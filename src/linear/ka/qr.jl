# Device-resident dense Householder QR factorisation and solve (R3) on `AbstractGPUArray`,
# implemented in `KernelAbstractions` with overflow-safe column scaling, column-parallel
# triangular backsolve, zero scalar indexing on the host, and one status readback per
# factorisation.

using KernelAbstractions: @localmem, @synchronize

@kernel function ka_qr_reflector_kernel!(A, τ, status, k::Int, n::Int)
    idx = @index(Global, Linear)
    if idx == 1
        F = eltype(A)
        R = real(F)
        @inbounds begin
            amax = zero(R)
            for i in k:n
                amax = max(amax, abs(A[i, k]))
            end
            σ = zero(R)
            if !iszero(amax)
                s2 = zero(R)
                for i in k:n
                    s2 += abs2(A[i, k] / amax)
                end
                σ = amax * sqrt(s2)
            end
            if !iszero(σ)
                x₁ = A[k, k]
                phase = iszero(x₁) ? one(F) : x₁ / abs(x₁)
                s_norm = σ / amax
                u₁ = abs(x₁) / amax
                v₁ = phase * (s_norm + u₁)
                τk = (s_norm + u₁) / s_norm
                τ[k] = F(τk)
                for i in (k + 1):n
                    A[i, k] = (A[i, k] / amax) / v₁
                end
                A[k, k] = -phase * σ
            else
                τ[k] = zero(F)
            end
            if iszero(A[k, k]) && status[1] == Int32(0)
                status[1] = Int32(k)
            end
        end
    end
end

@kernel function ka_qr_apply_kernel!(A, @Const(τ), k::Int, n::Int)
    dj = @index(Global, Linear)
    j = k + dj
    @inbounds begin
        τk = τ[k]
        if !iszero(τk)
            akj = A[k, j]
            s = zero(akj)
            for i in (k + 1):n
                s += conj(A[i, k]) * A[i, j]
            end
            A[k, j] = (one(τk) - τk) * akj - τk * s
            for i in (k + 1):n
                tik = A[i, k] * τk
                A[i, j] -= tik * akj + tik * s
            end
        end
    end
end

@kernel function ka_qr_qadj_kernel!(
        d, @Const(A), @Const(τ), n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    l_sum = @localmem eltype(d) (G,)
    l_dk = @localmem eltype(d) (1,)
    for k in 1:n
        @inbounds begin
            τk = τ[k]
            if tid == 1
                l_dk[1] = d[k]
            end
            acc = zero(eltype(d))
            if !iszero(τk)
                i = k + tid
                while i <= n
                    acc += conj(A[i, k]) * d[i]
                    i += G
                end
            end
            l_sum[tid] = acc
        end
        @synchronize()
        for step in 1:trailing_zeros(G)
            s = G >> step
            @inbounds if tid <= s
                l_sum[tid] += l_sum[tid + s]
            end
            @synchronize()
        end
        @inbounds begin
            τk = τ[k]
            if !iszero(τk)
                dk = l_dk[1]
                s = l_sum[1]
                if tid == 1
                    d[k] = (one(τk) - τk) * dk - τk * s
                end
                i = k + tid
                while i <= n
                    tik = A[i, k] * τk
                    d[i] -= tik * dk + tik * s
                    i += G
                end
            end
        end
        @synchronize()
    end
end

@kernel function ka_qr_trsv_r_col_kernel!(d, @Const(A), k::Int)
    i = @index(Global, Linear)
    @inbounds begin
        dk = d[k] / A[k, k]
        d[i] -= A[i, k] * dk
    end
end

@kernel function ka_qr_trsv_r_diag_kernel!(d, @Const(A))
    k = @index(Global, Linear)
    @inbounds d[k] = d[k] / A[k, k]
end

function ka_qr!(A::AbstractGPUArray, τ::AbstractVector, status::AbstractVector{Int32})
    n = size(A, 1)
    n == 0 && return 0
    backend = get_backend(A)
    fill!(status, Int32(0))
    ref_kernel! = ka_qr_reflector_kernel!(backend)
    app_kernel! = ka_qr_apply_kernel!(backend)
    for k in 1:n
        ref_kernel!(A, τ, status, k, n; ndrange = 1)
        if k < n
            app_kernel!(A, τ, k, n; ndrange = n - k)
        end
    end
    isgpu(backend) && synchronize(backend)
    return Int(Array(status)[1])
end

function ka_factorize_qr!(fac::DeviceQR, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    info = ka_qr!(fac.factors, fac.τ, fac.status)
    return DeviceQR(fac.factors, fac.τ, fac.rhs, info, e, fac.status)
end

function factorize!!(fac::DeviceQR, ::QRFactorization, J::AbstractMatrix)
    return ka_factorize_qr!(fac, J)
end

const KA_QR_WORKGROUP = 64

function ka_qr_backsolve!(A::AbstractGPUArray, τ::AbstractVector, d::AbstractVector)
    n = length(d)
    n == 0 && return d
    backend = get_backend(A)
    ka_qr_qadj_kernel!(backend, KA_QR_WORKGROUP)(
        d, A, τ, n, Val(KA_QR_WORKGROUP); ndrange = KA_QR_WORKGROUP)
    trsv_r! = ka_qr_trsv_r_col_kernel!(backend)
    for k in n:-1:2
        trsv_r!(d, A, k; ndrange = k - 1)
    end
    ka_qr_trsv_r_diag_kernel!(backend)(d, A; ndrange = n)
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(fac::DeviceQR{<:AbstractMatrix{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_qr_backsolve!(fac.factors, fac.τ, d)
end
