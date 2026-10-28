# Device-resident dense Householder QR factorisation and solve (R3) on `AbstractGPUArray`,
# implemented in `KernelAbstractions` with overflow-safe column scaling, zero scalar indexing
# on the host, and one status readback per factorisation.

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

@kernel function ka_qr_backsolve_kernel!(d, @Const(A), @Const(τ), n::Int)
    idx = @index(Global, Linear)
    if idx == 1
        @inbounds begin
            for k in 1:n
                τk = τ[k]
                if !iszero(τk)
                    dk = d[k]
                    s = zero(dk)
                    for i in (k + 1):n
                        s += conj(A[i, k]) * d[i]
                    end
                    d[k] = (one(τk) - τk) * dk - τk * s
                    for i in (k + 1):n
                        tik = A[i, k] * τk
                        d[i] -= tik * dk + tik * s
                    end
                end
            end
            for k in n:-1:1
                dk = d[k] / A[k, k]
                d[k] = dk
                for i in 1:(k - 1)
                    d[i] -= A[i, k] * dk
                end
            end
        end
    end
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

function ka_factorize_qr!(fac::DenseQR, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    info = ka_qr!(fac.factors, fac.τ, fac.work)
    return DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, info, e)
end

function factorize!!(fac::DenseQR{<:AbstractGPUArray}, ::QRFactorization, J::AbstractMatrix)
    return ka_factorize_qr!(fac, J)
end

function ka_qr_backsolve!(A::AbstractGPUArray, τ::AbstractVector, d::AbstractVector)
    n = length(d)
    n == 0 && return d
    backend = get_backend(A)
    ka_qr_backsolve_kernel!(backend)(d, A, τ, n; ndrange = 1)
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(fac::DenseQR{<:AbstractGPUArray{T}},
        d::AbstractVector{T}) where {T <: BlasFloat}
    return ka_qr_backsolve!(fac.factors, fac.τ, d)
end

function backsolve!(fac::DenseQR{<:AbstractGPUArray{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_qr_backsolve!(fac.factors, fac.τ, d)
end
