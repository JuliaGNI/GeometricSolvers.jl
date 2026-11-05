# Device-resident dense Bunch–Kaufman LDLᵀ / LDLᴴ factorisation and solve (R3) on
# `AbstractGPUArray`, implemented in `KernelAbstractions` with 1 × 1 and 2 × 2 diagonal pivot
# blocks, column-parallel trailing updates, zero scalar indexing on the host, and one status
# readback per factorisation.

using KernelAbstractions: @localmem, @synchronize

const KA_LDLT_WORKGROUP = 64

@inline function ka_ldlt_is_hermitian(A, n::Int)
    F = eltype(A)
    F <: Real && return true
    R = real(F)
    err_h = zero(R)
    err_s = zero(R)
    @inbounds for j in 1:n
        for i in j:n
            aij = A[i, j]
            aji = A[j, i]
            dh = abs1(aij - conj(aji))
            ds = abs1(aij - aji)
            dh > err_h && (err_h = dh)
            ds > err_s && (err_s = ds)
        end
    end
    return !(err_s < err_h)
end

@inline function ka_ldlt_all_finite(A, n::Int)
    @inbounds for j in 1:n
        for i in 1:n
            isfinite(A[i, j]) || return false
        end
    end
    return true
end

@kernel function ka_ldlt_factor_kernel!(
        A, ipiv, status, n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    F = eltype(A)
    R = real(F)
    l_ctrl = @localmem Int32 (4,)
    l_coef = @localmem F (4,)
    if tid == 1
        @inbounds if !ka_ldlt_all_finite(A, n)
            for k in 1:n
                ipiv[k] = Int32(k)
            end
            status[1] = Int32(1)
            l_ctrl[1] = Int32(0)
        else
            herm = ka_ldlt_is_hermitian(A, n)
            if herm && F <: Complex
                for i in 1:n
                    A[i, i] = F(real(A[i, i]))
                end
            end
            l_ctrl[1] = Int32(1)
            l_ctrl[2] = Int32(0)
            l_ctrl[3] = Int32(0)
            l_ctrl[4] = herm ? Int32(1) : Int32(0)
        end
    end
    @synchronize()
    @inbounds if l_ctrl[1] != Int32(0)
        herm = l_ctrl[4] == Int32(1)
        α = bk_alpha(R)
        k = 1
        while k <= n
            if tid == 1
                r = k
                ω1 = zero(R)
                if k < n
                    r = k + 1
                    ω1 = abs(A[k + 1, k])
                    for i in (k + 2):n
                        v = abs(A[i, k])
                        if v > ω1
                            ω1 = v
                            r = i
                        end
                    end
                end
                abs_akk = herm ? abs(real(A[k, k])) : abs(A[k, k])
                if iszero(max(abs_akk, ω1))
                    l_ctrl[3] == Int32(0) && (l_ctrl[3] = Int32(k))
                    ipiv[k] = Int32(k)
                    l_ctrl[1] = Int32(1)
                    l_ctrl[2] = Int32(0)
                else
                    if abs_akk >= α * ω1
                        s = 1
                        p = k
                    else
                        ωr = zero(R)
                        for i in k:(r - 1)
                            v = abs(A[r, i])
                            v > ωr && (ωr = v)
                        end
                        for i in (r + 1):n
                            v = abs(A[i, r])
                            v > ωr && (ωr = v)
                        end
                        abs_arr = herm ? abs(real(A[r, r])) : abs(A[r, r])
                        if abs_akk * ωr >= α * ω1 * ω1
                            s = 1
                            p = k
                        elseif abs_arr >= α * ωr
                            s = 1
                            p = r
                        else
                            s = 2
                            p = r
                        end
                    end
                    m = s == 1 ? k : k + 1
                    ldlt_swap_lower!(A, m, p, n, herm)
                    if s == 1
                        ipiv[k] = Int32(p)
                        d11 = herm && F <: Complex ? F(real(A[k, k])) : A[k, k]
                        A[k, k] = d11
                        if iszero(d11)
                            l_ctrl[3] == Int32(0) && (l_ctrl[3] = Int32(k))
                            l_ctrl[2] = Int32(0)
                        else
                            l_coef[1] = inv(d11)
                            l_ctrl[2] = Int32(1)
                        end
                        l_ctrl[1] = Int32(1)
                    else
                        ipiv[k] = Int32(-p)
                        ipiv[k + 1] = Int32(-p)
                        d11 = herm && F <: Complex ? F(real(A[k, k])) : A[k, k]
                        d22 = herm && F <: Complex ? F(real(A[k + 1, k + 1])) :
                              A[k + 1, k + 1]
                        A[k, k] = d11
                        A[k + 1, k + 1] = d22
                        d21 = A[k + 1, k]
                        d12 = herm ? conj(d21) : d21
                        δ11 = d11 / d12
                        δ22 = d22 / d21
                        denom = δ11 * δ22 - one(F)
                        if iszero(d21) || iszero(denom)
                            l_ctrl[3] == Int32(0) && (l_ctrl[3] = Int32(k))
                            l_ctrl[2] = Int32(0)
                        else
                            l_coef[1] = δ11
                            l_coef[2] = δ22
                            l_coef[3] = inv(d12 * denom)
                            l_coef[4] = inv(d21 * denom)
                            l_ctrl[2] = Int32(1)
                        end
                        l_ctrl[1] = Int32(2)
                    end
                end
            end
            @synchronize()
            s = Int(l_ctrl[1])
            ok = l_ctrl[2] == Int32(1)
            if ok
                if s == 1
                    inv_d11 = l_coef[1]
                    j = k + tid
                    while j <= n
                        l_jk = A[j, k] * inv_d11
                        w_jk = herm ? conj(l_jk) : l_jk
                        for i in j:n
                            A[i, j] -= A[i, k] * w_jk
                        end
                        if herm && F <: Complex
                            A[j, j] = F(real(A[j, j]))
                        end
                        j += G
                    end
                    @synchronize()
                    j = k + tid
                    while j <= n
                        A[j, k] *= inv_d11
                        j += G
                    end
                else
                    δ11 = l_coef[1]
                    δ22 = l_coef[2]
                    inv_d12_det = l_coef[3]
                    inv_d21_det = l_coef[4]
                    j = k + 1 + tid
                    while j <= n
                        a_jk = A[j, k]
                        a_jk1 = A[j, k + 1]
                        l_jk = (a_jk * δ22 - a_jk1) * inv_d12_det
                        l_jk1 = (a_jk1 * δ11 - a_jk) * inv_d21_det
                        w_jk = herm ? conj(l_jk) : l_jk
                        w_jk1 = herm ? conj(l_jk1) : l_jk1
                        for i in j:n
                            A[i, j] -= A[i, k] * w_jk + A[i, k + 1] * w_jk1
                        end
                        if herm && F <: Complex
                            A[j, j] = F(real(A[j, j]))
                        end
                        j += G
                    end
                    @synchronize()
                    j = k + 1 + tid
                    while j <= n
                        a_jk = A[j, k]
                        a_jk1 = A[j, k + 1]
                        A[j, k] = (a_jk * δ22 - a_jk1) * inv_d12_det
                        A[j, k + 1] = (a_jk1 * δ11 - a_jk) * inv_d21_det
                        j += G
                    end
                end
            end
            @synchronize()
            k += s
        end
        if tid == 1
            info = l_ctrl[3]
            status[1] = herm ? info : -(info + Int32(1))
        end
    end
end

@kernel function ka_ldlt_factor_serial_kernel!(A, ipiv, status, n::Int)
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        if !ka_ldlt_all_finite(A, n)
            for k in 1:n
                ipiv[k] = Int32(k)
            end
            status[1] = Int32(1)
        else
            herm = ka_ldlt_is_hermitian(A, n)
            info = generic_ldlt!(A, ipiv, herm)
            status[1] = herm ? info : -(info + Int32(1))
        end
    end
end

@kernel function ka_ldlt_backsolve_kernel!(
        d, @Const(A), @Const(ipiv), n::Int, herm::Bool, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    F = eltype(d)
    if tid == 1
        @inbounds begin
            k = 1
            while k <= n
                pk = Int(ipiv[k])
                if pk > 0
                    if pk != k
                        d[k], d[pk] = d[pk], d[k]
                    end
                    k += 1
                else
                    p = -pk
                    if p != k + 1
                        d[k + 1], d[p] = d[p], d[k + 1]
                    end
                    k += 2
                end
            end
        end
    end
    @synchronize()
    @inbounds begin
        k = 1
        while k <= n
            pk = Int(ipiv[k])
            if pk > 0
                dk = d[k]
                i = k + tid
                while i <= n
                    d[i] -= A[i, k] * dk
                    i += G
                end
                @synchronize()
                if tid == 1
                    d[k] = dk / A[k, k]
                end
                @synchronize()
                k += 1
            else
                dk = d[k]
                dk1 = d[k + 1]
                i = k + 1 + tid
                while i <= n
                    d[i] -= A[i, k] * dk + A[i, k + 1] * dk1
                    i += G
                end
                @synchronize()
                if tid == 1
                    d21 = A[k + 1, k]
                    d12 = herm ? conj(d21) : d21
                    δ11 = A[k, k] / d12
                    δ22 = A[k + 1, k + 1] / d21
                    denom = δ11 * δ22 - one(F)
                    b1 = dk / d12
                    b2 = dk1 / d21
                    d[k] = (δ22 * b1 - b2) / denom
                    d[k + 1] = (δ11 * b2 - b1) / denom
                end
                @synchronize()
                k += 2
            end
        end
        if tid == 1
            k = n
            while k >= 1
                pk = Int(ipiv[k])
                if pk > 0
                    dk = d[k]
                    for i in (k + 1):n
                        l_ik = herm ? conj(A[i, k]) : A[i, k]
                        dk -= l_ik * d[i]
                    end
                    d[k] = dk
                    k -= 1
                else
                    dk_1 = d[k - 1]
                    dk = d[k]
                    for i in (k + 1):n
                        l_ik_1 = herm ? conj(A[i, k - 1]) : A[i, k - 1]
                        l_ik = herm ? conj(A[i, k]) : A[i, k]
                        dk_1 -= l_ik_1 * d[i]
                        dk -= l_ik * d[i]
                    end
                    d[k - 1] = dk_1
                    d[k] = dk
                    k -= 2
                end
            end
            k = n
            while k >= 1
                pk = Int(ipiv[k])
                if pk > 0
                    if pk != k
                        d[k], d[pk] = d[pk], d[k]
                    end
                    k -= 1
                else
                    p = -pk
                    if p != k
                        d[k], d[p] = d[p], d[k]
                    end
                    k -= 2
                end
            end
        end
    end
end

@kernel function ka_ldlt_backsolve_serial_kernel!(
        d, @Const(A), @Const(ipiv), n::Int, herm::Bool)
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        generic_ldlt_backsolve!(A, ipiv, d, herm)
    end
end

function ka_ldlt!(A::AbstractGPUArray, ipiv::AbstractVector{Int32},
        status::AbstractVector{Int32})
    n = size(A, 1)
    n == 0 && return 0, true
    backend = get_backend(A)
    fill!(status, Int32(0))
    if isgpu(backend) || n <= KA_LDLT_WORKGROUP
        ka_ldlt_factor_kernel!(backend, KA_LDLT_WORKGROUP)(
            A, ipiv, status, n, Val(KA_LDLT_WORKGROUP); ndrange = KA_LDLT_WORKGROUP)
    else
        ka_ldlt_factor_serial_kernel!(backend, 1)(A, ipiv, status, n; ndrange = 1)
    end
    isgpu(backend) && synchronize(backend)
    raw = Int(Array(status)[1])
    if raw >= 0
        return raw, true
    else
        return -raw - 1, false
    end
end

function ka_factorize_ldlt!(fac::DeviceLDLT, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    info, herm = ka_ldlt!(fac.factors, fac.ipiv, fac.status)
    return DeviceLDLT(fac.factors, fac.ipiv, fac.rhs, info, e, fac.status, herm)
end

function factorize!!(fac::DeviceLDLT, ::LDLTFactorization, J::AbstractMatrix)
    return ka_factorize_ldlt!(fac, J)
end

function ka_ldlt_backsolve!(A::AbstractGPUArray, ipiv::AbstractVector{Int32},
        d::AbstractVector, herm::Bool)
    n = length(d)
    n == 0 && return d
    backend = get_backend(A)
    if isgpu(backend) || n <= KA_LDLT_WORKGROUP
        ka_ldlt_backsolve_kernel!(backend, KA_LDLT_WORKGROUP)(
            d, A, ipiv, n, herm, Val(KA_LDLT_WORKGROUP); ndrange = KA_LDLT_WORKGROUP)
    else
        ka_ldlt_backsolve_serial_kernel!(backend, 1)(d, A, ipiv, n, herm; ndrange = 1)
    end
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(fac::DeviceLDLT{<:AbstractMatrix{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_ldlt_backsolve!(fac.factors, fac.ipiv, d, fac.hermitian)
end
