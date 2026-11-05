# The R1 Bunch–Kaufman LDLᵀ / LDLᴴ factorisation with 1 × 1 and 2 × 2 diagonal pivot blocks:
# a scalar loop on a local `MMatrix` (`local_array` of `factors.jl`) that one thread runs off the
# heap on the CPU, Metal, and CUDA, and on a `SizedMatrix` for a non-`isbits` element type such as
# `BigFloat`. The in-place `generic_ldlt!` and `generic_ldlt_backsolve!` routines also serve the
# generic (non-`BlasFloat`) R3 host path in `dense.jl`.

@inline bk_alpha(::Type{R}) where {R <: Real} = (one(R) + sqrt(R(17))) / R(8)

function is_hermitian_matrix(A::AbstractMatrix{<:Real})
    return true
end

function is_hermitian_matrix(A::AbstractMatrix{<:Complex{R}}) where {R <: Real}
    n = size(A, 1)
    tol = R(64) * eps(R)
    @inbounds for j in 1:n
        iszero(imag(A[j, j])) || return false
        for i in (j + 1):n
            aij = A[i, j]
            aji = A[j, i]
            if abs(aij - conj(aji)) > tol * max(abs(aij), abs(aji))
                return false
            end
        end
    end
    return true
end

@inline function ldlt_swap_lower!(
        A::AbstractMatrix{F}, m::Int, p::Int, n::Int, herm::Bool) where {F}
    p == m && return nothing
    @inbounds begin
        for j in 1:(m - 1)
            A[m, j], A[p, j] = A[p, j], A[m, j]
        end
        for i in (m + 1):(p - 1)
            tmp = A[i, m]
            A[i, m] = herm ? conj(A[p, i]) : A[p, i]
            A[p, i] = herm ? conj(tmp) : tmp
        end
        if herm && F <: Complex
            A[p, m] = conj(A[p, m])
        end
        amm = A[m, m]
        app = A[p, p]
        A[m, m] = herm && F <: Complex ? F(real(app)) : app
        A[p, p] = herm && F <: Complex ? F(real(amm)) : amm
        for i in (p + 1):n
            A[i, m], A[i, p] = A[i, p], A[i, m]
        end
    end
    return nothing
end

function generic_ldlt!(
        A::AbstractMatrix{F}, ipiv::AbstractVector, herm::Bool) where {F <: Number}
    n = size(A, 1)
    P = eltype(ipiv)
    if !all(isfinite, A)
        @inbounds for k in 1:n
            ipiv[k] = P(k)
        end
        return Int32(1)
    end
    R = real(F)
    α = bk_alpha(R)
    info = Int32(0)
    @inbounds if herm && F <: Complex
        for i in 1:n
            A[i, i] = F(real(A[i, i]))
        end
    end
    k = 1
    @inbounds while k <= n
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
            info == 0 && (info = Int32(k))
            ipiv[k] = P(k)
            k += 1
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
                ipiv[k] = P(p)
                d11 = herm && F <: Complex ? F(real(A[k, k])) : A[k, k]
                A[k, k] = d11
                if iszero(d11)
                    info == 0 && (info = Int32(k))
                else
                    inv_d11 = inv(d11)
                    for j in (k + 1):n
                        l_jk = A[j, k] * inv_d11
                        w_jk = herm ? conj(l_jk) : l_jk
                        for i in j:n
                            A[i, j] -= A[i, k] * w_jk
                        end
                        if herm && F <: Complex
                            A[j, j] = F(real(A[j, j]))
                        end
                        A[j, k] = l_jk
                    end
                end
                k += 1
            else
                ipiv[k] = P(-p)
                ipiv[k + 1] = P(-p)
                d11 = herm && F <: Complex ? F(real(A[k, k])) : A[k, k]
                d22 = herm && F <: Complex ? F(real(A[k + 1, k + 1])) : A[k + 1, k + 1]
                A[k, k] = d11
                A[k + 1, k + 1] = d22
                d21 = A[k + 1, k]
                d12 = herm ? conj(d21) : d21
                δ11 = d11 / d12
                δ22 = d22 / d21
                denom = δ11 * δ22 - one(F)
                if iszero(d21) || iszero(denom)
                    info == 0 && (info = Int32(k))
                else
                    inv_d12_det = inv(d12 * denom)
                    inv_d21_det = inv(d21 * denom)
                    for j in (k + 2):n
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
                        A[j, k] = l_jk
                        A[j, k + 1] = l_jk1
                    end
                end
                k += 2
            end
        end
    end
    return info
end

function generic_ldlt_backsolve!(A::AbstractMatrix{F}, ipiv::AbstractVector,
        x::AbstractVector{F}, herm::Bool) where {F <: Number}
    n = length(x)
    @inbounds begin
        k = 1
        while k <= n
            pk = Int(ipiv[k])
            if pk > 0
                if pk != k
                    x[k], x[pk] = x[pk], x[k]
                end
                k += 1
            else
                p = -pk
                if p != k + 1
                    x[k + 1], x[p] = x[p], x[k + 1]
                end
                k += 2
            end
        end
        k = 1
        while k <= n
            pk = Int(ipiv[k])
            if pk > 0
                xk = x[k]
                for i in (k + 1):n
                    x[i] -= A[i, k] * xk
                end
                x[k] = xk / A[k, k]
                k += 1
            else
                xk = x[k]
                xk1 = x[k + 1]
                for i in (k + 2):n
                    x[i] -= A[i, k] * xk + A[i, k + 1] * xk1
                end
                d21 = A[k + 1, k]
                d12 = herm ? conj(d21) : d21
                δ11 = A[k, k] / d12
                δ22 = A[k + 1, k + 1] / d21
                denom = δ11 * δ22 - one(F)
                b1 = xk / d12
                b2 = xk1 / d21
                x[k] = (δ22 * b1 - b2) / denom
                x[k + 1] = (δ11 * b2 - b1) / denom
                k += 2
            end
        end
        k = n
        while k >= 1
            pk = Int(ipiv[k])
            if pk > 0
                xk = x[k]
                for i in (k + 1):n
                    l_ik = herm ? conj(A[i, k]) : A[i, k]
                    xk -= l_ik * x[i]
                end
                x[k] = xk
                k -= 1
            else
                xk_1 = x[k - 1]
                xk = x[k]
                for i in (k + 1):n
                    l_ik_1 = herm ? conj(A[i, k - 1]) : A[i, k - 1]
                    l_ik = herm ? conj(A[i, k]) : A[i, k]
                    xk_1 -= l_ik_1 * x[i]
                    xk -= l_ik * x[i]
                end
                x[k - 1] = xk_1
                x[k] = xk
                k -= 2
            end
        end
        k = n
        while k >= 1
            pk = Int(ipiv[k])
            if pk > 0
                if pk != k
                    x[k], x[pk] = x[pk], x[k]
                end
                k -= 1
            else
                p = -pk
                if p != k
                    x[k], x[p] = x[p], x[k]
                end
                k -= 2
            end
        end
    end
    return x
end

function factorize!!(::Union{Nothing, StaticLDLT},
        method::LDLTFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    Â, e = factor_matrix(factor_eltype(method, T), J)
    herm = is_hermitian_matrix(Â)
    A = local_array(Â)
    ipiv = local_vector(Int32, Val(N))
    info = generic_ldlt!(A, ipiv, herm)
    return StaticLDLT(SMatrix(A), SVector(ipiv), info, e, herm)
end

function backsolve(fac::StaticLDLT{N, TF}, r::SVector{N, TF}) where {N, TF}
    x = local_array(r)
    generic_ldlt_backsolve!(fac.factors, fac.ipiv, x, fac.hermitian)
    return SVector(x)
end
