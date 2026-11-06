# The R1 one-sided Jacobi SVD of a rectangular `M × N` static matrix (`1 ≤ N ≤ M ≤ 32`).

"""
    StaticRectSVD{M, N, TF, L, LN, R}

The thin singular value decomposition `U * Diagonal(S) * V'` of a rectangular `M × N` static
matrix (`M ≥ N`), in element type `TF`, computed by one-sided Jacobi sweeps. `U` holds the
`M × N` left singular vectors, `V` the `N × N` right singular vectors, `S` the `N` singular
values of the matrix in `R = real(TF)`, `σ̂` the `N` singular values of the matrix divided by
`2^e`, `info` is 0 on convergence and 1 otherwise, `rank` is the numerical rank above
`√M · eps(R) · σ̂₁`, and `scale_exponent` is `e`.
"""
struct StaticRectSVD{M, N, TF, L, LN, R}
    U::SMatrix{M, N, TF, L}
    V::SMatrix{N, N, TF, LN}
    S::SVector{N, R}
    σ̂::SVector{N, R}
    info::Int32
    rank::Int32
    scale_exponent::Int32
end

@inline function Base.getproperty(fac::StaticRectSVD, s::Symbol)
    s === :factors && return getfield(fac, :U)
    return getfield(fac, s)
end

singular_values(fac::StaticRectSVD) = getfield(fac, :S)
exact_solve(fac::StaticRectSVD{M, N}) where {M, N} = getfield(fac, :rank) == N
returncode(fac::StaticRectSVD) = iszero(getfield(fac, :info)) ? SUCCESS : SINGULAR

@inline function factorize!!(
        method::SVDFactorization{<:Union{Nothing, AbstractFloat}},
        fac::Union{Nothing, StaticSVD, StaticRectSVD},
        J::StaticMatrix)
    return factorize!!(fac, method, J)
end

@inline function rotate_rect_columns!(
        A::StaticMatrix{M, N, F}, p, q, cs, sn) where {M, N, F}
    @inbounds for i in 1:M
        u = A[i, p]
        v = A[i, q]
        A[i, p] = cs * u - conj(sn) * v
        A[i, q] = sn * u + cs * v
    end
    return A
end

function svd_rect_matrix(::Type{F}, J::StaticMatrix{M, N, T}) where {F, M, N, T}
    e = matrix_exponent(J)
    S = SMatrix{M, N, T}(J)
    return SMatrix{M, N, F}(iszero(e) ? S : S * ldexp(one(float(real(T))), -e)), Int32(e)
end

@inline function rect_column_norm2(A::StaticMatrix{M, N, F}, j) where {M, N, F}
    x = local_array(zero(SVector{M, F}))
    @inbounds for i in 1:M
        x[i] = A[i, j]
    end
    return norm2(SVector(x))
end

@inline function jacobi_rect_sweep!(
        A::StaticMatrix{M, N, F}, V::StaticMatrix{N, N, F}, tol) where {M, N, F}
    R = real(F)
    norms = local_array(zero(SVector{N, R}))
    αmax = zero(R)
    @inbounds for j in 1:N
        norms[j] = rect_column_norm2(A, j)
        αmax = max(αmax, norms[j])
    end
    small = sqrt(eps(R))
    negligible = αmax * tol * tol
    rotations = 0
    m = N + isodd(N)
    for r in 0:(m - 2), k in 0:(m ÷ 2 - 1)

        p, q = jacobi_pair(N, r, k)
        p > N && continue
        α = @inbounds norms[p]
        β = @inbounds norms[q]
        (α ≤ negligible || β ≤ negligible) && continue
        γ = zero(F)
        @inbounds for i in 1:M
            γ += conj(A[i, p]) * A[i, q]
        end
        abs(γ) ≤ tol * sqrt(α * β) && continue
        cs, sn = jacobi_rotation(α, β, γ)
        rotate_rect_columns!(A, p, q, cs, sn)
        rotate_columns!(V, p, q, cs, sn)
        c = real(conj(sn) * γ) * cs
        α′ = cs^2 * α + abs2(sn) * β - 2c
        β′ = abs2(sn) * α + cs^2 * β + 2c
        @inbounds norms[p] = α′ < small * α ? rect_column_norm2(A, p) : α′
        @inbounds norms[q] = β′ < small * β ? rect_column_norm2(A, q) : β′
        rotations += 1
    end
    return rotations
end

function factorize!!(::Union{Nothing, StaticSVD, StaticRectSVD},
        method::SVDFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{M, N, T}) where {M, N, T}
    (M < N || N < 1) &&
        throw(DimensionMismatch("StaticRectSVD requires M >= N >= 1, got ($M, $N)"))
    M > 32 &&
        throw(ArgumentError("StaticRectSVD supports M <= 32, got M = $M"))
    F = factor_eltype(method, T)
    R = real(F)
    Â, e = svd_rect_matrix(F, J)
    A = local_array(Â)
    V = local_array(one(SMatrix{N, N, F}))
    tol = jacobi_tolerance(M, R)
    sweeps = 0
    converged = false
    while sweeps < JACOBI_MAX_SWEEPS
        sweeps += 1
        if iszero(jacobi_rect_sweep!(A, V, tol))
            converged = true
            break
        end
    end

    σ = zero(SVector{N, R})
    for j in 1:N
        σ = setindex(σ, sqrt(rect_column_norm2(A, j)), j)
    end
    for j in 1:(N - 1)
        k = j
        for i in (j + 1):N
            σ[i] > σ[k] && (k = i)
        end
        k == j && continue
        σj = σ[j]
        σ = setindex(setindex(σ, σ[k], j), σj, k)
        @inbounds for i in 1:M
            A[i, j], A[i, k] = A[i, k], A[i, j]
        end
        @inbounds for i in 1:N
            V[i, j], V[i, k] = V[i, k], V[i, j]
        end
    end

    ok = converged && all(isfinite, σ)
    info = ok ? Int32(0) : Int32(1)
    cutoff = jacobi_tolerance(M, R) * σ[1]
    rank = Int32(0)
    if ok
        for j in 1:N
            σ[j] > cutoff || break
            rank += Int32(1)
        end
    end

    for j in 1:N
        if j <= rank
            @inbounds for i in 1:M
                A[i, j] /= σ[j]
            end
        else
            @inbounds for i in 1:M
                A[i, j] = zero(F)
            end
        end
    end

    S = map(a -> ldexp2(a, e), σ)
    return StaticRectSVD(SMatrix(A), SMatrix(V), S, σ, info, rank, e)
end

function backsolve(fac::StaticRectSVD{M, N, TF}, r::SVector{M, TF}) where {M, N, TF}
    return first(svd_rect_solve(fac, r, false))
end

function backsolve_shifted(
        fac::StaticRectSVD{M, N, TF}, r::SVector{M, TF}) where {M, N, TF}
    return svd_rect_solve(fac, r, true)
end

function svd_rect_solve(
        fac::StaticRectSVD{M, N, TF}, r::SVector{M, TF}, shift::Bool) where {M, N, TF}
    U = getfield(fac, :U)
    V = getfield(fac, :V)
    σ̂_fac = getfield(fac, :σ̂)
    rk = getfield(fac, :rank)
    y = zero(SVector{N, TF})
    c = coordinate_bound(TF, N)
    q = zero(real(TF))
    for j in 1:rk
        s = zero(TF)
        @inbounds for i in 1:M
            s += conj(U[i, j]) * r[i]
        end
        y = setindex(y, s, j)
        shift && (q = max(q, coordinate_ratio(s, σ̂_fac[j], c)))
    end
    t = shift ? shift_exponent(q) : 0
    σ̂ = iszero(t) ? σ̂_fac : map(a -> ldexp2(a, t), σ̂_fac)
    d = zero(SVector{N, TF})
    for j in 1:rk
        d = setindex(d, y[j] / σ̂[j], j)
    end
    x = local_array(zero(SVector{N, TF}))
    @inbounds for i in 1:N, j in 1:rk

        x[i] += V[i, j] * d[j]
    end
    return SVector{N, TF}(x), t
end

function ldiv!!(
        ::Union{Nothing, StaticVector{N}},
        fac::StaticRectSVD{M, N, TF},
        r::StaticVector{M, TF}) where {M, N, TF}
    k = rhs_exponent(r)
    r̂ = iszero(k) ? SVector{M, TF}(r) : map(a -> ldexp2(a, -k), SVector{M, TF}(r))
    d, t = backsolve_shifted(fac, r̂)
    return unscale(fac, d, k + t)
end

function ldiv!!(
        ::Union{Nothing, StaticVector{N}},
        fac::StaticRectSVD{M, N, TF},
        r::StaticVector{M, T}) where {M, N, TF, T}
    s = maximum(abs, r)
    iszero(s) && return zero(SVector{N, T})
    d = backsolve(fac, SVector{M, TF}(SVector{M, T}(r) / s))
    frac, ex = frexp(s)
    return map(a -> T(ldexp2(a * frac, ex - getfield(fac, :scale_exponent))), d)
end

@inline function static_rect_svd_nullnorm(
        U::SMatrix{M, N, TF}, rk::Int32, r::SVector{M, TF}) where {M, N, TF}
    p = local_array(r)
    @inbounds for j in 1:rk
        s = zero(TF)
        for i in 1:M
            s += conj(U[i, j]) * r[i]
        end
        for i in 1:M
            p[i] -= U[i, j] * s
        end
    end
    return rnorm(SVector{M, TF}(p))
end

@inline function factor_diagnostics(
        ::Nothing, fac::StaticRectSVD{M, N, TF}, r::StaticVector{M, T},
        fnorm::R) where {
        M, N, TF, T, R <: Real}
    getfield(fac, :info) == Int32(0) || return Int32(-1), R(NaN)
    rk = getfield(fac, :rank)
    rk == Int32(0) && return Int32(0), fnorm
    U = getfield(fac, :U)
    if TF === T
        return rk, R(static_rect_svd_nullnorm(U, rk, SVector{M, TF}(r)))
    end
    s = maximum(abs, r)
    iszero(s) && return rk, zero(R)
    r_tf = map(a -> TF(a / s), SVector{M, T}(r))
    return rk, R(static_rect_svd_nullnorm(U, rk, r_tf)) * R(s)
end

function factor_diagnostics(
        fac::StaticRectSVD{M, N, TF}, r::StaticVector{M, T}) where {M, N, TF, T}
    R = real(T)
    fnorm = R(rnorm(SVector{M, T}(r)))
    rk, nn = factor_diagnostics(nothing, fac, r, fnorm)
    rk < Int32(0) && return rk, R(NaN)
    iszero(fnorm) && return rk, zero(R)
    return rk, nn / fnorm
end
