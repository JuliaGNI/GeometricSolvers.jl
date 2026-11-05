# The R1 Householder QR factorisation of a rectangular `M × N` static matrix (`1 ≤ N ≤ M ≤ 32`)
# without column pivoting: a scalar loop on a local `MMatrix` off the heap.

"""
    StaticRectQR{M, N, TF, L}

The Householder QR factors of a rectangular `M × N` static matrix (`M ≥ N`), in element type
`TF`: `factors` holds `R` on and above the diagonal of the leading `N × N` triangle and below it
the reflector tails `w[(k + 1):M]`, scaled to `w[k] = 1`; the reflector of step `k` is
`I - τ[k] w wᴴ`. `info` is the index of the first zero diagonal entry of `R`, or 0, and
`scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was divided by.
`L = M * N` makes the `SMatrix` field concrete, so the factor is `isbits`.
"""
struct StaticRectQR{M, N, TF, L}
    factors::SMatrix{M, N, TF, L}
    τ::SVector{N, TF}
    info::Int32
    scale_exponent::Int32
end

@inline function factorize!!(
        method::QRFactorization{<:Union{Nothing, AbstractFloat}},
        fac::Union{Nothing, StaticQR, StaticRectQR},
        J::StaticMatrix)
    return factorize!!(fac, method, J)
end

function factorize!!(::Union{Nothing, StaticQR, StaticRectQR},
        method::QRFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{M, N, T}) where {M, N, T}
    (M < N || N < 1) &&
        throw(DimensionMismatch("StaticRectQR requires M >= N >= 1, got ($M, $N)"))
    M > 32 &&
        throw(ArgumentError("StaticRectQR supports M <= 32, got M = $M"))
    Â, e = factor_matrix(factor_eltype(method, T), J)
    A = local_array(Â)
    F = eltype(A)
    τ = zero(SVector{N, F})
    info = Int32(0)
    @inbounds for k in 1:N
        amax = zero(real(F))
        for i in k:M
            amax = max(amax, abs(A[i, k]))
        end
        x = local_vector(F, Val(M))
        for i in 1:M
            x[i] = (i >= k && !iszero(amax)) ? A[i, k] / amax : zero(F)
        end
        σ = amax * rnorm(SVector(x))
        if !iszero(σ)
            x₁ = A[k, k]
            phase = iszero(x₁) ? one(F) : x₁ / abs(x₁)
            v₁ = phase * (σ + abs(x₁))
            τk = (σ + abs(x₁)) / σ
            τ = setindex(τ, F(τk), k)
            for i in (k + 1):M
                A[i, k] = A[i, k] / v₁
            end
            for j in (k + 1):N
                s = A[k, j]
                for i in (k + 1):M
                    s += conj(A[i, k]) * A[i, j]
                end
                s *= τk
                A[k, j] -= s
                for i in (k + 1):M
                    A[i, j] -= A[i, k] * s
                end
            end
            A[k, k] = -phase * σ
        end
        iszero(A[k, k]) && info == 0 && (info = Int32(k))
    end
    return StaticRectQR(SMatrix(A), τ, info, e)
end

function backsolve(fac::StaticRectQR{M, N, TF}, r::SVector{M, TF}) where {M, N, TF}
    A = fac.factors
    x = local_array(r)
    d = local_vector(TF, Val(N))
    @inbounds begin
        # Qᴴ r = H_N ⋯ H_1 r across all M rows
        for k in 1:N
            s = x[k]
            for i in (k + 1):M
                s += conj(A[i, k]) * x[i]
            end
            s *= fac.τ[k]
            x[k] -= s
            for i in (k + 1):M
                x[i] -= A[i, k] * s
            end
        end
        # R₁ d = (Qᴴ r)[1:N]
        for k in N:-1:1
            dk = x[k]
            for j in (k + 1):N
                dk -= A[k, j] * d[j]
            end
            d[k] = dk / A[k, k]
        end
    end
    return SVector(d)
end

function ldiv!!(
        ::Union{Nothing, StaticVector{N}},
        fac::StaticRectQR{M, N, TF},
        r::StaticVector{M, TF}) where {M, N, TF}
    k = rhs_exponent(r)
    r̂ = iszero(k) ? SVector{M, TF}(r) : map(a -> ldexp2(a, -k), SVector{M, TF}(r))
    d, t = backsolve_shifted(fac, r̂)
    return unscale(fac, d, k + t)
end

function ldiv!!(
        ::Union{Nothing, StaticVector{N}},
        fac::StaticRectQR{M, N, TF},
        r::StaticVector{M, T}) where {M, N, TF, T}
    s = maximum(abs, r)
    iszero(s) && return zero(SVector{N, T})
    d = backsolve(fac, SVector{M, TF}(SVector{M, T}(r) / s))
    frac, ex = frexp(s)
    return map(a -> T(ldexp2(a * frac, ex - fac.scale_exponent)), d)
end

function factor_diagnostics(
        fac::StaticRectQR{M, N, TF}, r::StaticVector{M, T}) where {M, N, TF, T}
    R = real(T)
    rk = fac.info == Int32(0) ? Int32(N) : fac.info - Int32(1)
    rk == Int32(N) && return rk, zero(R)
    rn = rnorm(SVector{M, T}(r))
    iszero(rn) && return rk, zero(R)
    A = fac.factors
    x = local_array(SVector{M, TF}(r))
    @inbounds for k in 1:rk
        s = x[k]
        for i in (k + 1):M
            s += conj(A[i, k]) * x[i]
        end
        s *= fac.τ[k]
        x[k] -= s
        for i in (k + 1):M
            x[i] -= A[i, k] * s
        end
    end
    tail = zero(real(TF))
    @inbounds for i in (rk + 1):M
        tail += abs2(x[i])
    end
    return rk, R(sqrt(tail)) / R(rn)
end
