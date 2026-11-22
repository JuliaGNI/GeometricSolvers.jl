# The R1 Cholesky factorisation `A = L Lᴴ` without pivoting: a scalar loop on a local `MMatrix`
# (`local_array` of `factors.jl`) that one thread runs off the heap, and on a `SizedMatrix` for a
# non-`isbits` element type such as `BigFloat`. The in-place `generic_cholesky!` and
# `generic_cholesky_backsolve!` routines also serve the generic (non-`BlasFloat`) R3 host path in
# `dense.jl`.

# Zero the entries of `A` above the diagonal, so that a factor matrix holds `L` alone.
@inline function zero_strict_upper!(A::AbstractMatrix{F}) where {F}
    @inbounds for j in 2:size(A, 2)
        for i in 1:(j - 1)
            A[i, j] = zero(F)
        end
    end
    return A
end

# The Cholesky factor `L` of the Hermitian matrix `A` in place, column by column, reading the lower
# triangle of `A` and the real part of its diagonal only, as LAPACK `potrf` with `uplo = 'L'` does,
# and setting the entries above the diagonal to zero. Returns `info`: `1` for a matrix with a
# non-finite entry anywhere, which is refused before any arithmetic or write, the order `j` of the
# first leading minor whose pivot `real(A[j, j]) - Σₖ |L[j, k]|²` is not positive (a zero or a
# `NaN` pivot included), with the columns from `j` on left unfactorised, or `0`.
@inline function generic_cholesky!(A::AbstractMatrix{F}) where {F <: Number}
    n = size(A, 1)
    all(isfinite, A) || return Int32(1)
    R = real(F)
    info = Int32(0)
    @inbounds for j in 1:n
        d = real(A[j, j])
        for k in 1:(j - 1)
            d -= abs2(A[j, k])
        end
        if !(d > zero(R))
            info = Int32(j)
            break
        end
        ljj = sqrt(d)
        A[j, j] = F(ljj)
        for i in (j + 1):n
            s = A[i, j]
            for k in 1:(j - 1)
                s -= A[i, k] * conj(A[j, k])
            end
            A[i, j] = s / ljj
        end
    end
    zero_strict_upper!(A)
    return info
end

# The solve of `L Lᴴ x = b` in place of `x = b`, with the factor `L` of `generic_cholesky!`: the
# forward substitution `L y = b`, then the back substitution `Lᴴ x = y`.
@inline function generic_cholesky_backsolve!(
        A::AbstractMatrix{F}, x::AbstractVector{F}) where {F <: Number}
    n = length(x)
    @inbounds begin
        for j in 1:n
            xj = x[j] / real(A[j, j])
            x[j] = xj
            for i in (j + 1):n
                x[i] -= A[i, j] * xj
            end
        end
        for j in n:-1:1
            xj = x[j]
            for i in (j + 1):n
                xj -= conj(A[i, j]) * x[i]
            end
            x[j] = xj / real(A[j, j])
        end
    end
    return x
end

function factorize!!(::Union{Nothing, StaticCholesky},
        method::CholeskyFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    Â, e = factor_matrix(factor_eltype(method, T), J)
    A = local_array(Â)
    info = generic_cholesky!(A)
    return StaticCholesky(SMatrix(A), info, e)
end

function backsolve(fac::StaticCholesky{N, TF}, r::SVector{N, TF}) where {N, TF}
    x = local_array(r)
    generic_cholesky_backsolve!(fac.factors, x)
    return SVector(x)
end
