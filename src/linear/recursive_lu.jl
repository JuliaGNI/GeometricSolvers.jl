# Recursive LU with partial pivoting on the host [Toledo1997, doi:10.1137/S0895479896297744]:
# the columns of a panel split in halves, the left half is factorised, its row exchanges and the
# triangular solve `L11⁻¹ A12` reach the right half, the Schur complement `A22 - A21 A12` is
# formed, and the right half is factorised, whose row exchanges then reach `A21`. A panel of at
# most `base` columns is factorised by an own unblocked kernel; the triangular solve and the
# product call BLAS (`trsm!`, `gemm!`). The pivot of a column is the first entry of largest
# `|Re| + |Im|`, as LAPACK's `i?amax` picks it, and `ipiv` follows LAPACK's convention: row `k` was
# exchanged with row `ipiv[k]` at step `k`. The recursion works on index ranges of the parent
# matrix, so that each view is a view of `A` and keeps its type.

# `|Re| + |Im|`, the magnitude by which LAPACK picks a pivot
abs1(x::Real) = abs(x)
abs1(z::Complex) = abs(real(z)) + abs(imag(z))

# The largest panel that `LUFactorization()` factorises with the unblocked kernel of the recursion:
# the base case of the measurement of `scripts/hardware/rl_host_lu.jl`.
const RECURSIVE_LU_BASE = 16

# The band `(lo, hi)` of `n` in which the measurement of `scripts/hardware/rl_host_lu.jl` gave
# `r = t(recursion) / t(getrf!) ≤ 0.90` at every `n` from `lo` to `hi`, with BLAS on one thread and
# on all threads (Apple M4 Max, OpenBLAS 0.3.30, Julia 1.13.4, base case 16), or `nothing` for an
# element type without such a band.
recursive_lu_band(::Type) = nothing
recursive_lu_band(::Type{Float32}) = (8, 32)
recursive_lu_band(::Type{Float64}) = (8, 32)
recursive_lu_band(::Type{ComplexF64}) = (8, 32)

# Whether `LUFactorization()` factorises an `n × n` host `Matrix{T}` by `recursive_lu!` rather than
# `LAPACK.getrf!`: where `n` lies in the band of `T`. `n = 0`, `n = 1`, an `n` outside the band and
# an element type without a band take `getrf!`; a non-BLAS element type takes the generic `lu!`.
function use_recursive_lu(::Type{T}, n::Integer) where {T}
    band = recursive_lu_band(T)
    return band !== nothing && band[1] <= n <= band[2]
end

# Factorise the square `A` in place as `P A = L U` with partial pivoting, by a recursion over
# halves of the columns down to panels of at most `base` columns, and return `info`: the index of
# the first exactly zero pivot, or 0. `A` then holds `U` on and above the diagonal and the
# multipliers of the unit lower `L` below it, and `ipiv[k]` the row exchanged with row `k` at step
# `k`, as `LAPACK.getrf!` writes them. A `NaN` or an `Inf` in `A` gives a non-finite factor, and
# `info` still counts only an exactly zero pivot.
function recursive_lu!(A::StridedMatrix{T}, ipiv::AbstractVector{BlasInt},
        base::Int) where {T <: BlasFloat}
    Base.require_one_based_indexing(A, ipiv)
    n = size(A, 1)
    size(A, 2) == n ||
        throw(DimensionMismatch("the LU factors have size $(size(A)), expected a square matrix"))
    length(ipiv) == n ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), expected $n"))
    stride(A, 1) == 1 ||
        throw(ArgumentError("the LU factors must have contiguous columns"))
    base >= 1 ||
        throw(ArgumentError("the base case must have at least one column, got $base"))
    return recursive_lu_panel!(A, ipiv, 0, n, base)
end

# The panel of the rows `o + 1:N` and the columns `o + 1:o + n` of the `N × N` matrix `A`: its
# pivots go to `ipiv[o + 1:o + n]`, as row indices counted from row `o`, and the returned `info`
# counts from column `o`.
function recursive_lu_panel!(A::StridedMatrix{T}, ipiv::AbstractVector{BlasInt}, o::Int,
        n::Int, base::Int) where {T}
    n <= base && return unblocked_lu_panel!(A, ipiv, o, n)
    N = size(A, 1)
    n1 = n ÷ 2
    c = o + n1
    info1 = recursive_lu_panel!(A, ipiv, o, n1, base)
    # the exchanges of the left half reach the right half of the panel
    swap_panel_rows!(A, ipiv, o, (o + 1):c, (c + 1):(o + n))
    A11 = view(A, (o + 1):c, (o + 1):c)
    A12 = view(A, (o + 1):c, (c + 1):(o + n))
    A21 = view(A, (c + 1):N, (o + 1):c)
    A22 = view(A, (c + 1):N, (c + 1):(o + n))
    BLAS.trsm!('L', 'L', 'N', 'U', one(T), A11, A12)
    BLAS.gemm!('N', 'N', -one(T), A21, A12, one(T), A22)
    info2 = recursive_lu_panel!(A, ipiv, c, n - n1, base)
    # the pivots of the right half count from row `c`, and from row `o` once `n1` is added
    @inbounds for k in (c + 1):(o + n)
        ipiv[k] += n1
    end
    # the exchanges of the right half reach the left half
    swap_panel_rows!(A, ipiv, o, (c + 1):(o + n), (o + 1):c)
    return iszero(info1) ? (iszero(info2) ? 0 : info2 + n1) : info1
end

# Exchange row `k` with row `o + ipiv[k]` of the columns `cols`, for `k` in `ks` in order.
function swap_panel_rows!(A::AbstractMatrix, ipiv::AbstractVector{BlasInt}, o::Int,
        ks::UnitRange{Int}, cols::UnitRange{Int})
    @inbounds for j in cols
        for k in ks
            p = o + Int(ipiv[k])
            if p != k
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
        end
    end
    return A
end

# The unblocked kernel of a panel of `n` columns: for each column the pivot of largest
# `|Re| + |Im|`, first maximum, the exchange of its row across the panel, the multipliers below
# the diagonal and the rank-1 update of the rest of the panel. An exactly zero pivot exchanges
# nothing and scales nothing, as `getf2`, and sets `info` at the first.
function unblocked_lu_panel!(A::StridedMatrix{T}, ipiv::AbstractVector{BlasInt}, o::Int,
        n::Int) where {T}
    N = size(A, 1)
    info = 0
    R = real(T)
    @inbounds for j in 1:n
        c = o + j
        p = c
        amax = abs1(A[c, c])
        for i in (c + 1):N
            a = abs1(A[i, c])
            if a > amax
                amax = a
                p = i
            end
        end
        ipiv[c] = p - o
        piv = A[p, c]
        if !iszero(piv)
            if p != c
                for k in (o + 1):(o + n)
                    A[c, k], A[p, k] = A[p, k], A[c, k]
                end
            end
            # the reciprocal where it is finite, else a division per entry, as `getf2`
            if abs(piv) >= floatmin(R)
                rp = inv(piv)
                for i in (c + 1):N
                    A[i, c] *= rp
                end
            else
                for i in (c + 1):N
                    A[i, c] /= piv
                end
            end
        elseif iszero(info)
            info = j
        end
        for k in (c + 1):(o + n)
            t = A[c, k]
            for i in (c + 1):N
                A[i, k] -= A[i, c] * t
            end
        end
    end
    return info
end
