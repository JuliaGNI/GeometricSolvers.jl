# The R1 cyclic two-sided Jacobi eigensolver for a Hermitian — or a real symmetric — matrix:
# the rotations and the round-robin ordering of the Jacobi SVD of `svd.jl`, applied to both sides
# of the matrix at once. It is a function on a matrix and not a `linsolve` value, it is the only
# consumer of the SVD's rotations and of nothing else in this package, and it adds no method to
# `LinearAlgebra.eigen`: a method on a `StaticArrays` type would be type piracy.

"""
    jacobi_eigen(A)

The eigenvalues and the eigenvectors of the Hermitian matrix `A`, or of a real symmetric one,
by cyclic two-sided Jacobi: the rotations and the round-robin ordering of the Jacobi SVD of
[`SVDFactorization`](@ref), applied to the rows and the columns of `A` at once. The eigenvectors
are unitary to working precision, which is what `ProjectedSkew` needs of them. A pair whose
off-diagonal entry is at or below `eps(real(eltype(A))) * ‖A‖_F` is skipped, which is what ends
the sweep on a rank-deficient or low-rank matrix, as the SVD's negligible-column guard ends its
own.

The call returns `(values, vectors, status)`: `values` holds the `N` eigenvalues in
`real(eltype(A))`, `vectors` the columns of the unitary `V` with `A = V * Diagonal(values) * V'`,
and `status` is `SUCCESS` where the sweep reached its fixed point, or `MAXITERS`
where it reached its cap of 30 sweeps first. All three are `isbits`, so the call runs inside a
kernel.

The matrix is assumed Hermitian: the call reads its lower triangle and sets the upper triangle to
the conjugate of the lower one on entry, so a matrix that is not Hermitian gives the eigensystem
of that Hermitian part and not the one of `A`. A `NaN` or an `Inf` entry is not checked: nothing
throws, and the sweep reaches its cap for `N ≥ 2`; at `N = 1` there is no pair, so the call
returns `SUCCESS`.

For a real skew matrix `M`, the Hermitian matrix is `im * M`, with the eigenvalues `Λ` of
`im * M` and

    exp(M) == real(V * Diagonal(exp.(-im .* values)) * V')

to working precision — the `ProjectedSkew` case, which is what `jacobi_eigen` exists for.
"""
function jacobi_eigen(A::StaticMatrix{N, N, T}) where {N, T}
    H = MMatrix{N, N, T, N * N}(A)
    @inbounds for j in 1:N, i in (j + 1):N

        H[j, i] = conj(H[i, j])
    end
    V = one(MMatrix{N, N, T, N * N})
    R = real(T)
    # The pair floor `eps(R) * ‖H‖_F`, taken once on entry so no rotation changes it. A pair of
    # zero eigenvalues holds only rounding noise on the diagonal, which a relative test never
    # satisfies, whereas `√|α| · √|β| ≤ ‖H‖_F` means the Frobenius floor subsumes it. Scaling
    # by the largest `|re|` or `|im|` keeps `s * sqrt(a)` finite across the range of `R`, and a
    # `NaN` or an `Inf` entry makes `floor` `NaN` so no pair is skipped.
    s = zero(R)
    @inbounds for j in 1:N, i in 1:N

        s = max(s, abs(real(H[i, j])), abs(imag(H[i, j])))
    end
    a = zero(R)
    @inbounds for j in 1:N, i in 1:N

        re = abs(real(H[i, j])) / s
        im = abs(imag(H[i, j])) / s
        a += re * re + im * im
    end
    floor = iszero(s) ? zero(R) : eps(R) * s * sqrt(a)
    # The sweep is written out here rather than in a function of its own, as the sweeps of
    # `lu.jl` and `qr.jl` are: a call that is not inlined over a local `MMatrix` lets the matrix
    # escape, and an escaped matrix is a heap allocation of `n²` values — 4096 bytes for the two
    # of this function at `n = 16` (measured). Every read of the working matrix is `@inbounds`:
    # a bounds-checked read with run-time indices escapes it too, by 2048 bytes at `n = 16`.
    sweeps = 0
    converged = false
    m = N + isodd(N)
    while sweeps < JACOBI_MAX_SWEEPS
        sweeps += 1
        rotations = 0
        for r in 0:(m - 2), k in 0:(m ÷ 2 - 1)

            p, q = jacobi_pair(N, r, k)
            p > N && continue
            α = @inbounds real(H[p, p])
            β = @inbounds real(H[q, q])
            γ = @inbounds H[p, q]
            # `|β − α| ≤ 2‖H‖_F` bounds `|ζ| = |β − α| / (2|γ|)` of `jacobi_rotation` by
            # `1 / eps(R)` whenever `|γ| > floor`, well below the `ζ²` overflow threshold where a
            # rotation degenerates to the identity (`t = 0`).
            abs(γ) ≤ floor && continue
            cs, sn = jacobi_rotation(α, β, γ)
            rotate_columns!(H, p, q, cs, sn)
            rotate_rows!(H, p, q, cs, sn)
            rotate_columns!(V, p, q, cs, sn)
            rotations += 1
        end
        if iszero(rotations)
            converged = true
            break
        end
    end
    # at the fixed point the diagonal holds the eigenvalues, which are real. The vector is grown
    # by `setindex`, as the pivots of `lu.jl` are: an `MVector` local escapes into the returned
    # `SVector` and is a heap allocation.
    λ = zero(SVector{N, real(T)})
    @inbounds for i in 1:N
        λ = setindex(λ, real(H[i, i]), i)
    end
    return λ, SMatrix(V), converged ? SUCCESS : MAXITERS
end

# The same rotation on the rows, `H ← Gᴴ H`: rows `p` and `q` are combined with `cs` and `-sn`,
# the transpose of the coefficients the columns take. `@inline` for the reason `rotate_columns!`
# of `svd.jl` carries.
@inline function rotate_rows!(H::MMatrix{N, N, T}, p, q, cs, sn) where {N, T}
    @inbounds for j in 1:N
        hp = H[p, j]
        hq = H[q, j]
        H[p, j] = cs * hp - sn * hq
        H[q, j] = conj(sn) * hp + cs * hq
    end
    return H
end
