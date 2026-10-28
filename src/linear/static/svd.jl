# The R1 one-sided Jacobi SVD: a scalar loop that one thread runs, on a local `MMatrix` that the
# compiler keeps off the heap on the CPU, on Metal and on CUDA, as the LU of `lu.jl`.
#
# The sweep rotates pairs of columns until they are mutually orthogonal, in a round-robin order:
# `n/2` disjoint pairs per round, and `n(n-1)/2` pairs per sweep, which is the order a parallel
# device kernel runs the same rotations in. A rotation is built from the Gram matrix of
# the pair — `α = ‖a_p‖²`, `β = ‖a_q‖²`, `γ = a_pᴴ a_q` — and it is the same rotation that a
# two-sided Jacobi eigensolver applies to a Hermitian matrix, so `eigen.jl` reuses it.
#
# The algorithm guards against two numerical traps:
#
# - The squared norms of the sweep underflow for a subnormal entry, so the matrix is first
#   divided by a power of two that brings its largest `|re|` or `|im|` into `[1, 2)`, or as close
#   as the exponent range allows. The division is exact, so the factors are those of the original
#   matrix up to that power of two. The exponent is clamped to the range of the type, because
#   `ldexp(one(T), -exponent(m) - 1)` is `Inf` for a subnormal `m`, and an infinite prescale
#   turns the whole matrix into `Inf`.
# - A pair of columns that hold nothing but rounding noise — the null directions of a
#   rank-deficient matrix — is asked to become orthogonal to each other, and each rotation
#   regenerates as much noise as it removes, so a per-pair relative test alone never terminates
#   there. A pair whose two columns are negligible against the largest column of the
#   matrix is skipped, which is what ends the sweep.

# The number of sweeps the iteration will run before giving up, the same cap LAPACK's `gesvj`
# uses. Six to twelve is typical, and reaching thirty means something is wrong rather than
# merely slow. A sweep that reaches it leaves `converged` false: a kernel cannot throw.
const JACOBI_MAX_SWEEPS = 30

# The relative tolerance of the sweep and the rank cutoff of `factorize!!`: `√n · eps(R)`, where
# `R = real(TF)`. `√n` and not a constant, because `n` roundings accumulate along a column.
jacobi_tolerance(n, ::Type{R}) where {R <: Real} = sqrt(R(n)) * eps(R)

"""
    jacobi_rotation(α, β, γ)

The rotation `(cs, sn)` that makes the Hermitian `2 × 2` matrix `[α γ; γ̄ β]` diagonal, with real
`α` and `β`: with `G = [cs sn; -conj(sn) cs]`, the product `Gᴴ [α γ; γ̄ β] G` is diagonal, and
`θ` with `tan θ = t` is at most `π/4` in modulus, which is what makes the sweep converge rather
than merely rearrange.

`γ` must not be zero: the phase of the rotation is `γ / |γ|`.
"""
function jacobi_rotation(α, β, γ)
    # t solves t² + 2ζt - 1 = 0 with ζ = (β - α)/2|γ|, and the root of smaller modulus is
    # copysign(1, ζ)/(|ζ| + √(1 + ζ²))
    ζ = (β - α) / (2 * abs(γ))
    # `copysign` and not `sign`: the two agree everywhere except at ζ = 0, where the two diagonal
    # entries are equal and the rotation that diagonalizes the pair is the full π/4 that a `sign`
    # reports as no rotation at all. A Hermitian matrix with a zero diagonal — `im * M` for a
    # real skew `M` — has ζ = 0 for every pair.
    t = copysign(one(ζ), ζ) / (abs(ζ) + sqrt(one(ζ) + ζ * ζ))
    cs = inv(sqrt(one(t) + t * t))
    return cs, (γ / abs(γ)) * (cs * t)
end

# The pair (p, q) of pair k of round r of the round-robin schedule over n indices: n/2 disjoint
# pairs per round for an even n, and for an odd n one pair per round holds the dummy index n + 1,
# which the caller skips. Each of the n(n-1)/2 real pairs then occurs exactly once per sweep.
function jacobi_pair(n::Int, r::Int, k::Int)
    m = n + isodd(n)
    k == 0 && return (m, r + 1)
    return (mod(r + k, m - 1) + 1, mod(r - k, m - 1) + 1)
end

# `a_p′ = cs a_p - conj(sn) a_q` and `a_q′ = sn a_p + cs a_q`, in place: the columns of the
# working matrix under a rotation, and the columns of the accumulated `V` under the same one.
# `@inline` for the same reason as `jacobi_sweep!` below.
@inline function rotate_columns!(A::MMatrix{N, N, F}, p, q, cs, sn) where {N, F}
    @inbounds for i in 1:N
        u = A[i, p]
        v = A[i, q]
        A[i, p] = cs * u - conj(sn) * v
        A[i, q] = sn * u + cs * v
    end
    return A
end

# `J` in the factor element type `F` as an `SMatrix`, and the exponent `e` of the power of two
# `2^e` that it was divided by, chosen so that its largest |re| or |im| lies in [1, 2), or as
# close as the exponent range allows: the largest entry of the result is below 2, and above
# `floatmin` wherever the type allows it, so that the squares of the sweep neither overflow nor
# underflow. `factor_matrix` of `factors.jl` is the same rule for `T ≠ TF` alone; the sweep needs
# it at `TF = T` too, because it squares the entries. Dividing in `T` before the conversion is
# what keeps a `Float64` matrix below the `Float32` range solvable at `TF = Float32`.
function svd_matrix(::Type{F}, J::StaticMatrix{N, N, T}) where {F, N, T}
    e = matrix_exponent(J)
    S = SMatrix{N, N, T}(J)
    # one return type for both branches: a `Union` of two static types here would make the
    # `MMatrix` below a runtime type construction, and the factorisation allocate
    return SMatrix{N, N, F}(iszero(e) ? S : S * ldexp(one(float(real(T))), -e)), Int32(e)
end

# The squared norm of column `j` through the reduction seam of `reductions.jl`: the column is
# copied into a local `MVector` first, and not passed as a `view`. The parent of a `SubArray`
# escapes into the generic reduction, and an `MMatrix` that escapes is a heap allocation of `n²`
# values — the LU and QR kernels of the same kind take their norms the same way.
@inline function column_norm2(A::MMatrix{N, N, F}, j) where {N, F}
    x = zero(MVector{N, F})
    @inbounds for i in 1:N
        x[i] = A[i, j]
    end
    return norm2(SVector(x))
end

# One sweep of the round-robin rounds, each pair rotated where its two columns are not yet
# orthogonal. Returns the number of rotations applied: a sweep that applies none is the fixed
# point of the iteration.
#
# `@inline`, and `rotate_columns!` with it: a call that is not inlined over a local `MMatrix`
# lets the matrix escape, and each escaped matrix is a heap allocation of `n²` values (measured
# with `--track-allocation=user`: 144 bytes at `n = 4` for each).
@inline function jacobi_sweep!(A::MMatrix{N, N, F}, V::MMatrix{N, N, F}, tol) where {N, F}
    R = real(F)
    αmax = zero(R)
    for j in 1:N
        αmax = max(αmax, column_norm2(A, j))
    end
    # `αmax` first, so that the product stays in the accumulator's type, and in the scale of the
    # matrix rather than in the scale of the pair
    negligible = αmax * tol * tol
    rotations = 0
    m = N + isodd(N)
    for r in 0:(m - 2), k in 0:(m ÷ 2 - 1)

        p, q = jacobi_pair(N, r, k)
        p > N && continue
        α = column_norm2(A, p)
        β = column_norm2(A, q)
        (α ≤ negligible || β ≤ negligible) && continue
        # the pair's own inner product, through no seam of `reductions.jl`: `rdot` returns its
        # real part alone, and the rotation needs the phase of the complex one
        γ = zero(F)
        @inbounds for i in 1:N
            γ += conj(A[i, p]) * A[i, q]
        end
        abs(γ) ≤ tol * sqrt(α * β) && continue
        cs, sn = jacobi_rotation(α, β, γ)
        rotate_columns!(A, p, q, cs, sn)
        rotate_columns!(V, p, q, cs, sn)
        rotations += 1
    end
    return rotations
end

function factorize!!(::Union{Nothing, StaticSVD},
        method::SVDFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    F = factor_eltype(method, T)
    Â, e = svd_matrix(F, J)
    A = MMatrix{N, N, F, N * N}(Â)
    V = one(MMatrix{N, N, F, N * N})
    tol = jacobi_tolerance(N, real(F))
    sweeps = 0
    converged = false
    while sweeps < JACOBI_MAX_SWEEPS
        sweeps += 1
        if iszero(jacobi_sweep!(A, V, tol))
            converged = true
            break
        end
    end

    # the columns of the working matrix are `σ̂ⱼ uⱼ` at the fixed point. `σ` is an `SVector` grown
    # by `setindex`, as the pivots of `lu.jl` are: an `MVector` local would escape into the
    # conversion below and be a heap allocation.
    σ = zero(SVector{N, real(F)})
    for j in 1:N
        σ = setindex(σ, sqrt(column_norm2(A, j)), j)
    end
    # non-increasing, with the matching columns of `U` and `V` carried along: the rank is a
    # leading run of `σ`, and the solve indexes it as one
    for j in 1:(N - 1)
        k = j
        for i in (j + 1):N
            σ[i] > σ[k] && (k = i)
        end
        k == j && continue
        σj = σ[j]
        σ = setindex(setindex(σ, σ[k], j), σj, k)
        @inbounds for i in 1:N
            A[i, j], A[i, k] = A[i, k], A[i, j]
            V[i, j], V[i, k] = V[i, k], V[i, j]
        end
    end

    # the rank cutoff: `σᵢ ≤ √n · eps(TF) · σ₁` counts as zero, which is the tolerance of
    # the sweep. `info` is the index of the first
    # such entry, as it is for the LU and QR factors.
    cutoff = tol * σ[1]
    info = Int32(0)
    for j in 1:N
        if !(σ[j] > cutoff)
            info = Int32(j)
            break
        end
    end

    # `A` becomes `U`. Beyond the rank the column is zeroed rather than normalized, which keeps
    # a division by rounding noise out of the solve and leaves a `NaN` out of a zero quotient.
    for j in 1:N
        if info == 0 || j < info
            @inbounds for i in 1:N
                A[i, j] /= σ[j]
            end
        else
            @inbounds for i in 1:N
                A[i, j] = zero(F)
            end
        end
    end

    # the singular values of the matrix the caller passed: a power of two scales one exactly.
    # `σ̂` keeps the ones the sweep computed, of the matrix as `factorize!!` divided it: the solve
    # divides by those, so that a `σ₁` above `floatmax(R)` — a `σ` the report cannot hold, where
    # every entry of the matrix is finite — still leaves a finite step
    σ̂ = σ
    for j in 1:N
        σ = setindex(σ, ldexp2(σ[j], e), j)
    end
    return StaticSVD(SMatrix(A), SMatrix(V), σ, σ̂, info, converged, e)
end

# The solve with the stored factors, of the matrix that `factorize!!` divided by `2^e`: the
# truncated pseudo-inverse `V Σ⁺ Uᴴ r`. Its coordinates are taken in the scaled domain, where the
# divisor is `σ̂`, the singular value the sweep computed, and the step is multiplied back by `2^-e`
# as the LU and QR solves are: `σ` itself can be `Inf` where `σ₁` is above `floatmax(R)`, and a
# division by it would make the step zero.
function backsolve(fac::StaticSVD{N, TF}, r::SVector{N, TF}) where {N, TF}
    U = fac.factors
    V = fac.V
    # the directions the rank cutoff counted as zero contribute nothing: dividing by their
    # rounding noise is what turns a null direction into an enormous step, or into a `NaN`
    rank = fac.info == 0 ? N : Int(fac.info) - 1
    d = zero(MVector{N, TF})
    for j in 1:rank
        s = zero(TF)
        @inbounds for i in 1:N
            s += conj(U[i, j]) * r[i]
        end
        d[j] = s / fac.σ̂[j]
    end
    x = zero(MVector{N, TF})
    @inbounds for i in 1:N, j in 1:N

        x[i] += V[i, j] * d[j]
    end
    return SVector(x)
end
