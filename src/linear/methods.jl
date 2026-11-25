"""
    TF32()

The factorisation-precision marker for NVIDIA's TensorFloat-32 tensor cores. It is a type only:
no factorisation method accepts a marker, so a marker selects no code path.
"""
struct TF32 end

"""
    FP16()

The factorisation-precision marker for half-precision tensor cores. See [`TF32`](@ref).
"""
struct FP16 end

"""
    BF16()

The factorisation-precision marker for bfloat16 tensor cores. See [`TF32`](@ref).
"""
struct BF16 end

const PrecisionMarker = Union{TF32, FP16, BF16}
const FactorisationPrecision = Union{Nothing, AbstractFloat, PrecisionMarker}

"""
    LinearMethod{TF}

The supertype of the linear-solver methods. `TF` is the factorisation precision: `Nothing` for
the working type of the problem, a real float type, or a [`PrecisionMarker`](@ref TF32). It is a
type parameter and not a field, because a type stored in a field is not `isbits`, and a method
must enter a kernel.

`TF` is a real type for a complex problem too: `LUFactorization(Float32)` factorises a
`ComplexF64` Jacobian in `ComplexF32`.
"""
abstract type LinearMethod{TF <: FactorisationPrecision} end

"""
    LUFactorization()
    LUFactorization(TF)

LU factorisation with partial pivoting. `LUFactorization()` factorises in the working type of the
problem, `LUFactorization(Float32)` in `Float32`, and `LUFactorization(TF32())` carries a
tensor-core marker (see [`TF32`](@ref)).
"""
struct LUFactorization{TF <: FactorisationPrecision} <: LinearMethod{TF} end

"""
    QRFactorization()
    QRFactorization(TF)

Householder QR factorisation, with the factorisation precision of [`LUFactorization`](@ref).
"""
struct QRFactorization{TF <: FactorisationPrecision} <: LinearMethod{TF} end

"""
    SVDFactorization(; rtol = nothing)
    SVDFactorization(TF; rtol = nothing)

Singular value decomposition, with the factorisation precision of [`LUFactorization`](@ref).

The factor counts a singular value at or below `rtol · σ₁` as zero: its `rank` is the number of
singular values above that cutoff, and its solve is the pseudo-inverse truncated at that rank.
`rtol = nothing` is the cutoff `√m · eps(R) · σ₁`, with `m` the number of rows of the matrix and
`R` the real type of the factor; it is also the tolerance of the Jacobi sweep of a static factor
and of the own-kernel device factor, and the only cutoff at the level of the rounding noise.
`rtol = 0` keeps every nonzero singular value, also one of rounding noise.

`rtol` is stored in `real(TF)` for a float `TF`, and in `Float64` for `TF = Nothing` and for a
precision marker; the factor converts it to its own real type, so that a `Float64` `rtol` that
rounds to `1` there gives rank 0 and the zero step in a direct `factorize!!`. [`ToReal`](@ref)
converts `rtol` to its type and keeps `TF`. A stored `rtol` outside `[0, 1)`, or not finite,
raises an `ArgumentError`, also where `ToReal` rounds it to `1` in a solve.
"""
struct SVDFactorization{TF <: FactorisationPrecision, R <: Union{Nothing, AbstractFloat}} <:
       LinearMethod{TF}
    rtol::R
    function SVDFactorization{
            TF, R}(rtol) where {
            TF <: FactorisationPrecision, R <: Union{Nothing, AbstractFloat}}
        r = convert(R, rtol)
        # the stored value, after the conversion: a value that rounds to 1 is refused. The
        # comparison is false for `NaN`, so the guard refuses it too. A kernel cannot format a
        # message, so the message is constant; `r + zero(r)` canonicalises `-0.0` to `+0.0`.
        r === nothing || zero(r) <= r < one(r) ||
            throw(ArgumentError(
                "SVDFactorization: rtol must be finite and lie in [0, 1)"))
        return new{TF, R}(r === nothing ? nothing : r + zero(r))
    end
end

# The storage type of `rtol`: `TF` for a float `TF`, `Float64` for `Nothing` and a marker.
svd_rtol_type(::Type, ::Nothing) = Nothing
svd_rtol_type(::Type{TF}, ::Real) where {TF <: AbstractFloat} = TF
svd_rtol_type(::Type, ::Real) = Float64

function SVDFactorization{TF}(;
        rtol::Union{Nothing, Real} = nothing) where {TF <: FactorisationPrecision}
    return SVDFactorization{TF, svd_rtol_type(TF, rtol)}(rtol)
end
SVDFactorization(; rtol::Union{Nothing, Real} = nothing) = SVDFactorization{Nothing}(; rtol)
function SVDFactorization(::Type{TF}; rtol::Union{
        Nothing, Real} = nothing) where {
        TF <: AbstractFloat}
    return SVDFactorization{TF}(; rtol)
end
function SVDFactorization(::TF; rtol::Union{Nothing, Real} = nothing) where {
        TF <: PrecisionMarker}
    return SVDFactorization{TF}(; rtol)
end
@inline function promote_linsolve(m::SVDFactorization, ::Type{T}) where {T}
    return SVDFactorization(real(T); rtol = m.rtol)
end

# `TF` is a type parameter and no field, so `Adapt.@adapt_structure`, which rebuilds through the
# bare name, would lose it: `ToReal` converts `rtol` and keeps `TF`.
function Adapt.adapt_structure(to::ToReal, m::SVDFactorization{TF}) where {TF}
    rtol = Adapt.adapt(to, m.rtol)
    return SVDFactorization{TF, typeof(rtol)}(rtol)
end

"""
    LDLTFactorization()
    LDLTFactorization(TF)

Bunch–Kaufman `L D Lᵀ` / `L D Lᴴ` factorisation with `1 × 1` and `2 × 2` diagonal pivot blocks
for symmetric and Hermitian matrices, with the factorisation precision of
[`LUFactorization`](@ref). The factorisation reads the lower triangle and does not check that a
real matrix is symmetric. A complex matrix whose diagonal entries are real to `64 eps` of each
entry is factorised as `L D Lᴴ` where each off-diagonal pair is Hermitian to `64 eps` of the pair,
or where its largest Hermitian mismatch `|A[i, j] - conj(A[j, i])|` is no larger than its largest
symmetric mismatch `|A[i, j] - A[j, i]|`, and as the complex symmetric `L D Lᵀ` otherwise.
"""
struct LDLTFactorization{TF <: FactorisationPrecision} <: LinearMethod{TF} end

"""
    CholeskyFactorization()
    CholeskyFactorization(TF)

Cholesky factorisation `L Lᴴ` without pivoting, for a Hermitian (real symmetric) positive definite
matrix, with the factorisation precision of [`LUFactorization`](@ref). It reads the lower triangle
and the real part of the diagonal, and does not check that the matrix is Hermitian. A matrix that
is not positive definite in the factorisation precision gives `info = k`, the order of its first
leading minor that is not positive, and `SINGULAR`, without throwing; a non-finite entry anywhere
in the matrix gives `info = 1`. A host `Array` iterate factorises with LAPACK `potrf` for a
`LinearAlgebra.BlasFloat` element type and with a generic loop for the others; a static iterate
always takes the generic loop. A device array iterate raises an `ArgumentError`, because the
factorisation has no device factor.
"""
struct CholeskyFactorization{TF <: FactorisationPrecision} <: LinearMethod{TF} end

"""
    KroneckerFactorization(; block = LUFactorization())
    KroneckerFactorization(TF; block = LUFactorization())

The linear solver of the stage system of an implicit Runge–Kutta method, for [`Newton`](@ref) on a
[`NonlinearProblem`](@ref) with `stage_matrix = A` and `stage_jacobian = jac!`. The stage unknowns
are the increments `Z = (Z₁, …, Z_s)`, `Zᵢ = Yᵢ - yₙ`, of `s` stages of `n` entries each, stacked
into one iterate of length `s n`, and the Newton matrix of the stage system
`Z - h (A ⊗ I) f(yₙ + Z) = 0` is taken as `I - h A ⊗ J`, with `J` the `n × n` matrix that
`jac!(J, Z, p)` writes at each refresh (Hairer and Wanner 1996, *Solving Ordinary Differential
Equations II*, §IV.8, [doi:10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7)).

With the eigendecomposition `A = V Λ V⁻¹`, computed once where the state is built, a solve with
`I - h A ⊗ J` is `s` solves with the `n × n` blocks `γᵢ I - J`, `γᵢ = 1 / (h λᵢ)`. For a real
element type `T` of the iterate, a real eigenvalue takes one block in `T`, and a complex-conjugate
pair takes one block in `complex(T)`, whose solution gives the other half of the pair as its
conjugate: a Gauss or Radau IIA tableau of `s` stages takes `⌊s/2⌋` blocks in `complex(T)` and
`s mod 2` in `T`. For a complex `T` each of the `s` eigenvalues takes one block in `T`. `block` is
the linear method of each block, a method that factorises a general square matrix in the working
precision, such as [`LUFactorization`](@ref), [`QRFactorization`](@ref) or
[`SVDFactorization`](@ref). A [`GMRES`](@ref), a `KroneckerFactorization`, a method with a
factorisation precision, a [`CholeskyFactorization`](@ref) or
[`LDLTFactorization`](@ref), which read one triangle of a block that is in general not Hermitian,
and an [`UMFPACKFactorization`](@ref), which needs a sparse pattern that a block does not have,
raise an `ArgumentError`.

The step `h` is read from the parameters as `p.h` at each refresh, and `jac!` is called at each
refresh, which the `reuse` rule of `Newton` governs. Each block is factorised as `γᵢ I - J` scaled
by a power of two, which keeps it finite for a tiny `h`. A step `h = 0` makes the Newton matrix the
identity. A non-finite `h` and a non-finite entry of `J` give a non-finite direction, and the step
returns `NONFINITE`. So does a residual whose product with `V⁻¹ ⊗ I` overflows, which a step `h`
near `floatmax(real(T))` can cause, also where the stacked `Newton` solve returns a finite step:
Gauss with 3 stages at `h = floatmax(Float64)` is one such case. A block whose factor fails gives
`SINGULAR`.

`init` raises an `ArgumentError` for a value iterate (a `Number` or a `StaticArray`), a device
iterate, a [`DogLeg`](@ref) or [`LevenbergMarquardt`](@ref) globalization, a refinement,
`regularization > 0`, a `jacobian` other than `nothing`, a factorisation precision `TF` other than
`Nothing`, a problem without `stage_matrix`, parameters without a property `h`, an iterate whose
length is not a multiple of `s`, a stage matrix that is not square, not invertible, or complex for
a real `T`, and a stage matrix whose eigenvectors have
`cond(V) > max(eps(real(T)), eps(Float64))^(-1/2)`, which also refuses a stage matrix that is not
diagonalisable: the solve multiplies by `V` and `V⁻¹` and loses `cond(V)` in accuracy. Gauss with 6
stages has `cond(V) ≈ 603`, and a two-stage SDIRK, whose `A` has a double eigenvalue with a single
eigenvector, about `10¹⁶`. The eigendecomposition runs in `Float64` (`ComplexF64` for a complex
`A`), so for a `real(T)` wider than `Float64`, such as `BigFloat`, the bound is that of `Float64`,
and each Newton direction is accurate to `Float64` only, so a solve can take more steps than with
an exact Newton matrix. `solve` raises an
`ArgumentError` for a `ForwardDiff.Dual` in `x₀` or `p`: the tangent solve of implicit
differentiation needs the Jacobian of the stacked residual.
"""
struct KroneckerFactorization{TF <: FactorisationPrecision, B <: LinearMethod} <:
       LinearMethod{TF}
    block::B
    function KroneckerFactorization{
            TF, B}(block::B) where {
            TF <: FactorisationPrecision, B <: LinearMethod}
        (block isa LinearMethod{Nothing} &&
         !(block isa Union{GMRES, KroneckerFactorization})) ||
            throw(ArgumentError(
                "KroneckerFactorization takes a block method that factorises in the working precision, such as LUFactorization()"))
        block isa Union{CholeskyFactorization, LDLTFactorization} && throw(ArgumentError(
            "KroneckerFactorization takes a block method for a general square matrix, such as LUFactorization(): a block γ I - J is in general not Hermitian, and $(nameof(typeof(block))) reads one triangle of it"))
        block isa UMFPACKFactorization && throw(ArgumentError(
            "KroneckerFactorization takes a block method for a dense square matrix, such as LUFactorization(): a block γ I - J has no sparse pattern, which UMFPACKFactorization needs"))
        return new{TF, B}(block)
    end
end

function KroneckerFactorization(; block::LinearMethod = LUFactorization())
    return KroneckerFactorization{Nothing, typeof(block)}(block)
end
function KroneckerFactorization(::Type{TF};
        block::LinearMethod = LUFactorization()) where {TF <: AbstractFloat}
    return KroneckerFactorization{TF, typeof(block)}(block)
end
function KroneckerFactorization(::TF;
        block::LinearMethod = LUFactorization()) where {TF <: PrecisionMarker}
    return KroneckerFactorization{TF, typeof(block)}(block)
end

"""
    UMFPACKFactorization()

Sparse LU factorisation of a `SparseMatrixCSC` Jacobian through UMFPACK, from `SparseArrays`, in
the working type of the problem, which is `Float64` or `ComplexF64` on a host `Vector`. The
Jacobian pattern is a `SparseMatrixCSC{<:Any, Int}` `jacobian_prototype`, or the pattern that the
sparsity detector of an `AutoSparse` back end finds. `init` raises `ArgumentError` for another
element type, such as `Float32` or `ComplexF32`, for another iterate, such as a static or scalar
one, and for a solve with no sparse pattern. A detected pattern that `Newton` does not use counts
as none: under `LevenbergMarquardt`, and under `regularization ≠ 0` where the pattern does not
hold the whole structural diagonal. The first factorisation runs the symbolic analysis, and every
later factorisation of the same factor reuses it (see [`UMFPACKLU`](@ref)). UMFPACK is
distributed under the GPL with `SparseArrays`.
"""
struct UMFPACKFactorization <: LinearMethod{Nothing} end

for M in (:LUFactorization, :QRFactorization, :LDLTFactorization, :CholeskyFactorization)
    @eval begin
        $M() = $M{Nothing}()
        $M(::Type{TF}) where {TF <: AbstractFloat} = $M{TF}()
        $M(::TF) where {TF <: PrecisionMarker} = $M{TF}()
        @inline promote_linsolve(::$M, ::Type{T}) where {T} = $M(real(T))
    end
end
