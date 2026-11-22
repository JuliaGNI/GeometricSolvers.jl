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
    SVDFactorization()
    SVDFactorization(TF)

Singular value decomposition, with the factorisation precision of [`LUFactorization`](@ref).
"""
struct SVDFactorization{TF <: FactorisationPrecision} <: LinearMethod{TF} end

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

for M in (:LUFactorization, :QRFactorization, :SVDFactorization, :LDLTFactorization,
    :CholeskyFactorization)
    @eval begin
        $M() = $M{Nothing}()
        $M(::Type{TF}) where {TF <: AbstractFloat} = $M{TF}()
        $M(::TF) where {TF <: PrecisionMarker} = $M{TF}()
        @inline promote_linsolve(::$M, ::Type{T}) where {T} = $M(real(T))
    end
end
