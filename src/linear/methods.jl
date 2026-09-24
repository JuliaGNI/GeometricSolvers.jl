"""
    TF32()

The factorisation-precision marker for NVIDIA's TensorFloat-32 tensor cores. Only the CUDA fast
path will accept it; with any other array there is no method, and the solve raises a
`MethodError` on the host.
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
problem, `LUFactorization(Float32)` in `Float32`, and `LUFactorization(TF32())` on tensor cores.
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

for M in (:LUFactorization, :QRFactorization, :SVDFactorization)
    @eval begin
        $M() = $M{Nothing}()
        $M(::Type{TF}) where {TF <: AbstractFloat} = $M{TF}()
        $M(::TF) where {TF <: PrecisionMarker} = $M{TF}()
    end
end
