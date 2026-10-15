using GeometricSolvers
using GeometricSolvers: factorize!!, ldiv!!, returncode, StaticLU, StaticQR
using StaticArrays: SMatrix, SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

const STATIC_METHODS = (LUFactorization, QRFactorization)

# the lower pairs: (T, Float32) for T = Float64 and ComplexF64
const LOWER_PAIRS = filter(p -> p[2] !== Nothing, PRECISION_PAIRS)

@testset "check 4: the T ≠ TF solve scales the right-hand side to unit size" begin
    @testset "$M, $T with TF = $TF, n = $n" for M in STATIC_METHODS, (T, TF) in LOWER_PAIRS,
        n in STATIC_SIZES

        m = M(TF)
        F = factortype(T, TF)
        A, b = first(conditioned_systems(4000n + 1, T, TF, n))
        fac = factorize!!(nothing, m, A)
        @test eltype(fac.factors) === F
        # 1e-40 is below floatmin(Float32), 1e40 above floatmax(Float32); rnorm(r) gives 0 for
        # fill(1e-163, n) and Inf for fill(1e155, n) in Float64, so the scale is no 2-norm; a
        # purely imaginary r has no real part to scale by
        rs = SVector{n, T}[T.(s * b / maximum(abs, b)) for s in (1e-40, 1e40)]
        append!(rs, [fill(T(1e-163), n), fill(T(1e155), n)])
        T <: Complex &&
            append!(rs, [T.(im * s * real.(b) / maximum(abs ∘ real, b))
                         for s in (1e-40, 1e40)])
        for r in rs
            x = ldiv!!(nothing, fac, r)
            @test eltype(x) === T
            @test all(isfinite, x)
            @test relerr(x, reference(A, r)) <= bound(A, T, TF)
        end
        # r = 0 gives a step of exactly 0
        @test ldiv!!(nothing, fac, zero(SVector{n, T})) === zero(SVector{n, T})
        # Inf or NaN in r gives a non-finite step, and no throw
        for v in (Inf, -Inf, NaN)
            x = ldiv!!(nothing, fac, Base.setindex(b, T(v), 1))
            @test !all(isfinite, x)
        end
    end
end

@testset "the factor element type is TF, made complex for a complex T" begin
    for M in STATIC_METHODS, T in ELTYPES

        A = SMatrix{2, 2, T}(1, 2, 3, 5)
        @test eltype(factorize!!(nothing, M(), A).factors) === T
        @test eltype(factorize!!(nothing, M(Float32), A).factors) === factortype(T, Float32)
    end
    @test eltype(factorize!!(nothing, LUFactorization(Float32),
        SMatrix{1, 1, ComplexF64}(1)).factors) === ComplexF32
end

@testset "a precision marker on an SMatrix has no method" begin
    for M in STATIC_METHODS, marker in (TF32(), FP16(), BF16()), T in ELTYPES
        @test_throws MethodError factorize!!(nothing, M(marker), SMatrix{2, 2, T}(1, 2, 3, 5))
    end
end

@testset "returncode maps info to SUCCESS or SINGULAR" begin
    A = SMatrix{2, 2, Float64}(1, 2, 3, 4)
    for M in STATIC_METHODS
        @test returncode(factorize!!(nothing, M(), A)) === SUCCESS
        @test returncode(factorize!!(nothing, M(), zero(A))) === SINGULAR
    end
    @test returncode(StaticLU(A, SVector{2, Int32}(1, 2), Int32(0))) === SUCCESS
    @test returncode(StaticLU(A, SVector{2, Int32}(1, 2), Int32(2))) === SINGULAR
    @test returncode(StaticQR(A, SVector(0.0, 0.0), Int32(1))) === SINGULAR
end
