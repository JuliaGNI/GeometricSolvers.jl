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
        # 1e-40 is below floatmin(Float32), 1e40 above floatmax(Float32); the unscaled
        # sqrt(norm2(r)) gives 0 for fill(1e-163, n) and Inf for fill(1e155, n) in Float64, so
        # the scale is no unscaled 2-norm; a purely imaginary r has no real part to scale by
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

    # r near floatmax(T) and a step entry above 1 in the scaled system: the step times s overflows
    # T unless the power of two of s and the 2^-e of J are applied together
    @testset "r near floatmax, $M, $T with TF = $TF" for M in (STATIC_METHODS...,
            SVDFactorization),
        (T, TF) in LOWER_PAIRS

        s = 0.6 * floatmax(real(T))
        A0 = SMatrix{2, 2, T}(1, 0.1, 0.1, 0.25)
        A = T(s) * A0
        r = SVector{2, T}(s, s)
        x = ldiv!!(nothing, factorize!!(nothing, M(TF), A), r)
        @test eltype(x) === T
        @test all(isfinite, x)
        @test relerr(x, reference(A, r)) <= bound(A0, T, TF)
    end

    # T narrower than TF: the step entry 2^-24 of the scaled system is the smallest subnormal
    # Float16, and half of it rounds to 0 unless frac and the power of two of s are applied before
    # the conversion to T. The identity returns r.
    @testset "T narrower than TF, $M" for M in (STATIC_METHODS..., SVDFactorization)
        A = SMatrix{2, 2, Float16}(1, 0, 0, 1)
        r = SVector{2, Float16}(floatmin(Float16), 1024)
        x = ldiv!!(nothing, factorize!!(nothing, M(Float32), A), r)
        @test eltype(x) === Float16
        @test all(isapprox.(x, r; rtol = 8 * eps(Float16)))
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
    @test returncode(StaticLU(A, SVector{2, Int32}(1, 2), Int32(0), Int32(0))) === SUCCESS
    @test returncode(StaticLU(A, SVector{2, Int32}(1, 2), Int32(2), Int32(0))) === SINGULAR
    @test returncode(StaticQR(A, SVector(0.0, 0.0), Int32(1), Int32(0))) === SINGULAR
end

@testset "the T ≠ TF factorisation divides J by a power of two, $M, $T with TF = $TF" for M in STATIC_METHODS,
    (T, TF) in LOWER_PAIRS

    m = M(TF)
    F = factortype(T, TF)
    A, b = first(conditioned_systems(5001, T, TF, 4))
    # 1e-50 and 1e-39 are below floatmin(Float32), 1e40 and 1e50 above floatmax(Float32): unscaled,
    # the conversion of s·A to Float32 flushes it to zero or overflows
    for s in (1e-50, 1e-39, 1e40, 1e50)
        sA = real(T)(s) * A
        fac, x = host_solve(m, sA, b)
        @test returncode(fac) === SUCCESS
        @test all(isfinite, x)
        @test relerr(x, reference(sA, b)) <= bound(sA, T, TF)
    end
    # the division is exact: 2^100 A has the factors of A, and an exponent larger by 100
    fac = factorize!!(nothing, m, A)
    fac100 = factorize!!(nothing, m, ldexp(one(real(T)), 100) * A)
    @test fac100.factors == fac.factors
    @test fac100.scale_exponent == fac.scale_exponent + 100
    # a right-hand side in the factor type takes the direct solve, which applies 2^-e too
    A20 = ldexp(one(real(T)), 20) * A
    r = SVector{4, F}(b)
    @test relerr(ldiv!!(nothing, factorize!!(nothing, m, A20), r), reference(A20, r)) <=
          bound(A20, T, TF)
    # without a conversion there is no division
    @test factorize!!(nothing, M(), real(T)(1e40) * A).scale_exponent == 0
end

@testset "the same-type solve scales r by a power of two, $T" for T in REAL_ELTYPES
    @testset "$label, $M" for (label, J, b, xref, with_qr) in extreme_rhs_systems(T),
        M in (SVDFactorization, QRFactorization)

        M === QRFactorization && !with_qr && continue
        fac = factorize!!(nothing, M(), SMatrix{4, 4, T}(J))
        x = ldiv!!(nothing, fac, SVector{4, T}(b))
        @test eltype(x) === T
        @test all(isfinite, x)
        @test relerr(x, xref) <= 100 * eps(T)
    end

    # a solution above floatmax overflows, by design: 2^-130 I has the solution 2^130
    @testset "a solution above floatmax overflows, $M" for M in (LUFactorization,
        QRFactorization, SVDFactorization)
        J = SMatrix{4, 4, T}(ldexp(one(T), -(exponent(floatmax(T)) + 3)) *
                             Matrix{T}(I, 4, 4))
        x = ldiv!!(nothing, factorize!!(nothing, M(), J), ones(SVector{4, T}))
        @test !any(isfinite, x)
    end

    # a zero, non-finite or empty r takes the path of an r that is not scaled
    @testset "a zero or non-finite r, $M" for M in (QRFactorization, SVDFactorization)
        fac = factorize!!(nothing, M(), SMatrix{4, 4, T}(2 * Matrix{T}(I, 4, 4)))
        @test ldiv!!(nothing, fac, zeros(SVector{4, T})) == zeros(SVector{4, T})
        for v in (T(Inf), T(NaN))
            @test !all(isfinite, ldiv!!(nothing, fac, SVector{4, T}(v, 1, 1, 1)))
        end
    end
end
