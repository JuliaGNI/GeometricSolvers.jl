using GeometricSolvers
using GeometricSolvers: LinearMethod, ToReal, promote_linsolve
using Adapt: Adapt
using Test

include("../helpers/matrix.jl")

@testset "$M carries its factorisation precision in the type" for M in (LUFactorization,
    QRFactorization, SVDFactorization)
    @test M() === M{Nothing}()
    for R in REAL_ELTYPES
        @test M(R) === M{R}()
    end
    @test M(TF32()) === M{TF32}()
    @test M(FP16()) === M{FP16}()
    @test M(BF16()) === M{BF16}()
    @test M(Float32) isa LinearMethod{Float32}
    for m in (M(), M(Float32), M(TF32()))
        @test isbits(m)
        @test sizeof(m) == 0
    end
    # the factorisation precision is a real float type, `Nothing` or a marker
    @test_throws MethodError M(ComplexF32)
    @test_throws MethodError M(Int)
    @test_throws TypeError M{Int}()
end

const RTOL_PHRASE = "SVDFactorization: rtol must be finite and lie in [0, 1), got "

"Whether `f()` raises the `ArgumentError` of the `rtol` guard, asserted by its phrase."
function refuses_rtol(f)
    try
        f()
    catch e
        return e isa ArgumentError && startswith(e.msg, RTOL_PHRASE)
    end
    return false
end

@testset "SVDFactorization(; rtol)" begin
    @testset "the default is the field-less method" begin
        m = SVDFactorization()
        @test m === SVDFactorization(; rtol = nothing) === SVDFactorization{Nothing}()
        @test m.rtol === nothing
        @test isbits(m)
        @test sizeof(m) == 0
        @test SVDFactorization(Float32; rtol = nothing) === SVDFactorization(Float32)
        @test SVDFactorization(Float32).rtol === nothing
    end
    @testset "rtol is stored in real(TF), and in Float64 for Nothing and a marker" begin
        @test SVDFactorization(; rtol = 1e-9).rtol === 1e-9
        @test SVDFactorization(; rtol = 1.0f-9).rtol === Float64(1.0f-9)
        @test SVDFactorization(; rtol = 0).rtol === 0.0
        @test SVDFactorization(; rtol = big"1e-9").rtol === 1e-9
        @test SVDFactorization(Float32; rtol = 1e-9).rtol === 1.0f-9
        @test SVDFactorization(Float64; rtol = 1.0f-9).rtol === Float64(1.0f-9)
        @test SVDFactorization(TF32(); rtol = 1e-9).rtol === 1e-9
        @test SVDFactorization(BF16(); rtol = 1e-9).rtol === 1e-9
        # a value below the range of the storage type is rtol = 0
        @test SVDFactorization(Float32; rtol = 1e-50).rtol === 0.0f0
        for m in (SVDFactorization(; rtol = 1e-9), SVDFactorization(Float32; rtol = 1e-9),
            SVDFactorization(TF32(); rtol = 1e-9))
            @test isbits(m)
        end
        @test SVDFactorization(; rtol = 1e-9) isa LinearMethod{Nothing}
        @test SVDFactorization(Float32; rtol = 1e-9) isa LinearMethod{Float32}
        @test SVDFactorization(TF32(); rtol = 1e-9) isa LinearMethod{TF32}
        @test SVDFactorization(; rtol = 1e-9) === SVDFactorization(; rtol = 1e-9)
        @test SVDFactorization(; rtol = 1e-9) !== SVDFactorization(; rtol = 1e-12)
        @test SVDFactorization(; rtol = 1e-9) !== SVDFactorization()
    end
    @testset "rtol outside [0, 1) or not finite is refused, rtol = $bad" for bad in (
        -1e-300, -1.0, 1.0, 1.5, NaN, Inf, -Inf)
        @test refuses_rtol(() -> SVDFactorization(; rtol = bad))
        @test refuses_rtol(() -> SVDFactorization(Float64; rtol = bad))
    end
    @testset "rtol inside [0, 1) is accepted, rtol = $good" for good in (
        0.0, nextfloat(0.0), 1e-20, 0.5, prevfloat(1.0))
        @test SVDFactorization(; rtol = good).rtol === good
    end
    @testset "the stored value is checked, after the conversion" begin
        # 0.99999999 lies in [0, 1) and rounds to 1f0
        @test Float32(0.99999999) == 1
        @test refuses_rtol(() -> SVDFactorization(Float32; rtol = 0.99999999))
    end
    @testset "rtol is a Real or nothing" begin
        @test_throws TypeError SVDFactorization(; rtol = 1e-9 + 0im)
        @test_throws TypeError SVDFactorization(Float32; rtol = "1e-9")
    end
    @testset "promote_linsolve carries rtol" begin
        @test promote_linsolve(SVDFactorization(), Float32) === SVDFactorization(Float32)
        @test promote_linsolve(SVDFactorization(Float32; rtol = 1e-9), Float64) ===
              SVDFactorization(Float64; rtol = Float64(1.0f-9))
        @test promote_linsolve(SVDFactorization(; rtol = 1e-9), ComplexF32) ===
              SVDFactorization(Float32; rtol = 1e-9)
        # through the checking constructor: a Float64 rtol that rounds to 1f0
        @test refuses_rtol(() -> promote_linsolve(
            SVDFactorization(; rtol = 0.99999999), Float32))
    end
    @testset "ToReal converts rtol and keeps TF" begin
        to32 = ToReal{Float32}()
        @test Adapt.adapt(to32, SVDFactorization()) === SVDFactorization()
        @test Adapt.adapt(to32, SVDFactorization(Float64)) === SVDFactorization(Float64)
        a = Adapt.adapt(to32, SVDFactorization(; rtol = 1e-9))
        @test a isa LinearMethod{Nothing}
        @test a.rtol === 1.0f-9
        @test isbits(a)
        a64 = Adapt.adapt(to32, SVDFactorization(Float64; rtol = 1e-9))
        @test a64 isa LinearMethod{Float64}
        @test a64.rtol === 1.0f-9
        @test isbits(a64)
        atf = Adapt.adapt(to32, SVDFactorization(TF32(); rtol = 1e-9))
        @test atf isa LinearMethod{TF32}
        @test atf.rtol === 1.0f-9
        @test isbits(atf)
        @test Adapt.adapt(to32, Newton(; linsolve = SVDFactorization(; rtol = 1e-12))).linsolve.rtol ===
              1.0f-12
    end
end

@testset "the markers are isbits" begin
    @test isbits(TF32())
    @test isbits(FP16())
    @test isbits(BF16())
end
