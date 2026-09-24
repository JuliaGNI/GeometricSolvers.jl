using GeometricSolvers
using GeometricSolvers: LinearMethod
using Test

@testset "$M carries its factorisation precision in the type" for M in (LUFactorization,
    QRFactorization, SVDFactorization)
    @test M() === M{Nothing}()
    @test M(Float32) === M{Float32}()
    @test M(Float64) === M{Float64}()
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

@testset "the markers are isbits" begin
    @test isbits(TF32())
    @test isbits(FP16())
    @test isbits(BF16())
end
