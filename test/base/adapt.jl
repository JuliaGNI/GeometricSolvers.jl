using Adapt: Adapt, adapt
using GeometricSolvers: ToReal
using Test

struct TestMethod{A, B, F}
    a::A
    b::B
    n::Int32
    f::F
end
Adapt.@adapt_structure TestMethod

@testset "ToReal converts every float field and nothing else" begin
    m = TestMethod(0.7, TestMethod(1.5, big"2.5", Int32(1), cos), Int32(3), sin)
    m32 = adapt(ToReal{Float32}(), m)
    @test m32.a === 0.7f0
    @test m32.b.a === 1.5f0
    @test m32.b.b === 2.5f0          # a BigFloat too
    @test m32.n === Int32(3)
    @test m32.b.n === Int32(1)
    @test m32.f === sin
    @test m32.b.f === cos
    @test isbits(m32)
    @test isbits(ToReal{Float32}())
    @test adapt(ToReal{Float64}(), 0.5f0) === 0.5
end
