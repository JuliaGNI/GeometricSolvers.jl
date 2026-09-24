using GeometricSolvers
using GeometricSolvers: converged, record, rnorm
using StaticArrays: SVector, SMatrix
using Test

@testset "Options(T) is Options{real(T)} and isbits" for T in (Float32, Float64, ComplexF32,
    ComplexF64)
    R = real(T)
    opt = Options(T)
    @test opt isa Options{R}
    @test isbits(opt)
    @test opt.f_abstol === zero(R)
    @test opt.f_reltol === sqrt(eps(R))
    @test opt.x_abstol === zero(R)
    @test opt.x_reltol === 2 * eps(R)
    @test opt.min_iterations === Int32(0)
    @test opt.max_iterations === Int32(100)
    @test opt.max_stalls === Int32(2)
end

@testset "keywords are converted to R and Int32" begin
    opt = Options(Float32; f_abstol = 1e-6, f_reltol = 1 // 1000, max_iterations = 20)
    @test opt.f_abstol === 1.0f-6
    @test opt.f_reltol === 1.0f-3
    @test opt.max_iterations === Int32(20)
end

@testset "Options converted from Float64 to Float32 has only Float32 fields" begin
    opt = convert(Options{Float32}, Options(Float64; f_abstol = 1e-9, max_stalls = 5))
    @test opt isa Options{Float32}
    for name in (:f_abstol, :f_reltol, :x_abstol, :x_reltol)
        @test getfield(opt, name) isa Float32
    end
    @test all(T -> T === Float32 || T === Int32, fieldtypes(typeof(opt)))
    @test opt.f_abstol === 1.0f-9
    @test opt.f_reltol === Float32(sqrt(eps(Float64)))
    @test opt.max_stalls === Int32(5)
    @test convert(Options{Float32}, opt) === opt
end

@testset "invalid values raise an ArgumentError" begin
    @test_throws ArgumentError Options(Float64; f_abstol = -1)
    @test_throws ArgumentError Options(Float64; f_reltol = NaN)
    @test_throws ArgumentError Options(Float64; x_abstol = -eps())
    @test_throws ArgumentError Options(Float64; x_reltol = -1)
    @test_throws ArgumentError Options(Float64; min_iterations = -1)
    @test_throws ArgumentError Options(Float64; min_iterations = 5, max_iterations = 4)
    @test_throws ArgumentError Options(Float64; max_stalls = 0)
end

# A minimal Newton loop over the stopping test. It runs the test where a solver loop runs it:
# once before the first step and once after each step.
function newton(F, J, x, opt::Options{R}) where {R}
    Fx = F(x)
    fnorm₀ = rnorm(Fx)
    st = SolverStatus{R}(MAXITERS, Int32(0), fnorm₀, zero(R), Int32(0), false)
    while !converged(opt, st, fnorm₀, rnorm(x)) && st.iterations < opt.max_iterations
        d = -(J(x) \ Fx)
        x = x + d
        Fx = F(x)
        st = record(st, StepInfo{R}(MAXITERS, rnorm(Fx), rnorm(d), Int32(0), false))
    end
    return x, st, converged(opt, st, fnorm₀, rnorm(x))
end

@testset "a start at the root returns after 0 iterations" begin
    F(x) = x .^ 2 .- 4
    J(x) = SMatrix{2, 2}(2x[1], 0, 0, 2x[2])
    x₀ = SVector(2.0, -2.0)
    x, st, ok = newton(F, J, x₀, Options(Float64))
    @test ok
    @test st.iterations == 0
    @test x === x₀
end

@testset "the default test converges when ‖J‖ ‖x‖ ≫ 1" begin
    # At the root x = (√π, √ℯ) · 1e10, eps ‖J‖ ‖x‖ ≈ 2e5: the attainable residual depends on the
    # scale of the problem (Tisseur 2001, Cor. 2.5), which the default test does not assume.
    c = SVector(π * 1e20, ℯ * 1e20)
    F(x) = x .^ 2 .- c
    J(x) = SMatrix{2, 2}(2x[1], 0, 0, 2x[2])
    x₀ = SVector(2.5e10, 2.5e10)
    x, st, ok = newton(F, J, x₀, Options(Float64))
    @test ok
    @test st.iterations < 20
    @test x ≈ sqrt.(c) rtol = 1e-8
    @test st.fnorm > 1                # the relative test, not a small absolute one, stopped it
end

@testset "min_iterations defers the test, and the step test waits for a step" begin
    opt = Options(Float64; min_iterations = 1)
    st₀ = SolverStatus{Float64}(MAXITERS, Int32(0), 0.0, 0.0, Int32(0), false)
    @test !converged(opt, st₀, 1.0, 1.0)
    @test converged(Options(Float64), st₀, 1.0, 1.0)
    # before the first step, a zero `stepnorm` means no step, not a converged one
    st₀ = SolverStatus{Float64}(MAXITERS, Int32(0), 1.0, 0.0, Int32(0), false)
    @test !converged(Options(Float64), st₀, 1.0, 1.0)
    st₁ = SolverStatus{Float64}(MAXITERS, Int32(1), 1.0, 0.0, Int32(0), false)
    @test converged(Options(Float64), st₁, 1.0, 1.0)
end
