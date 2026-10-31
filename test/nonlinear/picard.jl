# `Picard` in R1: constructor validation, `ToReal` adaptation, damped residual step,
# `MeasuredSlope()` globalization, and convergence on contractions.

using Adapt: Adapt
using KernelAbstractions: @index, @kernel, synchronize
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, NonlinearProblem, Options, Picard, SUCCESS,
                        Static, StrongWolfe, ToReal, init_state, rnorm, solve, solve!!,
                        step!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

@kernel function picard_solve_kernel!(
        codes, xs, @Const(prob), @Const(x₀), @Const(m), @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        x, st = solve(prob, x₀, m, p)
        codes[i] = st.code
        xs[i] = x
    end
end

@testset "Picard constructor, validation, and ToReal adaptation" begin
    m = Picard()
    @test m.damping === 1.0
    @test m.globalization === Static()
    @test isbits(m)

    m_int = Picard(; damping = 1)
    @test m_int.damping === 1.0
    @test m_int.globalization isa Static

    m_07 = Picard(; damping = 0.7)
    @test m_07.damping === 0.7
    @test isbits(m_07)

    a = Adapt.adapt(ToReal{Float32}(), m_07)
    @test a.damping === 0.7f0
    @test a.globalization === Static(1.0f0)
    @test isbits(a)

    for bad_d in (0, 0.0, -1, -0.5, NaN, Inf, -Inf)
        @test_throws ArgumentError Picard(; damping = bad_d)
    end
end

@testset "Picard solves contractions and scales direction by -damping * Fx, $T" for T in ELTYPES
    R = real(T)
    x₀ = SVector{2, T}(T(3), T(5))
    p = SVector{2, T}(one(T), one(T))

    # One Picard step with damping = 0.5 on F(x, p) = x - p from (3, 5) lands on (2, 3)
    prob_lin = NonlinearProblem((y, q) -> y .- q)
    x₁, st₁ = solve(
        prob_lin, x₀, Picard(; damping = 0.5), p;
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    @test x₁ === SVector{2, T}(T(2), T(3))
    @test st₁.iterations == 1

    # Contraction F(x, p) = 0.5 * (x - p) with Static, Backtracking, Bisection, StrongWolfe
    prob = NonlinearProblem((y, q) -> R(0.5) .* (y .- q))
    for glob in (Static(), Backtracking(), Bisection(), StrongWolfe())
        x_sol, st = solve(prob, x₀, Picard(; damping = 0.8, globalization = glob), p)
        @test st.code == SUCCESS
        @test st.iterations >= 1
        @test x_sol != x₀
        @test rnorm(prob.F(x_sol, p)) <=
              max(Options(T).f_abstol, Options(T).f_reltol * rnorm(prob.F(x₀, p)))
    end

    # Scalar iterate
    prob_s = NonlinearProblem((y, q) -> R(0.5) * (y - q))
    xs, sts = solve(
        prob_s, T(3), Picard(; damping = 0.8, globalization = Backtracking()), T(1))
    @test sts.code == SUCCESS
    @test isapprox(xs, T(1); rtol = 10 * sqrt(eps(R)))
end

@testset "the Picard state of a value iterate is isbits and allocation-free, $T" for T in ELTYPES
    allocations(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D} = (f(a, b, c, d);
        @allocated f(a, b, c, d))
    prob = NonlinearProblem((y, q) -> y .* y .- q)
    m = Adapt.adapt(ToReal{real(T)}(), Picard())
    for x in (SVector{2, T}(3, 5), T(3))
        s = @inferred init_state(m, prob, x, nothing)
        @test isbits(s)
        @test !(s isa NamedTuple)
        @test allocations(init_state, m, prob, x, nothing) == 0
    end
end

@testset "the damping of a complex problem is real" begin
    # with a complex damping, -ω F gives the real part -(0.5 * 0.0 - 0.0 * 1.0) = +0.0, and the
    # iterate's -0.0 + 0.0 = +0.0; with a real damping it gives -0.0, and the iterate stays -0.0
    for T in filter(t -> t <: Complex, ELTYPES)
        R = real(T)
        x₀ = T(complex(-zero(R), zero(R)))
        f₀ = T(complex(zero(R), one(R)))
        m = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.5))
        expected = x₀ - R(0.5) * f₀
        @test isequal(real(expected), -zero(R))
        # a value iterate
        for x in (SVector(x₀), x₀)
            prob = NonlinearProblem((y, q) -> x isa Number ? f₀ : SVector(f₀))
            s = init_state(m, prob, x, nothing)
            x₁, _, _, _ = @inferred step!!(s, m, prob, x, prob.F(x, nothing), nothing)
            @test isequal(x₁, x .- R(0.5) .* prob.F(x, nothing))
            @test isequal(real(x₁[1]), -zero(R))
            xs, _ = solve(prob, x, m; max_iterations = 1)
            @test isequal(xs, x₁)
        end
        # an array iterate
        prob! = NonlinearProblem((y, x, q) -> (y .= f₀; y))
        xa = [x₀]
        s = init_state(m, prob!, xa, nothing)
        x₁, _, _, _ = @inferred step!!(s, m, prob!, xa, [f₀], nothing)
        @test isequal(x₁, [expected])
    end
end

@testset "Picard Float32 inference and KernelAbstractions kernel" begin
    x₀ = SVector{2, Float32}(3.0f0, 5.0f0)
    p = SVector{2, Float32}(2.0f0, 4.0f0)
    prob = NonlinearProblem((y, q) -> y .- q)
    m_pic = Adapt.adapt(ToReal{Float32}(), Picard(; damping = 0.7))
    x_sol, st = @inferred solve(prob, x₀, Picard(; damping = 0.7), p)
    @test st.code == SUCCESS
    @test eltype(x_sol) === Float32

    codes = Vector{GeometricSolvers.ReturnCode}(undef, 2)
    xs = Vector{SVector{2, Float32}}(undef, 2)
    picard_solve_kernel!(KA_BACKEND)(codes, xs, prob, x₀, m_pic, p; ndrange = 2)
    synchronize(KA_BACKEND)
    @test all(==(st.code), codes)
    @test eltype(xs[1]) === Float32
    @test all(x -> isapprox(x, p; rtol = 10 * sqrt(eps(Float32))), xs)
end
