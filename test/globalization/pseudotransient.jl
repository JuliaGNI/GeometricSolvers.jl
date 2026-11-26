# `PseudoTransient`, the pseudo-transient continuation of `Newton` for a `StationarityProblem`:
# the double well whose saddle `Newton()` reaches and whose minimum the shifted steps reach, the
# parameters and their validation, the rule state that each solve starts at `δ₀` and that a raw
# step without `start!!` reads as `NaN`, the switched evolution relaxation of `δ` and its cap, a
# non-finite residual, the refusals, a linear gradient scaled to the ends of the float range, and
# the kernel path.

using Adapt: Adapt
using JET: JET, @test_opt
using JLArrays: JLArray
using KernelAbstractions: @Const, @index, @kernel, synchronize
using LinearAlgebra: Diagonal, isposdef
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: KroneckerFactorization, LeastSquaresProblem, LeftPreconditioned,
                        NONFINITE, NonlinearProblem, Options, PseudoTransient,
                        PseudoTimeStep, Refinement, SINGULAR, StationarityProblem, SUCCESS,
                        ToReal, init, init_state, pt_next_step, residual!!, solve, solve!,
                        solve!!, start!!, step!!, with_rule

include("../helpers/matrix.jl")

# The double well `S(x, y) = (x² − 1)²/4 + y²/2`: a saddle at `(0, 0)` and minima at `(±1, 0)`.
well(x, p) = (x[1]^2 - 1)^2 / 4 + x[2]^2 / 2
well_hessian(x) = Diagonal([3 * x[1]^2 - 1, one(x[1])])
I2(::Type{T}) where {T} = Diagonal(ones(T, 2))

@testset "the double well: the saddle and the minimum" begin
    @testset "$T $(typeof(x₀))" for T in REAL_ELTYPES,
        x₀ in (T[0.1, 1], SVector{2, T}(0.1, 1))

        prob = StationarityProblem(well)
        x, status = solve(prob, x₀, Newton(); max_iterations = 50)
        @test status.code == SUCCESS
        @test isapprox(x, T[0, 0]; atol = sqrt(eps(T)))
        x, status = solve(prob, x₀, Newton(; globalization = PseudoTransient(0.1));
            max_iterations = 50)
        @test status.code == SUCCESS
        @test isapprox(x, T[1, 0]; atol = sqrt(eps(T)))
        @test isposdef(well_hessian(x))
        @test status.iterations <= 50
    end
end

# `S(x) = s (½ xᵀ A x − bᵀ x)`, whose gradient `s (A x − b)` vanishes at `A⁻¹ b = (1, −2)` for
# every scale `s`.
@testset "a linear gradient scaled to the ends of the float range" begin
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, -3.0]
    @testset "$T s = $s" for T in REAL_ELTYPES,
        s in (4 * sqrt(floatmax(T)), floatmax(T) / 8)

        At, bt = T.(A), T.(b)
        S(x, p) = s * (x' * At * x / 2 - bt' * x)
        x, status = solve(StationarityProblem(S), zeros(T, 2),
            Newton(; globalization = PseudoTransient(0.1)); f_abstol = zero(T))
        @test status.code == SUCCESS
        @test isapprox(x, T[1, -2]; rtol = 64 * eps(T))
    end
end

@testset "the parameters" begin
    pt = PseudoTransient(0.1)
    @test pt.δ₀ == 0.1 && pt.δ_max == 1e12
    pt = PseudoTransient(0.5; δ_max = 4.0)
    @test pt.δ₀ == 0.5 && pt.δ_max == 4.0
    @test PseudoTransient(Float32, 0.1) isa PseudoTransient{Float32}
    for (δ₀, δ_max) in ((0.0, 1.0), (-1.0, 1.0), (NaN, 1.0), (Inf, Inf), (2.0, 1.0),
        (1.0, Inf), (1.0, NaN))
        @test_throws r"^ArgumentError: PseudoTransient requires 0 < δ₀ ≤ δ_max < Inf" PseudoTransient(
            δ₀; δ_max)
    end
    # the reuse and the refinement of a factor, and a method that forms no factor
    @test_throws r"^ArgumentError: Newton with a PseudoTransient globalization takes no reuse" Newton(;
        globalization = PseudoTransient(0.1), reuse = 2)
    @test_throws r"^ArgumentError: Newton with a PseudoTransient globalization takes no refinement" Newton(;
        globalization = PseudoTransient(0.1), refinement = Refinement())
    for linsolve in (GMRES(), MINRESQLP(), KroneckerFactorization())
        @test_throws r"^ArgumentError: Newton with a PseudoTransient globalization needs a factorising" Newton(;
            globalization = PseudoTransient(0.1), linsolve)
    end
end

@testset "the problems it refuses" begin
    m = Newton(; globalization = PseudoTransient(0.1))
    F!(y, x, p) = (y .= x .- 1; nothing)
    @test_throws r"^ArgumentError: PseudoTransient solves a StationarityProblem" init(
        NonlinearProblem(F!), [0.0, 0.0], m)
    @test_throws r"^ArgumentError: PseudoTransient solves a StationarityProblem" init(
        NonlinearProblem((x, p) -> x - 1), 0.0, m)
    lsq = LeastSquaresProblem((y, x, p) -> (y .= [x[1] - 1, x[1] - 2]; nothing), 2)
    @test_throws r"^ArgumentError: PseudoTransient solves a StationarityProblem" init(lsq, [0.0], m)
    # the problem that a combinator builds from a `NonlinearProblem`: the preconditioned residual
    # of `LeftPreconditioned` is no gradient either
    @test_throws r"^ArgumentError: PseudoTransient solves a StationarityProblem" solve(
        NonlinearProblem(F!), [0.0, 0.0], LeftPreconditioned(m, Newton()))
    @test_throws r"^ArgumentError: PseudoTransient solves a StationarityProblem" solve(
        NonlinearProblem(F!), [0.0, 0.0], LeftPreconditioned(Newton(), m))
    # a `StationarityProblem` with `E ≠ U`, whose residual is no gradient, is taken, and the
    # shifted steps reach a root of `F`: `S = x₁² x₂ + x₂³/3 − x₁` with `E = [1]` and `U = [2]`
    # gives `F(x₂) = 2 x₁ x₂ − 1`, whose root at `x₁ = 1/2` is `x₂ = 1`
    S(x, p) = x[1]^2 * x[2] + x[2]^3 / 3 - x[1]
    x, status = solve(StationarityProblem(S; E = [1], U = [2]), [0.5, 2.0], m)
    @test status.code == SUCCESS
    @test x[1] == 0.5 && isapprox(x[2], 1; atol = sqrt(eps()))
end

@testset "the rule state and the evolution of δ" begin
    @testset "$T" for T in REAL_ELTYPES
        m = Newton(; globalization = PseudoTransient(T(0.1); δ_max = T(1e3)))
        prob = StationarityProblem(well)
        s = init_state(m, prob, T[0.1, 1], nothing)
        u = s.u_work
        Fx = residual!!(similar(u), s.rprob, u, nothing)
        # a step on a state that no `start!!` saw reads `δ = NaN` and keeps its iterate
        @test isnan(s.inner.rule.δ)
        u₀ = copy(u)
        u₊, F₊, _, info = step!!(s.inner, m, s.rprob, u, Fx, nothing)
        @test info.code == NONFINITE
        @test u₊ == u₀ && F₊ == Fx
        # `start!!` sets `δ₀`, and each step multiplies `δ` by `‖F(xₖ)‖ / ‖F(xₖ₊₁)‖`, capped at
        # `δ_max`; the step solves `(J + I/δ) d = −F` and accepts it
        inner = start!!(s.inner, m, s.rprob, u, Fx, nothing, sqrt(sum(abs2, Fx)))
        @test inner.rule == PseudoTimeStep(T(0.1))
        δ = T(0.1)
        for k in 1:30
            f = sqrt(sum(abs2, Fx))
            d = -((well_hessian(u) + I2(T) / δ) \ Fx)
            u_expected = u + d
            u, Fx, inner, info = step!!(inner, m, s.rprob, u, Fx, nothing)
            @test info.code == SUCCESS
            @test isapprox(u, u_expected; rtol = 64 * eps(T), atol = 64 * eps(T))
            f₊ = sqrt(sum(abs2, Fx))
            δ = min(δ * (f / f₊), T(1e3))
            @test isapprox(inner.rule.δ, δ; rtol = 64 * eps(T))
            f₊ == 0 && break
        end
        @test inner.rule.δ == T(1e3)
        # a solve leaves `δ = NaN` in the state it returns, and the next solve starts at `δ₀` again
        solver = init(prob, T[0.1, 1], m)
        x, status = solve!(solver)
        @test status.code == SUCCESS
        @test isnan(solver.state.inner.rule.δ)
        x′, status′ = solve!(solver, T[0.1, 1])
        @test x′ == x && status′.iterations == status.iterations
    end
end

@testset "the evolution of δ at the ends of the float range" begin
    # `δ ‖F(xₖ)‖ / ‖F(xₖ₊₁)‖` where the ratio alone underflows to zero, `floatmin / √floatmax`,
    # or overflows, `√floatmax / floatmin`, while `δ₊` is a normal number below the cap; the
    # reference is the product in `BigFloat`, rounded once
    @testset "$T" for T in (Float16, REAL_ELTYPES...)
        pt = PseudoTransient(T, one(T); δ_max = floatmax(T) / 4)
        M = sqrt(floatmax(T))
        for (δ, f, f₊) in ((floatmax(T) / 4, floatmin(T), M), (floatmin(T), M, floatmin(T)))
            expected = T(big(δ) * big(f) / big(f₊))
            @test isfinite(expected) && expected >= floatmin(T) && expected < pt.δ_max
            @test isapprox(pt_next_step(pt, δ, f, f₊), expected; rtol = 8 * eps(T))
        end
        # a residual of zero after the step gives the cap
        @test pt_next_step(pt, one(T), one(T), zero(T)) == pt.δ_max
    end
end

@testset "the guards of a step" begin
    prob = StationarityProblem(well)
    @testset "$T" for T in REAL_ELTYPES
        m = Newton(; globalization = PseudoTransient(T(1)))
        s = init_state(m, prob, T[0, 1], nothing)
        u = s.u_work
        Fx = residual!!(similar(u), s.rprob, u, nothing)
        f = sqrt(sum(abs2, Fx))
        inner = start!!(s.inner, m, s.rprob, u, Fx, nothing, f)
        # `δ = 1` makes `J + I/δ = diag(3x² − 1 + 1, 2)` singular at `x = 0`
        u₀ = copy(u)
        u₊, _, inner₊, info = step!!(inner, m, s.rprob, u, Fx, nothing)
        @test info.code == SINGULAR && u₊ == u₀ && inner₊.rule == inner.rule
        # an infinite pseudo-time step, whose shift `1/δ = 0` is finite, a pseudo-time step of
        # zero, whose shift is not, and a non-finite residual
        for δ in (T(Inf), zero(T))
            u₊, _, _, info = step!!(with_rule(inner, PseudoTimeStep(δ)), m, s.rprob, u, Fx,
                nothing)
            @test info.code == NONFINITE && u₊ == u₀
        end
        u₊, _, _, info = step!!(inner, m, s.rprob, u, fill(T(NaN), 2), nothing)
        @test info.code == NONFINITE && u₊ == u₀
        # at a root the step is zero, and `δ` stays
        root = T[1, 0]
        Fr = residual!!(similar(root), s.rprob, root, nothing)
        r₊, _, inner₊, info = step!!(inner, m, s.rprob, copy(root), Fr, nothing)
        @test info.code == SUCCESS && iszero(info.stepnorm) && r₊ == root
        @test inner₊.rule == inner.rule
        # a regularisation `μ` adds to the shift: `(J + (μ + 1/δ) I) d = −F`
        mμ = Newton(; globalization = PseudoTransient(T(0.25)), regularization = 0.5)
        sμ = init_state(mμ, prob, T[0.1, 1], nothing)
        v = sμ.u_work
        Fv = residual!!(similar(v), sμ.rprob, v, nothing)
        innerμ = start!!(sμ.inner, mμ, sμ.rprob, v, Fv, nothing, sqrt(sum(abs2, Fv)))
        expected = v - (well_hessian(v) + (T(0.5) + 4) * I2(T)) \ Fv
        v₊, _, _, info = step!!(innerμ, mμ, sμ.rprob, copy(v), Fv, nothing)
        @test info.code == SUCCESS
        @test isapprox(v₊, expected; rtol = 64 * eps(T))
    end
end

@testset "a short step is no stall" begin
    # every step is `limited`, so the step test does not end the solve while the shift keeps the
    # steps short, even with a step tolerance far above them
    @testset "$T" for T in REAL_ELTYPES
        x, status = solve(StationarityProblem(well), T[0.1, 1],
            Newton(; globalization = PseudoTransient(0.1)); x_reltol = T(0.5),
            max_iterations = 50)
        @test status.code == SUCCESS
        @test isapprox(x, T[1, 0]; atol = sqrt(eps(T)))
    end
end

@testset "a non-finite residual" begin
    # the gradient `x − 2` turns `NaN` past `x = 1`, which the steps toward 2 cross
    S(x, p) = (x[1] - 2)^2 / 2 + ifelse(x[1] > 1, x[1] * eltype(x)(NaN), zero(x[1]))
    @testset "$T" for T in REAL_ELTYPES
        x, status = solve(StationarityProblem(S), T[0], Newton(;
            globalization = PseudoTransient(0.1)))
        @test status.code == NONFINITE
        @test all(isfinite, x)
    end
end

# On a Julia it does not support, JET loads empty stubs that throw. JET 0.12 exports
# `JET_AVAILABLE` to say so; JET 0.9 and 0.10 name the same flag `JET_LOADABLE`.
const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

@kernel function pt_kernel!(codes, xs, prob, method, state₀, opt, @Const(x₀s))
    i = @index(Global, Linear)
    @inbounds begin
        x, st, _ = solve!!(state₀, prob, x₀s[i], method, nothing, opt)
        codes[i] = st.code
        xs[i] = x
    end
end

function pt_allocations(state::S, prob::P, x₀::X, m::M, opt::O) where {S, P, X, M, O}
    (
        solve!!(state, prob, x₀, m, nothing, opt); @allocated solve!!(
            state, prob, x₀, m, nothing, opt))
end
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "the kernel path: host and kernel agree, inferred, allocation-free, $R" for R in REAL_ELTYPES
    prob = StationarityProblem(well)
    m = Adapt.adapt(ToReal{R}(), Newton(; globalization = PseudoTransient(0.1)))
    opt = Options(R; max_iterations = 50)
    x₀s = [SVector{2, R}(0.1, 1), SVector{2, R}(-0.1, 1), SVector{2, R}(0.5, -1),
        SVector{2, R}(-3, 4)]
    host = [solve!!(init_state(m, prob, x₀, nothing), prob, x₀, m, nothing, opt)
            for x₀ in x₀s]
    codes = Vector{GeometricSolvers.ReturnCode}(undef, 4)
    xs = Vector{SVector{2, R}}(undef, 4)
    state₀ = init_state(m, prob, x₀s[1], nothing)
    @test isbits(m) && isbits(state₀)
    pt_kernel!(KA_BACKEND)(codes, xs, prob, m, state₀, opt, x₀s; ndrange = 4)
    synchronize(KA_BACKEND)
    @test codes == [h[2].code for h in host]
    @test xs == [h[1] for h in host]
    @test all(==(SUCCESS), codes)

    @test @inferred(solve!!(state₀, prob, x₀s[1], m, nothing, opt))[2].code == SUCCESS
    # `--check-bounds=yes` lets the static factors' `MMatrix` escape to the heap, so a run that
    # forces it skips this assertion
    if Base.JLOptions().check_bounds != 1
        @test pt_allocations(state₀, prob, x₀s[1], m, opt) == 0
    end
    # the control: the barrier sees an allocation where there is one
    @test allocations(collect, x₀s[1]) > 0
    if JET_WORKS
        @test_opt solve!!(state₀, prob, x₀s[1], m, nothing, opt)
    else
        @test_skip JET_WORKS  # issue #8
    end

    # a device array, with an action that broadcasts: `S(x) = ‖x − 1‖²/2`, minimum at `(1, 1)`
    S(x, p) = sum(abs2, x .- 1) / 2
    x, status = solve(StationarityProblem(S), JLArray(R[0.1, 1]), m; max_iterations = 50)
    @test status.code == SUCCESS
    @test x isa JLArray{R}
    @test isapprox(Array(x), R[1, 1]; atol = sqrt(eps(R)))
end
