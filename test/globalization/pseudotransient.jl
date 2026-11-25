# `PseudoTransient`, the pseudo-transient continuation of `Newton` for a `StationarityProblem`:
# the double well whose saddle `Newton()` reaches and whose minimum the shifted steps reach, the
# parameters and their validation, the rule state that each solve starts at `δ₀` and that a raw
# step without `start!!` reads as `NaN`, the switched evolution relaxation of `δ` and its cap, a
# non-finite residual, the refusals, and a linear gradient scaled to the ends of the float range.

using LinearAlgebra: Diagonal, isposdef
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: KroneckerFactorization, LeastSquaresProblem, NONFINITE,
                        NonlinearProblem, PseudoTransient, PseudoTimeStep, Refinement,
                        SINGULAR,
                        StationarityProblem, SUCCESS, init, init_state, residual!!, solve,
                        solve!, start!!, step!!, with_rule

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
        @test u₊ == u₀
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
