# The contraction-rate stopping test, the round-off stop, and a factor kept across a sequence of
# `solve!` calls: the rate `θₖ = ‖Δₖ‖ / ‖Δₖ₋₁‖` that `record` measures, the `SUCCESS` of the rate
# test `θₖ / (1 - θₖ) ‖Δₖ‖ ≤ κ max(x_abstol, x_reltol ‖x‖)`, the `STALLED` of `roundoff = true`,
# and the factorisations of `Newton(; reuse = k, keep_factor = true)` and of
# `solve!(solver, x, p; refactorize = true)`.

using Adapt: Adapt
using JET: JET
using LinearAlgebra: I
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: L2Norm, LUFactorization, MAXITERS, Newton, NonlinearProblem,
                        Options, PerSystem,
                        QuasiNewton, STALLED, SUCCESS, SolverStatus, StepInfo, init,
                        init_state, record, solve, solve!, solve_batch!, stopping_test

include("../helpers/matrix.jl")

# The scalar linear residual `F(x) = a x` with the constant Jacobian `b`: a Newton step maps `x` to
# `(1 - a / b) x`, a linear contraction of rate `θ = |1 - a / b|` where `θ < 1`.
linear_problem(a) = NonlinearProblem((x, p) -> a * x)
linear_newton(b) = Newton(; jacobian = (x, p) -> b)

# Options under which only the iteration cap ends a solve of `linear_problem`: no residual test, no
# step test and a rate test whose bound is `0`.
function capped_options(::Type{T}, k) where {T}
    return Options(T; f_reltol = 0, x_reltol = 0, max_iterations = k)
end

@testset "the computed rate of a linear contraction is |1 - a/b|, $T" for T in REAL_ELTYPES
    # `(a, b)`: rates `1/4` and `1/2` that are exact in `T`, `2/3` and `0.7` that are not, and
    # `1/2` with alternating sign
    for (a, b) in ((T(3) / 4, one(T)), (one(T), T(3)), (T(3) / 10, one(T)),
        (T(3) / 2, one(T)))
        θ = abs(1 - big(a) / big(b))
        prob = linear_problem(a)
        m = linear_newton(b)
        x₁, st₁ = solve(prob, one(T), m; options = capped_options(T, 1))
        @test st₁.code == MAXITERS
        @test isnan(st₁.rate)
        @test x₁ != one(T)
        for k in 2:8
            _, st = solve(prob, one(T), m; options = capped_options(T, k))
            @test st.code == MAXITERS
            @test st.iterations == k
            @test st.rate isa T
            @test abs(st.rate - θ) / θ <= 8 * eps(T)
        end
    end
end

@testset "record measures the rate of a step against the step before it, $T" for T in REAL_ELTYPES
    st₀ = SolverStatus{T}(SUCCESS, Int32(0), one(T), zero(T), Int32(0), false, false)
    @test isnan(st₀.rate)
    @test !st₀.contracted && !st₀.reused
    st₁ = record(st₀, StepInfo{T}(SUCCESS, T(1) / 2, T(3) / 4, Int32(0), false, false))
    # the first step has no step before it
    @test isnan(st₁.rate)
    st₂ = record(st₁, StepInfo{T}(SUCCESS, T(1) / 4, T(3) / 8, Int32(0), false, false))
    @test st₂.rate === T(1) / 2
    st₃ = record(st₂, StepInfo{T}(SUCCESS, T(1) / 8, zero(T), Int32(0), false, false))
    @test st₃.rate === zero(T)
    # a step after a zero step has no rate
    st₄ = record(st₃, StepInfo{T}(SUCCESS, T(1) / 16, T(1) / 16, Int32(0), false, false))
    @test isnan(st₄.rate)
    @test (@inferred record(st₃, StepInfo{T}(
        SUCCESS, T(1), T(1), Int32(0), false, false))) isa
          SolverStatus{T}
end

@testset "the rate test returns SUCCESS before the residual test holds, $T" for T in REAL_ELTYPES
    # rate `θ = 1/100`: the step norms are about `0.99`, `0.0099` and `9.9e-5`. At the third step
    # `θ / (1 - θ) ‖Δ‖ ≈ 1e-6 ≤ κ x_abstol = 1e-5`, while the residual test, with `f_reltol = 0`,
    # does not hold at any nonzero iterate
    prob = linear_problem(T(99) / 100)
    m = linear_newton(one(T))
    opt = Options(T; f_reltol = 0, x_abstol = T(1e-4))
    @test opt.kappa === T(1) / 10
    x, st = solve(prob, one(T), m; options = opt)
    @test st.code == SUCCESS
    @test st.contracted
    @test !st.stationary
    @test stopping_test(st) == :contraction
    @test st.iterations == 3
    @test st.fnorm > 0
    @test x != 0
    @test abs(st.rate - T(1) / 100) <= T(1) / 1000
    # with `κ = 1/1000` the rate test fails at the third step, and the step test `‖Δ‖ ≤ x_abstol`,
    # which holds there, returns `STALLED`
    _, st_κ = solve(prob, one(T), m;
        options = Options(T; f_reltol = 0, x_abstol = T(1e-4),
            kappa = T(1) / 1000))
    @test st_κ.code == STALLED
    @test !st_κ.contracted
    @test stopping_test(st_κ) == :none
    @test st_κ.iterations == 3
    # `min_iterations = 4` bounds the rate test, which then holds at the fourth step
    _, st_min = solve(prob, one(T), m;
        options = Options(T; f_reltol = 0, x_abstol = T(1e-4),
            min_iterations = 4))
    @test st_min.code == SUCCESS
    @test st_min.contracted
    @test st_min.iterations == 4
    # the bound holds with equality: rate `1/2`, so `θ / (1 - θ) ‖Δ₂‖ = ‖Δ₂‖ = 1/4`, and
    # `κ x_abstol = 1/4`, all exact in `T`; `min_iterations = 2` keeps the step test off the first
    # step, and it would hold at the second too
    _, st_eq = solve(linear_problem(T(1) / 2), one(T), m;
        options = Options(T; f_reltol = 0, x_abstol = T(1) / 2, kappa = T(1) / 2,
            min_iterations = 2))
    @test st_eq.code == SUCCESS
    @test st_eq.contracted
    @test st_eq.iterations == 2
    # a zero step has rate `0`, and the rate test needs a step of nonzero norm: from `x₀ = 2` the
    # step along the Jacobian `4/3` lands at `1/2`, and the step along the infinite Jacobian there
    # is zero
    zero_step = Newton(; jacobian = (x, p) -> x == 2 ? T(4) / 3 : T(Inf))
    x_z, st_z = solve(linear_problem(one(T)), T(2), zero_step;
        options = Options(T; f_reltol = 0, x_abstol = T(1)))
    @test st_z.code == STALLED
    @test !st_z.contracted
    @test st_z.iterations == 2
    @test st_z.rate == 0
    @test x_z == T(1) / 2
    # the residual test still returns `:residual`
    _, st_res = solve(prob, one(T), m; options = Options(T; f_abstol = T(1e-1)))
    @test st_res.code == SUCCESS
    @test !st_res.contracted
    @test stopping_test(st_res) == :residual
end

@testset "the stationarity test comes before the rate test, $T" for T in REAL_ELTYPES
    # `r(x) = (x² - 2, x - 1)` has no root, and Gauss–Newton converges to its least-squares point
    # at a linear rate: the solve ends on the stationarity test, also where the rate test holds
    r(x, p) = SVector(x[1]^2 - T(2), x[1] - one(T))
    j(x, p) = SMatrix{2, 1, T, 2}(T(2) * x[1], one(T))
    prob = GeometricSolvers.LeastSquaresProblem(r, j)
    _, st = solve(prob, SVector{1, T}(3), GeometricSolvers.GaussNewton())
    @test st.code == SUCCESS
    @test st.stationary
    @test !st.contracted
    @test stopping_test(st) == :stationarity
    # where the stationarity test fails, the rate test under `x_abstol = 10⁻²` ends the solve
    _, st_c = solve(prob, SVector{1, T}(3), GeometricSolvers.GaussNewton();
        options = Options(T; x_abstol = T(1e-2)))
    @test st_c.code == SUCCESS
    @test st_c.contracted
    @test !st_c.stationary
end

@testset "a Deflated problem reports the rate of its steps, $T" for T in REAL_ELTYPES
    # its residual test reads a status of its own, which keeps the rate of the step
    prob = GeometricSolvers.Deflated(NonlinearProblem((x, p) -> SVector(x[1]^2 - 2)),
        (SVector{1, T}(-1.41),))
    _, st = solve(prob, SVector{1, T}(3), Newton())
    @test st.code == SUCCESS
    @test st.iterations >= 2
    @test 0 < st.rate < 1
end

@testset "a diverging scalar of rate 3/2 never returns SUCCESS, $T" for T in REAL_ELTYPES
    # `F(x) = -x/2` with `b = 1` maps `x` to `3x/2`. Without the guard `θ < 1`, the rate test
    # `θ / (1 - θ) ‖Δ‖ = -3 ‖Δ‖ ≤ κ x_reltol ‖x‖` would hold at the second step
    prob = linear_problem(-one(T) / 2)
    m = linear_newton(one(T))
    for k in 2:12
        _, st = solve(prob, one(T), m; options = Options(T; max_iterations = k))
        @test st.code == MAXITERS
        @test !st.contracted
        @test st.rate == T(3) / 2
    end
    _, st = solve(prob, one(T), m)
    @test st.code == MAXITERS
    @test !st.contracted
end

# A method whose step divides `x` by `100` and reports the code `code` and the flag `limited`, so
# that the solver loop sees steps of rate `1/100` that a radius limits or that stalled.
struct ScriptedStep
    code::GeometricSolvers.ReturnCode
    limited::Bool
end
GeometricSolvers.init_state(::ScriptedStep, prob, x, p) = nothing
function GeometricSolvers.step!!(state, m::ScriptedStep, prob, x, Fx, p)
    x₊ = x / 100
    F₊ = prob.F(x₊, p)
    T = typeof(x)
    return x₊, F₊, state,
    StepInfo{T}(m.code, abs(F₊), abs(x₊ - x), Int32(0), false, m.limited)
end

@testset "the rate test skips a limited and a STALLED step, $T" for T in REAL_ELTYPES
    # under `x_abstol = 1` the rate test holds at the second step of each method; the step test
    # does not read these steps either
    prob = linear_problem(one(T))
    opt = Options(T; f_reltol = 0, x_abstol = 1, max_iterations = 5)
    _, st_limited = solve(prob, one(T), ScriptedStep(SUCCESS, true); options = opt)
    @test st_limited.code == MAXITERS
    @test st_limited.iterations == 5
    @test !st_limited.contracted
    _, st_stalled = solve(prob, one(T), ScriptedStep(STALLED, false); options = opt)
    @test st_stalled.code == STALLED
    @test st_stalled.iterations == 2
    @test !st_stalled.contracted
    # the same steps, neither limited nor stalled, end on the rate test at the second step; under
    # `x_abstol = 1/2` the first step, of norm `0.99`, passes no step test
    _, st_plain = solve(prob, one(T), ScriptedStep(SUCCESS, false);
        options = Options(T; f_reltol = 0, x_abstol = T(1) / 2, max_iterations = 5))
    @test st_plain.code == SUCCESS
    @test st_plain.contracted
    @test st_plain.iterations == 2
end

@testset "a NaN rate never stops a solve" begin
    # the first step of the rate-3/2 scalar has no rate, and under `roundoff = true` neither it nor
    # the next step of rate `3/2 ≥ 1` stops it: no step before it had a rate below `1`
    prob = linear_problem(-1.0 / 2)
    m = linear_newton(1.0)
    _, st = solve(prob, 1.0, m;
        options = Options(Float64; roundoff = true, max_iterations = 5))
    @test st.code == MAXITERS
    @test st.iterations == 5
end

# A residual with a round-off floor: `x - 1` plus a perturbation of size `δ` whose phase turns
# `10⁶` times over a change of `x` by `δ`, so that it takes an unrelated value at each iterate, as
# the rounding error of a residual evaluated in finite precision does near its root. With the
# Jacobian `1`, every step after the first has a norm of order `δ`, and the rate of each is the
# ratio of two such perturbations. `δ = 64 √eps(T)` keeps the rate test off at the second step:
# there `θ / (1 - θ) ‖Δ‖ ≈ δ² / 2` is above `κ x_reltol ‖x‖ = eps(T) / 5`.
struct FloorResidual{T}
    δ::T
end
(r::FloorResidual{T})(x, p) where {T} = (x - 1) + r.δ * cos(x / (r.δ * T(1e-6)))

@testset "roundoff = true stops a solve whose increment stops decreasing, $T" for T in REAL_ELTYPES
    δ = 64 * sqrt(eps(T))
    prob = NonlinearProblem(FloorResidual(δ))
    m = linear_newton(one(T))
    opt(roundoff) = Options(T; f_reltol = 0, max_iterations = 60, roundoff)
    x₀ = T(3)
    # the iterates of the solve, by hand: `x₊ = x - F(x)`
    xs = [x₀]
    for _ in 1:60
        push!(xs, xs[end] - FloorResidual(δ)(xs[end], nothing))
    end
    # the norm of the direction `-F(x)`, which the solve measures, not `|x₊ - x|`, whose rounding
    # is `eps(T)` against a step of order `δ`
    steps = [abs(FloorResidual(δ)(xs[k], nothing)) for k in 1:60]
    θs = [k == 1 ? T(NaN) : steps[k] / steps[k - 1] for k in 1:60]
    # the first step of rate `θₖ ≥ 1` after a step of rate `θₖ₋₁ < 1`
    k_stop = findfirst(k -> k ≥ 2 && θs[k] ≥ 1 && θs[k - 1] < 1, 1:60)
    @test k_stop !== nothing && k_stop < 60
    @test all(!iszero, steps)

    x, st = solve(prob, x₀, m; options = opt(true))
    @test st.code == STALLED
    @test st.iterations == k_stop
    @test st.iterations < 60
    @test x == xs[k_stop + 1]
    @test isapprox(st.rate, θs[k_stop]; rtol = 8 * eps(T))
    @test st.rate >= 1

    x_off, st_off = solve(prob, x₀, m; options = opt(false))
    @test st_off.code == MAXITERS
    @test st_off.iterations == 60
    @test x_off == xs[end]

    # `min_iterations` bounds the stop: past `k_stop`, the first step of rate `θₖ ≥ 1` after a
    # step of rate below `1`
    k_min = findfirst(k -> k > k_stop && θs[k] ≥ 1 && θs[k - 1] < 1, 1:60)
    x_min, st_min = solve(prob, x₀, m;
        options = Options(T; f_reltol = 0, max_iterations = 60, roundoff = true,
            min_iterations = k_stop + 1))
    @test st_min.code == STALLED
    @test st_min.iterations == k_min
    @test x_min == xs[k_min + 1]
end

@testset "roundoff = true stops at a rate of exactly 1, $T" for T in REAL_ELTYPES
    # `F(x) = x` from `x₀ = 4` with the Jacobian `-1` at `x = 1` and `2` elsewhere: the steps
    # `-2`, `-1` and `+1` have the rates `1/2` and `1`, all exact, and the iterates then cycle
    # between `1` and `2`
    m = Newton(; jacobian = (x, p) -> x == 1 ? -one(T) : T(2))
    x, st = solve(linear_problem(one(T)), T(4), m; options = Options(T; roundoff = true))
    @test st.code == STALLED
    @test st.iterations == 3
    @test st.rate == 1
    @test x == 2
    _, st_off = solve(linear_problem(one(T)), T(4), m)
    @test st_off.code == MAXITERS
end

@testset "Options take κ in (0, 1) and the roundoff flag, $T" for T in REAL_ELTYPES
    opt = Options(T)
    @test opt.kappa === T(1) / 10
    @test opt.roundoff === false
    @test isbitstype(typeof(opt))
    opt2 = Options(T; kappa = 0.5, roundoff = true)
    @test opt2.kappa === T(1) / 2
    @test opt2.roundoff === true
    @test opt2 isa Options{T, L2Norm}
    for κ in (0, 1, -0.5, 1.5, NaN, Inf, -Inf, prevfloat(0.0), nextfloat(1.0))
        @test_throws r"^ArgumentError: kappa must be finite and in \(0, 1\)" Options(T;
            kappa = κ)
    end
    @test Options(T; kappa = nextfloat(zero(T))).kappa === nextfloat(zero(T))
    @test Options(T; kappa = prevfloat(one(T))).kappa === prevfloat(one(T))
    # a `Float64` value that rounds to `0` or to `1` in `Float32` is that value
    if T === Float32
        for κ in (nextfloat(0.0), prevfloat(1.0))
            @test_throws r"^ArgumentError: kappa must be finite and in \(0, 1\)" Options(T;
                kappa = κ)
        end
    end
    # the positional form takes the defaults
    opt_p = Options{T}(0, 1, 0, 0, 0, 10, 2)
    @test opt_p.kappa === T(1) / 10
    @test opt_p.roundoff === false
end

@testset "keep_factor needs reuse" begin
    @test_throws r"^ArgumentError: Newton keep_factor = true needs reuse" Newton(;
        keep_factor = true)
    m = Newton(; reuse = 3, keep_factor = true)
    @test m.keep_factor === true
    @test Newton(; reuse = 3).keep_factor === false
    @test QuasiNewton(3; keep_factor = true).keep_factor === true
    @test isbits(m)
end

# A Jacobian callable that counts its calls: every factorisation of a `Newton` step reads the
# Jacobian once, so the count is the number of factorisations.
struct CountingJacobian{F}
    J::F
    calls::Base.RefValue{Int}
end
(c::CountingJacobian)(x, p) = (c.calls[] += 1; c.J(x, p))

# One implicit-midpoint step of the pendulum `q̈ = -sin(q)` in `q = (angle, velocity)`, from `q₀`
# with step `h`, both in `p`: `F(q₁) = q₁ - q₀ - h f((q₀ + q₁) / 2)`.
pendulum(q) = SVector(q[2], -sin(q[1]))
function pendulum_jacobian(q)
    SMatrix{2, 2}(zero(eltype(q)), -cos(q[1]), one(eltype(q)),
        zero(eltype(q)))
end
midpoint_residual(x, p) = x - p.q₀ - p.h * pendulum((p.q₀ + x) / 2)
midpoint_jacobian(x, p) = I - p.h / 2 * pendulum_jacobian((p.q₀ + x) / 2)

# Ten steps of the midpoint rule with `Newton(; reuse = k, keep_factor)`, from `q₀ = (1, 0)` with
# step `h`, each solve started at the last point. Returns the factorisations and the statuses.
function midpoint_sequence(
        ::Type{T}, keep_factor; k = 5, h = T(3) / 20, steps = 10) where {T}
    calls = Ref(0)
    m = Newton(; jacobian = CountingJacobian(midpoint_jacobian, calls), reuse = k,
        keep_factor)
    q₀ = SVector{2, T}(1, 0)
    solver = init(NonlinearProblem(midpoint_residual), q₀, m, (; q₀, h))
    statuses = SolverStatus{T}[]
    for _ in 1:steps
        q₁, st = solve!(solver, q₀, (; q₀, h))
        push!(statuses, st)
        q₀ = q₁
    end
    return calls[], statuses, q₀
end

# The factorisations of the ten steps with `reuse = 5`, without and with `keep_factor`. Without it,
# each solve factorises at its first step and converges before its factor reaches age `5`. With
# it, a solve that ends after a second step of rate `θ > 10⁻³` leaves age `0`, which some solves of
# each precision do, and every other solve hands its factor on.
const MIDPOINT_COUNTS = Dict(Float32 => (10, 5), Float64 => (10, 7))

@testset "a kept factor factorises fewer times over ten midpoint steps, $T" for T in REAL_ELTYPES
    n_fresh, st_fresh, q_fresh = midpoint_sequence(T, false)
    n_kept, st_kept, q_kept = midpoint_sequence(T, true)
    @test all(st -> st.code == SUCCESS, st_fresh)
    @test all(st -> st.code == SUCCESS, st_kept)
    @test n_fresh isa Int && n_kept isa Int
    @test (n_fresh, n_kept) == MIDPOINT_COUNTS[T]
    @test n_kept < n_fresh
    # some solve of the sequence ends after more than one step with a rate above `10⁻³`, so the
    # rule that drops its factor decides the count
    @test any(st -> st.iterations > 1 && st.rate > T(1 // 1000), st_kept)
    # both sequences reach the same point, to the tolerance of the residual test
    @test isapprox(q_kept, q_fresh; rtol = 100 * sqrt(eps(T)))
end

@testset "a step of rate θ ≥ 1 along a reused factor refreshes it, $T" for T in REAL_ELTYPES
    # `F(x) = x` with the Jacobian `2/5` at `x = 2` and the exact `1` elsewhere: from `x₀ = 2` the
    # fresh step lands at `x₁ = -3`, and the step along that factor at `x₂ = 9/2`, with rate
    # `θ₂ = 3/2`. The refresh at the third step reads the exact Jacobian and lands at the root;
    # along the old factor the iterates would grow by `3/2` at each step.
    calls = Ref(0)
    J = CountingJacobian((x, p) -> x == 2 ? T(2) / 5 : one(T), calls)
    x, st = solve(linear_problem(one(T)), T(2), QuasiNewton(10; jacobian = J))
    @test st.code == SUCCESS
    @test st.iterations == 3
    @test x == 0
    @test calls[] == 2
    @test !st.reused
    # the same refresh where the factorisation runs in `Float32` below the working precision
    if T === Float64
        calls[] = 0
        x_mixed, st_mixed = solve(linear_problem(one(T)), T(2),
            QuasiNewton(10; jacobian = J, linsolve = LUFactorization(Float32)))
        @test st_mixed.code == SUCCESS
        @test st_mixed.iterations == 3
        @test x_mixed == 0
        @test calls[] == 2
    end
end

# A preconditioner that the rule state of its `GMRES` names and that no step applies here.
struct UnappliedPreconditioner end

@testset "a GMRES step records whether it reused its preconditioner" begin
    # the rule state after a step of a `Newton` with a preconditioned `GMRES` and `reuse = 3`,
    # which the rate rule reads as it reads the rule state of a factor
    FA = GeometricSolvers.FactorAge
    rule = GeometricSolvers.preconditioner_rule
    m = Newton(; reuse = 3,
        linsolve = GeometricSolvers.GMRES(; preconditioner = UnappliedPreconditioner()))
    @test rule(FA(Int32(1), false, false), m, false, SUCCESS, Int32(0)).reused
    @test !rule(FA(Int32(1), false, true), m, true, SUCCESS, Int32(0)).reused
end

@testset "a step of rate θ ≥ 1 on a fresh factor refreshes nothing, $T" for T in REAL_ELTYPES
    # as above, with the Jacobian `1/4` at `x₂ = 9/2`: the refreshed third step lands at
    # `x₃ = -27/2`, with rate `θ₃ = 12/5 ≥ 1` on its fresh factor, and the fourth step steps along
    # that factor
    calls = Ref(0)
    J = CountingJacobian((x, p) -> x == 2 ? T(2) / 5 : (x == T(9) / 2 ? T(1) / 4 : one(T)),
        calls)
    x, st = solve(linear_problem(one(T)), T(2), QuasiNewton(10; jacobian = J);
        options = Options(T; max_iterations = 4))
    @test st.code == MAXITERS
    @test x == T(81) / 2
    @test calls[] == 2
    @test st.reused
    @test st.rate == 3
end

@testset "refactorize = true after a change of h adds one factorisation, $T" for T in REAL_ELTYPES
    # two solvers take the same steps; at the step after the change of `h`, one of them passes
    # `refactorize = true`
    function run(refactorize)
        calls = Ref(0)
        m = Newton(; jacobian = CountingJacobian(midpoint_jacobian, calls), reuse = 5,
            keep_factor = true)
        h = T(1) / 20
        q₀ = SVector{2, T}(1, 0)
        solver = init(NonlinearProblem(midpoint_residual), q₀, m, (; q₀, h))
        for _ in 1:3
            q₀, _ = solve!(solver, q₀, (; q₀, h))
        end
        h = T(1) / 40
        before = calls[]
        q₁, st = solve!(solver, q₀, (; q₀, h); refactorize)
        return calls[] - before, st, q₁
    end
    n_plain, st_plain, q_plain = run(false)
    n_refresh, st_refresh, q_refresh = run(true)
    @test st_plain.code == SUCCESS
    @test st_refresh.code == SUCCESS
    @test n_refresh == n_plain + 1
    @test isapprox(q_refresh, q_plain; rtol = 100 * sqrt(eps(T)))
    # a solver without reuse takes the keyword and factorises as without it
    calls = Ref(0)
    m = Newton(; jacobian = CountingJacobian(midpoint_jacobian, calls))
    q₀ = SVector{2, T}(1, 0)
    p = (; q₀, h = T(1) / 20)
    solver = init(NonlinearProblem(midpoint_residual), q₀, m, p)
    _, st = solve!(solver, q₀, p; refactorize = true)
    @test st.code == SUCCESS
    @test calls[] == st.iterations
end

@testset "no factor survives a solve that is not SUCCESS, $T" for T in REAL_ELTYPES
    # Two solves of one step each from the far start `q₀ + 1`, whose residual is about `1.4` and
    # about `10⁻²` after one step. Under `f_abstol = 1/2` the first solve ends with `SUCCESS` after
    # one step, and the second steps along its factor; under the default tolerances the first ends
    # with `MAXITERS`, and the second factorises afresh. Returns the factorisations of the second
    # solve and its status.
    function second_solve(f_abstol)
        calls = Ref(0)
        m = Newton(; jacobian = CountingJacobian(midpoint_jacobian, calls), reuse = 10,
            keep_factor = true)
        q₀ = SVector{2, T}(1, 0)
        p = (; q₀, h = T(1) / 20)
        solver = init(NonlinearProblem(midpoint_residual), q₀, m, p;
            options = Options(T; f_abstol, max_iterations = 1))
        _, st₁ = solve!(solver, q₀ .+ 1, p)
        @test calls[] == 1
        before = calls[]
        _, st₂ = solve!(solver, q₀ .+ 1, p)
        n₂ = calls[] - before
        # a start whose residual, about `0.04`, passes `f_abstol = 1/2` takes no step, and so no
        # step along a reused factor
        if f_abstol > 0
            _, st₀ = solve!(solver, q₀, p)
            @test st₀.iterations == 0
            @test !st₀.reused
        end
        return st₁, n₂, st₂
    end
    st_ok, n_ok, st₂_ok = second_solve(T(1) / 2)
    @test st_ok.code == SUCCESS
    @test st_ok.iterations == 1
    @test n_ok == 0
    @test st₂_ok.reused
    @test st₂_ok.iterations == 1
    st_fail, n_fail, st₂_fail = second_solve(zero(T))
    @test st_fail.code == MAXITERS
    @test n_fail == 1
    @test !st₂_fail.reused
end

@testset "a solve that promoted its factorisation keeps no factor" begin
    # A `Float32` factorisation of a Jacobian that is singular in `Float32` promotes to the
    # working precision `Float64`. Each solve converges in two steps with a last rate below
    # `10⁻³`, which would keep the factor, but the next solve starts unpromoted and does not read
    # the `Float64` factor: it factorises afresh, and promotes again.
    calls = Ref(0)
    A = SMatrix{2, 2}(1.0, 1.0, 1.0, 1.0 + 1e-9)
    F(x, p) = A * x - p .+ 1e-3 .* x .^ 3
    J(x, p) = A + SMatrix{2, 2}(3e-3 * x[1]^2, 0.0, 0.0, 3e-3 * x[2]^2)
    m = Newton(;
        jacobian = CountingJacobian(J, calls), linsolve = LUFactorization(Float32),
        reuse = 5, keep_factor = true)
    b = SVector(2.0, 2.0 + 1e-9)
    x₀ = SVector(0.0, 0.0)
    solver = init(NonlinearProblem(F), x₀, m, b)
    for _ in 1:3
        before = calls[]
        _, st = solve!(solver, x₀, b)
        @test st.code == SUCCESS
        @test st.promoted
        @test st.iterations == 2
        @test st.rate <= 1e-3
        @test !st.reused
        @test calls[] - before == 2
    end
end

allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "the R1 solve with keep_factor and roundoff is inferred and allocation-free, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem(midpoint_residual)
    q₀ = SVector{2, T}(1, 0)
    p = (; q₀, h = T(3) / 20)
    m = Newton(; jacobian = midpoint_jacobian, reuse = 3, keep_factor = true)
    opt = Options(T; roundoff = true)
    @test (@inferred solve(prob, q₀, m, p; options = opt)) isa Tuple
    # `solve!!` takes the method in the working precision, as `solve` adapts it
    m_T = Adapt.adapt(GeometricSolvers.ToReal{T}(), m)
    state = init_state(m_T, prob, q₀, p)
    run(s) = GeometricSolvers.solve!!(s, prob, q₀, m_T, p, opt)
    @test (@inferred run(state)) isa Tuple
    # under `--check-bounds=yes` a bounds check lets the local arrays of `StaticLU` escape to the
    # heap, so the check is skipped where a run forces `--check-bounds=yes`
    if Base.JLOptions().check_bounds != 1
        @test allocations(run, state) == 0
    end
    @test allocations(collect, [1, 2]) > 0
    JET.@test_opt run(state)
end

@testset "the R1 and R2 states with keep_factor stay isbits, $T" for T in REAL_ELTYPES
    @test isbitstype(SolverStatus{T})
    @test isbitstype(Options{T, L2Norm})
    m = Newton(; jacobian = midpoint_jacobian, reuse = 3, keep_factor = true)
    q₀ = SVector{2, T}(1, 0)
    p = (; q₀, h = T(1) / 20)
    prob = NonlinearProblem(midpoint_residual)
    state = init_state(m, prob, q₀, p)
    @test isbitstype(typeof(state))
    # R2: the state `solve_batch!` prepares for its kernel from a host batch
    X = [q₀, q₀ .+ T(1) / 10]
    m_prep, opt_prep, state_prep = GeometricSolvers._prepare_batch(
        X, prob, m, Options(T), (;))
    @test isbitstype(typeof(m_prep))
    @test isbitstype(typeof(opt_prep))
    @test isbitstype(typeof(state_prep))
    S = Vector{SolverStatus{T}}(undef, 2)
    solve_batch!(X, S, prob, m, PerSystem([p, (; q₀ = q₀ .+ T(1) / 10, h = T(1) / 20)]))
    @test all(st -> st.code == SUCCESS, S)
end
