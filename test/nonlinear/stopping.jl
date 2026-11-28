# The contraction-rate stopping test, the round-off stop, and a factor kept across a sequence of
# `solve!` calls: the rate `θₖ = ‖Δₖ‖ / ‖Δₖ₋₁‖` that `record` measures, the `SUCCESS` of the rate
# test `θ̂ₖ / (1 - θ̂ₖ) ‖Δₖ‖ ≤ κ max(x_abstol, x_reltol ‖x‖)` with `θ̂ₖ = max(θₖ, θₖ₋₁)`, the
# `STALLED` of `roundoff = true`, and the factorisations of
# `Newton(; reuse = k, keep_factor = true)` and of `solve!(solver, x, p; refactorize = true)`.

using Adapt: Adapt
using ForwardDiff: ForwardDiff
using JET: JET
using LinearAlgebra: I
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: L2Norm, LUFactorization, MAXITERS, Newton, NonlinearProblem,
                        Options, PerSystem,
                        QuasiNewton, SINGULAR, STALLED, SUCCESS, SolverStatus, StepInfo,
                        init,
                        init_state, record, solve, solve!, solve_batch!, stopping_test
using GeometricSolvers: Additive, Anderson, Broyden, Fallback, FixedWeights, Multiplicative,
                        Picard, Repeat, SVDFactorization, Switch

include("../helpers/matrix.jl")
include("../helpers/standard_problems.jl")

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
    # the bound holds with equality: rate `1/2`, so `θ / (1 - θ) ‖Δ₃‖ = ‖Δ₃‖ = 1/8`, and
    # `κ x_abstol = 1/8`, all exact in `T`; `min_iterations = 3` keeps the step test off the
    # second step, and it would hold at the third too
    _, st_eq = solve(linear_problem(T(1) / 2), one(T), m;
        options = Options(T; f_reltol = 0, x_abstol = T(1) / 4, kappa = T(1) / 2,
            min_iterations = 3))
    @test st_eq.code == SUCCESS
    @test st_eq.contracted
    @test st_eq.iterations == 3
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

@testset "the rate test reads an SVD factor of full column rank, $T" for T in REAL_ELTYPES
    # the problem above, along an SVD factor, which reports its rank `1`: the rate test ends the
    # solve as along an LU factor, which reports no rank. A rank-deficient SVD factor, of check 2
    # in `test/nonlinear/diagnostics.jl`, leaves the solve to the step test.
    opt = Options(T; f_reltol = 0, x_abstol = T(1e-4))
    svd = GeometricSolvers.SVDFactorization()
    _, st = solve(linear_problem(T(99) / 100), one(T),
        Newton(; jacobian = (x, p) -> one(T), linsolve = svd); options = opt)
    @test st.code == SUCCESS
    @test st.rank == 1
    @test stopping_test(st) == :contraction
    @test st.iterations == 3
    _, st_v = solve(NonlinearProblem((x, p) -> T(99) / 100 * x), SVector{1, T}(1),
        Newton(; jacobian = (x, p) -> SMatrix{1, 1, T}(1), linsolve = svd); options = opt)
    @test st_v.code == SUCCESS
    @test st_v.rank == 1
    @test stopping_test(st_v) == :contraction
end

# Least-squares residuals in `x = (y, z₁, z₂)` for a `VariableProjection` that eliminates `y`. The
# reduced residual of `vp_deficient` reads `z₁ + z₂` alone, so its Jacobian in `z` has rank `1`;
# that of `vp_full` has rank `2`. Neither has a root, and Gauss–Newton steps contract linearly.
vp_deficient(x, p) = SVector(x[1] - x[2], (x[2] + x[3])^2 - 2, (x[2] + x[3]) - 7 // 5)
vp_full(x, p) = SVector(x[1] - x[2], x[2]^2 - 2, x[2] - 7 // 5, x[3] - x[2])
vp_deficient!(y, x, p) = (y .= vp_deficient(x, p); y)
vp_full!(y, x, p) = (y .= vp_full(x, p); y)

@testset "the rate test of a VariableProjection reads the rank of its outer factor, $T" for T in REAL_ELTYPES
    # under `x_abstol = 10⁻²` the rate test holds at the fourth step, with `g_tol = 0` that keeps
    # the stationarity test off; along the outer SVD factor of rank `1` in two outer unknowns the
    # step test returns `STALLED` there, and along rank `2`, two of the three entries of the
    # iterate, the rate test ends the solve. R1 on `SVector`s and R3 on `Vector`s.
    opt = Options(T; x_abstol = T(1e-2), g_tol = 0)
    m = Newton(; linsolve = GeometricSolvers.SVDFactorization())
    x₀ = SVector{3, T}(0, 3, 1 // 2)
    for (r, r!, nr, rank, contracts) in ((vp_deficient, vp_deficient!, 3, 1, false),
        (vp_full, vp_full!, 4, 2, true))
        vp1 = GeometricSolvers.VariableProjection(
            GeometricSolvers.LeastSquaresProblem(r, nr), (1,); outer = (2, 3))
        vp3 = GeometricSolvers.VariableProjection(
            GeometricSolvers.LeastSquaresProblem(r!, nr), [1]; outer = [2, 3])
        for (vp, x) in ((vp1, x₀), (vp3, Vector(x₀)))
            _, st = solve(vp, x, m; options = opt)
            @test st.code == (contracts ? SUCCESS : STALLED)
            @test st.rank == rank
            @test st.contracted == contracts
            @test st.iterations == 4
            @test stopping_test(st) == (contracts ? :contraction : :none)
        end
    end
end

# One-unknown least-squares residuals with no root: `ls_square` is `(x² - 2 - c, x - 1)`, and
# `ls_affine` is `(x + a x² - 1 - c, x - 3)`, linear at `a = 0`. Steps along a factor of `J(x₀)`
# contract onto the point where `J(x₀)ᵀ r(x) = 0`, which is no stationary point of `‖r‖²`.
ls_square(x, p) = SVector(x[1]^2 - 2 - p.c, x[1] - 1)
ls_square_jacobian(x, p) = SMatrix{2, 1}(2x[1], one(x[1]))
ls_affine(x, p) = SVector(x[1] + p.a * x[1]^2 - 1 - p.c, x[1] - 3)
ls_affine_jacobian(x, p) = SMatrix{2, 1}(1 + 2p.a * x[1], one(x[1]))
# `‖J(x)ᵀ r(x)‖ / (‖J(x)‖_F ‖r(x)‖)`, the ratio of the stationarity test, in `BigFloat`
function ls_gradratio(r, j, x, p)
    xb = SVector(big(x[1]))
    rx = r(xb, p)
    J = j(xb, p)
    return abs((J' * rx)[1]) / (sqrt(sum(abs2, J)) * sqrt(sum(abs2, rx)))
end

# The status of a least-squares solve whose rate test reads no step along a reused factor: no
# `SUCCESS` by the rate test, and a `SUCCESS` only by the stationarity test, at a point that
# passes it (`2 g_tol` for the round-off of the solve's own measure).
function test_no_reused_contraction(st, r, j, x, p, opt)
    @test stopping_test(st) != :contraction
    @test !st.contracted
    @test st.code != SUCCESS || st.stationary
    @test st.code != SUCCESS || ls_gradratio(r, j, x, p) <= 2 * opt.g_tol
end

@testset "the rate test reads no step along a reused factor of a least-squares problem, $T" for T in REAL_ELTYPES
    # `r(x) = (x² - 5, x - 1)` from `3`: without the guard, the steps along a reused QR factor
    # meet the rate test where `‖r‖²` is not stationary, near `2.1753` for `reuse = 10` in
    # `Float64` (the stationary point solves `2x³ - 9x - 1 = 0`, `x* ≈ 2.1748`)
    p = (c = T(3),)
    opt = Options(T)
    prob = GeometricSolvers.LeastSquaresProblem(ls_square, ls_square_jacobian)
    for k in (3, 10)
        m = Newton(; linsolve = GeometricSolvers.QRFactorization(),
            jacobian = ls_square_jacobian, reuse = k)
        x, st = solve(prob, SVector{1, T}(3), m, p; options = opt)
        @test x[1] != 3
        @test st.reused
        test_no_reused_contraction(st, ls_square, ls_square_jacobian, x, p, opt)
    end
end

@testset "the rate test reads a step along a fresh factor of a least-squares problem, $T" for T in REAL_ELTYPES
    # `F(x) = 0.99 x` with the Jacobian `1`, from `1`: under `reuse = 2` the third step, of norm
    # `9.9e-5`, steps along a fresh factor and meets the rate test before the residual test
    lin(x, p) = SVector(T(99) / 100 * x[1])
    lin_j(x, p) = SMatrix{1, 1}(one(x[1]))
    m = Newton(; linsolve = GeometricSolvers.QRFactorization(), jacobian = lin_j, reuse = 2)
    _, st = solve(GeometricSolvers.LeastSquaresProblem(lin, lin_j), SVector{1, T}(1), m;
        options = Options(T; f_reltol = 0, x_abstol = T(1e-4)))
    @test st.code == SUCCESS
    @test stopping_test(st) == :contraction
    @test !st.reused
end

@testset "the rate test of a Reparametrized problem reads the steps that it reads for the inner problem, $T" for T in REAL_ELTYPES
    # `ls_square` from `3` under the identity chart: without the guard, the steps along a reused
    # QR factor meet the rate test where `‖r‖²` is not stationary
    p = (c = T(3),)
    opt = Options(T)
    chart(y, p) = y
    chart_jacobian(y, p) = SMatrix{1, 1}(one(y[1]))
    prob = GeometricSolvers.Reparametrized(
        GeometricSolvers.LeastSquaresProblem(ls_square, ls_square_jacobian), chart, chart_jacobian)
    for k in (3, 10)
        m = Newton(; linsolve = GeometricSolvers.QRFactorization(), reuse = k)
        x, st = solve(prob, SVector{1, T}(3), m, p; options = opt)
        @test x[1] != 3
        @test st.reused
        @test stopping_test(st) != :contraction
        @test !st.contracted
        @test st.code != SUCCESS || st.stationary
    end
    # A square inner problem keeps the rate test on a reused factor: `F(x) = 0.99 x` with the
    # Jacobian `1`, from `1`, ends on the rate test at its third step, along a reused factor
    lin(x, p) = SVector(T(99) / 100 * x[1])
    lin_j(x, p) = SMatrix{1, 1}(one(x[1]))
    square = GeometricSolvers.Reparametrized(NonlinearProblem(lin, lin_j), chart, chart_jacobian)
    _, st = solve(square, SVector{1, T}(1), Newton(; reuse = 3);
        options = Options(T; f_reltol = 0, x_abstol = T(1e-4)))
    @test st.code == SUCCESS
    @test stopping_test(st) == :contraction
    @test st.reused
end

@testset "the rate test reads no step along a kept factor of a least-squares problem, $T" for T in REAL_ELTYPES
    # The pair of `ls_affine`: the first solve, linear, ends on the stationarity test and keeps
    # its factor; the second starts along it. The pair of `ls_square`, from `3`, at `c = 0` and
    # then `c = 3`.
    opt = Options(T)
    pairs = (
        (ls_affine, ls_affine_jacobian, (a = zero(T), c = zero(T)),
            (a = T(1 // 20), c = T(10)), SVector{1, T}(0), (10,)),
        (ls_square, ls_square_jacobian, (c = zero(T),), (c = T(3),), SVector{1, T}(3),
            (3, 10)))
    for (r, j, p₁, p₂, x₀, ks) in pairs, k in ks,
        linsolve in (GeometricSolvers.QRFactorization(), SVDFactorization())
        m = Newton(; linsolve, jacobian = j, reuse = k, keep_factor = true)
        solver = init(GeometricSolvers.LeastSquaresProblem(r, j), x₀, m, p₁; options = opt)
        x₁, st₁ = solve!(solver, x₀, p₁)
        test_no_reused_contraction(st₁, r, j, x₁, p₁, opt)
        x₂, st₂ = solve!(solver, copy(x₁), p₂)
        @test x₂ != x₁
        test_no_reused_contraction(st₂, r, j, x₂, p₂, opt)
    end
end

@testset "the rate test reads no step along a reused outer factor of a VariableProjection, $T" for T in REAL_ELTYPES
    # `vp_full` from `(0, 3, 1/2)`: without the guard, the steps along a reused outer factor meet
    # the rate test in `Float64` at a point that is not stationary. R1 on `SVector`s and R3 on
    # `Vector`s.
    opt = Options(T)
    x₀ = SVector{3, T}(0, 3, 1 // 2)
    vp1 = GeometricSolvers.VariableProjection(
        GeometricSolvers.LeastSquaresProblem(vp_full, 4), (1,); outer = (2, 3))
    vp3 = GeometricSolvers.VariableProjection(
        GeometricSolvers.LeastSquaresProblem(vp_full!, 4), [1]; outer = [2, 3])
    for (vp, x) in ((vp1, x₀), (vp3, Vector(x₀))), k in (3, 10)

        _, st = solve(vp, x, Newton(; linsolve = SVDFactorization(), reuse = k); options = opt)
        @test st.reused
        @test stopping_test(st) != :contraction
        @test !st.contracted
        @test st.code != SUCCESS || st.stationary
    end
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
    # from `x₀ = 2` under `x_abstol = g_tol = 10⁻³` the stationarity test fails before the fourth
    # step and holds after it, where the rate test holds too: the stationarity test ends the solve
    _, st_both = solve(prob, SVector{1, T}(2), GeometricSolvers.GaussNewton();
        options = Options(T; x_abstol = T(1e-3), g_tol = T(1e-3)))
    @test st_both.code == SUCCESS
    @test st_both.iterations == 4
    @test st_both.stationary
    @test !st_both.contracted
    @test stopping_test(st_both) == :stationarity
end

# A least-squares method whose step divides `x` by `100` along a factor of rank `rank` whose
# diagnostics read the step's start stationary; their re-measure at the end of the step reports
# full column rank and the ratio `1`, which is not stationary.
struct RankStep
    rank::Int32
end
struct RankState{T}
    rank::Int32
    gradratio::T
end
GeometricSolvers.init_state(::RankStep, prob, x, p) = RankState(Int32(-1), eltype(x)(NaN))
GeometricSolvers.state_rank(s::RankState) = s.rank
GeometricSolvers.state_jnorm(s::RankState) = one(s.gradratio)
GeometricSolvers.state_gradratio(s::RankState) = s.gradratio
function GeometricSolvers.refresh_diagnostics!!(s::RankState, ::RankStep, prob, x, Fx, p)
    return RankState(Int32(length(x)), one(s.gradratio))
end
rank_residual(x, p) = SVector(x[1], one(eltype(x)) / 1000)
function GeometricSolvers.step!!(s, m::RankStep, prob, x, Fx, p)
    T = eltype(x)
    x₊ = x / 100
    F₊ = rank_residual(x₊, p)
    info = StepInfo{T}(SUCCESS, sqrt(sum(abs2, F₊)), sqrt(sum(abs2, x₊ - x)), Int32(0),
        false, false, false)
    return x₊, F₊, RankState(m.rank, zero(T)), info
end

@testset "the rate test reads the rank of the step's factor, not of its re-measure, $T" for T in REAL_ELTYPES
    # each step's factor reads its start stationary, so the solve re-measures the diagnostics at
    # the end of the step, where they report full rank and fail the stationarity test; the step
    # lowers `‖F‖`, so the solve goes on. Under `x_abstol = 1/200` the rate test holds at the third
    # step: along a step's factor of full rank it ends the solve there, and along one of rank `0`
    # the step test `‖Δ₃‖ ≤ x_abstol` returns `STALLED`
    prob = GeometricSolvers.LeastSquaresProblem(rank_residual, 2)
    opt = Options(T; f_abstol = 0, f_reltol = 0, x_abstol = T(1) / 200, max_iterations = 5)
    _, st_full = solve(prob, SVector{1, T}(1), RankStep(Int32(1)); options = opt)
    @test st_full.code == SUCCESS
    @test st_full.contracted
    @test st_full.iterations == 3
    x, st = solve(prob, SVector{1, T}(1), RankStep(Int32(0)); options = opt)
    @test st.code == STALLED
    @test !st.contracted
    @test !st.stationary
    @test st.iterations == 3
    @test x[1] ≈ T(1e-6)
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
    # `θ / (1 - θ) ‖Δ‖ = -3 ‖Δ‖ ≤ κ x_reltol ‖x‖` would hold at the third step, the first with two
    # rates
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
    # `best = true` returns the start, of residual `1/2`, with the rate of the last step, which
    # under `reuse = 3` stepped along the factor of an earlier step
    x_best, st_best = solve(prob, one(T),
        Newton(; jacobian = (x, p) -> one(T), best = true, reuse = 3);
        options = Options(T; max_iterations = 4))
    @test st_best.code == MAXITERS
    @test x_best == 1
    @test st_best.fnorm == T(1) / 2
    @test st_best.rate == T(3) / 2
    @test st_best.reused
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
    # under `x_abstol = 1` the rate test holds at the third step of each method; the step test
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
    # the same steps, neither limited nor stalled, end on the rate test at the third step; under
    # `x_abstol = 1/200` the steps of norm `0.99` and `0.0099` pass no step test, and at the second
    # step the bound `θ / (1 - θ) ‖Δ₂‖ ≈ 10⁻⁴ ≤ κ x_abstol = 5 · 10⁻⁴` of its own rate alone holds
    _, st_plain = solve(prob, one(T), ScriptedStep(SUCCESS, false);
        options = Options(T; f_reltol = 0, x_abstol = T(1) / 200, max_iterations = 5))
    @test st_plain.code == SUCCESS
    @test st_plain.contracted
    @test st_plain.iterations == 3
end

# A method whose step divides `x` by `100`, as `ScriptedStep` does, and reports `SUCCESS` and a
# step that a line search shortened.
struct ShortenedStep end
GeometricSolvers.init_state(::ShortenedStep, prob, x, p) = nothing
function GeometricSolvers.step!!(state, ::ShortenedStep, prob, x, Fx, p)
    x₊ = x / 100
    F₊ = prob.F(x₊, p)
    T = typeof(x)
    return x₊, F₊, state,
    StepInfo{T}(SUCCESS, abs(F₊), abs(x₊ - x), Int32(0), false, false, true)
end

@testset "the rate test skips a step that a line search shortened, and the step test reads it, $T" for T in REAL_ELTYPES
    # the steps of `ScriptedStep(SUCCESS, false)`, which end on the rate test at the third step
    # under `x_abstol = 1/200`: shortened, they pass no rate test, and at the third step, of norm
    # `9.9e-5`, the step test `‖Δ₃‖ ≤ x_abstol` returns `STALLED`
    prob = linear_problem(one(T))
    opt = Options(T; f_reltol = 0, x_abstol = T(1) / 200, max_iterations = 5)
    x, st = solve(prob, one(T), ShortenedStep(); options = opt)
    @test st.code == STALLED
    @test !st.contracted
    @test stopping_test(st) == :none
    @test st.iterations == 3
    @test x == one(T) / 100 / 100 / 100
end

@testset "a composition carries the shortened flag of its member to the rate test, $T" for T in REAL_ELTYPES
    # each composition of the steps above ends on the rate test at the third step, and with
    # shortened member steps the step test returns `STALLED` there; a `Multiplicative` of two
    # members divides by `10⁴` per step, so its steps of norm `1e-4` and `1e-8` meet the step test
    # at the third step under `x_abstol = 10⁻⁶`
    prob = linear_problem(one(T))
    half = FixedWeights((1 // 2, 1 // 2))
    compositions = (s -> Repeat(s, 1), s -> Additive((s, s), half), s -> Anderson(s),
        s -> Switch(s, s; threshold = 1e3), s -> Multiplicative(s, s))
    for (i, compose) in enumerate(compositions)
        x_abstol = i == length(compositions) ? T(1e-6) : T(1) / 200
        opt = Options(T; f_reltol = 0, x_abstol, max_iterations = 5)
        _, st_full = solve(prob, one(T), compose(ScriptedStep(SUCCESS, false)); options = opt)
        @test st_full.code == SUCCESS
        @test stopping_test(st_full) == :contraction
        @test st_full.iterations == 3
        _, st = solve(prob, one(T), compose(ShortenedStep()); options = opt)
        @test st.code == STALLED
        @test !st.contracted
        @test st.iterations == 3
    end
end

# A method whose steps from `x₀ = 1` go to `1/2`, `1/4`, `-1/4` and back to `1/4`: the step norms
# `1/2`, `1/4`, `1/2` and `1/2` have the rates `1/2`, `2` and `1`, all exact. Each step reports
# `SUCCESS` and the flag `limited`.
struct CycleStep
    limited::Bool
end
GeometricSolvers.init_state(::CycleStep, prob, x, p) = nothing
function GeometricSolvers.step!!(state, m::CycleStep, prob, x, Fx, p)
    T = typeof(x)
    x₊ = x == 1 ? T(1) / 2 : x == T(1) / 2 ? T(1) / 4 : x == T(1) / 4 ? -T(1) / 4 : T(1) / 4
    F₊ = prob.F(x₊, p)
    return x₊, F₊, state,
    StepInfo{T}(SUCCESS, abs(F₊), abs(x₊ - x), Int32(0), false, m.limited)
end

@testset "the round-off stop skips a limited step, $T" for T in REAL_ELTYPES
    # the third step has the rate `2` after a step of rate `1/2`: the round-off stop ends the solve
    # there, unless a radius limits the steps
    prob = linear_problem(one(T))
    opt = Options(T; f_reltol = 0, roundoff = true, max_iterations = 5)
    x, st = solve(prob, one(T), CycleStep(false); options = opt)
    @test st.code == STALLED
    @test st.iterations == 3
    @test st.rate == 2
    @test x == -T(1) / 4
    _, st_limited = solve(prob, one(T), CycleStep(true); options = opt)
    @test st_limited.code == MAXITERS
    @test st_limited.iterations == 5
    @test !st_limited.contracted
end

# A method whose steps from `x₀ = 0` go to `1`, `2` and `2 + 1/1024`, and then by `1` each: the step
# norms `1`, `1`, `1/1024` and `1` have the rates `1`, `1/1024` and `1024`, all exact.
struct JumpStep end
GeometricSolvers.init_state(::JumpStep, prob, x, p) = nothing
function GeometricSolvers.step!!(state, m::JumpStep, prob, x, Fx, p)
    T = typeof(x)
    x₊ = x == 0 ? one(T) : x == 1 ? T(2) : x == 2 ? T(2) + T(1) / 1024 : x + 1
    F₊ = prob.F(x₊, p)
    return x₊, F₊, state, StepInfo{T}(SUCCESS, abs(F₊), abs(x₊ - x), Int32(0), false, false)
end

@testset "the rate test reads the larger of the last two rates, $T" for T in REAL_ELTYPES
    # one small rate after a step of rate `1`: the third step of `JumpStep` has the rate `1/1024`,
    # and `θ / (1 - θ) ‖Δ₃‖ ≈ 10⁻⁶ ≤ κ x_abstol = 10⁻³`, but the larger rate `1` of the last two
    # fails `θ < 1`, and the step test `‖Δ₃‖ ≤ x_abstol` returns `STALLED`
    prob = NonlinearProblem((x, p) -> x - 100)
    opt = Options(T; f_reltol = 0, x_abstol = T(1e-2))
    x, st = solve(prob, zero(T), JumpStep(); options = opt)
    @test st.code == STALLED
    @test !st.contracted
    @test st.iterations == 3
    @test st.rate == T(1) / 1024
    @test x == T(2) + T(1) / 1024
    # a rate `θ₃ = 2 ≥ 1` after `θ₂ = 1/2`, at the step norm `1/2` where the bound of the smaller
    # rate holds with equality: `(1/2) / (1/2) · 1/2 ≤ κ x_abstol = 1/2`. The larger rate fails
    # `θ < 1`, and the step test returns `STALLED`; `min_iterations = 3` keeps both tests off the
    # first two steps
    _, st_cycle = solve(linear_problem(one(T)), one(T), CycleStep(false);
        options = Options(T; f_reltol = 0, x_abstol = 1, kappa = T(1) / 2,
            min_iterations = 3))
    @test st_cycle.code == STALLED
    @test !st_cycle.contracted
    @test st_cycle.iterations == 3
    @test st_cycle.rate == 2
    # the first step with a rate is the second, and the larger of its rate and the `NaN` of the
    # first step is no rate: at the second step of the rate-`1/100` contraction, of norm `0.0099`,
    # the bound `θ / (1 - θ) ‖Δ₂‖ ≈ 10⁻⁴ ≤ κ x_abstol = 1/20` of its own rate holds, and the step
    # test `‖Δ₂‖ ≤ x_abstol = 1/2` returns `STALLED`
    _, st_lin = solve(linear_problem(T(99) / 100), one(T), linear_newton(one(T));
        options = Options(T; f_reltol = 0, x_abstol = T(1) / 2))
    @test st_lin.code == STALLED
    @test !st_lin.contracted
    @test st_lin.iterations == 2
end

@testset "one small rate of a full step after a shortened step does not end a solve on the rate test" begin
    # the Powell badly scaled function from `u0` under a matrix-free `Newton` with `GMRES` and
    # `Backtracking`: at iteration 77 a full step of norm about `3e-9`, about `1` away from the
    # root, has the rate `8e-8` after a step that the line search shortened. The rate test reads
    # the full step, the larger rate `0.98` of the last two fails the bound, and the solve reaches
    # the root by the residual test
    sp = STANDARD_PROBLEMS[3]
    m = Newton(; linsolve = GeometricSolvers.GMRES(),
        globalization = GeometricSolvers.Backtracking())
    _, st = solve(sp.prob, sp.u0(Float64), m; max_iterations = 200)
    @test st.code == SUCCESS
    @test !st.contracted
    @test stopping_test(st) == :residual
    @test st.iterations > 77
end

@testset "steps that a line search shortens at a minimum of |F| do not end a solve on the rate test" begin
    # `F(x) = x³ - 2x + 2` has its one real root at `x ≈ -1.7693`, and `|F|` has a local minimum
    # of `0.9113` at `x = √(2/3)`, where `F′ = 0`. From these starts `Backtracking` shortens the
    # Newton steps near that minimum, and their small rates, which the line search and not a
    # contraction sets, do not pass the rate test: the step test `‖Δ‖ ≤ x_abstol` returns
    # `STALLED` at the minimum
    prob = NonlinearProblem((x, p) -> x^3 - 2x + 2)
    m = Newton(; globalization = GeometricSolvers.Backtracking())
    for x_abstol in (1e-4, 1e-3), x₀ in (0.6, 0.8, 1.0)

        x, st = solve(prob, x₀, m; options = Options(Float64; x_abstol))
        @test st.code == STALLED
        @test !st.contracted
        @test stopping_test(st) == :none
        @test abs(x - sqrt(2 / 3)) <= x_abstol
        @test st.fnorm > 0.9
    end
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
# ratio of two such perturbations. The second step, of rate at most about `δ / 2` after a first
# step of norm about `2`, is the one step of small rate, and the rate test, which needs two rates,
# does not read it alone; at the later steps it stays off for `δ = 64 √eps(T)`, as the stop at
# `k_stop` below shows.
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
    # the Jacobian `1/2` maps `x` to `-x`: every step after the first has the rate `1`, after a
    # step of rate `1` too, and the stop needs a rate below `1` before it
    _, st_one = solve(linear_problem(one(T)), one(T), linear_newton(T(1) / 2);
        options = Options(T; roundoff = true, max_iterations = 6))
    @test st_one.code == MAXITERS
    @test st_one.iterations == 6
    @test st_one.rate == 1
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
    # the same refresh under `reuse = 3` without `keep_factor`, whose factor would step twice
    calls[] = 0
    x_3, st_3 = solve(linear_problem(one(T)), T(2), Newton(; jacobian = J, reuse = 3))
    @test st_3.code == SUCCESS
    @test st_3.iterations == 3
    @test x_3 == 0
    @test calls[] == 2
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
    # a rate of exactly `1` refreshes too: the Jacobian `1/2` maps `x` to `-x`, and the second step,
    # along the factor of the first, has the rate `1`
    calls[] = 0
    J_half = CountingJacobian((x, p) -> T(1) / 2, calls)
    _, st_one = solve(
        linear_problem(one(T)), one(T), Newton(; jacobian = J_half, reuse = 10);
        options = Options(T; max_iterations = 3))
    @test st_one.code == MAXITERS
    @test st_one.rate == 1
    @test calls[] == 2
end

# A preconditioner that the rule state of its `GMRES` names and that no step applies here.
struct UnappliedPreconditioner end

@testset "reused names the factor of the last step, $T" for T in REAL_ELTYPES
    # the contraction of rate `1/4` along the constant Jacobian `1`: under `reuse = 3` the second
    # step steps along the factor of the first, and under `reuse = 1` each step factorises
    prob = linear_problem(T(3) / 4)
    J = (x, p) -> one(T)
    _, st_3 = solve(prob, one(T), Newton(; jacobian = J, reuse = 3);
        options = capped_options(T, 2))
    @test st_3.iterations == 2
    @test st_3.reused
    _, st_1 = solve(prob, one(T), Newton(; jacobian = J, reuse = 1);
        options = capped_options(T, 2))
    @test st_1.iterations == 2
    @test !st_1.reused
end

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

@testset "refactorize = true changes nothing for a method without a kept factor, $T" for T in REAL_ELTYPES
    # `Picard`, and a `Newton` without `reuse`: the solve with the keyword returns what the same
    # solve without it returns
    q₀ = SVector{2, T}(1, 0)
    p = (; q₀, h = T(1) / 20)
    prob = NonlinearProblem(midpoint_residual)
    for m in (GeometricSolvers.Picard(), Newton(; jacobian = midpoint_jacobian))
        solver_plain = init(prob, q₀, m, p)
        solver_refresh = init(prob, q₀, m, p)
        x_plain, st_plain = solve!(solver_plain, q₀ .+ 1, p)
        x_refresh, st_refresh = solve!(solver_refresh, q₀ .+ 1, p; refactorize = true)
        @test st_plain.code == SUCCESS
        @test st_plain.iterations > 1
        @test x_refresh == x_plain
        @test st_refresh === st_plain
    end
end

@testset "the factor seam of solve! is no method of refactorize!!" begin
    # `refactorize!!(solver, A)` of a `LinearSolver` takes a matrix; any other argument is a
    # `MethodError`, on a static and on an array solver
    A = [2.0 1.0; 1.0 3.0]
    for solver in (init(LinearProblem(SMatrix{2, 2}(A)), LUFactorization()),
        init(LinearProblem(A), LUFactorization()))
        @test solver isa LinearSolver
        @test_throws r"^MethodError: no method matching refactorize!!\(" refactorize!!(solver, 3)
    end
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
    # The Jacobian records the points it is read at, so that the factorisation at the first step of
    # each solve, at `x₀`, shows.
    points = SVector{2, Float64}[]
    A = SMatrix{2, 2}(1.0, 1.0, 1.0, 1.0 + 1e-9)
    F(x, p) = A * x - p .+ 1e-3 .* x .^ 3
    J(x, p) = (push!(points, x); A + SMatrix{2, 2}(3e-3 * x[1]^2, 0.0, 0.0, 3e-3 * x[2]^2))
    m = Newton(;
        jacobian = J, linsolve = LUFactorization(Float32), reuse = 3, keep_factor = true)
    b = SVector(2.0, 2.0 + 1e-9)
    x₀ = SVector(0.0, 0.0)
    solver = init(NonlinearProblem(F), x₀, m, b)
    for _ in 1:3
        empty!(points)
        _, st = solve!(solver, x₀, b)
        @test st.code == SUCCESS
        @test st.promoted
        @test st.iterations == 2
        @test st.rate <= 1e-3
        @test !st.reused
        @test length(points) == 2
        @test count(==(x₀), points) == 1
    end
end

# `F(x) = (g(s) - 2, 2 g(s) - 4 + c)` in `s = x₁ + x₂` with `g(s) = s + a s²`, and its Jacobian
# `g'(s) [1 1; 2 2]` of rank `1`. With `c = 0` the residual has roots; with `c ≠ 0` it has none,
# and the steps along an SVD factor contract onto a least-squares point.
rank_one_g(s, a) = s + a * s^2
function rank_one_residual(x, p)
    g = rank_one_g(x[1] + x[2], p.a)
    return SVector(g - 2, 2g - 4 + p.c)
end
function rank_one_jacobian(x, p)
    d = 1 + 2 * p.a * (x[1] + x[2])
    return SMatrix{2, 2}(d, 2d, d, 2d)
end

@testset "a kept rank-deficient factor leaves the solve to the step test, $T" for T in REAL_ELTYPES
    # The first solve, at `c = 0`, ends on the residual test after one step and keeps its SVD
    # factor of rank `1`. The second, at `c = 1/2` with no root, starts along that factor and
    # contracts onto a least-squares point: the rate test reads the rank of the kept factor and
    # leaves the solve to the step test, as a solve that factorises afresh does.
    opt = Options(T; x_abstol = T(1e-6), g_tol = 0)
    x₀ = SVector{2, T}(1 // 2, 1 // 2)
    p₁ = (a = zero(T), c = zero(T))
    p₂ = (a = T(1 // 100), c = T(1 // 2))
    for keep_factor in (true, false)
        m = Newton(; jacobian = rank_one_jacobian, linsolve = SVDFactorization(),
            reuse = 10, keep_factor)
        solver = init(NonlinearProblem(rank_one_residual), x₀, m, p₁; options = opt)
        _, st₁ = solve!(solver, x₀, p₁)
        @test st₁.code == SUCCESS
        @test st₁.iterations == 1
        @test st₁.rank == 1
        _, st₂ = solve!(solver, x₀, p₂)
        @test st₂.code == STALLED
        @test !st₂.contracted
        @test st₂.rank == 1
        if keep_factor
            # every step of the second solve is along the kept factor, which carries its rank
            # and no other diagnostic of the first solve
            @test isnan(st₂.nullnorm)
            @test isnan(st₂.gradratio)
        end
    end
end

@testset "a solve without keep_factor reports no rank of an earlier solve, $T" for T in REAL_ELTYPES
    # a solve that ends at a root `x₀` with no step reads no factor, after a solve whose last
    # factor has rank `1`
    m = Newton(; jacobian = rank_one_jacobian, linsolve = SVDFactorization())
    opt = Options(T; x_abstol = T(1e-6), g_tol = 0)
    p_none = (a = zero(T), c = T(1 // 2))
    p_root = (a = zero(T), c = zero(T))
    x₀ = SVector{2, T}(1 // 2, 1 // 2)
    solver = init(NonlinearProblem(rank_one_residual), x₀, m, p_none; options = opt)
    _, st₁ = solve!(solver, x₀, p_none)
    @test st₁.code == STALLED
    @test st₁.rank == 1
    _, st₂ = solve!(solver, SVector{2, T}(1, 1), p_root)
    @test st₂.code == SUCCESS
    @test st₂.iterations == 0
    @test st₂.rank == -1
end

struct RateTangentTag end

@testset "a SINGULAR tangent keeps the rate of the solve, $T" for T in REAL_ELTYPES
    # `F(x) = p x` at `p = 15/16` along the Jacobian `1`: a contraction of rate `1/16` that ends on
    # the rate test. The Jacobian `0` of the problem, which the tangent reads at `x*`, is
    # singular, so the solve with a dual `p` returns `SINGULAR`; it keeps the rate, the rate test
    # and the reused factor of the primal solve.
    opt = Options(T; f_reltol = 0, x_abstol = T(1e-4))
    a = T(15) / 16
    pd = ForwardDiff.Dual{RateTangentTag}(a, one(T))
    for (prob, x₀, J) in (
            (
            NonlinearProblem((x, p) -> p * x, (x, p) -> zero(T)), one(T), (x, p) -> one(T)),
            (NonlinearProblem((x, p) -> p * x, (x, p) -> zero(SMatrix{1, 1, T})),
            SVector{1, T}(1), (x, p) -> one(SMatrix{1, 1, T}))),
        reuse in (1, 3)

        m = Newton(; jacobian = J, reuse)
        _, st = solve(prob, x₀, m, a; options = opt)
        @test st.code == SUCCESS
        @test stopping_test(st) == :contraction
        @test st.reused == (reuse == 3)
        _, st_d = solve(prob, x₀, m, pd; options = opt)
        @test st_d.code == SINGULAR
        @test st_d.iterations == st.iterations
        @test st_d.rate == st.rate
        @test st_d.contracted
        @test st_d.reused == st.reused
    end
end

@testset "eliminated inner options keep κ and the round-off flag" begin
    opt = Options(Float64; kappa = 1 / 4, roundoff = true)
    prob = NonlinearProblem((x, p) -> SVector(x[1] - 2, x[2] - 2))
    elim = GeometricSolvers.BlockEliminated(prob, (1,); outer = (2,), inner_options = opt)
    inner_f32 = Adapt.adapt(GeometricSolvers.ToReal{Float32}(), elim).inner_options
    @test inner_f32 isa Options{Float32}
    @test inner_f32.kappa === 0.25f0
    @test inner_f32.roundoff === true
    moments = GeometricSolvers.DepositedMoments(zeros(2), zeros(2), zeros(2), zeros(2))
    batch = GeometricSolvers.BatchEliminated((F, x, m, s) -> F,
        NonlinearProblem((y, c) -> y .- c), [SVector(0.0) for _ in 1:3],
        (m, Y, s, ps) -> m, moments; inner_options = opt)
    inner = GeometricSolvers._batch_inner_options(batch, 1e-6)
    @test inner.kappa === 0.25
    @test inner.roundoff === true
end

@testset "refactorize drops the outer factor of an eliminated problem" begin
    F(x, p) = SVector(x[1]^2 - x[2], x[2]^3 + x[2] - 2 + x[1] / 10)
    F!(y, x, p) = (y .= F(x, p); y)
    for (prob, x₀) in ((NonlinearProblem(F), SVector(0.5, 0.5)),
        (NonlinearProblem(F!), [0.5, 0.5]))
        elim = GeometricSolvers.BlockEliminated(prob, [1]; outer = [2])
        solver = init(elim, copy(x₀), Newton(; reuse = 10, keep_factor = true))
        x, st = solve!(solver)
        @test st.code == SUCCESS
        # a solve of two steps from near the root, with a last rate below `10⁻³`, keeps its factor
        _, st₂ = solve!(solver, x .+ 1e-7)
        @test st₂.code == SUCCESS
        @test st₂.rate <= 1e-3
        @test solver.state.outer_state.rule.age > 0
        dropped = GeometricSolvers.drop_factor!!(solver.state, solver.method)
        @test dropped.outer_state.rule.age == 0
        _, st₃ = solve!(solver, x .+ 1e-7; refactorize = true)
        @test st₃.code == SUCCESS
    end
end

@testset "a step of rate θ ≥ 1 along a reused outer factor refreshes it, $T" for T in REAL_ELTYPES
    # the fixture of "a step of rate θ ≥ 1 along a reused factor refreshes it" as the outer block
    # `z = x₂`, beside the decoupled inner block `x₁ - 1`: the Jacobian of `z` is `2/5` at `z = 2`
    # and `1` elsewhere. The refresh at the third step reads the exact Jacobian and lands at
    # `z = 0`; along the first factor the iterates grow by `3/2` at each step. The Jacobian records
    # the `z` it is read at: the refresh reads it at `z₂ = 9/2`, where the static form also reads
    # it once for its Schur complement, and the dense form does not.
    points = T[]
    jac_z(z) = (push!(points, z); z == 2 ? T(2) / 5 : one(T))
    F(x, p) = SVector(x[1] - 1, x[2])
    J(x, p) = SMatrix{2, 2, T}(1, 0, 0, jac_z(x[2]))
    F!(y, x, p) = (y[1] = x[1] - 1; y[2] = x[2]; y)
    J!(Jm, x, p) = (Jm .= 0; Jm[1, 1] = 1; Jm[2, 2] = jac_z(x[2]); Jm)
    for (F_, J_, x₀, Y, Z, n) in ((F, J, SVector{2, T}(1, 2), (1,), (2,), 2),
        (F!, J!, T[1, 2], [1], [2], 1))
        empty!(points)
        elim = GeometricSolvers.BlockEliminated(NonlinearProblem(F_, J_), Y; outer = Z)
        x, st = solve(elim, x₀, Newton(; reuse = 10);
            options = Options(T; max_iterations = 6))
        @test st.code == SUCCESS
        @test x == [1, 0]
        @test st.iterations == 3
        @test count(==(T(9) / 2), points) == n
    end
end

# The stationary points of `a⁴/4 - a²/2 + c²(1 + a²)/2` are `(0, 0)` and `(±1, 0)`.
stationary_action(x, p) = x[1]^4 / 4 - x[1]^2 / 2 + x[2]^2 * (1 + x[1]^2) / 2

@testset "solve! on a stationarity problem takes refactorize" begin
    prob = GeometricSolvers.StationarityProblem(stationary_action)
    m = Newton(; globalization = GeometricSolvers.Backtracking(), reuse = 10, keep_factor = true)
    for (root, x₀, x_near) in (
        (SVector(0.0, 0.0), SVector(0.1, 0.3), SVector(1.0, 0.0) .+ 1e-9),
        ([0.0, 0.0], [0.1, 0.3], [1.0, 0.0] .+ 1e-9))
        deflated = GeometricSolvers.Deflated(prob, (root,))
        # the solve with no arguments starts at the solver's own iterate
        x, st = solve!(init(deflated, copy(x₀), m))
        @test st isa SolverStatus{Float64}
        @test st.code == SUCCESS
        @test isapprox(abs(x[1]), 1; atol = 1e-8)
        # a solve of one step from close to the root `(1, 0)` keeps its factor; the next solve steps
        # along it, unless it passes `refactorize = true`
        for problem in (prob, deflated), refactorize in (false, true)

            solver = init(problem, copy(x₀), m)
            _, st₁ = solve!(solver, copy(x_near))
            @test st₁.iterations == 1
            _, st₂ = solve!(solver, copy(x_near); refactorize)
            @test st₂.iterations == 1
            @test st₂.reused == !refactorize
        end
    end
end

@testset "a composition reports the rate test of its member, $T" for T in REAL_ELTYPES
    # the rate-`1/100` contraction of the testset "the rate test returns SUCCESS before the
    # residual test holds", through each composition: the status that a composition rebuilds from
    # its member's keeps the rate, `contracted` and `reused`
    prob = linear_problem(T(99) / 100)
    nb = linear_newton(one(T))
    opt = Options(T; f_reltol = 0, x_abstol = T(1e-4))
    _, st_plain = solve(prob, one(T), nb; options = opt)
    @test stopping_test(st_plain) == :contraction
    @test st_plain.iterations == 3
    half = FixedWeights((1 // 2, 1 // 2))
    # `Switch` switches at the first step under the threshold `10³`, and never under `10⁻³⁰`;
    # two equal candidates under fixed weights give the step of the member
    for m in (Repeat(nb, 1), Switch(nb, nb; threshold = 1e3), Switch(nb, nb; threshold = 1e-30),
        Fallback(nb, Picard()), Additive((nb, nb), half))
        _, st = solve(prob, one(T), m; options = opt)
        @test st.code == SUCCESS
        @test st.contracted
        @test stopping_test(st) == :contraction
        @test st.iterations == 3
        @test st.rate == st_plain.rate
    end
    # a `Multiplicative` of two steps of rate `1/10` on `F(x) = 9x/10`, of rate `1/100` together:
    # its `Newton` members report diagnostics, and its `Picard` members report none, which the
    # status of the `Multiplicative` itself then carries
    prob_9 = linear_problem(T(9) / 10)
    for m in (Multiplicative(nb, nb), Multiplicative(Picard(), Picard()))
        _, st = solve(prob_9, one(T), m; options = opt)
        @test st.code == SUCCESS
        @test st.contracted
        @test stopping_test(st) == :contraction
        @test st.iterations == 3
        @test isapprox(st.rate, T(1) / 100; rtol = 100 * eps(T))
    end
    # `Anderson` and `Broyden` end on the rate test too, with rates of their own mixed steps
    for m in (Anderson(nb), Broyden(nb))
        _, st = solve(prob, one(T), m; options = opt)
        @test st.code == SUCCESS
        @test st.contracted
        @test stopping_test(st) == :contraction
        @test 0 <= st.rate < 1
    end
    # the last corrector of a `Continuation` on `F(x) = x + x²` from `x₀ = 1`: its `Newton` with
    # `reuse = 10` steps along one factor from the end of the path and contracts linearly
    cont = GeometricSolvers.Continuation(GeometricSolvers.NewtonHomotopy();
        method = Newton(; reuse = 10), h₀ = 1 / 2, h_max = 1)
    _, st_cont = solve(NonlinearProblem((x, p) -> x + x^2), one(T), cont; options = opt)
    @test st_cont.code == SUCCESS
    @test st_cont.contracted
    @test stopping_test(st_cont) == :contraction
    @test 0 < st_cont.rate < 1
    @test st_cont.reused
    # a `reuse = 3` member steps along the factor of its first step at the third step
    nr = Newton(; jacobian = (x, p) -> one(T), reuse = 3)
    for m in (nr, Repeat(nr, 1), Switch(nr, nr; threshold = 1e3),
        Switch(nr, nr; threshold = 1e-30), Fallback(nr, Picard()))
        _, st = solve(prob, one(T), m; options = opt)
        @test st.code == SUCCESS
        @test st.contracted
        @test st.iterations == 3
        @test st.reused
    end
end

# A residual of rank `1` with no root, as `vp_deficient` above without its eliminated unknown:
# Newton steps along an SVD factor of rank `1` contract onto a least-squares point.
deficient(x, p) = SVector((x[1] + x[2])^2 - 2, (x[1] + x[2]) - eltype(x)(7) / 5)

@testset "a composition reads the rank of its member's factor in the rate test, $T" for T in REAL_ELTYPES
    # under `x_abstol = 10⁻²` and `g_tol = 0` the plain solve stalls at the fourth step on the
    # step test, at `fnorm ≈ 0.0134`, where the rate test would hold along a factor of full rank.
    # The `Switch` with the SVD member first never switches, and the one with it second switches
    # at the first step. In the `Additive`, the LU member reports no rank, and the SVD member
    # reports rank `1`, which keeps the rate test off.
    svdn = Newton(; linsolve = SVDFactorization())
    opt = Options(T; x_abstol = T(1e-2), g_tol = 0)
    x₀ = SVector{2, T}(0, 3)
    _, st_plain = solve(NonlinearProblem(deficient), x₀, svdn; options = opt)
    @test st_plain.code == STALLED
    @test st_plain.rank == 1
    for m in (Repeat(svdn, 1), Switch(svdn, Newton(); threshold = 1e-30),
        Switch(Newton(), svdn; threshold = 1e3), Additive(Newton(), svdn),
        Multiplicative(svdn, svdn))
        _, st = solve(NonlinearProblem(deficient), x₀, m; options = opt)
        @test st.code != SUCCESS
        @test !st.contracted
    end
    # the mixed steps of `Anderson` and `Broyden` reach the bound of the rate test under
    # `x_abstol = 10⁻³`, and the outer `Newton` of a `LeftPreconditioned` with a `Picard` inner,
    # which steps on `rˡ = F`, under `x_abstol = 10⁻²`
    opt_3 = Options(T; x_abstol = T(1e-3), g_tol = 0)
    for m in (Anderson(svdn), Broyden(svdn))
        _, st = solve(NonlinearProblem(deficient), x₀, m; options = opt_3)
        @test st.code != SUCCESS
        @test !st.contracted
    end
    deficient!(y, x, p) = (y .= deficient(x, p); y)
    _, st_lp = solve(NonlinearProblem(deficient!), Vector(x₀),
        GeometricSolvers.LeftPreconditioned(svdn, Picard()); options = opt)
    @test st_lp.code != SUCCESS
    @test !st_lp.contracted
end

@testset "a step of rate θ ≥ 1 along a reused factor refreshes a composition's member, $T" for T in REAL_ELTYPES
    # the fixture of "a step of rate θ ≥ 1 along a reused factor refreshes it": without the
    # refresh the iterates grow by `3/2` at each step. `Switch(qn, Picard())` never switches, and
    # `Switch(Newton(), qn)` switches at the first step. The `Additive` of two equal candidates
    # refreshes each of its members.
    for (wrap, n) in ((m -> Repeat(m, 1), 2), (
        m -> Switch(m, Picard(); threshold = 1e-30), 2),
        (m -> Switch(Newton(), m; threshold = 1e3), 2),
        (m -> Additive((m, m), FixedWeights((1 // 2, 1 // 2))), 4),
        (m -> Multiplicative(m), 2))
        calls = Ref(0)
        J = CountingJacobian((x, p) -> x == 2 ? T(2) / 5 : one(T), calls)
        x, st = solve(linear_problem(one(T)), T(2), wrap(QuasiNewton(10; jacobian = J));
            options = Options(T; max_iterations = 6))
        @test st.code == SUCCESS
        @test x == 0
        @test calls[] == n
    end
    # `Anderson` of depth `1` on `F(x) = x` in two unknowns, with the Jacobian `diag(1/3, 3)` at
    # `x₀ = (2, 1)` and the exact identity elsewhere: a mixed step along the first factor has a
    # rate `θ ≥ 1`, and the refresh of the member reads the identity and lands at the root.
    # Without it the member keeps the first factor.
    calls = Ref(0)
    x₀ = SVector{2, T}(2, 1)
    J = CountingJacobian(
        (x, p) -> x == x₀ ? SMatrix{2, 2, T}(T(1) / 3, 0, 0, 3) : SMatrix{2, 2, T}(I), calls)
    x, st = solve(
        NonlinearProblem((x, p) -> x), x₀, Anderson(QuasiNewton(10; jacobian = J), 1);
        options = Options(T; max_iterations = 10))
    @test st.code == SUCCESS
    @test x == zero(x)
    @test calls[] == 2
    # the outer `QuasiNewton` of a `LeftPreconditioned` with a `Picard` inner, whose residual
    # `rˡ = F` has the slope `2/5` at `x₁ = 2` and about `1` elsewhere: the second step, along the
    # first factor, has the rate `θ ≈ 3/2`, and the refresh brings the iterate to the root. Without
    # it the iterates grow.
    function bump!(y, x, p)
        d = x[1] - 2
        y[1] = x[1] - 3 * d * exp(-4 * d^2) / 5
        y[2] = x[2]
        return y
    end
    x_lp, st_lp = solve(NonlinearProblem(bump!), T[2, 1],
        GeometricSolvers.LeftPreconditioned(QuasiNewton(10), Picard());
        options = Options(T; max_iterations = 10))
    @test st_lp.code == SUCCESS
    @test maximum(abs, x_lp) < 1e-4
end

@testset "refactorize = true drops the factor of a composition's members, $T" for T in REAL_ELTYPES
    # the pair of "refactorize = true after a change of h adds one factorisation" through each
    # composition: one more factorisation for each member with a kept factor
    function run(wrap, refactorize)
        calls = Ref(0)
        m = wrap(Newton(; jacobian = CountingJacobian(midpoint_jacobian, calls), reuse = 5,
            keep_factor = true))
        h = T(1) / 20
        q₀ = SVector{2, T}(1, 0)
        solver = init(NonlinearProblem(midpoint_residual), q₀, m, (; q₀, h))
        for _ in 1:3
            q₀, _ = solve!(solver, q₀, (; q₀, h))
        end
        h = T(1) / 40
        before = calls[]
        _, st = solve!(solver, q₀, (; q₀, h); refactorize)
        return calls[] - before, st
    end
    half = FixedWeights((1 // 2, 1 // 2))
    for (wrap, members) in ((m -> Repeat(m, 1), 1), (m -> Switch(m, m; threshold = 1e3), 1),
        (m -> Switch(m, m; threshold = 1e-30), 1), (m -> Fallback(m, Picard()), 1),
        (m -> Anderson(m), 1), (m -> Broyden(m), 1),
        (m -> Additive((m, m), half), 2), (m -> Multiplicative(m, m), 2))
        n_plain, st_plain = run(wrap, false)
        n_refresh, st_refresh = run(wrap, true)
        @test st_plain.code == SUCCESS
        @test st_refresh.code == SUCCESS
        @test n_refresh == n_plain + members
    end
end

@testset "refactorize = true drops the factor of a Continuation" begin
    # the last corrector of a `Continuation` keeps the factor of its member: at the third step
    # of the midpoint rule the solve with `refactorize = true` factorises once more. In `Float32`
    # the solves of this sequence end `STALLED`, which keeps no factor.
    function run(refactorize)
        calls = Ref(0)
        m = GeometricSolvers.Continuation(GeometricSolvers.NewtonHomotopy();
            method = Newton(; jacobian = CountingJacobian(midpoint_jacobian, calls),
                reuse = 5, keep_factor = true))
        h = 1 / 20
        q₀ = SVector(1.0, 0.0)
        solver = init(NonlinearProblem(midpoint_residual), q₀, m, (; q₀, h))
        for _ in 1:2
            q₀, _ = solve!(solver, q₀, (; q₀, h))
        end
        @test solver.state.member.rule.age > 0
        before = calls[]
        _, st = solve!(solver, q₀, (; q₀, h); refactorize)
        return calls[] - before, st
    end
    n_plain, st_plain = run(false)
    n_refresh, st_refresh = run(true)
    @test st_plain.code == SUCCESS
    @test st_refresh.code == SUCCESS
    @test n_refresh == n_plain + 1
end

midpoint!(y, x, p) = (y .= midpoint_residual(SVector{2}(x), p); y)

@testset "refactorize = true drops the factor of a LeftPreconditioned, $T" for T in REAL_ELTYPES
    # the outer `Newton` of a `LeftPreconditioned`, whose Jacobian is a difference of the
    # preconditioned residual, keeps its factor across solves; the seam of `refactorize` drops it
    m = GeometricSolvers.LeftPreconditioned(
        Newton(; reuse = 5, keep_factor = true), Picard())
    q₀ = SVector{2, T}(1, 0)
    p = (; q₀, h = T(1) / 20)
    solver = init(NonlinearProblem(midpoint!), Vector(q₀), m, p)
    _, st = solve!(solver, Vector(q₀), p)
    @test st.code == SUCCESS
    @test solver.state.outer_state.rule.age > 0
    dropped = GeometricSolvers.drop_factor!!(solver.state, solver.method)
    @test dropped.outer_state.rule.age == 0
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
