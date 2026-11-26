# `Continuation`: the pseudo-arclength follower on the Newton, the fixed-point and a parameter
# homotopy; the standard row that every other column fails in `Float64`; the two-fold cubic; a
# system with no path to `λ = 1`; a non-finite start; the record of the steps and its step
# control, a failed tangent among them, and the record of a `StationarityProblem` solve; the
# residual and the Jacobian of each homotopy; the refusals; a linear system scaled to the ends of
# the float range; a start near the root; and the scale of the fixed-point homotopy where the
# Jacobian at the start is zero or not finite.

using ForwardDiff: ForwardDiff
using JLArrays: JLArray
using LinearAlgebra: I, norm
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Continuation, FixedPointHomotopy, LeastSquaresProblem,
                        NewtonHomotopy,
                        NONFINITE, NonlinearProblem, ParameterHomotopy, STALLED,
                        StationarityProblem, SUCCESS,
                        MAXITERS, bordered_solve!, continuation_steps, next_step_length,
                        path_corrector!, homotopy_jacobian!, homotopy_residual!,
                        homotopy_scale, init,
                        solve, solve!

include("../helpers/standard_cells.jl")

# The real root of `x³ − 3x − 5/2`: one root, since the local maximum `f(−1) = −1/2` is negative,
# by the hyperbolic form of Cardano's formula, `2 cosh(acosh(5/4) / 3)`.
cubic_root(::Type{T}) where {T} = T(2 * cosh(acosh(big(5) / 4) / 3))

cubic(x::Number, p) = x^3 - 3x - eltype(x)(5) / 2
cubic!(y, x, p) = (y[1] = x[1]^3 - 3x[1] - eltype(x)(5) / 2; nothing)
square(x::Number, p) = x^2 + 1
square!(y, x, p) = (y[1] = x[1]^2 + 1; nothing)

# The number of turns of `λ` along the accepted points of a record: one at each fold.
function folds(steps)
    λ = steps.λ[steps.accepted]
    s = sign.(diff(λ))
    return count(i -> s[i] != s[i + 1], 1:(length(s) - 1))
end

# Whether each rejected step of a record holds the `λ` of its prediction, which differs from the
# `λ` of the accepted point it starts from, the start at `λ = 0` before the first accepted step.
function rejected_steps_predict(steps)
    λ_from = zero(eltype(steps.λ))
    for k in eachindex(steps.λ)
        if steps.accepted[k]
            λ_from = steps.λ[k]
        elseif steps.λ[k] == λ_from
            return false
        end
    end
    return true
end

@testset "the standard row p3 at 100u0" begin
    sp = STANDARD_PROBLEMS[3]
    # every column of the reference fails the row in Float64
    reference = read_table(STANDARD_REFERENCE)
    row = filter(
        c -> (c.problem, c.start, c.eltype) == (3, "100u0", "Float64") &&
             !startswith(c.column, "Continuation"),
        reference)
    @test !isempty(row)
    @test all(c -> c.code != "SUCCESS", row)
    @testset "$T" for T in REAL_ELTYPES
        x₀ = 100 .* sp.u0(T)
        solver = init(sp.prob, x₀, Continuation(FixedPointHomotopy());
            options = Options(T; max_iterations = 200))
        x, status = solve!(solver)
        @test status.code == SUCCESS
        # the stated root has seven significant digits; `F` is symmetric under `x₁ ↔ x₂`, so the
        # mirror of the stated root is a root too, and the `Float32` path ends there
        @test isapprox(x, sp.root(T); rtol = max(sqrt(eps(T)), 1e-6)) ||
              isapprox(x, reverse(sp.root(T)); rtol = max(sqrt(eps(T)), 1e-6))
        T == Float64 && @test isapprox(x, sp.root(T); rtol = 1e-6)
        @test x != x₀
        steps = continuation_steps(solver)
        n = length(steps.λ)
        @test n == length(steps.h) == length(steps.accepted)
        # each step is accepted or rejected, and the rejected steps are the step failures
        @test count(.!steps.accepted) == status.step_failures
        @test count(steps.accepted) > 0
        @test all(>(0), steps.h)
        # the last step is accepted past `λ = 1`, and no accepted step before it reaches 1
        @test steps.accepted[end] && steps.λ[end] >= 1
        @test all(<(1), steps.λ[steps.accepted][1:(end - 1)])
        # a rejected step halves the step length; a step after an accepted one keeps it or
        # doubles it, and some step doubles it
        for k in 1:(n - 1)
            if steps.accepted[k]
                @test steps.h[k + 1] == steps.h[k] || steps.h[k + 1] == 2 * steps.h[k]
            else
                @test steps.h[k + 1] == steps.h[k] / 2
            end
        end
        @test any(k -> steps.accepted[k] && steps.h[k + 1] == 2 * steps.h[k], 1:(n - 1))
        # the iterations are the path steps and those of the last corrector, which takes one at
        # least from the chord's point
        @test n < status.iterations <= n + 200
        # a second solve on the solver records its own steps only
        λ₁, h₁ = copy(steps.λ), copy(steps.h)
        solve!(solver, copy(x₀))
        @test continuation_steps(solver).λ == λ₁ && continuation_steps(solver).h == h₁
        # `solve` returns what `solve!(init(...))` returns
        x′, status′ = solve(sp.prob, x₀, Continuation(FixedPointHomotopy());
            max_iterations = 200)
        @test x′ == x && status′.code == status.code &&
              status′.iterations == status.iterations
    end
end

@testset "the two-fold cubic through both folds" begin
    @testset "$T $(nameof(typeof(h)))" for T in REAL_ELTYPES,
        h in (FixedPointHomotopy(), NewtonHomotopy())

        method = Continuation(h)
        # the last corrector stops at `|f(x)| ≤ 64 eps`, at most `7 eps` in `x`, since
        # `f′ = 3x² − 3 ≈ 9.65` at the root; the default relative test would stop it earlier
        options = Options(T; f_abstol = 64 * eps(T), f_reltol = zero(T))
        # a value iterate
        solver = init(NonlinearProblem(cubic), T(-2.2), method; options)
        x, status = solve!(solver)
        @test status.code == SUCCESS
        @test x isa T
        @test isapprox(x, cubic_root(T); rtol = 8 * eps(T))
        steps = continuation_steps(solver)
        @test folds(steps) == 2
        # the fixed-point path rejects a step past its second fold, whose record holds the `λ` of
        # the prediction
        h isa FixedPointHomotopy &&
            @test count(.!steps.accepted) == status.step_failures > 0
        @test rejected_steps_predict(steps)
        # an array iterate
        solver = init(NonlinearProblem(cubic!), T[-2.2], method; options)
        x, status = solve!(solver)
        @test status.code == SUCCESS
        @test isapprox(only(x), cubic_root(T); rtol = 8 * eps(T))
        @test folds(continuation_steps(solver)) == 2
        # a static iterate
        x, status = solve(NonlinearProblem((x, p) -> SVector(cubic(x[1], p))),
            SVector{1, T}(-2.2), method; options)
        @test status.code == SUCCESS && x isa SVector{1, T}
        @test isapprox(only(x), cubic_root(T); rtol = 8 * eps(T))
    end
end

@testset "x² + 1 has no path to λ = 1" begin
    @testset "$T" for T in REAL_ELTYPES
        method = Continuation(NewtonHomotopy())
        for (prob, x₀) in ((NonlinearProblem(square), T(0.5)),
            (NonlinearProblem(square!), T[0.5]))
            solver = init(prob, x₀, method)
            x, status = solve!(solver)
            @test status.code == STALLED
            steps = continuation_steps(solver)
            # the solve ends on a rejected step whose halved length is below the minimum
            @test !steps.accepted[end]
            @test steps.h[end] / 2 < T(1e-6) <= steps.h[end]
            # the step norm of the status is the length of that last step
            @test status.stepnorm == steps.h[end]
            @test all(<(1), steps.λ)
            @test status.step_failures == count(.!steps.accepted)
            @test rejected_steps_predict(steps)
            @test all(isfinite, x)
        end
        # the fixed-point homotopy leaves to `|x| → ∞` with `λ → 0`, and does not succeed either
        _, status = solve(NonlinearProblem(square), T(0.5), Continuation(FixedPointHomotopy()))
        @test status.code != SUCCESS
        # step lengths of powers of two: a step of exactly `h_min` is tried, and `h_max` caps
        # every step
        method = Continuation(NewtonHomotopy(); h₀ = 2.0^-4, h_min = 2.0^-12, h_max = 2.0^-3)
        solver = init(NonlinearProblem(square), T(0.5), method)
        _, status = solve!(solver)
        steps = continuation_steps(solver)
        @test status.code == STALLED
        @test minimum(steps.h) == T(2)^-12
        @test maximum(steps.h) == T(2)^-3
    end
end

@testset "the step control" begin
    # an accepted step doubles the length after a corrector of at most 3 iterations, up to
    # `h_max`, and keeps it after a slower one
    @test next_step_length(0.25, 1, 1.0) == 0.5
    @test next_step_length(0.25, 3, 1.0) == 0.5
    @test next_step_length(0.25, 4, 1.0) == 0.25
    @test next_step_length(0.75, 2, 1.0) == 1.0
    # `max_iterations` path steps return `MAXITERS` at the last accepted point, here on the
    # fixed-point path of `x² + 1`, which leaves to `|x| → ∞`
    @testset "$T" for T in REAL_ELTYPES
        for N in (0, 15)
            solver = init(
                NonlinearProblem(square), T(0.5), Continuation(FixedPointHomotopy());
                max_iterations = N)
            x, status = solve!(solver)
            steps = continuation_steps(solver)
            @test status.code == MAXITERS
            @test status.iterations == N == length(steps.λ)
            @test status.stepnorm == (N == 0 ? 0 : steps.h[end])
            @test (x == T(0.5)) == (N == 0)
            @test isfinite(x) && isapprox(status.fnorm, x^2 + 1; rtol = 8 * eps(T))
        end
    end
end

@testset "a failed tangent at a corrected point rejects the step" begin
    # On `x − 2`, whose Newton-homotopy path is `x = 2λ`, the Jacobian is called once by the
    # start's corrector, once by the start's tangent, and then once by each step's corrector and
    # once by the tangent at its corrected point. A `NaN` at the fourth call, the tangent of the
    # first step, rejects that step and halves its length; the steps after it are accepted.
    @testset "$T" for T in REAL_ELTYPES
        calls = Ref(0)
        J(x, p) = (calls[] += 1; calls[] == 4 ? T(NaN) : one(T))
        solver = init(NonlinearProblem((x, p) -> x - 2, J), zero(T),
            Continuation(NewtonHomotopy()))
        x, status = solve!(solver)
        steps = continuation_steps(solver)
        @test status.code == SUCCESS && x == 2
        @test !steps.accepted[1] && all(steps.accepted[2:end])
        @test steps.h[2] == steps.h[1] / 2
        @test status.step_failures == 1
    end
end

@testset "the record of a StationarityProblem solve" begin
    # the double well `S = (x₁² − 1)²/4 + x₂²/2`, whose gradient `(x₁³ − x₁, x₂)` has the root
    # `(1, 0)` on the path from `(0.1, 1)`
    S(x, p) = (x[1]^2 - 1)^2 / 4 + x[2]^2 / 2
    @testset "$T $(typeof(x₀))" for T in REAL_ELTYPES,
        x₀ in (T[0.1, 1], SVector{2, T}(0.1, 1))

        solver = init(StationarityProblem(S), x₀, Continuation(FixedPointHomotopy()))
        x, status = solve!(solver)
        @test status.code == SUCCESS
        @test isapprox(x, T[1, 0]; atol = sqrt(eps(T)))
        steps = continuation_steps(solver)
        @test !isempty(steps.λ) && steps.accepted[end] && steps.λ[end] >= 1
    end
end

@testset "the bordered solve" begin
    @testset "$T" for T in REAL_ELTYPES
        r = T[3, 5]
        @test bordered_solve!(T[2 1; 1 3], r) && isapprox(r, T[4, 7] / 5; rtol = 8 * eps(T))
        # an infinite entry, whose LU factor solves `[Inf 1; 1 2] y = (1, 2)` with the finite
        # `y = (0, 1)`, an exactly singular matrix, and a non-finite right-hand side
        @test !bordered_solve!(T[Inf 1; 1 2], T[1, 2])
        @test !bordered_solve!(T[1 1; 1 1], T[1, 2])
        @test !bordered_solve!(T[2 1; 1 3], T[NaN, 2])
    end
    # a `BigFloat` matrix with a zero pivot, on which the generic back substitution throws
    # `SingularException`
    @test !bordered_solve!(BigFloat[1 1; 1 1], BigFloat[1, 2])
end

@testset "the corrector stops at √eps" begin
    # The corrector along `t = e_λ` at `λ = 1` is Newton's method on `x² − 2` from 1.5, whose
    # corrections are 8.3e-2, 2.5e-3, 2.1e-6 and 1.6e-12: the first below
    # `√eps · max(1, ‖z‖) ≈ √eps · 1.7` is the fourth in `Float64` and the third in `Float32`.
    @testset "$T" for T in REAL_ELTYPES
        F(x, q) = [x[1]^2 - 2]
        J(x, q) = reshape([2x[1]], 1, 1)
        ws = (; F₀ = T[0], σ = Ref(one(T)), z = zeros(T, 2),
            w = zeros(T, 2), zp = T[1.5, 1], t = T[0, 1],
            r = zeros(T, 2), A = zeros(T, 2, 2))
        ok, k = path_corrector!(
            ws, NewtonHomotopy(), F, J, zeros(T, 1), nothing, T[1.5], nothing, 1)
        @test ok
        @test k == (T == Float64 ? 4 : 3)
        @test isapprox(ws.w, T[sqrt(T(2)), 1]; rtol = sqrt(eps(T)))
        # on `x²` from 1.5, whose double root halves each correction, the corrector stops
        # unconverged after its sixth iteration, at `1.5 / 2⁶`
        G(x, q) = [x[1]^2]
        JG(x, q) = reshape([2x[1]], 1, 1)
        ws = (; F₀ = T[0], σ = Ref(one(T)), z = zeros(T, 2),
            w = zeros(T, 2), zp = T[1.5, 1], t = T[0, 1],
            r = zeros(T, 2), A = zeros(T, 2, 2))
        ok, k = path_corrector!(
            ws, NewtonHomotopy(), G, JG, zeros(T, 1), nothing, T[1.5], nothing, 1)
        @test !ok && k == 6
        @test ws.w == T[1.5 / 2 ^ 6, 1]
    end
end

@testset "a start that is not on a path" begin
    @testset "$T" for T in REAL_ELTYPES
        # `F(x₀)` is not finite where the fixed-point homotopy's `H(x₀, 0) = 0` is
        x, status = solve(NonlinearProblem((x, p) -> 1 / x), zero(T),
            Continuation(FixedPointHomotopy()))
        @test status.code == NONFINITE && status.iterations == 0 && iszero(x)
        # a parameter homotopy starts from `H(x₀, 0) = F(x₀, p_0)`, also where `F(x₀, p_1)` is
        # not finite: the path `x = 1/(1 − λ)` leaves to infinity, and is followed
        h = ParameterHomotopy((p, λ) -> λ)
        _, status = solve(NonlinearProblem((x, p) -> x - 1 / (1 - p)), one(T), Continuation(h))
        @test status.iterations > 0
        # nor is `F` evaluated at `(x₀, p_1)` before the path: `log(x − q)` is not defined at
        # `x₀ = 1` and `q = p_1 = 2`, and the path `x = 1 + 2λ` reaches the root 3
        h = ParameterHomotopy((p, λ) -> 2λ)
        for (prob, x₀) in ((NonlinearProblem((x, q) -> log(x - q)), one(T)),
            (NonlinearProblem((y, x, q) -> (y[1] = log(x[1] - q); nothing)), T[1]))
            x, status = solve(prob, x₀, Continuation(h))
            @test status.code == SUCCESS && isapprox(only(x), 3; rtol = 8 * eps(T))
        end
        # `x² + 1 = 0` at `λ = 0` has no root to correct the start onto
        h = ParameterHomotopy((p, λ) -> λ)
        solver = init(NonlinearProblem((x, p) -> x^2 + 1 - 2p), T(0.5), Continuation(h))
        x, status = solve!(solver)
        @test status.code == STALLED && status.iterations == 0 && x == T(0.5)
        @test isempty(continuation_steps(solver).λ)
        # at a fold of `λ` at the start, `H_x(x₀, 0)` and the bordered system along `e_λ` are
        # singular: the start takes no step
        for x₀ in (one(T), -one(T))
            solver = init(NonlinearProblem(cubic), x₀, Continuation(NewtonHomotopy()))
            x, status = solve!(solver)
            @test status.code == STALLED && status.iterations == 0 && x == x₀ &&
                  isempty(continuation_steps(solver).λ)
        end
    end
    # the same fold in `BigFloat`, whose generic LU factor of the singular bordered system has a
    # zero pivot
    for x₀ in (big(1.0), big(-1.0))
        solver = init(NonlinearProblem(cubic), x₀, Continuation(NewtonHomotopy()))
        x, status = solve!(solver)
        @test status.code == STALLED && status.iterations == 0 && x == x₀ &&
              isempty(continuation_steps(solver).λ)
    end
end

@testset "a NaN start takes no step" begin
    @testset "$T $(nameof(typeof(h)))" for T in REAL_ELTYPES,
        h in (FixedPointHomotopy(), NewtonHomotopy(), ParameterHomotopy((p, λ) -> λ))

        for (prob, x₀) in ((NonlinearProblem(cubic), T(NaN)),
            (NonlinearProblem(cubic!), T[NaN]))
            solver = init(prob, x₀, Continuation(h))
            x, status = solve!(solver)
            @test status.code == NONFINITE
            @test status.iterations == 0
            @test isnan(only(x))
            @test isempty(continuation_steps(solver).λ)
        end
    end
end

# `x[1]² = p_λ`, `x[2] = x[1]`, whose parameter moves from `p.a` at `λ = 0` to `p.b` at `λ = 1`.
parabola!(y, x, p) = (y[1] = x[1]^2 - p; y[2] = x[2] - x[1]; nothing)

@testset "a parameter homotopy" begin
    @testset "$T" for T in REAL_ELTYPES
        h = ParameterHomotopy((p, λ) -> p.a + λ * (p.b - p.a))
        p = (a = T(1), b = T(4))
        x, status = solve(NonlinearProblem(parabola!), T[1, 1], Continuation(h), p)
        @test status.code == SUCCESS
        @test isapprox(x, T[2, 2]; rtol = 8 * eps(T))
        # a start off the path at `λ = 0` is corrected onto it first
        x, status = solve(NonlinearProblem(parabola!), T[1.1, 0.9], Continuation(h), p)
        @test status.code == SUCCESS
        @test isapprox(x, T[2, 2]; rtol = 8 * eps(T))
    end
end

@testset "the residual and the Jacobian of each homotopy" begin
    # `F(x, p) = (x₁² − p x₂, x₁x₂)`, with `J = [2x₁ −p; x₂ x₁]`, at `x = (1.5, −0.5)` from
    # `x₀ = (0.25, 2)`, at `λ = 0.3` and `p = 5/2`. The residual and the Jacobian are given as
    # callables `(x, p_λ) -> F` and `(x, p_λ) -> J`, as the solve passes them. The fixed-point
    # homotopy divides `F` by the scale at the start,
    # `σ = max|J(x₀)| = max|[1/2 −5/2; 2 1/4]| = 5/2`, which differs from the
    # `max|J(x)| = max|[3 −5/2; −1/2 3/2]| = 3` at the current point and from
    # `‖F(x₀)‖∞ = ‖(−79/16, 1/2)‖∞ = 79/16`.
    @testset "$T" for T in REAL_ELTYPES
        x₀ = T[0.25, 2]
        x = T[1.5, -0.5]
        λ = T(0.3)
        F(x, p) = T[x[1] ^ 2 - p * x[2], x[1] * x[2]]
        J(x, p) = T[2x[1] -p; x[2] x[1]]
        p = T(5) / 2
        F₀ = F(x₀, p)
        σ = T(5) / 2
        @test homotopy_scale(FixedPointHomotopy(), J, x₀, F₀, p) == σ
        # the other homotopies read no scale
        @test homotopy_scale(NewtonHomotopy(), J, x₀, F₀, p) == 1
        tol = 8 * eps(T)
        for (h, Hλ, Hz) in (
            (NewtonHomotopy(), F(x, p) - (1 - λ) * F₀, [J(x, p) F₀]),
            (FixedPointHomotopy(), λ * F(x, p) / σ + (1 - λ) * (x - x₀),
            [λ * J(x, p) / σ + (1 - λ) * I F(x, p) / σ - (x - x₀)]),
            (ParameterHomotopy((p, λ) -> p + λ), F(x, p + λ),
            [J(x, p + λ) T[-x[2], 0]]))
            H = zeros(T, 2)
            homotopy_residual!(H, h, F, x, λ, x₀, F₀, homotopy_scale(h, J, x₀, F₀, p), p)
            @test isapprox(H, Hλ; rtol = tol)
            A = zeros(T, 3, 3)
            homotopy_jacobian!(A, h, F, J, x, λ, x₀, F₀, homotopy_scale(h, J, x₀, F₀, p), p)
            # the parameter column is a central difference of step `cbrt(eps)`
            rtol = h isa ParameterHomotopy ? 10 * cbrt(eps(T))^2 : tol
            @test isapprox(A[1:2, :], Hz; rtol)
        end
    end
end

@testset "the refusals" begin
    @test_throws r"^ArgumentError: Continuation needs" Continuation(FixedPointHomotopy(); method = Picard())
    @test_throws r"^ArgumentError: Continuation needs" Continuation(
        FixedPointHomotopy(); method = Newton(; linsolve = GMRES()))
    for kw in ((; h₀ = 0.0), (; h_min = 0.0), (; h_min = 0.2), (; h₀ = Inf), (; h₀ = NaN),
        (; h_max = 0.05), (; h_min = NaN))
        @test_throws r"^ArgumentError: Continuation requires finite step lengths" Continuation(
            FixedPointHomotopy(); kw...)
    end
    method = Continuation(FixedPointHomotopy())
    # a `ForwardDiff.Dual` input, before the primal solve
    @test_throws r"^ArgumentError: solve does not accept ForwardDiff.Dual inputs for the combinator Continuation" solve(
        NonlinearProblem(cubic!), [ForwardDiff.Dual(-2.2, 1.0)], method)
    # a `LeastSquaresProblem`
    lsq = LeastSquaresProblem((y, x, p) -> (y .= [x[1] - 1, x[1] - 2]; nothing), 2)
    @test_throws r"^ArgumentError: the combinator Continuation does not take a LeastSquaresProblem" init(
        lsq, [0.0], method)
    # a device array, and a complex iterate
    @test_throws r"^ArgumentError: Continuation runs on a host array" init(
        NonlinearProblem(cubic!), JLArray([-2.2]), method)
    @test_throws r"^ArgumentError: Continuation takes a real iterate" init(
        NonlinearProblem(cubic!), [-2.2 + 0im], method)
end

@testset "a linear system scaled to the ends of the float range" begin
    # `F(x) = s (A x − b)`, whose root `A⁻¹ b = (1, −2)` does not depend on `s`. Neither does the
    # path of the Newton homotopy, nor that of the fixed-point homotopy, which divides `F` by
    # `σ = max|J(x₀)| = 3s`. At `floatmax/8` the squares of `F(x₀)` and of the entries of `J(x₀)`
    # overflow, and their largest absolute entries do not.
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, -3.0]
    @testset "$T $(nameof(typeof(h))) s = $s" for T in REAL_ELTYPES,
        h in (NewtonHomotopy(), FixedPointHomotopy()),
        s in (T(1e-6), T(1e6), 4 * sqrt(floatmax(T)), floatmax(T) / 8)
        At, bt = T.(A), T.(b)
        F!(y, x, p) = (y .= s .* (At * x .- bt); nothing)
        x, status = solve(NonlinearProblem(F!), zeros(T, 2), Continuation(h);
            f_abstol = zero(T))
        @test status.code == SUCCESS
        @test isapprox(x, T[1, -2]; rtol = 64 * eps(T))
    end
    # a start at the root, where `F(x₀) = 0`: the path of the fixed-point homotopy stays at `x₀`
    @testset "$T a start at the root" for T in REAL_ELTYPES
        At, bt = T.(A), T.(b)
        F!(y, x, p) = (y .= At * x .- bt; nothing)
        x, status = solve(NonlinearProblem(F!), T[1, -2], Continuation(FixedPointHomotopy()))
        @test status.code == SUCCESS
        @test x == T[1, -2]
    end
end

@testset "a start near the root" begin
    # `F(x) = A x − b` from `(1, −2) + δ d`: the fixed-point path leaves along
    # `dx/dλ = −F(x₀)/σ = −δ A d / 3`, of the size of `δ`, and accepts each step. The member's
    # last corrector may stop at round-off a few ulps from the root with `STALLED`, so the test
    # reads the path and the iterate, not the code.
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, -3.0]
    @testset "$T" for T in REAL_ELTYPES
        At, bt = T.(A), T.(b)
        F!(y, x, p) = (y .= At * x .- bt; nothing)
        δs = T[k * eps(T) for k in (1, 4, 16, 256, 4096)]
        T == Float64 && append!(δs, (1e-8, 1e-7, 1e-6))
        for δ in δs, d in (T[1, 0], T[0, 1], T[1, -1])

            solver = init(NonlinearProblem(F!), T[1, -2] .+ δ .* d,
                Continuation(FixedPointHomotopy()))
            x, _ = solve!(solver)
            steps = continuation_steps(solver)
            @test all(steps.accepted) && steps.λ[end] >= 1
            @test isapprox(x, T[1, -2]; rtol = 8 * eps(T))
        end
    end
end

@testset "the scale of the fixed-point homotopy where J(x₀) = 0" begin
    # `s (x² − 1)` from 0, where `J(x₀) = 0`: `σ = max|F(x₀)| = s`, so the path of `x² − 1`
    # does not depend on `s`
    @testset "$T" for T in REAL_ELTYPES
        records = map((T(1e-6), one(T), T(1e6))) do s
            solver = init(NonlinearProblem((x, p) -> s * (x^2 - 1)), zero(T),
                Continuation(FixedPointHomotopy()))
            x, status = solve!(solver)
            @test status.code == SUCCESS
            # the member stops at its default `f_reltol = √eps` on `|F(x₀)| = s`, at most about
            # `√eps / 2` from the simple root 1 of `x² − 1`
            @test isapprox(abs(x), one(T); rtol = sqrt(eps(T)))
            continuation_steps(solver)
        end
        for r in records
            @test r.accepted == records[2].accepted
            @test isapprox(r.λ, records[2].λ; rtol = 8 * eps(T))
            @test isapprox(r.h, records[2].h; rtol = 8 * eps(T))
        end
        # `x²` from its root 0, where `J(x₀) = 0` and `F(x₀) = 0`: `σ = 1`, and the path stays at 0
        x, status = solve(NonlinearProblem((x, p) -> x^2), zero(T),
            Continuation(FixedPointHomotopy()))
        @test status.code == SUCCESS && iszero(x)
    end
end

@testset "a non-finite Jacobian at the start takes no step" begin
    # `√x − 1` from 0: `F(x₀) = −1` is finite, and `J(x₀) = 1/(2√0) = Inf` makes `σ` infinite
    @testset "$T" for T in REAL_ELTYPES
        solver = init(NonlinearProblem((x, p) -> sqrt(x) - 1), zero(T),
            Continuation(FixedPointHomotopy()))
        x, status = solve!(solver)
        @test status.code == NONFINITE && status.iterations == 0 && iszero(x)
        @test isempty(continuation_steps(solver).λ)
    end
end
