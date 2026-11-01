# The passes over the residual that a line-search step makes, and the trips of a search. The
# merit `φ(0) = ‖F(x)‖²` at the anchor of each step comes from the state of `Newton` and `Picard`:
# `start!!` reduces `F(x₀)` once, and each step carries the merit that its search computed for the
# residual it returns. A search loops while no result is found, for at most `maxiter` trips.

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, StrongWolfe, LineSearch, Options, ToReal,
                        init_state, solve!!, start!!, step!!, residual!!, linesearch, norm2,
                        rnorm
using LinearAlgebra: Diagonal
using StaticArrays: SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/linefunctions.jl")

const GS = GeometricSolvers
const adapt = GS.Adapt.adapt

# `F(x) = x² − (4, 9)` from `(1, 1)`, with its analytic Jacobian: the first full Newton step, to
# `(2.5, 5)`, raises the merit and is cut back. `F(x) = x − cos(x)/2` is a contraction for Picard.
newton_F!(y, x, p) = (y .= x .^ 2 .- [4, 9]; y)
newton_J!(J, x, p) = (J .= Diagonal(2 .* x); J)
picard_F!(y, x, p) = (y .= x .- cos.(x) ./ 2; y)
newton_F(x, p) = x .^ 2 .- SVector(4, 9)
picard_F(x, p) = x .- cos.(x) ./ 2

const SEARCHES = (Backtracking(), Bisection(), StrongWolfe())

# The solves of the count, each with the number of steps it takes.
const COUNTED = [
    ("Newton Backtracking", Newton(; globalization = Backtracking()), :newton, 5),
    ("Newton Bisection", Newton(; globalization = Bisection()), :newton, 4),
    ("Newton StrongWolfe", Newton(; globalization = StrongWolfe()), :newton, 4),
    ("Picard Backtracking",
        Picard(; damping = 1, globalization = Backtracking()), :picard, 12)
]

function array_problem(kind)
    kind === :newton ? NonlinearProblem(newton_F!, newton_J!) :
    NonlinearProblem(picard_F!)
end
function value_problem(kind)
    kind === :newton ? NonlinearProblem(newton_F) : NonlinearProblem(picard_F)
end

# The calls of `norm2` whose argument is the residual buffer of a solve's state, counted by
# identity: `rnorm` calls `norm2` too, on the residual and on the step. The counting method falls
# through to the package's own, and is deleted after the count, so that no other test sees it.
const FX = Ref{Any}(nothing)
const NFX = Ref(0)
function count_merits(method, prob, x₀)
    @eval function GS.norm2(a::Vector{Float64})
        a === $FX[] && ($NFX[] += 1)
        return invoke(GS.norm2, Tuple{Any}, a)
    end
    try
        state = init_state(method, prob, x₀, nothing)
        FX[] = state.Fx
        NFX[] = 0
        _, status, _ = Base.invokelatest(solve!!, state, prob, copy(x₀), method, nothing,
            Options(Float64))
        return NFX[], status
    finally
        Base.delete_method(which(GS.norm2, (Vector{Float64},)))
    end
end

@testset "a line-search solve reduces the residual buffer at most twice" begin
    @testset "$name" for (name, m, kind, steps) in COUNTED
        n, status = count_merits(m, array_problem(kind), [1.0, 1.0])
        @test status.code == SUCCESS
        @test status.step_failures == 0
        @test status.iterations == steps
        @test n ≤ 2
    end
    # the counting method is gone again
    @test which(norm2, (Vector{Float64},)).sig == Tuple{typeof(norm2), Any}

    @testset "with reuse, and from a converged iterate" begin
        prob = array_problem(:newton)
        n, status = count_merits(
            Newton(; reuse = 3, globalization = Backtracking()), prob, [1.0, 1.0])
        @test status.code == SUCCESS
        @test n ≤ 2
        n, status = count_merits(Newton(; globalization = Backtracking()), prob, [2.0, 3.0])
        @test status.code == SUCCESS
        @test status.iterations == 0
        @test n ≤ 2
    end
end

# The merit of the state after `start!!` and after each step is `norm2` of the residual that the
# step returns, within 8 ulps, on an array and on a value iterate.
function merits_follow(m, prob, x₀)
    R = real(eltype(x₀))
    rtol = 8 * eps(R)
    state = init_state(m, prob, x₀, nothing)
    x = copy(x₀)
    Fx = residual!!(GS.residual_buffer(state), prob, x, nothing)
    state = start!!(state, m, prob, x, Fx, nothing, rnorm(Fx))
    ok = state.merit isa R && isapprox(state.merit, norm2(Fx); rtol)
    moved = false
    for _ in 1:6
        x_old = copy(x)
        x, Fx, state, _ = step!!(state, m, prob, x, Fx, nothing)
        moved |= x != x_old
        ok &= state.merit isa R && isapprox(state.merit, norm2(Fx); rtol)
    end
    return ok && moved
end

@testset "the merit of the state follows the residual, $R" for R in REAL_ELTYPES
    for (name, m, kind, _) in COUNTED, reuse in (nothing, 2)

        reuse !== nothing && m isa Picard && continue
        mr = reuse === nothing ? m : Newton(; reuse, globalization = m.globalization)
        mR = adapt(ToReal{R}(), mr)
        @test merits_follow(mR, array_problem(kind), R[1, 1])
        @test merits_follow(mR, value_problem(kind), SVector{2, R}(1, 1))
    end
end

# `init_state` sets the merit of a line search to `NaN`, and only `start!!` sets it to
# `norm2(F(x₀))`: a step on a state that no `start!!` saw meets a non-finite anchor, returns
# `NONFINITE` and keeps its iterate. After `start!!`, the same step succeeds and moves.
const NO_START = [
    ("Newton Backtracking", Newton(; globalization = Backtracking())),
    ("Newton Bisection", Newton(; globalization = Bisection())),
    ("Newton StrongWolfe", Newton(; globalization = StrongWolfe())),
    ("Newton reuse = 3 Backtracking", Newton(; reuse = 3, globalization = Backtracking())),
    ("Picard Backtracking", Picard(; damping = 1, globalization = Backtracking()))
]

function first_step(m, prob, x₀; start)
    state = init_state(m, prob, x₀, nothing)
    x = copy(x₀)
    Fx = residual!!(GS.residual_buffer(state), prob, x, nothing)
    start && (state = start!!(state, m, prob, x, Fx, nothing, rnorm(Fx)))
    x₁, _, state₊, info = step!!(state, m, prob, x, Fx, nothing)
    return x₁, state₊, info
end

@testset "a step without start!! is NONFINITE, $R" for R in REAL_ELTYPES
    @testset "$name" for (name, m) in NO_START
        mR = adapt(ToReal{R}(), m)
        for (prob, x₀) in ((value_problem(:newton), SVector{2, R}(1, 1)),
            (array_problem(:newton), R[1, 1]))
            x₁, st₊, info = first_step(mR, prob, x₀; start = false)
            @test info.code == NONFINITE
            @test x₁ == x₀
            @test isnan(st₊.merit)
            x₁, st₊, info = first_step(mR, prob, x₀; start = true)
            @test info.code == SUCCESS
            @test x₁ != x₀
            @test isfinite(st₊.merit)
        end
    end
end

# `finish!!` sets the merit of a line search to `NaN` again: a step on the state that a solve
# returned, from an iterate of its own, fails closed as on a fresh state. The solve that stops at
# once at `(100, 100)` would otherwise leave an anchor far above the merit 73 at `(1, 1)`, and the
# converged solve one of zero. The returned state still starts a solve of its own.
@testset "a step on the state that a solve returned is NONFINITE, $R" for R in REAL_ELTYPES
    @testset "$name" for (name, m) in NO_START
        mR = adapt(ToReal{R}(), m)
        for (prob, x₀) in ((value_problem(:newton), SVector{2, R}(1, 1)),
            (array_problem(:newton), R[1, 1]))
            for (xs, opt) in ((100 .* x₀, Options(R; max_iterations = 0)),
                (x₀ .* [2, 3], Options(R)))
                state = init_state(mR, prob, x₀, nothing)
                xs = x₀ isa SVector ? SVector{2, R}(xs) : collect(R, xs)
                _, status, state = solve!!(state, prob, xs, mR, nothing, opt)
                @test status.code == (opt.max_iterations == 0 ? MAXITERS : SUCCESS)
                @test isnan(state.merit)
                x = copy(x₀)
                Fx = residual!!(GS.residual_buffer(state), prob, x, nothing)
                x₁, _, st₊, info = step!!(state, mR, prob, x, Fx, nothing)
                @test info.code == NONFINITE
                @test x₁ == x₀
                @test isnan(st₊.merit)
                _, status, _ = solve!!(state, prob, copy(x₀), mR, nothing, Options(R))
                @test status.code == SUCCESS
            end
        end
    end
    @testset "no merit to reset" begin
        prob = value_problem(:newton)
        x = SVector(1.0, 1.0)
        for m in (Newton(), Newton(; globalization = DogLeg()), Picard())
            _, _, state = solve!!(init_state(m, prob, x, nothing), prob, x, m)
            @test state.merit === nothing
        end
    end
end

@testset "only a line search carries a merit" begin
    prob = value_problem(:newton)
    x = SVector(1.0, 1.0)
    Fx = newton_F(x, nothing)
    for m in (Newton(), Newton(; globalization = DogLeg()), Picard())
        state = init_state(m, prob, x, nothing)
        @test state.merit === nothing
        @test start!!(state, m, prob, x, Fx, nothing, rnorm(Fx)).merit === nothing
    end
    @test init_state(Newton(; globalization = Backtracking()), prob, x, nothing).merit isa
          Float64
    @test isbits(init_state(Newton(; globalization = Backtracking()), prob, x, nothing))
    @test isbits(init_state(Picard(; globalization = Backtracking()), prob, x, nothing))
end

# A search that fails and still decreases the merit takes its step: the merit it returns is that
# of the residual it returns. From `(1, 1)` along `d = (0.5, 1)` the merit falls from 73 to 28.0625,
# short of the decrease of 100 that a slope of `-10⁶` demands at `α = 1`, and one trip ends it.
@testset "a failed search that takes its step returns the merit of its residual, $R" for R in REAL_ELTYPES
    prob = value_problem(:newton)
    x = SVector{2, R}(1, 1)
    Fx = newton_F(x, nothing)
    d = SVector{2, R}(0.5, 1)
    ls = adapt(ToReal{R}(), Backtracking(; maxiter = 1))
    x₊, F₊, _, info, merit = GS.globalize_merit!!(
        nothing, ls, prob, x, Fx, d, nothing, nothing, R(-1e6), norm2(Fx))
    @test info.code == LINESEARCH_FAILED
    @test info.failures == 1
    @test x₊ == x + d
    @test merit isa R
    @test isapprox(merit, norm2(F₊); rtol = 8 * eps(R))
    @test merit < norm2(Fx)
end

# A merit that rises along the line, `φ(α) = 1 + α`, with a slope `-1` that the caller passes: no
# trial is accepted, and no stop other than the trip bound applies within three trips, so each
# search spends exactly the evaluations of three trips: one `φ` per trip for `Backtracking` and
# `StrongWolfe`, one `φ′` per trip and the `φ` at the result for `Bisection`.
const TRIP_EVALUATIONS = Dict(Backtracking => 3, Bisection => 4, StrongWolfe => 3)

@testset "a search spends at most maxiter trips, $R" for R in REAL_ELTYPES
    for ls in (Backtracking(; maxiter = 3), Bisection(; maxiter = 3), StrongWolfe(;
        maxiter = 3))
        lsR = adapt(ToReal{R}(), ls)
        w = Watched(Line(α -> 1 + α, α -> one(α)), R)
        res = linesearch(lsR, w, -one(R), one(R), one(R))
        @test res.code in (LINESEARCH_FAILED, STALLED)
        @test res.evaluations == TRIP_EVALUATIONS[typeof(ls).name.wrapper]
        @test evaluated(w) == TRIP_EVALUATIONS[typeof(ls).name.wrapper]
    end
end
