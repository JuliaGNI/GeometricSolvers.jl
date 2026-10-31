# The loop hooks `start!!` and `finish!!` of `solve!!`, through `Wrapped`, a test-only method that
# runs a member method and records what the hooks receive.

using Adapt: Adapt
using LinearAlgebra: Diagonal
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: MAXITERS, NONFINITE, Newton, NonlinearProblem, Options, SUCCESS,
                        SolverStatus, ToReal, init, rnorm, solve, solve!, solve!!

const GS = GeometricSolvers

include("../helpers/matrix.jl")

# The hook calls of the last solve, in order.
const HOOK_LOG = Symbol[]

# A method that runs its member `m`. With `restore = true`, its `finish!!` returns the member's
# first iterate in place of the last one, with the residual norm of the first step.
struct Wrapped{M}
    m::M
    restore::Bool
end

Wrapped(m) = Wrapped(m, false)

Adapt.adapt_structure(to, w::Wrapped) = Wrapped(Adapt.adapt(to, w.m), w.restore)

struct WrappedState{S, X, R}
    inner::S
    first::X
    firstfnorm::R
    steps::Int32
    fnorm₀::R
end

function GS.init_state(w::Wrapped, prob, x, p)
    R = real(eltype(x))
    return WrappedState(GS.init_state(w.m, prob, x, p), copy(x), R(NaN), Int32(0), R(NaN))
end

GS.residual_buffer(s::WrappedState) = GS.residual_buffer(s.inner)

function GS.start!!(s::WrappedState, w::Wrapped, prob, x, Fx, p, fnorm₀)
    push!(HOOK_LOG, :start)
    inner = GS.start!!(s.inner, w.m, prob, x, Fx, p, fnorm₀)
    return WrappedState(inner, s.first, s.firstfnorm, Int32(0), fnorm₀)
end

function GS.step!!(s::WrappedState, w::Wrapped, prob, x, Fx, p)
    x₊, F₊, inner, info = GS.step!!(s.inner, w.m, prob, x, Fx, p)
    first, firstfnorm = iszero(s.steps) ? (GS.copy!!(s.first, x₊), info.fnorm) :
                        (s.first, s.firstfnorm)
    return x₊, F₊, WrappedState(inner, first, firstfnorm, s.steps + Int32(1), s.fnorm₀),
    info
end

function GS.finish!!(
        s::WrappedState, w::Wrapped, prob, x, Fx, status::SolverStatus{R}) where {R}
    push!(HOOK_LOG, :finish)
    w.restore || return x, status, s
    x = GS.copy!!(x, s.first)
    st = SolverStatus{R}(status.code, status.iterations, s.firstfnorm, status.stepnorm,
        status.step_failures, status.promoted)
    return x, st, s
end

# `x² - p`, in the out-of-place form for a value iterate and in place for an array iterate; the
# array form's `J` counts its calls in `count`.
square_problem() = NonlinearProblem((x, p) -> x .* x .- p)
function counted_square_problem(count)
    F!(y, x, p) = (y .= x .* x .- p; y)
    J!(J, x, p) = (count[] += 1; J .= Diagonal(2 .* x); J)
    return NonlinearProblem(F!, J!)
end

@testset "fnorm₀ reaches start!!, which runs once, $T" for T in ELTYPES
    prob = square_problem()
    x₀ = SVector{2, T}(30, 50)
    p = SVector{2, T}(4, 9)
    m = Adapt.adapt(ToReal{real(T)}(), Wrapped(Newton()))
    empty!(HOOK_LOG)
    _, status, s = solve!!(GS.init_state(m, prob, x₀, p), prob, x₀, m, p, Options(T))
    @test status.code == SUCCESS
    @test s.fnorm₀ == rnorm(prob.F(x₀, p))
    @test count(==(:start), HOOK_LOG) == 1
    @test HOOK_LOG[1] === :start
    @test HOOK_LOG[end] === :finish
end

@testset "a member Newton starts fresh in each solve, $T" for T in ELTYPES
    count = Ref(0)
    prob = counted_square_problem(count)
    x = T[30, 50]
    p = T[4, 9]
    # one step per solve, so the first solve ends with a factor of age 1 out of 3
    solver = init(prob, x, Wrapped(Newton(; reuse = 3)), p; max_iterations = 1)
    @test solve!(x, solver, p).code == MAXITERS
    @test count[] == 1
    @test solve!(x, solver, p).code == MAXITERS
    @test count[] == 2
end

@testset "finish!! replaces the returned iterate, $T" for T in ELTYPES
    # a value iterate through `solve`
    prob = square_problem()
    x₀ = SVector{2, T}(30, 50)
    p = SVector{2, T}(4, 9)
    m = Wrapped(Newton(), true)
    x₁, _ = solve(prob, x₀, Newton(), p; max_iterations = 1)
    x₃, _ = solve(prob, x₀, Newton(), p; max_iterations = 3)
    @test x₁ != x₃
    empty!(HOOK_LOG)
    x, status = solve(prob, x₀, m, p; max_iterations = 3)
    @test HOOK_LOG == [:start, :finish]
    @test status.code == MAXITERS
    @test status.iterations == 3
    @test x == x₁
    @test status.fnorm == rnorm(prob.F(x, p))

    # an array iterate through `solve!`: the caller's `x` holds the first iterate
    count = Ref(0)
    prob! = counted_square_problem(count)
    xa = T[30, 50]
    pa = T[4, 9]
    solver = init(prob!, xa, m, pa; max_iterations = 3)
    status! = solve!(xa, solver, pa)
    @test status!.code == MAXITERS
    @test xa ≈ collect(x₁) rtol = 8 * eps(real(T))
    Fa = similar(xa)
    @test status!.fnorm == rnorm(prob!.F(Fa, xa, pa))
end

@testset "each early return calls start!! and then finish!! once, $T" for T in ELTYPES
    prob = square_problem()
    x₀ = SVector{2, T}(30, 50)
    p = SVector{2, T}(4, 9)
    m = Wrapped(Newton())
    fnorm₀ = rnorm(prob.F(x₀, p))
    for (pᵢ, kwargs, code) in (
        (SVector{2, T}(Inf, 9), (;), NONFINITE),
        (p, (; f_abstol = 2 * fnorm₀), SUCCESS),
        (p, (; max_iterations = 0), MAXITERS))
        empty!(HOOK_LOG)
        x, status = solve(prob, x₀, m, pᵢ; kwargs...)
        @test status.code == code
        @test status.iterations == 0
        @test x === x₀
        @test HOOK_LOG == [:start, :finish]
    end
end
