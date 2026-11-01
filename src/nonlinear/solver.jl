"""
    NoRefinement()

The default iterative-refinement setting of [`Newton`](@ref), which returns the linear solve's
step unchanged.
"""
struct NoRefinement end

@inline refine!!(d_slot, ::NoRefinement, prob, x, Fx, d, fac, p, jvpprep) = d

"""
    init_state(method, prob::NonlinearProblem, x, p)

Build the state of `method` for [`NonlinearProblem`](@ref) `prob` at the initial iterate `x` and
parameters `p`: its workspace, its AD preparation and its rule state, in one concrete type per
method. In R1 (`SVector`, `MVector` or `Number`), `init_state` is allocation-free and returns an
`isbits` state when `prob` and `method` are `isbits`.
"""
function init_state end

"""
    residual_buffer(state)

The slot that [`solve!!`](@ref) evaluates `F(x₀)` into: the residual workspace of `state` for an
array iterate, and `nothing` for a value iterate. The default is `nothing`, which lets
[`residual!!`](@ref) return a new value.
"""
@inline residual_buffer(state) = nothing

"""
    step!!(state, method, prob::NonlinearProblem, x, Fx, p)

Compute one nonlinear solver step from `(x, Fx)` and return `(x₊, F₊, state₊, info)` where `info`
is a [`StepInfo`](@ref).
"""
function step!! end

"""
    start!!(state, method, prob::NonlinearProblem, x, Fx, p, fnorm₀) -> state

Prepare `state` for a solve from `x`, with residual `Fx` and its norm `fnorm₀`, and return the
state that the first step reads. [`solve!!`](@ref) calls it once, before its stopping test at 0
iterations, so a state reused across solves starts each solve afresh. The default returns
`state`; [`Newton`](@ref) resets its rule state.
"""
@inline start!!(state, method, prob, x, Fx, p, fnorm₀) = state

"""
    finish!!(state, method, prob::NonlinearProblem, x, Fx, status) -> (x, status, state)

Close a solve that ends at `x`, with residual `Fx` and `status`, and return the triple that
[`solve!!`](@ref) returns. [`solve!!`](@ref) calls it at every return. The default returns
`(x, status, state)`. A method may return another iterate; it then sets `status.fnorm` itself,
and for an array iterate it writes that iterate into `x` and returns `x`.
"""
@inline finish!!(state, method, prob, x, Fx, status) = (x, status, state)

@inline function ok_status(status::SolverStatus{R}) where {R}
    return SolverStatus{R}(
        SUCCESS, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted)
end

@inline function end_status(status::SolverStatus{R}, code::ReturnCode) where {R}
    return SolverStatus{R}(
        code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted)
end

"""
    solve!!(state, prob::NonlinearProblem, x₀, method, p = nothing, options = Options(eltype(x₀)))

Run the nonlinear solver loop from `x₀` using `state = init_state(method, prob, x₀, p)` and return
`(x, status, state)`. `solve!!` is the in-place and kernel entry point: inside a
`KernelAbstractions.@kernel`, a caller prepares `state₀ = init_state(m, prob, x₀, p)` on the host
and calls `solve!!(state₀, prob, x_i, m, p_i, opt)` in each work-item.

At the start of each call, `solve!!` evaluates `Fx = residual!!(residual_buffer(state), prob, x₀,
p)` and `fnorm₀ = rnorm(Fx)`, and calls [`start!!`](@ref) once, which resets the per-solve state
of the method (the reuse age and the stalled flag under factor reuse, and the radius under
[`DogLeg`](@ref)). Every return goes through [`finish!!`](@ref), and `solve!!` returns the triple
that `finish!!` returns.

When `fnorm₀` is not finite — an entry of `Fx` is not finite, or `‖F‖` is above `floatmax` — it
returns `NONFINITE` with `0` iterations. When the residual test of [`Options`](@ref) already holds
at `0` iterations, it returns `SUCCESS`. When `options.max_iterations == 0`, it returns `MAXITERS`
with `0` iterations. `options` must be an `Options{real(eltype(x₀))}`; any other raises an
`ArgumentError`.

Inside the loop, each trip calls [`step!!`](@ref) and updates `status` via [`record`](@ref).
`SINGULAR` and `NONFINITE` return immediately. A `LINESEARCH_FAILED` step whose `stepnorm` is zero
(meaning the failed line-search trial did not decrease the merit) stops immediately with
`LINESEARCH_FAILED`; one that decreased the merit keeps its new iterate, increments
`status.step_failures`, and continues. Only the residual test returns `SUCCESS`. When it fails
after a step that is neither `STALLED` nor `limited`, a step test that holds returns `STALLED`
at once, and so does a `SUCCESS` step of zero norm, which leaves the iterate where the next step
would start; `min_iterations` bounds both. After a `STALLED` step only the residual test applies,
and `options.max_stalls` consecutive `STALLED` steps return `STALLED`. After a step whose
`limited` is set only the residual test applies too, because a radius, not the distance to a
root, sets its norm; such a step is not a stall. Reaching `options.max_iterations` without
convergence returns `MAXITERS`.
"""
function solve!!(
        state, prob::NonlinearProblem, x₀, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    check_options(eltype(x₀), options)
    x = x₀
    Fx = residual!!(residual_buffer(state), prob, x, p)
    fnorm₀ = rnorm(Fx)
    state = start!!(state, method, prob, x, Fx, p, fnorm₀)
    status = SolverStatus{R}(SUCCESS, Int32(0), fnorm₀, zero(R), Int32(0), false)
    if !isfinite(fnorm₀)
        return finish!!(state, method, prob, x, Fx, end_status(status, NONFINITE))
    end
    if residual_converged(options, status, fnorm₀)
        return finish!!(state, method, prob, x, Fx, status)
    end
    if options.max_iterations == Int32(0)
        return finish!!(state, method, prob, x, Fx, end_status(status, MAXITERS))
    end
    stalls = Int32(0)
    while true
        x, Fx, state, info = step!!(state, method, prob, x, Fx, p)
        status = record(status, info)
        if info.code == SINGULAR || info.code == NONFINITE
            return finish!!(state, method, prob, x, Fx, status)
        end
        if info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
            return finish!!(state, method, prob, x, Fx, status)
        end
        if residual_converged(options, status, fnorm₀)
            return finish!!(state, method, prob, x, Fx, ok_status(status))
        end
        if !(info.code == STALLED || info.limited)
            # the residual test fails: the step test, or a zero step, ends the solve unconverged
            if step_converged(options, status, rnorm(x))
                return finish!!(state, method, prob, x, Fx, end_status(status, STALLED))
            end
            if info.code == SUCCESS && iszero(info.stepnorm) &&
               status.iterations >= options.min_iterations
                return finish!!(state, method, prob, x, Fx, end_status(status, STALLED))
            end
        end
        if info.code == STALLED
            stalls += one(Int32)
            if stalls >= options.max_stalls
                return finish!!(state, method, prob, x, Fx, end_status(status, STALLED))
            end
        else
            stalls = Int32(0)
        end
        if status.iterations >= options.max_iterations
            return finish!!(state, method, prob, x, Fx, end_status(status, MAXITERS))
        end
    end
end

# The options of a solve whose iterate has element type `T`: the keywords, or an `Options{real(T)}`,
# never both.
@inline function resolve_options(
        ::Type{T}, options::Union{Nothing, Options}, kwargs) where {T}
    options === nothing && return Options(T; kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("pass either `options` or keyword arguments, not both"))
    check_options(T, options)
    return options
end

"""
    NonlinearSolver(prob, x, method, options, state, p)

A preallocated nonlinear solver for [`NonlinearProblem`](@ref) `prob` with `method` and
[`Options`](@ref) `options`, holding the workspace and AD preparation in `state`, the iterate `x`
and the parameters `p`. It keeps the element type `T` and the length `n` of `x`, which
[`solve!`](@ref) checks, and the types `X` of `x` and `Q` of `p`, which its fields `x::X` and
`p::Q` keep. Build it with [`init(prob, x, method, p; options = nothing, kwargs...)`](@ref init)
and solve in place with [`solve!(solver)`](@ref solve!) or
[`solve!(solver, x, p)`](@ref solve!).
"""
mutable struct NonlinearSolver{T, P <: NonlinearProblem, M, O <: Options, S,
    X <: AbstractVector{T}, Q}
    const prob::P
    const method::M
    const options::O
    const n::Int
    state::S
    x::X
    p::Q
    function NonlinearSolver(prob::P, x::X, method::M, options::O, state::S,
            p::Q) where {T, P <: NonlinearProblem, M, O <: Options, S,
            X <: AbstractVector{T}, Q}
        new{T, P, M, O, S, X, Q}(prob, method, options, length(x), state, x, p)
    end
end

"""
    init(prob::NonlinearProblem, x::AbstractVector, method, p = nothing;
         options = nothing, kwargs...)

Allocate and prepare a [`NonlinearSolver`](@ref) for `prob` and `method` at the iterate `x` and
the parameters `p`, without mutating `x`. The solver holds `x` and `p` themselves, not copies:
[`solve!(solver)`](@ref solve!) writes the solution into the caller's `x`, and a change of `x`
between two calls is the start of the next solve. Passing both `options` and keyword tolerances
raises an `ArgumentError`, and so does an `options` that is not an `Options{real(eltype(x))}`. A
`StaticArray` `x` raises an `ArgumentError` too, because [`solve!`](@ref) cannot write the
solution into it; solve it with [`solve`](@ref).
"""
function init(
        prob::NonlinearProblem, x::AbstractVector, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    x isa StaticArray && throw(ArgumentError(
        "`init` builds a solver for an array iterate; solve a `StaticArray` iterate with `solve`"))
    R = real(eltype(x))
    opt = resolve_options(eltype(x), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    state = init_state(m, prob, x, p)
    return NonlinearSolver(prob, x, m, opt, state, p)
end

"""
    solve!(solver::NonlinearSolver)
    solve!(solver::NonlinearSolver, x::AbstractVector, p = solver.p)

Solve `solver.prob` in place, using the preallocated workspace and AD preparation in `solver`, and
return `status::SolverStatus`. `solve!(solver)` solves from `solver.x`, the iterate given to
[`init`](@ref), with `solver.p`, and writes the solution into `solver.x`.
`solve!(solver, x, p)` stores `x` in `solver.x` and `p` in `solver.p`, then solves from `x` with
`p` and writes the solution into `x`; without `p`, it solves with the parameters that the solver
holds. An `x` whose element type or length differs from the iterate given to `init` raises an
`ArgumentError`, and so do an `x` of another array type and a `p` of another type than those
given to `init`. A refused call changes neither `solver` nor `x`.
"""
solve!(solver::NonlinearSolver) = solve!(solver, solver.x, solver.p)

function solve!(solver::NonlinearSolver{T, P, M, O, S, X, Q}, x::AbstractVector,
        p = solver.p) where {T, P, M, O, S, X, Q}
    eltype(x) === T && length(x) == solver.n || throw(ArgumentError(
        "`solve!` needs an iterate like the template given to `init`, of length " *
        "$(solver.n) and element type $T; got length $(length(x)) and " *
        "element type $(eltype(x))"))
    x isa X || throw(ArgumentError(
        "`solve!` needs an iterate of type $X, like the one given to `init`; " *
        "got type $(typeof(x))"))
    p isa Q || throw(ArgumentError(
        "`solve!` needs parameters of type $Q, like those given to `init`; " *
        "got type $(typeof(p))"))
    solver.x = x
    solver.p = p
    _, status, state = solve!!(
        solver.state, solver.prob, x, solver.method, p, solver.options)
    solver.state === state || (solver.state = state)
    return status
end

"""
    solve(prob::NonlinearProblem, x₀, method, p = nothing; options = nothing, kwargs...)

Solve [`NonlinearProblem`](@ref) `prob` from `x₀` with `method` and parameters `p`, and return
`(x, status)` where `status` is a [`SolverStatus`](@ref). For an R3 array iterate `x₀`, `solve`
copies `x₀` so the caller's initial guess is not mutated.

`solve` adapts `method` to `real(eltype(x₀))` with [`ToReal`](@ref), builds [`Options`](@ref) in
`eltype(x₀)` from `kwargs...` or takes `options`, calls [`init_state`](@ref), and delegates to
[`solve!!`](@ref). Passing both `options` and keyword tolerances raises an `ArgumentError`, and
so does an `options` that is not an `Options{real(eltype(x₀))}`: a `Float32` iterate takes
`Options(Float32; …)`, because the default tolerances of `Options(Float64)` are below what a
`Float32` solve reaches. The
residual norm `rnorm` is finite up to `floatmax`, but the merit `‖F‖²` of a line search overflows
to `Inf` above `‖F‖ = sqrt(floatmax)`, and a solve with a line search then returns `NONFINITE`.
"""
function solve(
        prob::NonlinearProblem, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    R = real(eltype(x₀))
    opt = resolve_options(eltype(x₀), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    state = init_state(m, prob, x, p)
    x, status, _ = solve!!(state, prob, x, m, p, opt)
    return x, status
end
