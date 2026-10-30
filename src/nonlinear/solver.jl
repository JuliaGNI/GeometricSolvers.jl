"""
    NoRefinement()

The default iterative-refinement setting of [`Newton`](@ref), which returns the linear solve's
step unchanged.
"""
struct NoRefinement end

@inline refine!!(d_slot, ::NoRefinement, prob, x, Fx, d, fac, p, jvpprep) = d

"""
    init(method, prob::NonlinearProblem, x, p = nothing)

Prepare the state of `method` for [`NonlinearProblem`](@ref) `prob` at the initial iterate `x` and
parameters `p`. In R1 (`SVector`, `MVector` or `Number`), `init` is allocation-free and returns an
`isbits` named tuple when `prob` and `method` are `isbits`.
"""
function init end

"""
    step!!(state, method, prob::NonlinearProblem, x, Fx, p)

Compute one nonlinear solver step from `(x, Fx)` and return `(x₊, F₊, state₊, info)` where `info`
is a [`StepInfo`](@ref).
"""
function step!! end

@inline function residual_converged(
        opt::Options{R}, status::SolverStatus{R}, fnorm₀::R) where {R}
    status.iterations >= opt.min_iterations || return false
    return status.fnorm <= max(opt.f_abstol, opt.f_reltol * fnorm₀)
end

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

@inline state_Fx(state) = hasproperty(state, :Fx) ? state.Fx : nothing

@inline function reset_state(state)
    if hasproperty(state, :age)
        return (; state..., age = Int32(0), stalled = false)
    elseif hasproperty(state, :radius) && hasproperty(state, :radius₀)
        return (; state..., radius = state.radius₀)
    else
        return state
    end
end

"""
    solve!!(state, prob::NonlinearProblem, x₀, method, p = nothing, options = Options(eltype(x₀)))

Run the nonlinear solver loop from `x₀` using `state = init(method, prob, x₀, p)` and return
`(x, status, state)`. `solve!!` is the in-place and kernel entry point: inside a
`KernelAbstractions.@kernel`, a caller prepares `state₀ = init(m, prob, x₀, p)` on the host and
calls `solve!!(state₀, prob, x_i, m, p_i, opt)` in each work-item.

At the start of each call, `solve!!` resets per-solve state counters (`age = 0, stalled = false`
under factor reuse, and `radius = radius₀` under [`DogLeg`](@ref)), evaluates
`Fx = residual!!(state_Fx(state), prob, x₀, p)`, and computes `fnorm₀ = rnorm(Fx)`.
When `fnorm₀` is not finite — including in `Float32` when `‖F‖ > sqrt(floatmax(Float32)) ≈ 1.8e19`,
where `norm2` overflows to `Inf` — it returns `NONFINITE` with `0` iterations. When
`converged(options, status, fnorm₀, rnorm(x₀))` already holds at `0` iterations, it returns
`SUCCESS`. When `options.max_iterations == 0`, it returns `MAXITERS` with `0` iterations.

Inside the loop, each trip calls [`step!!`](@ref) and updates `status` via [`record`](@ref).
`SINGULAR` and `NONFINITE` return immediately. A `LINESEARCH_FAILED` step whose `stepnorm` is zero
(meaning the failed line-search trial did not decrease the merit) stops immediately with
`LINESEARCH_FAILED`; one that decreased the merit keeps its new iterate, increments
`status.step_failures`, and continues. After a `STALLED` step, only the residual tolerance test
can declare `SUCCESS`, so a zero stepnorm never satisfies `x_abstol` or `x_reltol` by accident;
`options.max_stalls` consecutive `STALLED` steps return `STALLED`. After a step whose `limited`
is set, only the residual tolerance test can declare `SUCCESS` too, because a radius, not the
distance to a root, sets its norm; such a step is not a stall. Reaching
`options.max_iterations` without convergence returns `MAXITERS`.
"""
function solve!!(
        state, prob::NonlinearProblem, x₀, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    state = reset_state(state)
    x = x₀
    Fx = residual!!(state_Fx(state), prob, x, p)
    fnorm₀ = rnorm(Fx)
    status = SolverStatus{R}(SUCCESS, Int32(0), fnorm₀, zero(R), Int32(0), false)
    if !isfinite(fnorm₀)
        return x, end_status(status, NONFINITE), state
    end
    if converged(options, status, fnorm₀, rnorm(x))
        return x, status, state
    end
    if options.max_iterations == Int32(0)
        return x, end_status(status, MAXITERS), state
    end
    stalls = Int32(0)
    while true
        x, Fx, state, info = step!!(state, method, prob, x, Fx, p)
        status = record(status, info)
        if info.code == SINGULAR || info.code == NONFINITE
            return x, status, state
        end
        if info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
            return x, status, state
        end
        is_conv = if info.code == STALLED || info.limited
            residual_converged(options, status, fnorm₀)
        else
            converged(options, status, fnorm₀, rnorm(x))
        end
        if is_conv
            return x, ok_status(status), state
        end
        if info.code == STALLED
            stalls += one(Int32)
            if stalls >= options.max_stalls
                return x, end_status(status, STALLED), state
            end
        else
            stalls = Int32(0)
        end
        if status.iterations >= options.max_iterations
            return x, end_status(status, MAXITERS), state
        end
    end
end

@inline function default_options(::Type{R}) where {R <: Real}
    return Options{R}(
        zero(R), sqrt(eps(R)), zero(R), 2 * eps(R), Int32(0), Int32(100), Int32(2))
end

@inline function resolve_options(
        x₀, options::Union{Nothing, Options}, kwargs, caller::Symbol)
    T = eltype(x₀)
    R = real(T)
    if options !== nothing
        isempty(kwargs) || throw(ArgumentError(
            "pass either `options` or keyword tolerances to `$caller`, not both"))
        return convert(Options{R}, options)
    elseif isempty(kwargs)
        return default_options(R)
    else
        return Options(T; kwargs...)
    end
end

"""
    NonlinearSolver(prob, method, options, state)

A preallocated nonlinear solver for [`NonlinearProblem`](@ref) `prob` with `method` and
[`Options`](@ref) `options`, holding the workspace and AD preparation in `state`. Build it with
[`init(prob, x, method, p; options = nothing, kwargs...)`](@ref init) and solve in place with
[`solve!(x, solver, p)`](@ref solve!).
"""
mutable struct NonlinearSolver{P <: NonlinearProblem, M, O <: Options, S}
    const prob::P
    const method::M
    const options::O
    state::S
end

"""
    init(prob::NonlinearProblem, x::AbstractVector, method, p = nothing;
         options = nothing, kwargs...)

Allocate and prepare a [`NonlinearSolver`](@ref) for `prob` and `method` at a template iterate `x`
and representative parameters `p`, without mutating `x`. Passing both `options` and keyword
tolerances raises an `ArgumentError`. A `StaticArray` `x` raises an `ArgumentError` too, because
[`solve!`](@ref) cannot write the solution into it; solve it with [`solve`](@ref).
"""
function init(
        prob::NonlinearProblem, x::AbstractVector, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    x isa StaticArray && throw(ArgumentError(
        "`init` builds a solver for an array iterate; solve a `StaticArray` iterate with `solve`"))
    R = real(eltype(x))
    opt = resolve_options(x, options, kwargs, :init)
    m = Adapt.adapt(ToReal{R}(), method)
    state = init(m, prob, x, p)
    return NonlinearSolver(prob, m, opt, state)
end

"""
    solve!(x::AbstractVector, solver::NonlinearSolver, p = nothing)

Solve `solver.prob` in place into `x` using the preallocated workspace and AD preparation in
`solver`, and return `status::SolverStatus`. The solver does not keep the `p` given to
[`init`](@ref): pass the parameters to every call, because `solve!(x, solver)` solves with
`p = nothing`. An `x` whose element type or length differs from the template given to `init`
raises an `ArgumentError`.
"""
function solve!(x::AbstractVector, solver::NonlinearSolver, p = nothing)
    Fx = solver.state.Fx
    eltype(x) === eltype(Fx) && length(x) == length(Fx) || throw(ArgumentError(
        "`solve!` needs an iterate like the template given to `init`, of length " *
        "$(length(Fx)) and element type $(eltype(Fx)); got length $(length(x)) and " *
        "element type $(eltype(x))"))
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
`eltype(x₀)` from `options` or `kwargs...`, calls [`init`](@ref), and delegates to
[`solve!!`](@ref). Passing both `options` and keyword tolerances raises an `ArgumentError`. In
`Float32`, a residual norm above `sqrt(floatmax(Float32)) ≈ 1.8e19` overflows `norm2` to `Inf` and
returns `NONFINITE`.
"""
function solve(
        prob::NonlinearProblem, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    R = real(eltype(x₀))
    opt = resolve_options(x₀, options, kwargs, :solve)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    state = init(m, prob, x, p)
    x, status, _ = solve!!(state, prob, x, m, p, opt)
    return x, status
end
