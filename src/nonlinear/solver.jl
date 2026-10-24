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

"""
    solve!!(state, prob::NonlinearProblem, x₀, method, p = nothing, options = Options(eltype(x₀)))

Run the nonlinear solver loop from `x₀` using `state = init(method, prob, x₀, p)` and return
`(x, status, state)`. `solve!!` is the in-place and kernel entry point: inside a
`KernelAbstractions.@kernel`, a caller prepares `state₀ = init(m, prob, x₀, p)` on the host and
calls `solve!!(state₀, prob, x_i, m, p_i, opt)` in each work-item.

Before the loop, `solve!!` evaluates `Fx = residual!!(nothing, prob, x₀, p)` and `fnorm₀ = rnorm(Fx)`.
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
`options.max_stalls` consecutive `STALLED` steps return `STALLED`. Reaching
`options.max_iterations` without convergence returns `MAXITERS`.
"""
function solve!!(
        state, prob::NonlinearProblem, x₀, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    x = x₀
    Fx = residual!!(nothing, prob, x, p)
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
        is_conv = if info.code == STALLED
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

"""
    solve(prob::NonlinearProblem, x₀, method, p = nothing; options = nothing, kwargs...)

Solve [`NonlinearProblem`](@ref) `prob` from `x₀` with `method` and parameters `p`, and return
`(x, status)` where `status` is a [`SolverStatus`](@ref).

`solve` adapts `method` to `real(eltype(x₀))` with [`ToReal`](@ref), builds [`Options`](@ref) in
`eltype(x₀)` from `options` or `kwargs...`, calls [`init`](@ref), and delegates to
[`solve!!`](@ref). In `Float32`, a residual norm above `sqrt(floatmax(Float32)) ≈ 1.8e19` overflows
`norm2` to `Inf` and returns `NONFINITE`.
"""
function solve(
        prob::NonlinearProblem, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    T = eltype(x₀)
    R = real(T)
    opt = if options !== nothing
        convert(Options{R}, options)
    elseif isempty(kwargs)
        default_options(R)
    else
        Options(T; kwargs...)
    end
    m = Adapt.adapt(ToReal{R}(), method)
    state = init(m, prob, x₀, p)
    x, status, _ = solve!!(state, prob, x₀, m, p, opt)
    return x, status
end
