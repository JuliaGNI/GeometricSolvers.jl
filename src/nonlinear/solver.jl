"""
    NoRefinement()

The default iterative-refinement setting of [`Newton`](@ref), which returns the linear solve's
step unchanged.
"""
struct NoRefinement end

"""
    init_state(method, prob::AbstractNonlinearProblem, x, p)

Build the state of `method` for [`AbstractNonlinearProblem`](@ref) `prob` at the initial iterate `x`
and parameters `p`: its workspace, its AD preparation and its rule state, in one concrete type per
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
    step!!(state, method, prob::AbstractNonlinearProblem, x, Fx, p)

Compute one nonlinear solver step from `(x, Fx)` and return `(x₊, F₊, state₊, info)` where `info`
is a [`StepInfo`](@ref). A line-search step of [`Newton`](@ref) or [`Picard`](@ref) reads the
anchor merit that the last [`start!!`](@ref) or `step!!` on its state set, so it takes only the
`(x, Fx)` that they returned; on a state that no `start!!` saw, or that [`solve!!`](@ref)
returned, it returns `NONFINITE` and keeps `x` and `Fx`.
"""
function step!! end

"""
    start!!(state, method, prob::AbstractNonlinearProblem, x, Fx, p, fnorm₀) -> state

Prepare `state` for a solve from `x`, with residual `Fx` and its norm `fnorm₀`, and return the
state that the first step reads. [`solve!!`](@ref) calls it once, before its stopping test at 0
iterations, so a state reused across solves starts each solve afresh. The default returns
`state`. [`Newton`](@ref) resets its rule state, and [`Newton`](@ref) and [`Picard`](@ref) set
the anchor merit `norm2(Fx)` of a line search, which `init_state` leaves at `NaN`.
"""
@inline start!!(state, method, prob, x, Fx, p, fnorm₀) = state

"""
    finish!!(state, method, prob::AbstractNonlinearProblem, x, Fx, status) -> (x, status, state)

Close a solve that ends at `x`, with residual `Fx` and `status`, and return the triple that
[`solve!!`](@ref) returns. [`solve!!`](@ref) calls it at every return. The default returns
`(x, status, state)`. A method may return another iterate; it then sets `status.fnorm` itself,
and for an array iterate it writes that iterate into `x` and returns `x`. [`Newton`](@ref) and
[`Picard`](@ref) set the anchor merit of a line search to `NaN` again, as `init_state` does.
"""
@inline finish!!(state, method, prob, x, Fx, status) = (x, status, state)

@inline state_gradratio(state) = NaN
@inline state_jnorm(state) = 0

@inline function ok_status(status::SolverStatus{R}) where {R}
    return SolverStatus{R}(
        SUCCESS, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, false)
end

@inline function stationary_status(status::SolverStatus{R}) where {R}
    return SolverStatus{R}(
        SUCCESS, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, true)
end

@inline function end_status(status::SolverStatus{R}, code::ReturnCode) where {R}
    return SolverStatus{R}(
        code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, false)
end

# The status that the residual test of `solve!!` reads at `x`. It is `status` itself, unless a
# problem steps on a scaled residual and converges on its unscaled one: a `Deflated` problem sets
# `fnorm` to the norm of its inner residual.
@inline converged_status(state, prob, x, status) = status

@inline function stationary_converged(
        ::AbstractNonlinearProblem, ::Options{R}, state, ::SolverStatus{R}) where {R}
    return false
end
@inline function stationary_converged(
        ::LeastSquaresProblem, options::Options{R}, state, status::SolverStatus{R}) where {R}
    status.iterations >= options.min_iterations || return false
    jnorm = state_jnorm(state)
    grad = state_gradratio(state)
    return jnorm > zero(R) && status.fnorm > zero(R) && grad <= options.g_tol
end

# `residual!!` at a primal iterate of a solve, refused when the residual contains a
# `ForwardDiff.Dual`. `solve!!` and the globalizations evaluate every residual of a solve through it.
# The test reads only the type, so it costs nothing on a residual of one primal type.
@inline function primal_residual!!(y, prob, x, p)
    Fx = residual!!(y, prob, x, p)
    has_dual_type(typeof(Fx)) && throw(ArgumentError(
        "solve!! residual contains a ForwardDiff.Dual; solve!! takes primal x₀ and p and needs a primal residual on them, so pass dual x₀ or p to solve(prob, x₀, method, p)"))
    return Fx
end

"""
    solve!!(state, prob::AbstractNonlinearProblem, x₀, method, p = nothing, options = Options(eltype(x₀)))

Run the nonlinear solver loop from `x₀` using `state = init_state(method, prob, x₀, p)` and return
`(x, status, state)`. `solve!!` is the in-place and kernel entry point: inside a
`KernelAbstractions.@kernel`, a caller prepares `state₀ = init_state(m, prob, x₀, p)` on the host
and calls `solve!!(state₀, prob, x_i, m, p_i, opt)` in each work-item.

At the start of each call, `solve!!` evaluates `Fx = residual!!(residual_buffer(state), prob, x₀,
p)` and `fnorm₀ = rnorm(Fx)`, and calls [`start!!`](@ref) once, which resets the per-solve state
of the method (the reuse age and the stalled flag under factor reuse, the radius under
[`DogLeg`](@ref), the damping parameter under [`LevenbergMarquardt`](@ref), and the anchor merit
of a line search). Every return goes through
[`finish!!`](@ref), and `solve!!` returns the triple that `finish!!` returns.

When `fnorm₀` is not finite — an entry of `Fx` is not finite, or `‖F‖` is above `floatmax` — it
returns `NONFINITE` with `0` iterations. When the residual test of [`Options`](@ref) already holds
at `0` iterations, it returns `SUCCESS`. When `options.max_iterations == 0`, it returns `MAXITERS`
with `0` iterations. `options` must be an `Options{real(eltype(x₀))}`; any other raises an
`ArgumentError`, and so does a residual that contains a `ForwardDiff.Dual`, at `x₀` or at any later
evaluation inside a step: [`solve`](@ref) takes dual inputs.

Inside the loop, each trip calls [`step!!`](@ref) and updates `status` via [`record`](@ref).
`SINGULAR` and `NONFINITE` return immediately. A `LINESEARCH_FAILED` step whose `stepnorm` is zero
(meaning the failed line-search trial did not decrease the merit) stops immediately with
`LINESEARCH_FAILED`, unless the stationarity test below holds, which a nonzero-residual
least-squares minimum reaches with every step a line search can try; one that decreased the merit
keeps its new iterate, increments
`status.step_failures`, and continues. The residual test returns `SUCCESS` with
`status.stationary == false`; on a [`LeastSquaresProblem`](@ref) where the residual test does not
hold, the stationarity test `‖Jᴴ F‖ / (‖J‖_F ‖F‖) ≤ options.g_tol` (with `‖J‖_F > 0` and
`‖F‖ > 0`) returns `SUCCESS` with `status.stationary == true`. The state's diagnostics are those
of its last fresh factorisation, at the start of the step, so the ratio is re-measured at the
iterate the solve returns before it ends on this test. When neither test holds
after a step that is neither `STALLED` nor `limited`, a step test that holds returns `STALLED`
at once, and so does a `SUCCESS` step of zero norm, which leaves the iterate where the next step
would start — unless the stationarity test, re-measured at that iterate, holds, which a
nonzero-residual least-squares minimum reaches with a step of any length, a zero one included;
`min_iterations` bounds both. After a `STALLED` step only the residual and
stationarity tests apply,
and `options.max_stalls` consecutive `STALLED` steps return `STALLED`. After a step whose
`limited` is set only the residual and stationarity tests apply too, because a radius or a
damping, not the distance to a root, sets its norm; such a step is not a stall. Reaching
`options.max_iterations` without convergence returns `MAXITERS`.
"""
function solve!!(
        state, prob::AbstractNonlinearProblem, x₀, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    check_options(eltype(x₀), options)
    x = x₀
    Fx = primal_residual!!(residual_buffer(state), prob, x, p)
    fnorm₀ = rnorm(Fx)
    state = start!!(state, method, prob, x, Fx, p, fnorm₀)
    status = SolverStatus{R}(SUCCESS, Int32(0), fnorm₀, zero(R), Int32(0), false, false)
    if !isfinite(fnorm₀)
        return finish!!(state, method, prob, x, Fx, end_status(status, NONFINITE))
    end
    status = converged_status(state, prob, x, status)
    ftest₀ = status.fnorm
    if residual_converged(options, status, ftest₀)
        return finish!!(state, method, prob, x, Fx, status)
    end
    if options.max_iterations == Int32(0)
        return finish!!(state, method, prob, x, Fx, end_status(status, MAXITERS))
    end
    stalls = Int32(0)
    while true
        x, Fx, state, info = step!!(state, method, prob, x, Fx, p)
        status = converged_status(state, prob, x, record(status, info))
        if info.code == SINGULAR || info.code == NONFINITE
            return finish!!(state, method, prob, x, Fx, status)
        end
        if info.code == LINESEARCH_FAILED && iszero(info.stepnorm) &&
           !stationary_converged(prob, options, state, status)
            return finish!!(state, method, prob, x, Fx, status)
        end
        if residual_converged(options, status, ftest₀)
            return finish!!(state, method, prob, x, Fx, ok_status(status))
        end
        # the diagnostics are re-measured at `x` at most once per step
        refreshed = false
        if stationary_converged(prob, options, state, status)
            state = refresh_diagnostics!!(state, method, prob, x, Fx, p)
            refreshed = true
            if stationary_converged(prob, options, state, status)
                return finish!!(state, method, prob, x, Fx, stationary_status(status))
            end
        end
        # the residual test fails: the step test, or a zero step, ends the solve unconverged,
        # unless the iterate it returns is stationary, which a step of zero length cannot show
        if !(info.code == STALLED || info.limited) &&
           (step_converged(options, status, rnorm(x)) ||
            (info.code == SUCCESS && iszero(info.stepnorm) &&
             status.iterations >= options.min_iterations))
            refreshed || (state = refresh_diagnostics!!(state, method, prob, x, Fx, p))
            if stationary_converged(prob, options, state, status)
                return finish!!(state, method, prob, x, Fx, stationary_status(status))
            end
            return finish!!(state, method, prob, x, Fx, end_status(status, STALLED))
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

# A one-variable least-squares problem iterates on the one-entry vector path, and its result goes
# back to the caller as the `Number` it is: the state is the lifted state that `init_state` built.
function solve!!(state, prob::LeastSquaresProblem, x₀::Number, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    prob, method, x₀ = lift_scalar_iterate(prob, method, x₀)
    x, status, state₊ = solve!!(state, prob, x₀, method, p, options)
    return first(x), status, state₊
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

# The iterate, the adapted method, the options and the method state of a solve from `x₀`, which
# `init` and `solve` share: an array `x₀` is copied, so that neither of them mutates it.
@inline function prepare_solve(prob, x₀, method, p, options, kwargs)
    R = real(eltype(x₀))
    opt = resolve_options(eltype(x₀), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    return x, m, opt, init_state(m, prob, x, p)
end

"""
    NonlinearSolver(prob, x, method, options, state, p)

A preallocated nonlinear solver for [`NonlinearProblem`](@ref), [`StationarityProblem`](@ref) or
[`LeastSquaresProblem`](@ref) `prob` with `method` and [`Options`](@ref) `options`, holding the
workspace and AD preparation in `state`, the iterate `x` and the parameters `p`, in the fields
`x::X` and `p::Q`. It keeps the element type `T` and the length `n` of `x` and the types `X` of `x`
and `Q` of `p`, which [`solve!`](@ref) checks.
Build it with [`init(prob, x₀, method, p; options = nothing, kwargs...)`](@ref init) and solve
with [`solve!(solver)`](@ref solve!) or [`solve!(solver, x, p)`](@ref solve!).
"""
mutable struct NonlinearSolver{T, P <: AbstractNonlinearProblem, M, O <: Options, S,
    X <: Union{AbstractVector{T}, T}, Q}
    const prob::P
    const method::M
    const options::O
    const n::Int
    state::S
    x::X
    p::Q
    function NonlinearSolver(prob::P, x::X, method::M, options::O, state::S,
            p::Q) where {P <: AbstractNonlinearProblem, M, O <: Options, S,
            X <: Union{AbstractVector, Number}, Q}
        new{eltype(X), P, M, O, S, X, Q}(prob, method, options, length(x), state, x, p)
    end
end

"""
    init(prob::AbstractNonlinearProblem, x₀, method, p = nothing; options = nothing, kwargs...)

Allocate and prepare a [`NonlinearSolver`](@ref) for `prob` and `method` from the initial iterate
`x₀` and the parameters `p`, with the arguments of [`solve`](@ref), so that
`solve!(init(args...; kwargs...))` returns what `solve(args...; kwargs...)` returns on primal
inputs. The solver holds a copy of an array `x₀`, so `init` and [`solve!(solver)`](@ref solve!) do
not mutate `x₀`; it holds a `StaticArray` or a `Number` `x₀` and `p` themselves. Passing both
`options` and keyword tolerances raises an `ArgumentError`, and so does an `options` that is not an
`Options{real(eltype(x₀))}`. Any `ForwardDiff.Dual` in `x₀` or `p` raises an `ArgumentError`; use
[`solve`](@ref) for implicit differentiation. `init` does not check `prob`, which may capture an
object whose type holds a `ForwardDiff.Dual`; a residual that contains a `ForwardDiff.Dual` on
primal `x` and `p` raises an `ArgumentError` in [`solve!`](@ref), from [`solve!!`](@ref).
"""
function init(
        prob::AbstractNonlinearProblem, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p))
        throw(ArgumentError(
            "init does not accept ForwardDiff.Dual inputs; use solve(prob, x₀, method, p) for implicit differentiation"))
    end
    x, m, opt, state = prepare_solve(prob, x₀, method, p, options, kwargs)
    return NonlinearSolver(prob, x, m, opt, state, p)
end

"""
    solve!(solver::NonlinearSolver; b = nothing)
    solve!(solver::NonlinearSolver, x, p = solver.p; b = nothing)

Solve `solver.prob`, using the preallocated workspace and AD preparation in `solver`, store the
solution in `solver.x`, and return `(solver.x, status)` where `status` is a
[`SolverStatus`](@ref). `solve!(solver)` solves from `solver.x` with `solver.p`, the iterate and
the parameters that [`init`](@ref) or the last `solve!(solver, x, p)` stored, and writes the
solution into `solver.x` when it is an array.
`solve!(solver, x, p)` stores `x` in `solver.x` and `p` in `solver.p`, then solves from `x` with
`p`; an array `x` receives the solution in place and stays in `solver.x`. Without `p`, it solves
with the parameters that the solver holds. For a [`StationarityProblem`](@ref), or a
[`BlockEliminated`](@ref) of one, the keyword `b` overrides the datum of the solve without
reallocating. An `x` whose element type or length differs
from the iterate of `init` raises an `ArgumentError`, and so do an `x` of another type, a `p` of
another type than those of `init`, and any `ForwardDiff.Dual` in `x` or `p`. A refused call
changes neither `solver` nor `x`.
"""
solve!(solver::NonlinearSolver; b = nothing) = solve!(solver, solver.x, solver.p; b = b)

function solve!(solver::NonlinearSolver{T, P, M, O, S, X, Q}, x,
        p = solver.p; b = nothing) where {T, P, M, O, S, X, Q}
    if has_dual_type(typeof(x)) || has_dual_type(typeof(p))
        throw(ArgumentError(
            "solve! does not accept ForwardDiff.Dual inputs; use solve(prob, x₀, method, p) for implicit differentiation"))
    end
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
    P <: StationarityProblem && return _solve_stationarity_solver!(solver, x, p, b)
    is_elim_stationarity_problem(P) &&
        return _solve_elim_stationarity_solver!(solver, x, p, b)
    b === nothing ||
        throw(ArgumentError(
            "keyword `b` is only supported for a `StationarityProblem` or a `BlockEliminated` `StationarityProblem`"))
    solver.x = x
    solver.p = p
    x₊, status, state = solve!!(
        solver.state, solver.prob, x, solver.method, p, solver.options)
    solver.x = x₊
    solver.state === state || (solver.state = state)
    return solver.x, status
end

"""
    solve(prob::AbstractNonlinearProblem, x₀, method, p = nothing; options = nothing, kwargs...)

Solve [`AbstractNonlinearProblem`](@ref) `prob` from `x₀` with `method` and parameters `p`, and return
`(x, status)` where `status` is a [`SolverStatus`](@ref). For an R3 array iterate `x₀`, `solve`
copies `x₀` so the caller's initial guess is not mutated. On primal inputs,
`solve(args...; kwargs...)` returns what `solve!(init(args...; kwargs...))` returns, without
building a [`NonlinearSolver`](@ref).

`solve` adapts `method` to `real(eltype(x₀))` with [`ToReal`](@ref), builds [`Options`](@ref) in
`eltype(x₀)` from `kwargs...` or takes `options`, calls [`init_state`](@ref), and delegates to
[`solve!!`](@ref). Passing both `options` and keyword tolerances raises an `ArgumentError`, and
so does an `options` that is not an `Options{real(eltype(x₀))}`: a `Float32` iterate takes
`Options(Float32; …)`, because the default tolerances of `Options(Float64)` are below what a
`Float32` solve reaches. The
residual norm `rnorm` is finite up to `floatmax`, but the merit `‖F‖²` of a line search overflows
to `Inf` above `‖F‖ = sqrt(floatmax)`, and a solve with a line search then returns `NONFINITE`.

When `x₀` or `p` carries a `ForwardDiff.Dual` (or `Complex{<:ForwardDiff.Dual}`) leaf, `solve`
strips duals via [`StripDual`](@ref), runs the primal nonlinear solve to `(x*, status)`, and on
`status.code === SUCCESS` solves the implicit-function-theorem tangent equation
``J(x^*, p)\\,\\dot{x}_k = -(\\partial F / \\partial p_k)(x^*, p)`` at `x*` from a single
unregularised factorisation of ``J(x^*, p)`` and one dual residual evaluation ``F(\\hat{x}, p)``
seeded with zero partials on ``\\hat{x}``. Each tangent is one back-substitution with that factor,
without `refinement`, so its accuracy is that of the factor element type of `linsolve`: with
`LUFactorization(Float32)` on a `Float64` iterate, `x*` is accurate to `Float64` and the tangent
to `Float32`. A non-`SUCCESS` primal solve returns primal values at
`x*` with `NaN` partials and the primal `status`; a `SINGULAR` tangent factorisation at `x*`
performs no back-substitution and returns primal values at `x*` with `NaN` partials and
`status.code === SINGULAR`. Nested duals, distinct dual tags or partial counts across `x₀` and `p`,
and custom structs not registered with `Adapt.@adapt_structure` raise an `ArgumentError`. A
combinator method, [`Additive`](@ref), [`Multiplicative`](@ref), [`Repeat`](@ref),
[`Switch`](@ref) or [`Anderson`](@ref), has no implicit differentiation: with a
`ForwardDiff.Dual` in `x₀` or `p`, `solve` raises an `ArgumentError` before the primal solve.

`prob` may capture an object whose type holds a `ForwardDiff.Dual`, such as a
`ForwardDiff.GradientConfig` or a dual work buffer, and `solve` does not check it. A residual
that contains a `ForwardDiff.Dual` on the primal `x` and `p` of the solve raises an
`ArgumentError` from [`solve!!`](@ref); an in-place residual that writes a `ForwardDiff.Dual` into
its primal buffer raises the `MethodError` of that conversion instead.
"""
function solve(
        prob::AbstractNonlinearProblem, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p))
        return solve_implicit(prob, x₀, method, p, options, kwargs)
    end
    x, m, opt, state = prepare_solve(prob, x₀, method, p, options, kwargs)
    x, status, _ = solve!!(state, prob, x, m, p, opt)
    return x, status
end
