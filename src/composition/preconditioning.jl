# Left preconditioning: the outer method runs on the residual `rˡ(y) = y - N(y)` of an inner
# method `N`, so every line-search trial of the outer costs one inner solve and its merit is
# `‖rˡ‖²` (`Brune:2015`, `CaiLi:2011`, `Dolean:2016`). With `N` the restricted additive Schwarz
# map of `src/composition/ras.jl` it is RASPEN, whose Jacobian is exact from the block factors of
# the last sweep; with any other `N` its derivative is a central difference of `rˡ`, one inner step
# at each of `x ± h v`.

"""
    LeftPreconditioned(outer, inner)

The left-preconditioned method of [`NonlinearProblem`](@ref) `prob`: the method `outer` runs on
the residual ``r^\\ell(y) = y - N(y)`` of the inner map ``N``, one [`step!!`](@ref) of `inner` from
`(y, F(y))`, instead of on `F` itself. Every trial point of `outer`'s globalization therefore
costs one inner step, and every merit value it reads is ``\\|r^\\ell\\|_2^2``. The zeros of
``r^\\ell`` are the fixed points of ``N``.

`LeftPreconditioned(Newton(), RAS(partition))` is RASPEN: the merit is on the restricted additive
Schwarz map, and the Jacobian of ``r^\\ell`` is the exact form of [`raspen_jvp!`](@ref) built from
the block factors of the last sweep (`Dolean:2016`, eq. 3.17). RASPEN needs converged local solves,
and at one inner step its outer step is Newton's step. With any other `inner` the derivative of
``r^\\ell`` is a central difference, one inner step at each of ``x \\pm h v``, which
[`prepare_ad`](@ref) prepares at `init_state`; a globalization that reads no slope, such as
[`Static`](@ref), pays for none.

The [`solve!!`](@ref) status of a `LeftPreconditioned` solve is the one of the loop: its `fnorm` is
``\\|F(x)\\|`` of the global residual at the returned iterate, and its convergence tests apply to
that residual. `init_state` raises the `ArgumentError` of an inner [`RAS`](@ref) for a problem
without block residuals.
"""
struct LeftPreconditioned{O, I}
    outer::O
    inner::I
end

Adapt.@adapt_structure LeftPreconditioned

"""
    LeftPreconditionedProblem(prob, inner, state, work)

The problem that `outer` of a [`LeftPreconditioned`](@ref) method solves: the user's problem
`prob`, the inner method `inner`, the inner method's state in the iterate's element type, and the
iterate buffer that one inner step runs on — a step of an array iterate writes into its `x`, so the
residual of this problem steps a copy. Its residual is ``r^\\ell(y) = y - N(y)``, one
[`step!!`](@ref) of `inner` from `(y, F(y))` subtracted from `y`, so [`residual!!`](@ref) evaluates
it and a [`LineFunction`](@ref) merits it.
"""
struct LeftPreconditionedProblem{P, I, S, W} <: AbstractNonlinearProblem
    prob::P
    inner::I
    state::S
    work::W
end

"""
    LeftPreconditionedState

The state of a [`LeftPreconditioned`](@ref) method: the [`LeftPreconditionedProblem`](@ref) it
runs on, the outer method's state, and the buffer of the global residual that a step returns. The
outer state's residual workspace holds ``r^\\ell``; the problem's inner state holds the workspace
of the inner step, and its counters are the ones [`ras_counts`](@ref) reports.
"""
struct LeftPreconditionedState{P, S, FX}
    problem::P
    outer_state::S
    Fx::FX
end

"""
    RASPenPreparation

The AD preparation of a [`LeftPreconditioned`](@ref) method whose inner state is a
[`RASState`](@ref): the RAS state, whose block factors the RASPEN Jacobian product reads.
[`jacobian!!`](@ref) and [`jvp!!`](@ref) apply the exact form of
[`raspen_jacobian!`](@ref) and [`raspen_jvp!`](@ref).
"""
struct RASPenPreparation{A}
    ras::A
end

"""
    LeftPreconditionedAD

The AD preparation of a [`LeftPreconditioned`](@ref) method whose inner state is not a
[`RASState`](@ref): the iterate and residual buffers of the two points of a central difference,
and the direction buffer that seeds one column at a time. [`jvp!!`](@ref) evaluates ``r^\\ell`` at
``x \\pm h v`` and forms ``(r^\\ell(x + hv) - r^\\ell(x - hv)) / 2h``, and [`jacobian!!`](@ref)
takes one column per call of it.

A central difference, because one step of an inner method is not `ForwardDiff.Dual`-generic: a
method carries the real type of the working precision in its type, so a dual iterate reaches the
`StepInfo` of its globalization and the right-hand-side scaling of its linear factor in the wrong
type.
"""
struct LeftPreconditionedAD{XP, XM, RP, RM, E}
    xp::XP
    xm::XM
    rp::RP
    rm::RM
    e::E
end

@inline function init_state(m::LeftPreconditioned, prob::NonlinearProblem, x, p)
    inner_state = init_state(m.inner, prob, x, p)
    lprob = LeftPreconditionedProblem(prob, m.inner, inner_state, buffer(x))
    outer_state = init_state(m.outer, lprob, x, p)
    return LeftPreconditionedState(lprob, outer_state, buffer(x))
end

@inline residual_buffer(state::LeftPreconditionedState) = state.Fx

# `rˡ(y) = y - N(y)`: the residual is the difference between the iterate and one inner step from
# it. The inner state is started at `y` on each evaluation, so successive evaluations do not share
# a merit anchor, and its buffers are reused. The two methods are the buffer and the value form of
# the residual seam, each stated on its own slot so that neither is ambiguous against it.
@inline function residual!!(y::AbstractArray, prob::LeftPreconditionedProblem, x, p)
    return preconditioned_residual(y, prob, x, p)
end

@inline function residual!!(::Nothing, prob::LeftPreconditionedProblem, x, p)
    return preconditioned_residual(nothing, prob, x, p)
end

# The counters of a method state that reports them, and the restart that keeps them: the inner map
# of a `LeftPreconditioned` solve is started again at every evaluation, and its counters belong to
# the solve, so they survive that restart.
@inline state_counts(st) = nothing
@inline restart_with_counts(st, ::Nothing) = st

@inline function preconditioned_residual(y, prob::LeftPreconditionedProblem, x, p)
    st = prob.state
    xw = copy!!(prob.work, x)
    Fx = residual!!(residual_buffer(st), prob.prob, xw, p)
    counts = state_counts(st)
    st = start!!(st, prob.inner, prob.prob, xw, Fx, p, rnorm(Fx))
    st = restart_with_counts(st, counts)
    x₊, _, _, _ = step!!(st, prob.inner, prob.prob, xw, Fx, p)
    return map!!(-, y, x, x₊)
end

@inline jacobian_prototype(::LeftPreconditionedProblem) = nothing

# A preconditioned problem has no analytic Jacobian of `rˡ`, so a `Picard` outer prepares its
# derivative through `prepare_ad`, and a globalization that reads no slope pays for none.
@inline function picard_jvpprep(m::Picard, prob::LeftPreconditionedProblem, Fx, x, p)
    return Fx === nothing ? nothing : prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
end

# `Picard` on the preconditioned problem: the same fixed-point step as on a `NonlinearProblem`,
# with the merit of the outer's line search taken on `rˡ` through the problem's residual.
@inline function init_state(m::Picard, prob::LeftPreconditionedProblem, x, p)
    Fx = buffer(x)
    jvpprep = picard_jvpprep(m, prob, Fx, x, p)
    merit = init_merit(m.globalization, x)
    if Fx === nothing
        return PicardState(jvpprep, nothing, nothing, nothing, merit)
    else
        return PicardState(
            jvpprep, Fx, buffer(x), init_globalization(m.globalization, x), merit)
    end
end

@inline function step!!(
        state::PicardState, m::Picard, prob::LeftPreconditionedProblem, x, Fx, p)
    ω = real(eltype(x))(m.damping)
    d = map!!(fi -> -ω * fi, state.d, Fx)
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, MeasuredSlope(),
        state.merit)
    return x₊, F₊, with_merit(state, merit), info
end

function prepare_ad(
        backend::AutoForwardDiff, prob::LeftPreconditionedProblem, Fx::AbstractArray,
        x::AbstractArray, p)
    # the preparation reads the inner state's block factors, so the state that is a `RASState` —
    # the state of a [`RAS`](@ref), and of any member that forwards to one — is a RASPEN inner map
    if prob.state isa RASState
        return RASPenPreparation(prob.state)
    end
    return LeftPreconditionedAD(
        similar(x), similar(x), similar(Fx), similar(Fx), similar(x))
end

function jvp!!(Jv, prep::RASPenPreparation, prob::LeftPreconditionedProblem, x, v, p)
    return raspen_jvp!(Jv, prep.ras, prob.prob, x, v, p)
end

function jacobian!!(J, prep::RASPenPreparation, prob::LeftPreconditionedProblem, x, p)
    return raspen_jacobian!(J, prep.ras, prob.prob, x, p)
end

# The central difference of `rˡ` along `v`, written into `col`: one inner step at each of
# `x ± h v`, with ``h = \\sqrt[3]{\\varepsilon} \\max(1, \\|x\\|) / \\|v\\|``, the step whose
# truncation and rounding errors balance.
function difference_jvp!(
        col, prep::LeftPreconditionedAD, prob::LeftPreconditionedProblem, x, v, p)
    R = real(eltype(x))
    vnorm = rnorm(v)
    if iszero(vnorm)
        fill!(col, zero(eltype(col)))
        return col
    end
    h = cbrt(eps(R)) * max(one(R), rnorm(x)) / vnorm
    prep.xp .= x .+ h .* v
    prep.xm .= x .- h .* v
    rp = residual!!(prep.rp, prob, prep.xp, p)
    rm = residual!!(prep.rm, prob, prep.xm, p)
    col .= rp
    col .-= rm
    col ./= 2h
    return col
end

function jvp!!(Jv, prep::LeftPreconditionedAD, prob::LeftPreconditionedProblem, x, v, p)
    return difference_jvp!(Jv, prep, prob, x, v, p)
end

function jacobian!!(J, prep::LeftPreconditionedAD, prob::LeftPreconditionedProblem, x, p)
    for j in axes(J, 2)
        copyto!(prep.e, unit_direction(eltype(prep.e), length(prep.e), j))
        difference_jvp!(view(J, :, j), prep, prob, x, prep.e, p)
    end
    return J
end

# The host unit vector of column `j`, which `copyto!` puts into a device direction buffer.
function unit_direction(::Type{T}, n::Integer, j::Integer) where {T}
    e = zeros(T, n)
    e[j] = one(T)
    return e
end

@inline function start!!(
        state::LeftPreconditionedState, m::LeftPreconditioned, prob, x, Fx, p, fnorm₀)
    lprob = state.problem
    Fxˡ = residual!!(residual_buffer(state.outer_state), lprob, x, p)
    outer_state = start!!(
        state.outer_state, m.outer, lprob, x, Fxˡ, p, rnorm(Fxˡ))
    return LeftPreconditionedState(lprob, outer_state, state.Fx)
end

@inline function step!!(
        state::LeftPreconditionedState, m::LeftPreconditioned, prob, x, Fx, p)
    lprob = state.problem
    R = real(eltype(x))
    Fxˡ = residual!!(residual_buffer(state.outer_state), lprob, x, p)
    x₊, _, outer_state, info = step!!(state.outer_state, m.outer, lprob, x, Fxˡ, p)
    F₊ = residual!!(state.Fx, prob, x₊, p)
    info₊ = StepInfo{R}(
        info.code, rnorm(F₊), info.stepnorm, info.failures, info.promoted, info.limited)
    return x₊, F₊, LeftPreconditionedState(lprob, outer_state, state.Fx), info₊
end

@inline function finish!!(
        state::LeftPreconditionedState, m::LeftPreconditioned, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    x_out, st, outer_state = finish!!(
        state.outer_state, m.outer, state.problem, x, Fx, status)
    return x_out, st, LeftPreconditionedState(state.problem, outer_state, state.Fx)
end

"""
    ras_counts(state::LeftPreconditionedState)

The counters of the [`RAS`](@ref) state of a [`LeftPreconditioned`](@ref) method's inner map, as
[`ras_counts`](@ref) reports them. A method whose inner map is not a `RAS` raises an
`ArgumentError`.
"""
ras_counts(state::LeftPreconditionedState) = ras_counts(state.problem.state)

"""
    ras_counts(state::MultiplicativeState)

The counters of the [`RAS`](@ref) member of a [`Multiplicative`](@ref) method, as
[`ras_counts`](@ref) reports them. A method with no `RAS` member raises an `ArgumentError`.
"""
ras_counts(state::MultiplicativeState) = _ras_member_counts(state.states)

function _ras_member_counts(states::Tuple)
    first(states) isa RASState && return ras_counts(first(states))
    return _ras_member_counts(Base.tail(states))
end

function _ras_member_counts(::Tuple{})
    throw(ArgumentError(
        "ras_counts needs a state with a RAS method: a RAS state, a LeftPreconditioned state whose inner method is a RAS, or a Multiplicative state with a RAS member"))
end

function ras_counts(state)
    throw(ArgumentError(
        "ras_counts needs a state with a RAS method: a RAS state, a LeftPreconditioned state whose inner method is a RAS, or a Multiplicative state with a RAS member"))
end
