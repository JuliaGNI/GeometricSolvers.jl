# The `Multiplicative` combinator and the `sync_merit` seam that keeps line-search members'
# merit baselines `φ(x) = ‖F(x)‖₂²` synchronised when another member moves `(x, Fx)`.

"""
    Multiplicative(m₁, m₂, ...)
    Multiplicative((m₁, m₂, ...))

Sequential nonlinear combinator (composite step ``G_m \\circ \\cdots \\circ G_1``) that
threads `(x, F(x))` through `m₁, m₂, ...` in order within a single outer solver iteration.
Before each member steps, [`sync_merit`](@ref) updates its line-search baseline merit to the
incoming ``\\|F(x)\\|_2^2`` when the member state has been started. The outer step is `STALLED`
when every member step is. An outer step of two or more member steps that is shorter than its
longest member step is `shortened` in its [`StepInfo`](@ref), so the rate test of
[`Options`](@ref) does not read it: its member steps cancel, as along a cycle of the members, and
its small norm is no sign of a contraction.
"""
struct Multiplicative{M <: Tuple}
    members::M
    function Multiplicative{M}(members::M) where {M <: Tuple}
        fieldcount(M) >= 1 ||
            throw(ArgumentError("Multiplicative requires at least one member"))
        return new{M}(members)
    end
end

Multiplicative(members::Tuple) = Multiplicative{typeof(members)}(members)
Multiplicative(m1, ms...) = Multiplicative((m1, ms...))
Multiplicative() = throw(ArgumentError("Multiplicative requires at least one member"))

Adapt.@adapt_structure Multiplicative

# State of `Multiplicative`: tuple of member states, residual buffer, and starting-iterate
# buffer `x₀` used to measure the outer step norm `‖x^{(m)} - x^{(0)}‖` in R3.
struct MultiplicativeState{S <: Tuple, FX, X0}
    states::S
    Fx::FX
    x₀::X0
end

"""
    sync_merit(state, method, merit::Real)

Update a started line-search member state's stored baseline merit
``\\phi(x) = \\|F(x)\\|_2^2`` to `merit` when another member has moved the iterate `(x, Fx)`
since `state`'s last step. Preserves `NaN` on an un-started state so calling `step!!`
without `start!!` still fails closed with `NONFINITE`, and leaves the merit of a `Static` or
`DogLeg` state (`merit === nothing`) as it is. A `Newton` state also sets the residual norm it
carries to `NaN`, so that its next step measures ``\\|F(x)\\|`` of the residual it is given.
"""
@inline sync_merit(state, method, merit::Real) = state

@inline function sync_merit(state::NewtonState, ::Newton, merit::Real)
    φ = state.merit isa Real && isfinite(state.merit) ? oftype(state.merit, merit) :
        state.merit
    return with_merit(state, φ, NaN)
end

@inline function sync_merit(state::PicardState, ::Picard, merit::Real)
    state.merit isa Real && isfinite(state.merit) || return state
    return with_merit(state, oftype(state.merit, merit))
end

@inline _sync_member_merits(::Tuple{}, ::Tuple{}, merit::Real) = ()
@inline function _sync_member_merits(states::Tuple, members::Tuple, merit::Real)
    s1 = sync_merit(first(states), first(members), merit)
    s_tail = _sync_member_merits(Base.tail(states), Base.tail(members), merit)
    return (s1, s_tail...)
end

@inline function sync_merit(state::AdditiveState, m::Additive, merit::Real)
    states = _sync_member_merits(state.states, m.members, merit)
    return AdditiveState(
        states, state.Fx, state.xs, state.Fxs, state.d, state.xt, state.work)
end

@inline function sync_merit(state::MultiplicativeState, m::Multiplicative, merit::Real)
    states = _sync_member_merits(state.states, m.members, merit)
    return MultiplicativeState(states, state.Fx, state.x₀)
end

@inline function init_state(m::Multiplicative, prob::CombinatorProblem, x, p)
    states = _init_member_states(m.members, prob, x, p)
    return MultiplicativeState(states, buffer(x), buffer(x))
end

@inline residual_buffer(state::MultiplicativeState) = state.Fx

@inline rate_test_rank(state::MultiplicativeState, x) = _members_rate_test_rank(state.states, x)
@inline function observe_rate!!(state::MultiplicativeState, m::Multiplicative, rate)
    states = _observe_member_rates!!(state.states, m.members, rate)
    return MultiplicativeState(states, state.Fx, state.x₀)
end
@inline function drop_factor!!(state::MultiplicativeState, m::Multiplicative)
    states = _drop_member_factors!!(state.states, m.members)
    return MultiplicativeState(states, state.Fx, state.x₀)
end

@inline function start!!(
        state::MultiplicativeState, m::Multiplicative, prob, x, Fx, p, fnorm₀)
    states = _start_member_states(state.states, m.members, prob, x, Fx, p, fnorm₀)
    return MultiplicativeState(states, state.Fx, state.x₀)
end

@inline function has_diagnostics(st::SolverStatus)
    return st.rank != Int32(-1) || isfinite(st.nullnorm) || isfinite(st.gradratio)
end

# As `_finish_member_states` of `Additive`, and the status is the last member's that carries
# diagnostics, with the residual norm at the iterate returned.
@inline function _finish_multiplicative_states(
        ::Tuple{}, ::Tuple{}, prob, x, Fx, status::SolverStatus{R},
        st_diag::SolverStatus{R}) where {R}
    return (), x, restored_best_status(st_diag, status.fnorm)
end

@inline function _finish_multiplicative_states(
        states::Tuple, members::Tuple, prob, x, Fx, status::SolverStatus{R},
        st_diag::SolverStatus{R}) where {R}
    x1, st1, s1 = finish!!(first(states), first(members), prob, x, Fx, status)
    st_next = has_diagnostics(st1) ? st1 : st_diag
    ss_tail, x_out, st_final = _finish_multiplicative_states(Base.tail(states),
        Base.tail(members), prob, x1, Fx, restored_best_status(status, st1.fnorm), st_next)
    return (s1, ss_tail...), x_out, st_final
end

@inline function finish!!(
        state::MultiplicativeState, m::Multiplicative, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    st_init = SolverStatus{R}(
        status.code, Int16(0), status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, false, Int16(0), Int32(-1), R(NaN), R(NaN),
        status.rate, status.contracted, status.reused)
    states, x_out, st_out = _finish_multiplicative_states(
        state.states, m.members, prob, x, Fx, status, st_init)
    # each member counts only its own sub-steps, so `Multiplicative` keeps no count
    return x_out, with_calls(st_out, 0, 0), MultiplicativeState(states, state.Fx, state.x₀)
end

@inline function _step_multiplicative_tail(
        ::Tuple{}, ::Tuple{}, prob::AbstractNonlinearProblem, x, Fx, p,
        fnorm::R, failures::Int32, promoted::Bool, limited::Bool, shortened::Bool,
        stalled::Bool, substeps::Int32, longest::Real) where {R}
    code = stalled ? STALLED : SUCCESS
    info = StepInfo{R}(code, fnorm, zero(R), failures, promoted, limited, shortened)
    return x, Fx, (), info, substeps, longest
end

# The members from the first one in `states` on, in order, with the number of member steps and the
# norm of the longest one. A hard failure of a member ends the sequence with that member's code
# and keeps the iterate that the members before it reached. Otherwise the sequence is `STALLED`
# when every member step is, and `SUCCESS` else.
@inline function _step_multiplicative_tail(
        states::Tuple, members::Tuple, prob::AbstractNonlinearProblem, x, Fx, p,
        fnorm::R, failures::Int32, promoted::Bool, limited::Bool, shortened::Bool,
        stalled::Bool, substeps::Int32, longest::Real) where {R}
    s_i = sync_merit(first(states), first(members), norm2(Fx))
    x_next, Fx_next, s_i_new, info_i = step!!(s_i, first(members), prob, x, Fx, p)
    failures_next = failures + info_i.failures
    promoted_next = promoted || info_i.promoted
    limited_next = limited || info_i.limited
    shortened_next = shortened || info_i.shortened
    substeps_next = substeps + one(Int32)
    # in the precision of the iterate, whatever the precision of the member's `StepInfo`
    longest_next = max(longest, oftype(longest, info_i.stepnorm))
    if is_hard_failure(info_i)
        states_sc = (s_i_new, Base.tail(states)...)
        info_sc = StepInfo{R}(
            info_i.code, info_i.fnorm, zero(R), failures_next, promoted_next, limited_next,
            shortened_next)
        return x_next, Fx_next, states_sc, info_sc, substeps_next, longest_next
    end
    x_out, Fx_out, ss_tail, info_out, substeps_out, longest_out = _step_multiplicative_tail(
        Base.tail(states), Base.tail(members), prob, x_next, Fx_next, p,
        info_i.fnorm, failures_next, promoted_next, limited_next, shortened_next,
        stalled && info_i.code == STALLED, substeps_next, longest_next)
    return x_out, Fx_out, (s_i_new, ss_tail...), info_out, substeps_out, longest_out
end

# Whether the `substeps` sub-steps of a sequential composite step cancel: two or more sub-steps
# whose outer step, of norm `stepnorm`, is shorter than the longest of them, of norm `longest`. A
# contraction whose sub-steps point the same way has an outer step at least as long as each of
# them, while sub-steps along a cycle of the members, such as the Newton cycle `0 → 1 → 0` of
# `x³ - 2x + 2`, have an outer step far shorter than each. The rate test does not read such a
# step, as it does not read a step that a line search shortened. One sub-step is the outer step
# itself, whose norm differs from the member's only by rounding. A `NaN` norm marks nothing.
@inline function cancelled_substeps(substeps::Int32, stepnorm::R, longest::R) where {R}
    return substeps >= Int32(2) && stepnorm < longest
end

# The step norm is the outer one, ``\|x^{(m)} - x^{(0)}\|``, also after a hard failure of a later
# member: a `LINESEARCH_FAILED` step that the members before it moved is not a zero step. Members
# whose steps cancel mark the outer step `shortened`, as `cancelled_substeps` says.
function step!!(
        state::MultiplicativeState, m::Multiplicative,
        prob::AbstractNonlinearProblem, x, Fx, p)
    R = real(eltype(x))
    x_start = copy!!(state.x₀, x)
    x_out, Fx_out, states, info, substeps, longest = _step_multiplicative_tail(
        state.states, m.members, prob, x, Fx, p, zero(R), Int32(0), false, false, false, true,
        Int32(0), zero(R))
    stepnorm = rnorm(map!!(-, state.x₀, x_out, x_start))
    info_out = StepInfo{R}(
        info.code, info.fnorm, stepnorm, info.failures, info.promoted, info.limited,
        info.shortened || cancelled_substeps(substeps, stepnorm, longest))
    return x_out, Fx_out, MultiplicativeState(states, state.Fx, state.x₀), info_out
end
