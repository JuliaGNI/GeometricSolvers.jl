# The `Multiplicative` combinator and the `sync_merit` seam that keeps line-search members'
# merit baselines `φ(x) = ‖F(x)‖₂²` synchronised when another member moves `(x, Fx)`.

"""
    Multiplicative(m₁, m₂, ...)
    Multiplicative((m₁, m₂, ...))

Sequential nonlinear combinator (composite step ``G_m \\circ \\cdots \\circ G_1``) that
threads `(x, F(x))` through `m₁, m₂, ...` in order within a single outer solver iteration.
Before each member steps, [`sync_merit`](@ref) updates its line-search baseline merit to the
incoming ``\\|F(x)\\|_2^2`` when the member state has been started.
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
without `start!!` still fails closed with `NONFINITE`, and is a no-op for `Static` and
`DogLeg` states (`merit === nothing`).
"""
@inline sync_merit(state, method, merit::Real) = state

@inline function sync_merit(state::NewtonState, ::Newton, merit::Real)
    state.merit isa Real && isfinite(state.merit) || return state
    return with_merit(state, oftype(state.merit, merit))
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

@inline function init_state(m::Multiplicative, prob::NonlinearProblem, x, p)
    states = _init_member_states(m.members, prob, x, p)
    return MultiplicativeState(states, buffer(x), buffer(x))
end

@inline residual_buffer(state::MultiplicativeState) = state.Fx

@inline function start!!(
        state::MultiplicativeState, m::Multiplicative, prob, x, Fx, p, fnorm₀)
    states = _start_member_states(state.states, m.members, prob, x, Fx, p, fnorm₀)
    return MultiplicativeState(states, state.Fx, state.x₀)
end

@inline function has_diagnostics(st::SolverStatus)
    return st.rank != Int32(-1) || isfinite(st.nullnorm) || isfinite(st.gradratio)
end

@inline function _finish_multiplicative_states(
        ::Tuple{}, ::Tuple{}, prob, x, Fx, status::SolverStatus{R},
        st_diag::SolverStatus{R}) where {R}
    return (), st_diag
end

@inline function _finish_multiplicative_states(
        states::Tuple, members::Tuple, prob, x, Fx, status::SolverStatus{R},
        st_diag::SolverStatus{R}) where {R}
    _, st1, s1 = finish!!(first(states), first(members), prob, x, Fx, status)
    st_next = has_diagnostics(st1) ? st1 : st_diag
    ss_tail, st_final = _finish_multiplicative_states(
        Base.tail(states), Base.tail(members), prob, x, Fx, status, st_next)
    return (s1, ss_tail...), st_final
end

@inline function finish!!(
        state::MultiplicativeState, m::Multiplicative, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    st_init = SolverStatus{R}(
        status.code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, Int32(-1), R(NaN), R(NaN))
    states, st_out = _finish_multiplicative_states(
        state.states, m.members, prob, x, Fx, status, st_init)
    return x, st_out, MultiplicativeState(states, state.Fx, state.x₀)
end

@inline function _step_multiplicative_tail(
        ::Tuple{}, ::Tuple{}, prob::NonlinearProblem, x, Fx, p,
        fnorm::R, failures::Int32, promoted::Bool, limited::Bool) where {R}
    info = StepInfo{R}(SUCCESS, fnorm, zero(R), failures, promoted, limited)
    return x, Fx, (), info, false
end

@inline function _step_multiplicative_tail(
        states::Tuple, members::Tuple, prob::NonlinearProblem, x, Fx, p,
        fnorm::R, failures::Int32, promoted::Bool, limited::Bool) where {R}
    s_i = sync_merit(first(states), first(members), norm2(Fx))
    x_next, Fx_next, s_i_new, info_i = step!!(s_i, first(members), prob, x, Fx, p)
    failures_next = failures + info_i.failures
    promoted_next = promoted || info_i.promoted
    limited_next = limited || info_i.limited
    if is_hard_failure(info_i)
        states_sc = (s_i_new, Base.tail(states)...)
        info_sc = StepInfo{R}(
            info_i.code, info_i.fnorm, zero(R), failures_next, promoted_next, limited_next)
        return x_next, Fx_next, states_sc, info_sc, true
    end
    x_out, Fx_out, ss_tail, info_out, sc = _step_multiplicative_tail(
        Base.tail(states), Base.tail(members), prob, x_next, Fx_next, p,
        info_i.fnorm, failures_next, promoted_next, limited_next)
    return x_out, Fx_out, (s_i_new, ss_tail...), info_out, sc
end

function step!!(
        state::MultiplicativeState{S, Nothing}, m::Multiplicative,
        prob::NonlinearProblem, x, Fx, p) where {S}
    R = real(eltype(x))
    x_start = x
    s1 = sync_merit(first(state.states), first(m.members), norm2(Fx))
    x1, Fx1, s1_new, info1 = step!!(s1, first(m.members), prob, x, Fx, p)
    if is_hard_failure(info1)
        states_sc = (s1_new, Base.tail(state.states)...)
        new_sc = MultiplicativeState(states_sc, state.Fx, state.x₀)
        info_sc = StepInfo{R}(
            info1.code, info1.fnorm, zero(R), info1.failures, info1.promoted, info1.limited)
        return x1, Fx1, new_sc, info_sc
    end
    x_out, Fx_out, ss_tail, info_tail, sc = _step_multiplicative_tail(
        Base.tail(state.states), Base.tail(m.members), prob, x1, Fx1, p,
        info1.fnorm, info1.failures, info1.promoted, info1.limited)
    states_new = (s1_new, ss_tail...)
    new_state = MultiplicativeState(states_new, state.Fx, state.x₀)
    sc && return x_out, Fx_out, new_state, info_tail
    stepnorm = rnorm(x_out - x_start)
    info_ok = StepInfo{R}(
        SUCCESS, info_tail.fnorm, stepnorm, info_tail.failures,
        info_tail.promoted, info_tail.limited)
    return x_out, Fx_out, new_state, info_ok
end

function step!!(
        state::MultiplicativeState, m::Multiplicative,
        prob::NonlinearProblem, x, Fx, p)
    R = real(eltype(x))
    copy!!(state.x₀, x)
    s1 = sync_merit(first(state.states), first(m.members), norm2(Fx))
    x1, Fx1, s1_new, info1 = step!!(s1, first(m.members), prob, x, Fx, p)
    if is_hard_failure(info1)
        states_sc = (s1_new, Base.tail(state.states)...)
        new_sc = MultiplicativeState(states_sc, state.Fx, state.x₀)
        info_sc = StepInfo{R}(
            info1.code, info1.fnorm, zero(R), info1.failures, info1.promoted, info1.limited)
        return x1, Fx1, new_sc, info_sc
    end
    x_out, Fx_out, ss_tail, info_tail, sc = _step_multiplicative_tail(
        Base.tail(state.states), Base.tail(m.members), prob, x1, Fx1, p,
        info1.fnorm, info1.failures, info1.promoted, info1.limited)
    states_new = (s1_new, ss_tail...)
    new_state = MultiplicativeState(states_new, state.Fx, state.x₀)
    sc && return x_out, Fx_out, new_state, info_tail
    @. state.x₀ = x_out - state.x₀
    stepnorm = rnorm(state.x₀)
    info_ok = StepInfo{R}(
        SUCCESS, info_tail.fnorm, stepnorm, info_tail.failures,
        info_tail.promoted, info_tail.limited)
    return x_out, Fx_out, new_state, info_ok
end
