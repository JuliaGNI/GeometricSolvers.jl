# The `Switch` combinator: steps `first` until `‖F(x)‖ <= threshold * ‖F(x₀)‖`, then latches
# permanently to `second` for the remainder of the solve.

"""
    Switch(first, second; threshold = 1e-2)
    Switch(first, second, threshold)

Two-stage nonlinear combinator that runs `first` while
``\\|F(x)\\| > \\text{threshold} \\cdot \\|F(x_0)\\|`` and latches permanently to `second`
for the remainder of the solve once
``\\|F(x)\\| \\le \\text{threshold} \\cdot \\|F(x_0)\\|``.
Calling `step!!` on a state that has not been initialised by `start!!`
(`!isfinite(state.fnorm₀)`) fails closed with `NONFINITE` and a zero step norm.
"""
struct Switch{M, N, R <: Real}
    first::M
    second::N
    threshold::R
    # `adapt` rebuilds a `Switch` through this constructor, and a device kernel cannot format a
    # message, so the message is constant.
    function Switch{M, N, R}(first::M, second::N, threshold::R) where {M, N, R <: Real}
        isfinite(threshold) && threshold > zero(R) ||
            throw(ArgumentError("Switch threshold must be finite and positive"))
        return new{M, N, R}(first, second, threshold)
    end
end

function Switch(first::M, second::N, threshold::Real) where {M, N}
    r = float(threshold)
    return Switch{M, N, typeof(r)}(first, second, r)
end

function Switch(first, second; threshold::Real = 1e-2)
    return Switch(first, second, threshold)
end

Adapt.@adapt_structure Switch

# State of `Switch`: member states `s1` and `s2`, residual buffer `Fx`, initial residual
# norm `fnorm₀` (`NaN` before `start!!` and after `finish!!`), and latch `switched`.
struct SwitchState{S1, S2, FX, R <: Real}
    s1::S1
    s2::S2
    Fx::FX
    fnorm₀::R
    switched::Bool
end

@inline function sync_merit(state::SwitchState, m::Switch, merit::Real)
    s1 = sync_merit(state.s1, m.first, merit)
    s2 = sync_merit(state.s2, m.second, merit)
    return SwitchState(s1, s2, state.Fx, state.fnorm₀, state.switched)
end

@inline function init_state(m::Switch, prob::CombinatorProblem, x, p)
    R = real(eltype(x))
    s1 = init_state(m.first, prob, x, p)
    s2 = init_state(m.second, prob, x, p)
    return SwitchState(s1, s2, buffer(x), R(NaN), false)
end

@inline residual_buffer(state::SwitchState) = state.Fx

@inline function start!!(state::SwitchState, m::Switch, prob, x, Fx, p, fnorm₀)
    R = typeof(state.fnorm₀)
    s1 = start!!(state.s1, m.first, prob, x, Fx, p, fnorm₀)
    s2 = start!!(state.s2, m.second, prob, x, Fx, p, fnorm₀)
    return SwitchState(s1, s2, state.Fx, R(fnorm₀), false)
end

@inline function finish!!(
        state::SwitchState, m::Switch, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    _, st1, s1 = finish!!(state.s1, m.first, prob, x, Fx, status)
    _, st2, s2 = finish!!(state.s2, m.second, prob, x, Fx, status)
    # each member counts only its own steps, so `Switch` keeps no count
    st_out = with_calls(state.switched ? st2 : st1, 0, 0)
    return x, st_out, SwitchState(s1, s2, state.Fx, R(NaN), state.switched)
end

function step!!(
        state::SwitchState{S1, S2, FX, R}, m::Switch,
        prob::AbstractNonlinearProblem, x, Fx, p) where {S1, S2, FX, R <: Real}
    fnorm = rnorm(Fx)
    if !isfinite(state.fnorm₀)
        return x, Fx, state,
        StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
    end
    switched = state.switched || (fnorm <= R(m.threshold) * state.fnorm₀)
    merit = norm2(Fx)
    if switched
        s2 = sync_merit(state.s2, m.second, merit)
        x₊, F₊, s2_new, info = step!!(s2, m.second, prob, x, Fx, p)
        return x₊, F₊, SwitchState(state.s1, s2_new, state.Fx, state.fnorm₀, true), info
    else
        s1 = sync_merit(state.s1, m.first, merit)
        x₊, F₊, s1_new, info = step!!(s1, m.first, prob, x, Fx, p)
        return x₊, F₊, SwitchState(s1_new, state.s2, state.Fx, state.fnorm₀, false), info
    end
end
