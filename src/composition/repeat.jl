# The `Repeat` combinator: runs `count` sub-steps of a single member per outer iteration.

"""
    Repeat(method, count)

Sequential nonlinear combinator that runs `count` sub-steps of `method` (`count >= 1`) per
outer solver iteration, synchronising any line-search baseline merit between sub-steps and
reporting the combined outer step norm ``\\|x^{(\\text{count})} - x^{(0)}\\|``.
"""
struct Repeat{M}
    method::M
    count::Int32
    function Repeat{M}(method::M, count::Integer) where {M}
        count >= 1 ||
            throw(ArgumentError("Repeat requires count >= 1, got $count"))
        return new{M}(method, Int32(count))
    end
end

Repeat(method::M, count::Integer) where {M} = Repeat{M}(method, count)

Adapt.@adapt_structure Repeat

# State of `Repeat`: inner member state, residual buffer, and starting-iterate buffer `x₀`
# used to measure the outer step norm in R3.
struct RepeatState{S, FX, X0}
    state::S
    Fx::FX
    x₀::X0
end

@inline function sync_merit(state::RepeatState, m::Repeat, merit::Real)
    return RepeatState(sync_merit(state.state, m.method, merit), state.Fx, state.x₀)
end

@inline function init_state(m::Repeat, prob::NonlinearProblem, x, p)
    return RepeatState(init_state(m.method, prob, x, p), buffer(x), buffer(x))
end

@inline residual_buffer(state::RepeatState) = state.Fx

@inline function start!!(state::RepeatState, m::Repeat, prob, x, Fx, p, fnorm₀)
    return RepeatState(
        start!!(state.state, m.method, prob, x, Fx, p, fnorm₀), state.Fx, state.x₀)
end

@inline function finish!!(
        state::RepeatState, m::Repeat, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    x_out, st_out, s_inner = finish!!(state.state, m.method, prob, x, Fx, status)
    return x_out, st_out, RepeatState(s_inner, state.Fx, state.x₀)
end

function step!!(
        state::RepeatState{S, Nothing}, m::Repeat,
        prob::NonlinearProblem, x, Fx, p) where {S}
    R = real(eltype(x))
    x_start = x
    s = state.state
    fnorm = zero(R)
    failures = Int32(0)
    promoted = false
    limited = false
    for _ in Base.OneTo(m.count)
        s = sync_merit(s, m.method, norm2(Fx))
        x, Fx, s, info = step!!(s, m.method, prob, x, Fx, p)
        fnorm = info.fnorm
        failures += info.failures
        promoted |= info.promoted
        limited |= info.limited
        if is_hard_failure(info)
            info_err = StepInfo{R}(info.code, fnorm, zero(R), failures, promoted, limited)
            return x, Fx, RepeatState(s, state.Fx, state.x₀), info_err
        end
    end
    stepnorm = rnorm(x - x_start)
    info_ok = StepInfo{R}(SUCCESS, fnorm, stepnorm, failures, promoted, limited)
    return x, Fx, RepeatState(s, state.Fx, state.x₀), info_ok
end

function step!!(
        state::RepeatState, m::Repeat,
        prob::NonlinearProblem, x, Fx, p)
    R = real(eltype(x))
    copy!!(state.x₀, x)
    s = state.state
    fnorm = zero(R)
    failures = Int32(0)
    promoted = false
    limited = false
    for _ in Base.OneTo(m.count)
        s = sync_merit(s, m.method, norm2(Fx))
        x, Fx, s, info = step!!(s, m.method, prob, x, Fx, p)
        fnorm = info.fnorm
        failures += info.failures
        promoted |= info.promoted
        limited |= info.limited
        if is_hard_failure(info)
            info_err = StepInfo{R}(info.code, fnorm, zero(R), failures, promoted, limited)
            return x, Fx, RepeatState(s, state.Fx, state.x₀), info_err
        end
    end
    @. state.x₀ = x - state.x₀
    stepnorm = rnorm(state.x₀)
    info_ok = StepInfo{R}(SUCCESS, fnorm, stepnorm, failures, promoted, limited)
    return x, Fx, RepeatState(s, state.Fx, state.x₀), info_ok
end
