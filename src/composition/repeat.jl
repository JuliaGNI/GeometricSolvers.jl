# The `Repeat` combinator: runs `count` sub-steps of a single member per outer iteration.

"""
    Repeat(method, count)

Sequential nonlinear combinator that runs `count` sub-steps of `method` (`count >= 1`) per
outer solver iteration, synchronising any line-search baseline merit between sub-steps and
reporting the combined outer step norm ``\\|x^{(\\text{count})} - x^{(0)}\\|``. The outer step is
`STALLED` when every sub-step is.
"""
struct Repeat{M}
    method::M
    count::Int32
    # `adapt` rebuilds a `Repeat` through this constructor, and a device kernel cannot format a
    # message, so the message is constant.
    function Repeat{M}(method::M, count::Integer) where {M}
        count >= 1 ||
            throw(ArgumentError("Repeat requires count >= 1"))
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

@inline function init_state(m::Repeat, prob::CombinatorProblem, x, p)
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

# A hard failure of a sub-step ends the outer step with its code and keeps the iterate that the
# sub-steps before it reached. The step norm is the outer one, also after such a failure: a
# `LINESEARCH_FAILED` step that earlier sub-steps moved is not a zero step. Otherwise the outer
# step is `STALLED` when every sub-step is, and `SUCCESS` else, so `Repeat(m, 1)` stalls where
# `m` does.
function step!!(
        state::RepeatState, m::Repeat,
        prob::AbstractNonlinearProblem, x, Fx, p)
    R = real(eltype(x))
    x_start = copy!!(state.x₀, x)
    s = state.state
    stalled = true
    code = SUCCESS
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
        stalled &= info.code == STALLED
        if is_hard_failure(info)
            code = info.code
            break
        end
    end
    stalled && (code = STALLED)
    stepnorm = rnorm(map!!(-, state.x₀, x, x_start))
    info_out = StepInfo{R}(code, fnorm, stepnorm, failures, promoted, limited)
    return x, Fx, RepeatState(s, state.Fx, state.x₀), info_out
end
