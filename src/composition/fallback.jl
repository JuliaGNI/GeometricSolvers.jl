# The `Fallback` combinator: runs the whole solve of each member in turn from `x₀`, and stops at
# the first member that returns `SUCCESS`.

"""
    Fallback(m₁, m₂, ...)
    Fallback((m₁, m₂, ...))

Whole-solve combinator that runs [`solve!!`](@ref) with each member in turn and stops at the
first member that returns `SUCCESS`. Every member starts from `x₀`, with the full
[`Options`](@ref) of the solve and its own state, which [`start!!`](@ref) resets as for a solve
with the member alone; `solve!!` copies an array `x₀` into the state before the first member and
back into the iterate before each later one, so each member's result is the result of its own
solve from `x₀`. Every code but `SUCCESS` moves on to the next member, and an exception that a
member throws is not caught.

The [`SolverStatus`](@ref) is that of the member that succeeded, or of the last member when none
did, with `iterations` and `step_failures` summed over the members that ran (each saturated at
`typemax(Int32)`), `promoted` set when any of them set it, and the call counts `f_calls` and
`jvp_calls` `0`. The state that `solve!!` returns holds the index of that member, which
`GeometricSolvers.fallback_member` reads.

`Fallback` needs at least one member, and raises an `ArgumentError` with none. It has no
[`step!!`](@ref), so it is a method of its own and not a member of a step-level combinator such
as [`Switch`](@ref). Like every combinator, it refuses a `ForwardDiff.Dual` in `x₀` or `p` in
[`solve`](@ref) and a [`LeastSquaresProblem`](@ref) in `init_state`.
"""
struct Fallback{M <: Tuple}
    members::M
    # `adapt` rebuilds a `Fallback` through this constructor, and a device kernel cannot format a
    # message, so the message is constant.
    function Fallback{M}(members::M) where {M <: Tuple}
        fieldcount(M) >= 1 || throw(ArgumentError("Fallback needs at least one member"))
        return new{M}(members)
    end
end

Fallback(members::M) where {M <: Tuple} = Fallback{M}(members)
Fallback(members...) = Fallback(members)

Adapt.@adapt_structure Fallback

# State of `Fallback`: the member states, the copy of an array `x₀` that each later member starts
# from (`nothing` for a value iterate), and the index of the member whose result the last solve
# returned (`0` before the first solve).
struct FallbackState{S <: Tuple, X}
    states::S
    x₀::X
    member::Int32
end

"""
    fallback_member(state::FallbackState)

The index of the member of a [`Fallback`](@ref) whose result the last [`solve!!`](@ref) on `state`
returned: the member that succeeded, or the last member when none did; `0` for a state that no
solve returned.
"""
fallback_member(state::FallbackState) = state.member

@inline function init_state(m::Fallback, prob::CombinatorProblem, x, p)
    return FallbackState(_init_member_states(m.members, prob, x, p), buffer(x), Int32(0))
end

# `a + b` for two non-negative counts, saturated at `typemax(Int32)`.
@inline _sat_add32(a::Int32, b::Int32) = Int32(min(Int64(a) + Int64(b), typemax(Int32)))

# The status of member `st`, with the counts of the members before it added, and no call counts.
@inline function _fallback_status(
        st::SolverStatus{R}, iterations::Int32, failures::Int32, promoted::Bool) where {R}
    return SolverStatus{R}(
        st.code, Int16(0), _sat_add32(st.iterations, iterations), st.fnorm, st.stepnorm,
        _sat_add32(st.step_failures, failures), st.promoted | promoted, st.stationary,
        Int16(0), st.rank, st.nullnorm, st.gradratio)
end

# Solve with the first member from `x`; on a code other than `SUCCESS`, restore `x₀` from `saved`
# into `x` and go on with the rest. Returns the iterate, the status, the member states and the
# index of the member whose result it returns.
@inline function _fallback_solve!!(states::Tuple{Any}, members::Tuple{Any}, prob, x, saved,
        p, options, k::Int32, iterations::Int32, failures::Int32, promoted::Bool)
    x₊, st, s = solve!!(only(states), prob, x, only(members), p, options)
    return x₊, _fallback_status(st, iterations, failures, promoted), (s,), k
end
@inline function _fallback_solve!!(states::Tuple, members::Tuple, prob, x, saved,
        p, options, k::Int32, iterations::Int32, failures::Int32, promoted::Bool)
    x₊, st, s = solve!!(first(states), prob, x, first(members), p, options)
    if st.code == SUCCESS
        status = _fallback_status(st, iterations, failures, promoted)
        return x₊, status, (s, Base.tail(states)...), k
    end
    x_end, st_end, rest, k_end = _fallback_solve!!(
        Base.tail(states), Base.tail(members), prob, copy!!(x, saved), saved, p, options,
        k + one(Int32), _sat_add32(iterations, st.iterations),
        _sat_add32(failures, st.step_failures), promoted | st.promoted)
    return x_end, st_end, (s, rest...), k_end
end

# `prob` is a `CombinatorProblem`, as in `init_state`, so that this method is not ambiguous with
# `solve!!(state, ::LeastSquaresProblem, x₀::Number, method, …)`.
function solve!!(
        state::FallbackState, prob::CombinatorProblem, x₀, method::Fallback, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    saved = copy!!(state.x₀, x₀)
    x, status, states, k = _fallback_solve!!(
        state.states, method.members, prob, x₀, saved, p, options,
        Int32(1), Int32(0), Int32(0), false)
    return x, status, FallbackState(states, state.x₀, k)
end
