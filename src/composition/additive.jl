# The `Additive` combinator: steps `m` members independently from `(x, Fx)` and combines
# their candidate directions with `OptimalWeights` or `FixedWeights`.

"""
    Additive(m₁, m₂, ...; weights = OptimalWeights())
    Additive((m₁, m₂, ...), weights = OptimalWeights())

Parallel nonlinear combinator that runs one step of each member from the same iterate
`(x, F(x))` to obtain candidates ``(x_i, F(x_i))`` and forms
``x_+ = x + \\sum_{i=1}^m w_i (x_i - x)`` using [`OptimalWeights`](@ref) or
[`FixedWeights`](@ref). Under `OptimalWeights`, a rank-deficient Gram matrix or a trial
iterate that violates the PETSc safeguard falls back to the candidate with the smallest
residual norm and increments `step_failures`.
"""
const WeightRule = Union{OptimalWeights, FixedWeights}

struct Additive{M <: Tuple, W <: WeightRule}
    members::M
    weights::W
    function Additive{M, W}(members::M, weights::W) where {M <: Tuple, W <: WeightRule}
        K = fieldcount(M)
        K >= 1 || throw(ArgumentError("Additive requires at least one member"))
        if weights isa FixedWeights && length(weights.weights) != K
            throw(ArgumentError(
                "FixedWeights length $(length(weights.weights)) does not match $K members"))
        end
        return new{M, W}(members, weights)
    end
end

function Additive(members::Tuple, weights::WeightRule = OptimalWeights())
    return Additive{typeof(members), typeof(weights)}(members, weights)
end

function Additive(members...; weights::WeightRule = OptimalWeights())
    return Additive(members, weights)
end

Adapt.@adapt_structure Additive

# State of `Additive`: member states, residual buffer, per-member R3 candidate buffers,
# combined direction and trial-iterate buffers, and the R3 Gram workspace.
struct AdditiveState{S <: Tuple, FX, XS, FXS, D, XT, W}
    states::S
    Fx::FX
    xs::XS
    Fxs::FXS
    d::D
    xt::XT
    work::W
end

@inline _init_member_states(::Tuple{}, prob, x, p) = ()
@inline function _init_member_states(members::Tuple, prob, x, p)
    s1 = init_state(first(members), prob, x, p)
    s_tail = _init_member_states(Base.tail(members), prob, x, p)
    return (s1, s_tail...)
end

@inline _start_member_states(::Tuple{}, ::Tuple{}, prob, x, Fx, p, fnorm₀) = ()
@inline function _start_member_states(states::Tuple, members::Tuple, prob, x, Fx, p, fnorm₀)
    s1 = start!!(first(states), first(members), prob, x, Fx, p, fnorm₀)
    s_tail = _start_member_states(
        Base.tail(states), Base.tail(members), prob, x, Fx, p, fnorm₀)
    return (s1, s_tail...)
end

@inline _finish_member_states(::Tuple{}, ::Tuple{}, prob, x, Fx, status) = ()
@inline function _finish_member_states(states::Tuple, members::Tuple, prob, x, Fx, status)
    _, _, s1 = finish!!(first(states), first(members), prob, x, Fx, status)
    s_tail = _finish_member_states(
        Base.tail(states), Base.tail(members), prob, x, Fx, status)
    return (s1, s_tail...)
end

@inline function init_state(m::Additive, prob::NonlinearProblem, x, p)
    M = length(m.members)
    states = _init_member_states(m.members, prob, x, p)
    Fx = buffer(x)
    xs = Fx === nothing ? nothing : ntuple(_ -> buffer(x), Val(M))
    Fxs = Fx === nothing ? nothing : ntuple(_ -> buffer(x), Val(M))
    d = buffer(x)
    xt = buffer(x)
    work = init_weight_workspace(m.weights, x, Val(M))
    return AdditiveState(states, Fx, xs, Fxs, d, xt, work)
end

@inline residual_buffer(state::AdditiveState) = state.Fx

@inline function start!!(state::AdditiveState, m::Additive, prob, x, Fx, p, fnorm₀)
    states = _start_member_states(state.states, m.members, prob, x, Fx, p, fnorm₀)
    return AdditiveState(
        states, state.Fx, state.xs, state.Fxs, state.d, state.xt, state.work)
end

@inline function finish!!(
        state::AdditiveState, m::Additive, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    states = _finish_member_states(state.states, m.members, prob, x, Fx, status)
    st = SolverStatus{R}(
        status.code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, Int32(-1), R(NaN), R(NaN))
    new_state = AdditiveState(
        states, state.Fx, state.xs, state.Fxs, state.d, state.xt, state.work)
    return x, st, new_state
end

@inline function is_hard_failure(info::StepInfo)
    return info.code == SINGULAR || info.code == NONFINITE ||
           (info.code == LINESEARCH_FAILED && iszero(info.stepnorm))
end

@inline function first_failure_code(infos::NTuple{M, StepInfo{R}}) where {M, R}
    @inbounds for i in 1:M
        is_hard_failure(infos[i]) && return infos[i].code
    end
    return SUCCESS
end

@inline function sum_failures(infos::NTuple{M, StepInfo{R}}) where {M, R}
    s = Int32(0)
    @inbounds for i in 1:M
        s += infos[i].failures
    end
    return s
end

@inline function any_promoted(infos::NTuple{M, StepInfo{R}}) where {M, R}
    @inbounds for i in 1:M
        infos[i].promoted && return true
    end
    return false
end

@inline function any_limited(infos::NTuple{M, StepInfo{R}}) where {M, R}
    @inbounds for i in 1:M
        infos[i].limited && return true
    end
    return false
end

@inline function best_candidate_index(fnorms::NTuple{M, R}) where {M, R <: Real}
    k_best = 1
    f_best = fnorms[1]
    @inbounds for i in 2:M
        fi = fnorms[i]
        if (isfinite(fi) && !isfinite(f_best)) || (fi < f_best)
            k_best = i
            f_best = fi
        end
    end
    return k_best, f_best
end

@inline _step_additive_r1(::Tuple{}, ::Tuple{}, prob, x, Fx, p, merit₀) = (), (), (), ()
@inline function _step_additive_r1(states::Tuple, members::Tuple, prob, x, Fx, p, merit₀)
    s1 = sync_merit(first(states), first(members), merit₀)
    x1, F1, s1_new, info1 = step!!(s1, first(members), prob, x, Fx, p)
    xs_t, Fs_t, ss_t, infos_t = _step_additive_r1(
        Base.tail(states), Base.tail(members), prob, x, Fx, p, merit₀)
    return (x1, xs_t...), (F1, Fs_t...), (s1_new, ss_t...), (info1, infos_t...)
end

@inline function _step_additive_r3(
        ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, prob, x, Fx, p, merit₀)
    return (), ()
end
@inline function _step_additive_r3(
        states::Tuple, members::Tuple, xs::Tuple, Fxs::Tuple, prob, x, Fx, p, merit₀)
    x1_buf = copy!!(first(xs), x)
    F1_buf = copy!!(first(Fxs), Fx)
    s1 = sync_merit(first(states), first(members), merit₀)
    x1, F1, s1_new, info1 = step!!(s1, first(members), prob, x1_buf, F1_buf, p)
    copy!!(first(xs), x1)
    copy!!(first(Fxs), F1)
    ss_t, infos_t = _step_additive_r3(
        Base.tail(states), Base.tail(members), Base.tail(xs), Base.tail(Fxs),
        prob, x, Fx, p, merit₀)
    return (s1_new, ss_t...), (info1, infos_t...)
end

@inline _combine_r1_dir(x, ::Tuple{}, ::Tuple{}) = zero(x)
@inline _combine_r1_dir(x, xs::Tuple{Any}, w::Tuple{Any}) = first(w) * (first(xs) - x)
@inline function _combine_r1_dir(x, xs::Tuple, w::Tuple)
    return first(w) * (first(xs) - x) + _combine_r1_dir(x, Base.tail(xs), Base.tail(w))
end

@inline function combine_directions!(
        d::AbstractArray{T}, x::AbstractArray{T},
        xs::NTuple{1, AbstractArray{T}}, w::NTuple{1, R}) where {T, R <: Real}
    w1 = w[1]
    x1 = xs[1]
    @. d = w1 * (x1 - x)
    return d
end

@inline function combine_directions!(
        d::AbstractArray{T}, x::AbstractArray{T},
        xs::NTuple{2, AbstractArray{T}}, w::NTuple{2, R}) where {T, R <: Real}
    w1, w2 = w
    x1, x2 = xs
    @. d = w1 * (x1 - x) + w2 * (x2 - x)
    return d
end

@inline function combine_directions!(
        d::AbstractArray{T}, x::AbstractArray{T},
        xs::NTuple{3, AbstractArray{T}}, w::NTuple{3, R}) where {T, R <: Real}
    w1, w2, w3 = w
    x1, x2, x3 = xs
    @. d = w1 * (x1 - x) + w2 * (x2 - x) + w3 * (x3 - x)
    return d
end

@inline function combine_directions!(
        d::AbstractArray{T}, x::AbstractArray{T},
        xs::NTuple{4, AbstractArray{T}}, w::NTuple{4, R}) where {T, R <: Real}
    w1, w2, w3, w4 = w
    x1, x2, x3, x4 = xs
    @. d = w1 * (x1 - x) + w2 * (x2 - x) + w3 * (x3 - x) + w4 * (x4 - x)
    return d
end

@inline function combine_directions!(
        d::AbstractArray{T}, x::AbstractArray{T},
        xs::NTuple{M, AbstractArray{T}}, w::NTuple{M, R}) where {M, T, R <: Real}
    w1 = w[1]
    x1 = xs[1]
    @. d = w1 * (x1 - x)
    @inbounds for i in 2:M
        wi = w[i]
        xi = xs[i]
        @. d = d + wi * (xi - x)
    end
    return d
end

function step!!(
        state::AdditiveState{S, Nothing}, m::Additive, prob::NonlinearProblem,
        x, Fx, p) where {S}
    R = real(eltype(x))
    merit₀ = norm2(Fx)
    xs_cand, Fxs_cand, states_new, infos = _step_additive_r1(
        state.states, m.members, prob, x, Fx, p, merit₀)
    new_state = AdditiveState(
        states_new, state.Fx, state.xs, state.Fxs, state.d, state.xt, state.work)
    if any(is_hard_failure, infos)
        bad_code = first_failure_code(infos)
        info_err = StepInfo{R}(
            bad_code, rnorm(Fx), zero(R), sum_failures(infos), any_promoted(infos), false)
        return x, Fx, new_state, info_err
    end
    fnorms = map(info -> info.fnorm, infos)
    if m.weights isa FixedWeights
        w = map(R, m.weights.weights)
        rank_fallback = false
    else
        w, rank_fallback = optimal_weights(nothing, Fx, Fxs_cand, fnorms)
    end
    d = _combine_r1_dir(x, xs_cand, w)
    x₊ = x + d
    F₊ = residual!!(nothing, prob, x₊, p)
    fnorm₊ = rnorm(F₊)
    safeguard_fallback = false
    if m.weights isa OptimalWeights
        k_best, f_best = best_candidate_index(fnorms)
        w_l1 = sum(abs, w)
        ρ = R(m.weights.ρ)
        safeguard_fallback = !isfinite(fnorm₊) || w_l1 < R(0.1) || fnorm₊ > ρ * f_best
        if safeguard_fallback
            x₊ = xs_cand[k_best]
            F₊ = Fxs_cand[k_best]
            fnorm₊ = f_best
        end
    end
    stepnorm = rnorm(x₊ - x)
    failures = sum_failures(infos) + Int32(rank_fallback) + Int32(safeguard_fallback)
    info_ok = StepInfo{R}(
        SUCCESS, fnorm₊, stepnorm, failures, any_promoted(infos), any_limited(infos))
    return x₊, F₊, new_state, info_ok
end

function step!!(
        state::AdditiveState, m::Additive, prob::NonlinearProblem, x, Fx, p)
    R = real(eltype(x))
    merit₀ = norm2(Fx)
    states_new, infos = _step_additive_r3(
        state.states, m.members, state.xs, state.Fxs, prob, x, Fx, p, merit₀)
    new_state = AdditiveState(
        states_new, state.Fx, state.xs, state.Fxs, state.d, state.xt, state.work)
    if any(is_hard_failure, infos)
        bad_code = first_failure_code(infos)
        info_err = StepInfo{R}(
            bad_code, rnorm(Fx), zero(R), sum_failures(infos), any_promoted(infos), false)
        return x, Fx, new_state, info_err
    end
    fnorms = map(info -> info.fnorm, infos)
    if m.weights isa FixedWeights
        w = map(R, m.weights.weights)
        rank_fallback = false
    else
        w, rank_fallback = optimal_weights(state.work, Fx, state.Fxs, fnorms)
    end
    combine_directions!(state.d, x, state.xs, w)
    x₊ = update_iterate!!(state.xt, x, one(R), state.d)
    F₊ = residual!!(state.Fx, prob, x₊, p)
    fnorm₊ = rnorm(F₊)
    safeguard_fallback = false
    if m.weights isa OptimalWeights
        k_best, f_best = best_candidate_index(fnorms)
        w_l1 = sum(abs, w)
        ρ = R(m.weights.ρ)
        safeguard_fallback = !isfinite(fnorm₊) || w_l1 < R(0.1) || fnorm₊ > ρ * f_best
        if safeguard_fallback
            x₊ = state.xs[k_best]
            F₊ = state.Fxs[k_best]
            fnorm₊ = f_best
        end
    end
    @. state.d = x₊ - x
    stepnorm = rnorm(state.d)
    x_out = copy!!(x, x₊)
    Fx_out = copy!!(Fx, F₊)
    failures = sum_failures(infos) + Int32(rank_fallback) + Int32(safeguard_fallback)
    info_ok = StepInfo{R}(
        SUCCESS, fnorm₊, stepnorm, failures, any_promoted(infos), any_limited(infos))
    return x_out, Fx_out, new_state, info_ok
end
