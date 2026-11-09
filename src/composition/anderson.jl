# Type-II Anderson acceleration over an inner nonlinear method, with the length and angle
# history filters of Pollock and Rebholz (2022), precision-scaled Tikhonov regularisation,
# affine-weight and model-decrease safeguards, residual-growth and stagnation restarts,
# optional growing depth, and best-iterate tracking.

"""
    Anderson(member = Picard(), ::Val{m} = Val(4);
             c_s = 0.2, κ_bar = nothing, restart_factor = 10.0,
             M_alpha = 25.0, safeguard_cos = nothing, grow_depth = false)
    Anderson(member, m::Integer; kwargs...)

Type-II Anderson acceleration of history depth `m` wrapping an inner nonlinear method
`member` ([`Picard`](@ref), [`Newton`](@ref), or a combinator). At each outer iteration,
`Anderson` runs one step of `member` from `(x, Fx)` to obtain the candidate `x_m` and
correction `w_k = x_m - x`, updates the history of consecutive iterate differences
`Δx_j` and correction differences `Δw_j`, filters the history by the length and angle
filters of Pollock and Rebholz (2022) from the Gram matrix `G_{ij} = Re⟨Δw_i, Δw_j⟩`,
solves the `eps(R)`-regularised normal equations `(G_K + λI) γ = b_K` with
`b_i = Re⟨Δw_i, w_k⟩`, and forms the accelerated iterate
`x₊ = x_m - ∑_j γ_j (Δx_{K[j]} + Δw_{K[j]})`.

In R1 (`StaticVector` or `Number`), `m` must satisfy `1 <= m <= 4`, and the state is
`isbits` for an `SVector` or `Number` iterate. In R3 (`Array` or `AbstractGPUArray`), any
`m >= 1` is supported with preallocated `n × m` difference matrices, and `step!!` on an
`Array` allocates nothing.
When a solve ends with a code other than `SUCCESS`, [`finish!!`](@ref) restores the
iterate with the smallest residual norm `‖F(x)‖` seen during the solve. A step is `STALLED`
when the step of `member` is.

Keywords:

  - `c_s`: angle-filter threshold in `(0, 1)`; a history column is kept only if its
    component orthogonal to the kept columns has squared norm `≥ c_s² G_jj`.
  - `κ_bar`: bound of the length filter; `nothing` selects `1 / sqrt(eps(R))` for the
    real element type `R` of the iterate.
  - `restart_factor` (`≥ 3`): the history is cleared when `‖F‖` exceeds
    `restart_factor` times the smallest residual norm seen.
  - `M_alpha` (`≥ 1`): bound on `∑_j |α_j|` for the affine weights `α_j` of the member
    candidates `g_{k-j}` that `x₊` combines, also after a filter has dropped a history
    column; a step whose weights exceed it, are not finite, or increase the residual of the
    linear model takes the step of `member` and counts one failure.
  - `safeguard_cos`: when set, the history is cleared if the cosine between consecutive
    corrections `w_k` exceeds it in modulus.
  - `grow_depth`: when `true`, the depth is capped at `ceil(-log10(‖w_k‖))` clamped to
    `1:m`, so the cap grows as the corrections shrink.
"""
struct Anderson{m, M, R <: Real, K <: Union{Nothing, R}, S <: Union{Nothing, R}}
    member::M
    c_s::R
    κ_bar::K
    restart_factor::R
    M_alpha::R
    safeguard_cos::S
    grow_depth::Bool

    # `adapt` rebuilds an `Anderson` through the keyword constructor and this one, and a device
    # kernel cannot format a message, so the messages of both are constant.
    function Anderson{m, M, R, K, S}(
            member::M, c_s::R, κ_bar::K, restart_factor::R,
            M_alpha::R, safeguard_cos::S,
            grow_depth::Bool) where {
            m, M, R <: Real, K <: Union{Nothing, R}, S <: Union{Nothing, R}}
        m isa Int ||
            throw(ArgumentError("Anderson history depth m must be an Int"))
        m >= 1 ||
            throw(ArgumentError("Anderson history depth m must be >= 1"))
        isfinite(c_s) && zero(R) < c_s < one(R) ||
            throw(ArgumentError("Anderson c_s must be finite and in (0, 1)"))
        if κ_bar !== nothing
            isfinite(κ_bar) && κ_bar > one(R) ||
                throw(ArgumentError("Anderson κ_bar must be nothing or finite and > 1"))
        end
        isfinite(restart_factor) && restart_factor >= R(3) ||
            throw(ArgumentError("Anderson restart_factor must be finite and >= 3"))
        isfinite(M_alpha) && M_alpha >= one(R) ||
            throw(ArgumentError("Anderson M_alpha must be finite and >= 1"))
        if safeguard_cos !== nothing
            isfinite(safeguard_cos) && zero(R) < safeguard_cos < one(R) ||
                throw(ArgumentError(
                    "Anderson safeguard_cos must be nothing or finite and in (0, 1)"))
        end
        return new{m, M, R, K, S}(
            member, c_s, κ_bar, restart_factor, M_alpha, safeguard_cos, grow_depth)
    end
end

function Anderson(
        member::M = Picard(), ::Val{m} = Val(4);
        c_s::Real = 0.2,
        κ_bar::Union{Nothing, Real} = nothing,
        restart_factor::Real = 10.0,
        M_alpha::Real = 25.0,
        safeguard_cos::Union{Nothing, Real} = nothing,
        grow_depth::Bool = false) where {m, M}
    m isa Int ||
        throw(ArgumentError("Anderson history depth m must be an Int"))
    R_base = promote_type(typeof(c_s), typeof(restart_factor), typeof(M_alpha))
    R_kb = κ_bar === nothing ? R_base : promote_type(R_base, typeof(κ_bar))
    R = float(safeguard_cos === nothing ? R_kb : promote_type(R_kb, typeof(safeguard_cos)))
    kb = κ_bar === nothing ? nothing : R(κ_bar)
    sc = safeguard_cos === nothing ? nothing : R(safeguard_cos)
    return Anderson{m, M, R, typeof(kb), typeof(sc)}(
        member, R(c_s), kb, R(restart_factor), R(M_alpha), sc, grow_depth)
end

function Anderson(member, m::Integer; kwargs...)
    m isa Bool && throw(ArgumentError("Anderson history depth m must be an Int, got Bool"))
    return Anderson(member, Val(Int(m)); kwargs...)
end

function Adapt.adapt_structure(to, a::Anderson{m}) where {m}
    member = Adapt.adapt(to, a.member)
    c_s = Adapt.adapt(to, a.c_s)
    κ_bar = Adapt.adapt(to, a.κ_bar)
    restart_factor = Adapt.adapt(to, a.restart_factor)
    M_alpha = Adapt.adapt(to, a.M_alpha)
    safeguard_cos = Adapt.adapt(to, a.safeguard_cos)
    return Anderson(
        member, Val(m);
        c_s = c_s, κ_bar = κ_bar, restart_factor = restart_factor,
        M_alpha = M_alpha, safeguard_cos = safeguard_cos, grow_depth = a.grow_depth)
end

@inline function effective_kappa_bar(a::Anderson, ::Type{R}) where {R <: Real}
    return a.κ_bar === nothing ? inv(sqrt(eps(R))) : R(a.κ_bar)
end

# Preallocated R3 buffers for `Anderson`: candidate and difference buffers on the iterate's
# device, `m × m` and `m` Gram buffers on the device for `mul!!`, and host workspace arrays
# for the `m × m` filters and regularised Cholesky solve.
struct AndersonWorkspace{V, M, T, R <: Real}
    x_m::V
    F_m::V
    w_k::V
    x_trial::V
    F_trial::V
    Δx_new::V
    Δw_new::V
    G_dev::M
    b_dev::V
    γ_dev::V
    G_raw::Matrix{T}
    b_raw::Vector{T}
    G::Matrix{R}
    b::Vector{R}
    γ::Vector{R}
    γ_full::Vector{T}
    B::Vector{R}
    L::Matrix{R}
    y::Vector{R}
    kept::Vector{Int}
    col_idx::Vector{Int}
    col_tmp::Vector{Int}
end

function init_anderson_workspace(x::AbstractVector{T}, ::Val{m}) where {T, m}
    R = real(T)
    x_m = similar(x)
    F_m = similar(x)
    w_k = similar(x)
    x_trial = similar(x)
    F_trial = similar(x)
    Δx_new = similar(x)
    Δw_new = similar(x)
    G_dev = similar(x, T, m, m)
    b_dev = similar(x, T, m)
    γ_dev = similar(x, T, m)
    fill!(G_dev, zero(T))
    fill!(b_dev, zero(T))
    fill!(γ_dev, zero(T))
    return AndersonWorkspace(
        x_m, F_m, w_k, x_trial, F_trial, Δx_new, Δw_new,
        G_dev, b_dev, γ_dev,
        zeros(T, m, m), zeros(T, m),
        zeros(R, m, m), zeros(R, m), zeros(R, m), zeros(T, m),
        zeros(R, m), zeros(R, m, m), zeros(R, m),
        zeros(Int, m), collect(1:m), collect(1:m))
end

# State of `Anderson{m}`: `isbits` in R1 (`work`, `Fx` and `Fx_best` are `nothing`, so the
# residual may have another type than the iterate) and preallocated in R3
# (`work isa AndersonWorkspace`). `ages[c]` is the age of the history column `c`: the column of
# age `a` is the difference `g_{k-a+1} - g_{k-a}` of the member candidates `g`.
struct AndersonState{m, MS, FX, X, FB, DX, DW, A, W, R <: Real}
    member_state::MS
    Fx::FX
    x_prev::X
    w_prev::X
    x_best::X
    Fx_best::FB
    ΔX::DX
    ΔW::DW
    ages::A
    work::W
    fnorm₀::R
    fnorm_best::R
    depth::Int32
    current_depth::Int32
    has_prev::Bool
end

function AndersonState{m}(
        member_state, Fx, x_prev::X, w_prev::X, x_best::X, Fx_best, ΔX, ΔW, ages, work,
        fnorm₀::R, fnorm_best::R, depth::Int32, current_depth::Int32,
        has_prev::Bool) where {m, X, R <: Real}
    return AndersonState{m, typeof(member_state), typeof(Fx), X, typeof(Fx_best),
        typeof(ΔX), typeof(ΔW), typeof(ages), typeof(work), R}(
        member_state, Fx, x_prev, w_prev, x_best, Fx_best, ΔX, ΔW, ages, work,
        fnorm₀, fnorm_best, depth, current_depth, has_prev)
end

@inline residual_buffer(state::AndersonState) = state.Fx

@inline function sync_merit(state::AndersonState{m}, a::Anderson{m}, merit::Real) where {m}
    ms = sync_merit(state.member_state, a.member, merit)
    return AndersonState{m}(
        ms, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
        state.ΔX, state.ΔW, state.ages, state.work, state.fnorm₀, state.fnorm_best,
        state.depth, state.current_depth, state.has_prev)
end

function init_state(
        a::Anderson{m}, prob::CombinatorProblem,
        x::Union{StaticVector, Number}, p) where {m}
    m <= 4 ||
        throw(ArgumentError("R1 Anderson history depth m must be <= 4, got $m"))
    R = real(eltype(x))
    ms = init_state(a.member, prob, x, p)
    z = zero(x)
    ΔX = ntuple(_ -> z, Val(m))
    ΔW = ntuple(_ -> z, Val(m))
    ages = ntuple(_ -> Int32(0), Val(m))
    return AndersonState{m}(
        ms, nothing, z, z, z, nothing, ΔX, ΔW, ages, nothing,
        R(NaN), R(NaN), Int32(0), Int32(0), false)
end

function init_state(
        a::Anderson{m}, prob::CombinatorProblem,
        x::AbstractVector{T}, p) where {m, T}
    R = real(T)
    n = length(x)
    ms = init_state(a.member, prob, x, p)
    Fx = similar(x)
    x_prev = similar(x)
    w_prev = similar(x)
    x_best = similar(x)
    Fx_best = similar(x)
    ΔX = similar(x, T, n, m)
    ΔW = similar(x, T, n, m)
    fill!(ΔX, zero(T))
    fill!(ΔW, zero(T))
    ages = zeros(Int32, m)
    work = init_anderson_workspace(x, Val(m))
    return AndersonState{m}(
        ms, Fx, x_prev, w_prev, x_best, Fx_best, ΔX, ΔW, ages, work,
        R(NaN), R(NaN), Int32(0), Int32(0), false)
end

@inline function start!!(
        state::AndersonState{m, MS, Nothing, X, Nothing, DX, DW, A, Nothing, R},
        a::Anderson{m}, prob, x, Fx, p, fnorm₀) where {m, MS, X, DX, DW, A, R <: Real}
    ms = start!!(state.member_state, a.member, prob, x, Fx, p, fnorm₀)
    z = zero(x)
    f0 = R(fnorm₀)
    return AndersonState{m}(
        ms, nothing, z, z, x, nothing, state.ΔX, state.ΔW, state.ages, nothing,
        f0, f0, Int32(0), Int32(0), false)
end

@inline function start!!(
        state::AndersonState{m, MS, FX, X, FB, DX, DW, A, W, R},
        a::Anderson{m}, prob, x, Fx, p,
        fnorm₀) where {
        m, MS, FX, X, FB, DX, DW, A, W <: AndersonWorkspace, R <: Real}
    ms = start!!(state.member_state, a.member, prob, x, Fx, p, fnorm₀)
    T = eltype(x)
    copyto!(state.x_best, x)
    copyto!(state.Fx_best, Fx)
    fill!(state.ΔX, zero(T))
    fill!(state.ΔW, zero(T))
    fill!(state.ages, Int32(0))
    @inbounds for j in 1:m
        state.work.col_idx[j] = j
    end
    f0 = R(fnorm₀)
    return AndersonState{m}(
        ms, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
        state.ΔX, state.ΔW, state.ages, state.work, f0, f0, Int32(0), Int32(0), false)
end

@inline function should_restore_best(status::SolverStatus{R}, fnorm_best::R) where {R}
    return status.code != SUCCESS && isfinite(fnorm_best) && fnorm_best < status.fnorm
end

@inline function restored_best_status(status::SolverStatus{R}, fnorm_best::R) where {R}
    return SolverStatus{R}(
        status.code, status.iterations, fnorm_best, status.stepnorm,
        status.step_failures, status.promoted, status.rank, status.nullnorm, status.gradratio)
end

@inline function finish!!(
        state::AndersonState{m, MS, Nothing, X, Nothing, DX, DW, A, Nothing, R},
        a::Anderson{m}, prob, x, Fx, status::SolverStatus{R}) where {
        m, MS, X, DX, DW, A, R <: Real}
    x_m, st_m, ms = finish!!(state.member_state, a.member, prob, x, Fx, status)
    if should_restore_best(st_m, state.fnorm_best)
        x_out = state.x_best
        st_out = restored_best_status(st_m, state.fnorm_best)
    else
        x_out = x_m
        st_out = st_m
    end
    state_out = AndersonState{m}(
        ms, nothing, state.x_prev, state.w_prev, state.x_best, nothing,
        state.ΔX, state.ΔW, state.ages, nothing,
        R(NaN), state.fnorm_best, state.depth, state.current_depth, state.has_prev)
    return x_out, st_out, state_out
end

@inline function finish!!(
        state::AndersonState{m, MS, FX, X, FB, DX, DW, A, W, R},
        a::Anderson{m}, prob,
        x,
        Fx,
        status::SolverStatus{R}) where {
        m, MS, FX, X, FB, DX, DW, A, W <: AndersonWorkspace, R <: Real}
    x_m, st_m, ms = finish!!(state.member_state, a.member, prob, x, Fx, status)
    if should_restore_best(st_m, state.fnorm_best)
        copyto!(x_m, state.x_best)
        copyto!(Fx, state.Fx_best)
        st_out = restored_best_status(st_m, state.fnorm_best)
    else
        st_out = st_m
    end
    state_out = AndersonState{m}(
        ms, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
        state.ΔX, state.ΔW, state.ages, state.work,
        R(NaN), state.fnorm_best, state.depth, state.current_depth, state.has_prev)
    return x_m, st_out, state_out
end

# Shared scalar predicates for the restarts, filters, regularisation, and safeguards.
@inline is_unstarted(state::AndersonState) = !isfinite(state.fnorm₀)

@inline function effective_depth_cap(a::Anderson{m}, wnorm::R) where {m, R <: Real}
    a.grow_depth || return m
    (!isfinite(wnorm) || wnorm <= zero(R)) && return 1
    d_grow = ceil(Int, -log10(max(wnorm, floatmin(R))))
    return clamp(d_grow, 1, m)
end

@inline function is_degenerate_newest(
        Δw_norm::R, f2_1::R, Δx_norm::R, x_norm::R) where {R <: Real}
    return !isfinite(Δw_norm) || iszero(Δw_norm) || !isfinite(f2_1) ||
           f2_1 <= floatmin(R) || !isfinite(Δx_norm) ||
           Δx_norm <= eps(R) * max(one(R), x_norm)
end

@inline function should_restart_growth(
        fnorm::R, restart_factor::R, fnorm_best::R) where {R <: Real}
    return !isfinite(fnorm) || fnorm > restart_factor * fnorm_best
end

@inline function should_restart_angle(
        a::Anderson, w_dot::R, wnorm::R, wprev_norm::R) where {R <: Real}
    a.safeguard_cos === nothing && return false
    (wnorm > zero(R) && wprev_norm > zero(R)) || return false
    return abs(w_dot) / (wnorm * wprev_norm) > R(a.safeguard_cos)
end

@inline function length_filter_accepts(C_F::R, κ_max2::R) where {R <: Real}
    return C_F <= κ_max2
end

@inline function angle_filter_accepts(r_jj2::R, c_s2::R, G_jj::R) where {R <: Real}
    return r_jj2 >= c_s2 * G_jj
end

@inline function tikhonov_lambda(tr_G::R, p::Int) where {R <: Real}
    return eps(R) * (tr_G / R(p))
end

@inline function filter_failures(inner_failures::Int32, d::Int, p::Int)
    return inner_failures + Int32(d - p)
end

@inline function restart_current_depth()
    return Int32(1)
end

@inline function accelerated_current_depth(p::Int, m::Int)
    return min(Int32(p + 1), Int32(m))
end

# The code of an accelerated step: `STALLED` when the member step is, as for the other
# combinators, and `SUCCESS` else.
@inline accelerated_code(info_m::StepInfo) = info_m.code == STALLED ? STALLED : SUCCESS

# Regularised Cholesky solve of `(G_K + λI) γ = b_K` in R1 (`SMatrix` / `SVector`).
@inline function solve_weights(
        G::SMatrix{m, m, R}, b::SVector{m, R},
        kept::SVector{m, Int}, p::Int) where {m, R <: Real}
    tr_G = zero(R)
    @inbounds for c in 1:p
        kc = kept[c]
        tr_G += G[kc, kc]
    end
    λ = tikhonov_lambda(tr_G, p)
    L = zero(SMatrix{m, m, R, m * m})
    y = zero(SVector{m, R})
    @inbounds for c in 1:p
        kc = kept[c]
        for r in 1:(c - 1)
            kr = kept[r]
            s = G[kc, kr]
            for q in 1:(r - 1)
                s -= L[c, q] * L[r, q]
            end
            L = setindex(L, s / L[r, r], c + (r - 1) * m)
        end
        s_diag = G[kc, kc] + λ
        for q in 1:(c - 1)
            s_diag -= L[c, q]^2
        end
        L = setindex(L, sqrt(max(s_diag, zero(R))), c + (c - 1) * m)
    end
    @inbounds for c in 1:p
        kc = kept[c]
        s = b[kc]
        for r in 1:(c - 1)
            s -= L[c, r] * y[r]
        end
        y = setindex(y, s / L[c, c], c)
    end
    γ = zero(SVector{m, R})
    @inbounds for c in p:-1:1
        s = y[c]
        for r in (c + 1):p
            s -= L[r, c] * γ[r]
        end
        γ = setindex(γ, s / L[c, c], c)
    end
    return γ
end

# Regularised Cholesky solve of `(G_K + λI) γ = b_K` in R3 (preallocated host buffers).
function solve_weights!(
        γ::Vector{R}, L::Matrix{R}, y::Vector{R},
        G::AbstractMatrix{R}, b::AbstractVector{R},
        kept::Vector{Int}, p::Int) where {R <: Real}
    tr_G = zero(R)
    @inbounds for c in 1:p
        kc = kept[c]
        tr_G += G[kc, kc]
    end
    λ = tikhonov_lambda(tr_G, p)
    fill!(γ, zero(R))
    @inbounds for c in 1:p
        kc = kept[c]
        for r in 1:(c - 1)
            kr = kept[r]
            s = G[kc, kr]
            for q in 1:(r - 1)
                s -= L[c, q] * L[r, q]
            end
            L[c, r] = s / L[r, r]
        end
        s_diag = G[kc, kc] + λ
        for q in 1:(c - 1)
            s_diag -= L[c, q]^2
        end
        L[c, c] = sqrt(max(s_diag, zero(R)))
    end
    @inbounds for c in 1:p
        kc = kept[c]
        s = b[kc]
        for r in 1:(c - 1)
            s -= L[c, r] * y[r]
        end
        y[c] = s / L[c, c]
    end
    @inbounds for c in p:-1:1
        s = y[c]
        for r in (c + 1):p
            s -= L[r, c] * γ[r]
        end
        γ[c] = s / L[c, c]
    end
    return γ
end

# `∑_j |α_j|` for the affine weights `α_j` of the member candidates `g_{k-j}` in
# `x₊ = g_k - ∑_c γ_c (g_{k-a_c+1} - g_{k-a_c})`, with `a_c = ages[kept[c]]` increasing in `c`.
# Two kept columns of consecutive ages share a candidate, whose weight is then `γ_c - γ_{c+1}`.
@inline function affine_weight_sum(γ, kept, ages, p::Int, ::Type{R}) where {R <: Real}
    s = zero(R)
    j = 0          # the index of the candidate whose weight `α_j` is still open
    α_j = one(R)
    @inbounds for c in 1:p
        a = Int(ages[kept[c]])
        γ_c = R(γ[c])
        if a - 1 == j
            s += abs(α_j - γ_c)
        else
            s += abs(α_j) + abs(γ_c)
        end
        j = a
        α_j = γ_c
    end
    return s + abs(α_j)
end

# Check `all(isfinite, γ)`, `∑_j |α_j| <= M_alpha`, and `Δmodel <= eps(R) * wnorm2`.
@inline function weights_acceptable(
        G, b, γ, kept, ages, p::Int, M_alpha::R, wnorm2::R) where {R <: Real}
    @inbounds for c in 1:p
        isfinite(γ[c]) || return false
    end
    affine_weight_sum(γ, kept, ages, p, R) <= M_alpha || return false

    Δmodel = zero(R)
    @inbounds for c in 1:p
        kc = kept[c]
        γ_c = γ[c]
        g_γ = zero(R)
        for r in 1:p
            g_γ += G[kc, kept[r]] * γ[r]
        end
        Δmodel += γ_c * g_γ - R(2) * b[kc] * γ_c
    end
    return Δmodel <= eps(R) * max(wnorm2, floatmin(R))
end

# Pollock & Rebholz (2022) Algorithm 2.5 (length filter, Prop. 1.2 eqs. 1.6–1.8) followed by
# Algorithm 2.4 (greedy Cholesky angle filter) in R1 (`SMatrix` / `SVector`).
@inline function filter_history_r1(
        a::Anderson{m}, G::SMatrix{m, m, R}, d::Int) where {m, R <: Real}
    c_s = R(a.c_s)
    c_s2 = c_s^2
    c_t2 = one(R) - c_s2
    c_t = sqrt(max(c_t2, zero(R)))
    ρ = ((c_t + c_s) / c_s)^2
    κ_max = effective_kappa_bar(a, R)
    κ_max2 = κ_max^2

    B = zero(SVector{m, R})
    f2_1 = max(G[1, 1], floatmin(R))
    B = setindex(B, inv(f2_1), 1)
    @inbounds for j in 2:d
        f2_j = max(G[j, j], floatmin(R))
        s = c_t2 * (ρ^(j - 2)) / f2_1 + inv(f2_j)
        for i in 2:(j - 1)
            f2_i = max(G[i, i], floatmin(R))
            s += c_t2 * (ρ^(j - i - 1)) / (c_s2 * f2_i)
        end
        B = setindex(B, s / c_s2, j)
    end

    d_len = 1
    @inbounds for k in d:-1:1
        sum_f2 = zero(R)
        sum_B = zero(R)
        for j in 1:k
            sum_f2 += G[j, j]
            sum_B += B[j]
        end
        C_F = sum_f2 * sum_B
        if length_filter_accepts(C_F, κ_max2)
            d_len = k
            break
        end
    end

    kept = zero(SVector{m, Int})
    kept = setindex(kept, 1, 1)
    L = zero(SMatrix{m, m, R, m * m})
    L = setindex(L, sqrt(f2_1), 1)
    p = 1
    @inbounds for j in 2:d_len
        G_jj = G[j, j]
        sum_l2 = zero(R)
        for c in 1:p
            kc = kept[c]
            s = G[j, kc]
            for r in 1:(c - 1)
                s -= L[p + 1, r] * L[c, r]
            end
            l_c = s / L[c, c]
            L = setindex(L, l_c, (p + 1) + (c - 1) * m)
            sum_l2 += l_c^2
        end
        r_jj2 = G_jj - sum_l2
        if angle_filter_accepts(r_jj2, c_s2, G_jj)
            p += 1
            kept = setindex(kept, j, p)
            L = setindex(L, sqrt(max(r_jj2, floatmin(R))), p + (p - 1) * m)
        end
    end
    return kept, p
end

# Pollock & Rebholz (2022) length and angle filters in R3 (preallocated host buffers).
function filter_history_r3!(
        kept::Vector{Int}, B::Vector{R}, L::Matrix{R},
        a::Anderson{m}, G::Matrix{R}, d::Int) where {m, R <: Real}
    c_s = R(a.c_s)
    c_s2 = c_s^2
    c_t2 = one(R) - c_s2
    c_t = sqrt(max(c_t2, zero(R)))
    ρ = ((c_t + c_s) / c_s)^2
    κ_max = effective_kappa_bar(a, R)
    κ_max2 = κ_max^2

    f2_1 = max(G[1, 1], floatmin(R))
    @inbounds B[1] = inv(f2_1)
    @inbounds for j in 2:d
        f2_j = max(G[j, j], floatmin(R))
        s = c_t2 * (ρ^(j - 2)) / f2_1 + inv(f2_j)
        for i in 2:(j - 1)
            f2_i = max(G[i, i], floatmin(R))
            s += c_t2 * (ρ^(j - i - 1)) / (c_s2 * f2_i)
        end
        B[j] = s / c_s2
    end

    d_len = 1
    @inbounds for k in d:-1:1
        sum_f2 = zero(R)
        sum_B = zero(R)
        for j in 1:k
            sum_f2 += G[j, j]
            sum_B += B[j]
        end
        C_F = sum_f2 * sum_B
        if length_filter_accepts(C_F, κ_max2)
            d_len = k
            break
        end
    end

    fill!(kept, 0)
    @inbounds kept[1] = 1
    @inbounds L[1, 1] = sqrt(f2_1)
    p = 1
    @inbounds for j in 2:d_len
        G_jj = G[j, j]
        sum_l2 = zero(R)
        for c in 1:p
            kc = kept[c]
            s = G[j, kc]
            for r in 1:(c - 1)
                s -= L[p + 1, r] * L[c, r]
            end
            l_c = s / L[c, c]
            L[p + 1, c] = l_c
            sum_l2 += l_c^2
        end
        r_jj2 = G_jj - sum_l2
        if angle_filter_accepts(r_jj2, c_s2, G_jj)
            p += 1
            kept[p] = j
            L[p, p] = sqrt(max(r_jj2, floatmin(R)))
        end
    end
    return p
end

# R1 `step!!` (`StaticVector` or `Number`). The residual `Fx` may have another type than the
# iterate `x`, such as an `SVector` residual of an `MVector` iterate.
function step!!(
        state::AndersonState{m, MS, Nothing, X, Nothing, DX, DW, A, Nothing, R},
        a::Anderson{m}, prob::AbstractNonlinearProblem, x::X, Fx,
        p) where {
        m, MS, X <: Union{StaticVector, Number}, DX, DW, A, R <: Real}
    fnorm_in = rnorm(Fx)
    if is_unstarted(state)
        return x, Fx, state,
        StepInfo{R}(NONFINITE, fnorm_in, zero(R), Int32(0), false, false)
    end

    x_best = state.x_best
    fnorm_best = state.fnorm_best
    if isfinite(fnorm_in) && fnorm_in < fnorm_best
        x_best = x
        fnorm_best = fnorm_in
    end

    ms_sync = sync_merit(state.member_state, a.member, norm2(Fx))
    x_m, F_m, ms_new, info_m = step!!(ms_sync, a.member, prob, x, Fx, p)
    if is_hard_failure(info_m)
        st_err = AndersonState{m}(
            ms_new, nothing, state.x_prev, state.w_prev, x_best, nothing,
            state.ΔX, state.ΔW, state.ages, nothing,
            state.fnorm₀, fnorm_best, state.depth, state.current_depth, state.has_prev)
        return x, Fx,
        st_err,
        StepInfo{R}(
            info_m.code, fnorm_in, zero(R), info_m.failures, info_m.promoted, false)
    end

    w_k = x_m - x
    fnorm_m = info_m.fnorm

    # Step 1 (`!state.has_prev`): accept the member step and record `(x, w_k)`.
    if !state.has_prev
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            x_best = x_m
            fnorm_best = fnorm_m
        end
        st_step1 = AndersonState{m}(
            ms_new, nothing, x, w_k, x_best, nothing,
            state.ΔX, state.ΔW, state.ages, nothing,
            state.fnorm₀, fnorm_best, Int32(0), restart_current_depth(), true)
        return x_m, F_m, st_step1, info_m
    end

    Δx_new = x - state.x_prev
    Δw_new = w_k - state.w_prev
    f2_1 = norm2(Δw_new)
    Δw_norm = rnorm(Δw_new)
    Δx_norm = rnorm(Δx_new)
    wnorm = rnorm(w_k)

    # Pre-filter restart checks: incoming residual growth, a zero or stagnated newest
    # column, or the optional `safeguard_cos` bound on the cosine between consecutive
    # corrections.
    if should_restart_growth(fnorm_in, R(a.restart_factor), fnorm_best) ||
       is_degenerate_newest(Δw_norm, f2_1, Δx_norm, rnorm(x)) ||
       should_restart_angle(a, rdot(w_k, state.w_prev), wnorm, rnorm(state.w_prev))
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            x_best = x_m
            fnorm_best = fnorm_m
        end
        ΔX_zero = ntuple(_ -> zero(x), Val(m))
        ΔW_zero = ntuple(_ -> zero(x), Val(m))
        st_rst = AndersonState{m}(
            ms_new, nothing, x, w_k, x_best, nothing,
            ΔX_zero, ΔW_zero, ntuple(_ -> Int32(0), Val(m)), nothing,
            state.fnorm₀, fnorm_best, Int32(0), restart_current_depth(), true)
        info_rst = StepInfo{R}(
            info_m.code, fnorm_m, info_m.stepnorm,
            info_m.failures + one(Int32), info_m.promoted, info_m.limited)
        return x_m, F_m, st_rst, info_rst
    end

    # Prepend `(Δx_new, Δw_new)` at index 1, of age 1, up to `m_cap`.
    m_cap = effective_depth_cap(a, wnorm)
    d = min(Int(state.depth) + 1, m_cap)
    ΔX_cand = ntuple(j -> j == 1 ? Δx_new : state.ΔX[j - 1], Val(m))
    ΔW_cand = ntuple(j -> j == 1 ? Δw_new : state.ΔW[j - 1], Val(m))
    ages_cand = ntuple(j -> j == 1 ? Int32(1) : state.ages[j - 1] + Int32(1), Val(m))

    G = zero(SMatrix{m, m, R, m * m})
    b = zero(SVector{m, R})
    @inbounds for j in 1:d
        dw_j = ΔW_cand[j]
        b = setindex(b, rdot(dw_j, w_k), j)
        for i in 1:j
            g_ij = i == j ? norm2(dw_j) : rdot(ΔW_cand[i], dw_j)
            G = setindex(G, g_ij, i + (j - 1) * m)
            if i != j
                G = setindex(G, g_ij, j + (i - 1) * m)
            end
        end
    end

    kept, p_kept = filter_history_r1(a, G, d)
    failures = filter_failures(info_m.failures, d, p_kept)
    ΔX_kept = ntuple(c -> c <= p_kept ? ΔX_cand[kept[c]] : zero(x), Val(m))
    ΔW_kept = ntuple(c -> c <= p_kept ? ΔW_cand[kept[c]] : zero(x), Val(m))
    ages_kept = ntuple(c -> c <= p_kept ? ages_cand[kept[c]] : Int32(0), Val(m))
    cur_depth = accelerated_current_depth(p_kept, m)

    γ = solve_weights(G, b, kept, p_kept)
    if !weights_acceptable(G, b, γ, kept, ages_cand, p_kept, R(a.M_alpha), norm2(w_k))
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            x_best = x_m
            fnorm_best = fnorm_m
        end
        st_fb = AndersonState{m}(
            ms_new, nothing, x, w_k, x_best, nothing,
            ΔX_kept, ΔW_kept, ages_kept, nothing,
            state.fnorm₀, fnorm_best, Int32(p_kept), cur_depth, true)
        info_fb = StepInfo{R}(
            info_m.code, fnorm_m, info_m.stepnorm,
            failures + one(Int32), info_m.promoted, info_m.limited)
        return x_m, F_m, st_fb, info_fb
    end

    # Form accelerated iterate `x₊ = x_m - ∑_{c=1}^p γ_c (Δx_{K[c]} + Δw_{K[c]})`.
    d_acc = zero(x)
    @inbounds for c in 1:p_kept
        d_acc = d_acc + γ[c] * (ΔX_kept[c] + ΔW_kept[c])
    end
    x₊ = x_m - d_acc
    F₊ = primal_residual!!(nothing, prob, x₊, p)
    fnorm₊ = rnorm(F₊)

    # Post-step residual-growth safeguard against `fnorm_best`.
    if should_restart_growth(fnorm₊, R(a.restart_factor), fnorm_best)
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            x_best = x_m
            fnorm_best = fnorm_m
        end
        ΔX_zero = ntuple(_ -> zero(x), Val(m))
        ΔW_zero = ntuple(_ -> zero(x), Val(m))
        st_post = AndersonState{m}(
            ms_new, nothing, x, w_k, x_best, nothing,
            ΔX_zero, ΔW_zero, ntuple(_ -> Int32(0), Val(m)), nothing,
            state.fnorm₀, fnorm_best, Int32(0), restart_current_depth(), true)
        info_post = StepInfo{R}(
            info_m.code, fnorm_m, info_m.stepnorm,
            info_m.failures + one(Int32), info_m.promoted, info_m.limited)
        return x_m, F_m, st_post, info_post
    end

    if fnorm₊ < fnorm_best
        x_best = x₊
        fnorm_best = fnorm₊
    end
    stepnorm = rnorm(x₊ - x)
    st_ok = AndersonState{m}(
        ms_new, nothing, x, w_k, x_best, nothing,
        ΔX_kept, ΔW_kept, ages_kept, nothing,
        state.fnorm₀, fnorm_best, Int32(p_kept), cur_depth, true)
    info_ok = StepInfo{R}(
        accelerated_code(info_m), fnorm₊, stepnorm, failures, info_m.promoted,
        info_m.limited)
    return x₊, F₊, st_ok, info_ok
end

# R3 helper methods to write/zero a column of `ΔX, ΔW` and combine columns without
# allocations on `Array` or scalar indexing on `AbstractGPUArray`.
@inline function _write_history_col!(A::Matrix{T}, col::Int, v::Vector{T}) where {T}
    @inbounds for i in axes(A, 1)
        A[i, col] = v[i]
    end
    return A
end

@inline function _write_history_col!(
        A::AbstractMatrix{T}, col::Int, v::AbstractVector{T}) where {T}
    @views copyto!(A[:, col], v)
    return A
end

@inline function _zero_history_col!(A::Matrix{T}, col::Int) where {T}
    @inbounds for i in axes(A, 1)
        A[i, col] = zero(T)
    end
    return A
end

@inline function _zero_history_col!(A::AbstractMatrix{T}, col::Int) where {T}
    @views fill!(A[:, col], zero(T))
    return A
end

@inline function _combine_anderson_r3!(
        work::AndersonWorkspace{Vector{T}, Matrix{T}, T, R},
        ΔX::Matrix{T}, ΔW::Matrix{T}, p_kept::Int) where {T, R <: Real}
    x_trial = work.x_trial
    x_m = work.x_m
    copyto!(x_trial, x_m)
    @inbounds for c in 1:p_kept
        phys_c = work.col_idx[c]
        γ_c = T(work.γ[c])
        for i in eachindex(x_trial)
            x_trial[i] -= γ_c * (ΔX[i, phys_c] + ΔW[i, phys_c])
        end
    end
    return x_trial
end

@inline function _combine_anderson_r3!(
        work::AndersonWorkspace{V, M, T, R},
        ΔX::M, ΔW::M,
        p_kept::Int) where {T, R <: Real, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    fill!(work.γ_full, zero(T))
    @inbounds for c in 1:p_kept
        work.γ_full[work.col_idx[c]] = T(work.γ[c])
    end
    copyto!(work.γ_dev, work.γ_full)
    mul!!(work.Δx_new, ΔX, work.γ_dev)
    mul!!(work.Δw_new, ΔW, work.γ_dev)
    @. work.x_trial = work.x_m - (work.Δx_new + work.Δw_new)
    return work.x_trial
end

@inline function _reset_r3_history!(
        state::AndersonState{m}, work::AndersonWorkspace) where {m}
    T = eltype(state.ΔX)
    fill!(state.ΔX, zero(T))
    fill!(state.ΔW, zero(T))
    fill!(state.ages, Int32(0))
    @inbounds for j in 1:m
        work.col_idx[j] = j
    end
    return nothing
end

# R3 `step!!` (`Array` or `AbstractGPUArray`).
function step!!(
        state::AndersonState{m, MS, FX, X, FB, DX, DW, A, W, R},
        a::Anderson{m}, prob::AbstractNonlinearProblem,
        x::AbstractVector{T}, Fx::AbstractVector{T},
        p) where {
        m, MS, FX, X, FB, DX, DW, A, W <: AndersonWorkspace, R <: Real, T}
    fnorm_in = rnorm(Fx)
    if is_unstarted(state)
        return x, Fx, state,
        StepInfo{R}(NONFINITE, fnorm_in, zero(R), Int32(0), false, false)
    end

    work = state.work
    fnorm_best = state.fnorm_best
    if isfinite(fnorm_in) && fnorm_in < fnorm_best
        copyto!(state.x_best, x)
        copyto!(state.Fx_best, Fx)
        fnorm_best = fnorm_in
    end

    copyto!(work.x_m, x)
    copyto!(work.F_m, Fx)
    ms_sync = sync_merit(state.member_state, a.member, norm2(Fx))
    x_m, F_m, ms_new, info_m = step!!(ms_sync, a.member, prob, work.x_m, work.F_m, p)
    if is_hard_failure(info_m)
        st_err = AndersonState{m}(
            ms_new, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
            state.ΔX, state.ΔW, state.ages, work,
            state.fnorm₀, fnorm_best, state.depth, state.current_depth, state.has_prev)
        return x, Fx,
        st_err,
        StepInfo{R}(
            info_m.code, fnorm_in, zero(R), info_m.failures, info_m.promoted, false)
    end

    copy!!(work.x_m, x_m)
    copy!!(work.F_m, F_m)
    @. work.w_k = work.x_m - x
    fnorm_m = info_m.fnorm

    # Step 1 (`!state.has_prev`): accept the member step and record `(x, w_k)`.
    if !state.has_prev
        copyto!(state.x_prev, x)
        copyto!(state.w_prev, work.w_k)
        copyto!(x, work.x_m)
        copyto!(Fx, work.F_m)
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            copyto!(state.x_best, x)
            copyto!(state.Fx_best, Fx)
            fnorm_best = fnorm_m
        end
        st_step1 = AndersonState{m}(
            ms_new, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
            state.ΔX, state.ΔW, state.ages, work,
            state.fnorm₀, fnorm_best, Int32(0), restart_current_depth(), true)
        return x, Fx, st_step1, info_m
    end

    @. work.Δx_new = x - state.x_prev
    @. work.Δw_new = work.w_k - state.w_prev
    f2_1 = norm2(work.Δw_new)
    Δw_norm = rnorm(work.Δw_new)
    Δx_norm = rnorm(work.Δx_new)
    wnorm = rnorm(work.w_k)

    # Pre-filter restart checks.
    if should_restart_growth(fnorm_in, R(a.restart_factor), fnorm_best) ||
       is_degenerate_newest(Δw_norm, f2_1, Δx_norm, rnorm(x)) ||
       should_restart_angle(a, rdot(work.w_k, state.w_prev), wnorm, rnorm(state.w_prev))
        copyto!(state.x_prev, x)
        copyto!(state.w_prev, work.w_k)
        copyto!(x, work.x_m)
        copyto!(Fx, work.F_m)
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            copyto!(state.x_best, x)
            copyto!(state.Fx_best, Fx)
            fnorm_best = fnorm_m
        end
        _reset_r3_history!(state, work)
        st_rst = AndersonState{m}(
            ms_new, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
            state.ΔX, state.ΔW, state.ages, work,
            state.fnorm₀, fnorm_best, Int32(0), restart_current_depth(), true)
        info_rst = StepInfo{R}(
            info_m.code, fnorm_m, info_m.stepnorm,
            info_m.failures + one(Int32), info_m.promoted, info_m.limited)
        return x, Fx, st_rst, info_rst
    end

    # Prepend `(Δx_new, Δw_new)` at logical column 1, of age 1, using `work.col_idx`.
    m_cap = effective_depth_cap(a, wnorm)
    d = min(Int(state.depth) + 1, m_cap)
    new_phys = @inbounds work.col_idx[d]
    @inbounds for j in d:-1:2
        work.col_idx[j] = work.col_idx[j - 1]
        state.ages[j] = state.ages[j - 1] + Int32(1)
    end
    @inbounds work.col_idx[1] = new_phys
    @inbounds state.ages[1] = Int32(1)
    @inbounds for j in (d + 1):m
        _zero_history_col!(state.ΔX, work.col_idx[j])
        _zero_history_col!(state.ΔW, work.col_idx[j])
    end
    _write_history_col!(state.ΔX, new_phys, work.Δx_new)
    _write_history_col!(state.ΔW, new_phys, work.Δw_new)

    # Form Gram matrix `G` and right-hand side `b` in one `mul!!` each, then permute on host.
    mul!!(work.G_dev, state.ΔW', state.ΔW)
    mul!!(work.b_dev, state.ΔW', work.w_k)
    copyto!(work.G_raw, work.G_dev)
    copyto!(work.b_raw, work.b_dev)
    fill!(work.G, zero(R))
    fill!(work.b, zero(R))
    @inbounds for j in 1:d
        pj = work.col_idx[j]
        work.b[j] = real(work.b_raw[pj])
        for i in 1:j
            pi = work.col_idx[i]
            g_ij = i == j ? real(work.G_raw[pj, pj]) :
                   R(0.5) * (real(work.G_raw[pi, pj]) + real(work.G_raw[pj, pi]))
            work.G[i, j] = g_ij
            work.G[j, i] = g_ij
        end
    end

    p_kept = filter_history_r3!(work.kept, work.B, work.L, a, work.G, d)
    failures = filter_failures(info_m.failures, d, p_kept)
    cur_depth = accelerated_current_depth(p_kept, m)

    solve_weights!(work.γ, work.L, work.y, work.G, work.b, work.kept, p_kept)
    acceptable = weights_acceptable(
        work.G, work.b, work.γ, work.kept, state.ages, p_kept, R(a.M_alpha),
        norm2(work.w_k))

    # Compact `work.col_idx` and `state.ages` so logical columns `1:p_kept` hold the kept
    # physical columns, and zero out any dropped physical columns in `ΔX, ΔW`. `kept` increases
    # and `kept[c] >= c`, so the ages compact in place.
    @inbounds for j in 1:m
        work.col_tmp[j] = work.col_idx[j]
    end
    @inbounds for c in 1:p_kept
        work.col_idx[c] = work.col_tmp[work.kept[c]]
        state.ages[c] = state.ages[work.kept[c]]
    end
    tail_pos = p_kept + 1
    @inbounds for j in 1:m
        is_kept = false
        for c in 1:p_kept
            if work.kept[c] == j
                is_kept = true
                break
            end
        end
        if !is_kept
            phys_drop = work.col_tmp[j]
            work.col_idx[tail_pos] = phys_drop
            tail_pos += 1
            if j <= d
                _zero_history_col!(state.ΔX, phys_drop)
                _zero_history_col!(state.ΔW, phys_drop)
            end
        end
    end

    if !acceptable
        copyto!(state.x_prev, x)
        copyto!(state.w_prev, work.w_k)
        copyto!(x, work.x_m)
        copyto!(Fx, work.F_m)
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            copyto!(state.x_best, x)
            copyto!(state.Fx_best, Fx)
            fnorm_best = fnorm_m
        end
        st_fb = AndersonState{m}(
            ms_new, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
            state.ΔX, state.ΔW, state.ages, work,
            state.fnorm₀, fnorm_best, Int32(p_kept), cur_depth, true)
        info_fb = StepInfo{R}(
            info_m.code, fnorm_m, info_m.stepnorm,
            failures + one(Int32), info_m.promoted, info_m.limited)
        return x, Fx, st_fb, info_fb
    end

    _combine_anderson_r3!(work, state.ΔX, state.ΔW, p_kept)
    primal_residual!!(work.F_trial, prob, work.x_trial, p)
    fnorm₊ = rnorm(work.F_trial)

    # Post-step residual-growth safeguard against `fnorm_best`.
    if should_restart_growth(fnorm₊, R(a.restart_factor), fnorm_best)
        copyto!(state.x_prev, x)
        copyto!(state.w_prev, work.w_k)
        copyto!(x, work.x_m)
        copyto!(Fx, work.F_m)
        if isfinite(fnorm_m) && fnorm_m < fnorm_best
            copyto!(state.x_best, x)
            copyto!(state.Fx_best, Fx)
            fnorm_best = fnorm_m
        end
        _reset_r3_history!(state, work)
        st_post = AndersonState{m}(
            ms_new, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
            state.ΔX, state.ΔW, state.ages, work,
            state.fnorm₀, fnorm_best, Int32(0), restart_current_depth(), true)
        info_post = StepInfo{R}(
            info_m.code, fnorm_m, info_m.stepnorm,
            info_m.failures + one(Int32), info_m.promoted, info_m.limited)
        return x, Fx, st_post, info_post
    end

    @. work.Δx_new = work.x_trial - x
    stepnorm = rnorm(work.Δx_new)
    copyto!(state.x_prev, x)
    copyto!(state.w_prev, work.w_k)
    copyto!(x, work.x_trial)
    copyto!(Fx, work.F_trial)
    if fnorm₊ < fnorm_best
        copyto!(state.x_best, x)
        copyto!(state.Fx_best, Fx)
        fnorm_best = fnorm₊
    end
    st_ok = AndersonState{m}(
        ms_new, state.Fx, state.x_prev, state.w_prev, state.x_best, state.Fx_best,
        state.ΔX, state.ΔW, state.ages, work,
        state.fnorm₀, fnorm_best, Int32(p_kept), cur_depth, true)
    info_ok = StepInfo{R}(
        accelerated_code(info_m), fnorm₊, stepnorm, failures, info_m.promoted,
        info_m.limited)
    return x, Fx, st_ok, info_ok
end
