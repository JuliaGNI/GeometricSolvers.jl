# The weight rules (`OptimalWeights`, `FixedWeights`) and the shared real-weight
# least-squares solver used by `Additive`.

"""
    OptimalWeights(ρ = 1.0)
    OptimalWeights(; ρ = 1.0)

Secant least-squares weight rule for [`Additive`](@ref). Minimises the linearised model
residual ``\\|F(x) + \\sum_i w_i (F(x_i) - F(x))\\|_2`` over real weights
``w_i \\in \\mathbb{R}``, drops rank-deficient candidate directions in favour of the
candidate with the smaller residual norm, and falls back to the best candidate when
``\\sum_i |w_i| < 0.1``, when ``\\|F(x_+)\\| > \\rho \\min_i \\|F(x_i)\\|``, or when
``F(x_+)`` is not finite.
"""
struct OptimalWeights{R <: Real}
    ρ::R
    function OptimalWeights{R}(ρ::R) where {R <: Real}
        isfinite(ρ) && ρ > zero(R) ||
            throw(ArgumentError("OptimalWeights ρ must be finite and positive, got $ρ"))
        return new{R}(ρ)
    end
end

function OptimalWeights(ρ::Real)
    r = float(ρ)
    return OptimalWeights{typeof(r)}(r)
end
OptimalWeights(; ρ::Real = 1.0) = OptimalWeights(ρ)

Adapt.@adapt_structure OptimalWeights

"""
    FixedWeights(w₁, w₂, ...)
    FixedWeights((w₁, w₂, ...))

Fixed real combination weights for [`Additive`](@ref), forming
``x_+ = x + \\sum_{i=1}^m w_i (x_i - x)`` without a Gram solve or safeguard evaluation.
"""
struct FixedWeights{M, R <: Real}
    weights::NTuple{M, R}
    function FixedWeights{M, R}(weights::NTuple{M, R}) where {M, R <: Real}
        M >= 1 || throw(ArgumentError("FixedWeights requires at least one weight"))
        all(isfinite, weights) ||
            throw(ArgumentError("FixedWeights requires finite real weights, got $weights"))
        return new{M, R}(weights)
    end
end

function FixedWeights(weights::Tuple{Vararg{Real}})
    isempty(weights) && throw(ArgumentError("FixedWeights requires at least one weight"))
    ws = map(float, promote(weights...))
    R = eltype(ws)
    return FixedWeights{length(ws), R}(ws)
end
FixedWeights(::Tuple{}) = throw(ArgumentError("FixedWeights requires at least one weight"))
FixedWeights(w1::Real, ws::Real...) = FixedWeights((w1, ws...))
FixedWeights() = throw(ArgumentError("FixedWeights requires at least one weight"))

Adapt.@adapt_structure FixedWeights

# Workspace for the R3 Gram matrix `G = Re(ΔFᴴ ΔF)` and right-hand side `b = -Re(ΔFᴴ Fx)`.
# On a CPU `Array`, `G_host === G_dev` and `b_host === b_dev`; on a GPU array, `G_dev` and
# `b_dev` live on the device and are copied once per step into `G_host` and `b_host`.
struct AdditiveWeightWorkspace{M, T, DM <: AbstractMatrix{T}, GM <: AbstractMatrix{T},
    BV <: AbstractVector{T}, GH <: Matrix{T}, BH <: Vector{T}}
    ΔF::DM
    G_dev::GM
    b_dev::BV
    G_host::GH
    b_host::BH
end

@inline function init_weight_workspace(
        ::OptimalWeights, x::Union{StaticArray, Number}, ::Val{M}) where {M}
    M <= 4 || throw(ArgumentError(
        "OptimalWeights on a value iterate supports at most 4 members, got $M"))
    return nothing
end

@inline init_weight_workspace(::FixedWeights, ::Any, ::Val) = nothing

function init_weight_workspace(
        ::OptimalWeights, x::AbstractArray{T}, ::Val{M}) where {T, M}
    M <= 4 || throw(ArgumentError(
        "OptimalWeights supports at most 4 members, got $M"))
    n = length(x)
    ΔF = buffer(x, T, n, M)
    G_dev = buffer(x, T, M, M)
    b_dev = buffer(x, T, M)
    G_host = G_dev isa Matrix{T} ? G_dev : Matrix{T}(undef, M, M)
    b_host = b_dev isa Vector{T} ? b_dev : Vector{T}(undef, M)
    return AdditiveWeightWorkspace{
        M, T, typeof(ΔF), typeof(G_dev), typeof(b_dev), typeof(G_host), typeof(b_host)}(
        ΔF, G_dev, b_dev, G_host, b_host)
end

@inline function single_weight(::Val{M}, k::Int, val::R) where {M, R}
    return ntuple(i -> ifelse(i == k, val, zero(R)), Val(M))
end

# Solve the small `M × M` real normal equations `G w = b` with the `sin²θ` rank rule
# (`τ_sin2 = 1e-3`) and Tikhonov regularisation `λ = eps(R) * (tr(G_kept) / p)`.
@inline function solve_optimal_weights(
        G::SMatrix{1, 1, R}, b::SVector{1, R},
        fnorms::NTuple{1, R}) where {R <: AbstractFloat}
    g11 = G[1, 1]
    b1 = b[1]
    if isfinite(g11) && g11 > zero(R) && isfinite(b1)
        w1 = b1 / (g11 * (one(R) + eps(R)))
        isfinite(w1) && return (w1,), false
    end
    return (one(R),), true
end

@inline function solve_optimal_weights(
        G::SMatrix{2, 2, R}, b::SVector{2, R},
        fnorms::NTuple{2, R}) where {R <: AbstractFloat}
    τ_sin2 = R(1e-3)
    g11 = G[1, 1]
    g22 = G[2, 2]
    g12 = R(0.5) * (G[1, 2] + G[2, 1])
    b1 = b[1]
    b2 = b[2]
    if isfinite(g11) && isfinite(g22) && isfinite(g12) && isfinite(b1) && isfinite(b2) &&
       g11 > zero(R) && g22 > zero(R)
        sin2 = one(R) - (g12 / g11) * (g12 / g22)
        if !(sin2 < τ_sin2) && isfinite(sin2)
            λ = eps(R) * R(0.5) * (g11 + g22)
            a11 = g11 + λ
            a22 = g22 + λ
            det = a11 * a22 - g12 * g12
            if isfinite(det) && det > zero(R)
                w1 = (a22 * b1 - g12 * b2) / det
                w2 = (a11 * b2 - g12 * b1) / det
                if isfinite(w1) && isfinite(w2)
                    return (w1, w2), false
                end
            end
        end
    end
    keep1 = fnorms[1] <= fnorms[2] || !isfinite(fnorms[2])
    if keep1
        w1 = (isfinite(g11) && g11 > zero(R) && isfinite(b1)) ?
             (b1 / (g11 * (one(R) + eps(R)))) : one(R)
        w1 = isfinite(w1) ? w1 : one(R)
        return (w1, zero(R)), true
    else
        w2 = (isfinite(g22) && g22 > zero(R) && isfinite(b2)) ?
             (b2 / (g22 * (one(R) + eps(R)))) : one(R)
        w2 = isfinite(w2) ? w2 : one(R)
        return (zero(R), w2), true
    end
end

@inline function solve_optimal_weights(
        G::SMatrix{M, M, R}, b::SVector{M, R},
        fnorms::NTuple{M, R}) where {M, R <: AbstractFloat}
    τ_sin2 = R(1e-3)
    order = SVector{M, Int}(ntuple(identity, Val(M)))
    @inbounds for i in 2:M
        key = order[i]
        fkey = fnorms[key]
        j = i - 1
        while j >= 1
            oj = order[j]
            foj = fnorms[oj]
            comes_before = (isfinite(fkey) && !isfinite(foj)) ||
                           (fkey < foj) ||
                           (fkey == foj && key < oj)
            comes_before || break
            order = setindex(order, oj, j + 1)
            j -= 1
        end
        order = setindex(order, key, j + 1)
    end
    if !all(isfinite, G) || !all(isfinite, b)
        k = order[1]
        gkk = G[k, k]
        bk = b[k]
        wk_nf_raw = (isfinite(gkk) && gkk > zero(R) && isfinite(bk)) ?
                    (bk / (gkk * (one(R) + eps(R)))) : one(R)
        wk_nf = isfinite(wk_nf_raw) ? wk_nf_raw : one(R)
        return single_weight(Val(M), k, wk_nf), true
    end
    kept = zero(SVector{M, Int})
    L = zero(G)
    p = 0
    rank_fallback = false
    @inbounds for q in 1:M
        idx = order[q]
        gii = G[idx, idx]
        if !(gii > zero(R))
            rank_fallback = true
            continue
        end
        proj2 = zero(R)
        ok = true
        for c in 1:p
            kc = kept[c]
            s = R(0.5) * (G[idx, kc] + G[kc, idx])
            for r in 1:(c - 1)
                s -= L[p + 1, r] * L[c, r]
            end
            lcc = L[c, c]
            if !(lcc > zero(R))
                ok = false
                break
            end
            l_pc = s / lcc
            L = setindex(L, l_pc, p + 1, c)
            proj2 += (l_pc / sqrt(gii))^2
        end
        sin2 = one(R) - proj2
        r_jj2 = gii * sin2
        if !ok || !isfinite(sin2) || (p >= 1 && sin2 < τ_sin2) || !(r_jj2 > zero(R))
            rank_fallback = true
        else
            p += 1
            kept = setindex(kept, idx, p)
            L = setindex(L, sqrt(r_jj2), p, p)
        end
    end
    if p == 0
        k = order[1]
        return single_weight(Val(M), k, one(R)), true
    elseif p == 1
        k = kept[1]
        gkk = G[k, k]
        wk_p1_raw = b[k] / (gkk * (one(R) + eps(R)))
        wk_p1 = isfinite(wk_p1_raw) ? wk_p1_raw : one(R)
        return single_weight(Val(M), k, wk_p1), true
    end
    tr_kept = zero(R)
    @inbounds for c in 1:p
        tr_kept += G[kept[c], kept[c]]
    end
    λ = eps(R) * (tr_kept / R(p))
    L_reg = zero(G)
    @inbounds for i in 1:p
        ki = kept[i]
        for j in 1:i
            kj = kept[j]
            s = i == j ? (G[ki, ki] + λ) : R(0.5) * (G[ki, kj] + G[kj, ki])
            for r in 1:(j - 1)
                s -= L_reg[i, r] * L_reg[j, r]
            end
            if i == j
                L_reg = setindex(L_reg, sqrt(max(s, floatmin(R))), i, i)
            else
                L_reg = setindex(L_reg, s / L_reg[j, j], i, j)
            end
        end
    end
    y = zero(SVector{M, R})
    @inbounds for i in 1:p
        s = b[kept[i]]
        for r in 1:(i - 1)
            s -= L_reg[i, r] * y[r]
        end
        y = setindex(y, s / L_reg[i, i], i)
    end
    x_kept = zero(SVector{M, R})
    @inbounds for i in p:-1:1
        s = y[i]
        for r in (i + 1):p
            s -= L_reg[r, i] * x_kept[r]
        end
        x_kept = setindex(x_kept, s / L_reg[i, i], i)
    end
    w_vec = zero(SVector{M, R})
    @inbounds for c in 1:p
        w_vec = setindex(w_vec, x_kept[c], kept[c])
    end
    return Tuple(w_vec), rank_fallback
end

@inline function fill_delta_columns!(
        ΔF::Matrix{T}, Fx::Vector{T}, Fxs::NTuple{M, Vector{T}}) where {M, T}
    @inbounds for j in 1:M
        Fj = Fxs[j]
        for i in axes(ΔF, 1)
            ΔF[i, j] = Fj[i] - Fx[i]
        end
    end
    return ΔF
end

@inline function fill_delta_columns!(
        ΔF::AbstractMatrix{T}, Fx::AbstractArray{T},
        Fxs::NTuple{M, AbstractArray{T}}) where {M, T}
    @inbounds for j in 1:M
        @views map!(-, ΔF[:, j], Fxs[j], Fx)
    end
    return ΔF
end

@inline function optimal_weights(
        ::Nothing, Fx, Fxs::Tuple{Any, Vararg{Any, N}},
        fnorms::Tuple{R, Vararg{R, N}}) where {N, R <: AbstractFloat}
    M = N + 1
    δFs = ntuple(i -> Fxs[i] - Fx, Val(M))
    G = SMatrix{M, M, R}(ntuple(Val(M * M)) do k
        i = mod1(k, M)
        j = cld(k, M)
        i <= j ? rdot(δFs[i], δFs[j]) : rdot(δFs[j], δFs[i])
    end)
    b = SVector{M, R}(ntuple(i -> -rdot(δFs[i], Fx), Val(M)))
    return solve_optimal_weights(G, b, fnorms)
end

@inline function optimal_weights(
        work::AdditiveWeightWorkspace{M, T}, Fx::AbstractArray{T},
        Fxs::NTuple{M, AbstractArray{T}},
        fnorms::Tuple{R, Vararg{R, N}}) where {M, T, R <: AbstractFloat, N}
    fill_delta_columns!(work.ΔF, Fx, Fxs)
    mul!!(work.G_dev, work.ΔF', work.ΔF)
    mul!!(work.b_dev, work.ΔF', Fx)
    if work.G_dev !== work.G_host
        copyto!(work.G_host, work.G_dev)
        copyto!(work.b_host, work.b_dev)
    end
    G = SMatrix{M, M, R}(ntuple(k -> real(work.G_host[k]), Val(M * M)))
    b = SVector{M, R}(ntuple(i -> -real(work.b_host[i]), Val(M)))
    return solve_optimal_weights(G, b, fnorms)
end
