"""
    LevenbergMarquardt{R, S}

Adaptive Levenberg–Marquardt damping for [`Newton`](@ref), computing the trial step from the
damped linear model `(J(x)ᴴ J(x) + μ s² Dᵀ D) δ = -J(x)ᴴ F(x)` with damping parameter `μ > 0`
adapted from the gain ratio

```math
\\rho = \\frac{\\|F(x)\\|_2^2 - \\|F(x + \\delta)\\|_2^2}{\\|F(x)\\|_2^2 - \\|F(x) + J(x)\\,\\delta\\|_2^2}.
```

`μ` is measured in the power-of-two scale `s = 2^e` that the factorisation of `J(x)` divides it
by, `e` the exponent of the largest entry of `J(x)`, so a step is the same for `J(x)` and for
`2^k J(x)`: the undamped truncated step of decision 44 is scale-free, and a damping of `μ` is
too. With `scaling = :diagonal`, `D` is instead the diagonal of the column norms of `J(x)`,
floored at `√eps` of the largest of them.

With [`SVDFactorization`](@ref) and `scaling = :none`, `J(x) = U Σ Vᴴ` is factorised once per
outer step and each trial `μ` evaluates the damped pseudoinverse step from `(U, Σ, V)` without
refactoring `J(x)`. With [`QRFactorization`](@ref) or [`LUFactorization`](@ref) (or when
`scaling = :diagonal`), the damped step is solved by Givens QR elimination of the stacked matrix
`[J(x); √μ s D]` without forming `J(x)ᴴ J(x)`; that path indexes scalars and runs on a host
array.
"""
struct LevenbergMarquardt{R <: AbstractFloat, S}
    μ₀::R
    μ_min::R
    μ_max::R
    μ_decrease::R
    μ_increase::R
    ρ_accept::R
    scaling::S
end

Adapt.@adapt_structure LevenbergMarquardt

function LevenbergMarquardt(
        ::Type{R} = Float64;
        μ₀::Real = R(1e-6),
        μ_min::Real = max(R(1e-15), 4 * eps(R)),
        μ_max::Real = R(1e12),
        μ_decrease::Real = R(1) / R(3),
        μ_increase::Real = R(4),
        ρ_accept::Real = zero(R),
        scaling::Symbol = :none) where {R <: AbstractFloat}
    μ₀_r = R(μ₀)
    μ_min_r = R(μ_min)
    μ_max_r = R(μ_max)
    μ_dec_r = R(μ_decrease)
    μ_inc_r = R(μ_increase)
    ρ_acc_r = R(ρ_accept)
    finite_bounds = isfinite(μ₀_r) && isfinite(μ_min_r) && isfinite(μ_max_r)
    if !(finite_bounds && zero(R) < μ_min_r <= μ₀_r <= μ_max_r)
        throw(ArgumentError(
            "LevenbergMarquardt requires finite damping bounds 0 < μ_min ≤ μ₀ ≤ μ_max < Inf; got μ_min = $μ_min, μ₀ = $μ₀, μ_max = $μ_max"))
    end
    if !(isfinite(μ_dec_r) && zero(R) < μ_dec_r < one(R))
        throw(ArgumentError(
            "LevenbergMarquardt requires 0 < μ_decrease < 1; got μ_decrease = $μ_decrease"))
    end
    if !(isfinite(μ_inc_r) && μ_inc_r > one(R))
        throw(ArgumentError(
            "LevenbergMarquardt requires μ_increase > 1; got μ_increase = $μ_increase"))
    end
    if !(isfinite(ρ_acc_r) && zero(R) <= ρ_acc_r < one(R))
        throw(ArgumentError(
            "LevenbergMarquardt requires 0 ≤ ρ_accept < 1; got ρ_accept = $ρ_accept"))
    end
    if !(scaling === :none || scaling === :diagonal)
        throw(ArgumentError(
            "LevenbergMarquardt requires scaling ∈ (:none, :diagonal); got scaling = $(repr(scaling))"))
    end
    return LevenbergMarquardt{R, Val{scaling}}(
        μ₀_r, μ_min_r, μ_max_r, μ_dec_r, μ_inc_r, ρ_acc_r, Val(scaling))
end

@inline function init_globalization(::LevenbergMarquardt, x, Fx = x)
    T = eltype(x)
    R = real(T)
    n = x isa Number ? 1 : length(x)
    m = Fx isa Number ? 1 : length(Fx)
    return (; Jδ = buffer(Fx), xt = buffer(x), Ft = buffer(Fx),
        Rmat = buffer(x, T, m, n), rhs = buffer(Fx),
        d_diag = buffer(x, R, n), w_row = buffer(x, T, n))
end

@inline lm_Jδ(::Nothing) = nothing
@inline lm_Jδ(g) = g.Jδ
@inline lm_xt(::Nothing) = nothing
@inline lm_xt(g) = g.xt
@inline lm_Ft(::Nothing) = nothing
@inline lm_Ft(g) = g.Ft

# The damping a trial step uses. The schedule floors `μ` at `max(μ_min, 4 eps(R))` after every
# accepted step, so a `μ` below that floor arrives only as the initial `μ₀` that `start!!` stores,
# and the schedule never returns to it. It is taken as no damping: the truncated step of decision
# 44, which is the `μ → 0` limit of the damped one.
@inline function lm_step_damping(μ::R) where {R <: AbstractFloat}
    return μ < 4 * eps(R) ? zero(R) : μ
end

# The damped coefficient of the SVD step: the coordinate `y = u_jᴴ F(x)` is multiplied by
# `2^-e σ̂_j / (σ̂_j² + μ)` from the stored singular value `σ̂_j` of the scaled matrix `J / 2^e`.
# That coefficient is `σ_j / (σ_j² + 2^{2e} μ)` in the units of `J`: the damping is measured in the
# factor's own power-of-two scale, which is what makes the step the same for `J` and for `2^k J`,
# as the undamped truncated step of decision 44 already is.
#
# `s = max(σ̂, √μ)` evaluates the ratio without forming `σ̂² + μ`: `σ̂` and `√μ` are rescaled by
# their larger one first, so neither a tiny `σ̂` nor a `μ` far above `σ̂²` can underflow or
# overflow the weight, and the `2^-e` is applied to the weight alone rather than to `σ̂`, which
# would overflow.
@inline function lm_svd_coord(σ̂::R, e::Int, y::T, μ::R) where {
        R <: AbstractFloat, T <: Number}
    damping = lm_step_damping(μ)
    s = max(σ̂, sqrt(damping))
    σ_s = σ̂ / s
    μ_s = sqrt(damping) / s
    w = (σ_s / (σ_s * σ_s + μ_s * μ_s)) / s
    return T(ldexp2(w, -e)) * y
end

# Evaluate δ = -∑_{j=1}^{rank} w_j (u_jᴴ Fx) v_j from a StaticSVD or StaticRectSVD factorisation.
function lm_svd_step(
        ::Nothing, fac::Union{StaticSVD{N, TF}, StaticRectSVD{<:Any, N, TF}},
        Fx::StaticVector{M, T}, μ::R) where {M, N, TF, T, R <: AbstractFloat}
    U = fac.factors
    V = fac.V
    rk = Int(fac.rank)
    e = Int(fac.scale_exponent)
    δ = local_array(zero(SVector{N, T}))
    @inbounds for j in 1:rk
        y_j = zero(T)
        for i in 1:M
            y_j += T(conj(U[i, j])) * Fx[i]
        end
        c_j = -lm_svd_coord(R(fac.σ̂[j]), e, y_j, μ)
        for i in 1:N
            δ[i] += T(V[i, j]) * c_j
        end
    end
    return SVector{N, T}(δ)
end

function lm_svd_step(
        δ_buf::StaticVector{N, T},
        fac::Union{StaticSVD{N, TF}, StaticRectSVD{<:Any, N, TF}},
        Fx::StaticVector{M, T}, μ::R) where {M, N, TF, T, R <: AbstractFloat}
    s = lm_svd_step(nothing, fac, Fx, μ)
    copyto!(δ_buf, s)
    return δ_buf
end

function lm_svd_step(
        ::Nothing, fac::StaticSVD{1, TF}, Fx::T,
        μ::R) where {TF, T <: Number, R <: AbstractFloat}
    s = lm_svd_step(nothing, fac, SVector{1, T}((Fx,)), μ)
    return s[1]
end

# Evaluate δ = -∑_{j=1}^{rank} w_j (u_jᴴ Fx) v_j from a DenseSVD factorisation.
function lm_svd_step(
        δ::AbstractVector{T}, fac::DenseSVD, Fx::AbstractVector{T},
        μ::R) where {T <: Number, R <: AbstractFloat}
    m = length(Fx)
    n = length(δ)
    rk = Int(fac.rank)
    e = Int(fac.scale_exponent)
    fill!(δ, zero(T))
    U = fac.U
    VT = fac.VT
    @inbounds for j in 1:rk
        y_j = zero(T)
        for i in 1:m
            y_j += T(conj(U[i, j])) * Fx[i]
        end
        c_j = -lm_svd_coord(R(fac.rwork[j]), e, y_j, μ)
        for i in 1:n
            δ[i] += T(conj(VT[j, i])) * c_j
        end
    end
    return δ
end

@kernel function ka_lm_svd_ut_kernel!(tmp, @Const(U), @Const(Fx), m::Int, n::Int, rk::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(tmp)
        if j <= rk
            acc = zero(F)
            for i in 1:m
                acc += conj(U[i, j]) * F(Fx[i])
            end
            tmp[j] = acc
        else
            tmp[j] = zero(F)
        end
    end
end

@kernel function ka_lm_svd_weight_kernel!(
        tmp, @Const(σ̂), rk::Int, e::Int, μ::R) where {R <: AbstractFloat}
    j = @index(Global, Linear)
    @inbounds if j <= rk
        F = eltype(tmp)
        tmp[j] = -F(lm_svd_coord(R(σ̂[j]), e, tmp[j], μ))
    end
end

function lm_svd_step(
        δ::AbstractVector{T}, fac::DeviceSVD, Fx::AbstractVector{T},
        μ::R) where {T <: Number, R <: AbstractFloat}
    m = length(Fx)
    n = length(δ)
    rk = Int(fac.rank)
    if rk == 0 || n == 0
        fill!(δ, zero(T))
        return δ
    end
    backend = get_backend(fac.U)
    ka_lm_svd_ut_kernel!(backend)(fac.tmp, fac.U, Fx, m, n, rk; ndrange = n)
    ka_lm_svd_weight_kernel!(backend)(
        fac.tmp, fac.σ̂, rk, Int(fac.scale_exponent), μ; ndrange = rk)
    ka_svd_v_mul_kernel!(backend)(δ, fac.VT, fac.tmp, n; ndrange = n)
    isgpu(backend) && synchronize(backend)
    return δ
end

# Givens rotation (cs, sn, r) annihilating b against a: [cs sn; -conj(sn) cs] [a; b] = [r; 0].
@inline function lm_givens(a::T, b::T) where {T <: Number}
    R = real(T)
    iszero(b) && return one(R), zero(T), a
    if iszero(a)
        ab = abs(b)
        return zero(R), conj(b) / ab, T(ab)
    end
    scale = max(abs(a), abs(b))
    a_s = a / scale
    b_s = b / scale
    nrm = sqrt(abs2(a_s) + abs2(b_s))
    r_val = scale * nrm
    phase = a_s / abs(a_s)
    cs = abs(a_s) / nrm
    sn = phase * conj(b_s) / nrm
    return cs, sn, phase * r_val
end

# The diagonal `D^{1/2}` of the damping term and the scalar `√μ` it is multiplied by, for the
# stacked solve of `[J; √μ s D] δ = [-Fx; 0]`. `scaling = :none` takes `D = I` and measures `μ` in
# the factor's power-of-two scale `2^e`, the scale the SVD path divides `J` by, so that both paths
# solve the same damped system and neither changes with the units of `J`. `scaling = :diagonal`
# takes Marquardt's column norms, floored at `√eps(R)` of the largest of them, so that a zero or
# near-zero column still receives positive damping; it is scale-invariant on its own.
#
# Returns the diagonal and `√μ` folded with that scale; the caller treats a damping entry that is
# not representable as a failed trial.
function lm_damping_diagonal(
        J::StaticMatrix{M, N, T}, scaling::Val{:diagonal}) where {M, N, T}
    R = real(T)
    d_diag = local_array(zero(SVector{N, R}))
    dmax = zero(R)
    @inbounds for j in 1:N
        col = local_array(zero(SVector{M, T}))
        for i in 1:M
            col[i] = J[i, j]
        end
        d_diag[j] = rnorm(SVector(col))
        dmax = max(dmax, d_diag[j])
    end
    floor = iszero(dmax) ? one(R) : sqrt(eps(R)) * dmax
    @inbounds for j in 1:N
        d_diag[j] = max(d_diag[j], floor)
    end
    return SVector{N, R}(d_diag), one(R)
end

function lm_damping_diagonal(J::StaticMatrix{M, N, T}, ::Val{:none}) where {M, N, T}
    R = real(T)
    return SVector{N, R}(ntuple(_ -> one(R), Val(N))), ldexp2(one(R), matrix_exponent(J))
end

# Stacked QR solve of [J; √μ s D] δ = [-Fx; 0] via Householder QR of J followed by Givens
# elimination of √μ s D, without forming Jᴴ J.
function lm_stacked_qr_step(
        ::Nothing, J::StaticMatrix{M, N, T}, Fx::StaticVector{M, T},
        μ::R, scaling::Union{Val{:none}, Val{:diagonal}}) where {M, N, T, R}
    M >= N || throw(DimensionMismatch("lm_stacked_qr_step requires M >= N, got ($M, $N)"))
    d_diag, d_scale = lm_damping_diagonal(J, scaling)
    sqrt_μ = sqrt(lm_step_damping(μ)) * d_scale
    damping_ok = isfinite(sqrt_μ) && all(d -> isfinite(sqrt_μ * d), d_diag)
    damping_ok || return zero(SVector{N, T}), false
    Rmat = local_array(SMatrix{M, N, T}(J))
    rhs = local_array(-SVector{M, T}(Fx))
    # Reduce Rmat to upper triangular by Givens rotations on rows and apply to rhs.
    @inbounds for j in 1:N
        for i in (j + 1):M
            a = Rmat[j, j]
            b = Rmat[i, j]
            iszero(b) && continue
            cs, sn, r_val = lm_givens(a, b)
            Rmat[j, j] = r_val
            Rmat[i, j] = zero(T)
            for k in (j + 1):N
                u = Rmat[j, k]
                v = Rmat[i, k]
                Rmat[j, k] = cs * u + sn * v
                Rmat[i, k] = -conj(sn) * u + cs * v
            end
            u_r = rhs[j]
            v_r = rhs[i]
            rhs[j] = cs * u_r + sn * v_r
            rhs[i] = -conj(sn) * u_r + cs * v_r
        end
    end
    # Eliminate the diagonal damping block √μ s D row by row with Givens rotations.
    w_row = local_array(zero(SVector{N, T}))
    @inbounds for j in 1:N
        fill!(w_row, zero(T))
        w_row[j] = T(sqrt_μ * d_diag[j])
        w_rhs = zero(T)
        for k in j:N
            a = Rmat[k, k]
            b = w_row[k]
            iszero(b) && continue
            cs, sn, r_val = lm_givens(a, b)
            Rmat[k, k] = r_val
            w_row[k] = zero(T)
            for l in (k + 1):N
                u = Rmat[k, l]
                v = w_row[l]
                Rmat[k, l] = cs * u + sn * v
                w_row[l] = -conj(sn) * u + cs * v
            end
            u_r = rhs[k]
            rhs[k] = cs * u_r + sn * w_rhs
            w_rhs = -conj(sn) * u_r + cs * w_rhs
        end
    end
    # Back-substitution on the damped upper-triangular factor Rmat[1:N, 1:N].
    δ = local_array(zero(SVector{N, T}))
    @inbounds for i in N:-1:1
        s = rhs[i]
        for j in (i + 1):N
            s -= Rmat[i, j] * δ[j]
        end
        r_ii = Rmat[i, i]
        if iszero(r_ii) || !isfinite(r_ii)
            return zero(SVector{N, T}), false
        end
        δ[i] = s / r_ii
    end
    res = SVector{N, T}(δ)
    return res, all(isfinite, res)
end

function lm_stacked_qr_step(
        δ_buf::StaticVector{N, T}, J::StaticMatrix{M, N, T}, Fx::StaticVector{M, T},
        μ::R, scaling::Union{Val{:none}, Val{:diagonal}}) where {M, N, T, R}
    res, ok = lm_stacked_qr_step(nothing, J, Fx, μ, scaling)
    copyto!(δ_buf, res)
    return δ_buf, ok
end

function lm_stacked_qr_step(
        ::Nothing, J::T, Fx::T, μ::R,
        scaling::Union{Val{:none}, Val{:diagonal}}) where {T <: Number, R}
    res, ok = lm_stacked_qr_step(
        nothing, SMatrix{1, 1, T}((J,)), SVector{1, T}((Fx,)), μ, scaling)
    return res[1], ok
end

@kernel function ka_lm_stacked_qr_kernel!(
        δ, Rmat, rhs, d_diag, w_row, @Const(J), @Const(Fx),
        m::Int, n::Int, sqrt_μ::R, diag_scaling::Bool) where {R <: AbstractFloat}
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        T = eltype(δ)
        for j in 1:n, i in 1:m
            Rmat[i, j] = J[i, j]
        end
        for i in 1:m
            rhs[i] = -Fx[i]
        end
        dmax = zero(R)
        for j in 1:n
            if diag_scaling
                s = zero(R)
                for i in 1:m
                    s = max(s, abs(Rmat[i, j]))
                end
                if iszero(s) || !isfinite(s)
                    d_diag[j] = one(R)
                else
                    ss = zero(R)
                    for i in 1:m
                        ss += abs2(Rmat[i, j] / s)
                    end
                    cn = s * sqrt(ss)
                    d_diag[j] = cn
                end
                dmax = max(dmax, d_diag[j])
            else
                d_diag[j] = one(R)
            end
        end
        if diag_scaling
            floor = iszero(dmax) ? one(R) : sqrt(eps(R)) * dmax
            for j in 1:n
                d_diag[j] = max(d_diag[j], floor)
            end
        end
        for j in 1:n
            for i in (j + 1):m
                a = Rmat[j, j]
                b = Rmat[i, j]
                iszero(b) && continue
                cs, sn, r_val = lm_givens(a, b)
                Rmat[j, j] = r_val
                Rmat[i, j] = zero(T)
                for k in (j + 1):n
                    u = Rmat[j, k]
                    v = Rmat[i, k]
                    Rmat[j, k] = cs * u + sn * v
                    Rmat[i, k] = -conj(sn) * u + cs * v
                end
                u_r = rhs[j]
                v_r = rhs[i]
                rhs[j] = cs * u_r + sn * v_r
                rhs[i] = -conj(sn) * u_r + cs * v_r
            end
        end
        for j in 1:n
            for l in 1:n
                w_row[l] = zero(T)
            end
            w_row[j] = T(sqrt_μ * d_diag[j])
            w_rhs = zero(T)
            for k in j:n
                a = Rmat[k, k]
                b = w_row[k]
                iszero(b) && continue
                cs, sn, r_val = lm_givens(a, b)
                Rmat[k, k] = r_val
                w_row[k] = zero(T)
                for l in (k + 1):n
                    u = Rmat[k, l]
                    v = w_row[l]
                    Rmat[k, l] = cs * u + sn * v
                    w_row[l] = -conj(sn) * u + cs * v
                end
                u_r = rhs[k]
                rhs[k] = cs * u_r + sn * w_rhs
                w_rhs = -conj(sn) * u_r + cs * w_rhs
            end
        end
        for i in n:-1:1
            s = rhs[i]
            for j in (i + 1):n
                s -= Rmat[i, j] * δ[j]
            end
            r_ii = Rmat[i, i]
            if iszero(r_ii) || !isfinite(r_ii)
                for k in 1:n
                    δ[k] = T(NaN)
                end
                break
            end
            δ[i] = s / r_ii
        end
    end
end

function lm_stacked_qr_step(
        δ::AbstractVector{T}, J::AbstractMatrix{T}, Fx::AbstractVector{T},
        μ::R, scaling::Union{Val{:none}, Val{:diagonal}}, g = nothing) where {T, R}
    m = length(Fx)
    n = length(δ)
    Rmat = g === nothing || g.Rmat === nothing ? buffer(δ, T, m, n) : g.Rmat
    rhs = g === nothing || g.rhs === nothing ? buffer(Fx) : g.rhs
    d_diag = g === nothing || g.d_diag === nothing ? buffer(δ, R, n) : g.d_diag
    w_row = g === nothing || g.w_row === nothing ? buffer(δ, T, n) : g.w_row
    if J isa AbstractGPUArray
        backend = get_backend(J)
        d_scale = scaling === Val(:diagonal) ? one(R) : ldexp2(one(R), matrix_exponent(J))
        sqrt_μ = sqrt(lm_step_damping(μ)) * d_scale
        ka_lm_stacked_qr_kernel!(backend)(
            δ, Rmat, rhs, d_diag, w_row, J, Fx, m, n,
            sqrt_μ, scaling === Val(:diagonal); ndrange = 1)
        isgpu(backend) && synchronize(backend)
        ok = all(isfinite, δ)
        if !ok
            fill!(δ, zero(T))
        end
        return δ, ok
    end
    copyto!(Rmat, J)
    @inbounds for i in 1:m
        rhs[i] = -Fx[i]
    end
    dmax = zero(R)
    @inbounds for j in 1:n
        if scaling === Val(:diagonal)
            cn = rnorm(@view(Rmat[:, j]))
            d_diag[j] = cn
            dmax = max(dmax, cn)
        else
            d_diag[j] = one(R)
        end
    end
    d_scale = one(R)
    if scaling === Val(:diagonal)
        floor = iszero(dmax) ? one(R) : sqrt(eps(R)) * dmax
        @inbounds for j in 1:n
            d_diag[j] = max(d_diag[j], floor)
        end
    else
        d_scale = ldexp2(one(R), matrix_exponent(J))
    end
    sqrt_μ = sqrt(lm_step_damping(μ)) * d_scale
    damping_ok = isfinite(sqrt_μ)
    @inbounds for j in 1:n
        damping_ok &= isfinite(sqrt_μ * d_diag[j])
    end
    damping_ok || begin
        fill!(δ, zero(T))
        return δ, false
    end
    @inbounds for j in 1:n
        for i in (j + 1):m
            a = Rmat[j, j]
            b = Rmat[i, j]
            iszero(b) && continue
            cs, sn, r_val = lm_givens(a, b)
            Rmat[j, j] = r_val
            Rmat[i, j] = zero(T)
            for k in (j + 1):n
                u = Rmat[j, k]
                v = Rmat[i, k]
                Rmat[j, k] = cs * u + sn * v
                Rmat[i, k] = -conj(sn) * u + cs * v
            end
            u_r = rhs[j]
            v_r = rhs[i]
            rhs[j] = cs * u_r + sn * v_r
            rhs[i] = -conj(sn) * u_r + cs * v_r
        end
    end
    @inbounds for j in 1:n
        fill!(w_row, zero(T))
        w_row[j] = T(sqrt_μ * d_diag[j])
        w_rhs = zero(T)
        for k in j:n
            a = Rmat[k, k]
            b = w_row[k]
            iszero(b) && continue
            cs, sn, r_val = lm_givens(a, b)
            Rmat[k, k] = r_val
            w_row[k] = zero(T)
            for l in (k + 1):n
                u = Rmat[k, l]
                v = w_row[l]
                Rmat[k, l] = cs * u + sn * v
                w_row[l] = -conj(sn) * u + cs * v
            end
            u_r = rhs[k]
            rhs[k] = cs * u_r + sn * w_rhs
            w_rhs = -conj(sn) * u_r + cs * w_rhs
        end
    end
    @inbounds for i in n:-1:1
        s = rhs[i]
        for j in (i + 1):n
            s -= Rmat[i, j] * δ[j]
        end
        r_ii = Rmat[i, i]
        if iszero(r_ii) || !isfinite(r_ii)
            fill!(δ, zero(T))
            return δ, false
        end
        δ[i] = s / r_ii
    end
    return δ, all(isfinite, δ)
end

# The value-iterate calls carry the workspace tuple of the R3 state, which those paths do not
# read: the two methods below drop it. Their signatures stay disjoint from the dense one above,
# which takes an `AbstractMatrix` and an `AbstractVector`.
@inline function lm_stacked_qr_step(
        δ_buf::Union{Nothing, StaticVector}, J::StaticMatrix,
        Fx::StaticVector, μ::R,
        scaling::Union{Val{:none}, Val{:diagonal}}, _) where {R <: AbstractFloat}
    return lm_stacked_qr_step(δ_buf, J, Fx, μ, scaling)
end

@inline function lm_stacked_qr_step(
        δ_buf::Union{Nothing, StaticVector}, J::Number, Fx::Number, μ::R,
        scaling::Union{Val{:none}, Val{:diagonal}}, _) where {R <: AbstractFloat}
    return lm_stacked_qr_step(δ_buf, J, Fx, μ, scaling)
end
