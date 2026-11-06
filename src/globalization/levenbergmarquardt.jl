"""
    LevenbergMarquardt{R, S}

Adaptive Levenberg–Marquardt damping for [`Newton`](@ref), computing the trial step from the
damped linear model `(J(x)ᴴ J(x) + μ Dᵀ D) δ = -J(x)ᴴ F(x)` with damping parameter `μ > 0`
adapted from the gain ratio

```math
\\rho = \\frac{\\|F(x)\\|_2^2 - \\|F(x + \\delta)\\|_2^2}{\\|F(x)\\|_2^2 - \\|F(x) + J(x)\\,\\delta\\|_2^2}.
```

`μ` is absolute, in the units of `J(x)ᴴ J(x)`: scaling the residual by `k` scales `J(x)ᴴ J(x)`
by `k²` and leaves `μ` as it is, so a damping of `μ` on `k J(x)` acts as `μ / k²` does on
`J(x)`, and the `μ → 0` limit is decision 44's truncated step whatever the scale. With
`scaling = :diagonal`, `D` is instead the diagonal of the column norms of `J(x)`, floored at
`√eps` of the largest of them.

With [`SVDFactorization`](@ref) and `scaling = :none`, `J(x) = U Σ Vᴴ` is factorised once per
outer step and each trial `μ` evaluates the damped pseudoinverse step from `(U, Σ, V)` without
refactoring `J(x)`. With [`QRFactorization`](@ref) or [`LUFactorization`](@ref) (or when
`scaling = :diagonal`), `[J(x) | -F(x)]` is reduced once per outer step to an upper-triangular
`(R₀, c₀)` and each trial `μ` copies that pair and eliminates only its own damping rows `√μ D` by
Givens rotations, without forming `J(x)ᴴ J(x)`. That path indexes scalars and runs on a host
array: an `init` on a device array iterate raises an `ArgumentError` for it. A `μ` below
`4 eps` of the working type, which only an initial `μ₀` below the schedule's floor can be, is
taken as no damping: the truncated step of decision 44.
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

# Whether the damped step of this method comes from the one SVD factor of `J(x)`, which every
# trial reuses, rather than from the stacked QR system that each trial damps again.
@inline lm_svd_path(::SVDFactorization, ::Val{:none}) = true
@inline lm_svd_path(::Any, ::Any) = false

# The stacked QR solve indexes the entries of its workspace one at a time, which a device array
# disallows, and only the SVD step has a device form. `init_state` refuses the combination, so
# that a device solve raises a clear error instead of a scalar-indexing error at its first trial.
@inline function check_lm_iterate(lm::LevenbergMarquardt, linsolve, x)
    if x isa AbstractGPUArray && !lm_svd_path(linsolve, lm.scaling)
        throw(ArgumentError(
            "a device array iterate takes LevenbergMarquardt() with SVDFactorization() and " *
            "scaling = :none only: the damped step of QRFactorization(), LUFactorization() and " *
            "scaling = :diagonal is the stacked QR solve, which indexes scalars. Solve on an " *
            "Array to use that step."))
    end
    return nothing
end

@inline function init_globalization(::LevenbergMarquardt, x)
    T = eltype(x)
    R = real(T)
    n = x isa Number ? 1 : length(x)
    # A device iterate takes the SVD step (see `check_lm_iterate`), which reads none of the
    # stacked buffers: allocate them on the host only.
    stacked = !(x isa AbstractGPUArray)
    return (; Jδ = buffer(x), xt = buffer(x), Ft = buffer(x),
        R0 = stacked ? buffer(x, T, n, n) : nothing,
        c0 = stacked ? buffer(x, T, n) : nothing,
        R = stacked ? buffer(x, T, n, n) : nothing,
        c = stacked ? buffer(x, T, n) : nothing,
        d_diag = stacked ? buffer(x, R, n) : nothing,
        w = stacked ? buffer(x, T, n) : nothing)
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
# `2^-e σ̂_j / (σ̂_j² + 2^{-2e} μ)` from the stored singular value `σ̂_j` of the scaled matrix
# `J / 2^e`. That coefficient is `σ_j / (σ_j² + μ)` with `σ_j = 2^e σ̂_j`: the damping is absolute,
# in the units of `Jᴴ J`, so it is one number whatever the scale of `J`, and the `2^{-e}` and
# `2^{-2e}` only keep it inside the float range.
#
# `s = max(σ̂, 2^{-e} √μ)` evaluates the ratio without forming `σ̂² + 2^{-2e} μ`: `σ̂` and the
# rescaled `√μ` are divided by their larger one first, so neither a tiny `σ̂` nor a `μ` far above
# `σ̂²` can underflow or overflow the weight, and the `2^-e` is applied to the weight alone rather
# than to `σ̂`, which would overflow.
@inline function lm_svd_coord(σ̂::R, e::Int, y::T, μ::R) where {
        R <: AbstractFloat, T <: Number}
    damping = lm_step_damping(μ)
    sqrt_damping = ldexp2(sqrt(damping), -e)
    # `2^{-e} √μ` overflows at the bottom of the factor's scale range, where `e = -1023` and a
    # `μ` near `μ_max` put it above `floatmax`; the weight is then below the rounding of `x`, so
    # the coordinate is dropped rather than formed as a `NaN`.
    isfinite(sqrt_damping) || return zero(T)
    s = max(σ̂, sqrt_damping)
    σ_s = σ̂ / s
    μ_s = sqrt_damping / s
    w = (σ_s / (σ_s * σ_s + μ_s * μ_s)) / s
    return T(ldexp2(w, -e)) * y
end

# Evaluate δ = -∑_{j=1}^{rank} w_j (u_jᴴ Fx) v_j from a StaticSVD factorisation.
function lm_svd_step(
        ::Nothing, fac::StaticSVD{N, TF}, Fx::StaticVector{N, T},
        μ::R) where {N, TF, T, R <: AbstractFloat}
    U = fac.factors
    V = fac.V
    rk = Int(fac.rank)
    e = Int(fac.scale_exponent)
    δ = local_array(zero(SVector{N, T}))
    @inbounds for j in 1:rk
        y_j = zero(T)
        for i in 1:N
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
        δ_buf::StaticVector{N, T}, fac::StaticSVD{N, TF},
        Fx::StaticVector{N, T}, μ::R) where {N, TF, T, R <: AbstractFloat}
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
    n = length(Fx)
    rk = Int(fac.rank)
    e = Int(fac.scale_exponent)
    fill!(δ, zero(T))
    U = fac.U
    VT = fac.VT
    @inbounds for j in 1:rk
        y_j = zero(T)
        for i in 1:n
            y_j += T(conj(U[i, j])) * Fx[i]
        end
        c_j = -lm_svd_coord(R(fac.rwork[j]), e, y_j, μ)
        for i in 1:n
            δ[i] += T(conj(VT[j, i])) * c_j
        end
    end
    return δ
end

@kernel function ka_lm_svd_ut_kernel!(tmp, @Const(U), @Const(Fx), n::Int, rk::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(tmp)
        if j <= rk
            acc = zero(F)
            for i in 1:n
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
    n = length(Fx)
    rk = Int(fac.rank)
    if rk == 0 || n == 0
        fill!(δ, zero(T))
        return δ
    end
    backend = get_backend(fac.U)
    ka_lm_svd_ut_kernel!(backend)(fac.tmp, fac.U, Fx, n, rk; ndrange = n)
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

# The diagonal `D` of the damping term for the stacked solve of `[J; √μ D] δ = [-Fx; 0]`, written
# in place. `scaling = :none` takes `D = I`, so the damping is `√μ` in the units of `Jᴴ J`, the
# same number the SVD weight `σ̂_j/(σ̂_j² + 2^{-2e} μ)` divides by. `scaling = :diagonal` takes
# Marquardt's column norms, floored at `√eps(R)` of the largest of them, so that a zero or
# near-zero column still receives positive damping; it is scale-invariant on its own.
#
# The caller folds `√μ` into the diagonal; a damping entry that is not representable is a failed
# trial.
function lm_damping_diagonal!(
        d::AbstractVector, J::AbstractMatrix, ::Val{:diagonal})
    R = real(eltype(d))
    n = length(d)
    dmax = zero(R)
    @inbounds for j in 1:n
        d[j] = rnorm(view(J, :, j))
        dmax = max(dmax, d[j])
    end
    floor = iszero(dmax) ? one(R) : sqrt(eps(R)) * dmax
    @inbounds for j in 1:n
        d[j] = max(d[j], floor)
    end
    return d
end

function lm_damping_diagonal!(
        d::AbstractVector, ::AbstractMatrix, ::Val{:none})
    return fill!(d, one(real(eltype(d))))
end

# The same diagonal of a static matrix, as a value: the damping of the R1 stacked solve.
function lm_damping_diagonal(
        J::StaticMatrix{N, N, T}, scaling::Val) where {N, T}
    R = real(T)
    d = local_array(zero(SVector{N, R}))
    lm_damping_diagonal!(d, J, scaling)
    return SVector{N, R}(d)
end

# The stacked damped solve `[J; √μ D] δ = [-F; 0]` without forming `Jᴴ J`, in two phases:
# `[J | -F]` is reduced once per outer step to the upper-triangular pair `(R0, c0)`, and each
# trial `μ` copies that pair and eliminates only its own diagonal damping rows, so a rejected
# trial costs no factorisation of `J`.

# Reduce `[J | -F]`, written into `R0` and `c0`, to the upper-triangular pair `(R0, c0)` by
# Givens rotations on its rows.
function lm_qr_factor!(
        R0::AbstractMatrix, c0::AbstractVector, J::AbstractMatrix, Fx::AbstractVector)
    n = length(c0)
    @inbounds for j in 1:n
        c0[j] = -Fx[j]
        for i in 1:n
            R0[i, j] = J[i, j]
        end
    end
    @inbounds for j in 1:n
        for i in (j + 1):n
            b = R0[i, j]
            iszero(b) && continue
            a = R0[j, j]
            cs, sn, r_val = lm_givens(a, b)
            R0[j, j] = r_val
            R0[i, j] = zero(eltype(R0))
            for k in (j + 1):n
                u = R0[j, k]
                v = R0[i, k]
                R0[j, k] = cs * u + sn * v
                R0[i, k] = -conj(sn) * u + cs * v
            end
            u_r = c0[j]
            v_r = c0[i]
            c0[j] = cs * u_r + sn * v_r
            c0[i] = -conj(sn) * u_r + cs * v_r
        end
    end
    return R0, c0
end

# Whether `√μ D` is representable: a damping entry that overflows is a failed trial, which raises
# `μ`, not a step along a non-finite direction.
@inline function lm_damping_ok(sqrt_μ::Real, d_diag)
    isfinite(sqrt_μ) || return false
    for j in eachindex(d_diag)
        isfinite(sqrt_μ * d_diag[j]) || return false
    end
    return true
end

# Copy the stored factor `(R0, c0)` into the trial buffers `(R, c)` of the state, eliminate the
# damping rows `√μ d_j e_jᵀ` of `[R; √μ D] δ = [c; 0]` against the copy by Givens rotations, and
# back-substitute into `δ`; `w` is the length-`n` scratch row of the elimination. Returns `false`
# and zeroes `δ` where a pivot is zero or not finite.
function lm_qr_solve!(
        δ::AbstractVector, R::AbstractMatrix, c::AbstractVector, w::AbstractVector,
        R0::AbstractMatrix, c0::AbstractVector, sqrt_μ::Real, d_diag::AbstractVector)
    n = length(c)
    T = eltype(c)
    copyto!(R, R0)
    copyto!(c, c0)
    @inbounds for j in 1:n
        fill!(w, zero(T))
        w[j] = T(sqrt_μ * d_diag[j])
        w_rhs = zero(T)
        for k in j:n
            b = w[k]
            iszero(b) && continue
            a = R[k, k]
            cs, sn, r_val = lm_givens(a, b)
            R[k, k] = r_val
            w[k] = zero(T)
            for l in (k + 1):n
                u = R[k, l]
                v = w[l]
                R[k, l] = cs * u + sn * v
                w[l] = -conj(sn) * u + cs * v
            end
            u_r = c[k]
            c[k] = cs * u_r + sn * w_rhs
            w_rhs = -conj(sn) * u_r + cs * w_rhs
        end
    end
    @inbounds for i in n:-1:1
        s = c[i]
        for j in (i + 1):n
            s -= R[i, j] * δ[j]
        end
        r_ii = R[i, i]
        if iszero(r_ii) || !isfinite(r_ii)
            fill!(δ, zero(T))
            return false
        end
        δ[i] = s / r_ii
    end
    return all(isfinite, δ)
end

# One trial of the stacked solve: the damping `√μ D` eliminated against the stored factor, and
# `δ` read out of the result. `false` where the damping is not representable or the damped factor
# is singular, and `δ` is then zero.
@inline function lm_qr_step!(p, μ::R) where {R <: Real}
    sqrt_μ = sqrt(lm_step_damping(μ))
    if !lm_damping_ok(sqrt_μ, p.d_diag)
        fill!(p.δ, zero(eltype(p.δ)))
        return false
    end
    return lm_qr_solve!(p.δ, p.R, p.c, p.w, p.R0, p.c0, sqrt_μ, p.d_diag)
end

# The workspace of the stacked solve for a value iterate, factored at the start of an outer step:
# the factor `(R0, c0)` of `[J | -F]` and the trial buffers, all static, which `local_array` keeps
# off the heap where the element type is `isbits`.
@inline function lm_qr_prepare(
        ::Nothing, g, J::StaticMatrix{N, N, T}, Fx::StaticVector{N, T},
        scaling::Union{Val{:none}, Val{:diagonal}}) where {N, T}
    R0 = local_array(zero(SMatrix{N, N, T}))
    c0 = local_array(zero(SVector{N, T}))
    Rm = local_array(zero(SMatrix{N, N, T}))
    cm = local_array(zero(SVector{N, T}))
    w = local_array(zero(SVector{N, T}))
    δ = local_array(zero(SVector{N, T}))
    d_diag = lm_damping_diagonal(J, scaling)
    lm_qr_factor!(R0, c0, J, Fx)
    return (; R0, c0, R = Rm, c = cm, w, δ, d_diag)
end

# The same workspace for an array iterate, whose buffers `init_globalization` holds in the state.
@inline function lm_qr_prepare(
        δ::AbstractVector, g, J::AbstractMatrix, Fx::AbstractVector,
        scaling::Union{Val{:none}, Val{:diagonal}})
    lm_damping_diagonal!(g.d_diag, J, scaling)
    lm_qr_factor!(g.R0, g.c0, J, Fx)
    return (; R0 = g.R0, c0 = g.c0, R = g.R, c = g.c, w = g.w, δ, d_diag = g.d_diag)
end

# A scalar iterate is the 1×1 case of the static workspace.
@inline function lm_qr_prepare(
        ::Nothing, g, J::T, Fx::T,
        scaling::Union{Val{:none}, Val{:diagonal}}) where {T <: Number}
    return lm_qr_prepare(
        nothing, g, SMatrix{1, 1, T}((J,)), SVector{1, T}((Fx,)), scaling)
end

# The trial step `δ(μ)`: a value for a value iterate, and the buffer of the state for an array
# iterate.
@inline function lm_qr_step(p, ::StaticMatrix, μ::R) where {R <: Real}
    ok = lm_qr_step!(p, μ)
    return SVector(p.δ), ok
end

@inline function lm_qr_step(p, ::Number, μ::R) where {R <: Real}
    ok = lm_qr_step!(p, μ)
    return p.δ[1], ok
end

@inline function lm_qr_step(p, ::AbstractMatrix, μ::R) where {R <: Real}
    ok = lm_qr_step!(p, μ)
    return p.δ, ok
end
