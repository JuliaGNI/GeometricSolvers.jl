"""
    LevenbergMarquardt{R, S}
    LevenbergMarquardt(::Type{R} = Float64; μ₀ = R(1e-6), μ_min = max(R(1e-15), 4 * eps(R)),
                       μ_max = R(1e12), μ_decrease = R(1) / R(3), μ_increase = R(4),
                       ρ_accept = zero(R), scaling = :none) where {R <: AbstractFloat}

Adaptive Levenberg–Marquardt damping for [`Newton`](@ref), computing the trial step from the
damped linear model `(J(x)ᴴ J(x) + μ Dᵀ D) δ = -J(x)ᴴ F(x)` with damping parameter `μ > 0`
adapted from the gain ratio

```math
\\rho = \\frac{\\|F(x)\\|_2^2 - \\|F(x + \\delta)\\|_2^2}{\\|F(x)\\|_2^2 - \\|F(x) + J(x)\\,\\delta\\|_2^2}.
```

`μ` is absolute, in the units of `J(x)ᴴ J(x)`: scaling the residual by `k` scales `J(x)ᴴ J(x)`
by `k²` and leaves `μ` as it is, so a damping of `μ` on `k J(x)` acts as `μ / k²` does on
`J(x)`, and the `μ → 0` limit is the Newton step, truncated to the numerical rank as the solve of
[`SVDFactorization`](@ref) truncates it, whatever the scale. With
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
taken as no damping: the truncated Newton step, the `μ → 0` limit.

A solve starts at `μ₀`. A trial is accepted when `ρ > ρ_accept`, and an accepted step sets `μ` to
`max(μ · μ_decrease, max(μ_min, 4 eps(R)))`. A rejected trial, or one whose step is not finite,
sets `μ` to `max(μ · μ_increase, max(μ_min, 4 eps(R)))` and solves again within the same step,
with no new Jacobian, until a trial is accepted or `μ > μ_max`, which returns `STALLED` with the
iterate kept. A step accepted at
a `μ` above `μ₀` sets `limited` in its [`StepInfo`](@ref). The parameters satisfy `0 < μ_min ≤ μ₀ ≤
μ_max < Inf`, `0 < μ_decrease < 1 < μ_increase`, `0 ≤ ρ_accept < 1` and `scaling ∈ (:none,
:diagonal)`, else the constructor raises an `ArgumentError`; [`ToReal`](@ref) converts them to the
real type of the iterate. `Newton(; globalization = LevenbergMarquardt(), reuse = k)` raises an
`ArgumentError`.
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

# The workspace of `LevenbergMarquardt` at an iterate `x`: the trial buffers that every path uses,
# and the buffers of the stacked solve, which only the QR and LU paths and Marquardt's scaling
# read. The SVD step reads none of them, so the stacked slots are allocated only where they are
# used; a device iterate never asks for them, because `init_state` refuses the stacked step before
# this function runs (see `check_lm_iterate`).
@inline function init_globalization(lm::LevenbergMarquardt, linsolve::LinearMethod, x)
    T = eltype(x)
    R = real(T)
    n = x isa Number ? 1 : length(x)
    stacked = !lm_svd_path(linsolve, lm.scaling)
    return (; Jδ = buffer(x), xt = buffer(x), Ft = buffer(x),
        R₀ = stacked ? buffer(x, T, n, n) : nothing,
        c₀ = stacked ? buffer(x, T, n) : nothing,
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
# accepted and every rejected trial, so a `μ` below that floor arrives only as the initial `μ₀`
# that `start!!` stores, and the schedule never returns to it. It is taken as no damping: the
# truncated Newton step, which is the `μ → 0` limit of the damped one.
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

# The diagonal `D` of the damping term for the stacked solve of `[J; √μ D] δ = [-Fx; 0]`, written
# in place. `scaling = :none` takes `D = I`, so the damping is `√μ` in the units of `Jᴴ J`, the
# same number the SVD weight `σ_j/(σ_j² + μ)` divides by. `scaling = :diagonal` takes
# Marquardt's column norms, floored at `√eps(R)` of the largest of them, so that a zero or
# near-zero column still receives positive damping; it is scale-invariant on its own.
#
# The caller folds `√μ` into the diagonal; a damping entry that is not representable is a failed
# trial.
@inline function lm_damping_diagonal!(
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

@inline function lm_damping_diagonal!(
        d::AbstractVector, ::AbstractMatrix, ::Val{:none})
    return fill!(d, one(real(eltype(d))))
end

# The same diagonal of a static matrix, as a value: the damping of the R1 stacked solve. Inlined,
# as the stacked routines are, so that the `MVector` it fills stays in the caller's frame.
@inline function lm_damping_diagonal(
        J::StaticMatrix{N, N, T}, scaling::Val{SC}) where {N, T, SC}
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
# Givens rotations on its rows. Inlined because the caller's `MMatrix` buffers must not escape a
# call for the R1 step to stay off the heap.
@inline function lm_qr_factor!(
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
            G, r_val = givens(a, b, 1, 2)
            cs = G.c
            sn = G.s
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
# and zeroes `δ` where a pivot is zero or not finite. Inlined because the caller's `MMatrix`
# buffers must not escape a call for the R1 step to stay off the heap.
@inline function lm_qr_solve!(
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
            G, r_val = givens(a, b, 1, 2)
            cs = G.c
            sn = G.s
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

# The reduction of `[J | -F]` is the `O(n³)` factorisation that an outer step of the stacked path
# does once, and each trial then pays only the elimination of its own damping rows. A test counts
# the reduction through a `LinearMethod` of its own, as it counts a factorisation through
# `factorize!!`.
@inline lm_stacked_factor!(::LinearMethod, R₀, c₀, J, Fx) = lm_qr_factor!(R₀, c₀, J, Fx)

# The workspace of the stacked solve for a vector iterate: the factor `(R₀, c₀)` of `[J | -F]` and
# the damping diagonal, as values. It is immutable, so it crosses a call and a loop iteration
# without a heap array, and each trial builds the copies it eliminates in the call that reads them.
# `L` is the length parameter of the static arrays the factor is stored in: without it the field
# type is the `SMatrix{N, N, T}` alias with `L` free, which is not concrete, and the workspace
# would not be `isbits`.
struct LMStacked{N, T, L, D}
    R₀::SMatrix{N, N, T, L}
    c₀::SVector{N, T}
    d_diag::D
end

# The same workspace for a scalar iterate, whose step is the scalar `δ` of a 1×1 system. That step
# is written out below rather than routed through `lm_qr_solve!`: at `N = 1` the elimination and
# the back-substitution of that routine are one rotation and one division, and the scalar path
# needs none of its four buffers.
struct LMScalar{T, D}
    R₀::SMatrix{1, 1, T, 1}
    c₀::SVector{1, T}
    d_diag::D
end

# The workspace of the stacked solve for a static vector iterate, factored at the start of an outer
# step.
@inline function lm_qr_prepare(
        linsolve, ::Nothing, g, J::StaticMatrix{N, N, T}, Fx::StaticVector{N, T},
        scaling::Union{Val{:none}, Val{:diagonal}}) where {N, T}
    R₀ = local_array(zero(SMatrix{N, N, T}))
    c₀ = local_array(zero(SVector{N, T}))
    lm_stacked_factor!(linsolve, R₀, c₀, J, Fx)
    d_diag = lm_damping_diagonal(J, scaling)
    return LMStacked(
        SMatrix{N, N, T}(R₀), SVector{N, T}(c₀), d_diag)
end

# A scalar iterate is the 1×1 case of the same reduction.
@inline function lm_qr_prepare(
        linsolve, ::Nothing, g, J::T, Fx::T,
        scaling::Union{Val{:none}, Val{:diagonal}}) where {T <: Number}
    Jm = SMatrix{1, 1, T}((J,))
    Fv = SVector{1, T}((Fx,))
    R₀ = local_array(zero(SMatrix{1, 1, T}))
    c₀ = local_array(zero(SVector{1, T}))
    lm_stacked_factor!(linsolve, R₀, c₀, Jm, Fv)
    d_diag = lm_damping_diagonal(Jm, scaling)
    return LMScalar{T, typeof(d_diag)}(
        SMatrix{1, 1, T}(R₀), SVector{1, T}(c₀), d_diag)
end

# The same workspace for an array iterate, whose buffers `init_globalization` holds in the state.
@inline function lm_qr_prepare(
        linsolve, δ::AbstractVector, g, J::AbstractMatrix, Fx::AbstractVector,
        scaling::Union{Val{:none}, Val{:diagonal}})
    lm_damping_diagonal!(g.d_diag, J, scaling)
    lm_stacked_factor!(linsolve, g.R₀, g.c₀, J, Fx)
    return (; R₀ = g.R₀, c₀ = g.c₀, R = g.R, c = g.c, w = g.w, δ, d_diag = g.d_diag)
end

# The workspace of the trial loop for the path that reads the factor of `J(x)` it already has: the
# SVD step needs no reduction, only the factor, the residual at `x` and the direction slot.
struct LMSvd{F, D, X}
    fac::F
    δ::D
    Fx::X
end

# The workspace of the trial loop of an outer step: the stored SVD factor on the one path that needs
# no reduction, and the reduction of `[J | -F]` that the stacked path does once.
@inline function lm_trial_workspace(linsolve, scaling, d, g, fac, J, Fx)
    lm_svd_path(linsolve, scaling) && return LMSvd(fac, d, Fx)
    return lm_qr_prepare(linsolve, d, g, J, Fx, scaling)
end

# One trial of the stacked solve for a static vector iterate: the damping rows `√μ d_j e_jᵀ`
# eliminated against a copy of the stored factor by Givens rotations, on `MMatrix` copies local to
# this call. `false` where the damping is not representable or the damped factor is singular, and
# `δ` is then zero.
@inline function lm_trial(p::LMStacked{N, T}, μ::R) where {N, T, R <: Real}
    sqrt_μ = sqrt(lm_step_damping(μ))
    lm_damping_ok(sqrt_μ, p.d_diag) || return zero(SVector{N, T}), false
    Rm = local_array(zero(SMatrix{N, N, T}))
    cm = local_array(zero(SVector{N, T}))
    w = local_array(zero(SVector{N, T}))
    δ = local_array(zero(SVector{N, T}))
    ok = lm_qr_solve!(δ, Rm, cm, w, p.R₀, p.c₀, sqrt_μ, p.d_diag)
    return SVector{N, T}(δ), ok
end

# The same trial for a scalar iterate: `[J; √μ d] δ = [-F; 0]` is one rotation of the pair, and
# the rotated right-hand side over the rotated pivot is `δ`. A zero pivot, or a `δ` that overflows,
# is a failed trial, as on the other paths.
@inline function lm_trial(p::LMScalar{T}, μ::R) where {T, R <: Real}
    sqrt_μ = sqrt(lm_step_damping(μ))
    lm_damping_ok(sqrt_μ, p.d_diag) || return zero(T), false
    G, r = givens(p.R₀[1, 1], T(sqrt_μ * p.d_diag[1]), 1, 2)
    δ = G.c * p.c₀[1] / r
    return δ, isfinite(δ)
end

# The same trial for an array iterate, whose buffers the state holds.
@inline function lm_trial(p::NamedTuple, μ::R) where {R <: Real}
    sqrt_μ = sqrt(lm_step_damping(μ))
    if !lm_damping_ok(sqrt_μ, p.d_diag)
        fill!(p.δ, zero(eltype(p.δ)))
        return p.δ, false
    end
    return p.δ, lm_qr_solve!(p.δ, p.R, p.c, p.w, p.R₀, p.c₀, sqrt_μ, p.d_diag)
end

# The same trial on the SVD path: the damped step over the truncated rank of the stored factor.
@inline function lm_trial(p::LMSvd, μ::R) where {R <: Real}
    δ = lm_svd_step(p.δ, p.fac, p.Fx, μ)
    return δ, (δ isa Number ? isfinite(δ) : all(isfinite, δ))
end
