"""
    LevenbergMarquardt{R, S}

Adaptive Levenberg–Marquardt damping for [`Newton`](@ref), computing the trial step from the
damped linear model `(J(x)ᴴ J(x) + μ Dᵀ D) δ = -J(x)ᴴ F(x)` with damping parameter `μ > 0`
adapted from the gain ratio

```math
\\rho = \\frac{\\|F(x)\\|_2^2 - \\|F(x + \\delta)\\|_2^2}{\\|F(x)\\|_2^2 - \\|F(x) + J(x)\\,\\delta\\|_2^2}.
```

With [`SVDFactorization`](@ref) and `scaling = :none`, `J(x) = U Σ Vᴴ` is factorised once per
outer step and each trial `μ` evaluates the damped pseudoinverse step from `(U, Σ, V)` without
refactoring `J(x)`. With [`QRFactorization`](@ref) or [`LUFactorization`](@ref) (or when
`scaling = :diagonal`), the damped step is solved via Givens QR elimination of the stacked matrix
`[J(x); √μ D]` without forming `J(x)ᴴ J(x)`.
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

@inline function init_globalization(::LevenbergMarquardt, x)
    T = eltype(x)
    R = real(T)
    n = x isa Number ? 1 : length(x)
    return (; Jδ = buffer(x), xt = buffer(x), Ft = buffer(x),
        Rmat = buffer(x, T, n, n), rhs = buffer(x, T, n),
        d_diag = buffer(x, R, n), w_row = buffer(x, T, n))
end

@inline lm_Jδ(::Nothing) = nothing
@inline lm_Jδ(g) = g.Jδ
@inline lm_xt(::Nothing) = nothing
@inline lm_xt(g) = g.xt
@inline lm_Ft(::Nothing) = nothing
@inline lm_Ft(g) = g.Ft

# Effective regularization damping: when μ < eps(R), squaring keeps the stacked-QR diagonal √μ_reg
# at or below eps(R) while remaining strictly positive.
@inline function lm_effective_mu(μ::R) where {R <: AbstractFloat}
    return μ < eps(R) ? μ * μ : μ
end

# Scale-safe SVD coordinate weight for a singular value σ > 0 and coordinate y = u_jᴴ F(x).
# Scaling by s = max(σ, √μ) avoids overflow and underflow of σ² + μ across the floating-point
# range; when 0 < σ < √floatmin(R) so that σ² underflows to zero in R, rescaling the singular
# equation by σ preserves a finite nonzero damped step coordinate.
@inline function lm_svd_coord(σ::R, y::T, μ::R) where {R <: AbstractFloat, T <: Number}
    μ_reg = lm_effective_mu(μ)
    if σ < sqrt(floatmin(R))
        σ_s = one(R)
        μ_s = sqrt(μ_reg)
        w = σ_s / (σ_s * σ_s + μ_s * μ_s)
        return T(w) * (y / σ)
    end
    s = max(σ, sqrt(μ_reg))
    σ_s = σ / s
    μ_s = sqrt(μ_reg) / s
    w = (σ_s / (σ_s * σ_s + μ_s * μ_s)) / s
    return T(w) * y
end

# Scale-safe one-sided Jacobi SVD for StaticMatrix that preserves columns whose norm is below
# √floatmin(R) or below tol * σ₁ when they do not cancel against a larger column.
function lm_static_svd(
        method::SVDFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    F = factor_eltype(method, T)
    R = real(F)
    Â, e = svd_matrix(F, J)
    A = local_array(Â)
    V = local_array(one(SMatrix{N, N, F}))
    tol = jacobi_tolerance(N, R)
    col_scale = local_array(zero(SVector{N, R}))
    @inbounds for j in 1:N
        col = local_array(zero(SVector{N, F}))
        for i in 1:N
            col[i] = A[i, j]
        end
        col_scale[j] = rnorm(SVector(col))
    end
    sweeps = 0
    converged = false
    m = N + isodd(N)
    while sweeps < JACOBI_MAX_SWEEPS
        sweeps += 1
        rotations = 0
        for r in 0:(m - 2), k in 0:(m ÷ 2 - 1)

            p, q = jacobi_pair(N, r, k)
            p > N && continue
            col_p = local_array(zero(SVector{N, F}))
            col_q = local_array(zero(SVector{N, F}))
            @inbounds for i in 1:N
                col_p[i] = A[i, p]
                col_q[i] = A[i, q]
            end
            np = rnorm(SVector(col_p))
            nq = rnorm(SVector(col_q))
            (iszero(np) || iszero(nq)) && continue
            s_pq = max(np, nq)
            γ = zero(F)
            @inbounds for i in 1:N
                γ += conj(A[i, p] / s_pq) * (A[i, q] / s_pq)
            end
            abs(γ) <= tol * (np / s_pq) * (nq / s_pq) && continue
            α = (np / s_pq)^2
            β = (nq / s_pq)^2
            cs, sn = jacobi_rotation(α, β, γ)
            rotate_columns!(A, p, q, cs, sn)
            rotate_columns!(V, p, q, cs, sn)
            s_init = max(col_scale[p], col_scale[q])
            col_scale[p] = s_init
            col_scale[q] = s_init
            rotations += 1
        end
        if iszero(rotations)
            converged = true
            break
        end
    end
    σ = zero(SVector{N, R})
    for j in 1:N
        col = local_array(zero(SVector{N, F}))
        @inbounds for i in 1:N
            col[i] = A[i, j]
        end
        σ = setindex(σ, rnorm(SVector(col)), j)
    end
    for j in 1:(N - 1)
        k = j
        for i in (j + 1):N
            σ[i] > σ[k] && (k = i)
        end
        k == j && continue
        σj = σ[j]
        σ = setindex(setindex(σ, σ[k], j), σj, k)
        col_scale[j], col_scale[k] = col_scale[k], col_scale[j]
        @inbounds for i in 1:N
            A[i, j], A[i, k] = A[i, k], A[i, j]
            V[i, j], V[i, k] = V[i, k], V[i, j]
        end
    end
    ok = converged && all(isfinite, σ)
    info = ok ? Int32(0) : Int32(1)
    rank = Int32(0)
    if ok
        for j in 1:N
            cutoff = tol * max(col_scale[j], floatmin(R))
            (σ[j] > cutoff || (σ[j] > zero(R) && σ[j] == col_scale[j])) || break
            rank += Int32(1)
        end
    end
    for j in 1:N
        if j <= rank
            @inbounds for i in 1:N
                A[i, j] /= σ[j]
            end
        else
            @inbounds for i in 1:N
                A[i, j] = zero(F)
            end
        end
    end
    return StaticSVD(SMatrix(A), SMatrix(V), σ, info, rank, e)
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
        σ_j = ldexp2(R(fac.σ̂[j]), e)
        c_j = -lm_svd_coord(σ_j, y_j, μ)
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
        σ_j = ldexp2(R(fac.rwork[j]), e)
        c_j = -lm_svd_coord(σ_j, y_j, μ)
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
        σ_j = ldexp2(R(σ̂[j]), e)
        tmp[j] = -F(lm_svd_coord(σ_j, tmp[j], μ))
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

# Stacked QR solve of [J; √μ D] δ = [-Fx; 0] via Householder QR of J followed by Givens
# elimination of √μ D, without forming Jᴴ J.
function lm_stacked_qr_step(
        ::Nothing, J::StaticMatrix{N, N, T}, Fx::StaticVector{N, T},
        μ::R, scaling::Union{Val{:none}, Val{:diagonal}}) where {N, T, R}
    μ_reg = lm_effective_mu(μ)
    sqrt_μ = sqrt(μ_reg)
    Rmat = local_array(SMatrix{N, N, T}(J))
    rhs = local_array(-SVector{N, T}(Fx))
    d_diag = local_array(zero(SVector{N, R}))
    @inbounds for j in 1:N
        if scaling === Val(:diagonal)
            col = local_array(zero(SVector{N, T}))
            for i in 1:N
                col[i] = J[i, j]
            end
            cn = rnorm(SVector(col))
            d_diag[j] = cn > zero(R) ? cn : one(R)
        else
            d_diag[j] = one(R)
        end
    end
    # Reduce Rmat to upper triangular by Givens rotations on rows and apply to rhs.
    @inbounds for j in 1:N
        for i in (j + 1):N
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
    # Eliminate the diagonal damping block √μ_reg * D row by row with Givens rotations.
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
    # Back-substitution on the damped upper-triangular factor Rmat.
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
        δ_buf::StaticVector{N, T}, J::StaticMatrix{N, N, T}, Fx::StaticVector{N, T},
        μ::R, scaling::Union{Val{:none}, Val{:diagonal}}) where {N, T, R}
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

function lm_stacked_qr_step(
        δ::AbstractVector{T}, J::AbstractMatrix{T}, Fx::AbstractVector{T},
        μ::R, scaling::Union{Val{:none}, Val{:diagonal}}, g = nothing) where {T, R}
    n = length(Fx)
    μ_reg = lm_effective_mu(μ)
    sqrt_μ = sqrt(μ_reg)
    Rmat = g === nothing || g.Rmat === nothing ? Matrix{T}(undef, n, n) : g.Rmat
    copyto!(Rmat, J)
    rhs = g === nothing || g.rhs === nothing ? Vector{T}(undef, n) : g.rhs
    @inbounds for i in 1:n
        rhs[i] = -Fx[i]
    end
    d_diag = g === nothing || g.d_diag === nothing ? Vector{R}(undef, n) : g.d_diag
    @inbounds for j in 1:n
        if scaling === Val(:diagonal)
            cn = rnorm(@view(Rmat[:, j]))
            d_diag[j] = cn > zero(R) ? cn : one(R)
        else
            d_diag[j] = one(R)
        end
    end
    @inbounds for j in 1:n
        for i in (j + 1):n
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
    w_row = g === nothing || g.w_row === nothing ? zeros(T, n) : g.w_row
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

@inline function lm_stacked_qr_step(
        δ_buf::Union{Nothing, StaticVector}, J, Fx, μ::R,
        scaling::Union{Val{:none}, Val{:diagonal}}, _) where {R <: AbstractFloat}
    return lm_stacked_qr_step(δ_buf, J, Fx, μ, scaling)
end
