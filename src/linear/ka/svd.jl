# KernelAbstractions parallel round-robin one-sided Jacobi SVD for DeviceSVD on AbstractGPUArray.
#
# Reuses `JACOBI_MAX_SWEEPS`, `jacobi_tolerance`, `jacobi_rotation` and `jacobi_pair` from
# `src/linear/static/svd.jl`. `fac.U` holds the working columns during the sweep, `fac.V` holds
# the right singular vectors `V`, `fac.flag` is the 1-element `Int32` rotation flag, and `fac.σ̂`
# holds the squared column norms during the sweep and the sorted scaled singular values `σ̂` of
# `J / 2^e` after it.

using KernelAbstractions: @localmem, @synchronize

@kernel function ka_jacobi_col_norms2_kernel!(
        σ̂, flag, U, V, m_rows::Int, n::Int, neg_e::Int, init_v::Bool, take_sqrt::Bool)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(U)
        R = eltype(σ̂)
        if init_v
            if neg_e != 0
                for i in 1:m_rows
                    U[i, j] = ldexp2(U[i, j], neg_e)
                end
            end
            for i in 1:n
                V[i, j] = ifelse(i == j, one(F), zero(F))
            end
        end
        s = zero(R)
        for i in 1:m_rows
            s += abs2(U[i, j])
        end
        σ̂[j] = ifelse(take_sqrt, sqrt(s), s)
        if !take_sqrt && j == 1
            flag[1] = Int32(0)
        end
    end
end

@kernel function ka_jacobi_svd_round_kernel!(
        U, V, flag, m_rows::Int, n::Int, r::Int, tol, negligible, ::Val{G}) where {G}
    pair = @index(Group, Linear)
    tid = @index(Local, Linear)
    l_α = @localmem real(eltype(U)) (G,)
    l_β = @localmem real(eltype(U)) (G,)
    l_γ = @localmem eltype(U) (G,)
    l_c = @localmem real(eltype(U)) (1,)
    l_s = @localmem eltype(U) (1,)
    l_rot = @localmem Int32 (1,)
    @inbounds begin
        F = eltype(U)
        R = real(F)
        p, q = jacobi_pair(n, r, pair - 1)
        α_loc = zero(R)
        β_loc = zero(R)
        γ_loc = zero(F)
        if p <= n
            i = tid
            while i <= m_rows
                up = U[i, p]
                uq = U[i, q]
                α_loc += abs2(up)
                β_loc += abs2(uq)
                γ_loc += conj(up) * uq
                i += G
            end
        end
        l_α[tid] = α_loc
        l_β[tid] = β_loc
        l_γ[tid] = γ_loc
    end
    @synchronize()
    for step in 1:trailing_zeros(G)
        s = G >> step
        @inbounds if tid <= s
            l_α[tid] += l_α[tid + s]
            l_β[tid] += l_β[tid + s]
            l_γ[tid] += l_γ[tid + s]
        end
        @synchronize()
    end
    @inbounds if tid == 1
        p, _ = jacobi_pair(n, r, pair - 1)
        α = l_α[1]
        β = l_β[1]
        γ = l_γ[1]
        if p <= n && !(α <= negligible || β <= negligible) &&
           !(abs(γ) <= tol * sqrt(α * β))
            c, s = jacobi_rotation(α, β, γ)
            l_c[1] = c
            l_s[1] = s
            l_rot[1] = Int32(1)
            flag[1] = Int32(1)
        else
            l_rot[1] = Int32(0)
        end
    end
    @synchronize()
    @inbounds if l_rot[1] == Int32(1)
        p, q = jacobi_pair(n, r, pair - 1)
        c = l_c[1]
        s = l_s[1]
        sc = conj(s)
        i = tid
        while i <= m_rows
            up = U[i, p]
            uq = U[i, q]
            U[i, p] = c * up - sc * uq
            U[i, q] = s * up + c * uq
            i += G
        end
        j = tid
        while j <= n
            vp = V[j, p]
            vq = V[j, q]
            V[j, p] = c * vp - sc * vq
            V[j, q] = s * vp + c * vq
            j += G
        end
    end
end

@kernel function ka_jacobi_sort_columns_kernel!(σ̂, U, V, m_rows::Int, n::Int)
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        for p in 1:(n - 1)
            q_max = p
            s_max = σ̂[p]
            for q in (p + 1):n
                sq = σ̂[q]
                if sq > s_max
                    q_max = q
                    s_max = sq
                end
            end
            if q_max != p
                σ̂[q_max] = σ̂[p]
                σ̂[p] = s_max
                for i in 1:m_rows
                    up = U[i, p]
                    U[i, p] = U[i, q_max]
                    U[i, q_max] = up
                end
                for i in 1:n
                    vp = V[i, p]
                    V[i, p] = V[i, q_max]
                    V[i, q_max] = vp
                end
            end
        end
    end
end

@kernel function ka_jacobi_finalize_svd_kernel!(
        U, S, VT, @Const(V), @Const(σ̂), m_rows::Int, n::Int, e::Int, tol)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(U)
        σj = σ̂[j]
        S[j] = ldexp2(σj, e)
        if σj > tol * σ̂[1]
            for i in 1:m_rows
                U[i, j] = U[i, j] / σj
            end
        else
            for i in 1:m_rows
                U[i, j] = zero(F)
            end
        end
        for i in 1:n
            VT[j, i] = conj(V[i, j])
        end
    end
end

Base.@assume_effects :foldable function ka_svd_workgroup(::B) where {B}
    return nameof(B) === :JLBackend ? Val(2) : Val(32)
end

ka_unwrap_val(::Val{N}) where {N} = N

function ka_svd!(U::AbstractGPUArray{F, 2}, V::AbstractMatrix{F}, S::AbstractVector{R},
        VT::AbstractMatrix{F}, flag::AbstractVector{Int32}, σ̂::AbstractVector{R},
        e::Int, neg_e::Int) where {F, R}
    m_rows, n = size(U)
    backend = get_backend(U)
    tol = jacobi_tolerance(m_rows, R)
    cutoff_tol = jacobi_tolerance(n, R)
    m = n + Int(isodd(n))
    n_pairs = m ÷ 2
    valG = ka_svd_workgroup(backend)
    g = ka_unwrap_val(valG)
    converged = n <= 1
    col_norms2_kernel! = ka_jacobi_col_norms2_kernel!(backend)
    if n >= 2
        round_kernel! = ka_jacobi_svd_round_kernel!(backend, g)
        for sweep in 1:JACOBI_MAX_SWEEPS
            col_norms2_kernel!(
                σ̂, flag, U, V, m_rows, n, neg_e, sweep == 1, false; ndrange = n)
            # `max` propagates `NaN`, so one reduction tests every column norm for finiteness.
            max_norm2 = maximum(σ̂)
            isfinite(max_norm2) || break
            negligible = max_norm2 * tol^2
            for r in 0:(m - 2)
                round_kernel!(
                    U, V, flag, m_rows, n, r, tol, negligible, valG; ndrange = n_pairs * g)
            end
            if iszero(Array(flag)[1])
                converged = true
                break
            end
        end
        col_norms2_kernel!(
            σ̂, flag, U, V, m_rows, n, 0, false, true; ndrange = n)
    else
        col_norms2_kernel!(
            σ̂, flag, U, V, m_rows, n, neg_e, true, true; ndrange = n)
    end
    if n >= 2
        ka_jacobi_sort_columns_kernel!(backend)(σ̂, U, V, m_rows, n; ndrange = 1)
    end
    ka_jacobi_finalize_svd_kernel!(backend)(
        U, S, VT, V, σ̂, m_rows, n, e, cutoff_tol; ndrange = n)
    isgpu(backend) && synchronize(backend)
    # the status and the rank from one host read of the sorted `σ̂`, and no scalar read of a
    # device array: `info` reports a sweep that did not converge or a singular value that is
    # not finite, and the rank is then 0
    σ̂_host = Array(σ̂)
    ok = converged && all(isfinite, σ̂_host)
    return ok ? 0 : 1, ok ? svd_rank(σ̂_host) : 0
end

function ka_factorize_svd!(fac::DeviceSVD, J::AbstractMatrix)
    n = size(fac.U, 2)
    n == 0 && return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V, fac.σ̂, fac.flag, fac.tmp, fac.rhs,
        0, 0, 0)
    F = eltype(fac.U)
    e = matrix_exponent(J)
    if eltype(J) === F
        J === fac.U || copyto!(fac.U, J)
        neg_e = -e
    else
        fac.U .= ldexp2.(J, -e)
        neg_e = 0
    end
    info, rank = ka_svd!(fac.U, fac.V, fac.S, fac.VT, fac.flag, fac.σ̂, e, neg_e)
    return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V, fac.σ̂, fac.flag, fac.tmp, fac.rhs,
        info, rank, e)
end

function factorize!!(fac::DeviceSVD, ::SVDFactorization, J::AbstractMatrix)
    return ka_factorize_svd!(fac, J)
end

@kernel function ka_svd_ut_mul_kernel!(
        tmp, @Const(U), @Const(r), m_rows::Int, n::Int, rank::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(tmp)
        if j <= rank
            acc = zero(F)
            for i in 1:m_rows
                acc += conj(U[i, j]) * r[i]
            end
            tmp[j] = acc
        else
            tmp[j] = zero(F)
        end
    end
end

@kernel function ka_svd_divide_kernel!(tmp, @Const(σ̂), rank::Int, t::Int)
    j = @index(Global, Linear)
    @inbounds if j <= rank
        tmp[j] = tmp[j] / (iszero(t) ? σ̂[j] : ldexp2(σ̂[j], t))
    end
end

@kernel function ka_svd_v_mul_kernel!(d, @Const(VT), @Const(tmp), n::Int)
    i = @index(Global, Linear)
    @inbounds if i <= n
        acc = zero(eltype(d))
        for j in 1:n
            acc += conj(VT[j, i]) * tmp[j]
        end
        d[i] = acc
    end
end

# The `divisor_exponent` of the coordinates `Uᴴ d` in `tmp`, from one reduction over `tmp` and
# `σ̂` once the kernel that writes `tmp` has run. The reduction runs over the whole of both, of
# length `n` on a device, and not over views of their first `rank` entries, so that only device
# arrays reach it: past the rank `tmp` holds the zero that the kernel wrote, which adds nothing.
function ka_divisor_exponent(
        fac::DeviceSVD{<:AbstractGPUArray}, ::Type{T}, n::Int) where {T}
    backend = get_backend(fac.U)
    isgpu(backend) && synchronize(backend)
    c = coordinate_bound(T, n)
    R = real(T)
    q = mapreduce((y, σ) -> iszero(y) ? zero(σ) : coordinate_ratio(y, σ, c), max,
        fac.tmp, fac.σ̂; init = zero(R))
    return shift_exponent(q)
end

# The solve with the factors of `J / 2^e` divides by their singular values `σ̂`, the
# same-type solve by `σ̂ 2^t`, `t` the `divisor_exponent` of the coordinates `Uᴴ r`; returns `t`.
function ka_svd_backsolve!(
        fac::DeviceSVD{<:AbstractGPUArray}, d::AbstractVector{T},
        r::AbstractVector{T}, shift::Bool) where {T <: Number}
    m_rows, n = size(fac.U)
    n == 0 && return 0
    # a failed factor, whose `σ̂` can hold a finite leading value beside a `NaN`, and a zero
    # matrix give the zero step; `rank` is a host field, so this reads no device array
    iszero(fac.rank) && (fill!(d, zero(T)); return 0)
    backend = get_backend(fac.U)
    ka_svd_ut_mul_kernel!(backend)(fac.tmp, fac.U, r, m_rows, n, fac.rank; ndrange = n)
    t = shift ? ka_divisor_exponent(fac, T, n) : 0
    ka_svd_divide_kernel!(backend)(fac.tmp, fac.σ̂, fac.rank, t; ndrange = fac.rank)
    ka_svd_v_mul_kernel!(backend)(d, fac.VT, fac.tmp, n; ndrange = n)
    isgpu(backend) && synchronize(backend)
    return t
end

function backsolve!(
        fac::DeviceSVD{<:AbstractMatrix{T}}, d::AbstractVector{T}) where {T <: Number}
    ka_svd_backsolve!(fac, d, d, false)
    return d
end

function backsolve_shifted!(fac::DeviceSVD{<:AbstractMatrix{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_svd_backsolve!(fac, d, d, true)
end

function backsolve_shifted!(fac::DeviceSVD{<:AbstractMatrix{T}},
        d::AbstractVector{T}, rhs::AbstractVector{T}) where {T <: Number}
    return ka_svd_backsolve!(fac, d, rhs, true)
end

@kernel function ka_svd_null_residual_kernel!(
        out, r, @Const(U), @Const(tmp), m_rows::Int, rank::Int)
    i = @index(Global, Linear)
    @inbounds if i <= m_rows
        acc = zero(eltype(out))
        for j in 1:rank
            acc += U[i, j] * tmp[j]
        end
        out[i] = r[i] - acc
    end
end

function ka_svd_nullnorm!(
        out::AbstractVector{TF}, fac::DeviceSVD{<:AbstractGPUArray{TF, 2}},
        r::AbstractVector{TF}, rk::Int) where {TF}
    m_rows, n = size(fac.U)
    m_rows == 0 && return zero(real(TF))
    backend = get_backend(fac.U)
    ka_svd_ut_mul_kernel!(backend)(fac.tmp, fac.U, r, m_rows, n, rk; ndrange = n)
    ka_svd_null_residual_kernel!(backend)(
        out, r, fac.U, fac.tmp, m_rows, rk; ndrange = m_rows)
    isgpu(backend) && synchronize(backend)
    return rnorm(out)
end

function factor_diagnostics(
        d::AbstractVector{T}, fac::DeviceSVD{<:AbstractMatrix{TF}},
        r::AbstractVector{T}, fnorm::R) where {T, TF, R <: Real}
    fac.info == 0 || return Int32(-1), R(NaN)
    rk = fac.rank
    rk == 0 && return Int32(0), fnorm
    if TF === T
        out_buf = length(d) == length(r) ? d : fac.rhs::AbstractVector{TF}
        return Int32(rk), R(ka_svd_nullnorm!(out_buf, fac, r, rk))
    end
    s = maximum(abs, r)
    iszero(s) && return Int32(rk), zero(R)
    rhs = fac.rhs::AbstractVector{TF}
    rhs .= TF.(r ./ s)
    return Int32(rk), R(ka_svd_nullnorm!(rhs, fac, rhs, rk)) * R(s)
end
