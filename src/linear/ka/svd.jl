# KernelAbstractions parallel round-robin one-sided Jacobi SVD for DenseSVD on AbstractGPUArray.
#
# Reuses `JACOBI_MAX_SWEEPS`, `jacobi_tolerance`, `jacobi_rotation` and `jacobi_pair` from
# `src/linear/static/svd.jl`. `fac.U` holds the working columns during the sweep, `fac.factors`
# holds the right singular vectors `V`, `fac.work` is the 1-element `Int32` rotation flag, and
# `fac.rwork` holds the column norms squared during the sweep and the sorted scaled singular values
# `σ̂` of `J / 2^e` after it.

@inline function ka_rotate_columns!(A, n::Int, p::Int, q::Int, c, s)
    sc = conj(s)
    @inbounds for i in 1:n
        u = A[i, p]
        v = A[i, q]
        A[i, p] = c * u - sc * v
        A[i, q] = s * u + c * v
    end
    return nothing
end

@kernel function ka_jacobi_col_norms2_kernel!(
        rwork, work, U, V, n::Int, neg_e::Int, init_v::Bool, take_sqrt::Bool)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(U)
        R = eltype(rwork)
        if init_v
            for i in 1:n
                if neg_e != 0
                    U[i, j] = ldexp2(U[i, j], neg_e)
                end
                V[i, j] = ifelse(i == j, one(F), zero(F))
            end
        end
        s = zero(R)
        for i in 1:n
            s += abs2(U[i, j])
        end
        rwork[j] = ifelse(take_sqrt, sqrt(s), s)
        if !take_sqrt && j == 1
            work[1] = Int32(0)
        end
    end
end

@kernel function ka_jacobi_svd_round_kernel!(U, V, work, n::Int, r::Int, tol, negligible)
    k = @index(Global, Linear)
    m = n + Int(isodd(n))
    @inbounds if k <= m ÷ 2
        p, q = jacobi_pair(n, r, k - 1)
        if p <= n
            F = eltype(U)
            R = real(F)
            α = zero(R)
            β = zero(R)
            γ = zero(F)
            for i in 1:n
                up = U[i, p]
                uq = U[i, q]
                α += abs2(up)
                β += abs2(uq)
                γ += conj(up) * uq
            end
            if !(α <= negligible || β <= negligible) && !(abs(γ) <= tol * sqrt(α * β))
                c, s = jacobi_rotation(α, β, γ)
                ka_rotate_columns!(U, n, p, q, c, s)
                ka_rotate_columns!(V, n, p, q, c, s)
                work[1] = Int32(1)
            end
        end
    end
end

@kernel function ka_jacobi_sort_columns_kernel!(rwork, U, V, n::Int)
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        for p in 1:(n - 1)
            q_max = p
            s_max = rwork[p]
            for q in (p + 1):n
                sq = rwork[q]
                if sq > s_max
                    q_max = q
                    s_max = sq
                end
            end
            if q_max != p
                rwork[q_max] = rwork[p]
                rwork[p] = s_max
                for i in 1:n
                    up = U[i, p]
                    U[i, p] = U[i, q_max]
                    U[i, q_max] = up
                    vp = V[i, p]
                    V[i, p] = V[i, q_max]
                    V[i, q_max] = vp
                end
            end
        end
    end
end

@kernel function ka_jacobi_finalize_svd_kernel!(
        U, S, VT, @Const(V), @Const(rwork), n::Int, e::Int, tol)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(U)
        σj = rwork[j]
        S[j] = ldexp2(σj, e)
        if σj > tol * rwork[1]
            inv_σ = inv(σj)
            for i in 1:n
                U[i, j] = inv_σ * U[i, j]
            end
        else
            for i in 1:n
                U[i, j] = zero(F)
            end
        end
        for i in 1:n
            VT[j, i] = conj(V[i, j])
        end
    end
end

@inline ka_kernel_backend(backend) = ka_kernel_backend(Val(isgpu(backend)), backend)
@inline ka_kernel_backend(::Val{true}, backend) = backend
@inline ka_kernel_backend(::Val{false}, _) = KernelAbstractions.CPU()

@inline ka_kernel_array(backend, A::AbstractArray) = ka_kernel_array(Val(isgpu(backend)), A)
@inline ka_kernel_array(::Val{true}, A::AbstractArray) = A
@inline ka_kernel_array(::Val{false}, A::AbstractArray) = unsafe_wrap(Array, pointer(A), size(A))

@inline ka_host_vec(v::Array) = v
@inline ka_host_vec(v::AbstractArray) = Array(v)

function ka_svd!(U::AbstractMatrix{F}, V::AbstractMatrix{F}, S::AbstractVector{R},
        VT::AbstractMatrix{F}, work::AbstractVector{Int32}, rwork::AbstractVector{R},
        e::Int, neg_e::Int) where {F, R}
    n = size(U, 1)
    backend = get_backend(U)
    kbackend = ka_kernel_backend(backend)
    tol = jacobi_tolerance(n, R)
    m = n + Int(isodd(n))
    n_pairs = m ÷ 2
    converged = n <= 1
    GC.@preserve U V S VT work rwork begin
        U_k = ka_kernel_array(backend, U)
        V_k = ka_kernel_array(backend, V)
        VT_k = ka_kernel_array(backend, VT)
        S_k = ka_kernel_array(backend, S)
        work_k = ka_kernel_array(backend, work)
        rwork_k = ka_kernel_array(backend, rwork)
        col_norms2_kernel! = ka_jacobi_col_norms2_kernel!(kbackend)
        if n >= 2
            round_kernel! = ka_jacobi_svd_round_kernel!(kbackend)
            for sweep in 1:JACOBI_MAX_SWEEPS
                col_norms2_kernel!(
                    rwork_k, work_k, U_k, V_k, n, neg_e, sweep == 1, false; ndrange = n)
                # `max` propagates `NaN`, so one reduction tests every column norm for finiteness.
                max_norm2 = maximum(rwork_k)
                isfinite(max_norm2) || break
                negligible = max_norm2 * tol^2
                for r in 0:(m - 2)
                    round_kernel!(
                        U_k, V_k, work_k, n, r, tol, negligible; ndrange = n_pairs)
                end
                if iszero(ka_host_vec(work_k)[1])
                    converged = true
                    break
                end
            end
            col_norms2_kernel!(
                rwork_k, work_k, U_k, V_k, n, 0, false, true; ndrange = n)
        else
            col_norms2_kernel!(
                rwork_k, work_k, U_k, V_k, n, neg_e, true, true; ndrange = n)
        end
        if n >= 2
            ka_jacobi_sort_columns_kernel!(kbackend)(rwork_k, U_k, V_k, n; ndrange = 1)
        end
        ka_jacobi_finalize_svd_kernel!(kbackend)(
            U_k, S_k, VT_k, V_k, rwork_k, n, e, tol; ndrange = n)
        synchronize(kbackend)
        # the status and the rank from one host read of the sorted `σ̂`, and no scalar read of a
        # device array: `info` reports a sweep that did not converge or a singular value that is
        # not finite, and the rank is then 0
        σ̂ = ka_host_vec(rwork_k)
        ok = converged && all(isfinite, σ̂)
        return ok ? 0 : 1, ok ? svd_rank(σ̂) : 0
    end
end

function ka_factorize_svd!(fac::DenseSVD, J::AbstractMatrix)
    n = size(fac.U, 1)
    n == 0 && return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, 0, 0, 0)
    F = eltype(fac.U)
    e = matrix_exponent(J)
    if eltype(J) === F
        J === fac.U || copyto!(fac.U, J)
        neg_e = -e
    else
        fac.U .= ldexp2.(J, -e)
        neg_e = 0
    end
    info, rank = ka_svd!(fac.U, fac.factors, fac.S, fac.VT, fac.work, fac.rwork, e, neg_e)
    return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs,
        info, rank, e)
end

function factorize!!(fac::DenseSVD{<:AbstractGPUArray}, ::SVDFactorization, J::AbstractMatrix)
    return ka_factorize_svd!(fac, J)
end

@kernel function ka_svd_ut_mul_kernel!(tmp, @Const(U), @Const(d), n::Int, rank::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(tmp)
        if j <= rank
            acc = zero(F)
            for i in 1:n
                acc += conj(U[i, j]) * d[i]
            end
            tmp[j] = acc
        else
            tmp[j] = zero(F)
        end
    end
end

@kernel function ka_svd_divide_kernel!(tmp, @Const(rwork), rank::Int, t::Int)
    j = @index(Global, Linear)
    @inbounds if j <= rank
        tmp[j] = tmp[j] / (iszero(t) ? rwork[j] : ldexp2(rwork[j], t))
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
# `rwork` once the kernel that writes `tmp` has run.
function ka_divisor_exponent(kbackend, fac::DenseSVD, ::Type{T}, n::Int) where {T}
    synchronize(kbackend)
    c = coordinate_bound(T, n)
    q = mapreduce((y, σ) -> coordinate_ratio(y, σ, c), max,
        view(fac.tmp, 1:(fac.rank)), view(fac.rwork, 1:(fac.rank)); init = zero(real(T)))
    return shift_exponent(q)
end

# The solve with the factors of `J / 2^e` divides by their singular values `σ̂` in `rwork`, the
# same-type solve by `σ̂ 2^t`, `t` the `divisor_exponent` of the coordinates `Uᴴ d`; returns `t`.
function ka_svd_backsolve!(fac::DenseSVD, d::AbstractVector{T}, shift::Bool) where {T <:
                                                                                    Number}
    n = length(d)
    n == 0 && return 0
    # a failed factor, whose `σ̂` can hold a finite leading value beside a `NaN`, and a zero
    # matrix give the zero step; `rank` is a host field, so this reads no device array
    iszero(fac.rank) && (fill!(d, zero(T)); return 0)
    backend = get_backend(d)
    kbackend = ka_kernel_backend(backend)
    t = 0
    GC.@preserve fac d begin
        tmp_k = ka_kernel_array(backend, fac.tmp)
        U_k = ka_kernel_array(backend, fac.U)
        VT_k = ka_kernel_array(backend, fac.VT)
        rwork_k = ka_kernel_array(backend, fac.rwork)
        d_k = ka_kernel_array(backend, d)
        ka_svd_ut_mul_kernel!(kbackend)(tmp_k, U_k, d_k, n, fac.rank; ndrange = n)
        shift && (t = ka_divisor_exponent(kbackend, fac, T, n))
        ka_svd_divide_kernel!(kbackend)(tmp_k, rwork_k, fac.rank, t; ndrange = fac.rank)
        ka_svd_v_mul_kernel!(kbackend)(d_k, VT_k, tmp_k, n; ndrange = n)
    end
    synchronize(kbackend)
    return t
end

function backsolve!(
        fac::DenseSVD{<:AbstractGPUArray{T}}, d::AbstractVector{T}) where {T <: BlasFloat}
    ka_svd_backsolve!(fac, d, false)
    return d
end

function backsolve!(
        fac::DenseSVD{<:AbstractGPUArray{T}}, d::AbstractVector{T}) where {T <: Number}
    ka_svd_backsolve!(fac, d, false)
    return d
end

function backsolve_shifted!(fac::DenseSVD{<:AbstractGPUArray{T}},
        d::AbstractVector{T}) where {T <: BlasFloat}
    return ka_svd_backsolve!(fac, d, true)
end

function backsolve_shifted!(fac::DenseSVD{<:AbstractGPUArray{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_svd_backsolve!(fac, d, true)
end
