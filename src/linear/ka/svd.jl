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

@kernel function ka_jacobi_init_v_kernel!(U, V, @Const(J), n::Int, neg_e::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(U)
        for i in 1:n
            jij = J[i, j]
            U[i, j] = F(ldexp2(jij, neg_e))
            V[i, j] = ifelse(i == j, one(F), zero(F))
        end
    end
end

@kernel function ka_jacobi_col_norms2_kernel!(rwork, work, @Const(U), n::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        R = eltype(rwork)
        s = zero(R)
        for i in 1:n
            s += abs2(U[i, j])
        end
        rwork[j] = s
        if j == 1
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

@kernel function ka_jacobi_col_norms_kernel!(rwork, @Const(U), n::Int)
    j = @index(Global, Linear)
    @inbounds if j <= n
        R = eltype(rwork)
        s = zero(R)
        for i in 1:n
            s += abs2(U[i, j])
        end
        rwork[j] = sqrt(s)
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
        U, VT, S, @Const(V), @Const(rwork), n::Int, tol, e::Int)
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

function ka_factorize_svd!(fac::DenseSVD, J::AbstractMatrix)
    n = size(fac.U, 1)
    n == 0 && return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, 0, 0)
    F = eltype(fac.U)
    R = real(F)
    e = matrix_exponent(J)
    backend = get_backend(fac.U)
    kbackend = ka_kernel_backend(backend)
    tol = jacobi_tolerance(n, R)
    m = n + Int(isodd(n))
    n_pairs = m ÷ 2
    converged = n <= 1
    GC.@preserve fac J begin
        U_k = ka_kernel_array(backend, fac.U)
        V_k = ka_kernel_array(backend, fac.factors)
        VT_k = ka_kernel_array(backend, fac.VT)
        S_k = ka_kernel_array(backend, fac.S)
        work_k = ka_kernel_array(backend, fac.work)
        rwork_k = ka_kernel_array(backend, fac.rwork)
        J_k = ka_kernel_array(backend, J)
        ka_jacobi_init_v_kernel!(kbackend)(U_k, V_k, J_k, n, -e; ndrange = n)
        if n >= 2
            col_norms2_kernel! = ka_jacobi_col_norms2_kernel!(kbackend)
            round_kernel! = ka_jacobi_svd_round_kernel!(kbackend)
            for _ in 1:JACOBI_MAX_SWEEPS
                col_norms2_kernel!(rwork_k, work_k, U_k, n; ndrange = n)
                max_norm2 = maximum(fac.rwork)
                negligible = (eps(R) * sqrt(max_norm2))^2
                for r in 0:(m - 2)
                    round_kernel!(
                        U_k, V_k, work_k, n, r, tol, negligible; ndrange = n_pairs)
                end
                if iszero(Array(fac.work)[1])
                    converged = true
                    break
                end
            end
        end
        ka_jacobi_col_norms_kernel!(kbackend)(rwork_k, U_k, n; ndrange = n)
        if n >= 2
            ka_jacobi_sort_columns_kernel!(kbackend)(rwork_k, U_k, V_k, n; ndrange = 1)
        end
        ka_jacobi_finalize_svd_kernel!(kbackend)(
            U_k, VT_k, S_k, V_k, rwork_k, n, tol, e; ndrange = n)
    end
    isgpu(backend) && synchronize(backend)
    info = converged ? 0 : 1
    return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, info, e)
end

function factorize!!(fac::DenseSVD{<:AbstractGPUArray}, ::SVDFactorization, J::AbstractMatrix)
    return ka_factorize_svd!(fac, J)
end

@kernel function ka_svd_ut_mul_scale_kernel!(
        tmp, @Const(U), @Const(rwork), @Const(d), n::Int, tol)
    j = @index(Global, Linear)
    @inbounds if j <= n
        F = eltype(tmp)
        σj = rwork[j]
        if σj > tol * rwork[1]
            acc = zero(F)
            for i in 1:n
                acc += conj(U[i, j]) * d[i]
            end
            tmp[j] = acc / σj
        else
            tmp[j] = zero(F)
        end
    end
end

@kernel function ka_svd_v_mul_kernel!(d, @Const(VT), @Const(tmp), n::Int)
    i = @index(Global, Linear)
    @inbounds if i <= n
        F = eltype(d)
        acc = zero(F)
        for j in 1:n
            acc += conj(VT[j, i]) * tmp[j]
        end
        d[i] = acc
    end
end

function ka_svd_backsolve!(fac::DenseSVD, d::AbstractVector{T}) where {T <: Number}
    n = length(d)
    n == 0 && return d
    R = real(T)
    tol = jacobi_tolerance(n, R)
    backend = get_backend(d)
    kbackend = ka_kernel_backend(backend)
    GC.@preserve fac d begin
        tmp_k = ka_kernel_array(backend, fac.tmp)
        U_k = ka_kernel_array(backend, fac.U)
        VT_k = ka_kernel_array(backend, fac.VT)
        rwork_k = ka_kernel_array(backend, fac.rwork)
        d_k = ka_kernel_array(backend, d)
        ka_svd_ut_mul_scale_kernel!(kbackend)(tmp_k, U_k, rwork_k, d_k, n, tol; ndrange = n)
        ka_svd_v_mul_kernel!(kbackend)(d_k, VT_k, tmp_k, n; ndrange = n)
    end
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(
        fac::DenseSVD{<:AbstractGPUArray{T}}, d::AbstractVector{T}) where {T <: BlasFloat}
    return ka_svd_backsolve!(fac, d)
end

function backsolve!(
        fac::DenseSVD{<:AbstractGPUArray{T}}, d::AbstractVector{T}) where {T <: Number}
    return ka_svd_backsolve!(fac, d)
end
