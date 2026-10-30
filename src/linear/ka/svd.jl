# KernelAbstractions parallel round-robin one-sided Jacobi SVD for DenseSVD on AbstractGPUArray.
#
# Reuses `JACOBI_MAX_SWEEPS`, `jacobi_tolerance`, `jacobi_acc_type`, `jacobi_rotation` and
# `jacobi_pair` from `src/linear/static/svd.jl`. `fac.U` holds the working columns during the sweep, `fac.factors`
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
        A = promote_type(F, jacobi_acc_type(R))
        s = zero(jacobi_acc_type(R))
        for i in 1:n
            s += abs2(A(U[i, j]))
        end
        rwork[j] = R(ifelse(take_sqrt, sqrt(s), s))
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
            R = jacobi_acc_type(real(F))
            A = promote_type(F, R)
            α = zero(R)
            β = zero(R)
            γ = zero(A)
            for i in 1:n
                up = A(U[i, p])
                uq = A(U[i, q])
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
                max_norm2 = jacobi_acc_type(R)(maximum(rwork_k))
                isfinite(max_norm2) || break
                negligible = max_norm2 * jacobi_acc_type(R)(tol)^2
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
        return (converged && isfinite(maximum(rwork_k))) ? 0 : 1
    end
end

function ka_factorize_svd!(fac::DenseSVD, J::AbstractMatrix)
    n = size(fac.U, 1)
    n == 0 && return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, 0, 0)
    F = eltype(fac.U)
    e = matrix_exponent(J)
    if eltype(J) === F
        J === fac.U || copyto!(fac.U, J)
        neg_e = -e
    else
        fac.U .= ldexp2.(J, -e)
        neg_e = 0
    end
    info = ka_svd!(fac.U, fac.factors, fac.S, fac.VT, fac.work, fac.rwork, e, neg_e)
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
        A = promote_type(F, jacobi_acc_type(real(F)))
        σj = rwork[j]
        if σj > tol * rwork[1]
            acc = zero(A)
            for i in 1:n
                acc += conj(A(U[i, j])) * A(d[i])
            end
            tmp[j] = F(acc / σj)
        else
            tmp[j] = zero(F)
        end
    end
end

@kernel function ka_svd_v_mul_kernel!(d, @Const(VT), @Const(tmp), n::Int)
    i = @index(Global, Linear)
    @inbounds if i <= n
        F = eltype(d)
        A = promote_type(F, jacobi_acc_type(real(F)))
        acc = zero(A)
        for j in 1:n
            acc += conj(A(VT[j, i])) * A(tmp[j])
        end
        d[i] = F(acc)
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
    synchronize(kbackend)
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

if ccall(:jl_generating_output, Cint, ()) == 1
    let kbackend = KernelAbstractions.CPU(), work_k = zeros(Int32, 1)
        for F in (Float32, Float64, ComplexF32, ComplexF64, Float16)
            R = real(F)
            U_k = zeros(F, 2, 2)
            V_k = zeros(F, 2, 2)
            VT_k = zeros(F, 2, 2)
            S_k = zeros(R, 2)
            rwork_k = zeros(R, 2)
            tmp_k = zeros(F, 2)
            d_k = zeros(F, 2)
            tol = jacobi_tolerance(2, R)
            ka_jacobi_col_norms2_kernel!(kbackend)(
                rwork_k, work_k, U_k, V_k, 2, 0, true, false; ndrange = 2)
            ka_jacobi_svd_round_kernel!(kbackend)(
                U_k, V_k, work_k, 2, 0, tol, tol * tol; ndrange = 1)
            ka_jacobi_sort_columns_kernel!(kbackend)(rwork_k, U_k, V_k, 2; ndrange = 1)
            ka_jacobi_finalize_svd_kernel!(kbackend)(
                U_k, S_k, VT_k, V_k, rwork_k, 2, 0, tol; ndrange = 2)
            ka_svd_ut_mul_scale_kernel!(kbackend)(
                tmp_k, U_k, rwork_k, d_k, 2, tol; ndrange = 2)
            ka_svd_v_mul_kernel!(kbackend)(d_k, VT_k, tmp_k, 2; ndrange = 2)
        end
        synchronize(kbackend)
    end
end
