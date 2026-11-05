# Device-resident dense LU factorisation with partial pivoting and triangular backsolve (R3)
# on `AbstractGPUArray`, implemented in `KernelAbstractions` with workgroup pivot reduction,
# column-parallel triangular solves, zero scalar indexing on the host, and one status readback
# per factorisation.

using KernelAbstractions: @localmem, @synchronize

@kernel function ka_lu_pivot_swap_scale_kernel!(
        A, ipiv, status, k::Int, n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    l_amax = @localmem real(eltype(A)) (G,)
    l_piv = @localmem Int32 (G,)
    R = real(eltype(A))
    @inbounds begin
        best_amax = -one(R)
        best_p = Int32(n + 1)
        i = k + tid - 1
        while i <= n
            a = abs1(A[i, k])
            if best_p > n || a > best_amax
                best_amax = a
                best_p = Int32(i)
            end
            i += G
        end
        l_amax[tid] = best_amax
        l_piv[tid] = best_p
    end
    @synchronize()
    for step in 1:trailing_zeros(G)
        s = G >> step
        @inbounds if tid <= s
            a_other = l_amax[tid + s]
            p_other = l_piv[tid + s]
            if a_other > l_amax[tid] ||
               (a_other == l_amax[tid] && p_other < l_piv[tid])
                l_amax[tid] = a_other
                l_piv[tid] = p_other
            end
        end
        @synchronize()
    end
    p = Int(@inbounds l_piv[1])
    if tid == 1
        @inbounds ipiv[k] = Int32(p)
    end
    if p != k
        j = tid
        @inbounds while j <= n
            A[k, j], A[p, j] = A[p, j], A[k, j]
            j += G
        end
    end
    @synchronize()
    @inbounds begin
        akk = A[k, k]
        if iszero(akk)
            if tid == 1 && status[1] == Int32(0)
                status[1] = Int32(k)
            end
        else
            i = k + tid
            while i <= n
                A[i, k] = multiplier(A[i, k], akk)
                i += G
            end
        end
    end
end

@kernel function ka_lu_ger_kernel!(A, k::Int)
    di, dj = @index(Global, NTuple)
    i = k + di
    j = k + dj
    @inbounds begin
        if !iszero(A[k, k])
            A[i, j] -= A[i, k] * A[k, j]
        end
    end
end

@kernel function ka_lu_perm_l_kernel!(
        d, @Const(A), @Const(ipiv), n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    if tid == 1
        @inbounds for k in 1:n
            p = Int(ipiv[k])
            if p != k
                d[k], d[p] = d[p], d[k]
            end
        end
    end
    @synchronize()
    for k in 1:(n - 1)
        i = k + tid
        while i <= n
            @inbounds d[i] -= A[i, k] * d[k]
            i += G
        end
        @synchronize()
    end
end

@kernel function ka_lu_trsv_u_col_kernel!(d, @Const(A), k::Int)
    i = @index(Global, Linear)
    @inbounds begin
        dk = d[k] / A[k, k]
        d[i] -= A[i, k] * dk
    end
end

@kernel function ka_lu_trsv_u_diag_kernel!(d, @Const(A))
    k = @index(Global, Linear)
    @inbounds d[k] = d[k] / A[k, k]
end

const KA_LU_WORKGROUP = 64

@kernel function ka_lu_pivot_swap_scale_serial_kernel!(
        A, ipiv, status, k::Int, n::Int)
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        p = k
        amax = abs1(A[k, k])
        for i in (k + 1):n
            a = abs1(A[i, k])
            if a > amax
                amax = a
                p = i
            end
        end
        ipiv[k] = Int32(p)
        if p != k
            for j in 1:n
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
        end
        a_kk = A[k, k]
        if iszero(a_kk)
            if status[1] == Int32(0)
                status[1] = Int32(k)
            end
        else
            for i in (k + 1):n
                A[i, k] = multiplier(A[i, k], a_kk)
            end
        end
    end
end

@kernel function ka_lu_perm_l_serial_kernel!(
        d, @Const(A), @Const(ipiv), n::Int)
    idx = @index(Global, Linear)
    @inbounds if idx == 1
        for k in 1:n
            p_k = Int(ipiv[k])
            if p_k != k
                d[k], d[p_k] = d[p_k], d[k]
            end
        end
        for k in 1:(n - 1)
            d_k = d[k]
            for i in (k + 1):n
                d[i] -= A[i, k] * d_k
            end
        end
    end
end

function ka_lu!(A::AbstractGPUArray, ipiv::AbstractVector{Int32},
        status::AbstractVector{Int32})
    n = size(A, 1)
    n == 0 && return 0
    backend = get_backend(A)
    fill!(status, Int32(0))
    use_workgroup = isgpu(backend) || n <= KA_LU_WORKGROUP
    piv_kernel! = ka_lu_pivot_swap_scale_kernel!(backend, KA_LU_WORKGROUP)
    piv_serial! = ka_lu_pivot_swap_scale_serial_kernel!(backend, 1)
    ger_kernel! = ka_lu_ger_kernel!(backend)
    for k in 1:n
        if use_workgroup
            piv_kernel!(
                A, ipiv, status, k, n, Val(KA_LU_WORKGROUP); ndrange = KA_LU_WORKGROUP)
        else
            piv_serial!(A, ipiv, status, k, n; ndrange = 1)
        end
        if k < n
            m = n - k
            ger_kernel!(A, k; ndrange = (m, m))
        end
    end
    isgpu(backend) && synchronize(backend)
    return Int(Array(status)[1])
end

function ka_factorize_lu!(fac::DeviceLU, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    info = ka_lu!(fac.factors, fac.ipiv, fac.status)
    return DeviceLU(fac.factors, fac.ipiv, fac.rhs, info, e, fac.status)
end

function factorize!!(fac::DeviceLU, ::LUFactorization, J::AbstractMatrix)
    return ka_factorize_lu!(fac, J)
end

function ka_lu_backsolve!(A::AbstractGPUArray, ipiv::AbstractVector{Int32}, d::AbstractVector)
    n = length(d)
    n == 0 && return d
    backend = get_backend(A)
    if isgpu(backend) || n <= KA_LU_WORKGROUP
        ka_lu_perm_l_kernel!(backend, KA_LU_WORKGROUP)(
            d, A, ipiv, n, Val(KA_LU_WORKGROUP); ndrange = KA_LU_WORKGROUP)
    else
        ka_lu_perm_l_serial_kernel!(backend, 1)(d, A, ipiv, n; ndrange = 1)
    end
    trsv_u! = ka_lu_trsv_u_col_kernel!(backend)
    for k in n:-1:2
        trsv_u!(d, A, k; ndrange = k - 1)
    end
    ka_lu_trsv_u_diag_kernel!(backend)(d, A; ndrange = n)
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(fac::DeviceLU{<:AbstractMatrix{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_lu_backsolve!(fac.factors, fac.ipiv, d)
end
