# Device-resident dense LU factorisation with partial pivoting and triangular backsolve (R3)
# on `AbstractGPUArray`, implemented in `KernelAbstractions` with zero scalar indexing on the
# host and one status readback per factorisation.

@kernel function ka_lu_pivot_swap_scale_kernel!(A, ipiv, status, k::Int, n::Int)
    idx = @index(Global, Linear)
    if idx == 1
        @inbounds begin
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
            akk = A[k, k]
            if iszero(akk)
                if status[1] == Int32(0)
                    status[1] = Int32(k)
                end
            else
                for i in (k + 1):n
                    A[i, k] = multiplier(A[i, k], akk)
                end
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

@kernel function ka_lu_backsolve_kernel!(d, @Const(A), @Const(ipiv), n::Int)
    idx = @index(Global, Linear)
    if idx == 1
        @inbounds begin
            for k in 1:n
                p = Int(ipiv[k])
                if p != k
                    d[k], d[p] = d[p], d[k]
                end
            end
            for k in 1:n
                dk = d[k]
                for i in (k + 1):n
                    d[i] -= A[i, k] * dk
                end
            end
            for k in n:-1:1
                dk = d[k] / A[k, k]
                d[k] = dk
                for i in 1:(k - 1)
                    d[i] -= A[i, k] * dk
                end
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
    piv_kernel! = ka_lu_pivot_swap_scale_kernel!(backend)
    ger_kernel! = ka_lu_ger_kernel!(backend)
    for k in 1:n
        piv_kernel!(A, ipiv, status, k, n; ndrange = 1)
        if k < n
            m = n - k
            ger_kernel!(A, k; ndrange = (m, m))
        end
    end
    isgpu(backend) && synchronize(backend)
    return Int(Array(status)[1])
end

function ka_factorize_lu!(fac::DenseLU, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    info = ka_lu!(fac.factors, fac.ipiv, fac.status)
    return DenseLU(fac.factors, fac.ipiv, fac.rhs, info, e, fac.status)
end

function factorize!!(fac::DenseLU{<:AbstractGPUArray}, ::LUFactorization, J::AbstractMatrix)
    return ka_factorize_lu!(fac, J)
end

function ka_lu_backsolve!(A::AbstractGPUArray, ipiv::AbstractVector{Int32}, d::AbstractVector)
    n = length(d)
    n == 0 && return d
    backend = get_backend(A)
    ka_lu_backsolve_kernel!(backend)(d, A, ipiv, n; ndrange = 1)
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(fac::DenseLU{<:AbstractGPUArray{T}},
        d::AbstractVector{T}) where {T <: BlasFloat}
    return ka_lu_backsolve!(fac.factors, fac.ipiv, d)
end

function backsolve!(fac::DenseLU{<:AbstractGPUArray{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_lu_backsolve!(fac.factors, fac.ipiv, d)
end
