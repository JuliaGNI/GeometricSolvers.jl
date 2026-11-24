# ROCm rocSOLVER and rocBLAS dense LU, QR and Cholesky factorisations and solves for
# `ROCMatrix{T}` and `ROCVector{T}` (`T <: LinearAlgebra.BlasFloat`). `geqrf!` factorises, and
# `ormqr!` with `trsv!` solves, at every shape `m × n`, `m ≥ n`, of the QR factor. Non-BLAS element
# types (`Float16`, `BFloat16`) and the SVD and LDLᵀ factorisations remain on the
# `KernelAbstractions` device kernels in `src/linear/ka/`.

module GeometricSolversAMDGPUExt

using AMDGPU: AMDGPU, ROCMatrix, ROCVector
using GeometricSolvers: GeometricSolvers, CholeskyFactorization, DeviceCholesky, DeviceLU,
                        DeviceQR, LUFactorization, QRFactorization
using LinearAlgebra: BlasFloat, diagind

function GeometricSolvers.factorize!!(
        fac::DeviceLU{<:ROCMatrix{T}},
        ::LUFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    e = GeometricSolvers.load_factors!(fac.factors, J)
    isempty(fac.factors) &&
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, 0, e, fac.status)
    # `getrf!` throws an `ArgumentError` for a negative status, so `info_r` is the first zero
    # pivot, or 0.
    _, _, info_r = AMDGPU.rocSOLVER.getrf!(fac.factors, fac.ipiv)
    return DeviceLU(fac.factors, fac.ipiv, fac.rhs, Int(info_r), e, fac.status)
end

function GeometricSolvers.backsolve!(
        fac::DeviceLU{<:ROCMatrix{T}},
        d::ROCVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    AMDGPU.rocSOLVER.getrs!('N', fac.factors, fac.ipiv, d)
    return d
end

# The first diagonal entry of `R` that is zero or not finite, or 0: geqrf reports no singularity.
function _rocsolver_qr_info(R::AbstractMatrix)
    diag_r = Array(@view(R[diagind(R)]))
    idx = findfirst(x -> iszero(x) || !isfinite(x), diag_r)
    return idx === nothing ? 0 : Int(idx)
end

function GeometricSolvers.factorize!!(
        fac::DeviceQR{<:ROCMatrix{T}},
        ::QRFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    e = GeometricSolvers.load_factors!(fac.factors, J)
    isempty(fac.factors) &&
        return DeviceQR(fac.factors, fac.τ, fac.rhs, 0, e, fac.status)
    AMDGPU.rocSOLVER.geqrf!(fac.factors, fac.τ)
    info = _rocsolver_qr_info(fac.factors)
    return DeviceQR(fac.factors, fac.τ, fac.rhs, info, e, fac.status)
end

# The `m × n` factor, `m ≥ n`, and the length-`m` right-hand side `d`: `ormqr!` applies Qᴴ to `d`
# and `trsv!` solves with the leading `n × n` block `R`, so the solution is in `d[1:n]`.
function GeometricSolvers.backsolve!(
        fac::DeviceQR{<:ROCMatrix{T}},
        d::ROCVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    n = size(fac.factors, 2)
    AMDGPU.rocSOLVER.ormqr!('L', T <: Real ? 'T' : 'C', fac.factors, fac.τ, d)
    AMDGPU.rocBLAS.trsv!('U', 'N', 'N', view(fac.factors, 1:n, 1:n), view(d, 1:n))
    return d
end

# The same-type `ldiv!!` of an `m × n` factor, `m > n`, solves through this method, which keeps
# it on the rocSOLVER and rocBLAS solve above. `ka_qr_backsolve!` would read the LAPACK `τ` that
# `geqrf!` writes as its conjugate, which gives a wrong solution for complex `T`.
function GeometricSolvers.backsolve_shifted!(
        fac::DeviceQR{<:ROCMatrix{T}},
        d::ROCVector{T},
        rhs::ROCVector{T}
) where {T <: BlasFloat}
    GeometricSolvers.backsolve!(fac, rhs)
    copyto!(d, 1, rhs, 1, length(d))
    return 0
end

# The load kernel of the own factorisation refuses a non-finite matrix with `info = 1` and writes
# `J / 2^e` with zeros above the diagonal; `potrf` then factorises the lower triangle, whose layout
# and `info` are LAPACK's, as the own kernel writes them.
function GeometricSolvers.factorize!!(
        fac::DeviceCholesky{<:ROCMatrix{T}},
        ::CholeskyFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    info, e = GeometricSolvers.ka_load_cholesky!(fac, J)
    (info != 0 || isempty(fac.factors)) &&
        return DeviceCholesky(fac.factors, fac.rhs, info, e, fac.status)
    # `potrf!` throws an `ArgumentError` for a negative status, so `info_r` is the order of the
    # first leading minor that is not positive, or 0.
    _, info_r = AMDGPU.rocSOLVER.potrf!('L', fac.factors)
    return DeviceCholesky(fac.factors, fac.rhs, Int(info_r), e, fac.status)
end

function GeometricSolvers.backsolve!(
        fac::DeviceCholesky{<:ROCMatrix{T}},
        d::ROCVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    AMDGPU.rocSOLVER.potrs!('L', fac.factors, d)
    return d
end

end # module GeometricSolversAMDGPUExt
