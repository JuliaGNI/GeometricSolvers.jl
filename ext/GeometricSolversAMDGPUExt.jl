# ROCm rocSOLVER and rocBLAS dense LU and QR factorisations and solves for `ROCMatrix{T}` and
# `ROCVector{T}` (`T <: LinearAlgebra.BlasFloat`). `geqrf!` factorises at every shape; the solve
# with an `m × n` QR factor, `m > n`, runs on the `KernelAbstractions` device kernel. Non-BLAS
# element types (`Float16`, `BFloat16`) and the SVD and LDLᵀ factorisations remain on the
# `KernelAbstractions` device kernels in `src/linear/ka/`.

module GeometricSolversAMDGPUExt

using AMDGPU: AMDGPU, ROCMatrix, ROCVector
using GeometricSolvers: GeometricSolvers, DeviceLU, DeviceQR, LUFactorization,
                        QRFactorization
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
    # `geqrf!` writes the LAPACK layout, H = I - τ v vᴴ. The `m × n` factor, `m > n`, solves on
    # `ka_qr_backsolve!`, whose `qadj` kernel applies I - τ v vᴴ as Hᴴ (the layout of `ka_qr!`),
    # so it reads Hᴴ = I - conj(τ) v vᴴ of the LAPACK factor from `conj.(τ)`. The square factor
    # keeps the LAPACK `τ` that `ormqr!` reads.
    size(fac.factors, 1) == size(fac.factors, 2) || (fac.τ .= conj.(fac.τ))
    info = _rocsolver_qr_info(fac.factors)
    return DeviceQR(fac.factors, fac.τ, fac.rhs, info, e, fac.status)
end

function GeometricSolvers.backsolve!(
        fac::DeviceQR{<:ROCMatrix{T}},
        d::ROCVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    # `rocBLAS.trsv!` takes a square matrix only, so the `m × n` factor of a least-squares problem
    # with `m > n` solves on the own kernel, which leaves the solution in `d[1:n]`.
    size(fac.factors, 1) == size(fac.factors, 2) ||
        return GeometricSolvers.ka_qr_backsolve!(fac.factors, fac.τ, d)
    AMDGPU.rocSOLVER.ormqr!('L', T <: Real ? 'T' : 'C', fac.factors, fac.τ, d)
    AMDGPU.rocBLAS.trsv!('U', 'N', 'N', fac.factors, d)
    return d
end

end # module GeometricSolversAMDGPUExt
