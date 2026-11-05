# CUDA cuSOLVER and cuBLAS dense LU, QR, and SVD factorisations and solves for
# `CuMatrix{T}` and `CuVector{T}` (`T <: LinearAlgebra.BlasFloat`). Non-BLAS element
# types (`Float16`, `BFloat16`) remain on the `KernelAbstractions` device kernels in
# `src/linear/ka/`.

module GeometricSolversCUDAExt

using CUDA: CUDA, CuMatrix, CuVector
using GeometricSolvers: GeometricSolvers, DeviceLU, DeviceQR, DeviceSVD,
                        LUFactorization, QRFactorization, SVDFactorization,
                        jacobi_tolerance
using LinearAlgebra: LinearAlgebra, BlasFloat, adjoint!, diagind

function _cusolver_lu_finish!(
        fac::DeviceLU{<:CuMatrix{T}},
        J::AbstractMatrix,
        info_c::Integer,
        e::Int = 0
) where {T <: BlasFloat}
    if info_c >= 0
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, Int(info_c), e, fac.status)
    else
        return GeometricSolvers.ka_factorize_lu!(fac, J)
    end
end

function GeometricSolvers.factorize!!(
        fac::DeviceLU{<:CuMatrix{T}},
        ::LUFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    e = GeometricSolvers.load_factors!(fac.factors, J)
    isempty(fac.factors) &&
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, 0, e, fac.status)
    _, _, info_c = CUDA.CUSOLVER.getrf!(fac.factors, fac.ipiv)
    return _cusolver_lu_finish!(fac, J, info_c, e)
end

function GeometricSolvers.ka_lu_backsolve!(
        fac::DeviceLU{<:CuMatrix},
        d::CuVector
)
    return GeometricSolvers.ka_lu_backsolve!(fac.factors, fac.ipiv, d)
end

function GeometricSolvers.backsolve!(
        fac::DeviceLU{<:CuMatrix{T}},
        d::CuVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    CUDA.CUSOLVER.getrs!('N', fac.factors, fac.ipiv, d)
    return d
end

function _cusolver_qr_info(R::AbstractMatrix)
    diag_r = Array(@view(R[diagind(R)]))
    idx = findfirst(x -> iszero(x) || !isfinite(x), diag_r)
    return idx === nothing ? 0 : Int(idx)
end

function GeometricSolvers.factorize!!(
        fac::DeviceQR{<:CuMatrix{T}},
        ::QRFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    e = GeometricSolvers.load_factors!(fac.factors, J)
    isempty(fac.factors) &&
        return DeviceQR(fac.factors, fac.τ, fac.rhs, 0, e, fac.status)
    CUDA.CUSOLVER.geqrf!(fac.factors, fac.τ)
    info = _cusolver_qr_info(fac.factors)
    return DeviceQR(fac.factors, fac.τ, fac.rhs, info, e, fac.status)
end

function GeometricSolvers.ka_qr_backsolve!(
        fac::DeviceQR{<:CuMatrix},
        d::CuVector
)
    return GeometricSolvers.ka_qr_backsolve!(fac.factors, fac.τ, d)
end

function GeometricSolvers.backsolve!(
        fac::DeviceQR{<:CuMatrix{T}},
        d::CuVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    CUDA.CUSOLVER.ormqr!('L', T <: Real ? 'T' : 'C', fac.factors, fac.τ, d)
    CUDA.CUBLAS.trsv!('U', 'N', 'N', fac.factors, d)
    return d
end

function _cusolver_svd_finish!(
        fac::DeviceSVD{<:CuMatrix{T}},
        J::AbstractMatrix,
        U::AbstractMatrix{T},
        S::AbstractVector{RT},
        V::AbstractMatrix{T},
        info_c::Integer,
        e::Int = 0
) where {T <: BlasFloat, RT <: Real}
    if info_c != 0
        return GeometricSolvers.ka_factorize_svd!(fac, J)
    end
    if !all(isfinite, S) || !all(isfinite, U) || !all(isfinite, V)
        fill!(fac.flag, Int32(1))
        fill!(fac.S, zero(RT))
        fill!(fac.σ̂, zero(RT))
        return DeviceSVD(
            fac.factors, fac.U, fac.S, fac.VT, fac.V,
            fac.σ̂, fac.flag, fac.tmp, fac.rhs, 1, 0, e)
    end
    copyto!(fac.U, U)
    copyto!(fac.V, V)
    adjoint!(fac.VT, V)
    copyto!(fac.σ̂, S)
    fac.S .= GeometricSolvers.ldexp2.(S, e)
    n = size(fac.factors, 1)
    tol = jacobi_tolerance(n, RT)
    σ1 = maximum(fac.σ̂)
    cutoff = tol * σ1
    r = σ1 > zero(RT) ? Int(count(>(cutoff), fac.σ̂)) : 0
    r < n && fill!(@view(fac.U[:, (r + 1):n]), zero(T))
    fill!(fac.flag, Int32(0))
    return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V,
        fac.σ̂, fac.flag, fac.tmp, fac.rhs, 0, r, e)
end

function GeometricSolvers.factorize!!(
        fac::DeviceSVD{<:CuMatrix{T}},
        ::SVDFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    isempty(fac.factors) && return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V,
        fac.σ̂, fac.flag, fac.tmp, fac.rhs, 0, 0, 0)
    RT = real(T)
    if !all(isfinite, J)
        fill!(fac.flag, Int32(1))
        fill!(fac.S, zero(RT))
        fill!(fac.σ̂, zero(RT))
        return DeviceSVD(
            fac.factors, fac.U, fac.S, fac.VT, fac.V,
            fac.σ̂, fac.flag, fac.tmp, fac.rhs, 1, 0, 0)
    end
    e = GeometricSolvers.matrix_exponent(J)
    fac.factors .= GeometricSolvers.ldexp2.(J, -e)
    if !all(isfinite, fac.factors)
        fill!(fac.flag, Int32(1))
        fill!(fac.S, zero(RT))
        fill!(fac.σ̂, zero(RT))
        return DeviceSVD(
            fac.factors, fac.U, fac.S, fac.VT, fac.V,
            fac.σ̂, fac.flag, fac.tmp, fac.rhs, 1, 0, e)
    end
    U, S, V = CUDA.CUSOLVER.gesvdj!('V', 1, fac.factors)
    info_c = Int(Array(CUDA.CUSOLVER.dense_handle().info)[1])
    return _cusolver_svd_finish!(fac, J, U, S, V, info_c, e)
end

function GeometricSolvers.ka_svd_backsolve!(
        fac::DeviceSVD{<:CuMatrix},
        d::CuVector
)
    GeometricSolvers.ka_svd_backsolve!(fac, d, false)
    return d
end

function GeometricSolvers.backsolve!(
        fac::DeviceSVD{<:CuMatrix{T}},
        d::CuVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    return GeometricSolvers.ka_svd_backsolve!(fac, d)
end

end # module GeometricSolversCUDAExt
