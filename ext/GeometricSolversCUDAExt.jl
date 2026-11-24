# CUDA cuSOLVER and cuBLAS dense LU, QR, SVD and Cholesky factorisations and solves for
# `CuMatrix{T}` and `CuVector{T}` (`T <: LinearAlgebra.BlasFloat`). Non-BLAS element
# types (`Float16`, `BFloat16`) remain on the `KernelAbstractions` device kernels in
# `src/linear/ka/`.

module GeometricSolversCUDAExt

using CUDA: CUDA, CuMatrix, CuVector
using GeometricSolvers: GeometricSolvers, CholeskyFactorization, DeviceCholesky, DeviceLU,
                        DeviceQR, DeviceSVD, LUFactorization, QRFactorization,
                        SVDFactorization
using LinearAlgebra: LinearAlgebra, BlasFloat, adjoint!, diagind

function GeometricSolvers.factorize!!(
        fac::DeviceLU{<:CuMatrix{T}},
        ::LUFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    e = GeometricSolvers.load_factors!(fac.factors, J)
    isempty(fac.factors) &&
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, 0, e, fac.status)
    # `getrf!` throws an `ArgumentError` for a negative status, so `info_c` is the first zero
    # pivot, or 0.
    _, _, info_c = CUDA.CUSOLVER.getrf!(fac.factors, fac.ipiv)
    return DeviceLU(fac.factors, fac.ipiv, fac.rhs, Int(info_c), e, fac.status)
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

function GeometricSolvers.backsolve!(
        fac::DeviceQR{<:CuMatrix{T}},
        d::CuVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    CUDA.CUSOLVER.ormqr!('L', T <: Real ? 'T' : 'C', fac.factors, fac.τ, d)
    CUDA.CUBLAS.trsv!('U', 'N', 'N', fac.factors, d)
    return d
end

# The factor of a matrix that cannot be factorised, a non-finite entry in the matrix or in its
# factors: `info = 1` and `rank = 0`, the zero step, with `flag` raised and no singular value.
function _cusolver_svd_failed!(fac::DeviceSVD{<:CuMatrix{T}}, e::Int) where {T}
    RT = real(T)
    fill!(fac.flag, Int32(1))
    fill!(fac.S, zero(RT))
    fill!(fac.σ̂, zero(RT))
    return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V,
        fac.σ̂, fac.flag, fac.tmp, fac.rhs, 1, 0, e)
end

function _cusolver_svd_finish!(
        fac::DeviceSVD{<:CuMatrix{T}},
        method::SVDFactorization,
        J::AbstractMatrix,
        U::AbstractMatrix{T},
        S::AbstractVector{<:Real},
        V::AbstractMatrix{T},
        info_c::Integer,
        e::Int = 0
) where {T <: BlasFloat}
    if info_c != 0
        return GeometricSolvers.ka_factorize_svd!(fac, J, method)
    end
    if !all(isfinite, S) || !all(isfinite, U) || !all(isfinite, V)
        return _cusolver_svd_failed!(fac, e)
    end
    copyto!(fac.U, U)
    copyto!(fac.V, V)
    adjoint!(fac.VT, V)
    copyto!(fac.σ̂, S)
    fac.S .= GeometricSolvers.ldexp2.(S, e)
    m_rows, n = size(fac.U, 1), size(fac.U, 2)
    r = GeometricSolvers.svd_rank(Array(fac.σ̂), m_rows, method.rtol)
    r < n && fill!(@view(fac.U[:, (r + 1):n]), zero(T))
    fill!(fac.flag, Int32(0))
    return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V,
        fac.σ̂, fac.flag, fac.tmp, fac.rhs, 0, r, e)
end

function GeometricSolvers.factorize!!(
        fac::DeviceSVD{<:CuMatrix{T}},
        method::SVDFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    isempty(fac.factors) && return DeviceSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.V,
        fac.σ̂, fac.flag, fac.tmp, fac.rhs, 0, 0, 0)
    # `J` goes into `fac.factors`, the Jacobian buffer that the factor documents and that a caller
    # may pass back as `J`, and `J / 2^e` into `fac.U`, the matrix that `gesvdj!` factorises.
    if eltype(J) === T
        J === fac.factors || copyto!(fac.factors, J)
    else
        fac.factors .= J
    end
    all(isfinite, J) || return _cusolver_svd_failed!(fac, 0)
    # In single precision `gesvdj!` returns a singular value far below `eps(R) · σ̂₁` as 0, whatever
    # its tolerance, while the own Jacobi kernel resolves it. A cutoff below the sweep tolerance
    # `√m · eps(R)`, such as `rtol = 0`, needs those values, so it runs on the own kernel. The
    # default `rtol = nothing` gives the sweep tolerance itself and stays on `gesvdj!`.
    RT, m_rows = real(T), size(fac.U, 1)
    GeometricSolvers.rank_tolerance(method.rtol, m_rows, RT) <
    GeometricSolvers.jacobi_tolerance(m_rows, RT) &&
        return GeometricSolvers.ka_factorize_svd!(fac, J, method)
    e = GeometricSolvers.matrix_exponent(J)
    fac.U .= GeometricSolvers.ldexp2.(J, -e)
    all(isfinite, fac.U) || return _cusolver_svd_failed!(fac, e)
    U, S, V = CUDA.CUSOLVER.gesvdj!('V', 1, fac.U)
    info_c = Int(Array(CUDA.CUSOLVER.dense_handle().info)[1])
    return _cusolver_svd_finish!(fac, method, J, U, S, V, info_c, e)
end

function GeometricSolvers.backsolve!(
        fac::DeviceSVD{<:CuMatrix{T}},
        d::CuVector{T}
) where {T <: BlasFloat}
    GeometricSolvers.ka_svd_backsolve!(fac, d, d, false)
    return d
end

# The load kernel of the own factorisation refuses a non-finite matrix with `info = 1` and writes
# `J / 2^e` with zeros above the diagonal; `potrf` then factorises the lower triangle, whose layout
# and `info` are LAPACK's, as the own kernel writes them.
function GeometricSolvers.factorize!!(
        fac::DeviceCholesky{<:CuMatrix{T}},
        ::CholeskyFactorization,
        J::AbstractMatrix
) where {T <: BlasFloat}
    info, e = GeometricSolvers.ka_load_cholesky!(fac, J)
    (info != 0 || isempty(fac.factors)) &&
        return DeviceCholesky(fac.factors, fac.rhs, info, e, fac.status)
    # `potrf!` throws an `ArgumentError` for a negative status, so `info_c` is the order of the
    # first leading minor that is not positive, or 0.
    _, info_c = CUDA.CUSOLVER.potrf!('L', fac.factors)
    return DeviceCholesky(fac.factors, fac.rhs, Int(info_c), e, fac.status)
end

function GeometricSolvers.backsolve!(
        fac::DeviceCholesky{<:CuMatrix{T}},
        d::CuVector{T}
) where {T <: BlasFloat}
    isempty(d) && return d
    CUDA.CUSOLVER.potrs!('L', fac.factors, d)
    return d
end

end # module GeometricSolversCUDAExt
