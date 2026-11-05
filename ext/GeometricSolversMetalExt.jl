# Metal Performance Shaders (MPS) dense LU factorisation and triangular solve for
# `MtlMatrix{Float32}` and `MtlVector{Float32}`. Other element types (`ComplexF32`,
# `Float16`, `BFloat16`) and other factorisations (`QRFactorization`, `SVDFactorization`)
# remain on the `KernelAbstractions` device kernels in `src/linear/ka/`.

module GeometricSolversMetalExt

using GeometricSolvers: GeometricSolvers, DeviceLU, LUFactorization
using LinearAlgebra: LinearAlgebra, BlasInt, LU
using Metal: MtlMatrix, MtlVector

function _mps_lu_finish!(
        fac::DeviceLU{<:MtlMatrix{Float32}},
        J::AbstractMatrix,
        F_ipiv::AbstractVector{<:Integer},
        status_code::Integer,
        e::Int = 0
)
    if status_code == 0
        fac.ipiv === F_ipiv || (fac.ipiv .= Int32.(F_ipiv))
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, 0, e, fac.status)
    elseif status_code == -2 || status_code > 0
        fac.ipiv === F_ipiv || (fac.ipiv .= Int32.(F_ipiv))
        info = status_code > 0 ? Int(status_code) : 1
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, info, e, fac.status)
    else
        return GeometricSolvers.ka_factorize_lu!(fac, J)
    end
end

function GeometricSolvers.factorize!!(
        fac::DeviceLU{<:MtlMatrix{Float32}},
        ::LUFactorization,
        J::AbstractMatrix
)
    e = GeometricSolvers.load_factors!(fac.factors, J)
    isempty(fac.factors) &&
        return DeviceLU(fac.factors, fac.ipiv, fac.rhs, 0, e, fac.status)
    F = LinearAlgebra.lu!(fac.factors; check = false)
    return _mps_lu_finish!(fac, J, F.ipiv, Int(F.info), e)
end

function GeometricSolvers.ka_lu_backsolve!(
        fac::DeviceLU{<:MtlMatrix},
        d::MtlVector
)
    return GeometricSolvers.ka_lu_backsolve!(fac.factors, fac.ipiv, d)
end

function GeometricSolvers.backsolve!(
        fac::DeviceLU{<:MtlMatrix{Float32}},
        d::MtlVector{Float32}
)
    isempty(d) && return d
    F = LU(fac.factors, fac.ipiv, BlasInt(fac.info))
    LinearAlgebra.ldiv!(F, d)
    return d
end

end # module GeometricSolversMetalExt
