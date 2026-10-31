# Replace LAPACK's LU and QR of the dense path of GeometricSolvers by pure Julia, for the
# diagnosis of `diagnose.jl` and `trace.jl`: LinearAlgebra's `generic_lufact!` and
# `qrfactUnblocked!`, and triangular solves and the application of Qᵀ as plain loops. The SVD
# column keeps LAPACK's `gesvd`, because LinearAlgebra has no generic SVD. The methods replace
# `factorize!!` and `backsolve!` of `src/linear/dense.jl` for strided BLAS matrices in this
# process only.
using LinearAlgebra

@eval GeometricSolvers import LinearAlgebra
@eval GeometricSolvers begin
    function factorize!!(fac::DenseLU{<:StridedMatrix{<:BlasFloat}}, ::LUFactorization,
            J::AbstractMatrix)
        e = load_factors!(fac.factors, J)
        F = LinearAlgebra.generic_lufact!(
            fac.factors, RowMaximum(), fac.ipiv; check = false,
            allowsingular = true)
        return DenseLU(fac.factors, fac.ipiv, fac.rhs, Int(F.info), e, fac.status)
    end

    function backsolve!(fac::DenseLU{<:StridedMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                       BlasFloat}
        A = fac.factors
        n = size(A, 1)
        for i in 1:n
            p = fac.ipiv[i]
            p != i && ((d[i], d[p]) = (d[p], d[i]))
        end
        for i in 2:n, j in 1:(i - 1)

            d[i] -= A[i, j] * d[j]
        end
        for i in n:-1:1
            s = d[i]
            for j in (i + 1):n
                s -= A[i, j] * d[j]
            end
            d[i] = s / A[i, i]
        end
        return d
    end

    function factorize!!(fac::DenseQR{<:StridedMatrix{<:BlasFloat}}, ::QRFactorization,
            J::AbstractMatrix)
        e = load_factors!(fac.factors, J)
        F = LinearAlgebra.qrfactUnblocked!(fac.factors)
        copyto!(fac.τ, F.τ)
        info = qr_zero_diag(fac.factors)
        return DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, info, e)
    end

    function backsolve!(fac::DenseQR{<:StridedMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                       BlasFloat}
        A = fac.factors
        n = size(A, 1)
        for k in 1:n
            s = d[k]
            for i in (k + 1):n
                s += A[i, k] * d[i]
            end
            s *= fac.τ[k]
            d[k] -= s
            for i in (k + 1):n
                d[i] -= s * A[i, k]
            end
        end
        for i in n:-1:1
            s = d[i]
            for j in (i + 1):n
                s -= A[i, j] * d[j]
            end
            d[i] = s / A[i, i]
        end
        return d
    end
end
