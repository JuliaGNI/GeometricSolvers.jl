# Dense CPU linear factorisations (R3): LAPACK (`getrf!`, `geqrf!`, `gesvd!`) for
# `StridedMatrix{<:BlasFloat}` with held workspace buffers, and `LinearAlgebra`'s generic `lu!`
# and `qr!` for any other `AbstractMatrix`.

for (getrf, getrs, geqrf, ormqr, trtrs, gesvd, elty, relty) in (
    (:dgetrf_, :dgetrs_, :dgeqrf_, :dormqr_, :dtrtrs_, :dgesvd_, :Float64, :Float64),
    (:sgetrf_, :sgetrs_, :sgeqrf_, :sormqr_, :strtrs_, :sgesvd_, :Float32, :Float32),
    (:zgetrf_, :zgetrs_, :zgeqrf_, :zunmqr_, :ztrtrs_, :zgesvd_, :ComplexF64, :Float64),
    (:cgetrf_, :cgetrs_, :cgeqrf_, :cunmqr_, :ctrtrs_, :cgesvd_, :ComplexF32, :Float32))
    trans_q = elty === :ComplexF64 || elty === :ComplexF32 ? 'C' : 'T'
    @eval begin
        function getrf!(A::StridedMatrix{$elty}, ipiv::AbstractVector{BlasInt};
                check::Bool = true)
            check && LAPACK.chkfinite(A)
            m, n = size(A)
            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((BLAS.@blasfunc($getrf), LAPACK.liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
                m, n, A, lda, ipiv, info)
            LAPACK.chkargsok(info[])
            return A, ipiv, info[]
        end

        function geqrf!(A::StridedMatrix{$elty}, tau::AbstractVector{$elty},
                work::AbstractVector{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            lwork = BlasInt(length(work))
            info = Ref{BlasInt}()
            ccall((BLAS.@blasfunc($geqrf), LAPACK.liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                m, n, A, lda, tau, work, lwork, info)
            LAPACK.chkargsok(info[])
            return A, tau
        end

        function ormqr_adj!(A::StridedMatrix{$elty}, tau::AbstractVector{$elty},
                c::StridedVector{$elty}, work::AbstractVector{$elty})
            m = length(c)
            k = length(tau)
            lda = max(1, stride(A, 2))
            lwork = BlasInt(length(work))
            info = Ref{BlasInt}()
            ccall((BLAS.@blasfunc($ormqr), LAPACK.liblapack), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                    Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ref{BlasInt}, Clong, Clong),
                'L', $trans_q, m, 1, k, A, lda, tau, c, max(1, m), work, lwork, info, 1, 1)
            LAPACK.chkargsok(info[])
            return c
        end

        function trtrs_upper!(A::StridedMatrix{$elty}, b::StridedVector{$elty})
            n = size(A, 1)
            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((BLAS.@blasfunc($trtrs), LAPACK.liblapack), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                    Clong, Clong, Clong),
                'U', 'N', 'N', n, 1, A, lda, b, max(1, n), info, 1, 1, 1)
            LAPACK.chkargsok(info[])
            return b
        end

        function lapack_gesvd(A::StridedMatrix{$elty}, S::AbstractVector{$relty},
                U::StridedMatrix{$elty}, VT::StridedMatrix{$elty},
                work::AbstractVector{$elty}, rwork::AbstractVector{$relty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldvt = max(1, stride(VT, 2))
            lwork = BlasInt(length(work))
            info = Ref{BlasInt}()
            if $elty <: Complex
                ccall((BLAS.@blasfunc($gesvd), LAPACK.liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt}, Clong, Clong),
                    'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info,
                    1, 1)
            else
                ccall((BLAS.@blasfunc($gesvd), LAPACK.liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                    'A', 'A', m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, info, 1, 1)
            end
            LAPACK.chkargsok(info[])
            return info[]
        end
    end
end

getrf!(A::StridedMatrix{<:BlasFloat}; check::Bool = true) = LAPACK.getrf!(A; check = check)

function qr_zero_diag(R::AbstractMatrix)
    n = min(size(R, 1), size(R, 2))
    @inbounds for i in 1:n
        iszero(R[i, i]) && return i
    end
    return 0
end

# --- DenseLU ---

function factorize!!(fac::DenseLU{<:StridedMatrix{<:BlasFloat}}, ::LUFactorization,
        J::AbstractMatrix)
    J === fac.factors || copyto!(fac.factors, J)
    _, ipiv, info = getrf!(fac.factors, fac.ipiv; check = false)
    return DenseLU(fac.factors, ipiv, fac.rhs, Int(info))
end

function factorize!!(fac::DenseLU{<:AbstractMatrix}, ::LUFactorization, J::AbstractMatrix)
    J === fac.factors || copyto!(fac.factors, J)
    F = lu!(fac.factors, RowMaximum(); check = false)
    copyto!(fac.ipiv, F.ipiv)
    return DenseLU(fac.factors, fac.ipiv, fac.rhs, Int(F.info))
end

function ldiv!!(d::AbstractVector{T}, fac::DenseLU{<:StridedMatrix{T}},
        r::AbstractVector{T}) where {T <: BlasFloat}
    d === r || copyto!(d, r)
    LAPACK.getrs!('N', fac.factors, fac.ipiv, d)
    return d
end

function ldiv!!(d::AbstractVector{T}, fac::DenseLU{<:AbstractMatrix{T}},
        r::AbstractVector{T}) where {T <: Number}
    d === r || copyto!(d, r)
    ldiv!(LU(fac.factors, fac.ipiv, BlasInt(fac.info)), d)
    return d
end

# --- DenseQR ---

function factorize!!(fac::DenseQR{<:StridedMatrix{<:BlasFloat}}, ::QRFactorization,
        J::AbstractMatrix)
    J === fac.factors || copyto!(fac.factors, J)
    geqrf!(fac.factors, fac.τ, fac.work)
    info = qr_zero_diag(fac.factors)
    return DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, info)
end

function factorize!!(fac::DenseQR{<:AbstractMatrix}, ::QRFactorization, J::AbstractMatrix)
    J === fac.factors || copyto!(fac.factors, J)
    F = qr!(fac.factors)
    copyto!(fac.τ, F.τ)
    info = qr_zero_diag(fac.factors)
    return DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, info)
end

function ldiv!!(d::AbstractVector{T}, fac::DenseQR{<:StridedMatrix{T}},
        r::AbstractVector{T}) where {T <: BlasFloat}
    d === r || copyto!(d, r)
    ormqr_adj!(fac.factors, fac.τ, d, fac.work)
    trtrs_upper!(fac.factors, d)
    return d
end

function ldiv!!(d::AbstractVector{T}, fac::DenseQR{<:AbstractMatrix{T}},
        r::AbstractVector{T}) where {T <: Number}
    d === r || copyto!(d, r)
    ldiv!(QR(fac.factors, fac.τ), d)
    return d
end

# --- DenseSVD ---

function factorize!!(fac::DenseSVD{<:StridedMatrix{<:BlasFloat}}, ::SVDFactorization,
        J::AbstractMatrix)
    J === fac.factors || copyto!(fac.factors, J)
    info = lapack_gesvd(fac.factors, fac.S, fac.U, fac.VT, fac.work, fac.rwork)
    return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, Int(info))
end

function ldiv!!(d::AbstractVector{T}, fac::DenseSVD{<:StridedMatrix{T}},
        r::AbstractVector{T}) where {T <: BlasFloat}
    R = real(T)
    n = length(fac.S)
    cutoff = sqrt(R(n)) * eps(R) * fac.S[1]
    mul!(fac.tmp, adjoint(fac.U), r)
    @inbounds for i in 1:n
        s = fac.S[i]
        fac.tmp[i] = s <= cutoff ? zero(T) : fac.tmp[i] / s
    end
    mul!(d, adjoint(fac.VT), fac.tmp)
    return d
end
