# Dense CPU linear factorisations (R3): LAPACK (`LAPACK.getrf!`, `geqrf!`, `gesvd!`) for
# `StridedMatrix{<:BlasFloat}` with held workspace buffers, and `LinearAlgebra`'s generic `lu!`
# and `qr!` for any other `AbstractMatrix`. `geqrf!` and `lapack_gesvd` take `lwork = -1` for a
# workspace query, which writes the optimal size into `work[1]` and factorises nothing.

for (geqrf, ormqr, gesvd, elty, relty) in (
    (:dgeqrf_, :dormqr_, :dgesvd_, :Float64, :Float64),
    (:sgeqrf_, :sormqr_, :sgesvd_, :Float32, :Float32),
    (:zgeqrf_, :zunmqr_, :zgesvd_, :ComplexF64, :Float64),
    (:cgeqrf_, :cunmqr_, :cgesvd_, :ComplexF32, :Float32))
    trans_q = elty === :ComplexF64 || elty === :ComplexF32 ? 'C' : 'T'
    @eval begin
        function geqrf!(A::StridedMatrix{$elty}, tau::AbstractVector{$elty},
                work::AbstractVector{$elty}, lwork::BlasInt = BlasInt(length(work)))
            m, n = size(A)
            lda = max(1, stride(A, 2))
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
            stride(c, 1) == 1 ||
                throw(ArgumentError("the QR step vector must have a unit stride"))
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

        function lapack_gesvd(A::StridedMatrix{$elty}, S::AbstractVector{$relty},
                U::StridedMatrix{$elty}, VT::StridedMatrix{$elty},
                work::AbstractVector{$elty}, rwork::AbstractVector{$relty},
                lwork::BlasInt = BlasInt(length(work)))
            m, n = size(A)
            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldvt = max(1, stride(VT, 2))
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

# The lengths of the host QR and SVD workspaces of `init`: the optimal sizes that a workspace query
# of `geqrf` and of `gesvd` returns, for the buffers that the factorisation then takes. Another
# element type factorises through `qr!`, or has no SVD method, and takes no workspace.
function qr_work_length(A::StridedMatrix{F}, τ::AbstractVector{F}) where {F <: BlasFloat}
    work = Vector{F}(undef, 1)
    geqrf!(A, τ, work, BlasInt(-1))
    return Int(real(work[1]))
end
qr_work_length(A::AbstractMatrix, τ::AbstractVector) = 1

function svd_work_length(A::StridedMatrix{F}, S, U, VT, rwork) where {F <: BlasFloat}
    work = Vector{F}(undef, 1)
    lapack_gesvd(A, S, U, VT, work, rwork, BlasInt(-1))
    return Int(real(work[1]))
end
svd_work_length(A::AbstractMatrix, S, U, VT, rwork) = 1

function qr_zero_diag(R::AbstractMatrix)
    n = min(size(R, 1), size(R, 2))
    @inbounds for i in 1:n
        iszero(R[i, i]) && return i
    end
    return 0
end

# `J` into the factor buffer `A`, and the exponent `e` of the power of two that it was divided by,
# as `factor_matrix` does for R1: 0 where `J` has the element type of `A`, which is then copied,
# else `matrix_exponent(J)`.
function load_factors!(A::AbstractMatrix{F}, J::AbstractMatrix) where {F}
    if eltype(J) === F || isempty(J)
        J === A || copyto!(A, J)
        return 0
    end
    e = matrix_exponent(J)
    A .= J .* ldexp(one(float(real(eltype(J)))), -e)
    return e
end

const DenseFactor{A} = Union{DenseLU{A}, DenseQR{A}, DenseSVD{A}}

# The `T == TF` method: `backsolve!` solves with the factors of `J / 2^e` in place, and the step
# for `J` is that solution times `2^-e`.
function ldiv!!(d::AbstractVector{T}, fac::DenseFactor{<:AbstractMatrix{T}},
        r::AbstractVector{T}) where {T <: Number}
    d === r || copyto!(d, r)
    backsolve!(fac, d)
    e = fac.scale_exponent
    iszero(e) || map!(a -> ldexp2(a, -e), d, d)
    return d
end

# --- DenseLU ---

function factorize!!(fac::DenseLU{<:StridedMatrix{<:BlasFloat}}, ::LUFactorization,
        J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    _, ipiv, info = LAPACK.getrf!(fac.factors, fac.ipiv; check = false)
    return DenseLU(fac.factors, ipiv, fac.rhs, Int(info), e, fac.status)
end

function factorize!!(fac::DenseLU{<:AbstractMatrix}, ::LUFactorization, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    F = lu!(fac.factors, RowMaximum(); check = false)
    copyto!(fac.ipiv, F.ipiv)
    return DenseLU(fac.factors, fac.ipiv, fac.rhs, Int(F.info), e, fac.status)
end

function backsolve!(fac::DenseLU{<:StridedMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                   BlasFloat}
    LAPACK.getrs!('N', fac.factors, fac.ipiv, d)
    return d
end

function backsolve!(fac::DenseLU{<:AbstractMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                    Number}
    ldiv!(LU(fac.factors, fac.ipiv, BlasInt(fac.info)), d)
    return d
end

# --- DenseQR ---

function factorize!!(fac::DenseQR{<:StridedMatrix{<:BlasFloat}}, ::QRFactorization,
        J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    geqrf!(fac.factors, fac.τ, fac.work)
    info = qr_zero_diag(fac.factors)
    return DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, info, e)
end

function factorize!!(fac::DenseQR{<:AbstractMatrix}, ::QRFactorization, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    F = qr!(fac.factors)
    copyto!(fac.τ, F.τ)
    info = qr_zero_diag(fac.factors)
    return DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, info, e)
end

function backsolve!(fac::DenseQR{<:StridedMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                   BlasFloat}
    ormqr_adj!(fac.factors, fac.τ, d, fac.work)
    BLAS.trsv!('U', 'N', 'N', fac.factors, d)
    return d
end

function backsolve!(fac::DenseQR{<:AbstractMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                    Number}
    ldiv!(QR(fac.factors, fac.τ), d)
    return d
end

# --- DenseSVD ---

function factorize!!(fac::DenseSVD{<:StridedMatrix{<:BlasFloat}}, ::SVDFactorization,
        J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    # an iterative LAPACK routine is never given a non-finite entry: the factor fails before it
    if !all(isfinite, fac.factors)
        return DenseSVD(
            fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, 1, 0, e)
    end
    info = Int(lapack_gesvd(fac.factors, fac.S, fac.U, fac.VT, fac.work, fac.rwork))
    # a finite matrix near `floatmax` can still overflow to a singular value that is not finite,
    # which is a failure too
    ok = iszero(info) && all(isfinite, fac.S)
    status = ok ? 0 : (iszero(info) ? 1 : info)
    rank = ok ? svd_rank(fac.S) : 0
    return DenseSVD(
        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs,
        status, rank, e)
end

# The number of the non-increasing singular values `S` above the rank cutoff `√n · eps(R) · S[1]`.
function svd_rank(S::AbstractVector{R}) where {R <: Real}
    isempty(S) && return 0
    cutoff = jacobi_tolerance(length(S), R) * S[1]
    return count(>(cutoff), S)
end

function backsolve!(fac::DenseSVD{<:StridedMatrix{T}}, d::AbstractVector{T}) where {T <:
                                                                                    BlasFloat}
    # a failed factor, whose vectors can hold `NaN`, and a zero matrix give the zero step
    iszero(fac.rank) && return fill!(d, zero(T))
    mul!(fac.tmp, adjoint(fac.U), d)
    @inbounds for i in 1:length(fac.S)
        fac.tmp[i] = i <= fac.rank ? fac.tmp[i] / fac.S[i] : zero(T)
    end
    mul!(d, adjoint(fac.VT), fac.tmp)
    return d
end
