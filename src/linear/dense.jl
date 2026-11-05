# Dense CPU linear factorisations (R3): LAPACK (`LAPACK.getrf!`, `geqrf!`, `lapack_gesvd`) for
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
                    'S', 'S', m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info,
                    1, 1)
            else
                ccall((BLAS.@blasfunc($gesvd), LAPACK.liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                    'S', 'S', m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, info, 1, 1)
            end
            LAPACK.chkargsok(info[])
            return info[]
        end
    end
end

# The lengths of the host QR and SVD workspaces of `dense_factor`: the optimal sizes that a
# workspace query of `geqrf` and of `gesvd` returns, for the buffers that the factorisation then
# takes. Another element type factorises through `qr!`, or has no SVD method, and does not use its
# one-element workspace.
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

const DenseFactor{A} = Union{
    DenseLU{A}, DenseQR{A}, DenseSVD{A}, DeviceLU{A}, DeviceQR{A}, DeviceSVD{A}}

# The `T == TF` method: `d` takes `r / 2^k`, `rhs_exponent` of `factors.jl`, `backsolve_shifted!`
# solves with the factors of `J / 2^e` in place, an SVD dividing by `σ̂ 2^t`, and the step for `J` is
# that solution times `2^(k + t - e)`.
function ldiv!!(d::AbstractVector{T}, fac::DenseFactor{<:AbstractMatrix{T}},
        r::AbstractVector{T}) where {T <: Number}
    k = rhs_exponent(r)
    t = if length(r) == length(d)
        if iszero(k)
            d === r || copyto!(d, r)
        else
            d .= ldexp2.(r, -k)
        end
        backsolve_shifted!(fac, d)
    else
        rhs = fac.rhs::AbstractVector{T}
        if iszero(k)
            rhs === r || copyto!(rhs, r)
        else
            rhs .= ldexp2.(r, -k)
        end
        backsolve_shifted!(fac, d, rhs)
    end
    e = k + t - fac.scale_exponent
    iszero(e) || map!(a -> ldexp2(a, e), d, d)
    return d
end

# `backsolve!` and the exponent `t` of the power of two its result is to be multiplied by: 0 for LU
# and QR, and for the SVD the `divisor_exponent` of its coordinates.
backsolve_shifted!(fac, d) = (backsolve!(fac, d); 0)
function backsolve_shifted!(fac, d, rhs)
    backsolve!(fac, rhs)
    @views copyto!(d, rhs[1:length(d)])
    return 0
end

# --- DenseLU ---

function factorize!!(fac::DenseLU{<:StridedMatrix{<:BlasFloat}}, ::LUFactorization,
        J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    _, ipiv, info = LAPACK.getrf!(fac.factors, fac.ipiv; check = false)
    return DenseLU(fac.factors, ipiv, fac.rhs, Int(info), e)
end

function factorize!!(fac::DenseLU{<:AbstractMatrix}, ::LUFactorization, J::AbstractMatrix)
    e = load_factors!(fac.factors, J)
    F = lu!(fac.factors, RowMaximum(); check = false)
    copyto!(fac.ipiv, F.ipiv)
    return DenseLU(fac.factors, fac.ipiv, fac.rhs, Int(F.info), e)
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
    m, n = size(fac.factors)
    if m == n
        BLAS.trsv!('U', 'N', 'N', fac.factors, d)
    else
        A = fac.factors
        @inbounds for k in n:-1:1
            dk = d[k] / A[k, k]
            d[k] = dk
            for i in 1:(k - 1)
                d[i] -= A[i, k] * dk
            end
        end
    end
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
    # `J / 2^e` for every element type of `J`, as the static and device SVDs divide it, so that a
    # matrix near either end of the exponent range neither overflows in `gesvd` nor loses its small
    # singular values to the subnormal range
    e = isempty(J) ? 0 : matrix_exponent(J)
    fac.factors .= ldexp2.(J, -e)
    # an iterative LAPACK routine is never given a non-finite entry: the factor fails before it
    if !all(isfinite, fac.factors)
        return DenseSVD(
            fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp, fac.rhs, 1, 0, e)
    end
    info = Int(lapack_gesvd(fac.factors, fac.S, fac.U, fac.VT, fac.work, fac.rwork))
    # `gesvd` wrote the singular values `σ̂` of `J / 2^e` into `S`: the status and the rank are read
    # from them, a singular value that is not finite being a failure too
    ok = iszero(info) && all(isfinite, fac.S)
    status = ok ? 0 : (iszero(info) ? 1 : info)
    rank = ok ? svd_rank(fac.S) : 0
    # `σ̂` into `rwork`, which `gesvd` no longer needs, for the solve, and the singular values
    # `σ̂ · 2^e` of `J` into `S`, which overflow to `Inf` where `σ̂ · 2^e` exceeds `floatmax`
    @inbounds for i in eachindex(fac.S)
        σ̂ = fac.S[i]
        fac.rwork[i] = σ̂
        fac.S[i] = ldexp2(σ̂, e)
    end
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
    n = size(fac.U, 2)
    if length(d) == n
        svd_solve!(fac, d, d, false)
    else
        svd_solve!(fac, view(d, 1:n), d, false)
    end
    return d
end

function backsolve_shifted!(fac::DenseSVD{<:StridedMatrix{T}},
        d::AbstractVector{T}) where {T <: BlasFloat}
    return svd_solve!(fac, d, d, true)
end

function backsolve_shifted!(fac::DenseSVD{<:StridedMatrix{T}},
        d::AbstractVector{T}, rhs::AbstractVector{T}) where {T <: BlasFloat}
    return svd_solve!(fac, d, rhs, true)
end

# The solve with the factors of `J / 2^e` divides by their singular values `σ̂` in `rwork`, the
# same-type solve by `σ̂ 2^t`, `t` the `divisor_exponent` of the coordinates `Uᴴ r`; returns `t`.
function svd_solve!(fac::DenseSVD{<:StridedMatrix{T}}, d::AbstractVector{T},
        r::AbstractVector{T}, shift::Bool) where {T <: BlasFloat}
    # a failed factor, whose vectors can hold `NaN`, and a zero matrix give the zero step
    iszero(fac.rank) && (fill!(d, zero(T)); return 0)
    mul!(fac.tmp, adjoint(fac.U), r)
    t = shift ? divisor_exponent(fac.tmp, fac.rwork, fac.rank) : 0
    @inbounds for i in 1:length(fac.S)
        σ = iszero(t) ? fac.rwork[i] : ldexp2(fac.rwork[i], t)
        fac.tmp[i] = i <= fac.rank ? fac.tmp[i] / σ : zero(T)
    end
    mul!(d, adjoint(fac.VT), fac.tmp)
    return t
end

function dense_svd_nullnorm!(
        p::AbstractVector{TF}, U::StridedMatrix{TF}, rk::Int,
        r::AbstractVector{TF}) where {TF}
    copyto!(p, r)
    n = length(r)
    @inbounds for j in 1:rk
        s = zero(TF)
        for i in 1:n
            s += conj(U[i, j]) * r[i]
        end
        for i in 1:n
            p[i] -= U[i, j] * s
        end
    end
    return rnorm(p)
end

function factor_diagnostics(
        d::AbstractVector{T}, fac::DenseSVD{<:StridedMatrix{TF}},
        r::AbstractVector{T}, fnorm::R) where {T, TF, R <: Real}
    fac.info == 0 || return Int32(-1), R(NaN)
    rk = fac.rank
    rk == 0 && return Int32(0), fnorm
    if TF === T
        p_buf = length(fac.tmp) == length(r) ? fac.tmp : fac.rhs::AbstractVector{TF}
        return Int32(rk), R(dense_svd_nullnorm!(p_buf, fac.U, rk, r))
    end
    s = maximum(abs, r)
    iszero(s) && return Int32(rk), zero(R)
    rhs = fac.rhs::AbstractVector{TF}
    @inbounds for i in eachindex(rhs, r)
        rhs[i] = TF(r[i] / s)
    end
    return Int32(rk), R(dense_svd_nullnorm!(rhs, fac.U, rk, rhs)) * R(s)
end

function factor_diagnostics(
        fac::Union{DenseQR, DeviceQR}, r::AbstractVector{T}) where {T}
    R = real(T)
    n = size(fac.factors, 2)
    rk = fac.info == 0 ? Int32(n) : Int32(fac.info - 1)
    return rk, zero(R)
end

function factor_diagnostics(
        fac::Union{DenseSVD, DeviceSVD}, r::AbstractVector{T}) where {T}
    R = real(T)
    fac.info == 0 || return Int32(-1), R(NaN)
    n = size(fac.factors, 2)
    rk = Int32(fac.rank)
    rk == Int32(n) && return rk, zero(R)
    fnorm = R(rnorm(r))
    iszero(fnorm) && return rk, zero(R)
    d_dummy = similar(r, n)
    _, nn = factor_diagnostics(d_dummy, fac, r, fnorm)
    return rk, nn / fnorm
end
