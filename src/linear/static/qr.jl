# The R1 Householder QR factorisation without column pivoting: a scalar loop that one thread
# runs, on a local `MMatrix` that the compiler keeps off the heap, as the LU of `lu.jl`.
#
# Step k reflects x = A[k:N, k] onto β e₁ with β = −phase · ‖x‖, where phase = x₁/|x₁|, or 1 at
# x₁ = 0. With v = x − β e₁ scaled to w = v / v₁, the reflector is I − τ w wᴴ with
# τ = (‖x‖ + |x₁|)/‖x‖, a real number in [1, 2], and every |w[i]| ≤ 1.

function factorize!!(::Union{Nothing, StaticQR},
        method::QRFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    Â, e = factor_matrix(factor_eltype(method, T), J)
    A = MMatrix(Â)
    F = eltype(A)
    τ = zero(SVector{N, F})
    info = Int32(0)
    @inbounds for k in 1:N
        # ‖A[k:N, k]‖ as LAPACK's `nrm2` takes it: the subcolumn, zero above k, is divided by its
        # largest |A[i, k]| before `rnorm` squares it, so that the squares neither overflow nor
        # underflow
        amax = zero(real(F))
        for i in k:N
            amax = max(amax, abs(A[i, k]))
        end
        # Fill `x` over `1:N` in one loop. On Metal in `Float32` this form gives correct factors
        # at n = 1:12 on Julia 1.13.2 and wrong ones at n = 3:9 on Julia 1.12.7 (K9 in
        # `KNOWN_ISSUES.md`; `scripts/spikes/metal_qr/README.md` compares other forms).
        x = MVector{N, F}(undef)
        for i in 1:N
            x[i] = (i >= k && !iszero(amax)) ? A[i, k] / amax : zero(F)
        end
        σ = amax * rnorm(SVector(x))
        if !iszero(σ)
            x₁ = A[k, k]
            phase = iszero(x₁) ? one(F) : x₁ / abs(x₁)
            v₁ = phase * (σ + abs(x₁))
            τk = (σ + abs(x₁)) / σ
            τ = setindex(τ, F(τk), k)
            for i in (k + 1):N
                A[i, k] = A[i, k] / v₁
            end
            # A[k:N, j] -= τ w (wᴴ A[k:N, j]) for the columns right of k
            for j in (k + 1):N
                s = A[k, j]
                for i in (k + 1):N
                    s += conj(A[i, k]) * A[i, j]
                end
                s *= τk
                A[k, j] -= s
                for i in (k + 1):N
                    A[i, j] -= A[i, k] * s
                end
            end
            A[k, k] = -phase * σ
        end
        # a zero subcolumn leaves τ[k] = 0 and R[k, k] = 0, with no 0/0
        iszero(A[k, k]) && info == 0 && (info = Int32(k))
    end
    return StaticQR(SMatrix(A), τ, info, e)
end

# The solve with the stored factors, of the matrix that `factorize!!` divided by `2^e`.
function backsolve(fac::StaticQR{N, TF}, r::SVector{N, TF}) where {N, TF}
    A = fac.factors
    x = MVector{N, TF}(r)
    @inbounds begin
        # Qᴴ r = H_N ⋯ H_1 r, each reflector Hermitian: x[k:N] -= τ w (wᴴ x[k:N])
        for k in 1:N
            s = x[k]
            for i in (k + 1):N
                s += conj(A[i, k]) * x[i]
            end
            s *= fac.τ[k]
            x[k] -= s
            for i in (k + 1):N
                x[i] -= A[i, k] * s
            end
        end
        # R x = Qᴴ r
        for k in N:-1:1
            xk = x[k] / A[k, k]
            x[k] = xk
            for i in 1:(k - 1)
                x[i] -= A[i, k] * xk
            end
        end
    end
    return SVector(x)
end
