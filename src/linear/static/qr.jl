# The R1 Householder QR factorisation without column pivoting: a scalar loop that one thread
# runs, on a local `MMatrix` that the compiler keeps off the heap, as the LU of `lu.jl`.
#
# Step k reflects x = A[k:N, k] onto β e₁ with β = −phase · ‖x‖, where phase = x₁/|x₁|, or 1 at
# x₁ = 0. With v = x − β e₁ scaled to w = v / v₁, the reflector is I − τ w wᴴ with
# τ = (‖x‖ + |x₁|)/‖x‖, a real number in [1, 2], and every |w[i]| ≤ 1.

# The reflector vector of step k, from the factors: 0 above k, 1 at k, the stored w below.
function reflector(A::SMatrix{N, N, TF}, k) where {N, TF}
    return SVector{N, TF}(ntuple(i -> i > k ? A[i, k] : (i == k ? one(TF) : zero(TF)), Val(N)))
end

function factorize!!(::Union{Nothing, StaticQR},
        method::QRFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    F = factor_eltype(method, T)
    A = MMatrix(SMatrix{N, N, F}(J))
    τ = zero(SVector{N, F})
    info = Int32(0)
    @inbounds for k in 1:N
        # the subcolumn A[k:N, k], zero above k, so that its norm goes through `rnorm`
        x = zero(MVector{N, F})
        for i in k:N
            x[i] = A[i, k]
        end
        σ = rnorm(SVector(x))
        if !iszero(σ)
            x₁ = A[k, k]
            phase = iszero(x₁) ? one(F) : x₁ / abs(x₁)
            v₁ = phase * (σ + abs(x₁))
            τk = (σ + abs(x₁)) / σ
            τ = Base.setindex(τ, F(τk), k)
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
    return StaticQR(SMatrix(A), τ, info)
end

function ldiv!!(::Nothing, fac::StaticQR{N, TF}, r::StaticVector{N, TF}) where {N, TF}
    A = fac.factors
    # Qᴴ r = H_N ⋯ H_1 r, each reflector Hermitian
    y = SVector{N, TF}(r)
    for k in 1:N
        w = reflector(A, k)
        y -= (fac.τ[k] * (w' * y)) * w
    end
    # R x = Qᴴ r
    x = MVector{N, TF}(y)
    @inbounds for k in N:-1:1
        xk = x[k] / A[k, k]
        x[k] = xk
        for i in 1:(k - 1)
            x[i] -= A[i, k] * xk
        end
    end
    return SVector(x)
end
