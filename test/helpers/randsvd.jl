using LinearAlgebra
using Random

function _haar_unitary(rng::AbstractRNG, ::Type{T}, n::Integer) where {T}
    G = randn(rng, T, n, n)
    F = qr(G)
    d = [F.R[i, i] / abs(F.R[i, i]) for i in 1:n]
    return Matrix(F.Q) * Diagonal(d)
end

"""
    randsvd(rng, ::Type{T}, n, κ; rank = n)

Construct a random `n × n` matrix `A` of element type `T` with singular values
geometrically distributed from `1` down to `1 / κ` on indices `1:rank` and zero on
`(rank + 1):n`, using Haar-distributed orthogonal or unitary factors `U` and `V`.
Returns `(A, U, σ, V)`.
"""
function randsvd(
        rng::AbstractRNG, ::Type{T}, n::Integer, κ::Real; rank::Integer = n
) where {T}
    R = real(T)
    κ_R = R(κ)
    U = _haar_unitary(rng, T, n)
    V = _haar_unitary(rng, T, n)
    σ = R[i <= rank ? κ_R^(-R(i - 1) / R(max(1, rank - 1))) : zero(R) for i in 1:n]
    A = U * Diagonal(T.(σ)) * V'
    return A, U, σ, V
end

"""
    randhermitian(rng, ::Type{T}, n, κ)

Construct a random `n × n` Hermitian positive-definite matrix `H` of element type `T`
with eigenvalues geometrically distributed from `1` down to `1 / κ`, using a
Haar-distributed orthogonal or unitary eigenvector factor `Q`.
Returns `(H, λ, Q)`.
"""
function randhermitian(rng::AbstractRNG, ::Type{T}, n::Integer, κ::Real) where {T}
    R = real(T)
    κ_R = R(κ)
    Q = _haar_unitary(rng, T, n)
    λ = R[κ_R^(-R(i - 1) / R(max(1, n - 1))) for i in 1:n]
    H0 = Q * Diagonal(T.(λ)) * Q'
    H = Matrix{T}(Hermitian((H0 + H0') / 2))
    return H, λ, Q
end

"""
    minnorm_solution(U, σ, V, b, r)

Return the exact minimum-norm least-squares solution `V[:, 1:r] * (Diagonal(σ[1:r]) \\ (U[:, 1:r]' * b))`
for a rank-`r` matrix `U * Diagonal(σ) * V'`.
"""
function minnorm_solution(
        U::AbstractMatrix, σ::AbstractVector, V::AbstractMatrix, b::AbstractVector, r::Integer
)
    Ut_b = (@view U[:, 1:r])' * b
    y = Ut_b ./ (@view σ[1:r])
    return (@view V[:, 1:r]) * y
end
