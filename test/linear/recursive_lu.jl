using GeometricSolvers
using GeometricSolvers: recursive_lu!
using LinearAlgebra: BlasInt, I, LAPACK, UnitLowerTriangular, UpperTriangular, opnorm
using Random: Xoshiro, randperm
using Test

include("../helpers/matrix.jl")

# The permutation `p` of the rows that the exchanges `ipiv` apply, so that `A[p, :] = L U`.
function rl_row_permutation(ipiv::AbstractVector)
    p = collect(1:length(ipiv))
    for k in eachindex(ipiv)
        q = Int(ipiv[k])
        p[k], p[q] = p[q], p[k]
    end
    return p
end

# An entry of `L` with `|Re| + |Im| ≤ 1/2`, so that the diagonal 1 of `L` is the pivot of each
# column by a margin of 1/2.
rl_multiplier(rng, ::Type{T}) where {T <: Real} = T(rand(rng) - 0.5)
rl_multiplier(rng, ::Type{T}) where {T <: Complex} = T(rand(rng) - 0.5, rand(rng) - 0.5) / 2

# `A = Pᵀ L U`, `A[p, :] = L U`, with `|L| ≤ 1/2` below its unit diagonal, `U` well conditioned,
# and `p` a permutation that exchanges the first and the last row, across every split.
function rl_designed(rng, ::Type{T}, n::Int) where {T}
    L = Matrix{T}(I, n, n)
    for j in 1:n, i in (j + 1):n

        L[i, j] = rl_multiplier(rng, T)
    end
    U = zeros(T, n, n)
    for j in 1:n, i in 1:j

        U[i, j] = i == j ? T(2 + rand(rng)) : rl_multiplier(rng, T)
    end
    p = randperm(rng, n)
    i1, in_ = findfirst(==(1), p), findfirst(==(n), p)
    p[1], p[i1] = p[i1], p[1]
    p[n], p[in_] = p[in_], p[n]
    p[1], p[n] = p[n], p[1]
    A = similar(L)
    A[p, :] = L * U
    return A, p
end

function rl_berr(A, F, ipiv)
    opnorm(
        A[rl_row_permutation(ipiv), :] -
        UnitLowerTriangular(F) * UpperTriangular(F), Inf) / opnorm(A, Inf)
end

@testset "recursive_lu! on A = Pᵀ L U agrees with getrf! (T = $T, n = $n, base = $base)" for T in ELTYPES,
    n in (8, 16, 35, 64),
    base in (4, 16)

    rng = Xoshiro(n + 1000base)
    A, p = rl_designed(rng, T, n)
    F, ipiv = copy(A), Vector{BlasInt}(undef, n)
    info = recursive_lu!(F, ipiv, base)
    G, ipiv_g = copy(A), Vector{BlasInt}(undef, n)
    _, _, info_g = LAPACK.getrf!(G, ipiv_g; check = false)
    @test info == info_g == 0
    @test ipiv == ipiv_g
    @test rl_row_permutation(ipiv) == p
    @test rl_berr(A, F, ipiv) <= 8n * eps(real(T))
end
