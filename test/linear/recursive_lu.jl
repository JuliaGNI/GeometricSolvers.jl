using GeometricSolvers
using GeometricSolvers: DenseLU, RECURSIVE_LU_BASE, dense_factor, factorize!!, ldiv!!,
                        recursive_lu!, use_recursive_lu
using LinearAlgebra: BlasInt, I, LAPACK, UnitLowerTriangular, UpperTriangular, opnorm
using Random: Xoshiro, randperm
using Test

include("../helpers/matrix.jl")

# The band `[lo, hi]` of `n` in which `LUFactorization()` factorises a host `Matrix` by the
# recursion, per element type, from the measurement of `scripts/hardware/rl_host_lu.jl`.
const RL_BANDS = Dict(Float32 => (8, 32), Float64 => (8, 32), ComplexF64 => (8, 32))

# Each working element type with its own factor (`TF = Nothing`), and the `Float32` factor of a
# `Float64` working precision, real and complex.
const RL_PAIRS = [(T, TF) for T in ELTYPES
                  for TF in (Nothing, Float32)
                  if TF === Nothing || real(T) === Float64]

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

# The exact solution `x` and the right-hand side `b = A x`, rounded once from `BigFloat`, and
# `κ∞(A)`, in `BigFloat`.
function rl_system(rng, A::Matrix{T}) where {T}
    n = size(A, 1)
    x = randn(rng, T, n)
    Ab = big.(A)
    b = T.(Ab * big.(x))
    κ = opnorm(Ab, Inf) * opnorm(inv(Ab), Inf)
    return x, b, κ
end

rl_relerr(y, x) = maximum(abs, y .- x) / maximum(abs, x)

# The pivots of `LAPACK.getrf!` on a copy of `A`, with its factor and `info`.
function rl_getrf(A)
    G, ipiv = copy(A), Vector{BlasInt}(undef, size(A, 1))
    _, _, info = LAPACK.getrf!(G, ipiv; check = false)
    return G, ipiv, Int(info)
end

function rl_factorize_allocs(fac::F, m::M, A::A_) where {F, M, A_}
    factorize!!(fac, m, A)
    return @allocated factorize!!(fac, m, A)
end

rl_allocs(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "recursive_lu! on A = Pᵀ L U agrees with getrf! (T = $T, n = $n, base = $base)" for T in ELTYPES,
    n in (8, 16, 32, 35, 64),
    base in (4, RECURSIVE_LU_BASE)

    rng = Xoshiro(n + 1000base)
    A, p = rl_designed(rng, T, n)
    F, ipiv = copy(A), Vector{BlasInt}(undef, n)
    info = recursive_lu!(F, ipiv, base)
    _, ipiv_g, info_g = rl_getrf(A)
    @test info == info_g == 0
    @test ipiv == ipiv_g
    @test rl_row_permutation(ipiv) == p
    @test rl_berr(A, F, ipiv) <= 8n * eps(real(T))

    # the solve with the recursion's factor and LAPACK's `getrs!`, within `8n·eps·κ∞`
    x, b, κ = rl_system(rng, A)
    LAPACK.getrs!('N', F, ipiv, b)
    @test rl_relerr(b, x) <= 8n * eps(real(T)) * κ
end

@testset "use_recursive_lu: the measured band and nothing else" begin
    for (T, (lo, hi)) in RL_BANDS
        @test !use_recursive_lu(T, 0)
        @test !use_recursive_lu(T, 1)
        @test !use_recursive_lu(T, lo - 1)
        @test use_recursive_lu(T, lo)
        @test use_recursive_lu(T, hi)
        @test !use_recursive_lu(T, hi + 1)
        @test all(n -> use_recursive_lu(T, n), lo:hi)
    end
    # a type without a measured band stays on `getrf!`, and `BigFloat` on the generic `lu!`
    for n in (0, 1, 8, 16, 32, 64)
        for T in filter(T -> !haskey(RL_BANDS, T), ELTYPES)
            @test !use_recursive_lu(T, n)
        end
        @test !use_recursive_lu(BigFloat, n)
        @test !use_recursive_lu(Float16, n)
    end
end

@testset "LUFactorization() on a host Matrix (T = $T, TF = $TF, n = $n)" for (T, TF) in RL_PAIRS,
    n in (7, 8, 32, 33)

    m = TF === Nothing ? LUFactorization() : LUFactorization(TF)
    F = TF === Nothing ? T : (T <: Complex ? complex(TF) : TF)
    rng = Xoshiro(7000 + n)
    A, _ = rl_designed(rng, T, n)
    x, b, κ = rl_system(rng, A)
    fac = dense_factor(m, nothing, zeros(T, n), nothing)
    fac = factorize!!(fac, m, A)
    @test fac isa DenseLU
    @test fac.info == 0

    # the factor of the kernel that the band selects, from the matrix that `load_factors!` wrote
    A_F = F === T ? A : F.(A .* ldexp(one(real(T)), -fac.scale_exponent))
    K, ipiv_k = copy(A_F), Vector{BlasInt}(undef, n)
    if use_recursive_lu(F, n)
        @test recursive_lu!(K, ipiv_k, RECURSIVE_LU_BASE) == 0
    else
        LAPACK.getrf!(K, ipiv_k; check = false)
    end
    @test fac.factors == K
    @test fac.ipiv == ipiv_k == rl_getrf(A_F)[2]
    @test rl_berr(A_F, fac.factors, fac.ipiv) <= 8n * eps(real(F))

    # the solve through `ldiv!!`, the mixed method where `F ≠ T`, within `8n·eps(F)·κ∞`
    d = similar(b)
    @test ldiv!!(d, fac, b) === d
    @test rl_relerr(d, x) <= 8n * eps(real(F)) * κ

    # a zero last column, past the first half of the recursion: its `info` reaches the `DenseLU`
    A[:, n] .= 0
    fac = factorize!!(fac, m, A)
    @test fac.info == rl_getrf(F.(A))[3] == n
end

@testset "factorize!! in the band: inferred and allocation-free (T = $T)" for T in filter(
    T -> haskey(RL_BANDS, T), ELTYPES)
    n = 32
    m = LUFactorization()
    A, _ = rl_designed(Xoshiro(41), T, n)
    fac = dense_factor(m, nothing, zeros(T, n), nothing)
    @test use_recursive_lu(T, n)
    fac = @inferred factorize!!(fac, m, A)
    @test rl_factorize_allocs(fac, m, A) == 0
    @test rl_factorize_allocs(fac, m, randn(Xoshiro(1), T, n, n)) == 0
    @test rl_allocs(collect, 1:3) > 0
end

@testset "a zero pivot gives info as getrf! (T = $T, n = $n, base = $base, zero column $j)" for T in ELTYPES,
    (n, base, j) in ((8, 4, 6), (8, 4, 3), (20, RECURSIVE_LU_BASE, 15), (32, 4, 32))

    A = randn(Xoshiro(j + 100n), T, n, n)
    A[:, j] .= 0
    F, ipiv = copy(A), Vector{BlasInt}(undef, n)
    info = recursive_lu!(F, ipiv, base)
    _, _, info_g = rl_getrf(A)
    @test info == info_g == j

    # two zero columns: the first gives `info`
    A[:, n] .= 0
    j < n && @test recursive_lu!(copy(A), Vector{BlasInt}(undef, n), base) == j
end

@testset "the pivot of largest |Re| + |Im|, first maximum" begin
    # `abs(1 + im) < 1.5 < |1| + |1|`: pivoting by `abs` keeps row 1, LAPACK takes row 2
    A = ComplexF64[1.5 0; 1+1im 1]
    F, ipiv = copy(A), Vector{BlasInt}(undef, 2)
    @test recursive_lu!(F, ipiv, 1) == 0
    @test ipiv == rl_getrf(A)[2] == [2, 2]

    # a tie: the first maximum
    B = ComplexF64[1+1im 0; 2 1]
    @test (recursive_lu!(copy(B), ipiv, 1); ipiv) == rl_getrf(B)[2] == [1, 2]
    C = Float64[-2 0; 2 1]
    @test (recursive_lu!(copy(C), ipiv, 1); ipiv) == rl_getrf(C)[2] == [1, 2]
end

@testset "the edges: n = 0, n = 1, NaN and Inf (T = $T)" for T in ELTYPES
    @test recursive_lu!(zeros(T, 0, 0), BlasInt[], RECURSIVE_LU_BASE) == 0
    F, ipiv = fill(T(3), 1, 1), Vector{BlasInt}(undef, 1)
    @test recursive_lu!(F, ipiv, RECURSIVE_LU_BASE) == 0
    @test ipiv == [1] && F == fill(T(3), 1, 1)
    @test recursive_lu!(zeros(T, 1, 1), ipiv, RECURSIVE_LU_BASE) == 1

    n = 32
    for v in (T(NaN), T(Inf))
        A, _ = rl_designed(Xoshiro(5), T, n)
        A[20, 3] = v
        F, ipiv = copy(A), Vector{BlasInt}(undef, n)
        @test recursive_lu!(F, ipiv, 4) == 0
        @test !all(isfinite, F)
    end
end

@testset "a NaN is no zero pivot, and the pivot only on the diagonal (T = $T, base = $base)" for T in ELTYPES,
    base in (4, RECURSIVE_LU_BASE)

    n = 16
    # a zero first column with a `NaN` below the diagonal: the search by `>` skips the `NaN`, so
    # the pivot is the zero on the diagonal
    A = randn(Xoshiro(base), T, n, n)
    A[:, 1] .= 0
    A[5, 1] = T(NaN)
    F, ipiv = copy(A), Vector{BlasInt}(undef, n)
    @test recursive_lu!(F, ipiv, base) == 1
    @test !all(isfinite, F)

    # a `NaN` on the diagonal and zeros below: the `NaN` is the pivot, and no zero pivot
    A[:, 1] .= 0
    A[1, 1] = T(NaN)
    F = copy(A)
    @test recursive_lu!(F, ipiv, base) == 0
    @test !all(isfinite, F)

    # a `NaN` on the diagonal and nonzero entries below: no entry is `>` the `NaN`, so the first
    # maximum stays in row 1
    A = randn(Xoshiro(base), T, n, n)
    A[1, 1] = T(NaN)
    @test all(!iszero, A[2:n, 1])
    F = copy(A)
    @test recursive_lu!(F, ipiv, base) == 0
    @test ipiv[1] == 1
    @test !all(isfinite, F)
end

@testset "a subnormal pivot divides, as getf2 (T = $T)" for T in ELTYPES
    # `1 / s` overflows for a subnormal `s`: the multiplier is `(s / 2) / s = 1/2`, not `Inf`.
    # OpenBLAS's `getrf!` leaves this multiplier unscaled, so its factor is no reference here.
    s = 4 * nextfloat(zero(real(T)))
    A = T[s 1; s/2 1]
    F, ipiv = copy(A), Vector{BlasInt}(undef, 2)
    @test recursive_lu!(F, ipiv, 1) == 0
    @test ipiv == rl_getrf(A)[2] == [1, 2]
    @test F == T[s 1; 1//2 1//2]
    @test rl_berr(A, F, ipiv) <= 8 * 2 * eps(real(T))
end

@testset "recursive_lu! rejects a malformed call" begin
    @test_throws r"^DimensionMismatch: the LU factors have size \(3, 4\), expected a square matrix$" recursive_lu!(
        zeros(3, 4), Vector{BlasInt}(undef, 3), 2)
    @test_throws r"^DimensionMismatch: ipiv has length 2, expected 3$" recursive_lu!(
        zeros(3, 3), Vector{BlasInt}(undef, 2), 2)
    @test_throws r"^ArgumentError: the LU factors must have contiguous columns$" recursive_lu!(
        view(zeros(6, 3), 1:2:5, :), Vector{BlasInt}(undef, 3), 2)
    @test_throws r"^ArgumentError: the base case must have at least one column, got 0$" recursive_lu!(
        zeros(3, 3), Vector{BlasInt}(undef, 3), 0)
end
