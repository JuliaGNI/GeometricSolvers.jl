# The R1 cyclic two-sided Jacobi eigensolver of `src/linear/static/eigen.jl` (decision 27): the
# rotations and the round-robin ordering of the Jacobi SVD of
# `src/linear/static/svd.jl`, applied to a Hermitian matrix from both sides.
#
# The check is the one of decision 27, `jacobi_eigen(im * M)` for a real skew `M`, which is the
# `ProjectedSkew` case: a real skew matrix has a zero diagonal, so the two diagonal entries of
# every pair are equal and the rotation is the full π/4 that a `sign` reports as no rotation at
# all. The eigenvalues and the eigenvectors are checked against a `BigFloat` `exp` of `M`, which
# no wrong rotation can pass, and against the eigenvalues of `LinearAlgebra`.
#
# The norms of the two checks of decision 27 are the Frobenius norm, and `max(1, ‖M‖)` takes the
# Frobenius norm of `M` as well.
#
# The function runs in `Float32`, `Float64`, `ComplexF32` and `ComplexF64` and on the array
# backends and the KernelAbstractions `CPU()` backend of `test/helpers/matrix.jl`.

using GeometricSolvers
using GeometricSolvers: jacobi_eigen, MAXITERS, ReturnCode, SUCCESS
using JET: JET
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using LinearAlgebra: Diagonal, Eigen, Hermitian, I, eigen, eigvals, norm, opnorm, qr
using Random: Xoshiro, randn
using StaticArrays: MMatrix, SMatrix, SVector
using Test

include("../../helpers/matrix.jl")
include("../../helpers/staticsystems.jl")

"""
    skew_matrix(seed, T, N)

A skew `N × N` matrix in `T` — real, so that `im * M` is Hermitian with a zero diagonal.
"""
function skew_matrix(seed, ::Type{T}, N) where {T}
    B = randn(Xoshiro(seed), T, N, N)
    return SMatrix{N, N, T}(B - B')
end

"""
    equal_blocks(seed, T, n)

The `2n × 2n` skew matrix of two copies of the same `n × n` block, so that every eigenvalue of
`im * M` has multiplicity two.
"""
function equal_blocks(seed, ::Type{T}, n) where {T}
    B = skew_matrix(seed, T, n)
    M = zero(MMatrix{2n, 2n, T})
    for j in 1:n, i in 1:n

        M[i, j] = B[i, j]
        M[n + i, n + j] = B[i, j]
    end
    return SMatrix(M)
end

"""
    hermitian_matrix(seed, T, N)

A Hermitian `N × N` matrix in `T`, real symmetric for a real `T`.
"""
function hermitian_matrix(seed, ::Type{T}, N) where {T}
    B = randn(Xoshiro(seed), T, N, N)
    return SMatrix{N, N, T}(B + B')
end

"""
    rank_deficient_hermitian(seed, T, N, r)

A Hermitian `N × N` matrix in `T` of rank `r`, with the non-zero eigenvalues `1:r`. A pair of
diagonal entries that both hold the rounding noise of the rotations is what a relative pair test
alone does not terminate on (check 2 of §5.2, which the SVD guards).
"""
function rank_deficient_hermitian(seed, ::Type{T}, N, r) where {T}
    Q = SMatrix{N, N, T}(Matrix(qr(randn(Xoshiro(seed), T, N, N)).Q))
    d = [i ≤ r ? real(T)(i) : zero(real(T)) for i in 1:N]
    H = Q * Diagonal(d) * Q'
    return SMatrix{N, N, T}((H + H') / 2)
end

"""
    low_rank_skew(seed, T, N, r)

A real skew `N × N` matrix in `T` of rank `r`, an even `r`, as `r/2` rank-2 skew terms. Its null
directions hold nothing but rounding noise, which is the `ProjectedSkew` shape the whole check 4
is for.
"""
function low_rank_skew(seed, ::Type{T}, N, r) where {T}
    rng = Xoshiro(seed)
    M = zeros(T, N, N)
    for _ in 1:(r ÷ 2)
        u = randn(rng, T, N)
        v = randn(rng, T, N)
        M .+= u * v' .- v * u'
    end
    return SMatrix{N, N, T}(M)
end

"""
    floor_skew(T, N)

A real skew `N × N` matrix in `T` with `M[1, 2] = 1`, `M[2, 1] = -1` and `5 · eps(T)` on every
off-diagonal pair of `3:N`, so that each entry of that block sits above the `eps(T)` pair floor
and below `√N · eps(T)` at `N = 16`.
"""
function floor_skew(::Type{T}, N) where {T}
    M = zero(MMatrix{N, N, T})
    M[1, 2] = one(T)
    M[2, 1] = -one(T)
    δ = 5 * eps(T)
    for j in 3:N, i in (j + 1):N

        M[i, j] = δ
        M[j, i] = -δ
    end
    return SMatrix(M)
end

"""
    big_exp(A)

`exp(A)` in `BigFloat`, real or complex, by scaling and squaring with a Taylor series:
`LinearAlgebra`'s `exp` has no method for a `BigFloat` matrix, and the reference of the check
must not be the eigendecomposition it judges. `A` is halved until its ∞-norm is at most 1/2,
which leaves the 50 terms of the series an error far below the tolerance of the check.
"""
function big_exp(A::AbstractMatrix{T}) where {T <: Union{BigFloat, Complex{BigFloat}}}
    n = size(A, 1)
    B = copy(A)
    k = 0
    while opnorm(B, Inf) > big"0.5"
        B = B ./ 2
        k += 1
    end
    E = Matrix{T}(I, n, n)
    term = copy(E)
    for j in 1:50
        term = term * B ./ j
        E = E + term
    end
    for _ in 1:k
        E = E * E
    end
    return E
end

@testset "check 4: jacobi_eigen on im·M for a real skew M, and on a Hermitian matrix" begin
    @testset "a real skew M, $T, N = $N, $(form)" for T in REAL_ELTYPES,
        (N, form, seed) in ((2, "one block", 21), (16, "eight blocks", 23),
            (8, "equal blocks", 24), (16, "near-floor block", 0))

        M = if form == "equal blocks"
            equal_blocks(seed, T, N ÷ 2)
        elseif form == "near-floor block"
            floor_skew(T, N)
        else
            skew_matrix(seed, T, N)
        end
        H = im * M
        λ, V, status = jacobi_eigen(H)
        @test status === SUCCESS
        @test eltype(λ) === T
        @test eltype(V) === complex(T)

        # the eigenvectors are unitary to working precision, which is what `ProjectedSkew` is for
        @test norm(V' * V - I) <= 4 * N * eps(T)

        # `exp(M) = Re(V exp(-i Λ) Vᴴ)`: `im * M` is the Hermitian `V Λ Vᴴ`, so `M` is
        # `V (-i Λ) Vᴴ`. The reference is `exp` of the stored `M` in `BigFloat`, and `M` is real,
        # so its exponential is real.
        Mb = big.(Matrix(M))
        Vb = big.(Matrix(V))
        Λb = big.(Vector(λ))
        @test norm(real(Vb * Diagonal(exp.(-im .* Λb)) * Vb') - big_exp(Mb)) <=
              4 * N * eps(T) * max(1, norm(Mb))

        # and the eigenvalues are those of the Hermitian matrix, to working precision
        scale = max(1, opnorm(Mb, Inf))
        @test isapprox(sort(Vector(λ)), sort(eigvals(Hermitian(Matrix(H))));
            rtol = 8 * N * eps(T) * scale, atol = 8 * N * eps(T) * scale)
    end

    @testset "a Hermitian matrix, $T, N = $N" for T in ELTYPES,
        (N, seed) in ((2, 31), (16, 32))

        H = hermitian_matrix(seed, T, N)
        λ, V, status = jacobi_eigen(H)
        @test status === SUCCESS
        @test eltype(λ) === real(T)
        @test norm(V' * V - I) <= 4 * N * eps(real(T))

        Hb = big.(Matrix(H))
        Vb = big.(Matrix(V))
        Λb = big.(Vector(λ))
        # `exp(H)` amplifies a perturbation of the eigenvalues by a factor of order `‖exp(H)‖`,
        # which for a Hermitian matrix of a spread spectrum is `exp(λmax)` and not 1: the bound
        # of decision 27, written for the skew case where `‖exp(M)‖` is that of an orthogonal
        # matrix, carries that factor here (measured: the error at N = 16 in `Float64` is 3.2e-11
        # against 3.2e-13 without it, with `‖exp(H)‖ = 8.5e4`)
        E = big_exp(Hb)
        @test norm(Vb * Diagonal(exp.(Λb)) * Vb' - E) <=
              4 * N * eps(real(T)) * max(1, norm(Hb)) * max(1, norm(E))

        scale = max(1, opnorm(Hb, Inf))
        @test isapprox(sort(Vector(λ)), sort(eigvals(Hermitian(Matrix(H))));
            rtol = 8 * N * eps(real(T)) * scale, atol = 8 * N * eps(real(T)) * scale)
    end

    @testset "a non-finite entry never throws and reports the sweep cap, $T" for T in ELTYPES
        # as for the factorisations: no finiteness check, and nothing throws
        A = SMatrix{2, 2, T}(fill(T(NaN), 2, 2))
        λ, _, status = jacobi_eigen(A)
        @test status === MAXITERS
        @test !all(isfinite, λ)
        # the same for a diagonal `Inf`, whose off-diagonal entries are exactly zero: the scale
        # of the pair test is not finite either, so no pair is skipped and the rotation of the
        # pair is not finite
        for v in (T(Inf), -T(Inf))
            B = SMatrix{2, 2, T}(v, zero(T), zero(T), one(T))
            λb, _, statusb = jacobi_eigen(B)
            @test statusb === MAXITERS
            @test !all(isfinite, λb)
        end
        # at N = 1 there is no off-diagonal pair, so a NaN or an Inf returns SUCCESS
        for v in (T(NaN), T(Inf), -T(Inf))
            λ1, V1, status1 = jacobi_eigen(SMatrix{1, 1, T}(v))
            @test status1 === SUCCESS
            @test !isfinite(λ1[1])
            @test V1 == one(SMatrix{1, 1, T})
        end
    end
end

@testset "the stopping test terminates on a rank-deficient or a low-rank matrix" begin
    # A pair of diagonal entries that hold nothing but the rounding noise of the rotations never
    # satisfies a relative test — each rotation regenerates as much noise as it removes — so the
    # absolute floor `eps(R) * ‖H‖_F` of the pair test is what ends the sweep.
    @testset "a rank-deficient Hermitian matrix, $T, rank $r" for T in ELTYPES,
        r in (0, 1, 5)

        N = 16
        H = rank_deficient_hermitian(3, T, N, r)
        λ, V, status = jacobi_eigen(H)
        @test status === SUCCESS
        @test norm(V' * V - I) <= 4 * N * eps(real(T))
        scale = max(1, opnorm(Matrix(H), Inf))
        @test isapprox(sort(Vector(λ)), sort(eigvals(Hermitian(Matrix(H))));
            rtol = 8 * N * eps(real(T)) * scale, atol = 8 * N * eps(real(T)) * scale)
    end

    @testset "a low-rank skew M, $T, rank $r" for T in REAL_ELTYPES, r in (2, 4)

        N = 16
        H = im * low_rank_skew(7, T, N, r)
        λ, V, status = jacobi_eigen(H)
        @test status === SUCCESS
        @test norm(V' * V - I) <= 4 * N * eps(T)
        scale = max(1, opnorm(Matrix(H), Inf))
        @test isapprox(sort(Vector(λ)), sort(eigvals(Hermitian(Matrix(H))));
            rtol = 8 * N * eps(T) * scale, atol = 8 * N * eps(T) * scale)
    end

    @testset "a negligible off-diagonal beside a diagonal difference, $T" for T in REAL_ELTYPES

        # `ζ = (β − α)/2|γ|` squares to `Inf` for an `|γ|` at the bottom of the type's range, so
        # an unguarded rotation degenerates to the identity (`t = 0`); the pair floor skips any
        # `|γ| ≤ eps(T) * ‖H‖_F` on sweep 1.
        H = SMatrix{2, 2, T}(one(T), nextfloat(zero(T)), nextfloat(zero(T)), zero(T))
        λ, V, status = jacobi_eigen(H)
        @test status === SUCCESS
        @test norm(V' * V - I) <= 8 * eps(T)
        @test isapprox(sort(Vector(λ)), [zero(T), one(T)]; atol = 8 * eps(T))
    end

    @testset "the pair floor eps(R) * ‖H‖_F is pinned from both sides at N = 16, $T" for T in ELTYPES
        N = 16
        R = real(T)
        for unit in (T <: Complex ? (one(T), T(im)) : (one(T),))
            # below the floor: `|γ| = 0.75 * eps(R) <= eps(R) * ‖H‖_F` is skipped without rotation
            γ_below = unit * R(0.75) * eps(R)
            H_below = zero(MMatrix{N, N, T})
            H_below[1, 1] = one(T)
            H_below[2, 1] = γ_below
            H_below[1, 2] = conj(γ_below)
            _, V_below, status_below = jacobi_eigen(SMatrix(H_below))
            @test status_below === SUCCESS
            @test iszero(V_below[2, 1])
            @test V_below == one(SMatrix{N, N, T})

            # above the floor: `|γ| = 1.5 * eps(R) > eps(R) * ‖H‖_F` is rotated on sweep 1
            γ_above = unit * R(1.5) * eps(R)
            H_above = zero(MMatrix{N, N, T})
            H_above[1, 1] = one(T)
            H_above[2, 1] = γ_above
            H_above[1, 2] = conj(γ_above)
            _, V_above, status_above = jacobi_eigen(SMatrix(H_above))
            @test status_above === SUCCESS
            @test !iszero(V_above[2, 1])
            @test isapprox(abs(V_above[2, 1]), abs(γ_above); rtol = 8 * eps(R))
        end
    end
end

@testset "the lower triangle is read, and the upper triangle filled from it, $T" for T in ELTYPES
    # The matrix is assumed Hermitian, and the upper triangle is set to the conjugate of the lower
    # triangle on entry. `[1 2; 0 3]` is not Hermitian: its lower triangle is the diagonal
    # `[1 0; 0 3]`, so the call returns the eigenvalues 1 and 3.
    A = SMatrix{2, 2, T}(one(T), T(2), zero(T), T(3))
    λ, V, status = jacobi_eigen(A)
    H = Hermitian(A, :L)
    @test status === SUCCESS
    @test norm(V * Diagonal(λ) * V' - Matrix(H)) <= 8 * eps(real(T)) * norm(H)
end

@testset "check 5: the name collides with neither Eigen nor eigen, and pirates none" begin
    @test jacobi_eigen !== eigen
    @test jacobi_eigen !== Eigen
    @test :jacobi_eigen in names(GeometricSolvers)
    for T in ELTYPES
        A = hermitian_matrix(41, T, 4)
        # a method of `eigen` for a `StaticArrays` type would be type piracy (decision 38)
        @test isempty(methods(eigen, Tuple{typeof(A)}, GeometricSolvers))
    end
    @test isempty(filter(m -> m.module === GeometricSolvers, methods(eigen)))
end

# one matrix per thread, as the LU and QR tests do: the same rotations in a kernel give the
# host's eigenvalues and eigenvectors bit for bit
@kernel function eigen_kernel!(values, vectors, codes, @Const(A))
    i = @index(Global)
    λ, V, status = jacobi_eigen(A[i])
    n = size(A[i], 1)
    for j in 1:n
        values[j, i] = λ[j]
    end
    for j in 1:n, k in 1:n

        vectors[k, j, i] = V[k, j]
    end
    codes[i] = status === SUCCESS ? 0x01 : 0x00
end

"""
    kernel_eigen(AT, As)

`jacobi_eigen` on each matrix of `As`, one thread of a KernelAbstractions kernel per matrix, and
the values, the vectors and the status codes read back to the host.
"""
function kernel_eigen(AT, As)
    n = size(first(As), 1)
    T = eltype(first(As))
    A = AT(As)
    values = AT(zeros(real(T), n, length(As)))
    vectors = AT(zeros(T, n, n, length(As)))
    codes = AT(zeros(UInt8, length(As)))
    backend = KernelAbstractions.get_backend(A)
    eigen_kernel!(backend)(values, vectors, codes, A; ndrange = length(As))
    # `JLBackend` runs a kernel to completion at its launch, and has no `synchronize` method
    applicable(KernelAbstractions.synchronize, backend) &&
        KernelAbstractions.synchronize(backend)
    return Array(values), Array(vectors), Array(codes)
end

# the allocation assertion goes through a function barrier whose argument has a concrete type, and
# which calls the function once before it measures
function eigen_allocations(A::A_) where {A_}
    jacobi_eigen(A)
    return @allocated jacobi_eigen(A)
end

@testset "jacobi_eigen is isbits, inferred and allocation-free" begin
    @testset "$T, N = $N" for T in ELTYPES, N in (2, 16)

        A = hermitian_matrix(51, T, N)
        result = @inferred jacobi_eigen(A)
        @test result isa Tuple{SVector{N, real(T)}, SMatrix{N, N, T, N * N}, ReturnCode}
        @test isbits(result)
        if Base.JLOptions().check_bounds != 1
            @test eigen_allocations(A) == 0
            # the control: the allocation barrier sees an allocation where there is one
            @test collect_allocations(A) > 0
        end
        # an MMatrix input gives the same result
        @test jacobi_eigen(MMatrix(A)) === result
    end

    @testset "one matrix per thread in a kernel on $AT gives the host result bit for bit" for AT in ARRAY_BACKENDS
        for T in ELTYPES, N in (2, 8)

            As = [hermitian_matrix(60 + i, T, N) for i in 1:3]
            values, vectors, codes = kernel_eigen(AT, As)
            host = map(jacobi_eigen, As)
            @test all(==(0x01), codes)
            @test all(i -> isequal(values[:, i], Vector(host[i][1])), eachindex(As))
            @test all(i -> isequal(vectors[:, :, i], Matrix(host[i][2])), eachindex(As))
        end
    end

    @testset "JET: jacobi_eigen is optimisable and error-free" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for T in ELTYPES
                A = hermitian_matrix(51, T, 4)
                JET.@test_opt jacobi_eigen(A)
                JET.@test_call jacobi_eigen(A)
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
