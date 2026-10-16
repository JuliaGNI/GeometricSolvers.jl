# The R1 one-sided Jacobi SVD of `src/linear/static/svd.jl`: the two traps that §5.2 names for
# the port — a subnormal `ldexp` overflow, and a per-pair relative test that does not terminate
# on a rank-deficient matrix — the rank-revealing solve, and the checks of a kernel path.
#
# The SVD runs in `Float32`, `Float64`, `ComplexF32` and `ComplexF64`, at the working precision
# and with `TF = Float32` for the two `Float64` element types (`PRECISION_PAIRS`), over the array
# backends and on the KernelAbstractions `CPU()` backend of `test/helpers/matrix.jl`.

using GeometricSolvers
using GeometricSolvers: factorize!!, ldiv!!, returncode, StaticSVD
using JET: JET
using LinearAlgebra: Diagonal, norm, opnorm, qr
using Random: Xoshiro, randn
using StaticArrays: MMatrix, SMatrix, SVector, setindex
using Test

include("../../helpers/matrix.jl")
include("../../helpers/staticsystems.jl")

svd_method(::Type{TF}) where {TF} = SVDFactorization{TF}()

"""
    zero_cutoff(n, T, TF)

`√n · eps(TF)`, the relative threshold at which `factorize!!` counts a singular value as zero:
`σᵢ ≤ √n · eps(TF) · σ₁` is the stopping tolerance of the sweep and the rank cutoff of the solve.
"""
function zero_cutoff(n, ::Type{T}, ::Type{TF}) where {T, TF}
    sqrt(real(factortype(T, TF))(n)) * epsf(T, TF)
end

"""
    integers(T, n)

The `n × n` matrix of the small integers `i + j` in `T`, for check 1.
"""
integers(::Type{T}, n) where {T} = SMatrix{n, n, T}([i + j for i in 1:n, j in 1:n])

"""
    rank_deficient(seed, T, n, r)

`U Σ Vᴴ` in `T` with `U` and `V` unitary and `Σ` holding the `r` values `1, 1/2, …` and zeros,
so that the stored matrix has the numerical rank `r` at `√n · eps` for every element type.
"""
function rank_deficient(seed, ::Type{T}, n, r) where {T}
    rng = Xoshiro(seed)
    U = SMatrix{n, n, T}(Matrix(qr(randn(rng, T, n, n)).Q))
    V = SMatrix{n, n, T}(Matrix(qr(randn(rng, T, n, n)).Q))
    Σ = real(T)[i ≤ r ? real(T)(0.5)^(i - 1) : zero(real(T)) for i in 1:n]
    return SMatrix{n, n, T}(U * Diagonal(Σ) * V')
end

@testset "check 1: the subnormal ldexp trap gives finite factors and an exact prescale" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = svd_method(TF)
        R = real(factortype(T, TF))
        # the scale of check 1: `emin + 8` below the smallest normal number of `R`, where `emin`
        # is the exponent of `floatmin(R)`, which is the `2^-(emin + 8)` of the plan. Every entry
        # of `2^E · B` is subnormal, `B`'s entries being below 2^8
        E = exponent(floatmin(R)) - 8
        B = integers(T, n)
        A = ldexp(one(R), E) * B
        @test all(x -> abs(x) < floatmin(R), A)

        # the factors are finite: the exponent of the prescale is clamped to the range of `R`,
        # where an unclamped `ldexp(one(R), -exponent(m) - 1)` is `Inf` for a subnormal `m`
        fac = factorize!!(nothing, m, A)
        @test all(isfinite, fac.factors)
        @test all(isfinite, fac.V)
        @test all(isfinite, fac.σ)

        # the power of two is exact, so the sweep runs on the integer matrix scaled, and the
        # orthogonal factors are the integer matrix's bit for bit, as the singular values are
        # `2^E` times its
        ref = factorize!!(nothing, m, B)
        @test fac.factors == ref.factors
        @test fac.V == ref.V
        @test fac.σ == map(x -> ldexp(x, E), ref.σ)
    end
end

@testset "check 2: a rank-deficient matrix ends below the sweep cap with the exact rank" begin
    @testset "$T, TF = $TF, rank $r" for (T, TF) in PRECISION_PAIRS, r in (5, 1, 0)

        n = 16
        m = svd_method(TF)
        tol = zero_cutoff(n, T, TF)
        A = rank_deficient(100r + n, T, n, r)
        fac = factorize!!(nothing, m, A)

        # the per-pair relative test alone never terminates here: the null directions are asked
        # to become orthogonal to each other, and every rotation regenerates their noise
        @test fac.converged
        @test issorted(fac.σ; rev = true)
        # the numerical rank is exact, and `info` is the index of the first singular value that
        # the cutoff counts as zero
        @test fac.info == r + 1
        @test returncode(fac) === SINGULAR
        @test all(σ -> σ ≤ tol * fac.σ[1], fac.σ[(r + 1):end])
        if r >= 1
            @test fac.σ[r] > tol * fac.σ[1]
        end
    end
end

@testset "the negligible-column guard skips a column that carries nothing" begin
    @testset "$T, TF = $TF" for (T, TF) in PRECISION_PAIRS

        # The guard of §5.2, which the relative test of a pair does not replace: a column whose
        # squared norm is at or below `αmax · tol²` is skipped, so a column *below* the rank
        # cutoff is left where it is instead of being rotated into the column it is parallel to.
        # Without the guard the relative test rotates the pair — `|γ| = ε/2` against
        # `tol · √(αβ) = eps²` — and the reported singular value of the last column becomes the
        # rounding noise of that rotation, `~ eps²`, instead of the `ε = eps(TF)/2` the column
        # carries. The sweep's `σ` are the column norms, and the rank cutoff truncates this one
        # either way, which is why the two agree on the rank of this matrix.
        n = 4
        F = factortype(T, TF)
        ε = eps(real(F)) / 2
        # the identity with its last diagonal entry zeroed and a single ε in the first row of the
        # last column: that column is `ε e₁`, whose squared norm is `eps²/4`, below the
        # `αmax · tol² = 4 eps²` of the guard, and parallel to the first column
        A = SMatrix{n, n, T}([i == j ? one(T) : zero(T) for i in 1:n, j in 1:n])
        A = setindex(setindex(A, zero(T), n, n), T(ε), 1, n)
        fac = factorize!!(nothing, svd_method(TF), A)
        @test fac.converged
        @test fac.info == n
        @test fac.σ[n] == ε
    end
end

@testset "a non-finite entry never throws and reports the sweep cap" begin
    @testset "$T, TF = $TF" for (T, TF) in PRECISION_PAIRS
        m = svd_method(TF)
        # as for the factorisations: no finiteness check, and nothing throws. The relative test
        # of a pair is never satisfied on a `NaN`, so the sweep reaches its cap.
        A = SMatrix{4, 4, T}(fill(T(NaN), 4, 4))
        fac = factorize!!(nothing, m, A)
        @test !fac.converged
        @test fac.info == 1
    end
end

@testset "check 3: the solve of a rank-deficient system and a full-rank system" begin
    @testset "a consistent rank-deficient system, $T, TF = $TF, rank $r" for (T, TF) in PRECISION_PAIRS,
        r in (5, 1, 0)

        n = 16
        m = svd_method(TF)
        A = rank_deficient(100r + n, T, n, r)
        x₀ = SVector{n, T}(randn(Xoshiro(400r + n), T, n))
        b = A * x₀
        fac = factorize!!(nothing, m, A)
        x = ldiv!!(nothing, fac, b)

        @test returncode(fac) === SINGULAR
        @test eltype(x) === T
        @test all(isfinite, x)
        # the solve is not rank-revealing: the truncated step is not `x₀`, but its residual is
        # bounded by the discarded directions, and an exactly zero singular value never divides
        @test norm(A * x - b, Inf) <= 4 * n * epsf(T, TF) * opnorm(A, Inf) * norm(x, Inf)
    end

    @testset "a full-rank system meets part I's bound, $T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS,
        n in STATIC_SIZES

        m = svd_method(TF)
        systems = conditioned_systems(1000n + 3, T, TF, n)
        @test !isempty(systems)
        for (A, b) in systems
            fac, x = host_solve(m, A, b)
            @test returncode(fac) === SUCCESS
            @test eltype(x) === T
            @test relerr(x, reference(A, b)) <= bound(A, T, TF)
        end
    end
end

@testset "the SVD factor is isbits, inferred and allocation-free" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = svd_method(TF)
        F = factortype(T, TF)
        A, b = first(conditioned_systems(1000n + 3, T, TF, n))
        fac = @inferred factorize!!(nothing, m, A)
        @test fac isa StaticSVD{n, F, real(F), n * n}
        @test isbits(fac)
        @test eltype(fac.factors) === F
        @test eltype(fac.σ) === real(F)
        @test @inferred(ldiv!!(nothing, fac, b)) isa SVector{n, T}
        # under --check-bounds=yes a bounds check lets the local MMatrix escape to the heap, so a
        # run that forces it (Pkg.test() up to Julia 1.12) does not check these
        if Base.JLOptions().check_bounds != 1
            @test factorize_allocations(m, A) == 0
            @test ldiv_allocations(fac, b) == 0
            @test collect_allocations(b) > 0
        end
        # an MMatrix Jacobian and a previous factor give the same factors
        @test factorize!!(nothing, m, MMatrix(A)) === fac
        @test factorize!!(fac, m, A) === fac
    end

    @testset "one thread per system in a kernel on $AT gives the host result bit for bit" for AT in ARRAY_BACKENDS
        for (T, TF) in PRECISION_PAIRS, n in (1, 4)

            m = svd_method(TF)
            systems = conditioned_systems(1000n + 3, T, TF, n)
            # the zero matrix and the rank-1 matrix of check 2, which the kernel's `info` reports.
            # `vcat` of two vectors: a `[...; matrix; ...]` literal concatenates a matrix's
            # elements when its size is 1, which the n = 1 case of this loop is
            As = vcat(first.(systems),
                [rank_deficient(100n + 1, T, n, 0), rank_deficient(100n + 5, T, n, 1)])
            bs = vcat(last.(systems), [zero(SVector{n, T}), SVector{n, T}(fill(one(T), n))])
            x, info = kernel_solve(AT, m, As, bs)
            host = map((A, b) -> host_solve(m, A, b), As, bs)
            @test isequal(x, map(last, host))
            @test info == map(h -> first(h).info, host)
            # the zero matrix is singular at every size; a rank-1 matrix is full rank at n = 1,
            # and its `info` is 0 there
            @test info[length(systems) + 1] > 0
            n > 1 && @test info[length(systems) + 2] > 0
        end
    end

    @testset "JET: factorize!! and ldiv!! are optimisable and error-free" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for (T, TF) in PRECISION_PAIRS, n in (1, 4)

                m = svd_method(TF)
                F = factortype(T, TF)
                A, b = first(conditioned_systems(1000n + 3, T, TF, n))
                JET.@test_opt factorize!!(nothing, m, A)
                JET.@test_call factorize!!(nothing, m, A)
                fac = factorize!!(nothing, m, A)
                JET.@test_opt ldiv!!(nothing, fac, b)
                JET.@test_call ldiv!!(nothing, fac, b)
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
