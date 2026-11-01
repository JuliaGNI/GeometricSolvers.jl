# The R1 one-sided Jacobi SVD of `src/linear/static/svd.jl`: the two traps that §5.2 names for
# the port — a subnormal `ldexp` overflow, and a per-pair relative test that does not terminate
# on a rank-deficient matrix — the rank-revealing solve, and the checks of a kernel path.
#
# The SVD runs in `Float32`, `Float64`, `ComplexF32` and `ComplexF64`, at the working precision
# and with `TF = Float32` for the two `Float64` element types (`PRECISION_PAIRS`), over the array
# backends and on the KernelAbstractions `CPU()` backend of `test/helpers/matrix.jl`.

using GeometricSolvers
using GeometricSolvers: factorize!!, ldiv!!, returncode, singular_values, StaticSVD
using JET: JET
using LinearAlgebra: Diagonal, norm, opnorm
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
        @test all(isfinite, singular_values(fac))

        # the power of two is exact, so the sweep runs on the integer matrix scaled, and the
        # orthogonal factors are the integer matrix's bit for bit, as the singular values are
        # `2^E` times its
        ref = factorize!!(nothing, m, B)
        @test fac.factors == ref.factors
        @test fac.V == ref.V
        @test singular_values(fac) == map(x -> ldexp(x, E), singular_values(ref))
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
        @test fac.info == 0
        σ = singular_values(fac)
        @test issorted(σ; rev = true)
        # the numerical rank is exact, and a rank deficiency is no failure of the factorisation
        @test fac.rank == r
        @test returncode(fac) === SUCCESS
        @test all(s -> s ≤ tol * σ[1], σ[(r + 1):end])
        if r >= 1
            @test σ[r] > tol * σ[1]
        end
        # the columns of `U` past the rank are zeroed rather than normalized: the divisor of a
        # null direction is its rounding noise, and dividing by it can put a `NaN` into the factor
        @test all(isfinite, fac.factors)
        for j in (r + 1):n
            @test all(iszero, fac.factors[:, j])
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
        # carries. The sweep's `σ̂` are the column norms, and the rank cutoff truncates this one
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
        @test fac.info == 0
        @test fac.rank == n - 1
        @test singular_values(fac)[n] == ε
    end
end

@testset "a non-finite entry never throws and reports the sweep cap" begin
    @testset "$T, TF = $TF" for (T, TF) in PRECISION_PAIRS
        m = svd_method(TF)
        # as for the factorisations: no finiteness check, and nothing throws. The relative test
        # of a pair is never satisfied on a `NaN`, so the sweep reaches its cap.
        A = SMatrix{4, 4, T}(fill(T(NaN), 4, 4))
        fac = factorize!!(nothing, m, A)
        @test fac.info == 1
        @test fac.rank == 0
        @test returncode(fac) === SINGULAR
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

        @test returncode(fac) === SUCCESS
        @test fac.rank == r
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

@testset "check 2 and 4 of part I, for the SVD" begin
    # §13.J: "Tests: as in I" — the shapes of part I's `test/linear/static/lu.jl` and
    # `test/linear/static/qr.jl` and of `test/linear/factors.jl` over the SVD: `im·A`, whose real
    # parts are all zero (check 2), the scale family `s·A` (check 1), and a right-hand side of
    # `1e±40`, outside the range of `Float32` (check 4).
    @testset "im·A meets the bound of check 1, $T, TF = $TF" for (T, TF) in filter(
        p -> p[1] <: Complex, PRECISION_PAIRS)
        # a complex `T` alone: `im * A` has a complex factor element type for a real `A`, and a
        # right-hand side of the working type then has no `T ≠ TF` method that returns its type
        n = 4
        m = svd_method(TF)
        A = im * first(pivoting_matrices(2000n + 1, real(T), TF, n))
        b = SVector{n, T}(randn(Xoshiro(7n), T, n))
        fac, x = host_solve(m, A, b)
        @test returncode(fac) === SUCCESS
        @test eltype(x) === T
        @test all(isfinite, x)
        @test relerr(x, reference(A, b)) <= bound(A, T, TF)
    end

    @testset "the scale family s·A meets the bound of check 1, $T, TF = $TF" for (T, TF) in PRECISION_PAIRS
        n = 4
        m = svd_method(TF)
        A, b = first(conditioned_systems(1000n + 3, T, TF, n))
        for s in (exp10.(-8:4:8)..., extreme_scales(real(factortype(T, TF)))...)
            sA = real(T)(s) * A
            fac, x = host_solve(m, sA, b)
            @test returncode(fac) === SUCCESS
            @test all(isfinite, x)
            @test relerr(x, reference(sA, b)) <= bound(sA, T, TF)
        end
    end

    @testset "a right-hand side of 1e±40, $T with TF = $TF" for (T, TF) in filter(
        p -> p[2] !== Nothing, PRECISION_PAIRS)
        n = 4
        m = svd_method(TF)
        F = factortype(T, TF)
        A, b = first(conditioned_systems(4000n + 1, T, TF, n))
        fac = factorize!!(nothing, m, A)
        @test eltype(fac.factors) === F
        for s in (1e-40, 1e40)
            r = SVector{n, T}(s * b / maximum(abs, b))
            x = ldiv!!(nothing, fac, r)
            @test eltype(x) === T
            @test all(isfinite, x)
            @test relerr(x, reference(A, r)) <= bound(A, T, TF)
        end
    end

    @testset "a singular value above floatmax(R) leaves the solve finite, $T" for T in REAL_ELTYPES

        # Every entry of `A` is finite, and `σ₁` is not representable in `R` — a full matrix of
        # `floatmax/2` has a `σ₁` near `2·floatmax`. `singular_values` overflows, as its docstring
        # states, while the solve divides by the scaled singular value `σ̂` and returns the step
        # within the check-1 bound.
        n = 4
        m = svd_method(Nothing)
        A = SMatrix{n, n, T}(fill(floatmax(T) / 2, n, n)) +
            SMatrix{n, n, T}(Diagonal(fill(floatmax(T) / 8, n)))
        b = SVector{n, T}(fill(floatmax(T) / 4, n))
        fac = factorize!!(nothing, m, A)
        x = ldiv!!(nothing, fac, b)
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
        # the reported singular value is the one limit: `σ₁` is not representable in `R`, and
        # every entry of `A` is finite
        @test !isfinite(singular_values(fac)[1])
        @test all(isfinite, fac.σ̂)
        @test all(isfinite, x)
        @test relerr(x, reference(A, b)) <= bound(A, T, Nothing)
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
        @test eltype(fac.σ̂) === real(F)
        @test @inferred(ldiv!!(nothing, fac, b)) isa SVector{n, T}
        # under --check-bounds=yes a bounds check lets the local MMatrix escape to the heap,
        # so these are skipped where a run forces it, as `julia-runtest` does by default
        if Base.JLOptions().check_bounds != 1
            @test (@allocated factorize!!(nothing, m, A)) == 0
            @test (@allocated ldiv!!(nothing, fac, b)) == 0
            @test (@allocated collect(b)) > 0
        end
        # an MMatrix Jacobian and a previous factor give the same factors
        @test factorize!!(nothing, m, MMatrix(A)) === fac
        @test factorize!!(fac, m, A) === fac
    end

    @testset "one thread per system in a kernel on $AT gives the host result bit for bit" for AT in ARRAY_BACKENDS
        for (T, TF) in PRECISION_PAIRS, n in (1, 4)

            m = svd_method(TF)
            systems = conditioned_systems(1000n + 3, T, TF, n)
            # the zero matrix and the rank-1 matrix of check 2, whose truncated steps the kernel
            # gives as the host does. `vcat` of two vectors: a `[...; matrix; ...]` literal
            # concatenates a matrix's elements when its size is 1, which the n = 1 case of this
            # loop is
            As = vcat(first.(systems),
                [rank_deficient(100n + 1, T, n, 0), rank_deficient(100n + 5, T, n, 1)])
            bs = vcat(last.(systems), [zero(SVector{n, T}), SVector{n, T}(fill(one(T), n))])
            x, info = kernel_solve(AT, m, As, bs)
            host = map((A, b) -> host_solve(m, A, b), As, bs)
            @test isequal(x, map(last, host))
            @test info == map(h -> first(h).info, host)
            # a rank deficiency is no failure: `info` is 0 for every system. The zero matrix has
            # rank 0 at every size; a rank-1 matrix is full rank at n = 1
            @test all(iszero, info)
            @test first(host[length(systems) + 1]).rank == 0
            @test first(host[length(systems) + 2]).rank == 1
        end
    end

    @testset "JET: factorize!! and ldiv!! are optimisable and error-free" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for (T, TF) in PRECISION_PAIRS, n in (1, 4)

                m = svd_method(TF)
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
