using GeometricSolvers
using GeometricSolvers: factorize!!, ldiv!!, returncode, StaticLU
using JET: JET
using StaticArrays: MMatrix, SMatrix, SVector
using Test

include("../../helpers/matrix.jl")
include("../../helpers/staticsystems.jl")

lu_method(::Type{TF}) where {TF} = LUFactorization{TF}()

@testset "check 1: the LU solve meets 4n·eps(TF)·κ∞ against BigFloat" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = lu_method(TF)
        systems = conditioned_systems(1000n + 1, T, TF, n)
        @test !isempty(systems)
        for (A, b) in systems
            fac, x = host_solve(m, A, b)
            @test returncode(fac) === SUCCESS
            @test eltype(x) === T
            @test relerr(x, reference(A, b)) <= bound(A, T, TF)
        end
        # the scale family s·A, s = 10^k: the bound is scale-free; extreme scales check that a
        # complex multiplier stays finite where an unscaled |p|² overflows or underflows
        A, b = first(systems)
        for s in (exp10.(-8:4:8)..., extreme_scales(real(factortype(T, TF)))...)
            sA = real(T)(s) * A
            fac, x = host_solve(m, sA, b)
            @test returncode(fac) === SUCCESS
            @test relerr(x, reference(sA, b)) <= bound(sA, T, TF)
        end
    end
end

@testset "check 2: pivoting on A[1,1] = 0, A[1,1] = eps(TF)^2 and im·A" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS,
        n in filter(>(1), STATIC_SIZES)

        m = lu_method(TF)
        b = SVector{n, T}(randn(Xoshiro(7n), T, n))
        for A in pivoting_matrices(2000n + 1, T, TF, n)
            fac, x = host_solve(m, A, b)
            @test returncode(fac) === SUCCESS
            @test all(isfinite, x)
            @test relerr(x, reference(A, b)) <= bound(A, T, TF)
        end
    end

    @testset "a tie in |re| + |im| keeps the first index, $T, TF = $TF" for (T, TF) in PRECISION_PAIRS
        m = lu_method(TF)
        for t in (T <: Complex ? (-1, im) : (-1,))
            # step 1 ties 1 with t
            @test factorize!!(nothing, m, SMatrix{2, 2, T}(1, t, 2, 3)).ipiv == [1, 2]
            # step 2 ties the subcolumn [1, t] that the elimination of column 1 leaves
            A = SMatrix{3, 3, T}(2, 1, 1, 0, 1, t, 0, 0, 1)
            @test factorize!!(nothing, m, A).ipiv == [1, 2, 3]
        end
    end
end

@testset "check 3: a singular input gives SINGULAR with info > 0, and no throw" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = lu_method(TF)
        S = singular_matrices(3000n + 1, T, n)
        named = [map(A -> ("zero column", A), S.zero_columns)...,
            ("equal rows", S.equal_rows), ("zero matrix", S.zero_matrix)]
        n == 1 && filter!(p -> p[1] != "equal rows", named)
        for (name, A) in named
            fac = factorize!!(nothing, m, A)
            @test returncode(fac) === SINGULAR
            @test fac.info > 0
            @test all(isfinite, fac.factors)
        end
        # the first zero pivot: a zero first column stops at k = 1, the zero matrix too
        @test factorize!!(nothing, m, S.zero_columns[1]).info == 1
        @test factorize!!(nothing, m, S.zero_matrix).info == 1
    end

    @testset "the equal-rows LU at n = 2 is SINGULAR, $T" for T in ELTYPES
        # Equal rows require a multiplier of exactly one; reciprocal multiplication and complex
        # division can round it differently and leave a nonzero second pivot.
        rng = Xoshiro(42)
        As = map(1:100) do _
            r = randn(rng, T, 2)
            SMatrix{2, 2, T}(r[1], r[1], r[2], r[2])
        end
        @test count(A -> factorize!!(nothing, lu_method(Nothing), A).info == 2, As) == 100
        # the same in a kernel on CPU()
        bs = fill(SVector{2, T}(1, 2), length(As))
        _, info = kernel_solve(Array, lu_method(Nothing), As, bs)
        @test count(==(2), info) == 100
    end

    @testset "two equal rows near floatmax are SINGULAR with finite factors, $T" for T in ELTYPES
        # the complex multiplier's scale max(|re p|, |im p|) stays finite where |re p| + |im p|
        # overflows; at n ≥ 4 the Schur updates overflow here for any multiplier, as in `getrf`
        c = real(T)(0.6) * floatmax(real(T))
        z = T <: Complex ? T(c, c) : T(c)
        fac = factorize!!(nothing, lu_method(Nothing), SMatrix{2, 2, T}(z, z, z, z))
        @test fac.info == 2
        @test returncode(fac) === SINGULAR
        @test all(isfinite, fac.factors)
    end

    @testset "non-finite and subnormal entries: no throw, no finiteness check" for T in ELTYPES
        R = real(T)
        m = lu_method(Nothing)
        A = random_matrix(Xoshiro(5), T, 4, 10.0)
        b = SVector{4, T}(randn(Xoshiro(6), T, 4))
        nonfinite_or_singular(fac, x) = fac.info != 0 || !all(isfinite, x)
        perturbed(v, i, j) = host_solve(m, Base.setindex(A, T(v), i, j), b)
        places = ((1, 1), (3, 2))
        # a NaN gives a non-finite step or info > 0
        @test all(((i, j),) -> nonfinite_or_singular(perturbed(R(NaN), i, j)...), places)
        # a ±Inf gives what the elimination gives, and nothing throws; at (1, 1) it is the pivot,
        # which is not zero, so info == 0: there is no finiteness check
        for v in (R(Inf), -R(Inf)), (i, j) in places

            fac = first(perturbed(v, i, j))
            @test fac isa StaticLU
            (i, j) == (1, 1) && @test fac.info == 0
        end
        # a subnormal non-zero pivot is not singular
        @test factorize!!(nothing, m, SMatrix{1, 1, T}(nextfloat(zero(R)))).info == 0
        @test factorize!!(nothing, m, SMatrix{1, 1, T}(floatmin(R) / 4)).info == 0
    end
end

@testset "check 5: the LU factor is isbits, inferred, allocation-free, Int32 pivots" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = lu_method(TF)
        F = factortype(T, TF)
        A, b = first(conditioned_systems(1000n + 1, T, TF, n))
        fac = @inferred factorize!!(nothing, m, A)
        @test fac isa StaticLU{n, F, n * n}
        @test isbits(fac)
        @test eltype(fac.ipiv) === Int32
        @test eltype(fac.factors) === F
        @test @inferred(ldiv!!(nothing, fac, b)) isa SVector{n, T}
        # under --check-bounds=yes a bounds check lets the local MMatrix escape to the heap,
        # so a run that forces it (Pkg.test() up to Julia 1.12) does not check these
        if Base.JLOptions().check_bounds != 1
            @test (@allocated factorize!!(nothing, m, A)) == 0
            @test (@allocated ldiv!!(nothing, fac, b)) == 0
            @test (@allocated collect(b)) > 0
        end
        # an MMatrix Jacobian and a previous factor give the same factors
        @test factorize!!(nothing, m, MMatrix(A)) === fac
        @test factorize!!(fac, m, A) === fac
    end

    @testset "LUFactorization(T) equals LUFactorization() at TF = T" for T in REAL_ELTYPES
        A = first(first(conditioned_systems(11, T, Nothing, 4)))
        @test factorize!!(nothing, LUFactorization(T), A) ===
              factorize!!(nothing, LUFactorization(), A)
    end

    @testset "one thread per system in a kernel on $AT gives the host result bit for bit" for AT in ARRAY_BACKENDS
        for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

            m = lu_method(TF)
            S = singular_matrices(3000n + 1, T, n)
            systems = conditioned_systems(1000n + 1, T, TF, n)
            singular = n == 1 ? [S.zero_matrix] : [S.equal_rows, S.zero_matrix]
            As = [first.(systems); S.zero_columns; singular]
            bs = [last.(systems); fill(last(first(systems)), length(As) - length(systems))]
            x, info = kernel_solve(AT, m, As, bs)
            host = map((A, b) -> host_solve(m, A, b), As, bs)
            @test isequal(x, map(last, host))
            @test info == map(h -> first(h).info, host)
            @test all(>(0), info[(length(systems) + 1):end])
        end
    end

    @testset "JET: factorize!! and ldiv!! are optimisable and error-free" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for (T, TF) in PRECISION_PAIRS, n in (1, 4)

                m = lu_method(TF)
                F = factortype(T, TF)
                JET.test_opt(factorize!!, (Nothing, typeof(m), SMatrix{n, n, T, n * n}))
                JET.test_call(factorize!!, (Nothing, typeof(m), SMatrix{n, n, T, n * n}))
                JET.test_opt(ldiv!!, (Nothing, StaticLU{n, F, n * n}, SVector{n, T}))
                JET.test_call(ldiv!!, (Nothing, StaticLU{n, F, n * n}, SVector{n, T}))
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
