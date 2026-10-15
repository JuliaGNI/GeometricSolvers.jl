using GeometricSolvers
using GeometricSolvers: factorize!!, ldiv!!, returncode, StaticQR
using JET: JET
using StaticArrays: MMatrix, SMatrix, SVector
using Test

include("../../helpers/matrix.jl")
include("../../helpers/staticsystems.jl")

qr_method(::Type{TF}) where {TF} = QRFactorization{TF}()

@testset "check 1: the QR solve meets 4n·eps(TF)·κ∞ against BigFloat" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = qr_method(TF)
        systems = conditioned_systems(1000n + 2, T, TF, n)
        @test !isempty(systems)
        for (A, b) in systems
            fac, x = host_solve(m, A, b)
            @test returncode(fac) === SUCCESS
            @test eltype(x) === T
            @test relerr(x, reference(A, b)) <= bound(A, T, TF)
        end
        # the scale family s·A, s = 10^k: the bound is scale-free
        A, b = first(systems)
        for k in -8:4:8
            sA = real(T)(10.0^k) * A
            fac, x = host_solve(m, sA, b)
            @test returncode(fac) === SUCCESS
            @test relerr(x, reference(sA, b)) <= bound(sA, T, TF)
        end
    end
end

@testset "check 2: the reflector on A[1,1] = 0, A[1,1] = eps(TF)^2 and im·A" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS,
        n in filter(>(1), STATIC_SIZES)

        m = qr_method(TF)
        b = SVector{n, T}(randn(Xoshiro(7n), T, n))
        for A in pivoting_matrices(2000n + 2, T, TF, n)
            fac, x = host_solve(m, A, b)
            @test returncode(fac) === SUCCESS
            @test all(isfinite, x)
            @test relerr(x, reference(A, b)) <= bound(A, T, TF)
        end
    end
end

@testset "check 3: a singular input gives no throw, and an exact zero of R gives SINGULAR" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = qr_method(TF)
        S = singular_matrices(3000n + 2, T, n)
        # a zero column and the zero matrix give an exact zero diagonal entry of R
        for A in (S.zero_columns..., S.zero_matrix)
            fac = factorize!!(nothing, m, A)
            @test returncode(fac) === SINGULAR
            @test fac.info > 0
            @test all(isfinite, fac.factors)
            @test all(isfinite, fac.τ)
        end
        # the first zero diagonal entry: a zero column c gives info == c
        @test [factorize!!(nothing, m, A).info for A in S.zero_columns] == 1:n
        @test factorize!!(nothing, m, S.zero_matrix).info == 1
        # equal rows need not give an exact zero in QR: no throw and finite factors
        if n > 1
            fac = factorize!!(nothing, m, S.equal_rows)
            @test all(isfinite, fac.factors)
            @test all(isfinite, fac.τ)
        end
    end

    @testset "non-finite and subnormal entries: no throw, never a finite step with info 0" for T in ELTYPES
        R = real(T)
        m = qr_method(Nothing)
        A = random_matrix(Xoshiro(5), T, 4, 10.0)
        b = SVector{4, T}(randn(Xoshiro(6), T, 4))
        for v in (R(NaN), R(Inf), -R(Inf)), (i, j) in ((1, 1), (3, 2))

            fac, x = host_solve(m, Base.setindex(A, T(v), i, j), b)
            @test fac.info != 0 || !all(isfinite, x)
        end
        # a subnormal non-zero diagonal is not singular
        @test factorize!!(nothing, m, SMatrix{1, 1, T}(nextfloat(zero(R)))).info == 0
        @test factorize!!(nothing, m, SMatrix{1, 1, T}(floatmin(R) / 4)).info == 0
    end
end

@testset "check 5: the QR factor is isbits, inferred and allocation-free" begin
    @testset "$T, TF = $TF, n = $n" for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

        m = qr_method(TF)
        F = factortype(T, TF)
        A, b = first(conditioned_systems(1000n + 2, T, TF, n))
        fac = @inferred factorize!!(nothing, m, A)
        @test fac isa StaticQR{n, F, n * n}
        @test isbits(fac)
        @test eltype(fac.factors) === F
        @test @inferred(ldiv!!(nothing, fac, b)) isa SVector{n, T}
        # under --check-bounds=yes a bounds check lets the local MMatrix escape to the heap,
        # so a run that forces it (Pkg.test() up to Julia 1.12) does not check these
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
        for (T, TF) in PRECISION_PAIRS, n in STATIC_SIZES

            m = qr_method(TF)
            S = singular_matrices(3000n + 2, T, n)
            systems = conditioned_systems(1000n + 2, T, TF, n)
            singular = [S.zero_columns; [S.zero_matrix]]
            # equal rows give no exact zero in QR: no throw, and the host's step and info
            equal = n > 1 ? [S.equal_rows] : empty(singular)
            As = [first.(systems); singular; equal]
            bs = [last.(systems); fill(last(first(systems)), length(As) - length(systems))]
            x, info = kernel_solve(AT, m, As, bs)
            host = map((A, b) -> host_solve(m, A, b), As, bs)
            @test isequal(x, map(last, host))
            @test info == map(h -> first(h).info, host)
            @test all(>(0), info[length(systems) .+ eachindex(singular)])
        end
    end

    @testset "JET: factorize!! and ldiv!! are optimisable and error-free" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for (T, TF) in PRECISION_PAIRS, n in (1, 4)

                m = qr_method(TF)
                F = factortype(T, TF)
                JET.test_opt(factorize!!, (Nothing, typeof(m), SMatrix{n, n, T, n * n}))
                JET.test_call(factorize!!, (Nothing, typeof(m), SMatrix{n, n, T, n * n}))
                JET.test_opt(ldiv!!, (Nothing, StaticQR{n, F, n * n}, SVector{n, T}))
                JET.test_call(ldiv!!, (Nothing, StaticQR{n, F, n * n}, SVector{n, T}))
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
