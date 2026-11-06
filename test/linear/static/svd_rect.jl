using GeometricSolvers
using GeometricSolvers: exact_solve, factor_diagnostics, factorize!!, ldiv!!, returncode,
                        singular_values, StaticRectSVD, StaticSVD
using JET: JET
using LinearAlgebra: Diagonal, I, norm, pinv
using Random: Xoshiro, randn
using StaticArrays: MMatrix, SMatrix, SVector
using Test

include("../../helpers/matrix.jl")

const RECT_SIZES = ((4, 2), (6, 3), (10, 4), (32, 8))

function rect_svd_system(seed::Int, ::Type{T}, ::Val{M}, ::Val{N}) where {T, M, N}
    rng = Xoshiro(seed)
    A = SMatrix{M, N, T, M * N}(randn(rng, T, M * N)) +
        SMatrix{M, N, T, M * N}([i == j ? T(2) : zero(T) for i in 1:M, j in 1:N])
    b = SVector{M, T}(randn(rng, T, M))
    return A, b
end

@testset "rectangular one-sided Jacobi SVD reconstruction and solve, $T, ($M, $N)" for T in REAL_ELTYPES,
    (M, N) in RECT_SIZES

    A, b = rect_svd_system(300 * M + N, T, Val(M), Val(N))
    fac = factorize!!(SVDFactorization(), nothing, A)
    @test fac isa StaticRectSVD{M, N, T, M * N, N * N, T}
    @test factorize!!(nothing, SVDFactorization(), A) === fac
    @test factorize!!(fac, SVDFactorization(), A) === fac
    @test factorize!!(nothing, SVDFactorization(), MMatrix(A)) === fac
    @test returncode(fac) === SUCCESS
    @test exact_solve(fac) === true
    @test fac.info == 0
    @test fac.rank == N
    @test singular_values(fac) === fac.S
    @test issorted(fac.S; rev = true)
    @test isapprox(fac.U * Diagonal(fac.S) * fac.V', A; rtol = 64 * sqrt(T(M)) * eps(T))
    @test isapprox(fac.U' * fac.U, SMatrix{N, N, T}(I); atol = 64 * sqrt(T(M)) * eps(T))
    @test isapprox(fac.V' * fac.V, SMatrix{N, N, T}(I); atol = 64 * sqrt(T(N)) * eps(T))

    d = zero(SVector{N, T})
    x = ldiv!!(d, fac, b)
    @test x isa SVector{N, T}
    @test x != d
    x_ref = Matrix(A) \ Vector(b)
    @test isapprox(Vector(x), x_ref; rtol = 16 * sqrt(eps(T)))

    # subnormal prescale on 4×2
    if (M, N) == (4, 2)
        E = exponent(floatmin(T)) - 8
        B = SMatrix{4, 2, T}([T(i + j) for i in 1:4, j in 1:2])
        sB = ldexp(one(T), E) * B
        fac_sub = factorize!!(SVDFactorization(), nothing, sB)
        ref_sub = factorize!!(SVDFactorization(), nothing, B)
        @test all(isfinite, fac_sub.U)
        @test all(isfinite, fac_sub.V)
        @test singular_values(fac_sub) == map(x -> ldexp(x, E), singular_values(ref_sub))
    end
end

@testset "rank-deficient minimum-norm pseudoinverse solve and diagnostics, $T" for T in REAL_ELTYPES
    # 6×3 matrix of rank 2: column 3 = column 1 + column 2
    c1 = SVector{6, T}(1, 2, -1, 3, 0, 2)
    c2 = SVector{6, T}(-2, 1, 3, 0, 2, -1)
    c3 = c1 + c2
    A = hcat(c1, c2, c3)
    b = SVector{6, T}(1, -1, 2, 3, -2, 4)

    fac = factorize!!(SVDFactorization(), nothing, A)
    @test returncode(fac) === SUCCESS
    @test fac.info == 0
    @test fac.rank == Int32(2)
    @test fac.S[3] <= sqrt(T(3)) * eps(T) * fac.S[1]
    @test all(iszero, fac.U[:, 3])

    d = zero(SVector{3, T})
    x = ldiv!!(d, fac, b)
    x_pinv = pinv(Matrix(A)) * Vector(b)
    @test isapprox(Vector(x), x_pinv; rtol = 32 * sqrt(eps(T)))
    # normal equations hold and x is orthogonal to null vector (1, 1, -1)
    @test norm(A' * (A * x - b)) <= 64 * eps(T) * norm(A) * norm(b)
    v_null = SVector{3, T}(1, 1, -1)
    @test abs(v_null' * x) <= 64 * eps(T) * norm(x)
    @test norm(x) < norm(x + T(0.1) * v_null)

    # factor_diagnostics on rank-deficient 6×3
    rk, nn = factor_diagnostics(fac, b)
    @test rk == Int32(2)
    U_r = Matrix(fac.U)[:, 1:2]
    nn_ref = norm(Vector(b) - U_r * (U_r' * Vector(b))) / norm(b)
    @test isapprox(nn, T(nn_ref); rtol = 32 * eps(T))

    # zero matrix (rank 0) returns zero step and nullnorm 1 for nonzero rhs
    A_zero = zero(SMatrix{4, 2, T})
    b_nz = SVector{4, T}(1, 2, 3, 4)
    fac_zero = factorize!!(SVDFactorization(), nothing, A_zero)
    @test fac_zero.rank == Int32(0)
    @test ldiv!!(zero(SVector{2, T}), fac_zero, b_nz) == zero(SVector{2, T})
    rk_z, nn_z = factor_diagnostics(fac_zero, b_nz)
    @test rk_z == Int32(0)
    @test isapprox(nn_z, one(T); rtol = 8 * eps(T))
end

@testset "square dispatch, dimension guards, and size limit, $T" for T in REAL_ELTYPES
    A_sq = SMatrix{3, 3, T}(I)
    fac_sq = factorize!!(SVDFactorization(), nothing, A_sq)
    @test fac_sq isa StaticSVD{3, T, T, 9}

    A_under = SMatrix{2, 4, T}(ntuple(i -> T(i), 8))
    @test_throws DimensionMismatch factorize!!(SVDFactorization(), nothing, A_under)
    @test_throws DimensionMismatch factorize!!(nothing, SVDFactorization(), A_under)
    A_empty = SMatrix{3, 0, T}()
    @test_throws DimensionMismatch factorize!!(SVDFactorization(), nothing, A_empty)

    A_big = SMatrix{33, 2, T}(ntuple(i -> T(i), 66))
    @test_throws ArgumentError factorize!!(SVDFactorization(), nothing, A_big)
end

@testset "StaticRectSVD is isbits, inferred, allocation-free, and JET-clean" begin
    @testset "$T, ($M, $N)" for T in REAL_ELTYPES, (M, N) in RECT_SIZES
        A, b = rect_svd_system(400 * M + N, T, Val(M), Val(N))
        m = SVDFactorization()
        d = zero(SVector{N, T})
        fac = @inferred factorize!!(m, nothing, A)
        @test isbits(fac)
        @test (@inferred ldiv!!(d, fac, b)) isa SVector{N, T}
        @test (@inferred factor_diagnostics(fac, b)) isa Tuple{Int32, T}
        # skipped where a run forces --check-bounds=yes
        if Base.JLOptions().check_bounds != 1
            alloc_fac = let m = m, A = A
                factorize!!(m, nothing, A)
                @allocated factorize!!(m, nothing, A)
            end
            alloc_ldiv = let d = d, fac = fac, b = b
                ldiv!!(d, fac, b)
                @allocated ldiv!!(d, fac, b)
            end
            @test alloc_fac == 0
            @test alloc_ldiv == 0
        end
    end

    @testset "JET: factorize!! and ldiv!! are optimisable and error-free" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for T in REAL_ELTYPES
                m = SVDFactorization()
                JET.test_opt(factorize!!, (typeof(m), Nothing, SMatrix{4, 2, T, 8}))
                JET.test_call(factorize!!, (typeof(m), Nothing, SMatrix{4, 2, T, 8}))
                JET.test_opt(ldiv!!, (SVector{2, T}, StaticRectSVD{4, 2, T, 8, 4, T}, SVector{4, T}))
                JET.test_call(ldiv!!, (SVector{2, T}, StaticRectSVD{4, 2, T, 8, 4, T}, SVector{4, T}))
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
