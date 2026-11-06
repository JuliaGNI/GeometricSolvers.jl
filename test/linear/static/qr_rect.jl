using GeometricSolvers
using GeometricSolvers: backsolve, backsolve_shifted, exact_solve, factor_diagnostics,
                        factorize!!, ldiv!!, returncode, StaticQR, StaticRectQR
using JET: JET
using LinearAlgebra: I, norm, qr
using Random: Xoshiro, randn
using StaticArrays: MMatrix, SMatrix, SVector
using Test

include("../../helpers/matrix.jl")

const RECT_SIZES = ((4, 2), (6, 3), (10, 4), (32, 8))

function rect_qr_system(seed::Int, ::Type{T}, ::Val{M}, ::Val{N}) where {T, M, N}
    rng = Xoshiro(seed)
    A = SMatrix{M, N, T, M * N}(randn(rng, T, M * N)) +
        SMatrix{M, N, T, M * N}([i == j ? T(2) : zero(T) for i in 1:M, j in 1:N])
    b = SVector{M, T}(randn(rng, T, M))
    return A, b
end

@testset "rectangular Householder QR solve matches normal equations, $T, ($M, $N)" for T in REAL_ELTYPES,
    (M, N) in RECT_SIZES

    A, b = rect_qr_system(100 * M + N, T, Val(M), Val(N))
    fac = factorize!!(QRFactorization(), nothing, A)
    @test fac isa StaticRectQR{M, N, T, M * N}
    @test factorize!!(nothing, QRFactorization(), A) === fac
    @test factorize!!(fac, QRFactorization(), A) === fac
    @test factorize!!(nothing, QRFactorization(), MMatrix(A)) === fac
    @test returncode(fac) === SUCCESS
    @test exact_solve(fac) === true
    @test fac.info == 0

    d = zero(SVector{N, T})
    x = ldiv!!(d, fac, b)
    @test x isa SVector{N, T}
    @test x != d
    x_ref = (Matrix(A)' * Matrix(A)) \ (Matrix(A)' * Vector(b))
    @test isapprox(Vector(x), x_ref; rtol = 8 * sqrt(eps(T)))
    # normal equations residual J' * (J * x - b) ≈ 0
    r = A * x - b
    @test norm(A' * r) <= 32 * eps(T) * norm(A) * norm(b)

    # scale family: subnormal and large scales preserve the least-squares solution
    for s in (floatmin(T), T(1e-6), T(1e6))
        sA = s * A
        sb = s * b
        sfac = factorize!!(QRFactorization(), nothing, sA)
        @test returncode(sfac) === SUCCESS
        sx = ldiv!!(d, sfac, sb)
        @test isapprox(Vector(sx), x_ref; rtol = 16 * sqrt(eps(T)))
    end
end

@testset "factor_diagnostics: full-rank and rank-deficient nullnorm over 1:M, $T" for T in REAL_ELTYPES
    # full-rank 5×2 system: rank == 2 and nullnorm == 0
    A_full, b_full = rect_qr_system(502, T, Val(5), Val(2))
    fac_full = factorize!!(QRFactorization(), nothing, A_full)
    rk_full, nn_full = factor_diagnostics(fac_full, b_full)
    @test rk_full == Int32(2)
    @test nn_full === zero(T)

    # rank-1 4×2 system where column 2 is zero and b has energy in rows 1:2 and 3:4:
    # nullnorm must accumulate over all M rows (including rows N+1:M), not only 1:N
    A_def = SMatrix{4, 2, T}(T(3), T(4), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T))
    b_def = SVector{4, T}(T(4), T(-3), T(5), T(12))
    fac_def = factorize!!(QRFactorization(), nothing, A_def)
    @test returncode(fac_def) === SINGULAR
    @test fac_def.info == Int32(2)
    rk_def, nn_def = factor_diagnostics(fac_def, b_def)
    @test rk_def == Int32(1)
    # column 1 is (3, 4, 0, 0), orthogonal to b_def = (4, -3, 5, 12), so the entire b_def
    # lies in the orthogonal complement of range(A_def) and nullnorm == 1
    @test isapprox(nn_def, one(T); rtol = 16 * eps(T))

    # partially aligned rhs: b_part has projection of norm 5 on col 1 and orthogonal norm 13
    # in rows 3:4, so nullnorm == 13 / sqrt(25 + 169)
    b_part = SVector{4, T}(T(3), T(4), T(5), T(12))
    rk_part, nn_part = factor_diagnostics(fac_def, b_part)
    @test rk_part == Int32(1)
    @test isapprox(nn_part, T(13) / sqrt(T(194)); rtol = 16 * eps(T))

    # zero rhs gives zero nullnorm
    _, nn_zero = factor_diagnostics(fac_def, zero(SVector{4, T}))
    @test nn_zero === zero(T)
end

@testset "square dispatch, dimension guards, and size limit, $T" for T in REAL_ELTYPES
    # M == N dispatches to square StaticQR
    A_sq = SMatrix{3, 3, T}(I)
    fac_sq = factorize!!(QRFactorization(), nothing, A_sq)
    @test fac_sq isa StaticQR{3, T, 9}

    # underdetermined M < N and empty N == 0 throw DimensionMismatch
    A_under = SMatrix{2, 4, T}(ntuple(i -> T(i), 8))
    @test_throws DimensionMismatch factorize!!(QRFactorization(), nothing, A_under)
    @test_throws DimensionMismatch factorize!!(nothing, QRFactorization(), A_under)
    A_empty = SMatrix{3, 0, T}()
    @test_throws DimensionMismatch factorize!!(QRFactorization(), nothing, A_empty)

    # M > 32 throws ArgumentError
    A_big = SMatrix{33, 2, T}(ntuple(i -> T(i), 66))
    @test_throws ArgumentError factorize!!(QRFactorization(), nothing, A_big)
end

@testset "StaticRectQR is isbits, inferred, allocation-free, and JET-clean" begin
    @testset "$T, ($M, $N)" for T in REAL_ELTYPES, (M, N) in RECT_SIZES

        A, b = rect_qr_system(200 * M + N, T, Val(M), Val(N))
        m = QRFactorization()
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
                m = QRFactorization()
                JET.test_opt(factorize!!, (typeof(m), Nothing, SMatrix{4, 2, T, 8}))
                JET.test_call(factorize!!, (typeof(m), Nothing, SMatrix{4, 2, T, 8}))
                JET.test_opt(ldiv!!, (
                    SVector{2, T}, StaticRectQR{4, 2, T, 8}, SVector{4, T}))
                JET.test_call(ldiv!!, (
                    SVector{2, T}, StaticRectQR{4, 2, T, 8}, SVector{4, T}))
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
