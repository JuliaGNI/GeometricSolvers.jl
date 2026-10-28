using GeometricSolvers
using GeometricSolvers: DenseLU, DenseQR, DenseSVD,
                        factor_eltype, factorize!!, init, ldiv!!, returncode
using Random: Xoshiro
using Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

const MIXED_ELTYPES = filter(T -> sizeof(real(T)) == 8, ELTYPES)

function mixed_factorize_allocs(fac::F, m::M, J::J_) where {F, M, J_}
    factorize!!(fac, m, J)
    return @allocated factorize!!(fac, m, J)
end

function mixed_ldiv_allocs(d::D, fac::F, r::R) where {D, F, R}
    ldiv!!(d, fac, r)
    return @allocated ldiv!!(d, fac, r)
end

@testset "The T ≠ TF method keeps the scaled conversion (check 4)" begin
    @testset "1e-40 and 1e40 with TF = Float32 ($T, n = $n)" for T in MIXED_ELTYPES,
        n in (1, 4, 16)

        TF = Float32
        x = zeros(T, n)
        for (A_s, b_s) in conditioned_systems(1000 + n + hash(T), T, TF, n)
            A = Matrix(A_s)
            b = Vector(b_s)
            b_unit = b ./ maximum(abs, b)
            bnd = bound(A, T, TF)

            for M in (LUFactorization, QRFactorization, SVDFactorization)
                m = M(TF)
                fac = init(m, nothing, x, nothing)
                copyto!(fac.factors, A)
                fac = factorize!!(fac, m, fac.factors)
                @test returncode(fac) === SUCCESS

                for s in (1e-40, 1e40)
                    r = T(s) .* b_unit
                    r0 = copy(r)
                    d = similar(r)
                    x_sol = @inferred ldiv!!(d, fac, r)
                    @test x_sol === d
                    @test eltype(x_sol) === T
                    @test r == r0
                    @test all(isfinite, x_sol)
                    @test relerr(x_sol, reference(A, r)) <= bnd
                    @test mixed_ldiv_allocs(d, fac, r) == 0
                end
            end
        end
    end

    @testset "a zero right-hand side ($T, n = $n)" for T in MIXED_ELTYPES, n in (1, 4, 16)

        TF = Float32
        x = zeros(T, n)
        A = Matrix(random_matrix(Xoshiro(1100 + n), T, n, 10.0))
        r = zeros(T, n)

        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = M(TF)
            fac = init(m, nothing, x, nothing)
            copyto!(fac.factors, A)
            fac = factorize!!(fac, m, fac.factors)
            fill!(fac.rhs, eltype(fac.rhs)(NaN))
            d = ones(T, n)
            x_sol = ldiv!!(d, fac, r)
            @test x_sol === d
            @test eltype(x_sol) === T
            @test !any(isnan, x_sol)
            @test x_sol == zeros(T, n)
        end
    end

    @testset "a non-finite right-hand side ($T)" for T in MIXED_ELTYPES
        TF = Float32
        n = 4
        R = real(T)
        x = zeros(T, n)
        A = Matrix(random_matrix(Xoshiro(1200 + hash(T)), T, n, 10.0))

        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = M(TF)
            fac = init(m, nothing, x, nothing)
            copyto!(fac.factors, A)
            fac = factorize!!(fac, m, fac.factors)

            for bad in (R(Inf), -R(Inf), R(NaN))
                r = ones(T, n)
                r[2] = T(bad)
                d = similar(r)
                x_sol = ldiv!!(d, fac, r)
                @test !all(isfinite, x_sol)
            end
        end
    end

    # Without the division by `2^e`, J overflows Float32 at c = 1e39 and is subnormal at c = 1e-40.
    @testset "a Jacobian at an extreme scale ($T, c = $c)" for T in MIXED_ELTYPES,
        c in (1e-40, 1e39)

        TF = Float32
        A = T(c) .* T[2 1 0; 1 3 1; 0 1 4]
        b = T[1, -2, 3]
        x = zeros(T, 3)

        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = M(TF)
            fac = factorize!!(init(m, nothing, x, nothing), m, A)
            @test returncode(fac) === SUCCESS
            @test fac.scale_exponent == exponent(4c)
            d = similar(b)
            x_sol = ldiv!!(d, fac, b)
            @test all(isfinite, x_sol)
            @test relerr(x_sol, reference(A, b)) <= bound(A, T, TF)
            @test mixed_factorize_allocs(fac, m, A) == 0
            @test mixed_ldiv_allocs(d, fac, b) == 0
        end
    end
end
