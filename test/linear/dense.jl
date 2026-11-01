using GeometricSolvers
using GeometricSolvers: DenseLU, DenseQR, DenseSVD,
                        factor_eltype, factorize!!, init, ldiv!!, returncode
using LinearAlgebra: BlasFloat, BlasInt, Diagonal, I, norm, opnorm, qr
using Random: Xoshiro
using Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

const DENSE_PRECISION_PAIRS = [(T, TF) for T in ELTYPES for TF in (Nothing, Float32)]

make_method(::Type{M}, ::Type{Nothing}) where {M} = M()
make_method(::Type{M}, ::Type{TF}) where {M, TF} = M(TF)

function dense_factorize_allocs(fac::F, m::M, A::A_) where {F, M, A_}
    copyto!(fac.factors, A)
    fac = factorize!!(fac, m, fac.factors)
    copyto!(fac.factors, A)
    return @allocated factorize!!(fac, m, fac.factors)
end

function dense_ldiv_allocs(d::D, fac::F, r::R) where {D, F, R}
    ldiv!!(d, fac, r)
    return @allocated ldiv!!(d, fac, r)
end

function bigfloat_matrix(rng, n::Int, κ::BigFloat)
    U = Matrix(qr(randn(rng, BigFloat, n, n)).Q)
    V = Matrix(qr(randn(rng, BigFloat, n, n)).Q)
    σ = n == 1 ? [one(BigFloat)] :
        [κ^(-(BigFloat(i - 1) / BigFloat(n - 1))) for i in 1:n]
    return U * Diagonal(σ) * V'
end

function bigfloat_ref_and_cond(A::Matrix{BigFloat}, b::Vector{BigFloat})
    default_prec = precision(BigFloat)
    return setprecision(BigFloat, 2 * default_prec) do
        A2 = BigFloat.(A)
        b2 = BigFloat.(b)
        x_ref = A2 \ b2
        κ = opnorm(A2, Inf) * opnorm(inv(A2), Inf)
        return x_ref, κ
    end
end

@testset "The Dense factor types and init" begin
    @testset "init builds the factor of its method ($T, TF = $TF)" for (T, TF) in DENSE_PRECISION_PAIRS
        n = 4
        x = zeros(T, n)
        lu_fac = init(make_method(LUFactorization, TF), nothing, x, nothing)
        qr_fac = init(make_method(QRFactorization, TF), nothing, x, nothing)
        svd_fac = init(make_method(SVDFactorization, TF), nothing, x, nothing)

        @test lu_fac isa DenseLU
        @test qr_fac isa DenseQR
        @test svd_fac isa DenseSVD
        @test size(lu_fac.factors) == (4, 4)
        @test size(qr_fac.factors) == (4, 4)
        @test size(svd_fac.factors) == (4, 4)
    end

    @testset "the types of every pair ($T, TF = $TF)" for (T, TF) in DENSE_PRECISION_PAIRS
        n = 4
        x = zeros(T, n)
        A_s, b_s = first(conditioned_systems(101 + hash((T, TF)), T, TF, n))
        A = Matrix(A_s)
        b = Vector(b_s)
        expect_no_rhs = TF === Nothing || sizeof(real(T)) == 4

        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = make_method(M, TF)
            F = factor_eltype(m, T)
            fac = init(m, nothing, x, nothing)
            @test eltype(fac.factors) === F
            @test (fac.rhs === nothing) == expect_no_rhs
            if !expect_no_rhs
                @test fac.rhs isa Vector{F}
                @test length(fac.rhs) == n
            end
            if fac isa DenseLU
                @test eltype(fac.ipiv) === BlasInt
            elseif fac isa DenseSVD
                @test length(fac.rwork) == (F <: Complex ? max(1, 5 * n) : n)
            end

            copyto!(fac.factors, A)
            fac = factorize!!(fac, m, fac.factors)
            @test returncode(fac) === SUCCESS
            @test fac.info == 0

            d = similar(b)
            x_sol = ldiv!!(d, fac, b)
            @test eltype(x_sol) === T
            @test x_sol === d
            @test relerr(x_sol, reference(A, b)) <= bound(A, T, TF)
        end
    end

    @testset "a Jacobian that is not the buffer ($T)" for T in ELTYPES
        n = 4
        x = zeros(T, n)
        A_s, b_s = first(conditioned_systems(201 + hash(T), T, Nothing, n))
        A = Matrix(A_s)
        A0 = copy(A)
        b = Vector(b_s)
        b0 = copy(b)

        for m in (LUFactorization(), QRFactorization(), SVDFactorization())
            fac = init(m, nothing, x, nothing)
            fill!(fac.factors, zero(T))
            @test A !== fac.factors
            fac = factorize!!(fac, m, A)
            @test A == A0
            @test !iszero(fac.factors)
            @test returncode(fac) === SUCCESS

            # Two solves from one factorisation leave r and factor buffers intact
            factors_after = copy(fac.factors)
            d1 = similar(b)
            x1 = ldiv!!(d1, fac, b)
            @test b == b0
            @test fac.factors == factors_after
            @test relerr(x1, reference(A, b)) <= bound(A, T, Nothing)

            b2 = reverse(b0)
            d2 = similar(b2)
            x2 = ldiv!!(d2, fac, b2)
            @test relerr(x2, reference(A, b2)) <= bound(A, T, Nothing)

            # In-place solve when d === r
            d_inplace = copy(b0)
            @test ldiv!!(d_inplace, fac, d_inplace) === d_inplace
            @test d_inplace == x1
        end
    end

    @testset "a step vector with a stride ($T)" for T in ELTYPES
        n = 4
        A_s, b_s = first(conditioned_systems(251 + hash(T), T, Nothing, n))
        A = Matrix(A_s)
        b = Vector(b_s)
        m = QRFactorization()
        fac = factorize!!(init(m, nothing, zeros(T, n), nothing), m, A)
        @test_throws ArgumentError ldiv!!(view(zeros(T, n, n), 1, :), fac, b)
    end
end

@testset "No allocation per factorisation and solve (check 1)" begin
    @testset "the allocation of every pair ($T, TF = $TF)" for (T, TF) in DENSE_PRECISION_PAIRS
        n = 8
        x = zeros(T, n)
        rng = Xoshiro(301 + hash((T, TF)))
        A = Matrix(random_matrix(rng, T, n, 10.0))
        b = randn(rng, T, n)
        d = similar(b)

        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = make_method(M, TF)
            fac = init(m, nothing, x, nothing)
            ipiv0 = fac isa DenseLU ? fac.ipiv : nothing

            copyto!(fac.factors, A)
            fac = @inferred factorize!!(fac, m, fac.factors)
            copyto!(fac.factors, A)
            fac = @inferred factorize!!(fac, m, fac.factors)
            if fac isa DenseLU
                @test fac.ipiv === ipiv0
            end

            @test @inferred(ldiv!!(d, fac, b)) === d
            @test dense_factorize_allocs(fac, m, A) == 0
            @test dense_ldiv_allocs(d, fac, b) == 0
        end
    end
end

@testset "The method that runs (check 2)" begin
    @testset "which method" begin
        for T in ELTYPES
            x = zeros(T, 4)
            for (m, D) in ((LUFactorization(), DenseLU), (QRFactorization(), DenseQR),
                (SVDFactorization(), DenseSVD))
                fac = init(m, nothing, x, nothing)
                w = which(factorize!!, Tuple{typeof(fac), typeof(m), typeof(fac.factors)})
                @test w.sig <: Tuple{Any, D{<:StridedMatrix{<:BlasFloat}}, Any, Any}
            end
        end

        xb = zeros(BigFloat, 4)
        for (m, D) in ((LUFactorization(), DenseLU), (QRFactorization(), DenseQR))
            fac = init(m, nothing, xb, nothing)
            w = which(factorize!!, Tuple{typeof(fac), typeof(m), typeof(fac.factors)})
            @test !(w.sig <: Tuple{Any, D{<:StridedMatrix{<:BlasFloat}}, Any, Any})
            @test w.sig <: Tuple{Any, D{<:AbstractMatrix}, Any, Any}
        end
    end

    @testset "no SVD on BigFloat" begin
        x = zeros(BigFloat, 4)
        m = SVDFactorization()
        fac = init(m, nothing, x, nothing)
        copyto!(fac.factors, Matrix{BigFloat}(I, 4, 4))
        @test_throws MethodError factorize!!(fac, m, fac.factors)
    end
end

@testset "A BigFloat system through the generic path (check 2)" begin
    @testset "n = $n" for n in (4, 16)
        rng = Xoshiro(400 + n)
        ε = eps(BigFloat)
        cap = BigFloat(1) / (10 * n * ε)
        κs = exp10.(range(BigFloat(1), log10(cap); length = 4))
        x = zeros(BigFloat, n)

        for κ_target in κs
            A = bigfloat_matrix(rng, n, κ_target)
            b = randn(rng, BigFloat, n)
            x_ref, κ = bigfloat_ref_and_cond(A, b)
            for m in (LUFactorization(), QRFactorization())
                fac = init(m, nothing, x, nothing)
                copyto!(fac.factors, A)
                fac = factorize!!(fac, m, fac.factors)
                @test returncode(fac) === SUCCESS
                d = similar(b)
                x_sol = ldiv!!(d, fac, b)
                err = maximum(abs, x_sol .- x_ref) / maximum(abs, x_ref)
                @test err <= 4 * n * ε * κ
            end
        end
    end
end

@testset "A singular matrix does not throw (check 3)" begin
    @testset "LU on a zero column and the zero matrix ($T)" for T in (ELTYPES..., BigFloat)
        n = 4
        rng = Xoshiro(501 + hash(T))
        A_col2 = randn(rng, T, n, n)
        A_col2[:, 2] .= zero(T)
        A_zero = zeros(T, n, n)
        x = zeros(T, n)
        m = LUFactorization()

        for A_sing in (A_col2, A_zero)
            fac = init(m, nothing, x, nothing)
            copyto!(fac.factors, A_sing)
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info > 0
            @test returncode(fac) === SINGULAR
        end
    end

    @testset "QR on a zero column and the zero matrix ($T)" for T in (ELTYPES..., BigFloat)
        n = 4
        rng = Xoshiro(551 + hash(T))
        A_col2 = randn(rng, T, n, n)
        A_col2[:, 2] .= zero(T)
        A_zero = zeros(T, n, n)
        x = zeros(T, n)
        m = QRFactorization()

        fac = init(m, nothing, x, nothing)
        copyto!(fac.factors, A_col2)
        fac = factorize!!(fac, m, fac.factors)
        @test fac.info == 2
        @test returncode(fac) === SINGULAR
        if T <: BlasFloat
            # The exact zero on the diagonal of `R` gives a non-finite step, as in LU.
            @test !all(isfinite, ldiv!!(zeros(T, n), fac, ones(T, n)))
        end

        copyto!(fac.factors, A_zero)
        fac_z = factorize!!(fac, m, fac.factors)
        @test fac_z.info == 1
        @test returncode(fac_z) === SINGULAR
        if T <: BlasFloat
            @test !all(isfinite, ldiv!!(zeros(T, n), fac_z, ones(T, n)))
        end
    end

    @testset "two equal rows ($T)" for T in ELTYPES
        x = zeros(T, 2)
        for m in (LUFactorization(), QRFactorization())
            rng = Xoshiro(42)
            fac = init(m, nothing, x, nothing)
            for _ in 1:100
                A = randn(rng, T, 2, 2)
                A[2, :] .= A[1, :]
                copyto!(fac.factors, A)
                fac = factorize!!(fac, m, fac.factors)
                @test all(isfinite, fac.factors)
            end
        end
    end
end

@testset "The size band and conditioned systems (check 5)" begin
    @testset "n = 1 ($T, TF = $TF)" for (T, TF) in DENSE_PRECISION_PAIRS
        rng = Xoshiro(601 + hash((T, TF)))
        a = randn(rng, T)
        iszero(a) && (a = one(T))
        A = fill(a, 1, 1)
        b = [randn(rng, T)]
        iszero(b[1]) && (b[1] = one(T))
        x = zeros(T, 1)
        @test κinf(A) ≈ 1

        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = make_method(M, TF)
            fac = init(m, nothing, x, nothing)
            copyto!(fac.factors, A)
            fac = factorize!!(fac, m, fac.factors)
            @test returncode(fac) === SUCCESS
            d = similar(b)
            x_sol = ldiv!!(d, fac, b)
            @test relerr(x_sol, reference(A, b)) <= 4 * epsf(T, TF) * κinf(A)
        end
    end

    @testset "conditioned systems n = $n ($T, TF = $TF)" for n in (4, 16),
        (T, TF) in DENSE_PRECISION_PAIRS

        x = zeros(T, n)
        for (A_s, b_s) in conditioned_systems(700 + n + hash((T, TF)), T, TF, n)
            A = Matrix(A_s)
            b = Vector(b_s)
            xref = reference(A, b)
            bnd = bound(A, T, TF)
            for M in (LUFactorization, QRFactorization, SVDFactorization)
                m = make_method(M, TF)
                fac = init(m, nothing, x, nothing)
                copyto!(fac.factors, A)
                fac = factorize!!(fac, m, fac.factors)
                @test returncode(fac) === SUCCESS
                d = similar(b)
                x_sol = ldiv!!(d, fac, b)
                @test relerr(x_sol, xref) <= bnd
            end
        end
    end

    @testset "scaled-column same-precision ldiv!! ($T)" for T in ELTYPES
        RT = real(T)
        s = RT === Float32 ? 1.0f20 : 1.0e200
        A_sc = T[T(s) one(T); T(2 * s) T(3)]
        x_expected = T[one(T) / T(s), T(2)]
        b_sc = A_sc * x_expected
        x = zeros(T, 2)
        for m in (LUFactorization(), QRFactorization())
            fac = init(m, nothing, x, nothing)
            fac = factorize!!(fac, m, A_sc)
            @test returncode(fac) === SUCCESS
            d = ldiv!!(similar(b_sc), fac, b_sc)
            @test isapprox(d, x_expected; rtol = 16 * eps(RT))
        end
    end
end

@testset "The rank-revealing SVD solve (check 5)" begin
    @testset "a consistent rank-deficient system ($T, TF = $TF)" for (T, TF) in DENSE_PRECISION_PAIRS
        n = 16
        r = 5
        m = make_method(SVDFactorization, TF)
        F = factor_eltype(m, T)
        R = real(F)
        ε = eps(R)
        x = zeros(T, n)

        # Standard random rank-5 product B * C' and an equal-row rank-5 product whose
        # smallest positive singular value is near subnormal so a 0 cutoff overflows A * x.
        for variant in (1, 2)
            rng = Xoshiro(42)
            B = R(10) .* randn(rng, T, n, r)
            C = R(10) .* randn(rng, T, n, r)
            if variant == 2
                for i in (r + 1):n
                    B[i, :] .= B[1, :]
                end
                B[r + 1, 1] += T(floatmin(R))
            end
            A = B * C'
            x0 = randn(rng, T, n)
            b = A * x0

            fac = init(m, nothing, x, nothing)
            copyto!(fac.factors, A)
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            @test fac.rank == r
            @test returncode(fac) === SUCCESS

            d = similar(b)
            x_sol = ldiv!!(d, fac, b)
            @test !any(isnan, x_sol)
            @test norm(A * x_sol - b) <= 4 * n * ε * opnorm(A, 2) * norm(x_sol)
            @test norm(x_sol) <= (1 + 4 * n * ε) * norm(x0)
        end
    end

    @testset "the zero matrix ($T)" for T in ELTYPES
        A = zeros(T, 4, 4)
        b = ones(T, 4)
        m = SVDFactorization()
        fac = init(m, nothing, zeros(T, 4), nothing)
        copyto!(fac.factors, A)
        fac = factorize!!(fac, m, fac.factors)
        @test fac.info == 0
        @test fac.rank == 0
        @test returncode(fac) === SUCCESS
        d = similar(b)
        x_sol = ldiv!!(d, fac, b)
        @test !any(isnan, x_sol)
        @test x_sol == zeros(T, 4)
    end

    @testset "n = 0 ($T)" for T in ELTYPES
        A = zeros(T, 0, 0)
        b = zeros(T, 0)
        m = SVDFactorization()
        fac = init(m, nothing, zeros(T, 0), nothing)
        fac = factorize!!(fac, m, A)
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
        d = similar(b)
        x_sol = ldiv!!(d, fac, b)
        @test x_sol === d
        @test isempty(x_sol)
    end
end

@testset "No finiteness check on A" begin
    @testset "NaN and ±Inf in A ($T)" for T in ELTYPES
        R = real(T)
        n = 4
        rng = Xoshiro(901 + hash(T))
        A_base = Matrix(random_matrix(rng, T, n, 10.0))
        b = ones(T, n)
        x = zeros(T, n)

        for m in (LUFactorization(), QRFactorization())
            A_nan = copy(A_base)
            A_nan[3, 2] = T(R(NaN))
            fac_nan = init(m, nothing, x, nothing)
            copyto!(fac_nan.factors, A_nan)
            fac_nan = factorize!!(fac_nan, m, fac_nan.factors)
            d_nan = ldiv!!(similar(b), fac_nan, b)
            @test !all(isfinite, d_nan) || fac_nan.info > 0

            for inf_val in (R(Inf), -R(Inf))
                A_inf = copy(A_base)
                A_inf[1, 1] = T(inf_val)
                fac_inf = init(m, nothing, x, nothing)
                copyto!(fac_inf.factors, A_inf)
                fac_inf = factorize!!(fac_inf, m, fac_inf.factors)
                # An `Inf` in `J` does not throw; the values of the step are not specified.
                @test (ldiv!!(similar(b), fac_inf, b); true)
                if m isa LUFactorization
                    @test fac_inf.info == 0
                end
            end
        end
    end
end

@testset "the same-type solve scales r by a power of two, $T" for T in REAL_ELTYPES
    @testset "$label, $M" for (label, J, b, xref, with_qr) in extreme_rhs_systems(T),
        M in (SVDFactorization, QRFactorization, LUFactorization)

        M === QRFactorization && !with_qr && continue
        M === LUFactorization && !with_qr && continue
        m = M()
        fac = factorize!!(init(m, nothing, zeros(T, 4), nothing), m, J)
        d = ldiv!!(similar(b), fac, b)
        @test all(isfinite, d)
        @test relerr(d, xref) <= 100 * eps(T)
        # in place, r === d
        r = copy(b)
        @test ldiv!!(r, fac, r) === r
        @test r == d
    end

    # a solution above floatmax overflows, by design: 2^-130 I has the solution 2^130
    @testset "a solution above floatmax overflows, $M" for M in (LUFactorization,
        QRFactorization, SVDFactorization)
        m = M()
        J = ldexp(one(T), -(exponent(floatmax(T)) + 3)) * Matrix{T}(I, 4, 4)
        fac = factorize!!(init(m, nothing, zeros(T, 4), nothing), m, J)
        @test !any(isfinite, ldiv!!(zeros(T, 4), fac, ones(T, 4)))
    end

    @testset "a zero, non-finite or empty r, $M" for M in (QRFactorization, SVDFactorization)
        m = M()
        fac = factorize!!(init(m, nothing, zeros(T, 4), nothing), m, 2 * Matrix{T}(I, 4, 4))
        @test ldiv!!(ones(T, 4), fac, zeros(T, 4)) == zeros(T, 4)
        for v in (T(Inf), T(NaN))
            @test !all(isfinite, ldiv!!(zeros(T, 4), fac, T[v, 1, 1, 1]))
        end
        fac0 = factorize!!(init(m, nothing, zeros(T, 0), nothing), m, zeros(T, 0, 0))
        @test isempty(ldiv!!(zeros(T, 0), fac0, zeros(T, 0)))
    end
end
