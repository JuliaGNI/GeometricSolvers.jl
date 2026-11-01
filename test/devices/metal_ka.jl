using GPUArraysCore: allowscalar
using LinearAlgebra: BLAS, Diagonal, I, cond, norm, opnorm, qr
using Metal
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization, SVDFactorization
using GeometricSolvers: backsolve!, factorize!!, init, jacobi_tolerance, ldiv!!, returncode

include("../helpers/matrix.jl")
include("../helpers/metal_device.jl")

if Metal.functional() && !on_virtual_gpu()
    allowscalar(false)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    const METAL_ELTYPES = filter(T -> real(T) === Float32, ELTYPES)

    function metal_well_conditioned(T::Type, n::Int, seed::Integer = 0x5eed)
        rng = Xoshiro(seed)
        if n <= 256
            A = randn(rng, T, n, n) + T(2 * n) * I
        else
            A = Matrix{T}(Diagonal(T(2 * n) .+ randn(rng, T, n)))
            A[1:16, 1:16] .+= randn(rng, T, 16, 16)
            A[(n - 15):n, (n - 15):n] .+= randn(rng, T, 16, 16)
        end
        b = randn(rng, T, n)
        return A, b
    end

    function metal_rank_deficient(seed::Integer, ::Type{T}, n::Int, r::Int) where {T}
        rng = Xoshiro(seed)
        U = Matrix{T}(qr(randn(rng, T, n, n)).Q)
        V = Matrix{T}(qr(randn(rng, T, n, n)).Q)
        Σ = real(T)[i <= r ? real(T)(0.5)^(i - 1) : zero(real(T)) for i in 1:n]
        return U * Diagonal(Σ) * V'
    end

    metal_wide_eltype(::Type{T}) where {T <: Real} = Float64
    metal_wide_eltype(::Type{T}) where {T <: Complex} = ComplexF64
    metal_noop(r, y, p) = nothing

    @testset "Metal KA LU and QR agree with CPU solves at n = $n" for n in (1, 16, 256)
        @testset "T = $T" for T in METAL_ELTYPES
            for M in (LUFactorization, QRFactorization)
                m = M()
                A_cpu, b_cpu = metal_well_conditioned(T, n)
                κ = Float64(cond(A_cpu, Inf))
                @test κ <= 10.0

                x_cpu_init = zeros(T, n)
                prob_cpu = NonlinearProblem(x_cpu_init, metal_noop)
                fac_cpu = init(m, prob_cpu, x_cpu_init, nothing)
                copyto!(fac_cpu.factors, A_cpu)
                fac_cpu = factorize!!(fac_cpu, m, fac_cpu.factors)
                @test fac_cpu.info == 0
                d_cpu = ldiv!!(similar(x_cpu_init), fac_cpu, copy(b_cpu))

                x_dev = MtlArray(zeros(T, n))
                b_dev = MtlArray(b_cpu)
                prob_dev = NonlinearProblem(x_dev, metal_noop)
                fac_dev = init(m, prob_dev, x_dev, nothing)
                copyto!(fac_dev.factors, MtlArray(A_cpu))
                fac_dev = factorize!!(fac_dev, m, fac_dev.factors)
                @test fac_dev.info == 0
                @test returncode(fac_dev) === SUCCESS
                d_dev = ldiv!!(similar(x_dev), fac_dev, b_dev)
                @test d_dev isa MtlVector{T}

                x_h = Array(d_dev)
                eps32 = Float64(eps(Float32))
                bwd = norm(A_cpu * x_h - b_cpu, Inf) / (opnorm(A_cpu, Inf) * norm(x_h, Inf))
                @test bwd <= 4 * n * eps32

                fwd = norm(x_h - d_cpu, Inf) / norm(d_cpu, Inf)
                @test fwd <= 8 * n * eps32 * κ
            end
        end
    end

    @testset "Metal KA SVD agrees with CPU LAPACK gesvd at n = $n" for n in (1, 16, 10^3)
        @testset "T = $T" for T in METAL_ELTYPES
            m = SVDFactorization()
            A_cpu, b_cpu = metal_well_conditioned(T, n)
            κ = Float64(cond(A_cpu, Inf))
            @test κ <= 10.0

            x_cpu_init = zeros(T, n)
            prob_cpu = NonlinearProblem(x_cpu_init, metal_noop)
            fac_cpu = init(m, prob_cpu, x_cpu_init, nothing)
            copyto!(fac_cpu.factors, A_cpu)
            fac_cpu = factorize!!(fac_cpu, m, fac_cpu.factors)
            @test fac_cpu.info == 0
            d_cpu = ldiv!!(similar(x_cpu_init), fac_cpu, copy(b_cpu))

            x_dev = MtlArray(zeros(T, n))
            b_dev = MtlArray(b_cpu)
            prob_dev = NonlinearProblem(x_dev, metal_noop)
            fac_dev = init(m, prob_dev, x_dev, nothing)
            copyto!(fac_dev.factors, MtlArray(A_cpu))
            fac_dev = factorize!!(fac_dev, m, fac_dev.factors)
            @test fac_dev.info == 0
            @test returncode(fac_dev) === SUCCESS
            d_dev = ldiv!!(similar(x_dev), fac_dev, b_dev)
            @test d_dev isa MtlVector{T}

            S_dev = Array(fac_dev.S)
            @test issorted(S_dev; rev = true)
            eps32 = Float64(eps(Float32))
            @test maximum(abs.(S_dev .- fac_cpu.S)) <= 8 * n * eps32 * fac_cpu.S[1]

            x_h = Array(d_dev)
            bwd = norm(A_cpu * x_h - b_cpu, Inf) / (opnorm(A_cpu, Inf) * norm(x_h, Inf))
            @test bwd <= 4 * n * eps32
            bwd2 = norm(A_cpu * x_h - b_cpu) / (fac_cpu.S[1] * norm(x_h))
            @test bwd2 <= 4 * n * eps32

            fwd = norm(x_h - d_cpu, Inf) / norm(d_cpu, Inf)
            @test fwd <= 8 * n * eps32 * κ
        end
    end

    @testset "Metal KA SVD bit-identical repeated runs and traps: $T" for T in METAL_ELTYPES
        m = SVDFactorization()
        W = metal_wide_eltype(T)
        for n in (15, 16)
            A_cpu, b_cpu = metal_well_conditioned(T, n, 0x2222 + n)
            x1 = MtlArray(zeros(T, n))
            prob1 = NonlinearProblem(x1, metal_noop)
            fac1 = factorize!!(init(m, prob1, x1, nothing), m, MtlArray(A_cpu))
            d1 = ldiv!!(similar(x1), fac1, MtlArray(b_cpu))

            x2 = MtlArray(zeros(T, n))
            prob2 = NonlinearProblem(x2, metal_noop)
            fac2 = factorize!!(init(m, prob2, x2, nothing), m, MtlArray(A_cpu))
            d2 = ldiv!!(similar(x2), fac2, MtlArray(b_cpu))

            @test fac1.info == 0
            @test fac2.info == 0
            @test Array(fac1.U) == Array(fac2.U)
            @test Array(fac1.S) == Array(fac2.S)
            @test Array(fac1.VT) == Array(fac2.VT)
            @test Array(d1) == Array(d2)
        end

        # Subnormal prescaling bit for bit against integer matrix
        E = exponent(floatmin(Float32)) - 8
        B_cpu = T[i + j for i in 1:4, j in 1:4]
        A_sub = ldexp(1.0f0, E) .* B_cpu
        x4 = MtlArray(zeros(T, 4))
        prob4 = NonlinearProblem(x4, metal_noop)
        fac_sub = factorize!!(init(m, prob4, x4, nothing), m, MtlArray(A_sub))
        fac_ref = factorize!!(init(m, prob4, x4, nothing), m, MtlArray(B_cpu))
        @test fac_sub.info == 0
        @test Array(fac_sub.U) == Array(fac_ref.U)
        @test Array(fac_sub.VT) == Array(fac_ref.VT)
        @test Array(fac_sub.S) == ldexp.(Array(fac_ref.S), E)

        # Singular value above floatmax(Float32) leaves solve finite via rwork
        A_hi = fill(T(0.5f0 * floatmax(Float32)), 4, 4) +
               Matrix{T}(Diagonal(fill(T(0.125f0 * floatmax(Float32)), 4)))
        b_hi = fill(T(0.25f0 * floatmax(Float32)), 4)
        x_hi_ref = fill(2.0 / 17.0, 4)
        κ_hi = 17.0
        fac_hi = factorize!!(init(m, prob4, x4, nothing), m, MtlArray(A_hi))
        @test fac_hi.info == 0
        @test !isfinite(Array(fac_hi.S)[1])
        @test all(isfinite, Array(fac_hi.rwork))
        d_hi = ldiv!!(similar(x4), fac_hi, MtlArray(b_hi))
        @test all(isfinite, Array(d_hi))
        rel_hi = norm(W.(Array(d_hi)) - x_hi_ref, Inf) / norm(x_hi_ref, Inf)
        @test rel_hi <= 8 * 4 * Float64(eps(Float32)) * κ_hi

        # Rank-deficient matrices at n = 16 for r in (5, 1, 0)
        n16 = 16
        tol16 = jacobi_tolerance(n16, Float32)
        x16 = MtlArray(zeros(T, n16))
        prob16 = NonlinearProblem(x16, metal_noop)
        for r in (5, 1, 0)
            A_rk = metal_rank_deficient(100r + n16, T, n16, r)
            fac_rk = factorize!!(init(m, prob16, x16, nothing), m, MtlArray(A_rk))
            @test fac_rk.info == 0
            @test returncode(fac_rk) === SUCCESS
            S_rk = Array(fac_rk.S)
            num_rk = S_rk[1] == 0.0f0 ? 0 : count(>(tol16 * S_rk[1]), S_rk)
            @test num_rk == r
            @test fac_rk.rank == r
            @test all(iszero, Array(fac_rk.U)[:, (r + 1):n16])
            x0_rk = randn(Xoshiro(400r + n16), T, n16)
            b_rk = A_rk * x0_rk
            d_rk = ldiv!!(similar(x16), fac_rk, MtlArray(b_rk))
            x_rk = Array(d_rk)
            @test all(isfinite, x_rk)
            if r == 0
                @test all(iszero, x_rk)
            else
                @test norm(A_rk * x_rk - b_rk) <=
                      4 * n16 * eps(Float32) * S_rk[1] * norm(x_rk)
                @test norm(x_rk) <= (1 + 4 * n16 * eps(Float32)) * norm(x0_rk)
            end
        end

        # Negligible column of norm 0.5 * eps(Float32) and 2 * eps(Float32) at n = 16
        ε32 = 0.5f0 * eps(Float32)
        A_neg = Matrix{T}(I, 4, 4)
        A_neg[4, 4] = zero(T)
        A_neg[1, 4] = T(ε32)
        fac_neg = factorize!!(init(m, prob4, x4, nothing), m, MtlArray(A_neg))
        @test fac_neg.info == 0
        @test Array(fac_neg.S)[4] == ε32
        @test all(iszero, Array(fac_neg.U)[:, 4])
        @test count(!iszero, Array(fac_neg.VT)) == 4

        ε16 = 2.0f0 * eps(Float32)
        A_neg16 = Matrix{T}(I, n16, n16)
        A_neg16[n16, n16] = zero(T)
        A_neg16[1, n16] = T(ε16)
        fac_neg16 = factorize!!(init(m, prob16, x16, nothing), m, MtlArray(A_neg16))
        @test fac_neg16.info == 0
        @test Array(fac_neg16.S)[n16] == ε16
        @test all(iszero, Array(fac_neg16.U)[:, n16])
        @test count(!iszero, Array(fac_neg16.VT)) == n16

        # Non-finite NaN (at n = 1, n = 4, and partial-NaN 2x2) sets info == 1
        x1 = MtlArray(zeros(T, 1))
        prob1 = NonlinearProblem(x1, metal_noop)
        fac_nan1 = factorize!!(
            init(m, prob1, x1, nothing), m, MtlArray(fill(T(NaN32), 1, 1)))
        @test fac_nan1.info == 1
        @test returncode(fac_nan1) === SINGULAR

        A_nan = fill(T(NaN32), 4, 4)
        fac_nan = factorize!!(init(m, prob4, x4, nothing), m, MtlArray(A_nan))
        @test fac_nan.info == 1
        @test returncode(fac_nan) === SINGULAR

        x2 = MtlArray(zeros(T, 2))
        prob2 = NonlinearProblem(x2, metal_noop)
        A_part = T[zero(T) T(NaN32); zero(T) zero(T)]
        fac_part = factorize!!(init(m, prob2, x2, nothing), m, MtlArray(A_part))
        @test fac_part.info == 1
        @test returncode(fac_part) === SINGULAR

        # Dispatch and workspace sizes on MtlArray
        @test length(fac_sub.work) == 1
        @test length(fac_sub.rwork) == 4
        m_svd_fac = which(
            factorize!!, (typeof(fac_sub), SVDFactorization{Nothing}, MtlMatrix{T}))
        m_svd_bs = which(backsolve!, (typeof(fac_sub), MtlVector{T}))
        @test endswith(replace(String(m_svd_fac.file), '\\' => '/'), "src/linear/ka/svd.jl")
        @test endswith(replace(String(m_svd_bs.file), '\\' => '/'), "src/linear/ka/svd.jl")
    end

    @testset "Metal pivot search, scaling, singular, and non-finite: $T" for T in METAL_ELTYPES
        W = metal_wide_eltype(T)
        for M in (LUFactorization, QRFactorization)
            m = M()
            n = 6
            tiny = eps(Float32)^2
            A_tiny = zeros(T, n, n)
            for k in 1:n
                A_tiny[k, k] = T(tiny)
                row = k < n ? k + 1 : 1
                A_tiny[row, k] = T(2) + T(k) / T(n)
            end
            b_tiny = T[T(k) for k in 1:n]
            x_ref = W.(A_tiny) \ W.(b_tiny)
            κ = cond(W.(A_tiny), Inf)

            x_dev = MtlArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, metal_noop)
            fac = init(m, prob, x_dev, nothing)
            copyto!(fac.factors, MtlArray(A_tiny))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            d = ldiv!!(similar(x_dev), fac, MtlArray(b_tiny))
            rel = norm(W.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf)
            @test rel <= 8 * n * Float64(eps(Float32)) * κ

            # Column scaled by 1.0f20
            A0, _ = metal_well_conditioned(T, 4)
            A0[:, 1] .*= 1.0f20
            x4 = MtlArray(zeros(T, 4))
            prob4 = NonlinearProblem(x4, metal_noop)
            fac_sc = init(m, prob4, x4, nothing)
            copyto!(fac_sc.factors, MtlArray(A0))
            fac_sc = factorize!!(fac_sc, m, fac_sc.factors)
            @test fac_sc.info == 0
            @test all(isfinite, Array(fac_sc.factors))

            # Singular zero matrix and zero column 2
            fac_z = init(m, prob4, x4, nothing)
            fill!(fac_z.factors, zero(T))
            fac_z = factorize!!(fac_z, m, fac_z.factors)
            @test fac_z.info == 1
            @test returncode(fac_z) === SINGULAR
            @test all(isfinite, Array(fac_z.factors))

            A_c2 = Matrix{T}(I, 4, 4)
            A_c2[:, 2] .= zero(T)
            copyto!(fac_z.factors, MtlArray(A_c2))
            fac_z = factorize!!(fac_z, m, fac_z.factors)
            @test fac_z.info == 2
            @test all(isfinite, Array(fac_z.factors))

            # 2x2 equal rows
            A_eq = T[1 2; 1 2]
            x2 = MtlArray(zeros(T, 2))
            prob2 = NonlinearProblem(x2, metal_noop)
            fac_eq = init(m, prob2, x2, nothing)
            copyto!(fac_eq.factors, MtlArray(A_eq))
            fac_eq = factorize!!(fac_eq, m, fac_eq.factors)
            if M === LUFactorization
                @test fac_eq.info == 2
                @test returncode(fac_eq) === SINGULAR
            end
            @test all(isfinite, Array(fac_eq.factors))

            # Non-finite input does not throw
            A_nan = Matrix{T}(I, 4, 4)
            A_nan[2, 2] = T(NaN32)
            copyto!(fac_z.factors, MtlArray(A_nan))
            fac_z = factorize!!(fac_z, m, fac_z.factors)
            d_nan = ldiv!!(similar(x4), fac_z, MtlArray(ones(T, 4)))
            @test d_nan isa MtlVector{T}
        end
    end
    BLAS.set_num_threads(old_blas_threads)
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
