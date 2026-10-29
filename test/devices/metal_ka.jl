using GPUArraysCore: allowscalar
using LinearAlgebra: I, cond, norm, opnorm
using Metal
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization
using GeometricSolvers: backsolve!, factorize!!, init, ldiv!!, returncode

include("../helpers/matrix.jl")
include("../helpers/metal_device.jl")

if Metal.functional() && !on_virtual_gpu()
    allowscalar(false)

    const METAL_ELTYPES = filter(T -> real(T) === Float32, ELTYPES)

    function metal_well_conditioned(T::Type, n::Int, seed::Integer = 0x5eed)
        rng = Xoshiro(seed)
        A = randn(rng, T, n, n) + T(2 * n) * I
        b = randn(rng, T, n)
        return A, b
    end

    @testset "Metal KA LU and QR agree with CPU solves at n = $n" for n in (1, 16, 256)
        @testset "T = $T" for T in METAL_ELTYPES
            for M in (LUFactorization, QRFactorization)
                m = M()
                A_cpu, b_cpu = metal_well_conditioned(T, n)
                κ = Float64(cond(A_cpu, Inf))
                @test κ <= 10.0

                x_cpu_init = zeros(T, n)
                prob_cpu = NonlinearProblem(x_cpu_init, (r, y, p) -> nothing)
                fac_cpu = init(m, prob_cpu, x_cpu_init, nothing)
                copyto!(fac_cpu.factors, A_cpu)
                fac_cpu = factorize!!(fac_cpu, m, fac_cpu.factors)
                @test fac_cpu.info == 0
                d_cpu = ldiv!!(similar(x_cpu_init), fac_cpu, copy(b_cpu))

                x_dev = MtlArray(zeros(T, n))
                b_dev = MtlArray(b_cpu)
                prob_dev = NonlinearProblem(x_dev, (r, y, p) -> nothing)
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

    @testset "Metal pivot search, scaling, singular, and non-finite: $T" for T in METAL_ELTYPES
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
            x_ref = big.(A_tiny) \ big.(b_tiny)
            κ = Float64(cond(big.(A_tiny), Inf))

            x_dev = MtlArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
            fac = init(m, prob, x_dev, nothing)
            copyto!(fac.factors, MtlArray(A_tiny))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            d = ldiv!!(similar(x_dev), fac, MtlArray(b_tiny))
            rel = Float64(norm(big.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf))
            @test rel <= 8 * n * Float64(eps(Float32)) * κ

            # Column scaled by 1.0f20
            A0, _ = metal_well_conditioned(T, 4)
            A0[:, 1] .*= 1.0f20
            x4 = MtlArray(zeros(T, 4))
            prob4 = NonlinearProblem(x4, (r, y, p) -> nothing)
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
            prob2 = NonlinearProblem(x2, (r, y, p) -> nothing)
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
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
