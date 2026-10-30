using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray, JLVector
using LinearAlgebra: BLAS, I, LAPACK, norm, opnorm
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization, SVDFactorization
using GeometricSolvers: factorize!!, init, ldiv!!, returncode

include("../helpers/matrix.jl")

const PRECISION_PAIRS = (map(T -> (T, Nothing), ELTYPES)...,
    map(T -> (T, Float32), filter(T -> real(T) === Float64, ELTYPES))...)

allowscalar(false)

ka_noop(r, y, p) = nothing

@testset "KA device SVD agrees with CPU LAPACK gesvd at n = 10^3 on JLArray" begin
    n = 10^3
    old_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(min(2, old_threads))
        @testset "SVD ($T, n = $n)" for T in REAL_ELTYPES
            rng = Xoshiro(0x5eed + hash(T))
            A_cpu = randn(rng, T, n, n) + T(2 * n) * I
            b_cpu = randn(rng, T, n)

            A64 = Float64.(A_cpu)
            anorm = opnorm(A64, Inf)
            _, ipiv, _ = LAPACK.getrf!(A64)
            LAPACK.getri!(A64, ipiv)
            κ = anorm * opnorm(A64, Inf)
            @test κ <= 10.0

            m = SVDFactorization()
            x_cpu_init = zeros(T, n)
            prob_cpu = NonlinearProblem(x_cpu_init, ka_noop)
            fac_cpu = init(m, prob_cpu, x_cpu_init, nothing)
            copyto!(fac_cpu.factors, A_cpu)
            fac_cpu = factorize!!(fac_cpu, m, fac_cpu.factors)
            @test fac_cpu.info == 0
            x_cpu = ldiv!!(similar(x_cpu_init), fac_cpu, copy(b_cpu))

            x_dev_init = JLArray(zeros(T, n))
            b_dev = JLArray(copy(b_cpu))
            prob_dev = NonlinearProblem(x_dev_init, ka_noop)
            fac_dev = init(m, prob_dev, x_dev_init, nothing)
            copyto!(fac_dev.factors, JLArray(A_cpu))
            fac_dev = factorize!!(fac_dev, m, fac_dev.factors)
            @test fac_dev.info == 0
            @test returncode(fac_dev) === SUCCESS
            d_dev = ldiv!!(similar(x_dev_init), fac_dev, b_dev)
            @test d_dev isa JLVector{T}

            S_dev = Array(fac_dev.S)
            @test issorted(S_dev; rev = true)
            @test maximum(abs.(S_dev .- fac_cpu.S)) <= 8 * n * eps(T) * fac_cpu.S[1]

            x_dev = Array(d_dev)
            @test norm(A_cpu * x_dev - b_cpu, Inf) <=
                  4 * n * eps(T) * anorm * norm(x_dev, Inf)
            @test norm(A_cpu * x_dev - b_cpu) <=
                  4 * n * eps(T) * opnorm(A_cpu) * norm(x_dev)
            @test norm(x_dev - x_cpu, Inf) / norm(x_cpu, Inf) <= 8 * n * eps(T) * κ

            GC.gc()
        end
    finally
        BLAS.set_num_threads(old_threads)
    end
end

@testset "JET.@test_opt on init, factorize!!, and ldiv!! for KA factorizations on JLArray" begin
    for (T, TF) in PRECISION_PAIRS
        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = TF === Nothing ? M() : M(TF)
            x_dev = JLArray(zeros(T, 4))
            b_dev = JLArray(ones(T, 4))
            prob_dev = NonlinearProblem(x_dev, ka_noop)
            JET.@test_opt target_modules = (GeometricSolvers,) init(
                m, prob_dev, x_dev, nothing)
            fac = init(m, prob_dev, x_dev, nothing)
            JET.@test_opt target_modules = (GeometricSolvers,) factorize!!(
                fac, m, fac.factors)
            JET.@test_opt target_modules = (GeometricSolvers,) ldiv!!(
                similar(x_dev), fac, b_dev)
        end
    end
end
