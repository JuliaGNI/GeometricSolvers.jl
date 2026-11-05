using GeometricSolvers
using GeometricSolvers:
                        DenseLU, DenseQR, DenseSVD, DeviceLU, DeviceQR, DeviceSVD,
                        LUFactorization, NonlinearProblem, QRFactorization, SUCCESS,
                        SVDFactorization, StaticLU, StaticQR, StaticSVD, dense_factor,
                        factorize!!, jacobi_eigen, ldiv!!, returncode, singular_values
using JLArrays
using LinearAlgebra
using Random
using StaticArrays
using Test

include("../helpers/matrix.jl")
include("../helpers/randsvd.jl")

const C_BOUND = 16
nop(y, x, p) = y
const NOP_PROB = NonlinearProblem(nop)

JLArrays.allowscalar(false)

backward_error(A, x, b) = norm(A * x - b) / (opnorm(Matrix(A), 2) * norm(x) + norm(b))

@testset "Linear and spectral stability" begin
    @testset "randsvd, randhermitian, and minnorm_solution helpers" begin
        for T in ELTYPES
            R = real(T)
            n = 8
            r = 5
            κ = R(100)

            G = randn(Xoshiro(101), T, n, n)
            Q_haar = _haar_unitary(Xoshiro(101), T, n)
            R_haar = Q_haar' * G
            @test opnorm(Q_haar' * Q_haar - I, 2) <= C_BOUND * n * eps(R)
            @test all(
                i -> real(R_haar[i, i]) > 0 &&
                     abs(imag(R_haar[i, i])) <= C_BOUND * n * eps(R),
                1:n
            )

            A, U, σ, V = randsvd(Xoshiro(102), T, n, κ; rank = r)
            @test size(A) == (n, n)
            @test eltype(A) === T
            @test eltype(σ) === R
            @test opnorm(U' * U - I, 2) <= C_BOUND * n * eps(R)
            @test opnorm(V' * V - I, 2) <= C_BOUND * n * eps(R)
            @test σ[1] == one(R)
            @test isapprox(σ[r], inv(κ); rtol = C_BOUND * eps(R))
            @test all(
                i -> isapprox(σ[i], κ^(-R(i - 1) / R(r - 1)); rtol = C_BOUND * eps(R)),
                1:r
            )
            @test all(iszero, @view σ[(r + 1):n])
            @test opnorm(A - U * Diagonal(T.(σ)) * V', 2) <= C_BOUND * n * eps(R)

            b = randn(Xoshiro(103), T, n)
            x_min = minnorm_solution(U, σ, V, b, r)
            @test eltype(x_min) === T
            @test norm((@view V[:, (r + 1):n])' * x_min) <=
                  C_BOUND * n * eps(R) * norm(x_min)
            @test norm(A' * (A * x_min - b)) <=
                  C_BOUND * n * eps(R) * opnorm(A, 2) * norm(b)

            H, λ, Q = randhermitian(Xoshiro(104), T, n, κ)
            @test size(H) == (n, n)
            @test eltype(H) === T
            @test ishermitian(H)
            @test opnorm(Q' * Q - I, 2) <= C_BOUND * n * eps(R)
            @test λ[1] == one(R)
            @test isapprox(λ[n], inv(κ); rtol = C_BOUND * eps(R))
            @test all(
                i -> isapprox(λ[i], κ^(-R(i - 1) / R(n - 1)); rtol = C_BOUND * eps(R)),
                1:n
            )
            @test opnorm(H - Q * Diagonal(T.(λ)) * Q', 2) <= C_BOUND * n * eps(R)
        end
    end

    @testset "Full-rank LU, QR, and SVD backward error and singular values" begin
        rng = Xoshiro(11)
        for T in ELTYPES
            R = real(T)
            kappas = R === Float32 ? (10, 10^4) : (10, 10^4, 10^8)
            for n in (4, 16, 64), κ in kappas

                A, _, σ, _ = randsvd(rng, T, n, R(κ))
                b = randn(rng, T, n)
                bound = C_BOUND * n * eps(R)
                for M in (LUFactorization, QRFactorization, SVDFactorization)
                    m = M()
                    if n in (4, 16)
                        fac_s = factorize!!(nothing, m, SMatrix{n, n, T}(A))
                        @test returncode(fac_s) === SUCCESS
                        x_s = Vector(ldiv!!(nothing, fac_s, SVector{n, T}(b)))
                        @test backward_error(A, x_s, b) <= bound
                        if M === SVDFactorization
                            @test fac_s isa StaticSVD
                            s_hat = Vector(singular_values(fac_s))
                            @test maximum(abs.(s_hat .- σ)) / σ[1] <= bound
                        elseif M === LUFactorization
                            @test fac_s isa StaticLU
                        else
                            @test fac_s isa StaticQR
                        end
                    end

                    fac_d = dense_factor(m, NOP_PROB, zeros(T, n), nothing)
                    fac_d = factorize!!(fac_d, m, copy(A))
                    @test returncode(fac_d) === SUCCESS
                    x_d = ldiv!!(zeros(T, n), fac_d, copy(b))
                    @test backward_error(A, x_d, b) <= bound
                    if M === SVDFactorization
                        @test fac_d isa DenseSVD
                        @test maximum(abs.(fac_d.S .- σ)) / σ[1] <= bound
                    elseif M === LUFactorization
                        @test fac_d isa DenseLU
                    else
                        @test fac_d isa DenseQR
                    end

                    fac_g = dense_factor(m, NOP_PROB, JLArray(zeros(T, n)), nothing)
                    fac_g = factorize!!(fac_g, m, JLArray(A))
                    @test returncode(fac_g) === SUCCESS
                    x_g = Array(ldiv!!(JLArray(zeros(T, n)), fac_g, JLArray(b)))
                    @test backward_error(A, x_g, b) <= bound
                    if M === SVDFactorization
                        @test fac_g isa DeviceSVD
                        @test maximum(abs.(Array(fac_g.S) .- σ)) / σ[1] <= bound
                    elseif M === LUFactorization
                        @test fac_g isa DeviceLU
                    else
                        @test fac_g isa DeviceQR
                    end
                end
            end
        end
    end

    @testset "Hermitian jacobi_eigen residual, orthogonality, and eigenvalues" begin
        rng = Xoshiro(12)
        for T in ELTYPES
            R = real(T)
            kappas = R === Float32 ? (10, 10^4) : (10, 10^4, 10^8)
            for n in (4, 16), κ in kappas

                H, λ_true, _ = randhermitian(rng, T, n, R(κ))
                Hs = SMatrix{n, n, T}(H)
                λ_hat, V_hat, st = jacobi_eigen(Hs)
                @test st === SUCCESS
                Hm = Matrix(Hs)
                Vm = Matrix(V_hat)
                bound = C_BOUND * n * eps(R)
                @test opnorm(Hm * Vm - Vm * Diagonal(Vector(λ_hat)), 2) / opnorm(Hm, 2) <=
                      bound
                @test opnorm(Vm' * Vm - I, 2) <= bound
                λ_sorted = sort(Vector(λ_hat); rev = true)
                @test maximum(abs.(λ_sorted .- λ_true)) / λ_true[1] <= bound
            end
        end
    end

    @testset "Rank-deficient SVD minimum-norm ldiv!!" begin
        rng = Xoshiro(13)
        for T in ELTYPES
            R = real(T)
            for n in (4, 16)
                r = n - 2
                A, U, σ, V = randsvd(rng, T, n, R(100); rank = r)
                b_range = U[:, 1:r] * randn(rng, T, r)
                b_null = U[:, (r + 1):n] * (R(0.1) .* randn(rng, T, n - r))
                b = b_range + b_null
                x_min = minnorm_solution(U, σ, V, b, r)
                rel_bound = C_BOUND * n * R(100) * eps(R) * (norm(b) / norm(A * x_min))
                m = SVDFactorization()

                fac_s = factorize!!(nothing, m, SMatrix{n, n, T}(A))
                @test returncode(fac_s) === SUCCESS
                @test fac_s.rank == r
                x_s = Vector(ldiv!!(nothing, fac_s, SVector{n, T}(b)))
                @test norm(x_s - x_min) / norm(x_min) <= rel_bound

                fac_d = dense_factor(m, NOP_PROB, zeros(T, n), nothing)
                fac_d = factorize!!(fac_d, m, copy(A))
                @test returncode(fac_d) === SUCCESS
                @test fac_d.rank == r
                x_d = ldiv!!(zeros(T, n), fac_d, copy(b))
                @test norm(x_d - x_min) / norm(x_min) <= rel_bound

                fac_g = dense_factor(m, NOP_PROB, JLArray(zeros(T, n)), nothing)
                fac_g = factorize!!(fac_g, m, JLArray(A))
                @test returncode(fac_g) === SUCCESS
                @test fac_g.rank == r
                x_g = Array(ldiv!!(JLArray(zeros(T, n)), fac_g, JLArray(b)))
                @test norm(x_g - x_min) / norm(x_min) <= rel_bound
            end
        end
    end

    @testset "Mixed-precision ldiv!! (TF = Float32, T in 64-bit ELTYPES)" begin
        rng = Xoshiro(14)
        for T in filter(T -> sizeof(real(T)) == 8, ELTYPES)
            for n in (4, 16, 64),
                κ in (10, 10^4),
                M in (LUFactorization, QRFactorization, SVDFactorization)
                A, _, _, _ = randsvd(rng, T, n, Float64(κ))
                b = randn(rng, T, n)
                m = M(Float32)
                bound = C_BOUND * n * eps(Float32)

                if n in (4, 16)
                    fac_s = factorize!!(nothing, m, SMatrix{n, n, T}(A))
                    @test returncode(fac_s) === SUCCESS
                    x_s = ldiv!!(nothing, fac_s, SVector{n, T}(b))
                    @test eltype(x_s) === T
                    @test backward_error(A, Vector(x_s), b) <= bound
                end

                fac_d = dense_factor(m, NOP_PROB, zeros(T, n), nothing)
                fac_d = factorize!!(fac_d, m, copy(A))
                @test returncode(fac_d) === SUCCESS
                x_d = ldiv!!(zeros(T, n), fac_d, copy(b))
                @test eltype(x_d) === T
                @test backward_error(A, x_d, b) <= bound

                fac_g = dense_factor(m, NOP_PROB, JLArray(zeros(T, n)), nothing)
                fac_g = factorize!!(fac_g, m, JLArray(A))
                @test returncode(fac_g) === SUCCESS
                x_g = Array(ldiv!!(JLArray(zeros(T, n)), fac_g, JLArray(b)))
                @test eltype(x_g) === T
                @test backward_error(A, x_g, b) <= bound
            end
        end
    end
end
