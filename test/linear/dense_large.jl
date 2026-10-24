using GeometricSolvers
using GeometricSolvers: factorize!!, init, ldiv!!, returncode
using LinearAlgebra: BLAS, LAPACK, opnorm
using Random: Xoshiro
using Test

include("../helpers/matrix.jl")

function twiceprec_mul(A::Matrix{T}, x::Vector{T}) where {T <: Real}
    m, n = size(A)
    acc = zeros(Base.TwicePrecision{Float64}, m)
    @inbounds for j in 1:n
        xj = Base.TwicePrecision{Float64}(Float64(x[j]))
        for i in 1:m
            acc[i] += Base.TwicePrecision{Float64}(Float64(A[i, j])) * xj
        end
    end
    return T[T(Float64(acc[i])) for i in 1:m]
end

@testset "Size n = 10^4, the top of the dense R3 band (check 5)" begin
    n = 10^4
    old_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(min(2, old_threads))
        @testset "LU and QR ($T, n = $n)" for T in REAL_ELTYPES
            rng = Xoshiro(10_000 + hash(T))
            x_ref = randn(rng, T, n)
            A = randn(rng, T, n, n)
            b = twiceprec_mul(A, x_ref)
            anorm = opnorm(A, Inf)
            x0 = zeros(T, n)
            d = similar(b)

            fac_lu = init(LUFactorization(), nothing, x0, nothing)
            copyto!(fac_lu.factors, A)
            fac_lu = factorize!!(fac_lu, LUFactorization(), fac_lu.factors)
            @test returncode(fac_lu) === SUCCESS
            κ = Float64(1 / LAPACK.gecon!('I', fac_lu.factors, anorm))
            bnd = 4 * n * eps(T) * κ
            x_lu = ldiv!!(d, fac_lu, b)
            err_lu = maximum(abs, Float64.(x_lu) .- Float64.(x_ref)) /
                     maximum(abs, Float64.(x_ref))
            @test err_lu <= bnd

            fac_qr = init(QRFactorization(), nothing, x0, nothing)
            copyto!(fac_qr.factors, A)
            fac_qr = factorize!!(fac_qr, QRFactorization(), fac_qr.factors)
            @test returncode(fac_qr) === SUCCESS
            x_qr = ldiv!!(d, fac_qr, b)
            err_qr = maximum(abs, Float64.(x_qr) .- Float64.(x_ref)) /
                     maximum(abs, Float64.(x_ref))
            @test err_qr <= bnd

            GC.gc()
        end
    finally
        BLAS.set_num_threads(old_threads)
    end
end
