# The R1 static Jacobi SVD and eigensolver in a KernelAbstractions kernel on an Apple GPU, one
# thread per system, in `Float32` and `ComplexF32`: Metal has no `Float64`. Each full-rank system
# meets the bound of part I, `4n · eps(real(T)) · κ∞(A)` against a `BigFloat` reference; a
# rank-deficient one ends below the sweep cap with the exact rank; the subnormal input of part J's
# check 1 gives finite factors; and `jacobi_eigen(im * M)` for a real skew `M` gives the
# eigenvalues of the host within working precision, with a unitary `V`.
#
# `runtests.jl` runs this file in the `metal` group. Where `Metal.functional()` is `false`, as
# inside a sandbox, the file runs no test and records the device skip.

using GeometricSolvers
using GeometricSolvers: factorize!!, ldiv!!, jacobi_eigen
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using LinearAlgebra: I, norm
using Metal, Test
using Random: Xoshiro, randn
using StaticArrays: SMatrix, SVector

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")
include("../helpers/metal_device.jl")

const METAL_ELTYPES = filter(T -> real(T) === Float32, ELTYPES)

@kernel function metal_eigen_kernel!(values, vectors, @Const(A))
    i = @index(Global)
    λ, V, _ = jacobi_eigen(A[i])
    n = size(A[i], 1)
    for j in 1:n
        values[j, i] = λ[j]
    end
    for j in 1:n, k in 1:n

        vectors[k, j, i] = V[k, j]
    end
end

if Metal.functional() && !on_virtual_gpu()
    @testset "the Jacobi SVD on Metal, $T, n = $n" for T in METAL_ELTYPES, n in (1, 4, 16)

        m = SVDFactorization()
        R = real(T)
        systems = conditioned_systems(1000n + 3, T, Nothing, n)
        rank0 = rank_deficient(100n + 1, T, n, 0)
        rank1 = rank_deficient(100n + 5, T, n, 1)
        # the subnormal matrix of check 1: every entry is below `floatmin`, and an unclamped
        # prescale would turn it into `Inf`
        E = exponent(floatmin(R)) - 8
        subnormal = ldexp(one(R), E) * SMatrix{n, n, T}([i + j for i in 1:n, j in 1:n])
        As = vcat(first.(systems), [rank0, rank1, subnormal])
        bs = vcat(last.(systems),
            [
                zero(SVector{n, T}), SVector{n, T}(fill(one(T), n)), SVector{n, T}(fill(one(T), n))])
        x, info = kernel_solve(MtlArray, m, As, bs)
        for (k, (A, b)) in enumerate(systems)
            @test info[k] == 0
            @test relerr(x[k], reference(A, b)) <= bound(A, T, Nothing)
        end
        # the rank of a rank-`r` matrix is `r`, `info` being the index of the first singular
        # value the cutoff counts as zero; a rank-1 matrix is full rank at n = 1
        @test info[length(systems) + 1] == Int32(1)
        n > 1 && @test info[length(systems) + 2] == Int32(2)
        # the subnormal input gives finite factors
        fac = factorize!!(nothing, m, subnormal)
        @test all(isfinite, fac.factors)
        @test all(isfinite, fac.σ)
        @test ldiv!!(nothing, fac, SVector{n, T}(fill(one(T), n))) isa SVector{n, T}
    end

    @testset "jacobi_eigen(im * M) on Metal, $T, n = $n" for T in METAL_ELTYPES,
        n in (1, 4, 8)

        M = randn(Xoshiro(20 + n), T, 2n, 2n)
        M = SMatrix{2n, 2n, T}(M - M')
        _, V, status = jacobi_eigen(im * M)
        @test status === SUCCESS
        @test norm(V' * V - I) <= 4 * (2n) * eps(real(T))

        # the same in a kernel, one matrix per thread, against the host
        As = [im * M, im * (2 * M)]
        A = MtlArray(As)
        values = MtlArray(zeros(T, 2n, length(As)))
        vectors = MtlArray(zeros(complex(T), 2n, 2n, length(As)))
        backend = KernelAbstractions.get_backend(A)
        metal_eigen_kernel!(backend)(values, vectors, A; ndrange = length(As))
        KernelAbstractions.synchronize(backend)
        host = map(jacobi_eigen, As)
        @test all(isfinite, Array(values))
        @test isapprox(Array(values)[:, 1], Vector(host[1][1]);
            rtol = 8 * 2n * eps(real(T)) * max(1, norm(M)), atol = 8 * 2n * eps(real(T)) *
                                                                   max(1, norm(M)))
        @test isapprox(Array(vectors)[:, :, 1], Matrix(host[1][2]);
            rtol = 8 * 2n * eps(real(T)), atol = 8 * 2n * eps(real(T)))
    end
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
