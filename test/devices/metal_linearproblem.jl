# The linear-solve object `LinearProblem` on `MtlArray`s on an Apple GPU, in `Float32`: Metal has
# no `Float64`. A square system with LU, QR, SVD and LDLᵀ, an `m > n` system with QR and SVD and an
# `m < n` system with the SVD, each to `64 eps(Float32)` against the solution in `BigFloat`
# (`κ ≤ 3`, `κ ≤ 9` for the Hermitian positive definite matrix of LDLᵀ); the rank of the SVD path on
# the wide fixture with the singular values `1, 1e-10, 1e-14`; and the refusal of the QR with
# `m < n` on a device array. The CPU tests of `test/linear/linearproblem.jl` run the same code on a
# `JLArray`; these check that the factors of the device fill the fields that the object reads.
# `runtests.jl` runs this file in the `metal` group.
#
# Where `Metal.functional()` is `false`, as inside a sandbox, the file runs no test and records the
# device skip. On a virtualized GPU it runs no test and records a skip for issue #34.

using GeometricSolvers
using GeometricSolvers: SUCCESS
using GPUArraysCore: allowscalar
using LinearAlgebra: norm
using Metal
using Test

include("../helpers/metal_device.jl")

function phase_free_wellcond(m, n)
    [Float32((i == j ? 2.0 : 0.0) + 0.3 * sin(i + 2j)) for i in 1:m, j in 1:n]
end

function metal_spd(n)
    Wb = big.(phase_free_wellcond(n, n))
    return Float32.(Wb' * Wb)
end

# the `m × length(σ)` matrix with the singular values exactly `σ`, a signed column permutation of a
# diagonal
function metal_fixture(σ, m)
    n = length(σ)
    A = zeros(Float32, m, n)
    perm = circshift(1:n, 1)
    for j in 1:n
        A[perm[j], j] = Float32(isodd(j) ? 1 : -1) * Float32(σ[j])
    end
    return A
end

function metal_xref(A, b)
    Ab = big.(A)
    bb = big.(b)
    m, n = size(Ab)
    m == n && return Ab \ bb
    m > n && return (Ab' * Ab) \ (Ab' * bb)
    return Ab' * ((Ab * Ab') \ bb)
end

metal_relerr(x, xr) = Float64(norm(big.(x) - xr) / norm(xr))

if Metal.functional() && !on_virtual_gpu()
    allowscalar(false)

    @testset "LinearProblem on Metal: $(nameof(typeof(method))), $m × $n" for (method, m, n) in (
        (LUFactorization(), 4, 4), (QRFactorization(), 4, 4), (SVDFactorization(), 4, 4),
        (LDLTFactorization(), 4, 4), (QRFactorization(), 5, 3), (SVDFactorization(), 5, 3),
        (SVDFactorization(), 3, 5))
        A = method isa LDLTFactorization ? metal_spd(4) : phase_free_wellcond(m, n)
        b = Float32.(1:m)
        solver = init(LinearProblem(MtlArray(A)), method)
        x, st = solve!(solver, MtlArray(b))
        @test x === solver.x
        @test st.code == SUCCESS
        @test st.rank == (method isa SVDFactorization ? min(m, n) : -1)
        @test metal_relerr(Array(x), metal_xref(A, b)) <= 64 * eps(Float32)
    end

    @testset "LinearProblem on Metal: the rank of the wide fixture, rtol = $rtol" for (rtol, want) in (
        (nothing, 1), (1e-9, 1), (1e-12, 2), (0.0, 3))
        A = Matrix(metal_fixture((1.0, 1e-10, 1e-14), 5)')
        method = rtol === nothing ? SVDFactorization() : SVDFactorization(Float32; rtol)
        x, st = solve!(init(LinearProblem(MtlArray(A)), method), MtlArray(Float32[1, 2, 3]))
        @test st.code == SUCCESS
        @test st.rank == want
        @test all(isfinite, Array(x))
    end

    @testset "LinearProblem on Metal: QR with m < n" begin
        @test_throws r"^ArgumentError: QRFactorization has no minimum-norm solve of an m × n system with m < n on a device array; take SVDFactorization\(\)$" init(
            LinearProblem(MtlArray(phase_free_wellcond(3, 5))), QRFactorization())
    end
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
