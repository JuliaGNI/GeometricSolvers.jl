# The R1 static LU and QR solves in a KernelAbstractions kernel on an Apple GPU, one thread per
# system, in `Float32` and `ComplexF32`: Metal has no `Float64`. Each step meets the bound of
# check 1, `4n · eps(TF) · κ∞(A)` against a `BigFloat` reference, and a singular input gives
# `info > 0` in the kernel, as on the host. `runtests.jl` runs this file in the `metal` group.
#
# Where `Metal.functional()` is `false`, as inside a sandbox, the file runs no test and records the
# device skip.

using GeometricSolvers
using Metal, Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

if Metal.functional()
    @testset "$M on Metal, $T, n = $n" for M in (LUFactorization, QRFactorization),
        T in filter(T -> real(T) === Float32, ELTYPES), n in STATIC_SIZES
        m = M()
        S = singular_matrices(3000n + 1, T, n)
        systems = conditioned_systems(1000n + 1, T, Nothing, n)
        As = [first.(systems); S.zero_columns; [S.zero_matrix]]
        bs = [last.(systems); fill(last(first(systems)), length(As) - length(systems))]
        x, info = kernel_solve(MtlArray, m, As, bs)
        for (k, (A, b)) in enumerate(systems)
            @test info[k] == 0
            @test relerr(x[k], reference(A, b)) <= bound(A, T, Nothing)
        end
        @test all(>(0), info[(length(systems) + 1):end])
    end
else
    @test_skip Metal.functional()
end
