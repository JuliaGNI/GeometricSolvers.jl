# The R1 static LU and QR solves in a KernelAbstractions kernel on an Apple GPU, one thread per
# system, in `Float32` and `ComplexF32`: Metal has no `Float64`. Each step meets the bound of
# check 1, `4n · eps(TF) · κ∞(A)` against a `BigFloat` reference, and a singular input gives
# `info > 0` in the kernel, as on the host. `QRFactorization` runs at every `n` in `1:12` beside
# `STATIC_SIZES`, because other fills of `x` in the static QR make it fail on Metal at sizes in
# `3:10` (`scripts/spikes/metal_qr/README.md`). `runtests.jl` runs this file in the `metal` group.
#
# Where `Metal.functional()` is `false`, as inside a sandbox, the file runs no test and records the
# device skip. On a virtualized GPU it runs no test and records a skip for issue #34.

using GeometricSolvers
using Metal, Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")
include("../helpers/metal_device.jl")

if Metal.functional() && !on_virtual_gpu()
    @testset "$M on Metal, $T, n = $n" for M in (LUFactorization, QRFactorization),
        T in filter(T -> real(T) === Float32, ELTYPES),
        n in (M === QRFactorization ? sort(union(1:12, STATIC_SIZES)) : STATIC_SIZES)
        m = M()
        S = singular_matrices(3000n + 1, T, n)
        systems = conditioned_systems(1000n + 1, T, Nothing, n)
        # two equal rows give an exact zero pivot in LU, and as a rule no exact zero in QR
        equal = M === LUFactorization && n > 1 ? [S.equal_rows] : empty(S.zero_columns)
        As = [first.(systems); S.zero_columns; equal; [S.zero_matrix]]
        bs = [last.(systems); fill(last(first(systems)), length(As) - length(systems))]
        x, info = kernel_solve(MtlArray, m, As, bs)
        for (k, (A, b)) in enumerate(systems)
            @test info[k] == 0
            @test relerr(x[k], reference(A, b)) <= bound(A, T, Nothing)
        end
        @test all(>(0), info[(length(systems) + 1):end])
    end
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
