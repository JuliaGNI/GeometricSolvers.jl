# Repeat each base TRSM kernel of NextLA on Metal, 10 times at each of three sizes, and count the
# runs whose backward error exceeds the census tolerance max(100, n) * eps(Float32). The kernels
# write and read one shared slot per step without a barrier between (NextLA 0.2.3,
# `src/trsm.jl`), so a single passing run says little.
#
# Usage, from the repository root:
#   JULIA_LOAD_PATH="@:$PWD/test/gpu/metal:@stdlib" julia --startup-file=no \
#       --project=scripts/spikes/nextla scripts/spikes/nextla/probes/trsm_repeat.jl

using LinearAlgebra
using Metal
using NextLA: NextLA
using Random

const T = Float32
const KERNELS = (
    ("LL", NextLA.LeftLowerTRSM!, :L, :left), ("LU", NextLA.LeftUpperTRSM!, :U, :left),
    ("RL", NextLA.RightLowerTRSM!, :L, :right), ("RU", NextLA.RightUpperTRSM!, :U, :right))

rng = Xoshiro(1)
for (name, f!, uplo, side) in KERNELS, n in (40, 256, 1000)

    wrong = 0
    worst = 0.0
    for _ in 1:10
        A = randn(rng, T, n, n) + n * I
        A = uplo == :L ? tril(A) : triu(A)
        B = side == :left ? randn(rng, T, n, 5) : randn(rng, T, 5, n)
        X = MtlArray(B)
        f!(MtlArray(A), X)
        Metal.synchronize()
        Xh, Ah, Bh = Float64.(Array(X)), Float64.(A), Float64.(B)
        R = side == :left ? Ah * Xh - Bh : Xh * Ah - Bh
        e = norm(R) / (norm(Ah) * norm(Xh) + norm(Bh))
        worst = max(worst, e)
        wrong += e > max(100, n) * eps(T)
    end
    println("TRSM $name n = $n: $wrong of 10 wrong, worst relerr $(round(worst; sigdigits = 2))")
end
println("Packages: NextLA ", pkgversion(NextLA), ", Metal ", pkgversion(Metal))
