# `NextLA.trsm(side, uplo, transa, diag, A, B)` documents `transa` and `diag`, but its dispatch
# reads only `side` and `uplo` (NextLA 0.2.3, `src/trsm.jl`). This probe solves a left lower
# system with each flag pair and compares with the solve the flags ask for, on the CPU backend.
# A relative error near 0 means the flags are honoured; near 1, or far above eps, means they are not.
#
# Usage, from the repository root:
#   julia --startup-file=no --project=scripts/spikes/nextla scripts/spikes/nextla/probes/trsm_flags.jl

using KernelAbstractions: KernelAbstractions
using LinearAlgebra
using NextLA: NextLA
using Random

const n = 40
L(A) = LowerTriangular(A)
const CASES = (('N', 'N') => A -> L(A), ('N', 'U') => A -> UnitLowerTriangular(A),
    ('T', 'N') => A -> transpose(L(A)))

for seed in 1:3
    rng = Xoshiro(seed)
    A = tril(randn(rng, n, n)) + n * I
    B = randn(rng, n, 5)
    for ((transa, diag), op) in CASES
        X = copy(B)                  # `trsm` overwrites B with the solution and returns nothing
        NextLA.trsm('L', 'L', transa, diag, copy(A), X)
        KernelAbstractions.synchronize(KernelAbstractions.get_backend(X))
        Xref = op(A) \ B
        e = norm(X - Xref) / norm(Xref)
        println("seed $seed, transa = '$transa', diag = '$diag': relerr $(round(e; sigdigits = 2))")
    end
end
println("Packages: NextLA ", pkgversion(NextLA))
