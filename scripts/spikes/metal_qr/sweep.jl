# For QRFactorization and LUFactorization in Float32 and ComplexF32, the n in 1:12 at which the
# Metal kernel of `test/helpers/staticsystems.jl` misses the bound of check 1 on at least one of
# the systems `conditioned_systems(1000n + 1, T, Nothing, n)`, or returns `info ≠ 0`.

using GeometricSolvers: QRFactorization, LUFactorization
using Metal

Metal.functional() || (println("Metal is not functional; skipping the sweep."); exit(3))

include(joinpath(@__DIR__, "..", "..", "..", "test", "helpers", "matrix.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "test", "helpers", "staticsystems.jl"))

println("Julia ", VERSION, ", check-bounds ",
    Base.JLOptions().check_bounds, ", ", Metal.device())
for M in (QRFactorization, LUFactorization), T in (Float32, ComplexF32)

    failing = filter(1:12) do n
        systems = conditioned_systems(1000n + 1, T, Nothing, n)
        x, info = kernel_solve(MtlArray, M(), first.(systems), last.(systems))
        any(eachindex(systems)) do k
            A, b = systems[k]
            info[k] != 0 || !(relerr(x[k], reference(A, b)) <= bound(A, T, Nothing))
        end
    end
    println(M, " ", T, ": failing n = ", failing)
end
