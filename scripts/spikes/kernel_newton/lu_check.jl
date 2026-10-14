# lu_check.jl — the pivoting witness for the two LU solves of solvers.jl.
#
# The test problem of problem.jl is diagonally dominant, so no row of the grid pivots. This script
# factors random matrices that do pivot with both storage variants, checks that the two variants
# choose the same pivots, and checks each solution against `\` to 16 n eps(T) κ∞(A) ‖x‖∞. It
# counts the row swaps, so it cannot pass on matrices that never pivot.
#
# Usage, from the repository root:
#     julia --startup-file=no --project=scripts/spikes/kernel_newton scripts/spikes/kernel_newton/lu_check.jl

using LinearAlgebra
using Random
using StaticArrays

include(joinpath(@__DIR__, "solvers.jl"))

function lu_check(::Type{T}, n::Int, rng) where {T}
    A = SMatrix{n, n, T}(randn(rng, T, n, n))
    b = SVector{n, T}(randn(rng, T, n))
    J = MMatrix{n, n, T}(A)
    piv_M, info_M = lu_!(J)
    x_M = lu_solve!(J, piv_M, b)
    J_S, piv_S, info_S = lu_S(A)
    x_S = lu_solve_S(J_S, piv_S, b)
    x = Matrix(A) \ Vector(b)
    tol = 16 * n * eps(T) * cond(Matrix(A), Inf) * maxabs(x)
    ok = info_M == 0 && info_S == 0 && piv_M == piv_S &&
         maxabs(x_M - x) <= tol && maxabs(x_S - x) <= tol
    return ok, count(k -> piv_M[k] != k, 1:n)
end

function main()
    rng = Xoshiro(0x5eed)
    failures = String[]
    systems = 0
    swaps = 0
    for T in (Float32, Float64), n in (2, 4, 6, 8, 12, 16, 24), _ in 1:100
        ok, s = lu_check(T, n, rng)
        systems += 1
        swaps += s
        ok || push!(failures, "$T n=$n")
    end
    swaps > 0 || error("no row swap in $systems systems: the witness did not pivot")
    isempty(failures) ||
        error("the LU solves disagree with `\\`: " * join(unique(failures), ", "))
    println("lu_check: $systems systems, $swaps row swaps; both LU variants agree with `\\`")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
