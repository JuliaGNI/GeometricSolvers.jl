# Checks `cuda_well_conditioned_cond` of test/helpers/own_kernel.jl against the dense
# `cond(A, Inf)` on the block-diagonal matrices that `cuda_well_conditioned` returns above
# n = 256, for the four BLAS element types at n = 300 and n = 1000. The two agree within
# `8 * eps(real(T))` relative. Two mutants of the closed form, each without one corner block,
# must disagree with `cond(A, Inf)` at each size and type where that block sets a norm; a mutant
# that no case separates is reported. A matrix with one entry outside the blocks must throw.
# Run with `julia --startup-file=no --project=test scripts/well_conditioned_cond.jl`.
using LinearAlgebra: cond, opnorm
using Printf

include(joinpath(@__DIR__, "..", "test", "helpers", "own_kernel.jl"))

# The closed form with the corner block `drop` (1 or 2) left out.
function cond_without(A, drop)
    n = size(A, 1)
    blocks = [A[1:16, 1:16], A[(n - 15):n, (n - 15):n]]
    deleteat!(blocks, drop)
    d = abs.([A[i, i] for i in 17:(n - 16)])
    norm_A = max(maximum(B -> opnorm(B, Inf), blocks), maximum(d))
    norm_inv = max(maximum(B -> opnorm(inv(B), Inf), blocks), inv(minimum(d)))
    return norm_A * norm_inv
end

failures = 0
caught = Dict(1 => 0, 2 => 0)
for T in (Float32, Float64, ComplexF32, ComplexF64), n in (300, 1000)

    A, _ = cuda_well_conditioned(T, n)
    κ_ref = cond(A, Inf)
    κ = cuda_well_conditioned_cond(A)
    tol = 8 * eps(real(T))
    rel = abs(κ - κ_ref) / κ_ref
    ok = rel <= tol
    global failures += !ok
    mutants = map((1, 2)) do drop
        rel_m = abs(cond_without(A, drop) - κ_ref) / κ_ref
        rel_m > tol && (caught[drop] += 1)
        rel_m
    end
    @printf("%-10s n = %4d  κ∞ = %.10f  rel = %.2e  tol = %.2e  %s  without B₁: %.2e  without B₂: %.2e\n",
        T, n, κ_ref, rel, tol, ok ? "pass" : "FAIL", mutants...)
end

for drop in (1, 2)
    println("mutant without B", drop, ": caught in ", caught[drop], " of 8 cases",
        caught[drop] == 0 ? " (SURVIVED)" : "")
end

A, _ = cuda_well_conditioned(Float64, 300)
A[20, 200] = 1.0e-3
threw = try
    cuda_well_conditioned_cond(A)
    false
catch err
    err isa ArgumentError && occursin("not block diagonal", err.msg)
end
println("entry outside the blocks: ", threw ? "throws" : "NO THROW")

pass = failures == 0 && all(>(0), values(caught)) && threw
println(pass ? "all checks pass" : "CHECKS FAIL")
exit(pass ? 0 : 1)
