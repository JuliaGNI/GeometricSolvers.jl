# The six checks of the reduced-space nonlinear elimination of `BatchEliminated`, on the
# one-dimensional implicit electrostatic PIC problem of `Chen2011`.
#
# Run from the root of the GeometricSolvers tree:
#
#     julia --project=. scripts/verify_nonlinear_elimination.jl
#
# Every check runs on the problem that `make_pic_elimination_problem` builds, and prints the number
# it reads. The script exits with status 1 when a check fails.
using GeometricSolvers, LinearAlgebra
using Printf
using StaticArrays: SVector

const GS = GeometricSolvers

failures = String[]
function check(condition::Bool, message::AbstractString)
    condition || push!(failures, String(message))
    return nothing
end

r5() = Newton(; linsolve = GMRES(; maxiter = 16), globalization = Backtracking())

println("GeometricSolvers ", pkgversion(GeometricSolvers),
    " · reduced-space elimination with an R2 batch inside an R5 outer Newton")
println()

# ─── The problem: one Crank–Nicolson Vlasov–Ampère step, 16 nodes, 64 particles ───
outer, E0 = GS.make_pic_elimination_problem(Float64, Float64)
println(@sprintf("problem: 16 field nodes, 64 particles, Δt = 0.1, ‖E₀‖ = %.3e, batch = Float64",
    norm(E0)))
println()

# ─── Check 1: the inner tolerance is quadratic in the previous outer residual ───
reset_batch!(outer)
_, status1 = solve(outer, copy(E0), r5(); f_abstol = 1e-8, f_reltol = 0.0)
pairs = collect(zip(outer.g₁_history, outer.g₂_prev_history))
println(@sprintf("Check 1 · %d outer evaluations, status %s after %d steps, ‖F‖ = %.3e",
    length(pairs), status1.code, status1.iterations, status1.fnorm))
for (k, (g1, g2)) in enumerate(pairs)
    println(@sprintf("  %2d: ‖g₁‖ = %.3e   ‖g₂(prev)‖ = %.3e   ‖g₂(prev)‖² = %.3e",
        k, g1, g2, g2^2))
end
check(status1.code == SUCCESS, "Check 1: the outer solve did not converge")
check(!isempty(pairs), "Check 1: no outer evaluation was recorded")
# The floor of the schedule binds below `‖g₂(prev)‖² = 4 eps(R_in) √N_p`, where no arithmetic in the
# batch's type reaches the quadratic; there the clause is the coarse ceiling instead.
bound = 4 * eps(Float64) * sqrt(length(outer.Y))
ceiling = 32 * eps(Float64) * sqrt(length(outer.Y))
above = filter(p -> p[2]^2 > bound, pairs)
below = filter(p -> !(p[2]^2 > bound), pairs)
println(@sprintf("          %d evaluations above the floor, %d below it",
    length(above), length(below)))
check(all(p -> p[1] < p[2]^2, above),
    "Check 1: an evaluation above the floor does not satisfy ‖g₁‖ < ‖g₂(prev)‖²")
check(all(p -> p[1] <= ceiling, below),
    "Check 1: an evaluation below the floor exceeds 32 eps(R_in) √N_p")
println()

# ─── Check 2: the reduction reports a failure, with the right count and norm ───
inner = NonlinearProblem((y, c) -> SVector(y[1]^2 - c[1], y[2]^2 - c[2]))
Y_batch = [SVector(1.2, 1.8) for _ in 1:16]
params = [i == 7 ? SVector(-1.0, 3.0) : SVector(2.0, 3.0) for i in 1:16]
statuses = similar(Y_batch, SolverStatus{Float64})
solve_batch!(Y_batch, statuses, inner, Newton(), PerSystem(params),
    Options(Float64; max_iterations = 8))
summary = batch_summary(statuses)
host = norm(reduce(vcat, [Vector(inner.F(Y_batch[i], params[i])) for i in 1:16]))
println(@sprintf("Check 2 · all_converged = %s, failure_count = %d",
    summary.all_converged, summary.failure_count))
println(@sprintf("          ‖g₁‖ = %.6e, host norm = %.6e, relative gap %.2e",
    summary.fnorm, host, abs(summary.fnorm - host) / host))
check(!summary.all_converged, "Check 2: a failing system did not clear all_converged")
check(summary.failure_count == 1, "Check 2: the failure count is not 1")
check(isapprox(summary.fnorm, host; rtol = 32 * eps(Float64)),
    "Check 2: the global norm disagrees with the host norm")
println()

# ─── Check 3: the deposit is deterministic over 100 evaluations at one x ───
F = similar(E0)
reset_batch!(outer)
GS.residual!!(F, outer, E0, nothing)
m0 = deposited_moments(outer)
reference = (copy(m0.current), copy(m0.density), copy(m0.flux), copy(m0.temperature))
function deposits_equal(prob, x, F, reference)
    for _ in 1:99
        reset_batch!(prob)
        GS.residual!!(F, prob, x, nothing)
        m = deposited_moments(prob)
        m.current == reference[1] && m.density == reference[2] &&
        m.flux == reference[3] && m.temperature == reference[4] || return false
    end
    return true
end
deterministic = deposits_equal(outer, E0, F, reference)
println(@sprintf("Check 3 · 100 deposits equal: %s", deterministic))
println(@sprintf("          max|current| = %.3e, max|density| = %.3e",
    maximum(abs, reference[1]), maximum(abs, reference[2])))
println(@sprintf("          max|flux| = %.3e, max|temperature| = %.3e",
    maximum(abs, reference[3]), maximum(abs, reference[4])))
check(deterministic, "Check 3: the deposited moments are not bit-for-bit equal")
check(!all(iszero, reference[4]), "Check 3: the temperature was not deposited")
println()

# ─── Check 4: a repeat evaluation at the same (x, tolerance) takes 0 inner steps ───
reset_batch!(outer)
GS.residual!!(F, outer, E0, nothing)
first_count = batch_summary(outer).total_iterations
GS.residual!!(F, outer, E0, nothing; repeat_same_tol = true)
repeat_summary = batch_summary(outer)
println(@sprintf("Check 4 · first evaluation: %d inner iterations in total; repeat: max %d, total %d",
    first_count, repeat_summary.max_iterations, repeat_summary.total_iterations))
check(first_count > 0, "Check 4: the first evaluation took no inner iteration")
check(repeat_summary.total_iterations == 0,
    "Check 4: the repeat evaluation took inner iterations")
check(all(st -> st.iterations == 0, outer.statuses),
    "Check 4: a particle took an inner iteration on the repeat")
println()

# ─── Check 5: the eliminated index set is the same at every outer evaluation ───
reset_batch!(outer)
solve(outer, copy(E0), r5(); f_abstol = 1e-8, f_reltol = 0.0)
fixed = eliminated_indices(outer)
same = !isempty(outer.eliminated_index_log) && all(==(fixed), outer.eliminated_index_log)
println(@sprintf("Check 5 · %d evaluations, first index %d, last index %d, equal everywhere: %s",
    length(outer.eliminated_index_log), first(fixed), last(fixed), same))
check(same, "Check 5: the eliminated index set changes over the solve")
println()

# ─── Check 6: a Float32 batch inside a Float64 outer Newton ───
mixed, E0m = GS.make_pic_elimination_problem(Float32, Float64)
reset_batch!(mixed)
_, loose = solve(mixed, copy(E0m), r5(); f_abstol = 2e-4, f_reltol = 0.0)
reset_batch!(mixed)
_, tight = solve(mixed, copy(E0m), r5(); f_abstol = 1e-10, f_reltol = 0.0,
    max_iterations = 25)
println(@sprintf("Check 6 · 2e-4: %s after %d steps, ‖F‖ = %.3e · 1e-10: %s after %d steps, ‖F‖ = %.3e",
    loose.code, loose.iterations, loose.fnorm, tight.code, tight.iterations, tight.fnorm))
check(loose.code == SUCCESS && loose.fnorm <= 2e-4,
    "Check 6: the solve at 2e-4 did not converge")
check(tight.code == STALLED, "Check 6: the solve at 1e-10 did not stall")
check(tight.code != SINGULAR, "Check 6: the solve at 1e-10 reports a factorisation failure")
println()

if isempty(failures)
    println("all six checks hold")
    exit(0)
end
for message in failures
    println("FALSE: ", message)
end
exit(1)
