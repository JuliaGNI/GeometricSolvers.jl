# The measurement behind the constant `c` of the breakdown test of `GMRES` on a reduced basis,
# `hj < c eps(TB) ‖w₀‖`: the ratio `hj / (eps(TB) ‖w₀‖)` at every step where the test acts, on
# the reduced-basis systems of `test/nonlinear/krylov.jl` and on the Float32-basis column of the
# standard test set, with each step classified as invariant or not by an Arnoldi process in
# Float64 from the same stored `v₁`.
#
# The test acts where the cycle would go on: the Givens estimate misses the tolerance,
# `j < restart` and `used < maxiter`. The script replaces `basis_breakdown` by a method that
# records the ratio and returns `false`, so every trajectory is that of a solve without the test.
# A step is invariant where the Float64 Arnoldi remainder `h` of step `j` from the stored `v₁`
# is at most `1e-13 ‖J‖₂`, about 450 eps(Float64) of the operator; the script prints the largest
# remainder of an invariant step and the smallest of a non-invariant one, so the gap shows. A
# step after an invariant one, which only a cycle without the test reaches, is left out.
#
# Run from the package root, with the test environment:
#   julia --startup-file=no --project=. -e 'using TestEnv; TestEnv.activate(); include("scripts/gmres_breakdown_ratio.jl")'

using GeometricSolvers
using GeometricSolvers: FixedForcing, GMRES, Newton, NonlinearProblem, gmres_solve!,
                        init_gmres_workspace, rnorm, solve
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, dot, mul!, norm, opnorm
using Printf: @printf

module Standard
include(joinpath(@__DIR__, "..", "test", "helpers", "standard_cells.jl"))
end

const RECORD = Any[]
const INVARIANT = 1e-13
# the Jacobian at the current step, as a dense Float64 matrix
const JACOBIAN = Ref{Any}(nothing)

# The relative Float64 remainder `h / ‖J‖₂` of step `j` of an Arnoldi process with two MGS
# passes on `J`, from the stored first column of the basis; `Inf` where an earlier step is
# already invariant, a step that only a cycle without the test reaches.
function float64_remainder(J, v₁, j)
    V = zeros(promote_type(eltype(J), eltype(v₁)), size(J, 1), j)
    V[:, 1] = v₁ ./ norm(v₁)
    scale = opnorm(J)
    h = 0.0
    for k in 1:j
        w = J * V[:, k]
        for _ in 1:2, i in 1:k

            w .-= dot(V[:, i], w) .* V[:, i]
        end
        h = norm(w)
        k < j && h <= INVARIANT * scale && return Inf
        k < j && (V[:, k + 1] = w ./ h)
    end
    return h / scale
end

# The test is called after the rotation of step `j`, which sets `g[j + 1]`; a zero `g[j + 1]`
# leaves a zero `hj`, which no rule of the test reads.
@eval GeometricSolvers @inline function basis_breakdown(ws::GMRESWorkspace{T, R, V}, hj,
        w₀) where {T, R, V}
    j = something(findlast(!iszero, ws.g), 1) - 1
    J = Main.JACOBIAN[]()
    v₁ = ComplexF64.(ws.V[:, 1])
    rel = iszero(hj) ? 0.0 :
          Main.float64_remainder(isreal(J) ? real(J) : J, isreal(v₁) ? real(v₁) : v₁, j)
    push!(Main.RECORD,
        (ratio = Float64(hj / (R(eps(real(eltype(V)))) * w₀)), j = j, rel = rel))
    return false
end

# One record set: the label, whether the system is built invariant, and the steps it recorded.
function measure(f, label, kind)
    empty!(RECORD)
    f()
    return (label = label, kind = kind, steps = copy(RECORD))
end

function linear_solve(A, b, TB, tol, maxiter, restart; mgs)
    T = eltype(b)
    JACOBIAN[] = () -> ComplexF64.(Matrix(A))
    op = (Jv, v) -> (mul!(Jv, A, v); Jv)
    ws = init_gmres_workspace(GMRES(; maxiter, restart, basis_eltype = TB), zeros(T, length(b)))
    gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(maxiter), Int32(restart); mgs)
    return nothing
end

# A Newton solve whose residual records its last primal iterate, the iterate of the next step.
function newton_solve(F!, x₀, TB, η, maxiter; mgs)
    last = Ref(copy(x₀))
    G! = (r, x, q) -> (eltype(x) <: AbstractFloat && (last[] = copy(x)); F!(r, x, q))
    JACOBIAN[] = () -> ComplexF64.(ForwardDiff.jacobian((r, x) -> F!(r, x, nothing),
        similar(last[], Float64), Float64.(last[])))
    m = Newton(;
        linsolve = GMRES(; maxiter, forcing = FixedForcing(η), basis_eltype = TB, mgs))
    solve(NonlinearProblem(G!), copy(x₀), m)
    return nothing
end

const PAIRS = ((Float64, Float16), (Float64, Float32), (Float32, Float16))
tolerance(T) = T === Float64 ? 1e-10 : 1.0f-5

sets = Any[]
for (T, TB) in PAIRS, mgs in (false, true)

    tag = "mgs = $mgs, $TB on $T"
    n = 100
    push!(sets, measure("J = 14 I, constant b, $tag", :along_basis) do
        linear_solve(Diagonal(fill(T(14), n)), fill(T(-9), n), TB, tolerance(T), 20, 20; mgs)
    end)
    push!(sets,
        measure("three eigenvalues, $tag", :direction_error) do
            linear_solve(Diagonal(T.(repeat([2, 5, 11], 34)[1:n])),
                T.(collect(range(-1, 1; length = n))), TB, tolerance(T), 20, 20; mgs)
        end)
    u₁ = [fill(T(0.2), 25); zeros(T, 25)]
    u₂ = [zeros(T, 25); fill(T(0.2), 25)]
    push!(sets, measure("singular J = u₂ (u₁ + u₂)ᵀ, $tag", :along_basis) do
        linear_solve(u₂ * (u₁ .+ u₂)', copy(u₁), TB, tolerance(T), 20, 20; mgs)
    end)
    cubic!(r, x, q) = (@. r = x^3 + 2x - 3; r)
    push!(sets, measure("x³ + 2x − 3 from a constant start, $tag", :along_basis) do
        newton_solve(cubic!, fill(T(2), 8), TB, tolerance(T), 10; mgs)
    end)
    push!(sets, measure("cluster diag(1, 1 + eps(TB), 2, 3), $tag", :not_invariant) do
        linear_solve(Diagonal(T[1, 1 + eps(TB), 2, 3]), ones(T, 4), TB, zero(T), 4, 4; mgs)
    end)
    push!(sets,
        measure("diagonal κ = 1e5, n = 40, $tag", :not_invariant) do
            linear_solve(
                Diagonal(T.(collect(range(1, 1e5; length = 40)))), ones(T, 40), TB,
                zero(T), 30, 30; mgs)
        end)
end
const FG_A6 = [4.0 1.0 0.5 0.2 0.0 0.1
               0.3 3.0 0.8 0.1 0.2 0.0
               0.2 0.4 2.5 0.6 0.0 0.3
               0.1 0.2 0.3 2.0 0.5 0.0
               0.0 0.1 0.2 0.3 1.5 0.4
               0.2 0.0 0.1 0.2 0.3 1.2]
const FG_B6 = [1.0, -2.0, 1.5, -0.5, 0.7, -1.1]
const FG_DIAGONAL_X = collect(range(1.5, 3.0; length = 8))
r5!(r, x, q) = (@. r = x^3 + 2x - 3; r)
for mgs in (false, true)
    for T in (ComplexF32, ComplexF64)
        TB = real(T) === Float64 ? Float32 : Float16
        A = T.(FG_A6) .+ T(0.3im) .* T.(FG_A6')
        push!(sets,
            measure("complex FG_A6, mgs = $mgs, $TB on $T", :not_invariant) do
                linear_solve(A, T.(FG_B6) .- T(0.5im), TB,
                    real(T) === Float64 ? 1e-12 : 1.0f-5, 40, 40; mgs)
            end)
    end
    jd = 3 .* FG_DIAGONAL_X .^ 2 .+ 2
    b = .-(FG_DIAGONAL_X .^ 3 .+ 2 .* FG_DIAGONAL_X .- 3)
    for tol in (1e-8, 1e-12)
        push!(sets, measure("R5Diagonal at the start, tol = $tol, mgs = $mgs", :not_invariant) do
            linear_solve(Diagonal(jd), b, Float32, tol, 30, 30; mgs)
        end)
    end
    for restart in (2, 4), tol in (1e-13, 1e-14)

        push!(sets,
            measure("R5Diagonal restarted, restart = $restart, tol = $tol, mgs = $mgs",
                :not_invariant) do
                linear_solve(Diagonal(jd), b, Float32, tol, 60, restart; mgs)
            end)
    end
    push!(sets, measure("Newton on R5Diagonal, mgs = $mgs", :not_invariant) do
        newton_solve(r5!, copy(FG_DIAGONAL_X), Float32, 1e-10, 20; mgs)
    end)
end
column = only(filter(c -> c[1] == "Newton Backtracking Float32-basis GMRES",
    Standard.STANDARD_COLUMNS))
for sp in Standard.STANDARD_PROBLEMS, (start, x₀) in Standard.standard_starts(sp, Float64)

    last = Ref{Any}((copy(x₀), nothing))
    F = sp.prob.F
    G = (r, x, q) -> (eltype(x) <: AbstractFloat && (last[] = (copy(x), q)); F(r, x, q))
    JACOBIAN[] = () -> ComplexF64.(ForwardDiff.jacobian((r, x) -> F(r, x, last[][2]),
        similar(last[][1], Float64), Float64.(last[][1])))
    push!(sets, measure("standard problem $(sp.number), $start", :standard) do
        Standard.run_cell(sp, start, x₀, column[1], column[2]; F = G)
    end)
end

all_steps = [(s.label, s.kind, st) for s in sets for st in s.steps]
beyond = filter(p -> p[3].rel == Inf, all_steps)
counted = filter(p -> p[3].rel != Inf, all_steps)
steps(pred) = [(p[1], p[3]) for p in counted if pred(p[2])]
isinvariant(p) = p[2].rel <= INVARIANT
along = filter(isinvariant, steps(k -> k === :along_basis))
direction = filter(isinvariant, steps(k -> k === :direction_error))
invariant_other = filter(isinvariant, steps(k -> k in (:not_invariant, :standard)))
noninvariant = filter(!isinvariant, steps(k -> true))
built_noninvariant = filter(p -> p[2].rel <= INVARIANT, steps(k -> k === :not_invariant))
maxratio(ps) = isempty(ps) ? NaN : maximum(p -> p[2].ratio, ps)
minpair(ps) = ps[argmin([p[2].ratio for p in ps])]

@printf("steps where the test acts: %d, of which %d follow an invariant step and are left out\n",
    length(all_steps), length(beyond))
@printf("invariant, remainder along the stored basis: %d steps, largest ratio %.3g\n",
    length(along), maxratio(along))
@printf("invariant, with a direction error of the stored basis: %d steps, ratios %.3g to %.3g\n",
    length(direction), minimum(p -> p[2].ratio, direction), maxratio(direction))
@printf("invariant steps of the other systems: %d, largest ratio %.3g\n",
    length(invariant_other), maxratio(invariant_other))
@printf("steps of the systems built non-invariant that classify invariant: %d\n",
    length(built_noninvariant))
m = minpair(noninvariant)
@printf("non-invariant: %d steps, smallest ratio %.3g (%s, j = %d, Float64 remainder %.2g ‖J‖)\n",
    length(noninvariant), m[2].ratio, m[1], m[2].j, m[2].rel)
@printf("non-invariant steps below 1: %d; below 10: %d\n",
    count(p -> p[2].ratio < 1, noninvariant), count(p -> p[2].ratio < 10, noninvariant))
@printf("largest Float64 remainder of an invariant step: %.2g; smallest of a non-invariant one: %.2g\n",
    maximum(p -> p[2].rel, vcat(along, direction, invariant_other)),
    minimum(p -> p[2].rel, noninvariant))
@printf("Julia %s, check-bounds %d\n", VERSION, Base.JLOptions().check_bounds)
