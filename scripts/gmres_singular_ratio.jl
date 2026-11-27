# The measurement behind the threshold of the singular decision of `GMRES` on a reduced basis: at a
# breakdown step whose rotated diagonal entry is at the round-off of the basis, the solve stops
# where the entry `g[j]` of the rotated right-hand side exceeds `sqrt(eps(TB)) β`, `β` the residual
# at the start of the cycle, and restarts below it. The script records `|g[j]| / β` at each such
# step, with the decision, on two families in the rotated frame of the tests:
#
# - singular: `J = u₂ (u₁ + u₂)ᵀ`, `b = u₁`, for the 0.2 vectors of `test/nonlinear/krylov.jl`
#   unrotated and in the frame of `Xoshiro(7)`, and for unit vectors in the frames of five seeds
#   at `n = 20, 50, 101`; `maxiter = restart = 20`;
# - consistent: `J = u₂ (u₁ + u₂)ᵀ + δ u₁ u₁ᵀ`, nonsingular on `span(u₁, u₂)`, `b = u₁`, in the
#   frame of `Xoshiro(7)`, over a grid of `δ`; `restart = 20`, `maxiter = 40` and `200`.
#
# The method of `singular_breakdown` is replaced by one that records and decides as the package
# does, so every trajectory is that of the package. Run from the package root:
#   julia --startup-file=no --project=. scripts/gmres_singular_ratio.jl

using GeometricSolvers
using GeometricSolvers: GMRES, GMRESWorkspace, basis_breakdown, gmres_solve!,
                        init_gmres_workspace, rnorm
using LinearAlgebra: qr
using Printf: @printf
using Random: Xoshiro

const RECORD = Tuple{Float64, Bool}[]
@eval GeometricSolvers @inline function singular_breakdown(
        ws::GMRESWorkspace{T, R, V}, a, w₀,
        gj, β) where {T, R, V}
    singular = basis_breakdown(ws, a, w₀) && gj > sqrt(R(eps(real(eltype(V))))) * β
    basis_breakdown(ws, a, w₀) && push!(Main.RECORD, (Float64(gj / β), singular))
    return singular
end

function frame(T, seed, n)
    Q = T.(Matrix(qr(randn(Xoshiro(seed), n, n)).Q))
    h = n ÷ 2
    return Q * [fill(T(1 / sqrt(h)), h); zeros(T, n - h)],
    Q * [zeros(T, h); fill(T(1 / sqrt(n - h)), n - h)]
end
function frame02(T)
    Q = T.(Matrix(qr(randn(Xoshiro(7), 50, 50)).Q))
    return Q * [fill(T(0.2), 25); zeros(T, 25)], Q * [zeros(T, 25); fill(T(0.2), 25)]
end

function solve_recorded(T, TB, mgs, u₁, u₂, δ, tol, maxiter)
    empty!(RECORD)
    s = u₁ .+ u₂
    op = (Jv, v) -> (Jv .= u₂ .* sum(s .* v) .+ T(δ) .* u₁ .* sum(u₁ .* v); Jv)
    b = copy(u₁)
    ws = init_gmres_workspace(GMRES(; maxiter, basis_eltype = TB), zeros(T, length(b)))
    _, info = gmres_solve!(
        ws, op, nothing, b, rnorm(b), tol, Int32(maxiter), Int32(20); mgs)
    return info, copy(RECORD)
end

const PAIRS = ((Float64, Float16), (Float64, Float32), (Float32, Float16))

singular_ratios = Dict{Any, Vector{Float64}}()
stopped = 0
rows = 0
for (T, TB) in PAIRS, mgs in (false, true)

    fixtures = [
        frame02(T), ([fill(T(0.2), 25); zeros(T, 25)], [zeros(T, 25); fill(T(0.2), 25)])]
    for seed in (1, 2, 3, 7, 11), n in (20, 50, 101)

        push!(fixtures, frame(T, seed, n))
    end
    tol = T === Float64 ? 1e-10 : 1.0f-5
    for (u₁, u₂) in fixtures
        info, record = solve_recorded(T, TB, mgs, u₁, u₂, 0, tol, 20)
        global rows += 1
        global stopped += info.breakdown && !info.converged && info.iterations < 20
        append!(get!(singular_ratios, TB, Float64[]), first.(record))
    end
end
@printf("singular: %d solves, %d stop with a breakdown before maxiter\n", rows, stopped)
for TB in (Float16, Float32)
    @printf("singular, %s basis: smallest |g[j]| / β %.3g, threshold sqrt(eps) %.3g\n", TB,
        minimum(singular_ratios[TB]), sqrt(eps(TB)))
end

restarted = Dict{Any, Vector{Float64}}()
read_singular = Tuple{String, Float64}[]
for (T, TB) in PAIRS, mgs in (false, true),
    δ in (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-7),
    maxiter in (40, 200)
    u₁, u₂ = frame02(T)
    tol = T === Float64 ? 1e-8 : 1.0f-4
    _, record = solve_recorded(T, TB, mgs, u₁, u₂, δ, tol, maxiter)
    append!(get!(restarted, TB, Float64[]), [g for (g, s) in record if !s])
    for (g, s) in record
        s && push!(read_singular, ("mgs = $mgs, $TB on $T, δ = $δ, maxiter = $maxiter", g))
    end
end
for TB in (Float16, Float32)
    @printf("consistent, %s basis: %d restarts, largest |g[j]| / β %.3g\n", TB,
        length(restarted[TB]), maximum(restarted[TB]; init = 0.0))
end
@printf("consistent steps read as singular: %d, |g[j]| / β from %.3g to %.3g, at\n",
    length(read_singular), minimum(last, read_singular), maximum(last, read_singular))
foreach(p -> @printf("  %s: %.3g\n", p...), read_singular)
@printf("Julia %s, check-bounds %d\n", VERSION, Base.JLOptions().check_bounds)
