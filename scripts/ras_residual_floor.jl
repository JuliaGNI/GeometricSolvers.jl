# The margins of the tests that end a positive-tolerance RAS local solve, sweep by sweep, on the
# problem of the RAS doctest of docs/src/algorithms/composition.md: the residual test
# ‖f_i‖ ≤ rtol ‖f_i⁰‖ and the rounding test ‖δ‖ ≤ eps ‖x_i‖ of the last Newton step. A ratio above 1
# fails its test. Where the residual ratio is above 1, the residual target lies below the floor of
# the block residual, and the rounding ratio, a quantity at the noise level, decides alone. The
# sweeps from 6 on have rounding ratios between about 0.25 and 1.05, so a row with both ratios
# above 1 appears or not by the last bits of the arithmetic; such a row passes by the floor test of
# `at_floor` in src/composition/ras.jl, and without it the step returns `MAXITERS`.
# Run with `julia --project=. scripts/ras_residual_floor.jl`.
using GeometricSolvers
using GeometricSolvers: init_state, start!!, _ras_sweep!!, rnorm, SUCCESS
using Printf

const n = 8
cell(u, j) = u[j] * (1 + (j == 1 ? 0.0 : u[j - 1])^2 + (j == n ? 0.0 : u[j + 1])^2) - 1
F!(r, x, p) = (for j in 1:n
        r[j] = cell(x, j)
    end; r)

const part = BlockPartition(n, 2; overlap = 1)
const blocks = ntuple(2) do i
    S = part.subdomains[i]
    function (f, x_loc, x, p)
        u = Vector{promote_type(eltype(x_loc), eltype(x))}(x)
        u[S] .= x_loc
        for (k, j) in enumerate(S)
            f[k] = cell(u, j)
        end
        return f
    end
end
const prob = NonlinearProblem(F!; block_residuals = blocks)

const rtol = 1e-12
m = RAS(part; max_iterations = 20, rtol)
x = zeros(n)
Fx = F!(similar(x), x, nothing)
fnorm₀ = rnorm(Fx)
state = start!!(init_state(m, prob, x, nothing), m, prob, x, Fx, nothing, fnorm₀)
@printf "%5s %5s %10s %22s %22s\n" "sweep" "block" "‖f_i⁰‖" "‖f_i‖/(rtol ‖f_i⁰‖)" "‖δ‖/(eps ‖x_i‖)"
for sweep in 1:20
    f₀ = [rnorm(blocks[i](zeros(length(S)), x[S], x, nothing))
          for (i, S) in enumerate(part.subdomains)]
    code = _ras_sweep!!(state, m, prob, x, nothing)
    b = state.blocks
    for (i, S) in enumerate(part.subdomains)
        f = blocks[i](zeros(length(S)), b.x_locals[i], b.couplings, nothing)
        residual = rnorm(f) / (rtol * f₀[i])
        rounding = rnorm(b.deltas[i]) / (eps() * rnorm(b.x_locals[i]))
        @printf "%5d %5d %10.2e %22.3g %22.3g\n" sweep i f₀[i] residual rounding
    end
    code === SUCCESS || (println("sweep $sweep: ", code); break)
    # the default stopping test of `solve`, `‖F(x)‖ ≤ √eps ‖F(x₀)‖`
    rnorm(F!(Fx, x, nothing)) <= sqrt(eps()) * fnorm₀ && break
end
