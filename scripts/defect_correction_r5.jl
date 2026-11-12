# A deterministic comparison of three variants of the matrix-free Newton–GMRES regime (R5) on one
# nonlinear reaction–diffusion system: the Jacobian–vector product in `Float64`, the product in
# `Float32` with the residual in `Float64` (the defect correction of Chen, Chacón and Barnes), and a
# `Float64` outer solve with a right preconditioner applied in `Float32`. It times nothing, and it
# runs on `Array` only: the comparison is of counts and errors.
#
#   julia --startup-file=no --project=. scripts/defect_correction_r5.jl
#
# `generate_defect_correction_raw_data` solves a `Float64` reference root, then each variant, and
# returns the raw per-variant output: the return code, the Newton iteration count, the count of
# Jacobian–vector products, the final iterate and the reference root.
# `format_defect_correction_table` renders it as a table.
# `recompute_and_verify_defect_correction_table` derives every cell again from the raw output alone,
# compares it with the cell of the table, requires that no cell is empty, and that the second
# variant alone forms its product in `Float32`.
#
# The forward error of the `Float64 JVP` row is `0.0` at `n = 256`, and it is not a measured
# agreement: the reference is solved with the same method from the same start, so it runs through
# the same Newton iterates, and at this `n` the step that meets `f_reltol = 1e-10` meets `1e-12`
# too, so both stop at the same iterate. At an `n` where the reference takes more steps, such as
# `n = 64`, the error is not `0.0`.

using GeometricSolvers
using LinearAlgebra: Tridiagonal, ldiv!, lu, norm

"the coefficient `λ` of the reaction term `λ u³`"
const DEFECT_CORRECTION_LAMBDA = 10

# `F(u)_i = (2u_i - u_{i-1} - u_{i+1}) / h² + λ u_i³ - 1` on `n` interior points of (0, 1), with
# `h = 1 / (n + 1)` and `u_0 = u_{n+1} = 0`, in the element type of `u`.
# `A = tridiag(-1, 2, -1) / h²` is SPD and `u³` is increasing, so the root is unique and the
# Jacobian `A + 3λ diag(u²)` is SPD at every iterate.
function defect_correction_residual!(r, u, n::Integer)
    T = eltype(u)
    h = inv(T(n + 1))
    λ = T(DEFECT_CORRECTION_LAMBDA)
    for i in 1:n
        left = i == 1 ? zero(T) : u[i - 1]
        right = i == n ? zero(T) : u[i + 1]
        r[i] = (2u[i] - left - right) / h^2 + λ * u[i]^3 - one(T)
    end
    return r
end

"the `NonlinearProblem` of the comparison on `n` interior points"
function defect_correction_problem(n::Integer)
    return NonlinearProblem((r, u, p) -> defect_correction_residual!(r, u, n))
end

# `‖F(x)‖₂` in `Float64`, from the residual evaluated at `x`.
function defect_correction_fnorm(x)
    return Float64(norm(defect_correction_residual!(similar(x), x, length(x))))
end

# A right preconditioner in `Float32`: `refresh!` factorises the analytic tridiagonal Jacobian
# `A + 3λ diag(u²)` at the iterate in `Float32`, and `apply_preconditioner!` solves with that factor
# in `Float32` and writes the result into the `Float64` output.
mutable struct DefectCorrectionPreconditioner{F}
    fac::F
    v::Vector{Float32}
end

function DefectCorrectionPreconditioner(n::Integer)
    return DefectCorrectionPreconditioner(defect_correction_jacobian32(zeros(n)), zeros(Float32, n))
end

# The `Float32` LU factor of the analytic Jacobian at `x`.
function defect_correction_jacobian32(x)
    n = length(x)
    h = inv(Float32(n + 1))
    λ = Float32(DEFECT_CORRECTION_LAMBDA)
    off = fill(-1 / h^2, n - 1)
    diag = Float32[2 / h^2 + 3λ * Float32(x[i])^2 for i in 1:n]
    return lu(Tridiagonal(off, diag, copy(off)))
end

function GeometricSolvers.refresh!(P::DefectCorrectionPreconditioner, prob, x, Fx, p)
    P.fac = defect_correction_jacobian32(x)
    return nothing
end

function GeometricSolvers.apply_preconditioner!(z, P::DefectCorrectionPreconditioner, v)
    P.v .= v
    ldiv!(P.fac, P.v)
    z .= P.v
    return z
end

"the three variants of the comparison on `n` interior points, as `(name, method)` pairs"
function defect_correction_methods(n::Integer)
    return [
        ("Float64 JVP", Newton(; linsolve = GMRES(; maxiter = n))),
        ("Float32 JVP", Newton(; linsolve = GMRES(; maxiter = n, jvp_eltype = Float32))),
        ("Float32 preconditioner",
            Newton(;
                linsolve = GMRES(;
                maxiter = n, preconditioner = DefectCorrectionPreconditioner(n))))
    ]
end

"""
    generate_defect_correction_raw_data(; n = 256)

The raw output of the comparison. It solves the `Float64` reference root with the first variant's
method to `‖F‖ ≤ 1e-12 ‖F(u₀)‖` and throws an `ErrorException` where it does not reach that. It then
solves each variant from `u₀ = zeros(n)` with `f_reltol = 1e-10`, `f_abstol = 0` and
`max_iterations = 100`, and returns one row per variant with the fields `variant`, `code` and
`iterations` (from the returned status), `jvp_calls` (from `jvp_calls(solver)`), `x` (the returned
iterate, converged or not) and `xref` (the reference root).
"""
function generate_defect_correction_raw_data(; n = 256)
    prob = defect_correction_problem(n)
    x₀ = zeros(n)
    methods = defect_correction_methods(n)
    sref = init(prob, copy(x₀), methods[1][2]; f_reltol = 1e-12, f_abstol = 0.0,
        max_iterations = 100)
    xref, _ = solve!(sref, copy(x₀))
    fref = defect_correction_fnorm(xref)
    f₀ = defect_correction_fnorm(x₀)
    fref <= 1e-12 * f₀ ||
        error("the Float64 reference reaches ‖F‖ = $fref, above 1e-12 ‖F(u₀)‖ = $(1e-12 * f₀)")
    xref = copy(xref)
    return map(methods) do (name, m)
        solver = init(
            prob, copy(x₀), m; f_reltol = 1e-10, f_abstol = 0.0, max_iterations = 100)
        x, status = solve!(solver, copy(x₀))
        (; variant = name, code = status.code, iterations = status.iterations,
            jvp_calls = jvp_calls(solver), x = copy(x), xref = xref)
    end
end

"""
    format_defect_correction_table(raw)

The Markdown table of the raw output: one row per variant, with its return code, the Newton
iteration count, the GMRES iteration count (one Jacobian–vector product per inner iteration), the
final `‖F‖` and the forward error `‖x - xref‖ / ‖xref‖`.
"""
function format_defect_correction_table(raw)
    lines = String[
        "| variant | return code | Newton iterations | GMRES iterations | ‖F‖ | forward error |",
        "|:--|:--|--:|--:|--:|--:|"
    ]
    for row in raw
        fnorm = defect_correction_fnorm(row.x)
        ferr = Float64(norm(row.x - row.xref) / norm(row.xref))
        push!(lines,
            "| $(row.variant) | $(row.code) | $(row.iterations) | $(row.jvp_calls) | $fnorm | $ferr |")
    end
    return join(lines, "\n") * "\n"
end

# The six cells of one raw row, derived again from the raw row alone: `‖F‖` from a fresh evaluation
# of the residual at `x`, and the forward error from `x` and `xref`.
function defect_correction_recompute(row)
    x = row.x
    r = defect_correction_residual!(zeros(length(x)), x, length(x))
    return [row.variant, string(row.code), string(row.iterations), string(row.jvp_calls),
        string(Float64(norm(r))), string(Float64(norm(x .- row.xref) / norm(row.xref)))]
end

# The cells of the data rows of a table, with the header and the separator dropped.
function defect_correction_table_rows(table)
    rows = Vector{Vector{String}}()
    for line in split(table, '\n')
        text = strip(line)
        startswith(text, "|") || continue
        cells = [String(strip(c)) for c in split(text, '|')[2:(end - 1)]]
        (isempty(cells) || cells[1] == "variant" ||
         all(c -> occursin(r"^:?-+:?$", c), cells)) && continue
        push!(rows, cells)
    end
    return rows
end

"""
    recompute_and_verify_defect_correction_table(raw, table) -> Bool

`true` where the table is the raw output: it has one row per raw row and three in all, in order,
every cell equals the one recomputed from its raw row alone (`‖F‖` from the residual at `x`, the
forward error from `x` and `xref`), no cell is empty, and the variants' methods carry the
`jvp_eltype` `Nothing`, `Float32` and `Nothing` in that order. `false` otherwise.
"""
function recompute_and_verify_defect_correction_table(raw, table)
    rows = defect_correction_table_rows(table)
    length(rows) == length(raw) == 3 || return false
    methods = defect_correction_methods(length(first(raw).x))
    [m.linsolve.jvp_eltype for (_, m) in methods] == [Nothing, Float32, Nothing] ||
        return false
    for (cells, row, (name, _)) in zip(rows, raw, methods)
        length(cells) == 6 || return false
        any(isempty, cells) && return false
        row.variant == name || return false
        cells == defect_correction_recompute(row) || return false
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    raw = generate_defect_correction_raw_data()
    table = format_defect_correction_table(raw)
    print(table)
    ok = recompute_and_verify_defect_correction_table(raw, table)
    println(ok ? "recomputation holds" : "recomputation fails")
    ok || exit(1)
end
