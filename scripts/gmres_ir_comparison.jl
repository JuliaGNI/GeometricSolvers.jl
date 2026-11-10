# A deterministic comparison of the SDD GMRES-IR of [`GMRESRefinement`](@ref) with `Float32` LU-IR
# by the Newton iteration count and the linear residual of every step, and the recomputation of both
# from the raw output of the solves. It times nothing, and it names no device: the comparison is of
# counts and residuals, which are the same on every backend.
#
#   julia --project=. scripts/gmres_ir_comparison.jl
#
# `generate_gmres_ir_raw_data` runs both methods on the four conditioning numbers `κ∞` below and
# returns the raw per-solve output: for every step, the true linear residual `‖-F(x) - J d‖` of the
# direction it took and the residual `‖F(x)‖` it reached. `format_gmres_ir_table` renders the raw
# output as a table. `recompute_and_verify_gmres_ir_table` derives the two columns again from the
# raw output alone, compares them with the cells of the table, requires that no cell is empty, and
# that the step count each solve recorded is the number of steps its history holds.

using GeometricSolvers
using LinearAlgebra: Diagonal, mul!, norm

const GS_IR = GeometricSolvers

"the four conditioning numbers of the comparison"
const GMRES_IR_KAPPAS = (1.0e4, 5.0e7, 2.0e8, 5.0e10)

# The DCT-II matrix of `n` points, orthogonal to rounding: the singular vectors of the fixture.
function gmres_ir_basis(n::Integer)
    return [i == 1 ? sqrt(1 / n) : sqrt(2 / n) * cos(pi * (2j - 1) * (i - 1) / (2n))
            for
            j in 1:n, i in 1:n]
end

# The problem of the comparison at `κ`: `F(x) = A x` with `A = Q diag(σ) Qᵀ` of `κ∞ ≈ κ`, its
# analytic Jacobian, and the start `2 Q[:, n]`, along the smallest singular vector of `A`. A
# right-hand side in that direction is the one where the relative residual of a solve in `Float32`
# is of order `κ eps(Float32)` rather than of order `eps(Float32)`, which is the regime that
# separates the two methods: `LU-IR` fails there and refactorises the Jacobian in `Float64`, and
# GMRES-IR either meets its tolerance with the low-precision factors or fails and promotes too. A
# residual the size of `‖F(x₀)‖` along the well-conditioned directions of `A` would leave both
# methods in their successful branch at every `κ`, and compare nothing.
function gmres_ir_fixture(κ::Float64; n::Integer = 8)
    Q = gmres_ir_basis(n)
    σ = [κ^(-(i - 1) / (n - 1)) for i in 1:n]
    A = Q * Diagonal(σ) * Q'
    function F!(r, x, p)
        mul!(r, A, x)
        return r
    end
    function J!(J, x, p)
        copyto!(J, A)
        return J
    end
    return NonlinearProblem(F!, J!), J!, 2.0 .* Q[:, n]
end

# The raw output of one solve: the Newton steps it took, and for each step the true linear residual
# `‖-F(x) - J d‖` of the direction `d` and the residual `‖F‖` it reached. The solve steps by hand,
# because `step!!` overwrites the residual of the iterate it is given.
function gmres_ir_record(prob, J!, x0, m; maxsteps::Integer = 12)
    x = copy(x0)
    state = GS_IR.init_state(m, prob, x, nothing)
    Fx = prob.F(similar(x), x, nothing)
    fnorm = GS_IR.rnorm(Fx)
    state = GS_IR.start!!(state, m, prob, x, Fx, nothing, fnorm)
    linear = Float64[]
    residual = Float64[Float64(fnorm)]
    atol = sqrt(eps(Float64)) * fnorm
    Jm = similar(x, length(x), length(x))
    steps = 0
    while GS_IR.rnorm(Fx) > atol && steps < maxsteps
        J!(Jm, x, nothing)
        Fb = copy(Fx)
        before = copy(x)
        x, Fx, state, _ = GS_IR.step!!(state, m, prob, x, Fx, nothing)
        push!(linear, Float64(GS_IR.rnorm(Fb .+ Jm * (x .- before))))
        push!(residual, Float64(GS_IR.rnorm(Fx)))
        steps += 1
    end
    return (steps = steps, linear = linear, residual = residual)
end

"the two methods of the comparison, at `κ`"
function gmres_ir_methods()
    return (
        ("GMRES-IR",
            Newton(; linsolve = LUFactorization(Float32),
                refinement = GMRESRefinement(Float64; gmres_maxiter = 8, gmres_tol = 1e-8))),
        ("LU-IR", Newton(; linsolve = LUFactorization(Float32), refinement = Refinement()))
    )
end

"""
    generate_gmres_ir_raw_data(; kappas = GMRES_IR_KAPPAS)

The raw output of the comparison: one row per conditioning number and method, with the fields
`κ`, `method`, `steps` (the Newton steps the solve took), `linear` (the true linear residual of
every step) and `residual` (the residual norm after every step, from the start). It is
deterministic: the fixture, the start and both methods are fixed.
"""
function generate_gmres_ir_raw_data(; kappas = GMRES_IR_KAPPAS)
    rows = NamedTuple[]
    for κ in kappas
        prob, J!, x0 = gmres_ir_fixture(Float64(κ))
        for (name, m) in gmres_ir_methods()
            rec = gmres_ir_record(prob, J!, x0, m)
            push!(rows,
                (κ = Float64(κ), method = name, steps = rec.steps, linear = rec.linear,
                    residual = rec.residual))
        end
    end
    return rows
end

# The two columns of one row, as the table prints them.
function gmres_ir_summary(row)
    return (iterations = row.steps,
        linear = isempty(row.linear) ? NaN : last(row.linear))
end

# The two columns of one row, derived again from the raw output alone: the iteration count is the
# number of steps the history holds, not the count the solve recorded, and the linear residual is
# the last of the step residuals.
function gmres_ir_recompute(row)
    return (iterations = length(row.residual) - 1,
        linear = isempty(row.linear) ? NaN : last(row.linear))
end

"""
    format_gmres_ir_table(raw_data)

The Markdown table of the raw output: one row per conditioning number and method, with the Newton
iteration count and the linear residual of the last step.
"""
function format_gmres_ir_table(raw_data)
    lines = String[
        "| κ∞ | method | Newton iterations | linear residual |",
        "|--:|:--|--:|--:|"
    ]
    for row in raw_data
        s = gmres_ir_summary(row)
        push!(lines, "| $(row.κ) | $(row.method) | $(s.iterations) | $(s.linear) |")
    end
    return join(lines, "\n") * "\n"
end

# The cells of the data rows of a table, with the separator cells dropped.
function gmres_ir_table_rows(table_str)
    rows = Vector{Vector{String}}()
    for line in split(table_str, '\n')
        text = strip(line)
        startswith(text, "|") || continue
        cells = [strip(c) for c in split(text, '|')[2:(end - 1)]]
        (isempty(cells) || startswith(cells[1], "κ") ||
         all(c -> occursin(r"^:?-+:?$", c), cells)) && continue
        push!(rows, cells)
    end
    return rows
end

"""
    recompute_and_verify_gmres_ir_table(raw_data, table_str) -> Bool

`true` where the table is the raw output of the solves: every row of the table is a row of
`raw_data`, in order, its Newton iteration count and linear residual are the ones recomputed from
that raw output, the step count each solve recorded is the number of steps its history holds, and
no cell is empty. `false` otherwise, which is what a mismatched or truncated table gives.
"""
function recompute_and_verify_gmres_ir_table(raw_data, table_str)
    rows = gmres_ir_table_rows(table_str)
    length(rows) == length(raw_data) || return false
    for (cells, row) in zip(rows, raw_data)
        length(cells) == 4 || return false
        any(isempty, cells) && return false
        cells[1] == string(row.κ) || return false
        cells[2] == row.method || return false
        row.steps == length(row.residual) - 1 || return false
        s = gmres_ir_recompute(row)
        cells[3] == string(s.iterations) || return false
        (tryparse(Float64, cells[4]) === s.linear) || return false
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    raw = generate_gmres_ir_raw_data()
    table = format_gmres_ir_table(raw)
    print(table)
    ok = recompute_and_verify_gmres_ir_table(raw, table)
    println(ok ? "recomputation holds" : "recomputation fails")
    ok || exit(1)
end
