# The Broyden-update table of the report: GeometricSolvers alone, on every row of `problems.jl`, in
# `Float32` and `Float64`, with the columns `Broyden` and `Newton reuse 3 Broyden Backtracking LU`.
# It writes one line per cell to `<outdir>/broyden_update.csv`: the fields of `correctness.csv`
# and `chunks`, the number of ForwardDiff chunks of the shared Jacobian of the row. It prints the
# table, with the success count and the Jacobian count `Σ dual_evals / chunks` of each column and
# precision.
#
# Usage: julia --project=benchmark benchmark/broyden_update.jl [outdir]   (default benchmark/results)

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "packages.jl"))

using Printf: @sprintf

const BROYDEN_COLUMNS = ("Broyden", "Newton reuse 3 Broyden Backtracking LU")
const BROYDEN_HEADER = CELLS_HEADER * ",chunks"

# The chunks of `ForwardDiff.jacobian!` in `SharedJacobian`, which takes the default chunk of the
# length of the iterate.
jacobian_chunks(row::Row) = cld(length(row.x0), ForwardDiff.pickchunksize(length(row.x0)))

function main(outdir)
    mkpath(outdir)
    cells = Tuple{Cell, Int}[]
    for T in (Float32, Float64), row in benchmark_rows(T), column in BROYDEN_COLUMNS
        push!(cells, (run_cell(row, column, "GeometricSolvers"), jacobian_chunks(row)))
    end
    open(joinpath(outdir, "broyden_update.csv"), "w") do io
        println(io, BROYDEN_HEADER)
        for (c, k) in cells
            println(io, chomp(sprint(write_cell, c)), ",", k)
        end
    end
    println("| column | Float32 successes | Float32 Jacobian count | Float64 successes | Float64 Jacobian count |")
    println("|:--|--:|--:|--:|--:|")
    for column in BROYDEN_COLUMNS
        entries = map(("Float32", "Float64")) do el
            cs = [(c, k) for (c, k) in cells if c.column == column && cell_eltype(c) == el]
            jac = sum(c.dual_evals / k for (c, k) in cs)
            "$(count(((c, _),) -> c.success, cs))/$(length(cs)) | $(@sprintf("%.1f", jac))"
        end
        println("| $column | ", join(entries, " | "), " |")
    end
    println("\n$(length(cells)) cells")
    return nothing
end

cell_eltype(::Cell{T}) where {T} = string(T)

main(isempty(ARGS) ? joinpath(@__DIR__, "results") : ARGS[1])
