# The correctness sweep: every row of `problems.jl`, in `Float32` and `Float64`, with every column
# of the standard test set, run by every package. It writes one line per cell to
# `<outdir>/correctness.csv` and one line per row, with its verdict and explanation, to
# `<outdir>/rows.csv`, and prints the report's tables.
#
# Usage: julia --project=benchmark benchmark/correctness.jl [outdir]   (default benchmark/results)

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "packages.jl"))

"""
    sweep()

Run every cell, and return the cells and the rows, each row a named tuple of its key, its verdict
and its explanation.
"""
function sweep()
    cells = Cell[]
    rows = NamedTuple[]
    for T in (Float32, Float64), row in benchmark_rows(T), column in COLUMNS
        rc = [run_cell(row, column, package) for package in PACKAGES]
        key = (; row.set, row.problem, row.start, eltype = string(T), column)
        v = verdict(rc)
        append!(cells, rc)
        push!(rows, (; key..., verdict = v, explanation = explain(row, column, rc, v)))
    end
    return cells, rows
end

function main(outdir)
    mkpath(outdir)
    cells, rows = sweep()
    open(joinpath(outdir, "correctness.csv"), "w") do io
        println(io, CELLS_HEADER)
        foreach(c -> write_cell(io, c), cells)
    end
    open(joinpath(outdir, "rows.csv"), "w") do io
        println(io, ROWS_HEADER)
        for r in rows
            println(io, join(values(r), ","))
        end
    end
    report_tables(stdout, cells, rows)
    unexplained = count(
        r -> r.verdict in ("disagree", "different root") &&
             isempty(r.explanation), rows)
    println("\n$(length(cells)) cells, $(length(rows)) rows, $unexplained unexplained")
    return nothing
end

main(isempty(ARGS) ? joinpath(@__DIR__, "results") : ARGS[1])
