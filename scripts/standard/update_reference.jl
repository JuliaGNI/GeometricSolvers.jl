# Regenerate the reference table of the standard test set,
# `test/verification/standard_reference.csv`. This script writes the table; only the `platform`
# flag and its comment are set by hand. From the root of the repository:
#
#     julia --project=test -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test scripts/standard/update_reference.jl
#
# It runs every cell of `test/helpers/standard_cells.jl` with its 20 perturbed runs (16 for
# `Picard`), sets the `unstable` flag of each cell that the marker marks, and writes the table. A
# cell that the old table flags `platform` keeps the flag and its comment, which holds the values
# observed on the CI runners. It prints the summary, the number of `unstable` cells, and every
# cell that differs from the old table, for the pull request that changes the table.

include(joinpath(@__DIR__, "..", "..", "test", "helpers", "standard_cells.jl"))

function update_reference()
    old = isfile(STANDARD_REFERENCE) ? read_table(STANDARD_REFERENCE) : StandardCell[]
    cells, iterates = run_standard_cells(; mark = true)
    cells = keep_platform(old, cells)
    open(io -> write_table(io, cells), STANDARD_REFERENCE, "w")
    print_summary(stdout, cells, iterates)
    println("Cells flagged unstable: ", count(c -> c.unstable, cells), " of ", length(cells))
    println("Cells flagged platform: ", count(c -> c.platform, cells))
    differences = compare_tables(old, cells)
    println("Cells that differ from the old table: ", length(differences))
    foreach(d -> println("  ", d), differences)
    println("Wrote ", length(cells), " cells to ", normpath(STANDARD_REFERENCE))
    return nothing
end

update_reference()
