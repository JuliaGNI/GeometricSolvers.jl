# The timed sweep: every cell of `<dir>/correctness.csv` with `status = ran`, timed in one fresh
# Julia process per package and precision, with one Julia thread and one BLAS thread. Per cell:
# one warm-up solve, the batch `k` of solves per run, and five runs. It writes one line per ran
# cell, in the order of `correctness.csv`, to `<dir>/timings.csv`.
#
# Usage: julia --project=benchmark benchmark/timing.jl <dir>
#        julia --threads=1 --project=benchmark benchmark/timing.jl <dir> --worker <package> <eltype>
#        julia --project=benchmark benchmark/timing.jl <dir> --merge
#
# The first form runs every worker, then merges. A worker times the ran cells of one package and
# precision into `<dir>/timings-<package>-<eltype>.part`; `--merge` writes `timings.csv` from the
# parts and removes them.

# The self-test includes this file after the harness.
@isdefined(benchmark_rows) || include(joinpath(@__DIR__, "problems.jl"))
@isdefined(run_invoked) || include(joinpath(@__DIR__, "packages.jl"))

const TIMINGS_HEADER = "set,problem,start,eltype,column,package,t1,t2,t3,t4,t5,median_s,spread,solves_per_run,note"

# The ran cells of `correctness.csv`, each the vector of its fields.
function ran_cells(dir)
    lines = readlines(joinpath(dir, "correctness.csv"))
    lines[1] == CELLS_HEADER || error("$dir/correctness.csv: header $(lines[1])")
    return filter(f -> f[7] == "ran", [split(l, ",") for l in lines[2:end]])
end

part_path(dir, package, eltype) = joinpath(dir, "timings-$package-$eltype.part")

"""
    timing_note(warm, code, iterations, f_evals)

The note of a timed cell: `throws` when its warm-up solve throws, and `differs from
correctness.csv` with each of `code`, `iterations` and `f_evals` whose value in
`correctness.csv` (first) differs from the warm-up's (second).
"""
function timing_note(warm::Cell, code, iterations, f_evals)
    notes = String[]
    warm.code == "ERROR" && push!(notes, "throws")
    diffs = ["$name $was → $now"
             for (name, was, now) in (("code", code, warm.code),
                     ("iterations", iterations, warm.iterations),
                     ("f_evals", f_evals, string(warm.f_evals)))
             if was != now]
    isempty(diffs) || push!(notes, "differs from correctness.csv: " * join(diffs, " "))
    return join(notes, "; ")
end

"""
    timing_line(row, f)

The line of `timings.csv` for the ran cell with the fields `f` of `correctness.csv` on `row`: a
warm-up solve, then the times of `time_cell`, their median and spread, the batch and the note.
"""
function timing_line(row::Row{T}, f) where {T}
    column, package = f[5], f[6]
    entry = PAIRING[column][findfirst(==(package), PACKAGES)]
    invoke = adapter(package, entry.build(T), row; tol = tolerance(T))
    warm = run_invoked(row, column, package, invoke; tol = tolerance(T))
    (; k, times) = time_cell(row, invoke)
    note = timing_note(warm, f[9], f[11], f[12])
    return join(
        [f[1:6]; repr.(times); repr(run_median(times)); repr(run_spread(times));
         string(k); note], ",")
end

function worker(dir, package, eltype)
    start_worker("timing $package $eltype")
    T = eltype_named(eltype)
    rows = Dict((r.set, r.problem, r.start) => r for r in benchmark_rows(T))
    cells = filter(f -> f[6] == package && f[4] == eltype, ran_cells(dir))
    open(part_path(dir, package, eltype), "w") do io
        for f in cells
            println(io, timing_line(rows[(f[1], f[2], f[3])], f))
        end
    end
    println("timing $package $eltype: $(length(cells)) cells")
end

# The package and precision pairs that have a ran cell, in the order of their first cell.
groups(cells) = unique([(String(f[6]), String(f[4])) for f in cells])

function merge_parts(dir)
    cells = ran_cells(dir)
    merge_part_files(joinpath(dir, "timings.csv"), TIMINGS_HEADER, cells,
        f -> (String(f[6]), String(f[4])), f -> join(f[1:6], ","), g -> part_path(dir, g...))
    println("timings.csv: $(length(cells)) cells")
end

function timing_main(args)
    dir = args[1]
    if length(args) == 4 && args[2] == "--worker"
        worker(dir, args[3], args[4])
    elseif length(args) == 2 && args[2] == "--merge"
        merge_parts(dir)
    elseif length(args) == 1
        for (package, eltype) in groups(ran_cells(dir))
            spawn_worker(@__FILE__, dir, "--worker", package, eltype)
        end
        merge_parts(dir)
    else
        println(stderr, "usage: timing.jl <dir> [--worker <package> <eltype> | --merge]")
        exit(2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    timing_main(ARGS)
end
