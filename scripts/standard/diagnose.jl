# The diagnosis of the cells that differ between CI runners, which README.md reports. From the
# root of the repository, after `julia --project=test -e 'using Pkg; Pkg.instantiate()'`:
#
#     julia --project=test scripts/standard/diagnose.jl perturb
#     julia --project=test scripts/standard/diagnose.jl rerun [blas1 | generic]
#
# `perturb` runs every cell with each family of perturbed runs of `perturbed_runs`, and prints,
# per family, the cells whose code changes or whose iteration count moves by more than 1, how
# many of them are CI cells (`ci_cells.txt`), and the cells that the marker marks.
#
# `rerun` runs every cell once and prints the cells whose code, iteration count, evaluation
# counts or `fnorm` differ from the reference table. `blas1` runs with one BLAS thread;
# `generic` replaces LAPACK's LU and QR by pure Julia (`generic_lapack.jl`). The variable
# `OPENBLAS_CORETYPE` of the environment selects an OpenBLAS kernel, and the process prints the
# kernel it loaded.

include(joinpath(@__DIR__, "..", "..", "test", "helpers", "standard_cells.jl"))
using LinearAlgebra: BLAS
using Libdl: dlopen, dlsym

const KEY = r"\((\d+), \"([^\"]+)\", \"([^\"]+)\", \"([^\"]+)\"\)"

function ci_cells(path = joinpath(@__DIR__, "ci_cells.txt"))
    keys = Set{Tuple{Int, String, String, String}}()
    for line in eachline(path)
        m = match(KEY, line)
        m === nothing || push!(keys, (parse(Int, m[1]), m[2], m[3], m[4]))
    end
    return keys
end

function openblas_core()
    unsafe_string(ccall(
        dlsym(dlopen(BLAS.get_config().loaded_libs[1].libname), :openblas_get_corename64_),
        Cstring, ()))
end

# The families of perturbed runs of one cell; together they are the runs of `perturbed_runs`.
const FAMILIES = (
    "x0 1 ulp" =>
        (sp, start, x₀, column, method) -> [run_cell(sp, start, perturb_start(x₀, 1, seed),
                                                column, method) for seed in 1:4],
    "x0 2, 4 ulps" =>
        (sp, start, x₀, column, method) -> [run_cell(sp, start, perturb_start(x₀, k, seed),
                                                column, method)
                                            for k in (2, 4) for seed in 1:4],
    "residual 1 ulp" =>
        (sp, start, x₀, column, method) -> [run_cell(sp, start, x₀, column, method;
                                                F = perturbed_residual(sp.prob.F, seed))
                                            for seed in 1:4],
    "direction 1 ulp" =>
        (sp, start, x₀, column, method) -> [run_cell(sp, start, x₀, column,
                                                perturbed_direction(method, seed))
                                            for seed in 1:4 if !(method isa Picard)]
)

flips(c, c′) = c.code != c′.code || abs(c.iterations - c′.iterations) > 1

function perturb()
    listed = ci_cells()
    flipped = Dict(name => Set() for (name, _) in FAMILIES)
    marked = Set()
    ncells = 0
    seconds = @elapsed for sp in STANDARD_PROBLEMS, T in REAL_ELTYPES,
        (start, x₀) in standard_starts(sp, T),
        (column, method) in STANDARD_COLUMNS
        ncells += 1
        key = (sp.number, start, column, string(T))
        c, x = run_cell(sp, start, x₀, column, method)
        for (name, family) in FAMILIES
            runs = family(sp, start, x₀, column, method)
            any(((c′, _),) -> flips(c, c′), runs) && push!(flipped[name], key)
            marks_unstable(c, x, runs) && push!(marked, key)
        end
    end
    println("cells ", ncells, ", CI cells ", length(listed), ", ",
        round(seconds; digits = 1), " s with compilation")
    report(name, s) = println(rpad(name, 34), "cells ", lpad(length(s), 4), "/", ncells,
        "   CI cells ", lpad(length(intersect(s, listed)), 2), "/", length(listed),
        "   other cells ", length(setdiff(s, listed)))
    for (name, _) in FAMILIES
        report("flipped by " * name, flipped[name])
    end
    report("flipped by any family", union(values(flipped)...))
    report("marked unstable", marked)
    println("CI cells the marker leaves: ", sort(collect(setdiff(listed, marked))))
end

function rerun(mode)
    listed = ci_cells()
    println("mode ", mode, ", BLAS threads ", BLAS.get_num_threads(), ", OpenBLAS core ",
        openblas_core(), ", OPENBLAS_CORETYPE ", get(ENV, "OPENBLAS_CORETYPE", "-"))
    reference = Dict(cell_key(c) => c for c in read_table(STANDARD_REFERENCE))
    cells, _ = run_standard_cells()
    same(r, c) = (r.code, r.iterations, r.f_evals, r.dual_evals) ==
                 (c.code, c.iterations, c.f_evals, c.dual_evals) &&
                 isequal(r.fnorm, c.fnorm)
    changed = filter(c -> !same(reference[cell_key(c)], c), cells)
    flipped = filter(c -> flips(reference[cell_key(c)], c), cells)
    println("cells not identical to the reference: ", length(changed), " of ", length(cells))
    println("cells flipped: ", length(flipped), ", of them CI cells ",
        count(c -> cell_key(c) in listed, flipped))
    for c in flipped
        r = reference[cell_key(c)]
        println("  ", cell_key(c), ": ", c.code, " ", c.iterations, ", reference ", r.code,
            " ", r.iterations, cell_key(c) in listed ? " [CI]" : "")
    end
end

mode = isempty(ARGS) ? "perturb" : ARGS[1]
if mode == "perturb"
    perturb()
elseif mode == "rerun"
    variant = length(ARGS) >= 2 ? ARGS[2] : "plain"
    variant == "blas1" && BLAS.set_num_threads(1)
    variant == "generic" && include(joinpath(@__DIR__, "generic_lapack.jl"))
    rerun(variant)
else
    error("unknown mode $mode; use perturb or rerun")
end
