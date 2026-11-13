# The four blocks of further cells: mixed precision, the stage tolerance, Anderson and the sparse
# problem. Each cell is solved once, as a cell of `correctness.csv` is, and then timed as a cell
# of the timed sweep is, in one fresh Julia process per block, package and precision. It writes
# `<dir>/mixed.csv`, `stage.csv`, `anderson.csv` and `sparse.csv`.
#
# Usage: julia --project=benchmark benchmark/blocks.jl <dir>
#        julia --threads=1 --project=benchmark benchmark/blocks.jl <dir> \
#            --worker <block> <package> <eltype>
#        julia --project=benchmark benchmark/blocks.jl <dir> --merge
#
# The first form runs every worker, then merges. A worker writes the cells of one block, package
# and precision into `<dir>/<block>-<package>-<eltype>.part`; `--merge` writes the four tables
# from the parts and removes them.

# The self-test includes this file after the harness.
@isdefined(benchmark_rows) || include(joinpath(@__DIR__, "problems.jl"))
@isdefined(run_invoked) || include(joinpath(@__DIR__, "packages.jl"))
import MultiPrecisionArrays

const BLOCK_HEADER = "block,set,problem,start,eltype,variant,package,status,method,code,success,iterations,f_evals,fnorm_inf,tol,t1,t2,t3,t4,t5,median_s,spread,comment"

const BLOCKS = ("mixed", "stage", "anderson", "sparse")

# GeometricSolvers' `Newton` with backtracking, the linear method `linsolve` and the refinement
# `refinement`.
function mixed_newton(linsolve, refinement = GS.NoRefinement())
    GS.Newton(; linsolve, globalization = GS.Backtracking(), refinement)
end

"""
    MIXED_VARIANTS

The variants of the mixed-precision block, each a name, its package and the objects that its
adapter passes: GeometricSolvers' `Newton` with backtracking and an LU in `Float64`, in
`Float32`, and in `Float32` with refinement; and `nsol` with a `Float32` Jacobian store, and with
a `Float64` store factored by MultiPrecisionArrays' `mplu`. Each `nsol` takes a new Jacobian at
every iteration.
"""
const MIXED_VARIANTS = [
    "gs-f64" => ("GeometricSolvers", (; method = mixed_newton(GS.LUFactorization()))),
    "gs-f32" => ("GeometricSolvers",
        (; method = mixed_newton(GS.LUFactorization(Float32), GS.NoRefinement()))),
    "gs-f32-refine" => ("GeometricSolvers",
        (; method = mixed_newton(GS.LUFactorization(Float32), GS.Refinement()))),
    "nsol-f32" => ("SIAMFANLEquations", (; nsol = Nsol(Float32, 1))),
    "nsol-mplu" =>
        ("SIAMFANLEquations", (; nsol = Nsol(Float64, 1, MultiPrecisionArrays.mplu)))
]

"""
    STAGE_TOL

The tolerance of the stage-tolerance block, `eps(Float16)` in `Float32`.
"""
const STAGE_TOL = Float32(eps(Float16))

const ANDERSON_K = (1, 2, 3, 5)
const ANDERSON_DEPTH = 4

function anderson_newton(k)
    GS.Newton(; reuse = k, globalization = GS.Backtracking(), jacobian = nothing)
end

"""
    ANDERSON_VARIANTS

The variants of the Anderson block, for each `k` of `ANDERSON_K`: `anderson-k`, `Anderson` of
depth `ANDERSON_DEPTH` over `Newton` with backtracking and the factor reused for `k` steps, and
`newton-k`, that `Newton` alone.
"""
const ANDERSON_VARIANTS = [v => ("GeometricSolvers", (; method = m))
                           for k in ANDERSON_K
                           for (v, m) in (
    ("anderson-$k", GS.Anderson(anderson_newton(k), ANDERSON_DEPTH)),
    ("newton-$k", anderson_newton(k)))]

"""
    SPARSE_VARIANTS

The variants of the sparse block, each a name and its package.
"""
const SPARSE_VARIANTS = [
    "gs-sparse" => "GeometricSolvers", "gs-dense" => "GeometricSolvers",
    "nlsolve-sparse" => "NLsolve", "simplesolvers" => "SimpleSolvers",
    "simplenonlinearsolve" => "SimpleNonlinearSolve"]

"""
    BlockCell

One cell of a block: the block, the element type, the row key `(set, problem, start)`, the
variant and the package.
"""
struct BlockCell
    block::String
    eltype::String
    key::NTuple{3, String}
    variant::String
    package::String
end

rowkey(r::Row) = (r.set, r.problem, r.start)

"""
    block_cells(block)

The cells of `block`, in the order of its table.
"""
function block_cells(block)
    if block == "mixed"
        return [BlockCell(block, "Float64", rowkey(r), v, p)
                for r in benchmark_rows(Float64) for (v, (p, _)) in MIXED_VARIANTS]
    elseif block == "stage"
        return [BlockCell(block, "Float32", rowkey(r), column, p)
                for r in midpoint_rows(Float32) for column in COLUMNS for p in PACKAGES]
    elseif block == "anderson"
        return [BlockCell(block, string(T), rowkey(r), v, p)
                for T in (Float32, Float64) for r in benchmark_rows(T)
                for (v, (p, _)) in ANDERSON_VARIANTS]
    elseif block == "sparse"
        return [BlockCell(block, string(T), ("sparse", "bracket N=$N", "z0"), v, p)
                for N in SPARSE_SIZES for T in (Float32, Float64)
                for (v, p) in SPARSE_VARIANTS]
    end
    throw(ArgumentError("no block $block"))
end

"""
    block_run(b, rows)

How the cell `b` runs: `(; status, comment)` for a cell that does not run, or
`(; row, invoke, tol, p)` for one that does, with `invoke` as `run_invoked` takes it. `rows` maps
the key of each standard and midpoint row of the cell's element type to the row.
"""
function block_run(b::BlockCell, rows)
    b.block == "sparse" && return sparse_run(b)
    row = rows[b.key]
    T = eltype(row.x0)
    if b.block == "mixed"
        package, objs = Dict(MIXED_VARIANTS)[b.variant]
        tol = tolerance(T)
        return (; row, invoke = adapter(package, objs, row; tol), tol, p = nothing)
    elseif b.block == "stage"
        b.package == "NLsolve" &&
            return (; status = "no Float32", comment = "NLsolve runs in Float64 only")
        entry = PAIRING[b.variant][findfirst(==(b.package), PACKAGES)]
        entry isa NoMethod && return (; status = "no method", comment = entry.reason)
        tol = STAGE_TOL
        invoke = adapter(b.package, entry.build(T), row; tol)
        return (; row, invoke, tol, p = nothing)
    elseif b.block == "anderson"
        package, objs = Dict(ANDERSON_VARIANTS)[b.variant]
        tol = tolerance(T)
        return (; row, invoke = adapter(package, objs, row; tol), tol, p = nothing)
    end
    throw(ArgumentError("no block $(b.block)"))
end

function sparse_run(b::BlockCell)
    b.variant in ("simplesolvers", "simplenonlinearsolve") &&
        return (; status = "no sparse path",
            comment = "the benchmark has no sparse adapter for $(b.package)")
    b.package == "NLsolve" && b.eltype == "Float32" &&
        return (; status = "no Float32", comment = "NLsolve runs in Float64 only")
    T = b.eltype == "Float32" ? Float32 : Float64
    sp = sparse_row(parse(Int, last(split(b.key[2], "="))), T)
    tol = tolerance(T)
    invoke = c -> sparse_call(Val(Symbol(b.variant)), sp, c, copy(sp.row.x0); tol)
    return (; sp.row, invoke, tol, sp.p)
end

"""
    block_line(b, rows)

The line of the cell `b`: its key, status, method, code, success, iterations, `f_evals`, `‖F‖∞`
and `tol` from one solve, as `run_invoked` gives them, then the times of `time_cell`, their
median and spread, and the comment. A cell that does not run has empty fields from `code` on and
its reason in `comment`.
"""
function block_line(b::BlockCell, rows)
    key = [b.block; collect(b.key); b.eltype; b.variant; b.package]
    run = block_run(b, rows)
    haskey(run, :status) &&
        return join([key; run.status; ""; fill("", 13); run.comment], ",")
    c = run_invoked(run.row, b.variant, b.package, run.invoke; run.tol, run.p)
    (; times) = time_cell(run.row, run.invoke)
    fields = [c.method, c.code, string(c.success), c.iterations, string(c.f_evals),
        c.fnorm === nothing ? "" : repr(c.fnorm), repr(run.tol), repr.(times)...,
        repr(run_median(times)), repr(run_spread(times)), c.comment]
    return join([key; "ran"; fields], ",")
end

block_part_path(dir, block, package, eltype) = joinpath(dir, "$block-$package-$eltype.part")

function block_worker(dir, block, package, eltype)
    start_worker("blocks $block $package $eltype")
    T = eltype_named(eltype)
    rows = Dict(rowkey(r) => r for r in benchmark_rows(T))
    cells = filter(b -> b.package == package && b.eltype == eltype, block_cells(block))
    open(block_part_path(dir, block, package, eltype), "w") do io
        for b in cells
            println(io, block_line(b, rows))
        end
    end
    println("blocks $block $package $eltype: $(length(cells)) cells")
end

# The package and precision pairs of a block, in the order of their first cell.
block_groups(block) = unique([(b.package, b.eltype) for b in block_cells(block)])

function block_merge_parts(dir)
    for block in BLOCKS
        cells = block_cells(block)
        merge_part_files(joinpath(dir, "$block.csv"), BLOCK_HEADER, cells,
            b -> (b.package, b.eltype),
            b -> join([b.block; collect(b.key); b.eltype; b.variant; b.package], ","),
            g -> block_part_path(dir, block, g...))
        println("$block.csv: $(length(cells)) cells")
    end
end

function blocks_main(args)
    dir = args[1]
    if length(args) == 5 && args[2] == "--worker"
        block_worker(dir, args[3:5]...)
    elseif length(args) == 2 && args[2] == "--merge"
        block_merge_parts(dir)
    elseif length(args) == 1
        mkpath(dir)
        for block in BLOCKS, (package, eltype) in block_groups(block)

            spawn_worker(@__FILE__, dir, "--worker", block, package, eltype)
        end
        block_merge_parts(dir)
    else
        println(stderr, "usage: blocks.jl <dir> [--worker <block> <package> <eltype> | --merge]")
        exit(2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    blocks_main(ARGS)
end
