# The cells of the standard test set: the columns, the counting wrapper, the cell runner, and the
# reader, writer and comparison of the reference table.
#
# A cell is one problem × start × column × element type of `test/helpers/standard_problems.jl`.
# `test/verification/standard.jl` runs every cell and compares it with the reference table, and
# `scripts/standard/update_reference.jl` runs every cell and writes the table.

using GeometricSolvers
using GeometricSolvers: NonlinearProblem, solve
using LinearAlgebra: norm

include(joinpath(@__DIR__, "standard_problems.jl"))
include(joinpath(@__DIR__, "matrix.jl"))

const STANDARD_REFERENCE = joinpath(@__DIR__, "..", "verification", "standard_reference.csv")

const STANDARD_HEADER = "problem,start,column,eltype,code,iterations,f_evals,dual_evals,fnorm,platform,comment"

const STANDARD_MAX_ITERATIONS = 200

"""
    STANDARD_COLUMNS

The nine columns of the standard test set, each a name and a method: `Newton` with each
globalisation on LU, `Newton` with `Backtracking` on QR and on SVD, `Newton` with `reuse = 3`,
and `Picard` with `Backtracking`.
"""
const STANDARD_COLUMNS = (
    "Newton Static LU" => Newton(; globalization = Static()),
    "Newton Backtracking LU" => Newton(; globalization = Backtracking()),
    "Newton Bisection LU" => Newton(; globalization = Bisection()),
    "Newton StrongWolfe LU" => Newton(; globalization = StrongWolfe()),
    "Newton DogLeg LU" => Newton(; globalization = DogLeg()),
    "Newton Backtracking QR" => Newton(;
        linsolve = QRFactorization(), globalization = Backtracking()),
    "Newton Backtracking SVD" => Newton(;
        linsolve = SVDFactorization(), globalization = Backtracking()),
    "Newton reuse 3 Backtracking LU" => Newton(; reuse = 3, globalization = Backtracking()),
    "Picard Backtracking" => Picard(; globalization = Backtracking())
)

"""
    CountedResidual(f)

A residual `f(y, x, p)` that counts its calls: `primal` where `eltype(x)` is a floating-point
type, `dual` otherwise, which is every call with dual numbers: the Jacobian chunks of the chunked
forward mode and the directional derivatives that a line search or `Picard` takes.

The field `f` is typed `Function`, so that every problem shares one `NonlinearProblem` type and
`solve` is not compiled again for the residual type of each problem.
"""
mutable struct CountedResidual
    f::Function
    primal::Int
    dual::Int
end

CountedResidual(f) = CountedResidual(f, 0, 0)

function (c::CountedResidual)(y, x, p)
    eltype(x) <: AbstractFloat ? (c.primal += 1) : (c.dual += 1)
    c.f(y, x, p)
    return nothing
end

"""
    StandardCell

One row of the reference table. `code` is the name of a `ReturnCode`, or `ERROR` for a solve
that throws, with the exception type in `comment`. A cell with `platform = true` is compared on
its code alone.
"""
struct StandardCell
    problem::Int
    start::String
    column::String
    eltype::String
    code::String
    iterations::Int
    f_evals::Int
    dual_evals::Int
    fnorm::Float64
    platform::Bool
    comment::String
end

cell_key(c::StandardCell) = (c.problem, c.start, c.column, c.eltype)

"""
    run_cell(sp, start, x₀, column, method)

Solve `sp` from `x₀` with `method` and `max_iterations = 200`, and return the cell and the
iterate. A solve that throws gives the code `ERROR`, the exception type in the comment, and the
iterate `nothing`; the exception is not rethrown.
"""
function run_cell(sp, start::AbstractString, x₀, column::AbstractString, method)
    counter = CountedResidual(sp.prob.F)
    prob = NonlinearProblem(counter)
    cell(code, iterations, fnorm, comment) = StandardCell(
        sp.number, start, column, string(eltype(x₀)), code, iterations,
        counter.primal, counter.dual, fnorm, false, comment)
    try
        x, status = solve(prob, x₀, method; max_iterations = STANDARD_MAX_ITERATIONS)
        c = cell(string(status.code), Int(status.iterations), Float64(status.fnorm), "")
        return c, x
    catch e
        return cell("ERROR", 0, NaN, string(nameof(typeof(e)))), nothing
    end
end

"""
    run_standard_cells(problems = STANDARD_PROBLEMS)

Run every problem × start × column × element type of `REAL_ELTYPES`, and return the cells and,
for each cell, its iterate.
"""
function run_standard_cells(problems = STANDARD_PROBLEMS)
    cells = StandardCell[]
    iterates = Any[]
    for sp in problems, T in REAL_ELTYPES,
        (start, x₀) in standard_starts(sp, T),
        (column, method) in STANDARD_COLUMNS
        c, x = run_cell(sp, start, x₀, column, method)
        push!(cells, c)
        push!(iterates, x)
    end
    return cells, iterates
end

"""
    write_table(io, cells)

Write the header and one line per cell. A comment holds no line break.
"""
function write_table(io::IO, cells)
    println(io, STANDARD_HEADER)
    for c in cells
        println(io,
            join(
                (c.problem, c.start, c.column, c.eltype, c.code, c.iterations,
                    c.f_evals, c.dual_evals, c.fnorm, c.platform, c.comment),
                ","))
    end
    return nothing
end

"""
    read_table(io)

Read a table that [`write_table`](@ref) wrote. The comment is the rest of a line after the tenth
comma, so it may hold commas.
"""
function read_table(io::IO)
    lines = readlines(io)
    first(lines) == STANDARD_HEADER ||
        throw(ArgumentError("the header is not the reference's: $(first(lines))"))
    return map(lines[2:end]) do line
        f = split(line, ","; limit = 11)
        StandardCell(parse(Int, f[1]), f[2], f[3], f[4], f[5], parse(Int, f[6]),
            parse(Int, f[7]), parse(Int, f[8]), parse(Float64, f[9]), parse(Bool, f[10]),
            f[11])
    end
end

read_table(path::AbstractString) = open(read_table, path)

"""
    compare_tables(reference, run)

The differences of a run from the reference, one line each; empty where the run agrees. A cell
differs where its code differs, or where its iteration count differs by more than 1 and the
reference does not flag it `platform`. A cell of the run that the reference lacks, and a cell of
the reference that the run lacks, are differences too. The counts and `fnorm` are not compared.
"""
function compare_tables(reference, run)
    differences = String[]
    ref = Dict(cell_key(c) => c for c in reference)
    got = Dict(cell_key(c) => c for c in run)
    for c in run
        key = cell_key(c)
        r = get(ref, key, nothing)
        if r === nothing
            push!(differences, "$key: not in the reference ($(c.code), $(c.iterations))")
        elseif r.code != c.code
            push!(differences, "$key: code $(c.code), reference $(r.code)")
        elseif !r.platform && abs(r.iterations - c.iterations) > 1
            push!(differences, "$key: $(c.iterations) iterations, reference $(r.iterations)")
        end
    end
    for r in reference
        haskey(got, cell_key(r)) || push!(differences, "$(cell_key(r)): not in the run")
    end
    return differences
end

"""
    near_root(sp, x)

Whether the iterate `x` is within `sqrt(eps(T))` relative distance of the stated root `x*` of
`sp`, that is `‖x − x*‖ ≤ sqrt(eps(T)) · max(1, ‖x*‖)`; `nothing` where the source states no
root.
"""
function near_root(sp, x)
    sp.root === nothing && return nothing
    T = eltype(x)
    r = sp.root(T)
    return norm(x - r) <= sqrt(eps(T)) * max(1, norm(r))
end

"""
    print_summary(io, cells, iterates, problems = STANDARD_PROBLEMS)

Print, for each column and element type, the number of `SUCCESS` cells out of the starts, and
how many of the `SUCCESS` cells of a problem with a stated root end near that root (see
[`near_root`](@ref)); then each `SUCCESS` cell that does not.
"""
function print_summary(io::IO, cells, iterates, problems = STANDARD_PROBLEMS)
    byproblem = Dict(sp.number => sp for sp in problems)
    eltypes = string.(REAL_ELTYPES)
    nstarts = Dict(T => count(c -> c.eltype == T, cells) ÷ length(STANDARD_COLUMNS)
    for T in eltypes)
    away = String[]
    println(
        io, "Standard test set: SUCCESS cells per column, and of those with a stated root ",
        "the number near it")
    println(io, rpad("column", 32), join((rpad(T, 22) for T in eltypes)))
    for (column, _) in STANDARD_COLUMNS
        entries = map(eltypes) do T
            success = 0
            rooted = 0
            near = 0
            for (c, x) in zip(cells, iterates)
                (c.column == column && c.eltype == T && c.code == "SUCCESS") || continue
                success += 1
                at = near_root(byproblem[c.problem], x)
                at === nothing && continue
                rooted += 1
                at ? (near += 1) :
                push!(away, "  $column $T problem $(c.problem) from $(c.start)")
            end
            rpad("$success/$(nstarts[T]), root $near/$rooted", 22)
        end
        println(io, rpad(column, 32), join(entries))
    end
    println(io, "SUCCESS cells away from the stated root: ", length(away))
    foreach(line -> println(io, line), away)
    return nothing
end

"""
    keep_platform(reference, run)

The cells of `run`, where each cell that `reference` flags `platform` keeps that flag and the
reference's comment, which holds the values observed on the CI runners.
"""
function keep_platform(reference, run)
    ref = Dict(cell_key(c) => c for c in reference)
    return map(run) do c
        r = get(ref, cell_key(c), nothing)
        (r === nothing || !r.platform) && return c
        StandardCell(
            c.problem, c.start, c.column, c.eltype, c.code, c.iterations, c.f_evals,
            c.dual_evals, c.fnorm, true, r.comment)
    end
end
