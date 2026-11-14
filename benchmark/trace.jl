# The iteration trace of one row: for each package that runs on it, `‖F‖∞` at each primal call of
# the residual, from the counting wrapper, with the package's code and counts.
#
# Usage: julia --project=benchmark benchmark/trace.jl <set> <problem> <start> <eltype> <column> [outfile]
#
# The row's key is the first line, as `key: <set>,<problem>,<start>,<eltype>,<column>`; each
# package that ran follows as a block `== <package>`, with one line per primal call: its index
# and `‖F‖∞`. Without `outfile` the trace goes to standard output.

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "packages.jl"))

function write_trace(io::IO, set, problem, start, eltype, column)
    T = Dict("Float32" => Float32, "Float64" => Float64)[eltype]
    i = findfirst(r -> (r.set, r.problem, r.start) == (set, problem, start), benchmark_rows(T))
    i === nothing && throw(ArgumentError("no row $set $problem $start"))
    row = benchmark_rows(T)[i]
    column in COLUMNS || throw(ArgumentError("no column $column"))
    println(io, "key: ", join((set, problem, start, eltype, column), ","))
    for package in PACKAGES
        c = run_cell(row, column, package)
        c.status == "ran" || continue
        println(io, "== ", package)
        println(io, "# method $(c.method), code $(c.code), success $(c.success), ",
            "iterations $(c.iterations), f_evals $(c.f_evals), dual_evals $(c.dual_evals), ",
            "fnorm_inf $(c.fnorm === nothing ? "" : repr(c.fnorm)) $(c.comment)")
        for (k, f) in enumerate(c.trace)
            println(io, k, " ", repr(f))
        end
    end
end

if length(ARGS) == 6
    open(io -> write_trace(io, ARGS[1:5]...), ARGS[6], "w")
elseif length(ARGS) == 5
    write_trace(stdout, ARGS...)
else
    println(stderr, "usage: trace.jl <set> <problem> <start> <eltype> <column> [outfile]")
    exit(2)
end
