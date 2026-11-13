# The tables of the report's Y2 section, from the tables of `<dir>`: the performance profiles of
# Dolan and Moré over `correctness.csv` and `timings.csv`, DogLeg against Backtracking, and the
# tables of the four blocks. It prints them as the Markdown that `benchmark/REPORT.md` holds.
#
# Usage: julia --project=benchmark benchmark/profiles.jl <dir>

using Printf: @sprintf

"""
    TAUS

The ratios `τ` at which a profile is read.
"""
const TAUS = (1, 2, 4, 8, 16, Inf)

"""
    profile_metric(success, value, iscount)

The metric of a cell in a profile: its `value` when the cell succeeded, at least 1 where the
metric is a count, and `Inf` when it did not.
"""
function profile_metric(success::Bool, value::Real, iscount::Bool)
    success || return Inf
    return iscount ? max(value, 1) : value
end

"""
    profile_ratios(t)

The ratios `r[p, s] = t[p, s] / minimum(t[p, :])` of the metrics `t` of the problems (rows) and
the solvers (columns): `Inf` where the solver failed, and so for every solver of a problem that
none solves.
"""
function profile_ratios(t::AbstractMatrix)
    r = similar(t, Float64)
    for p in axes(t, 1)
        best = minimum(t[p, :])
        for s in axes(t, 2)
            r[p, s] = isinf(t[p, s]) ? Inf : t[p, s] / best
        end
    end
    return r
end

"""
    profile(r, τ)

`ρ_s(τ)` of each solver `s` (column of the ratios `r`): the share of the problems with a finite
`r[p, s] ≤ τ`; at `τ = Inf`, the share that `s` solves.
"""
function profile(r::AbstractMatrix, τ)
    [count(x -> isfinite(x) && x ≤ τ, r[:, s]) / size(r, 1)
     for s in axes(r, 2)]
end

# Three decimals, and three significant digits.
dec3(x) = @sprintf("%.3f", x)
function sig3(x)
    isnan(x) && return "—"
    s = @sprintf("%#.3g", x)
    return endswith(s, ".") ? s[1:(end - 1)] : s
end

function median_of(v)
    isempty(v) ? NaN :
    (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
end

# A table of `dir` as named tuples of its fields, by the header.
function read_table(path)
    lines = readlines(path)
    names = Symbol.(split(lines[1], ","))
    return [NamedTuple{Tuple(names)}(Tuple(String.(split(l, ",")))) for l in lines[2:end]]
end

const PROFILE_PACKAGES = (
    "GeometricSolvers", "SimpleSolvers", "SimpleNonlinearSolve", "NLsolve")
const METRICS = ("iterations", "f_evals", "median_s")

cellkey(c) = (c.set, c.problem, c.start, c.eltype, c.column, c.package)
rowcol(c) = (c.set, c.problem, c.start, c.eltype, c.column)

# The metric of a correctness cell, from the cell and its timing.
function metric_value(c, timing, metric)
    metric == "median_s" ? parse(Float64, timing.median_s) :
    parse(Float64, getfield(c, Symbol(metric)))
end

"""
    profile_solvers(eltype, metric)

The solvers of a profile: the packages that run in `eltype`, without SimpleNonlinearSolve, which
reports no iteration count, for the metric `iterations`.
"""
function profile_solvers(eltype, metric)
    filter(collect(PROFILE_PACKAGES)) do p
        !(p == "NLsolve" && eltype == "Float32") &&
            !(p == "SimpleNonlinearSolve" && metric == "iterations")
    end
end

"""
    profile_table(cells, timings, eltype, metric)

The solvers and `ρ_s(τ)` at each `τ` of `TAUS` of the profile of `metric` in `eltype` over every
(row, column) pair of that precision.
"""
function profile_table(cells, timings, eltype, metric)
    solvers = profile_solvers(eltype, metric)
    pairs = unique(rowcol(c) for c in cells if c.eltype == eltype)
    byk = Dict(cellkey(c) => c for c in cells)
    t = fill(Inf, length(pairs), length(solvers))
    for (i, pair) in enumerate(pairs), (j, s) in enumerate(solvers)

        c = byk[(pair..., s)]
        c.status == "ran" || continue
        ok = c.success == "true"
        t[i, j] = profile_metric(ok,
            ok ? metric_value(c, get(timings, cellkey(c), nothing), metric) : 0.0,
            metric != "median_s")
    end
    r = profile_ratios(t)
    return (; solvers, n = length(pairs), rho = [profile(r, τ) for τ in TAUS])
end

# The median ratio `a / b` of a metric over the pairs `(a, b)` of cells, counts at least 1.
function median_ratio(pairs, metric)
    v = Float64[]
    for (a, b) in pairs
        x, y = getfield(a, Symbol(metric)), getfield(b, Symbol(metric))
        (x == "n/a" || y == "n/a") && return NaN
        xa, yb = parse(Float64, x), parse(Float64, y)
        if metric != "median_s"
            xa, yb = max(xa, 1), max(yb, 1)
        end
        push!(v, xa / yb)
    end
    return median_of(v)
end

# A correctness cell with its median time as the field `median_s`.
function timed(c, timings)
    merge(c,
        (; median_s = c.status == "ran" ? timings[cellkey(c)].median_s : ""))
end

function print_profiles(io, cells, timings)
    for eltype in ("Float32", "Float64"), metric in METRICS

        (; solvers, n, rho) = profile_table(cells, timings, eltype, metric)
        println(io, "#### Profile · $eltype · $metric\n")
        println(io, "$n problems.\n")
        println(
            io, "| solver | ", join(("τ = " * (isinf(τ) ? "∞" : string(τ)) for τ in TAUS),
                " | "), " |")
        println(io, "|:--|", repeat("--:|", length(TAUS)))
        for (j, s) in enumerate(solvers)
            println(io, "| $s | ", join((dec3(rho[i][j]) for i in eachindex(TAUS)), " | "), " |")
        end
        println(io)
    end
end

function print_dogleg(io, cells, timings)
    println(io,
        "| package | eltype | DogLeg successes | Backtracking successes | both | iterations | f_evals | median_s |")
    println(io, "|:--|:--|--:|--:|--:|--:|--:|--:|")
    byk = Dict(cellkey(c) => c for c in cells)
    for p in ("GeometricSolvers", "SimpleSolvers", "NLsolve"),
        eltype in ("Float32", "Float64")

        rows = unique((c.set, c.problem, c.start) for c in cells if c.eltype == eltype)
        dl = [byk[(r..., eltype, "Newton DogLeg LU", p)] for r in rows]
        bt = [byk[(r..., eltype, "Newton Backtracking LU", p)] for r in rows]
        all(c -> c.status == "ran", [dl; bt]) || continue
        both = [(timed(a, timings), timed(b, timings))
                for (a, b) in zip(dl, bt) if a.success == "true" && b.success == "true"]
        println(io, "| $p | $eltype | ", count(c -> c.success == "true", dl), " | ",
            count(c -> c.success == "true", bt), " | ", length(both), " | ",
            join((sig3(median_ratio(both, m)) for m in METRICS), " | "), " |")
    end
end

ok(c) = c.success == "true"

function print_mixed(io, mixed)
    println(io, "| variant | method | successes of 79 | both with gs-f64 | iterations | median_s |")
    println(io, "|:--|:--|--:|--:|--:|--:|")
    base = Dict((c.set, c.problem, c.start) => c for c in mixed if c.variant == "gs-f64")
    for v in unique(c.variant for c in mixed)
        cs = [c for c in mixed if c.variant == v]
        both = [(c, base[(c.set, c.problem, c.start)])
                for c in cs if ok(c) && ok(base[(c.set, c.problem, c.start)])]
        println(
            io, "| $v | `$(cs[1].method)` | ", count(ok, cs), " | ", length(both), " | ",
            sig3(median_ratio(both, "iterations")), " | ", sig3(median_ratio(both, "median_s")),
            " |")
    end
end

function print_stage(io, stage, cells, timings)
    println(io,
        "| package | successes at eps(Float16) | successes at √eps(Float32) | both | iterations | f_evals | median_s |")
    println(io, "|:--|--:|--:|--:|--:|--:|--:|")
    strict = Dict(cellkey(c) => c for c in cells)
    for p in ("GeometricSolvers", "SimpleSolvers", "SimpleNonlinearSolve")
        loose = [c for c in stage if c.package == p]
        tight = [timed(strict[(c.set, c.problem, c.start, c.eltype, c.variant, p)], timings)
                 for c in loose]
        both = [(a, b) for (a, b) in zip(loose, tight) if ok(a) && ok(b)]
        println(
            io, "| $p | ", count(ok, loose), " | ", count(ok, tight), " | ", length(both),
            " | ", join((sig3(median_ratio(both, m)) for m in METRICS), " | "), " |")
    end
end

function print_anderson(io, anderson)
    println(io,
        "| eltype | k | anderson-k successes | newton-k successes | both | iterations | f_evals | median_s |")
    println(io, "|:--|--:|--:|--:|--:|--:|--:|--:|")
    for eltype in ("Float32", "Float64"),
        k in unique(last(split(c.variant, "-")) for c in anderson)

        a = [c for c in anderson if c.eltype == eltype && c.variant == "anderson-$k"]
        n = [c for c in anderson if c.eltype == eltype && c.variant == "newton-$k"]
        both = [(x, y) for (x, y) in zip(a, n) if ok(x) && ok(y)]
        println(
            io, "| $eltype | $k | ", count(ok, a), " | ", count(ok, n), " | ", length(both),
            " | ", join((sig3(median_ratio(both, m)) for m in METRICS), " | "), " |")
    end
end

function print_sparse(io, sparse)
    println(io, "| variant | N | eltype | status | code | iterations | f_evals | median_s |")
    println(io, "|:--|--:|:--|:--|:--|--:|--:|--:|")
    for c in sparse
        N = last(split(c.problem, "="))
        t = isempty(c.median_s) ? "" : sig3(parse(Float64, c.median_s))
        println(io, "| $(c.variant) | $N | $(c.eltype) | $(c.status) | $(c.code) | ",
            "$(c.iterations) | $(c.f_evals) | $t |")
    end
end

function profiles_main(dir)
    cells = read_table(joinpath(dir, "correctness.csv"))
    timings = Dict(cellkey(t) => t for t in read_table(joinpath(dir, "timings.csv")))
    blocks = Dict(b => read_table(joinpath(dir, "$b.csv"))
    for b in ("mixed", "stage", "anderson", "sparse"))
    println("### Y2 · Performance profiles\n")
    print_profiles(stdout, cells, timings)
    println("### Y2 · DogLeg against Backtracking\n")
    print_dogleg(stdout, cells, timings)
    println("\n### Y2 · Mixed precision\n")
    print_mixed(stdout, blocks["mixed"])
    println("\n### Y2 · Stage tolerance\n")
    print_stage(stdout, blocks["stage"], cells, timings)
    println("\n### Y2 · Anderson\n")
    print_anderson(stdout, blocks["anderson"])
    println("\n### Y2 · The sparse problem\n")
    print_sparse(stdout, blocks["sparse"])
    timed_cells = values(timings)
    println("\nTimed cells whose warm-up solve differs from correctness.csv: ",
        count(t -> occursin("differs from correctness.csv", t.note), timed_cells))
end

if abspath(PROGRAM_FILE) == @__FILE__
    profiles_main(ARGS[1])
end
