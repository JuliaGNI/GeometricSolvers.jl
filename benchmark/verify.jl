# The verifier of the correctness results. It reads `<dir>/correctness.csv`, `<dir>/rows.csv`,
# `benchmark/REPORT.md` and `benchmark/traces/`, and checks them with code of its own: it loads
# no file of the harness and no solver package.
#
# Usage: julia benchmark/verify.jl <dir> [--structure]
#
# With `--structure` it checks everything but the explanations and the report's counts, for the
# output of a new run.

const DIR = ARGS[1]
const STRUCTURE = "--structure" in ARGS[2:end]
const REPORT = joinpath(@__DIR__, "REPORT.md")
const TRACES = joinpath(@__DIR__, "traces")

const PKGS = ["GeometricSolvers", "SimpleSolvers", "SimpleNonlinearSolve", "NLsolve"]
const CELLS_HEAD = "set,problem,start,eltype,column,package,status,method,code,success,iterations,f_evals,dual_evals,fnorm_inf,x,comment"
const ROWS_HEAD = "set,problem,start,eltype,column,verdict,explanation"
const N_ROWS = 2054

# The codes with which each package reports success.
const OK_CODES = Dict(
    "GeometricSolvers" => ["SUCCESS"],
    "SimpleSolvers" => ["converged"],
    "SimpleNonlinearSolve" => ["Success", "Terminated", "ExactSolutionLeft",
        "ExactSolutionRight", "FloatingPointLimit", "StalledSuccess"],
    "NLsolve" => ["converged"])

failures = String[]
check(ok::Bool, msg) = (ok || push!(failures, msg); ok)

# A number as `repr` writes it in its element type: `1.5f-5`, `NaN32` and `Inf32` in Float32.
function number(s::AbstractString, T)
    if T === Float32
        s = replace(s, "NaN32" => "NaN", "Inf32" => "Inf", "f" => "e")
    end
    return parse(T, s)
end

function eltype_of(name)
    name == "Float32" ? Float32 :
    name == "Float64" ? Float64 :
    error("unknown element type $name")
end

maxabs(v) = isempty(v) ? 0.0 : maximum(abs, v)

# --- the cells -----------------------------------------------------------------------------

lines = readlines(joinpath(DIR, "correctness.csv"))
check(lines[1] == CELLS_HEAD, "correctness.csv: header is $(lines[1])")
cells = [split(l, ",") for l in lines[2:end]]
check(all(c -> length(c) == 16, cells), "correctness.csv: a line without 16 fields")
cells = filter(c -> length(c) == 16, cells)
check(length(cells) == 4 * N_ROWS, "correctness.csv: $(length(cells)) cells, not $(4N_ROWS)")

rowkeys = String[]
cellsof = Dict{String, Vector{Vector{SubString{String}}}}()
for c in cells
    k = join(c[1:5], ",")
    haskey(cellsof, k) || push!(rowkeys, k)
    push!(get!(cellsof, k, []), c)
end
check(length(rowkeys) == N_ROWS, "correctness.csv: $(length(rowkeys)) rows, not $N_ROWS")
for k in rowkeys
    check([c[6] for c in cellsof[k]] == PKGS,
        "correctness.csv: row $k has the packages " *
        join([c[6] for c in cellsof[k]], " "))
end

# The pairing of the report: package and column => label, or `nothing` for none.
report = readlines(REPORT)
function table(heading)
    i = findfirst(==(heading), report)
    i === nothing && (push!(failures, "REPORT.md: no section $heading"); return [])
    rows = []
    for l in report[(i + 1):end]
        startswith(l, "#") && break
        startswith(l, "|") || continue
        fields = strip.(split(strip(l, '|'), "|"))
        all(f -> occursin(r"^:?-+:?$", f), fields) && continue
        push!(rows, fields)
    end
    return rows[2:end]   # the head line
end

pairing = Dict{Tuple{String, String}, Union{Nothing, String}}()
for f in table("### Pairing")
    m = f[3] == "none" ? nothing : strip(f[3], '`')
    pairing[(f[1], f[2])] = m
end
columns = unique(first.(keys(pairing)))
check(length(columns) == 13 && length(pairing) == 52,
    "REPORT.md: the pairing table has $(length(columns)) columns and $(length(pairing)) entries")

success = Dict{Tuple{String, String, String}, Vector{Bool}}()   # column, package, eltype
for c in cells
    set, problem, start, el, column, pkg, status, method,
    code, succ, iters, fe, de, fn, x, comment = c
    where = "cell $(join(c[1:6], ","))"
    check(status in ("ran", "no method", "no Float32"), "$where: status '$status'")
    check(column in columns, "$where: column not in the report's pairing table")
    check((pkg == "NLsolve" && el == "Float32") == (status == "no Float32"),
        "$where: status $status")
    paired = get(pairing, (String(column), String(pkg)), nothing)
    status == "no Float32" ||
        check((paired === nothing) == (status == "no method"),
            "$where: status $status, the report's pairing table has $(something(paired, "none"))")
    if status != "ran"
        check(all(isempty, (code, succ, iters, fe, de, fn, x)) && !isempty(comment),
            "$where: a cell that did not run has fields or no reason")
        continue
    end
    check(method == something(paired, ""), "$where: method $method, the report has $paired")
    T = eltype_of(el)
    ok = if code == "ERROR"
        check(isempty(x) && isempty(fn) && !isempty(comment), "$where: an ERROR with a result")
        false
    else
        xs = number.(split(x, ";"), T)
        r = number(fn, T)
        code in OK_CODES[pkg] && all(isfinite, xs) && r ≤ sqrt(eps(T))
    end
    check(succ == string(ok), "$where: success $succ, recomputed $ok")
    push!(get!(success, (String(column), String(pkg), String(el)), Bool[]), ok)
end

# --- the rows ------------------------------------------------------------------------------

rlines = readlines(joinpath(DIR, "rows.csv"))
check(rlines[1] == ROWS_HEAD, "rows.csv: header is $(rlines[1])")
rows = [split(l, ",") for l in rlines[2:end]]
check(length(rows) == N_ROWS && all(r -> length(r) == 7, rows),
    "rows.csv: $(length(rows)) rows, not $N_ROWS of 7 fields")
check([join(r[1:5], ",") for r in rows] == rowkeys, "rows.csv: the rows differ from the cells'")

function recomputed_verdict(k)
    T = eltype_of(split(k, ",")[4])
    ran = [c for c in cellsof[k] if c[7] == "ran"]
    oks = [c[10] == "true" for c in ran]
    length(ran) == 1 && return "single"
    any(oks) || return "all fail"
    all(oks) || return "disagree"
    xs = [number.(split(c[15], ";"), T) for c in ran]
    bound = sqrt(eps(T)) * max(one(T), maxabs(xs[1]))
    return all(x -> maxabs(x .- xs[1]) ≤ bound, xs) ? "agree" : "different root"
end

for r in rows
    k = join(r[1:5], ",")
    haskey(cellsof, k) || continue
    v = recomputed_verdict(k)
    check(r[6] == v, "rows.csv: row $k has $(r[6]), recomputed $v")
end

# --- the report's counts and the explanations --------------------------------------------------

if !STRUCTURE
    counts = table("### Success counts")
    check(length(counts) == 13, "REPORT.md: the success table has $(length(counts)) columns")
    for f in counts
        i = 2
        for pkg in PKGS, el in ("Float32", "Float64")

            oks = get(success, (String(f[1]), pkg, el), Bool[])
            want = isempty(oks) ? "—" : "$(count(oks))/$(length(oks))"
            check(f[i] == want, "REPORT.md: success of $pkg $el on $(f[1]) is $(f[i]), recomputed $want")
            i += 1
        end
    end

    vtable = table("### Verdicts")
    groups = [(s, el) for s in ("standard", "midpoint") for el in ("Float32", "Float64")]
    for f in vtable
        for (j, (s, el)) in enumerate(groups)
            n = count(r -> r[6] == f[1] && r[1] == s && r[4] == el, rows)
            check(f[j + 1] == string(n), "REPORT.md: $(f[1]) rows of $s $el is $(f[j + 1]), recomputed $n")
        end
    end
    check(sum(f -> sum(parse.(Int, f[2:end])), vtable; init = 0) == N_ROWS,
        "REPORT.md: the verdict table does not count $N_ROWS rows")

    named = Dict{String, Int}()
    for r in rows
        needs = r[6] in ("disagree", "different root")
        check(needs == !isempty(r[7]), "rows.csv: row $(join(r[1:5], ",")) $(r[6]) names '$(r[7])'")
        isempty(r[7]) || (named[r[7]] = get(named, r[7], 0) + 1)
    end
    headings = Dict{String, Int}()
    for (i, l) in enumerate(report)
        m = match(r"^### (E\d+) · .+$", l)
        m === nothing && continue
        id = m[1]
        block = report[(i + 1):(something(
            findnext(x -> startswith(x, "#"), report, i + 1), length(report) + 1) - 1)]
        rl = findfirst(x -> startswith(x, "- **Rows:** "), block)
        tl = findfirst(==("- **Trace:** benchmark/traces/$id.txt"), block)
        check(rl !== nothing && tl !== nothing, "REPORT.md: $id lacks its Rows or Trace line")
        rl === nothing || (headings[id] = parse(Int, block[rl][13:end]))
        path = joinpath(TRACES, "$id.txt")
        if check(isfile(path), "traces: no $id.txt")
            tr = readlines(path)
            key = startswith(tr[1], "key: ") ? tr[1][6:end] : ""
            i = findfirst(r -> join(r[1:5], ",") == key, rows)
            check(i !== nothing && rows[i][7] == id, "traces/$id.txt: key '$key' is no row of $id")
            starts = findall(l -> startswith(l, "== "), tr)
            check(length(starts) ≥ 2, "traces/$id.txt: $(length(starts)) package blocks")
            # Each block is a package that ran on the row, and holds at least one call.
            ran = haskey(cellsof, key) ? [c[6] for c in cellsof[key] if c[7] == "ran"] : []
            for (j, s) in enumerate(starts)
                stop = j < length(starts) ? starts[j + 1] - 1 : length(tr)
                check(tr[s][4:end] in ran && any(l -> occursin(r"^\d+ ", l), tr[s:stop]),
                    "traces/$id.txt: the block $(tr[s]) is no package that ran, or has no call")
            end
            check(
                all(l -> startswith(l, r"key: |== |# ") || occursin(r"^\d+ \S+$", l), tr),
                "traces/$id.txt: a line that is no call")
        end
    end
    for (id, n) in named
        check(get(headings, id, -1) == n,
            "REPORT.md: $id has Rows $(get(headings, id, "none")), named by $n rows")
    end
    for id in keys(headings)
        check(haskey(named, id), "REPORT.md: $id is named by no row")
    end
end

if isempty(failures)
    println("VERIFY OK: $(length(cells)) cells, $(length(rows)) rows",
        STRUCTURE ? " (structure)" : "")
else
    foreach(f -> println("FAIL ", f), first(failures, 50))
    length(failures) > 50 && println("… and $(length(failures) - 50) more")
    exit(1)
end
