# The verifier of the results. It reads `<dir>/correctness.csv`, `<dir>/rows.csv`,
# `<dir>/timings.csv`, the four block tables `<dir>/mixed.csv`, `stage.csv`, `anderson.csv` and
# `sparse.csv`, `benchmark/REPORT.md` and `benchmark/traces/`, and checks them with code of its
# own: it loads no file of the harness and no solver package.
#
# Usage: julia benchmark/verify.jl <dir> [--structure]
#
# With `--structure` it checks everything but the explanations and the report's counts and
# tables, for the output of a new run.

const DIR = ARGS[1]
const STRUCTURE = "--structure" in ARGS[2:end]

using Printf: @sprintf
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

# Whether `s` is a count: a non-negative integer.
iscount(s) = something(tryparse(Int, s), -1) ≥ 0

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
    # SimpleNonlinearSolve reports no iteration count, and a cell that threw has none.
    check(
        iscount(fe) && iscount(de) &&
            (code == "ERROR" ? isempty(iters) :
             pkg == "SimpleNonlinearSolve" ? iters == "n/a" : iscount(iters)),
        "$where: iterations '$iters', f_evals '$fe', dual_evals '$de'")
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
            check(length(starts) ≥ 2 && starts[1] == 2,
                "traces/$id.txt: $(length(starts)) package blocks, the first on line $(first(starts, 1))")
            # Each block is a package that ran on the row: its `#` line gives the cell's method,
            # code, success and counts, and one line `k ‖F‖∞` follows for each of its f_evals.
            ran = haskey(cellsof, key) ? [c for c in cellsof[key] if c[7] == "ran"] : []
            # `trace.jl` writes one block for each package that ran, in the order of `PKGS`.
            check([l[4:end] for l in tr[starts]] == [c[6] for c in ran],
                "traces/$id.txt: the blocks " * join([l[4:end] for l in tr[starts]], " ") *
                " are not the packages that ran, " * join([c[6] for c in ran], " "))
            for (j, s) in enumerate(starts)
                stop = j < length(starts) ? starts[j + 1] - 1 : length(tr)
                k = findfirst(c -> c[6] == tr[s][4:end], ran)
                if !check(k !== nothing && s < stop,
                    "traces/$id.txt: the block $(tr[s]) is no package that ran")
                    continue
                end
                c = ran[k]
                meta = match(
                    r"^# method (.*?), code (.*?), success (\S+), iterations (\S*), f_evals (\S+), dual_evals (\S+), fnorm_inf ",
                    tr[s + 1])
                check(meta !== nothing && meta.captures == c[8:13],
                    "traces/$id.txt: the block $(tr[s]) has '$(tr[s + 1])', the cell " *
                    join(c[8:13], ","))
                calls = tr[(s + 2):stop]
                check(
                    all(
                        ((i, l),) -> (f = split(l, " ");
                            length(f) == 2 &&
                                f[1] == string(i) &&
                                tryparse(Float64, f[2]) !== nothing),
                        enumerate(calls)) && string(length(calls)) == c[12],
                    "traces/$id.txt: the block $(tr[s]) has calls that are not 1 to f_evals = $(c[12])")
            end
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

# --- Y2: the timings -------------------------------------------------------------------------

const TIMINGS_HEAD = "set,problem,start,eltype,column,package,t1,t2,t3,t4,t5,median_s,spread,solves_per_run,note"
const N_RAN = 4661

# The median of the five run times, and their spread, recomputed.
function median5(ts)
    s = sort(ts)
    return s[3]
end
spread5(ts) = (maximum(ts) - minimum(ts)) / median5(ts)

# Whether the five times, their median and their spread, as `repr` writes them, are consistent.
function check_times(where, f)
    length(f) == 7 || return check(false, "$where: $(length(f)) time fields, not 7")
    ts = [tryparse(Float64, s) for s in f[1:5]]
    check(all(t -> t !== nothing && isfinite(t) && t > 0, ts),
        "$where: a time that is not finite and positive: $(join(f[1:5], " "))") ||
        return false
    check(f[6] == repr(median5(ts)), "$where: median_s $(f[6]), recomputed $(repr(median5(ts)))")
    check(f[7] == repr(spread5(ts)), "$where: spread $(f[7]), recomputed $(repr(spread5(ts)))")
end

ranlines = [c for c in cells if c[7] == "ran"]
check(length(ranlines) == N_RAN, "correctness.csv: $(length(ranlines)) ran cells, not $N_RAN")
tpath = joinpath(DIR, "timings.csv")
tlines = readlines(tpath)
check(tlines[1] == TIMINGS_HEAD, "timings.csv: header is $(tlines[1])")
timings = [split(l, ",") for l in tlines[2:end]]
check(length(timings) == length(ranlines),
    "timings.csv: $(length(timings)) lines, not one per ran cell, $(length(ranlines))")
check([join(t[1:6], ",") for t in timings] == [join(c[1:6], ",") for c in ranlines],
    "timings.csv: the keys are not those of the ran cells in their order")
median_s = Dict{String, Float64}()
for t in timings
    where = "timings.csv: $(join(t[1:min(6, end)], ","))"
    check(length(t) == 15, "$where: $(length(t)) fields, not 15") || continue
    check_times(where, t[7:13])
    k = tryparse(Int, t[14])
    check(k !== nothing && k ≥ 1 && ispow2(k), "$where: solves_per_run $(t[14])")
    median_s[join(t[1:6], ",")] = something(tryparse(Float64, t[12]), NaN)
end

# --- Y2: the block tables --------------------------------------------------------------------

const BLOCK_HEAD = "block,set,problem,start,eltype,variant,package,status,method,code,success,iterations,f_evals,fnorm_inf,tol,t1,t2,t3,t4,t5,median_s,spread,comment"
const BLOCK_COUNTS = Dict("mixed" => 395, "stage" => 624, "anderson" => 1264, "sparse" =>
    20)
const BLOCK_OK = Dict(OK_CODES..., "SIAMFANLEquations" => ["0", "-1"])
const TOL64 = repr(sqrt(eps(Float64)))
const TOL32 = repr(sqrt(eps(Float32)))
const STAGE = repr(Float32(eps(Float16)))

# The label of each variant of the mixed, Anderson and sparse blocks.
const LABELS = Dict(
    "gs-f64" => "Newton+Backtracking+LUFactorization",
    "gs-f32" => "Newton+Backtracking+LUFactorization{Float32}",
    "gs-f32-refine" => "Newton+Backtracking+LUFactorization{Float32}+Refinement",
    "nsol-f32" => "nsol(sham=1)+FPS{Float32}",
    "nsol-mplu" => "nsol(sham=1)+mplu",
    "gs-sparse" => "Newton+Static+LUFactorization+AutoSparse+SparseMatrixCSC",
    "gs-dense" => "Newton+Static+LUFactorization+SharedJacobian",
    "nlsolve-sparse" => "newton+AnalyticJacobian+SparseMatrixCSC",
    ("anderson-$k" => "Anderson{4}(Newton($k)+Backtracking+LUFactorization)"
    for k in 1:5)...,
    ("newton-$k" => "Newton($k)+Backtracking+LUFactorization" for k in 1:5)...)

# The rows `(set, problem, start)` of a precision, and the columns, in the order of the cells.
rows_of(el) = unique([String.(c[1:3]) for c in cells if c[4] == el])
const COLS = unique([String(c[5]) for c in cells])

# The keys of each block, in the order of its table.
function block_keys(block)
    if block == "mixed"
        return [["mixed"; r; "Float64"; v; p] for r in rows_of("Float64")
                for (v, p) in (("gs-f64", "GeometricSolvers"), (
            "gs-f32", "GeometricSolvers"),
            ("gs-f32-refine", "GeometricSolvers"), ("nsol-f32", "SIAMFANLEquations"),
            ("nsol-mplu", "SIAMFANLEquations"))]
    elseif block == "stage"
        return [["stage"; r; "Float32"; col; p]
                for r in rows_of("Float32")
                if r[1] == "midpoint" for col in COLS for p in PKGS]
    elseif block == "anderson"
        return [["anderson"; r; el; "$v-$k"; "GeometricSolvers"]
                for el in ("Float32", "Float64") for r in rows_of(el) for k in (1, 2, 3, 5)
                for v in ("anderson", "newton")]
    end
    return [["sparse", "sparse", "bracket N=$N", "z0", el, v, p]
            for N in (125, 768) for el in ("Float32", "Float64")
            for (v, p) in (("gs-sparse", "GeometricSolvers"), (
        "gs-dense", "GeometricSolvers"),
        ("nlsolve-sparse", "NLsolve"), ("simplesolvers", "SimpleSolvers"),
        ("simplenonlinearsolve", "SimpleNonlinearSolve"))]
end

# The tolerance of a block cell.
block_tol(block, el) = block == "stage" ? STAGE : el == "Float32" ? TOL32 : TOL64

# The status of a block cell that does not run, or "ran".
function block_status(block, el, v, p)
    block == "stage" && p == "NLsolve" && return "no Float32"
    block == "stage" && get(pairing, (v, p), nothing) === nothing && return "no method"
    v in ("simplesolvers", "simplenonlinearsolve") && return "no sparse path"
    v == "nlsolve-sparse" && el == "Float32" && return "no Float32"
    return "ran"
end

btables = Dict{String, Vector{Vector{SubString{String}}}}()
for block in ("mixed", "stage", "anderson", "sparse")
    path = joinpath(DIR, "$block.csv")
    check(isfile(path), "$block.csv: missing") || continue
    bl = readlines(path)
    check(bl[1] == BLOCK_HEAD, "$block.csv: header is $(bl[1])")
    bc = [split(l, ",") for l in bl[2:end]]
    check(all(c -> length(c) == 23, bc), "$block.csv: a line without 23 fields")
    bc = filter(c -> length(c) == 23, bc)
    btables[block] = bc
    check(length(bc) == BLOCK_COUNTS[block],
        "$block.csv: $(length(bc)) cells, not $(BLOCK_COUNTS[block])")
    check([String.(c[1:7]) for c in bc] == block_keys(block),
        "$block.csv: the keys are not the block's, each once, in order")
    for c in bc
        where = "$block.csv: $(join(c[1:7], ","))"
        _, set, problem, start, el, v, p, status, method,
        code, succ, iters, fe, fn, tol = c[1:15]
        expected = block_status(block, el, v, p)
        check(status == expected, "$where: status $status, not $expected")
        if status != "ran"
            check(all(isempty, c[9:22]) && !isempty(c[23]),
                "$where: a cell that did not run has fields or no reason")
            continue
        end
        label = block == "stage" ? something(get(pairing, (v, p), nothing), "") : LABELS[v]
        check(method == label, "$where: method $method, not $label")
        check(tol == block_tol(block, el), "$where: tol $tol, not $(block_tol(block, el))")
        check(
            iscount(fe) && (code == "ERROR" ? isempty(iters) :
             p == "SimpleNonlinearSolve" ? iters == "n/a" : iscount(iters)),
            "$where: iterations '$iters', f_evals '$fe'")
        T = eltype_of(el)
        ok = if code == "ERROR"
            check(isempty(fn) && !isempty(c[23]), "$where: an ERROR with a result")
            false
        else
            code in BLOCK_OK[p] && number(fn, T) ≤ number(tol, T)
        end
        check(succ == string(ok), "$where: success $succ, recomputed $ok")
        check_times(where, c[16:22])
    end
end

# --- Y2: the report's tables -------------------------------------------------------------------

# The rows of the Markdown table under the line `heading`, without its head line, each the vector
# of its fields.
function y2_table(heading)
    i = findfirst(==(heading), report)
    i === nothing && (push!(failures, "REPORT.md: no heading $heading"); return [])
    out = []
    for l in report[(i + 1):end]
        startswith(l, "#") && break
        startswith(l, "|") || continue
        fields = strip.(split(strip(l, '|'), "|"))
        all(f -> occursin(r"^:?-+:?$", f), fields) && continue
        push!(out, String.(fields))
    end
    return isempty(out) ? out : out[2:end]
end

function compare_table(name, got, want)
    check(length(got) == length(want),
        "REPORT.md: $name has $(length(got)) lines, recomputed $(length(want))") || return
    for (g, w) in zip(got, want)
        check(g == w, "REPORT.md: $name has '$(join(g, " | "))', recomputed '$(join(w, " | "))'")
    end
end

# A share with three decimals, and a positive `x` with three significant digits, trailing zeros
# kept and a trailing point dropped; `—` for `NaN`, the median of no ratio.
d3(x) = @sprintf("%.3f", x)
function s3(x)
    isnan(x) && return "—"
    s = @sprintf("%#.3g", x)
    return endswith(s, ".") ? chop(s) : s
end

function median_any(v)
    isempty(v) ? NaN :
    (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
end

# A cell as a named view: `success`, `iterations`, `f_evals` and `median_s`.
function corr_view(c)
    (; ok = c[10] == "true", iterations = String(c[11]), f_evals = String(c[12]),
        median_s = get(median_s, join(c[1:6], ","), NaN))
end
function block_view(c)
    (; ok = c[11] == "true", iterations = String(c[12]), f_evals = String(c[13]),
        median_s = something(tryparse(Float64, c[21]), NaN))
end

function ratio_median(pairs, metric)
    v = Float64[]
    for (a, b) in pairs
        x, y = getfield(a, metric), getfield(b, metric)
        if metric == :median_s
            push!(v, x / y)
        else
            (x == "n/a" || y == "n/a") && return NaN
            push!(v, max(parse(Float64, x), 1) / max(parse(Float64, y), 1))
        end
    end
    return median_any(v)
end
const MS = (:iterations, :f_evals, :median_s)

!STRUCTURE && length(btables) == 4 &&
    let
        bykey = Dict(join(c[1:6], ",") => c for c in cells)
        want = []
        # The profiles.
        for el in ("Float32", "Float64"), metric in MS

            solvers = filter(
                p -> !(p == "NLsolve" && el == "Float32") &&
                     !(p == "SimpleNonlinearSolve" && metric == :iterations),
                PKGS)
            pairs = unique([join(c[1:5], ",") for c in cells if c[4] == el])
            check(length(pairs) == 1027, "correctness.csv: $(length(pairs)) pairs in $el, not 1027")
            t = fill(Inf, length(pairs), length(solvers))
            for (i, pr) in enumerate(pairs), (j, s) in enumerate(solvers)

                c = bykey["$pr,$s"]
                (c[7] == "ran" && c[10] == "true") || continue
                v = getfield(corr_view(c), metric)
                t[i, j] = metric == :median_s ? v : max(parse(Float64, v), 1)
            end
            want = [[s;
                     [begin
                          r = [isinf(t[i, j]) ? Inf : t[i, j] / minimum(t[i, :])
                               for i in axes(t, 1)]
                          d3(count(x -> isfinite(x) && x ≤ τ, r) / length(pairs))
                      end
                      for τ in (1, 2, 4, 8, 16, Inf)]] for (j, s) in enumerate(solvers)]
            compare_table("the profile $el $metric", y2_table("#### Profile · $el · $metric"), want)
        end
        # DogLeg against Backtracking.
        want = []
        for p in ("GeometricSolvers", "SimpleSolvers", "NLsolve"),
            el in ("Float32", "Float64")

            p == "NLsolve" && el == "Float32" && continue
            dl = [bykey[join([r; el; "Newton DogLeg LU"; p], ",")] for r in rows_of(el)]
            bt = [bykey[join([r; el; "Newton Backtracking LU"; p], ",")]
                  for r in rows_of(el)]
            both = [(corr_view(a), corr_view(b))
                    for (a, b) in zip(dl, bt)
                    if a[10] == "true" && b[10] == "true"]
            push!(want,
                [p, el, string(count(c -> c[10] == "true", dl)),
                    string(count(c -> c[10] == "true", bt)), string(length(both)),
                    (s3(ratio_median(both, m)) for m in MS)...])
        end
        compare_table("the DogLeg table", y2_table("### Y2 · DogLeg against Backtracking"), want)
        # Mixed precision.
        mixed = btables["mixed"]
        base = Dict(join(c[2:4], ",") => c for c in mixed if c[6] == "gs-f64")
        want = []
        for v in ("gs-f64", "gs-f32", "gs-f32-refine", "nsol-f32", "nsol-mplu")
            cs = [c for c in mixed if c[6] == v]
            both = [(block_view(c), block_view(base[join(c[2:4], ",")]))
                    for c in cs
                    if c[11] == "true" && base[join(c[2:4], ",")][11] == "true"]
            push!(want,
                [v, "`$(LABELS[v])`", string(count(c -> c[11] == "true", cs)),
                    string(length(both)), s3(ratio_median(both, :iterations)),
                    s3(ratio_median(both, :median_s))])
        end
        compare_table("the mixed-precision table", y2_table("### Y2 · Mixed precision"), want)
        # The stage tolerance.
        want = []
        for p in ("GeometricSolvers", "SimpleSolvers", "SimpleNonlinearSolve")
            loose = [c for c in btables["stage"] if c[7] == p]
            tight = [bykey[join([c[2:5]; c[6]; p], ",")] for c in loose]
            both = [(block_view(a), corr_view(b))
                    for (a, b) in zip(loose, tight)
                    if a[11] == "true" && b[10] == "true"]
            push!(want,
                [p, string(count(c -> c[11] == "true", loose)),
                    string(count(c -> c[10] == "true", tight)), string(length(both)),
                    (s3(ratio_median(both, m)) for m in MS)...])
        end
        compare_table("the stage-tolerance table", y2_table("### Y2 · Stage tolerance"), want)
        # Anderson.
        want = []
        for el in ("Float32", "Float64"), k in (1, 2, 3, 5)

            a = [c for c in btables["anderson"] if c[5] == el && c[6] == "anderson-$k"]
            n = [c for c in btables["anderson"] if c[5] == el && c[6] == "newton-$k"]
            both = [(block_view(x), block_view(y))
                    for (x, y) in zip(a, n)
                    if x[11] == "true" && y[11] == "true"]
            push!(want,
                [el, string(k), string(count(c -> c[11] == "true", a)),
                    string(count(c -> c[11] == "true", n)), string(length(both)),
                    (s3(ratio_median(both, m)) for m in MS)...])
        end
        compare_table("the Anderson table", y2_table("### Y2 · Anderson"), want)
        # The sparse problem.
        want = [[c[6], last(split(c[3], "=")), c[5], c[8], c[10], c[12], c[13],
                    isempty(c[21]) ? "" : s3(parse(Float64, c[21]))]
                for c in btables["sparse"]]
        compare_table("the sparse table", y2_table("### Y2 · The sparse problem"), want)
        # The count of timed cells whose warm-up differs from correctness.csv.
        nd = count(t -> occursin("differs from correctness.csv", last(t)), timings)
        check(any(l -> occursin("$nd of the $N_RAN timed cells", l), report),
            "REPORT.md: no line states '$nd of the $N_RAN timed cells'")
    end

if isempty(failures)
    println("VERIFY OK: $(length(cells)) cells, $(length(rows)) rows, ",
        "$(length(timings)) timings, ",
        join(
            ("$(length(get(btables, b, []))) $b"
            for b in ("mixed", "stage", "anderson", "sparse")),
            ", "), " block cells", STRUCTURE ? " (structure)" : "")
else
    foreach(f -> println("FAIL ", f), first(failures, 50))
    length(failures) > 50 && println("… and $(length(failures) - 50) more")
    exit(1)
end
