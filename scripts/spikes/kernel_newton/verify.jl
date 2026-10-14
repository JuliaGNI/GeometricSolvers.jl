# verify.jl — re-derive every cell of the §10.E tables from the raw output of a run.
#
# Usage:
#     julia --startup-file=no --project=. verify.jl <results dir> <card> [storage] [band]
#
# <card> is the machine slug of test/gpu/README.md: it names the two files the run wrote,
# kernel_newton-<card>-gpu.txt and kernel_newton-<card>-cpu.txt. Each file's Card: line names the
# card itself, and the script records it with the run. [storage] is the §5.2 statement to confirm,
# either MMatrix or SMatrix. [band] is the §1.1 band to confirm, a number of systems or ge24.
# Without them the script prints the derived values and compares nothing.
#
# Exit status is 0 when every check passes and 1 otherwise.

const NS = (2, 4, 6, 8, 12, 16, 24)
const NSYS = (10^4, 10^5, 10^6)
const TIMING_COLUMNS = ("solves_1e4", "solves_1e5", "solves_1e6")
const COLUMNS = ("compiles", "answer", "allocates", "compile_s", TIMING_COLUMNS...)
const MARKS = (
    r"^does not compile: .+$",
    r"^wrong: maxiters$",
    r"^wrong: (?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$",
    r"^not run: compile cap$",
    r"^not timed: device Float64$"
)
const COMPILE_CAP_S = 60.0

struct Run
    card::String
    backend::String
    julia::String
    packages::String
    cardline::String
    rows::Vector{Dict{String, String}}
end

function parse_fields(line::AbstractString)
    Dict{String, String}(m.captures[1] => m.captures[2]
    for m in eachmatch(r"(\w+)=\"([^\"]*)\"", line))
end

# A missing file is an error that names it, never an empty table (§10.E, decided at the edges).
function read_run(path::String, card::String)
    isfile(path) || error("the raw output file $path is missing; the run did not push it")
    backend = ""
    julia = ""
    packages = ""
    cardline = ""
    rows = Dict{String, String}[]
    for line in eachline(path)
        if occursin("kernel_newton spike -- backend = ", line)
            backend = rstrip(split(line, "backend = ")[end])
        elseif startswith(line, "Julia ")
            julia = String(strip(line[length("Julia "):end]))
        elseif startswith(line, "Packages: ")
            packages = line[length("Packages: "):end]
        elseif startswith(line, "Card: ")
            cardline = strip(line[length("Card: "):end])
        elseif startswith(line, "ROW ")
            push!(rows, parse_fields(line))
        end
    end
    isempty(backend) && error("$path holds no spike header; it is not a run of this spike")
    isempty(cardline) && error("$path names no card; the run did not record the machine")
    return Run(card, backend, julia, packages, cardline, rows)
end
struct Checks
    failures::Vector{String}
    notes::Vector{String}
end
Checks() = Checks(String[], String[])
fail!(c::Checks, text) = (push!(c.failures, text); c)
note!(c::Checks, text) = (push!(c.notes, text); c)

is_mark(value::AbstractString) = any(m -> occursin(m, value), MARKS)
label(r::Dict{String, String}) = r["variant"] * " " * r["tmode"] * " n=" * r["n"]
function median(v)
    (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
end
first_five(v) = parse.(Float64, split(first(split(v)), ","))
# --- 1. the answer ------------------------------------------------------------------------------
#
# Each solve is checked against its own x* to the bound 16 n eps(T) xstar_inf, and against the CPU
# result with the same bound. A wrong answer marks the cell and keeps it.
function check_answer!(c::Checks, run::Run)
    for r in run.rows
        is_mark(get(r, "answer", "")) && !haskey(r, "err") && continue
        fields = ("n", "xstar_inf", "bound", "err", "err_cpu")
        if any(k -> tryparse(Float64, get(r, k, "")) === nothing, fields)
            fail!(c, "$(run.backend) $(label(r)): the answer has missing or nonnumeric evidence")
            continue
        end
        n = parse(Int, r["n"])
        T = r["tmode"] == "F32" ? Float32 : Float64
        bound = 16 * n * eps(T) * parse(Float64, r["xstar_inf"])
        err = parse(Float64, r["err"])
        err_cpu = parse(Float64, r["err_cpu"])
        if !all(isfinite, (bound, err, err_cpu, parse(Float64, r["bound"]))) ||
           bound <= 0 || min(err, err_cpu) < 0
            fail!(c, "$(run.backend) $(label(r)): answer evidence must be finite with a positive bound")
            continue
        end
        isapprox(parse(Float64, r["bound"]), bound; rtol = 1e-6) ||
            fail!(c,
                "$(run.backend) $(label(r)): the bound $(r["bound"]) is not 16 n eps(T) xstar_inf = $bound")
        if err <= bound && err_cpu <= bound
            r["answer"] == "right" ||
                fail!(c,
                    "$(run.backend) $(label(r)): the answer reads '$(r["answer"])' but err $err and err_cpu $err_cpu are within $bound")
        else
            startswith(r["answer"], "wrong: ") ||
                fail!(c, "$(run.backend) $(label(r)): the answer is wrong and the cell reads '$(r["answer"])'")
            note!(c, "$(run.backend) $(label(r)): a wrong answer is kept as the cell's mark")
        end
    end
    return c
end

# --- 2. allocates --------------------------------------------------------------------------------
#
# An allocation cell with no method means nothing. The cell names its method and reads "no" only
# for a count of exactly zero.
function check_allocates!(c::Checks, run::Run)
    for r in run.rows
        haskey(r, "allocates") || continue
        is_mark(r["allocates"]) && continue
        isempty(get(r, "alloc_method", "")) &&
            fail!(c, "$(run.backend) $(label(r)): the allocates cell names no method")
        count = tryparse(Int, get(r, "alloc_count", ""))
        count === nothing &&
            fail!(c,
                "$(run.backend) $(label(r)): the allocator count '$(get(r, "alloc_count", ""))' is not a number")
        method = get(r, "alloc_method", "")
        if method == "host_allocated_function_barrier"
            samples = tryparse.(Int, split(get(r, "alloc_samples", ""), ","))
            length(samples) == 5 && all(x -> x !== nothing && x >= 0, samples) &&
            minimum(samples) == count ||
                fail!(c, "$(run.backend) $(label(r)): allocation count does not match five raw samples")
        elseif method == "device_llvm_allocator_calls"
            calls = get(r, "alloc_calls", nothing)
            calls !== nothing &&
            (isempty(calls) ? 0 : length(split(calls, "|"))) == count ||
                fail!(c, "$(run.backend) $(label(r)): allocation count does not match the LLVM call sites")
        elseif method == "device_compile_allocator_error"
            occursin(
                r"jl_gc|gc_alloc|malloc|pool_alloc|jl_alloc_genericmemory|unsupported allocation",
                get(r, "alloc_error", "")) &&
            count == 1 ||
                fail!(c, "$(run.backend) $(label(r)): allocator compile failure has no allocator evidence")
        else
            fail!(c, "$(run.backend) $(label(r)): unknown allocation method '$method'")
        end
        if r["allocates"] == "no"
            count == 0 ||
                fail!(c, "$(run.backend) $(label(r)): allocates reads no with a count of $count")
        elseif r["allocates"] == "yes"
            count === nothing || count > 0 ||
                fail!(c, "$(run.backend) $(label(r)): allocates reads yes with a count of $count")
        else
            is_mark(r["allocates"]) ||
                fail!(c,
                    "$(run.backend) $(label(r)): the allocates cell '$(r["allocates"])' is not yes, no or a mark")
        end
    end
    return c
end
# --- 3. the grid has no hole ----------------------------------------------------------------------
#
# Every expected row is there once, and each of the seven columns has a value or a mark.
function check_grid!(c::Checks, run::Run)
    pairs = [("M", "F32"), ("S", "F32"), ("SNS", "F32")]
    if run.backend in ("cuda", "rocm")
        append!(pairs, [("M", "F64"), ("S", "F64")])
    end
    if run.backend == "cuda"
        append!(pairs, [("M", "MIXED"), ("S", "MIXED")])
    end
    expected = Set(v * " " * t * " " * string(n) for (v, t) in pairs for n in NS)
    seen = Set{String}()
    for r in run.rows
        key = get(r, "variant", "") * " " * get(r, "tmode", "") * " " * get(r, "n", "")
        key in seen && fail!(c, "$(run.backend): the row $key appears twice")
        key in expected || fail!(c, "$(run.backend): unexpected row $key")
        push!(seen, key)
        for col in COLUMNS
            value = get(r, col, "")
            if isempty(value)
                fail!(c, "$(run.backend) $key: the column $col is empty")
            elseif is_mark(value)
                continue
            elseif col == "compiles" && value != "yes" ||
                   col == "answer" && value != "right" ||
                   col == "allocates" && !(value in ("yes", "no")) ||
                   col == "compile_s" && (tryparse(Float64, value) === nothing ||
                    !isfinite(parse(Float64, value)) || parse(Float64, value) < 0)
                fail!(c, "$(run.backend) $key: invalid $col value '$value'")
            end
        end
    end
    for key in sort!(collect(setdiff(expected, seen)))
        fail!(c, "$(run.backend): missing row $key ($(length(COLUMNS)) empty cells)")
    end
    return c
end
function check_timings!(c::Checks, run::Run)
    is_device = run.backend != "cpu"
    for r in run.rows
        if get(r, "compiles", "") == "yes"
            first_launch = tryparse(Float64, get(r, "first_launch_s", ""))
            warm = tryparse.(Float64, split(get(r, "warm_s", ""), ","))
            compile = tryparse(Float64, get(r, "compile_s", ""))
            if first_launch === nothing || compile === nothing || length(warm) != 5 ||
               any(t -> t === nothing || !isfinite(t) || t <= 0, warm)
                fail!(c, "$(run.backend) $(label(r)): compile-time subtraction has missing evidence")
            elseif !isapprox(compile, max(0.0, first_launch - median(Float64.(warm))); rtol = 1e-6, atol = 1e-7)
                fail!(c, "$(run.backend) $(label(r)): compile time is not first launch minus the warm median")
            end
            if run.backend == "cpu" && (tryparse(Int, get(r, "threads", "")) === nothing ||
                get(r, "blas_threads", "") != "1")
                fail!(c, "cpu $(label(r)): timing lacks host threads or single-threaded BLAS")
            end
        end
        for (cell, nsys) in zip(TIMING_COLUMNS, NSYS)
            value = get(r, cell, "")
            isempty(value) && continue
            if is_device && r["tmode"] == "F64" && get(r, "compiles", "") == "yes" &&
               value != "not timed: device Float64" && value != "not run: compile cap"
                fail!(c, "$(run.backend) $(label(r)): device Float64 throughput must be untimed")
            end
            is_mark(value) && continue
            parts = split(value)
            if length(parts) != 2 || !startswith(parts[2], "solves_per_s=")
                fail!(c, "$(run.backend) $(label(r)): $cell does not hold five values and a rate")
                continue
            end
            raw = tryparse.(Float64, split(parts[1], ","))
            if length(raw) != 5 || any(v -> v === nothing || !isfinite(v) || v <= 0, raw)
                fail!(c, "$(run.backend) $(label(r)): $cell needs five finite positive raw values")
                continue
            end
            values = Float64.(raw)
            med = median(values)
            spread = (maximum(values) - minimum(values)) / med
            recorded_spread = tryparse(Float64, get(r, "spread_" * cell, ""))
            recorded_spread !== nothing &&
            isapprox(recorded_spread, spread; rtol = 1e-3, atol = 1e-7) ||
                fail!(c, "$(run.backend) $(label(r)): $cell spread does not match its five raw values")
            rate = tryparse(Float64, split(parts[2], "="; limit = 2)[2])
            recorded = tryparse(Float64, get(r, "rate_" * cell, ""))
            for x in (rate, recorded)
                x !== nothing && isapprox(x, nsys / med; rtol = 1e-4) ||
                    fail!(c, "$(run.backend) $(label(r)): $cell rate does not match its five raw values")
            end
            if is_device && r["tmode"] == "F64"
                fail!(c, "$(run.backend) $(label(r)): a device Float64 cell holds a timing")
            end
        end
    end
    return c
end
function check_rules!(c::Checks, runs::Vector{Run}, storage::String, band::String)
    cpuidx = findfirst(r -> r.backend == "cpu", runs)
    heap = String[]
    for run in runs
        run.backend == "cpu" && continue
        rows_M = [r for r in run.rows if r["variant"] == "M" && r["tmode"] == "F32"]
        any(r -> r["allocates"] == "yes", rows_M) && push!(heap, run.backend)
        n_cpu = 0
        n_cap = 0
        for r in run.rows
            r["tmode"] != "F32" && continue
            n = parse(Int, r["n"])
            s = tryparse(Float64, get(r, "compile_s", ""))
            if s !== nothing && s > COMPILE_CAP_S
                n_cap = n_cap == 0 ? n : min(n_cap, n)
            end
            cpuidx === nothing && continue
            host = findfirst(
                x -> x["variant"] == r["variant"] && x["tmode"] == "F32" &&
                     x["n"] == r["n"],
                runs[cpuidx].rows)
            dev = get(r, "solves_1e6", "")
            cpu = host === nothing ? "" : get(runs[cpuidx].rows[host], "solves_1e6", "")
            if host !== nothing && !isempty(dev) && !isempty(cpu) && !is_mark(dev) &&
               !is_mark(cpu) &&
               maximum(first_five(cpu)) < minimum(first_five(dev))
                # The fastest-CPU/slowest-device clause is necessary; the overlap-is-a-tie
                # clause requires disjoint ranges, so the CPU's slowest beats the device's fastest.
                n_cpu = n_cpu == 0 ? n : min(n_cpu, n)
            end
        end
        band_n = 0
        if n_cpu > 0 && n_cap > 0
            band_n = min(n_cpu, n_cap)
        elseif n_cpu > 0
            band_n = n_cpu
        elseif n_cap > 0
            band_n = n_cap
        end
        text = band_n == 0 ? "ge24" : string(band_n)
        note!(c,
            "$(run.backend): the CPU loop wins from n=$n_cpu, compile time passes one minute at n=$n_cap, so the R1 band is " *
            text * (run.backend == "rocm" ? " (provisional, part G)" : ""))
        if band == "ge24"
            text == "ge24" || fail!(c, "$(run.backend): the band is $text, not ge24")
        elseif !isempty(band)
            want = tryparse(Int, band)
            want === nothing ?
            fail!(c, "$(run.backend): the band argument '$band' is not a number or ge24") :
            (band_n == want || fail!(c, "$(run.backend): the band is $text, not $want"))
        end
        if all(r -> r["allocates"] == "no", rows_M)
            choose_M = false
            for m in rows_M
                n = parse(Int, m["n"])
                band_n > 0 && n >= band_n && continue
                idx = findfirst(
                    r -> r["variant"] == "S" && r["tmode"] == "F32" &&
                         r["n"] == m["n"], run.rows)
                idx === nothing && continue
                srow = run.rows[idx]
                sc = tryparse(Float64, get(srow, "compile_s", ""))
                choose_M |= sc !== nothing && sc > COMPILE_CAP_S
                mt = get(m, "solves_1e6", "")
                st = get(srow, "solves_1e6", "")
                if !isempty(mt) && !isempty(st) && !is_mark(mt) && !is_mark(st)
                    choose_M |= maximum(first_five(mt)) < minimum(first_five(st))
                end
            end
            derived = choose_M ? "MMatrix" : "SMatrix"
            note!(c, "$(run.backend): off-heap storage rule gives $derived")
            isempty(storage) || storage in (derived, "per-backend") ||
                fail!(c, "$(run.backend): off-heap storage rule gives $derived, not '$storage'")
        end
    end
    if length(heap) == 1
        note!(c, "MMatrix is on the heap on $(heap[1]) only, so §5.2 must say SMatrix or a per-backend choice")
        if !isempty(storage)
            storage in ("SMatrix", "per-backend") ||
                fail!(c, "MMatrix is on the heap on $(heap[1]) only, and §5.2 says '$storage'")
        end
    elseif isempty(heap)
        note!(c, "MMatrix stays off the heap on every backend measured here")
    else
        note!(c, "MMatrix is on the heap on $(join(heap, ", "))")
    end
    return c
end
function main(args)
    args == ["--self-check"] && return self_check()
    length(args) >= 2 || error("usage: verify.jl <results dir> <card> [storage] [band]")
    dir, card = args[1], args[2]
    storage = length(args) >= 3 ? args[3] : ""
    band = length(args) >= 4 ? args[4] : ""
    runs = Run[]
    for suffix in ("gpu", "cpu")
        push!(runs, read_run(joinpath(dir, "kernel_newton-$(card)-$(suffix).txt"), card))
    end
    c = Checks()
    for run in runs
        check_answer!(c, run)
        check_allocates!(c, run)
        check_grid!(c, run)
        check_timings!(c, run)
    end
    isempty(c.failures) || return report(c)
    check_rules!(c, runs, storage, band)
    return report(c)
end

function report(c::Checks)
    for note in c.notes
        println("note: ", note)
    end
    for failure in c.failures
        println("FAIL: ", failure)
    end
    println("checks: ", length(c.failures), " failure(s), ", length(c.notes), " note(s)")
    return isempty(c.failures) ? 0 : 1
end

function self_check()
    device = [Dict("variant" => "M", "tmode" => "F32", "n" => string(n),
                  "compile_s" => n == 8 ? "61" : "1", "allocates" => "no",
                  "solves_1e6" =>
                      n == 8 ? "not run: compile cap" : "2,3,4,5,6 solves_per_s=250000")
              for n in (2, 4, 8)]
    cpu = [Dict("variant" => "M", "tmode" => "F32", "n" => string(n),
               "solves_1e6" =>
                   n == 2 ? "3,4,5,6,7 solves_per_s=200000" :
                   "1,1,1,1,1 solves_per_s=1000000")
           for n in (2, 4, 8)]
    runs = [Run("fixture", "metal", "fixture", "fixture", "fixture", device),
        Run("fixture", "cpu", "fixture", "fixture", "fixture", cpu)]
    c = Checks()
    check_rules!(c, runs, "", "4")
    @assert isempty(c.failures) "overlap is a tie, and disjoint CPU timings give band 4"
    c_reversed = Checks()
    reversed_runs = [Run(r.card, r.backend, r.julia, r.packages,
                         r.cardline, reverse(r.rows)) for r in runs]
    check_rules!(c_reversed, reversed_runs, "", "4")
    @assert isempty(c_reversed.failures) "row ordering does not affect the first n"

    # Flip only the CPU-wins comparator; the same fixture must change the derived band to 2.
    conservative = "maximum(" * "first_five(cpu)) < minimum(first_five(dev))"
    overlapping = "minimum(" * "first_five(cpu)) < maximum(first_five(dev))"
    source = read(@__FILE__, String)
    @assert count(conservative, source) == 1
    mutant = Module(gensym(:KernelNewtonComparatorMutant))
    Base.include_string(mutant, replace(source, conservative => overlapping))
    mutant_Run = Base.invokelatest(getfield, mutant, :Run)
    mutant_runs = [Base.invokelatest(mutant_Run, r.card, r.backend, r.julia,
                       r.packages, r.cardline, r.rows) for r in runs]
    mutant_checks = Base.invokelatest(Base.invokelatest(getfield, mutant, :Checks))
    Base.invokelatest(Base.invokelatest(getfield, mutant, :check_rules!),
        mutant_checks, mutant_runs, "", "4")
    @assert any(s -> occursin("the band is 2, not 4", s), getfield(mutant_checks, :failures))
    println("self-check: conservative band 4; overlapping-comparator mutant band 2; row-order check passed")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    iszero(main(ARGS)) || error("kernel_newton verification failed")
end
