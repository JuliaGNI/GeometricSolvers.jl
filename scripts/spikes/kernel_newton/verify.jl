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
    r"^wrong: [0-9.eE+-].*$",
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
        haskey(r, "err") || continue
        n = parse(Int, r["n"])
        T = r["tmode"] == "F32" ? Float32 : Float64
        bound = 16 * n * eps(T) * parse(Float64, r["xstar_inf"])
        err = parse(Float64, r["err"])
        err_cpu = parse(Float64, r["err_cpu"])
        isapprox(parse(Float64, r["bound"]), bound; rtol = 1e-6) ||
            fail!(c,
                "$(run.backend) $(label(r)): the bound $(r["bound"]) is not 16 n eps(T) xstar_inf = $bound")
        if err <= bound && err_cpu <= bound
            r["answer"] == "right" ||
                fail!(c,
                    "$(run.backend) $(label(r)): the answer reads '$(r["answer"])' but err $err and err_cpu $err_cpu are within $bound")
        else
            occursin(r"^wrong: ", r["answer"]) ||
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
        isempty(get(r, "alloc_method", "")) &&
            fail!(c, "$(run.backend) $(label(r)): the allocates cell names no method")
        count = tryparse(Int, get(r, "alloc_count", ""))
        count === nothing &&
            fail!(c,
                "$(run.backend) $(label(r)): the allocator count '$(get(r, "alloc_count", ""))' is not a number")
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
    seen = Set{String}()
    for r in run.rows
        key = r["variant"] * " " * r["tmode"] * " " * r["n"]
        key in seen && fail!(c, "$(run.backend): the row $key appears twice")
        push!(seen, key)
        for col in COLUMNS
            value = get(r, col, nothing)
            if value === nothing || isempty(value)
                fail!(c, "$(run.backend) $key: the column $col is empty")
            elseif !is_mark(value)
                nothing
            end
        end
    end
    return c
end
function check_timings!(c::Checks, run::Run)
    is_device = run.backend != "cpu"
    for r in run.rows
        for (cell, nsys) in zip(TIMING_COLUMNS, NSYS)
            value = get(r, cell, nothing)
            value === nothing && continue
            is_mark(value) && continue
            parts = split(value)
            length(parts) == 2 && startswith(parts[2], "solves_per_s=") ||
                fail!(c, "$(run.backend) $(label(r)): $cell does not hold five values and a rate")
            values = tryparse.(Float64, split(parts[1], ","))
            if any(v -> v === nothing, values) || length(values) != 5
                fail!(c, "$(run.backend) $(label(r)): $cell holds $(length(values)) raw values, not 5")
                continue
            end
            values = convert(Vector{Float64}, values)
            med = median(values)
            spread = (maximum(values) - minimum(values)) / med
            isapprox(parse(Float64, get(r, "spread_" * cell, "")), spread; rtol = 1e-3) ||
                fail!(c,
                    "$(run.backend) $(label(r)): $cell records the spread $(get(r, "spread_" * cell, "")) but the five raw values give $spread")
            rate = tryparse(Float64, split(parts[2], "=")[2])
            rate === nothing || isapprox(rate, nsys / med; rtol = 1e-4) ||
                fail!(c,
                    "$(run.backend) $(label(r)): $cell records $rate solves per second, the five raw values give $(nsys / med)")
            recorded = tryparse(Float64, get(r, "rate_" * cell, ""))
            recorded === nothing || isapprox(recorded, nsys / med; rtol = 1e-4) ||
                fail!(c,
                    "$(run.backend) $(label(r)): $cell records $(get(r, "rate_" * cell, "")) solves per second, the five raw values give $(nsys / med)")
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
            if n_cap == 0
                s = tryparse(Float64, get(r, "compile_s", ""))
                if s !== nothing && s > COMPILE_CAP_S
                    n_cap = parse(Int, r["n"])
                end
            end
            if n_cpu == 0 && cpuidx !== nothing
                host = findfirst(x -> x["variant"] == r["variant"] && x["n"] == r["n"],
                    runs[cpuidx].rows)
                dev = get(r, "solves_1e6", "")
                cpu = host === nothing ? "" : get(runs[cpuidx].rows[host], "solves_1e6", "")
                if host !== nothing && !is_mark(dev) && !is_mark(cpu)
                    # the CPU loop wins when its fastest of five beats the device's slowest of five
                    if minimum(first_five(cpu)) < maximum(first_five(dev))
                        n_cpu = parse(Int, r["n"])
                    end
                end
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
    check_rules!(c, runs, storage, band)
    for note in c.notes
        println("note: ", note)
    end
    for failure in c.failures
        println("FAIL: ", failure)
    end
    println("checks: ", length(c.failures), " failure(s), ", length(c.notes), " note(s)")
    return isempty(c.failures) ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
