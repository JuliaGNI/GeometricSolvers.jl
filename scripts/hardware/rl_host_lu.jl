# The host LU of `LUFactorization`: LAPACK's `getrf!` against the recursive LU `recursive_lu!` of
# `src/linear/recursive_lu.jl`, on a random `n × n` matrix, in `Float32`, `Float64` and
# `ComplexF64`, with BLAS on one thread and on all threads.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=<env> scripts/hardware/rl_host_lu.jl \
#       [threads] [base] [n,…] [reps]
#
# `threads` is `both` by default: the script runs itself in two cold processes, BLAS on one thread
# (`1`) and on `Sys.CPU_THREADS` threads (`all`), prints both tables and one verdict line. `1` or
# `all` measures that setting alone, in this process, and gives no verdict. `base` is the largest
# panel that the recursion factorises with its unblocked kernel, 16 by default; `n,…` is
# `8,16,32,64,128,256,512,1024` by default; `reps` is the number of timed runs after one warm-up
# run, 5 by default. `<env>` holds GeometricSolvers and KernelAbstractions.
#
# Each column times a copy of the matrix into the factor buffer plus the factorisation. A timed run
# repeats that `k` times, with `k` chosen so that one run takes at least 1 ms, and the column gives
# the median and the minimum over the runs of the time of one call, in ms. A row whose median
# exceeds its minimum by more than 50 % in either column is measured once more, and its note says
# so. `r` is the median of the recursion over the median of `getrf!`. `berr` is the backward error
# `‖A[p, :] - L U‖∞ / ‖A‖∞` of each factor, in `Float64` for a `Float32` matrix; `ipiv` says
# whether the recursion's pivots equal those of `getrf!`.
#
# The verdict applies the rules decided before the measurement. Build rule: in `Float64`,
# `r ≤ 0.75` at two or more consecutive `n` in both thread settings, and `berr ≤ 8n·eps` in every
# row. Stop rule: no `Float64` `n` with `r ≤ 0.75` in both settings, or a row with
# `berr > 8n·eps`. Any other result is between the two. The band of a type is a run of consecutive
# `n` with `r ≤ 0.90` in both settings.

using GeometricSolvers
using GeometricSolvers: recursive_lu!
using KernelAbstractions: KernelAbstractions
using LinearAlgebra
using LinearAlgebra: BlasInt
using Printf
using Random: Xoshiro

include(joinpath(@__DIR__, "common.jl"))

const RL_ELTYPES = (Float32, Float64, ComplexF64)
const RL_NS = [8, 16, 32, 64, 128, 256, 512, 1024]
const RL_SETTINGS = ("1", "all")
const RL_TYPE_NAMES = Dict(string(T) => T for T in RL_ELTYPES)

blas_library() = join((basename(l.libname) for l in BLAS.get_config().loaded_libs), ", ")

# The time of one call of `kernel!(B, ipiv)` after `copyto!(B, A)`, median and minimum over
# `reps` runs of `k` calls each, `k` such that one run takes at least 1 ms at the median time of
# three calls.
function time_column(kernel!::K, A, B, ipiv; reps::Int) where {K}
    call = () -> (copyto!(B, A); kernel!(B, ipiv); nothing)
    call()
    t1 = sample_median([(t0 = time_ns(); call(); (time_ns() - t0) / 1e9) for _ in 1:3])
    k = max(1, ceil(Int, 1e-3 / max(t1, 1e-9)))
    t = time_calls(() -> (for _ in 1:k
            call()
        end), () -> nothing; reps)
    return (median = t.median / k, min = t.min / k)
end

noisy(t) = t.median > 1.5 * t.min

# The permutation `p` of the rows that the exchanges `ipiv` apply, so that `A[p, :] = L U`.
function row_permutation(ipiv::AbstractVector)
    p = collect(1:length(ipiv))
    for k in eachindex(ipiv)
        q = Int(ipiv[k])
        p[k], p[q] = p[q], p[k]
    end
    return p
end

# `‖A[p, :] - L U‖∞ / ‖A‖∞` of the factors `F` with the pivots `ipiv`, in `Float64` precision or
# wider.
function backward_error(A::AbstractMatrix{T}, F::AbstractMatrix{T}, ipiv) where {T}
    W = promote_type(T, Float64)
    L = UnitLowerTriangular(W.(F))
    U = UpperTriangular(W.(F))
    p = row_permutation(ipiv)
    return opnorm(W.(A[p, :]) - L * U, Inf) / opnorm(W.(A), Inf)
end

function measure_once(A, base; reps)
    n = size(A, 1)
    B = similar(A)
    ip = Vector{BlasInt}(undef, n)
    rec = time_column((B, ip) -> recursive_lu!(B, ip, base), A, B, ip; reps)
    lap = time_column((B, ip) -> LAPACK.getrf!(B, ip; check = false), A, B, ip; reps)
    return rec, lap
end

function measure(io, ::Type{T}, n::Int, setting::String, base::Int; reps::Int) where {T}
    A = randn(Xoshiro(n), T, n, n)
    rec, lap = measure_once(A, base; reps)
    rerun = noisy(rec) || noisy(lap)
    rerun && ((rec, lap) = measure_once(A, base; reps))
    Fr, ipr = copy(A), Vector{BlasInt}(undef, n)
    info_r = recursive_lu!(Fr, ipr, base)
    Fg, ipg = copy(A), Vector{BlasInt}(undef, n)
    _, _, info_g = LAPACK.getrf!(Fg, ipg; check = false)
    (info_r == 0 && info_g == 0) ||
        error("LU at n = $n in $T returned info = $info_r (recursion), $info_g (getrf!)")
    be_rec, be_lap = backward_error(A, Fr, ipr), backward_error(A, Fg, ipg)
    r = rec.median / lap.median
    note = rerun ? "measured twice" : ""
    @printf(io,
        "| %s | %d | %s | %d | %.5f / %.5f | %.5f / %.5f | %.2f | %.1e / %.1e | %s | %s |\n",
        T, n, setting, base, 1e3 * rec.median, 1e3 * rec.min, 1e3 * lap.median, 1e3 *
                                                                                lap.min,
        r, be_rec, be_lap, ipr == ipg ? "equal" : "differs", note)
    return (
        T = T, n = n, threads = setting, base = base, recursion = rec, getrf = lap, r = r,
        berr = be_rec, berr_getrf = be_lap, ipiv_equal = ipr == ipg, rerun = rerun)
end

const RL_TABLE_HEAD = "| T | n | threads | base | recursion ms (median / min) | getrf! ms (median / min) | r | berr (recursion / getrf!) | ipiv | note |"

# The rows of one thread setting, measured in this process.
function run_setting(io, setting::String, base::Int, ns, reps::Int)
    setting in RL_SETTINGS ||
        throw(ArgumentError("unknown thread setting $(repr(setting)); expected both, 1 or all"))
    BLAS.set_num_threads(setting == "1" ? 1 : Sys.CPU_THREADS)
    dev = (name = "cpu", array = Array, mod = nothing)
    print_header(
        io, "RL host LU: recursion against getrf! (scripts/hardware/rl_host_lu.jl)",
        dev; extra = ["BLAS library" => blas_library(), "thread setting" => setting,
            "base case" => string(base), "reps" => string(reps)])
    println(io, RL_TABLE_HEAD)
    println(io, "|:--|--:|:--|--:|--:|--:|--:|--:|:--|:--|")
    rows = NamedTuple[]
    for T in RL_ELTYPES, n in ns

        push!(rows, measure(io, T, n, setting, base; reps))
    end
    return rows
end

# The fields of a table row that the verdict reads, from the line that `measure` printed.
function parse_row(line::AbstractString)
    f = strip.(split(strip(line), '|')[2:(end - 1)])
    berr = parse(Float64, strip(split(f[8], '/')[1]))
    return (T = RL_TYPE_NAMES[f[1]], n = parse(Int, f[2]), threads = String(f[3]),
        r = parse(Float64, f[7]), berr = berr, ipiv_equal = f[9] == "equal")
end

# The runs of consecutive entries of `ns` at which `ok(n)` holds, as `(first, last)` pairs.
function runs_where(ok, ns)
    out = Tuple{Int, Int}[]
    start = 0
    for (i, n) in enumerate(ns)
        if ok(n)
            start == 0 && (start = i)
        elseif start != 0
            push!(out, (ns[start], ns[i - 1]))
            start = 0
        end
    end
    start != 0 && push!(out, (ns[start], ns[end]))
    return out
end

"""
    rl_verdict(rows)

The verdict of the rows of both thread settings: `(rule, bands, bound_ok)`, with `rule` one of
`:build`, `:stop` and `:between`, `bands` a `Dict` from each element type to the runs of `n` with
`r ≤ 0.90` in both settings, and `bound_ok` whether every row has `berr ≤ 8n·eps`.
"""
function rl_verdict(rows)
    ns = sort(unique(r.n for r in rows))
    key = Dict((r.T, r.n, r.threads) => r for r in rows)
    for T in unique(r.T for r in rows), n in ns, s in RL_SETTINGS
        haskey(key, (T, n, s)) ||
            throw(ArgumentError("no row for $T at n = $n with the thread setting $s"))
    end
    fast(T, n, lim) = all(s -> key[(T, n, s)].r <= lim, RL_SETTINGS)
    bound_ok = all(r -> r.berr <= 8 * r.n * eps(real(r.T)), rows)
    f64 = [fast(Float64, n, 0.75) for n in ns]
    pair = any(i -> f64[i] && f64[i + 1], 1:(length(ns) - 1))
    rule = bound_ok && pair ? :build : (!bound_ok || !any(f64)) ? :stop : :between
    bands = Dict(T => runs_where(n -> fast(T, n, 0.90), ns)
    for T in unique(r.T for r in rows))
    return (rule = rule, bands = bands, bound_ok = bound_ok)
end

function print_verdict(io, v)
    for T in RL_ELTYPES
        haskey(v.bands, T) || continue
        b = v.bands[T]
        println(io, "band ", T, ": ",
            isempty(b) ? "none (stays on getrf!)" :
            join(("[$lo, $hi]" for (lo, hi) in b), ", "))
    end
    words = Dict(:build => "build rule", :stop => "stop rule", :between => "between the rules")
    println(io, "verdict: ", words[v.rule], v.bound_ok ? "" :
                                            " (a backward error exceeds 8n·eps)")
    return nothing
end

# Both settings, each in a cold process of its own, then the verdict.
function run_both(io, base::Int, ns, reps::Int)
    rows = NamedTuple[]
    for s in RL_SETTINGS
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) $s $base $(join(ns, ',')) $reps`
        out = read(cmd, String)
        print(io, out)
        println(io)
        for line in split(out, '\n')
            startswith(line, "| ") && !startswith(line, "| T ") &&
                push!(rows, parse_row(line))
        end
    end
    v = rl_verdict(rows)
    print_verdict(io, v)
    return rows, v
end

function main(args = ARGS; io::IO = stdout)
    setting = length(args) >= 1 ? String(args[1]) : "both"
    base = length(args) >= 2 ? parse(Int, args[2]) : 16
    ns = length(args) >= 3 ? parse.(Int, split(args[3], ',')) : RL_NS
    reps = length(args) >= 4 ? parse(Int, args[4]) : 5
    base >= 1 || throw(ArgumentError("the base case must be at least 1, got $base"))
    setting == "both" && return run_both(io, base, ns, reps)
    return run_setting(io, setting, base, ns, reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
