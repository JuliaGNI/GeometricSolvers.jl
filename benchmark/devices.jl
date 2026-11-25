# The device rows of one machine: Y1's 79 `Float32` rows in R2 on the device (`r2`), the R2
# throughput grid on the §10.E problem (`grid`), the §10.E problem in R3 form (`r3`), and one LU
# factorisation and solve by each of two paths (`lu`). It writes `<dir>/devices-<machine>.csv`,
# one line per cell, and `<dir>/devices-<machine>.txt`, the conditions of the run.
#
# Usage, from the package root, in a cold process with the vendor package's environment stacked
# under `benchmark/` (none for `cpu`):
#
#   JULIA_LOAD_PATH="@:$PWD/test/gpu/<backend>:@stdlib" \
#       julia --startup-file=no --threads=auto --project=benchmark benchmark/devices.jl \
#       <backend> <machine> <dir>
#
# with the pairs `cpu cpu`, `metal mac`, `cuda spark`, `cuda rtx4090` and `rocm rx7900xtx`. The
# vendor package is loaded only for its backend. `cpu` runs the R2 rows and the grid with
# KernelAbstractions' `CPU()` backend on host arrays and the R3 and LU rows on `JLArray`s, and
# runs only a part of the grid and of the R3 and LU rows.

isdefined(@__MODULE__, :standard_rows) || include(joinpath(@__DIR__, "problems.jl"))
isdefined(@__MODULE__, :MIN_RUN) || include(joinpath(@__DIR__, "packages.jl"))
isdefined(@__MODULE__, :InlineAdapter) || include(joinpath(@__DIR__, "device_problems.jl"))

using GeometricSolvers
using GeometricSolvers: SolverStatus, PerSystem, init_linear, factorize!!, ldiv!!,
                        ka_factorize_lu!, ka_lu_backsolve!
using JLArrays: JLArrays, JLArray
using KernelAbstractions
using LinearAlgebra: BLAS, I, SingularException, lu, norm
using SciMLBase: SciMLBase
using SimpleNonlinearSolve: SimpleNonlinearSolve, SimpleNewtonRaphson
using StaticArrays: StaticArrays, SVector

const DEVICE_HEADER = "kind,machine,backend,card,set,problem,start,n,n_sys,eltype,factor,variant,package,status,code,success,iterations,host_code,host_iterations,fnorm_inf,dx_host,verdict,copies_same,t1,t2,t3,t4,t5,median_s,spread,solves_per_run,compile_s,comment"
const DEVICE_FIELDS = Symbol.(split(DEVICE_HEADER, ","))

"""
    PAIRS

The machines of the device rows, each with its backend.
"""
const PAIRS = (("cpu", "cpu"), ("metal", "mac"), ("cuda", "spark"), ("cuda", "rtx4090"),
    ("rocm", "rx7900xtx"))

# backend => the vendor package and its array type
const VENDORS = Dict("metal" => (:Metal, :MtlArray), "cuda" => (:CUDA, :CuArray),
    "rocm" => (:AMDGPU, :ROCArray))

const DT = Float32                  # the element type of the R2 rows and the grid
const R2_SYSTEMS = 64               # the copies of a row's start in one R2 batch
const R2_COLUMNS = ("Newton Static LU", "Newton Backtracking LU", "Newton DogLeg LU",
    "Picard Backtracking")
const NO_METHOD_SNS = "no method: SimpleNewtonRaphson runs in Newton Static LU only"
# SimpleNonlinearSolve's kernel faults on ROCm with `hipErrorIllegalAddress`, which leaves the
# device context unusable for the rest of the process, so no SimpleNonlinearSolve cell launches
# there.
const SNS_ROCM = "not run: SimpleNonlinearSolve kernel faults on ROCm"
const GRID_NSYS = (10^3, 10^4, 10^5, 10^6, 10^7)
const GRID_N = (2, 4, 6, 8, 12, 16)
const GRID_VARIANTS = ("gs-newton", "gs-reuse4", "gs-backtracking", "sns", "host-r1")
const CPU_GRID_NSYS = (10^3, 10^4)
const CPU_GRID_N = (2, 4, 8)
const TIME_CAP = 1800.0             # seconds, for the six launches of a cell
const R3_N = (256, 1024, 4096)
const CPU_R3_N = (256, 1024)
const R3_VARIANTS = ("device", "host", "device-mixed", "device-mixed-refine", "device-f64",
    "host-f64")
const LU_VARIANTS = ("own-ka", "dispatched")
const METAL_F64 = "no Float64 on Metal; no Metal hybrid (decision 15)"
const NOT_TIMED = "device Float64: not timed (decision 6)"
const INTERNAL = "internal, not user-selectable"

# The codes with which each package reports success, and that of an LU that factorised.
const SUCCESS_CODES = Dict(
    "GeometricSolvers" => ("SUCCESS", "info=0"),
    "SimpleNonlinearSolve" => ("Success", "Terminated", "ExactSolutionLeft",
        "ExactSolutionRight", "FloatingPointLimit", "StalledSuccess"))

"""
    check_pair(backend, machine)

Throw an `ArgumentError` that names both unless `backend` and `machine` are one of `PAIRS`.
"""
function check_pair(backend::AbstractString, machine::AbstractString)
    (backend, machine) in PAIRS && return nothing
    throw(ArgumentError("the backend $backend and the machine $machine are not a pair; " *
                        "the pairs are " * join(("$b $m" for (b, m) in PAIRS), ", ")))
end

"""
    Device(backend, machine, vendor, r2array, r3array, card)

The machine of a run: its `backend` and `machine` names, the `vendor` package or `nothing`, the
maps `r2array` and `r3array` from a host array to the array of the R2 and of the R3 rows, and
the `card`, the name of the device.
"""
struct Device{V, A2, A3}
    backend::String
    machine::String
    vendor::V
    r2array::A2
    r3array::A3
    card::String
end

# A number as it is written into a line: `repr` in its own type, so `1.0f-5` and `NaN32` in Float32.
field(x::AbstractFloat) = repr(x)
field(x) = string(x)

"""
    csv_line(; fields...)

One line of the CSV, with `-` in each field that `fields` does not give. The comment is quoted
when it holds a comma.
"""
function csv_line(; kw...)
    vals = [field(get(kw, f, "-")) for f in DEVICE_FIELDS]
    c = vals[end]
    occursin(',', c) && (vals[end] = "\"" * c * "\"")
    return join(vals, ",")
end

"""
    success_of(code, package, fnorm, T)

Y-4's success rule: `code` is a success of `package` and `fnorm`, the `‖F‖∞` that the host
measures, is at most `sqrt(eps(T))`.
"""
function success_of(code, package, fnorm, ::Type{T}) where {T}
    return code in SUCCESS_CODES[package] && fnorm ≤ sqrt(eps(T))
end

"""
    device_verdict(device_ok, host_ok, dx, xnorm, T)

The verdict of a cell against its reference: `agree` when both succeed and the distance `dx` of
the iterates in `‖·‖∞` is at most `sqrt(eps(T)) · max(1, xnorm)`, `xnorm` the `‖·‖∞` of the
reference (Y-2); `different root` when both succeed beyond it; `device fails`, `host fails` or
`both fail` otherwise.
"""
function device_verdict(device_ok::Bool, host_ok::Bool, dx, xnorm, ::Type{T}) where {T}
    device_ok && host_ok &&
        return dx ≤ sqrt(eps(T)) * max(1, xnorm) ? "agree" : "different root"
    device_ok && return "host fails"
    host_ok && return "device fails"
    return "both fail"
end

# The status of a launch that threw, or a rethrow: an interrupt, and any exception that is
# neither a kernel that does not compile nor a kernel that threw, ends the run with the cell named.
function failed_launch(e, label)
    e isa InterruptException && rethrow()
    s = launch_status(e)
    s === nothing || return s
    println(stderr, "devices.jl: the launch of $label threw")
    rethrow()
end

# The fields of a cell that compares with a reference: the reference's code and iterations, the
# distance of the iterates, which needs a reference that succeeded, the verdict, and the
# reference's `‖x‖∞` and `‖F‖∞`, which the verifier needs to recompute the verdict.
function compared(ok, x, ref, T)
    dx = ref.ok ? norm(x - ref.x, Inf) : "-"
    return (; host_code = ref.code, host_iterations = ref.iterations, dx_host = dx,
        verdict = device_verdict(ok, ref.ok, ref.ok ? dx : 0, norm(ref.x, Inf), T),
        ref = "ref_xnorm=$(field(norm(ref.x, Inf)));ref_fnorm=$(field(ref.fnorm))")
end

"""
    copies_same(X)

The copies of a batch whose iterate lies within Y-2's bound of system 1's, `agrees(X[1], y, DT)`.
A distance to an iterate that is not finite is within no bound, so the count is 0 when system 1's
iterate holds a `NaN`.
"""
copies_same(X) = count(y -> agrees(X[1], y, DT), X)

"""
    successes(codes, package, fnorms)

The count of systems that succeed by Y-4's rule, from their codes and their `‖F‖∞`.
"""
function successes(codes, package, fnorms)
    count(
        i -> success_of(codes[i], package, fnorms[i], DT), eachindex(codes, fnorms))
end

# --- R2: Y1's rows --------------------------------------------------------------------------

@kernel function sns_kernel!(X, codes, f, P, tol, maxiters)
    i = @index(Global)
    prob = SciMLBase.NonlinearProblem{false}(f, X[i], GeometricSolvers.parameters(P, i))
    sol = SciMLBase.solve(prob, SimpleNewtonRaphson(); abstol = tol, maxiters)
    X[i] = sol.u
    codes[i] = Int32(sol.retcode)
end

"""
    sns_batch!(X, f, P)

SimpleNonlinearSolve's `SimpleNewtonRaphson` on each start of `X` in a KernelAbstractions kernel,
one work-item per system, with `abstol = sqrt(eps(Float32))` and `maxiters = 200`, the parameters
`P` passed as `solve_batch!` passes them; it returns the `Int32` return codes on the backend of
`X`, which `sns_code` names.
"""
function sns_batch!(X, f, P)
    backend = get_backend(X)
    codes = KernelAbstractions.allocate(backend, Int32, length(X))
    sns_kernel!(backend)(X, codes, f, P, sqrt(eps(DT)), MAX_ITERATIONS; ndrange = length(X))
    KernelAbstractions.synchronize(backend)
    return codes
end

sns_code(c) = string(SciMLBase.ReturnCode.T(c))

# The host R1 solve of a row, of the same adapter, method and start as its device cell.
function host_r1(row, package, m)
    if package == "GeometricSolvers"
        x, st = solve(NonlinearProblem(row.F), row.x0, m, row.p; f_abstol = sqrt(eps(DT)),
            f_reltol = 0, max_iterations = MAX_ITERATIONS)
        code, iterations = string(st.code), string(st.iterations)
    else
        # SimpleNewtonRaphson throws on a singular Jacobian; the host cell then fails with the
        # code `ERROR`, as Y1's cells do.
        x, code = try
            sol = SciMLBase.solve(SciMLBase.NonlinearProblem{false}(row.F, row.x0, row.p),
                SimpleNewtonRaphson(); abstol = sqrt(eps(DT)), maxiters = MAX_ITERATIONS)
            sol.u, string(sol.retcode)
        catch e
            e isa SingularException || rethrow()
            fill(DT(NaN), typeof(row.x0)), "ERROR"
        end
        iterations = "-"
    end
    fnorm = norm(row.F(x, row.p), Inf)
    return (; x, code, iterations, fnorm, ok = success_of(code, package, fnorm, DT))
end

# The device batch of a row: 64 copies of its start, the iterate of each, and system 1's code
# and iterations.
function device_r2(dev, row, package, m)
    X = dev.r2array(fill(row.x0, R2_SYSTEMS))
    if package == "GeometricSolvers"
        S = KernelAbstractions.allocate(get_backend(X), SolverStatus{DT}, R2_SYSTEMS)
        solve_batch!(X, S, NonlinearProblem(row.F), m, row.p; f_abstol = sqrt(eps(DT)),
            f_reltol = 0, max_iterations = MAX_ITERATIONS)
        s = Array(S)[1]
        code, iterations = string(s.code), string(s.iterations)
    else
        code, iterations = sns_code(Array(sns_batch!(X, row.F, row.p))[1]), "-"
    end
    return Array(X), code, iterations
end

"""
    r2_row_lines(dev, row)

The sixteen `r2` lines of `row`, its four columns by the four packages, and whether a launch of
the row ended in a kernel error.
"""
function r2_row_lines(dev::Device, row)
    n = length(row.x0)
    methods = Dict(STANDARD_COLUMNS)
    lines, failed = String[], false
    for column in R2_COLUMNS, package in PACKAGES

        key = (; kind = "r2", dev.machine, dev.backend, dev.card, row.set, row.problem,
            row.start, n, n_sys = R2_SYSTEMS, eltype = "Float32", variant = column, package)
        if package in ("SimpleSolvers", "NLsolve")
            push!(lines, csv_line(; key..., status = "no device path"))
            continue
        elseif package == "SimpleNonlinearSolve" && column != "Newton Static LU"
            push!(lines, csv_line(; key..., status = NO_METHOD_SNS))
            continue
        end
        band = band_status(dev.backend, n)
        if band !== nothing
            push!(lines, csv_line(; key..., status = band))
            continue
        elseif package == "SimpleNonlinearSolve" && dev.backend == "rocm"
            push!(lines, csv_line(; key..., status = SNS_ROCM))
            continue
        end
        m = methods[column]
        status = "ran"
        X = code = iterations = nothing
        try
            X, code, iterations = device_r2(dev, row, package, m)
        catch e
            status = failed_launch(e, "r2 $(row.set) $(row.problem) $(row.start) $column $package")
            failed |= startswith(status, "kernel error")
        end
        ref = host_r1(row, package, m)
        refs = "ref_xnorm=$(field(norm(ref.x, Inf)));ref_fnorm=$(field(ref.fnorm))"
        if status != "ran"
            push!(lines,
                csv_line(; key..., status, host_code = ref.code,
                    host_iterations = ref.iterations, comment = refs))
            continue
        end
        x = X[1]
        fnorm = norm(row.F(x, row.p), Inf)
        ok = success_of(code, package, fnorm, DT)
        c = compared(ok, x, ref, DT)
        push!(lines,
            csv_line(; key..., status, code, success = ok, iterations, c.host_code,
                c.host_iterations, fnorm_inf = fnorm, c.dx_host, c.verdict,
                copies_same = copies_same(X), comment = c.ref))
    end
    return lines, failed
end

"""
    heap_counter(vendor)
    heap_size(vendor)

The counter of the device malloc heap in bytes, and the size of that heap, or `nothing` where the
backend's heap has no counter to read. Metal.jl keeps the counter in the first 4 bytes of a
shared buffer of `MALLOC_BUF_SIZE` bytes per process, with no free, and the counter grows past
the size once an allocation fails.
"""
function heap_counter(vendor)
    (vendor === nothing || nameof(vendor) !== :Metal) && return nothing
    buf = vendor.malloc_buffer(vendor.device())
    return Int(unsafe_load(convert(Ptr{UInt32}, vendor.MTL.contents(buf))))
end

function heap_size(vendor)
    return vendor === nothing || nameof(vendor) !== :Metal ? nothing :
           vendor.MALLOC_BUF_SIZE
end

"""
    heap_next(k, before, after, limit, fresh, failed)

The R2 row from which the run continues in a new process after row `k`, or `nothing` to go on in
this one, from the heap counter `before` and `after` the row, the heap size `limit` and the
counter `fresh` of an empty heap; `failed` is whether a launch of the row ended in a kernel
error. A row that failed on a heap that earlier rows used runs again on an empty heap (`k`); a
row after which the counter exceeds the size leaves a full heap (`k + 1`). Without a counter
(`limit === nothing`) a run never continues elsewhere.
"""
function heap_next(k, before, after, limit, fresh, failed)
    limit === nothing && return nothing
    failed && before > fresh && return k
    after > limit && return k + 1
    return nothing
end

"""
    r2_lines!(lines, heap, dev, rows, from)

Append the `r2` lines of `rows[from:end]` to `lines`, and the heap counter before and after each
row to `heap`, and return the row from which a new process continues, or `nothing` when every row
is done.
"""
function r2_lines!(lines, heap, dev::Device, rows, from)
    limit, fresh = heap_size(dev.vendor), heap_counter(dev.vendor)
    for k in from:length(rows)
        before = heap_counter(dev.vendor)
        rl, failed = r2_row_lines(dev, rows[k])
        after = heap_counter(dev.vendor)
        next = heap_next(k, before, after, limit, fresh, failed)
        next == k && return k
        append!(lines, rl)
        r = rows[k]
        push!(heap,
            "heap: $(r.set),$(r.problem),$(r.start),$(something(before, "-")),$(something(after, "-"))")
        next === nothing || return next > length(rows) ? nothing : next
    end
    return nothing
end

# --- timing ---------------------------------------------------------------------------------

"""
    measure_cell(fresh, launch!, sync; clock = time_ns)

The times of a cell by the §10 preamble: one warm-up launch, then runs of `k` launches, `k` the
smallest power of two for which two successive runs each last at least `MIN_RUN`, then `N_RUNS`
runs of `k` launches. Each launch takes its own `fresh()` start, all made before the clock starts,
and each run ends with `sync()` before the clock stops. It returns the warm-up launch's result,
its time `warm`, `k`, and the time of one launch in each of the `N_RUNS` runs, in seconds.
`clock()` gives the time in nanoseconds.
"""
function measure_cell(fresh, launch!, sync; clock = time_ns)
    function one_run(k)
        starts = [fresh() for _ in 1:k]
        t0 = clock()
        for s in starts
            launch!(s)
        end
        sync()
        return (clock() - t0) / 1e9 / k
    end
    s = fresh()
    t0 = clock()
    result = launch!(s)
    sync()
    warm = (clock() - t0) / 1e9
    k = 1
    while k * min(one_run(k), one_run(k)) < MIN_RUN
        k *= 2
    end
    return (; result, warm, k, times = [one_run(k) for _ in 1:N_RUNS])
end

# The time fields of a measured cell.
function time_fields(m)
    med = run_median(m.times)
    return (; t1 = m.times[1], t2 = m.times[2], t3 = m.times[3], t4 = m.times[4],
        t5 = m.times[5], median_s = med, spread = run_spread(m.times),
        solves_per_run = m.k, compile_s = m.warm - med)
end

"""
    projected(median)

The time of a cell projected from the median of the cell at a tenth of its systems: ten times
that median for each of its six launches, the warm-up and five runs.
"""
projected(median) = 10 * median * (1 + N_RUNS)

"""
    grid_series(measure, nsys)

The cells of one `n` and variant of the grid at the system counts `nsys`, in order: each
`measure(n_sys)`, a measured cell or a status, until the projection from the cell before it
exceeds `TIME_CAP`; that cell and every later one is `not run: time cap`. A cell after one whose
launch threw has no projection and runs.
"""
function grid_series(measure, nsys)
    out = Any[]
    prev = nothing
    for n_sys in nsys
        if prev !== nothing && projected(run_median(prev.times)) > TIME_CAP ||
           (!isempty(out) && out[end] == "not run: time cap")
            push!(out, "not run: time cap")
            continue
        end
        m = measure(n_sys)
        prev = m isa NamedTuple ? m : nothing
        push!(out, m)
    end
    return out
end

# --- the R2 throughput grid -------------------------------------------------------------------

# One launch of a grid variant on the starts `s.X`, writing the codes into `s.S`.
function grid_launch!(variant, F, B, s)
    tol, it = sqrt(eps(DT)), MAX_ITERATIONS
    if variant == "sns"
        backend = get_backend(s.X)
        sns_kernel!(backend)(s.X, s.S, F, PerSystem(B), tol, it; ndrange = length(s.X))
    elseif variant == "host-r1"
        Threads.@threads for i in eachindex(s.X)
            s.X[i], s.S[i] = solve(NonlinearProblem(F), s.X[i], Newton(), B[i];
                f_abstol = tol, f_reltol = 0, max_iterations = it)
        end
    else
        m = variant == "gs-newton" ? Newton() :
            variant == "gs-reuse4" ? Newton(; reuse = 4) :
            Newton(; globalization = Backtracking())
        solve_batch!(s.X, s.S, NonlinearProblem(F), m, PerSystem(B); f_abstol = tol,
            f_reltol = 0, max_iterations = it)
    end
    return s
end

"""
    grid_cell(dev, n, n_sys, variant)

The measured grid cell of `variant` on `n_sys` systems of dimension `n` of the §10.E problem,
from `x₀ = 0`, or the status of a launch that threw.
"""
function grid_cell(dev::Device, n, n_sys, variant)
    F, Bh = tenE_batch(n, n_sys, DT)
    host = variant == "host-r1"
    arr = host ? identity : dev.r2array
    B = arr(Bh)
    x0 = zero(SVector{n, DT})
    fresh() = begin
        X = arr(fill(x0, n_sys))
        S = variant == "sns" ? KernelAbstractions.allocate(get_backend(X), Int32, n_sys) :
            KernelAbstractions.allocate(get_backend(X), SolverStatus{DT}, n_sys)
        (; X, S)
    end
    sync() = KernelAbstractions.synchronize(get_backend(B))
    m = try
        measure_cell(fresh, s -> grid_launch!(variant, F, B, s), sync)
    catch e
        return failed_launch(e, "grid n = $n n_sys = $n_sys $variant")
    end
    X, S = Array(m.result.X), Array(m.result.S)
    fnorms = [norm(F(X[i], Bh[i]), Inf) for i in eachindex(X)]
    package = variant == "sns" ? "SimpleNonlinearSolve" : "GeometricSolvers"
    codes = variant == "sns" ? sns_code.(S) : [string(s.code) for s in S]
    ok = successes(codes, package, fnorms)
    tally = Dict(c => count(==(c), codes) for c in unique(codes))
    code = join(("$c=$(tally[c])" for c in sort!(collect(keys(tally)))), ";")
    iterations = variant == "sns" ? "-" : maximum(s -> s.iterations, S)
    return (; m..., code, ok, iterations, fnorm = maximum(fnorms))
end

"""
    grid_status(backend, n, variant)
    grid_nsys(backend)

The status of every grid cell of dimension `n` and `variant` on `backend` that is not launched,
`not run: cpu mode`, `above band <band>` or `SNS_ROCM`, or `nothing` for a series that
runs; and the system counts at which a series runs. The cpu mode runs `n ∈ CPU_GRID_N` at
`n_sys ∈ CPU_GRID_NSYS`, the band applies to every variant but `host-r1`, which runs on the host,
and `sns` within the band does not run on ROCm.
"""
function grid_status(backend, n, variant)
    backend == "cpu" && !(n in CPU_GRID_N) && return "not run: cpu mode"
    variant == "host-r1" && return nothing
    band = band_status(backend, n)
    return band === nothing && variant == "sns" && backend == "rocm" ? SNS_ROCM : band
end

grid_nsys(backend) = backend == "cpu" ? CPU_GRID_NSYS : GRID_NSYS

"""
    grid_lines(dev)

The 150 `grid` lines: the system counts by the dimensions by the variants.
"""
function grid_lines(dev::Device)
    cells = Dict{Tuple{Int, String}, Vector{Any}}()
    for n in GRID_N, variant in GRID_VARIANTS

        status = grid_status(dev.backend, n, variant)
        cells[(n, variant)] = if status !== nothing
            fill(status, length(GRID_NSYS))
        else
            s = grid_series(n_sys -> grid_cell(dev, n, n_sys, variant),
                grid_nsys(dev.backend))
            [s; fill("not run: cpu mode", length(GRID_NSYS) - length(s))]
        end
    end
    lines = String[]
    for (j, n_sys) in enumerate(GRID_NSYS), n in GRID_N, variant in GRID_VARIANTS
        key = (; kind = "grid", dev.machine, dev.backend, dev.card, problem = "10.E",
            start = "zero", n, n_sys, eltype = "Float32", variant,
            package = variant == "sns" ? "SimpleNonlinearSolve" : "GeometricSolvers")
        c = cells[(n, variant)][j]
        if c isa AbstractString
            push!(lines, csv_line(; key..., status = c))
        else
            push!(lines,
                csv_line(; key..., status = "ran", c.code, success = c.ok, c.iterations,
                    fnorm_inf = c.fnorm, time_fields(c)...))
        end
    end
    return lines
end

# --- the R3 and LU rows -----------------------------------------------------------------------

# A solve of the §10.E problem in R3 form in the element type `T` on the arrays of `arr`.
function r3_solve(arr, n, ::Type{T}, m) where {T}
    A, b = tenE_r3(n, T)
    p = (; A = arr(A), b = arr(b), Id = arr(Matrix{T}(I, n, n)))
    fresh() = fill!(similar(p.b), 0)
    launch!(x) = solve(NonlinearProblem(r3_F!, r3_J!), x, m, p; f_abstol = sqrt(eps(T)),
        f_reltol = 0, max_iterations = MAX_ITERATIONS)
    sync() = KernelAbstractions.synchronize(get_backend(p.b))
    result(x, st) = begin
        xh = Array(x)
        fnorm = norm(A * xh .+ sin.(xh) ./ 10 .- b, Inf)
        code = string(st.code)
        (; x = xh, code, iterations = string(st.iterations), fnorm,
            ok = success_of(code, "GeometricSolvers", fnorm, T))
    end
    return fresh, launch!, sync, result
end

"""
    r3_lines(dev, n)

The six `r3` lines of dimension `n`, in the order of `R3_VARIANTS`.
"""
function r3_lines(dev::Device, n)
    cells = Dict{String, Any}()
    for v in R3_VARIANTS
        T = v in ("device", "host") ? Float32 : Float64
        TF = v == "device-f64" || v == "host-f64" ? Float64 : Float32
        key = (; kind = "r3", dev.machine, dev.backend, dev.card, problem = "10.E",
            start = "zero", n, n_sys = 1, eltype = string(T), factor = string(TF),
            variant = v, package = "GeometricSolvers")
        if dev.backend == "cpu" && !(n in CPU_R3_N)
            cells[v] = (; key, status = "not run: cpu mode")
            continue
        elseif dev.backend == "metal" && T === Float64 && startswith(v, "device")
            cells[v] = (; key, status = METAL_F64)
            continue
        end
        m = Newton(; globalization = Backtracking(),
            linsolve = v in ("device-mixed", "device-mixed-refine") ?
                       LUFactorization(Float32) : LUFactorization(),
            refinement = v == "device-mixed-refine" ? Refinement() : NoRefinement())
        fresh, launch!, sync, result = r3_solve(
            startswith(v, "host") ? identity : dev.r3array,
            n, T, m)
        if v == "device-f64"
            x0 = fresh()
            cells[v] = (; key, status = "ran", r = result(launch!(x0)...), times = nothing)
        else
            t = measure_cell(fresh, launch!, sync)
            cells[v] = (;
                key, status = "ran", r = result(t.result...), times = time_fields(t))
        end
    end
    lines = String[]
    for v in R3_VARIANTS
        c = cells[v]
        if c.status != "ran"
            push!(lines, csv_line(; c.key..., c.status))
            continue
        end
        T = c.key.eltype == "Float32" ? Float32 : Float64
        refname = v == "device" ? "host" : startswith(v, "host") ? nothing : "host-f64"
        comment = v == "device-f64" ? NOT_TIMED : "-"
        cmp = if refname === nothing
            # A reference line carries its own `‖x‖∞`, for the verdicts against it.
            (; comment = "xnorm=$(field(norm(c.r.x, Inf)))")
        elseif cells[refname].status != "ran"
            (; comment)
        else
            k = compared(c.r.ok, c.r.x, cells[refname].r, T)
            (; k.host_code, k.host_iterations, k.dx_host, k.verdict, comment)
        end
        times = c.times === nothing ? (;) : c.times
        push!(lines,
            csv_line(; c.key..., status = "ran", c.r.code, success = c.r.ok,
                c.r.iterations, fnorm_inf = c.r.fnorm, cmp..., times...))
    end
    return lines
end

"""
    lu_once!(variant, fac, J, b)

One factorisation of `J` and one solve with the right-hand side `b`, from the factor `fac`, by
the path `variant`, and the factor and the solution: `dispatched` is the path of `solve`,
`factorize!!` and `ldiv!!`, which scales `b` by `2^(-rhs_exponent(b))` before the backsolve and
the solution by `2^(rhs_exponent(b) − e)` after it; `own-ka`, which no option of `solve` selects
where an extension's LU applies, calls the internal `ka_factorize_lu!` and `ka_lu_backsolve!`
and scales the solution by `2^(−e)`, `e` the factor's `scale_exponent`.
"""
function lu_once!(variant, fac, J, b)
    if variant == "own-ka"
        fac = ka_factorize_lu!(fac, J)
        d = copy(b)
        ka_lu_backsolve!(fac.factors, fac.ipiv, d)
        d .*= exp2(eltype(d)(-fac.scale_exponent))
    else
        fac = factorize!!(fac, LUFactorization(), J)
        d = similar(b)
        ldiv!!(d, fac, b)
    end
    return fac, d
end

"""
    lu_line(dev, n, variant)

The `lu` line of dimension `n` and path `variant`: one factorisation and one solve of
`J = A + I/10` with the right-hand side `b` of the §10.E problem in `Float32`, timed together,
against the host LAPACK solution of `J d = b`.
"""
function lu_line(dev::Device, n, variant)
    key = (;
        kind = "lu", dev.machine, dev.backend, dev.card, problem = "10.E", start = "-",
        n, n_sys = 1, eltype = "Float32", factor = "Float32", variant,
        package = "GeometricSolvers")
    dev.backend == "cpu" && !(n in CPU_R3_N) &&
        return csv_line(; key..., status = "not run: cpu mode")
    A, b = tenE_r3(n, DT)
    J = A + DT(1 / 10) * I
    Jd, bd = dev.r3array(J), dev.r3array(b)
    F = lu(J; check = false)
    dh = F.info == 0 ? F \ b : fill(DT(NaN), n)
    hcode = "info=$(F.info)"
    hfnorm = norm(J * dh - b, Inf)
    ref = (; x = dh, code = hcode, iterations = "-", fnorm = hfnorm,
        ok = success_of(hcode, "GeometricSolvers", hfnorm, DT))
    fresh() = init_linear(LUFactorization(), nothing, bd)
    sync() = KernelAbstractions.synchronize(get_backend(bd))
    t = measure_cell(fresh, fac -> lu_once!(variant, fac, Jd, bd), sync)
    fac, d = t.result
    x = Array(d)
    code = "info=$(fac.info)"
    fnorm = norm(J * x - b, Inf)
    ok = success_of(code, "GeometricSolvers", fnorm, DT)
    c = compared(ok, x, ref, DT)
    return csv_line(; key..., status = "ran", code, success = ok, iterations = "-",
        c.host_code, c.host_iterations, fnorm_inf = fnorm, c.dx_host, c.verdict,
        time_fields(t)..., comment = variant == "own-ka" ? "$INTERNAL; $(c.ref)" : c.ref)
end

# --- the run ----------------------------------------------------------------------------------

function load_vendor(backend)
    backend == "cpu" && return nothing
    package = first(VENDORS[backend])
    Core.eval(Main, Expr(:using, Expr(:., package)))
    return Base.invokelatest(getfield, Main, package)
end

function device_of(backend, machine, vendor)
    if vendor === nothing
        return Device(backend, machine, nothing, identity, JLArray,
            csv_text("host $(Sys.CPU_NAME)"))
    end
    A = getfield(vendor, last(VENDORS[backend]))
    return Device(
        backend, machine, vendor, x -> A(x), x -> A(x), csv_text(card_name(vendor)))
end

"""
    card_name(vendor)

The name of the vendor package's current device, as its driver gives it: CUDA's `name`, the
`name` of a Metal device, and HIP's `name` on AMDGPU.
"""
function card_name(vendor)
    d = vendor.device()
    nameof(vendor) === :CUDA && return vendor.CUDACore.name(d)
    nameof(vendor) === :Metal && return String(d.name)
    return vendor.HIP.name(d)
end

# The driver of the device, for the `.txt`.
function driver(dev::Device)
    v = dev.vendor
    v === nothing && return "none"
    nameof(v) === :CUDA && return "CUDA driver $(v.driver_version())"
    nameof(v) === :Metal && return "macOS $(v.macos_version())"
    return "HIP runtime $(v.HIP.runtime_version())"
end

"""
    txt_head(dev)

The lines of the `.txt` before the heap lines: the date, the `uptime`, the Julia version, the
threads, the BLAS threads, the device, its driver, the band and the versions of the packages.
"""
function txt_head(dev::Device)
    mods = Any[
        GeometricSolvers, KernelAbstractions, StaticArrays, SimpleNonlinearSolve, JLArrays]
    dev.vendor === nothing || push!(mods, dev.vendor)
    band = BANDS[dev.backend]
    return ["date: $(strip(read(`date -u`, String)))",
        "uptime: $(strip(read(`uptime`, String)))", "julia: $VERSION",
        "threads: $(Threads.nthreads())", "blas threads: $(BLAS.get_num_threads())",
        "device: $(dev.card)", "driver: $(driver(dev))",
        "band: $(band === nothing ? "none" : band)",
        "packages: " * join(("$(nameof(m)) $(pkgversion(m))" for m in mods), ", ")]
end

# The R2 lines from row `from` on in new processes, one after another, each from where the last
# one stopped: a process writes its lines, its heap lines and the row to continue from into
# `part`.
function r2_in_processes!(lines, heap, dir, backend, machine, from)
    part = joinpath(dir, "devices-$machine.r2-part")
    while from !== nothing
        run(`$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$(Base.active_project()) $(@__FILE__) $backend $machine $dir --r2-from $from`)
        from = nothing
        for l in readlines(part)
            tag, rest = split(l, " "; limit = 2)
            tag == "r2" ? push!(lines, rest) :
            tag == "heap" ? push!(heap, rest) : (from = parse(Int, rest))
        end
        rm(part)
    end
end

"""
    run_device(backend, machine, dir; r2_from = nothing)

Run the device rows of `machine` on `backend` and write `<dir>/devices-<machine>.csv` and `.txt`.
With `r2_from`, run only the `r2` rows from that row on and write them into the part file of
`r2_in_processes!`.
"""
function run_device(backend, machine, dir; r2_from = nothing)
    check_pair(backend, machine)
    BLAS.set_num_threads(1)
    vendor = load_vendor(backend)
    dev = Base.invokelatest(device_of, backend, machine, vendor)
    # The conditions of the run, read before the first row, so that a call that fails stops it.
    head = Base.invokelatest(txt_head, dev)
    rows = device_rows(DT)
    lines, heap = String[], String[]
    next = Base.invokelatest(r2_lines!, lines, heap, dev, rows, something(r2_from, 1))
    if r2_from !== nothing
        open(joinpath(dir, "devices-$machine.r2-part"), "w") do io
            foreach(l -> println(io, "r2 ", l), lines)
            foreach(l -> println(io, "heap ", l), heap)
            next === nothing || println(io, "next ", next)
        end
        return nothing
    end
    next === nothing || r2_in_processes!(lines, heap, dir, backend, machine, next)
    append!(lines, Base.invokelatest(grid_lines, dev))
    for n in R3_N
        append!(lines, Base.invokelatest(r3_lines, dev, n))
    end
    for n in R3_N, v in LU_VARIANTS

        push!(lines, Base.invokelatest(lu_line, dev, n, v))
    end
    write(joinpath(dir, "devices-$machine.csv"), join([DEVICE_HEADER; lines], "\n") * "\n")
    write(joinpath(dir, "devices-$machine.txt"), join([head; heap], "\n") * "\n")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (3, 5) ||
        error("usage: devices.jl <backend> <machine> <dir> [--r2-from <row>]")
    run_device(ARGS[1], ARGS[2], ARGS[3];
        r2_from = length(ARGS) == 5 ? parse(Int, ARGS[5]) : nothing)
end
