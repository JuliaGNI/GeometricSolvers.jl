# The three kernel premises of part Y3 (§13.Y, "Premises not checked"), probed on one backend:
#
#   P1  Y1's 79 `Float32` rows through an `MVector` adapter in GeometricSolvers' R2 `solve_batch!`
#   P2  the same rows with SimpleNonlinearSolve's `SimpleNewtonRaphson` in a KA kernel, one thread
#       per system, as part E ran it (`scripts/spikes/kernel_newton/sns.jl`)
#   P3  which dense LU runs for `LUFactorization()` on a device array with the vendor extension
#       loaded, whether a seam selects the own KA LU, and the §10.E residual in R3 form, n = 256
#
# Usage, from the repository root, with <backend> one of cpu, metal, cuda, rocm and <pass> one of
# p1-plain, p1-inline, p2, p3, and optionally `first:last`, the row lines of a pass by their
# position among its 79 lines (1-based, in the order of the output):
#   JULIA_LOAD_PATH="@:$PWD/test/gpu/<backend>:@stdlib" \
#       julia --startup-file=no --threads=1 --project=benchmark benchmark/y3_premises.jl <backend> <pass> [first:last]
#
# Run each pass in a process of its own. A kernel that allocates draws on the device malloc heap,
# and on Metal (Metal.jl 1.12) that heap is 1 MiB per device and process, with no `free` and no
# reset: once it is full, every later allocating kernel of the process fails. On Metal each row
# line records the heap counter before and after the row, and when a row fills the heap the script
# continues the pass in a new process with an empty heap, from that row again if it failed on a
# heap that earlier rows had used, else from the next row. A row that fails on an empty heap keeps
# its outcome.
#
# The vendor package comes only from the stacked `test/gpu/<backend>` environment (decision Y3-11)
# and is loaded only for its backend; `cpu` needs none and runs KA's `CPU()` on host arrays, which
# compiles its kernels through GPUCompiler, so a heap allocation in a kernel fails there as well.
# P1 runs each row through two adapters, a plain call and one that inlines the residual. Output is
# stdout only: one CSV line per cell, prefixed with the probe, a header line per probe and adapter,
# and a summary line per probe and adapter. One row's failure never stops the script.

using GeometricSolvers
using GeometricSolvers: ka_factorize_lu!, ka_lu_backsolve!, init_linear, factorize!!,
                        backsolve!
using KernelAbstractions
using LinearAlgebra
using Random: Xoshiro
using SciMLBase: SciMLBase
using SimpleNonlinearSolve: SimpleNonlinearSolve, SimpleNewtonRaphson
using StaticArrays

include(joinpath(@__DIR__, "problems.jl"))

# backend => (vendor package, array type, R1 size band; `nothing` is no band)
const BACKENDS = Dict("cpu" => (nothing, nothing, nothing),
    "metal" => (:Metal, :MtlArray, 24), "cuda" => (:CUDA, :CuArray, 24),
    "rocm" => (:AMDGPU, :ROCArray, 8))

const T = Float32
const TOL = sqrt(eps(T))       # Y-3
const MAXIT = 200              # Y-3
const COPIES = 4               # systems per batch

# The out-of-place adapter of Y3-3: copy `x` into an `MVector`, call the in-place residual on it,
# return an `SVector`. `MAdapter{true}` inlines the residual at the call site: an `MVector` that
# reaches a call which is not inlined escapes to the heap, which a kernel cannot allocate.
struct MAdapter{I, F}
    f!::F
end

MAdapter{I}(f!) where {I} = MAdapter{I, typeof(f!)}(f!)

@inline function (a::MAdapter{false})(x::SVector{N, S}, p) where {N, S}
    y = MVector{N, S}(undef)
    a.f!(y, MVector{N, S}(x), p)
    return SVector{N, S}(y)
end

@inline function (a::MAdapter{true})(x::SVector{N, S}, p) where {N, S}
    y = MVector{N, S}(undef)
    @inline a.f!(y, MVector{N, S}(x), p)
    return SVector{N, S}(y)
end

# The midpoint stage residuals of `midpoint_system` in `problems.jl`, with the initial state held
# as `SVector`s and the GeometricProblems parameters passed as `p`, because the benchmark's
# closures capture host `Vector`s and allocate. The vector fields are GeometricProblems' own, called
# with `@inline` so that the `MVector`s they write stay off the heap where the adapter inlines.
struct MidpointResidual{K, Fs, S, Q, P}
    fs::Fs
    t̃::S
    h::S
    q₀::Q
    p₀::P
end

function (r::MidpointResidual{:ode})(y, V, par)
    @inline r.fs.v(y, r.t̃, r.q₀ .+ (r.h / 2) .* SVector(V), par)
    y .-= V
    return nothing
end

function (r::MidpointResidual{:iode})(y, V, par)
    Q = r.q₀ .+ (r.h / 2) .* SVector(V)
    f = similar(y)
    @inline r.fs.ϑ(y, r.t̃, Q, V, par)
    @inline r.fs.f(f, r.t̃, Q, V, par)
    y .-= r.p₀ .+ (r.h / 2) .* f
    return nothing
end

function (r::MidpointResidual{:hode, Fs, S, <:SVector{D}})(y, X, par) where {Fs, S, D}
    V = SVector{D}(ntuple(i -> X[i], Val(D)))
    W = SVector{D}(ntuple(i -> X[D + i], Val(D)))
    v, f = MVector{D, eltype(y)}(undef), MVector{D, eltype(y)}(undef)
    @inline r.fs.v(v, r.t̃, r.q₀ .+ (r.h / 2) .* V, r.p₀ .+ (r.h / 2) .* W, par)
    @inline r.fs.f(f, r.t̃, r.q₀ .+ (r.h / 2) .* V, r.p₀ .+ (r.h / 2) .* W, par)
    for i in 1:D
        y[i] = v[i] - V[i]
        y[D + i] = f[i] - W[i]
    end
    return nothing
end

function midpoint_residual(prob)
    k = problem_kind(prob)
    q₀ = SVector{length(prob.ics.q)}(collect(prob.ics.q))
    p₀ = k == :ode ? nothing : SVector{length(prob.ics.p)}(collect(prob.ics.p))
    t̃ = prob.timespan[1] + prob.timestep / 2
    return MidpointResidual{k, typeof(prob.functions), typeof(t̃), typeof(q₀), typeof(p₀)}(
        prob.functions, t̃, prob.timestep, q₀, p₀)
end

# Every row as (label fields, start, in-place residual, parameters, whether the benchmark's own
# residual is `isbits`): the standard rows call their residual as it is, with `p = nothing`.
function probe_rows()
    rows = Any[(r.set, r.problem, r.start, r.x0, r.F, nothing, isbits(r.F))
               for r in standard_rows(T)]
    for ((name, h), r) in zip(MIDPOINT_SCENARIOS, midpoint_rows(T))
        prob = midpoint_problem(name, h, T)
        push!(rows, (r.set, r.problem, r.start, r.x0, midpoint_residual(prob),
            prob.parameters, isbits(r.F)))
    end
    return rows
end

# The exception that a kernel launch wraps, its type and its first line, and whether it is the
# form of a kernel that does not compile.
function unwrap(e)
    e isa TaskFailedException && return unwrap(e.task.exception)
    e isa CompositeException && return unwrap(first(e.exceptions))
    e isa LoadError && return unwrap(e.error)
    return e
end

function outcome(e)
    e = unwrap(e)
    line = first(split(sprint(showerror, e), '\n'))
    line = replace(length(line) > 160 ? first(line, 160) * "…" : line, '"' => '\'')
    kind = nameof(typeof(e)) in (:InvalidIRError, :KernelError) ? "no compile" : "error"
    return kind, "\"$(nameof(typeof(e))): $line\""
end

counts_string(c) = join(("$k=$v" for (k, v) in pairs(c) if v > 0), ";")

finf(F, X, p) = maximum(x -> norm(F(x, p), Inf), X)

@kernel function sns_kernel!(X, codes, f, p)
    i = @index(Global)
    prob = SciMLBase.NonlinearProblem{false}(f, X[i], p)
    sol = SciMLBase.solve(prob, SimpleNewtonRaphson(); abstol = TOL, maxiters = MAXIT)
    X[i] = sol.u
    codes[i] = Int32(sol.retcode)
end

function sns_batch!(X, f, p)
    backend = get_backend(X)
    codes = KernelAbstractions.allocate(backend, Int32, length(X))
    sns_kernel!(backend)(X, codes, f, p; ndrange = length(X))
    KernelAbstractions.synchronize(backend)
    names = string.(SciMLBase.ReturnCode.T.(Array(codes)))
    return Dict(k => count(==(k), names) for k in unique(names))
end

# The device malloc heap of the backend: its counter in bytes and its size, or `nothing` where the
# backend's heap has no counter to read. Metal.jl keeps the counter in the first 4 bytes of a
# shared buffer, and the counter grows past the size once an allocation fails.
function heap_counter(vendor)
    (vendor === nothing || nameof(vendor) !== :Metal) && return nothing
    buf = vendor.malloc_buffer(vendor.device())
    return Int(unsafe_load(convert(Ptr{UInt32}, vendor.MTL.contents(buf))))
end

function heap_size(vendor)
    vendor === nothing || nameof(vendor) !== :Metal ? nothing : vendor.MALLOC_BUF_SIZE
end

# P1 and P2: one batch of `COPIES` copies of each row's start. `lines` are the positions of `rows`
# among the 79. Where the backend's heap has a counter, a row that fills the heap ends the process's
# share of the pass, and the function returns the line to continue from in a fresh process: the
# same line when the row failed on a heap that earlier rows had used, so that it runs again on an
# empty heap, else the next line. Otherwise it returns `nothing`.
function probe_rows!(label, solver, inline, rows, lines, todevice, band, vendor)
    adapter = inline ? "inline" : "plain"
    println(label,
        ",adapter,set,problem,start,n,above_band,params_isbits,adapter_isbits,rowF_isbits,outcome,detail,max_Finf,heap_before,heap_after")
    tally = Dict("ran" => 0, "no compile" => 0, "error" => 0)
    limit = heap_size(vendor)
    fresh = heap_counter(vendor)
    show_heap(h) = h === nothing ? "-" : string(h)
    next = nothing
    done = 0
    for (k, (set, problem, start, x0, f!, p, rowF_isbits)) in enumerate(rows)
        n = length(x0)
        F = MAdapter{inline}(f!)
        above = band === nothing ? "-" : string(n > band)
        kind, detail, res = "ran", "", ""
        before = heap_counter(vendor)
        try
            X = todevice(fill(SVector{n, T}(x0), COPIES))
            detail = solver(X, F, p)
            res = string(finf(F, Array(X), p))
        catch e
            kind, detail = outcome(e)
        end
        after = heap_counter(vendor)
        if limit !== nothing && after >= limit && kind != "ran" && before > fresh
            next = lines[k]
            break
        end
        tally[kind] += 1
        done += 1
        println(label, ",", adapter, ",", set, ",\"", problem, "\",", start, ",", n, ",",
            above, ",", isbits(p), ",", isbits(F), ",", rowF_isbits, ",", kind, ",", detail,
            ",", res, ",", show_heap(before), ",", show_heap(after))
        if limit !== nothing && after >= limit && k < length(rows)
            next = lines[k + 1]
            break
        end
    end
    span = done == 0 ? "no lines" : "lines $(first(lines)):$(lines[done])"
    println(label, " (", adapter, " adapter): ", span, ", ", tally["ran"], " ran, ",
        tally["no compile"], " no compile, ", tally["error"], " error",
        next === nothing ? "" :
        "; the device heap is full, the pass continues in a new process from line $next")
    return next
end

function p1_solver(X, F, p)
    counts_string(solve_batch!(
        X, NonlinearProblem(F), Newton(), p; f_abstol = TOL,
        f_reltol = 0, max_iterations = MAXIT))
end

p2_solver(X, F, p) = counts_string(sns_batch!(X, F, p))

# P3: the §10.E residual in R3 form, `A = 2I + R` with off-diagonal row sums at most 1/2.
function r3_problem(n)
    rng = Xoshiro(1)
    R = 2 .* rand(rng, n, n) .- 1
    R[diagind(R)] .= 0
    R .*= 0.5 / (n - 1)
    A = T.(2I + R)
    xs = T.(2 .* rand(rng, n) .- 1)
    b = A * xs .+ sin.(xs) ./ 10
    return A, b
end

r3_F!(y, x, p) = (y .= p.A * x .+ sin.(x) ./ 10 .- p.b; nothing)
r3_J!(J, x, p) = (J .= p.A .+ p.Id .* reshape(cos.(x) ./ 10, 1, :); nothing)

# The own KA LU called directly in a host Newton loop: `ka_factorize_lu!` factorises `J / 2^e`.
function ka_newton(p, n)
    x = fill!(similar(p.b), 0)
    y, J = similar(x), similar(p.A)
    fac = init_linear(LUFactorization(), nothing, x)
    for k in 0:MAXIT
        r3_F!(y, x, p)
        maximum(abs, y) <= TOL && return x, "converged", k
        r3_J!(J, x, p)
        fac = ka_factorize_lu!(fac, J)
        fac.info == 0 || return x, "singular", k
        ka_lu_backsolve!(fac.factors, fac.ipiv, y)
        x .-= y .* T(2.0^(-fac.scale_exponent))
    end
    return x, "maxiters", MAXIT
end

function probe_lu(name, todevice, vendor)
    n = 256
    A, b = r3_problem(n)
    host_finf(x) = norm(A * x .+ sin.(x) ./ 10 .- b, Inf)
    p = (; A = todevice(A), b = todevice(b), Id = todevice(Matrix{T}(I, n, n)))
    fac = init_linear(LUFactorization(), nothing, p.b)
    mf = which(factorize!!, (typeof(fac), LUFactorization{Nothing}, typeof(p.A)))
    mb = which(backsolve!, (typeof(fac), typeof(p.b)))
    own = mf.module === GeometricSolvers
    println("P3,n,array,factor,factorize_method,backsolve_method")
    quoted(m) = "\"" * replace(string(m), '"' => '\'') * "\""
    println("P3,", n, ",", nameof(typeof(p.b)), ",", nameof(typeof(fac)), ",", quoted(mf),
        ",", quoted(mb))
    seam = vendor === nothing ? "no vendor package" :
           own ? "own KA LU dispatched; no vendor LU in GeometricSolvers for this backend" :
           "no seam: the extension's method is more specific than the KA method for this array type, and no solve-level option selects the KA one"
    println("P3 seam: ", seam)
    println("P3,path,status,iterations,host_Finf")
    dispatched = vendor === nothing ? "dispatched (host LU)" :
                 own ? "dispatched (own KA LU)" : "dispatched (vendor LU)"
    paths = Any[dispatched => function ()
        x, st = solve(NonlinearProblem(r3_F!, r3_J!), fill!(similar(p.b), 0), Newton(), p;
            f_abstol = TOL, f_reltol = 0, max_iterations = MAXIT)
        return x, string(st.code), st.iterations
    end]
    fac isa GeometricSolvers.DeviceLU && !own &&
        push!(paths, "own KA LU, called directly (internal, not a seam)" =>
            () -> ka_newton(p, n))
    for (label, solve_path) in paths
        try
            x, st, it = solve_path()
            println("P3,\"", label, "\",", st, ",", it, ",", host_finf(Array(x)))
        catch e
            kind, detail = outcome(e)
            println("P3,\"", label, "\",", kind, ",", detail, ",")
        end
    end
    println("P3: factorize!! runs ", own ? "the own " : "a vendor ", "method from ",
        mf.module, " (", mf.file, ":", mf.line, "); ", seam)
end

const PASSES = ("p1-plain", "p1-inline", "p2", "p3")

function probe_all(name, vendor, todevice, band, pass, selection)
    sh(cmd) =
        try
            strip(read(cmd, String))
        catch e
            "unavailable: $(nameof(typeof(e)))"
        end
    device = vendor === nothing ? "host CPU, $(Sys.CPU_NAME), KA CPU()" :
             try
        repr(vendor.device())
    catch e
        "unknown: $(nameof(typeof(e)))"
    end
    println("Y3 premises -- backend = ", name, ", pass = ", pass,
        selection === nothing ? "" : ", rows $(first(selection)):$(last(selection))")
    println("date: ", sh(`date -u`))
    println("uptime: ", sh(`uptime`))
    println("Julia ", VERSION, ", threads ", Threads.nthreads())
    println("device: ", device)
    println("band: ", band === nothing ? "none" : band)
    mods = (GeometricSolvers, KernelAbstractions, StaticArrays, SimpleNonlinearSolve)
    println("packages: ", join(("$(nameof(m)) $(pkgversion(m))" for m in mods), ", "),
        vendor === nothing ? ", no vendor package" :
        ", $(nameof(vendor)) $(pkgversion(vendor))")
    pass == "p3" && return probe_lu(name, todevice, vendor)
    rows = probe_rows()
    count(r -> r[1] == "standard", rows) == 67 &&
    count(r -> r[1] == "midpoint", rows) == 12 ||
        error("expected 67 standard and 12 midpoint rows")
    lines = selection === nothing ? (1:length(rows)) : selection
    label, solver, inline = pass == "p1-plain" ? ("P1", p1_solver, false) :
                            pass == "p1-inline" ? ("P1", p1_solver, true) :
                            ("P2", p2_solver, true)
    next = probe_rows!(label, solver, inline, rows[lines], lines, todevice, band, vendor)
    next === nothing && return nothing
    # A fresh process gets an empty device heap; it inherits stdout and JULIA_LOAD_PATH.
    flush(stdout)
    run(`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(Base.active_project()) $(@__FILE__) $name $pass $next:$(last(lines))`)
    return nothing
end

function parse_selection(s)
    parts = split(s, ':')
    length(parts) == 2 || error("rows must be first:last; got $s")
    a, b = parse.(Int, parts)
    1 <= a <= b <= 79 || error("rows must lie in 1:79; got $s")
    return a:b
end

function main(name, pass, selection)
    haskey(BACKENDS, name) ||
        error("backend must be one of $(join(sort!(collect(keys(BACKENDS))), ", ")); got $name")
    pass in PASSES || error("pass must be one of $(join(PASSES, ", ")); got $pass")
    package, array, band = BACKENDS[name]
    package === nothing && return probe_all(name, nothing, identity, band, pass, selection)
    Core.eval(Main, Expr(:using, Expr(:., package)))
    vendor = Base.invokelatest(getfield, Main, package)
    A = Base.invokelatest(getfield, vendor, array)
    Base.invokelatest(probe_all, name, vendor, x -> A(x), band, pass, selection)
end

length(ARGS) in (2, 3) ||
    error("usage: y3_premises.jl <cpu|metal|cuda|rocm> <p1-plain|p1-inline|p2|p3> [first:last]")
main(ARGS[1], ARGS[2], length(ARGS) == 3 ? parse_selection(ARGS[3]) : nothing)
