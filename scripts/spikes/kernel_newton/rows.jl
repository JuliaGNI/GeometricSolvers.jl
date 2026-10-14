# One row of the grid: the seven columns of §10.E check 3 and the evidence behind them.
#
# A CPU row is a threaded host loop over the same R1 code — the baseline of §10.E. A device row
# is the KA kernel of `kernels.jl`, one system per thread, whose first launch is captured in the
# device LLVM for the allocates cell.

# The first line of an error. A task failure hides its cause in the line after
# "nested task error", so take that one.
function firstline(e)
    text = sprint(showerror, e)
    if e isa TaskFailedException
        m = match(r"^nested task error: (.*)$"m, text)
        m === nothing || (text = m.captures[1])
    end
    return first(split(text, "\n"))
end

# A device compile failure caused by a heap allocation. The InvalidIRError header names no
# allocator; its Reason lines do: the GC allocator, the runtime's `Memory` allocator
# jl_alloc_genericmemory, or an object that a device without a GC cannot allocate.
const ALLOC_ERROR = r"jl_gc|gc_alloc|malloc|pool_alloc|jl_alloc_genericmemory|unsupported allocation"

# The first line of an error that names an allocator, or nothing.
function allocator_line(e)
    lines = split(sprint(showerror, e), "\n")
    i = findfirst(l -> occursin(ALLOC_ERROR, l), lines)
    return i === nothing ? nothing : String(strip(lines[i]))
end

# The CPU path: one system per host thread (§1.1, R2 on the host).
function cpu_launch(f, A, bs, out, iters, maxit)
    Threads.@threads for i in eachindex(bs)
        x, it, info = f(A, bs[i], maxit)
        out[i] = x
        iters[i] = info == 0 ? Int32(it) : Int32(-info)
    end
    return nothing
end

# The batches of one row, discarded before the next row to bound retained host storage.
struct Cache
    batches::Dict{Tuple{String, Int, Int}, Any}
end
Cache() = Cache(Dict{Tuple{String, Int, Int}, Any}())

function cached(cache::Cache, tmode::String, n::Int, count::Int)
    key = (tmode, n, count)
    haskey(cache.batches, key) && return cache.batches[key]
    value = make_batch(tmode, n, count)
    cache.batches[key] = value
    return value
end

# A fresh output pair for a batch, on the device and on the host.
function out_pair_for(kb, batch::Batch{T, n}, count::Int) where {T, n}
    out = KernelAbstractions.allocate(kb, SVector{n, T}, (count,))
    iters = KernelAbstractions.allocate(kb, Int32, (count,))
    copyto!(out, [zero(SVector{n, T}) for _ in 1:count])
    copyto!(iters, zeros(Int32, count))
    return out, iters
end

function out_pair_for(batch::Batch{T, n}, count::Int) where {T, n}
    out = [zero(SVector{n, T}) for _ in 1:count]
    iters = zeros(Int32, count)
    return out, iters
end

function measure_row(backend::Symbol, vendor, cache::Cache,
        variant::String, tmode::String, n::Int,
        compile_s::Union{Float64, Nothing})
    row = new_row(string(backend), variant, tmode, n)
    T = T_of(tmode)
    A = make_A(T, n)
    row["kappa"] = string(cond(Array(A), Inf))
    solver = SOLVERS[(variant, tmode)]
    kernel = KERNELS[tmode == "MIXED" ? "MIXED-" * variant : variant]
    kb = backend === :cpu ? KernelAbstractions.CPU() : ka_backend(backend, vendor)
    is_device = backend !== :cpu

    # --- compile_s: the child's first launch, less the warm median ---------------------------
    row["compile_s"] = compile_s === nothing ?
                       "does not compile: compile-time child failed" :
                       string(compile_s)
    row["compile_cap_s"] = string(COMPILE_CAP_S)

    # --- the first launch: compiles, answer, allocates -----------------------------------------

    ans = cached(cache, tmode, n, NS_ANSWER)
    if is_device
        out, iters = out_pair_for(kb, ans, NS_ANSWER)
        argb = to_device(kb, ans, NS_ANSWER)
    else
        out, iters = out_pair_for(ans, NS_ANSWER)
        argb = ans.b
    end

    ir = ""
    try
        if is_device
            io = IOBuffer()
            @device_code_llvm io = io kernel(kb, 64)(
                out, iters, A, argb, MAXIT; ndrange = NS_ANSWER)
            KernelAbstractions.synchronize(kb)
            ir = String(take!(io))
        else
            cpu_launch(solver, A, ans.b, out, iters, MAXIT)
        end
        row["compiles"] = "yes"
    catch e
        text = "does not compile: " * firstline(e)
        row["compiles"] = text
        mark_columns!(row, text, (
            "answer", "allocates", "solves_1e4", "solves_1e5", "solves_1e6"))
        reason = is_device ? allocator_line(e) : nothing
        if reason !== nothing
            row["alloc_method"] = "device_compile_allocator_error"
            row["alloc_count"] = "1"
            row["allocates"] = "yes"
            row["alloc_error"] = reason
        end
        return row
    end

    # --- allocates ------------------------------------------------------------------------------

    if is_device
        count_alloc = device_alloc_count(ir)
        row["alloc_method"] = "device_llvm_allocator_calls"
        row["alloc_calls"] = join((m.match for m in eachmatch(ALLOC_CALL, ir)), "|")
    else
        probe_alloc(solver, A, ans.b[1])
        samples = [probe_alloc(solver, A, ans.b[1]) for _ in 1:TIMING_RUNS]
        count_alloc = minimum(samples)
        row["alloc_samples"] = join(samples, ",")
        row["alloc_method"] = "host_allocated_function_barrier"
    end
    row["alloc_count"] = string(count_alloc)
    row["allocates"] = count_alloc == 0 ? "no" : "yes"
    if backend === :cuda
        obj = kernel(kb, 64)
        nd, _, space, _ = KernelAbstractions.launch_config(obj, NS_ANSWER, nothing)
        ctx = KernelAbstractions.mkcontext(obj, nd, space)
        args = vendor.cudaconvert.((ctx, out, iters, A, argb, MAXIT))
        compiled = vendor.cufunction(obj.f, Tuple{typeof.(args)...};
            maxthreads = 64, always_inline = kb.always_inline)
        row["cuda_registers"] = string(vendor.registers(compiled))
        row["cuda_local_bytes"] = string(getproperty(vendor.memory(compiled), :local))
    end

    # --- answer ---------------------------------------------------------------------------------

    xout = is_device ? Array(out) : out
    xsol = [solver(A, b, MAXIT)[1] for b in ans.b]
    err, err_cpu = answer_errors(xout, ans, xsol)
    it = is_device ? Array(iters) : iters
    iters_row = variant == "SNS" ? -1 : maximum(it)
    resnorm = maximum(maxabs(residual(A, ans.b[i], xout[i])) for i in eachindex(xout))
    # The strictest batch bound is a sufficient bound on both maximum errors.
    xs_inf = minimum(maxabs(x) for x in ans.xs)
    row["xstar_inf"] = string(xs_inf)
    row["bound"] = string(16 * n * eps(T) * xs_inf)
    row["err"] = string(err)
    row["err_cpu"] = string(err_cpu)
    row["iters"] = string(iters_row)
    row["iters_kind"] = variant == "SNS" ? "unknown" : "newton_iterations"
    if variant == "SNS"
        row["retcode"] = join(string.(ReturnCode.T.(sort!(unique(it)))), ",")
    else
        row["info"] = string(minimum(it) < 0 ? -minimum(it) : 0)
    end
    row["resnorm"] = string(resnorm)
    if err <= parse(Float64, row["bound"]) && err_cpu <= parse(Float64, row["bound"])
        row["answer"] = "right"
    elseif iters_row >= MAXIT
        row["answer"] = "wrong: maxiters"
    else
        row["answer"] = "wrong: $err"
    end

    # --- the three throughput cells -------------------------------------------------------------

    # The warm median of a launch, for the compile-time subtraction.
    warm = Float64[]
    for _ in 1:TIMING_RUNS
        t0 = time_ns()
        if is_device
            launch(kernel, kb, out, iters, A, argb, MAXIT)
        else
            cpu_launch(solver, A, ans.b, out, iters, MAXIT)
        end
        push!(warm, (time_ns() - t0) / 1e9)
    end
    row["warm_s"] = timings_text(warm)
    row["first_launch_s"] = compile_s === nothing ?
                            "does not compile: compile-time child failed" :
                            string(compile_s)
    if compile_s !== nothing
        row["compile_s"] = string(max(0.0, compile_s - median(warm)))
    end
    compile_cap = compile_s !== nothing && parse(Float64, row["compile_s"]) > COMPILE_CAP_S
    row["threads"] = string(Threads.nthreads())
    row["blas_threads"] = string(LinearAlgebra.BLAS.get_num_threads())

    # A device Float64 row gives an iteration count and a residual, but no timing (decision 6).
    not_timed_f64 = is_device && tmode == "F64"
    for (cell, count) in zip(("solves_1e4", "solves_1e5", "solves_1e6"), NSYS)
        if compile_cap
            row[cell] = "not run: compile cap"
        elseif not_timed_f64
            row[cell] = "not timed: device Float64"
        else
            big = cached(cache, tmode, n, count)
            if is_device
                bdev = to_device(kb, big, count)
                odev, idev = out_pair_for(kb, big, count)
                ts = device_solves(kernel, kb, odev, idev, A, bdev, count)
            else
                ts = cpu_solves(solver, A, big, count)
            end
            row["threads"] = string(Threads.nthreads())
            row["blas_threads"] = string(LinearAlgebra.BLAS.get_num_threads())
            row[cell] = timings_text(ts, count)
            row["rate_" * cell] = string(round(count / median(ts), sigdigits = 6))
            row["spread_" * cell] = string(round((maximum(ts) - minimum(ts)) / median(ts), sigdigits = 6))
        end
    end
    return row
end

# Five runs, launch to synchronize, on the device.
function device_solves(kernel, kb, out, iters, A, bdev, count)
    launch(kernel, kb, out, iters, A, bdev, MAXIT)
    ts = Float64[]
    for _ in 1:TIMING_RUNS
        t0 = time_ns()
        launch(kernel, kb, out, iters, A, bdev, MAXIT)
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return ts
end

# Five runs of the threaded host loop.
function cpu_solves(f, A, batch::Batch, count)
    bs = view(batch.b, 1:count)
    out, iters = out_pair_for(batch, count)
    cpu_launch(f, A, bs, out, iters, MAXIT)
    ts = Float64[]
    for _ in 1:TIMING_RUNS
        t0 = time_ns()
        cpu_launch(f, A, bs, out, iters, MAXIT)
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return ts
end

# The five raw values, and the solves per second when the batch size is known. All five values go
# into the row so verify.jl can recompute the median and the spread (§10.E check 4).
function timings_text(ts::Vector{Float64}, count::Int = 0)
    values = join((string(round(t, sigdigits = 8)) for t in ts), ",")
    if count == 0
        return values
    end
    return values * " solves_per_s=" * string(round(count / median(ts), sigdigits = 6))
end
