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

# The first line of an error that names an allocator, or nothing.
function allocator_line(e)
    lines = split(sprint(showerror, e), "\n")
    i = findfirst(l -> occursin(ALLOC_ERROR, l), lines)
    return i === nothing ? nothing : String(strip(lines[i]))
end

function measure_row(backend::Symbol, vendor, variant::String, tmode::String, n::Int,
        compile_s::Union{Float64, Nothing})
    row = new_row(string(backend), variant, tmode, n)
    T = T_of(tmode)
    A = make_A(T, n)
    row["kappa"] = string(cond(Array(A), Inf))
    solver = SOLVERS[(variant, tmode)]
    kb = ka_backend(backend, vendor)
    is_device = backend !== :cpu

    # --- compile_s: the child's first launch, less the warm median ---------------------------
    first_launch = compile_s === nothing ? "does not compile: compile-time child failed" :
                   string(compile_s)
    row["compile_s"] = first_launch
    row["compile_cap_s"] = string(COMPILE_CAP_S)

    # --- the first launch: compiles, answer, allocates -----------------------------------------

    ans = make_batch(A, NS_ANSWER)
    out, iters = out_pair(kb, ans, NS_ANSWER)
    argb = to_device(kb, ans, NS_ANSWER)

    ir = ""
    try
        if is_device
            io = IOBuffer()
            @device_code_llvm io = io newton_kernel!(kb, 64)(
                out, iters, solver, A, argb, MAXIT; ndrange = NS_ANSWER)
            KernelAbstractions.synchronize(kb)
            ir = String(take!(io))
        else
            launch(kb, solver, out, iters, A, argb, MAXIT)
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
        count_alloc = count(ALLOC_CALL, ir)
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
        obj = newton_kernel!(kb, 64)
        nd, _, space, _ = KernelAbstractions.launch_config(obj, NS_ANSWER, nothing)
        ctx = KernelAbstractions.mkcontext(obj, nd, space)
        args = vendor.cudaconvert.((ctx, out, iters, solver, A, argb, MAXIT))
        compiled = vendor.cufunction(obj.f, Tuple{typeof.(args)...};
            maxthreads = 64, always_inline = kb.always_inline)
        row["cuda_registers"] = string(vendor.registers(compiled))
        row["cuda_local_bytes"] = string(getproperty(vendor.memory(compiled), :local))
    end

    # --- answer ---------------------------------------------------------------------------------

    xout = Array(out)
    xsol = [solver(A, b, MAXIT)[1] for b in ans.b]
    err, err_cpu = answer_errors(xout, ans, xsol)
    it = Array(iters)
    iters_row = variant == "SNS" ? -1 : maximum(it)
    resnorm = maximum(maxabs(residual(A, ans.b[i], xout[i])) for i in eachindex(xout))
    # The strictest batch bound is a sufficient bound on both maximum errors.
    xs_inf = minimum(maxabs(x) for x in ans.xs)
    bound = 16 * n * eps(T) * xs_inf
    row["xstar_inf"] = string(xs_inf)
    row["bound"] = string(bound)
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
    if err <= bound && err_cpu <= bound
        row["answer"] = "right"
    elseif iters_row >= MAXIT
        row["answer"] = "wrong: maxiters"
    else
        row["answer"] = "wrong: $err"
    end

    # --- the three throughput cells -------------------------------------------------------------

    # The warm median of a launch, for the compile-time subtraction.
    warm = timings(() -> launch(kb, solver, out, iters, A, argb, MAXIT))
    row["warm_s"] = timings_text(warm)
    row["first_launch_s"] = first_launch
    compile_cap = false
    if compile_s !== nothing
        compile = max(0.0, compile_s - median(warm))
        row["compile_s"] = string(compile)
        compile_cap = compile > COMPILE_CAP_S
    end
    row["threads"] = string(Threads.nthreads())
    row["blas_threads"] = string(LinearAlgebra.BLAS.get_num_threads())

    # A device Float64 row gives an iteration count and a residual, but no timing (decision 6).
    not_timed_f64 = is_device && tmode == "F64"
    batch = compile_cap || not_timed_f64 ? nothing : make_batch(A, maximum(NSYS))
    for (cell, count) in zip(("solves_1e4", "solves_1e5", "solves_1e6"), NSYS)
        if compile_cap
            row[cell] = "not run: compile cap"
        elseif not_timed_f64
            row[cell] = "not timed: device Float64"
        else
            ts = solves(kb, solver, A, batch, count)
            row[cell] = timings_text(ts, count)
            row["rate_" * cell] = string(round(count / median(ts), sigdigits = 6))
            row["spread_" * cell] = string(round((maximum(ts) - minimum(ts)) / median(ts), sigdigits = 6))
        end
    end
    return row
end

# A warm launch, then TIMING_RUNS timed launches on the first `count` systems of a batch.
function solves(kb, solver, A, batch::Batch, count::Int)
    b = to_device(kb, batch, count)
    out, iters = out_pair(kb, batch, count)
    launch(kb, solver, out, iters, A, b, MAXIT)
    return timings(() -> launch(kb, solver, out, iters, A, b, MAXIT))
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
