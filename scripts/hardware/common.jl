# Shared helpers of the measurement scripts in `scripts/hardware/` and of
# `scripts/r2_throughput.jl`: the choice of a device, the header that names the commit, the
# versions, the device and the load of a table, and the timing of a call.
#
# A script asked for a device that is not present prints one line that names the device and
# returns, so that the process exits 0 with no table row for it. It never falls back to another
# device. A device is present when its package is installed in an environment on the load path and,
# for Metal and CUDA, `functional()` holds.

using KernelAbstractions: KernelAbstractions
using LinearAlgebra: LinearAlgebra

const DEVICES = ("cpu", "jlarray", "metal", "cuda")

# The device of a script: `(name, array, mod)`, with `array` the array type of the device and `mod`
# the module of its package (`nothing` for `cpu`), or `nothing` after one line that names a device
# that is not present.
function select_device(name::AbstractString; io::IO = stdout)
    lname = lowercase(strip(name))
    lname in DEVICES ||
        throw(ArgumentError("unknown device $(repr(name)); expected one of $(join(DEVICES, ", "))"))
    lname == "cpu" && return (name = "cpu", array = Array, mod = nothing)
    pkg = lname == "jlarray" ? "JLArrays" : lname == "metal" ? "Metal" : "CUDA"
    id = Base.identify_package(pkg)
    if id === nothing
        println(io, "device $lname: not present ($pkg is not installed on the load path); no rows")
        return nothing
    end
    mod = Base.require(id)
    if lname == "jlarray"
        return (name = lname, array = Base.invokelatest(() -> mod.JLArray), mod = mod)
    end
    if !Base.invokelatest(mod.functional)
        println(io, "device $lname: not present ($pkg.functional() is false); no rows")
        return nothing
    end
    array = lname == "metal" ? Base.invokelatest(() -> mod.MtlArray) :
            Base.invokelatest(() -> mod.CuArray)
    return (name = lname, array = array, mod = mod)
end

# The model name of the device: the CPU for `cpu` and `jlarray`, whose kernels run on the host.
function device_label(dev)
    dev.name == "metal" && return string(Base.invokelatest(dev.mod.device))
    dev.name == "cuda" && return Base.invokelatest(() -> dev.mod.name(dev.mod.device()))
    return Sys.cpu_info()[1].model
end

# The commit of the checkout that holds this script, with `-dirty` for uncommitted changes, or
# `unknown` where `git` cannot read it, such as in a copy of the files without the repository.
function script_commit()
    dir = @__DIR__
    try
        return readchomp(`git -C $dir describe --always --dirty --abbrev=7`)
    catch
        return "unknown"
    end
end

# The header of a table: the commit, the Julia, KernelAbstractions and device package versions,
# the device, the threads and the load, each on one line beginning with `#`.
function print_header(io::IO, title::AbstractString, dev; extra = Pair{String, String}[])
    println(io, "# ", title)
    println(io, "# commit: ", script_commit())
    println(io, "# julia: ", VERSION)
    println(io, "# KernelAbstractions: ", Base.pkgversion(KernelAbstractions))
    if dev.mod !== nothing
        println(io, "# ", nameof(dev.mod), ".jl: ", Base.pkgversion(dev.mod))
    end
    println(io, "# device: ", dev.name, " (", device_label(dev), ")")
    println(io, "# threads: julia ", Threads.nthreads(), ", BLAS ",
        LinearAlgebra.BLAS.get_num_threads())
    la = Sys.loadavg()
    println(io, "# load: ", join(round.(la; digits = 2), " "), " (1, 5, 15 min)")
    for (k, v) in extra
        println(io, "# ", k, ": ", v)
    end
    return nothing
end

# Wait for the work queued on the device of `a`. A host `Array` needs no wait.
device_sync(a::Array) = nothing
function device_sync(a::AbstractArray)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(a))
end

# The median and the minimum wall time in seconds of `f()` over `reps` calls after one warm-up
# call, each call followed by `sync()` inside the timed interval.
function time_calls(f, sync; reps::Int = 5)
    reps >= 1 || throw(ArgumentError("reps must be at least 1, got $reps"))
    f()
    sync()
    ts = Vector{Float64}(undef, reps)
    for r in 1:reps
        t0 = time_ns()
        f()
        sync()
        ts[r] = (time_ns() - t0) / 1e9
    end
    return (median = middle(ts), min = minimum(ts))
end

# The median of the values `ts`, which must not be empty.
function middle(ts::AbstractVector{<:Real})
    s = sort(ts)
    h = (length(s) + 1) ÷ 2
    return isodd(length(s)) ? s[h] : (s[h] + s[h + 1]) / 2
end

# Whether the phase times `phases` sum to within 10 % of the measured total `total`.
phases_agree(phases, total) = abs(sum(phases) - total) <= 0.1 * total

# The note of a table row whose phases do not sum to within 10 % of its measured total.
function phase_note(phases, total)
    phases_agree(phases, total) ? "" :
    "phases sum to $(round(100 * sum(phases) / total; digits = 1)) % of the total"
end
