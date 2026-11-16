# cuSOLVER `getrf` in `Float64` on a CUDA device, with the relative residual of the solve, under
# cuSOLVER's default math mode and under its `Float64` fixed-point emulation mode. The emulated
# row is not yet measured on a device.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=<env> scripts/hardware/cusolver_getrf.jl [n,…] [reps]
#
# `n,…` is a comma-separated list of sizes, `2048,8192` by default; `reps` is the number of timed
# factorisations after one warm-up, 5 by default. `<env>` holds GeometricSolvers,
# KernelAbstractions and CUDA.jl. Without a functional CUDA device the script prints one line and
# no rows.
#
# Each size has two rows. The math mode is a property of cuSOLVER's dense handle, which
# `CUDA.math_mode!` does not set: that call sets the cuBLAS handle alone. `default` runs with
# `CUSOLVER_DEFAULT_MATH` on the dense handle. `FixedPoint` sets
# `cusolverDnSetMathMode(dense_handle(), CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)` before the row
# and sets `CUSOLVER_DEFAULT_MATH` again after it, also when the row throws. The column `cuSOLVER
# mode read back` is the mode that `cusolverDnGetMathMode` returns for the row. Switches that
# cuBLAS reads from the environment act on the whole process, so a run with them is a second
# process: the header lists every environment variable whose name begins with `CUBLAS_`.
#
# `getrf` is the median and the minimum wall time in ms of a device copy of the matrix followed by
# `cuSOLVER.getrf!` on it, which returns the pivot status to the host and so waits for the
# factorisation.
# `residual` is `‖A x - b‖∞ / (‖A‖∞ ‖x‖∞)` of `getrs!`'s solution, computed on the host.

using KernelAbstractions: KernelAbstractions
using LinearAlgebra
using Printf
using Random: Xoshiro

include(joinpath(@__DIR__, "common.jl"))

function cublas_environment()
    vars = sort!([k for k in keys(ENV) if startswith(k, "CUBLAS_")])
    isempty(vars) && return "none"
    return join(("$k=$(ENV[k])" for k in vars), ", ")
end

function cusolver_math_mode(cuSOLVER)
    mode = Ref{cuSOLVER.cusolverMathMode_t}()
    cuSOLVER.cusolverDnGetMathMode(cuSOLVER.dense_handle(), mode)
    return mode[]
end

function measure(io, CUDA, n::Int, mode::String; reps::Int)
    cuSOLVER = CUDA.cuSOLVER
    if mode == "FixedPoint"
        cuSOLVER.cusolverDnSetMathMode(cuSOLVER.dense_handle(),
            cuSOLVER.CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)
    end
    try
        return measure_row(io, CUDA, n, mode, cusolver_math_mode(cuSOLVER); reps)
    finally
        cuSOLVER.cusolverDnSetMathMode(cuSOLVER.dense_handle(), cuSOLVER.CUSOLVER_DEFAULT_MATH)
    end
end

function measure_row(io, CUDA, n::Int, mode::String, read_back; reps::Int)
    rng = Xoshiro(n)
    A = randn(rng, Float64, n, n)
    b = randn(rng, Float64, n)
    A_dev = CUDA.CuArray(A)
    F_dev = similar(A_dev)
    ipiv = CUDA.CuArray{Int32}(undef, n)
    info = Ref(0)
    t = time_calls(
        () -> begin
            copyto!(F_dev, A_dev)
            _, _, info[] = CUDA.cuSOLVER.getrf!(F_dev, ipiv)
        end, () -> CUDA.synchronize(); reps)
    info[] == 0 || error("getrf! at n = $n returned info = $(info[])")
    x_dev = CUDA.CuArray(b)
    CUDA.cuSOLVER.getrs!('N', F_dev, ipiv, x_dev)
    x = Array(x_dev)
    res = norm(A * x - b, Inf) / (opnorm(A, Inf) * norm(x, Inf))
    @printf(io, "| %d | %s | %s | %.3f / %.3f | %.2e |\n", n, mode, read_back,
        1e3 * t.median, 1e3 * t.min, res)
    return (n = n, mode = mode, read_back = read_back, getrf = t, residual = res)
end

function run_tables(io, dev, ns, reps)
    CUDA = dev.mod
    print_header(io, "cuSOLVER getrf in Float64 (scripts/hardware/cusolver_getrf.jl)", dev;
        extra = ["CUDA runtime" => string(CUDA.runtime_version()),
            "cuBLAS" => string(CUDA.cuBLAS.version()),
            "cuSOLVER" => string(CUDA.cuSOLVER.version()),
            "environment" => cublas_environment(),
            "reps" => string(reps)])
    println(io,
        "| n | math mode | cuSOLVER mode read back | getrf ms (median / min) | residual |")
    println(io, "|--:|:--|:--|--:|--:|")
    return [measure(io, CUDA, n, mode; reps) for n in ns
            for mode in ("default", "FixedPoint")]
end

function main(args = ARGS; io::IO = stdout)
    dev = select_device("cuda"; io)
    dev === nothing && return nothing
    ns = length(args) >= 1 ? parse.(Int, split(args[1], ',')) : [2048, 8192]
    reps = length(args) >= 2 ? parse(Int, args[2]) : 5
    return Base.invokelatest(run_tables, io, dev, ns, reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
