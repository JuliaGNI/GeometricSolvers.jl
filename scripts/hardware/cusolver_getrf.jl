# cuSOLVER `getrf` in `Float64` on a CUDA device, with the relative residual of the solve, under
# the cuBLAS math mode of CUDA.jl that it runs with by default and under the `Float64` fixed-point
# emulation mode.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=<env> scripts/hardware/cusolver_getrf.jl [n,…] [reps]
#
# `n,…` is a comma-separated list of sizes, `2048,8192` by default; `reps` is the number of timed
# factorisations after one warm-up, 5 by default. `<env>` holds GeometricSolvers,
# KernelAbstractions and CUDA.jl. Without a functional CUDA device the script prints one line and
# no rows.
#
# Each size has two rows. `default` sets `CUDA.math_mode!(CUDA.DEFAULT_MATH)`, the mode that CUDA.jl
# starts with unless Julia runs with `--math-mode=fast`. `FixedPoint` sets
# `CUDA.math_mode!(CUDA.FAST_MATH; precision = :FixedPoint)`, which CUDA.jl 6.5 maps to
# `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` on its cuBLAS handle from cuBLAS 13.1 on. Whether that
# handle's mode reaches the products inside cuSOLVER's `getrf` is what the two rows show. Switches
# that cuBLAS reads from the environment act on the whole process, so a run with them is a second
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

function measure(io, CUDA, n::Int, mode::String; reps::Int)
    if mode == "FixedPoint"
        CUDA.math_mode!(CUDA.FAST_MATH; precision = :FixedPoint)
    else
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
    end
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
    @printf(io, "| %d | %s | %.3f / %.3f | %.2e |\n", n, mode, 1e3 * t.median, 1e3 * t.min,
        res)
    CUDA.math_mode!(CUDA.DEFAULT_MATH)
    return (n = n, mode = mode, getrf = t, residual = res)
end

function run_tables(io, dev, ns, reps)
    CUDA = dev.mod
    print_header(io, "cuSOLVER getrf in Float64 (scripts/hardware/cusolver_getrf.jl)", dev;
        extra = ["CUDA runtime" => string(CUDA.runtime_version()),
            "cuBLAS" => string(CUDA.cuBLAS.version()),
            "cuSOLVER" => string(CUDA.cuSOLVER.version()),
            "environment" => cublas_environment(),
            "reps" => string(reps)])
    println(io, "| n | math mode | getrf ms (median / min) | residual |")
    println(io, "|--:|:--|--:|--:|")
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
