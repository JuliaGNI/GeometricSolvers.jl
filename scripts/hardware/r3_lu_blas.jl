# The R3 LU on the CPU with the BLAS and LAPACK that Julia ships, OpenBLAS, and with Apple's
# Accelerate through AppleAccelerate.jl: `factorize!!` and `ldiv!!` of `LUFactorization` on an
# `Array`, in `Float32` and `Float64`.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=<env> scripts/hardware/r3_lu_blas.jl [library,…] [n,…] [reps]
#
# `library,…` is `openblas,accelerate` by default; `n,…` is `64,256,1024` by default; `reps` is the
# number of timed calls after one warm-up call, 5 by default. `<env>` holds GeometricSolvers and
# KernelAbstractions, and AppleAccelerate.jl for `accelerate`. The libraries run in the order given:
# loading AppleAccelerate forwards the BLAS and LAPACK of the whole process to Accelerate, so
# `openblas` after `accelerate` is refused. Where AppleAccelerate is not installed, the script
# prints one line for `accelerate` and no rows for it.
#
# `factorize` and `solve` are the median and the minimum wall time in ms of one call; `residual` is
# `‖A x - b‖∞ / (‖A‖∞ ‖x‖∞)` of the last solve, in `Float64`. A line before the rows of each library
# names the libraries that libblastrampoline forwards to.

using GeometricSolvers
using GeometricSolvers: LUFactorization, NonlinearProblem, dense_factor, factorize!!, ldiv!!
using KernelAbstractions: KernelAbstractions
using LinearAlgebra
using Printf
using Random: Xoshiro

include(joinpath(@__DIR__, "common.jl"))

const LIBRARIES = ("openblas", "accelerate")

lu_noop(r, x, p) = nothing

blas_libraries() = join((basename(l.libname) for l in BLAS.get_config().loaded_libs), ", ")

# Make `lib` the library of BLAS and LAPACK, or return `false` after one line where it is not
# present.
function select_library(io, lib::String)
    if lib == "openblas"
        occursin("openblas", lowercase(blas_libraries())) ||
            throw(ArgumentError("openblas after accelerate: run it in a process of its own"))
        return true
    end
    id = Base.identify_package("AppleAccelerate")
    if id === nothing
        println(
            io, "library accelerate: not present (AppleAccelerate is not installed on ",
            "the load path); no rows")
        return false
    end
    Base.require(id)
    occursin("accelerate", lowercase(blas_libraries())) ||
        error("AppleAccelerate loaded, but libblastrampoline forwards to $(blas_libraries())")
    return true
end

function measure(io, lib, ::Type{T}, n::Int; reps::Int) where {T}
    rng = Xoshiro(n)
    A = randn(rng, T, n, n)
    b = randn(rng, T, n)
    x = zeros(T, n)
    d = similar(x)
    m = LUFactorization()
    fac = dense_factor(m, NonlinearProblem(x, lu_noop), x, nothing)
    tf = time_calls(() -> (fac = factorize!!(fac, m, A)), () -> nothing; reps)
    fac.info == 0 || error("LU at n = $n returned info = $(fac.info)")
    ts = time_calls(() -> ldiv!!(d, fac, b), () -> nothing; reps)
    A64, d64, b64 = Float64.(A), Float64.(d), Float64.(b)
    res = norm(A64 * d64 - b64, Inf) / (opnorm(A64, Inf) * norm(d64, Inf))
    @printf(io, "| %s | %s | %d | %.4f / %.4f | %.4f / %.4f | %.2e |\n", lib, T, n,
        1e3 * tf.median, 1e3 * tf.min, 1e3 * ts.median, 1e3 * ts.min, res)
    return (library = lib, T = T, n = n, factorize = tf, solve = ts, residual = res)
end

function run_tables(io, libs, ns, reps)
    dev = (name = "cpu", array = Array, mod = nothing)
    print_header(io,
        "R3 LU on the CPU with OpenBLAS and Accelerate (scripts/hardware/r3_lu_blas.jl)",
        dev; extra = ["reps" => string(reps)])
    println(
        io, "| library | T | n | factorize ms (median / min) | solve ms (median / min) | ",
        "residual |")
    println(io, "|:--|:--|--:|--:|--:|--:|")
    rows = NamedTuple[]
    for lib in libs
        Base.invokelatest(select_library, io, lib) || continue
        println(io, "| $lib | libblastrampoline: $(blas_libraries()) | | | | |")
        for T in (Float32, Float64)
            for n in ns
                push!(rows, Base.invokelatest(measure, io, lib, T, n; reps))
            end
        end
    end
    return rows
end

function main(args = ARGS; io::IO = stdout)
    libs = length(args) >= 1 ? String.(split(args[1], ',')) : collect(LIBRARIES)
    for lib in libs
        lib in LIBRARIES || throw(ArgumentError(
            "unknown library $(repr(lib)); expected one of $(join(LIBRARIES, ", "))"))
    end
    ns = length(args) >= 2 ? parse.(Int, split(args[2], ',')) : [64, 256, 1024]
    reps = length(args) >= 3 ? parse(Int, args[3]) : 5
    return run_tables(io, libs, ns, reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
