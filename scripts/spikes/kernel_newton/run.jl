# R1 in a kernel — part E of the design (§10.E, §7 P0.2, P0.4, P0.5).
#
# The spike measures the R1 design on each backend: an SVector Newton iteration without a line
# search, a partially pivoted LU with Int32 pivots, and the analytic Jacobian, in two
# storage variants — a local MMatrix and functional SMatrix updates — plus the
# SimpleNonlinearSolve baseline. A KA kernel runs one system per thread (the R2 launcher of
# §1.1); the CPU rows are a threaded host loop over the same R1 code.
#
# Usage, from the repository root, as `test/gpu/README.md` step 5 gives it:
#     julia --startup-file=no --project=scripts/spikes/kernel_newton scripts/spikes/kernel_newton/run.jl <backend> [variant] [tmode] [n]
# with <backend> one of cpu, metal, cuda, rocm. A GPU backend stacks its vendor environment behind
# this one through JULIA_LOAD_PATH; where the two manifests differ, step 5's combined environment
# runs the same command. The optional variant (M, S, SNS), tmode (F32, F64, MIXED) and n restrict
# the run to the matching rows. The CPU rows need --threads=auto.
#
# The output is one ROW line per (backend, variant, tmode, n) with the seven columns of §10.E
# check 3 and five raw timing values per timing column, so verify.jl re-derives every cell.
# Compile time is measured in a fresh child process per row: the first launch with synchronize at
# top level, less the warm median the parent measures.

using KernelAbstractions
using StaticArrays
using ForwardDiff
using LinearAlgebra
using Random
using Statistics: median
using SciMLBase
using SimpleNonlinearSolve
using GPUCompiler
using GPUCompiler: @device_code_llvm

# The CPU loop runs no matrix routine, so one BLAS thread keeps the host timing honest.
LinearAlgebra.BLAS.set_num_threads(1)

for f in ("grid.jl", "problem.jl", "solvers.jl", "sns.jl", "measure.jl", "kernels.jl", "rows.jl")
    include(joinpath(@__DIR__, f))
end

# --- the backends ------------------------------------------------------------------------------

const VENDORS = Dict(:metal => :Metal, :cuda => :CUDA, :rocm => :AMDGPU)
const KA_BACKENDS = Dict(:metal => :MetalBackend, :cuda => :CUDABackend, :rocm =>
    :ROCBackend)

# Load the vendor package of a backend. Its methods exist only in the world after the load, so
# a caller reaches them through one `invokelatest`.
function load_backend(backend::Symbol)
    backend === :cpu && return nothing
    Core.eval(Main, :(using $(VENDORS[backend])))
    return Base.invokelatest(getfield, Main, VENDORS[backend])
end

function ka_backend(backend::Symbol, vendor)
    backend === :cpu && return KernelAbstractions.CPU()
    return getfield(vendor, KA_BACKENDS[backend])()
end

function card_name(backend::Symbol, vendor)
    backend === :cpu && return Sys.CPU_NAME
    backend === :metal && return sprint(show, vendor.device())
    backend === :cuda && return vendor.name(vendor.device())
    return string(vendor.device())
end

function package_versions(vendor)
    mods = Module[
        KernelAbstractions, StaticArrays, ForwardDiff, SimpleNonlinearSolve, SciMLBase,
        GPUCompiler]
    vendor === nothing || push!(mods, vendor)
    return join(("$(nameof(m)) $(pkgversion(m))" for m in mods), ", ")
end

# --- the compile-time child ----------------------------------------------------------------------

# A fresh Julia process per (backend, variant, tmode, n): the first launch with synchronize at
# top level. Nothing else runs in it, so no earlier compile can hide in the time.
function compile_time_child(
        project::String, backend::Symbol, variant::String, tmode::String, n::Int)
    cmd = `$(joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia")) --startup-file=no --project=$(project) -- $(@__DIR__)/run.jl
           --compile-one $(backend) $(variant) $(tmode) $(n)`
    out = try
        read(cmd, String)
    catch e
        @warn "the compile-time child failed" backend variant tmode n exception = e
        return nothing
    end
    m = match(r"^compile_s=([0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?)$"m, out)
    return m === nothing ? nothing : parse(Float64, m.captures[1])
end

# The child's launch arguments, prepared without launching.
function compile_one(backend::Symbol, vendor, variant::String, tmode::String, n::Int)
    kb = ka_backend(backend, vendor)
    A = make_A(T_of(tmode), n)
    batch = make_batch(A, 1)
    out, iters = out_pair(kb, batch, 1)
    return kb, SOLVERS[(variant, tmode)], out, iters, A, to_device(kb, batch, 1), MAXIT
end

# --- the output row ------------------------------------------------------------------------------

# key="value" with the value's double quotes turned into single ones, so verify.jl parses a field
# without an escape rule.
function row_text(row::Dict{String, String})
    parts = String[]
    for k in sort!(collect(keys(row)))
        v = replace(row[k], "\"" => "'")
        push!(parts, "$k=\"$v\"")
    end
    return join(parts, " ")
end

# The rows of a backend, or the ones that match `variant`, `tmode` and `n` where these are given.
function main(io::IO, backend::AbstractString; variant = "", tmode = "", n = -1)
    be = Symbol(backend)
    be in (:cpu, :metal, :cuda, :rocm) || error("backend must be cpu, metal, cuda or rocm")
    be === :cpu && Threads.nthreads() == 1 &&
        error("the CPU baseline needs one thread per core: start Julia with --threads=auto")
    vendor = load_backend(be)
    return Base.invokelatest(run_rows, io, be, vendor, variant, tmode, n)
end

main(backend::AbstractString; kwargs...) = main(stdout, backend; kwargs...)

function run_rows(io::IO, be::Symbol, vendor, want_variant, want_tmode, want_n)
    println(io, "GeometricSolvers kernel_newton spike -- backend = ", be)
    println(io, "Julia ", VERSION)
    println(io, "Card: ", card_name(be, vendor))
    println(io, "Packages: ", package_versions(vendor))
    println(io, "Compile cap: ", COMPILE_CAP_S, " s; timing runs: ", TIMING_RUNS)
    project = Base.active_project()
    for (variant, tmode) in rows_of(string(be)), n in NS

        (isempty(want_variant) || variant == want_variant) || continue
        (isempty(want_tmode) || tmode == want_tmode) || continue
        (want_n < 0 || n == want_n) || continue
        compile_s = compile_time_child(project, be, variant, tmode, n)
        row = measure_row(be, vendor, variant, tmode, n, compile_s)
        row["julia"] = string(VERSION)
        row["packages"] = package_versions(vendor)
        row["card"] = card_name(be, vendor)
        println(io, "ROW ", row_text(row))
        flush(io)
    end
    return nothing
end

# --- the command line ----------------------------------------------------------------------------
#
# The child loads its vendor package, prepares, reads the clock and launches in four top-level
# statements. Each statement runs in the world the one before left, and the launch compiles inside
# the timed statement.

const SCRIPT = abspath(PROGRAM_FILE) == @__FILE__
const COMPILE_CHILD = SCRIPT && get(ARGS, 1, "") == "--compile-one"
COMPILE_CHILD && (child_vendor = load_backend(Symbol(ARGS[2])))
COMPILE_CHILD &&
    (child = compile_one(
        Symbol(ARGS[2]), child_vendor, ARGS[3], ARGS[4], parse(Int, ARGS[5])))
child_start = time_ns()
COMPILE_CHILD && launch(child...)
COMPILE_CHILD && println("compile_s=", (time_ns() - child_start) / 1e9)

if SCRIPT && !COMPILE_CHILD
    main(get(ARGS, 1, "cpu"); variant = get(ARGS, 2, ""), tmode = get(ARGS, 3, ""),
        n = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : -1)
end
