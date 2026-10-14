# R1 in a kernel — part E of Tasks/Design GeometricSolvers.md (§10.E, §7 P0.2, P0.4, P0.5).
#
# The spike measures the R1 design on each backend: an SVector Newton iteration without a line
# search, a partially pivoted LU with Int32 pivots, and the Jacobian of `problem.jl`, in two
# storage variants — a local MMatrix and functional SMatrix updates — plus the
# SimpleNonlinearSolve baseline. A KA kernel runs one system per thread (the R2 launcher of
# §1.1); the CPU rows are a threaded host loop over the same R1 code.
#
# Usage, from the repository root, as `test/gpu/README.md` step 5 gives it:
#     julia --startup-file=no --project=scripts/spikes/kernel_newton scripts/spikes/kernel_newton/run.jl <backend>
# with <backend> one of cpu, metal, cuda, rocm. A GPU backend stacks its vendor environment behind
# this one through JULIA_LOAD_PATH; where the two manifests differ, step 5's combined environment
# runs the same command.
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
using Pkg

# The CPU loop runs no matrix routine, so one BLAS thread keeps the host timing honest.
LinearAlgebra.BLAS.set_num_threads(1)

for f in ("problem.jl", "solvers.jl", "sns.jl", "measure.jl", "kernels.jl", "rows.jl")
    include(joinpath(@__DIR__, f))
end

const NS = (2, 4, 6, 8, 12, 16, 24)

# --- the backends ------------------------------------------------------------------------------

function load_backend(backend::Symbol)
    name = backend === :metal ? :Metal :
           backend === :cuda ? :CUDA : backend === :rocm ? :AMDGPU : nothing
    name === nothing && return nothing
    Base.invokelatest() do
        Core.eval(Main, Expr(:using, Expr(:., name)))
    end
    return Base.invokelatest(getfield, Main, name)
end

function ka_backend(backend::Symbol, vendor)
    backend === :cpu && return KernelAbstractions.CPU()
    field = backend === :metal ? :MetalBackend :
            backend === :cuda ? :CUDABackend : :ROCBackend
    return Base.invokelatest(Base.invokelatest(getfield, vendor, field))
end

function card_name(backend::Symbol, vendor)
    backend === :cpu && return Sys.CPU_NAME
    backend === :metal && return Base.invokelatest(() -> sprint(show, vendor.device()))
    backend === :cuda && return Base.invokelatest(() -> vendor.name(vendor.device()))
    return Base.invokelatest(() -> string(vendor.device()))
end

function package_versions(backend::Symbol, vendor)
    mods = Module[
        KernelAbstractions,
        StaticArrays,
        ForwardDiff,
        SimpleNonlinearSolve,
        SciMLBase,
        GPUCompiler
    ]
    if backend !== :cpu && vendor !== nothing
        push!(mods, vendor)
    end
    return Base.invokelatest() do
        join(("$(nameof(m)) $(Pkg.Types.pkgversion(m))" for m in mods), ", ")
    end
end

# --- the rows of a backend ----------------------------------------------------------------------
#
# Metal has no Float64 (§1.4). CUDA carries the mixed F64/F32 and all-F64 rows (§10.E).
# SNS is a Float32 variant.

function rows_of(backend::Symbol)
    rows = (("M", "F32"), ("S", "F32"), ("SNS", "F32"))
    backend in (:cpu, :metal) && return rows
    rows = (rows..., ("M", "F64"), ("S", "F64"))
    backend === :cuda && return (rows..., ("M", "MIXED"), ("S", "MIXED"))
    return rows
end

# --- the compile-time child ----------------------------------------------------------------------

# A fresh Julia process per (backend, variant, tmode, n): the first launch with synchronize at
# top level. Nothing else runs in it, so no earlier compile can hide in the time.
function compile_time_child(
        project::String, backend::Symbol, variant::String, tmode::String, n::Int)
    cmd = `$(joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia")) --startup-file=no --project=$(project) -- $(@__DIR__)/run.jl
           --compile-one $(backend) $(variant) $(tmode) $(n)`
    out = try
        Base.invokelatest(read, cmd, String)
    catch
        return nothing
    end
    m = match(r"^compile_s=([0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?)$"m, out)
    return m === nothing ? nothing : parse(Float64, m.captures[1])
end

# Prepare the child's inputs without launching; time its first launch in separate top-level statements.
function compile_one(backend::Symbol, variant::String, tmode::String, n::Int)
    vendor = load_backend(backend)
    kb = ka_backend(backend, vendor)
    T = T_of(tmode)
    A = make_A(T, n)
    b = [make_b(A, make_xstar(T, n, 1))]
    out, iters, argb = if backend === :cpu
        [zero(SVector{n, T})], zeros(Int32, 1), b
    else
        o = KernelAbstractions.allocate(kb, SVector{n, T}, 1)
        it = KernelAbstractions.allocate(kb, Int32, 1)
        bd = KernelAbstractions.allocate(kb, SVector{n, T}, 1)
        copyto!(o, [zero(SVector{n, T})])
        copyto!(it, zeros(Int32, 1))
        copyto!(bd, b)
        o, it, bd
    end
    solver = SOLVERS[(variant, tmode)]
    key = tmode == "MIXED" ? "MIXED-" * variant : variant
    kernel = KERNELS[key]
    return solver, kernel, kb, A, out, iters, argb
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

function main(io::IO, backend::AbstractString)
    be = Symbol(backend)
    be in (:cpu, :metal, :cuda, :rocm) || error("backend must be cpu, metal, cuda or rocm")
    vendor = be === :cpu ? nothing : load_backend(be)
    println(io, "GeometricSolvers kernel_newton spike -- backend = ", backend)
    println(io, "Julia ", VERSION)
    println(io, "Card: ", card_name(be, vendor))
    println(io, "Packages: ", package_versions(be, vendor))
    println(io, "Compile cap: ", COMPILE_CAP_S, " s; timing runs: ", TIMING_RUNS)
    cache = Cache()
    project = Base.active_project()
    want_variant = length(ARGS) >= 2 ? ARGS[2] : ""
    want_tmode = length(ARGS) >= 3 ? ARGS[3] : ""
    want_n = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : -1
    for (variant, tmode) in rows_of(be), n in NS

        (isempty(want_variant) || variant == want_variant) || continue
        (isempty(want_tmode) || tmode == want_tmode) || continue
        (want_n < 0 || n == want_n) || continue

        empty!(cache.batches)
        compile_s = compile_time_child(project, be, variant, tmode, n)
        row = Base.invokelatest(
            measure_row, be, vendor, cache, variant, tmode, n, compile_s)
        row["julia"] = string(VERSION)
        row["packages"] = package_versions(be, vendor)
        row["card"] = card_name(be, vendor)
        println(io, "ROW ", row_text(row))
        flush(io)
    end
    return nothing
end

main(backend::AbstractString) = main(stdout, backend)

const COMPILE_CHILD = abspath(PROGRAM_FILE) == (@__FILE__) &&
                      !isempty(ARGS) && ARGS[1] == "--compile-one"
if COMPILE_CHILD
    be = Symbol(ARGS[2])
    load_backend(be)
    solver, kernel, kb, A, out, iters, argb = Base.invokelatest(compile_one,
        be, ARGS[3], ARGS[4], parse(Int, ARGS[5]))
end

# The clock and launch are separate top-level statements, so launch compilation is timed.
child_start = COMPILE_CHILD ? time_ns() : UInt64(0)
if COMPILE_CHILD
    if be === :cpu
        cpu_launch(solver, A, argb, out, iters, MAXIT)
    else
        Base.invokelatest(launch, kernel, kb, out, iters, A, argb, MAXIT)
    end
end

if COMPILE_CHILD
    println("compile_s=", (time_ns() - child_start) / 1e9)
elseif abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) >= 1 ? ARGS[1] : "cpu")
end
