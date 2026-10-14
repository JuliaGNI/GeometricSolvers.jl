# The seven columns of §10.E check 3 for one row, and the row's output.

const NS_ANSWER = 64              # systems in the answer batch

# The (problem, factorisation) element types of a tmode. F32 and F64 solve in their own type;
# MIXED solves a Float64 problem with a Float32 Jacobian and LU (decision 15).
T_of(tmode::String) = tmode == "F32" ? Float32 : Float64

# The solver of a (variant, tmode) pair, for the rows that `rows_of` gives.
const SOLVERS = Dict(
    ("M", "F32") => newton_M, ("S", "F32") => newton_S, ("SNS", "F32") => sns_newton,
    ("M", "F64") => newton_M, ("S", "F64") => newton_S,
    ("M", "MIXED") => newton_mixed_M, ("S", "MIXED") => newton_mixed_S)

# A batch of systems: per system an x* and a b built on the host in T. The seeds are per system,
# so a smaller batch is a prefix of a larger one.
struct Batch{T, n}
    xs::Vector{SVector{n, T}}
    b::Vector{SVector{n, T}}
end

function make_batch(A::SMatrix{n, n, T}, count::Int) where {n, T}
    xs = [make_xstar(T, n, s) for s in 1:count]
    return Batch(xs, [make_b(A, x) for x in xs])
end

# The right-hand sides of a batch prefix on the backend, and a zeroed output pair.
function to_device(kb, batch::Batch{T, n}, count::Int) where {T, n}
    b = KernelAbstractions.allocate(kb, SVector{n, T}, (count,))
    copyto!(b, first(batch.b, count))
    return b
end
to_device(::KernelAbstractions.CPU, batch::Batch, count::Int) = view(batch.b, 1:count)

function out_pair(kb, ::Batch{T, n}, count::Int) where {T, n}
    return KernelAbstractions.zeros(kb, SVector{n, T}, count),
    KernelAbstractions.zeros(kb, Int32, count)
end

# --- a launch, timed from the launch to the synchronize (§10.E check 4) ------------------------
#
# On a device, the KA kernel. On the CPU, a threaded host loop over the same R1 code: one system
# per host thread (§1.1, R2 on the host).

function launch(kb, solve, out, iters, A, b, maxit)
    newton_kernel!(kb, 64)(out, iters, solve, A, b, maxit; ndrange = length(b))
    KernelAbstractions.synchronize(kb)
    return nothing
end

function launch(::KernelAbstractions.CPU, solve, out, iters, A, b, maxit)
    Threads.@threads for i in eachindex(b)
        x, it, info = solve(A, b[i], maxit)
        out[i] = x
        iters[i] = status(it, info)
    end
    return nothing
end

# The wall times of TIMING_RUNS calls of f.
timings(f) = [@elapsed(f()) for _ in 1:TIMING_RUNS]

# --- the allocates cell ------------------------------------------------------------------------
#
# On a device: a call to the GC allocator in the kernel's device LLVM (§10.E). On the CPU:
# @allocated through a function barrier. The cell reads "no" only for a count of exactly zero,
# and it names its method.

const ALLOC_CALL = r"\b(?:call|invoke)\b[^\n]*@(?:\"[^\"]*(?:jl_gc_(?:pool|big|alloc)|gc_alloc|malloc|pool_alloc)[^\"]*\"|[^\s(]*(?:jl_gc_(?:pool|big|alloc)|gc_alloc|malloc|pool_alloc)[^\s(]*)\s*\("

@noinline probe_alloc(f::F, A, b) where {F} = @allocated f(A, b, MAXIT)

# --- the answer cell ---------------------------------------------------------------------------

# The maximum over the answer batch of ‖x − x*‖∞ and of ‖x − x_cpu‖∞, against the bound
# ‖x − x*‖∞ ≤ 16 n eps(T) ‖x*‖∞.
function answer_errors(xout, batch::Batch{T, n}, xsol) where {T, n}
    err = zero(T)
    err_cpu = zero(T)
    for i in eachindex(xout)
        err = max(err, maxabs(xout[i] - batch.xs[i]))
        err_cpu = max(err_cpu, maxabs(xout[i] - xsol[i]))
    end
    return err, err_cpu
end

# --- the row --------------------------------------------------------------------------------------

function new_row(backend::String, variant::String, tmode::String, n::Int)
    Dict{String, String}(
        "backend" => backend,
        "variant" => variant,
        "tmode" => tmode,
        "n" => string(n)
    )
end

const ROW_COLUMNS = ("compiles", "answer", "allocates", "compile_s",
    "solves_1e4", "solves_1e5", "solves_1e6")

# A row that does not compile carries the same mark in every column that depends on the launch,
# so no cell is empty (§10.E check 3).
function mark_columns!(row::Dict{String, String}, text::String, columns = ROW_COLUMNS)
    for c in columns
        row[c] = text
    end
    return row
end
