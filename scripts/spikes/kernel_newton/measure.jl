# The seven columns of §10.E check 3 for one row, and the row's output.

const NS_ANSWER = 64              # systems in the answer batch
const NSYS = (10^4, 10^5, 10^6)   # the three throughput cells
const TIMING_RUNS = 5             # runs per timing cell, from a cold process
const COMPILE_CAP_S = 60.0        # the compile cap of §10.E check 5

# The (problem, factorisation) element types of a tmode. F32 and F64 solve in their own type;
# MIXED solves a Float64 problem with a Float32 Jacobian and LU (decision 15).
T_of(tmode::String) = tmode == "F32" ? Float32 : Float64

# The solver of a (variant, tmode) pair. SNS exists in F32 only; the mixed and all-F64 rows carry
# the two storage variants, as §10.E decides at the edges.
const SOLVERS = Dict{Tuple{String, String}, Function}()
for v in ("M", "S", "SNS")
    SOLVERS[(v, "F32")] = v == "M" ? newton_M : v == "S" ? newton_S : sns_newton
end
for v in ("M", "S")
    SOLVERS[(v, "F64")] = v == "M" ? newton_M : newton_S
    SOLVERS[(v, "MIXED")] = v == "M" ? newton_mixed_M : newton_mixed_S
end

# A batch of systems: one A, and per system an x* and a b built on the host in T.
struct Batch{T, n, AT <: SMatrix{n, n, T}}
    A::AT
    xs::Vector{SVector{n, T}}
    b::Vector{SVector{n, T}}
end

function make_batch(tmode::String, n::Int, count::Int)
    T = T_of(tmode)
    A = make_A(T, n)
    xs = [make_xstar(T, n, s) for s in 1:count]
    b = [make_b(A, x) for x in xs]
    Batch(A, xs, b)
end

# The device copy of a batch prefix, and a fresh output pair on the device.
function to_device(kb, batch::Batch{T, n}, count::Int) where {T, n}
    b = KernelAbstractions.allocate(kb, SVector{n, T}, count)
    copyto!(b, view(batch.b, 1:count))
    return b
end

# --- a launch, timed from the launch to the synchronize (§10.E check 4) ------------------------

function launch(kernel, kb, out, iters, A, b, maxit)
    kernel(kb, 64)(out, iters, A, b, maxit; ndrange = length(b))
    KernelAbstractions.synchronize(kb)
    return nothing
end

# --- the allocates cell ------------------------------------------------------------------------
#
# On a device: a call to the GC allocator in the kernel's device LLVM (§10.E). On the CPU:
# @allocated through a function barrier. The cell reads "no" only for a count of exactly zero,
# and it names its method.

const ALLOC_CALL = r"\b(?:call|invoke)\b[^\n]*@(?:\"[^\"]*(?:jl_gc_(?:pool|big|alloc)|gc_alloc|malloc|pool_alloc)[^\"]*\"|[^\s(]*(?:jl_gc_(?:pool|big|alloc)|gc_alloc|malloc|pool_alloc)[^\s(]*)\s*\("

function device_alloc_count(ir::String)
    return count(ALLOC_CALL, ir)
end

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
