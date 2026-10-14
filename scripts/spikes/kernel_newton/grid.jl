# The grid of §10.E, shared by run.jl, which writes it, and verify.jl, which checks it.

const NS = (2, 4, 6, 8, 12, 16, 24)  # the system sizes
const NSYS = (10^4, 10^5, 10^6)      # the three throughput cells
const TIMING_RUNS = 5                # runs per timing cell
const COMPILE_CAP_S = 60.0           # the compile cap of §10.E check 5

# The (variant, tmode) rows of a backend. Metal has no Float64 (§1.4). CUDA and ROCm add the
# all-F64 rows, and CUDA the mixed F64/F32 rows (§10.E). SNS is a Float32 variant.
function rows_of(backend::AbstractString)
    rows = [("M", "F32"), ("S", "F32"), ("SNS", "F32")]
    backend in ("cuda", "rocm") && append!(rows, [("M", "F64"), ("S", "F64")])
    backend == "cuda" && append!(rows, [("M", "MIXED"), ("S", "MIXED")])
    return rows
end

# A device compile failure caused by a heap allocation. The InvalidIRError header names no
# allocator; its Reason lines do: the GC allocator, the runtime's `Memory` allocator
# jl_alloc_genericmemory, or an object that a device without a GC cannot allocate.
const ALLOC_ERROR = r"jl_gc|gc_alloc|malloc|pool_alloc|jl_alloc_genericmemory|unsupported allocation"
