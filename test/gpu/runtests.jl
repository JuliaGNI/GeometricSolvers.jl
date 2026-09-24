# Device tests, run by hand on a machine with the GPU. There is no device CI; `README.md` in this
# directory is the procedure.
#
# Usage, from the repository root:
#   julia --startup-file=no --project=test/gpu/<backend> test/gpu/runtests.jl <backend>
# with <backend> one of `cuda`, `rocm`, `metal`. Each backend has its own environment in
# `test/gpu/<backend>/`, so that no machine installs another vendor's stack. From a REPL whose
# active project is `test/gpu/<backend>`:
#   include("test/gpu/runtests.jl"); main("metal")
#
# The test runs a KernelAbstractions kernel on the device array and checks it against the same
# kernel on the KA `CPU()` backend: in `Float32` and `Float16`, plus `Float64` on CUDA and ROCm
# for correctness (Metal has no `Float64`). The output starts with the device, the driver and the
# package versions, so that a saved output records what it was run on.

using GeometricSolvers: GeometricSolvers
using KernelAbstractions
using Random: Xoshiro
using Test

# backend => (vendor package, its KA backend type, the element types it is tested in)
const BACKENDS = Dict(
    "cuda" => (:CUDA, :CUDABackend, (Float32, Float16, Float64)),
    "rocm" => (:AMDGPU, :ROCBackend, (Float32, Float16, Float64)),
    "metal" => (:Metal, :MetalBackend, (Float32, Float16)))

# The lengths include ones that are not a multiple of the workgroup size.
const LENGTHS = (1, 1000, 4097)
const WORKGROUP = 64

@kernel function axpy_kernel!(y, a, @Const(x))
    i = @index(Global)
    y[i] = muladd(a, x[i], y[i])
end

function axpy!(backend, y, a, x)
    axpy_kernel!(backend, WORKGROUP)(y, a, x; ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    y
end

function to_backend(backend, A)
    B = KernelAbstractions.allocate(backend, eltype(A), size(A))
    copyto!(B, A)
end

function versions(vendor::Module)
    mods = (GeometricSolvers, KernelAbstractions, vendor)
    join(("$(nameof(m)) $(pkgversion(m))" for m in mods), ", ")
end

function run_tests(name::AbstractString, vendor::Module, device, eltypes)
    println("GeometricSolvers device tests -- backend = ", name)
    println("Julia ", VERSION)
    println("Packages: ", versions(vendor))
    println()
    vendor.versioninfo()
    println()
    cpu = KernelAbstractions.CPU()
    @testset "KA kernel on $name" begin
        @testset "axpy, $T, n = $n" for T in eltypes, n in LENGTHS

            rng = Xoshiro(n)
            x = rand(rng, T, n)
            y = rand(rng, T, n)
            a = T(0.75)
            y_cpu = axpy!(cpu, copy(y), a, x)
            y_dev = axpy!(device, to_backend(device, y), a, to_backend(device, x))
            @test KernelAbstractions.get_backend(y_dev) == device
            # A device may fuse the multiply-add where the CPU rounds twice: one ulp apart.
            @test Array(y_dev) ≈ y_cpu rtol=2 * eps(T)
        end
    end
end

function main(name::AbstractString)
    haskey(BACKENDS, name) ||
        error("backend must be one of $(join(sort!(collect(keys(BACKENDS))), ", ")); got $name")
    package, backend_type, eltypes = BACKENDS[name]
    # Loaded here rather than at the top, so that one file serves every vendor's environment.
    # The rest of the run needs the newest world, which has the package's methods.
    Core.eval(Main, Expr(:using, Expr(:., package)))
    vendor = Base.invokelatest(getfield, Main, package)
    device = Base.invokelatest(getfield(vendor, backend_type))
    Base.invokelatest(run_tests, name, vendor, device, eltypes)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: runtests.jl <cuda|rocm|metal>")
    main(ARGS[1])
end
