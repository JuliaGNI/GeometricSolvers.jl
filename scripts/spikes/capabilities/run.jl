# Capability census for the dense linear-algebra operations GeometricSolvers needs from a vendor
# library or from its own generic kernels.
#
# For each element type -- F16, BF16, F32, F64, CF32, CF64 -- and each operation -- GEMM, lu!,
# qr!, svd!, A \ b, cholesky, batched LU -- a cell is *pass*, *wrong* (the relative residual
# exceeds a stated tolerance) or the first line of the thrown error, tagged with the step that
# threw it: moving the input to the device, the computation, or reading the result back. The
# residual is taken in ComplexF64 against the T-rounded input: the product for GEMM, the
# reconstruction L*U, Q*R, U*S*Vt or U'*U for a factorisation, A*x - b for A \ b. Two extra checks,
# not swept over every element type: a KernelAbstractions kernel doing BFloat16 arithmetic, and
# DifferentiationInterface's `jacobian!` with `AutoForwardDiff()` on the backend's own array type
# and on a `JLArray`.
#
# Usage: julia --startup-file=no --project=. run.jl <backend>
#   backend in {cpu, metal, cuda, rocm}.
#
# This environment has no backend package. `metal`, `cuda` and `rocm` need the vendor environment
# `test/gpu/<backend>` stacked behind it through `JULIA_LOAD_PATH` (see README.md). Without that,
# every cell reports the load error (`Package CUDA not found`), unless the global environment on
# the load path has the vendor package: then it loads from there. Batched LU has no check yet: it
# is *n/a* on `cpu` and `metal`, and *not measured* on `cuda` and `rocm`.
#
# `Float64` and `ComplexF64` on `metal` are marked *unsupported* without a run: a `Float64` scalar
# reaching a Metal kernel raises `InvalidIRError`.
#
# `lu!` and `cholesky` on a `Float16` `MtlArray` crash the Julia process rather than raise a
# catchable exception. This script does not attempt them: it reports *crash (confirmed manually,
# not re-run)* for both, and skips `A \ b` on the same (backend, F16) pair because it dispatches
# to `lu!` internally.
#
# No timings: this is a correctness census, not a benchmark.

using LinearAlgebra
using Random
using Printf
using BFloat16s: BFloat16s, BFloat16
using JLArrays: JLArrays, JLArray
using KernelAbstractions
using ADTypes: ADTypes, AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
import ForwardDiff   # DI's ForwardDiff back end needs this loaded, not only `using ADTypes`.

const N = 8
const SEED = 0x5eed
const ELTYPES = (:F16 => Float16, :BF16 => BFloat16, :F32 => Float32, :F64 => Float64,
    :CF32 => ComplexF32, :CF64 => ComplexF64)
const OPS = ("GEMM", "lu!", "qr!", "svd!", "A\\b", "cholesky", "batched LU")

# Float16 and BFloat16 follow the `max(8, n) * eps` scaling of
# `GeometricIntegratorsBase.default_options`; `svd!` sits near 5 eps at Float32, so the higher
# precisions keep 100 eps. Every comparison is in ComplexF64, against the T-rounded input.
rtol(T, n) = (real(T) in (Float16, BFloat16) ? max(8, n) : 100) * Float64(eps(real(T)))
function relerr(C, Cref)
    norm(ComplexF64.(C) .- ComplexF64.(Cref)) / max(norm(ComplexF64.(Cref)), eps())
end
firstline(e) = first(split(sprint(showerror, e), '\n'))

function judge(C, Cref, T)
    e = relerr(C, Cref)
    tol = rtol(T, N)
    e <= tol ? @sprintf("pass (relerr %.2g, tol %.2g)", e, tol) :
    @sprintf("wrong (relerr %.2g, tol %.2g)", e, tol)
end

# --- backend dispatch -------------------------------------------------------------------------

function to_device(backend::AbstractString, A)
    if backend == "cpu"
        Array(A)
    elseif backend == "metal"
        @eval Main using Metal
        # The closure body runs in the newest world, so it sees the binding `using` just made.
        Base.invokelatest(() -> Main.Metal.MtlArray(A))
    elseif backend == "cuda"
        @eval Main using CUDA
        Base.invokelatest(() -> Main.CUDA.CuArray(A))
    elseif backend == "rocm"
        @eval Main using AMDGPU
        Base.invokelatest(() -> Main.AMDGPU.ROCArray(A))
    else
        error("unknown backend $backend (expected cpu, metal, cuda or rocm)")
    end
end

ka_backend(backend::AbstractString) =
    if backend == "cpu"
        KernelAbstractions.CPU()
    elseif backend == "metal"
        Base.invokelatest(() -> Main.Metal.MetalBackend())
    elseif backend == "cuda"
        Base.invokelatest(() -> Main.CUDA.CUDABackend())
    elseif backend == "rocm"
        Base.invokelatest(() -> Main.AMDGPU.ROCBackend())
    end

function unsupported_eltype(backend::AbstractString, T::Type)
    backend == "metal" && real(T) == Float64
end

# `lu!` and `cholesky` crash the process outright for a Float16 MtlArray (confirmed manually,
# see the header comment above); do not attempt them, or the op that calls `lu!` internally.
function crashes(backend::AbstractString, T::Type, op::AbstractString)
    backend == "metal" && T == Float16 && op in ("lu!", "cholesky", "A\\b")
end

# --- test data ---------------------------------------------------------------------------------

function testdata(T::Type)
    rng = Xoshiro(SEED)
    A0 = randn(rng, N, N) + N * I
    b0 = randn(rng, N)
    if T <: Complex
        R = real(T)
        A = R.(A0) .+ im .* R.(reverse(A0, dims = 1))
        b = R.(b0) .+ im .* R.(reverse(b0))
    else
        A = T.(A0)
        b = T.(b0)
    end
    A, b
end

spd(A) = A * A' + size(A, 1) * one(eltype(A)) * I

# --- one grid cell -------------------------------------------------------------------------

function run_cell(backend::AbstractString, T::Type, op::AbstractString)
    if op == "batched LU"
        return backend in ("cuda", "rocm") ? "not measured (no batched LU check yet)" :
               "n/a (CUDA, ROCm only)"
    end
    unsupported_eltype(backend, T) && return "unsupported (no Float64 on Metal)"
    crashes(backend, T, op) && return "crash (confirmed manually, not re-run)"

    A, b = testdata(T)
    step = "to device"
    try
        Ad = to_device(backend, A)
        Aref = ComplexF64.(A)
        if op == "GEMM"
            step = "compute"
            C = Ad * Ad
            step = "read back"
            judge(Array(C), Aref * Aref, T)
        elseif op == "lu!"
            step = "compute"
            F = lu!(copy(Ad))
            step = "read back"
            L = Array(F.L)
            U = Array(F.U)
            p = Array(F.p)
            judge(ComplexF64.(L) * ComplexF64.(U), Aref[p, :], T)
        elseif op == "qr!"
            step = "compute"
            F = qr!(copy(Ad))
            step = "read back"
            Q = Array(Matrix(F.Q))
            R = Array(Matrix(F.R))
            judge(ComplexF64.(Q) * ComplexF64.(R), Aref, T)
        elseif op == "svd!"
            step = "compute"
            F = svd!(copy(Ad))
            step = "read back"
            U = Array(F.U)
            S = Array(F.S)
            Vt = Array(F.Vt)
            judge(ComplexF64.(U) * Diagonal(ComplexF64.(S)) * ComplexF64.(Vt), Aref, T)
        elseif op == "A\\b"
            bd = to_device(backend, b)
            step = "compute"
            x = Ad \ bd
            step = "read back"
            judge(Aref * ComplexF64.(Array(x)), ComplexF64.(b), T)
        elseif op == "cholesky"
            step = "compute"
            Aspd = spd(Ad)
            F = cholesky(Aspd)
            step = "read back"
            # `Array(F.U)` on a GPU array scalar-indexes through the generic triangular copy
            # path; go through the raw, unmasked factor and triangularise on the host instead.
            Uraw = Array(parent(F.U))
            U = triu(ComplexF64.(Uraw))
            verdict = judge(U' * U, ComplexF64.(Array(spd(A))), T)
            # LinearAlgebra factorises Float16 and BFloat16 in Float32 (`choltype`). A BFloat16
            # input keeps the Float32 factor, flagged here; a Float16 input is converted back to
            # Float16, so its cell cannot show that the arithmetic ran in Float32.
            eltype(F) == T ? verdict : verdict * " [factor eltype $(eltype(F))]"
        end
    catch e
        "error ($step): " * firstline(e)
    end
end

# --- a KA kernel with BFloat16 arithmetic (no rem, fma, atan(y,x), mod2pi or sincos --
# BFloat16s.jl lacks the first four and the fifth recurses to a stack overflow) --------------

@kernel function bf16_axpy_kernel!(c, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds c[i] = a[i] * b[i] + a[i]
end

function check_ka_bfloat16(backend::AbstractString)
    try
        rng = Xoshiro(SEED)
        a = BFloat16.(randn(rng, 100))
        b = BFloat16.(randn(rng, 100))
        ad = to_device(backend, a)
        bd = to_device(backend, b)
        cd = similar(ad)
        kab = ka_backend(backend)
        bf16_axpy_kernel!(kab)(cd, ad, bd; ndrange = length(ad))
        KernelAbstractions.synchronize(kab)
        cref = ComplexF64.(a) .* ComplexF64.(b) .+ ComplexF64.(a)
        judge(Array(cd), cref, BFloat16)
    catch e
        "error: " * firstline(e)
    end
end

# --- DI jacobian! with AutoForwardDiff() on the device array, and on a JLArray ---------

di_f!(y, x) = (y .= x .^ 2 .+ 1; nothing)

function check_di_jacobian(x)
    y = similar(x)
    backend_ad = AutoForwardDiff()
    try
        prep = DI.prepare_jacobian(di_f!, y, backend_ad, x)
        J = similar(x, length(x), length(x))
        DI.jacobian!(di_f!, y, J, prep, backend_ad, x)
        Jref = Diagonal(2 .* ComplexF64.(Array(x)))
        judge(Array(J), Matrix(Jref), eltype(x))
    catch e
        "error: " * firstline(e)
    end
end

function check_di_jacobian(backend::AbstractString)
    try
        check_di_jacobian(to_device(backend, Float32.(1:6)))
    catch e
        "error: " * firstline(e)
    end
end

# --- report --------------------------------------------------------------------------------

const BACKEND_PACKAGE = Dict("metal" => :Metal, "cuda" => :CUDA, "rocm" => :AMDGPU)

# The Manifest is not committed, so the output records the versions that decided each cell.
function package_versions(backend::AbstractString)
    mods = Module[KernelAbstractions, DI, ForwardDiff, ADTypes, BFloat16s, JLArrays]
    name = get(BACKEND_PACKAGE, backend, nothing)
    if name !== nothing && Base.invokelatest(isdefined, Main, name)
        push!(mods, Base.invokelatest(getfield, Main, name))
    end
    join(("$(nameof(m)) $(pkgversion(m))" for m in mods), ", ")
end

# Loading the backend package inside a cell leaves the rest of that call in a world without the
# package's methods. Load it first, so `main` can run the census in the newest world. A package
# that is not in the environment is left to `to_device`, which reports it in every cell.
function load_backend(backend::AbstractString)
    name = get(BACKEND_PACKAGE, backend, nothing)
    if name !== nothing && Base.find_package(string(name)) !== nothing
        Core.eval(Main, Expr(:using, Expr(:., name)))
    end
    nothing
end

function main(io::IO, backend::AbstractString)
    load_backend(backend)
    Base.invokelatest(census, io, backend)
end

function census(io::IO, backend::AbstractString)
    println(io, "GeometricSolvers capability census -- backend = ", backend)
    println(
        io, "Julia ", VERSION, "; tolerance = max(8, n)*eps(T) for F16 and BF16 (n = ", N,
        "), 100*eps(real(T)) otherwise")
    println(io)
    rows = [[run_cell(backend, T, op) for op in OPS] for (_, T) in ELTYPES]
    widths = [max(textwidth(op), maximum(r -> textwidth(r[j]), rows))
              for (j, op) in enumerate(OPS)]
    print(io, rpad("type", 8))
    foreach((op, w) -> print(io, "  ", rpad(op, w)), OPS, widths)
    println(io)
    for ((label, _), row) in zip(ELTYPES, rows)
        print(io, rpad(string(label), 8))
        foreach((cell, w) -> print(io, "  ", rpad(cell, w)), row, widths)
        println(io)
    end
    println(io)
    println(io, "KA kernel, BFloat16 arithmetic:               ", check_ka_bfloat16(backend))
    println(io, "DI jacobian!, AutoForwardDiff, device array:  ", check_di_jacobian(backend))
    println(io, "DI jacobian!, AutoForwardDiff, JLArray:       ",
        check_di_jacobian(JLArray(Float32.(1:6))))
    println(io)
    println(io, "Packages: ", package_versions(backend))
    nothing
end

main(backend::AbstractString) = main(stdout, backend)

if abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) >= 1 ? ARGS[1] : "cpu")
end
