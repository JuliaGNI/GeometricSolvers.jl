# Capability census for the R3 operations GeometricSolvers needs from a vendor library or its
# own generic kernel.
#
# For each element type -- F16, BF16, F32, F64, CF32, CF64 -- and each operation -- GEMM, lu!,
# qr!, svd!, A \ b, cholesky, batched LU (CUDA/ROCm only) -- a cell is *pass*, *wrong* (the
# relative residual exceeds a stated tolerance) or the first line of the thrown error. The
# residual is taken in ComplexF64 against the T-rounded input: the product for GEMM, the
# reconstruction L*U, Q*R, U*S*Vt or U'*U for a factorisation, A*x - b for A \ b. Two extra checks,
# not swept over every element type: a KernelAbstractions kernel doing BFloat16 arithmetic
# (P0.6), and DifferentiationInterface's `jacobian!` with `AutoForwardDiff()` on the backend's
# own array type and on a `JLArray` (P0.3).
#
# Usage: julia --startup-file=no --project=. run.jl <backend>
#   backend in {cpu, metal, cuda, rocm}.
#
# Only `cpu` and `metal` run on this machine. `cuda` and `rocm` are accepted so a runner with that
# hardware can call the same script unchanged; here they report the load error (`Package CUDA not
# found`) for every cell, because CUDA.jl and AMDGPU.jl are not dependencies of this spike
# environment (AMDGPU.jl does not support macOS at all, and this machine has no CUDA device to
# exercise).
#
# `Float64` and `ComplexF64` on `metal` are marked *unsupported* without a run: a `Float64` scalar
# reaching a Metal kernel raises `InvalidIRError`.
#
# `lu!` and `cholesky` on a `Float16` `MtlArray` crash the Julia process outright (not a catchable
# exception -- confirmed twice, in separate Kaimon sessions). This script does not attempt them:
# it reports *crash (confirmed manually, not re-run)* for both, and skips `A \ b` on the same
# (backend, F16) pair because it dispatches to `lu!` internally.
#
# No timings: this is a correctness census, not a benchmark.

using LinearAlgebra
using Random
using Printf
using BFloat16s: BFloat16
using JLArrays: JLArray
using KernelAbstractions
using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
import ForwardDiff   # DI's ForwardDiff back end needs this loaded, not only `using ADTypes`.

const N = 8
const SEED = 0x5eed
const ELTYPES = (:F16 => Float16, :BF16 => BFloat16, :F32 => Float32, :F64 => Float64,
    :CF32 => ComplexF32, :CF64 => ComplexF64)
const OPS = ("GEMM", "lu!", "qr!", "svd!", "A\\b", "cholesky", "batched LU")

# A coarse "does this basically work" bound, not a precision or cost study.
# Every comparison is in ComplexF64, against a reference built from the T-rounded input.
rtol(T) = 100 * Float64(eps(T <: Complex ? real(T) : T))
function relerr(C, Cref)
    norm(ComplexF64.(C) .- ComplexF64.(Cref)) / max(norm(ComplexF64.(Cref)), eps())
end
firstline(e) = first(split(sprint(showerror, e), '\n'))

function judge(C, Cref, T)
    e = relerr(C, Cref)
    e <= rtol(T) ? @sprintf("pass (relerr %.2g, tol %.2g)", e, rtol(T)) :
    @sprintf("wrong (relerr %.2g, tol %.2g)", e, rtol(T))
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
    op == "batched LU" && return "n/a (CUDA, ROCm only)"
    unsupported_eltype(backend, T) && return "unsupported (no Float64 on Metal)"
    crashes(backend, T, op) && return "crash (confirmed manually, not re-run)"

    A, b = testdata(T)
    try
        Ad = to_device(backend, A)
        Aref = ComplexF64.(A)
        if op == "GEMM"
            C = Ad * Ad
            judge(Array(C), Aref * Aref, T)
        elseif op == "lu!"
            F = lu!(copy(Ad))
            L = Array(F.L)
            U = Array(F.U)
            p = Array(F.p)
            judge(ComplexF64.(L) * ComplexF64.(U), Aref[p, :], T)
        elseif op == "qr!"
            F = qr!(copy(Ad))
            Q = Array(Matrix(F.Q))
            R = Array(Matrix(F.R))
            judge(ComplexF64.(Q) * ComplexF64.(R), Aref, T)
        elseif op == "svd!"
            F = svd!(copy(Ad))
            U = Array(F.U)
            S = Array(F.S)
            Vt = Array(F.Vt)
            judge(ComplexF64.(U) * Diagonal(ComplexF64.(S)) * ComplexF64.(Vt), Aref, T)
        elseif op == "A\\b"
            bd = to_device(backend, b)
            x = Ad \ bd
            judge(Aref * ComplexF64.(Array(x)), ComplexF64.(b), T)
        elseif op == "cholesky"
            Aspd = spd(Ad)
            F = cholesky(Aspd)
            # `Array(F.U)` on a GPU array scalar-indexes through the generic triangular copy
            # path; go through the raw, unmasked factor and triangularise on the host instead.
            Uraw = Array(parent(F.U))
            U = triu(ComplexF64.(Uraw))
            judge(U' * U, ComplexF64.(Array(spd(A))), T)
        end
    catch e
        "error: " * firstline(e)
    end
end

# --- P0.6: a KA kernel with BFloat16 arithmetic (no rem, fma, atan(y,x), mod2pi or sincos --
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

# --- P0.3: DI jacobian! with AutoForwardDiff() on the device array, and on a JLArray ---------

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

function main(io::IO, backend::AbstractString)
    println(io, "GeometricSolvers capability census -- backend = ", backend)
    println(io, "Julia ", VERSION, "; tolerance = 100*eps(real(T)) per element type")
    println(io)
    @printf(io, "%-8s", "type")
    foreach(op -> @printf(io, "  %-34s", op), OPS)
    println(io)
    for (label, T) in ELTYPES
        @printf(io, "%-8s", label)
        for op in OPS
            @printf(io, "  %-34s", run_cell(backend, T, op))
        end
        println(io)
    end
    println(io)
    println(io, "KA kernel, BFloat16 arithmetic (P0.6): ", check_ka_bfloat16(backend))
    println(io, "DI jacobian!, AutoForwardDiff (P0.3), device array: ",
        check_di_jacobian(backend))
    println(io, "DI jacobian!, AutoForwardDiff (P0.3), JLArray:      ",
        check_di_jacobian(JLArray(Float32.(1:6))))
    nothing
end

main(backend::AbstractString) = main(stdout, backend)

if abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) >= 1 ? ARGS[1] : "cpu")
end
