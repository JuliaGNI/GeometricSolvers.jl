# NextLA.jl census: do NextLA's KernelAbstractions kernels for GEMM, the triangular solves and
# the tile QR give correct results on a backend?
#
# For each element type -- F32, F64, CF32 -- and each operation, a cell is *pass*, *wrong* (the
# relative residual exceeds a stated tolerance) or the first line of the thrown error, tagged with
# the step that threw it: moving the input to the device, the computation, or reading the result
# back. The residual is taken in ComplexF64 on the host, against the T-rounded input:
#
#   GEMM        GEMM_ADD!(A, B, C), C := C + A*B               ‖C - C0 - A*B‖ / (‖A‖‖B‖ + ‖C0‖)
#   TRSM LL/LU  LeftLowerTRSM!, LeftUpperTRSM!,  A*X = B        ‖A*X - B‖ / (‖A‖‖X‖ + ‖B‖)
#   TRSM RL/RU  RightLowerTRSM!, RightUpperTRSM!, X*A = B       ‖X*A - B‖ / (‖A‖‖X‖ + ‖B‖)
#   rec TRSM    unified_rectrxm!('L', 'L', 'N', 1, 'S', A, B)   as TRSM LL, at n = N_REC
#   rec TRMM    unified_rectrxm!('L', 'U', 'N', 1, 'M', A, B)   ‖B - A*B0‖ / (‖A‖‖B0‖)
#   geqrt!      geqrt!(m, n, ib, A, T, tau, work), Q built on the host from the reflectors in A
#               and tau                                         max(‖A0 - Q*R‖/‖A0‖, ‖Q'Q - I‖)
#   unmqr!      unmqr!('L', 'N', A, T, I) on the geqrt! output  ‖A0 - Q*R‖ / ‖A0‖
#
# The operands are sized so that no dimension is a multiple of the 32-wide GEMM tile. The
# recursive TRSM runs at n = N_REC > 256, the size above which `unified_rectrxm!` recurses for a
# solve instead of calling the base kernel; the recursive TRMM recurses above 16, so N suffices.
# `geqrt!` runs on a tall matrix with a block size that does not divide it. NextLA 0.2.3 has no
# driver that chains the tile kernels (`tsqrt!`, `tsmqr!`, `ttqrt!`) into a full QR, so they have
# no row of their own.
#
# Usage: julia --startup-file=no --project=. run.jl <backend>
#   backend in {cpu, metal, cuda}.
#
# This environment has no backend package. `metal` and `cuda` need the vendor environment
# `test/gpu/<backend>` stacked behind it through `JULIA_LOAD_PATH` (see README.md); without that,
# every cell of that backend reports an `UndefVarError` for the vendor module, at the `to device`
# step.
#
# `Float64` on `metal` is marked *unsupported* without a run: a `Float64` scalar reaching a
# Metal kernel raises `InvalidIRError`.
#
# No timings: this is a correctness census, not a benchmark.

using LinearAlgebra
using Random
using Printf
using KernelAbstractions
using NextLA: NextLA

const N = 40        # square operands; not a multiple of the 32-wide GEMM tile
const N_REC = 300   # the recursive TRSM recurses only above n = 256
const NRHS = 5      # right-hand sides of a triangular solve
const QR_M, QR_N, QR_IB = 45, 37, 8
const SEED = 0x5eed
const ELTYPES = (:F32 => Float32, :F64 => Float64, :CF32 => ComplexF32)
const OPS = ("GEMM", "TRSM LL", "TRSM LU", "TRSM RL", "TRSM RU", "rec TRSM", "rec TRMM",
    "geqrt!", "unmqr!")

# The residuals are normwise backward errors, which grow at most linearly in n for these
# operations; 100 eps covers the small operands, n eps the recursive solve at n = N_REC.
rtol(T, n) = max(100, n) * Float64(eps(real(T)))
c64(A) = ComplexF64.(Array(A))
firstline(e) = first(split(sprint(showerror, e), '\n'))

function judge(e, T, n)
    tol = rtol(T, n)
    e <= tol ? @sprintf("pass (relerr %.2g, tol %.2g)", e, tol) :
    @sprintf("wrong (relerr %.2g, tol %.2g)", e, tol)
end

# --- backend dispatch -------------------------------------------------------------------------

function to_device(backend::AbstractString, A)
    if backend == "cpu"
        Array(A)
    elseif backend == "metal"
        Base.invokelatest(() -> Main.Metal.MtlArray(A))
    elseif backend == "cuda"
        Base.invokelatest(() -> Main.CUDA.CuArray(A))
    else
        error("unknown backend $backend (expected cpu, metal or cuda)")
    end
end

function unsupported_eltype(backend::AbstractString, T::Type)
    backend == "metal" && real(T) == Float64
end

# --- test data ---------------------------------------------------------------------------------

function randmat(rng, T::Type, m, n)
    T <: Complex ? T.(randn(rng, m, n), randn(rng, m, n)) : T.(randn(rng, m, n))
end

# Triangular with a dominant diagonal, so that the solve is well conditioned and a wrong result
# cannot hide behind a large condition number.
function triangular(rng, T::Type, n, uplo::Symbol)
    A = randmat(rng, T, n, n) + n * I
    uplo == :L ? tril(A) : triu(A)
end

# --- the operations ------------------------------------------------------------------------

function check_gemm(backend, T)
    rng = Xoshiro(SEED)
    A, B, C0 = randmat(rng, T, N, N - 3), randmat(rng, T, N - 3, N + 5),
    randmat(rng, T, N, N + 5)
    step = "to device"
    try
        Ad, Bd, Cd = to_device(backend, A), to_device(backend, B), to_device(backend, C0)
        step = "compute"
        NextLA.GEMM_ADD!(Ad, Bd, Cd)
        KernelAbstractions.synchronize(get_backend(Cd))
        step = "read back"
        C = c64(Cd)
        e = norm(C - c64(C0) - c64(A) * c64(B)) /
            (norm(c64(A)) * norm(c64(B)) + norm(c64(C0)))
        judge(e, T, N)
    catch err
        "error ($step): " * firstline(err)
    end
end

const TRSM = Dict("TRSM LL" => (NextLA.LeftLowerTRSM!, :L, :left),
    "TRSM LU" => (NextLA.LeftUpperTRSM!, :U, :left),
    "TRSM RL" => (NextLA.RightLowerTRSM!, :L, :right),
    "TRSM RU" => (NextLA.RightUpperTRSM!, :U, :right))

function trsm_residual(A, X, B, side)
    R = side == :left ? c64(A) * c64(X) - c64(B) : c64(X) * c64(A) - c64(B)
    norm(R) / (norm(c64(A)) * norm(c64(X)) + norm(c64(B)))
end

function check_trsm(backend, T, op)
    f, uplo, side = TRSM[op]
    rng = Xoshiro(SEED)
    A = triangular(rng, T, N, uplo)
    B = side == :left ? randmat(rng, T, N, NRHS) : randmat(rng, T, NRHS, N)
    step = "to device"
    try
        Ad, Xd = to_device(backend, A), to_device(backend, B)
        step = "compute"
        f(Ad, Xd)
        KernelAbstractions.synchronize(get_backend(Xd))
        step = "read back"
        judge(trsm_residual(A, Xd, B, side), T, N)
    catch err
        "error ($step): " * firstline(err)
    end
end

function check_rectrsm(backend, T)
    rng = Xoshiro(SEED)
    A = triangular(rng, T, N_REC, :L)
    B = randmat(rng, T, N_REC, NRHS)
    step = "to device"
    try
        Ad, Xd = to_device(backend, A), to_device(backend, B)
        step = "compute"
        NextLA.unified_rectrxm!('L', 'L', 'N', one(T), 'S', Ad, Xd)
        KernelAbstractions.synchronize(get_backend(Xd))
        step = "read back"
        judge(trsm_residual(A, Xd, B, :left), T, N_REC)
    catch err
        "error ($step): " * firstline(err)
    end
end

function check_rectrmm(backend, T)
    rng = Xoshiro(SEED)
    A = triangular(rng, T, N, :U)
    B0 = randmat(rng, T, N, NRHS)
    step = "to device"
    try
        Ad, Bd = to_device(backend, A), to_device(backend, B0)
        step = "compute"
        NextLA.unified_rectrxm!('L', 'U', 'N', one(T), 'M', Ad, Bd)
        KernelAbstractions.synchronize(get_backend(Bd))
        step = "read back"
        e = norm(c64(Bd) - c64(A) * c64(B0)) / (norm(c64(A)) * norm(c64(B0)))
        judge(e, T, N)
    catch err
        "error ($step): " * firstline(err)
    end
end

# Q = H_1 H_2 ... H_k with H_i = I - tau_i v_i v_i', v_i = [0; 1; A[i+1:m, i]], in ComplexF64.
function householder_q(Aqr, tau)
    m, k = size(Aqr, 1), length(tau)
    Q = Matrix{ComplexF64}(I, m, m)
    for i in 1:k
        v = zeros(ComplexF64, m)
        v[i] = 1
        v[(i + 1):m] .= Aqr[(i + 1):m, i]
        Q -= (Q * v) * (tau[i] * v')
    end
    Q
end

# One geqrt! run shared by the two QR cells: `nothing` on success, else the error cell.
function run_geqrt(backend, T)
    rng = Xoshiro(SEED)
    m, n, ib = QR_M, QR_N, QR_IB
    k = min(m, n)
    A0 = randmat(rng, T, m, n)
    step = "to device"
    try
        Ad = to_device(backend, A0)
        Td = to_device(backend, zeros(T, ib, k))
        taud = to_device(backend, zeros(T, k))
        workd = to_device(backend, zeros(T, ib * n))
        step = "compute"
        NextLA.geqrt!(m, n, ib, Ad, Td, taud, workd)
        KernelAbstractions.synchronize(get_backend(Ad))
        (A0, Ad, Td, taud), nothing
    catch err
        nothing, "error ($step): " * firstline(err)
    end
end

function check_geqrt(backend, T, run)
    run[2] === nothing || return run[2]
    A0, Ad, _, taud = run[1]
    try
        Aqr, tau = c64(Ad), vec(c64(taud))
        k = length(tau)
        R = triu(Aqr[1:k, :])
        Q = householder_q(Aqr, tau)
        e = max(norm(c64(A0) - Q[:, 1:k] * R) / norm(c64(A0)), norm(Q' * Q - I))
        judge(e, T, QR_M)
    catch err
        "error (read back): " * firstline(err)
    end
end

function check_unmqr(backend, T, run)
    run[2] === nothing || return run[2] * " [in geqrt!]"
    A0, Ad, Td, _ = run[1]
    m, n = size(A0)
    k = min(m, n)
    step = "to device"
    try
        Qd = to_device(backend, Matrix{T}(I, m, m))
        step = "compute"
        NextLA.unmqr!('L', 'N', Ad, Td, Qd)
        KernelAbstractions.synchronize(get_backend(Qd))
        step = "read back"
        R = triu(c64(Ad)[1:k, :])
        e = norm(c64(A0) - c64(Qd)[:, 1:k] * R) / norm(c64(A0))
        judge(e, T, QR_M)
    catch err
        "error ($step): " * firstline(err)
    end
end

# --- one row ---------------------------------------------------------------------------------

function run_row(backend::AbstractString, T::Type)
    unsupported_eltype(backend, T) &&
        return fill("unsupported (no Float64 on Metal)", length(OPS))
    qr = run_geqrt(backend, T)
    map(OPS) do op
        op == "GEMM" ? check_gemm(backend, T) :
        startswith(op, "TRSM") ? check_trsm(backend, T, op) :
        op == "rec TRSM" ? check_rectrsm(backend, T) :
        op == "rec TRMM" ? check_rectrmm(backend, T) :
        op == "geqrt!" ? check_geqrt(backend, T, qr) : check_unmqr(backend, T, qr)
    end
end

# --- report --------------------------------------------------------------------------------

const BACKEND_PACKAGE = Dict("metal" => :Metal, "cuda" => :CUDA)

# The Manifest is not committed, so the output records the versions that decided each cell.
function package_versions(backend::AbstractString)
    mods = Module[NextLA, KernelAbstractions]
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
    println(io, "GeometricSolvers NextLA census -- backend = ", backend)
    println(io, "Julia ", VERSION, "; n = ", N, " (rec TRSM n = ", N_REC, "), ", NRHS,
        " right-hand sides, geqrt! ", QR_M, "x", QR_N, " with ib = ", QR_IB,
        "; tolerance = max(100, n)*eps(real(T))")
    println(io)
    rows = [run_row(backend, T) for (_, T) in ELTYPES]
    println(io, "| type | ", join(OPS, " | "), " |")
    println(io, "|:--|", join((":--" for _ in OPS), "|"), "|")
    for ((label, _), row) in zip(ELTYPES, rows)
        println(io, "| ", label, " | ", join(row, " | "), " |")
    end
    println(io)
    println(io, "Packages: ", package_versions(backend))
    nothing
end

main(backend::AbstractString) = main(stdout, backend)

if abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) >= 1 ? ARGS[1] : "cpu")
end
