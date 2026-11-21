# Part RO, the RX 7900 XTX run: why the complex agreement row at n = 10⁴ gives `info = 65`.
#
# Run it from the root of the clone, on the branch device-run/rocm-rx7900xtx:
#
#     julia --startup-file=no --project=test/gpu/rocm device-run/probe_info65.jl \
#         > ../results/rocmext-rx7900xtx/probe_info65.txt 2>&1
#
# The base run of `test/gpu/rocm/runtests.jl` fails `fac_ext.info == 0` with `65 == 0` for
# ComplexF32 and ComplexF64 at n = 10⁴, and passes at n = 16 and 256 and for the real types. Its
# log does not say whether the LU or the QR row fails. The matrix above n = 256 is `4I` with two
# random 16 × 16 blocks, so column 65 holds only the diagonal 4: a zero there comes from the
# factorisation, not from the data. Column 65 is the first column after a 64-wide panel.
#
# Each part prints one line per case, ending in `ok` or `BAD`:
#
# A  the extension path of the failing row: `factorize!!` with LUFactorization and
#    QRFactorization on the row's matrix, for each element type. Names the failing method.
# B  rocSOLVER `getrf!` and `geqrf!` called directly, on the row's matrix and on `4I` alone,
#    over n, each against LAPACK on the host: `info`, and the first diagonal index of U (LU) or
#    of |R| (QR) whose relative difference exceeds `sqrt(eps)`. Finds the smallest failing n
#    and whether the random blocks matter.
# C  rocBLAS `gemm!` with `beta = 1` into `C = 4I`, k = 64, the shape of a trailing update
#    after the first panel: with zero operands C must stay `4I`; with random operands it must
#    match the host `gemm`. A BAD line here points at rocBLAS rather than rocSOLVER.
#
# The host references at n = 10⁴ take a few minutes; the whole probe takes about 30 minutes.

using AMDGPU
using LinearAlgebra
using Random: Xoshiro

using GeometricSolvers
using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization, SUCCESS,
                        dense_factor, factorize!!, returncode

include("../test/helpers/own_kernel.jl")

BLAS.set_num_threads(Sys.CPU_THREADS)
println("AMDGPU ", pkgversion(AMDGPU), ", Julia ", VERSION, ", device ", AMDGPU.device())
println("functional = ", AMDGPU.functional())
flush(stdout)

const ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)
errored = Ref(false)

function report(f, label)
    try
        line = f()
        println(label, "  ", line)
    catch err
        errored[] = true
        println(label, "  ERROR ", sprint(showerror, err))
    end
    flush(stdout)
end

verdict(bad) = bad ? "BAD" : "ok"

# The first index where `x` and `ref` differ by more than `rtol` relative to `ref`, or 0.
function first_bad(x, ref, rtol)
    idx = findfirst(i -> !(abs(x[i] - ref[i]) <= rtol * abs(ref[i])), eachindex(ref))
    return idx === nothing ? 0 : idx
end

# A: the extension path of the failing row.
println("\n# A  extension path, the agreement row's matrix")
rocm_noop(r, y, p) = nothing
for n in (256, 10^4), T in ELTYPES

    A_cpu, _ = cuda_well_conditioned(T, n)
    dA = ROCArray(A_cpu)
    prob = NonlinearProblem(ROCArray(zeros(T, n)), rocm_noop)
    for M in (LUFactorization, QRFactorization)
        report("A $(rpad(M, 16)) $(rpad(T, 10)) n = $n") do
            m = M()
            fac = dense_factor(m, prob, ROCArray(zeros(T, n)), nothing)
            fac = factorize!!(fac, m, dA)
            rc = returncode(fac)
            "info = $(fac.info)  returncode = $rc  " *
            verdict(fac.info != 0 || rc !== SUCCESS)
        end
    end
end

# B: rocSOLVER called directly, against LAPACK.
diag4(T, n) = Matrix{T}(Diagonal(fill(T(4), n)))
matrices = (("row", (T, n) -> first(cuda_well_conditioned(T, n))), ("4I", diag4))
println("\n# B  rocSOLVER getrf! and geqrf! against LAPACK")
for (name, make) in matrices, T in ELTYPES,
    n in (64, 65, 128, 256, 257, 320, 512, 1024, 2048, 4096, 10^4)
    A = make(T, n)
    rtol = sqrt(eps(real(T)))
    report("B getrf! $(rpad(name, 3)) $(rpad(T, 10)) n = $(lpad(n, 5))") do
        dA = ROCArray(A)
        _, dipiv, info = AMDGPU.rocSOLVER.getrf!(dA, ROCArray(zeros(Cint, n)))
        F = lu(A; check = false)
        d_dev = Array(view(dA, diagind(dA)))
        j = first_bad(d_dev, diag(F.U), rtol)
        # Both are LAPACK's 1-based row interchanges.
        same_piv = Int.(Array(dipiv)) == F.ipiv
        "info = $info  first bad diag(U) = $j" *
        (j > 0 ? " (dev $(d_dev[j]), host $(F.U[j, j]))" : "") *
        "  ipiv same = $same_piv  " * verdict(info != 0 || j > 0)
    end
    report("B geqrf! $(rpad(name, 3)) $(rpad(T, 10)) n = $(lpad(n, 5))") do
        dA = ROCArray(A)
        AMDGPU.rocSOLVER.geqrf!(dA, ROCArray(zeros(T, n)))
        R_host = qr(A).R
        d_dev = abs.(Array(view(dA, diagind(dA))))
        d_host = abs.(diag(R_host))
        j = first_bad(d_dev, d_host, rtol)
        "first bad |diag(R)| = $j" *
        (j > 0 ? " (dev $(d_dev[j]), host $(d_host[j]))" : "") *
        "  " * verdict(j > 0)
    end
end

# C: rocBLAS gemm! with beta = 1, the trailing update after a 64-wide panel.
println("\n# C  rocBLAS gemm!, C = 4I, k = 64, alpha = -1, beta = 1")
for T in ELTYPES, m in (64, 256, 1024, 4096, 10^4 - 64)

    k = 64
    report("C zero   $(rpad(T, 10)) m = $(lpad(m, 5))") do
        dC = ROCArray(diag4(T, m))
        AMDGPU.rocBLAS.gemm!('N', 'N', -one(T), ROCArray(zeros(T, m, k)),
            ROCArray(zeros(T, k, m)), one(T), dC)
        err = maximum(abs, Array(dC) - diag4(T, m))
        "max |C - 4I| = $err  " * verdict(err != 0)
    end
    report("C random $(rpad(T, 10)) m = $(lpad(m, 5))") do
        rng = Xoshiro(0x5eed + m)
        A = randn(rng, T, m, k) .* real(T)(0.05)
        B = randn(rng, T, k, m) .* real(T)(0.05)
        dC = ROCArray(diag4(T, m))
        AMDGPU.rocBLAS.gemm!('N', 'N', -one(T), ROCArray(A), ROCArray(B), one(T), dC)
        C_host = diag4(T, m) - A * B
        err = maximum(abs, Array(dC) - C_host) / maximum(abs, C_host)
        bound = 8 * k * eps(real(T))
        "relative error = $err  bound = $bound  " * verdict(!(err <= bound))
    end
end

println("\ndone; errored = ", errored[])
exit(errored[] ? 1 : 0)
