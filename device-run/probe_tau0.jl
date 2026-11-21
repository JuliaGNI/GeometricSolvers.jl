# Part RO, the RX 7900 XTX run: confirm the trigger of the complex `geqrf!` NaN.
#
# Run it from the root of the clone, on the branch device-run/rocm-rx7900xtx:
#
#     julia --startup-file=no --project=test/gpu/rocm device-run/probe_tau0.jl \
#         > ../results/rocmext-rx7900xtx/probe_tau0.txt 2>&1
#
# `probe_info65.txt` shows that rocSOLVER `geqrf!` gives `|R[65,65]| = NaN` for ComplexF32 and
# ComplexF64 on `4I` from n = 256 (not at n = 128), while real `geqrf!`, every `getrf!` and
# rocBLAS `gemm!` agree with the host. The hypothesis: a column whose part below the diagonal is
# exactly zero gives a reflector with τ = 0, and the blocked complex path, which starts somewhere
# in 129..256, turns that τ = 0 into a NaN when it applies the panel to the trailing columns.
#
# Each line ends in `ok` or `BAD` (BAD = a NaN anywhere, or |diag R| differs from LAPACK by more
# than `sqrt(eps)`). It also prints how many τ are exactly zero, the first NaN of τ, and the first
# NaN of the factor as (row, column).
#
# D  the threshold: `4I` for n in 128..256; the first BAD n is the blocked-path switch.
# E  the cure: `4I` plus 1e-3 just below the diagonal of every column, so no τ is zero but the
#    last one, τ_n, whose column has no part below the diagonal and no trailing column (LAPACK:
#    `τ zero = 1`). The hypothesis predicts ok at every n.
# F  the cause: a dense matrix (`randn + 2n I`, no zero τ) with one decoupled variable j: row j
#    and column j zero but for a real `A[j, j] = 4`, so τ is zero at j only (LAPACK: `τ zero = 1`;
#    in Float64 also τ_n, as in E). The
#    hypothesis predicts ok with no j and at n = 128, BAD for j in panel 1 with the first NaN in
#    column 65, and BAD for j = 100 (panel 2) with the first NaN in column 129.
# G  the same matrices as D, E and F in Float64: the hypothesis predicts ok everywhere.
#
# It takes a few minutes.

using AMDGPU
using LinearAlgebra
using Random: Xoshiro

println("AMDGPU ", pkgversion(AMDGPU), ", Julia ", VERSION, ", device ", AMDGPU.device())
flush(stdout)

errored = Ref(false)

function report(f, label)
    try
        println(label, "  ", f())
    catch err
        errored[] = true
        println(label, "  ERROR ", sprint(showerror, err))
    end
    flush(stdout)
end

# The first index where `x` and `ref` differ by more than `rtol` relative to `ref`, or 0.
function first_bad(x, ref, rtol)
    idx = findfirst(i -> !(abs(x[i] - ref[i]) <= rtol * abs(ref[i])), eachindex(ref))
    return idx === nothing ? 0 : idx
end

firstnan(x) = (i = findfirst(isnan, x); i === nothing ? "none" : Tuple(i))

function check_geqrf(A::Matrix{T}) where {T}
    n = size(A, 1)
    dA = ROCArray(A)
    dτ = ROCArray(zeros(T, n))
    AMDGPU.rocSOLVER.geqrf!(dA, dτ)
    F, τ = Array(dA), Array(dτ)
    d_dev = abs.(F[diagind(F)])
    d_host = abs.(diag(qr(A).R))
    j = first_bad(d_dev, d_host, sqrt(eps(real(T))))
    nan = any(isnan, F) || any(isnan, τ)
    "first bad |diag(R)| = $j  τ zero = $(count(iszero, τ))  " *
    "first NaN τ = $(firstnan(τ))  first NaN factor = $(firstnan(F))  " *
    (j > 0 || nan ? "BAD" : "ok")
end

diag4(T, n) = Matrix{T}(Diagonal(fill(T(4), n)))

function subdiag_filled(T, n)
    A = diag4(T, n)
    for j in 1:(n - 1)
        A[j + 1, j] = T(1e-3)
    end
    return A
end

# Row j and column j are zero but for a real `A[j, j] = 4`: no earlier reflector touches row j, so
# the reflector of column j has x = 0 and a real α, which gives τ = 0 there and only there.
function dense_decoupled(T, n, j)
    A = randn(Xoshiro(0x5eed + n), T, n, n) + T(2n) * I
    if j > 0
        A[j, :] .= zero(T)
        A[:, j] .= zero(T)
        A[j, j] = T(4)
    end
    return A
end

const NS_D = (128, 129, 144, 160, 176, 192, 193, 208, 224, 240, 255, 256)
const NS_EF = (128, 256, 512)
const COLS = (0, 1, 40, 64, 100)

for T in (ComplexF32, ComplexF64, Float64)
    tag = rpad(T, 10)
    T === Float64 && println("\n# G  the same matrices in Float64")
    T === Float64 || println("\n# D  threshold, 4I, $T")
    for n in NS_D
        report(() -> check_geqrf(diag4(T, n)), "D 4I          $tag n = $(lpad(n, 4))")
    end
    T === Float64 || println("\n# E  4I with a filled subdiagonal, $T")
    for n in NS_EF
        report(() -> check_geqrf(subdiag_filled(T, n)), "E 4I+subdiag  $tag n = $(lpad(n, 4))")
    end
    T === Float64 || println("\n# F  dense with one decoupled variable j (0 = none), $T")
    for n in NS_EF, j in COLS

        j < n || continue
        report(() -> check_geqrf(dense_decoupled(T, n, j)),
            "F dense j = $(lpad(j, 3)) $tag n = $(lpad(n, 4))")
    end
end

println("\ndone; errored = ", errored[])
exit(errored[] ? 1 : 0)
