# The element types and the array backends every test loops over, named once, and the systems of
# a right-hand side at an end of the exponent range, `extreme_rhs_systems`.
#
# A numeric test runs in each element type the package supports, and on each array backend the
# code runs on; writing those sets in each test file lets them drift, and `test/quality/matrix.jl`
# fails on a literal tuple of element types anywhere else under `test/`.
#
# `REAL_ELTYPES` is for a method that takes a real working precision, `ELTYPES` for one that also
# takes a complex element type. A test that needs one element type writes that type; a subset of a
# set is a `filter` of it, never a literal tuple.
#
# The array backends: an ordinary `Array`; `JLArray`, a CPU array that refuses scalar indexing and
# so catches a GPU-only fault in ordinary CI; and the KernelAbstractions `CPU()` backend, for a
# kernel itself rather than for the array it runs on.

using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: Diagonal, I

const REAL_ELTYPES = (Float32, Float64)
const ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)
const ARRAY_BACKENDS = (Array, JLArray)
const KA_BACKEND = KernelAbstractions.CPU()

# The 4 × 4 systems of a right-hand side at an end of the exponent range whose solution is
# representable: `(label, J, b, x, qr, atol)` with the exact solution `x` in `BigFloat`, `qr` whether
# the QR and LU solves are meant to solve it (they do not solve a singular `J`, nor `J_turned`, whose
# back substitution overflows, K23 of `KNOWN_ISSUES.md`), and `atol` the absolute error
# an entry may have beside the relative one. `b` near `floatmax` makes a solve that forms `Qᴴ b` or
# `Uᴴ b` without scaling `b` overflow, and with `J_lu` the forward substitution `b₂ - b₁` of an LU
# solve; with the large `2^e` of `J_k22` it makes an SVD solve of `J / 2^e` overflow in the step
# `2^e x` of the scaled system, and `J_k22` with a small `b` makes one that divides `b` by `2^e`
# lose it; `b` below `floatmin` makes one that scales `b` down lose it.
function extreme_rhs_systems(::Type{T}) where {T <: Real}
    fm = floatmax(T)
    big_rhs = fill(T(0.75) * fm, 4)
    hilbert = [one(BigFloat) / (i + j - 1) for i in 1:4, j in 1:4]
    J_hilbert = Matrix{T}(I, 4, 4) + T.(hilbert)
    J_lu = T[1 0 0 0; 1 4 0 0; 0 0 1 0; 0 0 0 1]
    lu_rhs = T[0.75 * fm, -0.75 * fm, 1, 1]
    D_k22 = Diagonal(big.([2^20, 2^10, 2^20, 2^20]))
    J_k22 = T.(D_k22)
    k22_rhs = T[0, 0.75 * fm, 0, 0]
    # `diag(2^20, 2^14, 2^20, 2^20)` turned by the Hadamard matrix `H₄ / 2`, exact in `T`, so that
    # the singular vectors mix the coordinates: the solution `0.75 floatmax / 2^14 · H₄[:, 2] / 2`;
    # `2^14`, not the `2^10` of `J_k22`, keeps the condition number, and so the error of the SVD
    # itself, below the bound of the tests
    H = big.([1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]) / 2
    J_turned = T.(H * Diagonal(big.([2^20, 2^14, 2^20, 2^20])) * H)
    turned_rhs = T.(big(T(0.75) * fm) * H[:, 2])
    small_rhs = T[0, T(1.2345) * ldexp(floatmin(T), 6), 0, 0]
    J_small = T.(Diagonal(big.([2^20, 1, 2^20, 2^20])))
    big_scale = ldexp(one(T), exponent(fm) - 8)
    tiny = fill(floatmin(T) / 4, 4)
    mixed = T[0.75 * fm, 3 * nextfloat(zero(T)), floatmin(T) / 4, 1]
    # the same-type solve divides the mixed `b` by `2^k`, `k = 3`: `0.75 floatmax` has the exponent
    # `emax - 1`, and the headroom of `N = 4` entries is `2 + 2` bits; the entry `3 nextfloat(0)`
    # then rounds to a multiple of `2^3 nextfloat(0)` on the way, and may be off by that much
    mixed_atol = ldexp(nextfloat(zero(T)), 3)
    return [
        ("I + Hilbert, b = 0.75 floatmax", J_hilbert,
            big_rhs, big.(J_hilbert) \ big.(big_rhs),
            true, zero(T)),
        ("L₂₁ = 1, U₂₂ = 4, b = ±0.75 floatmax", J_lu, lu_rhs, big.(J_lu) \ big.(lu_rhs),
            true, zero(T)),
        ("diag(2^20, 2^10, 2^20, 2^20), b₂ = 0.75 floatmax", J_k22, k22_rhs,
            big.(J_k22) \ big.(k22_rhs), true, zero(T)),
        ("H₄ diag(2^20, 2^14, 2^20, 2^20) H₄ / 4, b = 0.75 floatmax H₄[:, 2] / 2",
            J_turned,
            turned_rhs, big.(J_turned) \ big.(turned_rhs), false, zero(T)),
        ("diag(2^20, 1, 2^20, 2^20), b₂ = 1.2345 · 2^6 floatmin", J_small, small_rhs,
            big.(J_small) \ big.(small_rhs), true, zero(T)),
        ("ones, b = 0.75 floatmax", ones(T, 4, 4),
            big_rhs, fill(big(big_rhs[1]) / 4, 4), false, zero(T)),
        ("floatmax / 2 · ones, b = floatmax / 2", fill(fm / 2, 4, 4), fill(fm / 2, 4),
            fill(big(T(0.25)), 4), false, zero(T)),
        ("2^(emax - 8) I, b = 0.75 floatmax", big_scale * Matrix{T}(I, 4, 4), big_rhs,
            big.(big_rhs) / big(big_scale), true, zero(T)),
        ("I, b = floatmin / 4 (subnormal)",
            Matrix{T}(I, 4, 4), tiny, big.(tiny), true, zero(T)),
        ("I, b = 0.75 floatmax with entries down to 3 nextfloat(0)",
            Matrix{T}(I, 4, 4), mixed,
            big.(mixed), true, mixed_atol)
    ]
end
