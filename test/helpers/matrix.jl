# The element types and the array backends every test loops over, named once.
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

const REAL_ELTYPES = (Float32, Float64)
const ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)
const ARRAY_BACKENDS = (Array, JLArray)
const KA_BACKEND = KernelAbstractions.CPU()

using LinearAlgebra: I

# The 4 × 4 systems of a right-hand side at an end of the exponent range whose solution is
# representable: `(label, J, b, x, qr)` with the exact solution `x` in `BigFloat`, and `qr` whether
# the QR solve is meant to solve it (it does not solve a singular `J`). `b` near `floatmax` makes a
# solve that forms `Qᴴ b` or `Uᴴ b` without scaling `b` overflow; `b` below `floatmin` makes one
# that scales `b` down lose it.
function extreme_rhs_systems(::Type{T}) where {T <: Real}
    fm = floatmax(T)
    big_rhs = fill(T(0.75) * fm, 4)
    hilbert = [one(BigFloat) / (i + j - 1) for i in 1:4, j in 1:4]
    J_hilbert = Matrix{T}(I, 4, 4) + T.(hilbert)
    big_scale = ldexp(one(T), exponent(fm) - 8)
    tiny = fill(floatmin(T) / 4, 4)
    return [
        ("I + Hilbert, b = 0.75 floatmax", J_hilbert,
            big_rhs, big.(J_hilbert) \ big.(big_rhs),
            true),
        ("ones, b = 0.75 floatmax", ones(T, 4, 4),
            big_rhs, fill(big(big_rhs[1]) / 4, 4), false),
        ("floatmax / 2 · ones, b = floatmax / 2", fill(fm / 2, 4, 4), fill(fm / 2, 4),
            fill(big(T(0.25)), 4), false),
        ("2^(emax - 8) I, b = 0.75 floatmax", big_scale * Matrix{T}(I, 4, 4), big_rhs,
            big.(big_rhs) / big(big_scale), true),
        ("I, b = floatmin / 4 (subnormal)", Matrix{T}(I, 4, 4), tiny, big.(tiny), true)
    ]
end
