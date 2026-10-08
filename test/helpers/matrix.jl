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
