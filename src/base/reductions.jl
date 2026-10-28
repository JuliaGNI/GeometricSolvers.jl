# The one reduction seam: every norm, dot product and merit value of the package goes through
# these three functions, and each returns `real(T)`. A widened or compensated accumulator, if
# one is ever needed, is a change here and nowhere else. They use no scalar indexing, so they
# run on an `Array`, on a GPU array and on an `SVector` inside a kernel.

"""
    norm2(a)

The squared Euclidean norm ``\\sum_i |a_i|^2`` in `real(eltype(a))`.
"""
function norm2(a)
    R = real(eltype(a))
    return isempty(a) ? zero(R) : R(sum(abs2, a))
end

"""
    rnorm(a)

The Euclidean norm ``\\sqrt{\\sum_i |a_i|^2}`` in `real(eltype(a))`.
"""
rnorm(a) = sqrt(norm2(a))

"""
    rdot(a, b)

The real part of the inner product, ``\\mathrm{Re} \\sum_i \\bar{a}_i b_i``, in
`real(promote_type(eltype(a), eltype(b)))`. That is the dot product of the real vector space
under a complex `T`, which a merit derivative needs. Arguments with different axes raise a
`DimensionMismatch`.
"""
function rdot(a, b)
    # The broadcast below would otherwise extend a length-1 or a row argument without an error.
    axes(a) == axes(b) || throw(DimensionMismatch("rdot arguments have different axes"))
    R = real(promote_type(eltype(a), eltype(b)))
    isempty(a) && return zero(R)
    # A lazy broadcast: `mapreduce` over two `Array`s materialises `map(f, a, b)` first.
    products = Broadcast.instantiate(Broadcast.broadcasted((x, y) -> real(conj(x) * y), a, b))
    return R(sum(products))
end
