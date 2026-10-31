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

Where `norm2(a)` is finite and at least `length(a) * floatmin(R)`, it is `sqrt(norm2(a))`, one
reduction. Otherwise it scales the entries by ``2^{-e}``, where ``2^e`` is the power of two of the
largest ``|a_i|`` (of the largest real or imaginary part for a complex `a`), so that it neither
overflows where the norm is finite nor underflows to zero where the norm is positive. That path
takes two more reductions, the largest entry and the sum, and sums in at least `Float32`. Scaling
can round the squares of entries much smaller than the largest to zero, which changes the result
in the last bits at most. A `NaN` entry gives `NaN`, an `Inf` entry and no `NaN` gives
`Inf`. Where `real(eltype(a))` is no `AbstractFloat`, such as an integer or a
`ForwardDiff.Dual`, it is the unscaled `sqrt(norm2(a))`.
"""
function rnorm(a)
    R = real(eltype(a))
    R <: AbstractFloat || return sqrt(norm2(a))
    # The unscaled sum is accurate to the last bits when it is finite and at least
    # `length(a) * floatmin`: a square below `floatmin` loses at most `floatmin * eps / 2`, and an
    # entry has at most two such squares, so together they lose at most `eps` of the sum, relatively.
    s = norm2(a)
    isfinite(s) && s >= length(a) * floatmin(R) && return sqrt(s)
    isempty(a) && return zero(R)
    m = R(maximum(x -> max(abs(real(x)), abs(imag(x))), a))
    (iszero(m) || !isfinite(m)) && return m
    e = exponent(m)
    # Every scaled square is below 8, below 4 for a real entry, so a `Float16` sum of them can
    # overflow from 8200 entries on: it accumulates in `Float32`.
    S = promote_type(R, Float32)
    return R(ldexp2(sqrt(sum(x -> S(abs2(ldexp2(x, -e))), a)), e))
end

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
