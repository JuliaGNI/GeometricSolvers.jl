# The problem stubs the R3 AD tests differentiate.
#
# Part S adds the AD seam only: `prepare_ad`, `jacobian!!` and `jvp!!`. `NonlinearProblem` is part
# L, so the tests build the one thing the AD methods read from a problem — the field `F`, the
# in-place residual `F(r, x, p)` — and nothing else. When part L lands, these stubs stay: they keep
# the AD tests independent of the solver interface.
#
# Every residual here is a broadcast with no reduction, so the arithmetic per element is the same
# on every array backend and a comparison against `ForwardDiff.jacobian` on an `Array` is exact
# rather than approximate (§13.S, "Verify it").

using ForwardDiff: ForwardDiff

"A problem stub: the in-place residual `F(r, x, p)` and nothing else."
struct StubProblem{F}
    F::F
end

"""
    CallerTagged()

A stand-in for a caller who differentiates this solver from outside. `ForwardDiff.Tag(CallerTagged(),
T)` is a tag this solver would never build for itself, because its own tag names the residual, so a
preparation that carries it can only have taken it from the caller.
"""
struct CallerTagged end

"""
    Coupled(perm)

`r_i = x_i² + p_i x_i + 2 x_{perm_i}`, so the Jacobian is a diagonal plus the permutation matrix
of `perm`. A diagonal Jacobian cannot see a transposed or shifted column, which is what the
chunked mode can get wrong, so the coupling is what makes the comparison a test.
"""
struct Coupled{P}
    perm::P
end

(f::Coupled)(r, x, p) = (r .= x .* x .+ p .* x .+ 2 .* x[f.perm]; r)

"""
    Scaled()

`r_i = p_i x_i`, so the Jacobian is `Diagonal(p)`: the parameters appear in the result, and a
parameter that is differentiated by mistake shows up at once. It is a single fused broadcast, so
it allocates nothing, which is what the allocation check needs.
"""
struct Scaled end

(::Scaled)(r, x, p) = (r .= p .* x; r)

"The scalar term of [`Nested`](@ref): `x ⋅ d/ds(s² p)|_{s = x}`, which is `2 p x²`."
nested_term(xi, pi) = xi * ForwardDiff.derivative(s -> s * s * pi, xi)

"""
    Nested()

`r_i = 2 p_i x_i²`, written so that the residual itself calls `ForwardDiff.derivative`. The
Jacobian is `Diagonal(4 p_i x_i)`. Differentiating this is nested differentiation: with a shared
or missing tag the inner and the outer perturbation are confused, which is the fault this catches.
"""
struct Nested end

(::Nested)(r, x, p) = (r .= nested_term.(x, p); r)

"""
    Holomorphic()

`r_i = x_i² + p_i x_i`, holomorphic in `x`, so that in a complex element type the Jacobian is the
complex derivative `Diagonal(2 x_i + p_i)` and not the real Jacobian of the two components. The
chunked mode is the only path that differentiates it: ForwardDiff, and so
DifferentiationInterface, has no complex mode, which is why the complex tests build a
`ChunkedForwardDiff` directly rather than through `prepare_ad`.
"""
struct Holomorphic end

(::Holomorphic)(r, x, p) = (r .= x .* x .+ p .* x; r)

"""
    Counting(F)

`F` with a count of its calls, in `calls[]`. `jvp!!` is one pushforward and exactly one residual
evaluation, which is a statement about this counter.
"""
struct Counting{F}
    F::F
    calls::Base.RefValue{Int}
end

Counting(F) = Counting(F, Ref(0))

(c::Counting)(r, x, p) = (c.calls[] += 1; c.F(r, x, p))

"""
    ad_inputs(AT, T, n)

The iterate, the parameters and a residual buffer for the array type `AT` and the element type
`T`, with no zero and no equal entries, so that a wrong column cannot look right by accident.
"""
function ad_inputs(AT, ::Type{T}, n::Int) where {T}
    x = AT(T[(i + 1) / (n + 2) for i in 1:n])
    p = AT(T[(2i + 3) / (n + 5) for i in 1:n])
    return x, p, similar(x)
end

"A complex iterate and complex parameters: no real entry, so a path that drops the imaginary
component cannot look right by accident."
function ad_inputs(AT, ::Type{Complex{T}}, n::Int) where {T}
    x = AT(Complex{T}[(i + 1) / (n + 2) + im * (i + 3) / (n + 4) for i in 1:n])
    p = AT(Complex{T}[(2i + 3) / (n + 5) - im * (i + 2) / (n + 6) for i in 1:n])
    return x, p, similar(x)
end

"A cyclic shift by one, as the permutation of [`Coupled`](@ref): non-symmetric, so a transposed
Jacobian fails the comparison."
cyclic_perm(AT, n::Int) = AT(circshift(collect(1:n), -1))

"""
    forwarddiff_jacobian(F, x, p)

The reference: `ForwardDiff.jacobian` of the same residual on an `Array`, which is what the *Done
when* of part S compares against. `x` and `p` may live on any backend; the reference is taken on
the host copies, so `F` is the host copy of the residual — a [`Coupled`](@ref) on a device array
carries a device permutation, which cannot index a host array.
"""
function forwarddiff_jacobian(F, x, p)
    xh, ph = Array(x), Array(p)
    return ForwardDiff.jacobian((r, z) -> F(r, z, ph), similar(xh), xh)
end
