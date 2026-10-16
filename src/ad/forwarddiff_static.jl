# The R1 Jacobian: tagged ForwardDiff on a vector or `Number` iterate, the problem's own
# `J`, and the operator of a method. The tag and the complex seeding are the R3 chunked mode's —
# `tag_of`, `dual_type`, `seed` and `dual_partial` — so that R1 and R3 cannot drift apart. A back
# end other than `AutoForwardDiff` has no R1 method, and the call is a host `MethodError`.

"""
    JacobianFunction(f)

The Jacobian source of a method that carries an operator of its own: `f(x, p)` is that operator,
and [`jacobian!!`](@ref) calls it in place of the problem's `J`. The method's operator is not the
derivative of the residual, so it serves [`jacobian!!`](@ref) and never [`jvp!!`](@ref).

Internal: wrap the operator in this type to pass it as the source of the R1 `jacobian!!` call.
It holds a function and nothing else, so it is `isbits` exactly when that function is.
"""
struct JacobianFunction{A}
    f::A
end

"""
    prepare_ad(backend::AutoForwardDiff, prob, ::Nothing, x::Union{StaticVector, Number}, p)

The R1 preparation of `AutoForwardDiff`: with no residual buffer and a vector or `Number` iterate
there is nothing to allocate, so the preparation is the back end itself, and [`jacobian!!`](@ref)
and [`jvp!!`](@ref) resolve its tag per call against `prob` and `x`, through the same `tag_of` rule
as the R3 chunked mode. That keeps the R1 state `isbits` and carries a caller's tag unchanged.

A static matrix is not an iterate, so no method here takes one: `prepare_ad` on one is a host
`MethodError`.
"""
function prepare_ad(
        backend::AutoForwardDiff, prob, ::Nothing, x::Union{StaticVector, Number}, p)
    backend
end

function jacobian!!(::Nothing, ::Nothing, prob, x::Union{StaticVector, Number}, p)
    return prob.J(x, p)
end

function jacobian!!(::Nothing, s::JacobianFunction, prob, x, p)
    return s.f(x, p)
end

# One pass of `N` partials, `N = length(x)`: R1 ignores the chunk size, because a static iterate
# has no chunking to gain from and `SVector` arithmetic is unrolled. Column `j` perturbs entry `j`
# by the real one, so the result is the complex derivative where `x` is complex. The matrix is
# built column by column, as the constructor reads it, so that no mutable array is in the way: one
# that a bounds check could force onto the heap would allocate where this must not.
function jacobian!!(::Nothing, backend::AutoForwardDiff, prob, x::StaticVector{N, T},
        p) where {N, T}
    D = dual_type(T, typeof(tag_of(backend, prob, x)), N)
    xdual = SVector{N, D}(ntuple(j -> seed(zero(D), x[j], ntuple(==(j), Val(N))), Val(N)))
    rdual = prob.F(xdual, p)
    return SMatrix{N, N, T}(
        ntuple(k -> dual_partial(rdual[mod1(k, N)], fld1(k, N)), Val(N * N)))
end

# A `Number` iterate is `n = 1`, and its Jacobian is a `Number` in the same element type: the
# Jacobian–vector product in the direction one.
function jacobian!!(::Nothing, backend::AutoForwardDiff, prob, x::Number, p)
    return jvp!!(nothing, backend, prob, x, one(x), p)
end
