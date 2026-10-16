# The R3 Jacobian–vector product: one pushforward, and exactly one evaluation of the residual.
#
# The merit derivative of a line search is `φ′(α) = F(x + αd)ᵀ J(x + αd) d`, and one
# Jacobian–vector product gives it for the cost of one directional derivative, where the whole
# Jacobian costs `n`. The mixed-precision refinement residual is the other consumer.
#
# Through DifferentiationInterface that is its prepared `pushforward!`; through the chunked mode
# one `N = 1` dual pass in its own buffers, which leaves the Jacobian buffers untouched.

function jvp!!(Jv, prep::DIJacobian, prob, x, v, p)
    DI.pushforward!(
        prob.F, prep.y, (Jv,), prep.jvpprep, prep.backend, x, (v,), DI.Constant(p))
    return Jv
end

function jvp!!(Jv, prep::ChunkedForwardDiff, prob, x, v, p)
    prep.xdual1 .= seed.(prep.xdual1, x, tuple.(v))
    prob.F(prep.rdual1, prep.xdual1, p)
    Jv .= dual_partial.(prep.rdual1, 1)
    return Jv
end

# The R1 value methods. The preparation is built from the problem and never from a method's
# operator (§4.3), so `nothing` is the problem's own `J` and an `AutoForwardDiff` one dual
# evaluation of `F` — one partial, one residual evaluation, and no buffer. A `JacobianFunction`
# has no method here: a method's operator is not the derivative of the residual.
#
# The direction has the element type and the length of the iterate, as the line-search slope does
# (§13.L-a): `v` is typed with `x`, so a direction of another element type is a host `MethodError`
# and no `Float64` value reaches a `Float32` kernel.
function jvp!!(::Nothing, ::Nothing, prob, x::StaticVector{N, T}, v::StaticVector{N, T},
        p) where {N, T}
    return prob.J(x, p) * v
end

function jvp!!(::Nothing, ::Nothing, prob, x::T, v::T, p) where {T <: Number}
    return prob.J(x, p) * v
end

function jvp!!(::Nothing, backend::AutoForwardDiff, prob, x::StaticVector{N, T},
        v::StaticVector{N, T}, p) where {N, T}
    D = dual_type(T, typeof(tag_of(backend, prob, x)), 1)
    xdual = SVector{N, D}(ntuple(i -> seed(zero(D), x[i], tuple(v[i])), Val(N))...)
    rdual = prob.F(xdual, p)
    return SVector{N, T}(ntuple(i -> dual_partial(rdual[i], 1), Val(N))...)
end

function jvp!!(::Nothing, backend::AutoForwardDiff, prob, x::T, v::T, p) where {T <: Number}
    D = dual_type(typeof(x), typeof(tag_of(backend, prob, x)), 1)
    return dual_partial(prob.F(seed(zero(D), x, tuple(v)), p), 1)
end
