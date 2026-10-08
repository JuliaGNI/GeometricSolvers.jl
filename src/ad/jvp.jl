# The R3 Jacobian–vector product: one pushforward, and exactly one evaluation of the residual.
#
# Two consumers (§1.6, §1.4). The merit derivative of a line search is
# `φ′(α) = F(x + αd)ᵀ J(x + αd) d`: `SimpleSolvers` forms the whole Jacobian for it at every trial
# point, which costs `n` directional derivatives, where one Jacobian–vector product gives the same
# number for the cost of one. The mixed-precision refinement residual is the other.
#
# On an `Array` that is DI's prepared `pushforward!`; on a device array one `N = 1` dual pass
# through the chunked mode's own buffers, which leaves the Jacobian buffers untouched.
#
# Part L adds the value methods, for a `StaticArray` or a `Number` iterate, to this file.

function jvp!!(Jv, prep::DIJacobian, prob, x, v, p)
    DI.pushforward!(
        prob.F, prep.y, (Jv,), prep.jvpprep, prep.backend, x, (v,), context!(prep, p)
    )
    return Jv
end

# The `N = 1` seed: the value of the iterate, with the direction as its one partial. The dual
# buffer's own element carries the tag, so that no type travels into the broadcast.
@inline function dual_seed1(::ForwardDiff.Dual{Tg, V, 1}, xi, vi) where {Tg, V}
    return ForwardDiff.Dual{Tg}(convert(V, xi), convert(V, vi))
end

@inline function dual_seed1(d::Complex{<:ForwardDiff.Dual{Tg, V, 1}}, xi, vi) where {Tg, V}
    return Complex(
        ForwardDiff.Dual{Tg}(convert(V, real(xi)), convert(V, real(vi))),
        ForwardDiff.Dual{Tg}(convert(V, imag(xi)), convert(V, imag(vi)))
    )
end

function jvp!!(Jv, prep::ChunkedForwardDiff, prob, x, v, p)
    xdual, rdual = prep.xdual1, prep.rdual1
    xdual .= dual_seed1.(xdual, x, v)
    prob.F(rdual, xdual, p)
    Jv .= dual_partial.(rdual, 1)
    return Jv
end
