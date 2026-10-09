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
