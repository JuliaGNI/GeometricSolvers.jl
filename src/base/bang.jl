# The two functions of the AD seam, declared here with no method of their own.
#
# `src/ad/` adds the R3 methods, through DifferentiationInterface and through the own chunked
# forward mode. The declaration lives here rather than in either file, so that no file is the one
# that has to be included first.
#
# The `!!` of the name is the package's convention for a function that writes into its first
# argument where the storage allows it and returns a value where it does not: the caller uses the
# returned value in both cases.

"""
    jacobian!!(J, prep, prob, x, p)

Write the Jacobian of the residual of `prob` at the iterate `x`, for the parameters `p`, into `J`,
and return `J`.

`prep` is the AD preparation that [`prepare_ad`](@ref) built, and the only argument that selects
the code: a [`DIJacobian`](@ref) runs DifferentiationInterface's prepared `jacobian!`, a
[`ChunkedForwardDiff`](@ref) the own chunked forward mode. The back end is therefore chosen once,
where the preparation is built, and never per call.
"""
function jacobian!! end

"""
    jvp!!(Jv, prep, prob, x, v, p)

Write the Jacobian–vector product `J(x, p) * v` of the residual of `prob` into `Jv`, and return
`Jv`.

One pushforward, and exactly one evaluation of the residual: that is what makes the merit
derivative `φ′(α) = F(x + αd)ᵀ J(x + αd) d` of a line search cost one directional derivative
rather than `n`, and it is what the mixed-precision refinement residual uses.

`prep` selects the code exactly as in [`jacobian!!`](@ref).
"""
function jvp!! end
