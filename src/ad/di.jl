# The R3 AD path for every back end other than ForwardDiff: DifferentiationInterface, on an
# `Array`. DI is one API over every ADTypes back end, so `AutoEnzyme()`, `AutoFiniteDiff()` and the
# others need no line of their own here. `AutoForwardDiff` takes the own chunked mode of
# `src/ad/chunked.jl` on every array type.
#
# The path is selected once, in `prepare_ad`, and `jacobian!!` and `jvp!!` dispatch on the object
# it returns. The solver therefore takes no `ad` keyword and makes no runtime choice.

"""
    prepare_ad(backend, prob, r, x, p)

Build the AD preparation for the residual of `prob` at an iterate like `x`, with a residual buffer
like `r` and a representative parameter set `p`, and return the object that [`jacobian!!`](@ref)
and [`jvp!!`](@ref) dispatch on. The values of `p`, and the parameter object itself, may be
replaced on every call.

`AutoForwardDiff()` gives a [`ChunkedForwardDiff`](@ref) on every array type. Any other back end
gives a [`DIJacobian`](@ref) on a CPU iterate, and raises `ArgumentError` on a device array
iterate: DifferentiationInterface's `jacobian!` raises "Scalar indexing is disallowed" on an
`MtlArray`, a `CuArray` and a `JLArray`. The iterate must be a vector.
"""
function prepare_ad end

# A chunk is a range of Jacobian columns, one per entry of the iterate, so a matrix-shaped unknown
# is refused with its shape rather than with a `DimensionMismatch` out of a broadcast.
function check_iterate_shape(x::AbstractArray)
    x isa AbstractVector || throw(
        ArgumentError(
        "the R3 iterate is a vector; got an array of size $(size(x)). Reshape the unknown " *
        "and its residual into vectors, for example with `vec`, before preparing the Jacobian."
    )
    )
    return x
end

"""
    DIJacobian

The AD preparation of the R3 path through DifferentiationInterface: the back end, DI's prepared
`jacobian!` and `pushforward!`, and a residual buffer. [`prepare_ad`](@ref) builds it. The
parameters are a context and not a differentiated argument, so each call wraps them in DI's
`Constant`.
"""
struct DIJacobian{B <: AbstractADType, J, V, Y <: AbstractArray}
    backend::B
    jacprep::J
    jvpprep::V
    y::Y
end

function prepare_ad(backend::AbstractADType, prob, r::AbstractArray, x::AbstractArray, p)
    x isa AbstractGPUArray && throw(
        ArgumentError(
        "a device array iterate has an R3 Jacobian through AutoForwardDiff() only: " *
        "DifferentiationInterface raises a scalar-indexing error on an MtlArray, a " *
        "CuArray and a JLArray. Solve on an Array to use another back end."
    )
    )
    check_iterate_shape(x)
    context = DI.Constant(p)
    jacprep = DI.prepare_jacobian(prob.F, similar(r), backend, x, context)
    # the tangent is zeroed, so that preparation does not evaluate the residual on whatever is in
    # a fresh buffer
    jvpprep = DI.prepare_pushforward(prob.F, similar(r), backend, x, (zero(x),), context)
    return DIJacobian(backend, jacprep, jvpprep, similar(r))
end

function jacobian!!(J, prep::DIJacobian, prob, x, p)
    # an empty iterate has no column, so there is nothing to write; DI with `AutoFiniteDiff()`
    # raises "range must be non-empty" there
    isempty(x) && return J
    DI.jacobian!(prob.F, prep.y, J, prep.jacprep, prep.backend, x, DI.Constant(p))
    return J
end
