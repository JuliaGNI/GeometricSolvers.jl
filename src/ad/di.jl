# The R3 AD path on the CPU: DifferentiationInterface (§3.1). DI is a front end — one API over
# every ADTypes back end — so the package codes against it and gains `AutoEnzyme()`,
# `AutoFiniteDiff()` and, in part V, `AutoSparse(...)` without a line of its own (§3.2).
#
# The path is selected once, in `prepare_ad`, and `jacobian!!` and `jvp!!` dispatch on the object
# it returns. The solver therefore takes no `ad` keyword and makes no runtime choice.

"""
    prepare_ad(backend, prob, r, x, p)

Build the AD preparation for the residual of `prob` at an iterate like `x`, with a residual buffer
like `r` and a representative parameter set `p`, and return the object that [`jacobian!!`](@ref)
and [`jvp!!`](@ref) dispatch on. `init` calls this once per solver; the values of `p`, and the
parameter object itself, may be replaced on every solve.

An `Array` iterate takes DifferentiationInterface with `backend`, and gives a
[`DIJacobian`](@ref). A device array iterate with `AutoForwardDiff()` takes the own chunked
forward mode, and gives a [`ChunkedForwardDiff`](@ref): DI's `jacobian!` raises "Scalar indexing
is disallowed" on an `MtlArray`, a `CuArray` and a `JLArray`, and passes on an `Array` only
(§3.1). Any other back end on a device array has no path, and says so here rather than failing
inside DI at the first solve.
"""
function prepare_ad end

# The tag and the chunk size are fixed here, once, and are the same for the Jacobian and for the
# JVP. A back end that carries no tag gets this solver's own: the alternative is `Dual{Nothing}`,
# which confuses the perturbations of a residual that differentiates, and so breaks nested
# differentiation silently.
tag_of(backend::AutoForwardDiff, prob, x) = backend.tag
function tag_of(::AutoForwardDiff{C, Nothing}, prob, x) where {C}
    return ForwardDiff.Tag(prob.F, eltype(x))
end

# The caller's chunk size where it is positive, `ForwardDiff.pickchunksize(n)` otherwise, clamped
# to `n`: a chunk wider than the iterate would seed columns that do not exist.
function chunk_size(::AutoForwardDiff{C}, n::Integer) where {C}
    N = (C isa Integer && C > 0) ? C : ForwardDiff.pickchunksize(n)
    return max(1, min(N, n))
end

# The back end the preparation is built with. `AutoForwardDiff` gets the fixed tag and chunk size;
# any other back end passes through unchanged, because its options are the caller's alone.
prepared_backend(backend::AbstractADType, prob, x) = backend

function prepared_backend(backend::AutoForwardDiff, prob, x)
    return AutoForwardDiff(;
        chunksize = chunk_size(backend, length(x)), tag = tag_of(backend, prob, x)
    )
end

"""
    DIJacobian

The AD preparation of the R3 CPU path: DifferentiationInterface's prepared `jacobian!` and
`pushforward!` for one back end, a residual buffer, and the `Constant` wrapper of the current
parameters. [`prepare_ad`](@ref) builds it, and it lives in the solver state.

The parameters are a context and not a differentiated argument, so they are wrapped in DI's
`Constant`. The wrapper is rebuilt exactly when the caller passes a different parameter object, so
a solve that keeps its parameters allocates nothing and a caller that replaces its parameter array
needs no new solver.
"""
mutable struct DIJacobian{B <: AbstractADType, J, V, Y <: AbstractArray, P}
    const backend::B
    const jacprep::J
    const jvpprep::V
    const y::Y
    context::DI.Constant{P}
end

function prepare_ad(backend::AbstractADType, prob, r::AbstractArray, x::AbstractArray, p)
    b = prepared_backend(backend, prob, x)
    context = DI.Constant(p)
    jacprep = DI.prepare_jacobian(prob.F, similar(r), b, x, context)
    # the tangent is a prototype, not a value: zeroed, so that preparation cannot evaluate the
    # residual on whatever happened to be in a fresh buffer
    jvpprep = DI.prepare_pushforward(prob.F, similar(r), b, x, (zero(x),), context)
    return DIJacobian(b, jacprep, jvpprep, similar(r), context)
end

function prepare_ad(::AbstractADType, prob, ::AbstractGPUArray, ::AbstractGPUArray, p)
    throw(
        ArgumentError(
        "a device array iterate has an R3 Jacobian through AutoForwardDiff() only: " *
        "DifferentiationInterface raises a scalar-indexing error on an MtlArray, a " *
        "CuArray and a JLArray. Solve on an Array to use another back end."
    )
    )
end

# The `Constant` wrapper of the current parameters, rebuilt only where the caller passes a
# different object. `===` and not `==`: a parameter array whose values changed in place is the
# same context, so its wrapper need not be touched. The guard saves no allocation — a
# `DI.Constant{P}` is one pointer and is stored inline in the field, so rebuilding it per call
# allocates nothing either (measured: the allocation assertions of `test/ad/jacobian.jl` hold with
# the guard removed) — it keeps the stored context and the argument one object rather than two.
@inline function context!(prep::DIJacobian, p)
    prep.context.data === p || (prep.context = DI.Constant(p))
    return prep.context
end

function jacobian!!(J, prep::DIJacobian, prob, x, p)
    DI.jacobian!(prob.F, prep.y, J, prep.jacprep, prep.backend, x, context!(prep, p))
    return J
end
