# The storage functions of §4.1: the complete seam between an algorithm and its storage, and
# nothing else. The value methods of the R1 storage stand here. An algorithm calls them and
# rebinds the result, so the same algorithm runs on a value iterate and on a buffer iterate, and
# the difference stays at the leaves.
#
# The `!!` of the name is the package's convention for a function that writes into its first
# argument where the storage allows it and returns a value where it does not: the caller uses the
# returned value in both cases.
#
# `jacobian!!` and `jvp!!` are declared here with no method of their own, so that no file is the
# one that has to be included first. `src/ad/` adds the R1 and R3 methods, and the docstrings
# below cover both storages.

"""
    buffer(x)
    buffer(x, TF, dims...)

One storage slot for an iterate like `x`, in the element type `TF` and with the shape `dims`.
`init` calls it, once per slot; no algorithm does.

A value iterate — a `StaticArray` or a `Number` — has no slot: the call returns `nothing`, so the
R1 state carries no buffer and spends no register on one. A static matrix is never an iterate and
gets no slot either, which is why this method is written on `StaticArray` and not on
`StaticVector`.
"""
buffer(x::Union{StaticArray, Number}) = nothing
buffer(x::Union{StaticArray, Number}, ::Type{TF}, dims...) where {TF} = nothing

"""
    residual!!(y, prob, x, p)

The residual `F(x, p)` of `prob` at the iterate `x`, for the parameters `p`, in the storage of
`y`, and return it.

`y` is the storage slot the residual is to occupy, and it alone selects the call form of the
problem's `F`: with no buffer slot (`nothing`) the call is `F(x, p)`, for an `SVector`, an
`MVector`, a `Number` and an `Array` alike, because the iterate's type is read nowhere. There is
no `inplace` flag that could disagree with the iterate, and no `hasmethod`.
"""
residual!!(::Nothing, prob, x, p) = prob.F(x, p)

"""
    update_iterate!!(y, x, α, d)

The updated iterate `x + α d`, in the storage of `y`, and return it.

This is the one function that moves the iterate, and the one seam a manifold solver extends: a
retraction is a second method of it, and no algorithm changes.
"""
update_iterate!!(::Nothing, x, α, d) = x + α * d

"""
    map!!(f, y, xs...)

`map(f, xs...)` in the storage of `y`, and return it: the elementwise updates that are not an
iterate update of [`update_iterate!!`](@ref) — a negation, a difference, a copy.
"""
map!!(f, ::Nothing, xs...) = map(f, xs...)

"""
    mul!!(C, A, B)

The matrix product `A * B`, in the storage of `C`, and return it: the Gram matrix of the Anderson
and weighting histories, and a matrix–vector product where no [`jvp!!`](@ref) applies.
"""
mul!!(::Nothing, A, B) = A * B

# The `Number` iterate of §4.1: its Jacobian is a `Number`, and its factor is the `1 × 1` factor
# of part I, in the element type that `factor_eltype` gives. `linear/factors.jl` declares these
# two functions and names their factor types; it is included after this file. So these two methods
# state the storage — `nothing`, and a `Number` — and let the factor type dispatch inside, which
# keeps the factor types in the one file that owns them. A method type with no `1 × 1` factor,
# such as a `TF32` marker, then raises the same host `MethodError` it raises for a matrix.
factorize!!(::Nothing, method, J::Number) = factorize!!(nothing, method, SMatrix{1, 1}(J))
ldiv!!(::Nothing, fac, r::Number) = ldiv!!(nothing, fac, SVector{1}(r))[1]

"""
    jacobian!!(J, prep, prob, x, p)

The Jacobian of the residual of `prob` at the iterate `x`, for the parameters `p`, in the storage
of `J`, and return it.

`prep` is the Jacobian source as `init` prepared it, and the only argument that selects the code.
In R1 the output slot `J` is `nothing` and the call returns a new `SMatrix` in the element type
`T` of `x` — a `Number` for a `Number` iterate — which [`factorize!!`](@ref) converts to the
factorisation precision `TF`: `nothing` calls the problem's `J`, an `AutoForwardDiff` runs tagged
ForwardDiff over a vector iterate, and a [`JacobianFunction`](@ref) calls the method's own
function. In R3 `J` is an `n × n` buffer in `TF` and `prep` is a [`DIJacobian`](@ref), whose
prepared `jacobian!` runs DifferentiationInterface, or a [`ChunkedForwardDiff`](@ref), the own
chunked forward mode. A preparation selects the code once, where it is built, and never per call.
"""
function jacobian!! end

"""
    jvp!!(Jv, prep, prob, x, v, p)

The Jacobian–vector product `J(x, p) * v` of the residual of `prob`, in the storage of `Jv`, and
return it.

One pushforward, and exactly one evaluation of the residual: that is what makes the merit
derivative `φ′(α) = F(x + αd)ᵀ J(x + αd) d` of a line search cost one directional derivative
rather than `n`, and it is what the mixed-precision refinement residual uses.

`prep` is the JVP preparation that `init` built from the problem, never from a method's operator,
and it selects the code exactly as in [`jacobian!!`](@ref). In R1 the output slot `Jv` is
`nothing` and the call returns a new value: `nothing` gives `J(x, p) * v` from the problem's `J`,
and an `AutoForwardDiff` one dual evaluation of `F`. In R3 `Jv` is a buffer and `prep` is a
[`DIJacobian`](@ref) or a [`ChunkedForwardDiff`](@ref).
"""
function jvp!! end
