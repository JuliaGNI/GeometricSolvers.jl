# The storage functions: the complete seam between an algorithm and its storage, and
# nothing else. The value methods of the R1 storage and the array methods of the R3 storage stand
# here. An algorithm calls them and rebinds the result, so the same algorithm runs on a value
# iterate and on a buffer iterate, and the difference stays at the leaves.
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
A caller creates slots during initialization; an algorithm consumes their returned values.

A value iterate — a `StaticArray` or a `Number` — has no slot: the call returns `nothing`, so the
R1 state carries no buffer and spends no register on one. A static matrix is never an iterate and
gets no slot either, which is why this method is written on `StaticArray` and not on
`StaticVector`. An array iterate allocates a storage slot via `similar`.
"""
buffer(x::StaticArray) = nothing
buffer(x::Number) = nothing
buffer(x::StaticArray, ::Type{TF}, dims...) where {TF} = nothing
buffer(x::Number, ::Type{TF}, dims...) where {TF} = nothing
buffer(x::AbstractArray) = similar(x)
buffer(x::AbstractArray, ::Type{TF}, dims...) where {TF} = similar(x, TF, dims...)

"""
    residual!!(y, prob, x, p)

The residual `F(x, p)` of `prob` at the iterate `x`, for the parameters `p`, in the storage of
`y`, and return it.

`y` is the storage slot the residual is to occupy, and it alone selects the call form of the
problem's `F`: with no buffer slot (`nothing`) the call is `F(x, p)`, for an `SVector`, an
`MVector`, a `Number` and an `Array` alike, because the iterate's type is read nowhere; with an
array buffer slot `y` the call is `F(y, x, p)`. There is no `inplace` flag that could disagree
with the iterate, and no `hasmethod`.
"""
residual!!(::Nothing, prob, x, p) = prob.F(x, p)
residual!!(y::AbstractArray, prob, x, p) = (prob.F(y, x, p); y)

"""
    update_iterate!!(y, x, α, d)

The updated iterate `x + α d`, in the storage of `y`, and return it.

This is the one function that moves the iterate, and the one seam a manifold solver extends: a
retraction is a second method of it, and no algorithm changes.
"""
update_iterate!!(::Nothing, x, α, d) = x + α * d
update_iterate!!(y::AbstractArray, x, α, d) = (y .= x .+ α .* d; y)

"""
    map!!(f, y, xs...)

`map(f, xs...)` in the storage of `y`, and return it: the elementwise updates that are not an
iterate update of [`update_iterate!!`](@ref) — a negation, a difference, a copy.
"""
map!!(f::F, ::Nothing, xs::Vararg{Any, N}) where {F, N} = map(f, xs...)
map!!(f::F, y::AbstractArray, xs::Vararg{Any, N}) where {F, N} = (map!(f, y, xs...); y)

"""
    mul!!(C, A, B)

The matrix product `A * B`, in the storage of `C`, and return it: the Gram matrix of the Anderson
and weighting histories, and a matrix–vector product where no [`jvp!!`](@ref) applies.
"""
mul!!(::Nothing, A, B) = A * B
mul!!(C::AbstractVecOrMat, A, B) = mul!(C, A, B)

"""
    copy!!(dst, src)

Copy `src` into the storage of `dst`, and return it. On a value slot (`nothing`, a `StaticArray`,
or a `Number`) this returns `src` unchanged; on an array buffer slot it copies `src` into `dst`
via `copyto!`.
"""
copy!!(::Nothing, src) = src
copy!!(::StaticArray, src) = src
copy!!(::Number, src) = src
copy!!(y::AbstractArray, src) = copyto!(y, src)

"""
    scale!!(dst, α, x)

The scaled vector `α * x`, in the storage of `dst`, and return it.
"""
scale!!(::Nothing, α, x) = α * x
scale!!(dst::AbstractArray, α, x) = (dst .= α .* x; dst)

"""
    axpby!!(dst, α, x, β, y)

The linear combination `α * x + β * y`, in the storage of `dst`, and return it.
"""
axpby!!(::Nothing, α, x, β, y) = α * x + β * y
function axpby!!(dst::AbstractArray, α, x, β, y)
    dst .= α .* x .+ β .* y
    return dst
end

"""
    regularize!!(J, μ)

Add `μ * I` to the Jacobian `J` in its storage, and return it. When `iszero(μ)` holds, `J` is
returned untouched. On a `Matrix` the diagonal is shifted in a loop so bounds-checked runs
allocate nothing; on any other `AbstractMatrix` the diagonal view `view(J, diagind(J))` is updated
by broadcast so device arrays perform no scalar indexing.
"""
function regularize!!(J::SMatrix{N, N, T, L}, μ) where {N, T, L}
    iszero(μ) && return J
    μT = T(μ)
    return SMatrix{N, N, T, L}(ntuple(Val(L)) do k
        i, j = divrem(k - 1, N)
        @inbounds J[k] + ifelse(i == j, μT, zero(T))
    end)
end

regularize!!(J::T, μ) where {T <: Number} = iszero(μ) ? J : J + T(μ)

function regularize!!(J::AbstractMatrix{T}, μ) where {T}
    if !iszero(μ)
        μT = T(μ)
        if J isa Matrix
            @inbounds for i in axes(J, 1)
                J[i, i] += μT
            end
        else
            view(J, diagind(J)) .+= μT
        end
    end
    return J
end

# The `Number` iterate: its Jacobian is a `Number`, and its factor is the `1 × 1` static factor,
# in the element type that `factor_eltype` gives. `linear/factors.jl` declares these
# two functions and names their factor types; it is included after this file. So these two methods
# state the storage — `nothing`, and a `Number` — and let the factor type dispatch inside, which
# keeps the factor types in the one file that owns them. A method type with no `1 × 1` factor,
# such as a `TF32` marker, then raises the same host `MethodError` it raises for a matrix.
factorize!!(fac, method, J::Number) = factorize!!(fac, method, SMatrix{1, 1}(J))
ldiv!!(::Nothing, fac, r::Number) = ldiv!!(nothing, fac, SVector{1}(r))[1]

"""
    jacobian!!(J, prep, prob, x, p)

The Jacobian of the residual of `prob` at the iterate `x`, for the parameters `p`, in the storage
of `J`, and return it.

`prep` is the Jacobian source. In R1 the output slot `J` is `nothing`: `nothing` calls the
problem's `J(x, p)`, an `AutoForwardDiff` runs tagged ForwardDiff and returns an `SMatrix` in the
element type of a vector iterate or a `Number` for a `Number` iterate, and a
[`JacobianFunction`](@ref) calls the method's own function. [`factorize!!`](@ref) converts the
result to the factorisation precision `TF`. In R3 `J` is an `n × n` buffer in the element type
of `x`: `nothing` calls `prob.J(J, x, p)`, a [`JacobianFunction`](@ref) calls `s.f(J, x, p)`, a
[`DIJacobian`](@ref) runs DifferentiationInterface's prepared `jacobian!`, and a
[`ChunkedForwardDiff`](@ref) runs the own chunked forward mode. The preparation selects the code.
"""
function jacobian!! end

"""
    jvp!!(Jv, prep, prob, x, v, p)

The Jacobian–vector product `J(x, p) * v` of the residual of `prob`, in the storage of `Jv`, and
return it.

With an AD source, one pushforward and exactly one evaluation of the residual make the merit
derivative `φ′(α) = F(x + αd)ᵀ J(x + αd) d` of a line search cost one directional derivative
rather than `n`. With the source `nothing` or a [`JacobianMatrixJVP`](@ref), the call evaluates the
problem's `J` instead of `F`.

`prep` is the JVP source from the problem, never from a method's operator,
and it selects the code exactly as in [`jacobian!!`](@ref). In R1 the output slot `Jv` is
`nothing` and the call returns a new value: `nothing` gives `J(x, p) * v` from the problem's `J`,
and an `AutoForwardDiff` one dual evaluation of `F`. The R1 direction has the element type and the
length of the iterate, as the line-search slope does: a direction of another element type is a
host `MethodError`. In R3 `Jv` is a buffer and `prep` is a [`JacobianMatrixJVP`](@ref), a
[`DIJacobian`](@ref) or a [`ChunkedForwardDiff`](@ref).
"""
function jvp!! end
