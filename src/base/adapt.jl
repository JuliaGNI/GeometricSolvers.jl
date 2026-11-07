"""
    ToReal{R}()

The `Adapt` rule that converts every `AbstractFloat` in a method tree to `R`. Applied on the host
with `R = real(eltype(x))`, it lets a method written with `Float64` literals run in a `Float32`
kernel that never sees a `Float64`. An integer, a type and a function pass through unchanged.

Each method type that holds a float adds `Adapt.@adapt_structure` for itself.

```jldoctest
julia> GeometricSolvers.Adapt.adapt(GeometricSolvers.ToReal{Float32}(), (0.7, Int32(3), sin))
(0.7f0, 3, sin)
```
"""
struct ToReal{R <: AbstractFloat} end

Adapt.adapt_storage(::ToReal{R}, x::AbstractFloat) where {R} = convert(R, x)

"""
    StripDual()

The `Adapt` rule that strips every `ForwardDiff.Dual` and `Complex{<:ForwardDiff.Dual}` leaf to
its primal value: a `ForwardDiff.Dual` leaf via `ForwardDiff.value`, a
`Complex{<:ForwardDiff.Dual}` leaf in both real and imaginary parts, an array of duals elementwise,
and a `Tuple`, `NamedTuple` or struct with `Adapt.@adapt_structure` recursively. Any other value
passes through unchanged.

```jldoctest
julia> d = GeometricSolvers.ForwardDiff.Dual{Nothing}(3.5, 1.0);

julia> GeometricSolvers.Adapt.adapt(GeometricSolvers.StripDual(), (d, Complex(d, d)))
(3.5, 3.5 + 3.5im)
```
"""
struct StripDual end

Adapt.adapt_storage(::StripDual, d::ForwardDiff.Dual) = ForwardDiff.value(d)
function Adapt.adapt_storage(::StripDual, z::Complex{<:ForwardDiff.Dual})
    return Complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
end

function Adapt.adapt_storage(::StripDual, xs::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.value.(xs)
end

function Adapt.adapt_storage(::StripDual, xs::AbstractArray{<:Complex{<:ForwardDiff.Dual}})
    return Complex.(ForwardDiff.value.(real.(xs)), ForwardDiff.value.(imag.(xs)))
end
