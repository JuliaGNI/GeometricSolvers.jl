"""
    ToReal{R}()

The `Adapt` rule that converts every `AbstractFloat` in a method tree to `R`. `init` applies it on
the host, with `R = real(eltype(x))`, so that `Picard(; damping = 0.7)` runs in a `Float32`
kernel that never sees a `Float64`. An integer, a type and a function pass through unchanged.

Each method type that holds a float adds `Adapt.@adapt_structure` for itself.

```jldoctest
julia> GeometricSolvers.Adapt.adapt(GeometricSolvers.ToReal{Float32}(), (0.7, Int32(3), sin))
(0.7f0, 3, sin)
```
"""
struct ToReal{R <: AbstractFloat} end

Adapt.adapt_storage(::ToReal{R}, x::AbstractFloat) where {R} = convert(R, x)
