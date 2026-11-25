"""
    PseudoTransient{R}
    PseudoTransient([::Type{R} = Float64,] δ₀; δ_max = R(1e12)) where {R <: AbstractFloat}

Pseudo-transient continuation for [`Newton`](@ref) on a [`StationarityProblem`](@ref): each step
solves the shifted system

```math
\\left(J(x_k) + \\frac{1}{\\delta_k} I\\right) d_k = -F(x_k)
```

and takes `x_{k+1} = x_k + d_k`; every step is accepted. A small pseudo-time step `δ` follows the
gradient flow `ẋ = -∇S(x)`, which leaves a saddle of `S` along a direction of negative curvature,
and a large one gives the Newton step. `δ` starts at `δ₀` in each solve and follows the switched
evolution relaxation

```math
\\delta_{k+1} = \\min\\!\\left(\\delta_k \\, \\frac{\\|F(x_k)\\|_2}{\\|F(x_{k+1})\\|_2}, \\; \\delta_{\\max}\\right),
```

so it grows as the residual falls. A non-finite `δ` or `1/δ`, or a non-finite residual at the new
iterate, ends the solve with `NONFINITE`, and the iterate stays where it was. Every step sets
`limited` in its [`StepInfo`](@ref), because the shift and not the distance to a root sets its
norm, so a short step never ends the solve as a stall.

The parameters satisfy `0 < δ₀ ≤ δ_max < Inf`, else the constructor raises an `ArgumentError`;
[`ToReal`](@ref) converts them to the real type of the iterate. `Newton` with a `PseudoTransient`
takes no `reuse` and no refinement, and needs a factorising linear method: a [`GMRES`](@ref), a
[`MINRESQLP`](@ref) or a [`KroneckerFactorization`](@ref) raises an `ArgumentError`. A
[`NonlinearProblem`](@ref) or a [`LeastSquaresProblem`](@ref) raises an `ArgumentError` where
the state is built: the shift follows the gradient flow of `S` only where `F` is a gradient.
"""
struct PseudoTransient{R <: AbstractFloat}
    δ₀::R
    δ_max::R
    # `adapt` rebuilds a `PseudoTransient` through this constructor, inside a device kernel too,
    # so the check also refuses a value that `ToReal` narrows to `Inf` or to zero. A kernel cannot
    # format a message, so the message is constant.
    function PseudoTransient{R}(δ₀::R, δ_max::R) where {R <: AbstractFloat}
        zero(R) < δ₀ <= δ_max < R(Inf) ||
            throw(ArgumentError("PseudoTransient requires 0 < δ₀ ≤ δ_max < Inf"))
        return new{R}(δ₀, δ_max)
    end
end

function PseudoTransient(
        ::Type{R}, δ₀::Real; δ_max::Real = R(1e12)) where {R <: AbstractFloat}
    return PseudoTransient{R}(R(δ₀), R(δ_max))
end
PseudoTransient(δ₀::Real; δ_max::Real = 1e12) = PseudoTransient(Float64, δ₀; δ_max)

# The positional form, which `adapt` rebuilds a `PseudoTransient` from.
PseudoTransient(δ₀::R, δ_max::R) where {R <: AbstractFloat} = PseudoTransient{R}(δ₀, δ_max)

Adapt.@adapt_structure PseudoTransient

# The trial buffers of the full step: the iterate and the residual at `x + d`, as for `Static`.
@inline function init_globalization(::PseudoTransient, ::LinearMethod, x, Fx)
    return (; xt = buffer(x), Ft = buffer(Fx))
end

# The pseudo-time step after a step from a residual of norm `fnorm` to one of norm `fnorm₊`: the
# ratio is taken first, so that neither norm is squared or multiplied by `δ` alone.
@inline function pt_next_step(pt::PseudoTransient, δ::R, fnorm::R, fnorm₊::R) where {R}
    return min(δ * (fnorm / fnorm₊), R(pt.δ_max))
end
