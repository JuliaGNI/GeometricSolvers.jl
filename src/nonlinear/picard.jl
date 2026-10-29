"""
    Picard(; damping = 1, globalization = Static())

Damped fixed-point (Picard) iteration for [`NonlinearProblem`](@ref), with search direction
``d = -\\mathrm{damping}\\,F(x, p)`` and globalization `globalization` ([`Static`](@ref),
[`Backtracking`](@ref), [`Bisection`](@ref) or [`StrongWolfe`](@ref)).

# Fields

- `damping::D`: positive finite damping factor ``\\omega > 0``, stored in floating-point.
- `globalization::G`: the globalization method. Any line search uses [`MeasuredSlope`](@ref).
"""
struct Picard{D <: Real, G}
    damping::D
    globalization::G
    function Picard{D, G}(damping::D, globalization::G) where {D <: Real, G}
        damping > zero(D) && isfinite(damping) ||
            throw(ArgumentError("Picard damping must be positive and finite, got $damping"))
        new{D, G}(damping, globalization)
    end
end

function Picard(damping::D, globalization::G) where {D <: Real, G}
    Picard{D, G}(damping, globalization)
end

function Picard(; damping::Real = 1, globalization = Static())
    d = float(damping)
    return Picard{typeof(d), typeof(globalization)}(d, globalization)
end

Adapt.@adapt_structure Picard

@inline function picard_jvpprep(prob::NonlinearProblem, Fx, x, p)
    if prob.J !== nothing
        return Fx === nothing ? nothing :
               JacobianMatrixJVP(buffer(x, eltype(x), length(x), length(x)))
    else
        return prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
    end
end

function init(m::Picard, prob::NonlinearProblem, x, p = nothing)
    Fx = buffer(x)
    jvpprep = picard_jvpprep(prob, Fx, x, p)
    if Fx === nothing
        return (; jacprep = nothing, jvpprep)
    else
        d = buffer(x)
        g = init_globalization(m.globalization, x)
        return (; jacprep = nothing, jvpprep, Fx, d, g)
    end
end

@inline state_d(state) = hasproperty(state, :d) ? state.d : nothing
@inline state_g(state) = hasproperty(state, :g) ? state.g : nothing

@inline function step!!(state, m::Picard, prob::NonlinearProblem, x, Fx, p)
    ω = eltype(x)(m.damping)
    d = map!!(fi -> -ω * fi, state_d(state), Fx)
    x₊, F₊, _, info = globalize!!(
        state_g(state), m.globalization, prob, x, Fx, d, p, state.jvpprep, MeasuredSlope())
    return x₊, F₊, state, info
end
