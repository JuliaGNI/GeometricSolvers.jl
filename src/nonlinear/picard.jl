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

@inline function picard_jvpprep(m::Picard, prob::NonlinearProblem, Fx, x, p)
    if prob.J !== nothing
        return Fx === nothing ? nothing : jacobian_matrix_jvp(m.globalization, x)
    else
        return prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
    end
end

# The state of `Picard`: the Jacobian-vector preparation of its line search, the residual,
# direction and globalization workspace, each `nothing` for a value iterate, and the merit
# `norm2(Fx)` of the current residual for a line search, `nothing` for any other globalization.
struct PicardState{VP, FX, D, G, PH}
    jvpprep::VP
    Fx::FX
    d::D
    g::G
    merit::PH
end

@inline function init_state(m::Picard, prob::NonlinearProblem, x, p)
    Fx = buffer(x)
    jvpprep = picard_jvpprep(m, prob, Fx, x, p)
    merit = init_merit(m.globalization, x)
    if Fx === nothing
        return PicardState(jvpprep, nothing, nothing, nothing, merit)
    else
        return PicardState(
            jvpprep, Fx, buffer(x), init_globalization(m.globalization, x), merit)
    end
end

@inline residual_buffer(state::PicardState) = state.Fx

@inline function with_merit(s::PicardState, merit)
    return PicardState(s.jvpprep, s.Fx, s.d, s.g, merit)
end

@inline function start!!(state::PicardState, m::Picard, prob, x, Fx, p, fnorm₀)
    return with_merit(state, start_merit(m.globalization, Fx))
end

# The damping is real, so that `-ω F` of a complex `F` is the real scaling of each part.
@inline function step!!(state::PicardState, m::Picard, prob::NonlinearProblem, x, Fx, p)
    ω = real(eltype(x))(m.damping)
    d = map!!(fi -> -ω * fi, state.d, Fx)
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, MeasuredSlope(), state.merit)
    return x₊, F₊, with_merit(state, merit), info
end
