"""
    Static(α = 1.0)

The line search that takes the fixed step `α`, bounded by the caller's `αmax`, and evaluates
nothing: the common case inside an integrator. It ignores the trial step and returns `SUCCESS`
with `φ = NaN`.
"""
struct Static{R <: Real} <: LineSearch{R}
    α::R
    function Static{R}(α::R) where {R <: Real}
        α > 0 && isfinite(α) ||
            throw(ArgumentError("the step of Static must be positive and finite, got $α"))
        new{R}(α)
    end
end

Static(α::Real = 1.0) = Static{float(typeof(α))}(float(α))

Adapt.@adapt_structure Static

function linesearch(
        ls::Static{R}, lf, step::StepKind, φ₀::R, α::R, αmax::R = R(Inf)) where {R}
    usable_ceiling(αmax) || return LineSearchResult{R}(ls.α, R(NaN), LINESEARCH_FAILED, 0)
    LineSearchResult{R}(min(ls.α, αmax), R(NaN), SUCCESS, 0)
end
