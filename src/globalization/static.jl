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
        return new{R}(α)
    end
    # The constructor of the `adapt` rule: it rebuilds a `Static` whose step the checked constructor
    # validated and checks nothing, because a device kernel cannot format the message of the
    # checked constructor.
    Static{R}(α::Real, ::Val{:raw}) where {R <: Real} = new{R}(α)
end

Static(α::Real = 1.0) = Static{float(typeof(α))}(float(α))

Adapt.adapt_storage(::ToReal{R}, ls::Static) where {R <: Real} = Static{R}(ls.α, Val(:raw))

# The check of a `Static` after `adapt`, which the raw constructor above cannot make: `ToReal` may
# narrow a step to `Inf` or to zero.
function check_adapted(ls::Static{R}) where {R}
    ls.α > 0 && isfinite(ls.α) || throw(ArgumentError(
        "the step of Static must be positive and finite after adaptation to $R, got $(ls.α)"))
    return ls
end

function linesearch(
        ls::Static{R}, lf, step::StepKind{R}, φ₀::R, α::R, αmax::R = R(Inf)) where {R}
    F₀ = anchor_residual(lf)
    usable_ceiling(αmax) ||
        return LineSearchResult{R, typeof(F₀)}(ls.α, R(NaN), LINESEARCH_FAILED, 0, F₀)
    LineSearchResult{R, typeof(F₀)}(min(ls.α, αmax), R(NaN), SUCCESS, 0, F₀)
end
