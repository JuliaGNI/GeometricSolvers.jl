"""
    Backtracking(; c₁ = 1e-4, p = 0.5, maxiter = 100)

The Armijo line search with a safeguarded quadratic and cubic interpolation step. It shrinks the
trial step until

```math
φ(α) ≤ \\min\\bigl(φ(0),\\ φ(0) + c_1 α φ′(0) + τ\\bigr),
```

where ``τ`` is the [`roundoff`](@ref) of ``φ(0)``. Each backtrack takes the minimiser of the
model through ``φ(0)``, ``φ′(0)`` and the last one or two trials, clamped to ``[0.1 α, p α]``.
The search stops at the smallest informative step [`smallest_step`](@ref), after two trials
whose merit equals ``φ(0)`` bit for bit, or after `maxiter` trials.

The slope ``φ′(0)`` comes from the step kind: ``-2φ(0)`` for an [`ExactStep`](@ref), one
evaluation for a [`MeasuredSlope`](@ref). For an [`InexactStep`](@ref) with residual ``η`` the
test is Eisenstat and Walker's ``\\|r(x + s)\\| ≤ [1 - c_1 (1 - η)] \\|r\\|``, with
``η ← 1 - θ (1 - η)`` for each backtrack by ``θ`` and the model slope ``-2(1 - η) φ(0)``.

The model is built from differences of the merit divided by ``|φ′(0)|``, and the cubic step uses
the form that does not cancel on a near-quadratic merit, so the trials do not depend on the
scale of the merit.
"""
struct Backtracking{R <: Real} <: LineSearch
    c₁::R
    p::R
    maxiter::Int32
end

function Backtracking(; c₁::Real = 1e-4, p::Real = 0.5, maxiter::Integer = 100)
    0 < c₁ < 1 || throw(ArgumentError("Backtracking needs 0 < c₁ < 1, got c₁ = $c₁"))
    0 < p < 1 || throw(ArgumentError("Backtracking needs 0 < p < 1, got p = $p"))
    maxiter ≥ 1 || throw(ArgumentError("Backtracking needs maxiter ≥ 1, got $maxiter"))
    R = float(promote_type(typeof(c₁), typeof(p)))
    Backtracking{R}(c₁, p, maxiter)
end

Adapt.@adapt_structure Backtracking

"""
    backtrack_step(φ₀, d₀, α, φα, αp, φp, p)

The next trial step after the rejected trial `α` with merit `φα`; `αp` and `φp` are the trial
before it, and `αp` is `NaN` on the first backtrack. The model of ``φ`` is a quadratic through
``φ(0)``, ``φ′(0) = d_0`` and ``φ(α)``, then a cubic through ``φ(α_p)`` as well, both in the
merit divided by ``|d_0|``. The cubic minimiser is ``1 / (b + \\sqrt{b^2 + 3a})`` for ``b > 0``,
which does not cancel as ``a → 0``. The result is clamped to ``[0.1 α, p α]``; a model without a
minimiser gives ``p α``.
"""
function backtrack_step(φ₀::R, d₀::R, α::R, φα::R, αp::R, φp::R, p::R) where {R}
    αₙ = R(NaN)
    s = abs(d₀)
    # The model m(t) = φ₀/s - t + b t² + a t³ of the merit divided by s = |d₀|, and r(t) = b t² + a t³.
    r₁ = (φα - φ₀) / s + α
    if isfinite(r₁)
        if !(αp > zero(R))
            r₁ > zero(R) && (αₙ = α^2 / (2r₁))
        elseif isfinite(φp) && α != αp
            q₁ = r₁ / α^2
            q₂ = ((φp - φ₀) / s + αp) / αp^2
            a = (q₁ - q₂) / (α - αp)
            b = (α * q₂ - αp * q₁) / (α - αp)
            disc = b^2 + 3a
            if disc ≥ zero(R)
                αₙ = b > zero(R) ? inv(b + sqrt(disc)) : (-b + sqrt(disc)) / (3a)
            end
        end
    end
    (isfinite(αₙ) && αₙ > zero(R)) || (αₙ = p * α)
    min(max(αₙ, R(0.1) * α), p * α)
end

# The slope the model and the test of each step kind use, and its cost.
backtracking_slope(::ExactStep, lf, φ₀) = (-2φ₀, Int32(0))
backtracking_slope(::MeasuredSlope, lf, φ₀) = (φ′(lf, zero(φ₀)), Int32(1))
backtracking_slope(step::InexactStep, lf, φ₀) = (-2 * (1 - step.η) * φ₀, Int32(0))

# η of the trial step α: an inexact step d with residual η₀ gives the residual 1 - α(1 - η₀) along
# α d, for α ≤ 1. The other step kinds have none.
initial_residual(step::InexactStep{R}, α::R) where {R} = 1 - α * (1 - step.η)
initial_residual(step, α::R) where {R} = zero(R)

# The acceptance test at the trial step α with residual η.
function accepts(::Union{ExactStep, MeasuredSlope}, ls, φα, φ₀, d₀, α, η, τ)
    sufficient_decrease(φα, φ₀, ls.c₁ * α * d₀, τ)
end
function accepts(::InexactStep, ls, φα, φ₀, d₀, α, η, τ)
    φα ≤ min(φ₀, (1 - ls.c₁ * (1 - η))^2 * φ₀ + τ)
end

function linesearch(ls::Backtracking{R}, lf, step::StepKind, φ₀::R, α::R,
        αmax::R) where {R}
    usable_ceiling(αmax) ||
        return LineSearchResult{R}(trial_step(α, R(Inf)), R(NaN), LINESEARCH_FAILED, 0)
    α = trial_step(α, αmax)
    d₀, n = backtracking_slope(step, lf, φ₀)
    usable_anchor(φ₀, d₀) || return LineSearchResult{R}(α, R(NaN), anchor_code(φ₀, d₀), n)

    τ = roundoff(φ₀)
    αmin = smallest_step(ls.c₁, d₀, τ)
    η = initial_residual(step, α)
    αₐ, φₐ = α, R(NaN)    # the last trial and its merit
    αp, φp = R(NaN), R(NaN)
    frozen = 0             # consecutive trials whose merit equals φ₀ bit for bit
    code = LINESEARCH_FAILED
    done = false
    for _ in 1:ls.maxiter
        done && continue
        αₐ, φₐ = α, φ(lf, α)
        n += Int32(1)
        frozen = φₐ == φ₀ ? frozen + 1 : 0
        if accepts(step, ls, φₐ, φ₀, d₀, α, η, τ)
            code = φₐ ≤ φ₀ - τ ? SUCCESS : STALLED
            done = true
        elseif frozen ≥ 2
            # the trial point no longer differs from the anchor in floating point
            code = STALLED
            done = true
        elseif α ≤ αmin
            code = abs(φₐ - φ₀) ≤ τ ? STALLED : LINESEARCH_FAILED
            done = true
        else
            αₙ = max(backtrack_step(φ₀, d₀, α, φₐ, αp, φp, ls.p), αmin)
            η = 1 - (αₙ / α) * (1 - η)
            αp, φp, α = α, φₐ, αₙ
        end
    end
    LineSearchResult{R}(αₐ, φₐ, code, n)
end
