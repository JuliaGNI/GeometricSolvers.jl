"""
    Backtracking(; c₁ = 1e-4, p = 0.5, maxiter = 100)

The Armijo line search with a safeguarded quadratic and cubic interpolation step. It shrinks the
trial step until the [`sufficient_decrease`](@ref) test

```math
φ(α) - φ(0) ≤ \\min\\bigl(0,\\ c_1 α φ′(0) + τ\\bigr)
```

holds, where ``τ`` is the [`roundoff`](@ref) of ``φ(0)``. Each backtrack takes the minimiser of
the model through ``φ(0)``, ``φ′(0)`` and the last one or two trials, clamped to
``[0.1 α, p α]``. A trial whose merit is not finite gives no model; the next trial is then the
smaller of ``0.1 α`` and the geometric mean of ``α`` and the step floor. The search stops at the
step floor [`smallest_step`](@ref), after two trials at steps below ``\\sqrt{\\mathrm{eps}}``
whose merit equals ``φ(0)`` bit for bit, or after `maxiter` trials.

The slope ``φ′(0)`` comes from the step kind: ``-2φ(0)`` for an [`ExactStep`](@ref), one
evaluation for a [`MeasuredSlope`](@ref), the number itself for a slope passed in. For an
[`InexactStep`](@ref) with residual ``η`` the test is Eisenstat and Walker's
``\\|r(x + α d)\\| ≤ [1 - c_1 (1 - η(α))] \\|r\\|`` with ``η(α) = |1 - α| + α η``, and the model
slope is ``-2(1 - η) φ(0)``.

The model is built from differences of the merit divided by ``|φ′(0)|``, and the cubic step uses
the form that does not cancel on a near-quadratic merit, so the trials do not depend on the
scale of the merit.
"""
struct Backtracking{R <: Real} <: LineSearch{R}
    c₁::R
    p::R
    maxiter::Int32
    function Backtracking{R}(c₁::Real, p::Real, maxiter::Integer) where {R <: Real}
        0 < c₁ < 1 || throw(ArgumentError("Backtracking needs 0 < c₁ < 1, got c₁ = $c₁"))
        0 < p < 1 || throw(ArgumentError("Backtracking needs 0 < p < 1, got p = $p"))
        1 ≤ maxiter ≤ typemax(Int32) ||
            throw(ArgumentError("Backtracking needs 1 ≤ maxiter ≤ typemax(Int32), got $maxiter"))
        new{R}(c₁, p, maxiter)
    end
end

function Backtracking(c₁::Real, p::Real, maxiter::Integer)
    Backtracking{float(promote_type(typeof(c₁), typeof(p)))}(c₁, p, maxiter)
end
function Backtracking(; c₁::Real = 1e-4, p::Real = 0.5, maxiter::Integer = 100)
    Backtracking(c₁, p, maxiter)
end

Adapt.@adapt_structure Backtracking

"""
    backtrack_step(φ₀, d₀, α, φα, αp, φp, p)

The next trial step after the rejected trial `α` with merit `φα`; `αp` and `φp` are the trial
before it, and `αp` is `NaN` on the first backtrack. The model of ``φ`` is a quadratic through
``φ(0)``, ``φ′(0) = d_0`` and ``φ(α)``, then a cubic through ``φ(α_p)`` as well, both in the
merit divided by ``|d_0|``. The cubic minimiser is ``1 / (b + \\sqrt{b^2 + 3a})`` for ``b > 0``,
which does not cancel as ``a → 0``. The result is clamped to ``[0.1 α, p α]``; a model without a
minimiser gives ``p α``, and a merit that is not finite at `α` gives ``0.1 α``.
"""
function backtrack_step(φ₀::R, d₀::R, α::R, φα::R, αp::R, φp::R, p::R) where {R}
    isfinite(φα) || return R(0.1) * α
    αₙ = R(NaN)
    s = abs(d₀)
    # The model m(t) = φ₀/s - t + b t² + a t³ of the merit divided by s = |d₀|, and r(t) = b t² + a t³.
    r₁ = (φα - φ₀) / s + α
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
    (isfinite(αₙ) && αₙ > zero(R)) || (αₙ = p * α)
    min(max(αₙ, R(0.1) * α), p * α)
end

# An inexact step bounds the slope only; the model uses the bound.
function search_slope(::Backtracking, step::InexactStep, lf, φ₀)
    (-2 * (1 - oftype(φ₀, step.η)) * φ₀, Int32(0))
end

# The acceptance test at the trial step α.
accepts(step, ls, φα, φ₀, d₀, α, τ) = sufficient_decrease(φα, φ₀, ls.c₁ * α * d₀, τ)
function accepts(step::InexactStep, ls, φα, φ₀, d₀, α, τ)
    η = abs(1 - α) + α * oftype(α, step.η)
    η < 1 && sufficient_decrease(φα, φ₀, ((1 - ls.c₁ * (1 - η))^2 - 1) * φ₀, τ)
end

function search(ls::Backtracking{R}, lf, step, φ₀::R, d₀::R, τ::R, α::R, ceiling::R,
        n::Int32) where {R}
    αmin = smallest_step(d₀, τ)
    αₐ, φₐ = α, R(NaN)    # the last trial and its merit
    αp, φp = R(NaN), R(NaN)
    frozen = 0             # consecutive trials whose merit equals φ₀ bit for bit
    code = LINESEARCH_FAILED
    done = false
    for _ in 1:ls.maxiter
        done && continue
        αₐ, φₐ = α, φ(lf, α)
        n += Int32(1)
        # A merit equal to φ₀ is taken as a frozen trial point only for a step small enough that
        # x + αd may round to x; a larger step with φ = φ₀ is an ordinary rejected trial.
        frozen = φₐ == φ₀ && α ≤ sqrt(eps(R)) ? frozen + 1 : 0
        if accepts(step, ls, φₐ, φ₀, d₀, α, τ)
            code = classify(φₐ, φ₀, τ)
            done = true
        elseif frozen ≥ 2
            # the trial point no longer differs from the anchor in floating point
            code = STALLED
            done = true
        elseif α ≤ αmin
            code = floor_code(φₐ, φ₀, τ)
            done = true
        else
            # a merit that is not finite gives no model: shrink in log space towards the floor,
            # so that a trial step many decades too long costs log₂ of the decades, not tens
            αₙ = isfinite(φₐ) ? backtrack_step(φ₀, d₀, α, φₐ, αp, φp, ls.p) :
                 min(R(0.1) * α, sqrt(α) * sqrt(αmin))
            αp, φp, α = α, φₐ, max(αₙ, αmin)
        end
    end
    LineSearchResult{R}(αₐ, φₐ, code, n)
end
