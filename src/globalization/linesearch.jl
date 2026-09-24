# What the four line searches share: the line-function interface, the result, the three kinds
# of step a caller can hand in, and the anchor, ceiling and round-off rules. Every function here
# is scalar, allocates nothing and never throws, so the same code runs on the host and inside a
# kernel.

"""
    LineSearch

The supertype of the line searches [`Static`](@ref), [`Backtracking`](@ref),
[`Bisection`](@ref) and [`StrongWolfe`](@ref). A line search is a method: it is `isbits`, it
holds numbers and an `Int32` cap and nothing else, and [`linesearch`](@ref) runs it.

Every line search keeps six contracts:

1. It never throws. A constructor checks its parameters on the host and raises an
   `ArgumentError`; [`linesearch`](@ref) raises nothing.
2. It returns ``α > 0``.
3. It reports through the `code` of its [`LineSearchResult`](@ref) and never logs.
4. A non-finite or ascending anchor is reported, not searched: `NONFINITE` and
   `LINESEARCH_FAILED`. A stationary anchor, ``φ′(0) = 0``, is `STALLED`.
5. Its number of evaluations does not depend on the scale of the merit.
6. It returns ``α ≤ α_{max}``, for the caller's `αmax` and for its own.

It never evaluates the merit at ``α = 0``: the caller passes ``φ(0)``, which is the residual it
already has.
"""
abstract type LineSearch end

"""
    φ(lf, α)

The merit of the line function `lf` at the step `α`, ``φ(α) = \\|r(x + α d)\\|^2``, in the real
type of `α`. A line search sees a line function only through this and [`φ′`](@ref).
"""
function φ end

"""
    φ′(lf, α)

The derivative ``dφ/dα`` of the merit of the line function `lf` at the step `α`: one
Jacobian–vector product.
"""
function φ′ end

"""
    ExactStep()

The step kind of a direction from a fresh Jacobian and an exact linear solve. For the merit
``φ = \\|r\\|^2`` it gives ``φ′(0) = -2φ(0)``, so no line search evaluates ``φ′(0)``.
"""
struct ExactStep end

"""
    InexactStep(η)

The step kind of a direction `d` with the linear residual ``\\|r + J d\\| ≤ η \\|r\\|``,
``0 ≤ η < 1``. It gives only ``φ′(0) ≤ -2(1 - η) φ(0)`` (Eisenstat and Walker 1996, (1.2)).
[`Backtracking`](@ref) then tests ``\\|r(x + s)\\| ≤ [1 - c_1 (1 - η)] \\|r\\|`` and updates
``η ← 1 - θ (1 - η)`` for each backtrack by ``θ``, and evaluates no ``φ′``. From the trial step
``α`` the residual is ``η = 1 - α (1 - η_0)``; a step with ``c_1 α (1 - η_0) ≥ 1`` makes the
factor ``1 - c_1 (1 - η)`` non-positive and is rejected.
[`Bisection`](@ref) and [`StrongWolfe`](@ref) need the true slope and evaluate ``φ′(0)``.
"""
struct InexactStep{R <: Real}
    η::R
end

"""
    MeasuredSlope()

The step kind of any other descent direction, such as one from a reused Jacobian: every line
search except [`Static`](@ref) evaluates ``φ′(0)`` once.
"""
struct MeasuredSlope end

const StepKind = Union{ExactStep, InexactStep, MeasuredSlope}

"""
    LineSearchResult{R <: Real}

The result of one [`linesearch`](@ref). It is `isbits`.

# Fields

- `α::R`: the step, ``0 < α ≤ α_{max}``.
- `φ::R`: the merit at `α`, or `NaN` if the search did not evaluate it there.
- `code::ReturnCode`: `SUCCESS` for an accepted step that decreases the merit by more than its
  round-off; `STALLED` for a step that changes the merit by no more than its round-off, or a
  stationary anchor; `NONFINITE` for a non-finite anchor; `LINESEARCH_FAILED` for anything
  else: an ascending anchor, a spent cap, an increase of the merit, an invalid `αmax`, or a
  [`StrongWolfe`](@ref) step without the curvature condition. `φ` tells the caller whether the
  step of a failure still decreases the merit.
- `evaluations::Int32`: the evaluations of [`φ`](@ref) and [`φ′`](@ref) together.
"""
struct LineSearchResult{R <: Real}
    α::R
    φ::R
    code::ReturnCode
    evaluations::Int32
end

"""
    linesearch(ls::LineSearch, lf, step, φ₀::R, α::R, αmax::R = R(Inf))

Run the line search `ls` along the line function `lf` from the trial step `α`, and return a
[`LineSearchResult{R}`](@ref LineSearchResult). `φ₀` is the merit at ``α = 0``, which the caller
has; `step` is an [`ExactStep`](@ref), an [`InexactStep`](@ref) or a [`MeasuredSlope`](@ref);
`αmax` is the caller's ceiling on the step. A trial step that is not positive is replaced by 1.

A non-positive or `NaN` `αmax` is a caller error. The search then evaluates nothing and returns
`LINESEARCH_FAILED` with the trial step, bounded by the ceiling of the method.

Each search is one loop of exactly `ls.maxiter` trips with a done flag, so that the threads of
a warp stay together in a kernel; a trip after the flag is set evaluates nothing.
"""
function linesearch end

function linesearch(ls::LineSearch, lf, step::StepKind, φ₀::R, α::R) where {R <: Real}
    linesearch(ls, lf, step, φ₀, α, R(Inf))
end

# A ceiling is usable when it is positive; `NaN > 0` is false.
usable_ceiling(αmax) = αmax > zero(αmax)

# The trial step: a non-positive or non-finite one is replaced by the unit step, then bounded.
function trial_step(α::R, ceiling::R) where {R}
    min(α > zero(R) && isfinite(α) ? α : one(R), ceiling)
end

# The anchor may be searched from when it is finite and descending.
usable_anchor(φ₀, d₀) = isfinite(φ₀) & isfinite(d₀) & (d₀ < zero(d₀))

# The code of an anchor that may not be searched from.
function anchor_code(φ₀, d₀)
    isfinite(φ₀) && isfinite(d₀) || return NONFINITE
    d₀ > zero(d₀) ? LINESEARCH_FAILED : STALLED
end

# The slope at the anchor, and what it cost.
anchor_slope(::ExactStep, lf, φ₀) = (-2φ₀, Int32(0))
anchor_slope(::Union{InexactStep, MeasuredSlope}, lf, φ₀) = (φ′(lf, zero(φ₀)), Int32(1))

"""
    roundoff(φ₀)

The round-off resolution ``τ = 4\\,\\mathrm{ulp}(φ(0))`` of the merit. A step that changes the
merit by no more than ``τ`` is at the round-off floor of the merit.
"""
roundoff(φ₀) = 4 * eps(φ₀)

"""
    smallest_step(c, d₀, τ)

The smallest step ``τ / (c |φ′(0)|)`` at which a demanded decrease ``c α |φ′(0)|`` still
exceeds the round-off ``τ``, clamped to ``[\\mathrm{eps}(R), \\sqrt{\\mathrm{eps}(R)}]``. Below
it, a trial carries no information, so a search stops there instead of spending its cap.
"""
function smallest_step(c::R, d₀::R, τ::R) where {R}
    αmin = τ / (c * abs(d₀))
    isfinite(αmin) || (αmin = sqrt(eps(R)))
    clamp(αmin, eps(R), sqrt(eps(R)))
end

"""
    sufficient_decrease(φα, φ₀, demand, τ)

The Armijo test with the round-off allowance: ``φ(α) ≤ \\min(φ(0), φ(0) + \\mathrm{demand} + τ)``,
where `demand` is the negative decrease the test demands. The `min` lets ``τ`` lower the demand,
never accept an increase.
"""
sufficient_decrease(φα, φ₀, demand, τ) = φα ≤ min(φ₀, φ₀ + demand + τ)

"""
    classify(φα, φ₀, τ)

The code of an accepted step: `SUCCESS` for a decrease by more than ``τ``, `STALLED` for a
change by no more than ``τ``, and `LINESEARCH_FAILED` for an increase by more than ``τ`` or a
merit that is not finite.
"""
function classify(φα, φ₀, τ)
    φα ≤ φ₀ - τ && return SUCCESS
    floor_code(φα, φ₀, τ)
end

# The code of a step that is not accepted: `STALLED` at the round-off floor of the merit, where
# it changes by no more than τ, and `LINESEARCH_FAILED` otherwise.
floor_code(φα, φ₀, τ) = abs(φα - φ₀) ≤ τ ? STALLED : LINESEARCH_FAILED
