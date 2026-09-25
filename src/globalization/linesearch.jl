# What the four line searches share: the line-function interface, the result, the kinds of step
# a caller can hand in, the one entry point with its ceiling, trial-step and anchor checks, and
# the round-off rules. Every function here is scalar, allocates nothing and never throws, so the
# same code runs on the host and inside a kernel.

"""
    LineSearch{R}

The supertype of the line searches [`Static`](@ref), [`Backtracking`](@ref),
[`Bisection`](@ref) and [`StrongWolfe`](@ref), whose numbers are of the real type `R`. A line
search is a method: it is `isbits`, it holds numbers and an `Int32` cap and nothing else, and
[`linesearch`](@ref) runs it.

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
abstract type LineSearch{R <: Real} end

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
``φ = \\|r\\|^2`` it gives ``φ′(0) = -2φ(0)``, so no line search evaluates ``φ′(0)``. For
``φ(0) > \\mathrm{floatmax}(R)/2`` that slope overflows, and the search reports `NONFINITE`.
"""
struct ExactStep end

"""
    InexactStep(η)

The step kind of a direction `d` with the linear residual ``\\|r + J d\\| ≤ η \\|r\\|``,
``0 ≤ η < 1``. It gives only ``φ′(0) ≤ -2(1 - η) φ(0)`` (Eisenstat and Walker 1996, (1.2)).
Along ``α d`` the linear residual is ``(1 - α) r + α (r + J d)``, so its bound is
``η(α) = |1 - α| + α η``. [`Backtracking`](@ref) tests
``\\|r(x + α d)\\| ≤ [1 - c_1 (1 - η(α))] \\|r\\|`` with this ``η(α)``, which for ``α ≤ 1`` is the
update ``η ← 1 - θ (1 - η)`` of each backtrack by ``θ``, and evaluates no ``φ′``. A step with
``η(α) ≥ 1`` promises no decrease and is rejected. [`Bisection`](@ref) and
[`StrongWolfe`](@ref) need the true slope and evaluate ``φ′(0)``.
"""
struct InexactStep{R <: Real}
    η::R
end

"""
    MeasuredSlope()

The step kind of any other descent direction, such as one from a reused Jacobian: every line
search except [`Static`](@ref) evaluates ``φ′(0)`` once. A caller that already has ``φ′(0)``
passes the number itself as the step kind, and no search evaluates it.
"""
struct MeasuredSlope end

const StepKind = Union{ExactStep, InexactStep, MeasuredSlope, Real}

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
    linesearch(ls::LineSearch{R}, lf, step, φ₀::R, α::R, αmax::R = R(Inf))

Run the line search `ls` along the line function `lf` from the trial step `α`, and return a
[`LineSearchResult{R}`](@ref LineSearchResult). `φ₀` is the merit at ``α = 0``, which the caller
has; `step` is an [`ExactStep`](@ref), an [`InexactStep`](@ref), a [`MeasuredSlope`](@ref) or the
slope ``φ′(0)`` itself; `αmax` is the caller's ceiling on the step. A trial step that is not
positive or not finite is replaced by 1.

A non-positive or `NaN` `αmax` is a caller error. The search then evaluates nothing and returns
`LINESEARCH_FAILED` with the trial step, bounded by the ceiling of the method.

This is the one entry point: it applies the ceiling, the trial step and the anchor checks, then
runs the loop of the method. Each loop has exactly `ls.maxiter` trips with a done flag, so that
the threads of a warp stay together in a kernel; a trip after the flag is set evaluates nothing.
"""
function linesearch(ls::LineSearch{R}, lf, step::StepKind, φ₀::R, α::R,
        αmax::R = R(Inf)) where {R}
    usable_ceiling(αmax) ||
        return LineSearchResult{R}(trial_step(α, method_αmax(ls)), R(NaN), LINESEARCH_FAILED, 0)
    ceiling = min(method_αmax(ls), αmax)
    α = trial_step(α, ceiling)
    d₀, n = search_slope(ls, step, lf, φ₀)
    usable_anchor(φ₀, d₀) || return LineSearchResult{R}(α, R(NaN), anchor_code(φ₀, d₀), n)
    τ = roundoff(φ₀)
    search(ls, lf, step, φ₀, d₀, τ, α, ceiling, n)
end

# The ceiling of the method itself.
method_αmax(ls::LineSearch{R}) where {R} = R(Inf)

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

# The slope at the anchor, and what it cost. `search_slope` is what a method uses; a method with
# its own rule for a step kind adds a method of it.
anchor_slope(::ExactStep, lf, φ₀) = (-2φ₀, Int32(0))
anchor_slope(::Union{InexactStep, MeasuredSlope}, lf, φ₀) = (φ′(lf, zero(φ₀)), Int32(1))
anchor_slope(d₀::Real, lf, φ₀) = (oftype(φ₀, d₀), Int32(0))
search_slope(ls, step, lf, φ₀) = anchor_slope(step, lf, φ₀)

"""
    roundoff(φ₀)

The round-off resolution ``τ = 4\\,\\mathrm{eps}(R)\\,|φ(0)|`` of the merit, four units of
relative round-off. A step that changes the merit by no more than ``τ`` is at the round-off
floor of the merit. It is proportional to ``|φ(0)|``, not a count of ulps of ``φ(0)``, so that
it and the step floor [`smallest_step`](@ref) scale with the merit and do not jump by a factor 2
at a power of two. For a subnormal ``φ(0)``, where the product underflows, it is at least four
of the smallest subnormals, so it is never 0.
"""
roundoff(φ₀) = max(4 * eps(typeof(φ₀)) * abs(φ₀), 4 * nextfloat(zero(φ₀)))

"""
    smallest_step(d₀, τ)

The step floor ``τ / |φ′(0)|`` of every search (decision 40): below it the decrease that the
slope predicts, ``α |φ′(0)|``, is smaller than the round-off ``τ``, so no trial can show it. It
is at least `floatmin(R)`, so that no search tries a step of 0 when the quotient underflows.
"""
smallest_step(d₀, τ) = max(τ / abs(d₀), floatmin(typeof(τ)))

"""
    sufficient_decrease(φα, φ₀, demand, τ)

The Armijo test on the difference, ``φ(α) - φ(0) ≤ \\min(0, \\mathrm{demand} + τ)``, where
`demand` is the negative decrease the test demands and ``τ`` the round-off allowance. The
difference keeps a demand below one ulp of ``φ(0)``, which ``φ(0) + \\mathrm{demand}`` would
round away; the `min` lets ``τ`` lower the demand, never accept an increase. [`StrongWolfe`](@ref)
passes ``τ = 0``.
"""
sufficient_decrease(φα, φ₀, demand, τ) = φα - φ₀ ≤ min(zero(φ₀), demand + τ)

"""
    classify(φα, φ₀, τ)

The code of an accepted step: `SUCCESS` for a finite decrease by more than ``τ``, `STALLED` for
a change by no more than ``τ``, and `LINESEARCH_FAILED` for an increase by more than ``τ`` or a
merit that is not finite.
"""
function classify(φα, φ₀, τ)
    isfinite(φα) && φα - φ₀ ≤ -τ && return SUCCESS
    floor_code(φα, φ₀, τ)
end

# The code of a step that is not accepted: `STALLED` at the round-off floor of the merit, where
# it changes by no more than τ, and `LINESEARCH_FAILED` otherwise.
floor_code(φα, φ₀, τ) = abs(φα - φ₀) ≤ τ ? STALLED : LINESEARCH_FAILED

# Whether d and Δ have the same sign, or d is zero: the sign test of d · Δ ≥ 0 without the
# product, which can underflow to -0.0.
samesign(d, Δ) = iszero(d) || ((d > zero(d)) == (Δ > zero(Δ)))
