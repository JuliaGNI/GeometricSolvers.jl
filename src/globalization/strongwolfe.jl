"""
    StrongWolfe(; c₁ = 1e-4, c₂ = 0.9, αmax = 65536.0, maxiter = 100)

The line search for a step with the strong Wolfe conditions

```math
φ(α) ≤ φ(0) + c_1 α φ′(0), \\qquad |φ′(α)| ≤ c_2 |φ′(0)|,
```

``0 < c_1 < c_2 < 1``: Nocedal and Wright's Algorithms 3.5 and 3.6. The bracketing phase doubles
the trial step up to the smaller of `αmax` and the caller's ceiling until a trial fails the
first condition, does not decrease the merit, or has ``φ′ ≥ 0``. The zoom phase then takes the
minimiser of the cubic through the merit and its derivative at both ends of the bracket,
safeguarded to the inner 80 % of the bracket, in the Moré–Thuente style. The lower end is always
the trial with the lowest merit that meets the first condition; two merits within the
[`roundoff`](@ref) of the larger of them tie, and the sign of ``φ′`` at the trial decides the
side of the minimiser.
A zoom trial that meets both conditions is accepted whatever its merit.

It returns `SUCCESS` only for a step that meets both conditions and decreases the merit by more
than its round-off. Every other search returns its lowest trial that meets the first condition,
or else its last trial. At the ceiling, at a collapsed bracket or at the
[`smallest_step`](@ref) the code is `STALLED` if the merit is at its round-off floor and
`LINESEARCH_FAILED` otherwise; after `maxiter` trials it is `LINESEARCH_FAILED`. Each trial
costs one ``φ`` and one ``φ′``.
"""
struct StrongWolfe{R <: Real} <: LineSearch
    c₁::R
    c₂::R
    αmax::R
    maxiter::Int32
end

function StrongWolfe(; c₁::Real = 1e-4, c₂::Real = 0.9, αmax::Real = 65536.0,
        maxiter::Integer = 100)
    0 < c₁ < c₂ < 1 ||
        throw(ArgumentError("StrongWolfe needs 0 < c₁ < c₂ < 1, got c₁ = $c₁, c₂ = $c₂"))
    αmax > 0 || throw(ArgumentError("StrongWolfe needs αmax > 0, got αmax = $αmax"))
    maxiter ≥ 1 || throw(ArgumentError("StrongWolfe needs maxiter ≥ 1, got $maxiter"))
    R = float(promote_type(typeof(c₁), typeof(c₂), typeof(αmax)))
    StrongWolfe{R}(c₁, c₂, αmax, maxiter)
end

Adapt.@adapt_structure StrongWolfe

"""
    zoom_step(a, φa, da, b, φb, db)

The minimiser of the cubic through the merit `φa`, `φb` and its derivative `da`, `db` at the two
ends `a` and `b` of a bracket (Nocedal and Wright, (3.59)), clamped to the inner 80 % of the
bracket. The discriminant is scaled by the largest of ``|θ|``, ``|d_a|``, ``|d_b|``, as in
Moré and Thuente, so it neither overflows nor depends on the scale of the merit. A cubic without
a minimiser gives the middle of the bracket.
"""
function zoom_step(a::R, φa::R, da::R, b::R, φb::R, db::R) where {R}
    θ = 3 * (φa - φb) / (b - a) + da + db
    s = max(abs(θ), abs(da), abs(db))
    γ² = (θ / s)^2 - (da / s) * (db / s)
    lo, hi = minmax(a, b)
    δ = R(0.1) * (hi - lo)
    γ² ≥ zero(R) || return (a + b) / 2
    γ = b > a ? s * sqrt(γ²) : -s * sqrt(γ²)
    αc = b - (b - a) * (db + γ - θ) / (db - da + 2γ)
    isfinite(αc) ? clamp(αc, lo + δ, hi - δ) : (a + b) / 2
end

# The two strong Wolfe conditions, exact: no round-off allowance.
armijo(φα, φ₀, c₁, α, d₀) = φα ≤ φ₀ + c₁ * α * d₀
curvature(dα, c₂, d₀) = abs(dα) ≤ -c₂ * d₀

# The code of a step that does not meet both strong Wolfe conditions.
unmet_code(φα, φ₀, τ) = abs(φα - φ₀) ≤ τ ? STALLED : LINESEARCH_FAILED

function linesearch(ls::StrongWolfe{R}, lf, step::StepKind, φ₀::R, α::R, αmax::R) where {R}
    usable_ceiling(αmax) ||
        return LineSearchResult{R}(trial_step(α, ls.αmax), R(NaN), LINESEARCH_FAILED, 0)
    ceiling = min(ls.αmax, αmax)
    α = trial_step(α, ceiling)
    d₀, n = anchor_slope(step, lf, φ₀)
    usable_anchor(φ₀, d₀) || return LineSearchResult{R}(α, R(NaN), anchor_code(φ₀, d₀), n)

    c₁, c₂ = ls.c₁, ls.c₂
    τ = roundoff(φ₀)
    αmin = smallest_step(c₁, d₀, τ)

    # the previous trial of the bracketing phase, and the next one
    αp, φp, dp = zero(R), φ₀, d₀
    αi = α
    # the bracket of the zoom phase: `lo` has the lowest merit of the trials that meet `armijo`
    lo, φlo, dlo = zero(R), φ₀, d₀
    hi, φhi, dhi = zero(R), φ₀, d₀
    # the step to return if no trial meets both conditions
    αres, φres = α, R(NaN)
    code = LINESEARCH_FAILED
    zooming = false
    initial = true
    done = false
    for _ in 1:ls.maxiter
        done && continue
        if !zooming
            φi, di = φ(lf, αi), φ′(lf, αi)
            n += Int32(2)
            αres, φres = αi, φi
            if !armijo(φi, φ₀, c₁, αi, d₀) || (!initial && φi ≥ φp)
                zooming = true
                lo, φlo, dlo = αp, φp, dp
                hi, φhi, dhi = αi, φi, di
            elseif curvature(di, c₂, d₀)
                code = φi ≤ φ₀ - τ ? SUCCESS : STALLED
                done = true
            elseif di ≥ zero(R)
                zooming = true
                lo, φlo, dlo = αi, φi, di
                hi, φhi, dhi = αp, φp, dp
            elseif αi == ceiling
                code = unmet_code(φi, φ₀, τ)
                done = true
            else
                αp, φp, dp = αi, φi, di
                αi = min(2αi, ceiling)
            end
            initial = false
            # entering the zoom, the best step so far is its lower end, if it is a step
            zooming && lo > zero(R) && ((αres, φres) = (lo, φlo))
        else
            αj = zoom_step(lo, φlo, dlo, hi, φhi, dhi)
            φj, dj = φ(lf, αj), φ′(lf, αj)
            n += Int32(2)
            # the round-off of the two merits compared, which may be far from φ₀
            τj = roundoff(max(abs(φj), abs(φlo)))
            if !armijo(φj, φ₀, c₁, αj, d₀) || φj > φlo + τj
                hi, φhi, dhi = αj, φj, dj
                lo > zero(R) || ((αres, φres) = (αj, φj))
            elseif curvature(dj, c₂, d₀)
                αres, φres = αj, φj
                code = φj ≤ φ₀ - τ ? SUCCESS : STALLED
                done = true
            elseif φj ≥ φlo && dj * (αj - lo) ≥ zero(R)
                # a tie within the round-off, and φ rises at αj: the minimiser is below αj
                hi, φhi, dhi = αj, φj, dj
            else
                dj * (hi - lo) ≥ zero(R) && ((hi, φhi, dhi) = (lo, φlo, dlo))
                lo, φlo, dlo = αj, φj, dj
                αres, φres = αj, φj
            end
            if !done && (abs(hi - lo) ≤ eps(R) * max(lo, hi) || max(lo, hi) ≤ αmin)
                code = unmet_code(φres, φ₀, τ)
                done = true
            end
        end
    end
    LineSearchResult{R}(αres, φres, code, n)
end
