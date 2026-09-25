"""
    StrongWolfe(; c₁ = 1e-4, c₂ = 0.9, αmax = 65536.0, maxiter = 100)

The line search for a step with the strong Wolfe conditions

```math
φ(α) - φ(0) ≤ c_1 α φ′(0), \\qquad |φ′(α)| ≤ c_2 |φ′(0)|,
```

``0 < c_1 < c_2 < 1``: Nocedal and Wright's Algorithms 3.5 and 3.6. The first condition is the
[`sufficient_decrease`](@ref) test without a round-off allowance. The bracketing phase doubles
the trial step up to the smaller of `αmax` and the caller's ceiling until a trial fails the
first condition, does not decrease the merit, or has ``φ′ ≥ 0``. The zoom phase then takes the
minimiser of the cubic through the merit and its derivative at both ends of the bracket, or of
the quadratic through both merits and the derivative at the lower end where the upper end has
no derivative. The trial is safeguarded to the inner 80 % of the bracket, and it is the middle
after a trial that shrinks the bracket by less than a factor 0.66, as in Moré and Thuente. The
lower end is the trial with the lowest merit that meets the first condition, up to the
[`roundoff`](@ref) of the larger of two merits: a trial within it of the lower end ties with it,
and the sign of ``φ′`` at the trial decides the side of the minimiser. A zoom trial that meets
both conditions is accepted whatever its merit.

``φ′`` is evaluated only at a trial that meets the first condition and, in the bracketing phase,
decreases the merit: a trial that fails costs one ``φ``, a trial that passes one ``φ`` and one
``φ′``.

It returns `SUCCESS` only for a step that meets both conditions and decreases the merit by more
than its round-off. Every other search returns its lowest trial that meets the first condition,
or else its last trial. At the ceiling, at a collapsed bracket or at the step floor
[`smallest_step`](@ref) the code is `STALLED` if the merit is at its round-off floor and
`LINESEARCH_FAILED` otherwise; after `maxiter` trials it is `LINESEARCH_FAILED`.
"""
struct StrongWolfe{R <: Real} <: LineSearch{R}
    c₁::R
    c₂::R
    αmax::R
    maxiter::Int32
    function StrongWolfe{R}(c₁::Real, c₂::Real, αmax::Real, maxiter::Integer) where {R <:
                                                                                     Real}
        0 < c₁ < c₂ < 1 ||
            throw(ArgumentError("StrongWolfe needs 0 < c₁ < c₂ < 1, got c₁ = $c₁, c₂ = $c₂"))
        αmax > 0 || throw(ArgumentError("StrongWolfe needs αmax > 0, got αmax = $αmax"))
        maxiter ≥ 1 || throw(ArgumentError("StrongWolfe needs maxiter ≥ 1, got $maxiter"))
        new{R}(c₁, c₂, αmax, maxiter)
    end
end

function StrongWolfe(c₁::Real, c₂::Real, αmax::Real, maxiter::Integer)
    R = float(promote_type(typeof(c₁), typeof(c₂), typeof(αmax)))
    StrongWolfe{R}(c₁, c₂, αmax, maxiter)
end
function StrongWolfe(; c₁::Real = 1e-4, c₂::Real = 0.9, αmax::Real = 65536.0,
        maxiter::Integer = 100)
    StrongWolfe(c₁, c₂, αmax, maxiter)
end

Adapt.@adapt_structure StrongWolfe

method_αmax(ls::StrongWolfe) = ls.αmax

"""
    zoom_step(a, φa, da, b, φb, db)

The minimiser of the cubic through the merit `φa`, `φb` and its derivative `da`, `db` at the two
ends `a` and `b` of a bracket (Nocedal and Wright, (3.59)), clamped to the inner 80 % of the
bracket. The discriminant is scaled by the largest of ``|θ|``, ``|d_a|``, ``|d_b|``, as in
Moré and Thuente, so it neither overflows nor depends on the scale of the merit. A cubic without
a minimiser gives the middle of the bracket. Without a finite `db` it is the minimiser of the
quadratic through `φa`, `da` and `φb`.
"""
function zoom_step(a::R, φa::R, da::R, b::R, φb::R, db::R) where {R}
    lo, hi = minmax(a, b)
    δ = R(0.1) * (hi - lo)
    if isfinite(db)
        θ = 3 * (φa - φb) / (b - a) + da + db
        s = max(abs(θ), abs(da), abs(db))
        γ² = (θ / s)^2 - (da / s) * (db / s)
        γ² ≥ zero(R) || return midpoint(a, b)
        γ = b > a ? s * sqrt(γ²) : -s * sqrt(γ²)
        αc = b - (b - a) * (db + γ - θ) / (db - da + 2γ)
    else
        c = (φb - φa - da * (b - a)) / (b - a)^2
        c > zero(R) || return midpoint(a, b)
        αc = a - da / (2c)
    end
    isfinite(αc) ? clamp(αc, lo + δ, hi - δ) : midpoint(a, b)
end

# The two strong Wolfe conditions, exact: no round-off allowance.
armijo(φα, φ₀, c₁, α, d₀) = sufficient_decrease(φα, φ₀, c₁ * α * d₀, zero(φ₀))
curvature(dα, c₂, d₀) = abs(dα) ≤ -c₂ * d₀

function search(ls::StrongWolfe{R}, lf, step, φ₀::R, d₀::R, τ::R, α::R, ceiling::R,
        n::Int32) where {R}
    c₁, c₂ = ls.c₁, ls.c₂
    αmin = smallest_step(d₀, τ)
    top = min(ceiling, floatmax(R))    # without a ceiling the doubling stops at floatmax
    unknown = R(NaN)    # the slope at a trial where it was not evaluated

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
    bisect = false    # the last zoom trial shrank the bracket by less than 0.66
    initial = true
    done = false
    for _ in 1:ls.maxiter
        done && continue
        if !zooming
            φi = φ(lf, αi)
            n += Int32(1)
            αres, φres = αi, φi
            if !armijo(φi, φ₀, c₁, αi, d₀) || (!initial && φi ≥ φp)
                zooming = true
                lo, φlo, dlo = αp, φp, dp
                hi, φhi, dhi = αi, φi, unknown
            else
                di = φ′(lf, αi)
                n += Int32(1)
                if curvature(di, c₂, d₀)
                    code = classify(φi, φ₀, τ)
                    done = true
                elseif di ≥ zero(R)
                    zooming = true
                    lo, φlo, dlo = αi, φi, di
                    hi, φhi, dhi = αp, φp, dp
                elseif αi == top
                    code = floor_code(φi, φ₀, τ)
                    done = true
                else
                    αp, φp, dp = αi, φi, di
                    αi = min(2αi, top)
                end
            end
            initial = false
            # entering the zoom, the best step so far is its lower end, if it is a step
            zooming && lo > zero(R) && ((αres, φres) = (lo, φlo))
        elseif abs(hi - lo) ≤ eps(R) * max(lo, hi) || max(lo, hi) ≤ αmin
            # a collapsed bracket, or one below the step floor
            code = floor_code(φres, φ₀, τ)
            done = true
        else
            width = abs(hi - lo)
            # no trial below the step floor: there the merit cannot show the predicted decrease
            αj = max(bisect ? midpoint(lo, hi) : zoom_step(lo, φlo, dlo, hi, φhi, dhi), αmin)
            φj = φ(lf, αj)
            n += Int32(1)
            # the round-off of the two merits compared, which may be far from φ₀
            τj = roundoff(max(abs(φj), abs(φlo)))
            if !armijo(φj, φ₀, c₁, αj, d₀) || φj > φlo + τj
                hi, φhi, dhi = αj, φj, unknown
                lo > zero(R) || ((αres, φres) = (αj, φj))
            else
                dj = φ′(lf, αj)
                n += Int32(1)
                if curvature(dj, c₂, d₀)
                    αres, φres = αj, φj
                    code = classify(φj, φ₀, τ)
                    done = true
                else
                    # the sign of φ′ at αj decides the side of the minimiser, also for a merit
                    # that ties with φlo within the round-off
                    samesign(dj, hi - lo) && ((hi, φhi, dhi) = (lo, φlo, dlo))
                    lo, φlo, dlo = αj, φj, dj
                    αres, φres = αj, φj
                end
            end
            bisect = abs(hi - lo) > R(0.66) * width
        end
    end
    LineSearchResult{R}(αres, φres, code, n)
end
