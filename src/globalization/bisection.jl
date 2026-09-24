"""
    Bisection(; αmax = 65536.0, maxiter = 100)

The line search that bisects ``φ′`` for a minimiser of the merit along the line. It brackets from
the anchor: the lower end starts at ``0``, where ``φ′(0) < 0``, and the upper end at the trial
step, doubled while ``φ′ < 0``, up to the smaller of its own `αmax` and the caller's. So the
bracket always holds a sign change from ``-`` to ``+``: a minimum, never a maximum. It then
halves the bracket until its width is below ``\\sqrt{\\mathrm{eps}}`` times its upper end, or
the upper end is below the [`smallest_step`](@ref). The second stop is what ends the search when
the lower end stays at ``0``.

It returns the middle of the bracket, or the ceiling if ``φ′`` is still negative there, and
classifies it by the merit there: one evaluation of ``φ`` after one ``φ′`` per trial.
"""
struct Bisection{R <: Real} <: LineSearch
    αmax::R
    maxiter::Int32
end

function Bisection(; αmax::Real = 65536.0, maxiter::Integer = 100)
    αmax > 0 || throw(ArgumentError("Bisection needs αmax > 0, got αmax = $αmax"))
    maxiter ≥ 1 || throw(ArgumentError("Bisection needs maxiter ≥ 1, got $maxiter"))
    Bisection{float(typeof(αmax))}(αmax, maxiter)
end

Adapt.@adapt_structure Bisection

function linesearch(ls::Bisection{R}, lf, step::StepKind, φ₀::R, α::R, αmax::R) where {R}
    usable_ceiling(αmax) ||
        return LineSearchResult{R}(trial_step(α, ls.αmax), R(NaN), LINESEARCH_FAILED, 0)
    ceiling = min(ls.αmax, αmax)
    α = trial_step(α, ceiling)
    d₀, n = anchor_slope(step, lf, φ₀)
    usable_anchor(φ₀, d₀) || return LineSearchResult{R}(α, R(NaN), anchor_code(φ₀, d₀), n)

    τ = roundoff(φ₀)
    αmin = smallest_step(one(R), d₀, τ)
    rtol = sqrt(eps(R))
    lo, hi = zero(R), α    # φ′(lo) < 0 throughout; φ′(hi) ≥ 0 once `bracketing` is false
    αres = α
    bracketing = true
    done = false
    for _ in 1:ls.maxiter
        done && continue
        if bracketing
            dh = φ′(lf, hi)
            n += Int32(1)
            if iszero(dh)
                αres, done = hi, true
            elseif !(dh < zero(R))
                bracketing = false
            elseif hi == ceiling
                αres, done = hi, true
            else
                lo, hi = hi, min(2hi, ceiling)
            end
        else
            m = (lo + hi) / 2
            dm = φ′(lf, m)
            n += Int32(1)
            if iszero(dm)
                αres, done = m, true
            else
                dm < zero(R) ? (lo = m) : (hi = m)
                if hi - lo ≤ rtol * hi || hi ≤ αmin
                    αres, done = (lo + hi) / 2, true
                end
            end
        end
    end
    done || (αres = (lo + hi) / 2)
    φres = φ(lf, αres)
    n += Int32(1)
    LineSearchResult{R}(αres, φres, done ? classify(φres, φ₀, τ) : LINESEARCH_FAILED, n)
end
