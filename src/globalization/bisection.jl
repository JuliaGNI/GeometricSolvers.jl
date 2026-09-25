"""
    Bisection(; αmax = 65536.0, maxiter = 100)

The line search that bisects ``φ′`` for a minimiser of the merit along the line. It brackets from
the anchor: the lower end starts at ``0``, where ``φ′(0) < 0``, and the upper end at the trial
step, doubled while ``φ′ < 0``, up to the smaller of its own `αmax` and the caller's. So the
bracket always holds a sign change from ``-`` to ``+``: a minimum, never a maximum. A ``φ′``
that is not finite counts as not descending, so the bracket never grows into a region where
the merit is not finite.

It then bisects the bracket. While the bracket spans more than a factor 2, the next trial is the
geometric mean of its ends, with a lower end of ``0`` taken as the step floor
[`smallest_step`](@ref); after that it is the middle. It stops when the width is
below ``\\sqrt{\\mathrm{eps}}`` times the upper end, or when the lower end is still ``0`` and the
upper end is within a factor 2 of the floor. So the number of trials grows only with
``\\log_2 \\log_2`` of the ratio of the trial step to the floor, and it stays bounded for
`αmax = Inf`. While the lower end stays at ``0``, a search from the trial step ``α`` costs at most
``3 + ⌈\\log_2 \\log_2(α / \\mathrm{floatmin}(R))⌉`` evaluations, because the floor is at least
`floatmin(R)`: 10 in `Float32` and 13 in `Float64` for ``α = 1``.

It returns the lower end of the bracket, where ``φ′`` is finite and negative, its upper end if
the lower end is ``0``, or the ceiling if ``φ′`` is still negative there, and classifies it by
the merit there: one evaluation of ``φ`` after one ``φ′`` per trial.
"""
struct Bisection{R <: Real} <: LineSearch{R}
    αmax::R
    maxiter::Int32
    function Bisection{R}(αmax::Real, maxiter::Integer) where {R <: Real}
        αmax > 0 || throw(ArgumentError("Bisection needs αmax > 0, got αmax = $αmax"))
        maxiter ≥ 1 || throw(ArgumentError("Bisection needs maxiter ≥ 1, got $maxiter"))
        new{R}(αmax, maxiter)
    end
end

Bisection(αmax::Real, maxiter::Integer) = Bisection{float(typeof(αmax))}(αmax, maxiter)
Bisection(; αmax::Real = 65536.0, maxiter::Integer = 100) = Bisection(αmax, maxiter)

Adapt.@adapt_structure Bisection

method_αmax(ls::Bisection) = ls.αmax

# A slope that is finite and negative; any other points to a minimum below.
descending(d) = isfinite(d) && d < zero(d)

function search(ls::Bisection{R}, lf, step, φ₀::R, d₀::R, τ::R, α::R, ceiling::R,
        n::Int32) where {R}
    αmin = smallest_step(d₀, τ)
    rtol = sqrt(eps(R))
    top = min(ceiling, floatmax(R))
    lo, hi = zero(R), min(α, top)   # φ′(lo) < 0 throughout; φ′(hi) ≥ 0 once `bracketing` is false
    αres = hi
    bracketing = true
    done = false
    for _ in 1:ls.maxiter
        done && continue
        if bracketing
            dh = φ′(lf, hi)
            n += Int32(1)
            if iszero(dh)
                αres, done = hi, true
            elseif !descending(dh)
                bracketing = false
            elseif hi == top
                αres, done = hi, true
            else
                lo, hi = hi, min(2hi, top)
            end
        elseif hi - lo ≤ rtol * hi || (iszero(lo) && hi ≤ 2αmin)
            αres, done = iszero(lo) ? hi : lo, true
        else
            lower = lo > zero(R) ? lo : αmin
            m = hi > 2lower ? sqrt(hi) * sqrt(lower) : (lo + hi) / 2
            dm = φ′(lf, m)
            n += Int32(1)
            if iszero(dm)
                αres, done = m, true
            else
                descending(dm) ? (lo = m) : (hi = m)
            end
        end
    end
    done || (αres = iszero(lo) ? hi : lo)
    φres = φ(lf, αres)
    n += Int32(1)
    LineSearchResult{R}(αres, φres, done ? classify(φres, φ₀, τ) : LINESEARCH_FAILED, n)
end
