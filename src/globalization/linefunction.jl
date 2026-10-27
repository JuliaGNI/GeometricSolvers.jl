"""
    LineFunction(prob, x, Fx, d, p, jvpprep)

The line function along the ray from `x` in the direction `d` for [`NonlinearProblem`](@ref) `prob`
with parameters `p` and Jacobian-vector preparation `jvpprep`; `Fx` is the residual at `x`. It is
a callable struct, not a subtype of `Function`, so `isbits` holds when its fields are `isbits`.

[`φ(lf, α)`](@ref φ) advances `x` along `d` by `α` through `update_iterate!!`, evaluates the
residual through `residual!!`, and returns its merit [`norm2`](@ref), ``\\|F\\|_2^2``.
[`φ′(lf, α)`](@ref φ′) evaluates the derivative of the merit,
``2\\,\\mathrm{Re}\\langle F, J d\\rangle``, through `jvp!!` and [`rdot`](@ref). A line search
along `lf` returns the residual at its step in its [`LineSearchResult`](@ref), and `Fx` for a step
it did not evaluate. Calling `lf(α)` delegates to `φ(lf, α)`.
"""
struct LineFunction{P, X, FX, D, Q, JP}
    prob::P
    x::X
    Fx::FX
    d::D
    p::Q
    jvpprep::JP
end

@inline function merit_and_residual(lf::LineFunction, α)
    xα = update_iterate!!(nothing, lf.x, α, lf.d)
    Fα = residual!!(nothing, lf.prob, xα, lf.p)
    return norm2(Fα), Fα
end

anchor_residual(lf::LineFunction) = lf.Fx

@inline φ(lf::LineFunction, α) = merit_and_residual(lf, α)[1]

@inline (lf::LineFunction)(α) = φ(lf, α)

@inline function φ′(lf::LineFunction, α)
    xα = update_iterate!!(nothing, lf.x, α, lf.d)
    Fα = residual!!(nothing, lf.prob, xα, lf.p)
    Jv = jvp!!(nothing, lf.jvpprep, lf.prob, xα, lf.d, lf.p)
    return 2 * rdot(Fα, Jv)
end

"""
    globalize!!(g, ls, prob, x, Fx, d, p, jvpprep, stepkind = MeasuredSlope())

Advance `x` along `d` under the globalization method `ls` and return `(x₊, F₊, g, info)` where
`info` is a [`StepInfo`](@ref).

For [`Static`](@ref), `globalize!!` takes the fixed step `ls.α` through `update_iterate!!` and
evaluates `residual!!` once, without a line-search loop. When the step `d` or the new residual norm
is not finite, it returns `(x, Fx, g, StepInfo{R}(NONFINITE, rnorm(Fx), abs(ls.α) * rnorm(d), 0,
false, false))` without advancing the iterate or evaluating `F` at a non-finite trial point.

For [`Backtracking`](@ref), [`Bisection`](@ref) and [`StrongWolfe`](@ref), `globalize!!` returns
`NONFINITE` immediately when `rnorm(d)` is not finite; otherwise it builds a
[`LineFunction`](@ref) and calls [`linesearch`](@ref) from `α = 1`. An accepted step returns the
residual that the search evaluated at `res.α`, so it costs no further evaluation of `F`. A
`LINESEARCH_FAILED` result whose `φ` is finite and strictly less than `φ₀ = norm2(Fx)` takes the
step `res.α` and reports `failures = 1`; otherwise the iterate stays at `x` with `stepnorm = 0`.
"""
@inline function globalize!!(
        g, ls::Static{R}, prob, x, Fx, d, p, jvpprep, stepkind = MeasuredSlope()) where {R}
    s = ls.α
    dn = rnorm(d)
    stepnorm = abs(s) * dn
    if !isfinite(stepnorm)
        return x, Fx, g, StepInfo{R}(NONFINITE, rnorm(Fx), stepnorm, Int32(0), false, false)
    end
    x₊ = update_iterate!!(nothing, x, s, d)
    F₊ = residual!!(nothing, prob, x₊, p)
    fnorm₊ = rnorm(F₊)
    if !isfinite(fnorm₊)
        return x, Fx, g, StepInfo{R}(NONFINITE, rnorm(Fx), stepnorm, Int32(0), false, false)
    end
    return x₊, F₊, g, StepInfo{R}(SUCCESS, fnorm₊, stepnorm, Int32(0), false, false)
end

@inline function globalize!!(
        g, ls::LineSearch{R}, prob, x, Fx, d, p,
        jvpprep, stepkind = MeasuredSlope()) where {R}
    dn = rnorm(d)
    if !isfinite(dn)
        return x, Fx, g, StepInfo{R}(NONFINITE, rnorm(Fx), dn, Int32(0), false, false)
    end
    lf = LineFunction(prob, x, Fx, d, p, jvpprep)
    φ₀ = norm2(Fx)
    res = linesearch(ls, lf, stepkind, φ₀, one(R))
    code = res.code
    if code == SUCCESS
        x₊ = update_iterate!!(nothing, x, res.α, d)
        return x₊, res.residual,
        g,
        StepInfo{R}(SUCCESS, sqrt(res.φ), res.α * dn, Int32(0), false, false)
    elseif code == LINESEARCH_FAILED
        if isfinite(res.φ) && res.φ < φ₀
            x₊ = update_iterate!!(nothing, x, res.α, d)
            return x₊, res.residual,
            g,
            StepInfo{R}(LINESEARCH_FAILED, sqrt(res.φ), res.α * dn, Int32(1), false, false)
        else
            return x, Fx,
            g,
            StepInfo{R}(LINESEARCH_FAILED, sqrt(φ₀), zero(R), Int32(1), false, false)
        end
    elseif code == STALLED
        return x, Fx, g, StepInfo{R}(STALLED, sqrt(φ₀), zero(R), Int32(0), false, false)
    else
        return x, Fx, g, StepInfo{R}(NONFINITE, sqrt(φ₀), zero(R), Int32(0), false, false)
    end
end
