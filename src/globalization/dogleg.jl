"""
    DogLeg(; radius = nothing)

The trust-region globalization of [`Newton`](@ref) on the dogleg curve: `Newton(; globalization =
DogLeg())`. It is not a [`LineSearch`](@ref): it needs the Jacobian, its factor and a radius that
lives from one iteration to the next, so `Newton` holds the radius in its state and runs the trial
loop with its own Jacobian.

The defaults are Procedure 4.1 of Pawlowski, Simonis, Walker and Shadid (2008). The step is the
Newton step when it lies in the radius ``δ``, else the Cauchy step along ``-J^* F`` scaled to ``δ``
when the Cauchy point lies outside ``δ``, else the point of norm ``δ`` on the segment from the
Cauchy point to the Newton step. ``J`` is the matrix that was factorised, regularisation included.
A trial is accepted when the predicted reduction ``\\mathrm{pred} = \\|F(x)\\| - \\|F(x) + J s\\|``
is positive and the actual reduction ``\\mathrm{ared} = \\|F(x)\\| - \\|F(x + s)\\|`` satisfies
``\\mathrm{ared} ≥ 10^{-4} \\mathrm{pred}``, with unsquared norms; a rejected trial shrinks ``δ``
to ``\\max(δ/4, 10^{-6})`` and chooses again on the same curve, with no new Jacobian or solve. So
a rejected Newton step that lies well inside ``δ`` is tried again, at one evaluation of `F` each
time, until ``δ`` falls below its norm. A rejection at ``δ = 10^{-6}`` returns
`LINESEARCH_FAILED` with the iterate kept. After an accepted step with ratio
``ρ = \\mathrm{ared} / \\mathrm{pred}``, ``ρ < 0.1`` sets ``δ`` to ``\\max(\\|s^{N}\\|, 10^{-6})``
when the Newton step ``s^{N}`` is shorter than ``δ`` and to ``\\max(δ/4, 10^{-6})`` otherwise,
and ``ρ > 0.75`` on the boundary sets it to ``\\min(4δ, 10^{10})``. An accepted step other than
the whole Newton step sets `limited` in its [`StepInfo`](@ref): the radius or the Cauchy point, not
the distance to a root, sets its norm, so the solve applies only the residual test after it.

`radius` is the initial radius. It must be positive and finite as given; `DogLeg` stores it
converted to `float(typeof(radius))`, and [`ToReal`](@ref) to the iterate's real type, clamped to
``[10^{-6}, 10^{10}]`` after each conversion. With `nothing`, the first step starts the radius at
the norm of the first Newton step, clamped the same way, except that a norm below ``10^{-6}``
starts it at ``2 \\cdot 10^{-6}``, as Procedure 4.1 does. A singular factor, or a Newton step
whose norm is not finite, gives no Newton step: the step is then the Cauchy point, cut to the
radius when it lies outside, and a first radius starts at the length of the Cauchy point, by the same rule. The Cauchy
direction is formed from the unit residual, ``\\tilde g = J^* F / \\|F\\|``, so no norm of
``J^* F`` occurs, and a common scale of ``F`` and ``J`` leaves the Cauchy point unchanged up to
rounding, as long as the norm of ``\\tilde g`` is finite. Where the Cauchy point is needed and
``\\tilde g`` is zero or its norm is not finite, or ``J \\tilde g / \\|\\tilde g\\|`` is zero or its
norm is not finite, the step returns `STALLED` with the iterate kept. At ``F(x) = 0`` the step returns `SUCCESS` with the
iterate and the radius kept, and makes no trial.

`DogLeg` takes no factor reuse: `Newton(; globalization = DogLeg(), reuse = k)` raises an
`ArgumentError`.
"""
struct DogLeg{R <: Real, D <: Union{Nothing, R}}
    radius::D
    function DogLeg{R}(radius::Union{Nothing, Real}) where {R <: Real}
        if radius !== nothing
            radius > 0 && isfinite(radius) ||
                throw(ArgumentError("the radius of DogLeg must be positive and finite, got $radius"))
        end
        r = radius === nothing ? nothing : dogleg_clamp(R(radius))
        new{R, typeof(r)}(r)
    end
end

function DogLeg(; radius::Union{Nothing, Real} = nothing)
    radius === nothing ? DogLeg{Float64}(nothing) : DogLeg{float(typeof(radius))}(radius)
end

Adapt.adapt_structure(::ToReal{R}, dl::DogLeg) where {R} = DogLeg{R}(dl.radius)

# The constants of Procedure 4.1: the radius bounds, and the shrink and growth factors.
@inline dogleg_lo(::Type{R}) where {R} = R(1e-6)
@inline dogleg_hi(::Type{R}) where {R} = R(1e10)
@inline dogleg_bounds(::Type{R}) where {R} = (dogleg_lo(R), dogleg_hi(R))
@inline dogleg_clamp(δ::R) where {R} = clamp(δ, dogleg_lo(R), dogleg_hi(R))
@inline dogleg_shrink(δ::R) where {R} = max(R(0.25) * δ, dogleg_lo(R))
@inline dogleg_grow(δ::R) where {R} = min(R(4) * δ, dogleg_hi(R))

@inline function dogleg_initial_radius(dl::DogLeg, ::Type{R}) where {R}
    return dl.radius === nothing ? zero(R) : R(dl.radius)
end

@inline function init_globalization(::DogLeg, x)
    return (; g̃ = buffer(x), ĝ = buffer(x), Jg = buffer(x), s = buffer(x),
        xt = buffer(x), Ft = buffer(x))
end

@inline g_g̃(::Nothing) = nothing
@inline g_g̃(g) = g.g̃
@inline g_ĝ(::Nothing) = nothing
@inline g_ĝ(g) = g.ĝ
@inline g_Jg(::Nothing) = nothing
@inline g_Jg(g) = g.Jg
@inline g_s(::Nothing) = nothing
@inline g_s(g) = g.s

# The trial loop of `DogLeg` at `(x, Fx)`, with the model matrix `J`, the Newton step `sN` (read
# only when `newton` is `true`) and the radius `δ` (`0` when unset). Returns `(x₊, F₊, δ₊, info)`.
@inline function dogleg_step(prob, x, Fx, p, J, sN, newton::Bool, δ::R) where {R}
    return dogleg_step(nothing, prob, x, Fx, p, J, sN, newton, δ)
end

@inline function dogleg_step(g, prob, x, Fx, p, J, sN, newton::Bool, δ::R) where {R}
    lo = dogleg_lo(R)
    fnorm = rnorm(Fx)
    # at an exact root there is nothing to reduce: the iterate stays, and so does the radius
    if iszero(fnorm)
        return x, Fx, δ, StepInfo{R}(SUCCESS, fnorm, zero(R), Int32(0), false, false)
    end
    nN = newton ? rnorm(sN) : R(Inf)
    # The Cauchy point minimises the model norm along `-g`, `g = J' F`. Its direction is formed
    # from the unit residual, `g̃ = J' (F / ‖F‖)` and `ĝ = g̃ / ‖g̃‖`, and its length
    # `‖g‖ / ‖J ĝ‖² = ‖F‖ ‖g̃‖ / ‖J ĝ‖²`, so that no norm of the product `J' F` occurs. A `g̃`
    # that is zero or whose norm is not finite, or a `J ĝ` that is zero or whose norm is not finite,
    # gives no Cauchy point.
    u = scale!!(g_s(g), inv(fnorm), Fx)
    g̃ = mul!!(g_g̃(g), J', u)
    gnorm = rnorm(g̃)
    ĝ = scale!!(g_ĝ(g), inv(gnorm), g̃)
    Jg = mul!!(g_Jg(g), J, ĝ)
    jnorm = rnorm(Jg)
    cauchy = !iszero(gnorm) && isfinite(gnorm) && !iszero(jnorm) && isfinite(jnorm)
    nC = fnorm * ((gnorm / jnorm) / jnorm)
    if iszero(δ) && (newton || cauchy)
        n₀ = newton ? nN : nC
        δ = n₀ < lo ? 2 * lo : dogleg_clamp(n₀)
    end
    while true
        if newton && nN <= δ
            s = sN
            boundary = nN >= δ
        elseif !cauchy
            return x, Fx, δ, StepInfo{R}(STALLED, fnorm, zero(R), Int32(0), false, false)
        elseif !newton || nC >= δ
            # the Cauchy point, cut to the radius; with a Newton step, this branch has `nC ≥ δ`
            s = scale!!(g_s(g), -min(nC, δ), ĝ)
            boundary = nC >= δ
        else
            # the point of norm `δ` on the segment from the Cauchy point `δ a` to the Newton step,
            # in units of `δ` and along the unit `b̂` from the one to the other: the root `t ≥ 0`
            # of `‖a + t b̂‖ = 1`, where `‖a‖ < 1`; the root lies on the segment because the
            # Newton step is longer than `δ`. The Newton step minimises the model, so
            # `Re⟨a, b̂⟩ ≥ 0` (Nocedal and Wright, Lemma 4.2), and this form of the root does not
            # cancel. No norm of the Newton step in units of `δ` occurs.
            a = scale!!(g_g̃(g), -(nC / δ), ĝ)
            b = axpby!!(g_Jg(g), one(R), sN, nC, ĝ)
            b̂ = scale!!(g_Jg(g), inv(rnorm(b)), b)
            ab = rdot(a, b̂)
            c = 1 - norm2(a)
            ab2 = ab * ab
            t = c / (ab + sqrt(ab2 + c))
            s = axpby!!(g_s(g), δ, a, δ * t, b̂)
            boundary = true
        end
        Js = mul!!(g_Jg(g), J, s)
        Flin = axpby!!(g_Ft(g), one(R), Fx, one(R), Js)
        pred = fnorm - rnorm(Flin)
        x₊ = update_iterate!!(g_xt(g), x, one(R), s)
        F₊ = residual!!(g_Ft(g), prob, x₊, p)
        fnorm₊ = rnorm(F₊)
        ared = fnorm - fnorm₊
        # a non-finite `F₊` gives an `ared` of `-Inf` or `NaN`, which fails the test
        if pred > 0 && ared >= R(1e-4) * pred
            # the radius or the Cauchy point, not a root, sets the norm of any other step
            limited = !(newton && nN <= δ)
            ρ = ared / pred
            if ρ < R(0.1)
                δ = nN < δ ? max(nN, lo) : dogleg_shrink(δ)
            elseif ρ > R(0.75) && boundary
                δ = dogleg_grow(δ)
            end
            stepnorm = rnorm(s)
            x = copy!!(x, x₊)
            Fx = copy!!(Fx, F₊)
            return x, Fx, δ,
            StepInfo{R}(SUCCESS, fnorm₊, stepnorm, Int32(0), false, limited)
        end
        if δ == lo
            return x, Fx, δ,
            StepInfo{R}(LINESEARCH_FAILED, fnorm, zero(R), Int32(1), false, false)
        end
        δ = dogleg_shrink(δ)
    end
end
