"""
    Steihaug(; radius = nothing)

The trust-region globalization of [`Newton`](@ref) by truncated conjugate gradients (Steihaug
1983, doi:10.1137/0720042): `Newton(; linsolve = MINRESQLP(), globalization = Steihaug())`. It solves the symmetric
system of a [`StationarityProblem`](@ref) with `E == U`, and needs the linear method
[`MINRESQLP`](@ref), whose `maxiter` caps the conjugate-gradient iterations of a step and whose
forcing term sets their relative tolerance ``\\eta_k``; the conjugate gradients take the place of
the MINRES-QLP solve. Any other linear method raises an `ArgumentError` in the constructor of
`Newton`, and so does a factor reuse.

The step minimises the quadratic model ``q(d) = F^H d + \\tfrac12 d^H J d`` of the action `S`
inside the radius ``δ``, where ``F = (\\nabla S)_E + b`` and ``J`` is the Hessian: conjugate
gradients on ``J d = -F`` from ``d = 0`` with one Hessian–vector product per iteration, which stop
at ``\\|F + J d\\| \\le \\eta_k \\|F\\|`` inside the radius, the interior step; on a direction
``p`` of negative curvature, ``p^H J p \\le 0``, they go along ``p`` to ``\\|d\\| = δ``; and an
iterate that would leave the radius stops where its segment meets ``\\|d\\| = δ``. A trial is
accepted, and ``δ`` updated, by the rule of [`DogLeg`](@ref), with unsquared norms:
``\\mathrm{pred} = \\|F\\| - \\|F + J d\\|`` must be positive and
``\\mathrm{ared} = \\|F\\| - \\|F(x + d)\\| \\ge 10^{-4} \\mathrm{pred}``; a rejected trial shrinks
``δ`` to ``\\max(δ/4, 10^{-6})`` and runs the conjugate gradients again, unless it was an interior
step shorter than the new radius, which is tried again at one evaluation of `F`. A rejection at
``δ = 10^{-6}`` returns `LINESEARCH_FAILED` with the iterate kept. After an accepted step with
ratio ``ρ = \\mathrm{ared} / \\mathrm{pred}``, ``ρ < 0.1`` sets ``δ`` to
``\\max(\\|d\\|, 10^{-6})`` for an interior step and to ``\\max(δ/4, 10^{-6})`` otherwise, and
``ρ > 0.75`` on the boundary sets it to ``\\min(4δ, 10^{10})``. An accepted step other than an
interior step sets `limited` in its [`StepInfo`](@ref).

The merit is ``\\|F\\|``, never `S`, so a step toward a saddle point of `S` is accepted where it
reduces ``\\|F\\|``. A step that leaves along the first direction ``p = -F`` because it has negative
curvature predicts an increase of ``\\|F\\|``: ``\\|F - t J F\\|`` grows with `t > 0` where
``F^H J F < 0``, so that trial is rejected at every radius, and the step returns
`LINESEARCH_FAILED` after the radius falls to ``10^{-6}``.

`radius` is the initial radius, positive and finite, stored and clamped as `DogLeg` does it. With
`nothing`, the first step starts the radius at the length ``\\|F\\|^3 / (F^H J F)`` of the Cauchy
step along ``-F``, or at ``\\|F\\|`` where ``F^H J F \\le 0``, clamped to ``[10^{-6}, 10^{10}]``
with a length below ``10^{-6}`` starting it at ``2 \\cdot 10^{-6}``. A product that is not finite
ends the step with `NONFINITE`, and the iterate stays. At ``F(x) = 0`` the step returns `SUCCESS`
with the iterate and the radius kept, and takes no product.
"""
struct Steihaug{R <: Real, D <: Union{Nothing, R}}
    radius::D
    # `adapt` rebuilds a `Steihaug` through this constructor, inside a device kernel too. A kernel
    # cannot format a message, so the message is constant.
    function Steihaug{R}(radius::Union{Nothing, Real}) where {R <: Real}
        if radius !== nothing
            radius > 0 && isfinite(radius) ||
                throw(ArgumentError("the radius of Steihaug must be positive and finite"))
        end
        r = radius === nothing ? nothing : dogleg_clamp(R(radius))
        new{R, typeof(r)}(r)
    end
end

function Steihaug(; radius::Union{Nothing, Real} = nothing)
    radius === nothing ? Steihaug{Float64}(nothing) :
    Steihaug{float(typeof(radius))}(radius)
end

Adapt.adapt_structure(::ToReal{R}, st::Steihaug) where {R} = Steihaug{R}(st.radius)

@inline function steihaug_initial_radius(st::Steihaug, ::Type{R}) where {R}
    return st.radius === nothing ? zero(R) : R(st.radius)
end

# The workspace of a Steihaug step: the step `d`, the model residual `r = F + J d`, the direction
# `pv` and its product `Jp`, and the trial iterate and its residual.
@inline function init_globalization(::Steihaug, ::MINRESQLP, x, Fx)
    return (; d = buffer(x), r = buffer(Fx), pv = buffer(x), Jp = buffer(Fx),
        xt = buffer(x), Ft = buffer(Fx))
end

# What one run of the truncated conjugate gradients reports: the radius it ran with, which the
# first run of a solve without a radius sets, whether the step is interior, whether it met the
# tolerance there, whether every product was finite, and its iterations, which are its products.
struct SteihaugCG{R <: Real}
    radius::R
    interior::Bool
    converged::Bool
    finite::Bool
    iterations::Int32
end

# The step `t ≥ 0` along `p` from `d`, `‖d‖ < δ`, to `‖d + t p‖ = δ`. In units of `δ` along the
# unit direction `p̂ = p / ‖p‖`, with `a = d / δ`, it is the root `s` of `‖a + s p̂‖ = 1`:
# `s = c / (ab + √(ab² + c))` with `c = 1 - ‖a‖²` and `ab = ⟨a, p̂⟩` where `ab ≥ 0`, which does not
# cancel, and `s = √(ab² + c) - ab` otherwise; then `t = δ s / ‖p‖`.
@inline function boundary_step(d, p, δ::R) where {R}
    pn = R(rnorm(p))
    c = max(one(R) - (R(rnorm(d)) / δ)^2, zero(R))
    ab = R(rdot(d, p)) / (δ * pn)
    root = sqrt(ab * ab + c)
    s = ab >= zero(R) ? c / (ab + root) : root - ab
    return δ * s / pn
end

# The truncated conjugate gradients of Steihaug (1983) on `J d = -F` in the workspace `g`, with the
# product `A(y, v)`, which writes `J v` into `y`, from `d = 0`, to the absolute tolerance `atol` on
# `‖F + J d‖`, for at most `maxiter` iterations, inside the radius `δ`; `δ = 0` sets the radius from
# the first direction, as the docstring of `Steihaug` says. `g.d` holds the step and `g.r` its model
# residual `F + J d` on return.
@inline function steihaug_cg!(g, A, Fx, fnorm::R, atol::R, maxiter::Int32, δ::R) where {R}
    d = fill!(g.d, zero(eltype(g.d)))
    r = copy!!(g.r, Fx)
    p = map!!(-, g.pv, Fx)
    rr = fnorm * fnorm
    k = Int32(0)
    while k < maxiter
        k += one(Int32)
        Jp = A(g.Jp, p)
        κ = R(rdot(p, Jp))
        isfinite(κ) || return SteihaugCG{R}(δ, true, false, false, k)
        if iszero(δ)
            # the length of the Cauchy step `rr / κ ‖p‖` along `p = -F`, or `‖F‖` without one
            n₀ = κ > zero(R) ? (rr / κ) * fnorm : fnorm
            lo = dogleg_lo(R)
            δ = n₀ < lo ? 2 * lo : dogleg_clamp(n₀)
        end
        α = rr / κ
        # on negative curvature, and where the next iterate leaves the radius, the step ends on
        # the boundary along `p`
        if κ <= zero(R) ||
           R(rnorm(d)) ^ 2 + α * (2 * R(rdot(d, p)) + α * R(norm2(p))) >= δ^2
            t = boundary_step(d, p, δ)
            d .+= t .* p
            r .+= t .* Jp
            return SteihaugCG{R}(δ, false, false, true, k)
        end
        d .+= α .* p
        r .+= α .* Jp
        rr₊ = R(norm2(r))
        sqrt(rr₊) <= atol && return SteihaugCG{R}(δ, true, true, true, k)
        β = rr₊ / rr
        p .= β .* p .- r
        rr = rr₊
    end
    return SteihaugCG{R}(δ, true, false, true, k)
end

# The trial loop of `Steihaug` at `(x, Fx)` with the product `A`, the relative tolerance `η` of the
# conjugate gradients, their cap `maxiter` and the radius `δ` (`0` when unset). Returns `(x₊, F₊,
# δ₊, info, linres, iterations)`: the model residual `‖F + J d‖` of the last trial and the
# conjugate-gradient iterations of the step, which a `Newton` with `MINRESQLP` records for the
# forcing term of its next step.
@inline function steihaug_step!!(g, prob, x, Fx, p, A, η::R, maxiter::Int32, δ::R) where {R}
    lo = dogleg_lo(R)
    fnorm = R(rnorm(Fx))
    if iszero(fnorm)
        info = StepInfo{R}(SUCCESS, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, δ, info, zero(R), Int32(0)
    end
    iterations = Int32(0)
    cg = SteihaugCG{R}(δ, false, false, true, Int32(0))
    dnorm = zero(R)
    model = fnorm
    run = true
    while true
        if run
            cg = steihaug_cg!(g, A, Fx, fnorm, η * fnorm, maxiter, δ)
            iterations += cg.iterations
            δ = cg.radius
            if !cg.finite
                info = StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
                return x, Fx, δ, info, R(NaN), iterations
            end
            dnorm = R(rnorm(g.d))
            model = R(rnorm(g.r))
        end
        interior = cg.interior && cg.converged
        # an interior step that misses the tolerance at the cap is a step-rule failure
        failures = cg.interior && !cg.converged ? one(Int32) : Int32(0)
        pred = fnorm - model
        x₊ = update_iterate!!(g.xt, x, one(R), g.d)
        F₊ = primal_residual!!(g.Ft, prob, x₊, p)
        fnorm₊ = R(rnorm(F₊))
        ared = fnorm - fnorm₊
        # a non-finite `F₊` gives an `ared` of `-Inf` or `NaN`, which fails the test
        if pred > 0 && ared >= R(1e-4) * pred
            ρ = ared / pred
            if ρ < R(0.1)
                δ = interior ? max(dnorm, lo) : dogleg_shrink(δ)
            elseif ρ > R(0.75) && !cg.interior
                δ = dogleg_grow(δ)
            end
            x = copy!!(x, x₊)
            Fx = copy!!(Fx, F₊)
            info = StepInfo{R}(SUCCESS, fnorm₊, dnorm, failures, false, !interior)
            return x, Fx, δ, info, model, iterations
        end
        if δ == lo
            info = StepInfo{R}(
                LINESEARCH_FAILED, fnorm, zero(R), failures + one(Int32), false, false)
            return x, Fx, δ, info, model, iterations
        end
        δ = dogleg_shrink(δ)
        # an interior step shorter than the new radius is the step of that radius too
        run = !(interior && dnorm < δ)
    end
end
