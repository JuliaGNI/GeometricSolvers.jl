"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing)

Newton's method for [`NonlinearProblem`](@ref), with linear solver `linsolve`
([`LUFactorization`](@ref) or [`QRFactorization`](@ref) in R1), globalization `globalization`
([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref), [`StrongWolfe`](@ref) or the
trust region [`DogLeg`](@ref)), Jacobian selector `jacobian`, iterative refinement `refinement`
([`NoRefinement`](@ref)), diagonal regularisation `regularization`, and factor-reuse period
`reuse`.

# Fields

- `linsolve::L`: the linear-solver method.
- `globalization::G`: the globalization method.
- `jacobian::J`: `nothing` (use `prob.J` when present, or `AutoForwardDiff()` otherwise), an
  `AbstractADType`, or a custom Jacobian callable `J(x, p)`. Adapting `Newton` with
  [`ToReal`](@ref) (`adapt(ToReal{R}(), m)`) converts floating-point values captured inside a
  `jacobian` closure to `R`, while a function with no captured floats passes through unchanged.
- `refinement::RF`: the refinement setting ([`NoRefinement`](@ref)).
- `regularization::R`: non-negative finite diagonal shift ``μ ≥ 0`` added as ``J + μ I`` before
  factorisation when `μ > 0`.
- `reuse::K`: `nothing` (`K = Nothing`, factorise at every step) or `Int32(k)` (`K = Int32`,
  reuse the factor for up to `k` consecutive steps, refreshing early after a `STALLED` step).
  A line search along a reused factor that fails without a step is retried once, in the same
  step, from a fresh factor. A [`DogLeg`](@ref) globalization takes no reuse: its model needs the
  Jacobian of the current iterate, and the constructor raises an `ArgumentError`.
"""
struct Newton{L, G, J, RF, R <: Real, K <: Union{Nothing, Int32}}
    linsolve::L
    globalization::G
    jacobian::J
    refinement::RF
    regularization::R
    reuse::K
    function Newton{L, G, J, RF, R, K}(
            linsolve::L, globalization::G, jacobian::J,
            refinement::RF, regularization::R,
            reuse::K) where {
            L, G, J, RF, R <: Real, K <: Union{Nothing, Int32}}
        regularization >= zero(R) && isfinite(regularization) ||
            throw(ArgumentError("Newton regularization must be finite and ≥ 0, got $regularization"))
        if reuse !== nothing
            reuse >= Int32(1) ||
                throw(ArgumentError("Newton reuse must be ≥ 1 or nothing, got $reuse"))
            globalization isa DogLeg &&
                throw(ArgumentError("Newton with a DogLeg globalization takes no reuse, got reuse = $reuse"))
        end
        new{L, G, J, RF, R, K}(
            linsolve, globalization, jacobian, refinement, regularization, reuse)
    end
end

function Newton(
        linsolve::L, globalization::G, jacobian::J,
        refinement::RF, regularization::R,
        reuse::K) where {
        L, G, J, RF, R <: Real, K <: Union{Nothing, Int32}}
    return Newton{L, G, J, RF, R, K}(
        linsolve, globalization, jacobian, refinement, regularization, reuse)
end

function Newton(;
        linsolve = LUFactorization(),
        globalization = Static(),
        jacobian = nothing,
        refinement = NoRefinement(),
        regularization::Real = 0,
        reuse::Union{Nothing, Integer} = nothing)
    reg = float(regularization)
    r = reuse === nothing ? nothing : Int32(reuse)
    return Newton(linsolve, globalization, jacobian, refinement, reg, r)
end

Adapt.@adapt_structure Newton

@inline function newton_ad_preps(m::Newton, prob::NonlinearProblem, Fx, x, p)
    if m.jacobian === nothing && prob.J === nothing
        prep = prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
        return prep, prep
    elseif m.jacobian isa AbstractADType && prob.J === nothing
        prep = prepare_ad(m.jacobian, prob, Fx, x, p)
        return prep, prep
    else
        jacprep = if m.jacobian === nothing
            nothing
        elseif m.jacobian isa AbstractADType
            prepare_ad(m.jacobian, prob, Fx, x, p)
        else
            JacobianFunction(m.jacobian)
        end
        jvpprep = if prob.J !== nothing
            Fx === nothing ? nothing : jacobian_matrix_jvp(m.globalization, x)
        elseif m.jacobian isa AbstractADType
            prepare_ad(m.jacobian, prob, Fx, x, p)
        else
            prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
        end
        return jacprep, jvpprep
    end
end

@inline function empty_static_factor(ls::LUFactorization{P}, ::StaticVector{
        N, T}) where {P, N, T}
    TF = factor_eltype(ls, T)
    z = zeros(SMatrix{N, N, TF})
    piv = SVector{N, Int32}(ntuple(i -> Int32(i), Val(N)))
    return StaticLU{N, TF, N * N}(z, piv, Int32(0), Int32(0))
end

@inline function empty_static_factor(ls::LUFactorization{P}, ::T) where {P, T <: Number}
    TF = factor_eltype(ls, T)
    z = SMatrix{1, 1, TF, 1}((zero(TF),))
    piv = SVector{1, Int32}((Int32(1),))
    return StaticLU{1, TF, 1}(z, piv, Int32(0), Int32(0))
end

@inline function empty_static_factor(ls::QRFactorization{P}, ::StaticVector{
        N, T}) where {P, N, T}
    TF = factor_eltype(ls, T)
    z = zeros(SMatrix{N, N, TF})
    τ = zeros(SVector{N, TF})
    return StaticQR{N, TF, N * N}(z, τ, Int32(0), Int32(0))
end

@inline function empty_static_factor(ls::QRFactorization{P}, ::T) where {P, T <: Number}
    TF = factor_eltype(ls, T)
    z = SMatrix{1, 1, TF, 1}((zero(TF),))
    τ = SVector{1, TF}((zero(TF),))
    return StaticQR{1, TF, 1}(z, τ, Int32(0), Int32(0))
end

@inline function init_factor(
        ls::Union{LUFactorization, QRFactorization}, prob, x::Union{StaticVector, Number}, p)
    return empty_static_factor(ls, x)
end

@inline function init_factor(ls::LinearMethod, prob, x::AbstractVector, p)
    return init(ls, prob, x, p)
end

@inline function init_newton_J(m::Newton, fac, x)
    if !(m.globalization isa DogLeg) && factor_eltype(m.linsolve, eltype(x)) === eltype(x)
        return fac.factors
    end
    n = length(x)
    return buffer(x, eltype(x), n, n)
end

# The rule state of `Newton` with `reuse = k`: the age of the factor, and whether the last step
# stalled, which refreshes the factor early.
struct FactorAge
    age::Int32
    stalled::Bool
end

# The rule state of `Newton` with `DogLeg`: the radius, and the initial radius that each solve
# starts from, `0` when the first step sets it.
struct TrustRadius{R}
    radius::R
    radius₀::R
end

@inline newton_rule(m::Newton, x) = newton_rule(m.globalization, m.reuse, real(eltype(x)))
@inline newton_rule(globalization, ::Nothing, ::Type{R}) where {R} = nothing
@inline newton_rule(globalization, ::Int32, ::Type{R}) where {R} = FactorAge(Int32(0), false)
@inline function newton_rule(dl::DogLeg, ::Nothing, ::Type{R}) where {R}
    δ₀ = dogleg_initial_radius(dl, R)
    return TrustRadius(δ₀, δ₀)
end

@inline reset_rule(::Nothing) = nothing
@inline reset_rule(::FactorAge) = FactorAge(Int32(0), false)
@inline reset_rule(r::TrustRadius) = TrustRadius(r.radius₀, r.radius₀)

# The state of `Newton`: the AD preparations, the residual, direction, factor, Jacobian and
# globalization workspace, and the rule state of its reuse or its `DogLeg`. For a value iterate
# the workspace slots are `nothing`, and so is the factor slot unless the factor is reused. The
# factor slot is the storage that each factorisation writes into; only a step with reuse stores
# the factor it returns, because only such a step reads it again.
struct NewtonState{JP, VP, FX, D, FAC, JM, G, RS}
    jacprep::JP
    jvpprep::VP
    Fx::FX
    d::D
    fac::FAC
    J::JM
    g::G
    rule::RS
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule)
end

@inline function with_rule(s::NewtonState, rule)
    return NewtonState(s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, rule)
end

@inline function init_state(m::Newton, prob::NonlinearProblem, x, p)
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    rule = newton_rule(m, x)
    if Fx === nothing
        fac = m.reuse === nothing ? nothing : init_factor(m.linsolve, prob, x, p)
        return NewtonState(jacprep, jvpprep, nothing, nothing, fac, nothing, nothing, rule)
    else
        fac = init_factor(m.linsolve, prob, x, p)
        return NewtonState(jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, x),
            init_globalization(m.globalization, x), rule)
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    return with_rule(state, reset_rule(state.rule))
end

@inline function newton_stepkind(m::Newton, ::Type{T}, fresh::Bool) where {T}
    if fresh && (m.jacobian === nothing || m.jacobian isa AbstractADType) &&
       iszero(m.regularization) &&
       factor_eltype(m.linsolve, T) === T && m.refinement isa NoRefinement
        return ExactStep()
    else
        return MeasuredSlope()
    end
end

# The Jacobian at `x`, regularised, and its factor: the one factorisation of every `Newton` step.
@inline function newton_factor(m::Newton, state::NewtonState, prob, x, p)
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    Jreg = regularize!!(Jmat, m.regularization)
    return Jreg, factorize!!(state.fac, m.linsolve, Jreg)
end

# The step from the factor `fac`, fresh or reused: `SINGULAR` for a singular factor, else the
# globalized step along the refined direction, with the measured slope where the solve of `fac`
# is a truncated one.
@inline function newton_globalize(
        m::Newton, state::NewtonState, prob, x, Fx, p, fac, fresh::Bool)
    T = eltype(x)
    R = real(T)
    if fac.info > Int32(0)
        return x, Fx, StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    d₀ = ldiv!!(state.d, fac, Fx)
    d₁ = map!!(-, state.d, d₀)
    d = refine!!(state.d, m.refinement, prob, x, Fx, d₁, fac, p, state.jvpprep)
    sk = exact_solve(fac) ? newton_stepkind(m, T, fresh) : MeasuredSlope()
    x₊, F₊, _, info = globalize!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk)
    return x₊, F₊, info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    _, fac = newton_factor(m, state, prob, x, p)
    x₊, F₊, info = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    return x₊, F₊, state, info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A singular factor or a Newton step
# whose norm is zero or not finite leaves the Cauchy step.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    Jreg, fac = newton_factor(m, state, prob, x, p)
    if fac.info == Int32(0)
        d₀ = ldiv!!(state.d, fac, Fx)
        d₁ = map!!(-, state.d, d₀)
        d = refine!!(state.d, m.refinement, prob, x, Fx, d₁, fac, p, state.jvpprep)
        nd = rnorm(d)
        newton = isfinite(nd) && !iszero(nd)
    else
        d = Fx
        newton = false
    end
    x₊, F₊, radius, info = dogleg_step(
        state.g, prob, x, Fx, p, Jreg, d, newton, state.rule.radius)
    state₊ = with_rule(state, TrustRadius(radius, state.rule.radius₀))
    return x₊, F₊, state₊, info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    rule = state.rule
    fresh = rule.age == Int32(0) || rule.stalled
    fac = fresh ? newton_factor(m, state, prob, x, p)[2] : state.fac
    age_fresh = m.reuse == Int32(1) ? Int32(0) : Int32(1)
    age₊ = fresh ? age_fresh :
           (rule.age + one(Int32) >= m.reuse ? Int32(0) : rule.age + one(Int32))
    x₊, F₊, info = newton_globalize(m, state, prob, x, Fx, p, fac, fresh)
    # A reused factor can give a direction that does not descend. A line search along it that
    # fails without a step is retried once, in the same step, from a fresh factor.
    if !fresh && info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
        _, fac = newton_factor(m, state, prob, x, p)
        age₊ = age_fresh
        x₊, F₊, info = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    end
    if info.code == SINGULAR
        return x₊, F₊, with_rule(with_factor(state, fac), FactorAge(Int32(0), false)), info
    end
    state₊ = with_rule(with_factor(state, fac), FactorAge(age₊, info.code == STALLED))
    return x₊, F₊, state₊, info
end
