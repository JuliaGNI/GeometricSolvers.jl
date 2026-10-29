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
            Fx === nothing ? nothing :
            JacobianMatrixJVP(buffer(x, eltype(x), length(x), length(x)))
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

@inline function init(
        m::Newton{L, G, J, RF, R, Nothing}, prob::NonlinearProblem, x, p = nothing) where {
        L, G, J, RF, R}
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    if Fx === nothing
        return (; jacprep, jvpprep)
    else
        fac = init_factor(m.linsolve, prob, x, p)
        return (; jacprep, jvpprep, Fx, d = buffer(x), fac,
            J = init_newton_J(m, fac, x), g = init_globalization(m.globalization, x))
    end
end

@inline function init(
        m::Newton{L, G, J, RF, R, Int32}, prob::NonlinearProblem, x, p = nothing) where {
        L, G, J, RF, R}
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    fac = init_factor(m.linsolve, prob, x, p)
    if Fx === nothing
        return (; jacprep, jvpprep, fac, age = Int32(0), stalled = false)
    else
        return (; jacprep, jvpprep, fac, age = Int32(0), stalled = false,
            Fx, d = buffer(x), J = init_newton_J(m, fac, x),
            g = init_globalization(m.globalization, x))
    end
end

@inline function init(
        m::Newton{L, G, J, RF, R, Nothing}, prob::NonlinearProblem, x, p = nothing) where {
        L, G <: DogLeg, J, RF, R}
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    radius = dogleg_initial_radius(m.globalization, real(eltype(x)))
    if Fx === nothing
        return (; jacprep, jvpprep, radius)
    else
        fac = init_factor(m.linsolve, prob, x, p)
        return (; jacprep, jvpprep, radius, radius₀ = radius, Fx, d = buffer(x),
            fac, J = init_newton_J(m, fac, x),
            g = init_globalization(m.globalization, x))
    end
end

@inline state_J(state) = hasproperty(state, :J) ? state.J : nothing
@inline state_fac(state) = hasproperty(state, :fac) ? state.fac : nothing

@inline function newton_stepkind(m::Newton, ::Type{T}, fresh::Bool) where {T}
    if fresh && (m.jacobian === nothing || m.jacobian isa AbstractADType) &&
       iszero(m.regularization) &&
       factor_eltype(m.linsolve, T) === T && m.refinement isa NoRefinement
        return ExactStep()
    else
        return MeasuredSlope()
    end
end

@inline function step!!(
        state, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    Jmat = jacobian!!(state_J(state), state.jacprep, prob, x, p)
    Jreg = regularize!!(Jmat, m.regularization)
    fac = factorize!!(state_fac(state), m.linsolve, Jreg)
    state₊ = if hasproperty(state, :fac) && !(fac === state.fac)
        (; state..., fac)
    else
        state
    end
    if fac.info > Int32(0)
        return x, Fx, state₊,
        StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    d₀ = ldiv!!(state_d(state), fac, Fx)
    d₁ = map!!(-, state_d(state), d₀)
    d = refine!!(state_d(state), m.refinement, prob, x, Fx, d₁, fac, p, state.jvpprep)
    sk = newton_stepkind(m, T, true)
    x₊, F₊, _, info = globalize!!(
        state_g(state), m.globalization, prob, x, Fx, d, p, state.jvpprep, sk)
    return x₊, F₊, state₊, info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A singular factor or a Newton step
# whose norm is not finite leaves the Cauchy step.
@inline function step!!(
        state, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    Jmat = jacobian!!(state_J(state), state.jacprep, prob, x, p)
    Jreg = regularize!!(Jmat, m.regularization)
    fac = factorize!!(state_fac(state), m.linsolve, Jreg)
    if fac.info == Int32(0)
        d₀ = ldiv!!(state_d(state), fac, Fx)
        d₁ = map!!(-, state_d(state), d₀)
        d = refine!!(state_d(state), m.refinement, prob, x, Fx, d₁, fac, p, state.jvpprep)
        newton = isfinite(rnorm(d))
    else
        d = Fx
        newton = false
    end
    x₊, F₊, radius, info = dogleg_step(
        state_g(state), prob, x, Fx, p, Jreg, d, newton, state.radius)
    state₊ = if hasproperty(state, :fac)
        (; state..., fac, radius)
    else
        (; state.jacprep, state.jvpprep, radius)
    end
    return x₊, F₊, state₊, info
end

@inline function newton_factor(m::Newton, state, prob, x, p)
    Jmat = jacobian!!(state_J(state), state.jacprep, prob, x, p)
    Jreg = regularize!!(Jmat, m.regularization)
    return factorize!!(state_fac(state), m.linsolve, Jreg)
end

# The step from the factor `fac`, fresh or reused: `SINGULAR` for a singular factor, else the
# globalized step along the refined direction.
@inline function newton_globalize(m::Newton, state, prob, x, Fx, p, fac, fresh::Bool)
    T = eltype(x)
    R = real(T)
    if fac.info > Int32(0)
        return x, Fx, StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    d₀ = ldiv!!(state_d(state), fac, Fx)
    d₁ = map!!(-, state_d(state), d₀)
    d = refine!!(state_d(state), m.refinement, prob, x, Fx, d₁, fac, p, state.jvpprep)
    sk = newton_stepkind(m, T, fresh)
    x₊, F₊, _, info = globalize!!(
        state_g(state), m.globalization, prob, x, Fx, d, p, state.jvpprep, sk)
    return x₊, F₊, info
end

@inline function step!!(
        state, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    fresh = state.age == Int32(0) || state.stalled
    fac = fresh ? newton_factor(m, state, prob, x, p) : state.fac
    age_fresh = m.reuse == Int32(1) ? Int32(0) : Int32(1)
    age₊ = fresh ? age_fresh :
           (state.age + one(Int32) >= m.reuse ? Int32(0) : state.age + one(Int32))
    x₊, F₊, info = newton_globalize(m, state, prob, x, Fx, p, fac, fresh)
    # A reused factor can give a direction that does not descend. A line search along it that
    # fails without a step is retried once, in the same step, from a fresh factor.
    if !fresh && info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
        fac = newton_factor(m, state, prob, x, p)
        age₊ = age_fresh
        x₊, F₊, info = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    end
    if info.code == SINGULAR
        state_sing = (; state..., fac, age = Int32(0), stalled = false)
        return x₊, F₊, state_sing, info
    end
    stalled₊ = info.code == STALLED
    state₊ = (; state..., fac, age = age₊, stalled = stalled₊)
    return x₊, F₊, state₊, info
end
