"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing)

Newton's method for [`NonlinearProblem`](@ref), with linear solver `linsolve`
([`LUFactorization`](@ref) or [`QRFactorization`](@ref) in R1), globalization `globalization`
([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) or [`StrongWolfe`](@ref)), Jacobian
selector `jacobian`, iterative refinement `refinement` ([`NoRefinement`](@ref)), diagonal
regularisation `regularization`, and factor-reuse period `reuse`.

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
    r = reuse === nothing ? nothing :
        (reuse >= 1 ? Int32(reuse) :
         throw(ArgumentError("Newton reuse must be ≥ 1 or nothing, got $reuse")))
    return Newton(linsolve, globalization, jacobian, refinement, reg, r)
end

Adapt.@adapt_structure Newton

@inline function newton_jacprep(m::Newton, prob::NonlinearProblem, x, p)
    if m.jacobian === nothing
        return prob.J !== nothing ? nothing :
               prepare_ad(AutoForwardDiff(), prob, nothing, x, p)
    elseif m.jacobian isa AbstractADType
        return prepare_ad(m.jacobian, prob, nothing, x, p)
    else
        return JacobianFunction(m.jacobian)
    end
end

@inline function newton_jvpprep(m::Newton, prob::NonlinearProblem, x, p)
    if prob.J !== nothing
        return nothing
    elseif m.jacobian isa AbstractADType
        return prepare_ad(m.jacobian, prob, nothing, x, p)
    else
        return prepare_ad(AutoForwardDiff(), prob, nothing, x, p)
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

function init(
        m::Newton{L, G, J, RF, R, Nothing}, prob::NonlinearProblem, x, p = nothing) where {
        L, G, J, RF, R}
    jacprep = newton_jacprep(m, prob, x, p)
    jvpprep = newton_jvpprep(m, prob, x, p)
    return (; jacprep, jvpprep)
end

function init(
        m::Newton{L, G, J, RF, R, Int32}, prob::NonlinearProblem, x, p = nothing) where {
        L, G, J, RF, R}
    jacprep = newton_jacprep(m, prob, x, p)
    jvpprep = newton_jvpprep(m, prob, x, p)
    fac = empty_static_factor(m.linsolve, x)
    return (; jacprep, jvpprep, fac, age = Int32(0), stalled = false)
end

@inline function regularize_jacobian(J::SMatrix{N, N, T}, μ) where {N, T}
    iszero(μ) && return J
    shift = T(μ)
    return SMatrix{N, N, T, N * N}(ntuple(Val(N * N)) do k
        i = (k - 1) % N + 1
        j = (k - 1) ÷ N + 1
        @inbounds i == j ? J[k] + shift : J[k]
    end)
end

@inline function regularize_jacobian(J::T, μ) where {T <: Number}
    iszero(μ) && return J
    return J + T(μ)
end

@inline function newton_stepkind(m::Newton, ::Type{T}, fresh::Bool) where {T}
    if fresh && (m.jacobian === nothing || m.jacobian isa AbstractADType) &&
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
    Jmat = jacobian!!(nothing, state.jacprep, prob, x, p)
    Jreg = regularize_jacobian(Jmat, m.regularization)
    fac = factorize!!(nothing, m.linsolve, Jreg)
    if fac.info > Int32(0)
        return x, Fx, state, StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false)
    end
    d₀ = map!!(-, nothing, ldiv!!(nothing, fac, Fx))
    d = refine!!(nothing, m.refinement, prob, x, Fx, d₀, fac, p, state.jvpprep)
    sk = newton_stepkind(m, T, true)
    x₊, F₊, _, info = globalize!!(
        nothing, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk)
    return x₊, F₊, state, info
end

@inline function step!!(
        state, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    fresh = state.age == Int32(0) || state.stalled
    if fresh
        Jmat = jacobian!!(nothing, state.jacprep, prob, x, p)
        Jreg = regularize_jacobian(Jmat, m.regularization)
        fac = factorize!!(state.fac, m.linsolve, Jreg)
        age₊ = m.reuse == Int32(1) ? Int32(0) : Int32(1)
    else
        fac = state.fac
        age₊ = state.age + one(Int32) >= m.reuse ? Int32(0) : state.age + one(Int32)
    end
    if fac.info > Int32(0)
        state_sing = (; state.jacprep, state.jvpprep, fac, age = Int32(0), stalled = false)
        return x, Fx, state_sing,
        StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false)
    end
    d₀ = map!!(-, nothing, ldiv!!(nothing, fac, Fx))
    d = refine!!(nothing, m.refinement, prob, x, Fx, d₀, fac, p, state.jvpprep)
    sk = newton_stepkind(m, T, fresh)
    x₊, F₊, _, info = globalize!!(
        nothing, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk)
    stalled₊ = info.code == STALLED
    state₊ = (; state.jacprep, state.jvpprep, fac, age = age₊, stalled = stalled₊)
    return x₊, F₊, state₊, info
end
