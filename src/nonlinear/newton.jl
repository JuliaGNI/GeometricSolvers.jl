"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing)

Newton's method for [`NonlinearProblem`](@ref) and [`StationarityProblem`](@ref), with linear
solver `linsolve` ([`LUFactorization`](@ref), [`LDLTFactorization`](@ref),
[`QRFactorization`](@ref) or [`SVDFactorization`](@ref)), globalization `globalization`
([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref), [`StrongWolfe`](@ref), the
trust region [`DogLeg`](@ref) or the damped step [`LevenbergMarquardt`](@ref)), Jacobian selector
`jacobian`, iterative refinement `refinement`
([`NoRefinement`](@ref)), diagonal regularisation `regularization`, and factor-reuse period
`reuse`.

# Fields

- `linsolve::L`: the linear-solver method.
- `globalization::G`: the globalization method. A line search takes its merit at `α = 0` from
  the state, where [`start!!`](@ref) sets it to `norm2(F(x₀))` and each [`step!!`](@ref) to the
  merit of the residual it returns. A `step!!` with a line search therefore takes only the
  `(x, Fx)` that the last `start!!` or `step!!` on its state returned, as in [`solve!!`](@ref).
  A line-search step on a state that no `start!!` saw, or that `solve!!` returned, returns
  `NONFINITE` and keeps its iterate, and so does a [`LevenbergMarquardt`](@ref) step, whose
  damping parameter that state holds as `NaN`.
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
  step, from a fresh factor. A [`DogLeg`](@ref) or [`LevenbergMarquardt`](@ref) globalization takes
  no reuse: its model needs the Jacobian of the current iterate, and the constructor raises an
  `ArgumentError`.
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
            globalization isa LevenbergMarquardt &&
                throw(ArgumentError("Newton with a LevenbergMarquardt globalization takes no reuse, got reuse = $reuse"))
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

@inline _has_analytic_jacobian(prob::NonlinearProblem) = prob.J !== nothing
@inline _has_analytic_jacobian(::AbstractNonlinearProblem) = false

@inline function jacobian_matrix_jvp(g, x, prob)
    return jacobian_matrix_jvp(g, x)
end
@inline function jacobian_matrix_jvp(::LineSearch, x, prob)
    P = jacobian_prototype(prob)
    return JacobianMatrixJVP(
        P === nothing ? buffer(x, eltype(x), length(x), length(x)) : similar(P, eltype(x))
    )
end
@inline jacobian_matrix_jvp(::Static, x, prob) = nothing

@inline function newton_ad_preps(m::Newton, prob::AbstractNonlinearProblem, Fx, x, p)
    has_J = _has_analytic_jacobian(prob)
    if m.jacobian === nothing && !has_J
        prep = prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
        return prep, prep
    elseif m.jacobian isa AbstractADType && !has_J
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
        jvpprep = if has_J
            Fx === nothing ? nothing : jacobian_matrix_jvp(m.globalization, x, prob)
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

@inline function empty_static_factor(ls::LDLTFactorization{P}, ::StaticVector{
        N, T}) where {P, N, T}
    TF = factor_eltype(ls, T)
    z = zeros(SMatrix{N, N, TF})
    piv = SVector{N, Int32}(ntuple(i -> Int32(i), Val(N)))
    return StaticLDLT{N, TF, N * N}(z, piv, Int32(0), Int32(0), true)
end

@inline function empty_static_factor(ls::LDLTFactorization{P}, ::T) where {P, T <: Number}
    TF = factor_eltype(ls, T)
    z = SMatrix{1, 1, TF, 1}((zero(TF),))
    piv = SVector{1, Int32}((Int32(1),))
    return StaticLDLT{1, TF, 1}(z, piv, Int32(0), Int32(0), true)
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

@inline function empty_static_factor(ls::SVDFactorization{P}, ::StaticVector{
        N, T}) where {P, N, T}
    TF = factor_eltype(ls, T)
    z = zeros(SMatrix{N, N, TF})
    σ = zeros(SVector{N, real(TF)})
    return StaticSVD{N, TF, real(TF), N * N}(z, z, σ, Int32(0), Int32(0), Int32(0))
end

@inline function empty_static_factor(ls::SVDFactorization{P}, ::T) where {P, T <: Number}
    TF = factor_eltype(ls, T)
    z = SMatrix{1, 1, TF, 1}((zero(TF),))
    σ = SVector{1, real(TF)}((zero(real(TF)),))
    return StaticSVD{1, TF, real(TF), 1}(z, z, σ, Int32(0), Int32(0), Int32(0))
end

"""
    jacobian_pattern(prob, jacprep)

Return the Jacobian sparsity prototype for `prob` and `jacprep`: `prob.jacobian_prototype` for a
[`NonlinearProblem`](@ref), and `nothing` for any other problem and for `nothing`.
"""
@inline jacobian_pattern(prob::NonlinearProblem, jacprep) = prob.jacobian_prototype
@inline jacobian_pattern(::AbstractNonlinearProblem, jacprep) = nothing
@inline jacobian_pattern(::Nothing, jacprep) = nothing

@inline function init_factor(
        ls::LinearMethod, prob, x::Union{StaticVector, Number}, p, jacprep)
    return empty_static_factor(ls, x)
end

@inline function init_factor(ls::LinearMethod, prob, x::AbstractVector, p, jacprep)
    return init_linear(ls, jacobian_pattern(prob, jacprep), x)
end

@inline function init_newton_J(m::Newton, fac, x)
    same_eltype = factor_eltype(m.linsolve, eltype(x)) === eltype(x)
    if !(m.globalization isa Union{DogLeg, LevenbergMarquardt}) && same_eltype
        return fac.factors
    end
    if fac.factors isa SparseArrays.SparseMatrixCSC
        return similar(fac.factors, eltype(x))
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

# The rule state of `Newton` with `LevenbergMarquardt`: the current damping parameter `μ`,
# initialised to `NaN` in `init_state` and after `finish!!` so that a raw `step!!` outside
# `start!!` fails closed with `NONFINITE`, and set to `R(lm.μ₀)` by `start!!`.
struct DampingParameter{R}
    μ::R
end

@inline newton_rule(m::Newton, x) = newton_rule(m.globalization, m.reuse, real(eltype(x)))
@inline newton_rule(globalization, ::Nothing, ::Type{R}) where {R} = nothing
@inline newton_rule(globalization, ::Int32, ::Type{R}) where {R} = FactorAge(Int32(0), false)
@inline function newton_rule(dl::DogLeg, ::Nothing, ::Type{R}) where {R}
    δ₀ = dogleg_initial_radius(dl, R)
    return TrustRadius(δ₀, δ₀)
end
@inline function newton_rule(::LevenbergMarquardt, ::Nothing, ::Type{R}) where {R}
    return DampingParameter(R(NaN))
end

@inline reset_rule(::Nothing, _) = nothing
@inline reset_rule(::FactorAge, _) = FactorAge(Int32(0), false)
@inline reset_rule(r::TrustRadius, _) = TrustRadius(r.radius₀, r.radius₀)
@inline function reset_rule(::DampingParameter{R}, lm::LevenbergMarquardt) where {R}
    return DampingParameter(R(lm.μ₀))
end

# The range and spectrum diagnostics of the last `newton_factor` call of the solve, which
# `finish!!` copies into `SolverStatus`.
struct NewtonDiagnostics{R <: Real}
    rank::Int32
    nullnorm::R
    gradratio::R
end

@inline function NewtonDiagnostics(::Type{R}) where {R <: Real}
    return NewtonDiagnostics{R}(Int32(-1), R(NaN), R(NaN))
end

# The state of `Newton`: the AD preparations, the residual, direction, factor, Jacobian and
# globalization workspace, the rule state of its reuse, its `DogLeg` or its `LevenbergMarquardt`,
# the merit `norm2(Fx)` of the current residual for a line search (`nothing` for any other
# globalization), the norm `‖Fx‖` of the current residual, and the diagnostics of the last
# factorisation. For a value iterate the workspace slots are `nothing`, and so is the factor slot
# unless the factor is reused. The factor slot is the storage that each factorisation writes into;
# only a step with reuse stores the factor it returns, because only such a step reads it again.
# The norm is `NaN` on a state that no `start!!` saw or that `solve!!` returned.
struct NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG}
    jacprep::JP
    jvpprep::VP
    Fx::FX
    d::D
    fac::FAC
    J::JM
    g::G
    rule::RS
    merit::PH
    fnorm::R
    diag::DG
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag)
end

@inline function with_rule(s::NewtonState, rule)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, rule, s.merit, s.fnorm, s.diag)
end

# The merit and the norm of the residual that a step returns, or that a solve starts from.
@inline function with_merit(s::NewtonState, merit, fnorm)
    return NewtonState(s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, merit,
        oftype(s.fnorm, fnorm), s.diag)
end

@inline function with_diag(s::NewtonState, diag)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, diag)
end

@inline function init_state(m::Newton, prob::AbstractNonlinearProblem, x, p)
    m.globalization isa LevenbergMarquardt &&
        check_lm_iterate(m.globalization, m.linsolve, prob, x)
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    rule = newton_rule(m, x)
    merit = init_merit(m.globalization, x)
    R = real(eltype(x))
    diag = NewtonDiagnostics(R)
    if Fx === nothing
        fac = m.reuse === nothing ? nothing : init_factor(m.linsolve, prob, x, p, jacprep)
        return NewtonState(jacprep, jvpprep, nothing, nothing, fac, nothing, nothing, rule,
            merit, R(NaN), diag)
    else
        fac = init_factor(m.linsolve, prob, x, p, jacprep)
        return NewtonState(jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, x),
            init_globalization(m.globalization, m.linsolve, x), rule, merit, R(NaN), diag)
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    state = with_rule(state, reset_rule(state.rule, m.globalization))
    state = with_merit(state, start_merit(m.globalization, Fx), fnorm₀)
    return with_diag(state, NewtonDiagnostics(real(eltype(x))))
end

# The state that `solve!!` returns carries the merit and the norm of `init_state` again, so that
# a line-search step on it fails closed as on a fresh state, and `finish!!` copies the
# diagnostics of the last factorisation into the returned `SolverStatus`.
@inline function finish!!(
        state::NewtonState, m::Newton, prob, x, Fx, status::SolverStatus{R}) where {R}
    d = state.diag
    st = SolverStatus{R}(
        status.code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, d.rank, d.nullnorm, d.gradratio)
    state_out = with_merit(state, init_merit(m.globalization, x), NaN)
    if state.rule isa DampingParameter
        state_out = with_rule(state_out, DampingParameter(R(NaN)))
    end
    return x, st, state_out
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

# `‖Fx‖` with no extra host/device reduction over `Fx` where the state holds it: from the
# line-search merit `norm2(Fx)` when it is finite and in the normal range, and for any other
# globalization from the norm `fnorm` that the state carries. `rnorm(Fx)` otherwise: for a merit
# outside that range, and for a `NaN` norm on a state that no `start!!` saw or that `solve!!`
# returned.
@inline residual_norm(Fx, ::Nothing, fnorm) = isnan(fnorm) ? rnorm(Fx) : fnorm
@inline function residual_norm(Fx, merit::R, fnorm) where {R <: Real}
    isfinite(merit) && merit >= length(Fx) * floatmin(R) && return sqrt(merit)
    return rnorm(Fx)
end

# `‖Jᴴ Fx‖ / (‖J‖_F ‖Fx‖)` at the unregularised Jacobian `J`, before `regularize!!` or
# `factorize!!` touches `J`, and `0` when `‖J‖_F` or `‖Fx‖` is `0`. Where `Jᴴ Fx` overflows
# with both norms finite, it is computed again from `Fx` scaled by `2^{-e}`, with `2^e` the power
# of two of `‖Fx‖`, so that the scaled product is below `2‖J‖_F`; on an array iterate the
# scaled copy allocates. A mutable static `J` or `Fx` is read as an `SMatrix` or `SVector`, so
# that `Jᴴ Fx` allocates nothing.
@inline function grad_ratio(d, J, Fx, fnorm::R) where {R <: Real}
    Jmat = J isa StaticMatrix ? SMatrix(J) : J
    Fv = Fx isa StaticVector ? SVector(Fx) : Fx
    jnorm = rnorm(Jmat)
    (iszero(jnorm) || iszero(fnorm)) && return zero(R)
    gnorm = R(rnorm(mul!!(d, adjoint(Jmat), Fv)))
    isfinite(gnorm) && return gnorm / jnorm / fnorm
    (isfinite(jnorm) && isfinite(fnorm)) || return gnorm / jnorm / fnorm
    e = exponent(fnorm)
    gscaled = R(rnorm(mul!!(d, adjoint(Jmat), Fv .* ldexp(one(R), -e))))
    return gscaled / jnorm / ldexp(fnorm, -e)
end

# The Jacobian at `x`, regularised, its factor, and the diagnostics of both: the one
# factorisation of every `Newton` step.
@inline function newton_factor(m::Newton, state::NewtonState, prob, x, Fx, p)
    R = real(eltype(x))
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr = grad_ratio(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    fac = factorize!!(state.fac, m.linsolve, Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr)
end

# The step from the factor `fac`, fresh or reused: `SINGULAR` for a failed factor, else the
# globalized step along the refined direction, with the measured slope where the solve of `fac`
# is a truncated one. It returns the merit of the residual it returns, from the merit of `Fx`
# that the state carries.
@inline function newton_globalize(
        m::Newton, state::NewtonState, prob, x, Fx, p, fac, fresh::Bool)
    T = eltype(x)
    R = real(T)
    if fac.info > Int32(0)
        return x, Fx, StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false, false),
        state.merit
    end
    d₀ = ldiv!!(state.d, fac, Fx)
    d₁ = map!!(-, state.d, d₀)
    d = refine!!(state.d, m.refinement, prob, x, Fx, d₁, fac, p, state.jvpprep)
    sk = exact_solve(fac) ? newton_stepkind(m, T, fresh) : MeasuredSlope()
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk, state.merit)
    return x₊, F₊, info, merit
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    _, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    return x₊, F₊, with_diag(with_merit(state, merit, info.fnorm), diag), info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A failed factor or a Newton step
# whose norm is zero or not finite leaves the Cauchy step.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
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
    state₊ = with_diag(with_merit(state₊, state.merit, info.fnorm), diag)
    return x₊, F₊, state₊, info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    rule = state.rule
    fresh = rule.age == Int32(0) || rule.stalled
    _, fac, diag = if fresh
        newton_factor(m, state, prob, x, Fx, p)
    else
        (nothing, state.fac, state.diag)
    end
    age_fresh = m.reuse == Int32(1) ? Int32(0) : Int32(1)
    age₊ = fresh ? age_fresh :
           (rule.age + one(Int32) >= m.reuse ? Int32(0) : rule.age + one(Int32))
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, fresh)
    # A reused factor can give a direction that does not descend. A line search along it that
    # fails without a step is retried once, in the same step, from a fresh factor, and from the
    # same residual and merit.
    if !fresh && info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
        _, fac, diag = newton_factor(m, state, prob, x, Fx, p)
        age₊ = age_fresh
        x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    end
    state = with_diag(with_merit(with_factor(state, fac), merit, info.fnorm), diag)
    if info.code == SINGULAR
        return x₊, F₊, with_rule(state, FactorAge(Int32(0), false)), info
    end
    state₊ = with_rule(state, FactorAge(age₊, info.code == STALLED))
    return x₊, F₊, state₊, info
end

@inline function lm_svd_factor(state, m, ::Number, Jreg)
    return factorize!!(state.fac, m.linsolve, SMatrix{1, 1, typeof(Jreg)}((Jreg,)))
end
@inline lm_svd_factor(state, m, _, Jreg) = factorize!!(state.fac, m.linsolve, Jreg)

@inline function lm_newton_factor(
        m::Newton, state::NewtonState, prob, x, Fx, p, fnorm::R) where {R}
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr = grad_ratio(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    if lm_svd_path(m.linsolve, m.globalization.scaling)
        fac = lm_svd_factor(state, m, x, Jreg)
        rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
        return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr)
    end
    return Jreg, state.fac, NewtonDiagnostics{R}(Int32(-1), R(NaN), gr)
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {
        L, G <: LevenbergMarquardt, J, RF, Reg}
    R = real(eltype(x))
    fnorm = R(rnorm(Fx))
    μ = state.rule.μ
    if !isfinite(μ)
        info = StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state, info
    end
    # A root is a step of zero: it is answered before the Jacobian is read, so that a non-finite
    # `J` at a root is not what the step reports.
    if iszero(fnorm)
        info = StepInfo{R}(SUCCESS, zero(R), zero(R), Int32(0), false, false)
        return x, Fx, state, info
    end
    lm = m.globalization
    μ_min_eff = max(R(lm.μ_min), 4 * eps(R))
    Jreg, fac, diag = lm_newton_factor(m, state, prob, x, Fx, p, fnorm)
    state_d = with_diag(state, diag)
    j_ok = Jreg isa Number ? isfinite(Jreg) : all(isfinite, Jreg)
    if !j_ok
        info = StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state_d, info
    end
    use_svd = lm_svd_path(m.linsolve, lm.scaling)
    if use_svd && fac.info > 0
        info = StepInfo{R}(SINGULAR, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state_d, info
    end
    # The trial loop's workspace: the stored SVD factor, which needs no reduction, or the reduction
    # of `[J | -F]` that the stacked path does once per outer step.
    trial = lm_trial_workspace(m.linsolve, lm.scaling, state.d, state.g, fac, Jreg, Fx)
    failures = Int32(0)
    while true
        δ, step_ok = lm_trial(trial, μ)
        if step_ok
            Jδ = mul!!(lm_Jδ(state.g), Jreg, δ)
            pred = -2 * rdot(Fx, Jδ) - norm2(Jδ)
            if pred <= zero(R)
                Flin = axpby!!(lm_Ft(state.g), one(R), Fx, one(R), Jδ)
                flin_norm = rnorm(Flin)
                pred = (fnorm - flin_norm) * (fnorm + flin_norm)
            end
            x₊ = update_iterate!!(lm_xt(state.g), x, one(R), δ)
            F₊ = primal_residual!!(lm_Ft(state.g), prob, x₊, p)
            fnorm₊ = R(rnorm(F₊))
            if isfinite(fnorm₊) && pred > zero(R)
                ared = (fnorm - fnorm₊) * (fnorm + fnorm₊)
                ρ = ared / pred
                if ρ > R(lm.ρ_accept)
                    limited = μ > R(lm.μ₀)
                    μ_next = max(μ * R(lm.μ_decrease), μ_min_eff)
                    stepnorm = R(rnorm(δ))
                    x = copy!!(x, x₊)
                    Fx = copy!!(Fx, F₊)
                    state₊ = with_merit(state_d, state_d.merit, fnorm₊)
                    state₊ = with_rule(state₊, DampingParameter(μ_next))
                    info = StepInfo{R}(SUCCESS, fnorm₊, stepnorm, failures, false, limited)
                    return x, Fx, state₊, info
                end
            end
        end
        failures += Int32(1)
        μ = max(μ * R(lm.μ_increase), μ_min_eff)
        if !isfinite(μ) || μ > R(lm.μ_max)
            # The state keeps a finite damping: a `μ` that overflows is at the bound, and a
            # non-finite one would read as the un-started state on the stall that follows.
            state_fail = with_rule(state_d, DampingParameter(min(μ, floatmax(R))))
            info = StepInfo{R}(STALLED, fnorm, zero(R), failures, false, false)
            return x, Fx, state_fail, info
        end
    end
end
