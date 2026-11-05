"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing)

Newton's method for [`NonlinearProblem`](@ref), with linear solver `linsolve`
([`LUFactorization`](@ref), [`QRFactorization`](@ref) or [`SVDFactorization`](@ref)),
globalization `globalization`
([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref), [`StrongWolfe`](@ref) or the
trust region [`DogLeg`](@ref)), Jacobian selector `jacobian`, iterative refinement `refinement`
([`NoRefinement`](@ref)), diagonal regularisation `regularization`, and factor-reuse period
`reuse`.

# Fields

- `linsolve::L`: the linear-solver method.
- `globalization::G`: the globalization method. A line search takes its merit at `α = 0` from
  the state, where [`start!!`](@ref) sets it to `norm2(F(x₀))` and each [`step!!`](@ref) to the
  merit of the residual it returns. A `step!!` with a line search therefore takes only the
  `(x, Fx)` that the last `start!!` or `step!!` on its state returned, as in [`solve!!`](@ref).
  A line-search step on a state that no `start!!` saw, or that `solve!!` returned, returns
  `NONFINITE` and keeps its iterate.
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

@inline function init_factor(
        ls::Union{LUFactorization, QRFactorization, SVDFactorization},
        prob, x::Union{StaticVector, Number}, p)
    return empty_static_factor(ls, x)
end

@inline function init_factor(ls::LinearMethod, prob, x::AbstractVector, p)
    return dense_factor(ls, prob, x, p)
end

@inline function init_newton_J(m::Newton, fac, x)
    same_eltype = factor_eltype(m.linsolve, eltype(x)) === eltype(x)
    if !(m.globalization isa Union{DogLeg, LevenbergMarquardt}) && same_eltype
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
@inline function newton_rule(::LevenbergMarquardt, x)
    return DampingParameter(real(eltype(x))(NaN))
end
@inline function newton_rule(::LevenbergMarquardt, ::Nothing, ::Type{R}) where {R}
    return DampingParameter(R(NaN))
end

@inline reset_rule(::Nothing) = nothing
@inline reset_rule(::FactorAge) = FactorAge(Int32(0), false)
@inline reset_rule(r::TrustRadius) = TrustRadius(r.radius₀, r.radius₀)
@inline reset_rule(r, _) = reset_rule(r)
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
# globalization), and the diagnostics of the last factorisation. For a value iterate the
# workspace slots are `nothing`, and so is the factor slot unless the factor is reused. The
# factor slot is the storage that each factorisation writes into; only a step with reuse stores
# the factor it returns, because only such a step reads it again.
struct NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, DG}
    jacprep::JP
    jvpprep::VP
    Fx::FX
    d::D
    fac::FAC
    J::JM
    g::G
    rule::RS
    merit::PH
    diag::DG
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule, s.merit, s.diag)
end

@inline function with_rule(s::NewtonState, rule)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, rule, s.merit, s.diag)
end

@inline function with_merit(s::NewtonState, merit)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, merit, s.diag)
end

@inline function with_diag(s::NewtonState, diag)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, diag)
end

@inline function init_state(m::Newton, prob::NonlinearProblem, x, p)
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    rule = newton_rule(m, x)
    merit = init_merit(m.globalization, x)
    diag = NewtonDiagnostics(real(eltype(x)))
    if Fx === nothing
        fac = m.reuse === nothing ? nothing : init_factor(m.linsolve, prob, x, p)
        return NewtonState(
            jacprep, jvpprep, nothing, nothing, fac, nothing, nothing, rule, merit, diag)
    else
        fac = init_factor(m.linsolve, prob, x, p)
        return NewtonState(jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, x),
            init_globalization(m.globalization, x), rule, merit, diag)
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    state = with_rule(state, reset_rule(state.rule, m.globalization))
    state = with_merit(state, start_merit(m.globalization, Fx))
    return with_diag(state, NewtonDiagnostics(real(eltype(x))))
end

# The state that `solve!!` returns carries the merit of `init_state` again, so that a line-search
# step on it fails closed as on a fresh state, and `finish!!` copies the diagnostics of the last
# factorisation into the returned `SolverStatus`.
@inline function finish!!(
        state::NewtonState, m::Newton, prob, x, Fx, status::SolverStatus{R}) where {R}
    d = state.diag
    st = SolverStatus{R}(
        status.code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, d.rank, d.nullnorm, d.gradratio)
    state_out = with_merit(state, init_merit(m.globalization, x))
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

# `‖Fx‖` from the line-search merit `norm2(Fx)` when it lies in the normal range, so that a
# line-search step takes no extra host/device reduction over `Fx`, and `rnorm(Fx)` otherwise.
@inline residual_norm(Fx, ::Nothing) = rnorm(Fx)
@inline function residual_norm(Fx, merit::R) where {R <: Real}
    n = Fx isa Number ? 1 : length(Fx)
    isfinite(merit) && merit >= n * floatmin(R) && return sqrt(merit)
    return rnorm(Fx)
end

# `‖Jᴴ Fx‖ / (‖J‖_F ‖Fx‖)` at the unregularised Jacobian `J`, before `regularize!!` or
# `factorize!!` touches `J`, and `0` when `‖J‖_F` or `‖Fx‖` is `0`.
@inline function grad_ratio(d, J, Fx, fnorm::R) where {R <: Real}
    Jmat = J isa Number ? SMatrix{1, 1, typeof(J)}((J,)) :
           (J isa StaticMatrix ? SMatrix(J) : J)
    jnorm = rnorm(Jmat)
    (iszero(jnorm) || iszero(fnorm)) && return zero(R)
    Fx_vec = Fx isa Number ? SVector{1, typeof(Fx)}((Fx,)) :
             (Fx isa StaticVector ? SVector(Fx) : Fx)
    g = mul!!(d, adjoint(Jmat), Fx_vec)
    return R(rnorm(g)) / jnorm / fnorm
end

# The Jacobian at `x`, regularised, its factor, and the diagnostics of both: the one
# factorisation of every `Newton` step.
@inline function newton_factor(m::Newton, state::NewtonState, prob, x, Fx, p)
    R = real(eltype(x))
    fnorm = R(residual_norm(Fx, state.merit))
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
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    _, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    return x₊, F₊, with_diag(with_merit(state, merit), diag), info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A failed factor or a Newton step
# whose norm is zero or not finite leaves the Cauchy step.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
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
    state₊ = with_diag(with_rule(state, TrustRadius(radius, state.rule.radius₀)), diag)
    return x₊, F₊, state₊, info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
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
    state = with_diag(with_merit(with_factor(state, fac), merit), diag)
    if info.code == SINGULAR
        return x₊, F₊, with_rule(state, FactorAge(Int32(0), false)), info
    end
    state₊ = with_rule(state, FactorAge(age₊, info.code == STALLED))
    return x₊, F₊, state₊, info
end

@inline lm_svd_factor(_, m, ::StaticVector, Jreg) = lm_static_svd(m.linsolve, Jreg)
@inline function lm_svd_factor(_, m, ::Number, Jreg)
    return lm_static_svd(m.linsolve, SMatrix{1, 1, typeof(Jreg)}((Jreg,)))
end
@inline lm_svd_factor(state, m, _, Jreg) = factorize!!(state.fac, m.linsolve, Jreg)

@inline function lm_newton_factor(
        m::Newton, state::NewtonState, prob, x, Fx, p, fnorm::R) where {R}
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr = grad_ratio(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    if m.linsolve isa SVDFactorization && m.globalization.scaling === Val(:none)
        fac = lm_svd_factor(state, m, x, Jreg)
        rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
        return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr)
    end
    return Jreg, state.fac, NewtonDiagnostics{R}(Int32(-1), R(NaN), gr)
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: LevenbergMarquardt, J, RF, Reg}
    R = real(eltype(x))
    fnorm = R(rnorm(Fx))
    μ = state.rule.μ
    if !isfinite(μ)
        info = StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state, info
    end
    lm = m.globalization
    μ_min_eff = max(R(lm.μ_min), 4 * eps(R))
    Jreg, fac, diag = lm_newton_factor(m, state, prob, x, Fx, p, fnorm)
    state_d = with_diag(state, diag)
    j_ok = Jreg isa Number ? isfinite(Jreg) : all(isfinite, Jreg)
    if !j_ok || !isfinite(diag.gradratio)
        info = StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state_d, info
    end
    use_svd = m.linsolve isa SVDFactorization && lm.scaling === Val(:none)
    if use_svd && fac.info > 0
        info = StepInfo{R}(SINGULAR, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state_d, info
    end
    if iszero(fnorm)
        info = StepInfo{R}(SUCCESS, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state_d, info
    end
    failures = Int32(0)
    while true
        if use_svd
            δ = lm_svd_step(state.d, fac, Fx, μ)
            step_ok = δ isa Number ? isfinite(δ) : all(isfinite, δ)
        else
            δ, step_ok = lm_stacked_qr_step(state.d, Jreg, Fx, μ, lm.scaling, state.g)
        end
        if step_ok
            Jδ = mul!!(lm_Jδ(state.g), Jreg, δ)
            pred = -2 * rdot(Fx, Jδ) - norm2(Jδ)
            if pred <= zero(R)
                Flin = axpby!!(lm_Ft(state.g), one(R), Fx, one(R), Jδ)
                flin_norm = rnorm(Flin)
                pred = (fnorm - flin_norm) * (fnorm + flin_norm)
            end
            x₊ = update_iterate!!(lm_xt(state.g), x, one(R), δ)
            F₊ = residual!!(lm_Ft(state.g), prob, x₊, p)
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
                    state₊ = with_rule(state_d, DampingParameter(μ_next))
                    info = StepInfo{R}(SUCCESS, fnorm₊, stepnorm, failures, false, limited)
                    return x, Fx, state₊, info
                end
            end
        end
        failures += Int32(1)
        μ *= R(lm.μ_increase)
        if !isfinite(μ) || μ > R(lm.μ_max)
            state_fail = with_rule(state_d, DampingParameter(μ))
            info = StepInfo{R}(STALLED, fnorm, zero(R), failures, false, false)
            return x, Fx, state_fail, info
        end
    end
end

NonlinearProblem(F; jacobian = nothing) = NonlinearProblem(F, jacobian)

function solve(
        prob::NonlinearProblem, m::Newton, x₀::Union{AbstractVector, Number},
        p = nothing, options::Union{Nothing, Options} = nothing; kwargs...)
    return solve(prob, x₀, m, p; options = options, kwargs...)
end

function init(
        prob::NonlinearProblem, m::Newton, x₀::Union{AbstractVector, Number},
        p = nothing; options::Union{Nothing, Options} = nothing, kwargs...)
    return init(prob, x₀, m, p; options = options, kwargs...)
end

@inline function nonlinear_solver_status(s::NonlinearSolver)
    Fx = getfield(s, :state).Fx
    opts = getfield(s, :options)
    d = getfield(s, :state).diag
    fnorm = rnorm(Fx)
    R = typeof(fnorm)
    tol = max(opts.f_abstol, opts.f_reltol)
    code = !isfinite(fnorm) ? NONFINITE : (fnorm <= tol ? SUCCESS : MAXITERS)
    return SolverStatus{R}(
        code, Int32(0), fnorm, zero(R), Int32(0), false,
        d.rank, d.nullnorm, d.gradratio)
end

@inline function Base.getproperty(s::NonlinearSolver, name::Symbol)
    name === :Fx && return getfield(s, :state).Fx
    name === :status && return nonlinear_solver_status(s)
    return getfield(s, name)
end
