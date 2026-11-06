"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing)

Newton's method for [`NonlinearProblem`](@ref), with linear solver `linsolve`
([`LUFactorization`](@ref), [`QRFactorization`](@ref) or [`SVDFactorization`](@ref)),
globalization `globalization`
([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref), [`StrongWolfe`](@ref) or the
trust region [`DogLeg`](@ref)), Jacobian selector `jacobian`, iterative refinement `refinement`
([`NoRefinement`](@ref) or [`Refinement`](@ref)), diagonal regularisation `regularization`, and
factor-reuse period `reuse`.

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
- `refinement::RF`: the refinement setting ([`NoRefinement`](@ref) or [`Refinement`](@ref)).
- `regularization::R`: non-negative finite diagonal shift ``μ ≥ 0`` added as ``J + μ I`` before
  factorisation when `μ > 0`.
- `reuse::K`: `nothing` (`K = Nothing`, factorise at every step) or `Int32(k)` (`K = Int32`,
  reuse the factor for up to `k` consecutive steps, refreshing early after a `STALLED` step or a
  failed [`Refinement`](@ref) on a reused factor). A line search along a reused factor that fails
  without a step is retried once, in the same step, from a fresh factor. A [`DogLeg`](@ref)
  globalization takes no reuse: its model needs the Jacobian of the current iterate, and the
  constructor raises an `ArgumentError`.
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
    ref = refinement isa Bool ? (refinement ? Refinement() : NoRefinement()) : refinement
    return Newton(linsolve, globalization, jacobian, ref, reg, r)
end

"""
    QuasiNewton(k; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
                refinement = NoRefinement(), regularization = 0)

Construct [`Newton`](@ref) with factor-reuse period `reuse = Int32(k)` (`k ≥ 1`).
"""
function QuasiNewton(k::Integer; kwargs...)
    return Newton(; reuse = k, kwargs...)
end

Adapt.@adapt_structure Newton

@inline function has_analytic_jacobian(prob::NonlinearProblem)
    J = getfield(prob, :J)
    return J !== nothing && !(J isa Union{AbstractVector, Number})
end

@inline function newton_ad_preps(m::Newton, prob::NonlinearProblem, Fx, x, p)
    if m.refinement isa Refinement && m.jacobian isa AutoFiniteDiff
        throw(ArgumentError("Refinement requires an exact or forward-mode Jacobian-vector product; AutoFiniteDiff is not supported"))
    end
    has_J = has_analytic_jacobian(prob)
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
            if Fx === nothing
                nothing
            elseif m.refinement isa Refinement ||
                   factor_eltype(m.linsolve, eltype(x)) !== eltype(x)
                JacobianMatrixJVP(buffer(x, eltype(x), length(x), length(x)))
            else
                jacobian_matrix_jvp(m.globalization, x)
            end
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

@inline function init_newton_J(m::Newton, fac, fac_T, x)
    if !(m.globalization isa DogLeg)
        if factor_eltype(m.linsolve, eltype(x)) === eltype(x)
            return fac.factors
        elseif fac_T !== nothing && m.linsolve isa LUFactorization
            return fac_T.factors
        end
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
# globalization workspace, the rule state of its reuse or its `DogLeg`, the merit `norm2(Fx)` of
# the current residual for a line search (`nothing` for any other globalization), the diagnostics
# of the last factorisation, the refinement buffer (`nothing` for `NoRefinement` or R1), the
# working-precision factor slot for mixed-precision promotion, the promotion flag, and the count
# of consecutive slow steps a promotion waits for (`slow_steps = Int32(-1)` before `start!!` and
# after `finish!!`).
struct NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, DG, RB, FT}
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
    refbuf::RB
    fac_T::FT
    promoted::Bool
    slow_steps::Int32
end

@inline function NewtonState(
        jacprep::JP, jvpprep::VP, Fx::FX, d::D, fac::FAC,
        J::JM, g::G, rule::RS, merit::PH, diag::DG) where {
        JP, VP, FX, D, FAC, JM, G, RS, PH, DG}
    return NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, DG, Nothing, Nothing}(
        jacprep, jvpprep, Fx, d, fac, J, g, rule, merit, diag,
        nothing, nothing, false, Int32(-1))
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule, s.merit, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps)
end

@inline function with_factor_T(s::NewtonState, fac_T)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.diag,
        s.refbuf, fac_T, s.promoted, s.slow_steps)
end

@inline function update_stored_factor(
        s::NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, DG, RB, Nothing},
        fac, promoted::Bool) where {JP, VP, FX, D, FAC, JM, G, RS, PH, DG, RB}
    return promoted ? s : with_factor(s, fac)
end

@inline function update_stored_factor(s::NewtonState, fac, promoted::Bool)
    return promoted ? with_factor_T(s, fac) : with_factor(s, fac)
end

@inline function with_rule(s::NewtonState, rule)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, rule, s.merit, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps)
end

@inline function with_merit(s::NewtonState, merit)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, merit, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps)
end

@inline function with_diag(s::NewtonState, diag)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps)
end

@inline function with_mixed(s::NewtonState, promoted::Bool, slow_steps::Int32)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.diag,
        s.refbuf, s.fac_T, promoted, slow_steps)
end

@inline function init_state(m::Newton, prob::NonlinearProblem, x, p)
    Fx = buffer(x)
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    rule = newton_rule(m, x)
    merit = init_merit(m.globalization, x)
    diag = NewtonDiagnostics(real(eltype(x)))
    mixed = factor_eltype(m.linsolve, eltype(x)) !== eltype(x)
    if Fx === nothing
        fac = m.reuse === nothing ? nothing : init_factor(m.linsolve, prob, x, p)
        return NewtonState(
            jacprep, jvpprep, nothing, nothing, fac, nothing, nothing, rule, merit, diag,
            nothing, nothing, false, Int32(-1))
    else
        fac = init_factor(m.linsolve, prob, x, p)
        fac_T = mixed ?
                init_factor(promote_linsolve(m.linsolve, eltype(x)), prob, x, p) :
                nothing
        refbuf = init_refinement(m.refinement, x)
        return NewtonState(
            jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, fac_T, x),
            init_globalization(m.globalization, x), rule, merit, diag,
            refbuf, fac_T, false, Int32(-1))
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    state = with_rule(state, reset_rule(state.rule))
    state = with_merit(state, start_merit(m.globalization, Fx))
    state = with_mixed(state, false, Int32(0))
    return with_diag(state, NewtonDiagnostics(real(eltype(x))))
end

# The state that `solve!!` returns carries the merit and unstarted sentinel of `init_state` again,
# so that a `step!!` on it without `start!!` fails closed with `NONFINITE`, and `finish!!` copies
# the diagnostics of the last factorisation into the returned `SolverStatus`.
@inline function finish!!(
        state::NewtonState, m::Newton, prob, x, Fx, status::SolverStatus{R}) where {R}
    d = state.diag
    st = SolverStatus{R}(
        status.code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, d.rank, d.nullnorm, d.gradratio)
    state_end = with_mixed(
        with_merit(state, init_merit(m.globalization, x)),
        state.promoted, Int32(-1))
    return x, st, state_end
end

@inline function newton_stepkind(
        m::Newton, ::Type{T}, fresh::Bool, in_working_T::Bool = false) where {T}
    if fresh && (m.jacobian === nothing || m.jacobian isa AbstractADType) &&
       iszero(m.regularization) &&
       (in_working_T ||
        (factor_eltype(m.linsolve, T) === T && m.refinement isa NoRefinement))
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

# The Jacobian at `x`, regularised, its factor in `TF`, and the diagnostics of both: the
# factorisation that every `Newton` step takes, and the one a promotion may follow.
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

# The Jacobian at `x` and its factorisation in the working precision `T`, with the diagnostics of
# both: the one factorisation of a step whose factorisation already runs in `T`, because the
# solve was promoted to it.
@inline function newton_factor_T(m::Newton, state::NewtonState, prob, x, Fx, p)
    T = eltype(x)
    R = real(T)
    fnorm = R(residual_norm(Fx, state.merit))
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr = grad_ratio(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    ls_T = promote_linsolve(m.linsolve, T)
    fac = factorize!!(state.fac_T, ls_T, Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr)
end

# The factorisation in the working precision `T`, from the Jacobian the step already holds: the
# promotion of a mixed-precision step, which reads no Jacobian again. The factor is written into
# `state.fac_T`, the slot that `init_state` reserved, so that a promotion allocates nothing.
@inline function promote_factor!!(
        state::NewtonState, m::Newton, Jreg, x, Fx, fnorm::R,
        diag::NewtonDiagnostics{R}) where {R <: Real}
    T = eltype(x)
    fac = factorize!!(state.fac_T, promote_linsolve(m.linsolve, T), Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    return fac, NewtonDiagnostics{R}(rk, nn, diag.gradratio)
end

@inline function newton_direction(
        m::Newton, state::NewtonState, prob, x, Fx, fnorm::R, p, fac) where {R <: Real}
    d₀ = ldiv!!(state.d, fac, Fx)
    d₁ = map!!(-, state.d, d₀)
    return refine!!(
        d₁, m.refinement, fac, state.jvpprep, prob, x, Fx, fnorm, p, state.refbuf)
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
    fnorm = R(residual_norm(Fx, state.merit))
    d, _ = newton_direction(m, state, prob, x, Fx, fnorm, p, fac)
    sk = exact_solve(fac) ? newton_stepkind(m, T, fresh) : MeasuredSlope()
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk, state.merit)
    return x₊, F₊, info, merit
end

# Relative linear-model residual `‖F(x) + J d‖ / ‖F(x)‖` in the working precision `T`, from the
# regularised Jacobian the step already holds: the Jacobian is read, not evaluated again, and the
# trial buffer of the globalization is its scratch. It is taken before `globalize_merit!!`
# overwrites `Fx` in R3.
@inline function linear_residual_ratio(
        state::NewtonState, J, Fx, d, fnorm_old::R) where {R <: Real}
    (iszero(fnorm_old) || !isfinite(fnorm_old)) && return zero(R)
    T = eltype(d)
    tmp = g_Ft(state.g)
    Jd = mul!!(tmp, J, d)
    lin = axpby!!(tmp, one(T), Fx, one(T), Jd)
    return R(rnorm(lin)) / fnorm_old
end

# The two readings of one mixed-precision step, from the relative linear-model residual `lin_rel`
# of the direction it took. A step whose linear model is inaccurate in `TF` (`lin_rel > η_lin`)
# and which lowers the residual by less than half counts toward a promotion to `T`, and two such
# steps promote. A step whose linear model is accurate in `T` and whose residual does not fall at
# all has met the floor of `F` itself: it is `STALLED`, with no failure and no promotion, and the
# solve ends after `max_stalls` consecutive such steps. Any other step of `T` accuracy is
# unchanged, as a step in the working precision is.
@inline function read_mixed_stall(
        state::NewtonState, info::StepInfo{R}, fnorm_old::R, lin_rel::R) where {R <: Real}
    η_lin = max(sqrt(eps(R)), R(1e-4))
    if info.code == SUCCESS && lin_rel <= η_lin && info.fnorm >= fnorm_old
        return with_mixed(state, false, Int32(0)),
        StepInfo{R}(STALLED, info.fnorm, info.stepnorm, info.failures, false, info.limited)
    elseif info.fnorm > R(0.5) * fnorm_old && lin_rel > η_lin
        slow = state.slow_steps + one(Int32)
        if slow >= Int32(2)
            return with_mixed(state, true, Int32(0)),
            StepInfo{R}(
                info.code, info.fnorm, info.stepnorm,
                info.failures + one(Int32), true, info.limited)
        end
        return with_mixed(state, false, slow), info
    end
    return with_mixed(state, false, Int32(0)), info
end

# `true` where the factorisation of `m` runs in a precision other than the working type `T`: the
# step whose behaviour `state.promoted` and `slow_steps` govern. The predicate is a constant of
# the types, so the compiler removes the half it does not take.
@inline function is_mixed(m::Newton, ::Type{T}) where {T}
    return factor_eltype(m.linsolve, T) !== T
end

# The step of a mixed-precision solve on a state that no `start!!` saw: `start!!` sets
# `slow_steps` to `0` and `finish!!` returns it to `-1`, so such a step reports `NONFINITE` and
# never a result. A factorisation in the working type reads no counter, and has none to check.
@inline function mixed_unstarted(state::NewtonState, Fx, ::Type{R}) where {R <: Real}
    state.slow_steps < Int32(0) || return nothing
    fn = Fx === nothing ? zero(R) : R(rnorm(Fx))
    return StepInfo{R}(NONFINITE, fn, zero(R), Int32(0), state.promoted, false)
end

# The result of the mixed-precision half of a step: the direction, the regularised Jacobian it
# came from, the state that carries the factor a reuse path reuses and the promotion flag, and
# the flags of the step. No factor crosses this boundary: `state.fac` holds the one in `TF` and
# `state.fac_T` the one in `T`, so no call site holds a value of either type.
struct MixedNewton{S, D, J}
    d::D
    J::J
    state::S
    fresh::Bool
    exact::Bool
    ok::Bool
    failures::Int32
end

# The step-rule failures that one mixed-precision step's retries add, folded into its `StepInfo`.
@inline function bump_failures(info::StepInfo{R}, n::Int32) where {R <: Real}
    iszero(n) && return info
    return StepInfo{R}(
        info.code, info.fnorm, info.stepnorm, info.failures + n, info.promoted, info.limited)
end

# The state a mixed-precision step returns with `fac` stored, for a solve that reuses its factor:
# a step with no reuse keeps the slot of `init_state`, and its state keeps one type.
@inline function stored_factor(
        state::NewtonState, fac, promoted::Bool, store::Bool)
    return store ? update_stored_factor(state, fac, promoted) : state
end

# The mixed-precision half of one Newton step: the solve in `TF`, the promotion to `T` when the
# factor or the refinement fails there, and the flags of the direction it returns. A reused factor
# that fails is refactorised at the current iterate before any promotion, so that a stale factor
# costs a factorisation in `TF` and not the working precision. A state whose factorisation is
# already promoted solves in `T` at every step.
@inline function mixed_newton!!(
        state::NewtonState, m::Newton, prob, x, Fx, fnorm_old::R, p,
        fresh::Bool, store::Bool) where {R <: Real}
    if state.promoted
        Jreg, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
        if fac.info > Int32(0)
            return MixedNewton(
                state.d, Jreg, with_diag(state, diag), true, exact_solve(fac), false,
                Int32(0))
        end
        d, _ = newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
        return MixedNewton(
            d, Jreg, with_diag(state, diag), true, exact_solve(fac), true, Int32(0))
    end
    Jreg, fac, diag = fresh ? newton_factor(m, state, prob, x, Fx, p) :
                      (nothing, state.fac, state.diag)
    failures = Int32(0)
    if !fresh && fac.info > Int32(0)
        Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
        failures += one(Int32)
        fresh = true
    end
    d, ref_failed = if fac.info > Int32(0)
        state.d, true
    else
        newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
    end
    if !fresh && ref_failed
        Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
        failures += one(Int32)
        fresh = true
        if fac.info == Int32(0)
            d, ref_failed = newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
        end
    end
    if ref_failed || fac.info > Int32(0)
        fac_T, diag_T = promote_factor!!(state, m, Jreg, x, Fx, fnorm_old, diag)
        failures += one(Int32)
        st = with_mixed(
            with_diag(stored_factor(state, fac_T, true, store), diag_T),
            true, state.slow_steps)
        fac_T.info > Int32(0) && return MixedNewton(
            state.d, Jreg, st, true, exact_solve(fac_T), false, failures)
        d₀ = ldiv!!(state.d, fac_T, Fx)
        return MixedNewton(
            map!!(-, state.d, d₀), Jreg, st, true, exact_solve(fac_T), true, failures)
    end
    return MixedNewton(
        d, Jreg, with_diag(stored_factor(state, fac, false, store), diag),
        fresh, exact_solve(fac), true, failures)
end

# The step of a `Newton` whose factorisation runs in the working type `T`: one factorisation, one
# refined direction and one globalized step, with no mixed-precision state read.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
    _, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, true)
    return x₊, F₊, with_diag(with_merit(state, merit), diag), info
end

# The same step with a factorisation in `TF ≠ T`: the mixed-precision half of the step, the
# globalized direction it returns, and the reading of the linear-model residual that separates a
# slow `TF` step from the floor of `F`.
@inline function mixed_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    pre = mixed_unstarted(state, Fx, R)
    pre === nothing || return x, Fx, state, pre
    fnorm_old = R(residual_norm(Fx, state.merit))
    md = mixed_newton!!(state, m, prob, x, Fx, fnorm_old, p, true, false)
    if !md.ok
        info = StepInfo{R}(
            SINGULAR, R(rnorm(Fx)), zero(R), md.failures, md.state.promoted, false)
        return x, Fx, md.state, info
    end
    promoted_step = md.state.promoted && !state.promoted
    lin_rel = (promoted_step || state.promoted) ? zero(R) :
              linear_residual_ratio(md.state, md.J, Fx, md.d, fnorm_old)
    sk = md.exact ? newton_stepkind(m, T, true, md.state.promoted) : MeasuredSlope()
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, md.d, p, state.jvpprep, sk, state.merit)
    info = bump_failures(info, md.failures)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm,
            info.failures + one(Int32), true, info.limited)
        st = with_mixed(md.state, true, Int32(0))
    elseif state.promoted
        st = md.state
    else
        st, info = read_mixed_stall(md.state, info, fnorm_old, lin_rel)
    end
    return x₊, F₊, with_merit(st, merit), info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A failed factor or a Newton step
# whose norm is zero or not finite leaves the Cauchy step.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
    R = real(eltype(x))
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    d, newton = dogleg_direction(m, state, prob, x, Fx, R(rnorm(Fx)), p, fac)
    x₊, F₊, radius, info = dogleg_step(
        state.g, prob, x, Fx, p, Jreg, d, newton, state.rule.radius)
    state₊ = with_diag(with_rule(state, TrustRadius(radius, state.rule.radius₀)), diag)
    return x₊, F₊, state₊, info
end

# The direction a `DogLeg` step tries first, and whether it is a Newton step at all: a singular
# factor leaves the Cauchy step, and so does a direction of zero or non-finite norm.
@inline function dogleg_direction(m::Newton, state, prob, x, Fx, fnorm::R, p, fac) where {R}
    fac.info > Int32(0) && return Fx, false
    d, _ = newton_direction(m, state, prob, x, Fx, fnorm, p, fac)
    nd = rnorm(d)
    return d, isfinite(nd) && !iszero(nd)
end

@inline function dogleg_direction(md::MixedNewton, Fx)
    md.ok || return Fx, false
    nd = rnorm(md.d)
    return md.d, isfinite(nd) && !iszero(nd)
end

# The `DogLeg` step of a solve whose factorisation runs in `TF ≠ T`.
@inline function mixed_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    R = real(eltype(x))
    pre = mixed_unstarted(state, Fx, R)
    pre === nothing || return x, Fx, state, pre
    fnorm_old = R(rnorm(Fx))
    md = mixed_newton!!(state, m, prob, x, Fx, fnorm_old, p, true, false)
    promoted_step = md.state.promoted && !state.promoted
    d, newton = dogleg_direction(md, Fx)
    measured = md.ok && newton && !promoted_step && !state.promoted
    lin_rel = measured ?
              linear_residual_ratio(md.state, md.J, Fx, d, fnorm_old) : zero(R)
    x₊, F₊, radius, info = dogleg_step(
        state.g, prob, x, Fx, p, md.J, d, newton, state.rule.radius)
    info = bump_failures(info, md.failures)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm,
            info.failures + one(Int32), true, info.limited)
        st = with_mixed(md.state, true, Int32(0))
    elseif state.promoted
        st = md.state
    else
        st, info = read_mixed_stall(md.state, info, fnorm_old, lin_rel)
    end
    return x₊, F₊,
    with_rule(st, TrustRadius(radius, state.rule.radius₀)), info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
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

# The same step with a factorisation in `TF ≠ T`: the mixed-precision half of the step, and the
# one retry of a line search along a reused factor that its own failure asks for.
@inline function mixed_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    pre = mixed_unstarted(state, Fx, R)
    pre === nothing || return x, Fx, state, pre
    fnorm_old = R(residual_norm(Fx, state.merit))
    rule = state.rule
    fresh = state.promoted || rule.age == Int32(0) || rule.stalled
    age_fresh = m.reuse == Int32(1) ? Int32(0) : Int32(1)
    age₊ = fresh ? age_fresh :
           (rule.age + one(Int32) >= m.reuse ? Int32(0) : rule.age + one(Int32))
    while true
        was_fresh = fresh
        md = mixed_newton!!(state, m, prob, x, Fx, fnorm_old, p, fresh, true)
        st = md.state
        fresh = md.fresh
        if !md.ok
            info = StepInfo{R}(
                SINGULAR, R(rnorm(Fx)), zero(R), md.failures, st.promoted, false)
            return x, Fx, with_rule(st, FactorAge(Int32(0), false)), info
        end
        promoted_step = st.promoted && !state.promoted
        if promoted_step
            age₊ = Int32(0)
        elseif md.fresh && !was_fresh
            age₊ = age_fresh
        end
        lin_rel = (md.fresh && !promoted_step && !st.promoted) ?
                  linear_residual_ratio(st, md.J, Fx, md.d, fnorm_old) : zero(R)
        sk = md.exact ? newton_stepkind(m, T, md.fresh, st.promoted) : MeasuredSlope()
        x₊, F₊, _, info, merit = globalize_merit!!(
            state.g, m.globalization, prob, x, Fx, md.d, p, state.jvpprep, sk, state.merit)
        if !md.fresh && info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
            fresh = true
            age₊ = age_fresh
            continue
        end
        info = bump_failures(info, md.failures)
        if promoted_step
            info = StepInfo{R}(
                info.code, info.fnorm, info.stepnorm,
                info.failures + one(Int32), true, info.limited)
            st = with_mixed(st, true, Int32(0))
        elseif md.fresh && !st.promoted
            st, info = read_mixed_stall(st, info, fnorm_old, lin_rel)
        end
        st = with_merit(st, merit)
        if info.code == SINGULAR
            return x₊, F₊, with_rule(st, FactorAge(Int32(0), false)), info
        end
        return x₊, F₊, with_rule(st, FactorAge(age₊, info.code == STALLED)), info
    end
end
