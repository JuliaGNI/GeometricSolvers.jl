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
    return Newton(linsolve, globalization, jacobian, refinement, reg, r)
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
# working-precision factor slot for mixed-precision promotion, and the mixed-precision promotion
# and slow/floor step counters.
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
    floor_steps::Int32
end

@inline function NewtonState(
        jacprep::JP, jvpprep::VP, Fx::FX, d::D, fac::FAC,
        J::JM, g::G, rule::RS, merit::PH, diag::DG) where {
        JP, VP, FX, D, FAC, JM, G, RS, PH, DG}
    return NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, DG, Nothing, Nothing}(
        jacprep, jvpprep, Fx, d, fac, J, g, rule, merit, diag,
        nothing, nothing, false, Int32(0), Int32(0))
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule, s.merit, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.floor_steps)
end

@inline function with_factor_T(s::NewtonState, fac_T)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.diag,
        s.refbuf, fac_T, s.promoted, s.slow_steps, s.floor_steps)
end

@inline function with_rule(s::NewtonState, rule)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, rule, s.merit, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.floor_steps)
end

@inline function with_merit(s::NewtonState, merit)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, merit, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.floor_steps)
end

@inline function with_diag(s::NewtonState, diag)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.floor_steps)
end

@inline function with_mixed(
        s::NewtonState, promoted::Bool, slow_steps::Int32, floor_steps::Int32)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.diag,
        s.refbuf, s.fac_T, promoted, slow_steps, floor_steps)
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
        fac_T = (m.reuse === nothing || !mixed) ? nothing :
                init_factor(working_linsolve(m.linsolve), prob, x, p)
        return NewtonState(
            jacprep, jvpprep, nothing, nothing, fac, nothing, nothing, rule, merit, diag,
            nothing, fac_T, false, Int32(0), Int32(0))
    else
        fac = init_factor(m.linsolve, prob, x, p)
        fac_T = mixed ? init_factor(working_linsolve(m.linsolve), prob, x, p) : nothing
        refbuf = init_refinement(m.refinement, x)
        return NewtonState(
            jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, x),
            init_globalization(m.globalization, x), rule, merit, diag,
            refbuf, fac_T, false, Int32(0), Int32(0))
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    state = with_rule(state, reset_rule(state.rule))
    state = with_merit(state, start_merit(m.globalization, Fx))
    state = with_mixed(state, false, Int32(0), Int32(0))
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
    return x, st, with_merit(state, init_merit(m.globalization, x), NaN)
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

# Factorisation promoted to the working precision `T` of the iterate when a mixed-precision
# factorisation (`TF ≠ T`) fails refinement or stalls on linear-model accuracy.
@inline function newton_factor_T(m::Newton, state::NewtonState, prob, x, Fx, p)
    R = real(eltype(x))
    fnorm = R(residual_norm(Fx, state.merit))
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr = grad_ratio(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    ls_T = working_linsolve(m.linsolve)
    fac = factorize!!(state.fac_T, ls_T, Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr)
end

@inline function newton_direction(
        m::Newton, state::NewtonState, prob, x, Fx, fnorm::R, p, fac) where {R <: Real}
    d₀ = ldiv!!(state.d, fac, Fx)
    d₁ = map!!(-, state.d, d₀)
    return refine!!(
        d₁, m.refinement, fac, state.jvpprep, prob, x, Fx, fnorm, p, state.refbuf)
end

# Relative linear-model residual `‖F(x) + J(x) d‖ / ‖F(x)‖` in working precision `T`, evaluated
# before `globalize_merit!!` overwrites `Fx` in R3, using the globalization trial buffer `g_Ft`
# as scratch.
@inline function linear_residual_ratio(
        state::NewtonState, prob, x, Fx, d, fnorm_old::R, p) where {R <: Real}
    T = eltype(x)
    (iszero(fnorm_old) || !isfinite(fnorm_old)) && return zero(R)
    tmp = g_Ft(state.g)
    Jv = jvp!!(tmp, state.jvpprep, prob, x, d, p)
    lin_r = axpby!!(tmp, one(T), Fx, one(T), Jv)
    return R(rnorm(lin_r)) / fnorm_old
end

# Distinguish a stall caused by an inaccurate low-precision linear step (`lin_rel > η_lin`, which
# promotes the factorisation to working precision `T`) from a stall at the precision floor of the
# residual evaluation (`lin_rel <= η_lin`, which terminates with `STALLED` and `promoted = false`).
@inline function update_mixed_stall(
        state::NewtonState, info::StepInfo{R},
        fnorm_old::R, lin_rel::R, mixed::Bool) where {R <: Real}
    if !mixed || state.promoted || info.code != SUCCESS
        return state, info
    end
    if info.fnorm >= R(0.5) * fnorm_old
        η_lin = max(sqrt(eps(R)), R(1e-4))
        if lin_rel > η_lin
            slow = state.slow_steps + one(Int32)
            if slow >= Int32(2)
                info_p = StepInfo{R}(
                    info.code, info.fnorm, info.stepnorm,
                    info.failures + one(Int32), info.stalled, true)
                return with_mixed(state, true, slow, Int32(0)), info_p
            end
            return with_mixed(state, false, slow, Int32(0)), info
        else
            fl = state.floor_steps + one(Int32)
            if fl >= Int32(2)
                info_s = StepInfo{R}(
                    STALLED, info.fnorm, info.stepnorm,
                    info.failures, true, false)
                return with_mixed(state, false, Int32(0), fl), info_s
            end
            return with_mixed(state, false, Int32(0), fl), info
        end
    end
    return with_mixed(state, false, Int32(0), Int32(0)), info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    mixed = factor_eltype(m.linsolve, T) !== T
    fnorm_old = R(residual_norm(Fx, state.merit))
    _, fac, diag = state.promoted ?
                   newton_factor_T(m, state, prob, x, Fx, p) :
                   newton_factor(m, state, prob, x, Fx, p)
    promoted_step = false
    if fac.info > Int32(0)
        if mixed && !state.promoted
            _, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
            promoted_step = true
        end
        if fac.info > Int32(0)
            info_sing = StepInfo{R}(
                SINGULAR, rnorm(Fx), zero(R),
                promoted_step ? Int32(1) : Int32(0), false, promoted_step)
            state_sing = promoted_step ?
                         with_mixed(state, true, state.slow_steps, state.floor_steps) :
                         state
            return x, Fx, with_diag(state_sing, diag), info_sing
        end
    end
    d, ref_failed = if promoted_step
        d₀ = ldiv!!(state.d, fac, Fx)
        map!!(-, state.d, d₀), false
    else
        newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
    end
    if ref_failed && mixed && !state.promoted
        _, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
        promoted_step = true
        if fac.info > Int32(0)
            info_sing = StepInfo{R}(
                SINGULAR, rnorm(Fx), zero(R), Int32(1), false, true)
            state_sing = with_mixed(state, true, state.slow_steps, state.floor_steps)
            return x, Fx, with_diag(state_sing, diag), info_sing
        end
        d₀ = ldiv!!(state.d, fac, Fx)
        d = map!!(-, state.d, d₀)
    end
    lin_rel = (mixed && !state.promoted && !promoted_step) ?
              linear_residual_ratio(state, prob, x, Fx, d, fnorm_old, p) : zero(R)
    sk = exact_solve(fac) ?
         newton_stepkind(m, T, true, promoted_step || state.promoted) :
         MeasuredSlope()
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk, state.merit)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm,
            info.failures + one(Int32), info.stalled, true)
        state = with_mixed(state, true, Int32(0), Int32(0))
    else
        state, info = update_mixed_stall(state, info, fnorm_old, lin_rel, mixed)
    end
    return x₊, F₊, with_diag(with_merit(state, merit), diag), info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A failed factor or a Newton step
# whose norm is zero or not finite leaves the Cauchy step.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::NonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    mixed = factor_eltype(m.linsolve, T) !== T
    fnorm_old = R(rnorm(Fx))
    Jreg, fac, diag = state.promoted ?
                      newton_factor_T(m, state, prob, x, Fx, p) :
                      newton_factor(m, state, prob, x, Fx, p)
    promoted_step = false
    if fac.info > Int32(0) && mixed && !state.promoted
        Jreg, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
        promoted_step = true
    end
    if fac.info == Int32(0)
        d, ref_failed = if promoted_step
            d₀ = ldiv!!(state.d, fac, Fx)
            map!!(-, state.d, d₀), false
        else
            newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
        end
        if ref_failed && mixed && !state.promoted
            Jreg, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
            promoted_step = true
            if fac.info == Int32(0)
                d₀ = ldiv!!(state.d, fac, Fx)
                d = map!!(-, state.d, d₀)
            end
        end
        if fac.info == Int32(0)
            nd = rnorm(d)
            newton = isfinite(nd) && !iszero(nd)
        else
            d = Fx
            newton = false
        end
    else
        d = Fx
        newton = false
    end
    lin_rel = (mixed && !state.promoted && !promoted_step && newton) ?
              linear_residual_ratio(state, prob, x, Fx, d, fnorm_old, p) : zero(R)
    x₊, F₊, radius, info = dogleg_step(
        state.g, prob, x, Fx, p, Jreg, d, newton, state.rule.radius)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm,
            info.failures + one(Int32), info.stalled, true)
        state = with_mixed(state, true, Int32(0), Int32(0))
    else
        state, info = update_mixed_stall(state, info, fnorm_old, lin_rel, mixed)
    end
    state₊ = with_diag(with_rule(state, TrustRadius(radius, state.rule.radius₀)), diag)
    return x₊, F₊, state₊, info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::NonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    mixed = factor_eltype(m.linsolve, T) !== T
    fnorm_old = R(residual_norm(Fx, state.merit))
    rule = state.rule
    fresh = rule.age == Int32(0) || rule.stalled
    _, fac, diag = if fresh
        state.promoted ? newton_factor_T(m, state, prob, x, Fx, p) :
        newton_factor(m, state, prob, x, Fx, p)
    else
        (nothing, state.promoted ? state.fac_T : state.fac, state.diag)
    end
    age_fresh = m.reuse == Int32(1) ? Int32(0) : Int32(1)
    age₊ = fresh ? age_fresh :
           (rule.age + one(Int32) >= m.reuse ? Int32(0) : rule.age + one(Int32))
    promoted_step = false
    if fac.info > Int32(0)
        if !fresh
            fresh = true
            age₊ = age_fresh
            _, fac, diag = state.promoted ? newton_factor_T(m, state, prob, x, Fx, p) :
                           newton_factor(m, state, prob, x, Fx, p)
        end
        if fac.info > Int32(0) && mixed && !state.promoted
            _, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
            promoted_step = true
            age₊ = age_fresh
        end
        if fac.info > Int32(0)
            state_f = (promoted_step || state.promoted) ?
                      with_factor_T(state, fac) : with_factor(state, fac)
            state_m = promoted_step ?
                      with_mixed(state_f, true, state.slow_steps, state.floor_steps) :
                      state_f
            state_d = with_diag(state_m, diag)
            info_sing = StepInfo{R}(
                SINGULAR, rnorm(Fx), zero(R),
                promoted_step ? Int32(1) : Int32(0), false, promoted_step)
            return x, Fx, with_rule(state_d, FactorAge(Int32(0), false)), info_sing
        end
    end
    d, ref_failed = if promoted_step
        d₀ = ldiv!!(state.d, fac, Fx)
        map!!(-, state.d, d₀), false
    else
        newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
    end
    # When refinement fails on a reused factor (`!fresh`), refactorise early at the current
    # iterate in `TF` (`age` reset) and retry refinement before considering promotion to `T`.
    if !fresh && ref_failed
        fresh = true
        age₊ = age_fresh
        _, fac, diag = state.promoted ? newton_factor_T(m, state, prob, x, Fx, p) :
                       newton_factor(m, state, prob, x, Fx, p)
        if fac.info == Int32(0)
            d, ref_failed = newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
        end
    end
    if (fac.info > Int32(0) || ref_failed) && mixed && !state.promoted
        _, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
        promoted_step = true
        fresh = true
        age₊ = age_fresh
        if fac.info > Int32(0)
            state_f = with_factor_T(state, fac)
            state_m = with_mixed(state_f, true, state.slow_steps, state.floor_steps)
            state_d = with_diag(state_m, diag)
            info_sing = StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(1), false, true)
            return x, Fx, with_rule(state_d, FactorAge(Int32(0), false)), info_sing
        end
        d₀ = ldiv!!(state.d, fac, Fx)
        d = map!!(-, state.d, d₀)
    end
    lin_rel = (mixed && !state.promoted && !promoted_step) ?
              linear_residual_ratio(state, prob, x, Fx, d, fnorm_old, p) : zero(R)
    sk = exact_solve(fac) ?
         newton_stepkind(m, T, fresh, promoted_step || state.promoted) :
         MeasuredSlope()
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk, state.merit)
    # A reused factor can give a direction that does not descend. A line search along it that
    # fails without a step is retried once, in the same step, from a fresh factor, and from the
    # same residual and merit.
    if !fresh && info.code == LINESEARCH_FAILED && iszero(info.stepnorm)
        fresh = true
        age₊ = age_fresh
        _, fac, diag = state.promoted ? newton_factor_T(m, state, prob, x, Fx, p) :
                       newton_factor(m, state, prob, x, Fx, p)
        if fac.info > Int32(0) && mixed && !state.promoted
            _, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
            promoted_step = true
        end
        if fac.info > Int32(0)
            state_f = (promoted_step || state.promoted) ?
                      with_factor_T(state, fac) : with_factor(state, fac)
            state_m = promoted_step ?
                      with_mixed(state_f, true, state.slow_steps, state.floor_steps) :
                      state_f
            state_d = with_diag(with_merit(state_m, merit), diag)
            info_sing = StepInfo{R}(
                SINGULAR, rnorm(Fx), zero(R),
                promoted_step ? Int32(1) : Int32(0), false, promoted_step)
            return x, Fx, with_rule(state_d, FactorAge(Int32(0), false)), info_sing
        end
        d, ref_failed = if promoted_step
            d₀ = ldiv!!(state.d, fac, Fx)
            map!!(-, state.d, d₀), false
        else
            newton_direction(m, state, prob, x, Fx, fnorm_old, p, fac)
        end
        if ref_failed && mixed && !state.promoted
            _, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
            promoted_step = true
            if fac.info > Int32(0)
                state_f = with_factor_T(state, fac)
                state_m = with_mixed(state_f, true, state.slow_steps, state.floor_steps)
                state_d = with_diag(with_merit(state_m, merit), diag)
                info_sing = StepInfo{R}(
                    SINGULAR, rnorm(Fx), zero(R), Int32(1), false, true)
                return x, Fx, with_rule(state_d, FactorAge(Int32(0), false)), info_sing
            end
            d₀ = ldiv!!(state.d, fac, Fx)
            d = map!!(-, state.d, d₀)
        end
        lin_rel = (mixed && !state.promoted && !promoted_step) ?
                  linear_residual_ratio(state, prob, x, Fx, d, fnorm_old, p) : zero(R)
        sk = exact_solve(fac) ?
             newton_stepkind(m, T, true, promoted_step || state.promoted) :
             MeasuredSlope()
        x₊, F₊, _, info, merit = globalize_merit!!(
            state.g, m.globalization, prob, x, Fx, d, p, state.jvpprep, sk, state.merit)
    end
    state = (promoted_step || state.promoted) ?
            with_factor_T(state, fac) : with_factor(state, fac)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm,
            info.failures + one(Int32), info.stalled, true)
        state = with_mixed(state, true, Int32(0), Int32(0))
    else
        was_promoted = state.promoted
        state, info = update_mixed_stall(state, info, fnorm_old, lin_rel, mixed)
        if !was_promoted && state.promoted
            age₊ = Int32(0)
        end
    end
    state = with_diag(with_merit(state, merit), diag)
    if info.code == SINGULAR
        return x₊, F₊, with_rule(state, FactorAge(Int32(0), false)), info
    end
    state₊ = with_rule(state, FactorAge(age₊, info.code == STALLED))
    return x₊, F₊, state₊, info
end

function init(
        prob::NonlinearProblem, method::Newton,
        x₀::Union{AbstractVector, Number}, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    return init(prob, x₀, method, p; options = options, kwargs...)
end

function solve(
        prob::NonlinearProblem, method::Newton,
        x₀::Union{AbstractVector, Number}, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    return solve(prob, x₀, method, p; options = options, kwargs...)
end

function solve(
        prob::NonlinearProblem, method::Newton,
        x₀::Union{AbstractVector, Number}, p,
        options::Options; kwargs...)
    return solve(prob, x₀, method, p; options = options, kwargs...)
end
