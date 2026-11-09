"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing)

Newton's method for [`NonlinearProblem`](@ref), [`StationarityProblem`](@ref) and, as
[`GaussNewton`](@ref) configures it, for [`LeastSquaresProblem`](@ref), with linear solver
`linsolve` ([`LUFactorization`](@ref), [`LDLTFactorization`](@ref), [`QRFactorization`](@ref) or
[`SVDFactorization`](@ref)), globalization `globalization`
([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref), [`StrongWolfe`](@ref), the
trust region [`DogLeg`](@ref) or the damped step [`LevenbergMarquardt`](@ref)), Jacobian selector
`jacobian`, iterative refinement `refinement` ([`NoRefinement`](@ref) or [`Refinement`](@ref)),
diagonal regularisation `regularization`, and factor-reuse period `reuse`.

# Fields

- `linsolve::L`: the linear-solver method. A [`LeastSquaresProblem`](@ref) takes neither
  [`LUFactorization`](@ref) nor [`LDLTFactorization`](@ref) outside the stacked
  [`LevenbergMarquardt`](@ref) step. With more residuals than unknowns it takes no factor in a
  lower precision than the element type of the iterate, because a Gauss–Newton step with such a
  factor stops off the least-squares point; a factor in the same or a higher precision is taken,
  and the stacked `LevenbergMarquardt` step, which reads no factor, takes any. Each refusal raises
  an `ArgumentError` where the state is built. A factor in a higher precision on more residuals
  than unknowns is not promoted to the working precision after a stall: at a nonzero-residual
  least-squares point ``\\|F + J d\\| \\approx \\|F\\|``, which on a square problem reads as a stall.
  It is promoted where the factorisation fails.
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
- `refinement::RF`: the refinement setting ([`NoRefinement`](@ref) or [`Refinement`](@ref)).
  A `Refinement` refines the direction of the system that the factor solves, the regularised
  ``(J + μ I) d = -F`` when `μ > 0`. A [`LevenbergMarquardt`](@ref) globalization takes no
  `Refinement`: a refinement corrects the solve of ``J d = -F``, which the damped step does not
  take, and the constructor raises an `ArgumentError`. A [`LeastSquaresProblem`](@ref) takes no
  `Refinement` either: a refinement drives the linear residual ``J d + F`` to zero, which a
  least-squares step with more residuals than unknowns does not reach, and the refusal raises an
  `ArgumentError` where the state is built.
- `regularization::R`: non-negative finite diagonal shift ``μ ≥ 0`` added as ``J + μ I`` before
  factorisation when `μ > 0`. A [`LeastSquaresProblem`](@ref) with more residuals than unknowns
  takes none, and `μ > 0` raises an `ArgumentError` where the state is built.
- `reuse::K`: `nothing` (`K = Nothing`, factorise at every step) or `Int32(k)` (`K = Int32`,
  reuse the factor for up to `k` consecutive steps, refreshing early after a `STALLED` step and,
  within the step, after a failed [`Refinement`](@ref) on a reused factor, which counts one
  step-rule failure). A line search along a reused factor that fails without a step is retried
  once, in the same step, from a fresh factor. A [`DogLeg`](@ref) or
  [`LevenbergMarquardt`](@ref) globalization takes no reuse: its model needs the Jacobian of the
  current iterate, and the constructor raises an `ArgumentError`.
"""
struct Newton{L, G, J, RF, R <: Real, K <: Union{Nothing, Int32}}
    linsolve::L
    globalization::G
    jacobian::J
    refinement::RF
    regularization::R
    reuse::K
    # `adapt` rebuilds a method through this constructor, inside a device kernel too, so the check
    # also refuses a value that `ToReal` narrows to `Inf` or to zero. A kernel cannot format a
    # message, so each message is constant.
    function Newton{L, G, J, RF, R, K}(
            linsolve::L, globalization::G, jacobian::J,
            refinement::RF, regularization::R,
            reuse::K) where {
            L, G, J, RF, R <: Real, K <: Union{Nothing, Int32}}
        regularization >= zero(R) && isfinite(regularization) ||
            throw(ArgumentError("Newton regularization must be finite and ≥ 0"))
        if reuse !== nothing
            reuse >= Int32(1) ||
                throw(ArgumentError("Newton reuse must be ≥ 1 or nothing"))
            globalization isa DogLeg &&
                throw(ArgumentError("Newton with a DogLeg globalization takes no reuse"))
            globalization isa LevenbergMarquardt &&
                throw(ArgumentError("Newton with a LevenbergMarquardt globalization takes no reuse"))
        end
        globalization isa LevenbergMarquardt && refinement isa Refinement &&
            throw(ArgumentError("Newton with a LevenbergMarquardt globalization takes no Refinement: a Refinement corrects the solve of J d = -F, which the damped step does not take; use refinement = NoRefinement()"))
        if linsolve isa GMRES
            reuse === nothing ||
                throw(ArgumentError("Newton with GMRES takes no factor reuse: a GMRES step has no factor"))
            refinement isa Refinement &&
                throw(ArgumentError("Newton with GMRES takes no Refinement: a GMRES step has no factor to refine against"))
            if globalization isa DogLeg
                linsolve.restart == linsolve.maxiter ||
                    throw(ArgumentError("Newton with a DogLeg globalization needs unrestarted GMRES: restart must equal maxiter"))
                linsolve.preconditioner === nothing ||
                    throw(ArgumentError("Newton with a DogLeg globalization takes a GMRES without a preconditioner"))
            end
            globalization isa LevenbergMarquardt &&
                throw(ArgumentError("Newton with a LevenbergMarquardt globalization needs a Jacobian matrix, which a GMRES solve does not form"))
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
        reuse::Union{Nothing, Integer} = nothing,
        forcing::Union{Nothing, AbstractForcingTerm} = nothing,
        preconditioner = nothing)
    reg = float(regularization)
    r = reuse === nothing ? nothing : Int32(reuse)
    ref = refinement isa Bool ? (refinement ? Refinement() : NoRefinement()) : refinement
    if forcing !== nothing || preconditioner !== nothing
        linsolve isa GMRES || throw(ArgumentError(
            "the forcing and preconditioner keywords of Newton configure a GMRES linear method; pass them to GMRES, or set linsolve = GMRES(...)"))
        g = linsolve
        f = forcing === nothing ? g.forcing : forcing
        P = preconditioner === nothing ? g.preconditioner : preconditioner
        linsolve = GMRES{typeof(f), typeof(P), g.jvp_eltype}(
            g.maxiter, g.restart, f, P, g.jvp_eltype)
    end
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

@inline _has_analytic_jacobian(prob::NonlinearProblem) = prob.J !== nothing
@inline _has_analytic_jacobian(prob::LeastSquaresProblem) = prob.J !== nothing
@inline _has_analytic_jacobian(::AbstractNonlinearProblem) = false

# The JVP preparation from the analytic Jacobian of `prob`: the `m × n` buffer of the residual
# `Fx`, or one like the Jacobian prototype of `prob`, which a line search and a `Refinement` along a
# reused factor read.
@inline function prototype_jvp(x, Fx, prob)
    P = jacobian_prototype(prob)
    return JacobianMatrixJVP(
        P === nothing ? buffer(x, eltype(x), length(Fx), length(x)) : similar(P, eltype(x))
    )
end

@inline function jacobian_matrix_jvp(g, x, Fx, prob)
    return jacobian_matrix_jvp(g, x, Fx)
end
@inline jacobian_matrix_jvp(::LineSearch, x, Fx, prob) = prototype_jvp(x, Fx, prob)
@inline jacobian_matrix_jvp(::Static, x, Fx, prob) = nothing

@inline function newton_ad_preps(m::Newton, prob::AbstractNonlinearProblem, Fx, x, p)
    if m.refinement isa Refinement && m.jacobian isa AbstractADType &&
       dense_ad(m.jacobian) isa AutoFiniteDiff
        throw(ArgumentError("Refinement requires an exact or forward-mode Jacobian-vector product; AutoFiniteDiff is not supported"))
    end
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
            if Fx === nothing
                nothing
            elseif m.refinement isa Refinement && m.reuse !== nothing
                prototype_jvp(x, Fx, prob)
            else
                jacobian_matrix_jvp(m.globalization, x, Fx, prob)
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

@inline function empty_static_factor(ls::QRFactorization{P},
        ::StaticVector{N, T}, ::StaticVector{M}) where {P, M, N, T}
    M == N && return empty_static_factor(ls, zero(SVector{N, T}))
    TF = factor_eltype(ls, T)
    z = zeros(SMatrix{M, N, TF})
    τ = zeros(SVector{N, TF})
    return StaticRectQR{M, N, TF, M * N}(z, τ, Int32(0), Int32(0))
end

@inline function empty_static_factor(ls::SVDFactorization{P},
        ::StaticVector{N, T}, ::StaticVector{M}) where {P, M, N, T}
    M == N && return empty_static_factor(ls, zero(SVector{N, T}))
    TF = factor_eltype(ls, T)
    zu = zeros(SMatrix{M, N, TF})
    zv = zeros(SMatrix{N, N, TF})
    σ = zeros(SVector{N, real(TF)})
    return StaticRectSVD{M, N, TF, M * N, N * N, real(TF)}(
        zu, zv, σ, σ, Int32(0), Int32(0), Int32(0))
end

@inline function empty_static_factor(
        ls::Union{QRFactorization, SVDFactorization}, ::T,
        ::StaticVector{M}) where {M, T <: Number}
    return empty_static_factor(ls, SVector{1, T}((zero(T),)), zero(SVector{M, T}))
end

@inline empty_static_factor(ls, x, Fx) = empty_static_factor(ls, x)

"""
    jacobian_pattern(prob, jacprep)

Return the Jacobian sparsity prototype for `prob` and `jacprep`: `prob.jacobian_prototype` for a
[`NonlinearProblem`](@ref), and `nothing` for any other problem and for `nothing`. The
`GeometricSolversSparseADExt` extension adds a method for a [`NonlinearProblem`](@ref) with no
prototype and a [`DIJacobian`](@ref) of an `AutoSparse` back end, which returns the pattern that
back end's sparsity detector found; a back end with no detector finds only a pattern that is true
everywhere, so that method returns `nothing` and the dense path runs.
"""
@inline jacobian_pattern(prob::NonlinearProblem, jacprep) = prob.jacobian_prototype
@inline jacobian_pattern(::AbstractNonlinearProblem, jacprep) = nothing
@inline jacobian_pattern(::Nothing, jacprep) = nothing

@inline function init_factor(
        ls::LinearMethod, prob, x::Union{StaticVector, Number}, p, jacprep)
    return empty_static_factor(ls, x)
end

"""
    has_sparse_factor(method::LinearMethod, x)

Whether a package extension builds a sparse factor of `method` on the iterate `x` from a detected
Jacobian pattern: `false` by default. The `GeometricSolversSparspakExt` extension returns `true`
for an [`LUFactorization`](@ref) whose factor element type is `T` on a `Vector{T}`,
`T <: AbstractFloat`. Where it is `false`, a detected pattern gives the dense factor, as a solve
with no pattern does; an explicit `jacobian_prototype` does not consult it.
"""
@inline has_sparse_factor(::LinearMethod, x) = false

# The pattern that the factor of `ls` on `x` is built on: an explicit prototype as it is, and a
# detected pattern only where `has_sparse_factor` holds, else `nothing` and the dense factor. A
# `LeastSquaresProblem` has neither, because `jacobian_pattern` is `nothing` for it: the one sparse
# factor, `SparspakLU`, is square, so the `m × n` factor of a least-squares problem is dense.
@inline function factor_pattern(ls, prob, x, jacprep)
    P = jacobian_pattern(prob, jacprep)
    P === nothing || jacobian_prototype(prob) !== nothing || has_sparse_factor(ls, x) ||
        return nothing
    return P
end

@inline function init_factor(ls::LinearMethod, prob, x::AbstractVector, p, jacprep)
    return init_linear(ls, factor_pattern(ls, prob, x, jacprep), x, factor_rows(prob, x))
end

@inline has_structural_diagonal(::Nothing) = true
@inline function has_structural_diagonal(P::SparseArrays.SparseMatrixCSC)
    rows = SparseArrays.rowvals(P)
    return all(j -> j in view(rows, SparseArrays.nzrange(P, j)), axes(P, 2))
end

# The pattern of the factors of a Newton solve: the `factor_pattern` of `m.linsolve`, whose factor
# element type decides it also for the factor in the working precision that a mixed-precision
# solve promotes to, so that both factors and the Jacobian buffer share one layout. `regularize!!`
# writes `μ` onto the diagonal of `J`, so under `μ ≠ 0` a detected pattern without the whole
# structural diagonal gives the dense factor. A `LevenbergMarquardt` step reads no sparse factor
# (see `check_lm_iterate`), so a detected pattern gives it the dense factor and Jacobian buffer.
@inline function newton_pattern(m::Newton, prob, x, jacprep)
    P = factor_pattern(m.linsolve, prob, x, jacprep)
    m.globalization isa LevenbergMarquardt && jacobian_prototype(prob) === nothing &&
        return nothing
    iszero(m.regularization) || jacobian_prototype(prob) !== nothing ||
        has_structural_diagonal(P) || return nothing
    return P
end

# The factor of `ls` for a Newton solve with `m`: `m.linsolve`, or its promotion to the working
# precision. A pattern that the `init_factor` of `ls` would take and `newton_pattern` rejects gives
# the dense factor, and every other case the `init_factor` of `ls`.
@inline function init_newton_factor(
        m::Newton, ls, prob, x::Union{StaticVector, Number}, p, jacprep)
    return init_factor(ls, prob, x, p, jacprep)
end

@inline function init_newton_factor(m::Newton, ls, prob, x::AbstractVector, p, jacprep)
    newton_pattern(m, prob, x, jacprep) === nothing &&
        factor_pattern(ls, prob, x, jacprep) !== nothing &&
        return init_linear(ls, nothing, x, factor_rows(prob, x))
    return init_factor(ls, prob, x, p, jacprep)
end

# The Jacobian buffer of the step, `m × n` for the residual `Fx`. It is the matrix of a factor,
# which the factorisation overwrites, unless the step reads the Jacobian after it: a `DogLeg` or
# `LevenbergMarquardt` trial loop and a `Refinement` multiply by it.
@inline function init_newton_J(m::Newton, fac, fac_T, x, Fx = x)
    if !(m.globalization isa Union{DogLeg, LevenbergMarquardt}) &&
       m.refinement isa NoRefinement
        if factor_eltype(m.linsolve, eltype(x)) === eltype(x)
            return fac.factors
        elseif fac_T !== nothing && m.linsolve isa LUFactorization
            return fac_T.factors
        end
    end
    if fac.factors isa SparseArrays.SparseMatrixCSC
        return similar(fac.factors, eltype(x))
    end
    return buffer(x, eltype(x), length(Fx), length(x))
end

# The workspace of a globalization with the residual `Fx` beside the iterate `x`: the residual-
# shaped slots take the length of `Fx`, which a `LeastSquaresProblem` makes longer than `x`. A
# globalization without a method of its own here has the square workspace of `init_globalization`
# with the linear method.
@inline function init_globalization(::Static, ::LinearMethod, x, Fx)
    return (; xt = buffer(x), Ft = buffer(Fx))
end
@inline function init_globalization(::LineSearch, ::LinearMethod, x, Fx)
    return (; xt = buffer(x), Ft = buffer(Fx), Jv = buffer(Fx))
end
@inline function init_globalization(::DogLeg, ::LinearMethod, x, Fx)
    s = buffer(x)
    Jg = buffer(Fx)
    square = length(Fx) == length(x)
    return (; g̃ = buffer(x), ĝ = buffer(x), Jg, s, xt = buffer(x), Ft = buffer(Fx),
        u = square ? s : buffer(Fx), b = square ? Jg : buffer(x))
end
@inline init_globalization(g, linsolve::LinearMethod, x, Fx) = init_globalization(
    g, linsolve, x)

# The rule state of `Newton` with `reuse = k`: the age of the factor, and whether the last step
# stalled, which refreshes the factor early.
struct FactorAge
    age::Int32
    stalled::Bool
end

# The age of a factor that a step has just factorised: the next step reuses it unless the period
# is one step long.
@inline fresh_age(m::Newton) = m.reuse == Int32(1) ? Int32(0) : Int32(1)

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
# `finish!!` copies into `SolverStatus`, together with `jnorm = ‖J‖_F` for the stationarity
# stopping test on a `LeastSquaresProblem`.
struct NewtonDiagnostics{R <: Real}
    rank::Int32
    nullnorm::R
    gradratio::R
    jnorm::R
end

@inline function NewtonDiagnostics(::Type{R}) where {R <: Real}
    return NewtonDiagnostics{R}(Int32(-1), R(NaN), R(NaN), zero(R))
end

# The state of `Newton`: the AD preparations, the residual, direction, factor, Jacobian and
# globalization workspace, the rule state of its reuse, its `DogLeg` or its `LevenbergMarquardt`,
# the merit `norm2(Fx)` of the current residual for a line search (`nothing` for any other
# globalization), the norm `‖Fx‖` of the current residual, the diagnostics of the last
# factorisation, the refinement buffer (`nothing` for `NoRefinement` or R1), the working-precision
# factor slot for mixed-precision promotion (`nothing` where the factorisation runs in the working
# precision, and for `LevenbergMarquardt`, which promotes nothing), the promotion flag, the
# count of consecutive slow steps a promotion waits for, and the linear-solver workspace, which
# only a [`GMRES`](@ref) solve has (`nothing` for every factorising method). For a value iterate the
# workspace slots are `nothing`, and so is the factor slot unless the factor is reused. The factor
# slot is the storage that each factorisation writes into; only a step with reuse stores the factor
# it returns, because only such a step reads it again. The norm is `NaN` on a state that no
# `start!!` saw or that `solve!!` returned.
struct NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG, RB, FT, WS}
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
    ws::WS
end

# The state of a factorising method, whose linear solve keeps no workspace of its own: the fifteen
# fields that every `Newton` state held before `GMRES`.
@inline function NewtonState(jacprep::JP, jvpprep::VP, Fx::FX, d::D, fac::FAC, J::JM, g::G,
        rule::RS, merit::PH, fnorm::R, diag::DG, refbuf::RB, fac_T::FT, promoted::Bool,
        slow_steps::Int32) where {JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG, RB, FT}
    return NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG, RB, FT, Nothing}(
        jacprep, jvpprep, Fx, d, fac, J, g, rule, merit, fnorm, diag, refbuf, fac_T,
        promoted, slow_steps, nothing)
end

@inline state_gradratio(state::NewtonState) = state.diag.gradratio
@inline state_jnorm(state::NewtonState) = state.diag.jnorm

# The state whose diagnostics measure the stationarity test at `(x, Fx)`: `solve!!` re-measures
# them at the iterate it returns, because a step's diagnostics are those of the last fresh
# factorisation, at the start of that step. Every other method and problem keeps its reading.
@inline refresh_diagnostics!!(state, method, prob, x, Fx, p) = state
@inline function refresh_diagnostics!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing}, prob::LeastSquaresProblem,
        x, Fx, p) where {L, G, J, RF, Reg}
    R = real(eltype(x))
    _, _, diag = if m.globalization isa LevenbergMarquardt
        lm_newton_factor(m, state, prob, x, Fx, p, R(rnorm(Fx)))
    else
        newton_factor(m, state, prob, x, Fx, p)
    end
    return with_diag(state, diag)
end

# Under factor reuse the refresh's factorisation is also the factor that a later step reuses, so
# the state keeps it, as the retry path of `step!!` keeps the factor it recurs to. R1 drops a
# factor that no field holds and R3 writes into the factor's storage, so without the stored factor
# and its fresh age the two regimes continue from factors of different iterates.
@inline function refresh_diagnostics!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32}, prob::LeastSquaresProblem,
        x, Fx, p) where {L, G, J, RF, Reg}
    _, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    state = with_diag(with_factor(state, fac), diag)
    return with_rule(state, FactorAge(fresh_age(m), false))
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws)
end

@inline function with_factor_T(s::NewtonState, fac_T)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, fac_T, s.promoted, s.slow_steps, s.ws)
end

@inline function update_stored_factor(
        s::NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG, RB, Nothing},
        fac, promoted::Bool) where {JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG, RB}
    return promoted ? s : with_factor(s, fac)
end

@inline function update_stored_factor(s::NewtonState, fac, promoted::Bool)
    return promoted ? with_factor_T(s, fac) : with_factor(s, fac)
end

@inline function with_rule(s::NewtonState, rule)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, rule, s.merit, s.fnorm, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws)
end

# The merit of the current residual and its norm: the two values a step returns, or that a solve
# starts from.
@inline function with_merit(s::NewtonState, merit, fnorm)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, merit,
        oftype(s.fnorm, fnorm), s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws)
end

@inline function with_diag(s::NewtonState, diag)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws)
end

@inline function with_mixed(s::NewtonState, promoted::Bool, slow_steps::Int32)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, s.fac_T, promoted, slow_steps, s.ws)
end

# ─── A `Number` iterate on the one-entry vector path ─────────────────────────
#
# The steps, factorisations and line searches of a least-squares problem run on a vector iterate,
# so a one-variable least-squares problem is solved as the one-entry `SVector` problem it is: the
# solver lifts the iterate once, where it starts, and takes the result back to a `Number` at
# return. A residual and every Jacobian source enter that path through `ScalarResidual` and
# `ScalarJacobian`, which read the `Number` out of the one-entry iterate and lift what the caller's
# function returns.

# `ScalarResidual` is the residual `F(x, p)` of a one-variable problem, called on the one-entry
# iterate `x` the vector path carries: `(x, p) -> lift_residual(F(first(x), p))`, which
# `lift_scalar_iterate` builds.
struct ScalarResidual{F}
    F::F
end

@inline (r::ScalarResidual)(x::SVector{1}, p) = lift_residual(r.F(first(x), p))

# `ScalarJacobian` is the Jacobian `J(x, p)` of a one-variable problem, called on the one-entry
# iterate `x`: its value is the `M × 1` column of the `M`-entry residual's derivative, whether `J`
# returns that column, the one-entry matrix, or — for `M = 1` — the derivative as a `Number`.
# `lift_scalar_iterate` builds it, and so does `lift_scalar_operator` for the Jacobian operator of a
# method.
struct ScalarJacobian{J}
    J::J
end

@inline (J::ScalarJacobian)(x::SVector{1}, p) = lift_jacobian(J.J(first(x), p))

# The one-entry forms: a scalar residual is a one-entry vector, and the derivative of an `M`-entry
# residual along the one variable is its column, or — for a one-variable problem written with a
# `Number` residual — the derivative itself.
@inline lift_residual(r::Number) = SVector{1, typeof(r)}((r,))
@inline lift_residual(r::StaticVector) = r
@inline lift_jacobian(J::StaticMatrix) = J
@inline lift_jacobian(J::StaticVector{M, T}) where {M, T} = SMatrix{M, 1, T, M}(J)
@inline lift_jacobian(J::Number) = SMatrix{1, 1, typeof(J), 1}((J,))

# `lift_scalar_iterate` gives the problem, the method and the iterate of a `Number` iterate on the
# one-entry vector path: a `LeastSquaresProblem` whose residual and Jacobian take an `SVector{1}`
# and return the lifted residual and derivative, and a `Newton` method whose own Jacobian operator
# takes that iterate. Everything else about both is unchanged. `init_state` and `solve!!` lift with
# it, so that no step, factorisation or line search needs a `Number` method of its own.
@inline function lift_scalar_iterate(prob::LeastSquaresProblem, m::Newton, x::Number)
    F = ScalarResidual(prob.F)
    J = prob.J === nothing ? nothing : ScalarJacobian(prob.J)
    return LeastSquaresProblem(F, J, prob.output_length), lift_scalar_operator(m),
    SVector{1, typeof(x)}((x,))
end

# The Jacobian operator of a method takes the problem's signature, so it takes the one-entry
# iterate here. A back end differentiates the lifted residual unchanged.
@inline function lift_scalar_operator(m::Newton)
    j = m.jacobian
    (j === nothing || j isa AbstractADType) && return m
    return Newton(m.linsolve, m.globalization, ScalarJacobian(j), m.refinement,
        m.regularization, m.reuse)
end

# The state of a one-variable least-squares problem is the state of the lifted problem at the
# lifted iterate.
@inline function init_state(m::Newton, prob::LeastSquaresProblem, x::Number, p)
    prob, m, x = lift_scalar_iterate(prob, m, x)
    return init_state(m, prob, x, p)
end

# The diagonal shift `J + μ I` of `regularization = μ > 0` needs a square Jacobian, so a
# `LeastSquaresProblem` with `m_rows > n` residuals refuses it.
@inline function check_regularization(m::Newton, m_rows::Int, n::Int)
    if m_rows > n && m.regularization > zero(m.regularization)
        throw(ArgumentError(
            "Newton diagonal regularization requires a square Jacobian (m == n), got m = $m_rows, n = $n"))
    end
    return nothing
end

# A factor in a lower precision than the iterate perturbs the Jacobian that the step solves with.
# A square Newton solve still reaches the root, because it reads the residual in the working
# precision, but a Gauss–Newton step on `m_rows > n` residuals stops where the perturbed `Jᴴ F`
# vanishes, which is not the least-squares point. Such a factor is refused wherever it computes
# the step; the stacked `LevenbergMarquardt` step reduces `[J | -F]` in the working precision and
# reads no factor. A factor in a higher precision perturbs the Jacobian below the working
# precision, and is taken.
@inline function check_lsq_factor_eltype(m::Newton, m_rows::Int, x)
    T = eltype(x)
    F = factor_eltype(m.linsolve, T)
    (m_rows > length(x) && F !== T && eps(real(F)) > eps(real(T))) || return nothing
    m.globalization isa LevenbergMarquardt &&
        !lm_svd_path(m.linsolve, m.globalization.scaling) && return nothing
    throw(ArgumentError(
        "a LeastSquaresProblem with more residuals than unknowns (m = $m_rows, n = $(length(x))) takes no factor in a lower precision than the element type $T of the iterate; a factor in $F moves the Gauss–Newton fixed point off the least-squares point"))
end

@inline function init_state(m::Newton, prob::AbstractNonlinearProblem, x, p)
    m.linsolve isa GMRES && return init_gmres_state(m, m.linsolve, prob, x, p)
    if prob isa LeastSquaresProblem &&
       m.linsolve isa Union{LUFactorization, LDLTFactorization} &&
       !(m.globalization isa LevenbergMarquardt)
        throw(ArgumentError(
            "LeastSquaresProblem does not support $(nameof(typeof(m.linsolve))); use QRFactorization() or SVDFactorization()"))
    end
    # A `Refinement` drives the linear residual `J d + F` to zero, which a least-squares step with
    # more residuals than unknowns does not reach (known issue K80)
    if prob isa LeastSquaresProblem && m.refinement isa Refinement
        throw(ArgumentError("LeastSquaresProblem does not support Refinement"))
    end
    m.globalization isa LevenbergMarquardt &&
        check_lm_iterate(m.globalization, m.linsolve, prob, x)
    Fx = if x isa Union{StaticArray, Number}
        nothing
    elseif prob isa LeastSquaresProblem
        m_rows = output_length(prob, length(x))
        check_regularization(m, m_rows, length(x))
        check_lsq_factor_eltype(m, m_rows, x)
        buffer(x, eltype(x), m_rows)
    else
        buffer(x)
    end
    jacprep, jvpprep = newton_ad_preps(m, prob, Fx, x, p)
    rule = newton_rule(m, x)
    merit = init_merit(m.globalization, x)
    R = real(eltype(x))
    diag = NewtonDiagnostics(R)
    mixed = factor_eltype(m.linsolve, eltype(x)) !== eltype(x)
    if Fx === nothing
        if prob isa LeastSquaresProblem
            Fx0 = primal_residual!!(nothing, prob, x, p)
            check_static_residual_length(prob, Fx0, x)
            check_regularization(m, length(Fx0), length(x))
            check_lsq_factor_eltype(m, length(Fx0), x)
            fac = m.reuse === nothing ? nothing : empty_static_factor(m.linsolve, x, Fx0)
        else
            fac = m.reuse === nothing ? nothing :
                  init_factor(m.linsolve, prob, x, p, jacprep)
        end
        return NewtonState(
            jacprep, jvpprep, nothing, nothing, fac, nothing, nothing, rule, merit, R(NaN),
            diag, nothing, nothing, false, Int32(0), nothing)
    else
        fac = init_newton_factor(m, m.linsolve, prob, x, p, jacprep)
        fac_T = mixed && !(m.globalization isa LevenbergMarquardt) ?
                init_newton_factor(
            m, promote_linsolve(m.linsolve, eltype(x)), prob, x, p, jacprep) :
                nothing
        refbuf = init_refinement(m.refinement, x)
        return NewtonState(
            jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, fac_T, x, Fx),
            init_globalization(m.globalization, m.linsolve, x, Fx), rule, merit, R(NaN),
            diag, refbuf, fac_T, false, Int32(0), nothing)
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

# The per-solve reset that every `Newton` method shares: the rule state, the anchor merit of a
# line search, the promotion flags and the diagnostics.
@inline function newton_start!!(state::NewtonState, m::Newton, x, Fx, fnorm₀)
    state = with_rule(state, reset_rule(state.rule, m.globalization))
    state = with_merit(state, start_merit(m.globalization, Fx), fnorm₀)
    state = with_mixed(state, false, Int32(0))
    return with_diag(state, NewtonDiagnostics(real(eltype(x))))
end

@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    state = newton_start!!(state, m, x, Fx, fnorm₀)
    ws = gmres_workspace(state)
    ws === nothing && return state
    reset_gmres!(ws, fnorm₀)
    return state
end

# The state that `solve!!` returns carries the merit and the norm of `init_state` again. A step on
# it without `start!!` reads nothing that the solve left: a line-search step fails closed with
# `NONFINITE`, and any other step takes `‖Fx‖` from the residual it is given. `finish!!` copies the
# diagnostics of the last factorisation into the returned `SolverStatus`; this is the close that
# every `Newton` method shares.
@inline function newton_finish!!(
        state::NewtonState, m::Newton, x, status::SolverStatus{R}) where {R}
    d = state.diag
    st = SolverStatus{R}(
        status.code, status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, status.stationary,
        d.rank, d.nullnorm, d.gradratio)
    state_end = with_mixed(
        with_merit(state, init_merit(m.globalization, x), NaN),
        state.promoted, Int32(0))
    if state.rule isa DampingParameter
        state_end = with_rule(state_end, DampingParameter(R(NaN)))
    end
    return x, st, state_end
end

@inline function finish!!(
        state::NewtonState, m::Newton, prob, x, Fx, status::SolverStatus{R}) where {R}
    x₊, st, state_end = newton_finish!!(state, m, x, status)
    ws = gmres_workspace(state)
    ws === nothing && return x₊, st, state_end
    nf, nj = gmres_counts(ws)
    return x₊, with_calls(st, nf, nj), state_end
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

@inline function newton_stepkind(
        ::AbstractNonlinearProblem, m::Newton, ::Type{T}, fresh::Bool) where {T}
    return newton_stepkind(m, T, fresh)
end
@inline function newton_stepkind(::AbstractNonlinearProblem, m::Newton, ::Type{T},
        fresh::Bool, in_working_T::Bool) where {T}
    return newton_stepkind(m, T, fresh, in_working_T)
end
@inline function newton_stepkind(::LeastSquaresProblem, m::Newton, ::Type{T},
        fresh::Bool, in_working_T::Bool = false) where {T}
    return MeasuredSlope()
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
# `factorize!!` touches `J`, and `0` when `‖J‖_F` or `‖Fx‖` is `0`, returned together with
# `‖J‖_F`: the stationarity stopping test reads both. Where `Jᴴ Fx` overflows with both norms
# finite, it is computed again from `Fx` scaled by `2^{-e}`, with `2^e` the power of two of
# `‖Fx‖`, so that the scaled product is below `2‖J‖_F`; on an array iterate the scaled copy
# allocates. A mutable static `J` or `Fx` is read as an `SMatrix` or `SVector`, so that `Jᴴ Fx`
# allocates nothing.
@inline function grad_ratio_and_norm(d, J, Fx, fnorm::R) where {R <: Real}
    Jmat = J isa StaticMatrix ? SMatrix(J) : J
    Fv = Fx isa StaticVector ? SVector(Fx) : Fx
    jnorm = R(rnorm(Jmat))
    (iszero(jnorm) || iszero(fnorm)) && return zero(R), jnorm
    gnorm = R(rnorm(mul!!(d, adjoint(Jmat), Fv)))
    isfinite(gnorm) && return gnorm / jnorm / fnorm, jnorm
    (isfinite(jnorm) && isfinite(fnorm)) || return gnorm / jnorm / fnorm, jnorm
    e = exponent(fnorm)
    gscaled = R(rnorm(mul!!(d, adjoint(Jmat), Fv .* ldexp(one(R), -e))))
    return gscaled / jnorm / ldexp(fnorm, -e), jnorm
end

# The Jacobian at `x`, regularised, its factor in `TF`, and the diagnostics of both: the
# factorisation that every `Newton` step takes, and the one a promotion may follow.
@inline function newton_factor(m::Newton, state::NewtonState, prob, x, Fx, p)
    R = real(eltype(x))
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr, jnorm = grad_ratio_and_norm(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    fac = factorize!!(state.fac, m.linsolve, Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr, jnorm)
end

# The Jacobian at `x` and its factorisation in the working precision `T`, with the diagnostics of
# both: the one factorisation of a step whose factorisation already runs in `T`, because the
# solve was promoted to it.
@inline function newton_factor_T(m::Newton, state::NewtonState, prob, x, Fx, p)
    T = eltype(x)
    R = real(T)
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr, jnorm = grad_ratio_and_norm(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    ls_T = promote_linsolve(m.linsolve, T)
    fac = factorize!!(state.fac_T, ls_T, Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr, jnorm)
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
    return fac, NewtonDiagnostics{R}(rk, nn, diag.gradratio, diag.jnorm)
end

# The refined direction from the factor `fac`. A step that factorised at `x` passes the regularised
# Jacobian `Jreg` it factorised, and a `Refinement` multiplies by it; a step along a reused factor
# passes `nothing`, and a `Refinement` takes the JVP at `x`, shifted by `μ`.
@inline function newton_direction(
        m::Newton, state::NewtonState, prob, x, Fx, p, fac, Jreg)
    d₁ = newton_solve(state, fac, Fx)
    return refine!!(
        d₁, m.refinement, fac, HeldJacobianJVP(Jreg), prob, x, Fx, nothing, p, state.refbuf)
end

@inline function newton_direction(
        m::Newton, state::NewtonState, prob, x, Fx, p, fac, ::Nothing)
    d₁ = newton_solve(state, fac, Fx)
    μ = m.regularization
    iszero(μ) && return refine!!(
        d₁, m.refinement, fac, state.jvpprep, prob, x, Fx, nothing, p, state.refbuf)
    return refine!!(
        d₁, m.refinement, fac, RegularizedJVP(state.jvpprep, μ), prob, x, Fx, nothing, p,
        state.refbuf)
end

@inline newton_solve(state::NewtonState, fac, Fx) = map!!(-, state.d, ldiv!!(state.d, fac, Fx))

# The product with the regularised Jacobian that a step already holds: a `Refinement` of a step
# that factorised at `x` reads no Jacobian again.
struct HeldJacobianJVP{M}
    J::M
end

@inline jvp!!(Jv, h::HeldJacobianJVP, prob, x, v, p) = mul!!(Jv, h.J, v)

# The product with the regularised Jacobian `J + μ I` that `newton_factor` factorises: the JVP
# that a `Refinement` along a reused factor reads where `μ > 0`, so that it refines the direction
# of the system the factor solves.
struct RegularizedJVP{P, R <: Real}
    prep::P
    μ::R
end

@inline function jvp!!(Jv, r::RegularizedJVP, prob, x, v, p)
    T = eltype(x)
    Jv₀ = jvp!!(Jv, r.prep, prob, x, v, p)
    return axpby!!(Jv, one(T), Jv₀, T(r.μ), v)
end

# The step from the factor `fac` that the step factorised from `Jreg` at `x`: `SINGULAR` for a
# failed factor, else the globalized step along the refined direction.
@inline function newton_globalize(m::Newton, state::NewtonState, prob, x, Fx, p, fac, Jreg)
    R = real(eltype(x))
    if fac.info > Int32(0)
        return x, Fx, StepInfo{R}(SINGULAR, rnorm(Fx), zero(R), Int32(0), false, false),
        state.merit
    end
    d, _ = newton_direction(m, state, prob, x, Fx, p, fac, Jreg)
    return newton_line(m, state, prob, x, Fx, p, fac, d, true)
end

# The globalized step along the direction `d` from the factor `fac`, fresh or reused, with the
# measured slope where the solve of `fac` is a truncated one or the problem is a
# `LeastSquaresProblem`. It returns the merit of the residual it returns, from the merit of `Fx`
# that the state carries.
@inline function newton_line(
        m::Newton, state::NewtonState, prob, x, Fx, p, fac, d, fresh::Bool)
    sk = exact_solve(fac) ? newton_stepkind(prob, m, eltype(x), fresh) : MeasuredSlope()
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

# The reading of one mixed-precision step, from the relative linear-model residual `lin_rel` of
# the direction it took. A step whose linear model is inaccurate in `TF` (`lin_rel > η_lin`) and
# which lowers the residual by less than half counts toward a promotion to `T`, and two such steps
# promote. Any other step returns unchanged, as a step in the working precision is: the precision
# of `F` itself is not read here, because a step in `T` sees the same residual and keeps going.
@inline function read_mixed_stall(
        state::NewtonState, info::StepInfo{R}, fnorm_old::R, lin_rel::R) where {R <: Real}
    η_lin = max(sqrt(eps(R)), R(1e-4))
    if info.fnorm > R(0.5) * fnorm_old && lin_rel > η_lin
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

# The result of the mixed-precision half of a step: the direction, the regularised Jacobian the
# factorisation used, the state that carries the factor a reuse path reuses and the promotion
# flag, the relative linear-model residual of the direction, and the flags of the step. No factor
# crosses this boundary: `state.fac` holds the one in `TF` and `state.fac_T` the one in `T`, so no
# call site holds a value of either type. `lin_rel` is `0` on a step that reads no linear residual:
# one in `T` already, one that promoted, one along a reused factor, whose Jacobian is the state's
# own buffer, and one on a `LeastSquaresProblem` with more residuals than unknowns, where
# `‖F + J d‖ ≈ ‖F‖` at a nonzero-residual least-squares point and no step there is a stall.
struct MixedNewton{J, D, S, R <: Real}
    d::D
    J::J
    state::S
    lin_rel::R
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
# already promoted solves in `T` at every step. The linear-model residual is measured here, where
# the regularised Jacobian is at hand: a reused factor keeps no Jacobian of its own, and reads none.
@inline function mixed_newton!!(
        state::NewtonState, m::Newton, prob, x, Fx, fnorm_old::R, p,
        fresh::Bool, store::Bool) where {R <: Real}
    if state.promoted
        Jreg, fac, diag = newton_factor_T(m, state, prob, x, Fx, p)
        if fac.info > Int32(0)
            return MixedNewton(
                Fx, Jreg, with_diag(state, diag), zero(R), true, exact_solve(fac), false,
                Int32(0))
        end
        d, _ = newton_direction(m, state, prob, x, Fx, p, fac, Jreg)
        return MixedNewton(
            d, Jreg, with_diag(state, diag), zero(R), true, exact_solve(fac), true, Int32(0))
    end
    failures = Int32(0)
    if !fresh
        d, ref_failed = newton_direction(m, state, prob, x, Fx, p, state.fac, nothing)
        ref_failed || return MixedNewton(
            d, state.J,
            with_diag(stored_factor(state, state.fac, false, store), state.diag),
            zero(R), false, exact_solve(state.fac), true, failures)
    end
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    fresh || (failures += one(Int32))
    d, ref_failed = if fac.info > Int32(0)
        Fx, true
    else
        newton_direction(m, state, prob, x, Fx, p, fac, Jreg)
    end
    if ref_failed || fac.info > Int32(0)
        fac_T, diag_T = promote_factor!!(state, m, Jreg, x, Fx, fnorm_old, diag)
        failures += one(Int32)
        st = with_mixed(
            with_diag(stored_factor(state, fac_T, true, store), diag_T),
            true, state.slow_steps)
        if fac_T.info > Int32(0)
            return MixedNewton(
                Fx, Jreg, st, zero(R), true, exact_solve(fac_T), false, failures)
        end
        d₀ = ldiv!!(state.d, fac_T, Fx)
        return MixedNewton(
            map!!(-, state.d, d₀), Jreg, st, zero(R), true, exact_solve(fac_T), true, failures)
    end
    lin_rel = prob isa LeastSquaresProblem && length(Fx) > length(x) ? zero(R) :
              linear_residual_ratio(state, Jreg, Fx, d, fnorm_old)
    return MixedNewton(
        d, Jreg, with_diag(stored_factor(state, fac, false, store), diag), lin_rel,
        true, exact_solve(fac), true, failures)
end

# The step of a `Newton` whose factorisation runs in the working type `T`: one factorisation, one
# refined direction and one globalized step, with no mixed-precision state read.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    ws = gmres_workspace(state)
    ws === nothing || return gmres_step!!(state, ws, m, m.linsolve, prob, x, Fx, p)
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, Jreg)
    return x₊, F₊, with_diag(with_merit(state, merit, info.fnorm), diag), info
end

# The same step with a factorisation in `TF ≠ T`: the mixed-precision half of the step, the
# globalized direction it returns, and the reading of the linear-model residual it measured.
@inline function mixed_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    fnorm_old = R(residual_norm(Fx, state.merit, state.fnorm))
    md = mixed_newton!!(state, m, prob, x, Fx, fnorm_old, p, true, false)
    if !md.ok
        info = StepInfo{R}(
            SINGULAR, R(rnorm(Fx)), zero(R), md.failures, md.state.promoted, false)
        return x, Fx, md.state, info
    end
    promoted_step = md.state.promoted && !state.promoted
    sk = md.exact ? newton_stepkind(prob, m, T, true, md.state.promoted) : MeasuredSlope()
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, prob, x, Fx, md.d, p, state.jvpprep, sk, state.merit)
    info = bump_failures(info, md.failures)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm, info.failures, true, info.limited)
        st = with_mixed(md.state, true, Int32(0))
    elseif state.promoted
        st = md.state
    else
        st, info = read_mixed_stall(md.state, info, fnorm_old, md.lin_rel)
    end
    return x₊, F₊, with_merit(st, merit, info.fnorm), info
end

# The `DogLeg` step: one Jacobian, one factorisation and one Newton step, then the trial loop of
# `dogleg_step` on the regularised Jacobian that was factorised. A failed factor or a Newton step
# whose norm is zero or not finite leaves the Cauchy step.
@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    ws = gmres_workspace(state)
    ws === nothing || return gmres_step!!(state, ws, m, m.linsolve, prob, x, Fx, p)
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    d, newton = dogleg_direction(m, state, prob, x, Fx, p, fac, Jreg)
    x₊, F₊, radius, info = dogleg_step(
        state.g, prob, x, Fx, p, Jreg, d, newton, state.rule.radius)
    state₊ = with_rule(state, TrustRadius(radius, state.rule.radius₀))
    return x₊, F₊, with_diag(with_merit(state₊, state.merit, info.fnorm), diag), info
end

# The direction a `DogLeg` step tries first, and whether it is a Newton step at all: a singular
# factor leaves the Cauchy step, and so does a direction of zero or non-finite norm.
@inline function dogleg_direction(m::Newton, state, prob, x, Fx, p, fac, Jreg)
    fac.info > Int32(0) && return Fx, false
    d, _ = newton_direction(m, state, prob, x, Fx, p, fac, Jreg)
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
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G <: DogLeg, J, RF, Reg}
    R = real(eltype(x))
    fnorm_old = R(residual_norm(Fx, state.merit, state.fnorm))
    md = mixed_newton!!(state, m, prob, x, Fx, fnorm_old, p, true, false)
    promoted_step = md.state.promoted && !state.promoted
    d, newton = dogleg_direction(md, Fx)
    x₊, F₊, radius, info = dogleg_step(
        state.g, prob, x, Fx, p, md.J, d, newton, state.rule.radius)
    info = bump_failures(info, md.failures)
    if promoted_step
        info = StepInfo{R}(
            info.code, info.fnorm, info.stepnorm, info.failures, true, info.limited)
        st = with_mixed(md.state, true, Int32(0))
    elseif state.promoted
        st = md.state
    else
        st, info = read_mixed_stall(md.state, info, fnorm_old, md.lin_rel)
    end
    return x₊, F₊,
    with_merit(with_rule(st, TrustRadius(radius, state.rule.radius₀)), state.merit, info.fnorm),
    info
end

@inline function step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
    rule = state.rule
    age_fresh = fresh_age(m)
    failures = Int32(0)
    if !(rule.age == Int32(0) || rule.stalled)
        age₊ = rule.age + one(Int32) >= m.reuse ? Int32(0) : rule.age + one(Int32)
        # A reused factor whose `Refinement` fails is refactorised at `x`, within the step, and
        # counts one step-rule failure. A reused factor can also give a direction that does not
        # descend: a line search along it that fails without a step is retried once, in the same
        # step, from a fresh factor, and from the same residual and merit.
        d, ref_failed = newton_direction(m, state, prob, x, Fx, p, state.fac, nothing)
        if ref_failed
            failures = one(Int32)
        else
            x₊, F₊, info, merit = newton_line(m, state, prob, x, Fx, p, state.fac, d, false)
            (info.code == LINESEARCH_FAILED && iszero(info.stepnorm)) ||
                return reuse_result(state, state.fac, state.diag, x₊, F₊, info, merit, age₊)
        end
    end
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, Jreg)
    return reuse_result(
        state, fac, diag, x₊, F₊, bump_failures(info, failures), merit, age_fresh)
end

# The state and the result of a step with reuse from the factor `fac` it took: a `SINGULAR` step
# resets the age of the factor, and a `STALLED` one refreshes it at the next step.
@inline function reuse_result(state::NewtonState, fac, diag, x₊, F₊, info, merit, age₊)
    state = with_diag(with_merit(with_factor(state, fac), merit, info.fnorm), diag)
    info.code == SINGULAR &&
        return x₊, F₊, with_rule(state, FactorAge(Int32(0), false)), info
    return x₊, F₊, with_rule(state, FactorAge(age₊, info.code == STALLED)), info
end

# The same step with a factorisation in `TF ≠ T`: the mixed-precision half of the step, and the
# one retry of a line search along a reused factor that its own failure asks for.
@inline function mixed_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    T = eltype(x)
    R = real(T)
    fnorm_old = R(residual_norm(Fx, state.merit, state.fnorm))
    rule = state.rule
    fresh = state.promoted || rule.age == Int32(0) || rule.stalled
    age_fresh = fresh_age(m)
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
        lin_rel = md.lin_rel
        sk = md.exact ? newton_stepkind(prob, m, T, md.fresh, st.promoted) : MeasuredSlope()
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
                info.code, info.fnorm, info.stepnorm, info.failures, true, info.limited)
            st = with_mixed(st, true, Int32(0))
        elseif md.fresh && !st.promoted
            st, info = read_mixed_stall(st, info, fnorm_old, lin_rel)
        end
        st = with_merit(st, merit, info.fnorm)
        if info.code == SINGULAR
            return x₊, F₊, with_rule(st, FactorAge(Int32(0), false)), info
        end
        return x₊, F₊, with_rule(st, FactorAge(age₊, info.code == STALLED)), info
    end
end

@inline function lm_svd_factor(state, m, ::Number, Jreg)
    return factorize!!(state.fac, m.linsolve, SMatrix{1, 1, typeof(Jreg)}((Jreg,)))
end
@inline lm_svd_factor(state, m, _, Jreg) = factorize!!(state.fac, m.linsolve, Jreg)

@inline function lm_newton_factor(
        m::Newton, state::NewtonState, prob, x, Fx, p, fnorm::R) where {R}
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr, jnorm = grad_ratio_and_norm(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization)
    if lm_svd_path(m.linsolve, m.globalization.scaling)
        fac = lm_svd_factor(state, m, x, Jreg)
        rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
        return Jreg, fac, NewtonDiagnostics{R}(rk, nn, gr, jnorm)
    end
    return Jreg, state.fac, NewtonDiagnostics{R}(Int32(-1), R(NaN), gr, jnorm)
end

# The `LevenbergMarquardt` step. It has no mixed-precision half: the stacked step reduces the
# Jacobian in the working precision and reads no factor, and the SVD step reads a factor in `TF`
# for the damped direction and measures the gain ratio in `T`, so no step of it promotes.
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

# ─── R5 matrix-free Newton–Krylov ────────────────────────────────────────────
#
# A `Newton` whose linear method is a `GMRES` forms no Jacobian: the step solves `J d = -F` with a
# Jacobian–vector product, to the relative tolerance of the forcing term, and globalizes the
# direction under the step kind `InexactStep(η_ach)` of Eisenstat and Walker (1996, p. 24). The
# linear-solver workspace of the state is its `GMRESWorkspace`, and it is `nothing` for every
# factorising method. The hooks route to these bodies on that workspace and not on the type of the
# linear method: a method on `Newton{<:GMRES}` is ambiguous with the ones on a `DogLeg` or a
# `LevenbergMarquardt` globalization, which constrain another argument of the same signature.

@inline gmres_workspace(state::NewtonState) = state.ws

# The two call counts in one method whose argument is a workspace, so that the field access is typed.
@inline gmres_counts(ws::GMRESWorkspace) = (ws.f_calls, ws.jvp_calls)

# The per-solve reset of the R5 workspace. `solve!!` has evaluated `F(x₀)` once before `start!!`,
# which is the first of the residual evaluations that the workspace counts.
@inline function reset_gmres!(ws::GMRESWorkspace{T, R}, fnorm₀) where {T, R}
    ws.f_calls = Int32(1)
    ws.jvp_calls = Int32(0)
    ws.gmres_iterations = Int32(0)
    ws.max_inner = Int32(0)
    ws.arnoldi = Int32(0)
    ws.step = Int32(0)
    ws.prev_fnorm = R(fnorm₀)
    ws.last_eta = zero(R)
    ws.linres = zero(R)
    ws.beta = R(fnorm₀)
    return nothing
end

# The `DogLeg` workspace of an R5 solve: the matrix-free trial loop forms the trial step, the trial
# iterate and its residual, and reads no product buffer.
@inline function init_globalization(::DogLeg, ::GMRES, x, Fx)
    return (; s = buffer(x), xt = buffer(x), Ft = buffer(Fx))
end

# The Jacobian–vector preparation of an R5 solve: the caller's own product, or the product-only
# preparation of the AD seam, which is the chunked forward mode on every array type and
# DifferentiationInterface's pushforward on a host array.
@inline function newton_jvp_prep(m::Newton, prob, Fx, x::AbstractVector, p)
    j = m.jacobian
    j isa JVPFunction && return j
    backend = j === nothing ? AutoForwardDiff() : j
    backend isa AbstractADType || throw(ArgumentError(
        "a Newton with GMRES takes jacobian = nothing, an AbstractADType or a JVPFunction: a Jacobian function fills a matrix, and a GMRES solve forms none"))
    return prepare_jvp(backend, prob, Fx, x, p)
end

# The state of a `Newton` with `GMRES` at the iterate `x`: no Jacobian matrix, no factor and no
# factorization slot, a Jacobian–vector preparation, the workspace of the globalization, and the
# linear-solver workspace that every other body reads.
@inline function init_gmres_state(m::Newton, g::GMRES{F, P, TJ},
        prob::AbstractNonlinearProblem, x, p) where {F, P, TJ}
    (x isa AbstractVector && !(x isa StaticArray)) || throw(ArgumentError(
        "a Newton with GMRES solves an array linear system: the iterate must be an AbstractVector and not a value iterate, which is a Number or a StaticArray"))
    prob isa NonlinearProblem || throw(ArgumentError(
        "a Newton with GMRES solves the square system of a NonlinearProblem; a LeastSquaresProblem takes QRFactorization() or SVDFactorization()"))
    T = eltype(x)
    R = real(T)
    Fx = buffer(x)
    ws = init_gmres_workspace(g, x)
    prep = newton_jvp_prep(m, prob, Fx, x, p)
    iszero(m.regularization) || (prep = RegularizedJVP(prep, m.regularization))
    return NewtonState(
        nothing, r5_jvp(prep, ws, g), Fx, ws.d, nothing, nothing,
        init_globalization(m.globalization, g, x, Fx), newton_rule(m, x),
        init_merit(m.globalization, x), R(NaN), NewtonDiagnostics(R),
        init_refinement(m.refinement, x), nothing, false, Int32(0), ws)
end

# The Jacobian–vector preparation of the solve with its counters, and the `jvp_eltype` coupling
# where the method names one: `TJ` is a type parameter here, so the choice is a constant of the
# types and no branch reaches the inner loop.
@inline function r5_jvp(prep, ws, ::GMRES{F, P, TJ}) where {F, P, TJ}
    return TJ === Nothing ? R5JVP(prep, ws) : R5JVP{typeof(prep), typeof(ws), TJ}(prep, ws)
end

# The direction of one R5 step: the forcing term of the step, the refresh of the right
# preconditioner, and the inner solve. It returns whether the residual was finite, the attained
# relative tolerance `η_ach`, the step-rule failures of the inner solve, and `‖F_k‖`.
@inline function gmres_direction!(
        state::NewtonState, g::GMRES{F, P, TJ}, ws::GMRESWorkspace,
        prob, x, Fx, p) where {F, P, TJ}
    T = eltype(x)
    R = real(T)
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    isfinite(fnorm) || return false, zero(R), Int32(0), fnorm
    η = R(compute_forcing(g.forcing, ws.step, fnorm, ws.prev_fnorm, ws.last_eta, ws.linres))
    refresh!(g.preconditioner, prob, x, Fx, p)
    map!!(-, ws.bb, Fx)
    op = R5Operator(state.jvpprep, prob, x, p)
    _, info = gmres_solve!(
        ws, op, g.preconditioner, ws.bb, fnorm, η, g.maxiter, g.restart)
    ws.gmres_iterations = info.iterations
    ws.max_inner = max(ws.max_inner, info.iterations)
    ws.linres = info.resnorm
    ws.last_eta = η
    ws.prev_fnorm = fnorm
    ws.beta = fnorm
    ws.step += one(Int32)
    failures = info.iterations >= g.maxiter && !info.converged ? Int32(1) : Int32(0)
    η_ach = iszero(fnorm) ? zero(R) : min(info.resnorm / fnorm, η)
    return true, R(η_ach), failures, fnorm
end

# The step of a `Newton` with `GMRES`: the inner solve, the refined direction, and the globalization
# of the direction. A line search gets the inexact step kind `InexactStep(η_ach)`, whose test is
# `‖F(x + α d)‖ ≤ [1 - c₁(1 - η(α))] ‖F‖` with `η(α) = |1 - α| + α η_ach` (§5.1); a `DogLeg` runs
# its matrix-free trial loop on the Arnoldi relation of that same solve.
@inline function gmres_step!!(state::NewtonState, ws::GMRESWorkspace,
        m::Newton, g::GMRES{F, P, TJ}, prob, x, Fx, p) where {F, P, TJ}
    R = real(eltype(x))
    ok, η_ach, failures, fnorm = gmres_direction!(state, g, ws, prob, x, Fx, p)
    ok ||
        return x, Fx, state, StepInfo{R}(NONFINITE, fnorm, zero(R), failures, false, false)
    d, _ = refine!!(
        ws.d, m.refinement, nothing, state.jvpprep, prob, x, Fx, fnorm, p, state.refbuf)
    if m.globalization isa DogLeg
        nd = R(rnorm(d))
        newton = isfinite(nd) && !iszero(nd)
        x₊, F₊, radius, info = dogleg_step_mf(
            ws, state.g, prob, x, Fx, p, d, newton, state.rule.radius)
        state₊ = with_rule(state, TrustRadius(radius, state.rule.radius₀))
        info = bump_failures(info, failures)
        return x₊, F₊, with_merit(state₊, state.merit, info.fnorm), info
    end
    counted = counting_problem(prob, ws)
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, counted, x, Fx, d, p, state.jvpprep,
        InexactStep(η_ach), state.merit)
    info = bump_failures(info, failures)
    return x₊, F₊, with_merit(state, merit, info.fnorm), info
end

# The call counts of a solve, from the seed of `solve`: `max_inner_iterations` is the largest inner
# count of any Newton step, at most the `maxiter` of the linear method.
@inline max_inner_iterations(state::NewtonState) = max_inner_iterations(state.ws)
@inline max_inner_iterations(solver::NonlinearSolver) = max_inner_iterations(solver.state)
@inline f_calls(state::NewtonState) = f_calls(state.ws)
@inline f_calls(solver::NonlinearSolver) = f_calls(solver.state)
@inline jvp_calls(state::NewtonState) = jvp_calls(state.ws)
@inline jvp_calls(solver::NonlinearSolver) = jvp_calls(solver.state)

# `nothing` is the linear-solver workspace of a solve that keeps no count.
@inline max_inner_iterations(::Nothing) = throw(ArgumentError(
    "max_inner_iterations reads the count of a Newton with GMRES; this solve keeps none"))
@inline f_calls(::Nothing) = throw(ArgumentError(
    "f_calls reads the count of a Newton with GMRES; this solve keeps none"))
@inline jvp_calls(::Nothing) = throw(ArgumentError(
    "jvp_calls reads the count of a Newton with GMRES; this solve keeps none"))
