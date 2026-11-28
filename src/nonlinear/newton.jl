"""
    Newton(; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
             refinement = NoRefinement(), regularization = 0, reuse = nothing,
             forcing = nothing, preconditioner = nothing, best = false, keep_factor = false)

Newton's method for [`NonlinearProblem`](@ref), [`StationarityProblem`](@ref) and, as
[`GaussNewton`](@ref) configures it, for [`LeastSquaresProblem`](@ref), with linear solver
`linsolve` ([`LUFactorization`](@ref), [`LDLTFactorization`](@ref),
[`CholeskyFactorization`](@ref), [`QRFactorization`](@ref), [`SVDFactorization`](@ref), the
matrix-free [`GMRES`](@ref), [`BiCGStab`](@ref), [`DQGMRES`](@ref) and [`MINRESQLP`](@ref), or
[`KroneckerFactorization`](@ref) for the stage system of an implicit Runge–Kutta method),
globalization `globalization` ([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref),
[`StrongWolfe`](@ref), the trust regions [`DogLeg`](@ref) and [`Steihaug`](@ref), the damped
step [`LevenbergMarquardt`](@ref) or, for a [`StationarityProblem`](@ref), the pseudo-transient
continuation [`PseudoTransient`](@ref)), Jacobian selector `jacobian`, iterative refinement
`refinement` ([`NoRefinement`](@ref), [`Refinement`](@ref) or [`GMRESRefinement`](@ref)),
diagonal regularisation `regularization`, and factor-reuse period `reuse`.

# Fields

- `linsolve::L`: the linear-solver method. A [`LeastSquaresProblem`](@ref) takes none of
  [`LUFactorization`](@ref), [`LDLTFactorization`](@ref) and [`CholeskyFactorization`](@ref)
  outside the stacked [`LevenbergMarquardt`](@ref) step. With more residuals than unknowns it
  takes no factor in a lower precision than the element type of the iterate, because a
  Gauss–Newton step with such a factor stops off the least-squares point; a factor in the same or
  a higher precision is taken, and the stacked `LevenbergMarquardt` step, which reads no factor,
  takes any. Each refusal raises an `ArgumentError` where the state is built. A factor in a higher
  precision on more residuals than unknowns is not promoted to the working precision after a
  stall: at a nonzero-residual least-squares point ``\\|F + J d\\| \\approx \\|F\\|``, which on a
  square problem reads as a stall. It is promoted where the factorisation fails.
- `globalization::G`: the globalization method. A line search takes its merit at `α = 0` from
  the state, where [`start!!`](@ref) sets it to `norm2(F(x₀))` and each [`step!!`](@ref) to the
  merit of the residual it returns. A `step!!` with a line search therefore takes only the
  `(x, Fx)` that the last `start!!` or `step!!` on its state returned, as in [`solve!!`](@ref).
  A line-search step on a state that no `start!!` saw, or that `solve!!` returned, returns
  `NONFINITE` and keeps its iterate, and so does a [`LevenbergMarquardt`](@ref) step, whose
  damping parameter that state holds as `NaN`, and a [`PseudoTransient`](@ref) step, whose
  pseudo-time step it holds as `NaN`.
- `jacobian::J`: `nothing` (use `prob.J` when present, or `AutoForwardDiff()` otherwise), an
  `AbstractADType`, or a custom Jacobian callable: `J(x, p)`, which returns the Jacobian, on a
  value iterate (a `Number` or a `StaticArray`), and `J!(J, x, p)`, which writes it into `J`, on
  an array. A `Newton` with a matrix-free linear method ([`matrix_free`](@ref)) refuses a
  callable with an `ArgumentError`, because its solve forms no Jacobian matrix.
  `AutoForwardDiff()` writes a dense Jacobian only, so with a sparse `jacobian_prototype` it
  raises an `ArgumentError` where the state is built; such a problem takes an analytic `J` or an
  `AutoSparse` back end. Adapting `Newton` with [`ToReal`](@ref) (`adapt(ToReal{R}(), m)`)
  converts floating-point values captured inside a `jacobian` closure to `R`, while a function
  with no captured floats passes through unchanged.
- `refinement::RF`: the refinement setting ([`NoRefinement`](@ref), [`Refinement`](@ref) or
  [`GMRESRefinement`](@ref)). A `Refinement` refines the direction of the system that the factor
  solves, the regularised ``(J + μ I) d = -F`` when `μ > 0`; a `GMRESRefinement` runs the same
  passes and, where they fail on a factorisation below the working precision, a GMRES-IR stage
  with the promoted factors before the step refactorises in `T`. A
  [`LevenbergMarquardt`](@ref) globalization takes no refinement: a refinement corrects the solve
  of ``J d = -F``, which the damped step does not take, and the constructor raises an
  `ArgumentError`. A [`LeastSquaresProblem`](@ref) takes no refinement either: a refinement drives
  the linear residual ``J d + F`` to zero, which a least-squares step with more residuals than
  unknowns does not reach, and the refusal raises an `ArgumentError` where the state is built.
- `regularization::R`: non-negative finite diagonal shift ``μ ≥ 0`` added as ``J + μ I`` before
  factorisation when `μ > 0`. A [`LeastSquaresProblem`](@ref) with more residuals than unknowns
  takes none, and `μ > 0` raises an `ArgumentError` where the state is built.
- `reuse::K`: `nothing` (`K = Nothing`, factorise at every step) or `Int32(k)` (`K = Int32`,
  reuse the factor for up to `k` consecutive steps, refreshing early after a `STALLED` step and,
  within the step, after a failed refinement of a [`Refinement`](@ref) or a
  [`GMRESRefinement`](@ref) on a reused factor, which counts one
  step-rule failure). A line search along a reused factor that fails without a step is retried
  once, in the same step, from a fresh factor. A [`DogLeg`](@ref) or
  [`LevenbergMarquardt`](@ref) globalization takes no reuse: its model needs the Jacobian of the
  current iterate, and the constructor raises an `ArgumentError`. A [`PseudoTransient`](@ref)
  globalization takes no reuse and no refinement either, because its shift changes at every
  step. A [`GMRES`](@ref) with a preconditioner takes `reuse = k` as the period of the
  preconditioner: it keeps the preconditioner for up to `k` consecutive steps, so
  [`refresh!`](@ref) runs at the first step, `k` steps after each refresh, and early at the step
  after a `STALLED` step or an inner solve that reached its cap, with no retry within a step.
  A step along a reused factor or preconditioner whose rate
  ``θ_k = \\|Δ_k\\| / \\|Δ_{k-1}\\| ≥ 1`` refreshes it at the next step too, as a `STALLED` step
  does; a `NaN` rate, at the first step of a solve or after a step of zero norm, refreshes nothing.
- `best::B`: `Val(false)`, the default, or `Val(true)` from the keyword `best = true`, which
  returns the best iterate (below). The six-argument positional constructor gives `Val(false)`.
- `keep_factor::Bool`: `false`, the default, or `true` from the keyword `keep_factor = true`, which
  keeps the factor of `reuse = k` across a sequence of solves on one state, such as the
  [`solve!`](@ref) calls of an integrator. A solve then starts on the factor and its age that the
  last solve left, and the age counts toward `k` across solves. A solve leaves them only after
  `SUCCESS` in one step or with a last rate ``θ ≤ 10^{-3}`` (Hairer and Wanner 1996, *Solving
  Ordinary Differential Equations II*, p. 121), and not after a solve that promoted its
  factorisation to the working precision; every other solve leaves age `0`, so that the next solve
  factorises at its first step. A change of the parameters between two solves keeps the factor,
  and the rate rule above refreshes it where it no longer contracts; a caller that knows the
  Jacobian changed, such as after a change of the step size, passes
  `solve!(solver, x, p; refactorize = true)`. A solve that steps only along a kept factor reports
  in its [`SolverStatus`](@ref) the `rank` of that factor, which the rate test reads, and `NaN`
  for `nullnorm` and `gradratio`. `keep_factor = true` without `reuse` raises an
  `ArgumentError`. The six- and seven-argument positional constructors give `false`.

# The best iterate

With `best = true` a solve keeps the iterate of the smallest ``\\|F(x)\\|_2`` among `x₀` and every
iterate that a step returns, the earliest of equal norms, and [`finish!!`](@ref) returns it in
place of the last iterate when the solve ends with a code other than `SUCCESS` and its norm is
finite and below the norm of the last one. `status.fnorm` then reports that norm, and every other
field of the status is that of `best = false`: the code, the counts, the step norm and the
diagnostics of the last fresh factorisation. For an array iterate the residual buffer then holds
the residual of the returned iterate. Otherwise the solve returns what `best = false` returns. The
ranking reads the 2-norm under every [`StoppingNorm`](@ref) of the [`Options`](@ref), and each
solve starts a new record in [`start!!`](@ref), so a state reused across solves restores nothing
of an earlier one.
The state keeps a copy of `x` and of `F(x)` for an array iterate, which [`init_state`](@ref)
allocates once, and holds the record by value for a value iterate, where it stays `isbits`. A
combinator that steps a `Newton` member, such as [`Repeat`](@ref) or [`Switch`](@ref), gets the
member's restore through the member's `finish!!`. A [`Deflated`](@ref) problem and the outer
method of a [`BlockEliminated`](@ref), [`VariableProjection`](@ref) or
[`BatchEliminated`](@ref) problem refuse `best = true` with an `ArgumentError`, and so does a
[`LeftPreconditioned`](@ref) method for a `Newton` in its outer method, also as a member of a
combinator there, because the record would rank the preconditioned residual and `status.fnorm` is
the norm of the global residual; its inner method takes `best = true`, which has no effect there.

# Keywords of a matrix-free linear method

`forcing` and `preconditioner` configure the [`GMRES`](@ref), [`BiCGStab`](@ref) or
[`DQGMRES`](@ref) of `linsolve`, `forcing` configures a [`MINRESQLP`](@ref) too, and a `Newton`
that takes either of them with any other linear method, or a `preconditioner` with `MINRESQLP`,
raises an `ArgumentError`. A `Newton` with a matrix-free linear method ([`matrix_free`](@ref))
forms no factor, so it takes no `reuse`, except a `GMRES` with a preconditioner, where `reuse` is
the period of its refresh, no [`Refinement`](@ref), no
[`LevenbergMarquardt`](@ref) and no [`PseudoTransient`](@ref) globalization, and a
[`DogLeg`](@ref) needs an unrestarted `GMRES` without a preconditioner and with
`basis_eltype = Nothing`, or a `MINRESQLP`. A
[`Steihaug`](@ref) globalization needs a `MINRESQLP`. A [`JVPFunction`](@ref) in `jacobian` needs
a matrix-free linear method, because a factorising method needs a Jacobian matrix. Each refusal
is an `ArgumentError`.

- `forcing`: an [`AbstractForcingTerm`](@ref) that replaces the forcing term of `linsolve`, and so
  the relative tolerance of every Newton step's inner solve.
- `preconditioner`: the right preconditioner of `linsolve`, which [`refresh!`](@ref) is called on
  once per Newton step, or as `reuse` sets for a `GMRES`, and [`apply_preconditioner!`](@ref) on
  before every Jacobian-vector product.
"""
struct Newton{L, G, J, RF, R <: Real, K <: Union{Nothing, Int32},
    B <: Union{Val{false}, Val{true}}}
    linsolve::L
    globalization::G
    jacobian::J
    refinement::RF
    regularization::R
    reuse::K
    best::B
    keep_factor::Bool
    # `adapt` rebuilds a method through this constructor, inside a device kernel too, so the check
    # also refuses a value that `ToReal` narrows to `Inf` or to zero. A kernel cannot format a
    # message, so each message is constant.
    function Newton{L, G, J, RF, R, K, B}(
            linsolve::L, globalization::G, jacobian::J,
            refinement::RF, regularization::R,
            reuse::K, best::B,
            keep_factor::Bool) where {
            L, G, J, RF, R <: Real, K <: Union{Nothing, Int32},
            B <: Union{Val{false}, Val{true}}}
        regularization >= zero(R) && isfinite(regularization) ||
            throw(ArgumentError("Newton regularization must be finite and ≥ 0"))
        keep_factor && reuse === nothing &&
            throw(ArgumentError("Newton keep_factor = true needs reuse = k: only a factor that its steps reuse can survive a solve"))
        if reuse !== nothing
            reuse >= Int32(1) ||
                throw(ArgumentError("Newton reuse must be ≥ 1 or nothing"))
            globalization isa DogLeg &&
                throw(ArgumentError("Newton with a DogLeg globalization takes no reuse"))
            globalization isa LevenbergMarquardt &&
                throw(ArgumentError("Newton with a LevenbergMarquardt globalization takes no reuse"))
            globalization isa PseudoTransient &&
                throw(ArgumentError("Newton with a PseudoTransient globalization takes no reuse: its shift changes at every step"))
        end
        globalization isa LevenbergMarquardt && is_refining(refinement) &&
            throw(ArgumentError("Newton with a LevenbergMarquardt globalization takes no refinement: a refinement corrects the solve of J d = -F, which the damped step does not take; use refinement = NoRefinement()"))
        if globalization isa PseudoTransient
            is_refining(refinement) &&
                throw(ArgumentError("Newton with a PseudoTransient globalization takes no refinement; use refinement = NoRefinement()"))
            (matrix_free(linsolve) || linsolve isa KroneckerFactorization) &&
                throw(ArgumentError("Newton with a PseudoTransient globalization needs a factorising linear method other than KroneckerFactorization: its step factorises the shifted Jacobian J + I/δ"))
        end
        if linsolve isa MINRESQLP
            reuse === nothing ||
                throw(ArgumentError("Newton with MINRESQLP takes no factor reuse: a MINRESQLP step has no factor"))
            is_refining(refinement) &&
                throw(ArgumentError("Newton with MINRESQLP takes no refinement: a MINRESQLP step has no factor to refine against"))
            globalization isa LevenbergMarquardt &&
                throw(ArgumentError("Newton with a LevenbergMarquardt globalization needs a Jacobian matrix, which a MINRESQLP solve does not form"))
        elseif matrix_free(linsolve)
            (reuse === nothing || preconditioner_reuse(linsolve)) ||
                throw(ArgumentError("Newton with a matrix-free linear method takes no factor reuse: its step has no factor; a GMRES with a preconditioner takes reuse as the period of the preconditioner's refresh"))
            is_refining(refinement) &&
                throw(ArgumentError("Newton with a matrix-free linear method takes no refinement: its step has no factor to refine against"))
            if globalization isa DogLeg
                linsolve isa GMRES ||
                    throw(ArgumentError("Newton with a DogLeg globalization needs GMRES among the matrix-free linear methods: its trial loop reads the Arnoldi relation of the GMRES solve"))
                linsolve.restart == linsolve.maxiter ||
                    throw(ArgumentError("Newton with a DogLeg globalization needs unrestarted GMRES: restart must equal maxiter"))
                linsolve.preconditioner === nothing ||
                    throw(ArgumentError("Newton with a DogLeg globalization takes a GMRES without a preconditioner"))
                linsolve.basis_eltype === Nothing ||
                    throw(ArgumentError("Newton with a DogLeg globalization takes a GMRES with basis_eltype = Nothing: its trial loop reads the Arnoldi basis as orthonormal in the working precision"))
            end
            globalization isa LevenbergMarquardt &&
                throw(ArgumentError("Newton with a LevenbergMarquardt globalization needs a Jacobian matrix, which a matrix-free solve does not form"))
        elseif jacobian isa JVPFunction
            throw(ArgumentError("Newton with a JVPFunction needs a matrix-free linear method, such as GMRES(...): a factorising linear method needs a Jacobian matrix, which a JVPFunction does not give"))
        end
        globalization isa Steihaug && !(linsolve isa MINRESQLP) &&
            throw(ArgumentError("Newton with a Steihaug globalization needs linsolve = MINRESQLP(...): its conjugate gradients take the place of the MINRES-QLP solve"))
        new{L, G, J, RF, R, K, B}(
            linsolve, globalization, jacobian, refinement, regularization, reuse, best,
            keep_factor)
    end
end

# The positional forms: the eight fields, which `adapt` rebuilds a method from, the seven before
# `keep_factor`, which give `keep_factor = false`, and the six before `best`, which also give
# `best = Val(false)`.
function Newton(
        linsolve::L, globalization::G, jacobian::J,
        refinement::RF, regularization::R,
        reuse::K, best::B = Val(false),
        keep_factor::Bool = false) where {
        L, G, J, RF, R <: Real, K <: Union{Nothing, Int32},
        B <: Union{Val{false}, Val{true}}}
    return Newton{L, G, J, RF, R, K, B}(
        linsolve, globalization, jacobian, refinement, regularization, reuse, best,
        keep_factor)
end

function Newton(;
        linsolve = LUFactorization(),
        globalization = Static(),
        jacobian = nothing,
        refinement = NoRefinement(),
        regularization::Real = 0,
        reuse::Union{Nothing, Integer} = nothing,
        forcing::Union{Nothing, AbstractForcingTerm} = nothing,
        preconditioner = nothing,
        best::Bool = false,
        keep_factor::Bool = false)
    reg = float(regularization)
    r = reuse === nothing ? nothing : Int32(reuse)
    ref = refinement isa Bool ? (refinement ? Refinement() : NoRefinement()) : refinement
    if forcing !== nothing || preconditioner !== nothing
        matrix_free(linsolve) || throw(ArgumentError(
            "the forcing and preconditioner keywords of Newton configure a GMRES, BiCGStab or DQGMRES linear method, and forcing a MINRESQLP one; pass them to the method, or set linsolve = GMRES(...)"))
        linsolve = with_forcing(linsolve, forcing, preconditioner)
    end
    return Newton(linsolve, globalization, jacobian, ref, reg, r, Val(best), keep_factor)
end

# A matrix-free linear method with the forcing term `forcing` and the right preconditioner
# `preconditioner` in place of its own, where each is not `nothing`, and every other field kept.
# `MINRESQLP` takes no preconditioner.
@inline function with_forcing(g::GMRES, forcing, preconditioner)
    f = forcing === nothing ? g.forcing : forcing
    P = preconditioner === nothing ? g.preconditioner : preconditioner
    return GMRES(
        g.maxiter, g.restart, f, P, g.jvp_eltype, g.mgs, g.basis_eltype, g.flexible)
end
@inline function with_forcing(g::BiCGStab, forcing, preconditioner)
    f = forcing === nothing ? g.forcing : forcing
    P = preconditioner === nothing ? g.preconditioner : preconditioner
    return BiCGStab(g.maxiter, f, P)
end
@inline function with_forcing(g::DQGMRES, forcing, preconditioner)
    f = forcing === nothing ? g.forcing : forcing
    P = preconditioner === nothing ? g.preconditioner : preconditioner
    return DQGMRES(g.maxiter, g.memory, f, P)
end
@inline function with_forcing(g::MINRESQLP, forcing, preconditioner)
    preconditioner === nothing || throw(ArgumentError(
        "MINRESQLP takes no preconditioner: pass preconditioner = nothing, or use GMRES"))
    return MINRESQLP(g.maxiter, forcing === nothing ? g.forcing : forcing)
end

"""
    QuasiNewton(k; linsolve = LUFactorization(), globalization = Static(), jacobian = nothing,
                refinement = NoRefinement(), regularization = 0, best = false)

Construct [`Newton`](@ref) with factor-reuse period `reuse = Int32(k)` (`k ≥ 1`), passing every
keyword to it.
"""
function QuasiNewton(k::Integer; kwargs...)
    return Newton(; reuse = k, kwargs...)
end

Adapt.@adapt_structure Newton

@inline _has_analytic_jacobian(prob::NonlinearProblem) = prob.J !== nothing
@inline _has_analytic_jacobian(prob::LeastSquaresProblem) = prob.J !== nothing
@inline _has_analytic_jacobian(::AbstractNonlinearProblem) = false

# The JVP preparation from the analytic Jacobian of `prob`: the `m × n` buffer of the residual
# `Fx`, or one like the Jacobian prototype of `prob`, which a line search and a refinement along a
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

# The own chunked forward mode writes a dense Jacobian only, so a Jacobian preparation of it on a
# problem with a sparse prototype, whose Jacobian buffer is sparse, is refused here, at `init`.
@inline function check_jacobian_prep(prep, prob)
    prep isa ChunkedForwardDiff &&
        jacobian_prototype(prob) isa SparseArrays.AbstractSparseMatrix &&
        throw(ArgumentError(
            "a sparse jacobian_prototype takes no Jacobian from AutoForwardDiff(), the default of Newton's jacobian keyword without an analytic J: it writes a dense Jacobian only. Give the problem an analytic Jacobian J, or set jacobian = AutoSparse(...), which needs SparseMatrixColorings loaded (the GeometricSolversSparseADExt extension), or solve without the jacobian_prototype"))
    return prep
end

@inline function newton_ad_preps(m::Newton, prob::AbstractNonlinearProblem, Fx, x, p)
    if is_refining(m.refinement) && m.jacobian isa AbstractADType &&
       dense_ad(m.jacobian) isa AutoFiniteDiff
        throw(ArgumentError("A refinement requires an exact or forward-mode Jacobian-vector product; AutoFiniteDiff is not supported"))
    end
    has_J = _has_analytic_jacobian(prob)
    if m.jacobian === nothing && !has_J
        prep = check_jacobian_prep(prepare_ad(AutoForwardDiff(), prob, Fx, x, p), prob)
        return prep, prep
    elseif m.jacobian isa AbstractADType && !has_J
        prep = check_jacobian_prep(prepare_ad(m.jacobian, prob, Fx, x, p), prob)
        return prep, prep
    else
        jacprep = if m.jacobian === nothing
            nothing
        elseif m.jacobian isa AbstractADType
            check_jacobian_prep(prepare_ad(m.jacobian, prob, Fx, x, p), prob)
        else
            JacobianFunction(m.jacobian)
        end
        jvpprep = if has_J
            if Fx === nothing
                nothing
            elseif is_refining(m.refinement) && m.reuse !== nothing
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

@inline function empty_static_factor(ls::CholeskyFactorization{P}, ::StaticVector{
        N, T}) where {P, N, T}
    TF = factor_eltype(ls, T)
    return StaticCholesky{N, TF, N * N}(zeros(SMatrix{N, N, TF}), Int32(0), Int32(0))
end

@inline function empty_static_factor(
        ls::CholeskyFactorization{P}, ::T) where {P, T <: Number}
    TF = factor_eltype(ls, T)
    z = SMatrix{1, 1, TF, 1}((zero(TF),))
    return StaticCholesky{1, TF, 1}(z, Int32(0), Int32(0))
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

Whether `init_linear` builds a sparse factor of `method` on the iterate `x` from a detected
Jacobian pattern: `false` by default. It is `true` for [`UMFPACKFactorization`](@ref) on a
`Vector{Float64}` or `Vector{ComplexF64}`, and the `GeometricSolversSparspakExt` extension returns
`true` for an [`LUFactorization`](@ref) whose factor element type is `T` on a `Vector{T}`,
`T <: AbstractFloat`. Where it is `false`, a detected pattern gives the dense factor, as a solve
with no pattern does; an explicit `jacobian_prototype` does not consult it.
"""
@inline has_sparse_factor(::LinearMethod, x) = false

# Refuse, at `init`, an iterate that `method` builds no factor for, also where the step builds its
# factor later: no refusal by default.
@inline check_factor_iterate(::LinearMethod, x) = nothing

"""
    matrix_free(method::LinearMethod)

Whether a [`Newton`](@ref) with the linear method `method` solves its step through
Jacobian–vector products alone and forms no Jacobian matrix: `false` by default, and `true` for
[`GMRES`](@ref), [`BiCGStab`](@ref), [`DQGMRES`](@ref) and [`MINRESQLP`](@ref). Where it is
`true`, the `Newton` takes no factor reuse, except a `GMRES` with a preconditioner, where `reuse`
is the period of the preconditioner's refresh, and no [`Refinement`](@ref), its `forcing` keyword
configures the method, and so does its `preconditioner` keyword for every method but
`MINRESQLP`, a [`JVPFunction`](@ref) is a `jacobian` it takes, and `solve` refuses a
`ForwardDiff.Dual` in `x₀` or `p`. The trait is a query on the linear methods of this package,
not an extension point: the workspace and the step of a matrix-free solve are internal, and a
`Newton` with another method that extends it to `true` raises a `MethodError` in `init`.
"""
@inline matrix_free(::LinearMethod) = false
@inline matrix_free(::Union{GMRES, BiCGStab, DQGMRES, MINRESQLP}) = true

# Whether a matrix-free linear method takes `reuse`: a `GMRES` with a preconditioner, whose
# refresh the period of `reuse` sets.
@inline preconditioner_reuse(g::GMRES) = g.preconditioner !== nothing
@inline preconditioner_reuse(::LinearMethod) = false

# The pattern that the factor of `ls` on `x` is built on: an explicit prototype as it is, and a
# detected pattern only where `has_sparse_factor` holds, else `nothing` and the dense factor. A
# `LeastSquaresProblem` has neither, because `jacobian_pattern` is `nothing` for it: the sparse
# factors are square, so the `m × n` factor of a least-squares problem is dense.
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
# structural diagonal gives the dense factor, or the `ArgumentError` of a method that has no dense
# factor. A `LevenbergMarquardt` step reads no sparse factor (see `check_lm_iterate`), so a
# detected pattern gives it the dense factor and Jacobian buffer, or that `ArgumentError`.
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
# the dense factor, or the `ArgumentError` of a method that has no dense factor, and every other
# case the `init_factor` of `ls`.
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
# `LevenbergMarquardt` trial loop and a refinement multiply by it.
@inline function init_newton_J(m::Newton, fac, fac_T, x, Fx = x)
    if !(m.globalization isa Union{DogLeg, LevenbergMarquardt}) &&
       m.refinement isa NoRefinement
        if factor_eltype(m.linsolve, eltype(x)) === eltype(x)
            return fac.factors
        elseif fac_T !== nothing && m.linsolve isa LUFactorization
            return fac_T.factors
        end
    end
    if fac.factors isa SparseArrays.AbstractSparseMatrix
        return similar(fac.factors, eltype(x))
    end
    return buffer(x, eltype(x), length(Fx), length(x))
end

# The working-precision factor slot of a mixed-precision solve starts zeroed, so that
# [`promoted_factors`](@ref) reads zeros before a GMRES-IR stage or a promotion writes it, rather
# than the storage that `similar` left. A value iterate, a factorisation in the working precision
# and a factor that no promotion copies keep no such slot.
@inline function zero_promoted_factors!(fac::Union{DenseLU, DeviceLU})
    fill!(fac.factors, zero(eltype(fac.factors)))
    return nothing
end
@inline zero_promoted_factors!(fac) = nothing

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

# The rule state of `Newton` with `reuse = k`: the age of the factor, or of the preconditioner of a
# `GMRES`, whether the last step stalled, which refreshes it early, and whether the last step
# stepped along a factor or a preconditioner that an earlier step formed. A `GMRES` step also sets
# `stalled` where its inner solve reached its cap, and `observe_rate!!` where a step along a reused
# factor has the rate `θ ≥ 1`.
struct FactorAge
    age::Int32
    stalled::Bool
    reused::Bool
end

# The age of a factor that a step has just factorised: the next step reuses it unless the period
# is one step long.
@inline fresh_age(m::Newton) = m.reuse == Int32(1) ? Int32(0) : Int32(1)

# The age of a reused factor after one more step: `0` at the end of its period of `reuse` steps.
@inline next_age(rule::FactorAge, m::Newton) = rule.age + one(Int32) >= m.reuse ? Int32(0) :
                                               rule.age + one(Int32)

# The rule state of `Newton` with `DogLeg` or `Steihaug`: the radius, and the initial radius that
# each solve starts from, `0` when the first step sets it.
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

# The rule state of `Newton` with `PseudoTransient`: the pseudo-time step `δ`, `NaN` in
# `init_state` and after `finish!!`, as the damping of `LevenbergMarquardt` is, and `R(pt.δ₀)`
# after `start!!`.
struct PseudoTimeStep{R}
    δ::R
end

@inline newton_rule(m::Newton, x) = newton_rule(m.globalization, m.reuse, real(eltype(x)))
@inline newton_rule(globalization, ::Nothing, ::Type{R}) where {R} = nothing
@inline newton_rule(globalization, ::Int32, ::Type{R}) where {R} = FactorAge(
    Int32(0), false, false)
@inline function newton_rule(dl::DogLeg, ::Nothing, ::Type{R}) where {R}
    δ₀ = dogleg_initial_radius(dl, R)
    return TrustRadius(δ₀, δ₀)
end
@inline function newton_rule(st::Steihaug, ::Nothing, ::Type{R}) where {R}
    δ₀ = steihaug_initial_radius(st, R)
    return TrustRadius(δ₀, δ₀)
end
@inline function newton_rule(::LevenbergMarquardt, ::Nothing, ::Type{R}) where {R}
    return DampingParameter(R(NaN))
end
@inline function newton_rule(::PseudoTransient, ::Nothing, ::Type{R}) where {R}
    return PseudoTimeStep(R(NaN))
end

@inline reset_rule(::Nothing, _) = nothing
@inline reset_rule(::FactorAge, _) = FactorAge(Int32(0), false, false)
@inline reset_rule(r::TrustRadius, _) = TrustRadius(r.radius₀, r.radius₀)
@inline function reset_rule(::DampingParameter{R}, lm::LevenbergMarquardt) where {R}
    return DampingParameter(R(lm.μ₀))
end
@inline function reset_rule(::PseudoTimeStep{R}, pt::PseudoTransient) where {R}
    return PseudoTimeStep(R(pt.δ₀))
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
# globalization workspace, the rule state of its reuse, its `DogLeg` or `Steihaug`, its
# `LevenbergMarquardt` or its `PseudoTransient`, the merit `norm2(Fx)` of the current residual for a
# line search (`nothing` for any other globalization), the norm `‖Fx‖` of the current residual, the
# diagnostics of the last factorisation, the refinement buffer (`nothing` for `NoRefinement` or R1),
# the working-precision factor slot for mixed-precision promotion (`nothing` where the factorisation
# runs in the working precision, and for `LevenbergMarquardt`, which promotes nothing), the
# promotion flag, the count of consecutive slow steps a promotion waits for, the linear-solver
# workspace, which only a [`GMRES`](@ref) or a [`MINRESQLP`](@ref) solve has (`nothing` for every
# factorising method), and the slots that `solve!!` copies the start point of a step and its
# residual into (two buffers for an array iterate of a `LeastSquaresProblem`, `(nothing, nothing)`
# otherwise). For a value iterate the workspace slots are `nothing`, and so is the factor slot
# unless the factor is reused. The factor slot is the storage that each factorisation writes into;
# only a step with reuse stores the factor it returns, because only such a step reads it again.
# The norm is `NaN` on a state that no `start!!` saw or that `solve!!` returned. The last field is
# the record of the best iterate, a `BestIterate` under `best = true` and `nothing` else.
struct NewtonState{JP, VP, FX, D, FAC, JM, G, RS, PH, R, DG, RB, FT, WS, KS, BR}
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
    kept::KS
    best::BR
end

# A state with no slots for the start point of a step and no record of the best iterate: the
# fields up to the workspace `ws`, which defaults to `nothing` for a factorising method, whose
# linear solve keeps no workspace of its own.
@inline function NewtonState(jacprep, jvpprep, Fx, d, fac, J, g, rule, merit, fnorm, diag,
        refbuf, fac_T, promoted::Bool, slow_steps::Int32, ws = nothing)
    return NewtonState(
        jacprep, jvpprep, Fx, d, fac, J, g, rule, merit, fnorm, diag, refbuf, fac_T,
        promoted, slow_steps, ws, (nothing, nothing), nothing)
end

# The record of the best iterate of a solve under `best = true`: the iterate `x`, its residual
# `Fx` and its norm `fnorm`. For a value iterate `x` is the value, an `SVector` for an `MVector`
# iterate, and `Fx` is `nothing`, so the record is `isbits`; for an array iterate both are copies
# that the record owns. `fnorm` is `NaN` until `start!!` sets it to the norm of the start.
struct BestIterate{X, FX, R <: Real}
    x::X
    Fx::FX
    fnorm::R
end

# The record that `init_state` gives a state with the residual slot `Fx`, from the start `x`.
@inline init_best(::Val{false}, x, Fx) = nothing
@inline function init_best(::Val{true}, x, Fx)
    return BestIterate(best_slot(x), best_slot(Fx), real(eltype(x))(NaN))
end
@inline best_slot(x::Union{Number, StaticArray}) = x
@inline best_slot(x::StaticVector) = SVector(x)
@inline best_slot(x::AbstractArray) = similar(x)
@inline best_slot(::Nothing) = nothing

# The iterate `x` in the record's slot, and the record's iterate `b` in the type of `x`: a static
# vector is kept as an `SVector`, which an `MVector` iterate takes back as an `MVector`.
@inline keep_iterate!!(slot, x) = copy!!(slot, x)
@inline keep_iterate!!(::StaticVector, x::StaticVector) = SVector(x)
@inline restore_iterate!!(x, b) = copy!!(x, b)
@inline restore_iterate!!(x::StaticVector, b::StaticVector) = convert(typeof(x), b)

@inline function with_best(s::NewtonState, best)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws, s.kept, best)
end

# The record after an iterate `x` with residual `Fx` and norm `fnorm`: it takes `x` where `fnorm`
# is below the norm it holds, which a `NaN` norm never is, so the earliest of equal norms stays.
@inline take_best(::Nothing, x, Fx, fnorm) = nothing
@inline function take_best(b::BestIterate, x, Fx, fnorm)
    return fnorm < b.fnorm ? start_best(b, x, Fx, fnorm) : b
end
@inline keep_residual!!(::Nothing, Fx) = nothing
@inline keep_residual!!(slot, Fx) = copy!!(slot, Fx)

# The close of a solve under `best = true`: the iterate and the status that `finish!!` returns,
# the record's where the solve ends without `SUCCESS` and its norm is finite and below the last
# one, as `Anderson` restores its best iterate. An array iterate takes the record into `x` and its
# residual into `Fx`, the residual buffer of the solve.
@inline restore_best(::Nothing, x, Fx, status) = x, status
@inline function restore_best(b::BestIterate, x, Fx, status::SolverStatus)
    should_restore_best(status, b.fnorm) || return x, status
    restore_residual!!(Fx, b.Fx)
    return restore_iterate!!(x, b.x), with_fnorm(status, b.fnorm)
end
@inline restore_residual!!(Fx, ::Nothing) = nothing
@inline restore_residual!!(Fx, slot) = copy!!(Fx, slot)

@inline state_gradratio(state::NewtonState) = state.diag.gradratio
@inline state_jnorm(state::NewtonState) = state.diag.jnorm
@inline state_rank(state::NewtonState) = state.diag.rank
@inline reused_step(state::NewtonState) = state.rule isa FactorAge && state.rule.reused
@inline step_start_buffers(state::NewtonState) = state.kept

# The state whose diagnostics measure the stationarity test at `(x, Fx)`: `solve!!` re-measures
# them at the end of a step before it returns that iterate on the test, because a step's
# diagnostics are those of the last fresh factorisation, at the start of that step. Every other
# method and problem keeps its reading.
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
    return with_rule(state, FactorAge(fresh_age(m), false, state.rule.reused))
end

@inline function with_factor(s::NewtonState, fac)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws, s.kept, s.best)
end

@inline function with_factor_T(s::NewtonState, fac_T)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, fac_T, s.promoted, s.slow_steps, s.ws, s.kept, s.best)
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
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws, s.kept, s.best)
end

# The merit of the current residual and its norm: the two values a step returns, or that a solve
# starts from.
@inline function with_merit(s::NewtonState, merit, fnorm)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, merit,
        oftype(s.fnorm, fnorm), s.diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws, s.kept, s.best)
end

@inline function with_diag(s::NewtonState, diag)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, diag,
        s.refbuf, s.fac_T, s.promoted, s.slow_steps, s.ws, s.kept, s.best)
end

@inline function with_mixed(s::NewtonState, promoted::Bool, slow_steps::Int32)
    return NewtonState(
        s.jacprep, s.jvpprep, s.Fx, s.d, s.fac, s.J, s.g, s.rule, s.merit, s.fnorm, s.diag,
        s.refbuf, s.fac_T, promoted, slow_steps, s.ws, s.kept, s.best)
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
        m.regularization, m.reuse, m.best, m.keep_factor)
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

# A `PseudoTransient` step takes only the problem that a `StationarityProblem` solve hands to its
# method, the `_ReducedStationarityProblem`. Where `E == U` its `F` is the gradient of the action
# in the unknowns, and the shift follows the gradient flow of that action; for other index sets
# `F` is no gradient, and the shifted steps still aim at a root of `F`. Every other problem is
# refused, the problems that a combinator builds included. `stationarity.jl` defines that type
# after this file, so the test is an `isa` in the body and not a method signature.
@inline function check_pseudotransient_problem(prob::AbstractNonlinearProblem)
    prob isa _ReducedStationarityProblem || throw(ArgumentError(
        "PseudoTransient solves a StationarityProblem: its shifted step follows the gradient flow of the action S where E == U, and the residual of a problem of another type is no gradient; use another globalization"))
    return nothing
end

# The state of every linear method, with the record of the best iterate where `best = true`. On a
# `LeftPreconditionedProblem` the record would rank `‖rˡ‖₂` while `status.fnorm` is the global
# `‖F‖₂`, so `best = true` is refused there, for an outer `Newton` at any depth of a combinator.
@inline function init_state(m::Newton, prob::AbstractNonlinearProblem, x, p)
    m.best === Val(true) && prob isa LeftPreconditionedProblem &&
        throw(ArgumentError(
            "LeftPreconditioned does not take a Newton with best = true as its outer method: the best iterate would be ranked by the preconditioned residual, and status.fnorm is the norm of the global residual; use best = false"))
    state = newton_init_state(m, prob, x, p)
    return with_best(state, init_best(m.best, x, state.Fx))
end

@inline function newton_init_state(m::Newton, prob::AbstractNonlinearProblem, x, p)
    m.globalization isa PseudoTransient && check_pseudotransient_problem(prob)
    m.linsolve isa MINRESQLP && return init_minresqlp_state(m, m.linsolve, prob, x, p)
    matrix_free(m.linsolve) && return init_gmres_state(m, m.linsolve, prob, x, p)
    m.linsolve isa KroneckerFactorization &&
        return init_kronecker_state(m, m.linsolve, prob, x, p)
    if prob isa LeastSquaresProblem &&
       m.linsolve isa Union{LUFactorization, LDLTFactorization, CholeskyFactorization} &&
       !(m.globalization isa LevenbergMarquardt)
        throw(ArgumentError(
            "LeastSquaresProblem does not support $(nameof(typeof(m.linsolve))); use QRFactorization() or SVDFactorization()"))
    end
    # A `Refinement` drives the linear residual `J d + F` to zero, which a least-squares step with
    # more residuals than unknowns does not reach (known issue K80)
    if prob isa LeastSquaresProblem && is_refining(m.refinement)
        throw(ArgumentError("LeastSquaresProblem does not support refinement"))
    end
    check_factor_iterate(m.linsolve, x)
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
        zero_promoted_factors!(fac_T)
        refbuf = init_refinement(m.refinement, x)
        kept = prob isa LeastSquaresProblem ? (buffer(x), buffer(Fx)) : (nothing, nothing)
        return NewtonState(
            jacprep, jvpprep, Fx, buffer(x), fac, init_newton_J(m, fac, fac_T, x, Fx),
            init_globalization(m.globalization, m.linsolve, x, Fx), rule, merit, R(NaN),
            diag, refbuf, fac_T, false, Int32(0), nothing, kept, nothing)
    end
end

@inline residual_buffer(state::NewtonState) = state.Fx

"""
    linear_residual_buffer(solver)

The buffer of the linear residual ``r = -F(x) - J d`` of the last step of `solver`, in the working
precision `T`, which the defect-correction passes and the GMRES-IR stage reuse. It is `nothing` for
a solve that refines nothing, and for a value iterate, whose storage holds no buffer. It is not the
slot of `F(x₀)` that [`residual_buffer`](@ref) names. `linear_residual_buffer(state)` reads the
state that [`init_state`](@ref) returned instead.
"""
@inline linear_residual_buffer(state::NewtonState) = linear_residual_buffer(state.refbuf)
@inline function linear_residual_buffer(solver::NonlinearSolver)
    return linear_residual_buffer(solver.state)
end
@inline function linear_residual_buffer(refbuf::Union{RefinementBuffer, GMRESIRBuffer})
    return refbuf.r
end
@inline linear_residual_buffer(::Nothing) = nothing

"""
    stored_factors(solver)

The factors of the factorisation that the last step of `solver` took, in the factorisation
precision `TF`: [`stored_factors`](@ref) of the factor of the state, the packed `L` and `U` of an
[`LUFactorization`](@ref) with its `ipiv` in the factor itself. It is `nothing` for a solve that
holds no factor — a matrix-free [`GMRES`](@ref) solve, and a value iterate whose factor no step
stored — and for a factorisation that is no LU factorisation. `stored_factors(state)` reads the
state that [`init_state`](@ref) returned instead.
"""
@inline stored_factors(state::NewtonState) = stored_factors(state.fac)
@inline stored_factors(solver::NonlinearSolver) = stored_factors(solver.state)

"""
    promoted_factors(solver)

The factors of the working-precision copy of [`stored_factors`](@ref) that a [`GMRESRefinement`](@ref)
promotes, in the element type of the iterate `T`: `T.(stored_factors(solver))` while the
GMRES-IR stage that promoted them is the one that holds them, and the refactorised factors of the
Jacobian where the step promoted to `T` instead. The storage belongs to the state, and `init`
leaves it zeroed, so a read before the first promotion reads zeros. It is `nothing` for a solve
that keeps no working-precision factor slot: a factorisation already in `T`, a value iterate, and
a matrix-free [`GMRES`](@ref) solve; and for a factorisation that is no LU factorisation.
"""
@inline promoted_factors(state::NewtonState) = promoted_factors(state.fac_T)
@inline promoted_factors(solver::NonlinearSolver) = promoted_factors(solver.state)
@inline promoted_factors(fac) = stored_factors(fac)

"""
    preconditioned_buffer(solver)

The buffer of the preconditioned product ``\\hat{U}^{-1} \\hat{L}^{-1} P (J v)`` that the GMRES-IR
stage of a [`GMRESRefinement`](@ref) forms for each Krylov vector `v`, in the working precision
`T`, or `nothing` for a solve that forms none. `preconditioned_buffer(state)` reads the state that
[`init_state`](@ref) returned instead.
"""
@inline preconditioned_buffer(state::NewtonState) = preconditioned_buffer(state.refbuf)
@inline preconditioned_buffer(solver::NonlinearSolver) = preconditioned_buffer(solver.state)
@inline preconditioned_buffer(refbuf::GMRESIRBuffer) = refbuf.z
@inline preconditioned_buffer(refbuf) = nothing

# The per-solve reset that every `Newton` method shares: the rule state, the anchor merit of a
# line search, the promotion flags and the diagnostics. Under `keep_factor = true` the rule state
# is the one that the last `finish!!` left, so the first step reuses the factor that it kept, and
# the diagnostics keep the rank of the last fresh factor, which the rate test reads along it. That
# factor may be one that the last solve dropped, after a solve that kept no factor or after
# `refactorize = true`: the first step then factorises afresh, and only a solve that ends with no
# step reports the dropped factor's rank.
@inline function newton_start!!(state::NewtonState, m::Newton, x, Fx, fnorm₀)
    R = real(eltype(x))
    m.keep_factor || (state = with_rule(state, reset_rule(state.rule, m.globalization)))
    state = with_merit(state, start_merit(m.globalization, Fx), fnorm₀)
    state = with_mixed(state, false, Int32(0))
    rank = m.keep_factor ? state.diag.rank : Int32(-1)
    return with_diag(state, NewtonDiagnostics{R}(rank, R(NaN), R(NaN), zero(R)))
end

# A solve under `best = true` starts its record at `x₀`, whatever an earlier solve on the state
# recorded.
@inline function start!!(state::NewtonState, m::Newton, prob, x, Fx, p, fnorm₀)
    state = newton_start!!(state, m, x, Fx, fnorm₀)
    state = with_best(state, start_best(state.best, x, Fx, fnorm₀))
    ws = gmres_workspace(state)
    ws === nothing && return state
    reset_gmres!(ws, fnorm₀)
    return state
end

@inline start_best(::Nothing, x, Fx, fnorm₀) = nothing
@inline function start_best(b::BestIterate, x, Fx, fnorm₀)
    return BestIterate(keep_iterate!!(b.x, x), keep_residual!!(b.Fx, Fx), oftype(b.fnorm, fnorm₀))
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
        status.code, Int16(0), status.iterations, status.fnorm, status.stepnorm,
        status.step_failures, status.promoted, status.stationary, Int16(0),
        d.rank, d.nullnorm, d.gradratio, status.rate, status.contracted,
        reused_factor(state.rule, status))
    state_end = with_mixed(
        with_merit(state, init_merit(m.globalization, x), NaN),
        state.promoted, Int32(0))
    if state.rule isa DampingParameter
        state_end = with_rule(state_end, DampingParameter(R(NaN)))
    elseif state.rule isa PseudoTimeStep
        state_end = with_rule(state_end, PseudoTimeStep(R(NaN)))
    elseif state.rule isa FactorAge && m.keep_factor && !keeps_factor(state, status)
        state_end = with_rule(state_end, FactorAge(Int32(0), false, false))
    end
    return x, st, state_end
end

# Whether the last step of a solve that took `status.iterations` steps stepped along a reused factor
# or preconditioner.
@inline reused_factor(rule, status) = false
@inline function reused_factor(rule::FactorAge, status::SolverStatus)
    return rule.reused && status.iterations > Int32(0)
end

# Whether a solve under `keep_factor = true` leaves its factor, and the factor's age, to the next
# solve: after `SUCCESS` in one step, or with a last rate `θ ≤ 10⁻³` (Hairer and Wanner 1996,
# p. 121), and in the precision of the method's factorisation. A solve that is not `SUCCESS` keeps
# no factor, and neither does a promoted solve, whose factor is the one in the working precision
# that the next solve, which starts unpromoted, does not read.
@inline function keeps_factor(state::NewtonState, status::SolverStatus{R}) where {R}
    status.code == SUCCESS && !state.promoted || return false
    return status.iterations == Int32(1) || status.rate <= R(1 // 1000)
end

# Under `reuse`, a step of rate `θ ≥ 1` along a reused factor refreshes the factor at the next
# step, as a `STALLED` step does. A `NaN` rate refreshes nothing.
@inline function observe_rate!!(state::NewtonState, m::Newton, rate)
    rule = state.rule
    rule isa FactorAge || return state
    (rule.reused && rate >= one(rate)) || return state
    return with_rule(state, FactorAge(rule.age, true, rule.reused))
end

# The state whose next step factorises afresh: the age of a `reuse` factor at `0`.
@inline function drop_factor!!(state::NewtonState, m::Newton)
    state.rule isa FactorAge || return state
    return with_rule(state, FactorAge(Int32(0), false, false))
end

# The best iterate is restored after the `GMRES` counts are added, so the restored status keeps
# them.
@inline function finish!!(
        state::NewtonState, m::Newton, prob, x, Fx, status::SolverStatus{R}) where {R}
    x₊, st, state_end = newton_finish!!(state, m, x, status)
    ws = gmres_workspace(state)
    if ws !== nothing
        nf, nj = gmres_counts(ws)
        close_gmres!(ws)
        st = with_calls(st, nf, nj)
    end
    x_out, st_out = restore_best(state.best, x₊, Fx, st)
    return x_out, st_out, state_end
end

# The close of the R5 workspace: the per-solve values that `start!!` sets for the steps to read go
# back to `NaN`, so a step on the state that `solve!!` returned fails closed. The counters and the
# recorded counts stay readable.
@inline function close_gmres!(ws::GMRESWorkspace{T, R}) where {T, R}
    ws.prev_fnorm = R(NaN)
    ws.last_eta = R(NaN)
    ws.linres = R(NaN)
    ws.beta = R(NaN)
    return nothing
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
# Jacobian `Jreg` it factorised, and a refinement multiplies by it; a step along a reused factor
# passes `nothing`, and a refinement takes the JVP at `x`, shifted by `μ`.
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

# The product with the regularised Jacobian that a step already holds: a refinement of a step
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

# The product that the globalization of a matrix-free step reads, with the counters of the solve:
# that of `J` itself, without the shift `μ I` that the inner solve takes, so that the slope `φ′`
# of `Bisection` and `StrongWolfe` is that of `‖F‖²`.
@inline function merit_jvp(c::R5JVP{<:RegularizedJVP, W, TJ}) where {W, TJ}
    return R5JVP{typeof(c.prep.prep), W, TJ}(c.prep.prep, c.ws)
end
@inline merit_jvp(c) = c

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
                info.failures + one(Int32), true, info.limited, info.shortened)
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
        info.code, info.fnorm, info.stepnorm, info.failures + n, info.promoted, info.limited,
        info.shortened)
end

# The state a mixed-precision step returns with `fac` stored, for a solve that reuses its factor:
# a step with no reuse keeps the slot of `init_state`, and its state keeps one type.
@inline function stored_factor(
        state::NewtonState, fac, promoted::Bool, store::Bool)
    return store ? update_stored_factor(state, fac, promoted) : state
end

# The preconditioned product of a GMRES-IR stage: `M̂⁻¹ (J v)` in the working precision `T`, where
# `M̂` is the promoted copy of the low-precision factors and `J` the regularised Jacobian the step
# holds. A struct and not a closure, so that a step allocates none, and the slot `z` is the
# preconditioned buffer that [`preconditioned_buffer`](@ref) names.
struct PromotedProduct{J, F, M, Z}
    J::J
    fac::F
    m::M
    z::Z
end

@inline function (p::PromotedProduct)(Jv, v)
    mul!!(p.m, p.J, v)
    ldiv!!(p.z, p.fac, p.m)
    return copy!!(Jv, p.z)
end

"""
    gmres_ir!!(d, ref, state, fac, J, Fx, fnorm) -> (d_ir, ok, ran, linrel, state)

Stage 2 of the mixed-precision remedy chain of a [`Newton`](@ref) step, GMRES-IR after the LU-IR
passes of `d` have failed: a left-preconditioned GMRES in the working precision `T` on
``\\hat{U}^{-1}\\hat{L}^{-1}P J \\,\\delta d = \\hat{U}^{-1}\\hat{L}^{-1}P r_{\\mathrm{lu}}``, with
the copy of the factors in `TF` that [`promote_lu_factors!`](@ref) writes into the
working-precision factor slot, a cap of `min(ref.gmres_maxiter, n)` inner iterations, and the
tolerance `ref.gmres_tol` on the true linear residual
``r_{\\mathrm{true}} = -F(x) - J (d + \\delta d)`` recomputed in `T`, which the stage accepts up to
ten times its tolerance (Amestoy et al. 2024, p. 13; Kelley 2023, p. 7).

`d` is the direction the LU-IR passes reached, `fac` the factorisation in `TF` they solved with,
`J` the regularised Jacobian the step holds in `T`, and `fnorm` the norm ``\\|F(x)\\|`` the
tolerance is relative to. It returns the direction (the GMRES-IR one where `ok`, else `d`), whether
the stage met its bound, whether it ran at all (it does not where the solve keeps no
working-precision factor slot and no Krylov basis, where the factor is no LU factorisation, and for
any refinement that is not a [`GMRESRefinement`](@ref)), the relative true residual of the
direction it returned, and the state that carries the promoted factor slot it wrote.
"""
@inline function gmres_ir!!(d, ref, state, fac, J, Fx, fnorm::R) where {R <: Real}
    return gmres_ir_stage!(d, ref, state.refbuf, state.fac_T, fac, state, J, Fx, fnorm)
end

@inline function gmres_ir_stage!(
        d, ref, refbuf, fac_T, fac, state, J, Fx, fnorm::R) where {
        R <: Real}
    return d, false, false, zero(R), state
end

# Whether a GMRES-IR stage accepts the direction whose true relative residual is `linrel`: within ten
# times its tolerance, the allowance for the rounding between the Arnoldi estimate of the
# preconditioned system and the true residual of the direction.
@inline function accepts_gmres_ir(linrel::R, tol::R) where {R <: Real}
    return isfinite(linrel) && linrel <= R(10) * tol
end

@inline function gmres_ir_stage!(
        d, ref::GMRESRefinement, refbuf::GMRESIRBuffer, fac_T::DenseLU, fac::DenseLU, state,
        J, Fx, fnorm::R) where {R <: Real}
    return gmres_ir_promoted!(
        d, ref, refbuf, promote_lu_factors!(fac_T, fac), state, J, Fx, fnorm)
end

@inline function gmres_ir_stage!(
        d, ref::GMRESRefinement, refbuf::GMRESIRBuffer, fac_T::DeviceLU, fac::DeviceLU,
        state, J, Fx, fnorm::R) where {R <: Real}
    return gmres_ir_promoted!(
        d, ref, refbuf, promote_lu_factors!(fac_T, fac), state, J, Fx, fnorm)
end

# The stage itself, over the promoted copy of the low-precision factors: GMRES-IR corrections
# accumulate in `T` while the true linear residual `-F(x) - J d` misses `ref.gmres_tol * ‖F(x)‖`
# and the cap of inner iterations is not spent. The stop reads that residual and not the
# preconditioned estimate of the inner solve, whose norm is in the units of the direction `d` while
# the tolerance is in the units of `F(x)`: a stop on the estimate depends on the scale of the
# problem, and the same Jacobian then takes a different branch at different scales.
@inline function gmres_ir_promoted!(
        d, ref::GMRESRefinement, refbuf::GMRESIRBuffer, fac_promoted, state, J, Fx,
        fnorm::R) where {R <: Real}
    T = eltype(Fx)
    tol = R(ref.gmres_tol)
    st = with_factor_T(state, fac_promoted)
    n = length(d)
    m_cap = min(Int(ref.gmres_maxiter), n)
    op = PromotedProduct(J, fac_promoted, refbuf.tmp, refbuf.z)
    cand = copy!!(refbuf.d_try, d)
    used = 0
    linrel = zero(R)
    while true
        Jv = mul!!(refbuf.tmp, J, cand)
        rt = R(rnorm(axpby!!(refbuf.r, -one(T), Fx, -one(T), Jv)))
        linrel = iszero(fnorm) ? zero(R) : rt / fnorm
        isfinite(linrel) || return d, false, true, linrel, st
        (linrel <= tol || used >= m_cap) && break
        b = ldiv!!(refbuf.b, fac_promoted, refbuf.r)
        bnorm = R(rnorm(b))
        (isfinite(bnorm) && !iszero(bnorm)) || break
        δd, info = gmres_solve!(
            refbuf.ws, op, nothing, b, bnorm, tol, m_cap - used, m_cap - used; mgs = true)
        used += Int(info.iterations)
        axpby!!(cand, one(T), cand, one(T), δd)
    end
    accepts_gmres_ir(linrel, tol) || return d, false, true, linrel, st
    return cand, true, true, linrel, st
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
        # Stage 2 before Stage 3: a failed refinement asks for GMRES-IR with the promoted factors,
        # a failed factorisation for the refactorisation, which a singular factor cannot precondition.
        d_ir, ir_ok, ir_ran, ir_rel, st_ir = if ref_failed && !(fac.info > Int32(0))
            gmres_ir!!(d, m.refinement, state, fac, Jreg, Fx, fnorm_old)
        else
            (d, false, false, zero(R), state)
        end
        failures += one(Int32)
        # A stage that succeeds keeps the factor it solved with, as the LU-IR success below does: a
        # reused factor whose arrays were overwritten under the scale of the slot `init` made would
        # solve the next step of a scale the factor does not have.
        ir_ok && return MixedNewton(
            d_ir, Jreg, with_diag(stored_factor(st_ir, fac, false, store), diag), ir_rel,
            true, false, true, failures)
        ir_ran && (failures += one(Int32))
        fac_T, diag_T = promote_factor!!(st_ir, m, Jreg, x, Fx, fnorm_old, diag)
        st = with_mixed(
            with_diag(stored_factor(st_ir, fac_T, true, store), diag_T),
            true, st_ir.slow_steps)
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

# Every step of a `Newton`: the step of its globalization and its reuse, `newton_step!!`, and the
# record of the best iterate after it, which keeps the iterate the step returns where its norm is
# below the record's.
@inline function step!!(
        state::NewtonState, m::Newton, prob::AbstractNonlinearProblem, x, Fx, p)
    x₊, F₊, state₊, info = newton_step!!(state, m, prob, x, Fx, p)
    return x₊, F₊, with_best(state₊, take_best(state₊.best, x₊, F₊, info.fnorm)), info
end

# The step of a `Newton` whose factorisation runs in the working type `T`: one factorisation, one
# refined direction and one globalized step, with no mixed-precision state read.
@inline function newton_step!!(
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
            info.code, info.fnorm, info.stepnorm, info.failures, true, info.limited,
            info.shortened)
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
@inline function newton_step!!(
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
            info.code, info.fnorm, info.stepnorm, info.failures, true, info.limited,
            info.shortened)
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

@inline function newton_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Int32},
        prob::AbstractNonlinearProblem, x, Fx, p) where {L, G, J, RF, Reg}
    ws = gmres_workspace(state)
    ws === nothing || return gmres_step!!(state, ws, m, m.linsolve, prob, x, Fx, p)
    is_mixed(m, eltype(x)) && return mixed_step!!(state, m, prob, x, Fx, p)
    rule = state.rule
    age_fresh = fresh_age(m)
    failures = Int32(0)
    if !(rule.age == Int32(0) || rule.stalled)
        age₊ = next_age(rule, m)
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
                return reuse_result(
                    state, state.fac, state.diag, x₊, F₊, info, merit, age₊, true)
        end
    end
    Jreg, fac, diag = newton_factor(m, state, prob, x, Fx, p)
    x₊, F₊, info, merit = newton_globalize(m, state, prob, x, Fx, p, fac, Jreg)
    return reuse_result(
        state, fac, diag, x₊, F₊, bump_failures(info, failures), merit, age_fresh, false)
end

# The state and the result of a step with reuse from the factor `fac` it took, reused or fresh: a
# `SINGULAR` step resets the age of the factor, and a `STALLED` one refreshes it at the next step.
@inline function reuse_result(
        state::NewtonState, fac, diag, x₊, F₊, info, merit, age₊, reused::Bool)
    state = with_diag(with_merit(with_factor(state, fac), merit, info.fnorm), diag)
    info.code == SINGULAR &&
        return x₊, F₊, with_rule(state, FactorAge(Int32(0), false, reused)), info
    return x₊, F₊, with_rule(state, FactorAge(age₊, info.code == STALLED, reused)), info
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
    age₊ = fresh ? age_fresh : next_age(rule, m)
    while true
        was_fresh = fresh
        md = mixed_newton!!(state, m, prob, x, Fx, fnorm_old, p, fresh, true)
        st = md.state
        fresh = md.fresh
        if !md.ok
            info = StepInfo{R}(
                SINGULAR, R(rnorm(Fx)), zero(R), md.failures, st.promoted, false)
            return x, Fx, with_rule(st, FactorAge(Int32(0), false, !md.fresh)), info
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
                info.code, info.fnorm, info.stepnorm, info.failures, true, info.limited,
                info.shortened)
            st = with_mixed(st, true, Int32(0))
        elseif md.fresh && !st.promoted
            st, info = read_mixed_stall(st, info, fnorm_old, lin_rel)
        end
        st = with_merit(st, merit, info.fnorm)
        if info.code == SINGULAR
            return x₊, F₊, with_rule(st, FactorAge(Int32(0), false, !md.fresh)), info
        end
        return x₊, F₊, with_rule(st, FactorAge(age₊, info.code == STALLED, !md.fresh)), info
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
@inline function newton_step!!(
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

# The `PseudoTransient` step: one Jacobian, one factorisation of `J + (μ + 1/δ) I`, with `μ` the
# regularisation of `m`, and the full step along its direction, which is accepted. `δ` then follows
# the switched evolution relaxation. The constructor refuses reuse, refinement and a method with
# no factor, so the step reads none of them, and a mixed-precision factor solves in its own
# precision with no promotion.
@inline function newton_step!!(
        state::NewtonState, m::Newton{L, G, J, RF, Reg, Nothing},
        prob::AbstractNonlinearProblem, x, Fx, p) where {
        L, G <: PseudoTransient, J, RF, Reg}
    R = real(eltype(x))
    fnorm = R(rnorm(Fx))
    δ = state.rule.δ
    shift = inv(δ)
    if !(isfinite(δ) && isfinite(shift) && isfinite(fnorm))
        info = StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state, info
    end
    # A root is a step of zero, as for `LevenbergMarquardt`, and leaves `δ` as it is.
    if iszero(fnorm)
        info = StepInfo{R}(SUCCESS, zero(R), zero(R), Int32(0), false, false)
        return x, Fx, state, info
    end
    Jmat = jacobian!!(state.J, state.jacprep, prob, x, p)
    gr, jnorm = grad_ratio_and_norm(state.d, Jmat, Fx, fnorm)
    Jreg = regularize!!(Jmat, m.regularization + shift)
    fac = factorize!!(state.fac, m.linsolve, Jreg)
    rk, nn = factor_diagnostics(state.d, fac, Fx, fnorm)
    state_d = with_diag(state, NewtonDiagnostics{R}(rk, nn, gr, jnorm))
    if fac.info > Int32(0)
        info = StepInfo{R}(SINGULAR, fnorm, zero(R), Int32(0), false, false)
        return x, Fx, state_d, info
    end
    d = newton_solve(state, fac, Fx)
    x₊, F₊, _, info = globalize!!(state.g, Static(one(R)), prob, x, Fx, d, p, state.jvpprep)
    info.code == SUCCESS || return x₊, F₊, state_d, info
    δ₊ = pt_next_step(m.globalization, δ, fnorm, info.fnorm)
    state₊ = with_rule(with_merit(state_d, state.merit, info.fnorm), PseudoTimeStep(δ₊))
    info = StepInfo{R}(SUCCESS, info.fnorm, info.stepnorm, Int32(0), false, true)
    return x₊, F₊, state₊, info
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
        "a Newton with a matrix-free linear method takes jacobian = nothing, an AbstractADType or a JVPFunction: a Jacobian function fills a matrix, and a matrix-free solve forms none"))
    return prepare_jvp(backend, prob, Fx, x, p)
end

# Whether a `Newton` with `GMRES`, `BiCGStab` or `DQGMRES` solves `prob`: the square system of a
# `NonlinearProblem`, and the reduced problem of a `StationarityProblem`, which `stationarity.jl`
# adds.
@inline gmres_problem(::NonlinearProblem) = true
@inline gmres_problem(prob) = false

# The state of a `Newton` with `GMRES`, `BiCGStab` or `DQGMRES` at the iterate `x`: no Jacobian
# matrix, no factor and no factorization slot, a Jacobian–vector preparation, the workspace of the
# globalization, and the linear-solver workspace that every other body reads. The messages name
# the method, and the workspace and the element type of the product are the method's.
@inline function init_gmres_state(m::Newton, g::LinearMethod,
        prob::AbstractNonlinearProblem, x, p)
    name = nameof(typeof(g))
    (x isa AbstractVector && !(x isa StaticArray)) || throw(ArgumentError(
        "a Newton with $name solves an array linear system: the iterate must be an AbstractVector and not a value iterate, which is a Number or a StaticArray"))
    prob isa LeftPreconditionedProblem && throw(ArgumentError(
        "LeftPreconditioned does not take a Newton with $name as its outer method: a Newton with $name solves a NonlinearProblem, and the outer method solves the preconditioned residual; use a Newton with a factorization, such as the default LUFactorization(), or a Picard as the outer method, or put the Newton with $name in the inner method"))
    gmres_problem(prob) || throw(ArgumentError(
        "a Newton with $name solves the square system of a NonlinearProblem or a StationarityProblem; a LeastSquaresProblem takes QRFactorization() or SVDFactorization()"))
    T = eltype(x)
    R = real(T)
    TJ = krylov_jvp_eltype(g)
    Fx = buffer(x)
    ws = init_krylov_workspace(g, x)
    # a product in `jvp_eltype` is prepared on the iterate's own `TJ` slots, so the residual it
    # differentiates and the function it was prepared for share one element type
    prep = TJ === Nothing ? newton_jvp_prep(m, prob, Fx, x, p) :
           newton_jvp_prep(m, prob, ws.wT, ws.xT, p)
    iszero(m.regularization) || (prep = RegularizedJVP(prep, m.regularization))
    return NewtonState(
        nothing, r5_jvp(prep, ws, g), Fx, ws.d, nothing, nothing,
        init_globalization(m.globalization, g, x, Fx), newton_rule(m, x),
        init_merit(m.globalization, x), R(NaN), NewtonDiagnostics(R),
        nothing, nothing, false, Int32(0), ws)
end

# The Jacobian–vector preparation of the solve with its counters, and the `jvp_eltype` coupling
# where the method names one: `TJ` is a type parameter here, so the choice is a constant of the
# types and no branch reaches the inner loop.
@inline function r5_jvp(prep, ws, ::GMRES{F, P, TJ}) where {F, P, TJ}
    return TJ === Nothing ? R5JVP(prep, ws) : R5JVP{typeof(prep), typeof(ws), TJ}(prep, ws)
end
@inline r5_jvp(prep, ws, ::Union{BiCGStab, DQGMRES}) = R5JVP(prep, ws)

# The element type of the product of a matrix-free method, `Nothing` for the working type: only
# `GMRES` takes another, and the linear-solver workspace of each method.
@inline krylov_jvp_eltype(::GMRES{F, P, TJ}) where {F, P, TJ} = TJ
@inline krylov_jvp_eltype(::Union{BiCGStab, DQGMRES}) = Nothing
@inline init_krylov_workspace(g::GMRES, x) = init_gmres_workspace(g, x)
@inline init_krylov_workspace(g::BiCGStab, x) = init_bicgstab_workspace(g, x)
@inline init_krylov_workspace(g::DQGMRES, x) = init_dqgmres_workspace(g, x)

# The direction of one R5 step: the forcing term of the step, the refresh of the right
# preconditioner, and the inner solve, capped at `maxiter` inner iterations. It returns whether the
# residual, the linear residual and the direction were finite, the attained relative tolerance
# `η_ach`, the step-rule failures of the inner solve, and `‖F_k‖`.
@inline function gmres_direction!(
        state::NewtonState, g::GMRES{F, P, TJ}, ws::GMRESWorkspace,
        prob, x, Fx, p, maxiter::Int32) where {F, P, TJ}
    T = eltype(x)
    R = real(T)
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    # a value that `start!!` sets is `NaN` on a state it did not see, so a step reads no state of
    # an earlier solve
    (isfinite(fnorm) && isfinite(ws.prev_fnorm) && isfinite(ws.last_eta)) ||
        return false, zero(R), Int32(0), fnorm, false
    η = R(compute_forcing(g.forcing, ws.step, fnorm, ws.prev_fnorm, ws.last_eta, ws.linres))
    refreshed = refresh_due(state.rule)
    refreshed && refresh!(g.preconditioner, prob, x, Fx, p)
    map!!(-, ws.bb, Fx)
    op = R5Operator(state.jvpprep, prob, x, p)
    _, info = gmres_solve!(
        ws, op, g.preconditioner, ws.bb, fnorm, η, maxiter, g.restart; mgs = g.mgs)
    ws.gmres_iterations = info.iterations
    ws.max_inner = max(ws.max_inner, info.iterations)
    ws.linres = info.resnorm
    ws.last_eta = η
    ws.prev_fnorm = fnorm
    ws.beta = fnorm
    ws.step += one(Int32)
    failures = info.iterations >= maxiter && !info.converged ? Int32(1) : Int32(0)
    # a non-finite product, Hessenberg entry or preconditioner leaves a non-finite linear residual
    # or direction: the step ends with `NONFINITE` before a globalization reads either
    ok = isfinite(info.resnorm) && all(isfinite, ws.d)
    η_ach = iszero(fnorm) ? zero(R) : min(info.resnorm / fnorm, η)
    return ok, R(η_ach), failures, fnorm, refreshed
end

# Whether a step refreshes the preconditioner: every step without `reuse`, and with `reuse = k` a
# step at age 0 and the step after one that stalled or whose inner solve reached its cap.
@inline refresh_due(rule) = true
@inline refresh_due(rule::FactorAge) = rule.age == Int32(0) || rule.stalled

# The rule state after a step of a `Newton` with `GMRES`: unchanged without `reuse`; with it, the
# age of the preconditioner, as `FactorAge` holds the age of a factor, and whether the next step
# refreshes early, after a `STALLED` step or an inner solve that reached its cap.
@inline preconditioner_rule(rule, m::Newton, refreshed::Bool, code, failures) = rule
@inline function preconditioner_rule(
        rule::FactorAge, m::Newton, refreshed::Bool, code, failures)
    age₊ = refreshed ? fresh_age(m) : next_age(rule, m)
    return FactorAge(age₊, code == STALLED || failures > Int32(0), !refreshed)
end

# The step of a `Newton` with `GMRES`: the inner solve and the globalization of its direction. A line
# search gets the inexact step kind `InexactStep(η_ach)`, whose test is
# `‖F(x + α d)‖ ≤ [1 - c₁(1 - η(α))] ‖F‖` with `η(α) = |1 - α| + α η_ach`; a `DogLeg` runs
# its matrix-free trial loop on the Arnoldi relation of that same solve. That relation holds for
# one Arnoldi cycle only, so under a `DogLeg` the inner solve stops at the width of the basis,
# `min(maxiter, n)`, and never restarts.
@inline function gmres_step!!(state::NewtonState, ws::GMRESWorkspace,
        m::Newton, g::GMRES{F, P, TJ}, prob, x, Fx, p) where {F, P, TJ}
    R = real(eltype(x))
    cap = m.globalization isa DogLeg ? Int32(size(ws.V, 2) - 1) : g.maxiter
    ok, η_ach, failures, fnorm, refreshed = gmres_direction!(
        state, g, ws, prob, x, Fx, p, cap)
    ok ||
        return x, Fx, state, StepInfo{R}(NONFINITE, fnorm, zero(R), failures, false, false)
    d = ws.d
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
    dnorm = R(rnorm(d))
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, counted, x, Fx, d, p, merit_jvp(state.jvpprep),
        InexactStep(η_ach), state.merit)
    accepted_linres!(ws, info.stepnorm, dnorm)
    info = bump_failures(info, failures)
    rule = preconditioner_rule(state.rule, m, refreshed, info.code, failures)
    return x₊, F₊, with_merit(with_rule(state, rule), merit, info.fnorm), info
end

# The linear residual `‖F + α J d‖` of the step `α d` that a line search accepted, which the
# Choice 1 forcing term of the next step reads; `α = stepnorm / ‖d‖`. On one Arnoldi cycle,
# `F + α J d = V_{k+1} (-β₀ e₁ + α H̄_k y)` for `d = M⁻¹ V_k y`, and `d = Z_k y` on the flexible
# path, so the value is the norm of a reduced vector and costs no product. On a basis stored in a
# type other than the working one, the relation holds to the round-off of that type. A restarted
# solve has no such relation and keeps the residual of its direction, and so does a full step,
# whose residual that is.
@inline function accepted_linres!(ws::GMRESWorkspace{T, R}, stepnorm, dnorm) where {T, R}
    k = Int(ws.arnoldi)
    (k > 0 && ws.gmres_iterations == ws.arnoldi) || return nothing
    α = R(stepnorm) / R(dnorm)
    (isfinite(α) && α != one(R)) || return nothing
    scaled_weights!(@view(ws.yb[1:k]), @view(ws.y[1:k]), α)
    ws.linres = R(rnorm(reduced_model!(ws, ws.yb, k, ws.beta)))
    return nothing
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
    "max_inner_iterations reads the count of a Newton with a matrix-free linear method; this solve keeps none"))
@inline f_calls(::Nothing) = throw(ArgumentError(
    "f_calls reads the count of a Newton with a matrix-free linear method; this solve keeps none"))
@inline jvp_calls(::Nothing) = throw(ArgumentError(
    "jvp_calls reads the count of a Newton with a matrix-free linear method; this solve keeps none"))

# ─── R5 on a symmetric Jacobian: MINRES-QLP and Steihaug ─────────────────────
#
# A `Newton` whose linear method is a `MINRESQLP` solves the symmetric system of a
# `StationarityProblem` with `E == U` without a Jacobian matrix. Its state has the shape of the
# state of a `GMRES` solve, with a `MINRESQLPWorkspace` in the workspace slot, so `start!!`,
# `finish!!` and `newton_step!!` reach these bodies through the methods of the workspace hooks on
# that type. Under a `Steihaug` globalization the truncated conjugate gradients take the place of
# the MINRES-QLP solve, and the workspace keeps the counts.

# The workspaces whose counters and forcing record these hooks keep in the fields of the same
# names: those of `MINRESQLP`, `BiCGStab` and `DQGMRES`.
const CountedKrylovWorkspace{T, R} = Union{
    MINRESQLPWorkspace{T, R}, BiCGStabWorkspace{T, R}, DQGMRESWorkspace{T, R}}

@inline gmres_counts(ws::CountedKrylovWorkspace) = (ws.f_calls, ws.jvp_calls)

@inline function reset_gmres!(ws::CountedKrylovWorkspace{T, R}, fnorm₀) where {T, R}
    ws.f_calls = Int32(1)
    ws.jvp_calls = Int32(0)
    ws.iterations = Int32(0)
    ws.max_inner = Int32(0)
    ws.step = Int32(0)
    ws.prev_fnorm = R(fnorm₀)
    ws.last_eta = zero(R)
    ws.linres = zero(R)
    return nothing
end

@inline function close_gmres!(ws::CountedKrylovWorkspace{T, R}) where {T, R}
    ws.prev_fnorm = R(NaN)
    ws.last_eta = R(NaN)
    ws.linres = R(NaN)
    return nothing
end

# Whether a `Newton` with `MINRESQLP` solves `prob`: none here, and the reduced problem of a
# `StationarityProblem` with `E == U`, which `stationarity.jl` adds, raises its own error where the
# index sets differ.
@inline function check_symmetric_problem(prob)
    throw(ArgumentError(
        "a Newton with MINRESQLP solves the symmetric system of a StationarityProblem with E == U; a NonlinearProblem takes GMRES()"))
end

# The state of a `Newton` with `MINRESQLP` at the iterate `x`, as `init_gmres_state` builds it for
# `GMRES`: no Jacobian matrix and no factor, a counted Jacobian–vector preparation, the workspace of
# the globalization, and the linear-solver workspace.
@inline function init_minresqlp_state(m::Newton, g::MINRESQLP, prob, x, p)
    (x isa AbstractVector && !(x isa StaticArray)) || throw(ArgumentError(
        "a Newton with MINRESQLP solves an array linear system: the iterate must be an AbstractVector and not a value iterate, which is a Number or a StaticArray"))
    check_symmetric_problem(prob)
    R = real(eltype(x))
    Fx = buffer(x)
    ws = init_minresqlp_workspace(g, x)
    prep = newton_jvp_prep(m, prob, Fx, x, p)
    iszero(m.regularization) || (prep = RegularizedJVP(prep, m.regularization))
    return NewtonState(
        nothing, R5JVP(prep, ws), Fx, ws.x, nothing, nothing,
        init_globalization(m.globalization, g, x, Fx), newton_rule(m, x),
        init_merit(m.globalization, x), R(NaN), NewtonDiagnostics(R),
        nothing, nothing, false, Int32(0), ws)
end

# The forcing term of a step on the MINRES-QLP workspace, and `‖F_k‖`; `ok` is `false` where the
# residual norm or a value that `start!!` sets is not finite, so a step on a state that no `start!!`
# saw fails closed.
@inline function minresqlp_forcing(state::NewtonState, g::MINRESQLP, ws::MINRESQLPWorkspace, Fx)
    R = real(eltype(Fx))
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    ok = isfinite(fnorm) && isfinite(ws.prev_fnorm) && isfinite(ws.last_eta)
    ok || return false, zero(R), fnorm
    η = R(compute_forcing(g.forcing, ws.step, fnorm, ws.prev_fnorm, ws.last_eta, ws.linres))
    return true, η, fnorm
end

# The record of one step on the workspace: its inner iterations, its linear residual, its forcing
# term and its residual norm, which the forcing term of the next step reads.
@inline function record_inner!(ws::CountedKrylovWorkspace{T, R}, iterations, linres, η,
        fnorm) where {T, R}
    ws.iterations = Int32(iterations)
    ws.max_inner = max(ws.max_inner, Int32(iterations))
    ws.linres = R(linres)
    ws.last_eta = R(η)
    ws.prev_fnorm = R(fnorm)
    ws.step += one(Int32)
    return nothing
end

# The linear residual `‖F + α J d‖` of the step `α d` that a line search accepted, which the
# Choice 1 forcing term of the next step reads; `α = stepnorm / ‖d‖`. The direction `d` minimises
# `‖F + J d‖` over its Krylov space `K`, so the residual `r = F + J d` is orthogonal to `J K`, which
# holds `J d`: `Fᴴ r = ‖r‖²`, and `‖F + α J d‖² = (α - 1)² (‖F‖ - ‖r‖) (‖F‖ + ‖r‖) + ‖r‖²` costs no
# product. Both terms are nonnegative for every `α ≥ 0`, also for a line search that accepts more
# than twice the direction, and `‖r‖` enters as `min(‖r‖, ‖F‖)`, the bound that `Fᴴ r = ‖r‖²`
# gives by Cauchy–Schwarz. The two factors enter through their square roots, which do not overflow
# for `‖F‖ ≤ floatmax / 2`. A direction whose last component counts as zero does not minimise over
# the whole space, and the value is then an estimate. A full step, and a direction of length zero,
# where `α` is not finite, keep the residual of the direction.
@inline function accepted_linres!(ws::MINRESQLPWorkspace{T, R}, stepnorm, dnorm, fnorm,
        resnorm) where {T, R}
    α = R(stepnorm) / R(dnorm)
    (isfinite(α) && α != one(R)) || return nothing
    ρ = min(R(resnorm), R(fnorm))
    ws.linres = hypot((α - one(R)) * sqrt(R(fnorm) - ρ) * sqrt(R(fnorm) + ρ), ρ)
    return nothing
end

# The symmetric Jacobian of a `MINRESQLP` step as the operator of the factorised `DogLeg` loop:
# `mul!!(y, J, v)` is the counted Hessian–vector product, and `J'` is `J`, so the Cauchy direction
# `J' F` is one product. The loop checks every product for a non-finite entry.
struct SymmetricJVP{O}
    op::O
end

Base.adjoint(J::SymmetricJVP) = J
@inline mul!!(y::AbstractVecOrMat, J::SymmetricJVP, v) = J.op(y, v)
@inline finite_product(::SymmetricJVP, y) = all(isfinite, y)

# The step of a `Newton` with `MINRESQLP`: the inner solve and the globalization of its direction,
# or the Steihaug step in its place. A line search gets the inexact step kind `InexactStep(η_ach)`,
# as for `GMRES`; a `DogLeg` runs the factorised trial loop on `SymmetricJVP`, whose Cauchy point
# takes two products and whose predicted reduction takes one product per trial.
@inline function gmres_step!!(state::NewtonState, ws::MINRESQLPWorkspace,
        m::Newton, g::MINRESQLP, prob, x, Fx, p)
    R = real(eltype(x))
    counted = counting_problem(prob, ws)
    op = R5Operator(state.jvpprep, counted, x, p)
    ok, η, fnorm = minresqlp_forcing(state, g, ws, Fx)
    ok || return x, Fx, state,
    StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
    if m.globalization isa Steihaug
        x₊, F₊, radius, info, linres, iterations = steihaug_step!!(
            state.g, counted, x, Fx, p, op, η, g.maxiter, state.rule.radius)
        record_inner!(ws, iterations, linres, η, fnorm)
        state₊ = with_rule(state, TrustRadius(radius, state.rule.radius₀))
        return x₊, F₊, with_merit(state₊, state.merit, info.fnorm), info
    end
    map!!(-, ws.bb, Fx)
    d, inner = minresqlp_solve!(ws, op, ws.bb, fnorm, η, g.maxiter)
    record_inner!(ws, inner.iterations, inner.resnorm, η, fnorm)
    failures = inner.iterations >= g.maxiter && !inner.converged ? Int32(1) : Int32(0)
    # a non-finite product leaves a residual of `NaN`: the step ends before a globalization reads
    # the direction
    (isfinite(inner.resnorm) && all(isfinite, d)) ||
        return x, Fx, state, StepInfo{R}(NONFINITE, fnorm, zero(R), failures, false, false)
    if m.globalization isa DogLeg
        nd = R(rnorm(d))
        newton = isfinite(nd) && !iszero(nd)
        x₊, F₊, radius, info = dogleg_step(
            state.g, counted, x, Fx, p, SymmetricJVP(op), d, newton, state.rule.radius)
        state₊ = with_rule(state, TrustRadius(radius, state.rule.radius₀))
        info = bump_failures(info, failures)
        return x₊, F₊, with_merit(state₊, state.merit, info.fnorm), info
    end
    η_ach = iszero(fnorm) ? zero(R) : min(inner.resnorm / fnorm, η)
    dnorm = R(rnorm(d))
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, counted, x, Fx, d, p, merit_jvp(state.jvpprep),
        InexactStep(R(η_ach)), state.merit)
    accepted_linres!(ws, info.stepnorm, dnorm, fnorm, inner.resnorm)
    info = bump_failures(info, failures)
    return x₊, F₊, with_merit(state, merit, info.fnorm), info
end

# ─── R5 with fixed storage: BiCGStab and DQGMRES ─────────────────────────────
#
# A `Newton` whose linear method is a `BiCGStab` or a `DQGMRES` has the state of a `GMRES` solve,
# with the workspace of its method in the workspace slot, so `start!!`, `finish!!` and
# `newton_step!!` reach these bodies through the methods of the workspace hooks on those types. The
# step globalizes the direction with a line search or `Static`; a `DogLeg` is refused where the
# `Newton` is built.

const FixedKrylovWorkspace = Union{BiCGStabWorkspace, DQGMRESWorkspace}

@inline krylov_solve!(
    ws::BiCGStabWorkspace, op, P, b, bnorm, tol, maxiter) = bicgstab_solve!(
    ws, op, P, b, bnorm, tol, maxiter)
@inline krylov_solve!(ws::DQGMRESWorkspace, op, P, b, bnorm, tol, maxiter) = dqgmres_solve!(
    ws, op, P, b, bnorm, tol, maxiter)

# The step of a `Newton` with `BiCGStab` or `DQGMRES`: the forcing term of the step, the refresh of
# the right preconditioner, the inner solve, and the globalization of its direction under the
# inexact step kind `InexactStep(η_ach)`, as for `GMRES`. Neither method has a relation that gives
# the linear residual of a shortened step without a product, so the Choice 1 forcing term of the
# next step reads the residual of the direction, as after a restarted `GMRES` solve.
@inline function gmres_step!!(state::NewtonState, ws::FixedKrylovWorkspace,
        m::Newton, g::Union{BiCGStab, DQGMRES}, prob, x, Fx, p)
    R = real(eltype(x))
    fnorm = R(residual_norm(Fx, state.merit, state.fnorm))
    # a value that `start!!` sets is `NaN` on a state it did not see, so a step reads no state of
    # an earlier solve
    (isfinite(fnorm) && isfinite(ws.prev_fnorm) && isfinite(ws.last_eta)) ||
        return x, Fx, state, StepInfo{R}(NONFINITE, fnorm, zero(R), Int32(0), false, false)
    η = R(compute_forcing(g.forcing, ws.step, fnorm, ws.prev_fnorm, ws.last_eta, ws.linres))
    refresh!(g.preconditioner, prob, x, Fx, p)
    map!!(-, ws.bb, Fx)
    op = R5Operator(state.jvpprep, prob, x, p)
    d, inner = krylov_solve!(ws, op, g.preconditioner, ws.bb, fnorm, η, g.maxiter)
    record_inner!(ws, inner.iterations, inner.resnorm, η, fnorm)
    failures = inner.iterations >= g.maxiter && !inner.converged ? Int32(1) : Int32(0)
    # a non-finite product leaves a residual of `NaN`: the step ends before a globalization reads
    # the direction
    (isfinite(inner.resnorm) && all(isfinite, d)) ||
        return x, Fx, state, StepInfo{R}(NONFINITE, fnorm, zero(R), failures, false, false)
    η_ach = iszero(fnorm) ? zero(R) : min(inner.resnorm / fnorm, η)
    x₊, F₊, _, info, merit = globalize_merit!!(
        state.g, m.globalization, counting_problem(prob, ws), x, Fx, d, p,
        merit_jvp(state.jvpprep),
        InexactStep(R(η_ach)), state.merit)
    info = bump_failures(info, failures)
    return x₊, F₊, with_merit(state, merit, info.fnorm), info
end

# ─── The Kronecker stage solver ──────────────────────────────────────────────
#
# A `Newton` whose linear method is a `KroneckerFactorization` solves the stage system that a
# `NonlinearProblem` with `stage_matrix` carries. Its state holds the `n × n` Jacobian of the vector
# field in place of the `s n × s n` Jacobian of the stacked residual, and its factor holds the `s`
# blocks of `I - h A ⊗ J`. Every step body of `Newton` runs unchanged: only the factorisation of a
# refresh, `newton_factor`, has a method of its own.

# The state of a `Newton` with `KroneckerFactorization`: no Jacobian preparation, the product of the
# stacked residual by forward mode for a line search, which measures its slope, and none for
# `Static`.
function init_kronecker_state(
        m::Newton, k::KroneckerFactorization{TF}, prob, x, p) where {TF}
    x isa Union{Number, StaticArray} && throw(ArgumentError(
        "KroneckerFactorization solves the stage system in place, and a value iterate (a Number or a StaticArray) has no Kronecker solve; pass a Vector"))
    x isa AbstractGPUArray && throw(ArgumentError(
        "KroneckerFactorization has no device factor, so a device iterate has no Kronecker solve; pass a Vector"))
    TF === Nothing || throw(ArgumentError(
        "KroneckerFactorization takes no factorisation precision: its blocks factorise in the working precision"))
    m.globalization isa DogLeg && throw(ArgumentError(
        "Newton with KroneckerFactorization takes no DogLeg globalization: its trial loop needs the Jacobian of the stacked residual"))
    m.globalization isa LevenbergMarquardt && throw(ArgumentError(
        "Newton with KroneckerFactorization takes no LevenbergMarquardt globalization: its damped step needs the Jacobian of the stacked residual"))
    is_refining(m.refinement) && throw(ArgumentError(
        "Newton with KroneckerFactorization takes no refinement: its blocks factorise in the working precision"))
    iszero(m.regularization) || throw(ArgumentError(
        "Newton with KroneckerFactorization takes no regularization: a shift of I - h A ⊗ J has no block form"))
    m.jacobian === nothing || throw(ArgumentError(
        "Newton with KroneckerFactorization takes no jacobian: it reads the Jacobian of the vector field from the problem's stage_jacobian"))
    A = stage_matrix(prob)
    A === nothing && throw(ArgumentError(
        "KroneckerFactorization needs a NonlinearProblem with stage_matrix and stage_jacobian"))
    hasproperty(p, :h) || throw(ArgumentError(
        "KroneckerFactorization reads the step from p.h, and the parameters have no property h"))
    fac = kronecker_factor(k.block, A, x)
    T = eltype(x)
    R = real(T)
    n = length(x) ÷ size(A, 1)
    Fx = buffer(x)
    jvpprep = m.globalization isa Static ? nothing :
              prepare_jvp(AutoForwardDiff(), prob, Fx, x, p)
    return NewtonState(
        nothing, jvpprep, Fx, buffer(x), fac, similar(x, T, n, n),
        init_globalization(m.globalization, k, x, Fx), newton_rule(m, x),
        init_merit(m.globalization, x), R(NaN), NewtonDiagnostics(R), nothing, nothing, false,
        Int32(0))
end

# The refresh of a Kronecker step: `jac!` writes the Jacobian of the field into the state, and the
# blocks are factorised at the step `p.h`. The diagnostics are those of a factor without a rank.
@inline function newton_factor(
        m::Newton{<:KroneckerFactorization}, state::NewtonState, prob, x, Fx, p)
    R = real(eltype(x))
    stage_jacobian(prob)(state.J, x, p)
    fac = factorize_stages!!(state.fac, m.linsolve.block, state.J, R(p.h))
    return state.J, fac, NewtonDiagnostics(R)
end

# The tangent solve of implicit differentiation factorises the Jacobian of the stacked residual,
# which a `Newton` with `KroneckerFactorization` never forms, so `solve` refuses a dual input for
# it before the primal solve, as it does for `GMRES`.
function solve_implicit(
        ::NonlinearProblem, x₀, method::Newton{<:KroneckerFactorization}, p,
        options::Union{Nothing, Options}, kwargs)
    throw(ArgumentError(
        "solve does not accept ForwardDiff.Dual inputs for a Newton with KroneckerFactorization; its tangent solve needs the Jacobian of the stacked residual, which this solver does not form"))
end
