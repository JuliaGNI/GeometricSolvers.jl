# Deflation of known roots (Farrell, Birkisson and Funke 2015): a problem wrapper whose residual is
# `M(x) F(x)` and whose Jacobian is the product-rule exact one, so that a solve from the same start
# avoids the roots it is given.
#
# `M(x) = ∏ᵢ mᵢ(x)` with `mᵢ(x) = ‖x - rᵢ‖₂^{-power} + shift`, and
# `∇M(x) = Σᵢ (∏_{j≠i} mⱼ) ∇mᵢ(x)` with `∇mᵢ(x) = -power ‖u‖^{-power-2} u`, `u = x - rᵢ`. The
# accumulation is the division-free recurrence `g ← mᵢ g + M gᵢ`, `M ← M mᵢ`, so no root's product
# is formed twice and no factor is divided out.
#
# The distance `‖u‖` is taken on the iterate the solver steps on. For a `NonlinearProblem` that is
# the iterate itself; for a wrapped `StationarityProblem` it is the unknown part `x_U`, which is
# what the solve iterates on, so the roots of a stationarity problem are full states and the
# distance is over their `U` entries.

"""
    Deflated(prob, roots; power = 2, shift = 1)
    Deflated(prob, roots, power, shift)

Shifted norm deflation of the known roots `roots` around [`NonlinearProblem`](@ref) or
[`StationarityProblem`](@ref) `prob`: the problem whose residual is

```math
F_{\\mathrm{def}}(x) = M(x) F(x), \\qquad M(x) = \\prod_{i=1}^{k} \\left( \\|x - r_i\\|_2^{-p} + \\sigma \\right),
```

and whose Jacobian is the product rule of that product,

```math
J_{\\mathrm{def}}(x) = M(x) J(x) + F(x) \\nabla M(x)^\\top,
```

so that [`Newton`](@ref) started from the same `x₀` converges to a root other than `r₁, …, r_k`
(Farrell, Birkisson and Funke 2015, *Deflation techniques for finding distinct solutions of
nonlinear partial differential equations*). Near a simple root `‖M(x) F(x)‖` grows like
`‖x - rᵢ‖^{1-p}`, which is what pushes the iterate away from it.

The steps and the line search take the deflated residual, but the stopping test of
[`Options`](@ref) reads the inner one: `status.fnorm` is `‖F(x)‖`, `f_abstol` bounds it, and
`f_reltol` is relative to `‖F(x₀)‖`, so a deflated root meets the tolerance of a solve of `prob`.

`power` (`p > 0`) and `shift` (`σ > 0`) are stored in `T = float(real(eltype(first(roots))))`, so
`Float32` roots give `power::Float32` and a `Float32` product; a solve needs an iterate of that
element type and a real one.

# Roots

`roots` is a non-empty `Tuple` of roots — a `Number` for a `Number` iterate, an `SVector` for an R1
iterate, an array of the iterate's own back end for an R3 iterate (a `Vector` for a `Vector`, a
`JLArray` or an `MtlArray` for one on that device) — or an `AbstractVector` of real vectors. Every
root has the length of the iterate and finite real entries, and the roots of one set share their
element type and are all numbers or all vectors. A solve on an array iterate raises an
`ArgumentError` for a root on another back end or a static root, because the deflation broadcasts
`x .- r` on the iterate's back end. The `AbstractVector` form is not `isbits`, so
[`solve_batch!`](@ref) refuses it: the batch kernel takes the problem by value.

For a wrapped [`StationarityProblem`](@ref) a root is a full state, as [`solve`](@ref) returns it,
and the distance is taken on the unknowns `x_U`: `M` and `∇M` are functions of the `|U|` entries the
solve steps on, so `∇M` has length `|U|` whatever the length of the state.

# Evaluation

At a deflated root (`x = rᵢ`) the factor `mᵢ` is `Inf`, so a solve from `x₀ = rᵢ` returns
`NONFINITE` at `0` iterations, and a trial step onto `rᵢ` has a non-finite merit, which
`Backtracking` shrinks away from by the step `min(0.1α, √α √αmin)`.
A factor that overflows (`‖x - rᵢ‖^{-p} = Inf`, as at `10^{-200}` in `Float64`) does the same.
Every coordinate of `∇mᵢ` is evaluated unit-scaled, `-(p ((uⱼ/d) ϕ)/d)` with `d = ‖u‖` and
`ϕ = d^{-p}`, which forms no `ϕ/d²` and so leaves no `NaN` where `uⱼ` is zero: the unscaled
`-(p ϕ/d²) uⱼ` overflows to `Inf` at `d = 10^{-100}` and gives `Inf * 0.0 = NaN` there.
Where `‖u‖^{-p-1}` overflows although `‖u‖^{-p}` does not (between about `10^{-155}` and
`10^{-103}` in `Float64` for `p = 2`), `∇mᵢ` has an infinite coordinate and the Jacobian is not
finite.

A [`Newton`](@ref) with a callable `jacobian` raises an `ArgumentError`, because `Newton` would
take it as the deflated Jacobian. Give an analytic Jacobian of `F` to the wrapped
`NonlinearProblem` instead.

# Implicit differentiation

[`solve`](@ref) on a `Deflated(NonlinearProblem)` with a `ForwardDiff.Dual` in `x₀` or `p` runs the
primal deflated solve and differentiates the **undeflated** `prob` implicitly at the root it found,
as [`solve`](@ref) does on the problem itself. A `Deflated(StationarityProblem)` refuses a dual with
an `ArgumentError`, as a [`StationarityProblem`](@ref) does.

# R2

[`solve_batch!`](@ref) solves every system of a batch under one `Deflated`, so each system is
deflated at the same roots and ends at a root distinct from them. The batch of a wrapped
[`StationarityProblem`](@ref) takes the datum of its solves from the keyword `b`, shared or
[`PerSystem`](@ref), or from the problem's own `b`.

```jldoctest
julia> prob = NonlinearProblem((x, p) -> x^3 - x);

julia> def = Deflated(prob, (0.0,));

julia> solve(def, 0.1, Newton())[1] ≈ 1 || solve(def, 0.1, Newton())[1] ≈ -1
true

julia> solve(Deflated(prob, (0.0,)), 0.0, Newton())[2].code === NONFINITE
true
```
"""
struct Deflated{P <: AbstractNonlinearProblem, RTs, T <: AbstractFloat} <:
       AbstractNonlinearProblem
    prob::P
    roots::RTs
    power::T
    shift::T
    # `Adapt` rebuilds a `Deflated` through this constructor, and a device kernel cannot format a
    # message, so every message here is constant.
    function Deflated{P, RTs, T}(
            prob::P, roots::RTs, power::T,
            shift::T) where {
            P <: AbstractNonlinearProblem, RTs, T <: AbstractFloat}
        _check_deflated_problem(prob)
        (power > zero(T) && isfinite(power)) ||
            throw(ArgumentError("Deflated power must be finite and > 0"))
        (shift > zero(T) && isfinite(shift)) ||
            throw(ArgumentError("Deflated shift must be finite and > 0"))
        _check_deflated_roots(roots)
        return new{P, RTs, T}(prob, roots, power, shift)
    end
end

# The problems that define the iterate's residual and Jacobian themselves. A `LeastSquaresProblem`
# has more residuals than unknowns and a rank-1 update of its Jacobian is not a deflation of its
# roots; a `Reparametrized` and a `Deflated` are wrappers that a second wrapper would deflate in
# the iterate of the first. The reduced stationarity problem is the internal problem over the
# unknowns `x_U`.
@inline function _check_deflated_problem(prob::NonlinearProblem)
    prob.jacobian_prototype === nothing || throw(ArgumentError(
        "Deflated needs a NonlinearProblem with no Jacobian prototype: the rank-1 product-rule term writes a dense Jacobian, so a sparse prototype is refused"))
    return nothing
end
@inline function _check_deflated_problem(
        prob::Union{StationarityProblem, _ReducedStationarityProblem})
    return nothing
end
@inline function _check_deflated_problem(prob)
    throw(ArgumentError(
        "Deflated needs a NonlinearProblem or a StationarityProblem"))
end

@inline _is_root(r::Real) = true
@inline _is_root(r::AbstractVector) = true
@inline _is_root(r) = false

# A `Number` iterate has `Number` roots, and a vector iterate has vector roots: `x - r` has no
# method across the two.
@inline _scalar_iterate(x::Number) = true
@inline _scalar_iterate(x) = false

@inline function _finite_root(r::Real)
    return isfinite(r)
end
@inline function _finite_root(r::AbstractVector)
    return eltype(r) <: Real && all(isfinite, r)
end

# A root is non-empty, of the first root's length and element type, and finite. The length and the
# storage of the iterate itself are checked where a solve starts, because only then is the iterate
# known.
@inline function _check_deflated_roots(roots::Tuple)
    isempty(roots) && throw(ArgumentError("Deflated needs at least one root"))
    n = length(first(roots))
    n >= 1 || throw(ArgumentError("Deflated roots must be non-empty"))
    T = eltype(first(roots))
    scalar = _scalar_iterate(first(roots))
    for r in roots
        _is_root(r) || throw(ArgumentError(
            "Deflated roots must be real numbers or real vectors"))
        length(r) == n || throw(ArgumentError(
            "Deflated needs roots of one length"))
        eltype(r) === T || throw(ArgumentError(
            "Deflated needs roots of one element type"))
        _scalar_iterate(r) === scalar || throw(ArgumentError(
            "Deflated needs roots of one storage"))
        _finite_root(r) || throw(ArgumentError(
            "Deflated needs roots with finite real entries"))
    end
    return nothing
end

@inline function _check_deflated_roots(
        roots::AbstractVector{<:AbstractVector{<:Real}})
    isempty(roots) && throw(ArgumentError("Deflated needs at least one root"))
    n = length(first(roots))
    n >= 1 || throw(ArgumentError("Deflated roots must be non-empty"))
    T = eltype(first(roots))
    for r in roots
        length(r) == n || throw(ArgumentError(
            "Deflated needs roots of one length"))
        eltype(r) === T || throw(ArgumentError(
            "Deflated needs roots of one element type"))
        _finite_root(r) || throw(ArgumentError(
            "Deflated needs roots with finite real entries"))
    end
    return nothing
end

@inline function _check_deflated_roots(roots)
    throw(ArgumentError(
        "Deflated roots must be a non-empty Tuple of roots or an AbstractVector of real vectors"))
end

@inline function _root_float_type(roots)
    (roots isa Tuple || roots isa AbstractVector) || throw(ArgumentError(
        "Deflated roots must be a non-empty Tuple of roots or an AbstractVector of real vectors"))
    isempty(roots) && throw(ArgumentError("Deflated needs at least one root"))
    return float(real(eltype(first(roots))))
end

function Deflated(prob::P, roots, power::Real, shift::Real) where {P <:
                                                                   AbstractNonlinearProblem}
    T = _root_float_type(roots)
    return Deflated{P, typeof(roots), T}(prob, roots, T(power), T(shift))
end

function Deflated(prob::AbstractNonlinearProblem, roots; power::Real = 2, shift::Real = 1)
    return Deflated(prob, roots, power, shift)
end

Adapt.@adapt_structure Deflated

# ─── The factor `M(x)` and its gradient `∇M(x)` ──────────────────────────────

# One coordinate of `∇mᵢ(x)`, unit-scaled: `uⱼ/d` is at most `1` in magnitude, so no quotient of
# `ϕ` by `d²` is formed. `ifelse` evaluates both branches, as a device requires, and the zero
# coordinate is returned without a division by zero where `d` is small enough that `ϕ` overflows.
@inline function _deflate_gradient_entry(uj, d, ϕ, power)
    return ifelse(iszero(uj), zero(uj), -(power * ((uj / d) * ϕ) / d))
end

@inline function _deflate_gradient(u::Number, d, ϕ, power)
    return _deflate_gradient_entry(u, d, ϕ, power)
end
@inline function _deflate_gradient(u::StaticVector, d, ϕ, power)
    return _deflate_gradient_entry.(u, d, ϕ, power)
end

# `M(x)` alone: the residual is scaled by it and no gradient is needed.
@inline function _deflate_scale(
        def::Deflated{<:Any, <:Any, T},
        x::Union{StaticVector, Number}) where {T}
    M = one(T)
    for r in def.roots
        d = rnorm(x - r)
        M = M * (d^(-def.power) + def.shift)
    end
    return M
end

# `(M(x), ∇M(x))` for a value iterate: one value of the gradient's storage, so an `SVector` iterate
# gives an `SVector` and no allocation.
@inline function _deflate_factors(
        def::Deflated{<:Any, <:Any, T},
        x::Union{StaticVector, Number}) where {T}
    M = one(T)
    g = zero(x)
    for r in def.roots
        u = x - r
        d = rnorm(u)
        ϕ = d^(-def.power)
        m = ϕ + def.shift
        g = m * g + M * _deflate_gradient(u, d, ϕ, def.power)
        M = M * m
    end
    return M, g
end

# `M(x)` for an array iterate, with `us` the difference buffer the evaluation needs.
function _deflate_scale!(
        us::AbstractArray, def::Deflated{<:Any, <:Any, T},
        x::AbstractVector) where {T}
    M = one(T)
    for r in def.roots
        us .= x .- r
        M = M * (rnorm(us)^(-def.power) + def.shift)
    end
    return M
end

# `M(x)` for an array iterate, with `∇M(x)` written into `gs`: every update is one broadcast, so no
# temporary is allocated.
function _deflate_factors!(
        gs::AbstractArray, us::AbstractArray, def::Deflated{<:Any, <:Any, T},
        x::AbstractVector) where {T}
    gs .= zero(T)
    M = one(T)
    for r in def.roots
        us .= x .- r
        d = rnorm(us)
        ϕ = d^(-def.power)
        m = ϕ + def.shift
        gs .= m .* gs .+ M .* _deflate_gradient_entry.(us, d, ϕ, def.power)
        M = M * m
    end
    return M
end

# ─── R1: the residual, the Jacobian and the Jacobian-vector product ──────────

# The inner problem's residual and Jacobian at one point. A `StationarityProblem` reads both from
# one nested dual evaluation, `gradient_hessian!!`, which is what its Jacobian costs anyway.
@inline function _inner_res_jac(backend, prob::NonlinearProblem, x, p)
    F = residual!!(nothing, prob, x, p)
    J = prob.J === nothing ? jacobian!!(nothing, backend, prob, x, p) : prob.J(x, p)
    return F, J
end
@inline function _inner_res_jac(backend, prob, x, p)
    return gradient_hessian!!(nothing, nothing, backend, prob, x, p)
end

# With no residual buffer and a value iterate there is nothing to allocate for the inner derivative,
# so the preparation is the back end itself and the R1 state stays `isbits`.
@inline function prepare_ad(
        backend::AutoForwardDiff, ::Deflated, ::Nothing,
        x::Union{StaticVector, Number}, p)
    return backend
end

@inline function residual!!(
        ::Nothing, def::Deflated, x::Union{StaticVector, Number}, p)
    return _deflate_scale(def, x) * residual!!(nothing, def.prob, x, p)
end

# The Jacobian `M J + F ∇Mᵀ` and the product `M J v + ⟨∇M, v⟩ F` of the same point. `F` and `J` are
# the inner problem's own, and the rank-1 term is the outer product of the residual and `∇M`, so
# neither the Jacobian nor the product evaluates the deflated system it is forming.
@inline function _deflated_jacobian(def::Deflated, backend, x, p)
    F, J = _inner_res_jac(backend, def.prob, x, p)
    M, g = _deflate_factors(def, x)
    return M * J + F * transpose(g)
end

@inline function _deflated_jacobian(def::Deflated, ::Nothing, x, p)
    F = residual!!(nothing, def.prob, x, p)
    J = jacobian!!(nothing, nothing, def.prob, x, p)
    M, g = _deflate_factors(def, x)
    return M * J + F * transpose(g)
end

@inline function _deflated_jvp(def::Deflated, backend, x, v, p)
    F = residual!!(nothing, def.prob, x, p)
    Jv = jvp!!(nothing, backend, def.prob, x, v, p)
    M, g = _deflate_factors(def, x)
    return M * Jv + rdot(g, v) * F
end

# The two R1 storages are separate methods, with the iterate's parameters spelled out: the base's
# `AutoForwardDiff` methods are written on a `StaticVector{N, T}` and a `T <: Number`, so a method
# over the union, or with an untyped direction, is ambiguous with them.
@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, def::Deflated, x::StaticVector{N, T},
        p) where {N, T}
    return _deflated_jacobian(def, backend, x, p)
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, def::Deflated, x::T, p) where {T <: Number}
    return _deflated_jacobian(def, backend, x, p)
end

@inline function jacobian!!(
        ::Nothing, ::Nothing, def::Deflated, x::Union{StaticVector, Number}, p)
    return _deflated_jacobian(def, nothing, x, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, def::Deflated, x::StaticVector{N, T},
        v::StaticVector{N, T}, p) where {N, T}
    return _deflated_jvp(def, backend, x, v, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, def::Deflated, x::T, v::T,
        p) where {T <: Number}
    return _deflated_jvp(def, backend, x, v, p)
end

@inline function jvp!!(
        ::Nothing, ::Nothing, def::Deflated, x::StaticVector{N, T}, v::StaticVector{N, T},
        p) where {N, T}
    return _deflated_jvp(def, nothing, x, v, p)
end

@inline function jvp!!(
        ::Nothing, ::Nothing, def::Deflated, x::T, v::T, p) where {T <: Number}
    return _deflated_jvp(def, nothing, x, v, p)
end

# ─── R3: the residual, the Jacobian and the Jacobian-vector product ──────────

# The workspace of a deflated problem on an array iterate (R3): the inner derivative's
# preparation, the residual and the gradient of the rank-1 update, one difference buffer, and —
# for an inner problem with an analytic Jacobian only, whose Jacobian-vector product the base
# seam reads from a matrix — the matrix that product needs. The Jacobian itself needs no second
# matrix: the inner Jacobian is written into the caller's buffer, which the rank-1 update then
# completes in place.
struct DeflatedPrep{INNER, F, G, U, JX}
    inner::INNER
    Fs::F
    gs::G
    us::U
    Jx::JX
end

function _prepare_deflated_r3(
        backend, def::Deflated, r::AbstractArray, x::AbstractArray, p)
    check_iterate_shape(x)
    n = length(x)
    T = eltype(x)
    innerprob = def.prob
    Jx = innerprob isa NonlinearProblem && innerprob.J !== nothing ?
         similar(x, T, n, n) : nothing
    inner = _prepare_inner_r3(backend, innerprob, r, x, p)
    return DeflatedPrep(
        inner, similar(x, T, n), similar(x, T, n), similar(x, T, n), Jx)
end

# The back end is named in the signature as well as the problem: the base prepares a chunked
# `AutoForwardDiff` for any problem, and one method on `Deflated` alone would be ambiguous with it.
@inline function prepare_ad(
        backend::AutoForwardDiff, def::Deflated, r::AbstractArray, x::AbstractArray, p)
    return _prepare_deflated_r3(backend, def, r, x, p)
end

@inline function prepare_ad(
        backend::AbstractADType, def::Deflated, r::AbstractArray, x::AbstractArray, p)
    return _prepare_deflated_r3(backend, def, r, x, p)
end

# The residual is what `solve!!` evaluates before a preparation is handed to it, so this method takes
# no workspace: it reads `M(x)` with the output buffer itself as the difference scratch, writes the
# inner residual over it, and scales that. No allocation reaches a solve.
function residual!!(y::AbstractArray, def::Deflated, x::AbstractVector, p)
    M = _deflate_scale!(y, def, x)
    residual!!(y, def.prob, x, p)
    return y .*= M
end

function residual!!(
        y::AbstractArray, prep::DeflatedPrep, def::Deflated, x::AbstractVector, p)
    residual!!(y, def.prob, x, p)
    return y .*= _deflate_scale!(prep.us, def, x)
end

@inline function _inner_res_jac_r3!(
        prep::DeflatedPrep, prob::NonlinearProblem, J::AbstractMatrix,
        x::AbstractVector, p)
    if prob.J === nothing
        jacobian!!(J, prep.inner, prob, x, p)
    else
        prob.J(J, x, p)
    end
    residual!!(prep.Fs, prob, x, p)
    return prep.Fs
end

@inline function _inner_res_jac_r3!(
        prep::DeflatedPrep, prob, J::AbstractMatrix, x::AbstractVector, p)
    gradient_hessian!!(prep.Fs, J, prep.inner, prob, x, p)
    return prep.Fs
end

function jacobian!!(
        J::AbstractMatrix, prep::DeflatedPrep, def::Deflated, x::AbstractVector, p)
    Fs = _inner_res_jac_r3!(prep, def.prob, J, x, p)
    M = _deflate_factors!(prep.gs, prep.us, def, x)
    J .= M .* J .+ Fs .* transpose(prep.gs)
    return J
end

@inline function _inner_jvp_r3!(
        Jv, prep::DeflatedPrep, prob::NonlinearProblem, x::AbstractVector,
        v::AbstractVector, p)
    if prob.J === nothing
        return jvp!!(Jv, prep.inner, prob, x, v, p)
    end
    prob.J(prep.Jx, x, p)
    return mul!!(Jv, prep.Jx, v)
end

@inline function _inner_jvp_r3!(
        Jv, prep::DeflatedPrep, prob, x::AbstractVector, v::AbstractVector, p)
    return jvp!!(Jv, prep.inner, prob, x, v, p)
end

function jvp!!(
        Jv::AbstractVector, prep::DeflatedPrep, def::Deflated, x::AbstractVector,
        v::AbstractVector, p)
    F = residual!!(prep.Fs, def.prob, x, p)
    _inner_jvp_r3!(Jv, prep, def.prob, x, v, p)
    M = _deflate_factors!(prep.gs, prep.us, def, x)
    Jv .= M .* Jv .+ rdot(prep.gs, v) .* F
    return Jv
end

# ─── The stopping test on the inner residual ─────────────────────────────────

# The steps and the line search take the deflated residual `M F`, but a solve converges on the inner
# residual: `‖F‖ = ‖M F‖ / M` at the iterate, so `f_abstol` bounds `‖F(x)‖` and `f_reltol` is
# relative to `‖F(x₀)‖`, as in a solve of the inner problem. A non-finite norm is kept, so a start
# at a deflated root stays `NONFINITE`.
@inline function _inner_status(status::SolverStatus{R}, M) where {R}
    isfinite(status.fnorm) || return status
    return SolverStatus{R}(
        status.code, status.iterations, status.fnorm / M, status.stepnorm,
        status.step_failures, status.promoted, status.stationary, status.rank,
        status.nullnorm, status.gradratio)
end

@inline function converged_status(
        state, def::Deflated, x::Union{StaticVector, Number}, status::SolverStatus)
    return _inner_status(status, _deflate_scale(def, x))
end

# An array iterate reads `M` with the difference buffer of the deflated preparation, which is the
# Jacobian preparation of its `Newton` state. The state is not named in the signature, which would
# make a static iterate ambiguous between this method and the one above.
@inline function converged_status(
        state, def::Deflated, x::AbstractVector, status::SolverStatus)
    return _inner_status(status, _deflate_scale!(state.jacprep.us, def, x))
end

# ─── The checks of a solve ───────────────────────────────────────────────────

# The iterate of a solve: real, in the element type of the roots, as long as every root, and a
# number iterate needs number roots while a vector iterate needs vector roots. An array iterate
# needs array roots on its own back end, because the deflation broadcasts `x .- r`.
function _check_deflated_solve(def::Deflated, x₀)
    real_eltype = eltype(x₀)
    real_eltype <: Real || throw(ArgumentError(
        "Deflated needs a real iterate: M(x) is not holomorphic, so the complex Jacobian that " *
        "Newton steps with does not exist; got $(real_eltype)"))
    real_eltype === typeof(def.power) || throw(ArgumentError(
        "Deflated needs an iterate in the element type of its roots: the roots give " *
        "$(typeof(def.power)) and the start is $real_eltype"))
    n = length(x₀)
    scalar = _scalar_iterate(x₀)
    for r in def.roots
        length(r) == n || throw(ArgumentError(
            "Deflated needs roots of the length of the iterate: the start has $n entries and a " *
            "root has $(length(r))"))
        _scalar_iterate(r) === scalar || throw(ArgumentError(
            "Deflated needs roots of the storage of the iterate: the start is a $(typeof(x₀)) " *
            "and a root is a $(typeof(r))"))
    end
    _check_deflated_root_backend(def, x₀)
    return nothing
end

# A number or a static iterate takes its roots by value. An array iterate takes array roots on its
# own back end: a host root in a device broadcast cannot run, and a static root has no back end.
@inline _check_deflated_root_backend(def::Deflated, x₀::Union{
    Number, StaticArray}) = nothing
function _check_deflated_root_backend(def::Deflated, x₀::AbstractArray)
    backend = get_backend(x₀)
    for r in def.roots
        (!(r isa StaticArray) && get_backend(r) == backend) || throw(ArgumentError(
            "Deflated needs array roots on the back end of the iterate: the start is a " *
            "$(typeof(x₀)) and a root is a $(typeof(r)); give the roots in the array type of " *
            "the start"))
    end
    return nothing
end

# A `Newton` takes a callable `jacobian` as the Jacobian of the problem it solves, which here is the
# deflated one, `M J + F ∇Mᵀ`; a Jacobian of the inner `F` there steps in the wrong direction. An
# analytic Jacobian of `F` is the `J` of the wrapped `NonlinearProblem`.
@inline _check_deflated_method(method) = nothing
@inline function _check_deflated_method(m::Newton)
    (m.jacobian === nothing || m.jacobian isa AbstractADType) || throw(ArgumentError(
        "Deflated needs a Newton whose jacobian is nothing or an AD backend: a callable would be " *
        "taken as the deflated Jacobian; give the Jacobian of F to the NonlinearProblem instead"))
    return nothing
end

# ─── `Deflated(::NonlinearProblem)`: the entry points of the solver loop ─────

function solve(
        def::Deflated{<:NonlinearProblem}, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    _check_deflated_method(method)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p))
        return solve_implicit(def, x₀, method, p, options, kwargs)
    end
    _check_deflated_solve(def, x₀)
    return invoke(
        solve, Tuple{AbstractNonlinearProblem, Any, Any, Any}, def, x₀, method, p;
        options = options, kwargs...)
end

function init(
        def::Deflated{<:NonlinearProblem}, x₀, method, p = nothing;
        options::Union{Nothing, Options} = nothing, kwargs...)
    _check_deflated_method(method)
    _check_deflated_solve(def, x₀)
    return invoke(init, Tuple{AbstractNonlinearProblem, Any, Any, Any}, def, x₀, method, p;
        options = options, kwargs...)
end

# `solve` delegates to `solve!!`, but a caller may reach the loop directly with an iterate that
# `init` never saw, so the checks of a solve start are made here as well.
@inline function solve!!(
        state, def::Deflated{<:NonlinearProblem}, x₀, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    _check_deflated_solve(def, x₀)
    return invoke(solve!!,
        Tuple{Any, AbstractNonlinearProblem, Any, Any, Any, Options{R}},
        state, def, x₀, method, p, options)
end

# The derivative of the deflated solve at a dual input is the derivative of the **undeflated**
# problem at the root it found: the root is what the caller reads, and the deflation only steered
# the solve away from the other roots. The tangent therefore reads the inner problem's own residual
# and Jacobian at `x*`, in the deflated state's workspace.
@inline function _inner_jacprep_state(
        state::NewtonState, prob::NonlinearProblem, x_star, p_val)
    prob.J !== nothing && return state
    Jp = prepare_ad(AutoForwardDiff(), prob, state.Fx, x_star, p_val)
    return NewtonState(
        Jp, state.jvpprep, state.Fx, state.d, state.fac, state.J, state.g, state.rule,
        state.merit, state.fnorm, state.diag, state.refbuf, state.fac_T, state.promoted,
        state.slow_steps)
end
@inline function _inner_jacprep_state(state, prob, x_star, p_val)
    return state
end

function solve_implicit(
        def::Deflated{<:NonlinearProblem}, x₀, method, p,
        options::Union{Nothing, Options}, kwargs)
    D_in = validate_dual_types(typeof(x₀), typeof(p))
    x_val = Adapt.adapt(StripDual(), x₀)
    p_val = Adapt.adapt(StripDual(), p)
    if has_dual_type(typeof(x_val)) || has_dual_type(typeof(p_val))
        throw(ArgumentError(
            "Adapt.adapt(StripDual(), ...) left a ForwardDiff.Dual in x₀ or p; mark custom parameter structs with Adapt.@adapt_structure"))
    end
    _check_deflated_solve(def, x_val)
    x, m, opt, state = prepare_solve(def, x_val, method, p_val, options, kwargs)
    x_star, status, state = solve!!(state, def, x, m, p_val, opt)
    D = _working_dual_type(D_in, eltype(x_star))
    status.code === SUCCESS || return _make_nan_duals(D, x_star), status
    inner = def.prob
    return _implicit_tangent(
        D, _inner_jacprep_state(state, inner, x_star, p_val), inner, x_star, m, p, p_val,
        status)
end

# ─── `solve_batch!` of a `Deflated(::NonlinearProblem)` ──────────────────────

# The batch kernel takes the problem by value, so a `Deflated` whose roots or inner problem are not
# `isbits` is refused where the base refuses a Jacobian prototype that is host memory.
function _check_deflated_batch(def::Deflated, X)
    zero_x = zero(eltype(X))
    _check_deflated_solve(def, zero_x)
    isbits(def) || throw(ArgumentError(
        "solve_batch! takes the problem by value into a kernel, so a Deflated must be `isbits`: " *
        "pass a Tuple of static roots, or run `solve(prob, x₀, method, p)` per system"))
    return nothing
end

function _prepare_deflated_batch(X, def, method, options, kwargs)
    _check_deflated_method(method)
    T = eltype(eltype(X))
    R = real(T)
    opt = resolve_options(T, options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    state₀ = init_state(m, def, zero(eltype(X)), nothing)
    return m, opt, state₀
end

"""
    solve_batch!(X, S, def::Deflated, method, p = nothing; order, options, kwargs...)
    solve_batch!(X, def::Deflated, method, p = nothing; order, options, kwargs...)

Solve a batch of independent small systems under one [`Deflated`](@ref) problem, as
[`solve_batch!`](@ref) does on a [`NonlinearProblem`](@ref): every system is deflated at the same
roots, so each ends at a root distinct from them. The batch of a wrapped
[`StationarityProblem`](@ref) is deflated on the unknowns of the same full-state roots, and its
datum comes from the keyword `b`, shared or [`PerSystem`](@ref), or from the problem's own `b`.
The kernel takes the problem by value, so a `Deflated` whose roots or inner problem are not
`isbits` — the `AbstractVector` root storage of an R3 iterate, for one — raises an
`ArgumentError`; a root of another length than the iterate, or an iterate of another element type
than the roots, raises one too.
"""
function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        S::AbstractVector{SolverStatus{R}},
        def::Deflated{<:NonlinearProblem},
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        kwargs...
) where {R <: Real}
    _validate_batch(X, p, order)
    _check_deflated_batch(def, X)
    if R !== real(eltype(eltype(X)))
        throw(ArgumentError(
            "SolverStatus real type $R does not match batch real element type $(real(eltype(eltype(X))))"))
    end
    if length(S) != length(X)
        throw(DimensionMismatch(
            "status vector length $(length(S)) does not match batch length $(length(X))"))
    end
    if get_backend(S) != get_backend(X)
        throw(ArgumentError(
            "status vector backend $(get_backend(S)) does not match batch backend $(get_backend(X))"))
    end
    m, opt, state₀ = _prepare_deflated_batch(X, def, method, options, kwargs)
    isempty(X) && return S
    return _launch!(X, S, def, m, p, opt, state₀, order)
end

function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        def::Deflated{<:NonlinearProblem},
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        kwargs...
)
    _validate_batch(X, p, order)
    _check_deflated_batch(def, X)
    m, opt, state₀ = _prepare_deflated_batch(X, def, method, options, kwargs)
    isempty(X) && return _tally_return_codes(ReturnCode[])
    codes = allocate(get_backend(X), ReturnCode, length(X))
    return _tally_return_codes(Array(_launch!(X, codes, def, m, p, opt, state₀, order)))
end

# ─── `Deflated(::StationarityProblem)`: a solve on the unknowns ──────────────

# A `StationarityProblem` is solved on the `|U|` unknowns `x_U`, and so is its deflation: the
# wrapper builds the reduced problem over the unknowns and one `Deflated` over it, whose roots are
# the `U` entries of the given full states. The solve still starts and returns the full state.

# An R1 stationarity state: the state of the solve over the unknowns. The reduced problem is
# rebuilt from the start of each solve, because it holds that start by value.
struct StaticDeflatedState{ST}
    inner::ST
end

# An R3 stationarity state: the state over the unknowns, the deflated reduced problem with its
# workspace and its mapped roots, and the unknown buffer that the solve reuses.
struct DenseDeflatedState{ST, DP, U}
    inner::ST
    dprob::DP
    u_work::U
end

# The `U` entries of the full-state roots: the unknown part the distance is taken on.
@inline function _deflated_roots_r1(def::Deflated{<:StationarityProblem})
    U = def.prob.U
    return map(r -> _extract_unknowns(U, r), def.roots)
end

@inline function _deflated_roots_r3(def::Deflated{<:StationarityProblem})
    U = def.prob.U
    return map(def.roots) do r
        U isa Colon ? copy(r) : r[U]
    end
end

@inline function _deflated_reduced(
        def::Deflated{<:StationarityProblem}, x₀, b_eff)
    rprob = _ReducedStationarityProblem(def.prob, x₀, b_eff)
    return Deflated(
        rprob, _deflated_roots_r1(def); power = def.power, shift = def.shift)
end

# The index maps of the unknowns `x_U` in the full state, as the base's stationarity state builds
# them.
function _deflated_index_maps(U, x₀, M::Int, m_eq::Int)
    U isa Colon && return nothing, nothing
    u_cpu = Int[Int(i) for i in U]
    pos_cpu = zeros(Int, M)
    for (j, ui) in enumerate(u_cpu)
        pos_cpu[ui] = j
    end
    u_idx = similar(x₀, Int, m_eq)
    copyto!(u_idx, u_cpu)
    u_pos = similar(x₀, Int, M)
    copyto!(u_pos, pos_cpu)
    return u_idx, u_pos
end

# The preparation of a wrapped stationarity problem. The methods of `init_state` are named with
# `Newton` as well as with any method: the base's own preparation is written on
# `m::Newton, prob::AbstractNonlinearProblem`, and one method over every `m` and a `Deflated`
# problem alone would be ambiguous with it.
@inline function _deflated_stat_state_r1(m, def, x₀, p, b)
    b_raw = _resolve_datum(def.prob, p, b)
    _validate_stationarity_solve(def.prob, x₀, b_raw)
    b_eff = _static_datum(def.prob, x₀, b_raw)
    dprob = _deflated_reduced(def, x₀, b_eff)
    u₀ = _extract_unknowns(def.prob.U, x₀)
    return StaticDeflatedState(init_state(m, dprob, u₀, p))
end

function _deflated_stat_state_r3(m, def, x₀, p, b)
    b_raw = _resolve_datum(def.prob, p, b)
    _validate_stationarity_solve(def.prob, x₀, b_raw)
    M = length(x₀)
    m_eq = _num_equations(def.prob.E, M)
    x_work = copy(x₀)
    u_work = similar(x₀, eltype(x₀), m_eq)
    b_buf = similar(x₀, eltype(x₀), m_eq)
    has_b = Ref(false)
    _update_datum_buf!(b_buf, has_b, b_raw)
    u_idx, u_pos = _deflated_index_maps(def.prob.U, x₀, M, m_eq)
    _extract_u!(u_work, x_work, u_idx)
    prep = prepare_ad(_ad_backend_of(m), def.prob, u_work, x_work, p)
    rprob = _ReducedStationarityProblem(def.prob, x_work, b_buf, has_b, u_idx, u_pos, prep)
    dprob = Deflated(
        rprob, _deflated_roots_r3(def); power = def.power, shift = def.shift)
    return DenseDeflatedState(init_state(m, dprob, u_work, p), dprob, u_work)
end

@inline function init_state(
        m, def::Deflated{<:StationarityProblem}, x₀::Union{StaticVector, Number}, p;
        b = nothing)
    return _deflated_stat_state_r1(m, def, x₀, p, b)
end

@inline function init_state(
        m::Newton, def::Deflated{<:StationarityProblem}, x₀::Union{StaticVector, Number}, p;
        b = nothing)
    return _deflated_stat_state_r1(m, def, x₀, p, b)
end

@inline function init_state(
        m, def::Deflated{<:StationarityProblem}, x₀::AbstractVector, p; b = nothing)
    return _deflated_stat_state_r3(m, def, x₀, p, b)
end

@inline function init_state(
        m::Newton, def::Deflated{<:StationarityProblem}, x₀::AbstractVector, p; b = nothing)
    return _deflated_stat_state_r3(m, def, x₀, p, b)
end

@inline function solve!!(
        s::StaticDeflatedState, def::Deflated{<:StationarityProblem}, x₀, m, p,
        opt::Options{R}; b = nothing) where {R <: Real}
    p_sys = p isa _StationaritySysParams ? p.p : p
    b_sys = p isa _StationaritySysParams && b === nothing ? p.b : b
    b_raw = _resolve_datum(def.prob, p_sys, b_sys)
    b_eff = _static_datum(def.prob, x₀, b_raw)
    dprob = _deflated_reduced(def, x₀, b_eff)
    u₀ = _extract_unknowns(def.prob.U, x₀)
    u, status, inner₊ = solve!!(s.inner, dprob, u₀, m, p_sys, opt)
    x = _like_iterate(x₀, _embed_unknowns(def.prob.U, x₀, u))
    return x, status, StaticDeflatedState(inner₊)
end

@inline function solve!!(
        s::StaticDeflatedState, def::Deflated{<:StationarityProblem}, x₀, m,
        p = nothing; b = nothing)
    return solve!!(s, def, x₀, m, p, Options(eltype(x₀)); b = b)
end

function solve!!(
        s::DenseDeflatedState, def::Deflated{<:StationarityProblem}, x₀, method, p,
        options::Options{R}; b = nothing) where {R <: Real}
    b_raw = _resolve_datum(def.prob, p, b)
    rprob = s.dprob.prob
    if b_raw isa AbstractVector
        length(b_raw) == length(s.u_work) || throw(ArgumentError(
            "datum b must have length $(length(s.u_work)), got $(length(b_raw))"))
    end
    copyto!(rprob.x_ref, x₀)
    _extract_u!(s.u_work, rprob.x_ref, rprob.u_idx)
    _update_datum_buf!(rprob.b, rprob.has_b, b_raw)
    u, status, inner₊ = solve!!(s.inner, s.dprob, s.u_work, method, p, options)
    _embed_u!(x₀, rprob.x_ref, u, rprob.u_pos)
    state₊ = inner₊ === s.inner ? s : DenseDeflatedState(inner₊, s.dprob, s.u_work)
    return x₀, status, state₊
end

function solve!!(
        s::DenseDeflatedState, def::Deflated{<:StationarityProblem}, x₀, method,
        p = nothing; b = nothing)
    return solve!!(s, def, x₀, method, p, Options(eltype(x₀)); b = b)
end

# Implicit differentiation of a `StationarityProblem` is not implemented, so a deflated one refuses
# a dual as the problem itself does.
function _refuse_deflated_stationarity_duals(f::Symbol, x₀, p)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p))
        throw(ArgumentError(
            "$f does not accept ForwardDiff.Dual inputs for a Deflated(::StationarityProblem); " *
            "implicit differentiation of a StationarityProblem is not implemented"))
    end
    return nothing
end

function _init_deflated_stationarity(def, x₀, method, p, b, options, kwargs)
    _refuse_deflated_stationarity_duals(:init, x₀, p)
    _check_deflated_method(method)
    _check_deflated_solve(def, x₀)
    R = real(eltype(x₀))
    opt = resolve_options(eltype(x₀), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    state = init_state(m, def, x, p; b = b)
    # the datum that `init` was given is the datum of the solver's later solves, as it is for a
    # `StationarityProblem` itself, so the wrapper stores it on the problem
    stored = b === nothing ? def :
             Deflated(_with_datum(def.prob, b), def.roots; power = def.power, shift = def.shift)
    return NonlinearSolver(stored, x, m, opt, state, p)
end

function _solve_deflated_stationarity(def, x₀, method, p, b, options, kwargs)
    _refuse_deflated_stationarity_duals(:solve, x₀, p)
    _check_deflated_method(method)
    _check_deflated_solve(def, x₀)
    R = real(eltype(x₀))
    opt = resolve_options(eltype(x₀), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    state = init_state(m, def, x, p; b = b)
    x_sol, status, _ = solve!!(state, def, x, m, p, opt; b = b)
    return x_sol, status
end

function init(
        def::Deflated{<:StationarityProblem}, x₀, method, p = nothing;
        b = nothing, options::Union{Nothing, Options} = nothing, kwargs...)
    return _init_deflated_stationarity(def, x₀, method, p, b, options, kwargs)
end

function init(def::Deflated{<:StationarityProblem}, x₀; kwargs...)
    init(def, x₀, Newton(), nothing; kwargs...)
end

function solve(
        def::Deflated{<:StationarityProblem}, x₀, method, p = nothing;
        b = nothing, options::Union{Nothing, Options} = nothing, kwargs...)
    return _solve_deflated_stationarity(def, x₀, method, p, b, options, kwargs)
end

function solve(def::Deflated{<:StationarityProblem}, x₀; kwargs...)
    solve(def, x₀, Newton(), nothing; kwargs...)
end

# `solve!` on a solver of a deflated stationarity problem solves over the unknowns and returns the
# full state, as `solve!` on the problem itself does, and carries the per-solve datum `b`.
function solve!(
        solver::NonlinearSolver{T, P, M, O, S, X, Q}, x, p = solver.p;
        b = nothing) where {T, P <: Deflated{<:StationarityProblem}, M, O, S, X, Q}
    if has_dual_type(typeof(x)) || has_dual_type(typeof(p))
        throw(ArgumentError(
            "solve! does not accept ForwardDiff.Dual inputs; use solve(prob, x₀, method, p) for implicit differentiation"))
    end
    eltype(x) === T && length(x) == solver.n || throw(ArgumentError(
        "`solve!` needs an iterate like the template given to `init`, of length " *
        "$(solver.n) and element type $T; got length $(length(x)) and " *
        "element type $(eltype(x))"))
    x isa X || throw(ArgumentError(
        "`solve!` needs an iterate of type $X, like the one given to `init`; " *
        "got type $(typeof(x))"))
    p isa Q || throw(ArgumentError(
        "`solve!` needs parameters of type $Q, like those given to `init`; " *
        "got type $(typeof(p))"))
    solver.x = x
    solver.p = p
    x₊, status, state = solve!!(
        solver.state, solver.prob, x, solver.method, p, solver.options; b = b)
    solver.x = x₊
    solver.state === state || (solver.state = state)
    return solver.x, status
end

# ─── `solve_batch!` of a `Deflated(::StationarityProblem)` ───────────────────

# The batch steps of the base's stationarity batch, with the wrapped problem in the base problem's
# place: one `Deflated` for the whole batch, every system deflated at the same full-state roots on
# the unknowns, and the per-system datum from the keyword `b` or from the problem's own `b`.
function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        def::Deflated{<:StationarityProblem},
        method = Newton(),
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing, b = nothing, kwargs...
)
    b_src = b === nothing ? def.prob.b : b
    _validate_batch(X, p, order)
    _check_deflated_method(method)
    _check_deflated_batch(def, X)
    b_src isa PerSystem && _validate_batch(X, b_src, nothing)
    T = eltype(eltype(X))
    R = real(T)
    opt = resolve_options(T, options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    _validate_stationarity_batch(X, def.prob, b_src, p)
    isempty(X) && return _tally_return_codes(ReturnCode[])
    idx₀ = firstindex(X)
    p₀ = _sample_batch_item(p, idx₀)
    state₀ = init_state(
        m, def, zero(eltype(X)), p₀; b = _sample_batch_item(b_src, idx₀))
    codes = allocate(get_backend(X), ReturnCode, length(X))
    sp = _StationarityBatchParams(p, b_src)
    return _tally_return_codes(Array(_launch!(X, codes, def, m, sp, opt, state₀, order)))
end

function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        S::AbstractVector{SolverStatus{R}},
        def::Deflated{<:StationarityProblem},
        method = Newton(),
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing, b = nothing,
        kwargs...) where {R <: Real}
    b_src = b === nothing ? def.prob.b : b
    _validate_batch(X, p, order)
    _check_deflated_method(method)
    _check_deflated_batch(def, X)
    b_src isa PerSystem && _validate_batch(X, b_src, nothing)
    R_X = real(eltype(eltype(X)))
    R === R_X || throw(ArgumentError(
        "solve_batch! requires the status vector S to use the real type $R_X of the batch iterates, got SolverStatus{$R}"))
    length(S) == length(X) || throw(DimensionMismatch(
        "status vector S has length $(length(S)), expected $(length(X)) to match the batch X"))
    get_backend(S) == get_backend(X) || throw(ArgumentError(
        "status vector S must reside on the same backend as the batch X ($(get_backend(X))), got $(get_backend(S))"))
    T = eltype(eltype(X))
    opt = resolve_options(T, options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    _validate_stationarity_batch(X, def.prob, b_src, p)
    isempty(X) && return S
    idx₀ = firstindex(X)
    p₀ = _sample_batch_item(p, idx₀)
    state₀ = init_state(
        m, def, zero(eltype(X)), p₀; b = _sample_batch_item(b_src, idx₀))
    sp = _StationarityBatchParams(p, b_src)
    return _launch!(X, S, def, m, sp, opt, state₀, order)
end
