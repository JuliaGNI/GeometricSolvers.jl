# R5 matrix-free Newton–Krylov: a right-preconditioned GMRES with Eisenstat and Walker's forcing
# terms, on the array backends of R3 and on devices.
#
# `GMRES` is a linear-solver method without a factor: the Krylov basis is its storage, and the
# system matrix enters only through a Jacobian–vector product. A `Newton` with it therefore holds
# no `n × n` Jacobian, and its step solves `J d = -F` to the relative tolerance `η_k` that the
# forcing term of the step returns.
#
# The Arnoldi loop orthogonalises by classical Gram–Schmidt with a second pass (CGS2), two
# reductions per inner iteration, and with `GMRES(; mgs = true)` by the modified Gram–Schmidt
# process of Saad (2003), §6.5.2, `j + 1` reductions at iteration `j`. The Hessenberg matrix, its
# Givens rotations and the least-squares right-hand side are on the host, and every vector of
# length `n` is in the storage of the iterate: the loop allocates nothing after
# `init_gmres_workspace`, and it indexes no entry of a vector, so the same code runs under
# `allowscalar(false)` on a `JLArray` and on an `MtlArray`.

"""
    AbstractForcingTerm

The supertype of the forcing terms of [`GMRES`](@ref): the rule that chooses the relative
tolerance ``\\eta_k`` of the linear solve of Newton step `k`. Each term is `isbits` and its
constants are in the real type `R` of the iterate, and [`compute_forcing`](@ref) is the one
function that evaluates it.
"""
abstract type AbstractForcingTerm end

"""
    EisenstatWalkerChoice2(T = Float64; η₀ = 0.5, γ = 0.9, α = 2, η_max = 0.9)

Eisenstat and Walker's Choice 2 forcing term (Eisenstat and Walker 1996, (2.6), pp. 20–24) with
the safeguard of that choice and the floor ``2\\varepsilon(R)``:

```math
\\eta_k^{\\mathrm{raw}} = \\gamma \\Bigl(\\frac{\\|F_k\\|}{\\|F_{k-1}\\|}\\Bigr)^{\\alpha},
\\qquad
\\eta_k = \\max\\Bigl(\\min\\bigl(\\eta_k^{\\mathrm{safe}}, \\eta_{\\max}\\bigr), 2\\varepsilon(R)\\Bigr),
```

with ``\\eta_k^{\\mathrm{safe}} = \\max(\\eta_k^{\\mathrm{raw}}, \\gamma\\eta_{k-1}^{\\alpha})`` where
``\\gamma\\eta_{k-1}^{\\alpha} > 0.1`` and ``\\eta_k^{\\mathrm{raw}}`` elsewhere. The first step
returns ``\\eta_0``, floored at ``2\\varepsilon(R)`` like every other value of the rule. The
defaults are ``\\gamma = 0.9``, ``\\alpha = 2``,
``\\eta_0 = 0.5`` and ``\\eta_{\\max} = 0.9``; ``\\eta_{\\max}`` is 0.8 in Chen et al. (2011) and
0.2 in Chen et al. (2014).

`T` is the real type of the constants. A term with `0 ≤ η₀ < 1`, `0 < γ ≤ 1`, `α ≥ 1` and
`0 < η_max < 1` is built, and any other raises an `ArgumentError`.
"""
struct EisenstatWalkerChoice2{R <: Real} <: AbstractForcingTerm
    η₀::R
    γ::R
    α::R
    η_max::R
    # `adapt` rebuilds a forcing term through this constructor, inside a device kernel too, so the
    # messages are constant.
    function EisenstatWalkerChoice2{R}(
            η₀::Real, γ::Real, α::Real, η_max::Real) where {R <: Real}
        a = R(η₀)
        g = R(γ)
        e = R(α)
        m = R(η_max)
        (isfinite(a) && zero(R) <= a < one(R)) ||
            throw(ArgumentError("EisenstatWalkerChoice2 needs 0 ≤ η₀ < 1"))
        (isfinite(g) && zero(R) < g <= one(R)) ||
            throw(ArgumentError("EisenstatWalkerChoice2 needs 0 < γ ≤ 1"))
        (isfinite(e) && e >= one(R)) ||
            throw(ArgumentError("EisenstatWalkerChoice2 needs α ≥ 1"))
        (isfinite(m) && zero(R) < m < one(R)) ||
            throw(ArgumentError("EisenstatWalkerChoice2 needs 0 < η_max < 1"))
        return new{R}(a, g, e, m)
    end
end

function EisenstatWalkerChoice2(::Type{T} = Float64; η₀::Real = 0.5, γ::Real = 0.9,
        α::Real = 2, η_max::Real = 0.9) where {T <: Real}
    R = float(real(T))
    return EisenstatWalkerChoice2{R}(η₀, γ, α, η_max)
end

# The positional form, which `adapt` rebuilds the term through: the real type is that of the
# constants, which `ToReal` has already converted.
function EisenstatWalkerChoice2(η₀::Real, γ::Real, α::Real, η_max::Real)
    R = float(promote_type(typeof(η₀), typeof(γ), typeof(α), typeof(η_max)))
    return EisenstatWalkerChoice2{R}(η₀, γ, α, η_max)
end

Adapt.@adapt_structure EisenstatWalkerChoice2

"""
    EisenstatWalkerChoice1(T = Float64; η₀ = 0.5, η_max = 0.9)

Eisenstat and Walker's Choice 1 forcing term (Eisenstat and Walker 1996, §2.2), which reads the
linear residual of the last step `s_{k-1}`. After a line search on an unrestarted inner solve,
`s_{k-1}` is the step that the search accepted, whose linear residual the Arnoldi relation gives
without a product; after a restarted inner solve or a [`DogLeg`](@ref) step, it is the direction
GMRES returned:

```math
\\eta_k^{\\mathrm{raw}} =
  \\frac{\\bigl|\\,\\|F_k\\| - \\|F_{k-1} + J_{k-1} s_{k-1}\\|\\,\\bigr|}{\\|F_{k-1}\\|},
\\qquad
\\eta_k = \\max\\Bigl(\\min\\bigl(\\eta_k^{\\mathrm{safe}}, \\eta_{\\max}\\bigr), 2\\varepsilon(R)\\Bigr),
```

with ``\\eta_k^{\\mathrm{safe}} = \\max(\\eta_k^{\\mathrm{raw}}, \\eta_{k-1}^{\\phi})`` where
``\\eta_{k-1}^{\\phi} > 0.1`` and ``\\phi = (1 + \\sqrt{5})/2``, and ``\\eta_k^{\\mathrm{raw}}``
elsewhere. The first step returns ``\\eta_0``, floored at ``2\\varepsilon(R)``.

`T` is the real type of the constants, and the constructor validates `0 ≤ η₀ < 1` and
`0 < η_max < 1` as [`EisenstatWalkerChoice2`](@ref) does.
"""
struct EisenstatWalkerChoice1{R <: Real} <: AbstractForcingTerm
    η₀::R
    η_max::R
    function EisenstatWalkerChoice1{R}(η₀::Real, η_max::Real) where {R <: Real}
        a = R(η₀)
        m = R(η_max)
        (isfinite(a) && zero(R) <= a < one(R)) ||
            throw(ArgumentError("EisenstatWalkerChoice1 needs 0 ≤ η₀ < 1"))
        (isfinite(m) && zero(R) < m < one(R)) ||
            throw(ArgumentError("EisenstatWalkerChoice1 needs 0 < η_max < 1"))
        return new{R}(a, m)
    end
end

function EisenstatWalkerChoice1(::Type{T} = Float64; η₀::Real = 0.5,
        η_max::Real = 0.9) where {T <: Real}
    R = float(real(T))
    return EisenstatWalkerChoice1{R}(η₀, η_max)
end

function EisenstatWalkerChoice1(η₀::Real, η_max::Real)
    R = float(promote_type(typeof(η₀), typeof(η_max)))
    return EisenstatWalkerChoice1{R}(η₀, η_max)
end

Adapt.@adapt_structure EisenstatWalkerChoice1

"""
    FixedForcing(η)

The constant forcing term ``\\eta_k = \\max(\\eta, 2\\varepsilon(R))``: every Newton step solves its
linear system to the same relative tolerance, which is the choice of an inexact Newton method
without an adaptive rule (Eisenstat and Walker 1996, §1). `η` must be finite and in `[0, 1)`.
"""
struct FixedForcing{R <: Real} <: AbstractForcingTerm
    η::R
    function FixedForcing{R}(η::Real) where {R <: Real}
        e = R(η)
        (isfinite(e) && zero(R) <= e < one(R)) ||
            throw(ArgumentError("FixedForcing η must be finite and in [0, 1)"))
        return new{R}(e)
    end
end

FixedForcing(η::Real) = FixedForcing{float(typeof(η))}(η)

Adapt.@adapt_structure FixedForcing

"""
    compute_forcing(forcing, k, fnorm, fnorm_prev, η_prev, linres_prev) -> η

The relative tolerance ``\\eta_k`` of the linear solve of Newton step `k`, in the real type of the
forcing term's constants.

`k` is the zero-based index of the step, so that `k = 0` returns the initial tolerance
``\\eta_0`` of the term. `fnorm` is ``\\|F_k\\|``, `fnorm_prev` is ``\\|F_{k-1}\\|``, `η_prev` is
``\\eta_{k-1}``, and `linres_prev` is the linear residual ``\\|F_{k-1} + J_{k-1} s_{k-1}\\|`` of the
last step: the step that a line search accepted on an unrestarted inner solve, and the direction
GMRES returned after a restarted one or a [`DogLeg`](@ref) step. Only
[`EisenstatWalkerChoice1`](@ref) reads it. A zero, non-finite or `NaN`
`fnorm_prev` gives ``\\eta_{\\max}``: the ratio of the rule is then undefined, and a residual that
vanished ends the solve before the next step reads its tolerance, so the value is a guard and
never a step's choice. A non-finite ratio, which a non-finite residual gives, gives ``\\eta_{\\max}``
as well.
"""
function compute_forcing end

@inline function compute_forcing(f::FixedForcing{R}, k, fnorm, fnorm_prev, η_prev,
        linres_prev) where {R}
    return max(f.η, 2 * eps(R))
end

@inline function compute_forcing(
        f::EisenstatWalkerChoice2{R}, k, fnorm, fnorm_prev, η_prev,
        linres_prev) where {R}
    k > 0 || return max(f.η₀, 2 * eps(R))
    (isfinite(fnorm_prev) && !iszero(fnorm_prev) && isfinite(fnorm)) || return f.η_max
    raw = f.γ * (R(fnorm) / R(fnorm_prev))^f.α
    safeguard = f.γ * R(η_prev)^f.α
    safe = safeguard > R(0.1) ? max(raw, safeguard) : raw
    return clamp_forcing(R, max(min(safe, f.η_max), 2 * eps(R)), f.η_max)
end

@inline function compute_forcing(
        f::EisenstatWalkerChoice1{R}, k, fnorm, fnorm_prev, η_prev,
        linres_prev) where {R}
    k > 0 || return max(f.η₀, 2 * eps(R))
    (isfinite(fnorm_prev) && !iszero(fnorm_prev) && isfinite(fnorm)) || return f.η_max
    raw = abs(R(fnorm) - R(linres_prev)) / R(fnorm_prev)
    safeguard = R(η_prev)^R((1 + sqrt(5)) / 2)
    safe = safeguard > R(0.1) ? max(raw, safeguard) : raw
    return clamp_forcing(R, max(min(safe, f.η_max), 2 * eps(R)), f.η_max)
end

# The tolerance a rule returns: its value where it is finite and in `(0, 1)`, `η_max` where a
# non-finite ratio reached the cap, and nothing above `1 - eps`, which would make an inexact step
# promise no decrease.
@inline function clamp_forcing(::Type{R}, η, η_max) where {R}
    isfinite(η) || return η_max
    return min(max(η, 2 * eps(R)), one(R) - eps(R))
end

"""
    GMRES(; maxiter = 30, restart = nothing, forcing = EisenstatWalkerChoice2(),
            preconditioner = nothing, jvp_eltype = Nothing, mgs = false,
            basis_eltype = Nothing, flexible = false)

The R5 matrix-free Newton–Krylov linear solver of [`Newton`](@ref): right-preconditioned GMRES
on the Jacobian–vector product of the problem, with the forcing term `forcing`
([`EisenstatWalkerChoice2`](@ref) by default,
[`EisenstatWalkerChoice1`](@ref) or [`FixedForcing`](@ref)), a right preconditioner
`preconditioner`, the element type `jvp_eltype` of the product, the orthogonalisation of the
Arnoldi process that `mgs` selects, the storage type `basis_eltype` of the Krylov basis, and the
flexible variant that `flexible` selects (Saad 1993, SIAM J. Sci. Comput. 14(2), 461–469,
doi:10.1137/0914028).

# Fields

- `maxiter::Int32`: the cap on the inner GMRES iterations of one Newton step, `≥ 1`. Under a
  [`DogLeg`](@ref) the cap is `min(maxiter, n)`, one Arnoldi cycle, because its model reads the
  Arnoldi relation of that cycle. A step that reaches the cap without meeting the tolerance of its
  forcing term counts a step-rule failure.
- `restart::Int32`: the number of inner iterations between restarts, `1 ≤ restart ≤ maxiter`.
  `restart = nothing` stores `maxiter`, which is unrestarted GMRES, and which [`DogLeg`](@ref)
  needs. The Krylov basis holds `min(restart, n) + 1` columns.
- `forcing::F`: the rule that sets the relative tolerance ``\\eta_k`` of step `k`.
- `preconditioner::P`: the right preconditioner, `nothing` for none. `refresh!(P, prob, x, Fx, p)`
  is called once per Newton step, before GMRES starts, or once per `k` steps under
  `Newton(; reuse = k)`, and `apply_preconditioner!(z, P, v)` once per inner iteration; the solve
  forms ``J M^{-1} u = -F`` and returns ``d = M^{-1} u``. A type
  that carries a preconditioner extends those two functions, for example a stored factorisation
  that `ldiv!` applies, which is the frozen-Jacobian preconditioner of a Newton–Krylov method.
- `jvp_eltype::Type{TJ}`: `Nothing` evaluates the Jacobian–vector product in the working type of
  the iterate; a real type `TJ` evaluates it in `TJ` and promotes the product to the working type,
  which is a defect-correction solve: a product in a lower precision, a residual in the working
  one. The iterate, the residual and every norm stay in the working type.
- `mgs::Bool`: `false`, the default, orthogonalises each new Krylov vector by classical
  Gram–Schmidt with a second pass (CGS2), which takes two reductions over the basis per inner
  iteration: ``h_1 = V_j^H w`` and ``w_1 = w - V_j h_1``, then ``[V_j\\ w_1]^H w_1`` for the
  correction ``h_2`` and for ``s_1 = \\|w_1\\|^2``, and ``w_2 = w_1 - V_j h_2``. The norm of
  ``w_2`` is ``\\sqrt{s_1 - \\|h_2\\|^2}`` where ``s_1`` is finite, at least `n * floatmin` and
  ``\\|h_2\\|^2 \\le s_1 / 2``, and one more reduction, `rnorm` of ``w_2``, elsewhere. `true`
  orthogonalises by modified Gram–Schmidt, which takes ``j + 1`` reductions at inner iteration
  `j`, and ``2j + 2`` on a basis stored in a type other than `T` (`basis_eltype` below).
- `basis_eltype::Type{TB}`: `Nothing` stores the Krylov basis in the working type `T` of the
  iterate; a concrete `AbstractFloat` type `TB` stores it in `TB`, and a complex iterate in
  `Complex{TB}`, as the compressed basis of Aliaga et al. (2020, arXiv:2009.12101) does. Every
  product, coefficient, norm and update stays in `T`, and the preconditioner reads a copy of each
  basis vector in `T`. A basis in a type other than `T` is not orthonormal in `T`, so the Givens
  estimate of the residual is not the true one: each cycle ends on the true residual
  ``\\|b - J d\\|``, one product, and restarts where it misses the tolerance. With room for
  restarts the solve reaches the accuracy of `T`; one cycle of `j` CGS2 iterations reaches that
  of the cycle in `T` within the bound of Lindquist, Luszczek and Dongarra (2022, Theorem 1,
  doi:10.1109/TPDS.2021.3090757) for the unit roundoff of `TB`. A stored column has unit norm
  only to the round-off of `TB`, so one projection leaves a remainder of that order along the
  basis: the CGS2 step takes three reductions here, because column `j + 1` of the basis cannot
  hold ``w_1`` in `T`, and the MGS step takes a second pass. Each step takes one more reduction,
  the norm of ``w = J M^{-1} v_j`` before the projections, for a breakdown test: where the cycle
  would go on, its Givens estimate above the tolerance and neither `restart` nor `maxiter` reached,
  a remainder ``h_{j+1,j} < 2 \\cdot 10^{-3}\\, \\mathrm{eps}(TB) \\|w\\|`` is a breakdown, and
  the cycle ends on its true residual. That remainder is what the projections
  leave on a Krylov space that is invariant, as that of `J = c I` and a constant `b`, where each
  stored column is parallel to its vector and has unit norm only to the round-off of `TB`;
  without the test the step would take it for a new basis vector parallel to the last one. An
  invariant space whose stored columns also carry a direction error of order ``\\mathrm{eps}(TB)``
  leaves a remainder of that order, which the test does not see: the cycle goes on with it as a
  new direction. Any other
  `basis_eltype` raises an `ArgumentError`, and so does a [`Newton`](@ref) with a
  [`DogLeg`](@ref) globalization and a `basis_eltype` that is not `Nothing`.
- `flexible::Val{FL}`: `Val(false)`, from `flexible = false`, forms the direction
  ``d = M^{-1} u`` of the right-preconditioned solve. `Val(true)` keeps each preconditioned
  vector ``z_j = M_j^{-1} v_j`` in a matrix `Z` of the working type and forms
  ``d = Z_k y``, and the restart residual from it, so that a preconditioner that changes between
  inner iterations still gives the minimal-residual direction of its Krylov space (Saad 1993).
  Without a preconditioner `Z` is not allocated, and the solve is that of `flexible = false`.

A `Newton` with `GMRES` allocates no `n × n` Jacobian, and its iterate must be an
`AbstractVector`: a `Number` and a `StaticArray` have no Krylov basis here, and a
[`LeastSquaresProblem`](@ref) has no square system. The solve calls no Jacobian function of the
problem: the product comes from the `jacobian` of the `Newton`, a [`JVPFunction`](@ref) or an AD
back end, `AutoForwardDiff()` by default. A residual that the back end cannot differentiate
therefore needs a `JVPFunction`, even where the problem carries a Jacobian. For the same reason
[`solve`](@ref) refuses a `ForwardDiff.Dual` in `x₀` or `p` for a `Newton` with `GMRES` with an
`ArgumentError`, before the solve: implicit differentiation factorises the Jacobian at the
root. A constructor refuses
`maxiter < 1` and a
`restart` outside `1:maxiter` with an `ArgumentError`, and so does a [`Newton`](@ref) with a
[`DogLeg`](@ref) globalization and a preconditioner or a `restart < maxiter`; a kernel cannot
format a message, so each message is constant.
"""
struct GMRES{F <: AbstractForcingTerm, P, TJ, TB, FL} <: LinearMethod{Nothing}
    maxiter::Int32
    restart::Int32
    forcing::F
    preconditioner::P
    jvp_eltype::Type{TJ}
    mgs::Bool
    basis_eltype::Type{TB}
    flexible::Val{FL}
    # `adapt` rebuilds the method through this constructor, inside a device kernel too.
    function GMRES{F, P, TJ, TB, FL}(maxiter::Int32, restart::Int32, forcing::F,
            preconditioner::P, jvp_eltype::Type{TJ}, mgs::Bool, basis_eltype::Type{TB},
            flexible::Val{FL}) where {F <: AbstractForcingTerm, P, TJ, TB, FL}
        maxiter >= Int32(1) || throw(ArgumentError("GMRES maxiter must be ≥ 1"))
        (Int32(1) <= restart <= maxiter) ||
            throw(ArgumentError("GMRES needs 1 ≤ restart ≤ maxiter"))
        (TB === Nothing || (TB <: AbstractFloat && isconcretetype(TB))) ||
            throw(ArgumentError("GMRES basis_eltype must be Nothing or a concrete AbstractFloat type"))
        return new{F, P, TJ, TB, FL}(maxiter, restart, forcing, preconditioner, jvp_eltype,
            mgs, basis_eltype, flexible)
    end
end

function GMRES(; maxiter::Integer = 30, restart::Union{Nothing, Integer} = nothing,
        forcing::AbstractForcingTerm = EisenstatWalkerChoice2(),
        preconditioner = nothing, jvp_eltype::Type = Nothing, mgs::Bool = false,
        basis_eltype::Type = Nothing, flexible::Bool = false)
    m = to_int32(maxiter, "GMRES maxiter must fit in an Int32")
    r = restart === nothing ? m :
        to_int32(restart, "GMRES restart must fit in an Int32")
    return GMRES(
        m, r, forcing, preconditioner, jvp_eltype, mgs, basis_eltype, Val(flexible))
end

# An `Integer` constructor argument as an `Int32`, with a constant message where it does not fit: a
# kernel cannot format one.
@inline function to_int32(n::Integer, message::String)
    (typemin(Int32) <= n <= typemax(Int32)) || throw(ArgumentError(message))
    return Int32(n)
end

# The positional form, which `adapt` rebuilds the method through: the element types of the product
# and of the basis are types and the flexible flag a `Val`, so each is a type parameter too.
@inline function GMRES(maxiter::Int32, restart::Int32, forcing::AbstractForcingTerm,
        preconditioner, jvp_eltype::Type, mgs::Bool, basis_eltype::Type, flexible::Val{FL}) where {FL}
    return GMRES{typeof(forcing), typeof(preconditioner), jvp_eltype, basis_eltype, FL}(
        maxiter, restart, forcing, preconditioner, jvp_eltype, mgs, basis_eltype, flexible)
end

Adapt.@adapt_structure GMRES

# Whether the solve of `g` keeps the preconditioned vectors `Z`: a flexible `GMRES` with a
# preconditioner. A flexible `GMRES` without one keeps none, because `M⁻¹ v = v` for each `v`.
@inline keeps_z(::GMRES{F, P, TJ, TB, FL}) where {F, P, TJ, TB, FL} = FL && P !== Nothing

# The element type of the Krylov basis of `g` on an iterate of element type `T`: `T` where
# `basis_eltype` is `Nothing`, and `TB`, or `Complex{TB}` for a complex `T`, elsewhere.
@inline basis_type(::GMRES{F, P, TJ, Nothing}, ::Type{T}) where {F, P, TJ, T} = T
@inline function basis_type(::GMRES{F, P, TJ, TB}, ::Type{T}) where {F, P, TJ, TB, T}
    return T <: Complex ? Complex{TB} : TB
end

"""
    refresh!(P, prob, x, Fx, p)

Refresh the right preconditioner `P` of [`GMRES`](@ref) at the start of a Newton step, from the
iterate `x`, its residual `Fx` and the parameters `p`. [`Newton`](@ref) calls it once per step,
before the inner solve, so that the preconditioner of step `k` sees the data of the last residual
evaluation (Chen et al. 2014, pp. 558–560). Under `Newton(; reuse = k)` it calls it at the first
step, `k` steps after each call, and early at the step after one that stalled or whose inner
solve reached its cap. The method for `nothing` does nothing; a
preconditioner that holds data extends this function, and one that is constant across the solve
extends it with a method that does nothing.
"""
@inline refresh!(::Nothing, prob, x, Fx, p) = nothing

"""
    apply_preconditioner!(z, P, v)

The right-preconditioned vector `z = M⁻¹ v` of [`GMRES`](@ref), in the storage of `z` and return
it, for the preconditioner `P` and the vector `v`. The method for `nothing` copies `v`, so a GMRES
without a preconditioner solves `J d = -F` itself; any other type extends this function.
"""
@inline apply_preconditioner!(z, ::Nothing, v) = copy!!(z, v)

"""
    GMRESWorkspace{T, R}

The workspace of one R5 solve: the Krylov basis `V` of `min(restart, n) + 1` columns in the
storage of the iterate, in the type that `basis_eltype` sets, the preconditioned vectors `Z` of
`min(restart, n)` columns in the working type for a flexible `GMRES` with a preconditioner
(`nothing` otherwise), the host Hessenberg matrix `H` after the Givens rotations and `H0` before
them, the rotation tables `cs` and `sn`, the right-hand side `g` of the least-squares problem, the
solution weights `y`, the weights `cg`, `yN` and `yb` of the matrix-free [`DogLeg`](@ref) and its
reduced work vector `sm`, the vector slots `rr`, `z`, `w`, `u`, `d` and `bb` of length `n`, the
`jvp_eltype` slots `xT`, `vT` and `wT`, the vector `hd` of `min(restart, n) + 1` entries in the
storage of the iterate, which holds the coefficients ``V_j^H w`` of an Arnoldi step and the weights
`y` of a product ``V_j y``, its host copy `hh`, the flag `mgs` of the path of the last solve,
which `reduced_direction!` follows, and the counters of the solve.

`f_calls` and `jvp_calls` count the residual evaluations and the Jacobian–vector products of the
solve in `Int32`, `gmres_iterations` is the inner count of the last Newton step, `max_inner` the
largest inner count of the solve, `arnoldi` the number of Arnoldi steps of the last inner solve,
which the matrix-free `DogLeg` reads, and `step` the zero-based index of the current Newton step,
which the forcing term reads. `last_eta`, `prev_fnorm`, `linres` and `beta` are the ``\\eta_k``,
``\\|F_{k-1}\\|``, linear residual and ``\\beta_0`` of the last step, `linres` being that of the
step a line search accepted on an unrestarted inner solve and that of the direction otherwise;
`start!!` sets them for the solve and `finish!!` returns them to
`NaN`, so they are `NaN` on a state that no `start!!` saw and on the state that `solve!!` returned,
and a step there fails closed. The counters are mutable, so that a step reports them without
rebuilding the workspace.
"""
mutable struct GMRESWorkspace{T, R <: Real, V, M, SR, ST, NV, D, HD, ZM}
    V::V
    Z::ZM
    H::M
    H0::M
    cs::SR
    sn::ST
    g::ST
    y::ST
    cg::ST
    yN::ST
    yb::ST
    sm::ST
    rr::NV
    z::NV
    w::NV
    u::NV
    d::NV
    bb::NV
    xT::D
    vT::D
    wT::D
    hd::HD
    hh::ST
    mgs::Bool
    f_calls::Int32
    jvp_calls::Int32
    gmres_iterations::Int32
    max_inner::Int32
    arnoldi::Int32
    step::Int32
    last_eta::R
    prev_fnorm::R
    linres::R
    beta::R
end

"""
    init_gmres_workspace(g::GMRES, x::AbstractVector)

Allocate the [`GMRESWorkspace`](@ref) of a solve of `g` on an iterate like `x`: the Krylov basis
holds `min(g.restart, length(x)) + 1` columns of the basis type of `g`, and the host matrices `H`
and `H0`, the host vectors and the vector `hd` in the storage of `x` are of that width. The slots
of `jvp_eltype` are `nothing` unless `g.jvp_eltype` names a type, and `Z` is `nothing` unless `g`
is flexible and has a preconditioner.
"""
function init_gmres_workspace(g::GMRES{F, P, TJ}, x::AbstractVector{T}) where {F, P, TJ, T}
    R = real(T)
    n = length(x)
    mr = min(Int(g.restart), n)
    w = mr + 1
    V = buffer(x, basis_type(g, T), n, w)
    Z = keeps_z(g) ? buffer(x, T, n, mr) : nothing
    H = zeros(T, w, mr)
    H0 = zeros(T, w, mr)
    cs = zeros(R, mr)
    sn = zeros(T, mr)
    gb = zeros(T, w)
    y = zeros(T, mr)
    cg = zeros(T, mr)
    yN = zeros(T, mr)
    yb = zeros(T, mr)
    sm = zeros(T, w)
    rr = buffer(x)
    z = buffer(x)
    wv = buffer(x)
    u = buffer(x)
    d = buffer(x)
    bb = buffer(x)
    xT = TJ === Nothing ? nothing : buffer(x, TJ)
    vT = TJ === Nothing ? nothing : buffer(x, TJ)
    wT = TJ === Nothing ? nothing : buffer(x, TJ)
    hd = buffer(x, T, w)
    hh = zeros(T, w)
    return GMRESWorkspace{T, R, typeof(V), typeof(H), typeof(cs), typeof(sn),
        typeof(rr), typeof(xT), typeof(hd), typeof(Z)}(
        V, Z, H, H0, cs, sn, gb, y, cg, yN, yb, sm, rr,
        z, wv, u, d, bb, xT, vT, wT, hd, hh, false,
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
        R(NaN), R(NaN), R(NaN), R(NaN))
end

"""
    GMRESInfo

What one [`gmres_solve!`](@ref) reports: the inner iterations it used, its residual norm
``\\|b - J d\\|_2``, the ``\\beta_0 = \\|b\\|_2`` it was given, whether it met the tolerance, and
whether the Krylov space became invariant before the cap, a happy breakdown.
"""
struct GMRESInfo{R <: Real}
    iterations::Int32
    resnorm::R
    beta::R
    converged::Bool
    breakdown::Bool
end

"""
    gmres_solve!(ws::GMRESWorkspace, jvp_op!, P, b, bnorm, tol, maxiter, restart;
                 mgs = false) -> (d, GMRESInfo)

Solve `J d = b` with right-preconditioned GMRES in the workspace `ws`, and return the direction
`d = M⁻¹ u` in the slot `ws.d` and a [`GMRESInfo`](@ref). `jvp_op!(Jv, v)` writes the product
`J v` into `Jv`; `P` is the right preconditioner; `b` is the right-hand side and `bnorm` its norm
``\\|b\\|_2``, which the caller already holds; `tol` is the relative tolerance, so that the solve
stops at ``\\|b - J d\\|_2 \\le tol \\cdot bnorm``; `maxiter` caps the inner iterations and
`restart` the iterations between restarts, neither of which may exceed the width the workspace was
allocated for.

The Arnoldi process orthogonalises by classical Gram–Schmidt with a second pass (CGS2), as the
`mgs` field of [`GMRES`](@ref) describes, and by modified Gram–Schmidt with `mgs = true`; both
take the conjugated inner product, so a complex iterate is supported and every norm is in
`real(T)`. The solution update ``u \\mathrel{+}= V_j y`` of a cycle is one matrix–vector product
on the CGS2 path and one update per column of ``V_j`` on the MGS path, which `ws.mgs` records for
`reduced_direction!`. A zero `H[j+1, j]` is a breakdown: the Krylov space is invariant, and the
solve stops there rather than dividing by zero. On a basis stored in a type other than that of
`b`, a remainder at the round-off of the stored basis is a breakdown too, where the cycle would
go on, as the `basis_eltype` field of [`GMRES`](@ref) states; the rotation of the step then
takes `H[j+1, j]` as zero. The residual of
the reduced problem is then zero, unless `J` is singular on that space: the rotated diagonal entry
is then zero too, the residual is that of the last entry of the right-hand side, and the solve
reports no convergence. A restarted cycle forms its new residual with one more product,
`b - J M⁻¹ u`, so a restart costs one Jacobian–vector product. On a workspace with `Z`, the
flexible path, the solution update is ``u \\mathrel{+}= Z_k y``, the direction is `u` itself and the
restart residual is `b - J u`. On a basis stored in a type other than that of `b`, every cycle
ends on that product, and the solve converges where the true residual meets the tolerance; a
cycle that misses it restarts, unless it ended at `maxiter` or on a breakdown whose reduced
residual misses the tolerance too, and the reported residual is the true one. The least-squares
problem is solved from the Givens-rotated Hessenberg matrix, whose diagonal entry is `0` only
where the reduced system is singular, and the corresponding weight is then `0`.
"""
function gmres_solve!(ws::GMRESWorkspace{T, R}, jvp_op!, P, b, bnorm, tol, maxiter,
        restart; mgs::Bool = false) where {T, R}
    n = length(b)
    mr = min(min(Int(restart), n), size(ws.V, 2) - 1)
    atol = R(tol) * R(bnorm)
    ws.mgs = mgs
    fill!(ws.u, zero(T))
    copy!!(ws.rr, b)
    used = Int32(0)
    arnoldi = Int32(0)
    resnorm = R(bnorm)
    converged = resnorm <= atol
    breakdown = false
    while !converged && used < maxiter
        β = R(rnorm(ws.rr))
        if β <= atol
            resnorm = β
            converged = true
            break
        end
        @views ws.V[:, 1] .= ws.rr ./ β
        fill!(ws.g, zero(T))
        ws.g[1] = β
        j = 0
        hj = β
        linres = β
        while j < mr && used < maxiter
            j += 1
            used += one(Int32)
            precondition_column!(ws, P, j)
            jvp_op!(ws.w, ws.z)
            # the norm of `w` before the projections, which the breakdown test of a reduced basis
            # reads
            w₀ = reduced_basis(ws) ? R(rnorm(ws.w)) : zero(R)
            if mgs
                # modified Gram–Schmidt: one pass of projections, in the order of the basis
                for i in 1:j
                    h = arnoldi_dot(@view(ws.V[:, i]), ws.w)
                    ws.H[i, j] = h
                    @views ws.w .-= h .* ws.V[:, i]
                end
                # a basis stored in another type is orthonormal only to its own round-off, so one
                # pass leaves a remainder of that order along the basis, which a second removes
                if reduced_basis(ws)
                    for i in 1:j
                        h = arnoldi_dot(@view(ws.V[:, i]), ws.w)
                        ws.H[i, j] += h
                        @views ws.w .-= h .* ws.V[:, i]
                    end
                end
                hj = R(rnorm(ws.w))
            else
                hj = cgs2_step!(ws, j)
            end
            ws.H[j + 1, j] = hj
            @views ws.H0[:, j] .= ws.H[:, j]
            # the previous rotations on column j, then the rotation that removes H[j+1, j] from it
            for i in 1:(j - 1)
                t = ws.cs[i] * ws.H[i, j] + ws.sn[i] * ws.H[i + 1, j]
                ws.H[i + 1, j] = -conj(ws.sn[i]) * ws.H[i, j] + ws.cs[i] * ws.H[i + 1, j]
                ws.H[i, j] = t
            end
            a = ws.H[j, j]
            bj = ws.H[j + 1, j]
            radius = R(hypot(abs(a), abs(bj)))
            if iszero(radius)
                ws.cs[j] = one(R)
                ws.sn[j] = zero(T)
            else
                ws.cs[j] = R(abs(a) / radius)
                ws.sn[j] = iszero(abs(a)) ? conj(bj) / T(radius) :
                           (a / T(abs(a))) * conj(bj) / T(radius)
            end
            ws.H[j, j] = ws.cs[j] * a + ws.sn[j] * bj
            ws.H[j + 1, j] = zero(T)
            gj = ws.g[j]
            ws.g[j + 1] = -conj(ws.sn[j]) * gj
            ws.g[j] = ws.cs[j] * gj
            # a zero rotated diagonal leaves row j of the reduced system without a solution, so
            # the residual keeps `g[j]`: this happens only at a breakdown where `J` is singular on
            # the Krylov space
            linres = iszero(radius) ? R(hypot(abs(ws.g[j]), abs(ws.g[j + 1]))) :
                     R(abs(ws.g[j + 1]))
            # on a reduced basis, a remainder at the round-off of the stored basis is no new
            # direction where the cycle would go on: the Krylov space is invariant to that
            # round-off, and the step is a breakdown: the diagonal entry `H[j, j]`, `g[j]` and the
            # residual become those of a rotation of a zero `H[j+1, j]`, which the weights and the
            # stop of the cycle read
            if reduced_basis(ws) && linres > atol && j < mr && used < maxiter &&
               basis_breakdown(ws, hj, w₀)
                hj = zero(R)
                ws.H[j, j] = a
                ws.g[j] = gj
                linres = iszero(a) ? R(abs(gj)) : zero(R)
            end
            if iszero(hj) || linres <= atol
                break
            end
            @views ws.V[:, j + 1] .= ws.w ./ hj
        end
        arnoldi = Int32(j)
        # the weights of the reduced least-squares problem, from the triangular Hessenberg matrix
        for i in j:-1:1
            acc = ws.g[i]
            for k in (i + 1):j
                acc -= ws.H[i, k] * ws.y[k]
            end
            ws.y[i] = iszero(ws.H[i, i]) ? zero(T) : acc / ws.H[i, i]
        end
        add_solution!(ws.u, ws, ws.y, j)
        resnorm = linres
        breakdown = iszero(hj)
        if reduced_basis(ws)
            # a basis in a type other than the working one makes the Givens estimate inexact, so
            # every cycle ends on the true residual, and a cycle that misses the tolerance restarts
            restart_residual!(ws, jvp_op!, P, b)
            resnorm = R(rnorm(ws.rr))
            converged = resnorm <= atol
            # a breakdown whose Givens estimate meets the tolerance leaves a true residual to
            # restart from; one whose estimate misses it is singular on the Krylov space, and stops
            (converged || (breakdown && linres > atol) || used >= maxiter) && break
            continue
        end
        if linres <= atol || breakdown || used >= maxiter
            converged = linres <= atol
            break
        end
        # a restart: the residual at the solution so far, one product
        restart_residual!(ws, jvp_op!, P, b)
    end
    solution!(ws, P)
    ws.arnoldi = arnoldi
    return ws.d, GMRESInfo{R}(used, resnorm, R(bnorm), converged, breakdown)
end

# One CGS2 step of the Arnoldi process at inner iteration `j`: `ws.w` holds `w = J M⁻¹ v_j` on
# entry and `w₂` on return, column `j` of `ws.H` takes `h₁ + h₂`, and the norm of `w₂` is
# returned. The two reductions are the products `h₁ = V_jᴴ w` and `[V_j w₁]ᴴ w₁`, for which
# column `j + 1` of the basis holds `w₁`; the second gives `h₂` and `s₁ = ‖w₁‖²` together, so the
# norm is `√(s₁ - ‖h₂‖²)` by Pythagoras. That difference keeps its digits only where `w₂` holds
# at least half of `s₁`, and `s₁` is accurate only where it is finite and at least
# `n floatmin`, the threshold of `rnorm`; elsewhere a third reduction, `rnorm(w₂)`, norms it. A
# non-finite product fails one of those tests, so `rnorm` returns its non-finite norm.
function cgs2_step!(ws::GMRESWorkspace{T, R}, j::Int) where {T, R}
    reduced_basis(ws) && return cgs2_reduced_step!(ws, j)
    n = length(ws.w)
    Vj = @view ws.V[:, 1:j]
    hd = @view ws.hd[1:j]
    w₁ = @view ws.V[:, j + 1]
    mul!(hd, Vj', ws.w)
    copyto!(ws.hh, 1, ws.hd, 1, j)
    @views ws.H[1:j, j] .= ws.hh[1:j]
    mul!(ws.w, Vj, hd, -one(T), one(T))
    w₁ .= ws.w
    mul!(@view(ws.hd[1:(j + 1)]), (@view ws.V[:, 1:(j + 1)])', w₁)
    copyto!(ws.hh, 1, ws.hd, 1, j + 1)
    @views ws.H[1:j, j] .+= ws.hh[1:j]
    mul!(ws.w, Vj, hd, -one(T), one(T))
    h₂² = R(norm2(@view ws.hh[1:j]))
    s₁ = R(real(ws.hh[j + 1]))
    pythagoras = isfinite(s₁) && s₁ >= n * floatmin(R) && h₂² <= s₁ / 2
    return pythagoras ? sqrt(s₁ - h₂²) : R(rnorm(ws.w))
end

# The CGS2 step on a basis stored in a type other than the working type: column `j + 1` cannot
# hold `w₁` in the working type, so `w₁` stays in `ws.w`, the two projections `h₁ = V_jᴴ w` and
# `h₂ = V_jᴴ w₁` are two reductions, and a third, `rnorm(w₂)`, norms the result. The products read
# the stored basis and accumulate in the working type.
function cgs2_reduced_step!(ws::GMRESWorkspace{T, R}, j::Int) where {T, R}
    Vj = @view ws.V[:, 1:j]
    hd = @view ws.hd[1:j]
    @views ws.H[1:j, j] .= zero(T)
    for _ in 1:2
        mul!(hd, Vj', ws.w)
        copyto!(ws.hh, 1, ws.hd, 1, j)
        @views ws.H[1:j, j] .+= ws.hh[1:j]
        mul!(ws.w, Vj, hd, -one(T), one(T))
    end
    return R(rnorm(ws.w))
end

# `s += V_k y` for the host weights `y`, on the path of the last solve: after an MGS solve one
# update per column of `V_k`, in the order of the basis; after a CGS2 solve one matrix–vector
# product, for which the weights go to the storage of the basis first, in `ws.hd`.
@inline add_basis!(s, ws::GMRESWorkspace, y, k::Int) = add_columns!(s, ws.V, ws, y, k)

# `s += B_k y` for the first `k` columns of `B`, the basis `V` or the preconditioned vectors `Z`,
# on the path of the last solve, as `add_basis!` describes.
@inline function add_columns!(s, B, ws::GMRESWorkspace{T}, y, k::Int) where {T}
    if ws.mgs
        for i in 1:k
            @views s .+= y[i] .* B[:, i]
        end
        return s
    end
    copyto!(ws.hd, 1, y, 1, k)
    mul!(s, (@view B[:, 1:k]), (@view ws.hd[1:k]), one(T), one(T))
    return s
end

# Whether the basis of `ws` is stored in a type other than the working type `T`.
@inline reduced_basis(::GMRESWorkspace{T, R, V}) where {T, R, V} = eltype(V) !== T

# Whether the remainder `hj` of an Arnoldi step on a reduced basis is at the round-off of the
# stored basis, `hj < c eps(TB) ‖w₀‖` for the norm `w₀` of the vector before the projections and
# the unit roundoff `eps(TB)` of the real type of the basis. `c = 2e-3` lies between the largest
# `hj / (eps(TB) ‖w₀‖)` of an invariant step whose remainder lies along the stored basis,
# 2.44e-4, and the smallest of a non-invariant step, 0.0168, at the steps where the test acts on
# the reduced-basis systems of the tests and of the standard test set
# (`scripts/gmres_breakdown_ratio.jl`).
@inline function basis_breakdown(ws::GMRESWorkspace{T, R, V}, hj, w₀) where {T, R, V}
    return hj < R(2e-3) * R(eps(real(eltype(V)))) * w₀
end

# `z = M⁻¹ v_j` for column `j` of the basis, kept in column `j` of `Z` on the flexible path. The
# preconditioner reads a copy of `v_j` in the working type, in `ws.w`, where the basis is stored in
# another type, and the column itself elsewhere.
@inline function precondition_column!(ws::GMRESWorkspace, P, j::Int)
    if reduced_basis(ws)
        @views ws.w .= ws.V[:, j]
        apply_preconditioner!(ws.z, P, ws.w)
    else
        @views apply_preconditioner!(ws.z, P, ws.V[:, j])
    end
    ws.Z === nothing || (@views ws.Z[:, j] .= ws.z)
    return ws.z
end

# The update of the solution by the weights `y` of a cycle: `u += V_k y`, whose direction is
# `M⁻¹ u`, and on the flexible path `u += Z_k y`, which is the direction itself.
@inline function add_solution!(s, ws::GMRESWorkspace, y, k::Int)
    ws.Z === nothing && return add_basis!(s, ws, y, k)
    return add_columns!(s, ws.Z, ws, y, k)
end

# The residual `b - J d` of the solution so far in `ws.rr`, one product: `d = M⁻¹ u`, and `d = u`
# on the flexible path.
@inline function restart_residual!(ws::GMRESWorkspace, jvp_op!, P, b)
    if ws.Z === nothing
        apply_preconditioner!(ws.z, P, ws.u)
        jvp_op!(ws.w, ws.z)
    else
        jvp_op!(ws.w, ws.u)
    end
    @views ws.rr .= b .- ws.w
    return ws.rr
end

# The direction of the solve in `ws.d`: `M⁻¹ u`, and `u` on the flexible path.
@inline function solution!(ws::GMRESWorkspace, P)
    ws.Z === nothing && return apply_preconditioner!(ws.d, P, ws.u)
    return copy!!(ws.d, ws.u)
end

"""
    arnoldi_dot(a, b)

The conjugated inner product ``\\sum_i \\overline{a_i} b_i`` of two vectors, in the element type of
`b`: the coefficient of modified Gram–Schmidt, which is real for a real iterate and complex for a
complex one. It reduces through a lazy broadcast, so it allocates nothing and indexes no entry.
"""
function arnoldi_dot(a, b)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("arnoldi_dot arguments have different axes"))
    isempty(a) && return zero(eltype(b))
    products = Broadcast.instantiate(Broadcast.broadcasted((x, y) -> conj(x) * y, a, b))
    return sum(products)
end

# `H̄_k y` for a weight vector `y` of length `k`, in `ws.sm[1:k+1]`: the product of the unrotated
# Hessenberg matrix with the weights, whose norm is that of `J (V_k y)`. It costs `O(k²)` on the
# host and no Jacobian–vector product.
@inline function hessenberg_product!(ws::GMRESWorkspace{T}, y, k::Int) where {T}
    for i in 1:(k + 1)
        acc = zero(T)
        for j in 1:k
            acc += ws.H0[i, j] * y[j]
        end
        ws.sm[i] = acc
    end
    return @view ws.sm[1:(k + 1)]
end

# `-β₀ e₁ + H̄_k y`: the reduced vector whose norm is the norm of `F + J (V_k y)`, because `V` has
# orthonormal columns and `F = -β₀ V e₁`.
@inline function reduced_model!(ws::GMRESWorkspace{T}, y, k::Int, β₀) where {T}
    model = hessenberg_product!(ws, y, k)
    ws.sm[1] -= T(β₀)
    return model
end

# `‖H̄_k y‖₂` of a weight vector: the norm of `J (V_k y)`.
@inline function reduced_hessenberg_norm(ws::GMRESWorkspace, y, k::Int)
    return rnorm(hessenberg_product!(ws, y, k))
end

# `V_k y` in the storage of `s`, on the path of the last solve on `ws`. The accumulation is the one
# `gmres_solve!` makes for the solution from zero, so the weights of an unrestarted GMRES direction
# rebuild it bit for bit.
@inline function reduced_direction!(s, ws::GMRESWorkspace{T}, y, k::Int) where {T}
    fill!(s, zero(T))
    return add_basis!(s, ws, y, k)
end

"""
    CountingResidual(F, ws)

The residual `F` of an R5 solve with the workspace `ws`, whose `f_calls` it increments: every
evaluation of the residual of the solve goes through it, the trial evaluations of a globalization
included, so that the count of the status is the count of the solve. Both call forms of
[`residual!!`](@ref) are methods of it.
"""
struct CountingResidual{F, W}
    F::F
    ws::W
end

@inline function (c::CountingResidual)(y, x, p)
    c.ws.f_calls += Int32(1)
    c.F(y, x, p)
    return y
end

@inline function (c::CountingResidual)(x, p)
    c.ws.f_calls += Int32(1)
    return c.F(x, p)
end

# The problem a globalization of an R5 step reads: the residual of `prob` counted in `ws`, and
# `prob` itself, which the Jacobian–vector product is formed on. `globalize!!` and `LineFunction`
# read no other field of a problem, and a product is not an evaluation of `F`: a `jvp!!` that
# `DIJacobian` prepared must be called on the function it prepared, which the counting wrapper is
# not, and the chunked forward mode evaluates `F` on duals, which counts a product.
struct R5CountingProblem{P, F}
    prob::P
    F::F
end

@inline counting_problem(prob, ws) = R5CountingProblem(prob, CountingResidual(prob.F, ws))

"""
    R5JVP(prep, ws)
    R5JVP{P, W, TJ}(prep, ws)

The Jacobian–vector preparation `prep` of an R5 solve, which counts its products in the workspace
`ws` and, where `TJ` names an element type, evaluates the product in `TJ` and promotes it to the
working type of the iterate: the iterate, the direction and the product are converted to `TJ`, the
product is formed there, and its result is copied back. That is a defect-correction solve, where the
residual in the working precision keeps the accuracy of the outer solve.
"""
struct R5JVP{P, W, TJ}
    prep::P
    ws::W
end

R5JVP(prep, ws) = R5JVP{typeof(prep), typeof(ws), Nothing}(prep, ws)

# The problem a product is formed on: the one that a counting problem wraps, and any other as given.
@inline raw_problem(prob) = prob
@inline raw_problem(prob::R5CountingProblem) = prob.prob

@inline function jvp!!(Jv, c::R5JVP{P, W, Nothing}, prob, x, v, p) where {P, W}
    c.ws.jvp_calls += Int32(1)
    return jvp!!(Jv, c.prep, raw_problem(prob), x, v, p)
end

@inline function jvp!!(Jv, c::R5JVP{P, W, TJ}, prob, x, v, p) where {P, W, TJ}
    c.ws.jvp_calls += Int32(1)
    xT = copy!!(c.ws.xT, x)
    vT = copy!!(c.ws.vT, v)
    wT = jvp!!(c.ws.wT, c.prep, raw_problem(prob), xT, vT, p)
    return copy!!(Jv, wT)
end

"""
    R5Operator(prep, prob, x, p)

The Jacobian–vector product of one [`gmres_solve!`](@ref) at the Newton iterate `x`: the callable
`(Jv, v) -> jvp!!(Jv, prep, prob, x, v, p)`, whose fields are the preparation, the problem, the
iterate and the parameters. It is a struct and not a closure, so that a step allocates none.
"""
struct R5Operator{P, Q, X, W}
    prep::P
    prob::Q
    x::X
    p::W
end

@inline function (op::R5Operator)(Jv, v)
    return jvp!!(Jv, op.prep, op.prob, op.x, v, op.p)
end

"""
    max_inner_iterations(solver)

The largest count of inner iterations of any Newton step of the last solve of `solver`, a
[`NonlinearSolver`](@ref) whose method is a [`Newton`](@ref) with [`GMRES`](@ref),
[`BiCGStab`](@ref), [`DQGMRES`](@ref) or [`MINRESQLP`](@ref). It is at most the `maxiter` of the
linear method, except under a [`Steihaug`](@ref) globalization, where the count of a step adds the
conjugate-gradient iterations of every run of its trial loop.
"""
function max_inner_iterations end

max_inner_iterations(ws::GMRESWorkspace) = ws.max_inner

"""
    f_calls(solver)

The number of evaluations of the residual `F` that the last solve of `solver` made, for a
[`NonlinearSolver`](@ref) whose method is a [`Newton`](@ref) with [`GMRES`](@ref),
[`BiCGStab`](@ref), [`DQGMRES`](@ref) or [`MINRESQLP`](@ref): the count that
[`SolverStatus`](@ref) records, saturated there at `typemax(Int16)` and exact here below it.
"""
function f_calls end

f_calls(ws::GMRESWorkspace) = ws.f_calls

"""
    jvp_calls(solver)

The number of Jacobian–vector products that the last solve of `solver` made, for a
[`NonlinearSolver`](@ref) whose method is a [`Newton`](@ref) with [`GMRES`](@ref),
[`BiCGStab`](@ref), [`DQGMRES`](@ref) or [`MINRESQLP`](@ref): the count that
[`SolverStatus`](@ref) records, saturated there at `typemax(Int16)` and exact here below it.
"""
function jvp_calls end

jvp_calls(ws::GMRESWorkspace) = ws.jvp_calls
