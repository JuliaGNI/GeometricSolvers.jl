# R5 matrix-free Newton–Krylov: a right-preconditioned GMRES with Eisenstat and Walker's forcing
# terms, on the array backends of R3 and on devices.
#
# `GMRES` is a linear-solver method without a factor: the Krylov basis is its storage, and the
# system matrix enters only through a Jacobian–vector product. A `Newton` with it therefore holds
# no `n × n` Jacobian, and its step solves `J d = -F` to the relative tolerance `η_k` that the
# forcing term of the step returns.
#
# The Arnoldi loop is the modified Gram–Schmidt process of Saad (2003), §6.5.2, with the
# Hessenberg matrix, its Givens rotations and the least-squares right-hand side on the host, and
# every vector of length `n` in the storage of the iterate: the loop allocates nothing after
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
            preconditioner = nothing, jvp_eltype = Nothing)

The R5 matrix-free Newton–Krylov linear solver of [`Newton`](@ref): right-preconditioned GMRES
with modified Gram–Schmidt on the Jacobian–vector product of the problem, the forcing term
`forcing` ([`EisenstatWalkerChoice2`](@ref) by default,
[`EisenstatWalkerChoice1`](@ref) or [`FixedForcing`](@ref)), a right preconditioner
`preconditioner`, and the element type `jvp_eltype` of the product.

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
  is called once per Newton step, before GMRES starts, and `apply_preconditioner!(z, P, v)` once
  per inner iteration; the solve forms ``J M^{-1} u = -F`` and returns ``d = M^{-1} u``. A type
  that carries a preconditioner extends those two functions, for example a stored factorisation
  that `ldiv!` applies, which is the frozen-Jacobian preconditioner of a Newton–Krylov method.
- `jvp_eltype::Type{TJ}`: `Nothing` evaluates the Jacobian–vector product in the working type of
  the iterate; a real type `TJ` evaluates it in `TJ` and promotes the product to the working type,
  which is a defect-correction solve: a product in a lower precision, a residual in the working
  one. The iterate, the residual and every norm stay in the working type.

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
struct GMRES{F <: AbstractForcingTerm, P, TJ} <: LinearMethod{Nothing}
    maxiter::Int32
    restart::Int32
    forcing::F
    preconditioner::P
    jvp_eltype::Type{TJ}
    # `adapt` rebuilds the method through this constructor, inside a device kernel too.
    function GMRES{F, P, TJ}(maxiter::Int32, restart::Int32, forcing::F,
            preconditioner::P, jvp_eltype::Type{TJ}) where {
            F <: AbstractForcingTerm, P, TJ}
        maxiter >= Int32(1) || throw(ArgumentError("GMRES maxiter must be ≥ 1"))
        (Int32(1) <= restart <= maxiter) ||
            throw(ArgumentError("GMRES needs 1 ≤ restart ≤ maxiter"))
        return new{F, P, TJ}(maxiter, restart, forcing, preconditioner, jvp_eltype)
    end
end

function GMRES(; maxiter::Integer = 30, restart::Union{Nothing, Integer} = nothing,
        forcing::AbstractForcingTerm = EisenstatWalkerChoice2(),
        preconditioner = nothing, jvp_eltype::Type = Nothing)
    m = to_int32(maxiter, "GMRES maxiter must fit in an Int32")
    r = restart === nothing ? m :
        to_int32(restart, "GMRES restart must fit in an Int32")
    return GMRES{typeof(forcing), typeof(preconditioner), jvp_eltype}(
        m, r, forcing, preconditioner, jvp_eltype)
end

# An `Integer` constructor argument as an `Int32`, with a constant message where it does not fit: a
# kernel cannot format one.
@inline function to_int32(n::Integer, message::String)
    (typemin(Int32) <= n <= typemax(Int32)) || throw(ArgumentError(message))
    return Int32(n)
end

# The positional form, which `adapt` rebuilds the method through: the element type of the product
# is a type and not a field, so it travels as the fifth argument.
@inline function GMRES(maxiter::Int32, restart::Int32, forcing::AbstractForcingTerm,
        preconditioner, jvp_eltype::Type)
    return GMRES{typeof(forcing), typeof(preconditioner), jvp_eltype}(
        maxiter, restart, forcing, preconditioner, jvp_eltype)
end

Adapt.@adapt_structure GMRES

"""
    refresh!(P, prob, x, Fx, p)

Refresh the right preconditioner `P` of [`GMRES`](@ref) at the start of a Newton step, from the
iterate `x`, its residual `Fx` and the parameters `p`. [`Newton`](@ref) calls it once per step,
before the inner solve, so that the preconditioner of step `k` sees the data of the last residual
evaluation (Chen et al. 2014, pp. 558–560). The method for `nothing` does nothing; a
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
storage of the iterate, the host Hessenberg matrix `H` after the Givens rotations and `H0` before
them, the rotation tables `cs` and `sn`, the right-hand side `g` of the least-squares problem, the
solution weights `y`, the weights `cg`, `yN` and `yb` of the matrix-free [`DogLeg`](@ref) and its
reduced work vector `sm`, the vector slots `rr`, `z`, `w`, `u`, `d` and `bb` of length `n`, the
`jvp_eltype` slots `xT`, `vT` and `wT`, and the counters of the solve.

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
mutable struct GMRESWorkspace{T, R <: Real, V, M, SR, ST, NV, D}
    V::V
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
holds `min(g.restart, length(x)) + 1` columns, and the host matrices `H` and `H0` and the host
vectors are of that width. The slots of `jvp_eltype` are `nothing` unless `g.jvp_eltype` names a
type.
"""
function init_gmres_workspace(g::GMRES{F, P, TJ}, x::AbstractVector{T}) where {F, P, TJ, T}
    R = real(T)
    n = length(x)
    mr = min(Int(g.restart), n)
    w = mr + 1
    V = buffer(x, T, n, w)
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
    return GMRESWorkspace{T, R, typeof(V), typeof(H), typeof(cs), typeof(sn),
        typeof(rr), typeof(xT)}(
        V, H, H0, cs, sn, gb, y, cg, yN, yb, sm, rr, z, wv, u, d, bb, xT, vT, wT,
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
    gmres_solve!(ws::GMRESWorkspace, jvp_op!, P, b, bnorm, tol, maxiter, restart)
        -> (d, GMRESInfo)

Solve `J d = b` with right-preconditioned GMRES in the workspace `ws`, and return the direction
`d = M⁻¹ u` in the slot `ws.d` and a [`GMRESInfo`](@ref). `jvp_op!(Jv, v)` writes the product
`J v` into `Jv`; `P` is the right preconditioner; `b` is the right-hand side and `bnorm` its norm
``\\|b\\|_2``, which the caller already holds; `tol` is the relative tolerance, so that the solve
stops at ``\\|b - J d\\|_2 \\le tol \\cdot bnorm``; `maxiter` caps the inner iterations and
`restart` the iterations between restarts, neither of which may exceed the width the workspace was
allocated for.

The Arnoldi process is modified Gram–Schmidt with the conjugated inner product, so a complex
iterate is supported and every norm is in `real(T)`. A zero `H[j+1, j]` is a breakdown: the
Krylov space is invariant, and the solve stops there rather than dividing by zero. The residual of
the reduced problem is then zero, unless `J` is singular on that space: the rotated diagonal entry
is then zero too, the residual is that of the last entry of the right-hand side, and the solve
reports no convergence. A restarted cycle forms its new residual with one more product,
`b - J M⁻¹ u`, so a restart costs one Jacobian–vector product. The least-squares problem is solved
from the Givens-rotated Hessenberg matrix, whose diagonal entry is `0` only where the reduced
system is singular, and the corresponding weight is then `0`.
"""
function gmres_solve!(ws::GMRESWorkspace{T, R}, jvp_op!, P, b, bnorm, tol, maxiter,
        restart) where {T, R}
    n = length(b)
    mr = min(min(Int(restart), n), size(ws.V, 2) - 1)
    atol = R(tol) * R(bnorm)
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
            @views apply_preconditioner!(ws.z, P, ws.V[:, j])
            jvp_op!(ws.w, ws.z)
            # modified Gram–Schmidt: one pass of projections, in the order of the basis
            for i in 1:j
                h = arnoldi_dot(@view(ws.V[:, i]), ws.w)
                ws.H[i, j] = h
                @views ws.w .-= h .* ws.V[:, i]
            end
            hj = R(rnorm(ws.w))
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
        for i in 1:j
            @views ws.u .+= ws.y[i] .* ws.V[:, i]
        end
        resnorm = linres
        breakdown = iszero(hj)
        if linres <= atol || breakdown || used >= maxiter
            converged = linres <= atol
            break
        end
        # a restart: the residual b - J M⁻¹ u at the solution so far, one product
        apply_preconditioner!(ws.z, P, ws.u)
        jvp_op!(ws.w, ws.z)
        @views ws.rr .= b .- ws.w
    end
    apply_preconditioner!(ws.d, P, ws.u)
    ws.arnoldi = arnoldi
    return ws.d, GMRESInfo{R}(used, resnorm, R(bnorm), converged, breakdown)
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

# `V_k y` in the storage of `s`. The accumulation is the one `gmres_solve!` makes for the solution,
# in the same order, so the weights of an unrestarted GMRES direction rebuild it bit for bit.
@inline function reduced_direction!(s, ws::GMRESWorkspace{T}, y, k::Int) where {T}
    fill!(s, zero(T))
    for j in 1:k
        @views s .+= y[j] .* ws.V[:, j]
    end
    return s
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

The largest count of GMRES inner iterations of any Newton step of the last solve of `solver`, a
[`NonlinearSolver`](@ref) whose method is a [`Newton`](@ref) with [`GMRES`](@ref). It is at most
the `maxiter` of the linear method.
"""
function max_inner_iterations end

max_inner_iterations(ws::GMRESWorkspace) = ws.max_inner

"""
    f_calls(solver)

The number of evaluations of the residual `F` that the last solve of `solver` made, for a
[`NonlinearSolver`](@ref) whose method is a [`Newton`](@ref) with [`GMRES`](@ref): the count that
[`SolverStatus`](@ref) records, saturated there at `typemax(Int16)` and exact here below it.
"""
function f_calls end

f_calls(ws::GMRESWorkspace) = ws.f_calls

"""
    jvp_calls(solver)

The number of Jacobian–vector products that the last solve of `solver` made, for a
[`NonlinearSolver`](@ref) whose method is a [`Newton`](@ref) with [`GMRES`](@ref): the count that
[`SolverStatus`](@ref) records, saturated there at `typemax(Int16)` and exact here below it.
"""
function jvp_calls end

jvp_calls(ws::GMRESWorkspace) = ws.jvp_calls
