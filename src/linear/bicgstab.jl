# R5 matrix-free Newton–Krylov with fixed storage: the right-preconditioned BiCGStab of van der
# Vorst (1992, doi:10.1137/0913035), a linear method of `Newton` beside `GMRES`.
#
# The iteration keeps nine vectors of the length of the iterate and no basis, takes two
# Jacobian–vector products per iteration and one where the half step already meets the tolerance,
# and minimises no residual. The recurrence runs on the right-hand side scaled to unit norm, so
# that its scalars stay near the scale of the system, and `‖t‖` takes the place of `tᴴt`, which
# overflows from `‖t‖ ≈ √floatmax` on. Every vector is in the storage of the iterate, the loop
# allocates nothing after `init_bicgstab_workspace`, and it indexes no entry of a vector, so the
# same code runs on a `JLArray` under `allowscalar(false)`.

"""
    BiCGStab(; maxiter = 30, forcing = EisenstatWalkerChoice2(), preconditioner = nothing)

A matrix-free linear method of [`Newton`](@ref): the right-preconditioned BiCGStab of van der
Vorst (1992, doi:10.1137/0913035) on the Jacobian–vector product of the problem, with the forcing
term `forcing` ([`EisenstatWalkerChoice2`](@ref) by default, [`EisenstatWalkerChoice1`](@ref) or
[`FixedForcing`](@ref)) as the relative tolerance of each Newton step's inner solve, and the right
preconditioner `preconditioner`, which [`refresh!`](@ref) and [`apply_preconditioner!`](@ref) serve
as for [`GMRES`](@ref).

# Fields

- `maxiter::Int32`: the cap on the inner iterations of one Newton step, `≥ 1`. An iteration takes
  two Jacobian–vector products and two applications of the preconditioner, and one of each where
  its half step meets the tolerance and ends the solve. A step that reaches the cap without
  meeting the tolerance of its forcing term counts a step-rule failure.
- `forcing::F`: the rule that sets the relative tolerance ``\\eta_k`` of step `k`.
- `preconditioner::P`: the right preconditioner, `nothing` for none; the solve forms
  ``J M^{-1} u = -F`` and returns ``d = M^{-1} u``.

The solve stores nine vectors of the length of the iterate, whatever the iteration count, and
minimises no residual: its residual norm need not decrease from one iteration to the next. It
stops where ``\\|F + J d\\| \\le \\eta_k \\|F\\|`` by its recurrence. Four scalars of the
recurrence divide: ``\\sigma = \\hat r^H v``, ``\\|t\\|``, ``\\omega`` and ``\\rho = \\hat r^H r``.
Where one of them is exactly zero the solve stops, a breakdown: it returns the last finite
iterate, ``x + \\alpha \\hat p`` where ``\\omega = 0`` or ``t = 0``, with no convergence, and the
Newton step goes on with that direction as after a `GMRES` breakdown on a singular operator. A
product that is not finite ends the step with `NONFINITE`, and the iterate stays.

A `Newton` with `BiCGStab` takes the iterate and the problems of a `Newton` with `GMRES`, and the
`forcing` and `preconditioner` keywords of `Newton` replace those of the method. It takes no
`reuse`, no [`Refinement`](@ref), no [`LevenbergMarquardt`](@ref), no [`PseudoTransient`](@ref) and
no [`DogLeg`](@ref), whose trial loop reads the Arnoldi relation of a `GMRES` solve, and `solve`
refuses a `ForwardDiff.Dual` in `x₀` or `p`; each refusal raises an `ArgumentError`, and so does
`maxiter < 1`.
"""
struct BiCGStab{F <: AbstractForcingTerm, P} <: LinearMethod{Nothing}
    maxiter::Int32
    forcing::F
    preconditioner::P
    # `adapt` rebuilds the method through this constructor, inside a device kernel too.
    function BiCGStab{F, P}(maxiter::Int32, forcing::F,
            preconditioner::P) where {F <: AbstractForcingTerm, P}
        maxiter >= Int32(1) || throw(ArgumentError("BiCGStab maxiter must be ≥ 1"))
        return new{F, P}(maxiter, forcing, preconditioner)
    end
end

function BiCGStab(; maxiter::Integer = 30,
        forcing::AbstractForcingTerm = EisenstatWalkerChoice2(), preconditioner = nothing)
    m = to_int32(maxiter, "BiCGStab maxiter must fit in an Int32")
    return BiCGStab{typeof(forcing), typeof(preconditioner)}(m, forcing, preconditioner)
end

# The positional form, which `adapt` rebuilds the method through.
@inline function BiCGStab(maxiter::Int32, forcing::AbstractForcingTerm, preconditioner)
    return BiCGStab{typeof(forcing), typeof(preconditioner)}(maxiter, forcing, preconditioner)
end

Adapt.@adapt_structure BiCGStab

# The workspace of one BiCGStab solve: the solution `d`, the residual `r` of the scaled
# recurrence, which holds `s` after the half step, the shadow residual `rh`, the direction `p`, its
# preconditioned `ph`, the product `v = J ph`, the preconditioned half-step residual `sh`, its product
# `t`, and the right-hand side `bb`, all in the storage of the iterate; and the counters of a solve,
# as in `MINRESQLPWorkspace`.
mutable struct BiCGStabWorkspace{T, R <: Real, V}
    d::V
    r::V
    rh::V
    p::V
    ph::V
    v::V
    sh::V
    t::V
    bb::V
    f_calls::Int32
    jvp_calls::Int32
    iterations::Int32
    max_inner::Int32
    step::Int32
    last_eta::R
    prev_fnorm::R
    linres::R
end

"""
    init_bicgstab_workspace(g::BiCGStab, x::AbstractVector)

Allocate the workspace of a [`BiCGStab`](@ref) solve on an iterate like `x`: nine vectors of its
length and storage, whose sizes no solve changes.
"""
function init_bicgstab_workspace(::BiCGStab, x::AbstractVector{T}) where {T}
    R = real(T)
    vs = ntuple(_ -> buffer(x), 9)
    return BiCGStabWorkspace{T, R, typeof(vs[1])}(vs...,
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), R(NaN), R(NaN), R(NaN))
end

# The conjugated inner product `Σ conj(aᵢ / c) bᵢ` of the vector `a` scaled by the real `c`, in
# the element type of `b`: `aᴴb / c` without the product `aᴴb`, which can overflow where the
# quotient does not. A lazy broadcast, as in `arnoldi_dot`.
function scaled_dot(a, b, c)
    products = Broadcast.instantiate(Broadcast.broadcasted((x, y) -> conj(x / c) * y, a, b))
    return sum(products)
end

"""
    bicgstab_solve!(ws::BiCGStabWorkspace, jvp_op!, P, b, bnorm, tol, maxiter) -> (d, GMRESInfo)

Solve `J d = b` with right-preconditioned BiCGStab in the workspace `ws`, and return the direction
`d = M⁻¹ u` in the slot `ws.d` and a [`GMRESInfo`](@ref). `jvp_op!(Jv, v)` writes `J v` into
`Jv`, `P` is the right preconditioner, `bnorm` is ``\\|b\\|_2``, and the solve stops at a
residual of the recurrence ``\\|b - J d\\|_2 \\le tol \\cdot bnorm`` or after `maxiter`
iterations.

The recurrence runs on `b / bnorm`, with the shadow residual ``\\hat r = b / bnorm``, and
the solution adds the steps times `bnorm`. The inner products ``\\sigma = \\hat r^H v``,
``\\rho = \\hat r^H r`` and ``\\omega = t^H s / t^H t`` conjugate their first argument
([`arnoldi_dot`](@ref)), and ``\\omega`` is formed as ``((t / \\|t\\|)^H s) / \\|t\\|``. A
`σ`, `‖t‖`, `ω` or `ρ` that is exactly zero stops the solve with `breakdown = true` and the last
finite iterate; its residual norm is that of the recurrence at that iterate. A residual norm of
the half step or of the full step that is not finite stops the solve in that iteration, with that
norm; a `σ` or an `ω` that is not finite gives such a norm. `b = 0` takes no product.
"""
function bicgstab_solve!(ws::BiCGStabWorkspace{T, R}, jvp_op!, P, b, bnorm, tol,
        maxiter) where {T, R}
    β₀ = R(bnorm)
    rtol = R(tol)
    d, r, rh, p, ph, v, sh, t = ws.d, ws.r, ws.rh, ws.p, ws.ph, ws.v, ws.sh, ws.t
    fill!(d, zero(T))
    k = Int32(0)
    iszero(β₀) && return d, GMRESInfo{R}(k, zero(R), β₀, true, false)
    r .= b ./ β₀
    copy!!(rh, r)
    copy!!(p, r)
    ρ = arnoldi_dot(rh, r)
    rel = R(rnorm(r))
    converged = rel <= rtol
    breakdown = false
    while !converged && k < maxiter
        k += one(Int32)
        apply_preconditioner!(ph, P, p)
        jvp_op!(v, ph)
        σ = arnoldi_dot(rh, v)
        if iszero(σ)
            breakdown = true
            break
        end
        α = ρ / σ
        # the half step: `r` holds `s = r - α v` from here on
        r .-= α .* v
        rel = R(rnorm(r))
        if !isfinite(rel)
            break
        end
        if rel <= rtol
            d .+= (β₀ * α) .* ph
            converged = true
            break
        end
        apply_preconditioner!(sh, P, r)
        jvp_op!(t, sh)
        τ = R(rnorm(t))
        if iszero(τ)
            d .+= (β₀ * α) .* ph
            breakdown = true
            break
        end
        ω = scaled_dot(t, r, τ) / τ
        if iszero(ω)
            d .+= (β₀ * α) .* ph
            breakdown = true
            break
        end
        d .+= (β₀ * α) .* ph .+ (β₀ * ω) .* sh
        r .-= ω .* t
        rel = R(rnorm(r))
        if !isfinite(rel)
            break
        end
        if rel <= rtol
            converged = true
            break
        end
        ρ₊ = arnoldi_dot(rh, r)
        if iszero(ρ₊)
            breakdown = true
            break
        end
        β = (ρ₊ / ρ) * (α / ω)
        ρ = ρ₊
        p .= r .+ β .* (p .- ω .* v)
    end
    return d, GMRESInfo{R}(k, β₀ * rel, β₀, converged, breakdown)
end

# The counters of a solve, from the seed of `solve`, as for a `GMRESWorkspace`.
max_inner_iterations(ws::BiCGStabWorkspace) = ws.max_inner
f_calls(ws::BiCGStabWorkspace) = ws.f_calls
jvp_calls(ws::BiCGStabWorkspace) = ws.jvp_calls
