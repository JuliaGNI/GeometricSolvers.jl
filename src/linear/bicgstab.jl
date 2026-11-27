# R5 matrix-free Newton–Krylov with fixed storage: the right-preconditioned BiCGStab of van der
# Vorst (1992, doi:10.1137/0913035), a linear method of `Newton` beside `GMRES`.
#
# The iteration keeps nine vectors of the length of the iterate and no basis, takes two
# Jacobian–vector products per iteration and one where the half step already meets the tolerance,
# and minimises no residual. The recurrence runs on the right-hand side scaled to unit norm, and
# each preconditioned vector and each product is divided by the power of two of its norm, so that
# the scalars of the recurrence carry neither the scale of the system nor that of the
# preconditioner; `‖t‖` takes the place of `tᴴt`, which overflows from `‖t‖ ≈ √floatmax` on. The
# update of the solution takes the powers of two back in its coefficients, `scaled_coefficient`.
# A division by a power of two is exact, so in the normal range the recurrence rounds as it does
# unscaled. Every vector is in the storage of the iterate, the loop allocates nothing after
# `init_bicgstab_workspace`, and it indexes no entry of a vector, so the same code runs on a
# `JLArray` under `allowscalar(false)`.

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
iterate, ``d + \\alpha \\hat p`` where ``\\omega = 0`` or ``t = 0``, with no convergence, and the
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
# preconditioned `ph`, the product `v = J ph`, the preconditioned half-step residual `sh`, its
# product `t`, and the right-hand side `bb`, all in the storage of the iterate; and the counters of
# a solve, as in `MINRESQLPWorkspace`.
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

# The exponent `e` of the power of two `2^e ≤ ν < 2^(e+1)` that a vector of norm `ν` is divided
# by, and `0` for a `ν` that is zero or not finite, which the guards of the solve read unscaled.
@inline norm_exponent(ν) = isfinite(ν) && !iszero(ν) ? exponent(ν) : 0

# The coefficient `β α 2^-e` of a scaled vector in the solution, for the real `β ≥ 0`: with
# `β ≥ 1` the result is at least `|α 2^-e|`, and with `β < 1` the product `β α` is at most `|α|`,
# so an intermediate overflows only where the result does.
@inline function scaled_coefficient(β, α, e)
    return β >= one(β) ? β * ldexp2(α, -e) : ldexp2(β * α, -e)
end

# `β = ρq (α / ω) 2^Δ` of the scaled recurrence, with `ρq = ρ₊/ρ`: the shift applies after the
# division where it grows, and before it where it shrinks, so `α / ω` overflows only where `β`
# does. In the normal range both orders round as `ldexp2(α / ω, Δ)`.
@inline bicgstab_beta(ρq, α, ω, Δ) = ρq * (Δ >= 0 ? ldexp2(α / ω, Δ) : ldexp2(α, Δ) / ω)

# The next direction `p ← r + β (p - ω 2^-Δ v)` of the scaled recurrence. Where `ω 2^-Δ` leaves
# the normal range, the product `β ω 2^-Δ`, which is `ρq α`, takes its place, so that no factor
# overflows or flushes where the direction does not; otherwise the direction rounds as the
# unscaled recurrence does.
@inline function bicgstab_direction!(p, r, v, β, ρq, α, ω, Δ)
    ωv = ldexp2(ω, -Δ)
    if isfinite(ωv) && abs(ωv) >= floatmin(real(typeof(ωv)))
        p .= r .+ β .* (p .- ωv .* v)
    else
        p .= r .+ β .* p .- (ρq * α) .* v
    end
    return p
end

"""
    bicgstab_solve!(ws::BiCGStabWorkspace, jvp_op!, P, b, bnorm, tol, maxiter) -> (d, GMRESInfo)

Solve `J d = b` with right-preconditioned BiCGStab in the workspace `ws`, and return the direction
`d = M⁻¹ u` in the slot `ws.d` and a [`GMRESInfo`](@ref). `jvp_op!(Jv, v)` writes `J v` into
`Jv`, `P` is the right preconditioner, `bnorm` is ``\\|b\\|_2``, and the solve stops at a
residual of the recurrence ``\\|b - J d\\|_2 \\le tol \\cdot bnorm`` or after `maxiter`
iterations.

The recurrence runs on `b / bnorm`, with the shadow residual ``\\hat r = b / bnorm``, and
the solution adds the steps times `bnorm`. The preconditioned vectors `M⁻¹ p` and `M⁻¹ s` and
their products `v` and `t` are each divided by the power of two `2^e ≤ ‖·‖ < 2^(e+1)` of their
norm, and the scalars ``\\alpha`` and ``\\omega`` of the recurrence are formed on the scaled
vectors, so that they carry neither the scale of `J` nor that of `M⁻¹`; the coefficient of a step
in the solution takes `bnorm` and the powers of two back. The divisions are exact, so in the
normal range the iterates are those of the unscaled recurrence. The inner products
``\\sigma = \\hat r^H v``, ``\\rho = \\hat r^H r`` and ``\\omega = t^H s / t^H t`` conjugate their
first argument ([`arnoldi_dot`](@ref)), and ``\\omega`` is formed as
``((t / \\|t\\|)^H s) / \\|t\\|``. A `σ`, `‖t‖`, `ω` or `ρ` that is exactly zero stops the solve
with `breakdown = true` and the last finite iterate; its residual norm is that of the recurrence
at that iterate. A residual norm of the half step or of the full step that is not finite stops
the solve in that iteration, with that norm; a `σ` or an `ω` that is not finite gives such a norm.
`b = 0` takes no product.
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
        # `ph` and `v` hold `M⁻¹p 2^-eμ` and `J M⁻¹p 2^-(eμ+eν)`, so `α` is the step along `v`,
        # and `α 2^-(eμ+eν)` that along `J M⁻¹p`
        apply_preconditioner!(ph, P, p)
        eμ = norm_exponent(R(rnorm(ph)))
        iszero(eμ) || (ph .= ldexp2.(ph, -eμ))
        jvp_op!(v, ph)
        eν = norm_exponent(R(rnorm(v)))
        iszero(eν) || (v .= ldexp2.(v, -eν))
        σ = arnoldi_dot(rh, v)
        if iszero(σ)
            breakdown = true
            break
        end
        α = ρ / σ
        cα = scaled_coefficient(β₀, α, eν)
        # the half step: `r` holds `s = r - α v` from here on
        r .-= α .* v
        rel = R(rnorm(r))
        if !isfinite(rel)
            break
        end
        if rel <= rtol
            d .+= cα .* ph
            converged = true
            break
        end
        # `sh` and `t` hold `M⁻¹s 2^-eη` and `J M⁻¹s 2^-(eη+eτ)`, and `τ` is the norm of `t`
        apply_preconditioner!(sh, P, r)
        eη = norm_exponent(R(rnorm(sh)))
        iszero(eη) || (sh .= ldexp2.(sh, -eη))
        jvp_op!(t, sh)
        τ = R(rnorm(t))
        eτ = norm_exponent(τ)
        if !iszero(eτ)
            t .= ldexp2.(t, -eτ)
            τ = ldexp2(τ, -eτ)
        end
        if iszero(τ)
            d .+= cα .* ph
            breakdown = true
            break
        end
        ω = scaled_dot(t, r, τ) / τ
        if iszero(ω)
            d .+= cα .* ph
            breakdown = true
            break
        end
        d .+= cα .* ph .+ scaled_coefficient(β₀, ω, eτ) .* sh
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
        # `β = (ρ₊/ρ)(α'/ω')` and `p - ω' J M⁻¹p` of the unscaled steps `α' = α 2^-(eμ+eν)` and
        # `ω' = ω 2^-(eη+eτ)`, with `J M⁻¹p = v 2^(eμ+eν)`
        Δ = (eη + eτ) - (eμ + eν)
        ρq = ρ₊ / ρ
        β = bicgstab_beta(ρq, α, ω, Δ)
        ρ = ρ₊
        bicgstab_direction!(p, r, v, β, ρq, α, ω, Δ)
    end
    return d, GMRESInfo{R}(k, β₀ * rel, β₀, converged, breakdown)
end

# The counters of a solve, from the seed of `solve`, as for a `GMRESWorkspace`.
max_inner_iterations(ws::BiCGStabWorkspace) = ws.max_inner
f_calls(ws::BiCGStabWorkspace) = ws.f_calls
jvp_calls(ws::BiCGStabWorkspace) = ws.jvp_calls
