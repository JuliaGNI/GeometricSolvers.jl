# R5 matrix-free Newton–Krylov on a symmetric Jacobian: MINRES-QLP (Choi, Paige and Saunders
# 2011), the linear method of a `StationarityProblem` with `E == U`, whose Jacobian is a block of
# the Hessian of `S`.
#
# The Lanczos process builds `A V_k = V_{k+1} T̄_k` with four vectors and a short recurrence, so no
# Krylov basis is stored. Left reflections `Q_k` reduce `T̄_k` to the upper tridiagonal `R_k`, and
# right reflections `P_k` reduce `R_k` to the lower tridiagonal `L_k = R_k P_k`; the solution is
# `x_k = V_k P_k u_k` with `L_k u_k = Q_k β₁ e₁`, and only the last three columns of `W_k = V_k P_k`
# change from one iteration to the next. A diagonal entry `γ` of `L_k` at most `√eps(R) ‖T̄_k‖`
# counts as zero, its component of `u_k` is zero, and the iteration stops there: on a singular
# Jacobian the result is the solution of minimum length of the least-squares problem in the Krylov
# space, except where rounding makes an entry count as zero before that space holds it, which the
# docstring states as the limit of the rule. The norm `‖T̄_k‖` is estimated from the entries of
# `T̄_k` only, so `‖b‖` takes no part in it. While the estimate `‖T̄_k‖ / min|γ|` of the condition
# number stays below `minres_trancond` and no entry counts as zero, the solution is updated by the
# cheaper MINRES recurrence `x += τ w`, and the first iteration past it moves to the QLP update, as
# Choi, Paige and Saunders (2011, doi:10.1137/100787921) describe.
#
# Every vector of length `n` is in the storage of the iterate, the loop allocates nothing after
# `init_minresqlp_workspace`, and it indexes no entry of a vector, so the same code runs on a
# `JLArray` under `allowscalar(false)`. The method takes a real element type.

"""
    MINRESQLP(; maxiter = 30, forcing = EisenstatWalkerChoice2(), preconditioner = nothing)

The R5 matrix-free linear solver of [`Newton`](@ref) on a symmetric Jacobian: MINRES-QLP (Choi,
Paige and Saunders 2011, doi:10.1137/100787921) on the Hessian–vector product of a
[`StationarityProblem`](@ref) with `E == U`, with the forcing term `forcing`
([`EisenstatWalkerChoice2`](@ref) by default, [`EisenstatWalkerChoice1`](@ref) or
[`FixedForcing`](@ref)) as the relative tolerance of each Newton step's inner solve.

# Fields

- `maxiter::Int32`: the cap on the inner iterations of one Newton step, `≥ 1`; each iteration takes
  one Hessian–vector product. A step that reaches the cap without meeting the tolerance of its
  forcing term counts a step-rule failure.
- `forcing::F`: the rule that sets the relative tolerance ``\\eta_k`` of step `k`.

The solve stores no Krylov basis: the Lanczos process keeps four vectors, and the solution five
more. It solves `J d = -F` to ``\\|F + J d\\| \\le \\eta_k \\|F\\|``. A diagonal entry of the
reduced lower-triangular factor at most ``\\sqrt{\\varepsilon(R)}`` times the norm of the Lanczos
matrix, which a singular `J` gives, counts as zero: the solve then returns the solution of minimum
length of ``\\min \\|F + J d\\|`` over the Krylov space and ends. The norm of the Lanczos matrix
is estimated from its entries only, so the scale of `F` takes no part in that decision. An
incompatible system, where `F` has a component outside the range of a singular `J`, is the limit of
this rule: rounding in the Lanczos process can make an entry count as zero before the Krylov space
holds the solution of minimum length, and the solve then ends early. In `Float32` the result can
then be far from that solution, also where the nonzero eigenvalues are far from zero. A product
that is not finite ends the step with `NONFINITE`, and the iterate stays.

`E == U` compares the index sets as the problem stores them, so `E = (1, 2)` with `U = (2, 1)`
gives a nonsymmetric Jacobian and raises an `ArgumentError`; so do a [`NonlinearProblem`](@ref)
or any other problem, whose Jacobian is not known to be symmetric and which [`GMRES`](@ref)
takes, and a value iterate (a `Number` or a `StaticArray`). The product comes from the `jacobian`
of the `Newton`: a [`JVPFunction`](@ref) on the unknowns `x[U]`, or by default the Hessian–vector
product of the problem's automatic differentiation. `MINRESQLP` takes no preconditioner: its
keyword `preconditioner` accepts only its default `nothing`, and any other value raises an
`ArgumentError` that names [`GMRES`](@ref), which takes one; so does a `preconditioner` other
than `nothing` in a [`Newton`](@ref) with `MINRESQLP`. `maxiter < 1`, a factor reuse, a
[`Refinement`](@ref) and a [`LevenbergMarquardt`](@ref) globalization raise an `ArgumentError`
too. A [`DogLeg`](@ref) takes its Cauchy point from two
Hessian–vector products and its predicted reduction from one more per trial, and
[`Steihaug`](@ref) replaces the inner solve by its truncated conjugate gradients.
"""
struct MINRESQLP{F <: AbstractForcingTerm} <: LinearMethod{Nothing}
    maxiter::Int32
    forcing::F
    # `adapt` rebuilds the method through this constructor, inside a device kernel too.
    function MINRESQLP{F}(maxiter::Int32, forcing::F) where {F <: AbstractForcingTerm}
        maxiter >= Int32(1) || throw(ArgumentError("MINRESQLP maxiter must be ≥ 1"))
        return new{F}(maxiter, forcing)
    end
end

function MINRESQLP(; maxiter::Integer = 30,
        forcing::AbstractForcingTerm = EisenstatWalkerChoice2(), preconditioner = nothing)
    preconditioner === nothing || throw(ArgumentError(
        "MINRESQLP takes no preconditioner: pass preconditioner = nothing, or use GMRES"))
    m = to_int32(maxiter, "MINRESQLP maxiter must fit in an Int32")
    return MINRESQLP{typeof(forcing)}(m, forcing)
end

# The positional form, which `adapt` rebuilds the method through.
@inline function MINRESQLP(maxiter::Int32, forcing::AbstractForcingTerm)
    return MINRESQLP{typeof(forcing)}(maxiter, forcing)
end

Adapt.@adapt_structure MINRESQLP

# The workspace of one MINRES-QLP solve: the solution `x`, the three columns `w`, `wl` and `wl2` of
# `W_k` that still change, the part `xl2` of the solution that no longer changes, the Lanczos
# vectors `r1`, `r2`, `r3` and `v`, and the right-hand side `bb`, all in the storage of the iterate;
# and the counters of a solve, as in `GMRESWorkspace`: the residual evaluations and the products
# of the solve, the inner count of the last Newton step and the largest of the solve, the zero-based
# index of the step, and the `η_k`, `‖F_{k-1}‖` and linear residual of the last step, which the
# forcing term reads. `start!!` sets those three and `finish!!` returns them to `NaN`.
mutable struct MINRESQLPWorkspace{T, R <: Real, V}
    x::V
    w::V
    wl::V
    wl2::V
    xl2::V
    r1::V
    r2::V
    r3::V
    v::V
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

# The workspace of a `MINRESQLP` solve on an iterate like `x`: ten vectors of its length and
# storage.
function init_minresqlp_workspace(::MINRESQLP, x::AbstractVector{T}) where {T}
    R = real(T)
    vs = ntuple(_ -> buffer(x), 10)
    return MINRESQLPWorkspace{T, R, typeof(vs[1])}(vs...,
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), R(NaN), R(NaN), R(NaN))
end

# What one `minresqlp_solve!` reports: the inner iterations, which are its products, the estimate of
# the residual norm `‖b - A x‖`, whether it met the tolerance, and whether it stopped on a diagonal
# entry that counts as zero, where the result is the solution of minimum length.
struct MINRESQLPInfo{R <: Real}
    iterations::Int32
    resnorm::R
    converged::Bool
    singular::Bool
end

# The condition estimate below which the solution follows the MINRES recurrence.
@inline minres_trancond(::Type{R}) where {R} = R(1e7)

# The reflection `[c s; s -c]` that maps `(a, b)` to `(ρ, 0)`, with `ρ = hypot(a, b) ≥ 0`; `c = ±1`
# and `s = 0` where `b = 0`.
@inline function sym_reflection(a::R, b::R) where {R}
    if iszero(b)
        return iszero(a) ? one(R) : sign(a), zero(R), abs(a)
    elseif iszero(a)
        return zero(R), sign(b), abs(b)
    elseif abs(b) > abs(a)
        t = a / b
        s = sign(b) / sqrt(one(R) + t * t)
        return s * t, s, b / s
    else
        t = b / a
        c = sign(a) / sqrt(one(R) + t * t)
        return c, c * t, a / c
    end
end

# Solve the symmetric `A x = b` with MINRES-QLP in the workspace `ws` and return `(ws.x, info)`.
# `op!(y, v)` writes `A v` into `y` and is called once per iteration; `bnorm` is `‖b‖`; the solve
# stops at an estimated residual `‖b - A x‖ ≤ tol · bnorm`, at `maxiter` iterations, where the
# Lanczos vector vanishes (the Krylov space is invariant), or where a diagonal entry of `L_k` counts
# as zero. A vanishing Lanczos vector is one of norm at most `8 n eps(R) ‖T̄_k‖`, a few times the
# rounding of a product of length `n`. A product that is not finite ends the solve with a residual
# of `NaN`.
function minresqlp_solve!(ws::MINRESQLPWorkspace{T, R}, op!, b, bnorm, tol,
        maxiter) where {T, R}
    n = length(b)
    β₁ = R(bnorm)
    atol = R(tol) * β₁
    x, w, wl, wl2, xl2 = ws.x, ws.w, ws.wl, ws.wl2, ws.xl2
    r1, r2, r3, v = ws.r1, ws.r2, ws.r3, ws.v
    fill!(x, zero(T))
    iszero(β₁) && return x, MINRESQLPInfo{R}(Int32(0), zero(R), true, false)
    fill!(w, zero(T))
    fill!(wl, zero(T))
    fill!(wl2, zero(T))
    fill!(xl2, zero(T))
    fill!(r1, zero(T))
    copy!!(r2, b)
    copy!!(r3, b)
    z = zero(R)
    β, βn, ϕ = z, β₁, β₁
    τ, τl, τl2 = z, z, z
    cs, sn, cr1, sr1, cr2, sr2 = -one(R), z, -one(R), z, -one(R), z
    δn, ϵn = z, z
    γ, γl, γl2 = z, z, z
    η, ηl, ηl2 = z, z, z
    ϑ, ϑl, ϑl2 = z, z, z
    ul3, ul2, ul, u = z, z, z, z
    γl_qlp, ϑ_qlp, γ_qlp, ul_qlp, u_qlp = z, z, z, z, z
    Anorm, gmin, gminl, acond = z, z, z, one(R)
    resnorm = β₁
    qlp = false
    singular = false
    k = Int32(0)
    while k < maxiter
        k += one(Int32)
        # the Lanczos step: v_k, its product, and β_{k+1}
        βl, β = β, βn
        v .= r3 ./ β
        op!(r3, v)
        k > one(Int32) && (r3 .-= (β / βl) .* r1)
        α = rdot(r3, v)
        r3 .-= (α / β) .* r2
        copy!!(r1, r2)
        copy!!(r2, r3)
        βn = R(rnorm(r3))
        if !(isfinite(α) && isfinite(βn))
            resnorm = R(NaN)
            break
        end
        # the norm of column k of T̄_k, `(β_k, α_k, β_{k+1})`; `β₁ = ‖b‖` is no entry of T̄_k
        βk = k > one(Int32) ? β : z
        Anorm = max(Anorm, hypot(βk, α, βn))
        # a vanishing Lanczos vector leaves an invariant Krylov space: with `β_{k+1} = 0` the
        # rotated residual below is zero, or the last component counts as zero, and either stops
        βn <= 8 * n * eps(R) * Anorm && (βn = z)
        gtol = sqrt(eps(R)) * Anorm
        # the last left reflection on the new column, then the one that removes β_{k+1}
        δbar = δn
        δ = cs * δbar + sn * α
        ϵ = ϵn
        γbar = sn * δbar - cs * α
        ϵn = sn * βn
        δn = -cs * βn
        δ_minres = δ
        γl2, γl = γl, γ
        cs, sn, γ = sym_reflection(γbar, βn)
        γ_minres = γ
        τl2, τl = τl, τ
        τ = cs * ϕ
        ϕl = ϕ
        ϕ = sn * ϕ
        # the right reflection P_{k-2,k} of the last iteration on the new column, then P_{k-1,k}
        if k > Int32(2)
            ϑl2 = ϑl
            ηl2, ηl = ηl, η
            δt = sr2 * ϑ - cr2 * δ
            ϑl = cr2 * ϑ + sr2 * δ
            δ = δt
            η = sr2 * γ
            γ = -cr2 * γ
        end
        if k > one(Int32)
            cr1, sr1, γl = sym_reflection(γl, δ)
            ϑ = sr1 * γ
            γ = -cr1 * γ
        end
        # the last three components of u_k, of which the two earlier ones are recomputed
        ul4, ul3 = ul3, ul2
        k > Int32(2) && (ul2 = (τl2 - ηl2 * ul4 - ϑl2 * ul3) / γl2)
        # `|γl|` is at least the `|γ|` of the last iteration, which did not count as zero
        k > one(Int32) && (ul = (τl - ηl * ul3 - ϑl * ul2) / γl)
        singular = abs(γ) <= gtol
        u = singular ? z : (τ - η * ul2 - ϑ * ul) / γ
        if !qlp && !singular && acond < minres_trancond(R)
            # the MINRES update of the solution: w_k = (v_k - ε_k w_{k-2} - δ_k w_{k-1}) / γ_k
            copy!!(wl2, wl)
            copy!!(wl, w)
            w .= (v .- ϵ .* wl2 .- δ_minres .* wl) ./ γ_minres
            x .+= τ .* w
        else
            if !qlp
                # the first QLP iteration: the last two columns of W_{k-1} from the MINRES
                # directions, with the reflections of the last iteration, and the part of x that
                # stays; the update below overwrites the third column before it reads it
                qlp = true
                fill!(xl2, zero(T))
                if k > one(Int32)
                    k > Int32(2) && (wl .= γl_qlp .* wl .+ ϑ_qlp .* w)
                    w .= γ_qlp .* w
                    xl2 .= x .- ul_qlp .* wl .- u_qlp .* w
                end
            end
            if k == one(Int32)
                copy!!(wl2, wl)
                wl .= sr1 .* v
                w .= -cr1 .* v
            elseif k == Int32(2)
                copy!!(wl2, wl)
                wl .= cr1 .* w .+ sr1 .* v
                w .= sr1 .* w .- cr1 .* v
            else
                # `v` holds a column of W here; the next iteration sets it from `r3` again
                copy!!(wl2, wl)
                copy!!(wl, w)
                w .= sr2 .* wl2 .- cr2 .* v
                wl2 .= cr2 .* wl2 .+ sr2 .* v
                v .= cr1 .* wl .+ sr1 .* w
                w .= sr1 .* wl .- cr1 .* w
                copy!!(wl, v)
            end
            xl2 .+= ul2 .* wl2
            x .= xl2 .+ ul .* wl .+ u .* w
        end
        # the right reflection P_{k-1,k+1} of the next column, and the values a move to the QLP
        # update reads in the next iteration
        γl_qlp = γl
        cr2, sr2, γl = sym_reflection(γl, ϵn)
        ϑ_qlp, γ_qlp, ul_qlp, u_qlp = ϑ, γ, ul, u
        Anorm = max(Anorm, γl, abs(γ))
        if k == one(Int32)
            gmin = abs(γ)
            gminl = gmin
        else
            gminl2, gminl = gminl, gmin
            gmin = min(gminl2, γl, abs(γ))
        end
        acond = Anorm / gmin
        # a component that counts as zero leaves the residual of the last iteration
        resnorm = singular ? ϕl : ϕ
        (singular || resnorm <= atol) && break
    end
    return x, MINRESQLPInfo{R}(k, resnorm, resnorm <= atol, singular)
end

# The counters of a solve, from the seed of `solve`, as for a `GMRESWorkspace`.
max_inner_iterations(ws::MINRESQLPWorkspace) = ws.max_inner
f_calls(ws::MINRESQLPWorkspace) = ws.f_calls
jvp_calls(ws::MINRESQLPWorkspace) = ws.jvp_calls
