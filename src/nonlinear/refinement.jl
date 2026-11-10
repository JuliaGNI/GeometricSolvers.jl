"""
    Refinement(T = Float64; η = √eps(float(real(T))), maxiter = 5)
    Refinement(η, maxiter = 5)

Iterative refinement of the linear direction `d` in [`Newton`](@ref) using the existing factor
`fac` and Jacobian-vector products in the working precision `T` of the problem: a step that
factorised at `x` multiplies by the regularised Jacobian it holds, and a step along a reused
factor takes [`jvp!!`](@ref) at `x`.

The keyword constructor takes the real type `T` of `η`, and the positional one the type of `η`.
With `gmres_maxiter` or `gmres_tol`, the keyword constructor returns a
[`GMRESRefinement`](@ref).

At each inner pass (`1 ≤ iter ≤ maxiter`), `refine!!` forms the linear residual
`r = -F(x) - J(x) d` in `T`, solves `fac \\ r` in the factor precision `TF` for the correction
`δd`, updates `d ← d + δd`, and stops as soon as

```math
\\|F(x) + J(x) d\\| \\le \\max(\\eta,\\, \\sqrt{\\varepsilon(R)})\\, \\|F(x)\\|
```

for `R = real(T)`. If a pass fails to strictly decrease `‖F(x) + J(x) d‖` (`!(r_curr < r_prev)`)
or if `maxiter` passes finish above the threshold, `refine!!` returns the best direction reached
together with `failed = true`.

Both `maxiter ≥ 1` (`Int32`) and `0 < η < 1` (`R <: Real`) are validated at construction.
`Refinement{R}` is `isbits`, and [`ToReal`](@ref) adapts `η` to the real type of the iterate.
"""
struct Refinement{R <: Real}
    η::R
    maxiter::Int32
    # `adapt` rebuilds a `Refinement` through this constructor, inside a device kernel too. A
    # kernel cannot format a message, so the messages are constant.
    function Refinement{R}(η::Real, maxiter::Integer) where {R <: Real}
        maxiter >= 1 ||
            throw(ArgumentError("Refinement maxiter must be ≥ 1"))
        (isfinite(η) && zero(η) < η < one(η)) ||
            throw(ArgumentError("Refinement η must be finite and in (0, 1)"))
        return new{R}(R(η), Int32(maxiter))
    end
end

Refinement(η::R, maxiter::Integer = Int32(5)) where {R <: Real} = Refinement{R}(η, maxiter)

function Refinement(::Type{T} = Float64; η = sqrt(eps(float(real(T)))),
        maxiter::Integer = Int32(5), gmres_maxiter::Union{Nothing, Integer} = nothing,
        gmres_tol::Union{Nothing, Real} = nothing) where {T}
    R = float(real(T))
    if gmres_maxiter !== nothing || gmres_tol !== nothing
        return GMRESRefinement{R}(R(η), maxiter,
            gmres_maxiter === nothing ? GMRES_MAXITER : gmres_maxiter,
            gmres_tol === nothing ? GMRES_TOL : gmres_tol)
    end
    return Refinement{R}(R(η), maxiter)
end

Adapt.@adapt_structure Refinement

# The default cap of the inner GMRES iterations of one correction, which the solve lowers to the
# number of unknowns, and the default relative tolerance of the inner solve: the `1e-6` published
# for a double working precision (Carson and Higham 2018, p. A836).
const GMRES_MAXITER = Int32(30)
const GMRES_TOL = 1e-6

"""
    GMRESRefinement(T = Float64; η = √eps(float(real(T))), maxiter = 5, gmres_maxiter = 30,
                    gmres_tol = 1e-6)
    GMRESRefinement(η, maxiter = 5, gmres_maxiter = 30, gmres_tol = 1e-6)

The mixed-precision iterative refinement of [`Newton`](@ref) of Amestoy et al. 2024, their variant
`SDD`, in the five precisions `(u_f, u, u_p, u_g, u_r) = (Float32, Float64, Float64, Float64,
Float64)`: the factorisation is in a factorisation precision `TF` below the working precision `T`
of the iterate, and the linear direction of a step is corrected in `T` first by
[`Refinement`](@ref) passes with the low-precision factors, then, where those fail, by a
preconditioned GMRES in `T` on a copy of the same factors promoted to `T` (Kelley 2023, p. 7;
Higham and Mary 2022, pp. 374–375).

The keyword constructor takes the real type `T` of `η` and of `gmres_tol`, and the positional one
the types of its arguments. Both `maxiter ≥ 1` and `gmres_maxiter ≥ 1` (`Int32`) and
`0 < η < 1`, `0 < gmres_tol < 1` (`R <: Real`) are validated at construction.

# Fields

- `η::R`, `maxiter::Int32`: the tolerance and the pass count of the LU-IR passes, as in
  [`Refinement`](@ref).
- `gmres_maxiter::Int32`: the cap on the inner GMRES iterations of one correction, lowered to the
  number of unknowns of the solve. A correction that reaches the cap without meeting its tolerance
  is a refinement failure.
- `gmres_tol::R`: the relative tolerance of the inner GMRES, on the true linear residual
  ``\\|-F(x) - J d\\|_2`` of the direction it returns, which the stage recomputes in `T` and
  accepts up to ten times, so that the difference between the Arnoldi estimate and the true
  residual cannot pass a correction that the estimate alone would.

A `Newton` step whose factorisation is already in `T` runs the LU-IR passes alone: with no
low-precision factor there is no GMRES-IR stage, and the refinement is then a [`Refinement`](@ref).
A value iterate (`Number` or `StaticArray`) has no Krylov basis, so it too runs the LU-IR passes
alone, and a step that fails them refactorises in `T` as with a [`Refinement`](@ref).
`Refinement(; gmres_maxiter = k, gmres_tol = t)` builds a `GMRESRefinement` from either keyword.
"""
struct GMRESRefinement{R <: Real}
    η::R
    maxiter::Int32
    gmres_maxiter::Int32
    gmres_tol::R
    # `adapt` rebuilds a `GMRESRefinement` through this constructor, inside a device kernel too. A
    # kernel cannot format a message, so the messages are constant.
    function GMRESRefinement{R}(
            η::Real, maxiter::Integer, gmres_maxiter::Integer,
            gmres_tol::Real) where {R <: Real}
        maxiter >= 1 ||
            throw(ArgumentError("GMRESRefinement maxiter must be ≥ 1"))
        gmres_maxiter >= 1 ||
            throw(ArgumentError("GMRESRefinement gmres_maxiter must be ≥ 1"))
        (isfinite(η) && zero(η) < η < one(η)) ||
            throw(ArgumentError("GMRESRefinement η must be finite and in (0, 1)"))
        (isfinite(gmres_tol) && zero(gmres_tol) < gmres_tol < one(gmres_tol)) ||
            throw(ArgumentError("GMRESRefinement gmres_tol must be finite and in (0, 1)"))
        return new{R}(R(η), Int32(maxiter), Int32(gmres_maxiter), R(gmres_tol))
    end
end

function GMRESRefinement(η::R, maxiter::Integer = Int32(5),
        gmres_maxiter::Integer = GMRES_MAXITER,
        gmres_tol::Real = GMRES_TOL) where {R <: Real}
    return GMRESRefinement{R}(η, maxiter, gmres_maxiter, gmres_tol)
end

function GMRESRefinement(::Type{T} = Float64; η = sqrt(eps(float(real(T)))),
        maxiter::Integer = Int32(5), gmres_maxiter::Integer = GMRES_MAXITER,
        gmres_tol::Real = GMRES_TOL) where {T}
    R = float(real(T))
    return GMRESRefinement{R}(R(η), maxiter, gmres_maxiter, gmres_tol)
end

Adapt.@adapt_structure GMRESRefinement

# Whether a refinement corrects the direction of the system that the factor solves, which a damped
# `LevenbergMarquardt` step and a matrix-free `GMRES` step take no method of. A refinement that this
# package does not define refines nothing here, so a caller's own type keeps the paths of
# `NoRefinement`, as it did where the step read `refinement isa Refinement`.
@inline is_refining(::Union{Refinement, GMRESRefinement}) = true
@inline is_refining(refinement) = false

# Pre-allocated R3 scratch buffers for `refine!!`: `r` holds the linear residual `-Fx - J d` and
# `d_try` holds the correction `δd` and candidate direction `d + δd`. In R1 both slots are `nothing`.
struct RefinementBuffer{B}
    r::B
    d_try::B
end

# The workspace of a `GMRESRefinement`, added to those two slots: `b` the preconditioned
# right-hand side `M̂⁻¹ r` of the inner solve, `tmp` a product slot, `z` the preconditioned product
# `M̂⁻¹ (J v)` that the operator of the inner solve writes, and `ws` the Krylov workspace of
# [`gmres_solve!`](@ref). A value iterate has no storage for any of them and keeps `nothing`.
struct GMRESIRBuffer{B, W}
    r::B
    d_try::B
    b::B
    tmp::B
    z::B
    ws::W
end

@inline init_refinement(::NoRefinement, x) = nothing
@inline function init_refinement(::Refinement, x)
    r = buffer(x)
    r === nothing && return nothing
    return RefinementBuffer(r, buffer(x))
end

@inline function init_refinement(r::GMRESRefinement, x::AbstractVector)
    r_slot = buffer(x)
    r_slot === nothing && return nothing
    return GMRESIRBuffer(r_slot, buffer(x), buffer(x), buffer(x), buffer(x),
        init_gmres_ir_workspace(r, x))
end

# A value iterate has no buffer and no Krylov basis, so it keeps the LU-IR passes of a
# `Refinement` and, where they fail, its refactorisation in `T`.
@inline init_refinement(::GMRESRefinement, x) = nothing

# The Krylov workspace of the GMRES-IR stage: the basis of `min(gmres_maxiter, n)` columns in the
# working precision `T`, unrestarted, which is the widest basis that one correction forms.
@inline function init_gmres_ir_workspace(r::GMRESRefinement, x::AbstractVector)
    m = min(Int(r.gmres_maxiter), length(x))
    return init_gmres_workspace(GMRES(; maxiter = m, restart = m), x)
end

@inline function refine!!(
        d, ::NoRefinement, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf = nothing)
    return d, false
end

"""
    refine!!(d, ref, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf = nothing) -> (d, failed::Bool)

Refine the initial linear direction `d` for `J(x) d = -F(x)` using the factor `fac` and the JVP
preparation `jvpprep`. With [`NoRefinement`](@ref), returns `(d, false)` and reads neither `fnorm`
nor `Fx`. With [`Refinement`](@ref) or the LU-IR passes of [`GMRESRefinement`](@ref), performs up
to `ref.maxiter` defect-correction steps in working precision `T` and returns `(d_best, failed)`,
where `failed` is `true` if a step does not strictly reduce `‖F(x) + J(x) d‖` or if `ref.maxiter`
iterations finish above `max(ref.η, √eps(real(T))) * fnorm`; the GMRES-IR stage of a
[`GMRESRefinement`](@ref) follows a failed pass in the step, not here. `fnorm` is `‖F(x)‖` when the
caller already holds it and `nothing` otherwise, in which case this method forms it from `Fx`.
"""
@inline function refine!!(
        d, ref::Refinement, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf = nothing)
    return refine_passes!!(
        d, ref.η, ref.maxiter, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf)
end

@inline function refine!!(
        d, ref::GMRESRefinement, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf = nothing)
    return refine_passes!!(
        d, ref.η, ref.maxiter, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf)
end

# The `maxiter` LU-IR passes of a refinement, at the tolerance `η`: the shared body of the
# `Refinement` and the stage-1 method of the `GMRESRefinement`.
@inline function refine_passes!!(
        d, η, maxiter::Int32, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf)
    T = eltype(x)
    R = real(T)
    fnorm_R = fnorm === nothing ? R(rnorm(Fx)) : R(fnorm)
    iszero(fnorm_R) && return d, false
    isfinite(fnorm_R) || return d, true
    η_eff = max(R(η), sqrt(eps(R)))
    r_slot = refbuf === nothing ? buffer(x) : refbuf.r
    d_try_slot = refbuf === nothing ? buffer(x) : refbuf.d_try
    Jv = jvp!!(r_slot, jvpprep, prob, x, d, p)
    r = axpby!!(r_slot, -one(T), Fx, -one(T), Jv)
    r_prev = rnorm(r)
    isfinite(r_prev) || return d, true
    r_prev <= η_eff * fnorm_R && return d, false
    iter = Int32(1)
    while iter <= maxiter
        δd = ldiv!!(d_try_slot, fac, r)
        d_cand = axpby!!(d_try_slot, one(T), d, one(T), δd)
        Jv = jvp!!(r_slot, jvpprep, prob, x, d_cand, p)
        r = axpby!!(r_slot, -one(T), Fx, -one(T), Jv)
        r_curr = rnorm(r)
        if !(r_curr < r_prev)
            return d, true
        end
        d = copy!!(d, d_cand)
        r_prev = r_curr
        if r_curr <= η_eff * fnorm_R
            return d, false
        end
        iter += one(Int32)
    end
    return d, true
end
