"""
    Refinement(T = Float64; η = √eps(real(T)), maxiter = 5)
    Refinement(maxiter; η = √eps(Float64))
    Refinement(η, maxiter = 5)

Iterative refinement of the linear direction `d` in [`Newton`](@ref) using the existing factor
`fac` and matrix-free Jacobian-vector products [`jvp!!`](@ref) in the working precision `T` of the
problem.

The keyword constructor accepts `η` (or its ASCII alias `eta`) and `maxiter` (or `max_iterations`),
defaulting to `η = √eps(float(real(T)))` and `maxiter = 5`, and the properties `.eta` and
`.max_iterations` alias `.η` and `.maxiter`.

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
    function Refinement{R}(η::Real, maxiter::Integer) where {R <: Real}
        maxiter >= 1 ||
            throw(ArgumentError("Refinement maxiter must be ≥ 1, got $maxiter"))
        (isfinite(η) && zero(η) < η < one(η)) ||
            throw(ArgumentError("Refinement η must be finite and in (0, 1), got $η"))
        return new{R}(R(η), Int32(maxiter))
    end
end

Refinement(η::R, maxiter::Integer = Int32(5)) where {R <: Real} = Refinement{R}(η, maxiter)

function Refinement(maxiter::Integer; η = nothing, eta = nothing)
    η !== nothing && eta !== nothing &&
        throw(ArgumentError("Refinement: pass either η or eta, not both"))
    return Refinement(something(η, eta, sqrt(eps(Float64))), maxiter)
end

function Refinement(::Type{T}, maxiter::Integer; η = nothing, eta = nothing) where {T}
    η !== nothing && eta !== nothing &&
        throw(ArgumentError("Refinement: pass either η or eta, not both"))
    R = float(real(T))
    η_val = R(something(η, eta, sqrt(eps(R))))
    return Refinement{R}(η_val, maxiter)
end

function Refinement(::Type{T} = Float64;
        η = nothing,
        eta = nothing,
        maxiter = nothing,
        max_iterations = nothing) where {T}
    η !== nothing && eta !== nothing &&
        throw(ArgumentError("Refinement: pass either η or eta, not both"))
    maxiter !== nothing && max_iterations !== nothing &&
        throw(ArgumentError("Refinement: pass either maxiter or max_iterations, not both"))
    R = float(real(T))
    η_val = R(something(η, eta, sqrt(eps(R))))
    maxiter_val = something(maxiter, max_iterations, Int32(5))
    return Refinement{R}(η_val, maxiter_val)
end

@inline function Base.getproperty(ref::Refinement, s::Symbol)
    s === :eta && return getfield(ref, :η)
    s === :max_iterations && return getfield(ref, :maxiter)
    return getfield(ref, s)
end

function Base.propertynames(::Refinement, ::Bool = false)
    return (:η, :maxiter, :eta, :max_iterations)
end

Adapt.@adapt_structure Refinement

# Pre-allocated R3 scratch buffers for `refine!!`: `r` holds the linear residual `-Fx - J d` and
# `d_try` holds the correction `δd` and candidate direction `d + δd`. In R1 both slots are `nothing`.
struct RefinementBuffer{B}
    r::B
    d_try::B
end

@inline init_refinement(::NoRefinement, x) = nothing
@inline function init_refinement(::Refinement, x)
    r = buffer(x)
    r === nothing && return nothing
    return RefinementBuffer(r, buffer(x))
end

@inline function refine!!(
        d, ::NoRefinement, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf)
    return d, false
end

@inline function refine!!(
        d, ::NoRefinement, fac, jvpprep, prob::NonlinearProblem, x, Fx, fnorm, p)
    return d, false
end

"""
    refine!!(d, ref, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf = nothing) -> (d, failed::Bool)

Refine the initial linear direction `d` for `J(x) d = -F(x)` using the factor `fac` and the JVP
preparation `jvpprep`. With [`NoRefinement`](@ref), returns `(d, false)` and reads neither `fnorm`
nor `Fx`. With [`Refinement`](@ref), performs up to `ref.maxiter` defect-correction steps in
working precision `T` and returns `(d_best, failed)`, where `failed` is `true` if a step does not
strictly reduce `‖F(x) + J(x) d‖` or if `ref.maxiter` iterations finish above
`max(ref.η, √eps(real(T))) * fnorm`. `fnorm` is `‖F(x)‖` when the caller already holds it and
`nothing` otherwise, in which case this method forms it from `Fx`.
"""
@inline function refine!!(
        d, ref::Refinement, fac, jvpprep, prob, x, Fx, fnorm, p, refbuf = nothing)
    T = eltype(x)
    R = real(T)
    fnorm_R = fnorm === nothing ? R(rnorm(Fx)) : R(fnorm)
    iszero(fnorm_R) && return d, false
    isfinite(fnorm_R) || return d, true
    η_eff = max(R(ref.η), sqrt(eps(R)))
    r_slot = refbuf === nothing ? buffer(x) : refbuf.r
    d_try_slot = refbuf === nothing ? buffer(x) : refbuf.d_try
    Jv = jvp!!(r_slot, jvpprep, prob, x, d, p)
    r = axpby!!(r_slot, -one(T), Fx, -one(T), Jv)
    r_prev = rnorm(r)
    isfinite(r_prev) || return d, true
    r_prev <= η_eff * fnorm_R && return d, false
    iter = Int32(1)
    while iter <= ref.maxiter
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
