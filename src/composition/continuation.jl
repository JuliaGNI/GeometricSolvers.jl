# The `Continuation` method: follows the path of a homotopy `H(x, λ) = 0` from `(x₀, 0)` to
# `λ = 1` by pseudo-arclength continuation, and solves `F(x) = 0` at `λ = 1` with its member.

"""
    NewtonHomotopy()

The Newton homotopy ``H(x, λ) = F(x) - (1 - λ) F(x_0)`` of a [`Continuation`](@ref), whose
path starts at `x₀` and whose `λ = 1` end is a root of `F`. Along the path `F(x)` stays a
multiple of `F(x₀)`.
"""
struct NewtonHomotopy end

"""
    FixedPointHomotopy()

The fixed-point homotopy ``H(x, λ) = λ F(x) / σ + (1 - λ)(x - x_0)`` of a
[`Continuation`](@ref), with `σ` the largest absolute entry of the Jacobian `J(x₀)` of `F` at the
start, or of `F(x₀)` where `J(x₀) = 0`, or `σ = 1` where both are 0. Its path starts at `x₀`,
and its `λ = 1` end is a root of `F`. The division by `σ` makes the path independent of a scale
`s` of `F`: `s F` has the path of `F`. Since `σ` has the units of the Jacobian, the path leaves
the start along `dx/dλ = −F(x₀)/σ`, which near a root of a well-conditioned `J` has the size of
the distance of `x₀` from that root.
"""
struct FixedPointHomotopy end

"""
    ParameterHomotopy(map)

The homotopy ``H(x, λ) = F(x, p_λ)`` of a [`Continuation`](@ref) in a parameter of the user's
problem: `map(p, λ)` returns the parameters `p_λ` at `λ` from the parameters `p` of the solve. The
path starts at `x₀`, which is corrected onto `H(x, 0) = 0` first, and its `λ = 1` end is a root of
`F(x, map(p, 1))`. The derivative ``∂H/∂λ`` is the central difference of step
``\\sqrt[3]{\\varepsilon} \\max(1, |λ|)``, with `ε` the machine epsilon of the iterate.
"""
struct ParameterHomotopy{F}
    map::F
end

Adapt.@adapt_structure ParameterHomotopy

const Homotopy = Union{NewtonHomotopy, FixedPointHomotopy, ParameterHomotopy}

"""
    Continuation(homotopy; method = Newton(), h₀ = 0.1, h_min = 1e-6, h_max = Inf)

Whole-solve method that follows the path of `homotopy` ([`NewtonHomotopy`](@ref),
[`FixedPointHomotopy`](@ref) or [`ParameterHomotopy`](@ref)) from `(x₀, λ = 0)` to `λ = 1` by
pseudo-arclength continuation, and then solves `F(x) = 0` with `method` at `λ = 1`.

The path is followed in the unknowns `z = (x, λ)`. The tangent `t` solves
``[H_x \\; H_λ] t = 0`` with ``\\|t\\|_2 = 1``, oriented along the tangent before it (`λ`
increasing at the start). A step of length `h` predicts `z + h t`, and the bordered Newton
corrector solves

```math
\\begin{bmatrix} H_x & H_λ \\\\ t^\\top \\end{bmatrix} Δz
= -\\begin{bmatrix} H(z) \\\\ t^\\top (z - z_{\\text{pred}}) \\end{bmatrix}
```

for at most 6 iterations, until ``\\|Δz\\|_2 \\le \\sqrt{\\varepsilon} \\max(1, \\|z\\|_2)``. The
corrector fails on a singular or non-finite system, after six iterations, on a corrected point
farther than `h` from the prediction, and on a corrected `λ < 0`, where the path has turned back
to its start. A failed corrector, or a tangent at the corrected point whose bordered system is
singular or not finite, rejects the step and halves `h`; a step whose corrector converges in at
most 3 iterations is accepted and doubles `h` up to `h_max`, and any other accepted step keeps
`h`. The path passes a fold of `λ`, where ``H_x`` is singular, because the bordered matrix is
not. Once an accepted step reaches `λ ≥ 1`, `x` is interpolated linearly to `λ = 1` between that
point and the one before it, and [`solve!!`](@ref) with `method` from there, on the problem and
the parameters at `λ = 1`, is the last corrector: its iterate is the result.

The [`SolverStatus`](@ref) is that of the last corrector, with `iterations` the number of path
steps, accepted and rejected, plus the corrector's iterations, and `step_failures` the rejected
steps plus the corrector's (each saturated at `typemax(Int32)`). A homotopy residual at
`(x₀, 0)` that is not finite returns `NONFINITE` with no step taken; for the Newton and the
fixed-point homotopy, a non-finite `F(x₀)` makes it so, and so does a non-finite scale `σ` of the
fixed-point homotopy, which a non-finite `J(x₀)` gives. A start that the corrector cannot bring
onto the path at `λ = 0` returns `STALLED` with no step; so does a start whose bordered system at
`(x₀, 0)` along `e_λ` is singular, such as a fold of `λ` at the start of the Newton or the
parameter homotopy, where `H_x(x₀, 0)` is singular. The `fnorm` of these returns before the first
step is the norm of `F` at `x₀` and `λ = 1`; a parameter homotopy evaluates `F` there for them
alone, and its path does not. A halved `h` below `h_min` returns `STALLED`,
and `max_iterations` path steps of the [`Options`](@ref) return `MAXITERS`; both return the last
accepted point of the path, with `fnorm` the norm of `F` there at `λ = 1`, `stepnorm` the length
of the last step, and `step_failures` the rejected steps. [`continuation_steps`](@ref) reads `λ`
and the step length of every step of the last solve.

`method` is a [`Newton`](@ref) whose step forms the Jacobian matrix: its Jacobian source gives
``H_x``, and a matrix-free linear method ([`matrix_free`](@ref)), such as [`GMRES`](@ref) or
[`MINRESQLP`](@ref), or a [`KroneckerFactorization`](@ref) raises an `ArgumentError`. The step
lengths satisfy `0 < h_min ≤ h₀ ≤ h_max`, with `h_min` and `h₀` finite, else the constructor
raises an `ArgumentError`; [`ToReal`](@ref) converts them to the real type of the iterate. The
path and its bordered systems are dense, in `Vector` and `Matrix` storage of the iterate's
element type, so a solve allocates; the iterate is real and on the host, and a complex or a
device array iterate raises an `ArgumentError` in `init_state`.
Like every combinator, `Continuation` refuses a `ForwardDiff.Dual` in `x₀` or `p` in
[`solve`](@ref) and a [`LeastSquaresProblem`](@ref) in `init_state`.
"""
struct Continuation{H <: Homotopy, M, R <: AbstractFloat}
    homotopy::H
    method::M
    h₀::R
    h_min::R
    h_max::R
    # `adapt` rebuilds a `Continuation` through this constructor, so the check also refuses a step
    # length that `ToReal` narrows to `Inf` or to zero.
    function Continuation{H, M, R}(homotopy::H, method::M, h₀::R, h_min::R,
            h_max::R) where {H <: Homotopy, M, R <: AbstractFloat}
        method isa Newton &&
        !(matrix_free(method.linsolve) || method.linsolve isa KroneckerFactorization) ||
            throw(ArgumentError("Continuation needs a Newton method whose step forms the Jacobian matrix: not a matrix-free linear method or KroneckerFactorization"))
        # a finite `h₀` bounds `h_min`, and a `NaN` fails every comparison
        isfinite(h₀) && zero(R) < h_min <= h₀ <= h_max ||
            throw(ArgumentError("Continuation requires finite step lengths 0 < h_min ≤ h₀ ≤ h_max"))
        return new{H, M, R}(homotopy, method, h₀, h_min, h_max)
    end
end

function Continuation(homotopy::H, method::M, h₀::R, h_min::R, h_max::R) where {H, M, R}
    return Continuation{H, M, R}(homotopy, method, h₀, h_min, h_max)
end

function Continuation(homotopy; method = Newton(), h₀::Real = 0.1, h_min::Real = 1e-6,
        h_max::Real = Inf)
    h₀f, h_minf, h_maxf = promote(float(h₀), float(h_min), float(h_max))
    return Continuation(homotopy, method, h₀f, h_minf, h_maxf)
end

Adapt.@adapt_structure Continuation

# The parameters of the problem at `λ`.
@inline homotopy_parameters(::Union{NewtonHomotopy, FixedPointHomotopy}, p, λ) = p
@inline homotopy_parameters(h::ParameterHomotopy, p, λ) = h.map(p, λ)

"""
    homotopy_scale(homotopy, J, x₀, F₀, p)

The scale `σ` of `homotopy` at the start `x₀`, with `J(x, q)` the Jacobian of the problem and
`F₀` the vector of `F(x₀, p)`: for the [`FixedPointHomotopy`](@ref) the largest absolute entry of
`J(x₀, p)`, or of `F₀` where that is 0, or 1 where both are 0; for the other homotopies 1, which
they do not read. A non-finite `J(x₀, p)` gives a non-finite `σ`.
"""
homotopy_scale(::Union{NewtonHomotopy, ParameterHomotopy}, J, x₀, F₀, p) = one(eltype(F₀))

# The largest absolute entry cannot overflow for a finite `J(x₀, p)` or `F₀`, and `maximum`
# returns `NaN` where an entry is `NaN`.
function homotopy_scale(::FixedPointHomotopy, J, x₀, F₀, p)
    σ = maximum(abs, J(x₀, p))
    iszero(σ) || return σ
    σ = maximum(abs, F₀)
    return iszero(σ) ? one(σ) : σ
end

# Write `F(x₀, p)` into `F₀` for the Newton and the fixed-point homotopy, whose residual and
# Jacobian read it. A parameter homotopy reads only the length of `F₀`, and `F` is not evaluated.
function start_residual!(F₀, ::Union{NewtonHomotopy, FixedPointHomotopy}, F, x₀, p)
    F₁ = F(x₀, p)
    for i in eachindex(F₀)
        F₀[i] = F₁[i]
    end
    return F₀
end
start_residual!(F₀, ::ParameterHomotopy, F, x₀, p) = F₀

# The norm of `F(x₀, p₁)` that a return before the first step reports: that of `F₀` for the Newton
# and the fixed-point homotopy, whose `p₁` is `p`, and of a new evaluation for a parameter homotopy.
start_fnorm(::Union{NewtonHomotopy, FixedPointHomotopy}, F, x₀, p₁, F₀) = rnorm(F₀)
start_fnorm(::ParameterHomotopy, F, x₀, p₁, F₀) = rnorm(F(x₀, p₁))

"""
    homotopy_residual!(H, homotopy, F, x, λ, x₀, F₀, σ, p)

Write the residual `H(x, λ)` of `homotopy` into the vector `H` and return it. `F(x, q)` is the
residual of the problem with parameters `q`, `x₀` the start of the path, `F₀` the vector of
`F(x₀, p)` and `σ` the scale of [`homotopy_scale`](@ref).
"""
function homotopy_residual!(H, ::NewtonHomotopy, F, x, λ, x₀, F₀, σ, p)
    Fx = F(x, p)
    for i in eachindex(H)
        H[i] = Fx[i] - (1 - λ) * F₀[i]
    end
    return H
end

# `F` is divided by `σ` before `λ` multiplies it, so that no product of an unscaled `F` with `λ`
# can overflow.
function homotopy_residual!(H, ::FixedPointHomotopy, F, x, λ, x₀, F₀, σ, p)
    Fx = F(x, p)
    for i in eachindex(H)
        H[i] = λ * (Fx[i] / σ) + (1 - λ) * (x[i] - x₀[i])
    end
    return H
end

function homotopy_residual!(H, h::ParameterHomotopy, F, x, λ, x₀, F₀, σ, p)
    Fx = F(x, h.map(p, λ))
    for i in eachindex(H)
        H[i] = Fx[i]
    end
    return H
end

"""
    homotopy_jacobian!(A, homotopy, F, J, x, λ, x₀, F₀, σ, p)

Write the `n × (n + 1)` Jacobian ``[H_x \\; H_λ]`` of `homotopy` at `(x, λ)` into the first `n`
rows of `A` and return `A`. `J(x, q)` is the Jacobian of the problem with parameters `q`, and the
other arguments are those of [`homotopy_residual!`](@ref).
"""
function homotopy_jacobian!(A, ::NewtonHomotopy, F, J, x, λ, x₀, F₀, σ, p)
    Jx = J(x, p)
    n = length(F₀)
    for j in 1:n, i in 1:n

        A[i, j] = Jx[i, j]
    end
    for i in 1:n
        A[i, n + 1] = F₀[i]
    end
    return A
end

function homotopy_jacobian!(A, ::FixedPointHomotopy, F, J, x, λ, x₀, F₀, σ, p)
    Jx = J(x, p)
    n = length(F₀)
    for j in 1:n, i in 1:n

        A[i, j] = λ * (Jx[i, j] / σ) + (i == j ? 1 - λ : zero(λ))
    end
    Fx = F(x, p)
    for i in 1:n
        A[i, n + 1] = Fx[i] / σ - (x[i] - x₀[i])
    end
    return A
end

function homotopy_jacobian!(A, h::ParameterHomotopy, F, J, x, λ, x₀, F₀, σ, p)
    Jx = J(x, h.map(p, λ))
    n = length(F₀)
    for j in 1:n, i in 1:n

        A[i, j] = Jx[i, j]
    end
    # `F` may return the one buffer it writes into, so the forward value is stored before the
    # backward one is evaluated
    ε = cbrt(eps(typeof(λ))) * max(one(λ), abs(λ))
    F₊ = F(x, h.map(p, λ + ε))
    for i in 1:n
        A[i, n + 1] = F₊[i]
    end
    F₋ = F(x, h.map(p, λ - ε))
    for i in 1:n
        A[i, n + 1] = (A[i, n + 1] - F₋[i]) / (2ε)
    end
    return A
end

# State of `Continuation`: the state of its member on the problem, the copy of an array `x₀` and
# the array the path's iterates are written into (`nothing` for a value iterate), the residual
# buffer, the dense workspace of the path, and the record of its steps.
struct ContinuationState{S, X, FB, W, RC}
    member::S
    x₀::X
    xw::X
    fbuf::FB
    ws::W
    record::RC
end

"""
    continuation_steps(state)
    continuation_steps(solver)

The record of the path steps of the last [`solve!!`](@ref) of a [`Continuation`](@ref) on
`state`, or on the state of the [`NonlinearSolver`](@ref) `solver`: a named tuple of three
vectors of equal length, one entry per step in the order of the steps. `λ` is the parameter of the
corrected point of an accepted step and of the predicted point of a rejected one, `h` the step
length, and `accepted` whether the step was accepted. The vectors are the state's own, and the next
solve empties and fills them again.
"""
continuation_steps(state::ContinuationState) = state.record
# the state of a `StationarityProblem` solve holds the method's state as `inner`
function continuation_steps(state::Union{StaticStationarityState, DenseStationarityState})
    continuation_steps(state.inner)
end
continuation_steps(solver::NonlinearSolver) = continuation_steps(solver.state)

@inline function init_state(m::Continuation, prob::CombinatorProblem, x, p)
    x isa AbstractGPUArray && throw(ArgumentError(
        "Continuation runs on a host array: its path and bordered systems are dense host arrays; solve on an Array"))
    T = eltype(x)
    T <: Real || throw(ArgumentError(
        "Continuation takes a real iterate: its path parameter λ is real; got element type $T"))
    R = real(T)
    n = length(x)
    member = init_state(m.method, prob, x, homotopy_parameters(m.homotopy, p, one(R)))
    ws = (;
        F₀ = zeros(T, n), σ = Ref(one(T)), z = zeros(T, n + 1), w = zeros(T, n + 1),
        zp = zeros(T, n + 1), t = zeros(T, n + 1), r = zeros(T, n + 1),
        A = zeros(T, n + 1, n + 1))
    record = (; λ = R[], h = R[], accepted = Bool[])
    return ContinuationState(member, buffer(x), buffer(x), buffer(x), ws, record)
end

# The member runs the last corrector as a solve of its own, which reads its own rates;
# `refactorize` drops its factor.
@inline function drop_factor!!(state::ContinuationState, m::Continuation)
    return ContinuationState(drop_factor!!(state.member, m.method), state.x₀, state.xw,
        state.fbuf, state.ws, state.record)
end

# The iterate of the first `n` entries of `z`, in the storage of `x`: written into the array `xw`,
# or a new value.
@inline path_iterate(xw::AbstractArray, x, z, n) = copyto!(xw, 1, z, 1, n)
@inline path_iterate(::Nothing, x::Number, z, n) = typeof(x)(z[1])
@inline path_iterate(::Nothing, x::StaticArray, z, n) = typeof(x)(ntuple(i -> z[i], n))

# Solve `A y = r` in place in `r` from the dense `A`, which it overwrites. `false` where `A` is not
# finite, whose LU factor can give a finite `y`, where the LU factor has a zero pivot, on which the
# generic back substitution of a `BigFloat` factor throws `SingularException`, and where `y` is not
# finite, which a non-finite `r` gives.
function bordered_solve!(A, r)
    all(isfinite, A) || return false
    fac = lu!(A; check = false)
    issuccess(fac) || return false
    ldiv!(fac, r)
    return all(isfinite, r)
end

# The unit tangent at the point `z` of the path, oriented along `t` before it, written into `t`.
# `false`, with `t` unchanged, where the bordered system is singular or not finite.
function path_tangent!(ws, z, h, F, J, xw, x, x₀, p, n)
    xz = path_iterate(xw, x, z, n)
    homotopy_jacobian!(ws.A, h, F, J, xz, z[n + 1], x₀, ws.F₀, ws.σ[], p)
    for j in 1:(n + 1)
        ws.A[n + 1, j] = ws.t[j]
    end
    fill!(ws.r, zero(eltype(ws.r)))
    ws.r[n + 1] = one(eltype(ws.r))
    bordered_solve!(ws.A, ws.r) || return false
    # the last row gives `tᵀ y = 1`, so `y ≠ 0`; its norm is finite unless it overflows
    τ = rnorm(ws.r)
    isfinite(τ) || return false
    ws.t .= ws.r ./ τ
    return true
end

# The bordered Newton corrector from the prediction `zp` along the tangent `t`, written into `w`.
# It returns whether it converged and the number of its iterations.
function path_corrector!(ws, h, F, J, xw, x, x₀, p, n)
    T = eltype(ws.w)
    ws.w .= ws.zp
    for k in 1:6
        xk = path_iterate(xw, x, ws.w, n)
        λ = ws.w[n + 1]
        homotopy_residual!(view(ws.r, 1:n), h, F, xk, λ, x₀, ws.F₀, ws.σ[], p)
        ws.r[n + 1] = zero(T)
        for j in 1:(n + 1)
            ws.r[n + 1] += ws.t[j] * (ws.w[j] - ws.zp[j])
        end
        homotopy_jacobian!(ws.A, h, F, J, xk, λ, x₀, ws.F₀, ws.σ[], p)
        for j in 1:(n + 1)
            ws.A[n + 1, j] = ws.t[j]
        end
        bordered_solve!(ws.A, ws.r) || return false, k
        Δ = rnorm(ws.r)
        ws.w .-= ws.r
        Δ <= sqrt(eps(T)) * max(one(T), rnorm(ws.w)) && return true, k
    end
    return false, 6
end

# The status of a solve that ends on the path, before `λ = 1`.
@inline function path_status(code, steps, fnorm::R, h, rejected) where {R}
    return SolverStatus{R}(
        code, Int32(min(steps, typemax(Int32))), fnorm, R(h),
        Int32(min(rejected, typemax(Int32))), false, false)
end

# `prob` is a `CombinatorProblem`, as in `init_state`, so that this method is not ambiguous with
# `solve!!(state, ::LeastSquaresProblem, x₀::Number, method, …)`.
function solve!!(
        state::ContinuationState, prob::CombinatorProblem, x₀, method::Continuation,
        p = nothing, options::Options{R} = Options(eltype(x₀))) where {R <: Real}
    check_options(eltype(x₀), options)
    T = eltype(x₀)
    n = length(x₀)
    ws = state.ws
    rec = state.record
    foreach(empty!, rec)
    h = method.homotopy
    p₁ = homotopy_parameters(h, p, one(R))
    F = (x, q) -> primal_residual!!(state.fbuf, prob, x, q)
    J = (x, q) -> jacobian!!(state.member.J, state.member.jacprep, prob, x, q)
    xs = copy!!(state.x₀, x₀)
    xw = state.xw
    # the start: `F₀ = F(x₀, p)` and the scale of the Newton and the fixed-point homotopy, and the
    # homotopy residual at `λ = 0`, which a non-finite `F(x₀, p)` makes non-finite for those two
    # (`0 · Inf` is `NaN`). A parameter homotopy reads neither `F₀` nor a scale, so its path does
    # not evaluate `F` at `(x₀, p₁)`, where `F` need not be defined.
    start_residual!(ws.F₀, h, F, x₀, p)
    ws.σ[] = homotopy_scale(h, J, x₀, ws.F₀, p)
    if !isfinite(ws.σ[]) ||
       !all(isfinite, homotopy_residual!(
        view(ws.r, 1:n), h, F, x₀, zero(T), xs, ws.F₀, ws.σ[], p))
        f₁ = R(start_fnorm(h, F, x₀, p₁, ws.F₀))
        return x₀, path_status(NONFINITE, 0, f₁, zero(R), 0), state
    end
    # the start on the path: the corrector at `λ = 0`, along `t = e_λ`, which a start on the path
    # meets at its first iteration
    for i in 1:n
        ws.z[i] = x₀[i]
    end
    ws.z[n + 1] = zero(T)
    ws.zp .= ws.z
    fill!(ws.t, zero(T))
    ws.t[n + 1] = one(T)
    ok, _ = path_corrector!(ws, h, F, J, xw, x₀, xs, p, n)
    if !(ok && path_tangent!(ws, ws.w, h, F, J, xw, x₀, xs, p, n))
        f₁ = R(start_fnorm(h, F, x₀, p₁, ws.F₀))
        return x₀, path_status(STALLED, 0, f₁, zero(R), 0), state
    end
    ws.z .= ws.w
    step = T(method.h₀)
    steps = 0
    rejected = 0
    while true
        if steps >= options.max_iterations
            x = copy!!(x₀, path_iterate(xw, x₀, ws.z, n))
            last_step = isempty(rec.h) ? zero(R) : rec.h[end]
            status = path_status(MAXITERS, steps, R(rnorm(F(x, p₁))), last_step, rejected)
            return x, status, state
        end
        steps += 1
        ws.zp .= ws.z .+ step .* ws.t
        ok, k = path_corrector!(ws, h, F, J, xw, x₀, xs, p, n)
        # the distance from the prediction, in `r`, which the tangent and the next corrector
        # overwrite before they read it
        ok && (ws.r .= ws.w .- ws.zp)
        ok = ok && rnorm(ws.r) <= step && ws.w[n + 1] >= 0
        # the step ends the path at `λ ≥ 1`, or the tangent at its point continues it
        ok = ok && (ws.w[n + 1] >= 1 || path_tangent!(ws, ws.w, h, F, J, xw, x₀, xs, p, n))
        if !ok
            push!(rec.λ, ws.zp[n + 1])
            push!(rec.h, step)
            push!(rec.accepted, false)
            rejected += 1
            step /= 2
            if step < method.h_min
                x = copy!!(x₀, path_iterate(xw, x₀, ws.z, n))
                status = path_status(STALLED, steps, R(rnorm(F(x, p₁))), 2step, rejected)
                return x, status, state
            end
            continue
        end
        push!(rec.λ, ws.w[n + 1])
        push!(rec.h, step)
        push!(rec.accepted, true)
        if ws.w[n + 1] >= 1
            # the last corrector, at `λ = 1`, from the point of the chord between the last two
            # points of the path
            s = (1 - ws.z[n + 1]) / (ws.w[n + 1] - ws.z[n + 1])
            ws.zp .= ws.z .+ s .* (ws.w .- ws.z)
            x₁ = copy!!(x₀, path_iterate(xw, x₀, ws.zp, n))
            x, st, member = solve!!(state.member, prob, x₁, method.method, p₁, options)
            status = SolverStatus{R}(
                st.code, st.f_calls, _sat_add32(st.iterations, Int32(min(steps, typemax(Int32)))),
                st.fnorm, st.stepnorm,
                _sat_add32(st.step_failures, Int32(min(rejected, typemax(Int32)))),
                st.promoted, st.stationary, st.jvp_calls, st.rank, st.nullnorm, st.gradratio,
                st.rate, st.contracted, st.reused)
            state₊ = ContinuationState(member, state.x₀, xw, state.fbuf, ws, rec)
            return x, status, state₊
        end
        ws.z .= ws.w
        step = next_step_length(step, k, T(method.h_max))
    end
end

# The step length after an accepted step whose corrector took `k` iterations: doubled up to
# `h_max` after a fast corrector, of at most 3 iterations, and kept after a slower one.
@inline next_step_length(step, k, h_max) = k <= 3 ? min(2step, h_max) : step
