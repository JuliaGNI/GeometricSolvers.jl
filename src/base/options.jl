"""
    StoppingNorm

The norm that the residual test of [`Options`](@ref) reads: [`L2Norm`](@ref) or
[`MaxNorm`](@ref). Each is a singleton type, so the norm is a type parameter of `Options` and
adds no byte to it.
"""
abstract type StoppingNorm end

"""
    L2Norm()

The Euclidean norm ``\\|F\\|_2``, the default [`StoppingNorm`](@ref) of [`Options`](@ref). Under
it the residual test reads `status.fnorm` and takes no reduction of its own.
"""
struct L2Norm <: StoppingNorm end

"""
    MaxNorm()

The max norm ``\\|F\\|_\\infty = \\max_i |F_i|``, a [`StoppingNorm`](@ref) of [`Options`](@ref).
Under it the residual test takes one reduction of the residual at the start and after each step
that reaches the test, `GeometricSolvers.rmaxnorm`, which also runs on a device array.
"""
struct MaxNorm <: StoppingNorm end

"""
    Options{R <: Real, N <: StoppingNorm}

The stopping test of the loop, and nothing else. It is `isbits`, and every tolerance is in
`R = real(T)` for the element type `T` of the iterate, so a `Float64` tolerance never reaches a
`Float32` kernel. The iteration cap of a line search is a field of the line search.

# Fields

- `f_abstol::R`, `f_reltol::R`: the residual test ``\\|F(x)\\|_N ≤ \\max(f_{abstol}, f_{reltol} \\|F(x₀)\\|_N)`` in the norm `N`.
- `x_abstol::R`, `x_reltol::R`: the step test ``\\|Δx\\|_2 ≤ \\max(x_{abstol}, x_{reltol} \\|x\\|_2)``.
- `min_iterations::Int32`, `max_iterations::Int32`: the bounds on the number of steps.
- `max_stalls::Int32`: the number of consecutive stalled steps after which the solve stops.
- `g_tol::R`: the stationarity tolerance ``\\|J^\\top F\\|_2 / (\\|J\\|_F \\|F\\|_2) \\le g_{\\text{tol}}`` on a [`LeastSquaresProblem`](@ref).
- `norm::N`: the [`StoppingNorm`](@ref) of the residual test, [`L2Norm`](@ref) or
  [`MaxNorm`](@ref). It is a singleton, so `Options` has the size of its other fields.
- `kappa::R`: the factor ``κ ∈ (0, 1)`` of the rate test
  ``\\hat θ_k / (1 - \\hat θ_k) \\|Δ_k\\|_2 ≤ κ \\max(x_{abstol}, x_{reltol} \\|x\\|_2)``, with
  ``\\hat θ_k = \\max(θ_k, θ_{k-1})`` and the rate ``θ_k = \\|Δ_k\\|_2 / \\|Δ_{k-1}\\|_2`` of the
  step ``Δ_k``.
- `roundoff::Bool`: the round-off stop, which ends a solve with `STALLED` at the first step of
  rate ``θ_k ≥ 1`` after a step of rate ``θ_{k-1} < 1``.

[`Options(T; kwargs...)`](@ref Options(::Type{T}) where {T <: Number}) builds one from the element
type `T` of the iterate, so the caller never names `R`. A solve takes only an `Options{R}` for
`R = real(T)` and raises an `ArgumentError` for any other: the tolerances of another precision do
not carry over, because the `Float64` default `f_reltol = √eps(Float64)` lies below `eps(Float32)`
and a `Float32` solve cannot reach it. `Options{R}` names the options of every norm in `R`.
"""
struct Options{R <: Real, N <: StoppingNorm}
    f_abstol::R
    f_reltol::R
    x_abstol::R
    x_reltol::R
    min_iterations::Int32
    max_iterations::Int32
    max_stalls::Int32
    g_tol::R
    norm::N
    kappa::R
    roundoff::Bool
end

# The positional forms in `R`: eight values, a norm, `κ` and the round-off flag, which default to
# `L2Norm()`, `κ = 1/10` and `false`, and the seven values before `g_tol`, which take
# `g_tol = √eps(R)`.
function Options{R}(f_abstol::Real, f_reltol::Real, x_abstol::Real, x_reltol::Real,
        min_iterations::Integer, max_iterations::Integer, max_stalls::Integer, g_tol::Real,
        norm::N = L2Norm(), kappa::Real = R(1) / 10,
        roundoff::Bool = false) where {R <: Real, N <: StoppingNorm}
    κ = R(kappa)
    check_kappa(κ)
    return Options{R, N}(R(f_abstol), R(f_reltol), R(x_abstol), R(x_reltol),
        Int32(min_iterations), Int32(max_iterations), Int32(max_stalls), R(g_tol), norm,
        κ, roundoff)
end

function Options{R}(f_abstol::Real, f_reltol::Real, x_abstol::Real, x_reltol::Real,
        min_iterations::Integer, max_iterations::Integer, max_stalls::Integer) where {R <:
                                                                                      Real}
    return Options{R}(f_abstol, f_reltol, x_abstol, x_reltol,
        min_iterations, max_iterations, max_stalls, sqrt(eps(R)))
end

@inline function nonnegative(name::Symbol, tol::Real)
    tol >= 0 || throw(ArgumentError("$name must be non-negative, got $tol"))
    return nothing
end

# `κ` of the rate test, in the precision `R` of the options: the test bounds the error estimate
# `θ / (1 - θ) ‖Δ‖` by the fraction `κ` of the step tolerance, so `κ` lies in `(0, 1)`. `NaN` fails
# both comparisons, and a value that rounds to `0` or `1` in `R` is refused as that value.
@inline function check_kappa(kappa::Real)
    zero(kappa) < kappa < one(kappa) ||
        throw(ArgumentError("kappa must be finite and in (0, 1), got $kappa"))
    return nothing
end

"""
    Options(T = Float64; f_abstol, f_reltol, x_abstol, x_reltol, min_iterations, max_iterations, max_stalls, g_tol, norm, kappa, roundoff)

Build the [`Options`](@ref) of a solve whose iterate has element type `T`, as `Options{real(T)}`.
The relative tolerances scale with `eps(R)`:

| keyword | default |
|:--|:--|
| `f_abstol` | `0` |
| `f_reltol` | `√eps(R)` |
| `x_abstol` | `0` |
| `x_reltol` | `2 eps(R)` |
| `min_iterations` | `0` |
| `max_iterations` | `100` |
| `max_stalls` | `2` |
| `g_tol` | `√eps(R)` |
| `norm` | [`L2Norm()`](@ref L2Norm) |
| `kappa` | `1/10` |
| `roundoff` | `false` |

**The stopping norm is the norm of the residual test only.** With `norm = MaxNorm()` the residual
test reads ``\\|F(x)\\|_\\infty \\le \\max(f_{abstol}, f_{reltol} \\|F(x₀)\\|_\\infty)``, so the
default `f_reltol` is relative to the max norm of the start. The step test, the stationarity test,
the `NONFINITE` test of the start, the merit ``\\|F\\|_2^2`` of every line search and
`status.fnorm` stay in the 2-norm, so a `SUCCESS` under `MaxNorm()` can report a `status.fnorm`
above `f_abstol`. On a [`Deflated`](@ref) problem the test reads the norm of the inner residual,
``\\|M F\\|_N / M``.

**The default residual test is relative.** The residual cannot fall below about
``\\mathrm{eps}(T) \\|J\\| \\|x\\|``, whatever the precision of the residual (Tisseur 2001,
Cor. 2.5), and a default cannot know ``\\|J\\| \\|x\\|``. So the default `f_abstol` is `0`, which
turns the absolute test off, and the relative test ``f_{reltol} \\|F(x₀)\\|`` decides.

**The stopping test runs before the first step, and `min_iterations` is `0`.** From a start at the
limiting accuracy one Newton step makes the result worse (Wilkinson, quoted by Tisseur 2001). The
relative test cannot see such a start: before the first step it compares ``\\|F(x₀)\\|`` with a
fraction of itself. So a start at the limiting accuracy returns unchanged after 0 iterations only
when the caller sets `f_abstol` at or above its residual floor, as an integrator that knows the
scale of `F` can.

**The rate test joins the step test** (Hairer and Wanner 1996, *Solving Ordinary Differential
Equations II*, Sec. IV.8, (8.9)–(8.10), doi:10.1007/978-3-642-05221-7). From the second step
on, the rate of the step ``Δ_k`` is ``θ_k = \\|Δ_k\\|_2 / \\|Δ_{k-1}\\|_2``, and once the
residual test fails, a step with ``\\hat θ_k = \\max(θ_k, θ_{k-1}) < 1`` and
``\\hat θ_k / (1 - \\hat θ_k) \\|Δ_k\\|_2 ≤ κ \\max(x_{abstol}, x_{reltol} \\|x\\|_2)`` returns
`SUCCESS` with `status.contracted == true`. The left side estimates the distance of `x` to the
limit of a contraction of rate ``\\hat θ_k``. The larger of the last two rates keeps one small
rate, after a step that a line search shrank or after an irregular secant step, from ending a
solve. The rate is `NaN` at the first step and after a step of zero norm, and a `NaN` rate passes
no test, so the rate test holds at the third step at the earliest; `status.rate` is ``θ_k``. The
test reads only a solve whose last fresh factor reports no rank (`status.rank == -1`, as an LU or
a QR factor does) or full column rank: along a rank-deficient SVD factor the steps can contract
onto a point that is no root, and the step test decides. On a [`LeastSquaresProblem`](@ref), a
[`VariableProjection`](@ref) and a [`Reparametrized`](@ref) problem around a
`LeastSquaresProblem` it reads no step along a reused factor of `reuse = k`: such steps
contract onto the point where ``J_0^\\top F(x) = 0`` for the Jacobian ``J_0`` of the factor, which
is no stationary point of ``\\|F\\|_2^2``, so the stationarity, the step or the residual test
ends the solve there, as without `reuse`. A step along a fresh factor, and every step of a square
problem, reads the rate test.

**The round-off stop is off by default.** With `roundoff = true`, the first step of rate
``θ_k ≥ 1`` after a step of rate ``θ_{k-1} < 1`` returns `STALLED` with the iterate it reached
(Hairer, McLachlan and Razakarivony 2008, *BIT Numer. Math.* 48, p. 235,
doi:10.1007/s10543-008-0170-3): the increments have stopped decreasing, as they do once the
residual reaches its round-off floor. A step of zero norm returns `STALLED` under either setting,
as [`solve!!`](@ref) says.

The rate test and the round-off stop read the steps that the step test reads: not a `STALLED`
step, not a step that a radius or a damping limits, and not before `min_iterations` steps.

A negative or `NaN` tolerance, a negative `min_iterations`, a `max_iterations` below
`min_iterations`, a `max_stalls` below 1 and a `kappa` outside ``(0, 1)`` in `real(T)` raise an
`ArgumentError`.

```jldoctest
julia> opt = Options(ComplexF32; f_reltol = 1e-4);

julia> typeof(opt)
Options{Float32, L2Norm}

julia> opt.f_reltol
0.0001f0

julia> Options(Float64; norm = MaxNorm()).norm
MaxNorm()
```
"""
function Options(::Type{T} = Float64;
        f_abstol::Real = 0,
        f_reltol::Real = sqrt(eps(real(T))),
        x_abstol::Real = 0,
        x_reltol::Real = 2 * eps(real(T)),
        min_iterations::Integer = 0,
        max_iterations::Integer = 100,
        max_stalls::Integer = 2,
        g_tol::Real = sqrt(eps(real(T))),
        norm::StoppingNorm = L2Norm(),
        kappa::Real = real(T)(1) / 10,
        roundoff::Bool = false) where {T <: Number}
    R = real(T)
    # one call per tolerance, because a loop over their mixed types allocates
    nonnegative(:f_abstol, f_abstol)
    nonnegative(:f_reltol, f_reltol)
    nonnegative(:x_abstol, x_abstol)
    nonnegative(:x_reltol, x_reltol)
    nonnegative(:g_tol, g_tol)
    min_iterations >= 0 ||
        throw(ArgumentError("min_iterations must be non-negative, got $min_iterations"))
    max_iterations >= min_iterations ||
        throw(ArgumentError("max_iterations = $max_iterations is below min_iterations = $min_iterations"))
    max_stalls >= 1 ||
        throw(ArgumentError("max_stalls must be at least 1, got $max_stalls"))
    Options{R}(R(f_abstol), R(f_reltol), R(x_abstol), R(x_reltol),
        Int32(min_iterations), Int32(max_iterations), Int32(max_stalls), R(g_tol), norm,
        kappa, roundoff)
end

# `nothing` when `options` is an `Options{real(T)}`, the options of an iterate of element type
# `T`, and an `ArgumentError` otherwise. `R` and `T` are types, so the test is a constant where
# `solve!!` runs in a kernel.
@inline function check_options(::Type{T}, ::Options{R}) where {T, R}
    R === real(T) || throw(ArgumentError(
        "options are Options{$R}, but the iterate has element type $T: pass Options($T; …)"))
    return nothing
end

"""
    converged(options::Options{R}, status::SolverStatus{R}, fnorm₀::R, xnorm::R)

The stopping test of [`Options`](@ref) at `status`, for a start with residual norm `fnorm₀` and
an iterate of norm `xnorm`. It holds once `min_iterations` steps are taken and either the residual
test or, after a step of nonzero norm, the step test holds. At 0 iterations it is the test before
the first step: the relative residual test then holds only for ``F(x₀) = 0``, so a start passes it
through `f_abstol`. A zero step is no progress, so it never passes the step test. The residual
test compares `status.fnorm` with `fnorm₀` as they are given: under [`MaxNorm`](@ref), the caller
passes both in the max norm, as [`solve!!`](@ref) does.

[`solve!!`](@ref) returns `SUCCESS` through the residual test or the rate test of
[`Options`](@ref): a step test that holds while both fail returns `STALLED`.
"""
function converged(options::Options{R}, status::SolverStatus{R}, fnorm₀::R, xnorm::R) where {R}
    return residual_converged(options, status, fnorm₀) ||
           step_converged(options, status, xnorm)
end

# The residual test of `Options`, once `min_iterations` steps are taken.
@inline function residual_converged(
        options::Options{R}, status::SolverStatus{R}, fnorm₀::R) where {R}
    status.iterations >= options.min_iterations || return false
    return status.fnorm <= max(options.f_abstol, options.f_reltol * fnorm₀)
end

# The step test of `Options` for an iterate of norm `xnorm`, once `min_iterations` steps are taken.
# It needs a step of nonzero norm.
@inline function step_converged(
        options::Options{R}, status::SolverStatus{R}, xnorm::R) where {R}
    status.iterations >= options.min_iterations || return false
    return status.iterations > 0 && status.stepnorm > 0 &&
           status.stepnorm <= max(options.x_abstol, options.x_reltol * xnorm)
end

# The rate test of `Options` for an iterate of norm `xnorm`, once `min_iterations` steps are taken,
# after a step of rate `status.rate` where the step before it had the rate `θ_prev`: a step of
# nonzero norm and `θ < 1` with `θ / (1 - θ) ‖Δ‖ ≤ κ max(x_abstol, x_reltol ‖x‖)`, for the larger
# `θ` of the two rates. One small rate after a step that a line search shrank, or after an
# irregular secant step, does not pass it. A `NaN` rate, of either step, fails `θ < 1`.
@inline function rate_converged(
        options::Options{R}, status::SolverStatus{R}, θ_prev::R, xnorm::R) where {R}
    status.iterations >= options.min_iterations || return false
    θ = max(status.rate, θ_prev)
    status.stepnorm > 0 && θ < one(R) || return false
    return θ / (one(R) - θ) * status.stepnorm <=
           options.kappa * max(options.x_abstol, options.x_reltol * xnorm)
end

# The round-off stop of `Options` after a step of rate `status.rate`, where the step before it had
# the rate `θ_prev`: under `roundoff = true`, a step of rate `θ ≥ 1` after one of rate below `1`,
# once `min_iterations` steps are taken. A `NaN` rate fails both comparisons. A step of zero norm
# needs no rule here: `solve!!` stops on it under any `roundoff`.
@inline function roundoff_stop(
        options::Options{R}, status::SolverStatus{R}, θ_prev::R) where {R}
    options.roundoff || return false
    status.iterations >= options.min_iterations || return false
    return status.rate >= one(R) && θ_prev < one(R)
end
