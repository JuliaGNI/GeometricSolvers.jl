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
Under it the residual test takes one reduction of the residual at the start and after each step,
`GeometricSolvers.rmaxnorm`, which also runs on a device array.
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
end

# The positional forms in `R`: eight values and a norm, `L2Norm()` without one, and the seven
# values before `g_tol`, which take `g_tol = √eps(R)`.
function Options{R}(f_abstol::Real, f_reltol::Real, x_abstol::Real, x_reltol::Real,
        min_iterations::Integer, max_iterations::Integer, max_stalls::Integer, g_tol::Real,
        norm::N = L2Norm()) where {R <: Real, N <: StoppingNorm}
    return Options{R, N}(R(f_abstol), R(f_reltol), R(x_abstol), R(x_reltol),
        Int32(min_iterations), Int32(max_iterations), Int32(max_stalls), R(g_tol), norm)
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

"""
    Options(T = Float64; f_abstol, f_reltol, x_abstol, x_reltol, min_iterations, max_iterations, max_stalls, g_tol, norm)

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

A negative or `NaN` tolerance, a negative `min_iterations`, a `max_iterations` below
`min_iterations` and a `max_stalls` below 1 raise an `ArgumentError`.

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
        norm::StoppingNorm = L2Norm()) where {T <: Number}
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
        Int32(min_iterations), Int32(max_iterations), Int32(max_stalls), R(g_tol), norm)
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

[`solve!!`](@ref) returns `SUCCESS` only through the residual test: a step test that holds while
the residual test fails returns `STALLED`.
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
