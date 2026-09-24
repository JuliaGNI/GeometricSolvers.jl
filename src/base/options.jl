"""
    Options{R <: Real}

The stopping test of the loop, and nothing else. It is `isbits`, and every tolerance is in
`R = real(T)` for the element type `T` of the iterate, so a `Float64` tolerance never reaches a
`Float32` kernel. The iteration cap of a line search is a field of the line search.

# Fields

- `f_abstol::R`, `f_reltol::R`: the residual test ``\\|F(x)\\| ≤ \\max(f_{abstol}, f_{reltol} \\|F(x₀)\\|)``.
- `x_abstol::R`, `x_reltol::R`: the step test ``\\|Δx\\| ≤ \\max(x_{abstol}, x_{reltol} \\|x\\|)``.
- `min_iterations::Int32`, `max_iterations::Int32`: the bounds on the number of steps.
- `max_stalls::Int32`: the number of consecutive stalled steps after which the solve stops.

[`Options(T; kwargs...)`](@ref Options(::Type{T}) where {T <: Number}) builds one from the element
type `T` of the iterate, so the caller never names `R`, and `convert` turns an `Options` of
another `R` into `Options{R}`.
"""
struct Options{R <: Real}
    f_abstol::R
    f_reltol::R
    x_abstol::R
    x_reltol::R
    min_iterations::Int32
    max_iterations::Int32
    max_stalls::Int32
end

"""
    Options(T; f_abstol, f_reltol, x_abstol, x_reltol, min_iterations, max_iterations, max_stalls)

Build the [`Options`](@ref) of a solve whose iterate has element type `T`, as `Options{real(T)}`.
The two relative tolerances scale with `eps(R)`:

| keyword | default |
|:--|:--|
| `f_abstol` | `0` |
| `f_reltol` | `√eps(R)` |
| `x_abstol` | `0` |
| `x_reltol` | `2 eps(R)` |
| `min_iterations` | `0` |
| `max_iterations` | `100` |
| `max_stalls` | `2` |

**The default residual test is relative.** The residual cannot fall below about
``\\mathrm{eps}(T) \\|J\\| \\|x\\|``, whatever the precision of the residual (Tisseur 2001,
Cor. 2.5), and an absolute default cannot know ``\\|J\\| \\|x\\|``. Any fixed value of `f_abstol`
lies below that floor for some problem, which then ends as a stall. So `f_abstol` is `0` unless
the caller knows the scale of `F`.

**The stopping test runs before the first step, and `min_iterations` is `0`.** From a start at the
limiting accuracy one Newton step makes the result worse (Wilkinson, quoted by Tisseur 2001), and
an integrator starts each solve from an extrapolation. A start that passes the test returns
unchanged after 0 iterations.

A negative or `NaN` tolerance, a negative `min_iterations`, a `max_iterations` below
`min_iterations` and a `max_stalls` below 1 raise an `ArgumentError`.

```jldoctest
julia> opt = Options(ComplexF32; f_reltol = 1e-4);

julia> typeof(opt)
Options{Float32}

julia> opt.f_reltol
0.0001f0
```
"""
function Options(::Type{T};
        f_abstol::Real = 0,
        f_reltol::Real = sqrt(eps(real(T))),
        x_abstol::Real = 0,
        x_reltol::Real = 2 * eps(real(T)),
        min_iterations::Integer = 0,
        max_iterations::Integer = 100,
        max_stalls::Integer = 2) where {T <: Number}
    R = real(T)
    for (name, tol) in ((:f_abstol, f_abstol), (:f_reltol, f_reltol),
        (:x_abstol, x_abstol), (:x_reltol, x_reltol))
        tol >= 0 || throw(ArgumentError("$name must be non-negative, got $tol"))
    end
    min_iterations >= 0 ||
        throw(ArgumentError("min_iterations must be non-negative, got $min_iterations"))
    max_iterations >= min_iterations ||
        throw(ArgumentError("max_iterations = $max_iterations is below min_iterations = $min_iterations"))
    max_stalls >= 1 ||
        throw(ArgumentError("max_stalls must be at least 1, got $max_stalls"))
    Options{R}(R(f_abstol), R(f_reltol), R(x_abstol), R(x_reltol),
        Int32(min_iterations), Int32(max_iterations), Int32(max_stalls))
end

function Base.convert(::Type{Options{R}}, opt::Options) where {R <: Real}
    Options{R}(R(opt.f_abstol), R(opt.f_reltol), R(opt.x_abstol), R(opt.x_reltol),
        opt.min_iterations, opt.max_iterations, opt.max_stalls)
end

"""
    converged(options::Options{R}, status::SolverStatus{R}, fnorm₀::R, xnorm::R)

The stopping test of [`Options`](@ref) at `status`, for a start with residual norm `fnorm₀` and
an iterate of norm `xnorm`. It holds once `min_iterations` steps are taken and either the residual
test or, after the first step, the step test holds. At 0 iterations it is the test before the
first step, which a start at the limiting accuracy passes.
"""
function converged(options::Options{R}, status::SolverStatus{R}, fnorm₀::R, xnorm::R) where {R}
    status.iterations >= options.min_iterations || return false
    status.fnorm <= max(options.f_abstol, options.f_reltol * fnorm₀) && return true
    status.iterations > 0 &&
        status.stepnorm <= max(options.x_abstol, options.x_reltol * xnorm)
end
