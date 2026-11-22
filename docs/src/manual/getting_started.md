```@meta
CurrentModule = GeometricSolvers
```

# Getting started

A solve in GeometricSolvers has four parts:

- a **problem**, which holds the residual function and, optionally, its Jacobian,
- a **method**, which holds the algorithmic configuration, such as [`Newton`](@ref) with a linear
  solver and a globalization,
- an **initial iterate** `x₀`, whose storage selects the call form of the functions and the
  regime of the solve, and
- the **parameters** `p`, which arrive at solve time.

The problem holds no iterate and no parameters. One problem object therefore serves a whole
time-stepping loop, and a batch of systems shares one problem.

## A first solve

[`solve`](@ref) takes the problem, the initial iterate, the method and the parameters, and returns
the solution and a [`SolverStatus`](@ref):

```jldoctest getting_started
julia> using GeometricSolvers

julia> prob = NonlinearProblem((x, p) -> x^2 - p);

julia> x, status = solve(prob, 1.0, Newton(), 2.0);

julia> x ≈ sqrt(2)
true

julia> status.code === SUCCESS
true
```

The residual of a scalar or static problem has the form `F(x, p)` and returns a value. Without a
Jacobian on the problem, [`Newton`](@ref) differentiates `F` with ForwardDiff.

## Call forms

The storage of the initial iterate selects the call form of the residual and of the Jacobian.
The problem has no flag for it.

| iterate `x₀` | residual | Jacobian `J` |
|:--|:--|:--|
| `Number` | `F(x, p)` returns a number | `J(x, p)` returns a number |
| `SVector` (a `StaticVector`) | `F(x, p)` returns an `SVector` | `J(x, p)` returns an `SMatrix` |
| `AbstractVector` that is not static (`Vector`, `CuArray`, `MtlArray`, …) | `F!(y, x, p)` writes into `y` | `J!(J, x, p)` writes into `J` |

A static system:

```jldoctest getting_started
julia> using StaticArrays

julia> F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2]);

julia> x, status = solve(NonlinearProblem(F), SVector(1.0, 0.5), Newton(), 2.0);

julia> x ≈ SVector(1.0, 1.0)
true
```

The same system with in-place functions on a `Vector`, here with an analytic Jacobian:

```jldoctest getting_started
julia> function F!(y, x, p)
           y[1] = x[1]^2 + x[2]^2 - p
           y[2] = x[1] - x[2]
           return y
       end;

julia> function J!(J, x, p)
           J[1, 1] = 2x[1]; J[1, 2] = 2x[2]
           J[2, 1] = 1;     J[2, 2] = -1
           return J
       end;

julia> prob = NonlinearProblem(F!, J!);

julia> x, status = solve(prob, [1.0, 0.5], Newton(), 2.0);

julia> x ≈ [1.0, 1.0]
true
```

`solve` copies an array `x₀`, so the caller's initial guess stays unchanged.

## Parameters at solve time

The parameters `p` are the fourth positional argument of [`solve`](@ref), [`init`](@ref) and
[`solve_batch!`](@ref)`(X, prob, method, p)`, the fifth of `solve_batch!(X, S, prob, method, p)`
with a status vector `S`, and the last argument of every function of the problem. They can be any
value: a number, a tuple, a named tuple, an array or a struct. The default is `nothing`. Because
`p` is not part of the problem, the same problem solves for other parameters without a rebuild:

```jldoctest getting_started
julia> prob = NonlinearProblem((x, p) -> x^2 - p);

julia> [solve(prob, 1.0, Newton(), p)[1] for p in (4.0, 9.0)] ≈ [2.0, 3.0]
true
```

## Preallocated solves: `init` and `solve!`

[`init`](@ref), [`solve!`](@ref) and [`solve`](@ref) are the functions of
[CommonSolve.jl](https://github.com/SciML/CommonSolve.jl). `init` takes the arguments of `solve`
and returns a [`NonlinearSolver`](@ref) that holds all buffers, the factorisation workspace and
the automatic-differentiation preparation. `solve!(solver)` solves from the iterate that the
solver holds, and `solve!(solver, x, p)` solves from another iterate with other parameters of the
same types, in place into an array `x`:

```jldoctest getting_started
julia> solver = init(NonlinearProblem(F!), [1.0, 0.5], Newton(), 2.0);

julia> x, status = solve!(solver);

julia> x ≈ [1.0, 1.0]
true

julia> x, status = solve!(solver, [3.0, 1.0], 8.0);

julia> x ≈ [2.0, 2.0]
true
```

On primal inputs, `solve!(init(args...; kwargs...))` returns what `solve(args...; kwargs...)`
returns. The preallocated form is the one for a loop that solves many times, such as the stages
of an integrator: an R3 solve with `solve!` on an `Array` reuses its buffers.

## Options

The keywords of [`solve`](@ref) and [`init`](@ref) build the [`Options`](@ref) of the solve, typed
in `R = real(eltype(x₀))`:

| keyword | default | meaning |
|:--|:--|:--|
| `f_abstol` | `0` | absolute residual tolerance |
| `f_reltol` | `√eps(R)` | residual tolerance relative to ``\|F(x_0)\|`` |
| `x_abstol` | `0` | absolute step tolerance |
| `x_reltol` | `2 eps(R)` | step tolerance relative to ``\|x\|`` |
| `min_iterations` | `0` | lower bound on the number of steps |
| `max_iterations` | `100` | upper bound on the number of steps |
| `max_stalls` | `2` | consecutive stalled steps before the solve stops |
| `g_tol` | `√eps(R)` | stationarity tolerance of a [`LeastSquaresProblem`](@ref) |

The residual test is ``\|F(x)\| \le \max(f_{abstol}, f_{reltol} \|F(x_0)\|)``. It returns
`SUCCESS`, and so does the stationarity test of a [`LeastSquaresProblem`](@ref); the step test of
`x_abstol` and `x_reltol` returns `STALLED`. The default test is relative, because no default can
know the scale of `F`; a caller who knows the scale sets `f_abstol`.

```jldoctest getting_started
julia> x, status = solve(prob, 1.0, Newton(), 2.0; f_abstol = 1e-12, max_iterations = 20);

julia> status.code === SUCCESS
true
```

An [`Options`](@ref) object can also be passed as `options`. It must have the real type of the
iterate: [`Options(T; kwargs...)`](@ref Options(::Type{T}) where {T <: Number}) builds one for the
element type `T`. A `Float32` solve therefore takes `Options(Float32; …)`; an `Options{Float64}`
raises an `ArgumentError` there.

```jldoctest getting_started
julia> opt = Options(Float32; f_reltol = 1e-5);

julia> x, status = solve(prob, 1.0f0, Newton(), 2.0f0; options = opt);

julia> x isa Float32 && x ≈ sqrt(2.0f0)
true
```

Passing both `options` and keyword tolerances raises an `ArgumentError`.

## Status and return codes

Every solve returns a [`SolverStatus`](@ref). It is `isbits`, so a kernel can write one per
system. Its fields are:

- `code`: the [`ReturnCode`](@ref), why the solve ended,
- `iterations`: the number of steps,
- `fnorm`: ``\|F(x)\|`` at return, and `stepnorm`: the norm of the last step,
- `step_failures`: the count of step-rule failures, such as failed line searches, restarts or
  refinement failures,
- `promoted`: a factorisation ran in a higher precision than the method's factorisation
  precision,
- `stationary`: the solve converged by the stationarity test of a [`LeastSquaresProblem`](@ref),
- `f_calls` and `jvp_calls`: the residual and Jacobian–vector product counts of a matrix-free
  solve,
- `rank`, `nullnorm` and `gradratio`: range and spectrum diagnostics of the last fresh
  factorisation.

The return codes are:

| code | meaning |
|:--|:--|
| `SUCCESS` | the residual test (or the stationarity test of a least-squares problem) holds |
| `STALLED` | the iterate or the residual stopped improving before the test held |
| `MAXITERS` | the iteration cap was reached |
| `SINGULAR` | a factorisation failed; the solve does not throw |
| `NONFINITE` | a residual, a step or a norm is not finite |
| `LINESEARCH_FAILED` | the globalization found no acceptable step |

A solve does not throw for a failure that it detects. It returns the code instead:

```jldoctest getting_started
julia> x, status = solve(NonlinearProblem((x, p) -> x^2 + 1), 0.0, Newton());

julia> status.code === SINGULAR
true
```

An exception that a function of the problem raises, such as a `DomainError` of `sqrt` at a
negative trial iterate, propagates to the caller.

[`stopping_test`](@ref) names the test that ended a converged solve, and
`GeometricSolvers.isconverged` and `GeometricSolvers.isstalled` read the code:

```jldoctest getting_started
julia> stopping_test(solve(prob, 1.0, Newton(), 2.0)[2])
:residual
```

The section [Status and stopping test](@ref) of the API explains how `rank`, `nullnorm` and
`gradratio` classify a `STALLED` or `MAXITERS` return.

## The SciML `NonlinearProblem` extension

When SciMLBase is loaded, the package extension `GeometricSolversSciMLBaseExt` adds methods of
[`solve`](@ref) and [`init`](@ref) for a `SciMLBase.NonlinearProblem` with a [`Newton`](@ref) or
[`Picard`](@ref) method. The extension translates the problem and calls the native solve, so the
result is the native `(x, status)`:

```julia
using GeometricSolvers, SciMLBase

f(u, p) = u .^ 2 .- p
prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0], 2.0)

x, status = solve(prob, Newton(); abstol = 1e-12)
GeometricSolvers.sciml_retcode(status.code)   # SciMLBase.ReturnCode.Success
```

The call form follows `SciMLBase.isinplace(prob)` and the type of `prob.u0`. The keywords
`abstol`, `reltol`, `maxiters` and `options` map to the native [`Options`](@ref), and
[`sciml_retcode`](@ref) maps a [`ReturnCode`](@ref) to a `SciMLBase.ReturnCode.T`. The section
[SciMLBase extension](@ref) of the API gives the complete rules.

## Where to go next

- [The three regimes](@ref regimes) explains how the storage of the iterate selects the code
  path, and how to choose between value, batched and in-place solves.
- [Methods by regime](@ref guide) lists the methods that run in each regime, and the conditions
  under which each one applies.
- [Problem types](@ref problems) describes the least-squares, stationarity, reparametrized,
  deflated and eliminated problems.
- The pages under *Algorithms* describe each nonlinear method, globalization, linear solver and
  differentiation path, and the composition of methods.
