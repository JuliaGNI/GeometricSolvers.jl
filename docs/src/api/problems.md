```@meta
CurrentModule = GeometricSolvers
```

# Problems and solves

## The problem

A problem holds functions and, for a [`NonlinearProblem`](@ref), an optional Jacobian prototype:
the residual, optionally its derivative, and optionally `jacobian_prototype` for a sparse R3
Jacobian in [`NonlinearProblem`](@ref); the residual, optionally its derivative, and its length in
[`LeastSquaresProblem`](@ref); or the scalar action, its equation and unknown index sets, and an
optional datum in [`StationarityProblem`](@ref). [`Reparametrized`](@ref) wraps a
`NonlinearProblem` or a `StationarityProblem` in a coordinate chart `x = g(y, p)`, and
[`Deflated`](@ref) wraps either in a deflation of the roots it is given, so that a solve from the
same start can converge to another root. [`BlockEliminated`](@ref) partitions the unknowns of a
`NonlinearProblem` or a `StationarityProblem` into inner unknowns `y = x[Y]` and outer unknowns
`z = x[Z]`, eliminates `y` at each outer iterate by solving the inner equations, and steps on the
outer equations with the Schur complement `J_zz - J_zy J_yy⁺ J_yz`; [`VariableProjection`](@ref)
does the same for a `LeastSquaresProblem` across all `m` residual rows by Golub–Pereyra variable
projection or Kaufman's reduced Jacobian (`BlockEliminated` on a `LeastSquaresProblem` forwards to
`VariableProjection`). The storage of the iterate selects the call form of the functions, so one
problem serves every regime. `NonlinearProblem`, `StationarityProblem`, `LeastSquaresProblem`,
`Reparametrized`, `Deflated`, `BlockEliminated` and `VariableProjection` are exported;
`GeometricSolvers.jacobian_prototype` and `GeometricSolvers.jacobian_pattern` are internal
accessors.

```@docs
AbstractNonlinearProblem
NonlinearProblem
StationarityProblem
LeastSquaresProblem
Reparametrized
Deflated
BlockEliminated
VariableProjection
jacobian_prototype
jacobian_pattern
```

A [`NonlinearProblem`](@ref) also carries the local residuals of a [`BlockPartition`](@ref) when
it is built with the `block_residuals` keyword: its `F` is then a [`BlockResidual`](@ref) that
forwards every call to the global residual, and the block functions are read with
[`block_residuals`](@ref), [`block_jacobians`](@ref), [`block_residual`](@ref) and
[`block_jacobian`](@ref). `J` stays the caller's own Jacobian, so every method that reads it is
unchanged.

```@docs
BlockResidual
block_residuals
block_jacobians
block_residual
block_jacobian
```

A [`NonlinearProblem`](@ref) built with the `stage_matrix` and `stage_jacobian` keywords carries
the stage system of an implicit Runge–Kutta method: its `F` is then a [`StageResidual`](@ref) that
forwards every call to the stacked residual, and the stage matrix and the Jacobian of the vector
field are read with [`stage_matrix`](@ref) and [`stage_jacobian`](@ref), which a [`Newton`](@ref)
with [`KroneckerFactorization`](@ref) does. These names are exported.

```@docs
StageResidual
stage_matrix
stage_jacobian
```

### Reduced-space elimination with a batch inside the residual

[`BatchEliminated`](@ref) eliminates a whole batch of independent inner systems *from the residual
of an outer matrix-free [`Newton`](@ref)*, rather than a block of one unknown vector: the outer
solver steps on the `m`-vector alone, and every evaluation of that residual — the Krylov products
of the outer step included — solves the batch and deposits its moments. It is the pattern of the
implicit PIC method (`Chen2011`, pp. 7025 and 7033; `LanzkronRoseWilkes:1996`), where the outer
Newton works on the field and the per-particle orbit solves are the eliminated unknowns. The inner
tolerance of each evaluation is the adaptive quadratic
``\max(2\varepsilon(R_{\text{in}}), R_{\text{in}}(\tfrac{1}{2}\min(c, \|g_2(\text{prev})\|^2)/\sqrt{N_p}))``,
the batch is warm-started from the previous evaluation — so a second evaluation at the same
iterate and tolerance takes `0` inner iterations — and the outer Jacobian–vector product is a
one-sided difference at the step ``\varepsilon \max(1, \|x\|)/\|v\|``, with the relative step
``\varepsilon = \sqrt{\operatorname{eps}(R_{\text{eff}})}`` of the coarser of the two precisions,
so that a `Float32` batch inside a `Float64` outer solve converges at the
tolerance its own accuracy supports and, below it, ends on a stall rather than on a factorisation
failure — on the problem of [`make_pic_elimination_problem`](@ref), the solve at `2e-4` returns
`SUCCESS` and the one at `1e-10` `STALLED`. The batch's
[`BatchSummary`](@ref) reports its global residual norm, whether every system converged, and how
many did not; [`deposited_moments`](@ref) returns what the deposit wrote, and
[`reset_batch!`](@ref) returns the problem to its initial state.
[`make_pic_elimination_problem`](@ref) builds the one-dimensional implicit electrostatic PIC test
problem that the elimination checks solve.

```@docs
BatchEliminated
BatchSummary
batch_summary
BatchParams
DepositedMoments
deposited_moments
reset_batch!
eliminated_indices
make_pic_elimination_problem
```

## The solve interface

[`solve`](@ref) adapts the method
to `R = real(eltype(x₀))`, calls [`init_state`](@ref) and [`solve!!`](@ref) on a copy of `x₀` when
`x₀` is an array, and returns `(x, status)`; when `x₀` or `p` carries a `ForwardDiff.Dual` leaf,
[`solve`](@ref) on a [`NonlinearProblem`](@ref) strips duals via [`StripDual`](@ref), runs the
primal solve, and on `SUCCESS` differentiates the root implicitly at `x*`, while on a
[`StationarityProblem`](@ref) it raises an `ArgumentError`. With the same arguments,
`solver = init(prob, x₀, method, p; options, kwargs...)` pre-allocates a
[`NonlinearSolver`](@ref) (or a [`ReparametrizedSolver`](@ref) for a [`Reparametrized`](@ref)
problem) that holds a copy of an array `x₀` and `p`, `solve!(solver)` solves into
`solver.x` and returns `(solver.x, status)`, and `solve!(solver, x, p)` solves from another `x`
with other `p` of the same types, in place into an array `x`; on a [`NonlinearSolver`](@ref), with
`refactorize = true` its first step factorises afresh, where a [`Newton`](@ref) with
`keep_factor = true` would otherwise start on the factor of the last solve. `init`, `solve!` and
`solve` are the
functions of CommonSolve.jl, which the package extends on its own types, and
`solve!(init(args...))` returns what `solve(args...)` returns on primal inputs. [`solve!!`](@ref) takes a solver
state from `init_state(method_R, prob, x₀, p)` and runs the loop in R1 and R3 alike. On a batch
`X` of independent small systems, [`solve_batch!`](@ref) solves each in place on `get_backend(X)`
with one work-item per system, and [`PerSystem`](@ref) wraps an indexable collection of
per-system parameters so system `i` receives `p.data[i]`.

```@docs
NonlinearSolver
ReparametrizedSolver
init
solve
solve!
solve_batch!
solve!!
PerSystem
```

## Status and stopping test

Every type in this section is `isbits`, so a kernel can hold it, and every real field is in
`R = real(T)` for the element type `T` of the iterate. [`SolverStatus`](@ref) also carries the
range and spectrum diagnostics (`rank`, `nullnorm` and `gradratio`) recorded at the iterate
``x_k`` of the last fresh [`Newton`](@ref) factorisation, not at the iterate ``x`` at return (a
solve that steps only along a factor kept with `keep_factor = true` reports the `rank` of that
factor and `NaN` for the other two); a
solve of a [`LeastSquaresProblem`](@ref) that ends on the stationarity test has factorised afresh
at the iterate it returns. Where such a solve returns the start point of a step it discards, the
status carries the diagnostics of the factorisation at that point, not those of the later
re-measure at the end of the step; its `iterations` counts the steps that reached that point, and
its `stepnorm` is that of the last of them.
With `regularization = μ ≠ 0`, `rank` and `nullnorm` describe ``J(x_k) + μ I`` and `gradratio`
describes ``J(x_k)``, so the classification below holds for `μ = 0` only.
For `n ≥ 2` (for `n = 1`, a nonzero scalar Jacobian always gives `gradratio = 1`, and `rank = 1`
with an SVD factor), they classify the outcome and a `STALLED` or `MAXITERS` return:
a consistent rank-deficient system that converged has a small residual (`fnorm` small) and
nothing outside the range (`rank < n`, `nullnorm ≈ 0`); a least-squares point with no root has
`gradratio ≈ 0` (`fnorm > 0`) and the residual outside the range
(``\text{nullnorm} \approx \|F(x_k)\| > 0``, and `rank < n` for a square system; a
[`LeastSquaresProblem`](@ref) with more residuals than unknowns can have full rank there); a stall
in nearly singular directions has the residual inside the range (`nullnorm ≈ 0`) at a fine SVD
cutoff and outside it (`rank < n`, ``\text{nullnorm} \approx \|F(x_k)\|``) at a coarse one;
and an in-range stall from conditioning or step acceptance has `gradratio` bounded away from zero
with `nullnorm ≈ 0`.

The residual test of [`Options`](@ref) reads the 2-norm of the residual, or with
`norm = MaxNorm()` its max norm; `status.fnorm` is the 2-norm under either. [`L2Norm`](@ref) and
[`MaxNorm`](@ref) are exported, and their supertype `GeometricSolvers.StoppingNorm` is not.

Each step's rate ``θ_k = \|Δ_k\| / \|Δ_{k-1}\|`` goes into `status.rate`. The rate test of
[`Options`](@ref) joins the step test: a step with ``\hat θ_k = \max(θ_k, θ_{k-1}) < 1`` and
``\hat θ_k / (1 - \hat θ_k) \|Δ_k\| ≤ κ \max(x_{abstol}, x_{reltol} \|x\|)`` returns `SUCCESS` with
`status.contracted` set, and [`stopping_test`](@ref) then returns `:contraction`; it reads no step
along an SVD factor whose rank is below the number of the unknowns it solves for, the outer ones of
an eliminated problem, nor a step of a composition whose stepping member has such a factor. On a
[`LeastSquaresProblem`](@ref), a [`VariableProjection`](@ref) and a [`Reparametrized`](@ref)
problem around a `LeastSquaresProblem` it reads no step along a reused factor, whose steps
contract onto a point that need not be stationary. With
`roundoff = true`, a step whose increment stops decreasing returns `STALLED` at the iterate it
reached. `status.reused` is set where the last step of a [`Newton`](@ref) solve with `reuse`
stepped along a reused factor.

```@docs
ReturnCode
sciml_retcode
SolverStatus
stopping_test
StepInfo
Options
Options(::Type{T}) where {T <: Number}
StoppingNorm
L2Norm
MaxNorm
```

## SciMLBase extension

When [SciMLBase](https://github.com/SciML/SciMLBase.jl) is loaded, the package extension
`GeometricSolversSciMLBaseExt` adds methods of [`solve`](@ref) and [`init`](@ref) for a
`SciMLBase.NonlinearProblem` and a method [`Newton`](@ref) or [`Picard`](@ref). Both translate the
problem and call the native [`solve`](@ref) or [`init`](@ref), so `solve` returns `(x, status)`,
`init` returns a [`NonlinearSolver`](@ref) that holds a copy of an array `u0`, and
`solve!(init(prob, method; kwargs...))` returns what `solve(prob, method; kwargs...)` returns; the
native [`solve!`](@ref) serves the solver. A SciML caller maps `status.code` to a
`SciMLBase.ReturnCode.T` with [`sciml_retcode`](@ref)
(`GeometricSolvers.sciml_retcode(status.code)`), which is `public` and not exported.

The call form follows `SciMLBase.isinplace(prob)` and `u0 = prob.u0`: an out-of-place `f(u, p)`
with a `StaticArray` or a `Number` `u0` solves in R1 and returns a value of the type of `u0` (with
an `SVector` `u0`, an out-of-place `jac(u, p)` must return an `SMatrix`, as the native R1 `J`
does); an in-place `f(du, u, p)` with an array `u0` that is not a `StaticArray` solves in R3 on a
copy of `u0`; an out-of-place `f(u, p)` with such an array `u0` solves in R3 through a closure
wrapper `(y, x, p) -> (y .= f(x, p))` (and `(A, x, p) -> (A .= jac(x, p))` for `prob.f.jac`),
which allocates as `f` and `jac` do; and an in-place `f` with a `StaticArray` or a `Number` `u0`
raises an `ArgumentError`. `prob.f.jac` becomes the `J` of the native problem (overridden by a
method's own `jacobian` field), and `prob.p` is passed unchanged, `SciMLBase.NullParameters()`
included. A problem with `jac_prototype`, `sparsity`, `lb` or `ub` set raises an `ArgumentError`.

The extension merges `prob.kwargs` with the call's keywords (the call's winning) and accepts the
closed set `abstol`, `reltol`, `maxiters` and `options`: `abstol` maps to `f_abstol` (and sets
`f_reltol = 0` when `reltol` is not given), `reltol` maps to `f_reltol`, `maxiters` maps to
`max_iterations`, and `options` passes an [`Options`](@ref) alone; any other keyword, or `options`
together with another keyword, raises an `ArgumentError`.

## IntervalArithmetic extension

When [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl) is loaded, the
package extension `GeometricSolversIntervalArithmeticExt` adds a method of [`certify`](@ref) for a
[`NonlinearProblem`](@ref) and an iterate `x` that is a real floating-point `Number`, a
`StaticVector` or a host `Vector`. It runs Krawczyk's test in intervals of the element type of `x`
and returns `(certified, enclosure, guaranteed)`: whether a box around `x` holds exactly one root,
an enclosure of that root, and whether IntervalArithmetic guarantees the intervals of `F(x)` and
`J(X)`. `certify` is `public` and not exported (`GeometricSolvers.certify`); without the extension
it has no method.

```@docs
certify
```
