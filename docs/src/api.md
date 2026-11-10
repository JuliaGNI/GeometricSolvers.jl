```@meta
CurrentModule = GeometricSolvers
```

# API

## The problem

A problem holds functions and, for a [`NonlinearProblem`](@ref), an optional Jacobian prototype:
the residual, optionally its derivative, and optionally `jacobian_prototype` for a sparse R3
Jacobian in [`NonlinearProblem`](@ref); the residual, optionally its derivative, and its length in
[`LeastSquaresProblem`](@ref); or the scalar action, its equation and unknown index sets, and an
optional datum in [`StationarityProblem`](@ref). [`Reparametrized`](@ref) wraps a
`NonlinearProblem` or a `StationarityProblem` in a coordinate chart `x = g(y, p)`, and
[`Deflated`](@ref) wraps either in a deflation of the roots it is given, so that a solve from the
same start converges to another root. [`BlockEliminated`](@ref) partitions the unknowns of a
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

## The storage seam

The storage seam lets an algorithm read each function's result rather than the storage itself,
so the storage-specific operations stay at the leaves. Each function has a value method for a
`StaticArray` or a `Number` iterate (R1) and an in-place method for an `AbstractArray` buffer slot
(R3).

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
buffer
residual!!
update_iterate!!
map!!
mul!!
copy!!
scale!!
axpby!!
regularize!!
```

## Status and stopping test

Every type on this page is `isbits`, so a kernel can hold it, and every real field is in
`R = real(T)` for the element type `T` of the iterate. [`SolverStatus`](@ref) also carries the
range and spectrum diagnostics (`rank`, `nullnorm` and `gradratio`) recorded at the iterate
``x_k`` of the last fresh [`Newton`](@ref) factorisation, not at the iterate ``x`` at return; a
solve of a [`LeastSquaresProblem`](@ref) that ends on the stationarity test has factorised afresh
at the iterate it returns.
With `regularization = μ ≠ 0`, `rank` and `nullnorm` describe ``J(x_k) + μ I`` and `gradratio`
describes ``J(x_k)``, so the classification below holds for `μ = 0` only.
For `n ≥ 2` (for `n = 1`, a nonzero scalar Jacobian always gives `gradratio = 1`, and `rank = 1`
with an SVD factor), they classify the outcome and a `STALLED` or `MAXITERS` return:
a consistent rank-deficient system that converged has a small residual (`fnorm` small) and nothing outside the range
(`rank < n`, `nullnorm ≈ 0`); a least-squares point with no root has `gradratio ≈ 0` (`fnorm > 0`)
and the residual outside the range (`rank < n`, ``\text{nullnorm} \approx \|F(x_k)\| > 0``); a
stall in nearly singular directions has the residual inside the range (`nullnorm ≈ 0`) at a fine
SVD cutoff and outside it (`rank < n`, ``\text{nullnorm} \approx \|F(x_k)\|``) at a coarse one;
and an in-range stall from conditioning or step acceptance has `gradratio` bounded away from zero
with `nullnorm ≈ 0`.

```@docs
ReturnCode
sciml_retcode
SolverStatus
stopping_test
StepInfo
Options
Options(::Type{T}) where {T <: Number}
```

## Linear-solver methods

The factorisation precision `TF` of a linear-solver method is a type parameter. `Nothing` selects
the working type of the problem, a real float type a lower (or higher) precision, and a marker a
vendor tensor-core path. These names are exported.

```@docs
LUFactorization
LDLTFactorization
QRFactorization
SVDFactorization
TF32
FP16
BF16
```

### Factorisation and solve

A linear-solver method enters a solve through [`init_linear`](@ref) (and [`dense_factor`](@ref)),
[`factorize!!`](@ref) and [`ldiv!!`](@ref). For a static matrix (R1) [`factorize!!`](@ref) and
[`ldiv!!`](@ref) return new `isbits` values, so they run inside a kernel: [`StaticLU`](@ref) is LU
with partial pivoting and `Int32` pivots, [`StaticLDLT`](@ref) is Bunch–Kaufman `L D Lᵀ` /
`L D Lᴴ` with `1×1` and `2×2` diagonal blocks, [`StaticQR`](@ref) and [`StaticRectQR`](@ref) are
Householder QR without column pivoting, and [`StaticSVD`](@ref) and [`StaticRectSVD`](@ref) are a
one-sided Jacobi singular value decomposition whose solve truncates at the rank cutoff
`√n · eps(TF) · σ₁` for a square matrix and `√m · eps(TF) · σ₁` for a rectangular one, `m` its row
count, which is what makes a rank-revealing solve finite. For a vector state (R3) with no sparse
pattern (`jacobian_pattern(prob, jacprep) === nothing`),
[`init_linear`](@ref) pre-allocates [`DenseLU`](@ref), [`DenseLDLT`](@ref), [`DenseQR`](@ref) or
[`DenseSVD`](@ref) on a host `Array` and [`DeviceLU`](@ref), [`DeviceLDLT`](@ref),
[`DeviceQR`](@ref) or [`DeviceSVD`](@ref) on an `AbstractGPUArray`, and [`factorize!!`](@ref) and
[`ldiv!!`](@ref) reuse its buffers in place: through LAPACK on the host, and on a device through
the `GeometricSolversMetalExt` (`Metal` MPS `lu!` and `ldiv!` on `MtlMatrix{Float32}` and
`MtlVector{Float32}`, with every MPS failure status reported as a singular factor) and
`GeometricSolversCUDAExt` (`CUDA` cuSOLVER `getrf!`, `getrs!`, `geqrf!`, `ormqr!` with cuBLAS
`trsv!`, and `gesvdj!` on `CuMatrix{T}` and `CuVector{T}` for `T <:
LinearAlgebra.BlasFloat`, falling back to `ka_factorize_svd!` where `gesvdj!` does not converge)
package extensions where available and through `KernelAbstractions` otherwise. When `Sparspak` is
loaded and [`jacobian_pattern`](@ref) is a `SparseMatrixCSC{<:Any, Int}` on a host `Vector{T}` (`T <:
AbstractFloat`) — from an explicit `jacobian_prototype`, or detected automatically by
`GeometricSolversSparseADExt` when `SparseMatrixColorings` is loaded — the
`GeometricSolversSparspakExt` package extension pre-allocates a `SparspakLU`
workspace and unfactorised `Sparspak.SparseSolver` in `init_linear` for an `LUFactorization` whose
factor element type is `T` (`LUFactorization()` or `LUFactorization(T)`); the first `factorize!!`
call runs symbolic ordering and symbolic factorisation, and subsequent `factorize!!` and `ldiv!!`
calls reuse them across Newton steps. That configuration is the only one with a sparse factor, and
[`has_sparse_factor`](@ref) names it. A detected pattern is used only where `has_sparse_factor`
holds and, under a `Newton` `regularization ≠ 0`, only where the pattern holds the whole structural
diagonal. Everywhere else — `QRFactorization`, `SVDFactorization`, mixed precision such as
`LUFactorization(Float64)` on a `Float32` iterate, a complex iterate, or no `Sparspak` — a
detected pattern gives the dense factor and a dense Jacobian buffer, as a solve with no pattern
does. An explicit `jacobian_prototype` outside that configuration raises an error in `init_linear`
or at the first `factorize!!`. A [`LeastSquaresProblem`](@ref) takes no detected pattern, because
`SparspakLU` is square: its `m × n` factor is dense.
[`factorize!!`](@ref) states what a singular matrix gives: an SVD factor records its numerical rank
in `rank` apart from its status, and [`exact_solve`](@ref) tells `Newton` whether the solve is
exact or truncated; [`singular_values`](@ref) gives the singular values of a [`StaticSVD`](@ref)
or [`StaticRectSVD`](@ref). These names are not exported and are reached as
`GeometricSolvers.name`.

```@docs
init_linear
has_sparse_factor
dense_factor
factorize!!
ldiv!!
factor_eltype
StaticLU
StaticLDLT
StaticQR
StaticRectQR
StaticSVD
StaticRectSVD
DenseLU
DeviceLU
DenseLDLT
DeviceLDLT
DenseQR
DeviceQR
DenseSVD
DeviceSVD
returncode
exact_solve
singular_values
```

### Matrix-free Krylov solves

[`GMRES`](@ref) is a [`LinearMethod`](@ref) with no factor. The Jacobian enters a step only through
a Jacobian–vector product, so a [`Newton`](@ref) with it holds no `n × n` Jacobian (`state.J ===
nothing`) and serves a system that no array can hold the Jacobian of: a small system whose every
column costs a full evaluation of an expensive residual, or a device array. Its step solves
``J M^{-1} u = -F(x)`` and returns ``d = M^{-1} u`` to the relative tolerance that the forcing term
returns.

The forcing term sets that tolerance. [`EisenstatWalkerChoice2`](@ref) is the default, and
[`EisenstatWalkerChoice1`](@ref) and [`FixedForcing`](@ref) are the alternatives; a term is an
[`AbstractForcingTerm`](@ref), and [`compute_forcing`](@ref) is its rule. The right preconditioner
of `GMRES(; preconditioner = P)` is refreshed by [`refresh!`](@ref) once per Newton step and
applied by [`apply_preconditioner!`](@ref) before every product. A matrix-free problem supplies its
own product as [`JVPFunction`](@ref)`(jvp!)` in the `jacobian` field of [`Newton`](@ref), and the
solve reports the exact counts of its residual and product evaluations in
[`f_calls`](@ref)/[`jvp_calls`](@ref) and in the `f_calls` and `jvp_calls` fields of
[`SolverStatus`](@ref). A combinator other than [`Repeat`](@ref) reports `0` in those fields,
because its members count only their own steps. `GMRES(; maxiter)` caps the inner iterations of
one Newton step, and
[`max_inner_iterations`](@ref) gives the largest count of the last solve.

The matrix-free [`DogLeg`](@ref) runs on the Arnoldi relation of an unrestarted `GMRES` only
(`restart == maxiter`), and a `Newton` with a preconditioner or a restarted `GMRES` under it raises
an `ArgumentError`. These names are exported.

```@docs
GMRES
AbstractForcingTerm
EisenstatWalkerChoice2
EisenstatWalkerChoice1
FixedForcing
JVPFunction
refresh!
apply_preconditioner!
f_calls
jvp_calls
```

### Eigensolver

[`jacobi_eigen`](@ref) is a function on a matrix and not a linear-solver method: cyclic two-sided
Jacobi for a Hermitian or a real symmetric matrix, on the rotations and the round-robin ordering of
the Jacobi SVD with a Frobenius-norm pair floor `eps(R) * ‖H‖_F`, for the case
`jacobi_eigen(im * M)` of a real skew `M` that the `ProjectedSkew` exponential algorithm of
GeometricOptimizers needs. It adds no method to `LinearAlgebra.eigen`, because a method of `eigen`
for a `StaticArrays` type would be type piracy.

```@docs
jacobi_eigen
```

## Line searches

A line search is scalar code: [`linesearch`](@ref) sees the line only through the merit
[`φ`](@ref) and its derivative [`φ′`](@ref), allocates nothing and never throws, so the same
search runs on the host and inside a kernel. Its iteration cap, where it has one, is its own
field. Along a [`LineFunction`](@ref) its [`LineSearchResult`](@ref) also carries the residual at
the step, so the caller does not evaluate it again.

```@docs
Static
Backtracking
Bisection
StrongWolfe
```

[`DogLeg`](@ref) and [`LevenbergMarquardt`](@ref) are the trust-region and damped-step
globalizations of [`Newton`](@ref). Neither is a line search: each needs the Jacobian, its
factor and a parameter that lives across iterations (the trust-region radius or the damping
parameter `μ`), so `Newton` keeps that parameter in its state and runs the trial loop against
the factor itself.

```@docs
DogLeg
LevenbergMarquardt
```

## Nonlinear solvers

A nonlinear solver holds its algorithmic configuration (`linsolve`, `globalization`, `jacobian`,
`refinement`, `regularization` and `reuse` for [`Newton`](@ref); `damping` and `globalization` for
[`Picard`](@ref)) and nothing about the problem or the iterate. [`GaussNewton`](@ref) constructs a
[`Newton`](@ref) configured for [`LeastSquaresProblem`](@ref) (`linsolve = QRFactorization()`,
`globalization = Static()`). [`solve`](@ref) adapts the method
to `R = real(eltype(x₀))`, calls [`init_state`](@ref) and [`solve!!`](@ref) on a copy of `x₀` when
`x₀` is an array, and returns `(x, status)`; when `x₀` or `p` carries a `ForwardDiff.Dual` leaf,
[`solve`](@ref) on a [`NonlinearProblem`](@ref) strips duals via [`StripDual`](@ref), runs the
primal solve, and on `SUCCESS` differentiates the root implicitly at `x*`, while on a
[`StationarityProblem`](@ref) it raises an `ArgumentError`. With the same arguments,
`solver = init(prob, x₀, method, p; options, kwargs...)` pre-allocates a
[`NonlinearSolver`](@ref) (or a [`ReparametrizedSolver`](@ref) for a [`Reparametrized`](@ref)
problem) that holds a copy of an array `x₀` and `p`, `solve!(solver)` solves into
`solver.x` and returns `(solver.x, status)`, and `solve!(solver, x, p)` solves from another `x`
with other `p` of the same types, in place into an array `x`. `init`, `solve!` and `solve` are the
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
Newton
GaussNewton
QuasiNewton
Picard
NoRefinement
Refinement
GMRESRefinement
```

### Mixed-precision refinement

A [`Newton`](@ref) step whose factorisation runs in a precision `TF` below the working precision
`T` of the iterate — `linsolve = LUFactorization(Float32)` on a `Float64` problem — corrects its
linear direction in `T`. [`Refinement`](@ref) is the LU-IR loop of Kelley 2023: the factor's
`TF` solve, the residual `-F(x) - J d` in `T`, and the correction that solve gives, to the
tolerance `max(η, √eps(T)) ‖F(x)‖`. [`GMRESRefinement`](@ref) adds the SDD variant of Amestoy et
al. 2024 in front of the refactorisation of the Jacobian in `T`: where the LU-IR passes fail, a
left-preconditioned GMRES in `T` solves for a correction with a copy of the same `Float32` factors
promoted to `T`, so the step reaches the accuracy of a factorisation in `T` without forming one
(§1.4). The remedy chain of a step is therefore `LU-IR -> GMRES-IR -> refactorisation in T`, and
[`SolverStatus`](@ref) counts each failed stage in `step_failures` and records a factorisation in
`T` in `promoted`.

The buffers of a mixed-precision solve are readable after it: [`stored_factors`](@ref) is the
packed `L` and `U` of the factorisation in `TF`, [`promoted_factors`](@ref) the working-precision
copy of it that the GMRES-IR stage promotes (or the factors of the refactorisation, where the step
promoted), [`preconditioned_buffer`](@ref) the slot of the preconditioned product the inner GMRES
forms, and [`linear_residual_buffer`](@ref) the slot of the linear residual `-F(x) - J d`.
`scripts/gmres_ir_comparison.jl` runs both methods on the four conditioning numbers of the plan,
prints their Newton iteration counts and linear residuals, and recomputes both from the raw output
of the solves.

```@docs
stored_factors
promoted_factors
preconditioned_buffer
linear_residual_buffer
```

### Combinators

[`Additive`](@ref), [`Multiplicative`](@ref), [`Repeat`](@ref), [`Switch`](@ref) and
[`Anderson`](@ref) compose nonlinear methods through the same [`init_state`](@ref),
[`start!!`](@ref), [`step!!`](@ref) and [`finish!!`](@ref) interface as [`Newton`](@ref) and
[`Picard`](@ref), so a combinator runs in R1, R2 and R3 alike. [`Additive`](@ref) combines member
steps from the same iterate with [`OptimalWeights`](@ref) or [`FixedWeights`](@ref),
[`Multiplicative`](@ref) and [`Repeat`](@ref) thread `(x, F(x))` sequentially across sub-steps
while [`sync_merit`](@ref) updates any active line-search baseline merit, [`Switch`](@ref)
transitions permanently from `first` to `second` once
``\|F(x)\| \le \text{threshold} \cdot \|F(x_0)\|``, and [`Anderson`](@ref) accelerates its member
with filtered Anderson mixing, Tikhonov regularisation, restarts and best-iterate restoration. A
combinator has no implicit differentiation: [`solve`](@ref) with a combinator method raises an
`ArgumentError` for a `ForwardDiff.Dual` in `x₀` or `p`.

```@docs
Additive
OptimalWeights
FixedWeights
Multiplicative
Repeat
Switch
Anderson
```

### SciMLBase extension

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

## Automatic differentiation

In R1, [`prepare_ad`](@ref) returns the `AutoForwardDiff()` back end itself for a `StaticVector`
or a `Number` iterate. [`jacobian!!`](@ref), [`gradient_hessian!!`](@ref) and [`jvp!!`](@ref) run
tagged ForwardDiff directly with that source. The source `nothing` calls the problem's own `J`; a
[`JacobianFunction`](@ref) calls a method's operator for `jacobian!!` only. A back end other than
`AutoForwardDiff` has no R1 method, and the call is a host `MethodError`.

In R3 the AD path is chosen once, by [`prepare_ad`](@ref), from the
back end and the storage of the iterate: `AutoForwardDiff()` goes through an own chunked forward
mode on every array type ([`ChunkedForwardDiff`](@ref) for [`NonlinearProblem`](@ref) and
[`ChunkedHessianForwardDiff`](@ref) for [`StationarityProblem`](@ref)), and any other back end goes
through [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) on a
CPU iterate ([`DIJacobian`](@ref) and [`DIStationarity`](@ref)), so every ADTypes back end works
with no line of source here. A device array iterate takes
`AutoForwardDiff()` only, because DifferentiationInterface's `jacobian!` raises a scalar-indexing
error on an `MtlArray`, a `CuArray` and a `JLArray`. When the problem supplies an analytic
3-argument `J(Jbuf, x, p)`, [`JacobianMatrixJVP`](@ref) evaluates `prob.J` into its pre-allocated
matrix buffer and multiplies by the direction in place. Only a [`LineSearch`](@ref) other than
[`Static`](@ref) and a [`Refinement`](@ref) along a reused factor take a JVP, so under any other
configuration the analytic `J` gets no JVP preparation and no second `m × n` buffer. A
`Refinement` on a step that factorises at `x` multiplies by the regularised Jacobian that the step
holds, in a buffer of its own beside the factor, which the factorisation overwrites.
[`jacobian!!`](@ref), [`gradient_hessian!!`](@ref) and [`jvp!!`](@ref) then dispatch on the
object the preparation returned, so the solver makes no runtime choice and carries no `ad`
keyword.

For a sparse R3 Jacobian on a host `Array` iterate, an `AutoSparse` back end takes either an
explicit `jacobian_prototype` on the [`NonlinearProblem`](@ref) (which [`prepare_ad`](@ref) wraps in
`KnownJacobianSparsityDetector`) or, when `SparseMatrixColorings` is loaded and
`jacobian_prototype === nothing`, a sparsity detector such as `TracerSparsityDetector()` of
`SparseConnectivityTracer` alongside a colouring algorithm such as `GreedyColoringAlgorithm()`. The
`GeometricSolversSparseADExt` package extension then extracts the detected pattern from the prepared
[`DIJacobian`](@ref) via [`jacobian_pattern`](@ref) as a `SparseMatrixCSC{<:Any, Int}` — a
`SparseMatrixCSC{Bool, Int}` for `TracerSparsityDetector` — so
[`init_linear`](@ref) allocates the sparse factor and Jacobian buffer without a user-supplied
prototype where [`has_sparse_factor`](@ref) holds, and the dense factor elsewhere; when an explicit
`jacobian_prototype` is also given, the explicit prototype takes precedence.

In R3 the iterate is a vector; another shape raises `ArgumentError` where the preparation is
built. The parameters may be replaced on every call, which is what lets one preparation serve a
whole time-stepping loop.

`AutoEnzyme` reaches the `Array` path through DifferentiationInterface like any other back end, but
it needs runtime activity for the parameter context —
`AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))`. A plain `AutoEnzyme()` raises
`EnzymeRuntimeActivityError` on a residual that broadcasts constant parameters into its result; see
`KNOWN_ISSUES.md`. The mode is the caller's choice, so the package sets none.

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
prepare_ad
prepare_jvp
DIJacobian
DIStationarity
ChunkedForwardDiff
ChunkedHessianForwardDiff
JacobianFunction
JacobianMatrixJVP
jacobian!!
gradient_hessian!!
jvp!!
```

## Internals

These names are not exported. Reach them as `GeometricSolvers.name`.

A nonlinear method enters the loop of [`solve!!`](@ref) through [`init_state`](@ref), which builds
its state, [`step!!`](@ref), and the two hooks [`start!!`](@ref) and [`finish!!`](@ref), which run
once before the first step and at every return. [`residual_buffer`](@ref) names the slot of the
state that holds `F(x₀)`, and [`refine!!`](@ref) refines the linear direction `d` in working
precision before globalization; [`gmres_ir!!`](@ref) is the GMRES-IR stage that follows a failed
pass of a [`GMRESRefinement`](@ref).

```@docs
init_state
residual_buffer
step!!
start!!
finish!!
refine!!
sync_merit
init_globalization
LineFunction
globalize!!
LineSearch
linesearch
φ
φ′
ExactStep
InexactStep
MeasuredSlope
LineSearchResult
roundoff
smallest_step
sufficient_decrease
classify
backtrack_step
zoom_step
LinearMethod
compute_forcing
max_inner_iterations
GMRESWorkspace
init_gmres_workspace
GMRESInfo
gmres_solve!
gmres_ir!!
promote_lu_factors!
promote_lu_factors
arnoldi_dot
R5Operator
R5JVP
CountingResidual
sat16
with_calls
jacobi_rotation
ToReal
StripDual
parameters
record
converged
isconverged
isstalled
norm2
rnorm
rdot
```
