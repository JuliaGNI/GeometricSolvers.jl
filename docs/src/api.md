```@meta
CurrentModule = GeometricSolvers
```

# API

## The problem

A problem holds functions and nothing else: the residual, and optionally its derivative. The
storage of the iterate selects the call form of both, so one problem serves every regime. This
name is exported.

```@docs
NonlinearProblem
```

## The storage seam

The storage seam lets an algorithm read each function's result rather than the storage itself,
so the storage-specific operations stay at the leaves. The value methods, for a `StaticArray` or
a `Number` iterate, are the ones this page documents.

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
buffer
residual!!
update_iterate!!
map!!
mul!!
```

## Status and stopping test

Every type on this page is `isbits`, so a kernel can hold it, and every real field is in
`R = real(T)` for the element type `T` of the iterate.

```@docs
ReturnCode
SolverStatus
StepInfo
Options
Options(::Type{T}) where {T <: Number}
```

## Linear-solver methods

The factorisation precision `TF` of a linear-solver method is a type parameter. `Nothing` selects
the working type of the problem, a real float type a lower (or higher) precision, and a marker a
vendor tensor-core path.

```@docs
LUFactorization
QRFactorization
SVDFactorization
TF32
FP16
BF16
```

### Factorisation and solve

A linear-solver method enters a solve through [`init`](@ref), [`factorize!!`](@ref) and
[`ldiv!!`](@ref). For a static matrix (R1) [`factorize!!`](@ref) and [`ldiv!!`](@ref) return new
`isbits` values, so they run inside a kernel: [`StaticLU`](@ref) is LU with partial pivoting and
`Int32` pivots, [`StaticQR`](@ref) Householder QR without column pivoting, and [`StaticSVD`](@ref)
a one-sided Jacobi singular value decomposition whose solve truncates at the rank cutoff
`√n · eps(TF) · σ₁`, which is what makes a rank-revealing solve finite. For a vector state (R3)
[`init`](@ref) pre-allocates [`DenseLU`](@ref), [`DenseQR`](@ref) or [`DenseSVD`](@ref), and
[`factorize!!`](@ref) and [`ldiv!!`](@ref) reuse its buffers in place. [`factorize!!`](@ref) states
what a singular matrix gives. These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
init
factorize!!
ldiv!!
factor_eltype
StaticLU
StaticQR
StaticSVD
DenseLU
DenseQR
DenseSVD
returncode
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

A trust region is the other globalization of [`Newton`](@ref). [`DogLeg`](@ref) is no line search:
it needs the Jacobian, its factor and a radius that lives across iterations, so `Newton` keeps the
radius in its state and runs the trial loop on the dogleg curve itself.

```@docs
DogLeg
```

## Nonlinear solvers

A nonlinear solver holds its algorithmic configuration (`linsolve`, `globalization`, `jacobian`,
`refinement`, `regularization` and `reuse` for [`Newton`](@ref); `damping` and `globalization` for
[`Picard`](@ref)) and nothing about the problem or the iterate. [`solve`](@ref) adapts the method
to `R = real(eltype(x₀))`, calls [`init`](@ref) and [`solve!!`](@ref), and returns `(x, status)`.
[`solve!!`](@ref) takes a solver state from [`init`](@ref) and runs the loop on a `StaticVector`,
`MVector` or `Number` iterate in R1. [`solve!`](@ref) solves a batch of independent small systems in
place on `get_backend(X)` with one work-item per system, and [`PerSystem`](@ref) wraps an indexable
collection of per-system parameters so system `i` receives `p.data[i]`.

```@docs
solve
solve!
solve!!
PerSystem
Newton
Picard
NoRefinement
```

## Automatic differentiation

In R1, [`prepare_ad`](@ref) returns the `AutoForwardDiff()` back end itself for a `StaticVector`
or a `Number` iterate. [`jacobian!!`](@ref) and [`jvp!!`](@ref) run tagged ForwardDiff directly
with that source. The source `nothing` calls the problem's own `J`; a [`JacobianFunction`](@ref)
calls a method's operator for `jacobian!!` only. A back end other than `AutoForwardDiff` has no
R1 method, and the call is a host `MethodError`.

In R3 the AD path is chosen once, by [`prepare_ad`](@ref), from the
back end and the storage of the iterate: `AutoForwardDiff()` goes through an own chunked forward
mode on every array type, and any other back end goes through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) on a CPU
iterate, so every ADTypes back end works with no line of source here. A device array iterate takes
`AutoForwardDiff()` only, because DifferentiationInterface's `jacobian!` raises a scalar-indexing
error on an `MtlArray`, a `CuArray` and a `JLArray`. [`jacobian!!`](@ref) and [`jvp!!`](@ref) then
dispatch on the object the preparation returned, so the solver makes no runtime choice and carries
no `ad` keyword.

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
DIJacobian
ChunkedForwardDiff
JacobianFunction
jacobian!!
jvp!!
```

## Internals

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
step!!
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
jacobi_rotation
ToReal
parameters
record
converged
isconverged
isstalled
norm2
rnorm
rdot
```
