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

Every algorithm of the package — the loop, `Newton`, `Picard`, the globalisations, `DogLeg`, the
combinators and `Anderson` — touches storage only through the functions below, and reads their
result rather than the storage itself, so one algorithm serves every storage kind and the
difference between the kinds stays at the leaves. The value methods, for a `StaticArray` or a
`Number` iterate, are the ones this page documents.

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

A linear-solver method enters a solve through two functions, [`factorize!!`](@ref) and
[`ldiv!!`](@ref). For a static matrix (R1) both return new `isbits` values, so they run inside a
kernel: [`StaticLU`](@ref) is LU with partial pivoting and `Int32` pivots, [`StaticQR`](@ref)
Householder QR without column pivoting. [`factorize!!`](@ref) states what a singular matrix
gives. These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
factorize!!
ldiv!!
factor_eltype
StaticLU
StaticQR
returncode
```

## Line searches

A line search is scalar code: [`linesearch`](@ref) sees the line only through the merit
[`φ`](@ref) and its derivative [`φ′`](@ref), allocates nothing and never throws, so the same
search runs on the host and inside a kernel. Its iteration cap, where it has one, is its own
field.

```@docs
Static
Backtracking
Bisection
StrongWolfe
```

## Automatic differentiation

The Jacobian seam. In R1 there is no preparation object to build: a dual on a static array is an
`isbits` value, so [`jacobian!!`](@ref) and [`jvp!!`](@ref) take the `AutoForwardDiff()` back end
itself — or the problem's own `J`, or a method's operator — and run tagged ForwardDiff directly. A
back end other than `AutoForwardDiff` has no R1 method, and the call is a host `MethodError`.

In R3 the AD path is chosen once, by [`prepare_ad`](@ref), from the
back end and the storage of the iterate: `AutoForwardDiff()` goes through an own chunked forward
mode on every array type, and any other back end goes through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) on a CPU
iterate, so every ADTypes back end works with no line of source here. A device array iterate takes
`AutoForwardDiff()` only, because DifferentiationInterface's `jacobian!` raises a scalar-indexing
error on an `MtlArray`, a `CuArray` and a `JLArray`. [`jacobian!!`](@ref) and [`jvp!!`](@ref) then
dispatch on the object the preparation returned, so the solver makes no runtime choice and carries
no `ad` keyword.

The iterate is a vector; another shape raises `ArgumentError` where the preparation is built. The
parameters may be replaced on every call, which is what lets one solver serve a whole
time-stepping loop.

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
ToReal
record
converged
isconverged
isstalled
norm2
rnorm
rdot
```
