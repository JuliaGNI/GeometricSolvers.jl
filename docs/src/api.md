```@meta
CurrentModule = GeometricSolvers
```

# API

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

The Jacobian seam of the R3 solver. The AD path is chosen once, by [`prepare_ad`](@ref), from the
storage of the iterate: an `Array` goes through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl), so every
ADTypes back end works with no line of source here, and a device array goes through an own chunked
forward mode, because DifferentiationInterface's `jacobian!` raises a scalar-indexing error on an
`MtlArray`, a `CuArray` and a `JLArray`. [`jacobian!!`](@ref) and [`jvp!!`](@ref) then dispatch on
the object the preparation returned, so the solver makes no runtime choice and carries no `ad`
keyword.

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
prepare_ad
DIJacobian
ChunkedForwardDiff
ChunkedForwardDiff(::AutoForwardDiff, ::Any, ::AbstractArray, ::AbstractArray)
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
