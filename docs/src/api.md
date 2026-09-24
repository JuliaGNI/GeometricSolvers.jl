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

## Internals

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
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
