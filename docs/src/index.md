```@meta
CurrentModule = GeometricSolvers
```

# GeometricSolvers

Documentation for [GeometricSolvers](https://github.com/JuliaGNI/GeometricSolvers.jl).

```@docs
GeometricSolvers
```

GeometricSolvers provides line searches, linear solvers and nonlinear solvers that run on the CPU
and on CUDA, ROCm and Metal devices through KernelAbstractions. One solver loop serves three
regimes, which the storage of the initial iterate selects:

- **R1**: a `Number` or an `SVector`, with value semantics, on the host or in one GPU thread,
- **R2**: a batch of small systems, one work-item per system on the CPU or a GPU, through
  [`solve_batch!`](@ref),
- **R3**: an in-place `AbstractVector`, on the CPU or on a device array.

```jldoctest
julia> using GeometricSolvers

julia> x, status = solve(NonlinearProblem((x, p) -> x^2 - p), 1.0, Newton(), 2.0);

julia> x ≈ sqrt(2) && status.code === SUCCESS
true
```

## Contents of this documentation

**Manual**

- [Getting started](@ref): problems, call forms, `solve`, `init` and `solve!`, options,
  status and return codes, parameters, and the SciML extension.
- [The three regimes](@ref regimes): R1, R2 and R3, and how to choose one.
- [Methods by regime](@ref guide): the methods that run in R1, R2, R3 and the matrix-free R5,
  and the conditions of each.
- [Problem types](@ref problems): least squares, stationarity, reparametrization, deflation and
  elimination.

**Algorithms**

- [Nonlinear methods](@ref nonlinear): [`Newton`](@ref), [`GaussNewton`](@ref),
  [`Picard`](@ref), factor reuse and implicit differentiation.
- [Globalization](@ref globalization): line searches, the dogleg trust region and
  Levenberg–Marquardt.
- [Linear solvers](@ref linear): static, dense, device and sparse factorisations, and the
  matrix-free [`GMRES`](@ref).
- [Automatic differentiation](@ref ad): ForwardDiff, DifferentiationInterface and sparse
  Jacobians.
- [Composition and preconditioning](@ref composition): combinators, Anderson and Broyden
  acceleration, restricted additive Schwarz and RASPEN.

**Architecture**

- [CPU and GPU backends](@ref backends): the storage and reduction seams, dispatch, kernel-safe
  types, vendor libraries and device precision.
- [Mixed precision](@ref mixed): the factorisation precision, the scaled solve and iterative
  refinement.
- [Numerical Stability and Backward-Error Bounds](@ref stability): the bounds of each
  factorisation that the test suite verifies.

The API section documents every name, starting with [Problems and solves](@ref).

```@index
```
