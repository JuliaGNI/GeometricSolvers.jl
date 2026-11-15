```@meta
CurrentModule = GeometricSolvers
```

# Nonlinear solvers

A nonlinear solver holds its algorithmic configuration (`linsolve`, `globalization`, `jacobian`,
`refinement`, `regularization` and `reuse` for [`Newton`](@ref); `damping` and `globalization` for
[`Picard`](@ref)) and nothing about the problem or the iterate. [`GaussNewton`](@ref) constructs a
[`Newton`](@ref) configured for [`LeastSquaresProblem`](@ref) (`linsolve = QRFactorization()`,
`globalization = Static()`). The section [The solve interface](@ref) describes how a method enters
a solve.

```@docs
Newton
GaussNewton
QuasiNewton
Picard
NoRefinement
Refinement
GMRESRefinement
```

## Mixed-precision refinement

A [`Newton`](@ref) step whose factorisation runs in a precision `TF` below the working precision
`T` of the iterate — `linsolve = LUFactorization(Float32)` on a `Float64` problem — corrects its
linear direction in `T`. [`Refinement`](@ref) is the LU-IR loop of Kelley 2023 (*Newton's Method in Three
Precisions*, [doi:10.61208/pjo-2023-030](https://doi.org/10.61208/pjo-2023-030)): the factor's
`TF` solve, the residual `-F(x) - J d` in `T`, and the correction that solve gives, to the
tolerance `max(η, √eps(T)) ‖F(x)‖`. [`GMRESRefinement`](@ref) adds the SDD variant of Amestoy et
al. 2024 in front of the refactorisation of the Jacobian in `T`: where the LU-IR passes fail, a
left-preconditioned GMRES in `T` solves for a correction with a copy of the same `Float32` factors
promoted to `T`, so the step reaches the accuracy of a factorisation in `T` without forming one.
The remedy chain of a step is therefore `LU-IR -> GMRES-IR -> refactorisation in T`, and
[`SolverStatus`](@ref) counts each failed stage in `step_failures` and records a factorisation in
`T` in `promoted`.

The buffers of a mixed-precision solve are readable after it: [`stored_factors`](@ref) is the
packed `L` and `U` of the factorisation in `TF`, [`promoted_factors`](@ref) the working-precision
copy of it that the GMRES-IR stage promotes (or the factors of the refactorisation, where the step
promoted), [`preconditioned_buffer`](@ref) the slot of the preconditioned product the inner GMRES
forms, and [`linear_residual_buffer`](@ref) the slot of the linear residual `-F(x) - J d`.
`scripts/gmres_ir_comparison.jl` runs both methods at `κ∞ ∈ {1e4, 5e7, 2e8, 5e10}`, prints their
Newton iteration counts and linear residuals, and recomputes both from the raw output of the
solves.

```@docs
stored_factors
promoted_factors
preconditioned_buffer
linear_residual_buffer
```
