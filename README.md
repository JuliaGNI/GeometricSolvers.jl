# GeometricSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/GeometricSolvers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGNI/GeometricSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/GeometricSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/GeometricSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/GeometricSolvers.jl)

Line searches, linear and nonlinear solvers for CPUs and GPUs, based on KernelAbstractions.

GeometricSolvers solves nonlinear systems `F(x, p) = 0` and nonlinear least-squares problems. The
same solver loop runs on the CPU and on CUDA, ROCm and Metal devices: the device code is written
once with [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl). The package
collects many methods behind one interface, so that they can be compared on the same problem. The
[documentation](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/) gives the conditions under
which each method applies.

**Status.** The package is pre-1.0: a minor release can change the interface. The
[CHANGELOG](CHANGELOG.md) names each change.

## Overview

### Three regimes

A solve takes a problem, an initial iterate `x₀`, a method and the parameters `p`. The storage of
`x₀` selects the regime of the solve. The problem has no flag for it.

| regime | iterate `x₀` | entry point | where it runs |
|:--|:--|:--|:--|
| **R1** value semantics | a `Number` or an `SVector` | `solve` | on the host, or in one GPU thread |
| **R2** batched | an array of `Number` or `SVector`, one entry per system | `solve_batch!` | one work-item per system, on the CPU or a GPU |
| **R3** in place | an `AbstractVector` that is not static | `solve`, `init`, `solve!` | on the CPU or on a device array |

In R1 every operation returns a new value. The problem, the solver state and the status are
`isbits` when the functions of the problem are, so a kernel can hold them. R2 runs the R1 loop in
one kernel launch, one work-item per system. In R3 the residual and the Jacobian write into
buffers that `init` allocates once.

An R3 iterate with a Krylov linear solver, `GMRES` or `MINRESQLP`, gives the **matrix-free**
regime: the solve holds no Jacobian matrix, and each inner iteration takes one Jacobian–vector
product. See [The three regimes](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/manual/regimes/)
and [Methods by regime](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/guide/).

### Methods

- **Nonlinear methods**: `Newton`, `QuasiNewton` (Jacobian reuse), `GaussNewton` and `Picard`,
  with iterative refinement in a lower factorisation precision (`Refinement`,
  `GMRESRefinement`).
- **Globalization**: the line searches `Static`, `Backtracking`, `Bisection` and `StrongWolfe`,
  the trust regions `DogLeg` and `Steihaug`, and `LevenbergMarquardt`.
- **Linear solvers**: the LU, LDLT, Cholesky, QR and SVD factorisations (`LUFactorization`,
  `LDLTFactorization`, `CholeskyFactorization`, `QRFactorization`, `SVDFactorization`) on static,
  dense host and device matrices, sparse LU (`UMFPACKFactorization`, and Sparspak or cuDSS through
  `LUFactorization`), `KroneckerFactorization` for the stage systems of implicit Runge–Kutta
  methods, and the matrix-free `GMRES` and `MINRESQLP`.
- **Automatic differentiation**: ForwardDiff by default, other ADTypes back ends through
  DifferentiationInterface on a host array, sparse Jacobians by colouring, and a user-supplied
  Jacobian or Jacobian–vector product (`JVPFunction`).
- **Composition and acceleration**: the combinators `Additive`, `Multiplicative`, `Repeat`,
  `Switch` and `Fallback`, `Anderson` and `Broyden` acceleration, restricted additive Schwarz
  (`RAS`) and nonlinear left preconditioning (`LeftPreconditioned`).
- **Problem types**: `NonlinearProblem`, `LeastSquaresProblem` and `StationarityProblem`, and the
  wrappers `Reparametrized`, `Deflated`, `BlockEliminated`, `VariableProjection` and
  `BatchEliminated`.

`solve` and `solve!` return the solution and a `SolverStatus`, whose `code` (`SUCCESS`,
`STALLED`, `MAXITERS`, `SINGULAR`, `NONFINITE`, `LINESEARCH_FAILED`) says why the solve ended;
`solve_batch!` writes one status per system. A solve does not throw for a failure that it
detects.

### Package extensions

The package depends on no vendor package. Each of these packages, when loaded, activates an
extension:

| package | adds |
|:--|:--|
| `CUDA`, `AMDGPU`, `Metal` | the vendor factorisations (cuSOLVER, rocSOLVER, Metal Performance Shaders) on device arrays |
| `CUDA` and `CUDSS` | the sparse LU of cuDSS |
| `Sparspak` | a sparse LU on the host |
| `SparseMatrixColorings` | the sparsity pattern that an `AutoSparse` detector finds, on a problem with no prototype |
| `SciMLBase` | `solve` and `init` for a `SciMLBase.NonlinearProblem` |
| `IntervalArithmetic` | `GeometricSolvers.certify`, an interval enclosure of a root |

## Installation

```julia
pkg> add GeometricSolvers
```

The examples below also use StaticArrays (`pkg> add StaticArrays`). To solve on a GPU, also add
and load the package of the device: `CUDA`, `AMDGPU` or `Metal`. Without one, every solve runs on
the CPU.

## Examples

Each block below is complete. The output under it is that of a run of the block in a fresh Julia
session.

### One scalar or static system (R1)

```julia
using GeometricSolvers, StaticArrays

# A scalar equation x² = p
x, status = solve(NonlinearProblem((x, p) -> x^2 - p), 1.0, Newton(), 2.0)
@show x ≈ sqrt(2) status.code === SUCCESS

# A static system: the Jacobian comes from ForwardDiff and is factorised as an SMatrix
F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2])
x, status = solve(NonlinearProblem(F), SVector(1.0, 0.5), Newton(), 2.0)
@show x ≈ SVector(1.0, 1.0) status.code === SUCCESS
```

```text
x ≈ sqrt(2) = true
status.code === SUCCESS = true
x ≈ SVector(1.0, 1.0) = true
status.code === SUCCESS = true
```

### An in-place system with a line search (R3)

```julia
using GeometricSolvers

# atan(x - p) = 0, componentwise. From this start the full Newton step overshoots,
# and Backtracking shortens it until the merit ‖F‖² decreases.
F!(y, x, p) = (y .= atan.(x .- p); y)

method = Newton(; globalization = Backtracking())
solver = init(NonlinearProblem(F!), [3.0, 4.0, 5.0], method, [1.0, 2.0, 3.0])
x, status = solve!(solver)
@show x ≈ [1.0, 2.0, 3.0] status.code === SUCCESS

# The solver holds its buffers: solve again from another start with other parameters
x, status = solve!(solver, [2.0, 3.0, 4.0], [4.0, 5.0, 6.0])
@show x ≈ [4.0, 5.0, 6.0] status.code === SUCCESS
```

```text
x ≈ [1.0, 2.0, 3.0] = true
status.code === SUCCESS = true
x ≈ [4.0, 5.0, 6.0] = true
status.code === SUCCESS = true
```

### A batch of small systems (R2)

`solve_batch!` overwrites each initial guess with its solution. `PerSystem` gives each system its
own parameters. On a `Vector` the kernel runs on the KernelAbstractions `CPU()` backend.

```julia
using GeometricSolvers, StaticArrays

F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2])
prob = NonlinearProblem(F)

X = [SVector(1.0, 0.5), SVector(2.0, 1.0), SVector(0.5, 0.1)]   # one initial guess per system
S = Vector{SolverStatus{Float64}}(undef, length(X))             # one status per system

solve_batch!(X, S, prob, Newton(), PerSystem([2.0, 8.0, 0.5]))
@show X ≈ [SVector(1.0, 1.0), SVector(2.0, 2.0), SVector(0.5, 0.5)]
@show batch_summary(S).all_converged
```

```text
X ≈ [SVector(1.0, 1.0), SVector(2.0, 2.0), SVector(0.5, 0.5)] = true
(batch_summary(S)).all_converged = true
```

### Nonlinear least squares

A `LeastSquaresProblem` minimises `‖F(x, p)‖²/2` with at least as many residuals as unknowns.
`GaussNewton` takes the Gauss–Newton step; `LevenbergMarquardt` damps it.

```julia
using GeometricSolvers, StaticArrays

# Fit y ≈ a exp(b t) to four data points: four residuals, two unknowns
t = SVector(0.0, 1.0, 2.0, 3.0)
y = SVector(1.0, 2.7, 7.4, 20.1)
r(x, p) = x[1] .* exp.(x[2] .* p[1]) .- p[2]
prob = LeastSquaresProblem(r)

x, status = solve(prob, SVector(1.0, 0.9), GaussNewton(), (t, y))
@show status.code === SUCCESS && status.stationary
@show isapprox(x, SVector(1.0, 1.0); atol = 1e-2)

method = Newton(; linsolve = QRFactorization(), globalization = LevenbergMarquardt())
x, status = solve(prob, SVector(1.0, 0.9), method, (t, y))
@show status.code === SUCCESS && status.stationary
@show isapprox(x, SVector(1.0, 1.0); atol = 1e-2)
```

```text
status.code === SUCCESS && status.stationary = true
isapprox(x, SVector(1.0, 1.0); atol = 0.01) = true
status.code === SUCCESS && status.stationary = true
isapprox(x, SVector(1.0, 1.0); atol = 0.01) = true
```

### Matrix-free Newton–GMRES

With `linsolve = GMRES()` the solve forms no Jacobian matrix. The forcing term sets the tolerance
of each inner solve.

```julia
using GeometricSolvers

# A discrete nonlinear boundary-value problem -u'' + u³ = p on n interior points, u = 0 at the ends
function F!(y, u, p)
    n = length(u)
    h = 1 / (n + 1)
    for i in eachindex(u)
        left = i > 1 ? u[i - 1] : zero(eltype(u))
        right = i < n ? u[i + 1] : zero(eltype(u))
        y[i] = (2u[i] - left - right) / h^2 + u[i]^3 - p
    end
    return y
end

# No Jacobian matrix: each GMRES iteration takes one Jacobian–vector product by ForwardDiff
method = Newton(; linsolve = GMRES(; maxiter = 200, forcing = FixedForcing(1e-8)))
solver = init(NonlinearProblem(F!), zeros(500), method, 1.0)
u, status = solve!(solver)

y = similar(u)
@show status.code === SUCCESS
@show maximum(abs, F!(y, u, 1.0)) < 1e-6
@show jvp_calls(solver) > 0
```

```text
status.code === SUCCESS = true
maximum(abs, F!(y, u, 1.0)) < 1.0e-6 = true
jvp_calls(solver) > 0 = true
```

### A batch on a GPU

This block needs a GPU and its package. It was not run for this README. On Metal, the element type
is `Float32`.

```julia
using GeometricSolvers, StaticArrays, CUDA        # or Metal, AMDGPU

F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2])
prob = NonlinearProblem(F)

n = 10^6
X = CuArray(fill(SVector(1.0f0, 0.5f0), n))      # MtlArray, ROCArray
S = similar(X, SolverStatus{Float32})
p = PerSystem(CuArray(rand(Float32, n) .+ 1))

solve_batch!(X, S, prob, Newton(), p)
```

The [documentation](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/) has more examples: problem
types, factor reuse, mixed precision, composition and preconditioning.

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **Unicode NFC** — **blocks** a staged file that is not NFC-normalised. Sources here are NFC, so a
  file that is not is a regression: a pattern typed in NFC silently matches nothing in a decomposed
  file, which defeats `grep`, an editor search and Documenter's doctest comparison alike. Fix with
  `julia --startup-file=no ~/Research/Environment/Harness/githooks/nfc.jl --apply <files>`. The invariant
  is `s == Unicode.normalize(s, :NFC)`, not the absence of combining marks — `q̇`, `v̄` and `f̄` have
  no precomposed codepoint and are two codepoints in NFC as well.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
