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

A solve takes a problem, an initial iterate `x₀`, a method and the parameters `p`. The entry point
and the storage of `x₀` select the regime of the solve; the problem has no flag for it.
`solve_batch!` always solves a batch (R2): it treats each entry of its array as a system of its
own. `solve` takes R1 for a `Number` or an `SVector` and R3 for any other vector, so a
`Vector{Float64}` is one system of `n` unknowns for `solve` and `n` scalar systems for
`solve_batch!`.

| regime | iterate `x₀` | entry point | where it runs |
|:--|:--|:--|:--|
| **R1** value semantics | a `Number` or an `SVector` | `solve` | on the host, or in one GPU thread |
| **R2** batched | an array of `Number` or `SVector`, one entry per system | `solve_batch!` | one work-item per system, on the CPU or a GPU |
| **R3** in place | an `AbstractVector` that is not static | `solve`, `init`, `solve!` | on the CPU or on a device array |

In R1 every operation returns a new value. The problem, the solver state and the status are
`isbits` when the functions of the problem are, so a kernel can hold them. R2 runs the R1 loop in
one kernel launch, one work-item per system. In R3 the residual and the Jacobian write into
buffers that `init` allocates once.

An R3 iterate with a Krylov linear solver, `GMRES`, `BiCGStab`, `DQGMRES` or `MINRESQLP`, gives
the **matrix-free** regime: the solve holds no Jacobian matrix, and each inner iteration takes one
Jacobian–vector product, two for `BiCGStab`. See [The three regimes](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/manual/regimes/)
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
  methods, and the matrix-free `GMRES`, `BiCGStab`, `DQGMRES` and `MINRESQLP`.
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

`.githooks` holds two hooks, `pre-commit` and `pre-push`. Each is a shell launcher that runs its
Julia program, `pre-commit.jl` or `pre-push.jl`; `test-layout.jl` holds the test-layout check of
`pre-commit`. The hooks are **not active in a fresh clone**, because `core.hooksPath` is local
configuration and is not part of a clone. Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** runs when a commit stages a `.jl` file, a `.jl` deletion included, or changes
anything below `test/`. Any other commit passes at once. It refuses a staged `.jl` file that also
has unstaged changes, because the stages read the files on disk. Then it runs these stages:

- **Test layout**, on a commit that changes `test/`: **blocks** when the tests do not follow the
  test convention of the shared hook set.
- **JuliaFormatter**, on the staged `.jl` files, with this repository's `.JuliaFormatter.toml`:
  **blocks** a file that is not formatted. It needs JuliaFormatter 2.14 or newer, which it looks
  for first in the Julia environment of the hook set, which `RESEARCH_HARNESS_JULIA` can name, and
  then on Julia's load path; with none, or with an older one found first, every file fails.
- **`fatou lint`**, on the staged `.jl` files, when `fatou` is installed: **advisory**, because
  some of its findings are wrong.
- **Unicode NFC**, on the staged `.jl` files: **blocks** a file that is not NFC-normalised. A
  pattern typed in NFC matches nothing in a decomposed file. The invariant is
  `s == Unicode.normalize(s, :NFC)`: `q̇`, `v̄` and `f̄` have no precomposed codepoint and stay two
  codepoints. The script `nfc.jl` of the shared hook set fixes a file:
  `julia --startup-file=no nfc.jl --apply <files>`.
- **ast-grep**, on the staged `.jl` files, with the rules of an `sgconfig.yml` in the repository
  root or a directory above it: **blocks** on a hit of severity `error` on a line that the commit
  adds, and is **advisory** for a hit of severity `error` on another line and for a hit of
  severity `warning` or `hint`. Without `ast-grep` or an `sgconfig.yml`, the stage is skipped.
- **Package load**, when a `.jl` file is staged or deleted: **blocks** when `using GeometricSolvers`
  fails in a new process, such as on a syntax error or a broken `include`.

**`pre-push`** runs the full test suite, `Pkg.test` with `--check-bounds=auto`, **only when the
push updates `main` or `master`**. A push to another branch leaves the suite to CI. The suite
runs for a long time and can print little while it runs; that is not a network hang. An
interrupt stops the suite's processes, and the hook reports any process that does not stop.

Either hook can be bypassed for one command with `--no-verify`, for a change that it does not
apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from the shared hook set in
[michakraus/ResearchHarness](https://github.com/michakraus/ResearchHarness) (`githooks/`, and
`scripts/test-layout.jl`), and are identical across the related repositories. Edit them there:
the next install overwrites a local edit.
