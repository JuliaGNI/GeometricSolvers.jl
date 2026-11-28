# Comparison of four nonlinear solver packages

This benchmark runs GeometricSolvers and three other Julia packages for nonlinear systems on one
set of problems: SimpleSolvers, SimpleNonlinearSolve and NLsolve. It reports which package solves
which system, and whether the solutions agree. Each package runs with its own method that is
closest to each column of GeometricSolvers' standard test set. The report states the conditions of
each comparison; it ranks no package.

Three packages are left out:

- **NonlinearSolve** does not resolve in one environment with GeometricSolvers.
- **KINSOL**, through Sundials, does not resolve in one environment with GeometricSolvers either.
- **MINPACK** resolves, but its first call fails on aarch64 macOS with "cfunction: closures are not
  supported on this platform".

`benchmark/` is an environment of its own. Run each script from the package root with
`julia --project=benchmark benchmark/<script>.jl`, after `Pkg.instantiate()` in that environment.

| script | what it does |
|:--|:--|
| `correctness.jl [outdir]` | runs every cell, writes `correctness.csv` and `rows.csv` (by default to `benchmark/results/`), and prints the tables of this report |
| `broyden_update.jl [outdir]` | runs GeometricSolvers alone with `Broyden` and the column `Newton reuse 3 Broyden Backtracking LU`, writes `broyden_update.csv` and prints the table of section Y1's Broyden updates |
| `verify.jl <dir> [--structure] [--all-cards]` | checks the tables of `<dir>`, this report and the traces with code of its own |
| `selftest.jl` | tests the harness: the pairing, the label and the method of each call, the options of each package, the counting wrapper, the shared Jacobian, the measured `‖F‖∞`, the success rule, the agreement bound, the verdict, a throwing residual, the SVD solve of NLsolve on a non-finite matrix, and the element types and starts of the midpoint rows |
| `check_midpoint.jl` | checks each midpoint residual against one step of `ImplicitMidpoint` |
| `trace.jl <set> <problem> <start> <eltype> <column> [outfile]` | writes the iteration trace of one row |
| `devices.jl <backend> <machine> <dir>` | runs the device rows of one machine and writes `<dir>/devices-<machine>.csv` and `.txt` (section Y3) |

## Y1 · Correctness on the CPU

### Method

- **The rows.** The set `standard` holds the 67 problem × start pairs of the standard test set
  (`test/helpers/standard_problems.jl`), each from its start `u0`, `10u0` or `100u0`. The set
  `midpoint` holds the stage residual of the first step of `ImplicitMidpoint` for twelve
  scenarios of SolverBenchmark: the harmonic oscillator and the pendulum at `Δt = 0.1` and `1.0`,
  the two Lotka–Volterra systems at `0.01` and `0.1`, the double pendulum at `0.01` and `0.1`, and
  the Toda lattice with 16 particles at `0.1` and `1.0`. Each is built from GeometricProblems with
  `parameters = default_parameters(T)` and the initial conditions of SolverBenchmark. Its
  unknown is the stage velocity `V` (for an HODE the pair `(V, W)`), and its start is the vector
  field at the initial state: `v(t₀, q₀)`, the initial guess `v̄(t₀, q₀, p₀)` of an IODE, or
  `(v(t₀, q₀, p₀), f(t₀, q₀, p₀))`. `check_midpoint.jl` solves each residual in `Float64` and
  compares `q₁ = q₀ + h V` with one step of `ImplicitMidpoint` of GeometricIntegrators to
  `√eps(Float64) · max(1, ‖q₁‖∞)`.
- **The cells.** Each row runs in `Float32` and `Float64` with each of the 13 columns, (67 + 12) ×
  13 × 2 = 2054 rows, and each row has one cell per package, 8216 cells. NLsolve runs in `Float64`
  only; its `Float32` cells are `no Float32`. A package with no method for a column has a
  `no method` cell, with the reason in the pairing table.
- **The tolerance** is `τ = sqrt(eps(T))` on `‖F‖∞`: `3.45e-4` in `Float32`, `1.49e-8` in
  `Float64`, with `max_iterations = 200`. GeometricSolvers gets `f_abstol = τ`, `f_reltol = 0`;
  SimpleSolvers `f_abstol = τ`, `f_reltol = 0`, `verbosity = 0`; SimpleNonlinearSolve
  `abstol = τ`, `maxiters = 200`; NLsolve `ftol = τ`, `xtol = 0`, `iterations = 200`. Every other
  option keeps the package's default. GeometricSolvers and SimpleSolvers test `‖F‖₂`, which bounds
  `‖F‖∞`; SimpleSolvers also stops on a small relative change of `x` or `F`; NLsolve's
  `converged` can be true above `τ` after an exactly zero step.
- **The Jacobian.** Every package that accepts a Jacobian gets the same function `J!(J, x, p)`,
  ForwardDiff's `jacobian!` of the residual. GeometricSolvers' nine `Newton` columns with a
  factorisation get a problem with `J!`; its GMRES column, `Picard`, `Broyden` and
  `LeftPreconditioned` get a problem without it and keep their own Jacobian source.
- **The counts.** A counting wrapper on the residual, which every package calls, counts the calls
  with a floating-point iterate (`f_evals`) and with a dual one (`dual_evals`), and records
  `‖F‖∞` at each primal call: the trace of a row.
- **The success rule.** A cell succeeds when its package reports success, its iterate is finite,
  and the harness measures `‖F(x)‖∞ ≤ τ` at the returned `x` in `T`. A solve that throws has the
  code `ERROR` and fails.
- **The verdict** of a row is taken over the cells that ran, in the order GeometricSolvers,
  SimpleSolvers, SimpleNonlinearSolve, NLsolve: `single` when one ran; `all fail` when none
  succeeded; `disagree` when some succeeded and some did not; `agree` when all succeeded and every
  solution lies within `τ · max(1, ‖x_a‖∞)` in `‖·‖∞` of the first one, `x_a`; `different root`
  when all succeeded and one does not. Each `disagree` and `different root` row names one
  explanation below.

### Versions

Julia 1.13.3 on aarch64 macOS. GeometricSolvers at `301f366`, from `..`; SimpleSolvers 0.14.3,
SimpleNonlinearSolve 2.14.6, NLsolve 5.0.2, LineSearches 7.8.2, GeometricProblems 0.10.0,
GeometricIntegrators 0.19.0, ForwardDiff 1.4.7 and SciMLBase 3.58.0, each pinned in
`Project.toml`. No manifest is committed, so the other dependencies resolve afresh; the ones that
the solvers run are NonlinearSolveBase 2.55.1, LineSearch 0.1.20, DifferentiationInterface 0.7.22
and NLSolversBase 8.0.1.

### Pairing

The method label is built from the objects that each call passes to its package: their type
names, NLsolve's method symbol, and a GeometricSolvers method with its globalisation and linear
solver.

| column | package | method | difference from the column |
|:--|:--|:--|:--|
| Newton Static LU | GeometricSolvers | `Newton+Static+LUFactorization` | the column itself |
| Newton Static LU | SimpleSolvers | `Newton+Static+LapackLU` | the same full Newton step |
| Newton Static LU | SimpleNonlinearSolve | `SimpleNewtonRaphson` | a full Newton step; factorises with `\` |
| Newton Static LU | NLsolve | `newton+Static+BackslashSolve` | a full Newton step; factorises with `\` |
| Newton Backtracking LU | GeometricSolvers | `Newton+Backtracking+LUFactorization` | the column itself |
| Newton Backtracking LU | SimpleSolvers | `Newton+Backtracking+LapackLU` | SimpleSolvers' own backtracking line search |
| Newton Backtracking LU | SimpleNonlinearSolve | none | SimpleNewtonRaphson takes no line search |
| Newton Backtracking LU | NLsolve | `newton+BackTracking+BackslashSolve` | the backtracking line search of LineSearches; factorises with `\` |
| Newton Bisection LU | GeometricSolvers | `Newton+Bisection+LUFactorization` | the column itself |
| Newton Bisection LU | SimpleSolvers | `Newton+Bisection+LapackLU` | SimpleSolvers' own bisection line search |
| Newton Bisection LU | SimpleNonlinearSolve | none | SimpleNewtonRaphson takes no line search |
| Newton Bisection LU | NLsolve | none | LineSearches has no bisection line search |
| Newton StrongWolfe LU | GeometricSolvers | `Newton+StrongWolfe+LUFactorization` | the column itself |
| Newton StrongWolfe LU | SimpleSolvers | `Newton+StrongWolfe+LapackLU` | SimpleSolvers' own strong-Wolfe line search |
| Newton StrongWolfe LU | SimpleNonlinearSolve | none | SimpleNewtonRaphson takes no line search |
| Newton StrongWolfe LU | NLsolve | `newton+StrongWolfe+BackslashSolve` | the strong-Wolfe line search of LineSearches; factorises with `\` |
| Newton DogLeg LU | GeometricSolvers | `Newton+DogLeg+LUFactorization` | the column itself |
| Newton DogLeg LU | SimpleSolvers | `DogLeg` | SimpleSolvers' own dogleg trust region and radius update |
| Newton DogLeg LU | SimpleNonlinearSolve | `SimpleTrustRegion` | a dogleg trust region with its own radius update |
| Newton DogLeg LU | NLsolve | `trust_region` | a dogleg trust region with its own radius update |
| Newton Backtracking QR | GeometricSolvers | `Newton+Backtracking+QRFactorization` | the column itself |
| Newton Backtracking QR | SimpleSolvers | `Newton+Backtracking+PivotedQR` | SimpleSolvers' own backtracking line search; a column-pivoted QR that returns the minimum-norm solution at its rank tolerance |
| Newton Backtracking QR | SimpleNonlinearSolve | none | SimpleNewtonRaphson takes no line search |
| Newton Backtracking QR | NLsolve | `newton+BackTracking+PivotedQRSolve` | the backtracking line search of LineSearches; `qr(A, ColumnNorm()) \ b` |
| Newton Backtracking SVD | GeometricSolvers | `Newton+Backtracking+SVDFactorization` | the column itself |
| Newton Backtracking SVD | SimpleSolvers | `Newton+Backtracking+SVDSolver` | SimpleSolvers' own backtracking line search; an SVD that returns the minimum-norm solution at its rank tolerance |
| Newton Backtracking SVD | SimpleNonlinearSolve | none | SimpleNewtonRaphson takes no line search |
| Newton Backtracking SVD | NLsolve | `newton+BackTracking+SVDSolve` | the backtracking line search of LineSearches; `svd(A) \ b` |
| Newton LevenbergMarquardt SVD | GeometricSolvers | `Newton+LevenbergMarquardt+SVDFactorization` | the column itself |
| Newton LevenbergMarquardt SVD | SimpleSolvers | none | no Levenberg–Marquardt method |
| Newton LevenbergMarquardt SVD | SimpleNonlinearSolve | none | no Levenberg–Marquardt method |
| Newton LevenbergMarquardt SVD | NLsolve | none | no Levenberg–Marquardt method |
| Newton reuse 3 Backtracking LU | GeometricSolvers | `Newton(3)+Backtracking+LUFactorization` | the column itself |
| Newton reuse 3 Backtracking LU | SimpleSolvers | `Newton(3)+Backtracking+LapackLU` | refactorises at iterations 1, 3, 6, …, the column at 1, 4, 7, … |
| Newton reuse 3 Backtracking LU | SimpleNonlinearSolve | none | no factor reuse |
| Newton reuse 3 Backtracking LU | NLsolve | none | no factor reuse |
| Picard Backtracking | GeometricSolvers | `Picard+Backtracking` | the column itself |
| Picard Backtracking | SimpleSolvers | `Picard` | halves its step until `‖F‖₂` does not increase, a safeguard and not a line search |
| Picard Backtracking | SimpleNonlinearSolve | `SimpleDFSane` | a spectral residual method, not a Picard iteration |
| Picard Backtracking | NLsolve | none | `:anderson` with `m = 0` is a plain fixed-point step with no safeguard |
| Newton Backtracking GMRES | GeometricSolvers | `Newton+Backtracking+GMRES` | the column itself |
| Newton Backtracking GMRES | SimpleSolvers | none | no Krylov linear solver |
| Newton Backtracking GMRES | SimpleNonlinearSolve | none | no Krylov linear solver |
| Newton Backtracking GMRES | NLsolve | none | no Krylov linear solver |
| Broyden | GeometricSolvers | `Broyden` | the column itself |
| Broyden | SimpleSolvers | none | no Broyden method |
| Broyden | SimpleNonlinearSolve | `SimpleBroyden` | updates a full inverse from a scaled identity and uses no Jacobian; the column is a depth-4 history over `Picard` damped by 0.5 |
| Broyden | NLsolve | `broyden+Static` | updates a full inverse from the identity and uses no Jacobian; the column is a depth-4 history over `Picard` damped by 0.5 |
| LeftPreconditioned Picard Backtracking Newton | GeometricSolvers | `LeftPreconditioned+Picard+Backtracking+Newton+Static+LUFactorization` | the column itself |
| LeftPreconditioned Picard Backtracking Newton | SimpleSolvers | none | no nonlinear preconditioning |
| LeftPreconditioned Picard Backtracking Newton | SimpleNonlinearSolve | none | no nonlinear preconditioning |
| LeftPreconditioned Picard Backtracking Newton | NLsolve | none | no nonlinear preconditioning |

### Success counts

Each entry is `successes/ran` of one package, column and precision over the 79 rows; `—` marks a
pair that does not run.

| column | GeometricSolvers Float32 | GeometricSolvers Float64 | SimpleSolvers Float32 | SimpleSolvers Float64 | SimpleNonlinearSolve Float32 | SimpleNonlinearSolve Float64 | NLsolve Float32 | NLsolve Float64 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Newton Static LU | 71/79 | 70/79 | 66/79 | 70/79 | 69/79 | 71/79 | — | 71/79 |
| Newton Backtracking LU | 68/79 | 69/79 | 68/79 | 69/79 | — | — | — | 69/79 |
| Newton Bisection LU | 59/79 | 60/79 | 62/79 | 63/79 | — | — | — | — |
| Newton StrongWolfe LU | 67/79 | 68/79 | 71/79 | 70/79 | — | — | — | 65/79 |
| Newton DogLeg LU | 70/79 | 63/79 | 64/79 | 64/79 | 62/79 | 63/79 | — | 69/79 |
| Newton Backtracking QR | 67/79 | 69/79 | 61/79 | 61/79 | — | — | — | 69/79 |
| Newton Backtracking SVD | 68/79 | 69/79 | 61/79 | 61/79 | — | — | — | 69/79 |
| Newton LevenbergMarquardt SVD | 67/79 | 61/79 | — | — | — | — | — | — |
| Newton reuse 3 Backtracking LU | 67/79 | 70/79 | 70/79 | 69/79 | — | — | — | — |
| Picard Backtracking | 27/79 | 16/79 | 25/79 | 16/79 | 55/79 | 50/79 | — | — |
| Newton Backtracking GMRES | 67/79 | 68/79 | — | — | — | — | — | — |
| Broyden | 22/79 | 22/79 | — | — | 44/79 | 47/79 | — | 51/79 |
| LeftPreconditioned Picard Backtracking Newton | 50/79 | 51/79 | — | — | — | — | — | — |

### Broyden updates of a reused factor

The column `Newton reuse 3 Broyden Backtracking LU` (`Newton(; reuse = 3, update =
BroydenUpdate(2), globalization = Backtracking())`) is not in the tables above, which stay the run
at `301f366`. The three other packages have no method for it (no Broyden update of a reused
factor; `SimpleBroyden` and NLsolve's `:broyden` factorise no Jacobian). This table is
GeometricSolvers alone at `b86fc6c`, over the 79 rows in each precision, with the success rule
above, beside `Broyden` of the same run: `julia --project=benchmark benchmark/broyden_update.jl`
writes its 316 cells to `results/broyden_update.csv`. Each entry is the success count of the
column and precision, and the Jacobian count `Σ dual_evals / chunks`, where `chunks` is the number
of ForwardDiff chunks of the shared Jacobian of the row. That count also holds the directional
derivatives that the line search takes, so it is an upper bound on the Jacobian evaluations;
`Broyden` takes no Jacobian.

| column | Float32 successes | Float32 Jacobian count | Float64 successes | Float64 Jacobian count |
|:--|--:|--:|--:|--:|
| Broyden | 22/79 | 0.0 | 22/79 | 0.0 |
| Newton reuse 3 Broyden Backtracking LU | 66/79 | 1779.0 | 68/79 | 1829.0 |

### Verdicts

The number of rows of each verdict, per set and precision.

| verdict | standard Float32 | standard Float64 | midpoint Float32 | midpoint Float64 |
|:--|--:|--:|--:|--:|
| single | 201 | 201 | 36 | 36 |
| all fail | 122 | 118 | 5 | 5 |
| disagree | 117 | 119 | 10 | 12 |
| agree | 341 | 356 | 101 | 98 |
| different root | 90 | 77 | 4 | 5 |

### Explanations

Each `disagree` and `different root` row names the first explanation whose rule in
`benchmark/packages.jl` (`EXPLANATIONS`) applies to it. Each explanation gives its cause from the
trace of one of its rows, written by `trace.jl`, and from the counts of `correctness.csv`. A trace
holds, per package, `‖F‖∞` at each primal call of the residual, line-search trials included. The
problems of the set `standard` are named by their number in `STANDARD_PROBLEMS`.

### E1 · Sample problem 18 has two lines of roots

- **Rows:** 29
- **Trace:** benchmark/traces/E1.txt

Problem 18 sets `F₁ = 0` where `x₁ = 0` and `F₂ = 0` where `x₂ = 0`, and its formulas give
`F₁ = 0` on `x₂ = 0` and `F₂ = 0` on `x₁ = 0`. So every point of both axes is a root, and two
packages that succeed can end at different points of them. In the traced row (`10u0`, `Float64`,
`Newton StrongWolfe LU`) GeometricSolvers ends at `(1.07e-11, 0.0279)` and NLsolve at
`(1.93e-10, -0.877)`, both near the axis `x₁ = 0`, and SimpleSolvers at `(-40, 0)`, on the axis
`x₂ = 0`.

### E2 · One root reached to different accuracies

- **Rows:** 121
- **Trace:** benchmark/traces/E2.txt

All cells succeed, and the solutions differ by more than `τ · max(1, ‖x_a‖∞)`, but each solution
refines to the same root: the rule runs GeometricSolvers' `Newton()` from each solution in
`Float64` with no tolerance, and the refined points agree to `sqrt(eps(Float64)) · max(1, ‖r‖∞)`.
The test `‖F‖∞ ≤ τ` bounds the distance to the root only through the inverse Jacobian, so at an
ill-conditioned or singular root two solutions that pass it lie farther apart than `τ`. The
distance between the solutions is 1.0 to 9.1e4 times the bound. The rows are problems 2 (Powell
singular, root 0, 23 rows), 19 (`F = x ‖x‖²`, root 0, 27 rows) and 20 (the double root 5 of
`x (x − 5)²`), whose Jacobians are singular at the root; problems 3, 15 and 16, whose Jacobians
are ill-conditioned; problems 1, 7, 8, 9 and 22; and the two Lotka–Volterra systems, at 1.0 to 25
times the bound. In the traced row (problem 2, `u0`, `Float32`, `Newton Static LU`) the three
packages end at `‖F‖∞` of `1.9e-4`, `1.9e-4` and `4.8e-5`, below `τ = 3.45e-4`, near the
singular root 0.

### E3 · Different roots of one system

- **Rows:** 26
- **Trace:** benchmark/traces/E3.txt

All cells succeed, and the refined solutions of E2's rule lie at different roots. The systems have
more than one root, and the packages' paths from the same start end at different ones: problem 7
at `(-0.577, 0.577)` and `(0.577, -0.577)`; problem 8, Brown's almost linear function, at
`(1, …, 1)` and at `(0.979, …, 0.979, 1.21)`; problem 11, the trigonometric function, at roots
that differ by multiples of about `2π` in some components; problem 17 at `(0, 3)` and `(3, 0)`;
problem 22 at `(-0.707, 1.5)` and `(-1, 2)`; and problem 23 at two roots. In the traced row
(problem 8, `u0`, `Float64`, `Newton Backtracking LU`) GeometricSolvers and NLsolve reach the root
`(0.979, …, 0.979, 1.21)` in 10 and 14 iterations, SimpleSolvers the root `(1, …, 1)` in 77.

### E4 · The Picard column pairs three different fixed-point methods

- **Rows:** 78
- **Trace:** benchmark/traces/E4.txt

GeometricSolvers' `Picard` steps along `d = -F(x)` with a backtracking line search on `‖F‖₂`. This
direction decreases `‖F‖₂` only where `Fᵀ J F > 0`. At the last iterate of the 65 rows where
GeometricSolvers fails, `Fᵀ J F ≤ 0` in 54, all `LINESEARCH_FAILED`; one more ends
`LINESEARCH_FAILED` at `Fᵀ J F / (‖F‖² ‖J‖) = 3.3e-6`; the other 10 reach the iteration cap with
`Fᵀ J F > 0`, among them problem 19, `F = x ‖x‖²`, on which the step contracts slowly. The
midpoint residuals of the ODE and HODE scenarios are written as `F(V) = v(…) − V`, so their
Jacobian is near `−I` for small steps: at the start `Fᵀ J F / ‖F‖²` lies between −1.31 and −1.0,
and `-F` is a direction of increase. SimpleSolvers' `Picard` steps along the same direction and
halves the step until `‖F‖₂` does not increase; it fails in 67 rows, by its own stop rules or at
the cap. SimpleNonlinearSolve's `SimpleDFSane` is a spectral residual method: it tries both
`x + α d` and `x − α d` with a non-monotone acceptance test, so it can move where `-F` increases
`‖F‖`; it fails in 3 rows at the cap. The rows split so: 55 where GeometricSolvers and
SimpleSolvers fail and SimpleDFSane succeeds, 10 where only SimpleSolvers fails, 10 where only
GeometricSolvers fails, 1 where only SimpleDFSane fails, and 2 where SimpleSolvers and
SimpleDFSane fail. In the traced row (pendulum, `Δt = 1.0`, `Float64`) GeometricSolvers ends
`LINESEARCH_FAILED` after one iteration and SimpleSolvers stops after two, both at
`‖F‖∞ = 0.458`, and SimpleDFSane succeeds after 13 calls.

### E5 · GeometricSolvers' Broyden diverges or reaches the iteration cap

- **Rows:** 60
- **Trace:** benchmark/traces/E5.txt

GeometricSolvers' `Broyden()` is a depth-4 history over the member `Picard` damped by 0.5, whose
fixed-point map is `x − F(x)/2`. In 55 of the 60 rows this map does not contract at the start: the
spectral radius of `I − J(x₀)/2` exceeds 1. GeometricSolvers ends `NONFINITE` in 53 rows and at
the iteration cap in 7. In the traced row (problem 3, `u0`, `Float64`) `‖F‖∞` goes from 1.0 to
4.1e3 and then to `Inf` at the third call, while `SimpleBroyden` and NLsolve's `:broyden` reach
the root. In these rows `SimpleBroyden` also fails in 7 and NLsolve in 6. In 2 of them NLsolve
throws a `MethodError`: its `src/solvers/broyden.jl` (lines 143 and 145 in 5.0.2) calls
`ErrorException(message, value)` with two arguments when the norm of the residual is not finite,
and `ErrorException` has no such method.

### E6 · SimpleBroyden fails where GeometricSolvers' Broyden succeeds

- **Rows:** 6
- **Trace:** benchmark/traces/E6.txt

`SimpleBroyden` reaches its iteration cap in all 6 rows. In three midpoint rows (the two
Lotka–Volterra systems at `Δt = 0.01` in `Float32`, the Toda lattice at `Δt = 1.0` in `Float64`)
its residual becomes `NaN` and stays `NaN` until the cap: it tests its tolerance only after a step,
and where `‖F‖₂ ≥ 1e-5` it scales its first inverse Jacobian by `max(‖x‖, 1) / (2 ‖F‖₂)`, so from
a start with a small residual its first step is long. In the two Lotka–Volterra rows the start
already meets `τ = 3.45e-4`, and GeometricSolvers ends `SUCCESS` with 0 iterations. In the traced
row (Toda lattice, `Δt = 1.0`, `Float64`) `‖F‖∞` grows from 0.109 to above 1 within ten calls and
ends at `NaN`. In the three rows of problem 18 it ends at `‖F‖∞` between 2.4e3 and 1.5e4.

### E7 · Four dogleg trust regions with different radius updates

- **Rows:** 38
- **Trace:** benchmark/traces/E7.txt

The four packages take a dogleg step, but each with its own initial radius, acceptance test and
radius update. GeometricSolvers reaches the cap in 12 rows (problems 1, 15, 16 and 22); in problem
1 its `‖F‖∞` decreases by about 1 % per iteration at the cap. It ends `LINESEARCH_FAILED` in 4
rows of problem 11, where it rejects a trial at its smallest radius `1e-6`. SimpleSolvers fails in
14 rows at the cap or by its stop rules, and throws a `NonlinearSolverException` in 7 `Float32`
rows from `10u0` or `100u0`, at its second call ("non-finite direction₂ vector!" in problem 3 from
`100u0`). NLsolve reaches the cap in 5 rows. `SimpleTrustRegion` reaches the cap in 19 rows, ends
`ShrinkThresholdExceeded` in 3, and throws in 2. Its steepest-descent point is `-g`, `g = Jᵀ F`,
scaled to the radius, without the Cauchy step length `gᵀg / gᵀ Jᵀ J g`; where the radius exceeds the
model's minimiser along `-g`, the model predicts an increase of `‖F‖₂²`, the actual change is an
increase too, and its ratio test accepts the step. In the traced row (problem 9, `u0`, `Float64`)
the other three packages reach the root in 3 iterations; `SimpleTrustRegion`'s first step goes along
`-g` with radius 0.0150, where the model predicts a change of `+3.0e-4` in `‖F‖₂²/2` and the actual
change is `+3.0e-4`, and its `‖F‖∞` grows from 0.0123 to 0.140 over 401 calls.

### E8 · The undamped Newton step and the stop rules of each package

- **Rows:** 6
- **Trace:** benchmark/traces/E8.txt

The four packages take the same full Newton step, and far from a root it can increase `‖F‖` by
many orders. SimpleSolvers ends a solve when `‖F‖` increases, and does so in all 6 rows (problems
8 and 18 in `Float32`, problem 1 in `Float64`), while the others go on. In the two rows of problem
18 `SimpleNewtonRaphson` throws a `SingularException` at a singular Jacobian. In the traced row
(problem 1, `u0`, `Float64`) all four follow one path, `‖F‖∞` = 4.4, 3.8e6, 6.7e27, …, 5.3e176,
with values that differ in the last digits between the LU factorisations; SimpleSolvers stops at
iteration 5, GeometricSolvers' tenth call returns `Inf` and it ends `NONFINITE`, and
SimpleNonlinearSolve and NLsolve stay finite, `2.1e176`, and reach the root at the twelfth call.

### E9 · SimpleSolvers' rank-revealing solvers stall

- **Rows:** 35
- **Trace:** benchmark/traces/E9.txt

SimpleSolvers' `PivotedQR` and `SVDSolver` return the minimum-norm solution at their rank
tolerance. Near the roots of problems 1, 3, 4, 15 and 16 the Jacobian is ill-conditioned, so the
truncated step does not reduce `‖F‖`, and SimpleSolvers ends `not converged`. At its last iterate
the solver reports rank `n − 1` in 31 of the 35 rows, at condition numbers from 2.9e3 to 2.3e9; in
the other 4 (problems 15 and 16 from `10u0` in `Float64`) it reaches the cap at full rank. With
`rtol = 0` in place of the default rank tolerance, SimpleSolvers succeeds in 34 of the 35 rows. In
the traced row (problem 15, `u0`, `Float64`, `Newton Backtracking QR`) its `‖F‖∞` stops at 1.25e-6,
where the Jacobian has condition number 1.1e8 and rank 3 of 4 for `PivotedQR`, while
GeometricSolvers' `QRFactorization` and NLsolve reach `1.1e-16`.

### E10 · GeometricSolvers' line search ends the solve

- **Rows:** 26
- **Trace:** benchmark/traces/E10.txt

In each of these rows SimpleSolvers succeeds and GeometricSolvers ends `LINESEARCH_FAILED` (16),
`STALLED` (8), `SINGULAR` (1) or `NONFINITE` (1), in the `Bisection`, `StrongWolfe`, reuse, QR and
SVD columns. The line searches of the two packages are different implementations, so the iterates
differ from the first step on. In seven rows of problem 21, the Freudenstein–Roth function,
GeometricSolvers ends at `‖F‖∞` between 6.9 and 7.7. In the traced row (`u0`, `Float64`,
`Newton StrongWolfe LU`) it ends at `(13.5, -0.897)` with `‖F‖∞ = 7.06`; SimpleSolvers' trace
returns to `‖F‖∞ ≈ 7.08` many times, moves to `‖F‖∞ ≈ 3.6e24` near its 1040th call and from there
decreases to the root at its 1327th call; NLsolve throws a `LineSearchException` there, as in E12.
In problem 18 from `10u0` in `Float64`, SimpleSolvers' bisection reaches a point of the axis
`x₂ = 0`, a root (E1), at its eighth call, and GeometricSolvers' bisection ends
`LINESEARCH_FAILED` at `‖F‖∞ = 3.18`.

### E11 · SimpleSolvers' line search reaches the iteration cap or stops

- **Rows:** 6
- **Trace:** benchmark/traces/E11.txt

GeometricSolvers succeeds and SimpleSolvers does not: in four rows of problem 13 with `Bisection`
it stays at `‖F‖∞` between 0.913 and 0.915 until the cap; in problem 1 with reuse it reaches the
cap at `‖F‖∞ = 0.062`; in problem 16 from `100u0` with `Bisection` it stops at iteration 18 when
`‖F‖` increases. In problem 13 from `10u0`, at SimpleSolvers' last iterate the Jacobian has
condition number 1.4e5 in `Float32` and the Newton step has norm 2.6e4 against `‖x‖ = 2.8`. In
`Float64` the merit `‖F(x + α d)‖₂²/2` along it is 8.2e10 at `α = 1e-6` against 0.925 at
`α = 0`, so SimpleSolvers' bisection finds no step that reduces `‖F‖`. In the traced row (`10u0`,
`Float32`) GeometricSolvers succeeds in 3 iterations, and SimpleSolvers takes 7660 calls in 200
iterations.

### E12 · NLsolve's strong-Wolfe line search throws

- **Rows:** 3
- **Trace:** benchmark/traces/E12.txt

In problem 8 from `u0` and `10u0` and problem 23 from `10u0`, in `Float64`, LineSearches'
`StrongWolfe` throws "zoom reached maximum iterations (10) without satisfying Wolfe conditions"
(`LineSearchException`), which NLsolve does not catch, while GeometricSolvers and SimpleSolvers
succeed. In the traced row (problem 8, `u0`) NLsolve throws after 12 calls.

## Y2 · Timings and further rows on the CPU

This section times every cell of Y1 that ran, reads the timings and Y1's counts as performance
profiles, and adds four blocks of cells: mixed precision, a looser stage tolerance, Anderson
acceleration over Newton, and a sparse problem. Each table states the conditions of its
comparison. A time is a measurement under the load stated below, not a gate, and a second run
gives other times. The mixed-precision block adds SIAMFANLEquations' `nsol`, with
MultiPrecisionArrays' `mplu`, as a baseline.

| script | what it does |
|:--|:--|
| `timing.jl <dir>` | times every ran cell of `<dir>/correctness.csv` and writes `<dir>/timings.csv` |
| `blocks.jl <dir>` | runs and times the four blocks and writes `<dir>/mixed.csv`, `stage.csv`, `anderson.csv` and `sparse.csv` |
| `profiles.jl <dir>` | prints the tables of this section from the tables of `<dir>` |
| `verify.jl <dir> [--structure]` | checks the tables of this section too |
| `selftest.jl` | tests the timed runs and the batch, the note of a timed cell, the tolerance of each adapter, the variant labels, the `nsol` adapter, the sparse variants, the merge of the parts and the profiles too |

### Y2 · Method

- **The timed sweep.** `timing.jl` times each of the 4661 cells of `correctness.csv` with
  `status = ran`, in one fresh Julia process per package and precision, seven in all. Each
  process starts with `--startup-file=no` and `--threads=1`, and calls `BLAS.set_num_threads(1)`
  before any solve. Per cell it makes one warm-up solve, then finds the batch `k`, the smallest
  power of two for which two successive runs of `k` solves each last at least 1 ms, and then makes
  five runs of `k` solves. Two runs, so that one run slowed by a pause, such as a garbage
  collection or the compilation of the timing loop for the cell's call, does not decide `k`. Each
  solve gets its own counting wrapper of the residual and its own copy of the start, made before
  the timer starts; the timed call builds the package's problem object and solves, so a time
  includes the work of the counting wrapper. A run's time per solve is its elapsed time divided by
  `k`. `median_s` is the median of the five runs, and `spread` is `(max − min) / median`. A solve
  that throws is timed to
  the throw; the 45 cells of `correctness.csv` with the code `ERROR` have the note `throws`.
- **The warm-up against Y1.** The warm-up solve of a cell gives its code, iterations and
  `f_evals` again. A cell whose values differ from `correctness.csv` is timed all the same, and its
  note gives both values. 0 of the 4661 timed cells differ.
- **The blocks.** `blocks.jl` runs each block cell once, through the counting wrapper and the
  success rule of Y1, with the block's tolerance `tol`, 200 iterations and the shared ForwardDiff
  Jacobian, unless the block says otherwise. That solve is the warm-up of the cell, which is then
  timed as a cell of the timed sweep, in one fresh process per block, package and precision, 16 in
  all. Six of them run no solve and write only the status lines of their cells: stage NLsolve,
  sparse NLsolve `Float32`, and sparse SimpleSolvers and SimpleNonlinearSolve in both precisions.
  A cell succeeds when its package reports success, its iterate is finite and the harness
  measures `‖F(x)‖∞ ≤ tol`. A line of a block table gives the five times, `median_s`, `spread`
  and the batch `k` as `solves_per_run`, as a line of `timings.csv` does.
- **The labels.** The `method` of a block cell is read from the objects that its adapter passes: a
  GeometricSolvers `LUFactorization(Float32)` shows as `LUFactorization{Float32}`, and
  `LUFactorization()`, which factors in the element type of the problem, as `LUFactorization` in
  both precisions; a `Refinement` shows as `+Refinement`, a factor reuse of `k` steps as
  `Newton(k)`, an `Anderson` of depth `m` as `Anderson{m}(<member>)`, and `nsol` as
  `nsol(sham=<sham>)+FPS{<eltype of its Jacobian store>}` or `nsol(sham=<sham>)+mplu`.
- **The machine.** Apple M4 Max, aarch64 macOS 27.2. The timed sweep ran on Julia 1.13.3 and the
  blocks on Julia 1.13.4, in a later session. The machine runs other work, so each process prints
  the `uptime` load at its start. The 1-minute load averages were between 2.37 and 2.73 for the
  seven timing processes and between 4.69 and 5.54 for the 16 block processes.
  Over the cells of the timed sweep, the median of `spread` is 0.036 and its 90th percentile is
  0.56. In 34 of the 4661 timed cells, `k` times `median_s` is below 1 ms: both runs that decided
  `k` were slower than the median of the five runs.
- **The versions.** GeometricSolvers from `..`, whose `src/` is that of `301f366`, and the
  packages of Y1 at the versions of Y1. The new packages are pinned in `Project.toml` and resolve
  to MultiPrecisionArrays 0.1.4, SIAMFANLEquations 1.0.3, Sparspak 0.3.15, SparseMatrixColorings
  0.4.29 and ADTypes 1.24.0, with SparseArrays 1.13.0. The blocks ran with the same `src/` and
  the same package versions. Against an earlier run of the blocks on Julia 1.13.3, 31 block cells
  of the two `DoublePendulum` midpoint rows have a `‖F‖∞` that differs in the last digits, and two
  SimpleSolvers `Picard Backtracking` cells of the stage block that fail in both runs have other
  `f_evals` (50 → 47, 52 → 51); no code and no success changed.

| process | 1-minute load at its start |
|:--|--:|
| timing GeometricSolvers Float32 | 2.51 |
| timing GeometricSolvers Float64 | 2.59 |
| timing SimpleSolvers Float32 | 2.37 |
| timing SimpleSolvers Float64 | 2.44 |
| timing SimpleNonlinearSolve Float32 | 2.61 |
| timing SimpleNonlinearSolve Float64 | 2.73 |
| timing NLsolve Float64 | 2.59 |
| blocks mixed GeometricSolvers Float64 | 5.22 |
| blocks mixed SIAMFANLEquations Float64 | 5.54 |
| blocks stage GeometricSolvers Float32 | 5.22 |
| blocks stage SimpleSolvers Float32 | 4.95 |
| blocks stage SimpleNonlinearSolve Float32 | 5.28 |
| blocks stage NLsolve Float32 | 5.43 |
| blocks anderson GeometricSolvers Float32 | 5.35 |
| blocks anderson GeometricSolvers Float64 | 5.42 |
| blocks sparse GeometricSolvers Float32 | 4.91 |
| blocks sparse GeometricSolvers Float64 | 4.85 |
| blocks sparse NLsolve Float32 | 4.79 |
| blocks sparse NLsolve Float64 | 4.75 |
| blocks sparse SimpleSolvers Float32 | 4.90 |
| blocks sparse SimpleSolvers Float64 | 5.02 |
| blocks sparse SimpleNonlinearSolve Float32 | 4.91 |
| blocks sparse SimpleNonlinearSolve Float64 | 4.69 |

### Y2 · Performance profiles

The performance profiles, as tables. The problems of a precision are its 1027 (row,
column) pairs of Y1, and the solvers are the packages that run in it: all four in `Float64`, and
GeometricSolvers, SimpleSolvers and SimpleNonlinearSolve in `Float32`. SimpleNonlinearSolve
reports no iteration count, so it is not in the iteration tables. A package solves each pair with
its own method for the column, the method of the pairing table, so a profile compares the paired
methods of each package over all columns, not one algorithm.

For a pair `p` and a solver `s`, `t_{p,s}` is the metric of the cell when it succeeded by Y1's
rule, and `∞` when it did not: a failure, an `ERROR` or a `no method` cell. A count of 0, which 44
cells that ran and succeeded have, counts as 1, so every ratio is finite and at least 1.
`r_{p,s} = t_{p,s} / min_s t_{p,s}`, and `ρ_s(τ)` is the share of the 1027 pairs with
`r_{p,s} ≤ τ`; at `τ = ∞` it is the share of the pairs that `s` solves. A pair that no solver
solves stays in the 1027 and adds to no `ρ_s`. The metrics are the iterations, the residual
evaluations `f_evals` and the median time `median_s` of the timed sweep. `verify.jl` recomputes
each table from `correctness.csv` and `timings.csv`.

#### Profile · Float32 · iterations

1027 problems.

| solver | τ = 1 | τ = 2 | τ = 4 | τ = 8 | τ = 16 | τ = ∞ |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 0.683 | 0.729 | 0.746 | 0.749 | 0.750 | 0.750 |
| SimpleSolvers | 0.457 | 0.519 | 0.530 | 0.534 | 0.534 | 0.534 |

#### Profile · Float32 · f_evals

1027 problems.

| solver | τ = 1 | τ = 2 | τ = 4 | τ = 8 | τ = 16 | τ = ∞ |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 0.674 | 0.719 | 0.743 | 0.747 | 0.750 | 0.750 |
| SimpleSolvers | 0.069 | 0.130 | 0.422 | 0.521 | 0.532 | 0.534 |
| SimpleNonlinearSolve | 0.171 | 0.209 | 0.220 | 0.222 | 0.224 | 0.224 |

#### Profile · Float32 · median_s

1027 problems.

| solver | τ = 1 | τ = 2 | τ = 4 | τ = 8 | τ = 16 | τ = ∞ |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 0.639 | 0.710 | 0.742 | 0.748 | 0.750 | 0.750 |
| SimpleSolvers | 0.089 | 0.497 | 0.520 | 0.532 | 0.534 | 0.534 |
| SimpleNonlinearSolve | 0.097 | 0.179 | 0.211 | 0.220 | 0.224 | 0.224 |

#### Profile · Float64 · iterations

1027 problems.

| solver | τ = 1 | τ = 2 | τ = 4 | τ = 8 | τ = 16 | τ = ∞ |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 0.638 | 0.710 | 0.730 | 0.733 | 0.734 | 0.736 |
| SimpleSolvers | 0.449 | 0.512 | 0.525 | 0.529 | 0.529 | 0.529 |
| NLsolve | 0.420 | 0.442 | 0.450 | 0.451 | 0.451 | 0.451 |

#### Profile · Float64 · f_evals

1027 problems.

| solver | τ = 1 | τ = 2 | τ = 4 | τ = 8 | τ = 16 | τ = ∞ |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 0.584 | 0.688 | 0.716 | 0.734 | 0.734 | 0.736 |
| SimpleSolvers | 0.042 | 0.081 | 0.357 | 0.517 | 0.528 | 0.529 |
| SimpleNonlinearSolve | 0.162 | 0.208 | 0.217 | 0.221 | 0.223 | 0.225 |
| NLsolve | 0.407 | 0.422 | 0.433 | 0.441 | 0.443 | 0.451 |

#### Profile · Float64 · median_s

1027 problems.

| solver | τ = 1 | τ = 2 | τ = 4 | τ = 8 | τ = 16 | τ = ∞ |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 0.576 | 0.690 | 0.727 | 0.733 | 0.734 | 0.736 |
| SimpleSolvers | 0.071 | 0.460 | 0.518 | 0.526 | 0.529 | 0.529 |
| SimpleNonlinearSolve | 0.090 | 0.189 | 0.214 | 0.223 | 0.225 | 0.225 |
| NLsolve | 0.088 | 0.376 | 0.441 | 0.447 | 0.449 | 0.451 |

### Y2 · DogLeg against Backtracking

Each package that has a method in both columns `Newton DogLeg LU` and `Newton Backtracking LU`,
in each precision it runs: GeometricSolvers' `DogLeg` and `Backtracking`, SimpleSolvers' own
dogleg and backtracking, and NLsolve's `:trust_region` and `:newton` with LineSearches'
`BackTracking`. The successes of each column are over the 79 rows of the precision; the medians of
the ratios DogLeg / Backtracking are over the rows where both succeed, with a count of 0 taken as
1. No new run produces this table: it reads `correctness.csv` and `timings.csv`.

| package | eltype | DogLeg successes | Backtracking successes | both | iterations | f_evals | median_s |
|:--|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | Float32 | 70 | 68 | 66 | 1.00 | 1.00 | 1.14 |
| GeometricSolvers | Float64 | 63 | 69 | 61 | 1.00 | 1.00 | 1.16 |
| SimpleSolvers | Float32 | 64 | 68 | 62 | 1.00 | 0.667 | 0.849 |
| SimpleSolvers | Float64 | 64 | 69 | 61 | 1.00 | 0.639 | 0.782 |
| NLsolve | Float64 | 69 | 69 | 66 | 1.00 | 1.00 | 0.982 |

### Y2 · Mixed precision

The 79 `Float64` rows with five variants each, 395 cells in `mixed.csv`, each with
`tol = √eps(Float64)`, 200 iterations and the shared ForwardDiff Jacobian:

- `gs-f64`: GeometricSolvers' `Newton(; linsolve = LUFactorization(), globalization =
  Backtracking())`;
- `gs-f32`: the same with `LUFactorization(Float32)` and `NoRefinement()`;
- `gs-f32-refine`: the same with `LUFactorization(Float32)` and `Refinement()`;
- `nsol-f32`: SIAMFANLEquations' `nsol` with a `Float32` Jacobian store `FPS`;
- `nsol-mplu`: `nsol` with a `Float64` store and `jfact = mplu` of MultiPrecisionArrays, which
  factors a `Float32` copy and refines.

`nsol` gets `rtol = 0`, `atol = tol`, `maxit = 200`, `sham = 1` (a new Jacobian at every
iteration) and `printerr = false`, and keeps its own Armijo line search. It stops on `‖F‖₂`, which
bounds `‖F‖∞`. It counts as reporting success with `idid = true`, or with `errcode = -1`, which it
returns when the start already meets its test; its iterations are the entries of its history after
the first. The successes are of 79; the medians of the ratios to `gs-f64` are over the rows where
the variant and `gs-f64` both succeed, with a count of 0 taken as 1.

| variant | method | successes of 79 | both with gs-f64 | iterations | median_s |
|:--|:--|--:|--:|--:|--:|
| gs-f64 | `Newton+Backtracking+LUFactorization` | 69 | 69 | 1.00 | 1.00 |
| gs-f32 | `Newton+Backtracking+LUFactorization{Float32}` | 67 | 67 | 1.00 | 1.85 |
| gs-f32-refine | `Newton+Backtracking+LUFactorization{Float32}+Refinement` | 68 | 68 | 1.00 | 1.97 |
| nsol-f32 | `nsol(sham=1)+FPS{Float32}` | 62 | 62 | 1.00 | 0.957 |
| nsol-mplu | `nsol(sham=1)+mplu` | 60 | 60 | 1.00 | 3.17 |

Eight `nsol` cells end in `ERROR`, four per variant: the three starts of problem 20, the one
system of dimension 1, throw a `MethodError` in `nsol`, and problem 12 from `100u0` throws a
`SingularException`. Four `nsol` cells return `errcode = 0` after one iteration at an iterate
where the harness measures `‖F‖∞ = NaN`, and fail by the harness's rule: problem 3 from `100u0`
in both variants, and problems 18 from `100u0` and 22 from `u0` with `mplu`.

### Y2 · Stage tolerance

The 12 midpoint rows in `Float32`, with every column of the pairing table and every package, 624
cells in `stage.csv`, at `tol = eps(Float16)` = `9.77e-4` in place of Y1's `√eps(Float32)` =
`3.45e-4`, and Y1's adapters and options otherwise. The two tolerances differ by a factor of 2.8.
NLsolve's cells are `no Float32`, and a pair without a method is `no method`. Each line counts
the successes at each tolerance over the same 156 (row, column) pairs, and gives the medians of
the ratios loose / strict over the pairs where both succeed, with a count of 0 taken as 1; the
strict times are those of the timed sweep, from other processes of an earlier session.

| package | successes at eps(Float16) | successes at √eps(Float32) | both | iterations | f_evals | median_s |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 143 | 143 | 143 | 1.00 | 1.00 | 1.03 |
| SimpleSolvers | 98 | 98 | 98 | 1.00 | 1.00 | 1.04 |
| SimpleNonlinearSolve | 41 | 41 | 41 | — | 1.00 | 0.997 |

### Y2 · Anderson

The 79 rows in both precisions, GeometricSolvers only, 1264 cells in `anderson.csv`. For `k = 1,
2, 3, 5`, `anderson-k` is `Anderson(Newton(; reuse = k, globalization = Backtracking(), jacobian
= nothing))` with the depth 4, and `newton-k` is the same `Newton` alone; each solves a problem
with the shared Jacobian, with `tol = √eps(T)` and 200 iterations. The successes are of 79; the
medians of the ratios `anderson-k` / `newton-k` are over the rows where both succeed, with a count
of 0 taken as 1.

| eltype | k | anderson-k successes | newton-k successes | both | iterations | f_evals | median_s |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Float32 | 1 | 58 | 68 | 57 | 1.00 | 1.50 | 1.42 |
| Float32 | 2 | 61 | 68 | 61 | 1.20 | 1.72 | 1.79 |
| Float32 | 3 | 66 | 67 | 65 | 1.08 | 1.56 | 1.72 |
| Float32 | 5 | 65 | 67 | 63 | 0.905 | 1.25 | 1.32 |
| Float64 | 1 | 63 | 69 | 62 | 1.11 | 1.84 | 1.58 |
| Float64 | 2 | 63 | 71 | 63 | 1.22 | 1.93 | 1.75 |
| Float64 | 3 | 66 | 70 | 66 | 1.12 | 1.66 | 1.75 |
| Float64 | 5 | 67 | 69 | 66 | 0.963 | 1.27 | 1.36 |

### Y2 · The sparse problem

`sparse_bracket_problem(N, T)` of `test/helpers/sparse_problems.jl`, with its seed 1, for
`N = 125` and `768` (dimension `2N`) and `T = Float32` and `Float64`, from its start `z0` with its
parameter `p_val`, at `tol = √eps(T)` and 200 iterations; 20 cells in `sparse.csv`. The variants
differ in the Jacobian source and the linear solver:

- `gs-sparse`: GeometricSolvers' `Newton(; jacobian = AutoSparse(AutoForwardDiff();
  coloring_algorithm = GreedyColoringAlgorithm()))` on a problem with the sparsity prototype, which
  factors with Sparspak's LU through the package extension; each timed solve includes the
  colouring and the sparse-AD preparation of `prepare_ad`, which the Jacobian functions that the
  other two variants pass do not need;
- `gs-dense`: GeometricSolvers' `Newton()` on a problem with the shared dense ForwardDiff
  Jacobian;
- `nlsolve-sparse`: NLsolve's `:newton` on a `OnceDifferentiable` with the problem's analytic
  sparse Jacobian and a copy of the prototype, in `Float64` only;
- `simplesolvers` and `simplenonlinearsolve`: `no sparse path`, because the benchmark has no
  sparse adapter for these two packages.

`median_s` is in seconds.

| variant | N | eltype | status | code | iterations | f_evals | median_s |
|:--|--:|:--|:--|:--|--:|--:|--:|
| gs-sparse | 125 | Float32 | ran | SUCCESS | 2 | 3 | 0.000301 |
| gs-dense | 125 | Float32 | ran | SUCCESS | 2 | 3 | 0.000796 |
| nlsolve-sparse | 125 | Float32 | no Float32 |  |  |  |  |
| simplesolvers | 125 | Float32 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 125 | Float32 | no sparse path |  |  |  |  |
| gs-sparse | 125 | Float64 | ran | SUCCESS | 3 | 4 | 0.000416 |
| gs-dense | 125 | Float64 | ran | SUCCESS | 3 | 4 | 0.00187 |
| nlsolve-sparse | 125 | Float64 | ran | converged | 3 | 4 | 0.000428 |
| simplesolvers | 125 | Float64 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 125 | Float64 | no sparse path |  |  |  |  |
| gs-sparse | 768 | Float32 | ran | SUCCESS | 2 | 3 | 0.00165 |
| gs-dense | 768 | Float32 | ran | SUCCESS | 2 | 3 | 0.0665 |
| nlsolve-sparse | 768 | Float32 | no Float32 |  |  |  |  |
| simplesolvers | 768 | Float32 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 768 | Float32 | no sparse path |  |  |  |  |
| gs-sparse | 768 | Float64 | ran | SUCCESS | 3 | 4 | 0.00239 |
| gs-dense | 768 | Float64 | ran | SUCCESS | 3 | 4 | 0.181 |
| nlsolve-sparse | 768 | Float64 | ran | converged | 3 | 4 | 0.00246 |
| simplesolvers | 768 | Float64 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 768 | Float64 | no sparse path |  |  |  |  |

## Y3 · Device rows

This section runs Y1's rows in the batched R2 solver on a device, times R2 throughput on the §10.E
problem, and runs the §10.E problem in R3 form with the device LU and a mixed-precision Newton
step. `devices.jl <backend> <machine> <dir>` runs one machine and writes
`devices-<machine>.csv`, one line per cell, and `devices-<machine>.txt`, the conditions of the run.
The machines are the host in the cpu mode (`cpu cpu`) and four cards, each run by hand: the M4
Max's GPU (`metal mac`), the DGX Spark (`cuda spark`), the RTX 4090 (`cuda rtx4090`) and the
RX 7900 XTX (`rocm rx7900xtx`). Each table states the conditions of its comparison; a time is a
measurement of one run on one machine, and it names no package or method as faster.

### Y3 · Method

- **The environment.** `benchmark/Project.toml` adds KernelAbstractions, StaticArrays and
  JLArrays with the exact versions 0.10.7, 1.9.25 and 0.5.0. The vendor package comes only from
  the environment `test/gpu/<backend>`, stacked under `benchmark/` by
  `JULIA_LOAD_PATH="@:$PWD/test/gpu/<backend>:@stdlib"`, and only its backend loads it; the cpu
  mode loads none. The cpu mode runs the R2 rows and the grid with KernelAbstractions' `CPU()`
  backend on host arrays, which compiles each kernel through POCL, so a kernel that allocates on
  the heap fails there; it runs the R3 and LU rows on `JLArray`s.
- **The process.** Each machine runs in a cold process with `--startup-file=no` and
  `--threads=auto`, and `devices.jl` calls `BLAS.set_num_threads(1)` first; on Metal, further
  processes continue the R2 rows (*The Metal heap* below). The `.txt` records, read before the
  first row, the date, the `uptime` with the load of the machine, the Julia version, the threads
  of the run, the BLAS threads, the device (on a card, the name that the vendor package gives),
  the driver, the band and the versions of the packages.
- **The R2 rows.** Each of Y1's 79 `Float32` rows runs as one batch of 64 copies of its start.
  The residual runs out of place on an `SVector` through an adapter that copies the iterate into
  an `MVector`, calls the row's in-place residual with `@inline` at the call site, and returns an
  `SVector`; without the inlined call the `MVector` reaches the heap, which a kernel cannot
  allocate on `CPU()`. The midpoint rows use residuals that hold the initial state as `SVector`s
  and take GeometricProblems' parameters as `p`. GeometricSolvers runs `solve_batch!` with a
  status vector in the columns `Newton Static LU`, `Newton Backtracking LU`, `Newton DogLeg LU` and
  `Picard Backtracking`, with `f_abstol = √eps(Float32)`, `f_reltol = 0` and
  `max_iterations = 200`. SimpleNonlinearSolve runs `SimpleNewtonRaphson` in a KernelAbstractions
  kernel, one work-item per system, with `abstol = √eps(Float32)` and `maxiters = 200`, in
  `Newton Static LU` only; its other cells are `no method`. SimpleSolvers and NLsolve have no
  device path. Each cell has a reference: the host R1 `solve` of the same adapter, method and
  start, or SimpleNonlinearSolve's host `solve` on an `SVector`. `success` is the package's code of
  success with `‖F(x)‖∞ ≤ √eps(Float32)`, measured on the host at system 1's iterate. The verdict is
  `agree` when both succeed and `‖x − x_host‖∞ ≤ √eps(T) · max(1, ‖x_host‖∞)`, `different root`
  when both succeed beyond it, and otherwise `device fails`, `host fails` or `both fail`.
  `copies_same` counts the copies within that bound of system 1; no copy is within the bound of an
  iterate that holds a `NaN`, so such a batch counts 0. The comment holds the host's
  `‖x‖∞` and `‖F‖∞`, from which `verify.jl` recomputes the verdict.
- **The band.** A row whose dimension is above the R1 size band of its backend, 24 on Metal and
  CUDA and 8 on ROCm, is not launched: `above band <band>`. The cpu mode has no band. In the grid
  the band limits the device variants; `host-r1` runs on the host and has no band.
- **A launch that throws.** A kernel that does not compile (an `InvalidIRError`, a `KernelError`
  or POCL's `Failed to compile program`) is `no compile: <first line>`; a kernel that throws (a
  `KernelException`, or AMDGPU's `GPU Kernel Exception`) is `kernel error: <first line>`. The host
  solve still runs for both. Any other exception ends the run with the cell named.
- **SimpleNonlinearSolve on ROCm.** Its kernel faults with `hipErrorIllegalAddress` on the
  RX 7900 XTX, which leaves the device context unusable for the rest of the process, so no
  SimpleNonlinearSolve cell within the band launches there: `not run: SimpleNonlinearSolve kernel
  faults on ROCm`, in R2 and in the grid's `sns` variant.
- **The Metal heap.** Metal.jl gives a kernel's `malloc` a heap of 1 MiB per process with no free.
  On Metal the `.txt` records the heap counter before and after each R2 row. The run continues in
  a new process from the same row when a launch of the row ends in a kernel error on a heap that
  earlier rows used, and from the next row when the counter passes the size after the row. The
  other backends record `-`.
- **The grid.** The §10.E problem in R1 form, `F(x) = A x + sin.(x) / 10 − b` with
  `A = 2I + R`, off-diagonal row sums of `R` at most 1/2, one `A` per `n`, a per-system `x*` in
  `[-1, 1]ⁿ` from a fixed seed, `b` built in `Float32` on the host and passed as `PerSystem(B)`,
  from `x₀ = 0`, for `n_sys` from 10³ to 10⁷ systems and `n` from 2 to 16. The variants are
  `gs-newton` (`Newton()`), `gs-reuse4` (`Newton(; reuse = 4)`), `gs-backtracking`
  (`Newton(; globalization = Backtracking())`), `sns` (`SimpleNewtonRaphson` in a kernel) and
  `host-r1` (a `Threads.@threads` loop over the host R1 `solve` with `Newton()`). `success` counts
  the systems that succeed, and `fnorm_inf` is the largest `‖F(x)‖∞`.
- **The timing.** Each timed cell makes one warm-up launch, then five runs. Each launch starts from
  a fresh copy of the starts, made before the clock starts, and each run ends with
  `KernelAbstractions.synchronize` before the clock stops. Where one launch lasts less than 1 ms, a
  run repeats it `k` times, `k` the smallest power of two for which two successive runs each last
  at least 1 ms, and a time is the run's time divided by `k`. `median_s` is the median of the five
  runs, `spread` is `(max − min) / median`, and `compile_s` is the warm-up launch's time minus the
  median, the time of compilation apart from the runs. A grid cell whose time, projected as ten
  times the median of the cell at a tenth of its systems for each of six launches, exceeds 30
  minutes is `not run: time cap`, and so is every larger `n_sys` of its `n` and variant; a cell
  after one whose launch threw has no projection and runs. The
  throughput is `n_sys / median_s`, solves per second.
- **The R3 rows.** The §10.E residual `y .= A*x .+ sin.(x)./10 .- b` and its Jacobian
  `A + diag(cos.(x))/10`, with broadcasts only, for `n = 256`, 1024 and 4096, solved with
  `Newton(; globalization = Backtracking())`, `f_abstol = √eps(T)`, `f_reltol = 0` and
  `max_iterations = 200` from `x₀ = 0`. `device` is `Float32` on the device with
  `LUFactorization()` as `solve` dispatches it: MPS on Metal, cuSOLVER on CUDA, rocSOLVER on ROCm
  (`GeometricSolversAMDGPUExt`), and the package's own KernelAbstractions LU on `JLArray`. `host`
  is `Float32` with LAPACK.
  `device-mixed` is a `Float64` solve on the device with `LUFactorization(Float32)` and
  `NoRefinement()`, `device-mixed-refine` the same with `Refinement()`, `device-f64` a `Float64`
  solve with `LUFactorization()`, and `host-f64` a `Float64` solve on the host. `device` is
  compared with `host`, and the three `Float64` device variants with `host-f64`. Device `Float64`
  is never timed (decision 6): `device-f64` is a correctness row. On Metal the three `Float64`
  device variants are not run, because Metal has no `Float64` and the package has no Metal
  hybrid (decision 15).
- **The LU rows.** One factorisation and one solve, timed together, of `J = A + I/10` and `b` of
  the §10.E problem in `Float32` on the device, from `init_linear(LUFactorization(), nothing, b)`.
  `dispatched` is the path of `solve`, `factorize!!` and `ldiv!!`; `own-ka` calls the package's
  internal `ka_factorize_lu!` and `ka_lu_backsolve!`, which no option of `solve` selects while a
  vendor extension's LU applies: the path is internal, not user-selectable. Both are compared with
  the host LAPACK solution of `J d = b`.
- **The cpu mode.** The R2 rows run in full, the grid runs `n_sys ∈ {10³, 10⁴}` × `n ∈ {2, 4, 8}`,
  and the R3 and LU rows `n = 256` and 1024; every other cell is `not run: cpu mode`.

### Y3 · Cards and versions

One line per machine, from its `.txt`.

| machine | backend | device | driver | julia | threads | packages |
|:--|:--|:--|:--|:--|--:|:--|
| cpu | cpu | host apple-m4 | none | 1.13.4 | 16 | GeometricSolvers 0.1.0, KernelAbstractions 0.10.7, StaticArrays 1.9.25, SimpleNonlinearSolve 2.14.6, JLArrays 0.5.0 |
| mac | metal | Apple M4 Max | macOS 27.2.0 | 1.13.4 | 16 | GeometricSolvers 0.1.0, KernelAbstractions 0.10.7, StaticArrays 1.9.25, SimpleNonlinearSolve 2.14.6, JLArrays 0.5.0, Metal 1.12.1 |
| spark | cuda | NVIDIA GB10 | CUDA driver 13.5.0 | 1.13.4 | 20 | GeometricSolvers 0.1.0, KernelAbstractions 0.10.7, StaticArrays 1.9.25, SimpleNonlinearSolve 2.14.6, JLArrays 0.5.0, CUDA 6.5.1 |
| rtx4090 | cuda | NVIDIA GeForce RTX 4090 | CUDA driver 13.5.0 | 1.13.4 | 24 | GeometricSolvers 0.1.0, KernelAbstractions 0.10.7, StaticArrays 1.9.25, SimpleNonlinearSolve 2.14.6, JLArrays 0.5.0, CUDA 6.5.1 |
| rx7900xtx | rocm | AMD Radeon RX 7900 XTX | HIP runtime 7.16.26385 | 1.13.4 | 24 | GeometricSolvers 0.1.0, KernelAbstractions 0.10.7, StaticArrays 1.9.25, SimpleNonlinearSolve 2.14.6, JLArrays 0.5.0, AMDGPU 2.12.0 |

Each `.txt` records the load averages of its machine at the start of the run: 2.81, 3.18 and 6.95
on `cpu`, 19.41, 13.18 and 11.23 on `mac`, 0.93, 0.74 and 0.36 on `spark`, 1.07, 1.14 and 0.59 on
`rtx4090`, and 0.10, 0.04 and 0.01 on `rx7900xtx`. The `cpu` and `mac` runs share the M4 Max, and
the times of `mac` are measured under its load. The `rx7900xtx` run is of a `devices.jl` that
gives SimpleNonlinearSolve's cells on ROCm the status `not run: SimpleNonlinearSolve kernel faults
on ROCm`; the other three cards ran the same script without that status, which no other backend
reaches.

### Y3 · R2 correctness

Per machine, column and package: the cells that ran, the device successes, and the count of each
verdict against the host R1 solve.

| machine | column | package | ran | successes | agree | different root | device fails | host fails | both fail |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| cpu | Newton Static LU | GeometricSolvers | 63 | 57 | 57 | 0 | 1 | 0 | 5 |
| cpu | Newton Static LU | SimpleNonlinearSolve | 62 | 56 | 56 | 0 | 0 | 0 | 6 |
| cpu | Newton Backtracking LU | GeometricSolvers | 63 | 54 | 54 | 0 | 0 | 0 | 9 |
| cpu | Newton DogLeg LU | GeometricSolvers | 63 | 57 | 57 | 0 | 0 | 0 | 6 |
| cpu | Picard Backtracking | GeometricSolvers | 62 | 20 | 20 | 0 | 0 | 0 | 42 |
| mac | Newton Static LU | GeometricSolvers | 72 | 63 | 60 | 2 | 2 | 1 | 7 |
| mac | Newton Static LU | SimpleNonlinearSolve | 68 | 62 | 61 | 0 | 1 | 1 | 5 |
| mac | Newton Backtracking LU | GeometricSolvers | 70 | 59 | 57 | 2 | 0 | 0 | 11 |
| mac | Newton DogLeg LU | GeometricSolvers | 67 | 61 | 60 | 1 | 0 | 0 | 6 |
| mac | Picard Backtracking | GeometricSolvers | 67 | 24 | 24 | 0 | 0 | 0 | 43 |
| spark | Newton Static LU | GeometricSolvers | 70 | 63 | 61 | 2 | 1 | 0 | 6 |
| spark | Newton Static LU | SimpleNonlinearSolve | 68 | 63 | 61 | 0 | 1 | 2 | 4 |
| spark | Newton Backtracking LU | GeometricSolvers | 69 | 59 | 59 | 0 | 1 | 0 | 9 |
| spark | Newton DogLeg LU | GeometricSolvers | 69 | 63 | 63 | 0 | 0 | 0 | 6 |
| spark | Picard Backtracking | GeometricSolvers | 68 | 25 | 24 | 1 | 0 | 0 | 43 |
| rtx4090 | Newton Static LU | GeometricSolvers | 70 | 63 | 61 | 2 | 1 | 0 | 6 |
| rtx4090 | Newton Static LU | SimpleNonlinearSolve | 68 | 63 | 61 | 0 | 1 | 2 | 4 |
| rtx4090 | Newton Backtracking LU | GeometricSolvers | 69 | 59 | 59 | 0 | 1 | 0 | 9 |
| rtx4090 | Newton DogLeg LU | GeometricSolvers | 69 | 63 | 63 | 0 | 0 | 0 | 6 |
| rtx4090 | Picard Backtracking | GeometricSolvers | 68 | 25 | 24 | 1 | 0 | 0 | 43 |
| rx7900xtx | Newton Static LU | GeometricSolvers | 45 | 43 | 42 | 0 | 0 | 1 | 2 |
| rx7900xtx | Newton Static LU | SimpleNonlinearSolve | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx7900xtx | Newton Backtracking LU | GeometricSolvers | 45 | 39 | 39 | 0 | 0 | 0 | 6 |
| rx7900xtx | Newton DogLeg LU | GeometricSolvers | 45 | 41 | 41 | 0 | 0 | 0 | 4 |
| rx7900xtx | Picard Backtracking | GeometricSolvers | 45 | 12 | 12 | 0 | 0 | 0 | 33 |

The cells that ran differ between the machines: a row above the band, a cell that is not run, a
kernel that does not compile and a kernel that throws are not counted (*Not run* below), so two
machines' counts compare only over the rows that ran on both.

### Y3 · R2 throughput

Per machine, `n` and variant: solves per second, `n_sys / median_s`, at each `n_sys`; `—` for a
cell that did not run. The cpu mode runs the grid on the host's cores through `CPU()`, so its
throughput is no device throughput. On a card the four device variants run on the card and
`host-r1` on the host's threads, whose count is in the cards table; on `rx7900xtx` the three
GeometricSolvers device variants run up to the band, `n = 8`, `sns` does not run, and `host-r1`
runs at every `n`. A throughput is of one run of one machine at the load of the cards table, and
two machines differ in card, host and threads.

| machine | n | variant | 10³ | 10⁴ | 10⁵ | 10⁶ | 10⁷ |
|:--|--:|:--|--:|--:|--:|--:|--:|
| cpu | 2 | gs-newton | 2.53e+07 | 5.63e+07 | — | — | — |
| cpu | 2 | gs-reuse4 | 3.06e+07 | 7.45e+07 | — | — | — |
| cpu | 2 | gs-backtracking | 2.10e+07 | 6.55e+07 | — | — | — |
| cpu | 2 | sns | 2.90e+07 | 7.30e+07 | — | — | — |
| cpu | 2 | host-r1 | 6.85e+06 | 4.38e+07 | — | — | — |
| cpu | 4 | gs-newton | 1.88e+07 | 3.43e+07 | — | — | — |
| cpu | 4 | gs-reuse4 | 2.06e+07 | 3.90e+07 | — | — | — |
| cpu | 4 | gs-backtracking | 1.99e+07 | 3.02e+07 | — | — | — |
| cpu | 4 | sns | 1.84e+07 | 3.17e+07 | — | — | — |
| cpu | 4 | host-r1 | 6.77e+06 | 2.41e+07 | — | — | — |
| cpu | 8 | gs-newton | 9.95e+06 | 1.12e+07 | — | — | — |
| cpu | 8 | gs-reuse4 | 9.02e+06 | 1.29e+07 | — | — | — |
| cpu | 8 | gs-backtracking | 1.01e+07 | 1.15e+07 | — | — | — |
| cpu | 8 | sns | 1.06e+07 | 1.16e+07 | — | — | — |
| cpu | 8 | host-r1 | 5.77e+06 | 9.63e+06 | — | — | — |
| mac | 2 | gs-newton | 5.35e+06 | 4.66e+07 | 2.48e+08 | 5.50e+08 | 4.06e+08 |
| mac | 2 | gs-reuse4 | 4.55e+06 | 3.94e+07 | 2.25e+08 | 2.41e+08 | 4.67e+08 |
| mac | 2 | gs-backtracking | 2.65e+06 | 3.57e+07 | 1.29e+08 | 3.72e+08 | 3.10e+08 |
| mac | 2 | sns | 2.12e+07 | 6.50e+07 | 1.12e+08 | 6.29e+08 | 1.79e+09 |
| mac | 2 | host-r1 | 3.00e+06 | 1.38e+07 | 2.51e+07 | 3.59e+07 | 5.53e+07 |
| mac | 4 | gs-newton | 1.61e+06 | 2.21e+07 | 7.82e+07 | 9.00e+07 | 3.14e+08 |
| mac | 4 | gs-reuse4 | 3.00e+06 | 2.64e+07 | 9.25e+07 | 7.50e+07 | 4.09e+08 |
| mac | 4 | gs-backtracking | 3.05e+06 | 2.98e+07 | 1.08e+08 | 1.28e+08 | 1.16e+08 |
| mac | 4 | sns | 6.35e+06 | 4.48e+07 | 1.75e+08 | 3.48e+08 | 3.28e+08 |
| mac | 4 | host-r1 | 2.72e+06 | 7.98e+06 | 1.01e+07 | 1.35e+07 | 1.51e+07 |
| mac | 6 | gs-newton | 7.02e+05 | 1.15e+07 | 3.41e+07 | 8.32e+07 | 9.43e+07 |
| mac | 6 | gs-reuse4 | 7.50e+05 | 7.21e+06 | 2.23e+07 | 8.87e+07 | 6.87e+07 |
| mac | 6 | gs-backtracking | 8.52e+05 | 9.16e+06 | 3.46e+07 | 1.43e+08 | 1.03e+08 |
| mac | 6 | sns | 1.59e+06 | 4.71e+07 | 6.21e+07 | 7.50e+07 | 8.59e+07 |
| mac | 6 | host-r1 | 2.44e+05 | 2.25e+06 | 3.86e+06 | 1.23e+07 | 1.07e+07 |
| mac | 8 | gs-newton | 8.32e+05 | 7.63e+06 | 6.29e+07 | 9.78e+07 | 7.12e+07 |
| mac | 8 | gs-reuse4 | 6.46e+05 | 1.53e+07 | 3.20e+07 | 3.70e+07 | 3.51e+07 |
| mac | 8 | gs-backtracking | 8.20e+05 | 8.21e+06 | 2.15e+07 | 9.48e+07 | 6.12e+07 |
| mac | 8 | sns | 5.55e+05 | 4.07e+06 | 1.92e+07 | 2.92e+07 | 2.97e+07 |
| mac | 8 | host-r1 | 4.46e+06 | 8.46e+06 | 9.12e+06 | 9.49e+06 | 1.08e+07 |
| mac | 12 | gs-newton | 7.58e+05 | 6.52e+06 | 1.56e+07 | 1.74e+07 | 1.56e+07 |
| mac | 12 | gs-reuse4 | 7.11e+05 | 7.91e+06 | 1.01e+07 | 1.15e+07 | 1.10e+07 |
| mac | 12 | gs-backtracking | 6.63e+05 | 9.66e+06 | 1.55e+07 | 1.69e+07 | 1.54e+07 |
| mac | 12 | sns | 4.98e+05 | 4.57e+06 | 5.45e+06 | 5.40e+06 | 5.33e+06 |
| mac | 12 | host-r1 | 2.19e+06 | 2.24e+06 | 3.18e+06 | 4.01e+06 | 4.51e+06 |
| mac | 16 | gs-newton | 4.90e+05 | 4.32e+06 | 5.02e+06 | 5.61e+06 | 5.52e+06 |
| mac | 16 | gs-reuse4 | 4.85e+05 | 4.20e+06 | 4.16e+06 | 4.19e+06 | 4.07e+06 |
| mac | 16 | gs-backtracking | 4.80e+05 | 4.20e+06 | 4.97e+06 | 5.48e+06 | 5.38e+06 |
| mac | 16 | host-r1 | 1.56e+06 | 1.76e+06 | 1.74e+06 | 2.17e+06 | 2.44e+06 |
| spark | 2 | gs-newton | 4.50e+07 | 3.70e+08 | 1.13e+08 | 5.15e+08 | 6.01e+08 |
| spark | 2 | gs-reuse4 | 2.80e+07 | 2.13e+08 | 1.15e+08 | 1.51e+08 | 1.55e+08 |
| spark | 2 | gs-backtracking | 3.99e+07 | 3.63e+08 | 2.62e+08 | 4.66e+08 | 5.04e+08 |
| spark | 2 | sns | 8.16e+07 | 8.58e+08 | 1.95e+09 | 2.05e+09 | 1.99e+09 |
| spark | 2 | host-r1 | 2.87e+07 | 6.68e+07 | 5.97e+07 | 9.28e+07 | 1.09e+08 |
| spark | 4 | gs-newton | 2.35e+07 | 1.98e+08 | 1.03e+08 | 1.36e+08 | 1.36e+08 |
| spark | 4 | gs-reuse4 | 7.72e+06 | 3.30e+07 | 4.61e+07 | 4.98e+07 | 4.96e+07 |
| spark | 4 | gs-backtracking | 2.52e+07 | 1.55e+08 | 9.50e+07 | 1.14e+08 | 1.18e+08 |
| spark | 4 | sns | 1.96e+07 | 1.41e+08 | 8.02e+07 | 9.52e+07 | 9.51e+07 |
| spark | 4 | host-r1 | 2.05e+07 | 1.92e+07 | 2.14e+07 | 3.32e+07 | 3.86e+07 |
| spark | 6 | gs-newton | 2.18e+06 | 1.51e+07 | 4.79e+07 | 5.40e+07 | 5.60e+07 |
| spark | 6 | gs-reuse4 | 3.84e+06 | 1.27e+07 | 2.04e+07 | 2.26e+07 | 2.28e+07 |
| spark | 6 | gs-backtracking | 8.66e+06 | 4.70e+07 | 4.84e+07 | 5.25e+07 | 5.17e+07 |
| spark | 6 | sns | 5.85e+06 | 1.65e+07 | 2.23e+07 | 2.56e+07 | 2.58e+07 |
| spark | 6 | host-r1 | 1.46e+07 | 9.55e+06 | 1.68e+07 | 2.19e+07 | 2.44e+07 |
| spark | 8 | gs-newton | 1.25e+06 | 1.27e+07 | 3.01e+07 | 3.26e+07 | 3.38e+07 |
| spark | 8 | gs-reuse4 | 3.17e+06 | 7.68e+06 | 1.15e+07 | 1.25e+07 | 1.24e+07 |
| spark | 8 | gs-backtracking | 4.51e+06 | 1.27e+07 | 2.47e+07 | 2.87e+07 | 2.94e+07 |
| spark | 8 | sns | 4.82e+06 | 9.15e+06 | 1.11e+07 | 1.13e+07 | 1.13e+07 |
| spark | 8 | host-r1 | 9.90e+06 | 7.58e+06 | 1.07e+07 | 1.30e+07 | 1.48e+07 |
| spark | 12 | gs-newton | 2.22e+06 | 9.68e+06 | 1.19e+07 | 1.24e+07 | 1.25e+07 |
| spark | 12 | gs-reuse4 | 1.23e+06 | 4.39e+06 | 4.96e+06 | 5.08e+06 | 5.09e+06 |
| spark | 12 | gs-backtracking | 1.89e+06 | 5.83e+06 | 9.34e+06 | 1.00e+07 | 1.01e+07 |
| spark | 12 | sns | 1.34e+06 | 3.05e+06 | 3.52e+06 | 3.58e+06 | 3.59e+06 |
| spark | 12 | host-r1 | 4.00e+06 | 4.84e+06 | 5.98e+06 | 7.35e+06 | 7.96e+06 |
| spark | 16 | gs-newton | 8.37e+05 | 4.10e+06 | 5.11e+06 | 5.39e+06 | 5.43e+06 |
| spark | 16 | gs-reuse4 | 5.91e+05 | 2.13e+06 | 2.37e+06 | 2.45e+06 | 2.46e+06 |
| spark | 16 | gs-backtracking | 8.48e+05 | 3.63e+06 | 3.93e+06 | 4.13e+06 | 4.16e+06 |
| spark | 16 | host-r1 | 1.35e+06 | 2.67e+06 | 3.54e+06 | 4.48e+06 | 4.70e+06 |
| rtx4090 | 2 | gs-newton | 6.35e+07 | 6.18e+08 | 1.68e+09 | 2.10e+09 | 2.24e+09 |
| rtx4090 | 2 | gs-reuse4 | 4.68e+07 | 4.58e+08 | 6.76e+08 | 6.61e+08 | 7.09e+08 |
| rtx4090 | 2 | gs-backtracking | 6.02e+07 | 5.87e+08 | 1.47e+09 | 1.55e+09 | 1.80e+09 |
| rtx4090 | 2 | sns | 1.49e+08 | 9.65e+08 | 5.04e+09 | 5.94e+09 | 6.40e+09 |
| rtx4090 | 2 | host-r1 | 5.40e+07 | 8.82e+07 | 9.42e+07 | 1.36e+08 | 1.40e+08 |
| rtx4090 | 4 | gs-newton | 3.91e+07 | 3.76e+08 | 7.42e+08 | 9.06e+08 | 1.01e+09 |
| rtx4090 | 4 | gs-reuse4 | 1.14e+07 | 1.16e+08 | 1.85e+08 | 2.19e+08 | 2.29e+08 |
| rtx4090 | 4 | gs-backtracking | 3.49e+07 | 3.32e+08 | 6.22e+08 | 6.71e+08 | 7.24e+08 |
| rtx4090 | 4 | sns | 2.92e+07 | 2.87e+08 | 4.45e+08 | 4.21e+08 | 4.49e+08 |
| rtx4090 | 4 | host-r1 | 2.25e+07 | 2.76e+07 | 3.13e+07 | 2.56e+07 | 3.85e+07 |
| rtx4090 | 6 | gs-newton | 1.45e+07 | 1.31e+08 | 2.62e+08 | 2.93e+08 | 3.05e+08 |
| rtx4090 | 6 | gs-reuse4 | 7.04e+06 | 6.98e+07 | 8.31e+07 | 9.69e+07 | 1.01e+08 |
| rtx4090 | 6 | gs-backtracking | 1.17e+07 | 1.14e+08 | 2.07e+08 | 2.31e+08 | 2.39e+08 |
| rtx4090 | 6 | sns | 1.14e+07 | 1.00e+08 | 8.77e+07 | 9.80e+07 | 1.01e+08 |
| rtx4090 | 6 | host-r1 | 1.36e+07 | 1.58e+07 | 1.24e+07 | 2.14e+07 | 2.25e+07 |
| rtx4090 | 8 | gs-newton | 8.31e+06 | 7.89e+07 | 1.20e+08 | 1.43e+08 | 1.47e+08 |
| rtx4090 | 8 | gs-reuse4 | 4.26e+06 | 4.23e+07 | 4.64e+07 | 5.07e+07 | 5.19e+07 |
| rtx4090 | 8 | gs-backtracking | 6.68e+06 | 6.38e+07 | 9.88e+07 | 1.16e+08 | 1.20e+08 |
| rtx4090 | 8 | sns | 7.28e+06 | 5.00e+07 | 4.30e+07 | 4.49e+07 | 4.51e+07 |
| rtx4090 | 8 | host-r1 | 8.88e+06 | 9.90e+06 | 8.90e+06 | 1.17e+07 | 1.31e+07 |
| rtx4090 | 12 | gs-newton | 2.77e+06 | 2.61e+07 | 3.92e+07 | 4.49e+07 | 4.55e+07 |
| rtx4090 | 12 | gs-reuse4 | 1.87e+06 | 1.00e+07 | 1.85e+07 | 2.01e+07 | 2.03e+07 |
| rtx4090 | 12 | gs-backtracking | 2.45e+06 | 2.15e+07 | 3.23e+07 | 3.72e+07 | 3.76e+07 |
| rtx4090 | 12 | sns | 2.73e+06 | 1.26e+07 | 1.31e+07 | 1.34e+07 | 1.35e+07 |
| rtx4090 | 12 | host-r1 | 4.53e+06 | 5.94e+06 | 5.91e+06 | 6.23e+06 | 7.04e+06 |
| rtx4090 | 16 | gs-newton | 1.11e+06 | 1.01e+07 | 1.67e+07 | 2.06e+07 | 2.11e+07 |
| rtx4090 | 16 | gs-reuse4 | 8.86e+05 | 6.92e+06 | 9.22e+06 | 9.95e+06 | 1.01e+07 |
| rtx4090 | 16 | gs-backtracking | 9.97e+05 | 8.07e+06 | 1.32e+07 | 1.54e+07 | 1.56e+07 |
| rtx4090 | 16 | host-r1 | 1.80e+06 | 1.30e+06 | 3.01e+06 | 3.52e+06 | 3.95e+06 |
| rx7900xtx | 2 | gs-newton | 2.65e+07 | 2.14e+08 | 1.26e+09 | 2.30e+09 | 2.05e+09 |
| rx7900xtx | 2 | gs-reuse4 | 1.50e+07 | 1.49e+08 | 3.94e+08 | 4.47e+08 | 7.11e+08 |
| rx7900xtx | 2 | gs-backtracking | 3.49e+07 | 3.23e+08 | 1.17e+09 | 3.90e+09 | 2.32e+09 |
| rx7900xtx | 2 | host-r1 | 2.62e+07 | 6.88e+07 | 7.30e+07 | 8.45e+07 | 1.06e+08 |
| rx7900xtx | 4 | gs-newton | 6.04e+06 | 7.10e+07 | 1.90e+08 | 2.49e+08 | 4.58e+08 |
| rx7900xtx | 4 | gs-reuse4 | 1.97e+06 | 1.90e+07 | 8.76e+07 | 1.56e+08 | 1.68e+08 |
| rx7900xtx | 4 | gs-backtracking | 8.02e+06 | 6.62e+07 | 1.81e+08 | 2.07e+08 | 3.74e+08 |
| rx7900xtx | 4 | host-r1 | 1.92e+07 | 2.30e+07 | 2.68e+07 | 3.12e+07 | 4.34e+07 |
| rx7900xtx | 6 | gs-newton | 1.70e+06 | 1.82e+07 | 8.24e+07 | 1.45e+08 | 2.72e+08 |
| rx7900xtx | 6 | gs-reuse4 | 1.79e+06 | 1.77e+07 | 6.16e+07 | 8.62e+07 | 9.59e+07 |
| rx7900xtx | 6 | gs-backtracking | 2.15e+06 | 2.09e+07 | 7.87e+07 | 1.52e+08 | 1.98e+08 |
| rx7900xtx | 6 | host-r1 | 1.04e+07 | 1.33e+07 | 1.36e+07 | 1.62e+07 | 2.23e+07 |
| rx7900xtx | 8 | gs-newton | 7.80e+05 | 1.03e+07 | 4.30e+07 | 6.05e+07 | 6.23e+07 |
| rx7900xtx | 8 | gs-reuse4 | 7.50e+05 | 8.87e+06 | 1.66e+07 | 1.98e+07 | 2.08e+07 |
| rx7900xtx | 8 | gs-backtracking | 7.74e+05 | 9.38e+06 | 3.05e+07 | 4.45e+07 | 4.61e+07 |
| rx7900xtx | 8 | host-r1 | 8.26e+06 | 8.18e+06 | 1.45e+07 | 1.21e+07 | 1.56e+07 |
| rx7900xtx | 12 | host-r1 | 2.48e+06 | 3.53e+06 | 4.33e+06 | 5.74e+06 | 5.96e+06 |
| rx7900xtx | 16 | host-r1 | 1.53e+06 | 2.26e+06 | 2.35e+06 | 2.97e+06 | 3.23e+06 |

### Y3 · R3 rows

Per machine, `n` and variant: the status, the iterations, the success and `median_s`, in seconds;
`—` for a cell that is not timed. On `cpu` the device is a `JLArray`, whose kernels run on the
host. `host` and `host-f64` run LAPACK on the host of their machine with one BLAS thread, and the
device variants run on the card; a time is of one run of one machine.

| machine | n | variant | status | iterations | success | median_s |
|:--|--:|:--|:--|--:|:--|--:|
| cpu | 256 | device | ran | 2 | true | 0.00846 |
| cpu | 256 | host | ran | 2 | true | 0.000513 |
| cpu | 256 | device-mixed | ran | 3 | true | 0.0135 |
| cpu | 256 | device-mixed-refine | ran | 3 | true | 0.00933 |
| cpu | 256 | device-f64 | ran | 3 | true | — |
| cpu | 256 | host-f64 | ran | 3 | true | 0.00132 |
| cpu | 1024 | device | ran | 2 | true | 0.0716 |
| cpu | 1024 | host | ran | 2 | true | 0.0176 |
| cpu | 1024 | device-mixed | ran | 3 | true | 0.134 |
| cpu | 1024 | device-mixed-refine | ran | 3 | true | 0.132 |
| cpu | 1024 | device-f64 | ran | 3 | true | — |
| cpu | 1024 | host-f64 | ran | 3 | true | 0.0476 |
| cpu | 4096 | device | not run: cpu mode | - | - | — |
| cpu | 4096 | host | not run: cpu mode | - | - | — |
| cpu | 4096 | device-mixed | not run: cpu mode | - | - | — |
| cpu | 4096 | device-mixed-refine | not run: cpu mode | - | - | — |
| cpu | 4096 | device-f64 | not run: cpu mode | - | - | — |
| cpu | 4096 | host-f64 | not run: cpu mode | - | - | — |
| mac | 256 | device | ran | 2 | true | 0.0120 |
| mac | 256 | host | ran | 2 | true | 0.000541 |
| mac | 256 | device-mixed | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 256 | device-mixed-refine | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 256 | device-f64 | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 256 | host-f64 | ran | 3 | true | 0.00131 |
| mac | 1024 | device | ran | 2 | true | 0.0678 |
| mac | 1024 | host | ran | 2 | true | 0.0184 |
| mac | 1024 | device-mixed | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 1024 | device-mixed-refine | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 1024 | device-f64 | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 1024 | host-f64 | ran | 3 | true | 0.0492 |
| mac | 4096 | device | ran | 2 | true | 0.743 |
| mac | 4096 | host | ran | 2 | true | 0.921 |
| mac | 4096 | device-mixed | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 4096 | device-mixed-refine | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 4096 | device-f64 | no Float64 on Metal; no Metal hybrid (decision 15) | - | - | — |
| mac | 4096 | host-f64 | ran | 3 | true | 2.65 |
| spark | 256 | device | ran | 2 | true | 0.00295 |
| spark | 256 | host | ran | 2 | true | 0.000411 |
| spark | 256 | device-mixed | ran | 3 | true | 0.00236 |
| spark | 256 | device-mixed-refine | ran | 3 | true | 0.00388 |
| spark | 256 | device-f64 | ran | 3 | true | — |
| spark | 256 | host-f64 | ran | 3 | true | 0.00101 |
| spark | 1024 | device | ran | 2 | true | 0.00674 |
| spark | 1024 | host | ran | 2 | true | 0.0157 |
| spark | 1024 | device-mixed | ran | 3 | true | 0.0119 |
| spark | 1024 | device-mixed-refine | ran | 3 | true | 0.0155 |
| spark | 1024 | device-f64 | ran | 3 | true | — |
| spark | 1024 | host-f64 | ran | 3 | true | 0.0428 |
| spark | 4096 | device | ran | 2 | true | 0.0415 |
| spark | 4096 | host | ran | 2 | true | 0.791 |
| spark | 4096 | device-mixed | ran | 3 | true | 0.0833 |
| spark | 4096 | device-mixed-refine | ran | 3 | true | 0.0902 |
| spark | 4096 | device-f64 | ran | 3 | true | — |
| spark | 4096 | host-f64 | ran | 3 | true | 2.29 |
| rtx4090 | 256 | device | ran | 2 | true | 0.00100 |
| rtx4090 | 256 | host | ran | 2 | true | 0.000338 |
| rtx4090 | 256 | device-mixed | ran | 3 | true | 0.00196 |
| rtx4090 | 256 | device-mixed-refine | ran | 3 | true | 0.00239 |
| rtx4090 | 256 | device-f64 | ran | 3 | true | — |
| rtx4090 | 256 | host-f64 | ran | 3 | true | 0.000942 |
| rtx4090 | 1024 | device | ran | 2 | true | 0.00395 |
| rtx4090 | 1024 | host | ran | 2 | true | 0.0137 |
| rtx4090 | 1024 | device-mixed | ran | 3 | true | 0.00638 |
| rtx4090 | 1024 | device-mixed-refine | ran | 3 | true | 0.00719 |
| rtx4090 | 1024 | device-f64 | ran | 3 | true | — |
| rtx4090 | 1024 | host-f64 | ran | 3 | true | 0.0365 |
| rtx4090 | 4096 | device | ran | 2 | true | 0.0223 |
| rtx4090 | 4096 | host | ran | 2 | true | 0.648 |
| rtx4090 | 4096 | device-mixed | ran | 3 | true | 0.0421 |
| rtx4090 | 4096 | device-mixed-refine | ran | 3 | true | 0.0410 |
| rtx4090 | 4096 | device-f64 | ran | 3 | true | — |
| rtx4090 | 4096 | host-f64 | ran | 3 | true | 1.90 |
| rx7900xtx | 256 | device | ran | 2 | true | 0.00278 |
| rx7900xtx | 256 | host | ran | 2 | true | 0.000554 |
| rx7900xtx | 256 | device-mixed | ran | 3 | true | 0.00504 |
| rx7900xtx | 256 | device-mixed-refine | ran | 3 | true | 0.00611 |
| rx7900xtx | 256 | device-f64 | ran | 3 | true | — |
| rx7900xtx | 256 | host-f64 | ran | 3 | true | 0.00105 |
| rx7900xtx | 1024 | device | ran | 2 | true | 0.00897 |
| rx7900xtx | 1024 | host | ran | 2 | true | 0.0158 |
| rx7900xtx | 1024 | device-mixed | ran | 3 | true | 0.0145 |
| rx7900xtx | 1024 | device-mixed-refine | ran | 3 | true | 0.0154 |
| rx7900xtx | 1024 | device-f64 | ran | 3 | true | — |
| rx7900xtx | 1024 | host-f64 | ran | 3 | true | 0.0374 |
| rx7900xtx | 4096 | device | ran | 2 | true | 0.0816 |
| rx7900xtx | 4096 | host | ran | 2 | true | 0.709 |
| rx7900xtx | 4096 | device-mixed | ran | 3 | true | 0.129 |
| rx7900xtx | 4096 | device-mixed-refine | ran | 3 | true | 0.132 |
| rx7900xtx | 4096 | device-f64 | ran | 3 | true | — |
| rx7900xtx | 4096 | host-f64 | ran | 3 | true | 1.99 |

#### Y3 · LU rows

Per machine and `n`: `median_s` of one factorisation and one solve by each path, and their ratio
`own-ka / dispatched`. On `cpu` both paths run the package's own KernelAbstractions LU on a
`JLArray`; `dispatched` adds the steps of `ldiv!!` around the backsolve. On the cards
`dispatched` is MPS on `mac`, cuSOLVER on `spark` and `rtx4090` and rocSOLVER on `rx7900xtx`. A
ratio compares the two paths on one card in one run.

| machine | n | own-ka | dispatched | own-ka / dispatched |
|:--|--:|--:|--:|--:|
| cpu | 256 | 0.00423 | 0.00444 | 0.951 |
| cpu | 1024 | 0.0384 | 0.0388 | 0.990 |
| mac | 256 | 0.00288 | 0.00441 | 0.654 |
| mac | 1024 | 0.0168 | 0.0314 | 0.537 |
| mac | 4096 | 0.386 | 0.368 | 1.05 |
| spark | 256 | 0.00162 | 0.000671 | 2.42 |
| spark | 1024 | 0.0114 | 0.00297 | 3.83 |
| spark | 4096 | 0.801 | 0.0171 | 46.8 |
| rtx4090 | 256 | 0.00184 | 0.000358 | 5.15 |
| rtx4090 | 1024 | 0.0100 | 0.00186 | 5.38 |
| rtx4090 | 4096 | 0.128 | 0.0104 | 12.3 |
| rx7900xtx | 256 | 0.00438 | 0.00140 | 3.14 |
| rx7900xtx | 1024 | 0.0182 | 0.00364 | 4.99 |
| rx7900xtx | 4096 | 0.247 | 0.0403 | 6.13 |

### Y3 · Not run

Each status other than `ran`, per machine and kind, with its count of cells and its cause.

| machine | kind | status | count | cause |
|:--|:--|:--|--:|:--|
| cpu | r2 | kernel error: KernelException: exception thrown during kernel execution on device cpu | 62 | the residual allocates on the heap in the kernel: standard problems 1, 8, 10 and 14 at all three starts, 12 rows in five launched cells each; `Picard Backtracking` at LotkaVolterra2d h=0.1, whose kernel allocates on the heap (`gc_pool_alloc` in the device stack trace of `julia -g2`) where the host solve ends `LINESEARCH_FAILED` after one iteration; and SimpleNewtonRaphson's `SingularException` at the singular start of standard problem 12 at `100u0`, where GeometricSolvers returns `SINGULAR` |
| cpu | r2 | no device path | 632 | SimpleSolvers and NLsolve have no device path |
| cpu | r2 | no method: SimpleNewtonRaphson runs in Newton Static LU only | 237 | SimpleNonlinearSolve has one device method, paired with `Newton Static LU` |
| cpu | r2 | no compile: KernelError: GPU compilation of MethodInstance for GeometricSolvers.gpu__solve_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.Dyna… | 8 | the two DoublePendulum rows: GeometricProblems' vector fields are closures that are not `isbits` |
| cpu | r2 | no compile: KernelError: GPU compilation of MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstr… | 2 | the two DoublePendulum rows, as above |
| cpu | r2 | no compile: InvalidIRError: compiling MethodInstance for GeometricSolvers.gpu__solve_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSiz… | 6 | the two TodaLattice rows (n = 32) in the three Newton columns |
| cpu | r2 | no compile: InvalidIRError: compiling MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstraction… | 2 | the two TodaLattice rows (n = 32) |
| cpu | r2 | no compile: ErrorException: Failed to compile program | 2 | the two TodaLattice rows (n = 32) in `Picard Backtracking`: POCL's compile failure |
| cpu | grid | not run: cpu mode | 120 | the cpu mode runs 30 of the 150 grid cells |
| cpu | r3 | not run: cpu mode | 6 | the cpu mode runs `n = 256` and 1024 |
| cpu | lu | not run: cpu mode | 2 | the cpu mode runs `n = 256` and 1024 |
| mac | r2 | no device path | 632 | SimpleSolvers and NLsolve have no device path |
| mac | r2 | kernel error: KernelException: An exception was thrown by thread 1×1×1 in threadgroup 1×1×1 on device Apple M4 Max | 22 | with the 9 cells of the next line, whose message names thread 33 instead of thread 1: Metal's device `malloc` heap of 1 MiB per process fills within the row, in the rows of standard problems 1 and 8 at all three starts and of problem 14 at `10u0` and `100u0`, which start on an empty heap (counter 4 B) and end at 1 054 604 to 1 078 884 B (the heap lines of `devices-mac.txt`), 30 cells; and SimpleNewtonRaphson at the singular start of standard problem 12 at `100u0`, where the cpu mode's kernel throws its `SingularException` |
| mac | r2 | kernel error: KernelException: An exception was thrown by thread 33×1×1 in threadgroup 1×1×1 on device Apple M4 Max | 9 | as in the line above |
| mac | r2 | no method: SimpleNewtonRaphson runs in Newton Static LU only | 237 | SimpleNonlinearSolve has one device method, paired with `Newton Static LU` |
| mac | r2 | no compile: KernelError: GPU compilation of MethodInstance for GeometricSolvers.gpu__solve_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.Dyna… | 8 | the two DoublePendulum rows: GeometricProblems' vector fields are closures that are not `isbits` |
| mac | r2 | no compile: KernelError: GPU compilation of MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstr… | 2 | the two DoublePendulum rows, as above |
| mac | r2 | above band 24 | 10 | the two TodaLattice rows (n = 32) in their five launched cells |
| mac | grid | no compile: InvalidIRError: compiling MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstraction… | 5 | `sns` at `n = 16`, at each `n_sys`: SimpleNewtonRaphson's kernel does not compile at `n = 16` |
| mac | r3 | no Float64 on Metal; no Metal hybrid (decision 15) | 9 | the three `Float64` device variants at each `n`: Metal has no `Float64`, and the package has no Metal hybrid (decision 15) |
| spark | r2 | no device path | 632 | SimpleSolvers and NLsolve have no device path |
| spark | r2 | kernel error: KernelException: exception thrown during kernel execution on device NVIDIA GB10 | 31 | the rows of standard problems 1 and 8 at all three starts, 29 cells, whose kernels allocate on the device heap: the run's log prints `Out of dynamic GPU memory (trying to allocate 40 bytes)` with these errors; SimpleNewtonRaphson at the singular start of standard problem 12 at `100u0`, where the cpu mode's kernel throws its `SingularException`; and `Picard Backtracking` at LotkaVolterra2d h=0.1, whose kernel allocates on the heap in the cpu mode as well |
| spark | r2 | no method: SimpleNewtonRaphson runs in Newton Static LU only | 237 | SimpleNonlinearSolve has one device method, paired with `Newton Static LU` |
| spark | r2 | no compile: KernelError: GPU compilation of MethodInstance for GeometricSolvers.gpu__solve_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.Dyna… | 8 | the two DoublePendulum rows: GeometricProblems' vector fields are closures that are not `isbits` |
| spark | r2 | no compile: KernelError: GPU compilation of MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstr… | 2 | the two DoublePendulum rows, as above |
| spark | r2 | above band 24 | 10 | the two TodaLattice rows (n = 32) in their five launched cells |
| spark | grid | no compile: InvalidIRError: compiling MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstraction… | 5 | `sns` at `n = 16`, at each `n_sys`: SimpleNewtonRaphson's kernel does not compile at `n = 16` |
| rtx4090 | r2 | no device path | 632 | SimpleSolvers and NLsolve have no device path |
| rtx4090 | r2 | kernel error: KernelException: exception thrown during kernel execution on device NVIDIA GeForce RTX 4090 | 31 | the same 31 cells as on `spark`, for the same causes; this run's log prints `Out of dynamic GPU memory (trying to allocate 40 bytes)` as well |
| rtx4090 | r2 | no method: SimpleNewtonRaphson runs in Newton Static LU only | 237 | SimpleNonlinearSolve has one device method, paired with `Newton Static LU` |
| rtx4090 | r2 | no compile: KernelError: GPU compilation of MethodInstance for GeometricSolvers.gpu__solve_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.Dyna… | 8 | the two DoublePendulum rows: GeometricProblems' vector fields are closures that are not `isbits` |
| rtx4090 | r2 | no compile: KernelError: GPU compilation of MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstr… | 2 | the two DoublePendulum rows, as above |
| rtx4090 | r2 | above band 24 | 10 | the two TodaLattice rows (n = 32) in their five launched cells |
| rtx4090 | grid | no compile: InvalidIRError: compiling MethodInstance for gpu_sns_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize; KernelAbstraction… | 5 | `sns` at `n = 16`, at each `n_sys`: SimpleNewtonRaphson's kernel does not compile at `n = 16` |
| rx7900xtx | r2 | above band 8 | 160 | the 32 rows of dimension above 8 in their five launched cells: standard problems 1, 8 to 14 and 23 (n = 10) and 16 (n = 9) at all three starts, and the two TodaLattice rows (n = 32) |
| rx7900xtx | r2 | no device path | 632 | SimpleSolvers and NLsolve have no device path |
| rx7900xtx | r2 | no method: SimpleNewtonRaphson runs in Newton Static LU only | 237 | SimpleNonlinearSolve has one device method, paired with `Newton Static LU` |
| rx7900xtx | r2 | not run: SimpleNonlinearSolve kernel faults on ROCm | 47 | SimpleNonlinearSolve's cell of every row within the band: its kernel faults with `hipErrorIllegalAddress` on this card, at `standard/2/u0` (n = 4) as the first kernel of a process and after others, with 1 thread and with 24, and the fault leaves every later HIP call of the process failing; the same kernel returns `Success` on CUDA |
| rx7900xtx | r2 | no compile: KernelError: GPU compilation of MethodInstance for GeometricSolvers.gpu__solve_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.Dyna… | 8 | the two DoublePendulum rows: GeometricProblems' vector fields are closures that are not `isbits` |
| rx7900xtx | grid | not run: SimpleNonlinearSolve kernel faults on ROCm | 20 | `sns` at `n` = 2, 4, 6 and 8, at each `n_sys`: the fault of the line above |
| rx7900xtx | grid | above band 8 | 40 | the four device variants at `n = 12` and 16, at each `n_sys`; `host-r1` runs on the host and has no band |

The midpoint residual of the four Lotka–Volterra rows passes the iterate `V` to GeometricProblems'
`ϑ` and `f` as an `SVector`, and GeometricProblems 0.10.2 inlines LotkaVolterra4d's `ϑ`, so the
residual does not allocate on the heap and these rows run in a `CPU()` kernel; only the
`Picard Backtracking` cell of LotkaVolterra2d h=0.1 ends in the kernel error of the table's first
line, and on CUDA in a kernel error as well; on Metal and ROCm it runs and ends
`LINESEARCH_FAILED` after one iteration, as the host solve does. With the `MVector` `V` in these
calls all four rows allocate, and with GeometricProblems 0.10.0, whose LotkaVolterra4d `ϑ` passes
its output vector to a call that is not inlined, LotkaVolterra4d still allocates.
GeometricProblems 0.10.2 differs from 0.10.0, with which Y1's and Y2's tables were made, only in
that `@inline`. Those tables are not made again with 0.10.2, and `verify.jl` passes on the
committed tables.

### Y3 · Explanations

One line per `different root`, `device fails` and `host fails` verdict, from a host replay of the
cell with `max_iterations = j` (`maxiters = j` for SimpleNewtonRaphson), `j` = 1, 2, 3, 5, 10, 20,
40, 67, 100 and 200 and the iteration counts of the cell, beside the `CPU()` kernel's batch of one
system with the same limit, and from the device's iterations, status, `‖F‖∞` and
`‖x − x_host‖∞` in its machine's CSV line. The replay runs on the M4 Max's host, and the host
fields of each line below are equal on every machine where the cell is launched. On `cpu` the
`CPU()` kernel is the device; a card's CSV holds the norms of its iterate, not the iterate, so a
line states where the card's path ends and not where it leaves the host's. The agreement bound is
`√eps(Float32) · max(1, ‖x_host‖∞)`, `√eps(Float32) = 3.45e-4`.

Problem 18 (Y1's E1) has two lines of roots, `x₁ = 0` and `x₂ = 0`. At the host's end points on
`x₁ = 0` its Jacobian is near singular: its singular values are 0.0896 and 6.4e-8 at the end
point from `u0`, and 1.04 and 9.0e-7 at the end point of `Newton Backtracking LU` from `100u0`, so
the last steps do not fix the point along the line. From `10u0` and `100u0` the full Newton step
doubles `|x₁|` and flips its sign at each iteration, with `|x₂|` at 20 or 200 for at least 70
iterations; `‖x‖∞` reaches 4.6e22 at iteration 79 (`10u0`) and 2.4e23 at iteration 80 (`100u0`),
past `√floatmax(Float32) = 1.8e19`, where `x₁²` overflows. Then `x₂` falls, and the host's
successful solves end at `|x₁|` between 5.8e17 and 9.1e18 with `|x₂| < 1.73e-4`. There
`x₂² < 2.98e-8`, so `exp(−x₂²)` rounds to 1 in `Float32` and `F₂ = x₁ (1 − exp(−x₂²)) / x₂` is 0.
At the host's end points of `Newton Static LU` from `10u0` and `100u0` `‖F‖∞` is 3.0e-27 and
3.8e-27 in `Float32`, and `F₂` is 1.5e15 and −1.35e15 in `Float64`: they are roots of the
`Float32` residual near the line `x₂ = 0`, and where a solve ends on it depends on the rounding of
each step. The `CPU()` kernel's path from `10u0` lies 1.3e16 from the host's at iteration 85.

| machine | kind | set | problem | start | variant | package | verdict | explanation |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| cpu | r2 | standard | 11 | 10u0 | Newton Static LU | GeometricSolvers | device fails | The undamped Newton step amplifies a rounding difference between the kernel and the host solve: after one step the iterates differ by 4.2e-7, after three by 2.9e-3, and after ten by 314, the size of the root. The host reaches the root, `‖x‖∞ = 314`, with `SUCCESS` in 67 iterations; the device stops with `MAXITERS` after 200 iterations at `‖F‖∞ = 1.52`. |
| mac | r2 | standard | 11 | 10u0 | Newton Static LU | GeometricSolvers | device fails | As on `cpu`: the host reaches the root `‖x‖∞ = 314` with `SUCCESS` in 67 iterations, along a path whose `‖F‖∞` is 21.5 at iteration 3 and 9.81 at iteration 20, and the `CPU()` kernel's path leaves the host's by 4.2e-7 after one step and by 314 after ten. The card stops with `MAXITERS` after 200 iterations at `‖F‖∞ = 0.0211`, 289 from the host's root. |
| mac | r2 | standard | 18 | u0 | Newton Static LU | GeometricSolvers | different root | Problem 18 near the line `x₁ = 0` (above). The host and the `CPU()` kernel end after 5 iterations at `(−3.7e-7, −0.0885)`, `‖F‖∞ = 3.2e-8`. The card ends with `SUCCESS` after 5 iterations at `‖F‖∞ = 2.3e-6`, 5.0e-4 from the host's point, where the bound is 3.45e-4. |
| mac | r2 | standard | 18 | u0 | Newton Backtracking LU | GeometricSolvers | different root | As in the line above: on the host the line search takes the full step at each of the 5 iterations, so the iterates equal those of `Newton Static LU`, and the card ends at the same `‖F‖∞ = 2.3e-6`, 5.0e-4 from the host's point. |
| mac | r2 | standard | 18 | u0 | Newton DogLeg LU | GeometricSolvers | different root | As in the two lines above: on the host the dogleg step is the full Newton step at each of the 5 iterations, so the iterates equal those of `Newton Static LU`, and the card ends at the same `‖F‖∞ = 2.3e-6`, 5.0e-4 from the host's point. |
| mac | r2 | standard | 18 | 10u0 | Newton Static LU | GeometricSolvers | device fails | Problem 18 from `10u0` (above). The card stops with `SUCCESS` after 79 iterations, where the host's path is at `‖x‖∞ = 4.6e22`, and the host's `‖F‖∞` at the card's iterate is 9.06e11. A probe of this cell on the same card, which also stops at iteration 79 with a host `‖F‖∞` of 9.06e11, found the iterate `(−2.93e15, 1.93e-4)`: there the card's `exp(−3.73e-8)` is 1 and the host's 0.99999994, so `F₂` cancels to 0 on the card only. The host ends with `SUCCESS` after 91 iterations at `(−9.08e18, −1.65e-4)`. |
| mac | r2 | standard | 18 | 10u0 | Newton Static LU | SimpleNonlinearSolve | device fails | Problem 18 from `10u0` (above). On the host SimpleNewtonRaphson has the iterates of GeometricSolvers' `Newton Static LU` at each replayed iteration up to 70, at `‖x‖∞ = 2.4e22`, and ends with `Success` at `(5.83e17, 8.9e-5)`, where the `CPU()` kernel ends as well. The card returns `Success` with a `NaN` in system 1's iterate; a probe of this cell on the same card found `(−Inf, NaN)`. |
| mac | r2 | standard | 18 | 100u0 | Newton Static LU | GeometricSolvers | different root | Problem 18 from `100u0` (above). The host and the `CPU()` kernel end with `SUCCESS` after 97 iterations at `(7.85e18, −1.7e-4)`. The card ends with `SUCCESS` after 83 iterations, where the host's path is at `‖x‖∞ = 1.4e23`, at a point 7.85e18 from the host's and with a host `‖F‖∞` of 2.4e-23. |
| mac | r2 | standard | 18 | 100u0 | Newton Static LU | SimpleNonlinearSolve | host fails | Problem 18 from `100u0` (above). On the host SimpleNewtonRaphson has the iterates of GeometricSolvers' `Newton Static LU` at each replayed iteration up to 80, at `‖x‖∞ = 2.4e23`, and returns `Success` with the iterate `(−Inf, NaN)`, as the `CPU()` kernel does. The card returns `Success` at `‖F‖∞ = 1.2e-23`. |
| mac | r2 | standard | 18 | 100u0 | Newton Backtracking LU | GeometricSolvers | different root | Problem 18 near the line `x₁ = 0` (above). From `100u0` the line search takes the full step at the first 3 iterations, as `Newton Static LU` does, and not at every later one; the host and the `CPU()` kernel end after 13 iterations at `(2.2e-6, −0.665)`, `‖F‖∞ = 1.2e-6`. The card ends with `SUCCESS` after 13 iterations at `‖F‖∞ = 2.3e-6`, 1.0e-3 from the host's point, where the bound is 3.45e-4. |
| mac | r2 | standard | 22 | 10u0 | Newton Static LU | GeometricSolvers | host fails | The host's undamped Newton path on problem 22 reaches no root: from iteration 40 its iterates repeat about every four iterations, between `(1.81, 4.15)` and `(1.43, 2.91)` with `‖F‖∞` between 0.84 and 1.57, and it ends with `MAXITERS` after 200. The `CPU()` kernel's path leaves the host's by 1.5e-5 at iteration 5 and by 36 at iteration 20, and also ends with `MAXITERS`. The card ends with `SUCCESS` after 101 iterations at `‖F‖∞ = 1.5e-6`. |
| spark | r2 | standard | 11 | 10u0 | Newton Static LU | GeometricSolvers | device fails | As on `cpu` and `mac`: the host reaches the root `‖x‖∞ = 314` with `SUCCESS` in 67 iterations, and the `CPU()` kernel's path leaves the host's by 4.2e-7 after one step and by 314 after ten. The card stops with `MAXITERS` after 200 iterations at `‖F‖∞ = 0.0094`, 321 from the host's root. |
| spark | r2 | standard | 11 | 10u0 | Newton Static LU | SimpleNonlinearSolve | host fails | The host's SimpleNewtonRaphson takes the full Newton step on problem 11 and reaches no root in 200 iterations: its `‖F‖∞` is 0.39 at iteration 67, 23 at 100 and 1.37 at 200, with `‖x‖∞` growing to 176, and it ends with `MaxIters`. The `CPU()` kernel's path leaves the host's by 6.0e-7 after one step and ends with `MaxIters` as well. The card ends with `Success` at `‖F‖∞ = 4.8e-6`. |
| spark | r2 | standard | 11 | 100u0 | Newton Static LU | SimpleNonlinearSolve | host fails | The host's SimpleNewtonRaphson on problem 11 stays near `‖x‖∞ = 37.8` from iteration 10 on, with `‖F‖∞` between 0.011 and 1.1 at the replayed iterations (0.16 at 40, 0.13 at 67, 0.011 at 200), and ends with `MaxIters`. The `CPU()` kernel's path leaves the host's by 1.0e-5 after one step and ends with `MaxIters` as well. The card ends with `Success` at `‖F‖∞ = 6.0e-7`. |
| spark | r2 | standard | 18 | 10u0 | Newton Static LU | GeometricSolvers | different root | Problem 18 from `10u0` (above). The host and the card end with `SUCCESS` after 91 iterations near the line `x₂ = 0`: the host at `‖x‖∞ = 9.08e18`, the card at a point 2.5e18 from it, with a host `‖F‖∞` of 4.1e-27. |
| spark | r2 | standard | 18 | 10u0 | Newton Static LU | SimpleNonlinearSolve | device fails | Problem 18 from `10u0` (above). On the host SimpleNewtonRaphson has the iterates of GeometricSolvers' `Newton Static LU` at each replayed iteration up to 70, at `‖x‖∞ = 2.4e22`, and ends with `Success` at `(5.83e17, 8.9e-5)`, where the `CPU()` kernel ends as well. The card returns `Success` with a `NaN` in system 1's iterate. |
| spark | r2 | standard | 18 | 100u0 | Newton Static LU | GeometricSolvers | different root | Problem 18 from `100u0` (above). The host and the card end with `SUCCESS` after 97 iterations near the line `x₂ = 0`: the host at `‖x‖∞ = 7.85e18`, the card at a point 5.1e17 from it, with a host `‖F‖∞` of 3.5e-27. |
| spark | r2 | standard | 19 | 100u0 | Picard Backtracking | GeometricSolvers | different root | Problem 19, `F = x ‖x‖²`, has the one root 0, where its Jacobian is 0 (Y1's E2): since `‖F‖∞ ≥ ‖x‖∞³`, a success bounds `‖x‖∞` only by `3.45e-4^(1/3) = 0.070`. The host and the `CPU()` kernel follow one path with `x₁ = x₂`: `‖x‖∞` is 6.65 at iteration 21 and 0.706 at 22, and iteration 23 ends with `SUCCESS` at 0.0031, `‖F‖∞ = 6.1e-8`. The card ends with `SUCCESS` after 23 iterations at `‖F‖∞ = 6.4e-9`, so `‖x‖∞ ≤ 1.9e-3`, and 1.6e-3 from the host's point. |
| spark | r2 | standard | 22 | 10u0 | Newton Backtracking LU | GeometricSolvers | device fails | On problem 22 from `10u0` the host's `‖F‖∞` falls slowly for 14 iterations, from 81 to 53.6, with `x₂` from −79 to −52; iteration 15 takes it to 10.3, and the host ends with `SUCCESS` after 20 iterations at `(−0.707, 1.50)`; the `CPU()` kernel has the same iterates. The card ends with `LINESEARCH_FAILED` after 14 iterations at `‖F‖∞ = 57.8`, in the slow part of that path. |
| rtx4090 | r2 | standard | 11 | 10u0 | Newton Static LU | GeometricSolvers | device fails | As on `spark`, whose line this one equals in code, iterations, `‖F‖∞` and `‖x − x_host‖∞`: the host reaches the root `‖x‖∞ = 314` in 67 iterations, and the card stops with `MAXITERS` after 200 at `‖F‖∞ = 0.0094`, 321 from it. |
| rtx4090 | r2 | standard | 11 | 10u0 | Newton Static LU | SimpleNonlinearSolve | host fails | As on `spark`, whose line this one equals: the host's SimpleNewtonRaphson reaches no root in 200 iterations and ends with `MaxIters` at `‖F‖∞ = 1.37`, and the card ends with `Success` at `‖F‖∞ = 4.8e-6`. |
| rtx4090 | r2 | standard | 11 | 100u0 | Newton Static LU | SimpleNonlinearSolve | host fails | As on `spark`, whose line this one equals: the host's SimpleNewtonRaphson stays near `‖x‖∞ = 37.8` and ends with `MaxIters` at `‖F‖∞ = 0.011`, and the card ends with `Success` at `‖F‖∞ = 6.0e-7`. |
| rtx4090 | r2 | standard | 18 | 10u0 | Newton Static LU | GeometricSolvers | different root | As on `spark`, whose line this one equals: host and card end with `SUCCESS` after 91 iterations at two roots of the `Float32` residual near `x₂ = 0`, 2.5e18 apart, with `‖x_host‖∞ = 9.08e18`. |
| rtx4090 | r2 | standard | 18 | 10u0 | Newton Static LU | SimpleNonlinearSolve | device fails | As on `spark`, whose line this one equals: the host's path passes `‖x‖∞ = 2.4e22` and ends with `Success` at `(5.83e17, 8.9e-5)`, and the card returns `Success` with a `NaN` in system 1's iterate. |
| rtx4090 | r2 | standard | 18 | 100u0 | Newton Static LU | GeometricSolvers | different root | As on `spark`, whose line this one equals: host and card end with `SUCCESS` after 97 iterations at two roots of the `Float32` residual near `x₂ = 0`, 5.1e17 apart, with `‖x_host‖∞ = 7.85e18`. |
| rtx4090 | r2 | standard | 19 | 100u0 | Picard Backtracking | GeometricSolvers | different root | As on `spark`, whose line this one equals: both end with `SUCCESS` after 23 iterations near the singular root 0 of problem 19, the host at `‖x‖∞ = 0.0031` and the card within 1.9e-3 of 0, 1.6e-3 apart. |
| rtx4090 | r2 | standard | 22 | 10u0 | Newton Backtracking LU | GeometricSolvers | device fails | As on `spark`, whose line this one equals: the host ends with `SUCCESS` after 20 iterations, and the card with `LINESEARCH_FAILED` after 14 at `‖F‖∞ = 57.8`, where the host's `‖F‖∞` falls slowly from 81 to 53.6. |
| rx7900xtx | r2 | standard | 22 | 10u0 | Newton Static LU | GeometricSolvers | host fails | As on `mac`: the host's undamped Newton path on problem 22 repeats about every four iterations from iteration 40 and ends with `MAXITERS` after 200, and the `CPU()` kernel's path leaves it at iteration 5 and ends with `MAXITERS` as well. The card ends with `SUCCESS` after 139 iterations at `‖F‖∞ = 3.8e-6`. |
