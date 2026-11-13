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
| `verify.jl <dir> [--structure]` | checks the tables of `<dir>`, this report and the traces with code of its own |
| `selftest.jl` | tests the harness: the pairing, the label and the method of each call, the options of each package, the counting wrapper, the shared Jacobian, the measured `‖F‖∞`, the success rule, the agreement bound, the verdict, a throwing residual, the SVD solve of NLsolve on a non-finite matrix, and the element types and starts of the midpoint rows |
| `check_midpoint.jl` | checks each midpoint residual against one step of `ImplicitMidpoint` |
| `trace.jl <set> <problem> <start> <eltype> <column> [outfile]` | writes the iteration trace of one row |

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
  all. A cell succeeds when its package reports success, its iterate is finite and the harness
  measures `‖F(x)‖∞ ≤ tol`.
- **The labels.** The `method` of a block cell is read from the objects that its adapter passes: a
  GeometricSolvers LU that factors in `Float32` shows as `LUFactorization{Float32}`, a
  `Refinement` as `+Refinement`, a factor reuse of `k` steps as `Newton(k)`, an `Anderson` of depth
  `m` as `Anderson{m}(<member>)`, and `nsol` as `nsol(sham=<sham>)+FPS{<eltype of its Jacobian
  store>}` or `nsol(sham=<sham>)+mplu`.
- **The machine.** Apple M4 Max, aarch64 macOS 27.2, Julia 1.13.3. The machine runs other work, so
  each process prints the `uptime` load at its start. The 1-minute load averages were between 2.37
  and 2.73 for the seven timing processes and between 2.23 and 3.39 for the 16 block processes.
  Over the cells of the timed sweep, the median of `spread` is 0.036 and its 90th percentile is
  0.56. In 34 of the 4661 timed cells, `k` times `median_s` is below 1 ms: both runs that decided
  `k` were slower than the median of the five runs.
- **The versions.** GeometricSolvers from `..`, whose `src/` is that of `301f366`, and the
  packages of Y1 at the versions of Y1. The new packages are pinned in `Project.toml` and resolve
  to MultiPrecisionArrays 0.1.4, SIAMFANLEquations 1.0.3, Sparspak 0.3.15, SparseMatrixColorings
  0.4.29 and ADTypes 1.24.0, with SparseArrays 1.13.0.

| process | 1-minute load at its start |
|:--|--:|
| timing GeometricSolvers Float32 | 2.51 |
| timing GeometricSolvers Float64 | 2.59 |
| timing SimpleSolvers Float32 | 2.37 |
| timing SimpleSolvers Float64 | 2.44 |
| timing SimpleNonlinearSolve Float32 | 2.61 |
| timing SimpleNonlinearSolve Float64 | 2.73 |
| timing NLsolve Float64 | 2.59 |
| blocks mixed GeometricSolvers Float64 | 2.32 |
| blocks mixed SIAMFANLEquations Float64 | 2.23 |
| blocks stage GeometricSolvers Float32 | 2.24 |
| blocks stage SimpleSolvers Float32 | 2.28 |
| blocks stage SimpleNonlinearSolve Float32 | 2.35 |
| blocks stage NLsolve Float32 | 2.56 |
| blocks anderson GeometricSolvers Float32 | 2.36 |
| blocks anderson GeometricSolvers Float64 | 2.74 |
| blocks sparse GeometricSolvers Float32 | 3.39 |
| blocks sparse GeometricSolvers Float64 | 2.77 |
| blocks sparse NLsolve Float32 | 3.29 |
| blocks sparse NLsolve Float64 | 2.88 |
| blocks sparse SimpleSolvers Float32 | 3.17 |
| blocks sparse SimpleSolvers Float64 | 3.05 |
| blocks sparse SimpleNonlinearSolve Float32 | 2.91 |
| blocks sparse SimpleNonlinearSolve Float64 | 2.97 |

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
| gs-f32 | `Newton+Backtracking+LUFactorization{Float32}` | 67 | 67 | 1.00 | 1.84 |
| gs-f32-refine | `Newton+Backtracking+LUFactorization{Float32}+Refinement` | 68 | 68 | 1.00 | 1.97 |
| nsol-f32 | `nsol(sham=1)+FPS{Float32}` | 62 | 62 | 1.00 | 0.987 |
| nsol-mplu | `nsol(sham=1)+mplu` | 60 | 60 | 1.00 | 3.31 |

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
strict times are those of the timed sweep, from other processes.

| package | successes at eps(Float16) | successes at √eps(Float32) | both | iterations | f_evals | median_s |
|:--|--:|--:|--:|--:|--:|--:|
| GeometricSolvers | 143 | 143 | 143 | 1.00 | 1.00 | 0.993 |
| SimpleSolvers | 98 | 98 | 98 | 1.00 | 1.00 | 0.979 |
| SimpleNonlinearSolve | 41 | 41 | 41 | — | 1.00 | 0.988 |

### Y2 · Anderson

The 79 rows in both precisions, GeometricSolvers only, 1264 cells in `anderson.csv`. For `k = 1,
2, 3, 5`, `anderson-k` is `Anderson(Newton(; reuse = k, globalization = Backtracking(), jacobian
= nothing))` with the depth 4, and `newton-k` is the same `Newton` alone; each solves a problem
with the shared Jacobian, with `tol = √eps(T)` and 200 iterations. The successes are of 79; the
medians of the ratios `anderson-k` / `newton-k` are over the rows where both succeed, with a count
of 0 taken as 1.

| eltype | k | anderson-k successes | newton-k successes | both | iterations | f_evals | median_s |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Float32 | 1 | 58 | 68 | 57 | 1.00 | 1.50 | 1.36 |
| Float32 | 2 | 61 | 68 | 61 | 1.20 | 1.72 | 1.79 |
| Float32 | 3 | 66 | 67 | 65 | 1.08 | 1.56 | 1.71 |
| Float32 | 5 | 65 | 67 | 63 | 0.905 | 1.25 | 1.33 |
| Float64 | 1 | 63 | 69 | 62 | 1.11 | 1.84 | 1.61 |
| Float64 | 2 | 63 | 71 | 63 | 1.22 | 1.93 | 1.74 |
| Float64 | 3 | 66 | 70 | 66 | 1.12 | 1.66 | 1.76 |
| Float64 | 5 | 67 | 69 | 66 | 0.963 | 1.27 | 1.34 |

### Y2 · The sparse problem

`sparse_bracket_problem(N, T)` of `test/helpers/sparse_problems.jl`, with its seed 1, for
`N = 125` and `768` (dimension `2N`) and `T = Float32` and `Float64`, from its start `z0` with its
parameter `p_val`, at `tol = √eps(T)` and 200 iterations; 20 cells in `sparse.csv`. The variants
differ in the Jacobian source and the linear solver:

- `gs-sparse`: GeometricSolvers' `Newton(; jacobian = AutoSparse(AutoForwardDiff();
  coloring_algorithm = GreedyColoringAlgorithm()))` on a problem with the sparsity prototype, which
  factors with Sparspak's LU through the package extension;
- `gs-dense`: GeometricSolvers' `Newton()` on a problem with the shared dense ForwardDiff
  Jacobian;
- `nlsolve-sparse`: NLsolve's `:newton` on a `OnceDifferentiable` with the problem's analytic
  sparse Jacobian and a copy of the prototype, in `Float64` only;
- `simplesolvers` and `simplenonlinearsolve`: `no sparse path`, because the benchmark has no
  sparse adapter for these two packages.

`median_s` is in seconds.

| variant | N | eltype | status | code | iterations | f_evals | median_s |
|:--|--:|:--|:--|:--|--:|--:|--:|
| gs-sparse | 125 | Float32 | ran | SUCCESS | 2 | 3 | 0.000274 |
| gs-dense | 125 | Float32 | ran | SUCCESS | 2 | 3 | 0.000774 |
| nlsolve-sparse | 125 | Float32 | no Float32 |  |  |  |  |
| simplesolvers | 125 | Float32 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 125 | Float32 | no sparse path |  |  |  |  |
| gs-sparse | 125 | Float64 | ran | SUCCESS | 3 | 4 | 0.000420 |
| gs-dense | 125 | Float64 | ran | SUCCESS | 3 | 4 | 0.00174 |
| nlsolve-sparse | 125 | Float64 | ran | converged | 3 | 4 | 0.000441 |
| simplesolvers | 125 | Float64 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 125 | Float64 | no sparse path |  |  |  |  |
| gs-sparse | 768 | Float32 | ran | SUCCESS | 2 | 3 | 0.00164 |
| gs-dense | 768 | Float32 | ran | SUCCESS | 2 | 3 | 0.0638 |
| nlsolve-sparse | 768 | Float32 | no Float32 |  |  |  |  |
| simplesolvers | 768 | Float32 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 768 | Float32 | no sparse path |  |  |  |  |
| gs-sparse | 768 | Float64 | ran | SUCCESS | 3 | 4 | 0.00246 |
| gs-dense | 768 | Float64 | ran | SUCCESS | 3 | 4 | 0.175 |
| nlsolve-sparse | 768 | Float64 | ran | converged | 3 | 4 | 0.00241 |
| simplesolvers | 768 | Float64 | no sparse path |  |  |  |  |
| simplenonlinearsolve | 768 | Float64 | no sparse path |  |  |  |  |
