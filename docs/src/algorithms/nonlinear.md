```@meta
CurrentModule = GeometricSolvers
```

# [Nonlinear methods](@id nonlinear)

A nonlinear method holds its configuration and nothing about the problem or the iterate. It
enters the loop of [`solve!!`](@ref) through [`init_state`](@ref), [`start!!`](@ref),
[`step!!`](@ref) and [`finish!!`](@ref), so one method object serves each regime that its state
supports, and the [combinators](@ref composition) can compose any of them.

## `Newton`

[`Newton`](@ref) solves ``J(x) d = -F(x)`` at each step and updates the iterate along `d` with
its globalization. Its keywords are:

| keyword | default | meaning |
|:--|:--|:--|
| `linsolve` | `LUFactorization()` | the [linear solver](@ref linear): a factorisation, [`GMRES`](@ref), [`BiCGStab`](@ref), [`DQGMRES`](@ref) or [`MINRESQLP`](@ref) |
| `globalization` | `Static()` | the [globalization](@ref globalization): a line search, [`DogLeg`](@ref), [`Steihaug`](@ref) or [`LevenbergMarquardt`](@ref) |
| `jacobian` | `nothing` | the Jacobian source: `nothing` (the problem's `J`, else `AutoForwardDiff()`), an ADTypes back end, a callable (`J(x, p)` on a value iterate, `J!(J, x, p)` on an array) except with a matrix-free linear method, or a [`JVPFunction`](@ref) for a matrix-free linear method ([`matrix_free`](@ref)) |
| `refinement` | `NoRefinement()` | iterative refinement of the direction: [`Refinement`](@ref) or [`GMRESRefinement`](@ref), see [Mixed precision](@ref mixed) |
| `regularization` | `0` | the diagonal shift ``\mu`` of ``J + \mu I`` |
| `reuse` | `nothing` | the factor-reuse period `k`, or the refresh period of a preconditioned `GMRES` |
| `forcing`, `preconditioner` | `nothing` | the forcing term and the right preconditioner of a `GMRES`, `BiCGStab` or `DQGMRES`; the forcing term of a `MINRESQLP` |
| `best` | `false` | return the iterate of the smallest ``\|F\|_2`` of a solve that ends without `SUCCESS` |
| `keep_factor` | `false` | keep the factor of `reuse = k` across a sequence of solves on one solver |

```jldoctest nonlinear
julia> using GeometricSolvers

julia> F!(y, x, p) = (y[1] = x[1]^2 + x[2]^2 - p; y[2] = x[1] - x[2]; y);

julia> prob = NonlinearProblem(F!);

julia> method = Newton(; linsolve = QRFactorization(), globalization = Backtracking());

julia> x, status = solve(prob, [1.0, 0.5], method, 2.0);

julia> x ≈ [1.0, 1.0]
true
```

### Factor reuse

With `reuse = k`, a `Newton` step factorises the Jacobian and reuses the factor for up to `k`
consecutive steps. It refactorises early after a `STALLED` step, after a step along a reused factor
of rate ``θ_k = \|Δ_k\| / \|Δ_{k-1}\| ≥ 1``, and after a failed refinement on a reused factor.
[`QuasiNewton`](@ref)`(k)` is `Newton(; reuse = k)`. A reused factor saves the
evaluation and the factorisation of the Jacobian at the price of a linear rate, so it pays where
those dominate the cost of a step.

```jldoctest nonlinear
julia> x, status = solve(prob, [1.0, 0.5], QuasiNewton(3), 2.0);

julia> status.code === SUCCESS && x ≈ [1.0, 1.0]
true
```

[`DogLeg`](@ref) and [`LevenbergMarquardt`](@ref) take no reuse, because their model needs the
Jacobian at the current iterate.

With `keep_factor = true` as well, the factor and its age survive from one solve of a
[`NonlinearSolver`](@ref) to the next, as in the steps of an integrator: a solve that ends with
`SUCCESS` in one step, or with `SUCCESS` after more steps with a last rate ``θ ≤ 10^{-3}``, and
that did not promote its factorisation to the working precision, hands its factor on. Every other
solve leaves the next one to factorise at its first step: a solve that does not end with
`SUCCESS`, a promoted solve, and a solve that ends with `SUCCESS` at its start, with no step. A
caller that changes the Jacobian
between two solves, such as with a new step size, passes
`solve!(solver, x, p; refactorize = true)`.

### Regularisation

`regularization = μ` factorises ``J + \mu I`` instead of ``J``. With an
[`SVDFactorization`](@ref) a singular Jacobian gives the truncated minimum-norm step instead, and
[`SolverStatus`](@ref) reports the numerical rank in `rank`.

### The best iterate

A full Newton step can raise ``\|F\|``, so a solve that ends without `SUCCESS` can end on an
iterate worse than one it passed. With `best = true`, `Newton` keeps the iterate of the smallest
``\|F\|_2`` among the start and every iterate a step returned, and returns it where the solve ends
with a code other than `SUCCESS` and that norm is below the norm of the last iterate;
`status.fnorm` then reports that norm, and every other field of the status is that of
`best = false`. A solve that succeeds returns its last iterate. On ``F(x) = x^2 + 1``, which has
no real root, eight steps from `0.5` end on ``\|F\| \approx 2.49``, and the sixth iterate has
``\|F\| \approx 1.03``:

```jldoctest nonlinear
julia> noroot = NonlinearProblem((x, p) -> x^2 + 1);

julia> _, last_status = solve(noroot, 0.5, Newton(); max_iterations = 8);

julia> _, best_status = solve(noroot, 0.5, Newton(; best = true); max_iterations = 8);

julia> last_status.code === best_status.code === MAXITERS
true

julia> round(last_status.fnorm; digits = 2), round(best_status.fnorm; digits = 2)
(2.49, 1.03)
```

[`GaussNewton`](@ref) and [`QuasiNewton`](@ref) take the keyword too. A [`Deflated`](@ref) problem
and the outer method of a [`BlockEliminated`](@ref), a [`VariableProjection`](@ref) or a
[`BatchEliminated`](@ref) problem refuse it, and so does [`LeftPreconditioned`](@ref) for a
`Newton` in its outer method. The residual test can read the max norm instead of the 2-norm, with
`Options(T; norm = MaxNorm())` ([`Options`](@ref)); the ranking of `best` stays in the 2-norm.

## `GaussNewton`

[`GaussNewton`](@ref) is a [`Newton`](@ref) with `linsolve = QRFactorization()` and `Static()`,
for a [`LeastSquaresProblem`](@ref). [`SVDFactorization`](@ref) serves a rank-deficient Jacobian.
For a damped step, use `Newton(; globalization = LevenbergMarquardt())`, which takes any of the
four factorisations. See [Problem types](@ref problems) for an example.

## `Picard`

[`Picard`](@ref) is the damped fixed-point iteration ``x_{k+1} = x_k - \omega F(x_k)``. It needs
no Jacobian, and with ``F(x) = x - G(x)`` it is the fixed-point iteration of `G`. The step
converges where ``I - \omega J`` is a contraction, and its rate is linear.

```jldoctest nonlinear
julia> fp = NonlinearProblem((x, p) -> x - cos(x));

julia> x, status = solve(fp, 1.0, Picard(; damping = 0.8));

julia> status.code === SUCCESS && isapprox(x, 0.7390851332; atol = 1e-8)
true
```

A line search ([`Backtracking`](@ref), [`Bisection`](@ref) or [`StrongWolfe`](@ref)) along the
Picard direction uses a measured slope. [`Anderson`](@ref) and [`Broyden`](@ref) accelerate a
Picard iteration; see [Composition and preconditioning](@ref composition).

## Implicit differentiation

[`solve`](@ref) differentiates through a nonlinear solve. When `x₀` or `p` carries a
`ForwardDiff.Dual`, `solve` strips the duals, runs the primal solve to ``x^*``, and on `SUCCESS`
solves the tangent equation of the implicit function theorem,

```math
J(x^*, p)\, \dot x = -\frac{\partial F}{\partial p}(x^*, p) \, \dot p,
```

from one factorisation of ``J(x^*, p)``. The cost of the derivative is therefore one
factorisation and one back substitution per partial, independent of the number of iterations of
the primal solve.

```jldoctest nonlinear
julia> using ForwardDiff

julia> sq = NonlinearProblem((x, p) -> x^2 - p);

julia> dxdp = ForwardDiff.derivative(p -> solve(sq, 1.0, Newton(), p)[1], 2.0);

julia> dxdp ≈ 1 / (2 * sqrt(2.0))
true
```

A non-`SUCCESS` primal solve returns `NaN` partials. The accuracy of the tangent is that of the
factor element type of `linsolve`. [`init`](@ref) and [`solve!`](@ref) refuse dual inputs. A
[`StationarityProblem`](@ref), a [`LeastSquaresProblem`](@ref), a matrix-free solve
([`matrix_free`](@ref)), the combinators and the preconditioning methods have no implicit
differentiation, and `solve` raises an `ArgumentError` for a dual input there. A
[`Deflated`](@ref) problem differentiates the undeflated problem at the root it found.

## Methods on the other pages

- [Globalization](@ref globalization): [`Static`](@ref), [`Backtracking`](@ref),
  [`Bisection`](@ref), [`StrongWolfe`](@ref), [`DogLeg`](@ref), [`LevenbergMarquardt`](@ref).
- [Linear solvers](@ref linear): the factorisations, [`GMRES`](@ref) and the sparse LU.
- [Composition and preconditioning](@ref composition): [`Additive`](@ref),
  [`Multiplicative`](@ref), [`Repeat`](@ref), [`Switch`](@ref), [`Anderson`](@ref),
  [`Broyden`](@ref), [`RAS`](@ref), [`LeftPreconditioned`](@ref).
- [Problem types](@ref problems): least squares, stationarity, deflation, elimination and
  reparametrization.
