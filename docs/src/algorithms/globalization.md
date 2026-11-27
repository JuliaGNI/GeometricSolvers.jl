```@meta
CurrentModule = GeometricSolvers
```

# [Globalization](@id globalization)

A globalization decides how far a step goes along the direction of the method. The package has
two kinds:

- **line searches** ([`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref),
  [`StrongWolfe`](@ref)) search along the direction `d` on the merit
  ``\varphi(\alpha) = \|F(x + \alpha d)\|_2^2``. They serve [`Newton`](@ref) and [`Picard`](@ref).
- **trust-region and damped steps** ([`DogLeg`](@ref), [`LevenbergMarquardt`](@ref),
  [`PseudoTransient`](@ref) for a `StationarityProblem`) change the direction itself and need the
  Jacobian and its factor. They serve [`Newton`](@ref) only, which keeps the radius, the damping
  parameter or the pseudo-time step in its state.

A line search is scalar code: [`linesearch`](@ref) sees the line only through ``\varphi`` and its
derivative ``\varphi'``, allocates nothing and never throws, so the same search runs on the host
and inside a kernel. The slope ``\varphi'(0)`` of a Newton direction is known exactly
(``-2\varphi(0)`` for an exact step); for an inexact or a Picard step it is one Jacobian–vector
product. The iteration cap of a search is its own `maxiter` field.

The example of this page is ``F(x) = \arctan(x)`` from ``x_0 = 3``, where the full Newton step
overshoots and the iterates grow:

```jldoctest globalization
julia> using GeometricSolvers

julia> prob = NonlinearProblem((x, p) -> atan(x));

julia> solve(prob, 3.0, Newton())[2].code === SUCCESS
false
```

## `Static`

[`Static`](@ref)`(α = 1.0)` takes the fixed step `α` and evaluates nothing. It is the default and
the common case inside an integrator, where the start is close to the solution.

## `Backtracking`

[`Backtracking`](@ref)`(; c₁ = 1e-4, p = 0.5, maxiter = 100)` is the Armijo line search with
safeguarded quadratic and cubic interpolation. It shrinks the trial step until the sufficient
decrease condition ``\varphi(\alpha) - \varphi(0) \le c_1 \alpha \varphi'(0)`` holds, with a
round-off allowance. For an inexact step it tests the decrease of the residual norm against the
linear residual of the step instead. It needs one residual evaluation per trial.

```jldoctest globalization
julia> x, status = solve(prob, 3.0, Newton(; globalization = Backtracking()));

julia> status.code === SUCCESS && abs(x) < 1e-8
true
```

## `Bisection`

[`Bisection`](@ref)`(; αmax = 65536.0, maxiter = 100)` brackets a sign change of ``\varphi'`` from
``-`` to ``+`` and bisects it, so a bracketed search returns a minimiser of the merit along the
line, never a maximiser. Where ``\varphi'`` stays negative up to `αmax`, it returns the step
`αmax`, and after `maxiter` trials it ends with `LINESEARCH_FAILED`. It costs one ``\varphi'``
per trial: one for each doubling of the trial step while ``\varphi'`` stays negative, then one
for each bisection. While the lower end of the bracket is ``0``, the bisections grow with
``\log_2 \log_2`` of the ratio of the trial step to the step floor; a bracket with a positive
lower end is bisected to a relative width of ``\sqrt{\varepsilon}``. It can take a step longer
than the trial step.

```jldoctest globalization
julia> x, status = solve(prob, 3.0, Newton(; globalization = Bisection()));

julia> status.code === SUCCESS && abs(x) < 1e-8
true
```

## `StrongWolfe`

[`StrongWolfe`](@ref)`(; c₁ = 1e-4, c₂ = 0.9, αmax = 65536.0, maxiter = 100)` finds a step that
satisfies the strong Wolfe conditions

```math
\varphi(\alpha) - \varphi(0) \le c_1 \alpha \varphi'(0), \qquad
|\varphi'(\alpha)| \le c_2 |\varphi'(0)|,
```

in a bracketing phase, which doubles the trial step, and a zoom phase, which interpolates inside
the bracket. A trial that fails the first condition costs one ``\varphi``, and a trial that
passes it one ``\varphi`` and one ``\varphi'``.

```jldoctest globalization
julia> x, status = solve(prob, 3.0, Newton(; globalization = StrongWolfe()));

julia> status.code === SUCCESS && abs(x) < 1e-8
true
```

## `DogLeg`

[`DogLeg`](@ref)`(; radius = nothing)` is the trust-region globalization on the dogleg curve of
Pawlowski, Simonis, Walker and Shadid (2008). The step is the Newton step inside the radius, the
Cauchy step when the Cauchy point lies outside it, and otherwise the point of the radius on the
segment between them. A rejected trial shrinks the radius and chooses again on the same curve,
without a new Jacobian or factorisation. A singular factor gives the Cauchy step, so `DogLeg`
continues where a line search along a Newton direction has no direction.

```jldoctest globalization
julia> x, status = solve(prob, 3.0, Newton(; globalization = DogLeg()));

julia> status.code === SUCCESS && abs(x) < 1e-8
true
```

With a [`GMRES`](@ref) linear solver, `DogLeg` runs on the Arnoldi relation of one unrestarted
cycle without a preconditioner.

## `LevenbergMarquardt`

[`LevenbergMarquardt`](@ref)`(R = Float64; μ₀, μ_min, μ_max, μ_decrease, μ_increase, ρ_accept, scaling)`
computes the trial step from the damped model
``(J^\mathsf{H} J + \mu D^\mathsf{T} D)\, \delta = -J^\mathsf{H} F`` and adapts ``\mu`` from the
gain ratio of the actual and the predicted reduction. A rejected trial raises ``\mu`` and solves
again, without a new Jacobian. With [`SVDFactorization`](@ref) and `scaling = :none` each trial
reuses one SVD of ``J``; with the other factorisations, or with `scaling = :diagonal`, the stacked
path reduces `[J | -F]` once per step and each trial eliminates only its own damping rows by Givens
rotations, so ``J^\mathsf{H} J`` is never formed. It serves a [`NonlinearProblem`](@ref), a
[`StationarityProblem`](@ref) and a [`LeastSquaresProblem`](@ref).

```jldoctest globalization
julia> x, status = solve(prob, 3.0, Newton(; globalization = LevenbergMarquardt()));

julia> status.code === SUCCESS && abs(x) < 1e-8
true
```

`scaling = :diagonal` scales the damping by the column norms of ``J``. The stacked path indexes
scalars and runs on a host array only. `LevenbergMarquardt` takes no refinement and no factor
reuse.

## How to choose

- Close to the solution, as in the stages of an implicit integrator, [`Static`](@ref) costs
  nothing beyond the Newton step.
- From a distant start, a line search or a trust region makes the merit decrease. A line search
  needs only residual evaluations and slopes along one direction. [`DogLeg`](@ref) and
  [`LevenbergMarquardt`](@ref) change the direction and continue where the Jacobian is singular or
  ill-conditioned.
- For a least-squares problem, [`LevenbergMarquardt`](@ref) damps the Gauss–Newton step.
- On a device array, the stacked `LevenbergMarquardt` path is refused with an `ArgumentError`,
  because it indexes scalars: only [`SVDFactorization`](@ref) with `scaling = :none` runs there.

`KNOWN_ISSUES.md` in the repository lists the problems where a globalization is known to fail.

## References

- R. P. Pawlowski, J. P. Simonis, H. F. Walker and J. N. Shadid, *Inexact Newton Dogleg
  Methods*, SIAM Journal on Numerical Analysis 46(4), 2112–2132 (2008).
  [doi:10.1137/050632166](https://doi.org/10.1137/050632166)
