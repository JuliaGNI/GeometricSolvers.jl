```@meta
CurrentModule = GeometricSolvers
```

# [Problem types](@id problems)

Every problem subtypes [`AbstractNonlinearProblem`](@ref). The three base problems hold functions
only. The wrappers take a base problem and change the system that the solver sees.

| problem | solves | methods |
|:--|:--|:--|
| [`NonlinearProblem`](@ref) | ``F(x, p) = 0``, square | every method |
| [`LeastSquaresProblem`](@ref) | ``\min_x \tfrac12 \|F(x, p)\|_2^2``, ``m \ge n`` residuals | [`GaussNewton`](@ref), [`Newton`](@ref) with QR, SVD or [`LevenbergMarquardt`](@ref) |
| [`StationarityProblem`](@ref) | ``(\nabla S(x, p))_E + b = 0`` | [`Newton`](@ref), [`Picard`](@ref), the combinators |
| [`Reparametrized`](@ref) | the inner problem in chart coordinates ``x = g(y, p)`` | as the inner problem |
| [`Deflated`](@ref) | the inner problem with known roots removed | as the inner problem |
| [`BlockEliminated`](@ref) | the outer equations after the inner unknowns are eliminated | [`Newton`](@ref) |
| [`VariableProjection`](@ref) | a separable least-squares problem by variable projection | [`Newton`](@ref), [`GaussNewton`](@ref) |
| [`BatchEliminated`](@ref) | an outer system whose residual solves a batch of inner systems | a matrix-free [`Newton`](@ref) |

## `NonlinearProblem`

`NonlinearProblem(F)` and `NonlinearProblem(F, J)` hold the residual and, optionally, its
Jacobian. The call forms are those of [Getting started](@ref "Call forms"). Without `J`, the
Jacobian comes from the `jacobian` field of [`Newton`](@ref), whose default `nothing` takes
`AutoForwardDiff()`.

Three keywords add structure:

- `jacobian_prototype`: a matrix that gives the storage type and the sparsity pattern of the R3
  Jacobian. A `SparseMatrixCSC` prototype on a host `Vector`, with Sparspak loaded, gives a sparse
  LU factorisation (see [Sparse direct solves](@ref sparse)).
- `block_residuals`: the local residuals `F_i!(f_i, x_i, x, p)` of a
  [`BlockPartition`](@ref), which the restricted additive Schwarz method [`RAS`](@ref) reads (see
  [Composition and preconditioning](@ref composition)).
- `block_jacobians`: optionally, with `block_residuals`, the block Jacobians
  `J_i!(J_i, x_i, x, p)`, one per block, which fill the Jacobian of block `i`.

## `LeastSquaresProblem`

A [`LeastSquaresProblem`](@ref) has a residual of length ``m \ge n``. An R1 problem infers `m`
from the residual; an in-place problem needs `output_length = m`. [`GaussNewton`](@ref) is a
[`Newton`](@ref) with `QRFactorization()`. The solve converges by the residual test, or, at a
nonzero-residual minimum, by the stationarity test
``\|J^\mathsf{H} F\| / (\|J\|_F \|F\|) \le g_{tol}``, which sets `status.stationary`:

```jldoctest problems
julia> using GeometricSolvers, StaticArrays

julia> t = SVector(0.0, 1.0, 2.0, 3.0); y = SVector(1.0, 2.7, 7.4, 20.1);

julia> r(x, p) = x[1] .* exp.(x[2] .* p[1]) .- p[2];

julia> x, status = solve(LeastSquaresProblem(r), SVector(1.0, 0.9), GaussNewton(), (t, y));

julia> status.code === SUCCESS && status.stationary
true

julia> isapprox(x, SVector(1.0, 1.0); atol = 1e-2)
true
```

The test reads the factorisation that a step takes at its start point. Where the test passes
there and fails at the end of the step, the solve returns the start point of the step, unless that
point is `x₀`. A step that leaves a saddle of ``\|F\|^2``, which the Gauss–Newton model reads as a
minimum, so ends the solve at the saddle; [`solve!!`](@ref) states the rule.

[`LevenbergMarquardt`](@ref) is the damped globalization of the same step; see
[Globalization](@ref globalization).

## `StationarityProblem`

A [`StationarityProblem`](@ref) takes a scalar action `S(x, p)` and solves the stationarity
equations ``(\nabla S)_E + b = 0`` for the unknowns `x[U]`. The residual is the gradient and the
Jacobian is a block of the Hessian, both by automatic differentiation. This is the form of a
discrete variational principle, whose stationary points are often saddle points: the merit of a
line search is ``\|F\|_2^2`` and never `S` itself.

```jldoctest problems
julia> S(x, p) = (x[1]^2 + x[2]^2) / 2 + x[1]^4 / 4 - p * x[1] + x[1] * x[2];

julia> x, status = solve(StationarityProblem(S), SVector(0.5, 0.5), Newton(), 1.0);

julia> x ≈ SVector(1.0, -1.0)
true
```

With `E == U` the Jacobian is symmetric, so [`LDLTFactorization`](@ref) applies. The keyword `b`
of [`solve`](@ref) and [`init`](@ref) sets the datum for one solve, and `E` and `U` select a subset
of the equations and of the unknowns.

## `Reparametrized`

[`Reparametrized`](@ref)`(prob, g)` iterates in chart coordinates `y` with ``x = g(y, p)``. The
solve returns ``x^* = g(y^*, p)``, the solution in the original coordinates. The chart Jacobian is
`∂g` when given and forward-mode AD otherwise. A chart keeps the iterate on a constraint set,
here the positive half-line:

```jldoctest problems
julia> inner = NonlinearProblem((x, p) -> x - 2);

julia> x, status = solve(Reparametrized(inner, (y, p) -> exp(y)), 0.0, Newton());

julia> x ≈ 2.0
true
```

For a [`StationarityProblem`](@ref) the residual is pulled back covariantly,
``\tilde F = G^\mathsf{T} F(g(y))``, with ``G = \partial g / \partial y``, and the keyword
`curvature` selects whether the Jacobian includes the curvature term of the chart.

## `Deflated`

[`Deflated`](@ref)`(prob, roots)` multiplies the residual by the deflation factor
``M(x) = \prod_i (\|x - r_i\|^{-p} + \sigma)`` of Farrell, Birkisson and Funke (2015), so the known
roots repel the solve, and a solve from the same start can converge to another root. Deflation
does not guarantee that another root exists or that the solve converges to it, so read the
status. The stopping test reads the undeflated residual.

```jldoctest problems
julia> cubic = NonlinearProblem((x, p) -> x^3 - x);

julia> solve(cubic, 0.6, Newton())[1] ≈ 1
true

julia> abs(solve(Deflated(cubic, (1.0,)), 0.6, Newton())[1]) < 1e-8
true
```

`power` and `shift` set ``p`` and ``\sigma``. A [`Deflated`](@ref) problem with `isbits` roots also
runs in a batch: every system of the batch is deflated at the same roots.

## `BlockEliminated` and `VariableProjection`

[`BlockEliminated`](@ref)`(prob, inner)` splits the unknowns into inner unknowns `y = x[inner]`
and outer unknowns `z`. At each outer iterate it solves the inner equations for ``y^*(z)``, and
the outer [`Newton`](@ref) steps on the remaining equations with the Schur complement
``J_{zz} - J_{zy} J_{yy}^+ J_{yz}``. With `linear_inner = true` the inner solve is one linear solve,
for inner equations affine in `y`.

```jldoctest problems
julia> F(x, p) = SVector(x[1] - x[2]^2, x[1] + x[2] - p);

julia> x, status = solve(BlockEliminated(NonlinearProblem(F), (1,)), SVector(0.0, 0.5), Newton(), 2.0);

julia> x ≈ SVector(1.0, 1.0)
true
```

[`VariableProjection`](@ref) does the same for a [`LeastSquaresProblem`](@ref) whose residual is
linear (or easy) in the inner parameters: it minimises over them for each outer iterate, and the
outer solver steps on the reduced residual with Kaufman's reduced Jacobian
(`jacobian_mode = :kaufman`, Kaufman 1975) or the full Golub–Pereyra Jacobian
(`jacobian_mode = :golub_pereyra`, Golub and Pereyra 1973). `BlockEliminated` on a
`LeastSquaresProblem` forwards to `VariableProjection`.

[`BatchEliminated`](@ref) eliminates a whole batch of independent inner systems inside the
residual of an outer matrix-free [`Newton`](@ref): every evaluation of the outer residual runs
[`solve_batch!`](@ref) on the inner systems and deposits their moments. The section
[Reduced-space elimination with a batch inside the residual](@ref) of the API describes it.

## References

- P. E. Farrell, Á. Birkisson and S. W. Funke, *Deflation Techniques for Finding Distinct
  Solutions of Nonlinear Partial Differential Equations*, SIAM Journal on Scientific Computing
  37(4), A2026–A2045 (2015).
  [doi:10.1137/140984798](https://doi.org/10.1137/140984798)
- G. H. Golub and V. Pereyra, *The Differentiation of Pseudo-Inverses and Nonlinear Least
  Squares Problems Whose Variables Separate*, SIAM Journal on Numerical Analysis 10(2), 413–432
  (1973). [doi:10.1137/0710036](https://doi.org/10.1137/0710036)
- L. Kaufman, *A variable projection method for solving separable nonlinear least squares
  problems*, BIT 15(1), 49–57 (1975).
  [doi:10.1007/BF01932995](https://doi.org/10.1007/BF01932995)
