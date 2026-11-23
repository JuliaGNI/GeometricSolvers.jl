```@meta
CurrentModule = GeometricSolvers
```

# [Composition and preconditioning](@id composition)

A combinator is a nonlinear method whose members are nonlinear methods. It enters the loop through
the same [`init_state`](@ref), [`start!!`](@ref), [`step!!`](@ref) and [`finish!!`](@ref) as
[`Newton`](@ref) and [`Picard`](@ref), so a combinator runs in each regime that its members run
in, and combinators nest. The composition follows Brune, Knepley, Smith and Tu (2015). The last two
rows of the table are the preconditioning methods: [`RAS`](@ref) and [`LeftPreconditioned`](@ref)
take a [`NonlinearProblem`](@ref) with a mutable vector iterate, so they run in R3 only.

| combinator | one outer step |
|:--|:--|
| [`Additive`](@ref)`(m₁, m₂, …)` | each member steps from the same iterate; the candidates are combined with weights |
| [`Multiplicative`](@ref)`(m₁, m₂, …)` | the members step in sequence, each from the result of the last |
| [`Repeat`](@ref)`(m, k)` | `k` steps of `m` |
| [`Switch`](@ref)`(m₁, m₂; threshold)` | `m₁` until ``\|F(x)\| \le \text{threshold} \cdot \|F(x_0)\|``, then `m₂` for the rest of the solve |
| [`Anderson`](@ref)`(m, depth)` | one step of `m`, accelerated by Type-II Anderson mixing |
| [`Broyden`](@ref)`(m, depth)` | one step of `m`, accelerated by Type-II Broyden multisecant updates |
| [`RAS`](@ref)`(partition)` | restricted additive Schwarz: local Newton solves on overlapping blocks |
| [`LeftPreconditioned`](@ref)`(outer, inner)` | `outer` on the residual ``y - N(y)`` of one step `N` of `inner` |

[`Fallback`](@ref)`(m₁, m₂, …)` composes whole solves instead of steps. It has no
[`step!!`](@ref): its [`solve!!`](@ref) runs the solve of each member in turn, each from `x₀`,
and stops at the first member that returns `SUCCESS`. So it is not a member of the combinators of
the table.

A combinator takes a [`NonlinearProblem`](@ref) or a [`StationarityProblem`](@ref) and refuses a
[`LeastSquaresProblem`](@ref). A combinator and a preconditioning method have no implicit
differentiation. A combinator other than [`Repeat`](@ref) reports `0` in `f_calls` and
`jvp_calls`, because its members count only their own steps.

The examples of this page use the fixed-point problem ``x = \cos x``:

```jldoctest composition
julia> using GeometricSolvers

julia> fp = NonlinearProblem((x, p) -> x - cos(x));

julia> root = 0.7390851332151607;
```

## `Additive`

[`Additive`](@ref) runs one step of each member from the same iterate ``x`` and forms
``x_+ = x + \sum_i w_i (x_i - x)``. [`OptimalWeights`](@ref) chooses the weights that minimise the
linearised residual ``\|F(x) + \sum_i w_i (F(x_i) - F(x))\|_2``, with a safeguard that falls back
to the best candidate; [`FixedWeights`](@ref) holds them constant. A member whose step fails hard
is left out of the combination.

```jldoctest composition
julia> x, status = solve(fp, 1.0, Additive(Newton(), Picard()));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-8)
true

julia> x, status = solve(fp, 1.0, Additive(Newton(), Picard(); weights = FixedWeights(0.5, 0.5)));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-6)
true
```

## `Multiplicative` and `Repeat`

[`Multiplicative`](@ref) threads `(x, F(x))` through its members in order within one outer step:
`Multiplicative(Picard(), Newton())` is a Picard step followed by a Newton step.
[`Repeat`](@ref)`(m, k)` takes `k` steps of `m` per outer step. Between members, [`sync_merit`](@ref)
passes the current merit to the line search of the next member.

```jldoctest composition
julia> x, status = solve(fp, 1.0, Multiplicative(Picard(), Newton()));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-8)
true

julia> x, status = solve(fp, 1.0, Repeat(Picard(), 3));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-6)
true
```

## `Switch`

[`Switch`](@ref)`(first, second; threshold = 1e-2)` runs `first` until the residual has dropped by
the factor `threshold`, then latches to `second`. The common pattern is a robust method with a
large convergence region first, and Newton for the fast local convergence after it:

```jldoctest composition
julia> x, status = solve(fp, 1.0, Switch(Anderson(), Newton(); threshold = 1e-2));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-10)
true
```

## `Anderson` and `Broyden`

[`Anderson`](@ref)`(member, m)` accelerates the fixed-point map of `member` with Type-II Anderson
mixing of depth `m` (Walker and Ni 2011). The history of differences is filtered by the length and
angle filters of Pollock and Rebholz (2023) and the weight solve is regularised. The history
restarts when the residual grows, and on a failed solve the iterate with the smallest residual is
restored. `Anderson(m)` wraps `Picard()`.

[`Broyden`](@ref) shares that history, its filters and its safeguards, and solves the
lower-triangular Type-II Broyden multisecant system of Fang and Saad (2009) by forward
substitution instead of the full normal equations. `Broyden(m)` wraps `Picard(; damping = 0.5)`.

```jldoctest composition
julia> x, status = solve(fp, 1.0, Anderson(Picard(), 2));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-8)
true

julia> x, status = solve(fp, 1.0, Broyden(2));

julia> status.code === SUCCESS && isapprox(x, root; atol = 1e-8)
true
```

In R1 the depth is `1 ≤ m ≤ 4` and the state is `isbits`; in R3 any depth applies, with
preallocated ``n \times m`` difference matrices.

## Restricted additive Schwarz: `RAS`

[`RAS`](@ref)`(partition)` is the one-level restricted additive Schwarz map of nonlinear
preconditioning. A [`BlockPartition`](@ref) splits `1:n` into overlapping subdomains, each with
an owner set; the owner sets are disjoint and cover `1:n`. One step restricts the iterate to each
subdomain, takes up to `max_iterations` local Newton steps with that block's own factor, and
writes back the owned indices only. The problem supplies one local residual
`F_i!(f_i, x_i, x, p)` per block through the `block_residuals` keyword of
[`NonlinearProblem`](@ref), which reads the couplings outside the block from the frozen full
iterate `x`. A block without its own Jacobian is differentiated with `AutoForwardDiff()`.

```jldoctest composition
julia> n = 8;

julia> cell(u, j) = u[j] * (1 + (j == 1 ? 0.0 : u[j - 1])^2 + (j == n ? 0.0 : u[j + 1])^2) - 1;

julia> F!(r, x, p) = (for j in 1:n; r[j] = cell(x, j); end; r);

julia> part = BlockPartition(n, 2; overlap = 1);

julia> blocks = ntuple(2) do i
           S = part.subdomains[i]
           function (f, x_loc, x, p)
               u = Vector{promote_type(eltype(x_loc), eltype(x))}(x)
               u[S] .= x_loc
               for (k, j) in enumerate(S)
                   f[k] = cell(u, j)
               end
               return f
           end
       end;

julia> prob = NonlinearProblem(F!; block_residuals = blocks);

julia> x_ras, status = solve(prob, zeros(n), RAS(part; max_iterations = 20, rtol = 1e-12));

julia> status.code === SUCCESS
true
```

With `rtol` or `atol` positive, a local solve that does not converge ends the solve with
`MAXITERS`, and [`ras_counts`](@ref) names the block.

## Nonlinear preconditioning: `LeftPreconditioned`

[`LeftPreconditioned`](@ref)`(outer, inner)` runs `outer` on the left-preconditioned residual
``r^\ell(y) = y - N(y)``, where ``N`` is one step of `inner`. The zeros of ``r^\ell`` are the fixed
points of ``N``. Every trial point of the globalization of `outer` costs one step of `inner`.

- `LeftPreconditioned(Newton(), RAS(partition))` is RASPEN (Dolean, Gander, Kheriji, Kwok and
  Masson 2016). Its Jacobian is the exact form from the block factors of the last sweep,
  [`raspen_jvp!`](@ref) and [`raspen_jacobian!`](@ref). RASPEN needs converged local solves.
- `Multiplicative(RAS(partition), Newton())` is the composition of the same map with one Newton
  step, the nonlinear restricted additive Schwarz preconditioning of Cai and Li (2011).
- With an inner method other than `RAS`, the derivative of ``r^\ell`` is a central difference.

```jldoctest composition
julia> raspen = LeftPreconditioned(Newton(), RAS(part; max_iterations = 20, rtol = 1e-12));

julia> x, status = solve(prob, zeros(n), raspen);

julia> status.code === SUCCESS && isapprox(x, x_ras; atol = 1e-7)
true

julia> x, status = solve(prob, zeros(n), Multiplicative(RAS(part; max_iterations = 2), Newton()));

julia> status.code === SUCCESS && isapprox(x, x_ras; atol = 1e-7)
true
```

The status of a preconditioned solve is that of the global residual ``F`` at the returned iterate.

## How to choose

A combinator changes the convergence region and the cost of a step, and the right choice depends
on the problem. Some conditions under which each one applies:

- [`Switch`](@ref) and [`Multiplicative`](@ref) with a cheap first member apply where Newton alone
  does not converge from the start but converges near the solution.
- [`Anderson`](@ref) and [`Broyden`](@ref) apply to a fixed-point iteration whose map is cheap and
  whose Jacobian is expensive or unavailable.
- [`Additive`](@ref) with [`OptimalWeights`](@ref) combines methods whose steps are good in
  different directions.
- [`RAS`](@ref) and [`LeftPreconditioned`](@ref) apply to problems with local nonlinearities that
  a partition into blocks isolates, where a global Newton step is slowed by those blocks.

## References

- P. R. Brune, M. G. Knepley, B. F. Smith and X. Tu, *Composing Scalable Nonlinear Algebraic
  Solvers*, SIAM Review 57(4), 535–565 (2015).
  [doi:10.1137/130936725](https://doi.org/10.1137/130936725)
- X.-C. Cai and X. Li, *Inexact Newton Methods with Restricted Additive Schwarz Based Nonlinear
  Elimination for Problems with High Local Nonlinearity*, SIAM Journal on Scientific Computing
  33(2), 746–762 (2011). [doi:10.1137/080736272](https://doi.org/10.1137/080736272)
- V. Dolean, M. J. Gander, W. Kheriji, F. Kwok and R. Masson, *Nonlinear Preconditioning: How
  to Use a Nonlinear Schwarz Method to Precondition Newton's Method*, SIAM Journal on Scientific
  Computing 38(6), A3357–A3380 (2016).
  [doi:10.1137/15M102887X](https://doi.org/10.1137/15M102887X)
- H.-r. Fang and Y. Saad, *Two classes of multisecant methods for nonlinear acceleration*,
  Numerical Linear Algebra with Applications 16(3), 197–221 (2009).
  [doi:10.1002/nla.617](https://doi.org/10.1002/nla.617)
- S. Pollock and L. G. Rebholz, *Filtering for Anderson Acceleration*, SIAM Journal on Scientific
  Computing 45(4), A1571–A1590 (2023).
  [doi:10.1137/22M1536741](https://doi.org/10.1137/22M1536741)
- H. F. Walker and P. Ni, *Anderson Acceleration for Fixed-Point Iterations*, SIAM Journal on
  Numerical Analysis 49(4), 1715–1735 (2011).
  [doi:10.1137/10078356X](https://doi.org/10.1137/10078356X)
