```@meta
CurrentModule = GeometricSolvers
```

# Combinators and preconditioning

## Combinators

[`Additive`](@ref), [`Multiplicative`](@ref), [`Repeat`](@ref), [`Switch`](@ref),
[`Anderson`](@ref) and [`Broyden`](@ref) compose nonlinear methods through the same
[`init_state`](@ref), [`start!!`](@ref), [`step!!`](@ref) and [`finish!!`](@ref) interface as
[`Newton`](@ref) and [`Picard`](@ref), so a combinator runs in R1, R2 and R3 alike.
[`Additive`](@ref) combines member steps from the same iterate with [`OptimalWeights`](@ref) or
[`FixedWeights`](@ref), [`Multiplicative`](@ref) and [`Repeat`](@ref) thread `(x, F(x))`
sequentially across sub-steps while [`sync_merit`](@ref) updates any active line-search baseline
merit, [`Switch`](@ref) transitions permanently from `first` to `second` once
``\|F(x)\| \le \text{threshold} \cdot \|F(x_0)\|``, [`Anderson`](@ref) accelerates its member with
filtered Type-II Anderson mixing, Tikhonov regularisation, restarts and best-iterate restoration,
and [`Broyden`](@ref) accelerates its member over the same shared [`AndersonHistory`](@ref),
filters and safeguards by solving the lower-triangular Type-II Broyden multisecant equations by
forward substitution. A combinator has no implicit differentiation: [`solve`](@ref) with a
combinator method raises an `ArgumentError` for a `ForwardDiff.Dual` in `x₀` or `p`.

[`Fallback`](@ref) composes whole solves instead of steps: it runs [`solve!!`](@ref) with each
member in turn, each from `x₀`, and stops at the first member that returns `SUCCESS`;
[`fallback_member`](@ref) reads the index of the member whose result a solve returned.

[`Continuation`](@ref) is a whole solve too: it follows the path of a homotopy from `(x₀, λ = 0)`
to `λ = 1` by pseudo-arclength continuation, which passes a fold of `λ`, and solves `F(x) = 0`
with its [`Newton`](@ref) member at `λ = 1`. The homotopy is the [`NewtonHomotopy`](@ref), the
[`FixedPointHomotopy`](@ref), or a [`ParameterHomotopy`](@ref) in a parameter of the user's
problem; [`continuation_steps`](@ref) reads `λ` and the length of every accepted and rejected
step of the path.

```@docs
Additive
OptimalWeights
FixedWeights
Multiplicative
Repeat
Switch
Fallback
fallback_member
Continuation
NewtonHomotopy
FixedPointHomotopy
ParameterHomotopy
continuation_steps
homotopy_residual!
homotopy_jacobian!
homotopy_scale
Anderson
Broyden
AndersonHistory
```

## Nonlinear preconditioning

[`LeftPreconditioned`](@ref)`(outer, inner)` runs `outer` on the residual
``r^\ell(y) = y - N(y)`` of one step of `inner` instead of on `F`, so every trial point of its
globalization costs one inner step and every merit value it reads is ``\|r^\ell\|_2^2``. With
the inner method a one-level [`RAS`](@ref) map over a [`BlockPartition`](@ref) it is RASPEN,
`LeftPreconditioned(Newton(), RAS(partition))`, whose Jacobian is the exact form from the block
factors of the last sweep; RASPEN needs converged local solves.
`Multiplicative(RAS(partition), Newton())` is NKS-RAS, the same map
followed by one Newton step. A [`RAS`](@ref) sweep restricts each subdomain, solves each block
with the package's own factor, and prolongates through the owner sets only, so each index of `1:n`
is written once and the sweep calls the global residual no times; [`ras_counts`](@ref) reports the
counters of its last solve. A problem with block residuals supplies the block functions, and a
block without a Jacobian is differentiated by `AutoForwardDiff()`.

```@docs
LeftPreconditioned
LeftPreconditionedProblem
LeftPreconditionedState
LeftPreconditionedAD
RASPenPreparation
RAS
RASState
BlockPartition
ras_sweep!!
ras_prolongate!
ras_counts
raspen_jvp!
raspen_jacobian!
```
