```@meta
CurrentModule = GeometricSolvers
```

# Line searches

A line search is scalar code: [`linesearch`](@ref) sees the line only through the merit
[`φ`](@ref) and its derivative [`φ′`](@ref), allocates nothing and never throws, so the same
search runs on the host and inside a kernel. Its iteration cap, where it has one, is its own
field. Along a [`LineFunction`](@ref) its [`LineSearchResult`](@ref) also carries the residual at
the step, so the caller does not evaluate it again.

```@docs
Static
Backtracking
Bisection
StrongWolfe
```

## Trust region and damped steps

[`DogLeg`](@ref) and [`LevenbergMarquardt`](@ref) are the trust-region and damped-step
globalizations of [`Newton`](@ref). Neither is a line search: each needs the Jacobian, its
factor and a parameter that lives across iterations (the trust-region radius or the damping
parameter `μ`), so `Newton` keeps that parameter in its state and runs the trial loop against
the factor itself.

[`Steihaug`](@ref) is the trust region of a [`Newton`](@ref) with [`MINRESQLP`](@ref) on a
[`StationarityProblem`](@ref) with `E == U`: truncated conjugate gradients on the quadratic model
of the action, with the acceptance and radius rule of [`DogLeg`](@ref) on the merit ``\|F\|``.

[`PseudoTransient`](@ref) is the pseudo-transient continuation of a [`Newton`](@ref) on a
[`StationarityProblem`](@ref): each step solves ``(J + I/δ) d = -F`` and is accepted, and the
pseudo-time step `δ`, which `Newton` keeps in its state, grows as the residual falls. A small
`δ` follows the gradient flow of the action, so the steps leave a saddle that a Newton step
converges to.

```@docs
DogLeg
LevenbergMarquardt
Steihaug
PseudoTransient
```
