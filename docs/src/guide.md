```@meta
CurrentModule = GeometricSolvers
```

# [Methods by regime](@id guide)

This page lists, for each regime, the methods that run in it and the conditions under which each
one applies. It names no method as the right one: the choice depends on the problem, its size, the
cost of its residual and its Jacobian, and the start. The pages under *Algorithms* describe each
method in full.

The storage of the iterate selects the regime (see [The three regimes](@ref regimes)). R5 is an R3
iterate with a matrix-free linear solver:

| regime | iterate | entry point | linear solver |
|:--|:--|:--|:--|
| **R1** value semantics | `Number` or `SVector` | [`solve`](@ref) | a factorisation of a static matrix |
| **R2** batched | an array of `Number` or `SVector`, one entry per system | [`solve_batch!`](@ref) | a factorisation of a static matrix, per system |
| **R3** in place | an `AbstractVector` that is not static | [`solve`](@ref), [`init`](@ref), [`solve!`](@ref) | a factorisation of a dense or sparse matrix |
| **R5** matrix-free | an `AbstractVector` that is not static | [`solve`](@ref), [`init`](@ref), [`solve!`](@ref) | [`GMRES`](@ref) or [`MINRESQLP`](@ref), with no Jacobian matrix |

`Newton()` takes `linsolve = LUFactorization()` and `globalization = Static()` unless the caller
gives others (see [Nonlinear methods](@ref nonlinear)).

## The method of a call that names none

[`init`](@ref)`(prob, x₀)` and [`solve`](@ref)`(prob, x₀)` on a [`NonlinearProblem`](@ref), a
[`StationarityProblem`](@ref) or a [`LeastSquaresProblem`](@ref) take the method
[`default_method`](@ref)`(prob, x₀)` and the parameters `p = nothing`. The method follows from the problem type, its `jacobian_prototype` and
the storage of the iterate:

| problem | iterate | method |
|:--|:--|:--|
| [`NonlinearProblem`](@ref) with a `SparseMatrixCSC` `jacobian_prototype` | a `Vector{Float64}` or a `Vector{ComplexF64}` | `Newton(; linsolve = UMFPACKFactorization())` |
| [`NonlinearProblem`](@ref), every other prototype and iterate | any other | `Newton()` |
| [`StationarityProblem`](@ref) | any | `Newton()` |
| [`LeastSquaresProblem`](@ref) | any | [`GaussNewton`](@ref)`()` |

The rules read no size. The `LUFactorization()` of `Newton()` selects its factor from the storage
of the Jacobian: a static factor in R1, and on a dense array the factor that
[What runs where](@ref) lists for its array type. A `SparseMatrixCSC` prototype on any other
iterate takes `Newton()` as well: a `Vector{Float32}` gets the Sparspak factor where Sparspak is
loaded, and `init` raises otherwise, as it does when the caller names `Newton()`. A
`CuSparseMatrixCSR` prototype takes `Newton()`, which has the cuDSS factor where `CUDA` and
`CUDSS` are loaded (see [Sparse direct solves](@ref sparse)). The default method is never
matrix-free and never a combinator: the problem carries no information that selects either.

## R1: one small system as a value

**Methods.**

| method | condition |
|:--|:--|
| [`Newton`](@ref) with [`LUFactorization`](@ref) | a square Jacobian |
| [`Newton`](@ref) with [`LDLTFactorization`](@ref) | a symmetric or Hermitian Jacobian; a real Jacobian is read from the lower triangle only, a complex one must be given in full, because both triangles decide between ``L D L^\mathsf{T}`` and ``L D L^\mathsf{H}`` |
| [`Newton`](@ref) with [`CholeskyFactorization`](@ref) | a Hermitian (real symmetric) positive definite Jacobian at every iterate, read from the lower triangle; a Jacobian that is not positive definite gives `SINGULAR`; in R3 a host `Array` only, a device array raises an `ArgumentError` |
| [`Newton`](@ref) with [`QRFactorization`](@ref) | a square Jacobian, or a [`LeastSquaresProblem`](@ref) |
| [`Newton`](@ref) with [`SVDFactorization`](@ref) | a rank-deficient or nearly singular Jacobian, or a [`LeastSquaresProblem`](@ref); the status reports the numerical rank |
| [`Newton`](@ref) with `regularization = μ` | a singular or nearly singular Jacobian, where the step of ``J + \mu I`` is acceptable |
| [`QuasiNewton`](@ref)`(k)`, or [`Newton`](@ref) with `reuse = k` | the Jacobian and its factorisation dominate the cost of a step; the rate is then linear |
| [`Refinement`](@ref), [`GMRESRefinement`](@ref) | a factorisation precision below the working type, and not a [`LeastSquaresProblem`](@ref); in R1 a `GMRESRefinement` runs the LU-IR passes alone |
| [`GaussNewton`](@ref) | a [`LeastSquaresProblem`](@ref) |
| [`Picard`](@ref) | ``I - \omega J`` is a contraction; it needs no Jacobian, and the rate is linear |
| [`Anderson`](@ref), [`Broyden`](@ref) | a fixed-point iteration whose map is cheap and whose Jacobian is expensive or unavailable; in R1 the depth is `1 ≤ m ≤ 4` |
| [`Additive`](@ref), [`Multiplicative`](@ref), [`Repeat`](@ref), [`Switch`](@ref) | members that run in R1, and a problem that is not a [`LeastSquaresProblem`](@ref) |
| [`Fallback`](@ref) | members that run in R1, and a problem that is not a [`LeastSquaresProblem`](@ref); each member solves from ``x_0`` in turn until one returns `SUCCESS` |

**Globalizations.** [`Static`](@ref) applies close to the solution. [`Backtracking`](@ref),
[`Bisection`](@ref) and [`StrongWolfe`](@ref) search along the direction from a distant start.
[`DogLeg`](@ref) and [`LevenbergMarquardt`](@ref) change the direction and continue where the
Jacobian is singular or ill-conditioned; they take no factor reuse (see
[Globalization](@ref globalization)).

**Conditions of the regime.** The static factorisations unroll over the size of the system, so
the compile time and the register use grow with it. A [`GMRES`](@ref) solve and
[`LeftPreconditioned`](@ref) need an array iterate and raise an `ArgumentError` here. A
factorisation precision below the working type saves registers and memory, not operations.
[`solve`](@ref) differentiates through the solve when `x₀` or `p` carries a `ForwardDiff.Dual`
(see [Implicit differentiation](@ref)); a [`LeastSquaresProblem`](@ref) raises an
`ArgumentError` there.

```jldoctest guide_r1
julia> using GeometricSolvers, StaticArrays

julia> F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2]);

julia> prob = NonlinearProblem(F);

julia> candidates = (Newton(), Newton(; linsolve = SVDFactorization()), QuasiNewton(3),
                     Newton(; globalization = DogLeg()));

julia> all(candidates) do m
           x, status = solve(prob, SVector(1.0, 0.5), m, 2.0)
           status.code === SUCCESS && x ≈ SVector(1.0, 1.0)
       end
true
```

## R2: many small systems in one launch

**Methods.** [`solve_batch!`](@ref) runs the R1 loop in one work-item per system, so a batch takes
each method of R1 above, with the same conditions. A [`LeastSquaresProblem`](@ref) has no batch
form. [`PerSystem`](@ref) and [`BatchParams`](@ref) give each system its own parameters, and
[`batch_summary`](@ref) reduces the status vector.

**Conditions of the regime.** The systems are independent and of the same form. On a `Vector` the
kernel runs on the KernelAbstractions `CPU()` backend. On a GPU backend the problem enters the
kernel by value, so each part of it is `isbits`: a residual that captures a host array, or a
sparse `jacobian_prototype`, cannot enter a device kernel. On Metal the element type is `Float32`.
This page runs no device example.

```jldoctest guide_r2
julia> using GeometricSolvers, StaticArrays

julia> F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2]);

julia> prob = NonlinearProblem(F);

julia> X = [SVector(1.0, 0.5), SVector(2.0, 1.0), SVector(0.5, 0.1)];

julia> S = Vector{SolverStatus{Float64}}(undef, length(X));

julia> solve_batch!(X, S, prob, Newton(; globalization = Backtracking()), PerSystem([2.0, 8.0, 0.5]));

julia> batch_summary(S).all_converged && X ≈ [SVector(1.0, 1.0), SVector(2.0, 2.0), SVector(0.5, 0.5)]
true
```

## R3: one system in place

**Methods.** Each method of R1 runs in R3, with no limit on the depth of [`Anderson`](@ref) and
[`Broyden`](@ref), and a [`GMRESRefinement`](@ref) runs its GMRES-IR stage. R3 adds:

| method | condition |
|:--|:--|
| [`LUFactorization`](@ref) on a sparse `jacobian_prototype` | a host `Vector` iterate of a real element type, Sparspak loaded, the element type of the iterate as the factorisation precision, and a Jacobian from an analytic `J!` that writes the nonzeros of the prototype or from `jacobian = AutoSparse(...)` (see [Sparse direct solves](@ref sparse)) |
| [`KroneckerFactorization`](@ref) | the stage system of an implicit Runge–Kutta method, from a [`NonlinearProblem`](@ref) with `stage_matrix` and `stage_jacobian`, whose stage matrix is diagonalisable with `cond(V) ≤ max(eps(real(T)), eps(Float64))^(-1/2)` and invertible; a host array iterate, [`Static`](@ref) or a line search, and the step `h` in `p.h` |
| [`RAS`](@ref) | local nonlinearities that a [`BlockPartition`](@ref) isolates, and one local residual per block |
| [`LeftPreconditioned`](@ref) | an inner step `N` whose fixed points are the wanted roots; each trial point of the outer globalization costs one inner step, and with [`RAS`](@ref) it needs converged local solves |

[`Refinement`](@ref) converges while the error of the solve in the factorisation precision,
roughly the condition number of ``J`` times its unit round-off, stays below one, and
[`GMRESRefinement`](@ref) extends that range of condition numbers; where both fail, the step
refactorises in the working type. [Mixed precision](@ref mixed) gives the conditions in full.

**Conditions of the regime.** [`init`](@ref) allocates each buffer once, so a system that is solved
more than once goes through [`init`](@ref) and [`solve!`](@ref). On a device array the
factorisation methods run on device kernels. On CUDA, [`LUFactorization`](@ref),
[`QRFactorization`](@ref) and [`SVDFactorization`](@ref) call cuSOLVER for BLAS element types,
except an [`SVDFactorization`](@ref) whose `rtol` lies below its default cutoff `√m · eps(R)`,
with `m` the row count and `R` the real type of the factor, which runs on the own kernel; on Metal,
[`LUFactorization`](@ref) calls Metal Performance Shaders for `Float32` (see
[What runs where](@ref)). The residual of a device solve indexes no scalars. On a device array,
[`LevenbergMarquardt`](@ref) runs with [`SVDFactorization`](@ref) and `scaling = :none` only, and
the sparse LU does not run. This page runs no device example.

```jldoctest guide_r3
julia> using GeometricSolvers

julia> F!(y, x, p) = (y .= x .^ 2 .- p; y);

julia> method = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement());

julia> solver = init(NonlinearProblem(F!), ones(4), method, [1.0, 4.0, 9.0, 16.0]);

julia> x, status = solve!(solver);

julia> status.code === SUCCESS && x ≈ [1.0, 2.0, 3.0, 4.0]
true
```

## R5: one system without a Jacobian matrix

**Methods.** [`Newton`](@ref) with `linsolve = GMRES()`, `BiCGStab()`, `DQGMRES()` or
`MINRESQLP()` holds no ``n \times n`` Jacobian. Each inner iteration of [`GMRES`](@ref),
[`DQGMRES`](@ref) or [`MINRESQLP`](@ref) takes one Jacobian–vector product, and each of
[`BiCGStab`](@ref) two, from automatic differentiation or from a [`JVPFunction`](@ref).

| method or option | condition |
|:--|:--|
| [`GMRES`](@ref) | an `AbstractVector` iterate, on the host or on a device, and the square system of a [`NonlinearProblem`](@ref) or of a [`StationarityProblem`](@ref), with any index sets `E` and `U` |
| [`MINRESQLP`](@ref) | an `AbstractVector` iterate of a real element type and a [`StationarityProblem`](@ref) with `E == U`, the same indices in the same order, whose Jacobian is symmetric and may be indefinite or singular; no preconditioner; under [`DogLeg`](@ref) the Cauchy point takes two Hessian–vector products |
| [`BiCGStab`](@ref) | as `GMRES`, with storage of eight vectors whatever the iteration count and two products per iteration; no minimal residual; no [`DogLeg`](@ref) |
| [`DQGMRES`](@ref) | as `GMRES`, with storage of `min(memory, n) + 1` basis vectors and `min(memory, n)` directions whatever the iteration count and one product per iteration; a quasi-minimal residual; no [`DogLeg`](@ref) |
| [`EisenstatWalkerChoice2`](@ref) (the default), [`EisenstatWalkerChoice1`](@ref) | an inner tolerance that is loose far from the root and tight near it |
| [`FixedForcing`](@ref)`(η)` | a constant inner tolerance `η` |
| a right preconditioner with methods of [`refresh!`](@ref) and [`apply_preconditioner!`](@ref) | an operator that is refreshed once per Newton step and applied before every product |
| [`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref), [`StrongWolfe`](@ref) | as in R1 |
| [`DogLeg`](@ref) | as in R1; it runs on the Arnoldi relation of one unrestarted cycle without a preconditioner |
| [`Steihaug`](@ref) | `linsolve = MINRESQLP()`; truncated conjugate gradients that minimise the model of the action `S`, with trials accepted by the ratio of `DogLeg` on ``\|F\|``; a solve does not converge to a root where the Hessian of `S` is indefinite, also from a start close to it, and ends with `LINESEARCH_FAILED`; `DogLeg` does not have this limit |

**Conditions of the regime.** R5 applies to a large system whose Jacobian no array can hold. A
`GMRES`, `BiCGStab` or `DQGMRES` solve takes no factor reuse, no [`Refinement`](@ref), no
[`LevenbergMarquardt`](@ref) and no [`LeastSquaresProblem`](@ref), and it has no implicit
differentiation; each of them raises an `ArgumentError`. A `Newton` with one of them is not the
outer method of a [`LeftPreconditioned`](@ref), and raises an `ArgumentError` there; it can be the
inner method.
On a device array no factor is formed. [`f_calls`](@ref) and [`jvp_calls`](@ref) count the
evaluations of the last solve.

```jldoctest guide_r5
julia> using GeometricSolvers

julia> F!(y, x, p) = (y .= x .^ 2 .- p; y);

julia> method = Newton(; linsolve = GMRES(; forcing = FixedForcing(1e-6)));

julia> solver = init(NonlinearProblem(F!), ones(4), method, [1.0, 4.0, 9.0, 16.0]);

julia> x, status = solve!(solver);

julia> status.code === SUCCESS && x ≈ [1.0, 2.0, 3.0, 4.0] && jvp_calls(solver) > 0
true
```
