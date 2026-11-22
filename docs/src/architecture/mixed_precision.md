```@meta
CurrentModule = GeometricSolvers
```

# [Mixed precision](@id mixed)

A solve has up to three precisions:

| precision | holds | set by |
|:--|:--|:--|
| working precision `T` | the iterate, the residual, the Jacobian buffer, the stopping test and the status | the element type of `x₀` |
| factorisation precision `TF` | the factorisation and the triangular solves | the type parameter of the linear method |
| refinement residual | ``-F(x) - J d`` in a refinement pass | `T` |

The working precision sets the attainable accuracy of Newton's method, because the residual and
the update are computed in `T`. The factorisation precision changes the rate of convergence, not
the limit (Kelley 2022; Higham and Mary 2022).

## The factorisation precision is a type parameter

`LUFactorization(Float32)` is `LUFactorization{Float32}()`, and the same holds for
[`LDLTFactorization`](@ref), [`QRFactorization`](@ref) and [`SVDFactorization`](@ref). The
default `LUFactorization()` is `LUFactorization{Nothing}()`, which factorises in `T`. For a complex
problem the precision stays real: `LUFactorization(Float32)` factorises a `ComplexF64` Jacobian in
`ComplexF32`. [`factor_eltype`](@ref) gives the element type of the factors:

```jldoctest mixed
julia> using GeometricSolvers

julia> GeometricSolvers.factor_eltype(LUFactorization(Float32), ComplexF64) === ComplexF32
true

julia> GeometricSolvers.factor_eltype(LUFactorization(), Float64)
Float64
```

The precision is a type parameter and not a field, so the method stays `isbits` and enters a
kernel, and the factor type that [`init_linear`](@ref) builds carries both the precision and the
backend, which selects any vendor method by dispatch.

## The scaled mixed-precision solve

The boundary between the two precisions is one method of [`ldiv!!`](@ref), the one for a factor
whose element type differs from that of the right-hand side (`src/linear/mixed.jl`):

1. [`factorize!!`](@ref) divides the matrix by a power of two ``2^e`` near its largest entry,
   converts it to `TF`, and factorises. A power of two is exact in every binary format, so the
   scaling adds no rounding error, and the largest entries of a `Float64` matrix stay in the
   `Float32` range. One factor of scale does not cover an arbitrary dynamic range: an entry far
   below the largest one underflows in `TF`, and the factor can then be singular, as for
   ``\operatorname{diag}(2^{128}, 2^{-128})``. An LU, LDLᵀ or QR factor records the zero pivot,
   and the [remedy chain](@ref "The remedy chain") then refactorises in `T`. An SVD factor drops
   the direction and records no failure, so the chain refactorises in `T` only after a failed
   refinement or two slow steps, and the solve can end on `STALLED` before that.
2. `ldiv!!` scales the right-hand side by ``s = \|b\|_\infty``, converts it to `TF`, and solves.
   The triangular solve in `TF` can overflow where the solution in `T` is finite: for
   ``\operatorname{diag}(1, 2^{-128})`` and ``b = (0, 1)``, the step ``2^{128}`` exceeds
   `floatmax(Float32)`.
3. The solution is multiplied by ``s \cdot 2^{-e}`` in the wider of the two precisions and only
   then converted to `T`, so this rescaling adds no overflow or underflow beyond that of the
   solution itself.

The bound that the test suite verifies for this solve is on the page
[Numerical Stability and Backward-Error Bounds](@ref stability).

```jldoctest mixed
julia> F!(y, x, p) = (y[1] = x[1]^2 + x[2]^2 - p; y[2] = x[1] - x[2]; y);

julia> prob = NonlinearProblem(F!);

julia> solver = init(prob, [1.0, 0.5], Newton(; linsolve = LUFactorization(Float32)), 2.0);

julia> x, status = solve!(solver);

julia> status.code === SUCCESS && x ≈ [1.0, 1.0]
true

julia> eltype(stored_factors(solver))
Float32
```

[`stored_factors`](@ref) returns the packed `L` and `U` of the last factorisation in `TF`.

## Refinement: `NoRefinement`, `Refinement`, `GMRESRefinement`

The `refinement` field of [`Newton`](@ref) corrects the linear direction in `T` before the
globalization:

- [`NoRefinement`](@ref) (the default) takes the direction of the solve in `TF` as it is. The step
  is then an inexact Newton step, and the rate slows towards linear as the error of the solve
  grows.
- [`Refinement`](@ref)`(T; η, maxiter)` is the LU-IR loop: it forms the residual
  ``r = -F(x) - J d`` in `T`, solves for a correction with the factor in `TF`, and stops when
  ``\|F(x) + J d\| \le \max(\eta, \sqrt{\varepsilon(T)})\,\|F(x)\|``, or with a failure when a pass
  does not decrease that norm. The theory of iterative refinement in three precisions is that of
  Carson and Higham (2018).
- [`GMRESRefinement`](@ref)`(T; η, maxiter, gmres_maxiter, gmres_tol)` runs the same passes first
  and, where they fail, a GMRES in `T` left-preconditioned by the factors promoted to `T`: the
  GMRES-IR variant SDD of Amestoy, Buttari, Higham, L'Excellent, Mary and Vieublé (2024). The
  promoted copy is [`promoted_factors`](@ref).

```jldoctest mixed
julia> for ref in (Refinement(), GMRESRefinement())
           method = Newton(; linsolve = LUFactorization(Float32), refinement = ref)
           x, status = solve(prob, [1.0, 0.5], method, 2.0)
           println(status.code, " ", x ≈ [1.0, 1.0])
       end
SUCCESS true
SUCCESS true
```

### The remedy chain

A step whose factorisation in `TF` fails, a step whose refinement stages all fail, or two
consecutive steps whose linear model is inaccurate in `TF` and which lower the residual by less
than half, make the solve refactorise the Jacobian in `T`. The stages of [`Refinement`](@ref) are
the LU-IR passes; [`GMRESRefinement`](@ref) runs GMRES-IR after them, so it refactorises in `T`
only when GMRES-IR fails too. The chain of a step is therefore

```
LU-IR  →  GMRES-IR (with GMRESRefinement)  →  refactorisation in T
```

[`SolverStatus`](@ref) counts each failed stage in `step_failures` and records a factorisation in
`T` in `promoted`. A refinement drives the linear residual ``J d + F`` to zero, so it does not
apply to a [`LeastSquaresProblem`](@ref) with more residuals than unknowns, nor to the damped
[`LevenbergMarquardt`](@ref) step; a [`GMRES`](@ref) solve has no factor to refine with.

A value iterate (`Number`, `SVector`) has no Krylov basis, so a `GMRESRefinement` there runs the
LU-IR passes alone.

## When a lower factorisation precision pays

A factorisation in `TF` below `T` saves time and memory only where the factorisation dominates the
cost of a step, that is where the evaluation of `F` and `J` costs less than the
``O(n^3)`` factorisation (Kelley 2022). In R1 the gain is registers and memory, not operations.

Its applicability depends on the condition of the Jacobian. Without refinement, Newton's method
converges while the error of the solve in `TF`, roughly ``\kappa(J)\,\varepsilon(T_F)`` times a
modest function of `n`, stays below one (Higham and Mary 2022); the convergence slows as it grows.
LU-IR converges under a condition of the same form and then recovers the iteration count of a
solve in `T` (Carson and Higham 2018). GMRES-IR with promoted factors extends the range of
condition numbers further (Amestoy et al. 2024). Where these conditions fail, the remedy chain
above refactorises in `T`, and `status.promoted` records it.

The repository holds `scripts/gmres_ir_comparison.jl`, which compares the Newton iteration counts
and the linear residuals of `Refinement` and `GMRESRefinement` over a range of condition numbers.

## Mixed precision by backend

The pairs `T = Float64` with `TF = Float32` (and their complex forms) run on the CPU, on CUDA and on
ROCm. On CUDA, a `Float32` factor of a `Float64` problem reaches the cuSOLVER method for
`Float32`, and on ROCm a `Float32` LU or QR factor reaches the rocSOLVER method. Metal has no
`Float64`, so `T = Float32` there and the pair with `TF = Float32` does not apply; see
[Device precision](@ref "Device precision").

A residual can itself be less accurate than `T`, for instance when it calls a `Float32`
computation inside a `Float64` solve. The limit of the solve is then the accuracy of `F`, and the
tolerance must respect it: a tolerance below that floor ends on `STALLED`. The
[`BatchEliminated`](@ref) problem shows this with a `Float32` batch inside a `Float64` outer
solve.

## References

- P. Amestoy, A. Buttari, N. J. Higham, J.-Y. L'Excellent, T. Mary and B. Vieublé,
  *Five-Precision GMRES-Based Iterative Refinement*, SIAM Journal on Matrix Analysis and
  Applications 45(1), 529–552 (2024).
  [doi:10.1137/23M1549079](https://doi.org/10.1137/23M1549079)
- E. Carson and N. J. Higham, *Accelerating the Solution of Linear Systems by Iterative
  Refinement in Three Precisions*, SIAM Journal on Scientific Computing 40(2), A817–A847 (2018).
  [doi:10.1137/17M1140819](https://doi.org/10.1137/17M1140819)
- N. J. Higham and T. Mary, *Mixed precision algorithms in numerical linear algebra*, Acta
  Numerica 31, 347–414 (2022).
  [doi:10.1017/S0962492922000022](https://doi.org/10.1017/S0962492922000022)
- C. T. Kelley, *Newton's Method in Mixed Precision*, SIAM Review 64(1), 191–211 (2022).
  [doi:10.1137/20M1342902](https://doi.org/10.1137/20M1342902)
