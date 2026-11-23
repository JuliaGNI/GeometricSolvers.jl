```@meta
CurrentModule = GeometricSolvers
```

# Linear-solver methods

The factorisation precision `TF` of a linear-solver method is a type parameter. `Nothing` selects
the working type of the problem, a real float type a lower (or higher) precision, and a marker a
vendor tensor-core path. These names are exported.

```@docs
LUFactorization
LDLTFactorization
QRFactorization
SVDFactorization
TF32
FP16
BF16
```

## Factorisation and solve

A linear-solver method enters a solve through [`init_linear`](@ref) (and [`dense_factor`](@ref)),
[`factorize!!`](@ref) and [`ldiv!!`](@ref). For a static matrix (R1) [`factorize!!`](@ref) and
[`ldiv!!`](@ref) return new `isbits` values, so they run inside a kernel: [`StaticLU`](@ref) is LU
with partial pivoting and `Int32` pivots, [`StaticLDLT`](@ref) is Bunch–Kaufman `L D Lᵀ` /
`L D Lᴴ` with `1×1` and `2×2` diagonal blocks, [`StaticQR`](@ref) and [`StaticRectQR`](@ref) are
Householder QR without column pivoting, and [`StaticSVD`](@ref) and [`StaticRectSVD`](@ref) are a
one-sided Jacobi singular value decomposition whose solve truncates at the rank cutoff
`√n · eps(TF) · σ₁` for a square matrix and `√m · eps(TF) · σ₁` for a rectangular one, `m` its row
count, which is what makes a rank-revealing solve finite. `SVDFactorization(; rtol)` sets the
cutoff to `rtol · σ₁` on every route, R1 and R3. For a vector state (R3) with no sparse
pattern (`jacobian_pattern(prob, jacprep) === nothing`),
[`init_linear`](@ref) pre-allocates [`DenseLU`](@ref), [`DenseLDLT`](@ref), [`DenseQR`](@ref) or
[`DenseSVD`](@ref) on a host `Array` and [`DeviceLU`](@ref), [`DeviceLDLT`](@ref),
[`DeviceQR`](@ref) or [`DeviceSVD`](@ref) on an `AbstractGPUArray`, and [`factorize!!`](@ref) and
[`ldiv!!`](@ref) reuse its buffers in place: through LAPACK on the host, and on a device through
the `GeometricSolversMetalExt` (`Metal` MPS `lu!` and `ldiv!` on `MtlMatrix{Float32}` and
`MtlVector{Float32}`, with every MPS failure status reported as a singular factor) and
`GeometricSolversCUDAExt` (`CUDA` cuSOLVER `getrf!`, `getrs!`, `geqrf!`, `ormqr!` with cuBLAS
`trsv!`, and `gesvdj!` on `CuMatrix{T}` and `CuVector{T}` for `T <:
LinearAlgebra.BlasFloat`, falling back to `ka_factorize_svd!` where `gesvdj!` does not converge,
and calling it in place of `gesvdj!` where the `rtol` cutoff lies below `√m · eps(real(T))`, `m`
the number of rows)
package extensions where available and through `KernelAbstractions` otherwise. When `Sparspak` is
loaded and [`jacobian_pattern`](@ref) is a `SparseMatrixCSC{<:Any, Int}` on a host `Vector{T}` (`T <:
AbstractFloat`) — from an explicit `jacobian_prototype`, or detected automatically by
`GeometricSolversSparseADExt` when `SparseMatrixColorings` is loaded — the
`GeometricSolversSparspakExt` package extension pre-allocates a `SparspakLU`
workspace and unfactorised `Sparspak.SparseSolver` in `init_linear` for an `LUFactorization` whose
factor element type is `T` (`LUFactorization()` or `LUFactorization(T)`); the first `factorize!!`
call runs symbolic ordering and symbolic factorisation, and subsequent `factorize!!` and `ldiv!!`
calls reuse them across Newton steps. That configuration is the only one with a sparse factor, and
[`has_sparse_factor`](@ref) names it. A detected pattern is used only where `has_sparse_factor`
holds and, under a `Newton` `regularization ≠ 0`, only where the pattern holds the whole structural
diagonal. Everywhere else — `QRFactorization`, `SVDFactorization`, mixed precision such as
`LUFactorization(Float64)` on a `Float32` iterate, a complex iterate, or no `Sparspak` — a
detected pattern gives the dense factor and a dense Jacobian buffer, as a solve with no pattern
does. An explicit `jacobian_prototype` outside that configuration raises an error in `init_linear`
or at the first `factorize!!`. A [`LeastSquaresProblem`](@ref) takes no detected pattern, because
`SparspakLU` is square: its `m × n` factor is dense.
[`factorize!!`](@ref) states what a singular matrix gives: an SVD factor records its numerical rank
in `rank` apart from its status, and [`exact_solve`](@ref) tells `Newton` whether the solve is
exact or truncated; [`singular_values`](@ref) gives the singular values of a [`StaticSVD`](@ref)
or [`StaticRectSVD`](@ref). These names are not exported and are reached as
`GeometricSolvers.name`.

```@docs
init_linear
has_sparse_factor
dense_factor
factorize!!
ldiv!!
factor_eltype
StaticLU
StaticLDLT
StaticQR
StaticRectQR
StaticSVD
StaticRectSVD
DenseLU
DeviceLU
DenseLDLT
DeviceLDLT
DenseQR
DeviceQR
DenseSVD
DeviceSVD
returncode
exact_solve
singular_values
```

## Matrix-free Krylov solves

[`GMRES`](@ref) is a [`LinearMethod`](@ref) with no factor. The Jacobian enters a step only through
a Jacobian–vector product, so a [`Newton`](@ref) with it holds no `n × n` Jacobian (`state.J ===
nothing`) and serves a system that no array can hold the Jacobian of: a small system whose every
column costs a full evaluation of an expensive residual, or a device array. Its step solves
``J M^{-1} u = -F(x)`` and returns ``d = M^{-1} u`` to the relative tolerance that the forcing term
returns.

The forcing term sets that tolerance. [`EisenstatWalkerChoice2`](@ref) is the default, and
[`EisenstatWalkerChoice1`](@ref) and [`FixedForcing`](@ref) are the alternatives; a term is an
[`AbstractForcingTerm`](@ref), and [`compute_forcing`](@ref) is its rule. The right preconditioner
of `GMRES(; preconditioner = P)` is refreshed by [`refresh!`](@ref) once per Newton step and
applied by [`apply_preconditioner!`](@ref) before every product. A matrix-free problem supplies its
own product as [`JVPFunction`](@ref)`(jvp!)` in the `jacobian` field of [`Newton`](@ref), and the
solve reports the exact counts of its residual and product evaluations in
[`f_calls`](@ref)/[`jvp_calls`](@ref) and in the `f_calls` and `jvp_calls` fields of
[`SolverStatus`](@ref). A combinator other than [`Repeat`](@ref) reports `0` in those fields:
the members of a step-level combinator count only their own steps, and [`Fallback`](@ref) does
not add the counts of the solves of its members. `GMRES(; maxiter)` caps the inner iterations of
one Newton step, and
[`max_inner_iterations`](@ref) gives the largest count of the last solve.

The Arnoldi process of `GMRES` orthogonalises each new Krylov vector by classical Gram–Schmidt with
a second pass (CGS2), which takes two reductions over the basis per inner iteration, and one more,
or three where [`rnorm`](@ref) scales, on an iteration where the norm of the new vector is formed
explicitly. `GMRES(; mgs = true)` selects modified Gram–Schmidt, which takes `j + 1` reductions at
inner iteration `j`; on a device, each reduction is a synchronisation. The inner GMRES of a
[`GMRESRefinement`](@ref) orthogonalises by modified Gram–Schmidt whatever `GMRES` selects.

The matrix-free [`DogLeg`](@ref) runs on the Arnoldi relation of an unrestarted `GMRES` only
(`restart == maxiter`), and a `Newton` with a preconditioner or a restarted `GMRES` under it raises
an `ArgumentError`. These names are exported.

```@docs
GMRES
AbstractForcingTerm
EisenstatWalkerChoice2
EisenstatWalkerChoice1
FixedForcing
JVPFunction
refresh!
apply_preconditioner!
f_calls
jvp_calls
```

## Eigensolver

[`jacobi_eigen`](@ref) is a function on a matrix and not a linear-solver method: cyclic two-sided
Jacobi for a Hermitian or a real symmetric matrix, on the rotations and the round-robin ordering of
the Jacobi SVD with a Frobenius-norm pair floor `eps(R) * ‖H‖_F`, for the case
`jacobi_eigen(im * M)` of a real skew `M` that the `ProjectedSkew` exponential algorithm of
GeometricOptimizers needs. It adds no method to `LinearAlgebra.eigen`, because a method of `eigen`
for a `StaticArrays` type would be type piracy.

```@docs
jacobi_eigen
```
