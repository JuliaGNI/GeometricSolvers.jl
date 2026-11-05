```@meta
CurrentModule = GeometricSolvers
```

# [Numerical Stability and Backward-Error Bounds](@id stability)

This page states the backward-error and spectral-accuracy bounds satisfied by the linear and
spectral kernels in `GeometricSolvers`, together with their literature sources and the
verification domain covered by the test suite (`test/linear/stability.jl`).

Throughout this page:

- ``T \in \{\texttt{Float32}, \texttt{Float64}, \texttt{ComplexF32}, \texttt{ComplexF64}\}``
  denotes the problem element type and ``R = \texttt{real}(T)`` its real precision, with machine
  epsilon ``\varepsilon_R = \texttt{eps}(R)``.
- ``n`` denotes the matrix dimension and ``\kappa_2(A) = \sigma_1(A) / \sigma_r(A)`` the 2-norm
  condition number of a matrix ``A \in T^{n \times n}`` of numerical rank ``r``.
- The empirical backward-error and spectral constant used across all bounds is
  ```math
  c = 16.
  ```

---

## Kernel Summary

| Kernel family | Factor / function types | Bound verified (``c = 16``) | Literature source |
|:---|:---|:---|:---|
| LU with partial pivoting | [`StaticLU`](@ref), [`DenseLU`](@ref), [`DeviceLU`](@ref) | ``\dfrac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_R`` | Higham (2002), §9.3–9.4 |
| Householder QR | [`StaticQR`](@ref), [`DenseQR`](@ref), [`DeviceQR`](@ref) | ``\dfrac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_R`` | Higham (2002), §19.3 |
| SVD (one-sided Jacobi and LAPACK) | [`StaticSVD`](@ref), [`DenseSVD`](@ref), [`DeviceSVD`](@ref) | ``\dfrac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_R``, ``\;\dfrac{\max_i |\hat{\sigma}_i - \sigma_i|}{\sigma_1} \le c \cdot n \cdot \varepsilon_R`` | Demmel and Veselić (1992); Higham (2002) |
| Hermitian Jacobi eigensolver | [`jacobi_eigen`](@ref) | ``\dfrac{\|H \hat{V} - \hat{V} \operatorname{Diagonal}(\hat{\lambda})\|_2}{\|H\|_2} \le c \cdot n \cdot \varepsilon_R``, ``\;\|\hat{V}^* \hat{V} - I\|_2 \le c \cdot n \cdot \varepsilon_R``, ``\;\dfrac{\max_i |\hat{\lambda}_i - \lambda_i|}{\lambda_1} \le c \cdot n \cdot \varepsilon_R`` | Demmel and Veselić (1992) |
| Rank-deficient SVD minimum-norm solve | [`StaticSVD`](@ref), [`DenseSVD`](@ref), [`DeviceSVD`](@ref) | ``\dfrac{\|\hat{x} - x_{\min}\|_2}{\|x_{\min}\|_2} \le c \cdot n \cdot \kappa_2(A) \cdot \varepsilon_R \cdot \dfrac{\|b\|_2}{\|A x_{\min}\|_2}`` | Demmel and Veselić (1992); Higham (2002), §20.4 |
| Mixed-precision scaled solve (`TF ≠ T`) | [`ldiv!!`](@ref) on [`LUFactorization`](@ref), [`QRFactorization`](@ref), [`SVDFactorization`](@ref) | ``\dfrac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_{\operatorname{real}(F)}`` | Kelley (2023), eqs. 2.17–2.18 |

---

## LU with Partial Pivoting (`StaticLU`, `DenseLU`, `DeviceLU`)

[`LUFactorization`](@ref) dispatches to:

- [`StaticLU`](@ref) on `SMatrix{n, n, T}` (register-resident Gaussian elimination with partial
  pivoting),
- [`DenseLU`](@ref) on host `Array{T, 2}` (LAPACK `getrf!` / `getrs!`), and
- [`DeviceLU`](@ref) on GPU `AbstractGPUArray{T, 2}` backends such as `JLArray` and `MtlArray`
  (`KernelAbstractions` panel pivoting and trailing-submatrix update).

By Higham (2002, §9.3–9.4), Gaussian elimination with partial pivoting produces a computed
solution ``\hat{x}`` to ``A x = b`` that is the exact solution of ``(A + \Delta A)\hat{x} = b``,
where ``\|\Delta A\|_2 / \|A\|_2`` is bounded by ``O(n \, \rho_n \, \varepsilon_R)`` for growth
factor ``\rho_n``. On systems with moderate pivot growth, the normwise backward error satisfies

```math
\frac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_R, \qquad c = 16.
```

---

## Householder QR (`StaticQR`, `DenseQR`, `DeviceQR`)

[`QRFactorization`](@ref) dispatches to:

- [`StaticQR`](@ref) on `SMatrix{m, n, T}` (register-resident Householder reflections with
  standard complex sign selection),
- [`DenseQR`](@ref) on host `Array{T, 2}` (LAPACK `geqrf!` / `ormqr!` / `unmqr!` and
  `trtrs!`), and
- [`DeviceQR`](@ref) on GPU `AbstractGPUArray{T, 2}` backends (`KernelAbstractions` column-norm
  reduction and reflector application).

By Higham (2002, Theorem 19.4, §19.3), Householder QR factorization and triangular
back-substitution are unconditionally backward stable. For any nonsingular square system
``A x = b``, the computed solution ``\hat{x}`` satisfies the normwise backward error bound

```math
\frac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_R, \qquad c = 16.
```

---

## SVD and Hermitian Jacobi Eigensolver (`StaticSVD`, `DenseSVD`, `DeviceSVD`, `jacobi_eigen`)

[`SVDFactorization`](@ref) dispatches to:

- [`StaticSVD`](@ref) on `SMatrix{n, n, T}` (one-sided Jacobi orthogonalization of ``A`` followed
  by column normalization and descending sort),
- [`DenseSVD`](@ref) on host `Array{T, 2}` (LAPACK `gesdd!` divide-and-conquer with `gesvd!`
  fallback), and
- [`DeviceSVD`](@ref) on GPU `AbstractGPUArray{T, 2}` backends (`KernelAbstractions` one-sided
  Jacobi sweeps with complex phase annihilation).

For Hermitian matrices `H::SMatrix{n, n, T}`, [`jacobi_eigen`](@ref) computes the spectral
decomposition ``H = V \operatorname{Diagonal}(\lambda) V^*`` by two-sided cyclic Jacobi sweeps.

By Demmel and Veselić (1992) and Higham (2002):

1. **Full-rank linear solve (`ldiv!!`)**: For nonsingular ``A``, the pseudoinverse backsolve
   ``\hat{x} = \hat{V} \hat{\Sigma}^{-1} \hat{U}^* b`` satisfies the normwise backward error bound
   ```math
   \frac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_R, \qquad c = 16.
   ```
2. **Singular values ([`singular_values`](@ref) on [`StaticSVD`](@ref), `fac.S` on
   [`DenseSVD`](@ref) and [`DeviceSVD`](@ref))**: The computed singular values
   ``\hat{\sigma}_1 \ge \dots \ge \hat{\sigma}_n`` satisfy the Weyl relative bound
   ```math
   \frac{\max_{1 \le i \le n} |\hat{\sigma}_i - \sigma_i|}{\sigma_1} \le c \cdot n \cdot \varepsilon_R, \qquad c = 16.
   ```
3. **Hermitian eigendecomposition ([`jacobi_eigen`](@ref))**: On a Hermitian positive-definite
   matrix ``H`` with exact eigenvalues ``\lambda_1 \ge \dots \ge \lambda_n > 0``, the computed
   eigenpairs ``(\hat{\lambda}, \hat{V})`` satisfy
   ```math
   \frac{\|H \hat{V} - \hat{V} \operatorname{Diagonal}(\hat{\lambda})\|_2}{\|H\|_2} \le c \cdot n \cdot \varepsilon_R, \qquad
   \|\hat{V}^* \hat{V} - I\|_2 \le c \cdot n \cdot \varepsilon_R, \qquad
   \frac{\max_{1 \le i \le n} |\hat{\lambda}_i - \lambda_i|}{\lambda_1} \le c \cdot n \cdot \varepsilon_R,
   ```
   with ``c = 16``.

---

## Rank-Deficient SVD Minimum-Norm `ldiv!!`

When ``A \in T^{n \times n}`` has exact rank ``r < n`` with singular values
``\sigma_1 \ge \dots \ge \sigma_r > 0 = \sigma_{r+1} = \dots = \sigma_n`` and
``\sigma_r / \sigma_1 \gg \tau_{\text{rel}} = \sqrt{\varepsilon_R}``, [`SVDFactorization`](@ref)
across [`StaticSVD`](@ref), [`DenseSVD`](@ref), and [`DeviceSVD`](@ref) detects the numerical
rank `fac.rank == r` by truncating singular values satisfying
``\hat{\sigma}_i \le \max(\tau_{\text{abs}}, \tau_{\text{rel}} \hat{\sigma}_1)``.

By Demmel and Veselić (1992) and Higham (2002, §20.4), the truncated pseudoinverse solve
[`ldiv!!`](@ref) approximates the exact minimum-norm least-squares solution
``x_{\min} = V_{1:r} \Sigma_{1:r}^{-1} U_{1:r}^* b`` with relative forward error bounded by

```math
\frac{\|\hat{x} - x_{\min}\|_2}{\|x_{\min}\|_2} \le c \cdot n \cdot \kappa_2(A) \cdot \varepsilon_R \cdot \frac{\|b\|_2}{\|A x_{\min}\|_2}, \qquad c = 16,
```

where ``\kappa_2(A) = \sigma_1 / \sigma_r`` is the 2-norm condition number of the nonzero
singular spectrum.

---

## Mixed-Precision Factorization and Scaled Solve (`TF ≠ T`)

When a factorization method is constructed with reduced factor precision `TF` (for example
`LUFactorization(Float32)`, `QRFactorization(Float32)`, or `SVDFactorization(Float32)` applied
to a problem with ``T \in \{\texttt{Float64}, \texttt{ComplexF64}\}`` and factor storage type
``F = \texttt{Float32}`` or ``\texttt{ComplexF32}``), [`factorize!!`](@ref) scales ``A`` by a power
of two ``2^{-e}``, converts the scaled matrix into precision ``F``, and computes the factorization
in ``F``.

During [`ldiv!!`](@ref), `GeometricSolvers` implements the scaled mixed-precision solve of
Kelley (2023, eqs. 2.17–2.18):

1. Compute the right-hand side scale ``s = \max_i |b_i| = \|b\|_\infty`` in working precision ``T``
   (returning ``0`` immediately if ``s = 0``).
2. Scale and downcast ``\tilde{b} = F.(b / s)`` into factor precision ``F``, preventing exponent
   underflow or overflow in reduced precision.
3. Solve the factored system for ``\tilde{x}`` in precision ``F``.
4. Rescale by ``s \cdot 2^{-e}`` and convert back to working precision ``T``.

The resulting solution ``\hat{x} \in T^n`` satisfies the normwise backward error bound at the
factor precision ``\varepsilon_{\operatorname{real}(F)} = \texttt{eps}(\operatorname{real}(F))``:

```math
\frac{\|A \hat{x} - b\|_2}{\|A\|_2 \|\hat{x}\|_2 + \|b\|_2} \le c \cdot n \cdot \varepsilon_{\operatorname{real}(F)}, \qquad c = 16.
```

---

## Verification Matrix (`test/linear/stability.jl`)

Every bound above is verified in `test/linear/stability.jl` using random matrices with
Haar-distributed orthogonal or unitary factors (`test/helpers/randsvd.jl`) and geometrically
distributed singular values or eigenvalues:

- **Element types**:
  ``T \in \{\texttt{Float32}, \texttt{Float64}, \texttt{ComplexF32}, \texttt{ComplexF64}\}``
  (`ELTYPES`).
- **Condition numbers**: ``\kappa_2 \in \{10, 10^4\}`` for 32-bit real precision
  (``R = \texttt{Float32}``) and ``\kappa_2 \in \{10, 10^4, 10^8\}`` for 64-bit real precision
  (``R = \texttt{Float64}``); ``\kappa_2 = 100`` with rank ``r = n - 2`` for rank-deficient SVD;
  ``\kappa_2 \in \{10, 10^4\}`` with ``T \in \{\texttt{Float64}, \texttt{ComplexF64}\}`` and
  ``\texttt{TF} = \texttt{Float32}`` for mixed-precision solves.
- **System dimensions**: ``n \in \{4, 16, 64\}`` across dense (`Array`) and device (`JLArray`
  under `allowscalar(false)`) backends, and ``n \in \{4, 16\}`` across static (`SMatrix{n, n, T}`)
  kernels.

---

## References

- **Demmel, J. and Veselić, K.** (1992). *Jacobi's Method is More Accurate than QR*. SIAM
  Journal on Matrix Analysis and Applications, 13(4), 1204–1245.
- **Higham, N. J.** (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. Society
  for Industrial and Applied Mathematics (SIAM), Philadelphia. (§9.3–9.4 for LU with partial
  pivoting; §19.3 for Householder QR; §20.4 for rank-deficient least squares.)
- **Kelley, C. T.** (2023). *Solving Nonlinear Equations with Iterative Methods: Solvers and
  Examples in Julia*. Society for Industrial and Applied Mathematics (SIAM), Philadelphia.
  (Eqs. 2.17–2.18 for scaled mixed-precision linear solves.)
