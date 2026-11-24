```@meta
CurrentModule = GeometricSolvers
```

# [Linear solvers](@id linear)

The linear solver of a [`Newton`](@ref) step is its `linsolve`. A linear-solver method is a
[`LinearMethod`](@ref): either a factorisation, which the state factorises into a factor of the
type that the iterate selects, or the matrix-free [`GMRES`](@ref).

## Factorisation methods

| method | factorisation | matrix |
|:--|:--|:--|
| [`LUFactorization`](@ref) | LU with partial pivoting | square |
| [`LDLTFactorization`](@ref) | Bunch–Kaufman ``L D L^\mathsf{T}`` or ``L D L^\mathsf{H}`` with ``1 \times 1`` and ``2 \times 2`` pivots | symmetric or Hermitian, reads the lower triangle of a real matrix and both triangles of a complex one |
| [`CholeskyFactorization`](@ref) | Cholesky ``L L^\mathsf{H}`` without pivoting (LAPACK `potrf` on the host) | Hermitian or real symmetric positive definite, reads the lower triangle and the real part of the diagonal |
| [`QRFactorization`](@ref) | Householder QR | square, or ``m \times n`` with ``m \ge n`` for least squares |
| [`SVDFactorization`](@ref) | one-sided Jacobi SVD (static, device) or LAPACK `gesvd` (host) | square or ``m \ge n``, rank-revealing |

Each method takes an optional factorisation precision: `LUFactorization()` factorises in the
working type of the problem, and `LUFactorization(Float32)` in `Float32`. The precision is the
type parameter `TF` of the method, so the method stays `isbits` and enters a kernel. The page
[Mixed precision](@ref mixed) describes when a lower precision applies.

The type of the iterate selects the factor:

| iterate | LU | LDLT | Cholesky | QR | SVD |
|:--|:--|:--|:--|:--|:--|
| `Number`, `SVector` (R1, R2) | [`StaticLU`](@ref) | [`StaticLDLT`](@ref) | [`StaticCholesky`](@ref) | [`StaticQR`](@ref), [`StaticRectQR`](@ref) | [`StaticSVD`](@ref), [`StaticRectSVD`](@ref) |
| host `Vector` (R3) | [`DenseLU`](@ref) | [`DenseLDLT`](@ref) | [`DenseCholesky`](@ref) | [`DenseQR`](@ref) | [`DenseSVD`](@ref) |
| GPU array (R3) | [`DeviceLU`](@ref) | [`DeviceLDLT`](@ref) | none: `ArgumentError` | [`DeviceQR`](@ref) | [`DeviceSVD`](@ref) |

- The **static factors** are scalar loops over `SMatrix` data with `Int32` pivots. They return
  new values, `isbits` for an `isbits` element type, so they run in one GPU thread.
- The **dense factors** call LAPACK (`getrf`, `sytrf`/`hetrf`, `potrf`, `geqrf`, `gesvd`) on
  buffers that [`init`](@ref) allocates once. An LU of `n` from 8 to 32 in `Float32`, `Float64`
  or `ComplexF64` runs an own recursive LU that pivots by LAPACK's rule, in place of `getrf`; see
  [CPU and GPU backends](@ref backends).
- The **device factors** are KernelAbstractions kernels: panel pivoting and trailing updates
  for LU, column-norm reductions and reflector application for QR, the Bunch–Kaufman pivots for
  LDLT, and parallel one-sided Jacobi sweeps for the SVD. On CUDA and Metal, vendor libraries
  replace some of them; see [CPU and GPU backends](@ref backends).

A numerical failure of a factorisation does not throw: a zero pivot gives the status `SINGULAR`.
A configuration or pattern error can throw an `ArgumentError`, for example a sparse factor whose
pattern the regularisation changes. The SVD of an
``m \times n`` matrix truncates at the rank cutoff ``\sqrt{m}\,\varepsilon(T_F)\,\sigma_1``, or at
``\mathrm{rtol}\,\sigma_1`` with `SVDFactorization(; rtol)`, so its solve is the minimum-norm step
on the numerical range. The bounds that the test suite verifies for
each factor are on the page [Numerical Stability and Backward-Error Bounds](@ref stability).

```jldoctest linear
julia> using GeometricSolvers, StaticArrays

julia> F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2]);

julia> prob = NonlinearProblem(F);

julia> results = [solve(prob, SVector(1.0, 0.5), Newton(; linsolve = m), 2.0)[1]
                  for m in (LUFactorization(), QRFactorization(), SVDFactorization())];

julia> all(x -> x ≈ SVector(1.0, 1.0), results)
true
```

[`LDLTFactorization`](@ref) reads only the lower triangle, so it serves a symmetric Jacobian, such
as the Hessian block of a [`StationarityProblem`](@ref) with `E == U`:

```jldoctest linear
julia> S(x, p) = (x[1]^2 + x[2]^2) / 2 + x[1]^4 / 4 - p * x[1] + x[1] * x[2];

julia> x, status = solve(StationarityProblem(S), [0.5, 0.5], Newton(; linsolve = LDLTFactorization()), 1.0);

julia> x ≈ [1.0, -1.0]
true
```

With an [`SVDFactorization`](@ref), the status reports the numerical rank of the last factor in
`rank`, together with `nullnorm` and `gradratio`, which classify a stall; see
[Status and stopping test](@ref).

### Choosing a factorisation

- [`LUFactorization`](@ref) for a general square Jacobian.
- [`LDLTFactorization`](@ref) for a symmetric or Hermitian Jacobian; it needs about half the
  work of LU and handles an indefinite matrix with its ``2 \times 2`` pivots.
- [`CholeskyFactorization`](@ref) for a Hermitian or real symmetric Jacobian that is positive
  definite at every iterate, such as the Hessian of a [`StationarityProblem`](@ref) with `E == U`,
  or ``J + \mu I`` with a Hermitian positive semidefinite ``J`` and ``\mu > 0``. It takes no
  pivots, and a Jacobian that is not positive definite gives `SINGULAR`, with the order of the
  first leading minor that is not positive in `info`. It runs on a host `Array` and in R1 and R2;
  in R3 a GPU array raises an `ArgumentError`.
- [`QRFactorization`](@ref) for a least-squares problem, and where an orthogonal factorisation is
  wanted.
- [`SVDFactorization`](@ref) for a rank-deficient or nearly singular Jacobian. It costs more than
  the others, and it gives the rank diagnostics.

## [Sparse direct solves](@id sparse)

When Sparspak is loaded, the package extension `GeometricSolversSparspakExt` provides a sparse LU
for a host `Vector` iterate whose problem carries a `SparseMatrixCSC{<:Any, Int}` pattern, with
an [`LUFactorization`](@ref) in the element type of the iterate. The first factorisation runs the
symbolic ordering and factorisation, and every later step reuses them. The pattern comes from an
explicit `jacobian_prototype`, or, when SparseMatrixColorings is loaded, from a sparsity detector
of an `AutoSparse` back end (see [Automatic differentiation](@ref ad)).

The Jacobian then has to fill a sparse matrix: give the problem an analytic `J!` that writes the
nonzeros of the prototype, or give `Newton` an `AutoSparse` back end.

```jldoctest linear
julia> using SparseArrays, Sparspak

julia> n = 8;

julia> function Fs!(r, u, p)
           for j in 1:n
               l = j == 1 ? 0.0 : u[j - 1]
               q = j == n ? 0.0 : u[j + 1]
               r[j] = u[j] * (1 + l^2 + q^2) - 1
           end
           return r
       end;

julia> function Js!(J, u, p)
           rows, vals = rowvals(J), nonzeros(J)
           for c in 1:n, k in nzrange(J, c)
               j = rows[k]
               l = j == 1 ? 0.0 : u[j - 1]
               q = j == n ? 0.0 : u[j + 1]
               vals[k] = c == j ? 1 + l^2 + q^2 : 2 * u[j] * u[c]
           end
           return J
       end;

julia> P = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1));

julia> solver = init(NonlinearProblem(Fs!, Js!; jacobian_prototype = P), zeros(n), Newton());

julia> x, status = solve!(solver);

julia> status.code === SUCCESS
true

julia> nameof(typeof(solver.state.fac))
:SparspakLU
```

[`UMFPACKFactorization`](@ref) gives the sparse LU of UMFPACK, from `SparseArrays`, on the same
pattern for a host `Vector{Float64}` or `Vector{ComplexF64}` iterate, with no package to load; it
raises `ArgumentError` at `init` for another element type and for a problem with no sparse pattern.
With `CUDA` and `CUDSS` loaded, an [`LUFactorization`](@ref) on a `CuVector` iterate with a
`CuSparseMatrixCSR` `jacobian_prototype` and an analytic `J!` gives the sparse LU of cuDSS.
[`QRFactorization`](@ref), [`SVDFactorization`](@ref), a factorisation precision other than the
element type of the iterate, and every other complex or device iterate have no sparse factor. The
section [Factorisation and solve](@ref) of the API lists which configurations take a detected
pattern and which raise an error.

## Matrix-free Newton–Krylov: `GMRES`

[`GMRES`](@ref)`(; maxiter = 30, restart = nothing, forcing, preconditioner, jvp_eltype)` is a
linear method without a factor. A [`Newton`](@ref) with it holds no ``n \times n`` Jacobian: each
inner iteration takes one Jacobian–vector product, from automatic differentiation or from a
[`JVPFunction`](@ref) that the caller writes. The step solves ``J M^{-1} u = -F(x)`` to the relative
tolerance ``\eta_k`` of the forcing term and returns ``d = M^{-1} u``.

```jldoctest linear
julia> F!(y, x, p) = (y[1] = x[1]^2 + x[2]^2 - p; y[2] = x[1] - x[2]; y);

julia> jvp!(Jv, x, v, p) = (Jv[1] = 2x[1] * v[1] + 2x[2] * v[2]; Jv[2] = v[1] - v[2]; Jv);

julia> method = Newton(; linsolve = GMRES(; maxiter = 10), jacobian = JVPFunction(jvp!));

julia> solver = init(NonlinearProblem(F!), [1.0, 0.5], method, 2.0);

julia> x, status = solve!(solver);

julia> x ≈ [1.0, 1.0] && status.jvp_calls == jvp_calls(solver) > 0
true
```

The **forcing term** sets ``\eta_k``. [`EisenstatWalkerChoice2`](@ref) is the default and
[`EisenstatWalkerChoice1`](@ref) the other adaptive choice of Eisenstat and Walker (1996);
[`FixedForcing`](@ref)`(η)` holds ``\eta`` constant. An adaptive term solves loosely far from the
root and tightly near it, which saves inner iterations where a tight solve buys nothing.

A **right preconditioner** `M` is any object with methods of [`refresh!`](@ref), called once per
Newton step, and [`apply_preconditioner!`](@ref), called before every product. A stored Jacobian
factor that `ldiv!` applies is the frozen-Jacobian preconditioner of a Newton–Krylov method. The
example below takes the diagonal of the Jacobian instead:

```jldoctest linear
julia> struct Jacobi
           d::Vector{Float64}
       end

julia> GeometricSolvers.refresh!(P::Jacobi, prob, x, Fx, p) = (P.d .= (2x[1], -1.0); P);

julia> GeometricSolvers.apply_preconditioner!(z, P::Jacobi, v) = (z .= v ./ P.d; z);

julia> method = Newton(; linsolve = GMRES(), preconditioner = Jacobi(zeros(2)));

julia> x, status = solve(NonlinearProblem(F!), [1.0, 0.5], method, 2.0);

julia> x ≈ [1.0, 1.0]
true
```

`jvp_eltype = Float32` evaluates the product in `Float32` and keeps the residual and the iterate
in the working precision: a defect-correction solve. [`f_calls`](@ref) and [`jvp_calls`](@ref)
count the evaluations of the last solve, and the status holds the same counts.

`GMRES` needs an `AbstractVector` iterate, on the host or on a device. A `Number` or a static
iterate has no Krylov basis here, and a [`LeastSquaresProblem`](@ref) has no square system. It
takes no factor reuse and no [`Refinement`](@ref).

## Eigensolver

[`jacobi_eigen`](@ref) is a function on a static Hermitian or real symmetric matrix, not a linear
method. It computes the eigenvalues and the unitary eigenvectors by cyclic two-sided Jacobi, on the
rotations of the Jacobi SVD, and returns `(values, vectors, status)`, all `isbits`, so it runs in a
kernel.

```jldoctest linear
julia> A = SMatrix{2, 2}(2.0, 1.0, 1.0, 2.0);

julia> values, vectors, status = jacobi_eigen(A);

julia> sort(values) ≈ [1.0, 3.0] && status === SUCCESS
true
```

## References

- S. C. Eisenstat and H. F. Walker, *Choosing the Forcing Terms in an Inexact Newton Method*,
  SIAM Journal on Scientific Computing 17(1), 16–32 (1996).
  [doi:10.1137/0917003](https://doi.org/10.1137/0917003)
