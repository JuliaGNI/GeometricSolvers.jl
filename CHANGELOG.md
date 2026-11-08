# Release Notes

All notable changes to GeometricSolvers.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from an interface change.


## [Unreleased] — targeting 0.1.0

### Changed

- **`KernelAbstractions` 0.10.5 replaces 0.9, the `Metal` bound becomes `1.12`, and the `CUDA`
  range starts at `6.5.1`.** The floor rises from `0.9.38`, because JLArrays 0.5 needs
  `KernelAbstractions 0.10.5` and `GPUArrays 12.1`; Metal 1.12 is the first Metal that takes
  `GPUArrays` 12, and CUDA 6.5.1 the first CUDA whose `CUDACore` takes KernelAbstractions 0.10.
  `isgpu` is no longer part of the KernelAbstractions API, so the package states it as
  `!(backend isa CPU)`: the CPU backend answers `false`, and CUDA, ROCm, Metal and the `JLArray`
  backend answer `true`, as before. The `CPU()` backend now compiles its kernels, so a kernel
  argument must be `isbits` and a kernel body may not format an error message. Three consequences
  for a caller: a kernel that runs a whole `solve` no longer compiles there, because building its
  options formats a message — the device tests build the state and the options on the host and call
  `solve!!` in the kernel; the validation of `Newton` and of `Static` moves to the constructor that
  `adapt` does not rebuild, so a positional `Newton(...)` and `Static{R}(α)` no longer validate
  (`Newton(; ...)` and `Static(α)` do, as before); and the `order` of a batch is copied into an
  array of the backend, because a kernel reads the backend's own memory. A callable
  `StationarityProblem` datum is now checked for every system of a batch on the host, so a length
  that differs between systems is an `ArgumentError` before the launch instead of inside it.
- **`rnorm` works on a device array whose norm is finite but whose sum of squares is not.** Its
  scaled path summed through a closure that carried the accumulation type as a field, and a device
  kernel argument must be `isbits`, so `rnorm` threw `KernelError: passing non-bitstype argument`
  on an `MtlArray` of such entries — an entry near `floatmax` or near `floatmin` — while the host
  path returned the right value. The summand is now `ScaledSquares{S}`, whose type parameter holds
  the accumulation type and whose only field is the exponent `Int`.

- **`LeastSquaresProblem`, `GaussNewton`, rectangular static, host and device QR and SVD, and the
  stationarity stopping test** (decisions 44 and 57 of the design). `AbstractNonlinearProblem` is
  the common supertype of `NonlinearProblem`, `StationarityProblem` and
  `LeastSquaresProblem{F, J, M}` (exported), which constructs an overdetermined or square
  least-squares problem `min_x ½ ‖F(x, p)‖₂²` with `m ≥ n` residuals (2-argument `F(x, p)` in R1,
  and 3-argument `F!(r, x, p)` with `m::Int` in R3).
  `GaussNewton(; linsolve = QRFactorization(), globalization = Static(), jacobian = nothing,
  refinement = NoRefinement(), regularization = 0, reuse = nothing)` (exported) constructs a
  `Newton` method configured for least-squares problems, using `MeasuredSlope()` for line-search
  globalizations (`φ′(0) = 2 Re⟨F, J d⟩ = -2 ‖J d‖₂²`). `StaticRectQR{M, N, TF, L}` and
  `StaticRectSVD{M, N, TF, L, LN, R}` factorise rectangular static matrices `SMatrix{M, N, T}` with
  `M > N ≥ 1` and `M ≤ 32` without allocation (delegating `M == N` to `StaticQR` and `StaticSVD`,
  and throwing `DimensionMismatch` for `M < N` or `N == 0` and `ArgumentError` for `M > 32`), and
  solve the minimum-norm least-squares problem in `ldiv!!`; `DenseQR`, `DeviceQR`, `DenseSVD` and
  `DeviceSVD` support rectangular `m × n` dimensions (`m ≥ n`) on `Array` and `AbstractGPUArray`
  alike, and `LevenbergMarquardt` supports rectangular `m × n` Jacobians across R1 and R3 in both
  its SVD and stacked-QR paths, except that a device array iterate takes the SVD path only
  (`SVDFactorization()` with `scaling = :none`), and its stacked-QR path raises an `ArgumentError`.
  A `LeastSquaresProblem` holds no Jacobian prototype and takes no pattern that an `AutoSparse`
  detector finds (the one sparse factor, `SparspakLU`, is square), so its factors are dense, and
  it has no
  implicit differentiation: `solve` with a `ForwardDiff.Dual` in `x₀` or `p` raises an
  `ArgumentError` (known issue K65). `Options{R}` gains `g_tol::R = sqrt(eps(R))` (`≥ 0`), and
  `SolverStatus{R}` gains `stationary::Bool` (`sizeof(SolverStatus{Float32}) == 36`,
  `sizeof(SolverStatus{Float64}) == 56`) together with `stopping_test(status)` (exported), which
  returns `:stationarity`, `:residual` or `:none`. On a `LeastSquaresProblem`, `solve!!` tests the
  residual criterion first (`stationary = false`) and then the relative normal-equation gradient
  `‖J(x)ᴴ F(x)‖ / (‖J(x)‖_F ‖F(x)‖) ≤ options.g_tol` when `‖J(x)‖_F > 0` and `‖F(x)‖ > 0`
  (`stationary = true`). With more residuals than unknowns (`m > n`), a `LeastSquaresProblem`
  refuses a factor in a lower precision than the iterate's (`eps(real(TF)) > eps(real(T))`, for
  example `QRFactorization(Float32)` on a `Float64` iterate) with an `ArgumentError` where the
  state is built, because the Gauss–Newton fixed point with a lower-precision factor is not the
  least-squares point. A factor in a higher precision, such as `QRFactorization(Float64)` on a
  `Float32` iterate, is taken. A square problem (`m == n`) and the stacked `LevenbergMarquardt`
  step (`QRFactorization()`, `LUFactorization()`, `LDLTFactorization()`, or `scaling = :diagonal`,
  which reads no factor) are exempt; `LevenbergMarquardt` with `SVDFactorization()` and
  `scaling = :none` is refused. A factor in a higher precision on more residuals than unknowns is
  not promoted to the working precision after a stall (`StepInfo.promoted` and
  `SolverStatus.promoted` stay `false`, and no step-rule failure is counted), because at a
  nonzero-residual least-squares point the relative linear residual `‖F + J d‖/‖F‖ ≈ 1`, which
  the mixed-precision stall test of a square solve reads as a stall. A square
  `LeastSquaresProblem` (`m == n`) keeps the stall promotion, and a failed factorisation still
  promotes. A `LeastSquaresProblem` with more residuals than unknowns takes no `regularization`
  under any globalization, the stacked `LevenbergMarquardt` step included: `regularization > 0`
  raises an `ArgumentError` where the state is built, on static and array iterates. An
  `LDLTFactorization` is refused outside `LevenbergMarquardt`, as `LUFactorization` is, and a
  combinator (`Additive`, `Multiplicative`, `Repeat`, `Switch`, `Anderson`) refuses a `LeastSquaresProblem` with
  an `ArgumentError`. A `LeastSquaresProblem` refuses a `Refinement` in `Newton` on static and
  array iterates and for every `m`, with an `ArgumentError`, because a `Refinement` drives the
  linear residual `J d + F` to zero, which a least-squares step with more residuals than unknowns
  does not reach (known issue K80). An analytic `J` on the problem is the Jacobian used, not
  automatic differentiation.

- **Automatic Jacobian sparsity detection and colouring for R3 nonlinear solves via the
  `GeometricSolversSparseADExt` package extension (`SparseMatrixColorings`)** (decision 53 of the
  design). `Project.toml` lists `SparseMatrixColorings` (`"0.4.23"`) as the weak dependency that
  triggers the extension, through its `[extensions]` entry `GeometricSolversSparseADExt =
  "SparseMatrixColorings"`. `SparseConnectivityTracer` is a test dependency, not a dependency of the
  package. The extension uses nothing from it, so any sparsity detector works once
  `SparseMatrixColorings` is loaded: `TracerSparsityDetector()` (which needs `SparseConnectivityTracer`
  loaded), `DenseSparsityDetector`, or `KnownJacobianSparsityDetector(P)`. When `SparseMatrixColorings`
  is loaded and a `NonlinearProblem` has `jacobian_prototype === nothing`,
  `GeometricSolvers.jacobian_pattern(::NonlinearProblem{<:Any, <:Any, Nothing},
  prep::DIJacobian{<:AutoSparse})` returns `SparseMatrixCSC(SparseMatrixColorings.sparsity_pattern(
  prep.jacprep))` whenever the `AutoSparse` back end carries a sparsity detector, and `nothing`
  otherwise. The matrix holds the entries the detector supplies, `Bool` from `TracerSparsityDetector`
  and the prototype's element type from `KnownJacobianSparsityDetector(P)`, and `init_linear` needs
  only its integer indices. A back end left at ADTypes' `NoSparsityDetector` default finds only a
  pattern that is true everywhere, which has no sparsity to exploit, so the solve keeps the dense
  factor and Jacobian. `Newton(; jacobian = AutoSparse(AutoForwardDiff(); sparsity_detector =
  TracerSparsityDetector(), coloring_algorithm = GreedyColoringAlgorithm()))` detects the sparsity
  pattern once in `prepare_ad` and colours the columns, so `jacobian!!` evaluates the Jacobian in
  compressed form without a user-supplied `jacobian_prototype`. The detected pattern builds the sparse
  factor only where `GeometricSolvers.has_sparse_factor` holds: an `LUFactorization` on a `Vector`
  iterate of a real `AbstractFloat` type, with the factor in that same precision, and with `Sparspak`
  loaded, which defines `SparspakLU`. Where it does not hold, the solve takes the dense factor and a
  dense Jacobian buffer, as it does with no pattern. That covers `QRFactorization`,
  `SVDFactorization`, mixed precision such as `LUFactorization(Float64)` on a `Float32` iterate, a
  complex iterate, `Sparspak` not loaded, and `Newton(; globalization = LevenbergMarquardt(...))`
  without an explicit `jacobian_prototype`, where the damped step reads a dense `J` and no sparse
  factor. Under `Newton(; regularization = μ)` with `μ ≠ 0`, a
  detected pattern that does not hold the whole structural diagonal also gives the dense factor,
  because `regularize!!` writes `μ` onto the diagonal. When a `NonlinearProblem` carries an explicit
  `jacobian_prototype = P` alongside `AutoSparse`, `prepare_ad` wraps `P` in
  `KnownJacobianSparsityDetector(P)` and the extension method (restricted to
  `NonlinearProblem{<:Any, <:Any, Nothing}`) does not match, so
  `jacobian_pattern(prob::NonlinearProblem, jacprep) = prob.jacobian_prototype` returns `P` directly.
  An explicit prototype is subject to neither rule above: it keeps its errors, namely `init_linear`'s
  `MethodError` and `ArgumentError` for unsupported configurations, and the pattern-mismatch
  `ArgumentError` under regularization when `P` has no structural diagonal (K41).

- **Package extensions `GeometricSolversMetalExt` and `GeometricSolversCUDAExt` dispatch dense
  device factorisations and solves to vendor libraries on supported GPU element types** (decision 48
  of the design). `Project.toml` adds `CUDA` (`5.5, 6`) and `Metal` (`1.10.3 - 1.11`) under
  `[weakdeps]` and `[compat]` with `GeometricSolversCUDAExt` and `GeometricSolversMetalExt` under
  `[extensions]`. When `Metal` is loaded, `factorize!!` and `backsolve!` on
  `DeviceLU{<:MtlMatrix{Float32}}` and `MtlVector{Float32}` dispatch to Metal Performance Shaders
  (`LinearAlgebra.lu!(fac.factors; check = false)` and `LinearAlgebra.ldiv!(LU(fac.factors,
  fac.ipiv, BlasInt(fac.info)), d)`), mapping every MPS failure status to `info > 0`
  (`SINGULAR`) with no fallback, and no copy of `J` where the caller passes the factor buffer
  as `J`. MPS status `> 0` gives `info` = that status, `-2` gives the first index of the
  smallest `|aᵢᵢ|` of the factors, and any other negative status gives `info = 1`, while
  `QRFactorization`, `SVDFactorization`, and `ComplexF32`, `Float16` and `BFloat16`
  `LUFactorization` remain on `KernelAbstractions`. When `CUDA` is loaded, `factorize!!` and
  `backsolve!` on `DeviceLU{<:CuMatrix{T}}`, `DeviceQR{<:CuMatrix{T}}` and
  `DeviceSVD{<:CuMatrix{T}}` for `T <: LinearAlgebra.BlasFloat` dispatch to cuSOLVER and cuBLAS
  (`CUDA.CUSOLVER.getrf!` with `info` the first zero pivot, `CUDA.CUSOLVER.getrs!`,
  `CUDA.CUSOLVER.geqrf!` with diagonal zero and non-finite checks, `CUDA.CUSOLVER.ormqr!` and
  `CUDA.CUBLAS.trsv!`, and `CUDA.CUSOLVER.gesvdj!` with power-of-two input prescaling by `2^-e`,
  rank truncation at `jacobi_tolerance(m, real(T)) * σ̂₁` for an `m × n` matrix, zeroing of
  `U[:, (r + 1):n]`, and
  `ka_svd_backsolve!`, falling back to `ka_factorize_svd!` when `gesvdj!` does not converge),
  while `Float16` and `BFloat16` remain on `KernelAbstractions`.

- **`LevenbergMarquardt{R, S}` damped trust-region globalization for `Newton`** (decision 57 of the
  design). `LevenbergMarquardt{R <: AbstractFloat, S}` (`R = Float64` by default, adapted to
  `real(eltype(x₀))` by `ToReal{R}()`) carries the damping parameters `(μ₀, μ_min, μ_max,
  μ_decrease, μ_increase, ρ_accept, scaling)` with `scaling = Val(s)` for `s ∈ (:none, :diagonal)`
  and validates `0 < μ_min <= μ₀ <= μ_max < Inf`, `0 < μ_decrease < 1 < μ_increase` and `0 <=
  ρ_accept < 1`. `Newton` with `globalization = LevenbergMarquardt(...)` requires `reuse ===
  nothing` and tracks its damping parameter in `state.rule::DampingParameter{R}` (`NaN` from
  `init_state` and `finish!!`, and `R(lm.μ₀)` from `start!!`). Such a `Newton` takes no
  `Refinement`: `Newton(; globalization = LevenbergMarquardt(), refinement = Refinement())` and
  `refinement = true` raise an `ArgumentError` at construction, because a `Refinement` corrects the
  solve of `J d = -F` from the factor and the damped step does not take that solve (K57). `μ` is
  absolute, in the units of
  `J(x)ᴴ J(x)`: the damped model `(Jᴴ J + μ D²) δ = -Jᴴ F(x)` measures it there, so a residual
  scaled by `k` sees the damping of the unscaled one divided by `k²`, and the `μ → 0` limit is the
  truncated Newton step at every scale. With `scaling = :diagonal`, `D` is the diagonal of
  the column norms of `J(x)` floored at `√eps` of the largest of them. A `μ` below `4 eps(R)` —
  which only the initial `μ₀` can be, since the schedule floors at `max(μ_min, 4 eps(R))` — is
  taken as no damping, the truncated Newton step. On `SVDFactorization()` with
  `scaling = :none`, each step factorises `J(x)` once and each trial `μ` computes
  `δ = -V diag(2^-e σ̂_j / (σ̂_j² + 2^{-2e} μ)) Uᴴ F(x)` from the stored `σ̂_j` of the scaled matrix
  `J(x)/2^e` via the scale-safe coordinate weight `lm_svd_coord` across static (`StaticSVD`), host
  (`DenseSVD`) and device (`DeviceSVD`) factors; where `2^{-e} √μ` overflows, at the bottom of the
  factor's scale range, the weight is taken as zero and never as a `NaN`. On `QRFactorization()`,
  `LUFactorization()` or `LDLTFactorization()`, or when `scaling = :diagonal`, the step reduces
  `[J | -F]` once per outer step to an upper-triangular `(R₀, c₀)` and each trial `μ` copies that
  pair and eliminates only its own damping rows `√μ D` by Givens rotations, without forming `Jᴴ J`,
  so a rejected trial costs no factorisation of `J(x)`; that path indexes scalars, and an `init` on
  a device array iterate raises an `ArgumentError` that names `SVDFactorization()` with
  `scaling = :none` as the device step. An `init` on a non-static array iterate whose
  `jacobian_prototype` is sparse raises an `ArgumentError` for every linear method and both
  scalings. The message names the dense alternatives, a dense Jacobian with `LUFactorization()` or
  `QRFactorization()` (the stacked step) or `SVDFactorization()` (the SVD step), or another
  globalization for the sparse LU solve. A `StaticArray` iterate ignores the prototype and solves.
  This closes K55, whose dense stacked solve ran beside a sparse factor that it never read. A
  `StationarityProblem` solves with `Newton(; globalization = LevenbergMarquardt())` on R1 and R3
  iterates for every linear method, with the merit `‖F(x)‖²` that the problem prescribes; the
  stacked reduction reads no symmetry. Steps are accepted when the gain ratio
  `ρ = (‖F(x)‖² - ‖F(x + δ)‖²) / (‖F(x)‖² - ‖F(x) + J δ‖²)` satisfies `ρ > ρ_accept`, decreasing
  `μ` to `max(μ · μ_decrease, max(μ_min, 4 eps(R)))`, and rejected steps set `μ` to
  `max(μ · μ_increase, max(μ_min, 4 eps(R)))` inside the step until a trial step is accepted or
  `μ > μ_max` returns `STALLED` with a finite `μ` in the rule state, so that a solve confined
  to a stationary point ends `STALLED` and not `NONFINITE`. With a linear method whose factor type
  `TF` differs from the working type `T`, such as `LUFactorization(Float32)` on a `Float64` iterate,
  the stacked step reduces `J(x)` in `T` and reads no factor, and the SVD step takes its damped
  direction from the `TF` factor. No LM step promotes, so `StepInfo.promoted` stays `false`, and
  `init` reserves no working-precision factor (`NewtonState.fac_T === nothing`).
  `STANDARD_COLUMNS` in `test/helpers/standard_cells.jl` gains the 10th column
  `"Newton LevenbergMarquardt SVD" => Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())`.

- **The documentation build allows HTML pages up to 400 KiB** (`size_threshold` in
  `docs/make.jl`). The API page renders to about 200 KiB, Documenter's default hard limit, and the
  open pull requests add docstrings to it; the warning limit stays at 100 KiB.

- **Sparse R3 nonlinear solves through `NonlinearProblem(...; jacobian_prototype)`, compressed
  `AutoSparse` Jacobian evaluation with `KnownJacobianSparsityDetector`, and the `Sparspak` package
  extension `GeometricSolversSparspakExt`** (decision 53 of the design). `NonlinearProblem{F, J, P
  <: Union{Nothing, AbstractMatrix}}` gains a third field `jacobian_prototype::P` defaulting to
  `nothing` by keyword (`NonlinearProblem(F, J = nothing; jacobian_prototype = nothing)`), so
  `NonlinearProblem(F)` and `NonlinearProblem(F, J)` remain `isbits` when `F` and `J` are `isbits`.
  `GeometricSolvers.init_linear(method, pattern, x)` dispatches on `(method, pattern, x)` (with
  `dense_factor` forwarding `jacobian_prototype(prob)` and `Newton`'s `init_factor` forwarding
  `factor_pattern(ls, prob, x, jacprep)`), building dense host and device factors when `pattern ===
  nothing`. When `Sparspak` (`0.3.15`) is loaded, `GeometricSolversSparspakExt` defines `SparspakLU`
  and extends `init_linear(m::LUFactorization, P::SparseMatrixCSC{<:Any, Int}, x::Vector{T}) where
  {T <: AbstractFloat}`, `factorize!!` and `ldiv!!`: `init_linear` allocates a `similar(P, T)`
  buffer (never mutating `P`), records `colptr` and `rowval`, and builds the unfactorised
  `Sparspak.SparseSolver` via `sparspaklu(factors; factorize = false)`; `factorize!!` checks that
  the sparsity pattern matches `jacobian_prototype` and that `nonzeros(factors)` is finite before
  calling `sparspaklu!(lu, factors; allow_pattern_change = false)` (which runs symbolic ordering
  and factorisation on the first call and reuses them thereafter), catches an `ErrorException`
  only when `lu.slvr.errflag` is set (a zero pivot) and rethrows any other error, and checks
  `lu.slvr.errflag == 0` together with finiteness of `lu.slvr.lnz` and `lu.slvr.unz` so
  singular and non-finite sparse Jacobians return `info = 1` and `SINGULAR`
  without throwing. `init_newton_J` allocates `similar(fac.factors, eltype(x))` when `fac.factors`
  is a `SparseMatrixCSC` and cannot be aliased directly (such as under `DogLeg`). `prepare_ad` on a
  DifferentiationInterface back end validates `size(P) == (length(r), length(x))` whenever
  `jacobian_prototype` is set, and an `AutoSparse` backend then takes `P` as its
  `ADTypes.KnownJacobianSparsityDetector(P)` for column-coloured sparse AD, replacing any detector
  it carries, while `ChunkedForwardDiff`'s `jacobian!!` is restricted to `J::DenseMatrix`.
  `SparseArrays` joins `[deps]` and `Sparspak` joins `[weakdeps]` with the `[extensions]` entry
  `GeometricSolversSparspakExt`.

- **Numerical stability documentation (`docs/src/stability.md`) and backward-error test suite
  (`test/linear/stability.jl`, `test/helpers/randsvd.jl`).** `docs/src/stability.md` states the
  normwise backward-error and spectral-accuracy bounds (`c = 16`) for `StaticLU`, `StaticQR`,
  `StaticSVD`, `jacobi_eigen`, `DenseLU`, `DenseQR`, `DenseSVD`, `DeviceLU`, `DeviceQR`,
  `DeviceSVD`, rank-deficient SVD minimum-norm `ldiv!!`, and scaled mixed-precision `ldiv!!`
  (`TF ≠ T`), citing Higham (2002), Demmel and Veselić (1992), and Kelley (2023).
  `test/helpers/randsvd.jl` provides Haar-distributed `randsvd`, `randhermitian`, and
  `minnorm_solution` constructors, and `test/linear/stability.jl` verifies every documented bound
  group by group: the full-rank LU, QR and SVD solves and their singular values over `ELTYPES`,
  at condition numbers `κ₂ ∈ {10, 10⁴}` (`Float32`) and `{10, 10⁴, 10⁸}` (`Float64`), with
  `n ∈ {4, 16, 64}` (dense and device) and `n ∈ {4, 16}` (static); `jacobi_eigen` over the same
  condition numbers with `n ∈ {4, 16}`; the rank-deficient SVD minimum-norm solve at `κ₂ = 100`
  with rank `n - 2` and `n ∈ {4, 16}`; and the mixed-precision solve (`TF = Float32`) on the 64-bit
  element types at `κ₂ ∈ {10, 10⁴}` with the dimensions of the full-rank solves.

- **`solve(prob, x₀, method, p)` on a `NonlinearProblem` with a `Newton` or `Picard` method
  differentiates implicitly through a converged solve when `x₀` or `p` contains
  `ForwardDiff.Dual` values** (decision 50 of the design); on a `StationarityProblem` it raises
  an `ArgumentError` (K31).
  `GeometricSolvers.StripDual` (`src/base/adapt.jl`) strips one `ForwardDiff.Dual` layer from
  `ForwardDiff.Dual`, `Complex{<:ForwardDiff.Dual}`, and arrays of either through
  `Adapt.adapt(StripDual(), p)` while leaving tuples, named tuples, numbers and other arrays
  unchanged. When `x₀` or `p` carries
  `ForwardDiff.Dual{Tg, V, N}` values, where `V` is the promoted value type of the dual leaves of
  `x₀` and `p`, `solve` strips the dual layer from `x₀` and `p`, runs the primal solve via `solve!!`
  at the precision of the stripped `x₀` (`status::SolverStatus{R}` with `R = real(eltype(x₀))`),
  and on `status.code === SUCCESS` computes the tangent `ẋ` of the solution `x*` from the implicit
  function theorem `J(x*, p_val) ẋ_k = -∂_k F(x*, p)` across R1 (`Number`, `StaticVector`) and R3
  (`Array`, `JLArray`) iterates: it evaluates one unregularised Jacobian `J(x*, p_val)` and one
  factorisation at `(x*, p_val)` (via `method.linsolve` for `Newton` and `LUFactorization()` with
  `prob.J` or `AutoForwardDiff()` for `Picard`), evaluates `F(x̂, p)` once with `x̂` seeded at `x*`
  with zero partials, and solves the `N` right-hand sides `-∂_k F(x*, p)` against that single
  factor. The tangent depends only on `(x*, p)`, never on the partials of `x₀`. On the `SUCCESS`
  path and on the `NaN`-partials paths alike, the returned `x*` carries `ForwardDiff.Dual{Tg, W, N}`
  with `W = promote_type(real(eltype(x*)), V)`. For `x₀::Float64` and `p` carrying
  `ForwardDiff.Dual{Tg, Float32, 1}`, `W` is `Float64`. When `status.code !== SUCCESS`, `solve`
  skips the tangent solve and returns `(x*, status)` with `NaN` partials. If the tangent
  factorisation at `(x*, p_val)` fails (`returncode(fac) === SINGULAR`), `solve` performs no
  back-substitution and returns `(x*, status)` with `status.code === SINGULAR` and `NaN` partials on
  `x*`; a rank-deficient `SVDFactorization` factor with `info == 0` produces the minimum-norm
  pseudoinverse tangent `-J(x*)⁺ ∂_k F(x*, p)` with `status.code === SUCCESS`. Nested duals
  (`V <: ForwardDiff.Dual`), distinct tags or partial counts across `x₀` and `p`, and a user
  parameter type holding duals without an `Adapt.adapt_structure` method throw an `ArgumentError`.
  `solve`, `init`, `solve!` and `solve_batch!` do not check the type of `prob`, so a `prob` whose
  closure captures an object whose type holds a `ForwardDiff.Dual`, such as a
  `ForwardDiff.GradientConfig` or a dual work buffer, solves on primal inputs. A residual that
  contains a `ForwardDiff.Dual` on primal `x` and `p` throws an `ArgumentError` from the run-time
  check in `solve!!`, which runs at `x₀` and at every residual a `Newton` or `Picard` step
  evaluates, through one helper, and covers `solve` on both its primal and dual paths and `solve!`.
  So a residual that is primal at `x₀` but contains a `ForwardDiff.Dual` only at a later iterate
  throws the same `ArgumentError`. An in-place residual that writes a captured dual into its primal
  buffer throws the `MethodError` of that conversion instead. `init` and `solve!` throw an
  `ArgumentError` for a `ForwardDiff.Dual` in `x₀` (or `x`) or in `p`, and `solve_batch!` for one in
  `X` or `p` only, so a `ForwardDiff.Dual` captured in `prob` is not detected in a batch.

- **Device linear factors `DeviceLU`, `DeviceQR` and `DeviceSVD` separate device workspaces from
  host `DenseLU`, `DenseQR` and `DenseSVD`, and the `KernelAbstractions` LU, QR and SVD kernels use
  workgroup reductions and column-parallel triangular solves** (decision 48 of the design).
  `dense_factor` dispatches on `x::AbstractGPUArray` to construct `DeviceLU` (`(factors, ipiv, rhs,
  info, scale_exponent, status)`), `DeviceQR` (`(factors, τ, rhs, info, scale_exponent, status)`)
  and `DeviceSVD` (`(factors, U, S, VT, V, σ̂, flag, tmp, rhs, info, rank, scale_exponent)`), and on
  `x::AbstractVector` to construct host `DenseLU` (which drops `status`), `DenseQR` and `DenseSVD`,
  removing the `x isa AbstractGPUArray` branches, the duplicate `BlasFloat` `backsolve!` and
  `backsolve_shifted!` methods, and the `unsafe_wrap` `JLArray` adapter in `src/linear/ka/svd.jl`.
  `ka_divisor_exponent` uses `zero(σ)` in place of `zero(R)` inside its `mapreduce` closure so the
  closure is `isbits` on Metal and CUDA. `ka_lu!` replaces its single-thread pivot/swap/scale kernel
  with a single workgroup kernel per column step (`@localmem` pivot reduction with lowest-row-index
  tie-breaking, parallel row swap across `1:n`, and parallel column scaling across `(k + 1):n`),
  `ka_lu_backsolve!` and `ka_qr_backsolve!` replace single-thread triangular solves with
  column-parallel kernels (a workgroup `P` and `L⁻¹` kernel for LU and a workgroup `Qᴴ` kernel for
  QR, followed by `ndrange = k - 1` column updates plus a final diagonal divide for `U⁻¹` and `R⁻¹`),
  and `ka_jacobi_svd_round_kernel!` assigns one workgroup of `G` threads per column pair `(p, q)`
  (`ndrange = (n ÷ 2) * G`), reducing `(α, β, γ)` in `@localmem` and rotating columns `p` and `q` of
  `U` and `V` in parallel across the workgroup.

- **`SolverStatus{R}` carries the range and spectrum diagnostics `rank::Int32`, `nullnorm::R` and
  `gradratio::R`, and `Newton` with `SVDFactorization` supports factor reuse on a static or scalar
  iterate** (decisions 54 and 57 of the design). `SolverStatus{R}` gains the fields `rank::Int32`,
  `nullnorm::R` and `gradratio::R` (`sizeof(SolverStatus{Float32}) == 36`,
  `sizeof(SolverStatus{Float64}) == 56`), with a 6-argument constructor that fills their sentinels
  `(Int32(-1), R(NaN), R(NaN))`, and `GeometricSolvers.record` preserves them across steps. At each
  fresh `Newton` factorisation, `gradratio` is `‖J(x)ᴴ F(x)‖ / (‖J(x)‖_F ‖F(x)‖)` at the
  unregularised Jacobian `J(x)` (`zero(R)` when `‖J(x)‖_F == 0` or `‖F(x)‖ == 0`). Where `J(x)ᴴ F(x)`
  overflows although both norms are finite, the ratio is formed from a copy of `F(x)` scaled by a power
  of two, so it stays finite; on an array iterate that copy allocates, and only then. When `linsolve`
  is `SVDFactorization{P}()` and the factor has `info == 0`, `rank` is `Int32(fac.rank)` and
  `nullnorm` is `‖(I - U_r U_rᴴ) F(x)‖` (`fnorm` when `fac.rank == 0`), computed in the factor
  precision after scaling a nonzero `F(x)` by `maximum(abs, F(x))` when `TF ≠ T`; for
  `LUFactorization`, `QRFactorization`, or a failed SVD factor (`info ≠ 0`), `rank == Int32(-1)`
  and `nullnorm` is `NaN`. Under `regularization = μ ≠ 0`, `rank` and `nullnorm` describe the factored
  matrix `J(x) + μI`, and `gradratio` describes `J(x)`, so the outcome classification in the
  `SolverStatus` documentation holds for `μ = 0` only. Every fresh `Newton` factorisation computes
  these, for LU and QR as well as SVD: `‖J(x)‖_F` and the product `J(x)ᴴ F(x)` with its norm. It takes
  `‖F(x)‖` from the line-search merit or from the norm that the solve already holds, and reads `F(x)`
  again where a merit is non-finite or below its normal range, and on a state that no `start!!` saw,
  or that a solve returned. That is a few matrix-vector passes, of order `n²` against the order `n³`
  of a dense factorisation. Reused steps keep the diagnostics of the last factorisation unless a
  line-search failure triggers a fresh retry, `start!!` resets them to their sentinels at the start
  of each solve, and `finish!!` writes them into the returned `SolverStatus{R}` (while `Picard` and
  any solve that returns before a factorisation keep the sentinels). `NewtonState` pre-allocates an
  empty `StaticSVD` factor slot for `Newton(; linsolve = SVDFactorization(), reuse = k)` on a
  `StaticVector` or `Number` iterate, as it does for `LUFactorization` and `QRFactorization`.
  Measured against `main` at 27ccb1d on Julia 1.13.3 (Apple silicon, `Float64`), with
  `F(x) = A x + x.^3 - b`, a diagonally dominant `A` (`A_ii = 4`, `A_ij = 1/(i+j)`), an analytic
  Jacobian, `x₀ = 0.5` in every entry, and `Newton` with `Static` and no reuse: 5 iterations, so 5
  factorisations per solve. Each run is a fresh process with `--check-bounds=auto`; each timing sample
  folds calls until it lasts at least 20 µs, with 300 samples per run. Each tree gives the minimum per
  run, then the median over 6 cold runs. The load average was 18 to 25, and the run-to-run spread of
  one tree was up to 25 %. A solve costs +27 % (LU) and +5 % (QR) at `SVector{3}`, from 287 to 366 ns
  and from 582 to 612 ns, and +34 % (LU) and +16 % (QR) at `SVector{8}`, from 1110 to 1487 ns and from
  2156 to 2495 ns. A factor step, one `newton_factor` call with its Jacobian evaluation, diagnostics
  and factorisation, costs +5.9 ns (LU) and +17.6 ns (QR) at `SVector{3}`, from 9.1 to 15.0 ns and from
  42.9 to 60.5 ns, and +64 ns (LU) and +67 ns (QR) at `SVector{8}`, from 97.2 to 161.1 ns and from
  280.6 to 347.4 ns. `grad_ratio` by itself measured 6.5 ns at n = 3 and 59 ns at n = 8, in one run.
  An earlier measurement on another problem gave a solve cost of +19 % and +25 % for LU and +9 % and
  +15 % for QR, at `SVector{3}` and `SVector{8}`; the cost depends on the problem.

- **`Newton` takes the truncated minimum-norm step of a rank-deficient SVD** (decision 44 of the
  design). The numerical rank of `StaticSVD` and `DenseSVD` is the new field `rank`, the number of
  singular values above `√n · eps(TF) · σ₁`; `info` is nonzero only where the factorisation
  failed — a sweep that did not converge, or a singular value that is not finite — and `rank` is
  then 0, with the zero step. A rank-deficient matrix gives `info == 0` and `SUCCESS` on a static
  matrix, a host array and a device array alike, where the static factor gave `info = r + 1` and
  `SINGULAR`, so `Newton(; linsolve = SVDFactorization())` no longer stops after one iteration with
  `SINGULAR` on an `SVector`. A host `DenseSVD` refuses a matrix with an `Inf` or `NaN` entry
  before `gesvd` and reports `info = 1` and `SINGULAR`, where an `Inf` gave `SUCCESS` with `NaN`
  singular values. A host `DenseSVD` divides every matrix by a power of two before `gesvd`, as a
  static or device SVD does, and takes its status and rank from the singular values of the scaled
  matrix. A finite matrix near `floatmax`, such as a 4×4 matrix of `floatmax(T) / 2`, gives
  `info == 0`, `SUCCESS`, rank 1 and a finite step, as an `SMatrix` and a `JLArray` do, where it
  gave `info = 1` and `SINGULAR`. The field `S` holds the singular values of the matrix itself,
  which overflow to `Inf` above `floatmax` without failing the factor. A matrix rounded to the
  subnormal range gets the same rank on the host as on the static and device routes.
  `GeometricSolvers.exact_solve(fac)` is `false` for an SVD factor of rank below `n`, and `Newton`
  then measures the slope of its step (`MeasuredSlope()`) in place of `−2φ(0)`. With `DogLeg`, a
  Newton step of zero norm is no Newton step: a zero Jacobian returns `STALLED` with the iterate
  kept. The regenerated `test/verification/standard_reference.csv` changes the code of two cells
  of the column `Newton Backtracking SVD` in `Float32`, `(3, u0)` from `STALLED` after 7 iterations
  to `LINESEARCH_FAILED` after 6 and `(8, u0)`, flagged `unstable`, from `STALLED` after 12 to
  `LINESEARCH_FAILED` after 11, and the evaluation counts of 17 more cells of that column, where
  the measured slope costs a Jacobian-vector product at the start of a line search; one of them,
  `(8, 10u0)`, flagged `unstable`, also ends at another residual norm.

- **The same-type `ldiv!!` no longer overflows on a finite right-hand side near `floatmax` in the
  `Qᴴ r` and `Uᴴ r` products (K21), or in the SVD solve (K22).** The solve formed `Qᴴ r` or
  `Uᴴ r` before it divided, so a finite `r` near `floatmax` whose solution is representable gave
  a non-finite step: `J = I + Hilbert(4)` with `b = fill(0.75 * floatmax(T), 4)` did, on the
  static, host and device routes, in `Float32` and `Float64`. `ldiv!!` now divides a finite `r`
  whose largest `|re|` or `|im|` is within `2 + ⌈log2 N⌉` bits of `floatmax` by the smallest
  power of two `2^k` that leaves that headroom, and multiplies the step by `2^(k - e)`, with `e`
  the scale exponent of the factorisation; any other `r` is solved as before. The host SVD of a
  `Float32` `J = fill(floatmax(Float32) / 2, 4, 4)` now agrees with the static and device routes.
  The scaling also applies to LU and prevents the LU overflow of this system: with
  `J = [1 0 0 0; 1 4 0 0; 0 0 1 0; 0 0 0 1]` and
  `b = (0.75 * floatmax(T), -0.75 * floatmax(T), 1, 1)`, the solve gave an all-`NaN` step without
  the scaling, because the forward substitution forms `b₂ − b₁`; with it, the step is exact, on
  the static, host and device routes. The SVD factors are those of `J / 2^e`, so the SVD solve
  formed the step times `2^(e - k)` and overflowed where `e > k`:
  with `J = diag(2^20, 2^10, 2^20, 2^20)` and `b = (0, 0.75 * floatmax(T), 0, 0)`, the solve gave
  `[NaN, Inf, NaN, NaN]` although the solution is representable. The same-type SVD solve now
  divides the coordinates `Uᴴ r` by the singular values times the smallest power of two `2^t` that
  keeps the coordinates within the same headroom below `floatmax`, and multiplies the step by
  `2^(k + t - e)`; `t` is 0 wherever the coordinates already are below that bound, and then the
  step is bit for bit the one without `t`. That was checked on 20 000 random systems per type and
  on 20 000 systems aimed at the overflow end, where every case that overflowed before also gives
  a finite step. The step of the SVD no longer overflows where it is representable, on the
  static, host and device routes. An `r` within `k` bits of the subnormal range can lose
  those bits, for LU too: with `J = I` and
  `b = (0.75 * floatmax(T), 3 * nextfloat(zero(T)), 1, 1)` the step's second entry is `0`, and
  the tests now bound that error by `2^k · nextfloat(0)`; a coordinate within `t` bits of it can
  lose those bits in the SVD's division. A solution above `floatmax` still overflows. K21 and K22
  are fixed and leave `KNOWN_ISSUES.md`; K23 is recorded there: the LU and QR solves can still
  overflow in the triangular solve for such an `r`. With `J = H diag(2^20, 2^14, 2^20, 2^20) H`,
  `H = H₄ / 2`, `H₄` the 4 × 4 Hadamard matrix, and `b = 0.75 * floatmax(T) · H[:, 2]`, they give
  `[Inf, -Inf, Inf, …]`, while the SVD solves it.

- **`StaticSVD` stores `σ̂` alone, and `GeometricSolvers.singular_values(fac)` gives the singular
  values** of the matrix as it was passed, `σ̂ · 2^e`; the fields `σ` and `converged` are gone, and
  K17 leaves `KNOWN_ISSUES.md`: the singular value above `floatmax` that overflows is now the value
  of a function, documented there.

- **The R1 LU, QR and SVD factorise a `BigFloat` static matrix** (K6 leaves `KNOWN_ISSUES.md`): their
  local arrays are `SizedMatrix` and `SizedVector` for an element type that is not `isbits`, and
  `MMatrix` and `MVector`, with no allocation, otherwise.

- **No `Float16` code** (decision 46): `jacobi_acc_type`, its uses in the device SVD and the
  precompile block of `src/linear/ka/svd.jl` are removed, and the device SVD accumulates in its own
  element type. The `Float16` SVD tests go with them; the `Float16` tests of generic code stay.

- **The dense LU calls `LAPACK.getrf!`**, and the package's own `getrf!` wrapper is removed.
  `GeometricSolvers.dense_factor` sizes the host QR and SVD workspaces by a workspace query of
  `geqrf` and `gesvd` in place of the fixed `64n` and `128n`.

- **`init`, `solve!` and `solve` are CommonSolve.jl's functions, and `solve!(init(…))` returns what
  `solve(…)` returns** (decision 49 of the design). CommonSolve is a
  new dependency (`[compat] CommonSolve = "0.2"`). The package imports and extends
  `CommonSolve.init`, `CommonSolve.solve` and `CommonSolve.solve!` and exports them as before, so a
  caller reaches this package through the same three names as another CommonSolve package. This
  holds for every iterate `solve` takes: an array, including a GPU array, an `SVector`, an
  `MVector` or a `Number`. `init(prob, x₀, method, p = nothing; options, kwargs...)` takes the
  arguments of `solve`. It holds a copy of an array `x₀`, so neither `init` nor `solve!(solver)`
  mutates `x₀`, and it holds a `StaticArray` or a `Number` `x₀` and `p` themselves.
  `solve!(solver)` and `solve!(solver, x, p = solver.p)` return `(solver.x, status)`, as `solve`
  returns `(x, status)`. `solve!(solver, x, p)` stores `x` and `p` in the fields `solver.x` and
  `solver.p` and solves from `x`: an array `x` receives the solution in place and stays in
  `solver.x`, and a value iterate is replaced in `solver.x` by the solution. It raises an
  `ArgumentError` for an `x` whose element type or length differs from the iterate given to `init`,
  and for an `x` of another type or a `p` of another type than those given to `init`, because the
  fields `x::X` and `p::Q` have the concrete types of the first ones; a refused call changes
  nothing. `NonlinearSolver` gains the type parameters `X <: Union{AbstractVector{T}, T}` and `Q`,
  and its constructor takes `p` as a sixth argument. `solve!(solver)` allocates nothing on an
  `Array` after one warm-up call.
  The batched R2 in-place solve is `solve_batch!` (exported), in two forms:
  `solve_batch!(X, S, prob, method, p = nothing; ...)` and
  `solve_batch!(X, prob, method, p = nothing; ...)`. The dense factor builders for
  `LUFactorization`, `QRFactorization` and `SVDFactorization` are `GeometricSolvers.dense_factor`,
  not exported. With these names apart from CommonSolve's, the package extends `CommonSolve.init`
  only on `NonlinearProblem` and `CommonSolve.solve!` only on `NonlinearSolver`. The R3 call form
  `solve!(x, solver, p)` is removed, with no deprecation, because the package is not registered.

- **Each nonlinear method has one state type, and the loop has two hooks** (decision 47 of the
  design). The state builder of a method is now `GeometricSolvers.init_state(method, prob, x, p)`,
  with four positional arguments, in place of `init(method, prob, x, p = nothing)`; `init` builds
  the solver (`init(prob, x, method, p; ...)`), and the linear factor is built by
  `GeometricSolvers.dense_factor`. `init_state` returns a
  `NewtonState` or a `PicardState`, not a `NamedTuple`, and for an `SVector` or a `Number` it is
  `isbits` and allocates nothing. `Newton`'s reuse age and stalled flag, and `DogLeg`'s radius
  and its initial radius, are its field `rule`: `state.rule.age`, `state.rule.radius`.
  `solve!!` calls `start!!(state, method, prob, x, Fx, p, fnorm₀)` once before its stopping test
  at 0 iterations, and `finish!!(state, method, prob, x, Fx, status)` at every return, and
  returns what `finish!!` returns; a method's own `residual_buffer(state)` names the slot of
  `F(x₀)`. `Newton`'s `start!!` resets its rule state, which `solve!!` did by field name before.
  These four names are internal and documented beside `step!!`. Every `Newton` step factorises
  through one function, and the iterates of the `Newton` tests are unchanged.

- **The static Jacobi sweep updates its column norms from each rotation** (audit F12). The squared
  norms of all columns are read once at the start of a sweep, and a rotation of a pair sets the
  pair's two norms from its Gram matrix and the rotation, `α - t|γ|` and `β + t|γ|`, where every
  pair read both columns again; a norm below `√eps(R)` times its old value is read from its column
  again. On the 112 matrices of the new testset of `test/linear/static/svd.jl`, no factorisation
  takes more sweeps than the recompute form, and the singular values agree with it within
  `4n · eps(R) · σ₁`.

- **A line-search step of `Newton` and `Picard` reduces the residual no more to get its merit at
  `α = 0`** (audit F13). `NewtonState` and `PicardState` have a ninth and a fifth field, `merit`:
  `norm2(Fx)` of the current residual for a line search, which `start!!` sets once per solve and
  each step sets from the merit that its search computed for the residual it returns; `nothing`
  for `Static`, `DogLeg` and any other globalization. `init_state` and `finish!!` set it to `NaN`,
  so a line-search `step!!` on a state that no `start!!` saw, or that `solve!!` returned, returns
  `NONFINITE` and keeps its iterate. A solve on an `Array` with `Backtracking`, `Bisection` or
  `StrongWolfe` calls `norm2` on its residual buffer at most twice, where it called it once per step
  and once more. The public `globalize!!(g, ls, prob, x, Fx, d, p, jvpprep, stepkind)` keeps its
  form and its results; the solvers call the internal `globalize_merit!!`.

- **`Backtracking`, `Bisection` and `StrongWolfe` loop while they have no result**, for at most
  `maxiter` trips, in place of `maxiter` trips with a done flag (audit F14). They evaluate the same
  trials and return the same results, and no cell of `test/verification/standard_reference.csv`
  changes. `scripts/r2_throughput.jl` reports a row of `Newton(; globalization = Backtracking())`,
  which runs a line search in the kernel.

- **A step test that holds while the residual test fails returns `STALLED`, not `SUCCESS`**
  (decision 54). `SUCCESS` now always means that the residual test holds. A `SUCCESS` step of
  zero norm, which leaves the iterate where the next step starts, also returns `STALLED` at that
  iteration: `solve(NonlinearProblem((x, p) -> x * x - 4, (x, p) -> Inf), 3.0, Newton())`
  returns `STALLED` after 1 iteration at `x = 3`, where it returned `SUCCESS` (K13 leaves
  `KNOWN_ISSUES.md`). `min_iterations` bounds both rules, and `converged` holds through the step
  test only after a step of nonzero norm. No cell of the standard test set changes: the
  regenerated `test/verification/standard_reference.csv` equals the old one.

- **A solve takes only an `Options{R}` for `R = real(eltype(x₀))`, and raises an `ArgumentError`
  for any other.** `solve`, `init`, both batched `solve_batch!` methods and `solve!!` check it; there is
  no `convert` between `Options` of different `R`. The `Float64` defaults `f_reltol = √eps(Float64)`
  and `x_reltol = 2 eps(Float64)` lie below `eps(Float32)`, so a `Float32` solve could not reach
  them: with `Options(Float64)` converted, `x² − 2` from 1000 starts in `[1, 3]` returned `STALLED`
  from 366, and with `Options(Float32)` it returned `SUCCESS` from all 1000. A `Float32` problem
  takes `Options(Float32; …)` or keyword tolerances. `solve`, `init` and both batched `solve_batch!`
  methods resolve `options` and keywords in one function, with one message when both are passed:
  "pass either `options` or keyword arguments, not both".

- The batched `solve_batch!` runs one kernel for both forms, waits for it with
  `isgpu(backend) && synchronize(backend)` as the KA factorisations do, counts the return codes
  over `instances(ReturnCode)`, and calls `get_backend` on an `order` without catching its error:
  an `order` whose backend cannot be read raises, where it fell back to the CPU, and an `order` on a
  device with a batch on the host is copied to the host. The range method of `_adapt_order`,
  which the `isbits` branch already covered, is gone (K18 leaves `KNOWN_ISSUES.md`).

- `Picard` scales its direction by a damping in `real(T)`, so a complex problem gets a real
  damping (decision 24): one step with `damping = 0.5` equals `x .- 0.5 .* F` bit for bit, so a
  real part `-0.0` of `x` stays `-0.0` where the real part of `F` is `+0.0`.

- `Pkg.test()` no longer runs the doctests. `test/quality/doctests.jl` is now the group
  `doctests`, which an empty `ARGS` does not run; `Pkg.test(test_args = ["doctests"])` runs it.
  In CI the Doctests job stays their runner, so the test matrix no longer runs them a second
  time.

- CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of `Julia min`, and a test
  job saves the Julia cache only when it succeeds.

- Raised lower bounds, so that the CI Downgrade job, which resolves `Project.toml` and
  `test/Project.toml` together at their lower bounds on Julia 1.13 and runs the tests there,
  reaches every floor and passes. `KernelAbstractions` goes from 0.9.15 to 0.9.38: the test
  dependency Metal.jl 1.10.3 requires KernelAbstractions 0.9.38 or later. In `test/Project.toml`,
  `JET` goes from `0.9, 0.10, 0.11, 0.12` to `0.12`: JET 0.9 requires Julia 1.10, and JET 0.10
  and 0.11 require Julia 1.12. `Aqua` goes from 0.8 to 0.8.14: 0.8.14 is the first release whose
  piracy check reads `Core.methodtable`, the method table of Julia 1.13; Aqua 0.8.0 fails there on
  the missing field `TypeName.mt`. `Documenter`, with no entry before, gets `1`: without one the
  merged resolve takes its oldest release.

- `test/backends.jl` is now `test/integration/backends.jl`. The test convention keeps a test file
  at the top level of `test/` only where it mirrors `src/<name>.jl`, and this file tests no source
  file: it checks the array backends and element-type sets of `test/helpers/matrix.jl`. It stays
  in the `core` group under the label "Backends", and asserts the same as before.

- The `lu!` cell of `scripts/spikes/capabilities/run.jl` reads the raw factors and pivots back
  and extracts L, U and p on the host. On a `ROCArray`, `F.L` and `F.p` scalar-indexed, so the
  cell reported the read-back and not the factorisation, in every element type that `lu!`
  supports there. On an `Array` the extracted L, U and p are `==` to those of the factorisation.

- `test/gpu/README.md` step 5 no longer says to instantiate both environments again when the
  spike's and the vendor's manifests differ: `Pkg.instantiate()` does not change a manifest. It
  says to run `Pkg.update()` in both, and, where the vendor package holds a shared package at an
  older version (AMDGPU.jl 2.8.0 holds GPUArrays at 11), to resolve the spike's packages and the
  vendor package together in one environment outside the clone.

- **Julia 1.13 is the floor** (`[compat] julia = "1.13"`, from 1.11). A breaking change under
  this package's pre-1.0 convention: a caller on Julia 1.11 can no longer install this version.
  The eight other environments of the repository (`test/gpu/cuda`, `test/gpu/metal`,
  `test/gpu/rocm`, `test/enzyme`, `scripts/spikes/nextla`, `scripts/spikes/capabilities`,
  `scripts/spikes/kernel_newton` and `scripts/spikes/metal_qr`) state the same floor. The reason
  is Metal: with Metal.jl 1.11.1, static `QRFactorization` in `Float32` misses its accuracy bound
  on Julia 1.12.7 for every fill of the scaled subcolumn measured that keeps the factorisation
  free of heap allocation on the host (the previous fill at `n = 7, 8`; this release's fill at
  `n = 3:9`). This release's fill passes at every `n` in `1:12` on Julia 1.13.2.
  `scripts/spikes/metal_qr/README.md` compares five fills. The allocation tests of
  `test/base/bang.jl`, `test/base/reductions.jl`, `test/ad/jacobian.jl`,
  `test/ad/forwarddiff_static.jl`, `test/linear/static/lu.jl` and
  `test/linear/static/qr.jl` no longer go through a function barrier and a warm-up call: on Julia
  1.12.7 a bare `@allocated` at testset scope reads the same figure as the barrier in every case
  measured (zero for each call asserted to allocate nothing, a positive figure for each `collect`
  control), and it does not count the compilation of a first call. The tolerances on the product
  of static matrices in `test/base/bang.jl` stay: on Julia 1.12.7 on aarch64 the seam kernel's
  product and the closure's still differ by up to 2 ulp in `ComplexF32`, and an x86_64 run, which
  CI includes, is not measured. Both sets of figures are from Julia 1.12.7, now below the floor,
  and no run on 1.13 measured either the allocation reading or the 2 ulp difference, so the
  tolerances stay.

- Static `QRFactorization` (`factorize!!` in `src/linear/static/qr.jl`) fills the scaled subcolumn
  buffer `x` in one loop over `1:N`: `x[i]` is `A[i, k] / amax` for `i >= k` when `amax` is
  nonzero, and zero otherwise. The previous code zeros `x` and fills only `k:N`, under
  `if !iszero(amax)`. On Metal (Metal.jl 1.11.1) this repairs `Float32` QR on Julia 1.13. In a
  sweep over `n = 1:12` of `conditioned_systems(1000n + 1, Float32, Nothing, n)`, a system counts
  as failed if it misses the bound of check 1 or returns `info ≠ 0`: the previous fill fails at
  `n = 3:9`, and this fill fails at no `n`. In `test/devices/metal_linear.jl` on Julia 1.13.2, the
  previous fill gives 139 passed and 5 failed, all in `QRFactorization on Metal, Float32, n = 4`,
  and this fill gives 144 passed. With this fill the sweep has no failing `n` for QR or LU, in
  `Float32` or `ComplexF32`. This closes `K9` in `KNOWN_ISSUES.md`. The cause is not located: no
  experiment separates Metal.jl's lowering from Apple's Metal compiler, and no upstream report
  exists. This change leaves `K10` and `K11`, which entries below close. `K10`: the Metal CI
  job runs with `--check-bounds=yes`, which hides this failure; with that flag the previous fill
  passes every `n` on Julia 1.13.2. `K11`: the Metal test sizes 1, 4, 16 and 24 reach only
  `n = 4` of the failing sizes.
  `scripts/spikes/metal_qr/` holds the 4 × 4 reproducer (`probe.jl`), the sweep (`sweep.jl`) and
  `README.md`, which compares five fills.
  On the host, the factors, `τ` and `info` are bitwise identical to the previous fill on 64 random
  matrices per element type (`Float64`, `Float32`, `ComplexF64`) and size (`N` in 2, 4, 8 and 16),
  and both fills allocate nothing. The median time per factorisation, this fill relative to the
  previous one on Julia 1.13.2 (two cold runs, load average 9 to 12), is 1.81 to 1.86 for `Float64`
  at `N = 2`, 1.95 for `Float32` at `N = 2`, 1.19 to 1.22 for `Float32` at `N = 4`, and 0.99 to
  1.05 for every other pair measured. So a caller of static QR at `N = 2` in `Float64` or
  `Float32`, or at `N = 4` in `Float32`, should expect the factorisation to take 1.2 to 2 times as
  long. Only `N` in 2, 4, 8 and 16 was measured.

- The Metal device tests in `test/devices/` (`metal.jl`, `metal_linear.jl`, `metal_svd.jl`,
  `metal_nonlinear.jl` and `metal_ka.jl`) skip on a virtualized GPU and run on a real Apple GPU.
  `test/helpers/metal_device.jl` detects a virtual device with `Metal.MTL.is_virtual`, and each
  skip is recorded as `@test_skip` against issue #34. The paravirtual GPU of the GitHub macOS
  runners throws a device exception at random in a kernel late in the Metal group; which kernel
  throws depends on the test order, and each one passes when it runs early and on a real GPU, and
  its algorithm passes on the CPU. Until issue #34 closes, the Metal workflow and the macOS-latest
  CI jobs run no device test, so a green macOS-latest check does not exercise Metal.
  `.github/workflows/Metal.yml` and `test/gpu/README.md` say so.

- `test/quality/jet.jl` fails on a JET report in this package's own code. It runs
  `JET.report_package` with `target_modules = (GeometricSolvers,)`, with no `try` around it, and
  compares the reports with `KNOWN_FALSE_POSITIVES`: five reports, each named by the function of
  its innermost frame and a fragment of JET's message, with its reason. Four are the launch of a
  kernel and `synchronize` on a KernelAbstractions `GPU` backend, which the backend package
  defines; one is a `nothing` status that the one caller of `ka_factorize_lu!` never passes. A
  report outside the list, or an entry that names more than one report, fails the file; so does
  an entry that names none, unless it is a launch or a `synchronize`, whose method a backend
  package loaded earlier in the same process (JLArrays, Metal) can define. A call of an
  undefined function added to `rnorm` or to `isconverged` now fails it. A JET that fails to load
  fails the file, on a pre-release Julia too. This closes `K1` in
  `KNOWN_ISSUES.md`.

- The test environment sets Metal's `precompile` preference to `false` in `test/Project.toml`,
  and `test/devices/metal.jl` asserts that the test process reads it. Metal.jl 1.11.1 then
  precompiles where `Metal.devices()` is empty, as in a sandbox, and the Metal files record their
  device skip; with the preference `true` and no Metal cache for it, the sandboxed run fails with
  `Failed to precompile Metal`. `K2` stays in `KNOWN_ISSUES.md`, worked around: the upstream fix,
  Metal.jl 1.12.0, needs KernelAbstractions 0.10.

- `test/devices/metal_linear.jl` runs `QRFactorization` in `Float32` and `ComplexF32` at every
  `n` in `1:12` beside `STATIC_SIZES`, because other fills of `x` in the static QR fail on Metal
  at sizes in `3:10`. This closes `K11` in `KNOWN_ISSUES.md`. The file skips on a virtual GPU and
  in a sandbox; it runs by hand on a real Apple GPU.

- `K10` and `K12` leave `KNOWN_ISSUES.md`. `K10`: since the Metal device tests skip on a
  virtualized GPU, the Metal CI job runs no device test, so its `--check-bounds=yes` hides no
  device result; run 41114546419 on `main` records one `Broken` and no other test in each of the
  five files of the `metal` group, in both jobs. `K12`: a cold `run-tests.jl . core` passes
  213741 assertions in 265 s on Julia 1.13.3, and its slowest file,
  `globalization/linesearch.jl`, takes 44.5 s, under the 60 s budget of a `core` file.

- `rnorm` scales the entries by the power of two of the largest `|a_i|` (of the largest real or
  imaginary part for a complex entry) before it squares them, as proposed in #29. It no longer
  overflows to `Inf` above `sqrt(floatmax(R))` (`rnorm(SVector(1f20, 0f0))` is `1f20`, not `Inf`)
  nor underflows to `0` below `sqrt(floatmin(R))`. On the tested vectors, whose largest entry is
  `floatmax(R) / 4`, `1`, `floatmin(R)` or subnormal, it agrees with `LinearAlgebra.norm` to
  `2 eps(R)`. Where `norm2(a)` is finite and at least `length(a) * floatmin(R)`, `rnorm` is
  `sqrt(norm2(a))`, one reduction: at that size the subnormal squares lose at most `eps(R)` of
  the sum. Only otherwise does it take the scaled path, with two more reductions, and that path
  sums in at least `Float32`, so a `Float16` sum of scaled squares does not overflow from about
  8200 complex or 16400 real entries on: `rnorm(fill(Float16(1.99), 20000))` was `Inf16` and is
  `281.5`. `norm2`, the merit `‖F‖²` of the line searches, stays unscaled, so a line search above
  `‖F‖ = sqrt(floatmax(R))` still returns `NONFINITE`, and its `NONFINITE` result reports
  `rnorm(Fx)`, finite, where it reported `sqrt(φ₀) = Inf`. Its `STALLED` result, and a
  `LINESEARCH_FAILED` that keeps no step, report `rnorm(Fx)` too. At a finite `φ₀` this is
  `sqrt(φ₀)` exactly, unless `φ₀ < length(Fx) * floatmin(R)`; there `rnorm(Fx)` takes the scaled
  path, and the two can differ by more than the last bits, as `φ₀` sums subnormal squares. Where
  `real(eltype(a))` is no `AbstractFloat`, such as an integer or a `ForwardDiff.Dual`, `rnorm`
  stays the unscaled `sqrt(norm2(a))`.

- `K14` and `K16` leave `KNOWN_ISSUES.md`: with the scaled norm, `DogLeg` in `Float32` solves the
  Rosenbrock system scaled by `1e18` to `3e18` to its root, and stops on `s · (2x₁ − 1, −1)` at
  `s = 1e19` at the minimum of `‖F‖`, `(0.5, 0)`, after 3 iterations as at `s = 1e17`.
  `test/globalization/dogleg.jl` runs both. A `Newton` solve with `Static()` or `DogLeg()` no
  longer returns `NONFINITE` only because `‖F‖` lies between `sqrt(floatmax(R))` and
  `floatmax(R)`: `1f20 · (y² − q)` from `(3, 5)` now converges. The tests that asserted the
  overflow and the underflow of the unscaled norm assert the scaled values.

- `test/verification/standard_reference.csv` is regenerated. 14 cells change, each `NONFINITE`
  before, where an unscaled norm left the range: in `Float32`, problem 1 with `Static` LU
  from `u0` and `10u0` ends `SINGULAR`, and from `100u0` with `Backtracking`, `Bisection`,
  `StrongWolfe` and reuse-3 `Backtracking` LU and problem 12 from `100u0` with `Backtracking` QR
  end `LINESEARCH_FAILED`; problems 8 (`u0`, `100u0`) and 18 (`10u0`, `100u0`) with `Static` LU
  end `SUCCESS`. In `Float64`, problem 1 with `Static` LU stays `NONFINITE` after more
  iterations. 138 cells are flagged `unstable`, 133 before.

### New Features

- **A `SciMLBase` package extension (`GeometricSolversSciMLBaseExt`) accepts a
  `SciMLBase.NonlinearProblem` in `solve` and `init`, and returns `(x, status)`** (decision 52 of
  the design). `SciMLBase` is a weak dependency (`[compat] SciMLBase = "3.55"`), so the package and
  its main test suite load without it. When `SciMLBase` is loaded,
  `solve(prob::SciMLBase.NonlinearProblem, method::Union{Newton, Picard}; kwargs...)` translates
  `prob` to a native `NonlinearProblem` and returns `(x, status)`, and `init` with the same
  arguments translates `prob` by the same rules and returns a `NonlinearSolver`, so
  `solve!(init(prob, method; kwargs...))` returns what `solve(prob, method; kwargs...)` returns and
  the native `solve!` serves the solver. An out-of-place `f(u, p)` with a `StaticArray` or a
  `Number` `u0` solves in R1; an in-place `f(du, u, p)` with an array `u0` that is not a
  `StaticArray` solves in R3 on a copy of `u0`; an out-of-place `f(u, p)` with such an array `u0`
  solves in R3 through `(y, x, p) -> (y .= f(x, p))` (and the same wrapper for `prob.f.jac`); and
  an in-place `f` with a `StaticArray` or a `Number` `u0`, or a problem with `jac_prototype`,
  `sparsity`, `lb` or `ub` set, raises an `ArgumentError`. `prob.p` reaches `f` and `jac`
  unchanged, `SciMLBase.NullParameters()` included. `prob.kwargs` and the call's keywords merge
  with the call's winning over the closed set `abstol` (`f_abstol`, with `f_reltol = 0` when
  `reltol` is absent), `reltol` (`f_reltol`), `maxiters` (`max_iterations`) and `options`; any
  other keyword raises an `ArgumentError`. `GeometricSolvers.sciml_retcode(code::ReturnCode)`
  (`public`, not exported, with its method in the extension) maps the six `ReturnCode` values to
  `SciMLBase.ReturnCode.Success`, `Stalled`, `MaxIters`, `InternalLinearSolveFailed`, `Unstable`
  and `InternalLineSearchFailed` (`sciml_retcode(status.code)`). `test/sciml/` and
  `.github/workflows/SciML.yml` test the extension in its own environment.

- The empty package skeleton: `Project.toml`, the module and its
  docstring, a test harness with Aqua and a diagnostic JET check, and `test/backends.jl` — the
  array backends a test loops over (`Array`, `JLArray`, and the KernelAbstractions `CPU()`
  backend). `scripts/float16_dot.jl` measures the accuracy of five `Float16` dot-product
  accumulation policies (naive, Neumaier compensated summation, Ogita–Rump–Oishi `Dot2`,
  `Base.TwicePrecision` and a `Float32` accumulator) against a `BigFloat` reference.

- `scripts/spikes/capabilities/run.jl` is a capability census: for each element type (Float16,
  BFloat16, Float32, Float64, ComplexF32, ComplexF64) and dense linear-algebra operation (GEMM,
  `lu!`, `qr!`, `svd!`, `A \ b`, `cholesky`, batched LU), it reports whether the backend array
  supports it — pass, wrong (tolerance: `8 * eps(real(T))` for Float16/BFloat16,
  `100 * eps(real(T))` for others), unsupported, n/a, or the first line of the thrown error
  (tagged with the step: `to device`, `compute`, or `read back`) — and checks
  a KernelAbstractions kernel doing `BFloat16` arithmetic and
  `DifferentiationInterface`'s `jacobian!` with `AutoForwardDiff()` on the backend's native array
  type and on a `JLArray`. Runs on `cpu` and `metal` backends.

- `scripts/spikes/kernel_newton/` measures the R1 design of part E on each backend: a Newton
  iteration on `SVector` systems with no line search, and a partially pivoted LU with `Int32`
  pivots and the analytic Jacobian, in two storage variants (a local `MMatrix` mutated in place,
  and functional updates on an `SMatrix`), against the SimpleNonlinearSolve baseline. One
  KernelAbstractions kernel runs one system per thread and takes the solver as an argument; the CPU
  rows are a threaded host loop over the same code. `run.jl <backend> [variant] [tmode] [n]` writes
  one `ROW` line per backend, variant, `tmode` and `n`, with the seven columns of the design's
  §10.E check 3 and the raw evidence; compile time is measured in a fresh child process for each
  row. `verify.jl <results dir> <card>` re-derives every cell from the raw output and applies the
  band and storage decision rules. The band counts the own R1 variants, `M` and `S`, not the SNS
  baseline; `verify.jl` notes the first n where the CPU loop beats SNS. `verify.jl --self-check`
  tests the band rule against a comparator mutant and against an SNS row that the band must not
  count. `lu_check.jl` checks the pivoting of both LU solves on random matrices that
  pivot. A time cap bounds the throughput cells: before a cell runs, the median time of the cell
  before it projects the cell's wall time, from a warm launch and five timed launches. A cell whose
  projection passes 30 minutes is not run, nor is any larger cell, and it reads
  `not run: time cap`; the 1e4 cell always runs. Each row records `time_cap_s`. A row without it
  comes from a run without the cap, and `verify.jl` keeps a timing past the cap in such a row and
  notes it; in a row with `time_cap_s`, a timing past the cap fails, and so does a timing after a
  time-cap mark. The CPU baseline runs one Julia thread per core (`--threads=auto`): `run.jl cpu`
  stops with an error in a single-threaded Julia, and `verify.jl` fails a CPU row that records one
  thread.

- The core types, all `isbits` so that a kernel can hold them, with every real field in
  `R = real(T)`: `ReturnCode` (one byte: `SUCCESS`, `STALLED`, `MAXITERS`, `SINGULAR`,
  `NONFINITE`, `LINESEARCH_FAILED`); `SolverStatus{R}`, with `step_failures` counting every
  step-rule failure and `promoted` recording a factorisation above its method's precision;
  `StepInfo{R}`, which `GeometricSolvers.record` folds into a status; and `Options{R}`, the
  stopping test only, built by `Options(T; kwargs...)`, which a solve of another element type
  rejects with an `ArgumentError`.
  The default residual test is relative (`f_reltol = √eps(R)`, `f_abstol = 0`), because the
  attainable residual depends on the scale of the problem, and `min_iterations` is `0`, so a
  start that already passes the test takes no step.
- The line searches `Static`, `Backtracking` (Armijo), `Bisection` and `StrongWolfe` are `isbits`
  types running in a KernelAbstractions kernel and on the host, allocating nothing and never
  throwing once built; constructors check parameters and raise `ArgumentError`. Each reports
  through a `ReturnCode` (`SUCCESS`, `STALLED`, `NONFINITE`, `LINESEARCH_FAILED`), not through a
  log. Every step is positive and within the caller's ceiling. `StrongWolfe` reports `SUCCESS`
  only for steps meeting both strong Wolfe conditions. The four method types are exported;
  the function that runs a search is internal; `Quadratic`
  and `BierlaireQuadratic` of SimpleSolvers are not ported.
- `LUFactorization`, `QRFactorization` and `SVDFactorization` carry their factorisation precision
  as a type parameter: `LUFactorization()` for the working type, `LUFactorization(Float32)` for a
  `Float32` factorisation, and the markers `TF32()`, `FP16()`, `BF16()` for vendor tensor cores.
- `LUFactorization` and `QRFactorization` solve a static system. The internal
  `GeometricSolvers.factorize!!(nothing, method, J)` and `GeometricSolvers.ldiv!!(nothing, fac, r)`
  take an `SMatrix` or `MMatrix` and return `isbits` factors and an `SVector` step, so both run
  in a KernelAbstractions kernel, one system per thread, and allocate nothing unless bounds checks
  are forced (`--check-bounds=yes`). LU uses partial pivoting; QR is Householder QR without column
  pivoting, and it is not rank-revealing. A singular matrix does not throw:
  `GeometricSolvers.returncode(fac)` gives `SINGULAR` when `info` holds the first zero pivot (LU)
  or the first zero diagonal entry of `R` (QR). A singular matrix whose `R` has no exact zero gives
  `SUCCESS`, and two equal rows, as a rule, give no exact zero. Two equal rows give `SINGULAR` in
  LU in every element type, also at n = 2 with entries near `floatmax`. The factorisation checks no
  entry for finiteness: a `NaN` gives a non-finite step or `SINGULAR`, and an `Inf` gives what the
  elimination gives, which for a real `Inf` pivot is the finite limit, as in LAPACK. With a
  factorisation precision below the working type, as `LUFactorization(Float32)` for a `Float64`
  system, the matrix is first divided by a power of two that brings its largest entry near 1, which
  is exact, and the factors are in `Float32` (`ComplexF32` for a complex system). A matrix of scale
  `1e-50` or `1e50` therefore factorises and solves to the `Float32` bound, though entries far below
  the largest can still underflow. The right-hand side is scaled by its largest absolute entry
  before the conversion, so a right-hand side of size `1e-40` or `1e40` solves to the `Float32`
  bound too. QR takes its column norm as LAPACK's `nrm2` does, so an entry below `sqrt(floatmin)`
  or above `sqrt(floatmax)` does not give a wrong step. A precision marker such as `TF32()` has no
  method for a static matrix. `StaticArrays` is a new dependency, at 1.9.8 or later.

- `SVDFactorization` on a static matrix: `GeometricSolvers.factorize!!(nothing, SVDFactorization(), J)`
  and `GeometricSolvers.ldiv!!(nothing, fac, r)` return an `isbits` `StaticSVD` and an `SVector`
  step, from a one-sided Jacobi iteration run by one thread per system in a KernelAbstractions
  kernel, and allocate nothing unless bounds checks are forced (`--check-bounds=yes`).
  `GeometricSolvers.singular_values(fac)` gives the singular values of `J`, in non-increasing
  order. A singular value at or below `√n · eps(TF) · σ₁` counts as zero: `rank` counts the
  others, and the solve leaves those directions out, so a rank-deficient matrix gives a finite
  truncated step and never a `NaN`. The matrix is first divided by a power of two, recorded as
  `scale_exponent`, so a subnormal matrix gives finite factors. A sweep that reaches the cap of 30
  sweeps, or a singular value that is not finite, gives `info = 1`, and
  `GeometricSolvers.returncode(fac)` gives `SINGULAR`.

- `jacobi_eigen(A)`, exported, is a cyclic two-sided Jacobi eigensolver for a Hermitian or real
  symmetric static matrix, on the rotations and round-robin ordering of the SVD, with a
  Frobenius-norm pair floor `eps(R) * ‖H‖_F`. The matrix is first divided by a power of two that
  brings its largest `|re|` or `|im|` into `[1, 2)`, and the eigenvalues are multiplied back, so
  a matrix with entries near `floatmax` or in the subnormal range converges, with unitary vectors
  and eigenvalues to working precision wherever they are representable. It returns
  `(values, vectors, status)`: the real eigenvalues, the unitary `vectors` with
  `A = V * Diagonal(values) * V'`, and `status`, which is `SUCCESS`, or `MAXITERS` at the cap of
  30 sweeps. It is `isbits` and allocates nothing unless bounds checks are forced, so it runs in
  a kernel. For a real skew `M`, `jacobi_eigen(im * M)` gives `exp(M)` as
  `real(V * Diagonal(exp.(-im .* values)) * V')`, to working precision. It is a function on a
  matrix and adds no method to `LinearAlgebra.eigen`.

- Internal: the one reduction seam (`norm2`, `rnorm`, `rdot`, each returning `real(T)` on an
  `Array`, a GPU array and an `SVector` in a kernel), and the `ToReal{R}` adaptor, which converts
  every float of a method tree to `R` through `Adapt`.
- `Adapt` is the first runtime dependency. `LAPACK.getrf!(A, ipiv)` with preallocated pivots
  needs Julia 1.11 or later.
- Device runs by hand, as no runner has a CUDA or ROCm GPU: `test/gpu/cuda`, `test/gpu/rocm` and
  `test/gpu/metal` are one environment per vendor, and `test/gpu/runtests.jl <backend>` runs a
  KernelAbstractions kernel on the device in `Float32` and `Float16` (and `Float64` on CUDA and
  ROCm) against a host reference. `test/gpu/README.md` gives the procedure for a run on a machine
  and for pushing its output to a `results/<machine>` branch. The spike environments have no
  vendor package: a GPU run of `scripts/spikes/capabilities/run.jl` stacks `test/gpu/<backend>`
  behind the spike's environment through `JULIA_LOAD_PATH`.

- The test suite follows the shared layout. The test dependencies move from `[extras]` and
  `[targets]` to `test/Project.toml`, with their bounds; `Adapt`, which the tests use directly,
  takes its bound from `Project.toml` alone, and `Documenter` is new. `test/runtests.jl` selects
  the groups `core`, `slow` and `metal` from `ARGS`, and the next entry gives the groups of a run
  with empty `ARGS`. `test/aqua_tests.jl` and `test/jet_tests.jl` are renamed to
  `test/quality/aqua.jl` and `test/quality/jet.jl`, the line-function fixture moves to
  `test/helpers/linefunctions.jl`, and the new `test/quality/doctests.jl` in `slow` runs the
  docstring doctests. No file of `test/gpu/` moves.
- A default test run on an Apple-silicon Mac runs the Metal device tests. When `ARGS` is empty,
  `test/runtests.jl` runs the groups `core`, `slow` and `metal` on an Apple-silicon Mac and `core`
  and `slow` everywhere else. The new `test/devices/metal.jl`, in the `metal` group, includes
  `test/gpu/runtests.jl` and runs its Metal case, the `axpy` kernel in `Float32` and `Float16`;
  `test/gpu/` stays the separate suite for a run by hand, and `test/gpu/README.md` and the header
  of `test/gpu/runtests.jl` say that the Metal case alone has device CI. Where
  `Metal.functional()` is `false`, as inside a sandbox, the file runs no test and records one skip
  (`@test_skip Metal.functional()`), also under `test_args = ["metal"]`. Metal (1.10 or later) is
  a test dependency on every platform; it installs on Linux and Windows, and no default run there
  loads it. A new `Metal` workflow runs the group on GitHub's `macos-15` runner, for the `min` and
  `1` Julia versions, and a step before the tests fails that job where Metal is not functional. The
  job is not a required check.
- The `Adapt` floor is 4.6.1. Metal 1.10 and later, and the GPUArrays 11.5.6 and later that they
  require, need Adapt 4.6.1, so a test environment with `Metal = "1.10"` does not resolve with
  Adapt held below it.
- The element types and array backends a test loops over are named once, in the new
  `test/helpers/matrix.jl`: `REAL_ELTYPES`, `ELTYPES`, `ARRAY_BACKENDS` and `KA_BACKEND`. Every
  test file under `test/base/`, `test/linear/` and `test/globalization/`, and `test/backends.jl`,
  loops over those names instead of writing its own tuple, and the new `test/quality/matrix.jl`
  in the `core` group fails with `file:line` for a literal tuple or vector of two or more element
  types anywhere under `test/`, outside `test/helpers/` and the separate suite `test/gpu/`. No
  package code changes, and no test file changes what it asserts: `test/backends.jl` gains testsets
  of its own for the named sets, beside the ones it had.
- The R3 Jacobian seam, internal: `GeometricSolvers.prepare_ad(backend, prob, r, x, p)` chooses the
  AD path once, and `jacobian!!(J, prep, prob, x, p)` and `jvp!!(Jv, prep, prob, x, v, p)` dispatch
  on what it returned, so the solver takes no `ad` keyword and makes no runtime choice.
  `AutoForwardDiff()` goes through an own chunked forward mode on every array type, `Array`
  included (`ChunkedForwardDiff`: four `Dual` buffers allocated once, seeded and read back by
  broadcasts, `N` Jacobian columns per residual evaluation), because `DifferentiationInterface`'s
  `jacobian!` raises a scalar-indexing error on an `MtlArray`, a `CuArray` and a `JLArray`. A
  complex iterate with a holomorphic residual is differentiated in its complex argument on that
  path. Any other back end goes through `DifferentiationInterface` on a CPU iterate (`DIJacobian`: a
  prepared `jacobian!` and `pushforward!`, with the parameters wrapped in a `Constant` context on
  every call), and on a device array iterate it raises `ArgumentError` at preparation. For these
  back ends the choice depends on the iterate alone, not on the residual buffer. A non-vector
  iterate raises `ArgumentError` on both paths. A replacement parameter object may be of any type on
  the chunked path; on the `DifferentiationInterface` path a parameter of another type raises
  `DifferentiationInterface.PreparationMismatchError`, which names both types. The chunked mode tags
  its duals with the caller's tag where the caller named one, and with this solver's own otherwise,
  never `Nothing`, so a residual that differentiates inside its own body is not confused. `jvp!!` is
  one pushforward on either path, with exactly one residual evaluation per call. On an `Array` the
  chunked mode agrees exactly with `ForwardDiff.jacobian`, and its `jacobian!!` and `jvp!!` allocate
  nothing, at a chunk size that covers the iterate and at one that does not; the
  `DifferentiationInterface` path is checked against that within a tolerance, and its allocations
  are not measured. An empty iterate writes nothing on either path. `ADTypes`,
  `DifferentiationInterface`, `ForwardDiff` and `GPUArraysCore` are new dependencies.
- `test/gpu/runtests.jl` runs the R3 AD checks on the device as well (`test/gpu/ad.jl`), in
  `Float32`, plus `Float64` on CUDA and ROCm.
- `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))` through
  `DifferentiationInterface` is tested in an environment of its own, `test/enzyme/`, by the new
  `Enzyme` workflow, which is not a required status check: an Enzyme break then fails that job
  alone and blocks no merge. Runtime activity is what a `Constant` parameter context needs from
  Enzyme's forward mode; a plain `AutoEnzyme()` raises `EnzymeRuntimeActivityError`.

- `NonlinearProblem(F)` and `NonlinearProblem(F, J)`, exported, hold the residual `F` of a nonlinear
  system and, optionally, its derivative `J`. A problem holds functions only, with no parameters
  and no iterate, so it is `isbits` when `F` and `J` are, and one problem serves every call that
  passes other parameters. The value method of `residual!!` calls `F(x, p)` for every iterate it is
  handed — an `SVector`, an `MVector`, a `Number`, and an `Array` with the value slot `nothing` —
  because the iterate's type is read nowhere: the dispatch is on the storage slot and on nothing
  else.

- Internal: the storage seam, the functions an algorithm calls on its iterate, its residual, its
  update and its factorisation: `residual!!`, `update_iterate!!`, `map!!`, `mul!!`, `buffer`,
  `factorize!!` and `ldiv!!`. Each has a value method for a `StaticArray` or a `Number` iterate, so
  an algorithm written against them is written once for both storages. A `Number` Jacobian is
  factorised as a `1 × 1` matrix.

- Internal: `jacobian!!` and `jvp!!` for a `StaticVector` or a `Number` iterate. With the
  source `nothing`, `jacobian!!` returns the problem's `J` and `jvp!!` returns `J(x, p) * v`; with
  `AutoForwardDiff()`, `jacobian!!` differentiates the residual with ForwardDiff, and `jvp!!` takes
  one dual evaluation of it. The R1 direction has the element type and the length of the iterate,
  as the line-search slope does: a direction of another element type is a host `MethodError`. The
  ForwardDiff path allocates nothing. A method's own operator is a `JacobianFunction`, which
  `jacobian!!` calls in place of `J`, and `jvp!!` never reads it.

- `Newton`, `Picard`, `NoRefinement` and `solve` (exported), and `solve!!` (`public`), together with
  the internal `init`, `step!!`, `refine!!`, `LineFunction` and `globalize!!`, run the nonlinear
  solver loop in R1 (`StaticVector`, `MVector` or `Number` iterate) on the host and inside a
  KernelAbstractions kernel, one system per thread. `solve(prob, x₀, method, p = nothing; options,
  kwargs...)` adapts `method` to `R = real(eltype(x₀))` through `ToReal{R}()`, builds `state` with
  `init(method_R, prob, x₀, p)`, runs `solve!!(state, prob, x₀, method_R, p, opt)`, and returns
  `(x, status)`. Passing both `options` and keyword tolerances raises `ArgumentError`;
  `solve!!` returns `(x, status, state)`. In R1 `init` returns an `isbits` named tuple
  `(; jacprep, jvpprep)` when `reuse === nothing`, or `(; jacprep, jvpprep, fac, age = Int32(0),
  stalled = false)` when `reuse` is an `Int32`. `LineFunction(prob, x, Fx, d, p, jvpprep)` takes the residual `Fx` at `x` as a positional argument
  and evaluates the merit `φ(lf, α) = norm2(F(x + α*d, p))`, which is always `norm2`, and its
  derivative `φ′(lf, α) = 2 * rdot(F(x + α*d, p), jvp)` by one pushforward (`jvp!!`).
  `globalize!!` dispatches on the line search: `Static(α)` evaluates `F(x + α*d, p)` directly
  without building a `LineFunction`, while `Backtracking`, `Bisection` and `StrongWolfe` build a
  `LineFunction` and call `linesearch`. A step that the search takes returns the residual the
  search evaluated there, so an accepted step costs no second evaluation of `F`. In `linesearch`, `StepKind{R}` accepts
  an `InexactStep{R}` or raw slope `R` only when its element type matches the line search's `R`;
  another real type is a `MethodError`.
  `Newton(; linsolve, globalization, jacobian, refinement, regularization, reuse)` evaluates `J`
  (or reuses `state.fac` while `0 < state.age < reuse` and `!state.stalled`), adds
  `regularization * I` to the diagonal when `regularization > 0`, factorises with `factorize!!`,
  solves `J * s = Fx` with `ldiv!!`, and sets `d₀ = -s` and `d = refine!!(nothing, refinement, ...)`.
  The step kind passed to `globalize!!` is `ExactStep()` on a fresh factorisation when `jacobian` is
  `nothing` or an `AbstractADType`, `regularization` is zero, `factor_eltype(linsolve, T) === T`,
  and `refinement isa NoRefinement`, and `MeasuredSlope()` otherwise. A regularised step
  `d = -(J + μI)⁻¹F` does not satisfy the `-2φ₀` slope of `ExactStep`, so it takes the measured
  one. When a line search returns `LINESEARCH_FAILED`
  with a finite trial merit strictly below `φ₀ = norm2(Fx)`, `globalize!!` accepts the trial point
  and records `LINESEARCH_FAILED` in `step_failures`; otherwise the solver stops with
  `LINESEARCH_FAILED`. A line search along a reused factor that fails without a step is retried
  once, in the same step, from a fresh factor: the iteration count does not grow, the step costs
  one more Jacobian and factorisation, and a solve that stopped with `LINESEARCH_FAILED` after a
  stale step now continues. `Picard(; damping = 1, globalization = Static())` takes `d = -damping * Fx`
  with `MeasuredSlope()`. On `SVector` and `Number` iterates, `solve` allocates zero bytes after
  warmup.

- `DogLeg(; radius = nothing)` (exported), the trust-region globalization of `Newton`:
  `Newton(; globalization = DogLeg())`, with the defaults of Procedure 4.1 of Pawlowski, Simonis,
  Walker and Shadid (2008). Each step evaluates and factorises `J` once and solves for the Newton
  step once; the step is the Newton step inside the radius, else the Cauchy step along `-J' F`
  scaled to the radius when the Cauchy point lies outside it, else the point of the radius on the
  segment between them. `J` is the factorised, regularised matrix, and its adjoint serves a
  complex `T`. A trial is accepted at `pred > 0` and `ared ≥ 10⁻⁴ pred`, with
  `pred = ‖F‖ − ‖F + J s‖` and `ared = ‖F‖ − ‖F(x + s)‖` in unsquared norms; a rejected trial
  shrinks the radius by 0.25 and chooses again on the same curve with no new Jacobian, and a
  rejection at the floor `10⁻⁶` returns `LINESEARCH_FAILED` with the iterate kept. After an
  accepted step other than the whole Newton step, only the residual test can end the solve with
  `SUCCESS`, as after a stall, and the step is not counted as a stall. The radius, in
  `[10⁻⁶, 10¹⁰]`, starts at the first Newton step's norm (the Cauchy length without one), or
  at `2·10⁻⁶` when that norm is below `10⁻⁶`, as Procedure 4.1 has it, unless `radius` gives
  it, and lives in the `Newton` state, `(; jacprep, jvpprep, radius)`. `DogLeg` refuses a `radius`
  that is not positive and finite as given, and stores it clamped to `[10⁻⁶, 10¹⁰]` after each
  conversion. A singular factor or a Newton step whose norm is not finite takes the Cauchy point,
  cut to the radius when it lies outside it. The Cauchy direction is formed from the unit
  residual, `J' (F / ‖F‖)`, so no norm of `J' F` occurs: in `Float32`, a scale of `F` and `J` by
  `10^k`, `k ∈ {−16, −12, −8, 8, 12, 16}`, gives the code and the iteration count of the unscaled
  Rosenbrock, Dennis–Schnabel and cubic problems. Where the Cauchy point is needed and that
  direction is zero or its norm is not finite, or `J` maps it to zero or to a vector whose norm is
  not finite, the step returns `STALLED`; at `F(x) = 0` it returns `SUCCESS` with the iterate kept.
  `Newton` refuses `DogLeg` with `reuse` with an `ArgumentError`, because a reused factor makes the
  model stale. `DogLeg` solves `atan(x)` from 10 and the Dennis–Schnabel system from `(2, 0.5)`,
  where a fixed step diverges or ends `NONFINITE`, and the Rosenbrock system from `(−1.2, 1)`, in
  `Float32` and `Float64`; on the Rosenbrock system a KernelAbstractions kernel gives the host's
  result bit for bit, and a solve allocates nothing. On `x³ − 2x + 2` from 0 it stops with
  `LINESEARCH_FAILED` at `√(2/3)`, the local minimum of `|F|`: the root lies beyond a local maximum
  of `|F|` above `|F(0)|`, so no path along which `|F|` decreases reaches it.

- Internal: R3 dense CPU linear solvers and mixed-precision solve. `GeometricSolvers.dense_factor(m, prob, x, p)`
  pre-allocates `DenseLU`, `DenseQR` or `DenseSVD` for an `AbstractVector` state `x` of element type
  `T = eltype(x)`, with factor precision `TF = factor_eltype(m, T)` and a right-hand-side conversion
  buffer `rhs` when `TF !== T`. On `BlasFloat` (`Float32`, `Float64`, `ComplexF32`,
  `ComplexF64`), `factorize!!` and `ldiv!!` call LAPACK's `getrf!`/`getrs!`, `geqrf!` with
  `ormqr_adj!` and a `BLAS.trsv!` backsolve, and QR-iteration `gesvd!` with rank cutoff
  `√n · eps(real(TF)) · σ₁`, reusing the pre-allocated workspaces without allocation. On other
  `AbstractMatrix` element types (`BigFloat`), `DenseLU` and `DenseQR` use `LinearAlgebra.lu!` (with
  `RowMaximum()` and `check = false`) and `LinearAlgebra.qr!`; `DenseSVD` has no `BigFloat` method.
  In mixed precision (`TF !== T`), `ldiv!!` also scales the right-hand side by its infinity norm
  before converting to the factor precision, and returns zero directly on a zero right-hand side.
  The scale `s` of the right-hand side is split as `s = frac · 2^ex` with `frac` in `[0.5, 1)`. The
  step is multiplied by `frac` and then by `2^(ex - e)` in one `ldexp`, so a right-hand side near
  `floatmax(T)` makes the step overflow only where the solution itself does.
  Where the element type of `J` differs from the factor type, `factorize!!` divides `J` by the power
  of two `2^e` that brings its largest `|re|` or `|im|` into `[1, 2)`, as the static factorisation
  does, and `ldiv!!` multiplies the step by `2^-e` in both paths. So a `Float64` or `ComplexF64` `J`
  at a scale where its entries overflow or become subnormal in `TF`, such as `1e39` or `1e-40` for
  `Float32`, solves to the factor precision's bound. Entries far below the largest of `J` can still
  underflow, and an `Inf` in `J` gives a step that is not specified. `LinearAlgebra` is now a
  direct dependency of the package.

- Internal: R3 dense `KernelAbstractions` linear solvers on `AbstractGPUArray`
  (`src/linear/ka/lu.jl` and `src/linear/ka/qr.jl`). For a device vector iterate `x isa
  AbstractGPUArray`, `GeometricSolvers.dense_factor(m, prob, x, p)` builds `DenseLU` or `DenseQR` with `Int32` pivot and
  status buffers on the same device (`similar(x, Int32, n)` and `similar(x, Int32, 1)`), and
  `factorize!!` and `ldiv!!` run column-stepped `KernelAbstractions` kernels in place of the host
  `LAPACK` / `LinearAlgebra` methods, with zero host scalar indexing
  (`GPUArraysCore.allowscalar(false)`) and one `Array(status)` readback per `factorize!!` call
  (`0` on `ldiv!!`). Partial-pivoting LU selects its pivot by `abs1(z) = abs(real(z)) +
  abs(imag(z))` (as LAPACK's `ixamax` does) and divides subdiagonal entries through the
  overflow-safe `multiplier(num, den)`. Unpivoted Householder QR scales each subcolumn by its
  largest entry magnitude `amax = maximum(abs, A[k:n, k])` before forming the reflector norm and
  records the first zero diagonal entry of `R` in `info`. Triangular backsolves (`L`, `U` and `R`)
  and the application of `Qᴴ` to the right-hand side run in one work-item (`ndrange = 1`) on the
  device; the factorisations' column updates are parallel over the trailing matrix. In mixed
  precision (`TF !== T`), `ldiv!!` converts and scales the right-hand side and the step by
  broadcasts without host scalar indexing. The public `solve` does not reach these kernels: its
  `init` for `Newton` builds no dense factor from the iterate, and its step factorises with
  `nothing`, which has methods for static matrices only. The device path is reached only through
  the internal `dense_factor`, `factorize!!` and `ldiv!!`, which a device-resident R3 `solve` would call.
  `norm2(a)` and `rdot(a, b)` call `sum` without an `init` keyword (`isempty(a) ? zero(R) :
  R(sum(...))`), so `Array` and `Broadcasted` reductions use Base's pairwise `_mapreduce` rather
  than sequential `mapfoldl_impl`. `scripts/spikes/metal_reductions/run.jl` measures `norm2` and
  `rdot` on `MtlVector{Float32}` at `n = 10⁶, 10⁷, 10⁸` against a `Float64` host reference on
  positive-mean (`rand`) and zero-mean (`randn`) inputs: every relative error stays within the
  `⌈log₂(n)⌉ · eps(Float32)` pairwise tree-reduction bound (`20`, `24` and `27 · eps(Float32)`),
  so no custom `KernelAbstractions` reduction kernel is needed.

- `PerSystem` and `solve_batch!` (exported), together with the internal `GeometricSolvers.parameters`,
  run the R1 nonlinear solver loop across a batch of independent small systems in place on
  `get_backend(X)` through KernelAbstractions, one work-item per system. `PerSystem(data)` wraps an
  indexable collection of per-system parameters so system `i` receives `data[i]`; any parameter
  value not wrapped in `PerSystem` (including a plain `AbstractVector`, `Tuple` or `SVector`) is
  shared unchanged across every system, and `Adapt.@adapt_structure PerSystem` adapts `PerSystem`
  to device arrays.
  `solve_batch!(X, S, prob, method, p = nothing; order = nothing, options = nothing, kwargs...) -> S`
  writes each system's `SolverStatus{R}` into `S[idx]` and returns `S`, raising `ArgumentError`
  when `R !== real(eltype(eltype(X)))`, or when `get_backend(S) != get_backend(X)`: a host `S` with
  a device `X`, or a device `S` with a host `X`, is refused, so no kernel writes a status into a
  host `S`.
  `solve_batch!(X, prob, method, p = nothing; order = nothing, options = nothing, kwargs...)` allocates a
  1-byte-per-system `ReturnCode` buffer on `get_backend(X)`, copies the codes once to the host
  after `synchronize(backend)`, and returns the six outcome counts as a `NamedTuple`
  `(; SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED)`, for which
  `GeometricSolvers.isconverged(counts)` returns `counts.SUCCESS == sum(counts)`. On the host,
  `solve_batch!` adapts `method` with `Adapt.adapt(ToReal{R}(), method)`, converts or constructs
  `Options{R}`, and builds `state₀ = init_state(m, prob, zero(eltype(X)), nothing)` once by value so
  `Newton(; reuse = k)` carries its chord factor across steps inside each work-item. When `order`
  is a permutation of `1:length(X)`, work-item `i` solves system `idx = order[i]`; an `isbits` range
  is passed by value, and a non-`isbits` host `order` vector passed with a device `X` is copied to
  `get_backend(X)` before kernel launch. A length mismatch in `S`, `PerSystem`, or `order` raises
  `DimensionMismatch`. An `order` of the right length that repeats an index or holds one outside
  `1:length(X)` raises `ArgumentError`; the check reads a device `order` on the host, one copy per
  call. Passing both `options` and keyword tolerances also raises `ArgumentError`.
  `scripts/r2_throughput.jl` benchmarks R2 `solve_batch!` (`Newton()` and `Newton(; reuse = 4)`) against
  a threaded CPU R1 loop and reports the R4 crossover dimension `n`.
  `KernelAbstractions` (0.9.15 or later, the first release whose compat admits Adapt 4) is now a
  direct dependency of the package.

- `NonlinearSolver`, `init` and `solve!` (exported), together with `solve`, run `Newton` and
  `Picard` in R3 on an `AbstractVector` iterate (`Array` and device arrays), sharing `solve!!` and
  `step!!` with R1 through the storage seam. `init(prob, x, method, p = nothing; options, kwargs...)`
  adapts `method` to `R = real(eltype(x))` and builds a `NonlinearSolver` holding `prob`, the
  adapted `method`, `Options{R}`, the iterate `x` and the parameters `p` (a copy of `x` when `x` is an
  array, `x` itself when it is a `StaticArray` or a `Number`), and the pre-allocated workspace state,
  without mutating `x`. In R3,
  `Newton` holds its factorisation buffer in `state.fac`
  (`GeometricSolvers.dense_factor(m.linsolve, prob, x, p)`), aliases
  `state.J` to `state.fac.factors` when `!(m.globalization isa DogLeg)` and
  `factor_eltype(m.linsolve, eltype(x)) === eltype(x)`, and allocates a separate `n × n` Jacobian
  buffer `state.J` when `m.globalization isa DogLeg` or the factor and working element types differ.
  `solve!(solver, x, p = solver.p)` resets per-solve scalar state (`age = Int32(0)` and
  `stalled = false` for `Newton` with `reuse`, and `radius = radius₀` for `Newton` with `DogLeg`),
  stores `x` and `p` in `solver.x` and `solver.p`, solves from `x`, and returns `(solver.x, status)`,
  with zero steady-state heap allocations on `Array` once `solver` is built. An `x` whose element
  type or length differs from the template given to `init` raises `ArgumentError` before any
  evaluation. `solve!(solver)` solves with the `x` and `p` held by `solver`, and the solver keeps the
  `p` of `init` in `solver.p`.
  `solve(prob, x₀, method, p = nothing; options, kwargs...)` copies `x₀` when `x₀` is not a
  `StaticArray` or a `Number`, so `x₀` is never mutated.
  The internal storage seam (`src/base/bang.jl`) adds the array methods of `buffer`, `residual!!`,
  `update_iterate!!`, `map!!` and `mul!!`, and the value and array methods of `copy!!`, `scale!!`,
  `axpby!!` and `regularize!!` (which shifts the diagonal in a loop on a `Matrix` so
  bounds-checked runs allocate nothing, and updates `view(J, diagind(J))` by broadcast on any other
  `AbstractMatrix` so device arrays perform no scalar indexing). In `src/ad/`, `jacobian!!` adds its
  `AbstractMatrix` methods for
  `::Nothing` (`prob.J(J, x, p)`) and `JacobianFunction` (`s.f(J, x, p)`), `JacobianMatrixJVP`
  evaluates `prob.J(prep.J, x, p)` and `mul!!(Jv, prep.J, v)` in place for `jvp!!`, and
  `Newton`'s `init` shares one AD preparation between `jacprep` and `jvpprep` when
  `prob.J === nothing` and `m.jacobian` is `nothing` or an `AbstractADType`. With `nothing` it is
  `prepare_ad(AutoForwardDiff(), …)`; with an `AbstractADType` it is `prepare_ad(m.jacobian, …)`,
  which on an array iterate is a `ChunkedForwardDiff` for `AutoForwardDiff()` and a `DIJacobian`
  for another back end (an `ArgumentError` on a device iterate). With an analytic `prob.J`,
  `Newton` and `Picard` `init` build the `JacobianMatrixJVP` (an `n × n` buffer) only when the
  globalization is a searching line search (`Backtracking`, `Bisection` or `StrongWolfe`); under
  `Static`, `DogLeg` or any other globalization the `jvpprep` is `nothing`, because only a
  searching line search calls `jvp!!`. In `StrongWolfe`,
  when `search` returns an earlier trial `αres != αlast` after a later trial overwrote the shared
  residual buffer `Ft`, `merit_and_residual(lf, αres)` refreshes `Ft` before returning so
  `globalize!!` copies the residual at `x + αres * d`; this evaluation of `F` is not counted in
  `evaluations`.

- Internal: R3 device `KernelAbstractions` `SVDFactorization` on `AbstractGPUArray`
  (`src/linear/ka/svd.jl`). For a device vector iterate `x isa AbstractGPUArray`,
  `GeometricSolvers.dense_factor(m::SVDFactorization, prob, x, p)` allocates `fac.work = similar(x, Int32, 1)` (the
  per-sweep rotation-flag buffer) and `fac.rwork = similar(x, real(F), n)` (column norms squared
  during the sweep, and the sorted scaled singular values `σ̂` of `J / 2^e` after it).
  `factorize!!` prescales `J` unconditionally by `2^-e` (`e = matrix_exponent(J)`) into `fac.U`
  through `ldexp2` before initializing `fac.factors` (`V`) to the identity so that
  `J === fac.factors` is safe, and runs up to `JACOBI_MAX_SWEEPS` (`30`) sweeps of the round-robin
  schedule `jacobi_pair` with `m ÷ 2` disjoint parallel column pairs per round (`m = n + isodd(n)`),
  skipping negligible columns (`α <= maximum(fac.rwork) * jacobi_tolerance(n, R)^2` or
  `β <= maximum(fac.rwork) * jacobi_tolerance(n, R)^2`) and nearly orthogonal pairs
  (`abs(γ) <= jacobi_tolerance(n, R) * √(α * β)`) and, once per sweep, reading the scalar
  `maximum(fac.rwork)` (a non-finite value ends the sweep with `info == 1`) and `Array(fac.work)[1]`.
  After convergence, `fac.rwork` (`σ̂`), `fac.U` and `fac.factors` (`V`) are sorted in
  non-increasing order, `fac.S` stores `ldexp2.(σ̂, e)`, columns of `fac.U` with
  `σ̂ⱼ <= jacobi_tolerance(n, R) * σ̂₁` are zeroed, `fac.VT` stores `V'`, and `info` is `0` when the
  sweep converges with finite singular values (including rank-deficient and zero matrices) and `1`
  when the sweep cap is reached or a non-finite entry is present. `ldexp2` and `matrix_exponent`
  operate on the unsigned bit pattern of `Base.IEEEFloat` so that devices with hardware
  flush-to-zero (such as Metal) preserve subnormal entries and singular values. `backsolve!`
  divides by the scaled singular values `fac.rwork` above `jacobi_tolerance(n, R) * fac.rwork[1]` in
  two parallel `KernelAbstractions` kernels with zero host-device scalar indexing
  (`GPUArraysCore.allowscalar(false)`), so a matrix with finite entries and `σ₁ > floatmax(R)` still
  solves to a finite step.

- The standard test set runs in the `core` group, under the label "Standard test set". It is the
  23 problems of NonlinearProblemLibrary.jl 0.1.8 (Burkardt's `test_nonlin`, after Moré, Garbow
  and Hillstrom), copied into `test/helpers/standard_problems.jl` with every non-integer constant
  in the element type of the iterate, from the starts `u0`, `10 u0` and `100 u0` (67 starts). It
  solves each start with nine columns — `Newton` with `Static`, `Backtracking`, `Bisection`,
  `StrongWolfe` and `DogLeg` on LU, `Newton` with `Backtracking` on QR and on SVD, `Newton` with
  `reuse = 3` and `Backtracking` on LU, and `Picard` with `Backtracking` — in `Float32` and
  `Float64`, with `max_iterations = 200`: 1206 cells. `test/verification/standard.jl` fails when
  a cell's return code differs from the committed table
  `test/verification/standard_reference.csv`, when its iteration count differs by more than 1, or
  when a cell is in only one of the run and the table; it prints the `SUCCESS` cells per column
  and element type. A cell whose outcome round-off decides is flagged `unstable` and checked only
  for its presence and a code other than `ERROR`: the update script runs each cell 20 more times
  (16 for `Picard`), with the start, the residual or the Newton direction moved by 1 to 4 ulps,
  and flags the cell where a run returns another code, an iteration count that differs by more
  than 1, or a final iterate farther than `√eps(T) · max(1, ‖x‖)`. It flags 133 of the 1206
  cells, among them the 42 cells in which the Linux and Windows runners of CI differed from a
  table written on Apple silicon; a cell that a runner moves and the marker leaves is flagged
  `platform` by hand, with the same check. `scripts/standard/README.md` reports the diagnosis:
  round-off amplification, seeded by CPU-dependent OpenBLAS and LAPACK kernels. A change that
  moves a cell regenerates the table with `scripts/standard/update_reference.jl`, so each later
  fix shows its effect as a list of changed cells. The library is no dependency:
  `scripts/standard/check_transcription.jl`, in an environment of its own, checks the copy
  against it.

- Mixed-precision iterative refinement (`Refinement(T = Float64; η = √eps(real(T)), maxiter = 5)`,
  `Refinement(η, maxiter = 5)`, and `GeometricSolvers.refine!!`), factorisation promotion to
  working precision `T` (`GeometricSolvers.promote_linsolve`), and the `QuasiNewton(k; kwargs...)`
  convenience constructor for `Newton(; reuse = k, kwargs...)`.
  `Refinement{R <: Real}` (`η::R`, `maxiter::Int32 ≥ 1`, `0 < η < 1`, `isbits`, adapted by
  `ToReal{R}`) performs up to `maxiter` defect-correction passes in working precision `T` using
  the existing factor `fac`: a step that factorised at `x` multiplies by the regularised Jacobian
  it holds, and a step along a reused factor takes the Jacobian-vector product `jvp!!` at `x`. Each
  pass forms `r = -F(x) - J(x) d` in `T`, solves
  `fac \ r` in `TF` for `δd`, updates `d ← d + δd`, and stops once
  `‖F(x) + J(x) d‖ ≤ max(η, √eps(R)) ‖F(x)‖`. With `regularization = μ > 0`, `J(x)` throughout
  this paragraph stands for `J(x) + μI`, the matrix that the factor solves: `Refinement` refines the
  direction of `(J(x) + μI) d = -F(x)`, and the linear-model residual that a promotion reads uses the
  same matrix. Combining `Refinement` (or `refinement = true`)
  with `AutoFiniteDiff`, bare or inside `AutoSparse`, raises an `ArgumentError` in `init` because
  finite-difference directional derivatives corrupt the defect correction near the
  finite-difference step floor. If a pass does not strictly decrease `‖F(x) + J(x) d‖`
  (`!(r_curr < r_prev)`) or `maxiter` passes finish above the threshold, `refine!!` returns the best
  direction reached with `failed = true`. In `Newton`, a failed refinement on a reused factor
  (`reuse = k`, `age > 0`) first triggers an early refactorisation at the current iterate in the
  factor precision (`age` reset) within the same step, which counts one in `StepInfo.failures`.
  With `TF ≠ T`, the refactored step is refined again, and if that refinement fails, or the
  factorisation fails in `TF`, `Newton` (not with `LevenbergMarquardt`, K57) promotes the
  factorisation to working precision `T` via `promote_linsolve(m.linsolve, T)`
  (`LUFactorization(real(T))` / `QRFactorization(real(T))` / `SVDFactorization(real(T))` /
  `LDLTFactorization(real(T))`), recomputes `d = -J \ F(x)` in `T`, increments `StepInfo.failures`,
  and sets `StepInfo.promoted = true` (sticky in `SolverStatus.promoted`). In R1 (`Fx === nothing`),
  `NewtonState.fac_T` is `nothing` even when `reuse` is set. With `NoRefinement` and any
  globalization but `DogLeg` and `LevenbergMarquardt`, `NewtonState.J` aliases the factor's matrix
  (`fac.factors` when `TF = T`, and `fac_T.factors` with `LUFactorization` when `TF ≠ T`), so
  promotion allocates no extra `n × n` `T` buffer. When `Newton` with `TF ≠ T`
  (not with `LevenbergMarquardt`, K57) makes slow progress (`‖F(x₊)‖ > 0.5 ‖F(x)‖`) on two
  consecutive steps that measured their relative linear-model residual
  `‖F(x) + J(x) d‖ / ‖F(x)‖` and found it above `max(√eps(R), 1e-4)`, it promotes the
  factorisation to `T` with `promoted = true`; every other step returns unchanged, as a step in
  `T` does, so a mixed step keeps going wherever its all-`T` control does, and the precision floor
  of `F` itself is never read.
  `scripts/mixed_refinement_eta.jl` compares pure-`Float64` Newton, `η_J = √eps(Float64)`,
  Kelley's `1e-6`, and Eisenstat–Walker Choice 2 with a `√eps(Float64)` floor on
  `F(x) = A * x .+ 0.5 .* (x .^ 2)` at `N = 16`, `f_abstol = 1e-12`, and
  `κ(J) ∈ (1e2, 1e4, 1e5, 1e6)`.

- `StationarityProblem(S; E = :, U = :, b = nothing)` (and positional `StationarityProblem(S, E, U, b = nothing)`)
  and `LDLTFactorization()`, with `LDLTFactorization(TF)` for a factorisation precision `TF`: solve
  the stationarity condition
  $(\nabla S(x, p))_E + b = 0$ over $u = x_U$ of a scalar action $S$, alongside Bunch–Kaufman
  $\mathrm{LDL}^\top$ / $\mathrm{LDL}^\dagger$ factorisation (`StaticLDLT` in R1, zero-allocation
  LAPACK `sytrf!`/`sytrs!` and `hetrf!`/`hetrs!` with generic fallback `DenseLDLT` in R3 host, and
  `KernelAbstractions` `DeviceLDLT` in R3 device). For an array iterate `StationarityProblem`
  accepts `E` and `U` as `Colon()`, an `SVector{N, <:Integer}`, an `NTuple{N, <:Integer}`, an
  `AbstractUnitRange{<:Integer}`, or a `Vector{Int}`, and for an R1 iterate as `:`, a `Tuple` or an
  `SVector`; it keeps inactive components $x_{\notin U}$ fixed, and solves through `solve`, `init`,
  `solve!` (with keyword `b = nothing`) and `solve_batch!`, which refuse a `ForwardDiff.Dual` in
  the iterate or in `p` with an `ArgumentError`: a `StationarityProblem` has no implicit
  differentiation in this release (K31). `StationarityProblem` and
  `NonlinearProblem` share the supertype `AbstractNonlinearProblem`: `solve!`, `solve!!` and
  `NonlinearSolver` take either, and `init` and `solve` have a method for each problem type. In R1,
  `gradient_hessian!!` evaluates both $(\nabla S)_E + b$ and $(\nabla^2 S)_{E, U}$ in one nested
  `ForwardDiff.Dual` call, and `jvp!!` evaluates $(\nabla^2 S)_{E, U} v$ with one outer directional
  partial. In R3, `ChunkedHessianForwardDiff` provides the automatic differentiation on `Array` and
  on an `AbstractGPUArray` without scalar indexing, and `DIStationarity` the other back ends on CPU
  `Array`.

- `OptimalWeights`, `FixedWeights`, `Additive`, `Multiplicative`, `Repeat` and `Switch` (exported),
  together with the internal `GeometricSolvers.sync_merit`, compose nonlinear step rules (`Newton`,
  `Picard` and any combinator) across R1 (`StaticVector`, `MVector` and `Number`), R2
  (`solve_batch!` on host and device arrays) and R3 (`Array` and `AbstractGPUArray`), with
  `Adapt.@adapt_structure` on every combinator and weight rule. `solve`, `init` and `solve_batch!`
  on a `StationarityProblem` take a combinator method, as they take `Newton` and `Picard`.
  `OptimalWeights(ρ = 1.0)` (`OptimalWeights(; ρ = 1.0)`, requiring finite `ρ > 0`, else
  `ArgumentError`) solves the real-weight (`R = real(T)`) secant least-squares problem
  $\min_{w \in \mathbb{R}^m} \|F(x) + \sum_{i=1}^m w_i (F(x_i) - F(x))\|_2$ from the Gram matrix
  $G_{ij} = \operatorname{Re}\langle \delta F_i, \delta F_j \rangle$ and right-hand side
  $b_i = -\operatorname{Re}\langle \delta F_i, F(x) \rangle$ ($\delta F_i = F(x_i) - F(x)$), up to
  $m = 4$ members (`m > 4` raises `ArgumentError` in `init_state`). In R1, $G$ (`SMatrix{m, m, R}`)
  and $b$ (`SVector{m, R}`) are formed by `rdot` with zero allocations; in R3, $\delta F_i$ are
  stored in `work.ΔF` ($n \times m$), $G = \operatorname{Re}(\Delta F^{\mathrm{H}} \Delta F)$ and
  $b = -\operatorname{Re}(\Delta F^{\mathrm{H}} F(x))$ are formed by `mul!!`, and on a device array
  the small $m \times m$ and $m$ buffers are copied to the host once per step. Under the
  $\sin^2\theta$ rank rule ($\tau_{\sin^2} = 10^{-3}$), for $m = 2$,
  $\sin^2\theta = 1 - (G_{12} / G_{11})(G_{12} / G_{22})$ (and for $m \ge 3$, incremental Cholesky
  ordered by non-decreasing $\|F(x_i)\|$, with ties keeping the lowest member index) drops
  collinear or zero directions ($\sin^2\theta < 10^{-3}$ for $p \ge 1$, or $r_{jj}^2 \le 0$),
  sets `rank_fallback = true` (incrementing `step_failures` by 1), and either solves the 1-weight
  secant step $w_k = b_k / (G_{kk}(1 + \operatorname{eps}(R)))$ along the kept smallest-residual
  candidate ($w_k = 1$ if $G_{kk} = 0$) or solves the kept $p \times p$ system ($p \ge 2$) with
  Tikhonov regularisation $\lambda = \operatorname{eps}(R) \cdot (\operatorname{tr}(G_{\text{kept}})
  / p)$. `FixedWeights(w₁, w₂, ...)` (`FixedWeights((w₁, w₂, ...))`, requiring at least one weight
  and all finite real values, else `ArgumentError`) combines member corrections with fixed real
  weights without a Gram solve or safeguard check.
  `Additive(m₁, m₂, ...; weights = OptimalWeights())` (`fieldcount(M) >= 1` and matching
  `FixedWeights` length, else `ArgumentError`) runs one step of each member from the same
  $(x, F(x))$ (stepping in preallocated per-member copies `(xs[i], Fxs[i])` in R3 so `x` is not
  mutated while members run), and leaves out a member that hard-fails (`SINGULAR`, `NONFINITE`, or
  `LINESEARCH_FAILED` with zero `stepnorm`); its candidate is $(x, F(x))$ at residual norm $\infty$
  and takes weight zero. `OptimalWeights` solves for the weights of the members kept.
  `FixedWeights` rescales the kept weights by $\sum_{i=1}^m w_i / \sum_{i \text{ kept}} w_i$, so
  that they sum to the total of all the weights, and leaves them unchanged when no member fails.
  The step fails, with `stepnorm = 0` and the code of the first member that failed, when every
  member fails or when that rescale factor is not finite or zero (the kept fixed weights sum to
  zero, or the weights sum to zero in total); otherwise it forms $d = \sum_{i=1}^m w_i (x_i - x)$,
  $x_+ = \texttt{update\_iterate!!}(\cdot, x, 1, d)$ and $F_+ = F(x_+, p)$. Under
  `OptimalWeights`, the PETSc safeguard checks whether `!isfinite(‖F(x₊)‖)`,
  $\sum_i |w_i| < 0.1$, or $\|F(x_+)\| > \rho \min_i \|F(x_i)\|$; when triggered, it replaces
  $(x_+, F_+)$ with the candidate $(x_{\text{best}}, F(x_{\text{best}}))$ of smallest residual norm
  (ties to lowest index) and increments `step_failures` by 1, superseding a `rank_fallback` that
  fired on the same step so that one composite fallback is counted once. Under `FixedWeights`, which
  has no safeguard, a non-finite $\|F(x_+)\|$ reports `NONFINITE` instead of `SUCCESS`; the
  `OptimalWeights` safeguard replaces such a trial before it is reported, so only `FixedWeights`
  reaches this.
  `Multiplicative(m₁, m₂, ...)` (`fieldcount(M) >= 1`, else `ArgumentError`) threads $(x, F(x))$
  sequentially through its members in one outer step, calling `sync_merit` before each member step
  so a started line-search member updates its stored baseline merit $\phi(x) = \|F(x)\|_2^2$ to the
  incoming residual without resetting `Newton`'s `FactorAge` (`reuse`) or `DogLeg`'s `TrustRadius`
  (while preserving `NaN` on an un-started state so calling `step!!` without `start!!` fails closed
  with `NONFINITE`), and a hard failure of a member (`SINGULAR`, `NONFINITE`, or
  `LINESEARCH_FAILED` with zero `stepnorm`) ends the outer step with that member's code, keeps the
  iterate that the members before it reached, and reports the outer step norm
  $\|x^{(m)} - x^{(0)}\|$, so a `LINESEARCH_FAILED` after an earlier member moved the iterate is not
  a zero step. `Repeat(method, count)` (`count >= 1`, else `ArgumentError`) runs `count` sub-steps
  of `method` per outer iteration with `sync_merit` between sub-steps, and a hard failure of a
  sub-step ends the outer step with its code, keeps the iterate that the earlier sub-steps reached,
  and reports the outer step norm.
  `Switch(first, second; threshold = 1e-2)` (finite `threshold > 0`, else `ArgumentError`) stores
  `fnorm₀ = NaN` in `init_state` and after `finish!!` (so an un-started `step!!` returns `NONFINITE`
  with `stepnorm = 0`), records `fnorm₀` and `switched = false` in `start!!`, steps `first` while
  $\|F(x)\| > \text{threshold} \cdot \|F(x_0)\|$, and latches `switched = true` permanently to
  `second` once $\|F(x)\| \le \text{threshold} \cdot \|F(x_0)\|$. In `finish!!`, `Additive` resets
  the spectrum diagnostics (`rank = -1`, `nullnorm = NaN`, `gradratio = NaN`), `Multiplicative`
  forwards the diagnostics of the last member in sequence that populated them, `Repeat` forwards
  its inner member's diagnostics, and `Switch` forwards those of the active branch.
  `Additive`, `Multiplicative` and `Repeat` report `STALLED` when every member step (every
  sub-step, for `Repeat`) is `STALLED`, and `SUCCESS` otherwise; a hard failure keeps its own
  code. So `Repeat(m, 1)` and `Multiplicative(m)` stall where `m` does and count toward
  `max_stalls`, and `Switch` passes the code of its active member through unchanged. `solve` with
  a combinator method (`Additive`, `Multiplicative`, `Repeat`, `Switch`) and a `ForwardDiff.Dual` in
  `x₀` or `p` raises an `ArgumentError` before the primal solve, since implicit differentiation
  through a combinator is not implemented (K60); `init`, `solve!` and `solve_batch!` refuse a `Dual`
  input for every method.
- `Anderson` (exported): `Anderson(member = Picard(),
  Val(m) = Val(4); c_s = 0.2, κ_bar = nothing, restart_factor = 10.0, M_alpha = 25.0,
  safeguard_cos = nothing, grow_depth = false)` (plus the convenience `Anderson(member, m::Integer;
  kwargs...)` forwarding to `Val(Int(m))`) accelerates an inner fixed-point or nonlinear step map
  $g(x) = x_{\mathrm{member}}$ with fixed-point residual $w(x) = x_{\mathrm{member}} - x$ in R1
  ($m \le 4$, `isbits` state on `SVector` and `Number`; the residual may have another type than the
  iterate, so an `MVector` iterate with an `SVector` residual works (K70)), R2 (`solve_batch!`
  kernel), and R3 (no upper bound on `m`, zero heap allocations in `step!!` and `solve!` after
  warm-up on `Array`, and a `JLArray` step with no scalar indexing under `allowscalar(false)`).
  Implements Pollock and Rebholz's length filter and Gram–Schmidt angle filter (`c_s`, `κ_bar`,
  defaulting to `1 / sqrt(eps(R))` per iterate element type via `effective_kappa_bar`), trace-scaled
  Tikhonov regularisation on the Gram normal equations $\Delta W^H \Delta W \gamma = \Delta W^H w_k$,
  extrapolation weight and model-decrease safeguards ($\sum_j |\alpha_j| \le M_\alpha$ over the
  affine weights of the member candidates, exact also after the angle filter drops an interior
  history column (K71), optional `safeguard_cos` and `grow_depth`), automatic history clears on
  residual growth ($\|F\| > \text{restart\_factor} \cdot \text{fnorm\_best}$) or stagnation,
  `StepInfo.failures` counting filter column drops, rejected weights and history restarts,
  `state.current_depth` in `1:m` after every step except one that ends in a hard member failure
  (`SINGULAR`, `NONFINITE`, or `LINESEARCH_FAILED` with no step taken), which leaves it unchanged,
  and best-iterate restoration in `finish!!` when `solve!!` exits with `status.code != SUCCESS` and
  a strictly smaller historical residual norm was recorded. It solves a `NonlinearProblem` or a
  `StationarityProblem`, with R1 tested on `SVector` and R3 on `Vector`. An accelerated step reports
  `STALLED` when its member step does and `SUCCESS` otherwise; a fallback or restart step and a hard
  failure keep the member's code. `solve` with `Anderson` and a `ForwardDiff.Dual` in `x₀` or `p`
  raises the same `ArgumentError` as the other combinators, before the primal solve, since implicit
  differentiation through a combinator is not implemented (K60). The method-first
  `solve(prob, method::Anderson, x₀, p, options)` is not provided (K73).

- `Reparametrized(prob, g, ∂g = nothing; covariant = prob isa StationarityProblem,
  curvature = true)` and `ReparametrizedSolver` (exported) solve a `NonlinearProblem` or
  `StationarityProblem` in chart coordinates `x = g(y, p)` and return `(x* = g(y*, p), status)`
  in the original coordinates from `solve` and `init` / `solve!` (`solve!!` returns
  `(x*, status, state)`), with `solve_batch!` writing each system's `x*` into the batch, and
  leave `update_iterate!!` untouched. In R1 (`StaticVector` or `Number`), `g(y, p)` returns
  `x` and `∂g(y, p)` returns `G = ∂g/∂y`; in R3 (`AbstractVector`), `g!(x, y, p)` writes into
  `x` and `∂g!(G, y, p)` writes into `G`, and when `∂g === nothing`, `G(y, p) = ∂g/∂y` is
  computed by tagged ForwardDiff in R1 and chunked forward-mode AD in R3. With
  `covariant = false` (default for `NonlinearProblem`), the transformed residual and Jacobian
  are `F̃(y, p) = F(g(y, p), p)` and `J̃(y, p) = J(g(y, p), p) G(y, p)`. With `covariant = true`
  (default for `StationarityProblem`), the residual is pulled back as
  `F̃(y, p) = transpose(G(y, p)) F(g(y, p), p)`, and the Jacobian is
  `J̃(y, p) = transpose(G) J G + (∂Gᵀ/∂y) F(g(y, p), p)` when `curvature = true` or
  `transpose(G) J G` when `curvature = false`. A `StationarityProblem` needs `U = :` or `U = 1:n`
  in order, where `n` is the chart dimension, and `E = :` or a permutation of `1:n`; `init`,
  `solve`, `solve_batch!` and `prepare_ad` raise an `ArgumentError` for any other index set. For
  an R1 iterate the index sets are `:`, a tuple or an `SVector`, as for a `StationarityProblem`
  without a chart, so a range such as `E = U = 1:2` raises an `ArgumentError` there.
