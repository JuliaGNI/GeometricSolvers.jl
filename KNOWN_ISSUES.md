# Known issues

### K2 · Metal.jl 1.11's precompile workload fails where no Metal device exists, and the test environment works around it

- **location:** `test/Project.toml` (`[preferences.Metal]`)
- **evidence:** Metal.jl 1.11.1's precompile workload (`src/precompile.jl:18`) calls
  `mtlfunction(identity, Tuple{Nothing})`, which calls `device()`. In the Bash sandbox
  `Metal.devices()` is empty, so Metal cannot precompile its workload there. The issue is
  worked around: `test/Project.toml` sets Metal's `precompile` preference to `false` (Metal.jl
  PR #915, `@load_preference("precompile", true)` at `src/Metal.jl:110`), which turns the workload
  off, and `test/devices/metal.jl` asserts that the test process reads the preference as `false`.
  With the preference `true` and no Metal cache for it, the sandboxed
  `run-tests.jl <repository> devices/metal.jl` fails with `Failed to precompile Metal` and
  `BoundsError: attempt to access 0-element Vector{Metal.MTL.MTLDevice} at index [1]`; with
  `false` it passes, with the assertion passed and the device skip recorded (Julia 1.13.3). The
  fix is upstream: Metal.jl 1.12.0 runs its workload only where `!isempty(devices())`
  (JuliaGPU/Metal.jl#1000, which closed JuliaGPU/Metal.jl#996). Metal.jl 1.12.0 needs
  KernelAbstractions 0.10, while this package bounds KernelAbstractions at 0.9, so the test
  environment resolves Metal.jl 1.11.1, and the preference stays until the package moves to
  KernelAbstractions 0.10.
- **kind:** upstream
- **found:** 2026-10-02

### K4 · A plain `AutoEnzyme()` cannot differentiate a residual whose parameters are a `Constant`

- **location:** `test/enzyme/runtests.jl` (`BACKEND`)
- **evidence:** `prepare_ad(AutoEnzyme(), prob, r, x, p)` followed by `jacobian!!` raises
  `EnzymeRuntimeActivityError: Detected potential need for runtime activity. Constant memory is
  stored (or returned) to a differentiable variable`, pointing at the broadcast
  `r .= x .* x .+ p .* x .+ 2 .* x[perm]` of `Coupled` in `test/helpers/adproblems.jl`, with
  Enzyme 0.13 and Julia 1.13.1. Enzyme's static activity analysis cannot prove that the constant
  parameter array broadcast into an active result is non-differentiable. The documented remedy is
  runtime activity, so `test/enzyme/runtests.jl` uses
  `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))`, with which every check
  passes. The same residuals differentiate through ForwardDiff with no such setting, so this is a
  property of Enzyme's analysis and not of this package. A caller who passes `AutoEnzyme()` to the
  R3 solver with parameters will meet the same error and the same remedy; nothing in the package
  sets the mode for the caller, because the mode is the caller's choice.
- **kind:** upstream
- **found:** 2026-10-08

### K7 · A value iterate that is a mutable static array allocates on every update.

- **location:** `src/base/bang.jl` (`update_iterate!!`, `map!!`)
- **evidence:** `update_iterate!!(nothing, v, α, v)` with an `MVector{4, Float64}` allocates 48
  bytes, 32 in `Float32`, 48 in `ComplexF32` and 80 in `ComplexF64`, and `map!!(-, nothing, v)` the
  same, measured with `@allocated` at testset scope in a cold process on Julia 1.13.2. `buffer(v)`
  is `nothing` and allocates nothing. The result of an arithmetic operation on a mutable static
  array is a new mutable array, which the heap holds, so no method of the seam can avoid this while
  the iterate keeps its type. An `MVector` is a value iterate for the call form and for the value
  methods, and the zero-allocated-bytes target of the R1 regime names `SVector` and `Number`, so
  the target does not cover this case.
- **kind:** found late
- **found:** 2026-10-16

### K23 · The same-type LU and QR solves overflow in the back substitution for a finite right-hand side near `floatmax` whose solution is representable

- **location:** `src/linear/factors.jl` (`ldiv!!`), `src/linear/dense.jl` (`ldiv!!`)
- **evidence:** With `H = H₄ / 2`, `H₄` the 4 × 4 Hadamard matrix,
  `J = H diag(2^20, 2^14, 2^20, 2^20) H` and `b = 0.75 floatmax(T) · H[:, 2]`, whose solution
  `0.75 floatmax(T) / 2^14 · H[:, 2]` is representable (`7.78844f33` and `4.11459e303` in modulus),
  the step of `ldiv!!` with `LUFactorization()` and `QRFactorization()` is
  `[Inf, -Inf, Inf, -7.78843f33]` in `Float32` and `[Inf, -Inf, Inf, -4.11459e303]` in `Float64`,
  on an `SMatrix` and an `Array` (Julia 1.13.3); the `JLArray` route fails the same test. `r` is
  divided by `2^k` with `k = 2`, which keeps `Qᴴ r` finite; a product of an entry of `U` or `R`
  near `2^19` with an entry of the scaled step near `floatmax / 2^16` exceeds `floatmax`, which
  points at the back substitution, but the operation that overflows was not traced.
  `SVDFactorization()` solves the system. The test systems of
  `test/helpers/matrix.jl` leave this system out for LU and QR.
- **kind:** defect
- **found:** PR #48

### K26 · A device vector wrapped in a `ComponentArray` gets the host factor, and the host QR and SVD then fail

- **location:** `src/linear/factors.jl` (`init_linear`)
- **evidence:** With ComponentArrays 0.15.50 and JLArrays 0.3.3, `x = ComponentVector(JLArray(...))`
  gives `x isa AbstractGPUArray == false`, `copy(x)` stays a `ComponentVector`, and
  `similar(x, Float32, 4, 4)` is a `JLArray{Float32, 2}`. `init_linear` selects the device factor
  by the type of `x`, not by the matrix it allocates, so all three builders return the host
  factors `DenseLU`, `DenseQR` and `DenseSVD` around a device matrix, and `factorize!!` on
  `DenseQR` and `DenseSVD` fails with "Scalar indexing is disallowed". The same predicate,
  `x isa AbstractGPUArray`, is on `main`, in `src/linear/factors.jl` and in `prepare_ad` of
  `src/ad/di.jl`. ComponentArrays is neither a dependency nor a test dependency of the package, so
  a wrapped device vector is not a supported input. Support is its own pull request, which selects
  the factor from the allocated matrix instead of from `x`.
- **kind:** defect
- **found:** PR #55 (CodeRabbit, on #56)

### K30 · A solve whose residual is wider than the start iterate fails in the `SolverStatus` constructor

- **location:** `src/nonlinear/solver.jl` (`SolverStatus`)
- **evidence:** With `F(x, p) = x^2 - p` and `prob = NonlinearProblem(F)`,
  `solve(prob, 1.5, Newton(), 4.0)` and `solve(prob, 1.5f0, Newton(), 4.0f0)` return `SUCCESS`,
  while `solve(prob, 1.5f0, Newton(), 4.0)` raises `MethodError: no method matching
  SolverStatus{Float32}(::ReturnCode, ::Int32, ::Float64, ::Float32, ::Int32, ::Bool)` (merge of
  `main` into `task/z`, Julia 1.13.3). The status takes its real type from the start iterate, and
  the residual norm of a `Float64` parameter is a `Float64`. The implicit-differentiation path of
  `solve` inherits the same failure. Whether the solve promotes the iterate, converts the norm or
  refuses the mix with an `ArgumentError` is open.
- **kind:** defect
- **found:** PR #53

### K55 · `LevenbergMarquardt` on a sparse `jacobian_prototype` runs the dense stacked solve

- **location:** `src/globalization/levenbergmarquardt.jl` (`init_globalization`, `lm_qr_factor!`)
- **evidence:** `Newton(; linsolve = LUFactorization(), globalization = LevenbergMarquardt())` on
  a `NonlinearProblem` with a `SparseMatrixCSC` `jacobian_prototype` takes the stacked path,
  because `lm_svd_path` is `false` for every method but `SVDFactorization` with `scaling = :none`.
  `init_globalization` then allocates the dense `n × n` buffers `R₀` and `R`, `lm_qr_factor!`
  reads the sparse Jacobian one entry at a time in an `O(n³)` Givens reduction, and the
  `SparspakLU` factor that `init_factor` builds is never factorised. The solve is correct, but its
  cost and storage are those of a dense solve, not of the sparse one. With
  `QRFactorization` or `SVDFactorization` a sparse prototype raises a `MethodError` in
  `init_linear`, as for every other globalization. Whether this combination is refused with an
  `ArgumentError`, as a device iterate is in `check_lm_iterate`, or gets a sparse damped solve is
  open.
- **kind:** defect
- **found:** PR #65

### K56 · The stacked `LevenbergMarquardt` step allocates a linear factor that it never reads

- **location:** `src/nonlinear/newton.jl` (`init_state`, `lm_newton_factor`)
- **evidence:** `init_state` calls `init_factor(m.linsolve, prob, x, p, jacprep)` for every
  `Newton` without reuse, and `init_newton_J` gives `LevenbergMarquardt` a Jacobian buffer of its
  own. On the stacked path (`QRFactorization`, `LUFactorization`, or `scaling = :diagonal`)
  `lm_newton_factor` returns `state.fac` without a `factorize!!`, and the trial loop reads only the
  stacked buffers of `init_globalization`. The factor is one `n × n` array of state that no step
  reads.
- **kind:** dead code
- **found:** PR #65

### K65 · A `LeastSquaresProblem` has no implicit differentiation

- **location:** `src/nonlinear/leastsquares.jl` (`solve_implicit`)
- **evidence:** `solve(LeastSquaresProblem(f), x₀, GaussNewton(), p)` with a `ForwardDiff.Dual` in
  `x₀` or `p` raises `ArgumentError: solve does not accept ForwardDiff.Dual inputs for a
  LeastSquaresProblem; implicit differentiation of a LeastSquaresProblem is not implemented`
  (`test/nonlinear/leastsquares.jl`, testset "LeastSquaresProblem, GaussNewton, Options.g_tol, and
  stopping_test"). The implicit tangent of `solve` is that of a root, `J(x*) ẋ = -∂F/∂p` with a
  square `J`. A least-squares solution is a stationary point of `½‖F‖²`, whose tangent equation
  is `(Jᴴ J + Σᵢ Fᵢ ∇²Fᵢ) ẋ = -(Jᴴ ∂F/∂p + Σᵢ Fᵢ ∂(∇Fᵢ)/∂p)`; its Gauss–Newton form drops the
  terms in `F` and is exact only at a zero residual. Which form `solve` should implement is open.
- **kind:** defect
- **found:** PR #62

### K66 · A factor precision below the working one cannot reach the default stationarity test on a nonzero-residual `LeastSquaresProblem`

- **location:** `src/linear/mixed.jl` (`ldiv!!`), `src/nonlinear/solver.jl` (`stationary_converged`)
- **evidence:** For `r(x) = (x₁ - 1, x₂ - 2, x₁ + x₂)` from `[0, 0]`, `x* = (0, 1)`, on a `Float64`
  iterate: array `QRFactorization(Float32)` gives `STALLED` after 6 iterations with
  `gradratio = 7.9e-8` and `‖x - x*‖ = 2.7e-7`; array `SVDFactorization(Float32)` gives `MAXITERS`
  after 100 with `gradratio = 5.0e-8`; static `QRFactorization(Float32)` gives `STALLED` after 3,
  and static `SVDFactorization(Float32)` gives `SUCCESS` after 4. With `g_tol = 1e-6`, array
  `QRFactorization(Float32)` gives `SUCCESS` after 2 (review of PR #62). The perturbed factor moves
  the Gauss–Newton fixed point off the least-squares point by about `eps(Float32)`, and the
  default `g_tol = √eps(Float64)` is below that; the square Newton solve recovers the working
  precision through the residual, and nothing does so here. Whether the solve refuses the mixed
  factor on a `LeastSquaresProblem` with `m > n` or documents it with a reachable tolerance is
  open.
- **kind:** defect
- **found:** PR #62

### K67 · The two-argument `factor_diagnostics` methods have no caller in `src/`

- **location:** `src/linear/dense.jl` (`factor_diagnostics(fac, r)`, two methods),
  `src/linear/static/qr_rect.jl`, `src/linear/static/svd_rect.jl`
- **evidence:** `grep -rn "factor_diagnostics(" src` finds only four-argument calls
  (`newton.jl`, `dense.jl`, `svd_rect.jl`); the two-argument methods are called from
  `test/linear/static/qr_rect.jl`, `test/linear/static/svd_rect.jl` and
  `test/nonlinear/leastsquares.jl`. Their meaning differs from the four-argument one that the
  solver reads: `nullnorm` is relative, and the QR method takes a "rank" from `info`, which the
  `factorize!!` docstring says QR does not reveal; the `DenseSVD` and `DeviceSVD` method allocates
  `similar(r, n)` per call. Removing them needs the rectangular tests to assert the four-argument
  diagnostics instead, and what replaces the QR rank assertions is open.
- **kind:** dead code
- **found:** PR #62

### K68 · `test/nonlinear/leastsquares.jl` may exceed the 60 s budget of a `core` file

- **location:** `test/runtests.jl` (`nonlinear/leastsquares.jl` in `core`)
- **evidence:** Cold `run-tests.jl <repository> test/nonlinear/leastsquares.jl` on Julia 1.13.3:
  `PASS in 121 s`, `Test Summary` 1m49.7s, at a load average of 15.3 at the start; the review of
  PR #62 measured `PASS in 110 s` at a load average of 9.95. Both runs were on a loaded machine,
  so whether the file stays above 60 s on a quiet one, and moves to `slow`, is open.
- **kind:** not verified
- **found:** PR #62

### K69 · A static `LeastSquaresProblem` evaluates its residual at `x₀` twice in a cold `solve`

- **location:** `src/nonlinear/newton.jl` (`init_state`)
- **evidence:** For a `StaticArray` iterate, `init_state` evaluates `F(x₀)` to check the residual
  length and the regularization of a rectangular Jacobian, with `reuse = nothing` too, and
  `solve!!` evaluates it again at the start of the loop. The residual length of a static problem
  is a type parameter of the value `F` returns, so it is known only from an evaluation.
- **kind:** defect
- **found:** PR #62
