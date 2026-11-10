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
  KernelAbstractions 0.10, and this package bounds KernelAbstractions at 0.10.5, so the test
  environment resolves Metal.jl 1.12.0, and the preference stays until a sandboxed
  `run-tests.jl <repository> devices/metal.jl` passes without it.
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

### K24 · `Bisection`, `StrongWolfe` and `DogLeg` fail on the saddle residual `(x₁ + 0.4x₁³ − 1, −x₂ + 0.4x₂³ + 2)` while `Static` and `Backtracking` reach its root

- **location:** `src/globalization/bisection.jl`, `src/globalization/strongwolfe.jl`,
  `src/globalization/dogleg.jl`
- **evidence:** With `F(x, p) = SVector(x[1] + 0.4x[1]^3 - 1, -x[2] + 0.4x[2]^3 + 2)`,
  `prob = NonlinearProblem(F)` and `solve(prob, SVector(0.8, 1.5), Newton(; linsolve =
  LUFactorization(), globalization = g))`, the tree of `613c564` returns
  `Bisection: [0.7985352209656604, 0.9128709291833161] LINESEARCH_FAILED it=3`,
  `StrongWolfe: [0.7972843024431895, 0.912870914067057] STALLED it=9` and
  `DogLeg: [0.7972810583101553, 0.9128713775004471] LINESEARCH_FAILED it=14`, while `Static` and
  `Backtracking` return `SUCCESS` at the root `[0.7972810583155764, -2.1876029597039026]` (Julia
  1.13.3). The same three fail on the stationarity problem of that action in
  `test/nonlinear/stationarity.jl`, with `LDLTFactorization()` and with the same iterates, so the
  outcome does not depend on the problem type. The three stop next to
  `x̂ = [0.7972810583155764, 1/√1.2]`, where `F(x̂) = [0, 1.3914193805498154]`, the Jacobian is
  singular (`−1 + 1.2x̂₂² = 0`), and `‖F‖₂²` has a strict local minimum: `‖F(x̂)‖₂² = 1.9360478926`,
  its `ForwardDiff` gradient is `[0, 0]` and its Hessian has the eigenvalues `6.0969` and `6.2148`.
  So the measurement shows the three searches stopping at a nonzero local minimum of their merit,
  not a defect in them.
- **kind:** found late
- **found:** 2026-11-06

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

### K27 · `init` accepts `LDLTFactorization` for a `StationarityProblem` with `E ≠ U`, whose Jacobian is not symmetric

- **location:** `src/nonlinear/stationarity.jl` (`StationarityProblem` docstring, `init`)
- **evidence:** `LDLTFactorization` reads only the lower triangle of `J = (∇²S)_{E, U}`, and `J`
  is symmetric only where `E == U`. The `StationarityProblem` docstring states this limit, but
  `init`, `solve`, `solve!` and `solve_batch!` do not check it, so a solve with `E ≠ U` and
  `LDLTFactorization` factorises the wrong matrix with no error. The check is for a later pull
  request: `init` raises an `ArgumentError` for `LDLTFactorization` when `E ≠ U`.
- **kind:** defect
- **found:** PR #58

### K28 · A `StationarityProblem` with a complex state fails in the default automatic differentiation

- **location:** `src/nonlinear/stationarity.jl` (`StationarityProblem` docstring)
- **evidence:** With `S(x, p) = sum(abs2, x) / 2 - sum(x)`, `solve(StationarityProblem(S), x0)`
  returns `SUCCESS` at `[1.0, 1.0]` for `x0 = Float64[0.1, 0.2]`. For `ComplexF64[0.1, 0.2]` it
  raises `ArgumentError: Cannot create a dual over scalar type ComplexF64`, and for an
  `SVector{2, ComplexF64}` or a `ComplexF64` it raises `MethodError: no method matching
  _stationarity_residual_r1(::ADTypes.AutoForwardDiff{nothing, Nothing}, …)` (tree of `989000e`,
  Julia 1.13.3). The docstring states that the state must be real. Support for a complex state is
  for a later pull request.
- **kind:** defect
- **found:** PR #58

### K29 · `solve!` and `solve_batch!` on a `Reparametrized` stationarity problem do not take the datum `b`

- **location:** `src/nonlinear/reparametrized.jl` (`solve!(::ReparametrizedSolver)`, `solve_batch!`)
- **evidence:** With `S(x, p) = sum(abs2, x) / 2 - sum(x)`, `g(y, p) = y` and
  `r = Reparametrized(StationarityProblem(S), g)`, `solve(r, SVector(0.1, 0.2); b = 1.0)` returns
  the root `[2.8e-17, 5.6e-17]` of `x - 1 + b`. `solve!(init(r, SVector(0.1, 0.2)); b = 1.0)`
  raises `MethodError: no method matching solve!(::ReparametrizedSolver{…}; b::Float64)`, and
  `solve_batch!([SVector(0.1, 0.2), SVector(0.3, 0.4)], r; b = 1.0)` raises
  `MethodError: no method matching Options(::Type{Float64}; b::Float64)` (tree of `9807131`,
  Julia 1.13.3). A plain `StationarityProblem` takes `b` on all four entry points. Support on
  `solve!` and `solve_batch!`, with a `PerSystem` datum for a batch, is for a later pull request.
- **kind:** defect
- **found:** PR #57

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

### K31 · `solve` on a `StationarityProblem` has no implicit differentiation

- **location:** `src/nonlinear/stationarity.jl` (`init`, `solve`)
- **evidence:** `solve` on a `NonlinearProblem` with a `ForwardDiff.Dual` in `x₀` or `p` runs
  the primal solve and then the implicit-function-theorem tangent solve (`solve_implicit` in
  `src/nonlinear/implicit.jl`). `solve` on a `StationarityProblem` has no such path. `init`,
  `solve`, `solve!` and `solve_batch!` on a `StationarityProblem` therefore raise an
  `ArgumentError` for a `ForwardDiff.Dual` in the iterate or in `p`. Implicit differentiation
  through the stationarity equations `(∇S)_E + b = 0`, with the Jacobian `(∇²S)_{E, U}`, is for a
  later pull request.
- **kind:** defect
- **found:** PR #58 (CodeRabbit)

### K40 · A line-search solve with an analytic `J` and a detected sparsity pattern gets a dense JVP buffer

- **location:** `src/nonlinear/newton.jl` (`jacobian_matrix_jvp`)
- **evidence:** `jacobian_matrix_jvp(::LineSearch, x, Fx, prob)` sizes its matrix from
  `jacobian_prototype(prob)`, not from `jacobian_pattern(prob, jacprep)`. The path runs where
  `prob.J` is given, `Newton` has an `AutoSparse` `jacobian` with a sparsity detector, and the
  globalization is a line search. For a tridiagonal residual with `n = 2000`, `Backtracking()` and
  `TracerSparsityDetector()`, the JVP buffer is a `Matrix{Float64}` of 32,000,048 bytes with no
  prototype and a `SparseMatrixCSC{Float64, Int64}` of 112,136 bytes with the tridiagonal
  prototype (`Base.summarysize`, Julia 1.13.3). Both solves use `SparspakLU` and return `SUCCESS`
  in 9 iterations, and the iterate differs from the dense solve by `1.06e-17` relative. The cost
  is memory and time, `O(n²)` per slope evaluation; the result is correct. A sparse buffer would
  hand the analytic `J` a `SparseMatrixCSC` that the caller did not declare, so whether to use the
  detected pattern there is open.
- **kind:** defect
- **found:** PR #59

### K41 · `regularization > 0` with an explicit sparse prototype that has no structural diagonal fails at the first factorisation

- **location:** `src/base/bang.jl` (`regularize!!`), `ext/GeometricSolversSparspakExt.jl`
  (`factorize!!`)
- **evidence:** With `Fperm!(r, x, p) = (r[1] = x[2]^2 - 1; r[2] = x[1]^2 - 1; r)`,
  `P = sparse([1, 2], [2, 1], [1.0, 1.0], 2, 2)` and
  `Newton(; jacobian = AutoSparse(AutoForwardDiff(); coloring_algorithm = GreedyColoringAlgorithm()), regularization = 1e-10)`,
  `solve!(init(NonlinearProblem(Fperm!; jacobian_prototype = P), [1.5, 2.0], m))` raises
  `ArgumentError: sparse Jacobian sparsity pattern differs from jacobian_prototype`; with
  `regularization = 0` the same solve returns `SUCCESS` with `SparspakLU` (Julia 1.13.3). The
  regularisation adds the diagonal to `J`, which inserts structural entries that the factor's
  pattern does not hold. The same failure is on `main`.
- **kind:** defect
- **found:** PR #59

### K42 · A device `solve_batch!` passes a problem with a sparse prototype to the kernel, which a CUDA kernel cannot take as an argument

- **location:** `src/nonlinear/batched.jl` (`_launch!`)
- **evidence:** `_launch!` passes `prob` to `_solve_kernel!` unchanged, and `NonlinearProblem`
  has no `Adapt` rule, so the prototype goes into the kernel arguments. A batched iterate ignores
  the prototype. With `F!(r, x, p) = (r .= x .* x .- 4; r)`, `isbits(NonlinearProblem(F!))` is
  `true`, while
  `isbits(NonlinearProblem(F!; jacobian_prototype = sparse(Diagonal(ones(2)))))` is `false`,
  because `SparseMatrixCSC` holds `Vector` fields (`isbitstype(SparseMatrixCSC{Float64, Int64})`
  is `false`; Julia 1.13.3). CUDA.jl refuses a kernel argument that is not `isbits` after
  `cudaconvert`. The CPU batch with the same prototype returns `SUCCESS` for both systems
  (`test/linear/sparspak.jl`, "R1, R2, and Picard ignore jacobian_prototype"). CUDA is not
  available on this machine, so the failure is not reproduced on a device. A fix passes the kernel
  a problem without the prototype, with a device batch test whose residual captures a field.
- **kind:** defect
- **found:** PR #56 (CodeRabbit, on #59)

### K46 · The CUDA `backsolve!` method for `DeviceSVD` repeats the own-kernel method

- **location:** `ext/GeometricSolversCUDAExt.jl` (`backsolve!` for `DeviceSVD`)
- **evidence:** its body, `ka_svd_backsolve!(fac, d, d, false); return d`, is that of
  `backsolve!(fac::DeviceSVD{<:AbstractMatrix{T}}, d::AbstractVector{T})` in
  `src/linear/ka/svd.jl`, so the method runs the own kernel and calls no cuSOLVER or cuBLAS
  routine. The same-type `ldiv!!` reaches `backsolve_shifted!` and not this method. The CUDA
  census of `test/gpu/cuda/runtests.jl` asserts that `which(backsolve!, ...)` for `DeviceSVD`
  names the extension, so removing the method changes that assertion too.
- **kind:** defect
- **found:** PR #61

### K47 · The CUDA SVD writes `J` into `fac.factors`, which the own-kernel SVD does not, and no later step reads it

- **location:** `ext/GeometricSolversCUDAExt.jl` (`factorize!!` for `DeviceSVD`)
- **evidence:** the CUDA `factorize!!` copies `J` into `fac.factors` (converted to the factor
  type in the mixed case, unscaled) before it factorises `J / 2^e` in `fac.U`; the fallback
  receives `J` itself. `ka_factorize_svd!` (`src/linear/ka/svd.jl`) writes `fac.U` and leaves
  `fac.factors` alone. The docstring of `factorize!!` says that `J` is written into
  `fac.factors`, and `test/gpu/cuda/runtests.jl` asserts that `fac.factors` keeps `J`. The copy
  costs one `n × n` device copy per call where `J` is not the factor buffer. Which of the two
  backends sets the contract for `fac.factors` of a `DeviceSVD` is open.
- **kind:** defect
- **found:** PR #61

### K48 · The CUDA extension calls the deprecated bindings `CUDA.CUSOLVER` and `CUDA.CUBLAS`

- **location:** `ext/GeometricSolversCUDAExt.jl`
- **evidence:** CUDA 6.4.2 declares `Base.@deprecate_binding CUBLAS cuBLAS true` and
  `Base.@deprecate_binding CUSOLVER cuSOLVER true` (`src/CUDA.jl`), and `[compat]` admits CUDA 6
  (`CUDA = "5.5, 6"`). Under `--depwarn=yes` each use prints `Use of CUDA.CUSOLVER is deprecated,
  use cuSOLVER instead.` CUDA 5.x has no `cuSOLVER` binding, so the names that work on both
  majors differ, and the choice between them depends on the CUDA bound.
- **kind:** defect
- **found:** PR #61

### K49 · `test/gpu/README.md` does not describe the by-hand CUDA suite `test/gpu/cuda/runtests.jl`

- **location:** `test/gpu/README.md`
- **evidence:** a search for `cuda/runtests` in `test/gpu/README.md` and `test/gpu/runtests.jl`
  finds nothing. The README describes the `test/gpu/cuda/` environment and the run of
  `test/gpu/runtests.jl` with the backend `cuda`, and that runner includes only `ad.jl`, so no
  file names the vendor suite or says how to run it on a CUDA machine.
- **kind:** docs
- **found:** PR #61

### K56 · The stacked `LevenbergMarquardt` step allocates a linear factor that it never reads

- **location:** `src/nonlinear/newton.jl` (`init_state`, `lm_newton_factor`)
- **evidence:** `init_state` calls `init_newton_factor(m, m.linsolve, prob, x, p, jacprep)` for
  every `Newton` without reuse, and `init_newton_J` gives `LevenbergMarquardt` a Jacobian buffer of its
  own. On the stacked path (`QRFactorization`, `LUFactorization`, `LDLTFactorization`, or
  `scaling = :diagonal`) `lm_newton_factor` returns `state.fac` without a `factorize!!`, and the
  trial loop reads only the stacked buffers of `init_globalization`. The factor is one `n × n`
  array of state that no step reads.
- **kind:** dead code
- **found:** PR #65

### K57 · `Newton` with `LevenbergMarquardt` takes no `Refinement`, and its mixed-precision step never promotes

- **location:** `src/nonlinear/newton.jl` (`Newton` inner constructor, the `LevenbergMarquardt`
  `step!!`)
- **evidence:** `Newton(; globalization = LevenbergMarquardt(), refinement = Refinement())` and
  `refinement = true` raise an `ArgumentError`, tested in `test/globalization/levenbergmarquardt.jl`
  (testset "A factor in Float32 on a Float64 iterate, and no Refinement (K57)"). A `Refinement`
  corrects the solve of `J d = -F` from the factor, while the damped step solves
  `(Jᴴ J + μ D²) δ = -Jᴴ F`: the stacked step reads no factor, and the SVD step weights the
  singular triplets by `σ / (σ² + μ)`. With a linear method in `TF ≠ T`, the stacked step is the
  working-precision step, and the SVD step takes its direction from the `TF` factor with no
  promotion to `T`, so `StepInfo.promoted` stays `false`. A defect correction of the damped system
  in `T`, and a promotion of the SVD factor when its direction stalls, are for a later pull request.
- **kind:** defect
- **found:** PR #65

### K60 · `solve` with a combinator method has no implicit differentiation

- **location:** `src/composition/implicit.jl` (`solve_implicit`)
- **evidence:** `solve` with a `ForwardDiff.Dual` in `x₀` or `p` and a `Newton` or `Picard`
  method runs the primal solve and then the implicit-function-theorem tangent solve
  (`solve_implicit` in `src/nonlinear/implicit.jl`). The tangent solve takes its Jacobian
  preparation from `_tangent_jacprep`, which has a method for a `NewtonState` and a `PicardState`
  only. `solve` with an `Additive`, `Multiplicative`, `Repeat`, `Switch`, `Anderson`,
  `LeftPreconditioned` or `RAS` method
  therefore raises an `ArgumentError` for a `ForwardDiff.Dual` in `x₀` or `p`, before the primal
  solve; `init`, `solve!` and `solve_batch!` refuse a dual input for every method. Implicit
  differentiation through a combinator, which needs the Jacobian of `F` at `x*` and not the steps
  of the members, is for a later pull request.
- **kind:** defect
- **found:** PR #60

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

### K72 · The R1 and R3 `Anderson` code duplicates the filter, the weight solve and the step skeleton

- **location:** `src/composition/anderson.jl` (`filter_history_r1`, `filter_history_r3!`,
  `solve_weights`, `solve_weights!`, both `step!!` methods)
- **evidence:** The history storage and the Gram build differ for a reason (an `isbits` `NTuple`
  in R1; device `n × m` matrices with a `col_idx` permutation and one `mul!!` in R3). The two
  filters differ in 25 of 65 lines and the two Cholesky solves in 19 of 45, mostly `setindex`
  against `setindex!`, and each `step!!` repeats the "take the member step" fallback 3 times and
  the best-iterate update 4 times. `Multiplicative` and `Repeat` are one `step!!` each through
  `copy!!` and `map!!`. A `!!` setter and `copy!!` on the state buffers would leave only the
  history update and the Gram build per storage; the cost of that change was not measured.
- **kind:** defect
- **found:** PR #63

### K75 · An R3 `StationarityProblem` with a device-resident `E` or `U` reads the index set by scalar indexing

- **location:** `src/nonlinear/stationarity.jl:410` (`_init_stationarity_state`),
  `src/ad/hessian.jl:316-317` (`_build_pos_maps`)
- **evidence:** With `prob = @allowscalar StationarityProblem(S, JLArray([1, 2]), JLArray([1, 2]))`
  and `allowscalar(false)`, `solve(prob, JLArray([1.0, 1.0]), Newton())` raises "Scalar indexing
  is disallowed" at `stationarity.jl:410` (`Int[Int(i) for i in prob.U]`), and
  `solve(Reparametrized(prob, g!), JLArray([1.5, 0.5]), Newton())` raises it at `hessian.jl:316`,
  where `prepare_ad` copies `E` and `U` to the host by iteration. The same problems with host
  index sets `[1, 2]` and a `JLArray` iterate return `SUCCESS` (merge of `main` into `task/ay`,
  Julia 1.13.3). The constructor also checks for duplicates by scalar indexing, so
  `Adapt.adapt(JLArray, prob)` raises the same error. `_validate_stationarity_solve` and the index
  check of `Reparametrized` compare a device index set on the device. A host copy of `E` and `U`
  with `Array` before the iteration is for a later pull request.
- **kind:** defect
- **found:** PR #57 (CodeRabbit)

### K80 · A `LeastSquaresProblem` takes no `Refinement`

- **location:** `src/nonlinear/newton.jl` (`init_state`)
- **evidence:** `solve(LeastSquaresProblem(f), x₀, GaussNewton(; refinement = Refinement()))` and
  `init` with an array iterate raise `ArgumentError: LeastSquaresProblem does not support
  Refinement` (`test/nonlinear/leastsquares.jl`, testset "LeastSquaresProblem, GaussNewton,
  Options.g_tol, and stopping_test"). A `Refinement` corrects the direction until the linear
  residual `J d + F` is below its tolerance, and the least-squares step with more residuals than
  unknowns leaves `J d + F ≠ 0` at the least-squares point. A refinement of the least-squares step
  would correct the normal-equation residual `Jᴴ (J d + F)` instead, from the rectangular factor;
  whether to implement it is open.
- **kind:** defect
- **found:** PR #62

### K81 · The CUDA QR and SVD paths are untested for a rectangular factor

- **location:** `ext/GeometricSolversCUDAExt.jl` (`factorize!!` and `backsolve!` for `DeviceQR`
  and `DeviceSVD`)
- **evidence:** the CUDA extension was written and tested for square factors: the by-hand suite
  `test/gpu/cuda/runtests.jl` passed 640 of 640 on an RTX 4090 for the square factors of PR #61.
  A `LeastSquaresProblem` with more residuals than unknowns builds an `m × n` `DeviceQR` or
  `DeviceSVD` on any `AbstractGPUArray`, so with CUDA loaded a rectangular `CuMatrix` factor
  dispatches into the extension. The merge of the two branches adapted the SVD rank cutoff and the
  zeroing of the columns of `U` to `m × n`; whether the QR solve (`ormqr!` and `trsv!`) is correct
  for `m > n`, and the adapted SVD path, have not run on a CUDA device, and no test in
  `test/gpu/cuda/runtests.jl` builds a rectangular factor.
- **kind:** not verified
- **found:** PR #62

### K82 · `Enzyme` 0.13 and `KernelAbstractions` 0.10 cannot share an environment, so the Enzyme suite does not resolve

- **location:** `test/enzyme/Project.toml`, `.github/workflows/Enzyme.yml`
- **evidence:** `KernelAbstractions 0.10.5` depends on `LLVM` and bounds it `LLVM = "10"`, while
  `Enzyme 0.13` bounds `LLVM = "9.10"` (`< 10`). The Enzyme environment contains this package
  through the path source `GeometricSolvers = {path = "../.."}`, so the two bounds meet in one
  resolve, and `Pkg.resolve` in `test/enzyme` stops with "Unsatisfiable requirements detected for
  package LLVM … restricted by compatibility requirements with Enzyme … to versions: 6.1.0 - 9.13.2
  … restricted by compatibility requirements with KernelAbstractions … to versions: 10.0.0 -
  10.4.3 — no versions left". The job is not a required status check, so it fails alone. Enzyme's
  `main` still bounds `LLVM = "9.10"` and `GPUArrays = "10, 11"` (its `Project.toml` of
  2026-11-07, version 0.13.219), so the remedy is upstream and no change here reaches it.
- **kind:** upstream
- **found:** 2026-11-07

### K84 · An explicitly tagged `AutoForwardDiff` over a residual that calls `ForwardDiff.derivative` throws in a compiled kernel

- **location:** `src/ad/forwarddiff_static.jl` (the own chunked path), reached by
  `AUTOForwardDiff` with a caller tag in `test/ad/forwarddiff_static.jl`
- **evidence:** in a kernel on `CPU()`,
  `jacobian!!(nothing, AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T)), nested, x, p)` with
  `nested = NonlinearProblem(R1Closing(), r1_closing_jacobian)` raises
  `KernelException: exception thrown during kernel execution`; the device stack of `julia -g2` ends
  at `ForwardDiff.partials` (`dual.jl:114`) under
  `extract_derivative` → `derivative` → `r1_closing_term` (`test/helpers/r1problems.jl:151`) →
  `R1Closing`. The same call with the *untagged* `AutoForwardDiff()`, and the tagged call over a
  residual without a nested derivative (`R1Tagged`), both return the right Jacobian in the same
  kernel; all three work on the host. So the perturbation tag of the nested `derivative` is not
  isolated from the caller's tag once the code is compiled, and `partials` receives a plain value.
  The test keeps the untagged pair and the tagged-over-`R1Tagged` pair inside the kernel, and the
  tagged-over-nested pair on the host.
- **kind:** upstream
- **found:** 2026-11-08

### K85 · With `--check-bounds=yes`, a `solve_batch!` with `Newton` on `CPU()` throws `KernelException` at run time

- **location:** `src/nonlinear/batched.jl`, `src/nonlinear/reparametrized.jl` (the batch kernels);
  `.github/workflows/CI.yml` (`check_bounds: auto`)
- **evidence:** in a fresh `julia --check-bounds=yes --project=test` (Julia 1.13.3,
  KernelAbstractions 0.10.5), with `c = SVector{2, T}(1, 2)`,
  `solve_batch!([SVector{2, T}(1, 1)], Reparametrized(NonlinearProblem((x, p) -> x - c), (y, p) -> 2 * y, nothing), Newton())`
  and the same batch over the plain `NonlinearProblem((x, p) -> x - c)` both raise
  `KernelException: exception thrown during kernel execution on device cpu`, at `Float32` and at
  `Float64`; POCL prints `ERROR: exception during kernel execution on work-item (1, 1, 1).`
  With `--check-bounds=auto` all four return `SUCCESS = 1`. `Picard()` on the plain problem
  returns `SUCCESS = 1` under `yes` too. The default `Newton()` fails, so any testset that runs a
  `Newton` batch on `CPU()` errors under the flag. The cause is not investigated; the general
  mechanism is that a bounds check defeats `@inbounds`, a static kernel's local `MMatrix` or
  `MVector` then escapes to the heap, and a compiled kernel cannot allocate. CI runs the suite
  with `check_bounds: auto`, so no CI job runs it with `yes`.
- **kind:** defect
- **found:** PR #70
