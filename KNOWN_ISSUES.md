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

### K60 · `solve` with a combinator method has no implicit differentiation

- **location:** `src/composition/implicit.jl` (`solve_implicit`)
- **evidence:** `solve` with a `ForwardDiff.Dual` in `x₀` or `p` and a `Newton` or `Picard`
  method runs the primal solve and then the implicit-function-theorem tangent solve
  (`solve_implicit` in `src/nonlinear/implicit.jl`). The tangent solve takes its Jacobian
  preparation from `_tangent_jacprep`, which has a method for a `NewtonState` and a `PicardState`
  only. `solve` with an `Additive`, `Multiplicative`, `Repeat` or `Switch` method therefore raises
  an `ArgumentError` for a `ForwardDiff.Dual` in `x₀` or `p`, before the primal solve; `init`,
  `solve!` and `solve_batch!` refuse a dual input for every method. Implicit differentiation
  through a combinator, which needs the Jacobian of `F` at `x*` and not the steps of the members,
  is for a later pull request.
- **kind:** defect
- **found:** PR #60
