# Known issues

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
  only. `solve` with an `Additive`, `Multiplicative`, `Repeat`, `Switch`, `Fallback`, `Anderson`,
  `Broyden`, `LeftPreconditioned` or `RAS` method therefore raises an `ArgumentError` for a
  `ForwardDiff.Dual` in `x₀` or `p`, before the primal solve; `init`, `solve!` and
  `solve_batch!` refuse a dual input for every method. Implicit
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

### K69 · A static `LeastSquaresProblem` evaluates its residual at `x₀` twice in a cold `solve`

- **location:** `src/nonlinear/newton.jl` (`init_state`)
- **evidence:** For a `StaticArray` iterate, `init_state` evaluates `F(x₀)` to check the residual
  length and the regularization of a rectangular Jacobian, with `reuse = nothing` too, and
  `solve!!` evaluates it again at the start of the loop. The residual length of a static problem
  is a type parameter of the value `F` returns, so it is known only from an evaluation.
- **kind:** defect
- **found:** PR #62

### K72 · The R1 and R3 `Anderson` code duplicates the filter, the weight solve and the step skeleton, and `Broyden` duplicates the `Anderson` struct

- **location:** `src/composition/anderson.jl` (`filter_history_r1`, `filter_history_r3!`,
  `solve_weights`, `solve_weights!`, `solve_broyden_weights`, `solve_broyden_weights!`, both
  `step_multisecant!!` methods; the `Anderson` and `Broyden` structs, their outer constructors
  and their `Adapt.adapt_structure` methods)
- **evidence:** The history storage and the Gram build differ for a reason (an `isbits` `NTuple`
  in R1; device `n × m` matrices with a `col_idx` permutation and one `mul!!` in R3). The two
  filters differ in 25 of 65 lines and the two Cholesky solves in 19 of 45, mostly `setindex`
  against `setindex!`, and each `step_multisecant!!` method repeats the "take the member step"
  fallback 3 times and the best-iterate update 4 times. `Multiplicative` and `Repeat` are one
  `step!!` each through `copy!!` and `map!!`. A `!!` setter and `copy!!` on the state buffers
  would leave only the history update and the Gram build per storage; the cost of that change
  was not measured. `Broyden` repeats the `Anderson` struct, its inner-constructor validation,
  its four outer constructors and its `adapt_structure` with only the name changed, about 110
  lines; the two differ only in the weight solve that `_solve_multisecant_weights` and
  `_solve_multisecant_weights!` select. One struct with a weight-solve type parameter, with
  `Anderson` and `Broyden` as aliases, would remove the copy.
- **kind:** defect
- **found:** PR #63; the `Broyden` copy, PR #74

### K75 · An R3 `StationarityProblem` with a device-resident `E` or `U` reads the index set by scalar indexing

- **location:** `src/nonlinear/stationarity.jl:491` (`_init_stationarity_state`),
  `src/ad/hessian.jl:318-319` (`_build_pos_maps`)
- **evidence:** With `prob = @allowscalar StationarityProblem(S, JLArray([1, 2]), JLArray([1, 2]))`
  and `allowscalar(false)`, `solve(prob, JLArray([1.0, 1.0]), Newton())` raises "Scalar indexing
  is disallowed" at `stationarity.jl:491` (`Int[Int(i) for i in prob.U]`), and
  `solve(Reparametrized(prob, g!), JLArray([1.5, 0.5]), Newton())` raises it at `hessian.jl:318`,
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

### K86 · `StrongWolfe` returns `SUCCESS` for a step that fails the curvature condition when `φ′(0)` is subnormal

- **location:** `src/globalization/strongwolfe.jl:94`
- **evidence:** `curvature(dα, c₂, d₀) = abs(dα) ≤ -c₂ * d₀` rounds the product. Where `d₀` is a
  subnormal of few significant bits, `c₂ * d₀` rounds to `d₀` itself, so a trial with
  `φ′(α) = φ′(0)` passes the test, although `|φ′(α)| ≤ c₂ |φ′(0)|` is false in exact arithmetic.
  On `main` (`0c9e130`, Julia 1.13.3), `StrongWolfe()` with `MeasuredSlope()` on
  `φ(α) = s (1 − α/L)²`, `s = 9.4e-323`, `L = 31.622776601683793`, `α₀ = 1` returns `SUCCESS` at
  `α = 4.0` with `φ′(0) = φ′(α) = -5.0e-324`; `0.9 * 5.0e-324` rounds to `5.0e-324`. The
  `Float32` case `s = 1.4f-44`, `L = 14.677993f0` does the same with `φ′(0) = -1.0f-45`. A sweep of
  `s` over the subnormal range and `L ∈ [1e-2, 1e2]` from `α₀ = 1` gives 953 `SUCCESS` results in
  `Float64`, 27 of which fail the exact test, and 912 in `Float32`, 28 of which fail. The merit
  there is far below any practical residual. Reproducer, with a line function `lf` whose `φ` and
  `φ′` are the merit above and its derivative:
  `r = linesearch(StrongWolfe(), lf, MeasuredSlope(), φ(lf, 0.0), 1.0)` gives `r.code == SUCCESS`
  and `abs(big(φ′(lf, r.α))) ≤ big(0.9) * abs(big(φ′(lf, 0.0)))` is `false`.
- **kind:** defect
- **found:** 2026-11-12 (a property-based test of the strong Wolfe conditions, checked in
  `BigFloat`)

### K87 · `JET.report_package` errors on Julia pre and nightly, so those CI jobs fail in `test/quality/jet.jl`

- **location:** `test/quality/jet.jl:13`
- **evidence:** JET 0.12.3 (`JET = "0.12"` in `test/Project.toml`, marked `⌅`, held back) warns
  "Full JET functionality is not available on Julia 1.14.0-beta1." and `JET report_package`
  then ends in an `Error During Test at test/quality/jet.jl:13` with the same message:
  "0 passed, 0 failed, 1 errored". The same on 1.15.0-DEV.729 (nightly). `main` fails the same
  way (run 43532996461, job 132255039570, 2026-11-14). The pre and nightly jobs are no required
  checks.
- **kind:** upstream
- **found:** PR #84

### K88 · On Metal, a residual that broadcasts over overlapping `@views` of an `MVector` reads wrong values, and `solve_batch!` returns `SUCCESS` at a point that is no root

- **location:** —
- **evidence:** The residual is the caller's (R2 `solve_batch!` on an `MtlArray`, Metal.jl
  1.12.0, Apple M4 Max). The generalized Rosenbrock residual (n = 10) with
  `@views @. out[2:n] = 10 * (x[2:n] - x[1:n-1] * x[1:n-1])` on `MVector`s, through the inline
  out-of-place adapter of the Y3 premises probe, returns `SUCCESS` at iteration 2 at
  `X = (1, 0, 1, 0, …)` from all three starts. The device residual there reads
  `F = (0, 0, 10, 0, …)`, the host residual `F = (0, -10, 10, -10, …)`: the `x[k-1]` terms read as
  0. The same kernel allocates 80 B per thread on the device heap. On CUDA (GB10, RTX 4090) and
  ROCm (RX 7900 XTX), where the kernel also allocates, the same rows end `SINGULAR` or
  `NONFINITE`, as the host R1 `solve` does from the same start. The same residual as a loop, or as
  a broadcast on views of an `SVector`, is correct on Metal and allocates nothing. The cause of
  the wrong read (an `unalias` copy of the view is suspected) is not established. A caller on
  Metal writes such a residual as a loop.
- **kind:** upstream
- **found:** 2026-11-15 (Y3 premises probe on four cards)

### K89 · SimpleNonlinearSolve's `SimpleNewtonRaphson` in a KernelAbstractions kernel returns `Success` with a NaN residual, so a comparison with it reads its status and its residual

- **location:** `benchmark/`
- **evidence:** With SimpleNonlinearSolve 2.14.6: sample problem 18 (`test/helpers/standard_problems.jl`) at `10u0`, in `Float32`,
  solved with `SimpleNewtonRaphson` in a KA kernel, returns `Success` with host ‖F‖∞ = NaN on Metal
  (M4 Max), CUDA (GB10, RTX 4090) and ROCm (RX 7900 XTX); at `100u0` it does the same on CUDA and
  ROCm. On Metal the returned iterate is `u = (-Inf, NaN)`. GeometricSolvers' `Newton` on the
  same rows returns `SUCCESS` with a finite residual (CUDA and ROCm about 1e-27; on Metal at
  `10u0` a false root of the cancelling form `1 - exp(-x²)`). A benchmark that counts statuses
  only counts these rows as solved.
- **kind:** upstream
- **found:** 2026-11-15 (Y3 premises probe on four cards)

### K91 · The CUDA `backsolve!` for an `m × n` `DeviceQR`, `m > n`, throws `DimensionMismatch`

- **location:** `ext/GeometricSolversCUDAExt.jl:60–61`
- **evidence:** the method calls `CUDA.CUBLAS.trsv!('U', 'N', 'N', fac.factors, d)` on the whole
  factor, and that wrapper throws `DimensionMismatch("Matrix A is $m by $n but must be square")`
  for `m ≠ n` (CUDA 6.5.1, `lib/cublas/src/wrappers.jl:1058–1059`). The mixed-precision `ldiv!!`
  of a `LeastSquaresProblem` with `QRFactorization(Float32)` and a `Float64` iterate calls this
  `backsolve!` on the `m × n` factor with the length-`m` right-hand side
  (`src/linear/mixed.jl:19`). The same-type solve goes to the own-kernel `backsolve_shifted!`
  (`src/linear/ka/qr.jl:184`), which throws no error but, for complex `T`, gives a wrong
  solution: the cuSOLVER `geqrf!` factor has the LAPACK `τ` (H = I − τ v vᴴ), and the own
  `qadj` kernel applies I − τ v vᴴ as Hᴴ, which holds for the LAPACK factor only with
  `conj.(τ)`. A host `LAPACK.geqrf!` factor solved with `ka_qr_backsolve!` on a `JLArray` fails
  for `ComplexF32` and `ComplexF64` at 6 × 3, 40 × 7 and 3 × 1 and passes with `conj.(τ)`
  (critic round 2 probe of the ROCm extension). The ROCm extension solves with `ormqr!` on the
  whole factor and `trsv!` on views of its leading `n × n` block and of `d[1:n]`, which the
  `rocBLAS.trsv!` of AMDGPU 2.12 accepts, and its own `backsolve_shifted!` keeps the same-type
  solve on that path.
- **kind:** defect
- **found:** 2026-11-16 (critic round 1 of the ROCm extension)

### K92 · The launch and wait counts of `scripts/hardware/sync_cost.jl` come from a reading of the source, and no test sees them drift from what the step runs

- **location:** `scripts/hardware/sync_cost.jl:142` (`phase_counts`)
- **evidence:** `phase_counts` returns constants per path and device; the testset
  "sync_cost.jl runs on jlarray at n = 64" of `test/integration/hardware_scripts.jl` asserts
  `r.waits == 8` and `r.launches == 3 * 64 + 11`, the same constants written again. A change of
  `src/` that adds or removes a kernel launch, a `synchronize` or a host read in the step leaves
  both green. A count of the real synchronisations, such as a wrapped backend that counts
  `KernelAbstractions.synchronize` and the host reads, would catch that drift. The vendor counts
  (CUDA, Metal) are not run on a device at all.
- **kind:** missing test
- **found:** 2026-11-16 (critic of the measurement scripts)

### K90 · The residual floor of a `RAS` local solve separates a converged block from one with an error of `√eps ‖x_i‖` only where `n_i κ_F(J_i) < 1/√eps`

- **location:** `src/composition/ras.jl` (`residual_floor`, `_ras_sweep!!`)
- **evidence:** A positive-tolerance local solve is converged at a block residual norm of at most
  `n_i eps ‖J_i‖_F ‖x_i‖`. A block with an error `e` has a residual of at least
  `σ_min(J_i) ‖e‖`, so an error of `√eps ‖x_i‖` lies above that floor only where
  `n_i κ_F(J_i) < 1/√eps`: about 2900 in `Float32`, 6.7e7 in `Float64`. A larger or worse
  conditioned `Float32` block can accept an iterate whose error is of the order of `√eps ‖x_i‖`,
  also from a diverging local solve. The tests reach blocks of 4 indices with `κ ≈ 14`; the
  bound for larger blocks is derived, not measured. The outer iteration still tests the global
  residual.
- **kind:** not verified
- **found:** #97

### K95 · Under Metal.jl 1.12, `factorize!!` never reaches the negative-status branches of `_mps_lu_finish!`, and a non-singular MPS failure throws instead of giving `info = 1`

- **location:** `ext/GeometricSolversMetalExt.jl:44`
- **evidence:** From Metal.jl 1.12.0, `LinearAlgebra.lu!` on an `MtlMatrix` passes the MPS status
  through `MPS.lu_info`, which turns the singular status `-2` into LAPACK's `info` and calls
  `error(...)` on every other failure status; the compat is `Metal = "1.12"`. So
  `LinearAlgebra.lu!(fac.factors; check = false)` at line 44 never returns a negative `info`, the
  `-2` branch and the "any other negative status" branch are reached only by a direct call of
  `_mps_lu_finish!`, and the comments at lines 12–14 and 21–23 describe the old behaviour. On an
  Apple M4 Max (macOS 27.2.0, Julia 1.13.4, Metal.jl 1.12.1), Metal's `info` equals LAPACK's for
  four singular `Float32` matrices (3, 4, 1, 2).
- **kind:** dead code
- **found:** #100 (issue #101)

### K96 · rocSOLVER's complex `geqrf!` above n = 128 writes NaN into the trailing columns when a column of a panel has nothing below its diagonal

- **location:** `ext/GeometricSolversAMDGPUExt.jl:52` (`factorize!!` for `DeviceQR`)
- **evidence:** on an AMD Radeon RX 7900 XTX (gfx1100; AMDGPU 2.12.0, rocSOLVER 3.37.0, rocBLAS
  5.7.0, HIP 7.16, Julia 1.13.4), `AMDGPU.rocSOLVER.geqrf!` on a `ComplexF32` or `ComplexF64`
  matrix gives `NaN` in R where a column has an exactly zero part below the diagonal and a real
  diagonal entry, so its reflector has τ = 0. `4I` is correct at n = 128 and wrong at every n from
  129 to 10⁴, with the first `NaN` at (1, 65). One decoupled variable j in a dense
  `randn + 2n I` (row j and column j zero but for `A[j, j] = 4`, so τ is zero at j only) is
  correct at n = 128, and at n = 256 and 512 gives the first `NaN` in the first column of the
  next 64-column panel, in the 16-row block that holds row j: (1, 65), (33, 65), (49, 65) and
  (97, 129) for j = 1, 40, 64 and 100. The matrix `4I` with 1e-3 below each diagonal entry, the
  same matrices in `Float64`, every `getrf!`, and a complex `rocBLAS.gemm!` of the trailing-update
  shape agree with the host. So the fault is in the blocked complex path of rocSOLVER, which
  starts above n = 128, not in the extension. `factorize!!` reports the `NaN` as the first
  non-finite diagonal entry of R, `info = 65` and `SINGULAR`. The ROCm agreement rows at
  n = 10⁴ (`test/gpu/rocm/runtests.jl`), whose matrix has such columns, mark the complex QR
  assertions broken. The probes are `device-run/probe_info65.jl` and `device-run/probe_tau0.jl`
  on the branch `device-run/rocm-rx7900xtx`, and their output is in
  `results/rocmext-rx7900xtx/` on the branch `results/rx7900xtx`.
- **kind:** upstream
- **found:** #96 (upstream issue https://github.com/ROCm/rocSOLVER/issues/1050)

### K97 · The ROCm QR solve of an `m × n` factor with `m ≥ max(32768, 8n)` rows takes AMDGPU's own blocked `ormqr!`, which no device row reaches

- **location:** `ext/GeometricSolversAMDGPUExt.jl:65` (`backsolve!` for `DeviceQR`)
- **evidence:** the extension applies Qᴴ with `AMDGPU.rocSOLVER.ormqr!('L', 'T'/'C', factors, τ, d)`.
  In AMDGPU 2.12.0 that wrapper calls `ormqr_blocked!`, AMDGPU's own blocked compact-WY
  implementation, in place of rocSOLVER's `ormqr` where `prefer_blocked(mA, k)` holds
  (`src/solver/highlevel.jl:132`, `prefer_blocked(m, k) = m >= max(2^15, 8k)`, read at
  `src/solver/highlevel.jl:318`). The device rows of `test/gpu/rocm/runtests.jl` stop at the square
  `n = 10⁴` and at `m = 40` for an `m × n` factor, so a least-squares Jacobian with
  `m ≥ max(32768, 8n)` rows solves through a path of the dependency that no test runs on a device.
  A device row at, for example, `40000 × 7` would reach it.
- **kind:** not verified
- **found:** #96

### K98 · A `SubArray` right-hand side of a ROCm `DeviceQR` solve falls to the own kernel, which reads the LAPACK `τ` as its conjugate

- **location:** `ext/GeometricSolversAMDGPUExt.jl:59–81` (`backsolve!` and `backsolve_shifted!` for
  `DeviceQR`)
- **evidence:** both extension methods take `d::ROCVector{T}`. A contiguous view, such as
  `view(x, 1:k)` or `view(A, :, j)`, is itself a `ROCVector` (GPUArrays 12.2.0,
  `src/host/base.jl:362`) and reaches them, but a `SubArray`, such as `view(x, 1:2:k)` or
  `view(A, i, :)`, matches neither, and the solve dispatches to the own-kernel `backsolve!` and
  `backsolve_shifted!` (`src/linear/ka/qr.jl`). The factor of the extension comes from
  `rocSOLVER.geqrf!`, with the LAPACK `τ`, and the own kernel applies it with `conj.(τ)` (the
  mechanism of K91), so the solution is wrong for complex `T` at every shape. The call sites of
  `ldiv!!` in `src/` (`grep -rn "ldiv!!(" src/`) pass no view to a `DeviceQR` solve; the implicit
  tangent, for example, passes a right-hand side that `similar` allocates
  (`src/nonlinear/implicit.jl:341`). So no solver path reaches it; a caller that calls `ldiv!!`
  or `backsolve!` with a `SubArray` does.
- **kind:** found late
- **found:** #96

### K99 · The CUDA rows that assert one cuDSS analysis per solver read the extension's own counter, not a count that cuDSS reports

- **location:** `ext/GeometricSolversCUDSSExt.jl:78–84` (`factorize!!` for `CUDSSLU`)
- **evidence:** `test/gpu/cuda/runtests.jl:593`, `:606`, `:632` and `:645` assert
  `fac.analyses[] == 1`.
  `factorize!!` increments the `analyses` cell in the branch that calls `cudss("analysis", …)`,
  and calls `cudss("refactorization", …)` in the other branch. A change that also calls
  `cudss("analysis", …)` in the refactorisation branch, and leaves the counter alone, passes these
  rows. None of the data parameters that CUDSS.jl 1.0.0 reads with `cudss_get`
  (`CUDSS_DATA_PARAMETERS`, `src/types.jl:3–9`) counts the analyses of a solver, so the rows check
  the branch that the extension takes, not the number of analyses that cuDSS runs.
- **kind:** missing test
- **found:** 2026-11-22

### K100 · A direct `factorize!!` of a 0×0 sparse matrix throws, under `UMFPACKFactorization` and under Sparspak's `LUFactorization`

- **location:** `src/linear/sparse.jl` (`factorize!!` for `UMFPACKLU`)
- **evidence:** `init_linear(m, spzeros(0, 0), Float64[])` builds the factor for both methods, and
  `factorize!!` of it on `spzeros(0, 0)` throws: `ArgumentError: the number of rows or columns of
  the matrix must be greater than zero` from UMFPACK, and `An empty problem, no ordering found.`
  from Sparspak. The Sparspak case is true on `main` too, because the branch does not change
  `ext/GeometricSolversSparspakExt.jl`. A Newton solve of order 0 with a 0×0
  `jacobian_prototype` and `UMFPACKFactorization` returns `SUCCESS` before it factorises, so only
  a direct call reaches the throw. Julia 1.13.4, with the test environment of the package.
- **kind:** found late
- **found:** 2026-11-22

### K101 · No test reaches a detected Jacobian pattern in `ComplexF64`, because the sparsity detection of the complex test problem throws

- **location:** `test/helpers/sparse_problems.jl:110` (`F!` of `sparse_bracket_problem`)
- **evidence:** `init` of `NonlinearProblem(prob.F)` from `sparse_bracket_problem(125, ComplexF64)`
  with `Newton(; linsolve = UMFPACKFactorization(), jacobian = AutoSparse(AutoForwardDiff();
  sparsity_detector = TracerSparsityDetector(), …))` raises
  `SparseConnectivityTracer.MissingPrimalError`: `Function iszero requires primal value(s).` The
  tracer vector of the residual is real (`Vector{GradientTracer{Int64, BitSet}}`), and `F!` stores
  into it a sum with a complex matrix entry, whose conversion calls `isreal` and then `iszero` on a
  tracer. The error message names `TracerLocalSparsityDetector`; no test has tried it. The rows of
  a detected pattern in `test/linear/sparse.jl` run in `Float64` only, so
  `has_sparse_factor(::UMFPACKFactorization, ::Vector{ComplexF64})` is asserted, but no solve
  takes a detected complex pattern.
- **kind:** found late
- **found:** 2026-11-23

### K102 · The cuDSS `factorize!!` does not check the pattern of a Jacobian that is the factor's own buffer

- **location:** `ext/GeometricSolversCUDSSExt.jl:65–66` (`factorize!!` for `CUDSSLU`)
- **evidence:** the check compares `J.rowPtr` and `J.colVal` with those of `A = fac.factors` only
  where `J !== A`. A Newton step with no `DogLeg` and no refinement passes the factor's own
  buffer as `J` (`init_newton_J` in `src/nonlinear/newton.jl`), so a `J!` that changes the
  `rowPtr` or `colVal` of that buffer in place goes unchecked, and from the second call on
  `factorize!!` runs cuDSS's `"refactorization"`, which reuses the analysis of the old pattern.
  `UMFPACKLU` in `src/linear/sparse.jl` keeps copies of the prototype's `colptr` and `rowval` and
  compares each Jacobian with them, which closes the same gap on the host. A `CuSparseMatrixCSR`
  has no normal path that inserts an entry in place, so only a `J!` that writes the index arrays
  reaches this. The fix keeps device copies of `rowPtr` and `colVal` in `CUDSSLU` and compares
  with them; it needs a run on a CUDA device.
- **kind:** found late
- **found:** PR #110 (CodeRabbit, the host twin)

### K103 · On the CGS2 path, the residual that `gmres_solve!` reports once the Krylov space is full lies far below the true residual, and a tolerance below the round-off then reports convergence

- **location:** `src/linear/gmres.jl:616` (`linres` from the rotated right-hand side)
- **evidence:** with `rng = Xoshiro(1)`, `A = T.(Diagonal(1:6) + 0.1 randn(rng, 6, 6))`, then
  `b = T.(randn(rng, 6))`, `GMRES(; maxiter = 6)` and `tol = 0`, the default path in `Float32`
  reports `resnorm = 8.7e-17`, while the true residual is `‖b − A d‖ = 5.8e-8 ‖b‖`; in `Float64`
  it reports `6.0e-35` against `2.4e-16 ‖b‖`. Modified Gram–Schmidt (`mgs = true`) reports
  `1.57e-7` against `1.46e-7 ‖b‖`, and `3.0e-16` against `2.4e-16 ‖b‖`. At `j = n` the vector `w₂`
  is round-off, and CGS2 orthogonalises it once more, so `H[n+1, n] = rnorm(w₂)` is the round-off
  of a round-off vector, and the recursive residual, which `H[n+1, n]` scales, falls with it.
  `Newton` stores the reported value as `ws.linres` (`src/nonlinear/newton.jl:2354`). The true
  residual stays at the round-off of the solve; the reported value is low only where the solve
  runs to `j = n`. The `converged` flag is wrong too where `tol` lies between the two: with
  `tol = 1e-20`, `restart = 6` and `maxiter = 12`, the default path reports `converged = true`:
  in `Float32` at iteration 12, with `resnorm = 9.7e-24 ‖b‖` against a true `5.1e-8 ‖b‖`; in
  `Float64` at iteration 6, with `2.6e-35 ‖b‖` against `2.4e-16 ‖b‖`, where the convergence test
  at `:549` and `:566` skips the restart. The MGS path has the same class of fault, and so has the base
  (`47d817b`) with the same code: in `Float64` it reports `converged = true` at iteration 12, with
  `resnorm = 2.68e-32 ‖b‖` against a true `9.46e-17 ‖b‖`; in `Float32` it does not converge
  (`1.5e-15 ‖b‖` reported, `2.6e-8 ‖b‖` true). Each true residual here is formed in `T`. Formed
  in `Float64`, the two `Float32` rows of `tol = 1e-20` give `3.07e-8 ‖b‖` (CGS2) and
  `4.63e-8 ‖b‖` (MGS). A check of the true residual `b − J M⁻¹ u` before convergence is accepted
  covers both paths and costs one Jacobian–vector product on each solve that reaches `j = n`.
- **kind:** found late
- **found:** #106; a limit of the user's decision of 2026-11-22 (CGS2 as the default of `GMRES`),
  with its edges "else one `rnorm(w₂)`" and "the second pass always runs"

### K104 · Two `SVDFactorization(BigFloat; rtol)` with the same `rtol` are not `===`

- **location:** `src/linear/methods.jl:82` (the inner constructor of `SVDFactorization`)
- **evidence:** `rtol` is stored in `real(TF)`, so for `TF = BigFloat` the field is a
  `BigFloat`, which `===` compares by identity.
  `SVDFactorization(BigFloat; rtol = 1e-9) === SVDFactorization(BigFloat; rtol = 1e-9)` gives
  `false`, while the same call with `Float32` or `Float64` gives `true`. The factor reads only the
  value, so the rank and the solve do not change.
- **kind:** found late
- **found:** 2026-11-23

### K105 · No test observes whether a CUDA SVD runs on `gesvdj!` or on the own Jacobi kernel, so a mutant of the route comparison that changes only the route passes

- **location:** `ext/GeometricSolversCUDAExt.jl:128` (the comparison of the rank cutoff with the
  sweep tolerance in `factorize!!`)
- **evidence:** the mutant `<` → `<=` moves the default `rtol = nothing` off `gesvdj!`, and the
  mutant `true &&` never runs `gesvdj!`. Every row of the two `rtol` loops of
  `test/gpu/cuda/runtests.jl`, run on the own kernel alone, passes 96 of 96, because both routes
  give the same rank within rounding. The only observable is the zero `σ̂₂` that `gesvdj!` returns
  in `Float32` for the fixture `1, 1e-10, 1e-14`. A seam that reports the route would catch both
  mutants.
- **kind:** missing test
- **found:** 2026-11-23

### K106 · An R3 `Newton` solve with a `BigFloat` or `Complex{BigFloat}` iterate and the default automatic differentiation throws `UndefRefError`

- **location:** `src/ad/chunked.jl:115` (`jacobian!!` for `ChunkedForwardDiff`), called from
  `newton_factor` in `src/nonlinear/newton.jl`
- **evidence:** with `A = BigFloat[4 1; 1 5]`, `b = BigFloat[1, 2]` and
  `G!(r, x, p) = (r .= A * x .- b; r)`,
  `solve(NonlinearProblem(G!), zeros(BigFloat, 2), Newton())` raises
  `UndefRefError: access to undefined reference` at `prep.xdual .= seed.(prep.xdual, x, …)`,
  which reads the dual buffer before any element of it is set. The buffer comes from
  `similar(x, dual_type(…))` (`src/ad/chunked.jl:65`), and a `BigFloat` dual is no bits type, so
  its elements are undefined references. The same holds for `Complex{BigFloat}` and for
  `LUFactorization()`, `LDLTFactorization()` and `CholeskyFactorization()` alike; `ComplexF64`
  gives `SUCCESS` (Julia 1.13.4, tree of `cd5834b`). The static R1 path and the dense factors
  themselves take `BigFloat` (`test/linear/cholesky.jl`).
- **kind:** found late
- **found:** 2026-11-22

### K107 · The docstrings of `StaticLU`, `StaticSVD`, `StaticLDLT` and `StaticRectQR` say that the factor is `isbits` for every element type, which is false for `BigFloat`

- **location:** `src/linear/factors.jl:132` (`StaticLU`), `:173` (`StaticSVD`), `:342`
  (`StaticLDLT`), `src/linear/static/qr_rect.jl:12` (`StaticRectQR`)
- **evidence:** each docstring says that the size parameter makes the `SMatrix` field concrete,
  "so the factor is `isbits`", with no condition on `TF`.
  `isbitstype(GeometricSolvers.StaticLU{2, BigFloat, 4})` and
  `isbitstype(GeometricSolvers.StaticQR{2, BigFloat, 4})` are `false`, and
  `isbitstype(GeometricSolvers.StaticLU{2, Float64, 4})` is `true` (Julia 1.13.4, tree of
  `c6a2c4f`). The R1 path takes `BigFloat` through `local_array` (`src/linear/factors.jl:818`), so
  such a factor exists. The `StaticCholesky` docstring (`src/linear/factors.jl:407-408`) states
  the condition.
- **kind:** docs
- **found:** #108

### K108 · A static factor with a factorisation type other than the matrix's element type throws `InexactError` for a `BigFloat` matrix whose largest entry has an exponent above `typemax(Int32)`

- **location:** `src/linear/factors.jl:802` (`factor_matrix`), and `src/linear/static/svd.jl:102`
  (`svd_matrix`) and `src/linear/static/svd_rect.jl:51` (`svd_rect_matrix`), which scale always
- **evidence:** the scale exponent of every static factor is an `Int32` field, and
  `factor_matrix` converts the exponent of `matrix_exponent` with `Int32(e)`. With
  `A = SMatrix{2, 2, BigFloat}(big(2)^(big(2)^31 + 5) * [4 1; 1 5])`,
  `factorize!!(nothing, m, A)` raises `InexactError: trunc(Int32, 2147483655)` for
  `m = SVDFactorization()`, `LUFactorization(Float64)`, `CholeskyFactorization(Float64)` and
  `SVDFactorization(Float64)`, and returns a factor for `LUFactorization()`,
  `LDLTFactorization()`, `CholeskyFactorization()` and `QRFactorization()`, which do not scale a
  matrix in its own element type whose exponent is positive or outside the `Int32` range (Julia
  1.13.4, tree of `c6a2c4f`). `origin/main` has the same
  `Int32(e)` in `factor_matrix` and `svd.jl`.
- **kind:** defect
- **found:** #108

### K109 · No public solve is known to fail a step rule or to promote on a step that a least-squares solve discards, so only a unit test of the status helper checks `status.step_failures` and `status.promoted` on that return

- **location:** `src/nonlinear/solver.jl:107` (`discarded_step_status`, which takes
  `step_failures` and `promoted` from the status after `record` folds in the discarded step)
- **evidence:** the solve discards only a step that does not lower `‖F‖`. A count of the calls
  of `discarded_step_status` on the mirror-symmetric saddle problem (b) of the testset "a step
  that leaves a stationary start point returns the start point", under `Static`, `Backtracking`,
  `StrongWolfe`, `StrongWolfe(; maxiter = 1)`, `Bisection(; maxiter = 1)`, `DogLeg` and
  `LevenbergMarquardt`, six perturbations and `Float64` and `Float32`, on `SVector` iterates,
  finds no discard in the 84 solves (Julia 1.13.4). The solves of that
  testset that discard a step, on the problems (a) and (c), take the full steps of `Static`, which
  fail no step rule and do not promote, so only the unit test of `discarded_step_status` with
  synthetic statuses reaches `step_failures == 2` and `promoted == true` there.
- **kind:** missing test
- **found:** 2026-11-24, a limit of the decision that tests the promotion on this return with
  synthetic statuses only; 2026-11-25 for the step-rule failures, a limit of the decision that
  the solve discards only a step that does not lower `‖F‖`

### K112 · OpenBLAS's `getrf!` gives a backward error of order one on a matrix whose entries are subnormal or near `floatmin`

- **location:** `src/linear/dense.jl:267` (`factorize!!` for `DenseLU`, the `getrf!` path)
- **evidence:** `randn(Xoshiro(n), T, n, n)` scaled by `s`, and
  `‖A[p, :] − L U‖∞ / ‖A‖∞` in `BigFloat` (libopenblas64_ 0.3.30, Julia 1.13.4, Apple M4 Max):
  `LAPACK.getrf!` gives 0.51, 1.1 (`Float32`, `s = floatmin`, n = 8, 32), 0.79 (`Float64`,
  `s = floatmin`, n = 32) and 0.56–0.80 (`s = 2^20 · nextfloat(0)`, n = 8, 32, 64, both types);
  `recursive_lu!` on the same matrices gives 1.2e-16 to 1.4e-6, the round-off of the subnormal
  range. At `s = floatmin` and n = 64, `getrf!` gives 2.8e-7 (`Float32`) and 4.9e-16 (`Float64`),
  of the recursion's size.
  The `2 × 2` matrix `[s 1; s/2 1]` with `s = 4 · nextfloat(0)` keeps the multiplier `s/2`
  unscaled in `getrf!`'s factor, where LAPACK's reference `getf2` divides it by the pivot. An
  `n × n` Jacobian in its own element type whose largest entry is 1 or above, with entries at such
  a scale beside it as in that `2 × 2` matrix, reaches `getrf!` unscaled for every `n` outside the
  band of `use_recursive_lu`; one whose largest entry lies below 1 is divided by the power of two
  of that entry first (`load_factors!`).
- **kind:** upstream
- **found:** 2026-11-24

### K113 · The outer iteration counts of Check 6 in `test/composition/preconditioning.jl` are round-off sensitive, so a change of the block factor's round-off moves them across their bounds

- **location:** `test/composition/preconditioning.jl:1083` (the testset "Check 6: the P8 counts
  and the parallel schedule"; the NKS-RAS bound at `:1113`, the RASPEN bound at `:1136`)
- **evidence:** a probe of 64 starts 1 ulp from `x0` (record of 2026-11-24): with `getrf!` as the
  dense block factor, `Float64` RASPEN takes 4–9 outer iterations and
  `Float32` NKS-RAS ends `MAXITERS` from 3 of the 64 starts; with `recursive_lu!` in the band
  `8 ≤ n ≤ 32`, RASPEN takes 4–7 and every `Float32` solve ends `SUCCESS`. From `x0` itself,
  `Float64` RASPEN takes 7 outer iterations under `recursive_lu!`, so a RASPEN bound of `<= 6`
  fails. The RASPEN bound is `<= 9`, the top of the spread under `getrf!`; the NKS-RAS bound is
  `<= 6`. The follow-up is a robust Check 6: a start or a `λ` whose measured spread of
  outer iterations lies inside the bounds.
- **kind:** missing test
- **found:** 2026-11-24

### K115 · The own KA LU solve does not check the length of the vector against the order of the factor

- **location:** `src/linear/ka/lu.jl:149-151` (`ka_lu_backsolve!`)
- **evidence:** `ka_lu_backsolve!` takes `n = length(d)` and launches its kernels with that `n`,
  which read the factor by index under `@inbounds`. A `DeviceLU` of `[4 0; 0 9]` on a `JLArray`,
  with `ldiv!!(JLArray(ones(k)), fac, JLArray(ones(k)))`: `k = 3` throws `ReadOnlyMemoryError()`
  in one cold run and ends the Julia process in another; `k = 1` returns `[0.25]`, the solve of
  the leading `1 × 1` block only, with no error. Julia 1.13.4, JLArrays 0.5.0, tree of `157ca76`.
  The own Cholesky solve, `ka_cholesky_backsolve!`, throws `DimensionMismatch` in this case.
- **kind:** defect
- **found:** #113

### K116 · The mixed-precision `ldiv!!` of a dense factor does not check the length of the vector against the order of the factor

- **location:** `src/linear/mixed.jl:12-27` (the `T ≠ TF` method of `ldiv!!` for every
  `DenseFactor`)
- **evidence:** `rhs .= F.(r ./ s)` broadcasts a length-1 `r` into the length-2 `fac.rhs`, and the
  method then writes `rhs[1:1]` into `d`. A factor of `[4 0; 0 9]` with
  `CholeskyFactorization(Float32)` or `LUFactorization(Float32)`, solved with
  `ldiv!!(A(ones(k)), fac, A(ones(k)))` in `Float64`, for `DenseCholesky`, `DeviceCholesky`,
  `DenseLU` and `DeviceLU`, with `A` an `Array` or a `JLArray`: `k = 1` returns a length-1 result
  with no error, `[0.25]` (`[0.2499999850988388]` for `DenseCholesky`); `k = 3` throws
  `DimensionMismatch` from the broadcast. Julia 1.13.4, JLArrays 0.5.0, tree of `b6aa30e`.
- **kind:** found late
- **found:** #113

### K117 · A same-type `ldiv!!` of a dense factor with vectors of two lengths throws a `TypeError`, not a `DimensionMismatch`

- **location:** `src/linear/dense.jl:234` (the `T == TF` method of `ldiv!!` for every `DenseFactor`)
- **evidence:** with `length(d) != length(r)` the method takes its branch for a rectangular factor
  and asserts `fac.rhs::AbstractVector{T}`, which is `nothing` for a square factor. A factor of
  `[4 0; 0 9]` in `Float32` with `CholeskyFactorization()` or `LUFactorization()`, solved with
  `ldiv!!(A(ones(2)), fac, A(ones(3)))` or `ldiv!!(A(ones(3)), fac, A(ones(2)))`, throws
  `TypeError: in typeassert, expected AbstractVector{Float32}, got a value of type Nothing` for
  `DenseCholesky`, `DeviceCholesky`, `DenseLU` and `DeviceLU`, with `A` an `Array` or a `JLArray`.
  Julia 1.13.4, JLArrays 0.5.0, tree of `b6aa30e`.
- **kind:** found late
- **found:** #113

### K118 · The vendor LU and QR solves return an empty vector with no error for a factor of order `n > 0`

- **location:** `ext/GeometricSolversCUDAExt.jl:32`, `:60`; `ext/GeometricSolversAMDGPUExt.jl:32`,
  `:63` (`backsolve!` of `DeviceLU` and `DeviceQR` on a `CuMatrix` and on a `ROCMatrix`)
- **evidence:** read, not run (no CUDA or ROCm device): each of the four methods starts with
  `isempty(d) && return d` and checks the length of `d` against the factor nowhere before it, so an
  empty `d` returns with no error for a factor of any order. The vendor Cholesky solves in the same
  files check `length(d) == size(fac.factors, 1)` before that return and throw `DimensionMismatch`.
- **kind:** found late
- **found:** #113

### K119 · The own KA LDLᵀ solve does not check the length of the vector against the order of the factor

- **location:** `src/linear/ka/ldlt.jl:394-397` (`ka_ldlt_backsolve!`)
- **evidence:** `ka_ldlt_backsolve!` takes `n = length(d)` and launches its kernel with that `n`,
  which reads the factor by index under `@inbounds`. A `DeviceLDLT` of `[4 0; 0 9]` with
  `LDLTFactorization()` on a `JLArray`, with `ldiv!!(JLArray(ones(k)), fac, JLArray(ones(k)))`:
  `k = 3` throws `ReadOnlyMemoryError()`; `k = 1` returns `[0.25]`, the solve of the leading
  `1 × 1` block only, with no error. Julia 1.13.4, JLArrays 0.5.0, tree of `2390f50`. The own
  Cholesky solve, `ka_cholesky_backsolve!`, throws `DimensionMismatch` in this case.
- **kind:** found late
- **found:** #113

### K120 · The own KA QR solve does not check the length of the vector against the rows of the factor

- **location:** `src/linear/ka/qr.jl:164-177` (`ka_qr_backsolve!`), `:184-194`
  (`backsolve_shifted!` of `DeviceQR`)
- **evidence:** `ka_qr_backsolve!` takes `m, n = size(A)` and its kernels write `d[1:m]` by index
  under `@inbounds`, whatever the length of `d`. A square `DeviceQR` of `[4 0; 0 9]` with
  `QRFactorization()` on a `JLArray`: `ldiv!!(JLArray(ones(1)), fac, JLArray(ones(1)))` returns
  `[0.25]` and the process then ends with signal 11 in a GC finalizer, in two of two cold runs, and
  `backsolve!(fac, JLArray(ones(1)))` ends it the same way; `k = 3` returns
  `[0.25, 0.1111111111111111, 1.0]`, whose third entry is the input, with no error. For the
  `3 × 2` least-squares factor of `[4 0; 0 9; 1 1]`, `backsolve_shifted!(fac, d, JLArray(ones(3)))`
  with `length(d) = 4` copies `rhs[1:4]` and returns `7.26e-322` as the fourth entry, a read past
  `rhs`. Julia 1.13.4, JLArrays 0.5.0, tree of `2390f50`. The own Cholesky solve,
  `ka_cholesky_backsolve!`, throws `DimensionMismatch` for a vector of another length.
- **kind:** found late
- **found:** #113

### K121 · The Cholesky test matrix `cuda_hpd` draws its random numbers in the element type, so each precision gets a different matrix

- **location:** `test/helpers/own_kernel.jl:70`, `:72` (`cuda_hpd`)
- **evidence:** `G = randn(rng, T, n, n)` and `randn(rng, T, n)` take the stream of `T`, so one
  seed gives a `Float32` matrix that is not the `Float64` matrix rounded. With the seed of `n = 4`,
  `randn(Xoshiro(0xc401 + 4), Float32, 4, 4)` and `Float32.(randn(Xoshiro(0xc401 + 4), 4, 4))`
  differ by up to 1.55 per entry (Julia 1.13.4). The convention is one draw in `Float64`
  (`ComplexF64` for a complex `T`), rounded to `T`. The rows that use the matrix compare the vendor
  factor with the own kernel in one `T`, so no result is wrong; a comparison across precisions on
  these matrices would compare two samples.
- **kind:** defect
- **found:** #113

### K122 · The vendor agreement matrix `cuda_well_conditioned` draws its random numbers in the element type, so each precision gets a different matrix

- **location:** `test/helpers/own_kernel.jl:33`, `:36`, `:37`, `:39` (`cuda_well_conditioned`)
- **evidence:** `randn(rng, T, n, n)`, `randn(rng, T, 16, 16)` and `randn(rng, T, n)` take the
  stream of `T`. With the seed of `n = 4`, `randn(Xoshiro(0x5eed + 4), Float32, 4, 4)` and
  `Float32.(randn(Xoshiro(0x5eed + 4), 4, 4))` differ by up to 2.90 per entry (Julia 1.13.4). The
  convention is one draw in `Float64` (`ComplexF64` for a complex `T`), rounded to `T`. The
  LU, QR and SVD agreement rows compare in one `T`, so no result is wrong. The function is on
  `main`.
- **kind:** defect
- **found:** #113

### K124 · The dense and device SVD solves of an `m < n` system do not scale a subnormal right-hand side up, so they lose the bits below the smallest subnormal

- **location:** `src/linear/linearproblem.jl:442` (`wide_solve!` of a `DenseSVD` or
  `DeviceSVD`, the same-type `m < n` solve of a dense or device `LinearSolver`)
- **evidence:** with `A = fill(2^kA, 1, 4)` and `b = [t]`, `t = nextfloat(zero(R))`, whose
  minimum-norm solution `2^(-kA - 2) t (1, 1, 1, 1)` lies in the normal range,
  `solve!(init(LinearProblem(A), SVDFactorization()), b)` gives `SUCCESS` and, in `Float64` and
  `ComplexF64` at `kA = -1000, -1026, -1062`, `x = 0` on an `Array` and a `JLArray`; in `Float32`
  and `ComplexF32` at `kA = -104, -130, -137`, `x = 0` on a `JLArray` and four times the solution
  (relative error `3.0`) on an `Array`. With `A = 2^(23 - emax) [2 1 0; 1 2 1]` and
  `b = t (1, -2)` the relative error is `0.289` on both routes in all four element types. `main`
  at `ae2a0db` gives the same on these routes (Julia 1.13.4). The static QR and SVD solves of
  `m < n` and the dense QR one lift such a right-hand side with the matrix: on PR #125 they meet
  `32 eps(R)` on the `1 × 4` cases and `2.2e-16` (`1.9e-7` in `Float32` and `ComplexF32`) on the
  `2 × 3` one, as do the square and tall solves.
- **kind:** defect
- **found:** 2026-11-25, the K114 5 × 5 case; the wide case from a review of PR #115

### K125 · The device `SVDFactorization(Float32)` of a `Float64` or `ComplexF64` matrix of subnormal entries gives `SINGULAR`

- **location:** `src/linear/ka/svd.jl:236` (`ka_factorize_svd!`)
- **evidence:** `ka_factorize_svd!` divides `J` by `2^e` with `e = matrix_exponent(J)`, which is
  clamped to `-exponent(floatmax)`, `-1023` in `Float64`; the largest entry of `J / 2^e` is then
  near `2^-43`, not in `[1, 2)`. With `A = 2^-1070 · A₀`, `A₀` the `4 × 4` and `5 × 3` matrices
  of the testset "a subnormal matrix" of `test/linear/linearproblem.jl` (`κ₁ ≤ 16`), a
  `LinearProblem` on a `JLArray` with `SVDFactorization(Float32)` gives `SINGULAR` in `Float64`
  and `ComplexF64`, where the static and the host factors give `SUCCESS` within
  `32 · eps(Float32) · κ₁` (Julia 1.13.4). The static SVD divides by the
  unclamped exponent of a subnormal largest entry (`svd_exponent` of
  `src/linear/static/svd.jl`). That the Jacobi sweep of the device kernel, in `Float32`, fails on
  products of column norms below `floatmin(Float32)` is not verified. The testset marks these
  rows `@test_broken`.
- **kind:** found late
- **found:** 2026-11-25, issue #122

### K126 · `DogLeg` returns `SUCCESS` with `status.stationary == true` at a saddle of `‖F‖²` that the Gauss–Newton model reads as a minimum

- **location:** `src/nonlinear/solver.jl:327` (the stationarity test of `solve!!` after a step,
  which returns `SUCCESS`), `src/globalization/dogleg.jl:180` (the `LINESEARCH_FAILED` zero step
  at the floor of the radius)
- **evidence:** on the mirror-symmetric saddle problem (b) of the testset "a step that leaves a
  stationary start point returns the start point" in `test/nonlinear/leastsquares.jl`, with
  `GaussNewton(; globalization = DogLeg())`, six perturbations of the data by a few ULPs and
  `SVector`, `Array` and `JLArray` iterates, the solve returns the first iterate near the saddle
  (`u = 1`, `|v|` from `6.0f-6` to `1.2f-5` in `Float32` and from `5.6e-12` to `2.2e-11` in
  `Float64`, where the minima are near `v = ±1`), with `SUCCESS`, `stationary == true`,
  `iterations == 2` and `step_failures == 1` in 9 of the 18 `Float32` solves (perturbations 1–3)
  and 15 of the 18 `Float64` solves (perturbations 1–5); the other solves reach a minimum near
  `v = ±1`. The tree of `8eddb7c` gives the same outcome in each of the 216 solves of the six
  methods of that testset (Julia 1.13.4). The model reduction of the step from the first iterate
  is at the rounding level of `‖r‖`, so `DogLeg` rejects every trial down to the floor of its
  radius and returns `LINESEARCH_FAILED` with a zero step. `solve!!` stops on such a step only
  where the stationarity test `‖Jᴴ F‖ / (‖J‖_F ‖F‖) ≤ g_tol` fails, and that test holds at the
  saddle, so the solve ends on it with `SUCCESS`. That test reads no curvature, so it cannot tell
  the saddle from a minimum. The testset asserts for each `DogLeg` solve only that its status
  holds for the point it returns, not that it reaches a minimum.
- **kind:** defect
- **found:** 2026-11-25, the review of the branch that returns the step start only where the
  discarded step does not lower `‖F‖`

### K127 · The static `SVDFactorization` of a `0 × 0` matrix throws `BoundsError`

- **location:** `src/linear/static/svd.jl:227` (`factorize!!` of `StaticSVD`, the cutoff
  `rank_tolerance(…) * σ[1]`)
- **evidence:** `factorize!!(nothing, m, SMatrix{0, 0, Float64}())` throws
  `BoundsError: attempt to access Tuple{} at index [1]` at that line, for `m = SVDFactorization()`
  and `SVDFactorization(Float32)` (Julia 1.13.4). On `bf08120` the same calls throw earlier,
  `ArgumentError: reducing over an empty collection`, from `matrix_exponent` in `svd_matrix`;
  `svd_exponent` returns 0 for an empty matrix. The static LU, QR, LDLᵀ and Cholesky factors of
  a `0 × 0` matrix give `SUCCESS`.
- **kind:** found late
- **found:** 2026-11-25

### K128 · A line search returns `NONFINITE` at a start whose residual norm is finite but above `√floatmax`, because its merit `norm2(F)` overflows

- **location:** `src/globalization/linefunction.jl:85` (`start_merit`; the trial merit at `:36`)
- **evidence:** `NonlinearProblem((F, x, _) -> (F .= sinh.(x); nothing))` from
  `fill(50.0f0, 3)`: `‖F‖ = √3 sinh(50) ≈ 4.5e21` is finite in `Float32`, and
  `norm2(F) = Inf32`. `solve` with `Newton(; globalization = Backtracking())` and with
  `Bisection()` returns `NONFINITE` after 1 iteration with a finite iterate; with `Static(1.0)` it
  returns `SUCCESS` after 8 iterations (`origin/main` at `e7af5e7`, Julia 1.13.4). The same start
  as `StationarityProblem((x, p) -> sum(cosh, x))` gives `NONFINITE` with `GMRES` and with
  `MINRESQLP` under both line searches.
- **kind:** defect
- **found:** 2026-11-25

### K129 · `_pinv_adj_mul_r1` does not scale a subnormal `w` up, so the products `Vᴴ w` of the Golub–Pereyra Jacobian run on subnormal numbers

- **location:** `src/nonlinear/eliminate.jl:523` and `:546` (`_pinv_adj_mul_r1` of
  `StaticRectSVD` and `StaticSVD`)
- **evidence:** with `A₀ = [4 1 0.5; 1 5 0.25; 0.5 0.25 3]`, `Φ = 2^k A₀` and
  `w = 2^k (1, -2, 0.5)`, `_pinv_adj_mul_r1(factorize!!(nothing, SVDFactorization(), Φ), Φ, w)`
  at `k = -1060` in `Float64` (`k = -135` in `Float32`) has a relative error against the
  `BigFloat` `Φᴴ \ w` of `2.89e-5` to `2.91e-5`, and `1.73e-5` to `1.74e-5` for the `3 × 2`
  matrix `A₀[:, 1:2]` and `w = 2^k (1, -2)` against `Φ (Φᴴ Φ)⁻¹ w`, in the four element
  types (`Φ = 2^k (1 + i/2) A₀` in the complex ones), the same values on `bf08120`, on
  `091997c` and on `c87aca8` (Julia 1.13.4). At
  `k = -1026` (`-130`) the error is below `1.1e-15` (`5.8e-7`). The coefficients `vₖᴴ w` hold
  only `k - exponent(nextfloat(zero(R)))` significant bits, 14 at `k = -1060` (`-135`). The
  testset "(Φ⁺)ᴴ w from a static SVD at every scale exponent" of `test/nonlinear/eliminate.jl`
  bounds this by `8 · 2^(exponent(nextfloat(zero(R))) - k)` beside `8 eps(R) κ`.
- **kind:** defect
- **found:** 2026-11-25, the fix of `_pinv_adj_mul_r1` for a static SVD exponent below
  `-exponent(floatmax)`

### K130 · The R3 `_pinv_adj_mul_r3!` of a `DenseSVD` in `Float32` of a `Float64` matrix near `2^-130` gives `NaN`

- **location:** `src/nonlinear/eliminate.jl:1296` (`_pinv_adj_mul_r3!` of `DenseSVD`)
- **evidence:** the method forms `scale = ldexp(one(R), -fac.scale_exponent)` with
  `R = real(eltype(fac.tmp))`, `Float32` for `SVDFactorization(Float32)`. For
  `Φ = 2^-130 A₀` in `Float64`, `A₀ = [4 1 0.5; 1 5 0.25; 0.5 0.25 3]`, the factor has
  `scale_exponent = -128`, `2^128` is `Inf` in `Float32`, and
  `_pinv_adj_mul_r3!(out, factorize!!(init_linear(m, nothing, w), m, Φ), w)` with
  `w = 2^-130 (1, -2, 0.5)` gives `[NaN, -Inf, NaN]`, on `bf08120` and on `091997c`
  (Julia 1.13.4). The `DeviceSVD` method at `:1311` forms the same product; that it gives `NaN`
  is not verified.
- **kind:** defect
- **found:** 2026-11-25

### K131 · A solve with `Steihaug` does not converge to a root where the Hessian of `S` is indefinite, also from a start close to it, and ends with `LINESEARCH_FAILED`

- **location:** `src/globalization/steihaug.jl:173` (the conjugate gradients on the model of `S`
  in `steihaug_step!!`; the acceptance on `‖F‖` at `:192`)
- **evidence:** the indefinite problem `indefinite_problem` of `test/nonlinear/symkrylov.jl`
  (Hessian eigenvalues at the root −0.998, −0.412, 1.78, 2.30), `Float64`,
  `Newton(; linsolve = MINRESQLP())`, 20 random starts (`Xoshiro(1)`) at each distance 1e-2,
  1e-4 and 1e-6 from the root: `DogLeg()` gives `SUCCESS` 20/20 at each distance, `Steihaug()`
  gives `LINESEARCH_FAILED` 20/20 at each distance (Julia 1.13.4, `--check-bounds=auto`, at
  32374d0). The conjugate gradients minimise the model `q(d) = Fᴴd + ½dᴴJd` of `S`, and a trial
  is accepted on `‖F‖`: a step along a direction of negative curvature decreases `q` and can
  increase `‖F + J d‖`, so `pred ≤ 0` and every trial is rejected down to the radius `10⁻⁶`.
- **kind:** defect
- **found:** #121; the limit of the decided form, conjugate gradients on the model of `S`

### K132 · After a rejected interior step, `Steihaug` and `DogLeg` evaluate `F` again at the same point, until the radius falls below the norm of the step

- **location:** `src/globalization/steihaug.jl:211` (`run = !(interior && dnorm < δ)` in
  `steihaug_step!!`; in `DogLeg`, `src/globalization/dogleg.jl:195` and `:316`)
- **evidence:** one step of the atan action (residual `atan.(x)`) from `fill(1.5, 3)` with
  `MINRESQLP(; forcing = FixedForcing(1e-3))`, radius 100 and the `JVPFunction`
  `v ./ (1 .+ x .^ 2)`, `Float64`: `Steihaug` and `DogLeg` each evaluate `F` 4 times at 2
  distinct points, 3 times at the rejected Newton step (`x₁ = −1.694…`) and once at the accepted
  boundary step of norm 1.5625; `f_calls` grows by 4 in the step (Julia 1.13.4, at 32374d0). The
  testset "The radius rules of Steihaug, $T" in `test/nonlinear/symkrylov.jl` asserts this step,
  issue #123.
- **kind:** defect
- **found:** #121, issue #123

### K133 · `Continuation` from a start at a fold of its homotopy returns `STALLED` with no step, although a path to `λ = 1` leaves that start

- **location:** `src/composition/continuation.jl` (`solve!!`, the start's corrector and tangent)
- **evidence:** `solve(NonlinearProblem((x, p) -> x^3 - 3x - 5/2), 1.0,
  Continuation(NewtonHomotopy()))` returns `STALLED` at `x = 1.0` with 0 iterations and an empty
  record, in `Float32` and `Float64`. At `x₀ = 1` the derivative `3x² − 3` is zero, so `H_x` is
  singular, and the start's bordered system along `t = e_λ`, `[H_x F(x₀); 0 1]`, is singular too.
  The Newton-homotopy path leaves `(1, 0)` along `x` (Julia 1.13.4).
- **kind:** found late
- **found:** 2026-11-25; the limit of the decided start: a start whose bordered system at
  `(x₀, 0)` along `e_λ` is singular (a fold of `λ` at the start) returns `STALLED` with no step
  (the user's decision on part CO, 2026-11-25)

### K134 · `Newton()` returns `STALLED` from a start whose residual is already at round-off, because its default residual test is relative to `‖F(x₀)‖` alone

- **location:** `src/base/options.jl` (the defaults `f_abstol = 0`, `f_reltol = √eps`)
- **evidence:** `F(x) = A x − b` with `A = T[3 1; 1 2] / 3` and `b = T[1, −3] / 3`, whose root lies
  within a few ulps of `(1, −2)`, solved with `solve(NonlinearProblem(F!), x₀, Newton())` from the
  289 starts `x₀ = (1, −2) + (k₁ eps(T), 2 k₂ eps(T))`, `k₁, k₂ ∈ −8:8`: all 289 return `STALLED`,
  in `Float32` and in `Float64` (`origin/main` at `e7af5e7`, Julia 1.13.4). There `‖F(x₀)‖` is of
  the order of `eps`, so the test `‖F‖ ≤ √eps ‖F(x₀)‖` asks for a residual below the round-off of
  `F`, which no step reaches. The last corrector of a `Continuation` runs this `Newton` from a
  point near the root, so a path that ends within round-off of the root can return `STALLED` too.
- **kind:** found late
- **found:** 2026-11-25

### K135 · `Continuation` does not reach the root of the linear check `F(x) = s (A x − b)` at a subnormal scale `s`, where `Newton()` reaches it in `Float64`

- **location:** `src/composition/continuation.jl:178` (`homotopy_residual!`; the fixed-point form
  `λ F(x)/σ + (1 − λ)(x − x₀)` at `:188`)
- **evidence:** `A = [3 1; 1 2]`, `b = [1, −3]`, root `(1, −2)`, start `zeros(T, 2)`,
  `f_abstol = 0`, at `s = 2^20 · nextfloat(zero(T))`. `Float64`:
  `Continuation(FixedPointHomotopy())` returns `MAXITERS` at `(0.263, −0.628)`,
  `Continuation(NewtonHomotopy())` returns `STALLED` at `(6.4e-7, −1.2e-6)`, and `Newton()` returns
  `SUCCESS` at `(1.0000000000000002, −2.0)`. `Float32`: `Continuation(NewtonHomotopy())` returns
  `STALLED` at `(1.2e-4, −1.4e-4)`; `Continuation(FixedPointHomotopy())` and `Newton()` return
  `SUCCESS` with a relative error of about `1e-5` and `1e-4`. At `s = 1e-8`, `1e-20` and
  `4 floatmin(T)` all three return `SUCCESS` at `(1, −2)` in both types (Julia 1.13.4, at
  35ed7e8).
- **kind:** found late
- **found:** 2026-11-25; the homotopy forms that the part's spec decides (its clause 3), at a
  scale outside those that its clauses and the scale row name

### K136 · The docstrings of `stored_factors` and `promoted_factors` and the regimes page name `GMRES` as the matrix-free solve that holds no factor, and leave out `MINRESQLP`, `BiCGStab` and `DQGMRES`

- **location:** `src/nonlinear/newton.jl:1299` (`stored_factors`), `:1315` (`promoted_factors`);
  `docs/src/manual/regimes.md:147`
- **evidence:** each line names "a matrix-free [`GMRES`](@ref) solve" as the solve that holds no
  factor. A `Newton` with any linear method for which `matrix_free` is `true` holds no factor.
  The two docstring lines read the same at `9023f76` (`newton.jl:1006`, `:1022`), where
  `MINRESQLP` already held no factor. The wording "a matrix-free solve ([`matrix_free`](@ref))"
  covers every such method.
- **kind:** docs
- **found:** 2026-11-26

### K137 · A comment in the GMRES tests says that no linear method other than `GMRES` takes the `forcing` and `preconditioner` keywords of `Newton`

- **location:** `test/nonlinear/krylov.jl:1374`
- **evidence:** the comment reads "the keywords configure a GMRES, and no other linear method takes
  them". `MINRESQLP` takes `forcing`, and `BiCGStab` and `DQGMRES` take both
  (`src/nonlinear/newton.jl:307–310`, `with_forcing`). The comment was already false at `9023f76`,
  for `MINRESQLP`. The two `@test_throws` below it use the default `LUFactorization()` and hold.
- **kind:** docs
- **found:** 2026-11-26

### K138 · A matrix-free Newton step with a `Float32` iterate and the default `Static()` or `Backtracking()` infers `Union{StepInfo{Float32}, StepInfo{Float64}}`

- **location:** `src/globalization/static.jl:20` (`Static(α::Real = 1.0)`),
  `src/globalization/backtracking.jl:46` (`Backtracking(; c₁ = 1e-4, p = 0.5)`)
- **evidence:** the default constructors fix `Float64` parameters whatever the type of the iterate.
  `globalize_merit!!` with `Static{Float64}` and a `Float32` iterate returns `StepInfo{Float64}`,
  while the early returns of the step build `StepInfo{Float32}`, so `code_typed` of `step!!` and
  of `gmres_step!!` (`src/nonlinear/newton.jl:2388` for `GMRES`, `:2635` for `BiCGStab` and
  `DQGMRES`) infers the union. With `Static(1f0)` or `Backtracking(; c₁ = 1f-4, p = 0.5f0)` each
  step infers `StepInfo{Float32}`. `solve` infers `Tuple{Vector{T}, SolverStatus{T}}` in every case,
  and the whole solve allocates 0 bytes for `BiCGStab` and `DQGMRES` × `Float32`, `Float64` ×
  `Static()`, `Backtracking()` in a fresh process with `--check-bounds=auto` (Julia 1.13.4, at
  b2c3bba). The cause is in the two default constructors and not in a step, so the `GMRES` step
  infers the same union.
- **kind:** defect
- **found:** PR #127

### K139 · `DeviceCholesky` of a `JLArray{BigFloat}` throws `InvalidIRError` while it compiles the load kernel

- **location:** `src/linear/ka/cholesky.jl:36` (`ka_cholesky_load_kernel!`)
- **evidence:**

  ```julia
  m = CholeskyFactorization()
  factorize!!(dense_factor(m, nothing, JLArray(BigFloat[1, 2]), nothing), m,
      JLArray(BigFloat[3 1; 1 3]))
  ```

  throws `InvalidIRError` compiling `gpu_ka_cholesky_load_kernel!`, with the reasons
  "unsupported call to an external C function" (`mpfr_set`, `mpfr_mul_2si`, …) and "unsupported
  calls into the Julia runtime", on `ae2a0db` and on the head of PR #125 (Julia 1.13.4).
  `BigFloat` is not `isbits`, and its arithmetic calls MPFR, which a kernel cannot call. The same
  calls give `SUCCESS` for `Float16`, and for `Float32` with `CholeskyFactorization(Float16)` and
  `Float16` with `CholeskyFactorization(Float32)`.
- **kind:** defect
- **found:** a review of PR #125

### K140 · `_pinv_adj_mul_r1` of a subnormal `Φ` and a normal `w` gives `NaN` where the true `(Φ⁺)ᴴ w` has entries above `floatmax`

- **location:** `src/nonlinear/eliminate.jl:523` and `:546` (`_pinv_adj_mul_r1` of
  `StaticRectSVD` and `StaticSVD`)
- **evidence:**

  ```julia
  A₀ = [4.0 1.0 0.5; 1.0 5.0 0.25; 0.5 0.25 3.0]
  Φ = SMatrix{3, 3, Float32}(Float32.(A₀) .* ldexp(1.0f0, -130))
  w = SVector{3, Float32}(1, -2, 0.5)
  GeometricSolvers._pinv_adj_mul_r1(factorize!!(nothing, SVDFactorization(), Φ), Φ, w)
  ```

  gives `[NaN, -Inf, NaN]` on `091997c`; the `BigFloat` `Φᴴ \ w` is `[Inf, -Inf, 2.01e38]` in
  `Float32`, and `bf08120` gives `[Inf, -Inf, Inf]`. In `Float64` at `2^-1026` the same holds:
  `[NaN, -Inf, NaN]` on `091997c` against `[Inf, -Inf, 1.06e308]`, `[Inf, -Inf, Inf]` on
  `bf08120`. At `2^-128` (`2^-1024`) the result is finite and agrees with the reference to six
  (fifteen) digits on `bf08120` and on `091997c` (Julia 1.13.4). Two entries of the true result
  lie above `floatmax`, so no finite result is right; the finite third entry is lost on `bf08120`
  and on `091997c`.
- **kind:** defect
- **found:** 2026-11-25, the fix of `_pinv_adj_mul_r1` for a static SVD exponent below
  `-exponent(floatmax)`

### K141 · A `ForwardDiff.Dual` matrix of subnormal values and partials 1 gives `NaN` partials where the true derivative lies above `floatmax`

- **location:** `src/linear/dense.jl:204` (`load_factors!`), `src/linear/factors.jl:802`
  (`factor_matrix`)
- **evidence:**

  ```julia
  D(x, p) = ForwardDiff.Dual{Nothing}(x, p)
  A₀ = [4 1; 1 5] .// 8
  s = 2.0^-1060
  J = SMatrix{2, 2}(D.(Float64.(s .* A₀), 1.0))
  b = SVector{2}(D.(Float64.(s .* [1, -2]), 0.0))
  m = CholeskyFactorization()
  ldiv!!(nothing, factorize!!(nothing, m, J), b)
  ```

  gives the values `(2.9473684210526314, -3.789473684210527)` and the partials `(NaN, NaN)` on
  PR #125, where the solve lifts `b` with the matrix; the `DenseCholesky` of `Matrix(J)` gives
  the same, the `DenseLU` the values `(2.9473684210526314, -3.789473684210526)` and the
  partials `(NaN, NaN)`, and the static LU the same values and the partials `(NaN, -Inf)`. On
  `ae2a0db` both Cholesky factors give the partials `(-Inf, Inf)`, the static LU `(Inf, -Inf)`
  with the first value `2.9473876953125`, and the `DenseLU` `NaN` values and partials (Julia
  1.13.4, ForwardDiff 1.5.0). The partial 1 is not scaled with the subnormal value, so the true
  derivative, `(1.75e319, 1.31e319)` in `BigFloat`, lies above `floatmax`, and no finite partial
  is right; each `-Inf` above has the wrong sign.
- **kind:** defect
- **found:** 2026-11-26, PR #125

### K142 · Inside an Apple GPU kernel, `matrix_exponent` and `cholesky_exponent` give `e = 0` for a matrix of subnormal `Float32` entries, so an R1 static factor there takes no subnormal scale

- **location:** `src/linear/factors.jl:842` (`matrix_exponent`), `:861` (`cholesky_exponent`)
- **evidence:** both return `isfinite(m) && !iszero(m) ? clamp(exponent(m), -emax, emax) : 0`,
  and in a Metal kernel `iszero` of a subnormal `Float32` is `true`: with
  `m = reinterpret(Float32, 0x00010000)` in a kernel, `iszero(m)`, `isfinite(m)` and `m == 0f0`
  are all `true`, while the entry reaches the kernel intact (`0x00010000` on the device and the
  host). A static `SVDFactorization()` of `2^-134 · [i + j]` (`n = 1, 4, 16`, `Float32` and
  `ComplexF32`) inside a Metal kernel gives `scale_exponent = 0` and rank 0, where the host gives
  `-133`, `-131`, `-129` and rank 1, 2, 2; `matrix_exponent` gives 0 in the kernel and `-127` on
  the host (Julia 1.13.4). `origin/main` at `0999817` has the same lines and the same result.
  `test/devices/metal_svd.jl` passes on Metal because it checks the subnormal factor on the host
  only. The load kernel of `DeviceCholesky` reads the bits (`abs_bits`, `bits_exponent`) and is
  not affected.
- **kind:** defect
- **found:** 2026-11-26, the pre-PR verification of PR #125

### K143 · A `StationarityProblem` evaluates a callable datum at the parameters `p` of the solve, which a `ParameterHomotopy` maps before its problem reads them

- **location:** `src/nonlinear/stationarity.jl` (`_init_stationarity_state`, the `solve!!` of a
  `DenseStationarityState` and `_solve_stationarity_solver!`, which evaluate `b(p)` to check its
  length and fill the datum buffer)
- **evidence:** With `h = ParameterHomotopy((p, λ) -> p.a + λ * (p.b - p.a))` and
  `p = (a = 0.5, b = 2.0)`, `solve(StationarityProblem((x, q) -> x^4 / 4 + x^2 / 2; b = q -> q),
  0.0, Continuation(h), p)` raises `MethodError: no method matching _static_datum(…, ::Float64,
  ::@NamedTuple{a::Float64, b::Float64})`, and `solve(StationarityProblem((x, q) -> sum(abs2, x) /
  2; b = q -> [q, q]), zeros(2), Continuation(h), p)` raises `MethodError: Cannot convert an
  object of type @NamedTuple{a::Float64, b::Float64} to an object of type Float64`. The path
  itself evaluates `b` at `map(p, λ)` only; a map with `map(p, 0) = p`, such as
  `(p, λ) -> p + 3λ / 2`, solves (Julia 1.13.4).
- **kind:** defect
- **found:** PR #126, the round that evaluates a callable datum at `map(p, λ)`

### K144 · A matrix-free Newton step whose inner solve stops short of the forcing term `η` passes `InexactStep(η)`, not the ratio it attained, to the line search

- **location:** `src/nonlinear/newton.jl:2363` (`GMRES`), `:2604` (`MINRESQLP`), `:2654`
  (`BiCGStab` and `DQGMRES`), each `η_ach = min(resnorm / fnorm, η)`
- **evidence:** `InexactStep(η)` states `‖r + J d‖ ≤ η ‖r‖`, so a smaller `η` than the attained
  ratio asks `Backtracking` for more decrease than the direction gives. With
  `F(x) = A x − b`, `A = [1e-3 1; −1 1e-3]`, `b = [1, 0]`, `x₀ = 0`, `Backtracking()` and
  `GMRES(; maxiter = 1, forcing = FixedForcing(0.5))`, the solve ends `STALLED` after 2 iterations
  at `fnorm = 1.0` (Julia 1.13.4). The attained ratio in its place, `η_ach = resnorm / fnorm`, is
  not a one-line fix: a ratio of one or more promises no decrease, and the search rejects every
  step, so the testset "The Newton step goes on after the BiCGStab guard on ω" of
  `test/nonlinear/fixedkrylov.jl`, whose direction has the ratio 1 and decreases `‖F‖`, ends
  `LINESEARCH_FAILED` instead of `SUCCESS` with it. A fix chooses what a step with a ratio at or
  above `η` passes, for example the measured slope, for all four methods together.
- **kind:** defect
- **found:** PR #127, its review and the CodeRabbit comment on the BiCGStab and DQGMRES step

### K145 · The test of the summary of the standard test set finds the line of a column by its name and a space, so a column whose name is another column's name, a space and more matches two lines

- **location:** `test/verification/standard.jl:409`
- **evidence:** `only(filter(startswith(column * " "), split(summary, "\n")))` raises
  `ArgumentError: Collection has multiple elements, must contain exactly 1 element` for the
  column `Newton Backtracking GMRES` once a column named `Newton Backtracking GMRES Float32 basis`
  is in `STANDARD_COLUMNS`: both summary lines start with `Newton Backtracking GMRES `. Observed on
  a draft of the branch that adds the `Float32`-basis column, which then names it
  `Newton Backtracking Float32-basis GMRES`, so no column of the table meets the case.
- **kind:** defect
- **found:** 2026-11-26

### K146 · On a reduced basis, the CGS2 step does not see a Krylov space that is invariant to the round-off of the basis type, and the rest of the cycle makes no progress

- **location:** `src/linear/gmres.jl:691` (`cgs2_reduced_step!`)
- **evidence:** with `J = 14 I`, `b = fill(-9.0, 100)`, `GMRES(; maxiter = 20, basis_eltype =
  Float16)` and `tol = 1e-10`, the stored `v₁` has `‖v₁‖² = 1 − 4.9e-4`, and the two CGS2 passes
  leave `hj = 3.3e-6` along `v₁`, a Givens estimate of `2.4e-7 ‖b‖`; the step stores `v₂ = v₁`
  bit for bit, classical Gram–Schmidt over the two equal columns doubles the projection, and the
  subdiagonal of the Hessenberg matrix grows to `1.1e3` at `j = 10` and `1.2e4` at `j = 30`,
  where `‖J v_j‖ = 14`. The cycle ends at its cap on a true residual of `2.4e-4 ‖b‖` without
  convergence; with `restart = 10` the restart converges at 12 iterations (Julia 1.13.4). With a
  `Float32` basis the remainder, `1.2e-14`, lies below the tolerance, and the solve converges in 2.
  The MGS step with its second pass converges in 4 on the same system. The docstring of `GMRES`
  states the limit, and the testset "A CGS2 cycle on a Krylov space invariant to a Float16 basis
  makes no progress, and a restart recovers" of `test/nonlinear/krylov.jl:2199` asserts it. A
  test of `hj` against the round-off of the stored basis would see the breakdown.
- **kind:** defect
- **found:** PR #129, the second fix round, where the remedy of the review's MGS finding was
  decided as a second MGS pass and this limit as a documented one

### K147 · With `regularization = μ > 0`, a matrix-free Newton step passes the residual ratio of the regularized system `J + μ I` as `InexactStep`, and a `DogLeg` on `GMRES` models `J + μ I`, so `Backtracking` stalls where the dense `Newton` makes progress

- **location:** `src/nonlinear/newton.jl:2363` (`GMRES`), `:2604` (`MINRESQLP`), `:2654`
  (`BiCGStab` and `DQGMRES`), each `η_ach` from the inner solve on `J + μ I`
  (`RegularizedJVP`, `:2307` and `:2505`); `:2400` (`dogleg_step_mf`, on the Arnoldi relation of
  that solve; read, not measured)
- **evidence:** `InexactStep(η)` states `‖F + J d‖ ≤ η ‖F‖`; the inner solve bounds
  `‖F + (J + μ I) d‖` instead, and `‖F + J d‖` exceeds it by up to `μ ‖d‖`. With
  `F(x) = x − 1` on three entries, `x₀ = 0`, `regularization = 10.0` and
  `Backtracking(; c₁ = 0.5)`, the dense `Newton` ends `MAXITERS` after 100 iterations at
  `fnorm = 1.3e-4`, and `GMRES`, `BiCGStab` and `DQGMRES` with `FixedForcing(1e-12)` each end
  `STALLED` after 2 iterations at `fnorm = √3`, the residual of `x₀` (Julia 1.13.4). `main` at
  907d20b gives the same for the dense `Newton` and `GMRES`. The line searches that evaluate the
  slope take the product of `J` alone (`merit_jvp`, `:1597`), so `Bisection` and `StrongWolfe` do
  not have this limit. A fix chooses the step kind of a regularized direction, such as the measured
  slope or the attained ratio of `J` at one more product, and is a choice for all four methods
  together with K144.
- **kind:** defect
- **found:** PR #127, the CodeRabbit comment on the regularized product of the matrix-free step;
  the part on `Bisection` and `StrongWolfe` is fixed there

### K150 · `solve!` on a `ReparametrizedSolver` takes no `refactorize` keyword, so a `Newton(; reuse = k, keep_factor = true)` inside it refreshes its kept factor only by the rate rule

- **location:** `src/nonlinear/reparametrized.jl:1061`
- **evidence:** `solve!(solver::ReparametrizedSolver)` and `solve!(solver::ReparametrizedSolver, y,
  p = solver.p)` take no keywords, so `solve!(solver, y, p; refactorize = true)` raises a
  `MethodError`. The `solve!` of a `NonlinearSolver` takes `refactorize`. A caller that changes the
  step size between two reparametrized solves cannot drop the kept factor: a step of rate `θ ≥ 1`
  along it refreshes it, and a solve keeps it only after `SUCCESS` in one step or with a last rate
  `θ ≤ 10⁻³`.
- **kind:** defect
- **found:** 2026-11-26

### K151 · A `Newton(; reuse = k, keep_factor = true)` solve that returns `SUCCESS` at its start, with no step, drops the kept factor, so the next solve factorises at its first step

- **location:** `src/nonlinear/newton.jl:1406` (`keeps_factor`)
- **evidence:** `keeps_factor` keeps the factor after `SUCCESS` only with `status.iterations == 1`
  or a last rate `θ ≤ 10⁻³`; a solve of no step has `iterations == 0` and the rate `NaN`, which
  fails both. With the midpoint residual of `test/nonlinear/stopping.jl`,
  `Newton(; jacobian, reuse = 10, keep_factor = true)` and `Options(Float64; f_abstol = 0.5,
  max_iterations = 1)`: a solve from `q₀ .+ 1` returns `SUCCESS` in one step with the factor age
  `1`; a solve from `q₀`, whose residual passes the test, returns `SUCCESS` with `0` iterations and
  leaves age `0`; the next solve from `q₀ .+ 1` factorises once at its first step. An integrator
  whose start is already converged refactorises at the next step. The rule follows the decided edge
  "after a `SUCCESS` the factor's age survives only after one iteration or a last `θ ≤ 10⁻³`" of
  the part's spec, which does not name a solve of no step.
- **kind:** found late
- **found:** 2026-11-27, the review of the branch that keeps a `Newton` factor across solves; the
  limit of the decided rule for a kept factor after `SUCCESS`

### K152 · A `Newton(; reuse = k)` solve with `k > 1`, or with a kept factor, of a least-squares problem with a nonzero residual at its minimum often returns `STALLED` near the stationary point

- **location:** `src/nonlinear/solver.jl:84` (`rate_test_step`)
- **evidence:** the steps along a reused factor converge to the fixed point of the frozen
  Gauss–Newton map, which is not the stationary point, and the rate test reads no such step. With
  `r(x) = (x² − 5, x − 1)`, its Jacobian, `Newton(; linsolve = QRFactorization(), jacobian, reuse
  = k)`, default `Options(T)`, from `3`: in `Float64`, `k = 1` returns `SUCCESS` by
  `:stationarity` at `2.1748339274858415` after 7 iterations, `k = 3` returns `STALLED` at
  `2.17483438331406` after 12, and `k = 10` returns `STALLED` at `2.17527521960677` after 17. In
  `Float32`, `k = 3` returns `SUCCESS` by `:stationarity` and `k = 10` returns `STALLED` after 13.
  A `VariableProjection` with `reuse = 3` or `10` also returns `STALLED`. The step test of the base
  stalls in the same way.
- **kind:** defect
- **found:** 2026-11-27, the review of the branch that keeps a `Newton` factor across solves; the
  limit of the decided rule that the rate test of a least-squares problem reads no step along a
  reused factor

### K153 · `solve!` on a static `ReparametrizedSolver` of a `LeastSquaresProblem` raises a `MethodError`

- **location:** `src/nonlinear/reparametrized.jl:955`
- **evidence:** with `r` and its Jacobian of K152 as static vectors, and the identity chart
  `id(y, p) = y`, `did(y, p) = SMatrix{1, 1}(one(y[1]))`,
  `s = init(Reparametrized(LeastSquaresProblem(r, j), id, did), SVector(3.0), Newton(; linsolve =
  QRFactorization(), reuse = k); options = Options(Float64))` followed by
  `solve!(s, SVector(3.0))` raises `MethodError: Cannot convert an object of type
  GeometricSolvers.StaticReparamState{…}` at the assignment `solver.state = state₊`, for `k = 1`
  and `3`. The same pair with a square `NonlinearProblem` returns `SUCCESS`, and `solve` of the
  least-squares problem returns a status. The error is the same on the commit that the branch
  that keeps a `Newton` factor across solves starts from.
- **kind:** defect
- **found:** 2026-11-27

### K160 · A run of full-length secant steps along the pairs of a `BroydenUpdate` can pass the rate test away from a root, because the larger of the last two rates guards one irregular step only

- **location:** `src/nonlinear/solver.jl:348` (the rate test of `solve!!`), `src/base/options.jl:281`
  (`rate_converged`)
- **evidence:** `scripts/broyden_rate_test.jl` solves `F(x) = g(x − X)`, with `g(t) = 1 − t + θ t²`
  up to a kink at each method's own second iterate and a line of slope `−σ` beyond it. In
  `Float32`, `Newton(; reuse = 3, update = BroydenUpdate(2))` with `Static()` or `Backtracking()`
  returns `SUCCESS` with `contracted = true` after 3 iterations: at `X = 1e3`, `θ = 0.025`,
  `σ = 0.05` with `fnorm = 6.298e-4`, above the residual tolerance `3.45e-4`, at
  `|x − root| = 1.257e-2`; at `X = 3e3`, `θ = 0.03` with `fnorm = 8.345e-4` at
  `|x − root| = 1.660e-2`. `Newton(; reuse = 3)` reaches the root after 4 iterations on both rows.
  In `Float64`, at `X = 1e6` and `1e8`, `Newton(; reuse = 3)` stops early with
  `contracted = true` as well (`|x − root| = 1.592e-5` and `3.978e-4`), and the same on the base
  of #148 at a935115, so that half is a property of the rate test along a reused factor. A fix
  that skips a step along stored pairs in the rate test, as it skips a shortened step, needs the
  step to report such a step.
- **kind:** defect
- **found:** #148
