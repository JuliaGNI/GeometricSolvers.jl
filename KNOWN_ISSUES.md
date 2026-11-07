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

### K60 · `solve` with a combinator method has no implicit differentiation

- **location:** `src/composition/implicit.jl` (`solve_implicit`)
- **evidence:** `solve` with a `ForwardDiff.Dual` in `x₀` or `p` and a `Newton` or `Picard`
  method runs the primal solve and then the implicit-function-theorem tangent solve
  (`solve_implicit` in `src/nonlinear/implicit.jl`). The tangent solve takes its Jacobian
  preparation from `_tangent_jacprep`, which has a method for a `NewtonState` and a `PicardState`
  only. `solve` with an `Additive`, `Multiplicative`, `Repeat`, `Switch` or `Anderson` method
  therefore raises an `ArgumentError` for a `ForwardDiff.Dual` in `x₀` or `p`, before the primal
  solve; `init`, `solve!` and `solve_batch!` refuse a dual input for every method. Implicit
  differentiation through a combinator, which needs the Jacobian of `F` at `x*` and not the steps
  of the members, is for a later pull request.
- **kind:** defect
- **found:** PR #60

### K61 · A hard failure of one `Additive` member discards the whole step, also at weight zero

- **location:** `src/composition/additive.jl` (`step!!`)
- **evidence:** With `F(x) = -x - x³`, `x₀ = SVector(1.0, 0.5)` and
  `pic = Picard(; damping = 1, globalization = Backtracking())`, whose line search fails at every
  step, `solve(prob, x₀, Additive(Newton(), pic))` and
  `solve(prob, x₀, Additive(Newton(), pic; weights = FixedWeights(1.0, 0.0)))` both return
  `LINESEARCH_FAILED` after 1 iteration at `x₀`, while `Newton()` alone returns `SUCCESS` after 5.
  A `SINGULAR`, `NONFINITE` or zero-norm `LINESEARCH_FAILED` member ends the `Additive` step at
  `x` with `stepnorm = 0`, whatever its weight. Whether the step drops the failed candidates and
  combines the others, and how it then sets the weights, is open.
- **kind:** defect
- **found:** PR #60

### K62 · A combinator reports a `STALLED` member step as `SUCCESS`

- **location:** `src/composition/repeat.jl`, `src/composition/multiplicative.jl`,
  `src/composition/additive.jl` (`step!!`)
- **evidence:** With `F(x) = x² - 2` from `1.0`, all four tolerances `0`, `max_stalls = 3` and
  `nb = Newton(; globalization = Backtracking())`, `nb` returns `STALLED` after 8 iterations,
  `Switch(nb, nb)` after 8, `Repeat(nb, 1)` and `Multiplicative(nb)` after 6, and `Additive(nb)`
  after 4. `Repeat`, `Multiplicative` and `Additive` report `SUCCESS` for a step whose members
  returned `STALLED`, so `solve!!` applies the step test after a stall and does not count
  `max_stalls`; `Switch` passes the code of its active branch through. Whether the composite step
  is `STALLED` when every member step is, or when the last one is, is open.
- **kind:** defect
- **found:** PR #60

### K70 · R1 `Anderson` fails on an `MVector` iterate whose residual returns an `SVector`

- **location:** `src/composition/anderson.jl` (`AndersonState`, R1 `start!!` and `step!!`)
- **evidence:** The R1 state gives `x_prev`, `w_prev`, `x_best` and `Fx_best` one type `X`, and
  the R1 `step!!` takes `x::X, Fx::X`. With
  `prob = NonlinearProblem((x, p) -> SVector(x[1] - cos(x[1]), x[2] - sin(x[2]) - 0.1))`,
  `solve(prob, MVector(0.5, 0.5), Picard(; damping = 0.8))` returns `SUCCESS`, and the same call
  with `Anderson(Picard(; damping = 0.8), 2)` raises
  `MethodError: no method matching GeometricSolvers.AndersonState(::PicardState…, ::Nothing, ::MVector…`.
  With a residual that returns an `MVector`, `Anderson` converges. Whether the residual gets its
  own type parameter in the state or `init_state` refuses a mutable `StaticVector` iterate is open.
- **kind:** defect
- **found:** PR #63

### K71 · `Anderson`'s `M_alpha` safeguard underestimates `∑|α_j|` once a filter drops an interior column

- **location:** `src/composition/anderson.jl` (`weights_acceptable`)
- **evidence:** `weights_acceptable` computes
  ``|1 - γ_1| + \sum_c |γ_c - γ_{c+1}| + |γ_p|``, the sum of the affine weights when the kept
  columns are consecutive differences ``g_{k-j+1} - g_{k-j}``. After the angle filter drops an
  interior column, the kept columns are no longer adjacent, in that step and in every later one.
  With kept columns 1 and 3 and `γ = (1.0, 20/3)`, the affine weights are
  `α = (0, 1, -20/3, 20/3)` with ``\sum |α_j| = 14.33``, and the code computes `12.33`: with
  `M_alpha = 13` the step passes a bound that its weights exceed. The exact sum needs the age of
  each stored column in the state; whether to add it, or to document the quantity as the
  telescoped sum, is open.
- **kind:** defect
- **found:** PR #63

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

### K73 · The method-first `solve(prob, method, x₀, p, options)` exists for `Anderson` only

- **location:** `src/composition/anderson.jl` (`solve`)
- **evidence:** `solve(prob, Anderson(...), x₀)` forwards to `solve(prob, x₀, method)`, and no
  docstring describes the method-first order. `solve(prob, Picard(), x₀)` raises
  `MethodError: no method matching real(::Type{Any})`. Whether the method-first order is removed
  or defined and documented once for every method is open.
- **kind:** defect
- **found:** PR #63
