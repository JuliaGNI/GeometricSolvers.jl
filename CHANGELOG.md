# Release Notes

All notable changes to GeometricSolvers.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from an interface change.


## [Unreleased] — targeting 0.1.0

### Changed

- `Pkg.test()` no longer runs the doctests. `test/quality/doctests.jl` is now the group
  `doctests`, which an empty `ARGS` does not run; `Pkg.test(test_args = ["doctests"])` runs it.
  In CI the Doctests job stays their runner, so the test matrix no longer runs them a second
  time.

- CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of `Julia min`, and a test
  job saves the Julia cache only when it succeeds.

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
  exists. `K10` and `K11` stay open. `K10`: the Metal CI job runs with `--check-bounds=yes`, which
  hides this failure; with that flag the previous fill passes every `n` on Julia 1.13.2. `K11`:
  the Metal test sizes 1, 4, 16 and 24 reach only `n = 4` of the failing sizes.
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

### New Features

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
  stopping test only, built by `Options(T; kwargs...)` and converted to another `R` by `convert`,
  which raises a relative tolerance below the default of `R` to that default.
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
  kernel, and allocate nothing unless bounds checks are forced (`--check-bounds=yes`). The
  singular values `σ` are those of `J`, in non-increasing order. A singular value at or below
  `√n · eps(TF) · σ₁` counts as zero: `info` is the index of the first one,
  `GeometricSolvers.returncode(fac)` gives `SINGULAR`, and the solve leaves those directions out,
  so a rank-deficient matrix gives a finite truncated step and never a `NaN`. The matrix is first
  divided by a power of two, recorded as `scale_exponent`, so a subnormal matrix gives finite
  factors. A sweep that reaches the cap of 30 sweeps leaves `converged` false.

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

- Internal: R3 dense CPU linear solvers and mixed-precision solve. `init(m, prob, x, p)`
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
  AbstractGPUArray`, `init(m, prob, x, p)` builds `DenseLU` or `DenseQR` with `Int32` pivot and
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
  the internal `init`, `factorize!!` and `ldiv!!`, which a device-resident R3 `solve` would call.
  `norm2(a)` and `rdot(a, b)` call `sum` without an `init` keyword (`isempty(a) ? zero(R) :
  R(sum(...))`), so `Array` and `Broadcasted` reductions use Base's pairwise `_mapreduce` rather
  than sequential `mapfoldl_impl`. `scripts/spikes/metal_reductions/run.jl` measures `norm2` and
  `rdot` on `MtlVector{Float32}` at `n = 10⁶, 10⁷, 10⁸` against a `Float64` host reference on
  positive-mean (`rand`) and zero-mean (`randn`) inputs: every relative error stays within the
  `⌈log₂(n)⌉ · eps(Float32)` pairwise tree-reduction bound (`20`, `24` and `27 · eps(Float32)`),
  so no custom `KernelAbstractions` reduction kernel is needed.

- `PerSystem` and `solve!` (exported), together with the internal `GeometricSolvers.parameters`,
  run the R1 nonlinear solver loop across a batch of independent small systems in place on
  `get_backend(X)` through KernelAbstractions, one work-item per system. `PerSystem(data)` wraps an
  indexable collection of per-system parameters so system `i` receives `data[i]`; any parameter
  value not wrapped in `PerSystem` (including a plain `AbstractVector`, `Tuple` or `SVector`) is
  shared unchanged across every system, and `Adapt.@adapt_structure PerSystem` adapts `PerSystem`
  to device arrays.
  `solve!(X, S, prob, method, p = nothing; order = nothing, options = nothing, kwargs...) -> S`
  writes each system's `SolverStatus{R}` into `S[idx]` and returns `S`, raising `ArgumentError`
  when `R !== real(eltype(eltype(X)))`, or when `get_backend(S) != get_backend(X)`: a host `S` with
  a device `X`, or a device `S` with a host `X`, is refused, so no kernel writes a status into a
  host `S`.
  `solve!(X, prob, method, p = nothing; order = nothing, options = nothing, kwargs...)` allocates a
  1-byte-per-system `ReturnCode` buffer on `get_backend(X)`, copies the codes once to the host
  after `synchronize(backend)`, and returns the six outcome counts as a `NamedTuple`
  `(; SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED)`, for which
  `GeometricSolvers.isconverged(counts)` returns `counts.SUCCESS == sum(counts)`. On the host,
  `solve!` adapts `method` with `Adapt.adapt(ToReal{R}(), method)`, converts or constructs
  `Options{R}`, and builds `state₀ = init(m, prob, zero(eltype(X)), nothing)` once by value so
  `Newton(; reuse = k)` carries its chord factor across steps inside each work-item. When `order`
  is a permutation of `1:length(X)`, work-item `i` solves system `idx = order[i]`; an `isbits` range
  is passed by value, and a non-`isbits` host `order` vector passed with a device `X` is copied to
  `get_backend(X)` before kernel launch. A length mismatch in `S`, `PerSystem`, or `order` raises
  `DimensionMismatch`. An `order` of the right length that repeats an index or holds one outside
  `1:length(X)` raises `ArgumentError`; the check reads a device `order` on the host, one copy per
  call. Passing both `options` and keyword tolerances also raises `ArgumentError`.
  `scripts/r2_throughput.jl` benchmarks R2 `solve!` (`Newton()` and `Newton(; reuse = 4)`) against
  a threaded CPU R1 loop and reports the R4 crossover dimension `n`.
  `KernelAbstractions` (0.9.15 or later, the first release whose compat admits Adapt 4) is now a
  direct dependency of the package.

- `NonlinearSolver`, `init` and `solve!` (exported), together with `solve`, run `Newton` and
  `Picard` in R3 on an `AbstractVector` iterate (`Array` and device arrays), sharing `solve!!` and
  `step!!` with R1 through the storage seam. `init(prob, x, method, p = nothing; options, kwargs...)`
  adapts `method` to `R = real(eltype(x))` and builds a `NonlinearSolver` holding `prob`, the
  adapted `method`, `Options{R}`, the element type and length of `x`, and the pre-allocated
  workspace state without mutating `x`. A
  `StaticArray` `x` raises `ArgumentError`, because `solve!` cannot write the solution into it;
  `solve` takes one. In R3,
  `Newton` holds its factorisation buffer in `state.fac` (`init(m.linsolve, prob, x, p)`), aliases
  `state.J` to `state.fac.factors` when `!(m.globalization isa DogLeg)` and
  `factor_eltype(m.linsolve, eltype(x)) === eltype(x)`, and allocates a separate `n × n` Jacobian
  buffer `state.J` when `m.globalization isa DogLeg` or the factor and working element types differ.
  `solve!(x, solver, p = nothing)` resets per-solve scalar state (`age = Int32(0)` and
  `stalled = false` for `Newton` with `reuse`, and `radius = radius₀` for `Newton` with `DogLeg`),
  mutates `x` in place and returns `SolverStatus{R}` with zero steady-state heap allocations on
  `Array` once `solver` is built. An `x` whose element type or length differs from the template
  given to `init` raises `ArgumentError` before any evaluation. The solver does not keep the `p`
  of `init`, so `solve!(x, solver)` solves with `p = nothing`.
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
  `init(m::SVDFactorization, prob, x, p)` allocates `fac.work = similar(x, Int32, 1)` (the
  per-sweep rotation-flag buffer) and `fac.rwork = similar(x, real(F), n)` (column norms squared
  during the sweep, and the sorted scaled singular values `σ̂` of `J / 2^e` after it).
  `factorize!!` prescales `J` unconditionally by `2^-e` (`e = matrix_exponent(J)`) into `fac.U`
  through `ldexp2` before initializing `fac.factors` (`V`) to the identity so that
  `J === fac.factors` is safe, and runs up to `JACOBI_MAX_SWEEPS` (`30`) sweeps of the round-robin
  schedule `jacobi_pair` with `m ÷ 2` disjoint parallel column pairs per round (`m = n + isodd(n)`),
  skipping negligible columns (`α <= maximum(fac.rwork) * jacobi_tolerance(n, R)^2` or
  `β <= maximum(fac.rwork) * jacobi_tolerance(n, R)^2`) and nearly orthogonal pairs
  (`abs(γ) <= jacobi_tolerance(n, R) * √(α * β)`) and reading `Array(fac.work)[1]` once per sweep.
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
