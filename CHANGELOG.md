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
  and extracts L, U and p on the host. On a `ROCArray`, `F.L` scalar-indexed, so the cell
  reported the read-back and not the factorisation, in every element type that `lu!` supports
  there. On an `Array` the extracted L, U and p are `==` to those of the factorisation.

- `test/gpu/README.md` step 5 no longer says to instantiate both environments again when the
  spike's and the vendor's manifests differ: `Pkg.instantiate()` does not change a manifest. It
  says to run `Pkg.update()` in both, and, where the vendor package holds a shared package at an
  older version (AMDGPU.jl 2.8.0 holds GPUArrays at 11), to resolve the spike's packages and the
  vendor package together in one environment outside the clone.

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
  No factorisation runs yet.
- Internal: the one reduction seam (`norm2`, `rnorm`, `rdot`, each returning `real(T)` on an
  `Array`, a GPU array and an `SVector` in a kernel), and the `ToReal{R}` adaptor, which converts
  every float of a method tree to `R` through `Adapt`.
- `Adapt` is the first runtime dependency. The Julia floor is 1.11, for
  `LAPACK.getrf!(A, ipiv)` with preallocated pivots.
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
