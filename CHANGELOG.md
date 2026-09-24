# Release Notes

All notable changes to GeometricSolvers.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from an interface change.


## [Unreleased] — targeting 0.1.0

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
- `LUFactorization`, `QRFactorization` and `SVDFactorization` carry their factorisation precision
  as a type parameter: `LUFactorization()` for the working type, `LUFactorization(Float32)` for a
  `Float32` factorisation, and the markers `TF32()`, `FP16()`, `BF16()` for vendor tensor cores.
  No factorisation runs yet.
- Internal: the one reduction seam (`norm2`, `rnorm`, `rdot`, each returning `real(T)` on an
  `Array`, a GPU array and an `SVector` in a kernel), and the `ToReal{R}` adaptor, which converts
  every float of a method tree to `R` through `Adapt`.
- `Adapt` is the first runtime dependency. The Julia floor is 1.11, for
  `LAPACK.getrf!(A, ipiv)` with preallocated pivots.
- Device runs by hand, as there is no GPU runner: `test/gpu/cuda`, `test/gpu/rocm` and
  `test/gpu/metal` are one environment per vendor, and `test/gpu/runtests.jl <backend>` runs a
  KernelAbstractions kernel on the device in `Float32` and `Float16` (and `Float64` on CUDA and
  ROCm) against a host reference. `test/gpu/README.md` gives the procedure for a run on a machine
  and for pushing its output to a `results/<machine>` branch. The spike environments have no
  vendor package: a GPU run of `scripts/spikes/capabilities/run.jl` stacks `test/gpu/<backend>`
  behind the spike's environment through `JULIA_LOAD_PATH`.
