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

- **Julia 1.12 is the floor** (`[compat] julia = "1.12"`, from 1.11). A breaking change under
  this package's pre-1.0 convention: a caller on Julia 1.11 can no longer install this version.
  The seven other environments of the repository (`test/gpu/cuda`, `test/gpu/metal`,
  `test/gpu/rocm`, `test/enzyme`, `scripts/spikes/nextla`, `scripts/spikes/capabilities` and
  `scripts/spikes/kernel_newton`) state the same floor. The repository records no reason for the
  raise. The allocation tests of `test/base/bang.jl`, `test/base/reductions.jl`,
  `test/ad/jacobian.jl`, `test/ad/forwarddiff_static.jl`, `test/linear/static/lu.jl` and
  `test/linear/static/qr.jl` no longer go through a function barrier and a warm-up call: on Julia
  1.12.7 a bare `@allocated` at testset scope reads the same figure as the barrier in every case
  measured (zero for each call asserted to allocate nothing, a positive figure for each `collect`
  control), and it does not count the compilation of a first call. The tolerances on the product
  of static matrices in `test/base/bang.jl` stay: on Julia 1.12.7 on aarch64 the seam kernel's
  product and the closure's still differ by up to 2 ulp in `ComplexF32`, and an x86_64 run, which
  CI includes, is not measured.

- Static `QRFactorization` (`factorize!!` in `src/linear/static/qr.jl`) fills the scaled subcolumn
  buffer `x` in one loop over `1:N`: `x[i]` is `A[i, k] / amax` for `i >= k` when `amax` is
  nonzero, and zero otherwise. The previous code zeros `x` and fills only `k:N`, under
  `if !iszero(amax)`. On the host, the factors, `τ` and `info` are bitwise identical to the
  previous fill on 64 random matrices per element type and size, and both versions allocate
  nothing.
  On Metal (Metal.jl 1.11.1) the fill repairs `Float32` QR on Julia 1.13.2 and breaks it on
  Julia 1.12.7, which is the package's floor (`[compat] julia = "1.12"`). In a sweep over
  `n = 1:12` of `Float32` static QR, a system counts as failed if it misses the bound of check 1
  or returns `info ≠ 0`. The previous fill fails at `n = 7, 8` on 1.12.7 and at `n = 3:9` on
  1.13.2; this fill fails at `n = 3:9` on 1.12.7 and at no `n` on 1.13.2. `ComplexF32` QR and
  `LUFactorization` in both element types pass at every `n` in all four cases. In
  `test/devices/metal_linear.jl`, the previous fill gives 139 passed and 5 failed on 1.13.2, and
  this fill gives 144 passed; on 1.12.7 the previous fill gives 144 passed, and this fill gives
  134 passed and 10 failed, all in `QRFactorization on Metal, Float32, n = 4`.
  The defect is open and recorded as `K9` in `KNOWN_ISSUES.md`. The fill removes the failure on
  1.13.2 in these measurements, but the cause is not located: no experiment separates Metal.jl's
  lowering from Apple's shader compiler, and no upstream report exists. `K10` and `K11` record why
  the Metal CI job does not see it: it runs with `--check-bounds=yes`, and its test sizes 1, 4,
  16 and 24 reach only `n = 4` of the affected sizes. `scripts/spikes/metal_qr/` holds the 4 × 4
  reproducer (`probe.jl`), the sweep (`sweep.jl`) and `README.md`, which compares five forms of
  the fill on both Julia versions; none passes on both and stays free of heap allocation on the
  host.
  On the host, on Julia 1.13.2 (two cold runs, load average 9 to 12), the median time per
  factorisation with this fill, relative to the previous one, is 1.81 to 1.86 for `Float64` at
  `N = 2`, 1.95 for `Float32` at `N = 2`, 1.19 to 1.22 for `Float32` at `N = 4`, and 0.99 to 1.05
  for every other pair of `T` in `Float64`, `Float32` and `ComplexF64` with `N` in 2, 4, 8 and 16.
  One cold run on Julia 1.12.7 showed no slowdown (ratios 0.70 to 1.01). Only these sizes were
  measured.
  For a caller: on Metal, `Float32` static QR on Julia 1.13.2 gets the repair. On Julia 1.12.7 it
  can miss the accuracy bound at `n = 3` to `9`, so a caller there should check the residual or
  use `LUFactorization`, which passed the sweep in both element types, until K9 closes.

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
  Where the element type of `J` differs from the factor type, `factorize!!` divides `J` by the power
  of two `2^e` that brings its largest `|re|` or `|im|` into `[1, 2)`, as the static factorisation
  does, and `ldiv!!` multiplies the step by `2^-e` in both paths. So a `Float64` or `ComplexF64` `J`
  at a scale where its entries overflow or become subnormal in `TF`, such as `1e39` or `1e-40` for
  `Float32`, solves to the factor precision's bound. Entries far below the largest of `J` can still
  underflow, and an `Inf` in `J` gives a step that is not specified. `LinearAlgebra` is now a
  direct dependency of the package.

