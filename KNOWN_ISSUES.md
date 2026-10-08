# Known issues

### K1 · The JET quality file cannot fail.

- **location:** `test/quality/jet.jl:9`
- **evidence:** lines 9–23 wrap `JET.report_package` in a `try` whose every path gives `true`,
  then run `@test ok`, so a defect that JET reports does not fail the suite. A mutant that adds a
  call of an undefined function to `rnorm` (`src/base/reductions.jl`) or to `isconverged`
  (`src/base/status.jl`) survives `test/quality/jet.jl`, although JET reports the call.
- **kind:** missing test
- **found:** 2026-09-30

### K2 · A sandboxed run with no matching Metal cache fails at the Metal precompile instead of skipping

- **location:** `test/devices/metal.jl:13` (`using Metal, Test`)
- **evidence:** Metal.jl 1.11.1's precompile workload (`src/precompile.jl:18`) calls
  `mtlfunction(identity, Tuple{Nothing})`, which calls `device()`. In the Bash sandbox
  `Metal.devices()` is empty, so Metal cannot precompile there, and `test/devices/metal.jl` never
  reaches its device skip. A default run on Apple silicon includes the `metal` group, so a
  sandboxed `Pkg.test()` is red whenever no Metal cache for its flags exists, as after a Metal
  update. A cache that a process outside the sandbox builds for the same flags loads in the
  sandbox, and the file then records its skip. Measured in GeometricOptimizers (its K15) with
  Julia 1.13.1: the first sandboxed `run-tests.jl <repository> metal` gave `Error 1`, with
  `Failed to precompile Metal` and `BoundsError: attempt to access 0-element
  Vector{Metal.MTL.MTLDevice} at index [1]` from `device()`; after one run of the same command
  outside the sandbox it gave `Broken 1`. Here, with a Metal cache present, the sandboxed
  `run-tests.jl <repository> devices/metal.jl` gives `Broken 1`. The fix is upstream: a workload
  that skips the kernel compilation where no device exists, asked for in JuliaGPU/Metal.jl#996.
  The workaround is the `precompile` local preference of Metal 1.11.1 (Metal.jl PR #915,
  `@load_preference("precompile", true)` at `src/Metal.jl:110`): set to `false`, it turns the
  precompile workload off.
- **kind:** upstream
- **found:** 2026-10-02

### K3 · A prepared DI `jacobian!` allocates 48 bytes per call when the chunk size is below `n`

- **location:** `src/ad/di.jl` (the `DI.jacobian!` call of `jacobian!!`)
- **evidence:** measured in a cold Julia 1.13.1 process with DifferentiationInterface 0.7.21 and
  ForwardDiff 1.4.6, on `F!(r, x, p) = (r .= p .* x)` with a `DI.Constant(p)` context and a
  preparation made once: `@allocated` of the prepared `jacobian!` is exactly `0` where the chunk
  size equals `n` (one vector-mode pass) and exactly `48` where it is below `n`, independent of
  the number of chunks (measured at `n = 16` with chunk sizes 16, 8 and 4, and `48` at
  `n = 7, chunk = 3`). `ForwardDiff.pickchunksize(n)`
  is `n` for `n ≤ 12`, so the default back end allocates nothing up to `n = 12` and 48 bytes per
  call above it. The 48 bytes do not grow with `n` or with the number of chunks, and are the same
  in `Float32` and `Float64`, which is what `test/ad/jacobian.jl` asserts beside the exact zero of
  the vector-mode case: §13.S of the design requires the exact zero where the chunk covers the
  iterate and at most 48 bytes where it does not. The chunked device path is unaffected: it never
  goes through DI. The fix is upstream, in DifferentiationInterface's chunked loop.
- **kind:** upstream
- **found:** 2026-10-08

### K4 · A plain `AutoEnzyme()` cannot differentiate a residual whose parameters are a `Constant`

- **location:** `test/enzyme/runtests.jl` (`BACKEND`)
- **evidence:** `prepare_ad(AutoEnzyme(), prob, r, x, p)` followed by `jacobian!!` raises
  `EnzymeRuntimeActivityError: Detected potential need for runtime activity. Constant memory is
  stored (or returned) to a differentiable variable`, pointing at the broadcast
  `r .= x .* x .+ p .* x .+ 2 .* x[perm]` of `test/helpers/adproblems.jl:30`, with Enzyme 0.13 and
  Julia 1.13.1. Enzyme's static activity analysis cannot prove that the constant parameter array
  broadcast into an active result is non-differentiable. The documented remedy is runtime
  activity, so `test/enzyme/runtests.jl` uses
  `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))`, with which every check
  passes. The same residuals differentiate through ForwardDiff with no such setting, so this is a
  property of Enzyme's analysis and not of this package. A caller who passes `AutoEnzyme()` to the
  R3 solver with parameters will meet the same error and the same remedy; nothing in the package
  sets the mode for the caller, because the mode is the caller's choice (§3.2 of the design).
- **kind:** upstream
- **found:** 2026-10-08
