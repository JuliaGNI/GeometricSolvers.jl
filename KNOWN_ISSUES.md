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
  that skips the kernel compilation where no device exists.
- **kind:** upstream
- **found:** 2026-10-02
