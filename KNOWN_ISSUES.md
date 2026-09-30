# Known issues

### K1 · The JET quality file cannot fail.

- **location:** `test/quality/jet.jl:9`
- **evidence:** lines 9–23 wrap `JET.report_package` in a `try` whose every path gives `true`,
  then run `@test ok`, so a defect that JET reports does not fail the suite. A mutant that adds a
  call of an undefined function to `rnorm` (`src/base/reductions.jl`) or to `isconverged`
  (`src/base/status.jl`) survives `test/quality/jet.jl`, although JET reports the call.
- **kind:** missing test
- **found:** 2026-09-30
