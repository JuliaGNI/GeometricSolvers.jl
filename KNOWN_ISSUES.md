# Known issues

## KI-1 · The JET quality file cannot fail

- **Kind:** missing test (pre-existing)
- **Where:** `test/quality/jet.jl:9-23`
- **Claim:** the file wraps `JET.report_package` in a `try` whose every path gives `true`, then
  runs `@test ok`. A defect that JET reports does not fail the suite.
- **Evidence:** a mutant that adds a call of an undefined function to `rnorm`
  (`src/base/reductions.jl`) or to `isconverged` (`src/base/status.jl`) survives
  `test/quality/jet.jl`, although JET reports the call. The file content is unchanged from
  `test/jet_tests.jl` on `main`; the test-suite migration renames it only.
- **Fix:** the JET part of the test-suite plan gives the file its checking form.
