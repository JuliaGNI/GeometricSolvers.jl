# Release Notes

All notable changes to GeometricSolvers.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from an interface change.


## [Unreleased] — targeting 0.1.0

### New Features

- The empty package skeleton: `Project.toml` with no runtime dependencies yet, the module and its
  docstring, a test harness with Aqua and a diagnostic JET check, and `test/backends.jl` — the
  list of array back ends later parts loop over (`Array`, `JLArray`, and the KernelAbstractions
  `CPU()` backend). `scripts/float16_dot.jl` measures the accuracy of five `Float16` dot-product
  accumulation policies against a `BigFloat` reference, ahead of the accumulation-policy decision
  in phase P0.
