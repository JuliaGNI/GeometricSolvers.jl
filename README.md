# GeometricSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/GeometricSolvers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNI.github.io/GeometricSolvers.jl/dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGNI/GeometricSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/GeometricSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/GeometricSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/GeometricSolvers.jl)

Line searches, linear and nonlinear solvers for CPUs and GPUs, based on KernelAbstractions.

This package is at an early stage of development.

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **Unicode NFC** — **blocks** a staged file that is not NFC-normalised. Sources here are NFC, so a
  file that is not is a regression: a pattern typed in NFC silently matches nothing in a decomposed
  file, which defeats `grep`, an editor search and Documenter's doctest comparison alike. Fix with
  `julia --startup-file=no ~/Research/Environment/Harness/githooks/nfc.jl --apply <files>`. The invariant
  is `s == Unicode.normalize(s, :NFC)`, not the absence of combining marks — `q̇`, `v̄` and `f̄` have
  no precomposed codepoint and are two codepoints in NFC as well.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
