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

### K4 · A plain `AutoEnzyme()` cannot differentiate a residual whose parameters are a `Constant`

- **location:** `test/enzyme/runtests.jl` (`BACKEND`)
- **evidence:** `prepare_ad(AutoEnzyme(), prob, r, x, p)` followed by `jacobian!!` raises
  `EnzymeRuntimeActivityError: Detected potential need for runtime activity. Constant memory is
  stored (or returned) to a differentiable variable`, pointing at the broadcast
  `r .= x .* x .+ p .* x .+ 2 .* x[perm]` of `Coupled` in `test/helpers/adproblems.jl`, with
  Enzyme 0.13 and Julia 1.13.1. Enzyme's static activity analysis cannot prove that the constant
  parameter array broadcast into an active result is non-differentiable. The documented remedy is
  runtime activity, so `test/enzyme/runtests.jl` uses
  `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))`, with which every check
  passes. The same residuals differentiate through ForwardDiff with no such setting, so this is a
  property of Enzyme's analysis and not of this package. A caller who passes `AutoEnzyme()` to the
  R3 solver with parameters will meet the same error and the same remedy; nothing in the package
  sets the mode for the caller, because the mode is the caller's choice.
- **kind:** upstream
- **found:** 2026-10-08

### K6 · A `BigFloat` static matrix has no R1 factorisation.

- **location:** `src/linear/static/lu.jl`, `src/linear/static/qr.jl` (`A = MMatrix(...)` in
  `factorize!!`)
- **evidence:** `factorize!!(nothing, LUFactorization(), SMatrix{2, 2, BigFloat}(2, 1, 1, 3))`,
  and the same with `QRFactorization()`, raise `setindex!() with non-isbitstype eltype is not
  supported by StaticArrays. Consider using SizedArray.` (Julia 1.13.1). Both kernels work on a
  local `MMatrix`, the R1 storage, and an `MMatrix` of a non-`isbits` element type has no
  `setindex!`. `BigFloat` is best effort on the CPU, and no test names it.
- **kind:** found late
- **found:** 2026-10-15

### K7 · A value iterate that is a mutable static array allocates on every update.

- **location:** `src/base/bang.jl` (`update_iterate!!`, `map!!`)
- **evidence:** `update_iterate!!(nothing, v, α, v)` with an `MVector{4, Float64}` allocates 48
  bytes, 32 in `Float32`, 48 in `ComplexF32` and 80 in `ComplexF64`, and `map!!(-, nothing, v)` the
  same, measured through the function barrier of `test/base/bang.jl` on Julia 1.13.2 in a cold
  process. `buffer(v)` is `nothing` and allocates nothing. The result of an arithmetic operation
  on a mutable static array is a new mutable array, which the heap holds, so no method of the seam
  can avoid this while the iterate keeps its type. An `MVector` is a value iterate for the call
  form and for the value methods, and the zero-allocated-bytes target of the R1 regime names
  `SVector` and `Number`, so the target does not cover this case.
- **kind:** found late
- **found:** 2026-10-16

### K9 · `QRFactorization` on Metal fails in `Float32` at `n = 4` on Julia 1.13.

- **location:** `test/devices/metal_linear.jl:28`
- **evidence:** `run-tests.jl <worktree> devices/metal_linear.jl` on Julia 1.13.2 gives
  `139 passed, 5 failed, 0 errored, 0 broken` in the one testset *Metal static LU and QR*, in the
  sub-row `GeometricSolvers.QRFactorization on Metal, Float32, n = 4`, where every failure is the
  same line. Two of them:
  `Expression: relerr(x[k], reference(A, b)) <= bound(A, T, Nothing)`,
  `Evaluated: 3.209016409353227442164204849116473172581928563889892965447639635540318498057716 <=
  2.604517060212300835169065004620252092783037480035186092144733663489957585888089e-5`, and
  `Evaluated: 0.4238920024962149501405469427484310203890761122195790054648217829001240852179092 <=
  0.0002651047319540524489503257446708858433467320775791476271978043854901525737284947`.
  Both the `metal` file and the static kernels it exercises are unchanged in this branch, and the
  same command on a tree of `origin/main` (`base-tree.jl` at `fea2905`) under 1.13.2 gives the
  same `139 passed, 5 failed` and the same two `Evaluated:` lines; on the `[compat]` floor,
  Julia 1.11.9, every testset of the full run passes.
- **kind:** found late
- **found:** 2026-10-16
