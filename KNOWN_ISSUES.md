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

### K5 · Static QR returns `SUCCESS` with a wrong finite step when a subcolumn lies below `√floatmin`.

- **location:** `src/linear/static/qr.jl` (`σ = rnorm(SVector(x))` in `factorize!!`)
- **evidence:** the column norm goes through `rnorm`, which squares the entries unscaled, so
  below about `√floatmin(TF)` it is inaccurate or 0, and a zero norm skips the reflector. With
  `A = floatmin(R) * [1 0; 1 1]` and `b = floatmin(R) * [1, 1]`, `factorize!!(nothing,
  QRFactorization(), A)` then `ldiv!!` gives `SUCCESS` and `x = [1, 1]`, not `[1, 0]`, in
  `Float32`, `Float64`, `ComplexF32` and `ComplexF64` (Julia 1.13.1). A critic's probe of part I
  scaled a matrix with `κ₂ = 10` by `1e-160` or `1e-170` in `Float64` and by `1e-20` or `1e-25`
  in `Float32`: the error is up to `1e13` times the check-1 bound `4n·eps·κ∞`, with `SUCCESS`,
  where LU solves the same matrices to the bound. The design decides only the overflow side ("an
  entry above `√floatmax(TF)` is not scaled here"), and the scales lie outside check 1's family
  `10^(−8:4:8)`. The fix is a max-abs scale of the subcolumn before the squares, as LAPACK's
  `nrm2`, or a decision of the edge for parts I and O.
- **kind:** found late
- **found:** 2026-10-15

### K6 · A `BigFloat` static matrix has no R1 factorisation.

- **location:** `src/linear/static/lu.jl`, `src/linear/static/qr.jl` (`A = MMatrix(...)` in
  `factorize!!`)
- **evidence:** `factorize!!(nothing, LUFactorization(), SMatrix{2, 2, BigFloat}(2, 1, 1, 3))`,
  and the same with `QRFactorization()`, raise `setindex!() with non-isbitstype eltype is not
  supported by StaticArrays. Consider using SizedArray.` (Julia 1.13.1). Both kernels work on a
  local `MMatrix`, the R1 storage of the design (§5.2), and an `MMatrix` of a non-`isbits` element
  type has no `setindex!`. Decision 12 makes `BigFloat` best effort on the CPU, and no check of
  part I names it.
- **kind:** found late
- **found:** 2026-10-15
