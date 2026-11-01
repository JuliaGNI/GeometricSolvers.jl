# Known issues

### K2 · Metal.jl 1.11's precompile workload fails where no Metal device exists, and the test environment works around it

- **location:** `test/Project.toml` (`[preferences.Metal]`)
- **evidence:** Metal.jl 1.11.1's precompile workload (`src/precompile.jl:18`) calls
  `mtlfunction(identity, Tuple{Nothing})`, which calls `device()`. In the Bash sandbox
  `Metal.devices()` is empty, so Metal cannot precompile its workload there. The issue is
  worked around: `test/Project.toml` sets Metal's `precompile` preference to `false` (Metal.jl
  PR #915, `@load_preference("precompile", true)` at `src/Metal.jl:110`), which turns the workload
  off, and `test/devices/metal.jl` asserts that the test process reads the preference as `false`.
  With the preference `true` and no Metal cache for it, the sandboxed
  `run-tests.jl <repository> devices/metal.jl` fails with `Failed to precompile Metal` and
  `BoundsError: attempt to access 0-element Vector{Metal.MTL.MTLDevice} at index [1]`; with
  `false` it passes, with the assertion passed and the device skip recorded (Julia 1.13.3). The
  fix is upstream: Metal.jl 1.12.0 runs its workload only where `!isempty(devices())`
  (JuliaGPU/Metal.jl#1000, which closed JuliaGPU/Metal.jl#996). Metal.jl 1.12.0 needs
  KernelAbstractions 0.10, while this package bounds KernelAbstractions at 0.9, so the test
  environment resolves Metal.jl 1.11.1, and the preference stays until the package moves to
  KernelAbstractions 0.10.
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

### K7 · A value iterate that is a mutable static array allocates on every update.

- **location:** `src/base/bang.jl` (`update_iterate!!`, `map!!`)
- **evidence:** `update_iterate!!(nothing, v, α, v)` with an `MVector{4, Float64}` allocates 48
  bytes, 32 in `Float32`, 48 in `ComplexF32` and 80 in `ComplexF64`, and `map!!(-, nothing, v)` the
  same, measured with `@allocated` at testset scope in a cold process on Julia 1.13.2. `buffer(v)`
  is `nothing` and allocates nothing. The result of an arithmetic operation on a mutable static
  array is a new mutable array, which the heap holds, so no method of the seam can avoid this while
  the iterate keeps its type. An `MVector` is a value iterate for the call form and for the value
  methods, and the zero-allocated-bytes target of the R1 regime names `SVector` and `Number`, so
  the target does not cover this case.
- **kind:** found late
- **found:** 2026-10-16

### K22 · The same-type SVD solve overflows for a finite right-hand side near `floatmax` where the factor's scale exponent exceeds the right-hand side's

- **location:** `src/linear/factors.jl` (`ldiv!!`, `rhs_exponent`), `src/linear/dense.jl` (`ldiv!!`)
- **evidence:** With `J = diag(2^20, 2^10, 2^20, 2^20)` and `b = (0, 0.75 floatmax(T), 0, 0)`,
  whose solution `(0, 0.75 floatmax(T) / 2^10, 0, 0)` is representable, the step of `ldiv!!` with
  `SVDFactorization()` is `[NaN, Inf, NaN, NaN]` on an `SMatrix` and an `Array`, in `Float32` and
  `Float64` (Julia 1.13.3, at eeaad5e); the `JLArray` route gives the same in the review of
  PR #48. `LUFactorization()` and `QRFactorization()` give the exact step. The SVD factor stores
  the factors of `J / 2^e`, here with `e = 20`, and `r` is divided by `2^k` with `k = 4` from
  `rhs_exponent`, so the solve forms the step times `2^(e - k)` before the multiplication by
  `2^(k - e)`, and that overflows. The review of PR #48 measured the same result at fe26534 on `main`. Dividing `r` by
  `2^max(k, e)` and multiplying the step by `2^(max(k, e) - e)` gives the exact step on this case,
  measured on the static route only in the review of PR #48; it trades the overflow for an
  underflow of `r / 2^e` where `J` is large and its solution is not small, and that trade is not
  decided.
- **kind:** defect
- **found:** PR #48
