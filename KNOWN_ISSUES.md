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

### K21 · The same-type QR and SVD solves overflow for a finite right-hand side near `floatmax` whose solution is representable

- **location:** `src/linear/factors.jl` (`ldiv!!`, the path where `r` has the factor element type)
- **evidence:** With `J = I + H`, `H` the 4 × 4 Hilbert matrix, and
  `b = fill(T(0.75) * floatmax(T), 4)`, the step of `ldiv!!` with `SVDFactorization()` or
  `QRFactorization()` is not finite on an `SMatrix`, an `Array` and a `JLArray`, in `Float32` and
  `Float64`, although `big.(J) \ big.(b)` is below `floatmax(T)`; `LUFactorization()` gives a
  finite step there. `J = ones(T, 4, 4)` with
  the same `b` gives the same for the SVD, whose minimum-norm solution is `0.75 · floatmax / 4`
  in every entry. The same-type path does not scale `r`, and the QR and SVD solves form `Qᴴ r` or
  `Uᴴ r` before they divide, so a sum of entries near `floatmax` overflows; the mixed-type path
  divides `r` by `maximum(abs, r)` first. One `Float32` instance differs between the routes:
  `J = fill(floatmax(Float32) / 2, 4, 4)` with `b = fill(floatmax(Float32) / 2, 4)`, whose
  minimum-norm solution is `0.25` in every entry, gives a non-finite step on an `Array` and a
  finite one on an `SMatrix` and a `JLArray`. Measured at 677ea03 (Julia 1.13.3, M4 Max). The QR
  solve is not changed by PR #46, so the defect is probably older than PR #46; it was not measured
  on `main`. The planned fix scales `r` by an exact power of two inside the same-type `ldiv!!` and
  multiplies the step by its inverse with the `2^-e` of the factor, in a follow-up pull request.
- **kind:** defect
- **found:** PR #46
