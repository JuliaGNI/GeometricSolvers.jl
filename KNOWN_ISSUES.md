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
  same, measured with `@allocated` at testset scope in a cold process on Julia 1.13.2. `buffer(v)`
  is `nothing` and allocates nothing. The result of an arithmetic operation on a mutable static
  array is a new mutable array, which the heap holds, so no method of the seam can avoid this while
  the iterate keeps its type. An `MVector` is a value iterate for the call form and for the value
  methods, and the zero-allocated-bytes target of the R1 regime names `SVector` and `Number`, so
  the target does not cover this case.
- **kind:** found late
- **found:** 2026-10-16

### K13 · `Newton` with `Static()` returns `SUCCESS` at a non-root when the Jacobian is infinite.

- **location:** `src/nonlinear/newton.jl:209` (`step!!` of `Newton` with `reuse = nothing`)
- **evidence:** `solve(NonlinearProblem((x, p) -> x * x - 4, (x, p) -> R(Inf)), R(3), Newton())`
  gives `SUCCESS` with `x == 3` after 1 iteration, in `Float32` and `Float64` (Julia 1.13.2). The
  factor of `Inf` is not singular, the step `-F / Inf` is `-0.0`, and `converged` in
  `src/base/options.jl` reads the zero `stepnorm` as `stepnorm ≤ x_reltol · ‖x‖`, although
  `‖F(3)‖ = 5`. `DogLeg` does not take this path: it returns `LINESEARCH_FAILED` there.
- **kind:** defect
- **found:** 2026-10-26

### K14 · `DogLeg` stalls on the Rosenbrock system in `Float32` scaled by 1e18 to 3e18.

- **location:** `src/globalization/dogleg.jl` (`dogleg_step`, the Cauchy direction `g̃`)
- **evidence:** `solve` with `Newton(; globalization = DogLeg())` on the Rosenbrock system
  `s · (10 (x₂ − x₁²), 1 − x₁)` with its analytic `J`, from `(−1.2, 1)` in `Float32`, gives
  `SUCCESS` after 9 iterations at `s = 1e16`, `1e17` and `3e17`, and `STALLED` after 2 at
  `s = 1e18`, `2e18` and `3e18`, where `‖F(x₀)‖` is 4.9e18 to 1.5e19, below the overflow of the
  norm of `F` near 1.8e19 (Julia 1.13.3). At `s = 1e18` the entries of `g̃ = J' (F / ‖F‖)` at
  `x₀` are finite, `(−2.19e19, −8.94e18)`, but `rnorm(g̃)` is `Inf`, because their squares exceed
  `floatmax(Float32)`. The first trial, the whole Newton step, is rejected, so the next trial needs
  the Cauchy point, which this norm leaves undefined: the step returns `STALLED` with `x₀` kept. At
  the same scales `Newton()` with `Static()` gives `NONFINITE` after 1 iteration, and with
  `Backtracking()` `MAXITERS` or `NONFINITE`. The Dennis–Schnabel system from `(2, 0.5)` gives
  `SUCCESS` after 4 iterations at every one of these scales. The tested scales are `10^k`,
  `k ∈ {−16, −12, −8, 8, 12, 16}`.
- **kind:** found late
- **found:** 2026-10-27

### K16 · `DogLeg` stalls where the norm of `J ĝ` overflows in the Cauchy length, although a descent step exists.

- **location:** `src/globalization/dogleg.jl` (`dogleg_step`, `jnorm = rnorm(J * ĝ)`)
- **evidence:** in `Float32`, `F(x) = s · (2x₁ − 1, −1)` with `J = s · diag(2, 0)` from `(0, 0)`
  moves to the minimum of `‖F‖` at `x₁ = 0.5` at `s = 1e17` and `1e18`. At `s = 1e19`,
  `J ĝ = (−2e19, 0)` and `rnorm(J ĝ)` is `Inf`, because the square of its entry exceeds
  `floatmax(Float32)`. The Cauchy point is then undefined, and the step returns `STALLED` with the
  iterate kept (Julia 1.13.3). `rnorm` squares its entries with no scaling, as in K14. A scaled
  `rnorm` closes both: issue #29 proposes one.
- **kind:** defect
- **found:** #28; one fix with K14

### K17 · A `StaticSVD`'s reported `σ` holds `Inf` where `σ₁` exceeds `floatmax(real(TF))`.

- **location:** `src/linear/static/svd.jl:218`
- **evidence:** `test/linear/static/svd.jl`, the testset *a singular value above `floatmax(R)`
  leaves the solve finite*: at a full `floatmax(T)/2` matrix in `Float32` and `Float64` it asserts
  `@test !isfinite(fac.σ[1])` beside `all(isfinite, x)` and the `4n · eps(TF) · κ∞` solve bound.
  `σ` is the `SVector{N, R}` of the singular values of the matrix as it was passed, in
  `R = real(TF)`, and `factorize!!` unscales them by the power of two it divided the matrix by;
  that product overflows `R` although every entry of the matrix is finite — the `σ₁` there is
  `≈ 2.125 · floatmax`. The solve is not affected: `ldiv!!` divides by `σ̂`, the singular values
  the sweep computed. Reporting a `σ₁` above `floatmax(R)` needs a field wider than the factor
  type.
- **kind:** found late
- **found:** 2026-10-16

### K18 · `_adapt_order(backend, order::AbstractRange{<:Integer})` is redundant with the `isbitstype` branch of `_adapt_order(backend, order::AbstractVector{<:Integer})`.

- **location:** `src/nonlinear/batched.jl:30`
- **evidence:** `_adapt_order(backend, order::AbstractRange{<:Integer}) = return order` at lines
  30–32 is redundant with the `isbitstype(typeof(order)) && return order` guard at line 35 of
  `_adapt_order(backend, order::AbstractVector{<:Integer})`, because standard integer ranges
  (`UnitRange{Int}`, `StepRange{Int,Int}`, `OneTo{Int}`) are `isbitstype`
  `AbstractVector{<:Integer}`. Mutating `isbitstype(typeof(order)) && return order` to
  `false && return order` in `src/nonlinear/batched.jl:35` survives `nonlinear/batched.jl` because
  `AbstractRange{<:Integer}` intercepts range inputs at line 30.
- **kind:** dead code
- **found:** 2026-10-28
