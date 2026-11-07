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

### K23 · The same-type LU and QR solves overflow in the back substitution for a finite right-hand side near `floatmax` whose solution is representable

- **location:** `src/linear/factors.jl` (`ldiv!!`), `src/linear/dense.jl` (`ldiv!!`)
- **evidence:** With `H = H₄ / 2`, `H₄` the 4 × 4 Hadamard matrix,
  `J = H diag(2^20, 2^14, 2^20, 2^20) H` and `b = 0.75 floatmax(T) · H[:, 2]`, whose solution
  `0.75 floatmax(T) / 2^14 · H[:, 2]` is representable (`7.78844f33` and `4.11459e303` in modulus),
  the step of `ldiv!!` with `LUFactorization()` and `QRFactorization()` is
  `[Inf, -Inf, Inf, -7.78843f33]` in `Float32` and `[Inf, -Inf, Inf, -4.11459e303]` in `Float64`,
  on an `SMatrix` and an `Array` (Julia 1.13.3); the `JLArray` route fails the same test. `r` is
  divided by `2^k` with `k = 2`, which keeps `Qᴴ r` finite; a product of an entry of `U` or `R`
  near `2^19` with an entry of the scaled step near `floatmax / 2^16` exceeds `floatmax`, which
  points at the back substitution, but the operation that overflows was not traced.
  `SVDFactorization()` solves the system. The test systems of
  `test/helpers/matrix.jl` leave this system out for LU and QR.
- **kind:** defect
- **found:** PR #48

### K26 · A device vector wrapped in a `ComponentArray` gets the host factor, and the host QR and SVD then fail

- **location:** `src/linear/factors.jl` (`init_linear`)
- **evidence:** With ComponentArrays 0.15.50 and JLArrays 0.3.3, `x = ComponentVector(JLArray(...))`
  gives `x isa AbstractGPUArray == false`, `copy(x)` stays a `ComponentVector`, and
  `similar(x, Float32, 4, 4)` is a `JLArray{Float32, 2}`. `init_linear` selects the device factor
  by the type of `x`, not by the matrix it allocates, so all three builders return the host
  factors `DenseLU`, `DenseQR` and `DenseSVD` around a device matrix, and `factorize!!` on
  `DenseQR` and `DenseSVD` fails with "Scalar indexing is disallowed". The same predicate,
  `x isa AbstractGPUArray`, is on `main`, in `src/linear/factors.jl` and in `prepare_ad` of
  `src/ad/di.jl`. ComponentArrays is neither a dependency nor a test dependency of the package, so
  a wrapped device vector is not a supported input. Support is its own pull request, which selects
  the factor from the allocated matrix instead of from `x`.
- **kind:** defect
- **found:** PR #55 (CodeRabbit, on #56)

### K30 · A solve whose residual is wider than the start iterate fails in the `SolverStatus` constructor

- **location:** `src/nonlinear/solver.jl` (`SolverStatus`)
- **evidence:** With `F(x, p) = x^2 - p` and `prob = NonlinearProblem(F)`,
  `solve(prob, 1.5, Newton(), 4.0)` and `solve(prob, 1.5f0, Newton(), 4.0f0)` return `SUCCESS`,
  while `solve(prob, 1.5f0, Newton(), 4.0)` raises `MethodError: no method matching
  SolverStatus{Float32}(::ReturnCode, ::Int32, ::Float64, ::Float32, ::Int32, ::Bool)` (merge of
  `main` into `task/z`, Julia 1.13.3). The status takes its real type from the start iterate, and
  the residual norm of a `Float64` parameter is a `Float64`. The implicit-differentiation path of
  `solve` inherits the same failure. Whether the solve promotes the iterate, converts the norm or
  refuses the mix with an `ArgumentError` is open.
- **kind:** defect
- **found:** PR #53

### K45 · On the default `Newton` path the Metal `Float32` LU copies the `n × n` Jacobian on every step

- **location:** `ext/GeometricSolversMetalExt.jl` (`factorize!!`)
- **evidence:** `factorize!!` takes `A = J === fac.factors ? copy(fac.factors) : J`, so that a
  fallback to `ka_factorize_lu!` after a non-singular MPS failure still has the matrix that `lu!`
  writes the factors over. `init_newton_J` returns `fac.factors` when the globalization is not
  `DogLeg` and the factor element type is the working one (`src/nonlinear/newton.jl`), and
  `regularize!!` returns `J` itself for `μ = 0`, the default (`src/base/bang.jl`), so
  `newton_factor` passes `fac.factors` as `J`, and every step allocates and fills one more
  `n × n` device matrix. `_implicit_tangent` passes `fac_slot.factors` the same way
  (`src/nonlinear/implicit.jl`). Metal.jl 1.11.1's `lu!` allocates an `n × n` transpose buffer on
  every call as well (`src/linalg.jl`). The copy is needed only for the fallback, and MPS status
  `-1` can reach it. Whether a non-singular MPS failure maps to `info = 1` (`SINGULAR`) with no
  copy, or keeps the fallback at the cost of the copy, is open.
- **kind:** defect
- **found:** PR #61

### K46 · The CUDA `backsolve!` method for `DeviceSVD` repeats the own-kernel method

- **location:** `ext/GeometricSolversCUDAExt.jl` (`backsolve!` for `DeviceSVD`)
- **evidence:** its body, `ka_svd_backsolve!(fac, d, false); return d`, is that of
  `backsolve!(fac::DeviceSVD{<:AbstractMatrix{T}}, d::AbstractVector{T})` in
  `src/linear/ka/svd.jl`, so the method runs the own kernel and calls no cuSOLVER or cuBLAS
  routine. The same-type `ldiv!!` reaches `backsolve_shifted!` and not this method. The CUDA
  census of `test/gpu/cuda/runtests.jl` asserts that `which(backsolve!, ...)` for `DeviceSVD`
  names the extension, so removing the method changes that assertion too.
- **kind:** defect
- **found:** PR #61

### K47 · The CUDA SVD writes `J` into `fac.factors`, which the own-kernel SVD does not, and no later step reads it

- **location:** `ext/GeometricSolversCUDAExt.jl` (`factorize!!` for `DeviceSVD`)
- **evidence:** the CUDA `factorize!!` copies `J` into `fac.factors` (converted to the factor
  type in the mixed case, unscaled) before it factorises `J / 2^e` in `fac.U`; the fallback
  receives `J` itself. `ka_factorize_svd!` (`src/linear/ka/svd.jl`) writes `fac.U` and leaves
  `fac.factors` alone. The docstring of `factorize!!` says that `J` is written into
  `fac.factors`, and `test/gpu/cuda/runtests.jl` asserts that `fac.factors` keeps `J`. The copy
  costs one `n × n` device copy per call where `J` is not the factor buffer. Which of the two
  backends sets the contract for `fac.factors` of a `DeviceSVD` is open.
- **kind:** defect
- **found:** PR #61

### K48 · The CUDA extension calls the deprecated bindings `CUDA.CUSOLVER` and `CUDA.CUBLAS`

- **location:** `ext/GeometricSolversCUDAExt.jl`
- **evidence:** CUDA 6.4.2 declares `Base.@deprecate_binding CUBLAS cuBLAS true` and
  `Base.@deprecate_binding CUSOLVER cuSOLVER true` (`src/CUDA.jl`), and `[compat]` admits CUDA 6
  (`CUDA = "5.5, 6"`). Under `--depwarn=yes` each use prints `Use of CUDA.CUSOLVER is deprecated,
  use cuSOLVER instead.` CUDA 5.x has no `cuSOLVER` binding, so the names that work on both
  majors differ, and the choice between them depends on the CUDA bound.
- **kind:** defect
- **found:** PR #61

### K49 · `test/gpu/README.md` does not describe the by-hand CUDA suite `test/gpu/cuda/runtests.jl`

- **location:** `test/gpu/README.md`
- **evidence:** a search for `cuda/runtests` in `test/gpu/README.md` and `test/gpu/runtests.jl`
  finds nothing. The README describes the `test/gpu/cuda/` environment and the run of
  `test/gpu/runtests.jl` with the backend `cuda`, and that runner includes only `ad.jl`, so no
  file names the vendor suite or says how to run it on a CUDA machine.
- **kind:** docs
- **found:** PR #61
