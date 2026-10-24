# Metal 4x4 Float32 Static QR Miscompilation Spike

## Cause

Upstream: yes

Under Julia 1.13.2 (LLVM 20 opaque pointers) with `Metal.jl @ v1.11.0` targeting `air64_v29-apple-macosx27.0.1`, Apple's Metal shader compiler miscompiles `factorize!!(::Nothing, ::QRFactorization, ::SMatrix{4, 4, Float32, 16})` in `src/linear/static/qr.jl:24`.

In `src/linear/static/qr.jl:21-30`, the outer Householder loop `for k in 1:N` runs two consecutive variable-lower-bound inner loops `for i in k:N` over column `k` of the stack-allocated mutable matrix `A = MMatrix(Â)` (`alloca [8 x i64], align 16`), separated by `x = zero(MVector{N, F})` (`llvm.memset.p0.i64`) and the guard `if !iszero(amax)` (`src/linear/static/qr.jl:25`), before reading `x₁ = A[k, k]` (`src/linear/static/qr.jl:32`) and updating the trailing submatrix over `for i in (k + 1):N` (`src/linear/static/qr.jl:37-51`).

On Julia 1.13.2, LLVM's loop-peeling pass (`!loop_constrainer.loop.clone`) splits `for k in 1:4` on the `k < 4` bound of `for i in (k + 1):4` into a main loop `L3.i` (`k = 1, 2, 3`) and a peeled final iteration `L3.postloop.i` (`k = 4`), and hoists a negative-offset base pointer `%invariant.gep56.i = getelementptr i8, ptr %"new::MArray.i", i64 -4` for the `for i in k:4` column reads while addressing the `for i in (k + 1):4` trailing-submatrix stores through a three-step `getelementptr` chain on `%"new::MArray.i"`. When the `if !iszero(amax)` branch (`src/linear/static/qr.jl:25`) and the second `for i in k:N` loop (`src/linear/static/qr.jl:26-28`) sit between the `amax` reduction loop and the reflector construction `if !iszero(σ)` (`src/linear/static/qr.jl:31`), the Metal AIR-to-AGX backend loses the live stack-slot values of column 3 (`A[3, 3]` at byte offset 40 and `A[4, 3]` at byte offset 44 of `%"new::MArray.i"`) after computing `amax` and `σ` on iteration `k = 3`.

As a result, at `k = 3`, `amax` and `σ = amax * rnorm(SVector(x))` read the true entries `A[3, 3]` and `A[4, 3]` produced by iteration `k = 2`, whereas the immediately following loads `x₁ = A[k, k]` (`src/linear/static/qr.jl:32`) and `A[i, k] = A[i, k] / v₁` (`src/linear/static/qr.jl:38`) read `0.0f0`. That sets `phase = 1.0f0`, `v₁ = σ`, `τ[3] = 1.0f0`, and `A[4, 3] = 0.0f0`, and in the `j = 4` trailing update (`src/linear/static/qr.jl:41-51`) subtracts `A[3, 4]` from itself (`R[3, 4] = 0.0f0`) and leaves `A[4, 4]` unupdated before `k = 4`.

On Julia 1.11.9 (LLVM 16 typed pointers), `for k in 1:4` stays a single unpeeled loop with non-negative `getelementptr inbounds float, float*` indices and produces the exact reference solution on Metal.

## Evidence

In `scripts/spikes/metal_qr/probe.jl`, for the explicit `4x4` `Float32` matrix `A` (`κ∞(A) = 3.4602630074328187`, error bound `4n * eps(Float32) * κ∞(A) = 6.599927916398656e-6`), the unfixed kernel on Julia 1.13.2 produces:

- `CPU τ       = Float32[1.8677217, 1.9024092, 1.9607503, 2.0]`
- `Metal τ     = Float32[1.8677217, 1.9024092, 1.0, 2.0]`
- `CPU R[:,3]  = Float32[8.940697f-8, 0.32292452, -1.2075676, 0.14148395]`
- `Metal R[:,3]= Float32[8.940697f-8, 0.32292452, -1.2075676, 0.0]`
- `CPU x       = Float32[0.03887943, -1.8078897, 0.9948545, 0.09376802]` (`relerr = 2.89314385835767e-7`)
- `Metal x     = Float32[-0.11811417, -2.079124, -0.0, 0.40906927]` (`relerr = 0.5502846299810247`)

Across the five `n = 4, Float32` conditioned systems (`seed = 0x1055`, target `κ₂ ∈ {1, 10, 10², 10³, 10⁴}`), the unfixed kernel fails all five systems with relative errors `3.2090164093532274`, `0.42389200249621495`, `1.0`, `1.0`, and `0.9245924408027268` against bounds `4.544634866844348e-5`, `0.0002798842456502579`, `0.0029354551690624903`, `0.025852968831572578`, and `0.3046427164551889`, with `τ[3] = 1.0f0`, `factors[4, 3] = 0.0f0`, `factors[3, 4] = 0.0f0`, and `x[3] = -0.0f0` in every system.

## Index dependence

Depends on the system's index: no

Launching `qr_solve_kernel!` on the five `n = 4, Float32` conditioned systems in forward order (`1:5`), in reversed order (`5:-1:1`), and as five separate single-item kernel launches (`ndrange = 1`) produces the bitwise-identical wrong solution vector `x` for each system (`[0.7698731f0, 5.9730535f0, -0.0f0, 3.482837f0]`, `[-7.333426f0, -3.2040617f0, -0.0f0, -21.6029f0]`, `[701.1707f0, -2483.0505f0, -0.0f0, -1005.90466f0]`, `[-3038.109f0, 1473.077f0, -0.0f0, -1380.1969f0]`, and `[-693.78f0, -3429.9412f0, -0.0f0, -988.7534f0]`).

## Two runs and check bounds

Same wrong result in two runs in one process: yes
Survives --check-bounds=yes: no

Running `qr_solve_kernel!` twice in one process on the same inputs returns bitwise-identical wrong outputs (`x_h1 == x_h2` is `true`). Running `probe.jl` or `metal_rows.jl` under `julia +1.13 --startup-file=no --check-bounds=yes` eliminates all failures (`144/144` assertions pass, and `probe.jl` exits `0` with `relerr = 2.89314385835767e-7`) because `--check-bounds=yes` overrides `@inbounds` in `factorize!!` (`src/linear/static/qr.jl:16`) and inserts bounds-check branches that prevent LLVM from peeling `for k in 1:N` and emitting the negative-offset GEP pattern that Apple's Metal backend miscompiles.

## Fix

Workaround: Fill the scaled column buffer `x = MVector{N, F}(undef)` over the static range `for i in 1:N` with `x[i] = (i >= k && !iszero(amax)) ? A[i, k] / amax : zero(F)` in `src/linear/static/qr.jl:27-30`, which removes both the `llvm.memset` and the `if !iszero(amax)` conditional `for i in k:N` loop between the `amax` reduction and `x₁ = A[k, k]` so LLVM 20 and the Metal shader compiler keep the live column entries of `A` intact across all iterations `k = 1:N` while preserving `σ = amax * rnorm(SVector(x))` and zero host allocations.

## Commands

Run the standalone probe from the repository root inside a session with Metal device access:

```bash
julia +1.13 --startup-file=no --project=scripts/spikes/metal_qr -e 'using Pkg; Pkg.instantiate()'
julia +1.13 --startup-file=no --project=scripts/spikes/metal_qr scripts/spikes/metal_qr/probe.jl
```

Run the standalone probe with bounds checking enabled:

```bash
julia +1.13 --startup-file=no --check-bounds=yes --project=scripts/spikes/metal_qr scripts/spikes/metal_qr/probe.jl
```
