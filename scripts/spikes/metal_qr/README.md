# Metal static QR miscompilation spike

Two scripts, both run in this environment:

- `probe.jl` factorises and solves one explicit `4 × 4` `Float32` system with the static
  `QRFactorization` of `src/linear/static/qr.jl`, once in a KernelAbstractions kernel on Metal and
  once on the host, and compares both with a `BigFloat` reference. It exits 0 when the Metal
  solution meets the bound `4n · eps(Float32) · κ∞(A)`, 1 when it does not, and 3 when Metal is not
  functional.
- `sweep.jl` prints, for `QRFactorization` and `LUFactorization` in `Float32` and `ComplexF32`, the
  `n` in `1:12` at which the kernel of `test/helpers/staticsystems.jl` misses the bound of check 1,
  or returns `info ≠ 0`, on one of the systems `conditioned_systems(1000n + 1, T, Nothing, n)`.
  Those are five systems with `κ₂` geometric from 10 to `0.1 / (n · eps(Float32))`, or one system
  with `κ₂ = 1` at `n = 1`.

## The defect

Step `k` of `factorize!!` divides the subcolumn `A[k:N, k]` by its largest modulus `amax` into a
buffer `x`, zero above `k`, and takes `σ = amax * rnorm(x)`. On Metal, `QRFactorization` in
`Float32` returns wrong factors at some `n`. The set depends on how `x` is filled and on the Julia
version. `QRFactorization` in `ComplexF32` and `LUFactorization` pass at every `n` in every row.

| fill of `x` | Julia 1.12.7 | Julia 1.13.2 |
|:--|:--|:--|
| `x = zero(MVector{N, F})`, then `x[i] = A[i, k] / amax` for `i in k:N` under `if !iszero(amax)` | 7, 8 | 3–9 |
| `x = MVector{N, F}(undef)`, then `x[i] = (i >= k && !iszero(amax)) ? A[i, k] / amax : zero(F)` for `i in 1:N`; the form in `qr.jl` | 3–9 | none |
| `SVector(ntuple(i -> i >= k && !iszero(amax) ? A[i, k] / amax : zero(F), Val(N)))`, under `let A = A, k = k, amax = amax` | none | none |
| the same `ntuple` over `d = Tuple(A)`, with `d[i + (k - 1) * N]` for `A[i, k]` | 3–10 | 3–9 |
| an `@generated` `SVector{N, F}(e₁, …, e_N)`, inlined, with `eᵢ` the entry `i` of the second row | 3–10 | 3–9 |

Each cell is the set of `n` that the criterion of `sweep.jl` gives for `QRFactorization` in
`Float32`, with Metal.jl 1.11.1 on an `AGXG16CDevice`, `--check-bounds=auto`, one fresh process
per cell. Julia 1.12.7 is below the package's floor, Julia 1.13; its column shows that the result
depends on the Julia version. The third row is the only
one with no failure, but on the host it puts `A` on the heap, because the closure captures it:
`@allocated factorize!!(nothing, QRFactorization(), A)` is 80 bytes for a `4 × 4` `Float32` `A`,
where `test/linear/static/qr.jl` asserts 0.

With `--check-bounds=yes`, which restores the bounds checks that `@inbounds` removes, the first
row fails at no `n` on Julia 1.13.2, and `test/devices/metal_linear.jl` passes. `julia-runtest`
sets `--check-bounds=yes` by default.

The measurements do not separate the two compilers that lower the kernel, Metal.jl's translation
of the LLVM IR to AIR and Apple's Metal compiler, and no upstream report exists.

## The `4 × 4` case

With the first row on Julia 1.13.2, `probe.jl` gives, with `κ∞(A) = 3.4602630074328187` and the
bound `6.599927916398656e-6`:

```
CPU τ       = Float32[1.8677217, 1.9024092, 1.9607503, 2.0]
Metal τ     = Float32[1.8677217, 1.9024092, 1.0, 2.0]
CPU R[:,3]  = Float32[8.940697f-8, 0.32292452, -1.2075676, 0.14148395]
Metal R[:,3]= Float32[8.940697f-8, 0.32292452, -1.2075676, 0.0]
CPU x       = Float32[0.03887943, -1.8078897, 0.9948545, 0.09376802] (relerr = 2.89314385835767e-7)
Metal x     = Float32[-0.11811417, -2.079124, -0.0, 0.40906927] (relerr = 0.5502846299810247)
```

`τ[3] = 1` and a zero in row 4 of column 3 are what step `k = 3` stores when it reads zeros in
`A[3, 3]` and `A[4, 3]`, which are not zero. With the form in `qr.jl`, the Metal values match the
host values on Julia 1.13.2.

## Commands

From the repository root, in a session with Metal access:

```bash
julia --startup-file=no --project=scripts/spikes/metal_qr -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=scripts/spikes/metal_qr scripts/spikes/metal_qr/probe.jl
julia --startup-file=no --project=scripts/spikes/metal_qr scripts/spikes/metal_qr/sweep.jl
```

Add `--check-bounds=yes` to run with bounds checks. A row of the table is measured with its form
in place of the fill of `x` in `factorize!!`. The environment, as the package, needs Julia 1.13.
