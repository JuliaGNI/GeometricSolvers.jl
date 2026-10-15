# kernel_newton — R1 in a kernel

The R1 design of part E of the design, measured. An `SVector`
Newton iteration without a line search, a partially pivoted LU with `Int32` pivots, and the
analytic Jacobian, in two storage variants — a local `MMatrix` mutated in place, and
functional updates on an `SMatrix` — plus the `SimpleNonlinearSolve` baseline. A KA kernel runs
one system per thread (the R2 launcher of §1.1); the CPU rows are a threaded host loop over the
same R1 code.

## The test problem

`F(x) = A x + sin.(x)/10 − b`, with `A = 2I + R`, `R` symmetric with a zero diagonal and
off-diagonal row sums of at most one half. One `A` per `n`, one `x*` per system in `[−1, 1]^n` from
a fixed host seed, `x₀ = 0`, and `b` built on the host in `T`. Every `J = A + diag(cos x)/10` has
`κ∞ < 2`, asserted on each `A` and recorded in the row. Newton stops at `‖Δx‖∞ ≤ 4n·eps(T)·‖x‖∞`
or 20 iterations, and the answer bound is `‖x − x*‖∞ ≤ 16n·eps(T)·‖x*‖∞`, against `x*` and against
the CPU result.

## Run it

The procedure for a run on a machine, and for saving and pushing its output, is in
[`test/gpu/README.md`](../../../test/gpu/README.md). This environment has no vendor package, so a
GPU run stacks the vendor environment behind it. From the repository root:

```sh
julia --startup-file=no --project=scripts/spikes/kernel_newton -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --threads=auto --project=scripts/spikes/kernel_newton scripts/spikes/kernel_newton/run.jl cpu
JULIA_LOAD_PATH="@:$PWD/test/gpu/metal:@stdlib" \
    julia --startup-file=no --project=scripts/spikes/kernel_newton \
    scripts/spikes/kernel_newton/run.jl metal
```

`<backend>` is `cpu`, `metal`, `cuda` or `rocm`. The rows of a backend follow §10.E: Metal carries
the `Float32` grid only, because Metal has no `Float64`; CUDA and ROCm add the all-`Float64` rows
and, on CUDA only, the mixed `Float64` problem with a `Float32` Jacobian and LU.

The CPU baseline runs one Julia thread per core, `--threads=auto`, because the thread count moves
the band. `run.jl cpu` stops with an error in a single-threaded Julia, and `verify.jl` fails a CPU
row that records one thread.

`run.jl <backend> [variant] [tmode] [n]` restricts a run to the rows that match: `variant` is `M`,
`S` or `SNS`, `tmode` is `F32`, `F64` or `MIXED`. To split a grid over several runs, concatenate
their outputs into the one file that `verify.jl` reads; a row that appears twice is a failure.

**Metal on this machine.** Metal 1.11.1 rejects the GPUCompiler that the spike environment resolves
on its own. Where step 5's manifest check fails, resolve the two together in one environment
outside the clone and pin the shared packages to the versions `test/gpu/metal` holds —
GPUCompiler 2.10.0, GPUArrays 11.5.16, KernelAbstractions 0.9.45, LLVM 9.13.2.

## The output

One `ROW` line per backend, storage variant, `tmode` and `n`, with the seven columns of §10.E
check 3 and the evidence behind them: the allocator method and count, the answer error against
`x*` and against the CPU result, the bound, the iteration count, the final residual, `κ∞`, the warm
median of a launch, and — for each timing cell — all five raw values, the rate and the spread, so
`verify.jl` can recompute them. A cell that has no value carries one of the marks of §10.E:
`does not compile: <first error line>`, `wrong: <‖x−x*‖∞>`, `wrong: maxiters`,
`not run: compile cap`, `not run: time cap`, `not timed: device Float64`.

The time cap bounds the wall time of one throughput cell at 30 minutes. Before a cell runs, the
median of the cell before it projects the cell's time — a warm launch and five timed launches;
a cell whose projection passes the cap is not run, and nor is any larger cell. The `1e4` cell
always runs. `verify.jl` derives each `time cap` mark from the raw values again. A row without
`time_cap_s` comes from a run without the cap; `verify.jl` keeps the timing of such a cell and
notes it.

CPU rows retain five allocation samples; device rows retain the matching LLVM allocator call sites.
CUDA rows also record registers and local-memory bytes from the compiled KA kernel; local-memory
bytes are the spill proxy for comparing all-`Float64` and mixed storage. The answer column checks
the 64-system answer batch with the strictest batch bound; the residual is the batch maximum.

Compile time is measured in a fresh child process per row — the first launch with `synchronize` at
top level — less the warm median the parent measures, so no earlier compile can hide in the time.

## Verify it

```sh
julia --startup-file=no --project=scripts/spikes/kernel_newton scripts/spikes/kernel_newton/verify.jl <results dir> <card>
```

`<card>` is the machine slug of `test/gpu/README.md`; it names the two files the run wrote.
`verify.jl` re-derives every cell from the raw output: the answer bound, the allocator method and
count, the grid's completeness, the median and the spread from the five raw values, and the two
decision rules. It takes the §5.2 storage statement and the §1.1 band as two optional arguments and
compares them; without them it prints the derived values. A missing output file is an error that
names it.

`verify.jl --self-check` exercises overlapping timing ranges, row-order independence, and a
comparator mutant on an in-memory fixture: the conservative band is 4, and the mutant gives 2.
The CPU wins only when its slowest run is faster than the device's fastest run; overlapping
ranges are a tie and keep that n on the device. The band counts the own R1 variants, `M` and `S`:
the SNS baseline enters neither the CPU comparison nor the compile cap of the band, and `verify.jl`
notes the first n where the CPU loop beats it. The fixture holds such an SNS row, so the self-check
fails if the band counts SNS.

The test problem never pivots, so the grid does not exercise the pivoting of the two LU solves.
`lu_check.jl` does: it solves random systems that pivot with both variants and compares the
solutions with `\`.

```sh
julia --startup-file=no --project=scripts/spikes/kernel_newton scripts/spikes/kernel_newton/lu_check.jl
```

