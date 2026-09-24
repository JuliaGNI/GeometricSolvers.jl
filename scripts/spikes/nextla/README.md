# NextLA.jl census

`run.jl` checks whether the KernelAbstractions kernels of
[NextLA.jl](https://github.com/NextLinearAlgebra/NextLA.jl), pinned to 0.2.3 in `Project.toml`,
give correct results on a backend. For each element type (F32, F64, CF32) it runs GEMM
(`GEMM_ADD!`), the four triangular solves (`LeftLowerTRSM!`, `LeftUpperTRSM!`, `RightLowerTRSM!`,
`RightUpperTRSM!`), the recursive `unified_rectrxm!` as a solve and as a multiply, the blocked QR
`geqrt!`, and `unmqr!`, which applies the `Q` of `geqrt!`.

NextLA 0.2.3 has no driver that chains its tile kernels (`tsqrt!`, `tsmqr!`, `ttqrt!`) into a full
QR, so the census has no row for them. Its LU is CPU code (`getrf2!` calls `BLAS.trsm!` and
`BLAS.gemm!`), so it has no row either.

## Run it

```sh
cd scripts/spikes/nextla
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only
julia --startup-file=no --project=. run.jl cpu
```

`cpu` and `metal` both run this way; `metal` needs an Apple GPU. From a REPL on this directory:

```julia
include("run.jl")
main("metal")               # prints to stdout
# or, to capture the table as a string:
io = IOBuffer(); main(io, "metal"); String(take!(io))
```

`cuda` is accepted as a backend, but this environment has no `CUDA.jl`. A CUDA run stacks the
vendor environment `test/gpu/cuda` behind this one through `JULIA_LOAD_PATH`, so no committed file
changes. The procedure for a run on a GPU machine, and for saving and pushing its output, is in
[`test/gpu/README.md`](../../../test/gpu/README.md). From the repository root:

```sh
julia --startup-file=no --project=scripts/spikes/nextla -e 'using Pkg; Pkg.instantiate()'
JULIA_LOAD_PATH="@:$PWD/test/gpu/cuda:@stdlib" \
    julia --startup-file=no --project=scripts/spikes/nextla \
    scripts/spikes/nextla/run.jl cuda
```

Without the vendor environment in the load path, every cell reports an `UndefVarError` for `CUDA`.

There are no timings in this census; it checks correctness only.

## Output

The output is a Markdown table. Each cell is *pass* or *wrong*, with the relative residual and the
tolerance printed; *unsupported* (only `Float64` on `metal`, without a run — Metal has no
`Float64`); or the first line of the thrown error, tagged with the step that threw it
(`to device`, `compute` or `read back`). An `unmqr!` cell tagged `[in geqrt!]` repeats the error of
the `geqrt!` run it needs as input.

The residual is a normwise backward error, computed in `ComplexF64` on the host against the
`T`-rounded input; the header comment of `run.jl` gives it per operation. The tolerance is
`max(100, n) * eps(real(T))`. The triangular matrices have a dominant diagonal, so that a wrong
result cannot hide behind a large condition number. No dimension is a multiple of the 32-wide GEMM
tile, and the recursive solve runs at n = 300, above the size of 256 where `unified_rectrxm!`
starts to recurse. The last line records the versions of the packages that decide each cell.

The script only prints to stdout; it writes nothing to disk. The tables in
[pull request #4](https://github.com/JuliaGNI/GeometricSolvers.jl/pull/4) are the record of the
`cpu` and `metal` runs.
