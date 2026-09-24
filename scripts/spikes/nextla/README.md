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

This environment has no vendor package. A GPU run stacks the vendor environment
`test/gpu/<backend>` behind it through `JULIA_LOAD_PATH`, so no committed file changes. The
procedure for a run on a GPU machine, and for saving and pushing its output, is in
[`test/gpu/README.md`](../../../test/gpu/README.md). From the repository root:

```sh
julia --startup-file=no --project=scripts/spikes/nextla -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=scripts/spikes/nextla scripts/spikes/nextla/run.jl cpu
JULIA_LOAD_PATH="@:$PWD/test/gpu/<backend>:@stdlib" \
    julia --startup-file=no --project=scripts/spikes/nextla \
    scripts/spikes/nextla/run.jl <backend>
```

`<backend>` is `metal` or `cuda`. Without the vendor environment in the load path, every cell of
that backend reports an `UndefVarError` for the vendor module. From a REPL started at the
repository root:

```julia
using Pkg; Pkg.activate("scripts/spikes/nextla"); Pkg.instantiate()
insert!(LOAD_PATH, 2, abspath("test/gpu/metal"))   # the vendor environment
include("scripts/spikes/nextla/run.jl")
main("metal")               # prints to stdout
# or, to capture the table as a string:
io = IOBuffer(); main(io, "metal"); String(take!(io))
```

## Probes

`probes/` holds three small scripts behind the findings of the census. Each prints its figures and
writes nothing to disk. They take no argument; the two Metal probes run with `test/gpu/metal`
stacked, as above.

| script | backend | checks |
|:--|:--|:--|
| `probes/trsm_repeat.jl` | `metal` | each base TRSM kernel 10 times at n = 40, 256, 1000 (F32), counting the runs above the tolerance |
| `probes/trsm_flags.jl` | `cpu` | `NextLA.trsm` with `transa` and `diag` other than `'N'`, against the solve those flags ask for |
| `probes/geqrt_shared.jl` | `metal` | `geqrt!` on `SharedStorage` arrays, 5 runs each for F32 and CF32: factorisation and orthogonality errors |

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
