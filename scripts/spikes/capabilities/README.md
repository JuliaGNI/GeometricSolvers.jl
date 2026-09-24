# Capability census

`run.jl` is a capability census: for each element type (F16, BF16, F32, F64, CF32, CF64) and each
dense linear-algebra operation (GEMM, `lu!`, `qr!`, `svd!`, `A \ b`, `cholesky`, batched LU), does
the backend array support it? Plus two one-off checks: a KernelAbstractions kernel doing `BFloat16` arithmetic, and
`DifferentiationInterface`'s `jacobian!` with `AutoForwardDiff()` on the backend array and on a
`JLArray`.

## Run it

The procedure for a run on a GPU machine, and for saving and pushing its output, is in
[`test/gpu/README.md`](../../../test/gpu/README.md). This environment has no vendor package, so a
GPU run stacks the vendor environment `test/gpu/<backend>` behind it. The two manifests must agree
on every package they share; step 5 of that procedure checks it. From the repository root:

```sh
julia --startup-file=no --project=scripts/spikes/capabilities -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=scripts/spikes/capabilities scripts/spikes/capabilities/run.jl cpu
JULIA_LOAD_PATH="@:$PWD/test/gpu/<backend>:@stdlib" \
    julia --startup-file=no --project=scripts/spikes/capabilities \
    scripts/spikes/capabilities/run.jl <backend>
```

`<backend>` is `metal`, `cuda` or `rocm`. From a REPL started in this directory with
`julia --project=.`:

```julia
insert!(LOAD_PATH, 2, abspath("../../../test/gpu/metal"))   # the vendor environment
include("run.jl")
main("metal")               # prints to stdout
# or, to capture the table as a string:
io = IOBuffer(); main(io, "metal"); String(take!(io))
```

Without the vendor environment in the load path, every cell of a GPU backend reports the load
error, unless the global environment has the vendor package: then the package loads from there.

There are no timings in this census; it checks correctness only.

## Output

Each cell is *pass*, *wrong* (with the relative-error and tolerance printed), *unsupported* (only
`Float64`/`ComplexF64` on `metal`, without a run — Metal has no `Float64`), *n/a* (batched LU on
`cpu` and `metal`, which do not have it), *not measured* (batched LU on `cuda` and `rocm`, which
have no check yet), or the first line of the thrown error, tagged with the step that threw it
(`to device`, `compute` or `read back`). A `cholesky` cell whose factor element type differs from
the input's names the factor type. `lu!`, `cholesky` and `A \ b` are skipped for `Float16` on
`metal`: they crash the Julia process rather than raise a catchable exception, so the script does
not attempt them. The last line records the versions of the packages that decide each cell.

The script only prints to stdout; it writes nothing to disk. The tables in this file's pull
request are the record of a run.
