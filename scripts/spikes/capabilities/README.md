# Capability census

`run.jl` is a capability census: for each element type (F16, BF16, F32, F64, CF32, CF64) and each
dense linear-algebra operation (GEMM, `lu!`, `qr!`, `svd!`, `A \ b`, `cholesky`, batched LU), does
the backend array support it? Plus two one-off checks: a KernelAbstractions kernel doing `BFloat16` arithmetic, and
`DifferentiationInterface`'s `jacobian!` with `AutoForwardDiff()` on the backend array and on a
`JLArray`.

## Run it

```sh
cd scripts/spikes/capabilities
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

`cuda` and `rocm` are accepted as backends, but `CUDA.jl` and `AMDGPU.jl` are not dependencies of
this environment; without them, every cell reports the load error. A runner with that hardware
adds the relevant package to `Project.toml` and runs the same script unchanged.

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
