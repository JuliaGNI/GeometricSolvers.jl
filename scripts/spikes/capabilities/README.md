# Capability census

`run.jl` is a capability census: for each element type (F16, BF16, F32, F64, CF32, CF64) and each
R3 operation (GEMM, `lu!`, `qr!`, `svd!`, `A \ b`, `cholesky`, batched LU), does the backend array
support it? Plus two one-off checks: a KernelAbstractions kernel doing `BFloat16` arithmetic, and
`DifferentiationInterface`'s `jacobian!` with `AutoForwardDiff()` on the backend array and on a
`JLArray`.

## Run it

```sh
cd scripts/spikes/capabilities
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only
julia --startup-file=no --project=. run.jl cpu
```

`cpu` and `metal` both run this way from a terminal. A sandboxed shell sees no Metal device (the
`metal` cells then fail with a `BoundsError` on an empty device list), so from inside the sandbox
run `metal` in a Kaimon session on this directory:

```julia
include("run.jl")
main("metal")               # prints to stdout
# or, to capture the table as a string:
io = IOBuffer(); main(io, "metal"); String(take!(io))
```

`cuda` and `rocm` are accepted as backends but not run from this machine: `CUDA.jl` and
`AMDGPU.jl` are not dependencies of this environment (`AMDGPU.jl` does not support macOS, and
this machine has no CUDA device). A runner with that hardware adds the relevant package to
`Project.toml` and runs the same script unchanged.

There are no timings in this census; it checks correctness only.

## Output

Each cell is *pass*, *wrong* (with the relative-error and tolerance printed), *unsupported* (only
`Float64`/`ComplexF64` on `metal`, without a run — Metal has no `Float64`), *n/a* (batched LU, on
a backend that does not have it), or the first line of the thrown error. `lu!`, `cholesky` and
`A \ b` are skipped for `Float16` on `metal`: they crash the Julia process outright rather than
raising a catchable exception (confirmed manually), so the script does not attempt them.

The script only prints to stdout; it writes nothing to disk. The tables in this file's pull
request are the record of a run.
