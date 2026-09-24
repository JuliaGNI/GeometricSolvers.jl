# Device runs by hand

There is no GitHub runner with a GPU, and there is no device CI. The GPU machines are reachable
only inside the institute. A person runs the device tests, or a spike, on the machine, and pushes
the output to a results branch. A session then copies the output into the design record, and
deletes the branch. **A results branch is never merged.**

| machine | slug | backend |
|:--|:--|:--|
| DGX Spark | `spark` | `cuda` |
| RTX 4090 workstation | `rtx4090` | `cuda` |
| RX 7900 XTX workstation | `rx7900xtx` | `rocm` |
| Apple silicon Mac | `mac` | `metal`, through Kaimon (see below) |

## The environments

`test/gpu/<backend>/Project.toml` is one environment for each vendor: `cuda`, `rocm` and `metal`.
A machine instantiates only its own, so it never installs another vendor's stack. Each environment
has `GeometricSolvers` from this clone, through `[sources]`, which needs Julia 1.11 or later.

The spike environments in `scripts/spikes/<spike>/` are backend-neutral: they have no vendor
package. A spike run stacks the vendor environment behind the spike's own through
`JULIA_LOAD_PATH`. The spike finds its own dependencies first, and the vendor package in the
second environment. A package that both environments have, such as `KernelAbstractions` or
`GPUArrays`, loads from the spike's environment, for the vendor package too, so the two manifests
must agree on its version. No committed file changes for a run.

## One run on a machine

The commands are for a shell on the machine. Replace `<machine>` with the slug from the table and
`<backend>` with its backend.

1. Install Julia through `juliaup`, once:

   ```sh
   curl -fsSL https://install.julialang.org | sh
   ```

   The AMD machine also needs a ROCm release that supports `gfx1100`, and the user in the `render`
   and `video` groups.

2. Clone the repository, once. A push to a results branch needs push access:

   ```sh
   git clone git@github.com:JuliaGNI/GeometricSolvers.jl.git
   cd GeometricSolvers.jl
   ```

3. Get the code to run. The outputs go to a directory outside the clone, so that the switch to the
   results branch in step 6 cannot overwrite them:

   ```sh
   git fetch origin
   git switch --detach origin/main        # or the branch under test
   mkdir -p ../results
   ```

4. Instantiate the vendor environment, once for each change of its dependencies:

   ```sh
   julia --startup-file=no --project=test/gpu/<backend> -e 'using Pkg; Pkg.instantiate()'
   ```

5. Run the device tests, or a spike. Save the output as
   `../results/<spike>-<machine>-<cpu|gpu>.txt`.
   The device tests are the spike `devicetests`, on the `gpu`.

   The device tests:

   ```sh
   julia --startup-file=no --project=test/gpu/<backend> test/gpu/runtests.jl <backend> \
       2>&1 | tee ../results/devicetests-<machine>-gpu.txt
   ```

   A spike, here `capabilities` on the GPU:

   ```sh
   julia --startup-file=no --project=scripts/spikes/capabilities -e 'using Pkg; Pkg.instantiate()'
   JULIA_LOAD_PATH="@:$PWD/test/gpu/<backend>:@stdlib" \
       julia --startup-file=no --project=scripts/spikes/capabilities \
       scripts/spikes/capabilities/run.jl <backend> \
       2>&1 | tee ../results/capabilities-<machine>-gpu.txt
   ```

   The same spike on the CPU of the machine needs no vendor environment:

   ```sh
   julia --startup-file=no --project=scripts/spikes/capabilities \
       scripts/spikes/capabilities/run.jl cpu \
       2>&1 | tee ../results/capabilities-<machine>-cpu.txt
   ```

   Each spike's `README.md` gives its own arguments.

6. Push the output to the branch `results/<machine>`. The first command makes the branch, and
   the second switches to it where it exists on the remote already. `results/` and `*.txt` are in
   `.gitignore`, so add the file with `-f`:

   ```sh
   git switch -c results/<machine>        # or: git switch results/<machine>
   mkdir -p results
   cp ../results/<spike>-<machine>-<cpu|gpu>.txt results/
   git add -f results/<spike>-<machine>-<cpu|gpu>.txt
   git commit -m "<spike> on <machine>"
   git push -u origin results/<machine>
   ```

Then tell the session that the branch is there. The session copies the output into the design
record and deletes the branch.

## Metal, on the Mac

Metal runs in a Kaimon session on this Mac, with the active project `test/gpu/metal`. A Julia
process that the Claude Code sandbox starts sees no Metal device, so a plain shell is not enough.

```julia
using Pkg
Pkg.activate("test/gpu/metal"); Pkg.instantiate()
include("test/gpu/runtests.jl"); main("metal")
```

A spike needs the vendor environment behind its own, as `JULIA_LOAD_PATH` gives it on a machine.
In a session whose load path already has the global environment, put `test/gpu/metal` second:

```julia
Pkg.activate("scripts/spikes/capabilities"); Pkg.instantiate()
insert!(LOAD_PATH, 2, abspath("test/gpu/metal"))
include("scripts/spikes/capabilities/run.jl"); main("metal")
```

A Kaimon session can also run the shell commands of step 5 in a child process, through `run`. The
child inherits the session's access to the device. Remove `JULIA_LOAD_PATH` and `JULIA_PROJECT`
from the child's environment first, because the session sets both.
