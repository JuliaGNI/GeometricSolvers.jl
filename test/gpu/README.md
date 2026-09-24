# Device runs by hand

There is no GitHub runner with a GPU, and there is no device CI. The GPU machines are reachable
only inside the institute. A person runs the device tests, or a spike, on the machine, and pushes
the output to a results branch. The maintainer then records the output and deletes the branch.
**A results branch is never merged.**

| machine | slug | backend |
|:--|:--|:--|
| DGX Spark | `spark` | `cuda` |
| RTX 4090 workstation | `rtx4090` | `cuda` |
| RX 7900 XTX workstation | `rx7900xtx` | `rocm` |
| Apple silicon Mac | `mac` | `metal` |

## The environments

`test/gpu/<backend>/Project.toml` is one environment for each vendor: `cuda`, `rocm` and `metal`.
A machine instantiates only its own, so it never installs another vendor's stack. Each environment
has `GeometricSolvers` from this clone, through `[sources]`, which needs Julia 1.11 or later.

The spike environments in `scripts/spikes/<spike>/` are backend-neutral: they have no vendor
package. A spike run stacks the vendor environment behind the spike's own through
`JULIA_LOAD_PATH`. The spike finds its own dependencies first, and the vendor package in the
second environment. A package that both environments have, such as `KernelAbstractions` or
`GPUArrays`, loads from the spike's environment, for the vendor package too, so the two manifests
must agree on its version. Step 5 checks that before a spike run. No committed file changes for a
run.

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
   git fetch --prune origin
   git switch --detach origin/main        # or the branch under test
   mkdir -p ../results
   ```

4. Resolve and instantiate the vendor environment, before every run:

   ```sh
   julia --startup-file=no --project=test/gpu/<backend> -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
   ```

   `Pkg.resolve()` is needed because `GeometricSolvers` comes from the clone: when the checked-out
   code has a new dependency, `Pkg.instantiate()` alone keeps the old manifest, and the load fails
   with `package GeometricSolvers does not have … in its dependencies`.

5. Run the device tests, or a spike. Save the output as
   `../results/<spike>-<machine>-<cpu|gpu>.txt`.
   The device tests are the spike `devicetests`, on the `gpu`.

   The device tests:

   ```sh
   julia --startup-file=no --project=test/gpu/<backend> test/gpu/runtests.jl <backend> \
       2>&1 | tee ../results/devicetests-<machine>-gpu.txt
   ```

   A spike, here `capabilities` on the GPU. The second command stops with an error when the two
   manifests differ on the version of a package they share; instantiate both again in that case:

   ```sh
   julia --startup-file=no --project=scripts/spikes/capabilities -e 'using Pkg; Pkg.instantiate()'
   julia --startup-file=no -e '
       using TOML
       versions(env) = Dict(name => get(only(entries), "version", "")
           for (name, entries) in TOML.parsefile(joinpath(env, "Manifest.toml"))["deps"])
       a, b = versions.(ARGS)
       differ = sort!([name for name in intersect(keys(a), keys(b)) if a[name] != b[name]])
       isempty(differ) || error("the manifests differ on ", join(differ, ", "))
   ' scripts/spikes/capabilities test/gpu/<backend>
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

6. Push the output to the branch `results/<machine>`. `git switch -c` makes the branch. Where the
   branch exists on the remote already, use `git switch results/<machine>` in its place.
   `results/` and `*.txt` are in `.gitignore`, so add the file with `-f`:

   ```sh
   git switch -c results/<machine>        # or: git switch results/<machine>
   mkdir -p results
   cp ../results/<spike>-<machine>-<cpu|gpu>.txt results/
   git add -f results/<spike>-<machine>-<cpu|gpu>.txt
   git commit -m "<spike> on <machine>"
   git push -u origin results/<machine>
   git switch --detach
   git branch -d results/<machine>
   ```

   The last two commands delete the local branch, so that the next run finds only the remote one.

Then tell the maintainer that the branch is there.

## From a REPL

The same runs work from a Julia REPL started at the repository root. The device tests:

```julia
using Pkg
Pkg.activate("test/gpu/metal"); Pkg.resolve(); Pkg.instantiate()      # or cuda, rocm
include("test/gpu/runtests.jl"); main("metal")
```

A spike needs the vendor environment behind its own, as `JULIA_LOAD_PATH` gives it in step 5.
`insert!(LOAD_PATH, 2, …)` puts `test/gpu/<backend>` right after the active project, before the
global environment:

```julia
Pkg.activate("scripts/spikes/capabilities"); Pkg.instantiate()
insert!(LOAD_PATH, 2, abspath("test/gpu/metal"))
include("scripts/spikes/capabilities/run.jl"); main("metal")
```
