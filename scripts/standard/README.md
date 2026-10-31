# The standard test set: scripts

| file | what it does |
|:--|:--|
| `update_reference.jl` | regenerates the reference table of `test/verification/standard.jl`, the `unstable` flags included; the table's only writer |
| `check_transcription.jl` | compares the 23 copied problems with NonlinearProblemLibrary.jl 0.1.8, in the environment of `Project.toml` |
| `diagnose.jl` | the diagnosis below: the perturbation families, and the reruns with other kernels |
| `generic_lapack.jl` | LAPACK's LU and QR replaced by pure Julia, for `diagnose.jl` and `trace.jl` |
| `trace.jl` | one cell traced with LAPACK and with pure Julia, line-search trials included |
| `ci_cells.txt` | the 42 cells that CI run 41196978054 reported as different, with their values on each runner |

Each script runs from the root of the repository, in the environment that its header names;
`generic_lapack.jl` runs only as an `include` of `diagnose.jl` and `trace.jl`.

## Why cells differ between runners

CI run 41196978054 compared a reference table written on an Apple M4 Max with the runners of
`CI.yml`. The Julia 1 and min jobs on `ubuntu-latest` and `windows-latest` differed in 34, 23, 23
and 23 cells, 42 in the union (`ci_cells.txt`); some differ in the code, some in the iteration
count by up to about 100. The macOS jobs agreed.

**The cause is round-off amplification, seeded by CPU-dependent OpenBLAS and LAPACK kernels.** A
change of one ulp in the start, in one residual or in one Newton direction changes the code or the
iteration count of the same cells on one machine. The kernels of the runners round differently
from the M4 Max's, and the solver amplifies that difference until a decision on round-off goes the
other way.

| | hypothesis | test | result | verdict |
|:--|:--|:--|:--|:--|
| H1 | the differing cells are sensitive to round-off | `diagnose.jl perturb`: each cell with `x₀` moved by 1 ulp (4 seeds), by 2 and 4 ulps (8 runs), the residual moved by 1 ulp (4 seeds) and the Newton direction moved by 1 ulp (4 seeds) | a code change or more than one iteration of difference in 60, 79, 76 and 69 of 1206 cells for the four families, and 92 for any family; of the 42 CI cells, 31, 37, 36, 38 and 42 | holds |
| H2 | BLAS threads | `diagnose.jl rerun blas1`, and `rerun` with the default 2 threads | both identical to the reference in all 1206 cells, `fnorm` included | ruled out |
| H3 | the OpenBLAS kernel or the LAPACK code | `diagnose.jl rerun generic` (pure-Julia LU and QR); `OPENBLAS_CORETYPE=THUNDERX2T99 … rerun` | pure Julia: 499 cells not identical, 27 flipped, 23 of them CI cells. ThunderX2 kernel: 58 not identical, 5 flipped, 4 of them CI cells. The `ARMV8` and `CORTEXA57` kernels give the reference in all 1206 cells (`VORTEX` loads `armv8`), and `NEOVERSEN2` and `NEOVERSEV1` stop with an illegal instruction on the M4 Max | holds: another kernel moves the cells that the perturbations move |
| H4 | a race or other nondeterminism | `diagnose.jl rerun` in new processes | identical to the reference in all 1206 cells | ruled out |
| H5 | fused multiply-add, SIMD reassociation or the system's libm | reading of the code | no residual and no file of `src/` uses `@fastmath`, `@simd`, `muladd` or `fma`; the residuals call `atan` and `exp`, which Julia implements itself, and `sqrt`, which is correctly rounded | ruled out |

**A traced flip.** `trace.jl 21 u0 "Newton Backtracking LU" Float32` gives `STALLED` after 6
iterations with LAPACK, and `LINESEARCH_FAILED` after 5 with the pure-Julia LU. The points of the
two runs agree to 1 ulp for 8 primal evaluations, differ by 14 ulps at the 9th, and by thousands
of ulps from the 10th. In the last line search both
reach the smallest step with `τ = 4 eps φ₀`: with LAPACK `φ(α) − φ₀` is −3 ulps of `φ₀` and the
step is accepted as `STALLED`; with pure Julia it is +27 ulps and the search ends at its floor as
`LINESEARCH_FAILED`. The code of this cell is a classification of round-off.

**What follows for the comparison.** A tolerance of a few ulps on the outcome cannot hold such a
cell, because one ulp of input changes its code or its count by up to about 100. So
`update_reference.jl` runs each cell 20 more times, as in H1 (16 for `Picard`, which has no Newton
direction), and flags it `unstable` where a perturbed run returns another code, an iteration count
that differs by more than 1, or a final iterate farther than `√eps(T) · max(1, ‖x‖)` from the
unperturbed one. `test/verification/standard.jl` checks an `unstable` cell only for its presence
and for a code other than `ERROR`, and every other cell for its code and for its iteration count
to within 1. `diagnose.jl perturb` reports 133 of 1206 cells marked, the 42 CI cells among them.
The Newton direction is perturbed through the `refinement` of `Newton` (`PerturbedDirection` in
`test/helpers/standard_cells.jl`), so no file of `src/` changes. A cell that a runner reports as
different and that the marker leaves is flagged `platform` by hand, with the observed values in
its comment.
