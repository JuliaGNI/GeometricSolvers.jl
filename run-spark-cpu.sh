#!/usr/bin/env bash
# Part E on the DGX Spark (GB10): the host CPU grid of `kernel_newton` again, with one Julia
# thread per core. The run of 2026-10-14 (commit b76b4cd) ran the host rows with one thread, so
# the CUDA size band of §10.E is open until these rows exist. Run it in a clone that has this
# branch checked out:
#
#     git fetch --prune origin
#     git switch results/spark-cpu
#     ./run-spark-cpu.sh
#
# This branch carries the spike with the thread rule, the time cap and the verify.jl handling of
# rows from a run without the cap, plus results/kernel_newton-spark-gpu.txt, a copy of the CUDA
# rows of b76b4cd. The script runs the CPU grid with --threads=auto, checks the output, runs
# verify.jl on the new host rows and the CUDA rows, commits the files on this branch and prints
# the push command. Each check stops the script when it fails, so a failed run is never
# committed. A results branch is never merged.

set -euo pipefail

cd "$(dirname "$0")"

machine=spark
spike=kernel_newton
spike_env=scripts/spikes/$spike
rows_expected=21

die() {
    echo "run-spark-cpu.sh: $*" >&2
    exit 1
}

# --- where the script runs -----------------------------------------------------------------

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = results/spark-cpu ] ||
    die "the branch is $branch; switch to results/spark-cpu first"
git diff --quiet && git diff --cached --quiet ||
    die "the working tree is not clean; commit or stash its changes first"
command -v julia >/dev/null 2>&1 || die "julia is not on PATH"
grep -q 'threads=auto' $spike_env/run.jl ||
    die "$spike_env/run.jl has no thread rule; this branch must carry the current spike"
[ -s results/$spike-$machine-gpu.txt ] ||
    die "results/$spike-$machine-gpu.txt is missing; this branch must carry the CUDA rows"

threads=$(julia --startup-file=no --threads=auto -e 'print(Threads.nthreads())')
[ "$threads" -gt 1 ] || die "julia --threads=auto gives $threads thread(s)"
echo "julia --threads=auto gives $threads threads"

# --- the environment record, before anything can fail --------------------------------------

{
    echo "part E, DGX Spark (GB10), host CPU rows, $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "branch $branch at $(git rev-parse HEAD)"
    echo "cores: $(nproc); julia threads: $threads"
    if command -v lscpu >/dev/null 2>&1; then
        echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | paste -sd',' -)"
    fi
    echo "cuda rows: results/$spike-$machine-gpu.txt, a copy of b76b4cd"
} > results/environment-$machine-cpu.txt

# --- the spike environment, resolved and instantiated ----------------------------------------

julia --startup-file=no --project=$spike_env -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# --- the host CPU grid, one thread per core --------------------------------------------------

julia --startup-file=no --threads=auto --project=$spike_env $spike_env/run.jl cpu \
    2>&1 | tee results/$spike-$machine-cpu.txt

# --- the output checks -----------------------------------------------------------------------

file=results/$spike-$machine-cpu.txt
[ -s $file ] || die "$file is empty; the run produced nothing"
grep -q '^Packages: ' $file || die "$file has no Packages line; the run did not start"
grep -q '^Card: ' $file || die "$file names no card; the run did not record the machine"
rows=$(grep -c '^ROW ' $file || true)
[ "$rows" -eq $rows_expected ] || die "$file holds $rows ROW lines, not $rows_expected"
with_threads=$(grep '^ROW ' $file | grep -c " threads=\"$threads\"" || true)
[ "$with_threads" -eq $rows_expected ] ||
    die "only $with_threads of $rows ROW lines record threads=\"$threads\""

# --- the spike's own verification ------------------------------------------------------------

julia --startup-file=no --project=$spike_env $spike_env/verify.jl results $machine \
    2>&1 | tee results/$spike-$machine-verify.txt
grep -q 'failure(s)' results/$spike-$machine-verify.txt ||
    die "verify.jl printed no summary; read results/$spike-$machine-verify.txt"
grep -q ' 0 failure(s)' results/$spike-$machine-verify.txt ||
    die "verify.jl reports failures; read results/$spike-$machine-verify.txt"

# --- commit the files on the results branch --------------------------------------------------

git add -f results/$spike-$machine-cpu.txt results/$spike-$machine-verify.txt \
    results/environment-$machine-cpu.txt
git commit -q -m "part E: kernel_newton host rows on $machine with $threads threads"

cat <<EOF

$branch holds:
$(ls -1 results)

The band, from verify.jl:
$(grep 'R1 band' results/$spike-$machine-verify.txt || echo "(no band line)")

Push the branch, then tell the maintainer that it is there:

    git push origin $branch
EOF
