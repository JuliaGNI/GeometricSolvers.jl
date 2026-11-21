#!/usr/bin/env bash
# Part RO, the RX 7900 XTX run: the tests of GeometricSolversAMDGPUExt on the device, then the same
# tests once per named mutant N1, N2, N3, N4a, N4b and N5 (one patch in device-run/mutants/ each).
#
# Run it from the root of the clone, on the branch device-run/rocm-rx7900xtx:
#
#     bash device-run/run.sh ../results/rocmext-rx7900xtx
#
# It writes one log per run into the output directory, and summary.txt with the exit status of
# each run. The tests of the base exit 0; a mutant is caught when its run exits non-zero. The
# script restores the two patched files after each mutant and leaves the clone as it found it.
set -u

out=${1:?usage: bash device-run/run.sh <output directory>}
cd "$(git rev-parse --show-toplevel)"
files=(ext/GeometricSolversAMDGPUExt.jl test/gpu/rocm/runtests.jl)

if [ -n "$(git status --short -- "${files[@]}")" ]; then
    echo "stop: ${files[*]} have local changes; run git restore on them first" >&2
    exit 2
fi
mkdir -p "$out"
: > "$out/summary.txt"

{
    date -u
    git log -1 --format='%H %s'
    uname -srm
    uptime
    echo "device groups: $(id -nG | tr ' ' '\n' | grep -x -e video -e render | tr '\n' ' ')"
    rocminfo | grep -E 'Name: +gfx'
    julia --version
} > "$out/system.txt" 2>&1

julia --startup-file=no --project=test/gpu/rocm \
    -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.status()' > "$out/pkg-status.txt" 2>&1
echo "instantiate exit $?" | tee -a "$out/summary.txt"

run_tests() {
    julia --startup-file=no --project=test/gpu/rocm test/gpu/rocm/runtests.jl > "$out/$1.txt" 2>&1
    echo "$1 exit $?" | tee -a "$out/summary.txt"
}

run_tests base

for m in N1 N2 N3 N4a N4b N5; do
    if git apply "device-run/mutants/$m.patch"; then
        run_tests "mutant-$m"
    else
        echo "mutant-$m patch did not apply" | tee -a "$out/summary.txt"
    fi
    git checkout -- "${files[@]}"
done

echo "git status after the run:" >> "$out/summary.txt"
git status --short -- "${files[@]}" >> "$out/summary.txt"
echo "done; the logs are in $out"
