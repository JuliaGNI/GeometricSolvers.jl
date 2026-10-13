#!/usr/bin/env bash
# Part E measurements on the DGX Spark (GB10), backend `cuda`. Run it in a clone that has this
# branch checked out:
#
#     git fetch --prune origin
#     git switch results/spark
#     ./run-spark.sh
#
# It runs the `kernel_newton` spike of §10.E twice: the Float32 grid and the mixed
# Float64/Float32 rows on the card, and the same grid on the machine's host CPU. Each measurement
# checks its own output and stops the script when the check fails, so a failed run is never
# committed. The script commits the files on this branch and prints the push command; a results
# branch is never merged.
#
# The Metal rows and the M4 Max CPU baseline are not here: Metal runs in the maintainer's session
# through Kaimon (decision 6b). Where the spike's `verify.jl` exists, the script runs it on the
# machine, with the results directory and the card name as §10.E states.
#
# A cell that does not compile is a value of the grid, not a run failure, so the checks below
# read only whether the run started and finished.

set -euo pipefail

cd "$(dirname "$0")"

backend=cuda
machine=spark
vendor_env=test/gpu/$backend
spike=kernel_newton
spike_env=scripts/spikes/$spike
combined_env=../spike-$backend

die() {
    echo "run-$machine.sh: $*" >&2
    exit 1
}

# --- where the script runs -----------------------------------------------------------------

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "results/$machine" ] ||
    die "the branch is $branch; switch to results/$machine first"
git diff --quiet && git diff --cached --quiet ||
    die "the working tree is not clean; commit or stash its changes first"
command -v julia >/dev/null 2>&1 || die "julia is not on PATH"
[ -s $spike_env/run.jl ] ||
    die "$spike_env/run.jl is missing or empty; this branch must carry part E's spike"

mkdir -p results

# --- the environment record, before anything can fail --------------------------------------

{
    echo "part E, DGX Spark (GB10), $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "branch results/$machine at $(git rev-parse HEAD)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
    else
        echo "nvidia-smi: not on PATH"
    fi
    true
} > results/environment-$machine.txt

# --- the two environments, resolved and instantiated before every run -----------------------

julia --startup-file=no --project=$vendor_env -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --startup-file=no --project=$spike_env -e 'using Pkg; Pkg.instantiate()'

# --- the CUDA rows ---------------------------------------------------------------------------

# Step 5 of `test/gpu/README.md`: the two manifests must agree on every package they share.
manifest_versions() {
    julia --startup-file=no -e '
        using TOML
        versions(env) = Dict(name => get(only(entries), "version", "")
            for (name, entries) in TOML.parsefile(joinpath(env, "Manifest.toml"))["deps"])
        a, b = versions.(ARGS)
        println(join(sort!([name for name in intersect(keys(a), keys(b)) if a[name] != b[name]]),
                     "\n"))
    ' "$1" "$2"
}

differ=$(manifest_versions "$spike_env" "$vendor_env")
if [ -z "$differ" ]; then
    echo "the stacked manifests agree on every shared package"
    JULIA_LOAD_PATH="@:$PWD/$vendor_env:@stdlib" \
        julia --startup-file=no --project=$spike_env $spike_env/run.jl $backend \
        2>&1 | tee results/$spike-$machine-gpu.txt
else
    # The vendor package holds a shared package at an older version than the spike resolves
    # alone. Resolve the two together, in one environment outside the clone, and pin the vendor
    # package to the version the vendor environment resolved. The combined environment is
    # rebuilt from scratch on every run.
    echo "the stacked manifests differ on: $(echo "$differ" | paste -sd' ' -)"
    rm -rf "$combined_env"
    mkdir -p "$combined_env"
    cp $spike_env/Project.toml $combined_env/Project.toml
    vendor_version=$(julia --startup-file=no -e '
        using TOML
        deps = TOML.parsefile(joinpath(ARGS[1], "Manifest.toml"))["deps"]
        version = get(only(deps[ARGS[2]]), "version", "")
        isempty(version) && error("no version for $(ARGS[2]) in $(ARGS[1])/Manifest.toml")
        println(version)
    ' "$vendor_env" CUDA)
    echo "combined environment: $spike_env/Project.toml plus CUDA $vendor_version"
    julia --startup-file=no --project=$combined_env \
        -e "using Pkg; Pkg.add(name = \"CUDA\", version = \"$vendor_version\")"
    julia --startup-file=no --project=$combined_env $spike_env/run.jl $backend \
        2>&1 | tee results/$spike-$machine-gpu.txt
    {
        echo "combined environment: $spike_env/Project.toml plus CUDA $vendor_version"
        julia --startup-file=no -e '
            using TOML
            versions(env) = Dict(name => get(only(entries), "version", "")
                for (name, entries) in TOML.parsefile(joinpath(env, "Manifest.toml"))["deps"])
            a, b = versions.(ARGS)
            shared = sort!(collect(intersect(keys(a), keys(b))))
            differ = [name for name in shared if a[name] != b[name]]
            println("shared packages with test/gpu/cuda: ",
                    length(shared), ", differing: ",
                    isempty(differ) ? "none" : join(differ, ", "))
        ' "$combined_env" "$vendor_env"
    } >> results/environment-$machine.txt
fi

# --- the host CPU rows -----------------------------------------------------------------------

julia --startup-file=no --project=$spike_env $spike_env/run.jl cpu \
    2>&1 | tee results/$spike-$machine-cpu.txt

# --- the output checks -----------------------------------------------------------------------

check_run() {
    file=$1
    [ -s $file ] || die "$file is empty; the run produced nothing"
    grep -q '^Julia ' $file ||
        die "$file has no Julia line; the run did not start"
    grep -q '^Packages: ' $file ||
        die "$file has no Packages line; the run did not finish"
}

check_run results/$spike-$machine-gpu.txt
check_run results/$spike-$machine-cpu.txt

# --- the spike's own verification, where it exists -------------------------------------------

if [ -s $spike_env/verify.jl ]; then
    julia --startup-file=no --project=$spike_env $spike_env/verify.jl results $machine \
        2>&1 | tee results/$spike-$machine-verify.txt
    grep -q '^Julia ' results/$spike-$machine-verify.txt ||
        die "verify.jl printed no Julia line; read results/$spike-$machine-verify.txt"
else
    echo "$spike_env/verify.jl is empty; §10.E check 3 needs it before this run counts"
fi

# --- commit the files on the results branch --------------------------------------------------

files="results/$spike-$machine-gpu.txt results/$spike-$machine-cpu.txt results/environment-$machine.txt"
[ -s results/$spike-$machine-verify.txt ] && files="$files results/$spike-$machine-verify.txt"
git add -f $files
git commit -q -m "part E: kernel_newton on $machine"

cat <<EOF

results/$machine holds:
$(ls -1 results)

Push the branch, then tell the maintainer that it is there:

    git push -u origin results/$machine
EOF
