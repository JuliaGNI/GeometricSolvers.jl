# Metal Float32 reduction accuracy spike for `norm2` and `rdot`.
#
# Usage:
#   julia --startup-file=no --project=test/gpu/metal scripts/spikes/metal_reductions/run.jl [auto|metal|cpu]
#
# Measures `norm2(a)` and `rdot(a, b)` on `MtlVector{Float32}` (and on `Vector{Float32}` in CPU
# smoke mode) with random signs at `n = 10^6, 10^7, 10^8` against a host `Float64` reference
# `ref`, divided by `sum(|a_i b_i|)`, and compares against the pairwise tree-reduction bound
# `ceil(log2(n)) * eps(Float32)` (`20 * eps(Float32)` at `10^6`, `24 * eps(Float32)` at `10^7`,
# `27 * eps(Float32)` at `10^8`).

const MODE = isempty(ARGS) ? "auto" : lowercase(first(ARGS))
MODE in ("auto", "metal", "cpu") ||
    error("unknown mode $(repr(MODE)): use auto, metal or cpu")

delete!(ENV, "JULIA_LOAD_PATH")
empty!(LOAD_PATH)
append!(LOAD_PATH, ["@", "@v#.#", "@stdlib"])

if Base.find_package("GeometricSolvers") === nothing ||
   (MODE != "cpu" && Base.find_package("Metal") === nothing)
    repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    metal_env = joinpath(repo_root, "test", "gpu", "metal")
    @eval Main using Pkg
    if isfile(joinpath(metal_env, "Project.toml"))
        @eval Main Base.invokelatest(Pkg.activate, $metal_env; io = devnull)
        @eval Main Base.invokelatest(Pkg.instantiate; io = devnull)
    else
        @eval Main Base.invokelatest(Pkg.activate, $repo_root; io = devnull)
    end
    if Base.find_package("GeometricSolvers") === nothing && !(repo_root in LOAD_PATH)
        pushfirst!(LOAD_PATH, repo_root)
    end
end

using GeometricSolvers: norm2, rdot
using LinearAlgebra: dot
using Random: Xoshiro

const SEED = 0x5eed_f320
const LENGTHS = (10^6, 10^7, 10^8)

function detect_metal(mode::AbstractString)
    mode == "cpu" && return false
    return try
        @eval Main using Metal
        @eval Main Base.invokelatest(Metal.functional)
    catch
        false
    end
end

const METAL_FUNCTIONAL = detect_metal(MODE)

function measure_case(to_vec, n::Int, dist::Symbol)
    rng = Xoshiro(SEED + UInt64(n) + (dist === :pos ? UInt64(1) : UInt64(2)))
    if dist === :pos
        x_h = rand(rng, Float32, n)
        y_h = rand(rng, Float32, n)
    else
        x_h = randn(rng, Float32, n)
        y_h = randn(rng, Float32, n)
    end

    x_64 = Float64.(x_h)
    y_64 = Float64.(y_h)
    ref_norm2 = sum(abs2, x_64)
    ref_rdot = real(dot(x_64, y_64))
    sum_abs_norm2 = ref_norm2
    sum_abs_rdot = sum(abs.(x_64) .* abs.(y_64))

    x_d = to_vec(x_h)
    y_d = to_vec(y_h)
    val_norm2 = Float64(norm2(x_d))
    val_rdot = Float64(rdot(x_d, y_d))

    rel_norm2 = abs(val_norm2 - ref_norm2) / sum_abs_norm2
    rel_rdot = abs(val_rdot - ref_rdot) / sum_abs_rdot
    eps32 = Float64(eps(Float32))
    bound_ulp = ceil(Int, log2(n))
    bound = bound_ulp * eps32

    ok = (rel_norm2 <= bound) && (rel_rdot <= bound)
    n2_ulp = round(rel_norm2 / eps32; digits = 2)
    rd_ulp = round(rel_rdot / eps32; digits = 2)
    ns = lpad(n, 9)
    ds = rpad(dist, 4)
    n2_s = "$(round(rel_norm2; sigdigits = 4)) ($n2_ulp eps)"
    rd_s = "$(round(rel_rdot; sigdigits = 4)) ($rd_ulp eps)"
    tag = ok ? "PASS" : "FAIL"
    msg = "  n = $ns  dist = $ds  norm2 = $n2_s  rdot = $rd_s  bound = $bound_ulp eps"
    println("$msg [$tag]")
    return ok
end

function run_suite(mode::AbstractString)
    mode == "metal" && !METAL_FUNCTIONAL &&
        error("mode metal was requested, but Metal is not functional")
    all_ok = true
    if mode == "cpu" || !METAL_FUNCTIONAL
        println("CPU smoke check (Vector{Float32}, n in $LENGTHS):")
        for n in LENGTHS
            for dist in (:pos, :zero)
                all_ok &= measure_case(identity, n, dist)
            end
        end
    else
        println("Metal reduction accuracy (MtlVector{Float32}, n in $LENGTHS):")
        to_mtl = v -> Main.Metal.MtlArray(v)
        for n in LENGTHS
            for dist in (:pos, :zero)
                all_ok &= measure_case(to_mtl, n, dist)
            end
        end
    end
    println("VERDICT: ", all_ok ? "PASS" : "FAIL")
    return all_ok ? 0 : 1
end

exit(run_suite(MODE))
