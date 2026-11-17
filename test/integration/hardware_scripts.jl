# The measurement scripts of `scripts/hardware/` and `scripts/r2_throughput.jl` run to completion
# at their smallest size: `ka_lu_large.jl` on `JLArray`, `sync_cost.jl` on the CPU and on
# `JLArray`, `r3_lu_blas.jl` with OpenBLAS, and `r2_throughput.jl` on the CPU, whose SoA
# right-hand sides must give the roots of the AoS ones. A device or a library that is not present
# gives one line and no rows. Scalar indexing of a device array is disallowed, so a script that
# reads a device array entry by entry fails here.

using GPUArraysCore: allowscalar
using LinearAlgebra: I, norm
using Random: Xoshiro
using Test

allowscalar(false)

module KALULarge
include(joinpath(@__DIR__, "..", "..", "scripts", "hardware", "ka_lu_large.jl"))
end

module SyncCost
include(joinpath(@__DIR__, "..", "..", "scripts", "hardware", "sync_cost.jl"))
end

module R2Throughput
include(joinpath(@__DIR__, "..", "..", "scripts", "r2_throughput.jl"))
end

module R3LUBlas
include(joinpath(@__DIR__, "..", "..", "scripts", "hardware", "r3_lu_blas.jl"))
end

module CusolverGetrf
include(joinpath(@__DIR__, "..", "..", "scripts", "hardware", "cusolver_getrf.jl"))
end

@testset "the median of the timed calls" begin
    @test KALULarge.sample_median([3.0, 1.0, 2.0]) == 2.0
    @test KALULarge.sample_median([4.0, 1.0, 3.0, 2.0]) == 2.5
    @test KALULarge.sample_median([7.0]) == 7.0
    calls = Ref(0)
    t = KALULarge.time_calls(() -> (calls[] += 1), () -> nothing; reps = 4)
    @test calls[] == 5
    @test 0 <= t.min <= t.median
    @test_throws r"^ArgumentError: reps must be at least 1, got 0" KALULarge.time_calls(
        () -> nothing, () -> nothing; reps = 0)
end

@testset "the phase note marks a sum outside 10 % of the total" begin
    note = KALULarge.phase_note
    # 0.1 * 10.0 == 1.0 exactly: a sum 10 % off the total still agrees
    @test note([5.5, 5.5], 10.0) == ""
    @test note([4.5, 4.5], 10.0) == ""
    @test note([0.5, 0.5], 1.0) == ""
    @test note([0.46, 0.46], 1.0) == ""
    @test note([0.54, 0.54], 1.0) == ""
    @test note([0.44, 0.44], 1.0) == "phases sum to 88.0 % of the total"
    @test note([0.56, 0.56], 1.0) == "phases sum to 112.0 % of the total"
end

@testset "an unknown device is refused, an absent one gives one line" begin
    @test_throws r"^ArgumentError: unknown device \"tpu\"" KALULarge.select_device(
        "tpu"; io = devnull)
    @test_throws r"^ArgumentError: the KA kernels run on a device array" KALULarge.main(
        ["cpu", "64"]; io = devnull)
    if Base.identify_package("CUDA") === nothing
        for (script, args) in ((KALULarge, ["cuda", "64"]), (SyncCost, ["cuda", "64"]),
            (R2Throughput, ["cuda", "64"]), (CusolverGetrf, ["64"]))
            buf = IOBuffer()
            @test script.main(args; io = buf) === nothing
            out = String(take!(buf))
            @test count(==('\n'), out) == 1
            @test startswith(out, "device cuda: not present (CUDA is not installed")
        end
    end
    metal_id = Base.identify_package("Metal")
    if Sys.isapple() && metal_id !== nothing &&
       !Base.invokelatest(Base.require(metal_id).functional)
        buf = IOBuffer()
        @test KALULarge.main(["metal", "64"]; io = buf) === nothing
        @test String(take!(buf)) ==
              "device metal: not present (Metal.functional() is false); no rows\n"
    end
end

@testset "r3_lu_blas.jl runs with OpenBLAS at n = 64" begin
    @test_throws r"^ArgumentError: unknown library \"mkl\"" R3LUBlas.main(["mkl"]; io = devnull)
    # AppleAccelerate keeps OpenBLAS as a fallback: with both forwarded, `openblas` is refused
    @test_throws r"^ArgumentError: openblas after accelerate" R3LUBlas.select_library(
        devnull, "openblas", () -> "libopenblas64_.dylib, Accelerate")
    @test R3LUBlas.select_library(devnull, "openblas", () -> "libopenblas64_.dylib")
    buf = IOBuffer()
    rows = R3LUBlas.main(["openblas", "64", "1"]; io = buf)
    out = String(take!(buf))
    @test occursin("| openblas | libblastrampoline: ", out)
    @test [(r.library, r.T, r.n) for r in rows] ==
          [("openblas", Float32, 64), ("openblas", Float64, 64)]
    for r in rows
        @test r.factorize.median > 0 && r.solve.median > 0
        @test r.residual <= 64 * eps(r.T)
    end
    if Base.identify_package("AppleAccelerate") === nothing
        buf = IOBuffer()
        @test isempty(R3LUBlas.main(["accelerate", "64", "1"]; io = buf))
        @test occursin(
            "\nlibrary accelerate: not present (AppleAccelerate is not installed on the load " *
            "path); no rows\n", String(take!(buf)))
    end
end

@testset "ka_lu_large.jl runs on JLArray at n = 64" begin
    buf = IOBuffer()
    rows = KALULarge.main(["jlarray", "64", "1"]; io = buf)
    out = String(take!(buf))
    @test occursin("# commit: ", out)
    @test occursin("# KernelAbstractions: ", out)
    @test occursin("# JLArrays.jl: ", out)
    @test occursin("# device: jlarray", out)
    @test occursin("# load: ", out)
    # JLArray has no vendor path: one KA row per method
    @test [(r.method, r.path) for r in rows] == [("LU", "KA"), ("QR", "KA")]
    for r in rows
        @test r.n == 64
        @test r.factorize.median > 0 && r.solve.median > 0
        @test all(>(0), r.phases)
        @test r.residual <= 64 * eps(Float32)
    end
    vendor_lines = filter(l -> occursin("| vendor | 64 | no extension method", l),
        split(out, '\n'))
    @test length(vendor_lines) == 2
end

@testset "the KA solve of ka_lu_large.jl is the solve of ldiv!!" begin
    G = KALULarge.GeometricSolvers
    jl = KALULarge.select_device("jlarray"; io = devnull)
    n = 8
    rng = Xoshiro(1)
    # `J` in `Float64` into `Float32` factors: the factorisation divides it by `2^e`, `e ≠ 0`
    J_host = 1024 .* (ones(n, n) + I)
    J = jl.array(J_host)
    x = jl.array(zeros(Float32, n))
    prob = G.NonlinearProblem(x, KALULarge.ka_noop)
    # `r` along the first column of `J`, near `floatmax`: `Qᴴ r` has an entry of `‖r‖₂`, about
    # `1.5 floatmax`, which overflows unless `ldiv!!` scales `r` (`k > 0`); and an `r` that it
    # copies (`k = 0`). The solution `0.9 floatmax / 2048 e₁` is finite.
    r_big = Float32.(0.9 * floatmax(Float32) .* J_host[:, 1] ./ maximum(J_host[:, 1]))
    @test norm(Float64.(r_big)) > floatmax(Float32)
    for r_host in (r_big, randn(rng, Float32, n))
        r = jl.array(r_host)
        @test (G.rhs_exponent(r) > 0) == (r_host === r_big)
        for M in (G.LUFactorization(), G.QRFactorization())
            fac = G.factorize!!(G.dense_factor(M, prob, x, nothing), M, J)
            @test fac.scale_exponent != 0
            d_ka = Array(KALULarge.ka_solve!(similar(x), fac, r))
            d_ldiv = Array(G.ldiv!!(similar(x), fac, r))
            @test d_ka == d_ldiv
            @test all(isfinite, d_ka)
        end
    end
end

@testset "script_commit outside a git checkout is unknown and writes no stderr" begin
    dir = mktempdir()
    log = joinpath(mktempdir(), "stderr.txt")
    c = open(log, "w") do f
        redirect_stderr(() -> KALULarge.script_commit(dir), f)
    end
    @test c == "unknown"
    @test isempty(read(log, String))
end

@testset "the vendor counts of sync_cost.jl name the device" begin
    @test SyncCost.phase_counts(:vendor, 64, "metal")[3] == (2, 1)
    @test SyncCost.phase_counts(:vendor, 64, "cuda")[3] == (1, 1)
    @test SyncCost.phase_counts(:ka, 64, "metal")[3] == (128, 2)
end

@testset "sync_cost.jl runs on $dev at n = 64" for dev in ("cpu", "jlarray")
    buf = IOBuffer()
    rows = SyncCost.main([dev, "64", "1"]; io = buf)
    out = String(take!(buf))
    @test occursin("# device: $dev", out)
    r = only(rows)
    @test r.n == 64
    @test length(r.phases) == 5 && all(>(0), r.phases)
    @test r.step > 0
    if dev == "cpu"
        @test r.path === :host
        @test r.waits == 0 && r.launches == 0 && r.sync == 0
    else
        @test r.path === :ka
        @test r.waits == 8
        @test r.launches == 3 * 64 + 11
        @test r.t_wait > 0
        # the launch count of the factorisation takes the Jacobian buffer as the factor storage
        jl = SyncCost.select_device("jlarray"; io = devnull)
        prob, p, x0 = SyncCost.make_problem(jl, 64)
        state = Base.invokelatest(SyncCost.init_state, SyncCost.Newton(), prob, x0, p)
        @test state.J === state.fac.factors
    end
    @test r.arithmetic ≈ r.step - r.sync
end

@testset "r2_throughput.jl: SoA and AoS give the same roots on the CPU" begin
    buf = IOBuffer()
    rows = R2Throughput.main(["cpu", "1000", "2"]; io = buf)
    out = String(take!(buf))
    @test occursin("# device: cpu", out)
    r = only(rows)
    @test (r.n_sys, r.n) == (1000, 2)
    @test r.root_diff <= R2Throughput.ROOT_RTOL
    rates = (r.rate_r2, r.rate_soa, r.rate_reuse, r.rate_ls, r.rate_cpu, r.rate_serial)
    @test all(>(0), rates)
end

@testset "SoAColumns gathers system i from the component arrays" begin
    SVector = R2Throughput.SVector
    B = [SVector{3, Float32}(i, 10i, 100i) for i in 1:4]
    soa = R2Throughput.soa_columns(B, Array)
    @test length(soa) == 4
    @test [soa[i] for i in 1:4] == B
end
