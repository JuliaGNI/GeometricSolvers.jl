# The measurement scripts of `scripts/hardware/`, `scripts/r2_throughput.jl` and
# `scripts/recycling_spike.jl` run to completion at their smallest size: `ka_lu_large.jl` on
# `JLArray`, `sync_cost.jl` on the CPU and on `JLArray`, `r3_lu_blas.jl` with OpenBLAS,
# `rl_host_lu.jl` on one thread, `r2_throughput.jl` on the CPU, whose SoA right-hand sides must
# give the roots of the AoS ones, and `recycling_spike.jl` at its smallest grid point, after its
# control. A device or a library that is not present gives one line and no rows. Scalar indexing
# of a device array is disallowed, so a script that reads a device array entry by entry fails here.

using GPUArraysCore: allowscalar
using LinearAlgebra: BLAS, I, norm, normalize, opnorm, qr, rank
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

module RLHostLU
include(joinpath(@__DIR__, "..", "..", "scripts", "hardware", "rl_host_lu.jl"))
end

module RecyclingSpike
include(joinpath(@__DIR__, "..", "..", "scripts", "recycling_spike.jl"))
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

@testset "rl_host_lu.jl runs on one thread at n = 8, and its verdict follows the rules" begin
    @test_throws r"^ArgumentError: unknown thread setting \"2\"" RLHostLU.main(["2"]; io = devnull)
    buf = IOBuffer()
    # the setting `1` sets BLAS to one thread; the rest of the suite keeps its own count
    nthreads = BLAS.get_num_threads()
    rows = try
        RLHostLU.main(["1", "4", "8", "1"]; io = buf)
    finally
        BLAS.set_num_threads(nthreads)
    end
    out = String(take!(buf))
    @test occursin("# BLAS library: ", out)
    @test occursin("# load: ", out)
    @test [(r.T, r.n, r.threads, r.base) for r in rows] ==
          [(T, 8, "1", 4) for T in RLHostLU.RL_ELTYPES]
    for r in rows
        @test r.recursion.median > 0 && r.getrf.median > 0
        @test r.ipiv_equal
        @test r.berr <= 8 * r.n * eps(real(r.T))
    end
    # each printed row reads back as the row the verdict takes
    lines = filter(l -> startswith(l, "| ") && !startswith(l, "| T "), split(out, '\n'))
    @test [(p.T, p.n, p.threads, p.ipiv_equal) for p in RLHostLU.parse_row.(lines)] ==
          [(r.T, r.n, r.threads, r.ipiv_equal) for r in rows]

    # a row runs once more where its median exceeds its minimum by more than 50 %
    @test RLHostLU.noisy((median = 1.51, min = 1.0))
    @test !RLHostLU.noisy((median = 1.5, min = 1.0))

    # synthetic rows of both settings: `r` per `n`, the same in both unless `r_all` is given
    function table(rs; r_all = rs, berr = 0.0, ns = [8, 16, 32, 64])
        [(T = T, n = n, threads = s, r = (s == "1" ? rs : r_all)[i], berr = berr)
         for T in RLHostLU.RL_ELTYPES, (i, n) in enumerate(ns), s in RLHostLU.RL_SETTINGS]
    end
    v = RLHostLU.rl_verdict(vec(table([0.5, 0.7, 0.85, 1.2])))
    @test v.rule === :build && v.bound_ok
    @test v.bands[Float64] == [(8, 32)]
    @test RLHostLU.rl_verdict(vec(table([0.5, 0.8, 0.5, 1.2]))).rule === :between
    @test RLHostLU.rl_verdict(vec(table([0.8, 0.8, 0.95, 1.2]))).rule === :stop
    # the slower setting decides: one setting over 0.75 breaks the pair
    r_all = [0.5, 0.9, 1.0, 1.0]
    @test RLHostLU.rl_verdict(vec(table([0.5, 0.5, 1.0, 1.0]; r_all))).rule === :between
    # a backward error over 8n·eps stops the build
    @test RLHostLU.rl_verdict(vec(table([0.5, 0.5, 1.0, 1.0]; berr = 1.0))).rule === :stop
    @test RLHostLU.rl_verdict(vec(table([0.5, 1.0, 0.5, 0.5]))).bands[Float64] ==
          [(8, 8), (32, 64)]
end

@testset "recycling_spike.jl runs at its smallest grid point, and its verdict follows the rules" begin
    buf = IOBuffer()
    out = RecyclingSpike.main(["125", "20", "5", "1e-2"]; io = buf)
    text = String(take!(buf))
    for key in ("commit", "machine", "julia", "BLAS")
        @test occursin("\n# $key: ", text)
    end
    # the control's order, which the script asserts, and its line of the header
    c = out.control
    @test c.recycling < c.no_recycling < c.gmres
    cc(products; converged = true) = (; products, converged)
    @test RecyclingSpike.check_control(cc(3), cc(2), cc(1)) === nothing
    for (g, n, r) in ((3, 2, 2), (2, 2, 1), (3, 1, 2))
        @test_throws r"^the control's order fails: GCRO-DR recycling" RecyclingSpike.check_control(
            cc(g), cc(n), cc(r))
    end
    for i in 1:3
        cols = [cc(3), cc(2), cc(1)]
        cols[i] = cc(cols[i].products; converged = false)
        @test_throws r"^the control does not converge in every column" RecyclingSpike.check_control(
            cols...)
    end
    @test occursin(
        "# control (m = 30, k = 10, η = 1e-8): GMRES(m) $(c.gmres), GCRO-DR no " *
        "recycling $(c.no_recycling), GCRO-DR recycling $(c.recycling)\n",
        text)
    row = only(out.rows)
    @test (row.N, row.m, row.k, row.η) == (125, 20, 5, 1e-2)
    # the preconditioner is kept across the ten solves: fewer refreshes than solves, and a solve
    # that starts on a kept preconditioner; refactorising at every solve refreshes at least once
    # per solve and starts none on a kept one
    nokeep = RecyclingSpike.run_cell(125, 20, 5, 1e-2; keep_factor = false)
    for (col, col0) in ((row.gmres, nokeep.gmres), (row.norec, nokeep.norec),
        (row.rec, nokeep.rec))
        @test col.converged
        @test col.products >= col.steps >= 10
        @test 1 <= col.refreshes < 10
        @test 1 <= col.kept <= 9
        @test col0.refreshes >= 10
        @test col0.kept == 0
    end
    three(f) = join((f(row.gmres), f(row.norec), f(row.rec)), " / ")
    @test occursin(
        "\n| 125 | 20 | 5 | 0.01 | $(row.gmres.products) | $(row.norec.products) | " *
        "$(row.rec.products) | $(RecyclingSpike.ratio_text(row)) | $(three(c -> c.steps)) | " *
        "$(three(c -> c.refreshes)) | $(three(c -> c.kept)) | $(row.rec.from_u) | " *
        "$(row.rec.from_u == 0 ? "vacuous" : "") |\n",
        text)
    @test out.verdict.rule === :not_evaluated
    @test out.verdict.vacuous == count(RecyclingSpike.vacuous, out.rows)
    @test occursin(
        "\nvacuous cells (no linear solve starts from a recycled space U): " *
        "$(out.verdict.vacuous) of 1\n", text)
    @test occursin("\nbuild rule: not evaluated (the grid is not the full grid)\n", text)

    # a GCRO-DR count is the operator's own count of its calls, `A·U` products included: ten
    # solves of the control, each from a recycled space after the first, take `k` products each
    prob, x0 = RecyclingSpike.control_problem()
    s = RecyclingSpike.GCRODR(30, 10; recycle = true)
    Jbuf = copy(prob.jacobian_prototype)
    calls = 0
    for p in 1:10
        before = RecyclingSpike.products(s)
        ok, _ = RecyclingSpike.gcrodr_newton!(s, prob.F, prob.J, Jbuf, nothing, x0, p, 1e-8)
        @test ok
        calls += RecyclingSpike.products(s) - before
    end
    @test s.au == 9 * 10
    @test RecyclingSpike.products(s) == calls == c.recycling
    # equal storage: no cycle holds more than the `m + 1` basis columns of GMRES(m)
    @test s.basis == 30 + 1
    # a solve whose right-hand side meets the tolerance takes no product, `A·U` included
    op = RecyclingSpike.CountingOperator(Jbuf, nothing, zeros(length(x0)), 0)
    y, capped = RecyclingSpike.gcrodr_solve!(s, op, zeros(length(x0)), 0.0)
    @test op.calls == 0 && iszero(y) && !capped

    # `F(x) = x − 1 + 10⁻²⁰ e₁` stays at `10⁻²⁰` at `x = 1`, where `x₁ − 10⁻²⁰` rounds to `1`:
    # the solve stops on the step test after one step, not after 100, and fails the residual test
    F1! = (r, x, p) -> (r .= x .- 1; r[1] += 1e-20; r)
    J1! = (J, x, p) -> copyto!(J, I)
    ok, steps = RecyclingSpike.gcrodr_newton!(
        RecyclingSpike.GCRODR(20, 5; recycle = false),
        F1!, J1!, zeros(4, 4), nothing, ones(4), nothing, 1e-8)
    @test (ok, steps) == (false, 1)
    # a count of products that the operator did not make stops the solve
    s3 = RecyclingSpike.GCRODR(20, 5; recycle = false)
    J3! = (J, x, p) -> (s3.arnoldi += 1; copyto!(J, I))
    @test_throws r"^GCRO-DR counted 2 products, the operator was called 1 times" RecyclingSpike.gcrodr_newton!(
        s3, F1!, J3!, zeros(4, 4), nothing, ones(4), nothing, 1e-8)

    # an identity operator takes one product, and the solve stops there
    n = 50
    op = RecyclingSpike.CountingOperator(Matrix(1.0I, n, n), nothing, zeros(n), 0)
    b = collect(range(1.0, 2.0; length = n))
    s1 = RecyclingSpike.GCRODR(20, 5; recycle = true)
    y, capped = RecyclingSpike.gcrodr_solve!(s1, op, b, 1e-10 * norm(b))
    @test op.calls == 1 && !capped
    @test isapprox(y, b; rtol = 8 * eps(Float64))
    # the cyclic shift `A eᵢ = eᵢ₊₁` from `e₁` spans `e₁ … e₆` in exact arithmetic and breaks
    # down at the sixth step, `‖w‖ = 0`: the solve is exact and leaves a finite recycled space of
    # five vectors, from eigenvalues that are the sixth roots of unity, complex pairs among them
    A = circshift(Matrix(1.0I, 6, 6), (1, 0))
    op = RecyclingSpike.CountingOperator(A, nothing, zeros(6), 0)
    s2 = RecyclingSpike.GCRODR(20, 5; recycle = true)
    e1 = [1.0; zeros(5)]
    y, capped = RecyclingSpike.gcrodr_solve!(s2, op, e1, 1e-14)
    @test op.calls == 6 && !capped
    @test norm(A * y - e1) <= 1e-12
    @test size(s2.U) == (6, 5) && all(isfinite, s2.U)

    # the harmonic Ritz basis: the eigenvalues of smallest modulus, a complex pair by its real and
    # imaginary parts, a pair cut at the `k`-th vector by its real part
    M = zeros(5, 5)
    M[1:2, 1:2] .= [0.1 -0.1; 0.1 0.1]
    M[3, 3], M[4, 4], M[5, 5] = 3.0, 1.0, 2.0
    # in a random orthonormal frame `Qr`, so that no basis vector is a coordinate vector
    Qr = Matrix(qr(randn(Xoshiro(3), 5, 5)).Q)
    for (k, span) in ((1, 1:2), (2, 1:2), (3, [1, 2, 4]), (4, [1, 2, 4, 5]))
        Pk = RecyclingSpike.harmonic_basis(Qr * M * Qr', Matrix(1.0I, 5, 5), k)
        Qs = Qr[:, span]
        @test size(Pk) == (5, k)
        @test isapprox(Pk' * Pk, I; atol = 1e-12)
        @test norm(Pk - Qs * (Qs' * Pk)) <= 1e-12
        @test rank(Qs' * Pk) == k
    end

    # the Newton loop of GCRO-DR takes the steps, the refreshes and the kept starts of
    # `Newton(; reuse, keep_factor)` with a preconditioned `GMRES`, early refreshes after an inner
    # cap included
    sp, _, z0, _, _ = RecyclingSpike.sparse_bracket_problem(125)
    for (maxiter, reuse, keep_factor, η) in Iterators.product(
        (2, 500), (1, 2, 5, RecyclingSpike.RC_REUSE), (true, false), (1e-2, 1e-6))
        kw = (; m = 20, k = 5, η, preconditioned = true, reuse, keep_factor, maxiter)
        a = RecyclingSpike.run_column(:gmres, sp, z0, RecyclingSpike.RC_PS; kw...)
        b = RecyclingSpike.run_column(:norec, sp, z0, RecyclingSpike.RC_PS; kw...)
        @test (a.converged, a.steps, a.refreshes, a.kept) ==
              (b.converged, b.steps, b.refreshes, b.kept)
        @test a.converged
        @test 0.9 * a.products <= b.products <= 1.1 * a.products
    end
    @test_throws r"^ArgumentError: GCRO-DR needs 0 < k < m, got m = 5, k = 5" RecyclingSpike.GCRODR(
        5, 5; recycle = true)

    # synthetic rows of the full grid: `r` per cell from `ratio`, recycling below no recycling,
    # and `from_u` solves from a recycled space
    function grid(ratio; norec = 1.0, conv = (_...) -> (true, true), from_u = (_...) -> 1)
        [(N = N, m = m, k = k, η = η, gmres = cc(1000; converged = conv(N, m, k, η)[1]),
             norec = cc(round(Int, 1000 * norec)),
             rec = (;
                 cc(round(Int, 1000 * ratio(N, m, k, η));
                     converged = conv(N, m, k, η)[2])...,
                 from_u = from_u(N, m, k, η)))
         for N in RecyclingSpike.RC_NS,
        m in RecyclingSpike.RC_MS, k in RecyclingSpike.RC_KS,
        η in RecyclingSpike.RC_ETAS] |> vec
    end
    # a grid in which no linear solve starts from a recycled space measures nothing, whatever its
    # ratios; one cell that does makes the rules apply
    for ratio in (1.0, 0.5)
        rows = grid((_...) -> ratio; from_u = (_...) -> 0)
        v = RecyclingSpike.rc_verdict(rows)
        @test v.rule === :vacuous && v.vacuous == 16 && isempty(v.pairs)
        vbuf = IOBuffer()
        RecyclingSpike.print_verdict(vbuf, rows, v)
        @test String(take!(vbuf)) ==
              "vacuous cells (no linear solve starts from a recycled space U): 16 of 16\n" *
              "build rule: not evaluated (every cell is vacuous)\n" *
              "stop rule: not evaluated (every cell is vacuous)\n" *
              "result: the further spike measures nothing\n"
    end
    hit = (768, 30, 10, 1e-6)
    rows = grid((_...) -> 1.08; from_u = (N, m, k, η) -> (N, m, k, η) == hit ? 3 : 0)
    v = RecyclingSpike.rc_verdict(rows)
    @test v.rule === :stop && v.vacuous == 15
    vbuf = IOBuffer()
    RecyclingSpike.print_verdict(vbuf, rows, v)
    @test startswith(String(take!(vbuf)),
        "vacuous cells (no linear solve starts from a recycled space U): 15 of 16\n" *
        "build rule (r ≤ 0.75 and below GCRO-DR without recycling, at both N and both η): " *
        "not met\nstop rule (r ≥ 0.90 in every cell): met (16 of 16 cells)\nresult: stop\n")
    verdict = RecyclingSpike.rc_verdict
    @test verdict(grid((_...) -> 1.08)).rule === :stop
    @test verdict(grid((_...) -> 0.90)).rule === :stop
    @test verdict(grid((_...) -> 0.89)).rule === :between
    # the build rule needs one pair `(m, k)` at both `N` and both `η`
    v = verdict(grid((N, m, k, η) -> (m, k) == (30, 10) ? 0.75 : 1.0))
    @test v.rule === :build && v.pairs == [(30, 10)]
    @test verdict(grid((N, m, k, η) -> (m, k) == (30, 10) ? 0.76 : 1.0)).rule === :between
    @test verdict(grid((N, m, k, η) -> (m, k, N) == (30, 10, 125) ? 0.5 : 1.0)).rule ===
          :between
    @test verdict(grid((N, m, k, η) -> (m, k, η) == (30, 10, 1e-2) ? 0.5 : 1.0)).rule ===
          :between
    # recycling must also take fewer products than no recycling
    @test verdict(grid((_...) -> 0.5; norec = 0.5)).rule === :between
    # a GMRES(m) that does not converge meets neither rule; a recycling column that does not
    # converge beside one that does counts as `r ≥ 0.90`
    @test verdict(grid((_...) -> 1.0; conv = (N, m, k, η) -> (N == 125, true))).rule ===
          :between
    @test verdict(grid((_...) -> 0.5; conv = (N, m, k, η) -> (N == 125, true))).rule ===
          :between
    @test verdict(grid((N, m, k, η) -> N == 125 ? 1.0 : 0.5;
        conv = (N, m, k, η) -> (true, N == 125))).rule === :stop
    @test verdict(grid((_...) -> 0.5)[1:(end - 1)]).rule === :not_evaluated
    nc = RecyclingSpike.cell_text(cc(7; converged = false))
    @test nc == "no convergence"
end

@testset "recycling_spike.jl: GCRO-DR on the control matrix scaled near floatmax" begin
    # `s A y = s b` for the control matrix `A` and three unit right-hand sides, each solve after
    # the first from the recycled space of the one before: at `s = 4 √floatmax` a product of two
    # Hessenberg entries overflows, and at `s = floatmax / 8` a Householder reflector of a
    # Hessenberg column and a product of a Hessenberg entry with a coefficient above 1 do too
    prob, _ = RecyclingSpike.control_problem()
    A = copy(prob.jacobian_prototype)
    n = size(A, 1)
    B = randn(Xoshiro(7), n, 3)
    B ./= norm.(eachcol(B))'
    function scaled_run(s)
        op = RecyclingSpike.CountingOperator(s .* A, nothing, zeros(n), 0)
        g = RecyclingSpike.GCRODR(30, 10; recycle = true)
        ys = map(eachcol(B)) do b
            y, capped = RecyclingSpike.gcrodr_solve!(g, op, s .* b, 1e-8 * s)
            @test !capped
            return y
        end
        return ys, g
    end
    ys1, g1 = scaled_run(1.0)
    @test g1.au == 2 * 10
    Q1 = Matrix(qr(g1.U).Q)
    for s in (4 * sqrt(floatmax(Float64)), floatmax(Float64) / 8)
        ys, g = scaled_run(s)
        @test g.au == g1.au
        @test all(isfinite, g.U)
        # the sine of the largest principal angle between the recycled spaces at `s` and at 1
        Qs = Matrix(qr(g.U).Q)
        @test opnorm(Qs - Q1 * (Q1' * Qs)) < sqrt(eps(Float64))
        # the solution is that at scale 1, to the round-off of `s A`, which the condition number
        # of `A`, about 10⁴, amplifies
        for (y, y1) in zip(ys, ys1)
            @test norm(y - y1) <= 1e-10 * norm(y1)
        end
    end
end

@testset "recycling_spike.jl: the second Gram–Schmidt pass of the Arnoldi process" begin
    # a unit vector at the distance 10⁻¹⁰ from the span of an orthonormal `V`: one pass of
    # classical Gram–Schmidt leaves a loss of orthogonality of about eps / 10⁻¹⁰
    n = 400
    V = Matrix(qr(randn(Xoshiro(11), n, 20)).Q)
    w0 = V * randn(Xoshiro(12), 20)
    u = randn(Xoshiro(13), n)
    u .-= V * (V' * u)
    w0 .+= 1e-10 .* norm(w0) .* normalize(u)
    loss(w) = (Vw = hcat(V, w ./ norm(w)); opnorm(Vw' * Vw - I))
    one_pass = w0 - V * (V' * w0)
    @test loss(one_pass) > sqrt(eps(Float64))
    w = copy(w0)
    h = RecyclingSpike.orthogonalize!(w, V)
    @test loss(w) < n * eps(Float64)
    # the coefficients of the two passes: `w0 = V h + w`
    @test norm(w0 - (V * h + w)) <= 8 * eps(Float64) * norm(w0)
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
        for M in (G.LUFactorization(), G.QRFactorization())
            fac = G.factorize!!(G.dense_factor(M, prob, x, nothing), M, J)
            @test fac.scale_exponent != 0
            @test (G.rhs_exponent(r, fac.scale_exponent) > 0) == (r_host === r_big)
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
