# Self-test of the benchmark harness: the pairing, the label and the method of each call, the
# options of each package, the counting wrapper, the shared Jacobian, the measured `‖F‖∞`, the
# success rule, the agreement bound, the verdict, a throwing residual, the SVD solve on a
# non-finite matrix, and the element type and the start of the midpoint rows; and for the
# timings and the blocks: the timed runs and the batch, the note of a timed cell, the tolerance
# of each adapter, the variant labels, the `nsol` adapter, the sparse variants and the profiles;
# and for the device rows: the machine–backend pairs, the inline adapter in a `CPU()` kernel, the
# verdict, the band, the classifier of a launch that throws, the Metal heap restart, the order of
# the clock and `synchronize` in a timed run, the time cap, the two LU paths, and the R2 and R3
# lines.
#
# Usage: julia --project=benchmark benchmark/selftest.jl

using Test

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "packages.jl"))
include(joinpath(@__DIR__, "timing.jl"))
include(joinpath(@__DIR__, "blocks.jl"))
include(joinpath(@__DIR__, "profiles.jl"))

# The 2-d Rosenbrock system, with its root (1, 1).
function rosenbrock!(y, x, p)
    y[1] = 1 - x[1]
    y[2] = 10 * (x[2] - x[1] * x[1])
    return nothing
end

rosenbrock_row(T) = Row{T}("selftest", "rosenbrock", "x0", rosenbrock!, T[-1.2, 1])

@testset "harness" begin
    @testset "pairing: the label of each call is the table's" begin
        @test COLUMNS == [
            "Newton Static LU", "Newton Backtracking LU", "Newton Bisection LU",
            "Newton StrongWolfe LU", "Newton DogLeg LU", "Newton Backtracking QR",
            "Newton Backtracking SVD", "Newton LevenbergMarquardt SVD",
            "Newton reuse 3 Backtracking LU", "Picard Backtracking",
            "Newton Backtracking GMRES", "Broyden",
            "LeftPreconditioned Picard Backtracking Newton"]
        @test MAX_ITERATIONS == 200
        @test Set(keys(PAIRING)) == Set(COLUMNS)
        for column in COLUMNS, (package, entry) in zip(PACKAGES, PAIRING[column])

            entry isa Paired || continue
            for T in (Float32, Float64)
                @test method_label(entry.build(T)) == entry.label
            end
        end
        # The GeometricSolvers method of each column is the standard test set's.
        for (column, m) in STANDARD_COLUMNS
            @test PAIRING[column][1].build(Float64).method === m
        end
        @test PAIRING["Newton Backtracking LU"][2].label == "Newton+Backtracking+LapackLU"
        @test PAIRING["Newton Static LU"][1].label == "Newton+Static+LUFactorization"
        @test PAIRING["Newton reuse 3 Backtracking LU"][1].label ==
              "Newton(3)+Backtracking+LUFactorization"
        @test count(e -> e isa Paired, Iterators.flatten(values(PAIRING))) == 13 + 20
        # The nine `Newton` columns with a factorisation take the shared Jacobian, and the
        # GeometricSolvers adapter hands it to no other column: a Jacobian that throws reaches
        # the solve on those nine alone.
        @test [column for (column, m) in STANDARD_COLUMNS if gs_takes_jacobian(m)] ==
              COLUMNS[1:9]
        nojac(J, x, p) = error("the Jacobian is called")
        for (k, column) in enumerate(COLUMNS)
            out = call_package(Val(:GeometricSolvers), PAIRING[column][1].build(Float64),
                Counted(rosenbrock!), nojac, [-1.2, 1.0])
            @test (out.result isa Thrown) == (k ≤ 9)
        end
    end

    @testset "every paired method solves the Rosenbrock system" begin
        for T in (Float32, Float64), column in COLUMNS, package in PACKAGES
            c = run_cell(rosenbrock_row(T), column, package)
            entry = PAIRING[column][findfirst(==(package), PACKAGES)]
            if package == "NLsolve" && T === Float32
                @test c.status == "no Float32"
            elseif entry isa NoMethod
                @test c.status == "no method" && c.comment == entry.reason
            else
                @test c.status == "ran" && c.code != "ERROR"
                @test c.method == entry.label
                @test c.f_evals == length(c.trace) && c.f_evals > 0
            end
        end
        # The paired Newton methods with a line search reach the root.
        for package in ("GeometricSolvers", "SimpleSolvers", "NLsolve")
            c = run_cell(rosenbrock_row(Float64), "Newton Backtracking LU", package)
            @test c.success && agrees(c.x, [1.0, 1.0], Float64)
        end
    end

    @testset "the label is the adapter's: it names the objects that the call passes" begin
        for T in (Float32, Float64)
            objs = PAIRING["Newton Backtracking LU"][2].build(T)
            out = call_package(Val(:SimpleSolvers), objs, Counted(rosenbrock!),
                SharedJacobian(Counted(rosenbrock!)), T[-1.2, 1])
            @test out.method == "Newton+Backtracking+LapackLU"
        end
        # The method that each package runs shows in its result: from this start the full Newton
        # steps of `atan` diverge, and every globalised or derivative-free method of the table
        # reaches the root. A call that passes a plain Newton step in place of its method fails.
        far(y, x, p) = (y[1] = atan(x[1]); nothing)
        row = Row{Float64}("selftest", "atan", "x0", far, [2.0])
        for column in COLUMNS, package in PACKAGES

            c = run_cell(row, column, package)
            c.status == "ran" || continue
            @test c.success == (column != "Newton Static LU")
        end
    end

    @testset "the options of each package: the absolute tolerance alone stops a solve" begin
        # A double root: Newton's error halves at each step, so ‖F‖ falls by 4 per step, and
        # GeometricSolvers' default relative tolerance, sqrt(eps) times ‖F(x₀)‖ = 8.1e5, would
        # end its solve as a success at a ‖F‖ near 0.01, far above τ.
        double(y, x, p) = (y[1] = 1e4 * (x[1] - 1)^2; nothing)
        row = Row{Float64}("selftest", "double", "x0", double, [10.0])
        for package in PACKAGES
            c = run_cell(row, "Newton Backtracking LU", package)
            c.status == "ran" || continue
            @test c.success && c.fnorm ≤ tolerance(Float64)
        end
        @test count(p -> run_cell(row, "Newton Backtracking LU", p).status == "ran",
            PACKAGES) == 3
        # A round-off floor: ‖F‖∞ cannot fall below about 4e-6 > τ near √2, while a relative
        # tolerance of sqrt(eps) on ‖F(x₀)‖ = 9.8e11 accepts it. GeometricSolvers and
        # SimpleSolvers report no success there, and no cell succeeds.
        floored(y, x, p) = (y[1] = 1e10 * (x[1]^2 - 2); nothing)
        row = Row{Float64}("selftest", "floor", "x0", floored, [10.0])
        cells = [run_cell(row, "Newton Backtracking LU", p) for p in PACKAGES]
        @test cells[1].code != "SUCCESS"
        @test cells[2].code == "not converged"
        @test !any(c -> c.success, cells)
    end

    @testset "fnorm is ‖F(x)‖∞ at the returned iterate" begin
        # No root, and ‖F‖₂ ≠ ‖F‖∞ at every x.
        noroot(y, x, p) = (y[1] = x[1]^2 + 1; y[2] = 2 * (x[2]^2 + 1); nothing)
        row = Row{Float64}("selftest", "noroot", "x0", noroot, [0.5, 0.5])
        for package in PACKAGES
            c = run_cell(row, "Newton Backtracking LU", package)
            c.status == "ran" && c.x !== nothing || continue
            y = zeros(2)
            noroot(y, c.x, nothing)
            @test norm(y) != norm(y, Inf)
            @test c.fnorm == norm(y, Inf)
        end
    end

    @testset "the SVD solve keeps a non-finite matrix from LAPACK" begin
        # LAPACK on Apple silicon throws an `ArgumentError` of its own on such a matrix, so the
        # message tells which one threw.
        p = fill(7.0, 2)
        @test_throws ArgumentError("the matrix has a non-finite entry") SVDSolve()(p,
            [NaN 1.0; 1.0 1.0], [1.0, 1.0])
        @test p == [7.0, 7.0]
        @test SVDSolve()(p, [2.0 0.0; 0.0 4.0], [1.0, 1.0]) ≈ [0.5, 0.25]
    end

    @testset "counting wrapper and shared Jacobian" begin
        c = Counted(rosenbrock!)
        J! = SharedJacobian(c)
        J = zeros(2, 2)
        J!(J, [2.0, 3.0], nothing)
        @test J == [-1.0 0.0; -40.0 10.0]
        @test c.primal == 0 && c.dual ≥ 1
        y = zeros(2)
        c(y, [2.0, 3.0], nothing)
        @test c.primal == 1 && c.trace == [10.0]
    end

    @testset "success rule" begin
        for T in (Float32, Float64)
            τ = tolerance(T)
            @test τ == sqrt(eps(T))
            @test cell_success(true, T[1, 2], τ, T)
            @test cell_success(true, T[1, 2], τ / 2, T)
            @test !cell_success(true, T[1, 2], 2τ, T)       # the package's success alone
            @test !cell_success(true, T[1, 2], T(NaN), T)
            @test !cell_success(true, T[1, 2], T(Inf), T)
            @test !cell_success(true, T[1, NaN], zero(T), T)  # a non-finite iterate
            @test !cell_success(true, T[Inf, 1], zero(T), T)
            @test !cell_success(false, T[1, 2], zero(T), T)
        end
    end

    @testset "agreement bound" begin
        for T in (Float32, Float64)
            τ = tolerance(T)
            for s in (one(T), T(1e3))
                xa = T[s, -s]
                @test agrees(xa, xa .+ 2 * eps(T) * s, T)
                @test agrees(xa, xa .+ τ * s / 2, T)
                @test !agrees(xa, xa .+ 2τ * s, T)
            end
            # a zero solution: the bound is τ, from max(1, ‖xa‖∞)
            @test agrees(zeros(T, 2), T[τ / 2, 0], T)
            @test !agrees(zeros(T, 2), T[2τ, 0], T)
            # below 1 the bound stays τ
            @test agrees(T[1e-3, 0], T[1e-3 + τ / 2, 0], T)
        end
    end

    @testset "verdict" begin
        T = Float64
        x, y = [1.0, 1.0], [1.0, 1.0 + 1e-3]
        @test verdict([true], [x], T) == "single"
        @test verdict([false], [x], T) == "single"
        @test verdict([false, false], [x, x], T) == "all fail"
        @test verdict([true, false], [x, x], T) == "disagree"
        @test verdict([false, true, true], [x, x, x], T) == "disagree"
        @test verdict([true, true, true], [x, x, x], T) == "agree"
        @test verdict([true, true], [x, y], T) == "different root"
        @test verdict([true, true, true], [x, x, y], T) == "different root"
        # the cells that did not run take no part
        cells = [run_cell(rosenbrock_row(Float32), "Broyden", p) for p in PACKAGES]
        @test count(c -> c.status == "ran", cells) == 2
        @test verdict(cells) in ("all fail", "disagree", "agree", "different root")
        cells = [run_cell(rosenbrock_row(Float64), "Newton LevenbergMarquardt SVD", p)
                 for p in PACKAGES]
        @test verdict(cells) == "single"
    end

    @testset "a throwing residual" begin
        boom(y, x, p) = throw(DomainError(x[1], "boom"))
        row = Row{Float64}("selftest", "boom", "x0", boom, [1.0, 2.0])
        for column in COLUMNS, package in PACKAGES

            c = run_cell(row, column, package)
            c.status == "ran" || continue
            @test c.code == "ERROR" && !c.success && c.comment == "DomainError"
            @test c.x === nothing && c.fnorm === nothing
        end
        stop(y, x, p) = throw(InterruptException())
        row = Row{Float64}("selftest", "stop", "x0", stop, [1.0, 2.0])
        for package in PACKAGES
            @test_throws InterruptException run_cell(row, "Newton Static LU", package)
        end
        # A residual that throws only at the harness's own evaluation of the returned iterate:
        # the first run counts the calls, and the second throws at the call after the solve's.
        calls = Ref(0)
        limit = Ref(typemax(Int))
        function late!(y, x, p)
            (calls[] += 1) > limit[] && throw(DomainError(x[1], "late"))
            return rosenbrock!(y, x, p)
        end
        row = Row{Float64}("selftest", "late", "x0", late!, [-1.2, 1])
        for package in PACKAGES
            calls[], limit[] = 0, typemax(Int)
            @test run_cell(row, "Newton Static LU", package).code != "ERROR"
            calls[], limit[] = 0, calls[] - 1
            c = run_cell(row, "Newton Static LU", package)
            @test c.code == "ERROR" && !c.success && c.comment == "DomainError"
            @test c.x === nothing && c.fnorm === nothing
        end
    end

    @testset "the rows" begin
        for T in (Float32, Float64)
            rows = benchmark_rows(T)
            @test length(rows) == 79
            @test count(r -> r.set == "standard", rows) == 67
            @test all(r -> eltype(r.x0) === T && all(isfinite, r.x0), rows)
            for r in rows
                y = similar(r.x0)
                r.F(y, r.x0, nothing)
                @test eltype(y) === T && all(isfinite, y)
            end
        end
    end

    @testset "element type of the midpoint rows" begin
        for (name, h) in MIDPOINT_SCENARIOS, T in (Float32, Float64)

            prob = midpoint_problem(name, h, T)
            @test all(v -> eltype(v) === T, values(prob.parameters))
            @test all(v -> eltype(collect(v)) === T, values(prob.ics))
            @test typeof(prob.timestep) === T
        end
    end

    @testset "the start of each midpoint row is the vector field at the initial state" begin
        for (name, h) in MIDPOINT_SCENARIOS, T in (Float32, Float64)

            prob = midpoint_problem(name, h, T)
            fs, par, t₀ = prob.functions, prob.parameters, prob.timespan[1]
            q₀ = collect(prob.ics.q)
            x0 = midpoint_system(prob).x0
            if haskey(fs, :ϑ)
                v̄ = similar(q₀)
                prob.initialguess.v(v̄, t₀, q₀, collect(prob.ics.p), par)
                @test x0 == v̄
            elseif haskey(fs, :f)
                p₀ = collect(prob.ics.p)
                v, f = similar(q₀), similar(q₀)
                fs.v(v, t₀, q₀, p₀, par)
                fs.f(f, t₀, q₀, p₀, par)
                @test x0 == [v; f]
            else
                v = similar(q₀)
                fs.v(v, t₀, q₀, par)
                @test x0 == v
            end
            @test any(!iszero, x0)
        end
    end
end

# A system on which `nsol` with its default `sham = 5` skips a Jacobian: (x₁² + x₂² − 2,
# exp(x₁ − 1) + x₂³ − 2), with the root (1, 1), from (2, 0.5).
function skips!(y, x, p)
    y[1] = x[1]^2 + x[2]^2 - 2
    y[2] = exp(x[1] - 1) + x[2]^3 - 2
    return nothing
end

# A double root in the first component: Newton's error halves at each step, so ‖F‖∞ falls by 4
# per step. Two components, because `nsol` takes no system of dimension 1.
double!(y, x, p) = (y[1] = 1e4 * (x[1] - 1)^2; y[2] = x[2] - 1; nothing)

# A clock in nanoseconds that only `elapse!` moves, so that every time the self-test checks is
# exact and no check depends on the wall clock.
mutable struct FakeClock
    ns::UInt64
end
(c::FakeClock)() = c.ns
elapse!(c::FakeClock, t) = (c.ns += round(UInt64, t * 1e9); nothing)

@testset "Y2" begin
    @testset "run count: five timed runs, and the batch reaches MIN_RUN" begin
        @test N_RUNS == 5 && MIN_RUN == 1e-3
        row = rosenbrock_row(Float64)
        # A solve of 0.2 ms: 4 solves last 0.8 ms, 8 solves 1.6 ms, and 5 solves, which no
        # power of two gives, 1.0 ms.
        clock = FakeClock(0)
        slow = c -> (; passed = (;), call = () -> elapse!(clock, 2e-4))
        (; k, times) = time_cell(row, slow; clock)
        @test k == 8
        @test times == fill(2e-4, 5)
        # A first call that takes 2 ms, as one that compiles does, does not decide the batch.
        first = Ref(true)
        compiles = c -> (; passed = (;),
            call = () -> elapse!(clock, first[] ? (first[] = false; 2e-3) : 2e-4))
        @test time_cell(row, compiles; clock).k == 8
        # One slow run in the search, as a pause for garbage collection gives, does not decide
        # the batch: the second call, the second run of the search at k = 1, takes 2 ms.
        calls = Ref(0)
        pauses = c -> (; passed = (;),
            call = () -> elapse!(clock, (calls[] += 1) == 2 ? 2e-3 : 2e-4))
        @test time_cell(row, pauses; clock).k == 8
        # The call and its copy of the start are made before the timer starts: a call that takes
        # 1 ms to make does not enter the time of its solve of 0.2 ms.
        makes = c -> (
            elapse!(clock, 1e-3); (; passed = (;), call = () -> elapse!(clock, 2e-4)))
        @test time_cell(row, makes; clock).times == fill(2e-4, 5)
        # A smoke test on the real clock, with no bound on the time. Each solve takes its own
        # copy of the start, and the row's start stays as it was.
        invoke = adapter(
            "GeometricSolvers", PAIRING["Newton Backtracking LU"][1].build(Float64),
            row; tol = tolerance(Float64))
        (; k, times) = time_cell(row, invoke)
        @test ispow2(k) && length(times) == 5
        @test row.x0 == [-1.2, 1.0]
        # A throw ends a solve, and the run goes on.
        boom(y, x, p) = throw(DomainError(x[1], "boom"))
        brow = Row{Float64}("selftest", "boom", "x0", boom, [1.0, 2.0])
        binvoke = adapter(
            "GeometricSolvers", PAIRING["Newton Static LU"][1].build(Float64),
            brow; tol = tolerance(Float64))
        @test length(time_cell(brow, binvoke).times) == 5
        @test run_median([1.0, 2.0, 3.0, 4.0, 100.0]) == 3.0
        @test run_spread([1.0, 2.0, 3.0, 4.0, 100.0]) == 33.0
        @test run_median([4.0, 1.0, 3.0, 2.0]) == 2.5
        # A worker is a fresh process of this environment, with no startup file and one thread;
        # it sets one BLAS thread, and refuses to time with more than one Julia thread.
        cmd = worker_cmd("w.jl", "a", "b")
        @test "--startup-file=no" in cmd.exec && "--threads=1" in cmd.exec
        @test "--project=$(@__DIR__)" in cmd.exec &&
              cmd.exec[(end - 2):end] == ["w.jl", "a", "b"]
        if Threads.nthreads() == 1
            start_worker("selftest")
        else
            @test_throws r"a timing process runs" start_worker("selftest")
        end
        @test BLAS.get_num_threads() == 1
        @test eltype_named("Float32") === Float32 && eltype_named("Float64") === Float64
        @test_throws r"no element type" eltype_named("Float16")
    end

    @testset "the note of a timed cell" begin
        row = rosenbrock_row(Float64)
        c = run_cell(row, "Newton Backtracking LU", "GeometricSolvers")
        @test timing_note(c, c.code, c.iterations, string(c.f_evals)) == ""
        note = timing_note(c, c.code, "99", string(c.f_evals))
        @test note == "differs from correctness.csv: iterations 99 → $(c.iterations)"
        @test timing_note(c, c.code, c.iterations, "999") ==
              "differs from correctness.csv: f_evals 999 → $(c.f_evals)"
        @test timing_note(c, "-7", c.iterations, string(c.f_evals)) ==
              "differs from correctness.csv: code -7 → $(c.code)"
        boom(y, x, p) = throw(DomainError(x[1], "boom"))
        e = run_cell(Row{Float64}("selftest", "boom", "x0", boom, [1.0, 2.0]),
            "Newton Static LU", "GeometricSolvers")
        @test timing_note(e, "ERROR", "", string(e.f_evals)) == "throws"
    end

    @testset "tol reaches every adapter and the success rule" begin
        row = Row{Float64}("selftest", "double", "x0", double!, [10.0, 3.0])
        for package in PACKAGES
            c = run_cell(row, "Newton Static LU", package; tol = 1e-2)
            @test c.success && tolerance(Float64) < c.fnorm ≤ 1e-2
        end
        for (v, (package, objs)) in MIXED_VARIANTS
            c = run_invoked(row, v, package, adapter(package, objs, row; tol = 1e-2);
                tol = 1e-2)
            @test c.success && tolerance(Float64) < c.fnorm ≤ 1e-2
        end
        @test cell_success(true, [1.0], 1e-3, Float64; tol = 1e-2)
        @test !cell_success(true, [1.0], 1e-3, Float64)
        @test STAGE_TOL === 0.0009765625f0
        stage = BlockCell("stage", "Float32", ("midpoint", "Pendulum h=0.1", "V0"),
            "Newton Backtracking LU", "GeometricSolvers")
        rows32 = Dict(rowkey(r) => r for r in benchmark_rows(Float32))
        @test block_run(stage, rows32).tol === 0.0009765625f0
        line = split(block_line(stage, rows32), ",")
        @test length(line) == 24 && line[8] == "ran" && line[15] == "0.0009765625f0"
        @test ispow2(parse(Int, line[23]))
        # The five times are numbers; their values are the real clock's, and no check reads them.
        @test all(t -> tryparse(Float64, t) !== nothing, line[16:20])
        # The package receives the stage tolerance: this cell stops above √eps(Float32).
        loose = BlockCell(
            "stage", "Float32", ("midpoint", "HarmonicOscillator h=0.1", "V0"),
            "Broyden", "GeometricSolvers")
        run = block_run(loose, rows32)
        c = run_invoked(run.row, loose.variant, loose.package, run.invoke; run.tol, run.p)
        @test c.success && sqrt(eps(Float32)) < c.fnorm ≤ STAGE_TOL
        # The stage cells that do not run: NLsolve's, and a pair without a method.
        for (package, column, status) in (
            ("NLsolve", "Newton Backtracking LU", "no Float32"),
            ("SimpleNonlinearSolve", "Newton Backtracking LU", "no method"))
            b = BlockCell("stage", "Float32", ("midpoint", "Pendulum h=0.1", "V0"), column,
                package)
            @test block_run(b, rows32).status == status
            line = split(block_line(b, rows32), ",")
            @test length(line) == 24 && line[8] == status
            @test all(isempty, line[9:23]) && !isempty(line[24])
        end
    end

    @testset "variant labels" begin
        labels = Dict(v => method_label(objs) for (v, (_, objs)) in MIXED_VARIANTS)
        @test labels == Dict(
            "gs-f64" => "Newton+Backtracking+LUFactorization",
            "gs-f32" => "Newton+Backtracking+LUFactorization{Float32}",
            "gs-f32-refine" => "Newton+Backtracking+LUFactorization{Float32}+Refinement",
            "nsol-f32" => "nsol(sham=1)+FPS{Float32}",
            "nsol-mplu" => "nsol(sham=1)+mplu")
        @test [v => method_label(objs) for (v, (_, objs)) in ANDERSON_VARIANTS] == [
            "anderson-1" => "Anderson{4}(Newton(1)+Backtracking+LUFactorization)",
            "newton-1" => "Newton(1)+Backtracking+LUFactorization",
            "anderson-2" => "Anderson{4}(Newton(2)+Backtracking+LUFactorization)",
            "newton-2" => "Newton(2)+Backtracking+LUFactorization",
            "anderson-3" => "Anderson{4}(Newton(3)+Backtracking+LUFactorization)",
            "newton-3" => "Newton(3)+Backtracking+LUFactorization",
            "anderson-5" => "Anderson{4}(Newton(5)+Backtracking+LUFactorization)",
            "newton-5" => "Newton(5)+Backtracking+LUFactorization"]
        # The labels name the objects that the solve takes: the factor precision and the
        # refinement of a GeometricSolvers `Newton`, the depth and the reuse of `Anderson`.
        for (_, (package, objs)) in MIXED_VARIANTS
            package == "GeometricSolvers" || continue
            m = objs.method
            @test (m.linsolve isa GS.LUFactorization{Float32}) ==
                  occursin("{Float32}", method_label(objs))
            @test (m.refinement isa GS.Refinement) ==
                  occursin("Refinement", method_label(objs))
        end
        for (v, (_, objs)) in ANDERSON_VARIANTS
            m = objs.method isa GS.Anderson ? objs.method.member : objs.method
            @test v[end] == only(string(m.reuse))
        end
        # Every Anderson variant takes the shared Jacobian: a Jacobian that throws ends its solve.
        nojac(J, x, p) = error("the Jacobian is called")
        for (_, (_, objs)) in ANDERSON_VARIANTS
            @test gs_takes_jacobian(objs.method)
            out = call_package(Val(:GeometricSolvers), objs, Counted(rosenbrock!), nojac,
                [-1.2, 1.0])
            @test out.result isa Thrown
        end
        # The block tables' counts.
        @test length.(block_cells.(BLOCKS)) == (395, 624, 1264, 20)
    end

    @testset "nsol: a Jacobian at every iteration, and a start that meets the test" begin
        row = Row{Float64}("selftest", "skips", "x0", skips!, [2.0, 0.5])
        for v in ("nsol-f32", "nsol-mplu")
            package, objs = Dict(MIXED_VARIANTS)[v]
            invoke = adapter(package, objs, row; tol = tolerance(Float64))
            c = run_invoked(row, v, package, invoke; tol = tolerance(Float64))
            @test c.success && c.code == "0"
            # The shared Jacobian of a 2-d residual takes one dual call.
            @test parse(Int, c.iterations) > 2 && c.dual_evals == parse(Int, c.iterations)
            @test agrees(c.x, [1.0, 1.0], Float64)
        end
        # The control: with `sham = 5` the same solve skips a Jacobian.
        invoke = adapter("SIAMFANLEquations", (; nsol = Nsol(Float32, 5)), row;
            tol = tolerance(Float64))
        c = run_invoked(
            row, "sham 5", "SIAMFANLEquations", invoke; tol = tolerance(Float64))
        @test c.dual_evals < parse(Int, c.iterations)
        # From the root, nsol returns errcode -1 with idid = false: a reported success, 0
        # iterations.
        root = Row{Float64}("selftest", "skips", "root", skips!, [1.0, 1.0])
        for v in ("nsol-f32", "nsol-mplu")
            package, objs = Dict(MIXED_VARIANTS)[v]
            invoke = adapter(package, objs, root; tol = tolerance(Float64))
            c = run_invoked(root, v, package, invoke; tol = tolerance(Float64))
            @test c.code == "-1" && c.success && c.iterations == "0"
        end
        # The adapter passes the method's `jfact` to `nsol`, and none for `nothing`.
        calls = Ref(0)
        spy(A) = (calls[] += 1; MultiPrecisionArrays.mplu(A))
        invoke = adapter("SIAMFANLEquations", (; nsol = Nsol(Float64, 1, spy)), row;
            tol = tolerance(Float64))
        c = run_invoked(row, "spy", "SIAMFANLEquations", invoke; tol = tolerance(Float64))
        @test c.success && calls[] == parse(Int, c.iterations)
        @test c.method == "nsol(sham=1)+spy"
        # The nsol wrappers return their first argument.
        y, J = zeros(2), zeros(2, 2)
        @test NsolResidual(Counted(rosenbrock!))(y, [2.0, 3.0]) === y
        @test NsolJacobian(SharedJacobian(Counted(rosenbrock!)))(J, y, [2.0, 3.0]) === J
        @test J == [-1.0 0.0; -40.0 10.0]
    end

    @testset "the sparse variants" begin
        want = Dict(
            "gs-sparse" => "Newton+Static+LUFactorization+AutoSparse+SparseMatrixCSC",
            "gs-dense" => "Newton+Static+LUFactorization+SharedJacobian",
            "nlsolve-sparse" => "newton+AnalyticJacobian+SparseMatrixCSC")
        for (v, p) in SPARSE_VARIANTS, T in (Float32, Float64)

            b = BlockCell("sparse", string(T), ("sparse", "bracket N=125", "z0"), v, p)
            run = block_run(b, Dict())
            if v in ("simplesolvers", "simplenonlinearsolve")
                @test run.status == "no sparse path"
            elseif v == "nlsolve-sparse" && T === Float32
                @test run.status == "no Float32"
            else
                sp = sparse_row(125, T)
                c = run_invoked(run.row, v, p, run.invoke; run.tol, run.p)
                @test c.method == want[v]
                @test c.success && norm(c.x - sp.z_star, Inf) ≤ sqrt(eps(T))
                @test run.p === sp.p && run.tol === sqrt(eps(T))
            end
        end
        # The size of a cell's problem follows the N of its key.
        for N in SPARSE_SIZES
            b = BlockCell(
                "sparse", "Float64", ("sparse", "bracket N=$N", "z0"), "gs-sparse",
                "GeometricSolvers")
            @test length(block_run(b, Dict()).row.x0) == 2N
        end
        sp = sparse_row(125, Float64)
        @test size(sp.prototype) == (250, 250)
        # NLsolve's residual and analytic Jacobian are the problem's at its parameter: the
        # Jacobian equals the dense ForwardDiff one at a point off the root.
        z = sp.row.x0 .+ 0.3
        J, Jd = copy(sp.prototype), zeros(250, 250)
        AnalyticJacobian(sp.J, sp.p)(J, z)
        SharedJacobian(sp.row.F)(Jd, z, sp.p)
        @test isapprox(Matrix(J), Jd; rtol = 8 * eps(Float64))
        y, yd = zeros(250), zeros(250)
        ResidualAt(sp.row.F, sp.p)(y, z)
        sp.row.F(yd, z, sp.p)
        @test y == yd && norm(y) > 0
    end

    @testset "the merge of the parts keeps the order of the cells, and checks it" begin
        # The key of a cell of problem 1 from u0, and the line of a cell that holds `rest`.
        key(pkg, el) = "standard,1,u0,$el,Newton Static LU,$pkg"
        line(pkg, el, rest) = key(pkg, el) * "," * rest
        mktempdir() do dir
            cells = [line("NLsolve", "Float64", "ran"),
                line("GeometricSolvers", "Float32", "ran"),
                line("SimpleSolvers", "Float32", "no method"),
                line("GeometricSolvers", "Float64", "ran")]
            write(joinpath(dir, "correctness.csv"),
                join([CELLS_HEADER; cells .* repeat(",", 9)], "\n") * "\n")
            @test groups(ran_cells(dir)) == [("NLsolve", "Float64"),
                ("GeometricSolvers", "Float32"), ("GeometricSolvers", "Float64")]
            function parts(gs64)
                write(part_path(dir, "NLsolve", "Float64"), line("NLsolve", "Float64", "a\n"))
                write(part_path(dir, "GeometricSolvers", "Float32"),
                    line("GeometricSolvers", "Float32", "b\n"))
                write(part_path(dir, "GeometricSolvers", "Float64"), gs64)
            end
            parts(line("GeometricSolvers", "Float64", "c\n"))
            merge_parts(dir)
            @test readlines(joinpath(dir, "timings.csv")) == [TIMINGS_HEADER,
                line("NLsolve", "Float64", "a"), line("GeometricSolvers", "Float32", "b"),
                line("GeometricSolvers", "Float64", "c")]
            @test !isfile(part_path(dir, "NLsolve", "Float64"))
            # A part whose line is another cell's, that has no line for a cell, or that has a line
            # left over, stops the merge, and leaves the table and the parts as they were.
            table = read(joinpath(dir, "timings.csv"), String)
            parts(line("GeometricSolvers", "Float32", "c\n"))
            @test_throws r"is not the cell" merge_parts(dir)
            @test read(joinpath(dir, "timings.csv"), String) == table
            parts("")
            @test_throws r"no line for the cell" merge_parts(dir)
            @test read(joinpath(dir, "timings.csv"), String) == table
            parts(line("GeometricSolvers", "Float64", "c\n") *
                  line("GeometricSolvers", "Float64", "d\n"))
            @test_throws r"lines left over" merge_parts(dir)
            @test read(joinpath(dir, "timings.csv"), String) == table
            @test isfile(part_path(dir, "NLsolve", "Float64"))
        end
        bkey(b) = join([b.block; collect(b.key); b.eltype; b.variant; b.package], ",")
        mktempdir() do dir
            keys = Dict(block => bkey.(block_cells(block)) for block in BLOCKS)
            function blockparts(swap)
                for block in BLOCKS, g in block_groups(block)

                    ks = [bkey(b) for b in block_cells(block) if (b.package, b.eltype) == g]
                    swap && block == "sparse" && g == ("GeometricSolvers", "Float32") &&
                        reverse!(ks)
                    write(block_part_path(dir, block, g...), join(ks .* ",x", "\n") * "\n")
                end
            end
            blockparts(false)
            block_merge_parts(dir)
            for block in BLOCKS
                @test readlines(joinpath(dir, "$block.csv")) ==
                      [BLOCK_HEADER; keys[block] .* ",x"]
            end
            blockparts(true)
            @test_throws r"is not the cell" block_merge_parts(dir)
        end
    end

    @testset "the verifier checks the residual of a failed block cell" begin
        # A failed cell of stage.csv, `LINESEARCH_FAILED` with ‖F‖∞ = 0.0125f0, given another
        # residual in a copy of the results; the output and exit code of verify.jl on that copy.
        cell = "stage,midpoint,HarmonicOscillator h=0.1,V0,Float32,Picard Backtracking,GeometricSolvers,"
        function verify_with(fn)
            mktempdir() do dir
                for f in readdir(joinpath(@__DIR__, "results"))
                    cp(joinpath(@__DIR__, "results", f), joinpath(dir, f))
                end
                path = joinpath(dir, "stage.csv")
                ls = readlines(path)
                i = only(findall(l -> startswith(l, cell), ls))
                @test occursin(",0.0125f0,", ls[i])
                ls[i] = replace(ls[i], ",0.0125f0," => ",$fn,")
                write(path, join(ls, "\n") * "\n")
                cmd = `$(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, "verify.jl")) $dir`
                out = IOBuffer()
                p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
                return String(take!(out)), p
            end
        end
        out, p = verify_with("NaN32")
        @test success(p) && startswith(out, "VERIFY OK")
        out, p = verify_with("-0.0125f0")
        @test p.termsignal == 0 && p.exitcode == 1 &&
              occursin("FAIL stage.csv: $(chop(cell)): fnorm_inf -0.0125f0", out)
    end

    @testset "profile: the ratio to the smallest metric, a failure is Inf" begin
        t = [1.0 2.0 Inf; 4.0 2.0 8.0; Inf Inf Inf]
        r = profile_ratios(t)
        @test isequal(r, [1.0 2.0 Inf; 2.0 1.0 4.0; Inf Inf Inf])
        @test profile(r, 1) == [1, 1, 0] ./ 3
        @test profile(r, 2) == [2, 2, 0] ./ 3
        @test profile(r, 4) == [2, 2, 1] ./ 3
        @test profile(r, Inf) == [2, 2, 1] ./ 3
        @test profile_metric(false, 3.0, true) == Inf
        @test profile_metric(false, 3e-6, false) == Inf
        @test profile_metric(true, 3.0, true) == 3.0
        @test profile_metric(true, 3e-6, false) == 3e-6
    end

    @testset "profile: a zero metric counts as 1" begin
        @test profile_metric(true, 0.0, true) == 1.0
        r = profile_ratios([profile_metric(true, 0.0, true) profile_metric(true, 2.0, true)
                            ])
        @test r == [1.0 2.0]
        @test profile(r, 1) == [1.0, 0.0]
    end

    @testset "the block tables count the rows where both succeed" begin
        bcell(variant, problem, success, iterations) = (;
            block = "mixed", set = "standard",
            problem, start = "u0", eltype = "Float64", variant,
            package = "GeometricSolvers", status = "ran", method = "m-$variant",
            code = "SUCCESS", success, iterations, f_evals = iterations, median_s = "1.0")
        # gs-f64 fails on problem 1, where gs-f32 succeeds: that row is in no ratio.
        mixed = [bcell("gs-f64", "1", "false", "9"), bcell("gs-f32", "1", "true", "4"),
            bcell("gs-f64", "2", "true", "2"), bcell("gs-f32", "2", "true", "3")]
        out = split(sprint(print_mixed, mixed), "\n")
        @test out[3] == "| gs-f64 | `m-gs-f64` | 1 | 1 | 1.00 | 1.00 |"
        @test out[4] == "| gs-f32 | `m-gs-f32` | 2 | 1 | 1.50 | 1.00 |"
        # The strict cell fails where the loose one succeeds: that pair is in no ratio.
        stage = [merge(bcell("Newton Static LU", p, "true", it), (; block = "stage"))
                 for (p, it) in (("1", "5"), ("2", "6"))]
        cells = [(; set = "standard", problem = p, start = "u0", eltype = "Float64",
                     column = "Newton Static LU", package = "GeometricSolvers",
                     status = "ran", success = s, iterations = "3", f_evals = "3")
                 for (p, s) in (("1", "false"), ("2", "true"))]
        timings = Dict(cellkey(c) => (; median_s = "2.0") for c in cells)
        out = split(sprint(print_stage, stage, cells, timings), "\n")
        @test out[3] == "| GeometricSolvers | 2 | 1 | 1 | 2.00 | 2.00 | 0.500 |"
    end

    @testset "formats" begin
        @test dec3(1 / 3) == "0.333" && dec3(1.0) == "1.000"
        @test sig3(1.0) == "1.00" && sig3(0.5) == "0.500" && sig3(123.4) == "123"
        @test sig3(1234.0) == "1.23e+03" && sig3(1.234e-5) == "1.23e-05" && sig3(NaN) == "—"
    end
end

include(joinpath(@__DIR__, "devices.jl"))

# Exceptions with the names of the vendor packages' own, for the classifier, which reads the name
# of the type and the message.
module VendorErrors
struct KernelException <: Exception end
struct InvalidIRError <: Exception end
Base.showerror(io::IO, ::KernelException) = print(io, "KernelException: on device")
Base.showerror(io::IO, ::InvalidIRError) = print(io, "InvalidIRError: unsupported call")
end

# Modules with the names of the vendor packages, and the calls of theirs that `card_name` and
# `driver` make; `Broken.CUDA` throws on its driver call.
module FakeVendors
struct MetalDevice
    name::String
end
module CUDA
module CUDACore
name(d) = d == 0 ? "NVIDIA GB10" : "another card"
end
const CuArray = Array
device() = 0
driver_version() = v"13.0.0"
end
module Metal
import ..FakeVendors: MetalDevice
device() = MetalDevice("Apple M4 Max")
macos_version() = v"26.1.0"
end
module AMDGPU
module HIP
name(d) = d == 0 ? "AMD Radeon RX 7900 XTX" : "another card"
runtime_version() = v"7.1.0"
end
device() = 0
end
module Broken
module CUDA
driver_version() = error("no driver")
end
end
end

# A factor that records which function of GeometricSolvers' LU each path calls.
struct RecordedMatrix end
struct RecordedLU
    factors::RecordedMatrix
    ipiv::Vector{Int32}
    info::Int
    scale_exponent::Int
end
const LU_CALLS = Symbol[]
function GeometricSolvers.ka_factorize_lu!(f::RecordedLU, J)
    (push!(LU_CALLS, :ka_factorize_lu!); f)
end
function GeometricSolvers.ka_lu_backsolve!(::RecordedMatrix, ipiv, d)
    (push!(
            LU_CALLS, :ka_lu_backsolve!); d)
end
function GeometricSolvers.factorize!!(f::RecordedLU, ::LUFactorization, J)
    (push!(
            LU_CALLS, :factorize!!); f)
end
GeometricSolvers.ldiv!!(d, ::RecordedLU, b) = (push!(LU_CALLS, :ldiv!!); copyto!(d, b))

@testset "Y3" begin
    T = Float32
    tol = sqrt(eps(T))
    cpu = Device("cpu", "cpu", nothing, identity, JLArray, "selftest card")
    jl = Device("cuda", "spark", nothing, identity, JLArray, "selftest card")

    @testset "the machine and the backend are a pair" begin
        for (b, m) in (("cpu", "cpu"), ("metal", "mac"), ("cuda", "spark"),
            ("cuda", "rtx4090"), ("rocm", "rx7900xtx"))
            @test check_pair(b, m) === nothing
        end
        for (b, m) in (("metal", "spark"), ("cuda", "mac"), ("rocm", "cpu"), ("cpu", "mac"))
            e = try
                check_pair(b, m)
            catch e
                e
            end
            @test e isa ArgumentError && occursin(b, e.msg) && occursin(m, e.msg)
        end
        # The refusal comes before a vendor package loads or a file is written.
        dir = mktempdir()
        e = try
            run_device("metal", "spark", dir)
        catch e
            e
        end
        @test e isa ArgumentError && occursin("metal", e.msg) && occursin("spark", e.msg)
        @test isempty(readdir(dir)) && !isdefined(Main, :Metal)
    end

    @testset "the inline adapter runs a standard row in a CPU() kernel" begin
        rows = device_rows(T)
        @test length(rows) == 79 && count(r -> r.set == "standard", rows) == 67
        @test all(r -> isbits(r.p), rows)
        row = rows[findfirst(r -> r.problem == "2" && r.start == "u0", rows)]
        X = fill(row.x0, 4)
        S = KernelAbstractions.allocate(CPU(), SolverStatus{T}, 4)
        solve_batch!(X, S, NonlinearProblem(row.F), Newton(), row.p; f_abstol = tol,
            f_reltol = 0, max_iterations = 200)
        @test all(s -> s.code == SUCCESS, S)
        @test norm(row.F(X[1], row.p), Inf) ≤ tol
    end

    @testset "the Lotka–Volterra midpoint rows run in a CPU() kernel" begin
        # The :iode residual keeps the iterate and GeometricProblems' outputs off the heap.
        rows = filter(r -> startswith(r.problem, "LotkaVolterra"), device_rows(T))
        @test length(rows) == 4
        for row in rows
            X = fill(row.x0, 4)
            S = KernelAbstractions.allocate(CPU(), SolverStatus{T}, 4)
            solve_batch!(X, S, NonlinearProblem(row.F), Newton(), row.p; f_abstol = tol,
                f_reltol = 0, max_iterations = 200)
            @test all(s -> s.code == SUCCESS, S)
            @test norm(row.F(X[1], row.p), Inf) ≤ tol
        end
    end

    @testset "the verdict: Y-2's bound scales with the host root" begin
        @test device_verdict(true, true, 5tol, 10, T) == "agree"
        @test device_verdict(true, true, 11tol, 10, T) == "different root"
        @test device_verdict(true, true, 0.5tol, 0.1, T) == "agree"
        @test device_verdict(true, true, 2tol, 0.1, T) == "different root"
        @test device_verdict(false, true, 0, 1, T) == "device fails"
        @test device_verdict(true, false, 0, 1, T) == "host fails"
        @test device_verdict(false, false, 0, 1, T) == "both fail"
        @test device_verdict(true, true, 1e-7, 1, Float64) == "different root"
        @test device_verdict(true, true, 1e-8, 1, Float64) == "agree"
        @test success_of("SUCCESS", "GeometricSolvers", tol, T)
        @test !success_of("SUCCESS", "GeometricSolvers", 2tol, T)
        @test !success_of("SUCCESS", "GeometricSolvers", T(NaN), T)
        @test !success_of("MAXITERS", "GeometricSolvers", 0, T)
        @test success_of("Success", "SimpleNonlinearSolve", 0, T)
        @test success_of("info=0", "GeometricSolvers", 0, T) &&
              !success_of("info=3", "GeometricSolvers", 0, T)
        # The bound itself agrees.
        @test device_verdict(true, true, 10tol, 10, T) == "agree"
        # A reference that failed gives no distance, and its `‖x‖∞` and `‖F‖∞` go to the comment.
        ref = (; x = T[NaN, 1], code = "MAXITERS",
            iterations = "200", fnorm = T(NaN), ok = false)
        c = compared(true, T[1, 1], ref, T)
        @test c.dx_host == "-" && c.verdict == "host fails" && c.host_code == "MAXITERS" &&
              c.host_iterations == "200" && c.ref == "ref_xnorm=NaN32;ref_fnorm=NaN32"
        ref = (; x = T[10, 1], code = "SUCCESS", iterations = "3", fnorm = T(0), ok = true)
        c = compared(true, T[10, 1 + 5tol], ref, T)
        @test isapprox(c.dx_host, 5tol; rtol = 1e-3) && c.verdict == "agree"
    end

    @testset "the band: a row above it is not launched" begin
        @test band_status("rocm", 10) == "above band 8" &&
              band_status("rocm", 8) === nothing
        @test band_status("metal", 25) == "above band 24" &&
              band_status("cuda", 24) === nothing
        @test band_status("cpu", 32) === nothing
        launched = Ref(false)
        F(x, p) = (launched[] = true; x)
        row = (;
            set = "selftest", problem = "n10", start = "x0", x0 = zero(SVector{10, T}),
            F, p = nothing)
        rocm = Device("rocm", "rx7900xtx", nothing, identity, identity, "selftest card")
        lines, failed = r2_row_lines(rocm, row)
        @test length(lines) == 16 && !failed && !launched[]
        f = [split(l, ","; limit = 33) for l in lines]
        @test all(c -> c[1:4] == ["r2", "rx7900xtx", "rocm", "selftest card"], f)
        @test [c[14] for c in f] ==
              vcat(["above band 8", "no device path", "above band 8", "no device path"],
            repeat(["above band 8", "no device path", NO_METHOD_SNS, "no device path"], 3))
        @test all(c -> all(==("-"), c[15:33]), f)
        # The grid: the cpu mode, the band of every variant but host-r1, and the system counts.
        @test grid_status("cpu", 6, "gs-newton") == "not run: cpu mode"
        @test grid_status("cpu", 6, "host-r1") == "not run: cpu mode"
        @test grid_status("cpu", 8, "sns") === nothing
        @test grid_status("rocm", 12, "sns") == "above band 8"
        @test grid_status("rocm", 12, "host-r1") === nothing
        @test grid_status("rocm", 8, "gs-newton") === nothing
        @test grid_status("metal", 16, "gs-reuse4") === nothing
        @test grid_nsys("cpu") == (10^3, 10^4) && grid_nsys("cuda") == GRID_NSYS
    end

    @testset "the classifier of a launch that throws" begin
        @test startswith(launch_status(ErrorException("Failed to compile program")),
            "no compile: ")
        @test startswith(launch_status(VendorErrors.InvalidIRError()), "no compile: ")
        @test launch_status(VendorErrors.KernelException()) ==
              "kernel error: KernelException: on device"
        @test startswith(launch_status(ErrorException("GPU Kernel Exception:")),
            "kernel error: ")
        @test startswith(
            launch_status(ErrorException("exception thrown during kernel execution")),
            "kernel error: ")
        @test launch_status(DomainError(1.0)) === nothing
        @test launch_status(ErrorException("a, \"b\"\nc")) === nothing
        @test !occursin(',', launch_status(ErrorException("Failed to compile program, a")))
        t = Threads.@spawn throw(ErrorException("GPU Kernel Exception: x"))
        e = try
            wait(t)
        catch e
            e
        end
        @test startswith(launch_status(e), "kernel error: ErrorException: GPU Kernel")
    end

    @testset "the Metal heap: where a run continues" begin
        L = 2^20
        # A failure on a heap that earlier rows used runs the row again on an empty heap.
        @test heap_next(5, 100, L + 8, L, 4, true) == 5
        @test heap_next(5, 100, 200, L, 4, true) == 5
        # A failure on the empty heap keeps its outcome; a full heap continues at the next row.
        @test heap_next(5, 4, L + 8, L, 4, true) == 6
        @test heap_next(5, 4, 200, L, 4, true) === nothing
        @test heap_next(5, 100, L + 1, L, 4, false) == 6
        @test heap_next(5, 100, L, L, 4, false) === nothing
        @test heap_next(5, nothing, nothing, nothing, nothing, true) === nothing
    end

    @testset "a timed run: fresh starts, then synchronize, then the clock" begin
        log = Symbol[]
        clock = FakeClock(0)
        tick() = (push!(log, :clock); clock())
        fresh() = (push!(log, :fresh); 1)
        launch!(_) = (push!(log, :launch); elapse!(clock, 0.6e-3); 2)
        sync() = (push!(log, :sync); elapse!(clock, 0.1e-3))
        m = measure_cell(fresh, launch!, sync; clock = tick)
        @test m.result == 2
        # The warm-up, then runs of 1 launch (0.7 ms) and of 2 launches (1.3 ms, 0.65 ms each).
        @test log[1:5] == [:fresh, :clock, :launch, :sync, :clock]
        @test log[6:10] == [:fresh, :clock, :launch, :sync, :clock]
        @test log[16:22] == [:fresh, :fresh, :clock, :launch, :launch, :sync, :clock]
        @test m.k == 2 && length(m.times) == 5 && all(≈(0.65e-3), m.times)
        @test m.warm ≈ 0.7e-3
        @test count(==(:sync), log) == 1 + 2 + 2 + 5
        # The fields of a measured cell: compile_s is the warm-up's time minus the median; the
        # times have the median 3 and the mean 5.
        t = time_fields((; warm = 5.0, k = 4, times = [3.0, 1.0, 2.0, 15.0, 4.0]))
        @test (t.t1, t.t5, t.median_s, t.spread, t.solves_per_run, t.compile_s) ==
              (3.0, 4.0, 3.0, 14 / 3, 4, 2.0)
    end

    @testset "the time cap: a projection above 1800 s stops the series" begin
        clock = FakeClock(0)
        measured = Int[]
        measure(n_sys) = begin
            push!(measured, n_sys)
            t = n_sys == 10^5 ? 400.0 : 1e-3
            measure_cell(() -> nothing, _ -> elapse!(clock, t), () -> nothing; clock)
        end
        out = grid_series(measure, GRID_NSYS)
        @test measured == [10^3, 10^4, 10^5]
        @test out[1].k == 1 && run_median(out[3].times) == 400
        @test out[4] == "not run: time cap" && out[5] == "not run: time cap"
        @test projected(29.0) < 1800 < projected(31.0)
        # A series below the cap runs every cell.
        clock2 = FakeClock(0)
        out = grid_series(GRID_NSYS) do n_sys
            measure_cell(() -> nothing, _ -> elapse!(clock2, 1e-3), () -> nothing;
                clock = clock2)
        end
        @test all(o -> o isa NamedTuple, out)
        # A projection of exactly 1800 s runs the cell.
        clock3 = FakeClock(0)
        out = grid_series(GRID_NSYS) do n_sys
            measure_cell(() -> nothing, _ -> elapse!(clock3, n_sys == 10^4 ? 30.0 : 1e-3),
                () -> nothing; clock = clock3)
        end
        @test out[3] isa NamedTuple
        # A cell after one whose launch threw has no projection and runs.
        clock4 = FakeClock(0)
        out = grid_series(GRID_NSYS) do n_sys
            n_sys == 10^4 && return "kernel error: selftest"
            measure_cell(() -> nothing, _ -> elapse!(clock4, 1e-3), () -> nothing;
                clock = clock4)
        end
        @test out[2] == "kernel error: selftest" &&
              all(o -> o isa NamedTuple, out[[1, 3, 4, 5]])
        # Each variant on 100 systems of dimension 2 on CPU(): every system succeeds, and the
        # factor reuse of gs-reuse4 takes one more step than Newton.
        for (v, it) in zip(GRID_VARIANTS, ("2", "3", "2", "-", "2"))
            c = grid_cell(cpu, 2, 100, v)
            @test c.ok == 100 && c.fnorm ≤ tol && string(c.iterations) == it
            @test c.code == (v == "sns" ? "Success=100" : "SUCCESS=100")
            @test c.k ≥ 1 && length(c.times) == 5
        end
    end

    @testset "the .txt of a run" begin
        BLAS.set_num_threads(1)
        ls = txt_head(cpu)
        names = ["date", "uptime", "julia", "threads", "blas threads", "device", "driver",
            "band", "packages"]
        @test length(ls) == 9 && all(((l, n),) -> startswith(l, "$n: "), zip(ls, names))
        @test all(l -> length(l) > length("uptime: "), ls[1:2])
        @test ls[5] == "blas threads: 1" && ls[6] == "device: selftest card"
        @test ls[7] == "driver: none" && ls[8] == "band: none"
        @test occursin("KernelAbstractions 0.10.7", ls[9]) &&
              occursin("JLArrays 0.5.0", ls[9])
        # The card and the driver as each vendor package names them, and a vendor call that
        # throws stops the run.
        for (vendor, card, drv) in (
            (FakeVendors.CUDA, "NVIDIA GB10", "CUDA driver 13.0.0"),
            (FakeVendors.Metal, "Apple M4 Max", "macOS 26.1.0"),
            (FakeVendors.AMDGPU, "AMD Radeon RX 7900 XTX", "HIP runtime 7.1.0"))
            @test card_name(vendor) == card
            @test driver(Device("x", "y", vendor, identity, identity, card)) == drv
        end
        @test device_of("cuda", "spark", FakeVendors.CUDA).card == "NVIDIA GB10"
        @test_throws ErrorException driver(Device(
            "x", "y", FakeVendors.Broken.CUDA, identity, identity, "card"))
    end

    @testset "the LU paths: own-ka calls the internal functions, dispatched those of solve" begin
        fac = RecordedLU(RecordedMatrix(), Int32[], 0, 0)
        empty!(LU_CALLS)
        lu_once!("own-ka", fac, nothing, T[1, 2])
        @test LU_CALLS == [:ka_factorize_lu!, :ka_lu_backsolve!]
        empty!(LU_CALLS)
        lu_once!("dispatched", fac, nothing, T[1, 2])
        @test LU_CALLS == [:factorize!!, :ldiv!!]
        # own-ka scales the backsolve by 2^(-e).
        _, d = lu_once!("own-ka", RecordedLU(RecordedMatrix(), Int32[], 0, 1), nothing, T[1, 2])
        @test d == T[0.5, 1]
        # Both paths solve J d = b on a JLArray to Y-2's bound of the host LAPACK solution; the
        # device is named `cuda` so that n = 32 is no cpu-mode cell.
        @test split(lu_line(cpu, 4096, "own-ka"), ",")[14] == "not run: cpu mode"
        for v in LU_VARIANTS
            line = lu_line(jl, 32, v)
            c = split(line, ","; limit = 33)
            @test c[14] == "ran" && c[15] == "info=0" && c[16] == "true" && c[22] == "agree"
            # J, b and the solution are Float32: `repr` writes the residual with an `f`.
            @test occursin(r"f-?\d+$", c[20]) && occursin(r"f-?\d+$", c[21])
            @test (v == "own-ka") == startswith(c[33], "\"internal, not user-selectable")
            # The matrix is J = A + I/10: the host solution's ‖d‖∞ is that of (A + I/10) \ b.
            A, b = tenE_r3(32, T)
            xnorm = match(r"ref_xnorm=([^;]+);", c[33]).captures[1]
            @test parse(Float64, replace(xnorm, "f" => "e")) ≈ norm((A + I / 10) \ b, Inf) rtol = 1e-5
        end
    end

    @testset "the R3 rows on a JLArray" begin
        # The Jacobian is ForwardDiff's of the residual, A + diag(cos.(x))/10.
        A, b = tenE_r3(8, Float64)
        p = (; A, b, Id = Matrix{Float64}(I, 8, 8))
        x = collect(range(-1.0, 1.0; length = 8))
        J = zeros(8, 8)
        r3_J!(J, x, p)
        @test J ≈ ForwardDiff.jacobian((y, z) -> r3_F!(y, z, p), zeros(8), x)
        y = zeros(8)
        r3_F!(y, x, p)
        @test y ≈ A * x + sin.(x) / 10 - b
        @test all(l -> split(l, ",")[14] == "not run: cpu mode", r3_lines(cpu, 4096))
        metal = Device("metal", "mac", nothing, identity, JLArray, "selftest card")
        @test [split(l, ",")[14] for l in r3_lines(metal, 8)] ==
              ["ran", "ran", METAL_F64, METAL_F64, METAL_F64, "ran"]
        lines = r3_lines(jl, 16)
        @test length(lines) == 6
        f = [split(l, ","; limit = 33) for l in lines]
        @test [c[12] for c in f] == collect(R3_VARIANTS)
        @test all(c -> c[14] == "ran" && c[15] == "SUCCESS" && c[16] == "true", f)
        @test [c[22] for c in f] == ["agree", "-", "agree", "agree", "agree", "-"]
        # device compares with host, the Float64 device variants with host-f64.
        @test f[1][18:19] == f[2][[15, 17]] && all(c -> c[18:19] == f[6][[15, 17]], f[3:5])
        @test startswith(f[2][33], "xnorm=") && startswith(f[6][33], "xnorm=")
        @test [c[10] for c in f] == ["Float32", "Float32", "Float64", "Float64", "Float64",
            "Float64"]
        @test [c[11] for c in f] == ["Float32", "Float32", "Float32", "Float32", "Float64",
            "Float64"]
        d64 = f[5]
        @test all(==("-"), d64[24:32]) &&
              d64[33] == "device Float64: not timed (decision 6)"
        @test all(c -> all(s -> tryparse(Float64, s) !== nothing, c[24:30]), f[[1:4; 6]])
    end

    @testset "an R2 row: sixteen lines in order, and the host solve fills its fields" begin
        rows = device_rows(T)
        row = rows[findfirst(r -> r.problem == "2" && r.start == "u0", rows)]
        lines, failed = r2_row_lines(cpu, row)
        f = [split(l, ","; limit = 33) for l in lines]
        @test !failed && length(f) == 16 && all(c -> length(c) == 33, f)
        @test [c[12] for c in f] == repeat(collect(R2_COLUMNS); inner = 4)
        @test [c[13] for c in f] == repeat(collect(PACKAGES), 4)
        gs = f[1]
        @test gs[14] == "ran" && gs[15] == "SUCCESS" && gs[16] == "true" &&
              gs[22] == "agree"
        @test gs[17] == gs[19] && gs[18] == "SUCCESS" && gs[23] == "64"
        @test f[3][14] == "ran" && f[3][17] == "-"
        @test all(c -> !any(isempty, c), f)
        # A row that throws in the kernel keeps its host fields.
        bad = rows[findfirst(r -> r.problem == "1" && r.start == "u0", rows)]
        lines, failed = r2_row_lines(cpu, bad)
        c = split(lines[1], ","; limit = 33)
        @test failed && startswith(c[14], "kernel error: ") && c[18] != "-"
        @test c[15] == "-" && c[22] == "-"
        # The copies within Y-2's bound of system 1: a copy one bound away counts, one twice the
        # bound away does not, and a system 1 that holds a NaN has no copy within its bound.
        x = SVector{2, T}(10, 1)
        X = fill(x, 64)
        X[2] = x + SVector{2, T}(5tol, 0)
        X[3] = x + SVector{2, T}(30tol, 0)
        @test copies_same(X) == 63
        # Below norm 1 the bound is √eps itself.
        y = SVector{2, T}(0.1, 0.05)
        Y = fill(y, 64)
        Y[2] = y + SVector{2, T}(0.5tol, 0)
        Y[3] = y + SVector{2, T}(2tol, 0)
        @test copies_same(Y) == 63
        X[1] = SVector{2, T}(NaN, 1)
        @test copies_same(X) == 0
        @test copies_same(fill(SVector{2, T}(NaN, 1), 64)) == 0
        # A system succeeds with a code of success and ‖F‖∞ within √eps.
        @test successes(["SUCCESS", "SUCCESS", "MAXITERS", "SUCCESS"], "GeometricSolvers",
            T[0, 2tol, 0, tol]) == 2
    end

    @testset "the verifier checks the device lines" begin
        # verify.jl on a copy of the results whose devices-cpu.csv or .txt `edit!` changes, its
        # output and whether it passed.
        function verify_devices(edit!, args...; file = "devices-cpu.csv")
            mktempdir() do dir
                for f in readdir(joinpath(@__DIR__, "results"))
                    cp(joinpath(@__DIR__, "results", f), joinpath(dir, f))
                end
                path = joinpath(dir, file)
                ls = readlines(path)
                edit!(ls, dir)
                isfile(path) && write(path, join(ls, "\n") * "\n")
                cmd = `$(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, "verify.jl")) $dir $args`
                out = IOBuffer()
                p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
                return String(take!(out)), success(p)
            end
        end
        # The first line of a cell by its kind, problem, start, variant and package: its smallest
        # n for the R3 and LU rows.
        function at(ls, kind, problem, start, variant, package)
            return first(findall(ls) do l
                f = split(l, ",")
                f[1] == kind && f[6] == problem && f[7] == start && f[12] == variant &&
                    f[13] == package
            end)
        end
        setcell!(ls, i, k, v) = (
            f = split(ls[i], ","; limit = 33); f[k] = v; ls[i] = join(f, ","))
        gs = ("GeometricSolvers",)
        out, ok = verify_devices((ls, _) -> nothing, "--structure")
        @test ok && occursin("device lines: cpu 1438 (1264 r2, 150 grid, 18 r3, 6 lu)", out)
        # A root of norm 10 at 5 √eps from the device's iterate agrees.
        out, ok = verify_devices("--structure") do ls, _
            i = at(ls, "r2", "2", "u0", "Newton Static LU", gs...)
            setcell!(ls, i, 21, "0.0017263349f0")
            setcell!(ls, i, 33, "ref_xnorm=10.0f0;ref_fnorm=0.0f0")
        end
        @test ok
        bad = [
            (ls -> deleteat!(ls, 2), "lines, not 1438"),
            (
                ls -> setcell!(ls, at(ls, "r3", "10.E", "zero", "device-f64", gs...), 24,
                    "0.5"),
                "a cell that is not timed has a time field"),
            (
                ls -> setcell!(ls, at(ls, "r2", "2", "u0", "Newton Static LU", gs...), 22,
                    "different root"),
                "verdict different root, recomputed agree"),
            (
                ls -> setcell!(ls, at(ls, "r2", "2", "u0", "Newton Static LU", gs...), 16,
                    "false"),
                "success false, recomputed true"),
            (ls -> setcell!(ls, 2, 4, "another card"), "card another card"),
            (
                ls -> setcell!(
                    ls, at(ls, "r2", "2", "u0", "Newton Static LU",
                        "SimpleSolvers"), 14, "ran"),
                "the cell is 'no device path'"),
            (
                ls -> setcell!(
                    ls, findfirst(l -> startswith(l, "grid,") && occursin(",ran,", l), ls),
                    29, "1.0"),
                "median_s 1.0"),
            (ls -> setcell!(ls, 2, 17, ""), "an empty field"),
            (ls -> setcell!(ls, at(ls, "lu", "10.E", "-", "own-ka", gs...), 33, "-"),
                "comment -"),
            (
                ls -> setcell!(ls, at(ls, "r2", "2", "u0", "Newton Static LU", gs...), 33,
                    "ref_xnorm=1.0f0;ref_fnorm=1.0f0"),
                "verdict agree, recomputed host fails")]
        for (edit!, msg) in bad
            out, ok = verify_devices((ls, _) -> edit!(ls), "--structure")
            @test (msg, !ok && occursin(msg, out)) == (msg, true)
        end
        out, ok = verify_devices((ls, _) -> (ls[5] = "blas threads: 2");
            file = "devices-cpu.txt")
        @test !ok && occursin("devices-cpu.txt: blas threads 2", out)
        out, ok = verify_devices((ls, dir) -> rm(joinpath(dir, "devices-cpu.csv")))
        @test !ok && occursin("devices-cpu.csv is missing", out)
        out, ok = verify_devices((ls, _) -> nothing, "--all-cards")
        @test !ok && occursin("devices-mac.csv is missing (--all-cards)", out)
        out, ok = verify_devices((ls, _) -> (ls[3] = join(split(ls[3], ",")[1:30], ",")),
            "--structure")
        @test !ok && occursin("a line without 33 fields", out)
        out, ok = verify_devices((ls, _) -> pop!(ls); file = "devices-cpu.txt")
        @test !ok && occursin("devices-cpu.txt: 87 lines, not 88", out)
        # The band limits the device variants of the grid, and host-r1 has none: on rocm a
        # host-r1 cell at n = 12 is launched, an sns cell is above the band.
        out, ok = verify_devices("--structure") do ls, dir
            # The R2 rows above the band as on rocm, so that the grid's lines are among the
            # first 50 failures that verify.jl prints.
            rocm = map(ls[2:end]) do l
                f = split(l, ","; limit = 33)
                f[2], f[3] = "rx7900xtx", "rocm"
                if f[1] == "r2" && parse(Int, f[8]) > 8 &&
                   (f[14] == "ran" || startswith(f[14], "kernel error") ||
                    startswith(f[14], "no compile"))
                    f[14] = "above band 8"
                    f[15:33] .= "-"
                end
                join(f, ",")
            end
            for v in ("host-r1", "sns")
                i = findfirst(rocm) do l
                    f = split(l, ",")
                    f[1] == "grid" && f[8] == "12" && f[9] == "1000" && f[12] == v
                end
                setcell!(rocm, i, 14, "above band 8")
            end
            write(joinpath(dir, "devices-rx7900xtx.csv"), join([ls[1]; rocm], "\n") * "\n")
        end
        @test !ok
        @test occursin(
            "(grid - 10.E zero host-r1 GeometricSolvers): status 'above band 8', the cell is 'launched'",
            out)
        @test !occursin(
            "(grid - 10.E zero sns SimpleNonlinearSolve): status 'above band 8'",
            out)

        # One copy with a corruption of each field on a line of its own, and of the .txt, and a
        # file of no machine: each check reports its own line.
        out, ok = verify_devices("--structure") do ls, dir
            r2(problem, start, column, package) = at(
                ls, "r2", problem, start, column, package)
            grid(n, ns, v) = findfirst(ls) do l
                f = split(l, ",")
                f[1] == "grid" && f[8] == string(n) && f[9] == string(ns) && f[12] == v
            end
            r3(kind, n, v) = findfirst(ls) do l
                f = split(l, ",")
                f[1] == kind && f[8] == string(n) && f[12] == v
            end
            setcell!(ls, r2("2", "u0", "Newton Static LU", gs...), 23, "65")
            setcell!(ls, r2("2", "u0", "Newton Static LU", "SimpleNonlinearSolve"), 17, "5")
            setcell!(ls, r2("2", "10u0", "Newton Static LU", gs...), 2, "mac")
            setcell!(ls, r2("2", "100u0", "Newton Static LU", gs...), 33, "-")
            setcell!(ls, r2("2", "u0", "Newton Backtracking LU", gs...), 21, "-")
            setcell!(ls, r2("3", "u0", "Picard Backtracking", "NLsolve"), 12, "Broyden")
            setcell!(ls, grid(2, 1000, "gs-newton"), 31, "3")
            setcell!(ls, grid(2, 1000, "gs-reuse4"), 32, "x")
            setcell!(ls, grid(2, 1000, "gs-backtracking"), 15, "SUCCESS=999")
            setcell!(ls, grid(2, 1000, "sns"), 16, "999")
            setcell!(ls, grid(2, 1000, "host-r1"), 18, "SUCCESS")
            i = grid(4, 10000, "gs-newton")
            setcell!(ls, i, 14, "not run: time cap")
            foreach(k -> setcell!(ls, i, k, "-"), 15:32)
            setcell!(ls, grid(6, 1000, "gs-newton"), 15, "SUCCESS=1")
            setcell!(ls, grid(8, 1000, "gs-newton"), 29, "40.0")
            setcell!(ls, r3("r3", 256, "host"), 33, "-")
            setcell!(ls, r3("r3", 256, "device"), 18, "MAXITERS")
            setcell!(ls, r3("r3", 256, "device-f64"), 33, "-")
            setcell!(ls, r3("r3", 1024, "device-mixed"), 14, "weird")
            ls[1] = replace(ls[1], "comment" => "note")
            txt = joinpath(dir, "devices-cpu.txt")
            t = readlines(txt)
            t[7], t[8] = "driver: x", "band: 8"
            t[3] = replace(t[3], "julia: " => "julia ")
            t[9] = replace(t[9], ", JLArrays 0.5.0" => "")
            t[10] = "heap: standard,1,u0,1,2"
            write(txt, join(t, "\n") * "\n")
            cp(joinpath(dir, "devices-cpu.csv"), joinpath(dir, "devices-foo.csv"))
        end
        @test !ok
        for msg in ["copies_same 65", "SimpleNonlinearSolve): iterations 5",
            "machine mac and backend cpu", "no ref_xnorm and ref_fnorm in the comment",
            "dx_host - with a reference that succeeded",
            "the keys are not those of the device rows in their order",
            "solves_per_run 3", "compile_s x", "code SUCCESS=999", "success 999 of 1000",
            "a grid cell has a reference",
            "grid n = 4 gs-newton at 10000 is not run: time cap",
            "grid n = 8 gs-newton at 10000 is ran, projected 2400.0 s", "header is",
            "a cell that did not run has measured fields",
            "a reference line has a reference or no ‖x‖∞", "host fields MAXITERS",
            "device-f64 GeometricSolvers): comment -", "status 'weird'",
            "devices-cpu.txt: driver x on cpu", "devices-cpu.txt: band 8",
            "devices-cpu.txt: the line 'julia 1", "devices-cpu.txt: packages lack JLArrays",
            "the heap line 'heap: standard,1,u0,1,2'", "devices-foo.csv: no machine foo"]
            @test (msg, occursin(msg, out)) == (msg, true)
        end

        # The report's Y3 tables, in a copy of the verifier, the report and the traces.
        mktempdir() do dir
            for f in ("verify.jl", "REPORT.md")
                cp(joinpath(@__DIR__, f), joinpath(dir, f))
            end
            cp(joinpath(@__DIR__, "traces"), joinpath(dir, "traces"))
            cp(joinpath(@__DIR__, "results"), joinpath(dir, "results"))
            path = joinpath(dir, "REPORT.md")
            r = read(path, String)
            for (a, b) in ("| cpu | cpu | host apple-m4 | none | 1.13" => "| cpu | cpu | x | none | 1.13",
                "| GeometricSolvers | 59 | 53 | 53 |" => "| GeometricSolvers | 59 | 52 | 53 |",
                "| cpu | 2 | gs-newton | " => "| cpu | 2 | gs-newton | 1 ",
                "| cpu | 256 | host | ran | 2 | true | " => "| cpu | 256 | host | ran | 2 | true | 1",
                "| cpu | 256 | 0.00252 |" => "| cpu | 256 | 0.00253 |",
                "on device cpu | 81 |" => "on device cpu | 80 |",
                "| device fails | The undamped" => "| device fails | |")
                @test occursin(a, r)
                r = replace(r, a => b)
            end
            write(path, r)
            cmd = `$(Base.julia_cmd()) --startup-file=no $(joinpath(dir, "verify.jl")) $(joinpath(dir, "results"))`
            out = IOBuffer()
            p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
            out = String(take!(out))
            @test !success(p)
            for msg in ["the Y3 cards has", "the Y3 R2 correctness has",
                "the Y3 R2 throughput has", "the Y3 R3 rows has", "the Y3 LU rows has",
                "the Y3 not-run table has", "a Y3 explanation without its text"]
                @test (msg, occursin(msg, out)) == (msg, true)
            end
        end
    end
end
