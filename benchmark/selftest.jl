# Self-test of the benchmark harness: the pairing, the label and the method of each call, the
# options of each package, the counting wrapper, the shared Jacobian, the measured `‖F‖∞`, the
# success rule, the agreement bound, the verdict, a throwing residual, the SVD solve on a
# non-finite matrix, and the element type and the start of the midpoint rows; and for the
# timings and the blocks: the timed runs and the batch, the note of a timed cell, the tolerance
# of each adapter, the variant labels, the `nsol` adapter, the sparse variants and the profiles.
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
