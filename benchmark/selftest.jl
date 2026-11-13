# Self-test of the benchmark harness: the pairing, the label and the method of each call, the
# options of each package, the counting wrapper, the shared Jacobian, the measured `‖F‖∞`, the
# success rule, the agreement bound, the verdict, a throwing residual, the SVD solve on a
# non-finite matrix, and the element type and the start of the midpoint rows.
#
# Usage: julia --project=benchmark benchmark/selftest.jl

using Test

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "packages.jl"))

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
