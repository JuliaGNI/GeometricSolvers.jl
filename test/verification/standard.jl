# The standard test set: the 23 problems of NonlinearProblemLibrary.jl, from three starts, with
# every column of `STANDARD_COLUMNS`, in each element type of `REAL_ELTYPES`, compared with the
# reference table `test/verification/standard_reference.csv`.
#
# A cell that changes is regenerated with `scripts/standard/update_reference.jl`, and the pull
# request that changes it lists the cell with its cause.

using GeometricSolvers
using GeometricSolvers: AutoForwardDiff, ReturnCode, chunk_size
using ForwardDiff: Dual
using LinearAlgebra: norm
using Test

include("../helpers/standard_cells.jl")

# Each floating-point literal of the expression `ex` that stands outside a call `T(...)`.
function float_literals_outside_T(ex, inside = false, found = Any[])
    if ex isa AbstractFloat
        inside || push!(found, ex)
    elseif ex isa Expr
        inside = inside || (ex.head === :call && first(ex.args) === :T)
        foreach(a -> float_literals_outside_T(a, inside, found), ex.args)
    end
    return found
end

@testset "The 23 problems" begin
    @test length(STANDARD_PROBLEMS) == 23
    @test [sp.number for sp in STANDARD_PROBLEMS] == 1:23
    @test [sp.n for sp in STANDARD_PROBLEMS] ==
          [10, 4, 2, 4, 3, 2, 2, 10, 10, 10, 10, 10, 10, 10, 4, 9, 2, 2, 2, 1, 2, 2, 10]
    @test [sp.number for sp in STANDARD_PROBLEMS if sp.root !== nothing] ==
          [1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21, 22]
    @testset "$T" for T in REAL_ELTYPES
        @test sum(sp -> length(standard_starts(sp, T)), STANDARD_PROBLEMS) == 67
        @test [label for (label, _) in standard_starts(STANDARD_PROBLEMS[6], T)] == ["u0"]
        @test [label for (label, _) in standard_starts(STANDARD_PROBLEMS[1], T)] ==
              ["u0", "10u0", "100u0"]
        for sp in STANDARD_PROBLEMS, (_, x) in standard_starts(sp, T)

            @test x isa Vector{T}
            @test length(x) == sp.n
        end
        @test standard_starts(STANDARD_PROBLEMS[1], T)[3].second ==
              100 .* standard_starts(STANDARD_PROBLEMS[1], T)[1].second
    end
end

@testset "Float32 arithmetic" begin
    # The checker finds a bare literal and accepts one inside `T(...)`.
    @test float_literals_outside_T(Meta.parse("f(x) = 1.0 - x")) == [1.0]
    @test isempty(float_literals_outside_T(Meta.parse("f(x) = T(1.0) - x")))
    @test isempty(float_literals_outside_T(Meta.parse("f(x) = sqrt(T(5)) * T(0.5 * 2.0)")))

    source = read(joinpath(@__DIR__, "..", "helpers", "standard_problems.jl"), String)
    @test isempty(float_literals_outside_T(Meta.parseall(source)))

    for sp in STANDARD_PROBLEMS
        x = sp.u0(Float32)
        y = zeros(Float32, sp.n)
        @test sp.prob.F(y, x, nothing) === nothing
        @test y isa Vector{Float32}
        @test all(isfinite, y)
    end
end

@testset "The stated roots" begin
    for sp in STANDARD_PROBLEMS
        sp.root === nothing && continue
        r = sp.root(Float64)
        y = zeros(sp.n)
        sp.prob.F(y, r, nothing)
        bound = sp.number == 3 ? 1e-6 : 100 * sp.n * eps(Float64) * max(1, norm(r, Inf))
        @test norm(y, Inf) <= bound
    end
    # Problem 3's stated root has seven significant digits, so its residual is not at round-off.
    r = STANDARD_PROBLEMS[3].root(Float64)
    y = zeros(2)
    STANDARD_PROBLEMS[3].prob.F(y, r, nothing)
    @test 1e-7 < norm(y, Inf) < 1e-6

    # The residual of a root does not see a coefficient of a term that vanishes there, such as
    # the 10 of the generalized Rosenbrock function, whose terms 10 (x[i+1] − x[i]²) are zero at
    # the root. At u0 = (−1.2, 1, …, 1) the residual is (2.2, −4.4, 0, …, 0) by hand.
    y = zeros(10)
    STANDARD_PROBLEMS[1].prob.F(y, STANDARD_PROBLEMS[1].u0(Float64), nothing)
    @test y ≈ [2.2; -4.4; zeros(8)] rtol = 8 * eps(Float64)
end

@testset "The cell runner" begin
    # The generalized Rosenbrock function from u0 with Newton and Static in Float64: one primal
    # call per iteration and one at the start, and one dual call per Jacobian chunk.
    sp = STANDARD_PROBLEMS[1]
    c, _ = run_cell(sp, "u0", sp.u0(Float64), "Newton Static LU",
        Newton(; globalization = Static()))
    chunks = cld(sp.n, chunk_size(AutoForwardDiff(), sp.n))
    @test c.iterations > 0
    @test c.f_evals == c.iterations + 1
    @test c.dual_evals == c.iterations * chunks
    @test (c.problem, c.start, c.column, c.eltype) ==
          (1, "u0", "Newton Static LU", "Float64")

    # A residual that throws gives the code ERROR in every column, and the runner goes on.
    throws = StandardProblem(0, "sqrt of a negative number", 1,
        (y, x, p) -> (y[1] = sqrt(x[1] - 10); nothing), T -> zeros(T, 1), nothing)
    for (column, method) in STANDARD_COLUMNS
        c, x = run_cell(throws, "u0", throws.u0(Float64), column, method)
        @test c.code == "ERROR"
        @test c.comment == "DomainError"
        @test x === nothing
    end
    c, _ = run_cell(throws, "u0", throws.u0(Float64), STANDARD_COLUMNS[1]...)
    @test compare_tables(StandardCell[], [c]) != String[]
    @test compare_tables([c], [c]) == String[]

    # An interrupt is rethrown, so that Ctrl-C stops a run.
    interrupts = StandardProblem(0, "interrupt", 1,
        (y, x, p) -> throw(InterruptException()), T -> zeros(T, 1), nothing)
    @test_throws InterruptException run_cell(
        interrupts, "u0", interrupts.u0(Float64), STANDARD_COLUMNS[1]...)
end

@testset "The perturbations of the marker" begin
    @testset "$T" for T in REAL_ELTYPES
        x₀ = T[-1.2, 1, 0, 3]
        for k in (1, 2, 4)
            x = perturb_start(x₀, k, 1)
            @test x isa Vector{T}
            @test all(x .== nextfloat.(x₀, k) .|| x .== prevfloat.(x₀, k))
            @test x == perturb_start(x₀, k, 1)
        end
        @test perturb_start(x₀, 1, 1) != perturb_start(x₀, 1, 2)

        # The residual moves by 1 ulp in a primal call only.
        sp = STANDARD_PROBLEMS[1]
        F = perturbed_residual(sp.prob.F, 1)
        x = sp.u0(T)
        y, y′ = zeros(T, sp.n), zeros(T, sp.n)
        sp.prob.F(y, x, nothing)
        F(y′, x, nothing)
        @test all(y′ .== nextfloat.(y) .|| y′ .== prevfloat.(y))
        xd = Dual.(x, one(T))
        yd, yd′ = zeros(eltype(xd), sp.n), zeros(eltype(xd), sp.n)
        sp.prob.F(yd, xd, nothing)
        F(yd′, xd, nothing)
        @test yd′ == yd

        # The perturbed direction keeps the method's step kind, and moves the first step.
        m = Newton(; globalization = Backtracking())
        mp = perturbed_direction(m, 1)
        @test mp.refinement isa PerturbedDirection
        @test GeometricSolvers.newton_stepkind(mp, T, true) ===
              GeometricSolvers.newton_stepkind(m, T, true) === GeometricSolvers.ExactStep()
        function primal_points(method)
            points = Vector{T}[]
            record = function (y, x, p)
                eltype(x) <: AbstractFloat && push!(points, copy(x))
                sp.prob.F(y, x, p)
            end
            run_cell(sp, "u0", sp.u0(T), "Newton Backtracking LU", method; F = record)
            return points
        end
        a, b = primal_points(m), primal_points(mp)
        @test a[1] == b[1]
        @test a[2] != b[2]
        @test isapprox(a[2], b[2]; rtol = sqrt(eps(T)))
        @test perturbed_direction(Picard(; globalization = Backtracking()), 1) === nothing
    end
    sp = STANDARD_PROBLEMS[1]
    method = Newton(; globalization = Static())
    @test length(perturbed_runs(sp, "u0", sp.u0(Float64), "Newton Static LU", method)) == 20
    picard = Picard(; globalization = Backtracking())
    @test length(perturbed_runs(sp, "u0", sp.u0(Float64), "Picard Backtracking", picard)) ==
          16
end

@testset "The rule of the marker" begin
    cell(code, iterations) = StandardCell(
        1, "u0", "Newton Static LU", "Float64", code,
        iterations, 7, 6, 1e-9, false, false, "")
    c = cell("SUCCESS", 5)
    x = [1.0, 2.0]
    tol = sqrt(eps(Float64)) * norm(x)
    @test !moved(c, x, c, copy(x))
    @test !moved(c, x, cell("SUCCESS", 6), x)
    @test !moved(c, x, cell("SUCCESS", 4), x)
    @test moved(c, x, cell("SUCCESS", 7), x)
    @test moved(c, x, cell("SUCCESS", 3), x)
    @test moved(c, x, cell("STALLED", 5), x)
    @test !moved(c, x, c, x .+ [tol / 2, 0])
    @test moved(c, x, c, x .+ [2 * tol, 0])
    # The distance is relative to max(1, ‖x‖): absolute below norm 1.
    small = [1e-3, 0.0]
    @test !moved(c, small, c, small .+ [sqrt(eps(Float64)) / 2, 0])
    @test moved(c, small, c, small .+ [2 * sqrt(eps(Float64)), 0])
    # A distance equal to the bound does not move.
    @test !moved(c, zeros(2), c, [sqrt(eps(Float64)), 0])
    # A final iterate finite in one run only moves; two that are not finite do not.
    @test moved(c, x, c, [NaN, 2.0])
    @test moved(c, [Inf, 2.0], c, x)
    @test !moved(c, [Inf, 2.0], c, [NaN, 2.0])
    # Two runs that throw have no iterate, and the same code.
    e = cell("ERROR", 0)
    @test !moved(e, nothing, e, nothing)
    @test moved(e, nothing, c, x)

    @test !marks_unstable(c, x, [(c, x), (cell("SUCCESS", 6), x)])
    @test marks_unstable(c, x, [(c, x), (cell("SUCCESS", 7), x)])
    @test !marks_unstable(c, x, [])
end

@testset "The marker" begin
    # Three cells that the diagnosis of `scripts/standard/README.md` found round-off-sensitive,
    # and three that no perturbation of it changed, in three other columns.
    sensitive = [(21, "u0", "Newton Backtracking LU"), (11, "100u0", "Newton Static LU"),
        (8, "u0", "Newton Backtracking LU")]
    stable = [(1, "u0", "Newton Backtracking LU"), (7, "u0", "Newton Bisection LU"),
        (20, "10u0", "Newton DogLeg LU")]
    columns = Dict(STANDARD_COLUMNS)
    function marked(problem, start, column)
        sp = STANDARD_PROBLEMS[problem]
        x₀ = Dict(standard_starts(sp, Float32))[start]
        c, x = run_cell(sp, start, x₀, column, columns[column])
        return marks_unstable(c, x, perturbed_runs(sp, start, x₀, column, columns[column]))
    end
    for cell in sensitive
        @test marked(cell...)
    end
    for cell in stable
        @test !marked(cell...)
    end

    # The run of the update script sets the flag of a marked cell; the run of the test file
    # sets none.
    sp = STANDARD_PROBLEMS[21]
    marked_cells, _ = run_standard_cells([sp]; mark = true)
    @test only(filter(
        c -> cell_key(c) == (21, "u0", "Newton Backtracking LU", "Float32"), marked_cells)).unstable
    plain_cells, _ = run_standard_cells([sp])
    @test !any(c -> c.unstable, plain_cells)
    @test [(c.code, c.iterations) for c in plain_cells] ==
          [(c.code, c.iterations) for c in marked_cells]
end

@testset "The comparison on synthetic tables" begin
    cell(code, iterations; unstable = false, platform = false, start = "u0") = StandardCell(
        1, start, "Newton Static LU", "Float64", code, iterations, 7, 6, 1e-9, unstable,
        platform, "")
    ref = [cell("SUCCESS", 5), cell("MAXITERS", 200; start = "10u0")]

    @test isempty(compare_tables(ref, ref))
    @test isempty(compare_tables(ref, [cell("SUCCESS", 6), ref[2]]))
    @test isempty(compare_tables(ref, [cell("SUCCESS", 4), ref[2]]))
    @test length(compare_tables(ref, [cell("SUCCESS", 7), ref[2]])) == 1
    @test length(compare_tables(ref, [cell("SUCCESS", 3), ref[2]])) == 1
    @test length(compare_tables(ref, [cell("STALLED", 5), ref[2]])) == 1
    # A cell of the run that the reference lacks, and one of the reference that the run lacks.
    @test length(compare_tables(ref, [ref; cell("SUCCESS", 5; start = "100u0")])) == 1
    @test length(compare_tables(ref, ref[1:1])) == 1

    # An unstable or a platform cell is checked only for its presence and a code other than
    # ERROR.
    for flag in ((; unstable = true), (; platform = true))
        fref = [cell("SUCCESS", 5; flag...), ref[2]]
        @test isempty(compare_tables(fref, [cell("SUCCESS", 9), ref[2]]))
        @test isempty(compare_tables(fref, [cell("NONFINITE", 5), ref[2]]))
        @test length(compare_tables(fref, [cell("ERROR", 0), ref[2]])) == 1
        @test length(compare_tables(fref, ref[2:2])) == 1
    end

    # A regeneration keeps the platform flag and its comment, and takes the unstable flag from
    # the run.
    flagged = StandardCell(
        1, "u0", "Newton Static LU", "Float64", "SUCCESS", 5, 7, 6, 1e-9,
        true, true, "ubuntu-latest: 6 iterations")
    kept = keep_platform([flagged, ref[2]],
        [
            cell("SUCCESS", 4), cell("MAXITERS", 200;
                start = "10u0", unstable = true)])
    @test kept[1].platform && kept[1].comment == "ubuntu-latest: 6 iterations"
    @test kept[1].iterations == 4
    @test !kept[1].unstable
    @test !kept[2].platform && kept[2].unstable

    # The table reads back what it writes, the flags and a comment with a comma included.
    io = IOBuffer()
    write_table(io, [flagged, kept[2]])
    seekstart(io)
    back = read_table(io)
    @test cell_key.(back) == cell_key.([flagged, kept[2]])
    @test back[1].unstable && back[1].platform
    @test back[1].comment == "ubuntu-latest: 6 iterations"
    @test back[2].unstable && !back[2].platform
    io = IOBuffer()
    write_table(io, [StandardCell(
        1, "u0", "c", "Float64", "SUCCESS", 5, 7, 6, 1e-9, false, true, "a, b")])
    seekstart(io)
    @test only(read_table(io)).comment == "a, b"
    @test_throws ArgumentError read_table(IOBuffer("problem,start\n"))
end

@testset "The reference" begin
    cells, iterates = run_standard_cells()
    codes = [collect(string.(instances(ReturnCode))); "ERROR"]
    @test length(cells) == 67 * length(STANDARD_COLUMNS) * length(REAL_ELTYPES) == 1206
    @test all(c -> c.code in codes, cells)
    # Every cell runs with max_iterations = 200.
    @test any(c -> c.code == "MAXITERS", cells)
    @test all(c -> c.code != "MAXITERS" || c.iterations == 200, cells)

    reference = read_table(STANDARD_REFERENCE)
    differences = compare_tables(reference, cells)
    foreach(d -> println("standard test set: ", d), differences)
    @test isempty(differences)

    # The summary counts the SUCCESS cells of each column and element type out of 67.
    io = IOBuffer()
    print_summary(io, cells, iterates)
    summary = String(take!(io))
    for (column, _) in STANDARD_COLUMNS
        line = only(filter(startswith(column * " "), split(summary, "\n")))
        for T in string.(REAL_ELTYPES)
            n = count(c -> (c.column, c.eltype, c.code) == (column, T, "SUCCESS"), cells)
            @test occursin(Regex("\\s$n/67,"), line)
        end
    end
    print(summary)
end

@testset "The distance to the stated root" begin
    rosenbrock = STANDARD_PROBLEMS[1]
    @testset "$T" for T in REAL_ELTYPES
        @test near_root(rosenbrock, ones(T, 10))
        @test near_root(rosenbrock, ones(T, 10) .+ sqrt(eps(T)) / 10)
        @test !near_root(rosenbrock, ones(T, 10) .+ 2 * sqrt(eps(T)))
        # A root of norm below 1 is measured in absolute distance.
        @test near_root(STANDARD_PROBLEMS[19], fill(sqrt(eps(T)) / 10, 2))
        @test !near_root(STANDARD_PROBLEMS[19], fill(sqrt(eps(T)), 2))
        @test near_root(STANDARD_PROBLEMS[6], zeros(T, 2)) === nothing
    end
end
