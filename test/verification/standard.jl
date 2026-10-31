# The standard test set: the 23 problems of NonlinearProblemLibrary.jl, from three starts, with
# every column of `STANDARD_COLUMNS`, in each element type of `REAL_ELTYPES`, compared with the
# reference table `test/verification/standard_reference.csv`.
#
# A cell that changes is regenerated with `scripts/standard/update_reference.jl`, and the pull
# request that changes it lists the cell with its cause.

using GeometricSolvers
using GeometricSolvers: AutoForwardDiff, ReturnCode, chunk_size
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
end

@testset "The comparison on synthetic tables" begin
    cell(code, iterations; platform = false, start = "u0") = StandardCell(
        1, start, "Newton Static LU", "Float64", code, iterations, 7, 6, 1e-9, platform, "")
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

    # A platform cell is compared on its code alone.
    pref = [cell("SUCCESS", 5; platform = true)]
    @test isempty(compare_tables(pref, [cell("SUCCESS", 9)]))
    @test length(compare_tables(pref, [cell("NONFINITE", 5)])) == 1

    # A regeneration keeps the platform flag and its comment.
    flagged = StandardCell(
        1, "u0", "Newton Static LU", "Float64", "SUCCESS", 5, 7, 6, 1e-9,
        true, "ubuntu-latest: 6 iterations")
    kept = keep_platform([flagged, ref[2]], [
        cell("SUCCESS", 4), cell("MAXITERS", 200;
            start = "10u0")])
    @test kept[1].platform && kept[1].comment == "ubuntu-latest: 6 iterations"
    @test kept[1].iterations == 4
    @test !kept[2].platform

    # The table reads back what it writes, a comment with a comma included.
    io = IOBuffer()
    write_table(io, [flagged, ref[2]])
    seekstart(io)
    back = read_table(io)
    @test cell_key.(back) == cell_key.([flagged, ref[2]])
    @test back[1].platform && back[1].comment == "ubuntu-latest: 6 iterations"
    io = IOBuffer()
    write_table(io, [StandardCell(
        1, "u0", "c", "Float64", "SUCCESS", 5, 7, 6, 1e-9, true, "a, b")])
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
