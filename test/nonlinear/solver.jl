# The nonlinear solver loop: `init_state`, `solve!!`, `solve`, the options, the step test, and
# the six `ReturnCode` exits.

using Adapt: Adapt
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using JET: JET
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, JacobianFunction, LINESEARCH_FAILED,
                        LUFactorization, MAXITERS, NONFINITE, Newton, NoRefinement,
                        NonlinearProblem, Options, Picard, QRFactorization, SINGULAR,
                        STALLED, SUCCESS, SolverStatus, Static, StrongWolfe, ToReal, init,
                        init_state, rnorm, solve, solve!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_SOLVER = 4

function solver_allocated(prob, x, m, p)
    solve(prob, x, m, p)
    return @allocated solve(prob, x, m, p)
end

function strip_comments_and_docstrings(path::AbstractString)
    lines = readlines(path)
    code_lines = String[]
    in_doc = false
    for raw in lines
        s = strip(raw)
        if startswith(s, "\"\"\"")
            # A single-line """...""" docstring toggles twice
            if endswith(s, "\"\"\"") && length(s) >= 6
                continue
            end
            in_doc = !in_doc
            continue
        end
        in_doc && continue
        startswith(s, "#") && continue
        # Strip trailing comment
        idx = findfirst('#', raw)
        line = idx === nothing ? raw : raw[1:prevind(raw, idx)]
        push!(code_lines, line)
    end
    return code_lines
end

@testset "init prepares AD or JacobianFunction for each (m.jacobian, prob.J) cell, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_SOLVER)
    prob_no_j = NonlinearProblem(R1Scaled())
    prob_with_j = NonlinearProblem(R1Scaled(), r1_scaled_problem().J)
    custom_j = (y, q) -> r1_scaled_jacobian(y, q)
    b = AutoForwardDiff()

    # The six combinations of m.jacobian and prob.J
    s1 = init_state(Newton(), prob_with_j, x, p)
    @test !(s1 isa NamedTuple)
    @test s1.jacprep === nothing
    @test s1.jvpprep === nothing
    @test isbits(s1)

    s2 = init_state(Newton(), prob_no_j, x, p)
    @test s2.jacprep === b
    @test s2.jvpprep === b
    @test isbits(s2)

    s3 = init_state(Newton(; jacobian = b), prob_no_j, x, p)
    @test s3.jacprep === b
    @test s3.jvpprep === b
    @test isbits(s3)

    s4 = init_state(Newton(; jacobian = b), prob_with_j, x, p)
    @test s4.jacprep === b
    @test s4.jvpprep === nothing
    @test isbits(s4)

    s5 = init_state(Newton(; jacobian = custom_j), prob_with_j, x, p)
    @test s5.jacprep === JacobianFunction(custom_j)
    @test s5.jvpprep === nothing
    @test isbits(s5)

    s6 = init_state(Newton(; jacobian = custom_j), prob_no_j, x, p)
    @test s6.jacprep === JacobianFunction(custom_j)
    @test s6.jvpprep === b
    @test isbits(s6)

    # Picard prepares only jvpprep
    sp1 = init_state(Picard(), prob_with_j, x, p)
    @test !hasfield(typeof(sp1), :jacprep)
    @test sp1.jvpprep === nothing
    @test isbits(sp1)

    sp2 = init_state(Picard(), prob_no_j, x, p)
    @test sp2.jvpprep === b
    @test isbits(sp2)
end

@testset "isbits state for a heap problem and Options conversion" begin
    x = SVector{2, Float64}(3.0, 5.0)
    c = [4.0, 9.0]
    Fc(y, q) = y .^ 2 .- SVector(c[1], c[2])
    prob_c = NonlinearProblem(Fc)
    for m in (Newton(), Picard())
        @test isbits(m)
        @test isbits(init_state(m, prob_c, x, [1.0]))
        @test isbits(solve(prob_c, x, m, [1.0])[2])
    end

    # Options{Float32} passed whole solve a Float32 problem
    prob32 = NonlinearProblem(
        (y, q) -> y .^ 2 .- q,
        (y, q) -> SMatrix{2, 2, Float32}(2 * y[1], 0.0f0, 0.0f0, 2 * y[2]))
    x32 = SVector{2, Float32}(3.0f0, 5.0f0)
    p32 = SVector{2, Float32}(4.0f0, 9.0f0)
    _, st32 = solve(
        prob32, x32, Newton(), p32; options = Options(Float32; f_reltol = 1e-5))
    @test st32 isa SolverStatus{Float32}
    @test st32.code == SUCCESS

    # Options and keyword tolerances together are rejected, not silently merged
    @test_throws ArgumentError solve(
        prob32, x32, Newton(), p32; options = Options(Float32), max_iterations = 1)
end

@testset "options and keywords together raise one message from solve and init" begin
    pr = NonlinearProblem((x, p) -> x .* x .- 2)
    both = (; options = Options(Float64), f_reltol = 1e-3)
    for call in (() -> solve(pr, [1.0], Newton(); both...),
        () -> solve(pr, SVector(1.0), Newton(); both...),
        () -> init(pr, [1.0], Newton(); both...))
        err = try
            call()
        catch e
            e
        end
        @test err isa ArgumentError
        @test err.msg == "pass either `options` or keyword arguments, not both"
    end
end

@testset "an f_reltol by keyword or by Options(Float32) solves alike (F4)" begin
    prob = NonlinearProblem((x, p) -> SVector(x[1]^2 - 2.0f0))
    sts = [solve(prob, SVector(10.0f0), Newton(); kw...)[2]
           for kw in ((; f_reltol = 1e-6), (; options = Options(Float32; f_reltol = 1e-6)))]
    @test all(st -> st.code == SUCCESS, sts)
    @test allequal(st.iterations for st in sts)
    @test allequal(st.fnorm for st in sts)
end

# The default `f_reltol = √eps(Float64)` lies below `eps(Float32)`, so a `Float32` solve that took
# `Options(Float64)` would stall on 37 % of the starts of `x² − 2`. No precision converts to another.
@testset "options of another precision than the iterate raise an ArgumentError: $T" for T in REAL_ELTYPES
    O = T === Float32 ? Float64 : Float32
    prob = NonlinearProblem((x, p) -> x .* x .- T(2))
    x₀ = SVector(T(1.5))
    m = Adapt.adapt(ToReal{T}(), Newton())
    wrong = Options(O)
    @test_throws ArgumentError solve(prob, x₀, Newton(); options = wrong)
    @test_throws ArgumentError init(prob, [T(1.5)], Newton(); options = wrong)
    @test_throws ArgumentError solve!!(
        init_state(m, prob, x₀, nothing), prob, x₀, m, nothing, wrong)
    err = try
        solve(prob, x₀, Newton(); options = wrong)
    catch e
        e
    end
    @test occursin("Options{$O}", err.msg)
    @test occursin("Options($T", err.msg)

    # the options of the iterate's own precision solve it
    right = Options(T)
    @test solve(prob, x₀, Newton(); options = right)[2].code == SUCCESS
    @test solve!!(init_state(m, prob, x₀, nothing), prob, x₀, m, nothing, right)[2].code ==
          SUCCESS
end

@testset "a zero step returns STALLED at once (K13), $R" for R in REAL_ELTYPES
    # the factor of an infinite Jacobian is not singular, and the step -F / Inf is zero
    prob = NonlinearProblem((x, p) -> x * x - 4, (x, p) -> R(Inf))
    x, st = solve(prob, R(3), Newton())
    @test st.code == STALLED
    @test st.iterations == 1
    @test x == 3
    @test st.fnorm == 5
    @test iszero(st.stepnorm)
    # `min_iterations` bounds it: the same zero step repeats until then
    x, st = solve(prob, R(3), Newton(); min_iterations = 3)
    @test st.code == STALLED
    @test st.iterations == 3
    @test x == 3
end

@testset "the step test returns STALLED where the residual test fails, $T" for T in ELTYPES
    R = real(T)
    prob = NonlinearProblem((x, p) -> x * x - 2)
    # Newton's iteration for √2 from 10, by hand: the first step whose norm is at most |x₊| / 2
    x = T(10)
    k = 0
    while true
        d = (x * x - 2) / (2x)
        x -= d
        k += 1
        abs(d) <= abs(x) / 2 && break
    end
    @test k == 4
    x₄, st = solve(prob, T(10), Newton(); x_reltol = 0.5, f_reltol = 1e-12)
    @test st.code == STALLED
    @test st.iterations == k
    @test x₄ ≈ x rtol = 8 * eps(R)
    @test st.fnorm > R(1e-12) * rnorm(prob.F(T(10), nothing))
    @test st.fnorm > R(0.08)
    # `min_iterations` bounds the step test
    _, st₆ = solve(
        prob, T(10), Newton(); x_reltol = 0.5, f_reltol = 1e-12, min_iterations = 6)
    @test st₆.code == STALLED
    @test st₆.iterations == 6
    # a step that a radius cuts is no step test: from 1 on x - 2, DogLeg's radius starts at 1e-6
    # and grows by 4 a step, each step far below |x| / 2, and the solve runs to MAXITERS
    if T <: Real
        lin = NonlinearProblem((x, p) -> x - 2, (x, p) -> one(x))
        x_dl, st_dl = solve(lin, T(1), Newton(; globalization = DogLeg(; radius = 1e-6));
            x_reltol = 0.5, max_iterations = 3)
        @test st_dl.code == MAXITERS
        @test st_dl.iterations == 3
        @test 1 < x_dl < 1 + R(1e-4)
    end
    # with the default options the residual test ends the solve
    _, st_ok = solve(prob, T(10), Newton())
    @test st_ok.code == SUCCESS
    @test st_ok.fnorm <= sqrt(eps(R)) * rnorm(prob.F(T(10), nothing))
end

@testset "solve!! and solve: the six ReturnCode exits and edge cases, $T" for T in ELTYPES
    R = real(T)
    x₀ = SVector{2, T}(T(3), T(5))
    p = SVector{2, T}(T(4), T(9))
    prob = NonlinearProblem((y, q) -> y .^ 2 .- q, (y, q) -> diagonal_matrix(T(2) .* y))

    # 1. SUCCESS: residual convergence and agreement between solve and solve!!
    x_sol, st_sol = solve(prob, x₀, Newton(), p)
    m_r = Adapt.adapt(ToReal{R}(), Newton())
    state₀ = init_state(m_r, prob, x₀, p)
    x_sol2, st_sol2, _ = solve!!(state₀, prob, x₀, m_r, p, Options(T))
    @test x_sol === x_sol2
    @test st_sol === st_sol2
    @test st_sol.code == SUCCESS
    @test st_sol.iterations >= 1
    @test st_sol.fnorm <=
          max(Options(T).f_abstol, Options(T).f_reltol * rnorm(prob.F(x₀, p)))
    @test st_sol.step_failures == 0
    @test !st_sol.promoted
    @test isapprox(x_sol, SVector{2, T}(T(2), T(3)); rtol = 10 * sqrt(eps(R)))

    # Exact root: 0 iterations, SUCCESS, x === x_root
    x_root = SVector{2, T}(T(2), T(3))
    x_z, st_z = solve(prob, x_root, Newton(), p)
    @test x_z === x_root
    @test st_z === SolverStatus{R}(SUCCESS, Int32(0), zero(R), zero(R), Int32(0), false)

    # One ulp from a root: default options take >= 1 step; f_abstol = floor takes 0 steps
    x_ulp = SVector{2, T}(T(nextfloat(R(2))), T(3))
    _, st_ulp1 = solve(prob, x_ulp, Newton(), p)
    @test st_ulp1.iterations >= 1
    floor_ulp = rnorm(prob.F(x_ulp, p))
    x_ulp0, st_ulp0 = solve(prob, x_ulp, Newton(), p; f_abstol = floor_ulp)
    @test st_ulp0.code == SUCCESS
    @test st_ulp0.iterations == 0
    @test x_ulp0 === x_ulp

    # Invalid tolerances raise ArgumentError
    @test_throws ArgumentError solve(prob, x_ulp, Newton(), p; f_abstol = NaN)
    @test_throws ArgumentError solve(prob, x_ulp, Newton(), p; x_reltol = -1)

    # All tolerances 0 give SUCCESS at F == 0
    aff = NonlinearProblem(
        (y, q) -> T(2) .* y .- q, (y, q) -> T(2) .* identity_matrix(T, 2))
    x_aff, st_aff = solve(
        aff, x₀, Newton(), SVector{2, T}(T(2), T(2));
        f_abstol = 0, f_reltol = 0, x_abstol = 0, x_reltol = 0)
    @test st_aff.code == SUCCESS
    @test x_aff == SVector{2, T}(one(T), one(T))

    # n = 0 gives SUCCESS after 0 iterations
    x_n0, st_n0 = solve(NonlinearProblem((y, q) -> y), SVector{0, T}(), Newton(), nothing)
    @test st_n0.code == SUCCESS
    @test st_n0.iterations == 0
    @test x_n0 === SVector{0, T}()

    # Already-converged x₀ with min_iterations = 2 runs 2 steps
    _, st_min = solve(prob, x_root, Newton(), p; min_iterations = 2)
    @test st_min.code == SUCCESS
    @test st_min.iterations == 2

    # 2. MAXITERS: max_iterations = 0 off a root vs at a root
    x_m0_start = SVector{2, T}(one(T), one(T))
    x_m0, st_m0 = solve(prob, x_m0_start, Newton(), p; max_iterations = 0)
    @test x_m0 === x_m0_start
    @test st_m0.code == MAXITERS
    @test st_m0.iterations == 0

    x_m0_root, st_m0_root = solve(prob, x_root, Newton(), p; max_iterations = 0)
    @test x_m0_root === x_root
    @test st_m0_root.code == SUCCESS
    @test st_m0_root.iterations == 0

    # 3. SINGULAR: singular Jacobian at x₀ returns SINGULAR at iteration 1 with x === x₀
    prob_sing = NonlinearProblem((y, q) -> SVector(y[1] - one(eltype(y)), y[1] -
                                                                          one(eltype(y))))
    x_sg, st_sg = solve(prob_sing, x₀, Newton(), p)
    @test x_sg === x₀
    @test st_sg.code == SINGULAR
    @test st_sg.iterations == 1
    @test st_sg.fnorm === rnorm(prob_sing.F(x₀, p))
    @test iszero(st_sg.stepnorm)

    # 4. NONFINITE at F(x₀): 0 steps, NONFINITE, x === x₀
    x_nf_start = SVector{2, T}(one(T), one(T))
    p_nf = SVector{2, T}(T(Inf), one(T))
    x_nf0, st_nf0 = solve(prob, x_nf_start, Newton(), p_nf)
    @test x_nf0 === x_nf_start
    @test st_nf0.code == NONFINITE
    @test st_nf0.iterations == 0
    @test !isfinite(st_nf0.fnorm)

    # NONFINITE on a trial step: returns the last finite iterate and its fnorm
    prob_step_nf = NonlinearProblem(
        (y, q) -> y == x₀ ? SVector{2, T}(one(T), one(T)) : SVector{2, T}(T(NaN), one(T)),
        (y, q) -> identity_matrix(T, 2))
    x_snf, st_snf = solve(prob_step_nf, x₀, Newton(), p)
    @test x_snf === x₀
    @test st_snf.code == NONFINITE
    @test st_snf.iterations == 1
    @test isfinite(st_snf.fnorm)
    @test st_snf.fnorm === rnorm(prob_step_nf.F(x₀, p))

    # 5. STALLED: tiny step where x + αd == x triggers STALLED after max_stalls (2)
    big_j = SMatrix{2, 2, T}(R(1e20), zero(T), zero(T), R(1e20))
    prob_stall = NonlinearProblem((y, q) -> y .- q, (y, q) -> big_j)
    x_st_start = SVector{2, T}(T(2), T(2))
    x_st, st_st = solve(
        prob_stall, x_st_start, Newton(; globalization = Backtracking()),
        SVector{2, T}(one(T), one(T)))
    @test x_st === x_st_start
    @test st_st.code == STALLED
    @test st_st.iterations == 2
    @test iszero(st_st.stepnorm)
    @test st_st.step_failures == 0

    # A stalled step whose residual already satisfies f_abstol DOES return SUCCESS
    p_st_one = SVector{2, T}(one(T), one(T))
    fn_st = rnorm(prob_stall.F(x_st_start, p_st_one))
    _, st_st_ok = solve(
        prob_stall, x_st_start, Newton(; globalization = Backtracking()),
        p_st_one; min_iterations = 1, f_abstol = fn_st)
    @test st_st_ok.code == SUCCESS
    @test st_st_ok.iterations == 1

    # Non-consecutive stalls: a successful step between two stalls resets the stall counter
    step_Toggle = Ref(0)
    id2 = identity_matrix(T, 2)
    prob_alt_stall = NonlinearProblem(
        (y, q) -> y .- q,
        (y, q) -> (step_Toggle[] += 1; isodd(step_Toggle[]) ? big_j : R(2) .* id2))
    _, st_alt = solve(
        prob_alt_stall, x_st_start, Newton(; globalization = Backtracking()), p_st_one;
        max_iterations = 3, max_stalls = 2,
        f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
    @test st_alt.code == MAXITERS
    @test st_alt.iterations == 3

    # 6. LINESEARCH_FAILED
    prob_up = NonlinearProblem((y, q) -> y, (y, q) -> -identity_matrix(T, 2))
    x_lf, st_lf = solve(
        prob_up, x₀, Newton(; globalization = Backtracking(; maxiter = 2)), p)
    @test x_lf === x₀
    @test st_lf.code == LINESEARCH_FAILED
    @test st_lf.iterations == 1
    @test st_lf.step_failures == 1
    @test iszero(st_lf.stepnorm)
end

@testset "Inference, zero allocation, and JET.@test_opt on solve, $T" for T in ELTYPES
    prob = NonlinearProblem((y, q) -> y .^ 2 .- q)
    x₀ = SVector{2, T}(T(3), T(5))
    p = SVector{2, T}(T(4), T(9))
    for m in (Newton(), Picard())
        @test (@inferred(solve(prob, x₀, m, p)); true)
        # Under `--check-bounds=yes` a bounds check lets the local `MMatrix` and `MVector`
        # of `StaticLU` escape to the heap, so the `Newton` check is skipped where a run
        # forces `--check-bounds=yes`.
        if Base.JLOptions().check_bounds != 1 || m isa Picard
            @test solver_allocated(prob, x₀, m, p) == 0
        end
    end
end

@testset "JET.@test_opt on solve for Newton() and Picard()" begin
    prob = NonlinearProblem((y, q) -> y .^ 2 .- q)
    for T in ELTYPES, m in (Newton(), Picard())

        x₀ = SVector{2, T}(T(3), T(5))
        p = SVector{2, T}(T(4), T(9))
        JET.@test_opt solve(prob, x₀, m, p)
    end
end

@testset "Storage clause: no direct storage or written-out iterate update" begin
    pkg_root = normpath(joinpath(@__DIR__, "..", ".."))
    files = (
        "src/nonlinear/solver.jl",
        "src/nonlinear/newton.jl",
        "src/nonlinear/picard.jl",
        "src/globalization/linefunction.jl",
        "src/globalization/dogleg.jl"
    )
    bad_storage = r"(\.=|\bsetindex!\b|\bcopyto!\b|\bfill!\b|\bmul!\b|\b[A-Za-z_][A-Za-z0-9_]*\[[^]]+\]\s*=[^=])"
    bad_update = r"[+-]\s*α[A-Za-z0-9_₀-₉]*\s*\*"
    for rel in files
        code_lines = strip_comments_and_docstrings(joinpath(pkg_root, rel))
        @test all(l -> !occursin(bad_storage, l), code_lines)
        @test all(l -> !occursin(bad_update, l), code_lines)
    end
end
