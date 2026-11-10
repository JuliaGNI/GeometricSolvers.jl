# R5 matrix-free Newton–GMRES: the forcing terms, the right-preconditioner slot, the inexact
# backtracking test, the inner-iteration cap, the matrix-free `DogLeg`, and the decided edges.
#
# The solves run on `Array` and on `JLArray` (a CPU array that refuses scalar indexing, so it
# catches a GPU-only fault in ordinary CI) in `REAL_ELTYPES`, and the matrix-free `DogLeg` is
# compared with the factorised one, which is the R3 control of the same problem.

using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: Diagonal, I, mul!, norm
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, ChunkedForwardDiff, DIJacobian, DogLeg,
                        EisenstatWalkerChoice1, EisenstatWalkerChoice2, FixedForcing, GMRES,
                        InexactStep, JVPFunction, LevenbergMarquardt, LineFunction,
                        NONFINITE, NonlinearProblem, Newton, Refinement, StrongWolfe,
                        SUCCESS, SolverStatus, apply_preconditioner!, f_calls, gmres_solve!,
                        init, init_gmres_workspace, init_state, jvp_calls,
                        max_inner_iterations, refresh!, rnorm, solve, solve!, solve!!,
                        start!!, step!!

include("../helpers/matrix.jl")

allowscalar(false)

# ─── The problems ────────────────────────────────────────────────────────────

# The matrix-free R5 problem of Check 1: a diagonal residual `F_i = x_i^3 + 2x_i - 3`, whose
# Jacobian `3x_i^2 + 2` no array holds, with the counted residual and the counted product.
#
# `R5Diagonal` is the residual of the counted wrapper, and the chunked forward mode differentiates
# it through the same buffer: a call on a `Dual` iterate is a Jacobian-vector product, so the
# wrapper counts it as one, and a call on the plain element type is one evaluation of `F`. The two
# counts then partition the calls, and each equals the count the solve reports.
mutable struct R5Counts
    f::Int
    jvp::Int
end

struct R5Diagonal{C}
    counts::C
end

function (p::R5Diagonal)(r, x, q)
    if eltype(x) <: ForwardDiff.Dual
        p.counts.jvp += 1
    else
        p.counts.f += 1
    end
    @. r = x^3 + 2x - 3
    return r
end

struct R5DiagonalJVP{C}
    counts::C
end

function (p::R5DiagonalJVP)(Jv, x, v, q)
    p.counts.jvp += 1
    @. Jv = (3x^2 + 2) * v
    return Jv
end

r5_problem(counts) = NonlinearProblem(R5Diagonal(counts))

# The 4 × 4 affine system of Check 4 and Check 6: `F(x) = A x - b`, whose Jacobian is `A`.
const R5_A4 = [4.0 1.0 0.5 0.2
               0.3 3.0 0.8 0.1
               0.2 0.4 2.5 0.6
               0.1 0.2 0.3 2.0]
const R5_B4 = [1.0, -2.0, 1.5, -0.5]

function r5_affine(counts = nothing)
    return NonlinearProblem((r, x, q) -> begin
        counts === nothing || (counts.f += 1)
        mul!(r, R5_A4, x)
        r .-= R5_B4
        r
    end)
end

# A residual whose Jacobian is the diagonal `4, 3, 2.5, 2` and whose root is exactly representable,
# so that `‖F(x₀)‖ = 0` holds at `x₀` in floating point (edge E2).
function r5_exact_root()
    return NonlinearProblem((r, x, q) -> begin
        r[1] = 4.0 * x[1] - 2.0
        r[2] = 3.0 * x[2] + 3.0
        r[3] = 2.5 * x[3] - 0.625
        r[4] = 2.0 * x[4] - 1.0
        r
    end)
end

const R5_EXACT_X = [0.5, -1.0, 0.25, 0.5]

# `atan`, whose Newton step from 1.5 overshoots to about −1.69, where the residual is larger, so
# the trial at the full step fails the comparison of the inexact test and the search backtracks.
const R5_FAIL_START = 1.5
r5_full_step_fails() = NonlinearProblem((r, x, q) -> (r[1] = atan(x[1]); r))

# A coupled cubic residual of four unknowns. Its Jacobian is no multiple of the identity, so its
# Newton solves take more than one step and a two-iteration GMRES stops short of the tolerance:
# the linear residual of a step is then a visible fraction of its residual norm, which is what the
# Choice 1 rule reads and what the inexact step kind reports.
function r5_coupled(::Type{T}) where {T}
    c = T(0.5)
    return NonlinearProblem((r, x, q) -> begin
        n = length(x)
        for i in 1:n
            next = i < n ? x[i + 1] : x[1]
            r[i] = x[i]^3 + 2 * x[i] - 3 + c * (next - x[i])
        end
        r
    end)
end

const R5_COUPLED_X = (0.3, 1.1, -0.7, 2.0)

# `atan` from a start whose Newton step overshoots to a point where the residual is 0.94 of the
# anchor one. A one-dimensional solve with one product is exact, so the attained tolerance is
# zero: the bound of the inexact test is then `1 - c₁ = 0.5` at the full step, and the 0.94 ratio
# fails it, while the bound of the forcing term's own tolerance, `1 - c₁ (1 - 0.9) = 0.95`,
# accepts it.
const R5_ATAN_START = 1.3
r5_atan() = NonlinearProblem((r, x, q) -> (r[1] = atan(x[1]); r))

# ─── The allocation row of the inner solve ──────────────────────────────────

# An `@allocated` assertion measures through a function barrier whose arguments have concrete types
# and which calls the function once before it measures.
function gmres_allocations(ws, op, b, β, tol, maxiter, restart)
    gmres_solve!(ws, op, nothing, b, β, tol, maxiter, restart)
    return @allocated gmres_solve!(ws, op, nothing, b, β, tol, maxiter, restart)
end

@testset "R5 matrix-free Newton–GMRES" begin
    @testset "Check 1: no Jacobian, exact F and JVP counts, $AT, $T" for T in REAL_ELTYPES,
        AT in ARRAY_BACKENDS

        counts = R5Counts(0, 0)
        prob = r5_problem(counts)
        x₀ = AT(fill(T(2), 8))
        m = Newton(;
            linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(1e-8)),
            jacobian = JVPFunction(R5DiagonalJVP(counts)))
        solver = init(prob, x₀, m)
        # MU8: the R5 state allocates no Jacobian matrix and no factorisation slot
        @test solver.state.J === nothing
        @test solver.state.fac_T === nothing
        @test solver.state.fac === nothing
        @test solver.state.jvpprep.prep isa JVPFunction

        counts.f = 0
        counts.jvp = 0
        _, st = solve!(solver, x₀)
        fnorm₀ = rnorm(AT(fill(T(-1.5), 8)))
        # MU1: the reported counts are those of the solve, in the status and in the accessors
        @test st.code == SUCCESS
        @test st.fnorm <= max(zero(T), sqrt(eps(T)) * fnorm₀)
        @test st.f_calls == counts.f
        @test st.jvp_calls == counts.jvp
        @test f_calls(solver) == counts.f
        @test jvp_calls(solver) == counts.jvp
        @test st.f_calls isa Int16 && st.jvp_calls isa Int16
    end

    @testset "Check 1: the default preparation is the chunked one-partial product, $T" for T in REAL_ELTYPES
        counts = R5Counts(0, 0)
        solver = init(r5_problem(counts), zeros(T, 8), Newton(; linsolve = GMRES()))
        prep = solver.state.jvpprep.prep
        @test prep isa ChunkedForwardDiff{1}
        # the Jacobian buffers of a solve that forms no Jacobian are not allocated
        @test prep.xdual === nothing && prep.rdual === nothing
        _, st = solve!(solver, zeros(T, 8))
        @test st.code == SUCCESS
        @test st.f_calls == counts.f
        @test st.jvp_calls == counts.jvp
        @test st.jvp_calls > 0
    end

    @testset "Check 1: another AD backend forms the product through DI, Float64" begin
        prob = r5_affine()
        solver = init(
            prob, zeros(4), Newton(; linsolve = GMRES(), jacobian = AutoFiniteDiff()))
        @test solver.state.jvpprep.prep isa DIJacobian
        x, st = solve!(solver, zeros(4))
        @test st.code == SUCCESS
        @test norm(x - R5_A4 \ R5_B4) <= 100 * sqrt(eps(Float64)) * norm(R5_A4 \ R5_B4)
    end

    @testset "Check 1: the status fields keep the size of the struct" begin
        @test sizeof(SolverStatus{Float32}) == 36
        @test sizeof(SolverStatus{Float64}) == 56
        @test isbitstype(SolverStatus{Float64})
        @test fieldoffset(SolverStatus{Float64}, :f_calls) == 2
        @test fieldoffset(SolverStatus{Float64}, :jvp_calls) == 30

        # the counts saturate at `typemax(Int16)` and never wrap or go negative
        st = SolverStatus{Float64}(SUCCESS, Int32(1), 0.0, 0.0, Int32(0), false)
        big = GeometricSolvers.with_calls(st, 40000, 40000)
        @test big.f_calls == typemax(Int16)
        @test big.jvp_calls == typemax(Int16)
        small = GeometricSolvers.with_calls(st, -1, -1)
        @test small.f_calls == 0
        @test small.jvp_calls == 0
    end

    @testset "Check 1: the counts of a DogLeg solve, $T" for T in REAL_ELTYPES
        # the matrix-free trial loop of `DogLeg` evaluates one residual per trial, and the status
        # counts it
        counts = R5Counts(0, 0)
        prob = r5_problem(counts)
        x₀ = fill(T(2), 8)
        m = Newton(;
            linsolve = GMRES(; maxiter = 8, forcing = FixedForcing(1e-10)),
            globalization = DogLeg(), jacobian = JVPFunction(R5DiagonalJVP(counts)))
        solver = init(prob, x₀, m)
        _, st = solve!(solver, copy(x₀))
        @test st.code == SUCCESS
        @test st.f_calls == counts.f
        @test st.jvp_calls == counts.jvp
        # a trial was evaluated, so the trial loop's own count is in the status
        @test st.f_calls > st.iterations
    end

    @testset "Check 1: the counts of a line search that measures the slope, $T" for T in REAL_ELTYPES
        # `Bisection` and `StrongWolfe` evaluate the slope of the merit at every trial. That
        # product is a Jacobian–vector product, not an evaluation of `F`, and the problem the line
        # search hands to the product is the one the preparation was built from
        for g in (Bisection(), StrongWolfe())
            counts = R5Counts(0, 0)
            prob = r5_problem(counts)
            x₀ = fill(T(2), 8)
            solver = init(
                prob, x₀, Newton(; linsolve = GMRES(; maxiter = 10), globalization = g))
            _, st = solve!(solver, copy(x₀))
            @test st.code == SUCCESS
            @test st.f_calls == counts.f
            @test st.jvp_calls == counts.jvp
            @test st.jvp_calls > st.iterations
        end
    end

    @testset "Check 1: a DI product under a line search, Float64" begin
        # the preparation of a DifferentiationInterface back end is called on the function it
        # prepared, which the counting wrapper of a line search is not
        x₀ = Float64[0.3, 1.1, -0.7, 2.0]
        for g in (Bisection(), StrongWolfe())
            solver = init(r5_coupled(Float64), copy(x₀),
                Newton(; linsolve = GMRES(; maxiter = 8), jacobian = AutoFiniteDiff(),
                    globalization = g))
            x, st = solve!(solver, copy(x₀))
            @test st.code == SUCCESS
            @test norm(x .^ 3 .+ 2 .* x .- 3 .+ Float64(0.5) .* (circshift(x, -1) .- x)) <=
                  1e-8
        end
    end
end

# ─── The forcing terms (Check 2) ─────────────────────────────────────────────

# Choice 2 and Choice 1 written out as the plan states them, for the tests to compare against.
function reference_choice2(::Type{R}, norms, η_max, η₀ = R(0.5)) where {R}
    γ = R(0.9)
    α = R(2)
    η = η₀
    out = R[η]
    for k in 1:(length(norms) - 1)
        raw = γ * (R(norms[k + 1]) / R(norms[k]))^α
        safe = γ * η^α > R(0.1) ? max(raw, γ * η^α) : raw
        η = max(min(safe, η_max), 2 * eps(R))
        push!(out, η)
    end
    return out
end

function reference_choice1(::Type{R}, norms, linres, η_max, η₀ = R(0.5)) where {R}
    φ = R((1 + sqrt(5)) / 2)
    η = η₀
    out = R[η]
    for k in 1:(length(norms) - 1)
        raw = abs(R(norms[k + 1]) - R(linres[k])) / R(norms[k])
        safe = η^φ > R(0.1) ? max(raw, η^φ) : raw
        η = max(min(safe, η_max), 2 * eps(R))
        push!(out, η)
    end
    return out
end

# The values a solve records for the forcing term: the tolerance of every step, the residual norm
# at the start of the next one, and the linear residual of the step. `norms[k + 1] = ‖F_k‖` and
# `lins[k + 1]` is the linear residual of step `k`, which is what the rule of step `k + 1` reads;
# `lins[1]` is the value a workspace holds before its first step.
function record_forcing(prob, x₀, m, ::Type{R}) where {R}
    solver = init(prob, x₀, m)
    x = copy(solver.x)
    Fx = similar(x)
    prob.F(Fx, x, nothing)
    state = start!!(solver.state, solver.method, prob, x, Fx, nothing, rnorm(Fx))
    ηs = R[]
    norms = R[rnorm(Fx)]
    lins = R[zero(R)]
    for _ in 1:40
        x, Fx, state, info = step!!(state, solver.method, prob, x, Fx, nothing)
        push!(ηs, R(state.ws.last_eta))
        push!(norms, R(info.fnorm))
        push!(lins, R(state.ws.linres))
        # a step's `code` is the outcome of the globalization, not the solve's: the loop ends
        # where the solve's residual test does
        (info.code == SUCCESS && info.fnorm > sqrt(eps(R)) * norms[1]) || break
    end
    return ηs, norms, lins
end

@testset "Check 2: the forcing terms, $T" for T in REAL_ELTYPES
    R = real(T)
    # a norm sequence with a real drop, so that the raw ratio falls below the safeguard twice
    norms = R[10, 9.8, 3, 0.2, 0.05]

    for η_max in (R(0.9), R(0.8))
        f = EisenstatWalkerChoice2(R; η_max = η_max)
        reference = reference_choice2(R, norms, η_max)
        η = f.η₀
        @test η == reference[1]
        for k in 1:(length(norms) - 1)
            η = GeometricSolvers.compute_forcing(
                f, k, norms[k + 1], norms[k], η, zero(R))
            # MU2: the safeguard and the cap are in the formula, within four units of round-off
            @test abs(η - reference[k + 1]) <= 4 * eps(R)
        end
        @test η <= η_max
    end

    # the cap changes the sequence: the raw ratio of the third step is above both caps
    @test reference_choice2(R, norms, R(0.8)) != reference_choice2(R, norms, R(0.9))

    # Choice 1 reads the linear residual of the last step
    linres = R[0, 0.3, 2.9, 0.19, 0.04]
    f1 = EisenstatWalkerChoice1(R)
    reference1 = reference_choice1(R, norms, linres, f1.η_max)
    η = f1.η₀
    for k in 1:(length(norms) - 1)
        η = GeometricSolvers.compute_forcing(f1, k, norms[k + 1], norms[k], η, linres[k])
        @test abs(η - reference1[k + 1]) <= 4 * eps(R)
    end

    # the floor: a raw ratio of zero is floored at 2 eps(R), never zero. The safeguard does not
    # reach the floor here, so `η_prev` is zero: `γ η_prev^α = 0` stays below 0.1
    @test GeometricSolvers.compute_forcing(
        EisenstatWalkerChoice2(R), 1, zero(R), one(R), zero(R), zero(R)) == 2 * eps(R)
    @test GeometricSolvers.compute_forcing(
        FixedForcing(zero(R)), 3, one(R), one(R), one(R),
        zero(R)) == 2 * eps(R)
end

@testset "Check 2: the solver's recorded η sequence, $T" for T in REAL_ELTYPES
    R = real(T)
    # a nonlinear problem, whose Newton steps are more than one: an affine one is solved in a
    # single step, and a sequence of length one says nothing about the rule
    prob = NonlinearProblem((r, x, q) -> (@. r = x^3 + 2x - 3; r))
    f2 = EisenstatWalkerChoice2(R)
    ηs, norms, _ = record_forcing(prob, zeros(T, 8),
        Newton(; linsolve = GMRES(; maxiter = 20, forcing = f2)), R)
    steps = length(ηs)
    @test steps >= 2
    # the recorded η_k of step k is the rule on the recorded norms of steps k and k − 1
    reference2 = reference_choice2(R, norms[1:steps], f2.η_max)
    for k in 1:steps
        @test abs(ηs[k] - reference2[k]) <= 4 * eps(R)
    end

    # Choice 1 reads the linear residual of the last step, which only an inexact inner solve makes
    # a visible fraction of the residual norm: two inner iterations on four unknowns stop short of
    # the tolerance, so the raw term of the rule decides the sequence rather than its safeguard
    f1 = EisenstatWalkerChoice1(R; η₀ = R(0.1))
    ηs1, norms1, lins1 = record_forcing(r5_coupled(T), T.(collect(R5_COUPLED_X)),
        Newton(; linsolve = GMRES(; maxiter = 2, forcing = f1)), R)
    steps1 = length(ηs1)
    @test steps1 >= 2
    # `lins1[k + 1]` is the linear residual of step `k`, which is the value the rule of step
    # `k + 1` reads: the recorded list starts with the value a solve has before its first step
    reference1 = reference_choice1(R, norms1[1:steps1], lins1[2:end], f1.η_max, f1.η₀)
    for k in 1:steps1
        @test abs(ηs1[k] - reference1[k]) <= 4 * eps(R)
    end
    # the linear residuals decide the sequence: a solver that reads none gives another one
    @test reference_choice1(R, norms1[1:steps1], zeros(R, steps1 + 1), f1.η_max,
        f1.η₀) != reference1
end

# ─── The inexact backtracking test (Check 3) ─────────────────────────────────

@testset "Check 3: inexact backtracking, $T" for T in REAL_ELTYPES
    R = real(T)
    prob = r5_full_step_fails()
    x₀ = [T(R5_FAIL_START)]
    solver = init(prob, x₀,
        Newton(; linsolve = GMRES(; maxiter = 4), globalization = Backtracking()))
    # `init` adapts the method to the iterate's real type, and a `step!!` reads the adapted one
    m = solver.method
    x = copy(x₀)
    Fx = similar(x)
    prob.F(Fx, x, nothing)
    fnorm₀ = rnorm(Fx)
    state = start!!(solver.state, m, prob, x, Fx, nothing, fnorm₀)

    # one step, with the iterate and the residual it starts from kept for the readings below
    x_prev = copy(x)
    Fx_prev = copy(Fx)
    _, Fx₁, state₁, info = step!!(state, m, prob, x, Fx, nothing)
    d = copy(state₁.ws.d)
    c₁ = m.globalization.c₁
    η_ach = state₁.ws.linres / fnorm₀
    α = info.stepnorm / rnorm(d)
    @test info.code == SUCCESS
    @test zero(R) < α <= one(R)

    # MU3: the step kind is `InexactStep(η_ach)`, not `ExactStep()`: the accepted step obeys
    # ‖F(x + s)‖ ≤ [1 - c₁(1 - η)] ‖F(x)‖ with η ← 1 - α(1 - η) at the trial step α
    ηα = abs(1 - α) + α * η_ach
    @test R(rnorm(Fx₁)) <= (1 - c₁ * (1 - ηα)) * fnorm₀ + 4 * eps(R) * fnorm₀
    # and the trial at α = 1 fails that test here, so the search backtracked
    x_full = x_prev .+ d
    F_full = similar(x_full)
    prob.F(F_full, x_full, nothing)
    @test rnorm(F_full) > (1 - c₁ * (1 - η_ach)) * fnorm₀
    @test α < one(R)

    # the search itself on the line of that step, for a range of inexact tolerances
    φ₀ = fnorm₀^2
    lf = LineFunction(prob, x_prev, Fx_prev, d, nothing, state.jvpprep,
        similar(x_prev), similar(x_prev), similar(x_prev))
    for η in (zero(R), R(1e-4), R(0.25), R(0.5), R(0.75))
        res = GeometricSolvers.linesearch(m.globalization, lf, InexactStep(η), φ₀, one(R))
        @test res.code == SUCCESS
        ηα_k = abs(1 - res.α) + res.α * η
        @test sqrt(res.φ) <= (1 - c₁ * (1 - ηα_k)) * sqrt(φ₀) + 4 * eps(R) * sqrt(φ₀)
    end

    # the solve converges from a start whose full step ascends
    x_s, st = solve(prob, x₀, m; max_iterations = 50)
    @test st.code == SUCCESS
    @test abs(x_s[1]) <= 100 * sqrt(eps(R))
end

@testset "Check 3: the inexact step kind is what accepts the full step, $T" for T in REAL_ELTYPES
    R = real(T)
    # `F = A x - b` with `A = diag(1, 10)` and `b = 𝟙`: one GMRES iteration leaves the relative
    # linear residual `0.633`, which is above the bound `1 - c₁ = 0.5` that the exact (Armijo) test
    # gives and below the inexact bound `1 - c₁ (1 - η_ach) = 0.817`. The full step is accepted
    # under `InexactStep(η)` and rejected under `ExactStep()`, so the accepted iterate is the full
    # step's.
    A = T[1 0; 0 10]
    b = T[1, 1]
    prob = NonlinearProblem((r, x, q) -> begin
        mul!(r, A, x)
        r .-= b
        r
    end)
    solver = init(prob, zeros(T, 2),
        Newton(;
            linsolve = GMRES(; maxiter = 1, forcing = FixedForcing(R(0.9))),
            globalization = Backtracking(; c₁ = 0.5)))
    m = solver.method
    x = copy(solver.x)
    Fx = similar(x)
    prob.F(Fx, x, nothing)
    fnorm₀ = rnorm(Fx)
    state = start!!(solver.state, m, prob, x, Fx, nothing, fnorm₀)
    _, Fx₁, state₁, info = step!!(state, m, prob, x, Fx, nothing)
    η_ach = state₁.ws.linres / fnorm₀
    @test info.code == SUCCESS
    @test R(0.6) < η_ach < R(0.7)
    # the exact test rejects this step, and the accepted iterate is the one of the full step
    @test R(rnorm(Fx₁)) > (1 - m.globalization.c₁) * fnorm₀
    @test abs(R(rnorm(Fx₁)) - η_ach * fnorm₀) <= 4 * eps(R) * fnorm₀
end

@testset "Check 3: the attained tolerance bounds the step, $T" for T in REAL_ELTYPES
    R = real(T)
    prob = r5_atan()
    x₀ = [T(R5_ATAN_START)]
    m = Newton(;
        linsolve = GMRES(; maxiter = 4, forcing = FixedForcing(R(0.9))),
        globalization = Backtracking(; c₁ = R(0.5)))
    solver = init(prob, x₀, m)
    mm = solver.method
    x = copy(x₀)
    Fx = similar(x)
    prob.F(Fx, x, nothing)
    fnorm₀ = rnorm(Fx)
    state = start!!(solver.state, mm, prob, x, Fx, nothing, fnorm₀)
    _, Fx₁, state₁, info = step!!(state, mm, prob, x, Fx, nothing)
    d = copy(state₁.ws.d)
    c₁ = mm.globalization.c₁
    η_ach = state₁.ws.linres / fnorm₀
    α = info.stepnorm / rnorm(d)
    @test info.code == SUCCESS
    # a one-dimensional solve with one product is exact, so the attained tolerance is zero
    @test η_ach <= 4 * eps(R)
    # the full step decreases the residual, and not by enough for the attained tolerance: the
    # ratio lies above `1 - c₁ (1 - η_ach) = 0.5` and below `1 - c₁ (1 - η_k) = 0.95`
    x_full = x₀ .+ d
    F_full = similar(x)
    prob.F(F_full, x_full, nothing)
    ratio = rnorm(F_full) / fnorm₀
    @test ratio > 1 - c₁ * (1 - η_ach)
    @test ratio <= 1 - c₁ * (1 - mm.linsolve.forcing.η)
    # so the accepted step is a backtracked one, and it obeys the bound of the attained tolerance
    @test zero(R) < α < one(R)
    ηα = abs(1 - α) + α * η_ach
    @test R(rnorm(Fx₁)) <= (1 - c₁ * (1 - ηα)) * fnorm₀ + 4 * eps(R) * fnorm₀
end

# ─── The inner cap (Check 4) ─────────────────────────────────────────────────

@testset "Check 4: the inner cap and its recording, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem((r, x, q) -> begin
        mul!(r, R5_A4, x)
        r .-= R5_B4
        r
    end)

    # MU4: a cap of one inner iteration cannot meet a tolerance of 1e-12, and the step records it
    _, st = solve(prob, zeros(T, 4),
        Newton(; linsolve = GMRES(; maxiter = 1, forcing = FixedForcing(1e-12)));
        max_iterations = 2)
    @test st.step_failures >= 1

    for maxiter in (1, 2, 3, 8)
        solver = init(prob, zeros(T, 4), Newton(; linsolve = GMRES(; maxiter = maxiter)))
        _, st = solve!(solver, zeros(T, 4))
        @test max_inner_iterations(solver) <= maxiter
        @test max_inner_iterations(solver) == max_inner_iterations(solver.state)
    end
end

@testset "Check 4: a restarted cycle forms its own residual, $T" for T in REAL_ELTYPES
    R = real(T)
    # a spectrum of three tight pairs: three inner iterations do not meet the tolerance, so a
    # second cycle starts from the residual the first one reached, and the solve converges only
    # where that residual is the true one
    A = T.(Matrix(Diagonal([1.0, 1.05, 2.0, 2.05, 3.0, 3.05])))
    b = ones(T, 6)
    tol = R(100) * eps(R)
    ws = init_gmres_workspace(GMRES(; maxiter = 20, restart = 3), zeros(T, 6))
    op = (Jv, v) -> (mul!(Jv, A, v); Jv)
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(20), Int32(3))
    @test info.converged
    @test info.iterations > 3
    @test rnorm(A * d - b) <= 10 * tol * rnorm(b)
end

# ─── The right preconditioner (Check 5) ──────────────────────────────────────

# A counting right preconditioner for `M = scale · I`: `refresh!` counts its calls, and
# `apply_preconditioner!` divides by `scale`.
struct ScaledRightPreconditioner{R}
    count::Base.RefValue{Int32}
    scale::R
end

function GeometricSolvers.refresh!(P::ScaledRightPreconditioner, prob, x, Fx, p)
    P.count[] += Int32(1)
    return nothing
end

function GeometricSolvers.apply_preconditioner!(z, P::ScaledRightPreconditioner, v)
    z .= v ./ P.scale
    return z
end

@testset "Check 5: the right preconditioner refreshes once per Newton step, $T" for T in REAL_ELTYPES
    R = real(T)
    # a nonlinear problem of more than one Newton step: an affine one is solved in a single step,
    # where a refresh at `init` and a refresh per step count the same
    x₀ = T.(collect(R5_COUPLED_X))
    prob = r5_coupled(T)
    count = Ref(Int32(0))
    P = ScaledRightPreconditioner(count, T(2))
    m = Newton(;
        linsolve = GMRES(; maxiter = 8, forcing = FixedForcing(R(1e-10)), preconditioner = P))
    solver = init(prob, copy(x₀), m)
    x, st = solve!(solver, copy(x₀))
    @test st.code == SUCCESS
    @test st.iterations >= 2
    # MU5: one refresh per Newton step, no more and no fewer
    @test count[] == st.iterations
    # MU6: `d = M⁻¹ u`, so the preconditioned solve reaches the same root as the plain one
    x_ref, st_ref = solve(prob, copy(x₀), Newton(; linsolve = GMRES(; maxiter = 8)))
    @test st_ref.code == SUCCESS
    @test norm(x - x_ref) <= 100 * sqrt(eps(R)) * norm(x_ref)
    # and the preconditioner of a solve with no preconditioner is the identity
    solver_id = init(prob, copy(x₀), Newton(; linsolve = GMRES(; maxiter = 8)))
    x_id, _ = solve!(solver_id, copy(x₀))
    @test norm(x_id - x_ref) == 0
end

# ─── The matrix-free DogLeg (Check 6) ────────────────────────────────────────

# The 2 × 2 problem of Check 6, with an analytic Jacobian for the R3 control and without it for
# the matrix-free solve.
function r5_dogleg_pair()
    F! = (r, x, p) -> begin
        r[1] = x[1]^2 + 0.3 * x[2] - 2.0
        r[2] = x[2]^2 - 0.2 * x[1] - 3.0
        r
    end
    J! = (J, x, p) -> begin
        J[1, 1] = 2 * x[1]
        J[1, 2] = 0.3
        J[2, 1] = -0.2
        J[2, 2] = 2 * x[2]
        J
    end
    return NonlinearProblem(F!, J!), NonlinearProblem(F!)
end

@testset "Check 6: the matrix-free DogLeg, $T" for T in REAL_ELTYPES
    prob_r3, prob_r5 = r5_dogleg_pair()
    x₀ = T[3.5, -2.8]
    x_r3, st_r3 = solve(prob_r3, copy(x₀), Newton(; globalization = DogLeg()))
    m = Newton(;
        linsolve = GMRES(; maxiter = 2, forcing = FixedForcing(1e-14)), globalization = DogLeg())
    x_r5, st_r5 = solve(prob_r5, copy(x₀), m)
    @test st_r3.code == SUCCESS
    @test st_r5.code == SUCCESS
    # MU7: the Arnoldi Cauchy point gives the iteration count and the root of the factorised loop
    @test st_r5.iterations == st_r3.iterations
    @test norm(x_r5 - x_r3) <= 1e-8 * max(1, norm(x_r3))

    # the same on the 4 × 4 affine system, where a matrix exists but only the product is given
    prob_4 = r5_affine()
    m4 = Newton(;
        linsolve = GMRES(; maxiter = 4, forcing = FixedForcing(1e-14)), globalization = DogLeg())
    x4, st4 = solve(prob_4, zeros(T, 4), m4)
    x4_r3, st4_r3 = solve(prob_4, zeros(T, 4), Newton(; globalization = DogLeg()))
    @test st4.code == SUCCESS && st4_r3.code == SUCCESS
    @test st4.iterations == st4_r3.iterations
    @test norm(x4 - x4_r3) <= 1e-6 * max(1, norm(x4_r3))

    # MU12: a preconditioner is refused with a DogLeg, and a restarted GMRES with it
    @test_throws ArgumentError Newton(;
        linsolve = GMRES(; maxiter = 4, preconditioner = I), globalization = DogLeg())
    @test_throws ArgumentError Newton(;
        linsolve = GMRES(; maxiter = 4, restart = 2), globalization = DogLeg())
    # and the nearest configurations that are not refused
    @test Newton(; linsolve = GMRES(; maxiter = 4, preconditioner = I)) isa Newton
    @test Newton(; linsolve = GMRES(; maxiter = 4, restart = 2),
        globalization = Backtracking()) isa Newton
end

# The Cauchy point of the matrix-free loop is `V_k c_g / ‖c_g‖` scaled to the radius, as the one of
# the factorised loop is `ĝ = g̃ / ‖g̃‖` scaled to it. A weight vector divided by `‖Jᴴ F‖ / ‖F‖`
# instead makes each step of the two Cauchy branches `‖F‖` times too long, so a radius that binds
# gives a different step and the two solves part. The first start below puts the Cauchy point
# outside the radius, the second puts it between the Cauchy point and the Newton step.
@testset "Check 6: the matrix-free DogLeg at a binding radius, $T" for T in REAL_ELTYPES
    R = real(T)
    prob_r3, prob_r5 = r5_dogleg_pair()
    for (x₀, radius) in ((T[3.5, -2.8], R(1e-3)), (T[0.1, -0.1], R(10)))
        m = Newton(;
            linsolve = GMRES(; maxiter = 2, forcing = FixedForcing(1e-14)),
            globalization = DogLeg(; radius = radius))
        x_r5, st_r5 = solve(prob_r5, copy(x₀), m)
        x_r3, st_r3 = solve(prob_r3, copy(x₀),
            Newton(; globalization = DogLeg(; radius = radius)))
        @test st_r5.code == SUCCESS
        @test st_r5.iterations == st_r3.iterations
        @test norm(x_r5 - x_r3) <= 8 * eps(R) * max(1, norm(x_r3))
    end
end

@testset "Check 6: a restarted GMRES solves with a line search, $T" for T in REAL_ELTYPES
    # `restart < maxiter` runs the restart cycle: its residual is formed with one more product
    prob = NonlinearProblem((r, x, q) -> begin
        r[1] = 4 * x[1] + x[2] - 1
        r[2] = x[1] + 3 * x[2] - 2
        r[3] = 2 * x[3] + x[4] - 3
        r[4] = x[3] + 2 * x[4] - 4
        r
    end)
    solver = init(prob, zeros(T, 4),
        Newton(; linsolve = GMRES(; maxiter = 8, restart = 2), globalization = Backtracking()))
    _, st = solve!(solver, zeros(T, 4))
    @test st.code == SUCCESS
    @test max_inner_iterations(solver) <= 8
    @test st.jvp_calls >= st.iterations
end

@testset "Newton with GMRES refuses what needs a factor, and takes the forcing keywords" begin
    # a GMRES step has no factor to reuse, to refine against or to damp
    @test_throws ArgumentError Newton(; linsolve = GMRES(), reuse = 2)
    @test_throws ArgumentError Newton(; linsolve = GMRES(), refinement = Refinement())
    @test_throws ArgumentError Newton(;
        linsolve = GMRES(), globalization = LevenbergMarquardt())
    # the keywords configure a GMRES, and no other linear method takes them
    @test_throws ArgumentError Newton(; forcing = FixedForcing(0.1))
    @test_throws ArgumentError Newton(; preconditioner = I)
    m = Newton(;
        linsolve = GMRES(; maxiter = 5), forcing = FixedForcing(0.1), preconditioner = I)
    @test m.linsolve.maxiter == 5
    @test m.linsolve.forcing === FixedForcing(0.1)
    @test m.linsolve.preconditioner === I
end

# ─── The decided edges ───────────────────────────────────────────────────────

@testset "E1: a non-finite residual or product, $T" for T in REAL_ELTYPES
    # a non-finite residual at x₀ ends the solve at 0 iterations with NONFINITE
    prob_nan = NonlinearProblem((r, x, q) -> begin
        @. r = x^3 + 2x - 3
        r[1] = T(NaN)
        r
    end)
    _, st_nan = solve(prob_nan, fill(T(2), 4), Newton(; linsolve = GMRES()))
    @test st_nan.code == NONFINITE
    @test st_nan.iterations == 0

    # MU9: a non-finite product ends the step with NONFINITE and the solve keeps a finite iterate
    counts = R5Counts(0, 0)
    prob = r5_problem(counts)
    bad = JVPFunction((Jv, x, v, q) -> (Jv .= T(NaN); Jv))
    x, st = solve(prob, fill(T(2), 8), Newton(; linsolve = GMRES(), jacobian = bad))
    @test st.code == NONFINITE
    @test all(isfinite, x)

    # a non-finite Hessenberg entry, which a product that overflows gives, ends the same way
    huge = JVPFunction((Jv, x, v, q) -> (Jv .= T(Inf) .* v .+ T(NaN); Jv))
    x_h, st_h = solve(prob, fill(T(2), 8), Newton(; linsolve = GMRES(), jacobian = huge))
    @test st_h.code == NONFINITE
    @test all(isfinite, x_h)

    # the same product under a `DogLeg`, whose Cauchy point reads the Arnoldi output of the inner
    # solve: a non-finite direction ends the step with NONFINITE, not with a stalled trial loop
    dog = Newton(;
        linsolve = GMRES(; maxiter = 8, forcing = FixedForcing(1e-8)),
        globalization = DogLeg(), jacobian = bad)
    x_d, st_d = solve(prob, fill(T(2), 8), dog)
    @test st_d.code == NONFINITE
    @test all(isfinite, x_d)
end

@testset "E2: a zero residual at x₀, $T" for T in REAL_ELTYPES
    solver = init(r5_exact_root(), T.(R5_EXACT_X), Newton(; linsolve = GMRES()))
    x, st = solve!(solver, T.(R5_EXACT_X))
    @test st.code == SUCCESS
    @test st.iterations == 0
    @test st.f_calls == 1
    @test st.jvp_calls == 0
    @test max_inner_iterations(solver) == 0
    @test x == T.(R5_EXACT_X)
end

@testset "E3: a zero ‖F_{k−1}‖ in the forcing term, $T" for T in REAL_ELTYPES
    R = real(T)
    for forcing in (EisenstatWalkerChoice2(R), EisenstatWalkerChoice1(R))
        η = GeometricSolvers.compute_forcing(
            forcing, 1, zero(R), zero(R), R(0.5), zero(R))
        @test η == forcing.η_max
        @test !isnan(η)
        η2 = GeometricSolvers.compute_forcing(
            forcing, 1, one(R), zero(R), R(0.5), zero(R))
        @test η2 == forcing.η_max
        # a non-finite residual gives the cap too, never a NaN
        η3 = GeometricSolvers.compute_forcing(
            forcing, 1, R(NaN), one(R), R(0.5), zero(R))
        @test η3 == forcing.η_max
    end
    # the first step is η₀ and reads none of them
    @test GeometricSolvers.compute_forcing(
        EisenstatWalkerChoice2(R), 0, zero(R), zero(R), R(NaN), zero(R)) == R(0.5)
    # the floor applies to the initial tolerance too: `η₀ = 0` would ask an inner solve for a zero
    # residual, so it is `2 ε(R)` like every other value of the rule
    for forcing in (EisenstatWalkerChoice2(R; η₀ = zero(R)),
        EisenstatWalkerChoice1(R; η₀ = zero(R)))
        @test GeometricSolvers.compute_forcing(forcing, 0, one(R), one(R), zero(R),
            zero(R)) == 2 * eps(R)
    end
end

@testset "E4: a happy breakdown, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem((r, x, q) -> (r .= 2 .* x .- 1; r))
    solver = init(prob, zeros(T, 4), Newton(; linsolve = GMRES()))
    _, st = solve!(solver, zeros(T, 4))
    # MU11: `H[2, 1] == 0` stops the Arnoldi loop at one step, and no cap failure is recorded
    @test st.code == SUCCESS
    @test st.step_failures == 0
    @test max_inner_iterations(solver) == 1
    @test st.iterations == 1
end

@testset "E5: maxiter above n, $T" for T in REAL_ELTYPES
    prob = r5_affine()
    solver = init(prob, zeros(T, 4), Newton(; linsolve = GMRES(; maxiter = 30)))
    _, st = solve!(solver, zeros(T, 4))
    @test st.code == SUCCESS
    @test max_inner_iterations(solver) <= 4
    # the basis holds min(restart, n) + 1 columns
    @test size(solver.state.ws.V, 2) == 5
end

@testset "E6: n = 1 solves, and a value iterate is refused" begin
    prob1 = NonlinearProblem((r, x, q) -> (r[1] = x[1]^2 - 4; r))
    x1, st1 = solve(prob1, [1.0], Newton(; linsolve = GMRES()))
    @test st1.code == SUCCESS
    @test abs(x1[1] - 2) <= 100 * sqrt(eps(Float64))

    # R5 is array-only: a Number and a StaticArray iterate are refused where the state is built
    @test_throws ArgumentError init(
        NonlinearProblem((x, q) -> x^3 - 1), 2.0, Newton(; linsolve = GMRES()))
    @test_throws ArgumentError init(
        NonlinearProblem((x, q) -> x .^ 3 .- 1), SVector(2.0, 2.0),
        Newton(; linsolve = GMRES()))
    # a least-squares problem has no square system
    @test_throws ArgumentError init(LeastSquaresProblem((r, x, q) -> (r .= x .- 1), 2),
        zeros(2), Newton(; linsolve = GMRES()))
end

@testset "E7: the element types of the product" begin
    # a holomorphic residual in a complex iterate: MGS with the conjugated inner product
    prob = NonlinearProblem((r, x, q) -> (r .= x .^ 3 .- 1; r))
    jvp = JVPFunction((Jv, x, v, q) -> (Jv .= 3 .* x .^ 2 .* v; Jv))
    x₀ = fill(ComplexF64(0.6, 0.9), 4)
    x, st = solve(prob, copy(x₀), Newton(; linsolve = GMRES(; maxiter = 10), jacobian = jvp))
    @test st.code == SUCCESS
    @test norm(x .^ 3 .- 1) <= 100 * sqrt(eps(Float64))

    # `jvp_eltype` evaluates the product in the narrower type and promotes it to T
    seen = Ref{DataType}(Float64)
    jvp32! = (Jv, x, v, q) -> begin
        seen[] = eltype(Jv)
        @. Jv = (3x^2 + 2) * v
        Jv
    end
    counts = R5Counts(0, 0)
    m = Newton(;
        linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(1e-8), jvp_eltype = Float32),
        jacobian = JVPFunction(jvp32!))
    solver = init(r5_problem(counts), fill(2.0, 8), m)
    x32, st32 = solve!(solver, fill(2.0, 8))
    @test seen[] === Float32
    @test st32.code == SUCCESS
    @test norm(x32 .^ 3 .+ 2 .* x32 .- 3) <= 1e-6

    # the default preparation runs `F` on duals of `jvp_eltype`, not of the iterate's element
    # type: the preparation is built from a `TJ` iterate and a `TJ` residual buffer
    duals = Ref{DataType}(Float64)
    prob_dual = NonlinearProblem((r, x, q) -> begin
        eltype(x) <: ForwardDiff.Dual && (duals[] = ForwardDiff.valtype(eltype(x)))
        @. r = x^3 + 2x - 3
        r
    end)
    solver_dual = init(prob_dual, fill(2.0, 8),
        Newton(;
            linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(1e-8),
            jvp_eltype = Float32)))
    @test solver_dual.state.jvpprep.prep isa ChunkedForwardDiff{1}
    x_dual, st_dual = solve!(solver_dual, fill(2.0, 8))
    @test duals[] === Float32
    @test st_dual.code == SUCCESS
    @test norm(x_dual .^ 3 .+ 2 .* x_dual .- 3) <= 1e-6

    # and a DifferentiationInterface back end prepares the pushforward it is called on
    solver_di = init(r5_coupled(Float64), Float64[0.3, 1.1, -0.7, 2.0],
        Newton(; linsolve = GMRES(; maxiter = 10, jvp_eltype = Float32),
            jacobian = AutoFiniteDiff()))
    x_di, st_di = solve!(solver_di, Float64[0.3, 1.1, -0.7, 2.0])
    @test st_di.code == SUCCESS
    @test norm(x_di .^ 3 .+ 2 .* x_di .- 3 .+
               0.5 .* (circshift(x_di, -1) .- x_di)) <= 1e-3
end

@testset "E8: a device-compatible array under allowscalar(false)" begin
    for T in REAL_ELTYPES
        counts = R5Counts(0, 0)
        prob = r5_problem(counts)
        solver = init(prob, JLArray(zeros(T, 4)),
            Newton(; linsolve = GMRES(; maxiter = 10)))
        x, st = solve!(solver, JLArray(zeros(T, 4)))
        @test st.code == SUCCESS
        @test st.f_calls == counts.f
        @test st.jvp_calls == counts.jvp
        @test x isa JLArray

        # the cap of Check 4 on the same array type. The 4 × 4 matrix of Check 4 is a host matrix,
        # and its `mul!` would index the device array one entry at a time, so the coupling here is
        # a rank-one term that a reduction gives: `J = diag(3x² + 2) + 0.1 · 𝟙𝟙ᵀ`. The start has
        # distinct entries: at `zeros` the residual is `-3 · 𝟙` and `𝟙` is an eigenvector of `J`,
        # so one Arnoldi step breaks down and the cap is never reached
        coupling = T(0.1)
        prob_4 = NonlinearProblem((r, x, q) -> begin
            s = sum(x)
            @. r = x^3 + 2x - 3 + coupling * s
            r
        end)
        x₀cap = JLArray(T[1, 2, 3, 4])
        solver_cap = init(prob_4, copy(x₀cap),
            Newton(; linsolve = GMRES(; maxiter = 1, forcing = FixedForcing(1e-12))))
        _, st_cap = solve!(solver_cap, copy(x₀cap))
        @test st_cap.step_failures >= 1
        @test max_inner_iterations(solver_cap) <= 1
    end
end

@testset "The KA CPU() backend has no R5 kernel path" begin
    # An R5 state holds the Krylov workspace, which is mutable: it is not `isbits` and so not a
    # kernel argument, and R5 is array-only, so the kernel entry of R1 and R2 does not reach it.
    prob = r5_affine()
    state = init_state(Newton(; linsolve = GMRES()), prob, zeros(4), nothing)
    @test !isbits(state)
    @test state.ws isa GeometricSolvers.GMRESWorkspace

    # the one host kernel entry refuses the method, and the batch keeps its iterates. The refusal
    # is the iterate test of the R5 state, which sees the batch's value iterate: without it the
    # state is built from that iterate and the product preparation raises a `MethodError`.
    X = [SVector(2.0, 2.0) for _ in 1:4]
    X₀ = copy(X)
    S = Vector{SolverStatus{Float64}}(undef, 4)
    prob_value = NonlinearProblem((x, q) -> x .^ 3 .- 1)
    @test_throws ArgumentError solve_batch!(X, S, prob_value, Newton(; linsolve = GMRES()))
    @test_throws ArgumentError solve_batch!(X, prob_value, Newton(; linsolve = GMRES()))
    @test X == X₀
end

# ─── The values that `start!!` sets ─────────────────────────────────────────

@testset "A step on a state that no start!! saw fails closed" begin
    # the values that `start!!` sets for the steps to read are `NaN` at `init_state` and again on
    # the state `solve!!` returns, so a step reads no state of an earlier solve
    prob = r5_coupled(Float64)
    x₀ = Float64[0.3, 1.1, -0.7, 2.0]
    solver = init(prob, copy(x₀), Newton(; linsolve = GMRES(; maxiter = 8)))
    @test !isfinite(solver.state.ws.prev_fnorm)
    x, st = solve!(solver, copy(x₀))
    @test st.code == SUCCESS
    @test !isfinite(solver.state.ws.prev_fnorm)
    @test !isfinite(solver.state.ws.last_eta)

    Fx = similar(x)
    prob.F(Fx, x, nothing)
    _, _, _, info = step!!(solver.state, solver.method, prob, copy(x), Fx, nothing)
    @test info.code == NONFINITE

    # and the same on the state `init` returned, before any `start!!`
    solver₀ = init(prob, copy(x₀), Newton(; linsolve = GMRES(; maxiter = 8)))
    x1 = copy(solver₀.x)
    Fx1 = similar(x1)
    prob.F(Fx1, x1, nothing)
    _, _, _, info₀ = step!!(solver₀.state, solver₀.method, prob, x1, Fx1, nothing)
    @test info₀.code == NONFINITE
end

# ─── The allocations of the inner solve ──────────────────────────────────────

# The bytes a whole `solve!` allocates, through a function barrier: the solve runs once before the
# measurement, and the copy that resets the iterate is inside the measured block because `copyto!`
# allocates nothing while `copy` would be measured too.
function solve_allocations(solver, x₀, work)
    copyto!(work, x₀)
    solve!(solver, work)
    return @allocated begin
        copyto!(work, x₀)
        solve!(solver, work)
    end
end

@testset "The inner solve allocates nothing after init on Array, $T" for T in REAL_ELTYPES
    Base.JLOptions().check_bounds == 1 && continue
    prob = r5_affine()
    m = Newton(; linsolve = GMRES(; maxiter = 4))
    solver = init(prob, zeros(T, 4), m)
    ws = solver.state.ws
    op = GeometricSolvers.R5Operator(solver.state.jvpprep, prob, zeros(T, 4), nothing)
    b = similar(zeros(T, 4))
    map!(-, b, R5_B4)
    β = rnorm(b)
    @test gmres_allocations(ws, op, b, β, T(1e-10), Int32(4), Int32(4)) == 0
    # the control: the helper sees an allocation
    @test (@allocated similar(b)) > 0
end

@testset "The whole solve allocates nothing after init on Array, $T" for T in REAL_ELTYPES
    Base.JLOptions().check_bounds == 1 && continue
    R = real(T)
    x₀ = T.(collect(R5_COUPLED_X))
    for m in (Newton(; linsolve = GMRES(; maxiter = 8)),
        Newton(; linsolve = GMRES(; maxiter = 8, forcing = FixedForcing(R(1e-10))),
        globalization = Backtracking()),
        Newton(; linsolve = GMRES(; maxiter = 8, forcing = FixedForcing(R(1e-10))),
        globalization = DogLeg()))
        solver = init(r5_coupled(T), copy(x₀), m)
        @test solve_allocations(solver, x₀, similar(x₀)) == 0
    end
end
