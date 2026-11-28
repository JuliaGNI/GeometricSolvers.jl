# R5 matrix-free Newton–GMRES: the forcing terms, the right-preconditioner slot, the inexact
# backtracking test, the inner-iteration cap, the matrix-free `DogLeg`, the Arnoldi
# orthogonalisation, and the decided edges.
#
# The solves run on `Array` and on `JLArray` (a CPU array that refuses scalar indexing, so it
# catches a GPU-only fault in ordinary CI) in `REAL_ELTYPES`, and the matrix-free `DogLeg` is
# compared with the factorised one, which is the R3 control of the same problem.

using ADTypes: AutoFiniteDiff
using Adapt: Adapt
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: GPUArraysCore, AnyGPUArray, allowscalar
using JLArrays: JLArrays, JLArray
using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, I, Transpose, mul!, norm, opnorm, qr
using Random: Xoshiro
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, ChunkedForwardDiff, DIJacobian, DogLeg,
                        EisenstatWalkerChoice1, EisenstatWalkerChoice2, FixedForcing, GMRES,
                        GMRESRefinement, InexactStep, JVPFunction, LevenbergMarquardt,
                        LineFunction, LUFactorization,
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

# `‖F(x) + J(x) d‖` of a direction `d`, with the product from ForwardDiff on `F` itself and not from
# the solver: the attained tolerance of a step, whatever the workspace records after it.
function direction_linres(prob, x, d)
    F_at(t) = (y = similar(x, typeof(t)); prob.F(y, x .+ t .* d, nothing); y)
    return rnorm(F_at(zero(eltype(x))) .+ ForwardDiff.derivative(F_at, zero(eltype(x))))
end

# ─── The allocation row of the inner solve ──────────────────────────────────

# An `@allocated` assertion measures through a function barrier whose arguments have concrete types
# and which calls the function once before it measures.
function gmres_allocations(ws, op, b, β, tol, maxiter, restart; mgs = false)
    gmres_solve!(ws, op, nothing, b, β, tol, maxiter, restart; mgs)
    return @allocated gmres_solve!(ws, op, nothing, b, β, tol, maxiter, restart; mgs)
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
        @test sizeof(SolverStatus{Float32}) == 44
        @test sizeof(SolverStatus{Float64}) == 72
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

# Choice 2 and Choice 1 written out as Eisenstat and Walker (1996) state them, with the safeguard,
# the cap and the floor of the docstrings, for the tests to compare against.
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
    η_ach = direction_linres(prob, x_prev, d) / fnorm₀
    α = info.stepnorm / rnorm(d)
    @test info.code == SUCCESS
    @test zero(R) < α <= one(R)
    # the workspace records the linear residual of the accepted step `α d` for Choice 1, and a
    # one-dimensional solve is exact, so that residual is `(1 - α) ‖F‖`
    @test abs(state₁.ws.linres - (1 - α) * fnorm₀) <= 16 * eps(R) * fnorm₀

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
    η_ach = direction_linres(prob, x₀, d) / fnorm₀
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

# Every segment trial reads the Newton weights `yN`, so the loop keeps them for the whole step. The
# Newton step `(-1, -1000)` is far longer than the Cauchy point, of length about 1, and
# the curvature of `F₂` rejects the segment trials at the radii 100, 25, 6.25 and 1.5625; the
# Cauchy trial at 0.390625 is accepted. A segment point left in `yN` lies on the segment itself,
# so it gives the same trial points up to rounding, and only `V_k yN` tells it from the Newton
# direction.
@testset "Check 6: consecutive segment trials keep the Newton weights, $T" for T in REAL_ELTYPES
    R = real(T)
    trials = Vector{Vector{T}}()
    F! = (r, x, p) -> begin
        x isa AbstractVector{T} && push!(trials, copy(x))
        r[1] = x[1] + 1
        r[2] = T(1e-3) * x[2] + x[2]^2 + 1
        r
    end
    J! = (J, x, p) -> begin
        J[1, 1] = 1
        J[1, 2] = 0
        J[2, 1] = 0
        J[2, 2] = T(1e-3) + 2 * x[2]
        J
    end
    x₀ = zeros(T, 2)
    radius = R(100)
    m = Newton(; linsolve = GMRES(; maxiter = 2, forcing = FixedForcing(1e-14)),
        globalization = DogLeg(; radius = radius))
    solver = init(NonlinearProblem(F!), copy(x₀), m; max_iterations = 1)
    empty!(trials)
    x_r5, _ = solve!(solver, copy(x₀))
    trials_r5 = copy(trials)
    solver_r3 = init(NonlinearProblem(F!, J!), copy(x₀),
        Newton(; globalization = DogLeg(; radius = radius)); max_iterations = 1)
    empty!(trials)
    x_r3, _ = solve!(solver_r3, copy(x₀))
    # the residual at `x₀`, then four segment trials of norm δ and the accepted Cauchy trial
    @test length(trials_r5) == length(trials) == 6
    @test norm.(trials_r5) ≈ R[0, 100, 25, 6.25, 1.5625, 0.390625] rtol = sqrt(eps(R))
    for (a, b) in zip(trials_r5, trials)
        @test norm(a - b) <= sqrt(eps(R)) * max(1, norm(b))
    end
    @test x_r5 != x₀
    @test norm(x_r5 - x_r3) <= sqrt(eps(R)) * max(1, norm(x_r3))
    ws = solver.state.ws
    k = Int(ws.arnoldi)
    @test k == 2
    @test norm(ws.V[:, 1:k] * ws.yN[1:k] - ws.d) <= 8 * eps(R) * norm(ws.d)
end

# The model of the matrix-free loop reads the Arnoldi relation of one cycle. With `maxiter > n` and a
# tolerance that `n` iterations miss in floating point, the inner solve would restart, and the loop
# would read the last cycle's basis against the first residual. The inner solve of a `DogLeg` step
# stops at `n` instead, so `maxiter = 10` and `maxiter = n` give the same solve.
@testset "Check 6: the matrix-free DogLeg runs one Arnoldi cycle, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem((r, x, q) -> begin
        r[1] = x[1] - 1
        r[2] = T(1e-10) * x[2] + x[1]^2 - 2
        r
    end)
    x₀ = T[3, 3]
    solve_with(maxiter) = begin
        solver = init(prob, copy(x₀),
            Newton(; linsolve = GMRES(; maxiter = maxiter, forcing = FixedForcing(0.0)),
                globalization = DogLeg()))
        x, st = solve!(solver, copy(x₀))
        return x, st, max_inner_iterations(solver)
    end
    x10, st10, inner10 = solve_with(10)
    x2, st2, _ = solve_with(2)
    @test inner10 <= 2
    @test st10.code == st2.code
    @test st10.iterations == st2.iterations
    @test x10 == x2
    # the factorised loop is the control in Float64; in Float32 the condition number of `J`, about
    # 1e11, is far above 1/eps(Float32), and the matrix-free loop fails there with or without a
    # restart
    if T == Float64
        _, st_r3 = solve(prob, copy(x₀), Newton(; globalization = DogLeg()))
        @test st_r3.code == SUCCESS
        @test st10.code == SUCCESS
        @test st10.iterations == st_r3.iterations
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

# ─── The Arnoldi orthogonalisation ───────────────────────────────────────────

# The reductions of the Arnoldi process, counted at the array level on `JLArray`: every `sum` and
# `maximum` of a `JLArray` or of a lazy broadcast over one is a `Base.mapreduce` of it, and every
# product of `JLArray` storage is the five-argument `mul!` of `LinearAlgebra`, which the
# three-argument one calls. A product with an adjoint or a transposed matrix, `Vᴴ w`, reduces
# over the length of the basis and counts as a reduction. A product with a plain matrix reduces
# nothing: the counter records its destination, so that a test reads which vector it updated. The
# product methods, one for a vector and one for a matrix destination, are more specific than the
# generic ones of `LinearAlgebra`, which they call; no method of GPUArrays takes these operands at
# five arguments, and the storage-level methods below them differ between GPUArrays releases. The
# file deletes the methods after the tests that read them.
const ARRAY_REDUCTIONS = Ref(0)
const BROADCAST_REDUCTIONS = Ref(0)
const ADJOINT_OPERANDS = Any[]
const PLAIN_PRODUCTS = Any[]
const AnyJL = Union{JLArray, SubArray{<:Any, <:Any, <:JLArray}}
const JLVector = Union{JLArray{<:Any, 1}, SubArray{<:Any, 1, <:JLArray}}
const JLMatrix = Union{JLArray{<:Any, 2}, SubArray{<:Any, 2, <:JLArray}}
const JLVecOrMat = Union{JLVector, JLMatrix}
const JLOperand = Union{
    JLVecOrMat, Adjoint{<:Any, <:JLVecOrMat}, Transpose{<:Any, <:JLVecOrMat}}

function Base.mapreduce(f, op, A::AnyJL; kwargs...)
    ARRAY_REDUCTIONS[] += 1
    return invoke(Base.mapreduce, Tuple{Any, Any, AnyGPUArray}, f, op, A; kwargs...)
end

function Base.mapreduce(f, op, A::Broadcast.Broadcasted{<:JLArrays.JLArrayStyle}; kwargs...)
    ARRAY_REDUCTIONS[] += 1
    BROADCAST_REDUCTIONS[] += 1
    S = Broadcast.Broadcasted{<:GPUArraysCore.AbstractGPUArrayStyle}
    return invoke(Base.mapreduce, Tuple{Any, Any, S}, f, op, A; kwargs...)
end

# Whether the destination `C` of a product is the vector `u`, by its storage.
is_storage_of(C, u) = Base.mightalias(C, u)

function count_product!(C, A)
    if A isa JLVecOrMat
        push!(PLAIN_PRODUCTS, C)
    else
        ARRAY_REDUCTIONS[] += 1
        push!(ADJOINT_OPERANDS, parent(A))
    end
    return nothing
end

function LinearAlgebra.mul!(C::JLVector, A::JLOperand, B::JLVector, α::Number, β::Number)
    count_product!(C, A)
    S = Tuple{AbstractVector, AbstractVecOrMat, AbstractVector, Number, Number}
    return invoke(LinearAlgebra.mul!, S, C, A, B, α, β)
end

function LinearAlgebra.mul!(C::JLMatrix, A::JLOperand, B::JLOperand, α::Number, β::Number)
    count_product!(C, A)
    S = Tuple{AbstractMatrix, AbstractVecOrMat, AbstractVecOrMat, Number, Number}
    return invoke(LinearAlgebra.mul!, S, C, A, B, α, β)
end

# The counters see every product of `JLArray` storage: no method of GPUArrays takes the call
# first, whatever the release.
@testset "The product counters see the products of JLArray storage" begin
    A = JLArray(randn(Xoshiro(3), 6, 4))
    x = JLArray(randn(Xoshiro(4), 6))
    X = JLArray(randn(Xoshiro(5), 6, 2))
    v = JLArray(randn(Xoshiro(6), 4))
    V = JLArray(randn(Xoshiro(7), 4, 2))
    for (C, B) in ((JLArray(zeros(4)), x), (JLArray(zeros(4, 2)), X)),
        At in (A', transpose(A))

        empty!(ADJOINT_OPERANDS)
        reduced = ARRAY_REDUCTIONS[]
        mul!(C, At, B)
        @test ARRAY_REDUCTIONS[] - reduced == 1
        @test only(ADJOINT_OPERANDS) === A
    end
    for (C, B) in ((JLArray(zeros(6)), v), (JLArray(zeros(6, 2)), V))
        empty!(PLAIN_PRODUCTS)
        reduced = ARRAY_REDUCTIONS[]
        mul!(C, A, B)
        @test ARRAY_REDUCTIONS[] == reduced
        @test only(PLAIN_PRODUCTS) === C
    end
end

# The product of a diagonal operator by broadcast, which reduces nothing, and which records the
# reduction count at each call: the difference of two marks is the count of the Arnoldi step
# between two products.
struct MarkingOperator{D}
    d::D
    marks::Vector{Int}
end

function (op::MarkingOperator)(Jv, v)
    push!(op.marks, ARRAY_REDUCTIONS[])
    Jv .= op.d .* v
    return Jv
end

# The reductions of the inner iterations `1:k-1` of one cycle of `n` distinct eigenvalues, on
# `JLArray`, of the path that `mgs` selects.
function cycle_reductions(::Type{T}, n, k; mgs, scale = one(real(T))) where {T}
    R = real(T)
    op = MarkingOperator(JLArray(T.(scale .* (1:n))), Int[])
    b = JLArray(ones(T, n))
    ws = init_gmres_workspace(GMRES(; maxiter = k), JLArray(zeros(T, n)))
    _, info = gmres_solve!(ws, op, nothing, b, rnorm(b), zero(R), Int32(k), Int32(k); mgs)
    return diff(op.marks), info
end

@testset "CGS2 takes two reductions per inner iteration, MGS takes j + 1, $T" for T in REAL_ELTYPES
    counts, info = cycle_reductions(T, 8, 6; mgs = false)
    @test info.iterations == 6
    @test counts == fill(2, 5)
    counts_mgs, info_mgs = cycle_reductions(T, 8, 6; mgs = true)
    @test info_mgs.iterations == 6
    @test counts_mgs == collect(2:6)
end

# Where `s₁ = ‖w₁‖²` underflows below `n floatmin` or overflows, the step norms `w₂` explicitly;
# `rnorm` takes its scaled path there, one reduction for the sum and two more, so the iteration
# counts 2 + 3.
@testset "CGS2 norms w₂ explicitly where s₁ is out of range, $T" for T in REAL_ELTYPES
    R = real(T)
    for scale in (sqrt(floatmin(R)) / 64, 4 * sqrt(floatmax(R)))
        counts, info = cycle_reductions(T, 8, 6; mgs = false, scale)
        @test info.iterations == 6
        @test !info.breakdown
        @test counts == fill(5, 5)
    end
end

# One CGS2 step at `j = 1` on a basis column `v` and a vector `w` set by hand, on `JLArray`: the
# reductions of the step, the entry `H[1, 1] = h₁ + h₂` and the norm it returns.
function cgs2_row(::Type{T}, v, w) where {T}
    n = length(w)
    ws = init_gmres_workspace(GMRES(; maxiter = 2), JLArray(zeros(T, n)))
    V = zeros(T, size(ws.V))
    V[:, 1] = v
    copyto!(ws.V, V)
    copyto!(ws.w, T.(w))
    reduced = ARRAY_REDUCTIONS[]
    hj = GeometricSolvers.cgs2_step!(ws, 1)
    return ARRAY_REDUCTIONS[] - reduced, ws.H[1, 1], hj
end

# The step takes `√(s₁ - ‖h₂‖²)` where `‖h₂‖² ≤ s₁/2` and `s₁ ≥ n floatmin(R)`, and `rnorm(w₂)`
# elsewhere. With `v = 2e₁` and `w = e₁ + y e₂`, the arithmetic is exact: `h₁ = 2`,
# `w₁ = -3e₁ + y e₂`, `h₂ = -6`, `s₁ = 9 + y²` and `w₂ = 9e₁ + y e₂`, so `y = 8` gives
# `‖h₂‖²/s₁ = 36/73 ≈ 0.49` and the norm `√37`, and `y = 7.5` gives `36/65.25 ≈ 0.55` and the
# norm `√137.25` of one more reduction. With `v = e₁` and `w = t e₂`, `s₁ = t²`, and
# `t² = 0.8 n floatmin(R)` takes `rnorm(w₂)` on its scaled path, three reductions, where
# `t² = 1.25 n floatmin(R)` takes the two of Pythagoras.
@testset "CGS2 brackets its thresholds and norms w₂ by Pythagoras, $T" for T in REAL_ELTYPES
    R = real(T)
    n = 8
    e(i) = [k == i ? one(T) : zero(T) for k in 1:n]
    for (y, reductions, norm₂) in ((8, 2, sqrt(R(37))), (7.5, 3, sqrt(R(137.25))))
        c, h, hj = cgs2_row(T, 2 .* e(1), e(1) .+ T(y) .* e(2))
        @test c == reductions
        @test isapprox(h, T(-4); rtol = 8 * eps(R))
        @test isapprox(hj, norm₂; rtol = 8 * eps(R))
    end
    for (f, reductions) in ((0.8, 5), (1.25, 2))
        t = sqrt(R(f) * n * floatmin(R))
        c, h, hj = cgs2_row(T, e(1), t .* e(2))
        @test c == reductions
        @test iszero(h)
        @test isapprox(hj, t; rtol = 8 * eps(R))
    end
    # Both thresholds hold with equality and take Pythagoras. `v = 5e₁` and `w = e₁ + 168e₂`
    # give `h₁ = 5`, `h₂ = -120` and `s₁ = 28800`, so `‖h₂‖² = s₁/2` and the norm is `120`, where
    # `rnorm(w₂) = ‖576e₁ + 168e₂‖` is `600`. With `n = 4` and `t = √(4 floatmin(R))`, a power of
    # two, `s₁ = t² = n floatmin(R)` exactly.
    c, h, hj = cgs2_row(T, 5 .* e(1), e(1) .+ T(168) .* e(2))
    @test c == 2
    @test isapprox(h, T(-115); rtol = 8 * eps(R))
    @test isapprox(hj, R(120); rtol = 8 * eps(R))
    t = sqrt(4 * floatmin(R))
    c, h, hj = cgs2_row(T, T[1, 0, 0, 0], T[0, t, 0, 0])
    @test c == 2
    @test iszero(h)
    @test isapprox(hj, t; rtol = 8 * eps(R))
end

# The keyword of `GMRES` reaches the inner solve of a `Newton`: the same wrapper counts the two
# reductions of CGS2 by default and the `j + 1` of MGS with `mgs = true`.
@testset "The mgs keyword of GMRES selects the path of a Newton step, $T" for T in REAL_ELTYPES
    n = 8
    d = JLArray(T.(1:n))
    for (mgs, expected) in ((false, j -> 2), (true, j -> j + 1))
        marks = Int[]
        jvp! = (Jv, x, v, q) -> (push!(marks, ARRAY_REDUCTIONS[]); Jv .= d .* v; Jv)
        prob = NonlinearProblem((r, x, q) -> (r .= d .* x .- 1; r))
        g = GMRES(; maxiter = n, forcing = FixedForcing(eps(real(T))), mgs)
        @test g.mgs == mgs
        solver = init(prob, JLArray(zeros(T, n)),
            Newton(; linsolve = g, jacobian = JVPFunction(jvp!)); max_iterations = 1)
        solve!(solver, JLArray(zeros(T, n)))
        k = Int(max_inner_iterations(solver))
        @test k >= 5
        @test diff(marks)[1:(k - 1)] == expected.(1:(k - 1))
        # the forcing keyword of `Newton` rebuilds the method, and `adapt` rebuilds it through the
        # positional form: both keep the field
        @test Newton(; linsolve = g, forcing = FixedForcing(0.1)).linsolve.mgs == mgs
        @test Adapt.adapt(Array, g).mgs == mgs
    end
end

# The inner GMRES of a `GMRESRefinement` orthogonalises by MGS whatever `GMRES` defaults to: its
# Arnoldi steps are the conjugated products of `arnoldi_dot`, lazy broadcasts, and no adjoint
# product of its basis. The step forms the gradient `Jᵀ F` by an adjoint product of the Jacobian,
# which is no product of the basis. The linear problem has `κ ≈ 5e7`, above the limit of the
# `Float32` LU-IR stage, so the GMRES-IR stage runs.
@testset "GMRESRefinement keeps MGS" begin
    n = 8
    Q = Matrix(qr(randn(Xoshiro(7), n, n)).Q)
    A = JLArray(Q * Diagonal([5.0e7^(-(i - 1) / (n - 1)) for i in 1:n]) * Q')
    prob = NonlinearProblem((r, x, p) -> (mul!(r, A, x); r), (J, x, p) -> copyto!(J, A))
    m = Newton(; linsolve = LUFactorization(Float32),
        refinement = GMRESRefinement(Float64; gmres_maxiter = 8, gmres_tol = 1e-8))
    x₀ = JLArray(2 .* Q[:, n])
    solver = init(prob, copy(x₀), m)
    BROADCAST_REDUCTIONS[] = 0
    empty!(ADJOINT_OPERANDS)
    _, st = solve!(solver, copy(x₀))
    @test st.code == SUCCESS
    @test st.step_failures >= 1
    @test BROADCAST_REDUCTIONS[] > 0
    @test !isempty(ADJOINT_OPERANDS)
    @test !any(A -> is_storage_of(A, solver.state.refbuf.ws.V), ADJOINT_OPERANDS)
end

# On the CGS2 path the solution update `u += V y` is one product per cycle, into `u` itself, and so
# is the direction `V_k y` that the matrix-free `DogLeg` rebuilds. On the MGS path both are a loop
# over the columns, which takes no product: the direction follows the path of the last solve on
# the workspace.
@testset "The update and the reduced direction follow the path, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    R = real(T)
    n = 8
    A = Diagonal(T.(1:n))
    dJ = JLArray(T.(1:n))
    op = (Jv, v) -> (Jv .= dJ .* v; Jv)
    b = JLArray(ones(T, n))
    for (maxiter, restart) in ((8, 8), (8, 2))
        ws = init_gmres_workspace(GMRES(; maxiter, restart), JLArray(zeros(T, n)))
        empty!(PLAIN_PRODUCTS)
        d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), R(1e-30), Int32(maxiter),
            Int32(restart); mgs)
        cycles = cld(Int(info.iterations), restart)
        @test count(c -> is_storage_of(c, ws.u), PLAIN_PRODUCTS) == (mgs ? 0 : cycles)
        @test restart == maxiter ? cycles == 1 : cycles == 4
        # the direction is the sum of the cycles' products: its true residual is the one the last
        # cycle reports
        @test abs(norm(A * Array(d) - ones(T, n)) - info.resnorm) <= 100 * eps(R) * sqrt(n)
    end
    ws = init_gmres_workspace(GMRES(; maxiter = 4), JLArray(zeros(T, n)))
    gmres_solve!(ws, op, nothing, b, rnorm(b), zero(R), Int32(4), Int32(4); mgs)
    y = T[0.5, -1.25, 2, 0.75]
    s = JLArray(ones(T, n))
    empty!(PLAIN_PRODUCTS)
    reduced = ARRAY_REDUCTIONS[]
    GeometricSolvers.reduced_direction!(s, ws, y, 4)
    @test length(PLAIN_PRODUCTS) == (mgs ? 0 : 1)
    @test mgs || is_storage_of(only(PLAIN_PRODUCTS), s)
    @test ARRAY_REDUCTIONS[] == reduced
    @test norm(Array(s) - Array(ws.V)[:, 1:4] * y) <= 8 * eps(R) * norm(y)
end

for f in (Base.mapreduce, LinearAlgebra.mul!), m in methods(f)

    m.module === (@__MODULE__) && Base.delete_method(m)
end

# A normal matrix with its spectrum log-spaced in `[1, 1e8]`, `A = Q Λ Qᴴ` with a unitary `Q`, and
# a right-hand side, both drawn in `Float64` and rounded to `T`. `Q` is the product of the
# Householder reflections of a QR factorisation of `Z`. The test's own loops run every sum in
# plain sequential order, with no BLAS, LAPACK, `@simd` or fused multiply–add, so the matrix is the
# same on every platform where Base's `exp10` and `abs` give the same result.
function arnoldi_test_matrix(::Type{T}, n, seed) where {T}
    rng = Xoshiro(seed)
    Z = T <: Complex ? complex.(randn(rng, n, n), randn(rng, n, n)) : randn(rng, n, n)
    b = T <: Complex ? complex.(randn(rng, n), randn(rng, n)) : randn(rng, n)
    S = eltype(Z)
    vs = Vector{Vector{S}}(undef, n - 1)
    for k in 1:(n - 1)
        s = 0.0
        for i in k:n
            s += abs2(Z[i, k])
        end
        x₁ = Z[k, k]
        α = -(iszero(x₁) ? one(S) : x₁ / abs(x₁)) * sqrt(s)
        v = Z[k:n, k]
        v[1] -= α
        sv = 0.0
        for i in eachindex(v)
            sv += abs2(v[i])
        end
        nv = sqrt(sv)
        for i in eachindex(v)
            v[i] /= nv
        end
        vs[k] = v
        for c in k:n
            p = zero(S)
            for i in k:n
                p += conj(v[i - k + 1]) * Z[i, c]
            end
            for i in k:n
                Z[i, c] -= 2 * v[i - k + 1] * p
            end
        end
    end
    Q = Matrix{S}(undef, n, n)
    for c in 1:n, i in 1:n

        Q[i, c] = i == c ? one(S) : zero(S)
    end
    for k in (n - 1):-1:1
        v = vs[k]
        for c in k:n
            p = zero(S)
            for i in k:n
                p += conj(v[i - k + 1]) * Q[i, c]
            end
            for i in k:n
                Q[i, c] -= 2 * v[i - k + 1] * p
            end
        end
    end
    λ = exp10.(range(0, 8; length = n))
    A = Matrix{T}(undef, n, n)
    for c in 1:n, i in 1:n

        a = zero(S)
        for l in 1:n
            a += Q[i, l] * λ[l] * conj(Q[c, l])
        end
        A[i, c] = T(a)
    end
    return A, T.(b)
end

# The loss of orthogonality `‖I - VᴴV‖₂` of a basis, with `VᴴV` formed in `BigFloat`, so that the
# rounding of the measurement stays far below the `eps(R)` it measures.
function orthogonality_loss(V)
    Vb = big.(V)
    G = I - Vb' * Vb
    return opnorm(eltype(V) <: Complex ? ComplexF64.(G) : Float64.(G))
end

# The control: the Arnoldi basis of classical Gram–Schmidt with one pass, `w = A v`, `h = Vⱼᴴ w`,
# `w -= Vⱼ h` and `w / ‖w‖`, in `T`. Every sum runs in a plain sequential loop, with no BLAS,
# `@simd` or fused multiply–add, so the control is the same on every platform.
function cgs_one_pass_basis(A, b, m)
    T = eltype(b)
    n = length(b)
    V = zeros(T, n, m + 1)
    w = zeros(T, n)
    h = zeros(T, m)
    s = zero(real(T))
    for i in 1:n
        s += abs2(b[i])
    end
    nb = sqrt(s)
    for i in 1:n
        V[i, 1] = b[i] / nb
    end
    for j in 1:m
        for i in 1:n
            a = zero(T)
            for l in 1:n
                a += A[i, l] * V[l, j]
            end
            w[i] = a
        end
        for i in 1:j
            a = zero(T)
            for l in 1:n
                a += conj(V[l, i]) * w[l]
            end
            h[i] = a
        end
        for l in 1:n
            a = w[l]
            for i in 1:j
                a -= V[l, i] * h[i]
            end
            w[l] = a
        end
        s = zero(real(T))
        for l in 1:n
            s += abs2(w[l])
        end
        nw = sqrt(s)
        for l in 1:n
            V[l, j + 1] = w[l] / nw
        end
    end
    return V
end

# The control's loss in eps(R), seeds 1:5, Julia 1.13.4, the same with and without
# `--check-bounds=yes`:
# Float32 4051.1, 12137.7, 11263.7, 8042.4, 3717.0; Float64 6794.7, 6554.6, 9048.8, 10338.0,
# 7172.4; ComplexF32 2682.8, 2617.4, 7624.4, 5889.7, 4042.2; ComplexF64 9140.6, 5041.1, 4196.4,
# 6581.8, 11350.3.
@testset "The CGS2 basis stays orthogonal where one-pass CGS loses it, $T" for T in ELTYPES
    R = real(T)
    n, m = 200, 30
    for seed in 1:5
        A, b = arnoldi_test_matrix(T, n, seed)
        ws = init_gmres_workspace(GMRES(; maxiter = m), zeros(T, n))
        op = (Jv, v) -> mul!(Jv, A, v)
        _, info = gmres_solve!(ws, op, nothing, b, rnorm(b), zero(R), Int32(m), Int32(m))
        @test info.iterations == m
        @test !info.breakdown
        @test orthogonality_loss(ws.V) <= 32 * eps(R)
        @test orthogonality_loss(cgs_one_pass_basis(A, b, m)) > 1000 * eps(R)
    end
end

# A reference GMRES with modified Gram–Schmidt and no preconditioner on dense storage, transcribed
# from the MGS path of `gmres_solve!` in the same order of operations: the projections, the Givens
# rotations, the back substitution, the update of `u` one column at a time and the residual of a
# restart. It returns the iterate and the inner iteration count.
function mgs_gmres_reference(A, b, tol, maxiter, restart)
    T = eltype(b)
    R = real(T)
    n = length(b)
    mr = min(restart, n)
    V = zeros(T, n, mr + 1)
    H = zeros(T, mr + 1, mr)
    cs = zeros(R, mr)
    sn = zeros(T, mr)
    g = zeros(T, mr + 1)
    y = zeros(T, mr)
    u = zeros(T, n)
    w = zeros(T, n)
    z = zeros(T, n)
    rr = copy(b)
    atol = R(tol) * R(rnorm(b))
    used = 0
    while used < maxiter
        β = R(rnorm(rr))
        β <= atol && break
        @views V[:, 1] .= rr ./ β
        fill!(g, zero(T))
        g[1] = β
        j = 0
        hj = β
        linres = β
        while j < mr && used < maxiter
            j += 1
            used += 1
            @views z .= V[:, j]
            mul!(w, A, z)
            for i in 1:j
                h = GeometricSolvers.arnoldi_dot(@view(V[:, i]), w)
                H[i, j] = h
                @views w .-= h .* V[:, i]
            end
            hj = R(rnorm(w))
            H[j + 1, j] = hj
            for i in 1:(j - 1)
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            end
            a = H[j, j]
            bj = H[j + 1, j]
            radius = R(hypot(abs(a), abs(bj)))
            if iszero(radius)
                cs[j] = one(R)
                sn[j] = zero(T)
            else
                cs[j] = R(abs(a) / radius)
                sn[j] = iszero(abs(a)) ? conj(bj) / T(radius) :
                        (a / T(abs(a))) * conj(bj) / T(radius)
            end
            H[j, j] = cs[j] * a + sn[j] * bj
            H[j + 1, j] = zero(T)
            gj = g[j]
            g[j + 1] = -conj(sn[j]) * gj
            g[j] = cs[j] * gj
            linres = iszero(radius) ? R(hypot(abs(g[j]), abs(g[j + 1]))) : R(abs(g[j + 1]))
            (iszero(hj) || linres <= atol) && break
            @views V[:, j + 1] .= w ./ hj
        end
        for i in j:-1:1
            acc = g[i]
            for k in (i + 1):j
                acc -= H[i, k] * y[k]
            end
            y[i] = iszero(H[i, i]) ? zero(T) : acc / H[i, i]
        end
        for i in 1:j
            @views u .+= y[i] .* V[:, i]
        end
        (linres <= atol || iszero(hj) || used >= maxiter) && break
        z .= u
        mul!(w, A, z)
        rr .= b .- w
    end
    return u, used
end

# The MGS path of the keyword is modified Gram–Schmidt with the update of `u` one column at a
# time, so on the matrices of the test above, restarted every 5 iterations, it gives the iterate of
# the reference within `8 eps(R)` relative, in the 2-norm, and its iteration count.
@testset "The MGS path gives the iterate of the reference MGS GMRES, $T" for T in ELTYPES
    R = real(T)
    n, m, restart = 200, 30, 5
    for seed in 1:3, tol in (zero(R), R(1e-4))

        A, b = arnoldi_test_matrix(T, n, seed)
        ws = init_gmres_workspace(GMRES(; maxiter = m, restart), zeros(T, n))
        op = (Jv, v) -> mul!(Jv, A, v)
        d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(m), Int32(restart);
            mgs = true)
        x, used = mgs_gmres_reference(A, b, tol, m, restart)
        @test info.iterations == used
        @test norm(d - x) <= 8 * eps(R) * norm(x)
    end
end

# A step whose `w` lies within round-off of the span of `V_j`: after one pass, `w₁` is round-off,
# most of which lies in that span again, so `‖h₂‖² > s₁/2` and `s₁ - ‖h₂‖²` keeps no correct
# digit. The step norms `w₂` explicitly there, and the last basis vector has unit norm. The
# operator is the normal matrix of the test above up to step `n - 2`, and `V_{n-1} c` at step
# `n - 1`.
@testset "CGS2 norms w₂ explicitly where w₁ lies in the span of V, $T" for T in REAL_ELTYPES
    R = real(T)
    n = 128
    for seed in 1:5
        A, b = arnoldi_test_matrix(T, n, seed)
        c = T.(randn(Xoshiro(100 + seed), n - 1))
        ws = init_gmres_workspace(GMRES(; maxiter = n - 1), zeros(T, n))
        calls = Ref(0)
        w = zeros(T, n)
        op = (Jv, v) -> begin
            calls[] += 1
            calls[] == n - 1 ? mul!(Jv, view(ws.V, :, 1:(n - 1)), c) : mul!(Jv, A, v)
            calls[] == n - 1 && copyto!(w, Jv)
            Jv
        end
        _, info = gmres_solve!(
            ws, op, nothing, b, rnorm(b), zero(R), Int32(n - 1), Int32(n - 1))
        @test info.iterations == n - 1
        @test !info.breakdown
        # the row is the one it is meant to be: one pass leaves more than half of `s₁` in the span
        Vj = ws.V[:, 1:(n - 1)]
        w₁ = w - Vj * (Vj' * w)
        h₂ = Vj' * w₁
        @test norm(h₂)^2 > norm(w₁)^2 / 2
        @test abs(norm(big.(ws.V[:, n])) - 1) <= 4 * eps(R)
    end
end

@testset "Both paths solve R5Diagonal, mgs = $mgs, $AT, $T" for T in REAL_ELTYPES,
    AT in ARRAY_BACKENDS, mgs in (false, true)
    counts = R5Counts(0, 0)
    x₀ = AT(fill(T(2), 8))
    m = Newton(; linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(1e-8), mgs))
    x, st = solve(r5_problem(counts), x₀, m)
    @test x isa AT
    @test st.code == SUCCESS
    @test st.fnorm <= sqrt(eps(T)) * rnorm(AT(fill(T(-1.5), 8)))
    @test norm(Array(x) .- 1) <= 100 * sqrt(eps(T))
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
    # a product alone gives a factorising method no matrix to factorise
    jvp! = (Jv, x, v, q) -> (Jv .= v; Jv)
    @test_throws ArgumentError Newton(; jacobian = JVPFunction(jvp!))
    @test_throws ArgumentError Newton(;
        linsolve = QRFactorization(), jacobian = JVPFunction(jvp!))
    @test Newton(; linsolve = GMRES(), jacobian = JVPFunction(jvp!)) isa Newton
end

# Implicit differentiation factorises the Jacobian at the root, which a GMRES solve does not form:
# `solve` refuses a dual input before the primal solve evaluates `F` once. The factorising control
# gives `dx/dq = 1/(3x² + 2) = 1/5` per entry at the root `x = 1` of `q = 3`.
@testset "solve refuses a dual input for a Newton with GMRES" begin
    calls = Ref(0)
    prob = NonlinearProblem((r, x, q) -> begin
        calls[] += 1
        @. r = x^3 + 2x - q
        r
    end)
    total(q, m) = sum(first(solve(prob, [1.0, 1.0], m, q)))
    @test ForwardDiff.derivative(q -> total(q, Newton()), 3.0) ≈ 0.4
    calls[] = 0
    @test_throws ArgumentError ForwardDiff.derivative(
        q -> total(q, Newton(; linsolve = GMRES())), 3.0)
    @test calls[] == 0
end

# A combinator's members count only their own steps, so every combinator but `Repeat` reports no
# count; `Repeat` runs one member and evaluates nothing itself, so its count is the member's. The
# residual counts its own evaluations on the working type, which a product on duals is not.
@testset "A combinator keeps no call count, and Repeat keeps its member's" begin
    calls = Ref(0)
    prob = NonlinearProblem((r, x, q) -> begin
        eltype(x) <: AbstractFloat && (calls[] += 1)
        @. r = x^3 + 2x - 3
        r
    end)
    g = Newton(; linsolve = GMRES(; maxiter = 10))
    x₀ = collect(R5_COUPLED_X)
    for method in (Anderson(g), Additive(g, g), Multiplicative(g, g), Switch(g, g))
        _, st = solve(prob, copy(x₀), method)
        @test st.iterations > 0
        @test st.f_calls == 0
        @test st.jvp_calls == 0
    end
    calls[] = 0
    _, st = solve(prob, copy(x₀), Repeat(g, 2))
    @test st.f_calls > 0
    @test st.f_calls == calls[]
end

# Choice 1 reads `‖F + J s‖` of the step `s` taken. After a line search on one Arnoldi cycle the
# workspace records that of the accepted step `α d`, from the Arnoldi relation; the control is the
# product of ForwardDiff on `F`. A restarted solve keeps the residual of its direction.
@testset "Choice 1 reads the linear residual of the accepted step, $T" for T in REAL_ELTYPES
    R = real(T)
    prob = r5_coupled(T)
    x₀ = T.(collect(R5_COUPLED_X))
    for (restart, accepted) in ((2, true), (1, false))
        solver = init(prob, copy(x₀),
            Newton(;
                linsolve = GMRES(;
                maxiter = 2, restart = restart, forcing = FixedForcing(R(1e-6)))))
        m = solver.method
        x = copy(x₀)
        Fx = similar(x)
        prob.F(Fx, x, nothing)
        state = start!!(solver.state, m, prob, x, Fx, nothing, rnorm(Fx))
        _, _, state₁, _ = step!!(state, m, prob, x, Fx, nothing)
        ws = state₁.ws
        d = copy(ws.d)
        dn = rnorm(d)
        # the full step of `Static` records the residual of the direction
        full = direction_linres(prob, x₀, d)
        @test isapprox(ws.linres, full; rtol = sqrt(eps(R)))
        GeometricSolvers.accepted_linres!(ws, dn / 2, dn)
        expected = accepted ? direction_linres(prob, x₀, d ./ 2) : full
        @test isapprox(ws.linres, expected; rtol = sqrt(eps(R)))
        @test accepted == !isapprox(ws.linres, full; rtol = sqrt(eps(R)))
    end
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

# A breakdown where `J` is singular on the Krylov space: `J v₁ = 0` gives `H[1, 1] = H[2, 1] = 0`,
# the reduced system has no solution, and the true residual of the direction stays `‖b‖`. The
# solve reports that residual and no convergence; the residual is compared with `‖b - A d‖`.
@testset "A breakdown on a singular operator does not converge, $T" for T in REAL_ELTYPES
    R = real(T)
    for A in (zeros(T, 3, 3), T[0 1; 0 0])
        n = size(A, 1)
        b = zeros(T, n)
        b[1] = one(T)
        ws = init_gmres_workspace(GMRES(; maxiter = 5), b)
        d, info = gmres_solve!(
            ws, (Jv, v) -> mul!(Jv, A, v), nothing, b, norm(b), R(1e-6), 5, 5)
        @test info.breakdown
        @test !info.converged
        @test info.resnorm ≈ norm(b - A * d)
        @test info.resnorm ≈ one(R)
    end
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

@testset "The inner solve allocates nothing after init on Array, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    Base.JLOptions().check_bounds == 1 && continue
    prob = r5_affine()
    m = Newton(; linsolve = GMRES(; maxiter = 4))
    solver = init(prob, zeros(T, 4), m)
    ws = solver.state.ws
    op = GeometricSolvers.R5Operator(solver.state.jvpprep, prob, zeros(T, 4), nothing)
    b = similar(zeros(T, 4))
    map!(-, b, R5_B4)
    β = rnorm(b)
    @test gmres_allocations(ws, op, b, β, T(1e-10), Int32(4), Int32(4); mgs) == 0
    @inferred gmres_solve!(ws, op, nothing, b, β, T(1e-10), Int32(4), Int32(4); mgs)
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

# ─── Flexible GMRES, a basis below the working precision, a lagged preconditioner ───

# A right preconditioner that changes at every application: call `k` divides by the diagonal
# `scales[mod1(k, end)]`, so that `M_j⁻¹` differs between the inner iterations of one solve.
# `refresh!` counts its calls.
mutable struct CyclingPreconditioner{S}
    scales::S
    calls::Int
    refreshes::Int
end

CyclingPreconditioner(scales) = CyclingPreconditioner(scales, 0, 0)

function GeometricSolvers.refresh!(P::CyclingPreconditioner, prob, x, Fx, p)
    P.refreshes += 1
    return nothing
end

function GeometricSolvers.apply_preconditioner!(z, P::CyclingPreconditioner, v)
    P.calls += 1
    z .= v ./ P.scales[mod1(P.calls, length(P.scales))]
    return z
end

# The diagonals of the cycle: scales of up to a factor 100 between entries and between calls, and
# a gentle cycle within `[1/2, 2]` for the rows at the scale of `floatmax`.
function cycling_scales(::Type{T}, AT) where {T}
    [AT(T[1, 1, 1, 1, 1, 1]),
        AT(T[1, 10, 0.1, 5, 0.2, 3]), AT(T[0.3, 4, 1, 0.1, 8, 2])]
end
function gentle_scales(::Type{T}) where {T}
    [T[1, 1, 1, 1, 1, 1], T[1, 2, 0.5, 2, 0.5, 1],
        T[0.5, 1, 2, 1, 2, 0.5]]
end

# A 6 × 6 nonsymmetric system, `F(x) = A x - b`, with its root in `Float64`.
const FG_A6 = [4.0 1.0 0.5 0.2 0.0 0.1
               0.3 3.0 0.8 0.1 0.2 0.0
               0.2 0.4 2.5 0.6 0.0 0.3
               0.1 0.2 0.3 2.0 0.5 0.0
               0.0 0.1 0.2 0.3 1.5 0.4
               0.2 0.0 0.1 0.2 0.3 1.2]
const FG_B6 = [1.0, -2.0, 1.5, -0.5, 0.7, -1.1]
const FG_X6 = FG_A6 \ FG_B6

function fg_affine(A, b)
    return NonlinearProblem((r, x, q) -> (mul!(r, A, x); r .-= b; r))
end

# The right preconditioner that records the type of every vector it is given.
struct RecordingPreconditioner
    seen::Vector{Type}
end

function GeometricSolvers.apply_preconditioner!(z, P::RecordingPreconditioner, v)
    push!(P.seen, typeof(v))
    z .= v
    return z
end

# The diagonal Jacobian of `R5Diagonal` at `x` and its residual, for the inner solves of the
# reduced-basis rows: `J = Diagonal(3x² + 2)`, `b = -F(x)`.
r5_diagonal_system(x) = (3 .* x .^ 2 .+ 2, .-(x .^ 3 .+ 2 .* x .- 3))

const FG_DIAGONAL_X = collect(range(1.5, 3.0; length = 8))

@testset "Flexible GMRES solves where GMRES with a changing preconditioner does not, mgs = $mgs, $AT, $T" for T in REAL_ELTYPES,
    AT in ARRAY_BACKENDS, mgs in (false, true)

    R = real(T)
    A = AT(T.(FG_A6))
    b = AT(T.(FG_B6))
    tol = R(1e-5)
    op = (Jv, v) -> (mul!(Jv, A, v); Jv)
    rel = Dict{Bool, R}()
    for flexible in (false, true)
        P = CyclingPreconditioner(cycling_scales(T, AT))
        g = GMRES(; maxiter = 30, preconditioner = P, flexible, mgs)
        ws = init_gmres_workspace(g, AT(zeros(T, 6)))
        d, info = gmres_solve!(ws, op, P, b, rnorm(b), tol, Int32(30), Int32(30); mgs)
        rel[flexible] = rnorm(b .- A * d) / rnorm(b)
        flexible && @test info.converged
    end
    # the true residual `‖b − J d‖ ≤ η ‖b‖` of the flexible direction; the direction `M_last⁻¹ u`
    # of GMRES misses it by orders of magnitude
    @test rel[true] <= tol
    @test rel[false] > 1000 * tol

    # the Newton status: one step of the flexible solve ends the solve, three steps with GMRES
    # and the same preconditioner do not
    prob = fg_affine(A, b)
    for (flexible, steps) in ((true, 1), (false, 3))
        P = CyclingPreconditioner(cycling_scales(T, AT))
        m = Newton(;
            linsolve = GMRES(; maxiter = 12, forcing = FixedForcing(tol),
            preconditioner = P, flexible, mgs))
        x, st = solve(prob, AT(zeros(T, 6)), m; max_iterations = steps)
        if flexible
            @test st.code == SUCCESS
            @test norm(Array(x) .- FG_X6) <= 100 * tol * norm(FG_X6)
        else
            @test st.code != SUCCESS
        end
    end
end

@testset "A restarted flexible cycle forms its residual from Z y, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    R = real(T)
    A = T.(FG_A6)
    b = T.(FG_B6)
    tol = R(1e-3)
    P = CyclingPreconditioner(cycling_scales(T, Array))
    ws = init_gmres_workspace(
        GMRES(; maxiter = 60, restart = 2, preconditioner = P, flexible = true, mgs),
        zeros(T, 6))
    op = (Jv, v) -> (mul!(Jv, A, v); Jv)
    d, info = gmres_solve!(ws, op, P, b, rnorm(b), tol, Int32(60), Int32(2); mgs)
    @test info.converged
    @test info.iterations > 2
    @test rnorm(b .- A * d) <= tol * rnorm(b)
    @test info.resnorm <= tol * rnorm(b)
end

@testset "The Z slot of the flexible path" begin
    x = zeros(6)
    P = CyclingPreconditioner(cycling_scales(Float64, Array))
    @test init_gmres_workspace(GMRES(; flexible = true), x).Z === nothing
    @test init_gmres_workspace(GMRES(; preconditioner = P), x).Z === nothing
    Z = init_gmres_workspace(
        GMRES(; maxiter = 4, preconditioner = P, flexible = true), x).Z
    @test Z isa Matrix{Float64} && size(Z) == (6, 4)
    @test init_gmres_workspace(
        GMRES(; maxiter = 10, preconditioner = P, flexible = true), x).Z isa Matrix{Float64}
    @test size(init_gmres_workspace(
        GMRES(; maxiter = 10, preconditioner = P, flexible = true), x).Z) == (6, 6)
end

@testset "The basis type: basis_eltype, and Complex{TB} for a complex iterate" begin
    @test eltype(init_gmres_workspace(GMRES(), zeros(4)).V) === Float64
    @test eltype(init_gmres_workspace(GMRES(; basis_eltype = Float32), zeros(4)).V) ===
          Float32
    @test eltype(init_gmres_workspace(
        GMRES(; basis_eltype = Float32), zeros(ComplexF64, 4)).V) === ComplexF32
    @test eltype(init_gmres_workspace(
        GMRES(; basis_eltype = Float32), zeros(ComplexF64, 4)).hd) === ComplexF64
    @test eltype(init_gmres_workspace(
        GMRES(; basis_eltype = Float16), zeros(ComplexF32, 4)).V) === ComplexF16
    # a complex solve on a reduced basis reaches a tolerance below the round-off of the basis type
    # on its true residual
    for T in filter(T -> T <: Complex, ELTYPES), mgs in (false, true)

        TB = real(T) === Float64 ? Float32 : Float16
        tol = real(T) === Float64 ? 1e-12 : 1.0f-5
        A = T.(FG_A6) .+ T(0.3im) .* T.(FG_A6')
        b = T.(FG_B6) .- T(0.5im)
        ws = init_gmres_workspace(GMRES(; maxiter = 40, basis_eltype = TB), zeros(T, 6))
        @test sizeof(real(eltype(ws.V))) < sizeof(real(T))
        op = (Jv, v) -> (mul!(Jv, A, v); Jv)
        d, info = gmres_solve!(
            ws, op, nothing, b, rnorm(b), tol, Int32(40), Int32(40); mgs)
        @test info.converged
        @test rnorm(b .- A * d) <= tol * rnorm(b)
    end
end

@testset "The preconditioner reads a working-precision copy of a reduced basis column, $T" for T in REAL_ELTYPES
    TB = T === Float64 ? Float32 : Float16
    P = RecordingPreconditioner(Type[])
    A = T.(FG_A6)
    b = T.(FG_B6)
    ws = init_gmres_workspace(
        GMRES(; maxiter = 6, basis_eltype = TB, preconditioner = P),
        zeros(T, 6))
    op = (Jv, v) -> (mul!(Jv, A, v); Jv)
    for mgs in (false, true)
        empty!(P.seen)
        gmres_solve!(ws, op, P, b, rnorm(b), T(1e-4), Int32(6), Int32(6); mgs)
        @test !isempty(P.seen)
        @test all(t -> t === Vector{T}, P.seen)
    end
end

@testset "The CGS2 step keeps a reduced basis orthogonal to the round-off of its type, $T" for T in REAL_ELTYPES
    # a diagonal operator with κ = 1e5 and a full cycle of 30 steps without a stop: one pass of
    # classical Gram–Schmidt leaves ‖VᵀV − I‖ near 7e-3 on a Float32 basis here, and the second
    # pass keeps it below one unit of the round-off of the stored type
    TB = T === Float64 ? Float32 : Float16
    n = 40
    jd = T.(collect(range(1, 1e5; length = n)))
    b = ones(T, n)
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 30, basis_eltype = TB), zeros(T, n))
    gmres_solve!(ws, op, nothing, b, rnorm(b), zero(T), Int32(30), Int32(30))
    @test ws.arnoldi == 30
    V = Float64.(ws.V[:, 1:30])
    @test opnorm(V' * V - I) <= 4 * eps(TB)
end

@testset "A Float32 basis reaches the forcing tolerance on R5Diagonal, mgs = $mgs, $AT" for AT in ARRAY_BACKENDS,
    mgs in (false, true)

    # the inner solve at the start: the Givens estimate of a Float32 basis is not the true
    # residual, so the solve ends only where the true residual meets a tolerance below eps(Float32)
    jd, b = r5_diagonal_system(FG_DIAGONAL_X)
    Jd = AT(jd)
    bb = AT(b)
    op = (Jv, v) -> (Jv .= Jd .* v; Jv)
    for tol in (1e-8, 1e-12)
        ws = init_gmres_workspace(GMRES(; maxiter = 30, basis_eltype = Float32), AT(zeros(8)))
        d, info = gmres_solve!(
            ws, op, nothing, bb, rnorm(bb), tol, Int32(30), Int32(30); mgs)
        @test info.converged
        @test rnorm(bb .- Jd .* d) <= tol * rnorm(bb)
        @test info.resnorm == rnorm(bb .- Jd .* d)
    end

    # the Newton solve: every step meets its forcing tolerance on the true residual
    counts = R5Counts(0, 0)
    m = Newton(;
        linsolve = GMRES(; maxiter = 20, forcing = FixedForcing(1e-10),
        basis_eltype = Float32, mgs))
    x, st = solve(r5_problem(counts), AT(copy(FG_DIAGONAL_X)), m)
    @test x isa AT
    @test st.code == SUCCESS
    @test norm(Array(x) .- 1) <= 100 * sqrt(eps(Float64))
    @test eltype(init(r5_problem(counts), AT(copy(FG_DIAGONAL_X)), m).state.ws.V) ===
          Float32
end

@testset "The attainable accuracy of a Float32 basis against the CGS2 bound of LLD2022" begin
    # Lindquist, Luszczek and Dongarra (2022), the simplified form of Theorem 1 (p. 1029): one
    # cycle of `j` CGSR iterations in a precision of unit roundoff `u` attains a normwise backward
    # error of at most 1.1 times that of the exact cycle plus
    # `(4 j^{3/2} + 30 j + 3 p j^{1/2} + 3 c₁(n, j)) u`, with the norm ‖A‖_F. Three readings make
    # it a number: the paper gives `c₁(n, j)` only as O(n j), and this test takes `n j`; it does
    # not define `p`, and this test takes the largest number of nonzeros in a row of the
    # Jacobian, 1 for a diagonal one; the cycle with the Float64 basis stands for the exact one.
    # The test drops the factor 1.1 to 1, so it asserts more than the theorem. The paper proves the
    # bound for CGSR only; the MGS rows check the same number.
    jd, b = r5_diagonal_system(FG_DIAGONAL_X)
    n = length(b)
    p = 1
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    backward(d) = rnorm(b .- jd .* d) / (norm(jd) * rnorm(d) + rnorm(b))
    for j in (4, 8), mgs in (false, true)

        ws64 = init_gmres_workspace(GMRES(; maxiter = j), zeros(n))
        d64, _ = gmres_solve!(ws64, op, nothing, b, rnorm(b), 0.0, Int32(j), Int32(j); mgs)
        ws32 = init_gmres_workspace(GMRES(; maxiter = j, basis_eltype = Float32), zeros(n))
        d32, info = gmres_solve!(
            ws32, op, nothing, b, rnorm(b), 0.0, Int32(j), Int32(j); mgs)
        @test info.iterations == j
        u = eps(Float32) / 2
        bound = (4 * j^1.5 + 30 * j + 3 * p * sqrt(j) + 3 * n * j) * u
        @test backward(copy(d32)) <= backward(copy(d64)) + bound
    end
    # with the true residual at the end of each cycle and the update `u += V y` in Float64, the
    # restarts reach the working precision (Lindquist, Luszczek and Dongarra 2022, §3; Aliaga
    # et al. 2020, §6.2), within 34 iterations at every point; a residual of the Givens recurrence
    # in its place reports convergence at a true residual near 2e-8 for `restart = 2` and
    # `tol = 1e-14`, and an update in Float32 stops near 3e-8
    for mgs in (false, true), restart in (2, 4), tol in (1e-13, 1e-14)
        ws = init_gmres_workspace(
            GMRES(; maxiter = 60, restart, basis_eltype = Float32), zeros(n))
        d, info = gmres_solve!(
            ws, op, nothing, b, rnorm(b), tol, Int32(60), Int32(restart); mgs)
        @test info.converged
        @test rnorm(b .- jd .* d) <= tol * rnorm(b)
        @test isapprox(info.resnorm, rnorm(b .- jd .* d); rtol = 8 * eps(Float64))
    end
end

@testset "Newton(; reuse = k) refreshes a GMRES preconditioner once per k steps, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    R = real(T)
    x₀ = T.(collect(R5_COUPLED_X))
    prob = r5_coupled(T)
    for k in (1, 2, 3)
        count = Ref(Int32(0))
        P = ScaledRightPreconditioner(count, T(2))
        m = Newton(; reuse = k,
            linsolve = GMRES(;
                maxiter = 8, forcing = FixedForcing(R(1e-6)), preconditioner = P, mgs))
        _, st = solve(prob, copy(x₀), m)
        @test st.code == SUCCESS
        @test st.iterations >= 3
        @test st.step_failures == 0
        @test count[] == cld(st.iterations, k)
    end
    # an inner solve that reaches its cap refreshes at the next step: every step of a cap of one
    # inner iteration is capped, so every step after the first adds one early refresh
    for k in (2, 3)
        count = Ref(Int32(0))
        P = ScaledRightPreconditioner(count, T(2))
        m = Newton(; reuse = k,
            linsolve = GMRES(;
                maxiter = 1, forcing = FixedForcing(R(1e-6)), preconditioner = P, mgs))
        _, st = solve(prob, copy(x₀), m; max_iterations = 6)
        @test st.iterations >= 3
        @test st.step_failures == st.iterations
        @test count[] == st.iterations
        @test count[] > cld(st.iterations, k)
    end
end

@testset "The age of the preconditioner after a step" begin
    FA = GeometricSolvers.FactorAge
    rule = GeometricSolvers.preconditioner_rule
    STALLED = GeometricSolvers.STALLED
    P = ScaledRightPreconditioner(Ref(Int32(0)), 2.0)
    m = Newton(; reuse = 3, linsolve = GMRES(; preconditioner = P))
    # a refresh starts a period, also an early one at a nonzero age: the next refresh is due
    # three steps after it
    # the third field records whether the step stepped along a preconditioner that an earlier step
    # refreshed
    @test rule(FA(Int32(0), false, false), m, true, SUCCESS, Int32(0)) ===
          FA(Int32(1), false, false)
    @test rule(FA(Int32(2), true, true), m, true, SUCCESS, Int32(0)) ===
          FA(Int32(1), false, false)
    @test rule(FA(Int32(1), false, false), m, false, SUCCESS, Int32(0)) ===
          FA(Int32(2), false, true)
    @test rule(FA(Int32(2), false, true), m, false, SUCCESS, Int32(0)) ===
          FA(Int32(0), false, true)
    # a step that stalled or whose inner solve reached its cap asks for a refresh at the next one
    @test rule(FA(Int32(1), false, false), m, false, SUCCESS, Int32(1)) ===
          FA(Int32(2), true, true)
    @test rule(FA(Int32(1), false, false), m, false, STALLED, Int32(0)) ===
          FA(Int32(2), true, true)
    # a period of one step refreshes at every step
    m1 = Newton(; reuse = 1, linsolve = GMRES(; preconditioner = P))
    @test rule(FA(Int32(0), false, false), m1, true, SUCCESS, Int32(0)) ===
          FA(Int32(0), false, false)
    # without reuse the rule state is that of the globalization, unchanged
    @test rule(nothing, Newton(; linsolve = GMRES()), true, SUCCESS, Int32(1)) === nothing
end

@testset "A reduced-basis breakdown on a singular operator stops without convergence, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    # `b = e₁` is exact in the basis type and `J e₁ = 0`, so the first Arnoldi step breaks down
    # exactly; the true residual is `‖b‖`, and the solve stops there rather than restart to its cap
    TB = T === Float64 ? Float32 : Float16
    jd = T[0, 1, 2]
    b = T[1, 0, 0]
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 10, restart = 3, basis_eltype = TB), zeros(T, 3))
    _, info = gmres_solve!(ws, op, nothing, b, rnorm(b), T(1e-4), Int32(10), Int32(3); mgs)
    @test info.breakdown
    @test !info.converged
    @test info.iterations == 1
    @test info.resnorm == 1
end

@testset "A reduced-basis breakdown that meets the Givens estimate restarts on its true residual, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    # `s` rounds to zero in the basis type, so the stored `v₁` is `e₁`, an eigenvector of `J`: the
    # first step breaks down with a zero Givens estimate, while the true residual is `s`, above
    # the tolerance `s²`; the restart from that residual reaches it
    TB = T === Float64 ? Float32 : Float16
    s = sqrt(sqrt(floatmin(T)))
    @test iszero(TB(s))
    jd = T[2, 3, 5]
    b = T[1, s, 0]
    tol = s^2
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 10, restart = 3, basis_eltype = TB), zeros(T, 3))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(10), Int32(3); mgs)
    @test info.converged
    @test rnorm(b .- jd .* d) <= tol * rnorm(b)
    @test info.resnorm <= tol * rnorm(b)
end

# Two systems whose Krylov space is invariant after a few steps: `J = 14 I` with a constant `b`, a
# space of dimension one, and a diagonal `J` of the three eigenvalues 2, 5 and 11, of dimension
# three. A stored column of a reduced basis has unit norm only to the round-off of its type, so
# the projection at the last step of the space leaves a remainder of that order along the basis.
function invariant_krylov_system(::Type{T}, kind::Symbol) where {T}
    n = 100
    kind === :scaled_identity && return fill(T(14), n), fill(T(-9), n)
    return T.(repeat([2, 5, 11], 34)[1:n]), T.(collect(range(-1, 1; length = n)))
end

@testset "A reduced basis on a Krylov space invariant to its round-off converges in one cycle, mgs = $mgs, $kind, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true), kind in (:scaled_identity, :three_eigenvalues)

    # one pass of MGS takes that remainder for a new direction, and the cycle ends at its cap
    # with a true residual of 1, Inf or near eps(TB) ‖b‖; its second pass removes the remainder,
    # and the cycle ends early and restarts from its true residual. On `J = 14 I` the stored `v₁`
    # is parallel to `b`, so the remainder lies along `v₁`, of order eps(TB)² ‖J v₁‖ after the two
    # passes of either step, and the breakdown test ends the cycle there: without it, the CGS2
    # cycle on a Float16 basis on Float64 takes the remainder for `v₂ = v₁` and ends at its cap
    jd, b = invariant_krylov_system(T, kind)
    tol = T === Float64 ? 1e-10 : 1.0f-5
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 20, basis_eltype = TB), zeros(T, length(b)))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(20), Int32(20); mgs)
    @test eltype(ws.V) === TB
    @test info.converged
    @test info.iterations < 20
    @test rnorm(b .- jd .* d) <= tol * rnorm(b)
    @test isapprox(info.resnorm, rnorm(b .- jd .* d); rtol = 8 * eps(T))
end

@testset "Newton with a reduced basis solves x³ + 2x − 3 = 0 on both paths, mgs = $mgs, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true)

    # the constant start makes the Krylov space of each step that of a multiple of I, which one
    # pass of MGS on a reduced basis does not see, and the step ends NONFINITE
    F!(r, x, q) = (@. r = x^3 + 2x - 3; r)
    η = T === Float64 ? 1e-10 : 1.0f-5
    m = Newton(;
        linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(η), basis_eltype = TB, mgs))
    x₀ = fill(T(2), 8)
    x, st = solve(NonlinearProblem(F!), copy(x₀), m)
    @test st.code == SUCCESS
    @test maximum(abs, x .- 1) <= sqrt(eps(T))
end

# `J = u₂ (u₁ + u₂)ᵀ` for the unit vectors `u₁` and `u₂`, constant on the first and on the second
# half of the entries, and `b = u₁`: the Krylov space is `span(u₁, u₂)`, invariant, and `J` is
# singular on it. Their entries, 0.2, have no exact value in the basis type, so each stored column
# is parallel to its vector and has unit norm only to the round-off of `TB`, and the second step
# leaves a remainder of order eps(TB)² ‖w₀‖ along `v₂`; the reduced system has no solution.
@testset "A breakdown at the round-off of a reduced basis on a singular operator stops without convergence, mgs = $mgs, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true)

    n = 50
    u₁ = [fill(T(0.2), 25); zeros(T, 25)]
    u₂ = [zeros(T, 25); fill(T(0.2), 25)]
    s = u₁ .+ u₂
    op = (Jv, v) -> (Jv .= u₂ .* sum(s .* v); Jv)
    b = copy(u₁)
    tol = T === Float64 ? 1e-10 : 1.0f-5
    ws = init_gmres_workspace(GMRES(; maxiter = 20, basis_eltype = TB), zeros(T, n))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(20), Int32(20); mgs)
    Jd = op(similar(d), d)
    @test info.breakdown
    @test !info.converged
    @test info.iterations == 2
    @test isapprox(info.resnorm, rnorm(b .- Jd); rtol = 8 * eps(T))
    @test isapprox(info.resnorm, rnorm(b); rtol = sqrt(eps(TB)))
end

# The singular operator above in a random orthonormal frame `Q`, `J = Q u₂ (u₁ + u₂)ᵀ Qᵀ` and
# `b = Q u₁`: the rotated diagonal entry of the breakdown step is then not exactly zero but at the
# round-off of the basis, and the breakdown test takes it as zero, so the solve stops there
# rather than dividing `g[j]` by it and restarting to `maxiter`. The residual is at least `‖b‖`,
# the minimum over every `d` since `b ⟂ range(J)`; an earlier step of the frame leaves a
# remainder with a direction error of order eps(TB), which the test does not see, so `d` holds a
# component of order 1 / eps(TB) (K149), and the residual exceeds `‖b‖`. The two breakdown
# tests read the norm `‖w₀‖` of the product, which does not scale with `b`: at the breakdown
# step it is about 2e-4, and with `‖b‖ = 1e-10` a pivot test on `‖b‖` would take the Float16
# pivot as nonzero.
@testset "A breakdown on a singular operator in a rotated frame stops without convergence, ‖b‖ = $σ, mgs = $mgs, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true),
    σ in (1, 1e-10)

    n = 50
    Q = T.(Matrix(qr(randn(Xoshiro(7), n, n)).Q))
    u₁ = Q * [fill(T(0.2), 25); zeros(T, 25)]
    u₂ = Q * [zeros(T, 25); fill(T(0.2), 25)]
    s = u₁ .+ u₂
    op = (Jv, v) -> (Jv .= u₂ .* sum(s .* v); Jv)
    b = T(σ) .* u₁
    tol = T === Float64 ? 1e-10 : 1.0f-5
    ws = init_gmres_workspace(GMRES(; maxiter = 20, basis_eltype = TB), zeros(T, n))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(20), Int32(20); mgs)
    Jd = op(similar(d), d)
    @test info.breakdown
    @test !info.converged
    @test info.iterations < 20
    @test isapprox(info.resnorm, rnorm(b .- Jd); rtol = 8 * eps(T))
    @test info.resnorm >= rnorm(b) * (1 - 8 * eps(T))
    # the reduced system is singular at the breakdown step: its diagonal entry and its weight are 0
    @test iszero(ws.H[ws.arnoldi, ws.arnoldi])
    @test iszero(ws.y[ws.arnoldi])
    @test all(isfinite, d)
end

# A consistent operator in the frame above, `J = Q (u₂ (u₁ + u₂)ᵀ + δ u₁ u₁ᵀ) Qᵀ` and `b = Q u₁`:
# on `span(u₁, u₂)` it is `[δ 0; 1 1]`, nonsingular, and `d = (u₁ - u₂) / δ` solves it. The stored
# Float16 basis leaves a rotated diagonal entry at its round-off at a breakdown step, but the
# entry `g[j]` of the rotated right-hand side there is about 2e-4 ‖r₀‖, below
# `sqrt(eps(Float16)) ‖r₀‖`: the cycle ends on its true residual and restarts, and the solve
# converges. Each row stops without convergence where every such step reads `J` as singular. The
# pivot of such a step counts as zero, so `d` takes no weight of order 1 / eps(TB) from it: `‖d‖`
# is 1.2e5–3.3e5 here and 1.3e8–7.6e8 with the pivot kept, the same in seven emulated summation
# orders and with either `--check-bounds`, and the bound 10 / eps(TB)² lies 12 times or more
# from both. The rows of the other basis types that restart there converge or not with the
# summation order and the platform (K149), and are not asserted.
@testset "A consistent operator whose rotated pivot is at the round-off of a reduced basis restarts and converges, δ = $δ, mgs = $mgs, Float16 on Float64" for (mgs, δ) in (
    (false, 0.3), (true, 0.1), (true, 0.3))
    T = Float64
    TB = Float16
    n = 50
    Q = T.(Matrix(qr(randn(Xoshiro(7), n, n)).Q))
    u₁ = Q * [fill(T(0.2), 25); zeros(T, 25)]
    u₂ = Q * [zeros(T, 25); fill(T(0.2), 25)]
    s = u₁ .+ u₂
    op = (Jv, v) -> (Jv .= u₂ .* sum(s .* v) .+ T(δ) .* u₁ .* sum(u₁ .* v); Jv)
    b = copy(u₁)
    tol = 1e-8
    ws = init_gmres_workspace(GMRES(; maxiter = 40, basis_eltype = TB), zeros(T, n))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(40), Int32(20); mgs)
    @test info.converged
    @test info.iterations < 40
    @test rnorm(b .- op(similar(d), d)) <= tol * rnorm(b)
    @test isapprox(info.resnorm, rnorm(b .- op(similar(d), d)); rtol = 8 * eps(T))
    @test norm(d) < 10 / T(eps(TB))^2
end

# A singular operator with a part that `J` reaches, `J = Q (u₂ (u₁ + u₂)ᵀ + u₃ u₃ᵀ) Qᵀ` and
# `b = Q (u₁ + c u₃)` for orthonormal `u₁`, `u₂`, `u₃`: `range(J) = span(Q u₂, Q u₃)`, so every
# residual is at least 1, and the rotated right-hand side keeps about 1 at the singular step of
# the first cycle, where `‖r₀‖ = ‖b‖ ≈ c`. With `c = 10` that is 0.094 ‖r₀‖, above
# `sqrt(eps(Float16)) ‖r₀‖ = 0.031 ‖r₀‖`, and the first cycle stops; with `c = 60` it is
# 0.016 ‖r₀‖ and with `c = 200` 0.0049 ‖r₀‖, below it, and the cycle ends on its true residual
# and restarts, and the second cycle, whose `r₀` is that residual, keeps 0.6–0.7 ‖r₀‖ and stops.
# On a reduced basis every cycle ends on one product, so the cycles are the products less the
# iterations.
@testset "A singular step of a reduced basis stops where its residual is a large part of r₀ and restarts where it is a small one, c = $c, mgs = $mgs" for (c, cycles) in ((
        10, 1), (60, 2), (200, 2)),
    mgs in (false, true)

    T = Float64
    TB = Float16
    n = 60
    Q = Matrix(qr(randn(Xoshiro(7), n, n)).Q)
    e(k) = Q * [i in (20(k - 1) + 1):(20k) ? 1 / sqrt(20) : 0.0 for i in 1:n]
    u₁, u₂, u₃ = e(1), e(2), e(3)
    s = u₁ .+ u₂
    products = Ref(0)
    op = (Jv, v) -> (products[] += 1; Jv .= u₂ .* sum(s .* v) .+ u₃ .* sum(u₃ .* v); Jv)
    b = u₁ .+ c .* u₃
    ws = init_gmres_workspace(GMRES(; maxiter = 20, basis_eltype = TB), zeros(T, n))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), 1e-10, Int32(20), Int32(20); mgs)
    @test products[] - info.iterations == cycles
    @test info.iterations < 20
    @test info.breakdown
    @test !info.converged
    Jd = op(similar(d), d)
    @test isapprox(info.resnorm, rnorm(b .- Jd); rtol = 8 * eps(T))
    @test info.resnorm >= 1 - 8 * eps(T)
end

# `J = diag(1, 1 + eps(TB), 2, 3)` and `b = 1`: the space is not invariant before step 4, but
# the cluster leaves a remainder at step 3 of about 0.8 eps(TB) ‖w₀‖, below the round-off of the
# basis and above the breakdown test's `2e-3 eps(TB) ‖w₀‖`; the one cycle of `maxiter = 4` runs
# through it to step 4
@testset "A reduced-basis remainder below the round-off of the basis on a cluster is no breakdown, mgs = $mgs, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true)

    jd = T[1, 1 + eps(TB), 2, 3]
    b = ones(T, 4)
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 4, basis_eltype = TB), zeros(T, 4))
    _, info = gmres_solve!(ws, op, nothing, b, rnorm(b), zero(T), Int32(4), Int32(4); mgs)
    @test ws.arnoldi == 4
    @test info.iterations == 4
    @test !info.breakdown
end

# `J = diag(1, 2)` and `b = (1, ε)` for `ε = eps(TB) / 128`, which the basis type holds: the
# stored `v₁` is `b`, of unit norm to `ε²`, and the remainder of step 1 is `ε ‖w₀‖` to first
# order, a ratio `hj / (eps(TB) ‖w₀‖)` of 1/128, between the breakdown test's 2e-3 and the
# smallest ratio of a non-invariant step of `scripts/gmres_breakdown_ratio.jl`, 0.0168; the one
# cycle of `maxiter = 2` runs through it to step 2
@testset "A reduced-basis remainder of eps(TB) / 128 is no breakdown, mgs = $mgs, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true)

    jd = T[1, 2]
    b = T[1, eps(TB) / 128]
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 2, basis_eltype = TB), zeros(T, 2))
    _, info = gmres_solve!(ws, op, nothing, b, rnorm(b), zero(T), Int32(2), Int32(2); mgs)
    @test ws.arnoldi == 2
    @test info.iterations == 2
end

# The breakdown test of a reduced basis acts only where the cycle would go on. On `J = 14 I` the
# last cycle meets the tolerance on its Givens estimate at the invariant step, so it reports no
# breakdown. On the singular operator `J = u₂ (u₁ + u₂)ᵀ` above, the second step is the last of
# the budget with `maxiter = 2`, and the last of each cycle with `restart = 2`: the cycle ends
# there without the test, so neither solve reports a breakdown.
@testset "The breakdown test of a reduced basis acts only where the cycle would go on, mgs = $mgs, $TB on $T" for T in REAL_ELTYPES,
    TB in filter(S -> sizeof(S) < sizeof(T), (Float16, REAL_ELTYPES...)),
    mgs in (false, true)

    tol = T === Float64 ? 1e-10 : 1.0f-5
    jd, b = invariant_krylov_system(T, :scaled_identity)
    op = (Jv, v) -> (Jv .= jd .* v; Jv)
    ws = init_gmres_workspace(GMRES(; maxiter = 20, basis_eltype = TB), zeros(T, length(b)))
    _, info = gmres_solve!(ws, op, nothing, b, rnorm(b), tol, Int32(20), Int32(20); mgs)
    @test info.converged
    @test !info.breakdown

    u₁ = [fill(T(0.2), 25); zeros(T, 25)]
    u₂ = [zeros(T, 25); fill(T(0.2), 25)]
    s = u₁ .+ u₂
    op = (Jv, v) -> (Jv .= u₂ .* sum(s .* v); Jv)
    for (maxiter, restart) in ((2, 20), (20, 2))
        ws = init_gmres_workspace(GMRES(; maxiter = 20, basis_eltype = TB), zeros(T, 50))
        _, info = gmres_solve!(ws, op, nothing, u₁, rnorm(u₁), tol, Int32(maxiter),
            Int32(restart); mgs)
        @test !info.converged
        @test !info.breakdown
        @test info.iterations == maxiter
    end
end

@testset "reuse takes a GMRES with a preconditioner only" begin
    P = ScaledRightPreconditioner(Ref(Int32(0)), 2.0)
    msg = r"^ArgumentError: Newton with a matrix-free linear method takes no factor reuse"
    @test_throws msg Newton(; linsolve = GMRES(), reuse = 2)
    @test_throws msg Newton(; linsolve = GMRES(; flexible = true), reuse = 2)
    @test_throws msg Newton(; linsolve = GeometricSolvers.BiCGStab(; preconditioner = P),
        reuse = 2)
    @test Newton(; linsolve = GMRES(; preconditioner = P), reuse = 2).reuse === Int32(2)
    @test Newton(; linsolve = GMRES(), preconditioner = P, reuse = 2).reuse === Int32(2)
    @test_throws r"^ArgumentError: Newton with a matrix-free linear method takes no refinement" Newton(;
        linsolve = GMRES(; preconditioner = P), reuse = 2, refinement = Refinement())
end

@testset "DogLeg refuses a reduced basis, and GMRES validates basis_eltype" begin
    @test_throws r"^ArgumentError: Newton with a DogLeg globalization takes a GMRES with basis_eltype = Nothing" Newton(;
        linsolve = GMRES(; basis_eltype = Float32), globalization = DogLeg())
    @test Newton(; linsolve = GMRES(; flexible = true), globalization = DogLeg()) isa Newton
    msg = r"^ArgumentError: GMRES basis_eltype must be Nothing or a concrete AbstractFloat type"
    @test_throws msg GMRES(; basis_eltype = Int)
    @test_throws msg GMRES(; basis_eltype = ComplexF32)
    @test_throws msg GMRES(; basis_eltype = Union{})
    @test_throws msg GMRES(; basis_eltype = AbstractFloat)
    @test GMRES(; basis_eltype = Float16).basis_eltype === Float16
end

@testset "The rebuild of Newton and adapt keep flexible and basis_eltype" begin
    P = ScaledRightPreconditioner(Ref(Int32(0)), 2.0)
    m = Newton(; linsolve = GMRES(; flexible = true, basis_eltype = Float32, mgs = true),
        forcing = FixedForcing(0.1), preconditioner = P)
    g = m.linsolve
    @test g.flexible === Val(true)
    @test g.basis_eltype === Float32
    @test g.mgs
    @test g.forcing === FixedForcing(0.1)
    @test g.preconditioner === P
    @test g isa GMRES{FixedForcing{Float64}, typeof(P), Nothing, Float32, true}
    g₀ = GMRES(; flexible = true, basis_eltype = Float32)
    @test Adapt.adapt(Array, g₀) === g₀
end

@testset "The forcing record of the flexible and the reduced-basis step, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    R = real(T)
    n = 12
    x₀ = T.(collect(range(0.2, 2.4; length = n)))
    prob = r5_coupled(T)
    f1 = EisenstatWalkerChoice1(R; η₀ = R(0.1))
    P = ScaledRightPreconditioner(Ref(Int32(0)), T(2))
    TB = T === Float64 ? Float32 : Float16
    gmres = (
        ("flexible",
            GMRES(; maxiter = 30, forcing = f1, preconditioner = P, flexible = true, mgs)),
        ("reduced", GMRES(; maxiter = 30, forcing = f1, basis_eltype = TB, mgs)))
    for (label, g) in gmres, globalization in (GeometricSolvers.Static(), Backtracking())

        solver = init(prob, copy(x₀), Newton(; linsolve = g, globalization))
        x = copy(x₀)
        Fx = similar(x)
        prob.F(Fx, x, nothing)
        state = start!!(solver.state, solver.method, prob, x, Fx, nothing, rnorm(Fx))
        ηs = R[]
        norms = R[rnorm(Fx)]
        lins = R[]
        for _ in 1:3
            # a step may update `x` in place, so the start of the step is kept
            x_start = copy(x)
            x₊, F₊, state, info = step!!(state, solver.method, prob, x, Fx, nothing)
            # the inner solve stops on the forcing test, short of n iterations
            @test 1 <= state.ws.gmres_iterations < n
            push!(ηs, R(state.ws.last_eta))
            push!(norms, R(info.fnorm))
            # the true linear residual of the accepted step, from ForwardDiff on `F` itself
            push!(lins, R(direction_linres(prob, x_start, x₊ .- x_start)))
            x, Fx = x₊, F₊
        end
        @test ηs[1] == f1.η₀
        for k in 2:3
            # the φ-safeguard does not bind
            @test ηs[k - 1]^R((1 + sqrt(5)) / 2) <= R(0.1)
            ref = GeometricSolvers.compute_forcing(
                f1, k - 1, norms[k], norms[k - 1], ηs[k - 1], lins[k - 1])
            # the recorded linear residual is the Givens estimate of the Arnoldi relation on the
            # flexible path and the true residual on the reduced basis; either is exact to the
            # round-off of the working type relative to `‖F_{k-1}‖`, which the rule divides by
            @test abs(ηs[k] - ref) <= 100 * eps(R)
        end
    end
end

# The bytes of one inner solve with the preconditioner `P`, through a function barrier that calls
# the solve once before it measures.
function preconditioned_allocations(ws, op, P, b, β, tol, maxiter, restart; mgs)
    gmres_solve!(ws, op, P, b, β, tol, maxiter, restart; mgs)
    return @allocated gmres_solve!(ws, op, P, b, β, tol, maxiter, restart; mgs)
end

@testset "The flexible and the reduced-basis solves allocate nothing after init on Array, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    Base.JLOptions().check_bounds == 1 && continue
    R = real(T)
    TB = T === Float64 ? Float32 : Float16
    prob = r5_affine()
    P = ScaledRightPreconditioner(Ref(Int32(0)), T(2))
    for g in (GMRES(; maxiter = 4, preconditioner = P, flexible = true, mgs),
        GMRES(; maxiter = 4, restart = 2, preconditioner = P, flexible = true, mgs),
        GMRES(; maxiter = 4, basis_eltype = TB, mgs))
        solver = init(prob, zeros(T, 4), Newton(; linsolve = g))
        ws = solver.state.ws
        op = GeometricSolvers.R5Operator(solver.state.jvpprep, prob, zeros(T, 4), nothing)
        b = T.(.-R5_B4)
        β = rnorm(b)
        @test preconditioned_allocations(
            ws, op, g.preconditioner, b, β, R(1e-10), Int32(4), g.restart; mgs) == 0
        @inferred gmres_solve!(
            ws, op, g.preconditioner, b, β, R(1e-10), Int32(4), g.restart; mgs)
    end
    # the control: the helper sees the allocation of an operator that allocates
    g = GMRES(; maxiter = 4, basis_eltype = TB, mgs)
    ws = init(prob, zeros(T, 4), Newton(; linsolve = g)).state.ws
    A = Diagonal(T[3, 4, 5, 6])
    alloc_op = (Jv, v) -> (Jv .= A * v; Jv)
    b = T.(.-R5_B4)
    @test preconditioned_allocations(
        ws, alloc_op, nothing, b, rnorm(b), R(1e-10), Int32(4), g.restart; mgs) > 0
end

@testset "The scale rows: A and b at 4√floatmax and floatmax/8, mgs = $mgs, $T" for T in REAL_ELTYPES,
    mgs in (false, true)

    R = real(T)
    TB = T === Float64 ? Float32 : Float16
    for s in (4 * sqrt(floatmax(R)), floatmax(R) / 8)
        c = s / R(4)
        A = c .* T.(FG_A6)
        b = c .* T.(FG_B6)
        op = (Jv, v) -> (mul!(Jv, A, v); Jv)
        tol = R(1e-4)
        P = CyclingPreconditioner(gentle_scales(T))
        for g in (GMRES(; maxiter = 30, preconditioner = P, flexible = true, mgs),
            GMRES(; maxiter = 30, basis_eltype = TB, mgs))
            ws = init_gmres_workspace(g, zeros(T, 6))
            d, info = gmres_solve!(
                ws, op, g.preconditioner, b, rnorm(b), tol, Int32(30), Int32(30); mgs)
            @test info.converged
            @test all(isfinite, d)
            # the solution of the unscaled system, within the tolerance times κ(A) ≤ 10
            @test norm(Float64.(d) .- FG_X6) <= 10 * tol * norm(FG_X6)
        end
    end
end
