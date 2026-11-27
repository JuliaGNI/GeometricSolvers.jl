# R5 matrix-free Newton–Krylov with the fixed-memory methods `BiCGStab` and `DQGMRES`: the Newton
# step to the forcing tolerance on the true residual, the forcing record of each step, the
# products per iteration, the fixed workspace, the four breakdown guards of `BiCGStab`, the
# conjugated inner products, the right preconditioner, the DQGMRES window, the scale of the scalar
# recurrences, and the refusals of the `matrix_free` trait, in `Newton`, `Continuation` and
# `KroneckerFactorization`.
#
# The Newton solves run on `Array` and on `JLArray` (a CPU array that refuses scalar indexing) in
# `REAL_ELTYPES`.

using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: LinearAlgebra, Diagonal, dot, mul!, norm
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, BiCGStab, Bisection, DQGMRES, DogLeg,
                        EisenstatWalkerChoice1, StrongWolfe,
                        FixedForcing, GMRES, JVPFunction, Static, compute_forcing,
                        LeastSquaresProblem, LeftPreconditioned, LevenbergMarquardt,
                        LUFactorization, MINRESQLP, NONFINITE, NonlinearProblem, Newton,
                        PseudoTransient,
                        QRFactorization, Refinement, SUCCESS, bicgstab_solve!,
                        dqgmres_solve!, init, init_bicgstab_workspace,
                        init_dqgmres_workspace, matrix_free, rnorm, solve, solve!, start!!,
                        step!!

include("../helpers/matrix.jl")

allowscalar(false)

const FIXED_METHODS = (BiCGStab, DQGMRES)

# ─── The problems ────────────────────────────────────────────────────────────

# The diagonal residual `F_i = x_i^3 + 2x_i - 3` of `test/nonlinear/krylov.jl`, whose Jacobian
# `3x_i^2 + 2` no array holds, with a counted product.
mutable struct FixedCounts
    jvp::Int
end

r5_diagonal!(r, x, q) = (@. r = x^3 + 2x - 3; r)

struct DiagonalJVP{C}
    counts::C
end

function (p::DiagonalJVP)(Jv, x, v, q)
    p.counts.jvp += 1
    @. Jv = (3x^2 + 2) * v
    return Jv
end

# A linear operator `A` as the `op!(Jv, v)` of a solve, which counts its products.
struct CountingMatrix{M}
    A::M
    count::Base.RefValue{Int}
end

CountingMatrix(A) = CountingMatrix(A, Ref(0))

function (op::CountingMatrix)(Jv, v)
    op.count[] += 1
    mul!(Jv, op.A, v)
    return Jv
end

# A nonsymmetric 6 × 6 system that neither method solves to `1e-12` in two iterations.
function nonsymmetric(::Type{T}) where {T}
    A = T[4 1 0 0 0 2
          -1 3 1 0 0 0
          0 -2 5 1 0 0
          0 0 1 2 -1 0
          0 0 0 3 6 1
          1 0 0 0 -2 3]
    b = T[1, -2, 3, -1, 2, 1]
    return A, b
end

# A method of each type with the given cap and a fixed forcing term.
fixed(::Type{BiCGStab}; maxiter = 30, kw...) = BiCGStab(; maxiter, kw...)
fixed(::Type{DQGMRES}; maxiter = 30, kw...) = DQGMRES(; maxiter, kw...)

workspace(::Type{BiCGStab}, g, x) = init_bicgstab_workspace(g, x)
workspace(::Type{DQGMRES}, g, x) = init_dqgmres_workspace(g, x)

function linear_solve!(::Type{BiCGStab}, ws, op, P, b, tol, maxiter)
    bicgstab_solve!(
        ws, op, P, b, rnorm(b), tol, maxiter)
end
function linear_solve!(::Type{DQGMRES}, ws, op, P, b, tol, maxiter)
    dqgmres_solve!(
        ws, op, P, b, rnorm(b), tol, maxiter)
end

# The sizes of every array field of a workspace, host and device.
function array_sizes(ws)
    [size(getfield(ws, f))
     for f in fieldnames(typeof(ws))
     if getfield(ws, f) isa AbstractArray]
end

# ─── The Newton step to the forcing tolerance ───────────────────────────────

@testset "The Newton step meets the forcing tolerance on the true residual, $M, $AT, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES, AT in ARRAY_BACKENDS

    η = T(1e-3)
    prob = NonlinearProblem(r5_diagonal!)
    x₀ = AT(T[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 1.25])
    m = Newton(; linsolve = fixed(M; maxiter = 20, forcing = FixedForcing(η)))
    solver = init(prob, copy(x₀), m)
    state = solver.state
    Fx = similar(x₀)
    r5_diagonal!(Fx, x₀, nothing)
    fnorm = rnorm(Fx)
    state = start!!(state, m, prob, x₀, Fx, nothing, fnorm)
    # the step writes its iterate into the one it is given
    x₁, _, _, info = step!!(state, m, prob, copy(x₀), copy(Fx), nothing)
    @test info.code == SUCCESS
    @test x₁ isa AT
    d = x₁ .- x₀
    @test rnorm(d) > 0
    # the true linear residual `‖F + J d‖`, with the Jacobian `3x² + 2` of the problem
    linres = rnorm(Fx .+ (3 .* x₀ .^ 2 .+ 2) .* d)
    @test linres <= η * fnorm * (1 + 100 * eps(T))

    x, st = solve(prob, copy(x₀), m)
    @test st.code == SUCCESS
    @test norm(Array(x) .- 1) <= 100 * sqrt(eps(T))
end

# The record of each step, which the Choice 1 forcing term of the next step reads: the step count,
# `‖F‖` of the step, its `η` and its linear residual. The residual `F = A x + x³/4 - b` of the
# nonsymmetric `A` has the Jacobian `A + 3/4 Diagonal(x²)`, and the inner solves of steps 1 and 2
# stop on the forcing test with a linear residual well above zero. `η₀ = 0.1` keeps the safeguard
# `η^φ` below `0.1` at step 2, and the smaller `η` of step 2 keeps it there at step 3, so each `η`
# after the first is the ratio of the rule. The full step of `Static` makes `d = x₊ - x` the
# direction of the inner solve. The `η` of steps 2 and 3 reads the residual that steps 1 and 2
# record, and there each method records the true residual to rounding: a DQGMRES window no narrower
# than the iterations of each inner solve (`n = 6`) makes its residual estimate that of GMRES, and
# the recurrence residual of BiCGStab after one or two iterations agrees with the true one.
@testset "Each step records the forcing term of the next, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    A, b = nonsymmetric(T)
    F!(r, x, q) = (mul!(r, A, x); r .+= x .^ 3 ./ 4 .- b; r)
    J(x) = A + Diagonal(3 .* x .^ 2 ./ 4)
    forcing = EisenstatWalkerChoice1(T; η₀ = 0.1, η_max = 0.9)
    g = M === DQGMRES ? DQGMRES(; maxiter = 30, memory = 30, forcing) :
        BiCGStab(; maxiter = 30, forcing)
    m = Newton(; linsolve = g, globalization = Static())
    prob = NonlinearProblem(F!)
    x = T[1, -1, 2, 0.5, -0.5, 1.5]
    solver = init(prob, copy(x), m)
    Fx = F!(similar(x), x, nothing)
    state = start!!(solver.state, m, prob, x, Fx, nothing, rnorm(Fx))
    fprev, ηprev, ℓprev = zero(T), zero(T), zero(T)
    for k in 1:3
        fk = norm(Fx)
        x₊, _, state, info = step!!(state, m, prob, copy(x), copy(Fx), nothing)
        @test info.code == SUCCESS
        ws = state.ws
        @test ws.step == k
        @test isapprox(ws.prev_fnorm, fk; rtol = 8 * eps(T))
        # `η` of the step from the norms and the true linear residual of the step before
        η = compute_forcing(forcing, k - 1, fk, fprev, ηprev, ℓprev)
        @test isapprox(ws.last_eta, η; rtol = sqrt(eps(T)))
        d = x₊ .- x
        @test norm(d) > 0
        ℓ = norm(Fx .+ J(x) * d)
        # the residual is not zero, so a record that drops it gives another `η`
        k < 3 && @test ℓ >= 10 * sqrt(eps(T)) * fk
        fprev, ηprev, ℓprev = fk, ws.last_eta, ℓ
        x = x₊
        Fx = F!(similar(x), x, nothing)
    end
end

# ─── Products per iteration, and a fixed workspace ──────────────────────────

@testset "BiCGStab takes two products per iteration, DQGMRES one, $T" for T in REAL_ELTYPES
    A, b = nonsymmetric(T)
    for k in (1, 2, 3)
        op = CountingMatrix(A)
        ws = init_bicgstab_workspace(BiCGStab(), b)
        _, info = bicgstab_solve!(ws, op, nothing, b, rnorm(b), T(1e-30), k)
        @test info.iterations == k
        @test !info.converged
        @test op.count[] == 2k

        op = CountingMatrix(A)
        ws = init_dqgmres_workspace(DQGMRES(; memory = 3), b)
        _, info = dqgmres_solve!(ws, op, nothing, b, rnorm(b), T(1e-30), k)
        @test info.iterations == k
        @test op.count[] == k
    end
end

# `F = 2x - 1`: the first half step solves it, so the Newton step takes one product.
@testset "BiCGStab exits at the half step with one product, $T" for T in REAL_ELTYPES
    counts = FixedCounts(0)
    jvp! = (Jv, x, v, q) -> (counts.jvp += 1; Jv .= 2 .* v; Jv)
    prob = NonlinearProblem((r, x, q) -> (r .= 2 .* x .- 1; r))
    m = Newton(; linsolve = BiCGStab(; forcing = FixedForcing(T(1e-6))),
        jacobian = JVPFunction(jvp!))
    x, st = solve(prob, T[3], m)
    @test st.code == SUCCESS
    @test st.iterations == 1
    @test counts.jvp == 1
    @test st.jvp_calls == 1
    @test Array(x) ≈ T[0.5]
end

# A window of two vectors, below the dimension of the system, so that 3k iterations would grow
# an uncapped basis.
memory_kw(::Type{BiCGStab}) = (;)
memory_kw(::Type{DQGMRES}) = (; memory = 2)

@testset "The workspace after 3k iterations has the sizes after k, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    A, b = nonsymmetric(T)
    k = 2
    g = fixed(M; memory_kw(M)...)
    ws = workspace(M, g, b)
    sizes₀ = array_sizes(ws)
    _, info = linear_solve!(M, ws, CountingMatrix(A), nothing, b, T(1e-30), k)
    @test info.iterations == k
    sizes_k = array_sizes(ws)
    _, info = linear_solve!(M, ws, CountingMatrix(A), nothing, b, T(1e-30), 3k)
    @test info.iterations == 3k
    @test array_sizes(ws) == sizes_k == sizes₀
end

@testset "A Newton solve counts the products of the inner iterations, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    counts = FixedCounts(0)
    fcalls = Ref(0)
    # a line search, whose trial evaluations count too
    prob = NonlinearProblem((r, x, q) -> (fcalls[] += 1; r5_diagonal!(r, x, q)))
    x₀ = T[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 1.25]
    m = Newton(; linsolve = fixed(M; maxiter = 20, forcing = FixedForcing(T(1e-4))),
        jacobian = JVPFunction(DiagonalJVP(counts)), globalization = Backtracking())
    solver = init(prob, copy(x₀), m)
    fcalls[] = 0
    _, st = solve!(solver, copy(x₀))
    @test st.code == SUCCESS
    @test st.jvp_calls == counts.jvp
    @test GeometricSolvers.jvp_calls(solver) == counts.jvp
    @test st.f_calls == fcalls[]
    @test GeometricSolvers.f_calls(solver) == fcalls[]
    @test 0 < GeometricSolvers.max_inner_iterations(solver) <= 20
end

# `Bisection` and `StrongWolfe` evaluate the slope of the merit, `φ′`, through a Jacobian–vector
# product, which `Backtracking` and `Static` never take: the status counts those products too, and
# the solve reaches the root `x = 1` of `x^3 + 2x - 3`.
@testset "A line search that measures the slope solves and counts, $M, $(nameof(G)), $T" for M in FIXED_METHODS,
    G in (Bisection, StrongWolfe),
    T in REAL_ELTYPES

    counts = FixedCounts(0)
    fcalls = Ref(0)
    prob = NonlinearProblem((r, x, q) -> (fcalls[] += 1; r5_diagonal!(r, x, q)))
    x₀ = T[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 1.25]
    m = Newton(; linsolve = fixed(M; maxiter = 20, forcing = FixedForcing(T(1e-4))),
        jacobian = JVPFunction(DiagonalJVP(counts)), globalization = G())
    solver = init(prob, copy(x₀), m)
    fcalls[] = 0
    x, st = solve!(solver, copy(x₀))
    @test st.code == SUCCESS
    @test st.jvp_calls == counts.jvp
    @test st.f_calls == fcalls[]
    @test st.jvp_calls > st.iterations
    # `F′ = 3x^2 + 2 ≥ 2` gives `|x_i - 1| ≤ |F_i(x)| / 2`, and the residual test of the solve
    # bounds `‖F(x)‖` by `√eps ‖F(x₀)‖`, in the max norm or the 2-norm
    F₀ = r5_diagonal!(similar(x₀), x₀, nothing)
    @test maximum(abs, x .- 1) <= sqrt(eps(T)) * norm(F₀) / 2 + 4 * eps(T)
end

# ─── The four guards of BiCGStab ────────────────────────────────────────────

# Each system makes one scalar of the recurrence exactly zero in Float32 and in Float64: `σ = r̂ᴴv`
# at the first product, `tᴴt` and `ω` at the second, and `ρ = r̂ᴴr` after the first iteration. In
# exact arithmetic `r̂ᴴs = ρ - ασ = 0`, so a zero `ω` leaves `ρ₊ = 0` and the guard on `ρ` stops
# the same iterate; on the second `ω` system rounding leaves `ρ₊ ≠ 0`, and only the guard on `ω`
# keeps `β = (ρ₊/ρ)(α/ω)` finite.
const GUARD_SYSTEMS = (
    "σ" => ([0 1; -1 0], [1, 0]),
    "ω" => ([1 1; -1 0], [1, 0]),
    "ω, ρ₊ ≠ 0" => ([2 0 2; 3 1 3; 3 2 2], [-1, -1, -1]),
    "tᴴt" => ([0 -1; 0 -1], [0, -1]),
    "ρ" => ([-1 0 -1; -1 -1 -1; 0 -1 -1], [1, 0, 0])
)

@testset "The BiCGStab guard on $name stops with a finite iterate, $T" for (name, (
        A0, b0)) in GUARD_SYSTEMS,
    T in REAL_ELTYPES

    A = T.(A0)
    b = T.(b0)
    ws = init_bicgstab_workspace(BiCGStab(), b)
    d, info = bicgstab_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), T(1e-6), 10)
    @test all(isfinite, d)
    @test info.breakdown
    @test !info.converged
    @test isfinite(info.resnorm)
    @test isapprox(info.resnorm, norm(b - A * d); rtol = 8 * eps(T), atol = 8 * eps(T))
end

# The same systems as the Newton problems `F(x) = A x - b` from zero: the step goes on with the
# direction of the guard, as after a `GMRES` breakdown on a singular operator. On the `ω` system the
# direction `α p̂ = e₁` leaves `‖b - A d‖ = ‖b‖`, and the line search takes the forcing term as the
# attained tolerance, which the ratio `1` of the direction exceeds; the solve then succeeds.
@testset "The Newton step goes on after the BiCGStab guard on $name, $T" for (name, (
        A0, b0)) in GUARD_SYSTEMS,
    T in REAL_ELTYPES

    A = T.(A0)
    b = T.(b0)
    prob = NonlinearProblem((r, x, q) -> (mul!(r, A, x); r .-= b; r))
    x, st = solve(prob, zeros(T, length(b)),
        Newton(; linsolve = BiCGStab(), globalization = Backtracking()); max_iterations = 5)
    @test st.code != NONFINITE
    @test all(isfinite, x)
    name == "ω" && @test st.code == SUCCESS
end

# ─── The conjugated inner products ──────────────────────────────────────────

# One BiCGStab iteration from zero: `α = bᴴb / bᴴAb`, `s = b - α A b`, `t = A s` and the
# least-squares `ω = tᴴs / tᴴt`, whose residual `s - ω t` is the smallest along `t`. An inner
# product without the conjugate gives another `α` or `ω`, and a larger residual.
@testset "BiCGStab conjugates its inner products, $T" for T in filter(T -> T <: Complex, ELTYPES)
    A = T[2+1im 1-2im 0; 0.5im 3 -1+1im; 1 -1im 1+2im]
    b = T[1 + 2im, -1im, 2 - 1im]
    α = dot(b, b) / dot(b, A * b)
    s = b - α * (A * b)
    t = A * s
    ω = dot(t, s) / dot(t, t)
    expected = norm(s - ω * t)
    ws = init_bicgstab_workspace(BiCGStab(), b)
    d, info = bicgstab_solve!(
        ws, CountingMatrix(A), nothing, b, rnorm(b), real(T)(1e-12), 1)
    @test info.iterations == 1
    @test isapprox(norm(b - A * d), expected; rtol = 100 * eps(real(T)))
    @test isapprox(info.resnorm, expected; rtol = 100 * eps(real(T)))
end

# `ρ₊ = r̂ᴴr` enters from the second iteration on, through `β`. On the complex system of the full
# DQGMRES window below, the conjugated recurrence meets `1e-5` within `CONJUGATED_CAP` iterations;
# without the conjugate it takes more than 40 in both precisions.
const CONJUGATED_CAP = 12

@testset "BiCGStab conjugates ρ over several iterations, $T" for T in filter(
    T -> T <: Complex, ELTYPES)
    A, b = nonsymmetric(T)
    A = A .+ T(0.5im) .* transpose(A)
    b = b .+ T(1im) .* reverse(b)
    tol = real(T)(1e-5)
    ws = init_bicgstab_workspace(BiCGStab(), b)
    d, info = bicgstab_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), tol,
        CONJUGATED_CAP)
    @test info.converged
    @test info.iterations > 1
    @test norm(b - A * d) <= 2 * tol * norm(b)
end

# ─── The right preconditioner ───────────────────────────────────────────────

# `M = Diagonal(scale)`: `refresh!` counts its calls, and `apply_preconditioner!` divides by the
# entries of `scale`, each of its own value.
struct DiagonalRightPreconditioner{V}
    count::Base.RefValue{Int}
    scale::V
end

function GeometricSolvers.refresh!(P::DiagonalRightPreconditioner, prob, x, Fx, p)
    P.count[] += 1
    return nothing
end

function GeometricSolvers.apply_preconditioner!(z, P::DiagonalRightPreconditioner, v)
    z .= v ./ P.scale
    return z
end

# A right-preconditioned solve returns `d = M⁻¹ u` for `A M⁻¹ u = b`, whose residual is the true
# `‖b - A d‖`; a direction that skips `M⁻¹` solves another system.
@testset "The right preconditioner gives the true residual, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    A, b = nonsymmetric(T)
    P = DiagonalRightPreconditioner(Ref(0), T[2, 0.5, 3, 1.5, 0.25, 4])
    tol = T(1e-4)
    ws = workspace(M, fixed(M), b)
    d, info = linear_solve!(M, ws, CountingMatrix(A), P, b, tol, 60)
    @test info.converged
    @test norm(b - A * d) <= 2 * tol * norm(b)
end

@testset "The forcing and preconditioner keywords of Newton take $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    P = DiagonalRightPreconditioner(Ref(0), T[2, 0.5, 3, 1.5, 0.25, 4, 1, 2])
    g = M === DQGMRES ? DQGMRES(; maxiter = 7, memory = 3) : BiCGStab(; maxiter = 7)
    m = Newton(; linsolve = g, forcing = FixedForcing(T(0.01)), preconditioner = P)
    @test m.linsolve isa M
    @test m.linsolve.maxiter == 7
    @test m.linsolve.forcing === FixedForcing(T(0.01))
    @test m.linsolve.preconditioner === P
    M === DQGMRES && @test m.linsolve.memory == 3
    # one refresh per Newton step, and the preconditioned solve reaches the root
    x₀ = fill(T(2), 8)
    x, st = solve(NonlinearProblem(r5_diagonal!), copy(x₀), m)
    @test st.code == SUCCESS
    @test P.count[] == st.iterations
    @test st.fnorm <= sqrt(eps(T)) * rnorm(r5_diagonal!(similar(x₀), x₀, nothing))
    # `F_i = x_i³ + 2x_i - 3` has `F_i(1) = 0` and `F_i' = 3x_i² + 2 ≥ 2`, so the mean value theorem
    # gives `|x_i - 1| ≤ |F_i(x)| / 2`. `F` evaluated in `T` is off by at most a few `eps(T)` times
    # the sum of the magnitudes of its terms.
    Fx = r5_diagonal!(similar(x), x, nothing)
    rounding = 8 * eps(T) * norm(abs.(x) .^ 3 .+ 2 .* abs.(x) .+ 3)
    @test norm(x .- 1) <= (norm(Fx) + rounding) / 2
end

# ─── The DQGMRES window ──────────────────────────────────────────────────────

@testset "The DQGMRES window is min(memory, n)" begin
    x = zeros(4)
    ws = init_dqgmres_workspace(DQGMRES(; memory = 30), x)
    @test size(ws.V) == (4, 5)
    @test size(ws.P) == (4, 4)
    ws = init_dqgmres_workspace(DQGMRES(; memory = 2), x)
    @test size(ws.V) == (4, 3)
    @test size(ws.P) == (4, 2)
    @test_throws r"^ArgumentError: DQGMRES memory must be ≥ 1$" DQGMRES(; memory = 0)
    @test_throws r"^ArgumentError: BiCGStab maxiter must be ≥ 1$" BiCGStab(; maxiter = 0)
    @test_throws r"^ArgumentError: DQGMRES maxiter must be ≥ 1$" DQGMRES(; maxiter = 0)
end

# With a window at least the iteration count, DQGMRES is unrestarted GMRES: the same iterate. A
# complex matrix has a skew part `0.5i` times the transpose, so that its inner products need the
# conjugate.
@testset "A full window gives the GMRES iterate, $T" for T in ELTYPES
    A, b = nonsymmetric(T)
    if T <: Complex
        A = A .+ T(0.5im) .* transpose(A)
        b = b .+ T(1im) .* reverse(b)
    end
    tol = real(T)(1e-5)
    ws = init_dqgmres_workspace(DQGMRES(; memory = 6), b)
    d, info = dqgmres_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), tol, 6)
    wg = GeometricSolvers.init_gmres_workspace(GMRES(; maxiter = 6, mgs = true), b)
    dg, infog = GeometricSolvers.gmres_solve!(
        wg, CountingMatrix(A), nothing, b, rnorm(b), tol, 6, 6; mgs = true)
    @test info.iterations == infog.iterations
    @test norm(d - dg) <= 100 * eps(real(T)) * norm(dg)
    @test norm(b - A * d) <= tol * norm(b) * (1 + 100 * eps(real(T)))
end

# A window below the iteration count orthogonalises against the last `memory` vectors only, and
# the residual it reports bounds the true one.
@testset "A short window reports a bound of the true residual, $T" for T in REAL_ELTYPES
    A, b = nonsymmetric(T)
    tol = T(1e-4)
    ws = init_dqgmres_workspace(DQGMRES(; memory = 2), b)
    d, info = dqgmres_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), tol, 60)
    @test info.converged
    @test info.iterations > 3
    @test norm(b - A * d) <= info.resnorm * (1 + 100 * eps(T))
    @test norm(b - A * d) <= tol * norm(b) * (1 + 100 * eps(T))
end

# ─── The decided edges ───────────────────────────────────────────────────────

@testset "b = 0 takes no product, $M, $T" for M in FIXED_METHODS, T in REAL_ELTYPES

    A, _ = nonsymmetric(T)
    b = zeros(T, 6)
    op = CountingMatrix(A)
    ws = workspace(M, fixed(M), b)
    d, info = linear_solve!(M, ws, op, nothing, b, T(1e-6), 10)
    @test op.count[] == 0
    @test iszero(d)
    @test info.converged
    @test info.iterations == 0
end

@testset "F(x₀) = 0 takes no product, $M, $T" for M in FIXED_METHODS, T in REAL_ELTYPES

    counts = FixedCounts(0)
    m = Newton(; linsolve = fixed(M), jacobian = JVPFunction(DiagonalJVP(counts)))
    _, st = solve(NonlinearProblem(r5_diagonal!), ones(T, 4), m)
    @test st.code == SUCCESS
    @test st.iterations == 0
    @test counts.jvp == 0
    @test st.jvp_calls == 0
end

@testset "A NaN product gives NONFINITE and a finite iterate, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    jvp! = (Jv, x, v, q) -> (Jv .= T(NaN); Jv)
    m = Newton(; linsolve = fixed(M), jacobian = JVPFunction(jvp!))
    x₀ = fill(T(2), 4)
    x, st = solve(NonlinearProblem(r5_diagonal!), copy(x₀), m)
    @test st.code == NONFINITE
    @test x == x₀
    # the solve itself stops with a residual of `NaN` and a finite direction
    A = fill(T(NaN), 4, 4)
    ws = workspace(M, fixed(M), x₀)
    _, info = linear_solve!(M, ws, CountingMatrix(A), nothing, x₀, T(1e-6), 10)
    @test isnan(info.resnorm)
    @test !info.converged
end

# An infinite entry of `b` gives `‖b‖ = Inf`, so that no residual meets `tol ⋅ ‖b‖`: the solve
# reports no convergence and a residual of `NaN`.
@testset "An infinite right-hand side does not converge, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    A, b = nonsymmetric(T)
    b[2] = T(Inf)
    ws = workspace(M, fixed(M), b)
    _, info = linear_solve!(M, ws, CountingMatrix(A), nothing, b, T(1e-6), 10)
    @test !info.converged
    @test isnan(info.resnorm)
end

# An operator whose product number `bad` is `NaN` and every other is `A v`.
function nan_at(A, bad)
    count = Ref(0)
    return CountingMatrix(A, count),
    (Jv, v) -> begin
        count[] += 1
        count[] == bad ? (Jv .= NaN) : mul!(Jv, A, v)
        Jv
    end
end

# A non-finite residual norm ends the solve in the iteration that formed it: at the half step after
# one product, at the full step after two.
@testset "A non-finite product ends the iteration that takes it, $T" for T in REAL_ELTYPES
    A, b = nonsymmetric(T)
    for bad in (1, 2)
        counter, op = nan_at(A, bad)
        ws = init_bicgstab_workspace(BiCGStab(), b)
        _, info = bicgstab_solve!(ws, op, nothing, b, rnorm(b), T(1e-12), 10)
        @test !isfinite(info.resnorm)
        @test !info.converged
        @test info.iterations == 1
        @test counter.count[] == bad
    end
    counter, op = nan_at(A, 1)
    ws = init_dqgmres_workspace(DQGMRES(), b)
    _, info = dqgmres_solve!(ws, op, nothing, b, rnorm(b), T(1e-12), 10)
    @test isnan(info.resnorm)
    @test info.iterations == 1
    @test counter.count[] == 1
end

# A residual norm equal to the tolerance meets it. `b = e₁` has the norm 1 exactly, so the residual
# norm that one capped solve reports is the relative tolerance that a second solve meets.
@testset "A residual equal to the tolerance converges, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    A, _ = nonsymmetric(T)
    b = T[1, 0, 0, 0, 0, 0]
    ws = workspace(M, fixed(M), b)
    _, info = linear_solve!(M, ws, CountingMatrix(A), nothing, b, T(1e-30), 1)
    @test !info.converged
    op = CountingMatrix(A)
    _, again = linear_solve!(M, ws, op, nothing, b, info.resnorm, 1)
    @test again.converged
    @test again.resnorm == info.resnorm
    # the full step of BiCGStab, which takes the second product
    M === BiCGStab && @test op.count[] == 2
end

# `A = [0 1; 0 0]` maps `e₁` to zero: the first Krylov vector and the rotated pivot are both zero,
# so DQGMRES has no direction, keeps the zero iterate and reports no convergence.
@testset "A zero pivot of DQGMRES keeps the iterate, $T" for T in REAL_ELTYPES
    A = T[0 1; 0 0]
    b = T[1, 0]
    ws = init_dqgmres_workspace(DQGMRES(), b)
    d, info = dqgmres_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), T(1e-6), 10)
    @test iszero(d)
    @test info.breakdown
    @test !info.converged
    @test info.resnorm == norm(b - A * d)
    @test info.iterations == 1
end

# `A = 2I` leaves the Krylov space of any `b` invariant after one product: the next Krylov vector
# is zero, the pivot is not, and DQGMRES ends with the solution and a breakdown, as GMRES does.
# `b / ‖b‖ = ±1/2` is exact, so the next Krylov vector is zero in floating point too.
@testset "An invariant Krylov space ends DQGMRES with the solution, $T" for T in REAL_ELTYPES
    A = T[2 0 0 0; 0 2 0 0; 0 0 2 0; 0 0 0 2]
    b = T[1, -1, 1, -1]
    ws = init_dqgmres_workspace(DQGMRES(), b)
    d, info = dqgmres_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), T(1e-6), 10)
    @test info.iterations == 1
    @test info.breakdown
    @test info.converged
    @test isapprox(d, b ./ 2; rtol = 8 * eps(T))
end

# `F = atan(x)` from `1.3`: the full Newton step overshoots to a residual `0.94` of the first. A
# one-dimensional solve with one product is exact, so the attained tolerance is zero, and the
# inexact test with `c₁ = 0.5` bounds the full step by `1 - c₁ = 0.5`, which rejects it; the
# forcing term `0.9` in its place would give `1 - c₁ (1 - 0.9) = 0.95`, which accepts it.
@testset "The line search takes the attained tolerance, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    prob = NonlinearProblem((r, x, q) -> (r .= atan.(x); r))
    x₀ = T[1.3]
    m = Newton(; linsolve = fixed(M; maxiter = 4, forcing = FixedForcing(T(0.9))),
        globalization = Backtracking(; c₁ = T(0.5), p = T(0.5)))
    solver = init(prob, copy(x₀), m)
    Fx = atan.(x₀)
    fnorm₀ = rnorm(Fx)
    state = start!!(solver.state, m, prob, x₀, Fx, nothing, fnorm₀)
    x₁, Fx₁, state₁, info = step!!(state, m, prob, copy(x₀), copy(Fx), nothing)
    d = state₁.ws.d
    @test info.code == SUCCESS
    @test abs(Fx[1] + d[1] / (1 + x₀[1]^2)) <= 4 * eps(T) * fnorm₀
    # the full step, which the forcing term's bound accepts and the attained tolerance's rejects
    ratio = abs(atan(x₀[1] + d[1])) / fnorm₀
    @test 1 - m.globalization.c₁ < ratio <= 1 - m.globalization.c₁ * (1 - T(0.9))
    # so the accepted step is a backtracked one
    α = (x₁[1] - x₀[1]) / d[1]
    @test zero(T) < α < one(T)
    @test rnorm(Fx₁) <= (1 - m.globalization.c₁ * α) * fnorm₀ + 4 * eps(T) * fnorm₀
end

@testset "A step at the cap counts a step-rule failure, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    x₀ = T[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 1.25]
    m = Newton(; linsolve = fixed(M; maxiter = 1, forcing = FixedForcing(T(1e-10))))
    solver = init(NonlinearProblem(r5_diagonal!), copy(x₀), m)
    _, st = solve!(solver, copy(x₀))
    @test st.step_failures >= 1
    @test GeometricSolvers.max_inner_iterations(solver) == 1
    # a cap that the solve meets counts none
    m = Newton(; linsolve = fixed(M; maxiter = 20, forcing = FixedForcing(T(1e-6))))
    _, st = solve(NonlinearProblem(r5_diagonal!), copy(x₀), m)
    @test st.code == SUCCESS
    @test st.step_failures == 0
end

# `start!!` sets the values that a step reads, and `init` leaves them `NaN`, so a step on a state
# that no `start!!` saw fails closed and keeps its iterate.
@testset "A step without start!! fails closed, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    x₀ = fill(T(2), 4)
    m = Newton(; linsolve = fixed(M))
    prob = NonlinearProblem(r5_diagonal!)
    state = init(prob, copy(x₀), m).state
    Fx = similar(x₀)
    r5_diagonal!(Fx, x₀, nothing)
    x, _, _, info = step!!(state, m, prob, copy(x₀), Fx, nothing)
    @test info.code == NONFINITE
    @test x == x₀
end

# ─── Allocations and inference ───────────────────────────────────────────────

# An `@allocated` assertion measures through a function barrier whose arguments have concrete types
# and which calls the function once before it measures.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

# The diagonal operator `J v = d .* v` of a solve, a struct and not a closure.
struct DiagonalOperator{D}
    d::D
end

(op::DiagonalOperator)(Jv, v) = (Jv .= op.d .* v; Jv)

@testset "The inner solve allocates nothing after init and infers, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    b = T[1, -2, 3, -1, 2, 1]
    op = DiagonalOperator(T[1, 2, 3, 4, 5, 6])
    ws = workspace(M, fixed(M; memory_kw(M)...), b)
    inner = bb -> linear_solve!(M, ws, op, nothing, bb, T(1e-10), 8)
    @inferred linear_solve!(M, ws, op, nothing, b, T(1e-10), 8)
    @test allocations(collect, b) > 0
    # skipped where a run forces `--check-bounds=yes`
    if Base.JLOptions().check_bounds != 1
        @test allocations(inner, b) == 0
    end
end

@testset "The whole solve allocates nothing after init, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    x₀ = T[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 1.25]
    for globalization in (GeometricSolvers.Static(), Backtracking())
        m = Newton(; linsolve = fixed(M; maxiter = 20), globalization)
        solver = init(NonlinearProblem(r5_diagonal!), copy(x₀), m)
        work = similar(x₀)
        run = w -> (copyto!(w, x₀); solve!(solver, w))
        _, st = run(work)
        @test st.code == SUCCESS
        if Base.JLOptions().check_bounds != 1
            @test allocations(run, work) == 0
        end
    end
end

# ─── The scale of the scalar recurrences ─────────────────────────────────────

# `A` and `b` scaled together keep the solution: the scalars of the recurrences must neither
# overflow at `4√floatmax` nor at `floatmax / 8`.
@testset "The solution is that of the unscaled system at scale $label, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES,
    (label, scale) in (("4√floatmax", T -> 4 * sqrt(floatmax(T))),
        ("floatmax/8", T -> floatmax(T) / 8))

    A0 = T[4 1 0.5 0.2
           0.3 3 0.8 0.1
           0.2 0.4 2.5 0.6
           0.1 0.2 0.3 2]
    # `‖A0‖₂ < 5` and `‖b0‖₂ < 2`, so that `‖A‖` and `‖b‖` stay finite at `floatmax / 8`
    x_true = T[0.25, -0.5, 0.125, 0.5]
    b0 = A0 * x_true
    c = scale(T)
    A = c .* A0
    b = c .* b0
    @test all(isfinite, A) && all(isfinite, b)
    tol = T(10) * sqrt(eps(T))
    ws = workspace(M, fixed(M), b)
    d, info = linear_solve!(M, ws, CountingMatrix(A), nothing, b, tol, 40)
    @test info.converged
    @test all(isfinite, d)
    @test norm(d - x_true) <= 1000 * tol * norm(x_true)
end

# `A = 2^k A0` with `k = -(emax + 3)`, `emax = exponent(floatmax(T))`, has subnormal entries, and
# `b = 2^-(emax ÷ 4) b0` keeps the solution `2^-(k + emax ÷ 4) A0⁻¹ b0` far below `floatmax`. The
# scalars `α`, `ω` and `1 / r_jj` of the unscaled recurrences are about `2^-k`, above `floatmax`.
# The reference is the `Float64` solve of the unscaled system, scaled exactly; its error is far
# below `tol`.
@testset "The solution at a subnormal scale of J, $M, $T" for M in FIXED_METHODS,
    T in REAL_ELTYPES

    A0 = [4 1 0; 1 3 1; 0 1 2]
    b0 = [1, -2, 1]
    emax = exponent(floatmax(T))
    k = -(emax + 3)
    sb = -(emax ÷ 4)
    A = ldexp.(T.(A0), k)
    b = ldexp.(T.(b0), sb)
    @test all(a -> iszero(a) || issubnormal(a), A)
    x_ref = ldexp.(Float64.(A0) \ Float64.(b0), sb - k)
    tol = T(10) * sqrt(eps(T))
    # a relative residual `tol` bounds the relative error by `cond(A0) tol`
    bound = 2 * LinearAlgebra.cond(Float64.(A0)) * tol
    ws = workspace(M, fixed(M), b)
    d, info = linear_solve!(M, ws, CountingMatrix(A), nothing, b, tol, 40)
    @test info.converged
    @test all(isfinite, d)
    @test norm(Float64.(d) - x_ref) <= bound * norm(x_ref)
    # the Newton step, whose residual test is `‖F‖ ≤ √eps ‖F(x₀)‖`
    prob = NonlinearProblem((r, x, q) -> (mul!(r, A, x); r .-= b; r))
    jvp! = (Jv, x, v, q) -> mul!(Jv, A, v)
    x, st = solve(prob, zeros(T, 3), Newton(; linsolve = fixed(M), jacobian = JVPFunction(jvp!)))
    @test st.code == SUCCESS
    @test norm(Float64.(x) - x_ref) <=
          2 * LinearAlgebra.cond(Float64.(A0)) * sqrt(eps(T)) * norm(x_ref)
end

# One dimension, where the coefficient of the step overflows unscaled and the solution does not.
# BiCGStab: `J = 1`, a right preconditioner that divides by `2^8` and `b = 2^(emax - 4)`, so that
# `bnorm α = 2^(emax + 4)` and the solution is `b`. DQGMRES: `J = c = 2^-(emax + 1)`, a subnormal,
# and `b = 2^-3`, so that the direction `z / r_jj = 2^(emax + 1)` and the solution is
# `b / c = 2^(emax - 2)`. Each solve is exact in one iteration.
@testset "A one-dimensional step whose unscaled coefficient overflows, $T" for T in REAL_ELTYPES
    emax = exponent(floatmax(T))
    b = [ldexp(one(T), emax - 4)]
    P = DiagonalRightPreconditioner(Ref(0), [ldexp(one(T), 8)])
    ws = init_bicgstab_workspace(BiCGStab(), b)
    d, info = bicgstab_solve!(ws, CountingMatrix(ones(T, 1, 1)), P, b, rnorm(b), T(1e-3), 5)
    @test info.converged
    @test isapprox(d, b; rtol = 8 * eps(T))
    c = ldexp(one(T), -(emax + 1))
    @test issubnormal(c)
    b = [ldexp(one(T), -3)]
    ws = init_dqgmres_workspace(DQGMRES(), b)
    d, info = dqgmres_solve!(
        ws, CountingMatrix(fill(c, 1, 1)), nothing, b, rnorm(b), T(1e-3), 5)
    @test info.converged
    @test isapprox(d, [ldexp(one(T), emax - 2)]; rtol = 8 * eps(T))
end

# `β = ρq (α / ω) 2^Δ` of the scaled BiCGStab recurrence: with `ω = 2^-(emax + 3)` and `Δ = -4`,
# `α / ω` is above `floatmax` and `β = 2^(emax - 1)` is not; with `Δ ≥ 0` the shift grows, and
# `β` overflows exactly where its value does.
@testset "BiCGStab forms β at both signs of the shift, $T" for T in REAL_ELTYPES
    emax = exponent(floatmax(T))
    ω = ldexp(one(T), -(emax + 3))
    @test isapprox(GeometricSolvers.bicgstab_beta(one(T), one(T), ω, -4),
        ldexp(one(T), emax - 1); rtol = 8 * eps(T))
    @test isapprox(GeometricSolvers.bicgstab_beta(T(0.5), one(T), T(0.125), 2), T(16);
        rtol = 8 * eps(T))
    @test isinf(GeometricSolvers.bicgstab_beta(one(T), one(T), ω, 4))
end

# The next direction `r + β (p - ω 2^-Δ v)`: with `ω = 1` and `Δ = -(emax + 4)`, `ω 2^-Δ` is above
# `floatmax` while `β ω 2^-Δ = ρq α` is not; with `Δ = emax + 3` it is below `floatmin`. The
# reference is `r + β p - ρq α v` in `Float64` of the same values; in the normal range, `Δ = 1`,
# the direction is the unscaled one, `r + β (p - (ω / 2) v)`.
@testset "The BiCGStab direction at both signs of the shift, $T" for T in REAL_ELTYPES
    emax = exponent(floatmax(T))
    r = T[0.5, -1]
    v = T[1, 1]
    for (α, ω, Δ) in ((T(0.75), one(T), -(emax + 4)), (
        ldexp(one(T), -(emax + 3)), one(T), emax + 3))
        ρq = T(1.5)
        β = GeometricSolvers.bicgstab_beta(ρq, α, ω, Δ)
        @test isfinite(β)
        p = T[1, 2]
        ref = Float64.(r) .+ Float64(β) .* Float64.(p) .-
              Float64(ρq) * Float64(α) .* Float64.(v)
        GeometricSolvers.bicgstab_direction!(p, r, v, β, ρq, α, ω, Δ)
        @test all(isfinite, p)
        @test isapprox(Float64.(p), ref; rtol = 8 * eps(T))
    end
    p = T[1, 2]
    β = GeometricSolvers.bicgstab_beta(T(1.5), T(0.75), one(T), 1)
    GeometricSolvers.bicgstab_direction!(p, r, v, β, T(1.5), T(0.75), one(T), 1)
    @test isapprox(p, r .+ β .* (T[1, 2] .- T(0.5) .* v); rtol = 8 * eps(T))
end

# `J = [1 1/2; δ 1]` and `b = e₁` with `δ = 2^-(emax + 2)`: the half step leaves `s = [0, -δ]`,
# so `ω ≈ 0.8δ` on the scaled vectors and the shift `Δ` is about `-(emax + 2)`; the scaled
# `α / ω` is above `floatmax`, while `β ≈ δ / 2`. Two iterations solve the system to rounding.
@testset "BiCGStab solves where the scaled α / ω overflows, $T" for T in REAL_ELTYPES
    emax = exponent(floatmax(T))
    δ = ldexp(one(T), -(emax + 2))
    A = T[1 0.5; δ 1]
    b = T[1, 0]
    ws = init_bicgstab_workspace(BiCGStab(), b)
    d, info = bicgstab_solve!(ws, CountingMatrix(A), nothing, b, rnorm(b), zero(T), 2)
    @test info.iterations == 2
    @test isfinite(info.resnorm)
    @test all(isfinite, d)
    @test norm(b - A * d) <= 8 * eps(T) * norm(b)
end

# DQGMRES with a right preconditioner that divides by `2^8`, at `b` near `floatmax / 2^5`: the
# pivots `r_jj` carry the scale `2^-8` of the preconditioner and the stored directions do not, so
# a coefficient `g_j / r_jj` of the unscaled direction lies above `floatmax`, while the solution
# `A⁻¹ b` does not. One and two dimensions, the second through the band term of the recurrence; a
# full window solves each to the rounding of `κ(A) < 3`.
@testset "DQGMRES with a preconditioner of small scale at a large b, n = $n, $T" for n in (1, 2),
    T in REAL_ELTYPES

    emax = exponent(floatmax(T))
    A0 = n == 1 ? fill(1, 1, 1) : [2 1; 1 3]
    b0 = n == 1 ? [1] : [1, -1]
    A = T.(A0)
    b = ldexp.(T.(b0), emax - 6)
    x_ref = ldexp.(Float64.(A0) \ Float64.(b0), emax - 6)
    P = DiagonalRightPreconditioner(Ref(0), fill(ldexp(one(T), 8), n))
    ws = init_dqgmres_workspace(DQGMRES(), b)
    d, info = dqgmres_solve!(ws, CountingMatrix(A), P, b, rnorm(b), 10 * eps(T), 5)
    @test info.converged
    @test all(isfinite, d)
    @test norm(Float64.(d) - x_ref) <= 32 * eps(T) * norm(x_ref)
end

# With `regularization = μ`, the inner solve takes `J + μ I`, and the line search takes the slope
# of `‖F‖²`, which `J` gives: one step of a matrix-free `Newton` matches that of the dense `Newton`
# with `LUFactorization`, whose line search reads `J`. The bound is the inner tolerance `10 eps`
# times `κ(J + μ I) < 4` on this start, with a factor for the rounding of the line search; with the
# slope of `J + μ I` the steps differ by more than `9000 eps`.
@testset "A regularized matrix-free step takes the slope of ‖F‖², $(nameof(L)), $(nameof(typeof(g))), $T" for L in (
        GMRES, BiCGStab, DQGMRES, MINRESQLP),
    g in (Bisection(), StrongWolfe(; c₁ = 0.4)),
    T in REAL_ELTYPES

    x₀ = T[0.5, 1.5, 2.5, 0.25]
    μ = T(10)
    opt = GeometricSolvers.Options(T; max_iterations = 1)
    prob = if L === MINRESQLP
        GeometricSolvers.StationarityProblem((x, q) -> sum(x .^ 4) / 4 + sum(x .^ 2) / 2 +
                                                       T(0.1) * sum(x .* circshift(x, 1)) -
                                                       sum(x))
    else
        NonlinearProblem((r, x, q) -> (
            r .= x .^ 3 .+ 2 .* x .- 3 .+
                 T(0.3) .* circshift(x, 1); r))
    end
    xd, _ = solve(prob, copy(x₀), Newton(; regularization = μ, globalization = g), nothing;
        options = opt)
    m = Newton(; linsolve = L(; forcing = FixedForcing(10 * eps(T))), regularization = μ,
        globalization = g)
    xm, _ = solve(prob, copy(x₀), m, nothing; options = opt)
    @test xm != x₀
    @test norm(xm - xd) <= 1000 * eps(T) * norm(xd - x₀)
end

# ─── The refusals of the trait ──────────────────────────────────────────────

@testset "The matrix_free trait" begin
    @test matrix_free(GMRES())
    @test matrix_free(MINRESQLP())
    @test matrix_free(BiCGStab())
    @test matrix_free(DQGMRES())
    @test !matrix_free(LUFactorization())
    @test !matrix_free(QRFactorization())
end

@testset "Newton with $M refuses DogLeg, reuse and a factor's options" for M in FIXED_METHODS
    @test_throws r"^ArgumentError: Newton with a DogLeg globalization needs GMRES among the matrix-free linear methods" Newton(;
        linsolve = fixed(M), globalization = DogLeg())
    @test_throws r"^ArgumentError: Newton with a matrix-free linear method takes no factor reuse" Newton(;
        linsolve = fixed(M), reuse = 2)
    @test_throws r"^ArgumentError: Newton with a matrix-free linear method takes no refinement" Newton(;
        linsolve = fixed(M), refinement = Refinement())
    @test_throws r"^ArgumentError: Newton with a LevenbergMarquardt globalization needs a Jacobian matrix, which a matrix-free solve does not form$" Newton(;
        linsolve = fixed(M), globalization = LevenbergMarquardt())
    @test_throws r"^ArgumentError: Newton with a PseudoTransient globalization needs a factorising linear method" Newton(;
        linsolve = fixed(M), globalization = PseudoTransient(0.1))
    jvp! = (Jv, x, v, q) -> (Jv .= v; Jv)
    @test Newton(; linsolve = fixed(M), jacobian = JVPFunction(jvp!)) isa Newton
end

@testset "Continuation refuses a Newton with $(nameof(typeof(linsolve)))" for linsolve in (
    BiCGStab(), DQGMRES(), KroneckerFactorization())
    @test_throws r"^ArgumentError: Continuation needs a Newton method whose step forms the Jacobian matrix" Continuation(
        FixedPointHomotopy(); method = Newton(; linsolve))
end

@testset "KroneckerFactorization refuses the block method $(nameof(typeof(block)))" for block in (
    BiCGStab(), DQGMRES(), MINRESQLP(), KroneckerFactorization())
    @test_throws r"^ArgumentError: KroneckerFactorization takes a block method that factorises in the working precision" KroneckerFactorization(;
        block)
end

# The refusals of the state that a `Newton` with `BiCGStab` or `DQGMRES` shares with `GMRES` name
# the method.
@testset "Newton with $M refuses what has no square vector system" for M in FIXED_METHODS
    name = string(nameof(M))
    m = Newton(; linsolve = fixed(M))
    @test_throws Regex("^ArgumentError: a Newton with $name solves the square system of a NonlinearProblem") init(
        LeastSquaresProblem((r, x, q) -> (r .= x .- 1), 2), zeros(2), m)
    @test_throws Regex("^ArgumentError: a Newton with $name solves an array linear system") init(
        NonlinearProblem((x, q) -> x .- 1), SVector(0.0, 0.0), m)
    prob = NonlinearProblem((r, x, p) -> (r .= x .^ 3 .- 2; nothing))
    @test_throws r"^ArgumentError: a Newton with a matrix-free linear method takes jacobian = nothing, an AbstractADType or a JVPFunction" solve(
        prob, [1.0, 1.5], Newton(; linsolve = fixed(M), jacobian = (J, x, p) -> J))
    @test_throws Regex("^ArgumentError: LeftPreconditioned does not take a Newton with $name as its outer method") solve(
        prob, [1.0, 1.5], LeftPreconditioned(m, Newton()))
    # a Newton with the method as the inner map is taken
    x, st = solve(prob, [1.0, 1.5], LeftPreconditioned(Newton(), m); max_iterations = 50)
    @test st.code == SUCCESS
    @test x ≈ fill(cbrt(2.0), 2) rtol = 1e-6
end

@testset "solve refuses a dual input for a Newton with $M" for M in FIXED_METHODS
    calls = Ref(0)
    prob = NonlinearProblem((r, x, q) -> begin
        calls[] += 1
        @. r = x^3 + 2x - q
        r
    end)
    total(q, m) = sum(first(solve(prob, [1.0, 1.0], m, q)))
    @test_throws r"^ArgumentError: solve does not accept ForwardDiff.Dual inputs for a Newton with a matrix-free linear method" ForwardDiff.derivative(
        q -> total(q, Newton(; linsolve = fixed(M))), 3.0)
    @test calls[] == 0
end

@testset "A BatchEliminated problem refuses $M by its own phrase" for M in FIXED_METHODS
    inner = NonlinearProblem((y, c) -> y .- c)
    moments = DepositedMoments(zeros(2), zeros(2), zeros(2), zeros(2))
    batch = BatchEliminated((F, x, m, s) -> F, inner, [SVector(0.0) for _ in 1:3],
        (m, Y, s, ps) -> m, moments)
    @test_throws r"^ArgumentError: a BatchEliminated problem takes GMRES among the matrix-free linear methods" GeometricSolvers.init_state(
        Newton(; linsolve = fixed(M)), batch, zeros(2), nothing)
    # a factorising method is refused by the phrase of the matrix-free test
    @test_throws r"^ArgumentError: a BatchEliminated problem takes a Newton with GMRES" GeometricSolvers.init_state(
        Newton(), batch, zeros(2), nothing)
end
