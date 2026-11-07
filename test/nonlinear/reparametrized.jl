using Adapt: Adapt
using CommonSolve: CommonSolve
using ForwardDiff: ForwardDiff
using GPUArraysCore: @allowscalar, allowscalar
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: Diagonal, I, norm, transpose
using SparseArrays: nonzeros, sparse
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: AbstractNonlinearProblem, AutoForwardDiff, Backtracking,
                        LDLTFactorization, LUFactorization, MAXITERS, Newton,
                        NonlinearProblem, Options, Picard, Reparametrized,
                        ReparametrizedSolver, SUCCESS, SolverStatus, StationarityProblem,
                        Static, ToReal, gradient_hessian!!, init, jacobian!!, jvp!!,
                        prepare_ad, residual!!, solve, solve!, solve!!, solve_batch!

const GS = GeometricSolvers

include("../helpers/matrix.jl")

allowscalar(false)

# Allocation measured through a barrier, after one call that compiles the path.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

# Polar chart x = g(r, θ) = (r cos θ, r sin θ) and its analytic Jacobian G(r, θ).
g_polar(y, p) = SVector(y[1] * cos(y[2]), y[1] * sin(y[2]))

function dg_polar(y, p)
    r, θ = y[1], y[2]
    s, c = sincos(θ)
    return SMatrix{2, 2, typeof(r), 4}(c, s, -r * s, r * c)
end

function dg_polar_wrong(y, p)
    r, θ = y[1], y[2]
    s, c = sincos(θ)
    return SMatrix{2, 2, typeof(r), 4}(-c, s, -r * s, r * c)
end

function f_polar_target(x, p)
    T = eltype(x)
    return SVector(x[1]^2 + x[2]^2 - T(4), x[1] - x[2])
end

function j_polar_target(x, p)
    T = eltype(x)
    return SMatrix{2, 2, T, 4}(2 * x[1], one(T), 2 * x[2], -one(T))
end

# R3 in-place polar chart and target residual: zero-allocation on Array, broadcast on JLArray.
function g_polar!(x::Array, y::Array, p)
    @inbounds begin
        r, θ = y[1], y[2]
        s, c = sincos(θ)
        x[1] = r * c
        x[2] = r * s
    end
    return x
end

function g_polar!(x, y, p)
    T = eltype(y)
    c1 = JLArray(T[1, 0])
    c2 = JLArray(T[0, 1])
    r = sum(y .* c1)
    θ = sum(y .* c2)
    s, c = sincos(θ)
    x .= (r * c) .* c1 .+ (r * s) .* c2
    return x
end

function dg_polar!(G::Array, y::Array, p)
    @inbounds begin
        r, θ = y[1], y[2]
        s, c = sincos(θ)
        G[1, 1] = c
        G[2, 1] = s
        G[1, 2] = -r * s
        G[2, 2] = r * c
    end
    return G
end

function dg_polar!(G, y, p)
    T = eltype(y)
    c1 = JLArray(T[1, 0])
    c2 = JLArray(T[0, 1])
    r = sum(y .* c1)
    θ = sum(y .* c2)
    s, c = sincos(θ)
    copyto!(G, T[c (-r * s); s (r * c)])
    return G
end

function dg_polar_wrong!(G::Array, y::Array, p)
    @inbounds begin
        r, θ = y[1], y[2]
        s, c = sincos(θ)
        G[1, 1] = -c
        G[2, 1] = s
        G[1, 2] = -r * s
        G[2, 2] = r * c
    end
    return G
end

function f_polar_target!(F::Array, x::Array, p)
    T = eltype(x)
    @inbounds begin
        x1, x2 = x[1], x[2]
        F[1] = x1^2 + x2^2 - T(4)
        F[2] = x1 - x2
    end
    return F
end

function f_polar_target!(F, x, p)
    T = eltype(x)
    m1 = JLArray(T[1, 0])
    m2 = JLArray(T[0, 1])
    x1 = sum(x .* m1)
    x2 = sum(x .* m2)
    f1 = x1^2 + x2^2 - T(4)
    f2 = x1 - x2
    F .= f1 .* m1 .+ f2 .* m2
    return F
end

# A chart with a dense Jacobian, so a chunk of columns that the AD loop does not fill stays at
# zero and no other column moves.
function g_chunk!(x, y, p)
    x .= y .^ 2 .+ sin.(y) .+ 0.1 * sum(y)
    return x
end

# The identity residual: its Jacobian is `I`, so the non-covariant chart Jacobian is exactly `G`.
f_identity!(F, x, p) = copyto!(F, x)

function action_polar(x, p)
    T = eltype(x)
    return T(1 // 4) * (x[1]^2 + x[2]^2 - T(4))^2 + T(1 // 2) * (x[1] - x[2])^2
end

function action_polar_r3(x::Array, p)
    T = eltype(x)
    @inbounds begin
        x1, x2 = x[1], x[2]
        return T(1 // 4) * (x1^2 + x2^2 - T(4))^2 + T(1 // 2) * (x1 - x2)^2
    end
end

function action_polar_r3(x, p)
    T = eltype(x)
    m1 = JLArray(T[1, 0])
    m2 = JLArray(T[0, 1])
    x1 = sum(x .* m1)
    x2 = sum(x .* m2)
    return T(1 // 4) * (x1^2 + x2^2 - T(4))^2 + T(1 // 2) * (x1 - x2)^2
end

# A convex quadratic action has one critical point, so a chart solve and an x-space solve name
# the same x* and a solve test can assert it without a second solve.
const X_QUAD = SVector(1.5, 0.75)

function action_quad(x, p)
    T = eltype(x)
    return (x[1] - T(3 // 2))^2 + 2 * (x[2] - T(3 // 4))^2
end

function action_quad_r3(x::Array, p)
    T = eltype(x)
    @inbounds begin
        x1, x2 = x[1], x[2]
        return (x1 - T(3 // 2))^2 + 2 * (x2 - T(3 // 4))^2
    end
end

function action_quad_r3(x, p)
    T = eltype(x)
    m1 = JLArray(T[1, 0])
    m2 = JLArray(T[0, 1])
    x1 = sum(x .* m1)
    x2 = sum(x .* m2)
    return (x1 - T(3 // 2))^2 + 2 * (x2 - T(3 // 4))^2
end

@testset "Reparametrized construction, properties, exports, and Adapt" begin
    @test Base.isexported(GS, :Reparametrized)
    @test Base.isexported(GS, :ReparametrizedSolver)
    @test Reparametrized <: AbstractNonlinearProblem

    prob_nl = NonlinearProblem(f_polar_target)
    rep_nl = Reparametrized(prob_nl, g_polar)
    @test rep_nl.prob === prob_nl
    @test rep_nl.g === g_polar
    @test rep_nl.∂g === nothing
    @test rep_nl.covariant === false
    @test rep_nl.curvature === true
    @test propertynames(rep_nl) == (:prob, :g, :∂g, :covariant, :curvature)
    @test isbits(rep_nl)

    rep_nl_an = Reparametrized(
        prob_nl, g_polar, dg_polar; covariant = true, curvature = false)
    @test rep_nl_an.∂g === dg_polar
    @test rep_nl_an.covariant === true
    @test rep_nl_an.curvature === false
    @test isbits(rep_nl_an)

    prob_stat = StationarityProblem(action_polar)
    rep_stat = Reparametrized(prob_stat, g_polar)
    @test rep_stat.covariant === true
    @test rep_stat.curvature === true

    # Type parameter validation on C and K
    P = typeof(prob_nl)
    G = typeof(g_polar)
    @test_throws ArgumentError Reparametrized{P, G, Nothing, 1, true}(prob_nl, g_polar, nothing)
    @test_throws ArgumentError Reparametrized{P, G, Nothing, false, 0}(prob_nl, g_polar, nothing)

    # Adapt preserves C and K and adapts captured floats
    shift = 2.0
    prob_cap = NonlinearProblem((x, p) -> SVector(x[1] - shift, x[2] - shift))
    g_cap = (y, p) -> SVector(y[1] + shift, y[2] + shift)
    rep_cap = Reparametrized(prob_cap, g_cap; covariant = true, curvature = false)
    rep_f32 = Adapt.adapt(ToReal{Float32}(), rep_cap)
    @test rep_f32 isa Reparametrized
    @test rep_f32.covariant === true
    @test rep_f32.curvature === false
    @test eltype(rep_f32.g(SVector(1.0f0, 2.0f0), nothing)) === Float32
end

@testset "Polar map root equivalence in R1 and R3 ($T)" for T in REAL_ELTYPES
    allowscalar(false)
    tol = T === Float64 ? 8 * eps(T) : 32 * eps(T)
    opt_t = Options(T; f_abstol = 8 * eps(T), f_reltol = 4 * eps(T))
    y0 = SVector{2, T}(T(3 // 2), T(1 // 2))
    x0 = g_polar(y0, nothing)

    prob_nl = NonlinearProblem(f_polar_target)
    prob_nl_j = NonlinearProblem(f_polar_target, j_polar_target)
    x_dir, st_dir = solve(prob_nl, x0, Newton(); options = opt_t)
    @test st_dir.code == SUCCESS

    for p_inner in (prob_nl, prob_nl_j), dg in (dg_polar, nothing)

        rep = Reparametrized(p_inner, g_polar, dg)
        x_sol, st_sol = solve(rep, y0, Newton(); options = opt_t)
        @test st_sol.code == SUCCESS
        @test norm(x_sol - x_dir) <= tol * max(one(T), norm(x_dir))
        # Returned value is x* = g(y*, p), not y*
        @test norm(x_sol - SVector{2, T}(T(2), T(pi / 4))) > T(1 // 10)

        # solve!! returns (x*, status, state)
        st0 = GS.init_state(Newton(), rep, y0, nothing)
        x_s2, status_s2, _ = solve!!(st0, rep, y0, Newton(), nothing, opt_t)
        @test status_s2.code == SUCCESS
        @test norm(x_s2 - x_dir) <= tol * max(one(T), norm(x_dir))

        # init and solve! via ReparametrizedSolver
        solver = init(rep, y0, Newton(); options = opt_t)
        @test solver isa ReparametrizedSolver
        @test solver.y == y0
        @test solver.x == x0
        x_init, st_init = solve!(solver)
        @test st_init.code == SUCCESS
        @test x_init === solver.x
        @test norm(x_init - x_dir) <= tol * max(one(T), norm(x_dir))
        @test norm(g_polar(solver.y, nothing) - solver.x) <= tol * max(one(T), norm(x_dir))

        # Second solve!(solver) starts at y* and converges in 0 iterations
        x_init2, st_init2 = solve!(solver)
        @test st_init2.code == SUCCESS
        @test st_init2.iterations == Int32(0)
        @test x_init2 == x_init

        # solve!(solver, y_new)
        y_new = SVector{2, T}(T(5 // 2), T(3 // 4))
        x_new, st_new = solve!(solver, y_new)
        @test st_new.code == SUCCESS
        @test norm(x_new - x_dir) <= tol * max(one(T), norm(x_dir))
    end

    # R1 solve_batch!
    rep_b = Reparametrized(prob_nl, g_polar, dg_polar)
    Y_batch = [y0, SVector{2, T}(T(9 // 5), T(7 // 10))]
    counts = solve_batch!(Y_batch, rep_b, Newton(); options = opt_t)
    @test counts.SUCCESS == 2
    for x_b in Y_batch
        @test norm(x_b - x_dir) <= tol * max(one(T), norm(x_dir))
    end
    S_batch = similar(Y_batch, SolverStatus{T})
    Y_batch2 = [y0, SVector{2, T}(T(9 // 5), T(7 // 10))]
    solve_batch!(Y_batch2, S_batch, rep_b, Newton(); options = opt_t)
    @test all(s -> s.code == SUCCESS, S_batch)
    for x_b in Y_batch2
        @test norm(x_b - x_dir) <= tol * max(one(T), norm(x_dir))
    end
    @test solve_batch!(SVector{2, T}[], rep_b, Newton()).SUCCESS == 0
    @test isempty(solve_batch!(SVector{2, T}[], SolverStatus{T}[], rep_b, Newton()))
    @test_throws DimensionMismatch solve_batch!(
        Y_batch2, similar(Y_batch2, SolverStatus{T}, 1), rep_b, Newton())
    R_other = T === Float64 ? Float32 : Float64
    @test_throws ArgumentError solve_batch!(
        Y_batch2, similar(Y_batch2, SolverStatus{R_other}), rep_b, Newton())

    # R1 Number (scalar) coordinate chart: g(y, p) = y^2 + 1 on F(x, p) = x^2 - 9
    prob_num = NonlinearProblem((x, p) -> x^2 - T(9))
    for dg_num in ((y, p) -> 2 * y, nothing)
        rep_num = Reparametrized(prob_num, (y, p) -> y^2 + one(T), dg_num)
        x_num, st_num = solve(rep_num, T(6 // 5), Newton(); options = opt_t)
        @test st_num.code == SUCCESS
        @test isapprox(x_num, T(3); rtol = tol)
        solver_num = init(rep_num, T(6 // 5), Newton(); options = opt_t)
        x_num2, st_num2 = solve!(solver_num)
        @test st_num2.code == SUCCESS
        @test isapprox(x_num2, T(3); rtol = tol)
        @test isapprox(solver_num.y, sqrt(T(2)); rtol = tol)
    end

    # R3 Array and JLArray
    prob_r3 = NonlinearProblem(f_polar_target!)
    for Arr in ARRAY_BACKENDS, dg_r3 in (dg_polar!, nothing)

        rep_r3 = Reparametrized(prob_r3, g_polar!, dg_r3)
        y0_r3 = Arr(Vector(y0))
        x_r3, st_r3 = solve(rep_r3, y0_r3, Newton(); options = opt_t)
        @test st_r3.code == SUCCESS
        @test Array(y0_r3) == Vector(y0)
        @test norm(Array(x_r3) - Vector(x_dir)) <= tol * max(one(T), norm(x_dir))

        solver_r3 = init(rep_r3, y0_r3, Newton(); options = opt_t)
        @test solver_r3 isa ReparametrizedSolver
        @test solver_r3.y !== y0_r3
        x_r3_init, st_r3_init = solve!(solver_r3)
        @test st_r3_init.code == SUCCESS
        @test x_r3_init === solver_r3.x
        @test norm(Array(x_r3_init) - Vector(x_dir)) <= tol * max(one(T), norm(x_dir))
        x_check = Arr(zeros(T, 2))
        g_polar!(x_check, solver_r3.y, nothing)
        @test norm(Array(x_check) - Array(solver_r3.x)) <= tol * max(one(T), norm(x_dir))

        # Second solve!(solver_r3) starts from solver_r3.y == y* and takes 0 iterations
        _, st_r3_second = solve!(solver_r3)
        @test st_r3_second.code == SUCCESS
        @test st_r3_second.iterations == Int32(0)

        # solve!(solver_r3, y_in)
        y_in = Arr(T[9 // 5, 7 // 10])
        x_r3_in, st_r3_in = solve!(solver_r3, y_in)
        @test st_r3_in.code == SUCCESS
        @test solver_r3.y === y_in
        @test x_r3_in === solver_r3.x
        @test norm(Array(x_r3_in) - Vector(x_dir)) <= tol * max(one(T), norm(x_dir))

        # Validation of mismatched iterate / parameter types on solve!
        @test_throws ArgumentError solve!(solver_r3, Arr(zeros(T, 3)))
        @test_throws ArgumentError solve!(solver_r3, SVector{2, T}(T(1), T(2)))
        @test_throws ArgumentError solve!(solver_r3, y_in, (a = 1,))
    end
end

@testset "Holomorphic chart across ELTYPES ($T)" for T in ELTYPES
    allowscalar(false)
    R = real(T)
    tol = 32 * eps(R)
    opt_h = Options(T; f_abstol = 8 * eps(R), f_reltol = 4 * eps(R))
    # Holomorphic chart x = g(y) = (y1^2, y2 + y1) on F(x) = (x1 - 4, x2 - 5) => y* = (2, 3), x* = (4, 5)
    g_holo(y, p) = SVector(y[1]^2, y[2] + y[1])
    dg_holo(y, p) = SMatrix{2, 2, T, 4}(2 * y[1], one(T), zero(T), one(T))
    f_holo(x, p) = SVector(x[1] - T(4), x[2] - T(5))
    prob_holo = NonlinearProblem(f_holo)
    y0_h = SVector{2, T}(T(3 // 2), T(5 // 2))
    for dg in (dg_holo, nothing)
        rep_h = Reparametrized(prob_holo, g_holo, dg)
        x_h, st_h = solve(rep_h, y0_h, Newton(); options = opt_h)
        @test st_h.code == SUCCESS
        @test isapprox(x_h, SVector{2, T}(T(4), T(5)); rtol = tol)
    end
end

@testset "R3 rejected trial step re-evaluates g(x_buf, y*, p) at returned y*" begin
    # A residual whose trial step produces NaN so Backtracking rejects every trial step and
    # returns y* = y0 while residual!! last wrote g(y_trial) into ws.x_buf.
    function g_shift!(x, y, p)
        x .= 2 .* y .+ 1
        return x
    end
    function f_nan_step!(F, x, p)
        # At x0 = g(y0) = [3.0, 5.0] (y0 = [1.0, 2.0]), F = [1.0, 1.0] and J = I,
        # so the Newton step moves y away from y0; for any x != [3.0, 5.0] return NaN
        # except during Jacobian dual seeding where eltype(x) is Dual.
        if eltype(x) <: AbstractFloat &&
           (abs(x[1] - 3.0) > 1e-12 || abs(x[2] - 5.0) > 1e-12)
            fill!(F, NaN)
        else
            F[1] = x[1] - 2.0
            F[2] = x[2] - 4.0
        end
        return F
    end
    prob_nan = NonlinearProblem(f_nan_step!)
    rep_nan = Reparametrized(prob_nan, g_shift!)
    y0_nan = [1.0, 2.0]
    m_bt = Newton(; globalization = Backtracking(; maxiter = 2))
    x_ret, st_ret = solve(rep_nan, y0_nan, m_bt; max_iterations = 1)
    # Because the line search fails without moving, y* == y0_nan == [1.0, 2.0],
    # so x* MUST be g(y*, p) == [3.0, 5.0], NOT the rejected trial's g(y_trial, p).
    @test st_ret.code != SUCCESS
    @test x_ret == [3.0, 5.0]

    solver_nan = init(rep_nan, copy(y0_nan), m_bt; max_iterations = 1)
    x_sol_nan, _ = solve!(solver_nan)
    @test solver_nan.y == [1.0, 2.0]
    @test x_sol_nan == [3.0, 5.0]
end

@testset "User-supplied ∂g is used when ∂g !== nothing" begin
    y0 = SVector(1.5, 0.5)
    prob_nl = NonlinearProblem(f_polar_target)
    rep_wrong_r1 = Reparametrized(prob_nl, g_polar, dg_polar_wrong)
    x_w_r1, st_w_r1 = solve(
        rep_wrong_r1, y0, Newton(; globalization = Static()); max_iterations = 15)
    @test st_w_r1.code != SUCCESS
    @test norm(f_polar_target(x_w_r1, nothing)) > 1e-3

    prob_r3 = NonlinearProblem(f_polar_target!)
    rep_wrong_r3 = Reparametrized(prob_r3, g_polar!, dg_polar_wrong!)
    x_w_r3, st_w_r3 = solve(
        rep_wrong_r3, [1.5, 0.5], Newton(; globalization = Static()); max_iterations = 15)
    @test st_w_r3.code != SUCCESS
    F_check = zeros(2)
    f_polar_target!(F_check, x_w_r3, nothing)
    @test norm(F_check) > 1e-3
end

@testset "Covariant StationarityProblem residual, curvature Jacobian, and Complex transpose" begin
    allowscalar(false)
    prob_stat = StationarityProblem(action_polar)
    y_test = SVector(1.5, 0.5)
    x_test = g_polar(y_test, nothing)
    G_exact = dg_polar(y_test, nothing)
    Fx_exact = residual!!(nothing, prob_stat, x_test, nothing)
    Jx_exact = gradient_hessian!!(
        nothing, nothing, AutoForwardDiff(), prob_stat, x_test, nothing)[2]
    Fy_expected = transpose(G_exact) * Fx_exact
    Jy_pullback = transpose(G_exact) * Jx_exact * G_exact

    # Central-difference reference of F̃(y) = G(y)ᵀ F(g(y))
    f_tilde_ref(y) = transpose(dg_polar(y, nothing)) *
                     residual!!(nothing, prob_stat, g_polar(y, nothing), nothing)
    h_fd = 1e-6
    e1 = SVector(1.0, 0.0)
    e2 = SVector(0.0, 1.0)
    col1 = (f_tilde_ref(y_test + h_fd * e1) - f_tilde_ref(y_test - h_fd * e1)) / (2 * h_fd)
    col2 = (f_tilde_ref(y_test + h_fd * e2) - f_tilde_ref(y_test - h_fd * e2)) / (2 * h_fd)
    J_fd = hcat(col1, col2)

    for dg in (nothing, dg_polar)
        rep_curv = Reparametrized(prob_stat, g_polar, dg)
        rep_nocurv = Reparametrized(prob_stat, g_polar, dg; curvature = false)

        Fy = residual!!(nothing, rep_curv, y_test, nothing)
        Jy_curv = jacobian!!(nothing, AutoForwardDiff(), rep_curv, y_test, nothing)
        Jy_nocurv = jacobian!!(nothing, AutoForwardDiff(), rep_nocurv, y_test, nothing)

        # F̃ = Gᵀ F(g(y))
        @test norm(Fy - Fy_expected, Inf) <=
              8 * eps(Float64) * max(1.0, norm(Fy_expected, Inf))
        @test norm(Fy - Fx_exact, Inf) > 0.1

        # J̃ with curvature == true is symmetric and matches central difference to <= 1e-6
        @test norm(Jy_curv - transpose(Jy_curv), Inf) <=
              64 * eps(Float64) * max(1.0, norm(Jy_curv, Inf))
        @test norm(Jy_curv - J_fd, Inf) <= 1e-6 * max(1.0, norm(J_fd, Inf))
        @test norm(Jy_curv - Jy_pullback, Inf) > 0.1

        # J̃ with curvature == false equals Gᵀ J G
        @test norm(Jy_nocurv - Jy_pullback, Inf) <=
              64 * eps(Float64) * max(1.0, norm(Jy_pullback, Inf))

        # jvp!! agrees with J̃ * v in both curvature modes
        v_dir = SVector(0.7, -0.4)
        jv_curv = jvp!!(nothing, AutoForwardDiff(), rep_curv, y_test, v_dir, nothing)
        jv_nocurv = jvp!!(nothing, AutoForwardDiff(), rep_nocurv, y_test, v_dir, nothing)
        @test isapprox(jv_curv, Jy_curv * v_dir; rtol = 16 * eps(Float64))
        @test isapprox(jv_nocurv, Jy_nocurv * v_dir; rtol = 16 * eps(Float64))
    end

    # A covariant chart solve converges to the critical point of the action. The quadratic
    # action has one, so the assertion needs no second solve and holds in every chart.
    for dg in (nothing, dg_polar)
        rep_quad = Reparametrized(StationarityProblem(action_quad), g_polar, dg)
        x_quad, st_quad = solve(rep_quad, y_test, Newton(; linsolve = LDLTFactorization()))
        @test st_quad.code == SUCCESS
        # The default residual test is relative at sqrt(eps(Float64)) = 1.5e-8, so 1e-8 on the
        # iterate is that scale.
        @test norm(x_quad - X_QUAD) <= 1e-8
    end

    # R3 covariant StationarityProblem across Array and JLArray, with dg_polar! and nothing
    prob_stat_r3 = StationarityProblem(action_polar_r3)
    prob_quad_r3 = StationarityProblem(action_quad_r3)
    for Arr in ARRAY_BACKENDS, dg_r3 in (nothing, dg_polar!)

        rep_stat_r3 = Reparametrized(prob_stat_r3, g_polar!, dg_r3)
        rep_nocurv_r3 = Reparametrized(prob_stat_r3, g_polar!, dg_r3; curvature = false)
        y_r3 = Arr(Vector(y_test))
        Fy_r3 = Arr(zeros(2))
        Jy_r3 = Arr(zeros(2, 2))
        Jy_nc_r3 = Arr(zeros(2, 2))
        prep_c = prepare_ad(AutoForwardDiff(), rep_stat_r3, Fy_r3, y_r3, nothing)
        prep_nc = prepare_ad(AutoForwardDiff(), rep_nocurv_r3, Fy_r3, y_r3, nothing)

        residual!!(Fy_r3, prep_c, rep_stat_r3, y_r3, nothing)
        jacobian!!(Jy_r3, prep_c, rep_stat_r3, y_r3, nothing)
        jacobian!!(Jy_nc_r3, prep_nc, rep_nocurv_r3, y_r3, nothing)

        @test norm(Array(Fy_r3) - Vector(Fy_expected), Inf) <=
              16 * eps(Float64) * max(1.0, norm(Fy_expected, Inf))
        @test norm(Array(Jy_r3) - Matrix(J_fd), Inf) <= 1e-6 * max(1.0, norm(J_fd, Inf))
        @test norm(Array(Jy_nc_r3) - Matrix(Jy_pullback), Inf) <=
              64 * eps(Float64) * max(1.0, norm(Jy_pullback, Inf))

        v_r3 = Arr([0.7, -0.4])
        Jv_r3 = Arr(zeros(2))
        jvp!!(Jv_r3, prep_c, rep_stat_r3, y_r3, v_r3, nothing)
        @test isapprox(Array(Jv_r3), Array(Jy_r3) * [0.7, -0.4]; rtol = 32 * eps(Float64))

        rep_quad_r3 = Reparametrized(prob_quad_r3, g_polar!, dg_r3)
        m_r3 = Arr === Array ? Newton(; linsolve = LDLTFactorization()) : Newton()
        x_sol_r3, st_sol_r3 = solve(rep_quad_r3, y_r3, m_r3)
        @test st_sol_r3.code == SUCCESS
        @test norm(Array(x_sol_r3) - Vector(X_QUAD)) <= 1e-8
    end

    # Complex covariant problem: transpose(G) vs adjoint(G)
    g_cplx(y, p) = SVector(y[1]^2 + im * y[2], y[1] - im * y[2]^2)
    dg_cplx(y, p) = SMatrix{2, 2, eltype(y), 4}(
        2 * y[1], one(eltype(y)), im *
                                  one(eltype(y)), -2 *
                                                  im *
                                                  y[2])
    f_cplx(x, p) = SVector(x[1]^2 - (1.0 + 0.5im), x[2]^2 - (0.5 - 0.25im))
    j_cplx(x, p) = SMatrix{2, 2, eltype(x), 4}(2 * x[1], 0.0 + 0.0im, 0.0 + 0.0im, 2 * x[2])
    prob_cplx = NonlinearProblem(f_cplx, j_cplx)
    y_c = SVector(1.1 + 0.3im, -0.4 + 0.7im)
    x_c = g_cplx(y_c, nothing)
    G_c = dg_cplx(y_c, nothing)
    Fx_c = f_cplx(x_c, nothing)
    Jx_c = j_cplx(x_c, nothing)
    Fy_trans = transpose(G_c) * Fx_c
    Fy_adj = adjoint(G_c) * Fx_c
    @test norm(Fy_trans - Fy_adj) > 0.5

    # The same complex chart in R3, over both array backends, with the chart Jacobian given and
    # computed: `transpose(G)` and `adjoint(G)` differ here, so this separates them.
    function g_cplx!(x::Array, y::Array, p)
        @inbounds begin
            y1, y2 = y[1], y[2]
            x[1] = y1^2 + im * y2
            x[2] = y1 - im * y2^2
        end
        return x
    end
    function g_cplx!(x, y, p)
        T = eltype(y)
        e1 = JLArray(T[1, 0])
        e2 = JLArray(T[0, 1])
        y1 = sum(y .* e1)
        y2 = sum(y .* e2)
        x .= (y1^2 + im * y2) .* e1 .+ (y1 - im * y2^2) .* e2
        return x
    end
    function dg_cplx!(G::Array, y::Array, p)
        @inbounds begin
            y1, y2 = y[1], y[2]
            G[1, 1] = 2 * y1
            G[2, 1] = one(y1)
            G[1, 2] = im
            G[2, 2] = -2 * im * y2
        end
        return G
    end
    function dg_cplx!(G, y, p)
        T = eltype(y)
        e1 = JLArray(T[1, 0])
        e2 = JLArray(T[0, 1])
        y1 = sum(y .* e1)
        y2 = sum(y .* e2)
        copyto!(G, T[2 * y1 im; one(T) (-2 * im * y2)])
        return G
    end
    function f_cplx!(F::Array, x::Array, p)
        @inbounds begin
            x1, x2 = x[1], x[2]
            F[1] = x1^2 - (1.0 + 0.5im)
            F[2] = x2^2 - (0.5 - 0.25im)
        end
        return F
    end
    function f_cplx!(F, x, p)
        T = eltype(x)
        e1 = JLArray(T[1, 0])
        e2 = JLArray(T[0, 1])
        x1 = sum(x .* e1)
        x2 = sum(x .* e2)
        F .= (x1^2 - (1.0 + 0.5im)) .* e1 .+ (x2^2 - (0.5 - 0.25im)) .* e2
        return F
    end
    function j_cplx!(J::Array, x::Array, p)
        @inbounds begin
            x1, x2 = x[1], x[2]
            Z = zero(eltype(x))
            J[1, 1] = 2 * x1
            J[2, 1] = Z
            J[1, 2] = Z
            J[2, 2] = 2 * x2
        end
        return J
    end
    function j_cplx!(J, x, p)
        T = eltype(x)
        e1 = JLArray(T[1, 0])
        e2 = JLArray(T[0, 1])
        x1 = sum(x .* e1)
        x2 = sum(x .* e2)
        Z = zero(T)
        copyto!(J, T[2 * x1 Z; Z (2 * x2)])
        return J
    end
    prob_cplx_r3 = NonlinearProblem(f_cplx!, j_cplx!)

    for dg_c in (dg_cplx, nothing)
        rep_c_curv = Reparametrized(
            prob_cplx, g_cplx, dg_c; covariant = true, curvature = true)
        rep_c_nocurv = Reparametrized(
            prob_cplx, g_cplx, dg_c; covariant = true, curvature = false)
        @test isapprox(residual!!(nothing, rep_c_curv, y_c, nothing), Fy_trans; rtol = 32 *
                                                                                       eps(Float64))
        Jy_c_nc = jacobian!!(nothing, AutoForwardDiff(), rep_c_nocurv, y_c, nothing)
        @test isapprox(Jy_c_nc, transpose(G_c) * Jx_c * G_c; rtol = 32 * eps(Float64))
        @test norm(Jy_c_nc - adjoint(G_c) * Jx_c * G_c) > 0.5

        # Complex central difference of F̃(y) = transpose(G(y)) * F(g(y))
        f_tilde_c(y) = transpose(dg_cplx(y, nothing)) * f_cplx(g_cplx(y, nothing), nothing)
        hc = 1e-6
        c1 = (f_tilde_c(y_c + hc * SVector(1.0 + 0.0im, 0.0 + 0.0im)) -
              f_tilde_c(y_c - hc * SVector(1.0 + 0.0im, 0.0 + 0.0im))) / (2 * hc)
        c2 = (f_tilde_c(y_c + hc * SVector(0.0 + 0.0im, 1.0 + 0.0im)) -
              f_tilde_c(y_c - hc * SVector(0.0 + 0.0im, 1.0 + 0.0im))) / (2 * hc)
        J_fd_c = hcat(c1, c2)
        Jy_c = jacobian!!(nothing, AutoForwardDiff(), rep_c_curv, y_c, nothing)
        @test norm(Jy_c - J_fd_c, Inf) <= 1e-6 * max(1.0, norm(J_fd_c, Inf))

        # The same chart in R3, with the in-place chart Jacobian that R3 takes: the covariant
        # pull-back is `transpose(G)` there too, and the curvature term keeps it so.
        dg_c_r3 = dg_c === nothing ? nothing : dg_cplx!
        for Arr in ARRAY_BACKENDS
            rep_c_curv_r3 = Reparametrized(
                prob_cplx_r3, g_cplx!, dg_c_r3; covariant = true, curvature = true)
            rep_c_nocurv_r3 = Reparametrized(
                prob_cplx_r3, g_cplx!, dg_c_r3; covariant = true, curvature = false)
            y_a = Arr(Vector(y_c))
            Fy_a = Arr(zeros(ComplexF64, 2))
            Jy_a = Arr(zeros(ComplexF64, 2, 2))
            Jy_nc_a = Arr(zeros(ComplexF64, 2, 2))
            prep_curv = prepare_ad(AutoForwardDiff(), rep_c_curv_r3, Fy_a, y_a, nothing)
            prep_nocurv = prepare_ad(AutoForwardDiff(), rep_c_nocurv_r3, Fy_a, y_a, nothing)
            residual!!(Fy_a, prep_curv, rep_c_curv_r3, y_a, nothing)
            jacobian!!(Jy_a, prep_curv, rep_c_curv_r3, y_a, nothing)
            jacobian!!(Jy_nc_a, prep_nocurv, rep_c_nocurv_r3, y_a, nothing)
            Jy_pullback_c = transpose(G_c) * Jx_c * G_c
            @test norm(Array(Fy_a) - Vector(Fy_trans), Inf) <=
                  64 * eps(Float64) * max(1.0, norm(Fy_trans, Inf))
            @test norm(Array(Fy_a) - Vector(Fy_adj), Inf) > 0.5
            @test norm(Array(Jy_nc_a) - Matrix(Jy_pullback_c), Inf) <=
                  64 * eps(Float64) * max(1.0, norm(Jy_pullback_c, Inf))
            @test norm(Array(Jy_nc_a) - Matrix(adjoint(G_c) * Jx_c * G_c), Inf) > 0.5
            @test norm(Array(Jy_a) - Matrix(J_fd_c), Inf) <=
                  1e-6 * max(1.0, norm(J_fd_c, Inf))
        end
    end
end

@testset "StationarityProblem index sets in a chart" begin
    allowscalar(false)
    y0 = SVector(1.5, 0.5)

    # Explicit full index sets, and an E that permutes 1:n, solve as E = U = : does. The
    # quadratic action has one critical point, so the assertion needs no second solve.
    for (E, U) in (((1, 2), (1, 2)), ((2, 1), (1, 2)))
        prob = StationarityProblem(action_quad, E, U)
        x_r1, st_r1 = solve(Reparametrized(prob, g_polar), y0)
        @test st_r1.code == SUCCESS
        @test norm(x_r1 - X_QUAD) <= 1e-8
        prob_r3 = StationarityProblem(action_quad_r3, collect(E), collect(U))
        x_r3, st_r3 = solve(Reparametrized(prob_r3, g_polar!), Vector(y0))
        @test st_r3.code == SUCCESS
        @test norm(x_r3 - Vector(X_QUAD)) <= 1e-8
    end

    # A proper subset leaves the residual shorter than the chart, and a U out of order puts the
    # Jacobian columns out of the order of x. Both raise an ArgumentError before any evaluation.
    for (E, U) in (((1,), (1,)), ((2,), (1,)), ((1, 2), (2, 1)))
        rep = Reparametrized(StationarityProblem(action_quad, E, U), g_polar)
        @test_throws ArgumentError solve(rep, y0)
        @test_throws ArgumentError init(rep, y0)
        @test_throws ArgumentError prepare_ad(nothing, rep, nothing, y0, nothing)
        @test_throws ArgumentError solve_batch!([y0, y0], rep)
        prob_r3 = StationarityProblem(action_quad_r3, collect(E), collect(U))
        rep_r3 = Reparametrized(prob_r3, g_polar!)
        @test_throws ArgumentError solve(rep_r3, Vector(y0))
        @test_throws ArgumentError init(rep_r3, Vector(y0))
        @test_throws ArgumentError residual!!(zeros(2), rep_r3, Vector(y0), nothing)
    end

    # An R1 iterate takes E and U as :, a Tuple or an SVector, as a StationarityProblem does.
    rep_range = Reparametrized(StationarityProblem(action_quad, 1:2, 1:2), g_polar)
    @test_throws ArgumentError solve(rep_range, y0)

    # A device index set is compared on the device, so a U out of order raises the ArgumentError
    # of the chart, and not a scalar-indexing error, under allowscalar(false). The constructor
    # checks for duplicates by scalar indexing, so the problem is built with it allowed.
    y_dev = JLArray(Vector(y0))
    prob_dev = @allowscalar StationarityProblem(
        action_quad_r3, JLArray([1, 2]), JLArray([2, 1]))
    rep_dev = Reparametrized(prob_dev, g_polar!)
    @test_throws ArgumentError solve(rep_dev, y_dev)
    @test_throws ArgumentError init(rep_dev, y_dev)
    prob_dev_in_order = @allowscalar StationarityProblem(
        action_quad_r3, JLArray([1, 2]), JLArray([1, 2]))
    @test GS._check_chart_indices(prob_dev_in_order, y_dev) === nothing
end

# The critical point of the quadratic action with the datum b is X_QUAD - (b₁ / 2, b₂ / 4); a
# scalar b is b₁ = b₂.
x_quad_datum(b) = X_QUAD - SVector(first(b) / 2, last(b) / 4)

@testset "Datum b of init and solve through the chart in $R, b = $b" for (R, prob, chart, it, b) in (
    ("R1", StationarityProblem(action_quad), g_polar, identity, SVector(1.0, 2.0)),
    ("R1", StationarityProblem(action_quad), g_polar, identity, 0.5),
    ("R3", StationarityProblem(action_quad_r3), g_polar!, Vector, [1.0, 2.0]),
    ("R3", StationarityProblem(action_quad_r3), g_polar!, Vector, 0.5))
    allowscalar(false)
    y0 = it(SVector(1.5, 0.5))
    rep = Reparametrized(prob, chart)
    # The plain problem from the chart's initial point, with the same datum.
    x_plain, _ = solve(prob, it(g_polar(SVector(1.5, 0.5), nothing)), Newton(); b = b)
    @testset "solve" begin
        x, st = solve(rep, y0, Newton(); b = b)
        @test st.code == SUCCESS
        @test norm(x - x_plain) <= 1e-8
        @test norm(x - x_quad_datum(b)) <= 1e-8
    end
    @testset "init and solve!" begin
        x, st = solve!(init(rep, y0; b = b))
        @test st.code == SUCCESS
        @test norm(x - x_plain) <= 1e-8
        @test norm(x - x_quad_datum(b)) <= 1e-8
    end
end

@testset "Datum b of a chart around a NonlinearProblem" begin
    rep_nl = Reparametrized(NonlinearProblem(f_polar_target), g_polar)
    @test_throws ArgumentError solve(rep_nl, SVector(1.5, 0.5); b = 1.0)
    @test_throws ArgumentError init(rep_nl, SVector(1.5, 0.5); b = 1.0)
end

@testset "R3 chart Jacobian off ∂g in more than one AD chunk ($Arr)" for Arr in ARRAY_BACKENDS
    # The AD chart Jacobian fills one chunk of columns per pass, and a chart of 16 coordinates
    # takes more than one pass at the default chunk size and at the two-column chunk below. A
    # pass the loop drops therefore leaves columns of the reference Jacobian at zero.
    n = 16
    y_base = [0.4 + 0.3i for i in 1:n]
    G_ref = fill(0.1, n, n)
    for i in 1:n
        G_ref[i, i] = 2 * y_base[i] + cos(y_base[i]) + 0.1
    end
    prob_identity = NonlinearProblem(f_identity!)
    reparam_chunk = Reparametrized(prob_identity, g_chunk!, nothing; covariant = false)
    for backend in (AutoForwardDiff(), AutoForwardDiff(; chunksize = 2))
        y_a = Arr(copy(y_base))
        Jy_a = Arr(zeros(n, n))
        prep = prepare_ad(backend, reparam_chunk, Arr(zeros(n)), y_a, nothing)
        jacobian!!(Jy_a, prep, reparam_chunk, y_a, nothing)
        @test norm(Array(Jy_a) - G_ref, Inf) <= 64 * eps(Float64) * norm(G_ref, Inf)
    end
end

# The gradient of `action_quad` as a residual, whose root is X_QUAD, and a Jacobian that writes
# only the stored entries of a sparse `jacobian_prototype`.
function f_quad_grad!(F, x, p)
    F[1] = 2 * x[1] - 3
    F[2] = 4 * x[2] - 3
    return F
end

function j_quad_nonzeros!(J, x, p)
    nz = nonzeros(J)
    nz[1] = 2.0
    nz[2] = 4.0
    return J
end

@testset "R3 chart around a NonlinearProblem with a sparse jacobian_prototype" begin
    # The chart Jacobian with the user J in the prototype's storage equals the one with the inner
    # Jacobian by automatic differentiation, which is exact for this affine residual.
    P = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    prob_sparse = NonlinearProblem(f_quad_grad!, j_quad_nonzeros!; jacobian_prototype = P)
    prob_ad = NonlinearProblem(f_quad_grad!)
    y0 = [1.5, 0.5]
    for (cov, curv) in ((false, true), (true, false), (true, true))
        rep_sparse = Reparametrized(
            prob_sparse, g_polar!, dg_polar!; covariant = cov, curvature = curv)
        rep_ad = Reparametrized(
            prob_ad, g_polar!, dg_polar!; covariant = cov, curvature = curv)
        prep_sparse = prepare_ad(AutoForwardDiff(), rep_sparse, zeros(2), y0, nothing)
        prep_ad = prepare_ad(AutoForwardDiff(), rep_ad, zeros(2), y0, nothing)
        Jy_sparse = jacobian!!(zeros(2, 2), prep_sparse, rep_sparse, y0, nothing)
        Jy_ad = jacobian!!(zeros(2, 2), prep_ad, rep_ad, y0, nothing)
        @test isapprox(Jy_sparse, Jy_ad; rtol = 8 * eps(Float64))
        x, st = solve(rep_sparse, y0, Newton())
        @test st.code == SUCCESS
        @test norm(x - Vector(X_QUAD)) <= 1e-8
    end
end

@testset "Inference and zero allocations on Reparametrized" begin
    y0 = SVector(1.5, 0.5)
    prob_nl = NonlinearProblem(f_polar_target)
    rep_nl = Reparametrized(prob_nl, g_polar, dg_polar)
    rep_nl_ad = Reparametrized(prob_nl, g_polar)
    prob_stat = StationarityProblem(action_polar)
    rep_stat = Reparametrized(prob_stat, g_polar)

    @test (@inferred solve(rep_nl, y0, Newton()))[2].code == SUCCESS
    @test (@inferred solve(rep_nl_ad, y0, Newton()))[2].code == SUCCESS
    @test (@inferred solve(rep_stat, y0, Newton(; linsolve = LDLTFactorization())))[2].code ==
          SUCCESS

    if Base.JLOptions().check_bounds != 1
        # skipped where a run forces `--check-bounds=yes`
        @test allocations(a -> solve(rep_nl, a, Newton()), y0) == 0
        @test allocations(a -> solve(rep_nl_ad, a, Newton()), y0) == 0
        @test allocations(a -> solve(rep_stat, a, Newton()), y0) == 0

        for rep_r3 in (
            Reparametrized(NonlinearProblem(f_polar_target!), g_polar!, dg_polar!),
            Reparametrized(NonlinearProblem(f_polar_target!), g_polar!),
            Reparametrized(StationarityProblem(action_polar_r3), g_polar!, dg_polar!),
            Reparametrized(StationarityProblem(action_polar_r3), g_polar!)
        )
            solver_r3 = init(rep_r3, [1.5, 0.5], Newton())
            solve!(solver_r3, [1.5, 0.5])
            y_in = [1.5, 0.5]
            @test allocations(a -> solve!(solver_r3, a), y_in) == 0
        end
    end
end

@testset "JET.@test_opt on the R1 chart path, which the R2 kernel runs ($T)" for T in REAL_ELTYPES
    y0 = SVector{2, T}(T(3 // 2), T(1 // 2))
    prob_nl = NonlinearProblem(f_polar_target)
    prob_stat = StationarityProblem(action_polar)
    for dg in (dg_polar, nothing)
        rep = Reparametrized(prob_nl, g_polar, dg)
        @test_opt residual!!(nothing, rep, y0, nothing)
        @test_opt jacobian!!(nothing, AutoForwardDiff(), rep, y0, nothing)
        @test_opt jvp!!(nothing, AutoForwardDiff(), rep, y0, y0, nothing)
        @test_opt solve(rep, y0, Newton())
        rep_nc = Reparametrized(prob_nl, g_polar, dg; covariant = true, curvature = false)
        @test_opt jacobian!!(nothing, AutoForwardDiff(), rep_nc, y0, nothing)
    end
    for dg in (dg_polar, nothing)
        rep = Reparametrized(prob_stat, g_polar, dg)
        @test_opt residual!!(nothing, rep, y0, nothing)
        @test_opt jacobian!!(nothing, AutoForwardDiff(), rep, y0, nothing)
        @test_opt solve(rep, y0, Newton())
    end
end
