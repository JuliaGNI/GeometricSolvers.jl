using GeometricSolvers
using GeometricSolvers: AndersonState, ToReal, effective_kappa_bar, finish!!, init_state,
                        residual_buffer, rnorm, solve!!, solve_weights, start!!, step!!,
                        sync_merit
using Adapt
using ForwardDiff: ForwardDiff
using GPUArraysCore
using JET
using JLArrays
using KernelAbstractions
using LinearAlgebra
using StaticArrays
using Test

include("../helpers/matrix.jl")
GPUArraysCore.allowscalar(false)

@kernel function _anderson_ka_residual_kernel!(F, @Const(x))
    i = @index(Global, Linear)
    @inbounds begin
        xi = x[i]
        F[i] = xi - cos(0.3 * xi) - 0.25
    end
end

struct _AndersonR1LinearF{T, R}
    c1::T
    c2::T
end

function (f::_AndersonR1LinearF{T, R})(z, p) where {T, R}
    return SVector(z[1] - R(1 // 5) * z[1] - f.c1, z[2] - R(1 // 4) * z[2] - f.c2)
end

struct _AndersonR3LinearF{T, R}
    c1::T
end

# A member that takes the step of `Picard(; damping = 0.5)` and reports every step `STALLED`
struct _StallingPicard end

function GeometricSolvers.init_state(::_StallingPicard, prob::NonlinearProblem, x, p)
    return init_state(Picard(; damping = 0.5), prob, x, p)
end

function GeometricSolvers.step!!(
        s::GeometricSolvers.PicardState, ::_StallingPicard, prob::NonlinearProblem, x, Fx, p)
    x₊, F₊, s₊, info = step!!(s, Picard(; damping = 0.5), prob, x, Fx, p)
    info_stalled = StepInfo{typeof(info.fnorm)}(
        STALLED, info.fnorm, info.stepnorm, info.failures, info.promoted, info.limited)
    return x₊, F₊, s₊, info_stalled
end

function (f::_AndersonR3LinearF{T, R})(F, z, p) where {T, R}
    @. F = R(4 // 5) * z - f.c1
    return F
end

function _nlsolve_anderson_reference(g_fun, x0::AbstractVector{Float64};
        m::Int = 8, beta::Float64 = 0.7, abstol::Float64 = 1e-11, max_iters::Int = 200)
    n = length(x0)
    m_eff = min(m, n)
    x = copy(x0)
    gx = g_fun(x)
    w = gx .- x
    fnorm = norm(w)
    fnorm <= abstol && return x, 0, fnorm

    ΔX = zeros(Float64, n, m_eff)
    ΔW = zeros(Float64, n, m_eff)
    x_prev = copy(x)
    w_prev = copy(w)

    # Step 1: plain damped Picard step
    @. x = x + beta * w
    iters = 1
    gx = g_fun(x)
    @. w = gx - x
    fnorm = norm(w)
    fnorm <= abstol && return x, iters, fnorm

    depth = 0
    for _ in 2:max_iters
        iters += 1
        if depth < m_eff
            depth += 1
            for j in depth:-1:2
                @views ΔX[:, j] .= ΔX[:, j - 1]
                @views ΔW[:, j] .= ΔW[:, j - 1]
            end
        else
            for j in m_eff:-1:2
                @views ΔX[:, j] .= ΔX[:, j - 1]
                @views ΔW[:, j] .= ΔW[:, j - 1]
            end
        end
        @views ΔX[:, 1] .= x .- x_prev
        @views ΔW[:, 1] .= w .- w_prev

        x_prev .= x
        w_prev .= w

        ΔW_k = @view ΔW[:, 1:depth]
        ΔX_k = @view ΔX[:, 1:depth]
        γ = ΔW_k \ w

        x .= (x .+ beta .* w) .- (ΔX_k * γ .+ beta .* (ΔW_k * γ))
        gx = g_fun(x)
        @. w = gx - x
        fnorm = norm(w)
        fnorm <= abstol && return x, iters, fnorm
    end
    return x, iters, fnorm
end

function _gmres_reference(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
        x0::AbstractVector{Float64}, k::Int)
    n = length(x0)
    r0 = b - A * x0
    k == 0 && return copy(x0)
    V = zeros(Float64, n, k)
    v = copy(r0)
    for j in 1:k
        V[:, j] .= v
        v = A * v
    end
    AV = A * V
    y = AV \ r0
    return x0 + V * y
end

function _r1_step_alloc_probe(state, method, prob, x, Fx)
    x1, F1, st1, _ = step!!(state, method, prob, x, Fx, nothing)
    _, _, st2, info2 = step!!(st1, method, prob, x1, F1, nothing)
    @test st2.depth >= 1
    @test info2.failures == 0
    return @allocated step!!(st1, method, prob, x1, F1, nothing)
end

function _r1_solve_alloc_probe(state, prob, x, method, opts)
    solve!!(state, prob, x, method, nothing, opts)
    return @allocated solve!!(state, prob, x, method, nothing, opts)
end

function _r3_step_alloc_probe(state, method, prob, x, Fx)
    _, _, st1, _ = step!!(state, method, prob, x, Fx, nothing)
    _, _, st2, info2 = step!!(st1, method, prob, x, Fx, nothing)
    @test st2.depth >= 1
    @test info2.failures == 0
    return @allocated step!!(st2, method, prob, x, Fx, nothing)
end

function _r3_solve_alloc_probe(solver, x_work, x0)
    copyto!(x_work, x0)
    solve!(solver, x_work)
    copyto!(x_work, x0)
    return @allocated solve!(solver, x_work)
end

@testset "Anderson Acceleration" begin
    @testset "Constructors, validation, effective_kappa_bar, and Adapt" begin
        p = Picard(; damping = 0.7)
        a_val = Anderson(p, Val(3))
        a_int = Anderson(p, 3)
        @test typeof(a_val) === typeof(a_int)
        @test a_val === a_int
        @test a_val.member === p
        @test a_val.c_s === 0.2
        @test a_val.κ_bar === nothing
        @test a_val.restart_factor === 10.0
        @test a_val.M_alpha === 25.0
        @test a_val.safeguard_cos === nothing
        @test a_val.grow_depth === false

        a_def = Anderson(p)
        @test a_def isa Anderson{4}

        # Without a member, every form wraps `Picard(; damping)`, undamped by default.
        @test Anderson() === Anderson(Picard(), Val(4))
        @test Anderson(4) === Anderson(Picard(), Val(4))
        @test Anderson(Val(3)) === Anderson(Picard(), Val(3))
        @test Anderson(2) isa Anderson{2, <:Picard}
        a_damp = Anderson(; damping = 0.5)
        @test a_damp isa Anderson{4, <:Picard}
        @test a_damp.member.damping == 0.5
        a_damp_val = Anderson(Val(3); damping = 0.5, c_s = 0.25)
        @test a_damp_val isa Anderson{3, <:Picard}
        @test a_damp_val.member.damping == 0.5
        @test a_damp_val.c_s == 0.25
        a_damp_int = Anderson(3; damping = 0.5, grow_depth = true)
        @test typeof(a_damp_int) === typeof(a_damp_val)
        @test a_damp_int.member.damping == 0.5
        @test a_damp_int.grow_depth === true
        for bad_m in (0, -1, true, false)
            @test_throws ArgumentError Anderson(bad_m)
        end
        @test_throws ArgumentError Anderson(Val(0))

        @test effective_kappa_bar(a_val, Float64) == inv(sqrt(eps(Float64)))
        @test effective_kappa_bar(a_val, Float32) == inv(sqrt(eps(Float32)))
        a_kb = Anderson(p, Val(2); κ_bar = 500.0, safeguard_cos = 0.942, grow_depth = true)
        @test effective_kappa_bar(a_kb, Float64) === 500.0
        @test effective_kappa_bar(a_kb, Float32) === 500.0f0

        # Invalid depths (including Bool)
        for bad_m in (0, -1, -4, true, false)
            @test_throws ArgumentError Anderson(p, bad_m)
            @test_throws ArgumentError Anderson(p, Val(bad_m))
        end
        @test_throws ArgumentError Anderson(p, Val(:three))
        @test_throws ArgumentError Anderson(p, Val(2.0))

        # Invalid c_s
        for bad_cs in (0.0, 1.0, -0.1, 1.2, NaN, Inf, -Inf)
            @test_throws ArgumentError Anderson(p, 2; c_s = bad_cs)
        end

        # Invalid restart_factor (must be >= 3 and finite)
        @test Anderson(p, 2; restart_factor = 3).restart_factor === 3.0
        for bad_rf in (2.999, 0.0, -5.0, NaN, Inf, -Inf)
            @test_throws ArgumentError Anderson(p, 2; restart_factor = bad_rf)
        end

        # Invalid M_alpha (must be >= 1 and finite)
        @test Anderson(p, 2; M_alpha = 1).M_alpha === 1.0
        for bad_ma in (0.999, 0.0, -1.0, NaN, Inf, -Inf)
            @test_throws ArgumentError Anderson(p, 2; M_alpha = bad_ma)
        end

        # Invalid κ_bar (must be nothing or > 1 and finite)
        @test Anderson(p, 2; κ_bar = 1.001).κ_bar === 1.001
        for bad_kb in (1.0, 0.5, 0.0, -2.0, NaN, Inf, -Inf)
            @test_throws ArgumentError Anderson(p, 2; κ_bar = bad_kb)
        end

        # Invalid safeguard_cos (must be nothing or in (0, 1) and finite)
        @test Anderson(p, 2; safeguard_cos = 0.942).safeguard_cos === 0.942
        for bad_sc in (0.0, 1.0, -0.2, 1.1, NaN, Inf, -Inf)
            @test_throws ArgumentError Anderson(p, 2; safeguard_cos = bad_sc)
        end

        # Adapt
        a32 = Adapt.adapt(ToReal{Float32}(), a_kb)
        @test a32 isa Anderson{2}
        @test a32.c_s === 0.2f0
        @test a32.κ_bar === 500.0f0
        @test a32.restart_factor === 10.0f0
        @test a32.M_alpha === 25.0f0
        @test a32.safeguard_cos === 0.942f0
        @test a32.grow_depth === true

        a32_noth = Adapt.adapt(ToReal{Float32}(), a_val)
        @test a32_noth isa Anderson{3}
        @test a32_noth.c_s === 0.2f0
        @test a32_noth.κ_bar === nothing
        @test a32_noth.safeguard_cos === nothing
        @test a32_noth.grow_depth === false
    end

    @testset "R1 depth limit and lifecycle guards" begin
        prob_r1 = NonlinearProblem((x, p) -> x - SVector(cos(x[1]), sin(x[2])))
        x0_r1 = SVector(0.5, 0.5)
        @test_throws ArgumentError init_state(
            Anderson(Picard(), Val(5)), prob_r1, x0_r1, nothing)
        @test_throws ArgumentError init_state(
            Anderson(Picard(), Val(5)), NonlinearProblem((x, p) -> x^2 - 2.0), 1.5, nothing)

        a_r1 = Anderson(Picard(; damping = 0.8), Val(3))
        st_r1 = init_state(a_r1, prob_r1, x0_r1, nothing)
        @test isbits(st_r1)
        @test residual_buffer(st_r1) === nothing

        # Un-started step!! guard
        Fx0_r1 = prob_r1.F(x0_r1, nothing)
        _, _, _, info_unstarted = step!!(st_r1, a_r1, prob_r1, x0_r1, Fx0_r1, nothing)
        @test info_unstarted.code == NONFINITE
        @test iszero(info_unstarted.stepnorm)

        # Start, step, finish, and verify post-finish step!! guard
        st_started = start!!(st_r1, a_r1, prob_r1, x0_r1, Fx0_r1, nothing, rnorm(Fx0_r1))
        x1, F1, st_step1, info1 = step!!(
            st_started, a_r1, prob_r1, x0_r1, Fx0_r1, nothing)
        @test info1.code == SUCCESS
        st_synced = sync_merit(st_step1, a_r1, rnorm(F1)^2)
        @test isbits(st_synced)
        _, _, st_finished = finish!!(
            st_step1, a_r1, prob_r1, x1, F1,
            SolverStatus{Float64}(
                SUCCESS, Int32(1), info1.fnorm, info1.stepnorm, Int32(0), false))
        _, _, _, info_after_finish = step!!(
            st_finished, a_r1, prob_r1, x1, F1, nothing)
        @test info_after_finish.code == NONFINITE

        # sync_merit forwards to member_state and preserves fnorm_best and x_best (R1)
        a_bt_r1 = Anderson(
            Newton(; globalization = Backtracking()), Val(3))
        st_bt_r1 = init_state(a_bt_r1, prob_r1, x0_r1, nothing)
        @test isnan(st_bt_r1.member_state.merit)
        st_bt_unstarted_sync = sync_merit(st_bt_r1, a_bt_r1, 0.25)
        @test isnan(st_bt_unstarted_sync.member_state.merit)
        st_bt_started = start!!(
            st_bt_r1, a_bt_r1, prob_r1, x0_r1, Fx0_r1, nothing, rnorm(Fx0_r1))
        _, _, st_bt_step1, _ = step!!(
            st_bt_started, a_bt_r1, prob_r1, x0_r1, Fx0_r1, nothing)
        st_bt_synced = sync_merit(st_bt_step1, a_bt_r1, 0.125)
        @test isbits(st_bt_synced)
        @test st_bt_synced.member_state.merit == 0.125
        @test st_bt_synced.member_state != st_bt_step1.member_state
        @test st_bt_synced.fnorm_best == st_bt_step1.fnorm_best
        @test st_bt_synced.x_best == st_bt_step1.x_best
        @test st_bt_synced.Fx_best == st_bt_step1.Fx_best
        @test st_bt_synced.fnorm₀ == st_bt_step1.fnorm₀

        # R3 lifecycle guard and sync_merit forwarding
        prob_r3 = NonlinearProblem((F, x, p) -> (F .= x .- cos.(x); F))
        x0_r3 = [0.5, 0.5]
        Fx0_r3 = similar(x0_r3)
        prob_r3.F(Fx0_r3, x0_r3, nothing)
        a_r3 = Anderson(
            Picard(; damping = 0.8, globalization = Backtracking()), Val(5))
        st_r3 = init_state(a_r3, prob_r3, x0_r3, nothing)
        @test residual_buffer(st_r3) === st_r3.Fx
        _, _, _, info_r3_unstarted = step!!(
            st_r3, a_r3, prob_r3, copy(x0_r3), copy(Fx0_r3), nothing)
        @test info_r3_unstarted.code == NONFINITE

        x_r3_1 = copy(x0_r3)
        F_r3_1 = copy(Fx0_r3)
        st_r3_started = start!!(
            st_r3, a_r3, prob_r3, x_r3_1, F_r3_1, nothing, rnorm(F_r3_1))
        x_r3_1, F_r3_1, st_r3_step1, info_r3_1 = step!!(
            st_r3_started, a_r3, prob_r3, x_r3_1, F_r3_1, nothing)
        @test info_r3_1.code == SUCCESS
        st_r3_synced = sync_merit(st_r3_step1, a_r3, 0.375)
        @test st_r3_synced.member_state.merit == 0.375
        @test st_r3_synced.member_state != st_r3_step1.member_state
        @test st_r3_synced.fnorm_best == st_r3_step1.fnorm_best
        @test st_r3_synced.x_best == st_r3_step1.x_best
        @test st_r3_synced.Fx_best == st_r3_step1.Fx_best
        _, _, st_r3_finished = finish!!(
            st_r3_step1, a_r3, prob_r3, x_r3_1, F_r3_1,
            SolverStatus{Float64}(
                SUCCESS, Int32(1), info_r3_1.fnorm, info_r3_1.stepnorm, Int32(0), false))
        _, _, _, info_r3_after_finish = step!!(
            st_r3_finished, a_r3, prob_r3, x_r3_1, F_r3_1, nothing)
        @test info_r3_after_finish.code == NONFINITE

        # R3 lifecycle guard with stateless inner globalization (member_state.merit === nothing)
        a_r3_stat = Anderson(Picard(; damping = 0.8), Val(5))
        st_r3_stat = init_state(a_r3_stat, prob_r3, x0_r3, nothing)
        @test isnothing(st_r3_stat.member_state.merit)
        _, _, _, info_r3_stat_unstarted = step!!(
            st_r3_stat, a_r3_stat, prob_r3, copy(x0_r3), copy(Fx0_r3), nothing)
        @test info_r3_stat_unstarted.code == NONFINITE
        x_r3_stat = copy(x0_r3)
        F_r3_stat = copy(Fx0_r3)
        st_r3_stat_started = start!!(
            st_r3_stat, a_r3_stat, prob_r3, x_r3_stat, F_r3_stat, nothing,
            rnorm(F_r3_stat))
        x_r3_stat, F_r3_stat, st_r3_stat_step1, info_r3_stat_1 = step!!(
            st_r3_stat_started, a_r3_stat, prob_r3, x_r3_stat, F_r3_stat, nothing)
        @test info_r3_stat_1.code == SUCCESS
        _, _, st_r3_stat_finished = finish!!(
            st_r3_stat_step1, a_r3_stat, prob_r3, x_r3_stat, F_r3_stat,
            SolverStatus{Float64}(
                SUCCESS, Int32(1), info_r3_stat_1.fnorm, info_r3_stat_1.stepnorm,
                Int32(0), false))
        _, _, _, info_r3_stat_after_finish = step!!(
            st_r3_stat_finished, a_r3_stat, prob_r3, x_r3_stat, F_r3_stat, nothing)
        @test info_r3_stat_after_finish.code == NONFINITE
    end

    @testset "Check 1: parity with a reference Anderson iteration at β = 0.7" begin
        # Coupled nonlinear fixed-point problem x = g(x), residual F(x) = x - g(x)
        K = @SMatrix [0.20 0.05 -0.04 0.02;
                      -0.03 0.25 0.06 -0.02;
                      0.04 -0.02 0.18 0.05;
                      0.01 0.04 -0.03 0.22]
        b = @SVector [0.4, -0.3, 0.25, -0.15]
        g_svec(x) = SVector(
            cos(0.35 * x[1]) + (K * sin.(x))[1] + b[1],
            cos(0.35 * x[2]) + (K * sin.(x))[2] + b[2],
            cos(0.35 * x[3]) + (K * sin.(x))[3] + b[3],
            cos(0.35 * x[4]) + (K * sin.(x))[4] + b[4]
        )
        g_vec(x) = [cos(0.35 * x[i]) + sum(K[i, j] * sin(x[j]) for j in 1:4) + b[i]
                    for i in 1:4]

        x0_vec = [0.1, -0.2, 0.3, -0.1]
        x0_svec = SVector{4, Float64}(x0_vec)

        # Reference Anderson iteration `_nlsolve_anderson_reference` at m = 8, β = 0.7
        x_ref8, iters_ref8, fnorm_ref8 = _nlsolve_anderson_reference(
            g_vec, x0_vec; m = 8, beta = 0.7, abstol = 1e-11)
        @test fnorm_ref8 <= 1e-11

        # R3 Anderson(Picard(; damping = 0.7), Val(8))
        prob_r3 = NonlinearProblem((F, x, p) -> (F .= x .- g_vec(x); F))
        a_r3 = Anderson(Picard(; damping = 0.7), Val(8))
        opts = Options(Float64; f_abstol = 1e-11, f_reltol = 0.0, x_abstol = 0.0,
            x_reltol = 0.0, max_iterations = 100)
        x_gs_r3, status_r3 = solve(prob_r3, x0_vec, a_r3; options = opts)
        @test status_r3.code == SUCCESS
        @test norm(x_gs_r3 - x_ref8) < 1e-10
        @test abs(Int(status_r3.iterations) - iters_ref8) <= 1

        # R1 Anderson(Picard(; damping = 0.7), Val(4)) vs the m = 4 reference
        x_ref4, iters_ref4, _ = _nlsolve_anderson_reference(
            g_vec, x0_vec; m = 4, beta = 0.7, abstol = 1e-11)
        prob_r1 = NonlinearProblem((x, p) -> x - g_svec(x))
        a_r1 = Anderson(Picard(; damping = 0.7), Val(4))
        x_gs_r1, status_r1 = solve(prob_r1, x0_svec, a_r1; options = opts)
        @test status_r1.code == SUCCESS
        @test norm(x_gs_r1 - x_ref4) < 1e-10
        @test abs(Int(status_r1.iterations) - iters_ref4) <= 1

        # 10D R3 fixed-point problem at m = 8 < n = 10, β = 0.7
        n10 = 10
        g_vec10(x) = [0.25 * cos(0.3 * x[i]) + 0.1 * sin(x[mod1(i + 1, n10)]) -
                      0.05 * x[mod1(i + 2, n10)] + 0.1 * i / n10 for i in 1:n10]
        x0_10 = [0.1 * (-1)^i * i / n10 for i in 1:n10]
        x_ref10, iters_ref10, _ = _nlsolve_anderson_reference(
            g_vec10, x0_10; m = 8, beta = 0.7, abstol = 1e-11)
        prob_r3_10 = NonlinearProblem((F, x, p) -> (F .= x .- g_vec10(x); F))
        x_gs_10, status_r3_10 = solve(prob_r3_10, x0_10, a_r3; options = opts)
        @test status_r3_10.code == SUCCESS
        @test norm(x_gs_10 - x_ref10) < 1e-10
        @test abs(Int(status_r3_10.iterations) - iters_ref10) <= 1
    end

    @testset "Check 2: Restart on residual growth, stagnation, step_failures, and best iterate" begin
        # 2a. Post-step residual growth restart (fnorm₊ > restart_factor * fnorm_best)
        F_growth_r1(x, p) = x[1] > 4.0 ? SVector(100.0 * x[1], 0.0) :
                            SVector(-1.0 + 0.1 * x[1], x[2])
        prob_growth_r1 = NonlinearProblem(F_growth_r1)
        a_growth = Anderson(Picard(; damping = 1.0), Val(3); restart_factor = 10.0)
        x0_g = SVector(0.0, 0.0)
        Fx0_g = prob_growth_r1.F(x0_g, nothing)
        st_g = init_state(a_growth, prob_growth_r1, x0_g, nothing)
        st_g = start!!(st_g, a_growth, prob_growth_r1, x0_g, Fx0_g, nothing, rnorm(Fx0_g))

        x1_g, F1_g, st_g1, info_g1 = step!!(
            st_g, a_growth, prob_growth_r1, x0_g, Fx0_g, nothing)
        @test info_g1.failures == 0
        @test st_g1.current_depth == 1

        # Step 2: accelerated trial x₊ ≈ [10.0, 0.0] has ‖F(x₊)‖ = 1000 > 10 * 0.9,
        # so step!! must discard x₊, fall back to x_m = [1.9, 0.0], clear history, and record failure!
        x2_g, F2_g, st_g2, info_g2 = step!!(
            st_g1, a_growth, prob_growth_r1, x1_g, F1_g, nothing)
        @test info_g2.failures == 1
        @test st_g2.depth == 0
        @test st_g2.current_depth == 1
        @test x2_g ≈ SVector(1.9, 0.0)
        @test F2_g ≈ SVector(-0.81, 0.0)

        # Also test pre-filter growth trigger when incoming Fx has rnorm(Fx) > restart_factor * fnorm_best
        Fx_huge = SVector(50.0, 0.0)
        _, _, st_g3, info_g3 = step!!(
            st_g2, a_growth, prob_growth_r1, x2_g, Fx_huge, nothing)
        @test info_g3.failures == 1
        @test st_g3.depth == 0
        @test st_g3.current_depth == 1

        # 2b. Stagnation restart when step returns x₊ ≈ x above tolerance
        prob_stag = NonlinearProblem((x, p) -> SVector(x[1] - 1.0, x[2] + 2.0))
        a_stag = Anderson(Picard(; damping = 1.0), Val(3))
        x0_s = SVector(2.0, -1.0)
        Fx0_s = prob_stag.F(x0_s, nothing)
        st_s = init_state(a_stag, prob_stag, x0_s, nothing)
        st_s = start!!(st_s, a_stag, prob_stag, x0_s, Fx0_s, nothing, rnorm(Fx0_s))
        _, _, st_s1, info_s1 = step!!(st_s, a_stag, prob_stag, x0_s, Fx0_s, nothing)
        @test info_s1.failures == 0

        # Next step enters with x_stag == x0_s (the shift of 0.25 * eps(Float64) rounds away),
        # so Δx_new = 0 while Δw_new is nonzero, and ONLY the Δx_new stagnation clause
        # catches it!
        x_stag = x0_s + SVector(0.25 * eps(Float64), 0.0)
        Fx_stag = SVector(0.8, 0.9)
        _, _, st_s2, info_s2 = step!!(
            st_s1, a_stag, prob_stag, x_stag, Fx_stag, nothing)
        @test info_s2.failures == 1
        @test st_s2.depth == 0
        @test st_s2.current_depth == 1

        # 2c. `failures` summed over steps equals the number of history clears
        step_counter = Ref(0)
        function F_multi_clear(x, p)
            step_counter[] += 1
            if x[1] > 3.0
                return SVector(50.0 * x[1], 0.0)
            end
            return SVector(-1.0 + 0.15 * x[1], 0.5 * x[2])
        end
        prob_mc = NonlinearProblem(F_multi_clear)
        st_mc = init_state(a_growth, prob_mc, SVector(0.0, 1.0), nothing)
        x_mc = SVector(0.0, 1.0)
        Fx_mc = prob_mc.F(x_mc, nothing)
        st_mc = start!!(st_mc, a_growth, prob_mc, x_mc, Fx_mc, nothing, rnorm(Fx_mc))
        total_failures = 0
        total_clears = 0
        for k in 1:6
            x_mc, Fx_mc, st_mc, info_mc = step!!(
                st_mc, a_growth, prob_mc, x_mc, Fx_mc, nothing)
            total_failures += Int(info_mc.failures)
            if k >= 2 && st_mc.depth == 0
                total_clears += 1
            end
        end
        @test total_clears >= 1
        @test total_failures == total_clears

        # 2d. Stopped early by max_iterations: returned iterate == iterate with smallest ‖F‖
        # recorded by a residual wrapper (in R1 and R3)
        recorded_r1 = Tuple{SVector{2, Float64}, Float64}[]
        opts_div = Options(Float64; f_abstol = 1e-14, f_reltol = 0.0,
            x_abstol = 0.0, x_reltol = 0.0, max_iterations = 5)
        function F_dip_then_grow_r1(x, p)
            Fx = norm(x) < 0.08 ? 50.0 * (x + SVector(1.0, 1.0)) :
                 (norm(x) > 1.5 ? 4.0 * x : x)
            push!(recorded_r1, (x, rnorm(Fx)))
            return Fx
        end
        prob_dip_r1 = NonlinearProblem(F_dip_then_grow_r1)
        x_best_r1, status_best_r1 = solve(
            prob_dip_r1, SVector(1.0, -0.5),
            Anderson(Picard(; damping = 0.95), Val(2));
            options = opts_div)
        @test status_best_r1.code == MAXITERS
        _, min_idx_r1 = findmin(last, recorded_r1)
        @test x_best_r1 == recorded_r1[min_idx_r1][1]
        @test status_best_r1.fnorm == recorded_r1[min_idx_r1][2]

        # Same check in R3
        recorded_r3 = Tuple{Vector{Float64}, Float64}[]
        function F_dip_then_grow_r3!(F, x, p)
            if norm(x) < 0.08
                @. F = 50.0 * (x + 1.0)
            elseif norm(x) > 1.5
                @. F = 4.0 * x
            else
                @. F = x
            end
            if eltype(x) === Float64
                push!(recorded_r3, (copy(x), rnorm(F)))
            end
            return F
        end
        prob_dip_r3 = NonlinearProblem(F_dip_then_grow_r3!)
        x_best_r3, status_best_r3 = solve(
            prob_dip_r3, [1.0, -0.5],
            Anderson(Picard(; damping = 0.95), Val(2));
            options = opts_div)
        @test status_best_r3.code == MAXITERS
        _, min_idx_r3 = findmin(last, recorded_r3)
        @test x_best_r3 == recorded_r3[min_idx_r3][1]
        @test status_best_r3.fnorm == recorded_r3[min_idx_r3][2]
    end

    @testset "Check 3: Length filter, angle filter, zero column, and M_alpha safeguard" begin
        # 3a. Orthogonal history: 0 columns dropped (failures == 0) in R1 (m = 4) and R3 (m = 6)
        prob4_r1 = NonlinearProblem((x, p) -> 0.5 * x)
        a4 = Anderson(Picard(; damping = 1.0), Val(4); c_s = 0.2, κ_bar = 1e4)
        e1 = SVector(1.0, 0.0, 0.0, 0.0)
        e2 = SVector(0.0, 1.0, 0.0, 0.0)
        e3 = SVector(0.0, 0.0, 1.0, 0.0)
        e4 = SVector(0.0, 0.0, 0.0, 1.0)

        st_orth = AndersonState{4}(
            init_state(a4.member, prob4_r1, e1, nothing),
            nothing,
            zero(e1),
            zero(e1),
            e1,
            nothing,
            (e2, e3, e4, zero(e1)),
            (e2, e3, e4, zero(e1)),
            Int32.((1, 2, 3, 0)),
            nothing,
            1.0,
            1.0,
            Int32(3),
            Int32(3),
            true
        )
        _, _, st_orth_next, info_orth = step!!(
            st_orth, a4, prob4_r1, e1, -e1, nothing)
        @test info_orth.failures == 0
        @test st_orth_next.depth == 4
        @test st_orth_next.current_depth == 4

        # R3 with m = 6 orthogonal columns in Float64 (default κ_bar = 1/sqrt(eps(Float64)))
        prob6_r3 = NonlinearProblem((F, x, p) -> (F .= 0.5 .* x; F))
        a6 = Anderson(Picard(; damping = 1.0), Val(6); c_s = 0.2)
        x6_in = zeros(Float64, 6)
        x6_in[1] = 1.0
        Fx6_in = -x6_in
        st6_init = init_state(a6, prob6_r3, x6_in, nothing)
        st6_init = start!!(st6_init, a6, prob6_r3, x6_in, Fx6_in, nothing, rnorm(Fx6_in))
        fill!(st6_init.x_prev, 0.0)
        fill!(st6_init.w_prev, 0.0)
        fill!(st6_init.ΔX, 0.0)
        fill!(st6_init.ΔW, 0.0)
        for j in 1:5
            st6_init.ΔX[j + 1, j] = 1.0
            st6_init.ΔW[j + 1, j] = 1.0
        end
        st6_init.ages[1:5] .= 1:5
        st6 = AndersonState{6}(
            st6_init.member_state, st6_init.Fx, st6_init.x_prev, st6_init.w_prev,
            st6_init.x_best, st6_init.Fx_best, st6_init.ΔX, st6_init.ΔW, st6_init.ages,
            st6_init.work, st6_init.fnorm₀, st6_init.fnorm_best, Int32(5), Int32(5), true)
        _, _, st6_next, info6 = step!!(
            st6, a6, prob6_r3, copy(x6_in), copy(Fx6_in), nothing)
        @test info6.failures == 0
        @test st6_next.depth == 6
        @test st6_next.current_depth == 6

        # 3b. Two almost parallel columns in the middle of the history:
        col3_par = normalize(e2 + 1e-6 * e3)
        st_par = AndersonState{4}(
            init_state(a4.member, prob4_r1, e1, nothing),
            nothing,
            zero(e1),
            zero(e1),
            e1,
            nothing,
            (e2, col3_par, e4, zero(e1)),
            (e2, col3_par, e4, zero(e1)),
            Int32.((1, 2, 3, 0)),
            nothing,
            1.0,
            1.0,
            Int32(3),
            Int32(3),
            true
        )
        _, _, st_par_next, info_par = step!!(
            st_par, a4, prob4_r1, e1, -e1, nothing)
        @test info_par.failures == 1
        @test st_par_next.depth == 3
        @test st_par_next.current_depth == 4
        # Newest (e1), col 2 (e2), and col 4 (e4) kept; col 3 permanently dropped!
        @test st_par_next.ΔW[1] == e1
        @test st_par_next.ΔW[2] == e2
        @test st_par_next.ΔW[3] == e4
        @test st_par_next.ages == Int32.((1, 2, 4, 0))
        F_kept = hcat(Vector(st_par_next.ΔW[1]), Vector(st_par_next.ΔW[2]), Vector(st_par_next.ΔW[3]))
        @test cond(F_kept) < effective_kappa_bar(a4, Float64)

        # 3c. Out-of-scale older column dropped by length filter and counted in failures:
        a4_len = Anderson(Picard(; damping = 1.0), Val(4); c_s = 0.5, κ_bar = 50.0)
        st_len = AndersonState{4}(
            init_state(a4_len.member, prob4_r1, e1, nothing),
            nothing,
            zero(e1),
            zero(e1),
            e1,
            nothing,
            (e2, 1e5 * e3, zero(e1), zero(e1)),
            (e2, 1e5 * e3, zero(e1), zero(e1)),
            Int32.((1, 2, 0, 0)),
            nothing,
            1.0,
            1.0,
            Int32(2),
            Int32(2),
            true
        )
        _, _, st_len_next, info_len = step!!(
            st_len, a4_len, prob4_r1, e1, -e1, nothing)
        @test info_len.failures == 1
        @test st_len_next.depth == 2
        @test st_len_next.current_depth == 3
        F_len_kept = hcat(Vector(st_len_next.ΔW[1]), Vector(st_len_next.ΔW[2]))
        @test cond(F_len_kept) < a4_len.κ_bar

        # 3d. Zero newest column triggers a counted restart and produces no NaN:
        st_zero_col = AndersonState{4}(
            init_state(a4.member, prob4_r1, e1, nothing),
            nothing,
            zero(e1),
            e1,
            e1,
            nothing,
            (e2, e3, zero(e1), zero(e1)),
            (e2, e3, zero(e1), zero(e1)),
            Int32.((1, 2, 0, 0)),
            nothing,
            1.0,
            1.0,
            Int32(2),
            Int32(2),
            true
        )
        x_zc, F_zc, st_zc_next, info_zc = step!!(
            st_zero_col, a4, prob4_r1, e1, -e1, nothing)
        @test info_zc.failures == 1
        @test st_zc_next.depth == 0
        @test st_zc_next.current_depth == 1
        @test all(isfinite, x_zc)
        @test all(isfinite, F_zc)

        # 3e. Post-solve M_alpha safeguard rejects oversized weights ∑|α_j| > M_alpha:
        # With Δw_new = 0.05 * e1 and w_k = 1.0 * e1, raw γ_1 ≈ 20.0, so α = (20, -19), ∑|α_j| = 39.
        # When M_alpha = 3.0, ∑|α_j| = 39 > 3.0 is rejected, falling back to x_m = 2.0 * e1 with failures == 1;
        # when M_alpha = 50.0, ∑|α_j| = 39 <= 50.0 is accepted, taking x₊ ≈ -19.0 * e1 with failures == 0.
        a_malpha_low = Anderson(Picard(; damping = 1.0), Val(2); M_alpha = 3.0, restart_factor = 100.0)
        a_malpha_high = Anderson(Picard(; damping = 1.0), Val(2); M_alpha = 50.0, restart_factor = 100.0)
        st_ma = AndersonState{2}(
            init_state(a_malpha_low.member, prob4_r1, e1, nothing),
            nothing,
            zero(e1),
            0.95 * e1,
            e1,
            nothing,
            (zero(e1), zero(e1)),
            (zero(e1), zero(e1)),
            Int32.((0, 0)),
            nothing,
            1.0,
            1.0,
            Int32(0),
            Int32(1),
            true
        )
        x_ma_low, _, _, info_ma_low = step!!(
            st_ma, a_malpha_low, prob4_r1, e1, -e1, nothing)
        @test info_ma_low.code == SUCCESS
        @test info_ma_low.failures == 1
        @test x_ma_low ≈ 2.0 * e1 atol = 1e-10

        x_ma_high, _, _, info_ma_high = step!!(
            st_ma, a_malpha_high, prob4_r1, e1, -e1, nothing)
        @test info_ma_high.code == SUCCESS
        @test info_ma_high.failures == 0
        @test x_ma_high ≈ -19.0 * e1 atol = 1e-6

        # 3g. M_alpha bounds the exact ∑|α_j| after an interior drop. The angle filter drops
        # column 2, which is parallel to the newest column f1, so the kept columns have ages 1
        # and 3, and γ = (1, 20/3) gives α = (1 - γ₁, γ₁, -γ₂, γ₂) = (0, 1, -20/3, 20/3) with
        # ∑|α_j| = 43/3 ≈ 14.33; the sum for adjacent columns would be 37/3 ≈ 12.33.
        f1 = SVector(1.0, 0.0, 0.0)
        f2 = SVector(0.0, 1.0, 0.0)
        f3 = SVector(0.0, 0.0, 1.0)
        w_drop = f1 + (20 / 3) * f2
        prob_drop_r1 = NonlinearProblem((x, p) -> -w_drop)
        prob_drop_r3 = NonlinearProblem((F, x, p) -> (F .= -w_drop; F))
        for (M_alpha, accepts) in ((13.0, false), (15.0, true))
            a_drop = Anderson(Picard(; damping = 1.0), Val(3); M_alpha, restart_factor = 100.0)
            st_drop = AndersonState{3}(
                init_state(a_drop.member, prob_drop_r1, f1, nothing), nothing,
                zero(f1), w_drop - f1, f1, nothing,
                (f1, f1, zero(f1)), (f1 + 0.01 * f3, f2, zero(f1)), Int32.((1, 2, 0)),
                nothing, 1.0, rnorm(w_drop), Int32(2), Int32(2), true)
            x_drop, _, st_drop_next, info_drop = step!!(
                st_drop, a_drop, prob_drop_r1, f1, -w_drop, nothing)
            @test st_drop_next.depth == 2
            @test st_drop_next.ages == Int32.((1, 3, 0))
            # one dropped column, and one rejected step when 14.33 > M_alpha
            @test info_drop.failures == (accepts ? 1 : 2)
            # the step of the member is f1 + w_drop
            @test x_drop ≈ (accepts ? -(20 / 3) * f1 : f1 + w_drop) atol = 1e-10

            x_r3 = Vector(f1)
            Fx_r3 = Vector(-w_drop)
            st3 = start!!(init_state(a_drop, prob_drop_r3, x_r3, nothing), a_drop,
                prob_drop_r3, x_r3, Fx_r3, nothing, rnorm(Fx_r3))
            st3.x_prev .= 0.0
            st3.w_prev .= w_drop .- f1
            st3.ΔX[:, 1] .= f1
            st3.ΔX[:, 2] .= f1
            st3.ΔW[:, 1] .= f1 .+ 0.01 .* f3
            st3.ΔW[:, 2] .= f2
            st3.ages[1:2] .= (1, 2)
            st3 = AndersonState{3}(
                st3.member_state, st3.Fx, st3.x_prev, st3.w_prev, st3.x_best, st3.Fx_best,
                st3.ΔX, st3.ΔW, st3.ages, st3.work, st3.fnorm₀, st3.fnorm_best,
                Int32(2), Int32(2), true)
            x3_out, _, st3_next, info3 = step!!(
                st3, a_drop, prob_drop_r3, x_r3, Fx_r3, nothing)
            @test st3_next.ages[1:2] == [1, 3]
            @test info3.failures == info_drop.failures
            @test x3_out ≈ Vector(x_drop) atol = 1e-10
        end

        # 3f. Angle safeguard (safeguard_cos) and growing depth (grow_depth)
        a_sc = Anderson(Picard(; damping = 1.0), Val(2); safeguard_cos = 0.9, grow_depth = true)
        x_sc, _, st_sc_next, info_sc = step!!(
            st_ma, a_sc, prob4_r1, e1, -e1, nothing)
        @test info_sc.failures == 1
        @test st_sc_next.depth == 0
        @test st_sc_next.current_depth == 1
        @test x_sc == 2.0 * e1
    end

    @testset "Check 4: Float32 Tikhonov regularisation on rank-deficient history" begin
        # Direct singular Gram matrix in Float32: two identical columns
        G32_sing = @SMatrix [1.0f0 1.0f0; 1.0f0 1.0f0]
        b32_sing = @SVector [0.5f0, 0.5f0]
        γ32 = solve_weights(G32_sing, b32_sing, SVector(1, 2), 2)
        @test eltype(γ32) === Float32
        @test all(isfinite, γ32)

        # Float32 `step!!` in R1 from a history of identical columns, which the angle filter
        # thins (the regularised solve itself is the `solve_weights` check above), and a
        # Float32 R3 `solve`
        prob32_r1 = NonlinearProblem((x, p) -> 0.5f0 * x - SVector(0.2f0, -0.1f0, 0.3f0))
        a32_r1 = Adapt.adapt(
            ToReal{Float32}(),
            Anderson(Picard(; damping = 1.0f0), Val(3); c_s = 1.0f-7, κ_bar = 1.0f12))
        e1_32 = SVector(1.0f0, 0.0f0, 0.0f0)
        st32_r1 = AndersonState{3}(
            init_state(a32_r1.member, prob32_r1, e1_32, nothing),
            nothing,
            zero(e1_32),
            zero(e1_32),
            e1_32,
            nothing,
            (e1_32, e1_32, e1_32),
            (e1_32, e1_32, e1_32),
            Int32.((1, 2, 3)),
            nothing,
            1.0f0,
            1.0f0,
            Int32(2),
            Int32(2),
            true
        )
        x32_out, F32_out, _, info32 = step!!(
            st32_r1, a32_r1, prob32_r1, e1_32, prob32_r1.F(e1_32, nothing), nothing)
        @test info32.code == SUCCESS
        @test info32.failures >= 1
        @test all(isfinite, x32_out)
        @test all(isfinite, F32_out)

        # Also in R3 Float32
        prob32_r3 = NonlinearProblem((F, x, p) -> (F .= 0.5f0 .* x .- 0.1f0; F))
        x0_32_r3 = Float32[1.0, -0.5, 0.25, 0.75]
        x32_sol, status32_r3 = solve(
            prob32_r3, x0_32_r3, a32_r1;
            options = Options(Float32; f_abstol = 1.0f-5, f_reltol = 0.0f0))
        @test status32_r3.code == SUCCESS
        @test all(isfinite, x32_sol)
    end

    @testset "Check 5: GMRES equivalence and affine contraction bound" begin
        # Non-normal asymmetric matrix A with κ(A) <= 10
        A = [1.2 0.25 -0.1 0.05;
             -0.15 0.9 0.2 -0.05;
             0.05 -0.2 1.1 0.15;
             -0.05 0.1 -0.15 0.85]
        κA = cond(A)
        @test κA <= 10.0
        b = [0.7, -0.4, 0.9, -0.6]
        x0 = [0.1, 0.2, -0.1, 0.3]

        # 5a. GMRES equivalence with no truncation / filter drop (ω = 1.0)
        a_gmres_r3 = Anderson(Picard(; damping = 1.0), Val(4);
            c_s = 1e-3, κ_bar = 1e20, M_alpha = 1e6, restart_factor = 1e6)
        prob_aff_r3 = NonlinearProblem((F, x, p) -> (mul!(F, A, x); F .-= b; F))
        st_gm = init_state(a_gmres_r3, prob_aff_r3, x0, nothing)
        x_curr = copy(x0)
        Fx_curr = A * x_curr - b
        st_gm = start!!(
            st_gm, a_gmres_r3, prob_aff_r3, x_curr, Fx_curr, nothing, rnorm(Fx_curr))

        x_curr, Fx_curr, st_gm, _ = step!!(
            st_gm, a_gmres_r3, prob_aff_r3, x_curr, Fx_curr, nothing)
        x_gm0 = _gmres_reference(A, b, x0, 0)
        @test norm(x_curr - (x_gm0 - (A * x_gm0 - b))) <=
              100 * κA^2 * eps(Float64) * norm(x_curr)

        for k in 1:4
            x_curr, Fx_curr, st_gm, info_gm = step!!(
                st_gm, a_gmres_r3, prob_aff_r3, x_curr, Fx_curr, nothing)
            @test info_gm.failures == 0
            x_gmk = _gmres_reference(A, b, x0, k)
            x_gmk_step = x_gmk - (A * x_gmk - b)
            @test norm(x_curr - x_gmk_step) <= 100 * κA^2 * eps(Float64) * norm(x_gmk_step)
            if k == 4
                @test norm(x_curr - x_gmk) <= 100 * κA^2 * eps(Float64) * norm(x_gmk)
            end
        end

        # Same GMRES equivalence in R1 (SVector{4, Float64})
        A_sv = SMatrix{4, 4, Float64}(A)
        b_sv = SVector{4, Float64}(b)
        x0_sv = SVector{4, Float64}(x0)
        prob_aff_r1 = NonlinearProblem((x, p) -> A_sv * x - b_sv)
        st_gm_r1 = init_state(a_gmres_r3, prob_aff_r1, x0_sv, nothing)
        x_sv = x0_sv
        Fx_sv = prob_aff_r1.F(x_sv, nothing)
        st_gm_r1 = start!!(
            st_gm_r1, a_gmres_r3, prob_aff_r1, x_sv, Fx_sv, nothing, rnorm(Fx_sv))
        x_sv, Fx_sv, st_gm_r1, _ = step!!(
            st_gm_r1, a_gmres_r3, prob_aff_r1, x_sv, Fx_sv, nothing)
        for k in 1:4
            x_sv, Fx_sv, st_gm_r1, _ = step!!(
                st_gm_r1, a_gmres_r3, prob_aff_r1, x_sv, Fx_sv, nothing)
            x_gmk = _gmres_reference(A, b, x0, k)
            x_gmk_step = x_gmk - (A * x_gmk - b)
            @test norm(Vector(x_sv) - x_gmk_step) <=
                  100 * κA^2 * eps(Float64) * norm(x_gmk_step)
        end

        # 5b. Affine contraction bound: ‖F(u₊)‖ <= ‖I - ωA‖₂ ‖F(u)‖ (1 + n eps(R))
        n = 4
        for ω in (0.5, 0.8), m_depth in (1, 2, 4), cs_val in (0.1, 0.6), gd in (false, true)
            q_factor = opnorm(I - ω * A)
            a_con = Anderson(Picard(; damping = ω), Val(m_depth);
                c_s = cs_val, κ_bar = 20.0, grow_depth = gd, restart_factor = 3.0)
            st_c = init_state(a_con, prob_aff_r1, x0_sv, nothing)
            u = x0_sv
            Fu = prob_aff_r1.F(u, nothing)
            st_c = start!!(st_c, a_con, prob_aff_r1, u, Fu, nothing, rnorm(Fu))
            for _ in 1:10
                fnorm_before = rnorm(Fu)
                fnorm_before < 1e-13 && break
                u, Fu, st_c, _ = step!!(st_c, a_con, prob_aff_r1, u, Fu, nothing)
                fnorm_after = rnorm(Fu)
                @test fnorm_after <= q_factor * fnorm_before * (1 + n * eps(Float64))
            end
        end
    end

    @testset "Check 6: current_depth invariant, R1/R3/JLArray/KA parity, 0-alloc, JET, and matrix sweep" begin
        # 6a. 1 <= current_depth <= m at every step, and current_depth == 1 at step 1
        # and at a restart step (depth == 0)
        F_cd(x, p) = x[1] > 2.5 ? SVector(40.0 * x[1], 0.0, 0.0) :
                     SVector(-1.0 + 0.2 * x[1], 0.4 * x[2] - 0.1, 0.6 * x[3] + 0.2)
        prob_cd = NonlinearProblem(F_cd)
        for m_d in (1, 2, 3, 4)
            a_cd = Anderson(Picard(; damping = 1.0), Val(m_d); restart_factor = 5.0)
            u = SVector(0.0, 0.5, -0.5)
            Fu = prob_cd.F(u, nothing)
            st_cd = init_state(a_cd, prob_cd, u, nothing)
            st_cd = start!!(st_cd, a_cd, prob_cd, u, Fu, nothing, rnorm(Fu))
            for k in 1:10
                u, Fu, st_cd, _ = step!!(st_cd, a_cd, prob_cd, u, Fu, nothing)
                @test 1 <= st_cd.current_depth <= m_d
                if k == 1 || st_cd.depth == 0
                    @test st_cd.current_depth == 1
                end
            end
        end

        # 6b. 1 step from the same state in R1 (SVector) and R3 (Array, JLArray, KA CPU())
        # agrees within n * eps(R) * ‖x₊‖ across steps 1..5
        n = 4
        prob_par_r1 = NonlinearProblem((x, p) -> SVector{4, Float64}(
            x[1] - cos(0.3 * x[1]) - 0.25,
            x[2] - cos(0.3 * x[2]) - 0.25,
            x[3] - cos(0.3 * x[3]) - 0.25,
            x[4] - cos(0.3 * x[4]) - 0.25
        ))
        prob_par_r3 = NonlinearProblem((F, x, p) -> begin
            @. F = x - cos(0.3 * x) - 0.25
            return F
        end)
        prob_par_ka = NonlinearProblem((F, x, p) -> begin
            _anderson_ka_residual_kernel!(CPU())(F, x; ndrange = length(x))
            KernelAbstractions.synchronize(CPU())
            return F
        end)

        a_par = Anderson(Picard(; damping = 0.8), Val(3))
        x0_vec = [0.2, 0.4, -0.1, 0.5]
        x_sv = SVector{4, Float64}(x0_vec)
        Fx_sv = prob_par_r1.F(x_sv, nothing)
        st_sv = init_state(a_par, prob_par_r1, x_sv, nothing)
        st_sv = start!!(st_sv, a_par, prob_par_r1, x_sv, Fx_sv, nothing, rnorm(Fx_sv))

        x_arr = copy(x0_vec)
        Fx_arr = similar(x_arr)
        prob_par_r3.F(Fx_arr, x_arr, nothing)
        st_arr = init_state(a_par, prob_par_r3, x_arr, nothing)
        st_arr = start!!(st_arr, a_par, prob_par_r3, x_arr, Fx_arr, nothing, rnorm(Fx_arr))

        x_jl = JLArray(x0_vec)
        Fx_jl = similar(x_jl)
        prob_par_r3.F(Fx_jl, x_jl, nothing)
        st_jl = init_state(a_par, prob_par_r3, x_jl, nothing)
        st_jl = start!!(st_jl, a_par, prob_par_r3, x_jl, Fx_jl, nothing, rnorm(Fx_jl))

        x_ka = copy(x0_vec)
        Fx_ka = similar(x_ka)
        prob_par_ka.F(Fx_ka, x_ka, nothing)
        st_ka = init_state(a_par, prob_par_ka, x_ka, nothing)
        st_ka = start!!(st_ka, a_par, prob_par_ka, x_ka, Fx_ka, nothing, rnorm(Fx_ka))

        for _ in 1:5
            x_sv, Fx_sv, st_sv, _ = step!!(
                st_sv, a_par, prob_par_r1, x_sv, Fx_sv, nothing)
            x_arr, Fx_arr, st_arr, _ = step!!(
                st_arr, a_par, prob_par_r3, x_arr, Fx_arr, nothing)
            x_jl, Fx_jl, st_jl, _ = step!!(
                st_jl, a_par, prob_par_r3, x_jl, Fx_jl, nothing)
            x_ka, Fx_ka, st_ka, _ = step!!(
                st_ka, a_par, prob_par_ka, x_ka, Fx_ka, nothing)

            tol = n * eps(Float64) * norm(x_sv)
            @test norm(x_arr - Vector(x_sv)) <= tol
            @test norm(Array(x_jl) - Vector(x_sv)) <= tol
            @test norm(x_ka - Vector(x_sv)) <= tol
        end

        # 6c. R1 isbits, @inferred, 0 allocations, and JET.@test_opt (SVector and Number)
        opts_r1 = Options(Float64; f_abstol = 1e-11, f_reltol = 0.0, max_iterations = 30)
        st_sv0 = init_state(a_par, prob_par_r1, SVector{4, Float64}(x0_vec), nothing)
        @test isbits(st_sv0)
        st_sv_started = start!!(
            st_sv0, a_par, prob_par_r1, SVector{4, Float64}(x0_vec),
            prob_par_r1.F(SVector{4, Float64}(x0_vec), nothing), nothing,
            rnorm(prob_par_r1.F(SVector{4, Float64}(x0_vec), nothing)))
        @inferred step!!(
            st_sv_started, a_par, prob_par_r1, SVector{4, Float64}(x0_vec),
            prob_par_r1.F(SVector{4, Float64}(x0_vec), nothing), nothing)
        @inferred solve!!(
            st_sv0, prob_par_r1, SVector{4, Float64}(x0_vec), a_par, nothing, opts_r1)
        if Base.JLOptions().check_bounds != 1
            # Skipped where a run forces `--check-bounds=yes`.
            @test _r1_step_alloc_probe(
                st_sv_started, a_par, prob_par_r1, SVector{4, Float64}(x0_vec),
                prob_par_r1.F(SVector{4, Float64}(x0_vec), nothing)) == 0
            @test _r1_solve_alloc_probe(
                st_sv0, prob_par_r1, SVector{4, Float64}(x0_vec), a_par, opts_r1) == 0
        end
        JET.@test_opt step!!(
            st_sv_started, a_par, prob_par_r1, SVector{4, Float64}(x0_vec),
            prob_par_r1.F(SVector{4, Float64}(x0_vec), nothing), nothing)

        # Scalar Number R1
        prob_num = NonlinearProblem((x, p) -> x - cos(0.5 * x))
        a_num = Anderson(Picard(; damping = 0.9), Val(2))
        st_num = init_state(a_num, prob_num, 0.5, nothing)
        @test isbits(st_num)
        x_num, status_num = solve(prob_num, 0.5, a_num; options = opts_r1)
        @test status_num.code == SUCCESS
        @test abs(prob_num.F(x_num, nothing)) <= 1e-11

        # R2 solve_batch! on CPU()
        X_batch = [SVector{4, Float64}(x0_vec) for _ in 1:4]
        S_batch = Vector{SolverStatus{Float64}}(undef, 4)
        @test begin
            solve_batch!(X_batch, S_batch, prob_par_r1, a_par; options = opts_r1)
            all(s -> s.code == SUCCESS, S_batch) &&
                all(x -> rnorm(prob_par_r1.F(x, nothing)) <= 1e-11, X_batch)
        end

        # 6d. R3 0 allocations for both step!! and solve! after warm-up
        old_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            x_work = copy(x0_vec)
            Fx_work = similar(x_work)
            prob_par_r3.F(Fx_work, x_work, nothing)
            st_arr0 = init_state(a_par, prob_par_r3, x_work, nothing)
            st_arr_started = start!!(
                st_arr0, a_par, prob_par_r3, x_work, Fx_work, nothing, rnorm(Fx_work))
            @test _r3_step_alloc_probe(
                st_arr_started, a_par, prob_par_r3, x_work, Fx_work) == 0

            solver_r3 = init(prob_par_r3, x0_vec, a_par; options = opts_r1)
            @test _r3_solve_alloc_probe(solver_r3, x_work, x0_vec) == 0
        finally
            BLAS.set_num_threads(old_threads)
        end

        # 6e. ELTYPES and ARRAY_BACKENDS coverage
        for T in ELTYPES
            R = real(T)
            c1 = T(R(4 // 5))
            c2 = T(R(3 // 5))
            prob_t_r1 = NonlinearProblem(_AndersonR1LinearF{T, R}(c1, c2))
            z0_r1 = SVector(T(R(1 // 10)), T(R(2 // 10)))
            a_t = Anderson(Picard(; damping = R(4 // 5)), Val(2))
            opts_t = Options(R; f_abstol = R(1e-5), f_reltol = zero(R), max_iterations = 30)
            z_sol_r1, st_t_r1 = solve(prob_t_r1, z0_r1, a_t; options = opts_t)
            @test st_t_r1.code == SUCCESS
            @test rnorm(prob_t_r1.F(z_sol_r1, nothing)) <= R(1e-5)

            prob_t_r3 = NonlinearProblem(_AndersonR3LinearF{T, R}(c1))
            for AT in ARRAY_BACKENDS
                z0_r3 = AT(T[T(R(1 // 10)), T(R(2 // 10))])
                _, st_t_r3 = solve(prob_t_r3, z0_r3, a_t; options = opts_t)
                @test st_t_r3.code == SUCCESS
            end
        end
    end

    @testset "the R1 trial residual is refused when it contains a ForwardDiff.Dual" begin
        # The residual is primal for its first three calls (x₀, the member steps of steps 1
        # and 2) and contains a captured Dual from the fourth on, which is the trial residual
        # F(x₊) of the first accelerated step.
        dA = ForwardDiff.Dual{Nothing}(0.0, 1.0)
        calls = Ref(0)
        F_late = (x, p) -> (calls[] += 1;
            Fx = x - SVector(cos(x[1]), sin(x[2])); calls[] <= 3 ? Fx : Fx .+ dA)
        @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
            NonlinearProblem(F_late), SVector(0.5, 0.5), Anderson(Picard(), Val(2)))
        @test calls[] == 4
    end

    @testset "R1 with an MVector iterate and an SVector residual" begin
        prob_mv = NonlinearProblem((x, p) -> SVector(x[1] - cos(x[1]), x[2] - sin(x[2]) -
                                                                       0.1))
        _, st_pic = solve(prob_mv, MVector(0.5, 0.5), Picard(; damping = 0.8))
        x_mv, st_mv = solve(prob_mv, MVector(0.5, 0.5), Anderson(Picard(; damping = 0.8), 2))
        @test st_pic.code === SUCCESS
        @test st_mv.code === SUCCESS
        @test x_mv isa MVector{2, Float64}
        @test x_mv != MVector(0.5, 0.5)
        @test rnorm(prob_mv.F(x_mv, nothing)) <=
              sqrt(eps()) * rnorm(prob_mv.F(SVector(0.5, 0.5), nothing))
    end

    # Every step of `_StallingPicard` is `STALLED`, so with `max_stalls = 2` the solve ends
    # `STALLED` after the second step, which `Anderson` accelerates
    @testset "Anderson is STALLED when its member step is, $(nameof(typeof(x₀)))" for (prob, x₀) in (
        (NonlinearProblem((x, p) -> x - 0.5 * cos.(x) - SVector(1.0, 2.0, 3.0)),
        SVector(10.0, -10.0, 5.0)),
        (NonlinearProblem((F, x, p) -> (F .= x .- 0.5 .* cos.(x) .- (1.0, 2.0, 3.0); F)),
        [10.0, -10.0, 5.0]))
        opt = Options(Float64; max_stalls = 2)
        _, st_m = solve(prob, x₀, _StallingPicard(); options = opt)
        x_a, st_a = solve(prob, x₀, Anderson(_StallingPicard(), 2); options = opt)
        @test st_m.code === STALLED
        @test st_a.code === STALLED
        @test st_m.iterations == 2
        @test st_a.iterations == 2
        @test x_a != x₀
    end

    # ∇S(x) = x³ + x - 1 componentwise, whose real root is 0.6823278038280193
    @testset "Anderson on a StationarityProblem, $(nameof(typeof(x₀)))" for x₀ in (
        SVector(0.1, 0.2), [0.1, 0.2])
        sp = StationarityProblem((x, p) -> sum(abs2, x) / 2 - sum(x) + sum(x .^ 4) / 4)
        x, st = solve(sp, x₀, Anderson(Picard(; damping = 0.5), 2); f_reltol = 1e-14)
        @test st.code === SUCCESS
        @test isapprox(Array(x), fill(0.6823278038280193, 2); rtol = 1e-12)
    end
end
