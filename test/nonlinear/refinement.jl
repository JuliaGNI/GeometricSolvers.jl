using ADTypes: AutoFiniteDiff
using Adapt: adapt
using GeometricSolvers
using GeometricSolvers: solve!!
using JET: JET
using KernelAbstractions: CPU, @index, @kernel, synchronize
using LinearAlgebra: Diagonal, norm
using StaticArrays: SMatrix, SVector
using Test

const GS = GeometricSolvers

include("../helpers/matrix.jl")

# Moderately conditioned system (cond(J) ≈ 1.8e5) where Float32 factorisation has initial
# relative linear residual > 1e-4, 1 refinement pass stays above √eps(Float64), and Refinement()
# (up to 5 passes) recovers a Float64 Newton step without promoting to Float64.
const A_MOD = SMatrix{2, 2, Float64, 4}(1.3, 1.1, 1.1, 1.1 * 1.1 / 1.3 + 2e-5)
const U_MOD_STAR = A_MOD * SVector{2, Float64}(1.0, 2.0)
const B_MOD = SVector{2, Float64}(
    U_MOD_STAR[1] + 0.1 * U_MOD_STAR[1]^2, U_MOD_STAR[2] - 0.2 * U_MOD_STAR[2]^2)

@inline function f_mod_r1(x::SVector{2, T}, _) where {T}
    u = SMatrix{2, 2, T, 4}(A_MOD) * x
    b = SVector{2, T}(B_MOD)
    return SVector{2, T}(u[1] + T(0.1) * u[1]^2 - b[1], u[2] - T(0.2) * u[2]^2 - b[2])
end

@inline function j_mod_r1(x::SVector{2, T}, _) where {T}
    u = SMatrix{2, 2, T, 4}(A_MOD) * x
    d = SVector{2, T}(one(T) + T(0.2) * u[1], one(T) - T(0.4) * u[2])
    return Diagonal(d) * SMatrix{2, 2, T, 4}(A_MOD)
end

function f_mod_r3!(F, x, _)
    T = eltype(x)
    a11, a21, a12, a22 = T.(Tuple(A_MOD))
    b1, b2 = T.(Tuple(B_MOD))
    xh = x isa Array ? x : Array(x)
    u1 = a11 * xh[1] + a12 * xh[2]
    u2 = a21 * xh[1] + a22 * xh[2]
    r1 = u1 + T(0.1) * u1^2 - b1
    r2 = u2 - T(0.2) * u2^2 - b2
    if F isa Array
        @inbounds F[1] = r1
        @inbounds F[2] = r2
    else
        copyto!(F, [r1, r2])
    end
    return F
end

# Ill-conditioned system (cond(J) ≈ 5e7) where Float32 factorisation is nonsingular
# (fac.info == 0) but loses accuracy, triggering promotion both via Refinement() failure and
# via NoRefinement() slow-step linear-quality detection.
const A_ILL = SMatrix{2, 2, Float64, 4}(1.0, 1.0, 1.0, 1.0 + 1e-7)
const U_ILL_STAR = A_ILL * SVector{2, Float64}(1.0, 2.0)
const B_ILL = SVector{2, Float64}(
    U_ILL_STAR[1] + 0.1 * U_ILL_STAR[1]^2, U_ILL_STAR[2] - 0.2 * U_ILL_STAR[2]^2)

@inline function f_ill_r1(x::SVector{2, T}, _) where {T}
    u = SMatrix{2, 2, T, 4}(A_ILL) * x
    b = SVector{2, T}(B_ILL)
    return SVector{2, T}(u[1] + T(0.1) * u[1]^2 - b[1], u[2] - T(0.2) * u[2]^2 - b[2])
end

@inline function j_ill_r1(x::SVector{2, T}, _) where {T}
    u = SMatrix{2, 2, T, 4}(A_ILL) * x
    d = SVector{2, T}(one(T) + T(0.2) * u[1], one(T) - T(0.4) * u[2])
    return Diagonal(d) * SMatrix{2, 2, T, 4}(A_ILL)
end

function f_ill_r3!(F, x, _)
    T = eltype(x)
    a11, a21, a12, a22 = T.(Tuple(A_ILL))
    b1, b2 = T.(Tuple(B_ILL))
    xh = x isa Array ? x : Array(x)
    u1 = a11 * xh[1] + a12 * xh[2]
    u2 = a21 * xh[1] + a22 * xh[2]
    r1 = u1 + T(0.1) * u1^2 - b1
    r2 = u2 - T(0.2) * u2^2 - b2
    if F isa Array
        @inbounds F[1] = r1
        @inbounds F[2] = r2
    else
        copyto!(F, [r1, r2])
    end
    return F
end

function j_ill_r3!(J, x, _)
    T = eltype(x)
    a11, a21, a12, a22 = T.(Tuple(A_ILL))
    x_h = Array(x)
    u1 = a11 * x_h[1] + a12 * x_h[2]
    u2 = a21 * x_h[1] + a22 * x_h[2]
    d1 = one(T) + T(0.2) * u1
    d2 = one(T) - T(0.4) * u2
    copyto!(J, [d1*a11 d1*a12; d2*a21 d2*a22])
    return J
end

# Precision-floor problem: residual rounds its internal state to Float32 before subtracting, so
# ‖F(x)‖ floors near eps(Float32) ≈ 1.19e-7 while the linear model F(x) + J(x)d = 0 is solved
# accurately in Float32.
@inline function f_floor_r1(x::SVector{2, Float64}, _)
    r1 = Float64(Float32(x[1])^2 - 2.0f0)
    r2 = Float64(Float32(x[2])^2 - 3.0f0)
    return SVector{2, Float64}(r1, r2)
end

@inline function j_floor_r1(x::SVector{2, Float64}, _)
    return SMatrix{2, 2, Float64, 4}(2.0 * x[1], 0.0, 0.0, 2.0 * x[2])
end

function f_floor_r3!(F, x, _)
    x_h = Array(x)
    r1 = Float64(Float32(x_h[1])^2 - 2.0f0)
    r2 = Float64(Float32(x_h[2])^2 - 3.0f0)
    copyto!(F, [r1, r2])
    return F
end

function j_floor_r3!(J, x, _)
    x_h = Array(x)
    copyto!(J, [2.0*x_h[1] 0.0; 0.0 2.0*x_h[2]])
    return J
end

@kernel function ka_mixed_newton_kernel!(out_x, out_iters, out_prom, prob, x0, m, opts)
    i = @index(Global, Linear)
    st = GS.init_state(m, prob, x0, nothing)
    x_sol, _, status = solve!!(st, prob, x0, nothing, m, opts)
    @inbounds out_x[i] = x_sol
    @inbounds out_iters[i] = status.iterations
    @inbounds out_prom[i] = status.promoted
end

@testset "Refinement and QuasiNewton constructors" begin
    ref = Refinement()
    @test ref isa Refinement{Float64}
    @test fieldnames(Refinement) == (:η, :maxiter)
    @test ref.maxiter === Int32(5)
    @test ref.η === sqrt(eps(Float64))
    @test ref.eta === ref.η
    @test ref.max_iterations === ref.maxiter
    @test propertynames(ref) == (:η, :maxiter, :eta, :max_iterations)
    @test isbits(ref)

    ref_f32 = Refinement(Float32)
    @test ref_f32 isa Refinement{Float32}
    @test ref_f32.maxiter === Int32(5)
    @test ref_f32.η === sqrt(eps(Float32))

    ref_kw = Refinement(; eta = 1e-3, max_iterations = 4)
    @test ref_kw.maxiter === Int32(4)
    @test ref_kw.η === 1e-3

    ref2 = Refinement(3; η = 1e-2)
    @test ref2.maxiter === Int32(3)
    @test ref2.η === 1e-2

    ref3 = Refinement(0.05f0, Int32(4))
    @test ref3 isa Refinement{Float32}
    @test ref3.maxiter === Int32(4)
    @test ref3.η === 0.05f0

    ref_rat = Refinement(1 // 100, 2)
    @test ref_rat isa Refinement{Rational{Int}}
    @test ref_rat.η == 1 // 100
    @test ref_rat.maxiter === Int32(2)

    @test_throws ArgumentError Refinement(0)
    @test_throws ArgumentError Refinement(-2)
    @test_throws ArgumentError Refinement(; η = 0.0)
    @test_throws ArgumentError Refinement(; η = 1.0)
    @test_throws ArgumentError Refinement(; maxiter = 0)
    @test_throws ArgumentError Refinement(; η = 1e-3, eta = 1e-3)
    @test_throws ArgumentError Refinement(; maxiter = 2, max_iterations = 2)
    @test_throws ArgumentError Refinement(2; η = 0.0)
    @test_throws ArgumentError Refinement(2; η = -1e-3)
    @test_throws ArgumentError Refinement(2; η = 1.0)
    @test_throws ArgumentError Refinement(2; η = 1.5)
    @test_throws ArgumentError Refinement(2; η = NaN)
    @test_throws ArgumentError Refinement(2; η = Inf)

    for R in REAL_ELTYPES
        adapted = adapt(GS.ToReal{R}(), Refinement(5; η = 1e-4))
        @test adapted isa Refinement{R}
        @test adapted.maxiter === Int32(5)
        @test adapted.η === R(1e-4)
    end
    @test adapt(nothing, ref2) == ref2

    for M in (LUFactorization, QRFactorization, SVDFactorization)
        @test GS.promote_linsolve(M(Float32), Float64) === M(Float64)
        @test GS.promote_linsolve(M(Float16), Float32) === M(Float32)
        @test GS.promote_linsolve(M(Float32), ComplexF64) === M(Float64)
    end

    @test_throws ArgumentError Newton(;
        linsolve = LUFactorization(Float32), jacobian = AutoFiniteDiff(),
        refinement = Refinement())
    @test_throws ArgumentError Newton(;
        linsolve = LUFactorization(Float32), jacobian = AutoFiniteDiff(),
        refinement = true)
    prob_fd = NonlinearProblem((F, x, p) -> (F .= x .^ 2 .- 2.0; F))
    @test_throws ArgumentError init(
        prob_fd,
        [1.5, 1.5],
        Newton(; jacobian = AutoFiniteDiff(), refinement = Refinement()))

    qn = QuasiNewton(3)
    @test qn isa Newton
    @test qn.reuse === Int32(3)
    @test qn.linsolve === LUFactorization()
    @test qn.refinement === NoRefinement()

    qn_ref = QuasiNewton(4; linsolve = LUFactorization(Float32), refinement = Refinement())
    @test qn_ref.reuse === Int32(4)
    @test qn_ref.linsolve === LUFactorization(Float32)
    @test qn_ref.refinement === Refinement()
    @test_throws ArgumentError QuasiNewton(0)
    @test_throws ArgumentError QuasiNewton(-1)
end

@testset "refine!! unit checks and stopping rules" begin
    prob_r1 = NonlinearProblem(f_mod_r1, j_mod_r1)
    x0 = SVector{2, Float64}(2.0, 3.0)
    Fx = f_mod_r1(x0, nothing)
    fnorm = GS.rnorm(Fx)
    J = j_mod_r1(x0, nothing)
    fac32 = GS.factorize!!(nothing, LUFactorization(Float32), J)
    d0 = -GS.ldiv!!(nothing, fac32, Fx)

    # NoRefinement leaves d0 unchanged and reports failed = false.
    d_noref, failed_noref = GS.refine!!(
        d0, NoRefinement(), fac32, nothing, prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test d_noref === d0
    @test !failed_noref

    # Refinement reduces the linear residual ‖F(x0) + J(x0) d‖ below max(η, √eps(R)) * ‖F(x0)‖.
    r0_rel = GS.rnorm(Fx + J * d0) / fnorm
    @test r0_rel > 1e-4
    d_ref, failed_ref = GS.refine!!(
        d0, Refinement(), fac32, nothing, prob_r1, x0, Fx, fnorm, nothing, nothing)
    r_ref_rel = GS.rnorm(Fx + J * d_ref) / fnorm
    @test !failed_ref
    @test r_ref_rel <= sqrt(eps(Float64))
    @test r_ref_rel < r0_rel / 100

    # Exact direction already satisfies r_prev <= η_eff * fnorm and takes 0 iterations.
    fac64 = GS.factorize!!(nothing, LUFactorization(), J)
    d_exact = -GS.ldiv!!(nothing, fac64, Fx)
    d_zero_it, failed_zero_it = GS.refine!!(
        d_exact, Refinement(), fac32, nothing, prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test d_zero_it === d_exact
    @test !failed_zero_it

    # Effective threshold floor η_eff = max(R(ref.η), sqrt(eps(R))): a direction with relative
    # residual 1.2e-8 <= sqrt(eps(Float64)) ≈ 1.49e-8 passes immediately even when η = 1e-12.
    d_near = d_exact + J \ (1.2e-8 * Fx)
    r_near_rel = GS.rnorm(Fx + J * d_near) / fnorm
    @test 1e-12 < r_near_rel <= sqrt(eps(Float64))
    d_floor_eta, failed_floor_eta = GS.refine!!(
        d_near, Refinement(1; η = 1e-12), fac32, nothing,
        prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test d_floor_eta === d_near
    @test !failed_floor_eta

    # Exhausting maxiter when residual decreases but stays above η_eff * fnorm returns
    # (d_improved, true).
    d_one, failed_one = GS.refine!!(
        d0, Refinement(1; η = 1e-12), fac32, nothing,
        prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test failed_one
    @test GS.rnorm(Fx + J * d_one) < GS.rnorm(Fx + J * d0)

    # Non-decreasing residual !(r_curr < r_prev) stops immediately, returns failed = true,
    # and keeps the best direction before the diverging correction.
    bad_J = SMatrix{2, 2, Float32, 4}(-1.0f0, 0.0f0, 0.0f0, -1.0f0)
    bad_fac = GS.factorize!!(nothing, LUFactorization(), bad_J)
    d_bad, failed_bad = GS.refine!!(
        d0, Refinement(3; η = sqrt(eps(Float64))), bad_fac, nothing,
        prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test failed_bad
    @test d_bad === d0

    # Zero or non-finite fnorm handling.
    d_z, failed_z = GS.refine!!(
        d0, Refinement(), fac32, nothing, prob_r1, x0, zero(Fx), 0.0, nothing, nothing)
    @test d_z === d0
    @test !failed_z

    d_nf, failed_nf = GS.refine!!(
        d0, Refinement(), fac32, nothing, prob_r1, x0, Fx, NaN, nothing, nothing)
    @test d_nf === d0
    @test failed_nf
end

@testset "Mixed-precision Newton in R1 and R3 (LU, QR, SVD) and state invariants" begin
    prob_mod_r1 = NonlinearProblem(f_mod_r1, j_mod_r1)
    prob_mod_r3 = NonlinearProblem(f_mod_r3!)
    x0_r1 = SVector{2, Float64}(2.0, 3.0)
    x_star = SVector{2, Float64}(1.0, 2.0)
    opts = Options(Float64; f_abstol = 1e-12, f_reltol = 0.0)

    # State layout checks: R1 fac_T === nothing even with reuse; R3 LU aliases fac_T.factors === J.
    st_qn_r1 = GS.init_state(
        QuasiNewton(2; linsolve = LUFactorization(Float32)), prob_mod_r1, x0_r1, nothing)
    @test st_qn_r1.fac_T === nothing
    solver_lu_r3 = init(
        prob_mod_r3, [2.0, 3.0], Newton(; linsolve = LUFactorization(Float32));
        options = opts)
    @test solver_lu_r3.state.fac_T.factors === solver_lu_r3.state.J

    # step!! without start!! and after finish!! fails closed with NONFINITE and zero stepnorm.
    m_stat = Newton(;
        linsolve = LUFactorization(Float32), refinement = Refinement(),
        globalization = Static())
    st_unstarted = GS.init_state(m_stat, prob_mod_r1, x0_r1, nothing)
    Fx0 = f_mod_r1(x0_r1, nothing)
    x_u, _, _, info_u = GS.step!!(
        st_unstarted, m_stat, prob_mod_r1, x0_r1, Fx0, nothing)
    @test info_u.code == NONFINITE
    @test info_u.stepnorm == 0.0
    @test x_u == x0_r1

    st_dirty = GS.with_mixed(st_unstarted, true, Int32(2), Int32(1))
    st_started = GS.start!!(
        st_dirty, m_stat, prob_mod_r1, x0_r1, Fx0, nothing, GS.rnorm(Fx0))
    @test !st_started.promoted
    @test st_started.slow_steps == Int32(0)
    @test st_started.floor_steps == Int32(0)
    _, st_status = solve(prob_mod_r1, x0_r1, m_stat; options = opts)
    _, _, st_finished = GS.finish!!(
        st_started, m_stat, prob_mod_r1, x0_r1, Fx0, st_status)
    x_f, _, _, info_f = GS.step!!(
        st_finished, m_stat, prob_mod_r1, x0_r1, Fx0, nothing)
    @test info_f.code == NONFINITE
    @test info_f.stepnorm == 0.0
    @test x_f == x0_r1

    x_ref64, st_ref64 = solve(
        prob_mod_r1, x0_r1, Newton(; linsolve = LUFactorization()); options = opts)
    @test st_ref64.code == SUCCESS
    @test norm(x_ref64 - x_star) < 1e-12

    for L in (LUFactorization, QRFactorization, SVDFactorization)
        m_ref = Newton(; linsolve = L(Float32), refinement = Refinement())
        x_r1, st_r1 = solve(prob_mod_r1, x0_r1, m_ref; options = opts)
        @test st_r1.code == SUCCESS
        @test st_r1.fnorm < 1e-11
        @test !st_r1.promoted
        @test st_r1.step_failures == 0
        @test st_r1.iterations == st_ref64.iterations
        @test norm(x_r1 - x_star) < 1e-10
        @test isfinite(st_r1.gradratio) && st_r1.gradratio > 0
        if L === SVDFactorization
            @test st_r1.rank == Int32(2)
            @test isfinite(st_r1.nullnorm)
        else
            @test st_r1.rank == Int32(-1)
            @test isnan(st_r1.nullnorm)
        end

        for B in ARRAY_BACKENDS
            x0_r3 = B(collect(x0_r1))
            x_r3, st_r3 = solve(prob_mod_r3, x0_r3, m_ref; options = opts)
            @test st_r3.code == SUCCESS
            @test st_r3.fnorm < 1e-11
            @test !st_r3.promoted
            @test st_r3.step_failures == 0
            @test st_r3.iterations == st_ref64.iterations
            @test norm(Array(x_r3) - collect(x_star)) < 1e-10
            @test isfinite(st_r3.gradratio) && st_r3.gradratio > 0
            if L === SVDFactorization
                @test st_r3.rank == Int32(2)
                @test isfinite(st_r3.nullnorm)
            else
                @test st_r3.rank == Int32(-1)
                @test isnan(st_r3.nullnorm)
            end
        end
    end
end

@testset "Promotion to working precision T on ill-conditioned systems" begin
    prob_ill_r1 = NonlinearProblem(f_ill_r1, j_ill_r1)
    prob_ill_r3 = NonlinearProblem(f_ill_r3!, j_ill_r3!)
    x0_r1 = SVector{2, Float64}(2.0, 3.0)
    x_star = SVector{2, Float64}(1.0, 2.0)
    opts = Options(Float64; f_abstol = 1e-12, f_reltol = 0.0)

    # With Refinement(): refinement fails on the Float32 factor and promotes within the step.
    for L in (LUFactorization, QRFactorization, SVDFactorization)
        m_with_ref = Newton(; linsolve = L(Float32), refinement = Refinement())
        x_r1, st_r1 = solve(prob_ill_r1, x0_r1, m_with_ref; options = opts)
        @test st_r1.code == SUCCESS
        @test st_r1.promoted
        @test st_r1.step_failures >= 1
        @test st_r1.fnorm < 1e-12
        @test norm(x_r1 - x_star) < 1e-7
        @test isfinite(st_r1.gradratio)
        if L === SVDFactorization
            @test st_r1.rank == Int32(2)
            @test isfinite(st_r1.nullnorm)
        end

        for B in ARRAY_BACKENDS
            x0_r3 = B(collect(x0_r1))
            x_r3, st_r3 = solve(prob_ill_r3, x0_r3, m_with_ref; options = opts)
            @test st_r3.code == SUCCESS
            @test st_r3.promoted
            @test st_r3.step_failures >= 1
            @test st_r3.fnorm < 1e-12
            @test norm(Array(x_r3) - collect(x_star)) < 1e-7
            @test isfinite(st_r3.gradratio)
            if L === SVDFactorization
                @test st_r3.rank == Int32(2)
                @test isfinite(st_r3.nullnorm)
            end
        end
    end

    # With NoRefinement(): two slow steps with large linear-model error trigger promotion
    # and persist promoted = true in NewtonState.
    m_no_ref = Newton(; linsolve = LUFactorization(Float32), refinement = NoRefinement())
    x_nr_r1, st_nr_r1 = solve(prob_ill_r1, x0_r1, m_no_ref; options = opts)
    @test st_nr_r1.code == SUCCESS
    @test st_nr_r1.promoted
    @test st_nr_r1.step_failures >= 1
    @test st_nr_r1.fnorm < 1e-12
    @test norm(x_nr_r1 - x_star) < 1e-7

    solver_nr = init(prob_ill_r3, collect(x0_r1), m_no_ref; options = opts)
    _, st_nr_s = solve!(solver_nr)
    @test st_nr_s.code == SUCCESS
    @test st_nr_s.promoted
    @test solver_nr.state.promoted

    for B in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_r1))
        x_nr_r3, st_nr_r3 = solve(prob_ill_r3, x0_r3, m_no_ref; options = opts)
        @test st_nr_r3.code == SUCCESS
        @test st_nr_r3.promoted
        @test st_nr_r3.step_failures >= 1
        @test st_nr_r3.fnorm < 1e-12
        @test norm(Array(x_nr_r3) - collect(x_star)) < 1e-7
    end
end

@testset "Precision-floor vs linear-quality detector" begin
    prob_fl_r1 = NonlinearProblem(f_floor_r1, j_floor_r1)
    prob_fl_r3 = NonlinearProblem(f_floor_r3!, j_floor_r3!)
    x0_r1 = SVector{2, Float64}(1.5, 1.8)
    opts = Options(
        Float64; f_abstol = 1e-14, f_reltol = 1e-14, x_abstol = 0.0, x_reltol = 0.0,
        max_iterations = 5, max_stalls = 5)
    m_floor = Newton(; linsolve = LUFactorization(Float32), refinement = NoRefinement())

    _, st_fl_r1 = solve(prob_fl_r1, x0_r1, m_floor; options = opts)
    @test st_fl_r1.code == STALLED
    @test !st_fl_r1.promoted
    @test st_fl_r1.fnorm < 1e-5

    for B in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_r1))
        _, st_fl_r3 = solve(prob_fl_r3, x0_r3, m_floor; options = opts)
        @test st_fl_r3.code == STALLED
        @test !st_fl_r3.promoted
        @test st_fl_r3.fnorm < 1e-5
    end

    # High-nonlinearity O(1) residual with exact linear solve is not misclassified as TF floor.
    prob_high = NonlinearProblem(
        (x, p) -> SVector(x[1]^20, x[2]^20),
        (x, p) -> SMatrix{2, 2, Float64, 4}(
            20.0 * x[1]^19, 0.0, 0.0, 20.0 * x[2]^19))
    x_h = SVector(1.2, 1.2)
    Fx_h = prob_high.F(x_h, nothing)
    st_h = GS.init_state(m_floor, prob_high, x_h, nothing)
    st_h = GS.start!!(
        st_h, m_floor, prob_high, x_h, Fx_h, nothing, GS.rnorm(Fx_h))
    x_h, Fx_h, st_h, info_h1 = GS.step!!(
        st_h, m_floor, prob_high, x_h, Fx_h, nothing)
    x_h, Fx_h, st_h, info_h2 = GS.step!!(
        st_h, m_floor, prob_high, x_h, Fx_h, nothing)
    @test info_h1.code == SUCCESS
    @test info_h2.code == SUCCESS
end

@testset "QuasiNewton early refactorisation and chord-step handling" begin
    opts = Options(Float64; f_abstol = 1e-12, f_reltol = 0.0)

    # Nonlinear system with strong Jacobian rotation so that reusing the step-1 factor at step 2
    # fails Refinement(1; η = 1e-2), while a fresh Float32 factor at step 2 succeeds without
    # promoting to Float64.
    function f_rot_r1(x, _)
        return SVector(
            x[1]^3 - 3.0 * x[1] * x[2]^2 - 1.0,
            3.0 * x[1]^2 * x[2] - x[2]^3 - 2.0)
    end
    prob_rot_r1 = NonlinearProblem(f_rot_r1)
    x0_rot = SVector{2, Float64}(2.0, 0.5)

    m_qn = QuasiNewton(
        5; linsolve = LUFactorization(Float32), refinement = Refinement(1; η = 1e-2))
    x_qn, st_qn = solve(prob_rot_r1, x0_rot, m_qn; options = opts)
    @test st_qn.code == SUCCESS
    @test !st_qn.promoted
    @test st_qn.fnorm < 1e-12
    @test norm(x_qn - x0_rot) > 0.1

    function f_rot_r3!(F, x, _)
        xh = x isa Array ? x : Array(x)
        r1 = xh[1]^3 - 3.0 * xh[1] * xh[2]^2 - 1.0
        r2 = 3.0 * xh[1]^2 * xh[2] - xh[2]^3 - 2.0
        if F isa Array
            @inbounds F[1] = r1
            @inbounds F[2] = r2
        else
            copyto!(F, [r1, r2])
        end
        return F
    end
    function j_rot_r3!(J, x, _)
        xh = x isa Array ? x : Array(x)
        j11 = 3.0 * xh[1]^2 - 3.0 * xh[2]^2
        j12 = -6.0 * xh[1] * xh[2]
        j21 = 6.0 * xh[1] * xh[2]
        j22 = 3.0 * xh[1]^2 - 3.0 * xh[2]^2
        copyto!(J, [j11 j12; j21 j22])
        return J
    end
    prob_rot_r3 = NonlinearProblem(f_rot_r3!, j_rot_r3!)
    for B in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_rot))
        x_qn_r3, st_qn_r3 = solve(prob_rot_r3, x0_r3, m_qn; options = opts)
        @test st_qn_r3.code == SUCCESS
        @test !st_qn_r3.promoted
        @test st_qn_r3.fnorm < 1e-12
        @test norm(Array(x_qn_r3) - collect(x0_rot)) > 0.1
    end

    # Chord steps with NoRefinement() trigger early refactorisation on slow progress rather than
    # counting stale chord steps toward promotion.
    prob_chord = NonlinearProblem((x, p) -> SVector(x[1]^2 - 2.0, x[2]^2 - 3.0))
    m_chord = QuasiNewton(
        3; linsolve = LUFactorization(Float32), refinement = NoRefinement(),
        globalization = Static())
    _, st_chord = solve(prob_chord, SVector(10.0, 10.0), m_chord; options = opts)
    @test st_chord.code == SUCCESS
    @test !st_chord.promoted

    # When even a fresh Float32 factor cannot refine (cond(J) ≈ 5e7), QuasiNewton promotes
    # to Float64 and converges.
    prob_ill_r1 = NonlinearProblem(f_ill_r1, j_ill_r1)
    prob_ill_r3 = NonlinearProblem(f_ill_r3!, j_ill_r3!)
    m_qn_ill = QuasiNewton(
        3; linsolve = LUFactorization(Float32), refinement = Refinement())
    _, st_qn_ill_r1 = solve(
        prob_ill_r1, SVector{2, Float64}(2.0, 3.0), m_qn_ill; options = opts)
    @test st_qn_ill_r1.code == SUCCESS
    @test st_qn_ill_r1.promoted
    @test st_qn_ill_r1.fnorm < 1e-12

    for B in ARRAY_BACKENDS
        x0_r3 = B([2.0, 3.0])
        _, st_qn_ill_r3 = solve(prob_ill_r3, x0_r3, m_qn_ill; options = opts)
        @test st_qn_ill_r3.code == SUCCESS
        @test st_qn_ill_r3.promoted
        @test st_qn_ill_r3.fnorm < 1e-12
    end
end

function _measure_solve_alloc!(solver, x, x0)
    copyto!(x, x0)
    solve!(solver, x)
    copyto!(x, x0)
    return @allocated solve!(solver, x)
end

function _measure_solve_r1_alloc(st, prob, x0, m, opts)
    solve!!(st, prob, x0, nothing, m, opts)
    return @allocated solve!!(st, prob, x0, nothing, m, opts)
end

@testset "Complex mixed-precision refinement, KA @kernel, two-size @allocated, and JET" begin
    x_star = SVector{2, Float64}(1.0, 2.0)
    opts = Options(Float64; f_abstol = 1e-12, f_reltol = 0.0)
    for T in ELTYPES
        R = real(T)
        R === Float64 || continue
        prob_c_r1 = NonlinearProblem(f_mod_r1, j_mod_r1)
        x0_c = T <: Complex ?
               SVector{2, T}(T(2.0 + 0.1 * im), T(3.0 - 0.1 * im)) :
               SVector{2, T}(T(2.0), T(3.0))
        m_c = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
        x_c, st_c = @inferred solve(prob_c_r1, x0_c, m_c; options = opts)
        @test st_c.code == SUCCESS
        @test st_c.fnorm < 1e-11
        @test !st_c.promoted
        @test norm(x_c - SVector{2, T}(x_star)) < 1e-10

        prob_c_r3 = NonlinearProblem(f_mod_r3!)
        x0_vec = collect(x0_c)
        solver = init(prob_c_r3, x0_vec, m_c; options = opts)
        _, st_s = @inferred solve!(solver, copy(x0_vec))
        @test st_s.code == SUCCESS
        @test st_s.fnorm < 1e-11
        x_work = copy(x0_vec)
        alloc = _measure_solve_alloc!(solver, x_work, x0_vec)
        @test alloc == 0
    end

    # KernelAbstractions @kernel on CPU() for R1 SVector with LUFactorization(Float32)
    # matches R3 iteration count and promoted flag with 0 allocations after warm-up.
    prob_mod_r1 = NonlinearProblem(f_mod_r1, j_mod_r1)
    prob_mod_r3 = NonlinearProblem(f_mod_r3!)
    x0_r1 = SVector{2, Float64}(2.0, 3.0)
    m_ka = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
    _, st_r3_ref = solve(prob_mod_r3, collect(x0_r1), m_ka; options = opts)

    out_x = Vector{SVector{2, Float64}}(undef, 1)
    out_iters = Vector{Int32}(undef, 1)
    out_prom = Vector{Bool}(undef, 1)
    kern! = ka_mixed_newton_kernel!(CPU(), 1)
    kern!(out_x, out_iters, out_prom, prob_mod_r1, x0_r1, m_ka, opts; ndrange = 1)
    synchronize(CPU())
    @test out_iters[1] == st_r3_ref.iterations
    @test out_prom[1] == st_r3_ref.promoted
    @test norm(out_x[1] - x_star) < 1e-10

    st_warm = GS.init_state(m_ka, prob_mod_r1, x0_r1, nothing)
    alloc_r1 = _measure_solve_r1_alloc(st_warm, prob_mod_r1, x0_r1, m_ka, opts)
    @test alloc_r1 == 0

    # Two-size (N₁ = 4, N₂ = 8) R3 @allocated solve!(solver, x) check.
    function make_nd_prob(N::Int)
        function f_nd!(F, x, _)
            @inbounds for i in 1:N
                F[i] = (1.0 + 0.1 * i) * x[i] + 0.05 * x[i]^2 - 1.0
            end
            return F
        end
        function j_nd!(J, x, _)
            fill!(J, 0.0)
            @inbounds for i in 1:N
                J[i, i] = (1.0 + 0.1 * i) + 0.1 * x[i]
            end
            return J
        end
        return NonlinearProblem(f_nd!, j_nd!)
    end
    prob_n1 = make_nd_prob(4)
    prob_n2 = make_nd_prob(8)
    x0_n1 = fill(1.5, 4)
    x0_n2 = fill(1.5, 8)
    solver_n1 = init(prob_n1, x0_n1, m_ka; options = opts)
    solver_n2 = init(prob_n2, x0_n2, m_ka; options = opts)
    w_n1 = copy(x0_n1)
    w_n2 = copy(x0_n2)
    b1 = _measure_solve_alloc!(solver_n1, w_n1, x0_n1)
    b2 = _measure_solve_alloc!(solver_n2, w_n2, x0_n2)
    @test b1 == 0
    @test b2 == 0
    @test abs(b2 - b1) <= 64

    # JET.@test_opt target_modules = (GeometricSolvers,)
    JET.@test_opt target_modules=(GeometricSolvers,) solve(
        prob_mod_r1, x0_r1, m_ka; options = opts)
    copyto!(w_n1, x0_n1)
    JET.@test_opt target_modules=(GeometricSolvers,) solve!(solver_n1, w_n1)
end
