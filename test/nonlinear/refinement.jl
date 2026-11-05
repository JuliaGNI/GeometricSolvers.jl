using Adapt: adapt
using GeometricSolvers
using KernelAbstractions: get_backend
using LinearAlgebra: Diagonal, norm
using StaticArrays: SMatrix, SVector
using Test

const GS = GeometricSolvers

include("../helpers/matrix.jl")

# Moderately conditioned system (cond(J) ≈ 5e4) where Float32 factorisation + Refinement()
# recovers a Float64 Newton step without promoting to Float64.
const A_MOD = SMatrix{2, 2, Float64, 4}(1.0, 1.0, 1.0, 1.0 + 1e-4)
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
    u1 = a11 * x[1] + a12 * x[2]
    u2 = a21 * x[1] + a22 * x[2]
    F .= [u1 + T(0.1) * u1^2 - b1, u2 - T(0.2) * u2^2 - b2]
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
    u1 = a11 * x[1] + a12 * x[2]
    u2 = a21 * x[1] + a22 * x[2]
    F .= [u1 + T(0.1) * u1^2 - b1, u2 - T(0.2) * u2^2 - b2]
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

@testset "Refinement and QuasiNewton constructors" begin
    ref = Refinement()
    @test ref isa Refinement{Float64}
    @test ref.maxiter === Int32(2)
    @test ref.η === 1e-3
    @test isbits(ref)

    ref2 = Refinement(3; η = 1e-2)
    @test ref2.maxiter === Int32(3)
    @test ref2.η === 1e-2

    ref3 = Refinement(Int32(4), 0.05f0)
    @test ref3 isa Refinement{Float32}
    @test ref3.maxiter === Int32(4)
    @test ref3.η === 0.05f0

    @test_throws ArgumentError Refinement(0)
    @test_throws ArgumentError Refinement(-2)
    @test_throws ArgumentError Refinement(2; η = 0.0)
    @test_throws ArgumentError Refinement(2; η = -1e-3)
    @test_throws ArgumentError Refinement(2; η = 1.0)
    @test_throws ArgumentError Refinement(2; η = 1.5)
    @test_throws ArgumentError Refinement(2; η = NaN)
    @test_throws ArgumentError Refinement(2; η = Inf)

    for R in REAL_ELTYPES
        adapted = adapt(GS.ToReal{R}(), ref)
        @test adapted isa Refinement{R}
        @test adapted.maxiter === Int32(2)
        @test adapted.η === R(1e-3)
    end

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
    @test r_ref_rel < 1e-3
    @test r_ref_rel < r0_rel / 10

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
        d0, Refinement(3; η = 1e-3), bad_fac, nothing,
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

@testset "Mixed-precision Newton in R1 and R3 (LU, QR, SVD)" begin
    prob_mod_r1 = NonlinearProblem(f_mod_r1, j_mod_r1)
    prob_mod_r3 = NonlinearProblem(f_mod_r3!)
    x0_r1 = SVector{2, Float64}(2.0, 3.0)
    x_star = SVector{2, Float64}(1.0, 2.0)

    x_ref64, st_ref64 = solve(prob_mod_r1, x0_r1, Newton(; linsolve = LUFactorization()))
    @test st_ref64.code == SUCCESS
    @test norm(x_ref64 - x_star) < 1e-12

    for L in (LUFactorization, QRFactorization, SVDFactorization)
        m_ref = Newton(; linsolve = L(Float32), refinement = Refinement())
        x_r1, st_r1 = solve(prob_mod_r1, x0_r1, m_ref)
        @test st_r1.code == SUCCESS
        @test st_r1.fnorm < 1e-11
        @test !st_r1.promoted
        @test st_r1.step_failures == 0
        @test st_r1.iterations == st_ref64.iterations
        @test norm(x_r1 - x_star) < 1e-11

        for (_, B) in ARRAY_BACKENDS
            x0_r3 = B(collect(x0_r1))
            x_r3, st_r3 = solve(prob_mod_r3, x0_r3, m_ref)
            @test st_r3.code == SUCCESS
            @test st_r3.fnorm < 1e-11
            @test !st_r3.promoted
            @test st_r3.step_failures == 0
            @test st_r3.iterations == st_ref64.iterations
            @test norm(Array(x_r3) - collect(x_star)) < 1e-11
        end
    end
end

@testset "Promotion to working precision T on ill-conditioned systems" begin
    prob_ill_r1 = NonlinearProblem(f_ill_r1, j_ill_r1)
    prob_ill_r3 = NonlinearProblem(f_ill_r3!, j_ill_r3!)
    x0_r1 = SVector{2, Float64}(2.0, 3.0)
    x_star = SVector{2, Float64}(1.0, 2.0)

    # With Refinement(): refinement fails on the Float32 factor and promotes within the step.
    m_with_ref = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
    x_r1, st_r1 = solve(prob_ill_r1, x0_r1, m_with_ref)
    @test st_r1.code == SUCCESS
    @test st_r1.promoted
    @test st_r1.step_failures >= 1
    @test st_r1.fnorm < 1e-12
    @test norm(x_r1 - x_star) < 1e-10

    for (_, B) in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_r1))
        x_r3, st_r3 = solve(prob_ill_r3, x0_r3, m_with_ref)
        @test st_r3.code == SUCCESS
        @test st_r3.promoted
        @test st_r3.step_failures >= 1
        @test st_r3.fnorm < 1e-12
        @test norm(Array(x_r3) - collect(x_star)) < 1e-10
    end

    # With NoRefinement(): two slow steps with large linear-model error trigger promotion.
    m_no_ref = Newton(; linsolve = LUFactorization(Float32), refinement = NoRefinement())
    x_nr_r1, st_nr_r1 = solve(prob_ill_r1, x0_r1, m_no_ref)
    @test st_nr_r1.code == SUCCESS
    @test st_nr_r1.promoted
    @test st_nr_r1.step_failures >= 1
    @test st_nr_r1.fnorm < 1e-12

    for (_, B) in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_r1))
        x_nr_r3, st_nr_r3 = solve(prob_ill_r3, x0_r3, m_no_ref)
        @test st_nr_r3.code == SUCCESS
        @test st_nr_r3.promoted
        @test st_nr_r3.step_failures >= 1
        @test st_nr_r3.fnorm < 1e-12
    end
end

@testset "Precision-floor vs linear-quality detector" begin
    prob_fl_r1 = NonlinearProblem(f_floor_r1, j_floor_r1)
    prob_fl_r3 = NonlinearProblem(f_floor_r3!, j_floor_r3!)
    x0_r1 = SVector{2, Float64}(1.5, 1.8)
    opts = Options(
        Float64; f_abstol = 1e-14, f_reltol = 1e-14, x_abstol = 0.0, x_reltol = 0.0)
    m_floor = Newton(; linsolve = LUFactorization(Float32), refinement = NoRefinement())

    _, st_fl_r1 = solve(prob_fl_r1, x0_r1, m_floor; options = opts)
    @test st_fl_r1.code == STALLED
    @test !st_fl_r1.promoted
    @test st_fl_r1.fnorm < 1e-5

    for (_, B) in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_r1))
        _, st_fl_r3 = solve(prob_fl_r3, x0_r3, m_floor; options = opts)
        @test st_fl_r3.code == STALLED
        @test !st_fl_r3.promoted
        @test st_fl_r3.fnorm < 1e-5
    end
end

@testset "QuasiNewton early refactorisation on refinement failure" begin
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
    x_qn, st_qn = solve(prob_rot_r1, x0_rot, m_qn)
    @test st_qn.code == SUCCESS
    @test !st_qn.promoted
    @test st_qn.fnorm < 1e-12

    function f_rot_r3!(F, x, _)
        F .= [x[1]^3 - 3.0 * x[1] * x[2]^2 - 1.0, 3.0 * x[1]^2 * x[2] - x[2]^3 - 2.0]
        return F
    end
    prob_rot_r3 = NonlinearProblem(f_rot_r3!)
    for (_, B) in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_rot))
        x_qn_r3, st_qn_r3 = solve(prob_rot_r3, x0_r3, m_qn)
        @test st_qn_r3.code == SUCCESS
        @test !st_qn_r3.promoted
        @test st_qn_r3.fnorm < 1e-12
    end

    # When even a fresh Float32 factor cannot refine (cond(J) ≈ 5e7), QuasiNewton promotes
    # to Float64 and converges.
    prob_ill_r1 = NonlinearProblem(f_ill_r1, j_ill_r1)
    prob_ill_r3 = NonlinearProblem(f_ill_r3!, j_ill_r3!)
    m_qn_ill = QuasiNewton(
        3; linsolve = LUFactorization(Float32), refinement = Refinement())
    _, st_qn_ill_r1 = solve(prob_ill_r1, SVector{2, Float64}(2.0, 3.0), m_qn_ill)
    @test st_qn_ill_r1.code == SUCCESS
    @test st_qn_ill_r1.promoted
    @test st_qn_ill_r1.fnorm < 1e-12

    for (_, B) in ARRAY_BACKENDS
        x0_r3 = B([2.0, 3.0])
        _, st_qn_ill_r3 = solve(prob_ill_r3, x0_r3, m_qn_ill)
        @test st_qn_ill_r3.code == SUCCESS
        @test st_qn_ill_r3.promoted
        @test st_qn_ill_r3.fnorm < 1e-12
    end
end

@testset "Complex mixed-precision refinement and inference / allocation" begin
    for T in ELTYPES
        R = real(T)
        R === Float64 || continue
        prob_c_r1 = NonlinearProblem(f_mod_r1, j_mod_r1)
        x0_c = SVector{2, T}(T(2.0 + 0.1 * im), T(3.0 - 0.1 * im))
        m_c = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
        x_c, st_c = @inferred solve(prob_c_r1, x0_c, m_c)
        @test st_c.code == SUCCESS
        @test st_c.fnorm < 1e-11
        @test !st_c.promoted

        prob_c_r3 = NonlinearProblem(f_mod_r3!)
        x0_vec = [T(2.0 + 0.1 * im), T(3.0 - 0.1 * im)]
        solver = init(prob_c_r3, x0_vec, m_c)
        _, st_s = @inferred solve!(solver, copy(x0_vec))
        @test st_s.code == SUCCESS
        @test st_s.fnorm < 1e-11
        x_work = copy(x0_vec)
        alloc = @allocated solve!(solver, x_work)
        @test alloc == 0
    end
end
