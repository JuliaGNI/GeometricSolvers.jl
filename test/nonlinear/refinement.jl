using ADTypes: AutoFiniteDiff
using Adapt: adapt
using GeometricSolvers
using GeometricSolvers: solve!!
using JET: JET
using KernelAbstractions: CPU, @index, @kernel, synchronize
using LinearAlgebra: Diagonal, norm
using Random: Xoshiro, randn
using StaticArrays: SMatrix, SVector
using Test

const GS = GeometricSolvers

include("../helpers/matrix.jl")

# The allocation probes sit behind a function barrier: `@allocated` measures the call it wraps, so
# each probe warms its own path first and then measures the same call again. The step count beside
# a measurement is the warm-up solve's; the measured call takes the same deterministic path.
# `allocations` is the control: it shows that the barrier sees an allocation on this build, so that
# a measured zero is the solver's and not the probe's.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@noinline function _measure_solve_alloc!(solver::S, x::X, x0::X) where {S, X}
    copyto!(x, x0)
    _, status = solve!(solver, x)
    copyto!(x, x0)
    alloc = @allocated solve!(solver, x)
    return alloc, Int(status.iterations)
end

@noinline function _measure_solve_r1_alloc(
        st::S, prob::P, x0::X, m::M, opts::O) where {S, P, X, M, O}
    solve!!(st, prob, x0, m, nothing, opts)
    return @allocated solve!!(st, prob, x0, m, nothing, opts)
end

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
    x_sol, status, _ = solve!!(st, prob, x0, m, nothing, opts)
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
    @test propertynames(ref) == (:η, :maxiter)
    @test isbits(ref)

    ref_f32 = Refinement(Float32)
    @test ref_f32 isa Refinement{Float32}
    @test ref_f32.maxiter === Int32(5)
    @test ref_f32.η === sqrt(eps(Float32))

    ref_kw = Refinement(; η = 1e-3, maxiter = 4)
    @test ref_kw.maxiter === Int32(4)
    @test ref_kw.η === 1e-3

    ref2 = Refinement(Float32; η = 1e-2, maxiter = 3)
    @test ref2 isa Refinement{Float32}
    @test ref2.maxiter === Int32(3)
    @test ref2.η === 1.0f-2

    ref3 = Refinement(0.05f0, Int32(4))
    @test ref3 isa Refinement{Float32}
    @test ref3.maxiter === Int32(4)
    @test ref3.η === 0.05f0

    ref_rat = Refinement(1 // 100, 2)
    @test ref_rat isa Refinement{Rational{Int}}
    @test ref_rat.η == 1 // 100
    @test ref_rat.maxiter === Int32(2)

    # The first positional argument is always `η`: an integer there is no `maxiter`, and the
    # keywords are `η` and `maxiter` alone.
    @test_throws ArgumentError Refinement(1)
    @test_throws ArgumentError Refinement(3)
    @test_throws MethodError Refinement(Float64, 3)
    @test_throws MethodError Refinement(; eta = 1e-3)
    @test_throws MethodError Refinement(; max_iterations = 4)

    @test_throws ArgumentError Refinement(; η = 0.0)
    @test_throws ArgumentError Refinement(; η = 1.0)
    @test_throws ArgumentError Refinement(; maxiter = 0)
    @test_throws ArgumentError Refinement(1e-3, 0)
    @test_throws ArgumentError Refinement(1e-3, -2)
    @test_throws ArgumentError Refinement(0.0)
    @test_throws ArgumentError Refinement(-1e-3)
    @test_throws ArgumentError Refinement(1.0)
    @test_throws ArgumentError Refinement(1.5)
    @test_throws ArgumentError Refinement(NaN)
    @test_throws ArgumentError Refinement(Inf)

    for R in REAL_ELTYPES
        adapted = adapt(GS.ToReal{R}(), Refinement(1e-4, 5))
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

    prob_fd = NonlinearProblem((F, x, p) -> (F .= x .^ 2 .- 2.0; F))
    @test_throws ArgumentError init(
        prob_fd,
        [1.5, 1.5],
        Newton(; jacobian = AutoFiniteDiff(), refinement = Refinement()))
    @test_throws ArgumentError init(
        prob_fd,
        [1.5, 1.5],
        Newton(;
            linsolve = LUFactorization(Float32), jacobian = AutoFiniteDiff(),
            refinement = true))
    @test_throws ArgumentError init(
        prob_fd, SVector{2, Float64}(1.5, 1.5),
        Newton(; linsolve = LUFactorization(Float32), jacobian = AutoFiniteDiff(),
            refinement = Refinement()))

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

f_reg_r1(x, p) = x .^ 2 .- p
j_reg_r1(x, p) = SMatrix{2, 2, eltype(x), 4}(2x[1], 0, 0, 2x[2])
f_reg_r3!(F, x, p) = (F .= x .^ 2 .- p; F)
function j_reg_r3!(J, x, p)
    fill!(J, 0)
    for i in eachindex(x)
        J[i, i] = 2x[i]
    end
    return J
end

# The factor of a regularised step is that of `J + μ I`, so a `Refinement` refines the direction
# `-(J + μ I) \ F` of that system: it neither undoes the shift nor fails and promotes a
# mixed-precision factorisation. At `x = (1/2, 1)`, `p = (0, 1)` and `μ = 1` the regularised
# direction is `(-1/8, 0)`, and the unregularised one `(-1/4, 0)`.
@testset "Refinement refines the regularised direction ($(lin), $(name))" for lin in (
        LUFactorization(), LUFactorization(Float32)),
    (name, prob, x, p) in (
        ("analytic J, Vector", NonlinearProblem(f_reg_r3!, j_reg_r3!),
        [0.5, 1.0], [0.0, 1.0]),
        ("AD, Vector", NonlinearProblem(f_reg_r3!), [0.5, 1.0], [0.0, 1.0]),
        ("analytic J, SVector", NonlinearProblem(f_reg_r1, j_reg_r1), SVector(0.5, 1.0),
        SVector(0.0, 1.0)),
        ("AD, SVector", NonlinearProblem(f_reg_r1), SVector(0.5, 1.0), SVector(0.0, 1.0)))

    m = Newton(; linsolve = lin, regularization = 1, refinement = Refinement())
    F = x isa SVector ? prob.F(x, p) : prob.F(similar(x), x, p)
    st = GS.start!!(
        GS.init_state(m, prob, copy(x), p), m, prob, copy(x), copy(F), p, norm(F))
    x₁, _, _, info = GS.step!!(st, m, prob, copy(x), copy(F), p)
    @test isapprox(x₁ - x, [-0.125, 0.0]; rtol = 8 * eps())
    @test !info.promoted
    @test info.failures == 0
end

@testset "an analytic J takes a JVP buffer only for a line search or a Refinement along a reused factor" begin
    prob = NonlinearProblem(f_reg_r3!, j_reg_r3!)
    x, p = [0.5, 1.0], [0.0, 1.0]
    # A step that factorises at `x` refines with the Jacobian it holds, and needs no JVP.
    for lin in (LUFactorization(), LUFactorization(Float32)), g in (Static(), DogLeg()),
        ref in (NoRefinement(), Refinement())
        st = GS.init_state(
            Newton(; linsolve = lin, globalization = g, refinement = ref), prob, x, p)
        @test st.jvpprep === nothing
    end
    for lin in (LUFactorization(), LUFactorization(Float32))
        st_ref = GS.init_state(
            Newton(; linsolve = lin, refinement = Refinement(), reuse = 2), prob, x, p)
        @test st_ref.jvpprep isa GS.JacobianMatrixJVP
    end
    # A `J` that is not callable is no Jacobian: `Newton` calls it, and does not differentiate `F`.
    @test_throws MethodError solve(NonlinearProblem(f_reg_r3!, [1.0, 1.0]), x, Newton(), p)
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
    # `refbuf` is optional in the documented signature, so the same call without it is the same
    # answer: the method reads neither `fnorm` nor `Fx` and no buffer.
    d_noref9, failed_noref9 = GS.refine!!(
        d0, NoRefinement(), fac32, nothing, prob_r1, x0, Fx, fnorm, nothing)
    @test d_noref9 === d0
    @test !failed_noref9

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
        d_near, Refinement(1e-12, 1), fac32, nothing,
        prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test d_floor_eta === d_near
    @test !failed_floor_eta

    # Exhausting maxiter when residual decreases but stays above η_eff * fnorm returns
    # (d_improved, true).
    d_one, failed_one = GS.refine!!(
        d0, Refinement(1e-12, 1), fac32, nothing,
        prob_r1, x0, Fx, fnorm, nothing, nothing)
    @test failed_one
    @test GS.rnorm(Fx + J * d_one) < GS.rnorm(Fx + J * d0)

    # Non-decreasing residual !(r_curr < r_prev) stops immediately, returns failed = true,
    # and keeps the best direction before the diverging correction.
    bad_J = SMatrix{2, 2, Float32, 4}(-1.0f0, 0.0f0, 0.0f0, -1.0f0)
    bad_fac = GS.factorize!!(nothing, LUFactorization(), bad_J)
    d_bad, failed_bad = GS.refine!!(
        d0, Refinement(sqrt(eps(Float64)), 3), bad_fac, nothing,
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

    # The method forms `fnorm` from `Fx` when the caller holds none: the same direction and flag.
    d_with, failed_with = GS.refine!!(
        -GS.ldiv!!(nothing, fac32, Fx), Refinement(), fac32, nothing,
        prob_r1, x0, Fx, fnorm, nothing, nothing)
    d_without, failed_without = GS.refine!!(
        -GS.ldiv!!(nothing, fac32, Fx), Refinement(), fac32, nothing,
        prob_r1, x0, Fx, nothing, nothing, nothing)
    @test d_without == d_with
    @test failed_without == failed_with
    @test !failed_without
end

@testset "The mixed step reads a slow TF step and nothing else" begin
    prob = NonlinearProblem(f_mod_r1, j_mod_r1)
    x0 = SVector{2, Float64}(2.0, 3.0)
    Fx0 = f_mod_r1(x0, nothing)
    m = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
    st = GS.init_state(m, prob, x0, nothing)
    st = GS.start!!(st, m, prob, x0, Fx0, nothing, GS.rnorm(Fx0))
    fnorm_old = 1.0
    η_lin = max(sqrt(eps(Float64)), 1e-4)
    # an accepted step that lowers ‖F‖ by less than half, on an inaccurate linear model
    slow = GS.StepInfo{Float64}(SUCCESS, 0.9, 0.1, Int32(0), false, false)
    # the same step on an accurate linear model
    accurate = GS.StepInfo{Float64}(SUCCESS, 0.9, 0.1, Int32(0), false, false)

    st1, info1 = GS.read_mixed_stall(st, slow, fnorm_old, 10 * η_lin)
    @test !st1.promoted
    @test st1.slow_steps == Int32(1)
    @test info1 === slow
    @test info1.code == SUCCESS

    st2, info2 = GS.read_mixed_stall(st1, slow, fnorm_old, 10 * η_lin)
    @test st2.promoted
    @test st2.slow_steps == Int32(0)
    @test info2.promoted
    @test info2.failures == Int32(1)
    @test info2.fnorm == slow.fnorm
    @test info2.stepnorm == slow.stepnorm

    # An accurate linear model returns the step unchanged and resets the count, so that slow,
    # accurate, slow does not promote.
    st3, info3 = GS.read_mixed_stall(st1, accurate, fnorm_old, η_lin / 10)
    @test !st3.promoted
    @test st3.slow_steps == Int32(0)
    @test info3 === accurate
    st4, info4 = GS.read_mixed_stall(st3, slow, fnorm_old, 10 * η_lin)
    @test !st4.promoted
    @test st4.slow_steps == Int32(1)
    @test info4 === slow

    # A step that lowers ‖F‖ by more than half reads nothing, whatever its linear model.
    fast = GS.StepInfo{Float64}(SUCCESS, 0.4, 0.1, Int32(0), false, false)
    st5, info5 = GS.read_mixed_stall(st1, fast, fnorm_old, 10 * η_lin)
    @test !st5.promoted
    @test st5.slow_steps == Int32(0)
    @test info5 === fast
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

    # `start!!` clears the promotion and the slow-step count that an earlier solve left.
    m_stat = Newton(;
        linsolve = LUFactorization(Float32), refinement = Refinement(),
        globalization = Static())
    st_unstarted = GS.init_state(m_stat, prob_mod_r1, x0_r1, nothing)
    @test !st_unstarted.promoted
    @test st_unstarted.slow_steps == Int32(0)
    Fx0 = f_mod_r1(x0_r1, nothing)
    st_dirty = GS.with_mixed(st_unstarted, true, Int32(2))
    st_started = GS.start!!(
        st_dirty, m_stat, prob_mod_r1, x0_r1, Fx0, nothing, GS.rnorm(Fx0))
    @test !st_started.promoted
    @test st_started.slow_steps == Int32(0)

    # The control solve: SUCCESS at f_abstol = 1e-12. cond(J) ≈ 1.8e5 amplifies the residual into
    # the iterate, so the bound on ‖x - x*‖ is cond(J) * atol ≈ 1.8e-7; the control is held to
    # 1e-10, the bound the mixed runs below are held to.
    x_ref64, st_ref64 = solve(
        prob_mod_r1, x0_r1, Newton(; linsolve = LUFactorization()); options = opts)
    @test st_ref64.code == SUCCESS
    @test norm(x_ref64 - x_star) < 1e-10

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

    # cond(J) ≈ 5e7 amplifies the residual tolerance into the iterate, so f_abstol = 1e-12 bounds
    # ‖x - x*‖ by cond(J) * atol ≈ 5e-5; every run below reaches SUCCESS with ‖F‖ ≤ 1e-12 and must
    # land within 1e-5 of the root.
    x_bound = 1e-5

    # With Refinement(): refinement fails on the Float32 factor and promotes within the step.
    for L in (LUFactorization, QRFactorization, SVDFactorization)
        m_with_ref = Newton(; linsolve = L(Float32), refinement = Refinement())
        x_r1, st_r1 = solve(prob_ill_r1, x0_r1, m_with_ref; options = opts)
        @test st_r1.code == SUCCESS
        @test st_r1.promoted
        @test st_r1.step_failures == 1
        @test st_r1.fnorm < 1e-12
        @test norm(x_r1 - x_star) < x_bound
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
            @test norm(Array(x_r3) - collect(x_star)) < x_bound
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
    @test norm(x_nr_r1 - x_star) < x_bound

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
        @test norm(Array(x_nr_r3) - collect(x_star)) < x_bound
    end

    # A promotion counts one step-rule failure at each site that can make it: the line search, the
    # `DogLeg` trial loop, and a reused factor (`reuse = 3`). The one step and the whole solve each
    # count it once, so a step that is counted twice fails a test here.
    for m_site in (
        Newton(; linsolve = LUFactorization(Float32), refinement = Refinement()),
        Newton(;
        linsolve = LUFactorization(Float32), refinement = Refinement(),
        globalization = DogLeg()),
        Newton(; linsolve = LUFactorization(Float32), refinement = Refinement(), reuse = 3))
        Fx_site = f_ill_r1(x0_r1, nothing)
        st_site = GS.start!!(
            GS.init_state(m_site, prob_ill_r1, x0_r1, nothing), m_site, prob_ill_r1, x0_r1,
            Fx_site, nothing, GS.rnorm(Fx_site))
        _, _, st_site, info_site = GS.step!!(
            st_site, m_site, prob_ill_r1, x0_r1, Fx_site, nothing)
        @test st_site.promoted
        @test info_site.failures == 1
        _, s_site = solve(prob_ill_r1, x0_r1, m_site; options = opts)
        @test s_site.promoted
        @test s_site.step_failures == 1
    end

    # The factor in `T` is the slot that `init_state` reserves, so a promotion allocates nothing:
    # the measured solve runs after warm-up solves have compiled and warmed the state, on a problem
    # whose own residual and Jacobian write their buffers in place.
    function f_prom!(F, x, _)
        T = eltype(x)
        a11, a21, a12, a22 = T.(Tuple(A_ILL))
        b1, b2 = T.(Tuple(B_ILL))
        u1 = a11 * x[1] + a12 * x[2]
        u2 = a21 * x[1] + a22 * x[2]
        @inbounds F[1] = u1 + T(0.1) * u1^2 - b1
        @inbounds F[2] = u2 - T(0.2) * u2^2 - b2
        return F
    end
    function j_prom!(J, x, _)
        T = eltype(x)
        a11, a21, a12, a22 = T.(Tuple(A_ILL))
        u1 = a11 * x[1] + a12 * x[2]
        u2 = a21 * x[1] + a22 * x[2]
        d1 = one(T) + T(0.2) * u1
        d2 = one(T) - T(0.4) * u2
        @inbounds J[1, 1] = d1 * a11
        @inbounds J[1, 2] = d1 * a12
        @inbounds J[2, 1] = d2 * a21
        @inbounds J[2, 2] = d2 * a22
        return J
    end
    prob_prom = NonlinearProblem(f_prom!, j_prom!)
    warm = init(prob_prom, collect(x0_r1), m_no_ref; options = opts)
    w_prom = collect(x0_r1)
    solve!(warm, w_prom)
    @test warm.state.promoted
    # The measurement sits behind the function barrier, and the control asserts that the barrier
    # sees an allocation on this build: a measured zero is then the solver's and not the probe's.
    alloc_prom, steps_prom = _measure_solve_alloc!(warm, w_prom, collect(x0_r1))
    @test steps_prom > 0
    @test alloc_prom == 0
    @test allocations(collect, x0_r1) > 0

    # A promotion on a refinement failure is the mixed step's other promotion branch, and it holds
    # the same `T` factor from `init_state`: the solve counts exactly one failure and measures the
    # same zero.
    m_ref_prom = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
    warm_ref_prom = init(prob_prom, collect(x0_r1), m_ref_prom; options = opts)
    w_ref_prom = collect(x0_r1)
    _, st_ref_prom = solve!(warm_ref_prom, w_ref_prom)
    @test st_ref_prom.promoted
    @test st_ref_prom.step_failures == 1
    alloc_ref_prom, steps_ref_prom = _measure_solve_alloc!(warm_ref_prom, w_ref_prom, collect(x0_r1))
    @test steps_ref_prom > 0
    @test alloc_ref_prom == 0
end

@testset "Check 1: a mixed step keeps going where its TF == T control does" begin
    opts = Options(Float64)
    # A well-conditioned `F(x) = A x + c .* x.^3 - b`, drawn from one generator: the pure `Float64`
    # Newton iteration overshoots `‖F‖` on some of these, and a mixed step that read an overshoot
    # as the floor of `F` would end `STALLED` there while the control converges.
    rng = Xoshiro(7)
    for _ in 1:3
        A = randn(rng, 2, 2) + 3I
        c = randn(rng, 2)
        b = 5randn(rng, 2)
        x0 = SVector{2, Float64}(randn(rng, 2))
        prob = NonlinearProblem(
            (x, p) -> A * x .+ c .* x .^ 3 .- b,
            (x, p) -> A + Diagonal(3 .* c .* x .^ 2))
        _, st64 = solve(prob, x0, Newton(; linsolve = LUFactorization()); options = opts)
        _, st32 = solve(prob, x0, Newton(; linsolve = LUFactorization(Float32)); options = opts)
        @test st64.code == SUCCESS
        @test st32.code == st64.code
        @test st32.iterations == st64.iterations
        @test !st32.promoted
    end

    # The same generator over a wider set, judged by the two counts of Check 1.
    codes = ReturnCode[]
    promoted = Bool[]
    for _ in 1:30
        A = randn(rng, 2, 2) + 3I
        c = randn(rng, 2)
        b = 5randn(rng, 2)
        x0 = SVector{2, Float64}(randn(rng, 2))
        prob = NonlinearProblem(
            (x, p) -> A * x .+ c .* x .^ 3 .- b,
            (x, p) -> A + Diagonal(3 .* c .* x .^ 2))
        _, st32 = solve(prob, x0, Newton(; linsolve = LUFactorization(Float32)); options = opts)
        push!(codes, st32.code)
        push!(promoted, st32.promoted)
    end
    @test count(==(STALLED), codes) == 0
    @test count(identity, promoted) == 0
end

@testset "The floor of F under a mixed step and its TF == T control" begin
    prob_fl_r1 = NonlinearProblem(f_floor_r1, j_floor_r1)
    prob_fl_r3 = NonlinearProblem(f_floor_r3!, j_floor_r3!)
    x0_r1 = SVector{2, Float64}(1.5, 1.8)
    # `F` rounds its internal state through `Float32`, so neither precision reaches a tolerance
    # below the floor of `F`: the control and the mixed run both end `MAXITERS` at the cap, and the
    # mixed run promotes nothing and counts no failure.
    opts = Options(
        Float64; f_abstol = 1e-14, f_reltol = 1e-14, x_abstol = 0.0, x_reltol = 0.0,
        max_iterations = 10, max_stalls = 2)
    m_control = Newton(; linsolve = LUFactorization())
    m_floor = Newton(; linsolve = LUFactorization(Float32), refinement = NoRefinement())
    floor_bound = 4 * eps(Float32) * norm(SVector(2.0, 3.0))

    _, st_ctl_r1 = solve(prob_fl_r1, x0_r1, m_control; options = opts)
    @test st_ctl_r1.code == MAXITERS
    @test st_ctl_r1.iterations == opts.max_iterations

    _, st_fl_r1 = solve(prob_fl_r1, x0_r1, m_floor; options = opts)
    @test st_fl_r1.code == st_ctl_r1.code
    @test st_fl_r1.iterations == st_ctl_r1.iterations
    @test !st_fl_r1.promoted
    @test st_fl_r1.step_failures == 0
    @test st_fl_r1.fnorm <= floor_bound

    for B in ARRAY_BACKENDS
        x0_r3 = B(collect(x0_r1))
        _, st_ctl_r3 = solve(prob_fl_r3, x0_r3, m_control; options = opts)
        @test st_ctl_r3.code == MAXITERS
        @test st_ctl_r3.iterations == opts.max_iterations
        _, st_fl_r3 = solve(prob_fl_r3, x0_r3, m_floor; options = opts)
        @test st_fl_r3.code == st_ctl_r3.code
        @test st_fl_r3.iterations == st_ctl_r3.iterations
        @test !st_fl_r3.promoted
        @test st_fl_r3.step_failures == 0
        @test st_fl_r3.fnorm <= floor_bound
    end

    # A step whose linear model is accurate is unchanged, whatever its residual does: an O(1)
    # overshoot of the Newton iteration is not a reading of the floor of `F`.
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
    # fails Refinement(1e-2, 1), while a fresh Float32 factor at step 2 succeeds without
    # promoting to Float64.
    function f_rot_r1(x, _)
        return SVector(
            x[1]^3 - 3.0 * x[1] * x[2]^2 - 1.0,
            3.0 * x[1]^2 * x[2] - x[2]^3 - 2.0)
    end
    prob_rot_r1 = NonlinearProblem(f_rot_r1)
    x0_rot = SVector{2, Float64}(2.0, 0.5)

    m_qn = QuasiNewton(
        5; linsolve = LUFactorization(Float32), refinement = Refinement(1e-2, 1))
    x_qn, st_qn = solve(prob_rot_r1, x0_rot, m_qn; options = opts)
    @test st_qn.code == SUCCESS
    @test !st_qn.promoted
    @test st_qn.fnorm < 1e-12
    @test norm(x_qn - x0_rot) > 0.1

    # Check 5: the refactorisation happens at the step the reused factor fails, in `TF` and not in
    # `T`. A host counter over the Jacobian of this problem reads one evaluation per
    # factorisation and one per refinement pass along a reused factor: a fresh factor refines with
    # the Jacobian it was factorised from. A step reads 1 when its factor is fresh, and 3 when a
    # reused factor fails its one refinement pass and is refactorised.
    j_calls = Ref(0)
    function j_rot_r1_counted(x, _)
        j_calls[] += 1
        return SMatrix{2, 2, Float64, 4}(
            3.0 * x[1]^2 - 3.0 * x[2]^2, 6.0 * x[1] * x[2],
            -6.0 * x[1] * x[2], 3.0 * x[1]^2 - 3.0 * x[2]^2)
    end
    prob_counted = NonlinearProblem(f_rot_r1, j_rot_r1_counted)
    st_count = GS.init_state(m_qn, prob_counted, x0_rot, nothing)
    Fx_count = f_rot_r1(x0_rot, nothing)
    x_count = x0_rot
    st_count = GS.start!!(
        st_count, m_qn, prob_counted, x_count, Fx_count, nothing, GS.rnorm(Fx_count))
    counts = Int[]
    for _ in 1:2
        before = j_calls[]
        x_count, Fx_count, st_count, info_count = GS.step!!(
            st_count, m_qn, prob_counted, x_count, Fx_count, nothing)
        push!(counts, j_calls[] - before)
        @test info_count.code == SUCCESS
    end
    @test counts == [1, 3]
    @test st_count.rule === GS.FactorAge(Int32(1), false)
    @test !st_count.promoted
    # each early refactorisation is one step-rule failure, and the solve counts them
    @test st_qn.step_failures >= 1

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

# `F(x) = x.^3 .- p` with an analytic Jacobian that counts its own evaluations.
const J_CALLS = Ref(0)
f_cube_r1(x, p) = x .^ 3 .- p
function j_cube_r1(x, p)
    J_CALLS[] += 1
    return SMatrix{2, 2, eltype(x), 4}(3x[1]^2, 0, 0, 3x[2]^2)
end
f_cube_r3!(F, x, p) = (F .= x .^ 3 .- p; F)
function j_cube_r3!(J, x, p)
    J_CALLS[] += 1
    fill!(J, 0)
    for i in eachindex(x)
        J[i, i] = 3x[i]^2
    end
    return J
end

const CUBE_CASES = (
    ("SVector", NonlinearProblem(f_cube_r1, j_cube_r1),
        SVector(10.0, 8.0), SVector(2.0, 3.0)),
    ("Vector", NonlinearProblem(f_cube_r3!, j_cube_r3!), [10.0, 8.0], [2.0, 3.0]))

# The evaluations of `J` in each of `n` steps from `x`, and the state and the `StepInfo` of each.
function counted_steps(m, prob, x, p, n)
    F = x isa SVector ? prob.F(x, p) : prob.F(similar(x), x, p)
    st = GS.start!!(GS.init_state(m, prob, x, p), m, prob, x, F, p, norm(F))
    calls, steps = Int[], Any[]
    for _ in 1:n
        before = J_CALLS[]
        x, F, st, info = GS.step!!(st, m, prob, x, F, p)
        push!(calls, J_CALLS[] - before)
        push!(steps, (st, info))
    end
    return calls, steps
end

# A step that factorises at `x` holds the regularised Jacobian it factorised, and a `Refinement`
# multiplies by it: the step evaluates `J` once, for its factorisation. The buffer it multiplies by
# is no factor's matrix, which the factorisation overwrites.
@testset "a fresh step evaluates J once under Refinement ($(name), $(lin))" for (name, prob, x, p) in CUBE_CASES,
    lin in (LUFactorization(), LUFactorization(Float32), QRFactorization(Float32))

    for kw in ((;), (; regularization = 1), (; globalization = DogLeg()))
        m = Newton(; linsolve = lin, refinement = Refinement(), kw...)
        calls, steps = counted_steps(m, prob, copy(x), p, 4)
        @test calls == [1, 1, 1, 1]
        @test all(s -> s[2].code == SUCCESS && s[2].failures == 0, steps)
        st = first(last(steps))
        if x isa Vector
            @test st.J !== st.fac.factors
            st.fac_T === nothing || @test st.J !== st.fac_T.factors
        end
    end
end

@testset "a promoted step evaluates J once under Refinement ($(name))" for (name, prob, x) in (
    ("SVector", NonlinearProblem(
        f_ill_r1, (x, p) -> (J_CALLS[] += 1; j_ill_r1(x, p))),
    SVector(2.0, 3.0)),
    ("Vector",
    NonlinearProblem(
        f_ill_r3!, (J, x, p) -> (J_CALLS[] += 1; j_ill_r3!(J, x, p))),
    [2.0, 3.0]))
    m = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())
    calls, steps = counted_steps(m, prob, x, nothing, 3)
    @test all(s -> s[1].promoted, steps)
    @test calls == [1, 1, 1]
end

# A `Refinement` that fails on a reused factor refreshes it within the step, in the working
# precision as in a mixed one: the step factorises at `x` again, counts one step-rule failure, and
# restarts the age of the factor. With `η = 1e-12` and one pass, each refinement along the reused
# factor fails here, and the refinement on the fresh factor does not.
@testset "a failed Refinement on a reused factor refreshes it ($(name), $(lin))" for (name, prob, x, p) in CUBE_CASES,
    lin in (LUFactorization(), LUFactorization(Float32))

    m = Newton(; linsolve = lin, reuse = 3, refinement = Refinement(1e-12, 1))
    calls, steps = counted_steps(m, prob, copy(x), p, 5)
    # The first step factorises. Each later step takes two products along the reused factor and
    # one evaluation for the factorisation that refreshes it.
    @test calls == [1, 3, 3, 3, 3]
    @test all(s -> s[1].rule === GS.FactorAge(Int32(1), false), steps)
    @test [s[2].failures for s in steps] == [0, 1, 1, 1, 1]
    @test !any(s -> s[2].promoted, steps)
    _, status = solve(prob, copy(x), m, p)
    @test status.code == SUCCESS
    @test status.step_failures >= 1
end

# The mixed-precision step takes a `StationarityProblem` as the working-precision step does: the
# stationary point `x + x³ = 1` of `S` in each component, `x = 0.6823278038280193…`.
@testset "a StationarityProblem takes the mixed-precision step ($(typeof(x₀).name.name))" for x₀ in (
    SVector(0.3, 0.6), [0.3, 0.6])
    prob = StationarityProblem((x, p) -> sum(abs2, x) / 2 + sum(x .^ 4) / 4 - sum(x))
    x_ref, st_ref = solve(prob, copy(x₀), Newton())
    @test st_ref.code == SUCCESS
    for m in (Newton(; linsolve = LUFactorization(Float32)),
        Newton(; linsolve = LUFactorization(Float32), refinement = Refinement()),
        Newton(; linsolve = LDLTFactorization(Float32), refinement = Refinement(), reuse = 2))
        x, st = solve(prob, copy(x₀), m)
        @test st.code == SUCCESS
        @test !st.promoted
        @test isapprox(Array(x), Array(x_ref); rtol = 1e-10)
    end
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
        alloc, steps = _measure_solve_alloc!(solver, x_work, x0_vec)
        # `init_state` preallocates `state.fac_T` and the Jacobian buffer `state.J` that a
        # `Refinement` multiplies by, so the step adds no `n × n` buffer and holds no value that
        # could box either factor: a solve allocates nothing.
        @test steps > 0
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
    @test allocations(collect, x0_r1) > 0
    # `julia-actions/julia-runtest` forces `--check-bounds=yes`, which lets `StaticArrays`' local
    # `MMatrix` escape to the heap, so this assertion holds only where bounds checking is off.
    if Base.JLOptions().check_bounds != 1
        @test alloc_r1 == 0
    end

    # The reuse path of a mixed R1 solve: the step's mixed half returns a direction of one type on
    # every branch, so the solve infers, JET finds no dispatch, and nothing allocates.
    m_qn32 = Newton(; linsolve = LUFactorization(Float32), reuse = 3, refinement = Refinement())
    f_qn(x, _) = SVector(x[1]^2 + 0.2x[2] - 2.0, 0.1x[1] + x[2]^2 - 3.0)
    prob_qn = NonlinearProblem(f_qn)
    x0_qn = SVector(1.5, 1.5)
    @test (@inferred solve(prob_qn, x0_qn, m_qn32)) isa Tuple
    st_qn32 = GS.init_state(m_qn32, prob_qn, x0_qn, nothing)
    JET.@test_opt target_modules=(GS,) GS.solve!!(
        st_qn32, prob_qn, x0_qn, m_qn32, nothing, opts)
    alloc_qn32 = _measure_solve_r1_alloc(st_qn32, prob_qn, x0_qn, m_qn32, opts)
    if Base.JLOptions().check_bounds != 1
        @test alloc_qn32 == 0
    end

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
    # The same zero at both sizes: the step's allocation does not scale with n.
    b1, n_steps1 = _measure_solve_alloc!(solver_n1, w_n1, x0_n1)
    b2, n_steps2 = _measure_solve_alloc!(solver_n2, w_n2, x0_n2)
    @test n_steps1 > 0 && n_steps2 > 0
    @test b1 == 0
    @test b2 == 0

    JET.@test_opt target_modules=(GeometricSolvers,) solve(
        prob_mod_r1, x0_r1, m_ka; options = opts)
    copyto!(w_n1, x0_n1)
    JET.@test_opt target_modules=(GeometricSolvers,) solve!(solver_n1, w_n1)
end

# The `norm2` seam, counted: the number of reductions one step makes when its factorisation runs
# in the working precision, which is the step rule's own count. A step in `T` adds no reduction of
# its own, so the count is a property of the step rule alone.
const NORM2_CALLS = Ref(0)

function _counted_norm2(a)
    NORM2_CALLS[] += 1
    return isempty(a) ? 0.0 : Float64(sum(abs2, a))
end

GS.norm2(a::Vector{Float64}) = _counted_norm2(a)
GS.norm2(a::JLArray{Float64, 1}) = _counted_norm2(a)

@testset "A step in the working precision adds no reduction" begin
    function f_red!(F, x, _)
        xh = Array(x)
        copyto!(F, [xh[1]^2 + 0.2xh[2] - 2.0, 0.1xh[1] + xh[2]^2 - 3.0])
        return F
    end
    function j_red!(J, x, _)
        xh = Array(x)
        copyto!(J, [2xh[1] 0.2; 0.1 2xh[2]])
        return J
    end
    prob = NonlinearProblem(f_red!, j_red!)
    cases = (
        ("Newton Static", Newton(), [3, 3, 3]),
        ("Newton Backtracking", Newton(; globalization = Backtracking()), [3, 3, 3]),
        ("Newton DogLeg", Newton(; globalization = DogLeg()), [9, 9, 9]),
        ("Newton reuse=3", Newton(; reuse = 3), [3, 2, 2])
    )
    for (name, m, expected) in cases, B in ARRAY_BACKENDS

        x = B([1.5, 1.5])
        st = GS.init_state(m, prob, x, nothing)
        Fx = prob.F(GS.residual_buffer(st), x, nothing)
        st = GS.start!!(st, m, prob, x, Fx, nothing, GS.rnorm(Fx))
        counts = Int[]
        for _ in 1:3
            NORM2_CALLS[] = 0
            x, Fx, st, _ = GS.step!!(st, m, prob, x, Fx, nothing)
            push!(counts, NORM2_CALLS[])
        end
        @test counts == expected
    end
end

# The counting methods are the test file's own: drop them, so that every later file on this
# process sees the package's `norm2`.
for V in (Vector{Float64}, JLArray{Float64, 1})
    Base.delete_method(which(GS.norm2, (V,)))
end
