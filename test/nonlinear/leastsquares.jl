using ADTypes: AutoFiniteDiff, AutoForwardDiff
using Adapt: Adapt
using GeometricSolvers
using GeometricSolvers: DenseQR, DenseSVD, DeviceQR, DeviceSVD, MeasuredSlope, StepInfo,
                        dense_factor, factor_diagnostics, factorize!!, isconverged, ldiv!!,
                        lm_stacked_qr_step, lm_svd_step, newton_stepkind,
                        record, returncode
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: Diagonal, I, mul!, norm, pinv
using Random: Xoshiro, randn
using StaticArrays: @SMatrix, SMatrix, SVector
using Test

include("../helpers/matrix.jl")

allowscalar(false)

# ─── Benchmark fixtures (Moré–Garbow–Hillstrom 1981) ─────────────────────────

# MGH #11: Linear function — full rank (m = 10, n = 5), min ‖F(x*)‖₂² = m - n = 5 at x* = -ones(5)
function mgh11_static(x::SVector{5, T}, _) where {T}
    s = T(2) / T(10) * sum(x)
    return SVector{10, T}(
        x[1] - s - one(T), x[2] - s - one(T), x[3] - s - one(T),
        x[4] - s - one(T), x[5] - s - one(T),
        -s - one(T), -s - one(T), -s - one(T), -s - one(T), -s - one(T))
end

function mgh11_inplace!(r, x, _)
    T = eltype(x)
    m = length(r)
    n = length(x)
    s = (T(2) / T(m)) * sum(x)
    r .= -s .- one(T)
    @views r[1:n] .+= x
    return r
end

# MGH #12: Linear function — rank 1 (m = 10, n = 5), min ‖F(x*)‖₂² = m(m-1)/(2(2m+1)) = 15/7
function mgh12_static(x::SVector{5, T}, _) where {T}
    u = x[1] + T(2) * x[2] + T(3) * x[3] + T(4) * x[4] + T(5) * x[5]
    return SVector{10, T}(ntuple(i -> T(i) * u - one(T), Val(10)))
end

function mgh12_inplace!(r, x, _)
    T = eltype(x)
    n = length(x)
    m = length(r)
    w = similar(x)
    copyto!(w, collect(T(1):T(n)))
    u = sum(w .* x)
    idx = similar(r)
    copyto!(idx, collect(T(1):T(m)))
    r .= idx .* u .- one(T)
    return r
end

# MGH #8: Bard function (m = 15, n = 3), min ‖F(x*)‖₂² ≈ 8.214877306578994e-3
const BARD_Y = (0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
    0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39)

function mgh8_static(x::SVector{3, T}, _) where {T}
    return SVector{15, T}(ntuple(Val(15)) do i
        u = T(i)
        v = T(16 - i)
        w = T(min(i, 16 - i))
        T(BARD_Y[i]) - (x[1] + u / (v * x[2] + w * x[3]))
    end)
end

function mgh8_inplace!(r, x, _)
    T = eltype(x)
    x_h = Array(x)
    r_h = Vector{T}(undef, 15)
    @inbounds for i in 1:15
        u = T(i)
        v = T(16 - i)
        w = T(min(i, 16 - i))
        r_h[i] = T(BARD_Y[i]) - (x_h[1] + u / (v * x_h[2] + w * x_h[3]))
    end
    copyto!(r, r_h)
    return r
end

# MGH #6: Jennrich and Sampson function (m = 10, n = 2), min ‖F(x*)‖₂² ≈ 124.3621823556148
function mgh6_static(x::SVector{2, T}, _) where {T}
    return SVector{10, T}(ntuple(Val(10)) do i
        ti = T(i)
        T(2) + T(2) * ti - (exp(ti * x[1]) + exp(ti * x[2]))
    end)
end

function mgh6_inplace!(r, x, _)
    T = eltype(x)
    x_h = Array(x)
    r_h = Vector{T}(undef, 10)
    @inbounds for i in 1:10
        ti = T(i)
        r_h[i] = T(2) + T(2) * ti - (exp(ti * x_h[1]) + exp(ti * x_h[2]))
    end
    copyto!(r, r_h)
    return r
end

# A rectangular R1 residual whose sizes are type parameters, so that `solve` infers over it: a size
# read from a loop variable is a value, and `SVector{M}` built from it is then a run-time type.
function r1_least_squares(::Val{M}, ::Val{N}, ::Type{T}) where {M, N, T}
    f = (x, p) -> SVector{M}(ntuple(i -> i <= N ? x[i] - one(T) : T(1) / T(10) * x[1], Val(M)))
    return LeastSquaresProblem(f)
end

# ─── Constructors, Options.g_tol, SolverStatus.stationary, stopping_test ─────

@testset "LeastSquaresProblem, GaussNewton, Options.g_tol, and stopping_test, $T" for T in REAL_ELTYPES
    f_oop = (x, p) -> SVector{3}(x[1], x[2], x[1] + x[2])
    j_oop = (x, p) -> @SMatrix [one(T) zero(T); zero(T) one(T); one(T) one(T)]
    f_ip = (r, x, p) -> (r .= zero(T); r)
    j_ip = (J, x, p) -> (J .= zero(T); J)

    p1 = LeastSquaresProblem(f_oop)
    @test p1 isa LeastSquaresProblem{typeof(f_oop), Nothing, Nothing}
    p2 = LeastSquaresProblem(f_oop, j_oop)
    @test p2 isa LeastSquaresProblem{typeof(f_oop), typeof(j_oop), Nothing}
    p3 = LeastSquaresProblem(f_ip, 5)
    @test p3 isa LeastSquaresProblem{typeof(f_ip), Nothing, Int}
    @test p3.output_length == 5
    p4 = LeastSquaresProblem(f_ip, j_ip, 5)
    @test p4 isa LeastSquaresProblem{typeof(f_ip), typeof(j_ip), Int}
    @test p4.output_length == 5
    p5 = LeastSquaresProblem(f_ip, j_ip; output_length = 5)
    @test p5.output_length == 5 && p5.J === j_ip
    @test LeastSquaresProblem(f_oop).output_length === nothing

    @test_throws ArgumentError LeastSquaresProblem(f_ip, 0)
    @test_throws ArgumentError LeastSquaresProblem(f_ip, -3)
    @test_throws ArgumentError LeastSquaresProblem(f_ip, j_ip, 0)
    @test_throws ArgumentError init(p1, ones(T, 2), GaussNewton())

    gn = GaussNewton()
    @test gn isa Newton
    @test gn.linsolve isa QRFactorization
    @test gn.globalization isa Static
    @test gn.jacobian === nothing
    @test gn.refinement isa NoRefinement
    gn_svd = GaussNewton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
    @test gn_svd.linsolve isa SVDFactorization
    @test gn_svd.globalization isa LevenbergMarquardt

    opt = Options(T)
    @test opt.g_tol === sqrt(eps(T))
    opt2 = Options(T; g_tol = T(1e-5))
    @test opt2.g_tol === T(1e-5)
    @test Options{T}(opt.f_abstol, opt.f_reltol, opt.x_abstol, opt.x_reltol,
        opt.min_iterations, opt.max_iterations, opt.max_stalls).g_tol === sqrt(eps(T))
    @test_throws ArgumentError Options(T; g_tol = T(-1e-5))
    @test_throws ArgumentError Options(T; g_tol = T(NaN))

    @test sizeof(SolverStatus{T}) == (T === Float32 ? 36 : 56)
    st6 = SolverStatus{T}(SUCCESS, Int32(2), zero(T), zero(T), Int32(0), false)
    @test st6.stationary === false
    @test stopping_test(st6) === :residual
    st9 = SolverStatus{T}(STALLED, Int32(3), T(1e-3), zero(T), Int32(0),
        false, Int32(2), zero(T), T(1e-2))
    @test st9.stationary === false
    @test stopping_test(st9) === :none
    st10 = SolverStatus{T}(SUCCESS, Int32(4), T(1), T(1e-6), Int32(0),
        false, true, Int32(2), T(0.5), T(1e-9))
    @test st10.stationary === true
    @test stopping_test(st10) === :stationarity
    @test record(st10, StepInfo{T}(SUCCESS, T(1), T(1e-6), Int32(0), false, false)).stationary ===
          true
    st_fail = SolverStatus{T}(MAXITERS, Int32(10), T(1), T(1), Int32(0),
        false, false, Int32(2), T(0.5), T(1e-1))
    @test stopping_test(st_fail) === :none

    # LUFactorization without LevenbergMarquardt is rejected on LeastSquaresProblem
    x0_s = SVector{2, T}(1, 2)
    @test_throws ArgumentError solve(p1, x0_s, Newton(; linsolve = LUFactorization()))
    @test_throws ArgumentError init(p3, ones(T, 2), Newton(; linsolve = LUFactorization()))

    # Regularization > 0 on rectangular m > n is rejected
    @test_throws ArgumentError solve(p1, x0_s, Newton(; linsolve = QRFactorization(), regularization = 1e-6))
    @test_throws ArgumentError init(
        p3, ones(T, 2), Newton(; linsolve = QRFactorization(), regularization = 1e-6))

    # A rectangular static Jacobian takes `μ` on its leading diagonal and nowhere else
    J_rect = SMatrix{3, 2, T}(one(T), T(2), T(3), T(4), T(5), T(6))
    @test GeometricSolvers.regularize!!(J_rect, T(10)) ==
          SMatrix{3, 2, T}(T(11), T(2), T(3), T(4), T(15), T(6))
    @test GeometricSolvers.regularize!!(J_rect, zero(T)) === J_rect

    # Underdetermined m < n throws DimensionMismatch
    p_under_s = LeastSquaresProblem((x, p) -> SVector{2}(x[1], x[2]))
    @test_throws DimensionMismatch solve(p_under_s, SVector{3, T}(1, 2, 3), GaussNewton())
    p_under_d = LeastSquaresProblem(f_ip, 2)
    @test_throws DimensionMismatch init(p_under_d, ones(T, 3), GaussNewton())
end

# ─── D1: a `Number` iterate, solved on the one-entry vector path ─────────────

# `r(x) = (x - 1, x - 2, 2x)` has its least-squares minimum at `x* = 1/2`, and a line search over a
# measured slope stops where `g_tol = √eps` allows rather than at that minimum, `6e-9` short of it
# in `Float64`. The assertion is therefore not an approximation of `x*` but the same problem written
# by hand on `SVector{1}`: code, iteration count and iterate agree bit for bit, whichever Jacobian
# source and method run, and the method's own operator takes the `Number` the caller's problem is
# written for.

@testset "Number iterate: the same solve as the hand-written SVector{1} problem, $T" for T in REAL_ELTYPES
    f_num(x, p) = SVector(x - one(T), x - T(2), T(2) * x)
    j_vec(x, p) = SVector(one(T), one(T), T(2))
    j_mat(x, p) = SMatrix{3, 1, T, 3}(one(T), one(T), T(2))
    f_one(x, p) = SVector(x[1] - one(T), x[1] - T(2), T(2) * x[1])
    j_one(x, p) = SMatrix{3, 1, T, 3}(one(T), one(T), T(2))
    x0_num = zero(T)
    x0_one = SVector{1, T}(zero(T))

    # The Jacobian sources: the problem's `J` or the method's operator, each written as the column
    # `(1, 1, 2)` or as the `3 × 1` matrix, and the two automatic-differentiation sources. The
    # hand-written problem of a source written as a column carries the matrix, which is that
    # source's lifted form.
    sources = [
        ("AD", LeastSquaresProblem(f_num), LeastSquaresProblem(f_one), (;), (;)),
        ("problem J as a vector", LeastSquaresProblem(f_num, j_vec),
            LeastSquaresProblem(f_one, j_one), (;), (;)),
        ("problem J as a matrix", LeastSquaresProblem(f_num, j_mat),
            LeastSquaresProblem(f_one, j_one), (;), (;)),
        ("method jacobian as a vector", LeastSquaresProblem(f_num),
            LeastSquaresProblem(f_one), (; jacobian = j_vec), (; jacobian = j_one)),
        ("method jacobian as a matrix", LeastSquaresProblem(f_num),
            LeastSquaresProblem(f_one), (; jacobian = j_mat), (; jacobian = j_one)),
        ("AutoForwardDiff", LeastSquaresProblem(f_num), LeastSquaresProblem(f_one),
            (; jacobian = AutoForwardDiff()), (; jacobian = AutoForwardDiff()))
    ]

    # Every Gauss–Newton globalization with both factorisations and, where the method takes it, a
    # factor reuse of two steps, then Levenberg–Marquardt, which takes no reuse.
    methods = [("GaussNewton Static $lname",
                   (; globalization = Static(), linsolve = ls), true)
               for (lname, ls) in (("QR", QRFactorization()), ("SVD", SVDFactorization()))]
    append!(methods,
        [("GaussNewton $gname $lname", (; globalization = g, linsolve = ls), reuse_ok)
         for (gname, g, reuse_ok) in (
            ("Backtracking", Backtracking(), true), ("Bisection", Bisection(), true),
            ("StrongWolfe", StrongWolfe(), true), ("DogLeg", DogLeg(), false)),
        (lname, ls) in (("QR", QRFactorization()), ("SVD", SVDFactorization()))])
    append!(methods,
        [("LevenbergMarquardt $lname",
             (; globalization = LevenbergMarquardt(), linsolve = ls), false)
         for (lname, ls) in (("QR", QRFactorization()), ("SVD", SVDFactorization()))])

    for (label, prob_num, prob_one, kw_num, kw_one) in sources,
        (mname, mkw, reuse_ok) in methods,
        reuse in (reuse_ok ? (nothing, 2) : (nothing,))
        rkw = reuse === nothing ? mkw : (; mkw..., reuse)
        @testset "$label, $mname, reuse = $(reuse === nothing ? "none" : reuse)" begin
            x_num, st_num = solve(prob_num, x0_num, GaussNewton(; rkw..., kw_num...))
            x_one, st_one = solve(prob_one, x0_one, GaussNewton(; rkw..., kw_one...))
            @test x_num isa T
            @test isconverged(st_num)
            @test st_num.code === st_one.code
            @test st_num.iterations == st_one.iterations
            @test x_num === x_one[1]
        end
    end

    # The inferred, allocation-free calls of the `Number` path, through a barrier of concrete types
    # whose warm-up call runs before the measurement.
    number_solve(prob, x0, m) = @inferred solve(prob, x0, m)
    prob_num = LeastSquaresProblem(f_num)
    for m in (GaussNewton(), GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()))
        @test number_solve(prob_num, x0_num, m) isa Tuple{T, SolverStatus{T}}
        # skipped where a run forces --check-bounds=yes
        if Base.JLOptions().check_bounds != 1
            @test allocations(solve, prob_num, x0_num, m) == 0
        end
    end
    # the control: `allocations` sees an allocation where there is one
    @test allocations(collect, SVector{3, T}(1, 2, 3)) > 0
end

# ─── Clause 1c: Dense and device rectangular QR and SVD ──────────────────────

# `factorize!!` of a mutable matrix writes its factors into that matrix, so a measured call needs
# the matrix restored first: the barrier's warm-up call does that copy, and the measured one does
# the same work again.
function factorize_fresh!!(fac, method, A, A_src)
    copyto!(A, A_src)
    return factorize!!(fac, method, A)
end

# A measured `solve!(solver)` runs from the initial iterate every time, so the barrier restores it
# into the solver before the call: the warm-up call does the same work as the measured one.
function solve_from_start!(solver, x0)
    copyto!(solver.x, x0)
    return solve!(solver)
end

@testset "Dense and device rectangular QR and SVD, $T" for T in REAL_ELTYPES
    m, n = 6, 3
    rng = Xoshiro(700 + sizeof(T))
    A_h = randn(rng, T, m, n) .+ [i == j ? T(3) : zero(T) for i in 1:m, j in 1:n]
    b_h = randn(rng, T, m)
    x_ref = (A_h' * A_h) \ (A_h' * b_h)
    prob = LeastSquaresProblem((r, x, p) -> r, m)

    # DenseQR and DenseSVD on Array
    for method in (QRFactorization(), SVDFactorization())
        d = zeros(T, n)
        fac = dense_factor(method, prob, d, nothing)
        @test size(fac.factors) == (m, n)
        @test length(fac.rhs) == m
        fac = factorize!!(fac, method, copy(A_h))
        @test returncode(fac) === SUCCESS
        ldiv!!(d, fac, b_h)
        @test isapprox(d, x_ref; rtol = 16 * sqrt(eps(T)))
        rk, nn = factor_diagnostics(fac, b_h)
        @test rk == Int32(n)
        @test nn === zero(T)

        A_work = copy(A_h)
        fac = factorize!!(fac, method, A_work)
        ldiv!!(d, fac, b_h)
        @test allocations(factorize_fresh!!, fac, method, A_work, A_h) == 0
        @test allocations(ldiv!!, d, fac, b_h) == 0
        # the control: `allocations` sees an allocation where there is one
        @test allocations(copy, b_h) > 0
    end

    # Rank-deficient rectangular DenseSVD and DeviceSVD
    c1 = T[1, 2, -1, 3, 0, 2]
    c2 = T[-2, 1, 3, 0, 2, -1]
    A_def_h = hcat(c1, c2, c1 .+ c2)
    x_pinv = pinv(A_def_h) * b_h

    d_svd = zeros(T, n)
    fac_svd = dense_factor(SVDFactorization(), prob, d_svd, nothing)
    fac_svd = factorize!!(fac_svd, SVDFactorization(), copy(A_def_h))
    @test fac_svd.rank[] == Int32(2)
    ldiv!!(d_svd, fac_svd, b_h)
    @test isapprox(d_svd, x_pinv; rtol = 32 * sqrt(eps(T)))
    rk_d, nn_d = factor_diagnostics(fac_svd, b_h)
    @test rk_d == Int32(2)
    @test nn_d > zero(T)

    # DeviceQR and DeviceSVD on JLArray
    A_d = JLArray(A_h)
    b_d = JLArray(b_h)
    for method in (QRFactorization(), SVDFactorization())
        d_dev = JLArray(zeros(T, n))
        fac_dev = dense_factor(method, prob, d_dev, nothing)
        @test size(fac_dev.factors) == (m, n)
        @test length(fac_dev.rhs) == m
        fac_dev = factorize!!(fac_dev, method, copy(A_d))
        @test returncode(fac_dev) === SUCCESS
        ldiv!!(d_dev, fac_dev, b_d)
        @test isapprox(Array(d_dev), x_ref; rtol = 32 * sqrt(eps(T)))
        rk_dev, nn_dev = factor_diagnostics(fac_dev, b_d)
        @test rk_dev == Int32(n)
        @test nn_dev === zero(T)
    end

    d_dev_svd = JLArray(zeros(T, n))
    fac_dev_svd = dense_factor(SVDFactorization(), prob, d_dev_svd, nothing)
    fac_dev_svd = factorize!!(fac_dev_svd, SVDFactorization(), JLArray(A_def_h))
    @test fac_dev_svd.rank[] == Int32(2)
    ldiv!!(d_dev_svd, fac_dev_svd, b_d)
    @test isapprox(Array(d_dev_svd), x_pinv; rtol = 64 * sqrt(eps(T)))

    # A rank-5 32 × 6 matrix: rounding leaves `σ̂₆/σ̂₁` above the cutoff of the 6 columns and below
    # that of the 32 rows, so the rank is read from the rows, and the truncated direction stays out
    # of the solve. The seed differs by precision so that the case sits inside that window.
    seed32 = T === Float32 ? 3 : 34
    rng32 = Xoshiro(seed32)
    A32 = randn(rng32, 32, 5) * randn(rng32, 5, 6)
    b32 = randn(rng32, 32)
    A_def32 = T.(A32)
    b_def32 = T.(b32)
    x32_ref = pinv(Float64.(A_def32); rtol = 10 * sqrt(6) * eps(T)) * Float64.(b_def32)
    prob32 = LeastSquaresProblem((r, x, p) -> r, 32)
    for AT in ARRAY_BACKENDS
        d32 = AT(zeros(T, 6))
        fac32 = factorize!!(
            dense_factor(SVDFactorization(), prob32, d32, nothing), SVDFactorization(),
            AT(copy(A_def32)))
        @test fac32.rank[] == Int32(5)
        ldiv!!(d32, fac32, AT(b_def32))
        # 256 eps against a measured 6.6e-7 (Float32) and 2.0e-15 (Float64); with the cutoff of the
        # columns the sixth direction stays in the solve, and the relative error is 6.2e6 and 4.0e15
        @test norm(Float64.(Array(d32)) - x32_ref) <= 256 * eps(T) * norm(x32_ref)
    end

    # Direct LM SVD step vs stacked QR step on rectangular LeastSquaresProblem (catches MU7)
    J_lm = copy(A_h)
    Fx_lm = copy(b_h)
    μ = T(0.7)
    # `LevenbergMarquardt` measures `μ` in the factor's power-of-two scale `s = 2^e`, `e` the
    # exponent of the largest entry of `J`, so the damped system is `(Jᴴ J + μ s² D²) δ = -Jᴴ F`.
    s_damp = ldexp(one(T), GeometricSolvers.matrix_exponent(J_lm))
    d_ref = -((J_lm' * J_lm + μ * s_damp^2 * I) \ (J_lm' * Fx_lm))
    # R1 LM SVD vs stacked QR
    J_s = SMatrix{m, n, T}(A_h)
    Fx_s = SVector{m, T}(b_h)
    d_s = zero(SVector{n, T})
    d_qr_s, ok_qr_s = lm_stacked_qr_step(nothing, J_s, Fx_s, μ, Val(:none))
    d_svd_s = lm_svd_step(nothing, factorize!!(nothing, SVDFactorization(), J_s), Fx_s, μ)
    @test ok_qr_s
    @test isapprox(Vector(d_qr_s), d_ref; rtol = 32 * sqrt(eps(T)))
    @test isapprox(Vector(d_svd_s), d_ref; rtol = 32 * sqrt(eps(T)))
    # R3 LM SVD vs stacked QR on Array and JLArray
    glob_qr = GeometricSolvers.init_globalization(LevenbergMarquardt(), zeros(T, n), Fx_lm)
    d_r3_qr = zeros(T, n)
    d_r3_svd = zeros(T, n)
    fac_r3_svd = dense_factor(SVDFactorization(), prob, d_r3_svd, nothing)
    fac_r3_svd = factorize!!(fac_r3_svd, SVDFactorization(), copy(J_lm))
    _, ok_r3_qr = lm_stacked_qr_step(d_r3_qr, copy(J_lm), Fx_lm, μ, Val(:none), glob_qr)
    lm_svd_step(d_r3_svd, fac_r3_svd, Fx_lm, μ)
    @test ok_r3_qr
    @test isapprox(d_r3_qr, d_ref; rtol = 32 * sqrt(eps(T)))
    @test isapprox(d_r3_svd, d_ref; rtol = 32 * sqrt(eps(T)))
    glob_qr_jl = GeometricSolvers.init_globalization(
        LevenbergMarquardt(), JLArray(zeros(T, n)), JLArray(Fx_lm))
    d_jl_qr = JLArray(zeros(T, n))
    _, ok_jl_qr = lm_stacked_qr_step(
        d_jl_qr, JLArray(J_lm), JLArray(Fx_lm), μ, Val(:none), glob_qr_jl)
    @test ok_jl_qr
    @test isapprox(Array(d_jl_qr), d_ref; rtol = 64 * sqrt(eps(T)))

    # `scaling = :diagonal`: `D = diag(dᵢ)`, `dᵢ = max(‖J[:, i]‖, √eps(R) maxⱼ ‖J[:, j]‖)`, and no
    # scale factor of `J`. A near-zero column exercises the floor, in the host path and in the
    # device kernel alike.
    A_floor = copy(A_h)
    A_floor[:, n] .= T(1e-9)
    colnorm = [norm(A_floor[:, i]) for i in 1:n]
    dfloor = iszero(maximum(colnorm)) ? one(T) : sqrt(eps(T)) * maximum(colnorm)
    D2 = Diagonal(collect(max(d, dfloor)^2 for d in colnorm))
    d_ref_diag = -((A_floor' * A_floor + μ * D2) \ (A_floor' * b_h))
    d_diag_s, ok_diag_s = lm_stacked_qr_step(
        nothing, SMatrix{m, n, T}(A_floor), SVector{m, T}(b_h), μ, Val(:diagonal))
    glob_diag = GeometricSolvers.init_globalization(LevenbergMarquardt(), zeros(T, n), b_h)
    d_diag_ar = zeros(T, n)
    _, ok_diag_ar = lm_stacked_qr_step(
        d_diag_ar, copy(A_floor), copy(b_h), μ, Val(:diagonal), glob_diag)
    glob_diag_jl = GeometricSolvers.init_globalization(
        LevenbergMarquardt(), JLArray(zeros(T, n)), JLArray(b_h))
    d_diag_jl = JLArray(zeros(T, n))
    _, ok_diag_jl = lm_stacked_qr_step(
        d_diag_jl, JLArray(A_floor), JLArray(b_h), μ, Val(:diagonal), glob_diag_jl)
    @test ok_diag_s && ok_diag_ar && ok_diag_jl
    @test isapprox(Vector(d_diag_s), d_ref_diag; rtol = 32 * sqrt(eps(T)))
    @test isapprox(d_diag_ar, d_ref_diag; rtol = 32 * sqrt(eps(T)))
    @test isapprox(Array(d_diag_jl), d_ref_diag; rtol = 64 * sqrt(eps(T)))
end

# ─── Clause 2: Moré–Garbow–Hillstrom nonzero-residual problems ───────────────

@testset "MGH nonzero-residual problems, $T" for T in REAL_ELTYPES
    opt = Options(T; max_iterations = 200)
    # `‖r*‖²` is a sum of squares of the residuals the solve reached, and every method below
    # reaches the published value to a few ulp of it: measured over the runs of this testset,
    # 1.59e-6 = 13.4 eps in Float32 and 2.75e-15 = 12.4 eps in Float64. The tolerance is 64 eps
    # in both precisions, which no wrong minimum reaches and which a 5× regression in the last
    # digits fails.
    val_rtol = 64 * eps(T)

    # MGH #11: Full-rank linear (m = 10, n = 5), f* = 5, x* = -ones(5)
    x0_11 = ones(SVector{5, T})
    prob11_s = LeastSquaresProblem(mgh11_static)
    for alg in (GaussNewton(),
        GaussNewton(; linsolve = SVDFactorization()),
        GaussNewton(; globalization = Backtracking()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        x, st = solve(prob11_s, x0_11, alg; options = opt)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test st.gradratio <= opt.g_tol
        @test isapprox(st.fnorm^2, T(5); rtol = val_rtol)
        @test isapprox(Vector(x), fill(T(-1), 5); atol = sqrt(eps(T)))
    end

    prob11_d = LeastSquaresProblem(mgh11_inplace!, 10)
    for AT in (Array, JLArray),
        alg in (GaussNewton(), Newton(; globalization = LevenbergMarquardt()))

        x0_d = AT(ones(T, 5))
        solver = init(prob11_d, x0_d, alg; options = opt)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test st.gradratio <= opt.g_tol
        @test isapprox(st.fnorm^2, T(5); rtol = val_rtol)
        @test isapprox(Array(x), fill(T(-1), 5); atol = sqrt(eps(T)))
    end

    # MGH #12: Rank-1 linear (m = 10, n = 5), f* = 15/7, rank = 1
    x0_12 = zeros(SVector{5, T})
    prob12_s = LeastSquaresProblem(mgh12_static)
    for alg in (GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        x, st = solve(prob12_s, x0_12, alg; options = opt)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test st.gradratio <= opt.g_tol
        @test isapprox(st.fnorm^2, T(15) / T(7); rtol = val_rtol)
        if alg.linsolve isa SVDFactorization && alg.globalization isa Static
            @test st.rank == Int32(1)
            x_minnorm = [T(j) / T(385) for j in 1:5]
            @test isapprox(Vector(x), x_minnorm; atol = 10 * sqrt(eps(T)))
        end
    end

    prob12_d = LeastSquaresProblem(mgh12_inplace!, 10)
    for alg in (GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        solver = init(prob12_d, zeros(T, 5), alg; options = opt)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, T(15) / T(7); rtol = val_rtol)
    end

    # MGH #8: Bard function (m = 15, n = 3), f* ≈ 8.214877306578994e-3
    x0_8 = ones(SVector{3, T})
    prob8_s = LeastSquaresProblem(mgh8_static)
    bard_min = T(8.214877306578994e-3)
    for alg in (GaussNewton(),
        GaussNewton(; linsolve = SVDFactorization()),
        GaussNewton(; globalization = Backtracking()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        x, st = solve(prob8_s, x0_8, alg; options = opt)
        @test isconverged(st)
        @test st.iterations >= 2
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test st.gradratio <= opt.g_tol
        @test isapprox(st.fnorm^2, bard_min; rtol = val_rtol)
    end

    prob8_d = LeastSquaresProblem(mgh8_inplace!, 15)
    for alg in (GaussNewton(),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        solver = init(prob8_d, ones(T, 3), alg; options = opt)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.iterations >= 2
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, bard_min; rtol = val_rtol)
    end

    # MGH #6: Jennrich–Sampson (m = 10, n = 2), f* ≈ 124.3621823556148 at x* ≈ (0.25782521, 0.25782521).
    # The Jacobian is singular at the minimum, so the undamped Gauss–Newton step is unbounded there:
    # only the damped step reaches f*. The damped runs start at the MGH standard start (0.3, 0.4) and
    # at (0.25, 0.26), nearer the minimum.
    prob6_s = LeastSquaresProblem(mgh6_static)
    js_min = T(124.3621823556148)
    for alg in (Newton(; globalization = LevenbergMarquardt()),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())),
        x0 in (SVector{2, T}(T(0.3), T(0.4)), SVector{2, T}(T(0.25), T(0.26)))

        x, st = solve(prob6_s, x0, alg; options = opt)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, js_min; rtol = val_rtol)
    end
    # The undamped step is unbounded at a minimum with equal Jacobian columns and a nonzero
    # residual, which is why no `GaussNewton()` run of this problem is asserted here.

    prob6_d = LeastSquaresProblem(mgh6_inplace!, 10)
    for (x0_js, alg) in ((T[0.3, 0.4], Newton(; globalization = LevenbergMarquardt())),
        (T[0.25, 0.26], Newton(; globalization = LevenbergMarquardt())))
        solver = init(prob6_d, x0_js, alg; options = opt)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, js_min; rtol = val_rtol)
    end
end

# ─── Clause 3: Zero-residual problem and stationarity guards (MU4, MU5, MU6) ─

@testset "Zero-residual and stationarity guards, $T" for T in REAL_ELTYPES
    # Check 3: a consistent overdetermined system, m = 6 > n = 3, with r(x*) = 0 at x* = (1, 2, -1)
    f_zero_s = (x, p) -> SVector{6}(
        x[1] - one(T),
        x[2] - T(2),
        x[3] + one(T),
        x[1] + x[2] - T(3),
        x[1] * x[3] + one(T),
        x[2] * x[3] + T(2))
    prob_zero_s = LeastSquaresProblem(f_zero_s)
    x0_z = SVector{3, T}(T(2), T(3), zero(T))
    for alg in (GaussNewton(), Newton(; globalization = LevenbergMarquardt()))
        x, st = solve(prob_zero_s, x0_z, alg)
        @test isconverged(st)
        @test st.code === SUCCESS
        @test st.stationary === false
        @test stopping_test(st) === :residual
        @test isapprox(Vector(x), T[1, 2, -1]; atol = 10 * sqrt(eps(T)))
    end

    f_zero_d! = (r, x, p) -> begin
        r[1] = x[1] - one(T)
        r[2] = x[2] - T(2)
        r[3] = x[3] + one(T)
        r[4] = x[1] + x[2] - T(3)
        r[5] = x[1] * x[3] + one(T)
        r[6] = x[2] * x[3] + T(2)
        return r
    end
    prob_zero_d = LeastSquaresProblem(f_zero_d!, 6)
    for alg in (GaussNewton(), Newton(; globalization = LevenbergMarquardt()))
        solver = init(prob_zero_d, T[2, 3, 0], alg)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.code === SUCCESS
        @test st.stationary === false
        @test stopping_test(st) === :residual
        @test isapprox(x, T[1, 2, -1]; atol = 10 * sqrt(eps(T)))
    end

    # MU6 guard: the residual test is tested before the stationarity test. The first step of
    # MGH #11 leaves both holding — ‖r‖ = √5 < f_abstol = 2.3, and the ratio 0.4 < g_tol = 0.5 —
    # and the status must name the residual test.
    opt_both = Options(T; f_abstol = T(2.3), g_tol = T(0.5))
    x_both, st_both = solve(LeastSquaresProblem(mgh11_static), ones(SVector{5, T}),
        GaussNewton(); options = opt_both)
    @test st_both.gradratio <= opt_both.g_tol
    @test st_both.code === SUCCESS
    @test stopping_test(st_both) === :residual

    # NonlinearProblem must NEVER pass the stationarity stopping test
    prob_nl_s = NonlinearProblem((x, p) -> SVector{2}(x[1]^2 + one(T), x[2]^2 + one(T)))
    _, st_nl_s = solve(prob_nl_s, SVector{2, T}(T(1e-4), T(1e-4)),
        Newton(; globalization = LevenbergMarquardt()); options = Options(T; max_iterations = 25))
    @test !isconverged(st_nl_s)
    @test st_nl_s.stationary === false
    @test stopping_test(st_nl_s) === :none

    prob_nl_d = NonlinearProblem((r, x, p) -> (
        r[1] = x[1]^2 + one(T); r[2] = x[2]^2 + one(T); r))
    solver_nl_d = init(prob_nl_d, T[1e-4, 1e-4],
        Newton(; globalization = LevenbergMarquardt()); options = Options(T; max_iterations = 25))
    _, st_nl_d = solve!(solver_nl_d)
    @test !isconverged(st_nl_d)
    @test st_nl_d.stationary === false
    @test stopping_test(st_nl_d) === :none

    # MU5 guard: LeastSquaresProblem with ‖J(x₀)‖_F == 0 at nonzero residual must NOT pass stationarity
    prob_zj_s = LeastSquaresProblem((x, p) -> SVector{3}(
        x[1]^2 + one(T), x[2]^2 + one(T), x[1]^2 + x[2]^2 + one(T)))
    for alg in (GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()))
        _, st_zj = solve(prob_zj_s, zero(SVector{2, T}), alg; options = Options(T; max_iterations = 10))
        @test !isconverged(st_zj)
        @test st_zj.stationary === false
        @test stopping_test(st_zj) === :none
    end

    prob_zj_d = LeastSquaresProblem((r, x, p) -> begin
            r[1] = x[1]^2 + one(T)
            r[2] = x[2]^2 + one(T)
            r[3] = x[1]^2 + x[2]^2 + one(T)
            return r
        end, 3)
    for alg in (GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()))
        solver_zj = init(prob_zj_d, zeros(T, 2), alg; options = Options(T; max_iterations = 10))
        _, st_zj = solve!(solver_zj)
        @test !isconverged(st_zj)
        @test st_zj.stationary === false
        @test stopping_test(st_zj) === :none
    end

    # MU4 guard: newton_stepkind on LeastSquaresProblem is MeasuredSlope(), so Backtracking() uses
    # φ'(0) = -‖J d‖₂² instead of -2 φ(0). On a problem with a large orthogonal residual C = 100,
    # 2 φ(0) = (x₁ - 1)² + C² >> ‖J d‖₂² = (x₁ - 1)², so ExactStep() rejects even α = 1!
    @test newton_stepkind(prob_zj_s, GaussNewton()) === MeasuredSlope()
    @test newton_stepkind(prob_zj_d, GaussNewton()) === MeasuredSlope()
    @test newton_stepkind(prob_zj_s, GaussNewton(), T, true) === MeasuredSlope()
    @test newton_stepkind(prob_zj_d, GaussNewton(), T, true) === MeasuredSlope()
    f_big_orth = (x, p) -> SVector{3}(x[1] - one(T), x[2] - T(2), T(100))
    prob_big_orth = LeastSquaresProblem(f_big_orth)
    x_bo, st_bo = solve(prob_big_orth, zero(SVector{2, T}),
        GaussNewton(; globalization = Backtracking()))
    @test isconverged(st_bo)
    @test st_bo.stationary === true
    @test isapprox(Vector(x_bo), T[1, 2]; atol = 10 * sqrt(eps(T)))

    # The stationarity test respects `min_iterations`: from the minimum of this problem every step
    # is zero, and the solve takes the minimum number of steps before it names the test (D5).
    for alg in (GaussNewton(), GaussNewton(; linsolve = SVDFactorization()))
        x_min, st_min = solve(prob_big_orth, SVector{2, T}(one(T), T(2)), alg;
            options = Options(T; min_iterations = 3))
        @test st_min.iterations >= 3
        @test st_min.code === SUCCESS
        @test stopping_test(st_min) === :stationarity
        @test isapprox(Vector(x_min), T[1, 2]; atol = 10 * sqrt(eps(T)))
    end
end

# ─── The iterate a `:stationarity` status names, and factor reuse ────────────

@testset "the iterate a :stationarity status returns is stationary, $T" for T in REAL_ELTYPES
    # `f` starts at 1e-9 with a slope of 1e-12 and grows quadratically, so the start is stationary
    # to `g_tol` and the undamped step is `-f / f′ = -1000`, where the ratio is 1. A status that
    # named the test on the reading at the start of the step would return that point.
    fs(t) = T(1e-9) + T(1e-12) * t + T(1e-3) * t^2
    r_st(x, p) = SVector(fs(x[1]), one(T))
    J_st(x, p) = SMatrix{2, 1, T, 2}(T(1e-12) + T(2e-3) * x[1], zero(T))
    function ratio_st(x)
        rv = r_st(x, nothing)
        Jm = J_st(x, nothing)
        return norm(Jm' * rv) / (norm(Jm) * norm(rv))
    end
    prob_st = LeastSquaresProblem(r_st, J_st)
    for alg in (GaussNewton(), GaussNewton(; linsolve = SVDFactorization()))
        @test ratio_st(SVector{1, T}(zero(T))) <= Options(T).g_tol
        x, st = solve(prob_st, SVector{1, T}(zero(T)), alg)
        @test st.code === SUCCESS
        @test stopping_test(st) === :stationarity
        @test ratio_st(x) <= Options(T).g_tol
        @test st.gradratio <= Options(T).g_tol
        # still at the start's residual, not at the 1000 of the step's end
        @test isapprox(st.fnorm, one(T); rtol = sqrt(eps(T)))
    end
end

@testset "factor reuse names the stationarity test at the minimum, $T" for T in REAL_ELTYPES
    # The diagnostics of a reused factor belong to the iterate before the step that reused it.
    # MGH #11 is linear, so the first step reaches the minimum and the second, on the reused factor,
    # is zero: the solve names the stationarity test there instead of stalling.
    for alg in (
        GaussNewton(; reuse = 2),
        GaussNewton(; linsolve = SVDFactorization(), reuse = 2))
        x, st = solve(LeastSquaresProblem(mgh11_static), ones(SVector{5, T}), alg)
        @test st.code === SUCCESS
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, T(5); rtol = 64 * eps(T))
        for AT in ARRAY_BACKENDS
            solver = init(LeastSquaresProblem(mgh11_inplace!, 10), AT(ones(T, 5)), alg)
            xa, sta = solve!(solver)
            @test sta.code === SUCCESS
            @test stopping_test(sta) === :stationarity
            @test isapprox(sta.fnorm^2, T(5); rtol = 64 * eps(T))
        end
    end
end

# ─── Check 5 under factor reuse, after a stationarity refresh that fails ─────

# r(t) = (1e-9 + 1e-12 t + 1e-3 t², 1) from t = 0: the start reads stationary, the first step goes
# to t = -1000, where the refreshed reading is not, so the solve goes on, and its next step reuses
# the factor. The refresh under reuse stores the factor it factorised, so R1 — whose factor is a
# value that no storage holds — and R3 — whose `factorize!!` writes into the factor's storage —
# continue from the factor of the same iterate and take the same steps.
@testset "R1 and R3 agree under reuse after a refresh, $T" for T in REAL_ELTYPES
    for reuse in (2, 3)
        fs(t) = T(1e-9) + T(1e-12) * t + T(1e-3) * t^2
        dfs(t) = T(1e-12) + T(2e-3) * t
        r_s(x, p) = SVector(fs(x[1]), one(T))
        J_s(x, p) = SMatrix{2, 1, T, 2}(dfs(x[1]), zero(T))
        function r_d!(r, x, p)
            view(r, 1:1) .= T(1e-9) .+ T(1e-12) .* x .+ T(1e-3) .* x .^ 2
            view(r, 2:2) .= one(T)
            return r
        end
        function J_d!(J, x, p)
            view(J, 1:1, 1:1) .= T(1e-12) .+ T(2e-3) .* reshape(x, 1, 1)
            view(J, 2:2, 1:1) .= zero(T)
            return J
        end
        for alg in (GaussNewton(; reuse), GaussNewton(; linsolve = SVDFactorization(), reuse))
            x1, st1 = solve(LeastSquaresProblem(r_s, J_s), SVector{1, T}(zero(T)), alg)
            for AT in ARRAY_BACKENDS
                @testset "reuse = $reuse, $(nameof(typeof(alg.linsolve))), $AT" begin
                    solver = init(LeastSquaresProblem(r_d!, J_d!, 2), AT(zeros(T, 1)), alg)
                    x3, st3 = solve!(solver)
                    @test st3.code === st1.code
                    @test st3.iterations == st1.iterations
                    @test isapprox(Array(x3)[1], x1[1]; rtol = 8 * eps(T), atol = 8 *
                                                                                  eps(T))
                end
            end
        end
    end
end

# The iterate that two steps of `m` from `x0` return, its residual, and the state that a
# [`refresh_diagnostics!!`](@ref) there leaves. The steps are the solve's own, so the state carries
# the factor the second step reused and the age that the step after the refresh reads.
function refreshed_state(prob, x0, m, p)
    state = GeometricSolvers.init_state(m, prob, x0, p)
    Fx = GeometricSolvers.residual!!(GeometricSolvers.residual_buffer(state), prob, x0, p)
    state = GeometricSolvers.start!!(state, m, prob, x0, Fx, p, GeometricSolvers.rnorm(Fx))
    for _ in 1:2
        x0, Fx, state, _ = GeometricSolvers.step!!(state, m, prob, x0, Fx, p)
    end
    return x0, Fx, GeometricSolvers.refresh_diagnostics!!(state, m, prob, x0, Fx, p)
end

# A factorisation of the Jacobian at the iterate, in the storage the regime's factors live in: R1's
# factor is a value, and R3 factorises into a dense buffer.
fresh_factor(ls, prob, ::SVector, J) = factorize!!(nothing, ls, J)

function fresh_factor(ls, prob, x::AbstractVector, J)
    return factorize!!(dense_factor(ls, prob, x, nothing), ls, J)
end

# The step of `fac` at the right-hand side `Fx`, in the storage of the regime: R1's `ldiv!!` returns
# the step, and R3's writes it into a buffer.
step_of(fac, ::SVector, Fx) = ldiv!!(nothing, fac, Fx)
step_of(fac, x::AbstractVector, Fx) = ldiv!!(similar(x), fac, Fx)

@testset "refresh_diagnostics!! stores the factor it factorises under reuse, $T" for T in REAL_ELTYPES

    # r(x) = (x² - 2, x - 1) from x = 3: two Gauss–Newton steps, the second on the reused factor,
    # leave an iterate whose reading is not stationary — the reading that a refresh makes before a
    # solve goes on. The refresh leaves the factor of that iterate and the age of a fresh
    # factorisation, or the reused step after it reads the factor of the earlier iterate.
    f_s(x, p) = SVector(x[1]^2 - T(2), x[1] - one(T))
    j_s(x, p) = SMatrix{2, 1, T, 2}(T(2) * x[1], one(T))
    function f_d!(r, x, p)
        view(r, 1:1) .= x .^ 2 .- T(2)
        view(r, 2:2) .= x .- one(T)
        return r
    end
    function j_d!(J, x, p)
        view(J, 1:1, 1:1) .= T(2) .* reshape(x, 1, 1)
        view(J, 2:2, 1:1) .= one(T)
        return J
    end
    alg = GaussNewton(; reuse = 3)
    prob_s = LeastSquaresProblem(f_s, j_s)
    prob_d = LeastSquaresProblem(f_d!, j_d!, 2)

    for (regime, prob, x0) in (("R1", prob_s, SVector{1, T}(T(3))),
        ("Array", prob_d, Array{T}([3])), ("JLArray", prob_d, JLArray(T[3])))
        x, Fx, state = refreshed_state(prob, x0, alg, nothing)
        @testset "$regime" begin
            # the refreshed reading fails, so a solve would go on from this iterate
            @test GeometricSolvers.state_gradratio(state) > Options(T).g_tol
            # the age of a factor that a step has just factorised, and no stall
            @test state.rule == GeometricSolvers.FactorAge(Int32(1), false)
            # the step of the factor the state kept is the step of a fresh factor at this iterate
            δ_stored = step_of(state.fac, x, Fx)
            J = GeometricSolvers.jacobian!!(state.J, state.jacprep, prob, x, nothing)
            fac = fresh_factor(alg.linsolve, prob, x, J)
            @test δ_stored == step_of(fac, x, Fx)
        end
        # a period of one step reuses no factor, so the refreshed factor is due at once
        for (reuse, age) in ((1, Int32(0)), (2, Int32(1)), (3, Int32(1)))
            m = GaussNewton(; reuse)
            _, _, state_k = refreshed_state(prob, x0, m, nothing)
            @test state_k.rule == GeometricSolvers.FactorAge(age, false)
        end
    end
end

# ─── Clause 4: 8-unit tanh network fit of sin(πx) on 101 points ──────────────

@testset "tanh network fit of sin(πx), $T" for T in REAL_ELTYPES
    m = 101
    K = 8
    n = 3 * K # 8 weights w_k, 8 biases b_k and 8 output weights c_k, and no output bias
    ts = range(T(-1), T(1); length = m)
    ys = sin.(T(π) .* collect(ts))

    function tanh_residual!(r, θ, _)
        @inbounds for i in 1:m
            ti = ts[i]
            pred = zero(T)
            for k in 1:K
                wk = θ[k]
                bk = θ[K + k]
                ck = θ[2 * K + k]
                pred += ck * tanh(wk * ti + bk)
            end
            r[i] = pred - ys[i]
        end
        return r
    end

    function tanh_jacobian!(J, θ, _)
        @inbounds for i in 1:m
            ti = ts[i]
            for k in 1:K
                wk = θ[k]
                bk = θ[K + k]
                ck = θ[2 * K + k]
                z = wk * ti + bk
                th = tanh(z)
                sech2 = one(eltype(θ)) - th * th
                J[i, k] = ck * sech2 * ti
                J[i, K + k] = ck * sech2
                J[i, 2 * K + k] = th
            end
        end
        return J
    end

    # The start is fixed and seeded: each hidden unit's weight starts at 2 and its bias at
    # -1.4 + 0.4 (k - 1), both jittered by `Xoshiro(1)` so that the eight units break symmetry from
    # the first step, and the output weights alternate ±0.5. The fit reaches 1.7e-5 of the start's
    # residual in Float32 and 1.6e-5 in Float64, inside the 1e-4 below.
    rng_tanh = Xoshiro(1)
    θ0 = Vector{T}(undef, n)
    for k in 1:K
        θ0[k] = T(2) * (one(T) + T(0.1) * T(randn(rng_tanh)))
        θ0[K + k] = T(-1.4) + T(0.4) * T(k - 1) + T(0.1) * T(randn(rng_tanh))
        θ0[2 * K + k] = isodd(k) ? T(0.5) : T(-0.5)
    end

    prob_tanh = LeastSquaresProblem(tanh_residual!, tanh_jacobian!, m)
    opt_tanh = Options(T; max_iterations = 200, f_abstol = T(1e-12),
        f_reltol = T(1e-6), g_tol = T(1e-10))
    solver = init(prob_tanh, copy(θ0), Newton(; globalization = LevenbergMarquardt()); options = opt_tanh)
    θ_sol, _ = solve!(solver)
    # The relative residual of the problem is measured against the start's residual; `‖r(θ₀)‖`
    # exceeds `‖y‖` by the factor `‖r(θ₀)‖ / ‖y‖`.
    r0 = tanh_residual!(zeros(T, m), θ0, nothing)
    r_sol = tanh_residual!(zeros(T, m), θ_sol, nothing)
    @test θ_sol != θ0
    @test norm(r_sol) / norm(r0) <= T(1e-4)
end

# ─── Clause 5: R1 inference/zero-alloc/JET and R3 Array/JLArray/AD ───────────

@testset "R1 inference, zero allocation, and JET" begin
    @testset "$T, M = $M" for T in REAL_ELTYPES, (M, N) in ((4, 2), (10, 5), (32, 4))

        prob = r1_least_squares(Val(M), Val(N), T)
        x0 = zeros(SVector{N, T})
        for alg in (GaussNewton(),
            GaussNewton(; linsolve = SVDFactorization()),
            Newton(; globalization = LevenbergMarquardt()),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
            res = @inferred solve(prob, x0, alg)
            @test isconverged(res[2])
            # skipped where a run forces --check-bounds=yes
            if Base.JLOptions().check_bounds != 1
                @test allocations(solve, prob, x0, alg) == 0
            end
        end
        # the control: `allocations` sees an allocation where there is one
        @test allocations(collect, x0) > 0
    end

    @testset "JET: solve on R1 LeastSquaresProblem is optimisable" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for T in REAL_ELTYPES
                f_jet = (x, p) -> SVector{4}(x[1] - one(T), x[2] - T(2), x[1] + x[2] - T(3), T(0.5))
                prob = LeastSquaresProblem(f_jet)
                x0 = zero(SVector{2, T})
                gn = GaussNewton()
                JET.test_opt(solve, (typeof(prob), typeof(x0), typeof(gn)))
            end
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end

@testset "R3 Array and JLArray zero allocation and AD backends, $T" for T in REAL_ELTYPES
    # AD backends on R1 and R3 rectangular LeastSquaresProblem. R1 differentiates with ForwardDiff
    # alone; the back-end comparison for the others is `test/ad/jacobian.jl`.
    prob_r1 = LeastSquaresProblem(mgh8_static)
    x0_r1 = ones(SVector{3, T})
    let ad = AutoForwardDiff()
        x, st = solve(prob_r1, x0_r1, GaussNewton(; jacobian = ad))
        @test isconverged(st)
        @test st.stationary === true
        @test isapprox(st.fnorm^2, T(8.214877306578994e-3); rtol = T === Float64 ? 1e-6 :
                                                                   1e-3)
    end

    # The finite-difference backend has no in-place path through DifferentiationInterface in this
    # version, so R3 is exercised with the chunked ForwardDiff backend; `test/ad/jacobian.jl` holds
    # the back-end comparison itself.
    prob_r3 = LeastSquaresProblem(mgh8_inplace!, 15)
    let ad = AutoForwardDiff()
        solver = init(prob_r3, ones(T, 3), GaussNewton(; jacobian = ad))
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.stationary === true
        @test isapprox(st.fnorm^2, T(8.214877306578994e-3); rtol = T === Float64 ? 1e-6 :
                                                                   1e-3)
    end

    # Zero allocation of solve!(solver) in R3 on Array with analytic Jacobian
    f_lin! = (r, x, _) -> begin
        r[1] = x[1] - one(T)
        r[2] = x[2] - T(2)
        r[3] = x[1] + x[2] - T(3)
        r[4] = T(0.5)
        return r
    end
    j_lin! = (J, x, _) -> begin
        fill!(J, zero(T))
        J[1, 1] = one(T)
        J[2, 2] = one(T)
        J[3, 1] = one(T)
        J[3, 2] = one(T)
        return J
    end
    prob_alloc = LeastSquaresProblem(f_lin!, j_lin!, 4)
    for alg in (GaussNewton(),
        GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        x0_a = zeros(T, 2)
        solver = init(prob_alloc, copy(x0_a), alg)
        solve!(solver)
        @test allocations(solve_from_start!, solver, x0_a) == 0
    end
    # the control: `allocations` sees an allocation where there is one
    @test allocations(copy, zeros(T, 2)) > 0

    # JLArray R3 under allowscalar(false) for GaussNewton (QR and SVD) and LM (QR and SVD)
    prob11_d = LeastSquaresProblem(mgh11_inplace!, 10)
    for alg in (GaussNewton(),
        GaussNewton(; linsolve = SVDFactorization()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        x0_jl = JLArray(ones(T, 5))
        solver_jl = init(prob11_d, x0_jl, alg)
        x_jl, st_jl = solve!(solver_jl)
        @test isconverged(st_jl)
        @test st_jl.stationary === true
        @test stopping_test(st_jl) === :stationarity
        @test isapprox(Array(x_jl), fill(T(-1), 5); atol = 10 * sqrt(eps(T)))
    end
end
