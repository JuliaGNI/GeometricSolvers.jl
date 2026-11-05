using ADTypes: AutoFiniteDiff, AutoForwardDiff
using Adapt: Adapt
using GeometricSolvers
using GeometricSolvers: DenseQR, DenseSVD, DeviceQR, DeviceSVD, MeasuredSlope, StepInfo,
                        dense_factor, factor_diagnostics, factorize!!, isconverged, ldiv!!,
                        lm_stacked_qr_step, lm_static_svd, lm_svd_step, newton_stepkind,
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

# ─── Constructors, Options.g_tol, SolverStatus.stationary, stopping_test ─────

@testset "LeastSquaresProblem, GaussNewton, Options.g_tol, and stopping_test, $T" for T in REAL_ELTYPES
    f_oop = (x, p) -> SVector{3, T}(x[1], x[2], x[1] + x[2])
    j_oop = (x, p) -> @SMatrix [one(T) zero(T); zero(T) one(T); one(T) one(T)]
    f_ip = (r, x, p) -> (r .= zero(T); r)
    j_ip = (J, x, p) -> (J .= zero(T); J)

    p1 = LeastSquaresProblem(f_oop)
    @test p1 isa LeastSquaresProblem{typeof(f_oop), Nothing, Nothing}
    p2 = LeastSquaresProblem(f_oop, j_oop)
    @test p2 isa LeastSquaresProblem{typeof(f_oop), typeof(j_oop), Nothing}
    p3 = LeastSquaresProblem(f_ip, 5)
    @test p3 isa LeastSquaresProblem{typeof(f_ip), Nothing, Int}
    @test p3.m == 5
    p4 = LeastSquaresProblem(f_ip, j_ip, 5)
    @test p4 isa LeastSquaresProblem{typeof(f_ip), typeof(j_ip), Int}
    @test p4.m == 5
    p5 = LeastSquaresProblem(f_ip; m = 5, jacobian = j_ip)
    @test p5.m == 5 && p5.J === j_ip

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
    @test record(st10, StepInfo{T}(SUCCESS, T(1), T(1e-6), Int32(0), false, false)).stationary === true
    st_fail = SolverStatus{T}(MAXITERS, Int32(10), T(1), T(1), Int32(0),
        false, false, Int32(2), T(0.5), T(1e-1))
    @test stopping_test(st_fail) === :none

    # LUFactorization without LevenbergMarquardt is rejected on LeastSquaresProblem
    x0_s = SVector{2, T}(1, 2)
    @test_throws ArgumentError solve(p1, x0_s, Newton(; linsolve = LUFactorization()))
    @test_throws ArgumentError init(p3, ones(T, 2), Newton(; linsolve = LUFactorization()))

    # Regularization > 0 on rectangular m > n is rejected
    @test_throws ArgumentError solve(p1, x0_s, Newton(; linsolve = QRFactorization(), regularization = 1e-6))
    @test_throws ArgumentError init(p3, ones(T, 2), Newton(; linsolve = QRFactorization(), regularization = 1e-6))

    # Underdetermined m < n throws DimensionMismatch
    p_under_s = LeastSquaresProblem((x, p) -> SVector{2, T}(x[1], x[2]))
    @test_throws DimensionMismatch solve(p_under_s, SVector{3, T}(1, 2, 3), GaussNewton())
    p_under_d = LeastSquaresProblem(f_ip, 2)
    @test_throws DimensionMismatch init(p_under_d, ones(T, 3), GaussNewton())
end

# ─── Clause 1c: Dense and device rectangular QR and SVD ──────────────────────

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
        factorize!!(fac, method, copy(A_h))
        @test returncode(fac) === SUCCESS
        ldiv!!(d, fac, b_h)
        @test isapprox(d, x_ref; rtol = 16 * sqrt(eps(T)))
        rk, nn = factor_diagnostics(fac, b_h)
        @test rk == Int32(n)
        @test nn === zero(T)

        A_work = copy(A_h)
        factorize!!(fac, method, A_work)
        ldiv!!(d, fac, b_h)
        alloc_fac = let fac = fac, method = method, A_work = A_work
            copyto!(A_work, A_h)
            factorize!!(fac, method, A_work)
            copyto!(A_work, A_h)
            @allocated factorize!!(fac, method, A_work)
        end
        alloc_ldiv = let d = d, fac = fac, b_h = b_h
            ldiv!!(d, fac, b_h)
            @allocated ldiv!!(d, fac, b_h)
        end
        @test alloc_fac == 0
        @test alloc_ldiv == 0
    end

    # Rank-deficient rectangular DenseSVD and DeviceSVD
    c1 = T[1, 2, -1, 3, 0, 2]
    c2 = T[-2, 1, 3, 0, 2, -1]
    A_def_h = hcat(c1, c2, c1 .+ c2)
    x_pinv = pinv(A_def_h) * b_h

    d_svd = zeros(T, n)
    fac_svd = dense_factor(SVDFactorization(), prob, d_svd, nothing)
    factorize!!(fac_svd, SVDFactorization(), copy(A_def_h))
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
        factorize!!(fac_dev, method, copy(A_d))
        @test returncode(fac_dev) === SUCCESS
        ldiv!!(d_dev, fac_dev, b_d)
        @test isapprox(Array(d_dev), x_ref; rtol = 32 * sqrt(eps(T)))
        rk_dev, nn_dev = factor_diagnostics(fac_dev, b_d)
        @test rk_dev == Int32(n)
        @test nn_dev === zero(T)
    end

    d_dev_svd = JLArray(zeros(T, n))
    fac_dev_svd = dense_factor(SVDFactorization(), prob, d_dev_svd, nothing)
    factorize!!(fac_dev_svd, SVDFactorization(), JLArray(A_def_h))
    @test fac_dev_svd.rank[] == Int32(2)
    ldiv!!(d_dev_svd, fac_dev_svd, b_d)
    @test isapprox(Array(d_dev_svd), x_pinv; rtol = 64 * sqrt(eps(T)))

    # Direct LM SVD step vs stacked QR step on rectangular LeastSquaresProblem (catches MU7)
    J_lm = copy(A_h)
    Fx_lm = copy(b_h)
    μ = T(0.7)
    d_ref = -((J_lm' * J_lm + μ * I) \ (J_lm' * Fx_lm))
    # R1 LM SVD vs stacked QR
    J_s = SMatrix{m, n, T}(A_h)
    Fx_s = SVector{m, T}(b_h)
    d_s = zero(SVector{n, T})
    d_qr_s, ok_qr_s = lm_stacked_qr_step(d_s, J_s, Fx_s, μ, Val(:none), nothing)
    d_svd_s = lm_svd_step(d_s, lm_static_svd(SVDFactorization(), J_s), Fx_s, μ)
    @test ok_qr_s
    @test isapprox(Vector(d_qr_s), d_ref; rtol = 32 * sqrt(eps(T)))
    @test isapprox(Vector(d_svd_s), d_ref; rtol = 32 * sqrt(eps(T)))
    # R3 LM SVD vs stacked QR on Array and JLArray
    glob_qr = GeometricSolvers.init_globalization(LevenbergMarquardt(), zeros(T, n), Fx_lm)
    d_r3_qr = zeros(T, n)
    d_r3_svd = zeros(T, n)
    fac_r3_svd = dense_factor(SVDFactorization(), prob, d_r3_svd, nothing)
    factorize!!(fac_r3_svd, SVDFactorization(), copy(J_lm))
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
end

# ─── Clause 2: Moré–Garbow–Hillstrom nonzero-residual problems ───────────────

@testset "MGH nonzero-residual problems, $T" for T in REAL_ELTYPES
    opt = Options(T; max_iterations = 200)
    rtol_val = T === Float64 ? 1e-10 : 1e-4

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
        @test isapprox(st.fnorm^2, T(5); rtol = rtol_val)
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
        @test isapprox(st.fnorm^2, T(5); rtol = rtol_val)
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
        @test isapprox(st.fnorm^2, T(15) / T(7); rtol = rtol_val)
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
        @test isapprox(st.fnorm^2, T(15) / T(7); rtol = rtol_val)
    end

    # MGH #8: Bard function (m = 15, n = 3), f* ≈ 8.214877306578994e-3
    x0_8 = ones(SVector{3, T})
    prob8_s = LeastSquaresProblem(mgh8_static)
    bard_min = T(8.214877306578994e-3)
    bard_rtol = T === Float64 ? 1e-8 : 5e-4
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
        @test isapprox(st.fnorm^2, bard_min; rtol = bard_rtol)
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
        @test isapprox(st.fnorm^2, bard_min; rtol = bard_rtol)
    end

    # MGH #6: Jennrich–Sampson (m = 10, n = 2), f* ≈ 124.3621823556148 at x* ≈ (0.25782521, 0.25782521)
    prob6_s = LeastSquaresProblem(mgh6_static)
    js_min = T(124.3621823556148)
    js_rtol = T === Float64 ? 1e-7 : 1e-3
    # LM from standard MGH start (0.3, 0.4)
    for alg in (Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()),
        GaussNewton(; globalization = Backtracking()))
        x, st = solve(prob6_s, SVector{2, T}(T(0.3), T(0.4)), alg; options = opt)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, js_min; rtol = js_rtol)
    end
    # Undamped GaussNewton() in the local basin (0.25, 0.26)
    x_gn6, st_gn6 = solve(prob6_s, SVector{2, T}(T(0.25), T(0.26)), GaussNewton(); options = opt)
    @test isconverged(st_gn6)
    @test st_gn6.stationary === true
    @test stopping_test(st_gn6) === :stationarity
    @test isapprox(st_gn6.fnorm^2, js_min; rtol = js_rtol)

    prob6_d = LeastSquaresProblem(mgh6_inplace!, 10)
    for (x0_js, alg) in ((T[0.3, 0.4], Newton(; globalization = LevenbergMarquardt())),
        (T[0.25, 0.26], GaussNewton()))
        solver = init(prob6_d, x0_js, alg; options = opt)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.stationary === true
        @test stopping_test(st) === :stationarity
        @test isapprox(st.fnorm^2, js_min; rtol = js_rtol)
    end
end

# ─── Clause 3: Zero-residual problem and stationarity guards (MU4, MU5, MU6) ─

@testset "Zero-residual and stationarity guards, $T" for T in REAL_ELTYPES
    # Consistent overdetermined problem: F(x*) = 0 at x* = (1, 2)
    f_zero_s = (x, p) -> SVector{4, T}(
        x[1] - one(T),
        x[2] - T(2),
        x[1] + x[2] - T(3),
        x[1]^2 - one(T))
    prob_zero_s = LeastSquaresProblem(f_zero_s)
    x0_z = SVector{2, T}(T(2), T(3))
    for alg in (GaussNewton(), Newton(; globalization = LevenbergMarquardt()))
        x, st = solve(prob_zero_s, x0_z, alg)
        @test isconverged(st)
        @test st.code === SUCCESS
        @test st.stationary === false
        @test stopping_test(st) === :residual
        @test isapprox(Vector(x), T[1, 2]; atol = 10 * sqrt(eps(T)))
    end

    f_zero_d! = (r, x, p) -> begin
        r[1] = x[1] - one(T)
        r[2] = x[2] - T(2)
        r[3] = x[1] + x[2] - T(3)
        r[4] = x[1]^2 - one(T)
        return r
    end
    prob_zero_d = LeastSquaresProblem(f_zero_d!, 4)
    for alg in (GaussNewton(), Newton(; globalization = LevenbergMarquardt()))
        solver = init(prob_zero_d, T[2, 3], alg)
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.code === SUCCESS
        @test st.stationary === false
        @test stopping_test(st) === :residual
        @test isapprox(x, T[1, 2]; atol = 10 * sqrt(eps(T)))
    end

    # NonlinearProblem must NEVER pass the stationarity stopping test
    prob_nl_s = NonlinearProblem((x, p) -> SVector{2, T}(x[1]^2 + one(T), x[2]^2 + one(T)))
    _, st_nl_s = solve(prob_nl_s, SVector{2, T}(T(1e-4), T(1e-4)),
        Newton(; globalization = LevenbergMarquardt()); options = Options(T; max_iterations = 25))
    @test !isconverged(st_nl_s)
    @test st_nl_s.stationary === false
    @test stopping_test(st_nl_s) === :none

    prob_nl_d = NonlinearProblem((r, x, p) -> (r[1] = x[1]^2 + one(T); r[2] = x[2]^2 + one(T); r))
    solver_nl_d = init(prob_nl_d, T[1e-4, 1e-4],
        Newton(; globalization = LevenbergMarquardt()); options = Options(T; max_iterations = 25))
    _, st_nl_d = solve!(solver_nl_d)
    @test !isconverged(st_nl_d)
    @test st_nl_d.stationary === false
    @test stopping_test(st_nl_d) === :none

    # MU5 guard: LeastSquaresProblem with ‖J(x₀)‖_F == 0 at nonzero residual must NOT pass stationarity
    prob_zj_s = LeastSquaresProblem((x, p) -> SVector{3, T}(
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
    @test newton_stepkind(prob_zj_s, QRFactorization(), NoRefinement()) === MeasuredSlope()
    @test newton_stepkind(prob_zj_d, QRFactorization(), NoRefinement()) === MeasuredSlope()
    f_big_orth = (x, p) -> SVector{3, T}(x[1] - one(T), x[2] - T(2), T(100))
    prob_big_orth = LeastSquaresProblem(f_big_orth)
    x_bo, st_bo = solve(prob_big_orth, zero(SVector{2, T}),
        GaussNewton(; globalization = Backtracking()))
    @test isconverged(st_bo)
    @test st_bo.stationary === true
    @test isapprox(Vector(x_bo), T[1, 2]; atol = 10 * sqrt(eps(T)))
end

# ─── Clause 4: 8-unit tanh network fit of sin(πx) on 101 points ──────────────

@testset "tanh network fit of sin(πx), $T" for T in REAL_ELTYPES
    m = 101
    K = 8
    n = 3 * K + 1 # 8 weights w_k, 8 biases b_k, 8 output weights c_k, 1 output bias d = 25 params
    ts = range(T(-1), T(1); length = m)
    ys = sin.(T(π) .* collect(ts))
    ynorm = norm(ys)

    function tanh_residual!(r, θ, _)
        d_bias = θ[3 * K + 1]
        @inbounds for i in 1:m
            ti = ts[i]
            pred = d_bias
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
            J[i, 3 * K + 1] = one(eltype(θ))
        end
        return J
    end

    # Fixed seeded/deterministic start breaking symmetry across the 8 hidden units
    θ0 = Vector{T}(undef, n)
    for k in 1:K
        θ0[k] = T(2)
        θ0[K + k] = T(-1.4 + 0.4 * (k - 1))
        θ0[2 * K + k] = isodd(k) ? T(0.5) : T(-0.5)
    end
    θ0[3 * K + 1] = zero(T)

    prob_tanh = LeastSquaresProblem(tanh_residual!, tanh_jacobian!, m)
    opt_tanh = Options(T; max_iterations = 200, f_abstol = T(1e-10), g_tol = T(1e-10))
    solver = init(prob_tanh, copy(θ0), Newton(; globalization = LevenbergMarquardt()); options = opt_tanh)
    θ_sol, st = solve!(solver)
    @test θ_sol != θ0
    @test st.fnorm / ynorm <= T(1e-4)
end

# ─── Clause 5: R1 inference/zero-alloc/JET and R3 Array/JLArray/AD ───────────

@testset "R1 inference, zero allocation, and JET" begin
    @testset "$T, M = $M" for T in REAL_ELTYPES, (M, N) in ((4, 2), (10, 5), (32, 4))
        f_r1 = (x, p) -> SVector{M, T}(ntuple(i -> i <= N ? x[i] - one(T) : T(0.1) * x[1], Val(M)))
        prob = LeastSquaresProblem(f_r1)
        x0 = zeros(SVector{N, T})
        for alg in (GaussNewton(),
            GaussNewton(; linsolve = SVDFactorization()),
            Newton(; globalization = LevenbergMarquardt()),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
            res = @inferred solve(prob, x0, alg)
            @test isconverged(res[2])
            # skipped where a run forces --check-bounds=yes
            if Base.JLOptions().check_bounds != 1
                alloc = let prob = prob, x0 = x0, alg = alg
                    solve(prob, x0, alg)
                    @allocated solve(prob, x0, alg)
                end
                @test alloc == 0
            end
        end
    end

    @testset "JET: solve on R1 LeastSquaresProblem is optimisable" begin
        JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        if JET_WORKS
            for T in REAL_ELTYPES
                f_jet = (x, p) -> SVector{4, T}(x[1] - one(T), x[2] - T(2), x[1] + x[2] - T(3), T(0.5))
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
    # AD backends on R1 and R3 rectangular LeastSquaresProblem
    prob_r1 = LeastSquaresProblem(mgh8_static)
    x0_r1 = ones(SVector{3, T})
    for ad in (AutoForwardDiff(), AutoFiniteDiff())
        x, st = solve(prob_r1, x0_r1, GaussNewton(; jacobian = ad))
        @test isconverged(st)
        @test st.stationary === true
        @test isapprox(st.fnorm^2, T(8.214877306578994e-3); rtol = T === Float64 ? 1e-6 : 1e-3)
    end

    prob_r3 = LeastSquaresProblem(mgh8_inplace!, 15)
    for ad in (AutoForwardDiff(), AutoFiniteDiff())
        solver = init(prob_r3, ones(T, 3), GaussNewton(; jacobian = ad))
        x, st = solve!(solver)
        @test isconverged(st)
        @test st.stationary === true
        @test isapprox(st.fnorm^2, T(8.214877306578994e-3); rtol = T === Float64 ? 1e-6 : 1e-3)
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
        alloc = let solver = solver, x0_a = x0_a
            copyto!(solver.x, x0_a)
            solve!(solver)
            copyto!(solver.x, x0_a)
            @allocated solve!(solver)
        end
        @test alloc == 0
    end

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
