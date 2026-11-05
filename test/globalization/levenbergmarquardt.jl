using Adapt: Adapt
using GeometricSolvers
using JLArrays: JLArray
using LinearAlgebra: norm
using StaticArrays: @SMatrix, @SVector, MVector, SMatrix, SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/standard_cells.jl")

@testset "LevenbergMarquardt" begin
    @testset "Constructor and Adapt (MU9)" begin
        lm = LevenbergMarquardt()
        @test lm.μ₀ === 1e-6
        @test lm.μ_min === 1e-15
        @test lm.μ_max === 1e12
        @test lm.μ_decrease === 1.0 / 3.0
        @test lm.μ_increase === 4.0
        @test lm.ρ_accept === 0.0
        @test lm.scaling === Val(:none)

        lm_diag = LevenbergMarquardt(Float32; μ₀ = 1e-4, scaling = :diagonal)
        @test lm_diag.μ₀ === 1.0f-4
        @test lm_diag.μ_min === 4 * eps(Float32)
        @test lm_diag.scaling === Val(:diagonal)

        @test_throws ArgumentError LevenbergMarquardt(; μ_min = 0.0)
        @test_throws ArgumentError LevenbergMarquardt(; μ_min = -1e-12)
        @test_throws ArgumentError LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-12)
        @test_throws ArgumentError LevenbergMarquardt(; μ₀ = 1e13, μ_max = 1e12)
        @test_throws ArgumentError LevenbergMarquardt(; μ_max = Inf)
        @test_throws ArgumentError LevenbergMarquardt(; μ_decrease = 0.0)
        @test_throws ArgumentError LevenbergMarquardt(; μ_decrease = 1.0)
        @test_throws ArgumentError LevenbergMarquardt(; μ_increase = 1.0)
        @test_throws ArgumentError LevenbergMarquardt(; ρ_accept = -0.1)
        @test_throws ArgumentError LevenbergMarquardt(; ρ_accept = 1.0)
        @test_throws ArgumentError LevenbergMarquardt(; scaling = :bad)
        @test_throws ArgumentError Newton(; globalization = LevenbergMarquardt(), reuse = 2)

        lm32 = Adapt.adapt(GeometricSolvers.ToReal{Float32}(), lm)
        @test lm32 isa LevenbergMarquardt{Float32}
        @test lm32.μ₀ === Float32(1e-6)
        @test lm32.μ_decrease === Float32(1.0 / 3.0)
        @test lm32.scaling === Val(:none)
    end

    @testset "Check 1: Quadratic tail at μ_min (MU1) and Powell singular 2D" begin
        prob_quad = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2.0, x[2]^2 - 3.0))
        m_quad = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 9e-15, μ_min = 1e-15))
        x_q = SVector(50.0, 50.0)
        Fx_q = prob_quad.F(x_q, nothing)
        st_q = GeometricSolvers.init_state(m_quad, prob_quad, x_q, nothing)
        st_q = GeometricSolvers.start!!(
            st_q, m_quad, prob_quad, x_q, Fx_q, nothing, GeometricSolvers.rnorm(Fx_q))
        fnorms_q = Float64[GeometricSolvers.rnorm(Fx_q)]
        mus_q = Float64[]
        for _ in 1:20
            x_q, Fx_q, st_q, info_q = GeometricSolvers.step!!(
                st_q, m_quad, prob_quad, x_q, Fx_q, nothing)
            push!(fnorms_q, info_q.fnorm)
            push!(mus_q, st_q.rule.μ)
            info_q.fnorm <= 1e-14 && break
        end
        @test length(mus_q) >= 4
        @test all(==(1e-15), mus_q[(end - 2):end])
        @test fnorms_q[end] <= 1e-14
        @test fnorms_q[end - 1] <= 2.0 * fnorms_q[end - 2]^2

        F_powell(x, _) = @SVector [x[1], 10.0 * x[1] / (x[1] + 0.1) + 2.0 * x[2]^2]
        function J_powell(x, _)
            return @SMatrix [1.0 0.0;
                             1.0 / (x[1] + 0.1)^2 4.0 * x[2]]
        end
        prob = NonlinearProblem(F_powell, J_powell)
        x0 = @SVector [3.0, 1.0]
        opts = Options(Float64; f_abstol = 1e-12, f_reltol = 1e-12, max_iterations = 200)
        for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
            u_sol, st_sol = solve(
                prob, x0,
                Newton(; linsolve = ls, globalization = LevenbergMarquardt()),
                nothing; options = opts)
            @test st_sol.code == SUCCESS
            @test norm(u_sol - @SVector [0.0, 0.0]) < 1e-3
            @test st_sol.fnorm <= 1e-12
        end
    end

    @testset "Check 2 & 3: Rank-one consistent system (MU2) and single SVD per step (MU3)" begin
        prob_f3_r1 = NonlinearProblem(
            (x, _) -> SVector(x[1] + x[2] - 2.0, 2.0 * (x[1] + x[2]) - 4.0),
            (_, _) -> @SMatrix [1.0 1.0; 2.0 2.0 + 1e-16])
        x_f3, st_f3 = solve(
            prob_f3_r1,
            SVector(0.0, 0.0),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_f3.code == SUCCESS
        @test norm(x_f3 - SVector(1.0, 1.0)) <= 1e-6

        prob_f3_r3 = NonlinearProblem(
            (F, x, _) -> begin
                F[1] = x[1] + x[2] - 2.0
                F[2] = 2.0 * (x[1] + x[2]) - 4.0
                return nothing
            end,
            (J, _, _) -> begin
                J[1, 1] = 1.0
                J[1, 2] = 1.0
                J[2, 1] = 2.0
                J[2, 2] = 2.0 + 1e-16
                return nothing
            end)
        x_f3_r3, st_f3_r3 = solve(
            prob_f3_r3, [0.0, 0.0],
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_f3_r3.code == SUCCESS
        @test norm(x_f3_r3 - [1.0, 1.0]) <= 1e-6

        jac_calls = Ref(0)
        prob_reject = NonlinearProblem(
            (x, _) -> (norm(x) > 1.5 ? SVector(NaN, NaN) : SVector(x[1] - 1.0, x[2] - 1.0)),
            (_, _) -> (jac_calls[] += 1; @SMatrix [0.01 0.0; 0.0 0.01]))
        m_svd_lm = Newton(;
            linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
        x0_rej = SVector(0.0, 0.0)
        Fx0_rej = prob_reject.F(x0_rej, nothing)
        st_rej = GeometricSolvers.init_state(m_svd_lm, prob_reject, x0_rej, nothing)
        st_rej = GeometricSolvers.start!!(
            st_rej, m_svd_lm, prob_reject, x0_rej, Fx0_rej, nothing,
            GeometricSolvers.rnorm(Fx0_rej))
        _, _, _, info_rej = GeometricSolvers.step!!(
            st_rej, m_svd_lm, prob_reject, x0_rej, Fx0_rej, nothing)
        @test info_rej.code == SUCCESS
        @test info_rej.failures >= 2
        @test jac_calls[] == 1
    end

    @testset "Damped step formula in REAL_ELTYPES" begin
        for R in REAL_ELTYPES
            A = @SMatrix [R(3) R(1); R(1) R(2)]
            b = @SVector [R(4), R(3)]
            x0 = @SVector [R(0), R(0)]
            μ0 = R(1e-2)
            prob_lin = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
            opts = Options(R; f_abstol = R(1e-30), f_reltol = R(1e-30), max_iterations = 1)
            δ_ref = -((transpose(A) * A + μ0 * SMatrix{2, 2, R}(1, 0, 0, 1)) \
                      (transpose(A) * (-b)))
            for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
                u_sol, _ = solve(
                    prob_lin, x0,
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(R; μ₀ = μ0, scaling = :none)),
                    nothing; options = opts)
                @test isapprox(u_sol, δ_ref; rtol = 50 * eps(R))
            end

            d1 = hypot(R(3), R(1))
            d2 = hypot(R(1), R(2))
            D2 = @SMatrix [d1^2 R(0); R(0) d2^2]
            δ_diag_ref = -((transpose(A) * A + μ0 * D2) \ (transpose(A) * (-b)))
            u_diag, _ = solve(
                prob_lin, x0,
                Newton(;
                    linsolve = QRFactorization(),
                    globalization = LevenbergMarquardt(R; μ₀ = μ0, scaling = :diagonal)),
                nothing; options = opts)
            @test isapprox(u_diag, δ_diag_ref; rtol = 50 * eps(R))
        end
    end

    @testset "Check 4: Underflow-safe SVD weight (MU7) and STANDARD_COLUMNS" begin
        J_sub = @SMatrix [2e-200 0.0; 0.0 1.0]
        prob_sub = NonlinearProblem((x, _) -> J_sub * x, (_, _) -> J_sub)
        x0_sub = @SVector [1.0, 0.0]
        u_sub, _ = solve(
            prob_sub, x0_sub,
            Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-6)),
            nothing;
            options = Options(Float64; f_abstol = 1e-300, f_reltol = 1e-12, max_iterations = 5))
        @test all(isfinite, u_sub)
        @test u_sub[1] < 1.0

        for scale in (ldexp(1.0, 400), ldexp(1.0, -400))
            J_ext = @SMatrix [scale 0.0; 0.0 scale]
            b_ext = J_ext * @SVector([1.0, -2.0])
            prob_ext = NonlinearProblem((x, _) -> J_ext * x - b_ext, (_, _) -> J_ext)
            u_ext, _ = solve(
                prob_ext,
                @SVector([0.0, 0.0]),
                Newton(;
                    linsolve = SVDFactorization(),
                    globalization = LevenbergMarquardt(; μ₀ = 1e-6)),
                nothing;
                options = Options(Float64; f_abstol = 1e-300, f_reltol = 1e-12, max_iterations = 1))
            @test all(isfinite, u_ext)
            @test isapprox(
                u_ext[1],
                scale^2 / (scale^2 + 1e-6 * scale^2);
                rtol = 1e-12)
            @test isapprox(
                u_ext[2],
                -2.0 * scale^2 / (scale^2 + 1e-6 * scale^2);
                rtol = 1e-12)
        end

        @test length(STANDARD_COLUMNS) == 10
        @test "Newton LevenbergMarquardt SVD" in first.(STANDARD_COLUMNS)
    end

    @testset "Check 5: Non-finite trial residual rejection and STALLED on μ > μ_max" begin
        function F_nan(x, _)
            if abs(x[1] - 2.0) > 0.25
                return @SVector [NaN, NaN]
            end
            return @SVector [x[1] - 1.0, x[2]]
        end
        J_nan(_, _) = @SMatrix [1.0 0.0; 0.0 1.0]
        prob_nan = NonlinearProblem(F_nan, J_nan)
        x0_nan = @SVector [2.0, 0.0]
        u_nan, st_nan = solve(
            prob_nan, x0_nan,
            Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-2)),
            nothing;
            options = Options(Float64; f_abstol = 1e-10, f_reltol = 1e-10, max_iterations = 50))
        @test st_nan.code == SUCCESS
        @test u_nan ≈ @SVector [1.0, 0.0] atol = 1e-8
        @test st_nan.step_failures >= 1

        prob_stall = NonlinearProblem(
            (_, _) -> @SVector([1.0, 1.0]),
            (_, _) -> @SMatrix([1.0 0.0; 0.0 1.0]))
        x0_stall = @SVector [2.0, 3.0]
        u_stall, st_stall = solve(
            prob_stall, x0_stall,
            Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(;
                    μ₀ = 1.0, μ_max = 16.0, μ_increase = 4.0)),
            nothing;
            options = Options(Float64; f_abstol = 1e-12, f_reltol = 1e-12, max_iterations = 10))
        @test st_stall.code == STALLED
        @test u_stall === x0_stall
    end

    @testset "Check 6a/6b/6c: Stacked QR accuracy, μ floor, and uninitialised rule guard" begin
        J_ill = @SMatrix [1.0 1.0; 1.0 1.0 + 1e-8]
        x_true = @SVector [1.0, -1.0]
        b_ill = J_ill * x_true
        prob_ill = NonlinearProblem((x, _) -> J_ill * x - b_ill, (_, _) -> J_ill)
        x0_ill = @SVector [0.0, 0.0]
        opts_ill = Options(Float64; f_abstol = 1e-30, f_reltol = 1e-30, max_iterations = 1)
        u_qr_ill, _ = solve(
            prob_ill, x0_ill,
            Newton(;
                linsolve = QRFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-16)),
            nothing; options = opts_ill)
        u_svd_ill, _ = solve(
            prob_ill, x0_ill,
            Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-16)),
            nothing; options = opts_ill)
        @test norm(u_qr_ill - x_true) < 1e-6
        @test isapprox(u_qr_ill, u_svd_ill; rtol = 1e-5)

        m_lm = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-6, μ_min = 5e-7, μ_decrease = 0.25))
        prob_quad = NonlinearProblem(
            (x, _) -> @SVector([x[1] - 1.0, x[2] - 2.0]),
            (_, _) -> @SMatrix([1.0 0.0; 0.0 1.0]))
        x_init = @SVector [0.0, 0.0]
        Fx_init = prob_quad.F(x_init, nothing)
        st0 = GeometricSolvers.init_state(m_lm, prob_quad, x_init, nothing)
        @test isnan(st0.rule.μ)
        x_bad, Fx_bad, st_bad, info_bad = GeometricSolvers.step!!(
            st0, m_lm, prob_quad, x_init, Fx_init, nothing)
        @test info_bad.code == NONFINITE
        @test x_bad === x_init
        @test Fx_bad === Fx_init
        @test isnan(st_bad.rule.μ)

        st1 = GeometricSolvers.start!!(
            st0, m_lm, prob_quad, x_init, Fx_init, nothing, GeometricSolvers.rnorm(Fx_init))
        @test st1.rule.μ === 1e-6
        x1, Fx1, st2, info1 = GeometricSolvers.step!!(
            st1, m_lm, prob_quad, x_init, Fx_init, nothing)
        @test info1.code == SUCCESS
        @test st2.rule.μ === 5e-7
        _, _, st_fin = GeometricSolvers.finish!!(
            st2, m_lm, prob_quad, x1, Fx1,
            SolverStatus{Float64}(
                SUCCESS, Int32(1), info1.fnorm, info1.stepnorm, Int32(0), false))
        @test isnan(st_fin.rule.μ)
    end

    @testset "Check 7: R1/R2/R3 storage and zero-allocation solve!" begin
        function F_r3!(F, x, _)
            F[1] = 10.0 * (x[2] - x[1]^2)
            F[2] = 1.0 - x[1]
            return nothing
        end
        function J_r3!(J, x, _)
            J[1, 1] = -20.0 * x[1]
            J[1, 2] = 10.0
            J[2, 1] = -1.0
            J[2, 2] = 0.0
            return nothing
        end
        prob_r3 = NonlinearProblem(F_r3!, J_r3!)
        x0_r3 = [-1.2, 1.0]
        solver_r3 = init(
            prob_r3, copy(x0_r3),
            Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-4));
            options = Options(
                Float64; f_abstol = 1e-10, f_reltol = 1e-10, max_iterations = 200))
        _, st_r3 = solve!(solver_r3)
        @test st_r3.code == SUCCESS
        @test norm(solver_r3.state.Fx) < 1e-10
        alloc_r3 = @allocated begin
            copyto!(solver_r3.x, x0_r3)
            solve!(solver_r3)
        end
        @test alloc_r3 == 0

        # Scalar Number, MVector, JLArray, and ComplexF64
        prob_scalar = NonlinearProblem((x, _) -> x^2 - 2.0, (x, _) -> 2.0 * x)
        x_sc, st_sc = solve(
            prob_scalar, 1.5,
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_sc.code == SUCCESS
        @test isapprox(x_sc, sqrt(2.0); atol = 1e-10)

        prob_mv = NonlinearProblem(
            (x, _) -> MVector{2, Float64}(x[1]^2 - 1.0, x[2] - 2.0),
            (x, _) -> SMatrix{2, 2, Float64}(2.0 * x[1], 0.0, 0.0, 1.0))
        x_mv, st_mv = solve(
            prob_mv, MVector{2, Float64}(2.0, 0.0),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_mv.code == SUCCESS
        @test isapprox(x_mv, MVector{2, Float64}(1.0, 2.0); atol = 1e-10)

        prob_jl = NonlinearProblem((F, x, _) -> begin
            F .= x .^ 2 .- 4.0
            return nothing
        end)
        x_jl, st_jl = solve(
            prob_jl, JLArray([3.0, 3.0]),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_jl.code == SUCCESS
        @test isapprox(Array(x_jl), [2.0, 2.0]; atol = 1e-10)
    end
end
