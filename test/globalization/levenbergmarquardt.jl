# `LevenbergMarquardt`, the damped-step globalization of `Newton`: the parameters and their
# validation, the adapted damping, the rule state and its fail-closed initialisation, the damped
# step against the damped normal equations in every element type and on both array ranks, the one
# factorisation per Jacobian across rejected trials, the truncation of the ranks the factor drops,
# the scale invariance of the step, the `STALLED` return at `μ_max`, the rejection of a non-finite
# trial residual, the standard reference column, and the storage of the state.
#
# The cases that need the step in closed form take a linear problem with an analytic `J` and one
# `step!!`, so that the expected `δ` is the solution of `(Jᴴ J + μ s² D²) δ = -Jᴴ F` computed
# directly, with `s = 2^e` the factor's power-of-two scale of `J` and `D` the identity or, under
# `scaling = :diagonal`, the diagonal of the floored column norms.

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: I, norm
using StaticArrays: @SMatrix, @SVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: finish!!, init_state, matrix_exponent, rnorm, start!!, step!!

include("../helpers/matrix.jl")
include("../helpers/standard_cells.jl")

allowscalar(false)

# The power-of-two scale `2^e` the factorisations divide a matrix by, from its largest entry: the
# unit in which `LevenbergMarquardt` measures `μ`.
scale_of(J) = ldexp(one(real(eltype(J))), matrix_exponent(J))

# The damped step `(Jᴴ J + μ s² I) δ = -Jᴴ F` solved directly, for a linear problem `F = J x - b`
# from `x₀ = 0` at the start of a solve, with the scale `s = scale_of(J)`.
function damped_reference(A, b, μ)
    s2 = scale_of(A)^2
    return -((transpose(A) * A + μ * s2 * SMatrix{2, 2, eltype(A)}(I)) \
             (transpose(A) * (-b)))
end

# The same with Marquardt's column scaling, `D = diag(dᵢ²)`, `dᵢ = max(‖A[:, i]‖, √eps · maxⱼ dⱼ)`.
function damped_reference_diagonal(A, b, μ)
    R = real(eltype(A))
    d = [norm(A[:, i]) for i in 1:2]
    floor = iszero(maximum(d)) ? one(R) : sqrt(eps(R)) * maximum(d)
    D = SMatrix{2, 2, R}(max(d[1], floor)^2, 0, 0, max(d[2], floor)^2)
    return -((transpose(A) * A + μ * D) \ (transpose(A) * (-b)))
end

@testset "LevenbergMarquardt" begin
    @testset "Constructor, validation and Adapt (MU9)" begin
        lm = LevenbergMarquardt()
        @test lm.μ₀ === 1e-6
        @test lm.μ_min === max(1e-15, 4 * eps(Float64))
        @test lm.μ_max === 1e12
        @test lm.μ_decrease === 1.0 / 3.0
        @test lm.μ_increase === 4.0
        @test lm.ρ_accept === 0.0
        @test lm.scaling === Val(:none)

        lm32 = LevenbergMarquardt(Float32; μ₀ = 1e-4, scaling = :diagonal)
        @test lm32 isa LevenbergMarquardt{Float32}
        @test lm32.μ₀ === 1.0f-4
        @test lm32.μ_min === 4 * eps(Float32)
        @test lm32.scaling === Val(:diagonal)

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

        # MU9: without `Adapt.@adapt_structure` the fields stay `Float64` under `ToReal{Float32}()`
        adapted = Adapt.adapt(GeometricSolvers.ToReal{Float32}(), lm)
        @test adapted isa LevenbergMarquardt{Float32}
        @test adapted.μ₀ === Float32(1e-6)
        @test adapted.μ_decrease === Float32(1.0 / 3.0)
        @test adapted.scaling === Val(:none)
    end

    @testset "Damped step against the damped normal equations in ELTYPES" begin
        for R in REAL_ELTYPES
            A = @SMatrix [R(3) R(1); R(1) R(2)]
            b = @SVector [R(4), R(3)]
            μ0 = R(1e-2)
            prob = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
            opts = Options(R; f_abstol = R(1e-30), f_reltol = R(0), max_iterations = 1)
            δ_none = damped_reference(A, b, μ0)
            δ_diag = damped_reference_diagonal(A, b, μ0)
            for (ls, D) in ((SVDFactorization(), δ_none), (QRFactorization(), δ_none),
                (LUFactorization(), δ_none))
                x, _ = solve(
                    prob, zero(SVector{2, R}),
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(R; μ₀ = μ0, μ_min = μ0)),
                    nothing; options = opts)
                @test isapprox(x, D; rtol = 100 * eps(R))
            end
            # `scaling = :diagonal` takes the stacked QR path for every linear method.
            for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
                x, _ = solve(
                    prob, zero(SVector{2, R}),
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(R; μ₀ = μ0, μ_min = μ0,
                            scaling = :diagonal)),
                    nothing; options = opts)
                @test isapprox(x, δ_diag; rtol = 100 * eps(R))
            end
        end
        # A zero Jacobian column still receives positive damping under `:diagonal`.
        A = @SMatrix [3.0 0.0; 1.0 0.0]
        b = @SVector [4.0, 3.0]
        prob = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
        x, _ = solve(
            prob, zero(SVector{2, Float64}),
            Newton(;
                linsolve = QRFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-2, μ_min = 1e-2,
                    scaling = :diagonal)),
            nothing; options = Options(Float64; f_abstol = 1e-30, f_reltol = 0.0,
                max_iterations = 1))
        @test all(isfinite, x)
        @test isapprox(x, damped_reference_diagonal(A, b, 1e-2); rtol = 1e-10)
    end

    @testset "Check 1: quadratic tail at μ_min in REAL_ELTYPES (MU1)" begin
        for R in REAL_ELTYPES
            prob = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2, x[2]^2 - 3))
            # `μ_min` below the floor `4 eps(R)` of the schedule, so the floor is what the damping
            # reaches and not the field.
            μ_bottom = 4 * eps(R)
            m = Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(R; μ₀ = 10 * eps(R),
                    μ_min = eps(R) / 16))
            x = SVector{2, R}(2, 2)
            Fx = prob.F(x, nothing)
            st = init_state(m, prob, x, nothing)
            st = start!!(st, m, prob, x, Fx, nothing, rnorm(Fx))
            fnorms = R[rnorm(Fx)]
            mus = R[]
            for _ in 1:30
                x, Fx, st, info = step!!(st, m, prob, x, Fx, nothing)
                push!(fnorms, info.fnorm)
                push!(mus, st.rule.μ)
                info.fnorm <= 100 * eps(R) && break
            end
            @test length(mus) >= 2
            # MU1: a damping that is not lowered after an accepted step stays at `μ₀`
            @test all(==(μ_bottom), mus[(end - 2):end])
            # the residual falls quadratically from the first step at the floor onwards
            for k in max(1, length(mus) - 2):(length(fnorms) - 1)
                fnorms[k] <= 100 * eps(R) && continue
                # the last step of a converged solve sits at the rounding level of `R`
                fnorms[k + 1] <= 100 * eps(R) && continue
                @test fnorms[k + 1] <= 2 * fnorms[k]^2
            end
        end
    end

    @testset "Check 2: rank-one consistent system in R1 and R3 (MU2)" begin
        prob_r1 = NonlinearProblem(
            (x, _) -> SVector(x[1] + x[2] - 2, 2 * (x[1] + x[2]) - 4),
            (_, _) -> @SMatrix [1.0 1.0; 2.0 2.0 + 1e-16])
        x_r1, st_r1 = solve(
            prob_r1, SVector(0.0, 0.0),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_r1.code == SUCCESS
        @test norm(x_r1 - SVector(1.0, 1.0)) <= 1e-6

        prob_r3 = NonlinearProblem(
            (F, x, _) -> (F[1] = x[1] + x[2] - 2; F[2] = 2 * (x[1] + x[2]) - 4; nothing),
            (J, _, _) -> (J[1, 1] = 1; J[1, 2] = 1; J[2, 1] = 2; J[2, 2] = 2 + 1e-16;
                nothing))
        x_r3, st_r3 = solve(
            prob_r3, [0.0, 0.0],
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_r3.code == SUCCESS
        @test norm(x_r3 - [1.0, 1.0]) <= 1e-6
    end

    @testset "Check 2b: a rank the factor drops is not stepped along (MU2)" begin
        # `J = diag(1, 1e-16)` is rank one at the cutoff `√n eps σ₁`, and the residual has a
        # component along the dropped left singular vector: the step must not move along it.
        J1 = @SMatrix [1.0 0.0; 0.0 1e-16]
        b1 = @SVector [1.0, 1.0]
        prob = NonlinearProblem((F, x, _) -> (F .= J1 * x .- b1; nothing),
            (J, _, _) -> (J .= J1; nothing))
        m = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-14, μ_min = 1e-14))
        solver = init(prob, [0.0, 0.0], m)
        prob.F(solver.state.Fx, solver.x, nothing)
        state = start!!(
            solver.state, solver.method, solver.prob, solver.x, solver.state.Fx, nothing,
            rnorm(solver.state.Fx))
        x1, _, _, info = step!!(
            state, solver.method, solver.prob, solver.x, solver.state.Fx, nothing)
        @test info.code == SUCCESS
        @test isapprox(x1[1], 1.0; atol = 1e-9)
        @test abs(x1[2]) <= 1e-6
    end

    @testset "Check 3: one Jacobian and one factorisation per step (MU3)" begin
        jac_calls = Ref(0)
        prob = NonlinearProblem(
            (x, _) -> norm(x) > 1.5 ? SVector(NaN, NaN) : SVector(x[1] - 1, x[2] - 1),
            (_, _) -> (jac_calls[] += 1; @SMatrix [0.01 0.0; 0.0 0.01]))
        m = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
        x = SVector(0.0, 0.0)
        Fx = prob.F(x, nothing)
        st = init_state(m, prob, x, nothing)
        st = start!!(st, m, prob, x, Fx, nothing, rnorm(Fx))
        _, _, _, info = step!!(st, m, prob, x, Fx, nothing)
        @test info.code == SUCCESS
        @test info.failures >= 2
        # MU3: a factorisation inside the trial loop evaluates the Jacobian once per trial
        @test jac_calls[] == 1
        # the accepted trial's damping had grown past `μ₀`, so the step is a limited one
        @test info.limited
    end

    @testset "Check 4: STALLED at μ_max and a non-finite trial (MU4, MU5)" begin
        prob = NonlinearProblem((x, _) -> SVector(x[1]^2 + 1, x[2]))
        m = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
        x0 = SVector(0.0, 0.0)
        x, st = solve(prob, x0, m)
        # MU4: another return code in place of `STALLED`
        @test st.code == STALLED
        @test x === x0
        @test st.gradratio == 0
        @test st.step_failures >= 25

        # A non-finite trial residual is a rejection, not an error: `F` is `NaN` above `x₁ = 5`,
        # and the first trials of `x₀ = (0.1, 0)` overshoot to `x₁ ≈ 16` from `μ₀ = 1e-2`.
        # MU5: returning `NONFINITE` at the first non-finite trial instead of rejecting it.
        function F_barrier(x, _)
            return x[1] > 5 ? SVector(NaN, NaN) : SVector(x[1]^2 - 4, x[2])
        end
        function J_barrier(x, _)
            return SMatrix{2, 2, Float64}(2 * x[1], 0.0, 0.0, 1.0)
        end
        prob_nan = NonlinearProblem(F_barrier, J_barrier)
        x0 = SVector(0.1, 0.0)
        m_nan = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-2))
        Fx0 = prob_nan.F(x0, nothing)
        st = init_state(m_nan, prob_nan, x0, nothing)
        st = start!!(st, m_nan, prob_nan, x0, Fx0, nothing, rnorm(Fx0))
        x1, _, _, info = step!!(st, m_nan, prob_nan, x0, Fx0, nothing)
        @test info.code == SUCCESS
        @test info.failures >= 1
        @test all(isfinite, x1)
        @test info.fnorm < rnorm(Fx0)
    end

    @testset "Check 5: the standard reference column" begin
        @test length(STANDARD_COLUMNS) == 10
        @test "Newton LevenbergMarquardt SVD" in first.(STANDARD_COLUMNS)
        lm_column = STANDARD_COLUMNS[findfirst(
            c -> first(c) == "Newton LevenbergMarquardt SVD", STANDARD_COLUMNS)]
        @test last(lm_column).linsolve isa SVDFactorization
        @test last(lm_column).globalization isa LevenbergMarquardt
    end

    @testset "Check 6a: stacked QR against the normal equations (MU6)" begin
        J_ill = @SMatrix [1.0 1.0; 1.0 1.0 + 1e-8]
        x_true = @SVector [1.0, -1.0]
        b_ill = J_ill * x_true
        prob = NonlinearProblem((x, _) -> J_ill * x - b_ill, (_, _) -> J_ill)
        # MU6: the normal equations lose this to a relative error of `κ ≈ 1e8`.
        @test norm((transpose(J_ill) * J_ill + 1e-16 * SMatrix{2, 2, Float64}(I)) \
                   (transpose(J_ill) * (-b_ill)) - x_true) > 1e-6
        for ls in (QRFactorization(), LUFactorization())
            x, st = solve(
                prob, SVector(0.0, 0.0),
                Newton(;
                    linsolve = ls,
                    globalization = LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-16)),
                nothing;
                options = Options(Float64; f_abstol = 1e-14, f_reltol = 0.0))
            @test st.code == SUCCESS
            @test norm(x - x_true) < 1e-6
        end
    end

    @testset "Check 6b: the factor's scale exponent (MU7)" begin
        for exponent_shift in (400, -400)
            s = ldexp(1.0, exponent_shift)
            J_ext = @SMatrix [s 0.0; 0.0 s]
            b_ext = J_ext * @SVector [1.0, -2.0]
            prob = NonlinearProblem((x, _) -> J_ext * x - b_ext, (_, _) -> J_ext)
            # MU7: dropping the factor's `2^-e` in the weight makes the step `2^400` too large at
            # the low end and `2^-400` too small at the high one.
            x, st = solve(
                prob, SVector(0.0, 0.0),
                Newton(;
                    linsolve = SVDFactorization(),
                    globalization = LevenbergMarquardt()),
                nothing;
                options = Options(Float64; f_abstol = s * 1e-12, f_reltol = 1e-12))
            @test st.code == SUCCESS
            @test norm(x - SVector(1.0, -2.0)) < 1e-6
            # One damped step of the same problem is `1/(1 + μ)` of the undamped one, whatever
            # `2^e` is: the step is scale-invariant.
            prob_r1 = NonlinearProblem((x, _) -> J_ext * x - b_ext, (_, _) -> J_ext)
            m = Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-6, μ_min = 1e-6))
            x0 = SVector(0.0, 0.0)
            Fx0 = prob_r1.F(x0, nothing)
            state = init_state(m, prob_r1, x0, nothing)
            state = start!!(state, m, prob_r1, x0, Fx0, nothing, rnorm(Fx0))
            x1, _, _, _ = step!!(state, m, prob_r1, x0, Fx0, nothing)
            @test isapprox(x1, SVector(1.0, -2.0) ./ (1 + 1e-6); rtol = 1e-8)
        end
    end

    @testset "Check 6c: fail-closed rule state (MU8)" begin
        prob = NonlinearProblem(
            (x, _) -> SVector(x[1] - 1, x[2] - 2),
            (_, _) -> SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0))
        m = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-6, μ_min = 5e-7,
                μ_decrease = 0.25))
        x0 = SVector(0.0, 0.0)
        Fx0 = prob.F(x0, nothing)
        st0 = init_state(m, prob, x0, nothing)
        @test isnan(st0.rule.μ)
        # MU8: a rule state initialised to `μ₀` would step here instead of failing closed
        x_bad, Fx_bad, st_bad, info_bad = step!!(st0, m, prob, x0, Fx0, nothing)
        @test info_bad.code == NONFINITE
        @test x_bad === x0
        @test Fx_bad === Fx0
        @test isnan(st_bad.rule.μ)

        st1 = start!!(st0, m, prob, x0, Fx0, nothing, rnorm(Fx0))
        @test st1.rule.μ === 1e-6
        x1, Fx1, st2, info1 = step!!(st1, m, prob, x0, Fx0, nothing)
        @test info1.code == SUCCESS
        @test st2.rule.μ === 5e-7
        # MU8: without the reset in `finish!!` the state keeps the last damping
        _, _, st_fin = finish!!(
            st2, m, prob, x1, Fx1,
            SolverStatus{Float64}(
                SUCCESS, Int32(1), info1.fnorm, info1.stepnorm, Int32(0), false))
        @test isnan(st_fin.rule.μ)

        # A residual that is zero already returns `SUCCESS` with no step, before any trial.
        Fx_zero = zero(x0)
        st_zero = start!!(st0, m, prob, x0, Fx_zero, nothing, rnorm(Fx_zero))
        x_z, _, _, info_z = step!!(st_zero, m, prob, x0, Fx_zero, nothing)
        @test info_z.code == SUCCESS
        @test info_z.stepnorm == 0
        @test x_z === x0
    end

    @testset "Check 6d: R1 and R3 storage, Float32/Float64, Array and JLArray" begin
        prob = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2, x[2]^2 - 3))
        m = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
        x0 = SVector(2.0, 2.0)
        Fx0 = prob.F(x0, nothing)
        st = init_state(m, prob, x0, nothing)
        @test isbits(st)
        st = @inferred start!!(st, m, prob, x0, Fx0, nothing, rnorm(Fx0))
        @inferred step!!(st, m, prob, x0, Fx0, nothing)
        if Base.JLOptions().check_bounds != 1
            step!!(st, m, prob, x0, Fx0, nothing)
            @test @allocated(step!!(st, m, prob, x0, Fx0, nothing)) == 0
        end
        @test_opt solve(prob, x0, m)
        @test_opt solve(NonlinearProblem((x, _) -> x^2 - 2, (x, _) -> 2x), 1.5, m)

        # A scalar iterate: the one-dimensional case of the same value-iterate path.
        prob_scalar = NonlinearProblem((x, _) -> x^2 - 2, (x, _) -> 2 * x)
        x_sc, st_sc = solve(prob_scalar, 1.5, m)
        @test st_sc.code == SUCCESS
        @test isapprox(x_sc, sqrt(2.0); atol = 1e-8)

        prob_r3 = NonlinearProblem((F, x, _) -> (F[1] = x[1]^2 - 2; F[2] = x[2]^2 - 3; F))
        for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
            solver = init(prob_r3, [2.0, 2.0], Newton(;
                linsolve = ls, globalization = LevenbergMarquardt()))
            _, st_r3 = solve!(solver)
            @test st_r3.code == SUCCESS
            solver.x .= [2.0, 2.0]
            prob_r3.F(solver.state.Fx, solver.x, nothing)
            solver.state = start!!(
                solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing, rnorm(solver.state.Fx))
            step!!(solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing)
            solver.x .= [2.0, 2.0]
            prob_r3.F(solver.state.Fx, solver.x, nothing)
            solver.state = start!!(
                solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing, rnorm(solver.state.Fx))
            @test @allocated(step!!(
                solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing)) == 0
        end

        # Float32 on R1 and R3, and the device path under `allowscalar(false)`.
        m32 = Newton(;
            linsolve = SVDFactorization(), globalization = LevenbergMarquardt(Float32))
        prob32 = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2.0f0, x[2]^2 - 3.0f0))
        x32, st32 = solve(prob32, SVector(2.0f0, 2.0f0), m32)
        @test st32.code == SUCCESS
        @test isapprox(x32, SVector(sqrt(2.0f0), sqrt(3.0f0)); atol = 1e-5)
        prob32_r3 = NonlinearProblem(
            (F, x, _) -> (F[1] = x[1]^2 - 2.0f0; F[2] = x[2]^2 - 3.0f0; F))
        x32b, st32b = solve(prob32_r3, Float32[2.0, 2.0], m32)
        @test st32b.code == SUCCESS
        @test isapprox(x32b, Float32[sqrt(2.0f0), sqrt(3.0f0)]; atol = 1e-5)

        prob_jl = NonlinearProblem((F, x, _) -> (F .= x .^ 2 .- 4.0; nothing))
        x_jl, st_jl = solve(prob_jl, JLArray([3.0, 3.0]), m)
        @test st_jl.code == SUCCESS
        @test isapprox(Array(x_jl), [2.0, 2.0]; atol = 1e-8)
    end

    @testset "Complex problems in ELTYPES" begin
        for T in filter(T -> T <: Complex, ELTYPES)
            A = SMatrix{2, 2, T}(T(2), T(1) + im, T(1) - im, T(3))
            b = SVector{2, T}(T(1), T(2))
            x_star = A \ b
            prob = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
            for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
                x, st = solve(
                    prob, zero(SVector{2, T}),
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(real(T))))
                @test st.code == SUCCESS
                @test isapprox(x, x_star; atol = sqrt(eps(real(T))))
            end
        end
    end

    @testset "Check 7: the Powell singular problem from a fixed start" begin
        function F_powell(x, _)
            return @SVector [x[1], 10 * x[1] / (x[1] + 0.1) + 2 * x[2]^2]
        end
        function J_powell(x, _)
            return @SMatrix [1.0 0.0;
                             1.0 / (x[1] + 0.1)^2 4 * x[2]]
        end
        prob = NonlinearProblem(F_powell, J_powell)
        opts = Options(Float64; f_abstol = 1e-10, f_reltol = 1e-10, max_iterations = 200)
        for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
            x, st = solve(
                prob, SVector(3.0, 1.0),
                Newton(; linsolve = ls, globalization = LevenbergMarquardt()),
                nothing; options = opts)
            @test st.code == SUCCESS
            @test st.fnorm <= 1e-9
            @test norm(x) < 1e-3
        end
    end
end
