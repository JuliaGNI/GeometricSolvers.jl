# `Newton` in R1: `SVector`, `MVector` and `Number` iterates, LU and QR linear solvers,
# analytic and AD Jacobians, diagonal regularisation, factor reuse, stall refresh,
# `ToReal` adaptation, and quadratic convergence.

using Adapt: Adapt
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using KernelAbstractions: @index, @kernel, synchronize
using StaticArrays: MVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, LINESEARCH_FAILED, LUFactorization,
                        MAXITERS, NONFINITE, Newton, NoRefinement, NonlinearProblem,
                        Options, Picard, QRFactorization, SINGULAR, STALLED, SUCCESS,
                        SolverStatus, Static, StrongWolfe, ToReal, init, jacobian!!, rnorm,
                        solve, solve!!, step!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_NEWTON = 4

struct CallerTag end

caller_tag_scale(::Type{<:ForwardDiff.Dual{<:ForwardDiff.Tag{CallerTag}}}) = 2
caller_tag_scale(::Type{<:Complex{<:ForwardDiff.Dual{<:ForwardDiff.Tag{CallerTag}}}}) = 2
caller_tag_scale(::Type) = 1

@kernel function newton_singular_kernel!(codes, xs, @Const(prob), @Const(x₀))
    i = @index(Global, Linear)
    @inbounds begin
        x, st = solve(prob, x₀, Newton())
        codes[i] = st.code
        xs[i] = x
    end
end

@kernel function newton_solve_kernel!(codes, xs, @Const(prob), @Const(x₀), @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        x, st = solve(prob, x₀, Newton(), p)
        codes[i] = st.code
        xs[i] = x
    end
end

@testset "Newton constructor, validation, and ToReal adaptation" begin
    m = Newton()
    @test m.linsolve === LUFactorization()
    @test m.globalization === Static()
    @test m.jacobian === nothing
    @test m.refinement === NoRefinement()
    @test m.regularization === 0.0
    @test m.reuse === nothing
    @test isbits(m)

    m_reg = Newton(; regularization = 0.5, reuse = 4)
    @test m_reg.regularization === 0.5
    @test m_reg.reuse === Int32(4)
    @test isbits(m_reg)

    a = Adapt.adapt(ToReal{Float32}(), m_reg)
    @test a.regularization === 0.5f0
    @test a.reuse === Int32(4)
    @test isbits(a)

    a_bt = Adapt.adapt(ToReal{Float32}(), Newton(; globalization = Backtracking()))
    @test a_bt.globalization isa Backtracking{Float32}
    @test a_bt.regularization === 0.0f0

    clos_j = let α = 0.5
        (y, q) -> diagonal_matrix(SVector(α * y[1], α * y[2]))
    end
    a_clos = Adapt.adapt(ToReal{Float32}(), Newton(; jacobian = clos_j))
    @test a_clos.regularization === 0.0f0
    @test eltype(a_clos.jacobian(SVector(1.0f0, 2.0f0), nothing)) === Float32

    for bad_reg in (-0.1, NaN, Inf, -Inf)
        @test_throws ArgumentError Newton(; regularization = bad_reg)
    end
    for bad_reuse in (0, -1, -5)
        @test_throws ArgumentError Newton(; reuse = bad_reuse)
    end
end

@testset "Newton solves coupled nonlinear systems on SVector, MVector, and Number, $T" for T in ELTYPES
    R = real(T)
    perm = r1_shift(T, N_NEWTON)
    x₀, p = r1_inputs(T, N_NEWTON)
    prob_ad = NonlinearProblem(R1Coupled(perm))
    prob_j = NonlinearProblem(R1Coupled(perm), (y, q) -> r1_coupled_jacobian(y, q, perm))

    for ls in (LUFactorization(), QRFactorization()),
        glob in (Static(), Backtracking(), Bisection(), StrongWolfe()),
        prob in (prob_ad, prob_j)
        m = Newton(; linsolve = ls, globalization = glob)
        x_sol, st = solve(prob, x₀, m, p)
        @test st.code == SUCCESS
        @test st.iterations >= 1
        @test x_sol != x₀
        @test rnorm(prob.F(x_sol, p)) <=
              max(Options(T).f_abstol, Options(T).f_reltol * rnorm(prob.F(x₀, p)))
    end

    # MVector iterate
    x_mv = MVector(x₀)
    x_msol, st_m = solve(prob_ad, x_mv, Newton(), p)
    @test st_m.code == SUCCESS
    @test rnorm(prob_ad.F(x_msol, p)) <=
          max(Options(T).f_abstol, Options(T).f_reltol * rnorm(prob_ad.F(x₀, p)))

    # Number iterate
    prob_num = NonlinearProblem((y, q) -> y * y - q, (y, q) -> y + y)
    for ls in (LUFactorization(), QRFactorization()), glob in (Static(), Backtracking())

        x_n, st_n = solve(
            prob_num, T(3), Newton(; linsolve = ls, globalization = glob), T(4))
        @test st_n.code == SUCCESS
        @test st_n isa SolverStatus{R}
        @test isapprox(x_n, T(2); rtol = 10 * sqrt(eps(R)))
    end
    x_n0, st_n0 = solve(prob_num, T(3), Newton(), T(4); f_reltol = zero(R))
    @test st_n0.code == SUCCESS
    @test st_n0 isa SolverStatus{R}
    @test abs(x_n0 - T(2)) <= 8 * eps(R)

    # Large iterate and residual norm x₀ = (2e4, 2e4), p = (1e8, 1e8)
    x_big = SVector{2, T}(T(2e4), T(2e4))
    p_big = SVector{2, T}(T(1e8), T(1e8))
    prob_sq = NonlinearProblem((y, q) -> y .* y .- q)
    x_bsol, st_b = solve(prob_sq, x_big, Newton(), p_big)
    @test st_b.code == SUCCESS
    @test isapprox(x_bsol, SVector{2, T}(T(1e4), T(1e4)); rtol = 10 * sqrt(eps(R)))
end

@testset "Newton quadratic convergence near a simple root, $T" for T in ELTYPES
    R = real(T)
    p = SVector{2, T}(T(2), T(3))
    x_star = SVector{2, T}(sqrt(T(2)), sqrt(T(3)))
    x₀ = x_star .+ SVector{2, T}(T(1) / T(4), -T(1) / T(5))
    prob = NonlinearProblem((y, q) -> y .* y .- q, (y, q) -> diagonal_matrix(y .+ y))
    m = Adapt.adapt(ToReal{R}(), Newton())
    st = init(m, prob, x₀, p)
    Fx = prob.F(x₀, p)
    x₁, Fx₁, st, _ = step!!(st, m, prob, x₀, Fx, p)
    x₂, _, st, _ = step!!(st, m, prob, x₁, Fx₁, p)
    e₀ = rnorm(x₀ .- x_star)
    e₁ = rnorm(x₁ .- x_star)
    e₂ = rnorm(x₂ .- x_star)
    @test e₁ < e₀ / 4
    @test e₂ <= 4 * e₁^2
end

@testset "Newton singular Jacobian and diagonal regularization, $T" for T in ELTYPES
    R = real(T)
    x₀ = SVector{2, T}(zero(T), zero(T))
    p = SVector{2, T}(one(T), one(T))
    prob = NonlinearProblem((y, q) -> y .* y .- q, (y, q) -> diagonal_matrix(y .+ y))

    # Singular start without regularization returns SINGULAR
    for m_s in (Newton(), Newton(; reuse = 2))
        x_sing, st_sing = solve(prob, x₀, m_s, p)
        @test st_sing.code == SINGULAR
        @test x_sing === x₀
        @test st_sing.iterations == 1
        @test st_sing.stepnorm == zero(R)
    end

    prob_s = NonlinearProblem((y, q) -> y * y - q, (y, q) -> y + y)
    xs_sing, sts_sing = solve(prob_s, zero(T), Newton(), one(T))
    @test sts_sing.code == SINGULAR
    @test xs_sing === zero(T)
    @test sts_sing.iterations == 1
    @test sts_sing.stepnorm == zero(R)

    # Regularised singular start: J(0) + 0.5 I = 0.5 I, so step 1 lands on (2, 2)
    x₁, st₁ = solve(
        prob, x₀, Newton(; regularization = 0.5), p;
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    @test x₁ === SVector{2, T}(T(2), T(2))
    @test st₁.iterations == 1

    x_sol, st_sol = solve(prob, x₀, Newton(; regularization = 0.5), p)
    @test st_sol.code == SUCCESS
    @test isapprox(x_sol, SVector{2, T}(one(T), one(T)); rtol = 10 * sqrt(eps(R)))

    xs₁, _ = solve(
        prob_s, zero(T), Newton(; regularization = 0.5), one(T);
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    @test xs₁ === T(2)
end

@testset "Newton non-finite step, trial point, and Float32 overflow" begin
    # Non-finite step from a regular Jacobian in Float32
    for m_nf in (Newton(), Newton(; globalization = Backtracking()))
        f_calls = Ref(0)
        x₀_nf = SVector{2, Float32}(0.0f0, 0.0f0)
        j_tiny = SMatrix{2, 2, Float32}(1.0f-30, 0.0f0, 0.0f0, 1.0f-30)
        prob_nf = NonlinearProblem(
            (y, _) -> (f_calls[] += 1; 1.0f-30 .* y .- 1.0f15),
            (y, _) -> j_tiny)
        x_nf, st_nf = solve(prob_nf, x₀_nf, m_nf)
        @test st_nf.code == NONFINITE
        @test x_nf === x₀_nf
        @test st_nf.iterations == 1
        @test f_calls[] == 1
    end

    # Non-finite trial point under Static
    for R in REAL_ELTYPES
        x₀_exp = SVector{2, R}(R(-50), R(-50))
        prob_exp = NonlinearProblem((y, _) -> exp.(y) .- one(R))
        x_exp, st_exp = solve(prob_exp, x₀_exp, Newton())
        @test st_exp.code == NONFINITE
        @test x_exp === x₀_exp
        @test st_exp.fnorm == rnorm(prob_exp.F(x₀_exp, nothing))
    end

    # Float32 overflow of residual norm
    x₀_ov = SVector{2, Float32}(3.0f0, 5.0f0)
    p_ov = SVector{2, Float32}(4.0f0, 9.0f0)
    prob_ov = NonlinearProblem((y, q) -> 1.0f20 .* (y .* y .- q))
    x_ov, st_ov = solve(prob_ov, x₀_ov, Newton(), p_ov)
    @test st_ov.code == NONFINITE
    @test x_ov === x₀_ov

    # Scale invariance in Float32
    iters_scale = Int32[]
    for s in (1.0f-8, 1.0f0, 1.0f8)
        prob_sc = NonlinearProblem((y, q) -> s .* (y .* y .- q))
        _, st_sc = solve(prob_sc, x₀_ov, Newton(), p_ov)
        @test st_sc.code == SUCCESS
        push!(iters_scale, st_sc.iterations)
    end
    @test all(==(iters_scale[1]), iters_scale)
end

@testset "Newton Jacobian selection, AD tag isolation, and reuse = 1, $T" for T in ELTYPES
    R = real(T)
    x₀ = SVector{2, T}(T(3), T(5))
    p = SVector{2, T}(T(4), T(9))
    sq = (y, q) -> y .* y .- q
    J_exact = (y, q) -> diagonal_matrix(y .+ y)
    prob_ad = NonlinearProblem(sq)

    # Bit-for-bit identical iterates across k = 0:6 for Newton(; jacobian = J), Newton(), and reuse = 1
    J_shifted = (y, q) -> diagonal_matrix((y .+ y) .* (one(R) + 4 * eps(R)))
    prob_shifted = NonlinearProblem(sq, J_shifted)
    @test jacobian!!(nothing, AutoForwardDiff(), prob_shifted, x₀, p) != J_shifted(x₀, p)
    for k in 0:6
        opts = Options(
            T; min_iterations = k, max_iterations = k,
            f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
        xk_sh_j, _ = solve(
            prob_shifted, x₀, Newton(; jacobian = J_shifted), p; options = opts)
        xk_sh, _ = solve(prob_shifted, x₀, Newton(), p; options = opts)
        xk_ad, _ = solve(prob_ad, x₀, Newton(), p; options = opts)
        xk_r1, _ = solve(prob_ad, x₀, Newton(; reuse = 1), p; options = opts)
        @test xk_sh_j === xk_sh
        @test xk_r1 === xk_ad
    end
    @test any(1:6) do k
        opts = Options(
            T; min_iterations = k, max_iterations = k,
            f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
        solve(prob_shifted, x₀, Newton(), p; options = opts)[1] !==
        solve(prob_ad, x₀, Newton(), p; options = opts)[1]
    end

    # reuse = 2 and reuse = 3 factorise every k-th non-stalled step
    reuse_calls = Ref(0)
    prob_rc = NonlinearProblem(sq, (y, q) -> (reuse_calls[] += 1; J_exact(y, q)))
    opts4 = Options(
        T; min_iterations = 4, max_iterations = 4,
        f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
    reuse_calls[] = 0
    solve(prob_rc, x₀, Newton(; reuse = 2), p; options = opts4)
    @test reuse_calls[] == 2
    reuse_calls[] = 0
    solve(prob_rc, x₀, Newton(; reuse = 3), p; options = opts4)
    @test reuse_calls[] == 2

    # Caller's AutoForwardDiff tag
    x₀_tag = SVector{2, T}(T(3), T(5))
    p_tag = SVector{2, T}(one(T), one(T))
    prob_tag = NonlinearProblem((x, q) -> x .* T(caller_tag_scale(eltype(x))) .- q)
    opts_tag = Options(
        T; max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    x_t1, _ = solve(
        prob_tag, x₀_tag,
        Newton(; jacobian = AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T))),
        p_tag; options = opts_tag)
    x_t2, _ = solve(prob_tag, x₀_tag, Newton(), p_tag; options = opts_tag)
    @test x_t1 === SVector{2, T}(T(2), T(3))
    @test x_t2 === SVector{2, T}(one(T), one(T))
end

@testset "Step-kind Jacobian call counts in one Backtracking step, $T" for T in ELTYPES
    R = real(T)
    x₀ = SVector{2, T}(T(3), T(5))
    p = SVector{2, T}(T(2), T(4))
    j_calls = Ref(0)
    J_count = (y, q) -> (j_calls[] += 1; diagonal_matrix(SVector{2, T}(T(2), T(2))))
    A_func = (y, q) -> diagonal_matrix(SVector{2, T}(T(2), T(2)))
    prob = NonlinearProblem((y, q) -> T(2) .* y .- q, J_count)
    opts = Options(
        T; max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    bt = Backtracking()

    # jacobian = nothing -> ExactStep -> J called 1 time
    j_calls[] = 0
    solve(prob, x₀, Newton(; globalization = bt), p; options = opts)
    @test j_calls[] == 1

    # reuse = 2 -> step 1 uses ExactStep (J called 1 time for factorisation),
    # step 2 reuses factor (fresh == false) and uses MeasuredSlope (J called 1 time by φ′)
    prob_reuse = NonlinearProblem(
        (y, q) -> y .* y .- q, (y, q) -> (j_calls[] += 1; diagonal_matrix(y .+ y)))
    opts2 = Options(
        T; min_iterations = 2, max_iterations = 2, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    j_calls[] = 0
    solve(prob_reuse, x₀, Newton(; reuse = 2, globalization = bt), p; options = opts)
    @test j_calls[] == 1
    j_calls[] = 0
    solve(prob_reuse, x₀, Newton(; reuse = 2, globalization = bt), p; options = opts2)
    @test j_calls[] == 2

    # jacobian = A_func -> MeasuredSlope -> J called 1 time (by LineFunction φ′)
    j_calls[] = 0
    solve(prob, x₀, Newton(; jacobian = A_func, globalization = bt), p; options = opts)
    @test j_calls[] == 1

    # jacobian = AutoForwardDiff() -> ExactStep -> J called 0 times
    j_calls[] = 0
    solve(
        prob, x₀, Newton(; jacobian = AutoForwardDiff(), globalization = bt), p;
        options = opts)
    @test j_calls[] == 0

    # linsolve = LUFactorization(Float32) on Float64 -> MeasuredSlope -> J called 2 times in Float64, 1 in Float32
    j_calls[] = 0
    solve(
        prob, x₀, Newton(; linsolve = LUFactorization(Float32), globalization = bt), p;
        options = opts)
    @test j_calls[] == (R === Float32 ? 1 : 2)

    # Picard(; globalization = Backtracking()) -> MeasuredSlope -> J called 1 time
    j_calls[] = 0
    solve(prob, x₀, Picard(; globalization = bt), p; options = opts)
    @test j_calls[] == 1
end

@testset "Newton failed line search, two stalled steps, and stall refresh, $T" for T in ELTYPES
    R = real(T)
    p4 = SVector{2, T}(T(4), T(4))
    prob_sq = NonlinearProblem((y, q) -> y .* y .- q)
    bt_strict = Backtracking(; c₁ = 0.99, maxiter = 1)

    # Failed line search with decreasing trial merit from (1, 1) accepts trial point
    x₀_1 = SVector{2, T}(one(T), one(T))
    x_ls1, st_ls1 = solve(
        prob_sq, x₀_1, Newton(; globalization = bt_strict), p4;
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    @test x_ls1 === SVector{2, T}(T(2.5), T(2.5))
    @test st_ls1.step_failures == 1
    @test st_ls1.iterations == 1
    @test st_ls1.code == MAXITERS

    # Failed line search with non-decreasing trial merit from (0.1, 0.1) returns LINESEARCH_FAILED
    x₀_01 = SVector{2, T}(T(1) / T(10), T(1) / T(10))
    x_ls2, st_ls2 = solve(prob_sq, x₀_01, Newton(; globalization = bt_strict), p4)
    @test st_ls2.code == LINESEARCH_FAILED
    @test x_ls2 === x₀_01
    @test st_ls2.iterations == 1
    @test st_ls2.step_failures == 1

    # Two stalled steps stop with STALLED
    x₀_st = SVector{2, T}(T(2), T(2))
    p_st = SVector{2, T}(one(T), one(T))
    big_j = diagonal_matrix(SVector{2, T}(T(R(1e20)), T(R(1e20))))
    prob_st = NonlinearProblem((y, q) -> y .- q, (y, q) -> big_j)
    x_st, st_st = solve(prob_st, x₀_st, Newton(; globalization = Backtracking()), p_st)
    @test st_st.code == STALLED
    @test st_st.iterations == 2
    @test x_st === x₀_st

    # Refresh after a stall under reuse = 4 and Backtracking
    stall_calls = Ref(0)
    id_j = diagonal_matrix(SVector{2, T}(one(T), one(T)))
    prob_sr = NonlinearProblem(
        (y, q) -> y .- q,
        (y, q) -> (stall_calls[] += 1; stall_calls[] == 1 ? big_j : id_j))
    x_sr, st_sr = solve(
        prob_sr, x₀_st, Newton(; reuse = 4, globalization = Backtracking()), p_st)
    @test st_sr.code == SUCCESS
    @test x_sr === p_st
    @test st_sr.iterations == 2
    @test stall_calls[] == 2
end

@testset "Newton with R1Closing (nested ForwardDiff tag isolation), $T" for T in ELTYPES
    R = real(T)
    x₀, p = r1_inputs(T, N_NEWTON)

    prob_close = NonlinearProblem(R1Closing())
    x_c, st_c = solve(
        prob_close, x₀, Newton(), p;
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    @test st_c.iterations == 1
    @test isapprox(x_c, x₀ ./ 2; rtol = 10 * eps(R))
end

@testset "Newton inside a KernelAbstractions kernel on CPU" begin
    for T in ELTYPES
        R = real(T)
        x₀_s = SVector{2, T}(zero(T), zero(T))
        prob_s = NonlinearProblem((y, _) -> y .* y .- one(eltype(y)))
        codes_s = Vector{GeometricSolvers.ReturnCode}(undef, 2)
        xs_s = Vector{SVector{2, T}}(undef, 2)
        newton_singular_kernel!(KA_BACKEND)(codes_s, xs_s, prob_s, x₀_s; ndrange = 2)
        synchronize(KA_BACKEND)
        @test all(==(SINGULAR), codes_s)
        @test all(x -> x === x₀_s, xs_s)

        x₀_c = SVector{2, T}(T(3), T(5))
        p_c = SVector{2, T}(T(4), T(9))
        prob_c = NonlinearProblem((y, q) -> y .* y .- q)
        _, st_host = solve(prob_c, x₀_c, Newton(), p_c)
        codes_c = Vector{GeometricSolvers.ReturnCode}(undef, 2)
        xs_c = Vector{SVector{2, T}}(undef, 2)
        newton_solve_kernel!(KA_BACKEND)(codes_c, xs_c, prob_c, x₀_c, p_c; ndrange = 2)
        synchronize(KA_BACKEND)
        @test all(==(st_host.code), codes_c)
        @test all(
            x -> isapprox(x, SVector{2, T}(T(2), T(3)); rtol = 10 * sqrt(eps(R))), xs_c)
    end
end
