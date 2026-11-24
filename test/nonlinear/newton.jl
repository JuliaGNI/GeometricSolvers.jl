# `Newton` in R1 — `SVector`, `MVector` and `Number` iterates — and its step on `Array` and
# `JLArray`: LU and QR linear solvers, analytic and AD Jacobians, diagonal regularisation, factor
# reuse, stall refresh, the residual norm a step carries into the state, `ToReal` adaptation, and
# quadratic convergence.

using Adapt: Adapt
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using KernelAbstractions: @index, @kernel, synchronize
using StaticArrays: MVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, DogLeg, LINESEARCH_FAILED, LUFactorization,
                        MAXITERS, NONFINITE, Newton, NoRefinement, NonlinearProblem,
                        Options, Picard, QRFactorization, SINGULAR, STALLED, SUCCESS,
                        SolverStatus, Static, StrongWolfe, ToReal, init_state, jacobian!!,
                        rnorm, solve, solve!!, step!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_NEWTON = 4

struct CallerTag end

caller_tag_scale(::Type{<:ForwardDiff.Dual{<:ForwardDiff.Tag{CallerTag}}}) = 2
caller_tag_scale(::Type{<:Complex{<:ForwardDiff.Dual{<:ForwardDiff.Tag{CallerTag}}}}) = 2
caller_tag_scale(::Type) = 1

@kernel function newton_singular_kernel!(codes, xs, prob, method, state₀, opt, @Const(x₀))
    i = @index(Global, Linear)
    @inbounds begin
        x, st, _ = solve!!(state₀, prob, x₀, method, nothing, opt)
        codes[i] = st.code
        xs[i] = x
    end
end

@kernel function newton_solve_kernel!(codes, xs, prob, method, state₀, opt, @Const(x₀),
        @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        x, st, _ = solve!!(state₀, prob, x₀, method, p, opt)
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

    # any value is a globalization, but a `DogLeg` takes no factor reuse
    @test Newton(; globalization = :anything).globalization === :anything
    @test Newton(; globalization = DogLeg()).reuse === nothing
    for reuse in (1, 3)
        err = try
            Newton(; globalization = DogLeg(), reuse)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("DogLeg", err.msg) && occursin("reuse", err.msg)
    end
end

# `adapt` rebuilds a method through the checked constructors of `Newton` and `Static`, so each
# host entry that adapts one rejects a value that `ToReal{Float32}` narrows out of range: a
# regularization of 1e300 to `Inf`, a `Static` step of 1e-50 to zero. Each call raises before any
# kernel launches.
@testset "every host entry rejects a method that ToReal narrows out of range" begin
    x₀ = SVector{2, Float32}(2, 2)
    X = [x₀, x₀]
    S = Vector{SolverStatus{Float32}}(undef, 2)
    prob = NonlinearProblem((y, _) -> y .* y .- one(eltype(y)))
    stat = StationarityProblem((x, _) -> (x[1]^2 + x[2]^2) / 2)
    rep = Reparametrized(prob, (y, _) -> y)
    for m in (Newton(; regularization = 1e300), Newton(; globalization = Static(1e-50)))
        @test_throws ArgumentError solve(prob, x₀, m)
        @test_throws ArgumentError solve_batch!(copy(X), prob, m)
        @test_throws ArgumentError init(stat, x₀, m)
        @test_throws ArgumentError solve(stat, x₀, m)
        @test_throws ArgumentError solve_batch!(copy(X), stat, m)
        @test_throws ArgumentError solve_batch!(copy(X), S, stat, m)
        @test_throws ArgumentError solve_batch!(copy(X), rep, m)
        @test_throws ArgumentError solve_batch!(copy(X), S, rep, m)
    end
end

# A composite method adapts its members through the same constructors, so a `Newton` or `Static`
# member that `ToReal{Float32}` narrows out of range is rejected inside `Picard`, `Repeat` and
# `Anderson` too.
@testset "a composite method rejects a member that ToReal narrows out of range" begin
    x₀ = SVector{2, Float32}(2, 2)
    prob = NonlinearProblem((y, _) -> y .* y .- one(eltype(y)))
    for m in (Picard(; globalization = Static(1e300)),
        Picard(; globalization = Static(1e-50)),
        Repeat(Newton(; regularization = 1e300), 2),
        Repeat(Newton(; globalization = Static(1e-50)), 2),
        Anderson(Newton(; regularization = 1e300)),
        Anderson(Newton(; globalization = Static(1e-50))))
        @test_throws ArgumentError solve(prob, x₀, m)
        @test_throws ArgumentError Adapt.adapt(ToReal{Float32}(), m)
    end
end

@testset "Newton state for each globalization and reuse" begin
    prob = NonlinearProblem((x, p) -> x * x - 4)
    st = init_state(Newton(), prob, 3.0, nothing)
    @test st.rule === nothing
    @test st.fac === nothing
    @test isbits(st)
    st_reuse = init_state(Newton(; reuse = 3), prob, 3.0, nothing)
    @test st_reuse.rule === GeometricSolvers.FactorAge(Int32(0), false)
    @test st_reuse.fac isa GeometricSolvers.StaticLU
    @test isbits(st_reuse)
    st_dl = init_state(Newton(; globalization = DogLeg()), prob, 3.0, nothing)
    @test st_dl.rule === GeometricSolvers.TrustRadius(0.0, 0.0)
    @test isbits(st_dl)
    @test typeof(st).name === typeof(st_reuse).name === typeof(st_dl).name
end

@testset "the Newton state of a value iterate is isbits and allocation-free, $T" for T in ELTYPES
    allocations(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D} = (f(a, b, c, d);
        @allocated f(a, b, c, d))
    prob = NonlinearProblem((y, q) -> y .* y .- q)
    for m in (Newton(), Newton(; reuse = 3), Newton(; globalization = DogLeg()),
        Newton(; globalization = Backtracking()))
        mT = Adapt.adapt(ToReal{real(T)}(), m)
        for x in (SVector{2, T}(3, 5), T(3))
            s = @inferred init_state(mT, prob, x, nothing)
            @test isbits(s)
            @test !(s isa NamedTuple)
            @test allocations(init_state, mT, prob, x, nothing) == 0
        end
    end
end

@testset "start!! of Newton resets its rule state" begin
    prob = NonlinearProblem((x, p) -> x * x - 4)
    GS = GeometricSolvers
    st = GS.init_state(Newton(; reuse = 3), prob, 3.0, nothing)
    aged = GS.NewtonState(st.jacprep, st.jvpprep, st.Fx, st.d, st.fac, st.J, st.g,
        GS.FactorAge(Int32(2), true), st.merit, st.fnorm, st.diag,
        st.refbuf, st.fac_T, st.promoted, st.slow_steps)
    @test GS.start!!(aged, Newton(; reuse = 3), prob, 3.0, 5.0, nothing, 5.0).rule ===
          GS.FactorAge(Int32(0), false)
    st_dl = GS.init_state(Newton(; globalization = DogLeg(; radius = 2.0)), prob, 3.0, nothing)
    @test st_dl.rule === GS.TrustRadius(2.0, 2.0)
    moved = GS.NewtonState(st_dl.jacprep, st_dl.jvpprep, st_dl.Fx, st_dl.d, st_dl.fac,
        st_dl.J, st_dl.g, GS.TrustRadius(8.0, 2.0), st_dl.merit, st_dl.fnorm, st_dl.diag,
        st_dl.refbuf, st_dl.fac_T, st_dl.promoted, st_dl.slow_steps)
    @test GS.start!!(moved, Newton(; globalization = DogLeg()), prob, 3.0, 5.0, nothing,
        5.0).rule === GS.TrustRadius(2.0, 2.0)
end

# The residual `y .* y .- q` of the norm test with its start, on one backend: the value iterate
# takes the out-of-place residual, and an array backend the in-place one that AD differentiates.
newton_norm_label(::Type{Nothing}) = "SVector"
newton_norm_label(::Type{B}) where {B} = string(nameof(B))
function newton_norm_case(::Type{Nothing})
    prob = NonlinearProblem((y, q) -> y .* y .- q)
    x₀, p = SVector(3.0, 5.0), SVector(4.0, 9.0)
    return prob, x₀, p, prob.F(x₀, p)
end
function newton_norm_case(::Type{B}) where {B}
    prob = NonlinearProblem((Fx, y, q) -> (Fx .= y .* y .- q; Fx))
    x₀, p = B([3.0, 5.0]), B([4.0, 9.0])
    return prob, x₀, p, prob.F(B(similar(x₀)), x₀, p)
end

# A step hands the state the norm of the residual it returns, and the next step reads it back:
# `grad_ratio` scales the gradient by it, and a mixed step takes it as `fnorm_old`, the scale of the
# linear-model error. The five rules cover the three step paths — no reuse, `DogLeg`, and a reused
# factor — at the working precision and mixed (`TF ≠ T`), because each path writes that norm into
# the state on its own.
@testset "every Newton step carries the norm of the residual it returns, $(newton_norm_label(AT))" for AT in (
    Nothing, ARRAY_BACKENDS...)
    GS = GeometricSolvers
    prob, x₀, p, F₀ = newton_norm_case(AT)
    for kw in ((;), (; globalization = Backtracking()), (; globalization = DogLeg()),
        (; reuse = 3), (; reuse = 3, refinement = Refinement()))
        for lin in (LUFactorization(), LUFactorization(Float32))
            m = Newton(; linsolve = lin, kw...)
            # An array backend writes the iterate and the residual in place, so each rule takes
            # its own copy of the start.
            x, F = copy(x₀), copy(F₀)
            st = GS.start!!(GS.init_state(m, prob, x, p), m, prob, x, F, p, rnorm(F))
            for _ in 1:3
                x, F, st, _ = GS.step!!(st, m, prob, x, F, p)
                @test st.fnorm == rnorm(F)
            end
        end
    end
end

# A step on a state that no `start!!` saw, or that `solve!!` returned, trusts nothing the state
# carries: a line search, whose merit at `α = 0` is the state's, fails closed with `NONFINITE` and
# keeps its iterate, and any other step takes `‖F‖` from the residual it is given and returns the
# Newton step `x - (x² - p) / 2x = (13/6, 17/5)`. The mixed step does what the working-precision
# step does.
@testset "a step without start!! is the same at either precision, $(newton_norm_label(AT))" for AT in (
    Nothing, ARRAY_BACKENDS...)
    GS = GeometricSolvers
    prob, x₀, p, F₀ = newton_norm_case(AT)
    for lin in (LUFactorization(), LUFactorization(Float32)),
        g in (Static(), Backtracking(), DogLeg())

        m = Newton(; linsolve = lin, globalization = g)
        x, F = copy(x₀), copy(F₀)
        st = GS.start!!(GS.init_state(m, prob, x, p), m, prob, x, F, p, rnorm(F))
        _, _, st_end = GS.solve!!(st, prob, x, m, p)
        # The state that `solve!!` returns carries the `NaN` norm of `init_state` again.
        @test isnan(st_end.fnorm)
        for st₀ in (GS.init_state(m, prob, copy(x₀), p), st_end)
            x, F = copy(x₀), copy(F₀)
            x₁, F₁, _, info = GS.step!!(st₀, m, prob, x, F, p)
            if g isa Backtracking
                @test info.code == NONFINITE
                @test Array(x₁) == Array(x₀)
            else
                @test info.code == SUCCESS
                @test isapprox(Array(x₁), [13 / 6, 17 / 5]; rtol = 4 * eps(Float32))
                @test info.fnorm ≈ rnorm(F₁)
            end
        end
    end
end

@testset "a singular step with reuse resets the age" begin
    GS = GeometricSolvers
    prob = NonlinearProblem((x, p) -> x * x - 4, (x, p) -> zero(x))
    m = Newton(; reuse = 3)
    st = GS.init_state(m, prob, 3.0, nothing)
    _, _, st₊, info = step!!(st, m, prob, 3.0, 5.0, nothing)
    @test info.code == SINGULAR
    @test st₊.rule === GS.FactorAge(Int32(0), false)
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
    for ls in (LUFactorization(), QRFactorization())
        x_nr, st_nr = solve(prob_num, T(3), Newton(; linsolve = ls, reuse = 2), T(4))
        @test st_nr.code == SUCCESS
        @test isapprox(x_nr, T(2); rtol = 10 * sqrt(eps(R)))
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
    st = init_state(m, prob, x₀, p)
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

    # A regularised step is not the Newton step, so the line search measures its slope:
    # F = x - p, J = I, μ = 1 gives d = -F / 2, and Armijo with c₁ = 1/2 accepts α = 1
    x₀_lin = SVector{2, T}(T(3), T(5))
    p_lin = SVector{2, T}(T(4), T(9))
    prob_lin = NonlinearProblem(
        (y, q) -> y .- q, (y, q) -> diagonal_matrix(SVector{2, T}(one(T), one(T))))
    m_lin = Newton(; regularization = 1, globalization = Backtracking(; c₁ = 0.5))
    x_lin₁, _ = solve(
        prob_lin, x₀_lin, m_lin, p_lin;
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    @test x_lin₁ === SVector{2, T}(T(7) / T(2), T(7))
    x_lin, st_lin = solve(prob_lin, x₀_lin, m_lin, p_lin)
    @test st_lin.code == SUCCESS
    @test isapprox(x_lin, p_lin; rtol = 10 * sqrt(eps(R)))
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
    # the square of `‖F(x₀)‖ ≈ 1.7e21` overflows, its norm does not: Newton converges near
    # `(2, 3)`, where the default residual test `‖F‖ ≤ √eps ‖F(x₀)‖` stops it
    x₀_ov = SVector{2, Float32}(3.0f0, 5.0f0)
    p_ov = SVector{2, Float32}(4.0f0, 9.0f0)
    prob_ov = NonlinearProblem((y, q) -> 1.0f20 .* (y .* y .- q))
    @test isinf(rnorm(prob_ov.F(x₀_ov, p_ov))^2)
    x_ov, st_ov = solve(prob_ov, x₀_ov, Newton(), p_ov)
    @test st_ov.code == SUCCESS
    @test x_ov ≈ SVector{2, Float32}(2, 3) rtol = sqrt(eps(Float32))

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

@testset "Residual evaluations of one globalized Newton step, $T" for T in ELTYPES
    R = real(T)
    x₀ = SVector{2, T}(T(3), T(5))
    p = SVector{2, T}(T(4), T(9))
    f_calls = Ref(0)
    prob = NonlinearProblem(
        (y, q) -> (f_calls[] += 1; y .* y .- q), (y, q) -> diagonal_matrix(y .+ y))
    opts = Options(
        T; max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    # F(x₀), and one F per trial: the accepted step reuses the residual of its trial. StrongWolfe
    # evaluates F again in φ′(1) for the curvature condition.
    for (glob, calls) in ((Static(), 2), (Backtracking(), 2), (StrongWolfe(), 3))
        f_calls[] = 0
        x₁, st = solve(prob, x₀, Newton(; globalization = glob), p; options = opts)
        @test st.iterations == 1
        @test x₁ != x₀
        @test f_calls[] == calls
    end
end

@testset "Newton reuse retries a failed line search from a fresh factor, $R" for R in REAL_ELTYPES
    p = SVector{2, R}(R(1) / R(2), R(1) / R(2))
    prob = NonlinearProblem((y, q) -> sin.(y) .- q)
    m_fresh = Newton(; globalization = Backtracking())
    m_reuse = Newton(; reuse = 3, globalization = Backtracking())

    # From (-4.8, 0.3) the third chord direction does not descend; the retry refactorises
    x_r, st_r = solve(prob, SVector{2, R}(R(-4.8), R(0.3)), m_reuse, p)
    @test st_r.code == SUCCESS
    @test all(isapprox.(sin.(x_r), R(1) / R(2); atol = 10 * sqrt(eps(R))))

    # Every start that Newton solves, Newton with reuse solves too
    lost = count(range(R(-6), R(6); length = 241)) do a
        x₀ = SVector{2, R}(a, R(0.3))
        solve(prob, x₀, m_fresh, p)[2].code == SUCCESS &&
            solve(prob, x₀, m_reuse, p)[2].code != SUCCESS
    end
    @test lost == 0
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
        m = Adapt.adapt(ToReal{R}(), Newton())
        opt = Options(T)
        x₀_s = SVector{2, T}(zero(T), zero(T))
        prob_s = NonlinearProblem((y, _) -> y .* y .- one(eltype(y)))
        codes_s = Vector{GeometricSolvers.ReturnCode}(undef, 2)
        xs_s = Vector{SVector{2, T}}(undef, 2)
        state₀_s = init_state(m, prob_s, x₀_s, nothing)
        newton_singular_kernel!(KA_BACKEND)(
            codes_s, xs_s, prob_s, m, state₀_s, opt, x₀_s; ndrange = 2)
        synchronize(KA_BACKEND)
        @test all(==(SINGULAR), codes_s)
        @test all(x -> x === x₀_s, xs_s)

        x₀_c = SVector{2, T}(T(3), T(5))
        p_c = SVector{2, T}(T(4), T(9))
        prob_c = NonlinearProblem((y, q) -> y .* y .- q)
        _, st_host = solve(prob_c, x₀_c, m, p_c)
        codes_c = Vector{GeometricSolvers.ReturnCode}(undef, 2)
        xs_c = Vector{SVector{2, T}}(undef, 2)
        state₀_c = init_state(m, prob_c, x₀_c, p_c)
        newton_solve_kernel!(KA_BACKEND)(
            codes_c, xs_c, prob_c, m, state₀_c, opt, x₀_c, p_c; ndrange = 2)
        synchronize(KA_BACKEND)
        @test all(==(st_host.code), codes_c)
        @test all(
            x -> isapprox(x, SVector{2, T}(T(2), T(3)); rtol = 10 * sqrt(eps(R))), xs_c)
    end
end

# ─── `Newton(; best = true)`: the iterate of the smallest ‖F‖₂ ─────────────────
#
# Each fixture runs on the four routes of an iterate: the value iterates `Number` and `SVector`,
# and the array iterates `Array` and `JLArray`, which refuses scalar indexing.

using LinearAlgebra: norm

best_iterate(::Type{Number}, v) = only(v)
best_iterate(::Type{SVector}, v) = SVector{length(v)}(v)
best_iterate(::Type{B}, v) where {B} = B(v)
best_problem(::Type{<:Union{Number, SVector}}, f, f!) = NonlinearProblem(f)
best_problem(::Type, f, f!) = NonlinearProblem(f!)
best_host(x::Number) = [x]
best_host(x) = Array(x)
const BEST_ROUTES = (Number, SVector, ARRAY_BACKENDS...)

# `x` and `y` agree to `8 eps`: the same arithmetic on two routes may differ in the last bits.
best_close(x, y) = all(isapprox.(x, y; rtol = 8 * eps(real(eltype(float(x)))), atol = 0))

# The no-root scalar `F(x) = x² + 1`: Newton from `x₀ = 0.5` jumps between iterates of growing and
# shrinking residual, and `‖F(x_k)‖₂` for `k = 0, …, 8` is `BEST_NR_F`, the smallest at `k = 6`.
best_nr(x, p) = x .* x .+ one(eltype(x))
best_nr!(y, x, p) = (y .= x .* x .+ one(eltype(x)); y)
best_nr_J(x::Number, p) = 2x
const BEST_NR_F = (
    1.25, 1.5625, 1.0850694444444444, 3.4600428713151907, 1.2166349630462578,
    1.7081737551644687, 1.0300641461766078, 8.823069005566088, 2.487724019230605)

# Every field of a status but `fnorm`.
function best_same_rest(a, b)
    a.code == b.code && a.iterations == b.iterations &&
        isequal(a.stepnorm, b.stepnorm) && a.step_failures == b.step_failures &&
        a.promoted == b.promoted && a.stationary == b.stationary &&
        a.f_calls == b.f_calls && a.jvp_calls == b.jvp_calls &&
        a.rank == b.rank && isequal(a.nullnorm, b.nullnorm) &&
        isequal(a.gradratio, b.gradratio)
end

# What `best = true` returns, from the solves of `best = false` capped at each `k`: the iterate of
# the smallest ‖F‖₂, the earliest among equal norms, where the uncapped solve ends without `SUCCESS`
# and that norm is below its last one; else the return of `best = false`.
function best_reference(prob, x₀, method, p, K; kwargs...)
    x_last, st_last = solve(prob, x₀, method, p; max_iterations = K, kwargs...)
    n = Int(st_last.iterations)
    xs = Vector{Vector}(undef, n + 1)
    fs = zeros(real(eltype(best_host(x₀))), n + 1)
    for k in 0:n
        x, st = solve(prob, x₀, method, p; max_iterations = k, kwargs...)
        xs[k + 1] = best_host(x)
        fs[k + 1] = st.fnorm
    end
    j = argmin(map(f -> isfinite(f) ? f : oftype(f, Inf), fs))
    restore = st_last.code != SUCCESS && isfinite(fs[j]) && fs[j] < st_last.fnorm
    return (restore ? (xs[j], fs[j]) : (best_host(x_last), st_last.fnorm)), st_last
end

function test_best(prob, x₀, method, best_method, p, K; kwargs...)
    (x_ref, f_ref), st_false = best_reference(prob, x₀, method, p, K; kwargs...)
    x, st = solve(prob, x₀, best_method, p; max_iterations = K, kwargs...)
    @test best_close(best_host(x), x_ref)
    @test best_close(st.fnorm, f_ref)
    @test best_same_rest(st, st_false)
    return x, st
end

@testset "Newton(; best): the keyword, the field and the constructors" begin
    @test Newton().best === Val(false)
    @test Newton(; best = true).best === Val(true)
    @test Newton(; best = false).best === Val(false)
    @test fieldnames(Newton)[end] === :best
    @test isbits(Newton(; best = true)) == isbits(Newton())
    # the type check of the keyword `best::Bool`
    @test_throws r"^TypeError: non-boolean \(Int64\) used in boolean context$" Newton(;
        best = 1)
    @test QuasiNewton(2; best = true).best === Val(true)
    @test GaussNewton(; best = true).best === Val(true)
    @test GaussNewton().best === Val(false)
    for glob in (Static(), Backtracking(), DogLeg(), LevenbergMarquardt())
        @test Newton(; globalization = glob, best = true).best === Val(true)
    end
    @test Newton(; linsolve = GMRES(), best = true).best === Val(true)
    @test Newton(; reuse = 2, best = true).best === Val(true)
    m = Newton(; best = true)
    m6 = Newton(m.linsolve, m.globalization, m.jacobian, m.refinement, m.regularization,
        m.reuse)
    @test m6.best === Val(false)
    m7 = Newton(m.linsolve, m.globalization, m.jacobian, m.refinement, m.regularization,
        m.reuse, m.best)
    @test m7 === m
    # `solve` adapts every method with `ToReal`, which rebuilds it field by field
    @test Adapt.adapt(ToReal{Float32}(), m).best === Val(true)
    @test Adapt.adapt(ToReal{Float32}(), Newton()).best === Val(false)
    # a one-variable least-squares problem rebuilds a callable-Jacobian method on its lifted path
    lifted = GeometricSolvers.lift_scalar_operator(GaussNewton(; jacobian = best_nr_J, best = true))
    @test lifted.best === Val(true)
end

@testset "best = true on the no-root scalar, $B" for B in BEST_ROUTES
    prob = best_problem(B, best_nr, best_nr!)
    x₀ = best_iterate(B, [0.5])
    x, st = test_best(prob, x₀, Newton(), Newton(; best = true), nothing, 8)
    @test st.code == MAXITERS
    @test st.iterations == 8
    @test best_close(st.fnorm, BEST_NR_F[7])
    x₆, _ = solve(prob, x₀, Newton(), nothing; max_iterations = 6)
    @test best_host(x) == best_host(x₆)
    test_best(prob, x₀, QuasiNewton(2), QuasiNewton(2; best = true), nothing, 8)
    test_best(prob, x₀, Newton(; globalization = Backtracking()),
        Newton(; globalization = Backtracking(), best = true), nothing, 8)
    # a solve that ends in SUCCESS restores nothing, whatever its sequence
    _, st_ok = solve(prob, x₀, Newton(; best = true), nothing; f_abstol = 1.3, f_reltol = 0)
    @test st_ok.code == SUCCESS && st_ok.iterations == 0 && st_ok.fnorm == 1.25
end

@testset "best = true with GMRES keeps the call counts" begin
    prob = NonlinearProblem(best_nr!)
    _, st = test_best(prob, [0.5], Newton(; linsolve = GMRES()),
        Newton(; linsolve = GMRES(), best = true), nothing, 8)
    @test st.f_calls > 0 && st.jvp_calls > 0
    @test best_close(st.fnorm, BEST_NR_F[7])
end

@testset "GaussNewton(; best = true) on a one-variable LeastSquaresProblem" begin
    prob = LeastSquaresProblem(best_nr, 1)
    # the AD Jacobian, and a callable one, with which the lifted path rebuilds the method
    for kw in ((;), (; jacobian = best_nr_J))
        x, st = test_best(
            prob, 0.5, GaussNewton(; kw...), GaussNewton(; kw..., best = true),
            nothing, 8)
        @test x isa Float64
        @test st.code == MAXITERS
        @test best_close(st.fnorm, BEST_NR_F[7])
    end
end

@testset "best = true in the combinators that wrap Newton, $B" for B in BEST_ROUTES
    prob = best_problem(B, best_nr, best_nr!)
    x₀ = best_iterate(B, [0.5])
    x_at(k) = best_host(first(solve(prob, x₀, Newton(), nothing; max_iterations = k)))
    # `Repeat` takes the best over its sub-steps: x₆ of eight
    x, st = solve(prob, x₀, Repeat(Newton(; best = true), 2), nothing; max_iterations = 4)
    @test st.code == MAXITERS && st.iterations == 4
    @test best_close(st.fnorm, BEST_NR_F[7])
    @test best_host(x) == x_at(6)
    # `Switch` restores its member's best in the member's `finish!!`: x₂ of four
    x, st = solve(prob, x₀, Switch(Newton(; best = true), Newton()), nothing;
        max_iterations = 4)
    @test st.code == MAXITERS
    @test best_close(st.fnorm, BEST_NR_F[3])
    @test best_host(x) == x_at(2)
    # `Fallback` solves each member from x₀: x₂ of the second member's four
    x, st = solve(prob, x₀, Fallback(Newton(), Newton(; best = true)), nothing;
        max_iterations = 4)
    @test st.code == MAXITERS && st.iterations == 8
    @test best_close(st.fnorm, BEST_NR_F[3])
    @test best_host(x) == x_at(2)
end

@testset "best = true with DogLeg and LevenbergMarquardt is best = false, $B" for B in BEST_ROUTES
    prob = best_problem(B, best_nr, best_nr!)
    x₀ = best_iterate(B, [0.5])
    ls = B === JLArray ? SVDFactorization() : LUFactorization()
    for glob in (DogLeg(), LevenbergMarquardt())
        x_f, st_f = solve(prob, x₀, Newton(; linsolve = ls, globalization = glob), nothing;
            max_iterations = 8)
        x_b, st_b = solve(
            prob, x₀, Newton(; linsolve = ls, globalization = glob, best = true),
            nothing; max_iterations = 8)
        @test best_host(x_b) == best_host(x_f)
        @test st_b === st_f
    end
end

# `F(x) = atan(x)` from `x₀ = 1.5`: the full Newton step overshoots, `|F|` grows until the
# Jacobian underflows and the solve ends `SINGULAR`, so the start is the best iterate.
best_atan(x, p) = atan.(x)
best_atan!(y, x, p) = (y .= atan.(x); y)

@testset "best = true restores the start: $B, $T" for B in BEST_ROUTES, T in REAL_ELTYPES

    prob = best_problem(B, best_atan, best_atan!)
    x₀ = best_iterate(B, [T(1.5)])
    x, st = test_best(prob, x₀, Newton(), Newton(; best = true), nothing, 100)
    @test st.code == SINGULAR
    @test best_host(x) == [T(1.5)]
    @test st.fnorm == atan(T(1.5))
end

# `F(x) = log(x)` for `x > 0` and `NaN` else: from 3 the first step lands left of 0, Newton keeps
# its iterate and ends `NONFINITE`; from -1 the start is not finite.
best_nan(x, p) = ifelse.(x .> 0, log.(abs.(x)), NaN .* x)
best_nan!(y, x, p) = (y .= ifelse.(x .> 0, log.(abs.(x)), NaN .* x); y)

@testset "best = true restores nothing on a non-finite residual, $B" for B in BEST_ROUTES
    prob = best_problem(B, best_nan, best_nan!)
    x, st = solve(prob, best_iterate(B, [3.0]), Newton(; best = true), nothing)
    @test st.code == NONFINITE && st.iterations == 1
    @test best_host(x) == [3.0] && st.fnorm == log(3.0)
    x, st = solve(prob, best_iterate(B, [-1.0]), Newton(; best = true), nothing)
    @test st.code == NONFINITE && st.iterations == 0
    @test best_host(x) == [-1.0] && isnan(st.fnorm)
end

# The chord fixture: `F(x) = x - c` with the constant Jacobian `(I - G)⁻¹`, so that `F(x_k) = Gᵏ e₀`
# with `e₀ = (1, 1)`: `‖F‖₂` is 1.4142, 0.8, 0.8485, 0.48 and `‖F‖∞` 1, 0.8, 0.6, 0.48. The max
# norm passes `f_abstol = 0.7` at `k = 2`, whose 2-norm lies above that of `k = 1`.
const BEST_CHORD_G = SMatrix{2, 2}(0.75, 0.75, 0.05, -0.75)
const BEST_CHORD_M = inv(SMatrix{2, 2, Float64}(I) - BEST_CHORD_G)
const BEST_CHORD_C = [1.0, 2.0]
best_chord(x, p) = x .- p
best_chord!(y, x, p) = (y .= x .- p; y)
best_chord_J(x, p) = BEST_CHORD_M
best_chord_J!(J, x, p) = (copyto!(J, Matrix(BEST_CHORD_M)); J)
best_chord_newton(::Type{SVector}; kwargs...) = Newton(; jacobian = best_chord_J, kwargs...)
best_chord_newton(::Type; kwargs...) = Newton(; jacobian = best_chord_J!, kwargs...)

@testset "best = true keeps a SUCCESS, $B" for B in (SVector, ARRAY_BACKENDS...)
    prob = best_problem(B, best_chord, best_chord!)
    x₀ = best_iterate(B, BEST_CHORD_C .+ 1)
    p = best_iterate(B, BEST_CHORD_C)
    x, st = solve(prob, x₀, best_chord_newton(B; best = true), p;
        f_abstol = 0.7, f_reltol = 0, norm = MaxNorm())
    @test st.code == SUCCESS
    @test st.iterations == 2
    @test abs(st.fnorm - norm([0.6, 0.6])) <= 1e-14
    @test all(abs.(best_host(x) .- BEST_CHORD_C .- [0.6, 0.6]) .<= 1e-14)
end

@testset "solve_batch! with best = true and MaxNorm, per system" begin
    X = [SVector{2}(BEST_CHORD_C .+ 1) for _ in 1:3]
    S = Vector{SolverStatus{Float64}}(undef, 3)
    m = best_chord_newton(SVector; best = true)
    solve_batch!(X, S, NonlinearProblem(best_chord), m, SVector{2}(BEST_CHORD_C);
        f_abstol = 0.7, f_reltol = 0, norm = MaxNorm())
    @test all(s -> s.code == SUCCESS && s.iterations == 2, S)
    @test all(x -> all(abs.(x .- BEST_CHORD_C .- 0.6) .<= 1e-14), X)
    # the no-root scalar: each system returns its best iterate
    Y = [SVector(0.5) for _ in 1:3]
    S₁ = Vector{SolverStatus{Float64}}(undef, 3)
    solve_batch!(
        Y, S₁, NonlinearProblem(best_nr), Newton(; best = true); max_iterations = 8)
    @test all(s -> s.code == MAXITERS && best_close(s.fnorm, BEST_NR_F[7]), S₁)
end

@testset "a state reused across solves restores nothing of an earlier solve" begin
    m = Newton(; best = true)
    opt₈ = Options(Float64; max_iterations = 8)
    opt₀ = Options(Float64; max_iterations = 0)
    prob = NonlinearProblem(best_nr)
    state = init_state(m, prob, SVector(0.5), nothing)
    _, st₁, state = solve!!(state, prob, SVector(0.5), m, nothing, opt₈)
    @test best_close(st₁.fnorm, BEST_NR_F[7])
    x₂, st₂, _ = solve!!(state, prob, SVector(3.0), m, nothing, opt₀)
    @test x₂ == SVector(3.0) && st₂.fnorm == 10.0 && st₂.code == MAXITERS
    prob! = NonlinearProblem(best_nr!)
    state = init_state(m, prob!, [0.5], nothing)
    _, st₁, state = solve!!(state, prob!, [0.5], m, nothing, opt₈)
    @test best_close(st₁.fnorm, BEST_NR_F[7])
    x₂, st₂, state = solve!!(state, prob!, [3.0], m, nothing, opt₀)
    @test x₂ == [3.0] && st₂.fnorm == 10.0 && st₂.code == MAXITERS
end

@testset "the residual buffer holds F of the restored iterate, $B" for B in ARRAY_BACKENDS
    m = Newton(; best = true)
    prob = NonlinearProblem(best_nr!)
    x₀ = B([0.5])
    state = init_state(m, prob, x₀, nothing)
    x, st, state = solve!!(
        state, prob, x₀, m, nothing, Options(Float64; max_iterations = 8))
    @test best_close(st.fnorm, BEST_NR_F[7])
    @test best_close(Array(GeometricSolvers.residual_buffer(state)), Array(x .* x .+ 1))
    # a restore of the start: the residual that `start!!` recorded
    prob_atan = NonlinearProblem(best_atan!)
    x₀ = B([1.5])
    state = init_state(m, prob_atan, x₀, nothing)
    x, st, state = solve!!(state, prob_atan, x₀, m, nothing, Options(Float64))
    @test st.code == SINGULAR && Array(x) == [1.5]
    @test Array(GeometricSolvers.residual_buffer(state)) == [atan(1.5)]
end

# A residual `F(x) = x` with a callable Jacobian that steps 3 → 1 → -1 → 2: the norms 3, 1, 1, 2
# tie at `k = 1` and `k = 2`, and the earliest of the two is restored.
best_tie_J(x::Number, p) = x == 3 ? 1.5 : x == 1 ? 0.5 : x == -1 ? 1 / 3 : 1.0
best_tie_J!(J, x, p) = (J .= best_tie_J.(x, p); J)

@testset "best = true restores the earliest of equal norms, $B" for B in BEST_ROUTES
    static = B in (Number, SVector)
    prob = static ? NonlinearProblem((x, p) -> x) :
           NonlinearProblem((y, x, p) -> (y .= x; y))
    jac = B === Number ? best_tie_J :
          B === SVector ? ((x, p) -> SMatrix{1, 1}(best_tie_J(x[1], p))) : best_tie_J!
    x₀ = best_iterate(B, [3.0])
    x, st = solve(
        prob, x₀, Newton(; jacobian = jac, best = true), nothing; max_iterations = 3)
    @test st.code == MAXITERS && st.iterations == 3
    @test st.fnorm == 1.0
    @test best_host(x) == [1.0]
end

@testset "dual inputs: best = true returns what best = false returns" begin
    prob = NonlinearProblem((x, p) -> x .* x .- p)
    x_b, st_b = solve(prob, 1.0, Newton(; best = true), ForwardDiff.Dual(4.0, 1.0))
    x_f, st_f = solve(prob, 1.0, Newton(), ForwardDiff.Dual(4.0, 1.0))
    @test st_b.code == SUCCESS && st_b === st_f
    @test x_b == x_f
    @test ForwardDiff.partials(x_b) == ForwardDiff.partials(x_f)
end

function best_solve_allocated(state::S, prob::P, x::X, m::M, opt::O) where {S, P, X, M, O}
    (
        solve!!(state, prob, x, m, nothing, opt);
        @allocated solve!!(state, prob, x, m, nothing, opt))
end

@testset "the R1 state of best = true is isbits, allocation-free and inferred" begin
    prob = NonlinearProblem(best_nr)
    x₀ = SVector(0.5)
    @test sizeof(init_state(Newton(), prob, x₀, nothing)) == 48
    for opt in (Options(Float64; max_iterations = 8),
        Options(Float64; max_iterations = 8, norm = MaxNorm()))
        m = Newton(; best = true)
        state = init_state(m, prob, x₀, nothing)
        @test isbits(state)
        _, st, _ = @inferred solve!!(state, prob, x₀, m, nothing, opt)
        @test best_close(st.fnorm, BEST_NR_F[7])
        if Base.JLOptions().check_bounds != 1
            @test best_solve_allocated(state, prob, x₀, m, opt) == 0
        end
    end
end

@testset "an array solve with best = true or MaxNorm allocates no more than best = false" begin
    prob = NonlinearProblem(best_nr!)
    function allocated(m, opt)
        state = init_state(m, prob, [0.5], nothing)
        return best_solve_allocated(state, prob, [0.5], m, opt)
    end
    opt = Options(Float64; max_iterations = 8)
    a_false = allocated(Newton(), opt)
    @test a_false <= 256
    @test allocated(Newton(; best = true), opt) <= a_false
    @test allocated(Newton(; best = true), Options(Float64; max_iterations = 8,
        norm = MaxNorm())) <= a_false
end
