# The line function and globalization bridge.
#
# `LineFunction` wraps `(prob, x, Fx, d, p, jvpprep)` and evaluates the merit
# `φ(α) = ‖F(x + αd, p)‖²` through `norm2` and its directional derivative
# `φ′(α) = 2 rdot(F(x + αd, p), J(x + αd, p)d)` through `jvp!!`. It is a callable struct (not
# `<: Function`), so `isbits(lf)` holds when its fields are `isbits`.

using Adapt: Adapt
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, ExactStep, LineFunction, LINESEARCH_FAILED,
                        MeasuredSlope, NONFINITE, Newton, NonlinearProblem, STALLED,
                        SUCCESS,
                        Static, StepInfo, StrongWolfe, ToReal, globalize!!, init, jvp!!,
                        norm2, prepare_ad, rdot, residual!!, rnorm, solve, update_iterate!!,
                        φ, φ′

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_LF = 4

@testset "LineFunction evaluates merit and directional derivative, $T" for T in ELTYPES
    R = real(T)
    perm = r1_shift(T, N_LF)
    x, p = r1_inputs(T, N_LF)
    d = -x ./ T(2)

    # Problem with analytic J = Diagonal(2x) on F(x, p) = x.^2 .- p
    sq_f(y, q) = y .^ 2 .- q
    sq_j(y, q) = diagonal_matrix(y .+ y)
    prob_sq = NonlinearProblem(sq_f, sq_j)
    st_sq = init(Newton(), prob_sq, x, p)
    lf_sq = LineFunction(prob_sq, x, sq_f(x, p), d, p, st_sq.jvpprep)
    @test !(lf_sq isa Function)
    @test isbits(lf_sq)

    e₁ = SVector{N_LF, T}(one(T), zero(T), zero(T), zero(T))
    lf_e₁ = LineFunction(prob_sq, x, sq_f(x, p), e₁, p, st_sq.jvpprep)
    for α in (zero(R), R(0.5), one(R))
        xα = update_iterate!!(nothing, x, α, d)
        Fα = sq_f(xα, p)
        Jd = sq_j(xα, p) * d
        if T <: Complex
            @test rdot(Fα, Jd) != real(sum(Fα .* Jd))
        end
        @test φ(lf_sq, α) == norm2(Fα)
        @test lf_sq(α) == norm2(Fα)
        if T <: Real
            @test φ′(lf_sq, α) == 2 * rdot(Fα, Jd)
        else
            @test isapprox(φ′(lf_sq, α), 2 * rdot(Fα, Jd); rtol = 8 * eps(R))
        end
        xα_e = update_iterate!!(nothing, x, α, e₁)
        @test φ′(lf_e₁, α) == 2 * rdot(sq_f(xα_e, p), sq_j(xα_e, p) * e₁)
    end

    # Member operator A != J: φ′ uses the problem's J, never the member's operator A
    id_op(y, q) = identity_matrix(T, N_LF)
    st_a = init(Newton(; jacobian = id_op), prob_sq, x, p)
    lf_a = LineFunction(prob_sq, x, sq_f(x, p), d, p, st_a.jvpprep)
    F₀ = sq_f(x, p)
    if T <: Real
        @test φ′(lf_a, zero(R)) == 2 * rdot(F₀, sq_j(x, p) * d)
    else
        @test isapprox(φ′(lf_a, zero(R)), 2 * rdot(F₀, sq_j(x, p) * d); rtol = 8 * eps(R))
    end
    @test φ′(lf_a, zero(R)) != 2 * rdot(F₀, id_op(x, p) * d)

    j_calls = Ref(0)
    aff_f(y, q) = T(2) .* y .- q
    aff_j(y, q) = (j_calls[] += 1; T(2) .* identity_matrix(T, 2))
    aff_a(y, q) = identity_matrix(T, 2)
    prob_aff = NonlinearProblem(aff_f, aff_j)
    solve(
        prob_aff, SVector{2, T}(T(3), T(5)),
        Newton(; jacobian = aff_a, globalization = Backtracking()),
        SVector{2, T}(T(2), T(2)); max_iterations = 1)
    @test j_calls[] == 1

    # AD-backed JVP on R1Coupled
    prob_ad = NonlinearProblem(R1Coupled(perm))
    jvpprep = prepare_ad(AutoForwardDiff(), prob_ad, nothing, x, p)
    lf_ad = LineFunction(prob_ad, x, prob_ad.F(x, p), d, p, jvpprep)
    @test !(lf_ad isa Function)
    @test isbits(lf_ad)

    for α in (zero(R), R(1) / R(4), R(1) / R(2), one(R))
        xα = update_iterate!!(nothing, x, α, d)
        Fα = prob_ad.F(xα, p)
        Jα = r1_coupled_jacobian(xα, p, perm)
        Jvα = jvp!!(nothing, jvpprep, prob_ad, xα, d, p)
        @test φ(lf_ad, α) === norm2(Fα)
        @test φ′(lf_ad, α) === 2 * rdot(Fα, Jvα)
        @test isapprox(φ′(lf_ad, α), 2 * rdot(Fα, Jα * d); rtol = 4 * eps(R))
    end

    # Scalar iterate (Number)
    prob_s = NonlinearProblem(R1ScalarScaled(), r1_scalar_scaled_jacobian)
    lf_s = LineFunction(prob_s, x[1], prob_s.F(x[1], p[1]), d[1], p[1], nothing)
    @test isbits(lf_s)
    αs = R(1) / R(2)
    xαs = update_iterate!!(nothing, x[1], αs, d[1])
    Fαs = prob_s.F(xαs, p[1])
    @test φ(lf_s, αs) === norm2(Fαs)
    @test φ′(lf_s, αs) === 2 * rdot(Fαs, p[1] * d[1])
end

@testset "globalize!! on Static, Backtracking, Bisection, StrongWolfe, $T" for T in ELTYPES
    R = real(T)
    inR(g) = Adapt.adapt(ToReal{R}(), g)
    x = SVector{2, T}(T(2), T(3))
    p = SVector{2, T}(one(T), one(T))
    prob = NonlinearProblem((y, q) -> y .* y .- q, (y, q) -> diagonal_matrix(y .+ y))
    Fx = prob.F(x, p)
    J = prob.J(x, p)
    d = -(J \ Fx)

    # Static takes x + ls.α * d without a line search
    for α in (one(R), R(1) / R(2))
        ls_st = Static(α)
        x₊, F₊, g₊, info = globalize!!(nothing, ls_st, prob, x, Fx, d, p, nothing)
        x_exp = update_iterate!!(nothing, x, α, d)
        F_exp = prob.F(x_exp, p)
        @test x₊ === x_exp
        @test F₊ === F_exp
        @test g₊ === nothing
        @test info ===
              StepInfo{R}(SUCCESS, rnorm(F_exp), abs(α) * rnorm(d), Int32(0), false)
    end

    # Static on non-finite trial residual keeps last finite iterate x and Fx
    prob_nf = NonlinearProblem((y, q) -> y[1] == x[1] ? y : SVector{2, T}(T(NaN), zero(T)))
    Fx_nf = prob_nf.F(x, p)
    x_nf, F_nf, _, info_nf = globalize!!(
        nothing, inR(Static()), prob_nf, x, Fx_nf, d, p, nothing)
    @test x_nf === x
    @test F_nf === Fx_nf
    @test info_nf === StepInfo{R}(NONFINITE, rnorm(Fx_nf), rnorm(d), Int32(0), false)

    # Backtracking, Bisection, StrongWolfe accept a step that decreases the merit
    for ls in (inR(Backtracking()), inR(Bisection()), inR(StrongWolfe()))
        x₊, F₊, _, info = globalize!!(nothing, ls, prob, x, Fx, d, p, nothing, ExactStep())
        @test info.code == SUCCESS
        @test F₊ === prob.F(x₊, p)
        @test info.fnorm == rnorm(F₊)
        @test info.fnorm < rnorm(Fx)
        @test info.stepnorm > 0
        @test info.failures == 0
        @test !info.promoted
        @test x₊ != x
    end

    # LINESEARCH_FAILED that still decreases φ₀ < φ(0) takes the step and counts 1 failure
    ls_strict = inR(Backtracking(; c₁ = 0.9999, maxiter = 1))
    x_f, F_f, _, info_f = globalize!!(
        nothing, ls_strict, prob, x, Fx, d, p, nothing, ExactStep())
    @test info_f.code == LINESEARCH_FAILED
    @test info_f.failures == Int32(1)
    @test info_f.fnorm < rnorm(Fx)
    @test x_f != x
    @test F_f === prob.F(x_f, p)

    # LINESEARCH_FAILED that does not decrease φ₀ keeps x and Fx with stepnorm = 0
    d_up = -d
    x_up, F_up, _, info_up = globalize!!(
        nothing, inR(Backtracking(; maxiter = 1)), prob, x, Fx, d_up, p, nothing, ExactStep())
    @test info_up.code == LINESEARCH_FAILED
    @test info_up.failures == Int32(1)
    @test x_up === x
    @test F_up === Fx
    @test info_up.fnorm === rnorm(Fx)
    @test iszero(info_up.stepnorm)
end
