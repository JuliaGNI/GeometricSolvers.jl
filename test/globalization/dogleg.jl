# `DogLeg`, the trust-region globalization of `Newton`: the method type and its adaptation, the
# initial radius, the three branches of the dogleg curve, acceptance and the trial loop, the radius
# update and its bounds, singular and degenerate Jacobians, scale invariance, convergence where a
# fixed step fails, the stop at a minimum of `‖F‖` that is not a root, and the kernel path.
#
# Most cases run one `step!!` on a problem with an analytic `J`, so that the step, the predicted
# reduction and the ratio are known in closed form. The one-dimensional problems `pw_F` take their
# residual at the start and at any trial point from the parameters, with `J = 1`, so a test sets
# the ratio `ρ = ared / pred` exactly: `pred` and `ared` are then small integers or exact binary
# fractions, and `ρ = 1/10` and `ρ = 3/4` are the correctly rounded `R(0.1)` and `R(0.75)`.

using Adapt: Adapt
using JET: JET, @test_opt
using KernelAbstractions: @index, @kernel, synchronize
using LinearAlgebra: dot
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: DogLeg, LINESEARCH_FAILED, LUFactorization, LineSearch, NONFINITE,
                        Newton, NonlinearProblem, QRFactorization, SINGULAR, STALLED,
                        SUCCESS, ToReal, factorize!!, init_state, ldiv!!, rnorm, solve,
                        step!!

include("../helpers/matrix.jl")

# On a Julia it does not support, JET loads empty stubs that throw. JET 0.12 exports
# `JET_AVAILABLE` to say so; JET 0.9 and 0.10 name the same flag `JET_LOADABLE`.
const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

# The Rosenbrock system, scaled by `s`: root `(1, 1)`.
rosen_F(x, s) = s * SVector(10 * (x[2] - x[1]^2), 1 - x[1])
rosen_J(x, s) = s * SMatrix{2, 2}(-20 * x[1], -one(x[1]), 10 * one(x[1]), zero(x[1]))

# The Dennis–Schnabel system, scaled by `s`: root `(1, 1)`. A fixed Newton step from `(2, 0.5)`
# ends `NONFINITE`.
ds_F(x, s) = s * SVector(x[1]^2 + x[2]^2 - 2, exp(x[1] - 1) + x[2]^3 - 2)
ds_J(x, s) = s * SMatrix{2, 2}(2 * x[1], exp(x[1] - 1), 2 * x[2], 3 * x[2]^2)

# `x³ − 2x + 2`, on which a fixed Newton step cycles `0 → 1 → 0` and whose root no path along which
# `|F|` decreases reaches from 0, and `atan`, on which a fixed step diverges from 10; both scaled by
# `s`.
cubic_F(x, s) = s * (x^3 - 2x + 2)
cubic_J(x, s) = s * (3x^2 - 2)
atan_F(x, s) = s * atan(x)
const CUBIC_ROOT = -1.7692923542386314

# A linear problem `F(x) = A x − b` with `p = (; A, b)`.
lin_F(x, p) = p.A * x - p.b
lin_J(x, p) = p.A

# The residual `p[1]` at `x = 0` and `p[2]` at any other point, with `J = 1`.
pw_F(x, p) = iszero(x) ? p[1] : p[2]
pw_J(x, p) = one(x)

function dogleg_newton(radius = nothing; kwargs...)
    Newton(; globalization = DogLeg(; radius), kwargs...)
end

# One step from `x₀` with a `Newton` adapted to the element type of `x₀`.
function one_step(prob, x₀, p, m)
    R = real(eltype(x₀))
    ma = Adapt.adapt(ToReal{R}(), m)
    st = init_state(ma, prob, x₀, p)
    Fx = prob.F(x₀, p)
    return step!!(st, ma, prob, x₀, Fx, p)
end

# A residual that counts its evaluations in `count`.
counted(f, count) = (x, p) -> (count[] += 1; f(x, p))

# The cosine of the angle between `a` and `b` in the real inner product.
cosine(a, b) = real(dot(a, b)) / (rnorm(a) * rnorm(b))

@testset "DogLeg construction, validation, and ToReal adaptation" begin
    d = DogLeg()
    @test d isa DogLeg{Float64}
    @test d.radius === nothing
    @test isbits(d)
    @test !(d isa LineSearch)

    d2 = DogLeg(; radius = 2)
    @test d2 isa DogLeg{Float64}
    @test d2.radius === 2.0
    @test isbits(d2)

    d32 = DogLeg(; radius = 2.0f0)
    @test d32 isa DogLeg{Float32}
    @test d32.radius === 2.0f0
    @test isbits(d32)

    # the radius is checked as given, before any conversion
    for bad in (0, 0.0, -1.0, NaN, Inf, -Inf, big(0.0), big(-1.0))
        @test_throws ArgumentError DogLeg(; radius = bad)
    end

    # and stored converted and clamped to `[10⁻⁶, 10¹⁰]`
    @test DogLeg(; radius = 1e12).radius === 1e10
    @test DogLeg(; radius = 1e-12).radius === 1e-6
    @test DogLeg(; radius = 1.0f-12).radius === Float32(1e-6)

    a = Adapt.adapt(ToReal{Float32}(), DogLeg(; radius = 2.0))
    @test a isa DogLeg{Float32}
    @test a.radius === 2.0f0
    a0 = Adapt.adapt(ToReal{Float32}(), DogLeg())
    @test a0 isa DogLeg{Float32}
    @test a0.radius === nothing
    # a radius that is positive and finite as given, but not in `Float32`, is clamped after the
    # conversion
    @test Adapt.adapt(ToReal{Float32}(), DogLeg(; radius = 1e50)).radius === 1.0f10
    @test Adapt.adapt(ToReal{Float32}(), DogLeg(; radius = 1e-50)).radius === Float32(1e-6)
    @test DogLeg{Float32}(1e50).radius === 1.0f10
    @test DogLeg{Float32}(1e-50).radius === Float32(1e-6)

    # a `BigFloat` radius gives a `DogLeg{BigFloat}`, which adapts to a bits type
    db = DogLeg(; radius = big(2.0))
    @test db isa DogLeg{BigFloat}
    @test db.radius == 2
    @test DogLeg(; radius = big(1e12)).radius == 1e10
    da = Adapt.adapt(ToReal{Float64}(), db)
    @test da isa DogLeg{Float64}
    @test da.radius === 2.0
    @test isbits(da)

    m = Adapt.adapt(ToReal{Float32}(), Newton(; globalization = DogLeg(; radius = 3)))
    @test m.globalization isa DogLeg{Float32}
    @test m.globalization.radius === 3.0f0
    @test isbits(m)
end

@testset "the initial radius, $R" for R in REAL_ELTYPES
    prob = NonlinearProblem(cubic_F)
    for (radius, expected) in ((nothing, zero(R)), (R(1e-12), R(1e-6)), (R(1e12), R(1e10)))
        m = Adapt.adapt(ToReal{R}(), dogleg_newton(radius))
        st = init_state(m, prob, R(3), one(R))
        @test st.rule isa GeometricSolvers.TrustRadius{R}
        @test st.rule.radius == expected
        @test st.rule.radius₀ == expected
        @test isbits(st)
    end

    # an unset radius starts at the first Newton step's norm: the step is the whole Newton step
    # `(1, 1)`, which lies on the boundary of that radius, and `ρ = 1` grows it by 4
    A = SMatrix{2, 2, R}(2, 0, 0, 4)
    p = (; A, b = SVector{2, R}(2, 4))
    x₀ = zero(SVector{2, R})
    x₊, _, st₊, info = one_step(NonlinearProblem(lin_F, lin_J), x₀, p, dogleg_newton())
    @test x₊ == SVector{2, R}(1, 1)
    @test info.code == SUCCESS
    @test info.failures == 0
    @test info.stepnorm ≈ sqrt(R(2)) rtol = 8 * eps(R)
    @test st₊.rule.radius ≈ 4 * sqrt(R(2)) rtol = 8 * eps(R)
    # the step keeps the initial radius, which the next solve starts from
    @test st₊.rule.radius₀ == 0

    # and that start is bounded at each end: a Newton step of 10¹² starts the radius at 10¹⁰, and
    # the step is truncated there; one of 10⁻⁹ starts it at 2 · 10⁻⁶, as Procedure 4.1 does, and
    # the step inside it keeps it
    one_d = NonlinearProblem(pw_F, pw_J)
    x₊, _, st₊, info = one_step(one_d, zero(R), (-R(1e12), zero(R)), dogleg_newton())
    @test x₊ == R(1e10)
    @test info.code == SUCCESS
    @test st₊.rule.radius == R(1e10)
    x₊, _, st₊, info = one_step(one_d, zero(R), (-R(1e-9), zero(R)), dogleg_newton())
    @test x₊ == R(1e-9)
    @test st₊.rule.radius == 2 * R(1e-6)

    # with no Newton step, a Cauchy point of 10⁻⁷ starts the radius at 2 · 10⁻⁶ by the same rule,
    # and the step is the Cauchy point inside it, here the root, not a step of the whole radius
    A₀ = SMatrix{2, 2, R}(1, 0, 0, 0)
    p₀ = (; A = A₀, b = SVector{2, R}(1e-7, 0))
    prob₀ = NonlinearProblem(lin_F, lin_J)
    x₊, F₊, st₊, info = one_step(prob₀, x₀, p₀, dogleg_newton())
    @test x₊ == SVector{2, R}(1e-7, 0)
    @test iszero(F₊)
    @test info.code == SUCCESS
    @test info.limited
    @test st₊.rule.radius == 2 * R(1e-6)
    x, st = solve(prob₀, x₀, dogleg_newton(), p₀)
    @test st.code == SUCCESS
    @test x == SVector{2, R}(1e-7, 0)
end

@testset "the three branches of the dogleg curve, $T" for T in ELTYPES
    R = real(T)
    c = T <: Complex ? T(im) : one(T)
    A = SMatrix{2, 2, T}(3, 0, c, 2)
    b = SVector{2, T}(3, 2)
    p = (; A, b)
    prob = NonlinearProblem(lin_F, lin_J)
    x₀ = zero(SVector{2, T})
    F₀ = lin_F(x₀, p)

    # the reference points, in a wider precision
    Ab, bb = big.(A), big.(b)
    sIN = T.(Ab \ bb)
    g = Ab' * (-bb)
    sCP = T.(-(rnorm(g)^2 / rnorm(Ab * g)^2) * g)
    nIN, nCP = rnorm(sIN), rnorm(sCP)
    @test nCP < nIN

    for (branch, δ) in ((:cauchy, nCP / 2), (:segment, (nCP + nIN) / 2), (:newton, 2nIN))
        x₊, _, st₊, info = one_step(prob, x₀, p, dogleg_newton(δ))
        s = x₊ - x₀
        @test info.code == SUCCESS
        @test info.stepnorm ≈ rnorm(s) rtol = 8 * eps(R)
        # every step reduces the model norm
        @test rnorm(F₀ + A * s) < rnorm(F₀)
        # the model is exact, so `ρ ≈ 1`: a step on the boundary grows the radius by 4, and the
        # Newton step inside it keeps it
        if branch === :cauchy
            @test rnorm(s) ≈ δ rtol = 8 * eps(R)
            @test 1 - cosine(s, T.(-g)) ≤ 8 * eps(R)
            @test st₊.rule.radius == 4δ
        elseif branch === :segment
            @test rnorm(s) ≈ δ rtol = 8 * eps(R)
            τ = real(dot(sIN - sCP, s - sCP)) / rnorm(sIN - sCP)^2
            @test 0 ≤ τ ≤ 1
            @test 1 - cosine(s - sCP, sIN - sCP) ≤ 64 * eps(R)
            @test st₊.rule.radius == 4δ
        else
            @test rnorm(A * x₊ - b) ≤ 8 * eps(R) * rnorm(b)
            @test st₊.rule.radius == δ
        end
    end
end

@testset "the segment point of a Newton step far outside the radius, $R" for R in REAL_ELTYPES
    # `A = diag(1, ε)`, `b = (1/2, 1)`: the Cauchy point has length about 1/2 and the Newton step
    # `(1/2, 1/ε)` has a finite norm, but the Newton step in units of a radius 0.51 does not: the
    # segment point still has norm 0.51
    ε = R === Float32 ? R(1e-19) : R(1e-154)
    p = (; A = SMatrix{2, 2, R}(1, 0, 0, ε), b = SVector{2, R}(0.5, 1))
    sIN = p.A \ p.b
    @test isfinite(rnorm(sIN))
    @test !isfinite(rnorm(sIN / R(0.51)))
    x₊, _, _, info = one_step(
        NonlinearProblem(lin_F, lin_J), zero(SVector{2, R}), p, dogleg_newton(0.51))
    @test info.code == SUCCESS
    @test rnorm(x₊) ≈ R(0.51) rtol = 8 * eps(R)
    @test x₊[1] > 0 && x₊[2] > 0
end

@testset "Jacobian evaluations equal Newton iterations, $R" for R in REAL_ELTYPES
    jcount, fcount = Ref(0), Ref(0)
    Jc = (x, s) -> (jcount[] += 1; rosen_J(x, s))
    prob = NonlinearProblem(counted(rosen_F, fcount), Jc)
    x, st = solve(prob, SVector{2, R}(-1.2, 1), dogleg_newton(), one(R))
    @test st.code == SUCCESS
    @test isapprox(x, SVector{2, R}(1, 1); rtol = sqrt(eps(R)))
    @test jcount[] == st.iterations
    # the first trial, the whole Newton step to `x₁ = 1`, raises `‖F‖` from 4.9 to 48.4 and is
    # rejected, so `F` is evaluated more often than once per iteration and once at the start
    @test fcount[] > st.iterations + 1
end

@testset "the acceptance test is unsquared, $R" for R in REAL_ELTYPES
    # `‖F‖ = 10`, the whole Newton step to `x = 10`, so `pred = 10` and the unsquared test
    # accepts `ared ≥ 10⁻³`. A trial with `ared = 2 · 10⁻³` is accepted at once.
    count = Ref(0)
    prob = NonlinearProblem(counted(pw_F, count), pw_J)
    x₊, _, _, info = one_step(prob, zero(R), (-R(10), -R(10) + R(2e-3)), dogleg_newton())
    @test info.code == SUCCESS
    @test x₊ == R(10)
    @test count[] == 2  # `F(x₀)` in `one_step`, and one trial

    # A trial with `ared = 8 · 10⁻⁴` fails the unsquared test, while a squared one —
    # `‖F‖² − ‖F₊‖² = 0.016 ≥ 10⁻⁴ pred² = 0.01`, or the same with `pred` squared as a
    # difference of squares — would accept it. It is rejected, the radius shrinks to 2.5, and
    # the next trial, at 2.5, is accepted.
    count[] = 0
    x₊, _, _, info = one_step(prob, zero(R), (-R(10), -R(10) + R(8e-4)), dogleg_newton())
    @test info.code == SUCCESS
    @test x₊ == R(2.5)
    @test count[] == 3

    # `ared == 10⁻⁴ pred` is accepted: `pred = 10⁴` and `ared = 1`, which equals
    # `R(1e-4) * 10⁴` in `Float64`
    if R === Float64
        @test R(1e-4) * R(10000) == 1
        x₊, _, _, info = one_step(prob, zero(R), (-R(10000), -R(9999)), dogleg_newton())
        @test info.code == SUCCESS
        @test x₊ == R(10000)
    end

    # a trial whose predicted reduction rounds to zero is rejected, though its `ared = 0` is not
    # below `10⁻⁴ pred = 0`: at the floor, the step fails
    F₀ = -R(1e12)
    @test -F₀ - abs(F₀ + R(1e-6)) == 0
    x₊, _, _, info = one_step(prob, zero(R), (F₀, F₀), dogleg_newton(1e-6))
    @test info.code == LINESEARCH_FAILED
    @test x₊ == 0
end

@testset "the radius floor ends the trial loop, $R" for R in REAL_ELTYPES
    # the residual grows at every trial point, so every trial is rejected
    count = Ref(0)
    prob = NonlinearProblem(counted(pw_F, count), pw_J)
    x₀ = zero(R)
    F₀ = -one(R)
    m = Adapt.adapt(ToReal{R}(), dogleg_newton(1e-6))
    st = init_state(m, prob, x₀, (F₀, -R(2)))
    count[] = 0
    x₊, F₊, st₊, info = step!!(st, m, prob, x₀, F₀, (F₀, -R(2)))
    @test info.code == LINESEARCH_FAILED
    @test info.failures == 1
    @test info.stepnorm == 0
    @test x₊ === x₀
    @test F₊ === F₀
    @test count[] == 1
    @test st₊.rule.radius == R(1e-6)
end

@testset "a rejected trial shrinks the radius by 0.25, also inside it, $R" for R in REAL_ELTYPES
    # the Newton step is 10 and `F` is worse at every `x ≥ 5`: the trials at 10 (radius 80), 10
    # (radius 20) and 5 (radius 5, the Cauchy step) are rejected, and the trial at 1.25 is
    # accepted with `ρ = 1` on the boundary. A rejection that shrank to `‖s^IN‖ / 4` would accept
    # 2.5 instead.
    count = Ref(0)
    F5 = counted((x, p) -> iszero(x) ? -R(10) : (x >= 5 ? -R(10.5) : x - 10), count)
    x₊, _, st₊, info = one_step(NonlinearProblem(F5, pw_J), zero(R), nothing, dogleg_newton(80))
    @test info.code == SUCCESS
    @test x₊ == R(1.25)
    @test count[] == 5  # `F(x₀)`, three rejected trials and the accepted one
    @test st₊.rule.radius == 5
end

@testset "the model is the regularised Jacobian that was factorised, $R" for R in REAL_ELTYPES
    # `J = 1` and `μ = 1`: the model matrix is `J + μ = 2`, the Newton step of `x − 10` from 0 is
    # 5, and `pred = 10 − |−10 + 2 · 5| = 10`, so `ρ = 5/10` keeps the radius 5. The unregularised
    # model has `pred = 5` and `ρ = 1`, and would grow it to 20.
    prob = NonlinearProblem((x, p) -> x - 10, pw_J)
    x₊, _, st₊, info = one_step(prob, zero(R), nothing, dogleg_newton(5; regularization = 1))
    @test info.code == SUCCESS
    @test x₊ == 5
    @test st₊.rule.radius == 5
end

@testset "an exact root before min_iterations, $R" for R in REAL_ELTYPES
    # at `F(x) == 0` the step keeps the iterate and succeeds, as a fixed step does
    for (F, x₀, k) in (((x, p) -> x * x - 4, R(2), 1), ((x, p) -> 2x - 4, zero(R), 3))
        prob = NonlinearProblem(F)
        x, st = solve(prob, x₀, dogleg_newton(); min_iterations = k)
        @test st.code == SUCCESS
        @test x == 2
        @test st.iterations == k
        _, sts = solve(prob, x₀, Newton(); min_iterations = k)
        @test sts.code == SUCCESS
        @test sts.iterations == k
    end

    # one step at the root: no trial, the iterate, residual and radius stay, and `J` is
    # evaluated once
    fcount, jcount = Ref(0), Ref(0)
    prob = NonlinearProblem(
        counted((x, p) -> x * x - 4, fcount), (x, p) -> (jcount[] += 1; 2x))
    x₊, F₊, st₊, info = one_step(prob, R(2), nothing, dogleg_newton(3))
    @test info.code == SUCCESS
    @test info.stepnorm == 0
    @test info.failures == 0
    @test x₊ === R(2)
    @test F₊ === zero(R)
    @test st₊.rule.radius == 3
    @test fcount[] == 1  # `F(x₀)` in `one_step`
    @test jcount[] == 1
end

@testset "the radius update: shrink, grow, keep, $R" for R in REAL_ELTYPES
    prob = NonlinearProblem(pw_F, pw_J)
    F₀ = -R(10)  # the Newton step is 10, `‖s^IN‖ = 10`
    cases = (
        # δ, F at the trial, the new radius
        (R(4), -R(9.9), R(1)),      # ‖s^IN‖ ≥ δ, ρ = 0.025 < 0.1: 0.25 δ
        (R(4), -R(6.5), R(16)),     # on the boundary, ρ = 0.875 > 0.75: 4 δ
        (R(10), -R(9), R(10)),      # the Newton step on the boundary, ρ = 1/10: kept
        (R(10), -R(2.5), R(10)),    # the Newton step on the boundary, ρ = 3/4: kept
        (R(10), -R(9.9), R(2.5)),   # the Newton step on the boundary, ρ = 0.01 < 0.1: 0.25 δ
        (R(80), -R(9.9), R(10)),    # ‖s^IN‖ < δ, ρ = 0.01 < 0.1: max(‖s^IN‖, 10⁻⁶), not 0.25 δ
        (R(40), -R(1), R(40)))      # ‖s^IN‖ < δ, ρ = 0.9 > 0.75, inside: kept
    for (δ, Fs, expected) in cases
        x₊, _, st₊, info = one_step(prob, zero(R), (F₀, Fs), dogleg_newton(δ))
        @test info.code == SUCCESS
        @test x₊ == min(δ, R(10))
        @test st₊.rule.radius == expected
    end
end

@testset "the radius bounds, $R" for R in REAL_ELTYPES
    prob = NonlinearProblem(pw_F, pw_J)
    # at 10¹⁰, a truncated step with ρ = 1 does not grow the radius
    x₊, _, st₊, info = one_step(prob, zero(R), (-R(2e10), -R(1e10)), dogleg_newton(1e10))
    @test info.code == SUCCESS
    @test x₊ == R(1e10)
    @test st₊.rule.radius == R(1e10)
    # at 10⁻⁶, a step with ρ ≈ 0.05 does not shrink it
    F₀ = -R(1e-5)
    x₊, _, st₊, info = one_step(
        prob, zero(R), (F₀, F₀ + R(1e-6) / 20), dogleg_newton(1e-6))
    @test info.code == SUCCESS
    @test x₊ == R(1e-6)
    @test st₊.rule.radius == R(1e-6)
    # and a Newton step of 10⁻⁹ inside it, with ρ ≈ 0.05, leaves it at 10⁻⁶, not at 10⁻⁹
    F₀ = -R(1e-9)
    x₊, _, st₊, info = one_step(
        prob, zero(R), (F₀, F₀ + R(1e-9) / 20), dogleg_newton(1e-6))
    @test info.code == SUCCESS
    @test x₊ == R(1e-9)
    @test st₊.rule.radius == R(1e-6)
end

@testset "a singular Jacobian at the start takes the Cauchy step, $R" for R in REAL_ELTYPES
    J = SMatrix{2, 2, R}(1, 1, 1, 1)
    p = (; A = J, b = SVector{2, R}(0, 1))
    prob = NonlinearProblem(lin_F, lin_J)
    x₀ = zero(SVector{2, R})
    g = J' * lin_F(x₀, p)
    @test g == SVector{2, R}(-1, -1)
    x₊, _, st₊, info = one_step(prob, x₀, p, dogleg_newton())
    s = x₊ - x₀
    @test info.code != SINGULAR
    @test info.code == SUCCESS
    @test 1 - cosine(s, -g) ≤ 8 * eps(R)
    # the start of the radius is the length of the Cauchy point, √2/4, and the step has that length;
    # the model is exact, so `ρ = 1` on the boundary grows the radius by 4
    @test rnorm(s) ≈ sqrt(R(2)) / 4 rtol = 8 * eps(R)
    @test isfinite(st₊.rule.radius)
    @test st₊.rule.radius ≈ sqrt(R(2)) rtol = 8 * eps(R)

    # with no Newton step, a radius beyond the Cauchy point does not lengthen the step: the step is
    # the Cauchy point `(1/4, 1/4)`, and `ρ < 0.1` shrinks the radius by 0.25
    δ = R(1) / 2
    s = SVector{2, R}(1, 1) / 4
    F₀ = lin_F(x₀, p)
    pred = rnorm(F₀) - rnorm(F₀ + J * s)
    sing = NonlinearProblem(pw_F, (x, q) -> J)
    x₊, _, st₊, info = one_step(
        sing, x₀, (F₀, SVector{2, R}(0, -(1 - pred / 20))), dogleg_newton(δ))
    @test info.code == SUCCESS
    @test x₊ ≈ s rtol = 8 * eps(R)
    @test st₊.rule.radius == δ / 4
end

@testset "a zero steepest-descent direction stalls, $R" for R in REAL_ELTYPES
    prob = NonlinearProblem((x, p) -> x * x + 1)
    x, st = solve(prob, zero(R), dogleg_newton())
    @test st.code == STALLED
    @test st.code != SINGULAR
    @test x == 0

    # one step: the iterate stays, the step is zero, and the radius stays unset
    x₊, _, st₊, info = one_step(prob, zero(R), nothing, dogleg_newton())
    @test info.code == STALLED
    @test info.stepnorm == 0
    @test x₊ == 0
    @test st₊.rule.radius == 0

    # the direction `g̃ = J' (F / ‖F‖)` and its image `J ĝ` underflow: a `g̃` whose norm is zero is
    # a zero direction, and so is a `g̃` whose squared norm is a subnormal and whose `‖J ĝ‖` is zero
    j₀, j₁ = R === Float32 ? (R(1e-30), R(3e-23)) : (R(1e-170), R(1.6e-162))
    @test iszero(rnorm(j₀))
    @test !iszero(rnorm(j₁))
    @test iszero(rnorm(j₁ * (j₁ / rnorm(j₁))))
    for j in (j₀, j₁)
        lin1 = NonlinearProblem((x, q) -> 1 + j * x, (x, q) -> j)
        x₊, _, _, info = one_step(lin1, zero(R), nothing, dogleg_newton())
        @test info.code == STALLED
        @test info.stepnorm == 0
        @test x₊ == 0
    end

    # a `g̃` whose norm overflows gives no Cauchy direction either: a singular `J` near `floatmax`
    # stalls, and the trial loop does not run on a radius that is not a number
    big_j = R === Float32 ? R(1e30) : R(1e300)
    J = SMatrix{2, 2, R}(big_j, big_j, big_j, big_j)
    p = (; A = J, b = SVector{2, R}(-1e10, 0))
    x₀ = zero(SVector{2, R})
    F₀ = lin_F(x₀, p)
    @test !isfinite(rnorm(J' * (F₀ / rnorm(F₀))))
    x₊, _, st₊, info = one_step(NonlinearProblem(lin_F, lin_J), x₀, p, dogleg_newton())
    @test info.code == STALLED
    @test x₊ == x₀
    @test st₊.rule.radius == 0

    # and so does a `J ĝ` whose norm overflows while the norm of `g̃` is finite: with
    # `J = diag(d, 0)` and `F = (−1, −1)`, `‖g̃‖ = d/√2` and `‖J ĝ‖ = d`, and the Cauchy length would
    # be zero, so that every trial would be a zero step
    d = R === Float32 ? R(2e19) : R(1.6e154)
    J = SMatrix{2, 2, R}(d, 0, 0, 0)
    p = (; A = J, b = SVector{2, R}(1, 1))
    F₀ = lin_F(x₀, p)
    g̃ = J' * (F₀ / rnorm(F₀))
    @test isfinite(rnorm(g̃))
    @test !isfinite(rnorm(J * (g̃ / rnorm(g̃))))
    x₊, _, st₊, info = one_step(NonlinearProblem(lin_F, lin_J), x₀, p, dogleg_newton())
    @test info.code == STALLED
    @test x₊ == x₀
    @test st₊.rule.radius == 0
end

@testset "a non-finite Jacobian, $R" for R in REAL_ELTYPES
    # a `NaN` or `Inf` in `J` makes `g̃` not finite, so the Cauchy point is undefined: the solve
    # ends with the iterate kept, and nothing throws
    for bad in (R(NaN), R(Inf))
        scalar = (NonlinearProblem((x, p) -> x * x - 4, (x, p) -> bad), R(3))
        system = (
            NonlinearProblem((x, p) -> SVector(x[1] - 1, x[2] - 2),
                (x, p) -> SMatrix{2, 2, R}(bad, 0, 0, 1)),
            zero(SVector{2, R}))
        for (prob, x₀) in (scalar, system)
            x, st = solve(prob, x₀, dogleg_newton())
            @test x == x₀
            @test st.code ∉ (SUCCESS, SINGULAR, NONFINITE)
            isnan(bad) && @test st.code == STALLED
        end
    end
end

@testset "a non-finite Newton step and a non-finite trial, $R" for R in REAL_ELTYPES
    # an upper-triangular `J` whose last pivot is the smallest subnormal: the factor has
    # `info == 0`, and back substitution gives infinite last two entries of opposite sign and
    # `Inf − Inf = NaN` in the first
    t = nextfloat(zero(R))
    J = SMatrix{3, 3, R}(1, 0, 0, 1, 1, 0, 1, 1, t)
    p = (; A = J, b = -SVector{3, R}(1, 1, 1))
    fac = factorize!!(nothing, LUFactorization(), J)
    @test fac.info == 0
    @test any(isnan, ldiv!!(nothing, fac, SVector{3, R}(1, 1, 1)))
    prob = NonlinearProblem(lin_F, lin_J)
    x₀ = zero(SVector{3, R})
    g = J' * lin_F(x₀, p)
    x₊, _, st₊, info = one_step(prob, x₀, p, dogleg_newton())
    s = x₊ - x₀
    @test info.code == SUCCESS
    @test 1 - cosine(s, -g) ≤ 8 * eps(R)
    # the radius starts at the Cauchy length `‖g‖³ / ‖J g‖² = 27/41`, and grows by 4
    @test rnorm(s) ≈ R(27) / 41 rtol = 8 * eps(R)
    @test st₊.rule.radius ≈ 4 * R(27) / 41 rtol = 8 * eps(R)

    # `F` is `Inf` beyond 2: the first trial, at 4, is rejected; the radius shrinks to 1 and
    # the next trial, on the same curve, is accepted
    count = Ref(0)
    inf_F = counted((x, p) -> x > 2 ? R(Inf) : x - 10, count)
    prob_inf = NonlinearProblem(inf_F, pw_J)
    x₊, F₊, st₊, info = one_step(prob_inf, zero(R), nothing, dogleg_newton(4))
    @test info.code == SUCCESS
    @test info.failures == 0
    @test x₊ == 1
    @test F₊ == -9
    @test count[] == 3  # `F(x₀)`, the rejected trial and the accepted one
    @test st₊.rule.radius == 4
end

@testset "scaled residuals in Float32" begin
    R = Float32
    # the two systems converge, and the cubic stops at the minimum of `|F|`, at every scale
    # `10^k` of `F` and `J`, `k ∈ {−16, −12, −8, 8, 12, 16}`, with the code and the iteration count
    # of the unscaled problem
    scales = (1, 1e-16, 1e-12, 1e-8, 1e8, 1e12, 1e16)
    cases = ((NonlinearProblem(rosen_F, rosen_J), SVector{2, R}(-1.2, 1), SUCCESS),
        (NonlinearProblem(ds_F, ds_J), SVector{2, R}(2, 0.5), SUCCESS),
        (NonlinearProblem(cubic_F, cubic_J), zero(R), LINESEARCH_FAILED))
    for (prob, x₀, code) in cases
        results = map(scales) do s
            solve(prob, x₀, dogleg_newton(), R(s))[2]
        end
        @test all(st -> st.code == code, results)
        @test all(st -> st.iterations == results[1].iterations, results)
    end
end

@testset "convergence where a fixed step fails, $R" for R in REAL_ELTYPES
    problems = ((NonlinearProblem(atan_F), R(10), zero(R)),
        (NonlinearProblem(ds_F), SVector{2, R}(2, 0.5), SVector{2, R}(1, 1)),
        (NonlinearProblem(rosen_F), SVector{2, R}(-1.2, 1), SVector{2, R}(1, 1)))
    for (prob, x₀, root) in problems, linsolve in (LUFactorization(), QRFactorization())

        m = dogleg_newton(; linsolve)
        x, st = solve(prob, x₀, m, one(R))
        @test st.code == SUCCESS
        @test rnorm(prob.F(x, one(R))) ≤ sqrt(eps(R)) * rnorm(prob.F(x₀, one(R)))
        @test rnorm(x - root) ≤ 100 * sqrt(eps(R)) * (1 + rnorm(root))
    end
    # a fixed step does not converge on the first two
    for (prob, x₀) in ((NonlinearProblem(atan_F), R(10)),
        (NonlinearProblem(ds_F), SVector{2, R}(2, 0.5)))
        @test solve(prob, x₀, Newton(), one(R); max_iterations = 50)[2].code != SUCCESS
    end
end

@testset "a step that the radius limits does not pass the step test, $R" for R in REAL_ELTYPES
    # `F = x − (x₀ + 1)` from `x₀` with radius 10⁻⁶: the first step is the Newton step `1`
    # truncated to 10⁻⁶, which lies below `x_reltol ‖x₀‖ = 2 eps(R) x₀`. Such a step is no measure
    # of the distance to the root, so the solve goes on to the root.
    x₀ = R === Float32 ? R(10) : R(1e10)
    @test R(1e-6) <= 2 * eps(R) * x₀
    prob = NonlinearProblem((x, p) -> x - (x₀ + 1))
    x₊, _, _, info = one_step(prob, x₀, nothing, dogleg_newton(1e-6))
    @test info.code == SUCCESS
    @test info.limited
    @test x₊ != x₀
    x, st = solve(prob, x₀, dogleg_newton(R(1e-6)))
    @test st.code == SUCCESS
    @test st.iterations > 1
    @test x == x₀ + 1
    @test st.fnorm == 0

    # the whole Newton step is not limited
    x₊, _, _, info = one_step(prob, x₀, nothing, dogleg_newton(2))
    @test info.code == SUCCESS
    @test !info.limited
    @test x₊ == x₀ + 1
end

@testset "x³ − 2x + 2 from 0 ends at the stationary point of ‖F‖, $R" for R in REAL_ELTYPES
    # `|F|` falls from 2 at 0 to its local minimum 0.911 at `√(2/3)`, and the root −1.769 lies
    # beyond the local maximum `|F(−√(2/3))| = 3.09`: no path along which `‖F‖` decreases reaches
    # it. The dogleg descends to the minimum, where every trial is rejected down to the floor.
    prob = NonlinearProblem(cubic_F)
    for linsolve in (LUFactorization(), QRFactorization())
        x, st = solve(prob, zero(R), dogleg_newton(; linsolve), one(R))
        @test st.code == LINESEARCH_FAILED
        @test abs(x - sqrt(R(2) / 3)) ≤ 100 * sqrt(eps(R))
        @test abs(x - R(CUBIC_ROOT)) > 1
    end
end

@kernel function dogleg_kernel!(codes, xs, @Const(prob), @Const(x₀s), @Const(m), @Const(s))
    i = @index(Global, Linear)
    @inbounds begin
        x, st = solve(prob, x₀s[i], m, s)
        codes[i] = st.code
        xs[i] = x
    end
end

function dogleg_allocations(prob::P, x₀::X, m::M, s::S) where {P, X, M, S}
    (
        solve(prob, x₀, m, s); @allocated solve(prob, x₀, m, s))
end
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "the kernel path: host and kernel agree, inferred, allocation-free, $R" for R in REAL_ELTYPES
    prob = NonlinearProblem(rosen_F, rosen_J)
    m = dogleg_newton()
    s = one(R)
    x₀s = [SVector{2, R}(-1.2, 1), SVector{2, R}(2, 2), SVector{2, R}(0.5, -1),
        SVector{2, R}(-3, 4)]
    host = [solve(prob, x₀, m, s) for x₀ in x₀s]
    codes = Vector{GeometricSolvers.ReturnCode}(undef, 4)
    xs = Vector{SVector{2, R}}(undef, 4)
    dogleg_kernel!(KA_BACKEND)(codes, xs, prob, x₀s, m, s; ndrange = 4)
    synchronize(KA_BACKEND)
    @test codes == [h[2].code for h in host]
    @test xs == [h[1] for h in host]
    @test all(==(SUCCESS), codes)

    @test @inferred(solve(prob, x₀s[1], m, s))[2].code == SUCCESS
    # `--check-bounds=yes` lets the static factors' `MMatrix` escape to the heap, so a run that
    # forces it skips this assertion
    if Base.JLOptions().check_bounds != 1
        @test dogleg_allocations(prob, x₀s[1], m, s) == 0
        @test dogleg_allocations(NonlinearProblem(cubic_F, cubic_J), zero(R), m, s) == 0
    end
    # the control: the barrier sees an allocation where there is one
    @test allocations(collect, x₀s[1]) > 0
    if JET_WORKS
        @test_opt solve(prob, x₀s[1], m, s)
        @test_opt solve(NonlinearProblem(cubic_F), zero(R), m, s)
    else
        @test_skip JET_WORKS  # issue #8
    end
end
