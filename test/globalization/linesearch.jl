using GeometricSolvers
using GeometricSolvers: linesearch, φ, φ′, ExactStep, InexactStep, MeasuredSlope,
                        LineSearchResult, ToReal, roundoff, smallest_step,
                        sufficient_decrease,
                        backtrack_step, zoom_step
using JET: JET
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using Random: Random
using Test

include("linefunctions.jl")

const adapt = GeometricSolvers.Adapt.adapt
inR(R, m) = adapt(ToReal{R}(), m)

const METHODS = (Static(), Backtracking(), Bisection(), StrongWolfe())
const SEARCHES = (Backtracking(), Bisection(), StrongWolfe())
const STEPS = (ExactStep(), MeasuredSlope(), InexactStep(0.1))
stepR(R, s::InexactStep) = InexactStep(R(s.η))
stepR(R, s) = s

# Run `m` in precision `R` along `lf` from the trial step `α`. The anchor merit `φ₀` is the
# caller's, computed outside any `Watched` wrapper, as a solver would have it.
function search(m, lf, R; step = MeasuredSlope(), α = 1, αmax = Inf)
    inner = lf isa Watched ? lf.lf : lf
    linesearch(inR(R, m), lf, stepR(R, step), φ(inner, zero(R)), R(α), R(αmax))
end

function strong_wolfe(lf, α, φ₀, d₀, c₁, c₂)
    φ(lf, α) ≤ φ₀ + c₁ * α * d₀ &&
        abs(φ′(lf, α)) ≤ c₂ * abs(d₀)
end

@testset "the methods are isbits and convert with ToReal" begin
    for m in METHODS, R in (Float32, Float64)

        mR = inR(R, m)
        @test isbits(mR)
        @test all(f -> !(getfield(mR, f) isa AbstractFloat) || getfield(mR, f) isa R,
            fieldnames(typeof(mR)))
    end
    @test Static().α === 1.0
    @test Static(0.8).α === 0.8
    @test inR(Float32, Backtracking()).maxiter === Int32(100)
    @test isbits(LineSearchResult{Float32}(1.0f0, 0.0f0, SUCCESS, Int32(1)))
    @test isbitstype(InexactStep{Float32})
end

@testset "constructors check their parameters on the host" begin
    for α in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError Static(α)
    end
    @test_throws ArgumentError Backtracking(; p = 1.5)
    @test_throws ArgumentError Backtracking(; p = 0.0)
    @test_throws ArgumentError Backtracking(; c₁ = -1e-4)
    @test_throws ArgumentError Backtracking(; c₁ = 1.0)
    @test_throws ArgumentError Backtracking(; maxiter = 0)
    @test_throws ArgumentError StrongWolfe(; c₁ = 0.9, c₂ = 0.1)
    @test_throws ArgumentError StrongWolfe(; c₂ = 1.0)
    @test_throws ArgumentError StrongWolfe(; αmax = -1.0)
    @test_throws ArgumentError StrongWolfe(; αmax = NaN)
    @test_throws ArgumentError Bisection(; αmax = 0.0)
    @test_throws ArgumentError Bisection(; maxiter = -1)
    @test Bisection(; αmax = Inf).αmax == Inf
end

# The six contracts of `SimpleSolvers/src/linesearch/linesearch.jl`, for every method, step kind
# and precision, on the pathological anchors of the `SimpleSolvers` contract test.
@testset "contracts 1-4 and 6 hold for every method on every pathological line" begin
    for R in (Float32, Float64)
        o = one(R)
        pathological = (
            Line(α -> R(NaN), α -> R(NaN)),                         # NaN merit
            Line(α -> o - α, α -> R(NaN)),                          # NaN derivative
            Line(α -> R(Inf), α -> -o),                             # Inf merit
            Line(α -> α + o, α -> o),                               # ascent anchor
            Line(α -> -o, α -> zero(R)),                            # stationary anchor
            Line(α -> α > 0 ? nextfloat(o) : o, α -> -2o),          # flat to round-off
            Line(α -> (α + o)^2, α -> 2 * (α + o)),                 # minimiser at α < 0
            Line(α -> o + α, α -> -2o),                             # slope contradicts values
            Line(α -> α > 0 ? R(NaN) : o, α -> -2o)                 # NaN beyond the anchor
        )
        for m in (METHODS..., Bisection(; αmax = Inf), StrongWolfe(; αmax = Inf)),
            step in STEPS, lf in pathological, αmax in (Inf, 0.5),
            α in (1.0, Inf, NaN, -1.0)

            r = @test_logs search(m, lf, R; step, αmax, α)         # 3: it logs nothing
            @test r isa LineSearchResult{R}                         # 1: it did not throw
            @test r.α > 0                                           # 2
            @test isfinite(r.α)
            @test r.α ≤ αmax                                        # 6
            @test r.code in instances(ReturnCode)
        end
        # 4: the anchor is reported, not searched
        for m in SEARCHES
            @test search(m, pathological[1], R).code == NONFINITE
            @test search(m, pathological[2], R).code == NONFINITE
            @test search(m, pathological[3], R).code == NONFINITE
            @test search(m, pathological[4], R).code == LINESEARCH_FAILED
            @test search(m, pathological[5], R).code == STALLED
            @test search(m, pathological[7], R).code == LINESEARCH_FAILED
            w = Watched(pathological[4], R)
            r = search(m, w, R; α = 0.7)
            @test r.α == R(0.7)                    # the caller's step, not the anchor
            @test isempty(w.atφ)                   # nothing beyond the slope at the anchor
            @test w.atφ′ == [zero(R)]
            @test r.evaluations == 1
        end
    end
end

@testset "Static returns its own step, bounded by the ceiling, and evaluates nothing" begin
    for R in (Float32, Float64)
        w = Watched(Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R)
        @test search(Static(), w, R; α = 0) == LineSearchResult{R}(1, NaN, SUCCESS, 0)
        @test search(Static(0.8), w, R).α == R(0.8)
        @test search(Static(0.8), w, R; αmax = 0.5).α == R(0.5)
        @test evaluated(w) == 0
    end
end

@testset "no method evaluates the merit at α = 0: φ₀ is the caller's" begin
    for R in (Float32, Float64), m in SEARCHES, step in STEPS, α in (0.25, 1.0, 4.0)
        w = Watched(Line(α -> 1 - 2α + 1000α^2, α -> -2 + 2000α), R)
        r = search(m, w, R; step, α)
        @test all(>(0), w.atφ)
        @test r.evaluations == evaluated(w)
    end
end

@testset "Backtracking: the exact step needs no φ′, a measured slope one" begin
    for R in (Float32, Float64), AT in (Array, JLArray)

        lf = Watched(NewtonLine(AT(R.(range(0.5, 3; length = 8)))), R)
        exact = search(Backtracking(), lf, R; step = ExactStep())
        @test isempty(lf.atφ′)
        @test exact.code == SUCCESS
        measured = search(Backtracking(), lf, R; step = MeasuredSlope())
        @test lf.atφ′ == [zero(R)]
        # the measured slope is -2φ₀ up to the rounding of two sums of 8 terms
        @test measured.α ≈ exact.α rtol = 8eps(R)
        @test measured.evaluations == exact.evaluations + 1
    end
end

@testset "Backtracking: Eisenstat and Walker's test for an inexact step" begin
    for R in (Float32, Float64)
        # The merit of a linear problem along a step with linear residual η‖r‖:
        # r(x + αd) = (1 - α(1 - η)) r. With η near 1 the Armijo test of an exact step rejects
        # α = 1, and Eisenstat and Walker's test accepts it, without a φ′.
        η = R(0.99999)
        linear = Watched(
            Line(α -> (1 - α * (1 - η))^2, α -> -2 * (1 - η) * (1 - α * (1 - η))),
            R)
        r = search(Backtracking(), linear, R; step = InexactStep(η))
        @test r == LineSearchResult{R}(1, (1 - (1 - η))^2, SUCCESS, 1)
        @test isempty(linear.atφ′)
        @test search(Backtracking(), linear, R; step = ExactStep()).α < 1

        # A backtrack must update η ← 1 - θ(1 - η): the step accepted here decreases the merit by
        # less than the test at α = 1 demands, and by more than the updated test demands.
        η₀ = R(0.2)
        c₁ = R(0.5)
        curved = Line(α -> (1 - (1 - η₀) * α)^2 + 3α^2, α -> -2(1 - η₀) *
                                                             (1 - (1 - η₀) * α) + 6α)
        w = Watched(curved, R)
        r = search(Backtracking(; c₁ = 0.5), w, R; step = InexactStep(η₀))
        @test r.code == SUCCESS
        @test isempty(w.atφ′)
        @test 0 < r.α < 1
        @test r.φ ≤ (1 - c₁ * r.α * (1 - η₀))^2          # the updated test holds
        @test r.φ > (1 - c₁ * (1 - η₀))^2                 # the test of α = 1 does not
        @test r.evaluations == 2

        # A trial step past α = 1 / (c₁(1 - η₀)) makes the factor 1 - c₁(1 - η) negative; its
        # square must not accept a step with almost no decrease.
        flat = Line(α -> α > 0 ? 1 - R(1e-6) : one(R), α -> -2 * one(R))
        for (c₁, α) in ((0.5, 5.0), (1e-4, 3e4))
            r = search(Backtracking(; c₁), flat, R; step = InexactStep(0.1), α)
            @test r.α < 1 / (c₁ * (1 - 0.1))
            # the test with its round-off allowance τ
            @test r.code != SUCCESS || r.φ ≤ (1 - R(c₁) * r.α * R(0.9))^2 + roundoff(one(R))
        end
    end
end

@testset "defect 1: StrongWolfe returns SUCCESS only for a strong Wolfe step" begin
    # The six functions of Moré and Thuente (1994), their curvature constant η and their initial
    # steps. Their μ equals η for five of the functions, which the c₁ < c₂ of Nocedal and Wright
    # excludes, so c₁ keeps its default.
    for R in (Float32, Float64), kind in 1:6, α₀ in MT_STEPS
        lf = MoreThuente(R, kind)
        c₁, c₂ = 1e-4, MT_C[kind][2]
        m = StrongWolfe(; c₂)
        φ₀, d₀ = φ(lf, zero(R)), φ′(lf, zero(R))
        r = search(m, lf, R; α = α₀)
        if r.code == SUCCESS
            @test strong_wolfe(lf, r.α, φ₀, d₀, R(c₁), R(c₂))
        end
        # Every case is solved in Float64, which is where Moré and Thuente ran them.
        @test r.code == SUCCESS
        @test r.α > 0
    end

    # The zoom interpolates: on a quadratic merit from an overshooting trial step the first
    # zoom trial is the minimiser. Plain bisection needs two more evaluations.
    for R in (Float32, Float64)
        w = Watched(Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R)
        r = search(StrongWolfe(), w, R; α = 4)
        @test r.code == SUCCESS
        @test r.α ≈ 1 atol = 4eps(R)
        @test r.evaluations == 5                           # φ′(0), then φ and φ′ at 4 and at 1
    end
    # The cubic step is exact on a cubic, in either orientation of the bracket and at any
    # scale: φ(α) = α³ - 3α has its minimiser at 1.
    cubic(α) = (α^3 - 3α, 3α^2 - 3)
    for (a, b) in ((0.0, 1.5), (1.5, 0.0), (0.5, 2.0)), s in (2.0^-40, 1.0, 2.0^40)

        (φa, da), (φb, db) = s .* cubic(a), s .* cubic(b)
        @test zoom_step(a, φa, da, b, φb, db) ≈ 1.0 rtol = 4eps()
    end
    # clamped to the inner 80 % of the bracket, and the middle for a cubic without a minimiser
    @test zoom_step(0.0, 0.0, -1.0, 1.0, -0.999, -1.0e-3) ≈ 0.9
    @test zoom_step(0.0, 0.0, 1.0, 1.0, 2 / 3, 1.0) == 0.5

    # A step that fails the curvature condition is never a success: at the ceiling, or after the
    # cap is spent.
    for R in (Float32, Float64)
        far = Line(α -> (α - R(1e7))^2 / R(1e14), α -> 2 * (α - R(1e7)) / R(1e14))
        r = search(StrongWolfe(), far, R)
        @test r.α == R(65536)
        @test r.code == LINESEARCH_FAILED
        @test r.φ < φ(far, zero(R))                       # the step still decreases the merit
        lf = MoreThuente(R, 3)
        r = search(StrongWolfe(; c₂ = 0.1, maxiter = 1), lf, R; α = 1e-3)
        @test r.code != SUCCESS
    end
end

@testset "defect 2: Backtracking's cost does not change with the merit scale" begin
    merits = ((α -> 1 - 2α + 1000α^2, α -> -2 + 2000α),        # near-quadratic: the cubic step
        (α -> (α - 100)^2 / 10^4, α -> 2 * (α - 100) / 10^4),
        (α -> 1 - 2α + 100α^2 + 50α^3, α -> -2 + 200α + 150α^2),
        (α -> exp(8α) - 9α, α -> 8exp(8α) - 9))
    for R in (Float32, Float64), (f, d) in merits, step in (MeasuredSlope(), ExactStep())
        scales = R == Float32 ? (1e-8, 1.0, 1e8) : (1e-200, 1e-8, 1.0, 1e8, 1e200)
        results = map(scales) do s
            search(Backtracking(), Line(α -> R(s) * f(α), α -> R(s) * d(α)), R; step)
        end
        @test all(r -> r.evaluations == results[1].evaluations, results)
        @test all(r -> r.code == results[1].code, results)
        @test all(r -> isapprox(r.α, results[1].α; rtol = 8eps(R)), results)
    end
    # On the near-quadratic merit the cubic model is the quadratic itself, so the ladder is
    # 1, 0.1, 0.01 and then the minimiser 0.001: the stable form of the cubic step, where the
    # textbook form (-b + √(b² - 3a d₀)) / 3a cancels.
    for R in (Float32, Float64), s in (1e-8, 1.0, 1e8)

        w = Watched(Line(α -> R(s) * (1 - 2α + 1000α^2), α -> R(s) * (-2 + 2000α)), R)
        r = search(Backtracking(), w, R)
        @test w.atφ ≈ R[1, 0.1, 0.01, 0.001] rtol = 64eps(R)
        @test r.code == SUCCESS
    end
    @test backtrack_step(1.0, -2.0, 0.1, 10.8, 1.0, 999.0, 0.5) ≈ 0.01
    @test backtrack_step(1e200, -2e200, 0.1, 10.8e200, 1.0, 999e200, 0.5) ≈ 0.01
end

@testset "defect 3: Bisection with the lower end at 0 returns α > 0 before its cap" begin
    for R in (Float32, Float64)
        # The minimiser lies below the resolution of the step, or φ′ is positive right after
        # the anchor: the lower end of the bracket stays at 0 and a relative-width stop cannot
        # fire. The search stops at the smallest informative step instead.
        near = Line(α -> 1 + (α - R(1e-12))^2, α -> 2 * (α - R(1e-12)))
        lying = Line(α -> α > 0 ? 1 + α : one(R), α -> α > 0 ? one(R) : -2 * one(R))
        for lf in (near, lying)
            w = Watched(lf, R)
            r = search(Bisection(), w, R)
            @test r.α > 0
            @test all(>(0), w.atφ)                    # φ₀ is reused, never evaluated
            @test r.evaluations < Bisection().maxiter
            @test r.code == STALLED                   # at the round-off floor, not a spent cap
        end
    end
    # Only a spent cap is a spent cap.
    r = search(Bisection(; maxiter = 3), Line(α -> (α - 1)^2, α -> 2 * (α - 1)), Float64; α = 1e-3)
    @test r.code == LINESEARCH_FAILED
    @test r.α > 0
end

@testset "defect 4: a large increase of the merit is a failure, never STALLED" begin
    for R in (Float32, Float64), step in (MeasuredSlope(), ExactStep())

        cliff = Line(α -> α > 0 ? 1 + 1000α : one(R), α -> -2 * one(R))
        steep = Line(α -> α > 0 ? 1 + α + R(1e6) * α^2 : one(R), α -> -2 * one(R))
        for lf in (cliff, steep), m in SEARCHES

            r = search(m, lf, R; step)
            @test r.code == LINESEARCH_FAILED
            @test r.φ > φ(lf, zero(R)) + roundoff(one(R))
        end
    end
end

@testset "contract 6 and the caller's αmax" begin
    far = Line(α -> (α - 1.0e7)^2 / 1.0e14, α -> 2(α - 1.0e7) / 1.0e14)
    near = Line(α -> (α - 1.0)^2, α -> 2(α - 1.0))
    # A ceiling is not a failure: the minimising search stops at its own and reports a decrease.
    r = search(Bisection(), far, Float64)
    @test r.α == 65536.0
    @test r.code == SUCCESS
    @test r.φ == φ(far, r.α)
    @test search(Bisection(; αmax = Inf), far, Float64).α > 1e6

    for m in METHODS, ceiling in (10.0, 0.25, 0.005)

        w = Watched(far, Float64)
        r = search(m, w, Float64; αmax = ceiling)
        @test 0 < r.α ≤ ceiling
        @test all(≤(ceiling), w.atφ) && all(≤(ceiling), w.atφ′)  # nothing is evaluated above it
    end
    for m in METHODS
        # it binds below the minimiser, and on an ascent anchor
        @test 0 < search(m, near, Float64; αmax = 0.5).α ≤ 0.5
        @test 0 <
              search(m, Line(α -> (α + 1)^2, α -> 2(α + 1)), Float64; α = 4, αmax = 0.5).α ≤
              0.5
        # a ceiling that does not bind changes nothing
        @test search(m, near, Float64; αmax = 1e6) === search(m, near, Float64)
        # a ceiling that is not a step is a caller error: nothing is evaluated
        for bad in (0.0, -1.0, NaN)
            w = Watched(near, Float64)
            r = search(m, w, Float64; αmax = bad)
            @test r.code == LINESEARCH_FAILED
            @test r.evaluations == 0 == evaluated(w)
            @test r.α > 0
        end
    end
    # a merit that only falls: the ceiling is the step
    for ceiling in (2.0, 1.0, 0.5, 0.05, 0.005)
        w = Watched(Line(α -> 1 - α, α -> -1.0), Float64)
        r = search(Bisection(), w, Float64; αmax = ceiling)
        @test r.α == ceiling
        @test r.code == SUCCESS
        @test r.φ == 1 - ceiling
        @test r.evaluations ≤ 4
    end
    @test search(Bisection(), far, Float64; αmax = Inf) ===
          search(Bisection(), far, Float64)
end

@testset "contract 5: the cost of every method is independent of the merit's scale" begin
    for m in SEARCHES, R in (Float32, Float64)

        scales = R == Float32 ? (1e-12, 1e-6, 1.0, 1e6, 1e12) :
                 (1e-200, 1e-12, 1.0, 1e12, 1e200)
        rs = [search(m, Line(α -> R(c) * (α - 1)^2, α -> R(c) * 2 * (α - 1)), R; α = 0.5)
              for c in scales]
        @test all(r -> r.α ≈ rs[1].α, rs)
        @test all(r -> r.evaluations == rs[1].evaluations, rs)
        @test all(r -> r.code == SUCCESS, rs)
        # and on the Moré–Thuente functions, scaled by powers of two: the arithmetic is then
        # exact, so any change of the cost is a threshold of the method that does not scale
        for kind in 1:6
            lf = MoreThuente(R, kind)
            rs = [search(m, Line(α -> R(c) * φ(lf, α), α -> R(c) * φ′(lf, α)), R; α = 1e-1)
                  for c in (2.0^-20, 1.0, 2.0^20)]
            @test all(r -> r === rs[2] || (r.α == rs[2].α && r.code == rs[2].code), rs)
            @test all(r -> r.evaluations == rs[1].evaluations, rs)
        end
        # and on the round-off-floor path: a merit one ulp above φ₀, and a cliff. Powers of two
        # scale the merit exactly; a decimal scale changes the relative size of one ulp, so
        # there only the code must agree.
        noise(s) = Line(α -> α > 0 ? R(s) * nextfloat(one(R)) : R(s), α -> -2 * R(s))
        cliff(s) = Line(α -> α > 0 ? R(s) * (1 + 1000α) : R(s), α -> -2 * R(s))
        exact = R == Float32 ? (2.0^-60, 1.0, 2.0^60) : (2.0^-600, 1.0, 2.0^600)
        decimal = R == Float32 ? (1e-20, 1e20) : (1e-200, 1e200)
        for merit in (noise, cliff)
            rs = [search(m, merit(s), R) for s in exact]
            @test all(
                r -> (r.α, r.code, r.evaluations) ==
                     (rs[2].α, rs[2].code, rs[2].evaluations), rs)
            @test all(s -> search(m, merit(s), R).code == rs[2].code, decimal)
        end
        @test all(
            s -> search(m, cliff(s), R).evaluations == search(m, cliff(1.0), R).evaluations,
            decimal)
    end
end

# The tests of `SimpleSolvers/test/linesearch_tests.jl` for the four methods, where the design
# keeps the behaviour. Not ported: `Quadratic` and `BierlaireQuadratic` (§5.1); the expansion
# phase of `Backtracking`, its `τ_ulps` key and its curvature warning, the `Float16` rows, the
# `Linesearch` and `LinesearchProblem` objects, `change_precision`, `bracket_minimum`,
# `triple_point_finder` and the message tests, none of which this package has.
@testset "ported from SimpleSolvers" begin
    f(x) = x^2 - 1
    g(x) = 2x
    δx(x) = -g(x) / 2
    parabola(x₀) = Line(α -> f(x₀ + α * δx(x₀)), α -> g(x₀ + α * δx(x₀)) * δx(x₀))
    function iterate(m, x, n)
        for _ in 1:n
            x += search(m, parabola(x), Float64).α * δx(x)
        end
        x
    end

    @testset "Static" begin
        @test Static() === Static(1.0)
        @test search(Static(), parabola(-3.0), Float64; α = 0).α == 1.0
        @test search(Static(0.8), parabola(-3.0), Float64; α = 0).α == 0.8
    end

    @testset "Bisection, Backtracking and StrongWolfe reach the minimiser" begin
        @test iterate(Bisection(), -3.0, 1) ≈ 0 atol = ∛(2eps())
        @test iterate(Backtracking(), -3.0, 20) ≈ 0 atol = ∛(2eps())
        @test iterate(StrongWolfe(), -3.0, 20) ≈ 0 atol = ∛(2eps())
        for α₀ in (0.25, 0.5, 1.0, 2.0, 4.0)
            @test -3.0 + search(Bisection(), parabola(-3.0), Float64; α = α₀).α * 3.0 ≈ 0 atol = ∛(2eps())
        end
    end

    @testset "Backtracking stall" begin
        r = search(Backtracking(), Line(α -> (α - 100.0)^2, α -> 2.0 * (α - 100.0)), Float64)
        @test r.α == 1.0
        @test r.code == SUCCESS
    end

    @testset "Backtracking: non-descent and stationary anchors are reported, not searched" begin
        ascent = Line(α -> α + 1.0, α -> 1.0)
        r = search(Backtracking(), ascent, Float64; α = 0.7)
        @test r.code == LINESEARCH_FAILED
        @test r.α == 0.7
        @test r.α == search(StrongWolfe(), ascent, Float64; α = 0.7).α
        @test search(Backtracking(), Line(α -> -1.0, α -> 0.0), Float64).code == STALLED
        @test search(Backtracking(), Line(α -> 1.0 + α, α -> -2.0), Float64).code ==
              LINESEARCH_FAILED
    end

    @testset "Backtracking: stagnation at the merit's round-off floor" begin
        # frozen below α = 1e-6, increasing above: no α decreases it
        w = Watched(Line(α -> α ≤ 1e-6 ? 1.0 : 1.0 + α, α -> -2.0), Float64)
        r = search(Backtracking(), w, Float64)
        @test r.code == STALLED
        @test length(w.atφ) < 43
        # the frozen stop counts only steps below √eps, where x + αd may round to x
        @test sqrt(eps()) / 10 < r.α ≤ sqrt(eps())
        # every α > 0 lands one ulp above φ₀: pure round-off
        w = Watched(Line(α -> α > 0 ? nextfloat(1.0) : 1.0, α -> -2.0), Float64)
        r = search(Backtracking(), w, Float64)
        @test r.code == STALLED
        @test length(w.atφ) < 53
        @test r.α == smallest_step(1e-4, -2.0, roundoff(1.0)) == 4.440892098500626e-12
    end

    @testset "no method accepts a step that increases the merit" begin
        for T in (Float32, Float64), m in SEARCHES, step in (MeasuredSlope(), ExactStep())
            τ = roundoff(one(T))
            creep = Line(α -> one(T) + (α > 0 ? τ / 2 : zero(T)), α -> -2 * one(T))
            r = search(m, creep, T; step)
            @test r.code != SUCCESS
        end
    end

    @testset "a merit equal to φ₀ at a large step is not the round-off floor" begin
        # φ = 1 - 2α + 6α² - 4α³ equals φ₀ at α = 1 and at α = 1/2, and has φ(0.21) ≈ 0.81
        for T in (Float32, Float64), s in (1e-8, 1.0, 1e8),
            step in (MeasuredSlope(), ExactStep())
            lf = Line(α -> T(s) * (1 - 2α + 6α^2 - 4α^3), α -> T(s) * (-2 + 12α - 12α^2))
            r = search(Backtracking(), lf, T; step)
            @test r.code == SUCCESS
            @test r.φ < T(0.9) * T(s)
        end
    end

    @testset "roundoff, smallest_step and the interpolation" begin
        @test roundoff(1.0) == 4eps(1.0)
        @test roundoff(1e-20) == 4eps(1e-20)
        τ = roundoff(1.0)
        @test smallest_step(1e-4, -2.0, τ) == 4.440892098500626e-12
        @test smallest_step(1e-4, -0.0, τ) == sqrt(eps(1.0))
        @test smallest_step(1e-4, -1e30, τ) == eps(1.0)
        @test smallest_step(1e-4, -1e-30, τ) == sqrt(eps(1.0))
        @test smallest_step(1e-4, -2.0, 0.0) == eps(1.0)
        for (φα, αp, φp) in ((3.0, NaN, NaN), (3.0, 2.0, 5.0), (NaN, NaN, NaN), (
            Inf, 1.5, 2.0))
            @test 0.1 ≤ backtrack_step(1.0, -2.0, 1.0, φα, αp, φp, 0.5) ≤ 0.5
        end
        w = Watched(Line(α -> 1.0 - 2α + 1000α^2, α -> -2.0 + 2000α), Float64)
        r = search(Backtracking(), w, Float64)
        @test r.code == SUCCESS
        @test length(w.atφ) < 10
        @test r.evaluations == length(w.atφ) + 1
        @test r.α ≤ 0.5
    end

    @testset "sufficient decrease with the round-off allowance" begin
        @test !sufficient_decrease(nextfloat(1.0), 1.0, 1e-4 * 1.0 * -2.0, 0.0)
        @test !sufficient_decrease(nextfloat(1.0), 1.0, 1e-4 * 1e-13 * -2.0, 0.0)
        target = 1.0 + 1e-4 * 0.1 * -2.0
        @test !sufficient_decrease(nextfloat(target, 2), 1.0, 1e-4 * 0.1 * -2.0, 0.0)
        @test sufficient_decrease(nextfloat(target, 2), 1.0, 1e-4 * 0.1 * -2.0, 4eps(1.0))
        @test !sufficient_decrease(nextfloat(1.0), 1.0, 1e-4 * 1.0 * -2.0, 4eps(1.0))
        @test !sufficient_decrease(nextfloat(1.0), 1.0, 1e-4 * 1e-13 * -2.0, 4eps(1.0))
        @test 1.0 + 1e-4 * 1e-13 * -2.0 == 1.0
        @test sufficient_decrease(1.0, 1.0, 1e-4 * 1e-13 * -2.0, 4eps(1.0))
    end

    @testset "every method reports a decrease as one" begin
        lf = Line(α -> (α - 0.7)^2, α -> 2 * (α - 0.7))
        for m in METHODS
            @test search(m, lf, Float64).code == SUCCESS
        end
    end

    @testset "the cap bounds the ladder, and a spent cap is not the floor" begin
        noise = Line(α -> α > 0 ? nextfloat(1.0) : 1.0, α -> -2.0)
        @test search(Backtracking(), noise, Float64).code == STALLED
        w = Watched(noise, Float64)
        r = search(Backtracking(; maxiter = 3), w, Float64)
        @test length(w.atφ) == 3
        @test r.code == LINESEARCH_FAILED
    end

    @testset "a Bisection converges onto a minimum, never onto a maximum" begin
        # φ′ has a minimum at 0.3 and a maximum at 2
        lf = Line(a -> (a - 1.0)^2, a -> -(a - 0.3) * (a - 2.0))
        for α₀ in (0.01, 1.0)
            r = search(Bisection(), lf, Float64; α = α₀)
            @test r.α ≈ 0.3 atol = 1e-8
            @test r.code == SUCCESS
        end
        @test search(Bisection(), Line(a -> (a - 1.0)^2, a -> 2(a - 1.0)), Float64).α ≈ 1.0 atol = ∛(2eps())
    end

    @testset "a Bisection that cannot bracket never reports a floor" begin
        for lf in (Line(α -> (α - 1.0)^2, α -> -1.0), Line(α -> (α + 1.0)^2, α -> -1.0))
            r = search(Bisection(), lf, Float64; α = 0.01)
            @test r.code == LINESEARCH_FAILED
            @test r.α > 0
            @test r.φ == φ(lf, r.α)
        end
        forever = Line(α -> 1.0 - α, α -> -1.0)
        r = search(Bisection(), forever, Float64)
        @test r.code == SUCCESS
        @test r.α == 65536.0
        @test r.φ == 1.0 - r.α
        r = search(Bisection(; αmax = Inf), forever, Float64)
        @test r.code == LINESEARCH_FAILED
        @test r.φ == 1.0 - r.α
    end

    @testset "StrongWolfe line search (bracket + zoom)" begin
        lf = parabola(-3.0)
        φ₀, d₀ = φ(lf, 0.0), φ′(lf, 0.0)
        for α₀ in (0.1, 0.5, 1.0, 2.0)
            r = search(StrongWolfe(), lf, Float64; α = α₀)
            @test strong_wolfe(lf, r.α, φ₀, d₀, 1e-4, 0.9)
            @test r.code == SUCCESS
        end
        @test search(StrongWolfe(; c₂ = 1e-2), lf, Float64; α = 2.0).α ≈ 1.0 atol = 1e-6
        @test search(StrongWolfe(), Line(a -> (a + 1.0)^2, a -> 2(a + 1.0)), Float64; α = 0.7).α ==
              0.7
    end

    @testset "StrongWolfe reports a non-finite anchor instead of asserting" begin
        for lf in (Line(α -> NaN, α -> NaN), Line(α -> 1.0 - α, α -> NaN))
            sw = search(StrongWolfe(), lf, Float64; α = 0.7)
            bt = search(Backtracking(), lf, Float64; α = 0.7)
            @test sw.code == bt.code == NONFINITE
            @test sw.α == bt.α == 0.7
        end
    end

    @testset "Linesearch Integration Tests" begin
        # the ‖F‖² merit of a scalar Newton step on F(x) = eˣ(x³/2 - 5x² + 2x) + 2
        Random.seed!(1234)
        x = -10 * rand()
        for T in (Float32, Float64)
            F(x) = exp(x) * (T(0.5) * x^3 - 5x^2 + 2x) + 2one(T)
            J(x) = exp(x) * (T(0.5) * x^3 - 5x^2 + 2x) + exp(x) * (T(1.5) * x^2 - 10x + 2)
            x₀ = T(x)
            d = -F(x₀) / J(x₀)
            lf = Line(α -> F(x₀ + α * d)^2, α -> 2F(x₀ + α * d) * J(x₀ + α * d) * d)
            r = search(Bisection(), lf, T)
            @test φ′(lf, r.α) ≈ 0 atol = ∛(2eps(T))
        end
    end

    @testset "Linesearch T-consistency" begin
        for T in (Float32, Float64), m in METHODS

            lf = Line(α -> (α - 2one(T))^2, α -> 2 * (α - 2one(T)))
            @test search(m, lf, T).α isa T
        end
    end

    @testset "evaluations is a real count for every method" begin
        for m in SEARCHES
            w = Watched(Line(α -> 1.0 - 2α + 1000α^2, α -> -2.0 + 2000α), Float64)
            r = search(m, w, Float64)
            @test r.evaluations == evaluated(w) > 0
        end
    end
end

@testset "R3: the searches on Array and JLArray" begin
    for R in (Float32, Float64), m in METHODS, step in (ExactStep(), MeasuredSlope())
        x₀ = R.(range(0.5, 3; length = 8))
        results = map((Array, JLArray)) do AT
            x = AT(copy(x₀))
            for _ in 1:50
                lf = NewtonLine(x)
                # the residual floor: |x² - 2| is a few eps near √2 in each of the 8 terms
                φ(lf, zero(R)) ≤ length(x) * (8eps(R))^2 && break
                r = search(m, lf, R; step)
                @test r.code == SUCCESS
                x = lf.x .+ r.α .* lf.d
            end
            Array(x)
        end
        # the Newton iteration converges to √2 on both: |x - √2| ≤ |x² - 2| / 2x
        @test all(r -> all(isapprox.(r, sqrt(R(2)); rtol = 8eps(R))), results)
        # and the first search takes the same step on both
        lfA, lfJ = NewtonLine(x₀), NewtonLine(JLArray(x₀))
        rA, rJ = search(m, lfA, R; step), search(m, lfJ, R; step)
        @test rA.code == rJ.code
        @test rA.evaluations == rJ.evaluations
        @test rA.α ≈ rJ.α rtol = 8eps(R)
    end
end

@kernel function search_kernel!(αs, codes, counts, @Const(lfs), @Const(α₀s), ls, step)
    i = @index(Global)
    lf = lfs[i]
    r = linesearch(ls, lf, step, φ(lf, zero(eltype(αs))), α₀s[i])
    αs[i] = r.α
    codes[i] = r.code
    counts[i] = r.evaluations
end

@testset "R1: the searches run inside a KernelAbstractions kernel on CPU()" begin
    backend = KernelAbstractions.CPU()
    for R in (Float32, Float64), m in METHODS, step in (MeasuredSlope(), InexactStep(0.1))
        ls = inR(R, m)
        stepr = stepR(R, step)
        lfs = [MoreThuente(R, kind) for kind in 1:6 for _ in MT_STEPS]
        α₀s = [R(α) for _ in 1:6 for α in MT_STEPS]
        n = length(lfs)
        αs, codes, counts = zeros(R, n), fill(MAXITERS, n), zeros(Int32, n)
        search_kernel!(backend)(αs, codes, counts, lfs, α₀s, ls, stepr; ndrange = n)
        KernelAbstractions.synchronize(backend)
        host = [linesearch(ls, lf, stepr, φ(lf, zero(R)), α) for (lf, α) in zip(lfs, α₀s)]
        @test αs == getfield.(host, :α)
        @test codes == getfield.(host, :code)
        @test counts == getfield.(host, :evaluations)
    end
end

@testset "R1: isbits, inferred and allocation-free" begin
    count_allocations(ls, lf, step, φ₀, α) = @allocated linesearch(ls, lf, step, φ₀, α)
    for R in (Float32, Float64), m in METHODS, step in STEPS
        ls, stepr = inR(R, m), stepR(R, step)
        lf = MoreThuente(R, 1)
        φ₀ = φ(lf, zero(R))
        r = @inferred linesearch(ls, lf, stepr, φ₀, one(R))
        @test isbits(r)
        count_allocations(ls, lf, stepr, φ₀, one(R))
        @test count_allocations(ls, lf, stepr, φ₀, one(R)) == 0
    end
end

@testset "JET: the searches are optimisable and error-free" begin
    # the flag of test/base/reductions.jl, which also covers the stub JET of an unsupported Julia
    JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
    if JET_WORKS
        for R in (Float32, Float64), m in METHODS, step in STEPS
            types = (typeof(inR(R, m)), MoreThuente{R}, typeof(stepR(R, step)), R, R, R)
            JET.test_opt(linesearch, types)
            JET.test_call(linesearch, types)
        end
    else
        @test_skip "JET does not load on Julia $(VERSION)"
    end
end
