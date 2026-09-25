using GeometricSolvers
using GeometricSolvers: linesearch, φ, φ′, ExactStep, InexactStep, MeasuredSlope,
                        LineSearchResult, ToReal, roundoff, smallest_step,
                        sufficient_decrease, classify, floor_code,
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

# `search` on the merit f and its slope d, both multiplied by the constant c.
function scaled(m, f, d, R, c; kwargs...)
    search(m, Line(α -> R(c) * f(α), α -> R(c) * d(α)), R; kwargs...)
end

function strong_wolfe(lf, α, φ₀, d₀, c₁, c₂)
    φ(lf, α) - φ₀ ≤ c₁ * α * d₀ &&
        abs(φ′(lf, α)) ≤ c₂ * abs(d₀)
end

# The most evaluations a method can spend, from the structure of its loop: the slope at the
# anchor, at most two per trip, and the merit at the step of Bisection.
max_evaluations(::Static) = 0
max_evaluations(m::Backtracking) = 1 + m.maxiter
max_evaluations(m::Bisection) = 1 + m.maxiter + 1
max_evaluations(m::StrongWolfe) = 1 + 2m.maxiter

# The code agrees with the merit at the step: a finite decrease by more than τ for SUCCESS, a
# change within τ for STALLED. Static evaluates nothing and carries φ = NaN, and so does a search
# that stops at its anchor.
function agrees(m, r, φ₀)
    τ = roundoff(φ₀)
    m isa Static && r.code == SUCCESS && return isnan(r.φ)
    r.code == SUCCESS && return isfinite(r.φ) && r.φ - φ₀ < -τ
    r.code == STALLED && return isnan(r.φ) || abs(r.φ - φ₀) ≤ τ
    true
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
            Line(α -> α > 0 ? R(NaN) : o, α -> -2o),                # NaN beyond the anchor
            Line(α -> nextfloat(zero(R)) + α, α -> -2nextfloat(zero(R))),  # subnormal anchor, rises
            Line(α -> α > 0 ? zero(R) : nextfloat(zero(R)), α -> -2nextfloat(zero(R))), # subnormal, falls
            Line(α -> o + α, α -> α < R(0.5) ? -2o : 2o),           # φ′ turns where φ rises
            Line(α -> α > 0 ? R(-Inf) : o, α -> -2o),               # -Inf beyond α = 0
            Line(α -> α > R(0.3) ? R(-Inf) : (α - o)^2, α -> 2 * (α - o)),  # -Inf beyond 0.3
            Line(α -> α > R(0.5) ? R(Inf) : (α - o)^2, α -> 2 * (α - o)),   # Inf beyond 0.5
            Line(α -> α > 2 ? R(NaN) : (α - 4)^2, α -> 2 * (α - 4)),        # NaN beyond 2
            Line(α -> α > 0 ? o + α : o, α -> α > 0 ? o : -o),      # minimum at α = 0⁺
            Line(α -> 2 - sin(3α), α -> -3cos(3α)),                 # not convex
            Line(α -> o - α, α -> -o),                              # falls forever
            Line(α -> (α - R(0.7))^2, α -> 2 * (α - R(0.7)))        # healthy
        )
        for m in (METHODS..., Bisection(; αmax = Inf), StrongWolfe(; αmax = Inf)),
            step in (STEPS..., -2.0), lf in pathological,
            αmax in (Inf, 0.5, floatmin(R), nextfloat(zero(R))),
            α in (1.0, 0.0, -3.0, Inf, NaN)

            w = Watched(lf, R)
            r = @test_logs search(m, w, R; step, αmax, α)          # 3: it logs nothing
            @test r isa LineSearchResult{R}                         # 1: it did not throw
            @test 0 < r.α < Inf                                     # 2
            @test r.α ≤ αmax                                        # 6
            @test r.α ≤ GeometricSolvers.method_αmax(inR(R, m))
            @test 0 ≤ r.evaluations ≤ max_evaluations(m)            # 5: the bound
            @test r.evaluations == evaluated(w)
            @test !(0 in w.atφ)                                     # φ(0) is the caller's
            @test all(≤(αmax), w.atφ) && all(≤(αmax), w.atφ′)
            @test agrees(m, r, φ(lf, zero(R)))
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

@testset "StrongWolfe returns SUCCESS only for a strong Wolfe step" begin
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
        # φ′(0), φ at 4, which fails the first condition and so costs no φ′, then φ and φ′ at 1
        @test r.evaluations == 4
        @test w.atφ′ ≈ R[0, 1] atol = 4eps(R)
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

@testset "Bisection at a lower end of 0 bisects in log space, within a stated bound" begin
    # The minimiser lies below the step floor, or φ′ is positive right after the anchor: the
    # lower end of the bracket stays at 0 and a relative-width stop cannot fire. The search
    # bisects in log space down to the floor instead. Its count is at most
    # 3 + ⌈log₂ log₂(α / floatmin(R))⌉: φ′(0), one bracket, the geometric trials from α down to
    # the floor, which is at least floatmin, and the merit at the step. That is 10 in Float32
    # and 13 in Float64 for α = 1, 11 and 14 for the default ceiling 2¹⁶, and 11 and 14 for
    # 1e16 with αmax = Inf: it stays bounded without a ceiling.
    bound(R, α) = 3 + ceil(Int, log2(log2(α) - log2(floatmin(R))))
    for R in (Float32, Float64)
        near = Line(α -> 1 + (α - R(1e-12))^2, α -> 2 * (α - R(1e-12)))
        lying = Line(α -> α > 0 ? 1 + α : one(R), α -> α > 0 ? one(R) : -2 * one(R))
        # a steep anchor puts the floor far down: φ′(0) = -1e30 in Float64, -1e10 in Float32
        s = R == Float64 ? 1e30 : 1e10
        kink = Line(α -> α > 0 ? 1 + α : one(R), α -> α > 0 ? one(R) : -R(s))
        steep = Line(α -> 1 + R(s)^3 * (α - R(1 / s)^2)^2, α -> 2R(s)^3 * (α - R(1 / s)^2))
        trials = (
            (Bisection(), 1.0), (Bisection(), 65536.0), (Bisection(; αmax = Inf), 1e16),
            (Bisection(; αmax = Inf), R == Float64 ? 1e300 : 1e38),
            # trial steps above the minimiser, so that the lower end stays at 0
            ((Bisection(; αmax = Inf), 10.0^e)
            for e in (R == Float64 ? (0:25:300) : (0:5:35)))...)
        for lf in (near, lying, kink, steep), (m, α) in trials

            w = Watched(lf, R)
            r = search(m, w, R; α)
            @test r.α > 0
            @test all(>(0), w.atφ)                    # φ₀ is reused, never evaluated
            @test r.evaluations ≤ bound(R, α)
            @test r.code != SUCCESS                   # no detectable decrease exists
        end
        @test bound(R, 1.0) == (R == Float64 ? 13 : 10)
        @test bound(R, 65536.0) == (R == Float64 ? 14 : 11)
        @test bound(R, 1e16) == (R == Float64 ? 14 : 11)
        # a trial step below the floor is raised to it: the search then brackets up from there
        for m in SEARCHES
            r = search(m, Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R; α = 1e-30)
            @test r.α ≥ smallest_step(-2one(R), roundoff(one(R)))
            @test r.evaluations < max_evaluations(m)
            m isa Backtracking || @test r.code == SUCCESS
        end
        # Only a spent cap is a spent cap: φ′(0), three trips, and the merit at the step.
        r = search(Bisection(; maxiter = 3), Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R; α = 1e-3)
        @test r.code == LINESEARCH_FAILED
        @test r.evaluations == 1 + 3 + 1
        @test r.α > 0
    end
end

@testset "a large increase of the merit is a failure, never STALLED" begin
    for R in (Float32, Float64), step in (MeasuredSlope(), ExactStep())

        cliff = Line(α -> α > 0 ? 1 + 1000α : one(R), α -> -2 * one(R))
        steep = Line(α -> α > 0 ? 1 + 10α + R(1e6) * α^2 : one(R), α -> -2 * one(R))
        for lf in (cliff, steep), m in SEARCHES

            r = search(m, lf, R; step)
            @test r.code == LINESEARCH_FAILED
            @test r.φ > φ(lf, zero(R)) + roundoff(one(R))
        end
    end
end

@testset "contract 6 and the caller's αmax" for R in (Float32, Float64)
    far = Line(α -> (α - R(1e7))^2 / R(1e14), α -> 2(α - R(1e7)) / R(1e14))
    near = Line(α -> (α - 1)^2, α -> 2(α - 1))
    # A ceiling is not a failure: the minimising search stops at its own and reports a decrease.
    r = search(Bisection(), far, R)
    @test r.α == R(65536)
    @test r.code == SUCCESS
    @test r.φ == φ(far, r.α)
    @test search(Bisection(; αmax = Inf), far, R).α > 1e6
    # the method's own ceiling binds under a caller's ceiling that does not
    @test search(Bisection(), far, R; αmax = 1e6) === r

    for m in METHODS, ceiling in (10.0, 0.25, 0.005)

        w = Watched(far, R)
        r = search(m, w, R; αmax = ceiling)
        @test 0 < r.α ≤ R(ceiling)
        @test all(≤(R(ceiling)), w.atφ) && all(≤(R(ceiling)), w.atφ′)  # nothing above it
    end
    for m in METHODS
        # it binds below the minimiser, and on an ascent anchor
        @test 0 < search(m, near, R; αmax = 0.5).α ≤ R(0.5)
        @test 0 < search(m, Line(α -> (α + 1)^2, α -> 2(α + 1)), R; α = 4, αmax = 0.5).α ≤
              R(0.5)
        # a ceiling that does not bind changes nothing
        @test search(m, near, R; αmax = 1e6) === search(m, near, R)
        # a ceiling that is not a step is a caller error: nothing is evaluated
        for bad in (0.0, -1.0, NaN)
            w = Watched(near, R)
            r = search(m, w, R; αmax = bad)
            @test r.code == LINESEARCH_FAILED
            @test r.evaluations == 0 == evaluated(w)
            @test r.α > 0
        end
    end
    # a merit that only falls: the ceiling is the step
    for ceiling in (2.0, 1.0, 0.5, 0.05, 0.005)
        w = Watched(Line(α -> 1 - α, α -> -one(R)), R)
        r = search(Bisection(), w, R; αmax = ceiling)
        @test r.α == R(ceiling)
        @test r.code == SUCCESS
        @test r.φ == 1 - R(ceiling)
        @test r.evaluations ≤ 4
    end
end

@testset "the step floor is τ/|φ′(0)|" begin
    # φ = 1 - 2α + kα² accepts only steps below 2/k, far below √eps, and decreases by about 1/k
    # there, many times τ. A floor τ/(c₁|φ′(0)|) clamped to √eps lies above those steps.
    for (R, ks) in ((Float32, (1e4, 1e5, 1e6)), (Float64, (1e10, 1e11, 1e12))),
        k in ks, m in (Backtracking(), StrongWolfe()),
        step in (ExactStep(), MeasuredSlope())
        lf = Line(α -> 1 - 2α + R(k) * α^2, α -> -2 + 2R(k) * α)
        r = search(m, lf, R; step)
        @test r.code == SUCCESS
        @test r.φ - 1 ≤ -2 * R(1e-4) * r.α                  # the Armijo condition holds
    end
end

@testset "a subnormal anchor: τ and the step floor stay positive" begin
    for R in (Float32, Float64), m in SEARCHES, step in (ExactStep(), MeasuredSlope())
        s = nextfloat(zero(R))
        @test roundoff(s) > 0
        # a steep slope makes τ/|φ′(0)| underflow as well
        for slope in (-2s, -R(1e10))
            w = Watched(Line(α -> s + α, α -> slope), R)
            r = search(m, w, R; step)
            @test r.α > 0 && !(0 in w.atφ) && !(0 in filter(!iszero, w.atφ′))
            @test r.code != SUCCESS                 # the merit rises at every step
            # with the true slope the floor stops it, not the cap; the lying slope -1e10 puts the
            # floor at floatmin, 300 decades below the trial step, so the cap may end it first
            @test r.evaluations ≤ (slope == -2s ? 20 : max_evaluations(m))
        end
    end
end

@testset "the Armijo test on the difference keeps a demand below one ulp of φ₀" begin
    # φ(0) + demand rounds to φ(0) for a demand of eps/16, so the sum form accepts no decrease
    for R in (Float32, Float64)
        o, demand = one(R), -eps(R) / 16
        @test o + demand == o
        @test !sufficient_decrease(o, o, demand, zero(R))
        @test sufficient_decrease(prevfloat(o), o, demand, zero(R))
    end
end

@testset "every exit of StrongWolfe" begin
    for R in (Float32, Float64)
        # A kink: φ′ jumps from -1 to 2 at α = a, so no step meets the curvature condition with
        # c₂ = 0.9. For a = 0.3 and 3e-5 the zoom collapses onto the kink with its lower end
        # above 0; for 3e5 the bracketing reaches the ceiling 2¹⁶. Neither is a success.
        vee(a) = Line(α -> α < a ? 1 - α : 1 - a + 2 * (α - a), α -> α < a ? -one(R) :
                                                                     2one(R))
        for a in (R(3e-5), R(0.3), R(3e5))
            lf = vee(a)
            r = search(StrongWolfe(; αmax = 65536.0), lf, R)
            φ₀ = one(R)
            @test r.code == LINESEARCH_FAILED
            @test r.φ < φ₀                                # the step still decreases the merit
            # the collapse and the ceiling end the search: no point is evaluated twice, which a
            # search that runs on to its cap does on a collapsed bracket, and far below the cap
            w = Watched(lf, R)
            @test search(StrongWolfe(; αmax = 65536.0), w, R) === r
            @test allunique(w.atφ)
            @test r.evaluations ≤ 100
            a < 1 && @test r.α ≈ a rtol = 8eps(R)
        end
        # A C¹ wall at α = 0.5: the steps with the curvature condition lie within 1e-8 above it
        # in Float64. The 0.66 safeguard bisects a bracket that does not shrink.
        K = R == Float64 ? 1e8 : 1e4
        wall = Line(α -> 1 - α + R(K) * max(α - R(0.5), 0)^2,
            α -> -1 + 2R(K) * max(α - R(0.5), 0))
        r = search(StrongWolfe(), wall, R)
        @test r.code == SUCCESS
        @test strong_wolfe(wall, r.α, one(R), -one(R), R(1e-4), R(0.9))
        # the safeguard bisects at least every other trial: about 2 log₂(0.5 / 1e-8) ≈ 51
        # trials of at most 2 evaluations bound it
        @test r.evaluations ≤ 80
        # the step floor: a merit one ulp above φ₀ at every step
        noise = Line(α -> α > 0 ? nextfloat(one(R)) : one(R), α -> -2one(R))
        r = search(StrongWolfe(), noise, R)
        @test r.code == STALLED
        @test r.α ≥ smallest_step(-2one(R), roundoff(one(R)))
        # the ceiling exit at the round-off floor: at α = 1.5eps the decrease 3eps is below τ
        r = search(StrongWolfe(), Line(α -> 1 - 2α, α -> -2one(R)), R; αmax = 1.5eps(R))
        @test r.α == R(1.5eps(R))
        @test r.code == STALLED
        # and a decrease at the ceiling that fails the curvature condition is a failure
        r = search(StrongWolfe(), Line(α -> 1 - 2α, α -> -2one(R)), R; αmax = 0.25)
        @test r.code == LINESEARCH_FAILED && r.φ < 1
    end
end

@testset "Eisenstat and Walker's η(α) = |1 - α| + α η₀" begin
    # F(x) = x² - 2 with the direction (1 + η₀) d_N, which overshoots the Newton step d_N:
    # ‖F + J d‖ = η₀ ‖F‖ exactly. Every trial the search rejects fails the test with η(α), and
    # the step it returns meets it.
    for R in (Float32, Float64)
        x = R[0.1, 0.2, 0.15]
        η₀ = R(0.5)
        newton = NewtonLine(x)
        lf = Watched(NewtonLine(x, (1 + η₀) .* newton.d), R)
        φ₀ = φ(lf.lf, zero(R))
        r = linesearch(inR(R, Backtracking()), lf, InexactStep(η₀), φ₀, one(R))
        @test r.code == SUCCESS
        @test length(lf.atφ) ≥ 2                     # it backtracks
        @test isempty(lf.atφ′)
        for (k, α) in enumerate(lf.atφ)
            η = abs(1 - α) + α * η₀
            meets = η < 1 && sqrt(φ(lf.lf, α)) ≤ (1 - R(1e-4) * (1 - η)) * sqrt(φ₀)
            @test meets == (k == length(lf.atφ))
        end
    end
    # beyond α = 1 the bound is |1 - α| + α η₀, not 1 - α(1 - η₀): on the linear model of this
    # direction the residual at α = 1.5 is |1 - 1.5(1 + η)| = 0.65 of ‖r‖, and the step passes
    for R in (Float32, Float64)
        η = R(0.1)
        linear = Line(α -> (1 - α * (1 + η))^2, α -> -2 * (1 + η) * (1 - α * (1 + η)))
        r = search(Backtracking(; c₁ = 0.5), linear, R; step = InexactStep(η), α = 1.5)
        @test r.α == R(1.5)
        @test r.code == SUCCESS
        @test r.evaluations == 1
    end
end

@testset "a slope passed as a number costs no evaluation" begin
    for R in (Float32, Float64), m in SEARCHES

        w = Watched(Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R)
        r = search(m, w, R; step = -2.0)
        @test all(>(0), w.atφ′)
        rm = search(m, w.lf, R)
        @test (r.α, r.φ, r.code, r.evaluations + 1) == (rm.α, rm.φ, rm.code, rm.evaluations)
    end
end

@testset "Bisection does not grow into a non-finite region" begin
    for R in (Float32, Float64)
        trap = Line(α -> α > R(0.5) ? R(Inf) : (α - 1)^2, α -> α > R(0.5) ? -R(Inf) :
                                                               2 * (α - 1))
        r = search(Bisection(), trap, R; α = 0.25)
        @test r.α ≤ R(0.5)
        @test isfinite(r.φ) && r.φ < 1
    end
end

@testset "the sign test does not underflow" begin
    for R in (Float32, Float64)
        s = R == Float64 ? 1e-200 : 1e-30
        a, b = R(s), -R(s)
        @test a * b == 0 && -zero(R) ≥ 0               # the product would say "same sign"
        @test !GeometricSolvers.samesign(a, b)
        @test GeometricSolvers.samesign(b, b)
        @test GeometricSolvers.samesign(zero(R), -one(R))
    end
end

@testset "the positional constructors check their parameters" begin
    @test_throws ArgumentError Bisection(-1.0, Int32(10))
    @test_throws ArgumentError Backtracking(0.5, 1.5, Int32(10))
    @test_throws ArgumentError StrongWolfe(0.9, 0.1, 1.0, Int32(10))
    @test_throws ArgumentError Backtracking{Float32}(1.0, 0.5, 10)
    # a merit that is not finite at the trial gives the shortest backtrack, 0.1α
    for R in (Float32, Float64), bad in (Inf, NaN)

        @test backtrack_step(one(R), -2one(R), one(R), R(bad), R(NaN), R(NaN), R(0.5)) ==
              R(0.1)
    end
end

@testset "an η outside [0, 1) is a failure that evaluates nothing" begin
    for R in (Float32, Float64), m in SEARCHES, η in (1.0, 1.5, -0.5, NaN, Inf)
        w = Watched(Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R)
        r = search(m, w, R; step = InexactStep(η), α = 0.7)
        @test r.code == LINESEARCH_FAILED
        @test r.evaluations == 0 == evaluated(w)
        @test r.α == R(0.7)
    end
end

@testset "Backtracking from floatmax shrinks a non-finite merit in log space" begin
    # φ is Inf at floatmax; the next trial is the geometric mean of the step and the floor, where
    # φ is finite: 9e15 in Float32, 2.8e146 in Float64. From there the 0.1α safeguard allows one
    # decade per backtrack. So Float32 needs 16 more trials, and Float64 needs 147, which the
    # default cap of 100 does not hold: that search reports a spent cap.
    line = Line(α -> (α - 1)^2, α -> 2 * (α - 1))
    r = search(Backtracking(), line, Float32; α = floatmax(Float32))
    @test r.code == SUCCESS
    @test r.evaluations ≤ 1 + 1 + 1 + 16 + 2
    r = search(Backtracking(), line, Float64; α = floatmax(Float64))
    @test r.code == LINESEARCH_FAILED
    @test r.evaluations == 1 + 100
    r = search(Backtracking(; maxiter = 200), line, Float64; α = floatmax(Float64))
    @test r.code == SUCCESS
    @test r.evaluations ≤ 1 + 1 + 1 + 147 + 2
end

@testset "a lying φ₀ below every merit gives no success" begin
    for R in (Float32, Float64), m in SEARCHES, step in (ExactStep(), MeasuredSlope())
        lf = Line(α -> (α - 1)^2 + 1, α -> 2 * (α - 1))
        r = linesearch(inR(R, m), lf, stepR(R, step), R(0.5), one(R))
        @test r.code != SUCCESS
    end
end

@testset "no point is evaluated twice, and a larger cap changes nothing" begin
    for R in (Float32, Float64)
        o, c = one(R), R(0.7)
        lines = (
            (α -> (α - c)^2, α -> 2(α - c), 1.0), (α -> (α - c)^2, α -> 2(α - c), 0.5),
            (α -> (α - 1)^2, α -> 2(α - 1), 0.5), (α -> (α - 100)^2, α -> 2(α - 100), 1.0),
            (α -> 1 - 2α + 1000α^2, α -> -2 + 2000α, 1.0),
            (α -> (α - 1)^2, α -> -(α - R(0.3)) * (α - 2), 0.01), (
                α -> 1 - α, α -> -o, 1.0),
            (α -> α > 0 ? nextfloat(o) : o, α -> -2o, 1.0))
        for m in SEARCHES, (f, d, α) in lines

            w = Watched(Line(f, d), R)
            r = search(m, w, R; α)
            @test allunique(w.atφ) && allunique(w.atφ′)
            long = typeof(m)(
                (getfield(m, k) for k in fieldnames(typeof(m))[1:(end - 1)])...,
                400)
            @test search(long, Line(f, d), R; α) === r
        end
    end
end

@testset "the cost does not depend on the merit's scale" for R in (Float32, Float64)
    o = one(R)
    # Merits whose tests lie far from their thresholds, with the trial step: a minimiser past
    # the trial step, an overshoot by 1000 that the cubic model answers, a minimiser 100 or 11
    # times beyond it and one a thousand times below it, a cubic and an exponential.
    steady = ((α -> (α - 1)^2, α -> 2 * (α - 1), R(0.4)),
        (α -> 1 - 2α + 1000α^2, α -> -2 + 2000α, o),
        (α -> (α - 100)^2 / 10^4, α -> 2 * (α - 100) / 10^4, o),
        (α -> (α - 11)^2, α -> 2 * (α - 11), o),
        (α -> (α - R(1e-3))^2, α -> 2 * (α - R(1e-3)), o),
        (α -> 1 - 2α + 100α^2 + 50α^3, α -> -2 + 200α + 150α^2, o),
        (α -> exp(8α) - 9α, α -> 8exp(8α) - 9, o))
    decimal = R == Float32 ? (1e-8, 1e8) : (1e-200, 1e-8, 1e8, 1e200)
    e = R == Float64 ? 12 : 8
    random = R.(10 .^ (2e .* rand(Random.Xoshiro(20260925), 500) .- e))
    same(r, b) = (r.α, r.code, r.evaluations) === (b.α, b.code, b.evaluations)
    for m in SEARCHES, (f, d, α) in steady

        base = scaled(m, f, d, R, o; α)
        # a power of two changes nothing; any other scale changes the step by rounding only
        @test all(k -> same(scaled(m, f, d, R, R(2)^k; α), base), -30:30)
        rs = map(c -> scaled(m, f, d, R, c; α), (decimal..., random...))
        @test all(r -> r.code == base.code && r.evaluations == base.evaluations, rs)
        @test all(r -> isapprox(r.α, base.α; rtol = 16eps(R)), rs)
    end
    for m in SEARCHES
        # the Moré–Thuente functions, and the round-off-floor path: a merit one ulp above φ₀
        # and a cliff. A power of two scales them exactly; a decimal scale changes the relative
        # size of one ulp, so there only the code of the noise merit must agree.
        for kind in 1:6
            lf = MoreThuente(R, kind)
            b = scaled(m, α -> φ(lf, α), α -> φ′(lf, α), R, 1; α = 0.1)
            @test all(
                c -> same(scaled(m, α -> φ(lf, α), α -> φ′(lf, α), R, c; α = 0.1), b),
                (2.0^-20, 2.0^20))
        end
        noise = (α -> α > 0 ? nextfloat(o) : o, α -> -2o)
        cliff = (α -> α > 0 ? 1 + 1000α : o, α -> -2o)
        exact = R == Float32 ? (2.0^-60, 2.0^60) : (2.0^-600, 2.0^600)
        for (f, d) in (noise, cliff)
            b = scaled(m, f, d, R, 1)
            @test all(c -> same(scaled(m, f, d, R, c), b), exact)
            @test all(c -> scaled(m, f, d, R, c).code == b.code, decimal)
        end
        n₁ = scaled(m, cliff..., R, 1).evaluations
        @test all(c -> scaled(m, cliff..., R, c).evaluations == n₁, decimal)
    end
    # On the near-quadratic merit the cubic model is the quadratic itself, so the ladder is
    # 1, 0.1, 0.01 and then the minimiser 0.001: the stable form of the cubic step, where the
    # textbook form (-b + √(b² - 3a d₀)) / 3a cancels.
    for s in (1e-8, 1.0, 1e8)
        w = Watched(Line(α -> R(s) * (1 - 2α + 1000α^2), α -> R(s) * (-2 + 2000α)), R)
        @test search(Backtracking(), w, R).code == SUCCESS
        @test w.atφ ≈ R[1, 0.1, 0.01, 0.001] rtol = 64eps(R)
    end
    @test backtrack_step(o, -2o, R(0.1), R(10.8), o, R(999), R(0.5)) ≈ R(0.01)
    # a Float32 Newton merit near the top of its range, where an unscaled square overflows
    if R == Float32
        δ = -10 * atan(3.0f0)
        for m in SEARCHES, step in (MeasuredSlope(), ExactStep())

            counts = map((1.0f0, 2.0f0^32)) do c
                f = α -> (c * atan(3 + α * δ))^2
                d = α -> 2 * c^2 * atan(3 + α * δ) * δ / (1 + (3 + α * δ)^2)
                search(m, Line(f, d), Float32; step).evaluations
            end
            @test allequal(counts)
        end
    end
end

@testset "the exits and boundaries that other tests do not reach" for R in (Float32, Float64)
    o = one(R)
    # classify and floor_code at their boundaries: a change by exactly τ
    τ = roundoff(o)
    @test classify(o - τ, o, τ) == SUCCESS
    @test floor_code(o + τ, o, τ) == STALLED
    # Bisection: a bracket trial where φ′ is exactly 0 is the step: φ′(0), φ′(1), φ(1)
    r = search(Bisection(), Line(α -> (α - 1)^2, α -> 2 * (α - 1)), R; α = 1)
    @test (r.α, r.code, r.evaluations) == (1, SUCCESS, 3)
    # and so is a bisection trial where φ′ is exactly 0: φ′ at 0, 0.5, 1 and 0.75, then φ
    r = search(Bisection(), Line(α -> (α - R(0.75))^2, α -> 2 * (α - R(0.75))), R; α = 0.5)
    @test (r.α, r.code, r.evaluations) == (R(0.75), SUCCESS, 5)
    # StrongWolfe: a trial that meets the first condition but has a higher merit than the one
    # before it ends the bracketing; the zoom then finds the lower merit below it
    bump = Line(α -> α ≤ R(0.6) ? 1 - α : R(0.7), α -> -o)
    r = search(StrongWolfe(), bump, R; α = 0.5)
    @test r.α < 1 && r.φ < R(0.5)
    # Backtracking: the frozen stop needs two trials below √eps whose merit equals φ₀. In
    # Float32 a merit equal to φ₀ is accepted first, because below 1e-6 the demanded decrease
    # is already below τ; in Float64 it is not, and the frozen stop ends the search.
    w = Watched(Line(α -> α ≤ R(1e-6) ? o : 1 + α, α -> -2o), R)
    r = search(Backtracking(), w, R)
    @test r.code == STALLED
    R == Float64 && @test count(α -> α ≤ sqrt(eps(R)) && φ(w.lf, α) == o, w.atφ) ≥ 2
end

@testset "the round-off τ decides a decrease" begin
    # the merit falls by `ulps` ulps of φ₀ = 1 at its minimiser: 2 is the floor, 8 a decrease
    # from α = 4, StrongWolfe reaches the minimiser through its zoom, and classifies it there
    for T in (Float32, Float64), (ulps, code) in ((2, STALLED), (8, SUCCESS)),
        (m, α) in ((Backtracking(), 1), (StrongWolfe(), 1), (StrongWolfe(), 4))
        a = T(ulps) * eps(T)
        r = search(m, Line(α -> one(T) - 2a * α + a * α^2, α -> -2a + 2a * α), T; α)
        @test r.code == code
    end
end

# The tests of `SimpleSolvers/test/linesearch_tests.jl` for the four methods, where the design
# keeps the behaviour. Not ported: `Quadratic` and `BierlaireQuadratic`; the expansion
# phase of `Backtracking`, its `τ_ulps` key and its curvature warning, the `Float16` rows, the
# `Linesearch` and `LinesearchProblem` objects, `change_precision`, `bracket_minimum`,
# `triple_point_finder` and the message tests, none of which this package has.
@testset "ported from SimpleSolvers: $R" for R in (Float32, Float64)
    o = one(R)
    f(x) = x^2 - 1
    g(x) = 2x
    δx(x) = -g(x) / 2
    parabola(x₀) = Line(α -> f(x₀ + α * δx(x₀)), α -> g(x₀ + α * δx(x₀)) * δx(x₀))
    function iterate(m, x, n)
        for _ in 1:n
            x += search(m, parabola(x), R).α * δx(x)
        end
        x
    end
    tol = ∛(2eps(R))

    @testset "Static" begin
        @test Static() === Static(1.0)
        @test search(Static(), parabola(-3o), R; α = 0).α == 1
        @test search(Static(0.8), parabola(-3o), R; α = 0).α == R(0.8)
    end

    @testset "Bisection, Backtracking and StrongWolfe reach the minimiser" begin
        @test iterate(Bisection(), -3o, 1) ≈ 0 atol = tol
        @test iterate(Backtracking(), -3o, 20) ≈ 0 atol = tol
        @test iterate(StrongWolfe(), -3o, 20) ≈ 0 atol = tol
        for α₀ in (0.25, 0.5, 1.0, 2.0, 4.0)
            @test -3 + search(Bisection(), parabola(-3o), R; α = α₀).α * 3 ≈ 0 atol = tol
        end
    end

    @testset "Backtracking stall" begin
        r = search(Backtracking(), Line(α -> (α - 100)^2, α -> 2 * (α - 100)), R)
        @test r.α == 1
        @test r.code == SUCCESS
    end

    @testset "Backtracking: non-descent and stationary anchors are reported, not searched" begin
        ascent = Line(α -> α + 1, α -> o)
        r = search(Backtracking(), ascent, R; α = 0.7)
        @test r.code == LINESEARCH_FAILED
        @test r.α == R(0.7)
        @test r.α == search(StrongWolfe(), ascent, R; α = 0.7).α
        @test search(Backtracking(), Line(α -> -o, α -> zero(R)), R).code == STALLED
        # the slope contradicts the values; at the step floor the rise is within τ
        @test search(Backtracking(), Line(α -> 1 + α, α -> -2o), R).code != SUCCESS
    end

    @testset "Backtracking: stagnation at the merit's round-off floor" begin
        # frozen below α = 1e-6, increasing above: no α decreases it
        t = R(1e-6)
        w = Watched(Line(α -> α ≤ t ? o : 1 + α, α -> -2o), R)
        r = search(Backtracking(), w, R)
        @test r.code == STALLED
        @test length(w.atφ) < 43
        # the frozen stop counts only steps below √eps, where x + αd may round to x
        @test min(t, sqrt(eps(R))) / 10 < r.α ≤ sqrt(eps(R))
        # every α > 0 lands one ulp above φ₀: pure round-off, down to the floor 2eps
        w = Watched(Line(α -> α > 0 ? nextfloat(o) : o, α -> -2o), R)
        r = search(Backtracking(), w, R)
        @test r.code == STALLED
        @test length(w.atφ) ≤ 2 + ceil(Int, log2(1 / (2eps(R))))
        @test r.α == smallest_step(-2o, roundoff(o)) == 2eps(R)
    end

    @testset "no method accepts a step that increases the merit" begin
        for m in SEARCHES, step in (MeasuredSlope(), ExactStep())

            τ = roundoff(o)
            creep = Line(α -> o + (α > 0 ? τ / 2 : zero(R)), α -> -2o)
            @test search(m, creep, R; step).code != SUCCESS
        end
    end

    @testset "a merit equal to φ₀ at a large step is not the round-off floor" begin
        # φ = 1 - 2α + 6α² - 4α³ equals φ₀ at α = 1 and at α = 1/2, and has φ(0.21) ≈ 0.81
        for s in (1e-8, 1.0, 1e8), step in (MeasuredSlope(), ExactStep())

            lf = Line(α -> R(s) * (1 - 2α + 6α^2 - 4α^3), α -> R(s) * (-2 + 12α - 12α^2))
            r = search(Backtracking(), lf, R; step)
            @test r.code == SUCCESS
            @test r.φ < R(0.9) * R(s)
        end
    end

    @testset "roundoff, smallest_step and the interpolation" begin
        @test roundoff(o) == 4eps(R)
        @test roundoff(R(1e-20)) == 4eps(R) * R(1e-20)       # relative, not a count of ulps
        @test roundoff(-3o) == 12eps(R)
        τ = roundoff(o)
        # τ/|φ′(0)|, with no c₁ and no clamp above floatmin
        @test smallest_step(-2o, τ) == 2eps(R)
        @test smallest_step(-R(1e30), τ) == max(τ / R(1e30), floatmin(R))
        @test smallest_step(-R(1e-30), τ) == τ / R(1e-30)
        for (φα, αp, φp) in ((3.0, NaN, NaN), (3.0, 2.0, 5.0), (NaN, NaN, NaN), (
            Inf, 1.5, 2.0))
            @test 0.1 ≤ backtrack_step(o, -2o, o, R(φα), R(αp), R(φp), R(0.5)) ≤ 0.5
        end
        w = Watched(Line(α -> 1 - 2α + 1000α^2, α -> -2 + 2000α), R)
        r = search(Backtracking(), w, R)
        @test r.code == SUCCESS
        @test length(w.atφ) < 10
        @test r.evaluations == length(w.atφ) + 1
        @test r.α ≤ 0.5
    end

    @testset "sufficient decrease with the round-off allowance" begin
        c = R(1e-4)
        @test !sufficient_decrease(nextfloat(o), o, c * -2, zero(R))
        @test !sufficient_decrease(nextfloat(o), o, c * R(1e-13) * -2, zero(R))
        target = 1 + c * R(0.1) * -2
        @test !sufficient_decrease(nextfloat(target, 2), o, c * R(0.1) * -2, zero(R))
        @test sufficient_decrease(nextfloat(target, 2), o, c * R(0.1) * -2, 4eps(R))
        @test !sufficient_decrease(nextfloat(o), o, c * -2, 4eps(R))
        @test !sufficient_decrease(nextfloat(o), o, c * R(1e-13) * -2, 4eps(R))
        @test 1 + c * R(1e-13) * -2 == 1
        @test sufficient_decrease(o, o, c * R(1e-13) * -2, 4eps(R))
    end

    @testset "every method reports a decrease as one" begin
        lf = Line(α -> (α - R(0.7))^2, α -> 2 * (α - R(0.7)))
        for m in METHODS
            @test search(m, lf, R).code == SUCCESS
        end
    end

    @testset "the cap bounds the ladder, and a spent cap is not the floor" begin
        noise = Line(α -> α > 0 ? nextfloat(o) : o, α -> -2o)
        @test search(Backtracking(), noise, R).code == STALLED
        w = Watched(noise, R)
        r = search(Backtracking(; maxiter = 3), w, R)
        @test length(w.atφ) == 3
        @test r.code == LINESEARCH_FAILED
    end

    @testset "a Bisection converges onto a minimum, never onto a maximum" begin
        # φ′ has a minimum at 0.3 and a maximum at 2; the bracket stops at √eps relative width
        lf = Line(a -> (a - 1)^2, a -> -(a - R(0.3)) * (a - 2))
        for α₀ in (0.01, 1.0)
            r = search(Bisection(), lf, R; α = α₀)
            @test r.α ≈ R(0.3) rtol = 2sqrt(eps(R))
            @test r.code == SUCCESS
        end
        @test search(Bisection(), Line(a -> (a - 1)^2, a -> 2(a - 1)), R).α ≈ 1 atol = tol
    end

    @testset "a Bisection that cannot bracket never reports a floor" begin
        for lf in (Line(α -> (α - 1)^2, α -> -o), Line(α -> (α + 1)^2, α -> -o))
            r = search(Bisection(), lf, R; α = 0.01)
            @test r.code == LINESEARCH_FAILED
            @test r.α > 0
            @test r.φ == φ(lf, r.α)
        end
        forever = Line(α -> 1 - α, α -> -o)
        r = search(Bisection(), forever, R)
        @test r.code == SUCCESS
        @test r.α == R(65536)
        @test r.φ == 1 - r.α
        r = search(Bisection(; αmax = Inf), forever, R)
        @test r.code == LINESEARCH_FAILED
        @test r.φ == 1 - r.α
    end

    @testset "StrongWolfe line search (bracket + zoom)" begin
        lf = parabola(-3o)
        φ₀, d₀ = φ(lf, zero(R)), φ′(lf, zero(R))
        for α₀ in (0.1, 0.5, 1.0, 2.0)
            r = search(StrongWolfe(), lf, R; α = α₀)
            @test strong_wolfe(lf, r.α, φ₀, d₀, R(1e-4), R(0.9))
            @test r.code == SUCCESS
        end
        @test search(StrongWolfe(; c₂ = 1e-2), lf, R; α = 2.0).α ≈ 1 atol = sqrt(eps(R))
        @test search(StrongWolfe(), Line(a -> (a + 1)^2, a -> 2(a + 1)), R; α = 0.7).α ==
              R(0.7)
    end

    @testset "StrongWolfe reports a non-finite anchor instead of asserting" begin
        for lf in (Line(α -> R(NaN), α -> R(NaN)), Line(α -> 1 - α, α -> R(NaN)))
            sw = search(StrongWolfe(), lf, R; α = 0.7)
            bt = search(Backtracking(), lf, R; α = 0.7)
            @test sw.code == bt.code == NONFINITE
            @test sw.α == bt.α == R(0.7)
        end
    end

    @testset "Linesearch Integration Tests" begin
        # the ‖F‖² merit of a scalar Newton step on F(x) = eˣ(x³/2 - 5x² + 2x) + 2
        x₀ = R(-10 * rand(Random.Xoshiro(1234)))
        F(x) = exp(x) * (R(0.5) * x^3 - 5x^2 + 2x) + 2o
        J(x) = exp(x) * (R(0.5) * x^3 - 5x^2 + 2x) + exp(x) * (R(1.5) * x^2 - 10x + 2)
        d = -F(x₀) / J(x₀)
        lf = Line(α -> F(x₀ + α * d)^2, α -> 2F(x₀ + α * d) * J(x₀ + α * d) * d)
        r = search(Bisection(), lf, R)
        @test φ′(lf, r.α) ≈ 0 atol = tol
    end

    @testset "Linesearch T-consistency" begin
        for m in METHODS
            lf = Line(α -> (α - 2o)^2, α -> 2 * (α - 2o))
            @test search(m, lf, R).α isa R
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
    # the Moré–Thuente set, the cubics of the step-floor and equal-merit tests, the kinks, the
    # cliff, and a Bisection line whose lower end stays at 0; one kernel launch per type
    lines(R) = (
        ([MoreThuente(R, kind) for kind in 1:6 for _ in MT_STEPS],
            [R(α) for _ in 1:6 for α in MT_STEPS]),
        (
            [Cubic((one(R), -2one(R), R(k), zero(R))) for k in (1e4, 1e6, 1e10, 1e12)] ∪
            [Cubic((one(R), -2one(R), 6one(R), -4one(R)))] ∪
            [Cubic(R(s) .* (1, -2, 1000, 0)) for s in (1e-8, 1.0, 1e8)],
            ones(R, 8)),
        ([Kink(R(a)) for a in (3e-5, 0.3, 3e5)], ones(R, 3)),
        ([Cliff()], ones(R, 1)),
        ([Rising(), Rising()], R[1, 1e16]))
    for R in (Float32, Float64), m in (METHODS..., Bisection(; αmax = Inf)),
        step in (MeasuredSlope(), InexactStep(0.1), ExactStep(), -0.5),
        (lfs, α₀s) in lines(R)

        ls = inR(R, m)
        stepr = stepR(R, step)
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
