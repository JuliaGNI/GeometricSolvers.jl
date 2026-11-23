# The Krawczyk root certificate of `GeometricSolvers.certify`, whose method comes from the
# extension `GeometricSolversIntervalArithmeticExt`.
#
# A certificate holds where the standard problem has an exact root in the iterate's element type:
# problems 1, 4, 17 and 22. Each enclosure must contain the root in `BigFloat`. The starts `u0` of
# those problems, and problem 20 at its double root `x = 5`, must not certify.

using ForwardDiff: ForwardDiff
using IntervalArithmetic: Interval, diam, in_interval, isbounded, isentire_interval,
                          isguaranteed
using StaticArrays: MVector, SMatrix, StaticVector, SVector
using Test

using GeometricSolvers
using GeometricSolvers: NonlinearProblem, certify

include("../helpers/matrix.jl")
include("../helpers/standard_problems.jl")

const CERTIFIED_PROBLEMS = (1, 4, 17, 22)

function standard_problem(number)
    return only(filter(sp -> sp.number == number, STANDARD_PROBLEMS))
end

encloses(enclosure::Interval, root) = in_interval(root, enclosure)
encloses(enclosure, root) = all(in_interval.(root, enclosure))

# The result type of a certificate whose enclosure has the type `enclosure`.
function result_type(enclosure)
    NamedTuple{(:certified, :enclosure, :guaranteed),
        Tuple{Bool, enclosure, Bool}}
end

@testset "a standard root certifies, and its enclosure holds the BigFloat root" begin
    for T in REAL_ELTYPES, number in CERTIFIED_PROBLEMS

        sp = standard_problem(number)
        x = sp.root(T)
        res = @inferred certify(sp.prob, x)
        @test res isa result_type(Vector{Interval{T}})
        @test res.certified
        @test length(res.enclosure) == sp.n
        @test encloses(res.enclosure, sp.root(BigFloat))
        # Every standard residual holds an integer constant, which IntervalArithmetic converts
        # without a guarantee, so the certificate holds without one.
        @test !res.guaranteed
        # The enclosure is the Krawczyk image, not the whole line. At an exact root the first box
        # has the radius of one `eps` of the root, and these roots certify on the second or third
        # box; the measured widths are 1.5 to 4 `eps(T)`, a factor 4 below the bound.
        @test all(isbounded, res.enclosure)
        @test all(c -> diam(c) <= 16 * eps(T), res.enclosure)
    end
end

@testset "a start and a double root do not certify" begin
    for T in REAL_ELTYPES
        for number in CERTIFIED_PROBLEMS
            sp = standard_problem(number)
            # No box forms an interval with NaN bounds, which IntervalArithmetic warns of.
            res = @test_logs @inferred certify(sp.prob, sp.u0(T))
            @test res isa result_type(Vector{Interval{T}})
            @test !res.certified
            @test !res.guaranteed
            @test all(isentire_interval, res.enclosure)
        end
        # Problem 20 has the double root 5, where its Jacobian vanishes: the midpoint Jacobian is
        # singular, and the certificate refuses without a throw.
        sp20 = standard_problem(20)
        res = @inferred certify(sp20.prob, T[5])
        @test res isa result_type(Vector{Interval{T}})
        @test !res.certified
        @test !res.guaranteed
        # The refusal comes before the first box: no interval Jacobian is evaluated.
        boxes = Ref(0)
        function F20!(y, x, p)
            eltype(x) <: ForwardDiff.Dual{<:Any, <:Interval} && (boxes[] += 1)
            return sp20.prob.F(y, x, p)
        end
        @test !certify(NonlinearProblem(F20!), T[5]).certified
        @test boxes[] == 0
    end
end

# The R1 forms: a scalar residual `x (x - 5)²`, the residual of problem 20, and an out-of-place
# version of problem 17.
p20_value(x, p) = x * (x - 5)^2
p17_value(x, p) = SVector(x[1] + x[2] - 3, x[1]^2 + x[2]^2 - 9)

@testset "a Number and a StaticVector certify in R1" begin
    for T in REAL_ELTYPES
        prob = NonlinearProblem(p20_value)
        res = @inferred certify(prob, zero(T))
        @test res isa result_type(Interval{T})
        @test res.certified
        @test encloses(res.enclosure, big(0))
        # The derivative vanishes at 5, so its inverse is infinite, and the test refuses it before
        # IntervalArithmetic sees it.
        res5 = @test_logs @inferred certify(prob, T(5))
        @test res5 isa result_type(Interval{T})
        @test !res5.certified

        prob17 = NonlinearProblem(p17_value)
        for x in (SVector{2, T}(0, 3), MVector{2, T}(0, 3))
            res = @inferred certify(prob17, x)
            @test res.certified
            @test res.enclosure isa StaticVector{2, Interval{T}}
            @test encloses(res.enclosure, BigFloat[0, 3])
            @test !(@inferred certify(prob17, 0 .* x)).certified
        end
    end
end

@testset "a given Jacobian is not read" begin
    # `Y` and `J(X)` come from `ForwardDiff` on `F`; the `J` of the problem is never called.
    for T in REAL_ELTYPES
        # `x - 1` off its root, with a constant `J` whose inverse is not finite or not 1.
        for c in (T(NaN), T(Inf), zero(T))
            res = @inferred certify(NonlinearProblem((x, p) -> x - 1, (x, p) -> c),
                one(T) + T(1e-3))
            @test res.certified
            @test encloses(res.enclosure, big(1))
        end
        # `x² + 10⁻¹²` has no real root. Its `J` here is the constant `2·10⁻³`, the derivative at
        # the iterate `10⁻³`: as `J(X)`, it would make `I - Y J(X)` vanish, and `K(X)` a point in
        # the interior of the box.
        cases = (
            (NonlinearProblem((x, p) -> x^2 + T(1e-12), (x, p) -> T(2e-3) + 0 * x),
                T(1e-3)),
            (
                NonlinearProblem((x, p) -> x .^ 2 .+ T(1e-12),
                    (x, p) -> SMatrix{1, 1}(T(2e-3) + 0 * x[1])),
                SVector(T(1e-3))))
        for (prob, x) in cases
            res = @test_logs @inferred certify(prob, x)
            @test !res.certified
            @test all(isentire_interval, res.enclosure)
        end
        # `x² - 2` at the rounded root √2, whose root is irrational, with a `J` that vanishes
        # everywhere: as `Y`, its inverse would be infinite.
        F(x, p) = x^2 - 2
        res = @inferred certify(NonlinearProblem(F, (x, p) -> zero(x)), sqrt(T(2)))
        @test res.certified
        @test encloses(res.enclosure, sqrt(big(2)))
        # The same in place, on a host `Vector`, where a singular `J` would refuse.
        F!(y, x, p) = (y .= x .^ 2 .- p; nothing)
        Z!(A, x, p) = (A .= 0; nothing)
        p = T[2, 3]
        res = @inferred certify(NonlinearProblem(F!, Z!), sqrt.(p), p)
        @test res isa result_type(Vector{Interval{T}})
        @test res.certified
        @test encloses(res.enclosure, sqrt.(big.(p)))
    end
end

@testset "guaranteed reads the NG flags of F(x) and J(X)" begin
    # At the root 1 of `x² - x`: written `x * x - x`, neither evaluation converts a number, and
    # the certificate is guaranteed (along with every entry of `enclosure`). A `+ 0` converts the
    # integer in `F(x)` only, and `x^2` converts the integer `2` of its derivative `2x` in `J(X)`
    # only.
    for T in REAL_ELTYPES
        cases = (((x, p) -> x * x - x, true), ((x, p) -> x * x - x + 0, false),
            ((x, p) -> x^2 - x, false))
        for (F, guaranteed) in cases
            for (prob, x, root) in (
                (NonlinearProblem(F), one(T), big(1)),
                (NonlinearProblem((x, p) -> SVector(F(x[1], p))), SVector(one(T)),
                BigFloat[1]),
                (NonlinearProblem((y, x, p) -> (y[1] = F(x[1], p); nothing)), [one(T)],
                BigFloat[1]))
                res = @inferred certify(prob, x)
                @test res.certified
                @test res.guaranteed == guaranteed
                @test all(isguaranteed, res.enclosure) == guaranteed
                @test encloses(res.enclosure, root)
            end
        end
    end
end

@testset "a residual with no real root reaches the box cap" begin
    # `2 + cos x ≥ 1` has no real root. Its Jacobian at `1` is regular, and bounded by 1 on every
    # box, so each radius is at most about 6 plus 4.4 times the one before, all 10 boxes stay
    # finite, and only the cap ends the search. A scalar box takes one call of `F` on a
    # dual number of intervals, so the calls count the boxes.
    for T in REAL_ELTYPES
        boxes = Ref(0)
        function F(x, p)
            x isa ForwardDiff.Dual{<:Any, <:Interval} && (boxes[] += 1)
            return 2 + cos(x)
        end
        res = @inferred certify(NonlinearProblem(F), one(T))
        @test res isa result_type(Interval{T})
        @test !res.certified
        @test !res.guaranteed
        @test boxes[] == 10
        res = @inferred certify(NonlinearProblem((x, p) -> SVector(2 + cos(x[1]))),
            SVector(one(T)))
        @test !res.certified
    end
end

@testset "an exact root certifies on the second box" begin
    # At the root 1 of `x * x - x`, `F(x) = 0`, so the first box has the radius `eps(T)`, whose
    # upper end `1 + eps(T)` is also the upper end of `K(X)` rounded outward. The second box is
    # wider and holds `K(X)` in its interior. Without the `eps` term of the first radius, the first
    # box would be the point 1, and the certificate would take a third box.
    for T in REAL_ELTYPES
        boxes = Ref(0)
        function F(x, p)
            x isa ForwardDiff.Dual{<:Any, <:Interval} && (boxes[] += 1)
            return x * x - x
        end
        @test certify(NonlinearProblem(F), one(T)).certified
        @test boxes[] == 2
    end
end

@testset "an iterate off the root certifies, and its enclosure holds the root" begin
    # Off the root, `F(x)` is not zero, and the Newton term `-Y F(x)` moves the enclosure from
    # the iterate to the root. With the sign of that term reversed, `x - 1` at `1 + 1/1000`
    # certifies the point `1 + 2/1000`, which does not hold the root.
    for T in REAL_ELTYPES
        res = @inferred certify(NonlinearProblem((x, p) -> x - 1), one(T) + T(1e-3))
        @test res.certified
        @test encloses(res.enclosure, big(1))
        for number in CERTIFIED_PROBLEMS
            sp = standard_problem(number)
            res = @inferred certify(sp.prob, sp.root(T) .+ T(1e-4))
            @test res.certified
            @test encloses(res.enclosure, sp.root(BigFloat))
        end
    end
end

# `x - p` with the term `0 * exp(x²)` in intervals: on the box around 30, `exp(x²)` overflows to an
# unbounded interval, and its product with 0 is the point 0 decorated `dac`, in `F(X)` and in
# `J(X)`. In floats the term is left out, where `0 * Inf` would be `NaN`.
dac_residual(x, p) = x - p
function dac_residual(x::Union{Interval, ForwardDiff.Dual{<:Any, <:Interval}}, p)
    return x - p + 0 * exp(x * x)
end

@testset "a residual undefined, not continuous or with a kink on the box does not certify" begin
    # IntervalArithmetic decorates a result on arguments outside the domain of a function `trv`,
    # and a result across a discontinuity `def`, in `F(X)`; the derivative of a kink is a jump,
    # decorated `def` in `J(X)`. Krawczyk's theorem needs `F` and its Jacobian continuous on the
    # box.
    for T in REAL_ELTYPES
        cases = (
            # `log` and `sqrt` are undefined on the negative part of the box around 0, and `1/x`
            # at 0, where `F(x)` is empty.
            (NonlinearProblem((x, p) -> log(x)), zero(T)),
            (NonlinearProblem((x, p) -> log.(x)), SVector(zero(T))),
            (NonlinearProblem((x, p) -> sqrt(x) + 1), zero(T)),
            (NonlinearProblem((x, p) -> 1 / x - 1), zero(T)),
            (NonlinearProblem((y, x, p) -> (y[1] = 1 / x[1] - 1; nothing)), [zero(T)]),
            # `x + floor(x) + 1/2` is `x - 1/2 < 0` on `[-1, 0)` and `x + 1/2 > 0` on `[0, 1)`:
            # no root. `F(x)` at the iterate and the `ForwardDiff` derivative `1` show no jump;
            # only `F(X)` sees it.
            (NonlinearProblem((x, p) -> x + floor(x) + T(1 / 2)), zero(T)),
            (NonlinearProblem((x, p) -> x .+ floor.(x) .+ T(1 / 2)), SVector(zero(T))),
            (NonlinearProblem((y, x, p) -> (y .= x .+ floor.(x) .+ T(1 / 2); nothing)),
                [zero(T)]),
            # A kink at the root, in a residual that is continuous: only `J(X)` sees it.
            (NonlinearProblem((x, p) -> abs(x) + 2x), zero(T)),
            (NonlinearProblem((x, p) -> abs(x - T(0.1)) + 2x - T(0.2)), T(0.1)))
        for (prob, x) in cases
            res = @test_logs @inferred certify(prob, x)
            @test !res.certified
            @test !res.guaranteed
            @test all(isentire_interval, res.enclosure)
        end
        # `dac` is enough.
        res = @test_logs @inferred certify(NonlinearProblem(dac_residual), T(30), T(30))
        @test res.certified
        @test encloses(res.enclosure, big(30))
        # `F` is called at the float iterate, and an error of `F` there is not caught.
        @test_throws r"^DomainError with -1\.0:\nlog was called with a negative real" certify(
            NonlinearProblem((x, p) -> log(x)), -one(T))
    end
end

@testset "hostile iterates return no certificate and do not throw" begin
    for T in REAL_ELTYPES
        prob = NonlinearProblem(p20_value)
        for x in (T(NaN), T(Inf), -T(Inf), floatmax(T))
            res = @inferred certify(prob, x)
            @test res isa result_type(Interval{T})
            @test !res.certified
            @test !res.guaranteed
        end
        sp = standard_problem(17)
        # The derivative of `x - p` stays 1 at a non-finite iterate, so `Y` is finite. The test of
        # the iterate refuses it before IntervalArithmetic sees it, which would warn of an
        # ill-formed interval.
        linear = NonlinearProblem((x, p) -> x - p)
        for bad in (T(NaN), T(Inf), -T(Inf))
            res = @inferred certify(sp.prob, T[0, bad])
            @test res isa result_type(Vector{Interval{T}})
            @test !res.certified
            res = @test_logs certify(linear, bad, one(T))
            @test !res.certified
            @test all(isentire_interval, res.enclosure)
        end
        # When `r` overflows (`-0.45 floatmax(T)` toward `0.45 floatmax(T)`), or when `x + r`
        # overflows with finite `r` (`floatmax(T)` on box 1, or `0.6 floatmax(T)` toward
        # `0.1 floatmax(T)` on box 2), the test refuses before it evaluates `F` on an unbounded box.
        for (x0, p0) in (
            (-T(0.45) * floatmax(T), T(0.45) * floatmax(T)),
            (floatmax(T), floatmax(T)),
            (T(0.6) * floatmax(T), T(0.1) * floatmax(T)))
            unbounded = Ref(0)
            function F_far(x, p)
                x isa Interval && !isbounded(x) && (unbounded[] += 1)
                return x - p
            end
            res = @test_logs @inferred certify(NonlinearProblem(F_far), x0, p0)
            @test !res.certified
            @test all(isentire_interval, res.enclosure)
            @test unbounded[] == 0
        end
        # `tanh((x - x0) / L) - 3/2 <= -1/2 < 0` has no real root anywhere, yet its `F(X)` and
        # `J(X)` stay bounded and `dac` on a half-infinite box `[a, ∞)`. When `x0 + r` overflows
        # `floatmax(T)`, the unbounded box must be refused without warnings rather than certifying.
        L = ldexp(one(T), exponent(floatmax(T)) - 3)
        x0 = T(14) * L
        F_unb(x, p) = tanh((x - x0) / L) - T(1.5)
        for (prob, x) in (
            (NonlinearProblem(F_unb), x0),
            (NonlinearProblem((x, p) -> SVector(F_unb(x[1], p))), SVector(x0)),
            (NonlinearProblem((x, p) -> SVector(F_unb(x[1], p), x[2])),
            SVector(x0, zero(T))),
            (NonlinearProblem((y, x, p) -> (y[1] = F_unb(x[1], p); nothing)), [x0]))
            res = @test_logs @inferred certify(prob, x)
            @test !res.certified
            @test !res.guaranteed
            @test all(isentire_interval, res.enclosure)
        end
        # Badly scaled systems where `Y`, `F(x)`, `X` and `J(X)` are all finite and bounded, while
        # `Y * J(X)` or `Y * F(x)` overflows: refuse cleanly on `Vector` with no `NaI` warnings.
        s = ldexp(one(T), 2 * exponent(floatmax(T)) ÷ 3)
        function F_overflow_M!(y, x, p)
            y[1] = inv(s) * x[1] + s * (x[2] - 1)^2
            y[2] = s * (x[2] - 1) + inv(s) * x[1]
            return nothing
        end
        res_M = @test_logs @inferred certify(NonlinearProblem(F_overflow_M!), T[1, 1])
        @test !res_M.certified
        @test !res_M.guaranteed
        @test all(isentire_interval, res_M.enclosure)
        F_overflow_c!(y, x, p) = (y[1] = inv(s) * x[1] - s; nothing)
        res_c = @test_logs @inferred certify(NonlinearProblem(F_overflow_c!), [zero(T)])
        @test !res_c.certified
        @test !res_c.guaranteed
        @test all(isentire_interval, res_c.enclosure)
        # An exact root certifies at each magnitude from zero to `floatmax(T) / 2`. The residual is
        # linear with the derivative 1, so `K(X)` is the point `s`, while every box is wider.
        roots = (zero(T), -zero(T), nextfloat(zero(T)), floatmin(T), eps(T), one(T),
            floatmax(T) / 2, (T(10)^k for k in -8:4:8)...)
        for s in roots
            res = certify(NonlinearProblem((x, p) -> x - p), s, s)
            @test res.certified
            @test encloses(res.enclosure, big(s))
            @test diam(res.enclosure) == 0
        end
    end
end

@testset "the iterates outside R1 and R3 on the CPU have no method" begin
    # The error is the one of `certify` itself, not of a function that it calls.
    sp = standard_problem(17)
    @test_throws r"^MethodError: no method matching certify\(" certify(
        sp.prob, ComplexF64[0, 3])
    @test_throws r"^MethodError: no method matching certify\(" certify(
        NonlinearProblem((x, p) -> x^2 - 1), 1.0 + 0im)
    @test_throws r"^MethodError: no method matching certify\(" certify(
        sp.prob, Float64[0 3; 3 0])
    @test_throws r"^MethodError: no method matching certify\(" certify(
        sp.prob, JLArray(Float64[0, 3]))
    @test_throws r"^MethodError: no method matching certify\(" certify(sp.prob, [0, 3])
    stat = StationarityProblem((x, p) -> sum(abs2, x), 1:2, 1:2)
    @test_throws r"^MethodError: no method matching certify\(" certify(stat, [0.0, 0.0])
    ls = LeastSquaresProblem((y, x, p) -> (y .= x; nothing), 2)
    @test_throws r"^MethodError: no method matching certify\(" certify(ls, [0.0, 0.0])
end

@testset "the method comes from the extension, and the stub is public" begin
    ext = Base.get_extension(GeometricSolvers, :GeometricSolversIntervalArithmeticExt)
    @test ext isa Module
    sp = standard_problem(17)
    @test which(certify, Tuple{typeof(sp.prob), Vector{Float64}}).module === ext
    @test Base.ispublic(GeometricSolvers, :certify)
    @test !Base.isexported(GeometricSolvers, :certify)
end

@testset "without IntervalArithmetic, certify has no method and nothing loads it" begin
    # A child process loads only the packages its code names. `JULIA_DEPOT_PATH` stays, so that
    # a child of a run in a custom depot finds its packages.
    julia_exe = joinpath(Sys.BINDIR, "julia")
    env_clean = filter(
        p -> (p.first == "JULIA_DEPOT_PATH" || !startswith(p.first, "JULIA_")) &&
             !startswith(p.first, "GIT_"),
        copy(ENV)
    )
    code = """
    using GeometricSolvers
    isempty(methods(GeometricSolvers.certify)) || error("certify has a method")
    Base.ispublic(GeometricSolvers, :certify) || error("certify is not public")
    any(k -> k.name == "IntervalArithmetic", keys(Base.loaded_modules)) &&
        error("IntervalArithmetic is loaded")
    """
    child = run(ignorestatus(setenv(
        `$julia_exe --startup-file=no --compiled-modules=existing --project=$(Base.active_project()) -e $code`,
        env_clean
    )))
    @test success(child)
end
