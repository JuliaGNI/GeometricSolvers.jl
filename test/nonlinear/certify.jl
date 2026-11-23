# The Krawczyk root certificate of `GeometricSolvers.certify`, whose method comes from the
# extension `GeometricSolversIntervalArithmeticExt`.
#
# A certificate holds where the standard problem has an exact root in the iterate's element type:
# problems 1, 4, 17 and 22. Each enclosure must contain the root in `BigFloat`. The starts `u0` of
# those problems, and problem 20 at its double root `x = 5`, must not certify.

using ForwardDiff: ForwardDiff
using IntervalArithmetic: Interval, diam, in_interval, isbounded, isentire_interval
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
        # box; the measured widths are 1.5 to 4 `eps(T)`, a factor 4 below the
        # bound.
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

@testset "an analytic Jacobian is the one that is read" begin
    for T in REAL_ELTYPES
        # `x² - 2` at the rounded root √2: the root is irrational, and the enclosure holds it.
        F(x, p) = x^2 - 2
        good = NonlinearProblem(F, (x, p) -> 2x)
        res = @inferred certify(good, sqrt(T(2)))
        @test res.certified
        @test encloses(res.enclosure, sqrt(big(2)))
        # A Jacobian that vanishes everywhere makes the midpoint Jacobian singular.
        bad = NonlinearProblem(F, (x, p) -> zero(x))
        @test !(@inferred certify(bad, sqrt(T(2)))).certified

        # The in-place form, on a host `Vector`.
        F!(y, x, p) = (y .= x .^ 2 .- p; nothing)
        J!(A, x, p) = (A .= 0; for i in eachindex(x)
                A[i, i] = 2x[i]
            end; nothing)
        p = T[2, 3]
        res = @inferred certify(NonlinearProblem(F!, J!), sqrt.(p), p)
        @test res isa result_type(Vector{Interval{T}})
        @test res.certified
        @test encloses(res.enclosure, sqrt.(big.(p)))
        Z!(A, x, p) = (A .= 0; nothing)
        @test !(@inferred certify(NonlinearProblem(F!, Z!), sqrt.(p), p)).certified
    end
end

@testset "guaranteed reads the NG flags of F(x) and J(X)" begin
    # At the root 1 of `x² - x`: written `x * x - x`, neither evaluation converts a number, and
    # the certificate is guaranteed. A `+ 0` converts the integer in `F(x)` only, and `x^2`
    # converts the integer `2` of its derivative `2x` in `J(X)` only.
    for T in REAL_ELTYPES
        cases = (((x, p) -> x * x - x, true), ((x, p) -> x * x - x + 0, false),
            ((x, p) -> x^2 - x, false))
        for (F, guaranteed) in cases
            res = @inferred certify(NonlinearProblem(F), one(T))
            @test res.certified
            @test res.guaranteed == guaranteed
            @test encloses(res.enclosure, big(1))
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

dac_jacobian(x::Interval, p) = 1 + 0 * exp(x * x)
dac_jacobian(x::AbstractFloat, p) = one(x)

@testset "a residual undefined or not continuous on the box does not certify" begin
    # IntervalArithmetic marks a result on arguments outside the domain of a function with the
    # decoration `trv`, and on a discontinuity with `def`; `F(x)` can then be empty, and with it
    # `K(X)`, which the interior of every box holds.
    for T in REAL_ELTYPES
        cases = (
            # `F(x)` is empty: `log` and `sqrt` of a negative number, and `1/x` at 0.
            (NonlinearProblem((x, p) -> log(x), (x, p) -> 1 / x), -one(T)),
            (NonlinearProblem((x, p) -> log.(x), (x, p) -> SMatrix{1, 1}(1 / x[1])),
                SVector(-one(T))),
            (NonlinearProblem((x, p) -> sqrt(x) + 1, (x, p) -> one(x)), -T(4)),
            (NonlinearProblem((x, p) -> 1 / x - 1), zero(T)),
            # `F(x)` is defined, and `J(X)` is bounded: `sqrt` is undefined on the negative part
            # of the box around 0, and `floor` is discontinuous at 0.
            (NonlinearProblem((x, p) -> x, (x, p) -> 1 + 0 * sqrt(x)), zero(T)),
            (NonlinearProblem((x, p) -> x, (x, p) -> 1 + 0 * floor(x)), zero(T)))
        for (prob, x) in cases
            res = @test_logs @inferred certify(prob, x)
            @test !res.certified
            @test !res.guaranteed
            @test all(isentire_interval, res.enclosure)
        end
        # `dac` is enough: `0 * exp(x²)` overflows on the box around 30 to the product of 0 with
        # an unbounded interval, the point 0 decorated `dac`, so `J(X)` is the point 1 decorated
        # `dac`. The float method gives `J(x)` without the overflow.
        res = @test_logs @inferred certify(NonlinearProblem((x, p) -> x - p, dac_jacobian),
            T(30), T(30))
        @test res.certified
        @test encloses(res.enclosure, big(30))
        # A constant analytic Jacobian is a number, not an interval: a finite one certifies, an
        # infinite one does not.
        @test certify(NonlinearProblem((x, p) -> x - 1, (x, p) -> one(T)), one(T)).certified
        res = @test_logs certify(NonlinearProblem((x, p) -> x - 1, (x, p) -> T(Inf)), one(T))
        @test !res.certified
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
        # A constant analytic Jacobian stays finite and regular at a non-finite iterate. The test
        # of the iterate refuses it before IntervalArithmetic sees it, which would warn of an
        # ill-formed interval.
        linear = NonlinearProblem((x, p) -> x - p, (x, p) -> one(x))
        for bad in (T(NaN), T(Inf), -T(Inf))
            res = @inferred certify(sp.prob, T[0, bad])
            @test res isa result_type(Vector{Interval{T}})
            @test !res.certified
            res = @test_logs certify(linear, bad, one(T))
            @test !res.certified
            @test all(isentire_interval, res.enclosure)
        end
        # From `-0.45 floatmax(T)` the root `0.45 floatmax(T)` asks for a second radius of about
        # `1.8 floatmax(T)`, which overflows: the test refuses it.
        far = T(0.45) * floatmax(T)
        res = @test_logs @inferred certify(NonlinearProblem((x, p) -> x - p), -far, far)
        @test !res.certified
        @test all(isentire_interval, res.enclosure)
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
