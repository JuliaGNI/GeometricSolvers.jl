# `Fallback` in R1, R2 and R3: construction, each non-`SUCCESS` code moving on to the next member,
# the restart of each member from `x₀`, the status of a success and of a total failure, a member's
# exception passing through, the refusal of a `ForwardDiff.Dual` input and of a
# `LeastSquaresProblem`, a `StationarityProblem`, and `isbits`, `@inferred`, zero allocations,
# `JLArray` under `allowscalar(false)` and `solve_batch!`.

using Adapt: Adapt
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: BLAS, diagind
using StaticArrays: StaticArray, SVector
using Test

using GeometricSolvers
using GeometricSolvers: CombinatorProblem, Fallback, LINESEARCH_FAILED, LeastSquaresProblem,
                        MAXITERS, NONFINITE, Newton, NonlinearProblem, Options, ReturnCode,
                        SINGULAR, STALLED,
                        SUCCESS, SVDFactorization, SolverStatus, StationarityProblem,
                        ToReal, fallback_member,
                        init, init_state, solve, solve!, solve!!, solve_batch!

include("../helpers/matrix.jl")

allowscalar(false)

# A member that ends its solve with `code` after `iterations` steps and `failures` step failures,
# with the residual norm `fnorm`. It moves the iterate by one in each entry (in place for an
# array), so that a member after it that did not start from `x₀` starts elsewhere.
struct FailWith
    code::ReturnCode
    iterations::Int32
    failures::Int32
    promoted::Bool
    fnorm::Float64
    stationary::Bool
end
function FailWith(code, iterations, failures, promoted, fnorm)
    FailWith(code, iterations, failures, promoted, fnorm, false)
end
FailWith(code) = FailWith(code, Int32(2), Int32(1), false, 7.0)

struct FailWithState end

GeometricSolvers.init_state(::FailWith, ::CombinatorProblem, x, p) = FailWithState()

bump(x::AbstractArray) = (x .+= one(eltype(x)); x)
bump(x::Union{StaticArray, Number}) = x .+ one(eltype(x))

function GeometricSolvers.solve!!(
        ::FailWithState, ::CombinatorProblem, x₀, m::FailWith, p,
        options::Options{R}) where {R <: Real}
    status = SolverStatus{R}(
        m.code, m.iterations, R(m.fnorm), R(1), m.failures, m.promoted, m.stationary)
    return bump(x₀), status, FailWithState()
end

# A member whose solve throws its own exception.
struct MemberError <: Exception end
Base.showerror(io::IO, ::MemberError) = print(io, "MemberError: the member's solve throws")
struct Throws end
GeometricSolvers.init_state(::Throws, ::CombinatorProblem, x, p) = FailWithState()
function GeometricSolvers.solve!!(
        ::FailWithState, ::CombinatorProblem, x₀, ::Throws, p,
        options::Options{R}) where {R <: Real}
    throw(MemberError())
end

const FAIL_CODES = (STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED)

struct R1QuadFbF{T}
    c1::T
    c2::T
end
(f::R1QuadFbF)(x, p) = SVector(x[1]^2 - f.c1, x[2]^2 - f.c2)

struct R3QuadFbF{T}
    c1::T
    c2::T
end
function (f::R3QuadFbF)(F, x, p)
    c = SVector(f.c1, f.c2)
    F .= x .* x .- c
    return F
end

struct R3QuadFbJ end
function (::R3QuadFbJ)(J, x, p)
    fill!(J, zero(eltype(J)))
    if J isa Matrix
        @inbounds for i in axes(J, 1)
            J[i, i] = 2 * x[i]
        end
    else
        view(J, diagind(J)) .= 2 .* x
    end
    return J
end

# The two iterates and problems of the file: `x² = (2, 3)` from `(1.5, 1.5)`.
r1_problem(::Type{T}) where {T} = NonlinearProblem(R1QuadFbF(T(2), T(3)))
r3_problem(::Type{T}) where {T} = NonlinearProblem(R3QuadFbF(T(2), T(3)), R3QuadFbJ())
start_value(::Type{T}) where {T} = SVector{2, T}(T(1.5), T(1.5))

# The member index that a solve with `m` from `x₀` ends on.
function member_of(prob, x₀, m; kwargs...)
    R = real(eltype(x₀))
    opt = Options(R; kwargs...)
    m_r = Adapt.adapt(ToReal{R}(), m)
    x = x₀ isa SVector ? x₀ : copy(x₀)
    _, _, state = solve!!(init_state(m_r, prob, x, nothing), prob, x, m_r, nothing, opt)
    return fallback_member(state)
end

function r1_fb_alloc(prob::P, x₀::X, m::M) where {P, X, M}
    solve(prob, x₀, m; max_iterations = 4)
    return @allocated solve(prob, x₀, m; max_iterations = 4)
end

function r3_fb_alloc(solver::S, x::X, x₀::X) where {S, X}
    copyto!(x, x₀)
    solve!(solver, x)
    copyto!(x, x₀)
    return @allocated solve!(solver, x)
end

@testset "Fallback construction and ToReal adaptation" begin
    @test Base.isexported(GeometricSolvers, :Fallback)
    @test !Base.isexported(GeometricSolvers, :fallback_member)
    @test_throws r"^ArgumentError: Fallback needs at least one member" Fallback()
    @test_throws r"^ArgumentError: Fallback needs at least one member" Fallback(())
    m = Fallback(Newton(), Newton(; globalization = DogLeg()))
    @test m.members === (Newton(), Newton(; globalization = DogLeg()))
    @test Fallback((Newton(), Newton())) === Fallback(Newton(), Newton())
    @test isbits(m)
    for T in REAL_ELTYPES
        am = @inferred Adapt.adapt(ToReal{T}(), m)
        @test am.members[1].regularization isa T
        @test am.members[2].regularization isa T
    end
end

@testset "Each non-SUCCESS code moves on to the next member, $T" for T in REAL_ELTYPES
    # the root to the residual tolerance of the default `Options`
    root = SVector(sqrt(T(2)), sqrt(T(3)))
    for code in FAIL_CODES
        m = Fallback(FailWith(code), Newton())
        x_s, st_s = solve(r1_problem(T), start_value(T), m)
        @test st_s.code === SUCCESS
        @test isapprox(x_s, root; rtol = sqrt(eps(T)))
        @test member_of(r1_problem(T), start_value(T), m) == 2
        x_a, st_a = solve(r3_problem(T), Vector(start_value(T)), m)
        @test st_a.code === SUCCESS
        @test isapprox(x_a, root; rtol = sqrt(eps(T)))
        @test member_of(r3_problem(T), Vector(start_value(T)), m) == 2
    end
    # a first member that succeeds ends the solve, with its own status and no call counts
    m = Fallback(Newton(; linsolve = GMRES()), FailWith(SINGULAR))
    x₀ = Vector(start_value(T))
    x, st = solve(r3_problem(T), x₀, m)
    x_solo, st_solo = solve(r3_problem(T), x₀, Newton(; linsolve = GMRES()))
    @test st.code === SUCCESS
    @test x == x_solo
    @test st_solo.f_calls > 0
    @test (st.f_calls, st.jvp_calls) === (Int16(0), Int16(0))
    @test (st.iterations, st.fnorm, st.stepnorm, st.step_failures) ===
          (st_solo.iterations, st_solo.fnorm, st_solo.stepnorm, st_solo.step_failures)
    @test member_of(r3_problem(T), x₀, m) == 1
    # the diagnostics of an SVD factor pass through from the member that succeeded
    svd = Newton(; linsolve = SVDFactorization())
    _, st = solve(r1_problem(T), start_value(T), Fallback(FailWith(STALLED), svd))
    _, st_solo = solve(r1_problem(T), start_value(T), svd)
    @test st_solo.rank === Int32(2)
    @test (st.rank, st.nullnorm, st.gradratio, st.stationary) ===
          (st_solo.rank, st_solo.nullnorm, st_solo.gradratio, st_solo.stationary)
    # the stationarity flag of the member that succeeded passes through
    m = Fallback(FailWith(STALLED), FailWith(SUCCESS, Int32(1), Int32(0), false, 0.5, true))
    _, st = solve(r1_problem(T), start_value(T), m)
    @test (st.code, st.stationary, st.iterations) === (SUCCESS, true, Int32(3))
end

@testset "Each member restarts from x₀, $T" for T in REAL_ELTYPES
    # two iterations, so that the result depends on the start and not only on the root
    kw = (; max_iterations = 2, f_abstol = zero(T), f_reltol = zero(T), x_abstol = zero(T),
        x_reltol = zero(T))
    second = Newton(; globalization = DogLeg())
    for code in FAIL_CODES
        m = Fallback(FailWith(code), second)
        x_solo, st_solo = solve(r1_problem(T), start_value(T), second; kw...)
        x, st = solve(r1_problem(T), start_value(T), m; kw...)
        @test x == x_solo
        @test st.code === st_solo.code === MAXITERS

        x₀ = Vector(start_value(T))
        x_solo, _ = solve(r3_problem(T), copy(x₀), second; kw...)
        x, _ = solve(r3_problem(T), copy(x₀), m; kw...)
        @test x == x_solo
        # `solve!!` restores an array `x₀` in place before the second member
        y = copy(x₀)
        m_r = Adapt.adapt(ToReal{T}(), m)
        state = init_state(m_r, r3_problem(T), y, nothing)
        y₊, _, _ = solve!!(state, r3_problem(T), y, m_r, nothing, Options(T; kw...))
        @test y₊ === y
        @test y == x_solo
    end
end

@testset "A total failure returns the last member's code with summed counts, $T" for T in REAL_ELTYPES
    m = Fallback(FailWith(STALLED, Int32(3), Int32(1), true, 5.0),
        FailWith(MAXITERS, Int32(4), Int32(0), false, 6.0),
        FailWith(SINGULAR, Int32(5), Int32(2), false, 9.0))
    for x₀ in (start_value(T), Vector(start_value(T)))
        prob = x₀ isa SVector ? r1_problem(T) : r3_problem(T)
        x, st = solve(prob, x₀, m)
        @test st isa SolverStatus{T}
        @test st.code === SINGULAR
        @test st.iterations === Int32(12)
        @test st.step_failures === Int32(3)
        @test st.promoted
        @test st.fnorm === T(9)
        @test (st.f_calls, st.jvp_calls) === (Int16(0), Int16(0))
        @test member_of(prob, x₀, m) == 3
        # the last member starts from x₀ too, and moves it once
        @test x == start_value(T) .+ one(T)
    end
    # the sums saturate at typemax(Int32)
    big = typemax(Int32)
    m = Fallback(FailWith(STALLED, big - Int32(1), big, false, 1.0),
        FailWith(SINGULAR, Int32(5), Int32(2), false, 1.0))
    _, st = solve(r1_problem(T), start_value(T), m)
    @test (st.iterations, st.step_failures) === (big, big)
    # promoted is or-ed: it is clear only when no member sets it
    m = Fallback(FailWith(STALLED), FailWith(SINGULAR))
    _, st = solve(r1_problem(T), start_value(T), m)
    @test !st.promoted
    # one member is that member's solve
    _, st = solve(r1_problem(T), start_value(T), Fallback(FailWith(NONFINITE)))
    @test (st.code, st.iterations, st.step_failures) === (NONFINITE, Int32(2), Int32(1))
end

@testset "A member that throws is not caught" begin
    for x₀ in (start_value(Float64), Vector(start_value(Float64)))
        prob = x₀ isa SVector ? r1_problem(Float64) : r3_problem(Float64)
        @test_throws r"^MemberError: the member's solve throws$" solve(
            prob, x₀, Fallback(Throws(), Newton()))
        @test_throws r"^MemberError: the member's solve throws$" solve(
            prob, x₀, Fallback(FailWith(STALLED), Throws()))
    end
    # a residual that throws in the first member's solve
    prob = NonlinearProblem((x, p) -> SVector(sqrt(x[1] - 10)))
    @test_throws r"^DomainError with -10\.0:\nsqrt was called with a negative real argument" solve(
        prob, SVector(0.0), Fallback(Newton(), Newton()))
end

struct FallbackDualTag end

@testset "Fallback refuses a ForwardDiff.Dual input and a LeastSquaresProblem" begin
    m = Fallback(Newton(), Newton(; globalization = DogLeg()))
    d(v) = ForwardDiff.Dual{FallbackDualTag}(v, 1.0)
    prob_s = NonlinearProblem((x, p) -> x .- p)
    prob_a = NonlinearProblem((r, x, p) -> (r .= x .- p; nothing))
    msg = r"^ArgumentError: solve does not accept ForwardDiff.Dual inputs for the combinator Fallback"
    @test_throws msg solve(prob_s, SVector(0.0, 0.0), m, SVector(d(1.0), d(2.0)))
    @test_throws msg solve(prob_a, [d(0.0), d(0.0)], m, [1.0, 2.0])
    x, st = solve(prob_s, SVector(0.0, 0.0), m, SVector(1.0, 2.0))
    @test st.code === SUCCESS
    @test x ≈ SVector(1.0, 2.0)

    lsq = LeastSquaresProblem((r, x, p) -> (r[1] = x[1] - 1; r[2] = x[2] - 2; nothing))
    lsq_msg = r"^ArgumentError: the combinator Fallback does not take a LeastSquaresProblem"
    @test_throws lsq_msg init(lsq, [0.0, 0.0], m)
    @test_throws lsq_msg solve(lsq, [0.0, 0.0], m)
    lsq_1 = LeastSquaresProblem((x, p) -> SVector(x - 1, 2x - 2))
    @test_throws lsq_msg solve(lsq_1, 0.0, m)
end

# ∇S(x) = x³ + x - 1 componentwise, whose real root is 0.6823278038280193
@testset "Fallback on a StationarityProblem, $(nameof(typeof(x₀)))" for x₀ in (
    SVector(0.1, 0.2), [0.1, 0.2])
    sp = StationarityProblem((x, p) -> sum(abs2, x) / 2 - sum(x) + sum(x .^ 4) / 4)
    x, st = solve(sp, x₀, Fallback(FailWith(STALLED), Newton()))
    @test st.code === SUCCESS
    @test isapprox(Array(x), fill(0.6823278038280193, 2); rtol = 1e-10)
end

@testset "R1/R2/R3 parity, isbits, @inferred, 0 allocs, $T" for T in REAL_ELTYPES
    prob_s = r1_problem(T)
    prob_a = r3_problem(T)
    x₀_s = start_value(T)
    x₀_a = Vector(x₀_s)
    m = Fallback(Newton(; globalization = Static(), linsolve = QRFactorization()),
        Newton(; globalization = DogLeg()))
    stalls = Fallback(FailWith(STALLED), Newton(; globalization = DogLeg()))
    m_r = Adapt.adapt(ToReal{T}(), m)
    st_s = @inferred init_state(m_r, prob_s, x₀_s, nothing)
    @test isbits(st_s)
    @test fallback_member(st_s) == 0

    kw = (; max_iterations = 2, f_abstol = zero(T), f_reltol = zero(T), x_abstol = zero(T),
        x_reltol = zero(T))
    xs, sts = @inferred solve(prob_s, x₀_s, m; kw...)
    xa, sta = solve(prob_a, copy(x₀_a), m; kw...)
    xjl, stjl = solve(prob_a, JLArray(x₀_a), m; kw...)
    @test xs != x₀_s
    @test isapprox(SVector{2, T}(xa), xs; rtol = 8 * eps(T))
    @test isapprox(Array(xjl), xa; rtol = 8 * eps(T))
    @test sts.iterations == sta.iterations == stjl.iterations == 4
    xjl, stjl = solve(prob_a, JLArray(x₀_a), stalls)
    xa, _ = solve(prob_a, copy(x₀_a), Newton(; globalization = DogLeg()))
    @test stjl.code === SUCCESS
    @test isapprox(Array(xjl), xa; rtol = 8 * eps(T))
    @test isapprox(xa, [sqrt(T(2)), sqrt(T(3))]; rtol = sqrt(eps(T)))

    if Base.JLOptions().check_bounds != 1
        # Skipped where a run forces `--check-bounds=yes`.
        @test r1_fb_alloc(prob_s, x₀_s, m_r) == 0
    end
    JET.@test_opt solve(prob_s, x₀_s, m_r; max_iterations = 4)

    old_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        x_work = copy(x₀_a)
        solver = init(prob_a, x_work, m; kw...)
        @test r3_fb_alloc(solver, x_work, x₀_a) == 0
    finally
        BLAS.set_num_threads(old_threads)
    end

    X_batch = [x₀_s .+ T(k) / 10 for k in 1:4]
    S_batch = Vector{SolverStatus{T}}(undef, 4)
    solve_batch!(X_batch, S_batch, prob_s, stalls)
    @test all(s -> s.code == SUCCESS, S_batch)
    @test all(x -> isapprox(x, SVector(sqrt(T(2)), sqrt(T(3))); rtol = sqrt(eps(T))), X_batch)
    counts = solve_batch!([x₀_s for _ in 1:4], prob_s, m)
    @test counts.SUCCESS == 4
end
