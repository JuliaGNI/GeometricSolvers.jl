# `Switch` combinator in R1, R2 and R3: constructor validation, relative threshold against
# `‖F(x₀)‖ = 10³` (Check 5, MU7), permanent latch on forced residual rise (Check 5, MU8),
# un-started `step!!` returning `NONFINITE` (Check 5, MU9), `sync_merit` on transition to a
# line-search `second` member (D7), active-branch `finish!!` diagnostics (D8), and R1/R2/R3
# `isbits`, `@inferred`, zero allocations, `JLArray` under `allowscalar(false)`, and
# `solve_batch!` (Check 6).

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: BLAS, diagind
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, LUFactorization, NONFINITE, Newton, NonlinearProblem,
                        Picard, SVDFactorization, SUCCESS, SolverStatus, Static, Switch,
                        ToReal, finish!!, init, init_state, residual!!, rnorm, solve,
                        solve!, solve_batch!, solve!!, start!!, step!!

include("../helpers/matrix.jl")

allowscalar(false)

struct R1QuadSwF{T}
    c1::T
    c2::T
end
(f::R1QuadSwF)(x, p) = SVector(x[1]^2 - f.c1, x[2]^2 - f.c2)

struct R3QuadSwF{T}
    c1::T
    c2::T
end
function (f::R3QuadSwF)(F, x, p)
    c = SVector(f.c1, f.c2)
    F .= x .* x .- c
    return F
end

struct R3QuadSwJ end
function (::R3QuadSwJ)(J, x, p)
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

function r1_sw_alloc(prob::P, x₀::X, m::M) where {P, X, M}
    solve(prob, x₀, m; max_iterations = 2)
    return @allocated solve(prob, x₀, m; max_iterations = 2)
end

function r3_sw_alloc(solver::S, x::X, x₀::X) where {S, X}
    copyto!(x, x₀)
    solve!(solver, x)
    copyto!(x, x₀)
    return @allocated solve!(solver, x)
end

@testset "Switch constructor validation and ToReal adaptation" begin
    @test Base.isexported(GeometricSolvers, :Switch)
    for bad_th in (0, 0.0, -0.1, -1, NaN, Inf, -Inf)
        @test_throws ArgumentError Switch(Picard(), Newton(); threshold = bad_th)
        @test_throws ArgumentError Switch(Picard(), Newton(), bad_th)
    end

    sw_def = Switch(Picard(; damping = 0.2), Newton())
    @test sw_def.threshold === 1e-2
    @test Switch(Picard(; damping = 0.2), Newton(), 0.25).threshold === 0.25

    for T in REAL_ELTYPES
        asw = @inferred Adapt.adapt(
            ToReal{T}(), Switch(Picard(; damping = 0.2), Newton(); threshold = 0.25))
        @test asw.threshold isa T
        @test asw.threshold === T(0.25)
        @test asw.first.damping isa T
    end
end

@testset "Check 5: relative threshold, latch, fail-closed (MU7-MU9), $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem((x, p) -> SVector{2, T}(R(0.5) * x[1], R(0.5) * x[2]))
    x₀_s = SVector{2, T}(T(2000), zero(T)) # ||F(x0)|| = 1000

    m₁ = Adapt.adapt(ToReal{R}(), Picard(; damping = 1.0))
    m₂ = Adapt.adapt(ToReal{R}(), Picard(; damping = 1.5))
    m_sw = Adapt.adapt(ToReal{R}(), Switch(m₁, m₂; threshold = 0.3))

    st_sw = init_state(m_sw, prob_s, x₀_s, nothing)
    Fx₀_s = residual!!(nothing, prob_s, x₀_s, nothing)

    # Calling step!! without start!! fails closed with NONFINITE and stepnorm == 0 (MU9)
    x_u, F_u, _, info_u = @inferred step!!(st_sw, m_sw, prob_s, x₀_s, Fx₀_s, nothing)
    @test info_u.code === NONFINITE
    @test iszero(info_u.stepnorm)
    @test x_u === x₀_s
    @test F_u === Fx₀_s

    st_sw = start!!(st_sw, m_sw, prob_s, x₀_s, Fx₀_s, nothing, rnorm(Fx₀_s))
    @test st_sw.fnorm₀ === R(1000)
    @test !st_sw.switched

    # Standalone states for bit-for-bit comparison
    s₁ = start!!(
        init_state(m₁, prob_s, x₀_s, nothing),
        m₁, prob_s, x₀_s, Fx₀_s, nothing, rnorm(Fx₀_s))

    # Step 1: ||F(x0)|| = 1000 > 0.3 * 1000 = 300 -> runs m1 (||F(x1)|| = 500)
    x₁, F₁, st_sw, _ = @inferred step!!(st_sw, m_sw, prob_s, x₀_s, Fx₀_s, nothing)
    x₁_m1, F₁_m1, s₁, _ = step!!(s₁, m₁, prob_s, x₀_s, Fx₀_s, nothing)
    @test !st_sw.switched
    @test x₁ === x₁_m1
    @test F₁ === F₁_m1

    # Step 2: ||F(x1)|| = 500 > 300 -> still runs m1 (x2 = (500, 0), ||F(x2)|| = 250)
    x₂, F₂, st_sw, _ = @inferred step!!(st_sw, m_sw, prob_s, x₁, F₁, nothing)
    x₂_m1, F₂_m1, s₁, _ = step!!(s₁, m₁, prob_s, x₁_m1, F₁_m1, nothing)
    @test !st_sw.switched
    @test x₂ === x₂_m1
    @test F₂ === F₂_m1

    # Step 3: ||F(x2)|| = 250 <= 300 (even though 250 > 0.3, catching MU7) -> switches
    s₂ = start!!(
        init_state(m₂, prob_s, x₂, nothing), m₂, prob_s, x₂, F₂, nothing, rnorm(F₂))
    x₃, F₃, st_sw, _ = @inferred step!!(st_sw, m_sw, prob_s, x₂, F₂, nothing)
    x₃_m2, F₃_m2, s₂, _ = step!!(s₂, m₂, prob_s, x₂, F₂, nothing)
    @test st_sw.switched
    @test x₃ === x₃_m2
    @test F₃ === F₃_m2

    # Step 4: Force residual rise to ||F|| = 2000 > 300; latch stays on m2 (catches MU8)
    x_high = SVector{2, T}(T(4000), zero(T))
    F_high = residual!!(nothing, prob_s, x_high, nothing)
    x₄, F₄, st_sw, _ = @inferred step!!(st_sw, m_sw, prob_s, x_high, F_high, nothing)
    x₄_m2, F₄_m2, s₂, _ = step!!(s₂, m₂, prob_s, x_high, F_high, nothing)
    @test st_sw.switched
    @test x₄ === x₄_m2
    @test F₄ === F₄_m2
end

@testset "D7 & D8: sync_merit and finish!! diagnostics, $T" for T in REAL_ELTYPES
    # Transition into Backtracking second member synchronises anchor merit at switch point
    prob_lin = NonlinearProblem((x, p) -> x)
    x₀ = SVector{2, T}(T(100), zero(T))
    m1 = Picard(; damping = 0.9)
    m2 = Picard(; damping = 11.0, globalization = Backtracking())
    sw = Adapt.adapt(ToReal{T}(), Switch(m1, m2; threshold = 0.2))

    Fx₀ = residual!!(nothing, prob_lin, x₀, nothing)
    st = start!!(
        init_state(sw, prob_lin, x₀, nothing), sw, prob_lin, x₀, Fx₀, nothing, rnorm(Fx₀))
    # Step 1: ||Fx0|| = 100 > 20 -> m1 steps to x1 = (10, 0), ||Fx1|| = 10
    x₁, F₁, st, _ = step!!(st, sw, prob_lin, x₀, Fx₀, nothing)
    @test !st.switched
    @test x₁ === SVector{2, T}(T(10), zero(T))
    # Step 2: ||Fx1|| = 10 <= 20 -> latches switched = true, sync_merit sets m2.merit = 100,
    # and m2 backtracks from d = -110 to α = 0.1 -> x2 ≈ (-1, 0)
    x₂, _, st, _ = step!!(st, sw, prob_lin, x₁, F₁, nothing)
    @test st.switched
    @test isapprox(x₂[1], T(-1); rtol = 8 * eps(T))
    @test iszero(x₂[2])

    # Diagnostics: before switch forwards `first`, after switch forwards `second`
    prob_q = NonlinearProblem(R1QuadSwF(T(2), T(3)))
    x_q = SVector{2, T}(T(2), T(2))
    sw_early = Switch(
        Newton(; linsolve = SVDFactorization()),
        Newton(; linsolve = LUFactorization());
        threshold = 1e-6)
    _, st_early = solve(prob_q, x_q, sw_early; max_iterations = 1)
    @test st_early.rank === Int32(2)
    @test isfinite(st_early.nullnorm)

    sw_late = Switch(
        Picard(; damping = 0.2),
        Newton(; linsolve = SVDFactorization());
        threshold = 0.99)
    _, st_late = solve(prob_q, x_q, sw_late; max_iterations = 3)
    @test st_late.rank === Int32(2)
    @test isfinite(st_late.nullnorm)
end

@testset "Check 6: R1/R2/R3 parity, isbits, @inferred, 0 allocs, $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem(R1QuadSwF(T(2), T(3)))
    prob_a = NonlinearProblem(R3QuadSwF(T(2), T(3)), R3QuadSwJ())
    x₀_s = SVector{2, T}(T(1.5), T(1.5))
    x₀_a = T[T(1.5), T(1.5)]
    x₀_jl = JLArray(x₀_a)

    m = Switch(Picard(; damping = 0.2), Newton(; globalization = Static()); threshold = 0.8)
    m_r = Adapt.adapt(ToReal{R}(), m)
    st_s = @inferred init_state(m_r, prob_s, x₀_s, nothing)
    @test isbits(st_s)

    xs_2, st_s2 = @inferred solve(
        prob_s, x₀_s, m;
        max_iterations = 2, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    xa_2, st_a2 = solve(
        prob_a, copy(x₀_a), m;
        max_iterations = 2, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    xjl_2, st_jl2 = solve(
        prob_a, copy(x₀_jl), m;
        max_iterations = 2, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))

    @test isapprox(SVector{2, T}(xa_2), xs_2; rtol = 8 * eps(R))
    @test isapprox(Array(xjl_2), xa_2; rtol = 8 * eps(R))
    @test isapprox(st_a2.stepnorm, st_s2.stepnorm; rtol = 8 * eps(R))
    @test isapprox(st_jl2.stepnorm, st_a2.stepnorm; rtol = 8 * eps(R))

    _, st_full = @inferred solve(prob_s, x₀_s, m)
    @test st_full.code === SUCCESS

    if Base.JLOptions().check_bounds != 1
        # Skipped where a run forces `--check-bounds=yes`.
        @test r1_sw_alloc(prob_s, x₀_s, m_r) == 0
    end
    JET.@test_opt solve(prob_s, x₀_s, m_r; max_iterations = 2)

    old_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        x_work = copy(x₀_a)
        solver = init(prob_a, x_work, m)
        @test r3_sw_alloc(solver, x_work, x₀_a) == 0
    finally
        BLAS.set_num_threads(old_threads)
    end

    X_batch = [x₀_s for _ in 1:4]
    S_batch = Vector{SolverStatus{R}}(undef, 4)
    solve_batch!(X_batch, S_batch, prob_s, m)
    @test all(s -> s.code == SUCCESS, S_batch)
    X_batch2 = [x₀_s for _ in 1:4]
    counts = solve_batch!(X_batch2, prob_s, m)
    @test counts.SUCCESS == 4
end
