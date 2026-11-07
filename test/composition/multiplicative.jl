# `Multiplicative` combinator in R1, R2 and R3: constructor validation, sequential step
# parity and `sync_merit` before member 2 (Check 4, MU6), Pollock operator-Picard + Newton
# convergence where standalone `Newton()` diverges (Check 4), short-circuiting on
# `SINGULAR`, `NONFINITE` and `LINESEARCH_FAILED` (D6), `finish!!` diagnostics from the last
# diagnostic member (D8), the `STALLED` rule and `Multiplicative(m) ≡ m`, a
# `StationarityProblem`, and R1/R2/R3 `isbits`, `@inferred`, zero allocations, `JLArray`, and
# `solve_batch!` (Check 6).

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: BLAS, Diagonal, diagind
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Additive, Backtracking, LUFactorization, Multiplicative, NONFINITE,
                        Newton, NonlinearProblem, Options, Picard, Repeat, SINGULAR,
                        STALLED,
                        SVDFactorization, SUCCESS, SolverStatus, Static,
                        StationarityProblem,
                        Switch, ToReal,
                        finish!!, init, init_state, residual!!, rnorm, solve, solve!,
                        solve_batch!, solve!!, start!!, step!!, sync_merit

include("../helpers/matrix.jl")

allowscalar(false)

struct R1PollockF end
function (::R1PollockF)(x, p)
    T = eltype(x)
    return SVector(x[1] / sqrt(one(T) + x[1]^2), x[2] / sqrt(one(T) + x[2]^2))
end

struct R1PollockA end
function (::R1PollockA)(x, p)
    T = eltype(x)
    d1 = inv(sqrt(one(T) + T(0.5) * x[1]^2))
    d2 = inv(sqrt(one(T) + T(0.5) * x[2]^2))
    return SMatrix{2, 2, T}(d1, zero(T), zero(T), d2)
end

struct R3PollockF end
function (::R3PollockF)(F, x, p)
    @. F = x / sqrt(one(eltype(x)) + x^2)
    return F
end

struct R3PollockA end
function (::R3PollockA)(J, x, p)
    T = eltype(J)
    fill!(J, zero(T))
    xh = Array(x)
    Jh = zeros(T, length(xh), length(xh))
    for i in eachindex(xh)
        Jh[i, i] = inv(sqrt(one(T) + T(0.5) * xh[i]^2))
    end
    copyto!(J, Jh)
    return J
end

struct R1QuadMultF{T}
    c1::T
    c2::T
end
(f::R1QuadMultF)(x, p) = SVector(x[1]^2 - f.c1, x[2]^2 - f.c2)

struct R3QuadMultF{T}
    c1::T
    c2::T
end
function (f::R3QuadMultF)(F, x, p)
    c = SVector(f.c1, f.c2)
    F .= x .* x .- c
    return F
end

struct R3QuadMultJ end
function (::R3QuadMultJ)(J, x, p)
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

function r1_mult_alloc(prob::P, x₀::X, m::M) where {P, X, M}
    solve(prob, x₀, m; max_iterations = 2)
    return @allocated solve(prob, x₀, m; max_iterations = 2)
end

function r3_mult_alloc(solver::S, x::X, x₀::X) where {S, X}
    copyto!(x, x₀)
    solve!(solver, x)
    copyto!(x, x₀)
    return @allocated solve!(solver, x)
end

@testset "Multiplicative constructor validation and ToReal adaptation" begin
    @test Base.isexported(GeometricSolvers, :Multiplicative)
    @test_throws ArgumentError Multiplicative()

    m = Multiplicative(Picard(; damping = 0.2), Newton())
    @test length(m.members) == 2
    @test Multiplicative((Picard(; damping = 0.2), Newton())) === m

    for T in REAL_ELTYPES
        a = @inferred Adapt.adapt(ToReal{T}(), m)
        @test a.members[1].damping isa T
        @test a.members[2].regularization isa T
    end
end

@testset "Check 4: sequential parity and sync_merit (MU6), $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem(R1QuadMultF(T(2), T(3)))
    x₀_s = SVector{2, T}(T(2), T(2))

    # Static members: 1 Multiplicative step == Picard step then Newton step bit for bit
    m₁ = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.2))
    m₂ = Adapt.adapt(ToReal{R}(), Newton(; globalization = Static()))
    m_mult = Multiplicative(m₁, m₂)

    x_m, st_m = @inferred solve(
        prob_s, x₀_s, m_mult;
        max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))

    Fx_seq = residual!!(nothing, prob_s, x₀_s, nothing)
    s₁ = start!!(
        init_state(m₁, prob_s, x₀_s, nothing),
        m₁, prob_s, x₀_s, Fx_seq, nothing, rnorm(Fx_seq))
    x_seq, Fx_seq, _, _ = step!!(s₁, m₁, prob_s, x₀_s, Fx_seq, nothing)
    s₂ = start!!(
        init_state(m₂, prob_s, x_seq, nothing),
        m₂, prob_s, x_seq, Fx_seq, nothing, rnorm(Fx_seq))
    x_seq, Fx_seq, _, _ = step!!(s₂, m₂, prob_s, x_seq, Fx_seq, nothing)

    @test x_m === x_seq
    @test st_m.stepnorm === rnorm(x_seq - x₀_s)

    # Line-search member 2: sync_merit updates member 2's anchor merit from φ(x₀) = 100
    # to φ(x₁) = 1 so member 2 backtracks when φ(x₁ + d₂) = 100 >= φ(x₁) = 1 (catches MU6!)
    prob_lin = NonlinearProblem((x, p) -> x)
    x₀_lin = SVector{2, T}(T(10), zero(T))
    m1_ls = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.9))
    m2_ls = Adapt.adapt(
        ToReal{R}(), Picard(; damping = 11.0, globalization = Backtracking()))
    m_mult_ls = Multiplicative(m1_ls, m2_ls)

    st_mls = init_state(m_mult_ls, prob_lin, x₀_lin, nothing)
    Fx₀_lin = residual!!(nothing, prob_lin, x₀_lin, nothing)
    st_mls = start!!(st_mls, m_mult_ls, prob_lin, x₀_lin, Fx₀_lin, nothing, rnorm(Fx₀_lin))
    x_mls, F_mls, _, info_mls = @inferred step!!(
        st_mls, m_mult_ls, prob_lin, x₀_lin, Fx₀_lin, nothing)

    # Standalone sequential steps with fresh anchor merit at x₁ = (1, 0)
    s1_sa = start!!(
        init_state(m1_ls, prob_lin, x₀_lin, nothing),
        m1_ls, prob_lin, x₀_lin, Fx₀_lin, nothing, rnorm(Fx₀_lin))
    x1_sa, F1_sa, _, _ = step!!(s1_sa, m1_ls, prob_lin, x₀_lin, Fx₀_lin, nothing)
    s2_sa = start!!(
        init_state(m2_ls, prob_lin, x1_sa, nothing),
        m2_ls, prob_lin, x1_sa, F1_sa, nothing, rnorm(F1_sa))
    x2_sa, F2_sa, _, _ = step!!(s2_sa, m2_ls, prob_lin, x1_sa, F1_sa, nothing)

    @test x_mls === x2_sa
    @test F_mls === F2_sa
    @test isapprox(x_mls[1], T(R(-0.1)); rtol = 8 * eps(R))
    @test iszero(x_mls[2])
    @test info_mls.stepnorm === rnorm(x2_sa - x₀_lin)

    # Recursive sync_merit through nested Additive, Multiplicative, Repeat, Switch states
    m_nest = Adapt.adapt(
        ToReal{R}(),
        Multiplicative(
            Additive(m2_ls, m1_ls),
            Repeat(m2_ls, 2),
            Switch(m2_ls, m2_ls; threshold = 0.5)))
    st_nest = start!!(
        init_state(m_nest, prob_lin, x₀_lin, nothing),
        m_nest, prob_lin, x₀_lin, Fx₀_lin, nothing, rnorm(Fx₀_lin))
    st_nest_sync = @inferred sync_merit(st_nest, m_nest, R(4))
    @test st_nest_sync.states[1].states[1].merit === R(4)
    @test st_nest_sync.states[2].state.merit === R(4)
    @test st_nest_sync.states[3].s1.merit === R(4)
    @test st_nest_sync.states[3].s2.merit === R(4)
end

@testset "sync_merit resets the residual norm of a Static Newton member, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem(R1QuadMultF(T(2), T(3)))
    m₁ = Adapt.adapt(ToReal{T}(), Picard(; damping = 0.2))
    m₂ = Adapt.adapt(ToReal{T}(), Newton(; globalization = Static()))
    m = Multiplicative(m₁, m₂)
    x = SVector{2, T}(T(2), T(2))
    Fx = residual!!(nothing, prob, x, nothing)
    st = start!!(init_state(m, prob, x, nothing), m, prob, x, Fx, nothing, rnorm(Fx))
    for _ in 1:2
        y, Fy, _, _ = step!!(st.states[1], m₁, prob, x, Fx, nothing)
        x, Fx, st, _ = step!!(st, m, prob, x, Fx, nothing)
        # Member 2 factorises J(y) = 2 diag(y) at the iterate y that member 1 returns, so its
        # gradratio is ‖J(y)ᴴ F(y)‖ / (‖J(y)‖_F ‖F(y)‖) with the norm of F(y), not of F(x).
        gr = rnorm(2 .* y .* Fy) / (2 * rnorm(y)) / rnorm(Fy)
        @test isapprox(st.states[2].diag.gradratio, gr; rtol = 8 * eps(T))
    end
end

@testset "Check 4: Pollock Picard + Newton vs Newton, $T" for T in REAL_ELTYPES
    x₀_s = SVector{2, T}(T(3), T(3))
    prob_s = NonlinearProblem(R1PollockF())
    _, st_pure_s = solve(
        prob_s, x₀_s, Newton(; globalization = Static()); max_iterations = 15)
    @test st_pure_s.code !== SUCCESS

    m_pollock_s = Multiplicative(
        Newton(; jacobian = R1PollockA(), globalization = Static()),
        Newton(; globalization = Static()))
    x_pol_s, st_pol_s = solve(
        prob_s, x₀_s, m_pollock_s;
        max_iterations = 30, f_abstol = eps(T), f_reltol = eps(T))
    @test st_pol_s.code === SUCCESS
    @test rnorm(x_pol_s) <= T(1e-7)

    x₀_a = T[3, 3]
    prob_a = NonlinearProblem(R3PollockF())
    _, st_pure_a = solve(
        prob_a, copy(x₀_a), Newton(; globalization = Static()); max_iterations = 15)
    @test st_pure_a.code !== SUCCESS

    m_pollock_a = Multiplicative(
        Newton(; jacobian = R3PollockA(), globalization = Static()),
        Newton(; globalization = Static()))
    x_pol_a, st_pol_a = solve(
        prob_a, copy(x₀_a), m_pollock_a;
        max_iterations = 30, f_abstol = eps(T), f_reltol = eps(T))
    @test st_pol_a.code === SUCCESS
    @test rnorm(x_pol_a) <= T(1e-7)
end

@testset "D6 & D8: short-circuit and finish!! diagnostics, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem(R1QuadMultF(T(2), T(3)))
    x_sing = zero(SVector{2, T})
    # Member 1 (`Newton`) is singular at `x = 0`, so Multiplicative short-circuits
    m_sc = Multiplicative(Newton(), Picard(; damping = 0.5))
    x_out, st_out = solve(prob, x_sing, m_sc)
    @test st_out.code === SINGULAR
    @test x_out === x_sing

    # Un-started line-search member fails closed with NONFINITE
    x₀ = SVector{2, T}(T(2), T(2))
    Fx₀ = residual!!(nothing, prob, x₀, nothing)
    m_ls = Adapt.adapt(
        ToReal{T}(),
        Multiplicative(Picard(; damping = 0.2), Newton(; globalization = Backtracking())))
    st_u = init_state(m_ls, prob, x₀, nothing)
    x_u, _, _, info_u = step!!(st_u, m_ls, prob, x₀, Fx₀, nothing)
    @test info_u.code === NONFINITE
    # the step norm is that of the iterate that member 1 reached before member 2 failed
    @test x_u != x₀
    @test info_u.stepnorm == rnorm(x_u - x₀)

    # A member that fails its line search after member 1 made progress does not end the solve:
    # the outer step keeps its nonzero norm, counts the failure, and the solve converges
    # like `Newton` alone. F = -x - x³ from (1, 1/2); Picard with Backtracking fails every step.
    prob_cubic = NonlinearProblem((x, p) -> -x .- x .^ 3)
    x_c = SVector{2, T}(T(1), T(1) / 2)
    pic_bt = Picard(; damping = 1, globalization = Backtracking())
    _, st_n = solve(prob_cubic, x_c, Newton())
    for m_c in (Multiplicative(Newton(), pic_bt), Repeat(Multiplicative(Newton(), pic_bt), 1))
        _, st_c = solve(prob_cubic, x_c, m_c)
        @test st_c.code === SUCCESS
        @test st_c.iterations == st_n.iterations
        @test st_c.step_failures == st_n.iterations
    end
    # with no progress before the failure, the step norm is zero and the solve stops
    _, st_p = solve(prob_cubic, x_c, Multiplicative(pic_bt, Newton()))
    @test st_p.code === GeometricSolvers.LINESEARCH_FAILED
    @test st_p.iterations == 1
    @test iszero(st_p.stepnorm)

    # Diagnostics: Multiplicative(Newton(SVD), Picard()) preserves Newton(SVD) diagnostics
    m_np = Multiplicative(Newton(; linsolve = SVDFactorization()), Picard(; damping = 0.1))
    _, st_np = solve(prob, x₀, m_np; max_iterations = 2)
    @test st_np.rank === Int32(2)
    @test isfinite(st_np.nullnorm)
    @test isfinite(st_np.gradratio)

    # Diagnostics: Multiplicative(Newton(LU), Newton(SVD)) takes the second member's SVD
    m_nn = Multiplicative(
        Newton(; linsolve = LUFactorization()),
        Newton(; linsolve = SVDFactorization()))
    _, st_nn = solve(prob, x₀, m_nn; max_iterations = 1)
    @test st_nn.rank === Int32(2)
    @test isfinite(st_nn.nullnorm)
end

function multiplicative_one_step(m, prob, x₀)
    Fx₀ = residual!!(x₀ isa Number ? nothing : similar(x₀), prob, x₀, nothing)
    s = start!!(init_state(m, prob, copy(x₀), nothing), m, prob, copy(x₀), copy(Fx₀),
        nothing, rnorm(Fx₀))
    return step!!(s, m, prob, copy(x₀), copy(Fx₀), nothing)[4]
end

# x² - 2 with every tolerance zero: the Backtracking line search stalls at the root, and
# `Newton()` with `Static` takes a step of round-off size there
@testset "Multiplicative is STALLED only when every member step is, Multiplicative(m) ≡ m, $T, $(nameof(AT))" for T in REAL_ELTYPES,
    AT in (Number, Array)

    nb = Newton(; globalization = Backtracking())
    opt = Options(T; f_abstol = 0, f_reltol = 0, x_abstol = 0, x_reltol = 0, max_stalls = 3)
    if AT === Number
        prob = NonlinearProblem((x, p) -> x^2 - 2)
        x₀, x_root = one(T), sqrt(T(2))
    else
        prob = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- 2; nothing))
        x₀, x_root = T[1], T[sqrt(T(2))]
    end
    nb_T = Adapt.adapt(ToReal{T}(), nb)
    n_T = Adapt.adapt(ToReal{T}(), Newton())
    @test multiplicative_one_step(nb_T, prob, x_root).code === STALLED
    @test multiplicative_one_step(n_T, prob, x_root).code === SUCCESS
    @test multiplicative_one_step(Multiplicative(nb_T), prob, x_root).code === STALLED
    @test multiplicative_one_step(Multiplicative(nb_T, nb_T), prob, x_root).code === STALLED
    @test multiplicative_one_step(Multiplicative(nb_T, n_T), prob, x_root).code === SUCCESS
    @test multiplicative_one_step(Multiplicative(n_T, nb_T), prob, x_root).code === SUCCESS

    _, st_nb = solve(prob, x₀, nb; options = opt)
    _, st_mul = solve(prob, x₀, Multiplicative(nb); options = opt)
    @test st_nb.code === STALLED
    @test st_mul.code === STALLED
    @test st_mul.iterations == st_nb.iterations
end

# ∇S(x) = x³ + x - 1 componentwise, whose real root is 0.6823278038280193
@testset "Multiplicative on a StationarityProblem, $(nameof(typeof(x₀)))" for x₀ in (
    SVector(0.1, 0.2), [0.1, 0.2])
    sp = StationarityProblem((x, p) -> sum(abs2, x) / 2 - sum(x) + sum(x .^ 4) / 4)
    x, st = solve(sp, x₀, Multiplicative(Picard(; damping = 0.5), Newton()))
    @test st.code === SUCCESS
    @test isapprox(Array(x), fill(0.6823278038280193, 2); rtol = 1e-10)
end

# A member `Newton` that factorises in `Float32` on a `Float64` problem takes the mixed-precision
# step inside every combinator, and `sync_merit` keeps its line search on the incoming residual
@testset "Combinators with a mixed-precision Newton member, $(nameof(typeof(x₀)))" for x₀ in (
    SVector(1.5, 1.5), [1.5, 1.5])
    prob = x₀ isa SVector ? NonlinearProblem(R1QuadMultF(2.0, 3.0)) :
           NonlinearProblem(R3QuadMultF(2.0, 3.0), R3QuadMultJ())
    root = [sqrt(2.0), sqrt(3.0)]
    pic = Picard(; damping = 0.2)
    for nm in (Newton(; linsolve = LUFactorization(Float32)),
        Newton(; linsolve = LUFactorization(Float32), globalization = Backtracking()))
        @test GeometricSolvers.is_mixed(nm, Float64)
        for m in (Additive(nm, pic), Multiplicative(pic, nm), Repeat(nm, 2),
            Switch(pic, nm; threshold = 0.5))
            x₁, st₁ = solve(prob, copy(x₀), m; max_iterations = 1)
            @test Array(x₁) != Array(x₀)
            x, st = solve(prob, copy(x₀), m)
            @test st.code === SUCCESS
            @test isapprox(Array(x), root; rtol = 1e-12)
        end
    end
end

@testset "Check 6: R1/R2/R3 parity, isbits, @inferred, 0 allocs, $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem(R1QuadMultF(T(2), T(3)))
    prob_a = NonlinearProblem(R3QuadMultF(T(2), T(3)), R3QuadMultJ())
    x₀_s = SVector{2, T}(T(1.5), T(1.5))
    x₀_a = T[T(1.5), T(1.5)]
    x₀_jl = JLArray(x₀_a)

    m = Multiplicative(Picard(; damping = 0.2), Newton(; globalization = Static()))
    m_r = Adapt.adapt(ToReal{R}(), m)
    st_s = @inferred init_state(m_r, prob_s, x₀_s, nothing)
    @test isbits(st_s)

    xs_1, st_s1 = @inferred solve(prob_s, x₀_s, m; max_iterations = 1)
    xa_1, st_a1 = solve(prob_a, copy(x₀_a), m; max_iterations = 1)
    xjl_1, st_jl1 = solve(prob_a, copy(x₀_jl), m; max_iterations = 1)

    @test isapprox(SVector{2, T}(xa_1), xs_1; rtol = 8 * eps(R))
    @test isapprox(Array(xjl_1), xa_1; rtol = 8 * eps(R))
    @test isapprox(st_a1.stepnorm, st_s1.stepnorm; rtol = 8 * eps(R))
    @test isapprox(st_jl1.stepnorm, st_a1.stepnorm; rtol = 8 * eps(R))

    xs_full, st_full = @inferred solve(prob_s, x₀_s, m)
    @test st_full.code === SUCCESS

    if Base.JLOptions().check_bounds != 1
        # Skipped where a run forces `--check-bounds=yes`.
        @test r1_mult_alloc(prob_s, x₀_s, m_r) == 0
    end
    JET.@test_opt solve(prob_s, x₀_s, m_r; max_iterations = 2)

    old_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        x_work = copy(x₀_a)
        solver = init(prob_a, x_work, m)
        @test r3_mult_alloc(solver, x_work, x₀_a) == 0
    finally
        BLAS.set_num_threads(old_threads)
    end

    X_batch = [x₀_s for _ in 1:4]
    S_batch = Vector{SolverStatus{R}}(undef, 4)
    solve_batch!(X_batch, S_batch, prob_s, m)
    @test all(s -> s.code == SUCCESS, S_batch)
    @test all(x -> isapprox(x, xs_full; rtol = 16 * eps(R)), X_batch)
    X_batch2 = [x₀_s for _ in 1:4]
    counts = solve_batch!(X_batch2, prob_s, m)
    @test counts.SUCCESS == 4
end
