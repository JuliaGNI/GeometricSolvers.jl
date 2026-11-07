# `Repeat` combinator in R1, R2 and R3: constructor validation, 3-step `Picard` bit-for-bit
# parity and `sync_merit` between sub-steps (Check 4), short-circuiting on `SINGULAR` and
# `NONFINITE` (D6), `finish!!` diagnostics forwarding (D8), the `STALLED` rule and
# `Repeat(m, 1) ≡ m`, a `StationarityProblem`, and R1/R2/R3 `isbits`,
# `@inferred`, zero allocations, `JLArray` under `allowscalar(false)`, and `solve_batch!`
# (Check 6).

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: BLAS, diagind
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, NONFINITE, Newton, NonlinearProblem, Options, Picard,
                        Repeat, SINGULAR, STALLED, SVDFactorization, SUCCESS, SolverStatus,
                        Static, StationarityProblem, ToReal,
                        finish!!, init, init_state, residual!!, rnorm, solve, solve!,
                        solve_batch!, solve!!, start!!, step!!

include("../helpers/matrix.jl")

allowscalar(false)

struct R1QuadRepF{T}
    c1::T
    c2::T
end
(f::R1QuadRepF)(x, p) = SVector(x[1]^2 - f.c1, x[2]^2 - f.c2)

struct R3QuadRepF{T}
    c1::T
    c2::T
end
function (f::R3QuadRepF)(F, x, p)
    c = SVector(f.c1, f.c2)
    F .= x .* x .- c
    return F
end

struct R3QuadRepJ end
function (::R3QuadRepJ)(J, x, p)
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

function r1_rep_alloc(prob::P, x₀::X, m::M) where {P, X, M}
    solve(prob, x₀, m; max_iterations = 2)
    return @allocated solve(prob, x₀, m; max_iterations = 2)
end

function r3_rep_alloc(solver::S, x::X, x₀::X) where {S, X}
    copyto!(x, x₀)
    solve!(solver, x)
    copyto!(x, x₀)
    return @allocated solve!(solver, x)
end

@testset "Repeat constructor validation and ToReal adaptation" begin
    @test Base.isexported(GeometricSolvers, :Repeat)
    @test_throws ArgumentError Repeat(Picard(), 0)
    @test_throws ArgumentError Repeat(Picard(), -3)

    r = Repeat(Picard(; damping = 0.25), 3)
    @test r.count === Int32(3)

    for T in REAL_ELTYPES
        ar = @inferred Adapt.adapt(ToReal{T}(), r)
        @test ar.method.damping isa T
        @test ar.count === Int32(3)
    end
end

@testset "Check 4: Repeat(Picard, 3) parity with 3 Picard steps, $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem(R1QuadRepF(T(2), T(3)))
    x₀_s = SVector{2, T}(T(1.5), T(1.5))

    m_p = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.15))
    m_rep = Repeat(m_p, 3)

    st_rep = init_state(m_rep, prob_s, x₀_s, nothing)
    Fx₀_s = residual!!(nothing, prob_s, x₀_s, nothing)
    st_rep = start!!(st_rep, m_rep, prob_s, x₀_s, Fx₀_s, nothing, rnorm(Fx₀_s))
    x_rep, F_rep, _, info_rep = @inferred step!!(
        st_rep, m_rep, prob_s, x₀_s, Fx₀_s, nothing)

    x_seq = x₀_s
    F_seq = Fx₀_s
    st_p = start!!(
        init_state(m_p, prob_s, x_seq, nothing),
        m_p, prob_s, x_seq, F_seq, nothing, rnorm(F_seq))
    for _ in 1:3
        x_seq, F_seq, st_p, _ = step!!(st_p, m_p, prob_s, x_seq, F_seq, nothing)
    end

    @test x_rep === x_seq
    @test F_rep === F_seq
    @test info_rep.stepnorm === rnorm(x_seq - x₀_s)

    # Line-search member: sync_merit updates anchor merit between sub-steps
    prob_lin = NonlinearProblem((x, p) -> x)
    x₀_lin = SVector{2, T}(T(10), zero(T))
    m_ls = Adapt.adapt(ToReal{R}(), Picard(; damping = 1.5, globalization = Backtracking()))
    m_rep_ls = Repeat(m_ls, 2)

    Fx₀_lin = residual!!(nothing, prob_lin, x₀_lin, nothing)
    st_rls = start!!(
        init_state(m_rep_ls, prob_lin, x₀_lin, nothing),
        m_rep_ls, prob_lin, x₀_lin, Fx₀_lin, nothing, rnorm(Fx₀_lin))
    x_rls, F_rls, _, info_rls = @inferred step!!(
        st_rls, m_rep_ls, prob_lin, x₀_lin, Fx₀_lin, nothing)

    st_sa = start!!(
        init_state(m_ls, prob_lin, x₀_lin, nothing),
        m_ls, prob_lin, x₀_lin, Fx₀_lin, nothing, rnorm(Fx₀_lin))
    x_sa1, F_sa1, st_sa, _ = step!!(st_sa, m_ls, prob_lin, x₀_lin, Fx₀_lin, nothing)
    x_sa2, F_sa2, st_sa, _ = step!!(st_sa, m_ls, prob_lin, x_sa1, F_sa1, nothing)
    @test x_rls === x_sa2
    @test F_rls === F_sa2
    @test info_rls.stepnorm === rnorm(x_sa2 - x₀_lin)
end

@testset "D6 & D8: short-circuit and finish!! diagnostics, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem(R1QuadRepF(T(2), T(3)))
    x_sing = zero(SVector{2, T})
    m_sc = Repeat(Newton(), 3)
    x_out, st_out = solve(prob, x_sing, m_sc)
    @test st_out.code === SINGULAR
    @test x_out === x_sing

    # Un-started line-search member fails closed with NONFINITE
    x₀ = SVector{2, T}(T(2), T(2))
    Fx₀ = residual!!(nothing, prob, x₀, nothing)
    m_ls = Adapt.adapt(ToReal{T}(), Repeat(Newton(; globalization = Backtracking()), 2))
    st_u = init_state(m_ls, prob, x₀, nothing)
    _, _, _, info_u = step!!(st_u, m_ls, prob, x₀, Fx₀, nothing)
    @test info_u.code === NONFINITE
    @test iszero(info_u.stepnorm)

    # Diagnostics: Repeat(Newton(SVD), 2) forwards SVD diagnostics
    m_svd = Repeat(Newton(; linsolve = SVDFactorization()), 2)
    _, st_svd = solve(prob, x₀, m_svd; max_iterations = 2)
    @test st_svd.code === SUCCESS
    @test st_svd.rank === Int32(2)
    @test isfinite(st_svd.nullnorm)
    @test isfinite(st_svd.gradratio)
end

function repeat_one_step(m, prob, x₀)
    Fx₀ = residual!!(x₀ isa Number ? nothing : similar(x₀), prob, x₀, nothing)
    s = start!!(init_state(m, prob, copy(x₀), nothing), m, prob, copy(x₀), copy(Fx₀),
        nothing, rnorm(Fx₀))
    return step!!(s, m, prob, copy(x₀), copy(Fx₀), nothing)[4]
end

# x² - 2 with every tolerance zero: the Backtracking line search stalls at the root
@testset "Repeat is STALLED only when every sub-step is, Repeat(m, 1) ≡ m, $T, $(nameof(AT))" for T in REAL_ELTYPES,
    AT in (Number, Array)

    nb = Newton(; globalization = Backtracking())
    opt = Options(T; f_abstol = 0, f_reltol = 0, x_abstol = 0, x_reltol = 0, max_stalls = 3)
    if AT === Number
        prob = NonlinearProblem((x, p) -> x^2 - 2)
        x₀, x_root, x_near = one(T), sqrt(T(2)), sqrt(T(2)) + 4 * eps(T)
    else
        prob = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- 2; nothing))
        x₀, x_root, x_near = T[1], T[sqrt(T(2))], T[sqrt(T(2)) + 4 * eps(T)]
    end
    nb_T = Adapt.adapt(ToReal{T}(), nb)
    @test repeat_one_step(nb_T, prob, x_root).code === STALLED
    @test repeat_one_step(Repeat(nb_T, 1), prob, x_root).code === STALLED
    @test repeat_one_step(Repeat(nb_T, 2), prob, x_root).code === STALLED
    # the first sub-step moves to the root, the second stalls there
    @test repeat_one_step(nb_T, prob, x_near).code === SUCCESS
    @test repeat_one_step(Repeat(nb_T, 2), prob, x_near).code === SUCCESS

    _, st_nb = solve(prob, x₀, nb; options = opt)
    _, st_rep = solve(prob, x₀, Repeat(nb, 1); options = opt)
    @test st_nb.code === STALLED
    @test st_rep.code === STALLED
    @test st_rep.iterations == st_nb.iterations
end

# ∇S(x) = x³ + x - 1 componentwise, whose real root is 0.6823278038280193
@testset "Repeat on a StationarityProblem, $(nameof(typeof(x₀)))" for x₀ in (
    SVector(0.1, 0.2), [0.1, 0.2])
    sp = StationarityProblem((x, p) -> sum(abs2, x) / 2 - sum(x) + sum(x .^ 4) / 4)
    x, st = solve(sp, x₀, Repeat(Newton(), 2))
    @test st.code === SUCCESS
    @test isapprox(Array(x), fill(0.6823278038280193, 2); rtol = 1e-10)
end

@testset "Check 6: R1/R2/R3 parity, isbits, @inferred, 0 allocs, $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem(R1QuadRepF(T(2), T(3)))
    prob_a = NonlinearProblem(R3QuadRepF(T(2), T(3)), R3QuadRepJ())
    x₀_s = SVector{2, T}(T(1.5), T(1.5))
    x₀_a = T[T(1.5), T(1.5)]
    x₀_jl = JLArray(x₀_a)

    for m in (
        Repeat(Picard(; damping = 0.2), 3),
        Repeat(Newton(; globalization = Static()), 2))
        m_r = Adapt.adapt(ToReal{R}(), m)
        st_s = @inferred init_state(m_r, prob_s, x₀_s, nothing)
        @test isbits(st_s)

        xs_1, st_s1 = @inferred solve(
            prob_s, x₀_s, m;
            max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
            x_abstol = zero(R), x_reltol = zero(R))
        xa_1, st_a1 = solve(
            prob_a, copy(x₀_a), m;
            max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
            x_abstol = zero(R), x_reltol = zero(R))
        xjl_1, st_jl1 = solve(
            prob_a, copy(x₀_jl), m;
            max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
            x_abstol = zero(R), x_reltol = zero(R))

        @test isapprox(SVector{2, T}(xa_1), xs_1; rtol = 8 * eps(R))
        @test isapprox(Array(xjl_1), xa_1; rtol = 8 * eps(R))
        @test isapprox(st_a1.stepnorm, st_s1.stepnorm; rtol = 8 * eps(R))
        @test isapprox(st_jl1.stepnorm, st_a1.stepnorm; rtol = 8 * eps(R))

        if Base.JLOptions().check_bounds != 1
            # Skipped where a run forces `--check-bounds=yes`.
            @test r1_rep_alloc(prob_s, x₀_s, m_r) == 0
        end
        JET.@test_opt solve(prob_s, x₀_s, m_r; max_iterations = 2)

        old_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            x_work = copy(x₀_a)
            solver = init(prob_a, x_work, m; max_iterations = 2)
            @test r3_rep_alloc(solver, x_work, x₀_a) == 0
        finally
            BLAS.set_num_threads(old_threads)
        end
    end

    m_batch = Repeat(Newton(; globalization = Static()), 2)
    X_batch = [x₀_s for _ in 1:4]
    S_batch = Vector{SolverStatus{R}}(undef, 4)
    solve_batch!(X_batch, S_batch, prob_s, m_batch)
    @test all(s -> s.code == SUCCESS, S_batch)
    X_batch2 = [x₀_s for _ in 1:4]
    counts = solve_batch!(X_batch2, prob_s, m_batch)
    @test counts.SUCCESS == 4
end
