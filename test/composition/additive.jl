# `Additive` combinator in R1, R2 and R3: constructor validation, affine exactness and PETSc
# safeguard (Check 1, MU1, MU2) in R1 and in R3, singular Gram fallback (Check 2, MU3, MU4),
# `FixedWeights` and complex real weights (Check 3), five `FixedWeights` members in R3, R3
# member isolation (Check 4, MU5), un-started fail-closed guard (Check 5), `finish!!`
# diagnostics (D8), and R1/R2/R3 `isbits`, `@inferred`, zero allocations, `JLArray` under
# `allowscalar(false)`, and `solve_batch!` (Check 6).

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: BLAS, diagind
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Additive, Backtracking, FixedWeights, NONFINITE, Newton
using GeometricSolvers: NonlinearProblem, OptimalWeights, Picard, SVDFactorization
using GeometricSolvers: SUCCESS, SolverStatus, Static, ToReal
using GeometricSolvers: init, solve, solve!, solve_batch!
import GeometricSolvers: finish!!, init_state, residual!!, residual_buffer
import GeometricSolvers: rnorm, start!!, step!!

include("../helpers/matrix.jl")

allowscalar(false)

struct RecordingMember{M, X}
    inner::M
    seen_x::Base.RefValue{X}
    seen_Fx::Base.RefValue{X}
end

function Adapt.adapt_structure(to, m::RecordingMember)
    return RecordingMember(Adapt.adapt(to, m.inner), m.seen_x, m.seen_Fx)
end

struct RecordingState{S}
    inner::S
end

function init_state(m::RecordingMember, prob::NonlinearProblem, x, p)
    return RecordingState(init_state(m.inner, prob, x, p))
end

residual_buffer(s::RecordingState) = residual_buffer(s.inner)

function start!!(s::RecordingState, m::RecordingMember, prob, x, Fx, p, fnorm₀)
    return RecordingState(start!!(s.inner, m.inner, prob, x, Fx, p, fnorm₀))
end

function finish!!(s::RecordingState, m::RecordingMember, prob, x, Fx, st::SolverStatus)
    x_out, st_out, inner = finish!!(s.inner, m.inner, prob, x, Fx, st)
    return x_out, st_out, RecordingState(inner)
end

function step!!(s::RecordingState, m::RecordingMember, prob::NonlinearProblem, x, Fx, p)
    m.seen_x[] = copy(Array(x))
    m.seen_Fx[] = copy(Array(Fx))
    x₊, F₊, inner, info = step!!(s.inner, m.inner, prob, x, Fx, p)
    return x₊, F₊, RecordingState(inner), info
end

struct R1QuadF{T}
    c1::T
    c2::T
end

(f::R1QuadF)(x, p) = SVector(x[1]^2 - f.c1, x[2]^2 - f.c2)

struct R3QuadF{T}
    c1::T
    c2::T
end

function (f::R3QuadF)(F, x, p)
    c = SVector(f.c1, f.c2)
    F .= x .* x .- c
    return F
end

struct R3QuadJ end

function (::R3QuadJ)(J, x, p)
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

function r1_solve_alloc(prob::P, x₀::X, m::M) where {P, X, M}
    solve(prob, x₀, m; max_iterations = 2)
    return @allocated solve(prob, x₀, m; max_iterations = 2)
end

function r3_add_alloc(solver::S, x::X, x₀::X) where {S, X}
    copyto!(x, x₀)
    solve!(solver, x)
    copyto!(x, x₀)
    return @allocated solve!(solver, x)
end

@testset "Additive constructor validation and ToReal adaptation" begin
    @test Base.isexported(GeometricSolvers, :Additive)

    @test_throws ArgumentError Additive()
    @test_throws ArgumentError Additive(Newton(), Picard(); weights = FixedWeights(1.0))
    @test_throws ArgumentError Additive(Newton(); weights = FixedWeights(0.5, 0.5))

    m_opt = Additive(Newton(), Picard(; damping = 0.5))
    @test m_opt.weights === OptimalWeights(1.0)
    @test length(m_opt.members) == 2

    m_fix = Additive(Newton(), Picard(; damping = 0.5); weights = FixedWeights(0.3, 0.7))
    @test m_fix.weights === FixedWeights(0.3, 0.7)

    for T in REAL_ELTYPES
        a_opt = @inferred Adapt.adapt(ToReal{T}(), m_opt)
        @test a_opt.weights isa OptimalWeights{T}
        @test a_opt.members[2].damping isa T

        a_fix = @inferred Adapt.adapt(ToReal{T}(), m_fix)
        @test a_fix.weights isa FixedWeights{2, T}
        @test eltype(a_fix.weights.weights) === T
    end

    # R1 OptimalWeights supports at most 4 members; 5 throws ArgumentError in init_state
    p5 = NonlinearProblem((x, p) -> x)
    m5_opt = Additive(Picard(), Picard(), Picard(), Picard(), Picard())
    @test_throws ArgumentError init_state(m5_opt, p5, SVector(1.0, 2.0), nothing)
    @test_throws ArgumentError init_state(m5_opt, p5, 1.0, nothing)

    # FixedWeights with 5 members in R1 is allowed
    w5 = FixedWeights(0.2, 0.2, 0.2, 0.2, 0.2)
    m5_fix = Additive(Picard(), Picard(), Picard(), Picard(), Picard(); weights = w5)
    @test isbits(init_state(m5_fix, p5, SVector(1.0, 2.0), nothing))
end

@testset "Check 1: Affine exactness and PETSc safeguard (MU1, MU2), $T" for T in ELTYPES
    R = real(T)
    A = SMatrix{2, 2, T}(T(3), T(0.5), T(1), T(2))
    b = SVector{2, T}(T(4), T(2.5))
    prob_aff = NonlinearProblem((x, p) -> A * x - b)
    x₀ = zero(SVector{2, T})

    n_st = Newton(; globalization = Static())
    p_02 = Picard(; damping = 0.2)
    m_add = Adapt.adapt(ToReal{R}(), Additive(n_st, p_02))
    st_add = init_state(m_add, prob_aff, x₀, nothing)
    Fx₀ = residual!!(nothing, prob_aff, x₀, nothing)
    st_add = start!!(st_add, m_add, prob_aff, x₀, Fx₀, nothing, rnorm(Fx₀))
    x₁, F₁, _, info₁ = @inferred step!!(st_add, m_add, prob_aff, x₀, Fx₀, nothing)

    m_n = Adapt.adapt(ToReal{R}(), n_st)
    st_n = init_state(m_n, prob_aff, x₀, nothing)
    st_n = start!!(st_n, m_n, prob_aff, x₀, Fx₀, nothing, rnorm(Fx₀))
    _, F_N, _, _ = step!!(st_n, m_n, prob_aff, x₀, Fx₀, nothing)
    @test rnorm(F₁) <= rnorm(F_N) + 8 * eps(R) * rnorm(b)
    @test isapprox(x₁, A \ b; rtol = 16 * eps(R))
    # The `ρ = 1` property of D5, exactly. A rounding-level excess of `‖F(x₊)‖` over an
    # already-solved `‖F(x_N)‖` is a safeguard fallback, so at most that one fallback is
    # counted (decision 25).
    @test rnorm(F₁) <= rnorm(F_N)
    @test info₁.failures <= Int32(1)

    # Scalar iterate (Number) in R1
    prob_num = NonlinearProblem((x, p) -> T(3) * x - T(6))
    x_num, st_num = @inferred solve(prob_num, zero(T), m_add; max_iterations = 1)
    @test isapprox(x_num, T(2); rtol = 16 * eps(R))
    @test st_num.code === SUCCESS

    # Misleading secant model where linear combination overshoots and triggers safeguard
    # (catches MU1 if safeguard is disabled, and MU2 if safeguard failure is not counted)
    function f_mis(x, p)
        y1 = x[1] - one(T) + T(50) * (x[1] - T(R(0.5))) * (x[1] - T(R(0.2)))
        return SVector{2, T}(y1, x[2])
    end
    prob_mis = NonlinearProblem(f_mis)
    m_mis = Adapt.adapt(ToReal{R}(), Additive(Picard(; damping = 0.5), p_02))
    st_mis = init_state(m_mis, prob_mis, x₀, nothing)
    Fm₀ = residual!!(nothing, prob_mis, x₀, nothing)
    st_mis = start!!(st_mis, m_mis, prob_mis, x₀, Fm₀, nothing, rnorm(Fm₀))
    xm₁, Fm₁, _, info_mis = @inferred step!!(st_mis, m_mis, prob_mis, x₀, Fm₀, nothing)
    x_c1 = x₀ - R(0.5) * Fm₀
    x_c2 = x₀ - R(0.2) * Fm₀
    F_c1 = residual!!(nothing, prob_mis, x_c1, nothing)
    F_c2 = residual!!(nothing, prob_mis, x_c2, nothing)
    @test rnorm(Fm₁) <= min(rnorm(F_c1), rnorm(F_c2))
    @test xm₁ === (rnorm(F_c1) <= rnorm(F_c2) ? x_c1 : x_c2)
    @test info_mis.failures == Int32(1)

    # Pure safeguard fallback with non-collinear candidate residual differences
    # (`rank_fallback == false` and `safeguard_fallback == true`)
    function f_nc(x, p)
        y1 = x[1] - one(T) + T(10) * x[1] * (x[1] - one(T))
        y2 = x[2] - one(T) - T(20) * x[2] * (x[2] - T(R(0.2)))
        return SVector{2, T}(y1, y2)
    end
    prob_nc = NonlinearProblem(f_nc)
    m_nc = Adapt.adapt(ToReal{R}(), Additive(Picard(; damping = 1.0), p_02))
    st_nc = init_state(m_nc, prob_nc, x₀, nothing)
    Fnc₀ = residual!!(nothing, prob_nc, x₀, nothing)
    st_nc = start!!(st_nc, m_nc, prob_nc, x₀, Fnc₀, nothing, rnorm(Fnc₀))
    xnc₁, Fnc₁, _, info_nc = @inferred step!!(st_nc, m_nc, prob_nc, x₀, Fnc₀, nothing)
    x_nc1 = x₀ - one(R) * Fnc₀
    x_nc2 = x₀ - R(0.2) * Fnc₀
    F_nc1 = residual!!(nothing, prob_nc, x_nc1, nothing)
    F_nc2 = residual!!(nothing, prob_nc, x_nc2, nothing)
    @test rnorm(Fnc₁) <= min(rnorm(F_nc1), rnorm(F_nc2))
    @test xnc₁ === (rnorm(F_nc1) <= rnorm(F_nc2) ? x_nc1 : x_nc2)
    @test info_nc.failures == Int32(1)

    # A candidate that has already reached an exact root goes through the Gram solve like any
    # other. `F(x) = 2x - 4` from `x₀ = 1` puts each member on `x = 2`, where `F(x) = 0`
    # exactly; the combined step lands beside it by rounding, so the safeguard returns the
    # candidate bit for bit and counts one fallback.
    prob_root = NonlinearProblem((x, p) -> T(2) * x - T(4))
    x_r0 = one(T)
    Fr0 = residual!!(nothing, prob_root, x_r0, nothing)
    function root_step(m_raw, x₀r, F₀r)
        m_r = Adapt.adapt(ToReal{R}(), m_raw)
        st_r = start!!(
            init_state(m_r, prob_root, x₀r, nothing),
            m_r, prob_root, x₀r, F₀r, nothing, rnorm(F₀r))
        return @inferred step!!(st_r, m_r, prob_root, x₀r, F₀r, nothing)
    end
    for m_raw in (
        Additive(n_st),
        Additive(n_st, p_02),
        Additive(n_st, p_02, Picard(; damping = 0.5)))
        x_r, F_r, _, info_r = root_step(m_raw, x_r0, Fr0)
        @test x_r === T(2)
        @test iszero(F_r)
        @test info_r.failures == Int32(1)
    end

    # Small weight L1 norm `sum(abs, w) < 0.1` also triggers PETSc safeguard
    function f_sw(x, p)
        y1 = iszero(x[1]) ? one(T) : T(100) * x[1]
        y2 = iszero(x[2]) ? one(T) : T(100) * x[2]
        return SVector{2, T}(y1, y2)
    end
    prob_smallw = NonlinearProblem(f_sw)
    st_sw = init_state(m_mis, prob_smallw, x₀, nothing)
    Fsw₀ = residual!!(nothing, prob_smallw, x₀, nothing)
    st_sw = start!!(st_sw, m_mis, prob_smallw, x₀, Fsw₀, nothing, rnorm(Fsw₀))
    xsw₁, Fsw₁, _, info_sw = step!!(st_sw, m_mis, prob_smallw, x₀, Fsw₀, nothing)
    @test info_sw.failures == Int32(1)
    @test all(isfinite, xsw₁)
    @test all(isfinite, Fsw₁)
end

@testset "Check 1 in R3: safeguard and one composite fallback, $T, $(nameof(AT))" for T in ELTYPES,
    AT in ARRAY_BACKENDS

    R = real(T)
    f_mis!(F, x, p) = (F .= x;
        F[1:1] .= (x[1:1] .- one(T)) .+
                  T(50) .* (x[1:1] .- T(R(0.5))) .* (x[1:1] .- T(R(0.2)));
        F)
    function f_mis(x, p)
        y1 = x[1] - one(T) + T(50) * (x[1] - T(R(0.5))) * (x[1] - T(R(0.2)))
        return SVector{2, T}(y1, x[2])
    end
    m_mis = Adapt.adapt(ToReal{R}(), Additive(Picard(; damping = 0.5), Picard(;
        damping = 0.2)))

    # R1 reference step from the same start
    prob_s = NonlinearProblem(f_mis)
    x₀s = zero(SVector{2, T})
    F₀s = residual!!(nothing, prob_s, x₀s, nothing)
    st_s = start!!(init_state(m_mis, prob_s, x₀s, nothing), m_mis, prob_s, x₀s, F₀s,
        nothing, rnorm(F₀s))
    x_s, _, _, info_s = @inferred step!!(st_s, m_mis, prob_s, x₀s, F₀s, nothing)
    @test info_s.failures == Int32(1)

    # Both members step along `-F(x)`, so the Gram matrix is rank deficient and the rank rule
    # keeps the smaller-residual candidate. The safeguard reacts to that candidate's one-weight
    # secant weight, `|w| < 0.1`, and not to the trial residual, which the combination improves
    # here (about 2.3 against the kept candidate's 63.2). The two rules count one fallback
    # together, and the safeguard returns that candidate bit for bit, as in R1. `step!!` writes
    # `F₊` into the residual buffer it is given, so it gets a copy of `F₀`, and the reference
    # candidates below are built from `F(x₀)`.
    prob_a = NonlinearProblem(f_mis!)
    x₀ = AT(zeros(T, 2))
    F₀ = residual!!(similar(x₀), prob_a, x₀, nothing)
    st_a = start!!(init_state(m_mis, prob_a, x₀, nothing), m_mis, prob_a, x₀, F₀,
        nothing, rnorm(F₀))
    x₊, F₊, _, info_a = step!!(st_a, m_mis, prob_a, copy(x₀), copy(F₀), nothing)
    @test info_a.failures == Int32(1)
    @test Array(x₊) == Array(x_s)
    @test Array(F₊) == Array(residual!!(similar(x₀), prob_a, x₊, nothing))
    x_c1 = x₀ .- R(0.5) .* F₀
    x_c2 = x₀ .- R(0.2) .* F₀
    F_c1 = residual!!(similar(x₀), prob_a, x_c1, nothing)
    F_c2 = residual!!(similar(x₀), prob_a, x_c2, nothing)
    @test rnorm(F₊) <= min(rnorm(F_c1), rnorm(F_c2)) * (1 + 8 * eps(R))
end

@testset "Check 1 in R3: relative-residual safeguard trigger, $T, $(nameof(AT))" for T in ELTYPES,
    AT in ARRAY_BACKENDS

    R = real(T)
    # Two `Picard` steps from `(0, 0)` give candidates whose Gram matrix has full rank and whose
    # weights are not small, while the combined iterate's residual is about three times the
    # smaller candidate's: only the `fnorm₊ > ρ * f_best` rule of the safeguard can reject it. A
    # large `ρ` keeps the combined iterate; `ρ = 1` falls back to that candidate.
    function f_rho!(F, x, p)
        F .= x
        F[1:1] .= (x[1:1] .- T(2)) .+ T(5) .* x[1:1] .* (x[1:1] .- one(T)) .+ x[2:2]
        F[2:2] .= (x[2:2] .- one(T)) .+ x[1:1]
        return F
    end
    prob_rho = NonlinearProblem(f_rho!)
    x₀ = AT(zeros(T, 2))
    F₀ = residual!!(similar(x₀), prob_rho, x₀, nothing)
    members_rho = (Picard(; damping = 0.5), Picard(; damping = 0.2))

    function step_rho(ρ)
        m = Adapt.adapt(ToReal{R}(), Additive(members_rho...; weights = OptimalWeights(ρ)))
        st = start!!(init_state(m, prob_rho, x₀, nothing), m, prob_rho, x₀, F₀,
            nothing, rnorm(F₀))
        return step!!(st, m, prob_rho, copy(x₀), copy(F₀), nothing)
    end

    f_best = min(rnorm(residual!!(similar(x₀), prob_rho, x₀ .- R(0.5) .* F₀, nothing)),
        rnorm(residual!!(similar(x₀), prob_rho, x₀ .- R(0.2) .* F₀, nothing)))

    x_big, F_big, _, info_big = step_rho(R(1e6))
    @test info_big.failures == Int32(0)
    @test all(isfinite, x_big) && all(isfinite, F_big)
    @test rnorm(F_big) > 2 * f_best

    x_sf, F_sf, _, info_sf = step_rho(one(R))
    @test info_sf.failures == Int32(1)
    @test rnorm(F_sf) <= f_best * (1 + 8 * eps(R))
    @test Array(F_sf) == Array(residual!!(similar(x₀), prob_rho, x_sf, nothing))
end

@testset "Check 2: Singular Gram matrix in Additive (MU3, MU4), $T" for T in ELTYPES
    R = real(T)
    prob_q = NonlinearProblem(R1QuadF(T(2), T(3)))
    x₀ = SVector{2, T}(T(2), T(2))
    Fx₀ = residual!!(nothing, prob_q, x₀, nothing)

    # Equal candidates (`Newton`, `Newton`)
    n_st = Newton(; globalization = Static())
    m_eq = Adapt.adapt(ToReal{R}(), Additive(n_st, n_st))
    st_eq = init_state(m_eq, prob_q, x₀, nothing)
    st_eq = start!!(st_eq, m_eq, prob_q, x₀, Fx₀, nothing, rnorm(Fx₀))
    x₁_eq, F₁_eq, _, info_eq = @inferred step!!(st_eq, m_eq, prob_q, x₀, Fx₀, nothing)
    @test all(isfinite, x₁_eq) && all(isfinite, F₁_eq)
    @test info_eq.failures == Int32(1)

    # Two equal candidates that both land on an exact root: the Gram matrix is singular, the
    # 1-weight step lands beside the root by rounding, and the safeguard returns the candidate
    # bit for bit, the pair of rules counting one fallback.
    prob_root = NonlinearProblem((x, p) -> T(2) * x - T(4))
    x_r0 = one(T)
    Fr0 = residual!!(nothing, prob_root, x_r0, nothing)
    m_root = Adapt.adapt(ToReal{R}(), Additive(n_st, n_st))
    st_root = init_state(m_root, prob_root, x_r0, nothing)
    st_root = start!!(st_root, m_root, prob_root, x_r0, Fr0, nothing, rnorm(Fr0))
    x₁_root, F₁_root, _, info_root = @inferred step!!(
        st_root, m_root, prob_root, x_r0, Fr0, nothing)
    @test x₁_root === T(2)
    @test iszero(F₁_root)
    @test info_root.failures == Int32(1)

    # Nearly-collinear candidates (`Picard(0.8)` vs `Picard(0.8 * (1 - 1e-5))`) where
    # candidate 1 has smaller residual than candidate 2 on `F(x) = x - [1, 1]` from `[2, 2]`
    prob_lin = NonlinearProblem((x, p) -> x .- SVector{2, T}(one(T), one(T)))
    Fl₀ = residual!!(nothing, prob_lin, x₀, nothing)
    p_a = Picard(; damping = 0.8)
    p_b = Picard(; damping = 0.8 * (1 - 1e-5))
    m_col12 = Adapt.adapt(ToReal{R}(), Additive(p_a, p_b))
    st_c12 = init_state(m_col12, prob_lin, x₀, nothing)
    st_c12 = start!!(st_c12, m_col12, prob_lin, x₀, Fl₀, nothing, rnorm(Fl₀))
    xc12, _, _, infoc12 = @inferred step!!(st_c12, m_col12, prob_lin, x₀, Fl₀, nothing)
    @test infoc12.failures == Int32(1)
    @test isapprox(xc12, SVector{2, T}(one(T), one(T)); rtol = 16 * eps(R))

    # Reverse order so candidate 2 has the smaller residual (catches MU4)
    p_c = Picard(; damping = 0.5 * (1 - 1e-5))
    p_d = Picard(; damping = 0.5)
    m_col21 = Adapt.adapt(ToReal{R}(), Additive(p_c, p_d))
    st_c21 = init_state(m_col21, prob_q, x₀, nothing)
    st_c21 = start!!(st_c21, m_col21, prob_q, x₀, Fx₀, nothing, rnorm(Fx₀))
    xc21, Fc21, _, infoc21 = @inferred step!!(st_c21, m_col21, prob_q, x₀, Fx₀, nothing)
    @test all(isfinite, xc21) && all(isfinite, Fc21)
    @test infoc21.failures == Int32(1)
end

@testset "Check 3: FixedWeights combination and real weights, $T" for T in ELTYPES
    R = real(T)
    prob_q = NonlinearProblem(R1QuadF(T(2), T(3)))
    x₀ = SVector{2, T}(T(2), T(2))
    Fx₀ = residual!!(nothing, prob_q, x₀, nothing)

    m₁ = Adapt.adapt(ToReal{R}(), Newton(; globalization = Static()))
    m₂ = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.25))
    w₁, w₂ = R(0.6), R(0.4)
    m_fw = Adapt.adapt(ToReal{R}(), Additive(m₁, m₂; weights = FixedWeights(w₁, w₂)))
    @test eltype(m_fw.weights.weights) === R

    s₁ = init_state(m₁, prob_q, x₀, nothing)
    s₁ = start!!(s₁, m₁, prob_q, x₀, Fx₀, nothing, rnorm(Fx₀))
    x_cand1, _, _, _ = step!!(s₁, m₁, prob_q, x₀, Fx₀, nothing)
    s₂ = init_state(m₂, prob_q, x₀, nothing)
    s₂ = start!!(s₂, m₂, prob_q, x₀, Fx₀, nothing, rnorm(Fx₀))
    x_cand2, _, _, _ = step!!(s₂, m₂, prob_q, x₀, Fx₀, nothing)

    s_fw = init_state(m_fw, prob_q, x₀, nothing)
    s_fw = start!!(s_fw, m_fw, prob_q, x₀, Fx₀, nothing, rnorm(Fx₀))
    x_fw, _, _, info_fw = @inferred step!!(s_fw, m_fw, prob_q, x₀, Fx₀, nothing)
    expected = x₀ + (w₁ * (x_cand1 - x₀) + w₂ * (x_cand2 - x₀))
    @test x_fw === expected
    @test info_fw.failures == Int32(0)
end

@testset "Check 3 in R3: five FixedWeights members, $T, $(nameof(AT))" for T in ELTYPES,
    AT in ARRAY_BACKENDS

    R = real(T)
    prob_a = NonlinearProblem(R3QuadF(T(2), T(3)))
    dampings = (R(0.5), R(0.4), R(0.3), R(0.2), R(0.1))
    weights = (R(0.1), R(0.2), R(0.3), R(0.2), R(0.2))
    members = ntuple(i -> Picard(; damping = dampings[i]), 5)
    m_fw = Adapt.adapt(ToReal{R}(), Additive(members...; weights = FixedWeights(weights...)))

    x₀ = AT(T[2, 2])
    x_ref = Array(x₀)
    F₀ = residual!!(similar(x₀), prob_a, x₀, nothing)
    x_in = copy(x₀)

    # The five candidates of the standalone members. The fifth member is the one a combination
    # loop that stops at `N` members loses, and its direction and weight are non-zero.
    xs = map(1:5) do i
        m_i = Adapt.adapt(ToReal{R}(), members[i])
        x_i = copy(x₀)
        Fx_i = copy(F₀)
        s_i = start!!(init_state(m_i, prob_a, x_i, nothing), m_i, prob_a, x_i, Fx_i,
            nothing, rnorm(Fx_i))
        x_cand, _, _, _ = step!!(s_i, m_i, prob_a, x_i, Fx_i, nothing)
        return Array(x_cand)
    end
    @test !iszero(weights[5])
    @test rnorm(xs[5] - x_ref) > 0

    st_fw = start!!(init_state(m_fw, prob_a, x_in, nothing), m_fw, prob_a, x_in, F₀,
        nothing, rnorm(F₀))
    x₊, _, _, info_fw = step!!(st_fw, m_fw, prob_a, copy(x₀), copy(F₀), nothing)
    @test info_fw.failures == Int32(0)

    d = zero(x_ref)
    for i in 1:5
        d = d .+ weights[i] .* (xs[i] - x_ref)
    end
    @test isapprox(Array(x₊), x_ref .+ d; rtol = 8 * eps(R))
end

@testset "Check 4: R3 isolation (MU5), $T" for T in ELTYPES
    R = real(T)
    prob_a = NonlinearProblem(R3QuadF(T(2), T(3)))
    for AT in ARRAY_BACKENDS
        x₀ = AT(T[2, 2])
        Fx₀ = AT(T[2, 1])

        seen_x2 = Ref(zeros(T, 0))
        seen_Fx2 = Ref(zeros(T, 0))
        m₁ = Picard(; damping = 0.5)
        m₂ = RecordingMember(Picard(; damping = 0.25), seen_x2, seen_Fx2)
        w_half = FixedWeights(R(0.5), R(0.5))
        m_iso = Adapt.adapt(ToReal{R}(), Additive(m₁, m₂; weights = w_half))

        x_in = copy(x₀)
        Fx_in = copy(Fx₀)
        st_iso = init_state(m_iso, prob_a, x_in, nothing)
        st_iso = start!!(st_iso, m_iso, prob_a, x_in, Fx_in, nothing, rnorm(Fx_in))
        x_out, _, st_iso, _ = step!!(st_iso, m_iso, prob_a, x_in, Fx_in, nothing)

        # Member 2 must see the exact unmodified `x₀` and `Fx₀` (catches MU5!)
        @test seen_x2[] == Array(x₀)
        @test seen_Fx2[] == Array(Fx₀)

        # Candidate buffers `st_iso.xs[1]` and `st_iso.xs[2]` match standalone steps
        x_sa1 = copy(x₀)
        Fx_sa1 = copy(Fx₀)
        m₁_r = Adapt.adapt(ToReal{R}(), m₁)
        s_sa1 = init_state(m₁_r, prob_a, x_sa1, nothing)
        s_sa1 = start!!(s_sa1, m₁_r, prob_a, x_sa1, Fx_sa1, nothing, rnorm(Fx_sa1))
        x_sa1, _, _, _ = step!!(s_sa1, m₁_r, prob_a, x_sa1, Fx_sa1, nothing)
        @test Array(st_iso.xs[1]) == Array(x_sa1)

        x_sa2 = copy(x₀)
        Fx_sa2 = copy(Fx₀)
        m₂_r = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.25))
        s_sa2 = init_state(m₂_r, prob_a, x_sa2, nothing)
        s_sa2 = start!!(s_sa2, m₂_r, prob_a, x_sa2, Fx_sa2, nothing, rnorm(Fx_sa2))
        x_sa2, _, _, _ = step!!(s_sa2, m₂_r, prob_a, x_sa2, Fx_sa2, nothing)
        @test Array(st_iso.xs[2]) == Array(x_sa2)
        d_sa1 = Array(x_sa1) .- Array(x₀)
        d_sa2 = Array(x_sa2) .- Array(x₀)
        x_comb = Array(x₀) .+ R(0.5) .* d_sa1 .+ R(0.5) .* d_sa2
        @test isapprox(Array(x_out), x_comb; rtol = 8 * eps(R))
    end
end

@testset "Check 5 & D8: un-started step!! and finish!!, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem(R1QuadF(T(2), T(3)))
    x₀ = SVector{2, T}(T(2), T(2))
    Fx₀ = residual!!(nothing, prob, x₀, nothing)

    n_bt = Newton(; globalization = Backtracking())
    p_02 = Picard(; damping = 0.2)
    m_ls = Adapt.adapt(ToReal{T}(), Additive(n_bt, p_02))
    st_unstarted = init_state(m_ls, prob, x₀, nothing)
    x_u, F_u, _, info_u = @inferred step!!(st_unstarted, m_ls, prob, x₀, Fx₀, nothing)
    @test info_u.code === NONFINITE
    @test iszero(info_u.stepnorm)
    @test x_u === x₀
    @test F_u === Fx₀

    m_svd = Additive(Newton(; linsolve = SVDFactorization()), p_02)
    _, status_svd = solve(prob, x₀, m_svd)
    @test status_svd.code === SUCCESS
    @test status_svd.rank === Int32(-1)
    @test isnan(status_svd.nullnorm)
    @test isnan(status_svd.gradratio)
end

@testset "Check 6: R1/R2/R3 parity, isbits, @inferred, 0 allocs, $T" for T in ELTYPES
    R = real(T)
    prob_s = NonlinearProblem(R1QuadF(T(2), T(3)))
    prob_a = NonlinearProblem(R3QuadF(T(2), T(3)), R3QuadJ())
    x₀_s = SVector{2, T}(T(1.5), T(1.5))
    x₀_a = T[T(1.5), T(1.5)]
    x₀_jl = JLArray(x₀_a)

    n_st = Newton(; globalization = Static())
    p_02 = Picard(; damping = 0.2)
    m_opt6 = Additive(n_st, p_02)
    m_fix6 = Additive(n_st, p_02; weights = FixedWeights(0.7, 0.3))
    for m in (m_opt6, m_fix6)
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
        x_exact = SVector{2, T}(sqrt(T(2)), sqrt(T(3)))
        @test isapprox(xs_full, x_exact; rtol = 16 * sqrt(eps(R)))

        if Base.JLOptions().check_bounds != 1
            # Skipped where a run forces `--check-bounds=yes`.
            @test r1_solve_alloc(prob_s, x₀_s, m_r) == 0
        end
        JET.@test_opt solve(prob_s, x₀_s, m_r; max_iterations = 2)

        old_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            x_work = copy(x₀_a)
            solver = init(prob_a, x_work, m)
            @test r3_add_alloc(solver, x_work, x₀_a) == 0
        finally
            BLAS.set_num_threads(old_threads)
        end

        # R2 solve_batch! on CPU()
        X_batch = [x₀_s for _ in 1:4]
        S_batch = Vector{SolverStatus{R}}(undef, 4)
        solve_batch!(X_batch, S_batch, prob_s, m)
        @test all(s -> s.code == SUCCESS, S_batch)
        @test all(x -> isapprox(x, xs_full; rtol = 16 * eps(R)), X_batch)
        X_batch2 = [x₀_s for _ in 1:4]
        counts = solve_batch!(X_batch2, prob_s, m)
        @test counts.SUCCESS == 4
        @test all(x -> isapprox(x, xs_full; rtol = 16 * eps(R)), X_batch2)
    end
end
