using Adapt: Adapt
using ForwardDiff: ForwardDiff
using GeometricSolvers
using GeometricSolvers: AndersonHistory, AndersonState, NONFINITE, LINESEARCH_FAILED,
                        MAXITERS, SINGULAR, STALLED, SUCCESS, StepInfo, ToReal, finish!!,
                        init_state, isconverged, push_history!!, rdot, rnorm, solve!!,
                        solve_broyden_weights, solve_broyden_weights!, solve_weights,
                        solve_weights!, start!!, step!!, step_multisecant!!
using GPUArraysCore: @allowscalar
using JET: JET
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions as KA
using LinearAlgebra: BLAS, LowerTriangular, cond, norm
using StaticArrays: @SMatrix, MVector, SMatrix, StaticVector, SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")
include("../helpers/adproblems.jl")

# Inner member whose step returns a prescribed ReturnCode and candidate for edge tests.
struct ScriptedBroydenMember{C, F}
    code::C
    step_fn::F
end

GeometricSolvers.init_state(::ScriptedBroydenMember, prob, x, p) = nothing
GeometricSolvers.start!!(st, ::ScriptedBroydenMember, prob, x, Fx, p, fnorm₀) = st
GeometricSolvers.sync_merit(st, ::ScriptedBroydenMember, merit::Real) = st
function GeometricSolvers.finish!!(st, ::ScriptedBroydenMember, prob, x, Fx, status)
    return x, status, st
end

function GeometricSolvers.step!!(
        st, m::ScriptedBroydenMember, prob, x::Union{StaticVector, Number}, Fx, p)
    R = real(eltype(x))
    x_cand = m.step_fn(x)
    F_cand = prob.F(x_cand, p)
    stepnorm = rnorm(x_cand - x)
    failures = m.code == SUCCESS ? Int32(0) : Int32(1)
    return x_cand, F_cand, st,
    StepInfo{R}(m.code, rnorm(F_cand), stepnorm, failures, false, false)
end

function GeometricSolvers.step!!(
        st, m::ScriptedBroydenMember, prob, x::AbstractVector{T}, Fx::AbstractVector{T},
        p) where {T}
    R = real(T)
    x_cand = m.step_fn(x)
    stepnorm = rnorm(x_cand - x)
    copyto!(x, x_cand)
    prob.F(Fx, x, p)
    failures = m.code == SUCCESS ? Int32(0) : Int32(1)
    return x, Fx, st,
    StepInfo{R}(m.code, rnorm(Fx), stepnorm, failures, false, false)
end

function _r1_broyden_step_alloc_probe(st, method, prob, x, Fx, p)
    step!!(st, method, prob, x, Fx, p)
    return @allocated step!!(st, method, prob, x, Fx, p)
end

function _r1_broyden_solve_alloc_probe(st, prob, x₀, method, p, opt)
    solve!!(st, prob, x₀, method, p, opt)
    return @allocated solve!!(st, prob, x₀, method, p, opt)
end

function _r3_broyden_step_alloc_probe(st, method, prob, x, x_save, Fx, Fx_save, p)
    nt = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        copyto!(x, x_save)
        copyto!(Fx, Fx_save)
        step!!(st, method, prob, x, Fx, p)
        copyto!(x, x_save)
        copyto!(Fx, Fx_save)
        return @allocated step!!(st, method, prob, x, Fx, p)
    finally
        BLAS.set_num_threads(nt)
    end
end

function _r3_broyden_solve_alloc_probe(solver, x, x₀)
    nt = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        copyto!(x, x₀)
        solve!(solver, x)
        copyto!(x, x₀)
        return @allocated solve!(solver, x)
    finally
        BLAS.set_num_threads(nt)
    end
end

@testset "Broyden constructors, validation, and Adapt" begin
    b_def = Broyden()
    @test b_def isa Broyden{4, <:Picard}
    @test b_def.member.damping == 1.0
    @test b_def.c_s == 0.2
    @test b_def.κ_bar === nothing
    @test b_def.restart_factor == 10.0
    @test b_def.M_alpha == 25.0
    @test b_def.safeguard_cos === nothing
    @test b_def.grow_depth === false

    b4 = Broyden(Picard(), Val(4))
    b4_int = Broyden(Picard(), 4)
    b4_val = Broyden(Val(4))
    b4_int_def = Broyden(4)
    @test typeof(b4) === typeof(b4_int) === typeof(b4_val) === typeof(b4_int_def) ===
          typeof(b_def)

    b_damp = Broyden(Val(3); damping = 0.5, c_s = 0.25, κ_bar = 1e5,
        restart_factor = 8.0, M_alpha = 15.0, safeguard_cos = 0.85, grow_depth = true)
    @test b_damp isa Broyden{3, <:Picard, Float64, Float64, Float64}
    @test b_damp.member.damping == 0.5
    @test b_damp.c_s == 0.25
    @test b_damp.κ_bar == 1e5
    @test b_damp.restart_factor == 8.0
    @test b_damp.M_alpha == 15.0
    @test b_damp.safeguard_cos == 0.85
    @test b_damp.grow_depth === true

    b_damp_int = Broyden(3; damping = 0.5, c_s = 0.25)
    @test b_damp_int isa Broyden{3, <:Picard}
    @test b_damp_int.member.damping == 0.5
    @test b_damp_int.c_s == 0.25

    # Edge 6: Other precision via Adapt.adapt(ToReal{Float32}(), ...)
    b32 = Adapt.adapt(ToReal{Float32}(), b_damp)
    @test b32 isa Broyden{3}
    @test b32.member.damping isa Float32
    @test b32.c_s isa Float32
    @test b32.κ_bar isa Float32
    @test b32.restart_factor isa Float32
    @test b32.M_alpha isa Float32
    @test b32.safeguard_cos isa Float32
    @test b32.grow_depth === true

    b32_def = Adapt.adapt(ToReal{Float32}(), b4)
    @test b32_def isa Broyden{4}
    @test b32_def.c_s isa Float32
    @test b32_def.κ_bar === nothing
    @test b32_def.safeguard_cos === nothing

    # ToReal{Float32} rejects out-of-range member or Broyden parameters
    @test_throws ArgumentError Adapt.adapt(
        ToReal{Float32}(), Broyden(Picard(), Val(2); κ_bar = 1e300))
    @test_throws ArgumentError Adapt.adapt(
        ToReal{Float32}(), Broyden(Picard(; damping = 1e-50), Val(2)))

    # Edge 8 & Edge 9: invalid depth and threshold parameters throw ArgumentError
    @test_throws ArgumentError Broyden(Picard(), Val(0))
    @test_throws ArgumentError Broyden(Picard(), Val(-2))
    @test_throws ArgumentError Broyden(Picard(), Val(:four))
    @test_throws ArgumentError Broyden(Picard(), 0)
    @test_throws ArgumentError Broyden(Picard(), -1)
    @test_throws ArgumentError Broyden(Picard(), true)
    @test_throws ArgumentError Broyden(0)
    @test_throws ArgumentError Broyden(-1)
    @test_throws ArgumentError Broyden(true)

    @test_throws ArgumentError Broyden(Picard(), 4; c_s = 0.0)
    @test_throws ArgumentError Broyden(Picard(), 4; c_s = 1.0)
    @test_throws ArgumentError Broyden(Picard(), 4; c_s = -0.1)
    @test_throws ArgumentError Broyden(Picard(), 4; c_s = NaN)
    @test_throws ArgumentError Broyden(Picard(), 4; c_s = Inf)

    @test_throws ArgumentError Broyden(Picard(), 4; κ_bar = 1.0)
    @test_throws ArgumentError Broyden(Picard(), 4; κ_bar = 0.5)
    @test_throws ArgumentError Broyden(Picard(), 4; κ_bar = NaN)
    @test_throws ArgumentError Broyden(Picard(), 4; κ_bar = Inf)

    @test_throws ArgumentError Broyden(Picard(), 4; restart_factor = 2.9)
    @test_throws ArgumentError Broyden(Picard(), 4; restart_factor = NaN)
    @test_throws ArgumentError Broyden(Picard(), 4; restart_factor = Inf)

    @test_throws ArgumentError Broyden(Picard(), 4; M_alpha = 0.9)
    @test_throws ArgumentError Broyden(Picard(), 4; M_alpha = NaN)
    @test_throws ArgumentError Broyden(Picard(), 4; M_alpha = Inf)

    @test_throws ArgumentError Broyden(Picard(), 4; safeguard_cos = 0.0)
    @test_throws ArgumentError Broyden(Picard(), 4; safeguard_cos = 1.0)
    @test_throws ArgumentError Broyden(Picard(), 4; safeguard_cos = -0.2)
    @test_throws ArgumentError Broyden(Picard(), 4; safeguard_cos = NaN)
    @test_throws ArgumentError Broyden(Picard(), 4; safeguard_cos = Inf)
end

@testset "Check 2 & Edges 3, 4, 5: solve_broyden_weights and solve_broyden_weights!" begin
    @testset "$R" for R in REAL_ELTYPES
        G = @SMatrix [R(4) R(2) R(1);
                      R(2) R(5) R(3 // 2);
                      R(1) R(3 // 2) R(6)]
        b = SVector{3, R}(R(4), R(7), R(17 // 2))
        kept1 = SVector{3, Int}(1, 0, 0)
        kept2 = SVector{3, Int}(1, 2, 0)
        kept3 = SVector{3, Int}(1, 2, 3)

        # Edge 3: p = 0 returns zero weights
        γ0 = solve_broyden_weights(G, b, kept1, 0)
        @test iszero(γ0)

        # Check 2 & Edge 4: p = 1 agrees with Anderson's solve_weights within 4 * eps(R) * abs(γ₁)
        γ1_b = solve_broyden_weights(G, b, kept1, 1)
        γ1_a = solve_weights(G, b, kept1, 1)
        @test abs(γ1_b[1] - γ1_a[1]) <= 4 * eps(R) * abs(γ1_a[1])
        @test iszero(γ1_b[2]) && iszero(γ1_b[3])

        # Check 2: p = 2 solves (L_G + λI) γ = b_kept by forward substitution (catches MU1, MU2)
        γ2_b = solve_broyden_weights(G, b, kept2, 2)
        γ2_a = solve_weights(G, b, kept2, 2)
        λ2 = eps(R) * ((G[1, 1] + G[2, 2]) / R(2))
        γ2_ref1 = b[1] / (G[1, 1] + λ2)
        γ2_ref2 = (b[2] - G[2, 1] * γ2_ref1) / (G[2, 2] + λ2)
        @test abs(γ2_b[1] - γ2_ref1) <= 4 * 2 * eps(R) * maximum(abs, γ2_b)
        @test abs(γ2_b[2] - γ2_ref2) <= 4 * 2 * eps(R) * maximum(abs, γ2_b)
        @test iszero(γ2_b[3])
        @test γ2_b != γ2_a

        # Check 2: p = 3 lower-triangular forward substitution agreement within 4 * p * eps(R) * maximum(abs, γ)
        γ3_b = solve_broyden_weights(G, b, kept3, 3)
        λ3 = eps(R) * ((G[1, 1] + G[2, 2] + G[3, 3]) / R(3))
        γ3_ref1 = b[1] / (G[1, 1] + λ3)
        γ3_ref2 = (b[2] - G[2, 1] * γ3_ref1) / (G[2, 2] + λ3)
        γ3_ref3 = (b[3] - G[3, 1] * γ3_ref1 - G[3, 2] * γ3_ref2) / (G[3, 3] + λ3)
        γ3_ref = SVector{3, R}(γ3_ref1, γ3_ref2, γ3_ref3)
        @test maximum(abs, γ3_b - γ3_ref) <= 4 * 3 * eps(R) * maximum(abs, γ3_b)

        # R3 in-place solve_broyden_weights! matches R1 solve_broyden_weights
        G_mat = Matrix(G)
        b_vec = Vector(b)
        γ_vec = fill(R(NaN), 3)
        kept_vec = [1, 2, 3]
        solve_broyden_weights!(γ_vec, G_mat, b_vec, kept_vec, 0)
        @test iszero(γ_vec)
        solve_broyden_weights!(γ_vec, G_mat, b_vec, kept_vec, 1)
        @test abs(γ_vec[1] - γ1_b[1]) <= 4 * eps(R) * abs(γ1_b[1])
        @test iszero(γ_vec[2]) && iszero(γ_vec[3])
        solve_broyden_weights!(γ_vec, G_mat, b_vec, kept_vec, 2)
        @test maximum(abs, γ_vec[1:2] - γ2_b[1:2]) <= 4 * 2 * eps(R) * maximum(abs, γ2_b)
        @test iszero(γ_vec[3])
        solve_broyden_weights!(γ_vec, G_mat, b_vec, kept_vec, 3)
        @test maximum(abs, γ_vec - Vector(γ3_ref)) <= 4 * 3 * eps(R) * maximum(abs, γ3_b)
        @test (@allocated solve_broyden_weights!(γ_vec, G_mat, b_vec, kept_vec, 3)) == 0

        # Edge 5: zero Gram diagonal with zero right-hand side gives finite 0 when tr(G_K) > 0,
        # and non-finite (NaN) when tr(G_K) == 0 so weights_acceptable rejects it.
        G_zero = @SMatrix [R(4) zero(R); zero(R) zero(R)]
        b_zero = SVector{2, R}(R(4), zero(R))
        kept_z = SVector{2, Int}(1, 2)
        γ_z = solve_broyden_weights(G_zero, b_zero, kept_z, 2)
        @test all(isfinite, γ_z)
        @test iszero(γ_z[2])

        γ_z_r3 = fill(R(NaN), 2)
        solve_broyden_weights!(γ_z_r3, Matrix(G_zero), Vector(b_zero), [1, 2], 2)
        @test all(isfinite, γ_z_r3)
        @test iszero(γ_z_r3[2])

        G_allzero = @SMatrix [zero(R) zero(R); zero(R) zero(R)]
        γ_allz = solve_broyden_weights(
            G_allzero, SVector{2, R}(zero(R), zero(R)), kept_z, 2)
        @test !all(isfinite, γ_allz)

        γ_allz_r3 = fill(zero(R), 2)
        solve_broyden_weights!(
            γ_allz_r3, Matrix(G_allzero), [zero(R), zero(R)], [1, 2], 2)
        @test !all(isfinite, γ_allz_r3)
    end
end

@testset "Check 2: shared AndersonState, AndersonHistory, push_history!!, and step_multisecant!!" begin
    prob_r1 = NonlinearProblem((x, p) -> SVector(x[1]^2 - 2.0, x[2] - 1.0))
    m_a = Anderson(Picard(; damping = 0.5), Val(3))
    m_b = Broyden(Picard(; damping = 0.5), Val(3))
    x0 = SVector(1.5, 0.5)

    st_a = init_state(m_a, prob_r1, x0, nothing)
    st_b = init_state(m_b, prob_r1, x0, nothing)
    @test isbits(st_b)
    @test typeof(st_b) === typeof(st_a)
    @test st_b.history isa AndersonHistory{3}
    @test isbits(st_b.history)
    @test typeof(st_b.history) === typeof(st_a.history)
    @test typeof(st_b.ΔX) === typeof(st_a.ΔX)
    @test typeof(st_b.ΔW) === typeof(st_a.ΔW)
    @test st_b.Δx === st_b.ΔX
    @test st_b.Δr === st_b.ΔW
    @test st_b.history.Δx === st_b.ΔX
    @test st_b.history.Δr === st_b.ΔW
    @test :history in propertynames(st_b)
    @test :Δx in propertynames(st_b)
    @test :Δr in propertynames(st_b)

    # Edge 8: R1 init_state with m > 4 throws ArgumentError
    @test_throws ArgumentError init_state(
        Broyden(Picard(), Val(5)), prob_r1, x0, nothing)
    @test_throws ArgumentError init(prob_r1, x0, Broyden(Picard(), Val(5)))

    # Edge 1: un-started step!! returns NONFINITE and leaves x, Fx unchanged
    Fx0 = prob_r1.F(x0, nothing)
    x_u, Fx_u, st_u, info_u = step!!(st_b, m_b, prob_r1, x0, Fx0, nothing)
    @test info_u.code == NONFINITE
    @test iszero(info_u.stepnorm)
    @test x_u == x0
    @test Fx_u == Fx0

    # Check 2 & Edge 4: Shared history on identical iterate sequence, p = 1 agreement, p >= 2 difference
    A_mat = @SMatrix [2.0 0.8 0.3; 0.4 1.7 0.5; 0.2 0.3 1.4]
    b_vec = SVector(1.0, -0.5, 0.8)
    prob_lin = NonlinearProblem((x, p) -> A_mat * x - b_vec)
    z3 = SVector(0.0, 0.0, 0.0)
    st_a1 = start!!(init_state(m_a, prob_lin, z3, nothing),
        m_a, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    st_b1 = start!!(init_state(m_b, prob_lin, z3, nothing),
        m_b, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    seq_x = (z3, SVector(0.5, -0.25, 0.4), SVector(0.35, -0.4, 0.55))
    for k in 1:3
        xk = seq_x[k]
        fk = prob_lin.F(xk, nothing)
        xa, _, st_a1, _ = step!!(st_a1, m_a, prob_lin, xk, fk, nothing)
        xb, _, st_b1, _ = step!!(st_b1, m_b, prob_lin, xk, fk, nothing)
        @test st_b1.ΔX === st_a1.ΔX
        @test st_b1.ΔW === st_a1.ΔW
        @test st_b1.history === st_a1.history
        if k == 2
            @test rnorm(xb - xa) <= 4 * eps(Float64) * rnorm(xb)
        elseif k == 3
            @test xb != xa
        end
    end

    # Edge 4: Broyden{1} and Anderson{1} agree within 4 * eps(R) * rnorm(x₊) in R1 and R3
    m_a1 = Anderson(Picard(; damping = 0.5), Val(1))
    m_b1 = Broyden(Picard(; damping = 0.5), Val(1))
    st_m1a = start!!(init_state(m_a1, prob_lin, z3, nothing),
        m_a1, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    st_m1b = start!!(init_state(m_b1, prob_lin, z3, nothing),
        m_b1, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    for k in 1:3
        xk = seq_x[k]
        fk = prob_lin.F(xk, nothing)
        xa, _, st_m1a, _ = step!!(st_m1a, m_a1, prob_lin, xk, fk, nothing)
        xb, _, st_m1b, _ = step!!(st_m1b, m_b1, prob_lin, xk, fk, nothing)
        @test rnorm(xb - xa) <= 4 * eps(Float64) * rnorm(xb)
    end

    # R3 shared state, history bit-for-bit equality on identical sequence, and un-started step!! guard
    prob_r3 = NonlinearProblem((F, x, p) -> (
        F .= Matrix(A_mat) * x .- Vector(b_vec); nothing))
    x0_r3 = [0.0, 0.0, 0.0]
    Fx0_r3 = similar(x0_r3)
    prob_r3.F(Fx0_r3, x0_r3, nothing)
    st_a_r3 = init_state(m_a, prob_r3, x0_r3, nothing)
    st_b_r3 = init_state(m_b, prob_r3, x0_r3, nothing)
    @test typeof(st_b_r3) === typeof(st_a_r3)
    @test typeof(st_b_r3.history) === typeof(st_a_r3.history)
    @test typeof(st_b_r3.ΔX) === typeof(st_a_r3.ΔX)
    @test typeof(st_b_r3.ΔW) === typeof(st_a_r3.ΔW)
    @test st_b_r3.Δx === st_b_r3.ΔX
    @test st_b_r3.Δr === st_b_r3.ΔW

    x_u3 = copy(x0_r3)
    Fx_u3 = copy(Fx0_r3)
    _, _, _, info_u3 = step!!(st_b_r3, m_b, prob_r3, x_u3, Fx_u3, nothing)
    @test info_u3.code == NONFINITE
    @test iszero(info_u3.stepnorm)
    @test x_u3 == x0_r3

    start!!(st_a_r3, m_a, prob_r3, x0_r3, Fx0_r3, nothing, rnorm(Fx0_r3))
    start!!(st_b_r3, m_b, prob_r3, x0_r3, Fx0_r3, nothing, rnorm(Fx0_r3))
    for k in 1:3
        xk_a = Vector(seq_x[k])
        xk_b = Vector(seq_x[k])
        fk_a = similar(xk_a)
        fk_b = similar(xk_b)
        prob_r3.F(fk_a, xk_a, nothing)
        prob_r3.F(fk_b, xk_b, nothing)
        step!!(st_a_r3, m_a, prob_r3, xk_a, fk_a, nothing)
        step!!(st_b_r3, m_b, prob_r3, xk_b, fk_b, nothing)
        @test isequal(st_b_r3.ΔX, st_a_r3.ΔX)
        @test isequal(st_b_r3.ΔW, st_a_r3.ΔW)
        if k == 2
            @test rnorm(xk_b - xk_a) <= 4 * eps(Float64) * rnorm(xk_b)
        elseif k == 3
            @test xk_b != xk_a
        end
    end
end

@testset "Check 1, Check 3, Check 4, and Edge 7: R1, R2, and R3 convergence, agreement, and allocations" begin
    @testset "R1 SVector and scalar convergence, allocations, and inference ($T)" for T in ELTYPES
        R = real(T)
        opt = Options(R)
        method_br = Broyden(Picard(; damping = 1 / 10), Val(4))

        # Scalar R1 problem: F(x, p) = x * x + p * x, J(x, p) = x + x + p, p = T(5) from x₀ = T(1/2)
        F_sc(x, p) = x * x + p * x
        J_sc(x, p) = x + x + p
        prob_sc = NonlinearProblem(F_sc, J_sc)
        p_sc = T(5)
        x0_sc = T(1 / 2)
        x_sc_n, st_sc_n = solve(prob_sc, x0_sc, Newton(), p_sc)
        x_sc_br, st_sc_br = @inferred solve(prob_sc, x0_sc, method_br, p_sc)
        abstol_sc = max(opt.f_abstol, opt.f_reltol * rnorm(F_sc(x0_sc, p_sc)))
        cond_sc = one(R)
        @test st_sc_n.code == SUCCESS
        @test st_sc_br.code == SUCCESS
        @test rnorm(F_sc(x_sc_br, p_sc)) <= abstol_sc
        @test rnorm(x_sc_br - x_sc_n) <= abstol_sc * cond_sc

        for m_val in (Val(1), Val(2), Val(3), Val(4))
            bm = Broyden(Picard(; damping = 1 / 10), m_val)
            st_m = init_state(bm, prob_sc, x0_sc, p_sc)
            @test isbits(st_m)
            Fx0_sc = F_sc(x0_sc, p_sc)
            st_ms = start!!(st_m, bm, prob_sc, x0_sc, Fx0_sc, p_sc, rnorm(Fx0_sc))
            @inferred step!!(st_ms, bm, prob_sc, x0_sc, Fx0_sc, p_sc)
            if Base.JLOptions().check_bounds != 1
                # Skipped where a run forces --check-bounds=yes.
                @test _r1_broyden_step_alloc_probe(
                    st_ms, bm, prob_sc, x0_sc, Fx0_sc, p_sc) == 0
                @test _r1_broyden_solve_alloc_probe(
                    st_m, prob_sc, x0_sc, bm, p_sc, opt) == 0
            end
        end

        for n in (2, 4)
            pv = fill(T(5), n)
            p_s = SVector{n, T}(pv)
            perm = r1_shift(Int, n)
            problems = (
                ("R1Holomorphic",
                    NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)),
                ("R1Coupled",
                    NonlinearProblem(R1Coupled(perm), (x, p) -> r1_coupled_jacobian(x, p, perm)))
            )
            for (name, prob) in problems
                x₀ = SVector{n, T}(fill(T(1 / 2), n))
                x_n, st_n = solve(prob, x₀, Newton(), p_s)
                x_br, st_br = @inferred solve(prob, x₀, method_br, p_s)
                abstol = max(opt.f_abstol, opt.f_reltol * rnorm(prob.F(x₀, p_s)))
                J_star = Matrix(prob.J(x_n, p_s))
                @test st_n.code == SUCCESS
                @test st_br.code == SUCCESS
                @test rnorm(prob.F(x_br, p_s)) <= abstol
                @test rnorm(x_br - x_n) <= abstol * cond(J_star)

                for m_val in (Val(1), Val(2), Val(3), Val(4))
                    bm = Broyden(Picard(; damping = 1 / 10), m_val)
                    st0 = init_state(bm, prob, x₀, p_s)
                    @test isbits(st0)
                    Fx₀ = prob.F(x₀, p_s)
                    st0_s = start!!(st0, bm, prob, x₀, Fx₀, p_s, rnorm(Fx₀))
                    @inferred step!!(st0_s, bm, prob, x₀, Fx₀, p_s)
                    if Base.JLOptions().check_bounds != 1
                        # Skipped where a run forces --check-bounds=yes.
                        @test _r1_broyden_step_alloc_probe(
                            st0_s, bm, prob, x₀, Fx₀, p_s) == 0
                        @test _r1_broyden_solve_alloc_probe(
                            st0, prob, x₀, bm, p_s, opt) == 0
                    end
                end
            end
        end
    end

    @testset "R3 Array and JLArray convergence on Holomorphic and Coupled ($T)" for T in ELTYPES
        R = real(T)
        opt = Options(R)
        method_br = Broyden(Picard(; damping = 1 / 10), Val(4))
        for n in (2, 4)
            p_vec = fill(T(5), n)
            perm = r1_shift(Int, n)
            problems = (
                ("Holomorphic",
                    NonlinearProblem(HolomorphicResidual!(), HolomorphicAnalyticalJacobian!()),
                    NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)),
                ("Coupled",
                    NonlinearProblem(CoupledResidual!(perm), CoupledAnalyticalJacobian!(perm)),
                    NonlinearProblem(R1Coupled(perm), (x, p) -> r1_coupled_jacobian(x, p, perm)))
            )
            for (name, prob_r3, prob_r1) in problems
                x₀_s = SVector{n, T}(fill(T(1 / 2), n))
                p_s = SVector{n, T}(p_vec)
                x_n_s, st_n_s = solve(prob_r1, x₀_s, Newton(), p_s)
                J_star = Matrix(prob_r1.J(x_n_s, p_s))

                x₀ = fill(T(1 / 2), n)
                Fx₀ = similar(x₀)
                prob_r3.F(Fx₀, x₀, p_vec)
                abstol = max(opt.f_abstol, opt.f_reltol * rnorm(Fx₀))

                solver = init(prob_r3, copy(x₀), method_br, p_vec)
                x_br, st_br = solve!(solver, copy(x₀))
                @test st_br.code == SUCCESS
                Fx_br = similar(x_br)
                prob_r3.F(Fx_br, x_br, p_vec)
                @test rnorm(Fx_br) <= abstol
                @test rnorm(x_br - Vector(x_n_s)) <= abstol * cond(J_star)

                # Check 4: R3 step!! and solve! allocate 0 bytes on Array after warm-up
                st_r3 = init_state(method_br, prob_r3, x₀, p_vec)
                start!!(st_r3, method_br, prob_r3, x₀, Fx₀, p_vec, rnorm(Fx₀))
                x_work = copy(x₀)
                Fx_work = copy(Fx₀)
                @test _r3_broyden_step_alloc_probe(
                    st_r3, method_br, prob_r3, x_work, x₀, Fx_work, Fx₀, p_vec) == 0
                @test _r3_broyden_solve_alloc_probe(solver, x_work, x₀) == 0

                # Edge 7: JLArray R3 solve with scalar indexing forbidden
                x₀_jl = JLArray(x₀)
                p_jl = JLArray(p_vec)
                solver_jl = @allowscalar false init(prob_r3, copy(x₀_jl), method_br, p_jl)
                x_jl, st_jl = @allowscalar false solve!(solver_jl, copy(x₀_jl))
                @test st_jl.code == SUCCESS
                @test rnorm(Array(x_jl) - Vector(x_n_s)) <= abstol * cond(J_star)
            end
        end
    end

    @testset "Check 3 & Edge 7: R1 vs R3 (Array, JLArray, KA CPU) one-step agreement ($T)" for T in ELTYPES
        R = real(T)
        n = 4
        perm = r1_shift(Int, n)
        prob_r1 = NonlinearProblem(R1Coupled(perm), (x, p) -> r1_coupled_jacobian(x, p, perm))
        prob_r3 = NonlinearProblem(CoupledResidual!(perm), CoupledAnalyticalJacobian!(perm))
        p_vec = fill(T(5), n)
        p_s = SVector{n, T}(p_vec)
        p_jl = JLArray(p_vec)
        p_ka = KA.allocate(KA_BACKEND, T, n)
        copyto!(p_ka, p_vec)

        for m in (1, 2, 3, 4)
            method = Broyden(Picard(; damping = 1 / 10), Val(m))
            x_r1 = SVector{n, T}(fill(T(1 / 2), n))
            Fx_r1 = prob_r1.F(x_r1, p_s)
            st_r1 = start!!(init_state(method, prob_r1, x_r1, p_s),
                method, prob_r1, x_r1, Fx_r1, p_s, rnorm(Fx_r1))

            x_arr = Vector(x_r1)
            Fx_arr = Vector(Fx_r1)
            st_arr = init_state(method, prob_r3, x_arr, p_vec)
            start!!(st_arr, method, prob_r3, x_arr, Fx_arr, p_vec, rnorm(Fx_arr))

            x_jl = JLArray(x_arr)
            Fx_jl = JLArray(Fx_arr)
            st_jl = @allowscalar false init_state(method, prob_r3, x_jl, p_jl)
            @allowscalar false start!!(
                st_jl, method, prob_r3, x_jl, Fx_jl, p_jl, rnorm(Fx_jl))

            x_ka = KA.allocate(KA_BACKEND, T, n)
            copyto!(x_ka, x_arr)
            Fx_ka = KA.allocate(KA_BACKEND, T, n)
            copyto!(Fx_ka, Fx_arr)
            st_ka = init_state(method, prob_r3, x_ka, p_ka)
            start!!(st_ka, method, prob_r3, x_ka, Fx_ka, p_ka, rnorm(Fx_ka))

            for step_idx in 1:(m + 1)
                x_r1, Fx_r1, st_r1, _ = step!!(st_r1, method, prob_r1, x_r1, Fx_r1, p_s)
                step!!(st_arr, method, prob_r3, x_arr, Fx_arr, p_vec)
                @allowscalar false step!!(st_jl, method, prob_r3, x_jl, Fx_jl, p_jl)
                step!!(st_ka, method, prob_r3, x_ka, Fx_ka, p_ka)

                tol = n * eps(R) * rnorm(x_r1)
                @test rnorm(x_arr - Vector(x_r1)) <= tol
                @test rnorm(Array(x_jl) - Vector(x_r1)) <= tol
                @test rnorm(Array(x_ka) - Vector(x_r1)) <= tol
            end
        end
    end

    @testset "R1 MVector iterate with SVector residual" begin
        prob_mv = NonlinearProblem((x, p) -> SVector(x[1] - 0.5 * cos(x[2]), x[2] -
                                                                             0.5 *
                                                                             sin(x[1])))
        x0_mv = MVector(0.2, 0.3)
        x_mv, st_mv = solve(prob_mv, x0_mv, Broyden(Picard(), Val(3)))
        @test isconverged(st_mv)
        @test rnorm(prob_mv.F(x_mv, nothing)) <= Options(Float64).f_abstol
    end

    @testset "R2 solve_batch! ($T)" for T in ELTYPES
        R = real(T)
        n = 2
        B = 8
        prob = NonlinearProblem(R1Holomorphic())
        method = Broyden(Picard(; damping = 1 / 10), Val(3))
        X0 = [SVector{n, T}(T(0.4 + 0.02 * b), T(0.5 - 0.01 * b)) for b in 1:B]
        p_s = SVector{n, T}(T(5), T(5))

        # JLArray batch runs in Julia without SPIR-V compilation
        X_jl = JLArray(copy(X0))
        S_jl = JLArray(Vector{SolverStatus{R}}(undef, B))
        solve_batch!(X_jl, S_jl, prob, method, p_s)
        @test all(st -> st.code == SUCCESS, Array(S_jl))
        X_jl_host = Array(X_jl)
        for b in 1:B
            x_ref, st_ref = solve(prob, X0[b], method, p_s)
            @test st_ref.code == SUCCESS
            @test isapprox(X_jl_host[b], x_ref; rtol = 16 * eps(R))
        end

        # R2 solve_batch! on CPU()
        X_cpu = copy(X0)
        S_cpu = Vector{SolverStatus{R}}(undef, B)
        @test begin
            solve_batch!(X_cpu, S_cpu, prob, method, p_s)
            all(st -> st.code == SUCCESS, S_cpu)
        end
    end

    @testset "Edge 8: R3 supports m >= 5" begin
        for T in ELTYPES
            R = real(T)
            n = 6
            prob = NonlinearProblem(
                (F, x, p) -> begin
                @inbounds for i in eachindex(x)
                    F[i] = 2 * x[i] + T(0.15) * x[i]^2 - p[i]
                end
                return nothing
            end)
            p_vec = [T(1 + 0.1 * i) for i in 1:n]
            x0_vec = zeros(T, n)
            method = Broyden(Picard(; damping = 0.4), Val(5); grow_depth = true)
            solver = init(prob, copy(x0_vec), method, p_vec)
            x_sol, st_sol = solve!(solver, copy(x0_vec))
            @test st_sol.code == SUCCESS
        end
    end

    @testset "Check 4: JET.@test_opt zero optimization issues" begin
        prob_r1 = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)
        x0_r1 = SVector{2, Float64}(0.5, 0.5)
        p_r1 = SVector{2, Float64}(5.0, 5.0)
        m_br = Broyden(Picard(; damping = 0.1), Val(4))
        st_r1 = init_state(m_br, prob_r1, x0_r1, p_r1)
        Fx0_r1 = prob_r1.F(x0_r1, p_r1)
        st_r1 = start!!(st_r1, m_br, prob_r1, x0_r1, Fx0_r1, p_r1, rnorm(Fx0_r1))
        opt = Options(Float64)
        JET.@test_opt target_modules = (GeometricSolvers,) step!!(
            st_r1, m_br, prob_r1, x0_r1, Fx0_r1, p_r1)
        JET.@test_opt target_modules = (GeometricSolvers,) solve!!(
            st_r1, prob_r1, x0_r1, m_br, p_r1, opt)

        prob_r3 = NonlinearProblem(HolomorphicResidual!(), HolomorphicAnalyticalJacobian!())
        x0_r3 = [0.5, 0.5]
        Fx0_r3 = similar(x0_r3)
        p_r3 = [5.0, 5.0]
        prob_r3.F(Fx0_r3, x0_r3, p_r3)
        st_r3 = init_state(m_br, prob_r3, x0_r3, p_r3)
        start!!(st_r3, m_br, prob_r3, x0_r3, Fx0_r3, p_r3, rnorm(Fx0_r3))
        JET.@test_opt target_modules = (GeometricSolvers,) step!!(
            st_r3, m_br, prob_r3, x0_r3, Fx0_r3, p_r3)
    end

    @testset "StationarityProblem and composition with Multiplicative, Repeat, Switch" begin
        F_stat(x, p) = SVector(2 * x[1] + 0.2 * x[2] - 1.0, 0.2 * x[1] + 3 * x[2] - 2.0)
        prob_stat = StationarityProblem(F_stat, :, :)
        x0 = SVector(0.0, 0.0)
        x_st, st_st = solve(prob_stat, x0, Broyden(Picard(; damping = 0.3), Val(3)))
        @test isconverged(st_st)
        @test rnorm(F_stat(x_st, nothing)) <= Options(Float64).f_abstol

        # Broyden composed inside Repeat, Multiplicative, and Switch
        prob_nl = NonlinearProblem(F_stat)
        for comb in (
            Repeat(Broyden(Picard(; damping = 0.3), Val(3)), 2),
            Multiplicative(Picard(; damping = 0.3), Broyden(Picard(; damping = 0.3), Val(3))),
            Switch(Broyden(Picard(; damping = 0.3), Val(3)), Newton(); threshold = 1e-1)
        )
            x_c, st_c = solve(prob_nl, x0, comb)
            @test isconverged(st_c)
            @test rnorm(F_stat(x_c, nothing)) <= Options(Float64).f_abstol
        end
    end
end

@testset "Decided edges 2, 3, 5, 6 and Mutants (restarts, safeguards, failures, finish!!, solver reuse)" begin
    # Hard failure of inner member returns immediately and preserves history
    prob_r1 = NonlinearProblem((x, p) -> SVector(x[1] - 1.0, x[2] - 2.0))
    x0 = SVector(0.0, 0.0)
    Fx0 = prob_r1.F(x0, nothing)
    for hard_code in (SINGULAR, NONFINITE, LINESEARCH_FAILED)
        member_hard = ScriptedBroydenMember(hard_code, x -> x)
        b_hard = Broyden(member_hard, Val(3))
        st0 = start!!(init_state(b_hard, prob_r1, x0, nothing),
            b_hard, prob_r1, x0, Fx0, nothing, rnorm(Fx0))
        x_out, Fx_out, st_out, info_out = step!!(st0, b_hard, prob_r1, x0, Fx0, nothing)
        @test info_out.code == hard_code
        @test iszero(info_out.stepnorm)
        @test x_out == x0
        @test st_out.depth == st0.depth
        @test st_out.current_depth == st0.current_depth
        @test st_out.has_prev == st0.has_prev
    end

    # Edge 2: Non-finite incoming residual (NaN or Inf in Fx):
    # - when member hard-fails, returns member's code and keeps x;
    # - when member produces a finite step, should_restart_growth clears history (depth == 0),
    #   takes x_m, and increments failures by 1.
    Fx_nan = SVector(NaN, 1.0)
    member_ok = ScriptedBroydenMember(SUCCESS, x -> SVector(1.0, 2.0))
    b_ok = Broyden(member_ok, Val(3))
    st_nan0 = start!!(init_state(b_ok, prob_r1, x0, nothing),
        b_ok, prob_r1, x0, Fx0, nothing, rnorm(Fx0))
    # Populate 1 history column first
    x_p1, Fx_p1, st_nan1, _ = step!!(
        st_nan0, b_ok, prob_r1, x0, Fx0, nothing)
    x_p2, Fx_p2, st_nan2, _ = step!!(
        st_nan1, b_ok, prob_r1, SVector(0.5, 1.0), prob_r1.F(SVector(0.5, 1.0), nothing), nothing)
    @test st_nan2.depth == 1
    x_nf, Fx_nf, st_nf, info_nf = step!!(
        st_nan2, b_ok, prob_r1, SVector(0.25, 0.5), Fx_nan, nothing)
    @test x_nf == SVector(1.0, 2.0)
    @test st_nf.depth == 0
    @test info_nf.failures == 1

    # Soft LINESEARCH_FAILED (stepnorm > 0) uses the moved candidate and preserves code
    member_soft = ScriptedBroydenMember(
        LINESEARCH_FAILED, x -> SVector(0.5 * (x[1] + 1.0), 0.5 * (x[2] + 2.0)))
    b_soft = Broyden(member_soft, Val(3))
    st_s = start!!(init_state(b_soft, prob_r1, x0, nothing),
        b_soft, prob_r1, x0, Fx0, nothing, rnorm(Fx0))
    x1, Fx1, st_s1, info1 = step!!(st_s, b_soft, prob_r1, x0, Fx0, nothing)
    @test info1.code == LINESEARCH_FAILED
    @test info1.stepnorm > 0
    @test st_s1.has_prev
    x2, Fx2, st_s2, info2 = step!!(st_s1, b_soft, prob_r1, x1, Fx1, nothing)
    @test info2.stepnorm > 0
    @test st_s2.depth >= 1

    # Edge 3: Empty history (first step after start!! is x_m; when filters drop all d columns,
    # p_kept == 0, takes x_m, and increments failures by d).
    A_mat = @SMatrix [2.0 0.8 0.3; 0.4 1.7 0.5; 0.2 0.3 1.4]
    b_vec = SVector(1.0, -0.5, 0.8)
    prob_lin = NonlinearProblem((x, p) -> A_mat * x - b_vec)
    z3 = SVector(0.0, 0.0, 0.0)
    # Two identical steps give Δw = 0 on the second step, so G_11 = 0 fails the angle filter (p_kept = 0, d = 1)
    member_const = ScriptedBroydenMember(SUCCESS, x -> SVector(0.2, 0.1, -0.1))
    b_drop = Broyden(member_const, Val(3))
    st_dr0 = start!!(init_state(b_drop, prob_lin, z3, nothing),
        b_drop, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    x_d1, F_d1, st_dr1, info_d1 = step!!(st_dr0, b_drop, prob_lin, z3, -b_vec, nothing)
    @test x_d1 == SVector(0.2, 0.1, -0.1)
    @test info_d1.failures == 0
    x_d2, F_d2, st_dr2, info_d2 = step!!(st_dr1, b_drop, prob_lin, z3, -b_vec, nothing)
    @test x_d2 == SVector(0.2, 0.1, -0.1)
    @test info_d2.failures == 1

    # M_alpha weight-bound rejection and safeguard_cos angle restart
    # Tight M_alpha = 1.0 forces weight rejection when any γ_c != 0 makes ∑|α_j| > 1
    b_tight = Broyden(Picard(; damping = 0.5), Val(3); M_alpha = 1.0)
    st_t = start!!(init_state(b_tight, prob_lin, z3, nothing),
        b_tight, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    xt1, Ft1, st_t1, _ = step!!(st_t, b_tight, prob_lin, z3, -b_vec, nothing)
    xt2, Ft2, st_t2, info_t2 = step!!(st_t1, b_tight, prob_lin, xt1, Ft1, nothing)
    @test info_t2.failures >= 1

    # Tight safeguard_cos = 1e-6 triggers angle restart on aligned consecutive corrections
    b_cos = Broyden(Picard(; damping = 0.5), Val(3); safeguard_cos = 1e-6)
    st_c = start!!(init_state(b_cos, prob_lin, z3, nothing),
        b_cos, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    xc1, Fc1, st_c1, _ = step!!(st_c, b_cos, prob_lin, z3, -b_vec, nothing)
    xc2, Fc2, st_c2, info_c2 = step!!(st_c1, b_cos, prob_lin, xc1, Fc1, nothing)
    @test info_c2.failures >= 1
    @test st_c2.depth == 0

    # Diverging map triggers residual-growth restart and restores x_best in finish!!
    prob_div = NonlinearProblem((x, p) -> 6.0 * x)
    x0_div = SVector(1.0, -0.5)
    x_div, st_div = solve(prob_div, x0_div,
        Broyden(Picard(; damping = 1.0), Val(3); restart_factor = 4.0), nothing;
        options = Options(Float64; max_iterations = 8, f_abstol = 1e-12, f_reltol = 0.0))
    @test st_div.code != SUCCESS
    @test st_div.step_failures >= 1
    @test x_div == x0_div
    @test st_div.fnorm == rnorm(prob_div.F(x0_div, nothing))

    prob_div_r3 = NonlinearProblem((F, x, p) -> (F .= 6.0 .* x; nothing))
    x0_div_r3 = [1.0, -0.5]
    x_div_r3, st_div_r3 = solve(prob_div_r3, copy(x0_div_r3),
        Broyden(Picard(; damping = 1.0), Val(3); restart_factor = 4.0), nothing;
        options = Options(Float64; max_iterations = 8, f_abstol = 1e-12, f_reltol = 0.0))
    @test st_div_r3.code != SUCCESS
    @test st_div_r3.step_failures >= 1
    @test x_div_r3 == x0_div_r3
    @test st_div_r3.fnorm == rnorm(6.0 .* x0_div_r3)

    # Refusal of ForwardDiff.Dual in solve and LeastSquaresProblem in init_state
    xd = SVector(ForwardDiff.Dual(1.0, 1.0), ForwardDiff.Dual(2.0, 0.0))
    @test_throws ArgumentError solve(prob_r1, xd, Broyden())
    prob_ls = LeastSquaresProblem((x, p) -> SVector(x[1] - 1.0, x[2] - 2.0, x[1] + x[2]))
    @test_throws ArgumentError init(prob_ls, SVector(1.0, 2.0), Broyden())

    # History reset in start!! across consecutive solve! calls on a reused solver
    prob_reuse = NonlinearProblem(
        (F, x, p) -> begin
        F[1] = 2 * x[1] + 0.3 * sin(x[2]) - 1.0
        F[2] = 2 * x[2] + 0.3 * cos(x[1]) - 1.5
        F[3] = 2 * x[3] + 0.2 * x[1] * x[2] - 0.5
        return nothing
    end)
    x0_reuse = [0.1, -0.2, 0.3]
    solver_reuse = init(prob_reuse, copy(x0_reuse), Broyden(Picard(; damping = 0.4), Val(4)))
    x_run1, st_run1 = solve!(solver_reuse, copy(x0_reuse))
    x_run2, st_run2 = solve!(solver_reuse, copy(x0_reuse))
    @test isconverged(st_run1)
    @test st_run1.iterations == st_run2.iterations
    @test st_run1.fnorm == st_run2.fnorm
    @test st_run1.step_failures == st_run2.step_failures
    @test x_run1 == x_run2
end
