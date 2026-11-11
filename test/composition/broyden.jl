using Adapt: Adapt
using ForwardDiff: ForwardDiff
using GeometricSolvers
using GeometricSolvers: AndersonHistory, NONFINITE, LINESEARCH_FAILED, SINGULAR, SUCCESS,
                        StepInfo, ToReal, finish!!, init_state, is_hard_failure,
                        isconverged,
                        norm2, push_history!!, rdot, rnorm, solve!!, solve_broyden_weights,
                        solve_broyden_weights!, solve_weights, start!!, step!!,
                        step_multisecant!!
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions as KA
using LinearAlgebra: BLAS, I, cond, mul!, norm
using Random: Xoshiro, randn
using StaticArrays: @SMatrix, MVector, SMatrix, StaticVector, SVector, setindex
using Test

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")
include("../helpers/adproblems.jl")
allowscalar(false)

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

# The only `step!!` of this member, on the R1 signature of `Broyden`: every state the file builds
# for it holds an `SVector` or a `Number` iterate. A second method on `AbstractVector` would make
# `SVector`, which is both a `StaticVector` and an `AbstractVector`, ambiguous.
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

# The R3 counterpart of the member above: its step takes the iterate and residual buffers of the
# R3 signature, writes its candidate into the iterate, and evaluates the in-place residual.
struct ScriptedR3BroydenMember{C, F}
    code::C
    step_fn::F
end

GeometricSolvers.init_state(::ScriptedR3BroydenMember, prob, x, p) = nothing
GeometricSolvers.start!!(st, ::ScriptedR3BroydenMember, prob, x, Fx, p, fnorm₀) = st
GeometricSolvers.sync_merit(st, ::ScriptedR3BroydenMember, merit::Real) = st
function GeometricSolvers.finish!!(st, ::ScriptedR3BroydenMember, prob, x, Fx, status)
    return x, status, st
end

function GeometricSolvers.step!!(
        st, m::ScriptedR3BroydenMember, prob, x::AbstractVector, Fx, p)
    R = real(eltype(x))
    x_cand = m.step_fn(Vector(x))
    stepnorm = rnorm(x_cand - Vector(x))
    copyto!(x, x_cand)
    prob.F(Fx, x, p)
    failures = m.code == SUCCESS ? Int32(0) : Int32(1)
    return x, Fx, st, StepInfo{R}(m.code, rnorm(Fx), stepnorm, failures, false, false)
end

# The map of the post-step restart testset: `A x + c x³ - b`, a damped `Picard` member and
# `restart_factor = 3`. From this start the third step's accelerated iterate grows the residual
# past `3 fnorm_best` while the third step's incoming residual does not, so only the post-step
# safeguard can fire there.
const POST_RESTART_A = [2.658268422170796 -1.0553858794800004 1.4510152505141574;
                        0.8940164711789151 1.4968130681990974 0.7636812440860139;
                        -0.46367073975036655 -0.480087178200877 2.410821387913771]
const POST_RESTART_C = 1.6842200338181805
const POST_RESTART_B = [-0.5532856214164881, -0.32122990014172237, 1.424486481479478]
const POST_RESTART_DAMPING = 0.9602548089183558
const POST_RESTART_X = [-0.866579836099137, -0.3325072429872186, 0.6114335821524323]

# The residual of Check 3: `Coupled`'s, `x_i² + p_i x_i + 2 x_{perm_i}`, minus a constant, so that
# the problem has no root at zero. The bound `n * eps(R) * rnorm(x₊)` is relative, and a
# root at zero takes it below the round-off floor of the weights, where no pair of summation
# orders meets it. One type with a value form and an in-place form, as the two storage forms of a
# residual take it.
struct CoupledShift{P, C}
    perm::P
    shift::C
end

(c::CoupledShift)(x, p) = x .* x .+ p .* x .+ 2 .* x[c.perm] .- c.shift
(c::CoupledShift)(r, x, p) = (r .= x .* x .+ p .* x .+ 2 .* x[c.perm] .- c.shift; r)

# The in-place Jacobians of the two R3 `Check 1` problems, so that `Newton()` runs on the same
# problem the R3 `Broyden` solve does: `Holomorphic` is `Diagonal(2x + p)`, and `Coupled` adds `2`
# at `(i, perm[i])`.
function r3_holomorphic_jacobian!(J, x, p)
    fill!(J, zero(eltype(J)))
    @inbounds for i in eachindex(x)
        J[i, i] = x[i] + x[i] + p[i]
    end
    return J
end

function r3_coupled_jacobian!(J, x, p, perm)
    fill!(J, zero(eltype(J)))
    @inbounds for i in eachindex(x)
        J[i, i] = x[i] + x[i] + p[i]
        J[i, perm[i]] += eltype(J)(2)
    end
    return J
end

# The size `S` of `Check 3`, from the R1 state the step returned: `x_m = x_prev + w_k`, and each
# kept column's `(abs(γ_c) + rnorm(w_k) / rnorm(Δw_c)) * rnorm(Δx_c + Δw_c)`. `x_m` and
# `γ_c (Δx_c + Δw_c)` are summed in `x₊`, and `b_c = Re⟨Δw_c, w_k⟩` in `γ_c`, so a different
# summation order is bounded by `n * eps(R) * S`, not by the size of the sum itself.
function _multisecant_step_scale(s1, ::Val{m}, ::Type{R}) where {m, R}
    p = Int(s1.depth)
    w_k = s1.w_prev
    x_m = s1.x_prev + w_k
    S = rnorm(x_m)
    p <= 0 && return S
    G = zero(SMatrix{m, m, R})
    b = zero(SVector{m, R})
    ΔXc = ntuple(c -> c <= p ? s1.ΔX[c] : zero(x_m), Val(m))
    ΔWc = ntuple(c -> c <= p ? s1.ΔW[c] : zero(x_m), Val(m))
    @inbounds for j in 1:p
        b = setindex(b, R(rdot(ΔWc[j], w_k)), j)
        for i in 1:j
            g = R(rdot(ΔWc[i], ΔWc[j]))
            G = setindex(G, g, i + (j - 1) * m)
            i == j || (G = setindex(G, g, j + (i - 1) * m))
        end
    end
    γ = solve_broyden_weights(G, b, SVector{m, Int}(ntuple(identity, m)), p)
    @inbounds for c in 1:p
        S += (abs(γ[c]) + rnorm(w_k) / rnorm(ΔWc[c])) * rnorm(ΔXc[c] + ΔWc[c])
    end
    return S
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

# The R3 weight solve on one call, through a barrier: the arguments are concrete and the call runs
# once before it is measured.
function _broyden_weights_alloc_probe(γ, G, b, kept)
    solve_broyden_weights!(γ, G, b, kept, 3)
    return @allocated solve_broyden_weights!(γ, G, b, kept, 3)
end

@testset "Broyden constructors, validation, and Adapt" begin
    b_def = Broyden()
    @test b_def isa Broyden{4, <:Picard}
    @test b_def.member.damping == 0.5
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
    # Without a member the default `Picard` is damped by 0.5; an explicit `Picard()` is not.
    @test b4_val.member.damping == b4_int_def.member.damping == 0.5
    @test b4.member.damping == b4_int.member.damping == 1

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

@testset "Edge 6: the other precision" begin
    # `Adapt.adapt(ToReal{Float32}(), …)` narrows every field and keeps `m`, and a `Float32` solve
    # runs with the `Float32` defaults of `Options`.
    b32 = Adapt.adapt(ToReal{Float32}(),
        Broyden(Picard(; damping = 0.25), Val(3);
            c_s = 0.25, κ_bar = 1e5, restart_factor = 8.0, M_alpha = 15.0,
            safeguard_cos = 0.85))
    @test b32 isa Broyden{3}
    @test b32.member.damping isa Float32
    @test b32.c_s isa Float32
    @test b32.κ_bar isa Float32
    @test b32.restart_factor isa Float32
    @test b32.M_alpha isa Float32
    @test b32.safeguard_cos isa Float32

    opt32 = Options(Float32)
    prob_r1 = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)
    x0 = SVector{2, Float32}(1 / 2, 1 / 2)
    p = SVector{2, Float32}(5, 5)
    x32, st32 = solve(prob_r1, x0, Broyden(Picard(; damping = 1 // 10), Val(3)), p)
    @test st32.code == SUCCESS
    @test eltype(x32) === Float32
    abstol_r1 = max(opt32.f_abstol, opt32.f_reltol * rnorm(prob_r1.F(x0, p)))
    @test rnorm(prob_r1.F(x32, p)) <= abstol_r1

    prob_r3 = NonlinearProblem(Holomorphic())
    x3 = Float32[1 / 2, 1 / 2]
    p3 = Float32[5, 5]
    F3 = similar(x3)
    prob_r3.F(F3, x3, p3)
    abstol_r3 = max(opt32.f_abstol, opt32.f_reltol * rnorm(F3))
    solver = init(prob_r3, copy(x3), Broyden(Picard(; damping = 1 // 10), Val(3)), p3)
    x_sol, st_sol = solve!(solver, copy(x3))
    @test st_sol.code == SUCCESS
    @test eltype(x_sol) === Float32
    F_sol = similar(x3)
    prob_r3.F(F_sol, x_sol, p3)
    @test rnorm(F_sol) <= abstol_r3
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

        # Check 2: a `kept` that is not a prefix. The kept columns are a subset of the candidate
        # Gram matrix in age order, so `G[K[c], K[r]]` and `b[K[c]]` differ from `G[c, r]` and
        # `b[c]`; the solver reads the indices of `kept`, never the logical position `c`.
        kept_np = SVector{3, Int}(1, 3, 0)
        λ_np = eps(R) * ((G[1, 1] + G[3, 3]) / R(2))
        γ_np1 = b[1] / (G[1, 1] + λ_np)
        γ_np2 = (b[3] - G[3, 1] * γ_np1) / (G[3, 3] + λ_np)
        γ_np = solve_broyden_weights(G, b, kept_np, 2)
        @test abs(γ_np[1] - γ_np1) <= 4 * 2 * eps(R) * maximum(abs, γ_np)
        @test abs(γ_np[2] - γ_np2) <= 4 * 2 * eps(R) * maximum(abs, γ_np)
        @test iszero(γ_np[3])

        # A kept column out of order: with `kept = (1, 3, 2)` the off-diagonal read at `r = 2` is
        # `G[2, 3]`, so an index taken from the logical position `r` reads `G[2, 2]` instead.
        kept_np3 = SVector{3, Int}(1, 3, 2)
        λ_np3 = eps(R) * ((G[1, 1] + G[3, 3] + G[2, 2]) / R(3))
        γ_np31 = b[1] / (G[1, 1] + λ_np3)
        γ_np32 = (b[3] - G[3, 1] * γ_np31) / (G[3, 3] + λ_np3)
        γ_np33 = (b[2] - G[2, 1] * γ_np31 - G[2, 3] * γ_np32) / (G[2, 2] + λ_np3)
        γ_np3_ref = SVector{3, R}(γ_np31, γ_np32, γ_np33)
        γ_np3 = solve_broyden_weights(G, b, kept_np3, 3)
        @test maximum(abs, γ_np3 - γ_np3_ref) <= 4 * 3 * eps(R) * maximum(abs, γ_np3)

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
        fill!(γ_vec, R(NaN))
        solve_broyden_weights!(γ_vec, G_mat, b_vec, [1, 3, 0], 2)
        @test abs(γ_vec[1] - γ_np1) <= 4 * 2 * eps(R) * maximum(abs, γ_vec)
        @test abs(γ_vec[2] - γ_np2) <= 4 * 2 * eps(R) * maximum(abs, γ_vec)
        @test iszero(γ_vec[3])
        fill!(γ_vec, R(NaN))
        solve_broyden_weights!(γ_vec, G_mat, b_vec, [1, 3, 2], 3)
        @test maximum(abs, γ_vec - Vector(γ_np3_ref)) <=
              4 * 3 * eps(R) * maximum(abs, γ_vec)
        if Base.JLOptions().check_bounds != 1
            # Skipped where a run forces --check-bounds=yes.
            @test _broyden_weights_alloc_probe(γ_vec, G_mat, b_vec, kept_vec) == 0
        end

        # Edge 5: zero Gram diagonal with zero right-hand side gives the finite weight 0 when
        # `tr(G_K) > 0`, because `λ = eps(R) tr(G_K) / p > 0`.
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
    end
end

# The weights against their defining property rather than the formula of the solver: the step
# `w - (ΔX + ΔW) γ` equals `-H w`, where `H` is built from `H₀ = -I` by the sequential rank-one
# Type-II Broyden updates `H += (Δx - H Δw) Δwᵀ / ‖Δw‖²`, oldest column first. Columns are in
# increasing age, so the newest is column 1.
@testset "solve_broyden_weights matches the sequential Type-II Broyden updates" begin
    rng = Xoshiro(74)
    n, m = 6, 4
    for _ in 1:20
        ΔX = randn(rng, n, m)
        ΔW = randn(rng, n, m)
        w = randn(rng, n)
        H = -Matrix{Float64}(I, n, n)
        for c in m:-1:1
            Δx, Δw = ΔX[:, c], ΔW[:, c]
            H += (Δx - H * Δw) * Δw' / (Δw' * Δw)
        end
        step_ref = -H * w
        G = SMatrix{m, m}(ΔW' * ΔW)
        b = SVector{m}(ΔW' * w)
        γ = solve_broyden_weights(G, b, SVector{m, Int}(1:m), m)
        @test isapprox(w - (ΔX + ΔW) * γ, step_ref; rtol = sqrt(eps(Float64)))
        γ_r3 = zeros(m)
        solve_broyden_weights!(γ_r3, Matrix(G), Vector(b), collect(1:m), m)
        @test isapprox(w - (ΔX + ΔW) * γ_r3, step_ref; rtol = sqrt(eps(Float64)))
    end
end

# One linear problem and one sequence of iterates per element type. A complex type carries no real
# entry, so a Gram matrix formed with `transpose` instead of `adjoint` cannot pass by accident.
function broyden_sequence(::Type{T}) where {T}
    c = T <: Complex ? T(0, 1) : zero(T)
    A = @SMatrix [T(2) T(0.8)+c/5 T(0.3);
                  T(0.4) T(1.7) T(0.5)-c/7;
                  T(0.2) T(0.3) T(1.4)]
    b = SVector{3, T}(T(1), T(-0.5) + c / 3, T(0.8))
    xs = (zero(SVector{3, T}),
        SVector{3, T}(T(0.5) + c / 9, T(-0.25), T(0.4)),
        SVector{3, T}(T(0.35), T(-0.4) - c / 4, T(0.55)))
    return A, b, xs
end

@testset "Check 2: shared AndersonState and AndersonHistory ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    x0 = xs[1]
    m_a = Adapt.adapt(ToReal{R}(), Anderson(Picard(; damping = 0.5), Val(3)))
    m_b = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3)))

    st_a = init_state(m_a, prob_r1, x0, nothing)
    st_b = init_state(m_b, prob_r1, x0, nothing)
    @test isbits(st_b)
    @test typeof(st_b) === typeof(st_a)
    @test st_b.history isa AndersonHistory{3}
    @test isbits(st_b.history)
    @test typeof(st_b.history) === typeof(st_a.history)
    @test typeof(st_b.ΔX) === typeof(st_a.ΔX)
    @test typeof(st_b.ΔW) === typeof(st_a.ΔW)
    @test st_b.history.ΔX === st_b.ΔX
    @test st_b.history.ΔW === st_b.ΔW
    @test :history in propertynames(st_b)
    @test !(:Δx in propertynames(st_b)) && !(:Δr in propertynames(st_b))
    @test !(:AndersonHistory in names(GeometricSolvers))

    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    x0_r3 = Vector(x0)
    Fx0_r3 = similar(x0_r3)
    prob_r3.F(Fx0_r3, x0_r3, nothing)
    st_a_r3 = init_state(m_a, prob_r3, x0_r3, nothing)
    st_b_r3 = init_state(m_b, prob_r3, x0_r3, nothing)
    @test typeof(st_b_r3) === typeof(st_a_r3)
    @test typeof(st_b_r3.history) === typeof(st_a_r3.history)
    @test typeof(st_b_r3.ΔX) === typeof(st_a_r3.ΔX)
    @test typeof(st_b_r3.ΔW) === typeof(st_a_r3.ΔW)
    @test st_b_r3.history.ΔX === st_b_r3.ΔX
    @test st_b_r3.history.ΔW === st_b_r3.ΔW
end

@testset "Check 2: equal history on one iterate sequence ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    m_a = Adapt.adapt(ToReal{R}(), Anderson(Picard(; damping = 0.5), Val(3)))
    m_b = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3)))

    z = xs[1]
    Fz = prob_r1.F(z, nothing)
    st_a1 = start!!(
        init_state(m_a, prob_r1, z, nothing), m_a, prob_r1, z, Fz, nothing, rnorm(Fz))
    st_b1 = start!!(
        init_state(m_b, prob_r1, z, nothing), m_b, prob_r1, z, Fz, nothing, rnorm(Fz))
    z_r3 = Vector(z)
    Fz_r3 = Vector(Fz)
    st_a3 = start!!(init_state(m_a, prob_r3, z_r3, nothing),
        m_a, prob_r3, z_r3, Fz_r3, nothing, rnorm(Fz_r3))
    st_b3 = start!!(init_state(m_b, prob_r3, z_r3, nothing),
        m_b, prob_r3, z_r3, Fz_r3, nothing, rnorm(Fz_r3))
    for k in 1:3
        xk = xs[k]
        fk = prob_r1.F(xk, nothing)
        xa, _, st_a1, _ = step!!(st_a1, m_a, prob_r1, xk, fk, nothing)
        xb, _, st_b1, _ = step!!(st_b1, m_b, prob_r1, xk, fk, nothing)
        @test st_b1.ΔX === st_a1.ΔX
        @test st_b1.ΔW === st_a1.ΔW
        @test st_b1.history === st_a1.history
        # An R3 step writes its iterate and residual arguments, so each method takes its own copy.
        xk_r3 = Vector(xk)
        fk_r3 = Vector(fk)
        xa3, _, st_a3, _ = step!!(st_a3, m_a, prob_r3, copy(xk_r3), copy(fk_r3), nothing)
        xb3, _, st_b3, _ = step!!(st_b3, m_b, prob_r3, copy(xk_r3), copy(fk_r3), nothing)
        @test isequal(st_b3.ΔX, st_a3.ΔX)
        @test isequal(st_b3.ΔW, st_a3.ΔW)
        # With one kept column the Cholesky solve rounds twice where the recurrence divides once,
        # so the iterates agree within `4 eps(R) rnorm(x₊)`; with `p >= 2` they differ. The
        # weights themselves are compared in the weight-solver testset.
        if k == 2
            @test rnorm(xb - xa) <= 4 * eps(R) * rnorm(xb)
            @test rnorm(xb3 - xa3) <= 4 * eps(R) * rnorm(xb3)
        elseif k == 3
            @test xb != xa
            @test xb3 != xa3
        end
    end
    @test st_b1.depth >= 2
    @test st_b3.depth >= 2
end

@testset "Check 2: push_history!! and step_multisecant!! ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    meth = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3)))

    # R1: the shared history update prepends `(x - x_prev, w - w_prev)` at column 1, of age 1, and
    # returns the column count capped at `m`.
    st = init_state(meth, prob_r1, xs[1], nothing)
    @test st.depth == 0
    x_new = SVector{3, T}(T(0.4), T(-0.2), T(0.5))
    w_new = SVector{3, T}(T(0.1), T(0.2), T(-0.3))
    ΔX_cand, ΔW_cand, ages_cand, d = push_history!!(st, x_new, w_new)
    @test d == 1
    @test ΔX_cand[1] == x_new - st.x_prev
    @test ΔW_cand[1] == w_new - st.w_prev
    @test ages_cand[1] == Int32(1)
    @test all(iszero, ΔX_cand[2:end]) && all(iszero, ΔW_cand[2:end])

    # R3: the four-argument form writes the newest column of `ΔX, ΔW` from `x` and `w`.
    x0_r3 = Vector(xs[1])
    Fx0_r3 = similar(x0_r3)
    prob_r3.F(Fx0_r3, x0_r3, nothing)
    st3 = init_state(meth, prob_r3, x0_r3, nothing)
    x_new3 = Vector(x_new)
    w_new3 = Vector(w_new)
    d3 = push_history!!(st3, x_new3, w_new3)
    @test d3 == 1
    @test st3.ages[1] == Int32(1)
    # `w_prev` fails closed at `NaN`, so the columns compare with `isequal`.
    @test isequal(st3.ΔX[:, st3.work.col_idx[1]], x_new3 - st3.x_prev)
    @test isequal(st3.ΔW[:, st3.work.col_idx[1]], w_new3 - st3.w_prev)

    # `step!!` of `Broyden` is `step_multisecant!!`: one state and one input give one step.
    Fz = prob_r1.F(xs[1], nothing)
    st_x = start!!(init_state(meth, prob_r1, xs[1], nothing),
        meth, prob_r1, xs[1], Fz, nothing, rnorm(Fz))
    st_y = start!!(init_state(meth, prob_r1, xs[1], nothing),
        meth, prob_r1, xs[1], Fz, nothing, rnorm(Fz))
    for k in 2:3
        xk = xs[k]
        fk = prob_r1.F(xk, nothing)
        x_s, _, st_x, info_s = step!!(st_x, meth, prob_r1, xk, fk, nothing)
        x_m, _, st_y, info_m = step_multisecant!!(st_y, meth, prob_r1, xk, fk, nothing)
        @test x_s == x_m
        @test info_s.code == info_m.code
        @test info_s.failures == info_m.failures
        @test st_x.depth == st_y.depth
        @test st_x.ΔX === st_y.ΔX
    end
    # The count is capped at `m`, and at `m_cap` where one is given.
    @test push_history!!(st_x, xs[3], st_x.w_prev)[4] == min(Int(st_x.depth) + 1, 3)
    @test push_history!!(st_x, xs[3], st_x.w_prev, 1)[4] == 1
end

@testset "Check 2: the history is bit-identical up to the first restart split ($T)" for T in ELTYPES
    R = real(T)
    compared = 0
    split_count = 0
    for seed in 1:10
        rng = Xoshiro(seed)
        A = Matrix{T}(I, 3, 3) + R(0.3) * randn(rng, T, 3, 3)
        b = randn(rng, T, 3)
        A_s = SMatrix{3, 3, T}(A)
        b_s = SVector{3, T}(b)
        prob_r1 = NonlinearProblem((x, p) -> A_s * x - b_s)
        prob_r3 = NonlinearProblem((F, x, p) -> (mul!(F, A, x); F .-= b; nothing))
        seq = [randn(rng, T, 3) for _ in 1:6]
        for m in 1:4
            m_a = Adapt.adapt(ToReal{R}(), Anderson(Picard(; damping = 0.5), Val(m)))
            m_b = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(m)))
            z = SVector{3, T}(seq[1])
            Fz = prob_r1.F(z, nothing)
            st_a1 = start!!(init_state(m_a, prob_r1, z, nothing),
                m_a, prob_r1, z, Fz, nothing, rnorm(Fz))
            st_b1 = start!!(init_state(m_b, prob_r1, z, nothing),
                m_b, prob_r1, z, Fz, nothing, rnorm(Fz))
            z_r3 = Vector{T}(seq[1])
            Fz_r3 = similar(z_r3)
            prob_r3.F(Fz_r3, z_r3, nothing)
            st_a3 = start!!(init_state(m_a, prob_r3, z_r3, nothing),
                m_a, prob_r3, z_r3, Fz_r3, nothing, rnorm(Fz_r3))
            st_b3 = start!!(init_state(m_b, prob_r3, z_r3, nothing),
                m_b, prob_r3, z_r3, Fz_r3, nothing, rnorm(Fz_r3))
            split1 = false
            split3 = false
            for k in 1:6
                xk = SVector{3, T}(seq[k])
                fk = prob_r1.F(xk, nothing)
                _, _, st_a1, _ = step!!(st_a1, m_a, prob_r1, xk, fk, nothing)
                _, _, st_b1, _ = step!!(st_b1, m_b, prob_r1, xk, fk, nothing)
                xk_r3 = Vector{T}(seq[k])
                fk_r3 = similar(xk_r3)
                prob_r3.F(fk_r3, xk_r3, nothing)
                _, _, st_a3, _ = step!!(
                    st_a3, m_a, prob_r3, copy(xk_r3), copy(fk_r3), nothing)
                _, _, st_b3, _ = step!!(
                    st_b3, m_b, prob_r3, copy(xk_r3), copy(fk_r3), nothing)

                if !split1
                    if (st_a1.depth == 0) != (st_b1.depth == 0)
                        split1 = true
                        split_count += 1
                        restarted = st_a1.depth == 0 ? st_a1 : st_b1
                        kept_on = st_a1.depth == 0 ? st_b1 : st_a1
                        @test restarted.depth == 0
                        @test all(iszero, restarted.ΔX)
                        @test all(iszero, restarted.ΔW)
                        @test kept_on.depth >= 1
                    else
                        @test st_b1.ΔX === st_a1.ΔX
                        @test st_b1.ΔW === st_a1.ΔW
                        Int(st_a1.depth) >= 1 && (compared += 1)
                    end
                end
                if !split3
                    if (st_a3.depth == 0) != (st_b3.depth == 0)
                        split3 = true
                        split_count += 1
                        restarted = st_a3.depth == 0 ? st_a3 : st_b3
                        kept_on = st_a3.depth == 0 ? st_b3 : st_a3
                        @test restarted.depth == 0
                        @test all(iszero, restarted.ΔX)
                        @test all(iszero, restarted.ΔW)
                        @test kept_on.depth >= 1
                    else
                        @test isequal(st_b3.ΔX, st_a3.ΔX)
                        @test isequal(st_b3.ΔW, st_a3.ΔW)
                        Int(st_a3.depth) >= 1 && (compared += 1)
                    end
                end
            end
        end
    end
    # The sweep must reach both shapes: a compared step at depth >= 1, and a split.
    @test compared >= 1
    @test split_count >= 1
end

# The iterates of a `Broyden` trajectory on the map of the post-step restart testset, fed to
# `Anderson` and `Broyden` alike. At the third step `Anderson` keeps its columns and `Broyden`
# takes the post-step residual-growth restart, so Check 2's split is reached by the post-step
# safeguard and not by the pre-filter restart that the sweep's splits take.
@testset "Check 2: a Broyden-only post-step split ($T)" for T in ELTYPES
    R = real(T)
    A = T.(POST_RESTART_A)
    c = T(POST_RESTART_C)
    b = T.(POST_RESTART_B)
    prob_r1 = NonlinearProblem(
        (x, p) -> SMatrix{3, 3, T}(A) * x .+ c .* x .^ 3 .- SVector{3, T}(b))
    prob_r3 = NonlinearProblem(
        (F, x, p) -> (mul!(F, A, x); @. F = F + c * x ^ 3 - b; nothing))
    m_br = Adapt.adapt(ToReal{R}(),
        Broyden(Picard(; damping = POST_RESTART_DAMPING), Val(3); restart_factor = 3.0))
    m_an = Adapt.adapt(ToReal{R}(),
        Anderson(Picard(; damping = POST_RESTART_DAMPING), Val(3); restart_factor = 3.0))

    x = SVector{3, T}(T.(POST_RESTART_X))
    Fx = prob_r1.F(x, nothing)
    st_b1 = start!!(init_state(m_br, prob_r1, x, nothing),
        m_br, prob_r1, x, Fx, nothing, rnorm(Fx))
    st_a1 = start!!(init_state(m_an, prob_r1, x, nothing),
        m_an, prob_r1, x, Fx, nothing, rnorm(Fx))
    x3 = Vector{T}(T.(POST_RESTART_X))
    F3 = similar(x3)
    prob_r3.F(F3, x3, nothing)
    st_b3 = start!!(init_state(m_br, prob_r3, x3, nothing),
        m_br, prob_r3, x3, F3, nothing, rnorm(F3))
    st_a3 = start!!(init_state(m_an, prob_r3, x3, nothing),
        m_an, prob_r3, x3, F3, nothing, rnorm(F3))
    for _ in 1:3
        _, _, st_a1, _ = step!!(st_a1, m_an, prob_r1, x, Fx, nothing)
        xr = Vector(x)
        Fr = Vector(Fx)
        _, _, st_a3, _ = step!!(st_a3, m_an, prob_r3, copy(xr), copy(Fr), nothing)
        _, _, st_b3, _ = step!!(st_b3, m_br, prob_r3, copy(xr), copy(Fr), nothing)
        x, Fx, st_b1, _ = step!!(st_b1, m_br, prob_r1, x, Fx, nothing)
    end
    @test st_a1.depth >= 1 && st_b1.depth == 0
    @test all(iszero, st_b1.ΔX) && all(iszero, st_b1.ΔW)
    @test st_a3.depth >= 1 && st_b3.depth == 0
    @test all(iszero, st_b3.ΔX) && all(iszero, st_b3.ΔW)
end

@testset "Edge 4: Broyden{1} and Anderson{1} agree per step ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    m_a1 = Adapt.adapt(ToReal{R}(), Anderson(Picard(; damping = 0.5), Val(1)))
    m_b1 = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(1)))

    # The per-step bound `8 eps(R) S₁` with `S₁ = rnorm(x_m) + |γ₁| rnorm(Δx₁ + Δw₁)` is the size
    # of the terms the two rounding paths sum in a different order: the member step `x_m` and the
    # one correction `γ₁ (Δx₁ + Δw₁)`. `|γ₁| rnorm(Δx₁ + Δw₁)` is `rnorm(x₊ - x_m)` and
    # `x_m = x_prev + w_k`, so both terms come from the state. The test asserts on the steps
    # where both methods take the accelerated step with its one kept column (`depth == 1`): a
    # restart or a rejected weight gives the member step for both, which says nothing about the
    # weights. The bound is per step, because the two paths compound over a solve.
    fixed_r1 = 0
    fixed_r3 = 0
    z = xs[1]
    Fz = prob_r1.F(z, nothing)
    st_a1 = start!!(init_state(m_a1, prob_r1, z, nothing),
        m_a1, prob_r1, z, Fz, nothing, rnorm(Fz))
    st_b1 = start!!(init_state(m_b1, prob_r1, z, nothing),
        m_b1, prob_r1, z, Fz, nothing, rnorm(Fz))
    z_r3 = Vector(z)
    Fz_r3 = Vector(Fz)
    st_a3 = start!!(init_state(m_a1, prob_r3, z_r3, nothing),
        m_a1, prob_r3, z_r3, Fz_r3, nothing, rnorm(Fz_r3))
    st_b3 = start!!(init_state(m_b1, prob_r3, z_r3, nothing),
        m_b1, prob_r3, z_r3, Fz_r3, nothing, rnorm(Fz_r3))
    for k in 1:3
        xk = xs[k]
        fk = prob_r1.F(xk, nothing)
        xa, _, st_a1, _ = step!!(st_a1, m_a1, prob_r1, xk, fk, nothing)
        xb, _, st_b1, _ = step!!(st_b1, m_b1, prob_r1, xk, fk, nothing)
        xk_r3 = Vector(xk)
        fk_r3 = Vector(fk)
        xa3, _, st_a3, _ = step!!(st_a3, m_a1, prob_r3, copy(xk_r3), copy(fk_r3), nothing)
        xb3, _, st_b3, _ = step!!(st_b3, m_b1, prob_r3, copy(xk_r3), copy(fk_r3), nothing)
        @test st_a1.depth == st_b1.depth
        @test st_a3.depth == st_b3.depth
        if st_a1.depth == 1 && st_b1.depth == 1
            x_m = st_b1.x_prev + st_b1.w_prev
            @test rnorm(xb - xa) <= 8 * eps(R) * (rnorm(x_m) + rnorm(xb - x_m))
            fixed_r1 += 1
        end
        if st_a3.depth == 1 && st_b3.depth == 1
            x_m3 = st_b3.x_prev + st_b3.w_prev
            @test rnorm(xb3 - xa3) <= 8 * eps(R) * (rnorm(x_m3) + rnorm(xb3 - x_m3))
            fixed_r3 += 1
        end
    end
    @test fixed_r1 >= 1
    @test fixed_r3 >= 1
    @test st_b1.depth == 1

    # Every step of a default `Broyden{1}` solve of Check 1's vector problems: one R1 state stepped
    # with both methods, and the same iterates fed to two R3 `Array` states up to their first split.
    opt = Options(R)
    datasets = T <: Complex ?
               ((T(1 / 2), T(5)), (T(1 / 2, 1 / 3), T(5, 1))) : ((T(1 / 2), T(5)),)
    compared1 = 0
    compared3 = 0
    for n in (2, 4)
        perm = r1_shift(Int, n)
        problems = (
            ("Holomorphic",
                NonlinearProblem(R1Holomorphic()), NonlinearProblem(Holomorphic())),
            ("Coupled",
                NonlinearProblem(R1Coupled(perm)),
                NonlinearProblem(Coupled(cyclic_perm(Array, n)))))
        for (x0_c, p_c) in datasets, (name, prob1, prob3) in problems

            pv = fill(p_c, n)
            p_s = SVector{n, T}(pv)
            m_a = Adapt.adapt(ToReal{R}(), Anderson(Picard(; damping = 1 / 10), Val(1)))
            m_b = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), Val(1)))
            x = SVector{n, T}(fill(x0_c, n))
            Fx = prob1.F(x, p_s)
            abstol = max(opt.f_abstol, opt.f_reltol * rnorm(Fx))
            st_a = start!!(
                init_state(m_a, prob1, x, p_s), m_a, prob1, x, Fx, p_s, rnorm(Fx))
            st_b = start!!(
                init_state(m_b, prob1, x, p_s), m_b, prob1, x, Fx, p_s, rnorm(Fx))
            x3 = Vector{T}(x)
            F3 = Vector{T}(Fx)
            st_a3 = start!!(
                init_state(m_a, prob3, x3, pv), m_a, prob3, x3, F3, pv, rnorm(F3))
            st_b3 = start!!(
                init_state(m_b, prob3, x3, pv), m_b, prob3, x3, F3, pv, rnorm(F3))
            split3 = false
            steps = 0
            while rnorm(Fx) > abstol && steps < opt.max_iterations
                steps += 1
                xa, _, st_a, _ = step!!(st_a, m_a, prob1, x, Fx, p_s)
                xb, Fb, st_b, info_b = step!!(st_b, m_b, prob1, x, Fx, p_s)
                if st_a.depth == 1 && st_b.depth == 1
                    x_m = st_b.x_prev + st_b.w_prev
                    @test rnorm(xb - xa) <= 8 * eps(R) * (rnorm(x_m) + rnorm(xb - x_m))
                    compared1 += 1
                end
                if !split3
                    xr = Vector(x)
                    Fr = Vector(Fx)
                    xa3, _, st_a3, _ = step!!(st_a3, m_a, prob3, copy(xr), copy(Fr), pv)
                    xb3, _, st_b3, _ = step!!(st_b3, m_b, prob3, copy(xr), copy(Fr), pv)
                    if (st_a3.depth == 0) != (st_b3.depth == 0)
                        split3 = true
                    elseif st_a3.depth == 1 && st_b3.depth == 1
                        x_m3 = st_b3.x_prev + st_b3.w_prev
                        @test rnorm(xb3 - xa3) <=
                              8 * eps(R) * (rnorm(x_m3) + rnorm(xb3 - x_m3))
                        compared3 += 1
                    end
                end
                (info_b.code == SUCCESS || info_b.code == STALLED) || break
                x, Fx = xb, Fb
            end
            @test steps >= 1
        end
    end
    # One compared step at `depth == 1` per element type, in each storage form.
    @test compared1 >= 1
    @test compared3 >= 1
end

@testset "Edge 1: a step on a state that no start!! saw ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    meth = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3)))

    x = xs[2]
    Fx = prob_r1.F(x, nothing)
    x_u, Fx_u, st_u, info_u = step!!(
        init_state(meth, prob_r1, x, nothing), meth, prob_r1, x, Fx, nothing)
    @test info_u.code == NONFINITE
    @test iszero(info_u.stepnorm)
    @test x_u == x
    @test Fx_u == Fx
    @test st_u.depth == 0

    x3 = Vector(x)
    Fx3 = Vector(Fx)
    x_u3, Fx_u3, st_u3, info_u3 = step!!(
        init_state(meth, prob_r3, x3, nothing), meth, prob_r3, x3, Fx3, nothing)
    @test info_u3.code == NONFINITE
    @test iszero(info_u3.stepnorm)
    @test x_u3 == x3
    @test Fx_u3 == Fx3
    @test st_u3.depth == 0
end

@testset "Edge 2: a non-finite incoming residual ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    meth = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3)))
    picard = Adapt.adapt(ToReal{R}(), Picard(; damping = 0.5))

    # The member fails hard on a non-finite residual: `Broyden` returns the code the member's own
    # `step!!` returns on the same input, and keeps `x` and the history.
    for bad in (T(NaN), T(Inf))
        x = xs[3]
        Fx = prob_r1.F(x, nothing)
        st = start!!(init_state(meth, prob_r1, x, nothing),
            meth, prob_r1, x, Fx, nothing, rnorm(Fx))
        for _ in 1:2
            x, Fx, st, _ = step!!(st, meth, prob_r1, x, Fx, nothing)
        end
        Fx_bad = setindex(Fx, bad, 2)
        st_pic = start!!(init_state(picard, prob_r1, x, nothing),
            picard, prob_r1, x, Fx, nothing, rnorm(Fx))
        _, _, _, info_pic = step!!(st_pic, picard, prob_r1, x, Fx_bad, nothing)
        x_out, _, st_out, info_out = step!!(st, meth, prob_r1, x, Fx_bad, nothing)
        @test is_hard_failure(info_pic)
        @test info_out.code == info_pic.code
        @test x_out == x
        @test st_out.depth == st.depth

        x3 = Vector(x)
        Fx3 = Vector(Fx)
        st3 = start!!(init_state(meth, prob_r3, x3, nothing),
            meth, prob_r3, x3, Fx3, nothing, rnorm(Fx3))
        for _ in 1:2
            x3, Fx3, st3, _ = step!!(st3, meth, prob_r3, x3, Fx3, nothing)
        end
        Fx_bad3 = copy(Fx3)
        Fx_bad3[2] = bad
        st_pic3 = start!!(init_state(picard, prob_r3, copy(x3), nothing),
            picard, prob_r3, copy(x3), copy(Fx3), nothing, rnorm(Fx3))
        _, _, _, info_pic3 = step!!(
            st_pic3, picard, prob_r3, copy(x3), copy(Fx_bad3), nothing)
        x_in3 = copy(x3)
        x_out3, _, st_out3, info_out3 = step!!(
            st3, meth, prob_r3, x3, Fx_bad3, nothing)
        @test is_hard_failure(info_pic3)
        @test info_out3.code == info_pic3.code
        @test x_out3 == x_in3
        @test st_out3.depth == st3.depth
    end

    # A member that survives the non-finite residual: the growth restart clears the history, the
    # step is the member's own candidate, and `failures` grows by one.
    x_mem = SVector{3, T}(T(0.6), T(1.2), T(-0.4))
    x_mem3 = Vector(x_mem)
    b_mem = Broyden(ScriptedBroydenMember(SUCCESS, x -> x_mem), Val(3))
    b_mem3 = Broyden(ScriptedR3BroydenMember(SUCCESS, x -> x_mem3), Val(3))

    x = xs[1]
    Fx = prob_r1.F(x, nothing)
    st = start!!(init_state(b_mem, prob_r1, x, nothing), b_mem, prob_r1, x, Fx, nothing,
        rnorm(Fx))
    for xk in xs[1:2]
        x, Fx, st, _ = step!!(st, b_mem, prob_r1, xk, prob_r1.F(xk, nothing), nothing)
    end
    @test st.depth == 1
    x_out, _, st_out, info_out = step!!(
        st, b_mem, prob_r1, xs[3], setindex(prob_r1.F(xs[3], nothing), T(NaN), 2), nothing)
    @test x_out == x_mem
    @test st_out.depth == 0
    @test info_out.failures == Int32(1)

    x03 = Vector{T}(xs[1])
    F03 = similar(x03)
    prob_r3.F(F03, x03, nothing)
    st3 = start!!(init_state(b_mem3, prob_r3, x03, nothing),
        b_mem3, prob_r3, x03, F03, nothing, rnorm(F03))
    for xk in xs[1:2]
        xk3 = Vector{T}(xk)
        fk3 = similar(xk3)
        prob_r3.F(fk3, xk3, nothing)
        _, _, st3, _ = step!!(st3, b_mem3, prob_r3, xk3, fk3, nothing)
    end
    @test st3.depth == 1
    xk3 = Vector{T}(xs[3])
    fk3 = similar(xk3)
    prob_r3.F(fk3, xk3, nothing)
    fk3[2] = T(NaN)
    x_out3, _, st_out3, info_out3 = step!!(st3, b_mem3, prob_r3, xk3, fk3, nothing)
    @test x_out3 == x_mem3
    @test st_out3.depth == 0
    @test info_out3.failures == Int32(1)
end

# Two steps of `meth_r1` and `meth_r3` from `xa` and then from `xb`, so that a test asserts one
# degenerate-column trigger's restart on both storage forms. Returns the state after the second
# step and its `StepInfo`, R1 first.
function _degenerate_restart(meth_r1, meth_r3, prob_r1, prob_r3, xa, xb)
    Fa = prob_r1.F(xa, nothing)
    st1 = start!!(init_state(meth_r1, prob_r1, xa, nothing),
        meth_r1, prob_r1, xa, Fa, nothing, rnorm(Fa))
    _, _, st1, _ = step!!(st1, meth_r1, prob_r1, xa, Fa, nothing)
    Fb = prob_r1.F(xb, nothing)
    _, _, st1, info1 = step!!(st1, meth_r1, prob_r1, xb, Fb, nothing)

    xa3 = Vector(xa)
    Fa3 = Vector(Fa)
    st3 = start!!(init_state(meth_r3, prob_r3, xa3, nothing),
        meth_r3, prob_r3, xa3, Fa3, nothing, rnorm(Fa3))
    _, _, st3, _ = step!!(st3, meth_r3, prob_r3, copy(xa3), copy(Fa3), nothing)
    _, _, st3, info3 = step!!(st3, meth_r3, prob_r3, Vector(xb), Vector(Fb), nothing)
    return st1, info1, st3, info3
end

@testset "Edge 3: the empty history, a degenerate newest column, and the filter count ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    prob_r1 = NonlinearProblem((x, p) -> A * x - b)
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    meth = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3)))

    # The first step after `start!!` is the member step, with the member's failure count.
    x = xs[1]
    Fx = prob_r1.F(x, nothing)
    x_m = x .- R(0.5) .* Fx
    st = start!!(
        init_state(meth, prob_r1, x, nothing), meth, prob_r1, x, Fx, nothing, rnorm(Fx))
    x_1, _, st_1, info_1 = step!!(st, meth, prob_r1, x, Fx, nothing)
    @test x_1 == x_m
    @test info_1.failures == Int32(0)
    @test st_1.depth == 0
    @test st_1.has_prev
    x3 = Vector(x)
    Fx3 = Vector(Fx)
    st3 = start!!(init_state(meth, prob_r3, x3, nothing),
        meth, prob_r3, x3, Fx3, nothing, rnorm(Fx3))
    x_13, _, st_13, info_13 = step!!(st3, meth, prob_r3, copy(x3), copy(Fx3), nothing)
    @test x_13 == Vector(x_m)
    @test info_13.failures == Int32(0)
    @test st_13.depth == 0
    @test st_13.has_prev

    # The degenerate newest column: the same iterate twice gives `Δw = 0`, so the degenerate-column
    # restart runs before the filters and clears the history, one failure more.
    xw = xs[2]
    Fw = prob_r1.F(xw, nothing)
    st = start!!(
        init_state(meth, prob_r1, x, nothing), meth, prob_r1, x, Fx, nothing, rnorm(Fx))
    _, _, st, _ = step!!(st, meth, prob_r1, xw, Fw, nothing)
    _, _, st, info_d = step!!(st, meth, prob_r1, xw, Fw, nothing)
    @test info_d.failures == Int32(1)
    @test st.depth == 0
    @test st.has_prev
    xw3 = Vector(xw)
    Fw3 = Vector(Fw)
    st3 = start!!(init_state(meth, prob_r3, Vector(x), nothing),
        meth, prob_r3, Vector(x), Vector(Fx), nothing, rnorm(Fx))
    _, _, st3, _ = step!!(st3, meth, prob_r3, copy(xw3), copy(Fw3), nothing)
    _, _, st3, info_d3 = step!!(st3, meth, prob_r3, copy(xw3), copy(Fw3), nothing)
    @test info_d3.failures == Int32(1)
    @test st3.depth == 0
    @test st3.has_prev

    # The other two triggers of the degenerate-column restart, each reached alone. The same iterate
    # twice fires both at once; here one column measure is tiny and the other is not.
    # (a) `0 < ‖Δw‖² <= floatmin(R)` with a `Δx` of normal size: the member shifts the zero `x₂`
    # by `sqrt(floatmin(R)) x₁`, so the shift is exact and a quarter of a unit of `x₁` moves `Δw`
    # by `sqrt(floatmin(R))/4` while `Δx` stays at a quarter. The residual is
    # `(s x₁, x₂ - 2, x₃ - 3)`, which the shift leaves unchanged, so the step reaches the
    # degenerate-column test and not a growth restart.
    s_tiny = sqrt(floatmin(R))
    tiny_r1 = NonlinearProblem(
        (x, p) -> SVector{3, T}(s_tiny * x[1], x[2] - T(2), x[3] - T(3)))
    tiny_r3 = NonlinearProblem((F, x, p) -> (
        F[1] = s_tiny * x[1]; F[2] = x[2] - T(2); F[3] = x[3] - T(3); nothing))
    meth_tiny = Broyden(
        ScriptedBroydenMember(
            SUCCESS, x -> x + SVector{3, T}(zero(T), s_tiny * x[1], zero(T))), Val(3))
    meth_tiny_r3 = Broyden(
        ScriptedR3BroydenMember(
            SUCCESS, x -> x + [zero(T), s_tiny * x[1], zero(T)]), Val(3))
    xa = SVector{3, T}(T(1) / 2, zero(T), T(3) / 4)
    xb = xa + SVector{3, T}(T(1) / 4, zero(T), zero(T))
    dw_tiny = s_tiny * SVector{3, T}(zero(T), xb[1] - xa[1], zero(T))
    @test zero(R) < norm2(dw_tiny) <= floatmin(R)
    @test rnorm(xb - xa) > eps(R) * max(one(R), rnorm(xb))
    st_t1, info_t1, st_t3, info_t3 = _degenerate_restart(
        meth_tiny, meth_tiny_r3, tiny_r1, tiny_r3, xa, xb)
    @test st_t1.depth == 0 && info_t1.failures == Int32(1)
    @test st_t3.depth == 0 && info_t3.failures == Int32(1)

    # (b) `0 < rnorm(Δx) <= eps(R) * max(1, rnorm(x))` with a `Δw` above `floatmin(R)`: the
    # iterate moves by `1e-17` in `x₁`, which moves only the member's third column measure, so
    # `γ₁` stays of order one and an accelerated step is accepted with `depth == 1`.
    stag_r1 = NonlinearProblem((x, p) -> x - SVector{3, T}(T(1), T(2), T(3)))
    stag_r3 = NonlinearProblem(
        (F, x, p) -> (F .= x .- SVector{3, T}(T(1), T(2), T(3)); nothing))
    meth_stag = Broyden(
        ScriptedBroydenMember(
            SUCCESS, x -> x + SVector{3, T}(T(3) / 10, zero(T), T(10)^10 * x[1])), Val(3))
    meth_stag_r3 = Broyden(
        ScriptedR3BroydenMember(
            SUCCESS, x -> x + [T(3) / 10, zero(T), T(10)^10 * x[1]]), Val(3))
    xs_a = SVector{3, T}(zero(T), T(-1) / 4, T(2) / 5)
    xs_b = xs_a + SVector{3, T}(T(1) / T(10)^17, zero(T), zero(T))
    dw_stag = SVector{3, T}(zero(T), zero(T), T(10)^10 * xs_b[1] - T(10)^10 * xs_a[1])
    @test norm2(dw_stag) > floatmin(R)
    @test zero(R) < rnorm(xs_b - xs_a) <= eps(R) * max(one(R), rnorm(xs_b))
    st_s1, info_s1, st_s3, info_s3 = _degenerate_restart(
        meth_stag, meth_stag_r3, stag_r1, stag_r3, xs_a, xs_b)
    @test st_s1.depth == 0 && info_s1.failures == Int32(1)
    @test st_s3.depth == 0 && info_s3.failures == Int32(1)

    # A step whose filters drop one of two columns counts `d - p_kept` failures and is the
    # accelerated step.
    u = SVector{3, T}(T(0.5), T(-0.25), T(0.375))
    seq_x = (zero(u), u, 3 * u)
    st = start!!(init_state(meth, prob_r1, seq_x[1], nothing),
        meth, prob_r1, seq_x[1], prob_r1.F(seq_x[1], nothing), nothing,
        rnorm(prob_r1.F(seq_x[1], nothing)))
    info_f = nothing
    for xk in seq_x
        _, _, st, info_f = step!!(
            st, meth, prob_r1, xk, prob_r1.F(xk, nothing), nothing)
    end
    @test info_f.code == SUCCESS
    @test info_f.failures == Int32(1)
    @test st.depth == 1
    st3 = start!!(init_state(meth, prob_r3, Vector(seq_x[1]), nothing),
        meth, prob_r3, Vector(seq_x[1]), Vector(prob_r1.F(seq_x[1], nothing)), nothing,
        rnorm(prob_r1.F(seq_x[1], nothing)))
    info_f3 = nothing
    for xk in seq_x
        xk3 = Vector(xk)
        fk3 = Vector(prob_r1.F(xk, nothing))
        _, _, st3, info_f3 = step!!(st3, meth, prob_r3, xk3, fk3, nothing)
    end
    @test info_f3.code == SUCCESS
    @test info_f3.failures == Int32(1)
    @test st3.depth == 1
end

@testset "Edge 7: a device array ($T)" for T in ELTYPES
    R = real(T)
    n = 4
    method = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), Val(4)))
    prob_r1 = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)
    prob_r3 = NonlinearProblem(Holomorphic())
    pv = fill(T(5), n)
    p_s = SVector{n, T}(pv)
    x0 = fill(T(1 / 2), n)
    x_n, _ = solve(prob_r1, SVector{n, T}(x0), Newton(), p_s)
    J_star = Matrix(prob_r1.J(x_n, p_s))
    Fx0 = similar(x0)
    prob_r3.F(Fx0, x0, pv)
    abstol = max(Options(R).f_abstol, Options(R).f_reltol * rnorm(Fx0))

    for AT in (Array, JLArray, :ka)
        conv = if AT === :ka
            v -> (a = KA.allocate(KA_BACKEND, T, n); copyto!(a, v); a)
        elseif AT === Array
            v -> Vector{T}(v)
        else
            v -> JLArray(v)
        end
        xa = conv(x0)
        pa = conv(pv)
        Fa = similar(xa)
        prob_r3.F(Fa, xa, pa)
        st = start!!(init_state(method, prob_r3, xa, pa), method, prob_r3, xa, Fa, pa,
            rnorm(Fa))
        x_step, Fx_step, _, info = step!!(st, method, prob_r3, copy(xa), copy(Fa), pa)
        @test info.code == SUCCESS
        @test all(isfinite, Array(x_step))
        @test rnorm(Array(Fx_step)) <= rnorm(Fx0)
        solver = init(prob_r3, copy(xa), method, pa)
        x_sol, st_sol = solve!(solver, copy(xa))
        @test st_sol.code == SUCCESS
        Fx_sol = similar(x0)
        prob_r3.F(Fx_sol, Array(x_sol), pv)
        @test rnorm(Fx_sol) <= abstol
        @test rnorm(Vector(Array(x_sol)) - Vector(x_n)) <= abstol * cond(J_star)
    end
end

@testset "MU5: the post-step residual-growth restart ($T)" for T in ELTYPES
    R = real(T)
    A = T.(POST_RESTART_A)
    c = T(POST_RESTART_C)
    b = T.(POST_RESTART_B)
    damping = R(POST_RESTART_DAMPING)
    x0 = T.(POST_RESTART_X)
    # `restart_factor = 3`, and this start: the third step's accelerated iterate grows the
    # residual past `3 fnorm_best` while the incoming residual of the third step does not, so only
    # the post-step safeguard can fire there.
    meth = Adapt.adapt(ToReal{R}(),
        Broyden(Picard(; damping = POST_RESTART_DAMPING), Val(3); restart_factor = 3.0))
    prob_r1 = NonlinearProblem((x, p) -> A * x .+ c .* x .^ 3 .- b)
    prob_r3 = NonlinearProblem(
        (F, x, p) -> (mul!(F, A, x); @. F = F + c * x ^ 3 - b; nothing))

    x = SVector{3, T}(x0)
    Fx = prob_r1.F(x, nothing)
    st = start!!(
        init_state(meth, prob_r1, x, nothing), meth, prob_r1, x, Fx, nothing, rnorm(Fx))
    for _ in 1:2
        x, Fx, st, _ = step!!(st, meth, prob_r1, x, Fx, nothing)
    end
    @test st.depth >= 1
    @test rnorm(Fx) <= 3 * st.fnorm_best
    x_m = x .- damping .* Fx
    x_out, _, st_out, info_out = step!!(st, meth, prob_r1, x, Fx, nothing)
    @test rnorm(x_out - x_m) <= 16 * eps(R) * rnorm(x_m)
    @test st_out.depth == 0
    @test info_out.failures == Int32(1)
    # The post-step safeguard clears the history: the restarted state holds no column.
    @test all(iszero, st_out.ΔX) && all(iszero, st_out.ΔW)

    x3 = Vector{T}(x0)
    Fx3 = similar(x3)
    prob_r3.F(Fx3, x3, nothing)
    st3 = start!!(init_state(meth, prob_r3, x3, nothing),
        meth, prob_r3, x3, Fx3, nothing, rnorm(Fx3))
    for _ in 1:2
        x3, Fx3, st3, _ = step!!(st3, meth, prob_r3, x3, Fx3, nothing)
    end
    @test st3.depth >= 1
    @test rnorm(Fx3) <= 3 * st3.fnorm_best
    x_m3 = x3 .- damping .* Fx3
    x_out3, _, st_out3, info_out3 = step!!(
        st3, meth, prob_r3, copy(x3), copy(Fx3), nothing)
    @test rnorm(x_out3 - x_m3) <= 16 * eps(R) * rnorm(x_m3)
    @test st_out3.depth == 0
    @test info_out3.failures == Int32(1)
    @test all(iszero, st_out3.ΔX) && all(iszero, st_out3.ΔW)
end

@testset "A rejected weight takes the member step in R3 ($T)" for T in ELTYPES
    R = real(T)
    A, b, xs = broyden_sequence(T)
    # `A` and `b` build the in-place residual; the rejection is a property of the R3 step alone.
    prob_r3 = NonlinearProblem((F, x, p) -> (
        mul!(F, Matrix(A), x); F .-= Vector(b); nothing))
    # `M_alpha = 1` refuses a weight whose affine sum exceeds one, which any nonzero `γ` gives
    # here. The step is then the member step `x_m`, counted as one failure.
    damp = R(1 // 2)
    meth = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 0.5), Val(3); M_alpha = 1.0))
    z = xs[1]
    z3 = Vector(z)
    Fz3 = similar(z3)
    prob_r3.F(Fz3, z3, nothing)
    st = start!!(init_state(meth, prob_r3, z3, nothing),
        meth, prob_r3, z3, Fz3, nothing, rnorm(Fz3))
    x_1, F_1, st_1, _ = step!!(st, meth, prob_r3, copy(z3), copy(Fz3), nothing)
    x_m = x_1 .- damp .* F_1
    x_2, _, _, info_2 = step!!(st_1, meth, prob_r3, copy(x_1), copy(F_1), nothing)
    @test rnorm(x_2 - x_m) <= 16 * eps(R) * rnorm(x_m)
    @test info_2.failures >= 1
end

@testset "Check 1, Check 3, Check 4, and Edge 7: R1, R2, and R3 convergence, agreement, and allocations" begin
    @testset "R1 SVector and scalar convergence, allocations, and inference ($T)" for T in ELTYPES
        R = real(T)
        opt = Options(R)
        # `solve` adapts the method to `real(eltype(x₀))`; a test that calls `step!!` itself does
        # the same, so that the member's `Static{Float64}` step does not promote a `Float32`
        # iterate.
        method_br = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), Val(4)))

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
            bm = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), m_val))
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

        # A complex row with a real start and real parameters leaves every iterate exactly real,
        # so the imaginary part of the Gram matrix and of the weights is never read. The complex
        # types therefore also take a start and parameters with no real entry.
        datasets = T <: Complex ?
                   ((T(1 / 2), T(5)), (T(1 / 2, 1 / 3), T(5, 1))) :
                   ((T(1 / 2), T(5)),)
        for n in (2, 4)
            perm = r1_shift(Int, n)
            problems = (
                ("R1Holomorphic",
                    NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)),
                ("R1Coupled",
                    NonlinearProblem(R1Coupled(perm), (x, p) -> r1_coupled_jacobian(x, p, perm)))
            )
            for (x0_c, p_c) in datasets
                pv = fill(p_c, n)
                p_s = SVector{n, T}(pv)
                for (name, prob) in problems
                    x₀ = SVector{n, T}(fill(x0_c, n))
                    x_n, st_n = solve(prob, x₀, Newton(), p_s)
                    x_br, st_br = @inferred solve(prob, x₀, method_br, p_s)
                    abstol = max(opt.f_abstol, opt.f_reltol * rnorm(prob.F(x₀, p_s)))
                    J_star = Matrix(prob.J(x_n, p_s))
                    @test st_n.code == SUCCESS
                    @test st_br.code == SUCCESS
                    @test rnorm(prob.F(x_br, p_s)) <= abstol
                    @test rnorm(x_br - x_n) <= abstol * cond(J_star)

                    for m_val in (Val(1), Val(2), Val(3), Val(4))
                        bm = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), m_val))
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
    end

    @testset "R3 Array and JLArray convergence on Holomorphic and Coupled ($T)" for T in ELTYPES
        R = real(T)
        opt = Options(R)
        method_br = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), Val(4)))
        datasets = T <: Complex ?
                   ((T(1 / 2), T(5)), (T(1 / 2, 1 / 3), T(5, 1))) :
                   ((T(1 / 2), T(5)),)
        for n in (2, 4)
            perm = r1_shift(Int, n)
            perm_r3 = cyclic_perm(Array, n)
            # The R3 stubs of `test/helpers/adproblems.jl`. A host permutation indexes an `Array`,
            # a `JLArray` and a KA `CPU()` array alike.
            problems = (
                ("Holomorphic",
                    NonlinearProblem(Holomorphic()),
                    NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian),
                    r3_holomorphic_jacobian!),
                ("Coupled",
                    NonlinearProblem(Coupled(perm_r3)),
                    NonlinearProblem(R1Coupled(perm), (x, p) -> r1_coupled_jacobian(x, p, perm)),
                    (J, x, p) -> r3_coupled_jacobian!(J, x, p, perm_r3))
            )
            for (x0_c, p_c) in datasets
                p_vec = fill(p_c, n)
                p_s = SVector{n, T}(p_vec)
                for (name, prob_r3, prob_r1, jac_r3) in problems
                    x₀_s = SVector{n, T}(fill(x0_c, n))
                    x_n_s, _ = solve(prob_r1, x₀_s, Newton(), p_s)
                    J_star = Matrix(prob_r1.J(x_n_s, p_s))

                    x₀ = fill(x0_c, n)
                    Fx₀ = similar(x₀)
                    prob_r3.F(Fx₀, x₀, p_vec)
                    abstol = max(opt.f_abstol, opt.f_reltol * rnorm(Fx₀))

                    # Check 1 runs `Newton()` on the R3 problem too, with its in-place Jacobian,
                    # and compares `Broyden` with that root.
                    prob_n3 = NonlinearProblem(prob_r3.F, jac_r3)
                    x_n3, st_n3 = solve(prob_n3, copy(x₀), Newton(), p_vec)
                    Fx_n3 = similar(x_n3)
                    prob_n3.F(Fx_n3, x_n3, p_vec)
                    @test st_n3.code == SUCCESS
                    @test rnorm(Fx_n3) <= abstol

                    solver = init(prob_r3, copy(x₀), method_br, p_vec)
                    x_br, st_br = solve!(solver, copy(x₀))
                    @test st_br.code == SUCCESS
                    Fx_br = similar(x_br)
                    prob_r3.F(Fx_br, x_br, p_vec)
                    @test rnorm(Fx_br) <= abstol
                    @test rnorm(x_br - x_n3) <= abstol * cond(J_star)
                    @test rnorm(x_br - Vector(x_n_s)) <= abstol * cond(J_star)

                    # Check 4: R3 step!! and solve! allocate 0 bytes on Array after warm-up. The
                    # residual is `Scaled()`, one fused broadcast: `Coupled`'s `x[perm]` gather
                    # allocates a new array at every evaluation — 64 bytes at n = 2, `Float32` —
                    # which no method can avoid.
                    prob_alloc = NonlinearProblem(Scaled())
                    p_alloc = fill(T(2), n)
                    Fx_alloc = similar(x₀)
                    prob_alloc.F(Fx_alloc, x₀, p_alloc)
                    st_alloc = init_state(method_br, prob_alloc, x₀, p_alloc)
                    st_alloc = start!!(
                        st_alloc, method_br, prob_alloc, x₀, Fx_alloc, p_alloc, rnorm(Fx_alloc))
                    x_work = copy(x₀)
                    Fx_work = copy(Fx_alloc)
                    if Base.JLOptions().check_bounds != 1
                        # Skipped where a run forces --check-bounds=yes.
                        @test _r3_broyden_step_alloc_probe(
                            st_alloc, method_br, prob_alloc, x_work, x₀, Fx_work, Fx_alloc,
                            p_alloc) == 0
                        solver_alloc = init(prob_alloc, copy(x₀), method_br, p_alloc)
                        @test _r3_broyden_solve_alloc_probe(solver_alloc, x_work, x₀) == 0
                    end

                    # Edge 7: the JLArray R3 solve, with scalar indexing forbidden, reaches the
                    # same root and satisfies the residual stop of its own start.
                    x₀_jl = JLArray(x₀)
                    p_jl = JLArray(p_vec)
                    solver_jl = init(prob_r3, copy(x₀_jl), method_br, p_jl)
                    x_jl, st_jl = solve!(solver_jl, copy(x₀_jl))
                    @test st_jl.code == SUCCESS
                    Fx_jl = similar(x_jl)
                    prob_r3.F(Fx_jl, x_jl, p_jl)
                    @test rnorm(Fx_jl) <= abstol
                    @test rnorm(Array(x_jl) - Vector(x_n_s)) <= abstol * cond(J_star)

                    # Check 1 names R3 `JLArray` for `Newton()` too: it runs on the same problem
                    # with its default AD Jacobian and reaches a root that `Broyden`'s `JLArray`
                    # root lies within the forward error of.
                    x_n_jl, st_n_jl = solve(prob_r3, copy(x₀_jl), Newton(), p_jl)
                    @test st_n_jl.code == SUCCESS
                    Fx_n_jl = similar(x_n_jl)
                    prob_r3.F(Fx_n_jl, x_n_jl, p_jl)
                    @test rnorm(Fx_n_jl) <= abstol
                    @test rnorm(Array(x_jl) - Array(x_n_jl)) <= abstol * cond(J_star)
                end
            end
        end
    end

    @testset "Check 3: R1 and R3 agree on every step of a default solve ($T)" for T in ELTYPES
        R = real(T)
        opt = Options(R)
        datasets = T <: Complex ?
                   ((T(1 / 2), T(5)), (T(1 / 2, 1 / 3), T(5, 1))) :
                   ((T(1 / 2), T(5)),)
        for n in (2, 4), m in (1, 2, 3, 4)

            perm = r1_shift(Int, n)
            for (x0_c, p_c) in datasets
                pv = fill(p_c, n)
                p_s = SVector{n, T}(pv)
                method = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), Val(m)))
                problems = (
                    ("Holomorphic", NonlinearProblem(R1Holomorphic()),
                        NonlinearProblem(Holomorphic())),
                    ("Coupled",
                        NonlinearProblem(R1Coupled(perm),
                            (x, p) -> r1_coupled_jacobian(x, p, perm)),
                        NonlinearProblem(Coupled(cyclic_perm(Array, n)))))
                for (name, prob_r1, prob_r3) in problems
                    x = SVector{n, T}(fill(x0_c, n))
                    Fx = prob_r1.F(x, p_s)
                    abstol = max(opt.f_abstol, opt.f_reltol * rnorm(Fx))
                    st = start!!(init_state(method, prob_r1, x, p_s),
                        method, prob_r1, x, Fx, p_s, rnorm(Fx))
                    # The R3 states take the R1 trajectory's iterates, so before every step the
                    # two histories hold the same columns: each step is one step from the same
                    # populated state. A host permutation indexes an `Array`, a `JLArray` and a
                    # KA `CPU()` array alike.
                    trackers = map((Array, JLArray, :ka)) do AT
                        conv = if AT === :ka
                            v -> (a = KA.allocate(KA_BACKEND, T, n); copyto!(a, v); a)
                        else
                            v -> AT(v)
                        end
                        pa = conv(pv)
                        xa = conv(Vector(x))
                        Fa = conv(Vector(Fx))
                        sr = Ref{Any}(start!!(init_state(method, prob_r3, xa, pa),
                            method, prob_r3, xa, Fa, pa, rnorm(Fx)))
                        (AT, conv, pa, sr)
                    end
                    steps = 0
                    while rnorm(Fx) > abstol && steps < opt.max_iterations
                        steps += 1
                        x₊, F₊, st_new, info = step!!(st, method, prob_r1, x, Fx, p_s)
                        (info.code == SUCCESS || info.code == STALLED) || break
                        tol = n * eps(R) * _multisecant_step_scale(st_new, Val(m), R)
                        for (AT, conv, pa, sr) in trackers
                            xr, _, sr_new, _ = step!!(
                                sr[], method, prob_r3, conv(Vector(x)), conv(Vector(Fx)), pa)
                            sr[] = sr_new
                            @test Int(sr_new.depth) == Int(st_new.depth)
                            if Int(st_new.depth) >= 1
                                @test rnorm(Array(xr) - Vector(x₊)) <= tol
                            end
                        end
                        x, Fx, st = x₊, F₊, st_new
                    end
                    @test steps >= 1
                end
            end
        end
    end

    @testset "Check 3: CoupledShift, R1 against R3 ($T)" for T in ELTYPES
        R = real(T)
        n = 4
        # This row's residual has no root at zero, so the relative bound
        # `n * eps(R) * rnorm(x₊)` stays above the round-off floor of the weights; and for a
        # complex element type the shift and the start carry an imaginary part, so a Gram matrix
        # formed with `transpose` instead of `adjoint` cannot pass.
        shift = T <: Complex ? T(8, 3) : T(8)
        x0_c = T <: Complex ? T(1 / 2, 1 / 3) : T(1 / 2)
        prob_r1 = NonlinearProblem(CoupledShift(r1_shift(Int, n), shift))
        prob_r3 = NonlinearProblem(CoupledShift(cyclic_perm(Array, n), shift))
        p_vec = fill(T(5), n)
        p_s = SVector{n, T}(p_vec)
        p_jl = JLArray(p_vec)
        p_ka = KA.allocate(KA_BACKEND, T, n)
        copyto!(p_ka, p_vec)

        for m in (1, 2, 3, 4)
            method = Adapt.adapt(ToReal{R}(), Broyden(Picard(; damping = 1 / 10), Val(m)))
            x_r1 = SVector{n, T}(fill(x0_c, n))
            Fx_r1 = prob_r1.F(x_r1, p_s)
            st_r1 = start!!(init_state(method, prob_r1, x_r1, p_s),
                method, prob_r1, x_r1, Fx_r1, p_s, rnorm(Fx_r1))

            x_arr = Vector(x_r1)
            Fx_arr = Vector(Fx_r1)
            st_arr = init_state(method, prob_r3, x_arr, p_vec)
            st_arr = start!!(st_arr, method, prob_r3, x_arr, Fx_arr, p_vec, rnorm(Fx_arr))

            x_jl = JLArray(x_arr)
            Fx_jl = JLArray(Fx_arr)
            st_jl = init_state(method, prob_r3, x_jl, p_jl)
            st_jl = start!!(st_jl, method, prob_r3, x_jl, Fx_jl, p_jl, rnorm(Fx_jl))

            x_ka = KA.allocate(KA_BACKEND, T, n)
            copyto!(x_ka, x_arr)
            Fx_ka = KA.allocate(KA_BACKEND, T, n)
            copyto!(Fx_ka, Fx_arr)
            st_ka = init_state(method, prob_r3, x_ka, p_ka)
            st_ka = start!!(st_ka, method, prob_r3, x_ka, Fx_ka, p_ka, rnorm(Fx_ka))

            for step_idx in 1:(m + 1)
                x_r1, Fx_r1, st_r1, _ = step!!(st_r1, method, prob_r1, x_r1, Fx_r1, p_s)
                x_arr, Fx_arr, st_arr, _ = step!!(
                    st_arr, method, prob_r3, x_arr, Fx_arr, p_vec)
                x_jl, Fx_jl, st_jl, _ = step!!(st_jl, method, prob_r3, x_jl, Fx_jl, p_jl)
                x_ka, Fx_ka, st_ka, _ = step!!(st_ka, method, prob_r3, x_ka, Fx_ka, p_ka)

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
        # The default `f_abstol` is `0.0`: the residual stop of the default `Options` is relative
        # to the start, as in `test/composition/anderson.jl`.
        @test rnorm(prob_mv.F(x_mv, nothing)) <=
              Options(Float64).f_reltol * rnorm(prob_mv.F(x0_mv, nothing))
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

    @testset "Edge 8: m > 4 is refused in R1 and runs in R3" begin
        for T in ELTYPES
            n = 6
            # R1: `m = 5` throws, from both entry points, with a message that names the depth.
            x0_s = SVector{n, T}(fill(T(1 / 2), n))
            prob_r1 = NonlinearProblem(R1Holomorphic())
            @test_throws ArgumentError init_state(
                Broyden(Picard(), Val(5)), prob_r1, x0_s, nothing)
            @test_throws ArgumentError init(prob_r1, x0_s, Broyden(Picard(), Val(5)))
            err = try
                init_state(Broyden(Picard(), Val(5)), prob_r1, x0_s, nothing)
            catch e
                e
            end
            @test err isa ArgumentError
            @test err.msg == "R1 Broyden history depth m must be <= 4, got 5"

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
            _, st_sol = solve!(solver, copy(x0_vec))
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

        prob_r3 = NonlinearProblem(Holomorphic())
        x0_r3 = [0.5, 0.5]
        Fx0_r3 = similar(x0_r3)
        p_r3 = [5.0, 5.0]
        prob_r3.F(Fx0_r3, x0_r3, p_r3)
        st_r3 = init_state(m_br, prob_r3, x0_r3, p_r3)
        st_r3 = start!!(st_r3, m_br, prob_r3, x0_r3, Fx0_r3, p_r3, rnorm(Fx0_r3))
        JET.@test_opt target_modules = (GeometricSolvers,) step!!(
            st_r3, m_br, prob_r3, x0_r3, Fx0_r3, p_r3)
    end

    @testset "Composition with Multiplicative, Repeat, Switch" begin
        F_stat(x, p) = SVector(2 * x[1] + 0.2 * x[2] - 1.0, 0.2 * x[1] + 3 * x[2] - 2.0)
        x0 = SVector(0.0, 0.0)
        abstol = Options(Float64).f_reltol * rnorm(F_stat(x0, nothing))

        # Broyden composed inside Repeat, Multiplicative, and Switch
        prob_nl = NonlinearProblem(F_stat)
        for comb in (
            Repeat(Broyden(Picard(; damping = 0.3), Val(3)), 2),
            Multiplicative(Picard(; damping = 0.3), Broyden(Picard(; damping = 0.3), Val(3))),
            Switch(Broyden(Picard(; damping = 0.3), Val(3)), Newton(); threshold = 1e-1)
        )
            x_c, st_c = solve(prob_nl, x0, comb)
            @test isconverged(st_c)
            @test rnorm(F_stat(x_c, nothing)) <= abstol
        end
    end
end

@testset "Restarts, safeguards, failures of the member, finish!!, and solver reuse" begin
    # Hard failure of inner member returns immediately and preserves history
    prob_r1 = NonlinearProblem((x, p) -> SVector(x[1] - 1.0, x[2] - 2.0))
    x0 = SVector(0.0, 0.0)
    Fx0 = prob_r1.F(x0, nothing)
    for hard_code in (SINGULAR, NONFINITE, LINESEARCH_FAILED)
        member_hard = ScriptedBroydenMember(hard_code, x -> x)
        b_hard = Broyden(member_hard, Val(3))
        st0 = start!!(init_state(b_hard, prob_r1, x0, nothing),
            b_hard, prob_r1, x0, Fx0, nothing, rnorm(Fx0))
        x_out, _, st_out, info_out = step!!(st0, b_hard, prob_r1, x0, Fx0, nothing)
        @test info_out.code == hard_code
        @test iszero(info_out.stepnorm)
        @test x_out == x0
        @test st_out.depth == st0.depth
        @test st_out.current_depth == st0.current_depth
        @test st_out.has_prev == st0.has_prev
    end

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
    _, _, st_s2, info2 = step!!(st_s1, b_soft, prob_r1, x1, Fx1, nothing)
    @test info2.stepnorm > 0
    @test st_s2.depth >= 1

    # A step that keeps every history column counts no filter failure: `filter_failures` adds
    # `d - p`, and here `d == p == 1`.
    A_mat = @SMatrix [2.0 0.8 0.3; 0.4 1.7 0.5; 0.2 0.3 1.4]
    b_vec = SVector(1.0, -0.5, 0.8)
    prob_lin = NonlinearProblem((x, p) -> A_mat * x - b_vec)
    z3 = SVector(0.0, 0.0, 0.0)
    member_moving = ScriptedBroydenMember(SUCCESS, x -> SVector(0.2, 0.1, -0.1) + 0.5 * x)
    b_keep = Broyden(member_moving, Val(3))
    st_k0 = start!!(init_state(b_keep, prob_lin, z3, nothing),
        b_keep, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    x_k1, F_k1, st_k1, _ = step!!(st_k0, b_keep, prob_lin, z3, -b_vec, nothing)
    _, _, st_k2, info_k2 = step!!(st_k1, b_keep, prob_lin, x_k1, F_k1, nothing)
    @test st_k2.depth == 1
    @test info_k2.failures == 0

    # M_alpha weight-bound rejection and safeguard_cos angle restart
    # Tight M_alpha = 1.0 forces weight rejection when any γ_c != 0 makes ∑|α_j| > 1
    b_tight = Broyden(Picard(; damping = 0.5), Val(3); M_alpha = 1.0)
    st_t = start!!(init_state(b_tight, prob_lin, z3, nothing),
        b_tight, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    xt1, Ft1, st_t1, _ = step!!(st_t, b_tight, prob_lin, z3, -b_vec, nothing)
    _, _, _, info_t2 = step!!(st_t1, b_tight, prob_lin, xt1, Ft1, nothing)
    @test info_t2.failures >= 1

    # Tight safeguard_cos = 1e-6 triggers angle restart on aligned consecutive corrections
    b_cos = Broyden(Picard(; damping = 0.5), Val(3); safeguard_cos = 1e-6)
    st_c = start!!(init_state(b_cos, prob_lin, z3, nothing),
        b_cos, prob_lin, z3, -b_vec, nothing, rnorm(-b_vec))
    xc1, Fc1, st_c1, _ = step!!(st_c, b_cos, prob_lin, z3, -b_vec, nothing)
    _, _, st_c2, info_c2 = step!!(st_c1, b_cos, prob_lin, xc1, Fc1, nothing)
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
