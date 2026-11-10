using Adapt: Adapt
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: norm
using Random: Xoshiro
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: AbstractNonlinearProblem, AutoForwardDiff, BlockEliminated,
                        GaussNewton,
                        LeastSquaresProblem, LevenbergMarquardt, NONFINITE, Newton,
                        NonlinearProblem, NonlinearSolver, Options, QRFactorization,
                        SUCCESS,
                        SVDFactorization, SolverStatus, StationarityProblem, ToReal,
                        VariableProjection, finish!!, init, init_state, jacobian!!, jvp!!,
                        prepare_ad, residual!!, rnorm, solve, solve!, solve!!, start!!,
                        step!!

const GS = GeometricSolvers

include("../helpers/matrix.jl")

allowscalar(false)

# Allocation measured through a barrier, after one call that compiles the path.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

# --- Test fixtures ---

# 4-unknown system parameterised by `p`:
# - `p === nothing` or `p isa Real`: cubic coefficient `c = p === nothing ? 0 : p` on (x1, x2).
#   With `c = 0`, both E_y = (1, 2) and E_y = (2, 3) are linear in y = (x1, x2); with `c = 1/5`,
#   both are genuinely nonlinear and strictly invertible in y = (x1, x2).
function f_elim(x, p)
    T = eltype(x)
    c = p === nothing ? zero(T) : T(p)
    return SVector{4, T}(
        2 * x[1] + c * x[1]^3 + T(1 // 2) * x[2] + sin(x[3]) - one(T),
        T(1 // 2) * x[1] + 3 * x[2] + c * x[2]^3 - cos(x[4]) - 2 * one(T),
        x[3]^3 + 2 * x[3] + x[1] - x[2] - T(1 // 2),
        x[4]^3 + 2 * x[4] + x[2] - T(3 // 2)
    )
end

function j_elim(x, p)
    T = eltype(x)
    c = p === nothing ? zero(T) : T(p)
    Z = zero(T)
    O = one(T)
    return SMatrix{4, 4, T, 16}(
        2 * O + 3 * c * x[1]^2, T(1 // 2), O, Z,
        T(1 // 2), 3 * O + 3 * c * x[2]^2, -O, O,
        cos(x[3]), Z, 3 * x[3]^2 + 2 * O, Z,
        Z, sin(x[4]), Z, 3 * x[4]^2 + 2 * O
    )
end

function f_elim!(F::Array, x::Array, p)
    T = eltype(x)
    c = p === nothing ? zero(T) : T(p)
    @inbounds begin
        F[1] = 2 * x[1] + c * x[1]^3 + T(1 // 2) * x[2] + sin(x[3]) - one(T)
        F[2] = T(1 // 2) * x[1] + 3 * x[2] + c * x[2]^3 - cos(x[4]) - 2 * one(T)
        F[3] = x[3]^3 + 2 * x[3] + x[1] - x[2] - T(1 // 2)
        F[4] = x[4]^3 + 2 * x[4] + x[2] - T(3 // 2)
    end
    return F
end

function f_elim!(F, x, p)
    T = eltype(x)
    c = p === nothing ? zero(T) : T(p)
    e1 = JLArray(T[1, 0, 0, 0])
    e2 = JLArray(T[0, 1, 0, 0])
    e3 = JLArray(T[0, 0, 1, 0])
    e4 = JLArray(T[0, 0, 0, 1])
    x1 = sum(x .* e1)
    x2 = sum(x .* e2)
    x3 = sum(x .* e3)
    x4 = sum(x .* e4)
    f1 = 2 * x1 + c * x1^3 + T(1 // 2) * x2 + sin(x3) - one(T)
    f2 = T(1 // 2) * x1 + 3 * x2 + c * x2^3 - cos(x4) - 2 * one(T)
    f3 = x3^3 + 2 * x3 + x1 - x2 - T(1 // 2)
    f4 = x4^3 + 2 * x4 + x2 - T(3 // 2)
    F .= f1 .* e1 .+ f2 .* e2 .+ f3 .* e3 .+ f4 .* e4
    return F
end

# Convex-with-quartic action for StationarityProblem block elimination.
function action_elim(x, p)
    T = eltype(x)
    return (x[1] - T(1 // 2) * x[3] - one(T))^2 +
           T(3 // 2) * (x[2] + T(1 // 3) * x[3] - T(1 // 2))^2 +
           (x[3] - T(3 // 4))^2 + T(1 // 4) * (x[3] - T(3 // 4))^4
end

function action_elim_r3(x::Array, p)
    T = eltype(x)
    @inbounds begin
        x1, x2, x3 = x[1], x[2], x[3]
        return (x1 - T(1 // 2) * x3 - one(T))^2 +
               T(3 // 2) * (x2 + T(1 // 3) * x3 - T(1 // 2))^2 +
               (x3 - T(3 // 4))^2 + T(1 // 4) * (x3 - T(3 // 4))^4
    end
end

# Separable exponential fit fixture for Check 1:
# r_i(y, z) = y_1 + y_2 exp(-z_1 t_i) + y_3 exp(-z_2 t_i) - d_i, m = 21 on [0, 2].
const T_VARPRO_21 = SVector{21, Float64}(range(0.0, 2.0; length = 21))
const X_VARPRO_TRUE = SVector{5, Float64}(0.5, 1.5, -1.0, 1.0, 2.0)
const D_VARPRO_21 = SVector{21, Float64}(
    ntuple(i -> 0.5 + 1.5 * exp(-1.0 * T_VARPRO_21[i]) - 1.0 * exp(-2.0 * T_VARPRO_21[i]),
    Val(21))
)

function r_exp5(x, p)
    T = eltype(x)
    y1, y2, y3, z1, z2 = x[1], x[2], x[3], x[4], x[5]
    return SVector{21, T}(ntuple(Val(21)) do i
        ti = T(T_VARPRO_21[i])
        di = T(D_VARPRO_21[i])
        y1 + y2 * exp(-z1 * ti) + y3 * exp(-z2 * ti) - di
    end)
end

# Compact 6-residual separable least-squares fixture (m = 6, n = 3: y = (x1, x2), z = (x3)).
# `p === :noisy` selects noisy target data (nonzero optimal residual); otherwise exact target data.
const T_LS6 = SVector{6, Float64}(0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
const D_LS6_EXACT = SVector{6, Float64}(ntuple(i -> 1.2 + 0.8 * exp(-1.5 * T_LS6[i]), Val(6)))
const D_LS6_NOISY = D_LS6_EXACT + SVector{6, Float64}(0.15, -0.10, 0.08, -0.12, 0.05, -0.06)

function r_ls6(x, p)
    T = eltype(x)
    d = p === :noisy ? D_LS6_NOISY : D_LS6_EXACT
    return SVector{6, T}(ntuple(i -> x[1] + x[2] * exp(-x[3] * T(T_LS6[i])) - T(d[i]), Val(6)))
end

function r_ls6!(r::Array, x::Array, p)
    T = eltype(x)
    d = p === :noisy ? D_LS6_NOISY : D_LS6_EXACT
    @inbounds for i in 1:6
        r[i] = x[1] + x[2] * exp(-x[3] * T(T_LS6[i])) - T(d[i])
    end
    return r
end

# Singular inner block fixtures for Check 4:
# `p === :dup` gives duplicate columns in J_yy (x1 + x2 = x3); otherwise a zero second column in J_yy.
function f_singular_inner(x, p)
    T = eltype(x)
    if p === :dup
        return SVector{3, T}(
            x[1] + x[2] - x[3],
            2 * (x[1] + x[2] - x[3]),
            x[3] + x[1] - 3 * one(T)
        )
    else
        return SVector{3, T}(
            2 * x[1] - x[3],
            zero(T) * x[2],
            x[3] + x[1] - 3 * one(T)
        )
    end
end

function f_singular_inner!(F::Array, x::Array, p)
    T = eltype(x)
    F[1] = 2 * x[1] - x[3]
    F[2] = zero(T)
    F[3] = x[3] + x[1] - 3 * one(T)
    return F
end

function r_vp_dead(x, p)
    T = eltype(x)
    return SVector{4, T}(
        x[1] + zero(T) * x[2] + x[3] - 3 * one(T),
        2 * x[1] + zero(T) * x[2] - x[3],
        x[1] + zero(T) * x[2] + 2 * x[3] - 5 * one(T),
        3 * x[1] + zero(T) * x[2] + x[3] - 5 * one(T)
    )
end

function r_vp_dead!(r::Array, x::Array, p)
    T = eltype(x)
    @inbounds begin
        r[1] = x[1] + zero(T) * x[2] + x[3] - 3 * one(T)
        r[2] = 2 * x[1] + zero(T) * x[2] - x[3]
        r[3] = x[1] + zero(T) * x[2] + 2 * x[3] - 5 * one(T)
        r[4] = 3 * x[1] + zero(T) * x[2] + x[3] - 5 * one(T)
    end
    return r
end

const T_VP_NL4 = SVector{4, Float64}(0.0, 0.5, 1.0, 1.5)
const Y_VP_NL4 = SVector{4, Float64}(ntuple(i -> 1.5^3 * exp(-0.8 * T_VP_NL4[i]), Val(4)))

function r_vp_nl(x, p)
    T = eltype(x)
    return SVector{4, T}(
        ntuple(i -> x[1]^3 * exp(-x[2] * T(T_VP_NL4[i])) - T(Y_VP_NL4[i]), Val(4))
    )
end

function r_vp_nl!(r::Array, x::Array, p)
    T = eltype(x)
    @inbounds for i in 1:4
        r[i] = x[1]^3 * exp(-x[2] * T(T_VP_NL4[i])) - T(Y_VP_NL4[i])
    end
    return r
end

# Zero-residual fit `r_i = y + 0.3 y³ + z t_i - d_i` with root (y, z) = (1.5, 0.7): nonlinear in y,
# with ∂r/∂y = 1 at y = 0, so one linear solve from y = 0 misses y*(z) without a rank-deficient Φ.
const T_VP_CUB5 = SVector{5, Float64}(0.0, 0.5, 1.0, 1.5, 2.0)
const D_VP_CUB5 = SVector{5, Float64}(ntuple(i -> 1.5 + 0.3 * 1.5^3 + 0.7 * T_VP_CUB5[i], Val(5)))

function r_vp_cub(x, p)
    T = eltype(x)
    return SVector{5, T}(ntuple(
        i -> x[1] + T(3 // 10) * x[1]^3 + x[2] * T(T_VP_CUB5[i]) - T(D_VP_CUB5[i]), Val(5)))
end

function r_vp_cub!(r::Array, x::Array, p)
    T = eltype(x)
    @inbounds for i in 1:5
        r[i] = x[1] + T(3 // 10) * x[1]^3 + x[2] * T(T_VP_CUB5[i]) - T(D_VP_CUB5[i])
    end
    return r
end

# Holomorphic complex fixture
function f_cplx_elim(x, p)
    T = eltype(x)
    return SVector{3, T}(
        T(2 + im) * x[1] + T(1 // 2) * x[2] - x[3]^2 - T(1 + im),
        T(1 // 3) * x[1] + T(3 - im) * x[2] + x[3] - T(2 - im),
        x[3]^2 + x[1] + x[2] - T(3 + 2im)
    )
end

function j_cplx_elim(x, p)
    T = eltype(x)
    return SMatrix{3, 3, T, 9}(
        T(2 + im), T(1 // 3), one(T),
        T(1 // 2), T(3 - im), one(T),
        -2 * x[3], one(T), 2 * x[3]
    )
end

@testset "BlockEliminated and VariableProjection construction, validation, and Adapt" begin
    @test Base.isexported(GS, :BlockEliminated)
    @test Base.isexported(GS, :VariableProjection)
    @test BlockEliminated <: AbstractNonlinearProblem
    @test VariableProjection <: AbstractNonlinearProblem

    prob_nl = NonlinearProblem(f_elim)
    elim_nl = BlockEliminated(prob_nl, (1, 2))
    @test elim_nl.prob === prob_nl
    @test elim_nl.inner == (1, 2)
    @test elim_nl.outer === nothing
    @test elim_nl.inner_eqs == (1, 2)
    @test elim_nl.outer_eqs === nothing
    @test elim_nl.inner_method isa Newton
    @test elim_nl.inner_method.linsolve isa SVDFactorization
    @test elim_nl.linear_inner === false
    @test elim_nl.inner_options === nothing
    @test propertynames(elim_nl) == (
        :prob, :inner, :outer, :inner_eqs, :outer_eqs,
        :inner_method, :linear_inner, :inner_options)
    @test isbits(elim_nl)

    elim_explicit = BlockEliminated(
        prob_nl, (1, 2);
        outer = (3, 4),
        inner_eqs = (2, 3),
        outer_eqs = (1, 4),
        linear_inner = true,
        inner_options = Options(Float64; f_abstol = 1e-10)
    )
    @test elim_explicit.outer == (3, 4)
    @test elim_explicit.inner_eqs == (2, 3)
    @test elim_explicit.outer_eqs == (1, 4)
    @test elim_explicit.linear_inner === true
    @test elim_explicit.inner_options isa Options{Float64}
    @test isbits(elim_explicit)

    prob_stat = StationarityProblem(action_elim)
    elim_stat = BlockEliminated(prob_stat, (1, 2); outer = (3,))
    @test elim_stat.prob === prob_stat
    @test isbits(elim_stat)

    prob_ls = LeastSquaresProblem(r_ls6, 6)
    vp = VariableProjection(prob_ls, (1, 2))
    @test vp.prob === prob_ls
    @test vp.inner == (1, 2)
    @test vp.outer === nothing
    @test vp.linsolve isa SVDFactorization
    @test vp.jacobian_mode === :kaufman
    @test vp.linear_inner === true
    @test vp.inner_method isa Newton
    @test vp.inner_options === nothing
    @test propertynames(vp) == (
        :prob, :inner, :outer, :linsolve, :jacobian_mode,
        :linear_inner, :inner_method, :inner_options)
    @test isbits(vp)

    # BlockEliminated on LeastSquaresProblem forwards to VariableProjection
    vp_fwd = BlockEliminated(prob_ls, (1, 2); outer = (3,), jacobian_mode = :golub_pereyra)
    @test vp_fwd isa VariableProjection
    @test vp_fwd.outer == (3,)
    @test vp_fwd.jacobian_mode === :golub_pereyra
    @test isbits(vp_fwd)

    # Partition and option validation at construction time
    @test_throws ArgumentError BlockEliminated(prob_nl, ())
    @test_throws ArgumentError BlockEliminated(prob_nl, Int[])
    @test_throws ArgumentError BlockEliminated(prob_nl, 1:0)
    @test_throws ArgumentError BlockEliminated(prob_nl, (0, 1))
    @test_throws ArgumentError BlockEliminated(prob_nl, (-1, 2))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 1))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); outer = ())
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); outer = (3, 3))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); outer = (2, 3))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); outer = (4, 5))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); inner_eqs = (1,))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); inner_eqs = (1, 1))
    @test_throws ArgumentError BlockEliminated(
        prob_nl, (1, 2); outer = (3, 4), inner_eqs = (1, 2), outer_eqs = (2, 3))
    @test_throws ArgumentError BlockEliminated(
        prob_nl, (1, 2); outer = (3, 4), inner_eqs = (1, 2), outer_eqs = (3,))
    @test_throws ArgumentError BlockEliminated(
        prob_nl, (1, 2); outer = (3, 4), inner_eqs = (1, 5))
    @test_throws ArgumentError BlockEliminated(prob_nl, (1, 2); inner_options = :invalid)

    @test_throws ArgumentError VariableProjection(prob_ls, ())
    @test_throws ArgumentError VariableProjection(prob_ls, (1, 1))
    @test_throws ArgumentError VariableProjection(prob_ls, (1, 2); outer = (2, 3))
    @test_throws ArgumentError VariableProjection(prob_ls, (1, 2); outer = (4,))
    @test_throws ArgumentError VariableProjection(prob_ls, (1, 2); jacobian_mode = :invalid)
    @test_throws ArgumentError VariableProjection(prob_ls, (1, 2); inner_options = :invalid)

    # Solve-time validation when x₀ length is incompatible with partition or non-vector
    @test_throws ArgumentError solve(elim_nl, SVector(1.0, 2.0), Newton())
    @test_throws ArgumentError solve(elim_explicit, SVector(1.0, 2.0, 3.0), Newton())
    @test_throws ArgumentError solve(
        BlockEliminated(prob_nl, (1, 5)), SVector(1.0, 2.0, 3.0, 4.0), Newton())
    @test_throws ArgumentError solve(elim_nl, 1.0, Newton())
    @test_throws ArgumentError solve(elim_nl, zeros(2, 2), Newton())
    @test_throws ArgumentError prepare_ad(AutoForwardDiff(), elim_nl, nothing, 1.0, 0.2)
    @test_throws ArgumentError prepare_ad(
        AutoForwardDiff(), elim_nl, zeros(2), zeros(2, 2), 0.2)
    @test_throws ArgumentError prepare_ad(AutoForwardDiff(), vp, nothing, 1.0, :noisy)
    @test_throws ArgumentError prepare_ad(
        AutoForwardDiff(), vp, zeros(6), zeros(2, 2), :noisy)

    # Dual-type rejection in init, solve, and solve!
    d1 = ForwardDiff.Dual{Nothing}(1.0, 1.0)
    x0_dual4 = SVector(d1, d1, d1, d1)
    x0_dual3 = SVector(d1, d1, d1)
    @test_throws ArgumentError init(elim_nl, x0_dual4, Newton(), 0.2)
    @test_throws ArgumentError init(elim_nl, SVector(0.2, 0.4, 0.5, 0.6), Newton(), d1)
    @test_throws ArgumentError init(vp, x0_dual3, GaussNewton())
    @test_throws ArgumentError init(vp, SVector(1.0, 0.5, 1.2), GaussNewton(), d1)

    elim_stat_v = BlockEliminated(prob_stat, (1, 2); outer = (3,))
    @test_throws ArgumentError init(
        elim_stat_v, SVector(0.2, 0.8, 0.25), Newton(); b = x0_dual3)
    @test_throws ArgumentError solve(
        elim_stat_v, SVector(0.2, 0.8, 0.25), Newton(); b = x0_dual3)
    solver_stat_v = init(elim_stat_v, SVector(0.2, 0.8, 0.25), Newton())
    @test_throws ArgumentError solve!(solver_stat_v; b = x0_dual3)

    # StationarityProblem E/U index validation and datum b length validation in R1 and R3
    prob_stat_bad_r1 = StationarityProblem(action_elim; U = (1, 2), E = (1, 4))
    elim_stat_bad_r1 = BlockEliminated(prob_stat_bad_r1, (1,); outer = (2,))
    @test_throws ArgumentError init(elim_stat_bad_r1, SVector(0.2, 0.8, 0.25), Newton())
    @test_throws ArgumentError solve(elim_stat_bad_r1, SVector(0.2, 0.8, 0.25), Newton())

    prob_stat_bad_r3 = StationarityProblem(action_elim; U = [1, 2], E = [1, 4])
    elim_stat_bad_r3 = BlockEliminated(prob_stat_bad_r3, [1]; outer = [2])
    @test_throws ArgumentError init(elim_stat_bad_r3, [0.2, 0.8, 0.25], Newton())
    @test_throws ArgumentError solve(elim_stat_bad_r3, [0.2, 0.8, 0.25], Newton())

    elim_stat_r3 = BlockEliminated(prob_stat, [1, 2]; outer = [3])
    @test_throws ArgumentError init(
        elim_stat_v, SVector(0.2, 0.8, 0.25), Newton(); b = [1.0, 2.0])
    @test_throws ArgumentError solve(
        elim_stat_v, SVector(0.2, 0.8, 0.25), Newton(); b = [1.0, 2.0])
    @test_throws ArgumentError init(
        elim_stat_r3, [0.2, 0.8, 0.25], Newton(); b = [1.0, 2.0])
    @test_throws ArgumentError solve(
        elim_stat_r3, [0.2, 0.8, 0.25], Newton(); b = [1.0, 2.0])

    # Adapt to Float32
    shift = 2.0
    prob_cap = NonlinearProblem((x, p) -> SVector(x[1] - shift, x[2] - shift))
    elim_cap = BlockEliminated(
        prob_cap, (1,); outer = (2,), inner_options = Options(Float64; f_abstol = 1e-5))
    elim_f32 = Adapt.adapt(ToReal{Float32}(), elim_cap)
    @test elim_f32 isa BlockEliminated
    @test elim_f32.inner_options isa Options{Float32}
    @test eltype(elim_f32.prob.F(SVector(1.0f0, 2.0f0), nothing)) === Float32

    ls_cap = LeastSquaresProblem((x, p) -> SVector(x[1] - shift, x[2] - shift, x[1] + x[2]), 3)
    vp_cap = VariableProjection(
        ls_cap, (1,); outer = (2,), inner_options = Options(Float64; f_abstol = 1e-5))
    vp_f32 = Adapt.adapt(ToReal{Float32}(), vp_cap)
    @test vp_f32 isa VariableProjection
    @test vp_f32.inner_options isa Options{Float32}
    @test eltype(vp_f32.prob.F(SVector(1.0f0, 2.0f0), nothing)) === Float32
end

@testset "Check 1: Separable exponential fit over 100 seeded starts" begin
    prob_exp5 = LeastSquaresProblem(r_exp5, 21)
    vp_kaufman = VariableProjection(prob_exp5, 1:3; outer = 4:5, jacobian_mode = :kaufman)
    vp_gp = VariableProjection(prob_exp5, 1:3; outer = 4:5, jacobian_mode = :golub_pereyra)
    m_lm = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
    opt = Options(
        Float64; max_iterations = 100, f_abstol = 1e-10, f_reltol = 1e-10, g_tol = 1e-10)

    rng = Xoshiro(42)
    starts = [SVector{5, Float64}(
                  10.0 * rand(rng) - 5.0,
                  10.0 * rand(rng) - 5.0,
                  10.0 * rand(rng) - 5.0,
                  0.2 + 3.3 * rand(rng),
                  0.2 + 3.3 * rand(rng)
              ) for _ in 1:100]

    n_joint = 0
    n_vp_kaufman = 0
    n_vp_gp = 0

    for x0 in starts
        x_j, st_j = solve(prob_exp5, x0, m_lm; options = opt)
        if st_j.code == SUCCESS && norm(r_exp5(x_j, nothing)) <= 1e-8
            n_joint += 1
        end

        x_vk, st_vk = solve(vp_kaufman, x0, m_lm; options = opt)
        if st_vk.code == SUCCESS && norm(r_exp5(x_vk, nothing)) <= 1e-8
            n_vp_kaufman += 1
            # Verify y*(z*) was written into x_vk[1:3] (catches MU1, MU4)
            @test norm(x_vk[SVector(1, 2, 3)] - x0[SVector(1, 2, 3)]) > 1e-3
        end

        x_vg, st_vg = solve(vp_gp, x0, m_lm; options = opt)
        if st_vg.code == SUCCESS && norm(r_exp5(x_vg, nothing)) <= 1e-8
            n_vp_gp += 1
            @test norm(x_vg[SVector(1, 2, 3)] - x0[SVector(1, 2, 3)]) > 1e-3
        end
    end

    @test n_vp_kaufman > n_joint
    @test n_vp_gp > n_joint
    @test n_vp_kaufman >= 85
    @test n_vp_gp >= 85
end

@testset "Check 2: Schur complement and reduced Jacobian vs central differences" begin
    allowscalar(false)
    x_test = SVector(0.3, -0.2, 0.6, -0.4)
    h_fd = 1e-6
    e1_2 = SVector(1.0, 0.0)
    e2_2 = SVector(0.0, 1.0)

    # 1. BlockEliminated on NonlinearProblem (linear_inner = true & false, E_y = Y & E_y != Y)
    for (prob_f, p_val, inner_eqs, outer_eqs, lin_in) in (
        (NonlinearProblem(f_elim), 0.0, (1, 2), (3, 4), true),
        (NonlinearProblem(f_elim), 0.0, (1, 2), (3, 4), false),
        (NonlinearProblem(f_elim, j_elim), 0.2, (1, 2), (3, 4), false),
        (NonlinearProblem(f_elim), 0.2, (1, 2), (3, 4), false),
        (NonlinearProblem(f_elim), 0.2, (2, 3), (1, 4), false)
    )
        elim = BlockEliminated(
            prob_f, (1, 2);
            outer = (3, 4),
            inner_eqs = inner_eqs,
            outer_eqs = outer_eqs,
            linear_inner = lin_in
        )
        S_analytic = jacobian!!(nothing, AutoForwardDiff(), elim, x_test, p_val)

        # Central difference of reduced residual wrt z = x[3:4]
        f_red_z(z) = residual!!(
            nothing, elim, SVector(x_test[1], x_test[2], z[1], z[2]), p_val)
        z0 = SVector(x_test[3], x_test[4])
        col1 = (f_red_z(z0 + h_fd * e1_2) - f_red_z(z0 - h_fd * e1_2)) / (2 * h_fd)
        col2 = (f_red_z(z0 + h_fd * e2_2) - f_red_z(z0 - h_fd * e2_2)) / (2 * h_fd)
        S_fd = hcat(col1, col2)

        @test norm(S_analytic - S_fd) <= 1e-6 * max(1.0, norm(S_fd))

        # The reduced Jacobian differs from the unreduced J_zz by O(1) (catches MU2); the
        # comparison with S_fd above fixes the sign of the J_zy J_yy^+ J_yz term
        J_full = jacobian!!(nothing, AutoForwardDiff(), prob_f, x_test, p_val)
        J_zz_raw = J_full[SVector{2, Int}(outer_eqs), SVector(3, 4)]
        @test norm(S_analytic - J_zz_raw) > 1e-2

        # jvp!! accepts both reduced-length and full-length tangent vectors v
        v_red = SVector(0.7, -0.3)
        v_full = SVector(9.0, -5.0, 0.7, -0.3)
        @test isapprox(jvp!!(nothing, AutoForwardDiff(), elim, x_test, v_red, p_val),
            S_analytic * v_red; rtol = 1e-12)
        @test isapprox(jvp!!(nothing, AutoForwardDiff(), elim, x_test, v_full, p_val),
            S_analytic * v_red; rtol = 1e-12)
    end

    # R3 Schur complement on Array (default and permuted equations) + jvp!!
    prob_r3 = NonlinearProblem(f_elim!)
    for (inner_eqs, outer_eqs) in (([1, 2], [3, 4]), ([2, 3], [1, 4]))
        elim_r3 = BlockEliminated(
            prob_r3, [1, 2];
            outer = [3, 4],
            inner_eqs = inner_eqs,
            outer_eqs = outer_eqs
        )
        x_a = Vector(x_test)
        Fz_a = zeros(2)
        Sz_a = zeros(2, 2)
        prep = prepare_ad(AutoForwardDiff(), elim_r3, Fz_a, x_a, 0.2)
        residual!!(Fz_a, prep, elim_r3, x_a, 0.2)
        jacobian!!(Sz_a, prep, elim_r3, x_a, 0.2)

        elim_r1 = BlockEliminated(
            NonlinearProblem(f_elim),
            (1, 2);
            outer = (3, 4),
            inner_eqs = Tuple(inner_eqs),
            outer_eqs = Tuple(outer_eqs)
        )
        Fz_r1 = residual!!(nothing, elim_r1, x_test, 0.2)
        Sz_r1 = jacobian!!(nothing, AutoForwardDiff(), elim_r1, x_test, 0.2)

        @test norm(Fz_a - Vector(Fz_r1)) <= 1e-10
        @test norm(Sz_a - Matrix(Sz_r1)) <= 1e-10

        Jv_a = zeros(2)
        jvp!!(Jv_a, prep, elim_r3, x_a, [0.7, -0.3], 0.2)
        @test isapprox(Jv_a, Sz_a * [0.7, -0.3]; rtol = 1e-10)
        jvp!!(Jv_a, prep, elim_r3, x_a, [9.0, -5.0, 0.7, -0.3], 0.2)
        @test isapprox(Jv_a, Sz_a * [0.7, -0.3]; rtol = 1e-10)
    end

    # Schur residual correction when F_y != 0 and R3 z_prev predictor buffer isolation
    f_schur_pred(x, _) = SVector(x[1] + x[2] - 3.0, 2.0 * x[1] + 3.0 * x[2] - 8.0)
    function f_schur_pred!(F, x, _)
        F[1] = x[1] + x[2] - 3.0
        F[2] = 2.0 * x[1] + 3.0 * x[2] - 8.0
        return F
    end
    in_opt_loose = Options(Float64; max_iterations = 1, f_abstol = 10.0, f_reltol = 10.0)
    out_opt_1step = Options(Float64; max_iterations = 1, f_abstol = 1e-12)
    elim_sp_r1 = BlockEliminated(
        NonlinearProblem(f_schur_pred), (1,);
        outer = (2,), linear_inner = false, inner_options = in_opt_loose
    )
    x_sp_r1, _ = solve(elim_sp_r1, SVector(0.0, 0.0), Newton(); options = out_opt_1step)
    @test isapprox(x_sp_r1[2], 2.0; atol = 1e-12)

    elim_sp_r3 = BlockEliminated(
        NonlinearProblem(f_schur_pred!), [1];
        outer = [2], linear_inner = false, inner_options = in_opt_loose
    )
    x_sp_r3, _ = solve(elim_sp_r3, [0.0, 0.0], Newton(); options = out_opt_1step)
    @test isapprox(x_sp_r3[2], 2.0; atol = 1e-12)

    st_sp_r3 = init_state(Newton(), elim_sp_r3, [0.0, 0.0], nothing)
    st_sp_r3 = start!!(st_sp_r3, Newton(), elim_sp_r3, [0.0, 0.0], [0.0], nothing, 1.0)
    copyto!(st_sp_r3.ws.z_prev, st_sp_r3.ws.z_curr)
    Jv_sp = zeros(1)
    jvp!!(Jv_sp, st_sp_r3.ws, elim_sp_r3, [0.0, 0.0], [0.5], nothing)
    @test st_sp_r3.ws.z_prev == st_sp_r3.ws.z_curr

    # 2. VariableProjection reduced Jacobian vs central differences (catches MU3)
    x_ls_test = SVector(1.0, 0.5, 1.2)
    z_ls0 = SVector(x_ls_test[3])
    e1_1 = SVector(1.0)

    # At zero residual (evaluating at the exact root z = 1.5 of r_ls6 with :exact), both :kaufman
    # and :golub_pereyra equal the exact derivative dr_red/dz
    prob_ls = LeastSquaresProblem(r_ls6, 6)
    x_ls_exact_z = SVector(0.0, 0.0, 1.5)
    for mode in (:kaufman, :golub_pereyra)
        vp_ex = VariableProjection(prob_ls, (1, 2); outer = (3,), jacobian_mode = mode)
        J_vp = jacobian!!(nothing, AutoForwardDiff(), vp_ex, x_ls_exact_z, :exact)
        r_red_ex(z) = residual!!(nothing, vp_ex, SVector(0.0, 0.0, z[1]), :exact)
        J_fd_ex = hcat((r_red_ex(SVector(1.5 + h_fd)) - r_red_ex(SVector(1.5 - h_fd))) /
                       (2 * h_fd))
        @test norm(J_vp - J_fd_ex) <= 1e-6 * max(1.0, norm(J_fd_ex))

        # Unprojected D(z) = J[:, Z] differs by O(1) (catches MU3a)
        x_full_ex = SVector(1.2, 0.8, 1.5)
        J_full_ex = jacobian!!(nothing, AutoForwardDiff(), prob_ls, x_full_ex, :exact)
        D_raw = J_full_ex[:, SVector(3)]
        @test norm(J_vp - D_raw) > 0.1

        # jvp!! on VariableProjection with reduced and full-length tangent vector v
        @test isapprox(
            jvp!!(nothing, AutoForwardDiff(), vp_ex, x_ls_exact_z, SVector(0.5), :exact),
            J_vp * SVector(0.5); rtol = 1e-12)
        @test isapprox(
            jvp!!(nothing, AutoForwardDiff(), vp_ex, x_ls_exact_z,
                SVector(2.0, -1.0, 0.5), :exact),
            J_vp * SVector(0.5); rtol = 1e-12)
    end

    # At nonzero residual (r_ls6 with :noisy at z = 1.2): :golub_pereyra matches central difference
    # to <= 1e-6 relative, while :kaufman omits the second-derivative term and differs by > 1e-2 (catches MU3b)
    vp_gp_n = VariableProjection(prob_ls, (1, 2); outer = (3,), jacobian_mode = :golub_pereyra)
    vp_kf_n = VariableProjection(prob_ls, (1, 2); outer = (3,), jacobian_mode = :kaufman)

    r_red_n(z) = residual!!(nothing, vp_gp_n, SVector(x_ls_test[1], x_ls_test[2], z[1]), :noisy)
    J_fd_n = hcat((r_red_n(z_ls0 + h_fd * e1_1) - r_red_n(z_ls0 - h_fd * e1_1)) /
                  (2 * h_fd))
    J_gp_n = jacobian!!(nothing, AutoForwardDiff(), vp_gp_n, x_ls_test, :noisy)
    J_kf_n = jacobian!!(nothing, AutoForwardDiff(), vp_kf_n, x_ls_test, :noisy)

    @test norm(J_gp_n - J_fd_n) <= 1e-6 * max(1.0, norm(J_fd_n))
    @test norm(J_kf_n - J_fd_n) > 1e-2

    # R3 VariableProjection reduced Jacobian on Array in both modes
    prob_ls_r3 = LeastSquaresProblem(r_ls6!, 6)
    for mode in (:kaufman, :golub_pereyra)
        vp_r3 = VariableProjection(prob_ls_r3, [1, 2]; outer = [3], jacobian_mode = mode)
        x_a = Vector(x_ls_test)
        r_a = zeros(6)
        J_a = zeros(6, 1)
        prep = prepare_ad(AutoForwardDiff(), vp_r3, r_a, x_a, :noisy)
        residual!!(r_a, prep, vp_r3, x_a, :noisy)
        jacobian!!(J_a, prep, vp_r3, x_a, :noisy)

        J_ref = mode === :golub_pereyra ? J_gp_n : J_kf_n
        @test norm(r_a - Vector(r_red_n(z_ls0))) <= 1e-12
        @test norm(J_a - Matrix(J_ref)) <= 1e-7 * max(1.0, norm(J_ref))

        Jv_vp = zeros(6)
        jvp!!(Jv_vp, prep, vp_r3, x_a, [0.5], :noisy)
        @test isapprox(Jv_vp, J_a * [0.5]; rtol = 1e-10)
        jvp!!(Jv_vp, prep, vp_r3, x_a, [2.0, -1.0, 0.5], :noisy)
        @test isapprox(Jv_vp, J_a * [0.5]; rtol = 1e-10)
    end
end

@testset "Check 3: Root equivalence in R1 and R3 ($T)" for T in REAL_ELTYPES
    allowscalar(false)
    tol = T === Float64 ? 1e-10 : 1e-4
    opt_t = Options(
        T; f_abstol = 16 * eps(T), f_reltol = 8 * eps(T),
        x_abstol = 16 * eps(T), x_reltol = 8 * eps(T)
    )
    x0 = SVector{4, T}(T(1 // 5), T(2 // 5), T(1 // 2), T(3 // 5))
    prob_j = NonlinearProblem(f_elim)
    prob_r3 = NonlinearProblem(f_elim!)

    # 1. BlockEliminated on NonlinearProblem (linear & nonlinear inner, default & permuted E_y)
    elim_cases = T === Float64 ?
                 (
        (zero(T), (1, 2), (3, 4), (false, true)),
        (T(1 // 5), (1, 2), (3, 4), (false,)),
        (T(1 // 5), (2, 3), (1, 4), (false,))
    ) : (
        (T(1 // 5), (1, 2), (3, 4), (false,)),
    )
    for (p_val, inner_eqs, outer_eqs, lin_modes) in elim_cases
        x_joint, st_joint = solve(prob_j, x0, Newton(), p_val; options = opt_t)
        @test st_joint.code == SUCCESS

        for lin_in in lin_modes
            elim = BlockEliminated(
                prob_j, (1, 2);
                outer = (3, 4),
                inner_eqs = inner_eqs,
                outer_eqs = outer_eqs,
                linear_inner = lin_in
            )
            x_elim, st_elim = solve(elim, x0, Newton(), p_val; options = opt_t)
            @test st_elim.code == SUCCESS
            @test norm(x_elim - x_joint) <= tol * max(one(T), norm(x_joint))
            @test norm(x_elim[SVector(1, 2)] - x0[SVector(1, 2)]) > T(1 // 20)

            if T === Float64
                # init and solve!
                solver = init(elim, x0, Newton(), p_val; options = opt_t)
                @test solver isa NonlinearSolver
                x_s1, st_s1 = solve!(solver)
                @test st_s1.code == SUCCESS
                @test norm(x_s1 - x_joint) <= tol * max(one(T), norm(x_joint))

                # Second solve!(solver) from converged root takes 0 outer iterations
                x_s2, st_s2 = solve!(solver)
                @test st_s2.code == SUCCESS
                @test st_s2.iterations == Int32(0)
                @test x_s2 == x_s1

                # 0-outer-iteration solve from z₀ = z* with un-eliminated y₀ != y*(z*) (catches MU4a in finish!!)
                x0_z_exact = SVector{4, T}(x0[1], x0[2], x_joint[3], x_joint[4])
                x_z0, st_z0 = solve(elim, x0_z_exact, Newton(), p_val; options = opt_t)
                @test st_z0.code == SUCCESS
                @test st_z0.iterations == Int32(0)
                @test norm(x_z0 - x_joint) <= tol * max(one(T), norm(x_joint))
                @test norm(x_z0[SVector(1, 2)] - x0_z_exact[SVector(1, 2)]) > T(1 // 20)

                # solve!! directly
                state0 = init_state(Newton(), elim, x0, p_val)
                x_s3, st_s3, _ = solve!!(state0, elim, x0, Newton(), p_val, opt_t)
                @test st_s3.code == SUCCESS
                @test norm(x_s3 - x_joint) <= tol * max(one(T), norm(x_joint))
            end

            # R3 on Array and JLArray
            elim_r3 = BlockEliminated(
                prob_r3, [1, 2];
                outer = [3, 4],
                inner_eqs = collect(inner_eqs),
                outer_eqs = collect(outer_eqs),
                linear_inner = lin_in
            )
            backends = (inner_eqs == (1, 2) && !lin_in && T === Float64) ? ARRAY_BACKENDS :
                       (Array,)
            for Arr in backends
                x0_r3 = Arr(Vector(x0))
                x_r3, st_r3 = solve(elim_r3, x0_r3, Newton(), p_val; options = opt_t)
                @test st_r3.code == SUCCESS
                @test Array(x0_r3) == Vector(x0)
                @test norm(Array(x_r3) - Vector(x_joint)) <=
                      tol * max(one(T), norm(x_joint))

                if Arr === Array && T === Float64
                    solver_r3 = init(elim_r3, x0_r3, Newton(), p_val; options = opt_t)
                    x_r3_i, st_r3_i = solve!(solver_r3)
                    @test st_r3_i.code == SUCCESS
                    @test x_r3_i === solver_r3.x
                    @test norm(Array(x_r3_i) - Vector(x_joint)) <=
                          tol * max(one(T), norm(x_joint))
                end
            end
        end
    end

    if T === Float64
        # 2. BlockEliminated on StationarityProblem (with and without datum b, plus solve!(solver; b = ...))
        prob_stat = StationarityProblem(action_elim)
        x0_stat = SVector{3, T}(T(1 // 5), T(4 // 5), T(1 // 4))
        b_vec = SVector{3, T}(T(1 // 10), -T(1 // 5), T(3 // 10))
        elim_stat = BlockEliminated(prob_stat, (1, 2); outer = (3,))
        elim_stat_r3 = BlockEliminated(prob_stat, [1, 2]; outer = [3])
        solver_stat = init(elim_stat, x0_stat, Newton(); options = opt_t)
        solver_stat_r3 = init(elim_stat_r3, Vector(x0_stat), Newton(); options = opt_t)

        for b_val in (nothing, b_vec)
            x_st_joint, st_st_joint = solve(
                prob_stat, x0_stat, Newton(); b = b_val, options = opt_t)
            @test st_st_joint.code == SUCCESS
            x_st_elim, st_st_elim = solve(
                elim_stat, x0_stat, Newton(); b = b_val, options = opt_t)
            @test st_st_elim.code == SUCCESS
            @test norm(x_st_elim - x_st_joint) <= tol * max(one(T), norm(x_st_joint))

            x_st_s, st_st_s = solve!(solver_stat, x0_stat, nothing; b = b_val)
            @test st_st_s.code == SUCCESS
            @test norm(x_st_s - x_st_joint) <= tol * max(one(T), norm(x_st_joint))

            b_r3 = b_val === nothing ? nothing : Vector(b_vec)
            x_st_r3, st_st_r3 = solve!(solver_stat_r3, Vector(x0_stat), nothing; b = b_r3)
            @test st_st_r3.code == SUCCESS
            @test norm(x_st_r3 - Vector(x_st_joint)) <= tol * max(one(T), norm(x_st_joint))
        end

        # Implicit differentiation through solve on BlockEliminated (R1 and R3)
        elim_imp_r1 = BlockEliminated(prob_j, (1, 2); outer = (3, 4))
        elim_imp_r3 = BlockEliminated(prob_r3, [1, 2]; outer = [3, 4])
        dx_dp_joint = ForwardDiff.derivative(
            p -> solve(prob_j, x0, Newton(), p; options = opt_t)[1], 0.2)
        dx_dp_r1 = ForwardDiff.derivative(
            p -> solve(elim_imp_r1, x0, Newton(), p; options = opt_t)[1], 0.2)
        dx_dp_r3 = ForwardDiff.derivative(
            p -> solve(elim_imp_r3, Vector(x0), Newton(), p; options = opt_t)[1], 0.2)
        @test isapprox(dx_dp_r1, dx_dp_joint; rtol = 1e-8)
        @test isapprox(dx_dp_r3, Vector(dx_dp_joint); rtol = 1e-8)
    end

    # 3. VariableProjection on LeastSquaresProblem (exact zero-residual and noisy nonzero-residual)
    x0_ls = SVector{3, T}(T(1 // 2), T(1 // 2), T(6 // 5))
    m_ls = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
    prob_ls = LeastSquaresProblem(r_ls6, 6)
    prob_ls_r3 = LeastSquaresProblem(r_ls6!, 6)
    ls_tags = T === Float64 ? (:exact, :noisy) : (:exact,)
    for tag in ls_tags
        x_ls_joint, st_ls_joint = solve(prob_ls, x0_ls, m_ls, tag; options = opt_t)
        @test st_ls_joint.code == SUCCESS

        lin_modes_vp = (T === Float64 && tag === :noisy) ? (true, false) : (true,)
        vp_modes = T === Float64 ? (:kaufman, :golub_pereyra) : (:kaufman,)
        for mode in vp_modes, lin_in in lin_modes_vp

            vp = VariableProjection(
                prob_ls, (1, 2); outer = (3,), jacobian_mode = mode, linear_inner = lin_in)
            x_vp, st_vp = solve(vp, x0_ls, m_ls, tag; options = opt_t)
            @test st_vp.code == SUCCESS
            tol_ls = tag === :noisy ? (T === Float64 ? T(1e-7) : T(5e-3)) : tol
            @test norm(x_vp - x_ls_joint) <= tol_ls * max(one(T), norm(x_ls_joint))
        end

        if T === Float64
            for mode in (:kaufman, :golub_pereyra)
                vp_r3 = VariableProjection(
                    prob_ls_r3, [1, 2]; outer = [3], jacobian_mode = mode)
                if tag === :exact
                    x_vp_r3, st_vp_r3 = solve(
                        vp_r3, Vector(x0_ls), m_ls, tag; options = opt_t)
                    @test st_vp_r3.code == SUCCESS
                    @test st_vp_r3.step_failures == 0
                    @test norm(x_vp_r3 - Vector(x_ls_joint)) <=
                          tol * max(one(T), norm(x_ls_joint))
                else
                    # at a nonzero residual the stationarity test sits at the noise floor of the
                    # reduced residual, so a start may end `STALLED`: assert a rate over 40 seeded
                    # starts within 1e-3 relative of `x0_ls`, and the distance of each `SUCCESS`
                    rng = Xoshiro(71)
                    n_success = 0
                    for _ in 1:40
                        x0_k = Vector(x0_ls) .* (1 .+ 1e-3 .* (2 .* rand(rng, 3) .- 1))
                        x_k, st_k = solve(vp_r3, x0_k, m_ls, tag; options = opt_t)
                        st_k.code == SUCCESS || continue
                        n_success += 1
                        @test norm(x_k - Vector(x_ls_joint)) <=
                              T(1e-7) * max(one(T), norm(x_ls_joint))
                    end
                    @test n_success >= 30
                end
            end
        end
    end
end

@testset "Holomorphic Complex root equivalence and Schur complement" begin
    T = ComplexF64
    R = real(T)
    tol = R(1e-10)
    opt_c = Options(T; f_abstol = 16 * eps(R), f_reltol = 8 * eps(R), g_tol = 16 * eps(R))
    prob_c = NonlinearProblem(f_cplx_elim, j_cplx_elim)
    x0_c = SVector{3, T}(T(0.5 + 0.2im), T(0.6 - 0.1im), T(1.2 + 0.4im))
    x_jc, st_jc = solve(prob_c, x0_c, Newton(); options = opt_c)
    @test st_jc.code == SUCCESS

    for lin_in in (true, false)
        elim_c = BlockEliminated(prob_c, (1, 2); outer = (3,), linear_inner = lin_in)
        x_ec, st_ec = solve(elim_c, x0_c, Newton(); options = opt_c)
        @test st_ec.code == SUCCESS
        @test norm(x_ec - x_jc) <= tol * max(one(R), norm(x_jc))
    end
end

@testset "Inner/outer contract: adaptive tolerance, 0-step warm start, and un-started step!!" begin
    # 1. Un-started step!! returns NONFINITE because init_state sets fnorm_z_prev = NaN (catches MU7)
    prob_nl = NonlinearProblem(f_elim)
    elim_nl = BlockEliminated(prob_nl, (1, 2); outer = (3, 4))
    x0 = SVector(0.2, 0.4, 0.5, 0.6)
    Fx0 = residual!!(nothing, elim_nl, x0, 0.2)

    st_r1 = init_state(Newton(), elim_nl, x0, 0.2)
    @test isnan(st_r1.fnorm_z_prev)
    @test all(isnan, st_r1.y_warm)
    x_un, Fx_un, _, info_un = step!!(st_r1, Newton(), elim_nl, x0, Fx0, 0.2)
    @test info_un.code == NONFINITE
    @test x_un == x0
    @test Fx_un == Fx0

    # After finish!!, fnorm_z_prev is reset to NaN so a subsequent step!! also returns NONFINITE
    st_started = start!!(st_r1, Newton(), elim_nl, x0, Fx0, 0.2, rnorm(Fx0))
    @test isfinite(st_started.fnorm_z_prev)
    @test all(isfinite, st_started.y_warm)
    _, _, st_finished = finish!!(
        st_started, Newton(), elim_nl, x0, Fx0,
        SolverStatus{Float64}(SUCCESS, Int32(1), 0.0, 0.0, Int32(0), false, false)
    )
    @test isnan(st_finished.fnorm_z_prev)
    _, _, _, info_after = step!!(st_finished, Newton(), elim_nl, x0, Fx0, 0.2)
    @test info_after.code == NONFINITE

    # Same un-started and post-finish!! check in R3
    prob_r3 = NonlinearProblem(f_elim!)
    elim_r3 = BlockEliminated(prob_r3, [1, 2]; outer = [3, 4])
    x0_r3 = Vector(x0)
    Fx0_r3 = zeros(2)
    st_r3 = init_state(Newton(), elim_r3, x0_r3, 0.2)
    @test isnan(st_r3.ws.fnorm_z_prev)
    @test all(isnan, st_r3.ws.y_warm)
    _, _, _, info_un_r3 = step!!(st_r3, Newton(), elim_r3, copy(x0_r3), copy(Fx0_r3), 0.2)
    @test info_un_r3.code == NONFINITE

    # 2. Adaptive inner tolerance scales as 0.5 * g2_prev^2 when inner_options === nothing at O(1) outer residual (catches MU4a)
    τ_at_o1, _ = GS._inner_tol_r1(nothing, Float64, 1.0, 0.8)
    @test isapprox(τ_at_o1, 0.5 * 0.8^2; rtol = 1e-12)
    τ_at_start, _ = GS._inner_tol_r1(nothing, Float64, 1.0, NaN)
    @test isapprox(τ_at_start, 10 * eps(Float64); rtol = 1e-12)

    _, _, _, _, iters_first_step = GS._eliminate_r1(
        elim_nl, st_r1.inner_state, SVector(0.2, 0.4), SVector(0.5, 0.6),
        0.2, 0.8, SVector(1, 2), SVector(3, 4), SVector(1, 2), SVector(3, 4)
    )
    _, _, _, _, iters_tight = GS._eliminate_r1(
        elim_nl, st_r1.inner_state, SVector(0.2, 0.4), SVector(0.5, 0.6),
        0.2, 1e-8, SVector(1, 2), SVector(3, 4), SVector(1, 2), SVector(3, 4)
    )
    @test iters_first_step <= 2
    @test iters_first_step < iters_tight

    # 3. Adaptive inner tolerance tightens as 0.5 * g2_prev^2 even when inner_options has a loose cap (catches MU4b)
    elim_loose = BlockEliminated(
        prob_nl, (1, 2);
        outer = (3, 4),
        linear_inner = false,
        inner_options = Options(Float64; f_abstol = 1e-1, f_reltol = 1e-1)
    )
    x_joint, _ = solve(prob_nl, x0, Newton(), 0.2)
    x_loose, st_loose = solve(elim_loose, x0, Newton(), 0.2)
    @test st_loose.code == SUCCESS
    @test norm(x_loose - x_joint) <= 1e-11
    @test norm(f_elim(x_loose, 0.2)) <= 1e-11

    elim_loose_r3 = BlockEliminated(
        prob_r3, [1, 2];
        outer = [3, 4],
        linear_inner = false,
        inner_options = Options(Float64; f_abstol = 1e-1, f_reltol = 1e-1)
    )
    x_loose_r3, st_loose_r3 = solve(elim_loose_r3, Vector(x0), Newton(), 0.2)
    @test st_loose_r3.code == SUCCESS
    @test norm(x_loose_r3 - Vector(x_joint)) <= 1e-11

    # 4. VariableProjection with linear_inner = false: adaptive tolerance, step_failures on non-convergence, and inner_method
    prob_vp_nl_r1 = LeastSquaresProblem(r_vp_nl, 4)
    prob_vp_nl_r3 = LeastSquaresProblem(r_vp_nl!, 4)
    m_gn_svd = GaussNewton(; linsolve = SVDFactorization())

    vp_fail_r1 = VariableProjection(
        prob_vp_nl_r1, (1,);
        outer = (2,),
        linear_inner = false,
        inner_method = Newton(; linsolve = QRFactorization()),
        inner_options = Options(
            Float64; max_iterations = 1, f_abstol = 1e-15, f_reltol = 1e-15)
    )
    _, st_vpf_r1 = solve(vp_fail_r1, SVector(10.0, 0.5), m_gn_svd; max_iterations = 2)
    @test st_vpf_r1.step_failures >= 1

    vp_fail_r3 = VariableProjection(
        prob_vp_nl_r3, [1];
        outer = [2],
        linear_inner = false,
        inner_method = Newton(; linsolve = QRFactorization()),
        inner_options = Options(
            Float64; max_iterations = 1, f_abstol = 1e-15, f_reltol = 1e-15)
    )
    _, st_vpf_r3 = solve(vp_fail_r3, [10.0, 0.5], m_gn_svd; max_iterations = 2)
    @test st_vpf_r3.step_failures >= 1

    vp_loose_r1 = VariableProjection(
        prob_vp_nl_r1, (1,);
        outer = (2,),
        linear_inner = false,
        inner_method = Newton(; linsolve = QRFactorization()),
        inner_options = Options(Float64; f_abstol = 1e-1, f_reltol = 1e-1)
    )
    x_vpl_r1, st_vpl_r1 = solve(vp_loose_r1, SVector(2.0, 0.5), m_gn_svd)
    @test st_vpl_r1.code == SUCCESS
    @test norm(x_vpl_r1 - SVector(1.5, 0.8)) <= 1e-8

    vp_loose_r3 = VariableProjection(
        prob_vp_nl_r3, [1];
        outer = [2],
        linear_inner = false,
        inner_method = Newton(; linsolve = QRFactorization()),
        inner_options = Options(Float64; f_abstol = 1e-1, f_reltol = 1e-1)
    )
    x_vpl_r3, st_vpl_r3 = solve(vp_loose_r3, [2.0, 0.5], m_gn_svd)
    @test st_vpl_r3.code == SUCCESS
    @test norm(x_vpl_r3 - [1.5, 0.8]) <= 1e-8

    # 5. 0-step warm start on repeat evaluation at the same z, and warm-start tracking (catches MU6)
    st_r3_warm = init_state(Newton(), elim_r3, copy(x0_r3), 0.2)
    st_r3_warm = start!!(st_r3_warm, Newton(), elim_r3, copy(x0_r3), copy(Fx0_r3), 0.2, 1.0)
    @test st_r3_warm.ws.last_inner_iters >= 1
    # Re-evaluating at the same z0 with y_warm == y*(z0) takes 0 inner steps:
    GS._eliminate_r3!(st_r3_warm.ws, elim_r3, st_r3_warm.ws.z_curr, 0.2)
    @test st_r3_warm.ws.last_inner_iters == 0

    # In R1, a repeat call to _eliminate_r1 from y_warm = y*(z0) at z0 also takes 0 inner steps:
    _, _, _, _, iters_r1_repeat = GS._eliminate_r1(
        elim_nl, st_started.inner_state, st_started.y_warm, st_started.z_curr,
        0.2, st_started.fnorm_z_prev, st_started.Y, st_started.Z,
        st_started.E_y, st_started.E_z
    )
    @test iters_r1_repeat == 0

    # 6. VariableProjection delegates Dual solve_implicit to LeastSquaresProblem ArgumentError
    d_tag = ForwardDiff.Dual{Nothing}(0.2, 1.0)
    @test_throws ArgumentError solve(vp_loose_r1, SVector(2.0, 0.5), m_gn_svd, d_tag)
    @test_throws ArgumentError solve(vp_loose_r3, [2.0, 0.5], m_gn_svd, d_tag)
    v_full = SVector(0.1, 0.2, 0.3, 0.4)
    @test jvp!!(nothing, nothing, elim_nl, x0, v_full, 0.2) ≈
          jvp!!(nothing, AutoForwardDiff(), elim_nl, x0, v_full[SVector(3, 4)], 0.2)
end

@testset "Check 4: Singular J_yy dead unit minimum-norm step and step_failures" begin
    allowscalar(false)

    # 1. Dead unit and duplicate columns in R1 and R3 with both linear_inner = false (default) and true (catches MU5, MU5a, MU5b)
    prob_sing = NonlinearProblem(f_singular_inner)
    prob_dead_r3 = NonlinearProblem(f_singular_inner!)
    for lin_in in (false, true)
        elim_dead = BlockEliminated(prob_sing, (1, 2); outer = (3,), linear_inner = lin_in)
        for z_init in (0.5, 2.0)
            x_dead, st_dead = solve(elim_dead, SVector(3.0, 7.0, z_init), Newton(), :dead)
            @test st_dead.code == SUCCESS
            @test isapprox(x_dead[1], 1.0; atol = 1e-10)
            @test abs(x_dead[2]) <= 1e-10
            @test isapprox(x_dead[3], 2.0; atol = 1e-10)
            @test st_dead.step_failures >= 1
        end

        # Duplicate columns in J_yy: x1 + x2 = x3, so minimum-norm solution has x1 == x2 == x3 / 2
        x_dup, st_dup = solve(elim_dead, SVector(5.0, -3.0, 0.5), Newton(), :dup)
        @test st_dup.code == SUCCESS
        @test isapprox(x_dup[1], 1.0; atol = 1e-10)
        @test isapprox(x_dup[2], 1.0; atol = 1e-10)
        @test isapprox(x_dup[3], 2.0; atol = 1e-10)
        @test st_dup.step_failures >= 1

        # R3 dead unit on Array
        elim_dead_r3 = BlockEliminated(
            prob_dead_r3, [1, 2]; outer = [3], linear_inner = lin_in)
        for z_init in (0.5, 2.0)
            x_d_h, st_d_r3 = solve(elim_dead_r3, [3.0, 7.0, z_init], Newton())
            @test st_d_r3.code == SUCCESS
            @test isapprox(x_d_h[1], 1.0; atol = 1e-10)
            @test abs(x_d_h[2]) <= 1e-10
            @test isapprox(x_d_h[3], 2.0; atol = 1e-10)
            @test st_d_r3.step_failures >= 1
        end
    end

    # 2. Dead unit in VariableProjection in R1 and R3 with linear_inner = true and false (catches MU5c)
    prob_vp_dead_r1 = LeastSquaresProblem(r_vp_dead, 4)
    prob_vp_dead_r3 = LeastSquaresProblem(r_vp_dead!, 4)
    m_gn_svd = GaussNewton(; linsolve = SVDFactorization())
    for lin_in in (true, false)
        vp_dead = VariableProjection(
            prob_vp_dead_r1, (1, 2); outer = (3,), linear_inner = lin_in)
        x_vpd, st_vpd = solve(vp_dead, SVector(4.0, 9.0, 0.5), m_gn_svd)
        @test st_vpd.code == SUCCESS
        @test isapprox(x_vpd[1], 1.0; atol = 1e-10)
        @test abs(x_vpd[2]) <= 1e-10
        @test isapprox(x_vpd[3], 2.0; atol = 1e-10)
        @test st_vpd.step_failures >= 1

        vp_dead_r3 = VariableProjection(
            prob_vp_dead_r3, [1, 2]; outer = [3], linear_inner = lin_in)
        x_vpd_r3, st_vpd_r3 = solve(vp_dead_r3, [4.0, 9.0, 0.5], m_gn_svd)
        @test st_vpd_r3.code == SUCCESS
        @test isapprox(x_vpd_r3[1], 1.0; atol = 1e-10)
        @test abs(x_vpd_r3[2]) <= 1e-10
        @test isapprox(x_vpd_r3[3], 2.0; atol = 1e-10)
        @test st_vpd_r3.step_failures >= 1
    end

    # 3. Inner nonlinear solver failure (max_iterations = 1 from a distant guess) increments step_failures
    elim_fail_in = BlockEliminated(
        NonlinearProblem(f_elim), (1, 2);
        outer = (3, 4),
        linear_inner = false,
        inner_options = Options(Float64; max_iterations = 1, f_abstol = 1e-15, f_reltol = 1e-15)
    )
    _, st_fail_in = solve(
        elim_fail_in, SVector(10.0, -10.0, 0.5, 0.6), Newton(), 0.2; max_iterations = 2)
    @test st_fail_in.step_failures >= 1

    # 4. linear_inner = true on an inner block that is nonlinear in y: the linear solve returns a
    # point that is no root, the missed inner residual increments step_failures, and the solve goes
    # on; an affine inner block, and linear_inner = false, count no failure
    x0_lin = SVector(0.2, 0.4, 0.5, 0.6)
    for (c, lin_in) in ((0.2, true), (0.2, false), (nothing, true))
        miss = c !== nothing && lin_in
        elim_lin = BlockEliminated(
            NonlinearProblem(f_elim), (1, 2); outer = (3, 4), linear_inner = lin_in)
        x_lin, st_lin = solve(elim_lin, x0_lin, Newton(), c)
        elim_lin_r3 = BlockEliminated(
            NonlinearProblem(f_elim!), [1, 2]; outer = [3, 4], linear_inner = lin_in)
        x_lin_r3, st_lin_r3 = solve(elim_lin_r3, Vector(x0_lin), Newton(), c)
        for (x_l, st_l) in ((x_lin, st_lin), (SVector{4}(x_lin_r3), st_lin_r3))
            @test st_l.code == SUCCESS
            if miss
                @test st_l.step_failures >= 1
                @test norm(f_elim(x_l, c)) > 1e-2
            else
                @test st_l.step_failures == 0
                @test norm(f_elim(x_l, c)) <= 1e-10
            end
        end
    end

    m_gn_cub = GaussNewton(; linsolve = SVDFactorization())
    for lin_in in (true, false)
        vp_cub = VariableProjection(
            LeastSquaresProblem(r_vp_cub, 5), (1,); outer = (2,), linear_inner = lin_in)
        x_cub, st_cub = solve(vp_cub, SVector(1.0, 0.5), m_gn_cub)
        vp_cub_r3 = VariableProjection(
            LeastSquaresProblem(r_vp_cub!, 5), [1]; outer = [2], linear_inner = lin_in)
        x_cub_r3, st_cub_r3 = solve(vp_cub_r3, [1.0, 0.5], m_gn_cub)
        for (x_c, st_c) in ((x_cub, st_cub), (SVector{2}(x_cub_r3), st_cub_r3))
            @test st_c.code == SUCCESS
            if lin_in
                @test st_c.step_failures >= 1
                @test norm(r_vp_cub(x_c, nothing)) > 1
            else
                @test st_c.step_failures == 0
                @test norm(r_vp_cub(x_c, nothing)) <= 1e-10
            end
        end
    end
end

@testset "Inference, isbits, and zero allocations" begin
    x0 = SVector(0.2, 0.4, 0.5, 0.6)
    prob_nl = NonlinearProblem(f_elim)
    elim_lin = BlockEliminated(prob_nl, (1, 2); outer = (3, 4), linear_inner = true)
    elim_nl = BlockEliminated(prob_nl, (1, 2); outer = (3, 4), linear_inner = false)
    elim_def = BlockEliminated(prob_nl, (1, 2); linear_inner = true)

    prob_ls = LeastSquaresProblem(r_ls6, 6)
    x0_ls = SVector(1.0, 0.5, 1.2)
    vp_kf = VariableProjection(prob_ls, (1, 2); outer = (3,), jacobian_mode = :kaufman)
    vp_gp = VariableProjection(prob_ls, (1, 2); outer = (3,), jacobian_mode = :golub_pereyra)
    m_gn = GaussNewton(; linsolve = SVDFactorization())

    @test isbits(elim_lin)
    @test isbits(elim_nl)
    @test isbits(vp_kf)
    @test isbits(vp_gp)
    @test isbits(init_state(Newton(), elim_lin, x0, 0.0))
    @test isbits(init_state(Newton(), elim_nl, x0, 0.2))
    @test isbits(init_state(m_gn, vp_kf, x0_ls, :noisy))
    @test isbits(init_state(m_gn, vp_gp, x0_ls, :noisy))

    @test (@inferred solve(elim_lin, x0, Newton(), 0.0))[2].code == SUCCESS
    @test (@inferred solve(elim_nl, x0, Newton(), 0.2))[2].code == SUCCESS
    @test (@inferred solve(elim_def, x0, Newton(), 0.0))[2].code == SUCCESS
    @test (@inferred solve(vp_kf, x0_ls, m_gn, :noisy))[2].code == SUCCESS
    @test (@inferred solve(vp_gp, x0_ls, m_gn, :noisy))[2].code == SUCCESS

    elim_r3_opt = BlockEliminated(
        NonlinearProblem(f_elim!), [1, 2]; outer = [3, 4], linear_inner = false)
    vp_r3_opt = VariableProjection(
        LeastSquaresProblem(r_ls6!, 6), [1, 2]; outer = [3], jacobian_mode = :kaufman)
    solver_r3_opt = init(elim_r3_opt, [0.2, 0.4, 0.5, 0.6], Newton(), 0.2)
    solver_vp_r3_opt = init(vp_r3_opt, [1.0, 0.5, 1.2], m_gn, :exact)
    JET.@test_opt target_modules = (GeometricSolvers,) solve(elim_lin, x0, Newton(), 0.0)
    JET.@test_opt target_modules = (GeometricSolvers,) solve(elim_nl, x0, Newton(), 0.2)
    JET.@test_opt target_modules = (GeometricSolvers,) solve(vp_kf, x0_ls, m_gn, :noisy)
    JET.@test_opt target_modules = (GeometricSolvers,) solve(vp_gp, x0_ls, m_gn, :noisy)
    JET.@test_opt target_modules = (GeometricSolvers,) solve!(
        solver_r3_opt, [0.2, 0.4, 0.5, 0.6], 0.2)
    JET.@test_opt target_modules = (GeometricSolvers,) solve!(
        solver_vp_r3_opt, [1.0, 0.5, 1.2], :exact)

    if Base.JLOptions().check_bounds != 1
        # skipped where a run forces `--check-bounds=yes`
        @test allocations(collect, x0) > 0
        @test allocations(a -> solve(elim_lin, a, Newton(), 0.0), x0) == 0
        @test allocations(a -> solve(elim_nl, a, Newton(), 0.2), x0) == 0
        @test allocations(a -> solve(elim_def, a, Newton(), 0.0), x0) == 0
        @test allocations(a -> solve(vp_kf, a, m_gn, :noisy), x0_ls) == 0
        @test allocations(a -> solve(vp_gp, a, m_gn, :noisy), x0_ls) == 0

        for (lin_in, p_val) in ((true, 0.0), (false, 0.2))
            prob_r3_wrap = BlockEliminated(
                NonlinearProblem(f_elim!), [1, 2]; outer = [3, 4], linear_inner = lin_in)
            x0_r3 = [0.2, 0.4, 0.5, 0.6]
            solver_r3 = init(prob_r3_wrap, copy(x0_r3), Newton(), p_val)
            x_in = copy(x0_r3)
            solve!(solver_r3, copyto!(x_in, x0_r3), p_val)
            @test allocations(a -> solve!(solver_r3, copyto!(a, x0_r3), p_val), x_in) == 0
        end

        for mode in (:kaufman, :golub_pereyra)
            vp_r3 = VariableProjection(
                LeastSquaresProblem(r_ls6!, 6), [1, 2]; outer = [3], jacobian_mode = mode)
            x0_ls_r3 = [1.0, 0.5, 1.2]
            solver_vp_r3 = init(vp_r3, copy(x0_ls_r3), m_gn, :exact)
            x_in_ls = copy(x0_ls_r3)
            solve!(solver_vp_r3, copyto!(x_in_ls, x0_ls_r3), :exact)
            @test allocations(
                a -> solve!(solver_vp_r3, copyto!(a, x0_ls_r3), :exact), x_in_ls) == 0
        end
    end
end

# ─── Reduced-space elimination: an R2 batch inside the outer residual ───────────────────────────

# One matrix-free outer Newton on a one-dimensional implicit PIC field, the method every check of
# this section solves with.
R5() = Newton(; linsolve = GMRES(; maxiter = 16), globalization = Backtracking())

# The batch of `Check 2`: `y[1]^2 = c[1]`, `y[2]^2 = c[2]`, so the parameter `(-1, 3)` of system 7
# has no real root and its inner solve cannot converge.
function failing_batch(::Type{T}, n = 16) where {T}
    prob = NonlinearProblem((y, c) -> SVector(y[1]^2 - c[1], y[2]^2 - c[2]))
    Y = [SVector{2, T}(T(12 // 10), T(18 // 10)) for _ in 1:n]
    ps = [i == 7 ? SVector{2, T}(T(-1), T(3)) : SVector{2, T}(T(2), T(3)) for i in 1:n]
    return prob, Y, ps
end

@testset "Batch reduction: BatchSummary, batch_summary, and BatchParams" begin
    ok_status(its) = SolverStatus{Float64}(SUCCESS, Int32(its), 0.0, 0.0, Int32(0), false)

    @testset "The reduction of a status vector" begin
        S = [ok_status(i) for i in 1:4]
        S[1] = SolverStatus{Float64}(SUCCESS, Int32(1), 3.0, 0.0, Int32(0), false)
        S[2] = SolverStatus{Float64}(SUCCESS, Int32(2), 4.0, 0.0, Int32(0), false)
        s = batch_summary(S)
        @test s isa BatchSummary{Float64}
        # the global norm is the 2-norm of the per-system norms, not their sum or their maximum
        @test s.fnorm == 5.0
        @test s.all_converged
        @test s.failure_count == 0
        @test s.max_iterations == Int32(4)
        @test s.total_iterations == 1 + 2 + 3 + 4
        empty_summary = batch_summary(SolverStatus{Float64}[])
        @test empty_summary.all_converged && empty_summary.fnorm == 0.0
        @test empty_summary.max_iterations == Int32(0) &&
              empty_summary.total_iterations == 0

        F = copy(S)
        F[3] = SolverStatus{Float64}(STALLED, Int32(8), 1.0, 0.0, Int32(0), false)
        f = batch_summary(F)
        @test !f.all_converged && f.failure_count == 1
        @test f.max_iterations == Int32(8)
        @test f.fnorm == sqrt(3.0^2 + 4.0^2 + 1.0^2 + 0.0^2)

        for B in ARRAY_BACKENDS
            @testset "backend $B" begin
                @test batch_summary(B(S)).fnorm == 5.0
                @test batch_summary(B(F)).failure_count == 1
            end
        end
    end

    @testset "BatchParams passes the shared value and the per-system value" begin
        shared = SVector(1.5, 2.5)
        per_system = [SVector(i, -i) for i in 1:4]
        p = BatchParams(shared, per_system)
        @test GS.parameters(p, 3) === (shared, per_system[3])
        @test GS.parameters(shared, 3) === shared
        @test Adapt.adapt(Array, p).per_system == per_system

        # a system sees both halves: `y = a + b` with `a` shared and `b` per system
        prob = NonlinearProblem((y, c) -> y .- (c[1] .+ c[2]))
        for B in ARRAY_BACKENDS
            @testset "backend $B" begin
                X = [zero(SVector{2, Float64}) for _ in 1:4]
                X = B(X)
                S = B(similar(X, SolverStatus{Float64}))
                solve_batch!(X, S, prob, Newton(), BatchParams(shared, B(per_system)))
                @test all(st -> st.code == SUCCESS, Array(S))
                @test Array(X)[3] == shared .+ per_system[3]
            end
        end
        @test_throws DimensionMismatch solve_batch!(
            [zero(SVector{2, Float64}) for _ in 1:4], similar(
                [zero(SVector{2, Float64}) for _ in 1:4], SolverStatus{Float64}),
            prob, Newton(), BatchParams(shared, per_system[1:2]))
        @test_throws ArgumentError solve_batch!(
            [zero(SVector{2, Float64}) for _ in 1:4], similar(
                [zero(SVector{2, Float64}) for _ in 1:4], SolverStatus{Float64}),
            prob, Newton(), BatchParams(shared, per_system); workgroupsize = 0)
        # the positional form of the options, which the checks call
        X = [zero(SVector{2, Float64}) for _ in 1:4]
        counts = solve_batch!(
            X, prob, Newton(), BatchParams(shared, per_system),
            Options(Float64; max_iterations = 4))
        @test counts.SUCCESS == 4
    end
end

@testset "Check 2: one failing particle of the batch" begin
    for T in REAL_ELTYPES, B in ARRAY_BACKENDS

        @testset "$T, backend $B" begin
            inner, Y_batch, params_vec = failing_batch(T)
            Y_batch = B(Y_batch)
            S = B(similar(Y_batch, SolverStatus{T}))
            solve_batch!(Y_batch, S, inner, Newton(), PerSystem(B(params_vec)),
                Options(T; max_iterations = 8))
            summary = batch_summary(S)
            host = reduce(vcat, [Vector(inner.F(Array(Y_batch)[i], params_vec[i]))
                                 for i in 1:length(Y_batch)])
            @test !summary.all_converged
            @test summary.failure_count == 1
            @test isapprox(summary.fnorm, norm(host); rtol = (2 * length(Y_batch)) * eps(T))
        end
    end
end

@testset "PIC1D: gather, deposit, and the outer residual, $T_in in Float64" for T_in in REAL_ELTYPES
    T_out = Float64
    prob, E0 = GS.make_pic_elimination_problem(T_in, T_out; N_x = 16, N_p = 64)
    @test length(E0) == 16 && length(prob.Y) == 64
    @test eltype(prob.Y) == SVector{2, T_in}
    @test eliminated_indices(prob) == Tuple(17:80)
    @test length(prob.statuses) == 64
    @test_throws ArgumentError GS.make_pic_elimination_problem(T_in, T_out; N_x = 2)
    # the batch's time step is finite in the batch's own type: `Δt = Inf` passes `> 0`, and a
    # `Float64` step that overflows `T_in` is `Inf` once converted
    @test_throws ArgumentError GS.make_pic_elimination_problem(T_in, T_out; Δt = Inf)
    @test_throws ArgumentError GS.make_pic_elimination_problem(T_in, T_out; Δt = NaN)
    @test_throws ArgumentError GS.make_pic_elimination_problem(T_in, T_out; Δt = -1)
    if T_in === Float32
        # a `Float64` step that overflows the batch's type is `Inf32` there
        @test_throws ArgumentError GS.make_pic_elimination_problem(T_in, T_out; Δt = 1e300)
    end

    # the gather reproduces a constant field, and its derivative is continuous across a cell edge
    Δx = T_in(2) * T_in(π) / T_in(16)
    E = fill(T_in(1 // 4), 16)
    for ξ in range(zero(T_in), Δx * T_in(15); length = 33)
        value, _ = GS.pic_gather(E, ξ, Δx, 16)
        @test value ≈ 1 // 4
    end
    edge = T_in === Float32 ? 8 * eps(T_in) : T_in(1e-9)
    _, slope_lo = GS.pic_gather(E, Δx - edge, Δx, 16)
    _, slope_hi = GS.pic_gather(E, Δx + edge, Δx, 16)
    @test slope_lo ≈ slope_hi atol = 1e-5

    # the deposit fills all four moments, and each is non-zero on this distribution
    reset_batch!(prob)
    F = similar(E0)
    residual!!(F, prob, E0, nothing)
    m = deposited_moments(prob)
    @test m === prob.moments
    @test all(!all(iszero, v) for v in (m.current, m.density, m.flux, m.temperature))
    # the S₂ weights sum to one, so a deposit of the whole batch reproduces `N_p` up to the
    # accumulated round-off of the batch's own type
    @test sum(m.density) ≈ 64 rtol = T_in === Float32 ? 2e-4 : 1e-12
    @test length(m.temperature) == 16

    # the analytic inner Jacobian matches the central difference of the inner residual
    h = T_in(1 // 10)
    grid = fill(T_in(3 // 10), 16)
    ps = GS.PIC1DParams(T_in(11 // 10), T_in(4 // 10), T_in(2 // 10))
    c = (grid, ps)
    y = SVector(T_in(115 // 100), T_in(45 // 100))
    J = GS.pic_inner_jacobian(y, c, h, Δx, 16)
    δ = sqrt(eps(T_in))
    for j in 1:2
        e = SVector(j == 1 ? δ : zero(δ), j == 2 ? δ : zero(δ))
        fd = (GS.pic_inner_residual(y + e, c, h, Δx, 16) .-
              GS.pic_inner_residual(y - e, c, h, Δx, 16)) ./ (2δ)
        @test J[:, j] ≈ fd rtol = T_in === Float32 ? 2e-3 : 1e-6
    end

    # one particle on a node deposits the S₂ weights on that node and its two neighbours, in the
    # canonical order; every other node stays empty, and an empty node's temperature is zero
    m = DepositedMoments(
        zeros(T_out, 16), zeros(T_out, 16), zeros(T_out, 16), zeros(T_out, 16))
    x_node = T_in(4) * Δx                 # node 5, whose position is `4 Δx`
    single = [SVector(x_node, T_in(2))]
    single_ps = [GS.PIC1DParams(x_node, T_in(2), zero(T_in))]
    GS.deposit_pic_moments!(m, single, zeros(T_in, 16), single_ps, h, Δx, 16)
    @test m.density[5] ≈ 3 / 4
    @test m.density[4] ≈ 1 / 8
    @test m.density[6] ≈ 1 / 8
    @test isapprox(sum(m.density), 1.0; atol = T_in === Float32 ? 2e-7 : 1e-15)
    @test m.current[5] ≈ 3 / 4 * 2.0
    @test m.flux[5] ≈ 3 / 4 * 4.0
    @test m.temperature[5] == 0            # one particle has no velocity spread
    for j in (1, 2, 3, 7, 16)
        @test m.density[j] == 0
        @test m.temperature[j] == 0
    end
    @test all(isfinite, m.temperature)

    # a position no cell holds — non-finite, or a cell count beyond the `Int` range — is a
    # non-finite moment, and not an `InexactError` from inside the gather or the deposit, which
    # no outer solve can report
    for ξ in (T_in(NaN), T_in(Inf), -T_in(Inf), T_in(1e20) * Δx)
        j, t = GS._pic_node(ξ, Δx, 16)
        @test 1 <= j <= 16
        @test isnan(t)
        value, slope = GS.pic_gather(E, ξ, Δx, 16)
        @test isnan(value) && isnan(slope)
    end
    hostile = DepositedMoments(
        zeros(T_out, 16), zeros(T_out, 16), zeros(T_out, 16), zeros(T_out, 16))
    GS.deposit_pic_moments!(
        hostile, [SVector(T_in(NaN), T_in(1))], zeros(T_in, 16), nothing, h, Δx, 16)
    @test !all(isfinite, hostile.density)
    @test !all(isfinite, hostile.current)
    @test !all(isfinite, hostile.flux)

    # the deposit accumulates in the canonical order of the batch, so the association of these
    # three velocities at one node is the batch's index order: the pairwise cancellation leaves the
    # third velocity's contribution, and any other order loses it
    mo = DepositedMoments(
        zeros(T_out, 16), zeros(T_out, 16), zeros(T_out, 16), zeros(T_out, 16))
    cancel = [SVector(x_node, T_in(1e16)), SVector(x_node, -T_in(1e16)),
        SVector(x_node, one(T_in))]
    cancel_ps = [GS.PIC1DParams(x_node, zero(T_in), zero(T_in)) for _ in 1:3]
    GS.deposit_pic_moments!(mo, cancel, zeros(T_in, 16), cancel_ps, h, Δx, 16)
    @test mo.current[5] == 3 / 4          # (3/4·1e16 − 3/4·1e16) + 3/4·1, and not 1.0
    @test mo.density[5] == 3 * 3 / 4

    # the inner options of a batch are absolute only, so a repeat evaluation stops at 0 iterations
    options = GS._batch_inner_options(prob, T_in(1e-6))
    @test options.f_abstol == T_in(1e-6)
    @test options.f_reltol == 0

    # the default eliminated set is the batch's own index block
    inner = NonlinearProblem((y, c) -> y .- c)
    blocks = BatchEliminated((F, x, m, s) -> F, inner, [SVector(0.0) for _ in 1:3],
        (m, Y, s, ps) -> m, DepositedMoments(zeros(2), zeros(2), zeros(2), zeros(2)))
    @test eliminated_indices(blocks) == (1, 2, 3)
end

@testset "Check 3: the deposit is deterministic, $T_in in Float64" for T_in in REAL_ELTYPES
    prob, E0 = GS.make_pic_elimination_problem(T_in, Float64; N_x = 16, N_p = 64)
    F = similar(E0)
    reset_batch!(prob)
    residual!!(F, prob, E0, nothing)
    m0 = deposited_moments(prob)
    reference = (copy(m0.current), copy(m0.density), copy(m0.flux), copy(m0.temperature))
    for _ in 1:100
        reset_batch!(prob)
        residual!!(F, prob, E0, nothing)
        mk = deposited_moments(prob)
        @test mk.current == reference[1]
        @test mk.density == reference[2]
        @test mk.flux == reference[3]
        @test mk.temperature == reference[4]
    end
    # a deposit that writes fewer than four moments, or in another order, changes these: the four
    # are the moments of the same batch state
    @test reference[1] != reference[2] && reference[3] != reference[4]
end

@testset "Check 4: the warm start reports 0 inner iterations on a repeat" begin
    T_out = Float64
    for T_in in REAL_ELTYPES
        @testset "$T_in in $T_out" begin
            prob, E0 = GS.make_pic_elimination_problem(T_in, T_out; N_x = 16, N_p = 64)
            F = similar(E0)
            reset_batch!(prob)
            # the tolerance of an evaluation is the schedule's, not the cap: at `E0` the cap is
            # sixteen times the schedule, so a repeat at the cap reports 0 iterations as well
            first_tol = GS._batch_inner_tolerance(prob)
            @test first_tol < prob.inner_atol_cap
            residual!!(F, prob, E0, nothing)
            @test batch_summary(prob).total_iterations > 0
            @test prob.last_inner_tol == first_tol
            # one evaluation on, the schedule is narrower still, and the repeat re-uses it
            scheduled = GS._batch_inner_tolerance(prob)
            @test scheduled < prob.inner_atol_cap
            residual!!(F, prob, E0, nothing)
            @test prob.last_inner_tol == scheduled
            first_F = copy(F)
            residual!!(F, prob, E0, nothing; repeat_same_tol = true)
            summary = batch_summary(prob)
            @test prob.last_inner_tol == scheduled     # identical across the two calls
            @test summary.max_iterations == 0
            @test summary.total_iterations == 0
            @test all(st -> st.iterations == 0, prob.statuses)
            @test F == first_F

            # a product does not disturb the warm start: after one it reports 0 iterations too
            prep = GS.prepare_jvp(AutoForwardDiff(), prob, F, E0, nothing)
            jv = similar(F)
            GS.jvp!!(jv, prep, prob, E0, one.(E0), nothing)
            residual!!(F, prob, E0, nothing; repeat_same_tol = true)
            @test batch_summary(prob).total_iterations == 0
        end
    end
end

@testset "Check 1: the adaptive quadratic inner tolerance, $T_in in Float64" for T_in in REAL_ELTYPES
    prob, E0 = GS.make_pic_elimination_problem(T_in, Float64; N_x = 16, N_p = 64)
    # the schedule: the floor `2 eps(R_in)` of the batch's own precision, and the quadratic term
    # `min(cap, g2^2)/2/sqrt(N_p)` of the outer residual above it
    exact = T_in === Float32 ? 1e-5 : 1e-12
    batch_floor = 2 * eps(T_in)
    cap_tol = 0.5 * prob.inner_atol_cap / sqrt(64)
    prob.g₂_prev = 0.05
    t_half = GS._batch_inner_tolerance(prob)
    prob.g₂_prev = 0.025
    t_quarter = GS._batch_inner_tolerance(prob)
    @test t_half > batch_floor
    @test t_quarter ≈ t_half / 4 rtol = exact        # quadratic in the outer residual
    @test t_quarter ≈ 0.5 * 0.025^2 / sqrt(64) rtol = exact
    prob.g₂_prev = 0.5
    @test GS._batch_inner_tolerance(prob) ≈ cap_tol rtol = exact
    prob.g₂_prev = 1e-12
    @test GS._batch_inner_tolerance(prob) == batch_floor   # floored at the batch's own round-off
    prob.g₂_prev = Inf
    @test GS._batch_inner_tolerance(prob) ≈ cap_tol rtol = exact
    prob.g₂_prev = NaN                  # an evaluation with no residual narrows nothing
    @test GS._batch_inner_tolerance(prob) ≈ cap_tol rtol = exact

    # and through a solve: `‖g₁‖ < ‖g₂(prev)‖²` strictly at every recorded evaluation where the
    # floor does not bind, and under the coarse ceiling where it does
    reset_batch!(prob)
    tol = T_in === Float32 ? 2e-4 : 1e-8
    _, status = solve(prob, copy(E0), R5(); f_abstol = tol, f_reltol = 0.0)
    @test status.code == SUCCESS
    @test status.f_calls > 0
    @test status.jvp_calls > 0
    @test length(prob.g₁_history) == length(prob.g₂_prev_history) ==
          length(prob.eliminated_index_log)
    @test length(prob.g₁_history) >= 2
    @test prob.g₂_prev_history[1] == Inf
    loose = collect(zip(prob.g₁_history, prob.g₂_prev_history))
    # The ceiling branch runs only on a record where `g₂² ≤ 4 eps(R_in) √N_p`, and the converging
    # solve has no such record: all of its evaluations sit above the floor. The solve Check 6 makes
    # at `1e-10` does reach the floored region (52 of 55 records at `Float32`, one of six at
    # `Float64`), so the testset solves at that tolerance too, and counts the records in the region
    # before it bounds one of them.
    reset_batch!(prob)
    solve(prob, copy(E0), R5(); f_abstol = 1e-10, f_reltol = 0.0, max_iterations = 25)
    tight = collect(zip(prob.g₁_history, prob.g₂_prev_history))
    @test length(tight) >= 2
    # the floor binds where `min(cap, g₂²)/2/√N_p` is below `2 eps(R_in)`, below `g₂² = 4 eps √N_p`
    n_p = length(prob.Y)
    bound = 4 * eps(T_in) * sqrt(n_p)
    ceiling = 32 * eps(T_in) * sqrt(n_p)
    @test count(p -> !(p[2]^2 > bound), tight) > 0
    for pairs in (loose, tight)
        for (g1, g2) in pairs
            if g2^2 > bound
                @test g1 < g2^2
            else
                @test g1 <= ceiling
            end
        end
        # and where the floor does not bind, the recorded inner residual is under the schedule
        for k in 2:length(pairs)
            g2 = pairs[k][2]
            if g2^2 > bound
                @test pairs[k][1] <= 0.5 * min(prob.inner_atol_cap, g2^2) * (1 + 1e-6)
            end
        end
    end
end

@testset "The ceiling of a failed batch: NaN above it, the deposit below it" begin
    # A batch that did not converge has no outer residual when its norm is above
    # `32 eps(R_in) √N_p`, and `NaN` is what the evaluation writes; at or below the ceiling the batch
    # stopped at the floor of its own arithmetic and its deposit is the residual to that accuracy.
    # The inner residual here is the constant `tiny`, which no step can change: every system fails
    # with the same norm, the batch's is `√N_p · tiny`, and the two cases sit at 14 times the
    # ceiling and at 0.56 of it.
    ceiling = 32 * eps(Float64) * sqrt(8)
    for (tiny, nan_expected) in ((1e-13, true), (4e-15, false))
        inner = NonlinearProblem((y, c) -> SVector(tiny, zero(tiny)))
        p = BatchEliminated(
            (F, x, m, s) -> (F .= x), inner, [SVector(0.5, 0.5) for _ in 1:8],
            (m, Y, s, ps) -> fill!(m.current, 1.0),
            DepositedMoments(zeros(4), zeros(4), zeros(4), zeros(4));
            inner_atol_cap = 1e-16)
        reset_batch!(p)
        F = zeros(4)
        residual!!(F, p, ones(4), nothing)
        summary = batch_summary(p)
        @test !summary.all_converged
        @test summary.failure_count == 8
        ratio = summary.fnorm / ceiling
        # the crossing case is above the ceiling and far below a thousand times it: a ceiling that
        # loose reports the failed batch as one stopped at its floor, and keeps its deposit
        @test nan_expected ? 1 < ratio < 1000 : ratio < 1
        @test nan_expected ? all(isnan, F) : all(isfinite, F)
    end
end

@testset "Check 5: the eliminated index set is fixed over a solve, $T_in in Float64" for T_in in REAL_ELTYPES
    prob, E0 = GS.make_pic_elimination_problem(T_in, Float64; N_x = 16, N_p = 64)
    fixed = eliminated_indices(prob)
    @test fixed == Tuple(17:80)
    reset_batch!(prob)
    @test isempty(prob.eliminated_index_log)
    solve(prob, copy(E0), R5(); f_abstol = T_in === Float32 ? 2e-4 : 1e-8, f_reltol = 0.0)
    @test !isempty(prob.eliminated_index_log)
    @test all(==(fixed), prob.eliminated_index_log)
    @test eliminated_indices(prob) == fixed
end

@testset "Check 6: a Float32 batch inside a Float64 outer Newton" begin
    prob, E0 = GS.make_pic_elimination_problem(Float32, Float64; N_x = 16, N_p = 64)
    reset_batch!(prob)
    _, status = solve(prob, copy(E0), R5(); f_abstol = 2e-4, f_reltol = 0.0)
    @test status.code == SUCCESS
    @test status.fnorm <= 2e-4
    reset_batch!(prob)
    _, stalled = solve(prob, copy(E0), R5(); f_abstol = 1e-10, f_reltol = 0.0,
        max_iterations = 25)
    @test stalled.code == STALLED
    @test stalled.code != SINGULAR
    @test stalled.fnorm > 1e-10

    # the product's step is the batch's own: `√eps(Float32)`, not `√eps(Float64)`, which the
    # plateau of a `Float32` residual swallows
    F = similar(E0)
    reset_batch!(prob)
    residual!!(F, prob, E0, nothing)
    # the whole residual is evaluated in the batch's precision, its linear term and its sum too,
    # so every value it writes is a `Float32` value whatever the outer iterate's type
    @test all(v -> Float32(v) == v, F)
    prep = GS.prepare_jvp(AutoForwardDiff(), prob, F, E0, nothing)
    @test prep.ε ≈ sqrt(eps(Float32))
    @test prep.tol == 2 * eps(Float32)
    jv = similar(F)
    v = copy(E0) ./ norm(E0)
    GS.jvp!!(jv, prep, prob, E0, v, nothing)
    @test !iszero(norm(jv))
    # against a central difference at a step the plateau cannot swallow, both evaluations at the
    # batch's floor so that the difference is the field's response and not the tolerance's
    far = 1e-2
    prob.last_inner_tol = 2 * eps(Float32)
    GS.residual!!(F, prob, E0 .+ far .* v, nothing; repeat_same_tol = true)
    Fp = copy(F)
    GS.residual!!(F, prob, E0 .- far .* v, nothing; repeat_same_tol = true)
    central = (Fp .- F) ./ (2 * far)
    @test isapprox(jv, central; rtol = 1e-2, atol = 1e-6 * norm(central))
end

@testset "Reduced-space elimination: construction, records, and hostile input" begin
    prob, E0 = GS.make_pic_elimination_problem(Float64, Float64; N_x = 8, N_p = 16)
    @test prob.workgroupsize == 64
    @test prob.inner_atol_cap == 0.01
    @test prob.g₂_prev == Inf
    @test isempty(prob.g₁_history) && isempty(prob.g₂_prev_history)
    @test prob.Y !== prob.Y_init && prob.Y_init == prob.Y_anchor
    # the initial guess is not mutated by the constructor or by a solve
    Y₀ = copy(prob.Y)
    reset_batch!(prob)
    @test prob.Y == Y₀
    residual!!(similar(E0), prob, E0, nothing)
    @test prob.outer_evaluations == 1
    reset_batch!(prob)
    @test prob.outer_evaluations == 0 && prob.g₂_prev == Inf
    @test all(st -> st.iterations == 0, prob.statuses)

    # a batch is solved by a matrix-free outer Newton only
    @test_throws ArgumentError solve(prob, copy(E0), Newton())
    # whose product is the batch's own difference: another AD back end is refused by name, not
    # left to the DifferentiationInterface preparation, which reads a field the batch does not have
    @test_throws ArgumentError solve(prob, copy(E0),
        Newton(; jacobian = GS.AutoFiniteDiff(), linsolve = GMRES(; maxiter = 16)))
    # a wrong inner options precision, an empty batch, a bad cap, a bad index set
    moments = DepositedMoments(zeros(2), zeros(2), zeros(2), zeros(2))
    inner = NonlinearProblem((y, c) -> y .- c)
    @test_throws ArgumentError BatchEliminated(
        (F, x, m, s) -> F, inner, [SVector(0.0)], (m, Y, s, ps) -> m, moments;
        inner_options = Options(Float32))
    @test_throws ArgumentError BatchEliminated(
        (F, x, m, s) -> F, inner, SVector{0, SVector{1, Float64}}(), (m, Y, s, ps) -> m,
        moments)
    @test_throws ArgumentError BatchEliminated(
        (F, x, m, s) -> F, inner, [SVector(0.0)], (m, Y, s, ps) -> m, moments;
        inner_atol_cap = -1.0)
    @test_throws ArgumentError BatchEliminated(
        (F, x, m, s) -> F, inner, [SVector(0.0)], (m, Y, s, ps) -> m, moments;
        eliminated_indices = (1, 1))
    @test_throws DimensionMismatch BatchEliminated(
        (F, x, m, s) -> F, inner, [SVector(0.0), SVector(1.0)], (m, Y, s, ps) -> m, moments;
        per_system = [1.0])
    @test_throws ArgumentError BatchEliminated(
        (F, x, m, s) -> F, inner, [SVector(0.0)], (m, Y, s, ps) -> m, moments;
        workgroupsize = 0)
    # the cap is finite: `Inf` and `NaN` are not a tolerance
    for bad_cap in (Inf, NaN)
        @test_throws ArgumentError BatchEliminated(
            (F, x, m, s) -> F, inner, [SVector(0.0)], (m, Y, s, ps) -> m, moments;
            inner_atol_cap = bad_cap)
    end

    # the workgroup size reaches the launch: the CPU device refuses one above its 4096
    # work-items, which a launch that dropped the keyword accepts
    @test GS._validate_workgroupsize(5000) == 5000
    @test_throws ArgumentError solve_batch!(
        [zero(SVector{2, Float64}) for _ in 1:4],
        similar([zero(SVector{2, Float64}) for _ in 1:4], SolverStatus{Float64}),
        NonlinearProblem((y, c) -> y .- c), Newton(),
        PerSystem([SVector(1.0, 2.0) for _ in 1:4]); workgroupsize = 5000)
    # every `solve_batch!` form takes it: each refuses a zero, which the backend would accept, and
    # the reparametrized and stationarity forms run a launch at a real size. The stationarity
    # objective is a scalar function of the pair `(x, p)`, whose gradient is the residual that the
    # batch of `SVector` roots solves: the residual evaluates it at a `Dual` field with a second
    # argument, so a one-argument `x -> ...` has no stationarity state for the kernel to compile.
    stat = GS.StationarityProblem((x, p) -> sum(abs2, x) / 2 - sum(x))
    Xs = [zero(SVector{2, Float64}) for _ in 1:4]
    Ss = similar(Xs, SolverStatus{Float64})
    rep = GS.Reparametrized(NonlinearProblem((x, p) -> x .- 1.0), (y, p) -> 2 * y, nothing)
    @test solve_batch!(Xs, rep, Newton(); workgroupsize = 2).SUCCESS == 4
    @test all(st -> st.code == SUCCESS,
        solve_batch!(Xs, Ss, rep, Newton(); workgroupsize = 2))
    @test solve_batch!(Xs, stat, Newton(); workgroupsize = 2).SUCCESS == 4
    @test all(st -> st.code == SUCCESS,
        solve_batch!(Xs, Ss, stat, Newton(); workgroupsize = 2))
    for f in (X -> solve_batch!(X, stat, Newton(); workgroupsize = 0),
        X -> solve_batch!(X, Ss, stat, Newton(); workgroupsize = 0),
        X -> solve_batch!(X, rep, Newton(); workgroupsize = 0),
        X -> solve_batch!(X, Ss, rep, Newton(); workgroupsize = 0))
        @test_throws ArgumentError f(Xs)
    end
    # the refusal comes before the return of an empty batch, as it does in the plain forms
    X_empty = eltype(Xs)[]
    S_empty = similar(X_empty, SolverStatus{Float64})
    for f in (() -> solve_batch!(X_empty, stat, Newton(); workgroupsize = 0),
        () -> solve_batch!(X_empty, S_empty, stat, Newton(); workgroupsize = 0),
        () -> solve_batch!(X_empty, rep, Newton(); workgroupsize = 0),
        () -> solve_batch!(X_empty, S_empty, rep, Newton(); workgroupsize = 0))
        @test_throws ArgumentError f()
    end
    # and the size it validated reaches the launch on every form: a launch that dropped the keyword
    # accepts 5000, which the CPU device refuses
    @test_throws ArgumentError solve_batch!(Xs, rep, Newton(); workgroupsize = 5000)
    @test_throws ArgumentError solve_batch!(
        copy(Xs), Ss, rep, Newton(); workgroupsize = 5000)
    @test_throws ArgumentError solve_batch!(Xs, stat, Newton(); workgroupsize = 5000)
    @test_throws ArgumentError solve_batch!(
        copy(Xs), Ss, stat, Newton(); workgroupsize = 5000)

    # an outer iterate of another length is refused, not resized into an `@inbounds` read past
    # the deposited moments and the shared field
    for n in (7, 9)
        reset_batch!(prob)
        x_bad = n < 8 ? E0[1:n] : vcat(E0, zeros(n - 8))
        @test_throws DimensionMismatch residual!!(zeros(n), prob, x_bad, nothing)
        @test_throws DimensionMismatch solve(prob, x_bad, R5())
    end
    @test isempty(prob.g₁_history)        # a refused evaluation records nothing

    # a Jacobian–vector product solves into a status slot of its own, so it leaves the statuses
    # of the anchor evaluation as they were
    reset_batch!(prob)
    Fr = similar(E0)
    residual!!(Fr, prob, E0, nothing)
    anchor_statuses = copy(prob.statuses)
    anchor_summary = batch_summary(prob)
    prep = GS.prepare_jvp(AutoForwardDiff(), prob, Fr, E0, nothing)
    GS.jvp!!(similar(Fr), prep, prob, E0, one.(E0), nothing)
    @test prob.statuses == anchor_statuses
    @test batch_summary(prob) == anchor_summary

    # an inner batch that fails leaves no outer residual: the deposit of a batch that did not
    # converge is not the value of the reduced residual. Its seventh system has no real root, and
    # the analytic Jacobian keeps the derivative out of the batch kernel; `c[2]` is the system's
    # parameter, the second half of what `BatchParams` passes
    inner_fail = NonlinearProblem(
        (y, c) -> SVector(y[1]^2 - c[2][1], y[2]^2 - c[2][2]),
        (y, c) -> SMatrix{2, 2}(2 * y[1], zero(y[1]), zero(y[1]), 2 * y[2]))
    Y_fail = [SVector(1.2, 1.8) for _ in 1:8]
    ps_fail = [i == 7 ? SVector(-1.0, 3.0) : SVector(2.0, 3.0) for i in 1:8]
    fail = BatchEliminated(
        (F, x, m, s) -> (F .= x), inner_fail, Y_fail,
        (m, Y, s, ps) -> fill!(m.current, 1.0),
        DepositedMoments(zeros(4), zeros(4), zeros(4), zeros(4));
        per_system = ps_fail, inner_atol_cap = 1e-6)
    reset_batch!(fail)
    Ffail = zeros(4)
    residual!!(Ffail, fail, ones(4), nothing)
    @test !batch_summary(fail).all_converged
    @test !all(isfinite, Ffail)
    @test !isfinite(fail.g₂_prev)
    _, fail_status = solve(fail, ones(4), R5())
    @test fail_status.code == NONFINITE

    # a particle at a position no cell holds is a non-finite residual and a status, not an
    # `InexactError` from inside the kernel
    nan_prob, nan_E0 = GS.make_pic_elimination_problem(Float64, Float64; N_x = 8, N_p = 16)
    nan_prob.Y[1] = SVector(NaN, 0.4)
    nan_prob.Y_init[1] = SVector(NaN, 0.4)
    reset_batch!(nan_prob)
    Fnan = similar(nan_E0)
    residual!!(Fnan, nan_prob, nan_E0, nothing)
    @test !all(isfinite, Fnan)
    @test !batch_summary(nan_prob).all_converged
    _, nan_status = solve(nan_prob, copy(nan_E0), R5())
    @test nan_status.code == NONFINITE

    # a counted residual adds to the `f_calls` of the workspace, so the trial points of a line
    # search reach the status: both of the residual's call forms count
    ws = GS.init_gmres_workspace(GMRES(; maxiter = 16), E0)
    ws.f_calls = Int32(0)
    counted = GS.counting_problem(prob, ws).F
    Fc = similar(E0)
    counted(Fc, E0, nothing)
    @test ws.f_calls == Int32(1)
    counted(E0, nothing)
    @test ws.f_calls == Int32(2)
end

@testset "Reduced-space elimination: inference, allocations, and JET" begin
    prob, E0 = GS.make_pic_elimination_problem(Float64, Float64; N_x = 8, N_p = 16)
    F = similar(E0)
    reset_batch!(prob)
    residual!!(F, prob, E0, nothing)
    S = prob.statuses
    h = 0.1
    Δx = 2π / 8
    grid = fill(0.3, 8)
    ps = GS.PIC1DParams(1.1, 0.4, 0.2)
    c = (grid, ps)
    y = SVector(1.15, 0.45)
    @test (@inferred GS.pic_inner_residual(y, c, h, Δx, 8)) isa SVector{2, Float64}
    @test (@inferred GS.pic_inner_jacobian(y, c, h, Δx, 8)) isa SMatrix{2, 2, Float64, 4}
    @test (@inferred batch_summary(S)) isa BatchSummary{Float64}
    @test (@inferred GS._batch_inner_tolerance(prob)) isa Float64
    JET.@test_opt target_modules = (GeometricSolvers,) GS.pic_inner_residual(y, c, h, Δx, 8)
    JET.@test_opt target_modules = (GeometricSolvers,) batch_summary(S)
    JET.@test_opt target_modules = (GeometricSolvers,) GS._batch_inner_tolerance(prob)
    # the barrier's argument has a concrete type and the call is `SMatrix` arithmetic, so these
    # hold under `--check-bounds=yes` as well
    @test allocations(collect, y) > 0
    inner_at(yy) = GS.pic_inner_residual(yy, c, h, Δx, 8)
    @test allocations(inner_at, y) == 0
    gather(ξ) = GS.pic_gather(grid, ξ, Δx, 8)
    @test allocations(gather, 1.1) == 0
end
