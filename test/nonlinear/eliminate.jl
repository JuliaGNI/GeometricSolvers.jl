using Adapt: Adapt
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: I, norm, pinv
using Random: Xoshiro
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: AbstractNonlinearProblem, AutoForwardDiff, Backtracking,
                        BlockEliminated, DogLeg, GaussNewton, LDLTFactorization,
                        LUFactorization, LeastSquaresProblem, LevenbergMarquardt, MAXITERS,
                        NONFINITE, Newton, NonlinearProblem, NonlinearSolver, Options,
                        Picard, QRFactorization, STALLED, SUCCESS, SVDFactorization,
                        SolverStatus, StationarityProblem, Static, ToReal,
                        VariableProjection, finish!!, init, init_state, jacobian!!,
                        prepare_ad, residual!!, rnorm, solve, solve!, solve!!,
                        start!!, step!!

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

    # Solve-time validation when x₀ length is incompatible with partition
    @test_throws ArgumentError solve(elim_nl, SVector(1.0, 2.0), Newton())
    @test_throws ArgumentError solve(elim_explicit, SVector(1.0, 2.0, 3.0), Newton())
    @test_throws ArgumentError solve(
        BlockEliminated(prob_nl, (1, 5)), SVector(1.0, 2.0, 3.0, 4.0), Newton())
    @test_throws ArgumentError solve(elim_nl, 1.0, Newton())

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
    end

    @test n_vp_kaufman > n_joint
    @test n_vp_kaufman >= 85
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

        # Unreduced J_zz and wrong-sign J_zz + J_zy J_yy^+ J_yz differ from S_fd (catches MU2)
        J_full = jacobian!!(nothing, AutoForwardDiff(), prob_f, x_test, p_val)
        J_zz_raw = J_full[SVector{2, Int}(outer_eqs), SVector(3, 4)]
        @test norm(S_analytic - J_zz_raw) > 1e-2
    end

    # R3 Schur complement on Array (default and permuted equations)
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
    end

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

                if Arr === Array
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
        # 2. BlockEliminated on StationarityProblem (with and without datum b)
        prob_stat = StationarityProblem(action_elim)
        x0_stat = SVector{3, T}(T(1 // 5), T(4 // 5), T(1 // 4))
        b_vec = SVector{3, T}(T(1 // 10), -T(1 // 5), T(3 // 10))
        for b_val in (nothing, b_vec)
            x_st_joint, st_st_joint = solve(
                prob_stat, x0_stat, Newton(); b = b_val, options = opt_t)
            @test st_st_joint.code == SUCCESS
            elim_stat = BlockEliminated(prob_stat, (1, 2); outer = (3,))
            x_st_elim, st_st_elim = solve(
                elim_stat, x0_stat, Newton(); b = b_val, options = opt_t)
            @test st_st_elim.code == SUCCESS
            @test norm(x_st_elim - x_st_joint) <= tol * max(one(T), norm(x_st_joint))
        end
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

        r3_modes = (T === Float64 && tag === :noisy) ? (:kaufman, :golub_pereyra) :
                   (:kaufman,)
        for mode in r3_modes
            vp_r3 = VariableProjection(prob_ls_r3, [1, 2]; outer = [3], jacobian_mode = mode)
            x_vp_r3, st_vp_r3 = solve(vp_r3, Vector(x0_ls), m_ls, tag; options = opt_t)
            @test st_vp_r3.code == SUCCESS
            tol_ls = tag === :noisy ? (T === Float64 ? T(1e-7) : T(5e-3)) : tol
            @test norm(x_vp_r3 - Vector(x_ls_joint)) <=
                  tol_ls * max(one(T), norm(x_ls_joint))
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

    # 2. Adaptive inner tolerance tightens as 0.5 * g2_prev^2 even when inner_options has a loose cap (catches MU6a)
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

    # 3. 0-step warm start on repeat evaluation at the same z, and warm-start tracking (catches MU6b)
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
end

@testset "Check 4: Singular J_yy dead unit minimum-norm step and step_failures" begin
    allowscalar(false)

    # 1. Dead unit (zero second column in J_yy) in R1 and R3 (catches MU5a and MU5b)
    prob_sing = NonlinearProblem(f_singular_inner)
    elim_dead = BlockEliminated(prob_sing, (1, 2); outer = (3,), linear_inner = true)
    x0_dead = SVector(3.0, 7.0, 0.5)
    x_dead, st_dead = solve(elim_dead, x0_dead, Newton(), :dead)
    @test st_dead.code == SUCCESS
    @test isapprox(x_dead[1], 1.0; atol = 1e-12)
    @test abs(x_dead[2]) <= 1e-12
    @test isapprox(x_dead[3], 2.0; atol = 1e-12)
    @test st_dead.step_failures >= 1

    # Duplicate columns in J_yy: x1 + x2 = x3, so minimum-norm solution has x1 == x2 == x3 / 2
    x_dup, st_dup = solve(elim_dead, SVector(5.0, -3.0, 0.5), Newton(), :dup)
    @test st_dup.code == SUCCESS
    @test isapprox(x_dup[1], 1.0; atol = 1e-12)
    @test isapprox(x_dup[2], 1.0; atol = 1e-12)
    @test isapprox(x_dup[3], 2.0; atol = 1e-12)
    @test st_dup.step_failures >= 1

    # R3 dead unit on Array
    prob_dead_r3 = NonlinearProblem(f_singular_inner!)
    elim_dead_r3 = BlockEliminated(prob_dead_r3, [1, 2]; outer = [3], linear_inner = true)
    x_d_h, st_d_r3 = solve(elim_dead_r3, [3.0, 7.0, 0.5], Newton())
    @test st_d_r3.code == SUCCESS
    @test isapprox(x_d_h[1], 1.0; atol = 1e-12)
    @test abs(x_d_h[2]) <= 1e-12
    @test isapprox(x_d_h[3], 2.0; atol = 1e-12)
    @test st_d_r3.step_failures >= 1

    # Dead unit in VariableProjection (second column of Φ(z) is identically zero)
    vp_dead = VariableProjection(LeastSquaresProblem(r_vp_dead, 4), (1, 2); outer = (3,))
    x_vpd, st_vpd = solve(vp_dead, SVector(4.0, 9.0, 0.5), GaussNewton(;
        linsolve = SVDFactorization()))
    @test st_vpd.code == SUCCESS
    @test isapprox(x_vpd[1], 1.0; atol = 1e-12)
    @test abs(x_vpd[2]) <= 1e-12
    @test isapprox(x_vpd[3], 2.0; atol = 1e-12)
    @test st_vpd.step_failures >= 1

    # Inner nonlinear solver failure (max_iterations = 1 from a distant guess) increments step_failures
    elim_fail_in = BlockEliminated(
        NonlinearProblem(f_elim), (1, 2);
        outer = (3, 4),
        linear_inner = false,
        inner_options = Options(Float64; max_iterations = 1, f_abstol = 1e-15, f_reltol = 1e-15)
    )
    _, st_fail_in = solve(
        elim_fail_in, SVector(10.0, -10.0, 0.5, 0.6), Newton(), 0.2; max_iterations = 2)
    @test st_fail_in.step_failures >= 1
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

    if Base.JLOptions().check_bounds != 1
        # skipped where a run forces `--check-bounds=yes`
        @test allocations(a -> solve(elim_lin, a, Newton(), 0.0), x0) == 0
        @test allocations(a -> solve(elim_nl, a, Newton(), 0.2), x0) == 0
        @test allocations(a -> solve(elim_def, a, Newton(), 0.0), x0) == 0
        @test allocations(a -> solve(vp_kf, a, m_gn, :noisy), x0_ls) == 0
        @test allocations(a -> solve(vp_gp, a, m_gn, :noisy), x0_ls) == 0

        for (lin_in, p_val) in ((true, 0.0), (false, 0.2))
            prob_r3_wrap = BlockEliminated(
                NonlinearProblem(f_elim!), [1, 2]; outer = [3, 4], linear_inner = lin_in)
            solver_r3 = init(prob_r3_wrap, [0.2, 0.4, 0.5, 0.6], Newton(), p_val)
            solve!(solver_r3, [0.2, 0.4, 0.5, 0.6], p_val)
            x_in = [0.2, 0.4, 0.5, 0.6]
            @test allocations(a -> solve!(solver_r3, a, p_val), x_in) == 0
        end

        for mode in (:kaufman, :golub_pereyra)
            vp_r3 = VariableProjection(
                LeastSquaresProblem(r_ls6!, 6), [1, 2]; outer = [3], jacobian_mode = mode)
            solver_vp_r3 = init(vp_r3, [1.0, 0.5, 1.2], m_gn, :exact)
            solve!(solver_vp_r3, [1.0, 0.5, 1.2], :exact)
            x_in_ls = [1.0, 0.5, 1.2]
            @test allocations(a -> solve!(solver_vp_r3, a, :exact), x_in_ls) == 0
        end
    end
end
