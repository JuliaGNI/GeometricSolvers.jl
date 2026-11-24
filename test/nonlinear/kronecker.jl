# The Kronecker stage solver: `Newton` with `KroneckerFactorization` on a `NonlinearProblem` with
# `stage_matrix` and `stage_jacobian`, against the stacked `Newton` solve of the same stage system,
# with a counting block factor, the refusals at `init`, and the edges of the stage system.

using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using GeometricSolvers
using GeometricSolvers: DenseLU, LinearMethod, factorize!!, init_linear
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: cond, eigen
using StaticArrays: SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/stage_problems.jl")

const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

# The tolerances of every solve here: the absolute residual test `‖G(Z)‖ ≤ f_abstol`, which ends a
# solve with `SUCCESS`, and the step test `‖ΔZ‖ ≤ x_abstol`, both at `eps^(3/4)`, above the
# residual floor of the stage systems below and far below their increments.
stage_xtol(::Type{T}) where {T} = eps(real(T))^(3 // 4)
function stage_options(::Type{T}) where {T}
    Options(
        T; f_abstol = stage_xtol(T), f_reltol = 0, x_abstol = stage_xtol(T), x_reltol = 0,
        max_iterations = 200, max_stalls = 5)
end

# The agreement bound `‖Z_kron - Z_stacked‖∞ ≤ C_AGREE · x_abstol` over every row of the testset
# below. The largest ratio measured over those rows is 0.64, in `Float64` (Gauss s = 3 on the
# two-species Lotka–Volterra field), and 0.22 in `Float32`: both solves stop at a residual of at
# most `f_abstol`, and `I - h A ⊗ J` is near the identity at `h = 0.1`, so each increment is
# within about `x_abstol` of the root. 4 leaves a factor of six over the measurement.
const C_AGREE = 4

# The options of a Kronecker solve in the agreement testsets: at most 8 steps. The simplified
# iteration takes at most 6 on those rows at `h = 0.1`, and an iteration whose Newton matrix is
# wrong converges at a rate that needs many more, so a wrong matrix that still reaches the root
# shows as `MAXITERS`.
function kronecker_options(::Type{T}) where {T}
    Options(
        T; f_abstol = stage_xtol(T), f_reltol = 0, x_abstol = stage_xtol(T), x_reltol = 0,
        max_iterations = 8, max_stalls = 5)
end

# The options of a solve that takes one step, and the bound `C_STEP · eps · ‖Z‖∞` on the
# difference of two first steps, ten times the largest measured ratio, 1.6.
function first_step(::Type{T}) where {T}
    Options(
        T; f_abstol = stage_xtol(T), f_reltol = 0, x_abstol = stage_xtol(T), x_reltol = 0,
        max_iterations = 1, max_stalls = 5)
end
const C_STEP = 16

# A block method that factorises with `LUFactorization()` and logs the element type and the size of
# each matrix it factorises.
struct CountingLU <: LinearMethod{Nothing} end
const FACTOR_LOG = Tuple{DataType, Tuple{Int, Int}}[]
function GeometricSolvers.init_linear(::CountingLU, ::Nothing, x::AbstractVector)
    return init_linear(LUFactorization(), nothing, x)
end
function GeometricSolvers.factorize!!(fac::DenseLU, ::CountingLU, J::AbstractMatrix)
    push!(FACTOR_LOG, (eltype(J), size(J)))
    return factorize!!(fac, LUFactorization(), J)
end

# The bytes of a call `f(a)`, after one warm call through the barrier.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

# A `solve!` of `solver` from `x₀` into `x`, at the parameters `p`.
struct SolveFrom{X, S, Q}
    x₀::X
    solver::S
    p::Q
end
(f::SolveFrom)(x) = (copyto!(x, f.x₀); solve!(f.solver, x, f.p))

stage_params(::Type{T}, h, y₀) where {T} = (h = real(T)(h), y = T.(y₀))

const KRON = Newton(; linsolve = KroneckerFactorization())
const ALL_TABLEAUX = [STAGE_TABLEAUX; [("midpoint s = 1", midpoint_tableau())]]

@testset "the stage increments agree with the stacked Newton solve" begin
    worst = 0.0
    for T in REAL_ELTYPES, (tname, A) in ALL_TABLEAUX, (name, f!, jac!, y₀) in ODE_FIELDS
        n, s = length(y₀), size(A, 1)
        prob, _ = stage_problem(f!, jac!, A, T, n)
        p = stage_params(T, 0.1, y₀)
        Z₀ = zeros(T, s * n)
        opt = stage_options(T)
        Zk, stk = solve(prob, Z₀, KRON, p; options = kronecker_options(T))
        Zs, sts = solve(prob, Z₀, Newton(), p; options = opt)
        @test stk.code == SUCCESS
        @test sts.code == SUCCESS
        # the oscillator is linear, so its frozen Jacobian is exact and one step solves it
        name == "HarmonicOscillator" && @test stk.iterations == 1
        ratio = maximum(abs, Zk .- Zs) / stage_xtol(T)
        worst = max(worst, ratio)
        @test ratio <= C_AGREE
        @test all(!iszero, Zk)
    end
    @test worst > 0
end

@testset "the stage increments agree for a complex element type" begin
    name, f!, jac!, y₀ = COMPLEX_FIELD
    for T in filter(T -> T <: Complex, ELTYPES), (tname, A) in ALL_TABLEAUX

        n, s = length(y₀), size(A, 1)
        prob, _ = stage_problem(f!, jac!, A, T, n)
        p = stage_params(T, 0.1, y₀)
        Z₀ = zeros(T, s * n)
        opt = stage_options(T)
        Zk, stk = solve(prob, Z₀, KRON, p; options = kronecker_options(T))
        Zs, sts = solve(prob, Z₀, Newton(), p; options = opt)
        @test stk.code == SUCCESS
        @test sts.code == SUCCESS
        @test stk.iterations == 1
        @test maximum(abs, Zk .- Zs) <= C_AGREE * stage_xtol(T)
    end
end

@testset "a step that is not 1 is read from p.h" begin
    for T in REAL_ELTYPES
        name, f!, jac!, y₀ = ODE_FIELDS[2]
        A = gauss_tableau(3)
        n = length(y₀)
        prob, _ = stage_problem(f!, jac!, A, T, n)
        Z₀ = zeros(T, 3n)
        opt = stage_options(T)
        Za, _ = solve(prob, Z₀, KRON, stage_params(T, 0.37, y₀); options = opt)
        Zb, _ = solve(prob, Z₀, Newton(), stage_params(T, 0.37, y₀); options = opt)
        Zc, _ = solve(prob, Z₀, Newton(), stage_params(T, 1.0, y₀); options = opt)
        @test maximum(abs, Za .- Zb) <= C_AGREE * stage_xtol(T)
        @test maximum(abs, Za .- Zc) > 100 * stage_xtol(T)
        # The first step from `Z = 0`, where `I - h A ⊗ J(yₙ)` is the Jacobian of the stacked
        # residual, so the two first steps solve one linear system and differ by rounding only:
        # at most 1.6 eps · ‖Z‖∞ measured over `h = 0.37`, `0.05` and `1`. A factor at another
        # `h` than the residual's moves the step by a fraction of itself.
        Zk, _ = solve(prob, Z₀, KRON, stage_params(T, 0.37, y₀); options = first_step(T))
        Zs, _ = solve(
            prob, Z₀, Newton(), stage_params(T, 0.37, y₀); options = first_step(T))
        @test maximum(abs, Zk .- Zs) <= C_STEP * eps(T) * maximum(abs, Zs)
        # with reuse, a second solve! at another `h` refreshes the factor at that `h`
        solver = init(prob, Z₀, Newton(; linsolve = KroneckerFactorization(), reuse = 3),
            stage_params(T, 0.37, y₀); options = first_step(T))
        solve!(solver)
        Zk, _ = solve!(solver, zeros(T, 3n), stage_params(T, 0.05, y₀))
        Zs, _ = solve(
            prob, Z₀, Newton(), stage_params(T, 0.05, y₀); options = first_step(T))
        @test maximum(abs, Zk .- Zs) <= C_STEP * eps(T) * maximum(abs, Zs)
    end
end

# The block factorisations of each refresh, in the order the factor takes them.
function refresh_chunks(log, nblocks)
    @assert length(log) % nblocks == 0
    return [log[(k - 1) * nblocks .+ (1:nblocks)] for k in 1:(length(log) ÷ nblocks)]
end

@testset "a refresh factorises ⌊s/2⌋ blocks in complex(T) and s mod 2 in T" begin
    counting = Newton(; linsolve = KroneckerFactorization(; block = CountingLU()))
    for T in ELTYPES, (tname, A) in ALL_TABLEAUX

        name, f!, jac!, y₀ = T <: Complex ? COMPLEX_FIELD : ODE_FIELDS[2]
        n, s = length(y₀), size(A, 1)
        prob, jac = stage_problem(f!, jac!, A, T, n)
        p = stage_params(T, 0.1, y₀)
        empty!(FACTOR_LOG)
        _, st = solve(prob, zeros(T, s * n), counting, p; options = stage_options(T))
        @test st.code == SUCCESS
        refreshes = jac.calls[]
        @test refreshes >= 1
        nblocks = T <: Complex ? s : s ÷ 2 + s % 2
        @test length(FACTOR_LOG) == refreshes * nblocks
        for chunk in refresh_chunks(FACTOR_LOG, nblocks)
            @test all(e -> e[2] == (n, n), chunk)
            if T <: Complex
                @test count(e -> e[1] === T, chunk) == s
            else
                @test count(e -> e[1] === complex(T), chunk) == s ÷ 2
                @test count(e -> e[1] === T, chunk) == s % 2
            end
        end
        # the stacked solve: one factorisation of the `s·n × s·n` Jacobian per step
        empty!(FACTOR_LOG)
        _, sts = solve(prob, zeros(T, s * n), Newton(; linsolve = CountingLU()), p;
            options = stage_options(T))
        @test sts.code == SUCCESS
        @test length(FACTOR_LOG) == sts.iterations
        @test all(e -> e == (T, (s * n, s * n)), FACTOR_LOG)
    end
end

@testset "a tableau whose eigenvectors are ill conditioned is refused at init" begin
    for T in REAL_ELTYPES
        y₀ = [0.5, 0.0]
        for A in (sdirk2_tableau(), [1.0 1.0; 0.0 1.0])
            prob, _ = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
            @test_throws "is not diagonalisable to the working precision" init(
                prob, zeros(T, 4), KRON, stage_params(T, 0.1, y₀))
        end
        A6 = gauss_tableau(6)
        @test 600 < cond(eigen(A6).vectors) < 610
        prob, _ = stage_problem(oscillator!, oscillator_jac!, A6, T, 2)
        solver = init(prob, zeros(T, 12), KRON, stage_params(T, 0.1, y₀))
        _, st = solve!(solver)
        @test st.code == SUCCESS
    end
end

# The stage matrix `V diag(1, 2) V⁻¹` whose unit eigenvectors `(1, 0)` and `(1, δ) / ‖(1, δ)‖`
# have a condition number near `2 / δ`.
function conditioned_tableau(δ)
    V = [1.0 1.0; 0.0 δ]
    return V * [1.0 0.0; 0.0 2.0] / V
end

@testset "the refusal bound is eps(real(T))^(-1/2)" begin
    for T in REAL_ELTYPES
        bound = inv(sqrt(eps(real(T))))
        for (factor, refused) in ((0.8, false), (1.25, true))
            A = conditioned_tableau(2 / (factor * bound))
            κ = cond(eigen(A).vectors)
            @test (κ > bound) == refused
            prob, _ = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
            p = stage_params(T, 0.1, [0.5, 0.0])
            if refused
                @test_throws "is not diagonalisable to the working precision" init(
                    prob, zeros(T, 4), KRON, p)
            else
                @test init(prob, zeros(T, 4), KRON, p) isa NonlinearSolver
            end
        end
    end
end

@testset "refusals at init" begin
    for T in REAL_ELTYPES
        A = gauss_tableau(2)
        prob, _ = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
        p = stage_params(T, 0.1, [0.5, 0.0])
        Z = zeros(T, 4)
        @test_throws "a value iterate" init(prob, SVector{4, T}(Z), KRON, p)
        @test_throws "a device iterate" init(prob, JLArray(Z), KRON, p)
        @test_throws "takes no DogLeg" init(
            prob, Z, Newton(; linsolve = KroneckerFactorization(), globalization = DogLeg()),
            p)
        @test_throws "takes no LevenbergMarquardt" init(
            prob, Z,
            Newton(; linsolve = KroneckerFactorization(),
                globalization = LevenbergMarquardt()),
            p)
        @test_throws "takes no refinement" init(
            prob, Z, Newton(; linsolve = KroneckerFactorization(), refinement = Refinement()),
            p)
        @test_throws "takes no regularization" init(
            prob, Z, Newton(; linsolve = KroneckerFactorization(), regularization = 0.1), p)
        @test_throws "takes no factorisation precision" init(
            prob, Z, Newton(; linsolve = KroneckerFactorization(Float32)), p)
        @test_throws "takes no jacobian" init(
            prob, Z,
            Newton(; linsolve = KroneckerFactorization(),
                jacobian = AutoForwardDiff()), p)
        @test_throws "needs a NonlinearProblem with stage_matrix" init(
            NonlinearProblem(StageSystem(oscillator!, A, T, 2)), Z, KRON, p)
        @test_throws "a multiple of the number of stages" init(prob, zeros(T, 5), KRON, p)
        @test_throws "reads the step from p.h" init(prob, Z, KRON, (; y = p.y))
        @test_throws "an invertible stage matrix" init(
            stage_problem(oscillator!, oscillator_jac!, [0.0 0.0; 0.5 0.5], T, 2)[1],
            Z, KRON, p)
        @test_throws "a real stage matrix" init(
            stage_problem(oscillator!, oscillator_jac!, complex.(A), T, 2)[1], Z, KRON, p)
    end
    @test_throws "a square stage matrix" init(
        stage_problem(oscillator!, oscillator_jac!, ones(2, 3), Float64, 2)[1],
        zeros(4), KRON, stage_params(Float64, 0.1, [0.5, 0.0]))
    @test_throws "takes a block method" KroneckerFactorization(; block = GMRES())
    @test_throws "takes a block method" KroneckerFactorization(;
        block = LUFactorization(Float32))
    f = StageSystem(oscillator!, gauss_tableau(2), Float64, 2)
    @test_throws "stage_matrix and stage_jacobian together" NonlinearProblem(
        f; stage_matrix = gauss_tableau(2))
    @test_throws "stage_matrix and stage_jacobian together" NonlinearProblem(
        f; stage_jacobian = FrozenJacobian(oscillator_jac!))
    @test_throws "either block_residuals or stage_matrix" NonlinearProblem(
        f; stage_matrix = gauss_tableau(2), stage_jacobian = FrozenJacobian(oscillator_jac!),
        block_residuals = (f,))
end

@testset "the stage accessors" begin
    A = gauss_tableau(2)
    jac = FrozenJacobian(oscillator_jac!)
    f = StageSystem(oscillator!, A, Float64, 2)
    prob = NonlinearProblem(f; stage_matrix = A, stage_jacobian = jac)
    @test prob.F isa StageResidual
    @test stage_matrix(prob) === A
    @test stage_jacobian(prob) === jac
    @test prob.F(zeros(4), ones(4), (h = 0.1, y = [0.5, 0.0])) ==
          f(zeros(4), ones(4), (h = 0.1, y = [0.5, 0.0]))
    @test stage_matrix(NonlinearProblem(f)) === nothing
    @test stage_jacobian(NonlinearProblem(f)) === nothing
    @test stage_matrix(NonlinearProblem((G, x, p) -> G)) === nothing
    @test stage_jacobian(NonlinearProblem((G, x, p) -> G)) === nothing
    @test stage_matrix(LeastSquaresProblem((r, x, p) -> r, 2)) === nothing
end

@testset "the reuse rule governs the refresh, and jac! runs at each refresh" begin
    counting = KroneckerFactorization(; block = CountingLU())
    for T in REAL_ELTYPES, (tname, A) in ALL_TABLEAUX

        name, f!, jac!, y₀ = ODE_FIELDS[3]
        n, s = length(y₀), size(A, 1)
        nblocks = s ÷ 2 + s % 2
        p = stage_params(T, 0.1, y₀)
        # no reuse: one refresh per step
        prob, jac = stage_problem(f!, jac!, A, T, n)
        empty!(FACTOR_LOG)
        _, st = solve(prob, zeros(T, s * n), Newton(; linsolve = counting), p;
            options = stage_options(T))
        @test st.code == SUCCESS
        @test jac.calls[] == st.iterations
        @test jac.calls[] == length(FACTOR_LOG) ÷ nblocks
        # reuse for up to three steps: fewer refreshes than steps, and one jac! per refresh
        prob, jac = stage_problem(f!, jac!, A, T, n)
        empty!(FACTOR_LOG)
        _, st = solve(prob, zeros(T, s * n), Newton(; linsolve = counting, reuse = 3), p;
            options = stage_options(T))
        @test st.code == SUCCESS
        @test jac.calls[] == length(FACTOR_LOG) ÷ nblocks
        @test jac.calls[] < st.iterations
    end
end

@testset "a zero step gives the zero increment" begin
    for T in REAL_ELTYPES, (tname, A) in ALL_TABLEAUX

        y₀ = [0.5, 0.0]
        s = size(A, 1)
        prob, jac = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
        p = stage_params(T, 0, y₀)
        Z, st = solve(prob, fill(T(0.25), 2s), KRON, p; options = stage_options(T))
        @test st.code == SUCCESS
        @test all(iszero, Z)
        Z, st = solve(prob, zeros(T, 2s), KRON, p; options = stage_options(T))
        @test st.code == SUCCESS
        @test all(iszero, Z)
        # a zero step after a refresh at a nonzero one, on the same solver
        solver = init(prob, zeros(T, 2s), KRON, stage_params(T, 0.1, y₀);
            options = stage_options(T))
        _, st = solve!(solver)
        @test st.code == SUCCESS
        Z, st = solve!(solver, fill(T(0.25), 2s), p)
        @test st.code == SUCCESS
        @test all(iszero, Z)
    end
end

# The linear field `f(y) = J y` of a constant Jacobian `J`.
struct LinearField{M}
    J::M
end
(L::LinearField)(dy, y) = (dy .= L.J * y; dy)
struct ConstantJacobian{M}
    J::M
end
(C::ConstantJacobian)(J, y) = (J .= C.J; J)

function linear_stage_problem(A, J::AbstractMatrix{T}) where {T}
    n = size(J, 1)
    return NonlinearProblem(StageSystem(LinearField(J), A, T, n);
        stage_matrix = A, stage_jacobian = FrozenJacobian(ConstantJacobian(J)))
end

@testset "a singular real block and a singular complex block give SINGULAR" begin
    A = gauss_tableau(3)
    for T in REAL_ELTYPES
        # the real block γ I - J with γ = 1 / (h λ) and J = γ
        probe = init(linear_stage_problem(A, ones(T, 1, 1)), zeros(T, 3), KRON,
            stage_params(T, 0.1, [1.0]))
        h = T(0.1)
        γ = only(probe.state.fac.μ) / h
        J = fill(γ, 1, 1)
        _, st = solve(linear_stage_problem(A, J), zeros(T, 3), KRON,
            stage_params(T, h, [1.0]); options = stage_options(T))
        @test st.code == SINGULAR
        # the complex block γ I - J with γ = a + 32i and J = [a -32; 32 a], whose eigenvalues are
        # γ and its conjugate: `h` makes the imaginary part of γ the power of two 32 exactly
        μ = only(probe.state.fac.pair_μ)
        h = imag(μ) / 32
        a = real(μ) / h
        @test imag(μ) / h == 32
        J = T[a -32; 32 a]
        _, st = solve(linear_stage_problem(A, J), zeros(T, 6), KRON,
            stage_params(T, h, [1.0, 0.5]); options = stage_options(T))
        @test st.code == SINGULAR
    end
end

@testset "the state holds the n × n Jacobian of the field" begin
    for T in ELTYPES, (tname, A) in ALL_TABLEAUX

        name, f!, jac!, y₀ = T <: Complex ? COMPLEX_FIELD : ODE_FIELDS[4]
        n, s = length(y₀), size(A, 1)
        prob, _ = stage_problem(f!, jac!, A, T, n)
        solver = init(prob, zeros(T, s * n), KRON, stage_params(T, 0.1, y₀))
        @test size(solver.state.J) == (n, n)
        @test eltype(solver.state.J) === T
    end
end

@testset "a non-finite residual or Jacobian gives NONFINITE" begin
    nan_field!(dy, y) = (dy .= y .* eltype(dy)(NaN); dy)
    nan_jac!(J, y) = (fill!(J, eltype(J)(NaN)); J)
    inf_jac!(J, y) = (fill!(J, zero(eltype(J))); J[1, 2] = eltype(J)(Inf); J)
    # an infinite entry on the diagonal, which enters each block as a pivot candidate
    inf_diag_jac!(J, y) = (fill!(J, zero(eltype(J))); J[1, 1] = eltype(J)(Inf); J)
    A = gauss_tableau(3)
    for T in REAL_ELTYPES
        y₀ = [0.5, 0.0]
        p = stage_params(T, 0.1, y₀)
        prob, _ = stage_problem(nan_field!, oscillator_jac!, A, T, 2)
        _, st = solve(prob, zeros(T, 6), KRON, p; options = stage_options(T))
        @test st.code == NONFINITE
        for bad! in (nan_jac!, inf_jac!, inf_diag_jac!)
            prob, _ = stage_problem(oscillator!, bad!, A, T, 2)
            _, st = solve(prob, zeros(T, 6), KRON, p; options = stage_options(T))
            @test st.code == NONFINITE
        end
        prob, _ = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
        _, st = solve(prob, zeros(T, 6), KRON, stage_params(T, NaN, y₀);
            options = stage_options(T))
        @test st.code == NONFINITE
    end
end

@testset "a line search along the Kronecker direction" begin
    for T in REAL_ELTYPES, glob in (Backtracking(), StrongWolfe())

        name, f!, jac!, y₀ = ODE_FIELDS[2]
        A = radau_iia3_tableau()
        prob, _ = stage_problem(f!, jac!, A, T, 2)
        p = stage_params(T, 0.1, y₀)
        opt = stage_options(T)
        Zk, stk = solve(prob, zeros(T, 6),
            Newton(; linsolve = KroneckerFactorization(), globalization = glob), p;
            options = opt)
        Zs, _ = solve(prob, zeros(T, 6), Newton(), p; options = opt)
        @test stk.code == SUCCESS
        @test maximum(abs, Zk .- Zs) <= C_AGREE * stage_xtol(T)
        # the direction solves `I - h A ⊗ J(yₙ)`, not the Jacobian of the residual at the
        # iterate, so the factor reports an inexact solve and the line search measures its slope
        solver = init(prob, zeros(T, 6), KRON, p)
        solve!(solver)
        @test !GeometricSolvers.exact_solve(solver.state.fac)
    end
end

@testset "the block methods" begin
    A = radau_iia3_tableau()
    name, f!, jac!, y₀ = ODE_FIELDS[2]
    for T in REAL_ELTYPES
        prob, _ = stage_problem(f!, jac!, A, T, 2)
        p = stage_params(T, 0.1, y₀)
        Zs, _ = solve(prob, zeros(T, 6), Newton(), p; options = stage_options(T))
        for block in (QRFactorization(), SVDFactorization())
            Zk, st = solve(prob, zeros(T, 6),
                Newton(; linsolve = KroneckerFactorization(; block)), p;
                options = kronecker_options(T))
            @test st.code == SUCCESS
            @test maximum(abs, Zk .- Zs) <= C_AGREE * stage_xtol(T)
        end
    end
    @test_throws "is in general not Hermitian" KroneckerFactorization(;
        block = CholeskyFactorization())
    @test_throws "is in general not Hermitian" KroneckerFactorization(;
        block = LDLTFactorization())
end

@testset "a tiny step agrees with the stacked solve" begin
    A = gauss_tableau(3)
    for T in REAL_ELTYPES, h in (nextfloat(zero(T)), floatmin(T), eps(T))

        prob, _ = stage_problem(pendulum!, pendulum_jac!, A, T, 2)
        p = (h = h, y = T[acos(0.4), 0.0])
        Zk, stk = solve(prob, fill(T(0.25), 6), KRON, p; options = stage_options(T))
        Zs, sts = solve(prob, fill(T(0.25), 6), Newton(), p; options = stage_options(T))
        @test stk.code == SUCCESS
        @test sts.code == SUCCESS
        @test maximum(abs, Zk .- Zs) <= C_AGREE * stage_xtol(T)
    end
end

@testset "a huge step keeps the blocks finite" begin
    # at `h = floatmax(T) / 2` the block `γ I - J` is near `-J`; a scale of `J` by the power of
    # two below `h` would overflow it
    A = gauss_tableau(3)
    for T in REAL_ELTYPES
        prob, _ = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
        solver = init(prob, zeros(T, 6), KRON, stage_params(T, 0.1, [0.5, 0.0]))
        fac = GeometricSolvers.factorize_stages!!(
            solver.state.fac, LUFactorization(), T[4 4; -4 4], floatmax(T) / 2)
        @test fac.info == 0
        @test all(isfinite, GeometricSolvers.ldiv!!(zeros(T, 6), fac, ones(T, 6)))
    end
end

@testset "a dual input is refused" begin
    A = gauss_tableau(2)
    prob, _ = stage_problem(pendulum!, pendulum_jac!, A, Float64, 2)
    y = [acos(0.4), 0.0]
    p = (h = ForwardDiff.Dual(0.1, 1.0), y = y)
    @test_throws "for a Newton with KroneckerFactorization" solve(
        prob, zeros(4), KRON, p; options = stage_options(Float64))
end

@testset "a solve! allocates nothing and infers" begin
    for T in REAL_ELTYPES, (tname, A) in ALL_TABLEAUX, reuse in (nothing, 3)
        s = size(A, 1)
        prob, _ = stage_problem(oscillator!, oscillator_jac!, A, T, 2)
        p = stage_params(T, 0.1, [0.5, 0.0])
        x₀ = zeros(T, 2s)
        solver = init(prob, copy(x₀), Newton(; linsolve = KroneckerFactorization(), reuse),
            p; options = stage_options(T))
        x = copy(x₀)
        @test allocations(SolveFrom(x₀, solver, p), x) == 0
        # the control: the barrier sees an allocation
        @test allocations(collect, x) > 0
        @test (@inferred solve!(solver, x, p))[2].code == SUCCESS
        JET_WORKS && @test_opt target_modules = (GeometricSolvers,) solve!(solver, x, p)
    end
end
