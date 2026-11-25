# R5 on a symmetric Jacobian: the MINRES-QLP linear method of a matrix-free `StationarityProblem`
# with `E == U`, the Steihaug trust region on the model of `S`, the matrix-free `DogLeg` on it,
# and `GMRES` on a `StationarityProblem` with `E ≠ U`.
#
# Each solve runs over `REAL_ELTYPES`, and the matrix-free ones on `Array` and `JLArray`. The
# actions of the device leg reach their entries by gathers and reductions only, since `x[i]` is
# scalar indexing on a device array.

using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: Diagonal, I, mul!, norm, pinv, qr
using Random: Xoshiro
using Test

using GeometricSolvers
using StaticArrays: SVector
using GeometricSolvers: Backtracking, Bisection, DogLeg, FixedForcing, GMRES, JVPFunction,
                        LevenbergMarquardt, MINRESQLP, NONFINITE, Refinement,
                        LINESEARCH_FAILED, MAXITERS, Newton, Steihaug, SUCCESS,
                        StationarityProblem,
                        NonlinearProblem, f_calls, init, init_minresqlp_workspace,
                        jvp_calls,
                        max_inner_iterations, minresqlp_solve!,
                        residual!!, rnorm, solve, solve!, start!!, Static, step!!,
                        steihaug_cg!

include("../helpers/matrix.jl")

allowscalar(false)

# ─── The problems ────────────────────────────────────────────────────────────

# The product of a matrix as a callable struct, and an allocation count through a function
# barrier that calls the function once before it measures.
struct MatrixOperator{M}
    A::M
end

(o::MatrixOperator)(y, v) = mul!(y, o.A, v)

allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

# A counter of the products of a `JVPFunction`.
mutable struct ProductCount
    n::Int
end

# The action of the indefinite problem: `S(x) = ½ Σ Dᵢ xᵢ² + ¼ Σ xᵢ⁴ + κ Σ xᵢ x_{π(i)}` with the
# cyclic shift `π`, and the datum `b = -c` that puts the root at `X_IND`. Its Hessian
# `diag(D + 3x²) + κ (P + Pᵀ)` is indefinite at the root, where `D + 3x²` is
# `(1.75, -0.97, 2.27, -0.38)`. `D` and `perm` are arrays of the backend, so the device leg
# evaluates the action with gathers and reductions only.
const D_IND = [1.0, -1.0, 2.0, -0.5]
const X_IND = [0.5, 0.1, -0.3, 0.2]
const PERM = [2, 3, 4, 1]
const PERMINV = [4, 1, 2, 3]
const KAPPA = 0.2

ind_gradient(x) = D_IND .* x .+ x .^ 3 .+ KAPPA .* (x[PERM] .+ x[PERMINV])

function indefinite_problem(AT, ::Type{T}) where {T}
    D = AT(T.(D_IND))
    perm = AT(PERM)
    S = (x, p) -> T(0.5) * sum(D .* x .^ 2) + T(0.25) * sum(x .^ 4) +
                  T(KAPPA) * sum(x .* x[perm])
    b = AT(T.(-ind_gradient(X_IND)))
    return StationarityProblem(S; b = b)
end

# The Hessian–vector product of the indefinite action, counted.
function indefinite_jvp(AT, ::Type{T}, count) where {T}
    D = AT(T.(D_IND))
    perm = AT(PERM)
    perminv = AT(PERMINV)
    return JVPFunction((Jv, x, v, p) -> begin
        count.n += 1
        Jv .= (D .+ 3 .* x .^ 2) .* v .+ T(KAPPA) .* (v[perm] .+ v[perminv])
        Jv
    end)
end

const X0_IND = X_IND .+ [0.02, -0.03, 0.01, 0.02]

# `S(x) = Σ (½ dᵢ xᵢ² + ¼ xᵢ⁴)` and its Hessian–vector product as loops that allocate nothing, for
# the allocation count of a whole solve.
struct QuarticAction{V}
    d::V
end

function (a::QuarticAction)(x, p)
    s = zero(eltype(x))
    for i in eachindex(x)
        s += a.d[i] * x[i]^2 / 2 + x[i]^4 / 4
    end
    return s
end

struct QuarticHVP{V}
    d::V
end

function (h::QuarticHVP)(Jv, x, v, p)
    for i in eachindex(Jv)
        Jv[i] = (h.d[i] + 3 * x[i]^2) * v[i]
    end
    return Jv
end

# The same action on a state of six entries, of which `E = U = (2, 3, 4, 5)` are unknown: the
# fixed entries 1 and 6 take no part in it.
function indexed_problem(::Type{T}) where {T}
    S = (x, p) -> begin
        u = x[2:5]
        T(0.5) * sum(T.(D_IND) .* u .^ 2) + T(0.25) * sum(u .^ 4) +
        T(KAPPA) * sum(u .* u[PERM]) + T(7) * x[1] * x[6]
    end
    return StationarityProblem(S; E = (2, 3, 4, 5), U = (2, 3, 4, 5),
        b = T.(-ind_gradient(X_IND)))
end

# A two-equation `StationarityProblem` with `E = (1, 2)` and `U = (3, 4)`: `F₁ = x₃ + x₃³/3 - c₁`
# and `F₂ = x₄ + x₃ x₄ - c₂`, whose Jacobian `[1 + u₁²  0; u₂  1 + u₁]` in `u = (x₃, x₄)` is not
# symmetric. The root is `u = (0.5, -0.4)`. The masks `e₁ … e₄` pick the entries of a device array.
const U_NONSYM = [0.5, -0.4]

function nonsymmetric_problem(AT, ::Type{T}) where {T}
    e = ntuple(i -> AT(T.([j == i ? 1 : 0 for j in 1:4])), 4)
    S = (x, p) -> begin
        x1, x2, x3, x4 = (sum(e[i] .* x) for i in 1:4)
        x1 * (x3 + x3^3 / 3) + x2 * (x4 + x3 * x4)
    end
    u = U_NONSYM
    c = [u[1] + u[1]^3 / 3, u[2] + u[1] * u[2]]
    return StationarityProblem(S; E = [1, 2], U = [3, 4], b = AT(T.(-c)))
end

function nonsymmetric_jvp(AT, ::Type{T}, count) where {T}
    e1 = AT(T[1, 0])
    e2 = AT(T[0, 1])
    return JVPFunction((Jv, u, v, p) -> begin
        count.n += 1
        u1, u2, v1, v2 = sum(e1 .* u), sum(e2 .* u), sum(e1 .* v), sum(e2 .* v)
        Jv .= e1 .* ((1 + u1^2) * v1) .+ e2 .* (u2 * v1 + (1 + u1) * v2)
        Jv
    end)
end

# `S(x) = Σ (xᵢ atan xᵢ - ½ log(1 + xᵢ²))`, whose residual is `atan.(x)` and whose Hessian
# `1/(1 + x²)` is positive: from `x = 1.5` the Newton step overshoots to `-1.69`, where `‖F‖` is
# larger, so a trust region that accepts that step increases `‖F‖`.
function atan_problem(AT, ::Type{T}) where {T}
    return StationarityProblem((x, p) -> sum(x .* atan.(x) .- log1p.(x .^ 2) ./ 2))
end

function atan_jvp(count)
    return JVPFunction((Jv, x, v, p) -> begin
        count.n += 1
        Jv .= v ./ (1 .+ x .^ 2)
        Jv
    end)
end

# A quadratic action with the Hessian `diag(1, -1/100)`, from `x = 0` where `F = (1, 1/100)`: the
# first conjugate-gradient direction `-F` has positive curvature, and the second has negative
# curvature, so a radius of 2 ends the step on the boundary along it, where the model residual is
# below `‖F‖` and the step is accepted.
function negative_curvature_problem(AT, ::Type{T}) where {T}
    h = AT(T[1, -1 // 100])
    return StationarityProblem((x, p) -> sum(h .* x .^ 2) / 2; b = AT(T[1, 1 // 100]))
end

const H_NEG = [1.0, -0.01]

# The state of the inner `Newton` of a stationarity solver and its reduced problem, with the
# residual at the start, so that a test can take single steps.
function first_state(solver)
    st = solver.state
    rprob = st.rprob
    u = copy(st.u_work)
    Fu = residual!!(similar(u), rprob, u, nothing)
    inner = start!!(st.inner, solver.method, rprob, u, Fu, nothing, rnorm(Fu))
    return inner, rprob, u, Fu
end

# The direction of a Steihaug conjugate-gradient run on the matrix `A`, from the residual `F`,
# with its workspace.
function cg_run(A, F, δ; tol = zero(eltype(F)), maxiter = 50)
    R = real(eltype(F))
    g = (; d = similar(F), r = similar(F), pv = similar(F), Jp = similar(F))
    op = (y, v) -> (y .= A * v; y)
    cg = steihaug_cg!(g, op, F, R(rnorm(F)), R(tol), Int32(maxiter), R(δ))
    return g, cg
end

@testset "R5 symmetric Krylov and the Steihaug trust region" begin
    @testset "MINRES-QLP gives the minimum-length solution of diag(2, -1, 0), $AT, $T" for T in REAL_ELTYPES,
        AT in ARRAY_BACKENDS

        A = AT(T[2 0 0; 0 -1 0; 0 0 0])
        b = AT(T[1, 1, 1])
        ws = init_minresqlp_workspace(MINRESQLP(), b)
        op = (y, v) -> (y .= A * v; y)
        x, info = minresqlp_solve!(ws, op, b, T(rnorm(b)), zero(T), Int32(10))
        # pinv(A) b, within 8 eps(T) ‖b‖: the null-space component of b is not solved for
        @test rnorm(x .- AT(T[0.5, -1, 0])) <= 8 * eps(T) * rnorm(b)
        @test all(isfinite, Array(x))
        @test info.iterations == 3
        @test !info.converged
        @test info.resnorm ≈ one(T) rtol = 8 * eps(T)
        # the scale of b takes no part in the rank decision: the solution scales with b
        for k in -8:4:8
            s = T(10.0^k)
            bs = s .* b
            xs, infos = minresqlp_solve!(ws, op, bs, T(rnorm(bs)), zero(T), Int32(10))
            @test rnorm(xs .- s .* AT(T[0.5, -1, 0])) <= 8 * eps(T) * rnorm(bs)
            @test infos.iterations == 3
        end
    end

    @testset "MINRES-QLP takes no rank decision from the scale of b, $T" for T in REAL_ELTYPES
        # a nonsingular indefinite system with condition number 6
        Q = Matrix(qr(randn(Xoshiro(3), 4, 4)).Q)
        A = T.(Q * Diagonal([2.0, -1.0, 0.5, 3.0]) * Q')
        A = (A + A') / 2
        op = (y, v) -> (y .= A * v; y)
        xe = [1.0, -1.0, 0.5, 2.0]
        for k in -8:4:8
            s = 10.0^k
            b = A * T.(s .* xe)
            ws = init_minresqlp_workspace(MINRESQLP(), b)
            x, info = minresqlp_solve!(ws, op, b, T(rnorm(b)), zero(T), Int32(12))
            @test !info.singular
            # the forward error of a system with condition number 6, from the rounding of b and
            # of the solve
            xb = Float64.(A) \ Float64.(b)
            @test norm(Float64.(x) .- xb) <= 6 * 8 * eps(T) * norm(xb)
        end
    end

    @testset "MINRES-QLP stops on a zero right-hand side and on an invariant space, $T" for T in REAL_ELTYPES
        ws = init_minresqlp_workspace(MINRESQLP(), zeros(T, 3))
        A = T.(Diagonal([2.0, -1.0, 3.0]))
        op = (y, v) -> (y .= A * v; y)
        x, info = minresqlp_solve!(ws, op, zeros(T, 3), zero(T), zero(T), Int32(10))
        @test x == zeros(T, 3)
        @test info.iterations == 0 && info.converged && info.resnorm == 0
        # a right-hand side in a two-dimensional invariant space of a rotated matrix: the third
        # Lanczos vector vanishes to rounding, and the solve stops after two products
        Q = Matrix(qr(randn(Xoshiro(3), 6, 6)).Q)
        A6 = T.(Q * Diagonal([1.0, -2.0, 3.0, 0.5, -4.0, 2.5]) * Q')
        A6 = (A6 + A6') / 2
        b6 = A6 * T.(Q[:, 1] .+ Q[:, 2])
        ws6 = init_minresqlp_workspace(MINRESQLP(), b6)
        x6, info6 = minresqlp_solve!(
            ws6, (y, v) -> (y .= A6 * v; y), b6, rnorm(b6), zero(T),
            Int32(20))
        @test info6.iterations == 2
        # the space is invariant, so the residual of the projected problem is zero
        @test info6.converged && info6.resnorm == 0
        @test rnorm(A6 * x6 .- b6) <= 64 * eps(T) * rnorm(b6)
    end

    @testset "The inner solves allocate nothing, $T" for T in REAL_ELTYPES
        A = T.(Diagonal([2.0, -1.0, 0.0, 3.0, 0.5])) + T(0.1) * ones(T, 5, 5)
        op = MatrixOperator(A)
        b = T[1, 1, 1, 2, -1]
        ws = init_minresqlp_workspace(MINRESQLP(), b)
        g = (; d = similar(b), r = similar(b), pv = similar(b), Jp = similar(b))
        solve_inner = bb -> minresqlp_solve!(ws, op, bb, rnorm(bb), zero(T), Int32(20))
        cg_inner = bb -> steihaug_cg!(g, op, bb, rnorm(bb), T(1e-6), Int32(20), T(5))
        @test allocations(collect, b) > 0
        # `--check-bounds=yes`, which `Pkg.test` sets up to Julia 1.12, skips the measurement
        if Base.JLOptions().check_bounds != 1
            @test allocations(solve_inner, b) == 0
            @test allocations(cg_inner, b) == 0
        end
    end

    # A random symmetric matrix of order n in 3:12 with r in 1:n eigenvalues `3 randn` and the rest
    # zero, a random right-hand side, and the condition number of the matrix on its range. The
    # callers make the right-hand side compatible, `A b`, on odd trials.
    function singular_system(rng)
        n = rand(rng, 3:12)
        r = rand(rng, 1:n)
        Q = Matrix(qr(randn(rng, n, n)).Q)
        λ = [3 .* randn(rng, r); zeros(n - r)]
        A = Q * Diagonal(λ) * Q'
        A = (A + A') / 2
        b = randn(rng, n)
        return A, b, maximum(abs, λ) / minimum(abs, λ[1:r])
    end

    function minresqlp_pinv_error(A, b, ::Type{T}) where {T}
        AT, bT = T.(A), T.(b)
        ws = init_minresqlp_workspace(MINRESQLP(), bT)
        op = (y, v) -> (y .= AT * v; y)
        x, info = minresqlp_solve!(ws, op, bT, rnorm(bT), eps(T), Int32(3 * length(b)))
        xp = pinv(A; rtol = 1e-10) * b
        return norm(Float64.(x) .- xp) / norm(xp), x, info
    end

    @testset "A whole solve is inferred and allocates nothing, $T" for T in REAL_ELTYPES
        d = T[1, 1.5, 2, 0.5]
        x₀ = T.(X0_IND)
        prob = StationarityProblem(QuarticAction(d); b = -(d .* T.(X_IND) .+
                                                           T.(X_IND) .^ 3))
        jvp = JVPFunction(QuarticHVP(d))
        for glob in (nothing, Backtracking(), DogLeg(), Steihaug())
            lin = MINRESQLP(; forcing = FixedForcing(1e-6))
            m = glob === nothing ? Newton(; linsolve = lin, jacobian = jvp) :
                Newton(; linsolve = lin, globalization = glob, jacobian = jvp)
            solver = init(prob, copy(x₀), m)
            run = w -> (copyto!(w, x₀); solve!(solver, w))
            x, st = @inferred run(similar(x₀))
            @test st.code == SUCCESS
            @test rnorm(x .- T.(X_IND)) <= sqrt(eps(T))
            # `--check-bounds=yes`, which `Pkg.test` sets up to Julia 1.12, skips the measurement
            if Base.JLOptions().check_bounds != 1
                @test allocations(run, similar(x₀)) == 0
            end
        end
    end

    @testset "MINRES-QLP against pinv on singular symmetric systems, $T" for T in REAL_ELTYPES
        rng = Xoshiro(7)
        for trial in 1:60
            A, b, cond = singular_system(rng)
            isodd(trial) && (b = A * b)
            err, x, _ = minresqlp_pinv_error(A, b, T)
            @test all(isfinite, x)
            if isodd(trial)
                # a compatible system: the forward error of a solve with this condition number
                @test err <= 64 * eps(T) * cond
            else
                # an incompatible one: a bound of the rank threshold √eps(T) times the condition
                # number, which every draw of this seed meets in both precisions
                @test err <= sqrt(eps(T)) * cond
            end
        end
    end

    @testset "MINRES-QLP on incompatible systems in Float32 stops early on at most a few" begin
        # The limit that the docstring states: on an incompatible system, rounding in the Lanczos
        # process can make a diagonal entry count as zero at √eps(Float32) ‖T̄‖ before the Krylov
        # space holds the solution of minimum length. In Float64 every draw below is within
        # 10⁻⁷ of pinv(A) b; in Float32 6 of the 150 are more than 10⁻³ from it, among them
        # draws 100 and 272.
        far = Dict(Float32 => 0, Float64 => 0)
        stopped_far = 0
        draws = 0
        for T in REAL_ELTYPES
            rng = Xoshiro(11)
            for trial in 1:300
                A, b, _ = singular_system(rng)
                isodd(trial) && continue
                T === Float32 && (draws += 1)
                err, x, info = minresqlp_pinv_error(A, b, T)
                @test all(isfinite, x)
                if err > 1e-3
                    far[T] += 1
                    T === Float32 && info.singular && (stopped_far += 1)
                end
            end
        end
        @test far[Float64] == 0
        @test far[Float32] <= draws ÷ 10
        # each far result comes from a stop on an entry that counts as zero
        @test stopped_far == far[Float32]
    end

    @testset "Clause 1: Newton with MINRESQLP on an indefinite Hessian, counted, $AT, $T" for T in REAL_ELTYPES,
        AT in ARRAY_BACKENDS

        count = ProductCount(0)
        prob = indefinite_problem(AT, T)
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-6)),
            jacobian = indefinite_jvp(AT, T, count))
        x, st = solve(prob, AT(T.(X0_IND)), m)
        @test st.code == SUCCESS
        @test rnorm(Array(x) .- T.(X_IND)) <= sqrt(eps(T))
        @test st.jvp_calls == count.n
        @test st.jvp_calls > 0
        # one evaluation of F at the start and one per step under Static
        @test st.f_calls == st.iterations + 1
        # every product of the solve is one inner iteration, and there is no Jacobian matrix
        solver = init(prob, AT(T.(X0_IND)), m)
        @test solver.state.inner.J === nothing
        count.n = 0
        _, st2 = solve!(solver)
        @test st2.jvp_calls == count.n
        # the accessors read the counts of the inner solve
        @test jvp_calls(solver) == count.n
        @test f_calls(solver) == st2.f_calls
        @test 0 < max_inner_iterations(solver) <= 30
    end

    @testset "Clause 1: the default product is the Hessian–vector product of the AD seam, $T" for T in REAL_ELTYPES
        prob = indefinite_problem(Array, T)
        x, st = solve(prob, T.(X0_IND), Newton(; linsolve = MINRESQLP()))
        @test st.code == SUCCESS
        @test rnorm(x .- T.(X_IND)) <= sqrt(eps(T))
        @test st.jvp_calls > 0
        x2, st2 = solve(indexed_problem(T), T[9; X0_IND; -4], Newton(;
            linsolve = MINRESQLP()))
        @test st2.code == SUCCESS
        @test rnorm(x2[2:5] .- T.(X_IND)) <= sqrt(eps(T))
        @test x2[1] == 9 && x2[6] == -4
    end

    @testset "Clause 1: Newton with MINRESQLP on an indefinite quadratic at every scale, $T" for T in REAL_ELTYPES
        # eigenvalues about -1.04, 2.03 and 3.01, so a condition number of about 3
        A = T[2 0.3 0; 0.3 -1 0.2; 0 0.2 3]
        for k in -8:4:8
            b = T.(10.0^k .* [1, 1, 1])
            prob = StationarityProblem((x, p) -> sum(x .* (A * x)) / 2; b = b)
            x, st = solve(prob, zeros(T, 3),
                Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-6))))
            xe = -(Float64.(A) \ Float64.(b))
            @test st.code == SUCCESS
            # the inner solve of a quadratic is exact after three products: one Newton step
            @test st.iterations == 1
            @test norm(Float64.(x) .- xe) <= 8 * eps(T) * 3 * norm(xe)
        end
    end

    @testset "MINRESQLP refuses what it does not solve" begin
        S = (x, p) -> sum(x .^ 2) / 2 + x[1] * x[2]
        # E = (1, 2), U = (2, 1): the same indices in another order give a nonsymmetric J
        prob_order = StationarityProblem(S; E = (1, 2), U = (2, 1))
        order = r"^ArgumentError: a Newton with MINRESQLP solves the symmetric system of a StationarityProblem with E == U, the same indices in the same order; got E = \[1, 2\] and U = \[2, 1\], whose Jacobian is not symmetric: use GMRES\(\)$"
        @test_throws order solve(prob_order, [0.3, 0.2, 0.1], Newton(;
            linsolve = MINRESQLP()))
        @test_throws order solve(prob_order, [0.3, 0.2, 0.1],
            Newton(; linsolve = MINRESQLP(), globalization = Steihaug()))
        # the same indices in the same order are taken
        prob_same = StationarityProblem(S; E = (1, 2), U = (1, 2))
        _, st_same = solve(prob_same, [0.3, 0.2, 0.1], Newton(; linsolve = MINRESQLP()))
        @test st_same.code == SUCCESS
        F!(y, x, p) = (y .= x .^ 2 .- 1; y)
        @test_throws r"^ArgumentError: a Newton with MINRESQLP solves the symmetric system of a StationarityProblem with E == U; a NonlinearProblem takes GMRES\(\)$" solve(
            NonlinearProblem(F!), [2.0, 2.0], Newton(; linsolve = MINRESQLP()))
        noprec = r"^ArgumentError: MINRESQLP takes no preconditioner: pass preconditioner = nothing, or use GMRES$"
        @test_throws noprec MINRESQLP(; preconditioner = I)
        @test_throws noprec Newton(; linsolve = MINRESQLP(), preconditioner = I)
        @test_throws r"^ArgumentError: MINRESQLP maxiter must be ≥ 1$" MINRESQLP(;
            maxiter = 0)
        @test_throws r"^ArgumentError: Newton with MINRESQLP takes no factor reuse: a MINRESQLP step has no factor$" Newton(;
            linsolve = MINRESQLP(), reuse = 2)
        @test_throws r"^ArgumentError: Newton with MINRESQLP takes no refinement: a MINRESQLP step has no factor to refine against$" Newton(;
            linsolve = MINRESQLP(), refinement = Refinement())
        @test_throws r"^ArgumentError: Newton with a LevenbergMarquardt globalization needs a Jacobian matrix, which a MINRESQLP solve does not form$" Newton(;
            linsolve = MINRESQLP(), globalization = LevenbergMarquardt())
        steihaug = r"^ArgumentError: Newton with a Steihaug globalization needs linsolve = MINRESQLP\(\.\.\.\): its conjugate gradients take the place of the MINRES-QLP solve$"
        @test_throws steihaug Newton(; linsolve = GMRES(), globalization = Steihaug())
        @test_throws steihaug Newton(; globalization = Steihaug())
        @test_throws r"^ArgumentError: Newton with a JVPFunction needs linsolve = GMRES\(\.\.\.\) or MINRESQLP\(\.\.\.\): a factorising linear method needs a Jacobian matrix, which a JVPFunction does not give$" Newton(;
            jacobian = JVPFunction((Jv, x, v, p) -> Jv))
        # a value iterate has no Krylov solve, and a Jacobian function fills a matrix
        @test_throws r"^ArgumentError: a Newton with MINRESQLP solves an array linear system: the iterate must be an AbstractVector and not a value iterate, which is a Number or a StaticArray$" solve(
            prob_same, SVector(0.3, 0.2, 0.1), Newton(; linsolve = MINRESQLP()))
        jacfun = r"^ArgumentError: a Newton with GMRES or MINRESQLP takes jacobian = nothing, an AbstractADType or a JVPFunction: a Jacobian function fills a matrix, and a matrix-free solve forms none$"
        @test_throws jacfun solve(prob_same, [0.3, 0.2, 0.1],
            Newton(; linsolve = MINRESQLP(), jacobian = (J, x, p) -> J))
        @test_throws jacfun solve(prob_same, [0.3, 0.2, 0.1],
            Newton(; linsolve = GMRES(), jacobian = (J, x, p) -> J))
        for radius in (0, -1, Inf, NaN)
            @test_throws r"^ArgumentError: the radius of Steihaug must be positive and finite$" Steihaug(;
                radius)
        end
        @test Steihaug(; radius = 1e-9).radius == 1e-6
        # the forcing keyword of Newton configures MINRESQLP as it configures GMRES
        m = Newton(; linsolve = MINRESQLP(), forcing = FixedForcing(1e-3))
        @test m.linsolve isa MINRESQLP && m.linsolve.forcing == FixedForcing(1e-3)
    end

    @testset "F(x₀) = 0 takes no product, $T" for T in REAL_ELTYPES
        count = ProductCount(0)
        x₀ = T[0.5, -0.25, 1]
        # the gradient of `½‖x‖²` is `x` exactly, so the datum `-x₀` makes `x₀` the root in
        # floating point
        prob0 = StationarityProblem((x, p) -> sum(x .^ 2) / 2; b = -x₀)
        jvp = JVPFunction((Jv, x, v, p) -> (count.n += 1; Jv .= v; Jv))
        for glob in (nothing, DogLeg(), Steihaug())
            m = glob === nothing ? Newton(; linsolve = MINRESQLP(), jacobian = jvp) :
                Newton(; linsolve = MINRESQLP(), globalization = glob, jacobian = jvp)
            count.n = 0
            _, st = solve(prob0, copy(x₀), m)
            @test st.code == SUCCESS
            @test st.jvp_calls == 0
            @test count.n == 0
        end
    end

    @testset "A non-finite Hessian–vector product gives NONFINITE and a finite iterate, $T" for T in REAL_ELTYPES
        prob = indefinite_problem(Array, T)
        count = ProductCount(0)
        bad = JVPFunction((Jv, x, v, p) -> (count.n += 1; Jv .= T(NaN); Jv))
        for glob in (nothing, DogLeg(), Steihaug())
            m = glob === nothing ? Newton(; linsolve = MINRESQLP(), jacobian = bad) :
                Newton(; linsolve = MINRESQLP(), globalization = glob, jacobian = bad)
            count.n = 0
            x, st = solve(prob, T.(X0_IND), m)
            @test st.code == NONFINITE
            @test all(isfinite, x)
            # the first product ends the step
            @test count.n == 1
        end
        # under DogLeg the Cauchy point and the trial take products after the inner solve: with
        # a radius of 0.001 the inner solve takes four, the Cauchy point the fifth and sixth, and
        # the first trial the seventh; a non-finite one of them ends the step too
        for N in 5:7
            calls = ProductCount(0)
            nan_on_call = JVPFunction((Jv, x, v, p) -> begin
                calls.n += 1
                Jv .= (T.(D_IND) .+ 3 .* x .^ 2) .* v .+
                      T(KAPPA) .* (v[PERM] .+ v[PERMINV])
                calls.n == N && (Jv .= T(NaN))
                Jv
            end)
            m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
                globalization = DogLeg(; radius = 0.001), jacobian = nan_on_call)
            solver = init(prob, T.(X0_IND), m)
            state, rprob, u, Fu = first_state(solver)
            calls.n = 0
            u₊, F₊, state₊, info = step!!(
                state, solver.method, rprob, copy(u), copy(Fu), nothing)
            @test calls.n == N
            @test info.code == NONFINITE
            @test u₊ == u && F₊ == Fu
            @test state₊.rule.radius == state.rule.radius
        end
    end

    @testset "A step on a state that no start!! saw fails closed, $T" for T in REAL_ELTYPES
        count = ProductCount(0)
        prob = indefinite_problem(Array, T)
        for glob in (nothing, DogLeg(), Steihaug())
            m = glob === nothing ?
                Newton(; linsolve = MINRESQLP(), jacobian = indefinite_jvp(Array, T, count)) :
                Newton(; linsolve = MINRESQLP(), globalization = glob,
                jacobian = indefinite_jvp(Array, T, count))
            solver = init(prob, T.(X0_IND), m)
            st = solver.state
            u = copy(st.u_work)
            Fu = residual!!(similar(u), st.rprob, u, nothing)
            # neither the state of init nor the one solve!! returns has the values of start!!
            for state in (st.inner, (solve!(solver); solver.state.inner))
                count.n = 0
                u₊, _, _, info = step!!(
                    state, solver.method, st.rprob, copy(u), copy(Fu), nothing)
                @test info.code == NONFINITE
                @test u₊ == u
                @test count.n == 0
            end
        end
    end

    @testset "A line search reads the attained tolerance of MINRESQLP, $T" for T in REAL_ELTYPES
        # From 1.3 the Newton step of atan lands where ‖F‖ is 0.94 of the start. The solve along
        # the eigenvector (1, 1, 1) is exact, so the attained tolerance is zero and the inexact
        # test bounds the full step by 1 - c₁ = 0.5, which it fails; the bound of the forcing
        # term's own tolerance, 1 - c₁ (1 - 0.9) = 0.95, would accept it.
        count = ProductCount(0)
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(0.9)),
            globalization = Backtracking(; c₁ = 0.5), jacobian = atan_jvp(count))
        solver = init(atan_problem(Array, T), fill(T(1.3), 3), m)
        state, rprob, u, Fu = first_state(solver)
        newton_length = sqrt(T(3)) * atan(T(1.3)) * (1 + T(1.3)^2)
        @test rnorm(atan.(fill(T(1.3), 3) .- newton_length / sqrt(T(3)))) >
              T(0.5) * rnorm(Fu)
        _, _, _, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS
        @test info.stepnorm < T(0.99) * newton_length
    end

    @testset "A line search records the linear residual of the step it accepted, $T" for T in REAL_ELTYPES
        # one inner iteration from a start off the eigenvector leaves a linear residual of about
        # 0.15 ‖F‖, and Backtracking cuts the step to about half of the direction
        count = ProductCount(0)
        m = Newton(; linsolve = MINRESQLP(; maxiter = 1, forcing = FixedForcing(1e-8)),
            globalization = Backtracking(), jacobian = atan_jvp(count))
        u₀ = T[1.3, 1.5, 1.7]
        solver = init(atan_problem(Array, T), copy(u₀), m)
        state, rprob, u, Fu = first_state(solver)
        F₀ = copy(Fu)
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        d = state₊.ws.x
        α = info.stepnorm / rnorm(d)
        @test info.code == SUCCESS
        @test 0 < α < T(0.9)
        @test rnorm(F₀ .+ d ./ (1 .+ u₀ .^ 2)) > T(0.1) * rnorm(F₀)
        # the residual ‖F + α J d‖, with the product formed here
        @test state₊.ws.linres ≈ rnorm(F₀ .+ α .* d ./ (1 .+ u₀ .^ 2)) rtol = 8 * eps(T)

        # a line search that accepts more than twice the direction: Static(2.5) takes α = 2.5
        m = Newton(; linsolve = MINRESQLP(; maxiter = 1, forcing = FixedForcing(1e-8)),
            globalization = Static(2.5), jacobian = atan_jvp(count))
        solver = init(atan_problem(Array, T), copy(u₀), m)
        state, rprob, u, Fu = first_state(solver)
        F₀ = copy(Fu)
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        d = state₊.ws.x
        α = info.stepnorm / rnorm(d)
        @test 0 < info.stepnorm
        @test α ≈ T(2.5) rtol = 8 * eps(T)
        @test state₊.ws.linres ≈ rnorm(F₀ .+ α .* d ./ (1 .+ u₀ .^ 2)) rtol = 8 * eps(T)

        # a residual norm that rounding puts above ‖F‖ enters as ‖F‖, the bound of Fᴴr = ‖r‖²
        ws = init_minresqlp_workspace(MINRESQLP(), zeros(T, 3))
        GeometricSolvers.accepted_linres!(ws, T(2), one(T), one(T), nextfloat(one(T)))
        @test ws.linres == 1

        # whole solves whose line search extends the step past 2d: for S = Σ cosh xᵢ from 10 the
        # direction is -tanh(10) ≈ -1 per entry, ten directions from the root, so Bisection
        # doubles its step. The fixed step of Static(2.5) maps x to x - 2.5 tanh x, which has the
        # two-cycle ±0.888, so its solve ends with SUCCESS or at its iteration cap.
        Scosh = StationarityProblem((x, p) -> sum(cosh, x))
        for (ls, codes) in ((Bisection(), (SUCCESS,)), (Static(2.5), (SUCCESS, MAXITERS)))
            m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
                globalization = ls)
            x, st = solve(Scosh, fill(T(10), 3), m)
            @test st.code in codes
            @test all(isfinite, x)
        end

        # a direction of length zero keeps the residual of the direction: for `S = ½ x₁²` with
        # `b = (0, 1)` from 0, `F = (0, 1)` lies in the null space of `J`, the solve stops on a
        # zero entry with `d = 0`, and the line search finds no step
        Snull = StationarityProblem((x, p) -> x[1]^2 / 2; b = T[0, 1])
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
            globalization = Backtracking())
        solver = init(Snull, zeros(T, 2), m)
        state, rprob, u, Fu = first_state(solver)
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test iszero(state₊.ws.x) && info.stepnorm == 0
        @test state₊.ws.linres == 1
    end

    @testset "MINRESQLP counts a step that reaches its cap as a step failure, $T" for T in REAL_ELTYPES
        prob = indefinite_problem(Array, T)
        m = Newton(; linsolve = MINRESQLP(; maxiter = 1, forcing = FixedForcing(1e-8)),
            globalization = Backtracking())
        _, st = solve(prob, T.(X0_IND), m)
        @test st.step_failures > 0
        # with Backtracking and room for the inner solve, the solve converges without one
        m2 = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
            globalization = Backtracking())
        x2, st2 = solve(prob, T.(X0_IND), m2)
        @test st2.code == SUCCESS
        @test st2.step_failures == 0
        @test rnorm(x2 .- T.(X_IND)) <= sqrt(eps(T))
    end

    @testset "The radius rules of Steihaug, $T" for T in REAL_ELTYPES
        # without a radius, the first one is the length of the Cauchy step along -F, or ‖F‖ on
        # negative curvature there, and at least 2·10⁻⁶ where that length is below 10⁻⁶
        A = T.(Diagonal([1.0, 2.0]))
        F = T[1, 2]
        _, cg = cg_run(A, F, zero(T))
        @test cg.radius ≈ rnorm(F)^3 / (F' * A * F) rtol = 8 * eps(T)
        _, cg = cg_run(-A, F, zero(T))
        @test cg.radius ≈ rnorm(F) rtol = 8 * eps(T)
        _, cg = cg_run(A, T(1e-8) .* F, zero(T))
        @test cg.radius == T(2e-6)

        # from 1.38 the Newton step of atan lands at -1.36, where ‖F‖ is a little lower: an
        # accepted interior step with a ratio below 0.1 sets the radius to its length
        count = ProductCount(0)
        prob = atan_problem(Array, T)
        newton_length = sqrt(T(3)) * atan(T(1.38)) * (1 + T(1.38)^2)
        for (radius, interior) in ((T(10), true), (T(0.999) * newton_length, false))
            m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
                globalization = Steihaug(; radius), jacobian = atan_jvp(count))
            solver = init(prob, fill(T(1.38), 3), m)
            state, rprob, u, Fu = first_state(solver)
            fnorm₀ = rnorm(Fu)
            _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
            @test info.code == SUCCESS
            @test info.limited == !interior
            # the step is a multiple of -F, so the model residual is below ‖F‖ / 1000 and the
            # predicted reduction is ‖F‖ to that accuracy
            ρ = (fnorm₀ - info.fnorm) / fnorm₀
            @test 0 < ρ < T(0.1)
            if interior
                @test state₊.rule.radius ≈ info.stepnorm rtol = 8 * eps(T)
            else
                # a boundary step with a ratio below 0.1 shrinks the radius fourfold
                @test state₊.rule.radius ≈ radius / 4 rtol = 8 * eps(T)
                @test info.stepnorm ≈ radius rtol = 8 * eps(T)
            end
        end

        # an interior step with a high ratio keeps the radius
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
            globalization = Steihaug(; radius = 10), jacobian = atan_jvp(count))
        solver = init(prob, fill(T(0.3), 3), m)
        state, rprob, u, Fu = first_state(solver)
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS && !info.limited
        @test state₊.rule.radius == T(10)

        # a conjugate-gradient iterate that lands exactly on the boundary stops there, as the
        # Newton step of DogLeg of length δ does: for `S = ½‖x‖² + x₁` from 0 the step is `-e₁`
        # of norm 1, so the radius 1 makes it a boundary step, and its ratio 1 grows the radius
        Sunit = StationarityProblem((x, p) -> sum(x .^ 2) / 2; b = T[1, 0])
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
            globalization = Steihaug(; radius = 1))
        solver = init(Sunit, zeros(T, 2), m)
        state, rprob, u, Fu = first_state(solver)
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS && info.limited
        @test info.stepnorm == 1 && info.fnorm == 0
        @test state₊.rule.radius == T(4)

        # a rejected interior step shorter than the new radius is tried again without a new
        # conjugate-gradient run: from 1.5 with a radius of 100, the trials at 100, 25 and 6.25
        # reuse the first run, and the trial at 1.5625 runs again and stops on the boundary
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-3)),
            globalization = Steihaug(; radius = 100), jacobian = atan_jvp(count))
        solver = init(prob, fill(T(1.5), 3), m)
        state, rprob, u, Fu = first_state(solver)
        count.n = 0
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS
        @test info.stepnorm ≈ T(1.5625) rtol = 8 * eps(T)
        @test count.n == 2

        # an interior step that misses the tolerance at the cap is limited and counts a failure
        Spd = StationarityProblem((x, p) -> (x[1]^2 + 2 * x[2]^2) / 2; b = T[1, 4])
        m = Newton(; linsolve = MINRESQLP(; maxiter = 1, forcing = FixedForcing(1e-8)),
            globalization = Steihaug(; radius = 100))
        solver = init(Spd, zeros(T, 2), m)
        state, rprob, u, Fu = first_state(solver)
        _, _, _, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS
        @test info.limited
        @test info.failures == 1

        # negative curvature along -F predicts an increase of ‖F‖ at every radius
        Sneg = StationarityProblem((x, p) -> -sum(x .^ 2) / 2; b = T[1, 2])
        m = Newton(; linsolve = MINRESQLP(), globalization = Steihaug(; radius = 1))
        solver = init(Sneg, zeros(T, 2), m)
        state, rprob, u, Fu = first_state(solver)
        u₊, F₊, state₊, info = step!!(
            state, solver.method, rprob, copy(u), copy(Fu), nothing)
        @test info.code == LINESEARCH_FAILED
        @test u₊ == u && F₊ == Fu
        @test state₊.rule.radius == T(1e-6)

        # a trial whose model predicts an increase is rejected even where ‖F‖ falls: for
        # `S = -cos x` from 2.5 the step along -F has negative curvature, and the trial at -2.4
        # lands at 0.1, where |sin| is lower, but `pred < 0`
        Scos = StationarityProblem((x, p) -> -sum(cos.(x)))
        m = Newton(; linsolve = MINRESQLP(), globalization = Steihaug(; radius = 2.4))
        solver = init(Scos, T[2.5], m)
        state, rprob, u, Fu = first_state(solver)
        @test abs(sin(T(0.1))) < abs(sin(T(2.5)))
        u₊, _, _, info = step!!(state, solver.method, rprob, copy(u), copy(Fu), nothing)
        @test info.code == LINESEARCH_FAILED
        @test u₊ == u

        # at a root the step keeps the iterate and the radius, and takes no product
        Sroot = StationarityProblem((x, p) -> sum(x .^ 2) / 2)
        m = Newton(; linsolve = MINRESQLP(), globalization = Steihaug(; radius = 3),
            jacobian = atan_jvp(count))
        solver = init(Sroot, zeros(T, 2), m)
        state, rprob, u, Fu = first_state(solver)
        count.n = 0
        u₊, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS && info.fnorm == 0
        @test count.n == 0
        @test state₊.rule.radius == T(3)
    end

    @testset "Clause 3: the Steihaug step stops on the boundary on negative curvature, $AT, $T" for T in REAL_ELTYPES,
        AT in ARRAY_BACKENDS

        # the first direction -F has positive curvature, the second negative curvature
        A = AT(T.(Diagonal(H_NEG)))
        F = AT(T[1, 1 // 100])
        for δ in (T(2), T(5), T(50))
            g, cg = cg_run(A, F, δ)
            @test cg.finite && !cg.interior
            @test cg.iterations == 2
            @test abs(rnorm(g.d) - δ) <= 8 * eps(T) * δ
            # the residual of the model is F + A d
            @test rnorm(g.r .- (F .+ A * g.d)) <= 8 * eps(T) * (rnorm(F) + rnorm(A * g.d))
        end
        # on negative curvature in the first direction, the step goes along -F to the boundary
        Aneg = AT(T.(Diagonal([-1.0, -2.0])))
        g, cg = cg_run(Aneg, AT(T[1, 2]), T(3))
        @test cg.iterations == 1 && !cg.interior
        @test abs(rnorm(g.d) - T(3)) <= 8 * eps(T) * T(3)
        @test Array(g.d) ≈ -T(3) .* T[1, 2] ./ sqrt(T(5)) rtol = 8 * eps(T)
        # a positive definite system converges inside a large radius
        Apos = AT(T.(Diagonal([1.0, 2.0])))
        g, cg = cg_run(Apos, AT(T[1, 2]), T(50); tol = sqrt(eps(T)))
        @test cg.interior && cg.converged
        @test Array(g.d) ≈ T[-1, -1] rtol = sqrt(eps(T))
    end

    @testset "Clause 3: the Newton step of Steihaug on negative curvature, $T" for T in REAL_ELTYPES
        count = ProductCount(0)
        prob = negative_curvature_problem(Array, T)
        jvp = JVPFunction((Jv, x, v, p) -> (count.n += 1; Jv .= T.(H_NEG) .* v; Jv))
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
            globalization = Steihaug(; radius = 2), jacobian = jvp)
        solver = init(prob, zeros(T, 2), m)
        state, rprob, u, Fu = first_state(solver)
        fnorm₀ = rnorm(Fu)
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        @test info.code == SUCCESS
        @test info.limited
        @test abs(info.stepnorm - T(2)) <= 8 * eps(T) * T(2)
        @test info.fnorm < fnorm₀
        # an accepted boundary step with a ratio above 0.75 grows the radius fourfold
        @test state₊.rule.radius == T(8)
    end

    @testset "Steihaug never increases ‖F‖ over accepted steps, $AT, $T" for T in REAL_ELTYPES,
        AT in ARRAY_BACKENDS

        count = ProductCount(0)
        prob = atan_problem(AT, T)
        # the full Newton step from 1.5 lies inside the radius 10, and it increases ‖F‖
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
            globalization = Steihaug(; radius = 10), jacobian = atan_jvp(count))
        x₀ = AT(fill(T(1.5), 3))
        @test rnorm(atan.(x₀ .- atan.(x₀) .* (1 .+ x₀ .^ 2))) > rnorm(atan.(x₀))
        solver = init(prob, x₀, m)
        state, rprob, u, Fu = first_state(solver)
        norms = T[rnorm(Fu)]
        for _ in 1:30
            u, Fu, state, info = step!!(state, solver.method, rprob, u, Fu, nothing)
            @test info.code == SUCCESS
            push!(norms, info.fnorm)
            info.fnorm <= sqrt(eps(T)) * norms[1] && break
        end
        @test all(norms[2:end] .<= norms[1:(end - 1)])
        @test norms[end] <= sqrt(eps(T)) * norms[1]
        # the solve converges, and its count is the count of the products
        count.n = 0
        x, st = solve(prob, AT(fill(T(1.5), 3)), m)
        @test st.code == SUCCESS
        @test rnorm(x) <= sqrt(eps(T))
        @test st.jvp_calls == count.n
    end

    @testset "Steihaug at a scale of S whose cube overflows, $T" for T in REAL_ELTYPES
        # with S scaled by s, F and J scale by s, so pᴴ J p of a direction p of the scale of F is
        # of the scale s³, above floatmax(T), while every product is finite
        s = floatmax(T)^T(0.4)
        @test isinf(s^3) && isfinite(s^2)
        count = ProductCount(0)
        prob = StationarityProblem((x, p) -> s * sum(x .* atan.(x) .- log1p.(x .^ 2) ./ 2))
        jvp = JVPFunction((Jv, x, v, p) -> (
            count.n += 1; Jv .= s .* v ./ (1 .+ x .^ 2); Jv))
        for glob in (Steihaug(), DogLeg())
            m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-8)),
                globalization = glob, jacobian = jvp)
            x, st = solve(prob, fill(T(1.5), 3), m)
            @test st.code == SUCCESS
            @test rnorm(x) <= sqrt(eps(T))
        end
    end

    @testset "Clause 4: Newton with GMRES on a StationarityProblem with E ≠ U, $AT, $T" for T in REAL_ELTYPES,
        AT in ARRAY_BACKENDS

        count = ProductCount(0)
        prob = nonsymmetric_problem(AT, T)
        x₀ = AT(T[0.7, -1.3, 0.6, -0.3])
        m = Newton(; linsolve = GMRES(; forcing = FixedForcing(1e-6)),
            jacobian = nonsymmetric_jvp(AT, T, count))
        x, st = solve(prob, copy(x₀), m)
        @test st.code == SUCCESS
        @test rnorm(Array(x)[3:4] .- T.(U_NONSYM)) <= sqrt(eps(T))
        # the entries outside U stay
        @test Array(x)[1:2] == T[0.7, -1.3]
        @test st.jvp_calls == count.n
        @test st.jvp_calls > 0
        # the same problem with the product of the AD seam
        if AT === Array
            x2, st2 = solve(prob, copy(x₀), Newton(; linsolve = GMRES()))
            @test st2.code == SUCCESS
            @test rnorm(x2[3:4] .- T.(U_NONSYM)) <= sqrt(eps(T))
            @test st2.jvp_calls > 0
            # and MINRESQLP refuses it, because its Jacobian is not symmetric
            @test_throws r"^ArgumentError: a Newton with MINRESQLP solves the symmetric system of a StationarityProblem with E == U, the same indices in the same order; got E = \[1, 2\] and U = \[3, 4\], whose Jacobian is not symmetric: use GMRES\(\)$" solve(
                prob, copy(x₀), Newton(; linsolve = MINRESQLP()))
        end
    end

    @testset "GMRES on a StationarityProblem refuses a jvp_eltype" begin
        prob = nonsymmetric_problem(Array, Float64)
        @test_throws r"^ArgumentError: a StationarityProblem takes a GMRES with jvp_eltype = Nothing: its Hessian–vector product is formed in the element type of the state$" solve(
            prob, [0.7, -1.3, 0.6, -0.3], Newton(;
                linsolve = GMRES(; jvp_eltype = Float32)))
    end

    @testset "Clause 5: DogLeg with MINRESQLP, the Cauchy products counted, $T" for T in REAL_ELTYPES
        count = ProductCount(0)
        prob = indefinite_problem(Array, T)
        δ = T(1 // 100)
        m = Newton(; linsolve = MINRESQLP(; forcing = FixedForcing(1e-6)),
            globalization = DogLeg(; radius = δ), jacobian = indefinite_jvp(Array, T, count))
        solver = init(prob, T.(X0_IND), m)
        state, rprob, u, Fu = first_state(solver)
        count.n = 0
        _, _, state₊, info = step!!(state, solver.method, rprob, u, Fu, nothing)
        # the Newton step is longer than δ, so the step lies on the boundary
        @test rnorm(solver.state.inner.d) > δ
        @test info.code == SUCCESS
        @test info.limited
        @test abs(info.stepnorm - δ) <= 8 * eps(T) * δ
        # the inner iterations, the two products of the Cauchy point and one per trial
        @test state₊.ws.jvp_calls == count.n
        @test count.n >= state₊.ws.iterations + 2
        # a solve converges and counts every product
        count.n = 0
        x, st = solve(prob, T.(X0_IND), m)
        @test st.code == SUCCESS
        @test rnorm(x .- T.(X_IND)) <= sqrt(eps(T))
        @test st.jvp_calls == count.n
    end
end
