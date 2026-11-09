using Adapt: Adapt
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: diagm, norm
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: AutoForwardDiff, Backtracking, Deflated, LeastSquaresProblem,
                        NONFINITE,
                        Newton, NonlinearProblem, Options, PerSystem, Reparametrized,
                        SolverStatus,
                        StationarityProblem, ToReal, init, init_state, jacobian!!, jvp!!,
                        prepare_ad, residual!!, solve, solve!, solve!!, solve_batch!

const GS = GeometricSolvers

include("../helpers/matrix.jl")

allowscalar(false)

const BT = Newton(; globalization = Backtracking())

# ─── The test problems of §2, and the test's own closed forms ────────────────

# P1: the scalar cubic, with the roots -1, 0 and 1.
p1_problem() = NonlinearProblem((x, p) -> x^3 - x)

# P2: two quadratics, with the roots (±1, ±2), and its Jacobian.
function p2_f(x, p)
    T = eltype(x)
    return SVector(x[1]^2 - T(1), x[2]^2 - T(4))
end

function p2_j(x, p)
    T = eltype(x)
    return SMatrix{2, 2, T, 4}(2 * x[1], zero(T), zero(T), 2 * x[2])
end

p2_problem() = NonlinearProblem(p2_f, p2_j)
p2_problem_ad() = NonlinearProblem(p2_f)

# P2 in R3: the offsets are the parameter `p`, an array of the iterate's storage.
p2_r3!(y, x, p) = (y .= x .^ 2 .- p; y)
p2_problem_r3() = NonlinearProblem(p2_r3!)

# The out-of-place residual of P2's R3 problem, for the independent reference.
r3_ref(y, q) = [y[1]^2 - q[1], y[2]^2 - q[2]]

# P3: the action a⁴/4 - a²/2 + c²(1 + a²)/2, whose stationary points (0, 0), (±1, 0) hold the dead
# branch a = 0, and its gradient.
p3_action(x, p) = x[1]^4 / 4 - x[1]^2 / 2 + x[2]^2 * (1 + x[1]^2) / 2
p3_problem() = StationarityProblem(p3_action)

function p3_residual(x, p)
    return SVector(x[1]^3 - x[1] + x[2]^2 * x[1], x[2] * (1 + x[1]^2))
end

# P4: the same action with the weights (s₁, s₂) as the fixed complement of the state, and the
# unknowns (a, c) = x[(1, 2)].
function p4_action(x, p)
    return x[3] * (x[1]^4 / 4 - x[1]^2 / 2) + x[4] * x[2]^2 * (1 + x[1]^2) / 2
end

p4_problem() = StationarityProblem(p4_action; E = (1, 2), U = (1, 2))
p4_unknowns(x) = x[SVector(1, 2)]

# P5: the edges of the cubic in the first variable, with the root (0, 0) at the origin, so that an
# offset of 1e-100 from that root survives in `Float64`.
p5_problem() = NonlinearProblem((x, p) -> SVector(x[1]^3 - x[1], x[2]))

# The gradients of the two stationarity actions on an array iterate, for the residual test of an R3
# solve: the test's own forms, independent of the code under test.
p3_gradient(x) = [x[1]^3 - x[1] + x[2]^2 * x[1], x[2] * (1 + x[1]^2)]
p4_gradient(x) = [x[3] * (x[1]^3 - x[1]) + x[4] * x[2]^2 * x[1], x[4] * x[2] * (1 + x[1]^2)]

# The test's own closed form of `M(x)` and its gradient: neither reads the code under test.
mass(x, roots, power, shift) = prod(r -> norm(x - r)^(-power) + shift, roots)
function mass_gradient(x, roots, power, shift)
    ForwardDiff.gradient(y -> mass(y, roots, power, shift), x)
end

# The residual norm of a point of the undeflated problem, and the pairwise distance of roots on the
# unknowns of the problem.
residual_norm(prob, x) = norm(residual!!(nothing, prob, x, nothing))

function pairwise_distinct(roots, dist = identity)
    n = length(roots)
    return all(norm(dist(roots[i]) - dist(roots[j])) > 1 / 2
    for i in 1:n for j in (i + 1):n)
end

# One solve: its root and its status.
function solve_root(prob, x0, method = BT, p = nothing; kwargs...)
    x, status = solve(prob, x0, method, p; kwargs...)
    return x, status
end

# The bound of clause 2: the two values are the same sum of two products in another order, so the
# difference is a small multiple of `eps(T)` times the size of the two terms, not of their sum,
# which can cancel.
function jacobian_bound(T, M, J, F, gM)
    return 64 * eps(T) * (M * norm(J) + norm(F) * norm(gM))
end

# The residual tolerance each precision reaches on these problems, and the tolerance every solve of
# clause 1 is asked for: a Newton step is quadratic, so the residual falls to the roundoff floor of
# the problem in `Float64` and to a few multiples of `√eps(Float32) ≈ 3.5e-4` in `Float32`.
residual_atol(::Type{Float64}) = 1e-8
residual_atol(::Type{Float32}) = 1e-3

# The keywords that make a solve stop on the absolute residual test alone, so that `‖F(r)‖` is
# bounded by the tolerance the solve was given.
residual_options(::Type{T}) where {T} = (f_abstol = residual_atol(T), f_reltol = zero(T))

# ─── Clause 1: sequential multi-root discovery from one start ────────────────

@testset "Clause 1: three roots from one start, P1 Number ($T)" for T in REAL_ELTYPES
    prob = p1_problem()
    x0 = T(0.1)
    atol = residual_atol(T)
    opts = residual_options(T)
    r1, s1 = solve_root(prob, x0; opts...)
    r2, s2 = solve_root(Deflated(prob, (r1,)), x0; opts...)
    r3, s3 = solve_root(Deflated(prob, (r1, r2)), x0; opts...)
    roots = (r1, r2, r3)
    @test all(GS.isconverged, (s1, s2, s3))
    @test pairwise_distinct(roots)
    @test all(r -> abs(abs(r) - 1) <= atol || abs(r) <= atol, roots)
    @test all(r -> residual_norm(prob, r) <= atol, roots)
end

@testset "Clause 1: three roots from one start, P2 SVector ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    x0 = SVector{2, T}(T(0.8), T(1.7))
    atol = residual_atol(T)
    opts = residual_options(T)
    r1, s1 = solve_root(prob, x0; opts...)
    r2, s2 = solve_root(Deflated(prob, (r1,)), x0; opts...)
    r3, s3 = solve_root(Deflated(prob, (r1, r2)), x0; opts...)
    roots = (r1, r2, r3)
    @test all(GS.isconverged, (s1, s2, s3))
    @test pairwise_distinct(roots)
    @test all(r -> residual_norm(prob, r) <= atol, roots)
    @test all(r -> abs(r[1]^2 - 1) <= atol && abs(r[2]^2 - 4) <= atol, roots)
end

@testset "Clause 1: two roots from one start, P5 SVector ($T)" for T in REAL_ELTYPES
    prob = p5_problem()
    x0 = SVector{2, T}(T(0.8), T(0.1))
    atol = residual_atol(T)
    opts = residual_options(T)
    r1, s1 = solve_root(prob, x0; opts...)
    r2, s2 = solve_root(Deflated(prob, (r1,)), x0; opts...)
    @test all(GS.isconverged, (s1, s2))
    @test norm(r1 - r2) > 1 / 2
    @test all(r -> residual_norm(prob, r) <= atol, (r1, r2))
end

@testset "Clause 1: three roots from one start, P3 StationarityProblem ($T)" for T in REAL_ELTYPES
    prob = p3_problem()
    x0 = SVector{2, T}(T(0.1), T(0.3))
    atol = residual_atol(T)
    opts = residual_options(T)
    r1, s1 = solve_root(prob, x0; opts...)
    r2, s2 = solve_root(Deflated(prob, (r1,)), x0; opts...)
    r3, s3 = solve_root(Deflated(prob, (r1, r2)), x0; opts...)
    roots = (r1, r2, r3)
    @test all(GS.isconverged, (s1, s2, s3))
    @test pairwise_distinct(roots)
    @test all(r -> residual_norm(prob, r) <= atol, roots)
    # the dead branch a = 0 and the two live branches a = ±1
    @test any(r -> abs(r[1]) <= atol, roots)
    live = filter(r -> abs(r[1]) > T(0.5), collect(roots))
    @test length(live) == 2
    @test all(r -> abs(abs(r[1]) - 1) <= atol, live)
end

@testset "Clause 1: P4, the distance on the unknowns of a StationarityProblem ($T)" for T in REAL_ELTYPES
    prob = p4_problem()
    x0 = SVector{4, T}(T(0.1), T(0.3), T(1), T(2))
    atol = residual_atol(T)
    opts = residual_options(T)
    r1, s1 = solve_root(prob, x0; opts...)
    r2, s2 = solve_root(Deflated(prob, (r1,)), x0; opts...)
    r3, s3 = solve_root(Deflated(prob, (r1, r2)), x0; opts...)
    roots = (r1, r2, r3)
    @test all(GS.isconverged, (s1, s2, s3))
    @test pairwise_distinct(roots, p4_unknowns)
    @test all(r -> residual_norm(prob, r) <= atol, roots)
    # the weights are fixed components of the state and stay at their start
    @test all(r -> r[3] == T(1) && r[4] == T(2), roots)
end

@testset "Clause 1 and edge 12: P3 and P4 on Array, the distance on the unknowns ($T)" for T in REAL_ELTYPES
    atol = residual_atol(T)
    opts = residual_options(T)
    # P3 on Array: the unknowns are the whole state, so the roots are full states too
    p3 = p3_problem()
    x03 = T[0.1, 0.3]
    s1, st1 = solve(p3, copy(x03), BT; opts...)
    s2, st2 = solve(Deflated(p3, [copy(s1)]), copy(x03), BT; opts...)
    s3, st3 = solve(Deflated(p3, [copy(s1), copy(s2)]), copy(x03), BT; opts...)
    roots3 = (s1, s2, s3)
    @test all(GS.isconverged, (st1, st2, st3))
    @test pairwise_distinct(roots3)
    @test all(r -> norm(p3_gradient(r)) <= atol, roots3)
    # P4 on Array: two unknowns of a four-entry state, and the weights stay put
    p4 = p4_problem()
    x04 = T[0.1, 0.3, 1, 2]
    a1, t1 = solve(p4, copy(x04), BT; opts...)
    a2, t2 = solve(Deflated(p4, [copy(a1)]), copy(x04), BT; opts...)
    a3, t3 = solve(Deflated(p4, [copy(a1), copy(a2)]), copy(x04), BT; opts...)
    roots4 = (a1, a2, a3)
    @test all(GS.isconverged, (t1, t2, t3))
    @test pairwise_distinct(roots4, r -> r[1:2])
    @test all(r -> norm(p4_gradient(r)) <= atol, roots4)
    @test all(r -> r[3] == T(1) && r[4] == T(2), roots4)
    # the datum reaches an R3 solve, and one of another length is refused
    b = T[0.05, -0.02]
    prob_b = StationarityProblem(p3_action; b = b)
    rb, stb = solve(Deflated(prob_b, [copy(s1)]), copy(x03), BT; b = b, opts...)
    @test GS.isconverged(stb)
    @test norm(rb - s1) > 1 / 2
    @test_throws ArgumentError solve(
        Deflated(prob_b, [copy(s1)]), copy(x03), BT; b = T[0.1, 0.2, 0.3])
end

@testset "Clause 1 and edge 10: P2 on $AT under allowscalar(false), ($AT, $T)" for AT in ARRAY_BACKENDS,
    T in REAL_ELTYPES

    prob = p2_problem_r3()
    AT === JLArray && @test_throws ErrorException AT(zeros(T, 2))[1]
    p = AT(T[1, 4])
    x0 = AT(T[0.8, 1.7])
    atol = residual_atol(T)
    opts = residual_options(T)
    r1, s1 = solve_root(prob, copy(x0), BT, p; opts...)
    r2, s2 = solve_root(Deflated(prob, [Array(r1)]), copy(x0), BT, p; opts...)
    r3, s3 = solve_root(Deflated(prob, [Array(r1), Array(r2)]), copy(x0), BT, p; opts...)
    @test all(GS.isconverged, (s1, s2, s3))
    @test norm(Array(r1) - Array(r2)) > 1 / 2
    @test norm(Array(r1) - Array(r3)) > 1 / 2
    @test norm(Array(r2) - Array(r3)) > 1 / 2
    @test all(r -> norm(r3_ref(Array(r), Array(p))) <= atol, (r1, r2, r3))
end

# ─── Clause 1: R2 and edge 11, one Deflated for a whole batch ───────────────

@testset "Clause 1 and edge 11: one Deflated for a batch ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    r1, _ = solve_root(prob, SVector{2, T}(T(0.8), T(1.7)))
    def = Deflated(prob, (r1,))
    X = fill(SVector{2, T}(T(0.8), T(1.7)), 4)
    S = Vector{SolverStatus{T}}(undef, 4)
    solve_batch!(X, S, def, BT)
    @test all(GS.isconverged, S)
    # every system starts at the same iterate and sees the same roots, so all four end at the one
    # root the deflation leaves them
    @test all(x -> norm(x - r1) > 1 / 2, X)
    @test all(x -> x == X[1], X)
    # the tally form counts the same four successes
    X2 = fill(SVector{2, T}(T(0.8), T(1.7)), 4)
    tally = solve_batch!(X2, def, BT)
    @test tally.SUCCESS == 4
    @test all(x -> norm(x - r1) > 1 / 2, X2)
    # a status vector of another real type, or of another length, is refused as it is for the problem
    other = T === Float32 ? Float64 : Float32
    @test_throws ArgumentError solve_batch!(
        X2, Vector{SolverStatus{other}}(undef, 4), def, BT)
    @test_throws DimensionMismatch solve_batch!(
        X2, Vector{SolverStatus{T}}(undef, 3), def, BT)
    # edge 11: the root storage of an R3 iterate is not `isbits`, and a kernel takes the problem by
    # value
    @test !isbits(Deflated(prob, [[T(1), T(2)]]))
end

# ─── Clause 2: the product-rule Jacobian and the JVP ─────────────────────────

@testset "Clause 2: P2 R1 Jacobian and JVP against the reference ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    xt = SVector{2, T}(T(0.4), T(1.3))
    vt = SVector{2, T}(T(0.7), T(-0.3))
    Fx = p2_f(xt, nothing)
    Jx = p2_j(xt, nothing)
    for roots in ((SVector{2, T}(1, 2),),
        (SVector{2, T}(1, 2), SVector{2, T}(-1, -2)))
        def = Deflated(prob, roots)
        M = mass(xt, roots, 2, 1)
        gM = mass_gradient(xt, roots, 2, 1)
        Jref = ForwardDiff.jacobian(y -> mass(y, roots, 2, 1) * p2_f(y, nothing), xt)
        bound = jacobian_bound(T, M, Jx, Fx, gM)
        for prep in (nothing, AutoForwardDiff())
            @test norm(Matrix(jacobian!!(nothing, prep, def, xt, nothing)) - Jref) <= bound
            @test norm(Vector(jvp!!(nothing, prep, def, xt, vt, nothing)) - Jref * vt) <=
                  bound * norm(vt)
        end
        # edge 5: the cross terms of the product rule appear only for k >= 2, and the factor and
        # the gradient the code forms are the ones the reference composes from the single roots
        if length(roots) == 2
            M1 = mass(xt, (roots[1],), 2, 1)
            M2 = mass(xt, (roots[2],), 2, 1)
            @test isapprox(residual!!(nothing, def, xt, nothing), M1 * M2 * Fx; rtol = 1e-6)
            @test norm(
                Matrix(jacobian!!(nothing, AutoForwardDiff(), def, xt, nothing)) -
                (M * Jx + Fx * transpose(gM))) <= bound
        end
    end
end

@testset "Clause 2: P3 Jacobian and JVP through the fused gradient_hessian!! ($T)" for T in REAL_ELTYPES
    prob = p3_problem()
    xt = SVector{2, T}(T(0.4), T(1.3))
    vt = SVector{2, T}(T(0.7), T(-0.3))
    roots = (SVector{2, T}(0, 0),)
    def = Deflated(prob, roots)
    M = mass(xt, roots, 2, 1)
    gM = mass_gradient(xt, roots, 2, 1)
    Jx = ForwardDiff.jacobian(y -> p3_residual(y, nothing), xt)
    Fx = p3_residual(xt, nothing)
    Jref = ForwardDiff.jacobian(y -> mass(y, roots, 2, 1) * p3_residual(y, nothing), xt)
    bound = jacobian_bound(T, M, Jx, Fx, gM)
    @test norm(Matrix(jacobian!!(nothing, AutoForwardDiff(), def, xt, nothing)) - Jref) <=
          bound
    @test norm(Vector(jvp!!(nothing, AutoForwardDiff(), def, xt, vt, nothing)) -
               Jref * vt) <=
          bound * norm(vt)
end

@testset "Clause 2: P2 R3 Jacobian and JVP against the reference ($AT, $T)" for AT in ARRAY_BACKENDS,
    T in REAL_ELTYPES

    prob = p2_problem_r3()
    p = AT(T[1, 4])
    xt = AT(T[0.4, 1.3])
    vt = AT(T[0.7, -0.3])
    for roots in ([[T(1), T(2)]], [[T(1), T(2)], [T(-1), T(-2)]])
        host_roots = [copy(r) for r in roots]
        def = Deflated(prob, roots)
        F = similar(xt)
        prep = prepare_ad(AutoForwardDiff(), def, F, xt, p)
        J = similar(xt, T, 2, 2)
        M = mass(Array(xt), host_roots, 2, 1)
        gM = mass_gradient(Array(xt), host_roots, 2, 1)
        Jx = ForwardDiff.jacobian(y -> r3_ref(y, Array(p)), Array(xt))
        Fx = r3_ref(Array(xt), Array(p))
        Jref = ForwardDiff.jacobian(
            y -> mass(y, host_roots, 2, 1) * r3_ref(y, Array(p)), Array(xt))
        bound = jacobian_bound(T, M, Jx, Fx, gM)
        @test norm(Array(jacobian!!(J, prep, def, xt, p)) - Jref) <= bound
        Jv = similar(xt)
        @test norm(Array(jvp!!(Jv, prep, def, xt, vt, p)) - Jref * Array(vt)) <=
              bound * norm(Array(vt))
        Fd = similar(xt)
        @test norm(Array(residual!!(Fd, prep, def, xt, p)) - M * Fx) <=
              64 * eps(T) * norm(M * Fx)
        # edge 5 in R3: the cross terms of the product rule appear only for k >= 2
        if length(roots) == 2
            @test norm(Array(jacobian!!(J, prep, def, xt, p)) -
                       (M * Jx + Fx * transpose(gM))) <=
                  bound
        end
    end
end

# The inner problem's own in-place Jacobian: the deflated Jacobian writes it into the caller's
# matrix and completes it there, and the product reads it in the matrix the preparation reserves.
function p2_r3_jac!(J, x, p)
    J[1, 1] = 2 * x[1]
    J[1, 2] = zero(eltype(J))
    J[2, 1] = zero(eltype(J))
    J[2, 2] = 2 * x[2]
    return J
end

@testset "Clause 2: an inner analytic Jacobian in R3 ($T)" for T in REAL_ELTYPES
    prob = NonlinearProblem(p2_r3!, p2_r3_jac!)
    p = T[1, 4]
    x0 = T[0.8, 1.7]
    def = Deflated(prob, [[T(1), T(2)]])
    x, status = solve(def, x0, BT, p)
    @test GS.isconverged(status)
    @test norm(x - T[-1, -2]) < 1 / 2
    # the Jacobian and the product at a point, against the same closed-form reference
    xt = T[0.4, 1.3]
    vt = T[0.7, -0.3]
    F = similar(xt)
    prep = prepare_ad(AutoForwardDiff(), def, F, xt, p)
    J = similar(xt, T, 2, 2)
    Jref = ForwardDiff.jacobian(
        y -> mass(y, [[T(1), T(2)]], 2, 1) * r3_ref(y, Array(p)), Array(xt))
    @test norm(Array(jacobian!!(J, prep, def, xt, p)) - Jref) <=
          64 * eps(T) * norm(Jref)
    Jv = similar(xt)
    @test norm(Array(jvp!!(Jv, prep, def, xt, vt, p)) - Jref * Array(vt)) <=
          64 * eps(T) * norm(Jref) * norm(Array(vt))
end

# ─── Clause 3 and the decided edges 1, 2, 3, 4 and 6 ────────────────────────

@testset "Clause 3 and edge 1: a start at a deflated root ($T)" for T in REAL_ELTYPES
    r1 = SVector{2, T}(1, 2)
    for m in (Newton(), BT)
        x, status = solve(Deflated(p2_problem(), (r1,)), r1, m)
        @test status.code == NONFINITE
        @test status.iterations == 0
        @test x == r1
    end
    x1, s1 = solve(Deflated(p1_problem(), (zero(T),)), zero(T), Newton())
    @test s1.code == NONFINITE
    @test s1.iterations == 0
    @test x1 == zero(T)
    # edge 1 on an array iterate of both backends, where the factor is `Inf` in the R3 workspace
    for AT in ARRAY_BACKENDS
        p = AT(T[1, 4])
        root = AT(T[1, 2])
        def = Deflated(p2_problem_r3(), [Array(root)])
        x = copy(root)
        xa, status = solve(def, x, BT, p)
        @test status.code == NONFINITE
        @test status.iterations == 0
        @test Array(xa) == Array(root)
    end
end

@testset "Edge 2: an overflowing d^-p gives NONFINITE at 0 iterations ($T)" for T in REAL_ELTYPES
    # the exponent whose `d^-2` overflows in this precision: 1e-200 in `Float64`, and 1e-25 in
    # `Float32`, whose `floatmin` is 1e-38
    e = T === Float32 ? 25 : 200
    x0 = T(1) / T(10)^e
    @test x0 > 0
    @test x0^(-2) == T(Inf)
    def = Deflated(p1_problem(), (zero(T),))
    x, status = solve(def, x0, Newton())
    @test status.code == NONFINITE
    @test status.iterations == 0
    @test x == x0
    @test !isfinite(residual!!(nothing, def, x0, nothing))
end

@testset "Edge 3 and MU4: P5 near the origin is finite, with no NaN" begin
    root = SVector{2, Float64}(0, 0)
    def = Deflated(p5_problem(), (root,))
    # the three decided points: the last two decades in which `∇mᵢ` is finite in `Float64`, and
    # the corresponding point in `Float32`
    for (T, d) in ((Float64, 1e-100), (Float64, 1e-102), (Float32, 1.0f-12))
        defT = T === Float64 ? def : Deflated(p5_problem(), (SVector{2, T}(0, 0),))
        x = SVector{2, T}(T(d), zero(T))
        F = residual!!(nothing, defT, x, nothing)
        J = jacobian!!(nothing, AutoForwardDiff(), defT, x, nothing)
        @test all(isfinite, F)
        @test all(isfinite, J)
        @test !any(isnan, J)
        # the unit-scaled gradient: `M` is `d^{-2}` and the first coordinate of `∇M` is `-2d^{-3}`,
        # where the unscaled `ϕ/d²` overflows and `Inf * 0.0` gives `NaN` for the second
        @test isapprox(J[1, 1], inv(T(d))^2; rtol = 1e-5)
        Jv = jvp!!(nothing, AutoForwardDiff(), defT, x, SVector{2, T}(1, 0), nothing)
        @test all(isfinite, Jv)
    end
    # the test's own closed form of `M` at the first point: 1e200, which the code agrees with
    x = SVector{2, Float64}(1e-100, 0)
    @test isapprox(mass(x, (root,), 2, 1), 1e200; rtol = 1e-6)
    @test isapprox(residual!!(nothing, def, x, nothing),
        1e200 * SVector(x[1]^3 - x[1], x[2]); rtol = 1e-6)
end

@testset "Edge 4 and MU3: a distant root gives shift * F to rtol 1e-5 ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    x_far = SVector{2, T}(T(1000), T(2000))
    F = p2_f(x_far, nothing)
    for shift in (T(2.5), T(1))
        def = Deflated(prob, (SVector{2, T}(1, 2),); shift = shift)
        @test isapprox(residual!!(nothing, def, x_far, nothing), shift * F; rtol = 1e-5)
    end
    def2 = Deflated(
        prob, (SVector{2, T}(1, 2), SVector{2, T}(-1, -2)); shift = T(2.5))
    @test isapprox(residual!!(nothing, def2, x_far, nothing), T(2.5)^2 * F; rtol = 1e-5)
end

@testset "Edge 6: a NaN in F gives NONFINITE at 0 iterations ($T)" for T in REAL_ELTYPES
    prob = NonlinearProblem((x, p) -> SVector(x[1]^2 - 1, x[2] / x[2] - 1))
    _, status = solve(
        Deflated(prob, (SVector{2, T}(1, 2),)), SVector{2, T}(T(0.8), zero(T)), Newton())
    @test status.code == NONFINITE
    @test status.iterations == 0
end

# ─── Edges 7, 8a, 13: the constructor's validation ──────────────────────────

@testset "Edges 7, 8a, 13 and MU7: the constructor validates" begin
    prob = p2_problem()
    r = SVector(1.0, 2.0)
    @test_throws ArgumentError Deflated(prob, ())
    @test_throws ArgumentError Deflated(prob, SVector{0, Float64}[])
    @test_throws ArgumentError Deflated(prob, (r,); power = 0.0)
    @test_throws ArgumentError Deflated(prob, (r,); power = -2.0)
    @test_throws ArgumentError Deflated(prob, (r,); power = Inf)
    @test_throws ArgumentError Deflated(prob, (r,); power = NaN)
    @test_throws ArgumentError Deflated(prob, (r,); shift = 0.0)
    @test_throws ArgumentError Deflated(prob, (r,); shift = Inf)
    @test_throws ArgumentError Deflated(prob, (SVector(1.0, NaN),))
    @test_throws ArgumentError Deflated(prob, (SVector(1.0, Inf),))
    @test_throws ArgumentError Deflated(prob, (SVector(Inf, 2.0),))
    @test_throws ArgumentError Deflated(prob, (r, SVector(1.0, 2.0, 3.0)))
    @test_throws ArgumentError Deflated(prob, [1.0, 2.0])
    @test_throws ArgumentError Deflated(prob, (r,), 0.0, 1.0)
    @test_throws ArgumentError Deflated(prob, (r,), 2.0, 0.0)
    # a root set of mixed element types or of mixed storage has no iterate it can serve
    @test_throws ArgumentError Deflated(
        prob, (SVector{2, Float32}(1, 2), SVector{2, Float64}(-1, -2)))
    @test_throws ArgumentError Deflated(prob, (1.0, SVector(1.0)))
    # MU7: the inner constructor's own refusals, which the outer one reaches first for an empty
    # root set: `_root_float_type` reads `first(roots)` before the constructor validates
    @test_throws ArgumentError Deflated{
        typeof(prob), Vector{Vector{Float64}}, Float64}(
        prob, Vector{Float64}[], 2.0, 1.0)
    @test_throws ArgumentError Deflated{
        typeof(prob), Tuple{SVector{2, Float64}}, Float64}(prob, (r,), 0.0, 1.0)
    @test_throws ArgumentError Deflated{
        typeof(prob), Tuple{SVector{2, Float64}}, Float64}(
        prob, (SVector(1.0, NaN),), 2.0, 1.0)
    # edge 13: an inner problem that is neither a NonlinearProblem nor a StationarityProblem
    @test_throws ArgumentError Deflated(Deflated(prob, (r,)), (r,))
    @test_throws ArgumentError Deflated(Reparametrized(prob, (y, p) -> y, nothing), (r,))
    @test_throws ArgumentError Deflated(
        LeastSquaresProblem((x, p) -> SVector(x[1]^2 - 1), 1), (r,))
    # a sparse rank-1 Jacobian update is out of scope and refused
    @test_throws ArgumentError Deflated(
        NonlinearProblem(p2_f, p2_j; jacobian_prototype = zeros(2, 2)), (r,))
    # a well-formed one, and its fields
    def = Deflated(prob, (r,))
    @test def.power === 2.0
    @test def.shift === 1.0
    @test def.roots === (r,)
    @test def.prob === prob
end

@testset "Edge 7 and MU7: the AbstractVector root storage validates too" begin
    prob = p2_problem()
    @test_throws ArgumentError Deflated(prob, [[1.0, 2.0], [3.0]])
    @test_throws ArgumentError Deflated(prob, [[1.0, NaN]])
    def = Deflated(prob, [[1.0, 2.0]])
    @test def.roots == [[1.0, 2.0]]
end

# ─── Edges 8b and 9: the refusals of a solve ────────────────────────────────

@testset "Edges 8b, 9 and MU9: a root, an element type, and a complex start" begin
    prob = p2_problem()
    x0 = SVector(0.8, 1.7)
    @test_throws ArgumentError solve(Deflated(prob, (SVector(1.0, 2.0, 3.0),)), x0, Newton())
    @test_throws ArgumentError init(Deflated(prob, (SVector(1.0, 2.0, 3.0),)), x0, Newton())
    @test_throws ArgumentError solve(
        Deflated(prob, (SVector{2, Float32}(1, 2),)), x0, Newton())
    @test_throws ArgumentError init(
        Deflated(prob, (SVector{2, Float32}(1, 2),)), x0, Newton())
    @test_throws ArgumentError solve(
        Deflated(prob, (SVector(1.0, 2.0),)), SVector(0.8 + 0im, 1.7 + 0im), Newton())
    @test_throws ArgumentError init(
        Deflated(prob, (SVector(1.0, 2.0),)), SVector(0.8 + 0im, 1.7 + 0im), Newton())
    # MU9: the complex refusal is the one that names holomorphy, not the element-type check that
    # a complex iterate fails as well
    complex_err = try
        solve(Deflated(prob, (SVector(1.0, 2.0),)), SVector(0.8 + 0im, 1.7 + 0im), Newton())
        nothing
    catch err
        err
    end
    @test complex_err isa ArgumentError
    @test occursin("not holomorphic", sprint(showerror, complex_err))
    # a root of another storage than the iterate has no `x - r`, which is a refusal, not a
    # `MethodError` deep in an evaluation
    @test_throws ArgumentError solve(
        Deflated(p1_problem(), (SVector(0.0),)), 0.1, Newton())
    @test_throws ArgumentError solve(
        Deflated(p1_problem(), (0.0,)), SVector(0.1), Newton())
    @test_throws ArgumentError solve_batch!(
        fill(SVector(0.8, 1.7), 2), Deflated(prob, (SVector{2, Float32}(1, 2),)), Newton())
    @test_throws ArgumentError solve_batch!(
        fill(SVector{2, Float32}(0.8, 1.7), 2), Deflated(prob, (SVector(1.0, 2.0),)), Newton())
    # the checks of a solve start hold on a direct `solve!!` as well
    m32 = Adapt.adapt(ToReal{Float32}(), Newton())
    def32_only = Deflated(p1_problem(), (zero(Float32),))
    state32 = init_state(m32, def32_only, Float32(0.1), nothing)
    @test_throws ArgumentError solve!!(
        state32, def32_only, 0.1, m32, nothing, Options(Float64))
    # a `Float32` root gives a `Float32` power and shift, and a `Float32` solve runs in that type
    def32 = Deflated(prob, (SVector{2, Float32}(1, 2),))
    @test def32.power isa Float32
    @test def32.shift isa Float32
    x32, s32 = solve(def32, SVector{2, Float32}(0.8, 1.7), BT)
    @test GS.isconverged(s32)
    @test norm(x32 - SVector{2, Float32}(-1, -2)) < 0.5
end

@testset "Edge 11 and MU10: a non-isbits Deflated is refused by solve_batch!" begin
    prob = p2_problem()
    X = fill(SVector(0.8, 1.7), 2)
    @test_throws ArgumentError solve_batch!(X, Deflated(prob, [[1.0, 2.0]]), Newton())
    @test_throws ArgumentError solve_batch!(
        X, Vector{SolverStatus{Float64}}(undef, 2), Deflated(prob, [[1.0, 2.0]]), Newton())
end

# ─── Edge 12: the distance on the unknowns, and the datum `b` ───────────────

@testset "Edge 12: the datum b reaches the wrapped StationarityProblem" begin
    b = SVector(0.05, -0.02)
    prob_b = StationarityProblem(p3_action; b = b)
    x0 = SVector(0.1, 0.3)
    r_plain, st_plain = solve(prob_b, x0, BT)
    r_kw, st_kw = solve(p3_problem(), x0, BT; b = b)
    @test GS.isconverged(st_plain) && GS.isconverged(st_kw)
    @test r_plain ≈ r_kw
    # the deflated solve carries the datum to the inner problem and leaves the plain root
    def = Deflated(prob_b, (r_plain,))
    r_def, st_def = solve(def, x0, BT; b = b)
    @test GS.isconverged(st_def)
    @test norm(r_def - r_plain) > 1 / 2
    # `init` and `solve!` carry the keyword datum too, and the datum `init` saw is the solver's
    solver = init(def, x0, BT; b = b)
    r_in, st_in = solve!(solver, x0; b = b)
    @test GS.isconverged(st_in)
    @test r_in ≈ r_def
    r_stored, st_stored = solve!(init(def, x0, BT; b = b), x0)
    @test GS.isconverged(st_stored)
    @test r_stored ≈ r_def
    # a datum of another length than the equations is refused
    @test_throws ArgumentError solve(def, x0, BT; b = SVector(0.1, 0.2, 0.3))
    # `solve!` takes an iterate and parameters of the types that `init` saw
    @test_throws ArgumentError solve!(solver, SVector(0.1, 0.3, 1.0), solver.p)
    @test_throws ArgumentError solve!(
        solver, ForwardDiff.Dual.(x0, 1.0), solver.p)
end

# `solve` fills the datum buffer of the dense state when it initialises it, so only a solver that
# `init` built without a datum shows the refill of the per-solve one: this test initialises without
# a datum, gives one to `solve!`, and reads the root against the one `solve` returns for it.
@testset "Edge 12: an R3 solve! refills its per-solve datum ($T)" for T in REAL_ELTYPES
    atol = residual_atol(T)
    opts = residual_options(T)
    prob = p3_problem()
    x0 = T[0.1, 0.3]
    r0, st0 = solve(prob, copy(x0), BT; opts...)
    @test GS.isconverged(st0)
    def = Deflated(prob, [copy(r0)])
    b = T[0.05, -0.02]
    r_def, st_def = solve(def, copy(x0), BT; b = b, opts...)
    @test GS.isconverged(st_def)
    r_st, st_st = solve!(init(def, copy(x0), BT; opts...), copy(x0); b = b)
    @test GS.isconverged(st_st)
    @test r_st ≈ r_def
    # the returned point is a stationary point of the problem with this solve's datum, not of the
    # undeflated one
    @test norm(p3_gradient(r_st) + b) <= atol
end

@testset "Edge 12: an R3 solve! refuses a datum and an iterate of another length ($T)" for T in REAL_ELTYPES
    opts = residual_options(T)
    prob = p3_problem()
    x0 = T[0.1, 0.3]
    r0, st0 = solve(prob, copy(x0), BT; opts...)
    @test GS.isconverged(st0)
    def = Deflated(prob, [copy(r0)])
    solver = init(def, copy(x0), BT; b = T[0.05, -0.02], opts...)
    # `init` refuses a datum of another length, so the datum below reaches the check of `solve!`
    @test_throws ArgumentError solve!(solver, copy(x0); b = T[0.1, 0.2, 0.3])
    # the length of an `Array` is not part of its type, so the template of `init` refuses an
    # iterate of another length
    @test_throws ArgumentError solve!(solver, zeros(T, 3))
end

# ─── Edge 14: the batch of a wrapped StationarityProblem ────────────────────

@testset "Edge 14: the batch of a wrapped StationarityProblem ($T)" for T in REAL_ELTYPES
    atol = residual_atol(T)
    opts = residual_options(T)
    other = T === Float32 ? Float64 : Float32
    # (a) P3: one Deflated for the whole batch, in both forms
    p3 = p3_problem()
    x03 = SVector{2, T}(T(0.1), T(0.3))
    r1, st1 = solve(p3, x03, BT; opts...)
    @test GS.isconverged(st1)
    def3 = Deflated(p3, (r1,))
    X = fill(x03, 4)
    S = Vector{SolverStatus{T}}(undef, 4)
    solve_batch!(X, S, def3, BT; opts...)
    @test all(GS.isconverged, S)
    @test all(x -> norm(x - r1) > 1 / 2, X)
    tally = solve_batch!(fill(x03, 4), def3, BT; opts...)
    @test tally.SUCCESS == 4
    # (b) P4: the state's fixed complement stays where it started, and the unknowns move off the
    # deflated root
    p4 = p4_problem()
    x04 = SVector{4, T}(T(0.1), T(0.3), T(1), T(2))
    a1, t1 = solve(p4, x04, BT; opts...)
    @test GS.isconverged(t1)
    X4 = fill(x04, 4)
    tally4 = solve_batch!(X4, Deflated(p4, (a1,)), BT; opts...)
    @test tally4.SUCCESS == 4
    @test all(x -> x[SVector(3, 4)] == SVector{2, T}(T(1), T(2)), X4)
    @test all(x -> norm(x[SVector(1, 2)] - a1[SVector(1, 2)]) > 1 / 2, X4)
    # (c) the datum: a `PerSystem` one then a shared one, each against the one `solve` takes
    b = SVector{2, T}(T(0.05), T(-0.02))
    prob_b = StationarityProblem(p3_action; b = b)
    defb = Deflated(prob_b, (r1,))
    per = [SVector{2, T}(T(0.05), T(-0.02)), SVector{2, T}(T(-0.03), T(0.04)),
        SVector{2, T}(zero(T), T(0.06)), SVector{2, T}(zero(T), T(-0.02))]
    Xb = fill(x03, 4)
    Sb = Vector{SolverStatus{T}}(undef, 4)
    solve_batch!(Xb, Sb, defb, BT; b = PerSystem(per), opts...)
    @test all(GS.isconverged, Sb)
    for i in 1:4
        @test Xb[i] ≈ solve(defb, x03, BT; b = per[i], opts...)[1]
        @test norm(p3_gradient(Xb[i]) + per[i]) <= atol
    end
    Xs = fill(x03, 4)
    solve_batch!(Xs, Deflated(prob_b, (r1,)), BT; b = b, opts...)
    @test all(x -> x ≈ solve(defb, x03, BT; b = b, opts...)[1], Xs)
    @test all(x -> norm(p3_gradient(x) + b) <= atol, Xs)
    # (d) a datum of the wrong length: a `PerSystem` of another length is a batch mismatch, and a
    # shared vector one is refused by the stationarity check
    @test_throws DimensionMismatch solve_batch!(
        fill(x03, 4), defb, BT; b = PerSystem(per[1:3]))
    @test_throws ArgumentError solve_batch!(
        fill(x03, 4), defb, BT; b = SVector{3, T}(T(0.1), T(0.2), T(0.3)))
    # (e) edge 8: a root of the unknowns alone is refused in both forms
    root_u = SVector{2, T}(T(1), zero(T))
    @test_throws ArgumentError solve_batch!(fill(x04, 4), Deflated(p4, (root_u,)), BT)
    @test_throws ArgumentError solve_batch!(
        fill(x04, 4), Vector{SolverStatus{T}}(undef, 4), Deflated(p4, (root_u,)), BT)
    # (f) edge 9: an iterate of another element type than the roots
    @test_throws ArgumentError solve_batch!(
        fill(SVector{2, other}(other(0.1), other(0.3)), 4), def3, BT)
    # (g) edge 11: a root storage that is not `isbits` is refused in both forms
    @test_throws ArgumentError solve_batch!(
        fill(x03, 4), Deflated(p3, [[zero(T), zero(T)]]), BT)
    @test_throws ArgumentError solve_batch!(
        fill(x03, 4), Vector{SolverStatus{T}}(undef, 4),
        Deflated(p3, [[zero(T), zero(T)]]), BT)
    # (h) the status vector: its real type and its length
    @test_throws ArgumentError solve_batch!(
        fill(x03, 4), Vector{SolverStatus{other}}(undef, 4), def3, BT)
    @test_throws DimensionMismatch solve_batch!(
        fill(x03, 4), Vector{SolverStatus{T}}(undef, 3), def3, BT)
    # (i) a dual parameter has no stationarity batch
    @test_throws ArgumentError solve_batch!(
        fill(x03, 4), def3, BT, ForwardDiff.Dual(T(1), one(T)))
    # (j) an empty batch: the all-zero tally, and the status vector itself
    empty_tally = solve_batch!(SVector{2, T}[], def3, BT)
    @test all(iszero, values(empty_tally))
    empty_S = SolverStatus{T}[]
    @test solve_batch!(SVector{2, T}[], empty_S, def3, BT) === empty_S
end

# ─── Clause 4: R1 kernel cleanliness ────────────────────────────────────────

# Allocation measured through a barrier, after one call that compiles the path. `--check-bounds=yes`
# lets the local `MMatrix` and `MVector` of the static factor escape to the heap, so the assertion
# is skipped there.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "Clause 4: R1 SVector isbits, @inferred and allocation-free ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    roots = (SVector{2, T}(1, 2),)
    def = Deflated(prob, roots)
    x0 = SVector{2, T}(T(0.8), T(1.7))
    @test isbits(def)
    @test isbits(Deflated(prob, (roots[1], SVector{2, T}(-1, -2))))
    @test isbits(init(def, x0, BT).state)
    x, status = @inferred solve(def, x0, BT)
    @test x isa SVector{2, T}
    @test status isa SolverStatus{T}
    @inferred residual!!(nothing, def, x0, nothing)
    @inferred jacobian!!(nothing, AutoForwardDiff(), def, x0, nothing)
    @inferred jvp!!(nothing, AutoForwardDiff(), def, x0, x0, nothing)
    if Base.JLOptions().check_bounds != 1
        @test allocations(a -> solve(def, a, BT), x0) == 0
        @test allocations(a -> residual!!(nothing, def, a, nothing), x0) == 0
        @test allocations(a -> jacobian!!(nothing, AutoForwardDiff(), def, a, nothing), x0) ==
              0
        @test allocations(a -> jvp!!(nothing, AutoForwardDiff(), def, a, a, nothing), x0) ==
              0
        # the control: the barrier sees an allocation
        @test allocations(collect, (x0[1], x0[2])) > 0
    end
end

@testset "Clause 4: R1 Number isbits, @inferred and allocation-free ($T)" for T in REAL_ELTYPES
    prob = p1_problem()
    x0 = T(0.1)
    # one deflated root and two: the three roots of the cubic are -1, 0 and 1
    for roots in ((zero(T),), (zero(T), one(T)))
        def = Deflated(prob, roots)
        @test isbits(def)
        @test isbits(init(def, x0, BT).state)
        x, status = @inferred solve(def, x0, BT)
        @test x isa T
        @test status isa SolverStatus{T}
        @inferred residual!!(nothing, def, x0, nothing)
        @inferred jacobian!!(nothing, AutoForwardDiff(), def, x0, nothing)
        @inferred jvp!!(nothing, AutoForwardDiff(), def, x0, x0, nothing)
        if Base.JLOptions().check_bounds != 1
            @test allocations(a -> solve(def, a, BT), x0) == 0
            @test allocations(a -> residual!!(nothing, def, a, nothing), x0) == 0
            @test allocations(
                a -> jacobian!!(
                    nothing, AutoForwardDiff(), def, a, nothing), x0) ==
                  0
            @test allocations(a -> jvp!!(nothing, AutoForwardDiff(), def, a, a, nothing), x0) ==
                  0
            # the control: the barrier sees an allocation
            @test allocations(collect, (x0,)) > 0
        end
        @test_opt residual!!(nothing, def, x0, nothing)
        @test_opt jacobian!!(nothing, AutoForwardDiff(), def, x0, nothing)
        @test_opt jvp!!(nothing, AutoForwardDiff(), def, x0, x0, nothing)
        @test_opt jacobian!!(nothing, nothing, def, x0, nothing)
        @test_opt solve(def, x0, Newton())
    end
end

@testset "Clause 4: R1 isbits and allocation-free, a wrapped StationarityProblem ($T)" for T in REAL_ELTYPES
    prob = p3_problem()
    def = Deflated(prob, (SVector{2, T}(0, 0),))
    x0 = SVector{2, T}(T(0.1), T(0.3))
    @test isbits(def)
    @test isbits(init(def, x0, BT).state)
    @inferred residual!!(nothing, def, x0, nothing)
    @inferred jacobian!!(nothing, AutoForwardDiff(), def, x0, nothing)
    if Base.JLOptions().check_bounds != 1
        @test allocations(a -> solve(def, a, BT), x0) == 0
    end
end

@testset "Clause 4: JET.@test_opt on the R1 path, one root and two ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    x0 = SVector{2, T}(T(0.4), T(1.3))
    for roots in ((SVector{2, T}(1, 2),),
        (SVector{2, T}(1, 2), SVector{2, T}(-1, -2)))
        def = Deflated(prob, roots)
        @test_opt residual!!(nothing, def, x0, nothing)
        @test_opt jacobian!!(nothing, AutoForwardDiff(), def, x0, nothing)
        @test_opt jvp!!(nothing, AutoForwardDiff(), def, x0, x0, nothing)
        @test_opt jacobian!!(nothing, nothing, def, x0, nothing)
        @test_opt solve(def, x0, Newton())
    end
    # and a wrapped stationarity problem, whose Jacobian is the fused `gradient_hessian!!`
    stat = Deflated(p3_problem(), (SVector{2, T}(0, 0),))
    @test_opt residual!!(nothing, stat, x0, nothing)
    @test_opt jacobian!!(nothing, AutoForwardDiff(), stat, x0, nothing)
    @test_opt solve(stat, x0, Newton())
end

# ─── Clause 4: R3 zero allocations ──────────────────────────────────────────

function solve!_allocations(x::X, x₀::X, solver::S, p::Q) where {X, S, Q}
    copyto!(x, x₀)
    solve!(solver, x, p)
    copyto!(x, x₀)
    return @allocated solve!(solver, x, p)
end

@testset "Clause 4: R3 solve! on Array allocates nothing after init ($T)" for T in REAL_ELTYPES
    prob = p2_problem_r3()
    p = T[1, 4]
    x0 = T[0.8, 1.7]
    def = Deflated(prob, [[T(1), T(2)]])
    solver = init(def, x0, BT, p)
    state = solver.state
    x = similar(x0)
    @test solve!_allocations(x, x0, solver, p) == 0
    # the workspace is the one `init` allocated
    @test solver.state.Fx === state.Fx
    @test solver.state.fac === state.fac
    @test solver.state.J === state.J
    @test solver.state.jacprep === state.jacprep
end

# ─── Clause 5: implicit differentiation ─────────────────────────────────────

@testset "Clause 5 and MU12: implicit differentiation of Deflated(::NonlinearProblem)" begin
    prob = NonlinearProblem((x, p) -> x^3 - p * x)
    def = Deflated(prob, (0.0,))
    xs, status = solve(def, 0.1, Newton(), 4.0)
    @test GS.isconverged(status)
    @test abs(abs(xs) - 2) < 1e-8
    d = ForwardDiff.derivative(p -> solve(def, 0.1, Newton(), p)[1], 4.0)
    dref = xs / (3 * xs^2 - 4.0)
    @test abs(d - dref) <= 8 * eps(Float64) * abs(dref)
    # the primal solve at a dual parameter returns the same root as the primal one
    xd = solve(def, 0.1, Newton(), ForwardDiff.Dual(4.0, 1.0))[1]
    @test ForwardDiff.value(xd) ≈ xs
end

@testset "Clause 5 and MU12: a Deflated StationarityProblem refuses a Dual" begin
    prob = p3_problem()
    def = Deflated(prob, (SVector(0.0, 0.0),))
    x0 = SVector(0.1, 0.3)
    @test_throws ArgumentError solve(def, x0, Newton(), ForwardDiff.Dual(1.0, 1.0))
    @test_throws ArgumentError init(def, x0, Newton(), ForwardDiff.Dual(1.0, 1.0))
    @test_throws ArgumentError solve(def, ForwardDiff.Dual.(x0, 1.0), Newton())
end

# `x² = p` at `p = (1, 4)`: the derivative of the root the deflated solve returns is that of the
# **undeflated** problem there, `∂x*/∂p = 1 / (2x*)`, whatever root the deflation left.
@testset "Clause 5: implicit differentiation of an R1 SVector Deflated solve" begin
    prob = NonlinearProblem((x, p) -> x .^ 2 .- p)
    q0 = SVector(1.0, 4.0)
    x0 = SVector(0.8, 1.7)
    r1, _ = solve(prob, x0, BT, q0)
    def = Deflated(prob, (r1,))
    xs, status = solve(def, x0, BT, q0)
    @test GS.isconverged(status)
    @test norm(xs - r1) > 1 / 2
    d = ForwardDiff.jacobian(q -> solve(def, x0, BT, q)[1], q0)
    dref = SMatrix{2, 2}(1 / (2 * xs[1]), 0.0, 0.0, 1 / (2 * xs[2]))
    @test norm(d - dref) <= 8 * eps(Float64) * norm(dref)
end

@testset "Clause 5: implicit differentiation of an R3 Deflated solve" begin
    prob = NonlinearProblem(p2_r3!)
    q0 = [1.0, 4.0]
    x0 = [0.8, 1.7]
    r1, _ = solve(prob, copy(x0), BT, q0)
    def = Deflated(prob, [copy(r1)])
    xs, status = solve(def, copy(x0), BT, q0)
    @test GS.isconverged(status)
    @test norm(xs - r1) > 1 / 2
    d = ForwardDiff.jacobian(q -> solve(def, copy(x0), BT, q)[1], q0)
    dref = diagm(0 => 1 ./ (2 .* xs))
    @test norm(d - dref) <= 8 * eps(Float64) * norm(dref)
end

# ─── The solver loop: `solve!!`, the hooks, and a second solve ──────────────

@testset "solve!! on a Deflated state, twice ($T)" for T in REAL_ELTYPES
    prob = p2_problem()
    def = Deflated(prob, (SVector{2, T}(1, 2),))
    x0 = SVector{2, T}(T(0.8), T(1.7))
    m = Adapt.adapt(ToReal{T}(), Newton(; globalization = Backtracking()))
    state = init_state(m, def, x0, nothing)
    x, status, state₊ = solve!!(
        state, def, x0, m, nothing, Options(T; f_abstol = residual_atol(T), f_reltol = zero(T)))
    @test GS.isconverged(status)
    @test norm(x - SVector{2, T}(-1, -2)) < T(1) / 2
    @test residual_norm(prob, x) <= residual_atol(T)
    # a second solve from the state that the first returned starts afresh
    y, status2, _ = solve!!(
        state₊, def, x0, m, nothing, Options(T; f_abstol = residual_atol(T), f_reltol = zero(T)))
    @test GS.isconverged(status2)
    @test y == x
end
