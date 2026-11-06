# Range and spectrum diagnostics (`rank`, `nullnorm`, `gradratio`) in `SolverStatus` and
# the static SVD factor slot of `Newton` with `reuse`.

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: mul!, norm
using Random: Xoshiro
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, DogLeg, LUFactorization, MAXITERS, NONFINITE, Newton,
                        NonlinearProblem, Options, Picard, QRFactorization, SINGULAR,
                        STALLED, SUCCESS, SVDFactorization, SolverStatus, Static, ToReal,
                        finish!!, init, init_state, rnorm, solve, solve!, solve_batch!,
                        solve!!

include("../helpers/matrix.jl")

allowscalar(false)

# Linear systems F(x) = A x - b in the value form (SVector) and array form (Array, JLArray).
linF(x, p) = p.A * x - p.b
linJ(x, p) = p.A
linF!(y, x, p) = (mul!(y, p.A, x); y .-= p.b; y)
linJ!(J, x, p) = copyto!(J, p.A)
const SPROB = NonlinearProblem(linF, linJ)
const APROB = NonlinearProblem(linF!, linJ!)

svd_newton(; kw...) = Newton(; linsolve = SVDFactorization(), kw...)
sentinels(st) = st.rank === Int32(-1) && isnan(st.nullnorm) && isnan(st.gradratio)
near(a, b, R; k = 32) = abs(a - b) <= k * eps(R) * max(abs(b), one(R))
ratio(J, F) = norm(J' * Vector(F)) / (norm(J) * norm(F))
adapted(m, ::Type{T}) where {T} = Adapt.adapt(ToReal{real(T)}(), m)
iterate_after(prob, x₀, m, p, k) = first(solve(prob, x₀, m, p; max_iterations = k))

# Check 1: consistent rank-one systems with exact integer data, A = u vᴴ and b = A x*.
function rank_one_cases(::Type{T}, n; count = 100) where {T}
    rng = Xoshiro(20261101 + n)
    cases = Tuple{SMatrix{n, n, T}, SVector{n, T}}[]
    while length(cases) < count
        u = T.(rand(rng, -4:4, n))
        v = T.(rand(rng, -4:4, n))
        xs = T.(rand(rng, -4:4, n))
        (iszero(u) || iszero(v)) && continue
        A = SMatrix{n, n, T}(u * v')
        b = A * SVector{n, T}(xs)
        iszero(b) && continue
        push!(cases, (A, b))
    end
    return cases
end

inc_p(::Type{T}) where {T} = (A = SMatrix{2, 2, T}(1, 1, 1, 1), b = SVector{2, T}(1, -1))
function diag_p(::Type{T}) where {T}
    return (A = SMatrix{2, 2, T}(1, 0, 0, T(1e-9)), b = SVector{2, T}(0, T(1e-3)))
end
const DOGLEG_SVD = Newton(;
    linsolve = SVDFactorization(), globalization = DogLeg(; radius = 1))
zero_p(::Type{T}) where {T} = (A = zero(SMatrix{2, 2, T}), b = SVector{2, T}(-1, -1))
inf_p(::Type{T}) where {T} = (A = SMatrix{2, 2, T}(Inf, 1, 1, 1), b = SVector{2, T}(3, 1))
infF(x, p) = x - p.b
infF!(y, x, p) = (y .= x .- p.b; y)
const INF_SPROB = NonlinearProblem(infF, linJ)
const INF_APROB = NonlinearProblem(infF!, linJ!)
reg_p(::Type{T}) where {T} = (A = SMatrix{2, 2, T}(2, 0, 1, 1), b = SVector{2, T}(1, 1))
cplx_p(::Type{T}) where {T} = (A = SMatrix{2, 2, T}(1, 0, im, 1), b = -SVector{2, T}(1, im))
root_p(::Type{T}) where {T} = (A = SMatrix{2, 2, T}(1, 1, 1, -1), b = SVector{2, T}(3, 1))
function nonfinite_p(::Type{T}) where {T}
    return (A = SMatrix{2, 2, T}(1, 0, 0, 1), b = SVector{2, T}(Inf, 0))
end

array_p(p) = (A = Matrix(p.A), b = Vector(p.b))
backend_p(::Type{Array}, p) = array_p(p)
backend_p(::Type{JLArray}, p) = (A = JLArray(Matrix(p.A)), b = JLArray(Vector(p.b)))

sqF(x, p) = x^2 + 1
sqJ(x, p) = 2x
sqF1(x, p) = x .* x .+ 1
const SQ_STARTS = (0.7, 3.0, -1.3)
const SQ_METHOD = Newton(; linsolve = SVDFactorization(), globalization = Backtracking())

const JACOBIANS = Ref(0)
quadF!(y, x, p) = (y .= x .* x .- [4, 9]; y)
function quadJ!(J, x, p)
    JACOBIANS[] += 1
    fill!(J, 0)
    J[1, 1] = 2x[1]
    J[2, 2] = 2x[2]
    return J
end
const QPROB = NonlinearProblem(quadF!, quadJ!)
quad_Jm(x) = [2x[1] 0; 0 2x[2]]
quad_F(x) = x .* x .- [4, 9]
quadF(x, p) = x .* x .- SVector{2, eltype(x)}(4, 9)
quadJ(x, p) = (JACOBIANS[] += 1; SMatrix{2, 2, eltype(x)}(2x[1], 0, 0, 2x[2]))
quadJ_pure(x, p) = SMatrix{2, 2, eltype(x)}(2x[1], 0, 0, 2x[2])
quadFn(x, p) = x * x - 4
quadJn_pure(x, p) = 2x
const QSPROB = NonlinearProblem(quadF, quadJ)
const QNPROB = NonlinearProblem(quadFn, (x, p) -> (JACOBIANS[] += 1; 2x))
const QSPROB_PURE = NonlinearProblem(quadF, quadJ_pure)
const QNPROB_PURE = NonlinearProblem(quadFn, quadJn_pure)

@testset "check 1 — consistent rank-one systems, $T, n = $n" for T in ELTYPES,
    n in (2, 3, 4)

    R = real(T)
    cases = rank_one_cases(T, n)
    @test all(cases) do (A, b)
        _, st = solve(SPROB, zero(SVector{n, T}), svd_newton(), (A = A, b = b))
        st.rank === Int32(1) && st.nullnorm < 10 * eps(R) * norm(b) &&
            isfinite(st.gradratio)
    end
    @test all(cases) do (A, b)
        _, st = solve(APROB, zeros(T, n), svd_newton(), array_p((A = A, b = b)))
        st.rank === Int32(1) && st.nullnorm < 10 * eps(R) * norm(b) &&
            isfinite(st.gradratio)
    end
end

@testset "mixed-precision SVD nullnorm" begin
    T = last(REAL_ELTYPES)
    TF = first(REAL_ELTYPES)
    A1 = SMatrix{3, 3, T}(T[1, 2, -1] * T[2, 1, 3]')
    p1 = (A = A1, b = A1 * SVector{3, T}(1, -1, 2))
    m = Newton(; linsolve = SVDFactorization(TF))
    for (prob, x₀, pp) in (
        (SPROB, zero(SVector{3, T}), p1),
        (APROB, zeros(T, 3), array_p(p1)),
        (APROB, JLArray(zeros(T, 3)), backend_p(JLArray, p1))
    )
        _, st = solve(prob, x₀, m, pp)
        @test st.rank === Int32(1)
        @test st.nullnorm isa T
        @test st.nullnorm < 10 * eps(TF) * norm(p1.b)
    end

    # Zero residual at an exact root with min_iterations = 1 in mixed precision and on JLArray
    p_root = root_p(T)
    for (prob, x₀, pp, method) in (
        (SPROB, SVector{2, T}(2, 1), p_root, m),
        (APROB, T[2, 1], array_p(p_root), m),
        (APROB, JLArray(T[2, 1]), backend_p(JLArray, p_root), svd_newton()),
        (APROB, JLArray(T[2, 1]), backend_p(JLArray, p_root), m)
    )
        _, st_z = solve(prob, x₀, method, pp; min_iterations = 1)
        @test st_z.rank === Int32(2)
        @test st_z.nullnorm === zero(T)
        @test st_z.gradratio === zero(T)
    end
end

@testset "check 2 — least-squares point with no root, $T" for T in ELTYPES
    R = real(T)
    p = inc_p(T)
    for (prob, x₀, pp) in (
        (SPROB, SVector{2, T}(1, 0), p),
        ((APROB, AT(T[1, 0]), backend_p(AT, p)) for AT in ARRAY_BACKENDS)...
    )
        _, st = solve(prob, x₀, svd_newton(), pp)
        @test st.code === STALLED
        @test st.rank === Int32(1)
        @test st.gradratio <= R(1e-6)
        @test near(st.nullnorm, st.fnorm, R; k = 8)
        @test near(st.nullnorm, sqrt(R(2)), R; k = 8)

        _, s1 = solve(prob, x₀, svd_newton(), pp; max_iterations = 1)
        @test near(s1.gradratio, inv(sqrt(R(2))), R)
    end
end

@testset "n = 1, x² + 1, $T" for T in REAL_ELTYPES
    for x0 in SQ_STARTS
        for (prob, x₀) in (
            (NonlinearProblem(sqF, sqJ), T(x0)),
            (NonlinearProblem(sqF1), SVector{1, T}(x0))
        )
            _, st = solve(prob, x₀, SQ_METHOD)
            @test st.code !== SUCCESS
            @test st.rank === Int32(1)
            @test st.nullnorm === zero(T)
            @test near(st.gradratio, one(T), T; k = 8)
        end
    end
end

@testset "check 3 — cutoff of the factor, $T" for T in ELTYPES
    R = real(T)
    p = diag_p(T)
    s = real(p.A[2, 2])
    full = R === last(REAL_ELTYPES)
    for (prob, x₀, pp) in (
        (SPROB, zero(SVector{2, T}), p),
        (APROB, zeros(T, 2), array_p(p))
    )
        _, st = solve(prob, x₀, DOGLEG_SVD, pp; max_iterations = 5)
        x₄ = SVector{2, T}(Array(iterate_after(prob, x₀, DOGLEG_SVD, pp, 4)))
        F₄ = norm(p.A * x₄ - p.b)
        @test st.code === MAXITERS
        @test st.rank === Int32(full ? 2 : 1)
        if full
            @test st.nullnorm <= 10 * eps(R) * F₄
        else
            @test near(st.nullnorm, F₄, R; k = 8)
        end
        @test near(st.gradratio, s / sqrt(one(R) + s^2), R)
    end
end

@testset "zero Jacobian, rank 0, $T" for T in ELTYPES
    R = real(T)
    p = zero_p(T)
    for (prob, x₀, pp) in (
        (SPROB, zero(SVector{2, T}), p),
        (APROB, zeros(T, 2), array_p(p))
    )
        _, st = solve(prob, x₀, svd_newton(), pp)
        @test st.rank === Int32(0)
        @test st.nullnorm === st.fnorm
        @test st.gradratio === zero(R)
        for ls in (LUFactorization(), QRFactorization())
            _, st_lq = solve(prob, x₀, Newton(; linsolve = ls), pp)
            @test st_lq.rank === Int32(-1)
            @test isnan(st_lq.nullnorm)
            @test st_lq.gradratio === zero(R)
        end
    end
    if T === last(REAL_ELTYPES)
        _, st_jl = solve(APROB, JLArray(zeros(T, 2)), svd_newton(), backend_p(JLArray, p))
        @test st_jl.rank === Int32(0)
        @test st_jl.nullnorm === st_jl.fnorm
        @test st_jl.gradratio === zero(R)
    end
end

@testset "LU, QR, failed SVD factor and regularization μ = 1, $T" for T in REAL_ELTYPES
    p_inf = inf_p(T)
    for (prob, x₀, pp) in (
        (INF_SPROB, zero(SVector{2, T}), p_inf),
        (INF_APROB, zeros(T, 2), array_p(p_inf)),
        (T === last(REAL_ELTYPES) ?
         ((INF_APROB, JLArray(zeros(T, 2)), backend_p(JLArray, p_inf)),) : ())...
    )
        _, st = solve(prob, x₀, svd_newton(), pp)
        @test st.code === SINGULAR
        @test st.rank === Int32(-1)
        @test isnan(st.nullnorm)
        @test !isfinite(st.gradratio)
    end

    p_reg = reg_p(T)
    g = ratio(p_reg.A, -p_reg.b)
    for ls in (LUFactorization(), QRFactorization())
        _, st = solve(
            SPROB, zero(SVector{2, T}), Newton(; linsolve = ls), p_reg; max_iterations = 1)
        @test st.rank === Int32(-1)
        @test isnan(st.nullnorm)
        @test near(st.gradratio, g, T)
    end
    for (prob, x₀, pp) in (
        (SPROB, zero(SVector{2, T}), p_reg),
        (APROB, zeros(T, 2), array_p(p_reg))
    )
        _, st = solve(
            prob, x₀, svd_newton(; regularization = 1), pp; max_iterations = 1)
        @test near(st.gradratio, g, T)
        @test st.rank === Int32(2)
        @test st.nullnorm <= 10 * eps(T) * norm(p_reg.b)
    end
end

@testset "complex conjugate transpose, $T" for T in filter(T -> T <: Complex, ELTYPES)
    R = real(T)
    p = cplx_p(T)
    for (prob, x₀, pp) in (
        (SPROB, zero(SVector{2, T}), p),
        (APROB, zeros(T, 2), array_p(p))
    )
        _, st = solve(prob, x₀, svd_newton(), pp)
        @test near(st.gradratio, inv(sqrt(R(6))), R)
    end
end

# J = s I and F(0) = -s (1, 1) with s² above floatmax: Jᴴ F overflows, the ratio is 1/√2.
@testset "gradratio where Jᴴ F overflows, $T" for T in ELTYPES
    R = real(T)
    s = 4 * sqrt(floatmax(R))
    p = (A = SMatrix{2, 2, T}(s, 0, 0, s), b = SVector{2, T}(s, s))
    for (prob, x₀, pp) in (
            (SPROB, zero(SVector{2, T}), p),
            (APROB, zeros(T, 2), array_p(p)),
            (APROB, JLArray(zeros(T, 2)), backend_p(JLArray, p))
        ),
        m in (Newton(), svd_newton())

        _, st = solve(prob, x₀, m, pp; max_iterations = 1)
        @test near(st.gradratio, inv(sqrt(R(2))), R)
    end
    _, st1 = solve(NonlinearProblem((x, p) -> p * x - p, (x, p) -> p), zero(T), Newton(),
        T(s); max_iterations = 1)
    @test near(st1.gradratio, one(R), R)
end

@testset "start!! resets and factorisation at F = 0, $T" for T in REAL_ELTYPES
    p = root_p(T)
    s = init(APROB, zeros(T, 2), svd_newton(), array_p(p))
    x = zeros(T, 2)
    _, s1 = solve!(s, x, array_p(p))
    x .= T[2, 1]
    _, s2 = solve!(s, x, array_p(p))
    @test s1.rank === Int32(2)
    @test s2.iterations === Int32(0)
    @test sentinels(s2)

    m = adapted(svd_newton(), T)
    opt = Options(T)
    state₀ = init_state(m, SPROB, zero(SVector{2, T}), p)
    _, sr1, state₁ = solve!!(state₀, SPROB, zero(SVector{2, T}), m, p, opt)
    _, sr2, _ = solve!!(state₁, SPROB, SVector{2, T}(2, 1), m, p, opt)
    @test sr1.rank === Int32(2)
    @test sr2.iterations === Int32(0)
    @test sentinels(sr2)

    for g in (Static(), DogLeg())
        _, st = solve(
            SPROB, SVector{2, T}(2, 1), svd_newton(; globalization = g), p;
            min_iterations = 1)
        @test st.iterations === Int32(1)
        @test st.rank === Int32(2)
        @test st.gradratio === zero(T)
        @test st.nullnorm === zero(T)
    end
end

@testset "factor reuse and static SVD factor slot" begin
    R64 = last(REAL_ELTYPES)
    x₀ = R64[1, 1]
    _, s2 = solve(QPROB, copy(x₀), svd_newton(; reuse = 2); max_iterations = 2)
    _, s3 = solve(QPROB, copy(x₀), svd_newton(; reuse = 2); max_iterations = 3)
    @test near(s2.gradratio, ratio(quad_Jm(x₀), quad_F(x₀)), R64)
    x₂ = first(solve(QPROB, copy(x₀), svd_newton(; reuse = 2); max_iterations = 2))
    @test near(s3.gradratio, ratio(quad_Jm(x₂), quad_F(x₂)), R64)

    for T in REAL_ELTYPES
        m = svd_newton(; reuse = 2)
        xa₂, _ = solve(QPROB, T[1, 1], m; max_iterations = 2)
        JACOBIANS[] = 0
        xs₂, ss₂ = solve(QSPROB, SVector{2, T}(1, 1), m; max_iterations = 2)
        @test ss₂.code === MAXITERS
        @test ss₂.iterations === Int32(2)
        @test JACOBIANS[] == 1
        @test isapprox(xs₂, SVector{2, T}(xa₂); rtol = 64 * eps(T))
        @test near(ss₂.gradratio, T(ratio(quad_Jm(T[1, 1]), quad_F(T[1, 1]))), T)
        @test ss₂.rank === Int32(2)

        _, sf = solve(QSPROB, SVector{2, T}(1, 1), m)
        @test sf.code === SUCCESS

        JACOBIANS[] = 0
        xn₂, sn₂ = solve(QNPROB, T(1), m; max_iterations = 2)
        @test sn₂.code === MAXITERS
        @test sn₂.iterations === Int32(2)
        @test JACOBIANS[] == 1
        @test isapprox(xn₂, T(11) / T(8); rtol = 64 * eps(T))
        _, sn = solve(QNPROB, T(1), m)
        @test sn.code === SUCCESS
        @test sn.rank === Int32(1)

        ma = adapted(m, T)
        state_s = init_state(ma, QSPROB, SVector{2, T}(1, 1), nothing)
        @test isbits(state_s)
        state_n = init_state(ma, QNPROB, T(1), nothing)
        @test isbits(state_n)

        x0_s = SVector{2, T}(1, 1)
        x0_n = T(1)
        @test (@inferred solve(QSPROB_PURE, x0_s, m))[2] isa SolverStatus{T}
        @test (@inferred solve(QNPROB_PURE, x0_n, m))[2] isa SolverStatus{T}
        solve(QSPROB_PURE, x0_s, m)
        solve(QNPROB_PURE, x0_n, m)
        if Base.JLOptions().check_bounds != 1
            # Skipped where a run forces `--check-bounds=yes`.
            @test (@allocated solve(QSPROB_PURE, x0_s, m)) == 0
            @test (@allocated solve(QNPROB_PURE, x0_n, m)) == 0
        end
        JET.@test_opt solve(QSPROB_PURE, x0_s, m)
        JET.@test_opt solve(QNPROB_PURE, x0_n, m)
    end
end

@testset "sentinels on Picard, non-finite start and init_state" begin
    R64 = last(REAL_ELTYPES)
    _, st_nf = solve(SPROB, zero(SVector{2, R64}), svd_newton(), nonfinite_p(R64))
    @test st_nf.code === NONFINITE
    @test sentinels(st_nf)

    _, st_pc = solve(
        NonlinearProblem((x, p) -> x .* x .- SVector{2, R64}(4, 9)),
        SVector{2, R64}(1, 1), Picard(; damping = 0.1); max_iterations = 3)
    @test sentinels(st_pc)

    for (x₀, pp, prob) in (
        (zero(SVector{2, R64}), diag_p(R64), SPROB),
        (zeros(R64, 2), array_p(diag_p(R64)), APROB)
    )
        m = svd_newton()
        state = init_state(m, prob, x₀, pp)
        s6 = SolverStatus{R64}(MAXITERS, Int32(0), one(R64), zero(R64), Int32(0), false)
        Fx = x₀ isa SVector ? linF(x₀, pp) : linF!(similar(x₀), x₀, pp)
        _, sf, _ = finish!!(state, m, prob, x₀, Fx, s6)
        @test sentinels(sf)
    end
end

@testset "R2 writes the R1 status and counts, $T" for T in REAL_ELTYPES
    for (p, m, k, x₀) in (
        (diag_p(T), DOGLEG_SVD, 5, zero(SVector{2, T})),
        (inc_p(T), svd_newton(), 100, SVector{2, T}(1, 0)),
        (reg_p(T), Newton(), 1, zero(SVector{2, T}))
    )
        _, st = solve(SPROB, x₀, m, p; max_iterations = k)
        @test isfinite(st.gradratio)
        X = [x₀ for _ in 1:4]
        S = Vector{SolverStatus{T}}(undef, 4)
        solve_batch!(X, S, SPROB, m, p; max_iterations = k)
        @test all(s -> s === st, S)
    end
    c = solve_batch!(
        [zero(SVector{2, T}) for _ in 1:4], SPROB, DOGLEG_SVD, diag_p(T); max_iterations = 5)
    @test c.MAXITERS == 4
    @test sum(c) == 4
end

@testset "R1 is isbits, inferred, allocation-free and JET-clean, $T" for T in REAL_ELTYPES
    p3 = diag_p(T)
    x3 = zero(SVector{2, T})
    m_dl = adapted(DOGLEG_SVD, T)
    @test isbits(init_state(m_dl, SPROB, x3, p3))
    _, st3 = @inferred solve(SPROB, x3, DOGLEG_SVD, p3; max_iterations = 5)
    @test st3 isa SolverStatus{T}
    @test st3.rank >= Int32(1)
    if Base.JLOptions().check_bounds != 1
        # Skipped where a run forces `--check-bounds=yes`.
        @test (@allocated solve(SPROB, x3, DOGLEG_SVD, p3; max_iterations = 5)) == 0
    end
    JET.@test_opt solve(SPROB, x3, DOGLEG_SVD, p3)

    A1, b1 = first(rank_one_cases(T, 2; count = 1))
    p1 = (A = A1, b = b1)
    x1 = zero(SVector{2, T})
    for ls in (SVDFactorization(), LUFactorization(), QRFactorization())
        m = Newton(; linsolve = ls)
        ma = adapted(m, T)
        @test isbits(init_state(ma, SPROB, x1, p1))
        _, st1 = @inferred solve(SPROB, x1, m, p1; max_iterations = 5)
        @test st1 isa SolverStatus{T}
        @test isfinite(st1.gradratio)
        if Base.JLOptions().check_bounds != 1
            # Skipped where a run forces `--check-bounds=yes`.
            @test (@allocated solve(SPROB, x1, m, p1; max_iterations = 5)) == 0
        end
        JET.@test_opt solve(SPROB, x1, m, p1)
    end
end
