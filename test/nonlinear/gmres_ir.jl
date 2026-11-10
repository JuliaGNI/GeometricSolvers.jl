# GMRES-IR: the SDD five-precision variant of iterative refinement, the three-stage remedy chain
# `LU-IR -> GMRES-IR -> T refactorisation`, the true linear residual bound of the inner GMRES, and
# the comparison script (§13.AD, Checks 1–5).
#
# The chain is only exercised by a Jacobian whose conditioning is above the LU-IR limit of about
# `2e7`, so the fixture is a linear problem whose right-hand side lies along the smallest singular
# vector: there the relative residual of a solve in `Float32` is of order `κ eps(Float32)`, and the
# LU-IR passes fail from `κ = 5e7` on. A right-hand side along the well-conditioned directions
# leaves LU-IR successful at every `κ`, and exercises no stage of the chain.

using ADTypes: AutoFiniteDiff
using Adapt: Adapt
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: @index, @kernel, synchronize
using LinearAlgebra: Diagonal, mul!, norm
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Options, PromotedProduct, ToReal, accepts_gmres_ir, factor_eltype,
                        factorize!!, gmres_ir!!, init_linear,
                        init_state, isconverged, ldiv!!, promote_lu_factors,
                        promote_lu_factors!, rnorm, solve!, solve!!, start!!, step!!

include("../helpers/matrix.jl")

const GS = GeometricSolvers

allowscalar(false)

# The working types of the chain: a `Float32` problem has no factorisation precision below it, and
# the SDD variant is `Float32` factors of a `Float64` problem (§1.4).
const SDD_ELTYPES = filter(T -> real(T) === Float64, ELTYPES)

# The conditioning numbers of the three branches: below the LU-IR limit, between the two limits, and
# beyond the limit of GMRES-IR with `Float32` factors (Amestoy et al. 2024, Tables 3.3–3.4).
const K_WELL = 1.0e4
const K_MID = 5.0e7
const K_HIGH = 1.0e11

const IR_TOL = 1e-8

# The DCT-II matrix of `n` points, orthogonal to rounding.
function ir_basis(n::Integer)
    return [i == 1 ? sqrt(1 / n) : sqrt(2 / n) * cos(pi * (2j - 1) * (i - 1) / (2n))
            for
            j in 1:n, i in 1:n]
end

# The linear problem `F(x) = A x` with `A = Q diag(σ) Qᵀ` of `κ∞ ≈ κ`, its analytic Jacobian `J!`,
# and the start `2 Q[:, n]` along the smallest singular vector of `A`, on the array type `AT` and in
# the working type `T`. A `scale` multiplies `J` and `F` and leaves every relative quantity and
# `κ∞` unchanged, which is what the scaling testsets below vary.
function ir_problem(κ::Float64, AT = Array, ::Type{T} = Float64; n::Integer = 8,
        scale::Float64 = 1.0) where {T}
    Q = ir_basis(n)
    σ = [κ^(-(i - 1) / (n - 1)) for i in 1:n]
    A = AT(T.(scale .* (Q * Diagonal(σ) * Q')))
    F!(r, x, p) = (mul!(r, A, x); r)
    J!(J, x, p) = (copyto!(J, A); J)
    return NonlinearProblem(F!, J!), J!, A, AT(T.(2.0 .* Q[:, n]))
end

# The method of the chain, with the cap and the tolerance of its inner GMRES.
function ir_method(cap::Integer = 8)
    return Newton(; linsolve = LUFactorization(Float32),
        refinement = GMRESRefinement(Float64; gmres_maxiter = cap, gmres_tol = IR_TOL))
end

# The direction of one step, and the residual and the iterate of its start: `step!!` overwrites the
# iterate and the residual it is given, so both are copied before it runs.
function one_step(prob, x0, m)
    x = copy(x0)
    Fx = prob.F(similar(x), x, nothing)
    state = init_state(m, prob, x, nothing)
    state = start!!(state, m, prob, x, Fx, nothing, rnorm(Fx))
    start_x = copy(x)
    start_F = copy(Fx)
    x₊, _, _, info = step!!(state, m, prob, x, Fx, nothing)
    return (; F = start_F, dir = x₊ .- start_x, info = info)
end

# The allocation probe of a solve, behind a function barrier that takes concrete types and warms its
# own path: `@allocated` measures the call it wraps.
@noinline function sdd_allocations(solver::S, x::X, x0::X) where {S, X}
    copyto!(x, x0)
    solve!(solver, x)
    copyto!(x, x0)
    return @allocated solve!(solver, x)
end

# One `solve!!` per index of the KA backend, on the value iterate `x₀`: the shape of a solve inside
# a kernel, which is the one a `GMRESRefinement` reaches without a Krylov basis.
@kernel function gmres_ir_kernel!(
        codes, xs, prob, method, state₀, opt, @Const(x₀), @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        x, st, _ = solve!!(state₀, prob, x₀, method, p, opt)
        codes[i] = st.code
        xs[i] = x
    end
end

@testset "GMRES-IR" begin
    # ─── Check 1: the SDD precision variant ──────────────────────────────────

    @testset "Check 1: factors in TF, buffers in T ($T, $AT)" for T in SDD_ELTYPES,
        AT in ARRAY_BACKENDS

        prob, J!, A, x0 = ir_problem(K_MID, AT, T)
        m = ir_method()
        solver = init(prob, copy(x0), m)
        _, st = solve!(solver, copy(x0))
        @test isconverged(st)

        # MU1, MU2: the stored factors are the `Float32` ones the step factorised, and the GMRES-IR
        # stage solves with a copy of them in `T`, not with the stored factor and not with a
        # factorisation of the Jacobian in `T`.
        @test eltype(stored_factors(solver)) === factor_eltype(LUFactorization(Float32), T)
        @test eltype(promoted_factors(solver)) === T
        @test eltype(preconditioned_buffer(solver)) === T
        @test eltype(linear_residual_buffer(solver)) === T
        # and it is the linear residual `r = -F(x) - J d` of the stage, not the `F(x₀)` slot that
        # `residual_buffer` names: a stage that keeps `r` in the factorisation precision fails here
        @test linear_residual_buffer(solver) === solver.state.refbuf.r
        @test linear_residual_buffer(solver) !== GS.residual_buffer(solver.state)
        @test Array(promoted_factors(solver)) == Array(stored_factors(solver))

        # the promotion itself: the packed factors, the pivots and the scaling travel as they are
        Jm = similar(x0, length(x0), length(x0))
        J!(Jm, x0, nothing)
        fac32 = factorize!!(
            init_linear(LUFactorization(Float32), nothing, x0), LUFactorization(Float32), Jm)
        fac_T = promote_lu_factors!(init_linear(LUFactorization(), nothing, x0), fac32)
        @test eltype(fac_T.factors) === T
        @test Array(fac_T.factors) == Array(fac32.factors)
        @test Array(fac_T.ipiv) == Array(fac32.ipiv)
        @test fac_T.scale_exponent == fac32.scale_exponent
        @test fac_T.info == fac32.info

        # MU1: the preconditioned product `M̂⁻¹ (J v)` that the inner GMRES forms is solved with the
        # promoted factors. A solve with the stored `Float32` factor differs from it well above
        # `eps(T)`, so the assertions below cannot hold for a stage that solves with `fac`.
        b = similar(x0)
        mul!(b, A, x0)
        z_T = ldiv!!(similar(x0), fac_T, b)
        z_32 = ldiv!!(similar(x0), fac32, b)
        @test rnorm(z_T - z_32) > 1e-6 * rnorm(z_T)
        out = similar(x0)
        PromotedProduct(Jm, fac_T, similar(x0), similar(x0))(out, x0)
        @test out ≈ z_T
        @test rnorm(out - z_32) > 1e-6 * rnorm(z_T)
    end

    @testset "Check 1: the R1 promotion of a value iterate" begin
        # the R1 form of the same copy, for a factorisation that is `isbits` and has no buffer
        A = SMatrix{4, 4, Float64}(
            1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 4.0, 2.0,
            0.0, 1.0, 0.0, 5.0)
        fac32 = factorize!!(nothing, LUFactorization(Float32), A)
        @test fac32 isa GS.StaticLU
        fac_T = promote_lu_factors(fac32, Float64)
        @test fac_T isa GS.StaticLU
        @test eltype(fac_T.factors) === Float64
        @test fac_T.factors == SMatrix{4, 4, Float64}(Array(fac32.factors))
        @test fac_T.ipiv == fac32.ipiv
        @test fac_T.scale_exponent == fac32.scale_exponent
        @test fac_T.info == fac32.info
        b = SVector{4, Float64}(1.0, -2.0, 3.0, 0.5)
        @test ldiv!!(nothing, fac_T, b) isa SVector{4, Float64}
        # the same factors, solved in each element type: the two differ by the rounding of the
        # `Float32` solve, and by nothing else
        @test isapprox(ldiv!!(nothing, fac_T, b),
            Float64.(ldiv!!(nothing, fac32, Float32.(b))); rtol = 8 * eps(Float32))
    end

    # ─── Check 2: the three-stage order ──────────────────────────────────────

    @testset "Check 2: LU-IR, then GMRES-IR, then the refactorisation ($T, $AT)" for T in SDD_ELTYPES,
        AT in ARRAY_BACKENDS

        # MU3: a well-conditioned Jacobian gives an LU-IR success, and a chain that skipped the
        # LU-IR passes and ran GMRES-IR unconditionally records a failure at that step.
        prob, _, _, x0 = ir_problem(K_WELL, AT, T)
        _, st = solve!(init(prob, copy(x0), ir_method()), copy(x0))
        @test isconverged(st)
        @test st.step_failures == 0
        @test !st.promoted

        # MU4: above the LU-IR limit and within the GMRES-IR limit, the first stage fails and the
        # second converges with the promoted factors, so no step refactorises in `T`.
        prob, _, _, x0 = ir_problem(K_MID, AT, T)
        _, st = solve!(init(prob, copy(x0), ir_method()), copy(x0))
        @test isconverged(st)
        @test st.step_failures >= 1
        @test !st.promoted

        # MU5: beyond the GMRES-IR limit both refinement stages fail, and the second remedy is the
        # refactorisation of the Jacobian in `T`.
        prob, _, _, x0 = ir_problem(K_HIGH, AT, T)
        _, st = solve!(init(prob, copy(x0), ir_method()), copy(x0))
        @test isconverged(st)
        @test st.step_failures >= 2
        @test st.promoted
    end

    @testset "Check 2: the chain takes the same branch at every scale ($T)" for T in SDD_ELTYPES

        # Multiplying `J` and `F` by `s` leaves `κ∞` and every relative quantity unchanged, so the
        # same Jacobian must reach the same stage at every `s`: a stop that compares a residual in
        # the units of `x` with a tolerance in the units of `F` reads a different system at each
        # `s`, and promotes at the large ones and not at the small ones.
        for scales in ((1.0e-6, 1.0, 1.0e6, 1.0e8), (2.0^-30, 2.0^-10, 2.0^10, 2.0^30))
            prob, _, _, x0 = ir_problem(K_MID, Array, T)
            _, st_ref = solve!(init(prob, copy(x0), ir_method()), copy(x0))
            @test isconverged(st_ref)
            @test st_ref.step_failures >= 1
            @test !st_ref.promoted
            for s in scales
                sprob, _, _, sx0 = ir_problem(K_MID, Array, T; scale = s)
                _, st = solve!(init(sprob, copy(sx0), ir_method()), copy(sx0))
                @test isconverged(st)
                @test st.promoted == st_ref.promoted
                @test st.step_failures == st_ref.step_failures
            end
        end
    end

    # ─── Check 3: the cap of the inner GMRES ─────────────────────────────────

    @testset "Check 3: a cap of one Krylov iteration ($T, $AT)" for T in SDD_ELTYPES,
        AT in ARRAY_BACKENDS

        prob, _, _, x0 = ir_problem(K_MID, AT, T)
        # MU6: one Krylov step cannot meet `IR_TOL` on this problem, and a cap that is ignored, or
        # whose exhaustion records no failure, leaves the solve on the low-precision direction.
        m = ir_method(1)
        @test m.refinement.gmres_maxiter === Int32(1)
        _, st = solve!(init(prob, copy(x0), m), copy(x0))
        @test isconverged(st)
        @test st.step_failures >= 2
        @test st.promoted
    end

    # ─── Check 4: the true linear residual bound in T ────────────────────────

    @testset "Check 4: the true linear residual in T ($T, κ = $κ, n = $n)" for T in SDD_ELTYPES,
        κ in (5.0e7, 1.0e8, 2.0e8, 5.0e8), n in (8, 16, 32)

        # The bound is a relative one, so it holds at every size and every `κ∞` of the range in
        # which GMRES-IR is the stage that supplies the direction. The cap is wide enough for the
        # whole Krylov space, so what the grid varies is the conditioning and the size, not the cap
        # (Check 3 owns the cap). A residual rounded to the factorisation precision meets the bound
        # at a small `κ` and misses it here: every cell of the grid meets it, the tightest at 0.92
        # of it.
        prob, J!, _, x0 = ir_problem(κ, Array, T; n = n)
        step = one_step(prob, copy(x0), ir_method(16))
        @test step.info.failures == 1
        @test !step.info.promoted
        Jm = similar(x0, length(x0), length(x0))
        J!(Jm, x0, nothing)
        # MU7: the bound is on the true residual recomputed in `T`, not on the Arnoldi estimate of
        # the preconditioned system, and the direction the stage accepts meets it.
        @test norm(step.F .+ Jm * step.dir) <= 10 * IR_TOL * norm(step.F)
        @test norm(step.F .+ Jm * step.dir) < norm(step.F)
    end

    @testset "Check 4: the accepted direction meets the bound at every scale ($T)" for T in SDD_ELTYPES

        # A true residual is scale-free, so a stage that stops on one accepts at every `s`. A stage
        # whose stop mixes the units of `x` and of `F` rejects at the large `s` and promotes.
        for s in (1.0e-6, 1.0, 1.0e6, 1.0e8), n in (8, 16)

            prob, J!, _, x0 = ir_problem(K_MID, Array, T; n = n, scale = s)
            step = one_step(prob, copy(x0), ir_method())
            @test step.info.failures == 1
            @test !step.info.promoted
            Jm = similar(x0, length(x0), length(x0))
            J!(Jm, x0, nothing)
            @test norm(step.F .+ Jm * step.dir) <= 10 * IR_TOL * norm(step.F)
        end
    end

    @testset "Check 4: the stage refuses a direction that misses the bound ($T)" for T in SDD_ELTYPES

        # Beyond the GMRES-IR limit the Arnoldi estimate of the preconditioned system meets its
        # tolerance while the true residual of the direction does not, so the stage reports a
        # failure and the step promotes. MU7 accepts this direction, and the assertions on `ok` and
        # on `promoted` fail then.
        prob, J!, _, x0 = ir_problem(K_HIGH, Array, T)
        x = copy(x0)
        Fx = prob.F(similar(x), x, nothing)
        fnorm = rnorm(Fx)
        m = ir_method()
        state = init_state(m, prob, x, nothing)
        state = start!!(state, m, prob, x, Fx, nothing, fnorm)
        Jm = similar(x0, length(x0), length(x0))
        J!(Jm, x0, nothing)
        fac = factorize!!(state.fac, m.linsolve, Jm)
        d₀ = -ldiv!!(similar(x), fac, Fx)
        d_ir, ok, ran, linrel, _ = gmres_ir!!(d₀, m.refinement, state, fac, Jm, Fx, fnorm)
        @test ran
        @test !ok
        @test d_ir === d₀
        @test linrel > 10 * IR_TOL
        # and below the limit the stage accepts, with a direction that meets the bound
        prob_mid, J_mid!, _, x0_mid = ir_problem(K_MID, Array, T)
        xm = copy(x0_mid)
        Fm = prob_mid.F(similar(xm), xm, nothing)
        fn_mid = rnorm(Fm)
        state = init_state(m, prob_mid, xm, nothing)
        state = start!!(state, m, prob_mid, xm, Fm, nothing, fn_mid)
        Jm = similar(x0_mid, length(x0_mid), length(x0_mid))
        J_mid!(Jm, x0_mid, nothing)
        fac = factorize!!(state.fac, m.linsolve, Jm)
        d₀ = -ldiv!!(similar(xm), fac, Fm)
        d_ir, ok, ran, linrel, state₊ = gmres_ir!!(
            d₀, m.refinement, state, fac, Jm, Fm, fn_mid)
        @test ran
        @test ok
        @test d_ir !== d₀
        @test linrel <= 10 * IR_TOL
        @test norm(Fm .+ Jm * d_ir) <= 10 * IR_TOL * fn_mid
        # the stage wrote the promoted factors into the working-precision slot of the state
        @test eltype(state₊.fac_T.factors) === T
        @test Array(state₊.fac_T.factors) == Array(state.fac.factors)
    end

    # ─── Check 5: the comparison table and its recomputation ─────────────────

    @testset "Check 5: the comparison table, recomputed from its raw output" begin
        include(joinpath(@__DIR__, "..", "..", "scripts", "gmres_ir_comparison.jl"))
        raw = generate_gmres_ir_raw_data()
        table = format_gmres_ir_table(raw)
        @test length(raw) == 2 * length(GMRES_IR_KAPPAS)
        @test all(row -> !isempty(row.linear) && !isempty(row.residual), raw)
        @test all(row -> row.steps == length(row.residual) - 1, raw)
        @test recompute_and_verify_gmres_ir_table(raw, table)

        # MU8: a cell that does not recompute, an empty cell, a table that lost a row, and a raw
        # output whose step count is not that of its history are each refused.
        @test !recompute_and_verify_gmres_ir_table(raw, replace(table, "| 1 |" => "| 2 |"))
        @test !recompute_and_verify_gmres_ir_table(raw, replace(table, "| 1 |" => "|  |"))
        @test !recompute_and_verify_gmres_ir_table(
            raw, join(split(table, '\n')[1:(end - 2)], '\n'))
        truncated = copy(raw)
        truncated[1] = merge(truncated[1], (residual = truncated[1].residual[1:(end - 1)],))
        @test !recompute_and_verify_gmres_ir_table(truncated, table)
        # a raw output whose own step count contradicts its history, against a table that agrees
        # with that history: the step-count cross-check alone refuses it
        miscounted = copy(raw)
        miscounted[1] = merge(miscounted[1], (steps = miscounted[1].steps + 1,))
        @test !recompute_and_verify_gmres_ir_table(miscounted, table)
    end

    # ─── Construction, keywords, refusals and allocations ────────────────────

    @testset "the API of the refinement" begin
        @test fieldnames(Refinement) === (:η, :maxiter)
        @test fieldnames(GMRESRefinement) === (:η, :maxiter, :gmres_maxiter, :gmres_tol)
        # the keyword forwarding of the plan's decision D1
        r = Refinement(Float64; gmres_maxiter = 4, gmres_tol = 1e-9)
        @test r isa GMRESRefinement
        @test r.η == sqrt(eps(Float64))
        @test r.maxiter === Int32(5)
        @test r.gmres_maxiter === Int32(4)
        @test r.gmres_tol === 1e-9
        @test Refinement(Float64; gmres_tol = 1e-5) isa GMRESRefinement
        @test Refinement(1e-6, 3) isa Refinement
        @test Refinement() isa Refinement
        g = GMRESRefinement(1e-6, 7, 3, 1e-8)
        @test g.η === 1e-6 && g.maxiter === Int32(7)
        @test g.gmres_maxiter === Int32(3) && g.gmres_tol === 1e-8
        @test GMRESRefinement(Float32).η === sqrt(eps(Float32))
        @test GMRESRefinement(Float64).gmres_maxiter === Int32(30)
        @test isbits(r)
        @test_throws ArgumentError GMRESRefinement(; maxiter = 0)
        @test_throws ArgumentError GMRESRefinement(; gmres_maxiter = 0)
        @test_throws ArgumentError GMRESRefinement(; η = 1.0)
        @test_throws ArgumentError GMRESRefinement(; gmres_tol = 0.0)
        @test_throws ArgumentError GMRESRefinement(; gmres_tol = 1.0)

        # the refusals of a method that takes a refinement reach a GMRESRefinement too
        @test_throws ArgumentError Newton(;
            globalization = LevenbergMarquardt(), refinement = GMRESRefinement())
        @test_throws ArgumentError Newton(; linsolve = GMRES(), refinement = GMRESRefinement())
        prob, _, _, x0 = ir_problem(K_WELL)
        @test_throws ArgumentError init(prob, x0,
            Newton(; linsolve = LUFactorization(Float32), refinement = GMRESRefinement(),
                jacobian = AutoFiniteDiff()))

        # a solve that forms no preconditioned product has no buffer to read
        plain = init(prob, copy(x0),
            Newton(; linsolve = LUFactorization(Float32), refinement = Refinement()))
        @test preconditioned_buffer(plain) === nothing
        # the working-precision slot is zeroed at `init`, so a read before the first promotion reads
        # zeros rather than the storage `similar` left
        sdd = init(prob, copy(x0), ir_method())
        @test all(iszero, promoted_factors(sdd))
        @test eltype(promoted_factors(sdd)) === Float64
        # and a solve whose factorisation runs in the working precision keeps no promoted slot
        same = init(prob, copy(x0),
            Newton(; linsolve = LUFactorization(), refinement = GMRESRefinement()))
        @test promoted_factors(same) === nothing
        # a matrix-free solve keeps no factor at all
        free = init(prob, copy(x0), Newton(; linsolve = GMRES(; maxiter = 4)))
        @test stored_factors(free) === nothing
        @test promoted_factors(free) === nothing
    end

    @testset "a mixed solve allocates nothing after init" begin
        prob, _, _, x0 = ir_problem(K_MID)
        solver = init(prob, copy(x0), ir_method())
        @test sdd_allocations(solver, copy(x0), x0) == 0
        # the control: the probe reads an allocation that is there
        @test Base.allocations(collect, x0) > 0
    end

    @testset "a reused factor keeps the scale of the factor it stored ($T)" for T in SDD_ELTYPES
        # Under reuse the state keeps the factor a step stored, and a step whose GMRES-IR stage
        # succeeds stores the factor it factorised as the LU-IR success does: a state that keeps
        # the slot `init` made solves the next reused step with a scale that factor does not have.
        prob, J!, _, x0 = ir_problem(K_MID, Array, T)
        m = Newton(; linsolve = LUFactorization(Float32), reuse = Int32(3),
            refinement = GMRESRefinement(Float64; gmres_maxiter = 8, gmres_tol = IR_TOL))
        solver = init(prob, copy(x0), m)
        _, st = solve!(solver, copy(x0))
        @test isconverged(st)
        @test st.step_failures >= 1
        @test !st.promoted
        Jm = similar(x0, length(x0), length(x0))
        J!(Jm, x0, nothing)
        fac_ref = factorize!!(init_linear(LUFactorization(Float32), nothing, x0),
            LUFactorization(Float32), Jm)
        @test Array(stored_factors(solver)) == Array(fac_ref.factors)
        @test solver.state.fac.scale_exponent == fac_ref.scale_exponent
    end

    @testset "the tenfold allowance of the true residual bound" begin
        # D3: a stage accepts a direction up to ten times its stated tolerance, the allowance for
        # the rounding between the Arnoldi estimate of the preconditioned system and the true
        # residual of the direction, and no further.
        @test accepts_gmres_ir(1e-9, 1e-8)
        @test accepts_gmres_ir(5.0e-8, 1e-8)
        @test accepts_gmres_ir(1.0e-7, 1e-8)
        @test !accepts_gmres_ir(1.01e-7, 1e-8)
        @test !accepts_gmres_ir(1e-6, 1e-8)
        @test !accepts_gmres_ir(NaN, 1e-8)
        @test !accepts_gmres_ir(Inf, 1e-8)
    end

    @testset "a singular factorisation goes straight to the refactorisation" begin
        # a factor with a zero pivot cannot precondition the inner solve, so the middle stage is
        # skipped and the step refactorises in `T` exactly as a `Refinement` does
        A = zeros(4, 4)
        A[1, 1] = A[2, 2] = A[3, 3] = 1.0
        F!(r, x, p) = (mul!(r, A, x); r)
        J!(J, x, p) = (copyto!(J, A); J)
        prob = NonlinearProblem(F!, J!)
        x0 = [1.0, 1.0, 1.0, 1.0]
        _, st = solve!(init(prob, copy(x0), ir_method(4)), copy(x0))
        @test st.code == SINGULAR
        @test st.step_failures == 1
        @test st.promoted
        _, st_ref = solve!(
            init(prob, copy(x0),
                Newton(; linsolve = LUFactorization(Float32), refinement = Refinement())),
            copy(x0))
        @test st_ref.code == st.code
        @test st_ref.step_failures == st.step_failures
        @test st_ref.promoted == st.promoted
    end

    @testset "a value iterate keeps the LU-IR passes and the refactorisation" begin
        # a `StaticArray` has no storage for a Krylov basis, so a `GMRESRefinement` on one runs the
        # LU-IR passes and, where they fail, the refactorisation in `T`, as a `Refinement` does
        x0 = SVector{2, Float64}(2.0, 3.0)
        F1(x, p) = SVector{2, Float64}(x[1]^2 - 2.0, x[2]^2 - 3.0)
        J1(x, p) = SMatrix{2, 2, Float64}(2.0 * x[1], 0.0, 0.0, 2.0 * x[2])
        prob = NonlinearProblem(F1, J1)
        x, st = solve(prob, x0,
            Newton(; linsolve = LUFactorization(Float32), refinement = GMRESRefinement()))
        @test isconverged(st)
        @test norm(Array(x) - sqrt.([2.0, 3.0])) < 1e-6
    end

    @testset "a GMRESRefinement solve inside a KernelAbstractions kernel on CPU" begin
        # §V names the KA backend: a value iterate carries no Krylov basis, so the kernel reaches
        # the LU-IR passes and the refactorisation, and `GMRESRefinement` has to be `isbits` for the
        # method to travel into the kernel at all.
        # Under `--check-bounds=yes` a bounds check stops the compiler eliding the local
        # `MVector` and `MMatrix` of the static solve, which reach the heap, and a kernel cannot
        # allocate: the launch fails in `gc_pool_alloc`. CI's `julia-runtest` step passes
        # `check_bounds: auto`, so the set runs there.
        for T in SDD_ELTYPES
            Base.JLOptions().check_bounds == 1 && continue
            R = real(T)
            m = Newton(; linsolve = LUFactorization(Float32),
                refinement = GMRESRefinement(T; gmres_maxiter = 4, gmres_tol = IR_TOL))
            @test isbits(m)
            mat = Adapt.adapt(ToReal{R}(), m)
            opt = Options(T)
            x0 = SVector{2, T}(T(3), T(5))
            p = SVector{2, T}(T(4), T(9))
            prob = NonlinearProblem((y, q) -> y .* y .- q)
            _, st_host = solve(prob, x0, mat, p)
            codes = Vector{GeometricSolvers.ReturnCode}(undef, 2)
            xs = Vector{SVector{2, T}}(undef, 2)
            state0 = init_state(mat, prob, x0, p)
            gmres_ir_kernel!(KA_BACKEND)(
                codes, xs, prob, mat, state0, opt, x0, p; ndrange = 2)
            synchronize(KA_BACKEND)
            @test all(==(st_host.code), codes)
            @test all(x -> isapprox(x, SVector{2, T}(T(2), T(3)); rtol = 10 * sqrt(eps(R))), xs)
        end
    end
end
