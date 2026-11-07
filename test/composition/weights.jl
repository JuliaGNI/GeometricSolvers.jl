# `OptimalWeights`, `FixedWeights`, and the shared real-weight least-squares solver in
# `src/composition/weights.jl`: constructor validation, `ToReal` adaptation, real weights on
# complex problems (Check 3), `sin²θ` rank fallback and smaller-residual candidate selection
# (Check 2, MU3, MU4), incremental Cholesky for `m = 3, 4`, the solve over the active
# candidates only, and R1/R3 parity on `Array` and
# `JLArray` under `allowscalar(false)`.

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: FixedWeights, OptimalWeights, ToReal, init_weight_workspace,
                        optimal_weights, rdot, rnorm, solve_optimal_weights

include("../helpers/matrix.jl")

allowscalar(false)

function r1_weights_alloc(G::GM, b::BV, fnorms::FN) where {GM, BV, FN}
    solve_optimal_weights(G, b, fnorms)
    return @allocated solve_optimal_weights(G, b, fnorms)
end

function r3_weights_alloc(work::W, Fx::V, Fxs::TUP, fnorms::FN) where {W, V, TUP, FN}
    optimal_weights(work, Fx, Fxs, fnorms)
    return @allocated optimal_weights(work, Fx, Fxs, fnorms)
end

@testset "OptimalWeights and FixedWeights constructors, validation, and ToReal" begin
    @test Base.isexported(GeometricSolvers, :OptimalWeights)
    @test Base.isexported(GeometricSolvers, :FixedWeights)

    ow_def = OptimalWeights()
    @test ow_def.ρ === 1.0
    @test isbits(ow_def)
    @test OptimalWeights(; ρ = 0.75) === OptimalWeights(0.75)
    @test OptimalWeights(2).ρ === 2.0

    for bad_ρ in (0, 0.0, -0.5, -1, NaN, Inf, -Inf)
        @test_throws ArgumentError OptimalWeights(bad_ρ)
        @test_throws ArgumentError OptimalWeights(; ρ = bad_ρ)
    end

    fw2 = FixedWeights(0.4, 0.6)
    @test fw2.weights === (0.4, 0.6)
    @test isbits(fw2)
    @test FixedWeights((0.4, 0.6)) === fw2
    @test FixedWeights(1, 2).weights === (1.0, 2.0)
    @test FixedWeights(3).weights === (3.0,)
    @test FixedWeights(-0.5, 1.5).weights === (-0.5, 1.5)

    @test_throws ArgumentError FixedWeights()
    @test_throws ArgumentError FixedWeights(())
    for bad_w in (NaN, Inf, -Inf)
        @test_throws ArgumentError FixedWeights(bad_w)
        @test_throws ArgumentError FixedWeights(1.0, bad_w)
        @test_throws ArgumentError FixedWeights((bad_w, 1.0))
    end

    for T in ELTYPES
        R = real(T)
        ow_r = @inferred Adapt.adapt(ToReal{R}(), OptimalWeights(0.8))
        @test ow_r isa OptimalWeights{R}
        @test typeof(ow_r.ρ) === R
        @test ow_r.ρ === R(0.8)

        fw_r = @inferred Adapt.adapt(ToReal{R}(), FixedWeights(0.4, 0.6))
        @test fw_r isa FixedWeights{2, R}
        @test eltype(fw_r.weights) === R
        @test fw_r.weights === (R(0.4), R(0.6))
    end
end

@testset "Check 2: sin²θ rank fallback (MU3, MU4), $R" for R in REAL_ELTYPES
    # Orthogonal candidates: full rank, no fallback
    G_orth = SMatrix{2, 2, R}(R(4), zero(R), zero(R), R(9))
    b_orth = SVector{2, R}(R(2), R(3))
    fn_orth = (R(1), R(2))
    w_orth, fb_orth = @inferred solve_optimal_weights(G_orth, b_orth, fn_orth)
    @test !fb_orth
    @test isapprox(w_orth[1], R(0.5); rtol = 8 * eps(R))
    @test isapprox(w_orth[2], R(1) / R(3); rtol = 8 * eps(R))
    if Base.JLOptions().check_bounds != 1
        # Skipped where a run forces `--check-bounds=yes`.
        @test r1_weights_alloc(G_orth, b_orth, fn_orth) == 0
    end

    # Identical candidates (`sin²θ == 0 < 1e-3`): tie keeps index 1
    G_eq = SMatrix{2, 2, R}(R(4), R(4), R(4), R(4))
    b_eq = SVector{2, R}(R(2), R(2))
    fn_tie = (R(1.5), R(1.5))
    w_tie, fb_tie = @inferred solve_optimal_weights(G_eq, b_eq, fn_tie)
    @test fb_tie
    w1_expected = R(2) / (R(4) * (one(R) + eps(R)))
    @test w_tie[1] === w1_expected
    @test iszero(w_tie[2])

    # Nearly-collinear candidates (`0 < sin²θ < 1e-3`) where candidate 2 has smaller
    # residual (catches MU3 if `sin²θ < τ_sin2` is disabled, and MU4 if candidate 1 is kept)
    δ1 = SVector{2, R}(one(R), zero(R))
    δ2 = SVector{2, R}(one(R), R(1e-3) * R(0.5))
    G_col = SMatrix{2, 2, R}(
        rdot(δ1, δ1), rdot(δ2, δ1),
        rdot(δ1, δ2), rdot(δ2, δ2))
    b_col = SVector{2, R}(R(0.8), R(0.9))
    # Candidate 2 has smaller residual norm `0.4 < 0.7`, so candidate 2 is kept
    w_col2, fb_col2 = @inferred solve_optimal_weights(G_col, b_col, (R(0.7), R(0.4)))
    @test fb_col2
    @test iszero(w_col2[1])
    @test w_col2[2] === b_col[2] / (G_col[2, 2] * (one(R) + eps(R)))

    # When candidate 1 has smaller residual norm `0.3 < 0.7`, candidate 1 is kept
    w_col1, fb_col1 = @inferred solve_optimal_weights(G_col, b_col, (R(0.3), R(0.7)))
    @test fb_col1
    @test w_col1[1] === b_col[1] / (G_col[1, 1] * (one(R) + eps(R)))
    @test iszero(w_col1[2])

    # Degenerate zero-Gram candidate falls back cleanly to weight 1 on the kept candidate
    G_zero = zero(SMatrix{2, 2, R})
    b_zero = zero(SVector{2, R})
    w_z, fb_z = @inferred solve_optimal_weights(G_zero, b_zero, (R(2), R(1)))
    @test fb_z
    @test w_z === (zero(R), one(R))

    # Non-finite Gram entry triggers fallback without throwing
    G_nan = SMatrix{2, 2, R}(R(NaN), zero(R), zero(R), R(4))
    b_nan = SVector{2, R}(one(R), R(2))
    w_nan, fb_nan = @inferred solve_optimal_weights(G_nan, b_nan, (R(1), R(2)))
    @test fb_nan
    @test all(isfinite, w_nan)
end

@testset "m = 1, 3, 4 solve_optimal_weights incremental Cholesky, $R" for R in REAL_ELTYPES
    # m = 1
    G1 = SMatrix{1, 1, R}((R(4),))
    b1 = SVector{1, R}((R(2),))
    w1, fb1 = @inferred solve_optimal_weights(G1, b1, (R(0.5),))
    @test !fb1
    @test w1[1] === R(2) / (R(4) * (one(R) + eps(R)))

    w1_z, fb1_z = @inferred solve_optimal_weights(
        zero(SMatrix{1, 1, R}), zero(SVector{1, R}), (R(0.5),))
    @test fb1_z
    @test w1_z === (one(R),)

    # m = 3: candidates 1 and 3 are collinear, candidate 3 has smaller residual than
    # candidate 1, and candidate 2 is orthogonal -> candidate 1 is dropped
    v1 = SVector{3, R}(one(R), zero(R), zero(R))
    v2 = SVector{3, R}(zero(R), R(2), zero(R))
    v3 = SVector{3, R}(R(2), zero(R), zero(R))
    vs = (v1, v2, v3)
    Fx3 = SVector{3, R}(-one(R), -R(2), zero(R))
    G3 = SMatrix{3, 3, R, 9}(ntuple(k -> rdot(vs[mod1(k, 3)], vs[cld(k, 3)]), Val(9)))
    b3 = SVector{3, R}(ntuple(i -> -rdot(vs[i], Fx3), Val(3)))
    fn3 = (R(0.9), R(0.6), R(0.3))
    w3, fb3 = @inferred solve_optimal_weights(G3, b3, fn3)
    @test fb3
    @test iszero(w3[1])
    @test isapprox(w3[2], one(R); rtol = 16 * eps(R))
    @test isapprox(w3[3], R(0.5); rtol = 16 * eps(R))

    # m = 4: four orthogonal basis directions in R^4 -> all 4 survive with no fallback
    G4 = SMatrix{4, 4, R, 16}(
        ntuple(k -> mod1(k, 4) == cld(k, 4) ? R(4) : zero(R), Val(16)))
    b4 = SVector{4, R}(R(1), R(2), R(3), R(4))
    fn4 = (R(0.4), R(0.3), R(0.2), R(0.1))
    w4, fb4 = @inferred solve_optimal_weights(G4, b4, fn4)
    @test !fb4
    for i in 1:4
        @test isapprox(w4[i], R(i) / R(4); rtol = 16 * eps(R))
    end
end

@testset "solve_optimal_weights over the active candidates only, $R" for R in REAL_ELTYPES
    # m = 3 as above with candidate 1 inactive: 2 and 3 are orthogonal, both survive, and
    # leaving candidate 1 out is no rank fallback
    v1 = SVector{3, R}(one(R), zero(R), zero(R))
    v2 = SVector{3, R}(zero(R), R(2), zero(R))
    v3 = SVector{3, R}(R(2), zero(R), zero(R))
    vs = (v1, v2, v3)
    Fx3 = SVector{3, R}(-one(R), -R(2), zero(R))
    G3 = SMatrix{3, 3, R, 9}(ntuple(k -> rdot(vs[mod1(k, 3)], vs[cld(k, 3)]), Val(9)))
    b3 = SVector{3, R}(ntuple(i -> -rdot(vs[i], Fx3), Val(3)))
    fn3 = (R(0.3), R(0.6), R(0.9))
    w3, fb3 = @inferred solve_optimal_weights(G3, b3, fn3, (false, true, true))
    @test !fb3
    @test iszero(w3[1])
    @test isapprox(w3[2], one(R); rtol = 16 * eps(R))
    @test isapprox(w3[3], R(0.5); rtol = 16 * eps(R))

    # one active candidate: the one-weight secant step along it, no rank fallback
    w3_1, fb3_1 = @inferred solve_optimal_weights(G3, b3, fn3, (false, false, true))
    @test !fb3_1
    @test w3_1 === (zero(R), zero(R), R(2) / (R(4) * (one(R) + eps(R))))

    # m = 4 orthogonal with candidates 2 and 4 inactive
    G4 = SMatrix{4, 4, R, 16}(
        ntuple(k -> mod1(k, 4) == cld(k, 4) ? R(4) : zero(R), Val(16)))
    b4 = SVector{4, R}(R(1), R(2), R(3), R(4))
    fn4 = (R(0.4), R(0.3), R(0.2), R(0.1))
    w4, fb4 = @inferred solve_optimal_weights(G4, b4, fn4, (true, false, true, false))
    @test !fb4
    @test iszero(w4[2]) && iszero(w4[4])
    @test isapprox(w4[1], R(1) / R(4); rtol = 16 * eps(R))
    @test isapprox(w4[3], R(3) / R(4); rtol = 16 * eps(R))

    # m = 2: the kept candidate's secant weight, whatever the residual norms; a zero direction
    # along it is a rank fallback to weight one
    G2 = SMatrix{2, 2, R}(R(4), R(1), R(1), R(2))
    b2 = SVector{2, R}(R(1), R(3))
    w2, fb2 = @inferred solve_optimal_weights(G2, b2, (R(1), R(2)), (false, true))
    @test !fb2
    @test w2 === (zero(R), R(3) / (R(2) * (one(R) + eps(R))))
    w2_z, fb2_z = @inferred solve_optimal_weights(
        zero(SMatrix{2, 2, R}), zero(SVector{2, R}), (R(1), R(2)), (true, false))
    @test fb2_z
    @test w2_z === (one(R), zero(R))
end

@testset "Check 3 & 6: R1/R3 optimal_weights parity on Array/JLArray, $T" for T in ELTYPES
    R = real(T)
    c = T <: Complex ? T(1, 1) : one(T)
    Fx_s = SVector{3, T}(c * T(2), c * T(-1), c * T(0.5))
    F1_s = SVector{3, T}(c * T(1), c * T(-0.5), c * T(0.5))
    F2_s = SVector{3, T}(c * T(2), c * T(-0.2), c * T(0.1))
    Fxs_s = (F1_s, F2_s)
    fnorms = (rnorm(F1_s), rnorm(F2_s))

    w_r1, fb_r1 = @inferred optimal_weights(nothing, Fx_s, Fxs_s, fnorms)
    @test !fb_r1
    @test eltype(w_r1) === R

    for AT in ARRAY_BACKENDS
        Fx_a = AT(collect(Fx_s))
        Fxs_a = (AT(collect(F1_s)), AT(collect(F2_s)))
        work = init_weight_workspace(OptimalWeights(one(R)), Fx_a, Val(2))
        w_r3, fb_r3 = @inferred optimal_weights(work, Fx_a, Fxs_a, fnorms)
        @test fb_r3 === fb_r1
        @test eltype(w_r3) === R
        @test isapprox(SVector{2, R}(w_r3), SVector{2, R}(w_r1); rtol = 16 * eps(R))
        if AT === Array && Base.JLOptions().check_bounds != 1
            # Skipped where a run forces `--check-bounds=yes`.
            @test r3_weights_alloc(work, Fx_a, Fxs_a, fnorms) == 0
        end
    end
end
