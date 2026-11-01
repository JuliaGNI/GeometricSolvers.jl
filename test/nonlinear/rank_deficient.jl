# `Newton` with a rank-deficient Jacobian: the truncated minimum-norm step of the SVD, with the
# measured slope, on an `SVector`, an `Array` and a `JLArray`; the LU, which reports `SINGULAR`
# where its factorisation meets a zero pivot; and the zero Jacobian, whose SVD factor has rank 0.

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, DogLeg, ExactStep, MeasuredSlope, SINGULAR,
                        STALLED, SUCCESS, Static, StrongWolfe, globalize!!,
                        init_globalization
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: norm
using StaticArrays: SMatrix, SVector
using Test

include("../helpers/matrix.jl")

allowscalar(false)

# The rank-1 consistent system `F(x) = (|x|² − 1) w` with `w = (1, 2, 3)`: its Jacobian `2 w xᵀ`
# has rank 1 everywhere, and `F(x) ∈ range J`, so the truncated step is the Newton step. The dense
# form takes the norm by `sum` and the weights by broadcast, so that a `JLArray` runs it without a
# scalar read.
sphere_r1(::Type{T}) where {T} = (x, p) -> SVector{3, T}(1, 2, 3) * (sum(abs2, x) - 1)
sphere_r3!(y, x, p) = (y .= (sum(abs2, x) .- 1) .* (1:length(x)); y)
"The start `(½, ½, ½)` and the problem on the storage `S`: `SVector`, or an array type."
function sphere(::Type{SVector}, ::Type{T}) where {T}
    return NonlinearProblem(sphere_r1(T)), SVector{3, T}(0.5, 0.5, 0.5)
end
function sphere(::Type{AT}, ::Type{T}) where {AT, T}
    return NonlinearProblem(sphere_r3!), AT(T[0.5, 0.5, 0.5])
end

# The full-rank system `x² − 2`, componentwise.
root_r1(x, p) = x .^ 2 .- 2
root_r3!(y, x, p) = (y .= x .^ 2 .- 2; y)
root(::Type{SVector}, ::Type{T}) where {T} = NonlinearProblem(root_r1), SVector{2, T}(3, 5)
root(::Type{AT}, ::Type{T}) where {AT, T} = NonlinearProblem(root_r3!), AT(T[3, 5])

const STORAGES = (SVector, ARRAY_BACKENDS...)

@testset "the rank-1 system" begin
    @testset "SVD with $(nameof(typeof(g))), $T, $(nameof(S))" for T in REAL_ELTYPES,
        g in (Static(), Backtracking(), DogLeg()), S in STORAGES
        prob, x0 = sphere(S, T)
        f0 = norm(sphere_r1(T)(SVector{3, T}(0.5, 0.5, 0.5), nothing))
        _, status = solve(prob, x0, Newton(; linsolve = SVDFactorization(), globalization = g))
        @test status.code === SUCCESS
        @test status.fnorm ≤ Options(T).f_reltol * f0
        @test status.iterations ≥ 1
    end
    # the host LU reports no zero pivot on this matrix: LAPACK's reciprocal multipliers leave a
    # rounding residue
    @testset "LU with $(nameof(typeof(g))), $T, $(nameof(S))" for T in REAL_ELTYPES,
        g in (Static(), Backtracking()), S in (SVector, JLArray)
        prob, x0 = sphere(S, T)
        _, status = solve(prob, x0, Newton(; linsolve = LUFactorization(), globalization = g))
        @test status.code === SINGULAR
    end
end

# A globalization that records the step kind `Newton` passes and takes the step of `Static`.
const KINDS = Any[]
struct Recording{G}
    g::G
end
GeometricSolvers.init_globalization(r::Recording, x) = init_globalization(r.g, x)
function GeometricSolvers.globalize!!(
        g, r::Recording, prob, x, Fx, d, p, jvpprep, stepkind = MeasuredSlope())
    push!(KINDS, stepkind)
    return globalize!!(g, r.g, prob, x, Fx, d, p, jvpprep, stepkind)
end

@testset "the step kind" begin
    @testset "the rank-1 system with the SVD, $T, $(nameof(S))" for T in REAL_ELTYPES,
        S in STORAGES

        prob, x0 = sphere(S, T)
        empty!(KINDS)
        solve(prob, x0,
            Newton(; linsolve = SVDFactorization(), globalization = Recording(Static(one(T)))))
        @test !isempty(KINDS)
        @test all(k -> k === MeasuredSlope(), KINDS)
    end
    @testset "the full-rank system with $(nameof(M)), $T, $(nameof(S))" for T in REAL_ELTYPES,
        M in (SVDFactorization, LUFactorization, QRFactorization), S in STORAGES

        prob, x0 = root(S, T)
        empty!(KINDS)
        _, status = solve(prob, x0,
            Newton(; linsolve = M(), globalization = Recording(Static(one(T)))))
        @test status.code === SUCCESS
        @test !isempty(KINDS)
        @test all(k -> k === ExactStep(), KINDS)
    end
end

# `F ≡ (1, 1)` with the analytic `J ≡ 0`: the SVD factor has rank 0 and `SUCCESS`, its step is
# zero, and the loop's zero-step rule returns `STALLED`; `DogLeg` takes no Newton step of zero
# norm, and its zero gradient `Jᵀ F` returns `STALLED` with the iterate kept.
F0_r1(x, p) = SVector(1.0, 1.0)
J0_r1(x, p) = zeros(SMatrix{2, 2, Float64})
F0_r3!(y, x, p) = (y .= 1; y)
J0_r3!(J, x, p) = (J .= 0; J)
zero_jacobian(::Type{SVector}) = NonlinearProblem(F0_r1, J0_r1), SVector(0.0, 0.0)
zero_jacobian(::Type{AT}) where {AT} = NonlinearProblem(F0_r3!, J0_r3!), AT([0.0, 0.0])

@testset "the zero Jacobian" begin
    @testset "SVD with $(nameof(typeof(g))), $(nameof(S))" for g in (
            Static(), Backtracking(), Bisection(), StrongWolfe()),
        S in STORAGES

        prob, x0 = zero_jacobian(S)
        _, status = solve(prob, x0, Newton(; linsolve = SVDFactorization(), globalization = g))
        @test status.code === STALLED
    end
    @testset "SVD with DogLeg, $(nameof(S))" for S in STORAGES
        prob, x0 = zero_jacobian(S)
        x, status = solve(prob, x0,
            Newton(; linsolve = SVDFactorization(), globalization = DogLeg()))
        @test status.code === STALLED
        @test iszero(Array(x))
    end
    @testset "LU on an SVector" begin
        prob, x0 = zero_jacobian(SVector)
        _, status = solve(prob, x0, Newton(; linsolve = LUFactorization()))
        @test status.code === SINGULAR
    end
end
