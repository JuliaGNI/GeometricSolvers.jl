# The R3 Jacobian–vector product: one pushforward, exactly one evaluation of the residual.
#
# `jvp!!` is what makes `StrongWolfe` and `Bisection` affordable (§1.6: the merit derivative
# `φ′(α) = F(x+αd)ᵀ J(x+αd) d` costs one directional derivative rather than n) and what the
# mixed-precision refinement residual uses (§1.4). Both uses are one pushforward per call, so the
# call count is part of the contract and is tested here, not only the value.

using ADTypes: AutoForwardDiff
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test

using GeometricSolvers: ChunkedForwardDiff, jacobian!!, jvp!!, prepare_ad

include("../helpers/matrix.jl")
include("../helpers/adproblems.jl")

const N_AD = 7
const CHUNK = 3

# As in `test/ad/jacobian.jl`: scalar indexing on a device array is an error for every test here.
allowscalar(false)

function jvp_allocations(Jv::A, prep::P, prob::Q, x::X, v::V, p::R) where {A, P, Q, X, V, R}
    jvp!!(Jv, prep, prob, x, v, p)
    return @allocated jvp!!(Jv, prep, prob, x, v, p)
end

@testset "jvp!! is the Jacobian times the vector, column by column" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        Jh = Array(J)
        Jv = similar(x)
        for j in 1:N_AD
            # a unit vector, so that the product is column j of the Jacobian exactly
            v = AT(T[i == j ? one(T) : zero(T) for i in 1:N_AD])
            @test @inferred(jvp!!(Jv, prep, prob, x, v, p)) === Jv
            @test Array(Jv) == Jh[:, j]
        end

        # and on a vector that is not a unit vector, against the dense product
        v = AT(T[(i + 2) / (N_AD + 1) for i in 1:N_AD])
        jvp!!(Jv, prep, prob, x, v, p)
        @test Array(Jv) ≈ Jh * Array(v) rtol = 4 * eps(T)
    end
end

@testset "jvp!! evaluates the residual exactly once" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        counting = Counting(Coupled(cyclic_perm(AT, N_AD)))
        prob = StubProblem(counting)
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        Jv = similar(x)
        v = AT(T[(i + 2) / (N_AD + 1) for i in 1:N_AD])
        # the preparation evaluates the residual as often as it needs; the count starts here
        counting.calls[] = 0
        jvp!!(Jv, prep, prob, x, v, p)
        @test counting.calls[] == 1
        jvp!!(Jv, prep, prob, x, v, p)
        @test counting.calls[] == 2
    end
end

@testset "jvp!! through the chunked mode uses the N = 1 buffers only" begin
    # The chunked mode's JVP is one dual pass with one partial, so the Jacobian buffers are not
    # touched and the `N = 1` buffers keep their identity. Built directly on an `Array` as well,
    # because `prepare_ad` selects DI there.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = ChunkedForwardDiff(AutoForwardDiff(; chunksize = CHUNK), prob, r, x)
        slots = (prep.xdual, prep.rdual, prep.xdual1, prep.rdual1)
        Jv = similar(x)
        v = AT(T[i == 2 ? one(T) : zero(T) for i in 1:N_AD])
        jvp!!(Jv, prep, prob, x, v, p)
        @test Array(Jv) == forwarddiff_jacobian(host, x, p)[:, 2]
        @test prep.xdual === slots[1]
        @test prep.rdual === slots[2]
        @test prep.xdual1 === slots[3]
        @test prep.rdual1 === slots[4]
    end
end

@testset "jvp!! allocates nothing on an Array, $T" for T in REAL_ELTYPES
    prob = StubProblem(Scaled())
    x, p, r = ad_inputs(Array, T, N_AD)
    prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
    Jv = similar(x)
    v = similar(x)
    v .= one(T)
    @test jvp_allocations(Jv, prep, prob, x, v, p) == 0
end
