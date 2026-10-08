# The R3 Jacobian–vector product: one pushforward, exactly one evaluation of the residual.
#
# `jvp!!` is what makes `StrongWolfe` and `Bisection` affordable — the merit derivative
# `φ′(α) = F(x+αd)ᵀ J(x+αd) d` costs one directional derivative rather than n — and what the
# mixed-precision refinement residual uses. Both uses are one pushforward per call, so the call
# count is part of the contract and is tested here, not only the value.

using ADTypes: AutoForwardDiff
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test

using GeometricSolvers: jacobian!!, jvp!!, prepare_ad

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
    # touched and the `N = 1` buffers keep their identity.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, r, x, p)
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

@testset "jvp!! on a holomorphic residual in a complex element type" begin
    # The complex counterpart of the Jacobian test in `test/ad/jacobian.jl`: `Complex{Dual{…,1}}`
    # buffers, one partial, and the complex derivative `2 x + p`.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in filter(T -> T <: Complex, ELTYPES)

        prob = StubProblem(Holomorphic())
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, r, x, p)
        d = Array(x) .+ Array(x) .+ Array(p)
        Jv = similar(x)
        for j in (1, N_AD)
            v = AT(T[i == j ? one(T) : zero(T) for i in 1:N_AD])
            jvp!!(Jv, prep, prob, x, v, p)
            # the residual is diagonal, so the product with a unit vector is one entry of `2x + p`
            @test Array(Jv) == [i == j ? d[j] : zero(T) for i in 1:N_AD]
        end

        # and on a complex vector that is not a unit vector: a seed that dropped the vector's
        # imaginary component, or kept only it, would differ here. Approximate, and the only
        # approximate comparison in these files: the dual pass multiplies the partial by the
        # vector inside the residual's arithmetic, so the complex products are rounded in a
        # different order from `d .* v`, which costs the last bit or two.
        v = AT(T[(i + 2) / (N_AD + 1) + im * (i + 1) / (N_AD + 3) for i in 1:N_AD])
        jvp!!(Jv, prep, prob, x, v, p)
        @test Array(Jv) ≈ d .* Array(v) rtol = 8 * eps(real(T))
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

    # and with a chunk below the iterate: the pushforward is one dual pass whatever the chunk
    # size, so it stays at zero — at `n` and at `4n` alike.
    @testset "chunk $CHUNK, n = $n" for n in (N_AD, 4 * N_AD)
        xn, pn, rn = ad_inputs(Array, T, n)
        prepn = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, rn, xn, pn)
        vn = similar(xn)
        vn .= one(T)
        @test jvp_allocations(similar(xn), prepn, prob, xn, vn, pn) == 0
    end
end
