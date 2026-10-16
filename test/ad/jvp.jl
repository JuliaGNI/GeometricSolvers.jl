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

# The R1 value methods of `jvp!!` — check 5 of §13.L-a, beside the R3 pushforward above.
#
# The preparation of a JVP is built from the problem and never from a method's operator (§4.3), so
# for a problem that carries `J` R1 computes `J(x, p) * v` itself and the comparison is exact;
# where the value has to come from a dual pass, the comparison is the one the R3 tests above make,
# for the same reason. The problems are the value forms of `../helpers/adproblems.jl`: its stubs
# are in-place and cannot be called with an `SVector` at all.

using StaticArrays: SVector

using GeometricSolvers: NonlinearProblem

include("../helpers/r1problems.jl")

# The allocation assertion goes through a function barrier, as in the R3 tests above.
function r1_jvp_allocations(ad::A, prob::P, x::X, v::V, p::Q) where {A, P, X, V, Q}
    jvp!!(nothing, ad, prob, x, v, p)
    return @allocated jvp!!(nothing, ad, prob, x, v, p)
end

@testset "the R1 jvp!! reads the problem's J, not a method's operator, $T" for T in ELTYPES
    x = r1_cube_input(T, N_AD)
    p = SVector{N_AD, T}(ones(T, N_AD))
    prob = NonlinearProblem(R1Cube(), r1_cube_jacobian)
    unit(j) = SVector{N_AD, T}([i == j ? one(T) : zero(T) for i in 1:N_AD])

    # A unit direction makes the dense product one column of `J`, with no sum to round, so this
    # comparison is `==` in every element type: `jvp!!` computes `J(x, p) * v` itself, in one
    # expression, and reads no dual. (A general direction is not exactly comparable for a complex
    # element type: `SMatrix * SVector` uses `muladd`, and whether that fuses into an FMA depends
    # on the compilation context, which costs the last bit. The product of the R3 tests above
    # carries a tolerance for the same reason.)
    for j in 1:N_AD
        @test @inferred(jvp!!(nothing, nothing, prob, x, unit(j), p)) ==
              prob.J(x, p) * unit(j)
    end

    # The analytic Jacobian and the one the duals build differ in the last bit at the entries of
    # `r1_cube_input`, so at least one unit direction separates the two: the assertions above are
    # a test, and a `nothing` preparation sent to ForwardDiff fails one of them.
    @test any(
        j -> jvp!!(nothing, AutoForwardDiff(), prob, x, unit(j), p) !=
             prob.J(x, p) * unit(j), 1:N_AD)
end

@testset "the R1 jvp!! seeds a complex direction on both its parts, $T" for T in filter(
    T -> T <: Complex, ELTYPES)
    x, p = r1_inputs(T, N_AD)
    prob = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)
    J = prob.J(x, p)

    # a unit direction gives one column of the Jacobian, exactly: `r_i = x_i² + p_i x_i`, so the
    # dual pass evaluates `(x_i + x_i) v_i + p_i v_i`
    for j in (1, N_AD)
        e = SVector{N_AD, T}([i == j ? one(T) : zero(T) for i in 1:N_AD])
        @test jvp!!(nothing, AutoForwardDiff(), prob, x, e, p) == J * e
    end

    # and a direction with a nonzero imaginary part: a seed that took the real part of the
    # direction in both components of the complex dual — the fault that `adjoint` spelled
    # `transpose` is the real form of — differs here, and in no real element type. Approximate,
    # and the only approximate comparison of the R1 tests: the dual pass multiplies the partial by
    # the direction inside the residual's arithmetic, which rounds in a different order from
    # `J * v`. The tolerance is the one of the R3 test above, eight ulps.
    R = real(T)
    v = SVector{N_AD, T}(
        [complex(R(i + 2) / R(N_AD + 1), R(i + 1) / R(N_AD + 3)) for i in 1:N_AD])
    @test isapprox(jvp!!(nothing, AutoForwardDiff(), prob, x, v, p), J * v;
        rtol = 8 * eps(R))
end

@testset "the R1 jvp!! is inferred and allocation-free, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_AD)
    prob = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)

    @test @inferred(jvp!!(nothing, AutoForwardDiff(), prob, x, x, p)) isa SVector{N_AD, T}
    @test r1_jvp_allocations(nothing, prob, x, x, p) == 0
    @test r1_jvp_allocations(AutoForwardDiff(), prob, x, x, p) == 0
end
