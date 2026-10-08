# The R3 Jacobian: DifferentiationInterface on an `Array`, the own chunked forward mode on a
# device array, and the ways the plan says the chunked mode can be wrong (§13.S, "Verify it").
#
# Every comparison is exact (`==`), not approximate: the residuals of `test/helpers/adproblems.jl`
# are broadcasts with no reduction, so the arithmetic per element is the same on every backend and
# the same as `ForwardDiff.jacobian`'s on an `Array`. A tolerance here would hide exactly the
# faults the file is for.

using ADTypes: AutoFiniteDiff, AutoForwardDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: Diagonal
using Test

using GeometricSolvers: ChunkedForwardDiff, DIJacobian, jacobian!!, prepare_ad

include("../helpers/matrix.jl")
include("../helpers/adproblems.jl")

# The tag a preparation carries, on either path: the type parameter of the chunked mode, the back
# end's `tag` field on the DI path.
ad_tag(::ChunkedForwardDiff{N, Tg}) where {N, Tg} = Tg
ad_tag(prep::DIJacobian) = typeof(prep.backend.tag)

# The size the plan names: a chunk size of 3 does not divide 7, so the last chunk is partial.
const N_AD = 7
const CHUNK = 3

# Scalar indexing on a device array is an error for every test in this file, so a path that falls
# back to a scalar loop — DI's `jacobian!` on a `JLArray`, an `MtlArray` or a `CuArray` — fails
# here rather than passing slowly. A non-interactive session disallows it already; this states it,
# and the testset below shows that the setting is in force rather than assumed.
allowscalar(false)

@testset "scalar indexing on a device array is an error in this file" begin
    @test_throws ErrorException JLArray(zeros(Float32, 2))[1]
end

# The allocation assertion goes through a function barrier whose arguments all have concrete
# types, and which calls the function once before it measures: at testset scope Julia 1.11 boxes
# the captured values and the assertion fails although the code allocates nothing.
function jacobian_allocations(J::A, prep::P, prob::Q, x::X, p::R) where {A, P, Q, X, R}
    jacobian!!(J, prep, prob, x, p)
    return @allocated jacobian!!(J, prep, prob, x, p)
end

@testset "the AD path is selected by the storage of the iterate, once" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        # `Array` goes through DI, every device array through the own chunked mode: DI's
        # `jacobian!` raises "Scalar indexing is disallowed" on a `JLArray`, an `MtlArray` and a
        # `CuArray` (§3.1, part D).
        if AT === Array
            @test prep isa DIJacobian
        else
            @test prep isa ChunkedForwardDiff
            @test !(prep isa DIJacobian)
        end
    end

    # A back end other than ForwardDiff has no device method at all, rather than a fallback that
    # would fail deep inside DI with a scalar-indexing error (§3.1, §4.6).
    prob = StubProblem(Scaled())
    x, p, r = ad_inputs(JLArray, Float64, N_AD)
    @test_throws ArgumentError prepare_ad(AutoFiniteDiff(), prob, r, x, p)
end

@testset "the Jacobian agrees with ForwardDiff on an Array" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        @test @inferred(jacobian!!(J, prep, prob, x, p)) === J
        @test Array(J) == forwarddiff_jacobian(host, x, p)
        # the comparison is a test only because the Jacobian is neither symmetric nor diagonal
        @test Array(J) != transpose(Array(J))
    end
end

@testset "the chunk size does not divide n: n = $N_AD, chunk = $CHUNK" begin
    # The partial last chunk is where a column gets no seed, or the same seed twice. Scalar
    # indexing is disallowed for the whole file, so a scalar fallback fails here.
    @testset "$T" for T in REAL_ELTYPES
        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(JLArray, N_AD)))
        x, p, r = ad_inputs(JLArray, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, r, x, p)
        @test prep isa ChunkedForwardDiff{CHUNK}
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == forwarddiff_jacobian(host, x, p)
    end

    # The same through the chunked mode on an `Array`, so that a fault in the tail chunk cannot
    # hide behind the device path alone: `ChunkedForwardDiff` is built directly here, because
    # `prepare_ad` selects DI for an `Array`.
    @testset "the chunked mode itself, on an Array, $T" for T in REAL_ELTYPES
        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(host)
        x, p, r = ad_inputs(Array, T, N_AD)
        prep = ChunkedForwardDiff(AutoForwardDiff(; chunksize = CHUNK), prob, r, x)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        @test J == forwarddiff_jacobian(host, x, p)
    end
end

@testset "a residual that differentiates: the tag is this solver's own" begin
    # `Nested` calls `ForwardDiff.derivative` in its body. Without an owned tag the inner and the
    # outer perturbation are confused, and the result is wrong rather than an error.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Nested())
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == forwarddiff_jacobian(Nested(), x, p)
        # and the reference is itself right: r_i = 2 p_i x_i², so J = Diagonal(4 p_i x_i). The two
        # expressions round differently, so this one carries a tolerance, named here: four ulps of
        # the element type.
        @test Array(J) ≈ Diagonal(4 .* Array(p) .* Array(x)) rtol = 4 * eps(T)

        # The value comparison above does not by itself show that the tag is owned: the
        # `ForwardDiff.derivative` inside the residual builds a tag of its own, which differs from
        # `Nothing` as much as from ours, so the nesting still comes out right with no tag at all.
        # (Measured: the mutant `tag_of(...) = nothing` survives the two tests above.) What §13.S
        # asks for is therefore asserted directly, on both paths: the tag is this solver's own,
        # named for the residual and the element type, and never `Nothing`.
        @test ad_tag(prep) === typeof(ForwardDiff.Tag(prob.F, T))
        @test ad_tag(prep) <: ForwardDiff.Tag
    end
end

@testset "the parameters are a constant, and a replacement takes effect" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Scaled())
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        # `r = p .* x`, so the Jacobian is `Diagonal(p)` exactly. A parameter differentiated by
        # mistake puts `x` into the diagonal instead.
        @test Array(J) == Diagonal(Array(p))

        # the same array, new values: no new solver, and no stale preparation
        p .= 2 .* p
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == Diagonal(Array(p))

        # a different array object, as a caller that rebuilt its parameters passes
        q = AT(3 .* Array(p))
        jacobian!!(J, prep, prob, x, q)
        @test Array(J) == Diagonal(Array(q))
        @test Array(J) != Diagonal(Array(p))
    end
end

@testset "the dual buffers are allocated once, in prepare_ad" begin
    # On an `Array`: exactly zero, in the cold process `run-tests.jl` starts for this file. At
    # `n = 7` the default chunk size is `n`, so DI runs one vector-mode pass; a chunk below `n`
    # costs 48 bytes per call inside DI, which is K3 of `KNOWN_ISSUES.md` and not reachable from
    # here.
    @testset "no allocation on an Array, $T" for T in REAL_ELTYPES
        prob = StubProblem(Scaled())
        x, p, r = ad_inputs(Array, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        @test jacobian_allocations(J, prep, prob, x, p) == 0
        # the control: the barrier does see an allocation
        @test (@allocated similar(x, N_AD, N_AD)) > 0
    end

    # On a device array the dual buffers are the state slots, and they stay the arrays that
    # `prepare_ad` made. A `JLArray` broadcast allocates host bookkeeping per call, so an exact
    # zero is not reachable there (§13.S); what is required is that no call allocates a dual
    # buffer — which identity shows — and that the host allocation does not grow from call to call.
    @testset "the state slots keep their identity on a JLArray, $T" for T in REAL_ELTYPES
        prob = StubProblem(Coupled(cyclic_perm(JLArray, N_AD)))
        x, p, r = ad_inputs(JLArray, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, r, x, p)
        slots = (prep.xdual, prep.rdual, prep.xdual1, prep.rdual1)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        jacobian!!(J, prep, prob, x, p)
        @test prep.xdual === slots[1]
        @test prep.rdual === slots[2]
        @test prep.xdual1 === slots[3]
        @test prep.rdual1 === slots[4]
        # stable host allocation: two later calls allocate the same
        a2 = jacobian_allocations(J, prep, prob, x, p)
        a3 = jacobian_allocations(J, prep, prob, x, p)
        @test a2 == a3

        # and the slots are not merely kept, they are the buffers the call writes: zeroed here, so
        # that a call working on a fresh `similar` copy instead would leave them zero.
        fill!(prep.xdual, zero(eltype(prep.xdual)))
        fill!(prep.rdual, zero(eltype(prep.rdual)))
        jacobian!!(J, prep, prob, x, p)
        rx = similar(r)
        prob.F(rx, x, p)
        @test Array(ForwardDiff.value.(prep.xdual)) == Array(x)
        @test Array(ForwardDiff.value.(prep.rdual)) == Array(rx)
    end
end
