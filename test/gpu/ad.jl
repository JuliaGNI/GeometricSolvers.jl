# The R3 AD path on a real device: checks 1–5 of §13.S, run by hand on a machine with the GPU.
#
# `test/ad/jacobian.jl` and `test/ad/jvp.jl` run the same checks on a `JLArray`, which is a
# reference implementation of the GPU array interface and not a GPU: it has the same scalar-indexing
# rules and the same broadcast machinery, but none of a device's arithmetic. This file is what says
# that the chunked forward mode really runs on an `MtlArray` and a `CuArray` — that a
# `Dual`-element array can be allocated, seeded by a broadcast and read back on the device.
#
# Included by `test/gpu/runtests.jl`, which supplies `to_backend` and the device.

using ADTypes: AutoFiniteDiff, AutoForwardDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: AbstractGPUArray, allowscalar
using LinearAlgebra: Diagonal

using GeometricSolvers: ChunkedForwardDiff, jacobian!!, jvp!!, prepare_ad

include("../helpers/adproblems.jl")

# The size §13.S names: a chunk of 3 does not divide 7, so the last chunk is partial.
const N_AD_DEVICE = 7
const CHUNK_DEVICE = 3

# A device permutation in `Int32`: Metal has no 64-bit integer arithmetic in a kernel.
device_perm(device, n) = to_backend(device, Int32.(cyclic_perm(Array, n)))

# The tag the preparation carries (§13.S, check 3).
device_tag(::ChunkedForwardDiff{N, Tg}) where {N, Tg} = Tg

function device_inputs(device, ::Type{T}, n) where {T}
    x, p, r = ad_inputs(Array, T, n)
    return to_backend(device, x), to_backend(device, p), to_backend(device, r)
end

function run_ad_tests(name::AbstractString, device, eltypes)
    # Half precision is not an AD element type here: the comparison against `ForwardDiff` on an
    # `Array` is exact, and `Float16` arithmetic on a device differs from the host's in the last
    # bit. Metal is therefore `Float32` alone, CUDA and ROCm `Float32` and `Float64`.
    ad_eltypes = filter(T -> T === Float32 || T === Float64, collect(eltypes))

    # check 2: a scalar fallback is an error, not a slow success
    allowscalar(false)

    @testset "R3 AD on $name" begin
        @testset "$T" for T in ad_eltypes
            host = Coupled(cyclic_perm(Array, N_AD_DEVICE))
            prob = StubProblem(Coupled(device_perm(device, N_AD_DEVICE)))
            x, p, r = device_inputs(device, T, N_AD_DEVICE)
            @test x isa AbstractGPUArray

            # check 2: the iterate's storage selects the chunked mode, and no other back end has
            # a device path
            prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK_DEVICE), prob, r, x, p)
            @test prep isa ChunkedForwardDiff{CHUNK_DEVICE}
            @test_throws ArgumentError prepare_ad(AutoFiniteDiff(), prob, r, x, p)

            # check 1: the Jacobian is what ForwardDiff gives on an Array, with a partial last
            # chunk
            J = similar(x, N_AD_DEVICE, N_AD_DEVICE)
            jacobian!!(J, prep, prob, x, p)
            @test Array(J) == forwarddiff_jacobian(host, x, p)
            @test Array(J) != transpose(Array(J))

            # check 5: the dual buffers are the state slots, at every call
            slots = (prep.xdual, prep.rdual, prep.xdual1, prep.rdual1)
            jacobian!!(J, prep, prob, x, p)
            @test (prep.xdual, prep.rdual, prep.xdual1, prep.rdual1) === slots

            # and the JVP is the same Jacobian, column by column
            Jv = similar(x)
            v = to_backend(device, T[i == 2 ? one(T) : zero(T) for i in 1:N_AD_DEVICE])
            jvp!!(Jv, prep, prob, x, v, p)
            @test Array(Jv) == forwarddiff_jacobian(host, x, p)[:, 2]
        end

        # check 3: a residual that differentiates inside its own body
        @testset "nested differentiation, $T" for T in ad_eltypes
            prob = StubProblem(Nested())
            x, p, r = device_inputs(device, T, N_AD_DEVICE)
            prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
            J = similar(x, N_AD_DEVICE, N_AD_DEVICE)
            jacobian!!(J, prep, prob, x, p)
            @test Array(J) == forwarddiff_jacobian(Nested(), x, p)
            @test device_tag(prep) === typeof(ForwardDiff.Tag(prob.F, T))
        end

        # check 4: the parameters are a constant, and a replacement takes effect
        @testset "the parameters are a constant, $T" for T in ad_eltypes
            prob = StubProblem(Scaled())
            x, p, r = device_inputs(device, T, N_AD_DEVICE)
            prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
            J = similar(x, N_AD_DEVICE, N_AD_DEVICE)
            jacobian!!(J, prep, prob, x, p)
            @test Array(J) == Diagonal(Array(p))

            p .= 2 .* p
            jacobian!!(J, prep, prob, x, p)
            @test Array(J) == Diagonal(Array(p))

            q = to_backend(device, 3 .* Array(p))
            jacobian!!(J, prep, prob, x, q)
            @test Array(J) == Diagonal(Array(q))
        end
    end
end
