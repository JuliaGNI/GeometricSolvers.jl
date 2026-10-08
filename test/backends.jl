# The array backends a test loops over are named in `test/helpers/matrix.jl`, with the element
# types: an ordinary `Array`; `JLArray`, a CPU array that refuses scalar indexing and so catches a
# GPU-only fault in ordinary CI; and the KernelAbstractions `CPU()` backend, for a kernel itself
# rather than for the array it runs on. This file checks that they are what they claim to be.

using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using Test

include("helpers/matrix.jl")

@testset "the backends are constructible" begin
    for AT in ARRAY_BACKENDS
        a = AT(zeros(Float32, 4))
        @test a isa AbstractVector{Float32}
        @test length(a) == 4
    end
    # `JLArray` carries its own KernelAbstractions backend (`JLBackend`), not `CPU()` — it is a
    # stand-in device array, not a `CPU()`-backed one, and its own backend is what its own kernels
    # dispatch on.
    @test KernelAbstractions.get_backend(JLArray(zeros(Float32, 1))) isa
          KernelAbstractions.Backend
    @test KA_BACKEND isa KernelAbstractions.CPU
end

# The sets themselves, so that a type silently dropped from one of them fails a test rather than
# only shrinking the number of cases every other test runs. The names are not written as a tuple
# here: `test/quality/matrix.jl` fails on a literal tuple of element types under `test/`.
@testset "the element-type sets are the matrix the package claims" begin
    @test length(REAL_ELTYPES) == 2
    @test Float32 in REAL_ELTYPES
    @test Float64 in REAL_ELTYPES
    @test all(T -> T <: AbstractFloat && isconcretetype(T), REAL_ELTYPES)
    # the complex types are the complexification of the real ones, in the same order
    @test ELTYPES === (REAL_ELTYPES..., map(complex, REAL_ELTYPES)...)
    @test length(ELTYPES) == 4
    @test ComplexF32 in ELTYPES
    @test ComplexF64 in ELTYPES
    @test length(ARRAY_BACKENDS) == 2
    @test Array in ARRAY_BACKENDS
    @test JLArray in ARRAY_BACKENDS
end
