# The array backends a test loops over: an ordinary `Array`; `JLArray`, a CPU array that
# refuses scalar indexing and so catches a GPU-only fault in ordinary CI; and the
# KernelAbstractions `CPU()` backend, for a kernel itself rather than for the array it runs on.

using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using Test

const ARRAY_BACKENDS = (Array, JLArray)
const KA_BACKEND = KernelAbstractions.CPU()

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
