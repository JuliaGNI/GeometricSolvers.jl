# The device tests of `test/gpu/runtests.jl` on a real Apple GPU: a KernelAbstractions kernel on
# an `MtlArray`, checked against a broadcast on the CPU, in `Float32` and `Float16`. `runtests.jl`
# runs this file in the `metal` group, which a default run on Apple silicon includes and
# `Pkg.test(test_args = ["metal"])` selects anywhere. `test/gpu/` stays the separate suite for a
# run by hand; this file includes it and calls its `main` itself. It then checks the scaled path of
# `rnorm` on an `MtlArray`.
#
# Where `Metal.functional()` is `true` on a real Apple GPU the file runs the device tests. Where it
# is `false` the file runs no device test and records one visible skip: on a Mac without a usable
# device, and inside a sandbox, where `Metal.devices()` is empty although the hardware is present.
# The skip does not fail the run, so a step of `.github/workflows/Metal.yml` before the tests fails
# that job where Metal is not functional. On a virtualized GPU, as on the GitHub macOS runners, the
# file records a skip for issue #34 and runs no device test.

using GeometricSolvers
using LinearAlgebra: norm
using Metal, Test

include("../helpers/metal_device.jl")

if Metal.functional() && !on_virtual_gpu()
    include("../gpu/runtests.jl")
    main("metal")

    # `rnorm`'s scaled path sums through a functor, not a closure: a closure that captures the
    # accumulation type is a non-isbits kernel argument, which a device rejects. Both `top` values
    # take that path, one by an overflowing sum of squares and one by an underflowing sum. Metal has
    # no `Float64`.
    @testset "rnorm scales on the device" begin
        for top in (floatmax(Float32) / 4, floatmin(Float32))
            v = Float32[top, top / 3, zero(Float32), top / 7]
            r = GeometricSolvers.rnorm(MtlArray(v))
            @test r isa Float32
            @test isapprox(r, norm(v); rtol = 8 * eps(Float32))
        end
    end
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
