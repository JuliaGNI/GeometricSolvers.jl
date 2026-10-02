# The device tests of `test/gpu/runtests.jl` on a real Apple GPU: a KernelAbstractions kernel on
# an `MtlArray`, checked against a broadcast on the CPU, in `Float32` and `Float16`. `runtests.jl`
# runs this file in the `metal` group, which a default run on Apple silicon includes and
# `Pkg.test(test_args = ["metal"])` selects anywhere. `test/gpu/` stays the separate suite for a
# run by hand; this file includes it and calls its `main` itself.
#
# Where `Metal.functional()` is `true` the file runs the device tests. Where it is `false` the file
# runs no test and records one visible skip: on a Mac without a usable device, and inside a
# sandbox, where `Metal.devices()` is empty although the hardware is present. The skip does not
# fail the run, so `.github/workflows/Metal.yml` keeps the guarantee instead: a step before the
# tests fails that job where Metal is not functional, so it cannot pass without having run them.

using Metal, Test

if Metal.functional()
    include("../gpu/runtests.jl")
    main("metal")
else
    @test_skip Metal.functional()
end
