# The problem and the call form that follows the iterate — check 2 of §13.L-a.
#
# `NonlinearProblem` holds the residual and, optionally, its derivative, and nothing else: no
# parameters, no iterate and no `inplace` flag. The storage of the iterate selects the call form,
# and it is `residual!!` that reads it: the value method of `src/base/bang.jl` calls `F(x, p)` for
# every iterate it is handed. This file tests the value slot only, not an in-place `F!` call.

using StaticArrays: MVector, SVector
using Test

using GeometricSolvers
using GeometricSolvers: NonlinearProblem, buffer, residual!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_PROBLEM = 4

@testset "the problem holds functions only" begin
    prob = NonlinearProblem(R1Scaled())
    @test prob.F isa R1Scaled
    @test prob.J === nothing
    @test NonlinearProblem(R1Scaled(), r1_scaled_jacobian).J === r1_scaled_jacobian
    # two fields and no flag: the call form depends on the storage, not on an `inplace` flag
    @test fieldnames(typeof(prob)) === (:F, :J)
    @test isbits(prob)
end

# A residual whose closure holds a heap array: the problem is not `isbits` then. The state holds no
# problem, so `isbits(prob)` is the whole question a kernel asks of one (§4.4).
heap_residual(::Type{T}) where {T} = (q = zeros(T, 2); (x, p) -> x .+ q[1])

@testset "the problem is isbits when its functions are" begin
    @test isbits(NonlinearProblem(R1Scaled(), r1_scaled_jacobian))
    @test !isbits(NonlinearProblem(heap_residual(Float64), nothing))
    @test !isbits(NonlinearProblem(heap_residual(ComplexF64), r1_scaled_jacobian))
end

@testset "the call form follows the storage of the iterate, $T" for T in ELTYPES
    prob = NonlinearProblem(R1Scaled(), r1_scaled_jacobian)
    x, p = r1_inputs(T, N_PROBLEM)
    v = MVector(x)

    # A value slot calls `F(x, p)`, whatever the iterate: an `SVector`, an `MVector`, which counts
    # as a value (§4.3), and a `Number`. Nothing in the call reads the iterate's type, so an
    # `Array` in a value slot calls `F(x, p)` too; these tests do not use an in-place residual.
    @test residual!!(nothing, prob, x, p) == prob.F(x, p)
    @test residual!!(nothing, prob, v, p) == prob.F(v, p)
    @test residual!!(nothing, prob, one(T), p[1]) == T(p[1])
    @test residual!!(nothing, prob, Array(x), p) == prob.F(Array(x), p)

    # and every value iterate has a `nothing` storage slot
    @test buffer(x) === nothing
    @test buffer(v) === nothing
    @test buffer(one(T)) === nothing
end
