# The matrix-free R5 Newton–GMRES on Metal in `Float32`, under `allowscalar(false)` (Check 7 of
# the part's spec).
#
# The R5 state holds the mutable Krylov workspace, so it is not `isbits` and not a kernel
# argument: the solve runs on the host and every array it touches is an `MtlArray`, so the
# scalar-indexing check covers the whole path. Metal has no `Float64`, and no device `Float64` is
# run.

using GPUArraysCore: allowscalar
using LinearAlgebra: mul!
using Metal: Metal
using Test

using GeometricSolvers
using GeometricSolvers: ChunkedForwardDiff, FixedForcing, GMRES, JVPFunction, NONFINITE,
                        NonlinearProblem, Newton, SUCCESS, init, max_inner_iterations,
                        rnorm,
                        solve!

include("../helpers/matrix.jl")
include("../helpers/metal_device.jl")

const N_METAL_R5 = 8

# The Check 1 problem: `F_i = x_i^3 + 2 x_i - 3`, whose Jacobian `3 x_i^2 + 2` no array holds.
metal_r5_residual!(r, x, q) = (@. r = x^3 + 2x - 3; r)

@testset "Metal R5 Newton-GMRES" begin
    if Metal.functional() && !on_virtual_gpu()
        allowscalar(false)

        @testset "Check 1 on MtlArray, Float32" begin
            prob = NonlinearProblem(metal_r5_residual!)
            x₀ = Metal.MtlArray(fill(2.0f0, N_METAL_R5))
            fnorm₀ = rnorm(x₀ .^ 3 .+ 2 .* x₀ .- 3)
            m = Newton(; linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(1e-6f0)))
            solver = init(prob, x₀, m)
            # the default preparation is the one-partial chunked product, on the device array
            @test solver.state.jvpprep.prep isa ChunkedForwardDiff{1}
            @test solver.state.J === nothing

            x, st = solve!(solver, copy(x₀))
            @test x isa Metal.MtlArray
            @test st.code == SUCCESS
            @test st.fnorm <= max(zero(Float32), sqrt(eps(Float32)) * fnorm₀)
            @test all(isfinite, Array(x))
            @test st.f_calls > 0 && st.jvp_calls > 0
            @test st.f_calls == GeometricSolvers.f_calls(solver)
            @test st.jvp_calls == GeometricSolvers.jvp_calls(solver)
            @test max_inner_iterations(solver) <= 10
        end

        @testset "Check 1 on MtlArray with the caller's product" begin
            prob = NonlinearProblem(metal_r5_residual!)
            x₀ = Metal.MtlArray(fill(2.0f0, N_METAL_R5))
            count = Ref(0)
            jvp! = (Jv, x, v, q) -> begin
                count[] += 1
                @. Jv = (3x^2 + 2) * v
                Jv
            end
            m = Newton(;
                linsolve = GMRES(; maxiter = 10, forcing = FixedForcing(1e-6f0)),
                jacobian = JVPFunction(jvp!))
            solver = init(prob, x₀, m)
            @test solver.state.jvpprep.prep isa JVPFunction
            _, st = solve!(solver, copy(x₀))
            @test st.code == SUCCESS
            @test st.jvp_calls == count[]
            @test all(isfinite, Array(st.fnorm))
        end

        @testset "Check 4 on MtlArray, Float32" begin
            A = Metal.MtlArray(Float32[4 1 0.5 0.2
                                       0.3 3 0.8 0.1
                                       0.2 0.4 2.5 0.6
                                       0.1 0.2 0.3 2.0])
            b = Metal.MtlArray(Float32[1, -2, 1.5, -0.5])
            prob = NonlinearProblem(
                (r, x, q) -> begin
                mul!(r, A, x)
                r .-= b
                r
            end)
            solver = init(
                prob, Metal.MtlArray(zeros(Float32, 4)),
                Newton(;
                    linsolve = GMRES(; maxiter = 1, forcing = FixedForcing(1e-7f0))))
            x, st = solve!(solver, Metal.MtlArray(zeros(Float32, 4)))
            # one inner iteration cannot meet the tolerance, and the step records the cap
            @test st.step_failures >= 1
            @test max_inner_iterations(solver) <= 1
            @test all(isfinite, Array(x))
        end

        @testset "E1 on MtlArray: a non-finite residual at the start" begin
            prob = NonlinearProblem(
                (r, x, q) -> begin
                @. r = x^3 + 2x - 3
                r[1] = Float32(NaN)
                r
            end)
            _, st = solve(prob, Metal.MtlArray(fill(2.0f0, 4)), Newton(;
                linsolve = GMRES()))
            @test st.code == NONFINITE
            @test st.iterations == 0
        end
    elseif Metal.functional()
        @test_skip on_virtual_gpu()  # issue #34
    else
        @test_skip Metal.functional()
    end
end
