# Check of the midpoint rows: each scenario's stage residual, solved by GeometricSolvers'
# `Newton()` in `Float64`, gives the `q₁` of one step of GeometricIntegrators' `ImplicitMidpoint`
# to `√eps(Float64) · max(1, ‖q₁‖∞)`, and each scenario's `Float32` problem holds parameters and
# initial conditions of element type `Float32`.
#
# Usage: julia --project=benchmark benchmark/check_midpoint.jl

import GeometricSolvers as GS
using GeometricIntegrators: integrate, ImplicitMidpoint
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "problems.jl"))

# Every parameter and every initial condition of `prob` has the element type `T`.
function holds_eltype(prob, ::Type{T}) where {T}
    return all(v -> eltype(v) === T, values(prob.parameters)) &&
           all(v -> eltype(collect(v)) === T, values(prob.ics))
end

@testset "midpoint rows against ImplicitMidpoint" begin
    for (name, h) in MIDPOINT_SCENARIOS
        @testset "$(midpoint_label(name, h))" begin
            @test holds_eltype(midpoint_problem(name, h, Float32), Float32)
            prob = midpoint_problem(name, h, Float64)
            sys = midpoint_system(prob)
            X, st = GS.solve(GS.NonlinearProblem(sys.F), copy(sys.x0), GS.Newton();
                f_abstol = 8eps(), f_reltol = 0, max_iterations = 200)
            @test st.code == GS.SUCCESS
            q₁ = sys.q1(X)
            q₁GI = collect(integrate(prob, ImplicitMidpoint()).q[end])
            @test norm(q₁ - q₁GI, Inf) ≤ sqrt(eps()) * max(1, norm(q₁GI, Inf))
        end
    end
end
