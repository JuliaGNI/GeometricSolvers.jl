using GeometricSolvers
using GeometricSolvers: record, isconverged, isstalled
using Test

@testset "ReturnCode is one byte" begin
    @test sizeof(ReturnCode) == 1
    @test isbits(SUCCESS)
    @test instances(ReturnCode) ==
          (SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED)
end

@testset "SolverStatus and StepInfo are isbits" for R in (Float32, Float64)
    st = SolverStatus{R}(SUCCESS, Int32(0), one(R), zero(R), Int32(0), false)
    info = StepInfo{R}(SUCCESS, one(R), zero(R), Int32(0), false)
    @test isbits(st)
    @test isbits(info)
    @test isbitstype(SolverStatus{R})
    @test isbitstype(StepInfo{R})
end

@testset "record folds a step into the status" begin
    st = SolverStatus{Float64}(MAXITERS, Int32(0), 4.0, 0.0, Int32(0), false)
    st = record(st, StepInfo{Float64}(MAXITERS, 2.0, 1.5, Int32(1), false))
    st = record(st, StepInfo{Float64}(SUCCESS, 0.5, 0.25, Int32(2), true))
    st = record(st, StepInfo{Float64}(SUCCESS, 0.1, 0.125, Int32(0), false))
    @test st.code == SUCCESS
    @test st.iterations === Int32(3)
    @test st.fnorm == 0.1
    @test st.stepnorm == 0.125
    @test st.step_failures === Int32(3)
    @test st.promoted             # once set, it stays set
    @test isconverged(st)
    @test !isstalled(st)
    @test (@inferred record(st, StepInfo{Float64}(STALLED, 0.1, 0.0, Int32(0), false))) isa
          SolverStatus{Float64}
end
