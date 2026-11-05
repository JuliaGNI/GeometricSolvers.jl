using GeometricSolvers
using GeometricSolvers: record, isconverged, isstalled
using Test

include("../helpers/matrix.jl")

@testset "ReturnCode is one byte" begin
    @test sizeof(ReturnCode) == 1
    @test isbits(SUCCESS)
    @test instances(ReturnCode) ==
          (SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED)
end

@testset "SolverStatus and StepInfo are isbits" for R in REAL_ELTYPES
    st = SolverStatus{R}(SUCCESS, Int32(0), one(R), zero(R), Int32(0), false)
    st9 = SolverStatus{R}(
        SUCCESS, Int32(1), one(R), zero(R), Int32(0), false, Int32(2), zero(R), one(R))
    info = StepInfo{R}(SUCCESS, one(R), zero(R), Int32(0), false, false)
    @test isbits(st)
    @test isbits(st9)
    @test isbits(info)
    @test isbitstype(SolverStatus{R})
    @test isbitstype(StepInfo{R})
    @test sizeof(SolverStatus{R}) == (R === Float32 ? 36 : 56)
    @test st.rank === Int32(-1)
    @test st.nullnorm === R(NaN)
    @test st.gradratio === R(NaN)
    @test st9.rank === Int32(2)
    @test st9.nullnorm === zero(R)
    @test st9.gradratio === one(R)
end

@testset "record folds a step into the status" begin
    st = SolverStatus{Float64}(MAXITERS, Int32(0), 4.0, 0.0, Int32(0), false)
    st = record(st, StepInfo{Float64}(MAXITERS, 2.0, 1.5, Int32(1), false, false))
    st = record(st, StepInfo{Float64}(SUCCESS, 0.5, 0.25, Int32(2), true, false))
    st = record(st, StepInfo{Float64}(SUCCESS, 0.1, 0.125, Int32(0), false, false))
    @test st.code == SUCCESS
    @test st.iterations === Int32(3)
    @test st.fnorm == 0.1
    @test st.stepnorm == 0.125
    @test st.step_failures === Int32(3)
    @test st.promoted             # once set, it stays set
    @test st.rank === Int32(-1)
    @test st.nullnorm === NaN
    @test st.gradratio === NaN
    @test isconverged(st)
    @test !isstalled(st)
    @test (@inferred record(st, StepInfo{Float64}(
        STALLED, 0.1, 0.0, Int32(0), false, false))) isa
          SolverStatus{Float64}
    st_diag = SolverStatus{Float64}(
        MAXITERS, Int32(1), 2.0, 1.0, Int32(0), false, Int32(2), 0.25, 0.75)
    st_diag = record(
        st_diag, StepInfo{Float64}(SUCCESS, 0.5, 0.25, Int32(0), false, false))
    @test st_diag.rank === Int32(2)
    @test st_diag.nullnorm === 0.25
    @test st_diag.gradratio === 0.75
end
