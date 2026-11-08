# JET.jl static analysis of this package's own code. See https://github.com/aviatesk/JET.jl.
#
# `target_modules = (GeometricSolvers,)` keeps the reports whose innermost frame is in this
# package; a report raised inside a dependency is not this file's concern. The file fails on any
# report: KernelAbstractions 0.10 defines the launch of a kernel and `synchronize` for every
# backend through KernelInterface, so the analysis finds neither missing where Metal or JLArrays
# is not loaded.

using GeometricSolvers
using JET
using Test

@testset "JET report_package" begin
    result = JET.report_package(GeometricSolvers; target_modules = (GeometricSolvers,),
        toplevel_logger = nothing)
    reports = JET.get_reports(result)
    isempty(reports) || show(result)
    @test isempty(reports)
end
