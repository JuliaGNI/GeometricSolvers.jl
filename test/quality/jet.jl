# JET.jl static analysis of this package's own code. See https://github.com/aviatesk/JET.jl.
#
# `target_modules = (GeometricSolvers,)` keeps the reports whose innermost frame is in this
# package; a report raised inside a dependency is not this file's concern. The file fails on any
# report. It used to hold a table of known false positives about the launch of a kernel and
# `synchronize` on a device backend: KernelAbstractions 0.9 left both to the backend package, so
# the analysis reported them where Metal or JLArrays was not loaded. KernelAbstractions 0.10
# defines them through KernelInterface, and the table is empty now.

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
