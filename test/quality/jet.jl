# JET.jl static analysis of this package's own code. See https://github.com/aviatesk/JET.jl.
#
# `target_modules = (GeometricSolvers,)` keeps the reports whose innermost frame is in this
# package; a report raised inside a dependency is not this file's concern. The file fails where
# the reports differ from `KNOWN_FALSE_POSITIVES`: a report that no entry names, or an entry
# that names no report or more than one. JET can fail to load on a pre-release Julia, so there
# alone the file records one skip and runs no analysis; on a released Julia, a JET that fails
# to load fails the file.

using GeometricSolvers
using Test

# KernelAbstractions leaves the launch of a kernel on a `GPU` backend and `synchronize(::GPU)`
# to the backend packages, such as Metal.jl and CUDA.jl, and this analysis loads none of them.
const LAUNCH = "the backend package defines the launch of a kernel on a `GPU` backend"
const SYNC = "the backend package defines `synchronize` on a `GPU` backend"

# Each known false positive, named by the function of its innermost frame and a fragment of
# JET's message, neither of which holds a line number, with the reason it is not a defect.
const KNOWN_FALSE_POSITIVES = [
    (f = "solve!", msg = "gpu__solve_status_kernel!", reason = LAUNCH),
    (f = "solve!", msg = "gpu__solve_codes_kernel!", reason = LAUNCH),
    (f = "ka_svd!", msg = "`synchronize(::KernelAbstractions.GPU)`", reason = SYNC),
    (f = "ka_svd_backsolve!",
        msg = "`synchronize(::KernelAbstractions.GPU)`", reason = SYNC),
    (f = "ka_factorize_lu!",
        msg = "`ka_lu!(::AbstractMatrix, ::AbstractVector, ::Nothing)`",
        reason = "its one caller, `factorize!!` of a `DenseLU{<:AbstractGPUArray}`, " *
                 "holds the `Int32` status vector that `init` gives a GPU array")
]

if !isempty(VERSION.prerelease)
    @testset "JET report_package" begin
        @test_skip isempty(VERSION.prerelease)  # JET on a pre-release: aviatesk/JET.jl#796
    end
else
    using JET

    # The name of the function of a report's innermost frame, with the `#name#n` of a keyword
    # body read as `name`, so that the name does not depend on the order of compilation.
    function frame_name(report)
        name = string(last(report.vst).linfo.def.name)
        m = match(r"^#(.+)#\d+$", name)
        return m === nothing ? name : m.captures[1]
    end

    function names_report(entry, report)
        return entry.f == frame_name(report) &&
               occursin(entry.msg, sprint(JET.print_report_message, report))
    end

    @testset "JET report_package" begin
        result = JET.report_package(GeometricSolvers; target_modules = (GeometricSolvers,),
            toplevel_logger = nothing)
        reports = JET.get_reports(result)
        unlisted = filter(r -> !any(e -> names_report(e, r), KNOWN_FALSE_POSITIVES), reports)
        counts = [count(r -> names_report(e, r), reports) for e in KNOWN_FALSE_POSITIVES]
        miscounted = [e.f => n for (e, n) in zip(KNOWN_FALSE_POSITIVES, counts) if n != 1]
        isempty(unlisted) || show(result)
        @test isempty(unlisted)
        @test isempty(miscounted)
    end
end
