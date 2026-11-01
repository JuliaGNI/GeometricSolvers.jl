# JET.jl static analysis of this package's own code. See https://github.com/aviatesk/JET.jl.
#
# `target_modules = (GeometricSolvers,)` keeps the reports whose innermost frame is in this
# package; a report raised inside a dependency is not this file's concern. The file fails where
# the reports differ from `KNOWN_FALSE_POSITIVES`: a report that no entry names, or an entry
# that names more reports, or fewer, than its reason gives. A JET that fails to load fails
# the file.

using GeometricSolvers
using JET
using Test

# KernelAbstractions leaves the launch of a kernel on a `GPU` backend and `synchronize(::GPU)`
# to the backend packages, such as Metal.jl and CUDA.jl, and this analysis loads none of them.
# A backend package that an earlier file of the same process loads defines a missing method,
# and JET then reports nothing for it: JLArrays defines the launch, Metal the launch and
# `synchronize`. So an entry with one of these two reasons names 1 report where the process has
# not loaded the package that defines its method, and 0 where it has; every other entry names
# exactly 1. `expected_reports` reads the loaded packages, so each count is exact.
const LAUNCH = "the backend package defines the launch of a kernel on a `GPU` backend"
const SYNC = "the backend package defines `synchronize` on a `GPU` backend"

# Each known false positive, named by the function of its innermost frame and a fragment of
# JET's message, neither of which holds a line number, with the reason it is not a defect.
const KNOWN_FALSE_POSITIVES = [
    (f = "_launch!", msg = "gpu__solve_kernel!", reason = LAUNCH),
    (f = "_launch!", msg = "`synchronize(::KernelAbstractions.GPU)`", reason = SYNC),
    (f = "ka_svd!", msg = "`synchronize(::KernelAbstractions.GPU)`", reason = SYNC),
    (f = "ka_svd_backsolve!",
        msg = "`synchronize(::KernelAbstractions.GPU)`", reason = SYNC),
    (f = "ka_factorize_lu!",
        msg = "`ka_lu!(::AbstractMatrix, ::AbstractVector, ::Nothing)`",
        reason = "its one caller, `factorize!!` of a `DenseLU{<:AbstractGPUArray}`, " *
                 "holds the `Int32` status vector that `dense_factor` gives a GPU array")
]

# The name of the function of a report's innermost frame, with the `#name#n` of a keyword
# body read as `name`, so that the name does not depend on the number that lowering gives
# the keyword body.
function frame_name(report)
    name = string(last(report.vst).linfo.def.name)
    m = match(r"^#(.+)#\d+$", name)
    return m === nothing ? name : m.captures[1]
end

# A toplevel report has no frames, and no entry names it.
function names_report(entry, report)
    report isa JET.InferenceErrorReport || return false
    return entry.f == frame_name(report) &&
           occursin(entry.msg, sprint(JET.print_report_message, report))
end

loaded(name) = any(id -> id.name == name, keys(Base.loaded_modules))

# The number of reports that an entry names: 0 where the process has loaded a package that
# defines the method its reason says is missing, 1 otherwise.
function expected_reports(entry)
    entry.reason == LAUNCH && return Int(!(loaded("JLArrays") || loaded("Metal")))
    entry.reason == SYNC && return Int(!loaded("Metal"))
    return 1
end

@testset "JET report_package" begin
    result = JET.report_package(GeometricSolvers; target_modules = (GeometricSolvers,),
        toplevel_logger = nothing)
    reports = JET.get_reports(result)
    unlisted = filter(r -> !any(e -> names_report(e, r), KNOWN_FALSE_POSITIVES), reports)
    counts = [count(r -> names_report(e, r), reports) for e in KNOWN_FALSE_POSITIVES]
    miscounted = [e.f => n
                  for (e, n) in zip(KNOWN_FALSE_POSITIVES, counts)
                  if n != expected_reports(e)]
    isempty(unlisted) || show(result)
    @test isempty(unlisted)
    @test isempty(miscounted)
end
