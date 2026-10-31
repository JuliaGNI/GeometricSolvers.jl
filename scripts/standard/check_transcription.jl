# The transcription check of the standard test set: compare the 23 residuals copied into
# `test/helpers/standard_problems.jl` with those of NonlinearProblemLibrary.jl 0.1.8.
#
# It runs in its own environment, which holds the library; the package and its tests do not
# depend on it. From the root of the repository:
#
#     julia --startup-file=no --project=scripts/standard -e 'using Pkg; Pkg.instantiate()'
#     julia --startup-file=no --project=scripts/standard scripts/standard/check_transcription.jl
#
# For each problem it checks that the copied title, size, start and root equal the library's,
# and that every component of the copied residual agrees with the library's in `Float64` to
# `4 eps(Float64) · max(1, ‖f(x)‖∞)`, at the starts `u0`, `10 u0` and `100 u0` and at 100 random
# points of the box `[−2, 2]ⁿ` from a fixed seed. A random point where the library's residual is
# not finite is drawn again. It prints the largest scaled difference per problem, in units of
# `eps(Float64)`, and exits 1 when any check fails.

using LinearAlgebra: norm
using NonlinearProblemLibrary: NonlinearProblemLibrary
using Random: MersenneTwister

include(joinpath(@__DIR__, "..", "..", "test", "helpers", "standard_problems.jl"))

const LIBRARY_F = NonlinearProblemLibrary.problems
const LIBRARY_DATA = NonlinearProblemLibrary.dicts
const NPOINTS = 100
const TOLERANCE = 4

# The largest difference of the two residuals at `x`, in units of `eps(Float64) · max(1, ‖f‖∞)`
# of the library's residual `f`. A component that is not finite in both residuals agrees where
# the two are `isequal`.
function scaled_difference(sp, f_library!, x)
    a = zeros(length(x))
    b = zeros(length(x))
    f_library!(a, x, nothing)
    sp.prob.F(b, x, nothing)
    finite = isfinite.(a)
    all(isequal.(a[.!finite], b[.!finite])) || return Inf
    scale = eps(Float64) * max(1, norm(a[finite], Inf))
    return maximum(abs.(a[finite] .- b[finite]); init = 0.0) / scale
end

function check_transcription()
    failures = String[]
    rng = MersenneTwister(20261031)
    length(STANDARD_PROBLEMS) == length(LIBRARY_F) == 23 ||
        push!(failures,
            "the copy has $(length(STANDARD_PROBLEMS)) problems, " *
            "the library $(length(LIBRARY_F))")
    for (sp, f_library!, data) in zip(STANDARD_PROBLEMS, LIBRARY_F, LIBRARY_DATA)
        k = sp.number
        sp.title == data["title"] || push!(failures, "problem $k: the title differs")
        sp.n == data["n"] || push!(failures, "problem $k: the size differs")
        sp.u0(Float64) == data["start"] || push!(failures, "problem $k: the start differs")
        if isempty(data["sol"])
            sp.root === nothing || push!(failures, "problem $k: the library states no root")
        else
            sp.root !== nothing && sp.root(Float64) == data["sol"] ||
                push!(failures, "problem $k: the root differs")
        end

        worst = 0.0
        for (_, x) in standard_starts(sp, Float64)
            worst = max(worst, scaled_difference(sp, f_library!, x))
        end
        drawn = 0
        while drawn < NPOINTS
            x = 4 .* rand(rng, sp.n) .- 2
            y = zeros(sp.n)
            f_library!(y, x, nothing)
            all(isfinite, y) || continue
            worst = max(worst, scaled_difference(sp, f_library!, x))
            drawn += 1
        end
        verdict = worst <= TOLERANCE ? "agrees" : "DIFFERS"
        println(lpad(k, 2), "  ", rpad(sp.title, 46), " n = ", lpad(sp.n, 2),
            "  starts ", length(standard_starts(sp, Float64)), "  points ", drawn,
            "  largest difference ", round(worst; digits = 2), " eps  ", verdict)
        worst <= TOLERANCE ||
            push!(failures, "problem $k: the residual differs by $worst eps")
    end
    isempty(failures) || (foreach(f -> println("FAIL: ", f), failures); return false)
    println("transcription holds: 23 problems agree with NonlinearProblemLibrary ",
        pkgversion(NonlinearProblemLibrary))
    return true
end

check_transcription() || exit(1)
