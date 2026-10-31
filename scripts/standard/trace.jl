# Trace one cell of the standard test set with LAPACK's LU and QR, then with the pure-Julia LU
# and QR of `generic_lapack.jl`, and print, per primal evaluation, the largest distance of the
# two runs' points in ulps and the two norms of the residual, then the last line-search trials of
# `Backtracking` of each run with the quantities that decide them. From the root of the
# repository:
#
#     julia --project=test scripts/standard/trace.jl 21 u0 "Newton Backtracking LU" Float32
#
# The logging `search` below is the body of `search(::Backtracking, …)` of
# `src/globalization/backtracking.jl` with one record per trial added; it traces that method
# while the two bodies agree.

include(joinpath(@__DIR__, "..", "..", "test", "helpers", "standard_cells.jl"))

const LOG = Ref(false)
const TRIALS = Any[]

@eval GeometricSolvers function search(ls::Backtracking{R}, lf, step, φ₀::R, d₀::R, τ::R,
        α::R, ceiling::R, n::Int32) where {R}
    αmin = smallest_step(d₀, τ)
    αₐ, φₐ, Fₐ = α, R(NaN), anchor_residual(lf)
    αp, φp = R(NaN), R(NaN)
    frozen = 0
    code = LINESEARCH_FAILED
    done = false
    for _ in 1:(ls.maxiter)
        done && continue
        αₐ = α
        φₐ, Fₐ = merit_and_residual(lf, α)
        n += Int32(1)
        frozen = φₐ == φ₀ && α ≤ sqrt(eps(R)) ? frozen + 1 : 0
        decision = "reject"
        if accepts(step, ls, φₐ, φ₀, d₀, α, τ)
            code = classify(φₐ, φ₀, τ)
            done = true
            decision = "accept -> $code"
        elseif frozen ≥ 2
            code = STALLED
            done = true
            decision = "frozen -> STALLED"
        elseif α ≤ αmin
            code = floor_code(φₐ, φ₀, τ)
            done = true
            decision = "floor -> $code"
        else
            αₙ = isfinite(φₐ) ? backtrack_step(φ₀, d₀, α, φₐ, αp, φp, ls.p) :
                 min(R(0.1) * α, sqrt(α) * sqrt(αmin))
            αp, φp, α = α, φₐ, max(αₙ, αmin)
        end
        Main.LOG[] && push!(Main.TRIALS,
            (α = αₐ, Δφ = φₐ - φ₀, τ = τ, ulp = eps(φ₀), αmin = αmin, decision = decision))
    end
    LineSearchResult{R, typeof(Fₐ)}(αₐ, φₐ, code, n, Fₐ)
end

function traced(sp, x₀, method)
    points = Any[]
    F = function (y, x, p)
        sp.prob.F(y, x, p)
        eltype(x) <: AbstractFloat && push!(points, (copy(x), norm(y)))
        return nothing
    end
    empty!(TRIALS)
    LOG[] = true
    _, status = solve(NonlinearProblem(F), x₀, method; max_iterations = STANDARD_MAX_ITERATIONS)
    LOG[] = false
    return (; points, trials = copy(TRIALS), status.code, status.iterations)
end

ulps(a, b) = a == b ? 0.0 : abs(a - b) / eps(max(abs(a), abs(b), floatmin(typeof(a))))

function trace(number, start, column, eltype)
    sp = STANDARD_PROBLEMS[number]
    x₀ = Dict(standard_starts(sp, eltype))[start]
    method = Dict(STANDARD_COLUMNS)[column]
    traced(sp, x₀, method)
    a = traced(sp, x₀, method)
    include(joinpath(@__DIR__, "generic_lapack.jl"))
    b = Base.invokelatest(traced, sp, x₀, method)
    println((number, start, column, eltype), ": LAPACK ", a.code, " ", a.iterations,
        ", pure Julia ", b.code, " ", b.iterations)
    println("evaluation  ulps of x  ‖F‖ LAPACK  ‖F‖ pure Julia")
    for (i, ((xa, Fa), (xb, Fb))) in enumerate(zip(a.points, b.points))
        println(lpad(i, 10), "  ", lpad(maximum(ulps.(xa, xb)), 9), "  ", Fa, "  ", Fb)
    end
    for (name, run) in (("LAPACK", a), ("pure Julia", b))
        println("last line-search trials, ", name, ":")
        for t in run.trials[max(1, end - 3):end]
            println("  α = ", t.α, ", φ(α) − φ₀ = ", t.Δφ,
                " (", round(t.Δφ / t.ulp), " ulps of φ₀)",
                ", τ = ", t.τ, ", αmin = ", t.αmin, ": ", t.decision)
        end
    end
end

trace(parse(Int, ARGS[1]), ARGS[2], ARGS[3], Dict("Float32" => Float32, "Float64" =>
    Float64)[ARGS[4]])
