# The rows of the correctness benchmark: the 67 starts of the standard test set, read from
# `test/helpers/standard_problems.jl`, and the first-step stage residuals of twelve
# implicit-midpoint scenarios of SolverBenchmark, built from GeometricProblems; and the sparse
# rows of `test/helpers/sparse_problems.jl`.
#
# A row is one nonlinear system `F(y, x, p) = 0` and one start `x0` in the element type `T`.

using GeometricProblems: HarmonicOscillator, Pendulum, LotkaVolterra2d, LotkaVolterra4d,
                         DoublePendulum, TodaLattice

include(joinpath(@__DIR__, "..", "test", "helpers", "standard_problems.jl"))

"""
    Row(set, problem, start, F, x0)

One system of the benchmark: the `set` it belongs to (`standard` or `midpoint`), its `problem`
and `start` labels, its residual `F(y, x, p)` and its start `x0`. The field `F` is typed
`Function`, so that every row shares one type.
"""
struct Row{T}
    set::String
    problem::String
    start::String
    F::Function
    x0::Vector{T}
end

"""
    standard_rows(T)

The 67 rows of the standard test set in the element type `T`: every problem of
`STANDARD_PROBLEMS` with every start of `standard_starts(sp, T)`.
"""
function standard_rows(::Type{T}) where {T}
    return [Row{T}("standard", string(sp.number), label, sp.prob.F, x0)
            for sp in STANDARD_PROBLEMS for (label, x0) in standard_starts(sp, T)]
end

"""
    MIDPOINT_SCENARIOS

The twelve implicit-midpoint scenarios of SolverBenchmark's `scripts/run_all.jl`, each a problem
name and a time step.
"""
const MIDPOINT_SCENARIOS = [("HarmonicOscillator", 0.1), ("HarmonicOscillator", 1.0),
    ("Pendulum", 0.1), ("Pendulum", 1.0), ("LotkaVolterra2d", 0.01), (
        "LotkaVolterra2d", 0.1),
    ("LotkaVolterra4d", 0.01), ("LotkaVolterra4d", 0.1), ("DoublePendulum", 0.01),
    ("DoublePendulum", 0.1), ("TodaLattice", 0.1), ("TodaLattice", 1.0)]

"""
    midpoint_problem(name, h, T)

The GeometricProblems problem of the scenario `name` over one step `(0, h)` in the element type
`T`, with `parameters = default_parameters(T)` and the initial conditions of SolverBenchmark.
"""
function midpoint_problem(name, h, ::Type{T}) where {T}
    ts = (zero(T), T(h))
    if name == "HarmonicOscillator"
        return HarmonicOscillator.odeproblem(T[0.5, 0.0]; timespan = ts, timestep = T(h),
            parameters = HarmonicOscillator.default_parameters(T))
    elseif name == "Pendulum"
        return Pendulum.odeproblem(T[acos(0.4), 0.0]; timespan = ts, timestep = T(h),
            parameters = Pendulum.default_parameters(T))
    elseif name == "LotkaVolterra2d"
        return LotkaVolterra2d.iodeproblem(T[2.0, 1.0]; timespan = ts, timestep = T(h),
            parameters = LotkaVolterra2d.default_parameters(T))
    elseif name == "LotkaVolterra4d"
        return LotkaVolterra4d.iodeproblem(T[2.0, 1.0, 1.0, 1.0]; timespan = ts,
            timestep = T(h), parameters = LotkaVolterra4d.default_parameters(T))
    elseif name == "DoublePendulum"
        return DoublePendulum.hodeproblem(T.(DoublePendulum.θ₀), T.(DoublePendulum.p₀);
            timespan = ts, timestep = T(h), parameters = DoublePendulum.default_parameters(T))
    elseif name == "TodaLattice"
        q₀ = T.(TodaLattice.compute_initial_q(0.3, 16))
        return TodaLattice.hodeproblem(16, q₀, zero(q₀); timespan = ts, timestep = T(h),
            parameters = TodaLattice.default_parameters(T))
    end
    throw(ArgumentError("no midpoint scenario $name"))
end

# The kind of a problem, from the functions it exposes.
function problem_kind(prob)
    haskey(prob.functions, :ϑ) ? :iode :
    haskey(prob.functions, :f) ? :hode : :ode
end

"""
    midpoint_system(prob)

The stage residual of the first step of `ImplicitMidpoint` on `prob`, with `t̃ = t₀ + h/2`, as a
named tuple of the residual `F(y, x, p)`, the start `x0` and the map `q1(x)` from a solution to
the end point `q₁ = q₀ + h V`:

- ODE, unknown `V`: `F(V) = v(t̃, q₀ + (h/2) V) − V`, from `x0 = v(t₀, q₀)`;
- IODE, unknown `V`, `Q = q₀ + (h/2) V`: `F(V) = ϑ(t̃, Q, V) − p₀ − (h/2) f(t̃, Q, V)`, from the
  problem's initial guess `x0 = v̄(t₀, q₀, p₀)`;
- HODE, unknown `(V, W)`, `Q = q₀ + (h/2) V`, `P = p₀ + (h/2) W`:
  `F(V, W) = (v(t̃, Q, P) − V, f(t̃, Q, P) − W)`, from `x0 = (v(t₀, q₀, p₀), f(t₀, q₀, p₀))`.

The residual computes in the element type of its argument, so a dual iterate stays dual.
"""
function midpoint_system(prob)
    fs, par, h = prob.functions, prob.parameters, prob.timestep
    t₀ = prob.timespan[1]
    t̃ = t₀ + h / 2
    q₀ = collect(prob.ics.q)
    d = length(q₀)
    k = problem_kind(prob)
    if k == :ode
        F = function (y, V, p)
            fs.v(y, t̃, q₀ .+ (h / 2) .* V, par)
            y .-= V
            return nothing
        end
        x0 = similar(q₀)
        fs.v(x0, t₀, q₀, par)
        return (; F, x0, q1 = V -> q₀ .+ h .* V)
    elseif k == :iode
        p₀ = collect(prob.ics.p)
        F = function (y, V, p)
            Q = q₀ .+ (h / 2) .* V
            f = similar(y)
            fs.ϑ(y, t̃, Q, V, par)
            fs.f(f, t̃, Q, V, par)
            y .-= p₀ .+ (h / 2) .* f
            return nothing
        end
        x0 = similar(q₀)
        prob.initialguess.v(x0, t₀, q₀, p₀, par)
        return (; F, x0, q1 = V -> q₀ .+ h .* V)
    else
        p₀ = collect(prob.ics.p)
        F = function (y, X, p)
            V, W = X[1:d], X[(d + 1):(2d)]
            Q, P = q₀ .+ (h / 2) .* V, p₀ .+ (h / 2) .* W
            v, f = similar(y, d), similar(y, d)
            fs.v(v, t̃, Q, P, par)
            fs.f(f, t̃, Q, P, par)
            y[1:d] .= v .- V
            y[(d + 1):(2d)] .= f .- W
            return nothing
        end
        v₀, f₀ = similar(q₀), similar(q₀)
        fs.v(v₀, t₀, q₀, p₀, par)
        fs.f(f₀, t₀, q₀, p₀, par)
        return (; F, x0 = [v₀; f₀], q1 = X -> q₀ .+ h .* X[1:d])
    end
end

# The problem label of a midpoint scenario.
midpoint_label(name, h) = "$name h=$h"

"""
    midpoint_rows(T)

The twelve midpoint rows in the element type `T`, each started from the vector field at the
initial state.
"""
function midpoint_rows(::Type{T}) where {T}
    return map(MIDPOINT_SCENARIOS) do (name, h)
        sys = midpoint_system(midpoint_problem(name, h, T))
        return Row{T}("midpoint", midpoint_label(name, h), "V0", sys.F, sys.x0)
    end
end

"""
    benchmark_rows(T)

Every row of the benchmark in the element type `T`: the standard rows, then the midpoint rows.
"""
benchmark_rows(::Type{T}) where {T} = [standard_rows(T); midpoint_rows(T)]

include(joinpath(@__DIR__, "..", "test", "helpers", "sparse_problems.jl"))

"""
    SPARSE_SIZES

The block sizes `N` of the sparse rows: systems of dimension `2N`.
"""
const SPARSE_SIZES = (125, 768)

"""
    sparse_row(N, T)

The sparse row of `sparse_bracket_problem(N, T)` with its seed 1, as a named tuple: the `row`,
whose residual `F(y, x, p)` and start `z0` are the problem's, the analytic sparse Jacobian
`J(J, x, p)`, the `prototype` of its sparsity, the parameter `p` and the planted solution
`z_star`.
"""
function sparse_row(N::Integer, ::Type{T}) where {T}
    prob, _, z0, p, z_star = sparse_bracket_problem(N, T)
    row = Row{T}("sparse", "bracket N=$N", "z0", prob.F, z0)
    return (; row, J = prob.J, prototype = prob.jacobian_prototype, p, z_star)
end
