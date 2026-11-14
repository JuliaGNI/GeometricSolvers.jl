# The four packages of the correctness benchmark: the columns of the standard test set, the
# pairing table, one adapter per package and one for SIAMFANLEquations' `nsol`, the counting
# wrapper, the shared Jacobian, the success rule, the timed solve, the verdict of a row, the
# explanations of the disagreements, and the writers of the tables.

import GeometricSolvers as GS
import SimpleSolvers as SS
import SimpleNonlinearSolve as SNS
import SciMLBase
import NLsolve, LineSearches, ForwardDiff
import SIAMFANLEquations
import ADTypes, SparseMatrixColorings, Sparspak
using LinearAlgebra: BLAS, norm, qr, svd, ColumnNorm

# The columns are read from the standard test set's own file, so that the benchmark runs the
# methods that the test set runs. That file also includes test helpers that need test-only
# packages, so only its two constants are evaluated, in a module of their own.
module StandardColumns
using GeometricSolvers
end

function standard_constant(name::Symbol)
    path = joinpath(@__DIR__, "..", "test", "helpers", "standard_cells.jl")
    for ex in Meta.parseall(read(path, String); filename = path).args
        ex = Meta.isexpr(ex, :macrocall) ? ex.args[end] : ex   # a docstring wraps the `const`
        if Meta.isexpr(ex, :const) && ex.args[1].args[1] === name
            return Core.eval(StandardColumns, ex.args[1].args[2])
        end
    end
    error("$path defines no constant $name")
end

"""
    STANDARD_COLUMNS

The thirteen columns of `test/helpers/standard_cells.jl`, each a name and a method of
GeometricSolvers, in the order of that file.
"""
const STANDARD_COLUMNS = standard_constant(:STANDARD_COLUMNS)
const COLUMNS = collect(first.(STANDARD_COLUMNS))
const MAX_ITERATIONS = standard_constant(:STANDARD_MAX_ITERATIONS)

"""
    PACKAGES

The four packages, in the order of a row's cells and of the verdict.
"""
const PACKAGES = ("GeometricSolvers", "SimpleSolvers", "SimpleNonlinearSolve", "NLsolve")

"""
    tolerance(T)

The tolerance of the correctness sweep, `sqrt(eps(T))`: the absolute tolerance on `‖F‖∞` that
every package is given and that the harness checks, and the factor of the agreement bound. It is
the default `tol` of `run_cell`; the stage-tolerance block passes another.
"""
tolerance(::Type{T}) where {T} = sqrt(eps(T))

"""
    Counted(F)

A residual `F(y, x, p)` that counts its calls, `primal` where `eltype(x)` is a floating-point
type and `dual` otherwise, and records `‖y‖∞` after each primal call in `trace`. Every package
calls the residual through it. The field `F` is typed `Function`, so that every row shares one
type and no package compiles again per problem. It subtypes `Function`, because SimpleSolvers'
`NonlinearProblem` takes only a `Function` or a `Type`.
"""
mutable struct Counted <: Function
    F::Function
    primal::Int
    dual::Int
    trace::Vector{Float64}
end

Counted(F) = Counted(F, 0, 0, Float64[])

function (c::Counted)(y, x, p)
    c.F(y, x, p)
    if eltype(x) <: AbstractFloat
        c.primal += 1
        push!(c.trace, Float64(norm(y, Inf)))
    else
        c.dual += 1
    end
    return nothing
end

"""
    SharedJacobian(F)

The one Jacobian `J!(J, x, p)` of the benchmark: ForwardDiff's `jacobian!` of the residual `F`,
given to every package that accepts a Jacobian.
"""
struct SharedJacobian{C} <: Function
    F::C
end

function (j::SharedJacobian)(J, x, p)
    ForwardDiff.jacobian!(J, (y, z) -> j.F(y, z, p), similar(x), x)
    return nothing
end

# NLsolve calls its residual and its Jacobian with two arguments.
struct NLResidual{C}
    F::C
end
(f::NLResidual)(y, x) = f.F(y, x, nothing)

struct NLJacobian{J}
    J!::J
end
(j::NLJacobian)(J, x) = j.J!(J, x, nothing)

# SIAMFANLEquations' `nsol` calls its residual as `F!(FS, x)` and its Jacobian as
# `J!(FP, FS, x)`, and takes the arrays that they return.
struct NsolResidual{C}
    F::C
end
(f::NsolResidual)(FS, x) = (f.F(FS, x, nothing); FS)

struct NsolJacobian{J}
    J!::J
end
(j::NsolJacobian)(FP, FS, x) = (j.J!(FP, x, nothing); FP)

"""
    Nsol(TF, sham, jfact = nothing)

The method of SIAMFANLEquations' `nsol` that the `nsol` adapter runs: a Jacobian store `FPS` of
element type `TF`, a new Jacobian every `sham` iterations, and the factorisation `jfact`, or
`nsol`'s default for `nothing`.
"""
struct Nsol{TF, F}
    sham::Int
    jfact::F
end
function Nsol(::Type{TF}, sham::Integer, jfact = nothing) where {TF}
    Nsol{TF, typeof(jfact)}(sham, jfact)
end

# A residual `F(y, x, p)` and an analytic Jacobian `J!(J, x, p)` of a problem with the parameter
# `p`, which NLsolve calls as `f!(y, x)` and `j!(J, x)`.
struct ResidualAt{C, P}
    F::C
    p::P
end
(f::ResidualAt)(y, x) = f.F(y, x, f.p)

struct AnalyticJacobian{F, P}
    J!::F
    p::P
end
(j::AnalyticJacobian)(J, x) = j.J!(J, x, j.p)

# The linear solvers that NLsolve's `:newton` takes: `linsolve(p, A, b)` writes the solution of
# `A p = b` into `p`.
struct BackslashSolve end
(::BackslashSolve)(p, A, b) = copyto!(p, A \ b)

struct PivotedQRSolve end
(::PivotedQRSolve)(p, A, b) = copyto!(p, qr(A, ColumnNorm()) \ b)

# A non-finite matrix never reaches LAPACK's iterative SVD, which need not return on such input
# (OpenBLAS 0.3.30 on x86); the solve throws an `ArgumentError` in its place.
struct SVDSolve end
function (::SVDSolve)(p, A, b)
    all(isfinite, A) || throw(ArgumentError("the matrix has a non-finite entry"))
    return copyto!(p, svd(A) \ b)
end

# The label part of an object that an adapter passes to its package: its type name, a method
# symbol as it is, and the parts of a GeometricSolvers method. A GeometricSolvers `Newton` shows
# `LUFactorization(Float32)` as `LUFactorization{Float32}`, a `Refinement` as `+Refinement` and
# a Jacobian selector other than the default `nothing` by its type name; an `Anderson` shows its
# depth and its member.
part(obj) = string(nameof(typeof(obj)))
part(s::Symbol) = string(s)
part(m::SS.Newton) = m.refactorize == 1 ? "Newton" : "Newton($(m.refactorize))"
part(::GS.LUFactorization{Float32}) = "LUFactorization{Float32}"
function part(m::GS.Newton)
    name = m.reuse === nothing ? "Newton" : "Newton($(m.reuse))"
    parts = [name, part(m.globalization), part(m.linsolve)]
    m.refinement isa GS.Refinement && push!(parts, "Refinement")
    m.jacobian === nothing || push!(parts, part(m.jacobian))
    return join(parts, "+")
end
part(a::GS.Anderson{m}) where {m} = "Anderson{$m}($(part(a.member)))"
function part(m::Nsol{TF}) where {TF}
    store = m.jfact === nothing ? "FPS{$TF}" : string(nameof(m.jfact))
    return "nsol(sham=$(m.sham))+$store"
end
part(m::GS.Picard) = "Picard+" * part(m.globalization)
function part(m::GS.LeftPreconditioned)
    join(("LeftPreconditioned", part(m.outer), part(m.inner)), "+")
end

"""
    method_label(objs)

The label of the objects `objs` that an adapter passes to its package, a named tuple in the order
of the call: their parts joined by `+`.
"""
method_label(objs::NamedTuple) = join(map(part, values(objs)), "+")

"""
    Paired(label, difference, build)

A paired method of a package for a column: `build(T)` returns the named tuple of the objects
that the adapter passes to the package, `label` is the label they must give, and `difference`
says where the method differs from the column.
"""
struct Paired
    label::String
    difference::String
    build::Function
end

"""
    NoMethod(reason)

A column for which a package has no method, and the reason.
"""
struct NoMethod
    reason::String
end

function gs_entry(column)
    method = Dict(STANDARD_COLUMNS)[column]
    Paired(method_label((; method)), "the column itself", T -> (; method))
end

"""
    PAIRING

The pairing table: for each column, the entry of each package of `PACKAGES`.
"""
const PAIRING = Dict{String, NTuple{4, Union{Paired, NoMethod}}}(
    "Newton Static LU" => (gs_entry("Newton Static LU"),
        Paired("Newton+Static+LapackLU", "the same full Newton step",
            T -> (; method = SS.Newton(), linesearch = SS.Static(T),
                linear_solver_method = SS.LapackLU())),
        Paired("SimpleNewtonRaphson", "a full Newton step; factorises with `\\`",
            T -> (; alg = SNS.SimpleNewtonRaphson())),
        Paired(
            "newton+Static+BackslashSolve", "a full Newton step; factorises with `\\`",
            T -> (; method = :newton, linesearch = LineSearches.Static(),
                linsolve = BackslashSolve()))),
    "Newton Backtracking LU" => (gs_entry("Newton Backtracking LU"),
        Paired("Newton+Backtracking+LapackLU",
            "SimpleSolvers' own backtracking line search",
            T -> (; method = SS.Newton(), linesearch = SS.Backtracking(T),
                linear_solver_method = SS.LapackLU())),
        NoMethod("SimpleNewtonRaphson takes no line search"),
        Paired("newton+BackTracking+BackslashSolve",
            "the backtracking line search of LineSearches; factorises with `\\`",
            T -> (; method = :newton, linesearch = LineSearches.BackTracking(),
                linsolve = BackslashSolve()))),
    "Newton Bisection LU" => (gs_entry("Newton Bisection LU"),
        Paired("Newton+Bisection+LapackLU", "SimpleSolvers' own bisection line search",
            T -> (; method = SS.Newton(), linesearch = SS.Bisection(T),
                linear_solver_method = SS.LapackLU())),
        NoMethod("SimpleNewtonRaphson takes no line search"),
        NoMethod("LineSearches has no bisection line search")),
    "Newton StrongWolfe LU" => (gs_entry("Newton StrongWolfe LU"),
        Paired("Newton+StrongWolfe+LapackLU",
            "SimpleSolvers' own strong-Wolfe line search",
            T -> (; method = SS.Newton(), linesearch = SS.StrongWolfe(T),
                linear_solver_method = SS.LapackLU())),
        NoMethod("SimpleNewtonRaphson takes no line search"),
        Paired("newton+StrongWolfe+BackslashSolve",
            "the strong-Wolfe line search of LineSearches; factorises with `\\`",
            T -> (; method = :newton, linesearch = LineSearches.StrongWolfe(),
                linsolve = BackslashSolve()))),
    "Newton DogLeg LU" => (gs_entry("Newton DogLeg LU"),
        Paired("DogLeg", "SimpleSolvers' own dogleg trust region and radius update",
            T -> (; method = SS.DogLeg())),
        Paired("SimpleTrustRegion", "a dogleg trust region with its own radius update",
            T -> (; alg = SNS.SimpleTrustRegion())),
        Paired("trust_region", "a dogleg trust region with its own radius update",
            T -> (; method = :trust_region))),
    "Newton Backtracking QR" => (gs_entry("Newton Backtracking QR"),
        Paired("Newton+Backtracking+PivotedQR",
            "SimpleSolvers' own backtracking line search; a column-pivoted QR that returns the minimum-norm solution at its rank tolerance",
            T -> (; method = SS.Newton(), linesearch = SS.Backtracking(T),
                linear_solver_method = SS.PivotedQR())),
        NoMethod("SimpleNewtonRaphson takes no line search"),
        Paired("newton+BackTracking+PivotedQRSolve",
            "the backtracking line search of LineSearches; `qr(A, ColumnNorm()) \\ b`",
            T -> (; method = :newton, linesearch = LineSearches.BackTracking(),
                linsolve = PivotedQRSolve()))),
    "Newton Backtracking SVD" => (gs_entry("Newton Backtracking SVD"),
        Paired("Newton+Backtracking+SVDSolver",
            "SimpleSolvers' own backtracking line search; an SVD that returns the minimum-norm solution at its rank tolerance",
            T -> (; method = SS.Newton(), linesearch = SS.Backtracking(T),
                linear_solver_method = SS.SVDSolver())),
        NoMethod("SimpleNewtonRaphson takes no line search"),
        Paired("newton+BackTracking+SVDSolve",
            "the backtracking line search of LineSearches; `svd(A) \\ b`",
            T -> (; method = :newton, linesearch = LineSearches.BackTracking(),
                linsolve = SVDSolve()))),
    "Newton LevenbergMarquardt SVD" => (gs_entry("Newton LevenbergMarquardt SVD"),
        NoMethod("no Levenberg–Marquardt method"),
        NoMethod("no Levenberg–Marquardt method"),
        NoMethod("no Levenberg–Marquardt method")),
    "Newton reuse 3 Backtracking LU" => (gs_entry("Newton reuse 3 Backtracking LU"),
        Paired("Newton(3)+Backtracking+LapackLU",
            "refactorises at iterations 1, 3, 6, …, the column at 1, 4, 7, …",
            T -> (; method = SS.Newton(3), linesearch = SS.Backtracking(T),
                linear_solver_method = SS.LapackLU())),
        NoMethod("no factor reuse"),
        NoMethod("no factor reuse")),
    "Picard Backtracking" => (gs_entry("Picard Backtracking"),
        Paired("Picard",
            "halves its step until `‖F‖₂` does not increase, a safeguard and not a line search",
            T -> (; method = SS.Picard())),
        Paired("SimpleDFSane", "a spectral residual method, not a Picard iteration",
            T -> (; alg = SNS.SimpleDFSane())),
        NoMethod("`:anderson` with `m = 0` is a plain fixed-point step with no safeguard")),
    "Newton Backtracking GMRES" => (gs_entry("Newton Backtracking GMRES"),
        NoMethod("no Krylov linear solver"),
        NoMethod("no Krylov linear solver"),
        NoMethod("no Krylov linear solver")),
    "Broyden" => (gs_entry("Broyden"),
        NoMethod("no Broyden method"),
        Paired("SimpleBroyden",
            "updates a full inverse from a scaled identity and uses no Jacobian; the column is a depth-4 history over `Picard` damped by 0.5",
            T -> (; alg = SNS.SimpleBroyden())),
        Paired("broyden+Static",
            "updates a full inverse from the identity and uses no Jacobian; the column is a depth-4 history over `Picard` damped by 0.5",
            T -> (; method = :broyden, linesearch = LineSearches.Static()))),
    "LeftPreconditioned Picard Backtracking Newton" => (
        gs_entry("LeftPreconditioned Picard Backtracking Newton"),
        NoMethod("no nonlinear preconditioning"),
        NoMethod("no nonlinear preconditioning"),
        NoMethod("no nonlinear preconditioning"))
)

# The nine `Newton` columns with a factorisation take the shared Jacobian. `Picard`, its
# compositions and Newton–GMRES read `prob.J` whenever the problem has one, so they get a problem
# without it and keep their own Jacobian source. An `Anderson` takes it where its member does.
gs_takes_jacobian(m) = m isa GS.Newton && !(m.linsolve isa GS.GMRES)
gs_takes_jacobian(a::GS.Anderson) = gs_takes_jacobian(a.member)

"""
    Thrown(exception)

The exception that a package call threw, in place of its result.
"""
struct Thrown
    exception::Any
end

"""
    labelled(call, passed)

Run the package call `call()` and return the label of the objects `passed` that it hands to the
package, as `method`, and its result, or the exception that it throws as a `Thrown`, as `result`.
An `InterruptException` is rethrown, so that Ctrl-C stops a run.
"""
function labelled(call, passed::NamedTuple)
    result = guarded(call)
    return (; method = method_label(passed), result)
end

"""
    guarded(call)

The result of `call()`, or the exception that it throws as a `Thrown`. An `InterruptException`
is rethrown, so that Ctrl-C stops a run.
"""
function guarded(call)
    try
        call()
    catch e
        e isa InterruptException && rethrow()
        Thrown(e)
    end
end

"""
    call_package(package, objs, c, J!, x0; tol = tolerance(eltype(x0)))

Run the adapter of `package`, a `Val`, on a copy of `x0`, and return `labelled` of its call: the
label of the objects that it passes, and its result or the exception that it throws.
"""
function call_package(package::Val, objs, c, J!, x0; tol = tolerance(eltype(x0)))
    (; passed, call) = package_call(package, objs, c, J!, copy(x0); tol)
    return labelled(call, passed)
end

# Each adapter solves from `x0`, which its call may overwrite, with the residual `c`, the Jacobian
# `J!`, the objects `objs` and the absolute tolerance `tol` on `‖F‖`. It names the objects that
# it hands to the package, `passed`, passes exactly those, and returns them with the call, which
# builds the package's problem, solves, and returns the iterate, the package's own code, whether
# the package reports success, and the package's iteration count.
function package_call(::Val{:GeometricSolvers}, objs, c, J!, x0; tol)
    passed = (; method = objs.method)
    call = function ()
        prob = gs_takes_jacobian(passed.method) ? GS.NonlinearProblem(c, J!) :
               GS.NonlinearProblem(c)
        x, st = GS.solve(prob, x0, passed.method; f_abstol = tol,
            f_reltol = 0, max_iterations = MAX_ITERATIONS)
        (; x, code = string(st.code), ok = st.code == GS.SUCCESS,
            iterations = string(st.iterations))
    end
    return (; passed, call)
end

function package_call(::Val{:SimpleSolvers}, objs, c, J!, x0; tol)
    passed = objs
    keywords = Base.structdiff(passed, NamedTuple{(:method,)})
    call = function ()
        st = SS.solve_with_status!(
            x0, SS.NonlinearProblem(c, J!, x0), passed.method, nothing;
            keywords..., f_abstol = tol, f_reltol = 0,
            max_iterations = MAX_ITERATIONS, verbosity = 0)
        ok = SS.isconverged(st)
        (; x = x0, code = ok ? "converged" : "not converged", ok,
            iterations = string(st.iterations))
    end
    return (; passed, call)
end

function package_call(::Val{:SimpleNonlinearSolve}, objs, c, J!, x0; tol)
    passed = (; alg = objs.alg)
    call = function ()
        f = SciMLBase.NonlinearFunction{true}(c; jac = J!)
        sol = SciMLBase.solve(SciMLBase.NonlinearProblem(f, x0, nothing), passed.alg;
            abstol = tol, maxiters = MAX_ITERATIONS)
        (; x = sol.u, code = string(sol.retcode), ok = SciMLBase.successful_retcode(sol),
            iterations = "n/a")
    end
    return (; passed, call)
end

function package_call(::Val{:NLsolve}, objs, c, J!, x0; tol)
    passed = objs
    call = function ()
        r = NLsolve.nlsolve(NLResidual(c), NLJacobian(J!), x0; passed...,
            ftol = tol, xtol = zero(eltype(x0)), iterations = MAX_ITERATIONS)
        ok = NLsolve.converged(r)
        (; x = r.zero, code = ok ? "converged" : "not converged", ok,
            iterations = string(r.iterations))
    end
    return (; passed, call)
end

# `nsol` stops on `‖F‖₂ ≤ atol` with `rtol = 0`, and reports success with `idid = true`, or with
# `errcode = -1` when the start meets that test; its iterations are the entries of its history
# after the first. It takes the Jacobian store `FPS`, `sham` and, where the method sets one,
# `jfact`.
function package_call(::Val{:SIAMFANLEquations}, objs, c, J!, x0; tol)
    m = objs.nsol
    passed = (; nsol = m)
    call = function ()
        n = length(x0)
        jfact = m.jfact === nothing ? (;) : (; m.jfact)
        r = SIAMFANLEquations.nsol(
            NsolResidual(c), x0, similar(x0), zeros(nsol_store(m), n, n),
            NsolJacobian(J!); rtol = 0, atol = tol, maxit = MAX_ITERATIONS, sham = m.sham,
            printerr = false, jfact...)
        (; x = r.solution, code = string(r.errcode), ok = r.idid || r.errcode == -1,
            iterations = string(length(r.history) - 1))
    end
    return (; passed, call)
end

nsol_store(::Nsol{TF}) where {TF} = TF

# The adapters of the sparse rows, one per variant, on the sparse row `sp` of `problems.jl` and
# its parameter `sp.p`. `gs-sparse` runs GeometricSolvers' `Newton` with a sparse ForwardDiff
# Jacobian of greedy colouring on a problem with the sparsity prototype, which factors with
# Sparspak's LU; `gs-dense` runs `Newton()` with the shared dense Jacobian; `nlsolve-sparse` runs
# NLsolve's `:newton` with the problem's analytic sparse Jacobian into a copy of the prototype.
const SPARSE_AD = ADTypes.AutoSparse(ADTypes.AutoForwardDiff();
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function sparse_call(::Val{Symbol("gs-sparse")}, sp, c, x0; tol)
    passed = (; method = GS.Newton(; jacobian = SPARSE_AD), prototype = sp.prototype)
    call = function ()
        prob = GS.NonlinearProblem(c; jacobian_prototype = passed.prototype)
        x, st = GS.solve(prob, x0, passed.method, sp.p; f_abstol = tol, f_reltol = 0,
            max_iterations = MAX_ITERATIONS)
        (; x, code = string(st.code), ok = st.code == GS.SUCCESS,
            iterations = string(st.iterations))
    end
    return (; passed, call)
end

function sparse_call(::Val{Symbol("gs-dense")}, sp, c, x0; tol)
    passed = (; method = GS.Newton(), jacobian = SharedJacobian(c))
    call = function ()
        prob = GS.NonlinearProblem(c, passed.jacobian)
        x, st = GS.solve(prob, x0, passed.method, sp.p; f_abstol = tol, f_reltol = 0,
            max_iterations = MAX_ITERATIONS)
        (; x, code = string(st.code), ok = st.code == GS.SUCCESS,
            iterations = string(st.iterations))
    end
    return (; passed, call)
end

function sparse_call(::Val{Symbol("nlsolve-sparse")}, sp, c, x0; tol)
    passed = (; method = :newton, jacobian = AnalyticJacobian(sp.J, sp.p),
        prototype = sp.prototype)
    call = function ()
        df = NLsolve.OnceDifferentiable(ResidualAt(c, sp.p), passed.jacobian, x0,
            similar(x0), copy(passed.prototype))
        r = NLsolve.nlsolve(df, x0; method = passed.method, ftol = tol,
            xtol = zero(eltype(x0)), iterations = MAX_ITERATIONS)
        ok = NLsolve.converged(r)
        (; x = r.zero, code = ok ? "converged" : "not converged", ok,
            iterations = string(r.iterations))
    end
    return (; passed, call)
end

"""
    cell_success(ok, x, fnorm, T; tol = tolerance(T))

The success rule: the package reports success, the iterate `x` is finite, and the harness's
`‖F(x)‖∞`, `fnorm`, is at most `tol`.
"""
function cell_success(ok::Bool, x, fnorm, ::Type{T}; tol = tolerance(T)) where {T}
    ok && all(isfinite, x) &&
        fnorm ≤ tol
end

"""
    Cell

One package's result on one row and column. `status` is `ran`, `no method` or `no Float32`; a
cell that did not run has `x = nothing` and its reason in `comment`. A cell whose solve threw
has `code = ERROR`, `x = nothing` and the exception's type in `comment`.
"""
struct Cell{T}
    set::String
    problem::String
    start::String
    column::String
    package::String
    status::String
    method::String
    code::String
    success::Bool
    iterations::String
    f_evals::Int
    dual_evals::Int
    fnorm::Union{Nothing, T}
    x::Union{Nothing, Vector{T}}
    comment::String
    trace::Vector{Float64}
end

function not_run(row::Row{T}, column, package, status, comment) where {T}
    return Cell{T}(
        row.set, row.problem, row.start, column, package, status, "", "", false, "",
        0, 0, nothing, nothing, comment, Float64[])
end

"""
    run_cell(row, column, package; tol = tolerance(T))

Run `package` on `row` with its method for `column` and the absolute tolerance `tol`, and return
the cell, whose `method` is the label that the adapter gives for the objects it passes. A solve,
or the residual at its iterate, that throws an exception gives the code `ERROR` and the
exception's type in the comment, and the sweep goes on; an `InterruptException` is rethrown, so
that Ctrl-C stops a run.
"""
function run_cell(row::Row{T}, column::AbstractString, package::AbstractString;
        tol = tolerance(T)) where {T}
    package == "NLsolve" && T === Float32 &&
        return not_run(row, column, package, "no Float32", "NLsolve runs in Float64 only")
    entry = PAIRING[column][findfirst(==(package), PACKAGES)]
    entry isa NoMethod && return not_run(row, column, package, "no method", entry.reason)
    return run_invoked(
        row, column, package, adapter(package, entry.build(T), row; tol); tol)
end

"""
    adapter(package, objs, row; tol)

The adapter of `package` on `row` with the objects `objs` and the tolerance `tol`, as a function
of a counting residual `c` that returns the objects passed and the call, from a fresh copy of the
start, with the shared Jacobian of `c`.
"""
function adapter(package::AbstractString, objs, row::Row; tol)
    return c -> package_call(
        Val(Symbol(package)), objs, c, SharedJacobian(c), copy(row.x0); tol)
end

"""
    run_invoked(row, column, package, invoke; tol, p = nothing)

The cell of one solve of `row`: `invoke(c)` returns the objects passed and the call on the
counting residual `c` of `row.F`, and the harness measures `‖F(x, p)‖∞` at the iterate that the
call returns and decides success with `tol`. The cell's `column` names the column or the variant.
"""
function run_invoked(row::Row{T}, column, package, invoke; tol, p = nothing) where {T}
    c = Counted(row.F)
    (; passed, call) = invoke(c)
    (; method, result) = labelled(call, passed)
    cell(code, success, iterations, fnorm, x, comment) = Cell{T}(row.set, row.problem,
        row.start, column, package, "ran", method, code, success, iterations,
        c.primal, c.dual, fnorm, x, comment, c.trace)
    result isa Thrown &&
        return cell(
            "ERROR", false, "", nothing, nothing, string(nameof(typeof(result.exception))))
    x = Vector{T}(result.x)
    y = similar(x)
    try
        row.F(y, x, p)
    catch e
        e isa InterruptException && rethrow()
        return cell("ERROR", false, "", nothing, nothing, string(nameof(typeof(e))))
    end
    fnorm = norm(y, Inf)
    return cell(
        result.code, cell_success(result.ok, x, fnorm, T; tol), result.iterations, fnorm, x,
        "")
end

"""
    N_RUNS

The number of timed runs of a cell.
"""
const N_RUNS = 5

"""
    MIN_RUN

The shortest timed run, in seconds: a run solves `k` times, with `k` the smallest power of two
for which two successive runs each last at least this long.
"""
const MIN_RUN = 1e-3

"""
    timed_run(row, invoke, k; clock = time_ns)

The time per solve, in seconds, of one run of `k` solves of `row`: each solve calls a call of
`invoke` on its own counting residual and its own copy of the start, all made before the timer
starts, and a throw ends that solve and not the run. `clock()` gives the time in nanoseconds.
"""
function timed_run(row::Row, invoke, k::Integer; clock = time_ns)
    calls = [invoke(Counted(row.F)).call for _ in 1:k]
    t0 = clock()
    for call in calls
        guarded(call)
    end
    return (clock() - t0) / 1e9 / k
end

"""
    time_cell(row, invoke; clock = time_ns)

The times of a cell after its warm-up solve: the batch `k`, the smallest power of two for which
two successive runs each last at least `MIN_RUN`, found once, and the time per solve of each of
the `N_RUNS` runs of `k` solves. Two runs, so that one run slowed by a pause, such as a garbage
collection or the compilation of `timed_run` for the call, does not decide `k`. `clock` goes to
`timed_run`.
"""
function time_cell(row::Row, invoke; clock = time_ns)
    one_run(k) = timed_run(row, invoke, k; clock)
    k = 1
    while k * min(one_run(k), one_run(k)) < MIN_RUN
        k *= 2
    end
    return (; k, times = [one_run(k) for _ in 1:N_RUNS])
end

"""
    run_median(ts)

The median of the run times `ts`.
"""
function run_median(ts)
    s = sort(ts)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"""
    run_spread(ts)

The spread of the run times `ts`: `(max − min) / median`.
"""
run_spread(ts) = (maximum(ts) - minimum(ts)) / run_median(ts)

"""
    worker_cmd(script, args...)

The command that runs `script` with `args` in a fresh Julia process of this environment, with
`--startup-file=no` and one thread.
"""
function worker_cmd(script, args...)
    `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(@__DIR__) $script $args`
end

"""
    spawn_worker(script, args...)

Run `worker_cmd(script, args...)` and wait for it.
"""
spawn_worker(script, args...) = run(worker_cmd(script, args...))

"""
    eltype_named(name)

The element type `Float32` or `Float64` that `name` names.
"""
function eltype_named(name)
    name == "Float32" ? Float32 :
    name == "Float64" ? Float64 :
    error("no element type $name")
end

"""
    merge_part_files(out, header, cells, group, key, part)

Write `header` to `out`, then, for each cell `c` of `cells` in order, the next line of the part
file `part(group(c))`, which must start with `key(c) * ","`. Throw an error, and leave `out` and
the parts as they are, when a part has no line for a cell or a line left over, or a line is
another cell's; remove the parts after the merge.
"""
function merge_part_files(out, header, cells, group, key, part)
    gs = unique(group.(cells))
    lines = Dict(g => readlines(part(g)) for g in gs)
    next = Dict(g => 0 for g in gs)
    merged = map(cells) do c
        g = group(c)
        next[g] < length(lines[g]) || error("$(part(g)): no line for the cell $(key(c))")
        line = lines[g][next[g] += 1]
        startswith(line, key(c) * ",") ||
            error("$(part(g)): line $(next[g]) is not the cell $(key(c))")
        line
    end
    for g in gs
        next[g] == length(lines[g]) || error("$(part(g)): lines left over")
    end
    write(out, join([header; merged], "\n") * "\n")
    foreach(g -> rm(part(g)), gs)
end

"""
    start_worker(label)

Set one BLAS thread, check that the process runs one Julia thread, and print `label` with the
`uptime` load at the start of the process.
"""
function start_worker(label)
    BLAS.set_num_threads(1)
    Threads.nthreads() == 1 || error("a timing process runs $(Threads.nthreads()) threads")
    println(
        label, ": uptime ", readchomp(`uptime`), "; BLAS threads ", BLAS.get_num_threads())
end

"""
    agrees(xa, xb, T)

Whether `xb` lies within `tolerance(T) · max(1, ‖xa‖∞)` of `xa` in `‖·‖∞`.
"""
function agrees(xa, xb, ::Type{T}) where {T}
    norm(xb - xa, Inf) ≤ tolerance(T) * max(1, norm(xa, Inf))
end

"""
    verdict(successes, xs, T)

The verdict of a row over the cells that ran, in the order of `PACKAGES`: their successes and
their iterates. `single` when one cell ran; `all fail` when none succeeded; `disagree` when some
succeeded and some did not; `agree` when all succeeded and every iterate agrees with the first;
`different root` when all succeeded and one does not.
"""
function verdict(successes::AbstractVector{Bool}, xs, ::Type{T}) where {T}
    length(successes) == 1 && return "single"
    any(successes) || return "all fail"
    all(successes) || return "disagree"
    return all(x -> agrees(xs[1], x, T), xs) ? "agree" : "different root"
end

function verdict(cells::AbstractVector{<:Cell{T}}) where {T}
    ran = filter(c -> c.status == "ran", cells)
    return verdict([c.success for c in ran], [c.x for c in ran], T)
end

"""
    Explanation(id, title, applies)

One cause of a disagreement: `applies(row, column, cells, verdict)` is true for the rows it
explains.
"""
struct Explanation
    id::String
    title::String
    applies::Function
end

# Whether `package` ran on the row and failed, and its code.
function fails(cells, package)
    any(c -> c.package == package && c.status == "ran" && !c.success, cells)
end
code_of(cells, package) = cells[findfirst(c -> c.package == package, cells)].code

"""
    polish(F, x)

`x` refined by GeometricSolvers' `Newton()` in `Float64` with no tolerance, for at most 200
iterations: the root that a solution `x` approximates.
"""
function polish(F, x)
    y, _ = GS.solve(GS.NonlinearProblem(F), Float64.(x), GS.Newton(); f_abstol = 0,
        f_reltol = 0, max_iterations = MAX_ITERATIONS)
    return y
end

"""
    one_root(row, cells)

Whether the solutions of the cells that ran polish to one root, within
`sqrt(eps(Float64)) · max(1, ‖r‖∞)` of the first one, `r`.
"""
function one_root(row, cells)
    roots = [polish(row.F, c.x) for c in cells if c.status == "ran"]
    return all(r -> agrees(roots[1], r, Float64), roots)
end

const LINESEARCH_COLUMNS = ("Newton Backtracking LU", "Newton Bisection LU",
    "Newton StrongWolfe LU", "Newton Backtracking QR", "Newton Backtracking SVD",
    "Newton reuse 3 Backtracking LU")

"""
    EXPLANATIONS

The causes of the disagreements, each with its id and title; `benchmark/REPORT.md` gives each
cause from an iteration trace. A row with the verdict `disagree` or `different root` names the
first explanation that applies to it.
"""
const EXPLANATIONS = [
    Explanation("E1", "Sample problem 18 has two lines of roots",
        (row, col, cells, v) -> v == "different root" && row.set == "standard" &&
                                row.problem == "18"),
    Explanation("E2", "One root reached to different accuracies",
        (row, col, cells, v) -> v == "different root" && one_root(row, cells)),
    Explanation("E3", "Different roots of one system",
        (row, col, cells, v) -> v == "different root"),
    Explanation("E4", "The Picard column pairs three different fixed-point methods",
        (row, col, cells, v) -> col == "Picard Backtracking"),
    Explanation("E5", "GeometricSolvers' Broyden diverges or reaches the iteration cap",
        (row, col, cells, v) -> col == "Broyden" && fails(cells, "GeometricSolvers")),
    Explanation("E6", "SimpleBroyden fails where GeometricSolvers' Broyden succeeds",
        (row, col, cells, v) -> col == "Broyden"),
    Explanation("E7", "Four dogleg trust regions with different radius updates",
        (row, col, cells, v) -> col == "Newton DogLeg LU"),
    Explanation("E8", "The undamped Newton step and the stop rules of each package",
        (row, col, cells, v) -> col == "Newton Static LU"),
    Explanation("E9", "SimpleSolvers' rank-revealing solvers stall",
        (row, col, cells, v) -> col in
                                ("Newton Backtracking QR", "Newton Backtracking SVD") &&
                                fails(cells, "SimpleSolvers")),
    Explanation("E10", "GeometricSolvers' line search ends the solve",
        (row, col, cells, v) -> col in LINESEARCH_COLUMNS &&
                                fails(cells, "GeometricSolvers")),
    Explanation("E11", "SimpleSolvers' line search reaches the iteration cap or stops",
        (row, col, cells, v) -> col in LINESEARCH_COLUMNS && fails(cells, "SimpleSolvers")),
    Explanation("E12", "NLsolve's strong-Wolfe line search throws",
        (row, col, cells, v) -> col in LINESEARCH_COLUMNS && fails(cells, "NLsolve") &&
                                code_of(cells, "NLsolve") == "ERROR")
]

"""
    explain(row, column, cells, verdict)

The id of the first explanation of `EXPLANATIONS` that applies to a row with the verdict
`disagree` or `different root`, or the empty string.
"""
function explain(row, column, cells, v)
    v in ("disagree", "different root") || return ""
    i = findfirst(e -> e.applies(row, column, cells, v), EXPLANATIONS)
    return i === nothing ? "" : EXPLANATIONS[i].id
end

const CELLS_HEADER = "set,problem,start,eltype,column,package,status,method,code,success,iterations,f_evals,dual_evals,fnorm_inf,x,comment"
const ROWS_HEADER = "set,problem,start,eltype,column,verdict,explanation"

function write_cell(io::IO, c::Cell{T}) where {T}
    ran = c.status == "ran"
    fields = (
        c.set, c.problem, c.start, string(T), c.column, c.package, c.status, c.method,
        c.code, ran ? string(c.success) : "", c.iterations,
        ran ? string(c.f_evals) : "", ran ? string(c.dual_evals) : "",
        c.fnorm === nothing ? "" : repr(c.fnorm),
        c.x === nothing ? "" : join(repr.(c.x), ";"), c.comment)
    println(io, join(fields, ","))
end

"""
    report_tables(io, cells, rows)

Write the report's tables: the pairing table, the success counts per package × column ×
precision, the rows per verdict and set, and the rows per explanation.
"""
function report_tables(io::IO, cells, rows)
    println(io, "### Pairing\n")
    println(io, "| column | package | method | difference from the column |")
    println(io, "|:--|:--|:--|:--|")
    for column in COLUMNS, (package, entry) in zip(PACKAGES, PAIRING[column])

        m, d = entry isa Paired ? ("`$(entry.label)`", entry.difference) :
               ("none", entry.reason)
        println(io, "| $column | $package | $m | $d |")
    end
    println(io, "\n### Success counts\n")
    heads = ["$p $T" for p in PACKAGES for T in ("Float32", "Float64")]
    println(io, "| column | ", join(heads, " | "), " |")
    println(io, "|:--|", repeat("--:|", length(heads)))
    for column in COLUMNS
        counts = map(Iterators.product(("Float32", "Float64"), PACKAGES)) do (T, p)
            ran = filter(
                c -> c.column == column && c.package == p && c.status == "ran" &&
                     cell_eltype(c) == T,
                cells)
            isempty(ran) ? "—" : "$(count(c -> c.success, ran))/$(length(ran))"
        end
        println(io, "| $column | ", join(vec(counts), " | "), " |")
    end
    println(io, "\n### Verdicts\n")
    groups = [(s, T) for s in ("standard", "midpoint") for T in ("Float32", "Float64")]
    println(io, "| verdict | ", join(("$s $T" for (s, T) in groups), " | "), " |")
    println(io, "|:--|", repeat("--:|", length(groups)))
    for v in ("single", "all fail", "disagree", "agree", "different root")
        n = [count(r -> r.verdict == v && r.set == s && r.eltype == T, rows)
             for (s, T) in groups]
        println(io, "| $v | ", join(n, " | "), " |")
    end
    println(io, "\n### Rows per explanation\n")
    for e in EXPLANATIONS
        println(io, "- $(e.id) · $(e.title): $(count(r -> r.explanation == e.id, rows))")
    end
end

cell_eltype(::Cell{T}) where {T} = string(T)
