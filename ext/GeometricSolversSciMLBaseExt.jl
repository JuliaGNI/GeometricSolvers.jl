module GeometricSolversSciMLBaseExt

import CommonSolve: init, solve
using GeometricSolvers: GeometricSolvers, LINESEARCH_FAILED, MAXITERS, NONFINITE,
                        Newton, NonlinearProblem, Picard, ReturnCode, SINGULAR,
                        STALLED, SUCCESS
using SciMLBase: SciMLBase
using StaticArrays: StaticArray

const SCIML_KEYWORDS = (:abstol, :reltol, :maxiters, :options)

function GeometricSolvers.sciml_retcode(code::ReturnCode)
    code === SUCCESS && return SciMLBase.ReturnCode.Success
    code === STALLED && return SciMLBase.ReturnCode.Stalled
    code === MAXITERS && return SciMLBase.ReturnCode.MaxIters
    code === SINGULAR && return SciMLBase.ReturnCode.InternalLinearSolveFailed
    code === NONFINITE && return SciMLBase.ReturnCode.Unstable
    return SciMLBase.ReturnCode.InternalLineSearchFailed
end

function translate(prob::SciMLBase.NonlinearProblem, kwargs)
    kws = merge(values(prob.kwargs), values(kwargs))
    bad = [k for k in keys(kws) if k ∉ SCIML_KEYWORDS]
    if !isempty(bad)
        list = join(("`$k`" for k in bad), ", ")
        throw(
            ArgumentError(
            "a SciML solve with GeometricSolvers takes the keywords `abstol`, `reltol`, `maxiters` and `options`; got $list"
        )
        )
    end

    f = prob.f
    f.jac_prototype === nothing || throw(
        ArgumentError(
        "GeometricSolvers does not support a SciML `NonlinearProblem` with `jac_prototype` set"
    )
    )
    f.sparsity === nothing || throw(
        ArgumentError(
        "GeometricSolvers does not support a SciML `NonlinearProblem` with `sparsity` set"
    )
    )
    prob.lb === nothing || throw(
        ArgumentError(
        "GeometricSolvers does not support a SciML `NonlinearProblem` with `lb` set"
    )
    )
    prob.ub === nothing || throw(
        ArgumentError(
        "GeometricSolvers does not support a SciML `NonlinearProblem` with `ub` set"
    )
    )

    u0 = prob.u0
    isval = u0 isa Union{StaticArray, Number}
    iip = SciMLBase.isinplace(prob)
    if isval && iip
        throw(
            ArgumentError(
            "an in-place SciML `f` needs an array `u0` that is not a `StaticArray`; got $(typeof(u0))"
        )
        )
    end
    wrap = !isval && !iip
    F = wrap ? (y, x, p) -> (y .= f(x, p)) : f
    J = f.jac === nothing ? nothing : (wrap ? (A, x, p) -> (A .= f.jac(x, p)) : f.jac)

    options = get(kws, :options, nothing)
    nt = NamedTuple()
    if haskey(kws, :abstol)
        nt = merge(nt, (; f_abstol = kws.abstol, f_reltol = 0))
    end
    if haskey(kws, :reltol)
        nt = merge(nt, (; f_reltol = kws.reltol))
    end
    if haskey(kws, :maxiters)
        nt = merge(nt, (; max_iterations = kws.maxiters))
    end

    return NonlinearProblem(F, J), u0, prob.p, options, nt
end

function solve(prob::SciMLBase.NonlinearProblem, method::Union{Newton, Picard}; kwargs...)
    gprob, u0, p, options, nt = translate(prob, kwargs)
    return solve(gprob, u0, method, p; options, nt...)
end

function init(prob::SciMLBase.NonlinearProblem, method::Union{Newton, Picard}; kwargs...)
    prob0, u0, p, options, nt = translate(prob, kwargs)
    return init(prob0, u0, method, p; options, nt...)
end

end
