module GeometricSolversSciMLBaseExt

import CommonSolve: init, solve
using GeometricSolvers: GeometricSolvers, Newton, NonlinearProblem, Picard, ReturnCode
using SciMLBase: SciMLBase
using StaticArrays: StaticArray

const SCIML_KEYWORDS = (:abstol, :reltol, :maxiters, :options)

# The SciML code of each `ReturnCode`, in the order of the `@enum` declaration. A code with no
# entry raises a `BoundsError`.
const SCIML_RETCODES = (
    SciMLBase.ReturnCode.Success,
    SciMLBase.ReturnCode.Stalled,
    SciMLBase.ReturnCode.MaxIters,
    SciMLBase.ReturnCode.InternalLinearSolveFailed,
    SciMLBase.ReturnCode.Unstable,
    SciMLBase.ReturnCode.InternalLineSearchFailed
)

GeometricSolvers.sciml_retcode(code::ReturnCode) = SCIML_RETCODES[Integer(code) + 1]

function refuse_set(value, name)
    value === nothing || throw(
        ArgumentError(
        "GeometricSolvers does not support a SciML `NonlinearProblem` with `$name` set"
    )
    )
end

function translate(prob::SciMLBase.NonlinearProblem, kwargs)
    kws = merge(values(prob.kwargs), values(kwargs))
    bad = filter(∉(SCIML_KEYWORDS), keys(kws))
    if !isempty(bad)
        list = join(("`$k`" for k in bad), ", ")
        throw(
            ArgumentError(
            "a SciML solve with GeometricSolvers takes the keywords `abstol`, `reltol`, `maxiters` and `options`; got $list"
        )
        )
    end

    f = prob.f
    refuse_set(f.jac_prototype, "jac_prototype")
    refuse_set(f.sparsity, "sparsity")
    refuse_set(prob.lb, "lb")
    refuse_set(prob.ub, "ub")

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
