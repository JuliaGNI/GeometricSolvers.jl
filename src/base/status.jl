"""
    SolverStatus{R <: Real}

The result of a solve. It is `isbits`, so a kernel can write one per system, and every real field
is in `R = real(T)` for the element type `T` of the iterate.

# Fields

- `code::ReturnCode`: why the solve ended.
- `iterations::Int32`: the number of steps taken.
- `fnorm::R`: ``\\|F(x)\\|`` at return.
- `stepnorm::R`: ``\\|Δx\\|`` of the last step.
- `step_failures::Int32`: every step-rule failure of the solve, such as a failed line search, in
  one count.
- `promoted::Bool`: a factorisation ran in a higher precision than the factorisation precision
  of its method, after a stall or as a vendor fallback.

The predicates `GeometricSolvers.isconverged` and `GeometricSolvers.isstalled` read the code. They
are not exported, because they are generic names that a package doing `using GeometricSolvers`
may want for itself.
"""
struct SolverStatus{R <: Real}
    code::ReturnCode
    iterations::Int32
    fnorm::R
    stepnorm::R
    step_failures::Int32
    promoted::Bool
end

"""
    StepInfo{R <: Real}

The result of one step, which [`record`](@ref) folds into a [`SolverStatus`](@ref).

# Fields

- `code::ReturnCode`: the outcome of the step.
- `fnorm::R`: ``\\|F(x₊)\\|`` at the new iterate, so that the loop needs no second reduction and,
  on a device, no second synchronisation.
- `stepnorm::R`: ``\\|x₊ - x\\|``.
- `failures::Int32`: the step-rule failures inside this step, those of its members included.
- `promoted::Bool`: the step factorised in a higher precision than its method's.
"""
struct StepInfo{R <: Real}
    code::ReturnCode
    fnorm::R
    stepnorm::R
    failures::Int32
    promoted::Bool
end

"""
    record(status::SolverStatus{R}, info::StepInfo{R})

Fold the result of one step into the status: count the iteration, take the code and the norms of
the step, add its failures, and keep `promoted` once it is set.
"""
function record(status::SolverStatus{R}, info::StepInfo{R}) where {R}
    SolverStatus{R}(info.code,
        status.iterations + one(Int32),
        info.fnorm,
        info.stepnorm,
        status.step_failures + info.failures,
        status.promoted | info.promoted)
end

"""
    isconverged(status::SolverStatus)

Whether the solve ended with `SUCCESS`.
"""
isconverged(status::SolverStatus) = status.code == SUCCESS

"""
    isstalled(status::SolverStatus)

Whether the solve ended with `STALLED`.
"""
isstalled(status::SolverStatus) = status.code == STALLED
