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
- `rank::Int32`: the numerical rank from the last fresh SVD factorisation, or `Int32(-1)`
  when no SVD factorisation completed.
- `nullnorm::R`: ``\\|(I - U_r U_r^H) F(x_k)\\|`` at the iterate ``x_k`` of the last fresh
  SVD factorisation, with ``U_r`` the first `rank` columns of ``U``, or `R(NaN)` when no
  SVD factorisation completed.
- `gradratio::R`: ``\\|J(x_k)^H F(x_k)\\| / (\\|J(x_k)\\|_F \\|F(x_k)\\|)`` at the iterate
  ``x_k`` of the last fresh factorisation, `zero(R)` when ``\\|J(x_k)\\|_F = 0`` or
  ``\\|F(x_k)\\| = 0``, or `R(NaN)` when no Jacobian was factorised.

The three numbers are those of the last fresh factorisation (``x_k``), not of the iterate
``x`` at return. For a [`Newton`](@ref) solve with `n ≥ 2` (for `n = 1`, any nonzero scalar
Jacobian has `gradratio == 1` and `rank == 1`), they classify the outcome and a `STALLED` or
`MAXITERS` return: a consistent rank-deficient system that converged has a small residual
(`fnorm` small) and nothing outside the range (`rank < n`, `nullnorm ≈ 0`); a least-squares
point with no root has a ratio near zero (`gradratio ≈ 0`, `fnorm > 0`) and the residual
outside the range (`rank < n`, ``\\text{nullnorm} \\approx \\|F(x_k)\\| > 0``); a stall in
nearly singular directions has the residual inside the range (`nullnorm ≈ 0`) at a fine SVD
cutoff and outside it (`rank < n`, ``\\text{nullnorm} \\approx \\|F(x_k)\\|``) at a coarse
cutoff; and an in-range stall from conditioning or step acceptance has `gradratio` bounded
away from zero and `nullnorm ≈ 0`.

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
    rank::Int32
    nullnorm::R
    gradratio::R
end

@inline function SolverStatus{R}(
        code::ReturnCode, iterations::Int32, fnorm::R, stepnorm::R,
        step_failures::Int32, promoted::Bool) where {R <: Real}
    return SolverStatus{R}(
        code, iterations, fnorm, stepnorm, step_failures, promoted,
        Int32(-1), R(NaN), R(NaN))
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
- `limited::Bool`: the step was accepted, but a trust-region radius truncated it, so its norm is
  set by the radius and does not measure the distance to a root.
"""
struct StepInfo{R <: Real}
    code::ReturnCode
    fnorm::R
    stepnorm::R
    failures::Int32
    promoted::Bool
    limited::Bool
end

"""
    record(status::SolverStatus{R}, info::StepInfo{R})

Fold the result of one step into the status: count the iteration, take the code and the norms of
the step, add its failures, keep `promoted` once it is set, and preserve `rank`, `nullnorm` and
`gradratio`.
"""
function record(status::SolverStatus{R}, info::StepInfo{R}) where {R}
    SolverStatus{R}(info.code,
        status.iterations + one(Int32),
        info.fnorm,
        info.stepnorm,
        status.step_failures + info.failures,
        status.promoted | info.promoted,
        status.rank,
        status.nullnorm,
        status.gradratio)
end

const ReturnCodeCounts = NamedTuple{
    (:SUCCESS, :STALLED, :MAXITERS, :SINGULAR, :NONFINITE, :LINESEARCH_FAILED),
    NTuple{6, Int}
}

"""
    isconverged(status::SolverStatus)
    isconverged(counts::NamedTuple{(:SUCCESS, :STALLED, :MAXITERS, :SINGULAR, :NONFINITE, :LINESEARCH_FAILED)})

Whether the solve ended with `SUCCESS`, or whether every system in a batch return-code count
summary ended with `SUCCESS` (`counts.SUCCESS == sum(counts)`).
"""
isconverged(status::SolverStatus) = status.code == SUCCESS
isconverged(counts::ReturnCodeCounts) = counts.SUCCESS == sum(counts)

"""
    isstalled(status::SolverStatus)

Whether the solve ended with `STALLED`.
"""
isstalled(status::SolverStatus) = status.code == STALLED
