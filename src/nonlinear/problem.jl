# The problem: the residual and its derivative, and nothing else. Parameters arrive at solve time,
# so one problem serves a whole time-stepping loop, and a batch shares one object.

"""
    AbstractNonlinearProblem

The supertype of the problems a [`NonlinearSolver`](@ref) solves: [`NonlinearProblem`](@ref), whose
residual has as many entries as the iterate has variables, and [`LeastSquaresProblem`](@ref), whose
residual may have more. [`init`](@ref), [`solve`](@ref), [`solve!`](@ref), [`solve!!`](@ref) and
[`solve_batch!`](@ref) take it, and so does every method — [`Newton`](@ref), [`Picard`](@ref) and
each globalization — so one solver loop serves both problem types.
"""
abstract type AbstractNonlinearProblem end

"""
    NonlinearProblem(F)
    NonlinearProblem(F, J)

The residual `F` of a nonlinear system, and optionally its derivative `J`. The problem holds
functions only — no parameters, no iterate and no flag — so it is `isbits` when `F` and `J` are
and a kernel can hold it.

The call form follows the storage of the iterate, and [`residual!!`](@ref) selects it on the
storage alone: with no buffer slot the call is `F(x, p)`, for a `Number`, an `MVector` — which
counts as a value — and an `Array` alike. `J` is the derivative of `F`, which
[`jvp!!`](@ref) and [`jacobian!!`](@ref) read for the source `nothing`; `AutoForwardDiff` instead
differentiates `F`. A method's own operator is a separate Jacobian source.

```jldoctest
julia> prob = NonlinearProblem((x, p) -> x .* x .- 4);

julia> prob.F(3.0, nothing)
5.0

julia> prob.J === nothing
true
```
"""
struct NonlinearProblem{F, J} <: AbstractNonlinearProblem
    F::F
    J::J
end

NonlinearProblem(F) = NonlinearProblem(F, nothing)

@inline output_length(::NonlinearProblem, n::Int) = n
