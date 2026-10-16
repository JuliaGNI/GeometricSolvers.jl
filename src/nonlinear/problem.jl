# The problem: the residual and its derivative, and nothing else. Parameters arrive at solve time
# (§4.3), so one problem serves a whole time-stepping loop, and a batch shares one object.

"""
    NonlinearProblem(F)
    NonlinearProblem(F, J)

The residual `F` of a nonlinear system, and optionally its derivative `J`. The problem holds
functions only — no parameters, no iterate and no flag — so it is `isbits` when `F` and `J` are
and `init` can hand it to a kernel.

The call form follows the storage of the iterate, and [`residual!!`](@ref) selects it on the
storage alone: with no buffer slot the call is `F(x, p)`, for a `Number`, an `MVector` — which
counts as a value (§4.3) — and an `Array` alike. `J` is the derivative of `F`, which
[`jvp!!`](@ref) always reads and [`jacobian!!`](@ref) reads for the source `nothing`; a method's
own operator is a separate argument of the method (§4.3).

```jldoctest
julia> prob = NonlinearProblem((x, p) -> x .* x .- 4);

julia> prob.F(3.0, nothing)
5.0

julia> prob.J === nothing
true
```
"""
struct NonlinearProblem{F, J}
    F::F
    J::J
end

NonlinearProblem(F) = NonlinearProblem(F, nothing)
