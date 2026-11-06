# The problem: the residual, optionally its derivative, and optionally a Jacobian prototype.
# Parameters arrive at solve time, so one problem serves a whole time-stepping loop, and a batch
# shares one object.

"""
    NonlinearProblem(F; jacobian_prototype = nothing)
    NonlinearProblem(F, J; jacobian_prototype = nothing)

The residual `F` of a nonlinear system, optionally its derivative `J`, and optionally a matrix
`jacobian_prototype` giving the storage type and sparsity pattern of the Jacobian. With no
prototype (`jacobian_prototype = nothing`), the problem holds functions only — no parameters, no
iterate and no flag — so it is `isbits` when `F` and `J` are and a kernel can hold it.

The call form follows the storage of the iterate, and [`residual!!`](@ref) selects it on the
storage alone: with no buffer slot the call is `F(x, p)`, for a `Number`, an `MVector` — which
counts as a value — and an `Array` alike. `J` is the derivative of `F`, which
[`jvp!!`](@ref) and [`jacobian!!`](@ref) read for the source `nothing`; `AutoForwardDiff` instead
differentiates `F`. A method's own operator is a separate Jacobian source. When
`jacobian_prototype` is a `SparseMatrixCSC`, `Newton` allocates its Jacobian buffer and linear
factorisation from that pattern and `AutoSparse` uses its sparsity pattern directly. A linear
solver exists only for a real host `Vector` iterate with a `SparseMatrixCSC{<:Any, Int}`
prototype, with `Sparspak` loaded and the default `LUFactorization`; a `QRFactorization`, an
`SVDFactorization`, a dense prototype and a device iterate raise `MethodError` when `Newton`
initialises. `Picard` and a static or batched iterate ignore the prototype.

```jldoctest
julia> prob = NonlinearProblem((x, p) -> x .* x .- 4);

julia> prob.F(3.0, nothing)
5.0

julia> prob.J === nothing && prob.jacobian_prototype === nothing
true
```
"""
struct NonlinearProblem{F, J, P <: Union{Nothing, AbstractMatrix}}
    F::F
    J::J
    jacobian_prototype::P
end

function NonlinearProblem(F, J = nothing; jacobian_prototype = nothing)
    NonlinearProblem(F, J, jacobian_prototype)
end

"""
    jacobian_prototype(prob)

Return the Jacobian prototype matrix stored on `prob`, or `nothing` when none is supplied.
"""
jacobian_prototype(prob::NonlinearProblem) = prob.jacobian_prototype
jacobian_prototype(::Nothing) = nothing
