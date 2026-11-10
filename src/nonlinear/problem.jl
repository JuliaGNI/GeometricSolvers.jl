# The problem: the residual, optionally its derivative, and optionally a Jacobian prototype.
# Parameters arrive at solve time, so one problem serves a whole time-stepping loop, and a batch
# shares one object.

"""
    AbstractNonlinearProblem

The supertype of the problems a [`NonlinearSolver`](@ref) solves: [`NonlinearProblem`](@ref), whose
residual has as many entries as the iterate has variables, [`StationarityProblem`](@ref), whose
residual is a part of the gradient of a scalar action, and [`LeastSquaresProblem`](@ref), whose
residual may have more entries than the iterate has variables. [`init`](@ref), [`solve`](@ref),
[`solve!`](@ref) and [`solve!!`](@ref) take it, and so does [`Newton`](@ref) with each of its
globalizations, so one solver loop serves every problem type; [`Picard`](@ref), the combinators
[`Additive`](@ref), [`Multiplicative`](@ref), [`Repeat`](@ref) and [`Switch`](@ref), and
[`solve_batch!`](@ref) take a [`NonlinearProblem`](@ref) or a [`StationarityProblem`](@ref), and
a combinator raises an `ArgumentError` for a [`LeastSquaresProblem`](@ref).
"""
abstract type AbstractNonlinearProblem end

"""
    NonlinearProblem(F; jacobian_prototype = nothing)
    NonlinearProblem(F, J; jacobian_prototype = nothing)
    NonlinearProblem(F, J; jacobian_prototype = nothing,
        block_residuals = nothing, block_jacobians = nothing)

The residual `F` of a nonlinear system, optionally its derivative `J`, and optionally a matrix
`jacobian_prototype` giving the storage type and sparsity pattern of the Jacobian. With no
prototype (`jacobian_prototype = nothing`), the problem holds functions only — no parameters, no
iterate and no flag — so it is `isbits` when `F` and `J` are and a kernel can hold it.

With `block_residuals` given, the problem also carries one local residual `F_i!(f_i, x_i, x, p)`
per block of a [`BlockPartition`](@ref) — writing the residual of block `i` on its own index set
into `f_i`, reading external couplings from the frozen full iterate `x` — and optionally one block
Jacobian `J_i!(J_i, x_i, x, p)` per block, and its `F` is the [`BlockResidual`](@ref) that holds
them. `J` stays as the caller gives it; [`RAS`](@ref) reads the block functions and every other
method the global ones. `block_jacobians` without `block_residuals`, and the two of different
lengths, raise an `ArgumentError`.

The call form follows the storage of the iterate, and [`residual!!`](@ref) selects it on the
storage alone: with no buffer slot the call is `F(x, p)`, for a `Number`, an `MVector` — which
counts as a value — and an `Array` alike. `J` is the derivative of `F`, which
[`jvp!!`](@ref) and [`jacobian!!`](@ref) read for the source `nothing`; `AutoForwardDiff` instead
differentiates `F`. A method's own operator is a separate Jacobian source. When
`jacobian_prototype` is a `SparseMatrixCSC`, `Newton` allocates its Jacobian buffer and linear
factorisation from that pattern and `AutoSparse` uses its sparsity pattern directly. A linear
solver exists only for a real host `Vector` iterate with a `SparseMatrixCSC{<:Any, Int}`
prototype, with `Sparspak` loaded and an `LUFactorization` in the iterate's element type; a
`QRFactorization`, an `SVDFactorization` and a dense prototype raise `MethodError` when `Newton`
initialises. On a device iterate, `Newton(; jacobian = nothing)` raises `MethodError` from the
factor, with an analytic `J` or without, and an `AutoSparse` `jacobian` raises `ArgumentError` in
[`prepare_ad`](@ref) first, as any back end other than `AutoForwardDiff()` does. `Picard` and a
static or batched iterate ignore the prototype.

```jldoctest
julia> prob = NonlinearProblem((x, p) -> x .* x .- 4);

julia> prob.F(3.0, nothing)
5.0

julia> prob.J === nothing && prob.jacobian_prototype === nothing
true
```
"""
struct NonlinearProblem{F, J, P <: Union{Nothing, AbstractMatrix}} <:
       AbstractNonlinearProblem
    F::F
    J::J
    jacobian_prototype::P
end

function NonlinearProblem(F, J = nothing; jacobian_prototype = nothing,
        block_residuals = nothing, block_jacobians = nothing)
    if block_residuals === nothing
        block_jacobians === nothing || throw(ArgumentError(
            "NonlinearProblem takes block_jacobians only together with block_residuals"))
        return NonlinearProblem(F, J, jacobian_prototype)
    end
    if block_jacobians !== nothing && length(block_jacobians) != length(block_residuals)
        throw(ArgumentError(
            "block_jacobians and block_residuals must hold one function per block, so they must have the same length"))
    end
    return NonlinearProblem(
        BlockResidual(F, block_residuals, block_jacobians), J, jacobian_prototype)
end

"""
    BlockResidual(F, block_residuals, block_jacobians)

The residual of a [`NonlinearProblem`](@ref) that also carries the local residuals of a
[`BlockPartition`](@ref): the global residual `F`, the tuple of block residuals, and the tuple of
block Jacobians or `nothing`. A call forwards every argument to the global `F` unchanged, so every
method that calls the problem's residual reads one residual per entry of the iterate;
[`RAS`](@ref) reads the block functions through [`block_residuals`](@ref) and
[`block_jacobians`](@ref) instead, and calls the global `F` no times during a sweep.
"""
struct BlockResidual{Fg, B, BJ} <: Function
    F::Fg
    block_residuals::B
    block_jacobians::BJ
end

@inline (b::BlockResidual)(args...) = b.F(args...)

"""
    block_residuals(prob)

The tuple of block residuals of `prob`, each of the form `F_i!(f_i, x_i, x, p)`, or `nothing` when
`prob` carries none. A problem built without the `block_residuals` keyword, and any problem that is
not a [`NonlinearProblem`](@ref), gives `nothing`.
"""
block_residuals(prob::NonlinearProblem) = _block_residuals(prob.F)
block_residuals(::AbstractNonlinearProblem) = nothing
@inline _block_residuals(F::BlockResidual) = F.block_residuals
@inline _block_residuals(F) = nothing

"""
    block_jacobians(prob)

The tuple of block Jacobians of `prob`, each of the form `J_i!(J_i, x_i, x, p)`, or `nothing` when
`prob` carries none. A block without a Jacobian of its own is differentiated by
[`RAS`](@ref) through `AutoForwardDiff()` on its block residual.
"""
block_jacobians(prob::NonlinearProblem) = _block_jacobians(prob.F)
block_jacobians(::AbstractNonlinearProblem) = nothing
@inline _block_jacobians(F::BlockResidual) = F.block_jacobians
@inline _block_jacobians(F) = nothing

"""
    block_residual(prob, i)

The block residual of block `i` of `prob`, in the form `F_i!(f_i, x_i, x, p)`.
"""
block_residual(prob::AbstractNonlinearProblem, i::Integer) = block_residuals(prob)[i]

"""
    block_jacobian(prob, i)

The block Jacobian of block `i` of `prob`, in the form `J_i!(J_i, x_i, x, p)`, or `nothing` when
block `i` has none.
"""
function block_jacobian(prob::AbstractNonlinearProblem, i::Integer)
    J = block_jacobians(prob)
    return J === nothing ? nothing : J[i]
end

"""
    jacobian_prototype(prob)

Return the Jacobian prototype matrix stored on `prob`, or `nothing` when none is supplied. A
[`StationarityProblem`](@ref) and a [`LeastSquaresProblem`](@ref) hold no prototype, so their
Jacobians are dense.
"""
jacobian_prototype(prob::NonlinearProblem) = prob.jacobian_prototype
jacobian_prototype(::AbstractNonlinearProblem) = nothing
jacobian_prototype(::Nothing) = nothing
