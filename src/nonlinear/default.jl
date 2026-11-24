# The method of a call that names none: `default_method(prob, x₀)` reads the problem type, the
# Jacobian prototype and the storage of the iterate, and `init` and `solve` with no method argument
# take it with `p = nothing`.

"""
    default_method(prob, x₀)

Return the method that [`init`](@ref)`(prob, x₀)` and [`solve`](@ref)`(prob, x₀)` take when no
method is given, from the type of `prob`, its Jacobian prototype and the storage of `x₀`:

| problem | iterate | method |
|:--|:--|:--|
| [`NonlinearProblem`](@ref) with a `SparseMatrixCSC` `jacobian_prototype` | `Vector{Float64}` or `Vector{ComplexF64}` | `Newton(; linsolve = UMFPACKFactorization())` |
| [`NonlinearProblem`](@ref), every other prototype and iterate | any other | `Newton()` |
| [`StationarityProblem`](@ref) | any | `Newton()` |
| [`LeastSquaresProblem`](@ref) | any | [`GaussNewton`](@ref)`()` |

The `LUFactorization()` of `Newton()` selects its factor from the storage of the Jacobian, so
the rule reads no array type and no size of a dense iterate. A `SparseMatrixCSC` prototype on any
other iterate takes `Newton()` as well: a `Vector{Float32}` gets the Sparspak factor where
Sparspak is loaded, and `init` raises otherwise, as it does when the caller names `Newton()`. No
default method is matrix-free and none is a combinator.

`init(prob, x₀; kwargs...)` and `solve(prob, x₀; kwargs...)` on a `NonlinearProblem`, a
`StationarityProblem` or a `LeastSquaresProblem` call
`init(prob, x₀, default_method(prob, x₀), nothing; kwargs...)` and
`solve(prob, x₀, default_method(prob, x₀), nothing; kwargs...)`.

```jldoctest
julia> prob = NonlinearProblem((y, x, p) -> (y .= x .^ 2 .- 4; y));

julia> default_method(prob, ones(3)) === Newton()
true

julia> x, status = solve(prob, ones(3));

julia> status.code === SUCCESS && x ≈ [2.0, 2.0, 2.0]
true
```
"""
default_method(::NonlinearProblem, x₀) = Newton()
function default_method(
        ::NonlinearProblem{<:Any, <:Any, <:SparseMatrixCSC},
        ::Vector{<:UMFPACKEltype})
    return Newton(; linsolve = UMFPACKFactorization())
end
default_method(::StationarityProblem, x₀) = Newton()
default_method(::LeastSquaresProblem, x₀) = GaussNewton()

function init(prob::Union{NonlinearProblem, LeastSquaresProblem},
        x₀::Union{Number, AbstractArray}; kwargs...)
    return init(prob, x₀, default_method(prob, x₀), nothing; kwargs...)
end

function solve(prob::Union{NonlinearProblem, LeastSquaresProblem},
        x₀::Union{Number, AbstractArray}; kwargs...)
    return solve(prob, x₀, default_method(prob, x₀), nothing; kwargs...)
end
