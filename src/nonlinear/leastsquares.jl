"""
    LeastSquaresProblem(F)
    LeastSquaresProblem(F, J)
    LeastSquaresProblem(F, m::Integer)
    LeastSquaresProblem(F, J, m::Integer)
    LeastSquaresProblem(F, J = nothing; output_length = nothing)

A nonlinear least-squares problem ``\\min_x \\tfrac{1}{2}\\|F(x, p)\\|_2^2`` with residual `F` of
length ``m`` in ``n`` variables (``m \\ge n``) and optional Jacobian `J`.

For out-of-place problems (`SVector` or `Number`), ``m`` is inferred from `F(x, p)` and the field
`output_length` may be `nothing`. For in-place problems (`F(r, x, p)` on mutable arrays),
`output_length` is required and must satisfy `m >= 1` and `m >= length(x)`.

A `LeastSquaresProblem` holds no Jacobian prototype, so its Jacobian is dense. It has no implicit
differentiation: [`solve`](@ref) with a `ForwardDiff.Dual` in `x₀` or `p` raises an
`ArgumentError`, as [`init`](@ref) and [`solve!`](@ref) do for every problem.
"""
struct LeastSquaresProblem{F, J, M <: Union{Nothing, Int}} <: AbstractNonlinearProblem
    F::F
    J::J
    output_length::M
    function LeastSquaresProblem(
            F::FT, J::JT, m::M) where {FT, JT, M <: Union{Nothing, Int}}
        m !== nothing && m < 1 &&
            throw(ArgumentError("LeastSquaresProblem: output_length must be >= 1, got $m"))
        return new{FT, JT, M}(F, J, m)
    end
end

LeastSquaresProblem(F, J, m::Integer) = LeastSquaresProblem(F, J, Int(m))
LeastSquaresProblem(F, m::Integer) = LeastSquaresProblem(F, nothing, Int(m))

function LeastSquaresProblem(
        F, J = nothing; output_length::Union{Nothing, Integer} = nothing)
    return LeastSquaresProblem(
        F, J, output_length === nothing ? nothing : Int(output_length))
end

@inline function output_length(prob::LeastSquaresProblem, n::Int)
    prob.output_length === nothing &&
        throw(ArgumentError("LeastSquaresProblem: output_length is required for in-place arrays"))
    m = prob.output_length::Int
    m < n &&
        throw(DimensionMismatch("LeastSquaresProblem: requires m >= n (overdetermined or square), got m = $m < n = $n"))
    return m
end

@inline factor_rows(prob::LeastSquaresProblem, x) = output_length(prob, length(x))

@inline function check_static_residual_length(prob::LeastSquaresProblem, Fx, x)
    m = length(Fx)
    n = length(x)
    if prob.output_length !== nothing && prob.output_length != m
        throw(DimensionMismatch("LeastSquaresProblem: residual length $m does not match output_length $(prob.output_length)"))
    end
    m < n &&
        throw(DimensionMismatch("LeastSquaresProblem: requires m >= n (overdetermined or square), got m = $m < n = $n"))
    return nothing
end

# Implicit differentiation solves the tangent equation `J(x*) ẋ = -∂F/∂p` of a root with the
# square Jacobian at `x*`. A least-squares solution is a stationary point of `‖F‖²` instead, whose
# tangent equation `solve` does not form, so a dual `x₀` or `p` is refused before any solve runs.
function solve_implicit(::LeastSquaresProblem, x₀, method, p, options, kwargs)
    throw(ArgumentError(
        "solve does not accept ForwardDiff.Dual inputs for a LeastSquaresProblem; implicit differentiation of a LeastSquaresProblem is not implemented"))
end

"""
    GaussNewton(; linsolve = QRFactorization(), globalization = Static(),
                  jacobian = nothing, refinement = NoRefinement(),
                  regularization = 0, reuse = nothing)

Gauss–Newton method for [`LeastSquaresProblem`](@ref) and [`NonlinearProblem`](@ref). Returns a
[`Newton`](@ref) solver configured with `linsolve = QRFactorization()` by default.
"""
function GaussNewton(;
        linsolve = QRFactorization(),
        globalization = Static(),
        jacobian = nothing,
        refinement = NoRefinement(),
        regularization::Real = 0,
        reuse::Union{Nothing, Integer} = nothing)
    return Newton(; linsolve, globalization, jacobian, refinement, regularization, reuse)
end
