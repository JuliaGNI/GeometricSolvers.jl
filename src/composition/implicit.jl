# `solve` with a `ForwardDiff.Dual` in `x₀` or `p` and a combinator method.
#
# The implicit-differentiation path of `solve` (`solve_implicit` in `src/nonlinear/implicit.jl`)
# takes the Jacobian preparation for the tangent solve from a `NewtonState` or a `PicardState`. A
# combinator state is neither, so `solve` refuses a dual input for a combinator method before the
# primal solve, as `init` and `solve!` refuse one for every method.
function solve_implicit(
        ::NonlinearProblem, x₀,
        method::Union{Additive, Multiplicative, Repeat, Switch, Anderson}, p,
        options::Union{Nothing, Options}, kwargs)
    throw(ArgumentError(
        "solve does not accept ForwardDiff.Dual inputs for the combinator $(nameof(typeof(method))); implicit differentiation through a combinator is not implemented"))
end
