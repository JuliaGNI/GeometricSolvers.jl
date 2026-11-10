# The problems and inputs that a combinator method refuses.
#
# The implicit-differentiation path of `solve` (`solve_implicit` in `src/nonlinear/implicit.jl`)
# takes the Jacobian preparation for the tangent solve from a `NewtonState` or a `PicardState`. A
# combinator state is neither, so `solve` refuses a dual input for a combinator method before the
# primal solve, as `init` and `solve!` refuse one for every method.
function solve_implicit(
        ::NonlinearProblem, x₀,
        method::Union{Additive, Multiplicative, Repeat, Switch, Anderson,
            LeftPreconditioned, RAS}, p,
        options::Union{Nothing, Options}, kwargs)
    throw(ArgumentError(
        "solve does not accept ForwardDiff.Dual inputs for the combinator $(nameof(typeof(method))); implicit differentiation through a combinator is not implemented"))
end

# A combinator state holds residual and candidate buffers of the iterate's length, and a `Picard`
# member takes a square `NonlinearProblem` alone, so a `LeastSquaresProblem`, whose residual may be
# longer than the iterate, is not a `CombinatorProblem`; `init_state` refuses it.
function init_state(
        method::Union{Additive, Multiplicative, Repeat, Switch, Anderson,
            LeftPreconditioned, RAS},
        ::LeastSquaresProblem, x, p)
    throw(ArgumentError(
        "the combinator $(nameof(typeof(method))) does not take a LeastSquaresProblem; solve it with GaussNewton() or Newton() and a QRFactorization() or SVDFactorization()"))
end
