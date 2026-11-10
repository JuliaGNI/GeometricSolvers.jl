# `solve`, `init`, `solve!` and `solve_batch!` with a combinator method refuse a
# `ForwardDiff.Dual` in `x₀` or `p`: a combinator has no implicit differentiation.

using ForwardDiff: ForwardDiff
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Additive, Anderson, BlockPartition, LeftPreconditioned,
                        LeastSquaresProblem, Multiplicative, Newton, NonlinearProblem,
                        Picard, RAS, Repeat, Switch
using GeometricSolvers: init, solve, solve!, solve_batch!

struct CombinatorDualTag end

const COMBINATORS = (
    Additive(Newton(), Picard(; damping = 0.5)),
    Multiplicative(Picard(; damping = 0.5), Newton()),
    Repeat(Newton(), 2),
    Switch(Picard(; damping = 0.5), Newton(); threshold = 0.5),
    Anderson(Picard(; damping = 0.5), 2),
    LeftPreconditioned(Picard(; damping = 0.5), Newton()),
    RAS(BlockPartition(2, 1)))

@testset "solve with $(nameof(typeof(m))) refuses ForwardDiff.Dual inputs before the primal solve" for m in COMBINATORS
    calls = Ref(0)
    f_s = (x, p) -> (calls[] += 1; x .- p)
    f_n = (x, p) -> (calls[] += 1; x - p)
    f_a! = (r, x, p) -> (calls[] += 1; r .= x .- p; nothing)
    prob_s = NonlinearProblem(f_s)
    prob_n = NonlinearProblem(f_n)
    prob_a = NonlinearProblem(f_a!)

    d(v) = ForwardDiff.Dual{CombinatorDualTag}(v, 1.0)
    msg = "does not accept ForwardDiff.Dual inputs for the combinator $(nameof(typeof(m)))"

    # dual p, primal x₀
    @test_throws ArgumentError solve(prob_s, SVector(0.0, 0.0), m, SVector(d(1.0), d(2.0)))
    @test_throws msg solve(prob_s, SVector(0.0, 0.0), m, SVector(d(1.0), d(2.0)))
    @test_throws msg solve(prob_n, 0.0, m, d(1.0))
    @test_throws msg solve(prob_a, [0.0, 0.0], m, [d(1.0), d(2.0)])
    # dual x₀, primal p
    @test_throws msg solve(prob_s, SVector(d(0.0), d(0.0)), m, SVector(1.0, 2.0))
    @test_throws msg solve(prob_a, [d(0.0), d(0.0)], m, [1.0, 2.0])
    @test calls[] == 0

    # init, solve! and solve_batch! refuse a dual input for every method. A `RAS` takes a problem
    # with block residuals, so the primal solver of the `solve!` refusal is built on one
    @test_throws ArgumentError init(prob_a, [0.0, 0.0], m, [d(1.0), d(2.0)])
    prob_primal = prob_a
    if m isa RAS
        prob_primal = NonlinearProblem(f_a!;
            block_residuals = ((f, x_l, x, p) -> (f .= x_l .- p; f),))
    end
    solver = init(prob_primal, [0.0, 0.0], m, [1.0, 2.0])
    @test_throws ArgumentError solve!(solver, [d(0.0), d(0.0)], [1.0, 2.0])
    @test_throws ArgumentError solve_batch!(
        [SVector(0.0, 0.0), SVector(0.0, 0.0)], prob_s, m, SVector(d(1.0), d(2.0)))

    # the primal solve with the same method still converges. A left-preconditioned method and a
    # `RAS` take a mutable vector iterate, which this file's `SVector` problems are not; their own
    # file solves them
    if !(m isa Union{LeftPreconditioned, RAS})
        x, _ = solve(prob_s, SVector(0.0, 0.0), m, SVector(1.0, 2.0))
        @test x ≈ SVector(1.0, 2.0)
    end
end

# A `LeastSquaresProblem` is refused by name for every member of the same union, a
# left-preconditioned method included: without the union, `init_state` has no method for the pair.
@testset "init refuses a LeastSquaresProblem for every combinator" begin
    f_lsq! = (r, x, p) -> (r[1] = x[1] - 1; r[2] = x[2] - 2; nothing)
    lsq = LeastSquaresProblem(f_lsq!)
    for m in COMBINATORS
        msg = "the combinator $(nameof(typeof(m))) does not take a LeastSquaresProblem"
        @test_throws msg init(lsq, [0.0, 0.0], m)
    end
end
