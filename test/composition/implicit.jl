# `solve`, `init`, `solve!` and `solve_batch!` with a combinator method refuse a
# `ForwardDiff.Dual` in `x₀` or `p`: a combinator has no implicit differentiation.

using ForwardDiff: ForwardDiff
using StaticArrays: SVector
using Test

using GeometricSolvers
using GeometricSolvers: Additive, Anderson, Multiplicative, Newton, NonlinearProblem,
                        Picard, Repeat, Switch
using GeometricSolvers: init, solve, solve!, solve_batch!

struct CombinatorDualTag end

const COMBINATORS = (
    Additive(Newton(), Picard(; damping = 0.5)),
    Multiplicative(Picard(; damping = 0.5), Newton()),
    Repeat(Newton(), 2),
    Switch(Picard(; damping = 0.5), Newton(); threshold = 0.5),
    Anderson(Picard(; damping = 0.5), 2))

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

    # init, solve! and solve_batch! refuse a dual input for every method
    @test_throws ArgumentError init(prob_a, [0.0, 0.0], m, [d(1.0), d(2.0)])
    solver = init(prob_a, [0.0, 0.0], m, [1.0, 2.0])
    @test_throws ArgumentError solve!(solver, [d(0.0), d(0.0)], [1.0, 2.0])
    @test_throws ArgumentError solve_batch!(
        [SVector(0.0, 0.0), SVector(0.0, 0.0)], prob_s, m, SVector(d(1.0), d(2.0)))

    # the primal solve with the same method still converges
    x, _ = solve(prob_s, SVector(0.0, 0.0), m, SVector(1.0, 2.0))
    @test x ≈ SVector(1.0, 2.0)
end
