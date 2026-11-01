# CommonSolve's interface of the R3 solver: `init`, `solve!` and `solve` are CommonSolve's
# functions, the `NonlinearSolver` holds its iterate and its parameters, `solve!(solver)` and
# `solve!(solver, x, p)` solve in place, and `solve!(solver, x, p)` refuses an iterate or
# parameters unlike those given to `init`, with one message per check.

using CommonSolve: CommonSolve
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using StaticArrays: MVector, SVector
using Test

using GeometricSolvers
using GeometricSolvers: DenseLU, LUFactorization, Newton, NonlinearProblem, NonlinearSolver,
                        SUCCESS, SolverStatus, init, solve, solve!

const GS = GeometricSolvers

include("../helpers/matrix.jl")

allowscalar(false)

# `y² = q`, in place, whose root from `[3, 5]` with `q = [4, 9]` is `[2, 3]`.
sq!(r, y, q) = (r .= y .* y .- q; r)

# The analytic Jacobian of `sq!`, for an `Array` iterate.
function sqj!(J, y, q)
    fill!(J, zero(eltype(J)))
    for i in axes(J, 1)
        J[i, i] = 2 * y[i]
    end
    return J
end

# The bytes of the second of two `solve!(solver)` calls from `x₀`.
function solve!_allocations(solver::S, x₀::X) where {S, X}
    copyto!(solver.x, x₀)
    solve!(solver)
    copyto!(solver.x, x₀)
    return @allocated solve!(solver)
end

# The message of the `ArgumentError` that `f()` raises.
function argument_error_message(f)
    try
        f()
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return "no exception"
end

@testset "the names are CommonSolve's" begin
    @test GS.init === CommonSolve.init
    @test GS.solve === CommonSolve.solve
    @test GS.solve! === CommonSolve.solve!
    for f in (GS.init, GS.solve, GS.solve!)
        @test parentmodule(f) === CommonSolve
        @test any(m -> m.module === GS, methods(f))
    end
    for name in (:init, :solve, :solve!)
        @test Base.isexported(GS, name)
    end
end

@testset "the factor builder and the R2 forms are methods of CommonSolve's functions" begin
    prob = NonlinearProblem(sq!, sqj!)
    @test CommonSolve.init(LUFactorization(), prob, [1.0, 2.0], [4.0, 9.0]) isa DenseLU
    pr = NonlinearProblem((x, p) -> x .* x .- 2)
    X = [SVector(1.0), SVector(2.0)]
    counts = CommonSolve.solve!(X, pr, Newton())
    @test counts.SUCCESS == 2
    @test all(x -> isapprox(x[1], sqrt(2.0); rtol = 1e-7), X)
    S = [SolverStatus{Float64}(SUCCESS, Int32(0), 0.0, 0.0, Int32(0), false) for _ in 1:2]
    @test CommonSolve.solve!([SVector(1.0), SVector(2.0)], S, pr, Newton()) === S
    @test all(s -> s.code == SUCCESS && s.iterations > 0, S)
end

@testset "init holds x and p, and solve! solves in place, $AT{$T}" for AT in ARRAY_BACKENDS,
    T in ELTYPES

    R = real(T)
    tol = 10 * sqrt(eps(R))
    prob = NonlinearProblem(sq!)
    x = AT(T[3, 5])
    p = AT(T[4, 9])

    # `init` aliases `x` and `p` and does not mutate `x`
    solver = init(prob, x, Newton(), p)
    @test solver isa NonlinearSolver{T}
    @test solver.x === x
    @test solver.p === p
    @test Array(x) == T[3, 5]

    # `solve!(solver)` writes the solution into `solver.x`, the caller's `x`
    st₁ = solve!(solver)
    @test st₁ isa SolverStatus{R}
    @test st₁.code == SUCCESS
    @test st₁.iterations > 0
    @test solver.x === x
    @test isapprox(Array(x), T[2, 3]; rtol = tol)

    # a second `solve!(solver)` starts from the solution, so it takes fewer iterations
    st₂ = solve!(solver)
    @test st₂.code == SUCCESS
    @test st₂.iterations < st₁.iterations
    @test isapprox(Array(x), T[2, 3]; rtol = tol)

    # the caller's change of `x` is the next start
    copyto!(x, AT(T[3, 5]))
    st₃ = solve!(solver)
    @test st₃.code == SUCCESS
    @test st₃.iterations > st₂.iterations
    @test isapprox(Array(x), T[2, 3]; rtol = tol)

    # `solve!(solver, x₂, p₂)` solves from `x₂` with `p₂` and keeps both
    x₂ = AT(T[5, 6])
    p₂ = AT(T[16, 25])
    st₄ = solve!(solver, x₂, p₂)
    @test st₄.code == SUCCESS
    @test isapprox(Array(x₂), T[4, 5]; rtol = tol)
    @test solver.x === x₂
    @test solver.p === p₂
    @test Array(x) ≈ T[2, 3] rtol = tol

    # `solve!(solver, x₃)` solves with the parameters that the solver holds
    x₃ = AT(T[1, 1])
    st₅ = solve!(solver, x₃)
    @test st₅.code == SUCCESS
    @test isapprox(Array(x₃), T[4, 5]; rtol = tol)
    @test solver.x === x₃
    @test solver.p === p₂

    # the refusals, each asserted by its whole message; a refused call changes nothing
    T_other = only(filter(S -> S !== T && (S <: Complex) == (T <: Complex), ELTYPES))
    @test argument_error_message(() -> solve!(solver, AT(T[1, 2, 3]))) ==
          "`solve!` needs an iterate like the template given to `init`, of length 2 and " *
          "element type $T; got length 3 and element type $T"
    @test argument_error_message(() -> solve!(solver, AT(T_other[1, 2]))) ==
          "`solve!` needs an iterate like the template given to `init`, of length 2 and " *
          "element type $T; got length 2 and element type $T_other"
    # an iterate of the right element type and length and another array type
    x_type = AT === Array ? view(T[1, 2, 3], 1:2) : T[1, 2]
    @test argument_error_message(() -> solve!(solver, x_type, p₂)) ==
          "`solve!` needs an iterate of type $(typeof(x)), like the one given to `init`; " *
          "got type $(typeof(x_type))"
    # parameters of another type with an iterate of the right type
    p_other = AT(T_other[16, 25])
    @test argument_error_message(() -> solve!(solver, x₃, p_other)) ==
          "`solve!` needs parameters of type $(typeof(p)), like those given to `init`; " *
          "got type $(typeof(p_other))"
    p_tuple = (R(16), R(25))
    @test argument_error_message(() -> solve!(solver, x₃, p_tuple)) ==
          "`solve!` needs parameters of type $(typeof(p)), like those given to `init`; " *
          "got type $(typeof(p_tuple))"
    @test solver.x === x₃
    @test solver.p === p₂
    @test isapprox(Array(x₃), T[4, 5]; rtol = tol)

    # the R3 call form with the iterate first is no method
    iterate_first = (copy(x₃), solver, p₂)
    @test_throws MethodError solve!(iterate_first...)
end

@testset "init with p = nothing, $T" for T in ELTYPES
    prob = NonlinearProblem((r, y, q) -> (r .= y .* y .- 4; r))
    x = T[3, 5]
    solver = init(prob, x, Newton())
    @test solver.p === nothing
    @test solve!(solver).code == SUCCESS
    @test isapprox(x, T[2, 2]; rtol = 10 * sqrt(eps(real(T))))
    @test argument_error_message(() -> solve!(solver, T[3, 5], (a = 1,))) ==
          "`solve!` needs parameters of type Nothing, like those given to `init`; " *
          "got type $(typeof((a = 1,)))"
    @test solver.x === x
    @test solver.p === nothing
    x₂ = T[3, 5]
    @test solve!(solver, x₂, nothing).code == SUCCESS
    @test solver.x === x₂
    @test isapprox(x₂, T[2, 2]; rtol = 10 * sqrt(eps(real(T))))
end

@testset "init refuses a StaticArray, $T" for T in ELTYPES
    prob = NonlinearProblem(sq!)
    msg = "`init` builds a solver for an array iterate; solve a `StaticArray` iterate with " *
          "`solve`"
    @test argument_error_message(() -> init(prob, SVector{2, T}(1, 2), Newton(), T[4, 9])) ==
          msg
    @test argument_error_message(() -> init(prob, MVector{2, T}(1, 2), Newton(), T[4, 9])) ==
          msg
end

@testset "solve!(solver) allocates nothing on an Array, $T" for T in ELTYPES
    prob = NonlinearProblem(sq!, sqj!)
    solver = init(prob, T[3, 5], Newton(), T[4, 9])
    @test solve!_allocations(solver, T[3, 5]) == 0
    @test isapprox(solver.x, T[2, 3]; rtol = 10 * sqrt(eps(real(T))))
    # control: the helper sees an allocation
    @test (copy(T[3, 5]); @allocated copy(T[3, 5])) > 0
end

@testset "solve returns (x, status) and copies an array x₀, $AT{$T}" for AT in ARRAY_BACKENDS,
    T in ELTYPES

    prob = NonlinearProblem(sq!)
    x₀ = AT(T[3, 5])
    p = AT(T[4, 9])
    x, st = solve(prob, x₀, Newton(), p)
    @test st.code == SUCCESS
    @test x !== x₀
    @test Array(x₀) == T[3, 5]
    @test isapprox(Array(x), T[2, 3]; rtol = 10 * sqrt(eps(real(T))))
    xc, stc = CommonSolve.solve(prob, x₀, Newton(), p)
    @test Array(xc) == Array(x)
    @test stc == st
    @test Array(x₀) == T[3, 5]

    xs, sts = solve(NonlinearProblem((y, q) -> y .* y .- q), SVector{2, T}(3, 5), Newton(),
        SVector{2, T}(4, 9))
    @test sts.code == SUCCESS
    @test xs isa SVector{2, T}
    @test isapprox(xs, SVector{2, T}(2, 3); rtol = 10 * sqrt(eps(real(T))))
end
