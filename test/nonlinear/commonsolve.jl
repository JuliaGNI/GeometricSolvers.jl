# CommonSolve's interface of the nonlinear solver: `init`, `solve!` and `solve` are CommonSolve's
# functions with methods on the package's own types only, `solve!(init(args...))` returns what
# `solve(args...)` returns, the `NonlinearSolver` holds its iterate and its parameters, and
# `solve!(solver, x, p)` refuses an iterate or parameters unlike those of `init`, with one
# message per check.

using CommonSolve: CommonSolve
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using StaticArrays: MVector, SVector
using Test

using GeometricSolvers
using GeometricSolvers: DenseLU, LUFactorization, Newton, NonlinearProblem, NonlinearSolver,
                        SUCCESS, SolverStatus, dense_factor, init, solve, solve!,
                        solve_batch!

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

@testset "CommonSolve's functions dispatch their first argument on the package's types" begin
    # the first positional argument of every method that the package adds
    for (f, owned) in ((CommonSolve.init, NonlinearProblem),
        (CommonSolve.solve, NonlinearProblem), (CommonSolve.solve!, NonlinearSolver))
        ours = filter(m -> m.module === GS, collect(methods(f)))
        @test !isempty(ours)
        for m in ours
            first_arg = Base.unwrap_unionall(Base.unwrap_unionall(m.sig).parameters[2])
            @test first_arg.name === Base.unwrap_unionall(owned).name
        end
    end
end

@testset "the factor builder and the R2 forms are the package's own functions, $T" for T in ELTYPES
    R = real(T)
    prob = NonlinearProblem(sq!, sqj!)
    @test dense_factor(LUFactorization(), prob, T[1, 2], T[4, 9]) isa DenseLU
    @test parentmodule(dense_factor) === GS
    @test parentmodule(solve_batch!) === GS
    @test Base.isexported(GS, :solve_batch!)
    pr = NonlinearProblem((x, p) -> x .* x .- 2)
    X = [SVector(T(1)), SVector(T(2))]
    counts = solve_batch!(X, pr, Newton())
    @test counts.SUCCESS == 2
    @test all(x -> isapprox(x[1], sqrt(R(2)); rtol = 10 * sqrt(eps(R))), X)
    S = [SolverStatus{R}(SUCCESS, Int32(0), zero(R), zero(R), Int32(0), false) for _ in 1:2]
    @test solve_batch!([SVector(T(1)), SVector(T(2))], S, pr, Newton()) === S
    @test all(s -> s.code == SUCCESS && s.iterations > 0, S)
    # neither reaches CommonSolve's functions
    @test_throws MethodError CommonSolve.solve!([SVector(T(1))], pr, Newton())
    @test_throws MethodError CommonSolve.init(LUFactorization(), prob, T[1, 2], T[4, 9])
end

@testset "init holds a copy of x and p itself, and solve! solves into solver.x, $AT{$T}" for AT in ARRAY_BACKENDS,
    T in ELTYPES

    R = real(T)
    tol = 10 * sqrt(eps(R))
    prob = NonlinearProblem(sq!)
    x = AT(T[3, 5])
    p = AT(T[4, 9])

    # `init` copies `x`, holds `p` itself, and does not mutate `x`
    solver = init(prob, x, Newton(), p)
    @test solver isa NonlinearSolver{T}
    @test solver.x !== x
    @test solver.x isa typeof(x)
    @test Array(solver.x) == T[3, 5]
    @test solver.p === p
    @test Array(x) == T[3, 5]

    # `solve!(solver)` writes the solution into `solver.x` and returns it, and leaves `x`
    x_held = solver.x
    x₁, st₁ = solve!(solver)
    @test x₁ === x_held === solver.x
    @test st₁ isa SolverStatus{R}
    @test st₁.code == SUCCESS
    @test st₁.iterations > 0
    @test isapprox(Array(x₁), T[2, 3]; rtol = tol)
    @test Array(x) == T[3, 5]

    # a second `solve!(solver)` starts from the solution, so it takes fewer iterations
    _, st₂ = solve!(solver)
    @test st₂.code == SUCCESS
    @test st₂.iterations < st₁.iterations
    @test isapprox(Array(solver.x), T[2, 3]; rtol = tol)

    # a change of `solver.x` is the next start
    copyto!(solver.x, AT(T[3, 5]))
    _, st₃ = solve!(solver)
    @test st₃.code == SUCCESS
    @test st₃.iterations > st₂.iterations
    @test isapprox(Array(solver.x), T[2, 3]; rtol = tol)

    # `solve!(solver, x₂, p₂)` solves in place into `x₂` with `p₂` and keeps both
    x₂ = AT(T[5, 6])
    p₂ = AT(T[16, 25])
    x₂_out, st₄ = solve!(solver, x₂, p₂)
    @test x₂_out === x₂
    @test st₄.code == SUCCESS
    @test isapprox(Array(x₂), T[4, 5]; rtol = tol)
    @test solver.x === x₂
    @test solver.p === p₂
    @test Array(x_held) ≈ T[2, 3] rtol = tol

    # `solve!(solver, x₃)` solves with the parameters that the solver holds
    x₃ = AT(T[1, 1])
    _, st₅ = solve!(solver, x₃)
    @test st₅.code == SUCCESS
    @test isapprox(Array(x₃), T[4, 5]; rtol = tol)
    @test solver.x === x₃
    @test solver.p === p₂

    # the refusals, each asserted by its whole message; a refused call changes nothing. Each
    # refused iterate is read back, and the refused parameters differ from `p₂` in value, so a
    # solve before a check moves an iterate.
    T_other = only(filter(S -> S !== T && (S <: Complex) == (T <: Complex), ELTYPES))
    template_error(m, S) = ArgumentError(
        "`solve!` needs an iterate like the template given to `init`, of length 2 and " *
        "element type $T; got length $m and element type $S")
    iterate_error(y) = ArgumentError(
        "`solve!` needs an iterate of type $(typeof(x)), like the one given to `init`; " *
        "got type $(typeof(y))")
    parameter_error(q) = ArgumentError(
        "`solve!` needs parameters of type $(typeof(p)), like those given to `init`; " *
        "got type $(typeof(q))")
    x_long = AT(T[1, 2, 3])
    @test_throws template_error(3, T) solve!(solver, x_long)
    @test Array(x_long) == T[1, 2, 3]
    x_eltype = AT(T_other[1, 2])
    @test_throws template_error(2, T_other) solve!(solver, x_eltype)
    @test Array(x_eltype) == T_other[1, 2]
    # an iterate of the right element type and length and another array type
    x_parent = T[1, 2, 3]
    x_type = AT === Array ? view(x_parent, 1:2) : T[1, 2]
    @test_throws iterate_error(x_type) solve!(solver, x_type, p₂)
    @test x_parent == T[1, 2, 3]
    @test x_type == T[1, 2]
    # parameters of another type with an iterate of the right type
    p_other = AT(T_other[4, 9])
    @test_throws parameter_error(p_other) solve!(solver, x₃, p_other)
    p_tuple = (R(4), R(9))
    @test_throws parameter_error(p_tuple) solve!(solver, x₃, p_tuple)
    # a call that fails two checks raises the error of the first
    @test_throws template_error(3, T) solve!(solver, x_long, p_other)
    @test_throws iterate_error(x_type) solve!(solver, x_type, p_other)
    @test Array(x_long) == T[1, 2, 3]
    @test x_parent == T[1, 2, 3]
    @test x_type == T[1, 2]
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
    @test solve!(solver)[2].code == SUCCESS
    @test isapprox(solver.x, T[2, 2]; rtol = 10 * sqrt(eps(real(T))))
    @test x == T[3, 5]
    x_held = solver.x
    x_refused = T[3, 5]
    @test_throws ArgumentError(
        "`solve!` needs parameters of type Nothing, like those given to `init`; " *
        "got type $(typeof((a = 1,)))") solve!(solver, x_refused, (a = 1,))
    @test x_refused == T[3, 5]
    @test solver.x === x_held
    @test solver.p === nothing
    x₂ = T[3, 5]
    @test solve!(solver, x₂, nothing)[2].code == SUCCESS
    @test solver.x === x₂
    @test isapprox(x₂, T[2, 2]; rtol = 10 * sqrt(eps(real(T))))
end

# The values of an iterate, comparable across `Number`, static and device arrays.
iterate_values(x::Number) = [x]
iterate_values(x::AbstractArray) = Array(x)

@testset "solve!(init(args...)) returns what solve(args...) returns, $T" for T in ELTYPES
    sqv(y, q) = y .* y .- q
    cases = (
        ((NonlinearProblem(sq!), AT(T[3, 5]), AT(T[4, 9])) for AT in ARRAY_BACKENDS)...,
        (NonlinearProblem(sqv), SVector{2, T}(3, 5), SVector{2, T}(4, 9)),
        (NonlinearProblem(sqv), MVector{2, T}(3, 5), SVector{2, T}(4, 9)),
        (NonlinearProblem(sqv), T(3), T(4)))
    for (prob, x₀, p) in cases, kwargs in ((;), (; max_iterations = 1))

        x₀_values = iterate_values(x₀)
        x, st = solve(prob, x₀, Newton(), p; kwargs...)
        solver = init(prob, x₀, Newton(), p; kwargs...)
        xi, sti = solve!(solver)
        @test typeof(xi) === typeof(x)
        @test xi === solver.x
        @test iterate_values(xi) == iterate_values(x)
        @test sti == st
        @test st.code == (isempty(kwargs) ? SUCCESS : GeometricSolvers.MAXITERS)
        @test iterate_values(x₀) == x₀_values
    end
end

@testset "a value iterate: solve!(solver, x, p) stores the solution in solver.x, $T" for T in ELTYPES
    tol = 10 * sqrt(eps(real(T)))
    prob = NonlinearProblem((y, q) -> y .* y .- q)
    solver = init(prob, SVector{2, T}(3, 5), Newton(), SVector{2, T}(4, 9))
    x₁, st₁ = solve!(solver)
    @test st₁.code == SUCCESS
    @test solver.x === x₁
    @test isapprox(x₁, SVector{2, T}(2, 3); rtol = tol)
    x₂, st₂ = solve!(solver, SVector{2, T}(5, 6), SVector{2, T}(16, 25))
    @test st₂.code == SUCCESS
    @test solver.x === x₂
    @test isapprox(x₂, SVector{2, T}(4, 5); rtol = tol)
    @test_throws ArgumentError solve!(solver, SVector{3, T}(5, 6, 7))
    @test solver.x === x₂
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
