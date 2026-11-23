# The default method of a problem and an iterate, `default_method(prob, x₀)`: its rule for each
# array type and problem type, the standard rows on which it is `Newton()`, and the `init` and
# `solve` calls with no method argument, which take it with `p = nothing`.

using GeometricSolvers
using GeometricSolvers: UMFPACKLU, default_method
using JLArrays: JLArray
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals, spdiagm
using StaticArrays: SVector
using Test

include("../helpers/standard_cells.jl")

# A residual that refuses any parameter but `nothing`, so a call that forwards another `p` fails.
square_minus_four!(y, x, p) = (p === nothing || error("p = $p"); y .= x .^ 2 .- 4; y)
square_minus_four(x, p) = (p === nothing || error("p = $p"); x .^ 2 .- 4)

# The sparse system `A x + x³/10 = b` on `n` unknowns, `A` tridiagonal with `4` on the diagonal
# and `-1` beside it, `b` such that the root is `ones(T, n)`, with its analytic Jacobian
# `A + 3 diag(x²)/10` written into the nonzeros of the prototype `A`.
function sparse_cubic_problem(::Type{T}, n::Int = 6) where {T}
    A = spdiagm(-1 => fill(-one(T), n - 1), 0 => fill(T(4), n), 1 => fill(-one(T), n - 1))
    b = A * ones(T, n) .+ ones(T, n) ./ 10
    F!(y, x, p) = (y .= A * x .+ x .^ 3 ./ 10 .- b; y)
    function J!(J::SparseMatrixCSC, x, p)
        copyto!(nonzeros(J), nonzeros(A))
        for j in 1:n, k in nzrange(J, j)

            rowvals(J)[k] == j && (nonzeros(J)[k] += 3 * x[j]^2 / 10)
        end
        return J
    end
    return NonlinearProblem(F!, J!; jacobian_prototype = copy(A))
end

@testset "default_method: a dense iterate takes Newton(): T = $T" for T in ELTYPES
    prob_ip = NonlinearProblem(square_minus_four!)
    prob_oop = NonlinearProblem(square_minus_four)
    @testset "$A" for A in ARRAY_BACKENDS
        x₀ = A(ones(T, 3))
        @test (@inferred default_method(prob_ip, x₀)) === Newton()
    end
    @test (@inferred default_method(prob_oop, SVector(one(T), one(T)))) === Newton()
    @test (@inferred default_method(prob_oop, one(T))) === Newton()
    # A dense prototype on a `Vector` is no sparse prototype.
    prob_dense = NonlinearProblem(square_minus_four!; jacobian_prototype = zeros(T, 3, 3))
    @test (@inferred default_method(prob_dense, ones(T, 3))) === Newton()
end

@testset "default_method: a StationarityProblem takes Newton(): T = $T" for T in ELTYPES
    prob = StationarityProblem((x, p) -> sum(abs2, x))
    @test (@inferred default_method(prob, SVector(one(T), one(T)))) === Newton()
    @test (@inferred default_method(prob, ones(T, 2))) === Newton()
end

@testset "default_method: a LeastSquaresProblem takes GaussNewton(): T = $T" for T in ELTYPES
    # The 3 × 2 system `(x₁ - 1, x₂ - 2, x₁ + x₂ - 3)`.
    prob = LeastSquaresProblem((x, p) -> SVector(x[1] - 1, x[2] - 2, x[1] + x[2] - 3))
    @test (@inferred default_method(prob, SVector(zero(T), zero(T)))) === GaussNewton()
    prob_ip = LeastSquaresProblem(
        (y, x, p) -> (y .= (x[1] - 1, x[2] - 2, x[1] + x[2] - 3); y), 3)
    @test (@inferred default_method(prob_ip, zeros(T, 2))) === GaussNewton()
    @test GaussNewton() !== Newton()
end

@testset "default_method: a SparseMatrixCSC prototype: T = $T" for T in ELTYPES
    prob = sparse_cubic_problem(T)
    m = @inferred default_method(prob, ones(T, 6))
    if T === Float64 || T === ComplexF64
        @test m === Newton(; linsolve = UMFPACKFactorization())
    else
        @test m === Newton()
    end
    # On a `JLArray` iterate the prototype gives no UMFPACK rule.
    @test (@inferred default_method(prob, JLArray(ones(T, 6)))) === Newton()
end

@testset "Standard rows: default_method is Newton(), the method of Newton Static LU" begin
    @test Dict(STANDARD_COLUMNS)["Newton Static LU"] === Newton()
    @testset "$T" for T in REAL_ELTYPES
        rows = 0
        for sp in STANDARD_PROBLEMS, (_, x₀) in standard_starts(sp, T)

            @test default_method(sp.prob, x₀) === Newton()
            rows += 1
        end
        @test rows == 67
    end
end

@testset "solve and init with no method: T = $T" for T in REAL_ELTYPES
    prob_ip = NonlinearProblem(square_minus_four!)
    prob_oop = NonlinearProblem(square_minus_four)
    root = fill(T(2), 3)
    x₀ = ones(T, 3)

    x, status = solve(prob_ip, x₀)
    @test status.code === SUCCESS
    @test x ≈ root
    @test (x, status) == solve(prob_ip, x₀, Newton(), nothing)

    # A keyword reaches the solve.
    x1, status1 = solve(prob_ip, x₀; max_iterations = 1)
    @test status1.code === MAXITERS
    @test (x1, status1) == solve(prob_ip, x₀, Newton(), nothing; max_iterations = 1)

    solver = init(prob_ip, x₀)
    @test solver.p === nothing
    @test typeof(solver) == typeof(init(prob_ip, x₀, Newton(), nothing))
    @test solve!(solver) == solve(prob_ip, x₀, Newton(), nothing)
    @test solve!(init(prob_ip, x₀; max_iterations = 1))[2].code === MAXITERS

    xs, ss = solve(prob_oop, SVector(one(T), one(T)))
    @test ss.code === SUCCESS && xs ≈ SVector(T(2), T(2))
    xn, sn = solve(prob_oop, one(T))
    @test sn.code === SUCCESS && xn ≈ T(2)

    prob_ls = LeastSquaresProblem(
        (y, x, p) -> (y .= (x[1] - 1, x[2] - 2, x[1] + x[2] - 3); y), 3)
    xl, sl = solve(prob_ls, zeros(T, 2))
    @test sl.code === SUCCESS && xl ≈ T[1, 2]
    @test (xl, sl) == solve(prob_ls, zeros(T, 2), GaussNewton(), nothing)
    @test typeof(init(prob_ls, zeros(T, 2))) ==
          typeof(init(prob_ls, zeros(T, 2), GaussNewton(), nothing))
end

@testset "solve and init with no method on a sparse prototype: T = $T" for T in (Float64, ComplexF64)
    prob = sparse_cubic_problem(T)
    x, status = solve(prob, zeros(T, 6))
    @test status.code === SUCCESS
    @test x ≈ ones(T, 6)
    @test init(prob, zeros(T, 6)).state.fac isa UMFPACKLU{T}
end

@testset "solve with no method on a sparse Float64 prototype without Sparspak" begin
    sub_code = """
    using GeometricSolvers, SparseArrays
    Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt) === nothing ||
        error("GeometricSolversSparspakExt is loaded")
    n = 6
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
    b = A * ones(n) .+ ones(n) ./ 10
    F!(y, x, p) = (y .= A * x .+ x .^ 3 ./ 10 .- b; y)
    function J!(J, x, p)
        copyto!(nonzeros(J), nonzeros(A))
        for j in 1:n, k in nzrange(J, j)
            rowvals(J)[k] == j && (nonzeros(J)[k] += 3 * x[j]^2 / 10)
        end
        return J
    end
    prob = NonlinearProblem(F!, J!; jacobian_prototype = copy(A))
    x, status = solve(prob, zeros(n))
    status.code === SUCCESS || error("status \$(status.code)")
    x ≈ ones(n) || error("x = \$x")
    """
    julia_exe = joinpath(Sys.BINDIR, "julia")
    env_clean = filter(
        p -> (p.first == "JULIA_DEPOT_PATH" || !startswith(p.first, "JULIA_")) &&
             !startswith(p.first, "GIT_"),
        copy(ENV)
    )
    child = run(ignorestatus(setenv(
        `$julia_exe --startup-file=no --compiled-modules=existing --project=$(Base.active_project()) -e $sub_code`,
        env_clean
    )))
    @test success(child)
end
