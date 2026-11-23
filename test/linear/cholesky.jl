using GeometricSolvers
using GeometricSolvers: DenseCholesky, StaticCholesky, dense_factor, factor_eltype
using GeometricSolvers: factorize!!, init_linear, ldiv!!, promote_linsolve, returncode
using GPUArraysCore: allowscalar
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: Diagonal, LowerTriangular, cond, norm, opnorm, qr, transpose
using Random: Xoshiro, randn
using StaticArrays: SMatrix, SVector
using Test

include("../helpers/matrix.jl")

# On a Julia it does not support, JET loads empty stubs that throw. JET 0.12 exports
# `JET_AVAILABLE` to say so; JET 0.9 and 0.10 name the same flag `JET_LOADABLE`.
const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

# The element types of the factorisation: the BLAS types of `ELTYPES`, which take LAPACK `potrf`
# on the host, and `BigFloat` and `Complex{BigFloat}`, which take the generic loop on every path.
const CHOLESKY_ELTYPES = (ELTYPES..., BigFloat, Complex{BigFloat})

# The bytes of a call, after one warm call of the same function through the barrier.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
function cholesky_factor_allocations(M::A) where {A}
    (factorize!!(nothing, CholeskyFactorization(), M);
        @allocated factorize!!(nothing, CholeskyFactorization(), M))
end
function cholesky_solve_allocations(fac::F, b::B) where {F, B}
    (ldiv!!(nothing, fac, b); @allocated ldiv!!(nothing, fac, b))
end
function dense_factor_allocations(fac::F, A::M) where {F, M}
    (factorize!!(fac, CholeskyFactorization(), A);
        @allocated factorize!!(fac, CholeskyFactorization(), A))
end
function dense_solve_allocations(d::D, fac::F, b::B) where {D, F, B}
    (ldiv!!(d, fac, b); @allocated ldiv!!(d, fac, b))
end

# A seeded Hermitian positive definite matrix `Q Λ Qᴴ` of element type `T`, with the eigenvalues
# `Λ` spaced logarithmically from 1 to `κ`, so that its 2-norm condition number is `κ`. The draws
# are `Float64` (or `ComplexF64`) and are converted to `T`.
function randhpd(::Type{T}, n::Int, κ::Real, seed::Int) where {T}
    rng = Xoshiro(seed)
    G = T <: Real ? T.(randn(rng, n, n)) : T.(randn(rng, ComplexF64, n, n))
    Q = Matrix(qr(G).Q)
    R = real(T)
    λ = n == 1 ? [R(κ)] : [exp10(R(i - 1) / R(n - 1) * log10(R(κ))) for i in 1:n]
    A = Q * Diagonal(T.(λ)) * Q'
    return (A + A') / 2
end

# The factorisation precision `TF` that the mixed rows take for the working type `T`: `Float32`
# below a `Float64` working type, and `Float64` for the others (above `Float32`, below `BigFloat`).
mixed_precision(::Type{T}) where {T} = real(T) === Float64 ? Float32 : Float64

# `‖x - x_ref‖∞ ≤ 8 n ε κ∞(A) ‖x_ref‖∞`, with `ε` the unit round-off of the coarser of the working
# and the factorisation precision: the forward-error bound of a backward-stable solve, for both
# solutions.
function agrees_with_backslash(A, x, b, ε)
    n = size(A, 1)
    x_ref = A \ b
    return norm(x - x_ref, Inf) <= 8 * n * ε * cond(A, Inf) * norm(x_ref, Inf)
end

# The static and the dense host factor of `A` for `method`, each with its solution for `b` as a
# host `Vector`.
function cholesky_paths(method, A::Matrix{T}, b::Vector{T}) where {T}
    n = size(A, 1)
    fs = factorize!!(nothing, method, SMatrix{n, n, T, n * n}(A))
    xs = Vector(ldiv!!(nothing, fs, SVector{n, T}(b)))
    fd = factorize!!(dense_factor(method, nothing, b, nothing), method, A)
    xd = similar(b)
    ldiv!!(xd, fd, b)
    return ((fs, xs), (fd, xd))
end

@testset "CholeskyFactorization method and factor types" begin
    @test CholeskyFactorization() === CholeskyFactorization{Nothing}()
    @test CholeskyFactorization(Float32) === CholeskyFactorization{Float32}()
    @test CholeskyFactorization(TF32()) === CholeskyFactorization{TF32}()
    @test isbits(CholeskyFactorization())
    @test isbits(CholeskyFactorization(Float32))
    for T in ELTYPES
        @test factor_eltype(CholeskyFactorization(), T) === T
        @test factor_eltype(CholeskyFactorization(Float32), T) ===
              (T <: Real ? Float32 : ComplexF32)
        @test promote_linsolve(CholeskyFactorization(Float32), T) ===
              CholeskyFactorization(real(T))
    end
    A2 = SMatrix{2, 2, Float64}(2.0, 1.0, 1.0, 2.0)
    # a precision marker selects no static method
    @test_throws r"^MethodError: no method matching factorize!!\(::Nothing, ::(GeometricSolvers\.)?CholeskyFactorization\{(GeometricSolvers\.)?TF32\}" factorize!!(
        nothing, CholeskyFactorization(TF32()), A2)
end

@testset "Hermitian positive definite systems agree with A \\ b ($T)" for T in CHOLESKY_ELTYPES
    R = real(T)
    TF = mixed_precision(T)
    for (method, ε) in ((CholeskyFactorization(), eps(R)),
        (CholeskyFactorization(TF), max(eps(R), eps(TF))))
        # the static and the dense factor at n = 1, 2, 5, and the dense one at n = 16 below
        for n in (1, 2, 5)
            A = randhpd(T, n, 1e3, 11 + n)
            b = T.(randn(Xoshiro(23 + n), n))
            for (fac, x) in cholesky_paths(method, A, b)
                @test returncode(fac) == SUCCESS
                @test agrees_with_backslash(A, x, b, ε)
            end
        end
        A = randhpd(T, 16, 1e3, 31)
        b = T.(randn(Xoshiro(37), 16))
        fd = factorize!!(dense_factor(method, nothing, b, nothing), method, A)
        @test fd isa DenseCholesky
        @test returncode(fd) == SUCCESS
        x = ldiv!!(similar(b), fd, b)
        @test agrees_with_backslash(A, x, b, ε)
        # the factor matrix is `L` with zeros above the diagonal, and `L Lᴴ` is `A`
        if method === CholeskyFactorization()
            L = fd.factors
            @test L == LowerTriangular(L)
            @test norm(L * L' - A, Inf) <= 8 * 16 * eps(R) * opnorm(A, Inf)
        end
    end
end

@testset "Not positive definite: diag(1, 2, -1, 3) gives info 3 and SINGULAR ($T)" for T in CHOLESKY_ELTYPES
    A = Matrix(Diagonal(T[1, 2, -1, 3]))
    b = ones(T, 4)
    for method in (CholeskyFactorization(), CholeskyFactorization(mixed_precision(T)))
        for (fac, _) in cholesky_paths(method, A, b)
            @test fac.info == 3
            @test returncode(fac) == SINGULAR
        end
    end
    # of two leading minors that are not positive, `info` names the first
    A2 = Matrix(Diagonal(T[1, -1, -2]))
    for (fac, _) in cholesky_paths(CholeskyFactorization(), A2, ones(T, 3))
        @test fac.info == 2
    end
    # a zero pivot is not positive either, and it is the first failing minor
    Z = Matrix(Diagonal(T[0, 1]))
    for (fac, _) in cholesky_paths(CholeskyFactorization(), Z, ones(T, 2))
        @test fac.info == 1
        @test returncode(fac) == SINGULAR
    end
end

@testset "A Hermitian matrix with non-real off-diagonal entries is L Lᴴ ($T)" for T in filter(
    T -> T <: Complex, CHOLESKY_ELTYPES)
    # Hermitian and strictly diagonally dominant with a positive diagonal, so positive definite
    A = T[4 1+2im 0.5-1im; 1-2im 5 2im; 0.5+1im -2im 6]
    b = T[1, -2im, 3 + 1im]
    R = real(T)
    for (fac, x) in cholesky_paths(CholeskyFactorization(), A, b)
        @test returncode(fac) == SUCCESS
        L = Matrix(fac.factors)
        @test L == LowerTriangular(L)
        @test all(i -> iszero(imag(L[i, i])) && real(L[i, i]) > 0, 1:3)
        @test norm(L * L' - A, Inf) <= 8 * 3 * eps(R) * opnorm(A, Inf)
        # `L Lᵀ` is not `A`: the factor is no complex symmetric one
        @test norm(L * transpose(L) - A, Inf) > 1
        @test agrees_with_backslash(A, x, b, eps(R))
    end
    # the imaginary part of the diagonal is not read, as in `zpotrf`
    A_imdiag = copy(A)
    for i in 1:3
        A_imdiag[i, i] += T(0, i)
    end
    for ((f, x), (f_im, x_im)) in zip(
        cholesky_paths(CholeskyFactorization(), A, b),
        cholesky_paths(CholeskyFactorization(), A_imdiag, b))
        @test Matrix(f_im.factors) == Matrix(f.factors)
        @test x_im == x
    end
end

@testset "Only the lower triangle is read ($T)" for T in CHOLESKY_ELTYPES
    A = randhpd(T, 5, 1e2, 41)
    b = T.(randn(Xoshiro(43), 5))
    # finite entries above the diagonal that have nothing to do with `A`
    A_junk = copy(A)
    junk = T.(T <: Real ? randn(Xoshiro(47), 5, 5) : randn(Xoshiro(47), ComplexF64, 5, 5))
    for j in 1:5
        for i in 1:(j - 1)
            A_junk[i, j] = 100 * junk[i, j]
        end
    end
    for ((f, x), (f_junk, x_junk)) in zip(
        cholesky_paths(CholeskyFactorization(), A, b),
        cholesky_paths(CholeskyFactorization(), A_junk, b))
        @test returncode(f_junk) == SUCCESS
        @test Matrix(f_junk.factors) == Matrix(f.factors)
        @test x_junk == x
    end
end

@testset "Edges: n = 0, n = 1, a non-finite entry, a factorisation precision ($T)" for T in CHOLESKY_ELTYPES
    # n = 0 gives SUCCESS and the empty step on the host
    b0 = T[]
    f0 = factorize!!(dense_factor(CholeskyFactorization(), nothing, b0, nothing),
        CholeskyFactorization(), zeros(T, 0, 0))
    @test returncode(f0) == SUCCESS
    @test isempty(ldiv!!(similar(b0), f0, b0))

    # n = 1 takes the same path: the square root of the entry
    for (fac, x) in cholesky_paths(CholeskyFactorization(), fill(T(4), 1, 1), T[2])
        @test returncode(fac) == SUCCESS
        @test fac.factors[1, 1] == 2
        @test x == T[1 // 2]
    end
    for (fac, _) in cholesky_paths(CholeskyFactorization(), fill(T(-4), 1, 1), T[2])
        @test fac.info == 1
    end

    # a non-finite entry anywhere gives info = 1 before the factorisation; `potrf` itself returns
    # 0 for a `NaN` below the first column's diagonal entry
    n = 4
    A = randhpd(T, n, 10, 53)
    b = ones(T, n)
    A_nan = copy(A)
    A_nan[n, 1] = T(NaN)
    A_inf_upper = copy(A)
    A_inf_upper[1, n] = T(Inf)
    for M in (A_nan, A_inf_upper)
        for (fac, _) in cholesky_paths(CholeskyFactorization(), M, b)
            @test fac.info == 1
            @test returncode(fac) == SINGULAR
        end
    end

    # a factor in `Float32` meets the zero pivot of `1 + 1e-9` rounded to 1, where the working
    # precision does not
    if real(T) !== Float32
        A_near = T[1 1; 1 1+1e-9]
        b2 = ones(T, 2)
        for (fac, _) in cholesky_paths(CholeskyFactorization(Float32), A_near, b2)
            @test fac.info == 2
            @test returncode(fac) == SINGULAR
        end
        for (fac, _) in cholesky_paths(CholeskyFactorization(), A_near, b2)
            @test returncode(fac) == SUCCESS
        end
    end
end

@testset "No device factor: a JLArray iterate raises ArgumentError" begin
    allowscalar(false)
    x = JLArray(ones(3))
    no_device = r"^ArgumentError: CholeskyFactorization has no device factor"
    @test_throws no_device init_linear(CholeskyFactorization(), nothing, x)
    @test_throws no_device init_linear(CholeskyFactorization(), nothing, x, 5)
    F!(r, x, p) = (r .= x .^ 3 .- p; r)
    @test_throws no_device init(
        NonlinearProblem(F!), x, Newton(; linsolve = CholeskyFactorization()), 1.0)
    # the stacked LevenbergMarquardt step refuses a device iterate first, and names the method
    @test_throws r"^ArgumentError: a device array iterate takes LevenbergMarquardt\(\).*LDLTFactorization\(\), CholeskyFactorization\(\) and scaling = :diagonal is the stacked QR solve" init(
        NonlinearProblem(F!), x,
        Newton(; linsolve = CholeskyFactorization(), globalization = LevenbergMarquardt()), 1.0)
end

@testset "Newton with CholeskyFactorization on a convex StationarityProblem ($T)" for T in REAL_ELTYPES
    # `S` is strictly convex, so its Hessian `I + 3 diag(x²)` is positive definite at every
    # iterate, and its stationary point solves `x + x³ = 1` in each component. A complex state
    # fails in the default automatic differentiation (K28), so the state is real.
    S(x, p) = sum(abs2, x) / 2 + sum(x .^ 4) / 4 - p * sum(x)
    prob = StationarityProblem(S)
    x_star = T(0.6823278038280193)
    # the residual test stops at `‖F(x)‖ ≤ √eps ‖F(x₀)‖`, with `‖F(x₀)‖ < 0.7` from both starts, and
    # `F′ = 1 + 3x² ≥ 1` gives `|x - x*| ≤ |F(x)|`: a relative error of at most `√eps · 0.7 / x*`.
    # The step test, `|Δx| ≤ 2 eps |x|`, stops closer to the root.
    rtol = 2 * sqrt(eps(T))
    p = one(T)
    # a `Number` iterate takes the `1 × 1` static factor
    for x₀ in (T(0.3), SVector{2, T}(0.3, 0.6), T[0.3, 0.6])
        x, st = solve(prob, copy(x₀), Newton(; linsolve = CholeskyFactorization()), p)
        @test st.code == SUCCESS
        @test x != x₀
        @test all(xi -> isapprox(xi, x_star; rtol), x)
        xm, stm = solve(prob, copy(x₀),
            Newton(; linsolve = CholeskyFactorization(mixed_precision(T)),
                refinement = Refinement()),
            p)
        @test stm.code == SUCCESS
        @test all(xi -> isapprox(xi, x_star; rtol), xm)
    end
    # R2: the static factor in one work-item per system
    for Arr in ARRAY_BACKENDS
        X = Arr([SVector{2, T}(0.3, 0.6) for _ in 1:3])
        counts = solve_batch!(X, prob, Newton(; linsolve = CholeskyFactorization()),
            PerSystem(Arr(ones(T, 3))))
        @test counts.SUCCESS == 3
        @test all(x -> all(xi -> isapprox(xi, x_star; rtol), x), Array(X))
    end
end

@testset "A Float32 factor that is not positive definite takes the promotion of Newton ($T)" for T in (
    Float64, ComplexF64)
    # `J` is Hermitian and positive definite in the working precision and has a zero pivot in
    # `Float32`, where `1 + 1e-9` rounds to 1: `1 - |J₂₁|² = 0`. The first factorisation in
    # `Float32` is SINGULAR, and the step factorises in the working precision.
    A = T <: Real ? T[1 1; 1 1+1e-9] : T[1 im; -im 1+1e-9]
    x_star = T <: Real ? T[1, -1] : T[1 + 2im, -1 + im]
    rhs = A * x_star
    F!(r, x, p) = (r .= A * x .- rhs; r)
    x, st = solve(NonlinearProblem(F!), zeros(T, 2),
        Newton(; linsolve = CholeskyFactorization(Float32)))
    @test st.code == SUCCESS
    @test st.promoted
    @test norm(x - x_star, Inf) <= 8 * 2 * eps(Float64) * cond(A, Inf) * norm(x_star, Inf)
end

@testset "A LeastSquaresProblem takes CholeskyFactorization in the LevenbergMarquardt step only ($T)" for T in REAL_ELTYPES
    f_oop = (x, p) -> SVector{3}(x[1] - 1, x[2] - 2, x[1] + x[2] - 3)
    f_ip = (r, x, p) -> (r[1] = x[1] - 1; r[2] = x[2] - 2; r[3] = x[1] + x[2] - 3; r)
    p_s = LeastSquaresProblem(f_oop)
    p_v = LeastSquaresProblem(f_ip, 3)
    m = Newton(; linsolve = CholeskyFactorization())
    refused = r"^ArgumentError: LeastSquaresProblem does not support CholeskyFactorization"
    @test_throws refused solve(p_s, SVector{2, T}(0, 0), m)
    @test_throws refused init(p_v, zeros(T, 2), m)
    lm = Newton(; linsolve = CholeskyFactorization(), globalization = LevenbergMarquardt(T))
    for x₀ in (SVector{2, T}(0, 0), zeros(T, 2))
        x, st = solve(x₀ isa SVector ? p_s : p_v, x₀, lm)
        @test st.code == SUCCESS
        @test norm(x - T[1, 2], Inf) <= 10 * sqrt(eps(T))
    end
end

@testset "StaticCholesky: isbits, inferred, JET clean and allocation-free ($T)" for T in ELTYPES
    A = SMatrix{3, 3, T, 9}(
        T(4), T(1), T(-1) / 2, T(1), T(3), T(1) / 4, T(-1) / 2, T(1) / 4, T(2))
    b = SVector{3, T}(one(T), T(-2), T(1) / 2)
    fac = @inferred factorize!!(nothing, CholeskyFactorization(), A)
    @test fac isa StaticCholesky{3, T, 9}
    @test isbits(fac)
    x = @inferred ldiv!!(nothing, fac, b)
    @test x isa SVector{3, T}
    # `--check-bounds=yes` lets the static factors' `MMatrix` escape to the heap, so a run that
    # forces it skips these assertions
    if Base.JLOptions().check_bounds != 1
        @test cholesky_factor_allocations(A) == 0
        @test cholesky_solve_allocations(fac, b) == 0
    end
    # the control: the barrier sees an allocation where there is one
    @test allocations(collect, b) > 0
    if JET_WORKS
        @test_opt factorize!!(nothing, CholeskyFactorization(), A)
        @test_opt ldiv!!(nothing, fac, b)
    else
        @test_skip JET_WORKS  # issue #8
    end
end

@testset "DenseCholesky: the LAPACK factorisation and solve allocate nothing ($T)" for T in ELTYPES
    A = randhpd(T, 8, 1e2, 59)
    b = T.(randn(Xoshiro(61), 8))
    d = similar(b)
    fac = factorize!!(dense_factor(CholeskyFactorization(), nothing, b, nothing),
        CholeskyFactorization(), A)
    @test returncode(fac) == SUCCESS
    @test dense_factor_allocations(fac, A) == 0
    @test dense_solve_allocations(d, fac, b) == 0
    @test allocations(collect, b) > 0
end
