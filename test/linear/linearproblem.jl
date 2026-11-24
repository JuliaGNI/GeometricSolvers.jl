# The linear-solve object: `LinearProblem(A)`, `init(prob, method)`, `solve!(solver, b)` and
# `refactorize!!(solver, A)` on R1 (an `SMatrix`), the R3 host (an `Array`) and the R3 device
# kernels on the CPU (a `JLArray`), in every element type of `ELTYPES`. Square systems through the
# five factor types, the least-squares solution of an `m × n` system with `m > n` through QR and
# SVD, and the minimum-norm solution with `m < n` through the SVD of `Aᴴ` on every route and a QR
# of `Aᴴ` on R1 and the host; the rank of the SVD path; the status codes; the refusals;
# refactorisation; the factorisation precision; the scale of the system; the values, inference and
# allocations of R1 and the allocations of the R3 host; and `BigFloat`.
#
# The matrices are well conditioned, `κ ≤ 3` (`κ ≤ 9` for the Hermitian positive definite one), so
# that a backward-stable solve meets a bound of a small multiple of `eps(R)` against the solution
# in `BigFloat` of the matrix and the vector as they are rounded.

using GeometricSolvers
using GeometricSolvers: NONFINITE, SINGULAR, SUCCESS, factor_eltype
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: norm
using SparseArrays: sparse
using StaticArrays: SMatrix, SVector
using Test

include("../helpers/matrix.jl")

allowscalar(false)

const ROUTES = ("SMatrix", map(AT -> string(nameof(AT)), ARRAY_BACKENDS)...)

# A unit phase `cis(0.2 i + 0.3 j)` for a complex `T`, 1 for a real one: `D₁ A D₂` with a unitary
# diagonal `D₁` and `D₂` has the singular values of `A`.
phase(::Type{T}, i, j) where {T} = T <: Complex ? T(cis(0.2 * i + 0.3 * j)) : one(T)

# `2` on the diagonal and `0.3 sin(i + 2j)` everywhere: `κ ≤ 3` at the sizes of this file.
function wellcond(::Type{T}, m::Int, n::Int) where {T}
    return [T(real(T)((i == j ? 2.0 : 0.0) + 0.3 * sin(i + 2j))) * phase(T, i, j)
            for i in 1:m, j in 1:n]
end

# `Wᴴ W` of `W = wellcond(T, n, n)` in `BigFloat`, rounded: exactly Hermitian, with a real diagonal.
function spd(::Type{T}, n::Int) where {T}
    Wb = big.(wellcond(T, n, n))
    return T.(Wb' * Wb)
end

# The `m × length(σ)` matrix whose singular values are exactly `σ`, `m ≥ length(σ)`: column `j`
# holds `ph_j σ_j` in row `perm[j]`, with `perm = circshift(1:n, 1)` and a sign or unit phase
# `ph_j`.
function fixture(::Type{T}, σ, m::Int = length(σ)) where {T}
    n = length(σ)
    A = zeros(T, m, n)
    perm = circshift(1:n, 1)
    for j in 1:n
        A[perm[j], j] = fixture_phase(T, j) * T(real(T)(σ[j]))
    end
    return A
end
function fixture_phase(::Type{T}, j) where {T}
    T <: Complex ? T(cis(0.3 * j)) : T(isodd(j) ? 1 : -1)
end

# The right-hand side `(1, 2, …, m)` with unit phases.
rhs(::Type{T}, m::Int) where {T} = [T(i) * phase(T, i, 1) for i in 1:m]

# The solution in `BigFloat`: `A \ b` for a square `A`, the least-squares solution for `m > n`, and
# the minimum-norm solution `Aᴴ (A Aᴴ)⁻¹ b` for `m < n`.
function xref(A::AbstractMatrix, b::AbstractVector)
    Ab = big.(Matrix(A))
    bb = big.(Vector(b))
    m, n = size(Ab)
    m == n && return Ab \ bb
    m > n && return (Ab' * Ab) \ (Ab' * bb)
    return Ab' * ((Ab * Ab') \ bb)
end

relerr(x, xr) = Float64(norm(big.(Vector(x)) - xr) / norm(xr))

# The truncated pseudo-inverse solution of the fixture with the singular values `σ` at rank `r`:
# the tall fixture `F` gives `x[j] = conj(ph_j) b[perm[j]] / σ_j`, its adjoint
# `x[perm[j]] = ph_j b[j] / σ_j`.
function trunc_ref(::Type{T}, σ, b::Vector{T}, r::Int, wide::Bool, n_x::Int) where {T}
    n = length(σ)
    perm = circshift(1:n, 1)
    x = zeros(Complex{BigFloat}, n_x)
    for j in 1:r
        ph = big(fixture_phase(T, j))
        s = big(real(T)(σ[j]))
        if wide
            x[perm[j]] = ph * big(b[j]) / s
        else
            x[j] = conj(ph) * big(b[perm[j]]) / s
        end
    end
    return x
end

# A host matrix or vector on a route.
function onroute(route, A::Matrix{T}) where {T}
    route == "SMatrix" && return SMatrix{size(A)..., T}(A)
    route == "Array" && return A
    return JLArray(A)
end
function onroute(route, b::Vector{T}) where {T}
    route == "SMatrix" && return SVector{length(b), T}(b)
    route == "Array" && return b
    return JLArray(b)
end

# The solution of `A x = b` on a route as a host `Vector`, the status and the solver.
function lp_solve(route, method, A::Matrix, b::Vector)
    solver = init(LinearProblem(onroute(route, A)), method)
    x, status = solve!(solver, onroute(route, b))
    return Vector(Array(x)), status, solver
end
function lp_resolve(route, solver, b::Vector)
    x, status = solve!(solver, onroute(route, b))
    return Vector(Array(x)), status
end

isnan_all(x) = all(isnan, real.(x))

const SQUARE = (LUFactorization(), QRFactorization(), SVDFactorization(),
    LDLTFactorization(), CholeskyFactorization())
const SQUARE_ONLY = (LUFactorization(), LDLTFactorization(), CholeskyFactorization())
const SIGMA = (1.0, 1e-10, 1e-14)

function square_matrix(T, n, method)
    method isa Union{LDLTFactorization, CholeskyFactorization} ?
    spd(T, n) : wellcond(T, n, n)
end
device_cholesky(route, method) = route == "JLArray" && method isa CholeskyFactorization
function device_wide_qr(route, method, m, n)
    route == "JLArray" && method isa QRFactorization && m < n
end

@testset "the object, its status and the exports" begin
    for name in (:LinearProblem, :LinearSolver, :LinearStatus, Symbol("refactorize!!"))
        @test Base.isexported(GeometricSolvers, name)
    end
    @test fieldnames(LinearStatus) == (:code, :rank)
    @test fieldtype(LinearStatus, :code) === GeometricSolvers.ReturnCode
    @test fieldtype(LinearStatus, :rank) === Int32
    A = wellcond(Float64, 3, 3)
    prob = LinearProblem(A)
    @test prob.A === A
    solver = init(prob, LUFactorization())
    @test solver isa LinearSolver
    x, st = solve!(solver, rhs(Float64, 3))
    @test st isa LinearStatus
    @test isbits(st)
    @test x === solver.x
end

@testset "a square system: $route, $T, $(nameof(typeof(method)))" for route in ROUTES,
    T in ELTYPES, method in SQUARE
    if device_cholesky(route, method)
        @test_throws r"^ArgumentError: CholeskyFactorization has no device factor" init(
            LinearProblem(JLArray(square_matrix(T, 4, method))),
            method)
        continue
    end
    R = real(T)
    A = square_matrix(T, 4, method)
    b = rhs(T, 4)
    x, st, _ = lp_solve(route, method, A, b)
    @test eltype(x) === T
    @test st.code == SUCCESS
    @test st.rank == (method isa SVDFactorization ? 4 : -1)
    # a backward-stable solve of a system of `κ ≤ 9`
    @test relerr(x, xref(A, b)) <= 128 * eps(R)
    x1, st1, _ = lp_solve(route, method, fill(T(3), 1, 1), T[6])
    @test st1.code == SUCCESS
    @test x1 ≈ T[2]
end

@testset "m > n: $route, $T, $(nameof(typeof(method))), $m × $n" for route in ROUTES,
    T in ELTYPES, method in (QRFactorization(), SVDFactorization()),
    (m, n) in ((5, 3), (3, 1))
    R = real(T)
    A = wellcond(T, m, n)
    b = rhs(T, m)
    x, st, _ = lp_solve(route, method, A, b)
    @test length(x) == n
    @test st.code == SUCCESS
    @test st.rank == (method isa SVDFactorization ? n : -1)
    @test relerr(x, xref(A, b)) <= 64 * eps(R)
end

@testset "LU, LDLᵀ and Cholesky refuse m ≠ n: $route, $(nameof(typeof(method))), $m × $n" for route in ROUTES,
    method in SQUARE_ONLY, (m, n) in ((5, 3), (3, 5))

    @test_throws r"^ArgumentError: \w+ factorises a square matrix alone, got the size \(\d, \d\); take QRFactorization\(\) or SVDFactorization\(\)$" init(
        LinearProblem(onroute(route, wellcond(Float64, m, n))), method)
end

@testset "m < n, full rank: $route, $T, $(nameof(typeof(method))), $m × $n" for route in ROUTES,
    T in ELTYPES, method in (SVDFactorization(), QRFactorization()),
    (m, n) in ((3, 5), (1, 3))

    A = wellcond(T, m, n)
    b = rhs(T, m)
    if device_wide_qr(route, method, m, n)
        @test_throws r"^ArgumentError: QRFactorization has no minimum-norm solve of an m × n system with m < n on a device array; take SVDFactorization\(\)$" init(
            LinearProblem(JLArray(A)), method)
        # a device array behind a wrapper that is no `AbstractGPUArray` itself
        @test_throws r"^ArgumentError: QRFactorization has no minimum-norm solve of an m × n system with m < n on a device array; take SVDFactorization\(\)$" init(
            LinearProblem(JLArray(Matrix(A'))'), method)
        continue
    end
    R = real(T)
    x, st, _ = lp_solve(route, method, A, b)
    @test length(x) == n
    @test st.code == SUCCESS
    @test st.rank == (method isa SVDFactorization ? m : -1)
    # the minimum-norm solution, and the residual, of a system of `κ ≤ 3`
    @test relerr(x, xref(A, b)) <= 64 * eps(R)
    @test Float64(norm(big.(A) * big.(x) - big.(b)) / norm(big.(b))) <= 64 * eps(R)
end

@testset "m < n, a QR of a rank-deficient matrix: $route, $T" for route in ("SMatrix", "Array"),
    T in ELTYPES

    A = Matrix(fixture(T, (1.0, 1e-10, 0.0), 5)')
    x, st, _ = lp_solve(route, QRFactorization(), A, rhs(T, 3))
    @test st.code == SINGULAR
    @test st.rank == -1
    @test isnan_all(x)
end

@testset "the rank of the SVD path: $route, $T, $shape, rtol = $rtol" for route in ROUTES,
    T in ELTYPES, shape in (:square, :tall, :wide),
    (rtol, r32, r64) in ((nothing, 1, 3), (1e-12, 2, 2), (1e-9, 1, 1), (0.0, 3, 3))
    R = real(T)
    F = fixture(T, SIGMA, shape == :square ? 3 : 5)
    A = shape == :wide ? Matrix(F') : F
    b = rhs(T, size(A, 1))
    method = rtol === nothing ? SVDFactorization() : SVDFactorization(; rtol)
    x, st, _ = lp_solve(route, method, A, b)
    want = R === Float32 ? r32 : r64
    @test st.code == SUCCESS
    @test st.rank == want
    # a complex LAPACK route loses `eps / 1e-14` of the accuracy of the coordinate along
    # `σ = 1e-14`: the truncated solution is checked on the real types alone
    if T <: Real
        xr = trunc_ref(T, SIGMA, b, want, shape == :wide, size(A, 2))
        @test relerr(x, xr) <= 8 * eps(R)
    end
end

@testset "the default cutoff counts the rows of the factored matrix: $route, $R, $shape" for route in ROUTES,
    R in REAL_ELTYPES, (shape, want) in ((:square, 2), (:tall, 1), (:wide, 1))

    # `σ₂ = 2 eps(R)` lies above `√3 eps(R)` and below `√5 eps(R)`
    F = fixture(R, (1.0, 2 * eps(R), 0.0), shape == :square ? 3 : 5)
    A = shape == :wide ? Matrix(F') : F
    _, st, _ = lp_solve(route, SVDFactorization(), A, rhs(R, size(A, 1)))
    @test st.rank == want
end

@testset "the status codes: $route, $T" for route in ROUTES, T in ELTYPES

    x, st, _ = lp_solve(route, LUFactorization(), T[1 2 0; 2 4 0; 0 0 1], rhs(T, 3))
    @test st.code == SINGULAR
    @test st.rank == -1
    @test isnan_all(x)
    if route != "JLArray"
        x, st, _ = lp_solve(route, CholeskyFactorization(), T[1 2 0; 2 1 0; 0 0 1], rhs(T, 3))
        @test st.code == SINGULAR
        @test st.rank == -1
        @test isnan_all(x)
    end
    for (m, n) in ((3, 3), (3, 5), (5, 3))
        x, st, _ = lp_solve(route, SVDFactorization(), zeros(T, m, n), rhs(T, m))
        @test st.code == SUCCESS
        @test st.rank == 0
        @test iszero(x)
    end
    for method in SQUARE, bad in (NaN, Inf)

        device_cholesky(route, method) && continue
        A = square_matrix(T, 3, method)
        A[2, 1] = T(bad)
        x, st, _ = lp_solve(route, method, A, rhs(T, 3))
        @test st.code == NONFINITE
        @test st.rank == -1
        @test isnan_all(x)
    end
    A = wellcond(T, 3, 5)
    A[1, 4] = T(NaN)
    x, st, _ = lp_solve(route, SVDFactorization(), A, rhs(T, 3))
    @test st.code == NONFINITE
    @test st.rank == -1
    @test isnan_all(x)
    for method in (LUFactorization(), SVDFactorization()), bad in (NaN, Inf)

        b = rhs(T, 4)
        b[2] = T(bad)
        x, st, _ = lp_solve(route, method, wellcond(T, 4, 4), b)
        @test st.code == NONFINITE
        @test !all(isfinite, x)
    end
    x, st, _ = lp_solve(route, LUFactorization(), wellcond(T, 4, 4), zeros(T, 4))
    @test st.code == SUCCESS
    @test iszero(x)
    # a non-finite `b` gives `x` all `NaN` also where the solve itself would give a finite `x`: the
    # zero step of an SVD of rank 0
    b = rhs(T, 3)
    b[1] = T(Inf)
    x, st, _ = lp_solve(route, SVDFactorization(), zeros(T, 3, 5), b)
    @test st.code == NONFINITE
    @test isnan_all(x)
end

@testset "a zero b in a lower factor precision with m < n: $route, $method" for route in ROUTES,
    method in (SVDFactorization(Float32), QRFactorization(Float32))

    device_wide_qr(route, method, 3, 5) && continue
    x, st, _ = lp_solve(route, method, wellcond(Float64, 3, 5), zeros(3))
    @test st.code == SUCCESS
    @test iszero(x)
end

@testset "A · 2^±200 in a Float32 factor with m < n: $route, $method, k = $k" for route in ROUTES,
    method in (SVDFactorization(Float32), QRFactorization(Float32)), k in (-200, 200)

    # `A` is divided by a power of two before it is converted to `Float32`, whose range it leaves
    device_wide_qr(route, method, 3, 5) && continue
    A = wellcond(Float64, 3, 5)
    b = rhs(Float64, 3)
    x0, _, _ = lp_solve(route, method, A, b)
    x, st, _ = lp_solve(route, method, ldexp.(A, k), b)
    @test st.code == SUCCESS
    @test isapprox(x, ldexp.(x0, -k); rtol = 8 * eps(Float64))
end

@testset "the refusals of LinearProblem and init" begin
    empty_message(dims) = "LinearProblem needs a matrix of at least one row and one column, " *
                          "got the size $dims"
    @test_throws ArgumentError(empty_message((0, 3))) LinearProblem(zeros(0, 3))
    @test_throws ArgumentError(empty_message((3, 0))) LinearProblem(zeros(3, 0))
    @test_throws ArgumentError("LinearProblem takes a dense matrix, got the type SparseMatrixCSC") LinearProblem(
        sparse(wellcond(Float64, 3, 3)))
    prob = LinearProblem(wellcond(Float64, 3, 3))
    # the refusal of `init` itself, not an error of the factor that it would build
    @test_throws ArgumentError(
        "a LinearProblem is solved by a dense factorisation, got GMRES") init(prob, GMRES())
    @test_throws ArgumentError(
        "a LinearProblem is solved by a dense factorisation, got UMFPACKFactorization") init(
        prob, UMFPACKFactorization())
    for M in (LUFactorization, QRFactorization, SVDFactorization)
        @test_throws ArgumentError(
            "a precision marker has no factorisation; give the precision as a float type") init(
            prob, M(TF32()))
    end
end

@testset "the refusals of solve! and refactorize!!: $route, $m × $n" for route in ROUTES,
    (m, n) in ((3, 3), (5, 3), (3, 5))

    A = wellcond(Float64, m, n)
    solver = init(LinearProblem(onroute(route, A)), SVDFactorization())
    @test_throws DimensionMismatch(
        "the right-hand side has $(m + 1) entries, the matrix $m rows") solve!(
        solver, onroute(route, rhs(Float64, m + 1)))
    @test_throws ArgumentError(
        "the right-hand side has the element type Float32, the matrix Float64") solve!(
        solver, onroute(route, rhs(Float32, m)))
    other_b = route == "JLArray" ? rhs(Float64, m) : JLArray(rhs(Float64, m))
    @test_throws (route == "SMatrix" ?
                  r"^ArgumentError: a solver of a static matrix takes a static right-hand side, got the type JLArray$" :
                  r"^ArgumentError: the right-hand side is on the backend \w+, the solver on \w+$") solve!(
        solver, other_b)
    @test_throws DimensionMismatch(
        "the solver factorises a matrix of the size $((m, n)), got $((m + 1, n))") refactorize!!(
        solver, onroute(route, wellcond(Float64, m + 1, n)))
    @test_throws ArgumentError(
        "the solver factorises a matrix of the element type Float64, got Float32") refactorize!!(
        solver, onroute(route, wellcond(Float32, m, n)))
    other_A = route == "JLArray" ? A : JLArray(A)
    @test_throws (route == "SMatrix" ?
                  r"^ArgumentError: a solver of a static matrix takes a static matrix, got the type JLArray$" :
                  r"^ArgumentError: the matrix is on the backend \w+, the solver on \w+$") refactorize!!(
        solver, other_A)
    route == "SMatrix" &&
        @test_throws ArgumentError(
            "a solver of a static matrix takes a static right-hand side, got the type Array") solve!(
            solver, rhs(Float64, m))
    if route == "Array"
        @test_throws ArgumentError(
            "a solver of a dense or device matrix takes no static right-hand side") solve!(
            solver, SVector{m}(rhs(Float64, m)))
        @test_throws ArgumentError(
            "a solver of a dense or device matrix takes no static matrix") refactorize!!(
            solver, SMatrix{m, n}(A))
        @test_throws ArgumentError(
            "a solver of a dense or device matrix takes no sparse matrix") refactorize!!(
            solver, sparse(A))
    end
end

@testset "a b of another length for an LU solver: $route" for route in ("SMatrix", "Array")
    solver = init(LinearProblem(onroute(route, wellcond(Float64, 3, 3))), LUFactorization())
    @test_throws DimensionMismatch("the right-hand side has 4 entries, the matrix 3 rows") solve!(
        solver, onroute(route, rhs(Float64, 4)))
end

@testset "refactorize!!: $route, $T, $(nameof(typeof(method))), $m × $n" for route in ROUTES,
    T in filter(T -> real(T) === Float64, ELTYPES),
    (method, m, n) in ((LUFactorization(), 4, 4), (SVDFactorization(), 3, 5),
        (QRFactorization(), 5, 3), (QRFactorization(), 3, 5))

    device_wide_qr(route, method, m, n) && continue
    A = wellcond(T, m, n)
    A2 = copy(A)
    A2[:, 1] .*= 3
    b = rhs(T, m)
    solver = init(LinearProblem(onroute(route, A)), method)
    solver2 = refactorize!!(solver, onroute(route, A2))
    x2, st2 = lp_resolve(route, solver2, b)
    @test st2.code == SUCCESS
    # `κ(A₂) ≤ 9`
    @test relerr(x2, xref(A2, b)) <= 256 * eps(Float64)
    if route == "SMatrix"
        x1, _ = lp_resolve(route, solver, b)
        @test relerr(x1, xref(A, b)) <= 64 * eps(Float64)
    else
        @test solver2.x === solver.x
    end
    # the status follows the last matrix: a non-finite one, then a finite one again
    Abad = copy(A)
    Abad[1, 1] = T(NaN)
    s3 = refactorize!!(solver2, onroute(route, Abad))
    x3, st3 = lp_resolve(route, s3, b)
    @test st3.code == NONFINITE
    @test st3.rank == -1
    @test isnan_all(x3)
    s4 = refactorize!!(s3, onroute(route, A))
    x4, st4 = lp_resolve(route, s4, b)
    @test st4.code == SUCCESS
    @test relerr(x4, xref(A, b)) <= 64 * eps(Float64)
end

@testset "Float32 factors of a $T system: $route, $method, $m × $n" for route in ROUTES,
    T in (Float64, ComplexF64),
    (method, m, n) in ((LUFactorization(Float32), 4, 4), (QRFactorization(Float32), 4, 4),
        (SVDFactorization(Float32), 4, 4), (SVDFactorization(Float32), 5, 3),
        (SVDFactorization(Float32), 3, 5), (QRFactorization(Float32), 5, 3),
        (QRFactorization(Float32), 3, 5))
    device_wide_qr(route, method, m, n) && continue
    A = wellcond(T, m, n)
    b = rhs(T, m)
    x, st, solver = lp_solve(route, method, A, b)
    @test eltype(x) === T
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 64 * eps(Float32)
    # `b` is divided by its largest absolute entry before it is converted to `Float32`; a power of
    # two scales a complex entry exactly
    for k in (-140, 140)
        xk, stk = lp_resolve(route, solver, b .* 2.0^k)
        @test stk.code == SUCCESS
        @test isapprox(xk, x .* 2.0^k; rtol = 8 * eps(Float64))
    end
end

@testset "the scale exponent of the factor of Aᴴ with m < n: $route, $method, $A_kind" for route in ROUTES,
    method in (SVDFactorization(), SVDFactorization(Float32), QRFactorization(),
        QRFactorization(Float32)),
    A_kind in ("largest entry prevfloat(2.0)", "subnormal")

    # The factor of a dense or device array divides `Aᴴ / 2^e` by a power of two of its own, which
    # the solve adds to `e`: where the largest entry of `A / 2^e` rounds up to 2 in a `Float32`
    # factor, and where `A` is subnormal. The static factors scale `A` themselves, the QR only for a
    # factor element type other than that of `A`; their QR in the element type of `A` and their
    # `Float32` SVD lose the accuracy of a subnormal matrix here and on the square path (K109 of
    # `KNOWN_ISSUES.md`), so that matrix is left out of every static row.
    device_wide_qr(route, method, 3, 5) && continue
    route == "SMatrix" && A_kind == "subnormal" && continue
    A = wellcond(Float64, 3, 5)
    b = rhs(Float64, 3)
    if A_kind == "subnormal"
        A = ldexp.(A, -1070)
        b = ldexp.(b, -1000)
    else
        A ./= maximum(abs, A)
        A .*= prevfloat(2.0)
    end
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 64 * eps(factor_eltype(method, Float64))
end

@testset "A · 2^k and b · 2^j: $route, $R, $(nameof(typeof(method))), $m × $n" for route in ROUTES,
    R in REAL_ELTYPES,
    (method, m, n) in ((LUFactorization(), 4, 4), (QRFactorization(), 4, 4),
        (SVDFactorization(), 4, 4), (QRFactorization(), 5, 3), (SVDFactorization(), 5, 3),
        (SVDFactorization(), 3, 5), (QRFactorization(), 3, 5))

    device_wide_qr(route, method, m, n) && continue
    A = wellcond(R, m, n)
    b = rhs(R, m)
    x0, _, _ = lp_solve(route, method, A, b)
    for k in (-40, 40), j in (-40, 40)

        x, st, _ = lp_solve(route, method, ldexp.(A, k), ldexp.(b, j))
        @test st.code == SUCCESS
        @test isapprox(x, ldexp.(x0, j - k); rtol = 64 * eps(R))
    end
end

# A function barrier with concrete argument types, which calls once before it measures.
allocations(f::F, args::Vararg{Any, K}) where {F, K} = (f(args...); @allocated f(args...))

@testset "R1 values and inference, R3 host allocations: $method, $m × $n" for (method, m, n) in (
    (LUFactorization(), 4, 4), (QRFactorization(), 4, 4), (SVDFactorization(), 4, 4),
    (LDLTFactorization(), 4, 4), (CholeskyFactorization(), 4, 4),
    (QRFactorization(), 5, 3), (SVDFactorization(), 5, 3), (SVDFactorization(), 3, 5),
    (QRFactorization(), 3, 5), (SVDFactorization(; rtol = 1e-12), 3, 5))
    A = m == n ? square_matrix(Float64, m, method) : wellcond(Float64, m, n)
    SA = SMatrix{m, n}(A)
    sb = SVector{m}(rhs(Float64, m))
    prob = LinearProblem(SA)
    @test isbits(prob)
    solver = @inferred init(prob, method)
    @test isbits(solver)
    @inferred solve!(solver, sb)
    @inferred refactorize!!(solver, SA)
    hsolver = init(LinearProblem(A), method)
    hb = rhs(Float64, m)
    @inferred solve!(hsolver, hb)
    # under --check-bounds=yes a bounds check lets a local `MMatrix` of the static kernels escape
    # to the heap, so the counts of R1 are skipped where a run forces it
    if Base.JLOptions().check_bounds != 1
        @test allocations(init, prob, method) == 0
        @test allocations(solve!, solver, sb) == 0
        @test allocations(refactorize!!, solver, SA) == 0
    end
    @test allocations(solve!, hsolver, hb) == 0
    @test allocations(refactorize!!, hsolver, A) == 0
    @test allocations(collect, SVector(1.0, 2.0)) > 0
end

@testset "BigFloat: $route, $(nameof(typeof(method))), $m × $n" for (route, method, m, n) in (
    ("SMatrix", LUFactorization(), 4, 4), ("SMatrix", SVDFactorization(), 3, 5),
    ("SMatrix", QRFactorization(), 3, 5), ("SMatrix", SVDFactorization(), 5, 3),
    ("Array", LUFactorization(), 4, 4), ("Array", QRFactorization(), 5, 3),
    ("Array", QRFactorization(), 3, 5))
    A = wellcond(BigFloat, m, n)
    b = rhs(BigFloat, m)
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 1024 * eps(BigFloat)
end

@testset "BigFloat: the host SVD has no method" begin
    @test_throws r"^MethodError: no method matching factorize!!\(::GeometricSolvers\.DenseSVD" init(
        LinearProblem(wellcond(BigFloat, 3, 3)), SVDFactorization())
end
