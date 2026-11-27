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
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: norm, opnorm
using Random: Xoshiro, randn
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

# `a · 2^k`, by component for a complex `a`, exact unless it leaves the floating-point range.
pow2(a::Real, k::Integer) = ldexp(a, k)
pow2(a::Complex, k::Integer) = complex(ldexp(real(a), k), ldexp(imag(a), k))

# `κ₁(A) = ‖A‖₁ ‖A⁺‖₁` of the matrix as it is rounded, in `BigFloat`, for `m ≥ n` of full rank.
function cond1(A::AbstractMatrix)
    Ab = big.(Matrix(A))
    P = size(Ab, 1) == size(Ab, 2) ? inv(Ab) : (Ab' * Ab) \ Matrix(Ab')
    return Float64(opnorm(Ab, 1) * opnorm(P, 1))
end

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

# A host array whose axes start at 0, on the CPU backend, so that only the index check refuses it.
struct ZeroBased{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(a::ZeroBased) = size(a.data)
Base.axes(a::ZeroBased) = map(n -> 0:(n - 1), size(a.data))
Base.getindex(a::ZeroBased{T, N}, i::Vararg{Int, N}) where {T, N} = a.data[(i .+ 1)...]
KernelAbstractions.get_backend(::ZeroBased) = KernelAbstractions.CPU()

@testset "an offset b or A: $(nameof(typeof(method))), $m × $n" for (method, m, n) in (
    (LUFactorization(), 3, 3), (QRFactorization(), 3, 3), (QRFactorization(), 5, 3),
    (QRFactorization(), 3, 5), (SVDFactorization(), 3, 5))
    A = wellcond(Float64, m, n)
    solver = init(LinearProblem(A), method)
    # the message of `Base.require_one_based_indexing`
    @test_throws r"^ArgumentError: offset arrays are not supported but got an array with index other than 1$" solve!(
        solver, ZeroBased(rhs(Float64, m)))
    @test_throws r"^ArgumentError: offset arrays are not supported but got an array with index other than 1$" LinearProblem(ZeroBased(A))
    @test_throws r"^ArgumentError: offset arrays are not supported but got an array with index other than 1$" refactorize!!(
        solver, ZeroBased(A))
end

@testset "a b that aliases solver.x: $T" for T in ELTYPES
    # the SVD of rank 0 writes a zero `x`; the finiteness of `b` is read before that write
    solver = init(LinearProblem(zeros(T, 1, 2)), SVDFactorization())
    fill!(solver.x, T(Inf))
    x, st = solve!(solver, view(solver.x, 1:1))
    @test st.code == NONFINITE
    @test isnan_all(x)
end

@testset "a b that overlaps solver.x in another order: the QR of Aᴴ with m < n" begin
    solver = init(LinearProblem([1.0 0 0; 0 1 0]), QRFactorization())
    solver.x .= [2.0, 3.0, 0.0]
    x, st = solve!(solver, view(solver.x, 2:-1:1))
    @test st.code == SUCCESS
    @test x == [3.0, 2.0, 0.0]
end

# `b` is `x[m:-1:1]`, which a solve that writes `x[1]` before it reads `b[m]` overwrites; with
# `m > n` it is `x[[n:-1:1; 1:(m - n)]]`, on the host alone, as a device array takes no index
# vector. The solution of a fresh solver for a copy of `b` is the reference. With `m == n`, `b`
# is also `x` itself.
@testset "a b that overlaps solver.x in another order: $route, $T, $(nameof(typeof(method))), $m × $n" for route in map(
        AT -> string(nameof(AT)), ARRAY_BACKENDS),
    T in ELTYPES,
    (method, m, n) in ((LUFactorization(), 4, 4), (QRFactorization(), 4, 4),
        (SVDFactorization(), 4, 4), (LDLTFactorization(), 4, 4),
        (CholeskyFactorization(), 4, 4), (LUFactorization(Float32), 4, 4),
        (QRFactorization(), 5, 3), (SVDFactorization(), 5, 3),
        (QRFactorization(Float32), 5, 3), (QRFactorization(), 3, 5),
        (SVDFactorization(), 3, 5), (QRFactorization(Float32), 3, 5),
        (SVDFactorization(Float32), 3, 5))

    device_wide_qr(route, method, m, n) && continue
    route == "JLArray" && m > n && continue
    A = m == n ? square_matrix(T, m, method) : wellcond(T, m, n)
    solver = init(LinearProblem(onroute(route, A)), method)
    copyto!(solver.x, onroute(route, rhs(T, n)))
    b = view(solver.x, m > n ? [n:-1:1; 1:(m - n)] : (m:-1:1))
    b_host = Vector(Array(b))
    x_ref, st_ref = lp_solve(route, method, A, b_host)
    x, st = solve!(solver, b)
    @test st.code == st_ref.code == SUCCESS
    @test isapprox(Vector(Array(x)), x_ref; rtol = 8 * eps(real(T)))
    @test relerr(x_ref, xref(A, b_host)) <= 128 * eps(real(factor_eltype(method, T)))
    if m == n
        copyto!(solver.x, onroute(route, b_host))
        x, st = solve!(solver, solver.x)
        @test st.code == SUCCESS
        @test isapprox(Vector(Array(x)), x_ref; rtol = 8 * eps(real(T)))
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
    T in filter(T -> real(T) === Float64, ELTYPES),
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
    # factor, and where `A` is subnormal. The static factors scale `Aᴴ` themselves.
    device_wide_qr(route, method, 3, 5) && continue
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

@testset "a subnormal matrix: $route, $T, $(nameof(typeof(method))), $m × $n" for route in ROUTES,
    T in ELTYPES,
    (method, m, n) in ((LUFactorization(), 4, 4), (QRFactorization(), 4, 4),
        (SVDFactorization(), 4, 4), (LDLTFactorization(), 4, 4),
        (CholeskyFactorization(), 4, 4), (QRFactorization(), 5, 3),
        (SVDFactorization(), 5, 3), (SVDFactorization(Float32), 4, 4),
        (SVDFactorization(Float32), 5, 3))

    # Every factor in the element type of the matrix divides it by the power of two of its largest
    # entry (Cholesky by the even power at or below it), so that it factorises a matrix in the
    # normal range, and so does the `Float32` SVD of a `Float64` matrix, whose sweep squares the
    # entries. The entries of `2^kA A₀` lie 4 bits above the smallest subnormal (`kA = -1070` in
    # `Float64`, `-145` in `Float32`), and `b = 2^(kA + 70) b₀` and the solution lie in the normal
    # range. The matrix rounded to the
    # subnormal grid keeps the diagonal dominance of `A₀`, so a backward-stable solve meets a
    # small multiple of `eps(RF) κ₁` against the solution in `BigFloat` of the rounded matrix, `RF`
    # the real type of the factor.
    R = real(T)
    RF = real(factor_eltype(method, T))
    kA = exponent(nextfloat(zero(R))) + 4
    A0 = m == n ? square_matrix(T, n, method) : wellcond(T, m, n)
    A = pow2.(A0, kA)
    b = pow2.(rhs(T, m), kA + 70)
    @test all(a -> abs(real(a)) < floatmin(R) && abs(imag(a)) < floatmin(R), A)
    κ = cond1(A)
    @test κ <= 16
    x, st, _ = lp_solve(route, method, A, b)
    if route == "JLArray" && RF !== R
        # the device `Float32` SVD of a `Float64` matrix gives `SINGULAR` here
        @test_broken st.code == SUCCESS  # issue #122, K125 in KNOWN_ISSUES.md
        @test_broken relerr(x, xref(A, b)) <= 32 * eps(RF) * κ  # issue #122, K125 in KNOWN_ISSUES.md
    else
        @test st.code == SUCCESS
        @test relerr(x, xref(A, b)) <= 32 * eps(RF) * κ
    end
end

@testset "the Cholesky factor of a 5 × 5 matrix of subnormal entries and a subnormal b: $route, $T" for route in ROUTES,
    T in ELTYPES

    # `A = 3s I + s (e₁e₃ᵀ + e₃e₁ᵀ)` and `b = s (1, …, 1)` with `s = 2^10 t`, `t = nextfloat(0)`:
    # the solution `(1/4, 1/3, 1/4, 1/3, 1/3)` lies in the normal range. The factor of `A / 2^e`
    # does too, and the solve lifts `b` by `2^-k`, `k` the exponent of `b` clamped to
    # `-exponent(floatmax(R))` and then to `[min(e, 0), 0]` (here `e + 1`), so the triangular
    # solves run in the normal range on `2^(e - k) x = x / 2` and meet a small multiple of
    # `eps(R) κ₁`. Solved with `b` as it is, they form products of the size of `s/3` on the
    # subnormal grid of spacing `t`, each with a relative error of up to `1.5 t/s`, `1.9e-4` here.
    R = real(T)
    t = nextfloat(zero(R))
    s = ldexp(t, 10)
    A = Matrix{T}(3s * I(5))
    A[3, 1] = A[1, 3] = s
    b = fill(T(s), 5)
    x, st, _ = lp_solve(route, CholeskyFactorization(), A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 8 * eps(R) * cond1(A)
end

@testset "the Cholesky factor of [2t t; t t] with t = nextfloat(0): $route, $T" for route in ROUTES,
    T in ELTYPES

    # The exact solution of `[2t t; t t] x = (0, t)` is `(-1, 2)`. The factor of `A / 2^e` lies in
    # the normal range, `e` the exponent of `2t` clamped to `-exponent(floatmax(R))` and rounded
    # down to an even one, and the solve lifts the right-hand side `(0, t)` by `2^-k`, `k` the
    # exponent of `t` clamped to `-exponent(floatmax(R))` and then to `[min(e, 0), 0]` (here
    # `e + 1`), so it runs in the normal range on `2^(e - k) x = x / 2` and meets a small multiple
    # of `eps(R) κ₁`. Solved with `(0, t)` as it is, the forward substitution gives an entry of the
    # size of `√(t 2^e)`, a subnormal whose spacing `t` is a relative error of
    # `√(t 2^-e) ≈ √(t floatmax(R))`. The factor of `A` unscaled loses `L₂₁² = t/2` below the
    # smallest subnormal and gives `(-0.5, 1)`, a relative error of 0.5.
    R = real(T)
    t = nextfloat(zero(R))
    A = T[2t t; t t]
    b = T[0, t]
    x, st, _ = lp_solve(route, CholeskyFactorization(), A, b)
    @test st.code == SUCCESS
    @test relerr(x, big.(T[-1, 2])) <= 8 * eps(R) * cond1(A)
end

const SMALL_SYSTEMS = (
    ([3.0;;], [1.0]), ([2.25;;], [1.0]), ([2.0 1.0; 1.0 2.0], [1.0, -2.0]),
    ([2.0 1.0; 1.0 2.0], [7.0, -3.0]))

@testset "a subnormal b with a matrix near floatmin: $route, $T, $(nameof(typeof(method))), A₀ = $A₀, b₀ = $b₀, kA = $j - emax" for route in ROUTES,
    T in ELTYPES,
    method in SQUARE,
    (A₀, b₀) in SMALL_SYSTEMS,
    j in (23, 1)

    # `A = 2^kA A₀`, `kA = j - emax`, is in the normal range, but its largest entry has an exponent
    # `e < 0` far below zero, and `b = t b₀`, `t = nextfloat(0)`, is subnormal; the solution `A⁻¹ b`
    # lies in the normal range. The own-type factor holds `A / 2^e` and the solve divides `b` by
    # `2^k` with `k = e`, which lifts it into the normal range by the same power of two, exactly,
    # so that the triangular solves run in the normal range and the solve meets a small multiple of
    # `eps(R) κ₁`. A complex `T` takes `b (1 + i)`.
    R = real(T)
    t = nextfloat(zero(R))
    A = T.(pow2.(R.(A₀), j - exponent(floatmax(R))))
    b = T.(R.(b₀) .* t) .* (T <: Complex ? T(1 + im) : one(T))
    κ = cond1(A)
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 32 * eps(R) * κ
end

@testset "a subnormal b with a tall matrix near floatmin: $route, $T, $(nameof(typeof(method)))" for route in ROUTES,
    T in ELTYPES,
    method in filter(M -> M isa Union{QRFactorization, SVDFactorization}, SQUARE)

    # The least-squares solve of the same-type QR and SVD factor of a `3 × 2` matrix at
    # `2^(23 - emax)` and a subnormal `b`: the solve lifts `b` with the matrix, as for a square one.
    R = real(T)
    t = nextfloat(zero(R))
    A = T.(pow2.(R[2 1; 1 2; 1 1], -exponent(floatmax(R)) + 23))
    b = T.(R[1, -2, 3] .* t)
    κ = cond1(A)
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 32 * eps(R) * κ
end

@testset "a b near floatmax with a matrix of exponent below 0: $route, $T, $(nameof(typeof(method)))" for route in ROUTES,
    T in ELTYPES,
    method in SQUARE

    # `A = [2 1; 1 2] / 4` has the exponent `e = -1` (`-2` for the even Cholesky one), and
    # `b = 0.6 floatmax (1, 1)` gives the finite solution `0.8 floatmax (1, 1)`. Lifted by `2^-e`,
    # `b` would overflow; of an exponent above 0 it is not lifted, the headroom bound divides it by
    # `2^3`, and the solve stays finite.
    R = real(T)
    A = T.(R[2 1; 1 2] ./ 4)
    b = fill(T(R(0.6) * floatmax(R)), 2)
    κ = cond1(A)
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test all(isfinite, x)
    @test relerr(x, xref(A, b)) <= 32 * eps(R) * κ
end

@testset "m < n, the static QR of a matrix near floatmin and a subnormal b: $T, $(size(A₀, 1)) × $(size(A₀, 2))" for T in ELTYPES,
    (A₀, b₀) in (([1.0 1.0 1.0 1.0], [1.0]), ([2.0 1.0 0.0; 1.0 2.0 1.0], [1.0, -2.0]))

    # The QR of `Aᴴ / 2^e` of `A = 2^(23 - emax) A₀` and `b = t b₀`, `t = nextfloat(0)`: the
    # minimum-norm solution lies in the normal range, and the solve lifts `b` by `2^-e` with the
    # matrix, so that the forward substitution `Rᴴ y = b / 2^k` runs in the normal range.
    R = real(T)
    t = nextfloat(zero(R))
    A = T.(pow2.(R.(A₀), 23 - exponent(floatmax(R))))
    b = T.(R.(b₀) .* t)
    κ = cond1(Matrix(A'))
    x, st, _ = lp_solve("SMatrix", QRFactorization(), A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 32 * eps(R) * κ
end

@testset "m < n, the QR of a matrix of exponent 0 or below and a b near floatmax: $route, $T, A₀ / $d" for route in filter(
        !=("JLArray"), ROUTES),
    T in ELTYPES,
    d in (4, 2)

    # `A = [2 1 0; 1 2 1] / d` has the exponent `e = -1` for `d = 4` and 0 for `d = 2`, and
    # `b = 0.6 floatmax (1, 1)` gives the finite minimum-norm solution
    # `0.6 floatmax · d/14 (5, 4, 1)`. Lifted by `2^-e`, `b` would overflow; of an exponent above 0
    # it is not lifted, the headroom bound divides it by `2^3`, the solve stays finite, and its
    # result is multiplied by `2^(3 - e)`, also where `e = 0`.
    R = real(T)
    A = T.(R[2 1 0; 1 2 1] ./ d)
    b = fill(T(R(0.6) * floatmax(R)), 2)
    κ = cond1(Matrix(A'))
    x, st, _ = lp_solve(route, QRFactorization(), A, b)
    @test st.code == SUCCESS
    @test all(isfinite, x)
    @test relerr(x, xref(A, b)) <= 32 * eps(R) * κ
end

@testset "m < n, the dense QR of a matrix of exponent above 0 and a b near floatmax: $T" for T in ELTYPES

    # `A = [2 1 0; 1 2 1]` has the exponent `e = 1`, and `b = 2^(emax - 1) (1, -2)` gives the
    # finite minimum-norm solution `2^(emax - 1) (1, -1, -1)`. Solved with `b` as it is, the solve
    # overflows; the headroom bound divides `b` by `2^k`, `k = 3`, and the result is multiplied by
    # `2^(k - e)`.
    R = real(T)
    A = T.(R[2 1 0; 1 2 1])
    b = T.(ldexp.(R[1, -2], exponent(floatmax(R)) - 1))
    κ = cond1(Matrix(A'))
    x, st, _ = lp_solve("Array", QRFactorization(), A, b)
    @test st.code == SUCCESS
    @test all(isfinite, x)
    @test relerr(x, xref(A, b)) <= 32 * eps(R) * κ
end

@testset "a b of exponent above 0 is not divided with a matrix of exponent below 0: $route, $T, $(nameof(typeof(method)))" for route in ROUTES,
    T in ELTYPES,
    method in SQUARE

    # `A = diag(1/2, 1/4)` has the exponent `e = -1` (`-2` for the even Cholesky one), and
    # `b = (2^20, 3 · 2^(exponent(t) + 14))`, `t = nextfloat(0)`, gives the solution
    # `(2^21, 12 · 2^(exponent(t) + 14))`, with a subnormal second entry. `b` of an exponent above 0
    # is solved as it is, and the solve of the diagonal factor gives the second entry exactly and
    # the first one within a few ulp (Cholesky's `√2`); divided by `2^20`, the second entry of `b`
    # would round to 0.
    R = real(T)
    s = ldexp(one(R), exponent(nextfloat(zero(R))) + 14)
    A = T[R(0.5) 0; 0 R(0.25)]
    b = T[ldexp(one(R), 20), 3s]
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test isapprox(x[1], ldexp(one(R), 21); rtol = 8 * eps(R))
    @test isapprox(x[2], 12s; rtol = 8 * eps(R))
end

# The Wilkinson growth matrix of size `n`: 1 on the diagonal and in the last column, -1 below the
# diagonal.
function wilkinson(n::Int)
    W = Matrix(1.0I, n, n)
    for i in 2:n
        W[i, 1:(i - 1)] .= -1
    end
    W[:, n] .= 1
    return W
end

@testset "the LU solve of 2^kA W and a solution near floatmax / 2^n: $route, $T, n = $n, kA = $kA" for route in ROUTES,
    T in REAL_ELTYPES,
    n in (8, 12),
    kA in (-40, 20 - exponent(floatmax(T)))

    # `A = 2^kA W` with the Wilkinson matrix `W` and `kA < 0`, so that the own-type factor holds
    # `A / 2^e` with `e < 0`, and `b = A x` with `x = 2^(emax - n - 1) v`, `v = (1, -2, 3, …)`.
    # Partial pivoting keeps `U = I + 2^(i - 1)` in the last column, so the back substitution forms
    # intermediates up to `2^(n - 1)` times the solution it gives. With `b` lifted by `2^-e`, that
    # solution is `x` itself and the intermediates overflow; `b` of an exponent above 0 is not
    # lifted, the solve gives `2^e x`, and its intermediates stay finite.
    emax = exponent(floatmax(T))
    A = T.(ldexp.(wilkinson(n), kA))
    v = T.(collect(1:n) .* (-1) .^ (0:(n - 1)))
    b = T.(big.(A) * (big.(v) .* big(2.0)^(emax - n - 1)))
    x, st, _ = lp_solve(route, LUFactorization(), A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 64 * eps(T) * cond1(A)
end

@testset "m < n, a matrix below floatmin and a subnormal b: $route, $T, $(nameof(typeof(method))), kA = $kA" for (route, method) in (
        ("SMatrix", QRFactorization()), ("SMatrix", SVDFactorization()),
        ("Array", QRFactorization())),
    T in ELTYPES,
    kA in (-exponent(floatmax(real(T))) + 23, -exponent(floatmax(real(T))) - 3,
        exponent(nextfloat(zero(real(T)))) + 12)

    # `A = 2^kA (1, 1, 1, 1)` near or below `floatmin` and `b = t`, `t = nextfloat(0)`: the
    # minimum-norm solution `2^(-kA - 2) t (1, 1, 1, 1)` lies in the normal range. The factor holds
    # `Aᴴ / 2^e`, and the solve lifts `b` with the matrix, so that the forward substitution of the
    # QR and the coordinates of the SVD do not run on the subnormal grid; with `b` as it is, they
    # lose the solution to 0, or to a power of two of it.
    R = real(T)
    t = nextfloat(zero(R))
    A = fill(T(ldexp(one(R), kA)), 1, 4)
    b = T[t]
    x, st, _ = lp_solve(route, method, A, b)
    @test st.code == SUCCESS
    @test relerr(x, xref(A, b)) <= 32 * eps(R)
end

@testset "a Cholesky factor in the element type of A · 2^-4 gives the bits of A: $route, $T" for route in ROUTES,
    T in ELTYPES

    # `A = Wᴴ W + 4I`, of a seeded `W`, has its largest entry in `[4, 32)` in these draws, so its
    # own-type Cholesky factor copies it, and `A · 2^-4` has an exponent `e` between -2 and 0. The
    # factor divides it by the even `2^(e & ~1)`, so it factorises `A` times a power of 4, whose
    # Cholesky factor and solve are those of `A` times a power of two, bit for bit; an odd `e = -1`,
    # which 26 of the 32 real and 14 of the 32 complex draws give, would factorise `A / 2` and round
    # otherwise. A rate across the sweep, because a draw whose `e` is even gives the same bits under
    # both rules.
    rng = Xoshiro(20261125)
    N = 32
    same = 0
    success = 0
    for _ in 1:N
        W = T <: Complex ? complex.(randn(rng, 4, 4), randn(rng, 4, 4)) : randn(rng, 4, 4)
        Wb = big.(T.(W))
        A = T.(Wb' * Wb + 4I)
        b = T.(T <: Complex ? complex.(randn(rng, 4), randn(rng, 4)) : randn(rng, 4))
        x0, st0, _ = lp_solve(route, CholeskyFactorization(), A, b)
        x1, st1, _ = lp_solve(route, CholeskyFactorization(), pow2.(A, -4), pow2.(b, -4))
        success += st0.code == SUCCESS
        same += x1 == x0 && st1.code == st0.code
    end
    @test success == N
    @test same >= 0.9 * N
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
    # a matrix that the factor divides by `2^e` with `e < 0`: `A / 8`, and `2^-1070 A`, whose
    # entries are subnormal
    for k in (-3, -1070)
        Ak = pow2.(A, k)
        SAk = SMatrix{m, n}(Ak)
        @inferred refactorize!!(solver, SAk)
        if Base.JLOptions().check_bounds != 1
            @test allocations(refactorize!!, solver, SAk) == 0
        end
        @test allocations(refactorize!!, hsolver, Ak) == 0
    end
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
