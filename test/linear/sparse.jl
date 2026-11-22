# The sparse LU of UMFPACK, `UMFPACKFactorization`: Newton solves of the sparse block problem in
# `Float64` and `ComplexF64` whose linear solves agree with Sparspak (real) or the dense `lu`
# (complex), the refusals at `init`, the reuse of the symbolic analysis, the detected pattern, the
# singular and non-finite edges, and the Sparspak factor that `LUFactorization()` keeps.

using ADTypes: AutoForwardDiff, AutoSparse
using GeometricSolvers
using GeometricSolvers: LeastSquaresProblem, LUFactorization, Newton, NonlinearProblem,
                        SINGULAR, SUCCESS,
                        UMFPACKFactorization, dense_factor, factorize!!, has_sparse_factor,
                        init, init_linear, ldiv!!, returncode, solve!
using LinearAlgebra: cond, lu, norm
using SparseArrays: SparseArrays, SparseMatrixCSC, getcolptr, nnz, nonzeros, rowvals,
                    sparse,
                    spzeros
using SparseConnectivityTracer: TracerSparsityDetector
using StaticArrays: SVector
using SparseMatrixColorings: GreedyColoringAlgorithm
using Sparspak: Sparspak
using Sparspak.SparseCSCInterface: sparspaklu
using Test

include("../helpers/matrix.jl")
include("../helpers/sparse_problems.jl")

# The element types that UMFPACK factorises, and those that `init` refuses.
const UMFPACK_ELTYPES = filter(T -> T === Float64 || T === ComplexF64, ELTYPES)
const REFUSED_ELTYPES = filter(T -> !(T in UMFPACK_ELTYPES), ELTYPES)

# The relative ∞-norm error that a backward-stable solve of `J d = b` may have: `8n·eps(T)·κ∞`,
# with `n` the order of `J` and `κ∞` the ∞-norm condition number of its dense copy.
agreement_bound(J) = 8 * size(J, 1) * eps(real(eltype(J))) * cond(Matrix(J), Inf)

# The reference solve: Sparspak for a real matrix, the dense LAPACK `lu` for a complex one.
reference_solve(J::SparseMatrixCSC{<:Real}, b) = sparspaklu(copy(J)) \ b
reference_solve(J::SparseMatrixCSC{<:Complex}, b) = lu(Matrix(J)) \ b

relerr(d, ref) = norm(d - ref, Inf) / norm(ref, Inf)

# The Jacobian of `prob` at `z` in a new matrix with the prototype's pattern.
function jacobian_at(prob, z, p)
    J = copy(prob.jacobian_prototype)
    prob.J(J, z, p)
    return J
end

@testset "UMFPACKFactorization solves the sparse block problem: N = $N, T = $T" for N in (
        125, 768),
    T in UMFPACK_ELTYPES

    tol = 1e-10
    prob, _, z0, p_val, z_star = sparse_bracket_problem(N, T)
    m = UMFPACKFactorization()
    solver = init(prob, copy(z0), Newton(; linsolve = m), p_val;
        f_abstol = tol, f_reltol = zero(real(T)))
    @test solver.state.fac isa GeometricSolvers.UMFPACKLU{T}
    @test solver.state.J === solver.state.fac.factors
    x, st = solve!(solver)
    @test st.code === SUCCESS
    @test x != z0
    r = similar(x)
    prob.F(r, x, p_val)
    @test norm(r, 2) <= tol
    @test isapprox(x, z_star; rtol = 1e-9)

    # Each linear solve of the path, at the start, half-way and at the root, agrees with the
    # reference within `8n·eps·κ∞`.
    rhs = T[mod1(7i, 13) / 13 + (T <: Complex ? im * mod1(5i, 11) / 11 : 0) for i in 1:(2N)]
    for z in (z0, (z0 + x) / 2, x)
        J = jacobian_at(prob, z, p_val)
        fac = factorize!!(solver.state.fac, m, J)
        @test returncode(fac) === SUCCESS
        d = ldiv!!(similar(rhs), fac, rhs)
        @test relerr(d, reference_solve(J, rhs)) <= agreement_bound(J)
    end

    # The root agrees with that of the reference factor: Sparspak for a real problem, the dense
    # LU of a problem with no prototype for a complex one.
    prob_ref = T <: Real ? prob :
               NonlinearProblem(prob.F,
        (Jd, z, p) -> copyto!(Jd, Matrix(jacobian_at(prob, z, p))))
    x_ref, st_ref = solve!(init(prob_ref, copy(z0), Newton(), p_val;
        f_abstol = tol, f_reltol = zero(real(T))))
    @test st_ref.code === SUCCESS
    @test relerr(x, x_ref) <= agreement_bound(jacobian_at(prob, x, p_val))
end

@testset "UMFPACKFactorization: in-place solve, a copied Jacobian, scale: T = $T" for T in UMFPACK_ELTYPES
    prob, _, z0, p_val, _ = sparse_bracket_problem(125, T)
    m = UMFPACKFactorization()
    J = jacobian_at(prob, z0, p_val)
    rhs = T.(range(0.1, 1.0; length = size(J, 1)))
    fac0 = init_linear(m, prob.jacobian_prototype, z0)
    @test nonzeros(fac0.factors) == zeros(T, nnz(J))

    # `J` that is not the factor's buffer is copied into it
    fac = factorize!!(fac0, m, J)
    @test fac.factors !== J
    @test nonzeros(fac.factors) == nonzeros(J)
    d = ldiv!!(zeros(T, length(rhs)), fac, rhs)
    @test relerr(d, reference_solve(J, rhs)) <= agreement_bound(J)
    d_ip = copy(rhs)
    @test ldiv!!(d_ip, fac, d_ip) === d_ip
    @test isapprox(d_ip, d; rtol = 8 * eps(real(T)))

    # `J === fac.factors` is factorised as it is
    copyto!(nonzeros(fac.factors), nonzeros(J))
    fac_same = factorize!!(fac, m, fac.factors)
    @test nonzeros(fac_same.factors) == nonzeros(J)
    @test isapprox(ldiv!!(similar(rhs), fac_same, rhs), d; rtol = 8 * eps(real(T)))

    # the scale family `s J`: the solve of `s J d = b` is `d / s` within the bound
    for k in -8:4:8
        s = 10.0^k
        Js = copy(J)
        nonzeros(Js) .*= s
        fac_s = factorize!!(fac, m, Js)
        @test returncode(fac_s) === SUCCESS
        ds = ldiv!!(similar(rhs), fac_s, rhs)
        @test relerr(ds, reference_solve(Js, rhs)) <= agreement_bound(Js)
    end

    @inferred init_linear(m, prob.jacobian_prototype, z0)
    @inferred factorize!!(fac, m, J)
    @inferred ldiv!!(similar(rhs), fac, rhs)
end

@testset "UMFPACKFactorization refuses Float32, ComplexF32 and no pattern at init" begin
    m = UMFPACKFactorization()
    msg = r"^ArgumentError: UMFPACKFactorization factorises a SparseMatrixCSC\{<:Any, Int\} Jacobian pattern on a Vector\{Float64\} or Vector\{ComplexF64\} iterate; got the pattern .* and the iterate .*$"
    @testset "T = $T" for T in REFUSED_ELTYPES
        prob, _, z0, p_val, _ = sparse_bracket_problem(125, T)
        @test_throws msg init(prob, copy(z0), Newton(; linsolve = m), p_val)
        @test_throws msg init_linear(m, prob.jacobian_prototype, z0)
        @test !has_sparse_factor(m, z0)
    end
    @testset "no sparse pattern: T = $T" for T in UMFPACK_ELTYPES
        prob, _, z0, p_val, _ = sparse_bracket_problem(125, T)
        prob_dense = NonlinearProblem(prob.F,
            (Jd, z, p) -> copyto!(Jd, Matrix(jacobian_at(prob, z, p))))
        @test_throws msg init(prob_dense, copy(z0), Newton(; linsolve = m), p_val)
        @test_throws msg init_linear(m, nothing, z0)
        @test_throws msg init_linear(m, Matrix(prob.jacobian_prototype), z0)
    end
    # a `LeastSquaresProblem` carries no sparse pattern, and its factor has `m_rows` rows
    @testset "least squares: T = $T" for T in UMFPACK_ELTYPES
        f_ls!(r, x, p) = (r[1] = x[1] - 1; r[2] = x[2] - 2; r[3] = x[1] + x[2] - 3; r)
        x0 = zeros(T, 2)
        @test_throws msg init(
            LeastSquaresProblem(f_ls!, 3), x0, Newton(; linsolve = m), nothing)
        @test_throws msg init_linear(m, nothing, x0, 3)
    end
    # a static or scalar iterate has no sparse pattern, with or without a reused factor
    @testset "static and scalar iterates" begin
        F_s(x, p) = SVector(x[1]^2 - 2, x[2] - 1)
        @test_throws msg init(
            NonlinearProblem(F_s), SVector(0.0, 0.0), Newton(; linsolve = m), nothing)
        @test_throws msg init(
            NonlinearProblem(F_s), SVector(0.0f0, 0.0f0), Newton(; linsolve = m), nothing)
        @test_throws msg init(
            NonlinearProblem((x, p) -> x^2 - 2), 0.0, Newton(; linsolve = m), nothing)
        @test_throws msg init(NonlinearProblem(F_s), SVector(0.0, 0.0),
            Newton(; linsolve = m, reuse = Int32(2)), nothing)
    end
end

@testset "UMFPACKFactorization reuses the symbolic analysis: T = $T" for T in UMFPACK_ELTYPES
    prob, _, z0, p_val, z_star = sparse_bracket_problem(125, T)
    m = UMFPACKFactorization()
    fac = factorize!!(init_linear(m, prob.jacobian_prototype, z0), m,
        jacobian_at(prob, z0, p_val))
    symbolic = fac.lu.symbolic
    @test symbolic.p != C_NULL
    rhs = ones(T, length(z0))
    for z in (z_star, (z0 + z_star) / 2, 2 .* z0)
        J = jacobian_at(prob, z, p_val)
        fac = factorize!!(fac, m, J)
        @test fac.lu.symbolic === symbolic
        @test returncode(fac) === SUCCESS
        @test relerr(ldiv!!(similar(rhs), fac, rhs), reference_solve(J, rhs)) <=
              agreement_bound(J)
    end

    # one Newton solve keeps the analysis of its first factorisation through every step
    solver = init(prob, copy(z0), Newton(; linsolve = m), p_val; f_abstol = 1e-10,
        f_reltol = 0.0)
    @test solver.state.fac.lu.symbolic.p == C_NULL
    _, st = solve!(solver)
    @test st.code === SUCCESS
    @test st.iterations >= 3
    @test solver.state.fac.lu.symbolic.p != C_NULL
end

@testset "has_sparse_factor of UMFPACKFactorization" begin
    m = UMFPACKFactorization()
    for T in UMFPACK_ELTYPES
        @test has_sparse_factor(m, zeros(T, 3)) === true
    end
    for T in REFUSED_ELTYPES
        @test has_sparse_factor(m, zeros(T, 3)) === false
    end

    # a detected pattern gives the UMFPACK factor, not a dense one
    _, prob_with_proto, z0, p_val, z_star = sparse_bracket_problem(125, Float64)
    prob = NonlinearProblem(prob_with_proto.F)
    backend = AutoSparse(AutoForwardDiff(); sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm())
    solver = init(prob, copy(z0), Newton(; linsolve = m, jacobian = backend), p_val;
        f_abstol = 1e-10, f_reltol = 0.0)
    @test solver.state.fac isa GeometricSolvers.UMFPACKLU{Float64}
    @test solver.state.J isa SparseMatrixCSC{Float64, Int}
    x, st = solve!(solver)
    @test st.code === SUCCESS
    @test isapprox(x, z_star; rtol = 1e-9)
end

@testset "LUFactorization() on a SparseMatrixCSC prototype stays Sparspak: T = $T" for T in REAL_ELTYPES
    ext = Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt)
    P = SparseMatrixCSC{T, Int}
    V = Vector{T}
    for M in (LUFactorization{Nothing}, LUFactorization{T})
        @test which(init_linear, (M, P, V)).module === ext
        @test which(init_linear, (M, P, V, Nothing)).module === GeometricSolvers
        @test which(has_sparse_factor, (M, V)).module === ext
    end
    prob, _, z0, p_val, _ = sparse_bracket_problem(125, T)
    @test dense_factor(LUFactorization(), prob, z0, p_val) isa ext.SparspakLU
    solver = init(prob, copy(z0), Newton(), p_val)
    @test solver.state.fac isa ext.SparspakLU
    @test which(factorize!!,
        (typeof(solver.state.fac), LUFactorization{Nothing}, P)).module === ext
end

@testset "UMFPACKFactorization: singular, non-finite and a changed pattern: T = $T" for T in UMFPACK_ELTYPES
    prob, _, z0, p_val, _ = sparse_bracket_problem(125, T)
    m = UMFPACKFactorization()
    fac0 = init_linear(m, prob.jacobian_prototype, z0)
    J = jacobian_at(prob, z0, p_val)
    fac = factorize!!(fac0, m, J)
    @test fac.info == 0
    @test returncode(fac) === SUCCESS

    # an exactly singular matrix: the stored entries of column 1 are zero
    Js = copy(J)
    nonzeros(Js)[SparseArrays.nzrange(Js, 1)] .= zero(T)
    fac_s = factorize!!(fac, m, Js)
    @test fac_s.info == 1
    @test returncode(fac_s) === SINGULAR

    # a non-finite entry gives `info = 1` without calling UMFPACK; the next finite matrix
    # factorises again with the same analysis
    for bad in (T(NaN), T(Inf), -T(Inf))
        Jn = copy(J)
        nonzeros(Jn)[3] = bad
        fac_n = factorize!!(fac, m, Jn)
        @test fac_n.info == 1
        @test returncode(fac_n) === SINGULAR
    end
    fac_back = factorize!!(fac, m, J)
    @test fac_back.info == 0
    @test fac_back.lu.symbolic === fac.lu.symbolic

    # UMFPACK factorises a non-finite entry above the diagonal of a triangular matrix as a
    # success, so only the check before it gives `info = 1` there
    P2 = sparse(T[1 1; 0 1])
    for bad in (T(NaN), T(Inf), -T(Inf))
        J2 = copy(P2)
        J2[1, 2] = bad
        fac2 = factorize!!(init_linear(m, P2, zeros(T, 2)), m, J2)
        @test fac2.info == 1
        @test returncode(fac2) === SINGULAR
    end

    # a Newton solve that meets the singular matrix stops with `SINGULAR`
    n = length(z0)
    prob_sing = NonlinearProblem(prob.F, (Jm, z, p) -> copyto!(nonzeros(Jm), nonzeros(Js));
        jacobian_prototype = prob.jacobian_prototype)
    _, st = solve!(init(prob_sing, copy(z0), Newton(; linsolve = m), p_val))
    @test st.code === SINGULAR

    # a Jacobian with another pattern or size raises; each of the last three differs from the
    # prototype in one of the size, the column pointers and the row indices alone
    msg = r"^ArgumentError: sparse Jacobian sparsity pattern differs"
    @test_throws msg factorize!!(fac, m, spzeros(T, n, n))
    @test_throws msg factorize!!(fac, m, sparse(1:(n - 1), 1:(n - 1), ones(T, n - 1)))
    colptr, rowval = getcolptr(J), rowvals(J)
    @test_throws msg factorize!!(fac, m,
        SparseMatrixCSC(n + 1, n, copy(colptr), copy(rowval), copy(nonzeros(J))))
    colptr_moved = copy(colptr)
    colptr_moved[2] += 1
    @test_throws msg factorize!!(fac, m,
        SparseMatrixCSC(n, n, colptr_moved, copy(rowval), copy(nonzeros(J))))
    rowval_moved = copy(rowval)
    rows1 = view(rowval_moved, SparseArrays.nzrange(J, 1))
    rows1[end] = first(setdiff(1:n, rows1))
    sort!(rows1)
    @test_throws msg factorize!!(fac, m,
        SparseMatrixCSC(n, n, copy(colptr), rowval_moved, copy(nonzeros(J))))
end

@testset "UMFPACKFactorization: prototype size and diagnostics: T = $T" for T in UMFPACK_ELTYPES
    prob, _, z0, p_val, _ = sparse_bracket_problem(125, T)
    m = UMFPACKFactorization()
    @test_throws r"^ArgumentError: jacobian_prototype has size \(250, 250\), expected \(249, 249\)" init_linear(
        m, prob.jacobian_prototype, z0[1:(end - 1)])

    # the factor has no rank or null-space diagnostics: the status reports rank `-1` and a
    # `NaN` null-space norm
    _, st = solve!(init(prob, copy(z0), Newton(; linsolve = m), p_val))
    @test st.code === SUCCESS
    @test st.rank == -1
    @test isnan(st.nullnorm)
end
