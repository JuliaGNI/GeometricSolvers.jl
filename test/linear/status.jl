# The status and the rank of a factor, apart from each other: the numerical rank of an SVD factor
# in its field `rank`, `info` and `returncode` reporting a failed factorisation alone, on a static
# matrix, a host `Array` and a `JLArray`; `exact_solve` of every factor type; the static kernels on
# `BigFloat` and their allocations on the `isbits` types; `singular_values`; the dense LU through
# `LAPACK.getrf!`; and the LAPACK workspaces of the dense QR and SVD factors.
#
# The matrices have small integer entries, or a diagonal of chosen values, so that they are exact
# in every precision and the numerical rank is the rank.

using GeometricSolvers
using GeometricSolvers: DenseSVD, NonlinearProblem, SINGULAR, SUCCESS, StaticLU, StaticQR,
                        StaticSVD, exact_solve, factorize!!, init, jacobi_tolerance, ldiv!!,
                        returncode, singular_values
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: BLAS, BlasInt, Diagonal, LAPACK, norm, opnorm, svdvals
using Random: Xoshiro
using StaticArrays: SMatrix, SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

allowscalar(false)

svd_method(::Type{Nothing}) = SVDFactorization()
svd_method(::Type{TF}) where {TF} = SVDFactorization{TF}()
nop(y, x, p) = y

vander(::Type{T}, n, r) where {T} = T[i^(k - 1) for i in 1:n, k in 1:r]
"An `n × n` integer matrix of rank `r`, with entries below 400 in modulus at `n = 4`."
rank_r(::Type{T}, n, r) where {T} = vander(T, n, r) * permutedims(vander(T, n, r))
"`I` plus the Hilbert matrix: full rank, with `κ∞` below 20 at `n = 4`."
function full_rank(::Type{T}, n) where {T}
    T[1 / (i + j - 1) + (i == j ? 1 : 0) for i in 1:n, j in 1:n]
end

function factor_static(T, TF, A)
    factorize!!(nothing, svd_method(TF),
        SMatrix{size(A, 1), size(A, 1), T}(A))
end
function factor_dense(T, TF, A, AT)
    m = svd_method(TF)
    fac = init(m, NonlinearProblem(nop), AT(zeros(T, size(A, 1))), nothing)
    return factorize!!(fac, m, AT(A))
end
"The SVD factors of `A` on the three routes: a static matrix, and each of `ARRAY_BACKENDS`."
function routes(T, TF, A)
    (("SMatrix", () -> factor_static(T, TF, A)),
        map(AT -> (string(nameof(AT)), () -> factor_dense(T, TF, A, AT)), ARRAY_BACKENDS)...)
end

"The solve of `A d = b` with the factor `fac` of a route, as a host `Vector`."
function route_solve(label, fac, b)
    n = length(b)
    label == "SMatrix" && return Vector(ldiv!!(nothing, fac, SVector{n}(b)))
    AT = label == "Array" ? Array : JLArray
    return Array(ldiv!!(AT(zeros(eltype(b), n)), fac, AT(copy(b))))
end

@testset "the rank of an SVD factor apart from its status" begin
    @testset "$T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        (label, make) in routes(T, TF, full_rank(T, 4))

        fac = make()
        @test fac.rank == 4
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
    end
    @testset "rank $r, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS, r in 1:3,
        (label, make) in routes(T, TF, rank_r(T, 4, r))
        fac = make()
        @test fac.rank == r
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
    end
    @testset "the zero matrix, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        (label, make) in routes(T, TF, zeros(T, 4, 4))

        fac = make()
        @test fac.rank == 0
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
    end
    @testset "a 1 × 1 matrix $v, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        (v, r) in ((0, 0), (3, 1)), (label, make) in routes(T, TF, fill(T(v), 1, 1))
        fac = make()
        @test fac.rank == r
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
    end
    @testset "the field types" begin
        fac = factor_static(Float64, Nothing, full_rank(Float64, 4))
        @test fac.rank isa Int32
        @test fac.info isa Int32
        @test factor_dense(Float64, Nothing, full_rank(Float64, 4), Array).rank isa Int
    end
end

@testset "a non-finite entry is SINGULAR with rank 0" begin
    @testset "$bad, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        bad in (Inf, -Inf, NaN),
        (label, make) in routes(T, TF, (A = full_rank(T, 4); A[1, 1] = T(bad); A))
        fac = make()
        @test fac.info != 0
        @test returncode(fac) === SINGULAR
        @test fac.rank == 0
        # the failed factor's step is the zero step, not a NaN
        @test route_solve(label, fac, ones(T, 4)) == zeros(T, 4)
    end
    # a 1 × 1 matrix has no pair to rotate, so its sweep converges at once and the finiteness of
    # its singular value is the only test that sees the entry
    @testset "1 × 1 $bad, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        bad in (Inf, NaN), (label, make) in routes(T, TF, fill(T(bad), 1, 1))
        fac = make()
        @test fac.info != 0
        @test returncode(fac) === SINGULAR
        @test fac.rank == 0
    end
    # a `NaN` outside the first column leaves the first singular value finite where the sweep
    # stops at the non-finite column norm, so a rank counted past the failure would not be 0, and
    # a solve that truncates by its own cutoff would divide out the finite columns
    @testset "NaN at $ij, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        ij in ((4, 4), (2, 3)),
        (label, make) in routes(T, TF, (A = full_rank(T, 4); A[ij...] = T(NaN); A))
        fac = make()
        @test fac.info != 0
        @test returncode(fac) === SINGULAR
        @test fac.rank == 0
        @test route_solve(label, fac, ones(T, 4)) == zeros(T, 4)
    end
    # the host SVD refuses a non-finite entry before `gesvd`: the sentinels in `S` and `U` show
    # that no LAPACK routine wrote them
    nonfinite = vcat(
        [(T, TF, (A = full_rank(T, 4); A[ij...] = T(bad); A))
         for (T, TF) in PRECISION_PAIRS
         for (ij, bad) in (((1, 1), Inf), ((1, 1), -Inf), ((1, 1), NaN),
            ((4, 4), NaN), ((2, 3), NaN))],
        [(T, TF, fill(T(bad), 1, 1)) for (T, TF) in PRECISION_PAIRS for bad in (Inf, NaN)])
    @testset "LAPACK is not reached for a non-finite entry, $T, TF = $TF, $(size(A, 1)) × $(size(A, 1))" for (
            T, TF, A) in nonfinite
        m = svd_method(TF)
        fac = init(m, NonlinearProblem(nop), zeros(T, size(A, 1)), nothing)
        fill!(fac.S, 7)
        fill!(fac.U, 7)
        fac = factorize!!(fac, m, A)
        @test all(==(7), fac.S)
        @test all(==(7), fac.U)
        @test fac.info != 0
        @test fac.rank == 0
    end
    # a finite matrix whose largest singular value overflows in `gesvd`, without prescaling
    @testset "an Array whose singular value overflows, $T" for T in REAL_ELTYPES
        fac = factor_dense(T, Nothing, T(floatmax(T) / 2) .* ones(T, 4, 4), Array)
        @test fac.info != 0
        @test returncode(fac) === SINGULAR
        @test fac.rank == 0
    end
    @testset "a static matrix of NaN, $T" for T in REAL_ELTYPES
        fac = factorize!!(nothing, SVDFactorization(), SMatrix{4, 4}(fill(T(NaN), 4, 4)))
        @test fac.info == 1
        @test fac.rank == 0
        @test returncode(fac) === SINGULAR
    end
end

@testset "the rank cutoff √n · eps(TF) · σ₁" begin
    # 1.5c lies above the cutoff and 0.75c below it: a cutoff halved or doubled moves one of them
    @testset "diag(1, 1.5c, 0.75c, 0), $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        (label, make) in routes(T, TF,
            (c = jacobi_tolerance(4, real(factortype(T, TF)));
                Matrix{T}(Diagonal(T.([1, 1.5c, 0.75c, 0])))))

        fac = make()
        @test fac.rank == 2
        @test fac.info == 0
    end
    # a scale that is not a power of two rounds the entries; the rounded matrix is rank 1 up to
    # `eps/2 · σ₁`, below the cutoff
    @testset "s · rank_r(T, 4, 1), s = $s, $T, $label" for T in REAL_ELTYPES,
        s in (1e-8, 1.0, 1e8), (label, make) in routes(T, Nothing, T(s) .* rank_r(T, 4, 1))
        fac = make()
        @test fac.rank == 1
        @test fac.info == 0
    end
end

@testset "the truncated step of a consistent rank-2 system" begin
    @testset "$T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        (label, make) in routes(T, TF, rank_r(T, 4, 2))

        A = rank_r(T, 4, 2)
        b = A * T[1, -1, 2, 0]
        fac = make()
        @test fac.rank == 2
        d = route_solve(label, fac, b)
        @test all(isfinite, d)
        @test norm(A * d - b) ≤ 64 * epsf(T, TF) * opnorm(A) * norm(d)
    end
end

@testset "exact_solve" begin
    @testset "the SVD factor of a $what matrix, $T, TF = $TF, $label" for (T, TF) in PRECISION_PAIRS,
        (A, want, what) in ((full_rank(T, 4), true, "full-rank"), (
            rank_r(T, 4, 1), false, "rank-1"),
            (zeros(T, 4, 4), false, "zero")),
        (label, make) in routes(T, TF, A)

        @test exact_solve(make()) === want
    end
    @testset "the $(nameof(M)) factors, $T" for M in (LUFactorization, QRFactorization),
        T in ELTYPES

        m = M()
        A = full_rank(T, 4)
        @test exact_solve(factorize!!(nothing, m, SMatrix{4, 4, T}(A))) === true
        for AT in ARRAY_BACKENDS
            fac = init(m, NonlinearProblem(nop), AT(zeros(T, 4)), nothing)
            @test exact_solve(factorize!!(fac, m, AT(A))) === true
        end
    end
end

@testset "the static kernels on BigFloat" begin
    A2 = SMatrix{2, 2, BigFloat}(2, 1, 1, 3)
    A4 = SMatrix{4, 4, BigFloat}(full_rank(BigFloat, 4))
    @testset "$(nameof(typeof(m))), n = $(size(A, 1))" for A in (A2, A4),
        m in (LUFactorization(), QRFactorization(), SVDFactorization())

        n = size(A, 1)
        r = SVector{n, BigFloat}(ntuple(BigFloat, n))
        fac = factorize!!(nothing, m, A)
        @test returncode(fac) === SUCCESS
        x = ldiv!!(nothing, fac, r)
        xref = Matrix(A) \ Vector(r)
        κ = opnorm(Matrix(A), Inf) * opnorm(inv(Matrix(A)), Inf)
        @test norm(x - xref, Inf) ≤ 100 * eps(BigFloat) * κ * norm(xref, Inf)
    end
end

# A function barrier with concrete argument types, which calls once before it measures.
allocations(f::F, args::Vararg{Any, K}) where {F, K} = (f(args...); @allocated f(args...))

@testset "the isbits static kernels allocate nothing" begin
    # under --check-bounds=yes a bounds check lets the local MMatrix escape to the heap, so these
    # are skipped where a run forces it
    if Base.JLOptions().check_bounds != 1
        @testset "$(nameof(typeof(m))), $T" for m in (
                LUFactorization(), QRFactorization(), SVDFactorization()),
            T in ELTYPES

            J = SMatrix{4, 4, T}(full_rank(T, 4))
            r = SVector{4, T}(1, 2, 3, 4)
            fac = factorize!!(nothing, m, J)
            @test allocations(factorize!!, nothing, m, J) == 0
            @test allocations(ldiv!!, nothing, fac, r) == 0
        end
        @test allocations(collect, SVector(1.0, 2.0)) > 0
    end
end

@testset "the dense factorisations allocate nothing on an Array" begin
    # skipped where a run forces --check-bounds=yes, as the static kernels above
    if Base.JLOptions().check_bounds != 1
        @testset "$(nameof(typeof(m))), $T" for m in (
                LUFactorization(), QRFactorization(), SVDFactorization()),
            T in ELTYPES

            fac = init(m, NonlinearProblem(nop), zeros(T, 4), nothing)
            J = rand(Xoshiro(2), T, 4, 4)
            factorize!!(fac, m, J)
            @test allocations(factorize!!, fac, m, J) == 0
        end
    end
end

@testset "singular_values and the fields of StaticSVD" begin
    @test fieldnames(StaticSVD) == (:factors, :V, :σ̂, :info, :rank, :scale_exponent)
    @test :rank in fieldnames(DenseSVD)
    B = [1 2; 3 4]
    @testset "2^$k · B, $T" for T in REAL_ELTYPES, k in (-50, 0, 50)

        fac = factorize!!(nothing, SVDFactorization(), SMatrix{2, 2, T}(ldexp.(T.(B), k)))
        sv = singular_values(fac)
        @test sv isa SVector{2, T}
        @test all(isapprox.(sv, ldexp.(svdvals(T.(B)), k); rtol = 16 * eps(T)))
    end
    @testset "a matrix of floatmax/2 entries, $T" for T in REAL_ELTYPES
        fac = factorize!!(nothing, SVDFactorization(), SMatrix{4, 4}(fill(floatmax(T) / 2, 4, 4)))
        @test singular_values(fac)[1] == Inf
        @test all(isfinite, ldiv!!(nothing, fac, SVector{4, T}(1, 1, 1, 1)))
    end
end

@testset "the dense LU through LAPACK.getrf!" begin
    @testset "a zero column, $T" for T in ELTYPES
        A = full_rank(T, 4)
        A[:, 3] .= 0
        fac = init(LUFactorization(), NonlinearProblem(nop), zeros(T, 4), nothing)
        fac = factorize!!(fac, LUFactorization(), A)
        @test fac.info > 0
        @test returncode(fac) === SINGULAR
    end
end

# The optimal workspace sizes from an independent LAPACK query (`lwork = -1`) of `?geqrf` and
# `?gesvd` with `jobu = jobvt = 'A'`, through the library that the package calls.
for (T, geqrf, gesvd) in ((Float64, :dgeqrf_, :dgesvd_), (Float32, :sgeqrf_, :sgesvd_),
    (ComplexF64, :zgeqrf_, :zgesvd_), (ComplexF32, :cgeqrf_, :cgesvd_))
    R = real(T)
    @eval function geqrf_query(::Type{$T}, n)
        A = zeros($T, n, n)
        tau = zeros($T, n)
        work = zeros($T, 1)
        info = Ref{BlasInt}()
        ccall((BLAS.@blasfunc($geqrf), LAPACK.liblapack), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ptr{$T},
                Ref{BlasInt}, Ptr{BlasInt}),
            n, n, A, n, tau, work, -1, info)
        return Int(real(work[1]))
    end
    @eval function gesvd_query(::Type{$T}, n)
        A = zeros($T, n, n)
        S = zeros($R, n)
        U = zeros($T, n, n)
        VT = zeros($T, n, n)
        work = zeros($T, 1)
        rwork = zeros($R, 5n)
        info = Ref{BlasInt}()
        if $T <: Complex
            ccall((BLAS.@blasfunc($gesvd), LAPACK.liblapack), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                    Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                    Ptr{$R}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T},
                    Ref{BlasInt}, Ptr{$R}, Ref{BlasInt}, Clong, Clong),
                'A', 'A', n, n, A, n, S, U, n, VT, n, work, -1, rwork, info, 1, 1)
        else
            ccall((BLAS.@blasfunc($gesvd), LAPACK.liblapack), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                    Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                    Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T},
                    Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                'A', 'A', n, n, A, n, S, U, n, VT, n, work, -1, info, 1, 1)
        end
        return Int(real(work[1]))
    end
end

@testset "the dense workspaces are the LAPACK query's sizes" begin
    @testset "$T, n = $n" for T in ELTYPES, n in (1, 5, 100)

        prob = NonlinearProblem(nop)
        @test length(init(QRFactorization(), prob, zeros(T, n), nothing).work) ==
              geqrf_query(T, n)
        @test length(init(SVDFactorization(), prob, zeros(T, n), nothing).work) ==
              gesvd_query(T, n)
    end
end
