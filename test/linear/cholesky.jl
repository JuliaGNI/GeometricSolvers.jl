using GeometricSolvers
using GeometricSolvers: DenseCholesky, DeviceCholesky, StaticCholesky, dense_factor,
                        factor_eltype
using GeometricSolvers: factorize!!, init_linear, ldiv!!, promote_linsolve, returncode
using GPUArraysCore: allowscalar
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: Diagonal, I, LowerTriangular, cond, norm, opnorm, qr, transpose
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

@testset "The prescaling reads the lower triangle and the real diagonal only ($T)" for T in CHOLESKY_ELTYPES
    # The Hermitian positive definite matrix of the lower triangle is `[4 2+i; 2-i 5]`, or its real
    # part. Above the diagonal, and in the imaginary part of the diagonal, sits an entry of the
    # largest exponent of `real(T)`, which the matrix is not divided by: divided by it, the lower
    # triangle of a factor in a narrower precision would be zero.
    R = real(T)
    huge = ldexp(one(R), exponent(floatmax(R)) - 1)
    A = T <: Real ? T[4 2; 2 5] : T[4 2+im; 2-im 5]
    A_huge = copy(A)
    A_huge[1, 2] = huge
    if T <: Complex
        A_huge[1, 1] += T(0, huge)
        A_huge[2, 2] -= T(0, huge)
    end
    b = T <: Real ? T[1, -2] : T[1 + im, -2]
    TF = mixed_precision(T)
    # a `Float16` factor is no `BlasFloat` and takes the generic loop on the dense path too
    for (method, ε) in ((CholeskyFactorization(), eps(R)),
        (CholeskyFactorization(TF), max(eps(R), eps(TF))),
        (CholeskyFactorization(Float16), eps(Float16)))
        for ((f, x), (f_huge, x_huge)) in zip(
            cholesky_paths(method, A, b), cholesky_paths(method, A_huge, b))
            @test returncode(f_huge) == SUCCESS
            @test f_huge.scale_exponent == f.scale_exponent
            @test Matrix(f_huge.factors) == Matrix(f.factors)
            @test x_huge == x
            @test agrees_with_backslash(A, x_huge, b, ε)
        end
        # the diagonal sets the scale where it is larger than the strict lower triangle: here
        # `5 · 2²⁰` is above `floatmax(Float16)`, and `2⁻¹⁰` far below it
        A_diag = T[4 2.0^-30; 2.0^-30 5] * 2^20
        for (f, x) in cholesky_paths(method, A_diag, b)
            @test returncode(f) == SUCCESS
            @test agrees_with_backslash(A_diag, x, b, ε)
        end
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

    # a non-finite entry anywhere gives info = 1 and scale_exponent = 0 before the factorisation;
    # `potrf` itself returns 0 for a `NaN` below the first column's diagonal entry
    n = 4
    A = randhpd(T, n, 10, 53)
    b = ones(T, n)
    A_nan = copy(A)
    A_nan[n, 1] = T(NaN)
    A_inf_upper = copy(A)
    A_inf_upper[1, n] = T(Inf)
    nonfinite_cases = if T <: Complex
        A_inf_diag_im = copy(A)
        A_inf_diag_im[1, 1] = complex(real(A[1, 1]), real(T)(Inf))
        (A_nan, A_inf_upper, A_inf_diag_im)
    else
        (A_nan, A_inf_upper)
    end
    for M in nonfinite_cases,
        method in (CholeskyFactorization(), CholeskyFactorization(mixed_precision(T)),
            CholeskyFactorization(Float16))

        for (fac, _) in cholesky_paths(method, M, b)
            @test fac.info == 1
            @test fac.scale_exponent == 0
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

# The dense host factor of `A` for `method` and the device factor on a `JLArray`, each with its
# solution for `b` as a host `Vector`.
function cholesky_host(method, A::Matrix{T}, b::Vector{T}) where {T}
    fac = factorize!!(dense_factor(method, nothing, b, nothing), method, A)
    return fac, ldiv!!(similar(b), fac, b)
end
function cholesky_device(method, A::Matrix{T}, b::Vector{T}) where {T}
    bjl = JLArray(b)
    fac = factorize!!(dense_factor(method, nothing, bjl, nothing), method, JLArray(A))
    return fac, Vector(ldiv!!(similar(bjl), fac, bjl))
end

# A seeded right-hand side of element type `T`, complex in both parts for a complex `T`.
function cholesky_rhs(::Type{T}, n::Int, seed::Int) where {T}
    rng = Xoshiro(seed)
    return T.(T <: Real ? randn(rng, n) : randn(rng, ComplexF64, n))
end

@testset "Dispatch: a Vector iterate gets DenseCholesky, a JLArray iterate DeviceCholesky" begin
    allowscalar(false)
    for T in ELTYPES, method in (CholeskyFactorization(), CholeskyFactorization(Float32))

        F = factor_eltype(method, T)
        @test init_linear(method, nothing, ones(T, 3)) isa DenseCholesky
        @test init_linear(method, nothing, ones(T, 3), 5) isa DenseCholesky
        x = JLArray(ones(T, 3))
        for fac in (init_linear(method, nothing, x), init_linear(method, nothing, x, 5))
            @test fac isa DeviceCholesky
            @test fac.factors isa JLArray{F, 2}
            @test size(fac.factors) == (3, 3)
            @test fac.rhs isa (F === T ? Nothing : JLArray{F, 1})
        end
    end
    # a device iterate solves with the device factor, end to end; `J = 3 diag(x²)` is positive
    # definite on the way from `x₀ = 1` to the root `2`
    F!(r, x, p) = (r .= x .^ 3 .- p; r)
    x, st = solve(NonlinearProblem(F!), JLArray(ones(3)),
        Newton(; linsolve = CholeskyFactorization()), 8.0)
    @test st.code == SUCCESS
    @test x isa JLArray
    # the residual test stops at `‖F(x)‖ ≤ √eps ‖F(x₀)‖`, `‖F(x₀)‖ = 7√3`, and `F′ = 3x² ≥ 3` on
    # `[1, 2]` gives `|x - 2| ≤ 7√3 √eps / 3 < 4.1 √eps`; the host solve takes the same steps
    @test all(xi -> isapprox(xi, 2; atol = 4.1 * sqrt(eps())), Array(x))
    xh, sth = solve(NonlinearProblem(F!), ones(3),
        Newton(; linsolve = CholeskyFactorization()), 8.0)
    @test sth.code == SUCCESS
    @test st.iterations == sth.iterations
    @test norm(Array(x) - xh, Inf) <= 8 * eps() * 2
    # the stacked LevenbergMarquardt step refuses a device iterate, and names the method
    @test_throws r"^ArgumentError: a device array iterate takes LevenbergMarquardt\(\).*LDLTFactorization\(\), CholeskyFactorization\(\) and scaling = :diagonal is the stacked QR solve" init(
        NonlinearProblem(F!), JLArray(ones(3)),
        Newton(; linsolve = CholeskyFactorization(), globalization = LevenbergMarquardt()), 1.0)
end

@testset "DeviceCholesky on a JLArray agrees with DenseCholesky at n = $n ($T)" for T in ELTYPES,
    n in (1, 4, 64, 65, 128)

    # one workgroup of 64 work-items: below it, at it and above it, where each work-item takes
    # more than one column
    allowscalar(false)
    R = real(T)
    A = randhpd(T, n, 1e2, 67 + n)
    b = cholesky_rhs(T, n, 71 + n)
    κ = cond(A, Inf)
    TF = mixed_precision(T)
    for (method, ε) in ((CholeskyFactorization(), eps(R)),
        (CholeskyFactorization(TF), max(eps(R), eps(TF))))
        fh, xh = cholesky_host(method, A, b)
        fd, xd = cholesky_device(method, A, b)
        @test fd isa DeviceCholesky
        @test returncode(fd) == SUCCESS
        @test fd.scale_exponent == fh.scale_exponent
        @test norm(xd - xh, Inf) <= 8 * n * ε * κ * norm(xh, Inf)
        # the factor matrix is `L`, with a real positive diagonal and zeros above it
        L = Array(fd.factors)
        @test L == LowerTriangular(L)
        @test all(i -> iszero(imag(L[i, i])) && real(L[i, i]) > 0, 1:n)
        if method === CholeskyFactorization()
            @test norm(L * L' - A, Inf) <= 8 * n * eps(R) * opnorm(A, Inf)
        end
    end
end

@testset "DeviceCholesky on a JLArray: not positive definite and non-finite ($T)" for T in ELTYPES
    allowscalar(false)
    for method in (CholeskyFactorization(), CholeskyFactorization(mixed_precision(T)))
        fac, _ = cholesky_device(method, Matrix(Diagonal(T[1, 2, -1, 3])), ones(T, 4))
        @test fac.info == 3
        @test returncode(fac) == SINGULAR
        # of two leading minors that are not positive, `info` names the first
        fac, _ = cholesky_device(method, Matrix(Diagonal(T[1, -1, -2])), ones(T, 3))
        @test fac.info == 2
        # a zero pivot is not positive either
        fac, _ = cholesky_device(method, Matrix(Diagonal(T[0, 1])), ones(T, 2))
        @test fac.info == 1
        @test returncode(fac) == SINGULAR
        # the zero matrix sets no scale: e = 0, as on the host
        fac, _ = cholesky_device(method, zeros(T, 3, 3), ones(T, 3))
        @test fac.info == 1
        @test fac.scale_exponent == 0
        # in a matrix that is not positive definite, a strictly lower entry, its imaginary part
        # for a complex `T`, can set the scale; the device takes it as the host does
        M = T <: Real ? T[1 8; 8 1] : T[1 -8im; 8im 1]
        fh, _ = cholesky_host(method, M, ones(T, 2))
        fac, _ = cholesky_device(method, M, ones(T, 2))
        @test fac.info == fh.info == 2
        @test fac.scale_exponent == fh.scale_exponent
        # past the first workgroup of columns: the pivot of order 70 is negative
        A = Matrix(Diagonal(fill(T(2), 80)))
        A[70, 70] = -1
        fac, _ = cholesky_device(method, A, ones(T, 80))
        @test fac.info == 70

        # a non-finite entry anywhere gives info = 1 and scale_exponent = 0 before the
        # factorisation, as on the host
        for n in (4, 65)
            A = randhpd(T, n, 10, 53 + n)
            b = ones(T, n)
            A_nan = copy(A)
            A_nan[n, 1] = T(NaN)
            A_inf_upper = copy(A)
            A_inf_upper[1, n] = T(Inf)
            cases = Any[A_nan, A_inf_upper]
            if T <: Complex
                A_inf_diag_im = copy(A)
                A_inf_diag_im[1, 1] = complex(real(A[1, 1]), real(T)(Inf))
                push!(cases, A_inf_diag_im)
            end
            for M in cases
                fh, _ = cholesky_host(method, M, b)
                fd, _ = cholesky_device(method, M, b)
                @test fh.info == 1
                @test fd.info == 1
                @test fd.scale_exponent == 0
                @test returncode(fd) == SINGULAR
                # the matrix is not factorised, and the factor holds it as it was loaded
                @test isequal(Array(fd.factors), fh.factors)
            end
        end
    end
end

@testset "DeviceCholesky on a JLArray: n = 0 and n = 1 ($T)" for T in ELTYPES
    allowscalar(false)
    m = CholeskyFactorization()
    b0 = JLArray(T[])
    f0 = dense_factor(m, nothing, b0, nothing)
    # n = 0 launches nothing: a kernel would write the status buffer
    fill!(f0.status, Int32(7))
    f0 = factorize!!(f0, m, JLArray(zeros(T, 0, 0)))
    @test Array(f0.status) == Int32[7, 7]
    @test f0 isa DeviceCholesky
    @test returncode(f0) == SUCCESS
    @test f0.scale_exponent == 0
    @test isempty(ldiv!!(similar(b0), f0, b0))
    fac, x = cholesky_device(m, fill(T(4), 1, 1), T[2])
    @test returncode(fac) == SUCCESS
    @test Array(fac.factors) == fill(T(2), 1, 1)
    @test x == T[1 // 2]
    fac, _ = cholesky_device(m, fill(T(-4), 1, 1), T[2])
    @test fac.info == 1
    # the load kernel reads `J` by index: a `J` of another size or on another backend is refused
    fac3 = dense_factor(m, nothing, JLArray(ones(T, 3)), nothing)
    @test_throws r"^DimensionMismatch: J has size \(3, 4\), the Cholesky factor \(3, 3\)" factorize!!(
        fac3, m, JLArray(ones(T, 3, 4)))
    @test_throws r"^DimensionMismatch: J has size \(2, 2\), the Cholesky factor \(3, 3\)" factorize!!(
        fac3, m, JLArray(ones(T, 2, 2)))
    @test_throws r"^ArgumentError: J is on the backend .+, the Cholesky factor on .*JLBackend" factorize!!(
        fac3, m, Matrix{T}(I(3)))
end

@testset "DeviceCholesky on a JLArray: a vector of another length ($T)" for T in ELTYPES
    allowscalar(false)
    m = CholeskyFactorization()
    # the backsolve kernel reads the factor by the length of `d`: a vector of another length,
    # the empty one included, is refused, by `ldiv!!` and by `backsolve!`
    fac2, _ = cholesky_device(m, T[4 0; 0 9], T[2, 3])
    for k in (0, 1, 3)
        msg = Regex("^DimensionMismatch: d has length $k, the Cholesky factor \\(2, 2\\)")
        @test_throws msg ldiv!!(JLArray(ones(T, k)), fac2, JLArray(ones(T, k)))
        @test_throws msg GeometricSolvers.backsolve!(fac2, JLArray(ones(T, k)))
    end
end

@testset "DeviceCholesky on a JLArray: a Hermitian matrix with non-real off-diagonal entries ($T)" for T in filter(
    T -> T <: Complex, ELTYPES)
    allowscalar(false)
    A = T[4 1+2im 0.5-1im; 1-2im 5 2im; 0.5+1im -2im 6]
    b = T[1, -2im, 3 + 1im]
    R = real(T)
    fh, xh = cholesky_host(CholeskyFactorization(), A, b)
    fd, xd = cholesky_device(CholeskyFactorization(), A, b)
    @test returncode(fd) == SUCCESS
    L = Array(fd.factors)
    @test L == LowerTriangular(L)
    @test norm(L - fh.factors, Inf) <= 8 * 3 * eps(R) * cond(A, Inf) * norm(fh.factors, Inf)
    @test norm(L * L' - A, Inf) <= 8 * 3 * eps(R) * opnorm(A, Inf)
    # `L Lᵀ` is not `A`: the factor is no complex symmetric one
    @test norm(L * transpose(L) - A, Inf) > 1
    @test norm(xd - xh, Inf) <= 8 * 3 * eps(R) * cond(A, Inf) * norm(xh, Inf)
    # the imaginary part of the diagonal is not read, as in `zpotrf`
    A_imdiag = copy(A)
    for i in 1:3
        A_imdiag[i, i] += T(0, i)
    end
    fd_im, xd_im = cholesky_device(CholeskyFactorization(), A_imdiag, b)
    @test Array(fd_im.factors) == L
    @test xd_im == xd
end

@testset "DeviceCholesky on a JLArray reads the lower triangle and the real diagonal only ($T)" for T in ELTYPES
    allowscalar(false)
    R = real(T)
    TF = mixed_precision(T)
    methods = ((CholeskyFactorization(), eps(R)),
        (CholeskyFactorization(TF), max(eps(R), eps(TF))),
        (CholeskyFactorization(Float16), eps(Float16)))
    # finite entries above the diagonal that have nothing to do with `A`
    A = randhpd(T, 5, 1e2, 41)
    b = cholesky_rhs(T, 5, 43)
    A_junk = copy(A)
    junk = cholesky_rhs(T, 25, 47)
    for j in 1:5, i in 1:(j - 1)

        A_junk[i, j] = 100 * junk[i + 5 * (j - 1)]
    end
    for (method, _) in methods
        f, x = cholesky_device(method, A, b)
        f_junk, x_junk = cholesky_device(method, A_junk, b)
        @test returncode(f_junk) == SUCCESS
        @test Array(f_junk.factors) == Array(f.factors)
        @test x_junk == x
    end
    # an entry of the largest exponent above the diagonal and in the imaginary part of the
    # diagonal sets no scale, as on the host
    huge = ldexp(one(R), exponent(floatmax(R)) - 1)
    A2 = T <: Real ? T[4 2; 2 5] : T[4 2+im; 2-im 5]
    A_huge = copy(A2)
    A_huge[1, 2] = huge
    if T <: Complex
        A_huge[1, 1] += T(0, huge)
        A_huge[2, 2] -= T(0, huge)
    end
    b2 = T <: Real ? T[1, -2] : T[1 + im, -2]
    # the diagonal sets the scale where it is larger than the strict lower triangle
    A_diag = T[4 2.0^-30; 2.0^-30 5] * 2^20
    for (method, ε) in methods
        f, x = cholesky_device(method, A2, b2)
        f_huge, x_huge = cholesky_device(method, A_huge, b2)
        fh_huge, _ = cholesky_host(method, A_huge, b2)
        @test returncode(f_huge) == SUCCESS
        @test f_huge.scale_exponent == fh_huge.scale_exponent == f.scale_exponent
        @test Array(f_huge.factors) == Array(f.factors)
        @test x_huge == x
        @test agrees_with_backslash(A2, x_huge, b2, ε)
        f_diag, x_diag = cholesky_device(method, A_diag, b2)
        fh_diag, _ = cholesky_host(method, A_diag, b2)
        @test returncode(f_diag) == SUCCESS
        @test f_diag.scale_exponent == fh_diag.scale_exponent
        @test agrees_with_backslash(A_diag, x_diag, b2, ε)
    end
    # a matrix of subnormal entries: the exponent is clamped to `-emax`, as on the host, so that
    # `2^-e` stays a float
    A_tiny = A2 * ldexp(one(R), -(exponent(floatmax(R)) + 12))
    method = CholeskyFactorization(TF)
    f_tiny, _ = cholesky_device(method, A_tiny, b2)
    fh_tiny, _ = cholesky_host(method, A_tiny, b2)
    @test returncode(f_tiny) == SUCCESS
    @test f_tiny.scale_exponent == fh_tiny.scale_exponent == -exponent(floatmax(R))
end

# For an IEEE element type, the load kernel takes the largest entry and its exponent from the bits,
# so that a device that flushes subnormals to zero in its arithmetic, as the Apple GPU does, reads
# a subnormal matrix.
# The CPU does not flush, so these rows check the bit path against `exponent` and the host's
# `cholesky_exponent`; the Metal rows of `test/devices/metal_ka.jl` check the flush.
@testset "The bit exponent of the load kernel is exponent, normal and subnormal ($R)" for R in REAL_ELTYPES
    bits_exponent = GeometricSolvers.bits_exponent
    abs_bits = GeometricSolvers.abs_bits
    xs = R[nextfloat(zero(R)), prevfloat(floatmin(R)), floatmin(R), nextfloat(floatmin(R)),
        one(R), prevfloat(floatmax(R)), floatmax(R)]
    append!(xs, (ldexp(one(R), k)
    for k in exponent(nextfloat(zero(R))):exponent(floatmax(R))))
    append!(xs, (3 * nextfloat(zero(R)) * k for k in 1:97))
    append!(xs, abs.(R.(randn(Xoshiro(83), 64))) .*
                ldexp.(one(R), rand(Xoshiro(89), -200:120, 64)))
    for x in filter(x -> isfinite(x) && !iszero(x), xs)
        @test bits_exponent(R, abs_bits(x)) == exponent(x)
        @test bits_exponent(R, abs_bits(-x)) == exponent(x)
    end
end

@testset "DeviceCholesky on a JLArray takes the host's exponent from normal to subnormal scales ($T)" for T in ELTYPES
    allowscalar(false)
    R = real(T)
    emax = exponent(floatmax(R))
    TF = mixed_precision(T)
    W = T <: Real ? T[2 0.3 0; 0.3 2 0.3; 0 0.3 2] :
        T[2 0.3+0.1im 0; 0.3-0.1im 2 0.3im; 0 -0.3im 2]
    b = cholesky_rhs(T, 3, 97)
    tiny = exponent(nextfloat(zero(R)))
    # every scale from the smallest subnormal to the normal range, then a coarse normal sweep
    ks = unique(vcat(collect(tiny:(-emax + 4)), collect((-emax + 4):37:(emax - 2))))
    for k in ks
        A = GeometricSolvers.ldexp2.(W, k)
        e = GeometricSolvers.cholesky_exponent(A)
        for (method, expected) in ((CholeskyFactorization(), e < 0 ? e & ~1 : 0),
            (CholeskyFactorization(TF), e))
            f = factorize!!(dense_factor(method, nothing, JLArray(b), nothing), method,
                JLArray(A))
            fh = factorize!!(dense_factor(method, nothing, b, nothing), method, A)
            @test f.scale_exponent == fh.scale_exponent == expected
        end
    end
end

@testset "DeviceCholesky: factorize!! and ldiv!! are inferred and JET clean ($T)" for T in ELTYPES
    allowscalar(false)
    A = randhpd(T, 6, 10, 73)
    b = JLArray(cholesky_rhs(T, 6, 79))
    for method in (CholeskyFactorization(), CholeskyFactorization(mixed_precision(T)))
        fac0 = dense_factor(method, nothing, b, nothing)
        fac = @inferred factorize!!(fac0, method, JLArray(A))
        @test typeof(fac) === typeof(fac0)
        @test (@inferred ldiv!!(similar(b), fac, b)) isa JLArray{T, 1}
        if JET_WORKS
            @test_opt target_modules = (GeometricSolvers,) factorize!!(fac0, method, JLArray(A))
            @test_opt target_modules = (GeometricSolvers,) ldiv!!(similar(b), fac, b)
        else
            @test_skip JET_WORKS  # issue #8
        end
    end
end

@testset "Newton with CholeskyFactorization on a convex StationarityProblem ($T)" for T in REAL_ELTYPES
    # The Hessian `I + 3 diag(x²)` of `S` is positive definite at every iterate, which strict
    # convexity alone does not give, and its stationary point solves `x + x³ = 1` in each
    # component. A complex state fails in the default automatic differentiation (K28), so the
    # state is real.
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

@testset "A Float32 factor that is not positive definite takes the promotion of Newton ($T)" for T in filter(
    T -> real(T) === Float64, ELTYPES)
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
