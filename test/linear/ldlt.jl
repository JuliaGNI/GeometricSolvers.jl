using Adapt
using GeometricSolvers
using GeometricSolvers: DenseLDLT, DeviceLDLT, StaticLDLT, backsolve, backsolve!
using GeometricSolvers: dense_factor, exact_solve, factor_diagnostics, factor_eltype
using GeometricSolvers: factorize!!, ldiv!!, returncode, rnorm
using GPUArraysCore: @allowscalar, allowscalar
using JLArrays: JLArray
using LinearAlgebra: Diagonal, Hermitian, I, Symmetric, cond, dot, norm, opnorm, qr
using Random: Xoshiro, randn
using StaticArrays: MMatrix, SMatrix, SVector
using Test

include("../helpers/matrix.jl")

# Seeded symmetric or Hermitian indefinite matrix with identically zero diagonal (A[i, i] == 0).
function zero_diag_indefinite(::Type{T}, n::Int, seed::Int) where {T}
    rng = Xoshiro(seed)
    R = real(T)
    M = randn(rng, T, n, n)
    A = T <: Real ? (M + transpose(M)) / R(2) : (M + M') / R(2)
    for i in 1:n
        A[i, i] = zero(T)
    end
    for i in 1:2:(n - 1)
        A[i, i + 1] += T <: Real ? R(2) : T(2 + im)
        A[i + 1, i] = T <: Real ? A[i, i + 1] : conj(A[i, i + 1])
    end
    return A
end

# Seeded complex-symmetric (non-Hermitian: A == transpose(A) and A != A') indefinite matrix.
function complex_symmetric_indefinite(::Type{T}, n::Int, seed::Int) where {T <: Complex}
    rng = Xoshiro(seed)
    R = real(T)
    M = randn(rng, T, n, n)
    A = (M + transpose(M)) / R(2)
    for i in 1:2:(n - 1)
        A[i, i] = zero(T)
        A[i + 1, i + 1] = zero(T)
        A[i, i + 1] += T(2 + im)
        A[i + 1, i] = A[i, i + 1]
    end
    return A
end

# Indefinite symmetric/Hermitian matrix with controlled 2-norm condition number κ and mixed signs.
function randhermitian(::Type{T}, n::Int, κ::Real, seed::Int) where {T}
    rng = Xoshiro(seed)
    R = real(T)
    G = randn(rng, T, n, n)
    Q_f = qr(G)
    Q = Matrix(Q_f.Q)
    exps = range(zero(R), log10(R(κ)); length = n)
    vals = [ifelse(isodd(i), one(R), -one(R)) * exp10(exps[i]) for i in 1:n]
    # Permute eigenvalues so leading submatrices can be indefinite and trigger 2x2 pivots
    perm = [isodd(i) ? cld(i, 2) : (n - div(i, 2) + 1) for i in 1:n]
    d = vals[perm]
    A = Q * Diagonal(T.(d)) * Q'
    if T <: Real
        return Matrix(Symmetric((A + transpose(A)) / R(2)))
    else
        return Matrix(Hermitian((A + A') / R(2)))
    end
end

function backward_error(A::AbstractMatrix, x::AbstractVector, b::AbstractVector)
    A_h = Matrix(A)
    x_h = Vector(x)
    b_h = Vector(b)
    denom = opnorm(A_h, 2) * norm(x_h, 2) + norm(b_h, 2)
    iszero(denom) && return zero(real(eltype(A)))
    return norm(A_h * x_h - b_h, 2) / denom
end

@testset "LDLTFactorization method and factor types" begin
    @test LDLTFactorization() === LDLTFactorization{Nothing}()
    @test LDLTFactorization(Float32) === LDLTFactorization{Float32}()
    @test LDLTFactorization(TF32()) === LDLTFactorization{TF32}()
    @test isbits(LDLTFactorization())
    @test isbits(LDLTFactorization(Float32))
    @test isbits(LDLTFactorization(TF32()))

    for T in ELTYPES
        @test factor_eltype(LDLTFactorization(), T) === T
        expected_f32 = T <: Real ? Float32 : ComplexF32
        @test factor_eltype(LDLTFactorization(Float32), T) === expected_f32
    end

    A2 = SMatrix{2, 2, Float64}(1.0, 2.0, 2.0, 1.0)
    @test_throws MethodError factorize!!(nothing, LDLTFactorization(TF32()), A2)
end

@testset "Check 4: Zero-diagonal symmetric and Hermitian indefinite matrices" begin
    allowscalar(false)
    for T in ELTYPES
        R = real(T)
        if T <: Real
            A_z = SMatrix{2, 2, T}(zero(T), one(T), one(T), zero(T))
        else
            A_z = SMatrix{2, 2, T}(zero(T), T(1 - im), T(1 + im), zero(T))
        end
        b_z = SVector{2, T}(T(2), T(-3))

        # R1 StaticLDLT
        fac_s = factorize!!(nothing, LDLTFactorization(), A_z)
        @test fac_s isa StaticLDLT{2, T, 4}
        @test isbits(fac_s)
        @test fac_s.info == Int32(0)
        @test fac_s.hermitian === true
        @test fac_s.ipiv[1] < Int32(0) && fac_s.ipiv[2] < Int32(0)
        @test returncode(fac_s) == SUCCESS
        @test exact_solve(fac_s) === true
        @test factor_diagnostics(nothing, fac_s, b_z, rnorm(b_z)) === (Int32(-1), R(NaN))

        x_s = ldiv!!(nothing, fac_s, b_z)
        @test x_s isa SVector{2, T}
        @test backward_error(A_z, x_s, b_z) <= 10 * 2 * eps(R)

        # Also test near-zero 1e-12 top-left entry that triggers 2x2 Bunch-Kaufman pivot
        δ = R(1e-12)
        if T <: Real
            A_near = SMatrix{2, 2, T}(T(δ), one(T), one(T), zero(T))
        else
            A_near = SMatrix{2, 2, T}(T(δ), T(1 - im), T(1 + im), zero(T))
        end
        fac_near = factorize!!(nothing, LDLTFactorization(), A_near)
        @test fac_near.ipiv[1] < Int32(0) && fac_near.ipiv[2] < Int32(0)
        x_near = ldiv!!(nothing, fac_near, b_z)
        @test backward_error(A_near, x_near, b_z) <= 10 * 2 * eps(R)

        # R3 Host DenseLDLT (LAPACK sytrf! / hetrf!)
        A_d = Matrix(A_z)
        b_d = Vector(b_z)
        d_d = similar(b_d)
        fac_d = dense_factor(LDLTFactorization(), nothing, b_d, nothing)
        @test fac_d isa DenseLDLT
        fac_d = factorize!!(fac_d, LDLTFactorization(), A_d)
        @test fac_d.info == 0
        @test fac_d.hermitian === true
        @test returncode(fac_d) == SUCCESS
        @test exact_solve(fac_d) === true
        @test factor_diagnostics(d_d, fac_d, b_d, rnorm(b_d)) === (Int32(-1), R(NaN))
        ldiv!!(d_d, fac_d, b_d)
        @test backward_error(A_d, d_d, b_d) <= 10 * 2 * eps(R)

        # R3 Device DeviceLDLT (JLArray)
        A_jl = JLArray(A_d)
        b_jl = JLArray(b_d)
        d_jl = similar(b_jl)
        fac_jl = dense_factor(LDLTFactorization(), nothing, b_jl, nothing)
        @test fac_jl isa DeviceLDLT
        fac_jl = factorize!!(fac_jl, LDLTFactorization(), A_jl)
        @test fac_jl.info == 0
        @test fac_jl.hermitian === true
        @test returncode(fac_jl) == SUCCESS
        @test exact_solve(fac_jl) === true
        @test factor_diagnostics(d_jl, fac_jl, b_jl, rnorm(b_jl)) === (Int32(-1), R(NaN))
        ldiv!!(d_jl, fac_jl, b_jl)
        @test backward_error(A_d, Array(d_jl), b_d) <= 10 * 2 * eps(R)

        # Seeded zero-diagonal matrices of sizes n in (4, 16, 64)
        for n in (4, 16, 64)
            A_n = zero_diag_indefinite(T, n, 400 + n)
            b_n = randn(Xoshiro(500 + n), T, n)
            if n <= 16
                As = SMatrix{n, n, T, n * n}(A_n)
                bs = SVector{n, T}(b_n)
                fs = factorize!!(nothing, LDLTFactorization(), As)
                @test fs.info == Int32(0)
                xs = ldiv!!(nothing, fs, bs)
                @test backward_error(A_n, xs, b_n) <= 10 * n * eps(R)
            end

            fd = dense_factor(LDLTFactorization(), nothing, b_n, nothing)
            fd = factorize!!(fd, LDLTFactorization(), A_n)
            @test fd.info == 0
            xd = similar(b_n)
            ldiv!!(xd, fd, b_n)
            @test backward_error(A_n, xd, b_n) <= 10 * n * eps(R)

            Ajl = JLArray(A_n)
            bjl = JLArray(b_n)
            fjl = dense_factor(LDLTFactorization(), nothing, bjl, nothing)
            fjl = factorize!!(fjl, LDLTFactorization(), Ajl)
            @test fjl.info == 0
            xjl = similar(bjl)
            ldiv!!(xjl, fjl, bjl)
            @test backward_error(A_n, Array(xjl), b_n) <= 10 * n * eps(R)
        end
    end
end

@testset "Check 5: Bunch-Kaufman stability across condition numbers" begin
    allowscalar(false)
    for T in ELTYPES
        R = real(T)
        kappas = sizeof(R) == 4 ? (10, 10_000) : (10, 10_000, 100_000_000)
        for κ in kappas, n in (4, 16, 64)

            A = randhermitian(T, n, κ, 700 + n + round(Int, log10(κ)))
            b = randn(Xoshiro(800 + n), T, n)
            bound = 16 * n * eps(R)

            if n <= 16
                As = SMatrix{n, n, T, n * n}(A)
                bs = SVector{n, T}(b)
                fs = factorize!!(nothing, LDLTFactorization(), As)
                @test fs.info == Int32(0)
                xs = ldiv!!(nothing, fs, bs)
                @test backward_error(A, xs, b) <= bound
            end

            fd = dense_factor(LDLTFactorization(), nothing, b, nothing)
            fd = factorize!!(fd, LDLTFactorization(), A)
            @test fd.info == 0
            xd = similar(b)
            ldiv!!(xd, fd, b)
            @test backward_error(A, xd, b) <= bound

            Ajl = JLArray(A)
            bjl = JLArray(b)
            fjl = dense_factor(LDLTFactorization(), nothing, bjl, nothing)
            fjl = factorize!!(fjl, LDLTFactorization(), Ajl)
            @test fjl.info == 0
            xjl = similar(bjl)
            ldiv!!(xjl, fjl, bjl)
            @test backward_error(A, Array(xjl), b) <= bound
        end
    end
end

@testset "Complex symmetric (L D L^T) vs complex Hermitian (L D L^H)" begin
    allowscalar(false)
    for T in filter(T -> T <: Complex, ELTYPES)
        R = real(T)
        for n in (2, 5, 12)
            A_sym = complex_symmetric_indefinite(T, n, 900 + n)
            b = randn(Xoshiro(950 + n), T, n)

            As = SMatrix{n, n, T, n * n}(A_sym)
            bs = SVector{n, T}(b)
            fs = factorize!!(nothing, LDLTFactorization(), As)
            @test fs.info == Int32(0)
            @test fs.hermitian === false
            xs = ldiv!!(nothing, fs, bs)
            @test backward_error(A_sym, xs, b) <= 16 * n * eps(R)

            fd = dense_factor(LDLTFactorization(), nothing, b, nothing)
            fd = factorize!!(fd, LDLTFactorization(), A_sym)
            @test fd.info == 0
            @test fd.hermitian === false
            xd = similar(b)
            ldiv!!(xd, fd, b)
            @test backward_error(A_sym, xd, b) <= 16 * n * eps(R)

            Ajl = JLArray(A_sym)
            bjl = JLArray(b)
            fjl = dense_factor(LDLTFactorization(), nothing, bjl, nothing)
            fjl = factorize!!(fjl, LDLTFactorization(), Ajl)
            @test fjl.info == 0
            @test fjl.hermitian === false
            xjl = similar(bjl)
            ldiv!!(xjl, fjl, bjl)
            @test backward_error(A_sym, Array(xjl), b) <= 16 * n * eps(R)
        end

        # D5: exact zero imaginary diagonal required, and 64 * eps(R) off-diagonal tolerance
        A_bad_diag = T[1 + R(1 // 2) * im 2 - im; 2 + im 3]
        fs_bad = factorize!!(nothing, LDLTFactorization(), SMatrix{2, 2, T, 4}(A_bad_diag))
        @test fs_bad.hermitian === false
        fd_bad = dense_factor(LDLTFactorization(), nothing, ones(T, 2), nothing)
        fd_bad = factorize!!(fd_bad, LDLTFactorization(), A_bad_diag)
        @test fd_bad.hermitian === false
        fjl_bad = dense_factor(LDLTFactorization(), nothing, JLArray(ones(T, 2)), nothing)
        fjl_bad = factorize!!(fjl_bad, LDLTFactorization(), JLArray(A_bad_diag))
        @test fjl_bad.hermitian === false

        noise = R(8) * eps(R)
        A_near_real = T[2 1 + noise * im; 1 + noise * im 3]
        fs_nr = factorize!!(nothing, LDLTFactorization(), SMatrix{2, 2, T, 4}(A_near_real))
        @test fs_nr.hermitian === true
        fd_nr = dense_factor(LDLTFactorization(), nothing, ones(T, 2), nothing)
        fd_nr = factorize!!(fd_nr, LDLTFactorization(), A_near_real)
        @test fd_nr.hermitian === true
        fjl_nr = dense_factor(LDLTFactorization(), nothing, JLArray(ones(T, 2)), nothing)
        fjl_nr = factorize!!(fjl_nr, LDLTFactorization(), JLArray(A_near_real))
        @test fjl_nr.hermitian === true
    end
end

@testset "Mixed-precision LDLTFactorization(Float32) and exponent scaling" begin
    allowscalar(false)
    for T in filter(T -> sizeof(real(T)) == 8, ELTYPES)
        R = real(T)
        m32 = LDLTFactorization(Float32)
        for scale_exp in (0, 50, -50)
            s = ldexp(one(R), scale_exp)
            A0 = zero_diag_indefinite(T, 4, 1100)
            A = s .* A0
            b = s .* randn(Xoshiro(1101), T, 4)

            As = SMatrix{4, 4, T, 16}(A)
            bs = SVector{4, T}(b)
            fs = factorize!!(nothing, m32, As)
            @test fs.info == Int32(0)
            xs = ldiv!!(nothing, fs, bs)
            @test backward_error(A, xs, b) <= 32 * 4 * eps(Float32)
            @test iszero(ldiv!!(nothing, fs, zero(bs)))

            fd = dense_factor(m32, nothing, b, nothing)
            fd = factorize!!(fd, m32, A)
            @test fd.info == 0
            xd = similar(b)
            ldiv!!(xd, fd, b)
            @test backward_error(A, xd, b) <= 32 * 4 * eps(Float32)
            ldiv!!(xd, fd, zero(b))
            @test iszero(xd)

            Ajl = JLArray(A)
            bjl = JLArray(b)
            fjl = dense_factor(m32, nothing, bjl, nothing)
            fjl = factorize!!(fjl, m32, Ajl)
            @test fjl.info == 0
            xjl = similar(bjl)
            ldiv!!(xjl, fjl, bjl)
            @test backward_error(A, Array(xjl), b) <= 32 * 4 * eps(Float32)
        end
    end
end

@testset "Singular, non-finite, 1x1 swapped, and BigFloat edge cases" begin
    allowscalar(false)
    for T in ELTYPES
        R = real(T)
        # 1x1 swapped pivot case: A = [0 0 1; 0 2 0; 1 0 10] has |a_11| = 0, r = 3, |a_33| = 10 >= α
        As = SMatrix{3, 3, T, 9}(0, 0, 1, 0, 2, 0, 1, 0, 10)
        A_swap1 = Matrix(As)
        b_swap1 = T[3, -4, 5]
        bs = SVector{3, T}(b_swap1)
        fs = factorize!!(nothing, LDLTFactorization(), As)
        @test fs.info == Int32(0)
        @test fs.ipiv[1] == Int32(3)
        xs = ldiv!!(nothing, fs, bs)
        @test backward_error(A_swap1, xs, b_swap1) <= 10 * 3 * eps(R)

        A_jl1 = JLArray(A_swap1)
        b_jl1 = JLArray(b_swap1)
        fjl = dense_factor(LDLTFactorization(), nothing, b_jl1, nothing)
        fjl = factorize!!(fjl, LDLTFactorization(), A_jl1)
        @test fjl.info == 0
        @test Array(fjl.ipiv)[1] == Int32(3)
        xjl = similar(b_jl1)
        ldiv!!(xjl, fjl, b_jl1)
        @test backward_error(A_swap1, Array(xjl), b_swap1) <= 10 * 3 * eps(R)

        # Singular zero matrix
        Z_s = zeros(SMatrix{3, 3, T, 9})
        fz_s = factorize!!(nothing, LDLTFactorization(), Z_s)
        @test fz_s.info > Int32(0)
        @test returncode(fz_s) == SINGULAR

        Z_d = zeros(T, 3, 3)
        bz_d = ones(T, 3)
        fz_d = dense_factor(LDLTFactorization(), nothing, bz_d, nothing)
        fz_d = factorize!!(fz_d, LDLTFactorization(), Z_d)
        @test fz_d.info > 0
        @test returncode(fz_d) == SINGULAR

        Z_jl = JLArray(Z_d)
        bz_jl = JLArray(bz_d)
        fz_jl = dense_factor(LDLTFactorization(), nothing, bz_jl, nothing)
        fz_jl = factorize!!(fz_jl, LDLTFactorization(), Z_jl)
        @test fz_jl.info > 0
        @test returncode(fz_jl) == SINGULAR

        # Non-finite input to DenseLDLT refuses before LAPACK and sets info = 1 (catches MU8)
        A_nf = Matrix{T}(I, 3, 3)
        A_nf[2, 2] = T(NaN)
        fnf = dense_factor(LDLTFactorization(), nothing, bz_d, nothing)
        fill!(fnf.ipiv, 77)
        fnf = factorize!!(fnf, LDLTFactorization(), A_nf)
        @test fnf.info == 1
        @test returncode(fnf) == SINGULAR
        @test all(==(77), fnf.ipiv)
    end

    # BigFloat on StaticLDLT and DenseLDLT (generic fallback)
    As_bf = SMatrix{3, 3, BigFloat, 9}(0, 2, -1, 2, 1, 3, -1, 3, -2)
    A_bf = Matrix(As_bf)
    b_bf = BigFloat[1, -2, 4]
    bs_bf = SVector{3, BigFloat}(b_bf)
    fs_bf = factorize!!(nothing, LDLTFactorization(), As_bf)
    @test fs_bf.info == Int32(0)
    xs_bf = ldiv!!(nothing, fs_bf, bs_bf)
    @test norm(A_bf * Vector(xs_bf) - b_bf) <= 10 * 3 * eps(BigFloat) * norm(b_bf)

    fd_bf = dense_factor(LDLTFactorization(), nothing, b_bf, nothing)
    fd_bf = factorize!!(fd_bf, LDLTFactorization(), A_bf)
    @test fd_bf.info == 0
    xd_bf = similar(b_bf)
    ldiv!!(xd_bf, fd_bf, b_bf)
    @test norm(A_bf * xd_bf - b_bf) <= 10 * 3 * eps(BigFloat) * norm(b_bf)

    # Zero allocations for R1 StaticLDLT and R3 DenseLDLT
    A_f64 = SMatrix{3, 3, Float64, 9}(0.0, 2.0, -1.0, 2.0, 1.0, 0.5, -1.0, 0.5, -3.0)
    b_f64 = SVector{3, Float64}(1.0, -2.0, 0.5)
    fac64 = factorize!!(nothing, LDLTFactorization(), A_f64)
    ldiv!!(nothing, fac64, b_f64)
    if Base.JLOptions().check_bounds != 1
        # skipped where a run forces `--check-bounds=yes`
        @test (@allocated factorize!!(nothing, LDLTFactorization(), A_f64)) == 0
        @test (@allocated ldiv!!(nothing, fac64, b_f64)) == 0
    end

    Ad_64 = Matrix(A_f64)
    bd_64 = Vector(b_f64)
    dd_64 = similar(bd_64)
    fd_64 = dense_factor(LDLTFactorization(), nothing, bd_64, nothing)
    factorize!!(fd_64, LDLTFactorization(), Ad_64)
    ldiv!!(dd_64, fd_64, bd_64)
    if Base.JLOptions().check_bounds != 1
        # skipped where a run forces `--check-bounds=yes`
        @test (@allocated factorize!!(fd_64, LDLTFactorization(), Ad_64)) == 0
        @test (@allocated ldiv!!(dd_64, fd_64, bd_64)) == 0
    end
end
