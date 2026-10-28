using GPUArraysCore: AbstractGPUArray, allowscalar
using JLArrays: JLArray, JLMatrix, JLVector
using LinearAlgebra: BlasFloat, I, cond, norm, opnorm
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: DenseLU, DenseQR, LUFactorization, NonlinearProblem, QRFactorization
using GeometricSolvers: backsolve!, factor_eltype, factorize!!, init, ldiv!!, returncode

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

allowscalar(false)

function ka_well_conditioned(T::Type, n::Int, seed::Integer = 0x5eed)
    rng = Xoshiro(seed)
    A = randn(rng, T, n, n) + T(2 * n) * I
    b = randn(rng, T, n)
    return A, b
end

function effective_factor_eltype(::Type{T}, ::Type{TF}) where {T, TF}
    TF === Nothing && return T
    F = promote_type(Float32, TF)
    return T <: Complex ? Complex{F} : F
end

@testset "base premise extension to JLArray" begin
    A_cpu = [2.0 1.0; 1.0 3.0]
    b_cpu = [5.0, 7.0]
    x_ref = A_cpu \ b_cpu

    for m in (LUFactorization(), QRFactorization())
        x = JLArray(zeros(Float64, 2))
        b = JLArray(copy(b_cpu))
        prob = NonlinearProblem(x, (r, y, p) -> nothing)
        fac = init(m, prob, x, nothing)
        copyto!(fac.factors, JLArray(A_cpu))
        fac = factorize!!(fac, m, fac.factors)
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
        d = ldiv!!(similar(x), fac, b)
        @test d isa JLVector{Float64}
        @test isapprox(Array(d), x_ref; rtol = 64 * eps(Float64))

        fac_z = init(m, prob, x, nothing)
        fill!(fac_z.factors, 0.0)
        fac_z = factorize!!(fac_z, m, fac_z.factors)
        @test fac_z.info > 0
        @test returncode(fac_z) === SINGULAR
        @test all(isfinite, Array(fac_z.factors))
    end
end

@testset "KA device LU and QR agree with CPU solves on JLArray" begin
    @testset "n = $n" for n in (1, 16)
        for (T, TF) in PRECISION_PAIRS
            for M in (LUFactorization, QRFactorization)
                m = TF === Nothing ? M() : M(TF)
                A_cpu, b_cpu = ka_well_conditioned(T, n)
                κ = Float64(cond(big.(A_cpu), Inf))
                @test κ <= 10.0

                # CPU solve with the same (T, TF) pair
                x_cpu_init = zeros(T, n)
                prob_cpu = NonlinearProblem(x_cpu_init, (r, y, p) -> nothing)
                fac_cpu = init(m, prob_cpu, x_cpu_init, nothing)
                copyto!(fac_cpu.factors, eltype(fac_cpu.factors).(A_cpu))
                fac_cpu = factorize!!(fac_cpu, m, fac_cpu.factors)
                @test fac_cpu.info == 0
                d_cpu = ldiv!!(similar(x_cpu_init), fac_cpu, copy(b_cpu))

                # Device solve on JLArray (in-place J === fac.factors)
                x_dev = JLArray(zeros(T, n))
                b_dev = JLArray(copy(b_cpu))
                prob_dev = NonlinearProblem(x_dev, (r, y, p) -> nothing)
                fac_dev = init(m, prob_dev, x_dev, nothing)
                copyto!(fac_dev.factors, JLArray(eltype(fac_dev.factors).(A_cpu)))
                fac_dev = factorize!!(fac_dev, m, fac_dev.factors)
                @test fac_dev.info == 0
                @test returncode(fac_dev) === SUCCESS
                d_dev = ldiv!!(similar(x_dev), fac_dev, b_dev)
                @test d_dev isa JLVector{T}

                x_h = Array(d_dev)
                TF_eff = effective_factor_eltype(T, TF)
                eps_eff = Float64(eps(real(TF_eff)))

                if TF === Nothing
                    bwd = norm(A_cpu * x_h - b_cpu, Inf) /
                          (opnorm(A_cpu, Inf) * norm(x_h, Inf))
                    @test bwd <= 4 * n * eps_eff
                end

                fwd_cpu = norm(x_h - d_cpu, Inf) / norm(d_cpu, Inf)
                @test fwd_cpu <= 8 * n * eps_eff * κ

                x_ref = big.(A_cpu) \ big.(b_cpu)
                fwd_ref = Float64(norm(big.(x_h) - x_ref, Inf) / norm(x_ref, Inf))
                @test fwd_ref <= 8 * n * eps_eff * κ

                # Out-of-place J !== fac_dev.factors also copies and factors cleanly
                if TF === Nothing
                    J_ext = JLArray(copy(A_cpu))
                    fill!(fac_dev.factors, zero(T))
                    fac_dev = factorize!!(fac_dev, m, J_ext)
                    @test fac_dev.info == 0
                    d_ext = ldiv!!(similar(x_dev), fac_dev, b_dev)
                    @test Array(d_ext) == x_h
                end
            end
        end
    end

    @testset "scaled mixed-precision solves on JLArray" begin
        f64_types = filter(T -> real(T) === Float64, ELTYPES)
        scales = ((1e-40, 1e-40), (1e39, 1e39))
        for T in f64_types
            for M in (LUFactorization, QRFactorization)
                for (sA, sb) in scales
                    n = 8
                    m = M(Float32)
                    A0, b0 = ka_well_conditioned(T, n)
                    A_cpu = T(sA) .* A0
                    b_cpu = T(sb) .* b0
                    κ = Float64(cond(big.(A0), Inf))

                    x_dev = JLArray(zeros(T, n))
                    b_dev = JLArray(copy(b_cpu))
                    prob_dev = NonlinearProblem(x_dev, (r, y, p) -> nothing)
                    fac_dev = init(m, prob_dev, x_dev, nothing)
                    fac_dev = factorize!!(fac_dev, m, JLArray(A_cpu))
                    @test fac_dev.info == 0
                    @test fac_dev.scale_exponent != 0

                    d_dev = ldiv!!(similar(x_dev), fac_dev, b_dev)
                    x_h = Array(d_dev)
                    @test all(isfinite, x_h)

                    x_ref = big.(A_cpu) \ big.(b_cpu)
                    rel = Float64(norm(big.(x_h) - x_ref, Inf) / norm(x_ref, Inf))
                    @test rel <= 8 * n * Float64(eps(Float32)) * κ

                    # Same-precision ldiv!! (T == TF) when factorize!! scaled a wider Jacobian (e != 0)
                    F = T <: Complex ? ComplexF32 : Float32
                    sb_f32 = sA < 1 ? 1e-20 : 1e20
                    b_f32_cpu = F.(T(sb_f32) .* b0)
                    x_f32 = JLArray(zeros(F, n))
                    b_f32 = JLArray(copy(b_f32_cpu))
                    prob_f32 = NonlinearProblem(x_f32, (r, y, p) -> nothing)
                    m_f32 = M()
                    fac_f32 = init(m_f32, prob_f32, x_f32, nothing)
                    fac_f32 = factorize!!(fac_f32, m_f32, JLArray(A_cpu))
                    @test fac_f32.info == 0
                    @test fac_f32.scale_exponent != 0
                    d_f32 = ldiv!!(similar(x_f32), fac_f32, b_f32)
                    x_f32_h = Array(d_f32)
                    @test all(isfinite, x_f32_h)
                    x_f32_ref = big.(A_cpu) \ big.(b_f32_cpu)
                    rel_f32 = Float64(norm(big.(x_f32_h) - x_f32_ref, Inf) /
                                      norm(x_f32_ref, Inf))
                    @test rel_f32 <= 8 * n * Float64(eps(Float32)) * κ
                end
            end
        end
    end
end

@testset "disambiguation and scalar-indexing absence on JLArray" begin
    @testset "dispatch to src/linear/ka/ for T = $T" for T in ELTYPES
        x = JLArray(zeros(T, 4))
        prob = NonlinearProblem(x, (r, y, p) -> nothing)

        fac_lu = init(LUFactorization(), prob, x, nothing)
        lu_sig = (typeof(fac_lu), LUFactorization{Nothing}, JLMatrix{T})
        m_lu_fac = which(factorize!!, lu_sig)
        m_lu_bs = which(backsolve!, (typeof(fac_lu), JLVector{T}))
        @test endswith(replace(String(m_lu_fac.file), '\\' => '/'), "src/linear/ka/lu.jl")
        @test endswith(replace(String(m_lu_bs.file), '\\' => '/'), "src/linear/ka/lu.jl")

        fac_qr = init(QRFactorization(), prob, x, nothing)
        qr_sig = (typeof(fac_qr), QRFactorization{Nothing}, JLMatrix{T})
        m_qr_fac = which(factorize!!, qr_sig)
        m_qr_bs = which(backsolve!, (typeof(fac_qr), JLVector{T}))
        @test endswith(replace(String(m_qr_fac.file), '\\' => '/'), "src/linear/ka/qr.jl")
        @test endswith(replace(String(m_qr_bs.file), '\\' => '/'), "src/linear/ka/qr.jl")
    end

    @testset "no scalar indexing and type inference in init, factorize!!, ldiv!!" begin
        for (T, TF) in PRECISION_PAIRS
            for M in (LUFactorization, QRFactorization)
                m = TF === Nothing ? M() : M(TF)
                A_cpu, b_cpu = ka_well_conditioned(T, 6)
                x_dev = JLArray(zeros(T, 6))
                b_dev = JLArray(copy(b_cpu))
                prob_dev = NonlinearProblem(x_dev, (r, y, p) -> nothing)
                fac = @inferred init(m, prob_dev, x_dev, nothing)
                fac = @inferred factorize!!(fac, m, JLArray(A_cpu))
                @test fac.info == 0
                d = @inferred ldiv!!(similar(x_dev), fac, b_dev)
                @test all(isfinite, Array(d))
                if TF === Nothing
                    fac_sc = fac isa DenseLU ?
                             DenseLU(
                        fac.factors, fac.ipiv, fac.rhs, fac.info, 3, fac.status) :
                             DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, fac.info, 3)
                    d_sc = ldiv!!(similar(x_dev), fac_sc, b_dev)
                    @test isapprox(Array(d_sc), Array(d) ./ real(T)(8); rtol = 8 *
                                                                               eps(real(T)))
                end
            end
        end
    end
end

@testset "pivot search and overflow-safe scaling on JLArray" begin
    @testset "tiny diagonal requiring row exchange in every column, T = $T" for T in ELTYPES
        n = 6
        RT = real(T)
        tiny = eps(RT)^2
        # Permutation matrix shifted by 1 with tiny diagonal entries
        A_cpu = zeros(T, n, n)
        for k in 1:n
            A_cpu[k, k] = T(tiny)
            row = k < n ? k + 1 : 1
            A_cpu[row, k] = T(2) + T(k) / T(n)
        end
        b_cpu = T[T(k) for k in 1:n]
        x_ref = big.(A_cpu) \ big.(b_cpu)
        κ = Float64(cond(big.(A_cpu), Inf))

        for m in (LUFactorization(), QRFactorization())
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
            fac = init(m, prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_cpu))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            d = ldiv!!(similar(x_dev), fac, JLArray(b_cpu))
            rel = Float64(norm(big.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf))
            @test rel <= 8 * n * Float64(eps(RT)) * κ
        end
    end

    @testset "purely imaginary and imaginary-dominant pivots via abs1" begin
        for T in filter(T -> T <: Complex, ELTYPES)
            RT = real(T)
            n = 4
            A_real, b_real = ka_well_conditioned(RT, n)
            A_real[1, 1] = zero(RT)
            A_real[2, 1] = RT(4)
            A_imag = T.(im .* A_real)
            b_imag = T.(b_real .+ im .* reverse(b_real))

            # Second matrix where |Re(A[1,1])| > |Re(A[2,1])| == 0 but |A[2,1]| == 4 >> |A[1,1]|
            A_dom = copy(A_imag)
            A_dom[1, 1] = T(eps(RT), zero(RT))
            A_dom[2, 1] = T(zero(RT), RT(4))

            for A_cpu in (A_imag, A_dom)
                for m in (LUFactorization(), QRFactorization())
                    x_ref = big.(A_cpu) \ big.(b_imag)
                    κ = Float64(cond(big.(A_cpu), Inf))
                    x_dev = JLArray(zeros(T, n))
                    prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
                    fac = init(m, prob, x_dev, nothing)
                    copyto!(fac.factors, JLArray(A_cpu))
                    fac = factorize!!(fac, m, fac.factors)
                    @test fac.info == 0
                    d = ldiv!!(similar(x_dev), fac, JLArray(b_imag))
                    rel = Float64(norm(big.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf))
                    @test rel <= 8 * n * Float64(eps(RT)) * κ
                end
            end
        end
    end

    @testset "overflow-safe column scaling in QR and multiplier in LU, T = $T" for T in ELTYPES
        RT = real(T)
        n = 4
        A0, b0 = ka_well_conditioned(T, n)
        big_col = RT === Float32 ? RT(1.0f20) : RT(1e200)
        A_col = copy(A0)
        A_col[:, 1] .*= big_col
        b_col = copy(b0)

        for m in (LUFactorization(), QRFactorization())
            x_ref = big.(A_col) \ big.(b_col)
            # Column-equilibrated condition number for column-scaled QR/LU
            A_eq = big.(A0)
            κ = Float64(cond(A_eq, Inf))
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
            fac = init(m, prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_col))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            @test all(isfinite, Array(fac.factors))
            d = ldiv!!(similar(x_dev), fac, JLArray(b_col))
            x_h = Array(d)
            @test all(isfinite, x_h)
            # Check componentwise/column-scaled residual against BigFloat reference
            err_col1 = Float64(abs(big(x_h[1]) - x_ref[1]) / abs(x_ref[1]))
            num_rest = norm(big.(x_h[2:end]) - x_ref[2:end], Inf)
            den_rest = norm(x_ref[2:end], Inf)
            err_rest = Float64(num_rest / den_rest)
            @test err_col1 <= 16 * n * Float64(eps(RT)) * κ
            @test err_rest <= 16 * n * Float64(eps(RT)) * κ
        end

        # Direct unscaled factorize!! on 1.0f20 / 1e200 column
        for m in (LUFactorization(), QRFactorization())
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
            fac = init(m, prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_col))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            @test all(isfinite, Array(fac.factors))
            if m isa QRFactorization
                @test all(isfinite, Array(fac.τ))
            end
        end

        # Near-floatmax (0.6 * floatmax(RT)) reflector, trailing apply, and solve without intermediate overflow
        s_hi = RT(0.6) * floatmax(RT)
        for (A_hi, x_hi) in (
            (T[T(s_hi) zero(T); zero(T) T(s_hi)], T[one(T), one(T)]),
            (T[T(s_hi) T(s_hi); zero(T) T(s_hi)], T[one(T), zero(T)]),
            (T[T(s_hi) zero(T); T(s_hi) T(s_hi)], T[one(T), zero(T)]),
            (T[T(s_hi) zero(T); T(s_hi) T(s_hi)], T[T(RT(0.5)), T(RT(0.5))])
        )
            b_hi = A_hi * x_hi
            for m in (LUFactorization(), QRFactorization())
                x2 = JLArray(zeros(T, 2))
                prob2 = NonlinearProblem(x2, (r, y, p) -> nothing)
                fac_hi = init(m, prob2, x2, nothing)
                fac_hi = factorize!!(fac_hi, m, JLArray(A_hi))
                @test returncode(fac_hi) === SUCCESS
                @test all(isfinite, Array(fac_hi.factors))
                if m isa QRFactorization
                    @test all(isfinite, Array(fac_hi.τ))
                end
                d_hi = ldiv!!(similar(x2), fac_hi, JLArray(b_hi))
                @test all(isfinite, Array(d_hi))
                @test isapprox(Array(d_hi), x_hi; rtol = sqrt(eps(RT)))
            end
        end

        # Complex overflow-safe multiplier in LU when pivot and subdiagonal are near floatmax
        if T <: Complex
            big_val = floatmax(RT) / RT(8)
            a11 = T(big_val, big_val)
            a12 = T(big_val / RT(2), -big_val / RT(4))
            a21 = T(big_val / RT(2), big_val / RT(2))
            a22 = T(big_val, big_val / RT(2))
            A_big = T[a11 a12; a21 a22]
            b_big = T[T(big_val, zero(RT)), T(zero(RT), big_val)]
            x_ref = big.(A_big) \ big.(b_big)
            x_dev = JLArray(zeros(T, 2))
            prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
            fac = init(LUFactorization(), prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_big))
            fac = factorize!!(fac, LUFactorization(), fac.factors)
            @test fac.info == 0
            @test all(isfinite, Array(fac.factors))
            d = copy(JLArray(b_big))
            backsolve!(fac, d)
            @test all(isfinite, Array(d))
            rel = Float64(norm(big.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf))
            @test rel <= 32 * Float64(eps(RT))
        end
    end
end

@testset "singular and non-finite inputs on JLArray: $T" for T in ELTYPES
    for M in (LUFactorization, QRFactorization)
        RT = real(T)
        m = M()

        # (a) n x n zero matrix -> info == 1 and finite factors
        n = 4
        x_dev = JLArray(zeros(T, n))
        prob = NonlinearProblem(x_dev, (r, y, p) -> nothing)
        fac_zero = init(m, prob, x_dev, nothing)
        fill!(fac_zero.factors, zero(T))
        fac_zero = factorize!!(fac_zero, m, fac_zero.factors)
        @test fac_zero.info == 1
        @test returncode(fac_zero) === SINGULAR
        @test all(isfinite, Array(fac_zero.factors))

        # (b) Exact zero column k = 2 (and k = 3 also zero so first zero is 2, not 3)
        A_col2 = Matrix{T}(I, n, n)
        A_col2[:, 2] .= zero(T)
        A_col2[:, 3] .= zero(T)
        fac_col2 = init(m, prob, x_dev, nothing)
        copyto!(fac_col2.factors, JLArray(A_col2))
        fac_col2 = factorize!!(fac_col2, m, fac_col2.factors)
        @test fac_col2.info == 2
        @test all(isfinite, Array(fac_col2.factors))

        # (b2) Zero column k = 2 followed by nonzero columns 3 and 4 -> info stays 2
        A_col2_nz = Matrix{T}(I, n, n)
        A_col2_nz[:, 2] .= zero(T)
        fac_col2_nz = init(m, prob, x_dev, nothing)
        copyto!(fac_col2_nz.factors, JLArray(A_col2_nz))
        fac_col2_nz = factorize!!(fac_col2_nz, m, fac_col2_nz.factors)
        @test fac_col2_nz.info == 2
        @test all(isfinite, Array(fac_col2_nz.factors))

        # (c) 2 x 2 matrix with two equal rows (scaled and random draws) -> info == 2 for LU
        x2 = JLArray(zeros(T, 2))
        prob2 = NonlinearProblem(x2, (r, y, p) -> nothing)
        for scale in (one(RT), floatmax(RT) / RT(8))
            A_eq = T[T(scale) T(2 * scale); T(scale) T(2 * scale)]
            fac_eq = init(m, prob2, x2, nothing)
            copyto!(fac_eq.factors, JLArray(A_eq))
            fac_eq = factorize!!(fac_eq, m, fac_eq.factors)
            if M === LUFactorization
                @test fac_eq.info == 2
                @test returncode(fac_eq) === SINGULAR
            end
            @test all(isfinite, Array(fac_eq.factors))
        end
        rng_eq = Xoshiro(42)
        for _ in 1:20
            r_eq = randn(rng_eq, T, 2)
            A_eq = T[r_eq[1] r_eq[2]; r_eq[1] r_eq[2]]
            fac_eq = init(m, prob2, x2, nothing)
            copyto!(fac_eq.factors, JLArray(A_eq))
            fac_eq = factorize!!(fac_eq, m, fac_eq.factors)
            if M === LUFactorization
                @test fac_eq.info == 2
                @test returncode(fac_eq) === SINGULAR
            end
            @test all(isfinite, Array(fac_eq.factors))
        end

        # (d) Non-finite inputs (NaN and Inf) do not throw
        for bad_val in (RT(NaN), RT(Inf))
            A_bad = Matrix{T}(I, n, n)
            A_bad[2, 2] = T(bad_val)
            fac_bad = init(m, prob, x_dev, nothing)
            copyto!(fac_bad.factors, JLArray(A_bad))
            fac_bad = factorize!!(fac_bad, m, fac_bad.factors)
            d_bad = ldiv!!(similar(x_dev), fac_bad, JLArray(ones(T, n)))
            @test d_bad isa JLVector{T}
        end

        # (e) 0 x 0 empty system does not launch a 0-range kernel or throw
        x0 = JLArray(zeros(T, 0))
        prob0 = NonlinearProblem(x0, (r, y, p) -> nothing)
        fac0 = init(m, prob0, x0, nothing)
        fac0 = factorize!!(fac0, m, fac0.factors)
        @test fac0.info == 0
        d0 = ldiv!!(similar(x0), fac0, JLArray(zeros(T, 0)))
        @test length(d0) == 0
    end
end
