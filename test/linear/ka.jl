using GPUArraysCore: AbstractGPUArray, allowscalar
using JLArrays: JLArray, JLMatrix, JLVector
using LinearAlgebra: BlasFloat, Diagonal, I, cond, norm, opnorm
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: DenseLU, DenseQR, DenseSVD, LUFactorization, NonlinearProblem,
                        QRFactorization, SVDFactorization
using GeometricSolvers: JACOBI_MAX_SWEEPS, backsolve!, factor_eltype, factorize!!, init,
                        jacobi_pair, jacobi_rotation, jacobi_tolerance, ldiv!!, returncode

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

ka_integers(::Type{T}, n::Int) where {T} = T[i + j for i in 1:n, j in 1:n]
ka_noop(r, y, p) = nothing

@testset "base premise extension to JLArray" begin
    A_cpu = [2.0 1.0; 1.0 3.0]
    b_cpu = [5.0, 7.0]
    x_ref = A_cpu \ b_cpu

    for m in (LUFactorization(), QRFactorization(), SVDFactorization())
        x = JLArray(zeros(Float64, 2))
        b = JLArray(copy(b_cpu))
        prob = NonlinearProblem(x, ka_noop)
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
        if m isa SVDFactorization
            @test fac_z.info == 0
            @test returncode(fac_z) === SUCCESS
            @test all(iszero, Array(fac_z.S))
            @test all(iszero, Array(fac_z.U))
            @test Array(fac_z.VT) == Matrix{Float64}(I, 2, 2)
            d_z = ldiv!!(similar(x), fac_z, b)
            @test all(iszero, Array(d_z))
        else
            @test fac_z.info > 0
            @test returncode(fac_z) === SINGULAR
            @test all(isfinite, Array(fac_z.factors))
        end
    end

    @test JACOBI_MAX_SWEEPS == 30
    @test jacobi_tolerance(16, Float64) == 4.0 * eps(Float64)
    c_rot, s_rot = jacobi_rotation(1.0, 1.0, 1.0)
    @test isapprox(c_rot, inv(sqrt(2.0)); atol = 4 * eps(Float64))
    @test isapprox(s_rot, inv(sqrt(2.0)); atol = 4 * eps(Float64))
    @test Set(jacobi_pair(4, 0, 0)) == Set((4, 1))
end

@testset "KA device LU, QR, and SVD agree with CPU solves on JLArray" begin
    @testset "n = $n" for n in (1, 16)
        for (T, TF) in PRECISION_PAIRS
            for M in (LUFactorization, QRFactorization, SVDFactorization)
                m = TF === Nothing ? M() : M(TF)
                A_cpu, b_cpu = ka_well_conditioned(T, n)
                κ = Float64(cond(big.(A_cpu), Inf))
                @test κ <= 10.0

                # CPU solve with the same (T, TF) pair
                x_cpu_init = zeros(T, n)
                prob_cpu = NonlinearProblem(x_cpu_init, ka_noop)
                fac_cpu = init(m, prob_cpu, x_cpu_init, nothing)
                copyto!(fac_cpu.factors, eltype(fac_cpu.factors).(A_cpu))
                fac_cpu = factorize!!(fac_cpu, m, fac_cpu.factors)
                @test fac_cpu.info == 0
                d_cpu = ldiv!!(similar(x_cpu_init), fac_cpu, copy(b_cpu))

                # Device solve on JLArray (in-place J === fac.factors)
                x_dev = JLArray(zeros(T, n))
                b_dev = JLArray(copy(b_cpu))
                prob_dev = NonlinearProblem(x_dev, ka_noop)
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

                if fac_dev isa DenseSVD
                    S_h = Array(fac_dev.S)
                    @test issorted(S_h; rev = true)
                    @test issorted(Array(fac_dev.rwork); rev = true)
                    if TF === Nothing
                        @test maximum(abs.(S_h .- fac_cpu.S)) <=
                              8 * n * eps_eff * fac_cpu.S[1]
                    end
                    bwd2 = norm(A_cpu * x_h - b_cpu) / (opnorm(A_cpu) * norm(x_h))
                    @test bwd2 <= 4 * n * eps_eff
                end

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
            for M in (LUFactorization, QRFactorization, SVDFactorization)
                for (sA, sb) in scales
                    n = 8
                    m = M(Float32)
                    A0, b0 = ka_well_conditioned(T, n)
                    A_cpu = T(sA) .* A0
                    b_cpu = T(sb) .* b0
                    κ = Float64(cond(big.(A0), Inf))

                    x_dev = JLArray(zeros(T, n))
                    b_dev = JLArray(copy(b_cpu))
                    prob_dev = NonlinearProblem(x_dev, ka_noop)
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
                    prob_f32 = NonlinearProblem(x_f32, ka_noop)
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
        prob = NonlinearProblem(x, ka_noop)

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

        fac_svd = init(SVDFactorization(), prob, x, nothing)
        @test length(fac_svd.work) == 1
        @test eltype(fac_svd.work) === Int32
        @test length(fac_svd.rwork) == 4
        @test eltype(fac_svd.rwork) === real(T)
        svd_sig = (typeof(fac_svd), SVDFactorization{Nothing}, JLMatrix{T})
        m_svd_fac = which(factorize!!, svd_sig)
        m_svd_bs = which(backsolve!, (typeof(fac_svd), JLVector{T}))
        @test endswith(replace(String(m_svd_fac.file), '\\' => '/'), "src/linear/ka/svd.jl")
        @test endswith(replace(String(m_svd_bs.file), '\\' => '/'), "src/linear/ka/svd.jl")
    end

    @testset "non-BLAS element type Float16 solves through src/linear/ka/" begin
        T = Float16
        n = 4
        A_cpu, b_cpu = ka_well_conditioned(T, n)
        κ = Float64(cond(big.(A_cpu), Inf))
        x_ref = big.(A_cpu) \ big.(b_cpu)
        for (M, file) in ((LUFactorization, "src/linear/ka/lu.jl"),
            (QRFactorization, "src/linear/ka/qr.jl"),
            (SVDFactorization, "src/linear/ka/svd.jl"))
            m = M()
            x = JLArray(zeros(T, n))
            prob = NonlinearProblem(x, ka_noop)
            fac = factorize!!(init(m, prob, x, nothing), m, JLArray(A_cpu))
            @test fac.info == 0
            m_bs = which(backsolve!, (typeof(fac), JLVector{T}))
            @test endswith(replace(String(m_bs.file), '\\' => '/'), file)
            d = ldiv!!(similar(x), fac, JLArray(b_cpu))
            @test d isa JLVector{T}
            rel = Float64(norm(big.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf))
            @test rel <= 8 * n * Float64(eps(T)) * κ
        end
    end

    @testset "no scalar indexing and type inference in init, factorize!!, ldiv!!" begin
        for (T, TF) in PRECISION_PAIRS
            for M in (LUFactorization, QRFactorization, SVDFactorization)
                m = TF === Nothing ? M() : M(TF)
                A_cpu, b_cpu = ka_well_conditioned(T, 6)
                x_dev = JLArray(zeros(T, 6))
                b_dev = JLArray(copy(b_cpu))
                prob_dev = NonlinearProblem(x_dev, ka_noop)
                fac = @inferred init(m, prob_dev, x_dev, nothing)
                if fac isa DenseSVD
                    F = effective_factor_eltype(T, TF)
                    @test length(fac.work) == 1
                    @test eltype(fac.work) === Int32
                    @test length(fac.rwork) == 6
                    @test eltype(fac.rwork) === real(F)
                end
                fac = @inferred factorize!!(fac, m, JLArray(A_cpu))
                @test fac.info == 0
                d = @inferred ldiv!!(similar(x_dev), fac, b_dev)
                @test all(isfinite, Array(d))
                if TF === Nothing
                    e_sc = fac.scale_exponent + 3
                    fac_sc = fac isa DenseLU ?
                             DenseLU(
                        fac.factors, fac.ipiv, fac.rhs, fac.info, e_sc, fac.status) :
                             fac isa DenseQR ?
                             DenseQR(
                        fac.factors, fac.τ, fac.work, fac.rhs, fac.info, e_sc) :
                             DenseSVD(
                        fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork, fac.tmp,
                        fac.rhs, fac.info, e_sc)
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
            prob = NonlinearProblem(x_dev, ka_noop)
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
                    prob = NonlinearProblem(x_dev, ka_noop)
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
            prob = NonlinearProblem(x_dev, ka_noop)
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
            prob = NonlinearProblem(x_dev, ka_noop)
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
                prob2 = NonlinearProblem(x2, ka_noop)
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
            prob = NonlinearProblem(x_dev, ka_noop)
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
        prob = NonlinearProblem(x_dev, ka_noop)
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
        prob2 = NonlinearProblem(x2, ka_noop)
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
        prob0 = NonlinearProblem(x0, ka_noop)
        fac0 = init(m, prob0, x0, nothing)
        fac0 = factorize!!(fac0, m, fac0.factors)
        @test fac0.info == 0
        d0 = ldiv!!(similar(x0), fac0, JLArray(zeros(T, 0)))
        @test length(d0) == 0
    end
end

@testset "Check 1: Jacobi round-robin schedule coverage and disjointness" begin
    for n in (1, 2, 3, 4, 5, 6, 15, 16)
        m = iseven(n) ? n : n + 1
        pair_counts = zeros(Int, n, n)
        for r in 0:(m - 2)
            round_counts = zeros(Int, n)
            for k in 0:(m ÷ 2 - 1)
                p0, q0 = jacobi_pair(n, r, k)
                p, q = ifelse(p0 < q0, (p0, q0), (q0, p0))
                if q <= n
                    @test 1 <= p < q <= n
                    round_counts[p] += 1
                    round_counts[q] += 1
                    pair_counts[p, q] += 1
                end
            end
            @test maximum(round_counts; init = 0) <= 1
        end
        @test sum(pair_counts) == n * (n - 1) ÷ 2
        for p in 1:n, q in (p + 1):n

            @test pair_counts[p, q] == 1
        end
    end
end

@testset "Check 2: no race across columns and bit-identical repeated runs on JLArray" begin
    for n in (15, 16), T in ELTYPES

        RT = real(T)
        A_cpu, b_cpu = ka_well_conditioned(T, n, 0x1234 + n)
        m = SVDFactorization()

        x1 = JLArray(zeros(T, n))
        prob1 = NonlinearProblem(x1, ka_noop)
        fac1 = init(m, prob1, x1, nothing)
        copyto!(fac1.factors, JLArray(A_cpu))
        fac1 = factorize!!(fac1, m, fac1.factors)
        d1 = ldiv!!(similar(x1), fac1, JLArray(b_cpu))

        x2 = JLArray(zeros(T, n))
        prob2 = NonlinearProblem(x2, ka_noop)
        fac2 = init(m, prob2, x2, nothing)
        copyto!(fac2.factors, JLArray(A_cpu))
        fac2 = factorize!!(fac2, m, fac2.factors)
        d2 = ldiv!!(similar(x2), fac2, JLArray(b_cpu))

        @test fac1.info == 0
        @test fac2.info == 0
        @test Array(fac1.U) == Array(fac2.U)
        @test Array(fac1.S) == Array(fac2.S)
        @test Array(fac1.VT) == Array(fac2.VT)
        @test Array(fac1.rwork) == Array(fac2.rwork)
        @test Array(d1) == Array(d2)

        U_h = Array(fac1.U)
        S_h = Array(fac1.S)
        VT_h = Array(fac1.VT)
        @test opnorm(U_h' * U_h - I, Inf) <= 8 * n * eps(RT)
        @test opnorm(VT_h * VT_h' - I, Inf) <= 8 * n * eps(RT)
        @test opnorm(U_h * Diagonal(S_h) * VT_h - A_cpu, Inf) <=
              8 * n * eps(RT) * opnorm(A_cpu, Inf)
    end

    @testset "equal-column-norm rotation (alpha == beta) on JLArray, T = $T" for T in ELTYPES
        RT = real(T)
        m = SVDFactorization()
        for A_eqnorm in (
            T[2 1; 1 2],
            T[2 -1; -1 2],
            T[2 1 0 0; 1 2 0 0; 0 0 3 1; 0 0 1 3]
        )
            n = size(A_eqnorm, 1)
            b_eq = T[T(i) for i in 1:n]
            x_ref = big.(A_eqnorm) \ big.(b_eq)
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = init(m, prob, x_dev, nothing)
            fac = factorize!!(fac, m, JLArray(A_eqnorm))
            @test fac.info == 0
            U_h = Array(fac.U)
            @test opnorm(U_h' * U_h - I, Inf) <= 8 * n * eps(RT)
            d = ldiv!!(similar(x_dev), fac, JLArray(b_eq))
            rel = Float64(norm(big.(Array(d)) - x_ref, Inf) / norm(x_ref, Inf))
            @test rel <= 8 * n * Float64(eps(RT))
        end
    end
end

@testset "Check 4: SVD subnormal prescaling and rank-deficient termination traps on JLArray" begin
    @testset "4(a) subnormal ldexp prescaling matches integer matrix bit for bit" begin
        for (T, TF) in PRECISION_PAIRS, n in (4, 16)

            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            E = exponent(floatmin(R)) - 8
            B_cpu = ka_integers(T, n)
            A_cpu = ldexp(one(R), E) .* B_cpu
            @test all(x -> abs(F(x)) < floatmin(R), A_cpu)

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_cpu))
            fac_ref = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(B_cpu))

            @test fac.info == 0
            @test all(isfinite, Array(fac.U))
            @test all(isfinite, Array(fac.VT))
            @test all(isfinite, Array(fac.S))
            @test Array(fac.U) == Array(fac_ref.U)
            @test Array(fac.VT) == Array(fac_ref.VT)
            @test Array(fac.S) == ldexp.(Array(fac_ref.S), E)
        end
    end

    @testset "4(b) singular value above floatmax(R) leaves solve finite via rwork" begin
        for (T, TF) in PRECISION_PAIRS
            n = 4
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            s_half = R(0.5) * floatmax(R)
            s_eighth = R(0.125) * floatmax(R)
            s_quarter = R(0.25) * floatmax(R)
            A_cpu = fill(T(s_half), n, n) + Matrix{T}(Diagonal(fill(T(s_eighth), n)))
            b_cpu = fill(T(s_quarter), n)
            x_ref = big.(A_cpu) \ big.(b_cpu)
            κ = Float64(cond(big.(A_cpu), Inf))

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_cpu))
            @test fac.info == 0
            @test returncode(fac) === SUCCESS
            @test !isfinite(Array(fac.S)[1])
            @test all(isfinite, Array(fac.rwork))

            d = ldiv!!(similar(x_dev), fac, JLArray(b_cpu))
            x_h = Array(d)
            @test all(isfinite, x_h)
            rel = Float64(norm(big.(x_h) - x_ref, Inf) / norm(x_ref, Inf))
            @test rel <= 8 * n * Float64(eps(R)) * κ
        end
    end

    @testset "4(c) rank-deficient matrices terminate with exact rank and minimum-norm step" begin
        for (T, TF) in PRECISION_PAIRS, r in (5, 1, 0)

            n = 16
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            tol = jacobi_tolerance(n, R)
            A_cpu = Matrix(rank_deficient(100r + n, T, n, r))

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_cpu))

            @test fac.info == 0
            @test returncode(fac) === SUCCESS
            S_h = Array(fac.S)
            @test issorted(S_h; rev = true)
            num_rank = S_h[1] == zero(R) ? 0 : count(>(tol * S_h[1]), S_h)
            @test num_rank == r
            @test all(s -> s <= tol * S_h[1], S_h[(r + 1):n])
            if r >= 1
                @test S_h[r] > tol * S_h[1]
            end

            U_h = Array(fac.U)
            @test all(isfinite, U_h)
            @test all(iszero, U_h[:, (r + 1):n])

            x0_cpu = randn(Xoshiro(400r + n), T, n)
            b_cpu = A_cpu * x0_cpu
            d = ldiv!!(similar(x_dev), fac, JLArray(b_cpu))
            x_h = Array(d)
            @test all(isfinite, x_h)
            if r == 0
                @test all(iszero, x_h)
                b_arb = randn(Xoshiro(999 + n), T, n)
                d_arb = ldiv!!(similar(x_dev), fac, JLArray(b_arb))
                @test all(iszero, Array(d_arb))
            else
                @test norm(A_cpu * x_h - b_cpu, Inf) <=
                      4 * n * eps(R) * opnorm(A_cpu, Inf) * norm(x_h, Inf)
                @test norm(A_cpu * x_h - b_cpu) <=
                      4 * n * eps(R) * opnorm(A_cpu) * norm(x_h)
                @test norm(x_h) <= (1 + 4 * n * eps(R)) * norm(x0_cpu)
            end
        end

        # Sub-cutoff positive singular value rwork[2] = tol / 4 > 0 is truncated by tol * rwork[1]
        for (T, TF) in PRECISION_PAIRS
            n = 4
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            tol = jacobi_tolerance(n, R)
            A_sub = zeros(T, n, n)
            A_sub[1, 1] = one(T)
            A_sub[2, 2] = T(tol / R(4))
            b_sub = ones(T, n)
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_sub))
            @test fac.info == 0
            @test Array(fac.rwork)[2] > zero(R)
            @test Array(fac.rwork)[2] <= tol * Array(fac.rwork)[1]
            @test all(iszero, Array(fac.U)[:, 2:n])
            d = ldiv!!(similar(x_dev), fac, JLArray(b_sub))
            d_h = Array(d)
            @test isapprox(d_h[1], one(T); atol = 4 * eps(R))
            @test all(iszero, d_h[2:n])

            # Direct backsolve! check when U[:, 2] is non-zero so ka_svd_ut_mul_scale_kernel!'s own
            # tol * rwork[1] guard is exercised independently of U column zeroing
            copyto!(fac.U, Matrix{F}(I, n, n))
            copyto!(fac.VT, Matrix{F}(I, n, n))
            d_bs = JLArray(ones(F, n))
            backsolve!(fac, d_bs)
            d_bs_h = Array(d_bs)
            @test isapprox(d_bs_h[1], one(F); atol = 4 * eps(R))
            @test all(iszero, d_bs_h[2:n])
        end
    end

    @testset "4(d) negligible column of norm 0.5 * eps(R) is skipped and zeroed in U" begin
        for (T, TF) in PRECISION_PAIRS
            n = 4
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            ε = eps(R) / R(2)
            A_cpu = Matrix{T}(I, n, n)
            A_cpu[n, n] = zero(T)
            A_cpu[1, n] = T(ε)

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_cpu))
            @test fac.info == 0
            @test Array(fac.S)[n] == ε
            @test all(iszero, Array(fac.U)[:, n])
        end
    end

    @testset "4(e) non-finite inputs and 0 x 0 system on JLArray" begin
        for (T, TF) in PRECISION_PAIRS
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            RT = real(T)
            for n in (1, 4)
                x_dev = JLArray(zeros(T, n))
                prob = NonlinearProblem(x_dev, ka_noop)

                A_nan = fill(T(RT(NaN)), n, n)
                fac_nan = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_nan))
                if n >= 2
                    @test fac_nan.info == 1
                    @test returncode(fac_nan) === SINGULAR
                end
                d_nan = ldiv!!(similar(x_dev), fac_nan, JLArray(ones(T, n)))
                @test d_nan isa JLVector{T}

                A_inf = Matrix{T}(I, n, n)
                A_inf[1, 1] = T(RT(Inf))
                fac_inf = factorize!!(init(m, prob, x_dev, nothing), m, JLArray(A_inf))
                d_inf = ldiv!!(similar(x_dev), fac_inf, JLArray(ones(T, n)))
                @test d_inf isa JLVector{T}
            end

            x0 = JLArray(zeros(T, 0))
            prob0 = NonlinearProblem(x0, ka_noop)
            fac0 = factorize!!(init(m, prob0, x0, nothing), m, JLArray(zeros(T, 0, 0)))
            @test fac0.info == 0
            d0 = ldiv!!(similar(x0), fac0, JLArray(zeros(T, 0)))
            @test length(d0) == 0
        end
    end
end
