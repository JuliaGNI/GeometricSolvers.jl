using CUDA
using GPUArraysCore: allowscalar
using LinearAlgebra: BLAS, BlasFloat, Diagonal, I, LU, cond, lu!, norm, opnorm, qr, qr!,
                     svd!
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: LUFactorization, Newton, NonlinearProblem, QRFactorization,
                        SVDFactorization
using GeometricSolvers: SINGULAR, SUCCESS, backsolve!, dense_factor, factor_eltype,
                        factorize!!, init, jacobi_tolerance, ka_factorize_lu!,
                        ka_factorize_qr!, ka_factorize_svd!, ldiv!!, returncode, solve,
                        solve!

include("../../helpers/own_kernel.jl")

@testset "GeometricSolversCUDAExt extension loading, dispatch, and CUDA census alignment" begin
    ext_cuda = Base.get_extension(GeometricSolvers, :GeometricSolversCUDAExt)
    @test ext_cuda !== nothing

    for Mem in (CUDA.DeviceMemory, CUDA.UnifiedMemory)
        VecI32 = CuArray{Int32, 1, Mem}
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            Mat = CuArray{T, 2, Mem}
            Vec = CuArray{T, 1, Mem}
            VecR = CuArray{real(T), 1, Mem}

            DevLU = DeviceLU{Mat, VecI32, Nothing, VecI32}
            @test which(factorize!!, (DevLU, LUFactorization{Nothing}, Mat)).module ===
                  ext_cuda
            @test which(backsolve!, (DevLU, Vec)).module === ext_cuda

            DevQR = DeviceQR{Mat, Vec, Nothing, VecI32}
            @test which(factorize!!, (DevQR, QRFactorization{Nothing}, Mat)).module ===
                  ext_cuda
            @test which(backsolve!, (DevQR, Vec)).module === ext_cuda

            DevSVD = DeviceSVD{Mat, VecR, VecI32, Vec, Nothing}
            @test which(factorize!!, (DevSVD, SVDFactorization{Nothing}, Mat)).module ===
                  ext_cuda
            @test which(backsolve!, (DevSVD, Vec)).module === ext_cuda
        end

        # Mixed-precision T = Float64, TF = Float32 reaches cuSOLVER Float32 methods
        MatF32 = CuArray{Float32, 2, Mem}
        VecF32 = CuArray{Float32, 1, Mem}
        MatF64 = CuArray{Float64, 2, Mem}
        DevLU_Mix = DeviceLU{MatF32, VecI32, VecF32, VecI32}
        DevQR_Mix = DeviceQR{MatF32, VecF32, VecF32, VecI32}
        DevSVD_Mix = DeviceSVD{MatF32, VecF32, VecI32, VecF32, VecF32}
        @test which(factorize!!, (DevLU_Mix, LUFactorization{Float32}, MatF64)).module ===
              ext_cuda
        @test which(backsolve!, (DevLU_Mix, VecF32)).module === ext_cuda
        @test which(factorize!!, (DevQR_Mix, QRFactorization{Float32}, MatF64)).module ===
              ext_cuda
        @test which(backsolve!, (DevQR_Mix, VecF32)).module === ext_cuda
        @test which(factorize!!, (DevSVD_Mix, SVDFactorization{Float32}, MatF64)).module ===
              ext_cuda
        @test which(backsolve!, (DevSVD_Mix, VecF32)).module === ext_cuda

        # Float16 remains on the KernelAbstractions device kernels
        MatF16 = CuArray{Float16, 2, Mem}
        VecF16 = CuArray{Float16, 1, Mem}
        DevLU_F16 = DeviceLU{MatF16, VecI32, Nothing, VecI32}
        DevQR_F16 = DeviceQR{MatF16, VecF16, Nothing, VecI32}
        DevSVD_F16 = DeviceSVD{MatF16, VecF16, VecI32, VecF16, Nothing}
        @test which(factorize!!, (DevLU_F16, LUFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevLU_F16, VecF16)).module === GeometricSolvers
        @test which(factorize!!, (DevQR_F16, QRFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevQR_F16, VecF16)).module === GeometricSolvers
        @test which(factorize!!, (DevSVD_F16, SVDFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevSVD_F16, VecF16)).module === GeometricSolvers
    end

    # CUDA capability census alignment: every pass factorisation cell for (F32, F64, CF32, CF64)
    # has an extension method in GeometricSolversCUDAExt, and F16 has none.
    VecI32_D = CuArray{Int32, 1, CUDA.DeviceMemory}
    for T in (Float16, Float32, Float64, ComplexF32, ComplexF64)
        Mat = CuArray{T, 2, CUDA.DeviceMemory}
        Vec = CuArray{T, 1, CUDA.DeviceMemory}
        VecR = CuArray{real(T), 1, CUDA.DeviceMemory}
        expected_ext = T in (Float32, Float64, ComplexF32, ComplexF64)
        for (Fac, M) in (
            (DeviceLU{Mat, VecI32_D, Nothing, VecI32_D}, LUFactorization{Nothing}),
            (DeviceQR{Mat, Vec, Nothing, VecI32_D}, QRFactorization{Nothing}),
            (DeviceSVD{Mat, VecR, VecI32_D, Vec, Nothing}, SVDFactorization{Nothing})
        )
            @test (which(factorize!!, (Fac, M, Mat)).module === ext_cuda) == expected_ext
        end
    end
end

if CUDA.functional()
    allowscalar(false)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    const CUDA_BLAS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

    function cuda_rank_deficient(seed::Integer, ::Type{T}, n::Int, r::Int) where {T}
        rng = Xoshiro(seed)
        U = Matrix{T}(qr(randn(rng, T, n, n)).Q)
        V = Matrix{T}(qr(randn(rng, T, n, n)).Q)
        Σ = real(T)[i <= r ? real(T)(0.5)^(i - 1) : zero(real(T)) for i in 1:n]
        return U * Diagonal(Σ) * V'
    end

    cuda_noop(r, y, p) = nothing

    @testset "CUDA cuSOLVER agreement with own KA kernels at n = $n" for n in (1, 16, 256)
        @testset "T = $T" for T in CUDA_BLAS_ELTYPES
            A_cpu, b_cpu = cuda_well_conditioned(T, n)
            κ = Float64(cond(A_cpu, Inf))
            @test κ <= 10.0
            epsT = Float64(eps(real(T)))
            bound = 8 * n * epsT * κ

            dA = CuArray(A_cpu)
            db = CuArray(b_cpu)
            prob_dev = NonlinearProblem(CuArray(zeros(T, n)), cuda_noop)

            for M in (LUFactorization, QRFactorization, SVDFactorization)
                m = M()
                dx_ext = CuArray(zeros(T, n))
                dx_ka = CuArray(zeros(T, n))
                fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
                fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)

                fac_ext = factorize!!(fac_ext, m, dA)
                @test fac_ext.info == 0
                @test returncode(fac_ext) === SUCCESS
                ldiv!!(dx_ext, fac_ext, db)

                if M === LUFactorization
                    fac_ka = ka_factorize_lu!(fac_ka, dA)
                elseif M === QRFactorization
                    fac_ka = ka_factorize_qr!(fac_ka, dA)
                else
                    fac_ka = ka_factorize_svd!(fac_ka, dA)
                    S_ext = Array(fac_ext.S)
                    S_ka = Array(fac_ka.S)
                    @test issorted(S_ext; rev = true)
                    @test maximum(abs.(S_ext .- S_ka)) <= 8 * n * epsT * S_ka[1]
                end
                @test fac_ka.info == 0
                # the own KA kernels, the reference of `test/devices/own_kernel.jl`
                own_kernel_ldiv!(dx_ka, fac_ka, db)

                x_ext = Array(dx_ext)
                x_ka = Array(dx_ka)
                rel_diff = norm(x_ext - x_ka, Inf) / max(norm(x_ka, Inf), epsT)
                @test rel_diff <= bound
            end
        end
    end

    @testset "CUDA cuSOLVER large-n LU, QR, SVD, and mixed TF = Float32" begin
        # LU and QR run at n = 10^4 and the SVD rows at n = 10^3; the comment above the SVD
        # rows gives the reason.
        for T in CUDA_BLAS_ELTYPES
            let n = 10^4
                A_cpu, b_cpu = cuda_well_conditioned(T, n)
                κ = Float64(cond(A_cpu, Inf))
                @test κ <= 10.0
                epsT = Float64(eps(real(T)))
                dA = CuArray(A_cpu)
                db = CuArray(b_cpu)
                dx_ext = CuArray(zeros(T, n))
                dx_ka = CuArray(zeros(T, n))
                prob_dev = NonlinearProblem(dx_ext, cuda_noop)
                for M in (LUFactorization, QRFactorization)
                    m = M()
                    fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
                    fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)
                    fac_ext = factorize!!(fac_ext, m, dA)
                    @test fac_ext.info == 0
                    @test returncode(fac_ext) === SUCCESS
                    ldiv!!(dx_ext, fac_ext, db)
                    x_ext = Array(dx_ext)
                    bwd = norm(A_cpu * x_ext - b_cpu, Inf) /
                          (opnorm(A_cpu, Inf) * norm(x_ext, Inf))
                    @test bwd <= 4 * n * epsT

                    # the own KA kernel; the extension's `backsolve!` calls cuSOLVER
                    if M === LUFactorization
                        fac_ka = ka_factorize_lu!(fac_ka, dA)
                    else
                        fac_ka = ka_factorize_qr!(fac_ka, dA)
                    end
                    @test fac_ka.info == 0
                    own_kernel_ldiv!(dx_ka, fac_ka, db)
                    x_ka = Array(dx_ka)
                    @test norm(x_ext - x_ka, Inf) / max(norm(x_ka, Inf), epsT) <=
                          8 * n * epsT * κ
                end
            end
        end

        # The own Jacobi SVD costs 6.5 s at n = 1024 on Metal and about ten times that per
        # doubling, so a row at n = 10^4 takes over an hour; n = 10^3 is the largest size at
        # which the own SVD is verified on a device.
        for T in CUDA_BLAS_ELTYPES
            let n = 10^3
                A_cpu, b_cpu = cuda_well_conditioned(T, n)
                κ = Float64(cond(A_cpu, Inf))
                @test κ <= 10.0
                epsT = Float64(eps(real(T)))
                dA = CuArray(A_cpu)
                db = CuArray(b_cpu)
                dx_ext = CuArray(zeros(T, n))
                dx_ka = CuArray(zeros(T, n))
                prob_dev = NonlinearProblem(dx_ext, cuda_noop)
                m = SVDFactorization()
                fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
                fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)
                fac_ext = factorize!!(fac_ext, m, dA)
                @test fac_ext.info == 0
                @test returncode(fac_ext) === SUCCESS
                ldiv!!(dx_ext, fac_ext, db)

                fac_ka = ka_factorize_svd!(fac_ka, dA)
                S_ext = Array(fac_ext.S)
                S_ka = Array(fac_ka.S)
                @test issorted(S_ext; rev = true)
                @test maximum(abs.(S_ext .- S_ka)) <= 8 * n * epsT * S_ka[1]
                @test fac_ka.info == 0

                # the own KA kernel; the extension's SVD `backsolve!` delegates to it
                own_kernel_ldiv!(dx_ka, fac_ka, db)

                x_ext = Array(dx_ext)
                x_ka = Array(dx_ka)
                @test norm(x_ext - x_ka, Inf) / max(norm(x_ka, Inf), epsT) <=
                      8 * n * epsT * κ
            end
        end

        # The mixed rows: a Float64 or ComplexF64 problem factorised in Float32
        for T in (Float64, ComplexF64), n in (1, 16, 10^4)

            A_cpu, b_cpu = cuda_well_conditioned(T, n)
            κ = Float64(cond(A_cpu, Inf))
            bound = 8 * n * Float64(eps(Float32)) * κ
            dA = CuArray(A_cpu)
            db = CuArray(b_cpu)
            dx_ext = CuArray(zeros(T, n))
            dx_ka = CuArray(zeros(T, n))
            prob_dev = NonlinearProblem(dx_ext, cuda_noop)
            for M in (LUFactorization, QRFactorization)
                m = M(Float32)
                fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
                fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)
                fac_ext = factorize!!(fac_ext, m, dA)
                @test fac_ext.info == 0
                @test returncode(fac_ext) === SUCCESS
                ldiv!!(dx_ext, fac_ext, db)

                # the own KA kernel; the extension's LU and QR `backsolve!` calls cuSOLVER
                if M === LUFactorization
                    fac_ka = ka_factorize_lu!(fac_ka, dA)
                else
                    fac_ka = ka_factorize_qr!(fac_ka, dA)
                end
                # the own KA kernels in the factor element type `Float32`
                own_kernel_ldiv!(dx_ka, fac_ka, db)

                x_ext = Array(dx_ext)
                x_ka = Array(dx_ka)
                @test norm(x_ext - x_ka, Inf) /
                      max(norm(x_ka, Inf), Float64(eps(Float32))) <= bound

                # Zero right-hand side with TF != T returns zero step
                dz = CuArray(ones(T, n))
                ldiv!!(dz, fac_ext, CuArray(zeros(T, n)))
                @test all(iszero, Array(dz))
            end
        end

        # The mixed SVD rows run at n = 10^3 too, for the reason given above.
        for T in (Float64, ComplexF64), n in (1, 16, 10^3)

            A_cpu, b_cpu = cuda_well_conditioned(T, n)
            κ = Float64(cond(A_cpu, Inf))
            bound = 8 * n * Float64(eps(Float32)) * κ
            dA = CuArray(A_cpu)
            db = CuArray(b_cpu)
            dx_ext = CuArray(zeros(T, n))
            dx_ka = CuArray(zeros(T, n))
            prob_dev = NonlinearProblem(dx_ext, cuda_noop)
            m = SVDFactorization(Float32)
            fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
            fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)
            fac_ext = factorize!!(fac_ext, m, dA)
            @test fac_ext.info == 0
            @test returncode(fac_ext) === SUCCESS
            ldiv!!(dx_ext, fac_ext, db)

            fac_ka = ka_factorize_svd!(fac_ka, dA)
            S_ext = Array(fac_ext.S)
            S_ka = Array(fac_ka.S)
            @test maximum(abs.(S_ext .- S_ka)) <= 8 * n * Float64(eps(Float32)) * S_ka[1]

            # the own KA kernel; the extension's SVD `backsolve!` delegates to it
            own_kernel_ldiv!(dx_ka, fac_ka, db)

            x_ext = Array(dx_ext)
            x_ka = Array(dx_ka)
            @test norm(x_ext - x_ka, Inf) /
                  max(norm(x_ka, Inf), Float64(eps(Float32))) <= bound
        end
    end

    @testset "CUDA cuSOLVER 0x0, singular, prescaling, rank-deficient, non-finite, and KA fallbacks: $T" for T in CUDA_BLAS_ELTYPES
        ext_cuda = Base.get_extension(GeometricSolvers, :GeometricSolversCUDAExt)
        RT = real(T)
        for M in (LUFactorization, QRFactorization, SVDFactorization)
            m = M()
            x0 = CuArray(zeros(T, 0))
            prob0 = NonlinearProblem(x0, cuda_noop)
            fac0 = factorize!!(
                dense_factor(m, prob0, x0, nothing), m, CuArray(zeros(T, 0, 0)))
            @test fac0.info == 0
            @test returncode(fac0) === SUCCESS
            d0 = ldiv!!(similar(x0), fac0, CuArray(zeros(T, 0)))
            @test length(d0) == 0
        end

        x4 = CuArray(zeros(T, 4))
        prob4 = NonlinearProblem(x4, cuda_noop)
        for M in (LUFactorization, QRFactorization)
            m = M()
            # Row of zeros / all-zero matrix
            fac_z = factorize!!(
                dense_factor(m, prob4, x4, nothing), m, CuArray(zeros(T, 4, 4)))
            @test fac_z.info > 0
            @test returncode(fac_z) === SINGULAR

            # Zero column 2 -> exact zero on diagonal at index 2
            A_c2 = Matrix{T}(I, 4, 4)
            A_c2[:, 2] .= zero(T)
            fac_c2 = factorize!!(dense_factor(m, prob4, x4, nothing), m, CuArray(A_c2))
            @test fac_c2.info == 2
            @test returncode(fac_c2) === SINGULAR

            # Two equal rows (for LU)
            if M === LUFactorization
                A_eq = T[1 2 3 4; 1 2 3 4; 0 1 1 1; 0 0 1 2]
                fac_eq = factorize!!(
                    dense_factor(m, prob4, x4, nothing), m, CuArray(A_eq))
                @test fac_eq.info > 0
                @test returncode(fac_eq) === SINGULAR
            end
        end

        # Direct check on _cusolver_qr_info for a non-finite R diagonal; the zero diagonal is
        # checked through factorize!! above
        A_qr_nf = Matrix{T}(I, 4, 4)
        A_qr_nf[3, 3] = T(RT(NaN))
        @test ext_cuda._cusolver_qr_info(CuArray(A_qr_nf)) == 3

        # Fallback to ka_factorize_lu! when getrf! info_c < 0 and ka_factorize_svd! when gesvdj! info_c > 0
        A_ok = T[4 1; 1 3]
        dA_ok = CuArray(A_ok)
        dx_ok = CuArray(zeros(T, 2))
        prob2 = NonlinearProblem(dx_ok, cuda_noop)

        fac_lu_fb = dense_factor(LUFactorization(), prob2, dx_ok, nothing)
        fac_lu_fb = ext_cuda._cusolver_lu_finish!(fac_lu_fb, dA_ok, -1, 0)
        @test fac_lu_fb.info == 0
        @test returncode(fac_lu_fb) === SUCCESS
        ldiv!!(dx_ok, fac_lu_fb, CuArray(T[5, 4]))
        @test isapprox(Array(dx_ok), T[1, 1]; rtol = 16 * eps(RT))

        fac_svd_fb = dense_factor(SVDFactorization(), prob2, dx_ok, nothing)
        fill!(fac_svd_fb.U, zero(T))
        fill!(fac_svd_fb.σ̂, zero(RT))
        fill!(fac_svd_fb.V, zero(T))
        fac_svd_fb = ext_cuda._cusolver_svd_finish!(
            fac_svd_fb, dA_ok, fac_svd_fb.U, fac_svd_fb.σ̂, fac_svd_fb.V, 1, 0)
        @test fac_svd_fb.info == 0
        @test fac_svd_fb.rank == 2
        @test returncode(fac_svd_fb) === SUCCESS
        ldiv!!(dx_ok, fac_svd_fb, CuArray(T[5, 4]))
        @test isapprox(Array(dx_ok), T[1, 1]; rtol = 16 * eps(RT))

        # SVD prescaling on a subnormal input
        m_svd = SVDFactorization()
        E_sub = exponent(floatmin(RT)) - 8
        B_cpu = T[i + j for i in 1:4, j in 1:4]
        A_sub = ldexp(one(RT), E_sub) .* B_cpu
        fac_sub = factorize!!(dense_factor(m_svd, prob4, x4, nothing), m_svd, CuArray(A_sub))
        fac_ref = factorize!!(dense_factor(m_svd, prob4, x4, nothing), m_svd, CuArray(B_cpu))
        @test fac_sub.info == 0
        @test isapprox(Array(fac_sub.S), ldexp.(Array(fac_ref.S), E_sub); rtol = 8 *
                                                                                 eps(RT))

        # SVD rank-deficient matrices at n = 16 for r in (5, 1, 0)
        n16 = 16
        x16 = CuArray(zeros(T, n16))
        prob16 = NonlinearProblem(x16, cuda_noop)
        for r in (5, 1, 0)
            A_rk = cuda_rank_deficient(100r + n16, T, n16, r)
            fac_rk = factorize!!(
                dense_factor(m_svd, prob16, x16, nothing), m_svd, CuArray(A_rk))
            @test fac_rk.info == 0
            @test returncode(fac_rk) === SUCCESS
            @test fac_rk.rank == r
            S_rk = Array(fac_rk.S)
            x0_rk = randn(Xoshiro(400r + n16), T, n16)
            b_rk = A_rk * x0_rk
            d_rk = ldiv!!(similar(x16), fac_rk, CuArray(b_rk))
            x_rk = Array(d_rk)
            @test all(isfinite, x_rk)
            if r == 0
                @test all(iszero, x_rk)
            else
                @test norm(A_rk * x_rk - b_rk) <= 4 * n16 * eps(RT) * S_rk[1] * norm(x_rk)
                @test norm(x_rk) <= (1 + 4 * n16 * eps(RT)) * norm(x0_rk)
            end
        end

        # SVD non-finite input sets flag[1] = 1, info = 1, SINGULAR without throwing
        A_nan = Matrix{T}(I, 4, 4)
        A_nan[2, 2] = T(RT(NaN))
        fac_nan = factorize!!(
            dense_factor(m_svd, prob4, x4, nothing), m_svd, CuArray(A_nan))
        @test fac_nan.info == 1
        @test Array(fac_nan.flag)[1] == Int32(1)
        @test returncode(fac_nan) === SINGULAR

        # `gesvdj!` factorises `J / 2^e` in `fac.U`, and `fac.factors` keeps `J`: both where the
        # caller passes the factor buffer back as `J` and where it passes another matrix.
        A_keep = T[4 1; 1 3]
        dx_keep = CuArray(zeros(T, 2))
        fac_keep = dense_factor(
            m_svd, NonlinearProblem(dx_keep, cuda_noop), dx_keep, nothing)
        copyto!(fac_keep.factors, CuArray(A_keep))
        fac_keep = factorize!!(fac_keep, m_svd, fac_keep.factors)
        @test fac_keep.info == 0
        @test Array(fac_keep.factors) == A_keep
        A_keep2 = T[3 1; 2 5]
        fac_keep = factorize!!(fac_keep, m_svd, CuArray(A_keep2))
        @test fac_keep.info == 0
        @test Array(fac_keep.factors) == A_keep2
    end

    @testset "CUDA capability census probe" begin
        for T in CUDA_BLAS_ELTYPES
            rng = Xoshiro(0x5eed)
            N = 8
            A0 = randn(rng, N, N) + N * I
            b0 = randn(rng, N)
            if T <: Complex
                R = real(T)
                A = R.(A0) .+ im .* R.(reverse(A0; dims = 1))
                b = R.(b0) .+ im .* R.(reverse(b0))
            else
                A = T.(A0)
                b = T.(b0)
            end
            Ad = CuArray(A)
            bd = CuArray(b)
            Aref = ComplexF64.(A)
            tol = 100 * Float64(eps(real(T)))
            relerr(C, Cref) = norm(ComplexF64.(C) .- ComplexF64.(Cref)) /
                              max(norm(ComplexF64.(Cref)), eps())

            # lu!
            F_lu = lu!(copy(Ad))
            Fh = LU(Array(F_lu.factors), Array(F_lu.ipiv), F_lu.info)
            @test relerr(ComplexF64.(Fh.L) * ComplexF64.(Fh.U), Aref[Fh.p, :]) <= tol

            # qr!
            F_qr = qr!(copy(Ad))
            Q = Array(Matrix(F_qr.Q))
            R_mat = Array(Matrix(F_qr.R))
            @test relerr(ComplexF64.(Q) * ComplexF64.(R_mat), Aref) <= tol

            # svd!
            F_svd = svd!(copy(Ad))
            @test relerr(
                ComplexF64.(Array(F_svd.U)) * Diagonal(ComplexF64.(Array(F_svd.S))) *
                ComplexF64.(Array(F_svd.Vt)), Aref) <= tol

            # A \ b
            xd = Ad \ bd
            @test relerr(Aref * ComplexF64.(Array(xd)), ComplexF64.(b)) <= tol
        end
    end

    BLAS.set_num_threads(old_blas_threads)
else
    @test_skip CUDA.functional()
end
