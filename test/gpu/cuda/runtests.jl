# The tests of `GeometricSolversCUDAExt` and `GeometricSolversCUDSSExt`, run by hand on a machine
# with a CUDA device: `julia --project=test/gpu/cuda test/gpu/cuda/runtests.jl`. The extension
# loading, dispatch and JET rows run without a device; every row that factorises runs where
# `CUDA.functional()`.

using CUDA
using CUDSS: CUDSS, CudssSolver
using CUDSS.cuSPARSE: CuSparseMatrixCSR
using GPUArraysCore: allowscalar
using JET: JET
using LinearAlgebra: BLAS, Diagonal, I, LU, cond, lu!, mul!, norm, opnorm, qr, qr!, svd!
using Random: Xoshiro
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse
using Test

using GeometricSolvers
using GeometricSolvers: DeviceCholesky, DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: CholeskyFactorization, LUFactorization, NonlinearProblem,
                        QRFactorization, SVDFactorization,
                        UMFPACKFactorization
using GeometricSolvers: SINGULAR, SUCCESS, backsolve!, dense_factor, factorize!!, init,
                        init_linear, ka_cholesky_backsolve!, ka_factorize_cholesky!,
                        ka_factorize_lu!, ka_factorize_qr!, ka_factorize_svd!,
                        ldexp2, ldiv!!, returncode, solve!

include("../../helpers/own_kernel.jl")
include("../../helpers/sparse_problems.jl")

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

        # Cholesky: cuSOLVER `potrf`/`potrs` for the BLAS types, the mixed pair included, and
        # the own kernel for `Float16`
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            Mat = CuArray{T, 2, Mem}
            Vec = CuArray{T, 1, Mem}
            DevChol = DeviceCholesky{Mat, Nothing, VecI32}
            @test which(factorize!!, (DevChol, CholeskyFactorization{Nothing}, Mat)).module ===
                  ext_cuda
            @test which(backsolve!, (DevChol, Vec)).module === ext_cuda
        end
        DevChol_Mix = DeviceCholesky{MatF32, VecF32, VecI32}
        @test which(factorize!!, (DevChol_Mix, CholeskyFactorization{Float32}, MatF64)).module ===
              ext_cuda
        @test which(backsolve!, (DevChol_Mix, VecF32)).module === ext_cuda
        DevChol_F16 = DeviceCholesky{MatF16, Nothing, VecI32}
        @test which(factorize!!, (DevChol_F16, CholeskyFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevChol_F16, VecF16)).module === GeometricSolvers
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

@testset "GeometricSolversCUDSSExt extension loading, dispatch and JET: $T" for T in (
    Float32, Float64, ComplexF32, ComplexF64)
    ext = Base.get_extension(GeometricSolvers, :GeometricSolversCUDSSExt)
    @test ext !== nothing
    for Ti in (Int32, Int64), Mem in (CUDA.DeviceMemory, CUDA.UnifiedMemory)

        Mat = CuSparseMatrixCSR{T, Ti}
        Vec = CuArray{T, 1, Mem}
        Fac = ext.CUDSSLU{T, Mat, CudssSolver{T, Ti}}
        @test which(init_linear, (LUFactorization{Nothing}, Mat, Vec)).module === ext
        @test which(init_linear, (LUFactorization{real(T)}, Mat, Vec)).module === ext
        @test which(factorize!!, (Fac, LUFactorization{Nothing}, Mat)).module === ext
        @test which(ldiv!!, (Vec, Fac, Vec)).module === ext
        @test which(returncode, (Fac,)).module === ext
        @test which(GeometricSolvers.factor_diagnostics,
            (Vec, Fac, Vec, real(T))).module === ext
        JET.test_call(init_linear, (LUFactorization{Nothing}, Mat, Vec); target_modules = (ext,))
        JET.test_call(factorize!!, (Fac, LUFactorization{Nothing}, Mat); target_modules = (ext,))
        JET.test_call(ldiv!!, (Vec, Fac, Vec); target_modules = (ext,))
        # a solve with no reuse keeps the factor that `init` built in its state, so the record of
        # the analysis is a cell that every factor of one `init_linear` shares
        @test fieldtype(Fac, :analyses) === Base.RefValue{Int}
        # a step that reads the Jacobian after the factorisation keeps a sparse Jacobian buffer
        for m in (Newton(; globalization = DogLeg()), Newton(; refinement = Refinement()))
            @test only(Base.return_types(GeometricSolvers.init_newton_J,
                (typeof(m), Fac, Nothing, Vec, Vec))) <: CuSparseMatrixCSR
        end
    end
    # a dense device iterate keeps the dense factor of the CUDA extension
    @test which(init_linear, (LUFactorization{Nothing}, Nothing, CuArray{T, 1})).module ===
          GeometricSolvers
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
            κ = Float64(cuda_well_conditioned_cond(A_cpu))
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
                κ = Float64(cuda_well_conditioned_cond(A_cpu))
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
                κ = Float64(cuda_well_conditioned_cond(A_cpu))
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
            κ = Float64(cuda_well_conditioned_cond(A_cpu))
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
            κ = Float64(cuda_well_conditioned_cond(A_cpu))
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

        A_ok = T[4 1; 1 3]
        dA_ok = CuArray(A_ok)
        dx_ok = CuArray(zeros(T, 2))
        prob2 = NonlinearProblem(dx_ok, cuda_noop)

        # The factor buffer passed back as `J`: `getrf!` writes the factors over it, and the
        # aliased call factorises and solves with the matrix it is given.
        fac_al = dense_factor(LUFactorization(), prob2, dx_ok, nothing)
        copyto!(fac_al.factors, dA_ok)
        fac_al = factorize!!(fac_al, LUFactorization(), fac_al.factors)
        @test fac_al.info == 0
        @test returncode(fac_al) === SUCCESS
        ldiv!!(dx_ok, fac_al, CuArray(T[5, 4]))
        @test isapprox(Array(dx_ok), T[1, 1]; rtol = 16 * eps(RT))

        # Fallback to ka_factorize_svd! when gesvdj! info_c != 0
        fac_svd_fb = dense_factor(SVDFactorization(), prob2, dx_ok, nothing)
        fill!(fac_svd_fb.U, zero(T))
        fill!(fac_svd_fb.σ̂, zero(RT))
        fill!(fac_svd_fb.V, zero(T))
        fac_svd_fb = ext_cuda._cusolver_svd_finish!(
            fac_svd_fb, SVDFactorization(), dA_ok, fac_svd_fb.U, fac_svd_fb.σ̂, fac_svd_fb.V,
            1, 0)
        @test fac_svd_fb.info == 0
        @test fac_svd_fb.rank == 2
        @test returncode(fac_svd_fb) === SUCCESS
        ldiv!!(dx_ok, fac_svd_fb, CuArray(T[5, 4]))
        @test isapprox(Array(dx_ok), T[1, 1]; rtol = 16 * eps(RT))

        # SVDFactorization(; rtol) through `factorize!!` and on the Jacobi fallback of `gesvdj!`:
        # the singular values 1, 1e-10, 1e-14, a diagonal with its columns permuted and signed,
        # whose default rank is 1 in Float32 and 3 in Float64. In single precision `gesvdj!`
        # returns 1e-10 and 1e-14 as 0, so a cutoff below the sweep tolerance (rtol 1e-9, 1e-12
        # and 0 in Float32, rtol 0 in Float64) runs on the own Jacobi kernel.
        A_rt = zeros(T, 3, 3)
        A_rt[2, 1], A_rt[3, 2], A_rt[1, 3] = 1, -1e-10, 1e-14
        x_rt = CuArray(zeros(T, 3))
        prob_rt = NonlinearProblem(x_rt, cuda_noop)
        for (m_rt, r_rt) in ((SVDFactorization(), RT === Float32 ? 1 : 3),
            (SVDFactorization(; rtol = 1e-9), 1), (SVDFactorization(; rtol = 1e-12), 2),
            (SVDFactorization(; rtol = 0), 3))
            fac_rt = factorize!!(dense_factor(m_rt, prob_rt, x_rt, nothing), m_rt, CuArray(A_rt))
            @test fac_rt.info == 0
            @test fac_rt.rank == r_rt
            @test all(iszero, Array(fac_rt.U)[:, (r_rt + 1):end])
            fac_fb = dense_factor(m_rt, prob_rt, x_rt, nothing)
            fac_fb = ext_cuda._cusolver_svd_finish!(
                fac_fb, m_rt, CuArray(A_rt), fac_fb.U, fac_fb.σ̂, fac_fb.V, 1, 0)
            @test fac_fb.info == 0
            @test fac_fb.rank == r_rt
            @test all(iszero, Array(fac_fb.U)[:, (r_rt + 1):end])
        end

        # The singular values 1, 1e-3, 1e-5, which `gesvdj!` resolves in every element type: the
        # cutoffs 1e-2 and 1e-4 lie above the sweep tolerance and run on `gesvdj!`, rtol 0 on the
        # own Jacobi kernel
        A_rtb = zeros(T, 3, 3)
        A_rtb[2, 1], A_rtb[3, 2], A_rtb[1, 3] = 1, -1e-3, 1e-5
        for (m_rt, r_rt) in ((SVDFactorization(), 3), (SVDFactorization(; rtol = 1e-2), 1),
            (SVDFactorization(; rtol = 1e-4), 2), (SVDFactorization(; rtol = 0), 3))
            fac_rt = factorize!!(
                dense_factor(m_rt, prob_rt, x_rt, nothing), m_rt, CuArray(A_rtb))
            @test fac_rt.info == 0
            @test fac_rt.rank == r_rt
            @test all(iszero, Array(fac_rt.U)[:, (r_rt + 1):end])
        end

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

    # A seeded Hermitian positive definite matrix `G Gᴴ / n + I` and a right-hand side.
    function cuda_hpd(::Type{T}, n::Int, seed::Integer = 0xc401) where {T}
        rng = Xoshiro(seed + n)
        G = randn(rng, T, n, n)
        A = G * G' / n + Matrix{T}(I, n, n)
        return (A + A') / 2, randn(rng, T, n)
    end

    # The own-kernel solve with the factor `fac`, the reference of the cuSOLVER rows: in `d` where
    # `d` holds the factor element type, else in `fac.rhs`, multiplied by `2^-e` either way.
    function own_cholesky_ldiv!(d::AbstractVector{T}, fac, r::AbstractVector) where {T}
        F = eltype(fac.factors)
        e = fac.scale_exponent
        if T === F
            copyto!(d, r)
            ka_cholesky_backsolve!(fac.factors, d)
            iszero(e) || (d .= ldexp2.(d, -e))
        else
            fac.rhs .= F.(r)
            ka_cholesky_backsolve!(fac.factors, fac.rhs)
            d .= T.(ldexp2.(fac.rhs, -e))
        end
        return d
    end

    @testset "CUDA cuSOLVER Cholesky agrees with the own KA kernel at n = $n" for n in (
        1, 16, 65, 256)
        @testset "T = $T, $method" for T in CUDA_BLAS_ELTYPES,
            method in (CholeskyFactorization(), CholeskyFactorization(Float32))

            A_cpu, b_cpu = cuda_hpd(T, n)
            κ = Float64(cond(A_cpu, Inf))
            # the round-off of the coarser of the working and the factorisation precision
            ε = Float64(method isa CholeskyFactorization{Nothing} ? eps(real(T)) :
                        eps(Float32))
            dA = CuArray(A_cpu)
            db = CuArray(b_cpu)
            fac_ext = dense_factor(method, nothing, CuArray(zeros(T, n)), nothing)
            fac_ka = dense_factor(method, nothing, CuArray(zeros(T, n)), nothing)
            fac_ext = factorize!!(fac_ext, method, dA)
            fac_ka = ka_factorize_cholesky!(fac_ka, dA)
            @test fac_ext.info == fac_ka.info == 0
            @test fac_ext.scale_exponent == fac_ka.scale_exponent
            @test returncode(fac_ext) === SUCCESS
            dx_ext = ldiv!!(CuArray(zeros(T, n)), fac_ext, db)
            dx_ka = own_cholesky_ldiv!(CuArray(zeros(T, n)), fac_ka, db)
            x_ext = Array(dx_ext)
            x_ka = Array(dx_ka)
            @test norm(x_ext - x_ka, Inf) <= 8 * n * ε * κ * norm(x_ka, Inf)
            # both write `L` with zeros above the diagonal
            L = Array(fac_ext.factors)
            @test all(iszero, (L[i, j] for j in 2:n for i in 1:(j - 1)))
        end
    end

    @testset "CUDA cuSOLVER Cholesky: info as the own kernel gives it: $T" for T in CUDA_BLAS_ELTYPES
        m = CholeskyFactorization()
        function both(A)
            fac_ext = factorize!!(
                dense_factor(m, nothing, CuArray(zeros(T, size(A, 1))), nothing), m, CuArray(A))
            fac_ka = ka_factorize_cholesky!(
                dense_factor(m, nothing, CuArray(zeros(T, size(A, 1))), nothing), CuArray(A))
            return fac_ext, fac_ka
        end
        # not positive definite: the order of the first leading minor that is not positive
        for (A, info) in ((Matrix(Diagonal(T[1, 2, -1, 3])), 3),
            (Matrix(Diagonal(T[0, 1])), 1), (fill(T(-4), 1, 1), 1))
            fac_ext, fac_ka = both(A)
            @test fac_ext.info == fac_ka.info == info
            @test returncode(fac_ext) === SINGULAR
        end
        # a non-finite entry gives info = 1 before `potrf`, which returns 0 for some of them
        n = 4
        A, _ = cuda_hpd(T, n)
        for (i, j, v) in ((n, 1, T(NaN)), (1, n, T(Inf)))
            A_bad = copy(A)
            A_bad[i, j] = v
            fac_ext, fac_ka = both(A_bad)
            @test fac_ext.info == fac_ka.info == 1
            @test fac_ext.scale_exponent == 0
            @test isequal(Array(fac_ext.factors), A_bad)
        end
        # n = 0 calls nothing and gives SUCCESS
        fac_ext, fac_ka = both(zeros(T, 0, 0))
        @test returncode(fac_ext) === SUCCESS
        @test isempty(ldiv!!(CuArray(T[]), fac_ext, CuArray(T[])))
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

    # The CSR arrays of a host `SparseMatrixCSC` `A`, `(rowptr, colval, nzval)`, and the position in
    # `nzval` of each diagonal entry, which the residual's Jacobian adds to.
    function csr_arrays(A::SparseMatrixCSC)
        At = sparse(transpose(A))
        diagpos = [only(k for k in nzrange(At, i) if rowvals(At)[k] == i)
                   for i in axes(A, 1)]
        return At.colptr, At.rowval, nonzeros(At), diagpos
    end

    # The residual `F(z) = A z + p z.^3 / 5 - b` and its analytic Jacobian, which writes the
    # nonzeros of `J`, the pattern of `A`, as `nzA`, the nonzeros of `A`, plus `3 p z.^2 / 5` at
    # `diagpos`.
    function cubic_sparse_problem(A, nzA, diagpos, b)
        F!(r, z, p) = (mul!(r, A, z); r .+= p .* z .^ 3 ./ 5 .- b; r)
        function J!(J, z, p)
            nz = nonzeros(J)
            copyto!(nz, nzA)
            view(nz, diagpos) .+= (3 * p / 5) .* z .^ 2
            return J
        end
        return NonlinearProblem(F!, J!; jacobian_prototype = A)
    end

    @testset "cuDSS sparse LU through LUFactorization: N = $N, T = $T" for N in (125, 768),
        T in CUDA_BLAS_ELTYPES

        ext = Base.get_extension(GeometricSolvers, :GeometricSolversCUDSSExt)
        R = real(T)
        rng = Xoshiro(0x5eed)
        Ah = _sparse_block_matrix(N, T)
        n = size(Ah, 1)
        p = one(R)
        z_star = T <: Complex ? T.(randn(rng, n) .+ im .* randn(rng, n)) ./ 4 :
                 T.(randn(rng, n)) ./ 4
        b = Ah * z_star .+ p .* z_star .^ 3 ./ 5
        z0 = z_star .+ one(R) / 5
        rowptr, colval, nzval, diagpos = csr_arrays(Ah)
        Ad = CuSparseMatrixCSR{T, Int32}(CuArray{Int32}(rowptr), CuArray{Int32}(colval),
            CuArray(nzval), size(Ah))
        prob_d = cubic_sparse_problem(Ad, CuArray(nzval), CuArray(diagpos), CuArray(b))
        tol = R === Float32 ? 1.0f-4 : 1e-10

        solver = init(prob_d, CuArray(z0), Newton(), p; f_abstol = tol, f_reltol = zero(R))
        fac0 = solver.state.fac
        @test fac0 isa ext.CUDSSLU{T}
        @test fac0.analyses[] == 0
        @test solver.state.J === fac0.factors
        x, st = solve!(solver)
        @test st.code === SUCCESS
        @test st.iterations >= 2
        @test Array(x) != z0
        # every step factorised, and only the first analysed
        @test solver.state.fac.analyses[] == 1
        @test isapprox(Array(x), z_star; rtol = R === Float32 ? 1.0f-3 : 1e-9)

        # The linear solves at the root, at the start and at the planted solution agree with
        # UMFPACK on the host copy of the Jacobian, in `Float64` or `ComplexF64` for a `Float32`
        # or `ComplexF32` `T`, within `8n·eps(T)·κ∞`; three more refactorisations keep the one
        # analysis.
        W = T <: Complex ? ComplexF64 : Float64
        function test_umfpack_agreement(solver, z)
            Jd = copy(solver.state.J)
            prob_d.J(Jd, z, p)
            fac = factorize!!(solver.state.fac, LUFactorization(), Jd)
            @test returncode(fac) === SUCCESS
            @test fac.analyses[] == 1
            rhs = T <: Complex ? T.(range(0.1, 1.0; length = n) .+ im) :
                  T.(range(0.1, 1.0; length = n))
            d = Array(ldiv!!(CuArray(zero(rhs)), fac, CuArray(rhs)))
            # the CSR arrays of `J` are the CSC arrays of its transpose
            Jt = SparseMatrixCSC(n, n, rowptr, colval, Array(Jd.nzVal))
            Jw = SparseMatrixCSC{W, Int}(sparse(transpose(Jt)))
            fac_h = factorize!!(init_linear(UMFPACKFactorization(), Jw, zeros(W, n)),
                UMFPACKFactorization(), Jw)
            d_ref = ldiv!!(zeros(W, n), fac_h, W.(rhs))
            κ = cond(Matrix(Jw), Inf)
            @test norm(W.(d) - d_ref, Inf) / norm(d_ref, Inf) <= 8 * n * eps(R) * κ
        end
        for z in (x, CuArray(z0), CuArray(z_star))
            test_umfpack_agreement(solver, z)
        end

        # a `DogLeg` trial loop and a refinement read `J` after the factorisation, so they keep a
        # sparse buffer of the factor's pattern apart from the factor's matrix
        for m in (Newton(; globalization = DogLeg()), Newton(; refinement = Refinement()))
            solver_m = init(prob_d, CuArray(z0), m, p; f_abstol = tol, f_reltol = zero(R))
            @test solver_m.state.J isa CuSparseMatrixCSR{T, Int32}
            @test solver_m.state.J !== solver_m.state.fac.factors
            x_m, st_m = solve!(solver_m)
            @test st_m.code === SUCCESS
            @test Array(x_m) != z0
            @test solver_m.state.fac.analyses[] == 1
            @test isapprox(Array(x_m), z_star; rtol = R === Float32 ? 1.0f-3 : 1e-9)
            for z in (x_m, CuArray(z0), CuArray(z_star))
                test_umfpack_agreement(solver_m, z)
            end
        end

        # a non-finite entry gives `SINGULAR` without a cuDSS call; mixed precision and another
        # pattern raise
        Jn = copy(solver.state.J)
        CUDA.@allowscalar Jn.nzVal[1] = T(NaN)
        fac_n = factorize!!(solver.state.fac, LUFactorization(), Jn)
        @test returncode(fac_n) === SINGULAR
        @test fac_n.analyses[] == 1
        @test_throws r"^ArgumentError: sparse LUFactorization with a CuSparseMatrixCSR" init(
            prob_d, CuArray(z0),
            Newton(; linsolve = LUFactorization(R === Float64 ? Float32 :
                                           Float64)), p)
        Ad2 = CuSparseMatrixCSR{T, Int32}(CuArray{Int32}(1:(n + 1)), CuArray{Int32}(1:n),
            CUDA.ones(T, n), (n, n))
        @test_throws r"^ArgumentError: sparse Jacobian sparsity pattern differs" factorize!!(
            solver.state.fac, LUFactorization(), Ad2)
    end

    BLAS.set_num_threads(old_blas_threads)
else
    @test_skip CUDA.functional()
end
