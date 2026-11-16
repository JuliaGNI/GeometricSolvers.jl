# The tests of `GeometricSolversAMDGPUExt`, run by hand on a machine with a ROCm device:
# `julia --project=test/gpu/rocm test/gpu/rocm/runtests.jl`. The extension loading, dispatch and
# JET rows run without a device; every row that factorises runs where `AMDGPU.functional()`.

using AMDGPU
using GPUArraysCore: allowscalar
using JET: JET
using LinearAlgebra: BLAS, I, cond, norm
using Test

using GeometricSolvers
using GeometricSolvers: DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization, SVDFactorization
using GeometricSolvers: SINGULAR, SUCCESS, backsolve!, dense_factor, factorize!!,
                        ka_factorize_lu!, ka_factorize_qr!, ldiv!!, returncode

include("../../helpers/own_kernel.jl")

println("AMDGPU ", pkgversion(AMDGPU), ", GeometricSolvers ", pkgversion(GeometricSolvers),
    ", Julia ", VERSION)
println("AMDGPU.functional() = ", AMDGPU.functional())

const ROCM_BLAS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

@testset "GeometricSolversAMDGPUExt extension loading, dispatch, and ROCm census alignment" begin
    ext_rocm = Base.get_extension(GeometricSolvers, :GeometricSolversAMDGPUExt)
    @test ext_rocm !== nothing

    for B in (AMDGPU.Mem.HIPBuffer, AMDGPU.Mem.HostBuffer)
        VecI32 = ROCArray{Int32, 1, B}
        for T in ROCM_BLAS_ELTYPES
            Mat = ROCArray{T, 2, B}
            Vec = ROCArray{T, 1, B}
            VecR = ROCArray{real(T), 1, B}

            DevLU = DeviceLU{Mat, VecI32, Nothing, VecI32}
            @test which(factorize!!, (DevLU, LUFactorization{Nothing}, Mat)).module ===
                  ext_rocm
            @test which(backsolve!, (DevLU, Vec)).module === ext_rocm

            DevQR = DeviceQR{Mat, Vec, Nothing, VecI32}
            @test which(factorize!!, (DevQR, QRFactorization{Nothing}, Mat)).module ===
                  ext_rocm
            @test which(backsolve!, (DevQR, Vec)).module === ext_rocm

            # The SVD stays on the own kernels on ROCm.
            DevSVD = DeviceSVD{Mat, VecR, VecI32, Vec, Nothing}
            @test which(factorize!!, (DevSVD, SVDFactorization{Nothing}, Mat)).module ===
                  GeometricSolvers
            @test which(backsolve!, (DevSVD, Vec)).module === GeometricSolvers
        end

        # A `Float64` Jacobian factorised in `Float32` reaches the rocSOLVER `Float32` methods.
        MatF32 = ROCArray{Float32, 2, B}
        VecF32 = ROCArray{Float32, 1, B}
        MatF64 = ROCArray{Float64, 2, B}
        DevLU_Mix = DeviceLU{MatF32, VecI32, VecF32, VecI32}
        DevQR_Mix = DeviceQR{MatF32, VecF32, VecF32, VecI32}
        @test which(factorize!!, (DevLU_Mix, LUFactorization{Float32}, MatF64)).module ===
              ext_rocm
        @test which(backsolve!, (DevLU_Mix, VecF32)).module === ext_rocm
        @test which(factorize!!, (DevQR_Mix, QRFactorization{Float32}, MatF64)).module ===
              ext_rocm
        @test which(backsolve!, (DevQR_Mix, VecF32)).module === ext_rocm

        # Float16 is no BLAS element type and stays on the KernelAbstractions device kernels.
        MatF16 = ROCArray{Float16, 2, B}
        VecF16 = ROCArray{Float16, 1, B}
        DevLU_F16 = DeviceLU{MatF16, VecI32, Nothing, VecI32}
        DevQR_F16 = DeviceQR{MatF16, VecF16, Nothing, VecI32}
        @test which(factorize!!, (DevLU_F16, LUFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevLU_F16, VecF16)).module === GeometricSolvers
        @test which(factorize!!, (DevQR_F16, QRFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevQR_F16, VecF16)).module === GeometricSolvers
        DevSVD_F16 = DeviceSVD{MatF16, VecF16, VecI32, VecF16, Nothing}
        @test which(factorize!!, (DevSVD_F16, SVDFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
    end
end

@testset "GeometricSolversAMDGPUExt rocSOLVER calls, JET: $T" for T in (Float64, ComplexF64)
    ext_rocm = Base.get_extension(GeometricSolvers, :GeometricSolversAMDGPUExt)
    VecI32 = ROCArray{Int32, 1, AMDGPU.Mem.HIPBuffer}
    Mat = ROCArray{T, 2, AMDGPU.Mem.HIPBuffer}
    Vec = ROCArray{T, 1, AMDGPU.Mem.HIPBuffer}
    DevLU = DeviceLU{Mat, VecI32, Nothing, VecI32}
    DevQR = DeviceQR{Mat, Vec, Nothing, VecI32}
    JET.test_call(factorize!!, (DevLU, LUFactorization{Nothing}, Mat);
        target_modules = (ext_rocm,))
    JET.test_call(backsolve!, (DevLU, Vec); target_modules = (ext_rocm,))
    JET.test_call(factorize!!, (DevQR, QRFactorization{Nothing}, Mat);
        target_modules = (ext_rocm,))
    JET.test_call(backsolve!, (DevQR, Vec); target_modules = (ext_rocm,))
end

if AMDGPU.functional()
    println()
    AMDGPU.versioninfo()
    println()

    allowscalar(false)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    rocm_noop(r, y, p) = nothing

    # Whether rocSOLVER accepts n = 0: the extension returns before it calls rocSOLVER on an empty
    # matrix, so this line only records the library's answer.
    let A0 = ROCArray(zeros(Float64, 0, 0)), ipiv0 = ROCArray(zeros(Int32, 0))
        msg = try
            _, _, info0 = AMDGPU.rocSOLVER.getrf!(A0, ipiv0)
            "rocSOLVER getrf! accepts n = 0, info = $info0"
        catch err
            "rocSOLVER getrf! rejects n = 0: " * sprint(showerror, err)
        end
        println(msg)
    end

    @testset "ROCm rocSOLVER agreement with own KA kernels at n = $n" for n in (16, 256, 10^4)
        @testset "T = $T" for T in ROCM_BLAS_ELTYPES
            A_cpu, b_cpu = cuda_well_conditioned(T, n)
            κ = Float64(cond(A_cpu, Inf))
            @test κ <= 10.0
            epsT = Float64(eps(real(T)))
            bound = 8 * n * epsT * κ

            dA = ROCArray(A_cpu)
            db = ROCArray(b_cpu)
            prob_dev = NonlinearProblem(ROCArray(zeros(T, n)), rocm_noop)

            for M in (LUFactorization, QRFactorization)
                m = M()
                dx_ext = ROCArray(zeros(T, n))
                dx_ka = ROCArray(zeros(T, n))
                fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
                fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)

                fac_ext = factorize!!(fac_ext, m, dA)
                @test fac_ext.info == 0
                @test returncode(fac_ext) === SUCCESS
                ldiv!!(dx_ext, fac_ext, db)

                fac_ka = M === LUFactorization ? ka_factorize_lu!(fac_ka, dA) :
                         ka_factorize_qr!(fac_ka, dA)
                @test fac_ka.info == 0
                own_kernel_ldiv!(dx_ka, fac_ka, db)

                x_ext = Array(dx_ext)
                x_ka = Array(dx_ka)
                @test norm(x_ext - x_ka, Inf) / max(norm(x_ka, Inf), epsT) <= bound
            end
        end
    end

    @testset "ROCm rocSOLVER mixed TF = Float32 at n = $n" for n in (16, 256)
        @testset "T = $T" for T in (Float64, ComplexF64)
            A_cpu, b_cpu = cuda_well_conditioned(T, n)
            κ = Float64(cond(A_cpu, Inf))
            bound = 8 * n * Float64(eps(Float32)) * κ
            dA = ROCArray(A_cpu)
            db = ROCArray(b_cpu)
            dx_ext = ROCArray(zeros(T, n))
            dx_ka = ROCArray(zeros(T, n))
            prob_dev = NonlinearProblem(dx_ext, rocm_noop)
            for M in (LUFactorization, QRFactorization)
                m = M(Float32)
                fac_ext = dense_factor(m, prob_dev, dx_ext, nothing)
                fac_ka = dense_factor(m, prob_dev, dx_ka, nothing)
                fac_ext = factorize!!(fac_ext, m, dA)
                @test fac_ext.info == 0
                @test returncode(fac_ext) === SUCCESS
                ldiv!!(dx_ext, fac_ext, db)

                fac_ka = M === LUFactorization ? ka_factorize_lu!(fac_ka, dA) :
                         ka_factorize_qr!(fac_ka, dA)
                own_kernel_ldiv!(dx_ka, fac_ka, db)

                x_ext = Array(dx_ext)
                x_ka = Array(dx_ka)
                @test norm(x_ext - x_ka, Inf) /
                      max(norm(x_ka, Inf), Float64(eps(Float32))) <= bound
            end
        end
    end

    # The `m × n` QR factor of a least-squares problem, `m > n`: `rocBLAS.trsv!` takes a square
    # matrix only, so its solve runs on the own kernel. The mixed `ldiv!!` and a direct `backsolve!`
    # reach the extension's method with the length-`m` right-hand side. The bound is that of the
    # `JLArray` least-squares rows of `test/nonlinear/leastsquares.jl`, in the factor's precision.
    @testset "ROCm rectangular QR, m = 6 > n = 3: $T" for T in ROCM_BLAS_ELTYPES
        m_rows, n = 6, 3
        A_h = T[(i == j ? 3 : 0) + (i + 2j) % 5 for i in 1:m_rows, j in 1:n]
        b_h = T[1, -2, 3, 0, 2, -1]
        x_ref = A_h \ b_h
        prob = LeastSquaresProblem((r, x, p) -> r, m_rows)
        methods = real(T) === Float64 ? (QRFactorization(), QRFactorization(Float32)) :
                  (QRFactorization(),)
        for method in methods
            d = ROCArray(zeros(T, n))
            fac = dense_factor(method, prob, d, nothing)
            @test size(fac.factors) == (m_rows, n)
            fac = factorize!!(fac, method, ROCArray(A_h))
            @test returncode(fac) === SUCCESS
            F = eltype(fac.factors)
            rtol = 32 * sqrt(eps(real(F)))
            ldiv!!(d, fac, ROCArray(b_h))
            @test isapprox(Array(d), x_ref; rtol)
            # The factors are those of `J / 2^e`, `e = fac.scale_exponent`.
            d_m = ROCArray(F.(b_h))
            backsolve!(fac, d_m)
            @test isapprox(Array(d_m)[1:n] .* F(2)^(-fac.scale_exponent), F.(x_ref); rtol)
        end
    end

    @testset "ROCm rocSOLVER 0x0, singular and non-finite rows: $T" for T in ROCM_BLAS_ELTYPES
        ext_rocm = Base.get_extension(GeometricSolvers, :GeometricSolversAMDGPUExt)
        RT = real(T)
        for M in (LUFactorization, QRFactorization)
            m = M()
            x0 = ROCArray(zeros(T, 0))
            prob0 = NonlinearProblem(x0, rocm_noop)
            fac0 = factorize!!(
                dense_factor(m, prob0, x0, nothing), m, ROCArray(zeros(T, 0, 0)))
            @test fac0.info == 0
            @test returncode(fac0) === SUCCESS
            d0 = ldiv!!(similar(x0), fac0, ROCArray(zeros(T, 0)))
            @test length(d0) == 0
        end

        x4 = ROCArray(zeros(T, 4))
        prob4 = NonlinearProblem(x4, rocm_noop)
        for M in (LUFactorization, QRFactorization)
            m = M()
            # The zero matrix
            fac_z = factorize!!(
                dense_factor(m, prob4, x4, nothing), m, ROCArray(zeros(T, 4, 4)))
            @test fac_z.info > 0
            @test returncode(fac_z) === SINGULAR

            # Column 2 zero: the first zero pivot, and the first zero of the diagonal of R, is 2
            A_c2 = Matrix{T}(I, 4, 4)
            A_c2[:, 2] .= zero(T)
            fac_c2 = factorize!!(dense_factor(m, prob4, x4, nothing), m, ROCArray(A_c2))
            @test fac_c2.info == 2
            @test returncode(fac_c2) === SINGULAR

            # Two equal rows, for LU
            if M === LUFactorization
                A_eq = T[1 2 3 4; 1 2 3 4; 0 1 1 1; 0 0 1 2]
                fac_eq = factorize!!(
                    dense_factor(m, prob4, x4, nothing), m, ROCArray(A_eq))
                @test fac_eq.info > 0
                @test returncode(fac_eq) === SINGULAR
            end
        end

        # The QR info helper on a non-finite diagonal of R; the zero diagonal is checked through
        # `factorize!!` above.
        A_qr_nf = Matrix{T}(I, 4, 4)
        A_qr_nf[3, 3] = T(RT(NaN))
        @test ext_rocm._rocsolver_qr_info(ROCArray(A_qr_nf)) == 3

        # The factor buffer passed back as `J`: `getrf!` writes the factors over it, and the
        # aliased call factorises and solves with the matrix it is given.
        A_ok = T[4 1; 1 3]
        dx_ok = ROCArray(zeros(T, 2))
        prob2 = NonlinearProblem(dx_ok, rocm_noop)
        fac_al = dense_factor(LUFactorization(), prob2, dx_ok, nothing)
        copyto!(fac_al.factors, ROCArray(A_ok))
        fac_al = factorize!!(fac_al, LUFactorization(), fac_al.factors)
        @test fac_al.info == 0
        @test returncode(fac_al) === SUCCESS
        ldiv!!(dx_ok, fac_al, ROCArray(T[5, 4]))
        @test isapprox(Array(dx_ok), T[1, 1]; rtol = 16 * eps(RT))
    end

    BLAS.set_num_threads(old_blas_threads)
else
    @test_skip AMDGPU.functional()
end
