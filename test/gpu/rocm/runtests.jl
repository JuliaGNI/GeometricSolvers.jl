# The tests of `GeometricSolversAMDGPUExt`, run by hand on a machine with a ROCm device:
# `julia --startup-file=no --project=test/gpu/rocm test/gpu/rocm/runtests.jl`. The extension
# loading, dispatch and JET rows run without a device; every row that factorises on a ROCm array
# runs where `AMDGPU.functional()`.

using AMDGPU
using GPUArraysCore: allowscalar
using JET: JET
using LinearAlgebra: BLAS, Diagonal, I, cond, norm, opnorm
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: DeviceCholesky, DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: CholeskyFactorization, LUFactorization, NonlinearProblem,
                        QRFactorization, SVDFactorization
using GeometricSolvers: SINGULAR, SUCCESS, backsolve!, backsolve_shifted!, dense_factor,
                        factorize!!, ka_cholesky_backsolve!, ka_factorize_cholesky!,
                        ka_factorize_lu!, ka_factorize_qr!, ldexp2, ldiv!!, returncode

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
            # The same-type `ldiv!!` of an `m × n` factor solves through this method.
            @test which(backsolve_shifted!, (DevQR, Vec, Vec)).module === ext_rocm

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
        @test which(backsolve_shifted!, (DevQR_F16, VecF16, VecF16)).module ===
              GeometricSolvers
        DevSVD_F16 = DeviceSVD{MatF16, VecF16, VecI32, VecF16, Nothing}
        @test which(factorize!!, (DevSVD_F16, SVDFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers

        # Cholesky: rocSOLVER `potrf`/`potrs` for the BLAS types, the mixed pair included, and
        # the own kernel for `Float16`
        for T in ROCM_BLAS_ELTYPES
            Mat = ROCArray{T, 2, B}
            Vec = ROCArray{T, 1, B}
            DevChol = DeviceCholesky{Mat, Nothing, VecI32}
            @test which(factorize!!, (DevChol, CholeskyFactorization{Nothing}, Mat)).module ===
                  ext_rocm
            @test which(backsolve!, (DevChol, Vec)).module === ext_rocm
        end
        DevChol_Mix = DeviceCholesky{MatF32, VecF32, VecI32}
        @test which(factorize!!, (DevChol_Mix, CholeskyFactorization{Float32}, MatF64)).module ===
              ext_rocm
        @test which(backsolve!, (DevChol_Mix, VecF32)).module === ext_rocm
        DevChol_F16 = DeviceCholesky{MatF16, Nothing, VecI32}
        @test which(factorize!!, (DevChol_F16, CholeskyFactorization{Nothing}, MatF16)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevChol_F16, VecF16)).module === GeometricSolvers
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
    JET.test_call(backsolve_shifted!, (DevQR, Vec, Vec); target_modules = (ext_rocm,))
    DevChol = DeviceCholesky{Mat, Nothing, VecI32}
    JET.test_call(factorize!!, (DevChol, CholeskyFactorization{Nothing}, Mat);
        target_modules = (ext_rocm,))
    JET.test_call(backsolve!, (DevChol, Vec); target_modules = (ext_rocm,))
end

# The data of the `m × n` least-squares QR rows, `m > n`. For complex `T` the entries have
# imaginary parts, so the LAPACK `τ` of `A` is not real and `conj.(τ) != τ`: a solve that reads
# the reflectors of the factor with `conj.(τ)` in place of `τ`, as `ka_qr_backsolve!` does with
# the factor of `geqrf!`, misses the least-squares solution.
function rect_qr_data(T, m_rows, n)
    A = T[(i == j ? 3 : 0) + (i + 2j) % 5 + (T <: Complex ? im * ((i * j) % 3) : 0)
          for i in 1:m_rows, j in 1:n]
    b = T[mod(7i, 5) - 2 + (T <: Complex ? im * (i % 2) : 0) for i in 1:m_rows]
    return A, b
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
            κ = Float64(cuda_well_conditioned_cond(A_cpu))
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

                # Above n = 256 the matrix has columns with nothing below the diagonal, and
                # rocSOLVER's blocked complex `geqrf!` turns their τ = 0 into NaN (K96).
                broken_ext = M === QRFactorization && T <: Complex && n > 256  # K96
                fac_ext = factorize!!(fac_ext, m, dA)
                @test fac_ext.info == 0 broken = broken_ext
                @test returncode(fac_ext) === SUCCESS broken = broken_ext
                ldiv!!(dx_ext, fac_ext, db)

                fac_ka = M === LUFactorization ? ka_factorize_lu!(fac_ka, dA) :
                         ka_factorize_qr!(fac_ka, dA)
                @test fac_ka.info == 0
                own_kernel_ldiv!(dx_ka, fac_ka, db)

                x_ext = Array(dx_ext)
                x_ka = Array(dx_ka)
                @test norm(x_ext - x_ka, Inf) / max(norm(x_ka, Inf), epsT) <= bound broken = broken_ext
            end
        end
    end

    @testset "ROCm rocSOLVER mixed TF = Float32 at n = $n" for n in (16, 256)
        @testset "T = $T" for T in (Float64, ComplexF64)
            A_cpu, b_cpu = cuda_well_conditioned(T, n)
            κ = Float64(cuda_well_conditioned_cond(A_cpu))
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

    # The `m × n` QR factor of a least-squares problem, `m > n`: `ormqr!` applies Qᴴ to the
    # length-`m` right-hand side and `rocBLAS.trsv!` solves with the leading `n × n` block. The
    # mixed `ldiv!!` and a direct `backsolve!` reach the extension's method with the length-`m`
    # right-hand side. The bound is that of the `JLArray` least-squares rows of
    # `test/nonlinear/leastsquares.jl`, in the factor's precision.
    @testset "ROCm rectangular QR, m = 6 > n = 3: $T" for T in ROCM_BLAS_ELTYPES
        m_rows, n = 6, 3
        A_h, b_h = rect_qr_data(T, m_rows, n)
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

    # The QR solves against the host solve `A \ b`: the same-type `ldiv!!` of an `m × n` factor,
    # `m > n`, which reaches the extension's `backsolve_shifted!`; the mixed `ldiv!!` of a
    # `Float32` factor and a `Float64` iterate, which reaches its `backsolve!` with the length-`m`
    # right-hand side; and the square solve. The bound is in the factor's precision.
    @testset "ROCm QR solves against the host solve: $T" for T in ROCM_BLAS_ELTYPES
        methods = real(T) === Float64 ? (QRFactorization(), QRFactorization(Float32)) :
                  (QRFactorization(),)
        shapes = ((6, 3), (40, 7), (16, 16))
        @testset "m = $m_rows, n = $n, $method" for (m_rows, n) in shapes, method in methods

            A_h, b_h = m_rows == n ? cuda_well_conditioned(T, n) :
                       rect_qr_data(T, m_rows, n)
            x_ref = A_h \ b_h
            d = ROCArray(zeros(T, n))
            prob = m_rows == n ? NonlinearProblem(d, rocm_noop) :
                   LeastSquaresProblem((r, x, p) -> r, m_rows)
            fac = dense_factor(method, prob, d, nothing)
            @test size(fac.factors) == (m_rows, n)
            fac = factorize!!(fac, method, ROCArray(A_h))
            @test returncode(fac) === SUCCESS
            rtol = 32 * sqrt(eps(real(eltype(fac.factors))))
            ldiv!!(d, fac, ROCArray(b_h))
            @test isapprox(Array(d), x_ref; rtol)
        end
    end

    # The `m × 0` factor of a least-squares problem with no unknowns: `factorize!!` returns before
    # `geqrf!`, and the same-type and the mixed `ldiv!!` reach the extension's `backsolve!` with
    # the nonzero length-`m` right-hand side, so `ormqr!` runs with `k = 0` and `trsv!` on `0 × 0`
    # views.
    @testset "ROCm QR of an m × 0 factor, m = 5: $T" for T in ROCM_BLAS_ELTYPES
        m_rows = 5
        _, b_h = rect_qr_data(T, m_rows, 1)
        methods = real(T) === Float64 ? (QRFactorization(), QRFactorization(Float32)) :
                  (QRFactorization(),)
        for method in methods
            d = ROCArray(zeros(T, 0))
            prob = LeastSquaresProblem((r, x, p) -> r, m_rows)
            fac = dense_factor(method, prob, d, nothing)
            @test size(fac.factors) == (m_rows, 0)
            fac = factorize!!(fac, method, ROCArray(zeros(T, m_rows, 0)))
            @test fac.info == 0
            @test returncode(fac) === SUCCESS
            @test length(ldiv!!(d, fac, ROCArray(b_h))) == 0
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

    # A seeded Hermitian positive definite matrix `G Gᴴ / n + I` and a right-hand side.
    function rocm_hpd(::Type{T}, n::Int, seed::Integer = 0xc401) where {T}
        rng = Xoshiro(seed + n)
        G = randn(rng, T, n, n)
        A = G * G' / n + Matrix{T}(I, n, n)
        return (A + A') / 2, randn(rng, T, n)
    end

    # The own-kernel solve with the factor `fac`, the reference of the rocSOLVER rows: in `d` where
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

    @testset "ROCm rocSOLVER Cholesky agrees with the own KA kernel at n = $n" for n in (
        1, 16, 65, 256)
        @testset "T = $T, $method" for T in ROCM_BLAS_ELTYPES,
            method in (CholeskyFactorization(), CholeskyFactorization(Float32))

            A_cpu, b_cpu = rocm_hpd(T, n)
            κ = Float64(cond(A_cpu, Inf))
            # the round-off of the coarser of the working and the factorisation precision
            ε = Float64(method isa CholeskyFactorization{Nothing} ? eps(real(T)) :
                        eps(Float32))
            dA = ROCArray(A_cpu)
            db = ROCArray(b_cpu)
            fac_ext = dense_factor(method, nothing, ROCArray(zeros(T, n)), nothing)
            fac_ka = dense_factor(method, nothing, ROCArray(zeros(T, n)), nothing)
            fac_ext = factorize!!(fac_ext, method, dA)
            fac_ka = ka_factorize_cholesky!(fac_ka, dA)
            @test fac_ext.info == fac_ka.info == 0
            @test fac_ext.scale_exponent == fac_ka.scale_exponent
            @test returncode(fac_ext) === SUCCESS
            dx_ext = ldiv!!(ROCArray(zeros(T, n)), fac_ext, db)
            dx_ka = own_cholesky_ldiv!(ROCArray(zeros(T, n)), fac_ka, db)
            x_ext = Array(dx_ext)
            x_ka = Array(dx_ka)
            @test norm(x_ext - x_ka, Inf) <= 8 * n * ε * κ * norm(x_ka, Inf)
            # both solve the host system to a backward error of `8nε`: one load kernel loads `dA`
            # for both paths, so the agreement above does not see an entry that the load drops or
            # misplaces
            for x in (x_ext, x_ka)
                @test norm(A_cpu * x - b_cpu, Inf) <=
                      8 * n * ε * opnorm(A_cpu, Inf) * norm(x, Inf)
            end
            # both write `L` with zeros above the diagonal
            L = Array(fac_ext.factors)
            @test all(iszero, (L[i, j] for j in 2:n for i in 1:(j - 1)))
        end
    end

    @testset "ROCm rocSOLVER Cholesky: info as the own kernel gives it: $T" for T in ROCM_BLAS_ELTYPES
        m = CholeskyFactorization()
        function both(A)
            fac_ext = factorize!!(
                dense_factor(m, nothing, ROCArray(zeros(T, size(A, 1))), nothing), m,
                ROCArray(A))
            fac_ka = ka_factorize_cholesky!(
                dense_factor(m, nothing, ROCArray(zeros(T, size(A, 1))), nothing),
                ROCArray(A))
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
        A, _ = rocm_hpd(T, n)
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
        @test isempty(ldiv!!(ROCArray(T[]), fac_ext, ROCArray(T[])))
    end

    @testset "ROCm rocSOLVER Cholesky: a vector of another length ($T)" for T in ROCM_BLAS_ELTYPES
        m = CholeskyFactorization()
        # the length is checked before `potrs`: a vector of another length, the empty one
        # included, is refused, by `ldiv!!` and by `backsolve!`, as the own KA solve refuses it
        fac2 = factorize!!(dense_factor(m, nothing, ROCArray(zeros(T, 2)), nothing), m,
            ROCArray(T[4 0; 0 9]))
        @test returncode(fac2) === SUCCESS
        for k in (0, 1, 3)
            msg = Regex("^DimensionMismatch: d has length $k, the Cholesky factor \\(2, 2\\)")
            @test_throws msg ldiv!!(ROCArray(ones(T, k)), fac2, ROCArray(ones(T, k)))
            @test_throws msg backsolve!(fac2, ROCArray(ones(T, k)))
        end
    end

    BLAS.set_num_threads(old_blas_threads)
else
    @test_skip AMDGPU.functional()
end
