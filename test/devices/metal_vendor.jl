using GPUArraysCore: allowscalar
using LinearAlgebra: LinearAlgebra, BLAS, Diagonal, I, LU, cond, lu!, norm, opnorm
using Metal
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization, SVDFactorization
using GeometricSolvers: SINGULAR, SUCCESS, backsolve!, dense_factor, factorize!!,
                        ka_factorize_lu!, ka_lu_backsolve!, ldiv!!, returncode

include("../helpers/matrix.jl")
include("../helpers/metal_device.jl")

if Metal.functional() && !on_virtual_gpu()
    allowscalar(false)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    metal_vendor_noop(r, y, p) = nothing

    # Asserts that `factorize!!` and `backsolve!` on a factor of `T` on storage `S` keep the own
    # KA kernels of `src/linear/ka/`, for an element type that no MPS path covers. `Float16` is
    # reached through `which` only, never through a factorisation: a `Float16` factorisation on
    # Metal crashes the process (§10.D finding 2).
    function ka_retention(T::Type, S::Type)
        Mat = MtlMatrix{T, S}
        Vec = MtlVector{T, S}
        VecR = MtlVector{real(T), S}
        VecI32 = MtlVector{Int32, S}
        DevLU = DeviceLU{Mat, VecI32, Nothing, VecI32}
        m_lu_fac = which(factorize!!, (DevLU, LUFactorization{Nothing}, Mat))
        m_lu_bs = which(backsolve!, (DevLU, Vec))
        @test m_lu_fac.module === GeometricSolvers
        @test m_lu_bs.module === GeometricSolvers
        @test endswith(replace(String(m_lu_fac.file), '\\' => '/'), "src/linear/ka/lu.jl")
        @test endswith(replace(String(m_lu_bs.file), '\\' => '/'), "src/linear/ka/lu.jl")
        DevQR = DeviceQR{Mat, Vec, Nothing, VecI32}
        @test which(factorize!!, (DevQR, QRFactorization{Nothing}, Mat)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevQR, Vec)).module === GeometricSolvers
        DevSVD = DeviceSVD{Mat, VecR, VecI32, Vec, Nothing}
        @test which(factorize!!, (DevSVD, SVDFactorization{Nothing}, Mat)).module ===
              GeometricSolvers
        @test which(backsolve!, (DevSVD, Vec)).module === GeometricSolvers
        return nothing
    end

    @testset "GeometricSolversMetalExt dispatch, KA retention, and Metal census alignment" begin
        ext_mtl = Base.get_extension(GeometricSolvers, :GeometricSolversMetalExt)
        @test ext_mtl !== nothing

        for S in (Metal.PrivateStorage, Metal.SharedStorage)
            MatF32 = MtlMatrix{Float32, S}
            VecF32 = MtlVector{Float32, S}
            VecI32 = MtlVector{Int32, S}
            DevLU_F32 = DeviceLU{MatF32, VecI32, Nothing, VecI32}
            DevLU_F32_Mix = DeviceLU{MatF32, VecI32, VecF32, VecI32}

            @test which(factorize!!, (DevLU_F32, LUFactorization{Nothing}, MatF32)).module ===
                  ext_mtl
            @test which(backsolve!, (DevLU_F32, VecF32)).module === ext_mtl
            @test which(
                factorize!!, (DevLU_F32_Mix, LUFactorization{Float32}, MatF32)).module ===
                  ext_mtl
            @test which(backsolve!, (DevLU_F32_Mix, VecF32)).module === ext_mtl

            ci_fac = string(only(code_typed(
                factorize!!, (DevLU_F32, LUFactorization{Nothing}, MatF32);
                optimize = false))[1])
            @test occursin("lu!", ci_fac) && !occursin("ka_factorize_lu!", ci_fac)
            ci_bs = string(only(code_typed(
                backsolve!, (DevLU_F32, VecF32); optimize = false))[1])
            @test occursin("ldiv!", ci_bs) && !occursin("ka_lu_backsolve!", ci_bs)

            # The element types that no MPS path covers keep the own KA kernels
            ka_retention(ComplexF32, S)
            ka_retention(Float16, S)
        end

        # Metal capability census alignment: `Float32` `lu!` is the sole pass factorisation cell
        # of `ELTYPES` that `GeometricSolversMetalExt` covers.
        VecI32_P = MtlVector{Int32, Metal.PrivateStorage}
        for T in ELTYPES
            Mat = MtlMatrix{T, Metal.PrivateStorage}
            Vec = MtlVector{T, Metal.PrivateStorage}
            VecR = MtlVector{real(T), Metal.PrivateStorage}
            DevLU = DeviceLU{Mat, VecI32_P, Nothing, VecI32_P}
            DevQR = DeviceQR{Mat, Vec, Nothing, VecI32_P}
            DevSVD = DeviceSVD{Mat, VecR, VecI32_P, Vec, Nothing}
            @test (which(factorize!!, (DevLU, LUFactorization{Nothing}, Mat)).module ===
                   ext_mtl) == (T === Float32)
            @test which(factorize!!, (DevQR, QRFactorization{Nothing}, Mat)).module ===
                  GeometricSolvers
            @test which(factorize!!, (DevSVD, SVDFactorization{Nothing}, Mat)).module ===
                  GeometricSolvers
        end
    end

    @testset "Metal MPS LU agreement with own KA kernel at n = $n" for n in (1, 16, 256, 10^4)
        m = LUFactorization()
        rng = Xoshiro(0x5eed + n)
        A_cpu = randn(rng, Float32, n, n) + Float32(2 * n) * Matrix{Float32}(I, n, n)
        b_cpu = randn(rng, Float32, n)
        κ = Float64(cond(A_cpu, Inf))
        @test κ <= 10.0

        dA = MtlArray(A_cpu)
        db = MtlArray(b_cpu)
        dx_ext = MtlArray(zeros(Float32, n))
        dx_ka = MtlArray(zeros(Float32, n))
        prob = NonlinearProblem(dx_ext, metal_vendor_noop)

        fac_ext = dense_factor(m, prob, dx_ext, nothing)
        fac_ka = dense_factor(m, prob, dx_ka, nothing)

        ext_mtl = Base.get_extension(GeometricSolvers, :GeometricSolversMetalExt)
        on_mps = which(factorize!!, (typeof(fac_ext), typeof(m), typeof(dA))).module ===
                 ext_mtl && ext_mtl !== nothing &&
                 !occursin("ka_factorize_lu!",
                     string(only(code_typed(
                         factorize!!, (typeof(fac_ext), typeof(m), typeof(dA));
                         optimize = false))[1]))
        @test on_mps

        fac_ext = factorize!!(fac_ext, m, dA)
        @test fac_ext.info == 0
        @test returncode(fac_ext) === SUCCESS
        ldiv!!(dx_ext, fac_ext, db)

        fac_ka = ka_factorize_lu!(fac_ka, dA)
        @test fac_ka.info == 0
        copyto!(dx_ka, db)
        ka_lu_backsolve!(fac_ka, dx_ka)

        x_ext = Array(dx_ext)
        x_ka = Array(dx_ka)
        eps32 = Float64(eps(Float32))
        bwd = norm(A_cpu * x_ext - b_cpu, Inf) / (opnorm(A_cpu, Inf) * norm(x_ext, Inf))
        @test bwd <= 4 * n * eps32

        rel_diff = norm(x_ext - x_ka, Inf) / max(norm(x_ka, Inf), eps32)
        @test rel_diff <= 8 * n * eps32 * κ

        if n == 16
            # SharedStorage MtlArray and in-place factorize!! / ldiv!! paths
            x_sh = MtlVector{Float32, Metal.SharedStorage}(zeros(Float32, n))
            A_sh = MtlMatrix{Float32, Metal.SharedStorage}(A_cpu)
            b_sh = MtlVector{Float32, Metal.SharedStorage}(b_cpu)
            prob_sh = NonlinearProblem(x_sh, metal_vendor_noop)
            fac_sh = dense_factor(m, prob_sh, x_sh, nothing)
            copyto!(fac_sh.factors, A_sh)
            fac_sh = factorize!!(fac_sh, m, fac_sh.factors)
            @test fac_sh.info == 0
            @test returncode(fac_sh) === SUCCESS
            d_sh = copy(b_sh)
            ldiv!!(d_sh, fac_sh, d_sh)
            @test norm(Array(d_sh) - x_ka, Inf) / max(norm(x_ka, Inf), eps32) <=
                  8 * n * eps32 * κ
        end
    end

    @testset "Metal capability census probe on Float32 lu! and A \\ b" begin
        rng = Xoshiro(0x5eed)
        N = 8
        A0 = randn(rng, N, N) + N * I
        b0 = randn(rng, N)
        A = Float32.(A0)
        b = Float32.(b0)
        Ad = MtlArray(A)
        bd = MtlArray(b)
        Aref = ComplexF64.(A)
        tol = 100 * Float64(eps(Float32))
        relerr(C, Cref) = norm(ComplexF64.(C) .- ComplexF64.(Cref)) /
                          max(norm(ComplexF64.(Cref)), eps())

        F = lu!(copy(Ad))
        Fh = LU(Array(F.factors), Array(F.ipiv), F.info)
        @test relerr(ComplexF64.(Fh.L) * ComplexF64.(Fh.U), Aref[Fh.p, :]) <= tol

        xd = Ad \ bd
        @test relerr(Aref * ComplexF64.(Array(xd)), ComplexF64.(b)) <= tol
    end

    @testset "Metal MPS LU edge cases, singular status mapping, and KA fallback" begin
        ext_mtl = Base.get_extension(GeometricSolvers, :GeometricSolversMetalExt)
        m = LUFactorization()

        # 0x0 empty system
        x0 = MtlArray(zeros(Float32, 0))
        A0 = MtlArray(zeros(Float32, 0, 0))
        b0 = MtlArray(zeros(Float32, 0))
        prob0 = NonlinearProblem(x0, metal_vendor_noop)
        fac0 = dense_factor(m, prob0, x0, nothing)
        fac0 = factorize!!(fac0, m, A0)
        @test fac0.info == 0
        @test returncode(fac0) === SUCCESS
        d0 = ldiv!!(similar(x0), fac0, b0)
        @test length(d0) == 0

        # Singular matrices: equal rows (MPS status -2), zero matrix, zero column 2
        x4 = MtlArray(zeros(Float32, 4))
        prob4 = NonlinearProblem(x4, metal_vendor_noop)
        A_sing = Float32[1 2 3 4; 1 2 3 4; 0 1 1 1; 0 0 1 2]
        fac_sing = factorize!!(dense_factor(m, prob4, x4, nothing), m, MtlArray(A_sing))
        @test fac_sing.info > 0
        @test returncode(fac_sing) === SINGULAR

        fac_z = factorize!!(
            dense_factor(m, prob4, x4, nothing), m, MtlArray(zeros(Float32, 4, 4)))
        @test fac_z.info > 0
        @test returncode(fac_z) === SINGULAR

        A_c2 = Matrix{Float32}(I, 4, 4)
        A_c2[:, 2] .= 0.0f0
        fac_c2 = factorize!!(dense_factor(m, prob4, x4, nothing), m, MtlArray(A_c2))
        @test fac_c2.info > 0
        @test returncode(fac_c2) === SINGULAR

        # Direct status mapping checks on _mps_lu_finish!: status 0, -2, >0, and <0 fallback
        x2 = MtlArray(zeros(Float32, 2))
        prob2 = NonlinearProblem(x2, metal_vendor_noop)
        if ext_mtl !== nothing && isdefined(ext_mtl, :_mps_lu_finish!)
            A_ok = Float32[4 1; 1 3]
            dA_ok = MtlArray(A_ok)
            dx_ok = MtlArray(zeros(Float32, 2))
            fac_st = dense_factor(m, prob2, dx_ok, nothing)
            F_ok = LinearAlgebra.lu!(copy(dA_ok); check = false)
            copyto!(fac_st.factors, F_ok.factors)

            fac_mps = dense_factor(m, prob2, dx_ok, nothing)
            fill!(fac_mps.status, Int32(7))
            fac_mps = factorize!!(fac_mps, m, dA_ok)
            @test fac_mps.info == 0
            @test Array(fac_mps.status)[1] == Int32(7)
            @test_throws DimensionMismatch backsolve!(fac_mps, MtlArray(Float32[1.0f0]))

            fac_st0 = ext_mtl._mps_lu_finish!(fac_st, dA_ok, F_ok.ipiv, 0)
            @test fac_st0.info == 0
            @test returncode(fac_st0) === SUCCESS

            fac_stm2 = ext_mtl._mps_lu_finish!(fac_st, dA_ok, F_ok.ipiv, -2)
            @test fac_stm2.info == 1
            @test returncode(fac_stm2) === SINGULAR

            fac_stpos = ext_mtl._mps_lu_finish!(fac_st, dA_ok, F_ok.ipiv, 2)
            @test fac_stpos.info == 2
            @test returncode(fac_stpos) === SINGULAR

            fac_fb = dense_factor(m, prob2, dx_ok, nothing)
            fac_fb = ext_mtl._mps_lu_finish!(fac_fb, dA_ok, fac_fb.ipiv, -1)
            @test fac_fb.info == 0
            @test returncode(fac_fb) === SUCCESS
            ldiv!!(dx_ok, fac_fb, MtlArray(Float32[5, 4]))
            @test isapprox(Array(dx_ok), Float32[1, 1]; rtol = 1.0f-5)

            fac_fb_sing = dense_factor(m, prob2, dx_ok, nothing)
            fac_fb_sing = ext_mtl._mps_lu_finish!(
                fac_fb_sing, MtlArray(zeros(Float32, 2, 2)), fac_fb_sing.ipiv, -3)
            @test fac_fb_sing.info == 1
            @test returncode(fac_fb_sing) === SINGULAR
        else
            @test ext_mtl !== nothing
        end
    end

    BLAS.set_num_threads(old_blas_threads)
elseif Metal.functional()
    @test_skip on_virtual_gpu()  # issue #34
else
    @test_skip Metal.functional()
end
