using GPUArraysCore: AbstractGPUArray, allowscalar
using JLArrays: JLArray, JLMatrix, JLVector
using LinearAlgebra: BlasFloat, I
using Random: Xoshiro
using Test

using GeometricSolvers
using GeometricSolvers: DenseLU, DenseQR, DenseSVD, LUFactorization, NonlinearProblem,
                        QRFactorization, SVDFactorization
using GeometricSolvers: JACOBI_MAX_SWEEPS, backsolve!, factor_eltype, factorize!!,
                        dense_factor,
                        jacobi_pair, jacobi_rotation, jacobi_tolerance, ldiv!!, returncode

include("../helpers/matrix.jl")

const PRECISION_PAIRS = (map(T -> (T, Nothing), ELTYPES)...,
    map(T -> (T, Float32), filter(T -> real(T) === Float64, ELTYPES))...)

allowscalar(false)

function ka_well_conditioned(T::Type, n::Int, seed::Integer = 0x5eed)
    rng = Xoshiro(seed)
    A = randn(rng, T, n, n)
    for i in 1:n
        A[i, i] += T(2 * n)
    end
    b = randn(rng, T, n)
    return A, b
end

function ka_orthonormal_cols(rng, ::Type{T}, n::Int, r::Int) where {T}
    Q = randn(rng, T, n, r)
    for j in 1:r
        for k in 1:(j - 1)
            dot_kj = zero(T)
            for i in 1:n
                dot_kj += conj(Q[i, k]) * Q[i, j]
            end
            for i in 1:n
                Q[i, j] -= dot_kj * Q[i, k]
            end
        end
        nrm2 = zero(real(T))
        for i in 1:n
            nrm2 += abs2(Q[i, j])
        end
        nrm = sqrt(nrm2)
        for i in 1:n
            Q[i, j] /= nrm
        end
    end
    return Q
end

function ka_rank_deficient(seed::Integer, ::Type{T}, n::Int, r::Int) where {T}
    A = zeros(T, n, n)
    r == 0 && return A
    rng = Xoshiro(seed)
    U = ka_orthonormal_cols(rng, T, n, r)
    V = ka_orthonormal_cols(rng, T, n, r)
    for k in 1:r
        σk = T(real(T)(0.5)^(k - 1))
        for j in 1:n
            vjk = σk * conj(V[j, k])
            for i in 1:n
                A[i, j] += U[i, k] * vjk
            end
        end
    end
    return A
end

function effective_factor_eltype(::Type{T}, ::Type{TF}) where {T, TF}
    TF === Nothing && return T
    F = promote_type(Float32, TF)
    return T <: Complex ? Complex{F} : F
end

ka_wide_eltype(::Type{T}) where {T <: Real} = Float64
ka_wide_eltype(::Type{T}) where {T <: Complex} = ComplexF64
ka_integers(::Type{T}, n::Int) where {T} = T[i + j for i in 1:n, j in 1:n]
ka_noop(r, y, p) = nothing

function ka_opnorm_inf(@nospecialize(A::AbstractMatrix))
    n = size(A, 1)
    m = size(A, 2)
    rmax = 0.0
    for i in 1:n
        s = 0.0
        for j in 1:m
            s += Float64(abs(A[i, j]))
        end
        rmax = max(rmax, s)
    end
    return rmax
end

function ka_norm_inf(@nospecialize(v::AbstractVector))
    rmax = 0.0
    for i in eachindex(v)
        rmax = max(rmax, Float64(abs(v[i])))
    end
    return rmax
end

function ka_norm_2(@nospecialize(v::AbstractVector))
    s = 0.0
    for i in eachindex(v)
        s += Float64(abs2(v[i]))
    end
    return sqrt(s)
end

function ka_rel_inf(@nospecialize(x::AbstractVector), @nospecialize(y::AbstractVector))
    num = 0.0
    for i in eachindex(x, y)
        num = max(num, abs(ComplexF64(x[i]) - ComplexF64(y[i])))
    end
    return num / ka_norm_inf(y)
end

function ka_res_norms(
        @nospecialize(A::AbstractMatrix), @nospecialize(x::AbstractVector),
        @nospecialize(b::AbstractVector))
    res_inf = 0.0
    res_2_sq = 0.0
    for i in axes(A, 1)
        s = zero(ComplexF64)
        for j in axes(A, 2)
            s += ComplexF64(A[i, j]) * ComplexF64(x[j])
        end
        ri = abs(s - ComplexF64(b[i]))
        res_inf = max(res_inf, ri)
        res_2_sq += ri * ri
    end
    return res_inf, sqrt(res_2_sq)
end

function ka_svd_orth_and_recon_errs(
        @nospecialize(U::AbstractMatrix), @nospecialize(S::AbstractVector),
        @nospecialize(VT::AbstractMatrix), @nospecialize(A::AbstractMatrix))
    n = size(A, 1)
    err_u = 0.0
    err_vt = 0.0
    err_rec = 0.0
    for i in 1:n
        ru = 0.0
        rvt = 0.0
        rrec = 0.0
        for j in 1:n
            su = zero(ComplexF64)
            svt = zero(ComplexF64)
            srec = zero(ComplexF64)
            for k in 1:n
                su += conj(ComplexF64(U[k, i])) * ComplexF64(U[k, j])
                svt += ComplexF64(VT[i, k]) * conj(ComplexF64(VT[j, k]))
                srec += ComplexF64(U[i, k]) * Float64(S[k]) * ComplexF64(VT[k, j])
            end
            ru += abs(su - (i == j ? 1.0 : 0.0))
            rvt += abs(svt - (i == j ? 1.0 : 0.0))
            rrec += abs(srec - ComplexF64(A[i, j]))
        end
        err_u = max(err_u, ru)
        err_vt = max(err_vt, rvt)
        err_rec = max(err_rec, rrec)
    end
    return err_u, err_vt, err_rec
end

function ka_max_abs_diff(@nospecialize(x::AbstractVector), @nospecialize(y::AbstractVector))
    dmax = 0.0
    for i in eachindex(x, y)
        dmax = max(dmax, abs(ComplexF64(x[i]) - ComplexF64(y[i])))
    end
    return dmax
end

function ka_count_nz(@nospecialize(A::AbstractArray))
    c = 0
    for i in eachindex(A)
        iszero(A[i]) || (c += 1)
    end
    return c
end

function _ka_gauss_solve_and_cond(Ac::Matrix{ComplexF64}, bc::Vector{ComplexF64})
    n = size(Ac, 1)
    anorm = ka_opnorm_inf(Ac)
    aug = zeros(ComplexF64, n, 2 * n + 1)
    for j in 1:n, i in 1:n

        aug[i, j] = Ac[i, j]
    end
    for i in 1:n
        aug[i, n + i] = 1.0 + 0.0im
        aug[i, 2 * n + 1] = bc[i]
    end
    for k in 1:n
        p = k
        amax = abs(aug[k, k])
        for i in (k + 1):n
            ai = abs(aug[i, k])
            if ai > amax
                amax = ai
                p = i
            end
        end
        if p != k
            for j in k:(2 * n + 1)
                aug[k, j], aug[p, j] = aug[p, j], aug[k, j]
            end
        end
        piv_inv = inv(aug[k, k])
        for j in k:(2 * n + 1)
            aug[k, j] *= piv_inv
        end
        for i in 1:n
            i == k && continue
            f = aug[i, k]
            iszero(f) && continue
            for j in k:(2 * n + 1)
                aug[i, j] -= f * aug[k, j]
            end
        end
    end
    x_ref = Vector{ComplexF64}(undef, n)
    Ainv = Matrix{ComplexF64}(undef, n, n)
    for i in 1:n
        x_ref[i] = aug[i, 2 * n + 1]
        for j in 1:n
            Ainv[i, j] = aug[i, n + j]
        end
    end
    return x_ref, anorm * ka_opnorm_inf(Ainv), anorm
end

function _ka_svd_ref_f64(Ar::Matrix{Float64})
    x0 = zeros(Float64, size(Ar, 1))
    prob = NonlinearProblem(x0, ka_noop)
    fac = dense_factor(SVDFactorization(), prob, x0, nothing)
    copyto!(fac.factors, Ar)
    fac = factorize!!(fac, SVDFactorization(), fac.factors)
    return fac.S
end

function ka_ref_solve_and_cond(
        @nospecialize(A::AbstractMatrix), @nospecialize(b::AbstractVector))
    n = size(A, 1)
    Ac = Matrix{ComplexF64}(undef, n, n)
    bc = Vector{ComplexF64}(undef, n)
    for j in 1:n
        bc[j] = ComplexF64(b[j])
        for i in 1:n
            Ac[i, j] = ComplexF64(A[i, j])
        end
    end
    return _ka_gauss_solve_and_cond(Ac, bc)
end

function ka_ref_solve_cond_svd(
        @nospecialize(A::AbstractMatrix), @nospecialize(b::AbstractVector))
    n = size(A, 1)
    x_ref, κ, anorm = ka_ref_solve_and_cond(A, b)
    S_ref = if eltype(A) <: Real
        Ar = Matrix{Float64}(undef, n, n)
        for j in 1:n, i in 1:n

            Ar[i, j] = Float64(A[i, j])
        end
        _ka_svd_ref_f64(Ar)
    else
        Ar2 = Matrix{Float64}(undef, 2 * n, 2 * n)
        for j in 1:n, i in 1:n

            z = ComplexF64(A[i, j])
            Ar2[i, j] = real(z)
            Ar2[i, n + j] = -imag(z)
            Ar2[n + i, j] = imag(z)
            Ar2[n + i, n + j] = real(z)
        end
        S2 = _ka_svd_ref_f64(Ar2)
        Float64[S2[2 * k - 1] for k in 1:n]
    end
    return x_ref, κ, anorm, S_ref
end

function ka_check_inferred_and_scaled(m, prob_dev, x_dev, A_dev, b_dev)
    fac = @inferred dense_factor(m, prob_dev, x_dev, nothing)
    fac = @inferred factorize!!(fac, m, A_dev)
    @test fac.info == 0
    d = @inferred ldiv!!(similar(x_dev), fac, b_dev)
    d_h = Array(d)
    @test all(isfinite, d_h)
    e_sc = fac.scale_exponent + 3
    fac_sc = fac isa DenseLU ?
             DenseLU(fac.factors, fac.ipiv, fac.rhs, fac.info, e_sc, fac.status) :
             fac isa DenseQR ?
             DenseQR(fac.factors, fac.τ, fac.work, fac.rhs, fac.info, e_sc) :
             DenseSVD(fac.factors, fac.U, fac.S, fac.VT, fac.work, fac.rwork,
        fac.tmp, fac.rhs, fac.info, fac.rank, e_sc)
    d_sc = ldiv!!(similar(x_dev), fac_sc, b_dev)
    @test ka_rel_inf(Array(d_sc), Float64[x / 8.0 for x in d_h]) <= 8 * eps(Float64)
end

@testset "base premise extension to JLArray" begin
    A_cpu = [2.0 1.0; 1.0 3.0]
    b_cpu = [5.0, 7.0]
    x_ref, _, _ = ka_ref_solve_and_cond(A_cpu, b_cpu)

    x_h0 = zeros(Float64, 2)
    prob_h = NonlinearProblem(x_h0, ka_noop)
    fac_h = dense_factor(SVDFactorization(), prob_h, x_h0, nothing)
    copyto!(fac_h.factors, A_cpu)
    fac_h = factorize!!(fac_h, SVDFactorization(), fac_h.factors)
    @test fac_h.info == 0
    @test returncode(fac_h) === SUCCESS
    d_h = ldiv!!(similar(x_h0), fac_h, copy(b_cpu))
    @test ka_rel_inf(d_h, x_ref) <= 64 * eps(Float64)
    fill!(fac_h.factors, 0.0)
    fac_hz = factorize!!(fac_h, SVDFactorization(), fac_h.factors)
    @test fac_hz.info == 0
    @test returncode(fac_hz) === SUCCESS
    @test all(iszero, ldiv!!(similar(x_h0), fac_hz, copy(b_cpu)))

    for m in (LUFactorization(), QRFactorization(), SVDFactorization())
        x = JLArray(zeros(Float64, 2))
        b = JLArray(copy(b_cpu))
        prob = NonlinearProblem(x, ka_noop)
        fac = dense_factor(m, prob, x, nothing)
        copyto!(fac.factors, JLArray(A_cpu))
        fac = factorize!!(fac, m, fac.factors)
        @test fac.info == 0
        @test returncode(fac) === SUCCESS
        d = ldiv!!(similar(x), fac, b)
        @test d isa JLVector{Float64}
        @test ka_rel_inf(Array(d), x_ref) <= 64 * eps(Float64)

        fac_z = dense_factor(m, prob, x, nothing)
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
    @test abs(c_rot - inv(sqrt(2.0))) <= 4 * eps(Float64)
    @test abs(s_rot - inv(sqrt(2.0))) <= 4 * eps(Float64)
    @test minmax(jacobi_pair(4, 0, 0)...) == (1, 4)
end

@testset "KA device LU, QR, and SVD agree with CPU solves on JLArray" begin
    @testset "n = $n" for n in (1, 16)
        for (T, TF) in PRECISION_PAIRS
            A_cpu, b_cpu = ka_well_conditioned(T, n)
            x_ref, κ, anorm_inf, S_ref = ka_ref_solve_cond_svd(A_cpu, b_cpu)
            @test κ <= 10.0

            for M in (LUFactorization, QRFactorization, SVDFactorization)
                m = TF === Nothing ? M() : M(TF)
                TF_eff = effective_factor_eltype(T, TF)
                eps_eff = Float64(eps(real(TF_eff)))

                # Device solve on JLArray (in-place J === fac.factors when TF === Nothing)
                x_dev = JLArray(zeros(T, n))
                b_dev = JLArray(copy(b_cpu))
                prob_dev = NonlinearProblem(x_dev, ka_noop)
                fac_dev = dense_factor(m, prob_dev, x_dev, nothing)
                if TF === Nothing
                    copyto!(fac_dev.factors, JLArray(A_cpu))
                    fac_dev = factorize!!(fac_dev, m, fac_dev.factors)
                else
                    fac_dev = factorize!!(fac_dev, m, JLArray(A_cpu))
                end
                d_dev = ldiv!!(similar(x_dev), fac_dev, b_dev)
                @test fac_dev.info == 0
                @test returncode(fac_dev) === SUCCESS
                @test d_dev isa JLVector{T}

                x_h = Array(d_dev)
                res_inf, res_2 = ka_res_norms(A_cpu, x_h, b_cpu)

                if fac_dev isa DenseSVD
                    @test length(fac_dev.work) == 1
                    @test eltype(fac_dev.work) === Int32
                    @test length(fac_dev.rwork) == n
                    @test eltype(fac_dev.rwork) === real(TF_eff)
                    S_h = Array(fac_dev.S)
                    @test issorted(S_h; rev = true)
                    @test issorted(Array(fac_dev.rwork); rev = true)
                    σ1 = S_ref[1]
                    @test ka_max_abs_diff(S_h, S_ref) <= 8 * n * eps_eff * σ1
                    bwd2 = res_2 / (σ1 * ka_norm_2(x_h))
                    @test bwd2 <= 4 * n * eps_eff
                end

                if TF === Nothing
                    bwd = res_inf / (anorm_inf * ka_norm_inf(x_h))
                    @test bwd <= 4 * n * eps_eff
                end

                @test ka_rel_inf(x_h, x_ref) <= 8 * n * eps_eff * κ

                # Out-of-place J !== fac_dev.factors also copies and factors cleanly
                if TF === Nothing && n == 1
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
        n = 8
        for T in f64_types
            A0, b0 = ka_well_conditioned(T, n)
            x0_ref, κ, _ = ka_ref_solve_and_cond(A0, b0)
            for M in (LUFactorization, QRFactorization, SVDFactorization)
                for (sA, sb) in scales
                    m = M(Float32)
                    A_cpu = T[T(sA) * A0[i, j] for i in 1:n, j in 1:n]
                    b_cpu = T[T(sb) * b0[i] for i in 1:n]
                    x_ref = [(sb / sA) * x0_ref[i] for i in 1:n]

                    x_dev = JLArray(zeros(T, n))
                    b_dev = JLArray(copy(b_cpu))
                    prob_dev = NonlinearProblem(x_dev, ka_noop)
                    fac_dev = dense_factor(m, prob_dev, x_dev, nothing)
                    fac_dev = factorize!!(fac_dev, m, JLArray(A_cpu))
                    @test fac_dev.info == 0
                    @test fac_dev.scale_exponent != 0

                    d_dev = ldiv!!(similar(x_dev), fac_dev, b_dev)
                    x_h = Array(d_dev)
                    @test all(isfinite, x_h)
                    @test ka_rel_inf(x_h, x_ref) <= 8 * n * Float64(eps(Float32)) * κ
                end
            end
        end
    end
end

@testset "disambiguation and scalar-indexing absence on JLArray" begin
    @testset "dispatch to src/linear/ka/ for T = $T" for T in ELTYPES
        x = JLArray(zeros(T, 4))
        prob = NonlinearProblem(x, ka_noop)

        fac_lu = dense_factor(LUFactorization(), prob, x, nothing)
        lu_sig = (typeof(fac_lu), LUFactorization{Nothing}, JLMatrix{T})
        m_lu_fac = which(factorize!!, lu_sig)
        m_lu_bs = which(backsolve!, (typeof(fac_lu), JLVector{T}))
        @test endswith(replace(String(m_lu_fac.file), '\\' => '/'), "src/linear/ka/lu.jl")
        @test endswith(replace(String(m_lu_bs.file), '\\' => '/'), "src/linear/ka/lu.jl")

        fac_qr = dense_factor(QRFactorization(), prob, x, nothing)
        qr_sig = (typeof(fac_qr), QRFactorization{Nothing}, JLMatrix{T})
        m_qr_fac = which(factorize!!, qr_sig)
        m_qr_bs = which(backsolve!, (typeof(fac_qr), JLVector{T}))
        @test endswith(replace(String(m_qr_fac.file), '\\' => '/'), "src/linear/ka/qr.jl")
        @test endswith(replace(String(m_qr_bs.file), '\\' => '/'), "src/linear/ka/qr.jl")

        fac_svd = dense_factor(SVDFactorization(), prob, x, nothing)
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

    @testset "non-BLAS element type Float16 solves through src/linear/ka/{lu,qr}.jl" begin
        T = Float16
        n = 4
        A_cpu, b_cpu = ka_well_conditioned(T, n)
        x_ref, κ, _ = ka_ref_solve_and_cond(A_cpu, b_cpu)
        for (M, file) in ((LUFactorization, "src/linear/ka/lu.jl"),
            (QRFactorization, "src/linear/ka/qr.jl"))
            m = M()
            x = JLArray(zeros(T, n))
            prob = NonlinearProblem(x, ka_noop)
            fac = factorize!!(dense_factor(m, prob, x, nothing), m, JLArray(A_cpu))
            @test fac.info == 0
            m_bs = which(backsolve!, (typeof(fac), JLVector{T}))
            @test endswith(replace(String(m_bs.file), '\\' => '/'), file)
            d = ldiv!!(similar(x), fac, JLArray(b_cpu))
            @test d isa JLVector{T}
            @test ka_rel_inf(Array(d), x_ref) <= 8 * n * Float64(eps(T)) * κ
        end
    end

    # JET.@test_opt on dense_factor, factorize!!, and ldiv!! is in test/linear/ka_large.jl (slow group)
    @testset "no scalar indexing and type inference in dense_factor, factorize!!, ldiv!!" begin
        T = Float64
        A_cpu, b_cpu = ka_well_conditioned(T, 6)
        x_dev = JLArray(zeros(T, 6))
        A_dev = JLArray(A_cpu)
        b_dev = JLArray(copy(b_cpu))
        prob_dev = NonlinearProblem(x_dev, ka_noop)
        ka_check_inferred_and_scaled(LUFactorization(), prob_dev, x_dev, A_dev, b_dev)
        ka_check_inferred_and_scaled(QRFactorization(), prob_dev, x_dev, A_dev, b_dev)
        ka_check_inferred_and_scaled(SVDFactorization(), prob_dev, x_dev, A_dev, b_dev)
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
        x_ref, κ, _ = ka_ref_solve_and_cond(A_cpu, b_cpu)

        for m in (LUFactorization(), QRFactorization())
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = dense_factor(m, prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_cpu))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            d = ldiv!!(similar(x_dev), fac, JLArray(b_cpu))
            @test ka_rel_inf(Array(d), x_ref) <= 8 * n * Float64(eps(RT)) * κ
        end
    end

    @testset "purely imaginary and imaginary-dominant pivots via abs1" begin
        for T in filter(T -> T <: Complex, ELTYPES)
            RT = real(T)
            n = 4
            A_real, b_real = ka_well_conditioned(RT, n)
            A_real[1, 1] = zero(RT)
            A_real[2, 1] = RT(4)
            A_imag = T[T(zero(RT), A_real[i, j]) for i in 1:n, j in 1:n]
            b_imag = T[T(b_real[i], b_real[n - i + 1]) for i in 1:n]

            # Second matrix where |Re(A[1,1])| > |Re(A[2,1])| == 0 but |A[2,1]| == 4 >> |A[1,1]|
            A_dom = copy(A_imag)
            A_dom[1, 1] = T(eps(RT), zero(RT))
            A_dom[2, 1] = T(zero(RT), RT(4))

            for A_cpu in (A_imag, A_dom)
                x_ref, κ, _ = ka_ref_solve_and_cond(A_cpu, b_imag)
                for m in (LUFactorization(), QRFactorization())
                    x_dev = JLArray(zeros(T, n))
                    prob = NonlinearProblem(x_dev, ka_noop)
                    fac = dense_factor(m, prob, x_dev, nothing)
                    copyto!(fac.factors, JLArray(A_cpu))
                    fac = factorize!!(fac, m, fac.factors)
                    @test fac.info == 0
                    d = ldiv!!(similar(x_dev), fac, JLArray(b_imag))
                    @test ka_rel_inf(Array(d), x_ref) <= 8 * n * Float64(eps(RT)) * κ
                end
            end
        end
    end

    @testset "overflow-safe column scaling in QR and multiplier in LU, T = $T" for T in ELTYPES
        RT = real(T)
        W = ka_wide_eltype(T)
        n = 4
        A0, b0 = ka_well_conditioned(T, n)
        big_col = RT === Float32 ? RT(1.0f20) : RT(1e200)
        A_col = copy(A0)
        for i in 1:n
            A_col[i, 1] *= T(big_col)
        end
        b_col = copy(b0)
        x0_ref, κ, _ = ka_ref_solve_and_cond(A0, b_col)
        x_ref_1 = x0_ref[1] / Float64(big_col)
        x_ref_rest = x0_ref[2:end]

        for m in (LUFactorization(), QRFactorization())
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = dense_factor(m, prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_col))
            fac = factorize!!(fac, m, fac.factors)
            @test fac.info == 0
            @test all(isfinite, Array(fac.factors))
            if m isa QRFactorization
                @test all(isfinite, Array(fac.τ))
            end
            d = ldiv!!(similar(x_dev), fac, JLArray(b_col))
            x_h = Array(d)
            @test all(isfinite, x_h)
            err_col1 = abs(W(x_h[1]) - x_ref_1) / abs(x_ref_1)
            err_rest = ka_rel_inf(x_h[2:end], x_ref_rest)
            @test err_col1 <= 16 * n * Float64(eps(RT)) * κ
            @test err_rest <= 16 * n * Float64(eps(RT)) * κ
        end

        # Near-floatmax (0.6 * floatmax(RT)) reflector, trailing apply, and solve without intermediate overflow
        s_hi = RT(0.6) * floatmax(RT)
        for (A_hi, x_hi) in (
            (T[T(s_hi) zero(T); zero(T) T(s_hi)], T[one(T), one(T)]),
            (T[T(s_hi) T(s_hi); zero(T) T(s_hi)], T[one(T), zero(T)]),
            (T[T(s_hi) zero(T); T(s_hi) T(s_hi)], T[one(T), zero(T)]),
            (T[T(s_hi) zero(T); T(s_hi) T(s_hi)], T[T(RT(0.5)), T(RT(0.5))])
        )
            b_hi = T[A_hi[1, 1] * x_hi[1] + A_hi[1, 2] * x_hi[2],
                A_hi[2, 1] * x_hi[1] + A_hi[2, 2] * x_hi[2]]
            for m in (LUFactorization(), QRFactorization())
                x2 = JLArray(zeros(T, 2))
                prob2 = NonlinearProblem(x2, ka_noop)
                fac_hi = dense_factor(m, prob2, x2, nothing)
                fac_hi = factorize!!(fac_hi, m, JLArray(A_hi))
                @test returncode(fac_hi) === SUCCESS
                @test all(isfinite, Array(fac_hi.factors))
                if m isa QRFactorization
                    @test all(isfinite, Array(fac_hi.τ))
                end
                d_hi = ldiv!!(similar(x2), fac_hi, JLArray(b_hi))
                @test all(isfinite, Array(d_hi))
                @test ka_rel_inf(Array(d_hi), x_hi) <= sqrt(Float64(eps(RT)))
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
            A_unit = W[W(1, 1) W(0.5, -0.25); W(0.5, 0.5) W(1, 0.5)]
            b_unit = W[W(1, 0), W(0, 1)]
            x_ref, _, _ = ka_ref_solve_and_cond(A_unit, b_unit)
            x_dev = JLArray(zeros(T, 2))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = dense_factor(LUFactorization(), prob, x_dev, nothing)
            copyto!(fac.factors, JLArray(A_big))
            fac = factorize!!(fac, LUFactorization(), fac.factors)
            @test fac.info == 0
            @test all(isfinite, Array(fac.factors))
            d = copy(JLArray(b_big))
            backsolve!(fac, d)
            @test all(isfinite, Array(d))
            @test ka_rel_inf(Array(d), x_ref) <= 32 * Float64(eps(RT))
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
        fac_zero = dense_factor(m, prob, x_dev, nothing)
        fill!(fac_zero.factors, zero(T))
        fac_zero = factorize!!(fac_zero, m, fac_zero.factors)
        @test fac_zero.info == 1
        @test returncode(fac_zero) === SINGULAR
        @test all(isfinite, Array(fac_zero.factors))

        # (b) Exact zero column k = 2 (and k = 3 also zero so first zero is 2, not 3)
        A_col2 = Matrix{T}(I, n, n)
        A_col2[:, 2] .= zero(T)
        A_col2[:, 3] .= zero(T)
        fac_col2 = dense_factor(m, prob, x_dev, nothing)
        copyto!(fac_col2.factors, JLArray(A_col2))
        fac_col2 = factorize!!(fac_col2, m, fac_col2.factors)
        @test fac_col2.info == 2
        @test all(isfinite, Array(fac_col2.factors))

        # (b2) Zero column k = 2 followed by nonzero columns 3 and 4 -> info stays 2
        A_col2_nz = Matrix{T}(I, n, n)
        A_col2_nz[:, 2] .= zero(T)
        fac_col2_nz = dense_factor(m, prob, x_dev, nothing)
        copyto!(fac_col2_nz.factors, JLArray(A_col2_nz))
        fac_col2_nz = factorize!!(fac_col2_nz, m, fac_col2_nz.factors)
        @test fac_col2_nz.info == 2
        @test all(isfinite, Array(fac_col2_nz.factors))

        # (c) 2 x 2 matrix with two equal rows (scaled and random draws) -> info == 2 for LU
        x2 = JLArray(zeros(T, 2))
        prob2 = NonlinearProblem(x2, ka_noop)
        for scale in (one(RT), floatmax(RT) / RT(8))
            A_eq = T[T(scale) T(2 * scale); T(scale) T(2 * scale)]
            fac_eq = dense_factor(m, prob2, x2, nothing)
            copyto!(fac_eq.factors, JLArray(A_eq))
            fac_eq = factorize!!(fac_eq, m, fac_eq.factors)
            if M === LUFactorization
                @test fac_eq.info == 2
                @test returncode(fac_eq) === SINGULAR
            end
            @test all(isfinite, Array(fac_eq.factors))
        end
        rng_eq = Xoshiro(42)
        for _ in 1:3
            r_eq = randn(rng_eq, T, 2)
            A_eq = T[r_eq[1] r_eq[2]; r_eq[1] r_eq[2]]
            fac_eq = dense_factor(m, prob2, x2, nothing)
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
            fac_bad = dense_factor(m, prob, x_dev, nothing)
            copyto!(fac_bad.factors, JLArray(A_bad))
            fac_bad = factorize!!(fac_bad, m, fac_bad.factors)
            d_bad = ldiv!!(similar(x_dev), fac_bad, JLArray(ones(T, n)))
            @test d_bad isa JLVector{T}
        end

        # (e) 0 x 0 empty system does not launch a 0-range kernel or throw
        x0 = JLArray(zeros(T, 0))
        prob0 = NonlinearProblem(x0, ka_noop)
        fac0 = dense_factor(m, prob0, x0, nothing)
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
        fac1 = dense_factor(m, prob1, x1, nothing)
        copyto!(fac1.factors, JLArray(A_cpu))
        fac1 = factorize!!(fac1, m, fac1.factors)
        d1 = ldiv!!(similar(x1), fac1, JLArray(b_cpu))

        x2 = JLArray(zeros(T, n))
        prob2 = NonlinearProblem(x2, ka_noop)
        fac2 = dense_factor(m, prob2, x2, nothing)
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

        err_u, err_vt, err_rec = ka_svd_orth_and_recon_errs(
            Array(fac1.U), Array(fac1.S), Array(fac1.VT), A_cpu)
        @test err_u <= 8 * n * eps(RT)
        @test err_vt <= 8 * n * eps(RT)
        @test err_rec <= 8 * n * eps(RT) * ka_opnorm_inf(A_cpu)
    end

    @testset "equal-column-norm rotation (alpha == beta) on JLArray, T = $T" for T in ELTYPES
        RT = real(T)
        m = SVDFactorization()
        for A_eqnorm in (
            T[2 1; 1 2],
            T[2 -1; -1 2]
        )
            n = size(A_eqnorm, 1)
            b_eq = T[T(i) for i in 1:n]
            x_ref, _, _ = ka_ref_solve_and_cond(A_eqnorm, b_eq)
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = dense_factor(m, prob, x_dev, nothing)
            fac = factorize!!(fac, m, JLArray(A_eqnorm))
            @test fac.info == 0
            err_u, _, _ = ka_svd_orth_and_recon_errs(
                Array(fac.U), Array(fac.S), Array(fac.VT), A_eqnorm)
            @test err_u <= 8 * n * eps(RT)
            d = ldiv!!(similar(x_dev), fac, JLArray(b_eq))
            @test ka_rel_inf(Array(d), x_ref) <= 8 * n * Float64(eps(RT))
        end
    end
end

@testset "Check 4: SVD subnormal prescaling and rank-deficient termination traps on JLArray" begin
    @testset "4(a) subnormal ldexp prescaling matches integer matrix bit for bit" begin
        for (T, TF) in PRECISION_PAIRS
            n = 4
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            E = exponent(floatmin(R)) - 8
            B_cpu = ka_integers(T, n)
            A_cpu = T[T(ldexp(one(R), E)) * B_cpu[i, j] for i in 1:n, j in 1:n]
            @test ka_opnorm_inf(A_cpu) / n < Float64(floatmin(R))

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_cpu))
            fac_ref = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(B_cpu))

            @test fac.info == 0
            @test all(isfinite, Array(fac.U))
            @test all(isfinite, Array(fac.VT))
            @test all(isfinite, Array(fac.S))
            @test Array(fac.U) == Array(fac_ref.U)
            @test Array(fac.VT) == Array(fac_ref.VT)
            S_ref = Array(fac_ref.S)
            @test Array(fac.S) == R[ldexp(S_ref[i], E) for i in 1:n]
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
            A_cpu = T[i == j ? T(s_half + s_eighth) : T(s_half) for i in 1:n, j in 1:n]
            b_cpu = fill(T(s_quarter), n)
            # Closed-form solution for (s_half * ones(n, n) + s_eighth * I) * x = s_quarter * ones(n)
            x_ref = fill(2.0 / (4 * n + 1), n)
            κ = Float64(4 * n + 1)

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_cpu))
            @test fac.info == 0
            @test returncode(fac) === SUCCESS
            @test !isfinite(Array(fac.S)[1])
            @test all(isfinite, Array(fac.rwork))

            d = ldiv!!(similar(x_dev), fac, JLArray(b_cpu))
            x_h = Array(d)
            @test all(isfinite, x_h)
            @test ka_rel_inf(x_h, x_ref) <= 8 * n * Float64(eps(R)) * κ
        end
    end

    @testset "4(c) rank-deficient matrices terminate with exact rank and minimum-norm step" begin
        for (T, TF) in PRECISION_PAIRS, r in (5, 1, 0)

            n = 16
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)
            tol = jacobi_tolerance(n, R)
            A_cpu = ka_rank_deficient(100r + n, T, n, r)

            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            fac = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_cpu))

            @test fac.info == 0
            @test returncode(fac) === SUCCESS
            S_h = Array(fac.S)
            @test issorted(S_h; rev = true)
            cutoff = tol * S_h[1]
            num_rank = 0
            if S_h[1] > zero(R)
                for s in S_h
                    s > cutoff && (num_rank += 1)
                end
            end
            @test num_rank == r
            @test fac.rank == r
            @test S_h[r + 1] <= cutoff
            if r >= 1
                @test S_h[r] > cutoff
            end

            U_h = Array(fac.U)
            @test all(isfinite, U_h)
            @test ka_count_nz(@view(U_h[:, (r + 1):n])) == 0

            x0_cpu = randn(Xoshiro(400r + n), T, n)
            b_cpu = zeros(T, n)
            for j in 1:n, i in 1:n

                b_cpu[i] += A_cpu[i, j] * x0_cpu[j]
            end
            d = ldiv!!(similar(x_dev), fac, JLArray(b_cpu))
            x_h = Array(d)
            @test all(isfinite, x_h)
            if r == 0
                @test all(iszero, x_h)
                b_arb = randn(Xoshiro(999 + n), T, n)
                d_arb = ldiv!!(similar(x_dev), fac, JLArray(b_arb))
                @test all(iszero, Array(d_arb))
            else
                res_inf, res_2 = ka_res_norms(A_cpu, x_h, b_cpu)
                @test res_inf <= 4 * n * eps(R) * ka_opnorm_inf(A_cpu) * ka_norm_inf(x_h)
                @test res_2 <= 4 * n * eps(R) * S_h[1] * ka_norm_2(x_h)
                @test ka_norm_2(x_h) <= (1 + 4 * n * eps(R)) * ka_norm_2(x0_cpu)
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
            fac = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_sub))
            @test fac.info == 0
            @test Array(fac.rwork)[2] > zero(R)
            @test Array(fac.rwork)[2] <= tol * Array(fac.rwork)[1]
            @test ka_count_nz(@view(Array(fac.U)[:, 2:n])) == 0
            d = ldiv!!(similar(x_dev), fac, JLArray(b_sub))
            d_h = Array(d)
            @test abs(d_h[1] - one(T)) <= 4 * eps(R)
            @test all(iszero, d_h[2:n])

            # Direct backsolve! check when U[:, 2] is non-zero so ka_svd_ut_mul_scale_kernel!'s own
            # guard on fac.rank is exercised independently of U column zeroing
            if TF === Nothing
                copyto!(fac.U, JLArray(Matrix{F}(I, n, n)))
                copyto!(fac.VT, JLArray(Matrix{F}(I, n, n)))
                d_bs = JLArray(ones(F, n))
                backsolve!(fac, d_bs)
                d_bs_h = Array(d_bs)
                @test abs(d_bs_h[1] - one(F)) <= 4 * eps(R)
                @test all(iszero, d_bs_h[2:n])
            end
        end
    end

    @testset "4(d) negligible column below jacobi_tolerance(n, R) is skipped and zeroed in U" begin
        for (T, TF) in PRECISION_PAIRS
            m = TF === Nothing ? SVDFactorization() : SVDFactorization(TF)
            F = effective_factor_eltype(T, TF)
            R = real(F)

            # n = 4 with column norm 0.5 * eps(R) * σ₁
            n4 = 4
            ε4 = eps(R) / R(2)
            A4 = Matrix{T}(I, n4, n4)
            A4[n4, n4] = zero(T)
            A4[1, n4] = T(ε4)
            x4 = JLArray(zeros(T, n4))
            prob4 = NonlinearProblem(x4, ka_noop)
            fac4 = factorize!!(dense_factor(m, prob4, x4, nothing), m, JLArray(A4))
            @test fac4.info == 0
            @test Array(fac4.S)[n4] == ε4
            @test all(iszero, Array(fac4.U)[:, n4])
            @test ka_count_nz(Array(fac4.VT)) == n4

            # n = 16 with column norm 2 * eps(R) = 0.5 * jacobi_tolerance(16, R) (between eps(R) and tol)
            n16 = 16
            ε16 = R(2) * eps(R)
            A16 = Matrix{T}(I, n16, n16)
            A16[n16, n16] = zero(T)
            A16[1, n16] = T(ε16)
            x16 = JLArray(zeros(T, n16))
            prob16 = NonlinearProblem(x16, ka_noop)
            fac16 = factorize!!(dense_factor(m, prob16, x16, nothing), m, JLArray(A16))
            @test fac16.info == 0
            @test Array(fac16.S)[n16] == ε16
            @test all(iszero, Array(fac16.U)[:, n16])
            @test ka_count_nz(Array(fac16.VT)) == n16
        end

        for T in REAL_ELTYPES
            n = 16
            tol = jacobi_tolerance(n, T)
            ε = T(0.5) * tol
            A_noise = zeros(T, n, n)
            A_noise[1, 1] = one(T)
            for j in 2:n, i in 1:n

                A_noise[i, j] = (ε / sqrt(T(n))) * ((i + j) % 2 == 0 ? one(T) : -one(T))
            end
            x_dev = JLArray(zeros(T, n))
            prob = NonlinearProblem(x_dev, ka_noop)
            m = SVDFactorization()
            fac_noise = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_noise))
            @test fac_noise.info == 0
            @test ka_count_nz(@view(Array(fac_noise.U)[:, 2:n])) == 0
            @test ka_count_nz(Array(fac_noise.VT)) == n
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
                fac_nan = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_nan))
                @test fac_nan.info == 1
                @test returncode(fac_nan) === SINGULAR
                d_nan = ldiv!!(similar(x_dev), fac_nan, JLArray(ones(T, n)))
                @test d_nan isa JLVector{T}

                A_inf = Matrix{T}(I, n, n)
                A_inf[1, 1] = T(RT(Inf))
                fac_inf = factorize!!(dense_factor(m, prob, x_dev, nothing), m, JLArray(A_inf))
                @test fac_inf.info == 1
                @test returncode(fac_inf) === SINGULAR
                d_inf = ldiv!!(similar(x_dev), fac_inf, JLArray(ones(T, n)))
                @test d_inf isa JLVector{T}
            end

            x2 = JLArray(zeros(T, 2))
            prob2 = NonlinearProblem(x2, ka_noop)
            A_part = T[zero(T) T(RT(NaN)); zero(T) zero(T)]
            fac_part = factorize!!(dense_factor(m, prob2, x2, nothing), m, JLArray(A_part))
            @test fac_part.info == 1
            @test returncode(fac_part) === SINGULAR

            x0 = JLArray(zeros(T, 0))
            prob0 = NonlinearProblem(x0, ka_noop)
            fac0 = factorize!!(dense_factor(m, prob0, x0, nothing), m, JLArray(zeros(T, 0, 0)))
            @test fac0.info == 0
            d0 = ldiv!!(similar(x0), fac0, JLArray(zeros(T, 0)))
            @test length(d0) == 0
        end
    end
end

@testset "the same-type solve scales r by a power of two on JLArray, $T" for T in REAL_ELTYPES
    @testset "$label, $M" for (label, J, b, xref, with_qr) in extreme_rhs_systems(T),
        M in (SVDFactorization, QRFactorization)

        M === QRFactorization && !with_qr && continue
        m = M()
        x = JLArray(zeros(T, 4))
        fac = factorize!!(init(m, NonlinearProblem(x, ka_noop), x, nothing), m, JLArray(J))
        d = ldiv!!(similar(x), fac, JLArray(b))
        @test d isa JLVector{T}
        h = Array(d)
        @test all(isfinite, h)
        @test maximum(abs, big.(h) .- xref) / maximum(abs, xref) <= 100 * eps(T)
        @test all(isapprox.(big.(h), xref; rtol = 100 * eps(T)))
    end

    # a solution above floatmax overflows, by design: 2^-(emax + 3) I has the solution 2^(emax + 3)
    @testset "a solution above floatmax overflows, $M" for M in (QRFactorization,
        SVDFactorization)
        m = M()
        x = JLArray(zeros(T, 4))
        J = ldexp(one(T), -(exponent(floatmax(T)) + 3)) * Matrix{T}(I, 4, 4)
        fac = factorize!!(init(m, NonlinearProblem(x, ka_noop), x, nothing), m, JLArray(J))
        @test !any(isfinite, Array(ldiv!!(similar(x), fac, JLArray(ones(T, 4)))))
    end
end
