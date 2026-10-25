# Self-contained reproducer for 4x4 Float32 static Householder QR on Metal.
# Exits 0 when the Metal kernel matches the BigFloat reference within
# 4n * eps(Float32) * κ∞(A), 1 when the Metal solve exceeds that bound,
# and 3 when Metal is unavailable or non-functional.

const METAL_LOADED = try
    @eval using Metal
    true
catch
    false
end

if !METAL_LOADED || !Metal.functional()
    println("Metal backend is unavailable or not functional; skipping probe.")
    exit(3)
end

using GeometricSolvers: QRFactorization, backsolve, factorize!!
using KernelAbstractions
using LinearAlgebra: norm, opnorm
using StaticArrays: SMatrix, SVector

@kernel function qr_probe_kernel!(x, τ_out, R_out, @Const(A), @Const(b))
    idx = @index(Global, Linear)
    @inbounds begin
        fac = factorize!!(nothing, QRFactorization(), A[idx])
        τ_out[idx] = fac.τ
        R_out[idx] = fac.factors
        x[idx] = backsolve(fac, b[idx])
    end
end

function run_probe()
    # Explicit well-conditioned 4x4 Float32 system (κ∞(A) ≈ 3.46).
    A = SMatrix{4, 4, Float32, 16}(
        1.0f0, 0.25f0, -0.5f0, 0.125f0,
        -0.25f0, 1.0f0, 0.25f0, -0.5f0,
        0.5f0, -0.25f0, 1.0f0, 0.5f0,
        0.125f0, 0.5f0, -0.25f0, 1.0f0
    )
    b = SVector{4, Float32}(1.0f0, -2.0f0, 0.5f0, 1.5f0)

    A_big = BigFloat.(Matrix(A))
    b_big = BigFloat.(Vector(b))
    x_ref = A_big \ b_big
    κ = Float64(opnorm(A_big, Inf) * opnorm(inv(A_big), Inf))
    bound = 4 * 4 * Float64(eps(Float32)) * κ

    fac_cpu = factorize!!(nothing, QRFactorization(), A)
    x_cpu = backsolve(fac_cpu, b)
    err_cpu = Float64(norm(BigFloat.(Vector(x_cpu)) - x_ref, Inf) / norm(x_ref, Inf))

    backend = MetalBackend()
    A_d = KernelAbstractions.allocate(backend, typeof(A), 1)
    b_d = KernelAbstractions.allocate(backend, typeof(b), 1)
    x_d = KernelAbstractions.allocate(backend, typeof(b), 1)
    τ_d = KernelAbstractions.allocate(backend, typeof(fac_cpu.τ), 1)
    R_d = KernelAbstractions.allocate(backend, typeof(fac_cpu.factors), 1)
    copyto!(A_d, [A])
    copyto!(b_d, [b])

    qr_probe_kernel!(backend)(x_d, τ_d, R_d, A_d, b_d; ndrange = 1)
    KernelAbstractions.synchronize(backend)

    x_mtl = Array(x_d)[1]
    τ_mtl = Array(τ_d)[1]
    R_mtl = Array(R_d)[1]
    err_mtl = Float64(norm(BigFloat.(Vector(x_mtl)) - x_ref, Inf) / norm(x_ref, Inf))

    println("κ∞(A)       = ", κ)
    println("bound       = ", bound)
    println("CPU τ       = ", fac_cpu.τ)
    println("Metal τ     = ", τ_mtl)
    println("CPU R[:,3]  = ", fac_cpu.factors[:, 3])
    println("Metal R[:,3]= ", R_mtl[:, 3])
    println("CPU x       = ", x_cpu, " (relerr = ", err_cpu, ")")
    println("Metal x     = ", x_mtl, " (relerr = ", err_mtl, ")")

    return err_mtl <= bound ? 0 : 1
end

exit(run_probe())
