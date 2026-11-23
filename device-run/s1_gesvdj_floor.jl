# S1 fix 2 probe for the DGX Spark (not for merge): the smallest singular value that cuSOLVER
# gesvdj! resolves in single precision, between √3·eps(Float32) ≈ 2.1e-7 (where the route moves to
# the own kernel) and 1e-5 (which the earlier probe resolved). Run from the repository root:
#   julia --startup-file=no --project=test/gpu/cuda device-run/s1_gesvdj_floor.jl
# Changes no file.
using CUDA, LinearAlgebra
for T in (Float32, ComplexF32)
    println("== ", T, "   (sqrt(3)*eps = ", sqrt(3) * eps(real(T)), ")")
    for σ in (3e-7, 1e-6, 3e-6, 1e-5)
        A = zeros(T, 3, 3)
        A[2, 1], A[3, 2], A[1, 3] = 1, -σ, σ / 2
        _, S, _ = CUDA.CUSOLVER.gesvdj!('V', 1, CuArray(A))
        _, S3, _ = CUDA.CUSOLVER.gesvd!('A', 'A', CuArray(A))
        println("  σ = (1, ", σ, ", ", σ / 2, "):  gesvdj S = ", Array(S),
            "   gesvd S = ", Array(S3))
    end
end
