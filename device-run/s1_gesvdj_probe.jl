# S1 device probe for the DGX Spark (not for merge): what cuSOLVER gesvdj! returns for the
# fixture with singular values 1, 1e-10, 1e-14, and what the CUDA path of SVDFactorization(; rtol)
# records for it. Run from the repository root:
#   julia --startup-file=no --project=test/gpu/cuda device-run/s1_gesvdj_probe.jl
# Changes no file.
using CUDA, LinearAlgebra, GeometricSolvers
using GeometricSolvers: NonlinearProblem, SVDFactorization, dense_factor, factorize!!
src = read(joinpath(pkgdir(GeometricSolvers), "ext", "GeometricSolversCUDAExt.jl"), String)
println("pkgdir = ", pkgdir(GeometricSolvers))
println("ext passes method.rtol at the gesvdj rank: ",
    occursin("svd_rank(Array(fac.σ̂), m_rows, method.rtol)", src))
noop(r, y, p) = nothing
for T in (Float32, Float64, ComplexF32, ComplexF64)
    RT = real(T)
    A = zeros(T, 3, 3)
    A[2, 1], A[3, 2], A[1, 3] = 1, -1e-10, 1e-14
    println("== ", T)
    _, S, _ = CUDA.CUSOLVER.gesvdj!('V', 1, CuArray(A))
    println("  raw gesvdj S, tol = eps  : ", Array(S))
    try
        _, S2, _ = CUDA.CUSOLVER.gesvdj!('V', 1, CuArray(A); tol = RT(1e-30))
        println("  raw gesvdj S, tol = 1e-30: ", Array(S2))
    catch err
        println("  raw gesvdj S, tol = 1e-30: ERROR ", sprint(showerror, err))
    end
    try
        _, S3, _ = CUDA.CUSOLVER.gesvd!('A', 'A', CuArray(A))
        println("  raw gesvd  S             : ", Array(S3))
    catch err
        println("  raw gesvd  S             : ERROR ", sprint(showerror, err))
    end
    x = CuArray(zeros(T, 3))
    prob = NonlinearProblem(x, noop)
    for m in (SVDFactorization(), SVDFactorization(; rtol = 1e-12), SVDFactorization(;
        rtol = 0))
        fac = factorize!!(dense_factor(m, prob, x, nothing), m, CuArray(A))
        println("  A, rtol = ", m.rtol, ": info = ", fac.info, ", rank = ", fac.rank,
            ", σ̂ = ", Array(fac.σ̂))
    end
    B = zeros(T, 3, 3)
    B[2, 1], B[3, 2], B[1, 3] = 1, -1e-3, 1e-5
    for (m, r) in ((SVDFactorization(), 3), (SVDFactorization(; rtol = 1e-2), 1),
        (SVDFactorization(; rtol = 1e-4), 2), (SVDFactorization(; rtol = 0), 3))
        fac = factorize!!(dense_factor(m, prob, x, nothing), m, CuArray(B))
        println("  B, rtol = ", m.rtol, ": rank = ", fac.rank, " (expected ", r, "), σ̂ = ",
            Array(fac.σ̂))
    end
end
