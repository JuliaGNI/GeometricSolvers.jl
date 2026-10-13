# The KA kernels of part E: one thread per system, R1 inside — the R2 launcher of §1.1. Each
# kernel writes x into `out` and the iteration count into `iters`, where a negative count stands
# for a singular factorisation (info in place of a throw, §5.2).

@kernel function newton_M_kernel!(out, iters, @Const(A), @Const(b), maxit)
    i = @index(Global)
    x, it, info = newton_M(A, b[i], maxit)
    out[i] = x
    iters[i] = info == 0 ? Int32(it) : Int32(-info)
end

@kernel function newton_S_kernel!(out, iters, @Const(A), @Const(b), maxit)
    i = @index(Global)
    x, it, info = newton_S(A, b[i], maxit)
    out[i] = x
    iters[i] = info == 0 ? Int32(it) : Int32(-info)
end

@kernel function newton_mixed_M_kernel!(out, iters, @Const(A), @Const(b), maxit)
    i = @index(Global)
    x, it, info = newton_mixed_M(A, b[i], maxit)
    out[i] = x
    iters[i] = info == 0 ? Int32(it) : Int32(-info)
end

@kernel function newton_mixed_S_kernel!(out, iters, @Const(A), @Const(b), maxit)
    i = @index(Global)
    x, it, info = newton_mixed_S(A, b[i], maxit)
    out[i] = x
    iters[i] = info == 0 ? Int32(it) : Int32(-info)
end

@kernel function newton_SNS_kernel!(out, iters, @Const(A), @Const(b), maxit)
    i = @index(Global)
    x, it, info = sns_newton(A, b[i], maxit)
    out[i] = x
    iters[i] = info == 0 ? Int32(it) : Int32(-info)
end

# The five kernels, by the variant name the output rows use.
KERNELS = Dict(
    "M" => newton_M_kernel!,
    "S" => newton_S_kernel!,
    "MIXED-M" => newton_mixed_M_kernel!,
    "MIXED-S" => newton_mixed_S_kernel!,
    "SNS" => newton_SNS_kernel!)
