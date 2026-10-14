# The KA kernel of part E: one thread per system, R1 inside — the R2 launcher of §1.1. It takes
# the solver as an argument, writes x into `out` and the status into `iters`.

# The iteration count, or minus the factorisation info for a singular factorisation. The SNS
# baseline returns its return-code value as the count.
status(it, info) = info == 0 ? Int32(it) : Int32(-info)

@kernel function newton_kernel!(out, iters, solve, @Const(A), @Const(b), maxit)
    i = @index(Global)
    x, it, info = solve(A, b[i], maxit)
    out[i] = x
    iters[i] = status(it, info)
end
