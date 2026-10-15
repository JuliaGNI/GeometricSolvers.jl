# The SimpleNonlinearSolve baseline of P0.5, run inside the same kernel as the own variants.
# SNS 2.x builds on SciMLBase's NonlinearProblem and differentiates F with ForwardDiff, so the
# Jacobian it forms is not the analytic one: that difference is the point of the baseline. SNS
# reports no statistics object for this problem, so the row carries its return code and the final
# residual, and the iteration count stays unknown.

@inline function sns_newton(A::SMatrix{n, n, T}, b::SVector{n, T}, maxit::Int) where {n, T}
    f = (u, p) -> residual(p[1], p[2], u)
    prob = NonlinearProblem{false}(f, zero(b), (A, b))
    sol = solve(prob, SimpleNewtonRaphson(); maxiters = maxit)
    x = SVector{n, T}(sol.u)
    # The kernel records the exact return-code value in place of an iteration count.
    return x, Int32(sol.retcode), Int32(0)
end
