# The R1 solvers of part E. Newton without a line search, a partially pivoted LU with Int32
# pivots, and the analytic Jacobian, in two storage variants: a local MMatrix mutated in
# place, and functional updates on an SMatrix. Both return x, the iteration count and the
# factorisation info, because a singular matrix is reported, not thrown (§5.2).

const MAXIT = 20

# The infinity norm, the form every bound in §10.E uses.
maxabs(v) = maximum(abs, v)

# --- the stopping rule: ‖Δx‖∞ ≤ 4 n eps(T) ‖x‖∞ --------------------------------------------------
#
# The step and the iterate may have different element types: the mixed row solves a Float64
# problem with a Float32 factorisation, so its step arrives in Float32 (decision 15).

function stop(::Type{T}, k::Int, dx::SVector{n, Tdx}, x::SVector{n, T}) where {T, Tdx, n}
    maxabs(dx) <= 4 * k * eps(T) * maxabs(x)
end

# --- the M variant: a local MMatrix, mutated in place -------------------------------------------

@inline function lu_!(J::MMatrix{n, n, T}) where {n, T}
    piv = zero(SVector{n, Int32})
    info = Int32(0)
    @inbounds for k in 1:n
        p = k
        big = abs(J[k, k])
        for i in (k + 1):n
            a = abs(J[i, k])
            if a > big
                big = a
                p = i
            end
        end
        piv = setindex(piv, Int32(p), k)
        if p != k
            for j in 1:n
                J[k, j], J[p, j] = J[p, j], J[k, j]
            end
        end
        akk = J[k, k]
        if iszero(akk)
            info = Int32(k)
        else
            for i in (k + 1):n
                J[i, k] /= akk
                Jik = J[i, k]
                for j in (k + 1):n
                    J[i, j] -= Jik * J[k, j]
                end
            end
        end
    end
    return piv, info
end

@inline function lu_solve!(J::MMatrix{n, n, T}, piv::SVector{n, Int32}, rhs::SVector{
        n, T}) where {n, T}
    x = MVector{n, T}(rhs)
    @inbounds for k in 1:n
        p = piv[k]
        if p != k
            x[k], x[p] = x[p], x[k]
        end
    end
    @inbounds for k in 1:n
        xk = x[k]
        for i in (k + 1):n
            x[i] -= J[i, k] * xk
        end
    end
    @inbounds for k in n:-1:1
        xk = x[k] / J[k, k]
        x[k] = xk
        for i in 1:(k - 1)
            x[i] -= J[i, k] * xk
        end
    end
    return SVector{n, T}(x)
end

# The Newton step: the Jacobian assembled in place, factored and solved for -f.
@inline function step_M(A::SMatrix{n, n, Tf}, x::SVector{n}, f::SVector{n}) where {n, Tf}
    J = MMatrix{n, n, Tf}(A)
    @inbounds for j in 1:n
        J[j, j] += Tf(cos(x[j]) / 10)
    end
    piv, info = lu_!(J)
    info == 0 || return zero(SVector{n, Tf}), info
    return lu_solve!(J, piv, SVector{n, Tf}(-f)), info
end

# --- the S variant: functional updates on an SMatrix ---------------------------------------------

@inline function lu_S(J0::SMatrix{n, n, T}) where {n, T}
    J = J0
    piv = zero(SVector{n, Int32})
    info = Int32(0)
    @inbounds for k in 1:n
        p = k
        big = abs(J[k, k])
        for i in (k + 1):n
            a = abs(J[i, k])
            if a > big
                big = a
                p = i
            end
        end
        piv = setindex(piv, Int32(p), k)
        if p != k
            for j in 1:n
                jk = J[k, j]
                J = setindex(setindex(J, J[p, j], k, j), jk, p, j)
            end
        end
        akk = J[k, k]
        if iszero(akk)
            info = Int32(k)
        else
            for i in (k + 1):n
                J = setindex(J, J[i, k] / akk, i, k)
                Jik = J[i, k]
                for j in (k + 1):n
                    J = setindex(J, J[i, j] - Jik * J[k, j], i, j)
                end
            end
        end
    end
    return J, piv, info
end

@inline function lu_solve_S(J::SMatrix{n, n, T}, piv::SVector{n, Int32}, rhs::SVector{
        n, T}) where {n, T}
    x = rhs
    @inbounds for k in 1:n
        p = piv[k]
        if p != k
            x = setindex(setindex(x, x[p], k), x[k], p)
        end
    end
    @inbounds for k in 1:n
        xk = x[k]
        for i in (k + 1):n
            x = setindex(x, x[i] - J[i, k] * xk, i)
        end
    end
    @inbounds for k in n:-1:1
        xk = x[k] / J[k, k]
        x = setindex(x, xk, k)
        for i in 1:(k - 1)
            x = setindex(x, x[i] - J[i, k] * xk, i)
        end
    end
    return x
end

@inline function step_S(A::SMatrix{n, n, Tf}, x::SVector{n}, f::SVector{n}) where {n, Tf}
    J = A
    @inbounds for j in 1:n
        J = setindex(J, J[j, j] + Tf(cos(x[j]) / 10), j, j)
    end
    J, piv, info = lu_S(J)
    info == 0 || return zero(SVector{n, Tf}), info
    return lu_solve_S(J, piv, SVector{n, Tf}(-f)), info
end

# --- the Newton iteration -----------------------------------------------------------------------
#
# The residual and the iterate are in the problem's type T; the Jacobian and its LU are in Tf. The
# F32 and F64 rows factor in T, and the mixed row factors a Float64 problem in Float32 (decision 15).

@inline function newton(step::F, A::SMatrix{n, n, T}, b::SVector{n, T}, maxit::Int,
        ::Type{Tf}) where {F, n, T, Tf}
    Af = SMatrix{n, n, Tf}(A)
    x = zero(SVector{n, T})
    iters = 0
    info = Int32(0)
    for i in 1:maxit
        iters = i
        dx, info = step(Af, x, residual(A, b, x))
        info == 0 || break
        x = x + SVector{n, T}(dx)
        stop(T, n, dx, x) && break
    end
    return x, iters, info
end

@inline newton_M(A::SMatrix, b, maxit) = newton(step_M, A, b, maxit, eltype(A))
@inline newton_S(A::SMatrix, b, maxit) = newton(step_S, A, b, maxit, eltype(A))
@inline newton_mixed_M(A::SMatrix, b, maxit) = newton(step_M, A, b, maxit, Float32)
@inline newton_mixed_S(A::SMatrix, b, maxit) = newton(step_S, A, b, maxit, Float32)
