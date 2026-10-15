# The R1 LU factorisation with partial pivoting: a scalar loop that one thread runs, on a local
# `MMatrix` that the compiler keeps off the heap (part E measured it on the CPU, Metal and CUDA).

# The pivot measure |re| + |im| of LAPACK's `izamax`: no `sqrt` per entry, as `abs` has, and no
# overflow above `sqrt(floatmax)`, as `abs2` has.
abs1(a) = abs(real(a)) + abs(imag(a))

function factorize!!(::Union{Nothing, StaticLU},
        method::LUFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    F = factor_eltype(method, T)
    # convert as an SMatrix: the converting MMatrix constructor allocates at n = 16 (measured)
    A = MMatrix(SMatrix{N, N, F}(J))
    ipiv = zero(SVector{N, Int32})
    info = Int32(0)
    @inbounds for k in 1:N
        # the first largest entry of the subcolumn
        p = k
        big = abs1(A[k, k])
        for i in (k + 1):N
            a = abs1(A[i, k])
            if a > big
                big = a
                p = i
            end
        end
        ipiv = Base.setindex(ipiv, Int32(p), k)
        if p != k
            for j in 1:N
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
        end
        akk = A[k, k]
        if iszero(akk)
            # a zero pivot: skip the column's division and go on, as LAPACK's `getrf`
            info == 0 && (info = Int32(k))
        else
            # a[i,k] / a[k,k], not a[i,k] * inv(a[k,k]): two equal rows then cancel exactly
            for i in (k + 1):N
                A[i, k] = A[i, k] / akk
            end
            for j in (k + 1):N
                for i in (k + 1):N
                    A[i, j] -= A[i, k] * A[k, j]
                end
            end
        end
    end
    return StaticLU(SMatrix(A), ipiv, info)
end

function ldiv!!(::Nothing, fac::StaticLU{N, TF}, r::StaticVector{N, TF}) where {N, TF}
    A = fac.factors
    x = MVector{N, TF}(r)
    @inbounds begin
        for k in 1:N
            p = fac.ipiv[k]
            if p != k
                x[k], x[p] = x[p], x[k]
            end
        end
        for k in 1:N
            xk = x[k]
            for i in (k + 1):N
                x[i] -= A[i, k] * xk
            end
        end
        for k in N:-1:1
            xk = x[k] / A[k, k]
            x[k] = xk
            for i in 1:(k - 1)
                x[i] -= A[i, k] * xk
            end
        end
    end
    return SVector(x)
end
