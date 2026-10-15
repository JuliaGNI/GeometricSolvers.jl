# The R1 LU factorisation with partial pivoting: a scalar loop that one thread runs, on a local
# `MMatrix` that the compiler keeps off the heap on the CPU, on Metal and on CUDA.

# The pivot measure |re| + |im| of LAPACK's `izamax`: no `sqrt` per entry, as `abs` has, and no
# overflow above `sqrt(floatmax)`, as `abs2` has.
abs1(a) = abs(real(a)) + abs(imag(a))

# The multiplier a / p, chosen so that a twin row cancels exactly: for `a == p` it is exactly one.
# A real `a / p` is. Julia's complex `/` is not (for ComplexF64 it gives z / z != 1 for some z), so
# a complex multiplier is (a′ conj(p′)) / |p′|², with a′ and p′ scaled by s = max(|re p|, |im p|).
# The larger component of p′ is then exactly ±1, so |p′|² lies in [1, 2], and s is finite for a
# finite p, where |re p| + |im p| overflows once both components exceed floatmax / 2. For a == p
# the product a′ conj(a′) has the imaginary part 0 and the real part |a′|², rounded as `abs2`
# rounds it. Not a * inv(p): two equal rows then do not cancel.
multiplier(a::Real, p::Real) = a / p
function multiplier(a::Complex, p::Complex)
    s = max(abs(real(p)), abs(imag(p)))
    a′ = a / s
    p′ = p / s
    return (a′ * conj(p′)) / abs2(p′)
end

function factorize!!(::Union{Nothing, StaticLU},
        method::LUFactorization{<:Union{Nothing, AbstractFloat}},
        J::StaticMatrix{N, N, T}) where {N, T}
    F = factor_eltype(method, T)
    # convert through an SMatrix to keep the element-type conversion off the heap
    A = MMatrix(SMatrix{N, N, F}(J))
    ipiv = zero(SVector{N, Int32})
    info = Int32(0)
    @inbounds for k in 1:N
        # the first largest entry of the subcolumn
        p = k
        amax = abs1(A[k, k])
        for i in (k + 1):N
            a = abs1(A[i, k])
            if a > amax
                amax = a
                p = i
            end
        end
        ipiv = setindex(ipiv, Int32(p), k)
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
            for i in (k + 1):N
                A[i, k] = multiplier(A[i, k], akk)
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
