# Device-resident dense Cholesky factorisation `A = L Lᴴ` and solve (R3) on `AbstractGPUArray`,
# implemented in `KernelAbstractions` as kernels of one workgroup: the load kernel refuses a
# non-finite matrix, takes the power-of-two scale and writes the factor buffer; the factor kernel
# runs the right-looking column loop with the trailing update shared by the work-items; and the
# status is read back once per factorisation. No entry is indexed on the host.

const KA_CHOLESKY_WORKGROUP = 64

# `J` into the factor buffer `A`, as `load_factors!` with `cholesky_exponent` and the finiteness
# test of the host `DenseCholesky` give it: where every entry of `J` is finite, the lower triangle
# of `J / 2^e` with zeros above the diagonal, `e` the exponent of the largest |re| or |im| of the
# strict lower triangle and |re| of the diagonal where the element types differ and 0 where they
# agree; where an entry is not finite, `J` as it is in the factor element type, and `e = 0`.
# `status` takes `[info, e]`, with `info = 1` for a non-finite `J` and 0 otherwise. Each work-item
# reads and writes only the entries of its columns `tid`, `tid + G`, …, and reads each entry before
# it writes it, so that `J` may be `A`.
@kernel function ka_cholesky_load_kernel!(A, J, status, n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    @uniform F = eltype(A)
    @uniform RJ = real(eltype(J))
    l_bad = @localmem Int32 (G,)
    l_max = @localmem RJ (G,)
    l_ctrl = @localmem Int32 (2,)
    @inbounds begin
        bad = Int32(0)
        m = zero(RJ)
        j = tid
        while j <= n
            for i in 1:n
                a = J[i, j]
                isfinite(a) || (bad = Int32(1))
                if i > j
                    m = max(m, abs(real(a)), abs(imag(a)))
                elseif i == j
                    m = max(m, abs(real(a)))
                end
            end
            j += G
        end
        l_bad[tid] = bad
        l_max[tid] = m
    end
    @synchronize()
    if tid == 1
        @inbounds begin
            bad = Int32(0)
            m = zero(RJ)
            for t in 1:G
                bad = max(bad, l_bad[t])
                m = max(m, l_max[t])
            end
            # `frexp(m)[2] - 1` is `exponent(m)` for a finite nonzero `m`, and has no `DomainError`
            # branch, which a kernel compiles to a trap
            e = Int32(0)
            if F !== eltype(J) && bad == Int32(0) && !iszero(m)
                emax = exponent(floatmax(RJ))
                e = Int32(clamp(frexp(m)[2] - 1, -emax, emax))
            end
            l_ctrl[1] = e
            l_ctrl[2] = bad
            status[1] = bad
            status[2] = e
        end
    end
    @synchronize()
    @inbounds begin
        e = Int(l_ctrl[1])
        refused = l_ctrl[2] == Int32(1)
        j = tid
        while j <= n
            for i in 1:n
                a = J[i, j]
                A[i, j] = if refused
                    F(a)
                elseif i < j
                    zero(F)
                else
                    iszero(e) ? F(a) : F(ldexp2(a, -e))
                end
            end
            j += G
        end
    end
end

# The Cholesky factor of the loaded buffer `A` in place, column by column, reading the lower
# triangle and the real part of the diagonal only, as LAPACK `potrf` with `uplo = 'L'` does. At step
# `k` the first work-item takes the pivot `real(A[k, k])`, which is the `k`-th leading minor over
# the `(k - 1)`-th: where it is not positive (a zero or a `NaN` included), `status[1] = k` and the
# loop stops, leaving the columns from `k` on as the trailing update left them. Otherwise column
# `k` is divided by its square root, and each work-item updates the trailing columns
# `j = k + tid, k + tid + G, …` by `A[j:n, j] -= A[j:n, k] conj(A[j, k])`. A `status[1] ≠ 0` from
# the load kernel, a refused matrix, skips the loop.
@kernel function ka_cholesky_factor_kernel!(A, status, n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    @uniform F = eltype(A)
    @uniform R = real(F)
    l_ctrl = @localmem Int32 (2,)
    l_pivot = @localmem R (1,)
    if tid == 1
        @inbounds l_ctrl[1] = status[1] == Int32(0) ? Int32(1) : Int32(n + 1)
        @inbounds l_ctrl[2] = Int32(0)
    end
    @synchronize()
    while l_ctrl[1] <= Int32(n)
        if tid == 1
            @inbounds begin
                k = Int(l_ctrl[1])
                d = real(A[k, k])
                if d > zero(R)
                    ljj = sqrt(d)
                    A[k, k] = F(ljj)
                    l_pivot[1] = ljj
                    l_ctrl[2] = Int32(1)
                else
                    status[1] = Int32(k)
                    l_ctrl[2] = Int32(0)
                end
            end
        end
        @synchronize()
        @inbounds if l_ctrl[2] == Int32(1)
            k = Int(l_ctrl[1])
            ljj = l_pivot[1]
            i = k + tid
            while i <= n
                A[i, k] /= ljj
                i += G
            end
        end
        @synchronize()
        @inbounds if l_ctrl[2] == Int32(1)
            k = Int(l_ctrl[1])
            j = k + tid
            while j <= n
                w = conj(A[j, k])
                for i in j:n
                    A[i, j] -= A[i, k] * w
                end
                j += G
            end
        end
        @synchronize()
        if tid == 1
            @inbounds l_ctrl[1] = l_ctrl[2] == Int32(1) ? l_ctrl[1] + Int32(1) :
                                  Int32(n + 1)
        end
        @synchronize()
    end
end

# The solve of `L Lᴴ x = b` in place of `d = b`, with the factor `L` in `A`: the forward
# substitution `L y = b` by columns of `L`, then the back substitution `Lᴴ x = y` by rows of `L`,
# the columns of `Lᴴ`. At each step the first work-item divides by the real diagonal entry, and the
# work-items update the remaining entries of `d` with the stride `G`. The two loops count with
# separate local counters, so that no work-item reads a counter that the other loop has reset.
@kernel function ka_cholesky_backsolve_kernel!(d, @Const(A), n::Int, ::Val{G}) where {G}
    tid = @index(Local, Linear)
    l_k = @localmem Int32 (2,)
    l_x = @localmem eltype(d) (1,)
    if tid == 1
        @inbounds l_k[1] = Int32(1)
        @inbounds l_k[2] = Int32(n)
    end
    @synchronize()
    while l_k[1] <= Int32(n)
        if tid == 1
            @inbounds begin
                k = Int(l_k[1])
                xk = d[k] / real(A[k, k])
                d[k] = xk
                l_x[1] = xk
            end
        end
        @synchronize()
        @inbounds begin
            k = Int(l_k[1])
            xk = l_x[1]
            i = k + tid
            while i <= n
                d[i] -= A[i, k] * xk
                i += G
            end
        end
        @synchronize()
        if tid == 1
            @inbounds l_k[1] += Int32(1)
        end
        @synchronize()
    end
    while l_k[2] >= Int32(1)
        if tid == 1
            @inbounds begin
                k = Int(l_k[2])
                xk = d[k] / real(A[k, k])
                d[k] = xk
                l_x[1] = xk
            end
        end
        @synchronize()
        @inbounds begin
            k = Int(l_k[2])
            xk = l_x[1]
            i = tid
            while i < k
                d[i] -= conj(A[k, i]) * xk
                i += G
            end
        end
        @synchronize()
        if tid == 1
            @inbounds l_k[2] -= Int32(1)
        end
        @synchronize()
    end
end

# `J` on the device of `fac` and of the size of its factors, or an exception: the load kernel
# reads `J` by index.
function check_cholesky_matrix(fac::DeviceCholesky, J::AbstractMatrix)
    size(J) == size(fac.factors) || throw(DimensionMismatch(
        "J has size $(size(J)), the Cholesky factor $(size(fac.factors))"))
    get_backend(J) == get_backend(fac.factors) || throw(ArgumentError(
        "J is on the backend $(get_backend(J)), the Cholesky factor on " *
        "$(get_backend(fac.factors))"))
    return nothing
end

function launch_cholesky_load!(fac::DeviceCholesky, J::AbstractMatrix, backend)
    G = KA_CHOLESKY_WORKGROUP
    n = size(fac.factors, 1)
    ka_cholesky_load_kernel!(backend, G)(fac.factors, J, fac.status, n, Val(G); ndrange = G)
    return nothing
end

# The status `[info, e]` of `fac`, read back from the device.
function cholesky_status(fac::DeviceCholesky, backend)
    isgpu(backend) && synchronize(backend)
    raw = Array(fac.status)
    return Int(raw[1]), Int(raw[2])
end

# `J` into `fac.factors` by the load kernel, and the status read back: `(info, e)`, `info = 1`
# where `J` has a non-finite entry and 0 otherwise, `e` the exponent of the power of two that `J`
# was divided by. A vendor `potrf` method calls it before the vendor call, so that a non-finite
# matrix never reaches `potrf`. `n = 0` launches nothing and gives `(0, 0)`.
function ka_load_cholesky!(fac::DeviceCholesky, J::AbstractMatrix)
    check_cholesky_matrix(fac, J)
    isempty(fac.factors) && return 0, 0
    backend = get_backend(fac.factors)
    launch_cholesky_load!(fac, J, backend)
    return cholesky_status(fac, backend)
end

# The factor of `J` by the own kernels: the load kernel and the factor kernel, launched one after
# the other, and one read-back of the status. `n = 0` launches nothing and gives `SUCCESS`.
function ka_factorize_cholesky!(fac::DeviceCholesky, J::AbstractMatrix)
    check_cholesky_matrix(fac, J)
    isempty(fac.factors) && return DeviceCholesky(fac.factors, fac.rhs, 0, 0, fac.status)
    G = KA_CHOLESKY_WORKGROUP
    n = size(fac.factors, 1)
    backend = get_backend(fac.factors)
    launch_cholesky_load!(fac, J, backend)
    ka_cholesky_factor_kernel!(backend, G)(fac.factors, fac.status, n, Val(G); ndrange = G)
    info, e = cholesky_status(fac, backend)
    return DeviceCholesky(fac.factors, fac.rhs, info, e, fac.status)
end

function factorize!!(fac::DeviceCholesky, ::CholeskyFactorization, J::AbstractMatrix)
    return ka_factorize_cholesky!(fac, J)
end

# The solve of `L Lᴴ x = d` in place, with the Cholesky factor `L` in the device matrix `A`, by the
# own backsolve kernel; returns `d`.
function ka_cholesky_backsolve!(A::AbstractGPUArray, d::AbstractVector)
    n = length(d)
    n == 0 && return d
    G = KA_CHOLESKY_WORKGROUP
    backend = get_backend(A)
    ka_cholesky_backsolve_kernel!(backend, G)(d, A, n, Val(G); ndrange = G)
    isgpu(backend) && synchronize(backend)
    return d
end

function backsolve!(fac::DeviceCholesky{<:AbstractMatrix{T}},
        d::AbstractVector{T}) where {T <: Number}
    return ka_cholesky_backsolve!(fac.factors, d)
end
