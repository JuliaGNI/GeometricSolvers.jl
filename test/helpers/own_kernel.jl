# The reference solve of the vendor fast-path tests, `own_kernel_ldiv!`, the matrix they run it
# on, `cuda_well_conditioned`, its `κ∞`, `cuda_well_conditioned_cond`, and the Hermitian positive
# definite matrix of the Cholesky rows, `cuda_hpd`.
#
# Every agreement row of `test/devices/metal_vendor.jl`, `test/gpu/cuda/runtests.jl` and
# `test/gpu/rocm/runtests.jl` takes its reference from `own_kernel_ldiv!`.
# `test/devices/own_kernel.jl` checks it against the host solve on a `JLArray`, in an ordinary
# run: both of its branches and all four backsolves, through the mixed LU, QR and Cholesky rows,
# the same-type Cholesky rows and the real SVD rows. The same-type LU and QR rows and the complex
# SVD rows run it on a device only.
#
# `factorize!!` divides `J` by `2^e` where the element type of `J` is not that of the factors, or
# is and `e < 0`, and a device `SVD` divides it always, so the `src/linear/ka/` kernels solve with the factors of
# `J / 2^e`; the solve for `J` is their result times `2^-e`, `e = fac.scale_exponent`. The solve
# runs in the factor element type `F`, so the mixed `T ≠ TF` row converts `r` into `fac.rhs`, as
# the `ldiv!!` of `src/linear/mixed.jl` does; that method also scales `r` by its largest entry,
# which changes no solution and which this reference leaves out.

using LinearAlgebra: Diagonal, I, cond, opnorm
using Random: Xoshiro

using GeometricSolvers: DeviceCholesky, DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: ka_cholesky_backsolve!, ka_lu_backsolve!, ka_qr_backsolve!,
                        ka_svd_backsolve!, ldexp2

# The `n × n` matrix and right-hand side of the vendor agreement tests: `randn` with `2n` on the
# diagonal, so that `κ∞` stays below the 10 that the rows assert; above `n = 256` a diagonal matrix
# with two random `16 × 16` blocks, where a dense `randn` costs more than the factorisation it
# feeds.
function cuda_well_conditioned(T::Type, n::Int, seed::Integer = 0x5eed)
    rng = Xoshiro(seed + n)
    if n <= 256
        A = randn(rng, T, n, n) + T(2 * n) * Matrix{T}(I, n, n)
    else
        A = Matrix{T}(Diagonal(fill(T(4), n)))
        A[1:16, 1:16] .+= randn(rng, T, 16, 16) .* real(T)(0.05)
        A[(n - 15):n, (n - 15):n] .+= randn(rng, T, 16, 16) .* real(T)(0.05)
    end
    b = randn(rng, T, n)
    return A, b
end

# `κ∞` of a matrix of `cuda_well_conditioned`. Above `n = 256` that matrix is block diagonal: two
# `16 × 16` corner blocks `B₁`, `B₂` around a diagonal `D`. Then `‖A‖∞` is the largest of `‖B₁‖∞`,
# `‖B₂‖∞` and `max|Dᵢᵢ|`, and `‖A⁻¹‖∞` the largest of `‖B₁⁻¹‖∞`, `‖B₂⁻¹‖∞` and `max 1/|Dᵢᵢ|`, so
# no dense `n × n` inverse is needed. The structure is checked first, so a matrix of another form
# throws instead of getting a wrong `κ∞`.
function cuda_well_conditioned_cond(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A is $(size(A)), not square"))
    n <= 256 && return cond(A, Inf)
    for j in 1:n, i in 1:n

        in_block = (i <= 16 && j <= 16) || (i > n - 16 && j > n - 16)
        i == j || in_block || iszero(A[i, j]) ||
            throw(ArgumentError("A[$i, $j] ≠ 0: A is not block diagonal with 16 × 16 corners"))
    end
    B₁ = A[1:16, 1:16]
    B₂ = A[(n - 15):n, (n - 15):n]
    d = [abs(A[i, i]) for i in 17:(n - 16)]
    norm_A = max(opnorm(B₁, Inf), opnorm(B₂, Inf), maximum(d))
    norm_inv = max(opnorm(inv(B₁), Inf), opnorm(inv(B₂), Inf), inv(minimum(d)))
    return norm_A * norm_inv
end

# The seeded Hermitian positive definite `n × n` matrix `G Gᴴ / n + I` and right-hand side of the
# Cholesky agreement tests.
function cuda_hpd(::Type{T}, n::Int, seed::Integer = 0xc401) where {T}
    rng = Xoshiro(seed + n)
    G = randn(rng, T, n, n)
    A = G * G' / n + Matrix{T}(I, n, n)
    return (A + A') / 2, randn(rng, T, n)
end

# The `src/` kernel that solves with the factors of `fac`, in the factor element type `F`: the
# device kernels of `src/linear/ka/`, called on the factor's own buffers.
own_kernel_backsolve!(fac::DeviceLU, d) = (ka_lu_backsolve!(fac.factors, fac.ipiv, d); d)
own_kernel_backsolve!(fac::DeviceQR, d) = (ka_qr_backsolve!(fac.factors, fac.τ, d); d)
own_kernel_backsolve!(fac::DeviceSVD, d) = (ka_svd_backsolve!(fac, d, d, false); d)
own_kernel_backsolve!(fac::DeviceCholesky, d) = (ka_cholesky_backsolve!(fac.factors, d); d)

"""
    own_kernel_ldiv!(d, fac, r)

Solve `A d = r` with the own `src/linear/ka/` kernels and the device factor `fac` of `A`, and
return `d`.

`d` is written in place and holds `T`, `r` holds `T`, and the factor holds its element type `F`.
Where `T === F` the solve runs in `d`; where they differ it runs in `fac.rhs` and `d` takes the
result converted to `T`. Either way the result is multiplied by `2^-fac.scale_exponent`, the power
of two that `factorize!!` divided `J` by, so that the returned step is the solve of `A`.
"""
function own_kernel_ldiv!(d::AbstractVector{T}, fac, r::AbstractVector) where {T <: Number}
    F = eltype(fac.factors)
    if T === F
        d === r || copyto!(d, r)
        own_kernel_backsolve!(fac, d)
        e = fac.scale_exponent
        iszero(e) || (d .= ldexp2.(d, -e))
    else
        rhs = fac.rhs::AbstractVector{F}
        rhs .= F.(r)
        own_kernel_backsolve!(fac, rhs)
        d .= T.(ldexp2.(rhs, -fac.scale_exponent))
    end
    return d
end
