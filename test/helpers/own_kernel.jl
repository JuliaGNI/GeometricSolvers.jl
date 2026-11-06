# The reference solve of the vendor fast-path tests, `own_kernel_ldiv!`, and the matrix they run
# it on, `cuda_well_conditioned`.
#
# Every agreement row of `test/devices/metal_vendor.jl` and of `test/gpu/cuda/runtests.jl` takes
# its reference from `own_kernel_ldiv!`. `test/devices/own_kernel.jl` checks it against the host
# solve on a `JLArray`, in an ordinary run: both of its branches and all three backsolves, through
# the mixed LU and QR rows and the real SVD rows. The same-type LU and QR rows and the complex SVD
# rows run it on a device only.
#
# `factorize!!` divides `J` by `2^e` where the element type of `J` is not that of the factors, and
# a device `SVD` divides it always, so the `src/linear/ka/` kernels solve with the factors of
# `J / 2^e`; the solve for `J` is their result times `2^-e`, `e = fac.scale_exponent`. The solve
# runs in the factor element type `F`, so the mixed `T ≠ TF` row converts `r` into `fac.rhs`, as
# the `ldiv!!` of `src/linear/mixed.jl` does; that method also scales `r` by its largest entry,
# which changes no solution and which this reference leaves out.

using LinearAlgebra: Diagonal, I
using Random: Xoshiro

using GeometricSolvers: DeviceLU, DeviceQR, DeviceSVD
using GeometricSolvers: ka_lu_backsolve!, ka_qr_backsolve!, ka_svd_backsolve!, ldexp2

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

# The `src/` kernel that solves with the factors of `fac`, in the factor element type `F`: the
# device kernels of `src/linear/ka/`, called on the factor's own buffers.
own_kernel_backsolve!(fac::DeviceLU, d) = (ka_lu_backsolve!(fac.factors, fac.ipiv, d); d)
own_kernel_backsolve!(fac::DeviceQR, d) = (ka_qr_backsolve!(fac.factors, fac.τ, d); d)
own_kernel_backsolve!(fac::DeviceSVD, d) = (ka_svd_backsolve!(fac, d, false); d)

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
