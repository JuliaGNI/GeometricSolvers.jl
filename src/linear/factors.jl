# The factors of the linear-solver methods, and the two functions of the linear-solver seam.
#
# `factorize!!` and `ldiv!!` follow the `!!` convention of `bang.jl`: the first argument is a
# buffer where the storage has one and `nothing` where it does not, and the caller always uses
# the returned value. R1 has no buffer: its factors are `isbits` values, returned new. R3 holds
# its Jacobian and workspace buffers in `DenseLU`, `DenseQR` and `DenseSVD`.

"""
    factorize!!(fac, method, J)

Factorise the matrix `J` with the linear-solver method `method` and return the factors.

For a static matrix (R1), `fac` is `nothing` or an earlier factor of the same type, whose value
is not used, and the result is a new [`StaticLU`](@ref), [`StaticQR`](@ref) or
[`StaticSVD`](@ref). `J` is converted to the element type of the factors,
[`factor_eltype`](@ref). Where that is not the element type of `J`, `J` is first divided by the
power of two `2^e` that brings its largest `|re|` or `|im|` into `[1, 2)`, or as close as the
exponent range allows, and `e` is kept in the factor, so that the overall scale of `J` neither
overflows nor underflows in the factor type. A zero or non-finite `J` gives `e = 0`. A singular
`J` does not throw. LU
records the first zero pivot in `info`, QR the first zero diagonal entry of `R`, and
[`returncode`](@ref) then gives `SINGULAR`. QR is not rank-revealing: a singular `J` whose `R` has
no exact zero on the diagonal gives `SUCCESS`, and two equal rows, as a rule, give no exact zero.
The SVD records the first singular value that its rank cutoff `√n · eps(TF) · σ₁` counts as zero,
and it is prescaled at the working precision too, because the sweep squares the entries.

For a dense factor (R3), `fac` is a [`DenseLU`](@ref), [`DenseQR`](@ref) or [`DenseSVD`](@ref)
built by [`init`](@ref). `J` is written into `fac.factors`, divided by `2^e` as for a static
matrix where its element type is not that of `fac.factors`, and `fac.factors` is factorised in
place; `J === fac.factors` is factorised as it is. For [`DenseSVD`](@ref), `info` is `gesvd`'s
non-convergence code, and a rank deficiency gives `info == 0` and `SUCCESS`. An `Inf` in a dense
`J` does not throw, and the step that [`ldiv!!`](@ref) then gives is not specified.

A precision marker such as [`TF32`](@ref) has no method for a static matrix: the call raises a
`MethodError`.
"""
function factorize!! end

"""
    ldiv!!(d, fac, r)

Solve `A d = r` with the factors `fac` of `A`, and return `d` in the element type of `r`.

For a static factor (R1), `d` is `nothing` and the result is a new `SVector`. For a dense factor
(R3), `d` is the vector buffer written in place and returned. Where the element
type of `r` is the element type of the factors, the solve runs directly in it. Where they differ,
a finite, nonzero `r` is scaled by `s = maximum(abs, r)`, converted to the factor type, solved,
converted back and multiplied by `s` (Kelley 2023, eqs. 2.17–2.18). Normalisation prevents
overflow from the overall right-hand-side scale in a lower factor precision; entries small
relative to the largest entry can still underflow. The scale takes no squares and overflows
only where an entry's absolute value does. In the mixed-type path, a zero `r` returns a zero
step without solving, and an `Inf` or `NaN` in `r` gives a non-finite step. In both paths the
step is multiplied by `2^-e`, the power of two that [`factorize!!`](@ref) divided `J` by, in the
element type of `r`. The SVD's solve leaves out the directions that its rank cutoff counted as
zero, so that a rank-deficient system gives a finite, truncated step rather than an enormous one.
"""
function ldiv!! end

"""
    factor_eltype(method, T)

The element type of the factors of a working element type `T`: `T` when the method's
factorisation precision is `Nothing`, the precision `TF` for a real `T`, and `complex(TF)` for a
complex `T`.
"""
factor_eltype(::LinearMethod{Nothing}, ::Type{T}) where {T <: Number} = T
factor_eltype(::LinearMethod{F}, ::Type{<:Real}) where {F <: AbstractFloat} = F
factor_eltype(::LinearMethod{F}, ::Type{<:Complex}) where {F <: AbstractFloat} = complex(F)

"""
    StaticLU{N, TF, L}

The LU factors with partial pivoting of an `N × N` static matrix, in element type `TF`:
`factors` holds `U` on and above the diagonal and the multipliers of the unit lower `L` below it,
`ipiv` the row exchanged with row `k` at step `k`, `info` the index of the first zero pivot,
or 0, and `scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was divided
by. `L = N * N` makes the `SMatrix` field concrete, so the factor is `isbits`.
"""
struct StaticLU{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    ipiv::SVector{N, Int32}
    info::Int32
    scale_exponent::Int32
end

"""
    StaticQR{N, TF, L}

The Householder QR factors of an `N × N` static matrix, in element type `TF`: `factors` holds `R`
on and above the diagonal and below it the reflector vectors `w`, scaled to `w[k] = 1`; the
reflector of step `k` is `I - τ[k] w wᴴ`. `info` is the index of the first zero diagonal entry of
`R`, or 0, and `scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was
divided by.
"""
struct StaticQR{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    τ::SVector{N, TF}
    info::Int32
    scale_exponent::Int32
end

"""
    StaticSVD{N, TF, R, L}

The singular value decomposition of an `N × N` static matrix, in element type `TF`, computed by
the one-sided Jacobi iteration of `src/linear/static/svd.jl`. `factors` holds `U`, `V` holds the
right singular vectors as columns, and `σ` holds the `N` singular values in non-increasing order
in `R = real(TF)`. `info` is the index of the first singular value that the rank cutoff
`√N · eps(TF) · σ₁` counts as zero, or 0 where there is none, so that
[`returncode`](@ref) gives `SINGULAR` for a rank-deficient matrix as it does for the LU and QR
factors; the truncated directions are zeroed in `U`, and the solve of [`ldiv!!`](@ref) leaves
their coordinates out. `converged` is `false` where the sweep reached its cap of 30 sweeps
before the columns were orthogonal. `scale_exponent` is the exponent `e` of the power of two
`2^e` that the matrix was divided by, and `σ` holds the singular values of the matrix as it was
passed, not of the scaled one: a `σ₁` above `floatmax(R)` — every entry of the matrix finite —
is not representable there, and `σ` then holds `Inf`. `σ̂` holds the singular values of the scaled
matrix, the ones the sweep computed and the ones the solve divides by, so that the limit of `σ`
is the limit of the report alone and not of the step.

`L = N * N` makes the `SMatrix` fields concrete, so the factor is `isbits`.
"""
struct StaticSVD{N, TF, R, L}
    factors::SMatrix{N, N, TF, L}
    V::SMatrix{N, N, TF, L}
    σ::SVector{N, R}
    σ̂::SVector{N, R}
    info::Int32
    converged::Bool
    scale_exponent::Int32
end

"""
    DenseLU{A, P, V, S}

The LU factors with partial pivoting of a dense `n × n` matrix, with `factors` the `n × n`
buffer in the factor element type, `ipiv` the pivot vector of length `n` (`BlasInt` on the host,
`Int32` on a device), `rhs` the right-hand-side buffer of the factor element type when that differs
from the working element type and `nothing` otherwise, `info` the index of the first zero pivot, or
0, `scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was divided by, and
`status` the 1-element `Int32` status buffer on a device array and `nothing` on the host.
"""
struct DenseLU{A <: AbstractMatrix, P <: AbstractVector,
    V <: Union{Nothing, AbstractVector}, S <: Union{Nothing, AbstractVector{Int32}}}
    factors::A
    ipiv::P
    rhs::V
    info::Int
    scale_exponent::Int
    status::S
end

"""
    DenseQR{A, T, W, V}

The Householder QR factors of a dense `n × n` matrix, with `factors` the `n × n` buffer in the
factor element type (`R` on and above the diagonal, reflectors below it), `τ` the reflector scale
vector of length `n`, `work` the workspace buffer (the LAPACK workspace on the host, and the
1-element `Int32` status buffer on a device), `rhs` the right-hand-side buffer of the factor
element type when that differs from the working element type and `nothing` otherwise, `info` the
index of the first zero diagonal entry of `R`, or 0, and `scale_exponent` the exponent `e` of the
power of two `2^e` that the matrix was divided by.
"""
struct DenseQR{A <: AbstractMatrix, T <: AbstractVector, W <: AbstractVector,
    V <: Union{Nothing, AbstractVector}}
    factors::A
    τ::T
    work::W
    rhs::V
    info::Int
    scale_exponent::Int
end

"""
    DenseSVD{A, S, W, RW, T, V}

The singular value decomposition factors `U * Diagonal(S) * VT` of a dense `n × n` matrix, with
`factors` the `n × n` Jacobian buffer in the factor element type, `U` and `VT` the `n × n` unitary
factors, `S` the vector of singular values in the real factor type, `work`, `rwork` and `tmp` the
workspace buffers, `rhs` the right-hand-side buffer of the factor element type when that differs
from the working element type and `nothing` otherwise, `info` `gesvd`'s non-convergence code, or
0, and `scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was divided by.
"""
struct DenseSVD{A <: AbstractMatrix, S <: AbstractVector, W <: AbstractVector,
    RW <: AbstractVector, T <: AbstractVector,
    V <: Union{Nothing, AbstractVector}}
    factors::A
    U::A
    S::S
    VT::A
    work::W
    rwork::RW
    tmp::T
    rhs::V
    info::Int
    scale_exponent::Int
end

"""
    init(method::LinearMethod, prob, x::AbstractVector, p)

Build the dense factor [`DenseLU`](@ref), [`DenseQR`](@ref) or [`DenseSVD`](@ref) of `method` for
the state vector `x`.
"""
function init(m::LUFactorization, prob, x::AbstractVector{T}, p) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    ipiv = x isa AbstractGPUArray ? similar(x, Int32, n) : Vector{BlasInt}(undef, n)
    rhs = F === T ? nothing : similar(x, F, n)
    status = x isa AbstractGPUArray ? similar(x, Int32, 1) : nothing
    return DenseLU(factors, ipiv, rhs, 0, 0, status)
end

function init(m::QRFactorization, prob, x::AbstractVector{T}, p) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    τ = similar(x, F, n)
    work = x isa AbstractGPUArray ? similar(x, Int32, 1) : similar(x, F, max(1, 64 * n))
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseQR(factors, τ, work, rhs, 0, 0)
end

function init(m::SVDFactorization, prob, x::AbstractVector{T}, p) where {T <: Number}
    F = factor_eltype(m, T)
    R = real(F)
    n = length(x)
    factors = similar(x, F, n, n)
    U = similar(x, F, n, n)
    S = similar(x, R, n)
    VT = similar(x, F, n, n)
    work = similar(x, F, max(1, 128 * n))
    rwork = similar(x, R, F <: Complex ? max(1, 5 * n) : 0)
    tmp = similar(x, F, n)
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseSVD(factors, U, S, VT, work, rwork, tmp, rhs, 0, 0)
end

"""
    returncode(fac)

`SUCCESS` for a factor with `info == 0`, and `SINGULAR` otherwise.
"""
function returncode(
        fac::Union{StaticLU, StaticQR, StaticSVD, DenseLU, DenseQR, DenseSVD})
    return iszero(fac.info) ? SUCCESS : SINGULAR
end

# `a · 2^e`, exact unless the result leaves the floating-point range; by component for a complex `a`.
ldexp2(a::Real, e) = ldexp(float(a), e)
ldexp2(a::Complex, e) = complex(ldexp2(real(a), e), ldexp2(imag(a), e))

# `J` in the factor element type `F`, as an `SMatrix`, which keeps the conversion off the heap, and
# the exponent `e` of the power of two that `J` was divided by: 0 where `F` is the element type of
# `J`, else the exponent of the largest |re| or |im| of `J`, which takes no `abs` that can overflow.
# `e` is clamped to the exponent range, so that `2^-e` is a float and the product with it is exact.
function factor_matrix(::Type{F}, J::StaticMatrix{N, N, T}) where {F, N, T}
    F === T && return SMatrix{N, N, F}(J), Int32(0)
    e = matrix_exponent(J)
    return SMatrix{N, N, F}(SMatrix(J) * ldexp(one(float(real(T))), -e)), Int32(e)
end

# The exponent of the largest |re| or |im| of a nonempty `J`, clamped to the exponent range, and 0
# for a zero or non-finite `J`.
function matrix_exponent(J::AbstractMatrix)
    m = float(maximum(a -> max(abs(real(a)), abs(imag(a))), J))
    emax = exponent(floatmax(m))
    return isfinite(m) && !iszero(m) ? clamp(exponent(m), -emax, emax) : 0
end

# The step for the matrix `J`, from the step `d` for `J / 2^e` that the stored factors solve.
function unscale(fac, d)
    e = fac.scale_exponent
    return iszero(e) ? d : map(a -> ldexp2(a, -e), d)
end

function ldiv!!(::Nothing, fac::Union{StaticLU{N, TF}, StaticQR{N, TF}, StaticSVD{N, TF}},
        r::StaticVector{N, TF}) where {N, TF}
    return unscale(fac, backsolve(fac, SVector{N, TF}(r)))
end

# The `T ≠ TF` method: the right-hand side has another element type than the factors. With
# `s = frac · 2^ex` and `frac` in `[0.5, 1)`, the step is multiplied by `frac` and then by
# `2^(ex - e)` in one step, with `e` the scale exponent of the factors, both in the wider of `TF`
# and `T`, and only then converted to `T`, so that it overflows or underflows only where the
# solution itself does.
function ldiv!!(::Nothing, fac::Union{StaticLU{N, TF}, StaticQR{N, TF}, StaticSVD{N, TF}},
        r::StaticVector{N, T}) where {N, TF, T}
    s = maximum(abs, r)
    iszero(s) && return zero(SVector{N, T})
    d = backsolve(fac, SVector{N, TF}(SVector{N, T}(r) / s))
    frac, ex = frexp(s)
    return map(a -> T(ldexp2(a * frac, ex - fac.scale_exponent)), d)
end
