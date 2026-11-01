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
The SVD records its numerical rank in `rank`, the number of singular values above the cutoff
`√n · eps(TF) · σ₁`, and a rank deficiency gives `info == 0` and `SUCCESS`; its `info` is nonzero
only where the sweep did not converge or a singular value is not finite, and `rank` is then 0. The
SVD is prescaled at the working precision too, because the sweep squares the entries.

For a dense factor (R3), `fac` is a [`DenseLU`](@ref), [`DenseQR`](@ref) or [`DenseSVD`](@ref)
built by [`init`](@ref). `J` is written into `fac.factors`, divided by `2^e` as for a static
matrix where its element type is not that of `fac.factors`, and `fac.factors` is factorised in
place; `J === fac.factors` is factorised as it is. A [`DenseSVD`](@ref) divides `J` by `2^e`
unconditionally, `J === fac.factors` too: into `fac.factors` on the host before `gesvd`, and into
`fac.U` on a device before the one-sided Jacobi sweep. For [`DenseSVD`](@ref),
`info` is `1` on the host where `J` has a non-finite entry, which is refused before `gesvd` and
leaves `U`, `S` and `VT` unwritten, `gesvd`'s non-convergence code where it fails, or `1` where a
singular value of `J / 2^e` is not finite; on a device it is `1` where the Jacobi
sweep reaches its sweep cap without converging or meets a non-finite entry; `rank` is the
numerical rank as for a static matrix. An `Inf` in a dense `J` does not throw; the SVD then
reports `SINGULAR`, and the step that the LU or QR factors give is not specified.

A precision marker such as [`TF32`](@ref) has no method for a static matrix: the call raises a
`MethodError`.
"""
function factorize!! end

"""
    ldiv!!(d, fac, r)

Solve `A d = r` with the factors `fac` of `A`, and return `d` in the element type of `r`.

For a static factor (R1), `d` is `nothing` and the result is a new `SVector`. For a dense factor
(R3), `d` is the vector buffer written in place and returned. Where the element
type of `r` is the element type of the factors, the solve runs directly in it. There `r` is not
scaled, unlike in the mixed-type path below: the step overflows on every route where the solution
itself exceeds `floatmax`, and the QR and SVD solves, which form `Qᴴ r` or `Uᴴ r` first, can also
overflow for a finite `r` near `floatmax` whose solution is representable, while the LU solve
forms no such product. Where they differ,
a finite, nonzero `r` is scaled by `s = maximum(abs, r)`, converted to the factor type, solved,
converted back and multiplied by `s` (Kelley 2023, eqs. 2.17–2.18). Normalisation prevents
overflow from the overall right-hand-side scale in a lower factor precision; entries small
relative to the largest entry can still underflow. The scale takes no squares and overflows
only where an entry's absolute value does. In the mixed-type path, a zero `r` returns a zero
step without solving, and an `Inf` or `NaN` in `r` gives a non-finite step. In both paths the
step is multiplied by `2^-e`, the power of two that [`factorize!!`](@ref) divided `J` by, in the
element type of `r`. The SVD's solve leaves out the directions that its rank cutoff counted as
zero, so that a rank-deficient system gives a finite, truncated step rather than an enormous one;
an SVD factor with `info ≠ 0` has rank 0 and gives the zero step.
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
right singular vectors as columns, and `σ̂` holds the `N` singular values of the matrix divided by
`2^e`, the ones the sweep computed and the solve divides by, in non-increasing order in
`R = real(TF)`; [`singular_values`](@ref) gives those of the matrix as it was passed. `rank` is
the number of singular values above the rank cutoff `√N · eps(TF) · σ̂₁`; the directions past it
are zeroed in `U`, and the solve of [`ldiv!!`](@ref) leaves their coordinates out. `info` is 1
where the sweep reached its cap of 30 sweeps before the columns were orthogonal or a singular
value is not finite, and `rank` is then 0; it is 0 otherwise, also for a rank-deficient matrix,
so that [`returncode`](@ref) reports a failed factorisation alone. `scale_exponent` is the
exponent `e` of the power of two `2^e` that the matrix was divided by.

`L = N * N` makes the `SMatrix` fields concrete, so the factor is `isbits`.
"""
struct StaticSVD{N, TF, R, L}
    factors::SMatrix{N, N, TF, L}
    V::SMatrix{N, N, TF, L}
    σ̂::SVector{N, R}
    info::Int32
    rank::Int32
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
`factors` the `n × n` Jacobian buffer in the factor element type (which holds `V` during the
device Jacobi sweep), `U` and `VT` the `n × n` unitary factors, `S` the vector of the singular
values `σ̂ · 2^e` of the matrix in the real factor type, which overflow to `Inf` where they exceed
`floatmax`, `work` (the LAPACK workspace on the host, and the 1-element `Int32` rotation-flag
buffer on a device), `rwork` (the real LAPACK workspace during `gesvd` on the host, and the
singular values `σ̂` of the matrix divided by `2^e` in its first `n` entries, on the host after
`gesvd` and on a device after the sweep) and `tmp` the workspace buffers, `rhs`
the right-hand-side buffer of the factor element type when that differs from the working element
type and `nothing` otherwise, `info` `1` where the host matrix has a non-finite entry, which is
refused before `gesvd`, `gesvd`'s non-convergence code, `1` where a host `σ̂` is not finite or the
device Jacobi sweep reaches its sweep cap without converging
or meets a non-finite entry, or `0`, `rank` the number of singular values above
the cutoff `√n · eps(real(TF)) · σ̂₁`, and 0 where `info ≠ 0`, and `scale_exponent` the exponent
`e` of the power of two `2^e` that the matrix was divided by.
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
    rank::Int
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
    work = x isa AbstractGPUArray ? similar(x, Int32, 1) :
           similar(x, F, qr_work_length(factors, τ))
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
    rwork = x isa AbstractGPUArray ? similar(x, R, n) :
            similar(x, R, F <: Complex ? max(1, 5 * n) : n)
    work = x isa AbstractGPUArray ? similar(x, Int32, 1) :
           similar(x, F, svd_work_length(factors, S, U, VT, rwork))
    tmp = similar(x, F, n)
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseSVD(factors, U, S, VT, work, rwork, tmp, rhs, 0, 0, 0)
end

"""
    returncode(fac)

`SUCCESS` for a factor with `info == 0`, and `SINGULAR` otherwise.
"""
function returncode(
        fac::Union{StaticLU, StaticQR, StaticSVD, DenseLU, DenseQR, DenseSVD})
    return iszero(fac.info) ? SUCCESS : SINGULAR
end

"""
    exact_solve(fac)

`true` where the solve of [`ldiv!!`](@ref) with the factor `fac` is the exact solve of the
factored matrix, and `false` for an SVD factor of numerical rank below `n`, whose solve is the
truncated minimum-norm step. `Newton` measures the slope of a truncated step, because `−2φ(0)`
is the slope of an exact Newton step alone.
"""
exact_solve(fac) = true
exact_solve(fac::StaticSVD{N}) where {N} = fac.rank == N
exact_solve(fac::DenseSVD) = fac.rank == length(fac.S)

"""
    singular_values(fac::StaticSVD)

The singular values of the matrix that [`factorize!!`](@ref) factorised, in non-increasing order,
as an `SVector` in `real(TF)`: the stored `σ̂` of the matrix divided by `2^e`, multiplied by `2^e`.
A singular value above `floatmax(real(TF))` overflows to `Inf`, although every entry of the matrix
is finite; the solve divides by `σ̂` and stays finite.
"""
singular_values(fac::StaticSVD) = map(a -> ldexp2(a, fac.scale_exponent), fac.σ̂)

# `a · 2^e`, exact unless the result leaves the floating-point range; by component for a complex `a`.
# Bitwise on `IEEEFloat` so that a device with flush-to-zero enabled (Metal) preserves subnormals.
@inline function ldexp2(x::T, e::Integer) where {T <: Base.IEEEFloat}
    xu = reinterpret(Unsigned, x)
    U = typeof(xu)
    sign = xu & Base.sign_mask(T)
    xs = xu & ~Base.sign_mask(T)
    xs >= Base.exponent_mask(T) && return x
    xs == zero(U) && return x
    sig_bits = Base.significand_bits(T)
    k = Int(xs >> sig_bits)
    if k == 0
        m = leading_zeros(xs) - Base.exponent_bits(T)
        xs <<= unsigned(m)
        k = 1 - m
    end
    sig = (xs & Base.significand_mask(T)) | (one(U) << sig_bits)
    k += Int(e)
    if k >= Base.exponent_raw_max(T)
        return reinterpret(T, sign | Base.exponent_mask(T))
    elseif k > 0
        return reinterpret(T, sign | (U(k) << sig_bits) | (sig & Base.significand_mask(T)))
    elseif k <= -sig_bits - 1
        return reinterpret(T, sign)
    else
        shift = 1 - k
        lower = sig & ((one(U) << shift) - one(U))
        half = one(U) << (shift - 1)
        sub = sig >> shift
        if lower > half || (lower == half && isodd(sub))
            sub += one(U)
        end
        return reinterpret(T, sign | sub)
    end
end
@inline ldexp2(a::AbstractFloat, e::Integer) = ldexp(a, e)
@inline ldexp2(a::Real, e::Integer) = ldexp2(float(a), e)
@inline ldexp2(a::Complex, e::Integer) = complex(ldexp2(real(a), e), ldexp2(imag(a), e))

@inline abs_bits(x::T) where {T <: Base.IEEEFloat} = reinterpret(Unsigned, x) &
                                                     ~Base.sign_mask(T)

# `J` in the factor element type `F`, as an `SMatrix`, which keeps the conversion off the heap, and
# the exponent `e` of the power of two that `J` was divided by: 0 where `F` is the element type of
# `J`, else the exponent of the largest |re| or |im| of `J`, which takes no `abs` that can overflow.
# `e` is clamped to the exponent range, so that `2^-e` is a float and the product with it is exact.
function factor_matrix(::Type{F}, J::StaticMatrix{N, N, T}) where {F, N, T}
    F === T && return SMatrix{N, N, F}(J), Int32(0)
    e = matrix_exponent(J)
    return SMatrix{N, N, F}(SMatrix(J) * ldexp(one(float(real(T))), -e)), Int32(e)
end

# A mutable local copy of a static array for the scalar loops of the R1 kernels: an `MMatrix` or
# `MVector` for an `isbits` element type, which the compiler keeps off the heap, and otherwise a
# `SizedMatrix` or `SizedVector` on a heap array, because a mutable static array of a non-`isbits`
# element type such as `BigFloat` has no `setindex!`. `isbitstype(F)` is a constant of the type,
# so each method compiles to one of its two branches.
@inline function local_array(A::SMatrix{N, M, F}) where {N, M, F}
    return isbitstype(F) ? MMatrix{N, M, F, N * M}(A) : SizedMatrix{N, M, F}(Matrix(A))
end
@inline function local_array(x::SVector{N, F}) where {N, F}
    return isbitstype(F) ? MVector{N, F}(x) : SizedVector{N, F}(Vector(x))
end

# An uninitialised local vector of `N` entries of type `F`, as `local_array` chooses its type.
@inline function local_vector(::Type{F}, ::Val{N}) where {F, N}
    return isbitstype(F) ? MVector{N, F}(undef) : SizedVector{N, F}(undef)
end

# The exponent of the largest |re| or |im| of a nonempty `J`, clamped to the exponent range, and 0
# for a zero or non-finite `J`.
function matrix_exponent(J::AbstractMatrix)
    R = real(eltype(J))
    m = if R <: Base.IEEEFloat
        reinterpret(R, maximum(a -> max(abs_bits(real(a)), abs_bits(imag(a))), J))
    else
        float(maximum(a -> max(abs(real(a)), abs(imag(a))), J))
    end
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
