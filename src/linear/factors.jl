# The factors of the linear-solver methods, and the two functions of the linear-solver seam.
#
# `factorize!!` and `ldiv!!` follow the `!!` convention of `bang.jl`: the first argument is a
# buffer where the storage has one and `nothing` where it does not, and the caller always uses
# the returned value. R1 has no buffer: its factors are `isbits` values, returned new.

"""
    factorize!!(fac, method, J)

Factorise the matrix `J` with the linear-solver method `method` and return the factors.

For a static matrix (R1), `fac` is `nothing` or an earlier factor of the same type, whose value
is not used, and the result is a new [`StaticLU`](@ref) or [`StaticQR`](@ref). `J` is converted
to the element type of the factors, [`factor_eltype`](@ref). Where that is not the element type
of `J`, `J` is first divided by the power of two `2^e` that brings its largest `|re|` or `|im|`
into `[1, 2)`, or as close as the exponent range allows, and `e` is kept in the factor, so that
the overall scale of `J` neither overflows nor underflows in the factor type. A zero or
non-finite `J` gives `e = 0`. A singular `J` does not throw. LU
records the first zero pivot in `info`, QR the first zero diagonal entry of `R`, and
[`returncode`](@ref) then gives `SINGULAR`. QR is not rank-revealing: a singular `J` whose `R` has
no exact zero on the diagonal gives `SUCCESS`, and two equal rows, as a rule, give no exact zero.

A precision marker such as [`TF32`](@ref) has no method for a static matrix: the call raises a
`MethodError`.
"""
function factorize!! end

"""
    ldiv!!(d, fac, r)

Solve `A d = r` with the factors `fac` of `A`, and return `d` in the element type of `r`.

For a static factor (R1), `d` is `nothing` and the result is a new `SVector`. Where the element
type of `r` is the element type of the factors, the solve runs directly in it. Where they differ,
a finite, nonzero `r` is scaled by `s = maximum(abs, r)`, converted to the factor type, solved,
converted back and multiplied by `s` (Kelley 2023, eqs. 2.17–2.18). Normalisation prevents
overflow from the overall right-hand-side scale in a lower factor precision; entries small
relative to the largest entry can still underflow. The scale takes no squares and overflows
only where an entry's absolute value does. In the mixed-type path, a zero `r` returns a zero
step without solving, and an `Inf` or `NaN` in `r` gives a non-finite step. In both paths the
step is multiplied by `2^-e`, the power of two that [`factorize!!`](@ref) divided `J` by.
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
    returncode(fac)

`SUCCESS` for a factor with `info == 0`, and `SINGULAR` otherwise.
"""
returncode(fac::Union{StaticLU, StaticQR}) = iszero(fac.info) ? SUCCESS : SINGULAR

# `a · 2^e`, exact unless the result leaves the floating-point range; by component for a complex `a`.
ldexp2(a::Real, e) = ldexp(float(a), e)
ldexp2(a::Complex, e) = complex(ldexp2(real(a), e), ldexp2(imag(a), e))

# `J` in the factor element type `F`, as an `SMatrix`, which keeps the conversion off the heap, and
# the exponent `e` of the power of two that `J` was divided by: 0 where `F` is the element type of
# `J`, else the exponent of the largest |re| or |im| of `J`, which takes no `abs` that can overflow.
# `e` is clamped to the exponent range, so that `2^-e` is a float and the product with it is exact.
function factor_matrix(::Type{F}, J::StaticMatrix{N, N, T}) where {F, N, T}
    F === T && return SMatrix{N, N, F}(J), Int32(0)
    m = float(maximum(a -> max(abs(real(a)), abs(imag(a))), J))
    emax = exponent(floatmax(m))
    e = isfinite(m) && !iszero(m) ? clamp(exponent(m), -emax, emax) : 0
    return SMatrix{N, N, F}(SMatrix(J) * ldexp(one(m), -e)), Int32(e)
end

# The step for the matrix `J`, from the step `d` for `J / 2^e` that the stored factors solve.
function unscale(fac, d)
    e = fac.scale_exponent
    return iszero(e) ? d : map(a -> ldexp2(a, -e), d)
end

function ldiv!!(::Nothing, fac::Union{StaticLU{N, TF}, StaticQR{N, TF}},
        r::StaticVector{N, TF}) where {N, TF}
    return unscale(fac, backsolve(fac, SVector{N, TF}(r)))
end

# The `T ≠ TF` method: the right-hand side has another element type than the factors.
function ldiv!!(::Nothing, fac::Union{StaticLU{N, TF}, StaticQR{N, TF}},
        r::StaticVector{N, T}) where {N, TF, T}
    s = maximum(abs, r)
    iszero(s) && return zero(SVector{N, T})
    d = backsolve(fac, SVector{N, TF}(SVector{N, T}(r) / s))
    return unscale(fac, SVector{N, T}(d) * s)
end
