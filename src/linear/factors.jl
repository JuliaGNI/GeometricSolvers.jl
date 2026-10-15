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
to the element type of the factors, [`factor_eltype`](@ref). A singular `J` does not throw. LU
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
type of `r` is the element type of the factors, the solve runs in it. Where they differ, `r` is
scaled by `s = maximum(abs, r)`, converted to the factor type, solved, converted back and
multiplied by `s` (Kelley 2023, eqs. 2.17–2.18): every entry of the scaled right-hand side has an
absolute value of at most 1, whatever the range of `r`, so it neither overflows nor underflows in
a lower factor precision. The scale takes no squares: it overflows only where the absolute value
of an entry does, and it is zero only for a zero `r`. A zero `r` gives a zero step, and an `Inf`
or `NaN` in `r` gives a non-finite step; nothing throws.
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
`ipiv` the row exchanged with row `k` at step `k`, and `info` the index of the first zero pivot,
or 0. `L = N * N` makes the `SMatrix` field concrete, so the factor is `isbits`.
"""
struct StaticLU{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    ipiv::SVector{N, Int32}
    info::Int32
end

"""
    StaticQR{N, TF, L}

The Householder QR factors of an `N × N` static matrix, in element type `TF`: `factors` holds `R`
on and above the diagonal and below it the reflector vectors `w`, scaled to `w[k] = 1`; the
reflector of step `k` is `I - τ[k] w wᴴ`. `info` is the index of the first zero diagonal entry of
`R`, or 0.
"""
struct StaticQR{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    τ::SVector{N, TF}
    info::Int32
end

"""
    returncode(fac)

`SUCCESS` for a factor with `info == 0`, and `SINGULAR` otherwise.
"""
returncode(fac) = iszero(fac.info) ? SUCCESS : SINGULAR

# The `T ≠ TF` method: the right-hand side has another element type than the factors.
function ldiv!!(::Nothing, fac::Union{StaticLU{N, TF}, StaticQR{N, TF}},
        r::StaticVector{N, T}) where {N, TF, T}
    s = maximum(abs, r)
    iszero(s) && return zero(SVector{N, T})
    d = ldiv!!(nothing, fac, SVector{N, TF}(SVector{N, T}(r) / s))
    return SVector{N, T}(d) * s
end
