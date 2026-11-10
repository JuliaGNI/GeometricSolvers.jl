# The mixed-precision seam of the linear solvers: the `T ≠ TF` method of `ldiv!!` for a dense
# factor, and the promotion of a factorisation in `TF` to a copy in the working precision `T`, which
# a GMRES-IR stage applies to its Krylov vectors.

# The `T ≠ TF` method of `ldiv!!` for dense factors: scales a finite, nonzero right-hand side by
# `s = maximum(abs, r)` before converting to the factor element type `F`, solves in `fac.rhs`, and
# converts the step into `d`, multiplied by `s` and by the `2^-e` of the factorisation. With
# `s = frac · 2^ex` and `frac` in `[0.5, 1)`, the step is multiplied by `frac` and then by
# `2^(ex - e)` in one step, both in the wider of `F` and `T`, and only then converted to `T`, so
# that it overflows or underflows only where the solution itself does.

function ldiv!!(
        d::AbstractVector{T}, fac::DenseFactor, r::AbstractVector{T}) where {T <: Number}
    s = maximum(abs, r)
    iszero(s) && return fill!(d, zero(T))
    rhs = fac.rhs::AbstractVector
    F = eltype(rhs)
    rhs .= F.(r ./ s)
    backsolve!(fac, rhs)
    frac, ex = frexp(s)
    if length(rhs) == length(d)
        d .= T.(ldexp2.(rhs .* frac, ex - fac.scale_exponent))
    else
        @views d .= T.(ldexp2.(rhs[1:length(d)] .* frac, ex - fac.scale_exponent))
    end
    return d
end

"""
    stored_factors(factor)

The `n × n` storage of the factors of a factorisation, in the element type of the factor: the
packed LU (`U` on and above the diagonal, the multipliers of the unit `L` below it) of an
[`LUFactorization`](@ref), which is the storage that [`promote_lu_factors!`](@ref) copies into the
working precision. It is `nothing` for a solve that holds no factor, and for a factor that is no
LU factor: a QR, SVD or LDLᵀ factorisation has no GMRES-IR stage that promotes it.
"""
@inline stored_factors(fac::Union{DenseLU, DeviceLU, StaticLU}) = fac.factors
@inline stored_factors(::Union{Nothing, AnyFactor, StaticRectQR, StaticRectSVD}) = nothing

"""
    promote_lu_factors!(fac_promoted, fac_TF)

Copy the LU factors of `fac_TF` into `fac_promoted`, whose element type is the working precision
`T`, and return the promoted factor. `L`, `U` and `ipiv` travel as they are, together with the
`info` and the `scale_exponent` of the factorisation, so that a solve with `fac_promoted` gives the
step of the factorisation in `TF` and never factorises the Jacobian again: the copy is what a
GMRES-IR stage applies to its Krylov vectors, in `T` (Amestoy et al. 2024, p. 13). Both factors are
dense or device factors of one shape, and the copy allocates nothing.
"""
@inline function promote_lu_factors!(
        fac::DenseLU{A, P, V}, fac_TF::DenseLU{B, Q, W}) where {A, P, V, B, Q, W}
    copyto!(fac.factors, fac_TF.factors)
    copyto!(fac.ipiv, fac_TF.ipiv)
    return typeof(fac)(
        fac.factors, fac.ipiv, fac.rhs, fac_TF.info, fac_TF.scale_exponent)
end

@inline function promote_lu_factors!(
        fac::DeviceLU{A, P, V, S}, fac_TF::DeviceLU{B, Q, W, U}) where {
        A, P, V, S, B, Q, W, U}
    copyto!(fac.factors, fac_TF.factors)
    copyto!(fac.ipiv, fac_TF.ipiv)
    return typeof(fac)(
        fac.factors, fac.ipiv, fac.rhs, fac_TF.info, fac_TF.scale_exponent, fac.status)
end

"""
    promote_lu_factors(fac_TF, T)

The LU factors of `fac_TF` as a new factor in the working precision `T`, for an R1 solve whose
factor is `isbits` and which therefore has no buffer to promote into: a [`StaticLU`](@ref) whose
matrix, pivot vector, `info` and `scale_exponent` are those of `fac_TF`, converted. It factorises
no Jacobian, as [`promote_lu_factors!`](@ref) does not.
"""
@inline function promote_lu_factors(fac::StaticLU{N, TF, L}, ::Type{T}) where {N, TF, L, T}
    return StaticLU{N, T, L}(SMatrix{N, N, T, L}(fac.factors), fac.ipiv, fac.info,
        fac.scale_exponent)
end
