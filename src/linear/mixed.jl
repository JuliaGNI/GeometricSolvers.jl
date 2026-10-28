# The `T ≠ TF` method of `ldiv!!` for dense factors: scales a finite, nonzero right-hand side by
# `s = maximum(abs, r)` before converting to the factor element type `F`, solves in `fac.rhs`, and
# converts the step into `d`, multiplied by `s` and by the `2^-e` of the factorisation.

function ldiv!!(d::AbstractVector{T}, fac::Union{DenseLU, DenseQR, DenseSVD},
        r::AbstractVector{T}) where {T <: Number}
    s = maximum(abs, r)
    iszero(s) && return fill!(d, zero(T))
    rhs = fac.rhs
    F = eltype(rhs)
    rhs .= F.(r ./ s)
    backsolve!(fac, rhs)
    e = fac.scale_exponent
    d .= ldexp2.(T.(rhs) .* s, -e)
    return d
end
