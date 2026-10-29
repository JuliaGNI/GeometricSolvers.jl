# The `T ≠ TF` method of `ldiv!!` for dense factors: scales a finite, nonzero right-hand side by
# `s = maximum(abs, r)` before converting to the factor element type `F`, solves in `fac.rhs`, and
# converts the step into `d`, multiplied by `s` and by the `2^-e` of the factorisation. With
# `s = frac · 2^ex` and `frac` in `[0.5, 1)`, the step is multiplied by `frac` and then by
# `2^(ex - e)` in one step, both in the wider of `F` and `T`, and only then converted to `T`, so
# that it overflows or underflows only where the solution itself does.

function ldiv!!(d::AbstractVector{T}, fac::Union{DenseLU, DenseQR, DenseSVD},
        r::AbstractVector{T}) where {T <: Number}
    s = maximum(abs, r)
    iszero(s) && return fill!(d, zero(T))
    rhs = fac.rhs::AbstractVector
    F = eltype(rhs)
    rhs .= F.(r ./ s)
    backsolve!(fac, rhs)
    frac, ex = frexp(s)
    d .= T.(ldexp2.(rhs .* frac, ex - fac.scale_exponent))
    return d
end
