# The `T ≠ TF` method of `ldiv!!` for dense factors: scales a finite, nonzero right-hand side by
# `s = maximum(abs, r)` before converting to the factor element type `F`, solves in `fac.rhs`, and
# converts and unscales the step into `d`.

function ldiv!!(d::AbstractVector{T}, fac::Union{DenseLU, DenseQR, DenseSVD},
        r::AbstractVector{T}) where {T <: Number}
    s = maximum(abs, r)
    iszero(s) && return fill!(d, zero(T))
    rhs = fac.rhs
    F = eltype(rhs)
    @inbounds for i in eachindex(rhs, r)
        rhs[i] = F(r[i] / s)
    end
    ldiv!!(rhs, fac, rhs)
    @inbounds for i in eachindex(d, rhs)
        d[i] = T(rhs[i]) * s
    end
    return d
end
