module GeometricSolversIntervalArithmeticExt

using ForwardDiff: ForwardDiff
using GeometricSolvers: GeometricSolvers, NonlinearProblem, buffer, residual!!
using IntervalArithmetic: Interval, dac, decoration, entireinterval, interval, isbounded,
                          isguaranteed, isinterior, mag
using LinearAlgebra: issuccess, lu
using StaticArrays: StaticVector

# The number of boxes the Krawczyk test tries before it gives up.
const MAX_BOXES = 10

# The iterates of a certificate: R1 (a real `Number`, a `StaticVector`) and R3 (a host `Vector`),
# each in a floating-point element type, which is the element type of the intervals.
const Iterate{T} = Union{T, StaticVector{<:Any, T}, Vector{T}} where {T <: AbstractFloat}

# The Jacobian of `prob` at `x`, in the storage of `x`: the analytic `J` where the problem has one,
# else `ForwardDiff` on `F`. `x` holds floats for the inverse `Y`, and intervals for `J(X)`.
jacobian(prob::NonlinearProblem, x, p) = jacobian(prob.J, prob, x, p)
jacobian(J, prob, x::Union{Number, StaticVector}, p) = J(x, p)
jacobian(J, prob, x::Vector, p) = (A = similar(x, length(x), length(x)); J(A, x, p); A)
jacobian(::Nothing, prob, x::Number, p) = ForwardDiff.derivative(ξ -> prob.F(ξ, p), x)
function jacobian(::Nothing, prob, x::StaticVector, p)
    return ForwardDiff.jacobian(ξ -> prob.F(ξ, p), x)
end
function jacobian(::Nothing, prob, x::Vector, p)
    return ForwardDiff.jacobian((y, ξ) -> prob.F(y, ξ, p), similar(x), x)
end

# The inverse of a point Jacobian, or `nothing` where its LU factorisation fails, because the
# inverse of a failed LAPACK factorisation throws. A zero scalar Jacobian gives an infinite
# inverse, which the caller refuses with every other inverse that is not finite.
inverse(A::Number) = inv(A)
function inverse(A::AbstractMatrix)
    fact = lu(A; check = false)
    return issuccess(fact) ? inv(fact) : nothing
end

# Whether every entry of `a`, a scalar or an array, is bounded and, for an interval, comes from a
# function that is defined and continuous on its arguments: the decoration `dac` or `com`.
# Krawczyk's conclusion needs `F` continuous and `J` its continuous Jacobian on the box; a lower
# decoration marks a function undefined on part of its arguments, where `F(x)` or `J(X)` can be
# empty. An unbounded entry makes `K(X)` unbounded, which no bounded box holds, or
# IntervalArithmetic's product of a float matrix with it forms NaN bounds, with a warning. A number
# that is not an interval, such as a constant analytic Jacobian, needs only to be finite.
usable(a::Interval) = decoration(a) >= dac && isbounded(a)
usable(a::Real) = isfinite(a)
usable(a::AbstractArray) = all(usable, a)

function GeometricSolvers.certify(prob::NonlinearProblem, x::Iterate{T},
        p = nothing) where {T <: AbstractFloat}
    unknown = map(_ -> entireinterval(T), x)
    refused = (; certified = false, enclosure = unknown, guaranteed = false)
    all(isfinite, x) || return refused

    Y = inverse(jacobian(prob, x, p))
    (Y === nothing || !all(isfinite, Y)) && return refused

    x̃ = map(interval, x)
    fx = residual!!(buffer(x̃), prob, x̃, p)
    usable(fx) || return refused
    r = eps.(x)
    for _ in 1:MAX_BOXES
        all(isfinite, r) || return refused
        X = x̃ .+ map(ρ -> interval(-ρ, ρ), r)
        JX = jacobian(prob, X, p)
        usable(JX) || return refused
        M = Y * JX
        K = x̃ - Y * fx + (one(M) - M) * (X - x̃)
        if all(isinterior.(K, X))
            guaranteed = all(isguaranteed, fx) && all(isguaranteed, JX)
            return (; certified = true, enclosure = convert(typeof(unknown), K), guaranteed)
        end
        r = 2 .* map(mag, K - x̃) .+ eps.(x)
    end
    return refused
end

end
