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

# The Jacobian of `F` at `x` by `ForwardDiff`, in the storage of `x`. `x` holds floats for the
# inverse `Y`, and intervals for `J(X)`. The `J` of the problem is not read: the test needs an
# enclosure of the Jacobian of `F` itself on the box, which a given `J` does not guarantee.
jacobian(prob, x::Number, p) = ForwardDiff.derivative(ξ -> prob.F(ξ, p), x)
jacobian(prob, x::StaticVector, p) = ForwardDiff.jacobian(ξ -> prob.F(ξ, p), x)
function jacobian(prob, x::Vector, p)
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

# Whether every interval of `a`, an interval or an array of them, is bounded and comes from a
# function that is defined and continuous on its arguments: the decoration `dac` or `com`.
# Krawczyk's conclusion needs `F` continuous and `J` its continuous Jacobian on the box; a lower
# decoration marks a function that can be undefined or not continuous on part of its arguments,
# such as a jump of `F` in `F(X)`, or a kink of `F` in `J(X)`. An unbounded entry makes `K(X)`
# unbounded, which no bounded box holds, or IntervalArithmetic's product of a float matrix with it
# forms NaN bounds, with a warning.
usable(a::Interval) = decoration(a) >= dac && isbounded(a)
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
    r = eps.(x)
    for _ in 1:MAX_BOXES
        all(isfinite, r) || return refused
        X = x̃ .+ map(ρ -> interval(-ρ, ρ), r)
        usable(residual!!(buffer(X), prob, X, p)) || return refused
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
