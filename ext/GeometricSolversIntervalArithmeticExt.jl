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
# such as a jump of `F` in `F(X)`, or a kink of `F` in `J(X)`. A bound that overflows keeps the
# decoration `dac`, so an unbounded `F(X)` or `J(X)` is refused by its bound.
usable(a::Interval) = decoration(a) >= dac && isbounded(a)
usable(a::AbstractArray) = all(usable, a)

# The product of an interval matrix with an interval vector or matrix, as sums of interval
# products. IntervalArithmetic's default product of `Matrix` arguments works on midpoints and
# radii, which overflow on bounded intervals near `floatmax` into NaN bounds, with a warning; sums
# of products overflow into infinite bounds only. A number and a `StaticArray` take their own
# product, which forms no midpoints.
mul(A, B) = A * B
function mul(A::Matrix{<:Interval}, b::Vector{<:Interval})
    return [sum(A[i, k] * b[k] for k in axes(A, 2); init = zero(eltype(b)))
            for i in axes(A, 1)]
end
function mul(A::Matrix{<:Interval}, B::Matrix{<:Interval})
    return [sum(A[i, k] * B[k, j] for k in axes(A, 2); init = zero(eltype(B)))
            for i in axes(A, 1), j in axes(B, 2)]
end

function GeometricSolvers.certify(prob::NonlinearProblem, x::Iterate{T},
        p = nothing) where {T <: AbstractFloat}
    unknown = map(_ -> entireinterval(T), x)
    refused = (; certified = false, enclosure = unknown, guaranteed = false)
    all(isfinite, x) || return refused

    Y = inverse(jacobian(prob, x, p))
    (Y === nothing || !all(isfinite, Y)) && return refused

    x̃ = map(interval, x)
    fx = residual!!(buffer(x̃), prob, x̃, p)
    Yi = map(interval, Y)
    c = x̃ - mul(Yi, fx)

    r = eps.(x)
    for _ in 1:MAX_BOXES
        X = x̃ .+ map(ρ -> interval(-ρ, ρ), r)
        all(isbounded, X) || return refused
        # Every box holds `x`, so `F(X)` holds `F(x)`, decorated no higher: this test refuses
        # also an `F(x)` that is unbounded or decorated below `dac`.
        usable(residual!!(buffer(X), prob, X, p)) || return refused
        JX = jacobian(prob, X, p)
        usable(JX) || return refused
        M = mul(Yi, JX)
        R = one(M) - M
        K = c + mul(R, X - x̃)
        if all(isinterior.(K, X))
            guaranteed = all(isguaranteed, fx) && all(isguaranteed, JX)
            return (; certified = true, enclosure = convert(typeof(unknown), K), guaranteed)
        end
        r = 2 .* map(mag, K - x̃) .+ eps.(x)
    end
    return refused
end

end
