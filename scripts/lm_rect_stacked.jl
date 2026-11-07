# The stacked `LevenbergMarquardt` step of an `m × n` Jacobian (`m ≥ n`), against the damped normal
# equations `(Jᵀ J + μ D²) δ = -Jᵀ F` solved in Float64: the R1 step of `lm_qr_prepare` on an
# `SMatrix` and the R3 step on an `Array`, over random `J`, `F` and `μ`, both scalings, and both
# precisions. The error is the largest `‖δ - δ_ref‖ / ‖δ_ref‖` of a configuration, in units of
# `eps(T)`; a damped system with `μ ≥ 1e-4` on a Gaussian `J` is well conditioned, so a few hundred
# `eps` is the expected scale and a wrong elimination of the rows past `n` is `O(1 / eps)`.
#
#     julia --startup-file=no --project=. scripts/lm_rect_stacked.jl
using GeometricSolvers, LinearAlgebra, Printf, Random, StaticArrays

const G = GeometricSolvers

function damping_diagonal(J, scaling)
    scaling === :none && return ones(eltype(J), size(J, 2))
    c = [norm(view(J, :, j)) for j in 1:size(J, 2)]
    return max.(c, sqrt(eps(eltype(J))) * maximum(c))
end

function relative_errors(rng, ::Type{T}, m, n, scaling) where {T}
    J = randn(rng, T, m, n)
    F = randn(rng, T, m)
    μ = T(10.0^rand(rng, -4:1))
    Jd = Float64.(J)
    D2 = Diagonal(Float64.(damping_diagonal(J, scaling)) .^ 2)
    ref = -((transpose(Jd) * Jd + μ * D2) \ (transpose(Jd) * Float64.(F)))
    p1 = G.lm_qr_prepare(QRFactorization(), nothing, nothing, SMatrix{m, n, T}(J),
        SVector{m, T}(F), Val(scaling))
    δ1, ok1 = G.lm_trial(p1, μ)
    ws = G.init_globalization(
        LevenbergMarquardt(; scaling), QRFactorization(), zeros(T, n),
        zeros(T, m))
    δ3 = zeros(T, n)
    p3 = G.lm_qr_prepare(QRFactorization(), δ3, ws, copy(J), copy(F), Val(scaling))
    _, ok3 = G.lm_trial(p3, μ)
    (ok1 && ok3) || error("a trial failed at $((T, m, n, scaling))")
    e(δ) = norm(Float64.(δ) - ref) / norm(ref) / eps(T)
    return e(δ1), e(δ3)
end

rng = Xoshiro(7)
for T in (Float32, Float64), scaling in (:none, :diagonal)

    worst1 = worst3 = 0.0
    for (m, n) in ((3, 2), (5, 3), (8, 4), (12, 6), (20, 3)), _ in 1:20

        e1, e3 = relative_errors(rng, T, m, n, scaling)
        worst1 = max(worst1, e1)
        worst3 = max(worst3, e3)
    end
    @printf("%-8s %-9s R1 %8.1f eps   R3 %8.1f eps\n", T, scaling, worst1, worst3)
end
