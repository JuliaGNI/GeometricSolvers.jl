# Can a run of full-length secant steps of `Newton(; reuse = 3, update = BroydenUpdate(2))`
# end a solve with `SUCCESS` and `contracted == true` away from a root? Run from the repository
# root with `julia --startup-file=no --project=. scripts/broyden_rate_test.jl`.
#
# F(x) = g(x - X) with g(t) = 1 - t + θ t² up to a kink t_k, and a nearly flat line
# g(t_k) - σ (t - t_k) beyond it. The root is at t_k + g(t_k)/σ, far from the iterates.
# Newton from t = 0 lands at t = 1 exactly. For each method the kink is placed at that method's
# own second iterate on the parabola alone (a solve with max_iterations = 2), so every method
# meets the same geometry: two steps on the smooth part, then a nearly flat residual. The columns
# after `|` repeat the solve under `Options(T; rate_test = false)`.
using GeometricSolvers
using GeometricSolvers: BroydenUpdate, Newton, NonlinearProblem, Options, Static,
                        Backtracking,
                        solve
using Printf

const METHODS = (
    ("Broyden reuse 3, Static", () -> Newton(; reuse = 3, update = BroydenUpdate(2))),
    ("Broyden reuse 3, Backtracking",
        () -> Newton(; reuse = 3, update = BroydenUpdate(2), globalization = Backtracking())),
    ("chord reuse 3, Static", () -> Newton(; reuse = 3)),
    ("chord reuse 3, Backtracking",
        () -> Newton(; reuse = 3, globalization = Backtracking())),
    ("Newton, Static", () -> Newton()),
    ("Newton, Backtracking", () -> Newton(; globalization = Backtracking())))

function probe(::Type{T}, X, θ, σ) where {T}
    X = T(X)
    θ = T(θ)
    σ = T(σ)
    gpar(t) = one(T) - t + θ * t^2
    rows = []
    for (name, mk) in METHODS
        x2, _ = solve(NonlinearProblem((x, p) -> gpar(x - X)), X, mk();
            options = Options(T; max_iterations = 2))
        tk = x2 - X
        g(t) = t <= tk ? gpar(t) : gpar(tk) - σ * (t - tk)
        root = X + tk + gpar(tk) / σ
        x, st = solve(NonlinearProblem((x, p) -> g(x - X)), X, mk(); options = Options(T))
        x_off, st_off = solve(NonlinearProblem((x, p) -> g(x - X)), X, mk();
            options = Options(T; rate_test = false))
        push!(rows,
            (name, st.code, st.iterations, st.contracted, Float64(st.fnorm),
                Float64(abs(x - root)), Float64(st.rate), st_off.code, st_off.iterations,
                Float64(abs(x_off - root))))
    end
    return rows
end

for (T, X, θ, σ) in ((Float64, 1e6, 2e-4, 5e-3), (Float64, 1e8, 1e-3, 5e-3),
    (Float32, 1e3, 2.5e-2, 5e-2), (Float32, 3e3, 3e-2, 5e-2))
    rows = probe(T, X, θ, σ)
    println("T = $T, X = $X, θ = $θ, σ = $σ; f_reltol·|F(x₀)| = ", sqrt(eps(T)))
    for r in rows
        @printf("  %-32s %-10s iters=%2d contracted=%-5s fnorm=%.3e |x - root|=%.3e rate=%.3e | %-10s iters=%2d |x - root|=%.3e\n",
            r[1], string(r[2]), r[3], string(r[4]), r[5], r[6], r[7], string(r[8]), r[9],
            r[10])
    end
end
