# On F(x) = A * x .+ 0.5 .* (x .^ 2) at N = 16, f_abstol = 1e-12, f_reltol = 0.0,
# κ(J) ∈ (1e2, 1e4, 1e5, 1e6), compare pure-Float64 Newton, constant η_J = √eps(Float64),
# Kelley's η_J = 1e-6, and Eisenstat-Walker Choice 2 with √eps(Float64) floor across
# outer Newton iterations, inner refinement passes, and final ‖F‖_∞.

using GeometricSolvers
using LinearAlgebra: Diagonal, lu, ldiv!, mul!, norm
using Printf: @printf
using StaticArrays: SMatrix, SVector

const GS = GeometricSolvers

function make_d2_matrix(N::Int, kappa::Float64)
    s = [kappa^((i - 1) / (N - 1)) for i in 1:N]
    Q = Matrix{Float64}(undef, N, N)
    for j in 1:N, i in 1:N

        Q[i, j] = sqrt(2.0 / (N + 1)) * sin(i * j * pi / (N + 1))
    end
    return Q * Diagonal(1.0 ./ s) * transpose(Q)
end

function make_d2_problem(A::Matrix{Float64})
    function f!(F, x, _)
        mul!(F, A, x)
        @. F = F + 0.5 * (x^2)
        return F
    end
    function j!(J, x, _)
        copyto!(J, A)
        @inbounds for i in axes(J, 1)
            J[i, i] += x[i]
        end
        return J
    end
    return NonlinearProblem(f!, j!)
end

function simulate_ew_choice2(A::Matrix{Float64}, x0::Vector{Float64}; atol::Float64 = 1e-12)
    N = length(x0)
    x = copy(x0)
    F = A * x .+ 0.5 .* (x .^ 2)
    fnorm = maximum(abs, F)
    fnorm_prev = fnorm
    eta_ew = 0.5
    eta_floor = sqrt(eps(Float64))
    outer_iters = 0
    inner_passes = 0
    while fnorm > atol && outer_iters < 50
        outer_iters += 1
        if outer_iters > 1
            eta_raw = 0.9 * (fnorm / fnorm_prev)^1.5
            eta_ew = max(eta_floor, min(0.5, eta_raw))
        end
        J = copy(A)
        @inbounds for i in 1:N
            J[i, i] += x[i]
        end
        fac32 = lu(Float32.(J))
        d = -Float64.(fac32 \ Float32.(F))
        r = F .+ J * d
        r_prev = maximum(abs, r)
        for _ in 1:5
            r_prev <= eta_ew * fnorm && break
            inner_passes += 1
            δ = Float64.(fac32 \ Float32.(r))
            d_try = d .- δ
            r_try = F .+ J * d_try
            r_curr = maximum(abs, r_try)
            r_curr < r_prev || break
            d = d_try
            r = r_try
            r_prev = r_curr
        end
        x .+= d
        fnorm_prev = fnorm
        F .= A * x .+ 0.5 .* (x .^ 2)
        fnorm = maximum(abs, F)
    end
    return outer_iters, inner_passes, fnorm
end

function count_refinement_passes(A::Matrix{Float64}, x0::Vector{Float64}, eta::Float64;
        atol::Float64 = 1e-12, maxiter::Int = 5)
    N = length(x0)
    x = copy(x0)
    F = A * x .+ 0.5 .* (x .^ 2)
    fnorm = maximum(abs, F)
    eta_eff = max(eta, sqrt(eps(Float64)))
    outer_iters = 0
    inner_passes = 0
    while fnorm > atol && outer_iters < 50
        outer_iters += 1
        J = copy(A)
        @inbounds for i in 1:N
            J[i, i] += x[i]
        end
        fac32 = lu(Float32.(J))
        d = -Float64.(fac32 \ Float32.(F))
        r = F .+ J * d
        r_prev = maximum(abs, r)
        for _ in 1:maxiter
            r_prev <= eta_eff * fnorm && break
            inner_passes += 1
            δ = Float64.(fac32 \ Float32.(r))
            d_try = d .- δ
            r_try = F .+ J * d_try
            r_curr = maximum(abs, r_try)
            r_curr < r_prev || break
            d = d_try
            r = r_try
            r_prev = r_curr
        end
        x .+= d
        F .= A * x .+ 0.5 .* (x .^ 2)
        fnorm = maximum(abs, F)
    end
    return outer_iters, inner_passes, fnorm
end

function run_d2_sweep()
    N = 16
    atol = 1e-12
    kappas = (1e2, 1e4, 1e5, 1e6)
    x0 = fill(0.1, N)
    opts = Options(Float64; f_abstol = atol, f_reltol = 0.0)

    println("Mixed-precision refinement η_J sweep: F(x) = A * x .+ 0.5 .* (x .^ 2), N = $N, atol = $atol")
    println("----------------------------------------------------------------------------------------")
    @printf("%-8s | %-22s | %-10s | %-12s | %-10s | %-12s\n",
        "κ(J)", "strategy", "outer_iter", "inner_passes", "promoted", "final ‖F‖_∞")
    println("----------------------------------------------------------------------------------------")

    m_f64 = Newton(; linsolve = LUFactorization(), refinement = NoRefinement())
    m_sqrt = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement(sqrt(eps(Float64)), 5))
    m_kelley = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement(1e-6, 5))

    for kappa in kappas
        A = make_d2_matrix(N, kappa)
        prob = make_d2_problem(A)

        _, st_f64 = solve(prob, copy(x0), m_f64; options = opts)
        _, st_sqrt = solve(prob, copy(x0), m_sqrt; options = opts)
        _, st_kelley = solve(prob, copy(x0), m_kelley; options = opts)

        _, passes_sqrt, _ = count_refinement_passes(A, x0, sqrt(eps(Float64)); atol = atol)
        _, passes_kelley, _ = count_refinement_passes(A, x0, 1e-6; atol = atol)
        ew_iters, ew_passes, ew_fnorm = simulate_ew_choice2(A, x0; atol = atol)

        @printf("%-8.0e | %-22s | %10d | %12d | %-10s | %12.2e\n",
            kappa, "Pure Float64", st_f64.iterations, 0, string(st_f64.promoted),
            st_f64.fnorm)
        @printf("%-8.0e | %-22s | %10d | %12d | %-10s | %12.2e\n",
            kappa, "η_J = √eps(Float64)", st_sqrt.iterations,
            passes_sqrt, string(st_sqrt.promoted), st_sqrt.fnorm)
        @printf("%-8.0e | %-22s | %10d | %12d | %-10s | %12.2e\n",
            kappa, "Kelley η_J = 1e-6", st_kelley.iterations, passes_kelley,
            string(st_kelley.promoted), st_kelley.fnorm)
        @printf("%-8.0e | %-22s | %10d | %12d | %-10s | %12.2e\n",
            kappa, "EW Choice 2 + floor", ew_iters, ew_passes, "false", ew_fnorm)
        println("----------------------------------------------------------------------------------------")
    end
    ref = Refinement()
    println("Selected default: η_J = √eps(real(T)) = $(ref.η) (maxiter = $(ref.maxiter))")
end

run_d2_sweep()
