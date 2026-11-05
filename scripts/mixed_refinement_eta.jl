# Sweep η_J ∈ (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, √eps(Float64)) across conditioned nonlinear
# problems in R1 and R3 with Float32 LU factorisation on Float64 state, comparing Newton
# iteration counts, refinement JVP evaluations, and promotion rates against pure Float64 LU.

using GeometricSolvers
using LinearAlgebra: Diagonal, norm
using Printf: @printf
using StaticArrays: SMatrix, SVector

const GS = GeometricSolvers

function make_problem_r1(eps_cond::Float64)
    A = SMatrix{2, 2, Float64, 4}(1.0, 1.0, 1.0, 1.0 + eps_cond)
    u_star = A * SVector{2, Float64}(1.0, 2.0)
    b = SVector{2, Float64}(
        u_star[1] + 0.1 * u_star[1]^2, u_star[2] - 0.2 * u_star[2]^2)
    f(x, _) = begin
        u = A * x
        SVector{2, Float64}(u[1] + 0.1 * u[1]^2 - b[1], u[2] - 0.2 * u[2]^2 - b[2])
    end
    j(x, _) = begin
        u = A * x
        d = SVector{2, Float64}(1.0 + 0.2 * u[1], 1.0 - 0.4 * u[2])
        Diagonal(d) * A
    end
    return NonlinearProblem(f, j)
end

function make_problem_r3(eps_cond::Float64, jvp_counter::Base.RefValue{Int})
    A = [1.0 1.0; 1.0 (1.0 + eps_cond)]
    u_star = A * [1.0, 2.0]
    b = [u_star[1] + 0.1 * u_star[1]^2, u_star[2] - 0.2 * u_star[2]^2]
    function f!(F, x, _)
        jvp_counter[] += 1
        u1 = A[1, 1] * x[1] + A[1, 2] * x[2]
        u2 = A[2, 1] * x[1] + A[2, 2] * x[2]
        F[1] = u1 + 0.1 * u1^2 - b[1]
        F[2] = u2 - 0.2 * u2^2 - b[2]
        return F
    end
    return NonlinearProblem(f!)
end

function run_sweep()
    etas = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, sqrt(eps(Float64)))
    eps_list = (1e-2, 1e-4, 1e-5, 2e-6)
    starts = (
        SVector{2, Float64}(2.0, 3.0),
        SVector{2, Float64}(-1.5, 4.0),
        SVector{2, Float64}(3.5, -1.0)
    )

    println("Mixed-precision Newton iterative refinement sweep (eta_J / η):")
    println("--------------------------------------------------------------------------")
    @printf("%-10s | %-8s | %-10s | %-10s | %-10s | %-10s\n",
        "eta_J (η)", "regime", "converged", "promoted", "total_iter", "total_eval")
    println("--------------------------------------------------------------------------")

    for η in etas
        conv_r1 = 0
        prom_r1 = 0
        iters_r1 = 0
        conv_r3 = 0
        prom_r3 = 0
        iters_r3 = 0
        evals_r3 = 0
        total_cases = length(eps_list) * length(starts)

        m = Newton(; linsolve = LUFactorization(Float32), refinement = Refinement(5; η = η))

        for eps_cond in eps_list
            prob_r1 = make_problem_r1(eps_cond)
            counter = Ref(0)
            prob_r3 = make_problem_r3(eps_cond, counter)
            for x0 in starts
                _, st1 = solve(prob_r1, x0, m)
                conv_r1 += (st1.code == SUCCESS)
                prom_r1 += st1.promoted
                iters_r1 += Int(st1.iterations)

                counter[] = 0
                _, st3 = solve(prob_r3, collect(x0), m)
                conv_r3 += (st3.code == SUCCESS)
                prom_r3 += st3.promoted
                iters_r3 += Int(st3.iterations)
                evals_r3 += counter[]
            end
        end

        @printf("%-10.0e | %-8s | %4d/%-5d | %4d/%-5d | %10d | %10s\n",
            η, "R1", conv_r1, total_cases, prom_r1, total_cases, iters_r1, "-")
        @printf("%-10.0e | %-8s | %4d/%-5d | %4d/%-5d | %10d | %10d\n",
            η, "R3", conv_r3, total_cases, prom_r3, total_cases, iters_r3, evals_r3)
    end
    println("--------------------------------------------------------------------------")
    ref = Refinement()
    println("Selected default: eta_J = √eps(real(T)) = $(ref.η) (maxiter = $(ref.maxiter))")
end

run_sweep()
