# R2 batched throughput benchmark:
#   F(x, b) = A * x + sin.(x) / 10 - b
# for diagonally dominant A in Float32 across batch sizes and system dimensions n.
#
# Usage:
#   julia --startup-file=no --threads=auto --project=. scripts/r2_throughput.jl [backend] [N_sys] [n]
#
# `backend` is `cpu` (the default), `metal` or `cuda`. The package depends on neither Metal.jl nor
# CUDA.jl: install the one you select in an environment on the load path, such as the default
# `@v1.x` environment, or `import Metal` / `import CUDA` fails before the benchmark starts.

using GeometricSolvers
using GeometricSolvers: Adapt, Options, SolverStatus, ToReal, init_state, isconverged,
                        solve!!
using KernelAbstractions
using LinearAlgebra
using Printf
using Random
using StaticArrays

function make_A(::Val{N}, ::Type{T}; seed::Int = 42) where {N, T}
    rng = Xoshiro(seed)
    M = randn(rng, T, N, N) ./ T(N)
    return SMatrix{N, N, T, N * N}(M + T(2) * I)
end

function make_batch(::Val{N}, ::Type{T}, n_sys::Int; seed::Int = 7) where {N, T}
    rng = Xoshiro(seed)
    X_star = [SVector{N, T}(randn(rng, T, N)) for _ in 1:n_sys]
    X0 = [x .+ SVector{N, T}(T(0.2) .* randn(rng, T, N)) for x in X_star]
    return X_star, X0
end

struct ThroughputResidual{N, T, L}
    A::SMatrix{N, N, T, L}
end

@inline function (f::ThroughputResidual)(x::SVector{N, T}, b::SVector{N, T}) where {N, T}
    return f.A * x + sin.(x) ./ T(10) - b
end

struct ThroughputJacobian{N, T, L}
    A::SMatrix{N, N, T, L}
end

@inline function (j::ThroughputJacobian{N, T})(
        x::SVector{N, T}, ::SVector{N, T}) where {N, T}
    return j.A + Diagonal(cos.(x) ./ T(10))
end

function make_problem(::Val{N}, ::Type{T}; seed::Int = 42) where {N, T}
    A = make_A(Val(N), T; seed)
    f = ThroughputResidual(A)
    j = ThroughputJacobian(A)
    return NonlinearProblem(f, j)
end

function make_rhs(prob::NonlinearProblem, X_star::Vector{SVector{N, T}}) where {N, T}
    return [prob.F(x, zero(SVector{N, T})) for x in X_star]
end

function cpu_threaded_loop!(
        X::Vector{SVector{N, T}},
        S::Vector{SolverStatus{R}},
        prob::NonlinearProblem,
        method,
        B::Vector{SVector{N, T}},
        opt::Options{R}
) where {N, T, R}
    m = Adapt.adapt(ToReal{R}(), method)
    state0 = init_state(m, prob, zero(SVector{N, T}), nothing)
    Threads.@threads :static for i in eachindex(X)
        @inbounds begin
            x, st, _ = solve!!(state0, prob, X[i], m, B[i], opt)
            X[i] = x
            S[i] = st
        end
    end
    return S
end

function _best_time_s(f!; reps::Int = 3)
    f!() # warmup
    best = Inf
    for _ in 1:reps
        t = @elapsed f!()
        best = min(best, t)
    end
    return best
end

function measure_row(
        ::Val{N},
        n_sys::Int,
        backend_name::String,
        array_ctor;
        reps::Int = 3
) where {N}
    T = Float32
    R = Float32
    prob = make_problem(Val(N), T)
    X_star, X0_host = make_batch(Val(N), T, n_sys)
    B_host = make_rhs(prob, X_star)
    opt = Options(R; f_abstol = 1.0f-5, f_reltol = 1.0f-5, max_iterations = 20)

    # 1. Threaded CPU R1 loop
    X_cpu = similar(X0_host)
    S_cpu = Vector{SolverStatus{R}}(undef, n_sys)
    t_cpu = _best_time_s(; reps) do
        copyto!(X_cpu, X0_host)
        cpu_threaded_loop!(X_cpu, S_cpu, prob, Newton(), B_host, opt)
    end
    rate_cpu = n_sys / t_cpu

    # 2. R2 solve_batch! with Newton()
    X0_dev = array_ctor(X0_host)
    X_dev = similar(X0_dev)
    S_dev = array_ctor(Vector{SolverStatus{R}}(undef, n_sys))
    P_dev = PerSystem(array_ctor(B_host))
    t_r2 = _best_time_s(; reps) do
        copyto!(X_dev, X0_dev)
        solve_batch!(X_dev, S_dev, prob, Newton(), P_dev; options = opt)
    end
    rate_r2 = n_sys / t_r2

    # 3. R2 solve_batch! with Newton(; reuse = 4)
    t_reuse = _best_time_s(; reps) do
        copyto!(X_dev, X0_dev)
        solve_batch!(X_dev, S_dev, prob, Newton(; reuse = 4), P_dev; options = opt)
    end
    rate_reuse = n_sys / t_reuse

    # 4. R2 solve! with Newton(; globalization = Backtracking()): the row that runs a line
    # search in the kernel
    t_ls = _best_time_s(; reps) do
        copyto!(X_dev, X0_dev)
        solve!(X_dev, S_dev, prob, Newton(; globalization = Backtracking()), P_dev;
            options = opt)
    end
    rate_ls = n_sys / t_ls

    # Verify convergence on the last run, the Backtracking row's
    S_got = Array(S_dev)
    n_conv = count(isconverged, S_got)

    @printf("backend=%-6s N_sys=%-8d n=%-2d | R2 Newton: %9.2f Msys/s | R2 reuse=4: %9.2f Msys/s | R2 Backtracking: %9.2f Msys/s | CPU-thr R1: %9.2f Msys/s | speedup=%5.2fx | Backtracking %d/%d conv\n",
        backend_name,
        n_sys,
        N,
        rate_r2 / 1e6,
        rate_reuse / 1e6,
        rate_ls / 1e6,
        rate_cpu / 1e6,
        rate_r2 / rate_cpu,
        n_conv,
        n_sys,)
    return (
        n = N,
        n_sys = n_sys,
        rate_r2 = rate_r2,
        rate_reuse = rate_reuse,
        rate_ls = rate_ls,
        rate_cpu = rate_cpu
    )
end

function resolve_backend(name::AbstractString)
    lname = lowercase(strip(name))
    if lname == "cpu"
        return "cpu", Array
    elseif lname == "metal"
        @eval import Metal
        if !Base.invokelatest(Metal.functional)
            error("Metal backend requested, but Metal.functional() is false")
        end
        return "metal", Base.invokelatest(() -> Metal.MtlArray)
    elseif lname == "cuda"
        @eval import CUDA
        if !Base.invokelatest(CUDA.functional)
            error("CUDA backend requested, but CUDA.functional() is false")
        end
        return "cuda", Base.invokelatest(() -> CUDA.CuArray)
    else
        error("Unknown backend $(repr(name)); expected cpu, metal, or cuda")
    end
end

function main(args = ARGS)
    backend_arg = length(args) >= 1 ? args[1] : "cpu"
    nsys_list = length(args) >= 2 ? [parse(Int, args[2])] : [1_000, 100_000]
    n_list = length(args) >= 3 ? [parse(Int, args[3])] : [2, 4, 8, 12, 16]

    bname, ctor = resolve_backend(backend_arg)
    println("R2 batched throughput benchmark (threads=$(Threads.nthreads()), backend=$bname)")

    for n_sys in nsys_list
        rows = NamedTuple[]
        for n in n_list
            row = measure_row(Val(n), n_sys, bname, ctor)
            push!(rows, row)
        end
        # Report R4 crossover n where R2 Newton throughput drops below CPU threaded R1 loop
        crossover = findfirst(r -> r.rate_r2 < r.rate_cpu, rows)
        if crossover === nothing
            println("R4 crossover (N_sys=$n_sys): none in n=$(n_list) (R2 >= CPU-thr across all tested n)")
        else
            println("R4 crossover (N_sys=$n_sys): n=$(rows[crossover].n)")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__) || !isempty(ARGS)
    main()
end
