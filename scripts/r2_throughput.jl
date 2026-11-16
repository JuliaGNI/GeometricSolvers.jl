# R2 batched throughput benchmark:
#   F(x, b) = A * x + sin.(x) / 10 - b
# for diagonally dominant A in Float32 across batch sizes and system dimensions n.
#
# Usage:
#   julia --startup-file=no --threads=auto --project=. scripts/r2_throughput.jl [device] [N_sys,…] [n,…]
#
# `N_sys,…` is a comma-separated list of batch sizes, `10³, 10⁴, …, 10⁷` by default, and `n,…` one of
# system dimensions, `2,4,8,12,16` by default.
# `device` is `cpu` (the default), `jlarray`, `metal` or `cuda`. The package depends on neither
# Metal.jl nor CUDA.jl: install the one you select in an environment on the load path, such as the
# default `@v1.x` environment. A device that is not present prints one line and no rows.
#
# The right-hand sides `b` reach the kernel in two layouts. The array of structures (AoS) is one
# `SVector` per system, `PerSystem(Vector{SVector{n}})`. The structure of arrays (SoA) is one array
# per component, `PerSystem(SoAColumns)`, whose entry `i` gathers component `j` of system `i` from
# array `j`. Both solve the same systems from the same starts, and the script stops with an error
# where their roots differ by more than `ROOT_RTOL` relative to the AoS root.
#
# Rows, each in Msys/s: R2 `Newton()` with AoS and SoA right-hand sides, R2 `Newton(; reuse = 4)`,
# R2 `Newton(; globalization = Backtracking())`, and two CPU baselines of the R1 `solve!!` loop over
# the systems: `Threads.@threads :static` over the Julia threads, and one thread.

using GeometricSolvers
using GeometricSolvers: Adapt, Options, SolverStatus, ToReal, init_state, isconverged,
                        solve!!
using KernelAbstractions
using LinearAlgebra
using Printf
using Random
using StaticArrays

include(joinpath(@__DIR__, "hardware", "common.jl"))

# The largest relative difference between the AoS and the SoA root of a system that the check
# accepts. Both layouts run the same arithmetic on the same values, so the roots agree to a few
# ulps (`cpu 1000,2000 2` gives a difference of 0.0); 8 ulps leave room for FMA contraction and
# reduction order.
const ROOT_RTOL = 8 * eps(Float32)

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

# The right-hand sides as a structure of arrays: `cols[j][i]` is component `j` of system `i`, and
# entry `i` is the `SVector` of system `i`, which is what `PerSystem` hands to the residual.
struct SoAColumns{N, V <: AbstractVector}
    cols::NTuple{N, V}
end

Adapt.@adapt_structure SoAColumns

Base.length(s::SoAColumns) = length(s.cols[1])

Base.@propagate_inbounds function Base.getindex(s::SoAColumns{N}, i::Integer) where {N}
    return SVector{N}(ntuple(j -> s.cols[j][i], Val(N)))
end

function soa_columns(B::Vector{SVector{N, T}}, array_ctor) where {N, T}
    return SoAColumns(ntuple(j -> array_ctor([b[j] for b in B]), Val(N)))
end

# The largest difference between the roots `X_aos` and `X_soa` of one system, relative to the norm
# of the AoS root.
function max_root_difference(X_aos::Vector{<:SVector}, X_soa::Vector{<:SVector})
    return maximum(i -> norm(X_aos[i] - X_soa[i]) / norm(X_aos[i]), eachindex(X_aos, X_soa))
end

# The R1 `solve!!` loop over the systems, on the Julia threads or on one thread.
function cpu_loop!(
        X::Vector{SVector{N, T}},
        S::Vector{SolverStatus{R}},
        prob::NonlinearProblem,
        method,
        B::Vector{SVector{N, T}},
        opt::Options{R};
        threaded::Bool
) where {N, T, R}
    m = Adapt.adapt(ToReal{R}(), method)
    state0 = init_state(m, prob, zero(SVector{N, T}), nothing)
    if threaded
        Threads.@threads :static for i in eachindex(X)
            @inbounds begin
                x, st, _ = solve!!(state0, prob, X[i], m, B[i], opt)
                X[i] = x
                S[i] = st
            end
        end
    else
        for i in eachindex(X)
            @inbounds begin
                x, st, _ = solve!!(state0, prob, X[i], m, B[i], opt)
                X[i] = x
                S[i] = st
            end
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
        reps::Int = 3,
        io::IO = stdout
) where {N}
    T = Float32
    R = Float32
    prob = make_problem(Val(N), T)
    X_star, X0_host = make_batch(Val(N), T, n_sys)
    B_host = make_rhs(prob, X_star)
    opt = Options(R; f_abstol = 1.0f-5, f_reltol = 1.0f-5, max_iterations = 20)

    # 1. CPU R1 loops: threaded, and on one thread
    X_cpu = similar(X0_host)
    S_cpu = Vector{SolverStatus{R}}(undef, n_sys)
    t_cpu = _best_time_s(; reps) do
        copyto!(X_cpu, X0_host)
        cpu_loop!(X_cpu, S_cpu, prob, Newton(), B_host, opt; threaded = true)
    end
    rate_cpu = n_sys / t_cpu
    t_serial = _best_time_s(; reps) do
        copyto!(X_cpu, X0_host)
        cpu_loop!(X_cpu, S_cpu, prob, Newton(), B_host, opt; threaded = false)
    end
    rate_serial = n_sys / t_serial

    # 2. R2 solve_batch! with Newton(), AoS right-hand sides
    X0_dev = array_ctor(X0_host)
    X_dev = similar(X0_dev)
    S_dev = array_ctor(Vector{SolverStatus{R}}(undef, n_sys))
    P_dev = PerSystem(array_ctor(B_host))
    t_r2 = _best_time_s(; reps) do
        copyto!(X_dev, X0_dev)
        solve_batch!(X_dev, S_dev, prob, Newton(), P_dev; options = opt)
    end
    rate_r2 = n_sys / t_r2
    X_aos = Array(X_dev)

    # 3. R2 solve_batch! with Newton(), SoA right-hand sides, and the check of its roots
    X_soa_dev = similar(X0_dev)
    P_soa = PerSystem(soa_columns(B_host, array_ctor))
    t_soa = _best_time_s(; reps) do
        copyto!(X_soa_dev, X0_dev)
        solve_batch!(X_soa_dev, S_dev, prob, Newton(), P_soa; options = opt)
    end
    rate_soa = n_sys / t_soa
    root_diff = max_root_difference(X_aos, Array(X_soa_dev))
    root_diff <= ROOT_RTOL || error(
        "SoA and AoS roots differ by $(root_diff) relative to the AoS root at N_sys = $n_sys, " *
        "n = $N; the check accepts $(ROOT_RTOL)")

    # 4. R2 solve_batch! with Newton(; reuse = 4)
    t_reuse = _best_time_s(; reps) do
        copyto!(X_dev, X0_dev)
        solve_batch!(X_dev, S_dev, prob, Newton(; reuse = 4), P_dev; options = opt)
    end
    rate_reuse = n_sys / t_reuse

    # 5. R2 solve_batch! with Newton(; globalization = Backtracking()): the row that runs a line
    # search in the kernel
    t_ls = _best_time_s(; reps) do
        copyto!(X_dev, X0_dev)
        solve_batch!(X_dev, S_dev, prob, Newton(; globalization = Backtracking()), P_dev;
            options = opt)
    end
    rate_ls = n_sys / t_ls

    # Verify convergence on the last run, the Backtracking row's
    S_got = Array(S_dev)
    n_conv = count(isconverged, S_got)

    @printf(io,
        "| %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.1e | %d/%d |\n",
        backend_name,
        n_sys,
        N,
        rate_r2 / 1e6,
        rate_soa / 1e6,
        rate_reuse / 1e6,
        rate_ls / 1e6,
        rate_cpu / 1e6,
        rate_serial / 1e6,
        rate_r2 / rate_cpu,
        root_diff,
        n_conv,
        n_sys)
    return (
        n = N,
        n_sys = n_sys,
        rate_r2 = rate_r2,
        rate_soa = rate_soa,
        rate_reuse = rate_reuse,
        rate_ls = rate_ls,
        rate_cpu = rate_cpu,
        rate_serial = rate_serial,
        root_diff = root_diff
    )
end

function run_tables(io, dev, nsys_list, n_list)
    print_header(io, "R2 batched throughput in Float32 (scripts/r2_throughput.jl)", dev)
    println(io, "| device | N_sys | n | R2 Newton AoS | R2 Newton SoA | R2 reuse=4 | ",
        "R2 Backtracking | CPU R1 threaded | CPU R1 one thread | R2 AoS / CPU threaded | ",
        "SoA-AoS root difference | Backtracking converged |")
    println(io, "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    rows = NamedTuple[]
    crossovers = String[]
    for n_sys in nsys_list
        rows_n = NamedTuple[]
        for n in n_list
            push!(rows_n, measure_row(Val(n), n_sys, dev.name, dev.array; io))
        end
        # R4 crossover n where R2 Newton throughput drops below CPU threaded R1 loop, printed
        # after the table so that its rows stay one Markdown table
        crossover = findfirst(r -> r.rate_r2 < r.rate_cpu, rows_n)
        push!(crossovers,
            crossover === nothing ?
            "R4 crossover (N_sys=$n_sys): none in n=$(n_list) (R2 >= CPU-thr across all tested n)" :
            "R4 crossover (N_sys=$n_sys): n=$(rows_n[crossover].n)")
        append!(rows, rows_n)
    end
    println(io)
    foreach(l -> println(io, l), crossovers)
    return rows
end

function main(args = ARGS; io::IO = stdout)
    dev = select_device(length(args) >= 1 ? args[1] : "cpu"; io)
    dev === nothing && return nothing
    nsys_list = length(args) >= 2 ? parse.(Int, split(args[2], ',')) : 10 .^ (3:7)
    n_list = length(args) >= 3 ? parse.(Int, split(args[3], ',')) : [2, 4, 8, 12, 16]
    return Base.invokelatest(run_tables, io, dev, nsys_list, n_list)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
