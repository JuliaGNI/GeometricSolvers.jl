# Times of the dense device LU and QR in `Float32`: `factorize!!` and the solve of the
# KernelAbstractions (KA) kernels of `src/linear/ka/`, and of the vendor path where an extension
# provides one, with the phase split of the KA factorisation into its two kernels.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=<env> scripts/hardware/ka_lu_large.jl [device] [n,…] [reps]
#
# `device` is `jlarray` (the default), `metal` or `cuda`; `n,…` is a comma-separated list of
# sizes, `1024,2048` by default; `reps` is the number of timed calls after one warm-up call, 5 by
# default. `<env>` is an environment that holds GeometricSolvers, KernelAbstractions and the device
# package (JLArrays, Metal or CUDA), such as `test/` for `jlarray` or `.` with Metal.jl in the
# default `@v1.x` environment. A device that is not present prints one line and no rows.
#
# Each row names its method (`LU` or `QR`) and its path. The `KA` path runs what `factorize!!` and
# `ldiv!!` run where no extension method applies: `ka_factorize_lu!`, or `ka_factorize_qr!`, and
# the solve of `ldiv!!` with `ka_lu_backsolve!`, or `ka_qr_backsolve!`, in place of `backsolve!`,
# the scaling of the right-hand side by `rhs_exponent` and its host read included, so that it is
# measured on a device whose `factorize!!` dispatches to a vendor library as well. The `vendor`
# path calls `factorize!!` and `ldiv!!`, and is measured only where their methods for the device's
# arrays are defined in an extension module; where none is, the script prints one line that says
# so.
#
# Columns: `factorize` and `solve` are the median and the minimum wall time in ms of one call,
# each followed by a device synchronisation. `phase 1` and `phase 2` split the KA factorisation:
# for LU the pivot-swap-scale kernel against the rank-1 trailing update, for QR the Householder
# reflector kernel against its application to the trailing columns. Each phase is the sum over the
# `n` columns of its kernel's time, each launch followed by a synchronisation, as the median over
# `reps` factorisations. The note says when the two phases do not sum to within 10 % of the
# measured `factorize` median, which a synchronisation per launch can cause. `residual` is
# `‖A x - b‖∞ / (‖A‖∞ ‖x‖∞)` of the last solve, computed on the host in `Float64`.

using GeometricSolvers
using GeometricSolvers: DeviceLU, DeviceQR, LUFactorization, NonlinearProblem,
                        QRFactorization, dense_factor, factorize!!, ldexp2, ldiv!!,
                        rhs_exponent
using KernelAbstractions: KernelAbstractions, get_backend
using LinearAlgebra
using Printf
using Random: Xoshiro

include(joinpath(@__DIR__, "common.jl"))

const T_KA = Float32

ka_noop(r, x, p) = nothing

# Whether the method of `f` for the argument types `argtypes` is defined outside the package, in
# one of its extensions.
is_vendor(f, argtypes) = parentmodule(which(f, argtypes)) !== GeometricSolvers

ka_factorize(fac::DeviceLU, J) = GeometricSolvers.ka_factorize_lu!(fac, J)
ka_factorize(fac::DeviceQR, J) = GeometricSolvers.ka_factorize_qr!(fac, J)

function ka_backsolve!(fac::DeviceLU, d)
    GeometricSolvers.ka_lu_backsolve!(fac.factors, fac.ipiv, d)
end
ka_backsolve!(fac::DeviceQR, d) = GeometricSolvers.ka_qr_backsolve!(fac.factors, fac.τ, d)

# The solve that `ldiv!!` runs for a square factor, with the KA backsolve kernels in place of
# `backsolve!`, which an extension can define for the device's arrays: `rhs_exponent(r, e)`, of a
# device-wide maximum that the host reads, gives `k`, with `e` the exponent that the factorisation
# divided `J` by; `d` takes `r / 2^k`; the kernels solve in place; and `d` is multiplied by
# `2^(k - e)`.
function ka_solve!(d, fac, r)
    k = rhs_exponent(r, fac.scale_exponent)
    if iszero(k)
        copyto!(d, r)
    else
        d .= ldexp2.(r, -k)
    end
    ka_backsolve!(fac, d)
    e = k - fac.scale_exponent
    iszero(e) || map!(a -> ldexp2(a, e), d, d)
    return d
end

# The two phases of `ka_lu!`, timed kernel by kernel: the pivot-swap-scale kernel of each column
# and the rank-1 trailing update after it. The launches are those of `ka_lu!`.
function ka_phases!(fac::DeviceLU, J)
    A, ipiv, status = fac.factors, fac.ipiv, fac.status
    n = size(A, 1)
    backend = get_backend(A)
    G = GeometricSolvers.KA_LU_WORKGROUP
    piv! = GeometricSolvers.ka_lu_pivot_swap_scale_kernel!(backend, G)
    ger! = GeometricSolvers.ka_lu_ger_kernel!(backend)
    copyto!(A, J)
    fill!(status, Int32(0))
    KernelAbstractions.synchronize(backend)
    t1 = 0.0
    t2 = 0.0
    for k in 1:n
        t0 = time_ns()
        piv!(A, ipiv, status, k, n, Val(G); ndrange = G)
        KernelAbstractions.synchronize(backend)
        t1 += (time_ns() - t0) / 1e9
        if k < n
            t0 = time_ns()
            ger!(A, k; ndrange = (n - k, n - k))
            KernelAbstractions.synchronize(backend)
            t2 += (time_ns() - t0) / 1e9
        end
    end
    return t1, t2
end

# The two phases of `ka_qr!`: the reflector kernel of each column and its application to the
# columns right of it.
function ka_phases!(fac::DeviceQR, J)
    A, τ, status = fac.factors, fac.τ, fac.status
    m, n = size(A)
    backend = get_backend(A)
    ref! = GeometricSolvers.ka_qr_reflector_kernel!(backend)
    app! = GeometricSolvers.ka_qr_apply_kernel!(backend)
    copyto!(A, J)
    fill!(status, Int32(0))
    KernelAbstractions.synchronize(backend)
    t1 = 0.0
    t2 = 0.0
    for k in 1:n
        t0 = time_ns()
        ref!(A, τ, status, k, m; ndrange = 1)
        KernelAbstractions.synchronize(backend)
        t1 += (time_ns() - t0) / 1e9
        if k < n
            t0 = time_ns()
            app!(A, τ, k, m; ndrange = n - k)
            KernelAbstractions.synchronize(backend)
            t2 += (time_ns() - t0) / 1e9
        end
    end
    return t1, t2
end

function median_phases(fac, J; reps::Int)
    ka_phases!(fac, J)
    p = [ka_phases!(fac, J) for _ in 1:reps]
    return sample_median(first.(p)), sample_median(last.(p))
end

function relative_residual(A::Matrix, x::Vector, b::Vector)
    A64, x64, b64 = Float64.(A), Float64.(x), Float64.(b)
    return norm(A64 * x64 - b64, Inf) / (opnorm(A64, Inf) * norm(x64, Inf))
end

ms(t) = @sprintf("%.3f", 1e3 * t)

function table_row(io, method, path, n, tf, ts, phases, residual, note)
    p1, p2 = phases === nothing ? ("–", "–") : (ms(phases[1]), ms(phases[2]))
    @printf(io, "| %s | %s | %d | %s / %s | %s / %s | %s | %s | %.2e | %s |\n",
        method, path, n, ms(tf.median), ms(tf.min), ms(ts.median), ms(ts.min), p1, p2,
        residual, note)
    return nothing
end

# One size: the KA rows of LU and QR, and the vendor rows where an extension defines them. It
# returns the rows as named tuples, for a caller that checks them.
function measure_size(io, dev, n::Int; reps::Int)
    rows = NamedTuple[]
    rng = Xoshiro(n)
    A_host = randn(rng, T_KA, n, n)
    b_host = randn(rng, T_KA, n)
    J = dev.array(A_host)
    b = dev.array(b_host)
    x = dev.array(zeros(T_KA, n))
    d = similar(x)
    prob = NonlinearProblem(x, ka_noop)
    sync = () -> device_sync(J)
    for (name, M) in (("LU", LUFactorization()), ("QR", QRFactorization()))
        fac = Ref(dense_factor(M, prob, x, nothing))
        # KA path
        tf = time_calls(() -> (fac[] = ka_factorize(fac[], J)), sync; reps)
        fac[].info == 0 || error("KA $name at n = $n returned info = $(fac[].info)")
        ts = time_calls(() -> ka_solve!(d, fac[], b), sync; reps)
        res = relative_residual(A_host, Array(d), b_host)
        phases = median_phases(fac[], J; reps)
        note = phase_note(phases, tf.median)
        table_row(io, name, "KA", n, tf, ts, phases, res, note)
        push!(rows,
            (method = name, path = "KA", n = n,
                factorize = tf, solve = ts, phases = phases, residual = res))
        # vendor path
        argtypes = (typeof(fac[]), typeof(M), typeof(J))
        if is_vendor(factorize!!, argtypes)
            fac[] = dense_factor(M, prob, x, nothing)
            tf = time_calls(() -> (fac[] = factorize!!(fac[], M, J)), sync; reps)
            fac[].info == 0 || error("vendor $name at n = $n returned info = $(fac[].info)")
            ts = time_calls(() -> ldiv!!(d, fac[], b), sync; reps)
            res = relative_residual(A_host, Array(d), b_host)
            table_row(io, name, "vendor", n, tf, ts, nothing, res, "")
            push!(rows,
                (method = name, path = "vendor", n = n, factorize = tf, solve = ts,
                    phases = nothing, residual = res))
        else
            println(
                io, "| $name | vendor | $n | no extension method of `factorize!!` for ",
                typeof(J), " | | | | | |")
        end
    end
    return rows
end

function run_tables(io, dev, ns, reps)
    print_header(
        io, "KA and vendor LU and QR in Float32 (scripts/hardware/ka_lu_large.jl)",
        dev; extra = ["reps" => string(reps)])
    println(io,
        "| method | path | n | factorize ms (median / min) | solve ms (median / min) | ",
        "phase 1 ms | phase 2 ms | residual | note |")
    println(io, "|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
    rows = NamedTuple[]
    for n in ns
        append!(rows, measure_size(io, dev, n; reps))
    end
    return rows
end

function main(args = ARGS; io::IO = stdout)
    dev = select_device(length(args) >= 1 ? args[1] : "jlarray"; io)
    dev === nothing && return nothing
    dev.name == "cpu" && throw(ArgumentError(
        "the KA kernels run on a device array; choose jlarray, metal or cuda"))
    ns = length(args) >= 2 ? parse.(Int, split(args[2], ',')) : [1024, 2048]
    reps = length(args) >= 3 ? parse(Int, args[3]) : 5
    return Base.invokelatest(run_tables, io, dev, ns, reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
