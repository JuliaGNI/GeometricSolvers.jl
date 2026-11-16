# One R3 Newton step split into host synchronisations and arithmetic, with the count of each.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=<env> scripts/hardware/sync_cost.jl [device] [n,…] [reps]
#
# `device` is `cpu` (the default), `jlarray`, `metal` or `cuda`; `n,…` is a comma-separated list of
# sizes, `64,256,1024` by default; `reps` is the number of timed steps after one warm-up step, 5 by
# default. `<env>` is an environment that holds GeometricSolvers, KernelAbstractions and the device
# package, as for `ka_lu_large.jl`. A device that is not present prints one line and no rows.
#
# The problem is `F(x) = A x + sin.(x) / 10 - b` in `Float32`, with `A = randn(n, n) / √n + 2I` and
# `b` such that a random `x⋆` is its root, and its analytic Jacobian `A + Diagonal(cos.(x) / 10)`.
# The step is one `step!!` of `Newton()`: `LUFactorization`, the `Static` full step, no refinement
# and no regularisation, from a start `x⋆ + 0.1 randn(n)`.
#
# The script times the step as one call, and then its phases one by one, each a call of the
# functions that `step!!` calls, in its order, each followed by a device synchronisation:
#
# | phase | calls |
# |:--|:--|
# | jacobian | `jacobian!!` |
# | gradient ratio | `grad_ratio_and_norm`: `rnorm(J)`, `Jᴴ F`, `rnorm(Jᴴ F)` |
# | factorize | `regularize!!`, `factorize!!`, `factor_diagnostics` |
# | direction | `newton_direction`: `ldiv!!`, the sign change, `refine!!` |
# | update | `newton_line`: `rnorm(d)`, `x + d`, the residual, `rnorm(F)`, two copies |
#
# A wait is a point at which the host waits for the device: a reduction whose value the host reads,
# a `synchronize`, or a copy of the factor status to the host. A launch is a device operation whose
# result stays on the device: a kernel of the package, a broadcast, a copy, a matrix-vector product,
# or one call of a vendor library. The counts per phase are those of the source of the step on that
# path; on `cpu` the step runs on the host, with no waits and no launches. A vendor call that
# returns a status to the host counts as one launch and one wait; any wait inside a vendor solve
# that returns nothing is not counted.
#
# `t_wait` is the median time of one wait with no work queued: a `synchronize` and a reduction of a
# one-entry array read by the host. The synchronisation time of the step is `waits × t_wait`, and
# its arithmetic time is the sum of the phases less that. The note says when the phases do not sum
# to within 10 % of the measured step.

using GeometricSolvers
using GeometricSolvers: Newton, NonlinearProblem, factorize!!, init_state, residual!!,
                        rnorm, step!!
using KernelAbstractions: KernelAbstractions
using LinearAlgebra
using Printf
using Random: Xoshiro

include(joinpath(@__DIR__, "common.jl"))

const T_SYNC = Float32

# The residual `A x + sin.(x) / 10 - b`, in place, from the parameters `(A, b)`.
struct SyncResidual end

function (::SyncResidual)(r, x, p)
    mul!(r, p.A, x)
    r .+= sin.(x) ./ 10 .- p.b
    return r
end

# Its Jacobian `A + Diagonal(cos.(x) / 10)`, in place.
struct SyncJacobian end

function (::SyncJacobian)(J, x, p)
    copyto!(J, p.A)
    view(J, diagind(J)) .+= cos.(x) ./ 10
    return J
end

# Launches of the residual and of the Jacobian above: `mul!` and one broadcast, and a copy and one
# broadcast.
const RESIDUAL_LAUNCHES = 2
const JACOBIAN_LAUNCHES = 2

function make_problem(dev, n::Int)
    rng = Xoshiro(n)
    A = randn(rng, T_SYNC, n, n) ./ sqrt(T_SYNC(n)) + T_SYNC(2) * I
    x_star = randn(rng, T_SYNC, n)
    b = A * x_star + sin.(x_star) ./ 10
    x0 = x_star + T_SYNC(0.1) * randn(rng, T_SYNC, n)
    p = (A = dev.array(A), b = dev.array(b))
    return NonlinearProblem(SyncResidual(), SyncJacobian()), p, dev.array(x0)
end

# The state, the iterate and the residual at the start of a solve from `x0`, as `solve!!` prepares
# them before its first step.
function start_step(m, prob, x0, p)
    state = init_state(m, prob, x0, p)
    x = copy(x0)
    Fx = residual!!(GeometricSolvers.residual_buffer(state), prob, x, p)
    state = GeometricSolvers.start!!(state, m, prob, x, Fx, p, rnorm(Fx))
    return state, x, Fx
end

# The phases of `step!!` for this method, each timed with a synchronisation after it.
function step_phases(m, prob, x0, p, sync)
    state, x, Fx = start_step(m, prob, x0, p)
    sync()
    G = GeometricSolvers
    times = zeros(5)
    R = real(eltype(x))
    fnorm = R(G.residual_norm(Fx, state.merit, state.fnorm))
    t0 = time_ns()
    Jmat = G.jacobian!!(state.J, state.jacprep, prob, x, p)
    sync()
    times[1] = (time_ns() - t0) / 1e9
    t0 = time_ns()
    G.grad_ratio_and_norm(state.d, Jmat, Fx, fnorm)
    sync()
    times[2] = (time_ns() - t0) / 1e9
    t0 = time_ns()
    Jreg = G.regularize!!(Jmat, m.regularization)
    fac = factorize!!(state.fac, m.linsolve, Jreg)
    G.factor_diagnostics(state.d, fac, Fx, fnorm)
    sync()
    times[3] = (time_ns() - t0) / 1e9
    fac.info == 0 || error("the factorisation of the step returned info = $(fac.info)")
    t0 = time_ns()
    d, _ = G.newton_direction(m, state, prob, x, Fx, p, fac, Jreg)
    sync()
    times[4] = (time_ns() - t0) / 1e9
    t0 = time_ns()
    G.newton_line(m, state, prob, x, Fx, p, fac, d, true)
    sync()
    times[5] = (time_ns() - t0) / 1e9
    return times
end

function step_time(m, prob, x0, p, sync)
    state, x, Fx = start_step(m, prob, x0, p)
    sync()
    t0 = time_ns()
    _, _, _, info = step!!(state, m, prob, x, Fx, p)
    sync()
    t = (time_ns() - t0) / 1e9
    info.code == SUCCESS || error("the step returned $(info.code)")
    return t
end

# The waits and the launches of each phase on the path of `fac`: the host (`cpu`), the KA kernels,
# or the vendor library of the device `device`.
function phase_counts(path::Symbol, n::Int, device::AbstractString)
    path === :host && return fill((0, 0), 5)
    # The Jacobian buffer of a `Newton` state on a device array is the factor's own storage, so
    # `factorize!!` copies nothing before it factorises.
    if path === :ka
        # factorize: status fill, n pivot and n - 1 update kernels; synchronize and status copy.
        # direction: `rhs_exponent`'s maximum and the backsolve's synchronize; copy, permuted
        # forward solve, n - 1 column and one diagonal kernel, sign change.
        fac = (2n, 2)
        dir = (n + 3, 2)
    else
        # factorize: the library factorisation, whose status the host reads; on Metal also the
        # broadcast of the pivots that MPS returns in a new array into the factor's own
        # (`_mps_lu_finish!` of the Metal extension).
        # direction: `rhs_exponent`'s maximum; copy, the library solve, sign change.
        fac = device == "metal" ? (2, 1) : (1, 1)
        dir = (3, 1)
    end
    return [
        (JACOBIAN_LAUNCHES, 0),
        (1, 2),
        fac,
        dir,
        (1 + RESIDUAL_LAUNCHES + 2, 2)
    ]
end

function factor_path(fac, m, J)
    J isa Array && return :host
    method = which(factorize!!, (typeof(fac), typeof(m.linsolve), typeof(J)))
    return parentmodule(method) === GeometricSolvers ? :ka : :vendor
end

# One wait with no work queued: a synchronisation and a one-entry reduction read by the host.
function wait_time(dev; reps::Int)
    one_entry = dev.array(ones(T_SYNC, 1))
    return time_calls(() -> sum(one_entry), () -> device_sync(one_entry); reps).median
end

const PHASES = ("jacobian", "gradient ratio", "factorize", "direction", "update")

ms(t) = @sprintf("%.4f", 1e3 * t)

function measure_size(io, dev, n::Int; reps::Int)
    m = Newton()
    prob, p, x0 = make_problem(dev, n)
    sync = () -> device_sync(x0)
    state0 = init_state(m, prob, x0, p)
    path = factor_path(state0.fac, m, state0.J)
    counts = phase_counts(path, n, dev.name)
    t_wait = path === :host ? 0.0 : wait_time(dev; reps)
    # one warm-up call of each, then `reps` timed ones
    t_step = middle([step_time(m, prob, x0, p, sync) for _ in 1:(reps + 1)][2:end])
    phase_list = [step_phases(m, prob, x0, p, sync) for _ in 1:(reps + 1)][2:end]
    phases = [middle([pl[k] for pl in phase_list]) for k in 1:5]
    for k in 1:5
        launches, waits = counts[k]
        @printf(io, "| %d | %s | %s | %s | %d | %d | %s |\n", n, path, PHASES[k],
            ms(phases[k]), launches, waits, "")
    end
    waits = sum(last, counts)
    launches = sum(first, counts)
    t_sync = waits * t_wait
    t_arith = sum(phases) - t_sync
    @printf(io, "| %d | %s | step (measured) | %s | | | |\n", n, path, ms(t_step))
    @printf(io, "| %d | %s | synchronisations | %s | | %d | t_wait = %s ms |\n", n, path,
        ms(t_sync), waits, ms(t_wait))
    @printf(io, "| %d | %s | arithmetic | %s | %d | | %s |\n", n, path, ms(t_arith),
        launches, phase_note(phases, t_step))
    return (n = n, path = path, phases = phases, step = t_step, t_wait = t_wait,
        waits = waits, launches = launches, sync = t_sync, arithmetic = t_arith)
end

function run_tables(io, dev, ns, reps)
    print_header(io,
        "One R3 Newton step in Float32: synchronisations and arithmetic " *
        "(scripts/hardware/sync_cost.jl)",
        dev;
        extra = ["reps" => string(reps)])
    println(io, "| n | path | phase | ms (median) | launches | waits | note |")
    println(io, "|--:|:--|:--|--:|--:|--:|:--|")
    return [measure_size(io, dev, n; reps) for n in ns]
end

function main(args = ARGS; io::IO = stdout)
    dev = select_device(length(args) >= 1 ? args[1] : "cpu"; io)
    dev === nothing && return nothing
    ns = length(args) >= 2 ? parse.(Int, split(args[2], ',')) : [64, 256, 1024]
    reps = length(args) >= 3 ? parse(Int, args[3]) : 5
    return Base.invokelatest(run_tables, io, dev, ns, reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
