# The Krylov-recycling spike: the Jacobian–vector products of ten Newton solves along a parameter
# path, with restarted GMRES and with GCRO-DR, the GCRO method with deflated restarts that carries
# a recycled space from one linear solve to the next, under a lagged LU right preconditioner. It
# times nothing: the comparison is of counts.
#
# Usage, from the root of the repository:
#   julia --startup-file=no --project=. scripts/recycling_spike.jl [N,…] [m,…] [k,…] [η,…]
#
# The defaults are the full grid: `N = 125,768`, `m = 20,30`, `k = 5,10`, `η = 1e-2,1e-6`.
#
# The problem is `sparse_bracket_problem(N)` of `test/helpers/sparse_problems.jl`, of dimension
# `n = 2N`. A cell solves it ten times, at `p = 1.00, 1.05, …, 1.45`, the first solve from the
# problem's `z0` and each later one from the root of the solve before. Every solve is a Newton
# iteration with the forcing term `FixedForcing(η)`, the default `Options` (the residual test
# `‖F‖ ≤ √eps ‖F(x₀)‖`, at most 100 steps) and the right preconditioner `M = LU(J(x))` of the
# analytic Jacobian at the iterate of the step that refreshes it: the first step of a solve, every
# `k` steps after, and the step after one whose inner solve reached its cap, as
# `Newton(; reuse = k)` refreshes a `GMRES` preconditioner. That `Newton` also refreshes after a
# `STALLED` step, which the script's own Newton loop does not; the Newton steps and the refreshes
# of a row show where the loops part. Every product is `J(x) v` with the analytic Jacobian. The
# three columns of a cell, each the sum over the ten solves:
#
# - `GMRES(m)`: `Newton(; linsolve = GMRES(; restart = m, maxiter = 500, forcing =
#   FixedForcing(η), preconditioner = M), reuse = k)` of the package, its products counted by
#   `jvp_calls`.
# - `GCRO-DR no recycling`: the GCRO-DR(m, k) of this script, in a Newton loop of this script that
#   takes the steps of the package's `Newton` above, with its recycled space discarded at the start
#   of each linear solve, so that it deflates across the restarts of one solve only.
# - `GCRO-DR recycling`: the same, with the recycled space `U` kept from one linear solve to the
#   next, across Newton steps and across the ten solves.
#
# GCRO-DR(m, k) keeps `k` recycled vectors and runs `m − k` Arnoldi steps per cycle, so that its
# basis, `C` and `V`, holds `m + 1` columns, as that of `GMRES(m)` does. A solve without a
# recycled space starts with one GMRES(m) cycle. A solve that starts with `U` takes `k` products
# for `C = A U`, which its count includes. Each GCRO-DR count is the script's own count of its
# Arnoldi and `A·U` products, and the script stops with an error where that count differs from a
# direct count of the operator's calls. A cell that does not converge in a column shows
# `no convergence` there. Beside the three columns, a row gives the Newton steps and the refreshes
# of the preconditioner of each column, in the order of the columns, and the linear solves of
# GCRO-DR with recycling that started from a recycled space. A cycle of fewer than `k` Arnoldi
# steps leaves no recycled space, so a solve after such cycles alone starts without one.
#
# The control runs first: ten solves of `A x = b_i` with a dense `400 × 400` matrix `A` with six
# eigenvalues in `[0.01, 0.05]` and the others in `[1, 3]`, ten random right-hand sides, each
# solve one Newton step from the root before, with `η = 1e-8`, `m = 30`, `k = 10` and no
# preconditioner. The script stops with an error unless the products of GCRO-DR with recycling
# are fewer than those without, those fewer than those of GMRES(m), and every solve converges.
#
# The verdict applies the rules decided before the run, on the full grid only; on any other grid
# it says that it is not evaluated. `r` is the ratio of the products of GCRO-DR with recycling to
# those of GMRES(m). Build rule: one pair `(m, k)` at which, at both `N` and both `η`, every column
# converges, `r ≤ 0.75`, and GCRO-DR with recycling takes fewer products than without. Stop rule:
# `r ≥ 0.90` in every cell, where a cell in which GMRES(m) converges and GCRO-DR with recycling
# does not counts as `r ≥ 0.90`, and a cell in which GMRES(m) does not converge counts as neither.
# Any other result is between the two.

using GeometricSolvers
using GeometricSolvers: FixedForcing, JVPFunction, jvp_calls
using KernelAbstractions: KernelAbstractions
using LinearAlgebra
using Printf
using Random: Xoshiro
using SparseArrays: SparseMatrixCSC

include(joinpath(@__DIR__, "hardware", "common.jl"))
include(joinpath(@__DIR__, "..", "test", "helpers", "sparse_problems.jl"))

const RC_NS = [125, 768]
const RC_MS = [20, 30]
const RC_KS = [5, 10]
const RC_ETAS = [1e-2, 1e-6]
const RC_PS = [1.0 + 0.05 * i for i in 0:9]
"the cap on the inner iterations of one linear solve, in every column"
const RC_MAXITER = 500
const RC_BUILD_RATIO = 0.75
const RC_STOP_RATIO = 0.90

# ─── The preconditioner ────────────────────────────────────────────────────────

# The right preconditioner `M = LU(J(x))`: `refresh!` evaluates the analytic Jacobian `J!` at the
# iterate and factorises it, and `apply_preconditioner!` solves with that factor.
mutable struct LUPreconditioner{FJ, M, F}
    J!::FJ
    J::M
    fac::F
    refreshes::Int
end

function LUPreconditioner(J!, prototype, x, p)
    J = copy(prototype)
    J!(J, x, p)
    return LUPreconditioner(J!, J, lu(J), 0)
end

function lu_refresh!(P::LUPreconditioner, x, p)
    P.J!(P.J, x, p)
    P.fac = lu(P.J)
    P.refreshes += 1
    return nothing
end

GeometricSolvers.refresh!(P::LUPreconditioner, prob, x, Fx, p) = lu_refresh!(P, x, p)
GeometricSolvers.apply_preconditioner!(z, P::LUPreconditioner, v) = ldiv!(z, P.fac, v)

precondition!(z, ::Nothing, v) = copyto!(z, v)
precondition!(z, P::LUPreconditioner, v) = ldiv!(z, P.fac, v)

# ─── GCRO-DR ───────────────────────────────────────────────────────────────────

# The right-preconditioned operator `v ↦ J M⁻¹ v` of a Newton solve, whose `J` each step
# overwrites, with a direct count of its calls.
mutable struct CountingOperator{M, P}
    J::M
    P::P
    z::Vector{Float64}
    calls::Int
end

function apply_operator!(y, op::CountingOperator, v)
    precondition!(op.z, op.P, v)
    mul!(y, op.J, op.z)
    op.calls += 1
    return y
end

# The GCRO-DR(m, k) solver: the recycled space `U` that a solve leaves, whether the next solve
# starts from it (`recycle`), the script's own count of the Arnoldi products and of the `A·U`
# products, and the largest number of basis columns `[C V]` that a cycle held.
mutable struct GCRODR
    m::Int
    k::Int
    maxiter::Int
    recycle::Bool
    U::Union{Nothing, Matrix{Float64}}
    arnoldi::Int
    au::Int
    basis::Int
end

function GCRODR(m::Integer, k::Integer; recycle::Bool, maxiter::Integer = RC_MAXITER)
    0 < k < m || throw(ArgumentError("GCRO-DR needs 0 < k < m, got m = $m, k = $k"))
    return GCRODR(m, k, maxiter, recycle, nothing, 0, 0, 0)
end

products(s::GCRODR) = s.arnoldi + s.au

# A real orthonormal basis of the `k` eigenvectors of the pencil `(A, B)` whose eigenvalues have
# the smallest modulus. A complex pair contributes its real and imaginary parts, and a pair cut at
# the `k`-th vector its real part only.
function harmonic_basis(A::AbstractMatrix, B::AbstractMatrix, k::Integer)
    E = eigen(A, B)
    cols = Vector{Vector{Float64}}()
    taken = ComplexF64[]
    for i in sortperm(abs.(E.values))
        length(cols) == k && break
        λ = E.values[i]
        v = E.vectors[:, i]
        if imag(λ) == 0
            push!(cols, real.(v))
        elseif !(conj(λ) in taken)
            push!(taken, λ)
            push!(cols, real.(v))
            length(cols) < k && push!(cols, imag.(v))
        end
    end
    return Matrix(qr(reduce(hcat, cols)).Q)[:, 1:k]
end

# One cycle from the residual `r` of norm `β`: `nsteps` Arnoldi steps at most, orthogonal to `C`
# where a recycled space `(C, U)` is given, until the least-squares residual reaches `tol`. It
# updates `y` and `r`, and returns the number of steps and the new recycled space `(C, U)` from the
# harmonic Ritz vectors of the cycle, or `nothing` where the cycle spans fewer than `k` vectors.
function gcrodr_cycle!(s::GCRODR, op, y, r, β, C, U, nsteps, tol)
    n = length(r)
    kc = C === nothing ? 0 : size(C, 2)
    W = zeros(n, kc + nsteps + 1)
    s.basis = max(s.basis, size(W, 2))
    G = zeros(kc + nsteps + 1, kc + nsteps)
    rhs = zeros(kc + nsteps + 1)
    rhs[kc + 1] = β
    if kc > 0
        W[:, 1:kc] .= C
        d = [1 / norm(u) for u in eachcol(U)]
        G[1:kc, 1:kc] .= Diagonal(d)
        Û = U .* d'
    end
    W[:, kc + 1] .= r ./ β
    w = zeros(n)
    j = 0
    while j < nsteps
        j += 1
        apply_operator!(w, op, view(W, :, kc + j))
        s.arnoldi += 1
        basis = view(W, :, 1:(kc + j))
        h = basis' * w
        w .-= basis * h
        h2 = basis' * w
        w .-= basis * h2
        G[1:(kc + j), kc + j] .= h .+ h2
        hn = norm(w)
        G[kc + j + 1, kc + j] = hn
        hn > 0 && (W[:, kc + j + 1] .= w ./ hn)
        Gj = G[1:(kc + j + 1), 1:(kc + j)]
        res = norm(rhs[1:(kc + j + 1)] - Gj * (Gj \ rhs[1:(kc + j + 1)]))
        (res <= tol || hn == 0) && break
    end
    J = kc + j
    Gj = G[1:(J + 1), 1:J]
    Wj = W[:, 1:(J + 1)]
    z = Gj \ rhs[1:(J + 1)]
    V̂ = kc > 0 ? hcat(Û, W[:, (kc + 1):(kc + j)]) : W[:, 1:j]
    y .+= V̂ * z
    r .= Wj * (rhs[1:(J + 1)] - Gj * z)
    J < s.k && return j, nothing
    P = harmonic_basis(Gj' * Gj, Gj' * (Wj' * V̂), s.k)
    F = qr(Gj * P)
    return j, (Wj * Matrix(F.Q)[:, 1:(s.k)], (V̂ * P) / UpperTriangular(F.R))
end

# The GCRO-DR solve of `A y = b` from `y = 0` to `‖b − A y‖ ≤ tol`, with `A` the operator `op`.
# It returns `y` and whether the solve reached its cap of `maxiter` Arnoldi steps without `tol`.
function gcrodr_solve!(s::GCRODR, op, b, tol)
    s.recycle || (s.U = nothing)
    n = length(b)
    y = zeros(n)
    r = copy(b)
    rn = norm(r)
    rn <= tol && return y, false
    C = U = nothing
    if s.U !== nothing
        Ũ = s.U
        AU = similar(Ũ)
        for i in axes(Ũ, 2)
            apply_operator!(view(AU, :, i), op, view(Ũ, :, i))
            s.au += 1
        end
        F = qr(AU)
        C = Matrix(F.Q)[:, 1:(s.k)]
        U = Ũ / UpperTriangular(F.R)
        c = C' * r
        y .+= U * c
        r .-= C * c
        rn = norm(r)
    end
    iters = 0
    while rn > tol && iters < s.maxiter
        nsteps = min(C === nothing ? s.m : s.m - s.k, s.maxiter - iters)
        j, CU = gcrodr_cycle!(s, op, y, r, rn, C, U, nsteps, tol)
        iters += j
        CU === nothing || ((C, U) = CU)
        rn = norm(r)
    end
    U === nothing || (s.U = U)
    return y, rn > tol
end

# ─── The Newton loops ──────────────────────────────────────────────────────────

# A Newton solve of `F!(r, x, p) = 0` from `x` with GCRO-DR, in the steps of `Newton(; reuse)`
# with a preconditioned `GMRES` and `FixedForcing(η)`: the inner tolerance `η ‖F‖`, the full step,
# the residual test `‖F‖ ≤ √eps ‖F(x₀)‖`, the step test `‖d‖ ≤ 2 eps ‖x‖`, at most 100 steps, and
# the refresh of `P` at age 0 and after an inner solve that reached its cap. `P = nothing` takes
# no preconditioner. It writes the last iterate into `x` and returns whether the residual test
# holds and the number of steps.
function gcrodr_newton!(s::GCRODR, F!, J!, Jbuf, P, x, p, η; reuse = nothing)
    op = CountingOperator(Jbuf, P, zeros(length(x)), 0)
    counted = products(s)
    F = similar(x)
    F!(F, x, p)
    fnorm = norm(F)
    tol = sqrt(eps(Float64)) * fnorm
    age, early, steps = 0, false, 0
    while fnorm > tol && steps < 100
        if P !== nothing
            if reuse === nothing || age == 0 || early
                lu_refresh!(P, x, p)
                age = (reuse === nothing || reuse == 1) ? 0 : 1
            else
                age = age + 1 >= reuse ? 0 : age + 1
            end
        end
        J!(Jbuf, x, p)
        y, early = gcrodr_solve!(s, op, -F, η * fnorm)
        d = similar(x)
        precondition!(d, P, y)
        x .+= d
        F!(F, x, p)
        fnorm = norm(F)
        steps += 1
        norm(d) <= 2 * eps(Float64) * norm(x) && break
    end
    counted = products(s) - counted
    counted == op.calls || error(
        "GCRO-DR counted $counted products, the operator was called $(op.calls) times")
    return fnorm <= tol, steps
end

# The ten solves of one column along `ps`, each from the root before, starting from `x0`.
# `column` is `:gmres`, `:norec` or `:rec`. Returns the products, whether every solve converged,
# the Newton steps, the refreshes of the preconditioner, and the linear solves that started from a
# recycled space. `maxiter` caps the inner iterations of each linear solve, and the `restart` of
# `GMRES` at `min(m, maxiter)`.
function run_column(column::Symbol, prob, x0, ps; m, k, η, preconditioned::Bool, reuse,
        maxiter = RC_MAXITER)
    x = copy(x0)
    Jbuf = copy(prob.jacobian_prototype)
    P = preconditioned ? LUPreconditioner(prob.J, prob.jacobian_prototype, x, first(ps)) :
        nothing
    converged, steps = true, 0
    if column === :gmres
        Jjvp = copy(prob.jacobian_prototype)
        jvp! = (Jv, x, v, p) -> (prob.J(Jjvp, x, p); mul!(Jv, Jjvp, v); Jv)
        g = GMRES(; restart = min(m, maxiter), maxiter, forcing = FixedForcing(η),
            preconditioner = P)
        method = preconditioned ?
                 Newton(; linsolve = g, reuse, jacobian = JVPFunction(jvp!)) :
                 Newton(; linsolve = g, jacobian = JVPFunction(jvp!))
        solver = init(prob, x, method, first(ps))
        count = 0
        for p in ps
            _, status = solve!(solver, x, p)
            count += jvp_calls(solver)
            converged &= status.code == GeometricSolvers.SUCCESS
            steps += status.iterations
        end
        return (products = count, converged, steps, refreshes = refreshes(P), from_u = 0)
    end
    s = GCRODR(m, k; recycle = column === :rec, maxiter)
    for p in ps
        ok, n = gcrodr_newton!(s, prob.F, prob.J, Jbuf, P, x, p, η;
            reuse = preconditioned ? reuse : nothing)
        converged &= ok
        steps += n
    end
    return (products = products(s), converged, steps, refreshes = refreshes(P),
        from_u = s.au ÷ s.k)
end

refreshes(::Nothing) = 0
refreshes(P::LUPreconditioner) = P.refreshes

# ─── The control ───────────────────────────────────────────────────────────────

# The control problem: `F(x) = A x − b_p` for the `p`-th of ten random right-hand sides, with
# `A = S D S⁻¹`, `D` six eigenvalues in `[0.01, 0.05]` and `n − 6` in `[1, 3]`, and
# `S = I + E / √n` for a random `E`.
function control_problem(n::Integer = 400)
    rng = Xoshiro(20061122)
    D = vcat(collect(range(0.01, 0.05; length = 6)), 1 .+ 2 .* rand(rng, n - 6))
    S = I + randn(rng, n, n) ./ sqrt(n)
    A = S * Diagonal(D) / S
    B = randn(rng, n, 10)
    F! = (r, x, p) -> (mul!(r, A, x); r .-= view(B, :, p); r)
    J! = (J, x, p) -> copyto!(J, A)
    return NonlinearProblem(F!, J!; jacobian_prototype = copy(A)), zeros(n)
end

"""
    run_control(; m = 30, k = 10, η = 1e-8)

The products of the three columns on the control problem, ten solves without a preconditioner.
It throws an `ErrorException` unless GCRO-DR with recycling takes fewer products than without,
that fewer than GMRES(m), and every solve converges.
"""
function run_control(; m = 30, k = 10, η = 1e-8)
    prob, x0 = control_problem()
    ps = 1:10
    cols = map((:gmres, :norec, :rec)) do c
        run_column(c, prob, x0, ps; m, k, η, preconditioned = false, reuse = nothing)
    end
    check_control(cols...)
    gmres, norec, rec = cols
    return (gmres = gmres.products, no_recycling = norec.products, recycling = rec.products)
end

# The assertion of the control on its three columns: every solve converges, and GCRO-DR with
# recycling takes fewer products than without, that fewer than GMRES(m).
function check_control(gmres, norec, rec)
    all(c -> c.converged, (gmres, norec, rec)) ||
        error("the control does not converge in every column")
    rec.products < norec.products < gmres.products || error(
        "the control's order fails: GCRO-DR recycling $(rec.products), no recycling " *
        "$(norec.products), GMRES(m) $(gmres.products)")
    return nothing
end

# ─── The grid, the table and the verdict ───────────────────────────────────────

"one cell of the grid: the three columns of `N`, `m`, `k` and `η`"
function run_cell(N, m, k, η)
    prob, _, z0, _, _ = sparse_bracket_problem(N, Float64)
    cols = map((:gmres, :norec, :rec)) do c
        run_column(c, prob, z0, RC_PS; m, k, η, preconditioned = true, reuse = k)
    end
    return (N = N, m = m, k = k, η = η, gmres = cols[1], norec = cols[2], rec = cols[3])
end

cell_text(c) = c.converged ? string(c.products) : "no convergence"

function ratio_text(row)
    (row.gmres.converged && row.rec.converged) || return "–"
    return @sprintf("%.2f", row.rec.products / row.gmres.products)
end

three(row, f) = join((f(row.gmres), f(row.norec), f(row.rec)), " / ")

function print_row(io::IO, row)
    println(io, "| ", row.N, " | ", row.m, " | ", row.k, " | ", row.η, " | ",
        cell_text(row.gmres), " | ", cell_text(row.norec), " | ", cell_text(row.rec), " | ",
        ratio_text(row), " | ", three(row, c -> c.steps), " | ", three(row, c -> c.refreshes),
        " | ", row.rec.from_u, " |")
end

# Whether a cell meets the build rule: every column converges, `r ≤ 0.75`, and recycling takes
# fewer products than no recycling.
function build_cell(row)
    all(c -> c.converged, (row.gmres, row.norec, row.rec)) || return false
    return row.rec.products <= RC_BUILD_RATIO * row.gmres.products &&
           row.rec.products < row.norec.products
end

# Whether a cell meets the stop rule, `r ≥ 0.90`: a GMRES(m) that does not converge meets neither
# rule, and a GCRO-DR with recycling that does not converge, beside a GMRES(m) that does, meets it.
function stop_cell(row)
    row.gmres.converged || return false
    row.rec.converged || return true
    return row.rec.products >= RC_STOP_RATIO * row.gmres.products
end

function full_grid(rows)
    sort([(r.N, r.m, r.k, r.η) for r in rows]) ==
    sort(vec([(N, m, k, η) for N in RC_NS, m in RC_MS, k in RC_KS, η in RC_ETAS]))
end

"""
    rc_verdict(rows)

The verdict of the rows of a grid: `rule` is `:build`, `:stop`, `:between`, or `:not_evaluated`
on a grid other than the full one; `pairs` holds the pairs `(m, k)` that meet the build rule.
"""
function rc_verdict(rows)
    full_grid(rows) || return (rule = :not_evaluated, pairs = Tuple{Int, Int}[])
    pairs = [(m, k) for m in RC_MS
             for k in RC_KS
             if all(build_cell, filter(r -> (r.m, r.k) == (m, k), rows))]
    rule = !isempty(pairs) ? :build : all(stop_cell, rows) ? :stop : :between
    return (rule = rule, pairs = pairs)
end

function print_verdict(io::IO, rows, v)
    if v.rule === :not_evaluated
        println(io, "build rule: not evaluated (the grid is not the full grid)")
        println(io, "stop rule: not evaluated (the grid is not the full grid)")
        return nothing
    end
    pairs = join(("(m, k) = ($m, $k)" for (m, k) in v.pairs), ", ")
    println(
        io, "build rule (r ≤ 0.75 and below GCRO-DR without recycling, at both N and both ",
        "η): ", isempty(v.pairs) ? "not met" : "met at " * pairs)
    nstop = count(stop_cell, rows)
    println(io, "stop rule (r ≥ 0.90 in every cell): ",
        nstop == length(rows) ? "met" : "not met",
        " (", nstop, " of ", length(rows), " cells)")
    println(io, "result: ", v.rule)
    return nothing
end

blas_config() = join((basename(l.libname) for l in BLAS.get_config().loaded_libs), ", ")

parse_list(T, s) = [parse(T, strip(x)) for x in split(s, ',')]

"""
    main(args = ARGS; io = stdout)

Run the control, then the grid of `args` (`N,…`, `m,…`, `k,…`, `η,…`, each with the full grid's
default), print the header, the table and the verdict to `io`, and return
`(control, rows, verdict)`.
"""
function main(args = ARGS; io::IO = stdout)
    Ns = length(args) >= 1 ? parse_list(Int, args[1]) : RC_NS
    ms = length(args) >= 2 ? parse_list(Int, args[2]) : RC_MS
    ks = length(args) >= 3 ? parse_list(Int, args[3]) : RC_KS
    ηs = length(args) >= 4 ? parse_list(Float64, args[4]) : RC_ETAS
    println(io, "# Krylov recycling spike: Jacobian–vector products of ten Newton solves")
    println(io, "# commit: ", script_commit())
    println(io, "# machine: ", Sys.cpu_info()[1].model, " (", Sys.MACHINE, ")")
    println(io, "# julia: ", VERSION)
    println(io, "# BLAS: ", blas_config(), ", ", BLAS.get_num_threads(), " threads")
    control = run_control()
    println(io, "# control (m = 30, k = 10, η = 1e-8): GMRES(m) ", control.gmres,
        ", GCRO-DR no recycling ", control.no_recycling, ", GCRO-DR recycling ",
        control.recycling)
    println(io)
    println(
        io, "| N | m | k | η | GMRES(m) | GCRO-DR no recycling | GCRO-DR recycling | r | ",
        "Newton steps | refreshes | solves from U |")
    println(io, "|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|--:|")
    rows = []
    for N in Ns, m in ms, k in ks, η in ηs
        row = run_cell(N, m, k, η)
        print_row(io, row)
        push!(rows, row)
    end
    println(io)
    verdict = rc_verdict(rows)
    print_verdict(io, rows, verdict)
    return (control = control, rows = rows, verdict = verdict)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
