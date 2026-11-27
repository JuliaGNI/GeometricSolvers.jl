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
"the period of the preconditioner, `Newton(; reuse)`, in Newton steps"
const RC_REUSE = 1000
const RC_BUILD_RATIO = 0.75
const RC_STOP_RATIO = 0.90

# ─── The preconditioner ────────────────────────────────────────────────────────

# The right preconditioner `M = LU(J(x))`: `refresh!` evaluates the analytic Jacobian `J!` at the
# iterate and factorises it, and `apply_preconditioner!` solves with that factor. It counts its
# refreshes, and those at the iterate `start`, which the caller sets to the start of each solve:
# a refresh there is one at the first step of a solve, since a later step of a solve leaves that
# point or ends the solve.
mutable struct LUPreconditioner{FJ, M, F}
    J!::FJ
    J::M
    fac::F
    refreshes::Int
    start::Vector{Float64}
    start_refreshes::Int
end

function LUPreconditioner(J!, prototype, x, p)
    J = copy(prototype)
    J!(J, x, p)
    return LUPreconditioner(J!, J, lu(J), 0, copy(x), 0)
end

function lu_refresh!(P::LUPreconditioner, x, p)
    P.J!(P.J, x, p)
    P.fac = lu(P.J)
    P.refreshes += 1
    x == P.start && (P.start_refreshes += 1)
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

# The operator of a solve may be scaled near `floatmax`, where a product of two of its entries
# overflows, a Householder reflector of one of its columns overflows, and a vector of the size of
# its inverse is subnormal. So every QR factorisation and every least-squares solve below takes a
# matrix divided by its largest entry, the harmonic Ritz pencil is formed from such a matrix, and a
# recycled space `U` is held as unit columns `Û` and the inverses `d` of the norms of the columns
# of `U`, `U = Û Diagonal(1 ./ d)`.

# The `Q` factor and the scaled `R` factor `R / σ` of the QR factorisation of `M`, and `σ`.
function scaled_qr(M)
    σ = maximum(abs, M)
    F = qr(M ./ σ)
    return Matrix(F.Q), F.R, σ
end

# The least-squares solution `z` of `G z ≈ rhs` and its residual `rhs − G z`. The residual is
# formed in the scaled system too: a product `G z` overflows where `z` holds an entry above 1.
function scaled_lsq(G, rhs)
    σ = maximum(abs, G)
    Ĝ = G ./ σ
    r̂ = rhs ./ σ
    z = Ĝ \ r̂
    return z, σ .* (r̂ - Ĝ * z)
end

# The recycled space `U = X (σ R)⁻¹` of an upper-triangular `R` as `(Û, d)`.
function unit_recycled(X, R, σ)
    τ = maximum(abs, R)
    Y = X / UpperTriangular(R ./ τ)
    ny = [norm(c) for c in eachcol(Y)]
    return Y ./ ny', σ * τ ./ ny
end

# Classical Gram–Schmidt with a second pass (CGS2): `w` made orthogonal to the orthonormal columns
# of `basis`, in place, and the coefficients of the two passes summed. One pass leaves `w` with a
# loss of orthogonality of about `eps` times the ratio of `‖w‖` before and after it; the second
# pass takes it back to about `eps`.
function orthogonalize!(w, basis)
    h = basis' * w
    w .-= basis * h
    h2 = basis' * w
    w .-= basis * h2
    return h .+ h2
end

# One cycle from the residual `r` of norm `β`: `nsteps` Arnoldi steps at most, orthogonal to `C`
# where a recycled space `(C, Û, d)` is given, with `A Û Diagonal(1 ./ d) = C`, until the
# least-squares residual reaches `tol`. It updates `y` and `r`, and returns the number of steps
# and the new recycled space `(C, Û, d)` from the harmonic Ritz vectors of the cycle, or `nothing`
# where the cycle spans fewer than `k` vectors. Dividing the Hessenberg matrix of the pencil by its
# largest entry leaves the eigenvectors and the order of the eigenvalues by modulus as they are.
function gcrodr_cycle!(s::GCRODR, op, y, r, β, C, Û, d, nsteps, tol)
    n = length(r)
    kc = C === nothing ? 0 : size(C, 2)
    W = zeros(n, kc + nsteps + 1)
    s.basis = max(s.basis, size(W, 2))
    G = zeros(kc + nsteps + 1, kc + nsteps)
    rhs = zeros(kc + nsteps + 1)
    rhs[kc + 1] = β
    if kc > 0
        W[:, 1:kc] .= C
        G[1:kc, 1:kc] .= Diagonal(d)
    end
    W[:, kc + 1] .= r ./ β
    w = zeros(n)
    j = 0
    while j < nsteps
        j += 1
        apply_operator!(w, op, view(W, :, kc + j))
        s.arnoldi += 1
        G[1:(kc + j), kc + j] .= orthogonalize!(w, view(W, :, 1:(kc + j)))
        hn = norm(w)
        G[kc + j + 1, kc + j] = hn
        hn > 0 && (W[:, kc + j + 1] .= w ./ hn)
        Gj = G[1:(kc + j + 1), 1:(kc + j)]
        _, e = scaled_lsq(Gj, rhs[1:(kc + j + 1)])
        (norm(e) <= tol || hn == 0) && break
    end
    J = kc + j
    Gj = G[1:(J + 1), 1:J]
    Wj = W[:, 1:(J + 1)]
    z, e = scaled_lsq(Gj, rhs[1:(J + 1)])
    V̂ = kc > 0 ? hcat(Û, W[:, (kc + 1):(kc + j)]) : W[:, 1:j]
    y .+= V̂ * z
    r .= Wj * e
    J < s.k && return j, nothing
    Ĝ = Gj ./ maximum(abs, Gj)
    P = harmonic_basis(Ĝ' * Ĝ, Ĝ' * (Wj' * V̂), s.k)
    Q, R, σ = scaled_qr(Gj * P)
    return j, (Wj * Q[:, 1:(s.k)], unit_recycled(V̂ * P, R, σ)...)
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
    C = U = d = nothing
    if s.U !== nothing
        Ũ = s.U
        AU = similar(Ũ)
        for i in axes(Ũ, 2)
            apply_operator!(view(AU, :, i), op, view(Ũ, :, i))
            s.au += 1
        end
        Q, R, σ = scaled_qr(AU)
        C = Q[:, 1:(s.k)]
        U, d = unit_recycled(Ũ, R, σ)
        c = C' * r
        y .+= U * (c ./ d)
        r .-= C * c
        rn = norm(r)
    end
    iters = 0
    while rn > tol && iters < s.maxiter
        nsteps = min(C === nothing ? s.m : s.m - s.k, s.maxiter - iters)
        j, CU = gcrodr_cycle!(s, op, y, r, rn, C, U, d, nsteps, tol)
        iters += j
        CU === nothing || ((C, U, d) = CU)
        rn = norm(r)
    end
    U === nothing || (s.U = U)
    return y, rn > tol
end

# ─── The Newton loops ──────────────────────────────────────────────────────────

# The rule of the preconditioner of `Newton(; reuse, keep_factor)` with a preconditioned `GMRES`,
# whose state a sequence of solves shares: the period `reuse`, whether a solve keeps the
# preconditioner for the next one (`keep`), the age of the preconditioner, whether the next step
# refreshes it early (`stalled`), and whether the last step stepped along a preconditioner that an
# earlier step formed (`reused`).
mutable struct PreconditionerRule
    reuse::Int
    keep::Bool
    age::Int
    stalled::Bool
    reused::Bool
end

function PreconditionerRule(reuse::Integer; keep::Bool)
    PreconditionerRule(reuse, keep, 0, false, false)
end

function drop!(rule::PreconditionerRule)
    rule.age, rule.stalled, rule.reused = 0, false, false
    return rule
end

# A Newton solve of `F!(r, x, p) = 0` from `x` with GCRO-DR, in the steps of
# `Newton(; reuse, keep_factor)` with a preconditioned `GMRES`, `FixedForcing(η)` and the default
# `Options`: the inner tolerance `η ‖F‖` and the full step. A step refreshes `P` where the rule's
# age is 0 or the rule is `stalled`; after the step the age counts on, or starts at 1 after a
# refresh, and returns to 0 after `reuse` steps; `stalled` is set by an inner solve that reached
# its cap and by a step along a reused preconditioner whose rate `θ = ‖d_k‖ / ‖d_{k−1}‖` is at
# least 1. The solve ends with `SUCCESS` on the residual test `‖F‖ ≤ √eps ‖F(x₀)‖` or on the rate
# test `θ / (1 − θ) ‖d‖ ≤ κ x_reltol ‖x‖`, with `θ` the larger rate of the step and of the one
# before it, `κ = 1/10` and `x_reltol = 2 eps`; it ends unconverged on the step test
# `‖d‖ ≤ 2 eps ‖x‖`, on a zero step, and after 100 steps. A rule that keeps its preconditioner
# keeps it for the next solve only after `SUCCESS` in one step or with a last rate `θ ≤ 10⁻³`;
# otherwise, and at the start of a solve under a rule that does not keep, the age is 0, so that the
# next step refreshes. `P = nothing` takes no preconditioner and no rule. It writes the last
# iterate into `x` and returns whether the solve converged and the number of steps.
function gcrodr_newton!(s::GCRODR, F!, J!, Jbuf, P, x, p, η; rule = nothing)
    op = CountingOperator(Jbuf, P, zeros(length(x)), 0)
    counted = products(s)
    F = similar(x)
    F!(F, x, p)
    fnorm = norm(F)
    tol = sqrt(eps(Float64)) * fnorm
    rule === nothing || rule.keep || drop!(rule)
    steps, converged = 0, fnorm <= tol
    rate, prevnorm = NaN, 0.0
    while !converged && steps < 100
        refresh = P !== nothing && (rule.age == 0 || rule.stalled)
        refresh && lu_refresh!(P, x, p)
        J!(Jbuf, x, p)
        y, capped = gcrodr_solve!(s, op, -F, η * fnorm)
        d = similar(x)
        precondition!(d, P, y)
        x .+= d
        F!(F, x, p)
        fnorm = norm(F)
        steps += 1
        dnorm = norm(d)
        prevrate = rate
        rate = prevnorm > 0 ? dnorm / prevnorm : NaN
        prevnorm = dnorm
        if P !== nothing
            rule.age = refresh ? (rule.reuse == 1 ? 0 : 1) :
                       (rule.age + 1 >= rule.reuse ? 0 : rule.age + 1)
            rule.reused = !refresh
            rule.stalled = capped || (rule.reused && rate >= 1)
        end
        converged = fnorm <= tol
        converged && break
        xnorm = norm(x)
        θ = max(rate, prevrate)
        converged = dnorm > 0 && θ < 1 &&
                    θ / (1 - θ) * dnorm <= 0.1 * 2 * eps(Float64) * xnorm
        (converged || dnorm <= 2 * eps(Float64) * xnorm) && break
    end
    keeps = converged && (steps == 1 || rate <= 1e-3)
    rule === nothing || keeps || drop!(rule)
    counted = products(s) - counted
    counted == op.calls || error(
        "GCRO-DR counted $counted products, the operator was called $(op.calls) times")
    return converged, steps
end

# The ten solves of one column along `ps`, each from the root before, starting from `x0`.
# `column` is `:gmres`, `:norec` or `:rec`; a preconditioned column takes the rule of
# `Newton(; reuse, keep_factor)`. Returns the products, whether every solve converged, the Newton
# steps, the refreshes of the preconditioner, the solves that started on a kept preconditioner,
# and the linear solves that started from a recycled space. `maxiter` caps the inner iterations of
# each linear solve, and the `restart` of `GMRES` at `min(m, maxiter)`.
function run_column(column::Symbol, prob, x0, ps; m, k, η, preconditioned::Bool, reuse,
        keep_factor::Bool = true, maxiter = RC_MAXITER)
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
                 Newton(; linsolve = g, reuse, keep_factor, jacobian = JVPFunction(jvp!)) :
                 Newton(; linsolve = g, jacobian = JVPFunction(jvp!))
        solver = init(prob, x, method, first(ps))
        count = 0
        for p in ps
            P === nothing || copyto!(P.start, x)
            _, status = solve!(solver, x, p)
            count += jvp_calls(solver)
            converged &= status.code == GeometricSolvers.SUCCESS
            steps += status.iterations
        end
        return (products = count, converged, steps, refreshes = refreshes(P),
            kept = kept_starts(P, ps), from_u = 0)
    end
    s = GCRODR(m, k; recycle = column === :rec, maxiter)
    rule = preconditioned ? PreconditionerRule(reuse; keep = keep_factor) : nothing
    for p in ps
        P === nothing || copyto!(P.start, x)
        ok, n = gcrodr_newton!(s, prob.F, prob.J, Jbuf, P, x, p, η; rule)
        converged &= ok
        steps += n
    end
    return (products = products(s), converged, steps, refreshes = refreshes(P),
        kept = kept_starts(P, ps), from_u = s.au ÷ s.k)
end

refreshes(::Nothing) = 0
refreshes(P::LUPreconditioner) = P.refreshes

# The solves that started on a kept preconditioner: those whose first step refreshed nothing.
kept_starts(::Nothing, ps) = 0
kept_starts(P::LUPreconditioner, ps) = length(ps) - P.start_refreshes

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

"""
    run_cell(N, m, k, η; keep_factor = true)

One cell of the grid: the three columns of `N`, `m`, `k` and `η`, each with the preconditioner
of `Newton(; reuse = RC_REUSE, keep_factor)`.
"""
function run_cell(N, m, k, η; keep_factor::Bool = true)
    prob, _, z0, _, _ = sparse_bracket_problem(N, Float64)
    cols = map((:gmres, :norec, :rec)) do c
        run_column(c, prob, z0, RC_PS; m, k, η, preconditioned = true, reuse = RC_REUSE,
            keep_factor)
    end
    return (N = N, m = m, k = k, η = η, gmres = cols[1], norec = cols[2], rec = cols[3])
end

# A cell in which no linear solve of GCRO-DR with recycling starts from a recycled space: its
# recycling column measures GCRO-DR without recycling.
vacuous(row) = row.rec.from_u == 0

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
        " | ", three(row, c -> c.kept), " | ", row.rec.from_u, " | ",
        vacuous(row) ? "vacuous" : "", " |")
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

The verdict of the rows of a grid: `rule` is `:build`, `:stop`, `:between`, `:vacuous` on the
full grid where every cell is vacuous, or `:not_evaluated` on a grid other than the full one;
`pairs` holds the pairs `(m, k)` that meet the build rule, and `vacuous` the number of vacuous
cells.
"""
function rc_verdict(rows)
    nvac = count(vacuous, rows)
    full_grid(rows) ||
        return (rule = :not_evaluated, pairs = Tuple{Int, Int}[], vacuous = nvac)
    nvac == length(rows) &&
        return (rule = :vacuous, pairs = Tuple{Int, Int}[], vacuous = nvac)
    pairs = [(m, k) for m in RC_MS
             for k in RC_KS
             if all(build_cell, filter(r -> (r.m, r.k) == (m, k), rows))]
    rule = !isempty(pairs) ? :build : all(stop_cell, rows) ? :stop : :between
    return (rule = rule, pairs = pairs, vacuous = nvac)
end

function print_verdict(io::IO, rows, v)
    println(
        io, "vacuous cells (no linear solve starts from a recycled space U): ", v.vacuous,
        " of ", length(rows))
    if v.rule === :not_evaluated
        println(io, "build rule: not evaluated (the grid is not the full grid)")
        println(io, "stop rule: not evaluated (the grid is not the full grid)")
        return nothing
    end
    if v.rule === :vacuous
        println(io, "build rule: not evaluated (every cell is vacuous)")
        println(io, "stop rule: not evaluated (every cell is vacuous)")
        println(io, "result: the further spike measures nothing")
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
        "Newton steps | refreshes | kept starts | solves from U | cell |")
    println(io, "|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|:--|--:|:--|")
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
