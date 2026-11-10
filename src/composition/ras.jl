# The restricted additive Schwarz (RAS) map of one-level nonlinear preconditioning: the block
# partition it sweeps, the factor of each block Jacobian, the restricted prolongation, and the
# counters of the sweep.
#
# A block residual `F_i!(f_i, x_i, x, p)` writes the residual of block `i` on its own index set
# into `f_i`, reading external couplings from the frozen full iterate `x`. A sweep restricts
# `x_i = R_i x`, takes up to `max_iterations` local Newton steps with the package's own factor, and
# prolongates through the owner sets `S̃_i` only, so that every index of `1:n` is written once and
# the map is the restricted form (`CaiKeyes:2002`, `Dolean:2016`, `CaiLi:2011`).

"""
    BlockPartition(subdomains, owners)
    BlockPartition(n, nblocks; overlap = 1)

The index sets of a one-level Schwarz method over `1:n`: one subdomain `S_i` per block, which
holds the index set of the local solve, and one owner set `S̃_i ⊆ S_i` per block, which holds the
indices that the block writes back. The owner sets are pairwise disjoint and cover `1:n` exactly,
so a [`RAS`](@ref) sweep writes each index once and the map is the restricted form. A subdomain is
non-empty and repeats no index, and every index is in `1:n`, with `n` the largest index of any
subdomain; any other input raises an `ArgumentError`. The field `local_owners[i]` holds the
positions of `owners[i]` in `subdomains[i]`, so that `subdomains[i][local_owners[i]] == owners[i]`.

`BlockPartition(n, nblocks; overlap = 1)` splits `1:n` into `nblocks` contiguous owner sets of
equal or nearly equal size, and extends each into its subdomain by `overlap` indices on each side,
clipped to `1:n`. `overlap = 0` gives `subdomains == owners`, which is block Jacobi; `n < 1`,
`nblocks < 1`, `nblocks > n` and `overlap < 0` raise an `ArgumentError`.
"""
struct BlockPartition{S, O, L}
    n::Int
    subdomains::S
    owners::O
    local_owners::L
end

function BlockPartition(subdomains, owners)
    length(subdomains) == length(owners) || throw(ArgumentError(
        "BlockPartition needs one owner set per subdomain"))
    isempty(subdomains) && throw(ArgumentError("BlockPartition needs at least one block"))
    n = 0
    for S in subdomains
        isempty(S) && throw(ArgumentError("a BlockPartition subdomain is empty"))
        length(unique(S)) == length(S) ||
            throw(ArgumentError("a BlockPartition subdomain repeats an index"))
        for i in S
            i >= 1 || throw(ArgumentError("a BlockPartition index is below 1"))
            n = max(n, Int(i))
        end
    end
    covered = falses(n)
    for (i, O) in enumerate(owners)
        isempty(O) && throw(ArgumentError("a BlockPartition owner set is empty"))
        length(unique(O)) == length(O) ||
            throw(ArgumentError("a BlockPartition owner set repeats an index"))
        S = subdomains[i]
        for o in O
            1 <= o <= n || throw(ArgumentError("a BlockPartition index is outside 1:n"))
            o in S || throw(ArgumentError(
                "a BlockPartition owner index is not in its subdomain"))
            covered[o] && throw(ArgumentError("BlockPartition owner sets overlap"))
            covered[o] = true
        end
    end
    all(covered) ||
        throw(ArgumentError("BlockPartition owner sets do not cover 1:n"))
    subs = [Int[i for i in S] for S in subdomains]
    owns = [Int[o for o in O] for O in owners]
    local_owners = Vector{Vector{Int}}(undef, length(owns))
    for i in eachindex(owns)
        local_owners[i] = Int[findfirst(==(o), subs[i]) for o in owns[i]]
    end
    return BlockPartition(n, subs, owns, local_owners)
end

function BlockPartition(n::Integer, nblocks::Integer; overlap::Integer = 1)
    n >= 1 || throw(ArgumentError("BlockPartition needs n >= 1"))
    nblocks >= 1 || throw(ArgumentError("BlockPartition needs nblocks >= 1"))
    nblocks <= n || throw(ArgumentError("BlockPartition needs nblocks <= n"))
    overlap >= 0 || throw(ArgumentError("BlockPartition overlap must be >= 0"))
    owners = [collect((((i - 1) * n) ÷ nblocks + 1):((i * n) ÷ nblocks)) for i in 1:nblocks]
    subdomains = [collect(max(1, first(O) - overlap):min(n, last(O) + overlap))
                  for O in owners]
    return BlockPartition(subdomains, owners)
end

"""
    RAS(partition::BlockPartition; max_iterations = 1, rtol = 0, atol = 0,
        balance_threshold = nothing)

The restricted additive Schwarz map of one-level nonlinear preconditioning, as a nonlinear solver
method of [`NonlinearProblem`](@ref) with block residuals: one [`step!!`](@ref) restricts the
iterate to each subdomain, takes up to `max_iterations` local Newton steps with that block's own
factor, and prolongates through the owner sets only, so every index is written once. A sweep
evaluates each block residual once per inner iteration and calls the global residual no times; the
step then evaluates the global residual once at the swept iterate and returns it, so a member of
[`Multiplicative`](@ref) passes the next member a residual of the iterate that the sweep reached.
`LeftPreconditioned(Newton(), RAS(partition))` is RASPEN and
`Multiplicative(RAS(partition), Newton())` is NKS-RAS.

An inner iteration of block `i` stops before its linear solve when the block residual norm is at
most `atol + rtol * ‖f_i⁰‖`, with `f_i⁰` the block's residual at its first inner iteration. With a
positive tolerance (`atol > 0` or `rtol > 0`), a local solve that takes all `max_iterations` steps
is converged when its last Newton step ``\\delta`` is at the rounding level of its iterate,
``\\|\\delta\\| \\le \\varepsilon \\|x_i\\|`` with ``\\varepsilon`` the `eps` of the element type,
where the residual can fall no further; otherwise it evaluates its block residual once more, and
where that norm is above the tolerance the local solve has not converged: the step returns the
iterate unchanged with `MAXITERS`, which ends the solve, and [`ras_counts`](@ref) names the block
in `failed_block`. With `atol = rtol = 0` every local
solve takes `max_iterations` steps and makes no claim of convergence. With
`balance_threshold = ρ` the step returns the iterate unchanged, as a `SUCCESS` of zero norm, when
``\\max_i \\|R_i F(x)\\| \\le \\rho \\|F(x)\\|`` (`CaiLi:2011`); the block residuals cost one call
per block in that test, and the global residual costs none. [`LeftPreconditioned`](@ref) refuses a
`RAS` with a `balance_threshold`.

The counters of the last [`start!!`](@ref) are read with [`ras_counts`](@ref). A problem without
block residuals, a partition whose block count or index count differs from the problem's, and an
iterate that is not a vector raise an `ArgumentError` at [`init_state`](@ref); the partition's
validation raises one at construction.
"""
struct RAS{P, R <: Real, B <: Union{Nothing, Real}}
    partition::P
    max_iterations::Int
    rtol::R
    atol::R
    balance_threshold::B
    # `adapt` rebuilds a `RAS` through this constructor, inside a device kernel too, so the checks
    # also refuse a value that `ToReal` narrows to zero, to `Inf` or to `NaN`. The messages are
    # constant: a kernel cannot format one.
    function RAS{P, R, B}(partition::P, max_iterations::Int, rtol::R, atol::R,
            balance_threshold::B) where {P, R <: Real, B <: Union{Nothing, Real}}
        max_iterations >= 1 ||
            throw(ArgumentError("RAS max_iterations must be at least 1"))
        (isfinite(rtol) && rtol >= zero(R)) ||
            throw(ArgumentError("RAS rtol must be finite and >= 0"))
        (isfinite(atol) && atol >= zero(R)) ||
            throw(ArgumentError("RAS atol must be finite and >= 0"))
        if balance_threshold !== nothing
            (isfinite(balance_threshold) && zero(R) < balance_threshold <= one(R)) ||
                throw(ArgumentError("RAS balance_threshold must be finite and in (0, 1]"))
        end
        return new{P, R, B}(partition, max_iterations, rtol, atol, balance_threshold)
    end
end

function RAS(partition::BlockPartition; max_iterations::Integer = 1, rtol::Real = 0,
        atol::Real = 0, balance_threshold::Union{Nothing, Real} = nothing)
    r = float(rtol)
    a = float(atol)
    R = promote_type(typeof(r), typeof(a))
    b = balance_threshold
    rb = b === nothing ? nothing : convert(R, float(b))
    B = rb === nothing ? Nothing : typeof(rb)
    return RAS{typeof(partition), R, B}(
        partition, Int(max_iterations), convert(R, r), convert(R, a), rb)
end

# The positional form that `adapt` rebuilds a method through: the fields as they stand, in the
# element type that `ToReal` gives them.
function RAS(partition::P, max_iterations::Integer, rtol::R, atol::R,
        balance_threshold::B) where {P, R <: Real, B <: Union{Nothing, Real}}
    return RAS{P, R, B}(partition, Int(max_iterations), rtol, atol, balance_threshold)
end

Adapt.@adapt_structure RAS

# The block residual with its external couplings frozen at the iterate of a sweep, in the
# three-argument call form of a `NonlinearProblem`'s residual: the block without a Jacobian of its
# own takes its factor from `AutoForwardDiff()` on this function. `x` is the state's coupling
# buffer, which every sweep fills from its own iterate before its first block, so a state swept
# from another array differentiates each block at that array's couplings.
struct FrozenBlockResidual{F, X} <: Function
    f::F
    x::X
end

@inline (b::FrozenBlockResidual)(f_i, x_i, p) = b.f(f_i, x_i, b.x, p)

# The counters of a `RASState`, reset by every `start!!`:
#
# - `sweeps`: the number of sweeps run.
# - `block_solves`: the number of block linear solves, summed over the blocks and the sweeps, and —
#   on a state whose block factors a RASPEN preparation reads — one more per block and sweep for
#   the block Jacobian that the sweep forms at the local iterate it returns.
# - `parallel_block_solves`: the block solves of the parallel schedule, one per inner iteration of
#   the slowest block, summed over the sweeps, and — on such a state — one per sweep for the block
#   Jacobian that the slowest block forms at the local iterate its sweep returns.
# - `failed_block`: the block whose local solve did not converge in the last sweep, or `0`.
mutable struct RASCounter
    sweeps::Int
    block_solves::Int
    parallel_block_solves::Int
    failed_block::Int
end

RASCounter() = RASCounter(0, 0, 0, 0)

# The workspace of one block: the index sets as arrays of the iterate's storage, the local
# iterate, residual, Jacobian and step buffers, the subdomain rows of the global Jacobian that a
# RASPEN product reads, and — where the problem names no block Jacobian — the residual with its
# couplings frozen and its AD preparation. `couplings` is the buffer those
# residuals read, filled from the sweep's own iterate; `swept` is the iterate that the step sweeps,
# so that a step leaves the iterate it was given unchanged; `raspen` is set where a RASPEN
# preparation reads the block factors, which is where a sweep forms the factor at the local iterate
# it returns; and `formed` is set by a complete sweep, so that a Jacobian product on a state that no
# sweep formed raises instead of reading a factor nothing has written.
mutable struct RASBlocks{SI, OI, LI, XL, FL, JL, DL, RL, FP, PP, C, X}
    subdomains::SI
    owners::OI
    local_owners::LI
    x_locals::XL
    f_locals::FL
    jacobians::JL
    deltas::DL
    rows::RL
    problems::FP
    preps::PP
    couplings::C
    swept::X
    raspen::Bool
    formed::Bool
end

"""
    RASState

The state of a [`RAS`](@ref) method: the workspace of every block, the factors of the block
Jacobians, which [`raspen_jvp!`](@ref) and [`raspen_jacobian!`](@ref) read, the global Jacobian
buffer and its preparation — `nothing` where the problem has its own `J`, and `AutoForwardDiff()`
on `F` where it has none — the composite iterate that a RASPEN Jacobian product forms per block,
and the counters that [`ras_counts`](@ref) reports. Every block factor is re-formed in
every inner iteration, and once more at the local iterate that the sweep returns where the block
factors feed a RASPEN preparation, which is the factor at the local solution. A factor that no
sweep has written fails closed: the product raises an `ArgumentError` before it reads one.
"""
mutable struct RASState{B, F, V, JP, G}
    blocks::B
    factors::F
    Fx::V
    x₀::V
    dx::V
    jac_prep::JP
    G::G
    composite::V
    counts::RASCounter
end

# `x`'s index sets, on `x`'s storage: an `Array` keeps a `Vector{Int}`, and a device array keeps a
# device index array, so that a gather and a scatter are non-scalar indexing.
function device_indices(x::AbstractVector, I::AbstractVector{Int})
    Ix = similar(x, Int, length(I))
    copyto!(Ix, I)
    return Ix
end

@inline function init_state(m::RAS, prob::NonlinearProblem, x, p)
    x isa AbstractVector ||
        throw(ArgumentError("RAS takes a vector iterate and its problem's block residuals"))
    block_residuals(prob) === nothing && throw(ArgumentError(
        "RAS needs a problem with block residuals: build it with NonlinearProblem(F, J; block_residuals = ...)"))
    part = m.partition
    nblocks = length(part.owners)
    length(block_residuals(prob)) == nblocks || throw(ArgumentError(
        "the BlockPartition and the problem must have one block residual per block"))
    part.n == length(x) || throw(ArgumentError(
        "the BlockPartition's index count must equal the length of the iterate"))
    T = eltype(x)
    T <: Number || throw(ArgumentError("RAS needs an iterate of numbers"))
    subdomains = [device_indices(x, part.subdomains[i]) for i in 1:nblocks]
    owners = [device_indices(x, part.owners[i]) for i in 1:nblocks]
    local_owners = [device_indices(x, part.local_owners[i]) for i in 1:nblocks]
    x_locals = [similar(x, length(part.subdomains[i])) for i in 1:nblocks]
    f_locals = [similar(x, length(part.subdomains[i])) for i in 1:nblocks]
    jacobians = [similar(x, T, length(part.subdomains[i]), length(part.subdomains[i]))
                 for i in 1:nblocks]
    deltas = [similar(x, length(part.subdomains[i])) for i in 1:nblocks]
    rows = [similar(x, T, length(part.subdomains[i]), part.n) for i in 1:nblocks]
    factors = [fail_closed!!(dense_factor(LUFactorization(), prob, x_locals[i], p))
               for i in 1:nblocks]
    couplings = similar(x)
    jacfuns = block_jacobians(prob)
    if jacfuns === nothing
        problems = [NonlinearProblem(FrozenBlockResidual(block_residual(prob, i), couplings))
                    for i in 1:nblocks]
        preps = [prepare_ad(AutoForwardDiff(), problems[i], f_locals[i], x_locals[i], p)
                 for i in 1:nblocks]
    else
        length(jacfuns) == nblocks || throw(ArgumentError(
            "the BlockPartition and the problem must have one block Jacobian per block"))
        problems = nothing
        preps = nothing
    end
    Fx = similar(x)
    P = jacobian_prototype(prob)
    G = P === nothing ? similar(x, T, part.n, part.n) : similar(P, T)
    jac_prep = _has_analytic_jacobian(prob) ? nothing :
               prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
    blocks = RASBlocks(subdomains, owners, local_owners, x_locals, f_locals, jacobians,
        deltas, rows, problems, preps, couplings, similar(x), false, false)
    return RASState(blocks, factors, Fx, similar(x), similar(x), jac_prep, G, similar(x),
        RASCounter())
end

# A block factor that no sweep has formed, with a written pivot vector and zero factors: a read
# that reached it would solve with a zero matrix and give a non-finite direction, never LAPACK's
# unset pivot vector, and a pivot of one is an in-range row for every solve. `_ras_sweep!!` writes
# a formed factor over it in every inner iteration.
@inline function fail_closed!!(fac)
    fill!(fac.factors, zero(eltype(fac.factors)))
    fill!(fac.ipiv, one(eltype(fac.ipiv)))
    return fac
end

# Every solve starts the counters again, so that `ras_counts` reports the solve that ended, and
# leaves no block factor of an earlier solve readable.
@inline function start!!(state::RASState, m::RAS, prob, x, Fx, p, fnorm₀)
    state.counts = RASCounter()
    state.blocks.formed = false
    return state
end

@inline state_counts(state::RASState) = state.counts

@inline function restart_with_counts(state::RASState, counts::RASCounter)
    state.counts = counts
    return state
end

@inline residual_buffer(state::RASState) = state.Fx

# The block Jacobian of block `i`: the problem's own function, or `AutoForwardDiff()` on the block
# residual with its couplings frozen.
@inline function block_jacobian!!(J, blocks::RASBlocks, prob, i, x_loc, x, p)
    f = blocks.problems
    if f === nothing
        block_jacobian(prob, i)(J, x_loc, x, p)
    else
        jacobian!!(J, blocks.preps[i], f[i], x_loc, p)
    end
    return J
end

"""
    ras_prolongate!(dest, x_locals, partition)

Write each block's local iterate back into `dest` at that block's owner set,
`dest[owners[i][k]] = x_locals[i][local_owners[i][k]]`, for the host index sets of `partition`.
Each index of `1:n` is written once, because the owner sets of a [`BlockPartition`](@ref) are
disjoint and cover `1:n`. A [`RAS`](@ref) sweep calls the method that takes the device index arrays
of its state, so a device iterate is never scattered element by element.
"""
function ras_prolongate!(dest, x_locals, partition::BlockPartition)
    return ras_prolongate!(dest, x_locals, partition.owners, partition.local_owners)
end

function ras_prolongate!(dest, x_locals, owners, local_owners)
    for i in eachindex(x_locals)
        dest[owners[i]] .= view(x_locals[i], local_owners[i])
    end
    return dest
end

# One sweep, and the code it ends with: `SINGULAR` where a block's factor reports a zero pivot,
# `NONFINITE` where a block residual is not finite, and `MAXITERS` where a local solve with a
# positive tolerance ends its iterations above it and its last step above the rounding level of
# its iterate, with that block in `failed_block`. A failed
# sweep prolongates nothing, so the iterate it was given is unchanged, and leaves no factor
# readable. The couplings of the frozen block residuals are the iterate of this call, and the
# factor at the returned local iterate is formed only where a RASPEN preparation reads the block
# factors.
function _ras_sweep!!(state::RASState, m::RAS, prob, x, p)
    blocks = state.blocks
    blocks.formed = false
    state.counts.failed_block = 0
    blocks.problems === nothing || copyto!(blocks.couplings, x)
    nblocks = length(blocks.subdomains)
    total = 0
    parallel = 0
    for i in 1:nblocks
        x_loc = blocks.x_locals[i]
        f_i = blocks.f_locals[i]
        J_i = blocks.jacobians[i]
        x_loc .= view(x, blocks.subdomains[i])
        f₀ = zero(real(eltype(x)))
        k = 0
        converged = false
        for it in 1:m.max_iterations
            block_residual(prob, i)(f_i, x_loc, x, p)
            all(isfinite, f_i) || return NONFINITE
            fnorm = rnorm(f_i)
            it == 1 && (f₀ = fnorm)
            if fnorm <= m.atol + m.rtol * f₀
                converged = true
                break
            end
            block_jacobian!!(J_i, blocks, prob, i, x_loc, x, p)
            fac = factorize!!(state.factors[i], LUFactorization(), J_i)
            iszero(fac.info) || return SINGULAR
            state.factors[i] = fac
            x_loc .-= ldiv!!(blocks.deltas[i], fac, f_i)
            k += 1
        end
        # a positive tolerance makes the local solve a claim of convergence, which the residual
        # after its last step decides, unless that step is at the rounding level of the local
        # iterate, `‖δ‖ ≤ eps ‖x_i‖`, where the residual can fall no further; zero tolerances take
        # `max_iterations` steps and claim none
        rounded = k > 0 && rnorm(blocks.deltas[i]) <= eps(real(eltype(x))) * rnorm(x_loc)
        if !converged && !rounded && (m.atol > 0 || m.rtol > 0)
            block_residual(prob, i)(f_i, x_loc, x, p)
            all(isfinite, f_i) || return NONFINITE
            if rnorm(f_i) > m.atol + m.rtol * f₀
                state.counts.failed_block = i
                return MAXITERS
            end
        end
        # the block Jacobian that RASPEN reads is `R_i J(u⁽ⁱ⁾) P_i` at the local iterate this
        # sweep returns, not at the iterate one Newton step before it, so it is formed here
        if blocks.raspen
            block_jacobian!!(J_i, blocks, prob, i, x_loc, x, p)
            fac = factorize!!(state.factors[i], LUFactorization(), J_i)
            iszero(fac.info) || return SINGULAR
            state.factors[i] = fac
        end
        calls = k + (blocks.raspen ? 1 : 0)
        total += calls
        parallel = max(parallel, calls)
    end
    ras_prolongate!(x, blocks.x_locals, blocks.owners, blocks.local_owners)
    blocks.formed = true
    state.counts.sweeps += 1
    state.counts.block_solves += total
    state.counts.parallel_block_solves += parallel
    return SUCCESS
end

"""
    ras_sweep!!(state, m::RAS, prob, x, p)

Run one restricted additive Schwarz sweep from the iterate `x`, overwrite `x` with the swept
iterate, and return it. The sweep restricts each subdomain, evaluates that block's residual once
per inner iteration, solves the block system with the block's own factor, and prolongates through
the owner sets. It calls the global residual no times. A block factor that reports a zero pivot,
a block residual with a non-finite entry, and a local solve with a positive tolerance that does
not converge leave `x` unchanged; the [`step!!`](@ref) of `RAS` reports them as `SINGULAR`,
`NONFINITE` and `MAXITERS`.
"""
function ras_sweep!!(state::RASState, m::RAS, prob, x, p)
    _ras_sweep!!(state, m, prob, x, p)
    return x
end

# The balance test of `CaiLi:2011`: with `balance_threshold = ρ`, the sweep runs only while
# `max_i ‖R_i F(x)‖ > ρ ‖F(x)‖`. It reads the global residual the step was given and never
# evaluates it again.
function ras_balanced(state::RASState, m::RAS, prob, x, Fx, p)
    ρ = m.balance_threshold
    ρ === nothing && return true
    blocks = state.blocks
    R = real(eltype(x))
    largest = zero(R)
    for i in eachindex(blocks.subdomains)
        x_loc = blocks.x_locals[i]
        f_i = blocks.f_locals[i]
        x_loc .= view(x, blocks.subdomains[i])
        block_residual(prob, i)(f_i, x_loc, x, p)
        largest = max(largest, rnorm(f_i))
    end
    return largest > ρ * rnorm(Fx)
end

# One step: the sweep runs on the state's own copy of `x`, so the iterate the step was given is
# unchanged and the swept iterate it returns is its own buffer. A skipped or failed sweep returns
# the iterate and its residual as they were.
@inline function step!!(state::RASState, m::RAS, prob, x, Fx, p)
    R = real(eltype(x))
    x₊ = state.blocks.swept
    copy!!(x₊, x)
    if !ras_balanced(state, m, prob, x, Fx, p)
        return x, Fx, state,
        StepInfo{R}(SUCCESS, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    copy!!(state.x₀, x)
    code = _ras_sweep!!(state, m, prob, x₊, p)
    if code != SUCCESS
        return x, Fx, state, StepInfo{R}(code, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    F₊ = residual!!(state.Fx, prob, x₊, p)
    stepnorm = rnorm(map!!(-, state.dx, state.x₀, x₊))
    return x₊, F₊, state, StepInfo{R}(SUCCESS, rnorm(F₊), stepnorm, Int32(0), false, false)
end

"""
    ras_counts(state)

The counters of the last solve of a [`RAS`](@ref) state, as a named tuple:

- `sweeps`: the sweeps run.
- `parallel_block_solves`, `parallel_inner_iterations`, `inner_iterations`: the block solves of the
  parallel schedule — one per inner iteration of the slowest block of each sweep, summed over the
  sweeps — and, where a RASPEN preparation reads the block factors of the state, one more per
  sweep for the block Jacobian that each block forms at the local iterate its sweep returns.
- `total_inner_iterations`, `block_solves`: every block solve, summed over the blocks and the
  sweeps, with the same one more per block on such a state.
- `failed_block`: the block whose local solve did not converge in the last sweep, which made the
  step return `MAXITERS`, or `0`.

[`start!!`](@ref) resets them, so a second [`solve!`](@ref) on one solver reports its own solve. A
state of [`LeftPreconditioned`](@ref) with an inner [`RAS`](@ref), and a state of
[`Multiplicative`](@ref) with a `RAS` member, report that member's counters.
"""
function ras_counts(state::RASState)
    c = state.counts
    return (; sweeps = c.sweeps,
        parallel_block_solves = c.parallel_block_solves,
        parallel_inner_iterations = c.parallel_block_solves,
        total_inner_iterations = c.block_solves,
        inner_iterations = c.parallel_block_solves,
        block_solves = c.block_solves,
        failed_block = c.failed_block)
end

"""
    raspen_jvp!(Jv, state, prob, x, v, p)

The Jacobian-vector product of the left-preconditioned residual
``r^\\ell(y) = y - N(y)`` of the exact RASPEN Jacobian,
``J_{r^\\ell}(x) v = \\sum_{i=1}^M \\tilde{R}_i^\\top (R_i J(u^{(i)}) P_i)^{-1} R_i J(u^{(i)}) v``
(`Dolean:2016`, eq. 3.17), with the block factors that the last sweep of `state` formed and
``u^{(i)}`` the iterate `x` with the local solve `x_locals[i]` of block `i` written over its
subdomain ``S_i``. Per block it evaluates the global Jacobian at ``u^{(i)}``, keeps the rows of
``S_i``, multiplies them by `v`, solves with that block's factor, and prolongates through the owner
set. The global Jacobian is the problem's own `J` where it has one, so a residual that takes no
`ForwardDiff.Dual` serves RASPEN, and `AutoForwardDiff()` on `F` where it has none. `Jv` is
overwritten and returned. A state that no complete sweep formed has no block factor to read, so it
raises an `ArgumentError`.
"""
function raspen_jvp!(Jv, state::RASState, prob, x, v, p)
    check_formed(state)
    blocks = state.blocks
    for i in eachindex(blocks.subdomains)
        r_i = mul!(blocks.f_locals[i], raspen_rows!(state, prob, x, p, i), v)
        δ = ldiv!!(blocks.deltas[i], state.factors[i], r_i)
        Jv[blocks.owners[i]] .= view(δ, blocks.local_owners[i])
    end
    return Jv
end

# The rows `R_i J(u⁽ⁱ⁾)` of block `i`, written into the block's `rows` buffer: the global Jacobian
# at the composite iterate, `x` with block `i`'s local solve over its subdomain, restricted to the
# subdomain's rows. They hold the couplings outside the block, which the block Jacobian does not.
function raspen_rows!(state::RASState, prob, x, p, i)
    blocks = state.blocks
    u = copy!!(state.composite, x)
    u[blocks.subdomains[i]] .= blocks.x_locals[i]
    G = jacobian!!(state.G, state.jac_prep, prob, u, p)
    B = blocks.rows[i]
    B .= view(G, blocks.subdomains[i], :)
    return B
end

# A block factor of a `RASState` is a formed factor only after a complete sweep, so a product that
# ran before one would hand LAPACK a factor nothing has written.
@inline function check_formed(state::RASState)
    state.blocks.formed || throw(ArgumentError(
        "a RASPEN Jacobian product needs the block factors of a complete RAS sweep: no sweep of this state has formed them, and a sweep that `balance_threshold` skips forms none"))
    return nothing
end

"""
    raspen_jacobian!(J, state, prob, x, p)

The Jacobian of the left-preconditioned residual ``r^\\ell`` at `x` of the exact RASPEN form of
[`raspen_jvp!`](@ref). Per block it evaluates the global Jacobian once, at ``u^{(i)}``, and solves
all columns of its subdomain rows with that block's one factor, so a call takes one global Jacobian
per block. It overwrites `J` and returns it. A state that no complete sweep formed raises an
`ArgumentError`.
"""
function raspen_jacobian!(J, state::RASState, prob, x, p)
    check_formed(state)
    blocks = state.blocks
    for i in eachindex(blocks.subdomains)
        B = raspen_rows!(state, prob, x, p, i)
        r_i = blocks.f_locals[i]
        for j in axes(J, 2)
            r_i .= view(B, :, j)
            δ = ldiv!!(blocks.deltas[i], state.factors[i], r_i)
            view(J, blocks.owners[i], j) .= view(δ, blocks.local_owners[i])
        end
    end
    return J
end
