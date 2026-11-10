# Left preconditioning and the restricted additive Schwarz map in R3: the block-residual
# interface of `NonlinearProblem` and `BlockResidual`, `BlockPartition` and `RAS` (restricted
# prolongation, call counts, the exact RASPEN Jacobian product, the NKS-RAS composition and the
# parallel counters), `LeftPreconditioned` with its merit on `rˡ`, the decided edges, and the P8
# stiff cubic reaction-diffusion problem.

using Adapt: Adapt
using ADTypes: AutoForwardDiff
using GeometricSolvers
using GeometricSolvers: BlockPartition, BlockResidual, LeftPreconditioned,
                        LeftPreconditionedProblem, LeftPreconditionedState,
                        Multiplicative, NONFINITE, Newton, NonlinearProblem, Picard, RAS,
                        RASState, SINGULAR, SUCCESS, StepInfo, block_jacobians,
                        block_residual, block_residuals, init_state, jacobian!!, jvp!!,
                        norm2, prepare_ad, ras_counts,
                        ras_prolongate!, ras_sweep!!, raspen_jacobian!, raspen_jvp!,
                        residual!!, residual_buffer, rnorm, solve, solve!, start!!, step!!
using JLArrays: JLArray
using GPUArraysCore: allowscalar
using LinearAlgebra: cond, dot, mul!, norm
using Random: Xoshiro
using StaticArrays: SVector

include("../helpers/matrix.jl")

allowscalar(false)

# --- the storage helpers: one problem definition for an `Array` and a `JLArray` ---------------

function storage_matrix(x, M::AbstractMatrix)
    copyto!(similar(x, eltype(x), size(M, 1), size(M, 2)), M)
end
storage_vector(x, v::AbstractVector) = copyto!(similar(x, eltype(x), length(v)), v)
storage_indices(x, I::AbstractVector{<:Integer}) = copyto!(similar(x, Int, length(I)), I)

# --- the affine test system: symmetric tridiagonal, 4 on the diagonal and -1 beside it ---------

function affine_matrix(::Type{T}, n::Integer) where {T}
    A = zeros(T, n, n)
    for j in 1:n
        A[j, j] = T(4)
        if j > 1
            A[j, j - 1] = -one(T)
            A[j - 1, j] = -one(T)
        end
    end
    return A
end

affine_rhs(::Type{T}, n::Integer) where {T} = T[T(sin(j)) for j in 1:n]

# The block functions of the affine system: the residual of block `i` on its subdomain, with the
# couplings outside it frozen in `x`, and the block Jacobian.
struct AffineBlocks{M, R, I, W}
    rows::M
    local_jacobians::M
    rhs::R
    subdomains::I
    work::W
end

function affine_blocks(x, part::BlockPartition, A, b)
    nblocks = length(part.owners)
    subdomains = [storage_indices(x, part.subdomains[i]) for i in 1:nblocks]
    rows = [storage_matrix(x, A[part.subdomains[i], :]) for i in 1:nblocks]
    local_jacobians = [storage_matrix(x, A[part.subdomains[i], part.subdomains[i]])
                       for i in 1:nblocks]
    rhs = [storage_vector(x, b[part.subdomains[i]]) for i in 1:nblocks]
    return AffineBlocks(rows, local_jacobians, rhs, subdomains, similar(x))
end

function affine_block(ab::AffineBlocks, i, counters)
    return function (f, x_loc, x, p)
        counters === nothing || (counters[i][] += 1)
        ab.work .= x
        ab.work[ab.subdomains[i]] .= x_loc
        mul!(f, ab.rows[i], ab.work)
        f .-= ab.rhs[i]
        return f
    end
end

function affine_block_jacobian(ab::AffineBlocks, i)
    (J, x_loc, x, p) -> (copyto!(J, ab.local_jacobians[i]); J)
end

function affine_global(x, A, b)
    Ad = storage_matrix(x, A)
    bd = storage_vector(x, b)
    F! = (r, y, p) -> (mul!(r, Ad, y); r .-= bd; nothing)
    J! = (J, y, p) -> (copyto!(J, Ad); J)
    return F!, J!
end

function affine_problem(x, A, b, part::BlockPartition; counters = nothing, jacobians = true)
    F!, J! = affine_global(x, A, b)
    ab = affine_blocks(x, part, A, b)
    nblocks = length(part.owners)
    residuals = ntuple(i -> affine_block(ab, i, counters), nblocks)
    block_jacobians = jacobians ?
                      ntuple(i -> affine_block_jacobian(ab, i), nblocks) : nothing
    return NonlinearProblem(F!, J!; block_residuals = residuals,
        block_jacobians = block_jacobians),
    ab
end

# One started step of `method` from `(x, Fx)`, through the loop's own hooks.
function one_step(prob, method, x, Fx)
    s = init_state(method, prob, x, nothing)
    s = start!!(s, method, prob, x, Fx, nothing, rnorm(Fx))
    x₊, Fx₊, _, info = step!!(s, method, prob, x, Fx, nothing)
    return copy(x₊), copy(Fx₊), info
end

# A coupled nonlinear problem whose block Jacobian depends on the couplings outside its block,
# `F_j(u) = u_j + u_j (u_{j-1}² + u_{j+1}²) - 1`, its block residuals and its block Jacobians: the
# problem that tells a block Jacobian differentiated at the sweep's couplings from one
# differentiated at a stale array.
function coupled_block(u, j, n)
    l = j == 1 ? zero(eltype(u)) : u[j - 1]
    r = j == n ? zero(eltype(u)) : u[j + 1]
    return u[j] + u[j] * (l^2 + r^2) - one(eltype(u))
end

function coupled_problem(::Type{T}, part::BlockPartition; jacobians::Bool) where {T}
    n = part.n
    global_ = (r, x, p) -> begin
        for j in 1:n
            r[j] = coupled_block(x, j, n)
        end
        nothing
    end
    residuals = ntuple(length(part.owners)) do i
        S = part.subdomains[i]
        (f, x_loc, x, p) -> begin
            u = Vector{promote_type(eltype(x_loc), eltype(x))}(x)
            u[S] .= x_loc
            for (k, j) in enumerate(S)
                f[k] = coupled_block(u, j, n)
            end
            f
        end
    end
    block_jacobians = ntuple(length(part.owners)) do i
        S = part.subdomains[i]
        (J, x_loc, x, p) -> begin
            u = copy(x)
            u[S] .= x_loc
            fill!(J, zero(T))
            for (k, j) in enumerate(S)
                l = j == 1 ? zero(T) : u[j - 1]
                r = j == n ? zero(T) : u[j + 1]
                J[k, k] = one(T) + l^2 + r^2
                k > 1 && (J[k, k - 1] = 2u[j] * l)
                k < length(S) && (J[k, k + 1] = 2u[j] * r)
            end
            J
        end
    end
    return NonlinearProblem(global_; block_residuals = residuals,
        block_jacobians = jacobians ? block_jacobians : nothing)
end

# The direct residual `rˡ(x) = x - N(x)` of the inner map `N`, computed through a `LeftPreconditioned`
# method's own problem.
function rl_direct(lprob::LeftPreconditionedProblem, x)
    y = similar(x)
    residual!!(y, lprob, x, nothing)
    return y
end

# --- the block-residual interface ------------------------------------------------------------

@testset "BlockResidual and the block accessors" begin
    F! = (r, x, p) -> (r .= x; nothing)
    J! = (J, x, p) -> (copyto!(J, one.(J)); J)
    f1 = (f, x1, x, p) -> (f .= x1; f)
    j1 = (J1, x1, x, p) -> (copyto!(J1, one.(J1)); J1)

    prob = NonlinearProblem(F!, J!; block_residuals = (f1,), block_jacobians = (j1,))
    @test prob.F isa BlockResidual
    @test prob.J === J!
    @test block_residuals(prob) === (f1,)
    @test block_jacobians(prob) === (j1,)
    @test block_residual(prob, 1) === f1
    @test GeometricSolvers.block_jacobian(prob, 1) === j1

    plain = NonlinearProblem(F!, J!)
    @test block_residuals(plain) === nothing
    @test block_jacobians(plain) === nothing
    @test GeometricSolvers.block_jacobian(plain, 1) === nothing

    # a BlockResidual forwards every argument to the global residual
    let r = zeros(2), x = [1.0, 2.0]
        prob.F(r, x, nothing)
        @test r == x
    end

    @test_throws ArgumentError NonlinearProblem(F!, J!; block_jacobians = (j1,))
    @test_throws ArgumentError NonlinearProblem(F!, J!;
        block_residuals = (f1,), block_jacobians = (j1, j1))
end

@testset "BlockPartition: owners, subdomains and local_owners" begin
    p = BlockPartition(6, 3; overlap = 0)
    @test p.n == 6
    @test p.subdomains == p.owners == [[1, 2], [3, 4], [5, 6]]
    @test p.local_owners == [[1, 2], [1, 2], [1, 2]]

    q = BlockPartition(6, 3; overlap = 1)
    @test q.subdomains == [[1, 2, 3], [2, 3, 4, 5], [4, 5, 6]]
    @test q.owners == [[1, 2], [3, 4], [5, 6]]
    @test q.local_owners == [[1, 2], [2, 3], [2, 3]]
    @test all(i -> q.subdomains[i][q.local_owners[i]] == q.owners[i], 1:3)

    r = BlockPartition([[1, 2, 3], [2, 3, 4]], [[1, 2], [3, 4]])
    @test r.n == 4
    @test r.local_owners == [[1, 2], [2, 3]]

    # the owner sets are disjoint and cover 1:n exactly: each index of 1:n is written once
    @test sort(reduce(vcat, r.owners)) == [1, 2, 3, 4]
    @test length(unique(reduce(vcat, r.owners))) == 4
end

@testset "BlockPartition and RAS validation" begin
    @test_throws ArgumentError BlockPartition([[1, 2], [2, 3]], [[1, 2], [2, 3]])
    @test_throws ArgumentError BlockPartition([[1, 2], [2, 3]], [[1], [3]])
    @test_throws ArgumentError BlockPartition([[1, 2], [2, 3]], [[1, 3], [2]])
    @test_throws ArgumentError BlockPartition([Int[], [1, 2]], [Int[], [1, 2]])
    @test_throws ArgumentError BlockPartition([[1, 1, 2]], [[1, 2]])
    @test_throws ArgumentError BlockPartition([[0, 1]], [[1]])
    # Edge 4: each count guard is asserted by its own anchored message, so that a later guard that
    # refuses the same input cannot stand in for it
    @test_throws ArgumentError("BlockPartition needs n >= 1") BlockPartition(0, 1)
    @test_throws ArgumentError("BlockPartition needs nblocks >= 1") BlockPartition(3, 0)
    @test_throws ArgumentError("BlockPartition needs nblocks <= n") BlockPartition(3, 4)
    @test_throws ArgumentError("BlockPartition overlap must be >= 0") BlockPartition(6, 3;
        overlap = -1)

    part = BlockPartition(4, 2)
    @test_throws ArgumentError RAS(part; max_iterations = 0)
    @test_throws ArgumentError RAS(part; rtol = -1.0)
    @test_throws ArgumentError RAS(part; atol = -1.0)
    @test_throws ArgumentError RAS(part; rtol = Inf)
    @test_throws ArgumentError RAS(part; atol = NaN)
    @test_throws ArgumentError RAS(part; balance_threshold = 0.0)
    @test_throws ArgumentError RAS(part; balance_threshold = 1.5)
    @test_throws ArgumentError RAS(part; balance_threshold = NaN)
    @test RAS(part; balance_threshold = 0.75).balance_threshold == 0.75
    @test RAS(part).balance_threshold === nothing

    # a problem without block residuals, and a partition of another length, are refused at init
    x = [1.0, 2.0, 3.0, 4.0]
    plain = NonlinearProblem((r, y, p) -> (r .= y; nothing))
    @test_throws ArgumentError init_state(RAS(part), plain, x, nothing)
    F!, J! = affine_global(x, affine_matrix(Float64, 4), affine_rhs(Float64, 4))
    prob, _ = affine_problem(x, affine_matrix(Float64, 4), affine_rhs(Float64, 4),
        BlockPartition(4, 2))
    @test_throws ArgumentError init_state(RAS(BlockPartition(6, 3)), prob, x, nothing)
    # a `LeastSquaresProblem` is refused
    lsq = LeastSquaresProblem((r, y, p) -> (r .= y; nothing))
    @test_throws ArgumentError init_state(RAS(part), lsq, x, nothing)
end

# --- Check 1 and Check 2: the restricted map and the call counts -----------------------------

@testset "Checks 1 and 2: restricted prolongation and call counts, $T, $(nameof(B))" for T in REAL_ELTYPES,
    B in ARRAY_BACKENDS

    n = 8
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 3; overlap = 1)
    x0 = B(T[0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.125, -0.125])
    counters = [Ref(0) for _ in 1:length(part.owners)]
    prob, _ = affine_problem(x0, A, b, part; counters = counters)

    global_calls = Ref(0)
    F!, J! = affine_global(x0, A, b)
    prob = NonlinearProblem((r, y, p) -> (global_calls[] += 1; F!(r, y, p)),
        (J, y, p) -> J!(J, y, p); block_residuals = block_residuals(prob),
        block_jacobians = block_jacobians(prob))

    ras = RAS(part; max_iterations = 1)
    x = copy(x0)
    state = init_state(ras, prob, x, nothing)
    state = start!!(
        state, ras, prob, x, residual!!(residual_buffer(state), prob, x, nothing),
        nothing, rnorm(residual!!(similar(x), prob, x, nothing)))

    # Check 2: one sweep calls the global residual no times, and each block residual once
    global_calls[] = 0
    foreach(c -> c[] = 0, counters)
    ras_sweep!!(state, ras, prob, x, nothing)
    @test global_calls[] == 0
    @test all(c -> c[] == 1, counters)

    # Check 1: at the owner indices of block i, the swept entries are block i's local solve
    xh = Array(x)
    xh0 = Array(x0)
    for i in 1:length(part.owners)
        S = part.subdomains[i]
        rhs = A[S, :] * xh0 - b[S]
        local_solution = xh0[S] - A[S, S] \ rhs
        for (k, o) in enumerate(part.owners[i])
            @test xh[o] === local_solution[part.local_owners[i][k]]
        end
    end

    # no index of 1:n is written twice: the owner sets are disjoint and cover 1:n
    written = reduce(vcat, part.owners)
    @test sort(written) == collect(1:n)
    @test length(unique(written)) == n

    # Edge 9: the gather and scatter index arrays have the storage type of the iterate, so that a
    # device iterate is indexed by device index arrays
    Ix = similar(x, Int, 2)
    @test all(I -> I isa typeof(Ix), state.blocks.subdomains)
    @test all(I -> I isa typeof(Ix), state.blocks.owners)
    @test all(I -> I isa typeof(Ix), state.blocks.local_owners)

    # a step of `RAS` returns `F(x₊)` and evaluates the global residual exactly once
    Fx = copy(residual!!(similar(x0), prob, x0, nothing))
    xr = copy(x0)
    global_calls[] = 0
    x₊, Fx₊, info = one_step(prob, ras, xr, Fx)
    @test info.code === SUCCESS
    @test global_calls[] == 1
    @test Array(Fx₊) == A * Array(x₊) - b
    # the step leaves the iterate it was given alone: its sweep runs on the state's own copy
    @test Array(xr) == Array(x0)
end

# --- the AD block Jacobian of a sweep: the couplings of the sweep's own iterate --------------

@testset "The AD block Jacobian reads the couplings of the sweep's own iterate, $T" for T in REAL_ELTYPES
    n = 8
    part = BlockPartition(n, 3; overlap = 1)
    prob_ad = coupled_problem(T, part; jacobians = false)
    prob_j = coupled_problem(T, part; jacobians = true)
    ras = RAS(part; max_iterations = 1)
    x_init = fill(T(0.1), n)
    x_step = T[0.9 + j / 5 for j in 1:n]

    # one sweep from `x_step` on a state built at another array is the same map as the one with
    # explicit block Jacobians, bit for bit to the rounding of the local solve
    function sweep(prob)
        s = init_state(ras, prob, copy(x_init), nothing)
        Fx = residual!!(similar(x_step), prob, x_step, nothing)
        s = start!!(s, ras, prob, x_step, Fx, nothing, rnorm(Fx))
        x₊, _, _, info = step!!(s, ras, prob, x_step, Fx, nothing)
        return copy(x₊), info
    end
    ref, ref_info = sweep(prob_j)
    ad, _ = sweep(prob_ad)
    @test ref_info.code === SUCCESS
    @test norm(ad - ref) <= 8 * eps(T) * norm(ref)

    # the same of the residual `rˡ` that a line search evaluates, at a point other than the one
    # the state was built at
    m = LeftPreconditioned(Newton(), ras)
    st_ad = init_state(m, prob_ad, copy(x_init), nothing)
    st_j = init_state(m, prob_j, copy(x_init), nothing)
    r_ad = residual!!(similar(x_step), st_ad.problem, x_step, nothing)
    r_j = residual!!(similar(x_step), st_j.problem, x_step, nothing)
    @test norm(r_ad - r_j) <= 8 * eps(T) * norm(r_j)

    # a solve agrees with the one on explicit block Jacobians: the two sources are one method
    solve_m = LeftPreconditioned(Newton(; globalization = Backtracking()),
        RAS(part; max_iterations = 1))
    _, s_ad = solve(prob_ad, copy(x_init), solve_m; max_iterations = 100)
    _, s_j = solve(prob_j, copy(x_init), solve_m; max_iterations = 100)
    @test (s_ad.code, s_ad.iterations) == (s_j.code, s_j.iterations)
end

# --- Check 3: the exact RASPEN Jacobian product ----------------------------------------------

@testset "Check 3: the exact RASPEN Jacobian product, $T, $(nameof(B))" for T in REAL_ELTYPES,
    B in ARRAY_BACKENDS

    n = 8
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 3; overlap = 1)
    x0 = B(T[0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.125, -0.125])
    prob, _ = affine_problem(x0, A, b, part)
    ras = RAS(part; max_iterations = 1)
    m = LeftPreconditioned(Newton(), ras)
    x = copy(x0)
    state = init_state(m, prob, x, nothing)
    lprob = state.problem
    # one sweep at `x0` forms the block factors that the product reads
    ras_sweep!!(lprob.state, ras, prob, copy(x0), nothing)
    prep = prepare_ad(AutoForwardDiff(), lprob, similar(x0), x0, nothing)
    @test prep isa GeometricSolvers.RASPenPreparation

    v = B(T[0.7, -1.2, 0.4, 0.9, -0.3, 0.6, -0.8, 0.2])
    Jv = similar(v)
    jvp!!(Jv, prep, lprob, x0, v, nothing)

    vh = Array(v)
    Av = A * vh
    expected = zeros(T, n)
    κ_max = zero(T)
    for i in 1:length(part.owners)
        S = part.subdomains[i]
        z = A[S, S] \ Av[S]
        expected[part.owners[i]] .+= z[part.local_owners[i]]
        κ_max = max(κ_max, cond(A[S, S]))
    end
    atol = 10 * n * κ_max * eps(T) * norm(vh)
    @test norm(Array(Jv) - expected) <= atol

    # the same product one column at a time is the Jacobian of `rˡ`: column `j` is the product
    # along the unit direction `e_j`, so it takes the clause's per-product bound with `‖v‖ = 1`
    J = similar(x0, eltype(x0), n, n)
    jacobian!!(J, prep, lprob, x0, nothing)
    expected_matrix = zeros(T, n, n)
    for i in 1:length(part.owners)
        S = part.subdomains[i]
        expected_matrix[part.owners[i], :] .+= (A[S, S] \ A[S, :])[part.local_owners[i], :]
    end
    column_bound = 10 * n * κ_max * eps(T)
    for j in 1:n
        @test norm(Array(J)[:, j] - expected_matrix[:, j]) <= column_bound
    end
end

# --- Check 4: the merit is on rˡ --------------------------------------------------------------

# A diagonal nonlinear residual, so that one Newton step from `x` is exact and `rˡ` is a function
# of `x` alone.
struct DiagonalResidual{T}
    c::Vector{T}
end

(f::DiagonalResidual)(r, x, p) = (r .= x .^ 2 .- f.c; nothing)

@testset "Check 4: the line-search merit is on rˡ, $T, $(nameof(B))" for T in REAL_ELTYPES,
    B in ARRAY_BACKENDS

    c = T[1.0, 2.0, 3.0, 4.0, 5.0]
    x0 = B(T[1.5, 2.5, 3.5, 4.5, 5.5])
    prob = NonlinearProblem(DiagonalResidual(c))
    m = LeftPreconditioned(Picard(; damping = T(2.0), globalization = Backtracking()),
        Newton())
    m = Adapt.adapt(GeometricSolvers.ToReal{T}(), m)

    x = copy(x0)
    state = init_state(m, prob, x, nothing)
    Fx = residual!!(residual_buffer(state), prob, x, nothing)
    state = start!!(state, m, prob, x, Fx, nothing, rnorm(Fx))

    # the anchor merit is norm2(rˡ(x₀)), not norm2(F(x₀))
    @test state.outer_state.merit === norm2(rl_direct(state.problem, x))
    @test state.outer_state.merit != norm2(residual!!(similar(x), prob, x, nothing))

    x₁, Fx₁, state₁, info = @inferred step!!(state, m, prob, x, Fx, nothing)
    @test x₁ != x0
    # the merit that the search accepted is norm2(rˡ(x₁)): every trial value of the search goes
    # through the same merit function of the preconditioned problem
    @test state₁.outer_state.merit === norm2(rl_direct(state.problem, x₁))
    # the step reports the global residual at the iterate it returns
    @test Array(Fx₁) == Array(x₁) .^ 2 .- c
    @test info.fnorm === rnorm(Array(x₁) .^ 2 .- c)
end

# --- the derivative of a non-RAS inner method: a central difference ---------------------------

@testset "A non-RAS inner is differentiated by a central difference, $T" for T in REAL_ELTYPES
    c = T[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    prob = NonlinearProblem((r, x, p) -> (r .= x .^ 3 .- c; nothing))
    ω = T(1) / 2
    m = LeftPreconditioned(Picard(; damping = ω), Picard(; damping = ω))
    x = T[1.1, 1.9, 3.2, 3.8, 5.3, 5.7]
    state = init_state(m, prob, x, nothing)
    lprob = state.problem
    prep = prepare_ad(AutoForwardDiff(), lprob, similar(x), x, nothing)
    @test prep isa GeometricSolvers.LeftPreconditionedAD
    # `rˡ(x) = x - N(x) = ω F(x)` for the inner Picard step `N(x) = x - ω F(x)`, so the product
    # is `ω · 3x² ⊙ v`, a central difference of `rˡ` to its own truncation and rounding error
    v = T[0.3, -0.7, 0.5, 0.9, -0.2, 0.6]
    Jv = similar(v)
    jvp!!(Jv, prep, lprob, x, v, nothing)
    expected = ω .* 3 .* x .^ 2 .* v
    @test norm(Jv - expected) <= 10 * eps(T)^(2 / 3) * norm(expected)
end

# --- Check 5: NKS-RAS is a composition -------------------------------------------------------

@testset "Check 5: Multiplicative(RAS, Newton) is the sequential form, $T, $(nameof(B))" for T in REAL_ELTYPES,
    B in ARRAY_BACKENDS

    n = 6
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 2; overlap = 1)
    x0 = B(T[0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
    prob, _ = affine_problem(x0, A, b, part)
    ras = RAS(part; max_iterations = 1)
    Fx0 = residual!!(similar(x0), prob, x0, nothing)

    x_r, F_r, _ = one_step(prob, ras, copy(x0), copy(Fx0))
    x_seq, _, _ = one_step(prob, Newton(), x_r, F_r)
    x_mult, _, _ = one_step(prob, Multiplicative(ras, Newton()), copy(x0), copy(Fx0))
    @test Array(x_mult) == Array(x_seq)
end

# --- the decided edges -----------------------------------------------------------------------

@testset "Edge 1: one block is one Newton step, $T" for T in REAL_ELTYPES
    n = 4
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    x0 = T[0.5, -0.5, 0.25, -0.25]
    F!, J! = affine_global(x0, A, b)
    part = BlockPartition([[1, 2, 3, 4]], [[1, 2, 3, 4]])
    prob = NonlinearProblem(F!, J!;
        block_residuals = ((f, x1, x, p) -> (
            mul!(f, storage_matrix(x, A), x); f .-= storage_vector(x, b); f),),
        block_jacobians = ((J1, x1, x, p) -> (copyto!(J1, storage_matrix(x, A)); J1),))
    ras = RAS(part; max_iterations = 1)
    x_r, _, _ = one_step(prob, ras, copy(x0), residual!!(similar(x0), prob, x0, nothing))
    x_n, _, _ = one_step(prob, Newton(), copy(x0), residual!!(similar(x0), prob, x0, nothing))
    @test norm(x_r - x_n) <= 8 * eps(T) * norm(x_n)
end

@testset "Edge 2: a one-index block, $T" for T in REAL_ELTYPES
    n = 4
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    x0 = T[0.5, -0.5, 0.25, -0.25]
    part = BlockPartition([[1], [2, 3], [4]], [[1], [2, 3], [4]])
    prob, _ = affine_problem(x0, A, b, part)
    ras = RAS(part; max_iterations = 1)
    x, _, info = one_step(prob, ras, copy(x0), residual!!(similar(x0), prob, x0, nothing))
    @test info.code === SUCCESS
    # a one-index block solves `x_i - f_i / J_ii` exactly
    f1 = dot(A[1, :], x0) - b[1]
    @test x[1] === x0[1] - f1 / A[1, 1]
end

@testset "Edges 3 and 4: overlap 0, and the invalid partitions" begin
    @test BlockPartition(6, 3; overlap = 0).subdomains ==
          BlockPartition(6, 3; overlap = 0).owners
    @test_throws ArgumentError BlockPartition([[1, 2, 3]], [[1, 2]])
    @test_throws ArgumentError BlockPartition([[1, 2], [3, 4]], [[1, 2], [3, 5]])
end

@testset "Edges 6 and 7: a singular block and a non-finite block, $T, $(nameof(B))" for T in REAL_ELTYPES,
    B in ARRAY_BACKENDS

    n = 4
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 2; overlap = 1)
    x0 = B(T[0.5, -0.5, 0.25, -0.25])
    F!, J! = affine_global(x0, A, b)
    for (kind, code) in ((:singular, SINGULAR), (:nonfinite, NONFINITE))
        singular_jacobian = (J1, x1, x, p) -> (fill!(J1, zero(T)); J1)
        nan_block = (f, x1, x, p) -> (fill!(f, T(NaN)); f)
        ab = affine_blocks(x0, part, A, b)
        residuals = ntuple(
            i -> kind === :nonfinite && i == 1 ?
                 nan_block : affine_block(ab, i, nothing), 2)
        jacobians = ntuple(
            i -> kind === :singular && i == 2 ?
                 singular_jacobian : affine_block_jacobian(ab, i),
            2)
        prob = NonlinearProblem(F!, J!; block_residuals = residuals,
            block_jacobians = jacobians)
        ras = RAS(part; max_iterations = 1)
        Fx = residual!!(similar(x0), prob, x0, nothing)
        x₊, Fx₊, info = one_step(prob, ras, copy(x0), copy(Fx))
        @test info.code === code
        @test Array(x₊) == Array(x0)
        @test Array(Fx₊) == Array(Fx)
    end
end

@testset "Edge 5: init refuses a problem without block residuals and a bad option, $T" for T in REAL_ELTYPES
    x = T[1.0, 2.0, 3.0, 4.0]
    plain = NonlinearProblem((r, y, p) -> (r .= y; nothing))
    @test_throws ArgumentError init(plain, copy(x), RAS(BlockPartition(4, 2)))
    @test_throws ArgumentError RAS(BlockPartition(4, 2); max_iterations = 0)
    @test_throws ArgumentError RAS(BlockPartition(4, 2); atol = -one(T))
end

@testset "MU9: balance_threshold skips the sweep, $T" for T in REAL_ELTYPES
    n = 6
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 3; overlap = 1)
    x0 = T[0.5, -0.5, 0.25, -0.25, 0.75, -0.75]
    plain, _ = affine_problem(x0, A, b, part)
    # the balance test reads the global residual it was given and evaluates it no times (C2b)
    global_calls = Ref(0)
    F!, J! = affine_global(x0, A, b)
    prob = NonlinearProblem((r, y, p) -> (global_calls[] += 1; F!(r, y, p)),
        (J, y, p) -> J!(J, y, p); block_residuals = block_residuals(plain),
        block_jacobians = block_jacobians(plain))
    Fx = residual!!(similar(x0), prob, x0, nothing)
    # each block residual is smaller than the global residual, so rho = 1 skips the sweep
    skipped = RAS(part; max_iterations = 1, balance_threshold = one(T))
    global_calls[] = 0
    x, _, info = one_step(prob, skipped, copy(x0), copy(Fx))
    @test Array(x) == x0
    @test info.code === SUCCESS
    @test iszero(info.stepnorm)
    @test global_calls[] == 0
    # a small threshold runs the sweep, whose step evaluates the global residual once
    run = RAS(part; max_iterations = 1, balance_threshold = T(1e-8))
    global_calls[] = 0
    x₂, _, _ = one_step(prob, run, copy(x0), copy(Fx))
    @test Array(x₂) != x0
    @test global_calls[] == 1
end

@testset "MU10: start!! resets the counters, $T" for T in REAL_ELTYPES
    n = 6
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 3; overlap = 1)
    x0 = T[0.5, -0.5, 0.25, -0.25, 0.75, -0.75]
    prob, _ = affine_problem(x0, A, b, part)
    # a plain `RAS`, and RASPEN, whose inner state is started again at every evaluation of `rˡ`
    for m in (RAS(part; max_iterations = 2),
        LeftPreconditioned(Newton(), RAS(part; max_iterations = 2)))
        solver = init(prob, copy(x0), m)
        solve!(solver)
        first = ras_counts(solver.state)
        @test first.sweeps > 0
        # the same solve from the same start reports the same counters: start!! reset them
        solve!(solver, copy(x0), nothing)
        second = ras_counts(solver.state)
        @test first == second
    end
end

@testset "MU11: the block residuals do not wrap the user's J, $T" for T in REAL_ELTYPES
    n = 4
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    x0 = T[0.5, -0.5, 0.25, -0.25]
    F!, J! = affine_global(x0, A, b)
    part = BlockPartition(n, 2; overlap = 1)
    ab = affine_blocks(x0, part, A, b)
    residuals = ntuple(i -> affine_block(ab, i, nothing), 2)
    jacobians = ntuple(i -> affine_block_jacobian(ab, i), 2)
    prob = NonlinearProblem(F!, J!; block_residuals = residuals,
        block_jacobians = jacobians)
    @test prob.J === J!
    @test prob.F isa BlockResidual
    @test block_residuals(prob) === residuals
end

# A state that no complete sweep formed holds no block factor, so a RASPEN product raises instead
# of handing LAPACK a pivot vector nothing has written.
@testset "A RASPEN product on a state that no sweep formed raises, $T" for T in REAL_ELTYPES
    n = 6
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 3; overlap = 1)
    x0 = T[0.5, -0.5, 0.25, -0.25, 0.75, -0.75]
    prob, _ = affine_problem(x0, A, b, part)
    ras = RAS(part; max_iterations = 1)
    state = init_state(ras, prob, copy(x0), nothing)
    v = T[0.7, -1.2, 0.4, 0.9, -0.3, 0.6]
    Jv = similar(v)
    @test_throws "a RASPEN Jacobian product needs" raspen_jvp!(
        Jv, state, prob, x0, v, nothing)
    @test_throws "a RASPEN Jacobian product needs" raspen_jacobian!(
        similar(x0, T, n, n), state, prob, x0, nothing)
    # the factor a read would have touched fails closed
    @test all(iszero, state.factors[1].factors)
    @test all(==(one(eltype(state.factors[1].ipiv))), state.factors[1].ipiv)
    # one complete sweep forms them, and the product runs
    ras_sweep!!(state, ras, prob, copy(x0), nothing)
    @test all(isfinite, raspen_jvp!(Jv, state, prob, x0, v, nothing))
end

@testset "A sweep skipped inside RASPEN is the zero residual, $T" for T in REAL_ELTYPES
    n = 6
    A = affine_matrix(T, n)
    b = affine_rhs(T, n)
    part = BlockPartition(n, 3; overlap = 1)
    x0 = T[0.5, -0.5, 0.25, -0.25, 0.75, -0.75]
    prob, _ = affine_problem(x0, A, b, part)
    m = LeftPreconditioned(Newton(), RAS(part; max_iterations = 1, balance_threshold = one(T)))
    state = init_state(m, prob, copy(x0), nothing)
    # the balance test skips the sweep at `x0`, so `rˡ(x0) = x0 - N(x0) = 0`
    @test all(iszero, rl_direct(state.problem, x0))
    # a zero preconditioned residual is reported as it is: the solve asks for no block factor
    x, status = solve(prob, copy(x0), m; max_iterations = 20)
    @test Array(x) == x0
    @test status.code === STALLED
end

@testset "LeftPreconditioned refuses an R1 iterate, $T" for T in REAL_ELTYPES
    prob = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- 2; nothing))
    m = LeftPreconditioned(Picard(; damping = one(T) / 2), Newton())
    @test_throws "LeftPreconditioned takes a mutable vector iterate" init_state(
        m, prob, SVector(one(T), one(T)), nothing)
    @test_throws "LeftPreconditioned takes a mutable vector iterate" init_state(
        m, prob, one(T), nothing)
end

# --- Check 6: the P8 benchmark counters -------------------------------------------------------

# The P8 1D stiff cubic reaction-diffusion problem, `-u'' + λ (u³ - u) = 0` on `n` interior points
# with Dirichlet ends, `h = 1 / (n + 1)`, and its block residuals and block Jacobians.
function p8_blocks(::Type{T}, n::Integer, λ, part::BlockPartition) where {T}
    h = inv(T(n + 1))
    d = 2 / h^2
    e = -1 / h^2
    nblocks = length(part.owners)
    S = part.subdomains
    residuals = Vector{Function}(undef, nblocks)
    for i in 1:nblocks
        residuals[i] = let i = i, Si = S[i]
            function (f, x_loc, x, p)
                x_full = copy(x)
                for (k, j) in enumerate(Si)
                    x_full[j] = x_loc[k]
                end
                for (k, j) in enumerate(Si)
                    u = x_full[j]
                    left = j == 1 ? zero(T) : x_full[j - 1]
                    right = j == n ? zero(T) : x_full[j + 1]
                    f[k] = d * u + e * (left + right) - λ * (u^3 - u)
                end
                return f
            end
        end
    end
    jacobians = Vector{Function}(undef, nblocks)
    for i in 1:nblocks
        jacobians[i] = let i = i, Si = S[i]
            function (J, x_loc, x, p)
                fill!(J, zero(T))
                for (k, j) in enumerate(Si)
                    u = x_loc[k]
                    J[k, k] = d - λ * (3 * u^2 - 1)
                    k > 1 && j - 1 == Si[k - 1] && (J[k, k - 1] = e)
                    k < length(Si) && j + 1 == Si[k + 1] && (J[k, k + 1] = e)
                end
                return J
            end
        end
    end
    global_ = let n = n, d = d, e = e, λ = λ
        (r, x, p) -> begin
            for j in 1:n
                left = j == 1 ? zero(T) : x[j - 1]
                right = j == n ? zero(T) : x[j + 1]
                r[j] = d * x[j] + e * (left + right) - λ * (x[j]^3 - x[j])
            end
            nothing
        end
    end
    return global_, (residuals...,), (jacobians...,)
end

# A member that runs a `RAS` and counts, from the block-Jacobian calls it owns, the sweeps, the
# block solves of the parallel schedule and every block solve. The sweep forms one block Jacobian
# per block solve, and on a state that a RASPEN preparation reads one more per block at the local
# iterate it returns, so a count with no correction term is the one `ras_counts` reports.
struct CountingRAS{M, C, R}
    ras::M
    calls::C
    counts::R
end

CountingRAS(ras, calls) = CountingRAS(ras, calls, Ref((0, 0, 0)))

GeometricSolvers.init_state(m::CountingRAS, prob, x, p) = init_state(m.ras, prob, x, p)
function GeometricSolvers.start!!(state, m::CountingRAS, prob, x, Fx, p, fnorm₀)
    start!!(state, m.ras, prob, x, Fx, p, fnorm₀)
end

function GeometricSolvers.step!!(state, m::CountingRAS, prob, x, Fx, p)
    before = [c[] for c in m.calls]
    out = step!!(state, m.ras, prob, x, Fx, p)
    per_block = [c[] for c in m.calls] .- before
    sweeps, parallel, total = m.counts[]
    m.counts[] = (sweeps + 1, parallel + maximum(per_block), total + sum(per_block))
    return out
end

@testset "Check 6: the P8 counts and the parallel schedule, $T" for T in REAL_ELTYPES
    n = 20
    part = BlockPartition(n, 5; overlap = 2)
    λ = T(200)
    global_, residuals, jacobians = p8_blocks(T, n, λ, part)
    counters = [Ref(0) for _ in 1:length(part.owners)]
    counted = ntuple(
        i -> let i = i, J_i = jacobians[i]
            (J, x_loc, x, p) -> (counters[i][] += 1; J_i(J, x_loc, x, p))
        end,
        length(part.owners))
    prob = NonlinearProblem(global_; block_residuals = residuals, block_jacobians = counted)

    h = inv(T(n + 1))
    x0 = T[sin(T(pi) * j * h) for j in 1:n]
    _, newton_status = solve(prob, copy(x0), Newton(); max_iterations = 200)
    @test newton_status.code === SUCCESS
    @test newton_status.iterations > 15

    tol = sqrt(eps(T))

    # NKS-RAS: Multiplicative(RAS(partition), Newton())
    ras = RAS(part; max_iterations = 50, rtol = tol)
    counting = CountingRAS(ras, counters)
    m_nks = Multiplicative(counting, Newton())
    _, nks_status, state_nks = GeometricSolvers.solve!!(
        init_state(m_nks, prob, x0, nothing), prob, copy(x0), m_nks, nothing,
        GeometricSolvers.Options(T))
    @test nks_status.code === SUCCESS
    @test nks_status.iterations <= 6

    # Check 6: the counters of the state equal the counting inner member bit for bit
    sweeps, parallel, total = counting.counts[]
    c = ras_counts(state_nks)
    @test c.sweeps == sweeps
    @test c.parallel_block_solves == parallel
    @test c.total_inner_iterations == total
    @test c.inner_iterations == parallel
    @test c.block_solves == total
    # the inner stop test ends each local solve before its last inner iteration
    @test c.total_inner_iterations < 50 * length(part.owners) * c.sweeps

    # RASPEN: LeftPreconditioned(Newton(), RAS(partition))
    ras2 = RAS(part; max_iterations = 50, rtol = tol)
    counting2 = CountingRAS(ras2, counters)
    m = LeftPreconditioned(Newton(), counting2)
    _, raspen_status, state_raspen = GeometricSolvers.solve!!(
        init_state(m, prob, x0, nothing), prob, copy(x0), m, nothing,
        GeometricSolvers.Options(T))
    @test raspen_status.code === SUCCESS
    @test raspen_status.iterations <= 6
    sweeps2, parallel2, total2 = counting2.counts[]
    c2 = ras_counts(state_raspen)
    @test c2.sweeps == sweeps2
    @test c2.parallel_block_solves == parallel2
    @test c2.total_inner_iterations == total2
    @test c2.total_inner_iterations < 50 * length(part.owners) * c2.sweeps

    # the counts at one inner step are recorded as a measurement, not asserted
    let
        ras1 = RAS(part; max_iterations = 1)
        counting1 = CountingRAS(ras1, [Ref(0) for _ in 1:length(part.owners)])
        m1 = Multiplicative(counting1, Newton())
        _, status1, state1 = GeometricSolvers.solve!!(
            init_state(m1, prob, copy(x0), nothing), prob, copy(x0), m1, nothing,
            GeometricSolvers.Options(T; max_iterations = 200))
        @info "the P8 NKS-RAS counts at one inner step" T status1.code status1.iterations ras_counts(
            state1)
    end
end

# --- the RASPEN Jacobian follows eq. 3.17 -----------------------------------------------------

@testset "The RASPEN Jacobian is a central difference of rˡ, $T" for T in REAL_ELTYPES
    n = 20
    part = BlockPartition(n, 5; overlap = 2)
    λ = T(200)
    global_, residuals, jacobians = p8_blocks(T, n, λ, part)
    prob = NonlinearProblem(global_; block_residuals = residuals,
        block_jacobians = jacobians)
    h = inv(T(n + 1))
    x0 = T[sin(T(pi) * j * h) for j in 1:n]
    xstar, newton_status = solve(prob, copy(x0), Newton(); max_iterations = 200)
    @test newton_status.code === SUCCESS
    # away from the root, so that the difference is not dominated by a converged residual
    x = xstar .+ T.(sin.(1:n)) ./ 10

    ras = RAS(part; max_iterations = 50, rtol = zero(T))
    m = LeftPreconditioned(Newton(), ras)
    state = init_state(m, prob, x, nothing)
    lprob = state.problem
    # the local solves of the point, and the block factors they formed
    ras_sweep!!(lprob.state, ras, prob, copy(x), nothing)
    prep = prepare_ad(AutoForwardDiff(), lprob, similar(x), x, nothing)
    @test prep isa GeometricSolvers.RASPenPreparation

    v = T[cos(T(3) * j) for j in 1:n]
    Jv = similar(v)
    jvp!!(Jv, prep, lprob, x, v, nothing)

    # the central difference of `rˡ` at the same point, with the step of `difference_jvp!`
    R = real(T)
    hv = cbrt(eps(R)) * max(one(R), norm(x)) / norm(v)
    rp = similar(x)
    rm = similar(x)
    residual!!(rp, lprob, x .+ hv .* v, nothing)
    residual!!(rm, lprob, x .- hv .* v, nothing)
    cd = (rp .- rm) ./ (2hv)
    @test norm(Jv - cd) <= 10 * sqrt(eps(R)) * norm(cd)
end
