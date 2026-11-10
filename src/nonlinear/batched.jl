"""
    PerSystem(data)

Wrap an indexable collection of per-system parameters so [`solve_batch!`](@ref) passes `data[i]`
to system `i`. Any parameter value not wrapped in `PerSystem` — a number, a tuple, an `SVector`,
or a plain `AbstractVector` — is shared unchanged across every system in the batch.
"""
struct PerSystem{P}
    data::P
end

Adapt.@adapt_structure PerSystem

"""
    parameters(p, i::Integer)

Return the parameter value for system `i`: `p.data[i]` when `p isa PerSystem`, and `p` unchanged
otherwise.
"""
@inline parameters(p::PerSystem, i::Integer) = @inbounds p.data[i]
@inline parameters(p, ::Integer) = p

"""
    BatchParams(shared, per_system)

Wrap a parameter value that every system of a batch shares together with an indexable collection
of per-system values, so [`solve_batch!`](@ref) passes `(p.shared, p.per_system[i])` to system
`i`. It is what a batch whose systems differ in one field and agree in another takes, such as the
particles of an implicit PIC step, which each carry their own previous position and velocity and
share the field they gather. Both parts are adapted to the batch's backend with `Adapt`, so a
`shared` value that is an `Array` follows `X` onto a device.
"""
struct BatchParams{S, P}
    shared::S
    per_system::P
end

Adapt.@adapt_structure BatchParams

"""
    parameters(p::BatchParams, i::Integer)

The parameter value of system `i` of a [`BatchParams`](@ref): the pair `(p.shared,
p.per_system[i])`.
"""
@inline parameters(p::BatchParams, i::Integer) = (p.shared, @inbounds p.per_system[i])

"""
    BatchSummary{R <: Real}

The reduction of a batch's per-system [`SolverStatus`](@ref) vector to one report.

# Fields

- `fnorm::R`: ``\\sqrt{\\sum_i \\|g_i\\|_2^2}``, the norm of the concatenated per-system residuals,
  from the `fnorm` of each system's status in `real(eltype(...))` of the batch.
- `all_converged::Bool`: every system ended with `SUCCESS`.
- `failure_count::Int`: the number of systems that did not.
- `max_iterations::Int32`, `total_iterations::Int`: the largest and the sum of the per-system
  iteration counts.
"""
struct BatchSummary{R <: Real}
    fnorm::R
    all_converged::Bool
    failure_count::Int
    max_iterations::Int32
    total_iterations::Int
end

"""
    batch_summary(S::AbstractVector{SolverStatus{R}}) -> BatchSummary{R}

Reduce the statuses `S` of a batch to a [`BatchSummary`](@ref), on the backend of `S`: the global
residual norm is the 2-norm of the per-system norms through the [`rnorm`](@ref) seam, and the
counts are the `mapreduce` forms of `count`, `maximum` and `sum`, so no entry of `S` is indexed by
the host and the reduction runs on a device array as it does on an `Array`. An empty batch
converged, with `fnorm = 0` and both iteration counts `0`.
"""
function batch_summary(S::AbstractVector{SolverStatus{R}}) where {R <: Real}
    isempty(S) && return BatchSummary{R}(zero(R), true, 0, Int32(0), 0)
    batch_norm = rnorm(map(s -> s.fnorm, S))
    all_converged = all(isconverged, S)
    failure_count = count(s -> !isconverged(s), S)
    max_iterations = maximum(s -> s.iterations, S)
    total_iterations = sum(s -> Int(s.iterations), S)
    return BatchSummary{R}(batch_norm, all_converged, Int(failure_count), max_iterations,
        total_iterations)
end

@inline _system_index(::Nothing, i::Integer) = Int(i)
@inline _system_index(order::AbstractVector{<:Integer}, i::Integer) = Int(@inbounds order[i])

function _adapt_order(backend, ::Nothing)
    return nothing
end

# An `isbits` order, such as a range, goes to the kernel by value; another order goes as an array
# of the batch's backend, which is the order itself when it is one and a copy otherwise. A kernel
# reads only the backend's own arrays and the CPU backend compiles its kernels, so a host container
# that is not an `Array`, such as a `view`, is copied there too.
function _adapt_order(backend, order::AbstractVector{<:Integer})
    isbitstype(typeof(order)) && return order
    order_dev = allocate(backend, eltype(order), length(order))
    order isa typeof(order_dev) && return order
    # An order that is not the backend's own array is read through its own backend, which raises
    # where the container names none.
    get_backend(order)
    copyto!(order_dev, order isa Array ? order : Array(order))
    return order_dev
end

function _validate_batch(
        X::AbstractVector,
        p,
        order::Union{Nothing, AbstractVector{<:Integer}}
)
    if has_dual_type(typeof(X)) || has_dual_type(typeof(p))
        throw(ArgumentError(
            "solve_batch! does not accept ForwardDiff.Dual inputs; use solve(prob, x₀, method, p) per system"))
    end
    if p isa PerSystem && length(p.data) != length(X)
        throw(DimensionMismatch(
            "PerSystem parameter length $(length(p.data)) does not match batch length $(length(X))"))
    end
    if p isa BatchParams && length(p.per_system) != length(X)
        throw(DimensionMismatch(
            "BatchParams per-system length $(length(p.per_system)) does not match batch length $(length(X))"))
    end
    if order !== nothing && length(order) != length(X)
        throw(DimensionMismatch(
            "order length $(length(order)) does not match batch length $(length(X))"))
    end
    if order !== nothing && !isperm(Adapt.adapt(Array, order))
        throw(ArgumentError("order is not a permutation of 1:$(length(X))"))
    end
    return nothing
end

# A batch runs its solves in one kernel, which takes the problem by value, so every part of the
# problem the kernel reads must be `isbits`: a `jacobian_prototype` that is a `SparseMatrixCSC`, for
# one, is host memory that no kernel reads.
function _validate_batch_problem(prob)
    P = jacobian_prototype(prob)
    (P === nothing || isbits(P)) || throw(ArgumentError(
        "solve_batch! takes the problem's Jacobian prototype by value into a kernel, so the " *
        "prototype must be `isbits`; got $(typeof(P)) in $(typeof(prob)). Run " *
        "`solve(prob, x₀, method, p)` per system instead"))
    return nothing
end

function _prepare_batch(
        X::AbstractVector,
        prob::NonlinearProblem,
        method,
        options::Union{Nothing, Options},
        kwargs
)
    T = eltype(eltype(X))
    R = real(T)
    opt = resolve_options(T, options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    state₀ = init_state(m, prob, zero(eltype(X)), nothing)
    return m, opt, state₀
end

# The result of system `idx` in the output of each form: its status, or its code.
@inline _write_result!(S::AbstractVector{<:SolverStatus}, idx, st) = (@inbounds S[idx] = st)
@inline _write_result!(codes::AbstractVector{ReturnCode}, idx, st) = (@inbounds codes[idx] = st.code)

@kernel function _solve_kernel!(X, out, prob, m, p, opt, state₀, order)
    i = @index(Global, Linear)
    @inbounds begin
        idx = _system_index(order, i)
        x, st, _ = solve!!(state₀, prob, X[idx], m, parameters(p, idx), opt)
        X[idx] = x
        _write_result!(out, idx, st)
    end
end

# Solve every system of `X` in one launch on `get_backend(X)`, write each result into `out`, and
# wait for the kernel. A `workgroupsize` of `nothing` leaves the launch at the backend's default.
function _launch!(X, out, prob, m, p, opt, state₀, order, workgroupsize)
    backend = get_backend(X)
    order_dev = _adapt_order(backend, order)
    kernel! = _solve_kernel!(backend)
    if workgroupsize === nothing
        kernel!(X, out, prob, m, p, opt, state₀, order_dev; ndrange = length(X))
    else
        kernel!(X, out, prob, m, p, opt, state₀, order_dev;
            ndrange = length(X), workgroupsize = workgroupsize)
    end
    isgpu(backend) && synchronize(backend)
    return out
end

function _tally_return_codes(codes_host::Vector{ReturnCode})
    return ReturnCodeCounts(map(c -> count(==(c), codes_host), instances(ReturnCode)))
end

# The workgroup size of a launch, refused unless it is a positive count: a kernel backend rounds a
# size above its maximum down and raises nothing, so a nonsensical value is caught here.
@inline function _validate_workgroupsize(workgroupsize::Union{Nothing, Integer})
    workgroupsize === nothing && return nothing
    workgroupsize >= 1 || throw(ArgumentError(
        "workgroupsize must be a positive integer or `nothing`, got $workgroupsize"))
    return Int(workgroupsize)
end

"""
    solve_batch!(X::AbstractVector{<:Union{SVector, Number}},
                 S::AbstractVector{SolverStatus{R}},
                 prob::NonlinearProblem,
                 method,
                 p = nothing;
                 order = nothing,
                 options = nothing,
                 workgroupsize = nothing,
                 kwargs...) -> S

    solve_batch!(X::AbstractVector{<:Union{SVector, Number}},
                 prob::NonlinearProblem,
                 method,
                 p = nothing;
                 order = nothing,
                 options = nothing,
                 workgroupsize = nothing,
                 kwargs...) -> NamedTuple{(:SUCCESS, :STALLED, :MAXITERS, :SINGULAR, :NONFINITE, :LINESEARCH_FAILED), NTuple{6, Int}}

Solve a batch of independent small nonlinear systems in place on `get_backend(X)`, assigning one
work-item per system.

Each element `X[idx]` is the initial guess for system `idx` and is overwritten with its final
iterate. Wrap an indexable per-system parameter collection in [`PerSystem`](@ref) so system `idx`
receives `p.data[idx]`, or in [`BatchParams`](@ref) so it receives `(p.shared, p.per_system[idx])`;
any other `p` is shared across all systems. When `order` is a
permutation of `1:length(X)`, work-item `i` solves system `idx = order[i]`; any other `order`
throws. An `order` that is not `isbits` needs a method of `get_backend`, and one that is not an
array of the backend of `X`, such as an array of another backend or a host `view`, is copied into
one. Any `ForwardDiff.Dual` in `X` or `p` raises an `ArgumentError`. The kernel takes `prob` by
value, so a `jacobian_prototype` that is not `isbits`, such as a `SparseMatrixCSC`, raises an
`ArgumentError`. `solve_batch!` does not look for a `ForwardDiff.Dual` captured in `prob`, so one is
not detected in a batch. A `workgroupsize` of `nothing` leaves the launch at the backend's
default; any other positive value is the workgroup size of the single launch. The `Options` of the
solve may also be passed as the sixth positional argument, `solve_batch!(X, S, prob, method, p,
options)`, instead of the `options` keyword.

The first form writes each system's [`SolverStatus`](@ref) into `S[idx]` and returns `S` (requiring
`R === real(eltype(eltype(X)))` and `get_backend(S) == get_backend(X)`). The second form allocates a 1-byte-per-system [`ReturnCode`](@ref)
buffer on `get_backend(X)`, copies the codes once to the host after kernel completion, and returns
the six outcome counts as a `NamedTuple`.
"""
function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        S::AbstractVector{SolverStatus{R}},
        prob::NonlinearProblem,
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        workgroupsize::Union{Nothing, Integer} = nothing,
        kwargs...
) where {R <: Real}
    _validate_batch(X, p, order)
    _validate_batch_problem(prob)
    R_X = real(eltype(eltype(X)))
    if R !== R_X
        throw(ArgumentError(
            "SolverStatus real type $R does not match batch real element type $R_X"))
    end
    if length(S) != length(X)
        throw(DimensionMismatch(
            "status vector length $(length(S)) does not match batch length $(length(X))"))
    end
    if get_backend(S) != get_backend(X)
        throw(ArgumentError(
            "status vector backend $(get_backend(S)) does not match batch backend $(get_backend(X))"))
    end
    ws = _validate_workgroupsize(workgroupsize)
    m, opt, state₀ = _prepare_batch(X, prob, method, options, kwargs)
    isempty(X) && return S
    return _launch!(X, S, prob, m, p, opt, state₀, order, ws)
end

function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        prob::NonlinearProblem,
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        workgroupsize::Union{Nothing, Integer} = nothing,
        kwargs...
)
    _validate_batch(X, p, order)
    _validate_batch_problem(prob)
    ws = _validate_workgroupsize(workgroupsize)
    m, opt, state₀ = _prepare_batch(X, prob, method, options, kwargs)
    isempty(X) && return _tally_return_codes(ReturnCode[])
    codes = allocate(get_backend(X), ReturnCode, length(X))
    return _tally_return_codes(Array(_launch!(
        X, codes, prob, m, p, opt, state₀, order, ws)))
end

# The options of a batch are the sixth positional argument where a caller passes them positionally,
# so that a batch call reads like the `solve` call whose keywords it shares.
function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        S::AbstractVector{SolverStatus{R}},
        prob::NonlinearProblem,
        method,
        p,
        options::Options;
        kwargs...
) where {R <: Real}
    return solve_batch!(X, S, prob, method, p; options = options, kwargs...)
end

function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        prob::NonlinearProblem,
        method,
        p,
        options::Options;
        kwargs...
)
    return solve_batch!(X, prob, method, p; options = options, kwargs...)
end
