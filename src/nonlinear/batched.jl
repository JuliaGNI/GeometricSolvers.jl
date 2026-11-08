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

@inline _system_index(::Nothing, i::Integer) = Int(i)
@inline _system_index(order::AbstractVector{<:Integer}, i::Integer) = Int(@inbounds order[i])

function _adapt_order(backend, ::Nothing)
    return nothing
end

# An `isbits` order, such as a range, goes to the kernel by value; another order is copied into an
# array of the batch's backend unless it is one already. A kernel reads the backend's own array,
# not the host's memory, and the CPU backend compiles its kernels, so a host order is copied there
# as well.
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
    if order !== nothing && length(order) != length(X)
        throw(DimensionMismatch(
            "order length $(length(order)) does not match batch length $(length(X))"))
    end
    if order !== nothing && !isperm(Adapt.adapt(Array, order))
        throw(ArgumentError("order is not a permutation of 1:$(length(X))"))
    end
    return nothing
end

# A batch runs its solves in one kernel, which takes the problem by value, so the problem must be
# `isbits`: a `jacobian_prototype` that is a `SparseMatrixCSC`, for one, is host memory that no
# kernel reads.
function _validate_batch_problem(prob)
    isbits(prob) || throw(ArgumentError(
        "solve_batch! takes the problem by value into a kernel, so the problem must be `isbits`; " *
        "got $(typeof(prob)). Run `solve(prob, x₀, method, p)` per system instead"))
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
# wait for the kernel.
function _launch!(X, out, prob, m, p, opt, state₀, order)
    backend = get_backend(X)
    order_dev = _adapt_order(backend, order)
    _solve_kernel!(backend)(X, out, prob, m, p, opt, state₀, order_dev; ndrange = length(X))
    isgpu(backend) && synchronize(backend)
    return out
end

function _tally_return_codes(codes_host::Vector{ReturnCode})
    return ReturnCodeCounts(map(c -> count(==(c), codes_host), instances(ReturnCode)))
end

"""
    solve_batch!(X::AbstractVector{<:Union{SVector, Number}},
                 S::AbstractVector{SolverStatus{R}},
                 prob::NonlinearProblem,
                 method,
                 p = nothing;
                 order = nothing,
                 options = nothing,
                 kwargs...) -> S

    solve_batch!(X::AbstractVector{<:Union{SVector, Number}},
                 prob::NonlinearProblem,
                 method,
                 p = nothing;
                 order = nothing,
                 options = nothing,
                 kwargs...) -> NamedTuple{(:SUCCESS, :STALLED, :MAXITERS, :SINGULAR, :NONFINITE, :LINESEARCH_FAILED), NTuple{6, Int}}

Solve a batch of independent small nonlinear systems in place on `get_backend(X)`, assigning one
work-item per system.

Each element `X[idx]` is the initial guess for system `idx` and is overwritten with its final
iterate. Wrap an indexable per-system parameter collection in [`PerSystem`](@ref) so system `idx`
receives `p.data[idx]`; any other `p` is shared across all systems. When `order` is a
permutation of `1:length(X)`, work-item `i` solves system `idx = order[i]`; any other `order`
throws. An `order` that is not `isbits` needs a method of `get_backend`, and an `order` on another
backend than `X` is copied to the backend of `X`. Any `ForwardDiff.Dual` in `X` or `p` raises an
`ArgumentError`. `solve_batch!` does not check `prob`, so a `ForwardDiff.Dual` captured in `prob`
is not detected in a batch.

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
    m, opt, state₀ = _prepare_batch(X, prob, method, options, kwargs)
    isempty(X) && return S
    return _launch!(X, S, prob, m, p, opt, state₀, order)
end

function solve_batch!(
        X::AbstractVector{<:Union{SVector, Number}},
        prob::NonlinearProblem,
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        kwargs...
)
    _validate_batch(X, p, order)
    _validate_batch_problem(prob)
    m, opt, state₀ = _prepare_batch(X, prob, method, options, kwargs)
    isempty(X) && return _tally_return_codes(ReturnCode[])
    codes = allocate(get_backend(X), ReturnCode, length(X))
    return _tally_return_codes(Array(_launch!(X, codes, prob, m, p, opt, state₀, order)))
end
