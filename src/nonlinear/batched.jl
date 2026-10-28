"""
    PerSystem(data::AbstractVector)

Wrap an `AbstractVector` of per-system parameters so [`solve!`](@ref) passes `data[i]` to system
`i`. Any parameter value not wrapped in `PerSystem` — a number, a tuple, an `SVector`, or a plain
`AbstractVector` — is shared unchanged across every system in the batch.
"""
struct PerSystem{P <: AbstractVector}
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

function _adapt_order(backend, order::AbstractVector{<:Integer})
    if backend isa KernelAbstractions.CPU || get_backend(order) == backend
        return order
    end
    order_dev = allocate(backend, eltype(order), length(order))
    copyto!(order_dev, order)
    return order_dev
end

function _validate_batch(
        X::AbstractVector,
        p,
        order::Union{Nothing, AbstractVector{<:Integer}},
        options::Union{Nothing, Options},
        kwargs
)
    if p isa PerSystem && length(p.data) != length(X)
        throw(DimensionMismatch(
            "PerSystem parameter length $(length(p.data)) does not match batch length $(length(X))"))
    end
    if order !== nothing && length(order) != length(X)
        throw(DimensionMismatch(
            "order length $(length(order)) does not match batch length $(length(X))"))
    end
    if options !== nothing && !isempty(kwargs)
        throw(ArgumentError("pass either `options` or keyword arguments, not both"))
    end
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
    opt = options === nothing ? Options(R; kwargs...) : convert(Options{R}, options)
    m = Adapt.adapt(ToReal{R}(), method)
    state₀ = init(m, prob, zero(eltype(X)), nothing)
    return m, opt, state₀
end

@kernel function _solve_status_kernel!(
        X, S, @Const(prob), @Const(m), @Const(p), @Const(opt), @Const(state₀), @Const(order))
    i = @index(Global, Linear)
    @inbounds begin
        idx = _system_index(order, i)
        x, st, _ = solve!!(state₀, prob, X[idx], m, parameters(p, idx), opt)
        X[idx] = x
        S[idx] = st
    end
end

@kernel function _solve_codes_kernel!(
        X, codes, @Const(prob), @Const(m), @Const(p),
        @Const(opt), @Const(state₀), @Const(order))
    i = @index(Global, Linear)
    @inbounds begin
        idx = _system_index(order, i)
        x, st, _ = solve!!(state₀, prob, X[idx], m, parameters(p, idx), opt)
        X[idx] = x
        codes[idx] = st.code
    end
end

function _tally_return_codes(codes_host::Vector{ReturnCode})
    n_success = 0
    n_stalled = 0
    n_maxiters = 0
    n_singular = 0
    n_nonfinite = 0
    n_linesearch_failed = 0
    @inbounds for c in codes_host
        if c == SUCCESS
            n_success += 1
        elseif c == STALLED
            n_stalled += 1
        elseif c == MAXITERS
            n_maxiters += 1
        elseif c == SINGULAR
            n_singular += 1
        elseif c == NONFINITE
            n_nonfinite += 1
        else
            n_linesearch_failed += 1
        end
    end
    return (
        SUCCESS = n_success,
        STALLED = n_stalled,
        MAXITERS = n_maxiters,
        SINGULAR = n_singular,
        NONFINITE = n_nonfinite,
        LINESEARCH_FAILED = n_linesearch_failed
    )
end

"""
    solve!(X::AbstractVector{<:Union{SVector, Number}},
           S::AbstractVector{SolverStatus{R}},
           prob::NonlinearProblem,
           method,
           p = nothing;
           order = nothing,
           options = nothing,
           kwargs...) -> S

    solve!(X::AbstractVector{<:Union{SVector, Number}},
           prob::NonlinearProblem,
           method,
           p = nothing;
           order = nothing,
           options = nothing,
           kwargs...) -> NamedTuple{(:SUCCESS, :STALLED, :MAXITERS, :SINGULAR, :NONFINITE, :LINESEARCH_FAILED), NTuple{6, Int}}

Solve a batch of independent small nonlinear systems in place on `get_backend(X)`, assigning one
work-item per system.

Each element `X[idx]` is the initial guess for system `idx` and is overwritten with its final
iterate. Wrap a per-system parameter vector in [`PerSystem`](@ref) so system `idx` receives
`p.data[idx]`; any other `p` is shared across all systems. When `order` is an integer vector of
length `length(X)`, work-item `i` solves system `idx = order[i]`.

The first form writes each system's [`SolverStatus`](@ref) into `S[idx]` and returns `S`. The
second form allocates a 1-byte-per-system [`ReturnCode`](@ref) buffer on `get_backend(X)`, copies
the codes once to the host after kernel completion, and returns the six outcome counts as a
`NamedTuple`.
"""
function solve!(
        X::AbstractVector{<:Union{SVector, Number}},
        S::AbstractVector{SolverStatus{R}},
        prob::NonlinearProblem,
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        kwargs...
) where {R <: Real}
    if length(S) != length(X)
        throw(DimensionMismatch(
            "status vector length $(length(S)) does not match batch length $(length(X))"))
    end
    _validate_batch(X, p, order, options, kwargs)
    m, opt, state₀ = _prepare_batch(X, prob, method, options, kwargs)
    isempty(X) && return S
    backend = get_backend(X)
    order_dev = _adapt_order(backend, order)
    _solve_status_kernel!(backend)(
        X, S, prob, m, p, opt, state₀, order_dev; ndrange = length(X))
    applicable(synchronize, backend) && synchronize(backend)
    return S
end

function solve!(
        X::AbstractVector{<:Union{SVector, Number}},
        prob::NonlinearProblem,
        method,
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        kwargs...
)
    _validate_batch(X, p, order, options, kwargs)
    m, opt, state₀ = _prepare_batch(X, prob, method, options, kwargs)
    if isempty(X)
        return (
            SUCCESS = 0,
            STALLED = 0,
            MAXITERS = 0,
            SINGULAR = 0,
            NONFINITE = 0,
            LINESEARCH_FAILED = 0
        )
    end
    backend = get_backend(X)
    codes = allocate(backend, ReturnCode, length(X))
    order_dev = _adapt_order(backend, order)
    _solve_codes_kernel!(backend)(
        X, codes, prob, m, p, opt, state₀, order_dev; ndrange = length(X))
    applicable(synchronize, backend) && synchronize(backend)
    return _tally_return_codes(Array(codes))
end
