"""
    StationarityProblem(S; E = :, U = :, b = nothing)
    StationarityProblem(S, E, U, b = nothing)

A stationarity problem `(∇S(x, p))_E + b = 0` for the unknown entries `x[U]` of a state `x`,
with scalar action `S(x, p)`, stationarity-equation index set `E`, unknown index set `U`, and
optional datum `b`.

The residual and its Jacobian are

```math
F_a(u, p) = \\frac{\\partial S}{\\partial x_{E_a}}(x(u), p) + b_a, \\qquad
J_{a, c}(u, p) = \\frac{\\partial^2 S}{\\partial x_{E_a}\\,\\partial x_{U_c}}(x(u), p),
\\qquad a, c \\in \\{1, \\dots, |E|\\},
```

where ``x(u)_i = u_c`` for ``i = U_c`` and ``x(u)_i = (x_0)_i`` for ``i \\notin U``. The action
`S(x, p)` is never used as a line-search or trust-region merit function: the merit is always
``\\|F\\|_2^2 = \\|(\\nabla S)_E + b\\|_2^2``, because stationarity points of physical actions
include saddle points where ``S`` is indefinite.

Where `E == U`, `J` is a block of the Hessian of `S` and is symmetric. For other index sets it is
not symmetric in general, so [`LDLTFactorization`](@ref), which reads only the lower triangle,
does not apply, and [`LUFactorization`](@ref) or [`QRFactorization`](@ref) does.

The state `x` must be real. For a complex state the default automatic differentiation raises an
error: an `ArgumentError` from ForwardDiff for an array iterate, and a `MethodError` for an R1
iterate. A `StationarityProblem` has no implicit differentiation: [`init`](@ref),
[`solve`](@ref), [`solve!`](@ref) and [`solve_batch!`](@ref) raise an `ArgumentError` for a
`ForwardDiff.Dual` in the iterate or in `p`.

# Arguments

- `S`: a scalar callable `S(x, p)` returning a scalar in the element type of `x`.
- `E`: `:` (all indices `1:length(x₀)`) or a duplicate-free collection of positive integer indices
  (`Tuple`, `SVector`, `AbstractUnitRange`, or `AbstractVector{<:Integer}`). A `Tuple` of integers
  is converted to `SVector{N, Int}` so an R1 problem is `isbits`. An R1 iterate (a `StaticVector`
  or a `Number`) takes `:`, a `Tuple` or an `SVector`, and any other index set raises an
  `ArgumentError` at the solve.
- `U`: `:` or a duplicate-free collection of positive integer indices of the same length as `E`.
  `E` and `U` are both `:` or both index sets.
- `b`: `nothing` (zero datum), a vector of length `|E|`, a `Number`, or a callable `b(p)` returning
  a vector of length `|E|` or a `Number`. The keyword `b` on [`init`](@ref), [`solve`](@ref),
  [`solve!`](@ref) and [`solve_batch!`](@ref) overrides `prob.b` for that solve; in
  [`solve_batch!`](@ref) a [`PerSystem`](@ref) `b` gives system `i` the datum `b.data[i]`.
"""
struct StationarityProblem{S, E, U, B} <: AbstractNonlinearProblem
    S::S
    E::E
    U::U
    b::B
    function StationarityProblem{S, E, U, B}(s::S, e::E, u::U, b::B) where {S, E, U, B}
        return new{S, E, U, B}(s, e, u, b)
    end
end

const StatProb = StationarityProblem
const StaticOrNumber = Union{StaticVector, Number}

@inline _normalize_indices(::Colon) = Colon()
@inline _normalize_indices(::Nothing) = Colon()
@inline function _normalize_indices(idx::Tuple{Vararg{Integer, N}}) where {N}
    N >= 1 || throw(ArgumentError("index set must be non-empty, got ()"))
    return SVector{N, Int}(map(Int, idx))
end
@inline function _normalize_indices(idx::SVector{N, <:Integer}) where {N}
    return SVector{N, Int}(idx)
end
@inline _normalize_indices(idx::AbstractVector{<:Integer}) = idx
@inline function _normalize_indices(idx)
    throw(ArgumentError("invalid index set type $(typeof(idx))"))
end

@inline _has_duplicates(::AbstractUnitRange) = false
function _has_duplicates(v)
    n = length(v)
    @inbounds for i in 1:n
        vi = v[i]
        for j in (i + 1):n
            vi == v[j] && return true
        end
    end
    return false
end

@inline _validate_stationarity_indices(::Colon, ::Colon) = nothing
function _validate_stationarity_indices(::Colon, U)
    throw(ArgumentError("StationarityProblem requires both E and U to be :"))
end
function _validate_stationarity_indices(E, ::Colon)
    throw(ArgumentError("StationarityProblem requires both E and U to be :"))
end
function _validate_stationarity_indices(E, U)
    ne = length(E)
    nu = length(U)
    ne == nu || throw(ArgumentError("length(E) == length(U) required, got $ne and $nu"))
    ne >= 1 || throw(ArgumentError("index sets E and U must be non-empty"))
    all(i -> i >= 1, E) || throw(ArgumentError("indices in E must be >= 1, got $E"))
    all(i -> i >= 1, U) || throw(ArgumentError("indices in U must be >= 1, got $U"))
    _has_duplicates(E) && throw(ArgumentError("index set E has duplicate indices"))
    _has_duplicates(U) && throw(ArgumentError("index set U has duplicate indices"))
    return nothing
end

function StationarityProblem(S, E, U, b = nothing)
    En = _normalize_indices(E)
    Un = _normalize_indices(U)
    _validate_stationarity_indices(En, Un)
    TProb = StationarityProblem{typeof(S), typeof(En), typeof(Un), typeof(b)}
    return TProb(S, En, Un, b)
end

function StationarityProblem(S; E = Colon(), U = Colon(), b = nothing)
    return StationarityProblem(S, E, U, b)
end

Adapt.@adapt_structure StationarityProblem

@inline _num_equations(::Colon, M::Int) = M
@inline _num_equations(E, ::Int) = length(E)

@inline _eval_datum(::Nothing, p) = nothing
@inline _eval_datum(b::Union{AbstractArray, Number}, p) = b
@inline _eval_datum(b, p) = b(p)

@inline function _resolve_datum(prob::StationarityProblem, p, b_kw)
    b_raw = b_kw === nothing ? prob.b : b_kw
    return _eval_datum(b_raw, p)
end

function _validate_stationarity_solve(prob::StationarityProblem, x₀, b_val)
    if x₀ isa StaticOrNumber && !(prob.E isa Union{Colon, SVector{<:Any, Int}} &&
         prob.U isa Union{Colon, SVector{<:Any, Int}})
        throw(ArgumentError(
            "R1/R2 StationarityProblem with $(typeof(x₀)) requires E and U to be : or SVector{N, Int} (or NTuple{N, Int})"))
    end
    M = x₀ isa Number ? 1 : length(x₀)
    if !(prob.E isa Colon)
        all(i -> 1 <= i <= M, prob.E) || throw(ArgumentError("index in E exceeds $M"))
    end
    if !(prob.U isa Colon)
        all(i -> 1 <= i <= M, prob.U) || throw(ArgumentError("index in U exceeds $M"))
    end
    m = _num_equations(prob.E, M)
    if b_val isa AbstractVector
        nb = length(b_val)
        nb == m || throw(ArgumentError("datum b must have length $m, got $nb"))
    end
    return nothing
end

# The datum of an R1/R2 solve as an `SVector` of the `m` equations, or a scalar for a scalar
# iterate. A vector datum of another length than `m` raises an `ArgumentError`. For a
# `StaticVector` datum both lengths are constants, so the check folds away where they agree and a
# device kernel compiles no `throw`; a mismatch is refused on the host first, when the state for
# the first system of a batch is initialised.
@inline function _check_datum_length(b::AbstractVector, m::Int)
    length(b) == m ||
        throw(ArgumentError("datum b must have length $m, got $(length(b))"))
    return nothing
end

@inline _static_datum(::StationarityProblem, x₀, ::Nothing) = nothing
@inline function _static_datum(p, ::StaticVector{M, T}, b::AbstractVector) where {M, T}
    m = _num_equations(p.E, M)
    _check_datum_length(b, m)
    return SVector{m, T}(ntuple(i -> T(@inbounds(b[i])), Val(m)))
end
@inline function _static_datum(p, ::StaticVector{M, T}, b::Number) where {M, T}
    m = _num_equations(p.E, M)
    tb = T(b)
    return SVector{m, T}(ntuple(_ -> tb, Val(m)))
end
@inline function _static_datum(::Any, ::T, b::AbstractVector) where {T <: Number}
    _check_datum_length(b, 1)
    return T(@inbounds(b[1]))
end
@inline _static_datum(::Any, ::T, b::Number) where {T <: Number} = T(b)

@inline _idx_pos(::Colon, i::Int) = i
@inline function _idx_pos(idx::SVector{N, Int}, i::Int) where {N}
    pos = 0
    @inbounds for k in 1:N
        pos = ifelse(idx[k] == i, k, pos)
    end
    return pos
end

@inline _idx_len(::Colon, ::Val{M}) where {M} = Val(M)
@inline _idx_len(::SVector{N, Int}, ::Val{M}) where {N, M} = Val(N)

@inline _extract_unknowns(::Colon, x::StaticVector{M, T}) where {M, T} = SVector{M, T}(x)
@inline function _extract_unknowns(U::SVector{N}, x::StaticVector{M, T}) where {N, M, T}
    return SVector{N, T}(ntuple(j -> @inbounds(x[U[j]]), Val(N)))
end
@inline _extract_unknowns(::Colon, x::Number) = x
@inline _extract_unknowns(::SVector{1, Int}, x::Number) = x

@inline function _embed_entry(U, x₀, u, ::Type{T}, i::Int) where {T}
    pos = _idx_pos(U, i)
    return pos == 0 ? convert(T, @inbounds(x₀[i])) : convert(T, @inbounds(u[pos]))
end

@inline _embed_unknowns(::Colon, ::StaticVector, u::StaticVector) = SVector(u)
@inline function _embed_unknowns(U::SVector{N}, x₀::StaticVector{M, T}, u) where {N, M, T}
    T_out = promote_type(T, eltype(u))
    return SVector{M, T_out}(ntuple(i -> _embed_entry(U, x₀, u, T_out, i), Val(M)))
end
@inline _embed_unknowns(::Colon, x_ref::Number, u::Number) = u
@inline _embed_unknowns(::SVector{1, Int}, x_ref::Number, u::Number) = u

# The solution `x` of an R1/R2 solve in the static array type of the initial iterate `x₀`, so that
# an `MVector` start gives an `MVector`, as `solve!` on a `NonlinearSolver` does.
@inline _like_iterate(x₀::StaticVector, x::StaticVector) = similar_type(x₀, eltype(x))(x)
@inline _like_iterate(::Number, x::Number) = x

@kernel function _extract_u_kernel!(u, @Const(x), @Const(u_idx))
    j = @index(Global, Linear)
    @inbounds u[j] = x[u_idx[j]]
end

# `x_ref` is not `@Const`: the R3 reduced problem passes its workspace as both `x` and `x_ref`.
@kernel function _embed_u_kernel!(x, x_ref, @Const(u), @Const(u_pos))
    i = @index(Global, Linear)
    @inbounds begin
        pos = u_pos[i]
        x[i] = pos == 0 ? x_ref[i] : u[pos]
    end
end

function _extract_u!(u::AbstractVector, x::AbstractVector, ::Nothing)
    copyto!(u, x)
    return u
end

function _extract_u!(u::AbstractVector, x::AbstractVector, u_idx::AbstractVector{Int})
    if x isa AbstractGPUArray
        backend = get_backend(x)
        _extract_u_kernel!(backend)(u, x, u_idx; ndrange = length(u))
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for j in eachindex(u, u_idx)
            u[j] = x[u_idx[j]]
        end
    end
    return u
end

function _embed_u!(x::AbstractVector, x_ref::AbstractVector, u::AbstractVector, ::Nothing)
    copyto!(x, u)
    return x
end

function _embed_u!(x::AbstractVector, x_ref::AbstractVector, u::AbstractVector, u_pos)
    if x isa AbstractGPUArray
        backend = get_backend(x)
        _embed_u_kernel!(backend)(x, x_ref, u, u_pos; ndrange = length(x))
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in eachindex(x, x_ref, u_pos)
            pos = u_pos[i]
            x[i] = pos == 0 ? x_ref[i] : u[pos]
        end
    end
    return x
end

function _update_datum_buf!(b_buf::AbstractVector{T}, has_b, ::Nothing) where {T}
    has_b[] = false
    return nothing
end

function _update_datum_buf!(b_buf::AbstractVector{T}, has_b, b_val::Number) where {T}
    has_b[] = true
    fill!(b_buf, T(b_val))
    return nothing
end

function _update_datum_buf!(b_buf::AbstractVector{T}, has_b, b_val::AbstractVector) where {T}
    has_b[] = true
    copyto!(b_buf, b_val)
    return nothing
end

# Internal reduced problem over the `m = |U|` unknowns `u = x[U]`, holding the full-state
# reference `x_ref` (or workspace `x_work` in R3) whose complement `x[∉ U]` stays fixed at
# `x₀[∉ U]`, together with the active datum `b` and, in R3, the preallocated index maps and
# AD preparation.
struct _ReducedStationarityProblem{P, X, B, HB, IV, UP, PR} <: AbstractNonlinearProblem
    prob::P
    x_ref::X
    b::B
    has_b::HB
    u_idx::IV
    u_pos::UP
    prep::PR
end

const ReducedStatProb = _ReducedStationarityProblem

@inline function _ReducedStationarityProblem(prob::P, x_ref::X, b::B) where {P, X, B}
    N = Nothing
    TR = _ReducedStationarityProblem{P, X, B, N, N, N, N}
    return TR(prob, x_ref, b, nothing, nothing, nothing, nothing)
end

struct StaticStationarityState{ST}
    inner::ST
end

struct DenseStationarityState{ST, RP, U}
    inner::ST
    rprob::RP
    u_work::U
end

const StaticStatState = StaticStationarityState
const DenseStatState = DenseStationarityState

struct _StationaritySysParams{P, B}
    p::P
    b::B
end

struct _StationarityBatchParams{P, B}
    p::P
    b::B
end

Adapt.@adapt_structure _StationaritySysParams
Adapt.@adapt_structure _StationarityBatchParams

@inline function parameters(sp::_StationarityBatchParams, i::Integer)
    return _StationaritySysParams(parameters(sp.p, i), parameters(sp.b, i))
end

@inline _sample_batch_item(x, ::Integer) = x
@inline function _sample_batch_item(p::PerSystem, i::Integer)
    if p.data isa AbstractGPUArray
        return Array(@view(p.data[i:i]))[1]
    else
        return @inbounds p.data[i]
    end
end

@inline function _ad_backend_of(m::Newton)
    return m.jacobian isa AbstractADType ? m.jacobian : AutoForwardDiff()
end
@inline _ad_backend_of(m) = AutoForwardDiff()

@inline function _with_datum(p::StationarityProblem, b)
    b === nothing && return p
    TProb = StationarityProblem{typeof(p.S), typeof(p.E), typeof(p.U), typeof(b)}
    return TProb(p.S, p.E, p.U, b)
end

@inline function init_state(m::Picard, prob::ReducedStatProb, u, p)
    Fx = buffer(u)
    jvpprep = prepare_ad(AutoForwardDiff(), prob, Fx, u, p)
    merit = init_merit(m.globalization, u)
    if Fx === nothing
        return PicardState(jvpprep, nothing, nothing, nothing, merit)
    else
        g = init_globalization(m.globalization, u)
        return PicardState(jvpprep, Fx, buffer(u), g, merit)
    end
end

@inline function step!!(s::PicardState, m::Picard, prob::ReducedStatProb, u, Fx, p)
    ω = real(eltype(u))(m.damping)
    d = map!!(fi -> -ω * fi, s.d, Fx)
    g, gl, jvp, mer = s.g, m.globalization, s.jvpprep, s.merit
    sl = MeasuredSlope()
    u₊, F₊, _, info, merit = globalize_merit!!(g, gl, prob, u, Fx, d, p, jvp, sl, mer)
    return u₊, F₊, with_merit(s, merit), info
end

@inline function _init_stationarity_state(m, prob::StatProb, x₀::StaticOrNumber, p, b)
    b_raw = _resolve_datum(prob, p, b)
    _validate_stationarity_solve(prob, x₀, b_raw)
    b_eff = _static_datum(prob, x₀, b_raw)
    rprob = _ReducedStationarityProblem(prob, x₀, b_eff)
    u₀ = _extract_unknowns(prob.U, x₀)
    return StaticStationarityState(init_state(m, rprob, u₀, p))
end

@inline function init_state(m::Newton, prob::StatProb, x₀::StaticOrNumber, p; b = nothing)
    return _init_stationarity_state(m, prob, x₀, p, b)
end

@inline function init_state(m, prob::StatProb, x₀::StaticOrNumber, p; b = nothing)
    return _init_stationarity_state(m, prob, x₀, p, b)
end

function _init_stationarity_state(
        m, prob::StatProb, x₀::AbstractVector{T}, p, b) where {T}
    b_raw = _resolve_datum(prob, p, b)
    _validate_stationarity_solve(prob, x₀, b_raw)
    M = length(x₀)
    m_eq = _num_equations(prob.E, M)
    x_work = copy(x₀)
    u_work = similar(x₀, T, m_eq)
    b_buf = similar(x₀, T, m_eq)
    has_b = Ref(false)
    _update_datum_buf!(b_buf, has_b, b_raw)
    if prob.U isa Colon
        u_idx = nothing
        u_pos = nothing
    else
        u_cpu = Int[Int(i) for i in prob.U]
        pos_cpu = zeros(Int, M)
        for (j, ui) in enumerate(u_cpu)
            pos_cpu[ui] = j
        end
        u_idx = similar(x₀, Int, m_eq)
        copyto!(u_idx, u_cpu)
        u_pos = similar(x₀, Int, M)
        copyto!(u_pos, pos_cpu)
    end
    _extract_u!(u_work, x_work, u_idx)
    prep = prepare_ad(_ad_backend_of(m), prob, u_work, x_work, p)
    rprob = _ReducedStationarityProblem(prob, x_work, b_buf, has_b, u_idx, u_pos, prep)
    inner = init_state(m, rprob, u_work, p)
    return DenseStationarityState(inner, rprob, u_work)
end

function init_state(
        m::Newton, prob::StatProb, x₀::AbstractVector{T}, p; b = nothing) where {T}
    return _init_stationarity_state(m, prob, x₀, p, b)
end

function init_state(m, prob::StatProb, x₀::AbstractVector{T}, p; b = nothing) where {T}
    return _init_stationarity_state(m, prob, x₀, p, b)
end

@inline function solve!!(
        s::StaticStatState, prob::StatProb, x₀, m, p,
        opt::Options{R}; b = nothing) where {R <: Real}
    p_sys = p isa _StationaritySysParams ? p.p : p
    b_sys = p isa _StationaritySysParams && b === nothing ? p.b : b
    b_raw = _resolve_datum(prob, p_sys, b_sys)
    b_eff = _static_datum(prob, x₀, b_raw)
    rprob = _ReducedStationarityProblem(prob, x₀, b_eff)
    u₀ = _extract_unknowns(prob.U, x₀)
    u, status, inner₊ = solve!!(s.inner, rprob, u₀, m, p_sys, opt)
    x = _like_iterate(x₀, _embed_unknowns(prob.U, x₀, u))
    return x, status, StaticStationarityState(inner₊)
end

@inline function solve!!(s::StaticStatState, pr::StatProb, x₀, m, p = nothing; b = nothing)
    return solve!!(s, pr, x₀, m, p, Options(eltype(x₀)); b = b)
end

function solve!!(
        s::DenseStatState, prob::StatProb, x₀, method, p,
        options::Options{R}; b = nothing) where {R <: Real}
    b_raw = _resolve_datum(prob, p, b)
    rprob = s.rprob
    if b_raw isa AbstractVector
        m = length(s.u_work)
        nb = length(b_raw)
        nb == m || throw(ArgumentError("datum b must have length $m, got $nb"))
    end
    copyto!(rprob.x_ref, x₀)
    _extract_u!(s.u_work, rprob.x_ref, rprob.u_idx)
    _update_datum_buf!(rprob.b, rprob.has_b, b_raw)
    u, status, inner₊ = solve!!(s.inner, rprob, s.u_work, method, p, options)
    _embed_u!(x₀, rprob.x_ref, u, rprob.u_pos)
    if inner₊ === s.inner
        state₊ = s
    else
        state₊ = DenseStationarityState(inner₊, rprob, s.u_work)
    end
    return x₀, status, state₊
end

function solve!!(s::DenseStatState, pr::StatProb, x₀, m, p = nothing; b = nothing)
    return solve!!(s, pr, x₀, m, p, Options(eltype(x₀)); b = b)
end

function _init_stationarity(prob::StatProb, x₀, method, p, b, options, kwargs)
    R = real(eltype(x₀))
    opt = resolve_options(eltype(x₀), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    state = init_state(m, prob, x, p; b = b)
    prob_stored = _with_datum(prob, b)
    return NonlinearSolver(prob_stored, x, m, opt, state, p)
end

function _solve_stationarity(prob::StatProb, x₀, method, p, b, options, kwargs)
    R = real(eltype(x₀))
    opt = resolve_options(eltype(x₀), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    x = x₀ isa Union{StaticArray, Number} ? x₀ : copy(x₀)
    state = init_state(m, prob, x, p; b = b)
    x_sol, status, _ = solve!!(state, prob, x, m, p, opt; b = b)
    return x_sol, status
end

function _solve_stationarity_solver!(solver::NonlinearSolver, x, p, b)
    b_raw = _resolve_datum(solver.prob, p, b)
    _validate_stationarity_solve(solver.prob, x, b_raw)
    solver.x = x
    solver.p = p
    st, pr, m, opt = solver.state, solver.prob, solver.method, solver.options
    x₊, status, state = solve!!(st, pr, x, m, p, opt; b = b)
    solver.x = x₊
    solver.state === state || (solver.state = state)
    return solver.x, status
end

# The index sets of `prob` against the iterate type of the batch `X`, and the length of a shared
# vector datum or of every vector of a `PerSystem` datum `b_src`, which the batch checks before it
# initialises a state. A callable datum is evaluated once per system here, on the host: a device
# kernel cannot format the message of a rejected datum, and one system's datum may differ in
# length from another's.
function _validate_stationarity_batch(X, prob::StationarityProblem, b_src, p)
    x₀ = zero(eltype(X))
    b_shared = b_src isa AbstractVector ? b_src : nothing
    _validate_stationarity_solve(prob, x₀, b_shared)
    m = _num_equations(prob.E, x₀ isa Number ? 1 : length(x₀))
    for nb in _datum_lengths(b_src)
        nb == m || throw(ArgumentError("datum b must have length $m, got $nb"))
    end
    if _is_callable_datum(b_src)
        for i in eachindex(X)
            bi = _eval_datum(b_src, parameters(p, i))
            bi isa AbstractVector || continue
            length(bi) == m ||
                throw(ArgumentError("datum b must have length $m, got $(length(bi))"))
        end
    end
    return nothing
end

# Whether a datum source is a callable `b(p)`: a `Number`, a vector, a `PerSystem` of those and
# `nothing` are not.
_is_callable_datum(::Nothing) = false
_is_callable_datum(::Number) = false
_is_callable_datum(::AbstractArray) = false
_is_callable_datum(::PerSystem) = false
_is_callable_datum(::Any) = true

# The lengths of the vectors of a `PerSystem` datum: one, from the type, for static vectors, which
# a device array also holds, and one per system for a host array of other vectors.
_datum_lengths(b) = ()
_datum_lengths(b::PerSystem) = _per_system_lengths(b.data)
_per_system_lengths(data) = ()
_per_system_lengths(data::AbstractArray{<:StaticVector}) = (length(eltype(data)),)
_per_system_lengths(data::AbstractArray{<:AbstractVector}) = (length(bi) for bi in data)

function solve_batch!(
        X::AbstractArray, prob::StationarityProblem, method = Newton(), p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing, b = nothing, kwargs...)
    b_src = b === nothing ? prob.b : b
    _validate_batch(X, p, order)
    b_src isa PerSystem && _validate_batch(X, b_src, nothing)
    T = eltype(eltype(X))
    R = real(T)
    opt = resolve_options(T, options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    _validate_stationarity_batch(X, prob, b_src, p)
    isempty(X) && return _tally_return_codes(ReturnCode[])
    idx₀ = firstindex(X)
    p₀ = _sample_batch_item(p, idx₀)
    state₀ = init_state(m, prob, zero(eltype(X)), p₀; b = _sample_batch_item(b_src, idx₀))
    codes = allocate(get_backend(X), ReturnCode, length(X))
    sp = _StationarityBatchParams(p, b_src)
    return _tally_return_codes(Array(_launch!(X, codes, prob, m, sp, opt, state₀, order)))
end

function solve_batch!(
        X::AbstractArray, S::AbstractArray{SolverStatus{R}},
        prob::StationarityProblem, method = Newton(), p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing, b = nothing,
        kwargs...) where {R <: Real}
    b_src = b === nothing ? prob.b : b
    _validate_batch(X, p, order)
    b_src isa PerSystem && _validate_batch(X, b_src, nothing)
    R_X = real(eltype(eltype(X)))
    R === R_X || throw(ArgumentError(
        "solve_batch! requires the status vector S to use the real type $R_X of the batch iterates, got SolverStatus{$R}"))
    length(S) == length(X) || throw(DimensionMismatch(
        "status vector S has length $(length(S)), expected $(length(X)) to match the batch X"))
    get_backend(S) == get_backend(X) || throw(ArgumentError(
        "status vector S must reside on the same backend as the batch X ($(get_backend(X))), got $(get_backend(S))"))
    opt = resolve_options(eltype(eltype(X)), options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    _validate_stationarity_batch(X, prob, b_src, p)
    isempty(X) && return S
    idx₀ = firstindex(X)
    p₀ = _sample_batch_item(p, idx₀)
    state₀ = init_state(m, prob, zero(eltype(X)), p₀; b = _sample_batch_item(b_src, idx₀))
    sp = _StationarityBatchParams(p, b_src)
    return _launch!(X, S, prob, m, sp, opt, state₀, order)
end

# `CommonSolve.init` and `CommonSolve.solve` on `StationarityProblem`, the counterparts of the
# `NonlinearProblem` methods in `src/nonlinear/solver.jl`. `init` and `solve` are `CommonSolve`'s,
# imported in `src/GeometricSolvers.jl`, so these definitions add methods to them for a problem
# type of this package instead of defining a second `init` or `solve`.

# Implicit differentiation of a `StationarityProblem` is not implemented, so `init` and `solve`
# refuse a `ForwardDiff.Dual` in `x₀` or `p`, as `solve!` does, rather than return a primal result.
function _refuse_stationarity_duals(f::Symbol, x₀, p)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p))
        throw(ArgumentError(
            "$f does not accept ForwardDiff.Dual inputs for a StationarityProblem; implicit differentiation of a StationarityProblem is not implemented"))
    end
    return nothing
end

function init(pr::StatProb, x₀, m, p = nothing; b = nothing, options = nothing, kw...)
    _refuse_stationarity_duals(:init, x₀, p)
    return _init_stationarity(pr, x₀, m, p, b, options, kw)
end

init(pr::StatProb, x₀; kw...) = init(pr, x₀, Newton(), nothing; kw...)

function solve(pr::StatProb, x₀, m, p = nothing; b = nothing, options = nothing, kw...)
    _refuse_stationarity_duals(:solve, x₀, p)
    return _solve_stationarity(pr, x₀, m, p, b, options, kw)
end

solve(pr::StatProb, x₀; kw...) = solve(pr, x₀, Newton(), nothing; kw...)
