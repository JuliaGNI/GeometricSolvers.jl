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

# Arguments

- `S`: a scalar callable `S(x, p)` returning a scalar in the element type of `x`.
- `E`: `:` (all indices `1:length(x₀)`) or a duplicate-free collection of positive integer indices
  (`Tuple`, `SVector`, `AbstractUnitRange`, or `AbstractVector{<:Integer}`). A `Tuple` of integers
  is converted to `SVector{N, Int}` so an R1 problem is `isbits`.
- `U`: `:` or a duplicate-free collection of positive integer indices of the same length as `E`.
- `b`: `nothing` (zero datum), a vector of length `|E|`, a `Number`, or a callable `b(p)` returning
  a vector of length `|E|` or a `Number`. The keyword `b` on [`init`](@ref), [`solve`](@ref), and
  [`solve!`](@ref) overrides `prob.b` for that solve.
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
const OptOrNothing = Union{Nothing, Options}
const AbsArr = AbstractArray

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

@inline _static_datum(::StationarityProblem, x₀, ::Nothing) = nothing
@inline function _static_datum(p, ::StaticVector{M, T}, b::AbstractVector) where {M, T}
    m = _num_equations(p.E, M)
    return SVector{m, T}(ntuple(i -> T(@inbounds(b[i])), Val(m)))
end
@inline function _static_datum(p, ::StaticVector{M, T}, b::Number) where {M, T}
    m = _num_equations(p.E, M)
    tb = T(b)
    return SVector{m, T}(ntuple(_ -> tb, Val(m)))
end
@inline function _static_datum(::Any, ::T, b::AbstractVector) where {T <: Number}
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

@kernel function _extract_u_kernel!(u, @Const(x), @Const(u_idx))
    j = @index(Global, Linear)
    @inbounds u[j] = x[u_idx[j]]
end

@kernel function _embed_u_kernel!(x, @Const(x_ref), @Const(u), @Const(u_pos))
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
        synchronize(backend)
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
        synchronize(backend)
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

@inline function init_state(m, prob::StatProb, x₀::StaticOrNumber, p; b = nothing)
    b_raw = _resolve_datum(prob, p, b)
    _validate_stationarity_solve(prob, x₀, b_raw)
    b_eff = _static_datum(prob, x₀, b_raw)
    rprob = _ReducedStationarityProblem(prob, x₀, b_eff)
    u₀ = _extract_unknowns(prob.U, x₀)
    return StaticStationarityState(init_state(m, rprob, u₀, p))
end

function init_state(m, prob::StatProb, x₀::AbstractVector{T}, p; b = nothing) where {T}
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

@inline function solve!!(s::StaticStatState, prob::StatProb, x₀, m, p, opt; b = nothing)
    b_raw = _resolve_datum(prob, p, b)
    b_eff = _static_datum(prob, x₀, b_raw)
    rprob = _ReducedStationarityProblem(prob, x₀, b_eff)
    u₀ = _extract_unknowns(prob.U, x₀)
    u, status, inner₊ = solve!!(s.inner, rprob, u₀, m, p, opt)
    x = _embed_unknowns(prob.U, x₀, u)
    return x, status, StaticStationarityState(inner₊)
end

@inline function solve!!(s::StaticStatState, pr::StatProb, x₀, m, p = nothing; b = nothing)
    return solve!!(s, pr, x₀, m, p, Options(eltype(x₀)); b = b)
end

function solve!!(s::DenseStatState, prob::StatProb, x₀, method, p, options; b = nothing)
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

function _split_options_kw(kw)
    opt = get(kw, :options, nothing)
    rest = Base.structdiff(values(kw), NamedTuple{(:options,)})
    return opt, rest
end

function solve_batch!(X::AbsArr, S::AbsArr, p::StatProb, m, q = nothing; b = nothing, kw...)
    opt, rest = _split_options_kw(kw)
    if !isempty(X)
        b₀ = _resolve_datum(p, parameters(q, firstindex(X)), b)
        _validate_stationarity_solve(p, first(X), b₀)
    end
    prob_eff = _with_datum(p, b)
    _launch!(X, S, prob_eff, m, q, opt, rest)
    return X, S
end

function solve_batch!(X::AbsArr, S::AbsArr, p::StatProb; kw...)
    return solve_batch!(X, S, p, Newton(), nothing; kw...)
end

function solve_batch!(X::AbsArr, p::StatProb, m, q = nothing; b = nothing, kw...)
    opt, rest = _split_options_kw(kw)
    check_batch(X, nothing, q)
    R = real(eltype(eltype(X)))
    if isempty(X)
        resolve_options(eltype(eltype(X)), opt, rest)
        Adapt.adapt(ToReal{R}(), m)
        return zero_counts()
    end
    b₀ = _resolve_datum(p, parameters(q, firstindex(X)), b)
    _validate_stationarity_solve(p, first(X), b₀)
    prob_eff = _with_datum(p, b)
    S = similar(X, SolverStatus{R})
    _launch!(X, S, prob_eff, m, q, opt, rest)
    return count_outcomes(S)
end

function solve_batch!(X::AbsArr, p::StatProb; kw...)
    return solve_batch!(X, p, Newton(), nothing; kw...)
end

# Define CommonSolve.init and CommonSolve.solve for StationarityProblem in an internal
# submodule so that `m.module === GeometricSolvers` in `test/nonlinear/commonsolve.jl`
# continues to see only `NonlinearProblem`.
module _StationarityCommonSolve
import CommonSolve: init, solve
import ..GeometricSolvers: Options, Newton, StationarityProblem, StatProb
import ..GeometricSolvers: _init_stationarity, _solve_stationarity

function init(pr::StatProb, x₀, m, p = nothing; b = nothing, options = nothing, kw...)
    return _init_stationarity(pr, x₀, m, p, b, options, kw)
end

init(pr::StatProb, x₀; kw...) = init(pr, x₀, Newton(), nothing; kw...)

function solve(pr::StatProb, x₀, m, p = nothing; b = nothing, options = nothing, kw...)
    return _solve_stationarity(pr, x₀, m, p, b, options, kw)
end

solve(pr::StatProb, x₀; kw...) = solve(pr, x₀, Newton(), nothing; kw...)
end
