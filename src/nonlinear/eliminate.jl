# Block elimination inside a single nonlinear or stationarity system (`BlockEliminated`) and
# Golub–Pereyra / Kaufman variable projection for separable least squares (`VariableProjection`).

"""
    BlockEliminated(prob, inner; outer = nothing, inner_eqs = inner, outer_eqs = outer,
                    inner_method = Newton(; linsolve = SVDFactorization()),
                    linear_inner = false, inner_options = nothing)

Partition the unknowns `x` of a [`NonlinearProblem`](@ref) or [`StationarityProblem`](@ref) `prob`
into inner indices `Y = inner` (`y = x[Y]`) and outer indices `Z = outer` (`z = x[Z]`, defaulting
to the complement of `inner` in `1:n`), and the residual equations into inner equations
`E_y = inner_eqs` and outer equations `E_z = outer_eqs` (`|E_y| == |Y|`, `|E_z| == |Z|`). On a
[`LeastSquaresProblem`](@ref), `BlockEliminated(prob, inner; kwargs...)` forwards to
[`VariableProjection`](@ref).

At each outer iterate `z`, the inner equations ``F_{E_y}(y, z) = 0`` are solved for ``y^*(z)``
(in one linear solve ``y^*(z) = -J_{yy}(0, z)^+ F_{E_y}(0, z)`` when `linear_inner = true`, or by
warm-starting from the previous inner solution `y_warm` with adaptive tolerance
``\\tau_{\\text{inner}} \\le 0.5 \\|F_{E_z}(\\text{prev})\\|_2^2`` when `linear_inner = false`), and
the outer solver steps on the reduced residual ``F_{\\text{red}}(z) = F_{E_z}(y^*(z), z)`` using the
Schur complement

```math
S(z) = J_{zz}(y^*(z), z) - J_{zy}(y^*(z), z)\\, J_{yy}(y^*(z), z)^+\\, J_{yz}(y^*(z), z).
```

At return, ``x^*[Y] = y^*(z^*)`` and ``x^*[Z] = z^*``. Any inner solver failure or rank-deficient
inner Jacobian (`fac_yy.rank < |Y|` under SVD) increments `status.step_failures`.
"""
struct BlockEliminated{P <: AbstractNonlinearProblem, YI, ZI, EY, EZ, M, LI, IO} <:
       AbstractNonlinearProblem
    prob::P
    inner::YI
    outer::ZI
    inner_eqs::EY
    outer_eqs::EZ
    inner_method::M
    inner_options::IO
end

@inline function Base.getproperty(
        elim::BlockEliminated{P, YI, ZI, EY, EZ, M, LI, IO}, s::Symbol) where {
        P, YI, ZI, EY, EZ, M, LI, IO}
    s === :linear_inner && return LI
    return getfield(elim, s)
end

function Base.propertynames(::BlockEliminated)
    return (:prob, :inner, :outer, :inner_eqs, :outer_eqs,
        :inner_method, :linear_inner, :inner_options)
end

"""
    VariableProjection(prob::LeastSquaresProblem, inner; outer = nothing,
                       linsolve = SVDFactorization(), jacobian_mode = :kaufman,
                       linear_inner = true,
                       inner_method = GaussNewton(; linsolve = SVDFactorization()),
                       inner_options = nothing)

Golub–Pereyra variable projection for a [`LeastSquaresProblem`](@ref) `prob`, partitioning the
unknowns `x` into inner (linear or fast) parameters `Y = inner` (`y = x[Y]`) and outer parameters
`Z = outer` (`z = x[Z]`, defaulting to the complement of `inner` in `1:n`) across all `m` residual
rows.

For each outer iterate `z`, the inner subproblem ``\\min_y \\tfrac{1}{2}\\|r(y, z)\\|_2^2`` is
solved for ``y^*(z)`` (in one linear least-squares solve ``y^*(z) = -\\Phi(z)^+ r(0, z)`` with
``\\Phi(z) = \\partial r / \\partial y`` when `linear_inner = true`, or via `inner_method` warm-
started from `y_warm` when `linear_inner = false`). The outer solver steps on the reduced residual
``r_{\\text{red}}(z) = r(y^*(z), z)`` with reduced Jacobian (`m × |Z|`):

- `jacobian_mode = :kaufman` (default): Kaufman's first-term reduced Jacobian
  ``J_{\\text{red}}(z) = (I - \\Phi(z)\\Phi(z)^+) D(z)``, where ``D(z) = \\partial r(y^*(z), z)/\\partial z``.
- `jacobian_mode = :golub_pereyra`: the full Golub–Pereyra reduced Jacobian
  ``J_{\\text{red}}(z) = (I - \\Phi(z)\\Phi(z)^+) D(z) - (\\Phi(z)^+)^H (\\partial\\Phi/\\partial z)^H r_{\\text{red}}(z)``.
"""
struct VariableProjection{P <: LeastSquaresProblem, YI, ZI, LS, JM, LI, M, IO} <:
       AbstractNonlinearProblem
    prob::P
    inner::YI
    outer::ZI
    linsolve::LS
    inner_method::M
    inner_options::IO
end

@inline function Base.getproperty(
        vp::VariableProjection{P, YI, ZI, LS, JM, LI, M, IO}, s::Symbol) where {
        P, YI, ZI, LS, JM, LI, M, IO}
    s === :jacobian_mode && return JM
    s === :linear_inner && return LI
    return getfield(vp, s)
end

function Base.propertynames(::VariableProjection)
    return (:prob, :inner, :outer, :linsolve, :jacobian_mode,
        :linear_inner, :inner_method, :inner_options)
end

# --- Partition normalization and validation ---

function _normalize_part_indices(idx, name::Symbol)
    idx === nothing && return nothing
    if idx isa Tuple
        isempty(idx) && throw(ArgumentError("$name indices must be non-empty"))
        all(x -> x isa Integer, idx) ||
            throw(ArgumentError("$name indices must be integers"))
        return map(Int, idx)
    elseif idx isa StaticVector{<:Any, <:Integer}
        isempty(idx) && throw(ArgumentError("$name indices must be non-empty"))
        return Tuple(Int.(idx))
    elseif idx isa AbstractUnitRange{<:Integer}
        isempty(idx) && throw(ArgumentError("$name indices must be non-empty"))
        if length(idx) <= 32
            f = Int(first(idx))
            return ntuple(i -> f + i - 1, length(idx))
        else
            return collect(Int, idx)
        end
    elseif idx isa AbstractVector{<:Integer}
        isempty(idx) && throw(ArgumentError("$name indices must be non-empty"))
        return Int[Int(k) for k in idx]
    else
        throw(ArgumentError(
            "$name must be a non-empty Tuple, UnitRange, or AbstractVector of positive integers"))
    end
end

function _check_unique_positive(idx, name::Symbol)
    n = length(idx)
    n >= 1 || throw(ArgumentError("$name must be non-empty"))
    @inbounds for i in 1:n
        v = idx[i]
        v >= 1 || throw(ArgumentError("$name indices must be >= 1, got $v"))
        for j in 1:(i - 1)
            idx[j] == v &&
                throw(ArgumentError("$name contains duplicate index $v"))
        end
    end
    return nothing
end

function _check_disjoint_cover(a, b, name_a::Symbol, name_b::Symbol)
    na = length(a)
    nb = length(b)
    ntot = na + nb
    @inbounds for i in 1:na
        ai = a[i]
        ai <= ntot || throw(ArgumentError(
            "$name_a index $ai exceeds combined partition length $ntot"))
        for j in 1:nb
            b[j] == ai && throw(ArgumentError(
                "$name_a and $name_b must be disjoint; index $ai appears in both"))
        end
    end
    @inbounds for j in 1:nb
        bj = b[j]
        bj <= ntot || throw(ArgumentError(
            "$name_b index $bj exceeds combined partition length $ntot"))
    end
    return nothing
end

function _validate_block_eliminated_ctor(inner, outer, inner_eqs, outer_eqs, inner_options)
    _check_unique_positive(inner, :inner)
    _check_unique_positive(inner_eqs, :inner_eqs)
    length(inner_eqs) == length(inner) || throw(ArgumentError(
        "inner_eqs length ($(length(inner_eqs))) must match inner length ($(length(inner)))"))
    if outer !== nothing
        _check_unique_positive(outer, :outer)
        _check_disjoint_cover(inner, outer, :inner, :outer)
        ntot = length(inner) + length(outer)
        @inbounds for k in inner_eqs
            k <= ntot || throw(ArgumentError(
                "inner_eqs index $k exceeds total dimension $ntot"))
        end
    end
    if outer_eqs !== nothing
        _check_unique_positive(outer_eqs, :outer_eqs)
        _check_disjoint_cover(inner_eqs, outer_eqs, :inner_eqs, :outer_eqs)
        if outer !== nothing && length(outer_eqs) != length(outer)
            throw(ArgumentError(
                "outer_eqs length ($(length(outer_eqs))) must match outer length ($(length(outer)))"))
        end
    end
    inner_options === nothing || inner_options isa Options ||
        throw(ArgumentError(
            "inner_options must be `nothing` or an `Options`, got $(typeof(inner_options))"))
    return nothing
end

function _validate_varpro_ctor(inner, outer, jacobian_mode::Symbol, inner_options)
    _check_unique_positive(inner, :inner)
    if outer !== nothing
        _check_unique_positive(outer, :outer)
        _check_disjoint_cover(inner, outer, :inner, :outer)
    end
    (jacobian_mode === :kaufman || jacobian_mode === :golub_pereyra) ||
        throw(ArgumentError(
            "VariableProjection requires jacobian_mode ∈ (:kaufman, :golub_pereyra), got $(repr(jacobian_mode))"))
    inner_options === nothing || inner_options isa Options ||
        throw(ArgumentError(
            "inner_options must be `nothing` or an `Options`, got $(typeof(inner_options))"))
    return nothing
end

function BlockEliminated(
        prob::Union{NonlinearProblem, StationarityProblem},
        inner;
        outer = nothing,
        inner_eqs = inner,
        outer_eqs = outer,
        inner_method = Newton(; linsolve = SVDFactorization()),
        linear_inner::Bool = false,
        inner_options = nothing
)
    inner_n = _normalize_part_indices(inner, :inner)
    outer_n = _normalize_part_indices(outer, :outer)
    ieqs_n = _normalize_part_indices(inner_eqs, :inner_eqs)
    oeqs_n = _normalize_part_indices(outer_eqs, :outer_eqs)
    _validate_block_eliminated_ctor(inner_n, outer_n, ieqs_n, oeqs_n, inner_options)
    return BlockEliminated{
        typeof(prob), typeof(inner_n), typeof(outer_n),
        typeof(ieqs_n), typeof(oeqs_n), typeof(inner_method),
        linear_inner, typeof(inner_options)}(
        prob, inner_n, outer_n, ieqs_n, oeqs_n, inner_method, inner_options)
end

function VariableProjection(
        prob::LeastSquaresProblem,
        inner;
        outer = nothing,
        linsolve = SVDFactorization(),
        jacobian_mode::Symbol = :kaufman,
        linear_inner::Bool = true,
        inner_method = GaussNewton(; linsolve = SVDFactorization()),
        inner_options = nothing
)
    inner_n = _normalize_part_indices(inner, :inner)
    outer_n = _normalize_part_indices(outer, :outer)
    _validate_varpro_ctor(inner_n, outer_n, jacobian_mode, inner_options)
    return VariableProjection{
        typeof(prob), typeof(inner_n), typeof(outer_n),
        typeof(linsolve), jacobian_mode, linear_inner,
        typeof(inner_method), typeof(inner_options)}(
        prob, inner_n, outer_n, linsolve, inner_method, inner_options)
end

function BlockEliminated(prob::LeastSquaresProblem, inner; kwargs...)
    return VariableProjection(prob, inner; kwargs...)
end

@inline function _adapt_inner_prob(to, p::NonlinearProblem)
    return NonlinearProblem(
        Adapt.adapt(to, p.F), Adapt.adapt(to, p.J), Adapt.adapt(to, p.jacobian_prototype))
end
@inline function _adapt_inner_prob(to, p::LeastSquaresProblem)
    return LeastSquaresProblem(Adapt.adapt(to, p.F), Adapt.adapt(to, p.J), p.output_length)
end
@inline function _adapt_inner_prob(to, p::StationarityProblem)
    return StationarityProblem{typeof(Adapt.adapt(to, p.S)), typeof(p.E),
        typeof(p.U), typeof(Adapt.adapt(to, p.b))}(
        Adapt.adapt(to, p.S), p.E, p.U, Adapt.adapt(to, p.b))
end
@inline _adapt_inner_prob(to, p) = Adapt.adapt(to, p)

@inline _adapt_inner_options(_, ::Nothing) = nothing
@inline function _adapt_inner_options(to, opt::Options)
    f_abs = Adapt.adapt(to, opt.f_abstol)
    f_rel = Adapt.adapt(to, opt.f_reltol)
    x_abs = Adapt.adapt(to, opt.x_abstol)
    x_rel = Adapt.adapt(to, opt.x_reltol)
    g_t = Adapt.adapt(to, opt.g_tol)
    R = typeof(f_abs)
    return Options{R}(
        f_abs, f_rel, x_abs, x_rel,
        opt.min_iterations, opt.max_iterations, opt.max_stalls, g_t)
end

function Adapt.adapt_structure(
        to, elim::BlockEliminated{P, YI, ZI, EY, EZ, M, LI, IO}) where {
        P, YI, ZI, EY, EZ, M, LI, IO}
    prob = _adapt_inner_prob(to, getfield(elim, :prob))
    m = Adapt.adapt(to, getfield(elim, :inner_method))
    iopt = _adapt_inner_options(to, getfield(elim, :inner_options))
    return BlockEliminated{
        typeof(prob), YI, ZI, EY, EZ, typeof(m), LI, typeof(iopt)}(
        prob, getfield(elim, :inner), getfield(elim, :outer),
        getfield(elim, :inner_eqs), getfield(elim, :outer_eqs), m, iopt)
end

function Adapt.adapt_structure(
        to, vp::VariableProjection{P, YI, ZI, LS, JM, LI, M, IO}) where {
        P, YI, ZI, LS, JM, LI, M, IO}
    prob = _adapt_inner_prob(to, getfield(vp, :prob))
    ls = Adapt.adapt(to, getfield(vp, :linsolve))
    m = Adapt.adapt(to, getfield(vp, :inner_method))
    iopt = _adapt_inner_options(to, getfield(vp, :inner_options))
    return VariableProjection{
        typeof(prob), YI, ZI, typeof(ls), JM, LI, typeof(m), typeof(iopt)}(
        prob, getfield(vp, :inner), getfield(vp, :outer), ls, m, iopt)
end

# Datum `b` support on `BlockEliminated` wrapping a `StationarityProblem`
function _with_datum(elim::BlockEliminated, b)
    b === nothing || throw(ArgumentError(
        "keyword `b` is only supported for a `BlockEliminated` `StationarityProblem`"))
    return elim
end

function _with_datum(
        elim::BlockEliminated{P, YI, ZI, EY, EZ, M, LI, IO},
        b) where {P <: StationarityProblem, YI, ZI, EY, EZ, M, LI, IO}
    prob = _with_datum(getfield(elim, :prob), b)
    return BlockEliminated{typeof(prob), YI, ZI, EY, EZ, M, LI, IO}(
        prob, getfield(elim, :inner), getfield(elim, :outer),
        getfield(elim, :inner_eqs), getfield(elim, :outer_eqs),
        getfield(elim, :inner_method), getfield(elim, :inner_options))
end

function _with_datum(vp::VariableProjection, b)
    b === nothing || throw(ArgumentError(
        "keyword `b` is not supported for `VariableProjection`"))
    return vp
end

# --- Solve-time partition resolution ---

@inline function _validate_x_partition(inner, outer, n::Int)
    ny = length(inner)
    n > ny || throw(ArgumentError(
        "iterate length must exceed inner dimension so outer variables are non-empty"))
    @inbounds for i in 1:ny
        inner[i] <= n || throw(ArgumentError(
            "inner index exceeds iterate length"))
    end
    if outer !== nothing
        nz = length(outer)
        ny + nz == n || throw(ArgumentError(
            "inner length + outer length must equal iterate length"))
        @inbounds for j in 1:nz
            outer[j] <= n || throw(ArgumentError(
                "outer index exceeds iterate length"))
        end
    end
    return nothing
end

@inline function _validate_elim_x(elim::BlockEliminated, x, p = nothing)
    x isa AbstractVector || throw(ArgumentError(
        "BlockEliminated requires a vector iterate x₀"))
    n = length(x)
    _validate_x_partition(elim.inner, elim.outer, n)
    _validate_x_partition(elim.inner_eqs, elim.outer_eqs, n)
    prob_in = getfield(elim, :prob)
    if prob_in isa StationarityProblem
        _validate_stationarity_solve(prob_in, x, _resolve_datum(prob_in, p, nothing))
    end
    return nothing
end

@inline function _validate_elim_x(vp::VariableProjection, x, _ = nothing)
    x isa AbstractVector || throw(ArgumentError(
        "VariableProjection requires a vector iterate x₀"))
    n = length(x)
    _validate_x_partition(vp.inner, vp.outer, n)
    if vp.prob.output_length !== nothing
        m = vp.prob.output_length
        m >= n || throw(DimensionMismatch(
            "VariableProjection requires m >= n"))
    end
    return nothing
end

@inline function _complement_r1(
        Y::SVector{Ny, Int}, ::Val{N}, ::Val{Nz}) where {Ny, N, Nz}
    z = local_array(zero(SVector{Nz, Int}))
    k = 0
    @inbounds for i in 1:N
        found = false
        for j in 1:Ny
            if Y[j] == i
                found = true
                break
            end
        end
        if !found
            k += 1
            z[k] = i
        end
    end
    return SVector{Nz, Int}(z)
end

@inline function _to_svec_r1(idx::NTuple{K, Int}, ::Val{K}) where {K}
    return SVector{K, Int}(idx)
end
@inline function _to_svec_r1(idx::AbstractVector{Int}, ::Val{K}) where {K}
    return SVector{K, Int}(ntuple(i -> @inbounds(idx[i]), Val(K)))
end

@inline _part_len(::NTuple{K, Int}) where {K} = Val(K)
@inline _part_len(v::AbstractVector{Int}) = Val(length(v))

@inline function _resolve_partition_r1(
        elim::BlockEliminated, x::StaticVector{N}, p = nothing,
        ::Val{Ny} = _part_len(elim.inner)) where {N, Ny}
    _validate_elim_x(elim, x, p)
    Nz = N - Ny
    Y = _to_svec_r1(elim.inner, Val(Ny))
    Z = elim.outer === nothing ? _complement_r1(Y, Val(N), Val(Nz)) :
        _to_svec_r1(elim.outer, Val(Nz))
    E_y = _to_svec_r1(elim.inner_eqs, Val(Ny))
    E_z = elim.outer_eqs === nothing ? _complement_r1(E_y, Val(N), Val(Nz)) :
          _to_svec_r1(elim.outer_eqs, Val(Nz))
    return Y, Z, E_y, E_z
end

@inline function _resolve_partition_r1(
        vp::VariableProjection, x::StaticVector{N}, p = nothing,
        ::Val{Ny} = _part_len(vp.inner)) where {N, Ny}
    _validate_elim_x(vp, x, p)
    Nz = N - Ny
    Y = _to_svec_r1(vp.inner, Val(Ny))
    Z = vp.outer === nothing ? _complement_r1(Y, Val(N), Val(Nz)) :
        _to_svec_r1(vp.outer, Val(Nz))
    return Y, Z
end

function _complement_r3(Y_vec::Vector{Int}, n::Int)
    present = falses(n)
    @inbounds for idx in Y_vec
        present[idx] = true
    end
    Z_vec = Vector{Int}(undef, n - length(Y_vec))
    k = 0
    @inbounds for i in 1:n
        if !present[i]
            k += 1
            Z_vec[k] = i
        end
    end
    return Z_vec
end

function _resolve_partition_r3(elim::BlockEliminated, x::AbstractVector, p = nothing)
    _validate_elim_x(elim, x, p)
    n = length(x)
    Y_h = collect(Int, elim.inner)
    Z_h = elim.outer === nothing ? _complement_r3(Y_h, n) : collect(Int, elim.outer)
    Ey_h = collect(Int, elim.inner_eqs)
    Ez_h = elim.outer_eqs === nothing ? _complement_r3(Ey_h, n) :
           collect(Int, elim.outer_eqs)
    return Y_h, Z_h, Ey_h, Ez_h
end

function _resolve_partition_r3(vp::VariableProjection, x::AbstractVector, p = nothing)
    _validate_elim_x(vp, x, p)
    n = length(x)
    Y_h = collect(Int, vp.inner)
    Z_h = vp.outer === nothing ? _complement_r3(Y_h, n) : collect(Int, vp.outer)
    return Y_h, Z_h
end

# --- R1 scatter/gather and inner linear algebra helpers ---

@inline function _scatter_yz_r1(
        y::SVector{Ny, T}, z::SVector{Nz, T},
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int}, ::Val{N}) where {Ny, Nz, N, T}
    x = local_array(zero(SVector{N, T}))
    @inbounds for i in 1:Ny
        x[Y[i]] = y[i]
    end
    @inbounds for i in 1:Nz
        x[Z[i]] = z[i]
    end
    return SVector{N, T}(x)
end

@inline _inner_linsolve(m::Newton) = m.linsolve
@inline _inner_linsolve(ls::LinearMethod) = ls
@inline _inner_linsolve(_) = SVDFactorization()

@inline _is_rank_deficient(fac::Union{StaticSVD, StaticRectSVD}, ny::Int) = Int(getfield(fac, :rank)) <
                                                                            ny
@inline _is_rank_deficient(fac::Union{DenseSVD, DeviceSVD}, ny::Int) = fac.rank < ny
@inline _is_rank_deficient(_, ::Int) = false

# Apply (Φ^+)^H * w for StaticRectSVD / StaticSVD / StaticRectQR / StaticQR in R1:
# Given Φ of size M × Ny and w of length Ny, returns an SVector{M, T}.
@inline function _pinv_adj_mul_r1(
        fac::StaticRectSVD{M, Ny, TF, L, LN, R},
        ::SMatrix{M, Ny, T}, w::SVector{Ny, T}) where {M, Ny, TF, L, LN, R, T}
    U = getfield(fac, :U)
    V = getfield(fac, :V)
    σ̂ = getfield(fac, :σ̂)
    rk = Int(getfield(fac, :rank))
    e = getfield(fac, :scale_exponent)
    out = local_array(zero(SVector{M, T}))
    rk == 0 && return SVector{M, T}(out)
    scale = ldexp(one(R), -e)
    @inbounds for k in 1:rk
        ck = zero(T)
        for i in 1:Ny
            ck += conj(T(V[i, k])) * w[i]
        end
        ck = (ck * scale) / σ̂[k]
        for row in 1:M
            out[row] += T(U[row, k]) * ck
        end
    end
    return SVector{M, T}(out)
end

@inline function _pinv_adj_mul_r1(
        fac::StaticSVD{Ny, TF, R, L},
        ::SMatrix{Ny, Ny, T}, w::SVector{Ny, T}) where {Ny, TF, R, L, T}
    U = fac.factors
    V = fac.V
    σ̂ = fac.σ̂
    rk = Int(fac.rank)
    e = fac.scale_exponent
    out = local_array(zero(SVector{Ny, T}))
    rk == 0 && return SVector{Ny, T}(out)
    scale = ldexp(one(R), -e)
    @inbounds for k in 1:rk
        ck = zero(T)
        for i in 1:Ny
            ck += conj(T(V[i, k])) * w[i]
        end
        ck = (ck * scale) / σ̂[k]
        for row in 1:Ny
            out[row] += T(U[row, k]) * ck
        end
    end
    return SVector{Ny, T}(out)
end

@inline function _pinv_adj_mul_r1(
        fac, Φ::SMatrix{M, Ny, T}, w::SVector{Ny, T}) where {M, Ny, T}
    G = adjoint(Φ) * Φ
    fac_g = factorize!!(nothing, SVDFactorization(), G)
    u = ldiv!!(nothing, fac_g, w)
    return Φ * u
end

# --- R1 Inner Elimination & Reduced Operators ---

@inline function _inner_tol_r1(
        inner_options, ::Type{R}, fnorm_y0::R, fnorm_z_prev::R) where {R <: Real}
    τ_floor = R(10) * eps(R) * max(one(R), fnorm_y0)
    if inner_options === nothing
        max_it = Int32(50)
        if !isfinite(fnorm_z_prev)
            return τ_floor, max_it
        end
        inner_atol_cap = typemax(R)
    else
        opt_r = _adapt_inner_options(ToReal{R}(), inner_options)
        max_it = opt_r.max_iterations
        inner_atol_cap = max(opt_r.f_abstol, opt_r.f_reltol * max(one(R), fnorm_y0))
        if !isfinite(fnorm_z_prev)
            return max(τ_floor, inner_atol_cap), max_it
        end
    end
    τ_cap = R(0.5) * fnorm_z_prev * fnorm_z_prev
    return max(τ_floor, min(inner_atol_cap, τ_cap)), max_it
end

@inline function _schur_res_corr_r1(
        backend, ls_y, prob_in,
        y::SVector{Ny, T}, z::SVector{Nz, T}, x::SVector{N, T}, p,
        Fy::SVector{Ny, T}, Fz::SVector{Nz, T},
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int},
        E_y::SVector{Ny, Int}, E_z::SVector{Nz, Int},
        fnorm_y::R, τ_floor::R) where {N, Ny, Nz, T, R <: Real}
    (fnorm_y <= τ_floor || !isfinite(fnorm_y)) && return y, Fz, x, false
    J_full = _inner_jac_r1(backend, prob_in, x, p)
    J_yy = J_full[E_y, Y]
    fac_yy = factorize!!(nothing, ls_y, J_yy)
    fail_c = returncode(fac_yy) != SUCCESS || _is_rank_deficient(fac_yy, Ny)
    y_base = fail_c ? ldiv!!(nothing, fac_yy, J_yy * y) : y
    wy = ldiv!!(nothing, fac_yy, Fy)
    all(isfinite, wy) || return y_base, Fz, x, true
    y_corr = y_base - wy
    x_corr = _scatter_yz_r1(y_corr, z, Y, Z, Val(N))
    F_corr = _inner_res_r1(prob_in, x_corr, p)
    return y_corr, F_corr[E_z], x_corr, fail_c
end

@inline function _eliminate_r1(
        elim::BlockEliminated{P, YI, ZI, EY, EZ, M, true, IO}, _inner_state,
        ::SVector{Ny, T}, z::SVector{Nz, T}, p, ::R,
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int},
        E_y::SVector{Ny, Int}, E_z::SVector{Nz, Int},
        ::Union{Nothing, SVector{Nz, T}} = nothing) where {
        P, YI, ZI, EY, EZ, M, IO, Ny, Nz, T, R <: Real}
    N = Ny + Nz
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    backend = AutoForwardDiff()
    y_zero = zero(SVector{Ny, T})
    x_zero = _scatter_yz_r1(y_zero, z, Y, Z, Val(N))
    F0, J0 = _inner_res_jac_r1(backend, getfield(elim, :prob), x_zero, p)
    Fy0 = F0[E_y]
    J_yy = J0[E_y, Y]
    fac_yy = factorize!!(nothing, ls_y, J_yy)
    dy = ldiv!!(nothing, fac_yy, Fy0)
    y_star = -dy
    x_star = _scatter_yz_r1(y_star, z, Y, Z, Val(N))
    F_star = _inner_res_r1(getfield(elim, :prob), x_star, p)
    Fz = F_star[E_z]
    failed = (returncode(fac_yy) != SUCCESS) || _is_rank_deficient(fac_yy, Ny) ||
             !all(isfinite, y_star)
    return y_star, Fz, x_star, failed, 1
end

@inline function _eliminate_r1(
        elim::BlockEliminated{P, YI, ZI, EY, EZ, M, false, IO}, _inner_state,
        y_init::SVector{Ny, T}, z::SVector{Nz, T}, p, fnorm_z_prev::R,
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int},
        E_y::SVector{Ny, Int}, E_z::SVector{Nz, Int},
        z_prev::Union{Nothing, SVector{Nz, T}} = nothing) where {
        P, YI, ZI, EY, EZ, M, IO, Ny, Nz, T, R <: Real}
    N = Ny + Nz
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    backend = AutoForwardDiff()
    prob_in = getfield(elim, :prob)
    y = all(isfinite, y_init) ? y_init : zero(SVector{Ny, T})
    x = _scatter_yz_r1(y, z, Y, Z, Val(N))
    F_full = _inner_res_r1(prob_in, x, p)
    Fy = F_full[E_y]
    fnorm_y0 = R(rnorm(Fy))
    τ_floor = R(10) * eps(R) * max(one(R), fnorm_y0)
    τ_in, max_it = _inner_tol_r1(getfield(elim, :inner_options), R, fnorm_y0, fnorm_z_prev)
    if fnorm_y0 <= τ_in
        fail_0 = !isfinite(fnorm_y0)
        if !isfinite(fnorm_z_prev) && isfinite(fnorm_y0)
            J_full0 = _inner_jac_r1(backend, prob_in, x, p)
            J_yy0 = J_full0[E_y, Y]
            fac_yy0 = factorize!!(nothing, ls_y, J_yy0)
            if returncode(fac_yy0) != SUCCESS || _is_rank_deficient(fac_yy0, Ny)
                fail_0 = true
                y = ldiv!!(nothing, fac_yy0, J_yy0 * y)
                x = _scatter_yz_r1(y, z, Y, Z, Val(N))
                F_full = _inner_res_r1(prob_in, x, p)
                Fy = F_full[E_y]
            end
        end
        if fnorm_y0 > τ_floor && isfinite(fnorm_y0)
            y_c, Fz_c, x_c, fail_c = _schur_res_corr_r1(
                backend, ls_y, prob_in, y, z, x, p, Fy, F_full[E_z],
                Y, Z, E_y, E_z, fnorm_y0, τ_floor)
            return y_c, Fz_c, x_c, (fail_0 | fail_c), 0
        end
        return y, F_full[E_z], x, fail_0, 0
    end
    if z_prev !== nothing && all(isfinite, y_init) && z != z_prev && max_it >= 1
        x_prev = _scatter_yz_r1(y, z_prev, Y, Z, Val(N))
        F_prev, J_prev = _inner_res_jac_r1(backend, prob_in, x_prev, p)
        J_yy_p = J_prev[E_y, Y]
        J_yz_p = J_prev[E_y, Z]
        fac_p = factorize!!(nothing, ls_y, J_yy_p)
        fail_p = returncode(fac_p) != SUCCESS || _is_rank_deficient(fac_p, Ny)
        y_base = fail_p ? ldiv!!(nothing, fac_p, J_yy_p * y) : y
        dy_p = ldiv!!(nothing, fac_p, F_prev[E_y] + J_yz_p * (z - z_prev))
        if all(isfinite, dy_p)
            y_p = y_base - dy_p
            x_p = _scatter_yz_r1(y_p, z, Y, Z, Val(N))
            F_p = _inner_res_r1(prob_in, x_p, p)
            Fy_p = F_p[E_y]
            fnorm_p = R(rnorm(Fy_p))
            if fnorm_p <= τ_in
                y_c, Fz_p, x_c, fail_c = _schur_res_corr_r1(
                    backend, ls_y, prob_in, y_p, z, x_p, p, Fy_p, F_p[E_z],
                    Y, Z, E_y, E_z, fnorm_p, τ_floor)
                return y_c, Fz_p, x_c, (fail_p | fail_c), 1
            end
        end
    end
    failed = false
    iters = 0
    fnorm_y = fnorm_y0
    for _ in 1:max_it
        iters += 1
        J_full = _inner_jac_r1(backend, prob_in, x, p)
        J_yy = J_full[E_y, Y]
        fac_yy = factorize!!(nothing, ls_y, J_yy)
        if returncode(fac_yy) != SUCCESS || _is_rank_deficient(fac_yy, Ny)
            failed = true
            y = ldiv!!(nothing, fac_yy, J_yy * y)
        end
        dy = ldiv!!(nothing, fac_yy, Fy)
        if !all(isfinite, dy)
            failed = true
            break
        end
        α = one(R)
        y_next = y - dy
        x_next = _scatter_yz_r1(y_next, z, Y, Z, Val(N))
        F_next = _inner_res_r1(prob_in, x_next, p)
        Fy_next = F_next[E_y]
        fnorm_next = R(rnorm(Fy_next))
        for _ in 1:8
            (isfinite(fnorm_next) && fnorm_next <= fnorm_y) && break
            α *= R(0.5)
            y_next = y - T(α) * dy
            x_next = _scatter_yz_r1(y_next, z, Y, Z, Val(N))
            F_next = _inner_res_r1(prob_in, x_next, p)
            Fy_next = F_next[E_y]
            fnorm_next = R(rnorm(Fy_next))
        end
        y = y_next
        x = x_next
        F_full = F_next
        Fy = Fy_next
        fnorm_y = fnorm_next
        if fnorm_y <= τ_in || R(rnorm(dy)) <= R(2) * eps(R) * max(one(R), R(rnorm(y)))
            break
        end
    end
    if !(fnorm_y <= τ_in)
        failed = true
    end
    y_out, Fz_out, x_out, fail_c = _schur_res_corr_r1(
        backend, ls_y, prob_in, y, z, x, p, Fy, F_full[E_z],
        Y, Z, E_y, E_z, fnorm_y, τ_floor)
    return y_out, Fz_out, x_out, (failed | fail_c), iters
end

@inline function _schur_jacobian_r1(
        elim::BlockEliminated, x_star::SVector{N, T}, p,
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int},
        E_y::SVector{Ny, Int}, E_z::SVector{Nz, Int}) where {N, Ny, Nz, T}
    R = real(T)
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    J_full = _inner_jac_r1(AutoForwardDiff(), getfield(elim, :prob), x_star, p)
    J_yy = J_full[E_y, Y]
    J_yz = J_full[E_y, Z]
    J_zy = J_full[E_z, Y]
    J_zz = J_full[E_z, Z]
    fac_yy = factorize!!(nothing, ls_y, J_yy)
    W = local_array(zero(SMatrix{Ny, Nz, T, Ny * Nz}))
    @inbounds for j in 1:Nz
        col_j = SVector{Ny, T}(ntuple(i -> J_yz[i, j], Val(Ny)))
        w_j = ldiv!!(nothing, fac_yy, col_j)
        for i in 1:Ny
            W[i, j] = w_j[i]
        end
    end
    failed = (returncode(fac_yy) != SUCCESS) || _is_rank_deficient(fac_yy, Ny)
    return J_zz - J_zy * SMatrix{Ny, Nz, T, Ny * Nz}(W), failed
end

@inline function _eliminate_vp_r1(
        vp::VariableProjection{P, YI, ZI, LS, JM, true, M, IO},
        ::SVector{Ny, T}, z::SVector{Nz, T}, p,
        ::R, Y::SVector{Ny, Int},
        Z::SVector{Nz, Int}) where {
        P, YI, ZI, LS, JM, M, IO, Ny, Nz, T, R <: Real}
    N = Ny + Nz
    ls_y = Adapt.adapt(ToReal{R}(), getfield(vp, :linsolve))
    backend = AutoForwardDiff()
    prob_in = getfield(vp, :prob)
    y_zero = zero(SVector{Ny, T})
    x_zero = _scatter_yz_r1(y_zero, z, Y, Z, Val(N))
    r0 = prob_in.F(x_zero, p)
    J0 = _inner_jac_r1(backend, prob_in, x_zero, p)
    Φ = J0[:, Y]
    fac_phi = factorize!!(nothing, ls_y, Φ)
    dy = ldiv!!(nothing, fac_phi, r0)
    y_star = -dy
    x_star = _scatter_yz_r1(y_star, z, Y, Z, Val(N))
    r_star = prob_in.F(x_star, p)
    failed = (returncode(fac_phi) != SUCCESS) || _is_rank_deficient(fac_phi, Ny) ||
             !all(isfinite, y_star)
    return y_star, r_star, x_star, failed, 1
end

@inline function _eliminate_vp_r1(
        vp::VariableProjection{P, YI, ZI, LS, JM, false, M, IO},
        y_init::SVector{Ny, T}, z::SVector{Nz, T}, p,
        fnorm_z_prev::R, Y::SVector{Ny, Int},
        Z::SVector{Nz, Int}) where {
        P, YI, ZI, LS, JM, M, IO, Ny, Nz, T, R <: Real}
    N = Ny + Nz
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(vp, :inner_method)))
    backend = AutoForwardDiff()
    prob_in = getfield(vp, :prob)
    y = all(isfinite, y_init) ? y_init : zero(SVector{Ny, T})
    x = _scatter_yz_r1(y, z, Y, Z, Val(N))
    r_curr = prob_in.F(x, p)
    fnorm_r0 = R(rnorm(r_curr))
    inner_opt = getfield(vp, :inner_options)
    if inner_opt === nothing
        τ_in = R(10) * eps(R) * max(one(R), fnorm_r0)
        max_it = Int32(50)
    else
        τ_in, max_it = _inner_tol_r1(inner_opt, R, fnorm_r0, fnorm_z_prev)
    end
    failed = false
    converged = isfinite(fnorm_r0) && fnorm_r0 <= τ_in
    iters = 0
    for k in 1:(max_it + 1)
        J_full = _inner_jac_r1(backend, prob_in, x, p)
        Φ = J_full[:, Y]
        g_y = adjoint(Φ) * r_curr
        if R(rnorm(g_y)) <= τ_in * max(one(R), R(rnorm(Φ)))
            if k == 1 && !isfinite(fnorm_z_prev)
                fac_phi0 = factorize!!(nothing, ls_y, Φ)
                if returncode(fac_phi0) != SUCCESS || _is_rank_deficient(fac_phi0, Ny)
                    failed = true
                    y = ldiv!!(nothing, fac_phi0, Φ * y)
                    x = _scatter_yz_r1(y, z, Y, Z, Val(N))
                    r_curr = prob_in.F(x, p)
                    fnorm_r0 = R(rnorm(r_curr))
                end
            end
            converged = isfinite(fnorm_r0)
            break
        end
        k > max_it && break
        iters += 1
        fac_phi = factorize!!(nothing, ls_y, Φ)
        if returncode(fac_phi) != SUCCESS || _is_rank_deficient(fac_phi, Ny)
            failed = true
            y = ldiv!!(nothing, fac_phi, Φ * y)
        end
        dy = ldiv!!(nothing, fac_phi, r_curr)
        if !all(isfinite, dy)
            failed = true
            break
        end
        α = one(R)
        y_next = y - dy
        x_next = _scatter_yz_r1(y_next, z, Y, Z, Val(N))
        r_next = prob_in.F(x_next, p)
        fnorm_next = R(rnorm(r_next))
        for _ in 1:8
            (isfinite(fnorm_next) && fnorm_next <= fnorm_r0) && break
            α *= R(0.5)
            y_next = y - T(α) * dy
            x_next = _scatter_yz_r1(y_next, z, Y, Z, Val(N))
            r_next = prob_in.F(x_next, p)
            fnorm_next = R(rnorm(r_next))
        end
        y = y_next
        x = x_next
        r_curr = r_next
        fnorm_r0 = fnorm_next
        if isfinite(fnorm_r0) && fnorm_r0 <= τ_in
            converged = true
            break
        end
        if R(rnorm(dy)) <= R(2) * eps(R) * max(one(R), R(rnorm(y)))
            J_end = _inner_jac_r1(backend, prob_in, x, p)
            Φ_end = J_end[:, Y]
            if R(rnorm(adjoint(Φ_end) * r_curr)) <= τ_in * max(one(R), R(rnorm(Φ_end)))
                converged = isfinite(fnorm_r0)
            end
            break
        end
    end
    if !converged || !all(isfinite, y)
        failed = true
    end
    return y, r_curr, x, failed, iters
end

struct VarProOuterTag{F, R} end

@inline function _varpro_kaufman_r1(
        vp::VariableProjection, x_star::SVector{N, T}, p,
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int}, ::Val{M}) where {N, M, Ny, Nz, T}
    R = real(T)
    raw_ls = vp.linear_inner ? getfield(vp, :linsolve) :
             _inner_linsolve(getfield(vp, :inner_method))
    ls_y = Adapt.adapt(ToReal{R}(), raw_ls)
    J_full = _inner_jac_r1(AutoForwardDiff(), getfield(vp, :prob), x_star, p)
    Φ = J_full[:, Y]
    D = J_full[:, Z]
    fac_phi = factorize!!(nothing, ls_y, Φ)
    P_D = local_array(zero(SMatrix{M, Nz, T, M * Nz}))
    @inbounds for j in 1:Nz
        d_j = SVector{M, T}(ntuple(k -> D[k, j], Val(M)))
        c_j = ldiv!!(nothing, fac_phi, d_j)
        proj_j = Φ * c_j
        for k in 1:M
            P_D[k, j] = D[k, j] - proj_j[k]
        end
    end
    J_kaufman = SMatrix{M, Nz, T, M * Nz}(P_D)
    failed = (returncode(fac_phi) != SUCCESS) || _is_rank_deficient(fac_phi, Ny)
    return J_kaufman, Φ, fac_phi, failed
end

@inline function _varpro_jacobian_r1(
        vp::VariableProjection{P, YI, ZI, LS, :kaufman, LI, M_T, IO},
        x_star::SVector{N, T}, r_star::SVector{M, T}, p,
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int}) where {
        P, YI, ZI, LS, LI, M_T, IO, N, M, Ny, Nz, T}
    J_kaufman, _, _, failed = _varpro_kaufman_r1(vp, x_star, p, Y, Z, Val(M))
    return J_kaufman, failed
end

@inline function _varpro_jacobian_r1(
        vp::VariableProjection{P, YI, ZI, LS, :golub_pereyra, LI, M_T, IO},
        x_star::SVector{N, T}, r_star::SVector{M, T}, p,
        Y::SVector{Ny, Int}, Z::SVector{Nz, Int}) where {
        P, YI, ZI, LS, LI, M_T, IO, N, M, Ny, Nz, T}
    R = real(T)
    J_kaufman, Φ, fac_phi, failed = _varpro_kaufman_r1(vp, x_star, p, Y, Z, Val(M))
    prob_in = getfield(vp, :prob)
    TgOut = VarProOuterTag{typeof(prob_in.F), R}
    D_out = dual_type(T, TgOut, Nz)
    x_dual = SVector{N, D_out}(ntuple(Val(N)) do idx
        pos_z = _idx_pos(Z, idx)
        seed(zero(D_out), x_star[idx], ntuple(==(pos_z), Val(Nz)))
    end)
    J_dual = _inner_jac_r1(AutoForwardDiff(), prob_in, x_dual, p)
    GP = local_array(J_kaufman)
    @inbounds for j in 1:Nz
        w_j = local_array(zero(SVector{Ny, T}))
        for i in 1:Ny
            yi = Y[i]
            acc = zero(T)
            for k in 1:M
                dphi_ki = dual_partial(J_dual[k, yi], j)
                acc += conj(dphi_ki) * r_star[k]
            end
            w_j[i] = acc
        end
        term2_j = _pinv_adj_mul_r1(fac_phi, Φ, SVector{Ny, T}(w_j))
        for k in 1:M
            GP[k, j] -= term2_j[k]
        end
    end
    return SMatrix{M, Nz, T, M * Nz}(GP), failed
end

# --- R1 `residual!!`, `jacobian!!`, `jvp!!`, `prepare_ad` ---

@inline function prepare_ad(
        ::Nothing, prob::Union{BlockEliminated, VariableProjection},
        ::Nothing, x::Union{StaticVector, Number}, p)
    _validate_elim_x(prob, x, p)
    return AutoForwardDiff()
end

@inline function prepare_ad(
        backend::AutoForwardDiff, prob::Union{BlockEliminated, VariableProjection},
        ::Nothing, x::Union{StaticVector, Number}, p)
    _validate_elim_x(prob, x, p)
    return backend
end

@inline function prepare_ad(
        backend::AbstractADType, prob::Union{BlockEliminated, VariableProjection},
        ::Nothing, x::Union{StaticVector, Number}, p)
    _validate_elim_x(prob, x, p)
    return backend
end

@inline function residual!!(
        ::Nothing, elim::BlockEliminated, x::StaticVector{N, T}, p) where {N, T}
    R = real(T)
    Y, Z, E_y, E_z = _resolve_partition_r1(elim, x, p)
    _, Fz, _, _, _ = _eliminate_r1(
        elim, nothing, x[Y], x[Z], p, R(NaN), Y, Z, E_y, E_z)
    return Fz
end

@inline function residual!!(
        ::Nothing, vp::VariableProjection, x::StaticVector{N, T}, p) where {N, T}
    R = real(T)
    Y, Z = _resolve_partition_r1(vp, x, p)
    _, r_star, _, _, _ = _eliminate_vp_r1(vp, x[Y], x[Z], p, R(NaN), Y, Z)
    return r_star
end

@inline function _elim_jac_r1(elim::BlockEliminated, x::StaticVector{N, T}, p) where {N, T}
    R = real(T)
    Y, Z, E_y, E_z = _resolve_partition_r1(elim, x, p)
    _, _, x_star, _, _ = _eliminate_r1(
        elim, nothing, x[Y], x[Z], p, R(NaN), Y, Z, E_y, E_z)
    S, _ = _schur_jacobian_r1(elim, x_star, p, Y, Z, E_y, E_z)
    return S
end

@inline function _elim_jac_r1(vp::VariableProjection, x::StaticVector{N, T}, p) where {N, T}
    R = real(T)
    Y, Z = _resolve_partition_r1(vp, x, p)
    _, r_star, x_star, _, _ = _eliminate_vp_r1(vp, x[Y], x[Z], p, R(NaN), Y, Z)
    J_red, _ = _varpro_jacobian_r1(vp, x_star, r_star, p, Y, Z)
    return J_red
end

@inline function jacobian!!(
        ::Nothing, ::AutoForwardDiff,
        elim::BlockEliminated, x::StaticVector{N, T}, p) where {N, T}
    return _elim_jac_r1(elim, x, p)
end

@inline function jacobian!!(
        ::Nothing, ::Nothing,
        elim::BlockEliminated, x::StaticVector{N, T}, p) where {N, T}
    return _elim_jac_r1(elim, x, p)
end

@inline function jacobian!!(
        ::Nothing, ::AutoForwardDiff,
        vp::VariableProjection, x::StaticVector{N, T}, p) where {N, T}
    return _elim_jac_r1(vp, x, p)
end

@inline function jacobian!!(
        ::Nothing, ::Nothing,
        vp::VariableProjection, x::StaticVector{N, T}, p) where {N, T}
    return _elim_jac_r1(vp, x, p)
end

@inline function _jvp_vec_r1(
        elim::BlockEliminated, x::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    _, Z, _, _ = _resolve_partition_r1(elim, x, p)
    return v[Z]
end

@inline function _jvp_vec_r1(
        vp::VariableProjection, x::StaticVector{N, T},
        v::StaticVector{N, T}, p) where {N, T}
    _, Z = _resolve_partition_r1(vp, x, p)
    return v[Z]
end

@inline function _jvp_vec_r1(
        ::Union{BlockEliminated, VariableProjection},
        ::StaticVector{N, T}, v::StaticVector{Nz, T}, _) where {N, Nz, T}
    return v
end

@inline function jvp!!(
        ::Nothing, ::AutoForwardDiff,
        prob::Union{BlockEliminated, VariableProjection},
        x::StaticVector{N, T}, v::StaticVector{Nv, T}, p) where {N, Nv, T}
    return _elim_jac_r1(prob, x, p) * _jvp_vec_r1(prob, x, v, p)
end

@inline function jvp!!(
        ::Nothing, ::AutoForwardDiff,
        prob::Union{BlockEliminated, VariableProjection},
        x::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return _elim_jac_r1(prob, x, p) * _jvp_vec_r1(prob, x, v, p)
end

@inline function jvp!!(
        ::Nothing, ::Nothing,
        prob::Union{BlockEliminated, VariableProjection},
        x::StaticVector{N, T}, v::StaticVector{Nv, T}, p) where {N, Nv, T}
    return _elim_jac_r1(prob, x, p) * _jvp_vec_r1(prob, x, v, p)
end

@inline function jvp!!(
        ::Nothing, ::Nothing,
        prob::Union{BlockEliminated, VariableProjection},
        x::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return _elim_jac_r1(prob, x, p) * _jvp_vec_r1(prob, x, v, p)
end

# --- R3 Kernels and Helpers (0-allocation on Array, 0 scalar indexing on GPU/JLArray) ---

@kernel function _gather_vec_kernel!(dst, @Const(src), @Const(idx))
    i = @index(Global, Linear)
    @inbounds dst[i] = src[idx[i]]
end

@kernel function _scatter_yz_kernel!(
        x, @Const(y), @Const(z), @Const(Y_idx), @Const(Z_idx), ny::Int, nz::Int)
    i = @index(Global, Linear)
    @inbounds if i <= ny
        x[Y_idx[i]] = y[i]
    elseif i <= ny + nz
        k = i - ny
        x[Z_idx[k]] = z[k]
    end
end

@kernel function _extract_submat_kernel!(
        sub, @Const(J), @Const(row_idx), @Const(col_idx), nr::Int, nc::Int)
    idx = @index(Global, Linear)
    @inbounds if idx <= nr * nc
        r = rem(idx - 1, nr) + 1
        c = div(idx - 1, nr) + 1
        sub[r, c] = J[row_idx[r], col_idx[c]]
    end
end

@kernel function _extract_col_kernel!(col, @Const(M), j::Int, nr::Int)
    i = @index(Global, Linear)
    @inbounds if i <= nr
        col[i] = M[i, j]
    end
end

@kernel function _store_col_kernel!(M, @Const(col), j::Int, nr::Int)
    i = @index(Global, Linear)
    @inbounds if i <= nr
        M[i, j] = col[i]
    end
end

@kernel function _sub_col_kernel!(M, @Const(D), @Const(col), j::Int, nr::Int)
    i = @index(Global, Linear)
    @inbounds if i <= nr
        M[i, j] = D[i, j] - col[i]
    end
end

@kernel function _dec_col_kernel!(M, @Const(col), j::Int, nr::Int)
    i = @index(Global, Linear)
    @inbounds if i <= nr
        M[i, j] -= col[i]
    end
end

@kernel function _svd_pinv_adj_scale_kernel!(
        tmp, @Const(s_hat), rk::Int, ny::Int, scale)
    i = @index(Global, Linear)
    @inbounds if i <= ny
        tmp[i] = i <= rk ? (tmp[i] * scale) / s_hat[i] : zero(eltype(tmp))
    end
end

function _gather_vec!(
        dst::AbstractVector, src::AbstractVector,
        idx_h::Vector{Int}, idx_d::AbstractVector{Int})
    if src isa AbstractGPUArray
        backend = get_backend(src)
        _gather_vec_kernel!(backend)(dst, src, idx_d; ndrange = length(dst))
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in eachindex(dst, idx_h)
            dst[i] = src[idx_h[i]]
        end
    end
    return dst
end

function _scatter_yz!(
        x::AbstractVector, y::AbstractVector, z::AbstractVector,
        Y_h::Vector{Int}, Z_h::Vector{Int},
        Y_d::AbstractVector{Int}, Z_d::AbstractVector{Int})
    ny = length(Y_h)
    nz = length(Z_h)
    if x isa AbstractGPUArray
        backend = get_backend(x)
        _scatter_yz_kernel!(backend)(x, y, z, Y_d, Z_d, ny, nz; ndrange = ny + nz)
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in 1:ny
            x[Y_h[i]] = y[i]
        end
        @inbounds for j in 1:nz
            x[Z_h[j]] = z[j]
        end
    end
    return x
end

function _extract_submat!(
        sub::AbstractMatrix, J::AbstractMatrix,
        r_h::Vector{Int}, c_h::Vector{Int},
        r_d::AbstractVector{Int}, c_d::AbstractVector{Int})
    nr = length(r_h)
    nc = length(c_h)
    if J isa AbstractGPUArray
        backend = get_backend(J)
        _extract_submat_kernel!(backend)(sub, J, r_d, c_d, nr, nc; ndrange = nr * nc)
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for j in 1:nc
            cj = c_h[j]
            for i in 1:nr
                sub[i, j] = J[r_h[i], cj]
            end
        end
    end
    return sub
end

function _extract_col!(col::AbstractVector, M::AbstractMatrix, j::Int)
    nr = size(M, 1)
    if M isa AbstractGPUArray
        backend = get_backend(M)
        _extract_col_kernel!(backend)(col, M, j, nr; ndrange = nr)
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in 1:nr
            col[i] = M[i, j]
        end
    end
    return col
end

function _store_col!(M::AbstractMatrix, col::AbstractVector, j::Int)
    nr = size(M, 1)
    if M isa AbstractGPUArray
        backend = get_backend(M)
        _store_col_kernel!(backend)(M, col, j, nr; ndrange = nr)
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in 1:nr
            M[i, j] = col[i]
        end
    end
    return M
end

function _sub_col!(M::AbstractMatrix, D::AbstractMatrix, col::AbstractVector, j::Int)
    nr = size(M, 1)
    if M isa AbstractGPUArray
        backend = get_backend(M)
        _sub_col_kernel!(backend)(M, D, col, j, nr; ndrange = nr)
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in 1:nr
            M[i, j] = D[i, j] - col[i]
        end
    end
    return M
end

function _dec_col!(M::AbstractMatrix, col::AbstractVector, j::Int)
    nr = size(M, 1)
    if M isa AbstractGPUArray
        backend = get_backend(M)
        _dec_col_kernel!(backend)(M, col, j, nr; ndrange = nr)
        isgpu(backend) && synchronize(backend)
    else
        @inbounds for i in 1:nr
            M[i, j] -= col[i]
        end
    end
    return M
end

@inline function _pinv_adj_mul_r3!(
        out::AbstractVector{T}, fac::DenseSVD, w::AbstractVector{T}) where {T}
    rk = fac.rank
    if rk == 0
        fill!(out, zero(T))
        return out
    end
    mul!(fac.tmp, fac.VT, w)
    R = real(eltype(fac.tmp))
    scale = ldexp(one(R), -fac.scale_exponent)
    @inbounds for i in 1:length(fac.S)
        fac.tmp[i] = i <= rk ? (fac.tmp[i] * scale) / fac.rwork[i] : zero(eltype(fac.tmp))
    end
    mul!(out, fac.U, fac.tmp)
    return out
end

@inline function _pinv_adj_mul_r3!(
        out::AbstractVector{T}, fac::DeviceSVD, w::AbstractVector{T}) where {T}
    rk = fac.rank
    if rk == 0
        fill!(out, zero(T))
        return out
    end
    mul!(fac.tmp, fac.VT, w)
    R = real(eltype(fac.tmp))
    scale = ldexp(one(R), -fac.scale_exponent)
    ny = length(fac.S)
    backend = get_backend(fac.tmp)
    _svd_pinv_adj_scale_kernel!(backend)(fac.tmp, fac.σ̂, rk, ny, scale; ndrange = ny)
    isgpu(backend) && synchronize(backend)
    mul!(out, fac.U, fac.tmp)
    return out
end

# --- R3 Workspaces (`BlockElimPrep` and `VarProPrep`) ---

mutable struct BlockElimPrep{
    T, R <: Real, V <: AbstractVector{T}, M <: AbstractMatrix{T},
    IV <: AbstractVector{Int}, IP, FAC
}
    const x_buf::V
    const F_buf::V
    const J_buf::M
    const y_warm::V
    const y_trial::V
    const dy_buf::V
    const Fy_buf::V
    const Fz_buf::V
    const w_col::V
    const J_yy::M
    const J_yz::M
    const J_zy::M
    const J_zz::M
    const W_mat::M
    const Sz_jvp::M
    const z_curr::V
    const z_prev::V
    const Y_h::Vector{Int}
    const Z_h::Vector{Int}
    const Ey_h::Vector{Int}
    const Ez_h::Vector{Int}
    const Y_d::IV
    const Z_d::IV
    const Ey_d::IV
    const Ez_d::IV
    const inner_prep::IP
    const fac_yy::FAC
    fnorm_z_prev::R
    last_inner_iters::Int
    inner_failures::Int32
    init_failures::Int32
end

struct VarProInnerTag{F, R} end

@inline function _seed_vp_nest1(
        ::ForwardDiff.Dual{TgOut, ForwardDiff.Dual{TgIn, T, 1}, 1},
        xi::T, idx::Int, yi_idx::Int, zj_idx::Int) where {TgOut, TgIn, T <: Real}
    D_in = ForwardDiff.Dual{TgIn, T, 1}
    D_out = ForwardDiff.Dual{TgOut, D_in, 1}
    v_in = D_in(xi, ForwardDiff.Partials{1, T}((ifelse(idx == yi_idx, one(T), zero(T)),)))
    p_out = ForwardDiff.Partials{1, D_in}((ifelse(idx == zj_idx, one(D_in), zero(D_in)),))
    return D_out(v_in, p_out)
end

@inline function _seed_vp_nest1(
        ::Complex{ForwardDiff.Dual{TgOut, ForwardDiff.Dual{TgIn, T, 1}, 1}},
        xi::Complex{T}, idx::Int, yi_idx::Int, zj_idx::Int) where {TgOut, TgIn, T <: Real}
    D_in = ForwardDiff.Dual{TgIn, T, 1}
    D_out = ForwardDiff.Dual{TgOut, D_in, 1}
    re_in = D_in(
        real(xi), ForwardDiff.Partials{1, T}((ifelse(idx == yi_idx, one(T), zero(T)),)))
    im_in = D_in(imag(xi))
    p_out = ForwardDiff.Partials{1, D_in}((ifelse(idx == zj_idx, one(D_in), zero(D_in)),))
    return Complex(D_out(re_in, p_out), D_out(im_in))
end

mutable struct VarProPrep{
    T, R <: Real, V <: AbstractVector{T}, M <: AbstractMatrix{T},
    IV <: AbstractVector{Int}, IP, FAC, XN, RN
}
    const x_buf::V
    const r_buf::V
    const Fz_buf::V
    const J_buf::M
    const y_warm::V
    const y_trial::V
    const dy_buf::V
    const w_vec::V
    const col_m::V
    const Φ_mat::M
    const D_mat::M
    const dΦ_mat::M
    const Sz_jvp::M
    const z_curr::V
    const Y_h::Vector{Int}
    const Z_h::Vector{Int}
    const M_h::Vector{Int}
    const Y_d::IV
    const Z_d::IV
    const M_d::IV
    const idx_n::IV
    const inner_prep::IP
    const fac_phi::FAC
    const x_nest1::XN
    const r_nest1::RN
    fnorm_z_prev::R
    last_inner_iters::Int
    inner_failures::Int32
    init_failures::Int32
end

struct _R3BlockOuterProb{E <: BlockEliminated, WS <: BlockElimPrep} <:
       AbstractNonlinearProblem
    elim::E
    ws::WS
end

@inline _has_analytic_jacobian(::_R3BlockOuterProb) = true

struct _R3VPRes{VP <: VariableProjection, WS <: VarProPrep}
    vp::VP
    ws::WS
end

struct _R3VPJac{VP <: VariableProjection, WS <: VarProPrep}
    vp::VP
    ws::WS
end

@inline function (f::_R3VPRes)(r_out::AbstractVector, z::AbstractVector, p)
    _eliminate_vp_r3!(f.ws, f.vp, z, p)
    r_out === f.ws.Fz_buf || copyto!(r_out, f.ws.Fz_buf)
    return r_out
end

@inline function (j::_R3VPJac)(J_out::AbstractMatrix, z::AbstractVector, p)
    _eliminate_vp_r3!(j.ws, j.vp, z, p)
    _varpro_jacobian_r3!(J_out, j.ws, j.vp, p)
    return J_out
end

@inline _make_r3_outer_prob(elim::BlockEliminated, ws::BlockElimPrep) = _R3BlockOuterProb(elim, ws)
@inline function _make_r3_outer_prob(vp::VariableProjection, ws::VarProPrep)
    return LeastSquaresProblem(_R3VPRes(vp, ws), _R3VPJac(vp, ws), length(ws.M_h))
end

function _build_block_elim_prep(
        backend::AbstractADType, elim::BlockEliminated, x::AbstractVector{T}, p) where {T}
    R = real(T)
    Y_h, Z_h, Ey_h, Ez_h = _resolve_partition_r3(elim, x, p)
    n = length(x)
    ny = length(Y_h)
    nz = length(Z_h)
    x_buf = similar(x, T, n)
    F_buf = similar(x, T, n)
    J_buf = similar(x, T, n, n)
    y_warm = similar(x, T, ny)
    fill!(y_warm, T(NaN))
    y_trial = similar(x, T, ny)
    dy_buf = similar(x, T, ny)
    Fy_buf = similar(x, T, ny)
    Fz_buf = similar(x, T, nz)
    w_col = similar(x, T, ny)
    J_yy = similar(x, T, ny, ny)
    J_yz = similar(x, T, ny, nz)
    J_zy = similar(x, T, nz, ny)
    J_zz = similar(x, T, nz, nz)
    W_mat = similar(x, T, ny, nz)
    Sz_jvp = similar(x, T, nz, nz)
    z_curr = similar(x, T, nz)
    z_prev = similar(x, T, nz)
    Y_d = copyto!(similar(x, Int, ny), Y_h)
    Z_d = copyto!(similar(x, Int, nz), Z_h)
    Ey_d = copyto!(similar(x, Int, ny), Ey_h)
    Ez_d = copyto!(similar(x, Int, nz), Ez_h)
    _gather_vec!(z_curr, x, Z_h, Z_d)
    copyto!(z_prev, z_curr)
    copyto!(x_buf, x)
    inner_prep = prepare_ad(backend, getfield(elim, :prob), F_buf, x_buf, p)
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    fac_yy = init_linear(ls_y, nothing, y_warm)
    return BlockElimPrep{
        T, R, typeof(x_buf), typeof(J_buf), typeof(Y_d),
        typeof(inner_prep), typeof(fac_yy)}(
        x_buf, F_buf, J_buf, y_warm, y_trial, dy_buf, Fy_buf, Fz_buf, w_col,
        J_yy, J_yz, J_zy, J_zz, W_mat, Sz_jvp, z_curr, z_prev,
        Y_h, Z_h, Ey_h, Ez_h, Y_d, Z_d, Ey_d, Ez_d,
        inner_prep, fac_yy, R(NaN), 0, Int32(0), Int32(0))
end

function _build_varpro_prep(
        backend::AbstractADType, vp::VariableProjection, x::AbstractVector{T}, p) where {T}
    R = real(T)
    Y_h, Z_h = _resolve_partition_r3(vp, x, p)
    n = length(x)
    ny = length(Y_h)
    nz = length(Z_h)
    prob_in = getfield(vp, :prob)
    m = output_length(prob_in, n)
    M_h = collect(1:m)
    x_buf = similar(x, T, n)
    r_buf = similar(x, T, m)
    Fz_buf = similar(x, T, m)
    J_buf = similar(x, T, m, n)
    y_warm = similar(x, T, ny)
    fill!(y_warm, T(NaN))
    y_trial = similar(x, T, ny)
    dy_buf = similar(x, T, ny)
    w_vec = similar(x, T, ny)
    col_m = similar(x, T, m)
    Φ_mat = similar(x, T, m, ny)
    D_mat = similar(x, T, m, nz)
    dΦ_mat = similar(x, T, m, ny)
    Sz_jvp = similar(x, T, m, nz)
    z_curr = similar(x, T, nz)
    Y_d = copyto!(similar(x, Int, ny), Y_h)
    Z_d = copyto!(similar(x, Int, nz), Z_h)
    M_d = copyto!(similar(x, Int, m), M_h)
    idx_n = copyto!(similar(x, Int, n), collect(1:n))
    _gather_vec!(z_curr, x, Z_h, Z_d)
    copyto!(x_buf, x)
    inner_prep = prepare_ad(backend, prob_in, r_buf, x_buf, p)
    raw_ls = vp.linear_inner ? getfield(vp, :linsolve) :
             _inner_linsolve(getfield(vp, :inner_method))
    ls_y = Adapt.adapt(ToReal{R}(), raw_ls)
    fac_phi = init_linear(ls_y, nothing, y_warm, m)
    if vp.jacobian_mode === :golub_pereyra
        TgIn = VarProInnerTag{typeof(prob_in.F), R}
        TgOut = VarProOuterTag{typeof(prob_in.F), R}
        DN1 = _nested_dual_type(T, TgIn, TgOut, 1, 1)
        x_nest1 = similar(x, DN1, n)
        r_nest1 = similar(x, DN1, m)
    else
        x_nest1 = nothing
        r_nest1 = nothing
    end
    return VarProPrep{
        T, R, typeof(x_buf), typeof(J_buf), typeof(Y_d),
        typeof(inner_prep), typeof(fac_phi), typeof(x_nest1), typeof(r_nest1)}(
        x_buf, r_buf, Fz_buf, J_buf, y_warm, y_trial, dy_buf, w_vec, col_m,
        Φ_mat, D_mat, dΦ_mat, Sz_jvp, z_curr,
        Y_h, Z_h, M_h, Y_d, Z_d, M_d, idx_n,
        inner_prep, fac_phi, x_nest1, r_nest1,
        R(NaN), 0, Int32(0), Int32(0))
end

@inline function _build_block_elim_prep(
        ::AbstractADType, elim::BlockEliminated, x::AbstractArray, p)
    return _validate_elim_x(elim, x, p)
end

@inline function _build_varpro_prep(
        ::AbstractADType, vp::VariableProjection, x::AbstractArray, p)
    return _validate_elim_x(vp, x, p)
end

@inline function prepare_ad(
        backend::AutoForwardDiff, elim::BlockEliminated,
        ::AbstractArray, x::AbstractArray, p)
    return _build_block_elim_prep(backend, elim, x, p)
end

@inline function prepare_ad(
        backend::AbstractADType, elim::BlockEliminated,
        ::AbstractArray, x::AbstractArray, p)
    return _build_block_elim_prep(backend, elim, x, p)
end

@inline function prepare_ad(
        ::Nothing, elim::BlockEliminated,
        ::AbstractArray, x::AbstractArray, p)
    return _build_block_elim_prep(AutoForwardDiff(), elim, x, p)
end

@inline function prepare_ad(
        backend::AutoForwardDiff, vp::VariableProjection,
        ::AbstractArray, x::AbstractArray, p)
    return _build_varpro_prep(backend, vp, x, p)
end

@inline function prepare_ad(
        backend::AbstractADType, vp::VariableProjection,
        ::AbstractArray, x::AbstractArray, p)
    return _build_varpro_prep(backend, vp, x, p)
end

@inline function prepare_ad(
        ::Nothing, vp::VariableProjection,
        ::AbstractArray, x::AbstractArray, p)
    return _build_varpro_prep(AutoForwardDiff(), vp, x, p)
end

# --- R3 Inner Elimination & Schur / VarPro Jacobians ---

function _eliminate_r3!(
        ws::BlockElimPrep{T, R}, elim::BlockEliminated{P, YI, ZI, EY, EZ, M, true, IO},
        z::AbstractVector{T}, p, ::Bool = false) where {T, R, P, YI, ZI, EY, EZ, M, IO}
    ny = length(ws.Y_h)
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    prob_in = getfield(elim, :prob)
    fill!(ws.y_trial, zero(T))
    _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
    _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
    _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
    _extract_submat!(ws.J_yy, ws.J_buf, ws.Ey_h, ws.Y_h, ws.Ey_d, ws.Y_d)
    fac_yy = factorize!!(ws.fac_yy, ls_y, ws.J_yy)
    ldiv!!(ws.dy_buf, fac_yy, ws.Fy_buf)
    ws.y_warm .= .-ws.dy_buf
    _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
    _gather_vec!(ws.Fz_buf, ws.F_buf, ws.Ez_h, ws.Ez_d)
    if returncode(fac_yy) != SUCCESS || _is_rank_deficient(fac_yy, ny) ||
       !all(isfinite, ws.y_warm)
        ws.inner_failures += one(Int32)
    end
    ws.last_inner_iters = 1
    return ws.Fz_buf
end

function _schur_res_corr_r3!(
        ws::BlockElimPrep{T, R}, ls_y, prob_in, z::AbstractVector{T},
        p, fnorm_y::R, τ_floor::R) where {T, R}
    (fnorm_y <= τ_floor || !isfinite(fnorm_y)) && return ws.Fz_buf
    ny = length(ws.Y_h)
    _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
    _extract_submat!(ws.J_yy, ws.J_buf, ws.Ey_h, ws.Y_h, ws.Ey_d, ws.Y_d)
    fac_yy = factorize!!(ws.fac_yy, ls_y, ws.J_yy)
    if returncode(fac_yy) != SUCCESS || _is_rank_deficient(fac_yy, ny)
        ws.inner_failures += one(Int32)
        mul!(ws.w_col, ws.J_yy, ws.y_warm)
        ldiv!!(ws.y_warm, fac_yy, ws.w_col)
    end
    ldiv!!(ws.w_col, fac_yy, ws.Fy_buf)
    if all(isfinite, ws.w_col)
        ws.y_warm .-= ws.w_col
        _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
        _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
        _gather_vec!(ws.Fz_buf, ws.F_buf, ws.Ez_h, ws.Ez_d)
    else
        ws.inner_failures += one(Int32)
    end
    return ws.Fz_buf
end

function _eliminate_r3!(
        ws::BlockElimPrep{T, R}, elim::BlockEliminated{P, YI, ZI, EY, EZ, M, false, IO},
        z::AbstractVector{T}, p,
        has_schur_pred::Bool = false) where {T, R, P, YI, ZI, EY, EZ, M, IO}
    ny = length(ws.Y_h)
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    prob_in = getfield(elim, :prob)
    if !all(isfinite, ws.y_warm)
        fill!(ws.y_warm, zero(T))
    end
    _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
    _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
    _gather_vec!(ws.Fz_buf, ws.F_buf, ws.Ez_h, ws.Ez_d)
    fnorm_y0 = R(rnorm(ws.Fy_buf))
    τ_floor = R(10) * eps(R) * max(one(R), fnorm_y0)
    τ_in, max_it = _inner_tol_r1(
        getfield(elim, :inner_options), R, fnorm_y0, ws.fnorm_z_prev)
    if fnorm_y0 <= τ_in
        ws.last_inner_iters = 0
        if !isfinite(fnorm_y0)
            ws.inner_failures += one(Int32)
        elseif !isfinite(ws.fnorm_z_prev)
            _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
            _extract_submat!(ws.J_yy, ws.J_buf, ws.Ey_h, ws.Y_h, ws.Ey_d, ws.Y_d)
            fac_yy0 = factorize!!(ws.fac_yy, ls_y, ws.J_yy)
            if returncode(fac_yy0) != SUCCESS || _is_rank_deficient(fac_yy0, ny)
                ws.inner_failures += one(Int32)
                mul!(ws.w_col, ws.J_yy, ws.y_warm)
                ldiv!!(ws.y_warm, fac_yy0, ws.w_col)
                _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
                _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
                _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
                _gather_vec!(ws.Fz_buf, ws.F_buf, ws.Ez_h, ws.Ez_d)
            end
        end
        _schur_res_corr_r3!(ws, ls_y, prob_in, z, p, fnorm_y0, τ_floor)
        return ws.Fz_buf
    end
    if has_schur_pred && max_it >= 1 && z != ws.z_prev
        ws.Fz_buf .= z .- ws.z_prev
        mul!(ws.dy_buf, ws.W_mat, ws.Fz_buf)
        ws.dy_buf .+= ws.w_col
        ws.y_trial .= ws.y_warm .- ws.dy_buf
        if all(isfinite, ws.y_trial)
            _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
            _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
            _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
            fnorm_p = R(rnorm(ws.Fy_buf))
            if fnorm_p <= τ_in
                copyto!(ws.y_warm, ws.y_trial)
                _gather_vec!(ws.Fz_buf, ws.F_buf, ws.Ez_h, ws.Ez_d)
                _schur_res_corr_r3!(ws, ls_y, prob_in, z, p, fnorm_p, τ_floor)
                ws.last_inner_iters = 1
                return ws.Fz_buf
            end
            _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
            _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
            _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
        end
    end
    iters = 0
    fnorm_y = fnorm_y0
    failed = false
    for _ in 1:max_it
        iters += 1
        _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
        _extract_submat!(ws.J_yy, ws.J_buf, ws.Ey_h, ws.Y_h, ws.Ey_d, ws.Y_d)
        fac_yy = factorize!!(ws.fac_yy, ls_y, ws.J_yy)
        if returncode(fac_yy) != SUCCESS || _is_rank_deficient(fac_yy, ny)
            failed = true
            mul!(ws.w_col, ws.J_yy, ws.y_warm)
            ldiv!!(ws.y_warm, fac_yy, ws.w_col)
        end
        ldiv!!(ws.dy_buf, fac_yy, ws.Fy_buf)
        if !all(isfinite, ws.dy_buf)
            failed = true
            break
        end
        α = one(R)
        ws.y_trial .= ws.y_warm .- ws.dy_buf
        _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
        _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
        _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
        fnorm_next = R(rnorm(ws.Fy_buf))
        for _ in 1:8
            (isfinite(fnorm_next) && fnorm_next <= fnorm_y) && break
            α *= R(0.5)
            ws.y_trial .= ws.y_warm .- T(α) .* ws.dy_buf
            _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
            _inner_res_r3!(ws.F_buf, ws.inner_prep, prob_in, ws.x_buf, p)
            _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
            fnorm_next = R(rnorm(ws.Fy_buf))
        end
        copyto!(ws.y_warm, ws.y_trial)
        fnorm_y = fnorm_next
        if fnorm_y <= τ_in ||
           R(rnorm(ws.dy_buf)) <= R(2) * eps(R) * max(one(R), R(rnorm(ws.y_warm)))
            break
        end
    end
    _gather_vec!(ws.Fz_buf, ws.F_buf, ws.Ez_h, ws.Ez_d)
    if failed || !(fnorm_y <= τ_in)
        ws.inner_failures += one(Int32)
    end
    _schur_res_corr_r3!(ws, ls_y, prob_in, z, p, fnorm_y, τ_floor)
    ws.last_inner_iters = iters
    return ws.Fz_buf
end

function _schur_jacobian_r3!(
        Sz::AbstractMatrix{T}, ws::BlockElimPrep{T, R},
        elim::BlockEliminated, p) where {T, R}
    ny = length(ws.Y_h)
    nz = length(ws.Z_h)
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(elim, :inner_method)))
    _inner_jac_r3!(ws.J_buf, ws.inner_prep, getfield(elim, :prob), ws.x_buf, p)
    _extract_submat!(ws.J_yy, ws.J_buf, ws.Ey_h, ws.Y_h, ws.Ey_d, ws.Y_d)
    _extract_submat!(ws.J_yz, ws.J_buf, ws.Ey_h, ws.Z_h, ws.Ey_d, ws.Z_d)
    _extract_submat!(ws.J_zy, ws.J_buf, ws.Ez_h, ws.Y_h, ws.Ez_d, ws.Y_d)
    _extract_submat!(ws.J_zz, ws.J_buf, ws.Ez_h, ws.Z_h, ws.Ez_d, ws.Z_d)
    fac_yy = factorize!!(ws.fac_yy, ls_y, ws.J_yy)
    if returncode(fac_yy) != SUCCESS || _is_rank_deficient(fac_yy, ny)
        ws.inner_failures += one(Int32)
        mul!(ws.Fy_buf, ws.J_yy, ws.y_warm)
        ldiv!!(ws.y_warm, fac_yy, ws.Fy_buf)
    end
    for j in 1:nz
        _extract_col!(ws.Fy_buf, ws.J_yz, j)
        ldiv!!(ws.w_col, fac_yy, ws.Fy_buf)
        _store_col!(ws.W_mat, ws.w_col, j)
    end
    _gather_vec!(ws.Fy_buf, ws.F_buf, ws.Ey_h, ws.Ey_d)
    ldiv!!(ws.w_col, fac_yy, ws.Fy_buf)
    Sz === ws.J_zz || copyto!(Sz, ws.J_zz)
    mul!(Sz, ws.J_zy, ws.W_mat, -one(T), one(T))
    return Sz
end

function _eliminate_vp_r3!(
        ws::VarProPrep{T, R}, vp::VariableProjection{P, YI, ZI, LS, JM, true, M, IO},
        z::AbstractVector{T}, p) where {T, R, P, YI, ZI, LS, JM, M, IO}
    ny = length(ws.Y_h)
    ls_y = Adapt.adapt(ToReal{R}(), getfield(vp, :linsolve))
    prob_in = getfield(vp, :prob)
    fill!(ws.y_trial, zero(T))
    _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    prob_in.F(ws.r_buf, ws.x_buf, p)
    _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
    _extract_submat!(ws.Φ_mat, ws.J_buf, ws.M_h, ws.Y_h, ws.M_d, ws.Y_d)
    fac_phi = factorize!!(ws.fac_phi, ls_y, ws.Φ_mat)
    ldiv!!(ws.dy_buf, fac_phi, ws.r_buf)
    ws.y_warm .= .-ws.dy_buf
    _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    prob_in.F(ws.Fz_buf, ws.x_buf, p)
    if returncode(fac_phi) != SUCCESS || _is_rank_deficient(fac_phi, ny) ||
       !all(isfinite, ws.y_warm)
        ws.inner_failures += one(Int32)
    end
    ws.last_inner_iters = 1
    return ws.Fz_buf
end

function _eliminate_vp_r3!(
        ws::VarProPrep{T, R}, vp::VariableProjection{P, YI, ZI, LS, JM, false, M, IO},
        z::AbstractVector{T}, p) where {T, R, P, YI, ZI, LS, JM, M, IO}
    ny = length(ws.Y_h)
    ls_y = Adapt.adapt(ToReal{R}(), _inner_linsolve(getfield(vp, :inner_method)))
    prob_in = getfield(vp, :prob)
    if !all(isfinite, ws.y_warm)
        fill!(ws.y_warm, zero(T))
    end
    _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    prob_in.F(ws.Fz_buf, ws.x_buf, p)
    fnorm_r0 = R(rnorm(ws.Fz_buf))
    inner_opt = getfield(vp, :inner_options)
    if inner_opt === nothing
        τ_in = R(10) * eps(R) * max(one(R), fnorm_r0)
        max_it = Int32(50)
    else
        τ_in, max_it = _inner_tol_r1(inner_opt, R, fnorm_r0, ws.fnorm_z_prev)
    end
    iters = 0
    failed = false
    converged = isfinite(fnorm_r0) && fnorm_r0 <= τ_in
    for k in 1:(max_it + 1)
        _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
        _extract_submat!(ws.Φ_mat, ws.J_buf, ws.M_h, ws.Y_h, ws.M_d, ws.Y_d)
        mul!(ws.w_vec, adjoint(ws.Φ_mat), ws.Fz_buf)
        if R(rnorm(ws.w_vec)) <= τ_in * max(one(R), R(rnorm(ws.Φ_mat)))
            if k == 1 && !isfinite(ws.fnorm_z_prev)
                fac_phi0 = factorize!!(ws.fac_phi, ls_y, ws.Φ_mat)
                if returncode(fac_phi0) != SUCCESS || _is_rank_deficient(fac_phi0, ny)
                    failed = true
                    mul!(ws.col_m, ws.Φ_mat, ws.y_warm)
                    ldiv!!(ws.y_warm, fac_phi0, ws.col_m)
                    _scatter_yz!(ws.x_buf, ws.y_warm, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
                    prob_in.F(ws.Fz_buf, ws.x_buf, p)
                    fnorm_r0 = R(rnorm(ws.Fz_buf))
                end
            end
            converged = isfinite(fnorm_r0)
            break
        end
        k > max_it && break
        iters += 1
        fac_phi = factorize!!(ws.fac_phi, ls_y, ws.Φ_mat)
        if returncode(fac_phi) != SUCCESS || _is_rank_deficient(fac_phi, ny)
            failed = true
            mul!(ws.col_m, ws.Φ_mat, ws.y_warm)
            ldiv!!(ws.y_warm, fac_phi, ws.col_m)
        end
        ldiv!!(ws.dy_buf, fac_phi, ws.Fz_buf)
        if !all(isfinite, ws.dy_buf)
            failed = true
            break
        end
        α = one(R)
        ws.y_trial .= ws.y_warm .- ws.dy_buf
        _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
        prob_in.F(ws.r_buf, ws.x_buf, p)
        fnorm_next = R(rnorm(ws.r_buf))
        for _ in 1:8
            (isfinite(fnorm_next) && fnorm_next <= fnorm_r0) && break
            α *= R(0.5)
            ws.y_trial .= ws.y_warm .- T(α) .* ws.dy_buf
            _scatter_yz!(ws.x_buf, ws.y_trial, z, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
            prob_in.F(ws.r_buf, ws.x_buf, p)
            fnorm_next = R(rnorm(ws.r_buf))
        end
        copyto!(ws.y_warm, ws.y_trial)
        copyto!(ws.Fz_buf, ws.r_buf)
        fnorm_r0 = fnorm_next
        if isfinite(fnorm_r0) && fnorm_r0 <= τ_in
            converged = true
            break
        end
        if R(rnorm(ws.dy_buf)) <= R(2) * eps(R) * max(one(R), R(rnorm(ws.y_warm)))
            _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, ws.x_buf, p)
            _extract_submat!(ws.Φ_mat, ws.J_buf, ws.M_h, ws.Y_h, ws.M_d, ws.Y_d)
            mul!(ws.w_vec, adjoint(ws.Φ_mat), ws.Fz_buf)
            if R(rnorm(ws.w_vec)) <= τ_in * max(one(R), R(rnorm(ws.Φ_mat)))
                converged = isfinite(fnorm_r0)
            end
            break
        end
    end
    if failed || !converged || !all(isfinite, ws.y_warm)
        ws.inner_failures += one(Int32)
    end
    ws.last_inner_iters = iters
    return ws.Fz_buf
end

function _varpro_kaufman_r3!(
        J_red::AbstractMatrix{T}, ws::VarProPrep{T, R},
        vp::VariableProjection, p) where {T, R}
    ny = length(ws.Y_h)
    nz = length(ws.Z_h)
    raw_ls = vp.linear_inner ? getfield(vp, :linsolve) :
             _inner_linsolve(getfield(vp, :inner_method))
    ls_y = Adapt.adapt(ToReal{R}(), raw_ls)
    _inner_jac_r3!(ws.J_buf, ws.inner_prep, getfield(vp, :prob), ws.x_buf, p)
    _extract_submat!(ws.Φ_mat, ws.J_buf, ws.M_h, ws.Y_h, ws.M_d, ws.Y_d)
    _extract_submat!(ws.D_mat, ws.J_buf, ws.M_h, ws.Z_h, ws.M_d, ws.Z_d)
    fac_phi = factorize!!(ws.fac_phi, ls_y, ws.Φ_mat)
    if returncode(fac_phi) != SUCCESS || _is_rank_deficient(fac_phi, ny)
        ws.inner_failures += one(Int32)
    end
    for j in 1:nz
        _extract_col!(ws.col_m, ws.D_mat, j)
        ldiv!!(ws.w_vec, fac_phi, ws.col_m)
        mul!(ws.col_m, ws.Φ_mat, ws.w_vec)
        _sub_col!(J_red, ws.D_mat, ws.col_m, j)
    end
    return J_red, fac_phi
end

function _varpro_jacobian_r3!(
        J_red::AbstractMatrix{T}, ws::VarProPrep{T, R},
        vp::VariableProjection{P, YI, ZI, LS, :kaufman, LI, M, IO},
        p) where {T, R, P, YI, ZI, LS, LI, M, IO}
    return _varpro_kaufman_r3!(J_red, ws, vp, p)[1]
end

function _varpro_jacobian_r3!(
        J_red::AbstractMatrix{T}, ws::VarProPrep{T, R},
        vp::VariableProjection{P, YI, ZI, LS, :golub_pereyra, LI, M, IO},
        p) where {T, R, P, YI, ZI, LS, LI, M, IO}
    _, fac_phi = _varpro_kaufman_r3!(J_red, ws, vp, p)
    ny = length(ws.Y_h)
    nz = length(ws.Z_h)
    prob_in = getfield(vp, :prob)
    x_nest1 = ws.x_nest1
    r_nest1 = ws.r_nest1
    for j in 1:nz
        zj_idx = ws.Z_h[j]
        for i in 1:ny
            yi_idx = ws.Y_h[i]
            x_nest1 .= _seed_vp_nest1.(x_nest1, ws.x_buf, ws.idx_n, yi_idx, zj_idx)
            prob_in.F(r_nest1, x_nest1, p)
            ws.col_m .= _nested_hess1.(r_nest1, 1)
            _store_col!(ws.dΦ_mat, ws.col_m, i)
        end
        mul!(ws.w_vec, adjoint(ws.dΦ_mat), ws.Fz_buf)
        _pinv_adj_mul_r3!(ws.col_m, fac_phi, ws.w_vec)
        _dec_col!(J_red, ws.col_m, j)
    end
    return J_red
end

# --- R3 `residual!!`, `jacobian!!`, `jvp!!` on `BlockEliminated` and `VariableProjection` ---

struct _ElimResBuffer{WS <: Union{BlockElimPrep, VarProPrep}}
    ws::WS
end

@inline function residual!!(
        Fz::AbstractVector, ws::BlockElimPrep,
        elim::BlockEliminated, x::AbstractVector, p)
    if !isfinite(ws.fnorm_z_prev)
        _gather_vec!(ws.y_warm, x, ws.Y_h, ws.Y_d)
    end
    _gather_vec!(ws.z_curr, x, ws.Z_h, ws.Z_d)
    _eliminate_r3!(ws, elim, ws.z_curr, p)
    Fz === ws.Fz_buf || copyto!(Fz, ws.Fz_buf)
    return Fz
end

@inline function residual!!(
        r_out::AbstractVector, ws::VarProPrep,
        vp::VariableProjection, x::AbstractVector, p)
    if !isfinite(ws.fnorm_z_prev)
        _gather_vec!(ws.y_warm, x, ws.Y_h, ws.Y_d)
    end
    _gather_vec!(ws.z_curr, x, ws.Z_h, ws.Z_d)
    _eliminate_vp_r3!(ws, vp, ws.z_curr, p)
    r_out === ws.Fz_buf || copyto!(r_out, ws.Fz_buf)
    return r_out
end

@inline function residual!!(
        rb::_ElimResBuffer{<:BlockElimPrep}, elim::BlockEliminated,
        x::AbstractVector, p)
    return residual!!(rb.ws.Fz_buf, rb.ws, elim, x, p)
end

@inline function residual!!(
        rb::_ElimResBuffer{<:VarProPrep}, vp::VariableProjection,
        x::AbstractVector, p)
    return residual!!(rb.ws.Fz_buf, rb.ws, vp, x, p)
end

@inline function residual!!(
        Fz::AbstractVector, elim::BlockEliminated,
        x::AbstractVector, p)
    ws = prepare_ad(AutoForwardDiff(), elim, Fz, x, p)
    return residual!!(Fz, ws, elim, x, p)
end

@inline function residual!!(
        Fz::AbstractVector, vp::VariableProjection,
        x::AbstractVector, p)
    ws = prepare_ad(AutoForwardDiff(), vp, Fz, x, p)
    return residual!!(Fz, ws, vp, x, p)
end

@inline function jacobian!!(
        Sz::AbstractMatrix, ws::BlockElimPrep,
        elim::BlockEliminated, x::AbstractVector, p)
    if !isfinite(ws.fnorm_z_prev)
        _gather_vec!(ws.y_warm, x, ws.Y_h, ws.Y_d)
    end
    _gather_vec!(ws.z_curr, x, ws.Z_h, ws.Z_d)
    _eliminate_r3!(ws, elim, ws.z_curr, p)
    return _schur_jacobian_r3!(Sz, ws, elim, p)
end

@inline function jacobian!!(
        J_red::AbstractMatrix, ws::VarProPrep,
        vp::VariableProjection, x::AbstractVector, p)
    if !isfinite(ws.fnorm_z_prev)
        _gather_vec!(ws.y_warm, x, ws.Y_h, ws.Y_d)
    end
    _gather_vec!(ws.z_curr, x, ws.Z_h, ws.Z_d)
    _eliminate_vp_r3!(ws, vp, ws.z_curr, p)
    return _varpro_jacobian_r3!(J_red, ws, vp, p)
end

@inline function jvp!!(
        Jv::AbstractVector, ws::BlockElimPrep,
        elim::BlockEliminated,
        x::AbstractVector, v::AbstractVector, p)
    jacobian!!(ws.J_zz, ws, elim, x, p)
    if length(v) == length(x)
        _gather_vec!(ws.z_curr, v, ws.Z_h, ws.Z_d)
        mul!(Jv, ws.J_zz, ws.z_curr)
    else
        mul!(Jv, ws.J_zz, v)
    end
    return Jv
end

@inline function jvp!!(
        Jv::AbstractVector, ws::VarProPrep,
        vp::VariableProjection,
        x::AbstractVector, v::AbstractVector, p)
    jacobian!!(ws.Sz_jvp, ws, vp, x, p)
    if length(v) == length(x)
        _gather_vec!(ws.z_curr, v, ws.Z_h, ws.Z_d)
        mul!(Jv, ws.Sz_jvp, ws.z_curr)
    else
        mul!(Jv, ws.Sz_jvp, v)
    end
    return Jv
end

# R3 outer problem methods for `_R3BlockOuterProb`
@inline function prepare_ad(
        ::AutoForwardDiff, oprob::_R3BlockOuterProb, ::AbstractArray, ::AbstractArray, _)
    return oprob.ws
end
@inline function prepare_ad(
        ::AbstractADType, oprob::_R3BlockOuterProb, ::AbstractArray, ::AbstractArray, _)
    return oprob.ws
end
@inline function prepare_ad(
        ::Nothing, oprob::_R3BlockOuterProb, ::AbstractArray, ::AbstractArray, _)
    return oprob.ws
end

@inline function residual!!(
        Fz::AbstractVector, oprob::_R3BlockOuterProb, z::AbstractVector, p)
    has_pred = !oprob.elim.linear_inner && isfinite(oprob.ws.fnorm_z_prev)
    _eliminate_r3!(oprob.ws, oprob.elim, z, p, has_pred)
    Fz === oprob.ws.Fz_buf || copyto!(Fz, oprob.ws.Fz_buf)
    return Fz
end

@inline function jacobian!!(
        Sz::AbstractMatrix, ::BlockElimPrep,
        oprob::_R3BlockOuterProb, z::AbstractVector, p)
    _eliminate_r3!(oprob.ws, oprob.elim, z, p)
    return _schur_jacobian_r3!(Sz, oprob.ws, oprob.elim, p)
end

@inline function jacobian!!(
        Sz::AbstractMatrix, ::Nothing,
        oprob::_R3BlockOuterProb, z::AbstractVector, p)
    _eliminate_r3!(oprob.ws, oprob.elim, z, p)
    return _schur_jacobian_r3!(Sz, oprob.ws, oprob.elim, p)
end

@inline function jvp!!(
        Jv::AbstractVector, ::Union{BlockElimPrep, Nothing}, oprob::_R3BlockOuterProb,
        z::AbstractVector, v::AbstractVector, p)
    jacobian!!(oprob.ws.J_zz, oprob.ws, oprob, z, p)
    mul!(Jv, oprob.ws.J_zz, v)
    return Jv
end

# --- R1 Outer Problem Wrappers for `StaticElimState` ---

struct _R1OuterBlockProb{E <: BlockEliminated, YW, R, YV, ZV, EYV, EZV} <:
       AbstractNonlinearProblem
    elim::E
    y_warm::YW
    fnorm_z_prev::R
    Y::YV
    Z::ZV
    E_y::EYV
    E_z::EZV
end

@inline _has_analytic_jacobian(::_R1OuterBlockProb) = true

@inline function prepare_ad(
        ::Nothing, ::_R1OuterBlockProb, ::Nothing, ::StaticVector, _)
    return nothing
end
@inline function prepare_ad(
        ::AutoForwardDiff, ::_R1OuterBlockProb, ::Nothing, ::StaticVector, _)
    return nothing
end
@inline function prepare_ad(
        ::AbstractADType, ::_R1OuterBlockProb, ::Nothing, ::StaticVector, _)
    return nothing
end

@inline function residual!!(
        ::Nothing, oprob::_R1OuterBlockProb, z::StaticVector, p)
    _, Fz, _, _, _ = _eliminate_r1(
        oprob.elim, nothing, oprob.y_warm, z, p, oprob.fnorm_z_prev,
        oprob.Y, oprob.Z, oprob.E_y, oprob.E_z)
    return Fz
end

@inline function _outer_block_jac_r1(oprob::_R1OuterBlockProb, z::StaticVector, p)
    _, _, x_star, _, _ = _eliminate_r1(
        oprob.elim, nothing, oprob.y_warm, z, p, oprob.fnorm_z_prev,
        oprob.Y, oprob.Z, oprob.E_y, oprob.E_z)
    S, _ = _schur_jacobian_r1(
        oprob.elim, x_star, p, oprob.Y, oprob.Z, oprob.E_y, oprob.E_z)
    return S
end

@inline function jacobian!!(
        ::Nothing, ::AutoForwardDiff,
        oprob::_R1OuterBlockProb, z::StaticVector, p)
    return _outer_block_jac_r1(oprob, z, p)
end

@inline function jacobian!!(
        ::Nothing, ::Nothing,
        oprob::_R1OuterBlockProb, z::StaticVector, p)
    return _outer_block_jac_r1(oprob, z, p)
end

@inline function jvp!!(
        ::Nothing, ::AutoForwardDiff,
        oprob::_R1OuterBlockProb, z::StaticVector{Nz, T},
        v::StaticVector{Nz, T}, p) where {Nz, T}
    return _outer_block_jac_r1(oprob, z, p) * v
end

@inline function jvp!!(
        ::Nothing, ::Nothing,
        oprob::_R1OuterBlockProb, z::StaticVector{Nz, T},
        v::StaticVector{Nz, T}, p) where {Nz, T}
    return _outer_block_jac_r1(oprob, z, p) * v
end

struct _R1OuterVPRes{VP <: VariableProjection, YW, R, YV, ZV}
    vp::VP
    y_warm::YW
    fnorm_z_prev::R
    Y::YV
    Z::ZV
end

struct _R1OuterVPJac{VP <: VariableProjection, YW, R, YV, ZV}
    vp::VP
    y_warm::YW
    fnorm_z_prev::R
    Y::YV
    Z::ZV
end

@inline function (f::_R1OuterVPRes)(z::StaticVector, p)
    _, r_star, _, _, _ = _eliminate_vp_r1(
        f.vp, f.y_warm, z, p, f.fnorm_z_prev, f.Y, f.Z)
    return r_star
end

@inline function (j::_R1OuterVPJac)(z::StaticVector, p)
    _, r_star, x_star, _, _ = _eliminate_vp_r1(
        j.vp, j.y_warm, z, p, j.fnorm_z_prev, j.Y, j.Z)
    J_red, _ = _varpro_jacobian_r1(j.vp, x_star, r_star, p, j.Y, j.Z)
    return J_red
end

@inline function _make_r1_outer_prob(
        elim::BlockEliminated, y_warm, fnorm_z_prev, Y, Z, E_y, E_z)
    return _R1OuterBlockProb(elim, y_warm, fnorm_z_prev, Y, Z, E_y, E_z)
end

@inline function _make_r1_outer_prob(
        vp::VariableProjection, y_warm, fnorm_z_prev, Y, Z, _E_y, _E_z)
    return LeastSquaresProblem(
        _R1OuterVPRes(vp, y_warm, fnorm_z_prev, Y, Z),
        _R1OuterVPJac(vp, y_warm, fnorm_z_prev, Y, Z),
        getfield(vp, :prob).output_length)
end

# --- Solver States (`StaticElimState` and `DenseElimState`) ---

struct StaticElimState{OST, IST, YW, ZC, R, YV, ZV, EYV, EZV}
    outer_state::OST
    inner_state::IST
    y_warm::YW
    z_curr::ZC
    fnorm_z_prev::R
    Y::YV
    Z::ZV
    E_y::EYV
    E_z::EZV
    init_failures::Int32
end

struct DenseElimState{OST, WS <: Union{BlockElimPrep, VarProPrep}, OP}
    outer_state::OST
    ws::WS
    oprob::OP
end

@inline residual_buffer(::StaticElimState) = nothing
@inline residual_buffer(st::DenseElimState) = _ElimResBuffer(st.ws)

@inline state_gradratio(st::Union{
    StaticElimState, DenseElimState}) = state_gradratio(st.outer_state)
@inline state_jnorm(st::Union{
    StaticElimState, DenseElimState}) = state_jnorm(st.outer_state)

@inline function stationary_converged(
        ::VariableProjection, options::Options{R}, state,
        status::SolverStatus{R}) where {R}
    status.iterations >= options.min_iterations || return false
    jnorm = state_jnorm(state)
    grad = state_gradratio(state)
    return jnorm > zero(R) && status.fnorm > zero(R) && grad <= options.g_tol
end

@inline function refresh_diagnostics!!(
        st::StaticElimState, m, vp::VariableProjection, x, Fx, p)
    R = real(eltype(x))
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(
        vp, st.y_warm, st.fnorm_z_prev, st.Y, st.Z, st.E_y, st.E_z)
    outer_state = refresh_diagnostics!!(st.outer_state, m_r, oprob, st.z_curr, Fx, p)
    return StaticElimState(
        outer_state, st.inner_state, st.y_warm, st.z_curr, st.fnorm_z_prev,
        st.Y, st.Z, st.E_y, st.E_z, st.init_failures)
end

@inline function _refresh_elim_r3!!(
        outer_state, ::BlockElimPrep, oprob, m, ::BlockEliminated, x, Fx, p)
    return outer_state
end

@inline function _refresh_elim_r3!!(
        outer_state, ws::VarProPrep, oprob, m, ::VariableProjection, x, Fx, p)
    R = real(eltype(x))
    m_r = Adapt.adapt(ToReal{R}(), m)
    return refresh_diagnostics!!(outer_state, m_r, oprob, ws.z_curr, Fx, p)
end

@inline function refresh_diagnostics!!(
        st::DenseElimState, m, vp::VariableProjection, x, Fx, p)
    outer_state = _refresh_elim_r3!!(st.outer_state, st.ws, st.oprob, m, vp, x, Fx, p)
    return DenseElimState(outer_state, st.ws, st.oprob)
end

@inline function _init_elim_r1(
        m, elim::BlockEliminated, x₀::StaticVector{N, T}, p) where {N, T}
    R = real(T)
    Y, Z, E_y, E_z = _resolve_partition_r1(elim, x₀, p)
    Ny = length(Y)
    y_nan = SVector{Ny, T}(ntuple(_ -> T(NaN), Val(Ny)))
    z₀ = x₀[Z]
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(elim, y_nan, R(NaN), Y, Z, E_y, E_z)
    outer_state = init_state(m_r, oprob, z₀, p)
    return StaticElimState(
        outer_state, nothing, y_nan, z₀, R(NaN), Y, Z, E_y, E_z, Int32(0))
end

@inline function _init_elim_r1(
        m, vp::VariableProjection, x₀::StaticVector{N, T}, p) where {N, T}
    R = real(T)
    Y, Z = _resolve_partition_r1(vp, x₀, p)
    Ny = length(Y)
    y_nan = SVector{Ny, T}(ntuple(_ -> T(NaN), Val(Ny)))
    z₀ = x₀[Z]
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(vp, y_nan, R(NaN), Y, Z, Y, Z)
    outer_state = init_state(m_r, oprob, z₀, p)
    return StaticElimState(
        outer_state, nothing, y_nan, z₀, R(NaN), Y, Z, Y, Z, Int32(0))
end

function _init_elim_r3(
        m, elim::BlockEliminated, x₀::AbstractVector{T}, p) where {T}
    R = real(T)
    m_r = Adapt.adapt(ToReal{R}(), m)
    ws = _build_block_elim_prep(_ad_backend_of(m_r), elim, x₀, p)
    oprob = _make_r3_outer_prob(elim, ws)
    outer_state = init_state(m_r, oprob, ws.z_curr, p)
    fill!(ws.y_warm, T(NaN))
    ws.fnorm_z_prev = R(NaN)
    ws.last_inner_iters = 0
    ws.inner_failures = Int32(0)
    ws.init_failures = Int32(0)
    return DenseElimState(outer_state, ws, oprob)
end

function _init_elim_r3(
        m, vp::VariableProjection, x₀::AbstractVector{T}, p) where {T}
    R = real(T)
    m_r = Adapt.adapt(ToReal{R}(), m)
    ws = _build_varpro_prep(_ad_backend_of(m_r), vp, x₀, p)
    oprob = _make_r3_outer_prob(vp, ws)
    outer_state = init_state(m_r, oprob, ws.z_curr, p)
    fill!(ws.y_warm, T(NaN))
    ws.fnorm_z_prev = R(NaN)
    ws.last_inner_iters = 0
    ws.inner_failures = Int32(0)
    ws.init_failures = Int32(0)
    return DenseElimState(outer_state, ws, oprob)
end

@inline function init_state(
        m::Newton, prob::BlockEliminated, x₀::StaticVector, p)
    return _init_elim_r1(m, prob, x₀, p)
end

@inline function init_state(
        m::Newton, prob::VariableProjection, x₀::StaticVector, p)
    return _init_elim_r1(m, prob, x₀, p)
end

@inline function init_state(
        m, prob::BlockEliminated, x₀::StaticVector, p)
    return _init_elim_r1(m, prob, x₀, p)
end

@inline function init_state(
        m, prob::VariableProjection, x₀::StaticVector, p)
    return _init_elim_r1(m, prob, x₀, p)
end

@inline function init_state(
        ::Newton, prob::Union{BlockEliminated, VariableProjection}, x₀::Number, p)
    return _validate_elim_x(prob, x₀, p)
end

@inline function init_state(
        _, prob::Union{BlockEliminated, VariableProjection}, x₀::Number, p)
    return _validate_elim_x(prob, x₀, p)
end

function init_state(
        m::Newton, prob::BlockEliminated, x₀::AbstractVector, p)
    return _init_elim_r3(m, prob, x₀, p)
end

function init_state(
        m::Newton, prob::VariableProjection, x₀::AbstractVector, p)
    return _init_elim_r3(m, prob, x₀, p)
end

function init_state(
        m, prob::BlockEliminated, x₀::AbstractVector, p)
    return _init_elim_r3(m, prob, x₀, p)
end

function init_state(
        m, prob::VariableProjection, x₀::AbstractVector, p)
    return _init_elim_r3(m, prob, x₀, p)
end

# --- `start!!`, `step!!`, `finish!!` ---

@inline function start!!(
        st::StaticElimState, m, elim::BlockEliminated,
        x::StaticVector{N, T}, Fx, p, fnorm₀::R) where {N, T, R <: Real}
    y₀ = x[st.Y]
    z₀ = x[st.Z]
    y_star, Fz, _, inner_fail, _ = _eliminate_r1(
        elim, st.inner_state, y₀, z₀, p, R(NaN), st.Y, st.Z, st.E_y, st.E_z)
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(elim, y_star, fnorm₀, st.Y, st.Z, st.E_y, st.E_z)
    outer_state = start!!(st.outer_state, m_r, oprob, z₀, Fz, p, R(rnorm(Fz)))
    return StaticElimState(
        outer_state, st.inner_state, y_star, z₀, fnorm₀,
        st.Y, st.Z, st.E_y, st.E_z, inner_fail ? Int32(1) : Int32(0))
end

@inline function start!!(
        st::StaticElimState, m, vp::VariableProjection,
        x::StaticVector{N, T}, Fx, p, fnorm₀::R) where {N, T, R <: Real}
    y₀ = x[st.Y]
    z₀ = x[st.Z]
    y_star, r_star, _, inner_fail, _ = _eliminate_vp_r1(
        vp, y₀, z₀, p, R(NaN), st.Y, st.Z)
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(vp, y_star, fnorm₀, st.Y, st.Z, st.E_y, st.E_z)
    outer_state = start!!(st.outer_state, m_r, oprob, z₀, r_star, p, R(rnorm(r_star)))
    return StaticElimState(
        outer_state, st.inner_state, y_star, z₀, fnorm₀,
        st.Y, st.Z, st.E_y, st.E_z, inner_fail ? Int32(1) : Int32(0))
end

@inline function _start_elim_r3!!(
        outer_state, ws::BlockElimPrep, oprob_stored, m, elim::BlockEliminated,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p, fnorm₀::R) where {T, R <: Real}
    _gather_vec!(ws.y_warm, x, ws.Y_h, ws.Y_d)
    _gather_vec!(ws.z_curr, x, ws.Z_h, ws.Z_d)
    ws.fnorm_z_prev = R(NaN)
    ws.inner_failures = Int32(0)
    _eliminate_r3!(ws, elim, ws.z_curr, p)
    copyto!(ws.z_prev, ws.z_curr)
    Fx === ws.Fz_buf || copyto!(Fx, ws.Fz_buf)
    ws.fnorm_z_prev = fnorm₀
    ws.init_failures = ws.inner_failures > Int32(0) ? Int32(1) : Int32(0)
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = elim === oprob_stored.elim ? oprob_stored : _R3BlockOuterProb(elim, ws)
    return start!!(outer_state, m_r, oprob, ws.z_curr, Fx, p, R(rnorm(Fx)))
end

@inline function _start_elim_r3!!(
        outer_state, ws::VarProPrep, oprob, m, vp::VariableProjection,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p, fnorm₀::R) where {T, R <: Real}
    _gather_vec!(ws.y_warm, x, ws.Y_h, ws.Y_d)
    _gather_vec!(ws.z_curr, x, ws.Z_h, ws.Z_d)
    ws.fnorm_z_prev = R(NaN)
    ws.inner_failures = Int32(0)
    _eliminate_vp_r3!(ws, vp, ws.z_curr, p)
    Fx === ws.Fz_buf || copyto!(Fx, ws.Fz_buf)
    ws.fnorm_z_prev = fnorm₀
    ws.init_failures = ws.inner_failures > Int32(0) ? Int32(1) : Int32(0)
    m_r = Adapt.adapt(ToReal{R}(), m)
    return start!!(outer_state, m_r, oprob, ws.z_curr, Fx, p, R(rnorm(Fx)))
end

@inline function start!!(
        st::DenseElimState{OST, <:BlockElimPrep}, m, elim::BlockEliminated,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p,
        fnorm₀::R) where {OST, T, R <: Real}
    outer_state = _start_elim_r3!!(
        st.outer_state, st.ws, st.oprob, m, elim, x, Fx, p, fnorm₀)
    return DenseElimState(outer_state, st.ws, st.oprob)
end

@inline function start!!(
        st::DenseElimState{OST, <:VarProPrep}, m, vp::VariableProjection,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p,
        fnorm₀::R) where {OST, T, R <: Real}
    outer_state = _start_elim_r3!!(
        st.outer_state, st.ws, st.oprob, m, vp, x, Fx, p, fnorm₀)
    return DenseElimState(outer_state, st.ws, st.oprob)
end

@inline function _sync_outer_merit(outer_state::NewtonState, Fz, fnorm₊)
    merit = outer_state.merit === nothing ? nothing : norm2(Fz)
    return with_merit(outer_state, merit, fnorm₊)
end
@inline _sync_outer_merit(outer_state, _, _) = outer_state

@inline function step!!(
        st::StaticElimState, m, elim::BlockEliminated,
        x::StaticVector{N, T}, Fx, p) where {N, T}
    R = real(T)
    if !isfinite(st.fnorm_z_prev)
        return x, Fx, st, StepInfo{R}(NONFINITE, R(NaN), zero(R), Int32(0), false, false)
    end
    y_curr = st.y_warm
    Fx_curr = Fx
    pre_fail = false
    if !elim.linear_inner
        y_curr, Fx_curr, _, pre_fail, _ = _eliminate_r1(
            elim, st.inner_state, st.y_warm, st.z_curr, p, st.fnorm_z_prev,
            st.Y, st.Z, st.E_y, st.E_z)
    end
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(
        elim, y_curr, st.fnorm_z_prev, st.Y, st.Z, st.E_y, st.E_z)
    _, jac_fail = _schur_jacobian_r1(
        elim, _scatter_yz_r1(y_curr, st.z_curr, st.Y, st.Z, Val(N)),
        p, st.Y, st.Z, st.E_y, st.E_z)
    z₊, _, outer_state₊, info_outer = step!!(
        st.outer_state, m_r, oprob, st.z_curr, Fx_curr, p)
    y₊, Fz₊, x₊, step_fail, _ = _eliminate_r1(
        elim, st.inner_state, y_curr, z₊, p, st.fnorm_z_prev,
        st.Y, st.Z, st.E_y, st.E_z, st.z_curr)
    fnorm₊ = R(rnorm(Fz₊))
    outer_state₊ = _sync_outer_merit(outer_state₊, Fz₊, fnorm₊)
    extra_fail = st.init_failures +
                 ((pre_fail | jac_fail | step_fail) ? Int32(1) : Int32(0))
    info = StepInfo{R}(
        info_outer.code, fnorm₊, info_outer.stepnorm,
        info_outer.failures + extra_fail, info_outer.promoted, info_outer.limited)
    st₊ = StaticElimState(
        outer_state₊, st.inner_state, y₊, z₊, fnorm₊,
        st.Y, st.Z, st.E_y, st.E_z, Int32(0))
    return x₊, Fz₊, st₊, info
end

@inline function step!!(
        st::StaticElimState, m, vp::VariableProjection,
        x::StaticVector{N, T}, Fx, p) where {N, T}
    R = real(T)
    if !isfinite(st.fnorm_z_prev)
        return x, Fx, st, StepInfo{R}(NONFINITE, R(NaN), zero(R), Int32(0), false, false)
    end
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(
        vp, st.y_warm, st.fnorm_z_prev, st.Y, st.Z, st.E_y, st.E_z)
    x_curr = _scatter_yz_r1(st.y_warm, st.z_curr, st.Y, st.Z, Val(N))
    _, jac_fail = _varpro_jacobian_r1(vp, x_curr, Fx, p, st.Y, st.Z)
    z₊, _, outer_state₊, info_outer = step!!(
        st.outer_state, m_r, oprob, st.z_curr, Fx, p)
    y₊, r₊, x₊, step_fail, _ = _eliminate_vp_r1(
        vp, st.y_warm, z₊, p, st.fnorm_z_prev, st.Y, st.Z)
    fnorm₊ = R(rnorm(r₊))
    outer_state₊ = _sync_outer_merit(outer_state₊, r₊, fnorm₊)
    gr = state_gradratio(outer_state₊)
    fnorm_prev_vp = (isfinite(gr) && gr > zero(R)) ? min(fnorm₊, gr * fnorm₊) : fnorm₊
    extra_fail = st.init_failures + ((jac_fail | step_fail) ? Int32(1) : Int32(0))
    info = StepInfo{R}(
        info_outer.code, fnorm₊, info_outer.stepnorm,
        info_outer.failures + extra_fail, info_outer.promoted, info_outer.limited)
    st₊ = StaticElimState(
        outer_state₊, st.inner_state, y₊, z₊, fnorm_prev_vp,
        st.Y, st.Z, st.E_y, st.E_z, Int32(0))
    return x₊, r₊, st₊, info
end

@inline function _step_elim_r3!!(
        outer_state, ws::BlockElimPrep, oprob_stored, m, elim::BlockEliminated,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p) where {T}
    R = real(T)
    if !isfinite(ws.fnorm_z_prev)
        return x, Fx, outer_state,
        StepInfo{R}(NONFINITE, R(NaN), zero(R), Int32(0), false, false)
    end
    ws.inner_failures = Int32(0)
    if !elim.linear_inner
        _eliminate_r3!(ws, elim, ws.z_curr, p)
        Fx === ws.Fz_buf || copyto!(Fx, ws.Fz_buf)
        copyto!(ws.z_prev, ws.z_curr)
    end
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = elim === oprob_stored.elim ? oprob_stored : _R3BlockOuterProb(elim, ws)
    z₊, _, outer_state₊, info_outer = step!!(outer_state, m_r, oprob, ws.z_curr, Fx, p)
    z₊ === ws.z_curr || copyto!(ws.z_curr, z₊)
    _eliminate_r3!(ws, elim, ws.z_curr, p, false)
    copyto!(ws.z_prev, ws.z_curr)
    fnorm₊ = R(rnorm(ws.Fz_buf))
    Fx === ws.Fz_buf || copyto!(Fx, ws.Fz_buf)
    _scatter_yz!(x, ws.y_warm, ws.z_curr, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    ws.fnorm_z_prev = fnorm₊
    outer_state₊ = _sync_outer_merit(outer_state₊, Fx, fnorm₊)
    extra_fail = ws.init_failures + (ws.inner_failures > Int32(0) ? Int32(1) : Int32(0))
    ws.init_failures = Int32(0)
    info = StepInfo{R}(
        info_outer.code, fnorm₊, info_outer.stepnorm,
        info_outer.failures + extra_fail, info_outer.promoted, info_outer.limited)
    return x, Fx, outer_state₊, info
end

@inline function _step_elim_r3!!(
        outer_state, ws::VarProPrep, oprob, m, vp::VariableProjection,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p) where {T}
    R = real(T)
    if !isfinite(ws.fnorm_z_prev)
        return x, Fx, outer_state,
        StepInfo{R}(NONFINITE, R(NaN), zero(R), Int32(0), false, false)
    end
    ws.inner_failures = Int32(0)
    m_r = Adapt.adapt(ToReal{R}(), m)
    z₊, _, outer_state₊, info_outer = step!!(outer_state, m_r, oprob, ws.z_curr, Fx, p)
    z₊ === ws.z_curr || copyto!(ws.z_curr, z₊)
    _eliminate_vp_r3!(ws, vp, ws.z_curr, p)
    fnorm₊ = R(rnorm(ws.Fz_buf))
    Fx === ws.Fz_buf || copyto!(Fx, ws.Fz_buf)
    _scatter_yz!(x, ws.y_warm, ws.z_curr, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    outer_state₊ = _sync_outer_merit(outer_state₊, Fx, fnorm₊)
    gr = state_gradratio(outer_state₊)
    ws.fnorm_z_prev = (isfinite(gr) && gr > zero(R)) ? min(fnorm₊, gr * fnorm₊) :
                      fnorm₊
    extra_fail = ws.init_failures + (ws.inner_failures > Int32(0) ? Int32(1) : Int32(0))
    ws.init_failures = Int32(0)
    info = StepInfo{R}(
        info_outer.code, fnorm₊, info_outer.stepnorm,
        info_outer.failures + extra_fail, info_outer.promoted, info_outer.limited)
    return x, Fx, outer_state₊, info
end

@inline function step!!(
        st::DenseElimState{OST, <:BlockElimPrep}, m, elim::BlockEliminated,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p) where {OST, T}
    x, Fx, outer_state₊, info = _step_elim_r3!!(
        st.outer_state, st.ws, st.oprob, m, elim, x, Fx, p)
    st₊ = outer_state₊ === st.outer_state ? st :
          DenseElimState(outer_state₊, st.ws, st.oprob)
    return x, Fx, st₊, info
end

@inline function step!!(
        st::DenseElimState{OST, <:VarProPrep}, m, vp::VariableProjection,
        x::AbstractVector{T}, Fx::AbstractVector{T}, p) where {OST, T}
    x, Fx, outer_state₊, info = _step_elim_r3!!(
        st.outer_state, st.ws, st.oprob, m, vp, x, Fx, p)
    st₊ = outer_state₊ === st.outer_state ? st :
          DenseElimState(outer_state₊, st.ws, st.oprob)
    return x, Fx, st₊, info
end

@inline function finish!!(
        st::StaticElimState, m, prob::Union{BlockEliminated, VariableProjection},
        x::StaticVector{N, T}, Fx, status::SolverStatus{R}) where {N, T, R}
    m_r = Adapt.adapt(ToReal{R}(), m)
    oprob = _make_r1_outer_prob(
        prob, st.y_warm, st.fnorm_z_prev, st.Y, st.Z, st.E_y, st.E_z)
    _, status_out, outer_end = finish!!(
        st.outer_state, m_r, oprob, st.z_curr, Fx, status)
    if st.init_failures > Int32(0)
        status_out = SolverStatus{R}(
            status_out.code, status_out.iterations, status_out.fnorm,
            status_out.stepnorm, status_out.step_failures + st.init_failures,
            status_out.promoted, status_out.stationary,
            status_out.rank, status_out.nullnorm, status_out.gradratio)
    end
    x_end = all(isfinite, st.y_warm) ?
            _scatter_yz_r1(st.y_warm, st.z_curr, st.Y, st.Z, Val(N)) : x
    st_end = StaticElimState(
        outer_end, st.inner_state, st.y_warm, st.z_curr, R(NaN),
        st.Y, st.Z, st.E_y, st.E_z, Int32(0))
    return _like_iterate(x, x_end), status_out, st_end
end

@inline function _finish_elim_r3!!(
        st_init::DenseElimState, outer_state, m,
        ::Union{BlockEliminated, VariableProjection},
        x::AbstractVector{T}, Fx::AbstractVector{T},
        status::SolverStatus{R}) where {T, R}
    ws = st_init.ws
    m_r = Adapt.adapt(ToReal{R}(), m)
    _, status_out, outer_end = finish!!(
        outer_state, m_r, st_init.oprob, ws.z_curr, Fx, status)
    if ws.init_failures > Int32(0)
        status_out = SolverStatus{R}(
            status_out.code, status_out.iterations, status_out.fnorm,
            status_out.stepnorm, status_out.step_failures + ws.init_failures,
            status_out.promoted, status_out.stationary,
            status_out.rank, status_out.nullnorm, status_out.gradratio)
    end
    if all(isfinite, ws.y_warm)
        _scatter_yz!(x, ws.y_warm, ws.z_curr, ws.Y_h, ws.Z_h, ws.Y_d, ws.Z_d)
    end
    ws.fnorm_z_prev = R(NaN)
    ws.init_failures = Int32(0)
    st_end = outer_end === st_init.outer_state ? st_init :
             DenseElimState(outer_end, ws, st_init.oprob)
    return x, status_out, st_end
end

@inline function finish!!(
        st::DenseElimState, m, prob::Union{BlockEliminated, VariableProjection},
        x::AbstractVector{T}, Fx::AbstractVector{T},
        status::SolverStatus{R}) where {T, R}
    return _finish_elim_r3!!(st, st.outer_state, m, prob, x, Fx, status)
end

# R3 `solve!!` keeps `outer_state` unboxed on its stack frame across `start!!`, `step!!`,
# and `finish!!` rather than re-wrapping `DenseElimState` at every step.
function solve!!(
        st::DenseElimState, prob::Union{BlockEliminated, VariableProjection},
        x₀::AbstractVector{T}, method, p = nothing,
        options::Options{R} = Options(eltype(x₀))) where {T, R <: Real}
    check_options(eltype(x₀), options)
    x = x₀
    ws = st.ws
    oprob = st.oprob
    Fx = primal_residual!!(residual_buffer(st), prob, x, p)
    fnorm₀ = rnorm(Fx)
    outer_state = _start_elim_r3!!(
        st.outer_state, ws, oprob, method, prob, x, Fx, p, fnorm₀)
    status = SolverStatus{R}(SUCCESS, Int32(0), fnorm₀, zero(R), Int32(0), false, false)
    if !isfinite(fnorm₀)
        return _finish_elim_r3!!(
            st, outer_state, method, prob, x, Fx, end_status(status, NONFINITE))
    end
    if residual_converged(options, status, fnorm₀)
        return _finish_elim_r3!!(st, outer_state, method, prob, x, Fx, status)
    end
    if options.max_iterations == Int32(0)
        return _finish_elim_r3!!(
            st, outer_state, method, prob, x, Fx, end_status(status, MAXITERS))
    end
    stalls = Int32(0)
    while true
        x, Fx, outer_state, info = _step_elim_r3!!(
            outer_state, ws, oprob, method, prob, x, Fx, p)
        status = record(status, info)
        if info.code == SINGULAR || info.code == NONFINITE
            return _finish_elim_r3!!(st, outer_state, method, prob, x, Fx, status)
        end
        if info.code == LINESEARCH_FAILED && iszero(info.stepnorm) &&
           !stationary_converged(prob, options, outer_state, status)
            return _finish_elim_r3!!(st, outer_state, method, prob, x, Fx, status)
        end
        if residual_converged(options, status, fnorm₀)
            return _finish_elim_r3!!(
                st, outer_state, method, prob, x, Fx, ok_status(status))
        end
        refreshed = false
        if stationary_converged(prob, options, outer_state, status)
            outer_state = _refresh_elim_r3!!(
                outer_state, ws, oprob, method, prob, x, Fx, p)
            refreshed = true
            if stationary_converged(prob, options, outer_state, status)
                return _finish_elim_r3!!(
                    st, outer_state, method, prob, x, Fx, stationary_status(status))
            end
        end
        if !(info.code == STALLED || info.limited) &&
           (step_converged(options, status, rnorm(x)) ||
            (info.code == SUCCESS && iszero(info.stepnorm) &&
             status.iterations >= options.min_iterations))
            refreshed || (outer_state = _refresh_elim_r3!!(
                outer_state, ws, oprob, method, prob, x, Fx, p))
            if stationary_converged(prob, options, outer_state, status)
                return _finish_elim_r3!!(
                    st, outer_state, method, prob, x, Fx, stationary_status(status))
            end
            return _finish_elim_r3!!(
                st, outer_state, method, prob, x, Fx, end_status(status, STALLED))
        end
        if info.code == STALLED
            stalls += one(Int32)
            if stalls >= options.max_stalls
                return _finish_elim_r3!!(
                    st, outer_state, method, prob, x, Fx, end_status(status, STALLED))
            end
        else
            stalls = Int32(0)
        end
        if status.iterations >= options.max_iterations
            return _finish_elim_r3!!(
                st, outer_state, method, prob, x, Fx, end_status(status, MAXITERS))
        end
    end
end

# --- `init`, `solve!`, `solve_implicit`, and `solve` ---

function init(
        prob::Union{BlockEliminated, VariableProjection}, x₀, m = Newton(), p = nothing;
        b = nothing, options = nothing, kwargs...)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p)) || has_dual_type(typeof(b))
        throw(ArgumentError(
            "init does not accept ForwardDiff.Dual inputs; use solve(prob, x₀, method, p) for implicit differentiation"))
    end
    prob_b = _with_datum(prob, b)
    x, m_ad, opt, state = prepare_solve(prob_b, x₀, m, p, options, kwargs)
    return NonlinearSolver(prob_b, x, m_ad, opt, state, p)
end

function _solve_elim_stationarity_solver!(
        solver::NonlinearSolver{T, P, M, O, S, X, Q}, x, p, b) where {T, P, M, O, S, X, Q}
    if has_dual_type(typeof(b))
        throw(ArgumentError(
            "solve! does not accept ForwardDiff.Dual inputs; use solve(prob, x₀, method, p) for implicit differentiation"))
    end
    prob_b = _with_datum(solver.prob, b)
    _validate_elim_x(prob_b, x, p)
    solver.x = x
    solver.p = p
    x₊, status, state = solve!!(
        solver.state, prob_b, x, solver.method, p, solver.options)
    solver.x = x₊
    solver.state === state || (solver.state = state)
    return solver.x, status
end

@inline function _elim_implicit_tangent_r1(
        ::Type{D}, prob_in::AbstractNonlinearProblem,
        x_star::StaticVector{S, T}, p, p_val,
        status::SolverStatus) where {Tg, V, N, D <: ForwardDiff.Dual{Tg, V, N}, S, T}
    Jmat = _inner_jac_r1(AutoForwardDiff(), prob_in, x_star, p_val)
    fac = factorize!!(nothing, LUFactorization(), Jmat)
    if returncode(fac) === SINGULAR
        return _make_nan_duals(D, x_star), _singular_status(status)
    end
    x_hat = _seed_zero_duals(D, x_star)
    r_dual = _inner_res_r1(prob_in, x_hat, p)
    tangents = _solve_static_tangents(Val(N), T, fac, r_dual)
    return _assemble_static_duals(D, x_star, tangents), status
end

function _elim_implicit_tangent_r3(
        ::Type{D}, ws::BlockElimPrep,
        prob_in::AbstractNonlinearProblem,
        x_star::AbstractVector{T}, p, p_val,
        status::SolverStatus) where {Tg, V, N, D <: ForwardDiff.Dual{Tg, V, N}, T}
    _inner_jac_r3!(ws.J_buf, ws.inner_prep, prob_in, x_star, p_val)
    m_rows, n_cols = size(ws.J_buf)
    ls = LUFactorization()
    fac_slot = init_linear(ls, nothing, x_star)
    fac = factorize!!(fac_slot, ls, ws.J_buf)
    if returncode(fac) === SINGULAR
        return _make_nan_duals(D, x_star), _singular_status(status)
    end
    x_hat = _seed_zero_duals(D, x_star)
    r_dual = similar(x_hat, m_rows)
    prob_in.F(r_dual, x_hat, p)
    b_buf = similar(x_star, m_rows)
    d_buf = similar(x_star, n_cols)
    x_out = similar(x_hat)
    x_out .= ZeroDual{D}().(x_star)
    for k in 1:N
        b_buf .= ExtractPartial{T}(k).(r_dual)
        d_k = ldiv!!(d_buf, fac, b_buf)
        x_out .= SetPartial{D}(k).(x_out, d_k)
    end
    return x_out, status
end

function solve_implicit(
        vp::VariableProjection, x₀, method, p,
        options::Union{Nothing, Options}, kwargs)
    return solve_implicit(getfield(vp, :prob), x₀, method, p, options, kwargs)
end

function solve_implicit(
        prob::BlockEliminated, x₀, method, p,
        options::Union{Nothing, Options}, kwargs)
    prob_in = getfield(prob, :prob)
    if prob_in isa StationarityProblem
        _refuse_stationarity_duals(:solve, x₀, p)
    end
    D_in = validate_dual_types(typeof(x₀), typeof(p))
    x_val = Adapt.adapt(StripDual(), x₀)
    p_val = Adapt.adapt(StripDual(), p)
    if has_dual_type(typeof(x_val)) || has_dual_type(typeof(p_val))
        throw(ArgumentError(
            "Adapt.adapt(StripDual(), ...) left a ForwardDiff.Dual in x₀ or p; mark custom parameter structs with Adapt.@adapt_structure"))
    end
    x, m, opt, state = prepare_solve(prob, x_val, method, p_val, options, kwargs)
    x_star, status, state = solve!!(state, prob, x, m, p_val, opt)
    D = _working_dual_type(D_in, eltype(x_star))
    if status.code !== SUCCESS
        return _make_nan_duals(D, x_star), status
    end
    if !has_dual_type(typeof(p))
        return _seed_zero_duals(D, x_star), status
    end
    if x_star isa StaticVector
        return _elim_implicit_tangent_r1(D, prob_in, x_star, p, p_val, status)
    else
        return _elim_implicit_tangent_r3(D, state.ws, prob_in, x_star, p, p_val, status)
    end
end

function solve(
        prob::Union{BlockEliminated, VariableProjection}, x₀, m = Newton(), p = nothing;
        b = nothing, options = nothing, kwargs...)
    if has_dual_type(typeof(b))
        throw(ArgumentError(
            "solve does not accept ForwardDiff.Dual in keyword `b`"))
    end
    prob_b = _with_datum(prob, b)
    if has_dual_type(typeof(x₀)) || has_dual_type(typeof(p))
        return solve_implicit(prob_b, x₀, m, p, options, kwargs)
    end
    x, m_ad, opt, state = prepare_solve(prob_b, x₀, m, p, options, kwargs)
    x_star, status, _ = solve!!(state, prob_b, x, m_ad, p, opt)
    return x_star, status
end
