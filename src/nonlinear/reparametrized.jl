"""
    Reparametrized(prob, g, ∂g = nothing;
                   covariant = prob isa StationarityProblem, curvature = true)

Coordinate-chart wrapper `x = g(y, p)` around an [`AbstractNonlinearProblem`](@ref) `prob`
([`NonlinearProblem`](@ref) or [`StationarityProblem`](@ref)).

Solving a `Reparametrized` problem iterates in the chart coordinates `y`. [`solve`](@ref) and
[`init`](@ref) / [`solve!`](@ref) return `(x*, status)` and [`solve!!`](@ref) returns
`(x*, status, state)`, where `x* = g(y*, p)` is evaluated at the returned chart iterate `y*`;
[`solve_batch!`](@ref) writes each system's `x*` into the batch. For a
[`StationarityProblem`](@ref), the keyword `b` of [`init`](@ref) and [`solve`](@ref) sets the
datum of the inner problem, as it does without the chart; with any other inner problem, `b`
raises an `ArgumentError`.

# Arguments and keywords

- `prob`: the inner [`NonlinearProblem`](@ref) or [`StationarityProblem`](@ref) in `x`-coordinates.
  A [`StationarityProblem`](@ref) needs `U = :` or `U = 1:n` in order, for the chart dimension
  `n`, and `E = :` or a permutation of `1:n`; [`init`](@ref), [`solve`](@ref),
  [`solve_batch!`](@ref) and [`prepare_ad`](@ref) raise an `ArgumentError` for any other index set.
  For an R1 iterate the index sets are `:`, a tuple or an `SVector`, as for a
  [`StationarityProblem`](@ref) without a chart, so a range such as `1:n` raises an
  `ArgumentError` there.
- `g`: the chart map `x = g(y, p)` in R1 (`StaticVector` or `Number`) or `g!(x, y, p)` in R3
  (`AbstractVector`).
- `∂g = nothing`: optional chart Jacobian `G = ∂g(y, p)` in R1 or `∂g!(G, y, p)` in R3. When
  `∂g === nothing`, `G(y, p) = ∂g/∂y` is computed by forward-mode automatic differentiation.
- `covariant::Bool`: when `false` (default for [`NonlinearProblem`](@ref)), the transformed residual
  and Jacobian are
  ``\\tilde{F}(y, p) = F(g(y, p), p)`` and ``\\tilde{J}(y, p) = J(g(y, p), p) G(y, p)``.
  When `true` (default for [`StationarityProblem`](@ref)), the residual is pulled back covariantly as
  ``\\tilde{F}(y, p) = G(y, p)^\\top F(g(y, p), p)``.
- `curvature::Bool = true`: when `covariant == true`, includes the chart-curvature term
  ``(\\partial G^\\top / \\partial y) F(g(y, p), p)`` in
  ``\\tilde{J}(y, p) = G^\\top J G + (\\partial G^\\top / \\partial y) F`` when `true`, or uses the
  pull-back approximation ``\\tilde{J}(y, p) = G^\\top J G`` when `false`.
"""
struct Reparametrized{P <: AbstractNonlinearProblem, G, DG, C, K} <:
       AbstractNonlinearProblem
    prob::P
    g::G
    ∂g::DG
    # `adapt` rebuilds a `Reparametrized` through this constructor, and a device kernel cannot
    # format a message, so the messages are constant.
    function Reparametrized{P, G, DG, C, K}(
            prob::P, g::G, ∂g::DG) where {P <: AbstractNonlinearProblem, G, DG, C, K}
        C isa Bool || throw(ArgumentError("Reparametrized type parameter C must be a Bool"))
        K isa Bool || throw(ArgumentError("Reparametrized type parameter K must be a Bool"))
        return new{P, G, DG, C, K}(prob, g, ∂g)
    end
end

@inline is_stationarity_problem(::StationarityProblem) = true
@inline is_stationarity_problem(::AbstractNonlinearProblem) = false

# The prototype of the problem the chart wraps: a batch takes the problem by value into a kernel, so
# a sparse prototype of the inner problem must be refused there as it is for the inner problem
# itself.
@inline jacobian_prototype(rep::Reparametrized) = jacobian_prototype(rep.prob)

# The chart evaluates the inner problem at the full state `x = g(y, p)` and reads its Jacobian
# as `∂F/∂x`, so a `StationarityProblem` needs `U = :` or `U = 1:n` in order; with the checks of
# `_validate_stationarity_solve`, `E` is then `:` or a permutation of `1:n`.
@inline _check_chart_indices(::AbstractNonlinearProblem, y) = nothing

# Whether the index set `U` is `1:n` in order. A device index set is compared by a broadcast and a
# reduction on the device, as scalar indexing of a device array is disallowed.
@inline _in_order(U, n::Int) = length(U) == n && all(c -> U[c] == c, eachindex(U))
@inline _in_order(U::AbstractGPUArray, n::Int) = length(U) == n && all(U .== (1:n))

function _check_chart_indices(prob::StationarityProblem, y)
    _validate_stationarity_solve(prob, y, nothing)
    U = prob.U
    n = length(y)
    U isa Colon || _in_order(U, n) ||
        throw(ArgumentError(
            "Reparametrized needs a StationarityProblem with U = : or U = 1:n in order, " *
            "where n = $n is the chart dimension; got U = $U"))
    return nothing
end

function Reparametrized(
        prob::P, g::G, ∂g::DG = nothing;
        covariant::Bool = is_stationarity_problem(prob),
        curvature::Bool = true) where {P <: AbstractNonlinearProblem, G, DG}
    return Reparametrized{P, G, DG, covariant, curvature}(prob, g, ∂g)
end

@inline function Base.getproperty(
        r::Reparametrized{P, G, DG, C, K}, s::Symbol) where {P, G, DG, C, K}
    s === :covariant && return C
    s === :curvature && return K
    s === :prob && return getfield(r, :prob)
    s === :g && return getfield(r, :g)
    s === :∂g && return getfield(r, :∂g)
    throw(ArgumentError("type Reparametrized has no field $s"))
end

Base.propertynames(::Reparametrized) = (:prob, :g, :∂g, :covariant, :curvature)

function Adapt.adapt_structure(
        to, r::Reparametrized{P, G, DG, C, K}) where {P, G, DG, C, K}
    prob = Adapt.adapt(to, getfield(r, :prob))
    g = Adapt.adapt(to, getfield(r, :g))
    ∂g = Adapt.adapt(to, getfield(r, :∂g))
    return Reparametrized{typeof(prob), typeof(g), typeof(∂g), C, K}(prob, g, ∂g)
end

struct ReparamMapTag{F, R} end
struct ReparamCurvTag{F, R} end

@inline _dual_val(d::ForwardDiff.Dual) = ForwardDiff.value(d)
@inline function _dual_val(d::Complex{<:ForwardDiff.Dual})
    return Complex(ForwardDiff.value(real(d)), ForwardDiff.value(imag(d)))
end

@inline function _nested_grad(d::ForwardDiff.Dual, i::Int)
    return ForwardDiff.partials(ForwardDiff.value(d), i)
end
@inline function _nested_grad(d::Complex{<:ForwardDiff.Dual}, i::Int)
    return Complex(
        ForwardDiff.partials(ForwardDiff.value(real(d)), i),
        ForwardDiff.partials(ForwardDiff.value(imag(d)), i))
end

@inline function _nested_hess(d::ForwardDiff.Dual, i::Int, j::Int)
    return ForwardDiff.partials(ForwardDiff.partials(d, j), i)
end
@inline function _nested_hess(d::Complex{<:ForwardDiff.Dual}, i::Int, j::Int)
    return Complex(
        ForwardDiff.partials(ForwardDiff.partials(real(d), j), i),
        ForwardDiff.partials(ForwardDiff.partials(imag(d), j), i))
end

@inline _nested_hess1(d, i::Int) = _nested_hess(d, i, 1)

@inline function _nested_dual_type(::Type{T}, ::Type{TgIn}, ::Type{TgOut}, NIn::Int,
        NOut::Int) where {
        T <: Real, TgIn, TgOut}
    D_in = ForwardDiff.Dual{TgIn, T, NIn}
    return ForwardDiff.Dual{TgOut, D_in, NOut}
end

@inline function _nested_dual_type(
        ::Type{Complex{T}}, ::Type{TgIn}, ::Type{TgOut}, NIn::Int, NOut::Int) where {
        T <: Real, TgIn, TgOut}
    D_in = ForwardDiff.Dual{TgIn, T, NIn}
    return Complex{ForwardDiff.Dual{TgOut, D_in, NOut}}
end

@inline function _seed_nest_r1(
        ::Type{T}, ::Type{TgIn}, ::Type{TgOut}, yi::T, ::Val{N},
        i::Int) where {
        T <: Real, TgIn, TgOut, N}
    D_in = ForwardDiff.Dual{TgIn, T, N}
    D_out = ForwardDiff.Dual{TgOut, D_in, N}
    p_in = ForwardDiff.Partials{N, T}(
        ntuple(j -> ifelse(j == i, one(T), zero(T)), Val(N)))
    v_in = D_in(yi, p_in)
    p_out = ForwardDiff.Partials{N, D_in}(
        ntuple(j -> ifelse(j == i, one(D_in), zero(D_in)), Val(N)))
    return D_out(v_in, p_out)
end

@inline function _seed_nest_r1(
        ::Type{Complex{T}}, ::Type{TgIn}, ::Type{TgOut}, yi::Complex{T},
        ::Val{N}, i::Int) where {
        T <: Real, TgIn, TgOut, N}
    D_in = ForwardDiff.Dual{TgIn, T, N}
    D_out = ForwardDiff.Dual{TgOut, D_in, N}
    p_in = ForwardDiff.Partials{N, T}(
        ntuple(j -> ifelse(j == i, one(T), zero(T)), Val(N)))
    re_in = D_in(real(yi), p_in)
    im_in = D_in(imag(yi))
    p_out = ForwardDiff.Partials{N, D_in}(
        ntuple(j -> ifelse(j == i, one(D_in), zero(D_in)), Val(N)))
    return Complex(D_out(re_in, p_out), D_out(im_in))
end

@inline _inner_res_r1(prob::NonlinearProblem, x, p) = prob.F(x, p)
@inline function _inner_res_r1(prob::AbstractNonlinearProblem, x, p)
    return residual!!(nothing, prob, x, p)
end

@inline function _inner_jac_r1(backend, prob::NonlinearProblem, x, p)
    prob.J !== nothing && return prob.J(x, p)
    return jacobian!!(nothing, backend, prob, x, p)
end
@inline function _inner_jac_r1(backend, prob::AbstractNonlinearProblem, x, p)
    return jacobian!!(nothing, backend, prob, x, p)
end

@inline function _inner_res_jac_r1(backend, prob::StationarityProblem, x, p)
    return gradient_hessian!!(nothing, nothing, backend, prob, x, p)
end
@inline function _inner_res_jac_r1(backend, prob::AbstractNonlinearProblem, x, p)
    return _inner_res_r1(prob, x, p), _inner_jac_r1(backend, prob, x, p)
end

@inline function _eval_dg_r1(
        reparam::Reparametrized{P, G, DG, C, K}, y, p) where {P, G, DG, C, K}
    return reparam.∂g(y, p)
end

@inline function _eval_dg_r1(
        reparam::Reparametrized{P, G, Nothing, C, K},
        y::StaticVector{N, T}, p) where {P, G, C, K, N, T}
    R = real(T)
    Tg = ReparamMapTag{G, R}
    D = dual_type(T, Tg, N)
    y_dual = SVector{N, D}(ntuple(
        i -> seed(zero(D), y[i], ntuple(==(i), Val(N))), Val(N)))
    x_dual = reparam.g(y_dual, p)
    return SMatrix{N, N, T, N * N}(ntuple(
        idx -> dual_partial(x_dual[rem(idx - 1, N) + 1], div(idx - 1, N) + 1), Val(N * N)))
end

@inline function _eval_dg_r1(
        reparam::Reparametrized{P, G, Nothing, C, K},
        y::T, p) where {P, G, C, K, T <: Number}
    R = real(T)
    Tg = ReparamMapTag{G, R}
    D = dual_type(T, Tg, 1)
    y_dual = seed(zero(D), y, (one(R),))
    x_dual = reparam.g(y_dual, p)
    return dual_partial(x_dual, 1)
end

@inline function _curv_entry_r1(
        x_nest::StaticVector{N}, Fx::StaticVector{N, T}, i::Int, j::Int) where {N, T}
    s = zero(T)
    @inbounds for k in 1:N
        s += _nested_hess(x_nest[k], i, j) * Fx[k]
    end
    return s
end

@inline function _eval_dg_and_curv_r1(
        reparam::Reparametrized{P, G, DG, C, K},
        y::StaticVector{N, T}, Fx::StaticVector{N, T}, p) where {P, G, DG, C, K, N, T}
    R = real(T)
    Tg = ReparamCurvTag{DG, R}
    D = dual_type(T, Tg, N)
    y_dual = SVector{N, D}(ntuple(
        i -> seed(zero(D), y[i], ntuple(==(i), Val(N))), Val(N)))
    G_dual = reparam.∂g(y_dual, p)
    G_val = SMatrix{N, N, T, N * N}(ntuple(idx -> _dual_val(G_dual[idx]), Val(N * N)))
    psi_dual = transpose(G_dual) * Fx
    H_curv = SMatrix{N, N, T, N * N}(ntuple(
        idx -> dual_partial(psi_dual[rem(idx - 1, N) + 1], div(idx - 1, N) + 1), Val(N * N)))
    return G_val, H_curv
end

@inline function _eval_dg_and_curv_r1(
        reparam::Reparametrized{P, G, DG, C, K},
        y::T, Fx::T, p) where {P, G, DG, C, K, T <: Number}
    R = real(T)
    Tg = ReparamCurvTag{DG, R}
    D = dual_type(T, Tg, 1)
    y_dual = seed(zero(D), y, (one(R),))
    G_dual = reparam.∂g(y_dual, p)
    return _dual_val(G_dual), dual_partial(G_dual, 1) * Fx
end

@inline function _eval_dg_and_curv_r1(
        reparam::Reparametrized{P, G, Nothing, C, K},
        y::StaticVector{N, T}, Fx::StaticVector{N, T}, p) where {P, G, C, K, N, T}
    R = real(T)
    TgIn = ReparamMapTag{G, R}
    TgOut = ReparamCurvTag{G, R}
    DN = _nested_dual_type(T, TgIn, TgOut, N, N)
    y_nest = SVector{N, DN}(ntuple(
        i -> _seed_nest_r1(T, TgIn, TgOut, y[i], Val(N), i), Val(N)))
    x_nest = reparam.g(y_nest, p)
    G_val = SMatrix{N, N, T, N * N}(ntuple(
        idx -> _nested_grad(x_nest[rem(idx - 1, N) + 1], div(idx - 1, N) + 1), Val(N * N)))
    H_curv = SMatrix{N, N, T, N * N}(ntuple(
        idx -> _curv_entry_r1(x_nest, Fx, rem(idx - 1, N) + 1, div(idx - 1, N) + 1),
        Val(N * N)))
    return G_val, H_curv
end

@inline function _eval_dg_and_curv_r1(
        reparam::Reparametrized{P, G, Nothing, C, K},
        y::T, Fx::T, p) where {P, G, C, K, T <: Number}
    R = real(T)
    TgIn = ReparamMapTag{G, R}
    TgOut = ReparamCurvTag{G, R}
    y_nest = _seed_nest_r1(T, TgIn, TgOut, y, Val(1), 1)
    x_nest = reparam.g(y_nest, p)
    return _nested_grad(x_nest, 1), _nested_hess(x_nest, 1, 1) * Fx
end

# --- R1 residual!!, jacobian!!, jvp!!, prepare_ad ---

@inline function prepare_ad(
        ::Nothing, reparam::Reparametrized, ::Nothing, y::Union{StaticVector, Number}, p)
    _check_chart_indices(reparam.prob, y)
    return AutoForwardDiff()
end

@inline function residual!!(
        ::Nothing, reparam::Reparametrized{P, G, DG, false, K},
        y::Union{StaticVector, Number}, p) where {P, G, DG, K}
    x = reparam.g(y, p)
    return _inner_res_r1(reparam.prob, x, p)
end

@inline function residual!!(
        ::Nothing, reparam::Reparametrized{P, G, DG, true, K},
        y::Union{StaticVector, Number}, p) where {P, G, DG, K}
    x = reparam.g(y, p)
    Fx = _inner_res_r1(reparam.prob, x, p)
    G_mat = _eval_dg_r1(reparam, y, p)
    return transpose(G_mat) * Fx
end

@inline function _reparam_jac_r1(
        backend, reparam::Reparametrized{P, G, DG, false, K},
        y::Union{StaticVector, Number}, p) where {P, G, DG, K}
    b = backend === nothing ? AutoForwardDiff() : backend
    x = reparam.g(y, p)
    Jx = _inner_jac_r1(b, reparam.prob, x, p)
    G_mat = _eval_dg_r1(reparam, y, p)
    return Jx * G_mat
end

@inline function _reparam_jac_r1(
        backend, reparam::Reparametrized{P, G, DG, true, false},
        y::Union{StaticVector, Number}, p) where {P, G, DG}
    b = backend === nothing ? AutoForwardDiff() : backend
    x = reparam.g(y, p)
    Jx = _inner_jac_r1(b, reparam.prob, x, p)
    G_mat = _eval_dg_r1(reparam, y, p)
    return transpose(G_mat) * Jx * G_mat
end

@inline function _reparam_jac_r1(
        backend, reparam::Reparametrized{P, G, DG, true, true},
        y::Union{StaticVector, Number}, p) where {P, G, DG}
    b = backend === nothing ? AutoForwardDiff() : backend
    x = reparam.g(y, p)
    Fx, Jx = _inner_res_jac_r1(b, reparam.prob, x, p)
    G_mat, H_curv = _eval_dg_and_curv_r1(reparam, y, Fx, p)
    return transpose(G_mat) * Jx * G_mat + H_curv
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, reparam::Reparametrized,
        y::StaticVector{N, T}, p) where {N, T}
    return _reparam_jac_r1(backend, reparam, y, p)
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, reparam::Reparametrized,
        y::Number, p)
    return _reparam_jac_r1(backend, reparam, y, p)
end

@inline function jacobian!!(
        ::Nothing, ::Nothing, reparam::Reparametrized,
        y::Union{StaticVector, Number}, p)
    return _reparam_jac_r1(AutoForwardDiff(), reparam, y, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, reparam::Reparametrized,
        y::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return _reparam_jac_r1(backend, reparam, y, p) * v
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, reparam::Reparametrized,
        y::T, v::T, p) where {T <: Number}
    return _reparam_jac_r1(backend, reparam, y, p) * v
end

@inline function jvp!!(
        ::Nothing, ::Nothing, reparam::Reparametrized,
        y::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return _reparam_jac_r1(AutoForwardDiff(), reparam, y, p) * v
end

@inline function jvp!!(
        ::Nothing, ::Nothing, reparam::Reparametrized,
        y::T, v::T, p) where {T <: Number}
    return _reparam_jac_r1(AutoForwardDiff(), reparam, y, p) * v
end

# --- R3 workspace and AD preparation (`ReparametrizedPrep`) ---

struct ReparametrizedPrep{N, VX, MG, MJ, VI, IP, YD, XD, GD, YN, XN}
    x_buf::VX
    Fx_buf::VX
    v_work::VX
    G_buf::MG
    Jx_buf::MJ
    W_mat::MG
    J_jvp::MG
    idx::VI
    inner_prep::IP
    y_dual::YD
    x_dual::XD
    G_dual::GD
    y_nest1::YN
    x_nest1::XN
end

@inline function _seed_reparam_1(
        ::ForwardDiff.Dual{Tg, R, N}, yi::R, i::Int, j0::Int) where {Tg, R <: Real, N}
    p_in = ForwardDiff.Partials{N, R}(
        ntuple(k -> ifelse(i == j0 + k, one(R), zero(R)), Val(N)))
    return ForwardDiff.Dual{Tg, R, N}(yi, p_in)
end

@inline function _seed_reparam_1(
        ::Complex{ForwardDiff.Dual{Tg, R, N}}, yi::Complex{R}, i::Int, j0::Int) where {
        Tg, R <: Real, N}
    p_in = ForwardDiff.Partials{N, R}(
        ntuple(k -> ifelse(i == j0 + k, one(R), zero(R)), Val(N)))
    return Complex(
        ForwardDiff.Dual{Tg, R, N}(real(yi), p_in),
        ForwardDiff.Dual{Tg, R, N}(imag(yi)))
end

@inline function _seed_reparam_nest1(
        ::ForwardDiff.Dual{TgOut, ForwardDiff.Dual{TgIn, R, N}, 1},
        yi::R, i::Int, a0::Int, j::Int) where {TgOut, TgIn, R <: Real, N}
    D_in = ForwardDiff.Dual{TgIn, R, N}
    p_in = ForwardDiff.Partials{N, R}(
        ntuple(k -> ifelse(i == a0 + k, one(R), zero(R)), Val(N)))
    v_in = D_in(yi, p_in)
    p_out = ForwardDiff.Partials{1, D_in}((ifelse(i == j, one(D_in), zero(D_in)),))
    return ForwardDiff.Dual{TgOut, D_in, 1}(v_in, p_out)
end

@inline function _seed_reparam_nest1(
        ::Complex{ForwardDiff.Dual{TgOut, ForwardDiff.Dual{TgIn, R, N}, 1}},
        yi::Complex{R}, i::Int, a0::Int, j::Int) where {TgOut, TgIn, R <: Real, N}
    D_in = ForwardDiff.Dual{TgIn, R, N}
    D_out = ForwardDiff.Dual{TgOut, D_in, 1}
    p_in = ForwardDiff.Partials{N, R}(
        ntuple(k -> ifelse(i == a0 + k, one(R), zero(R)), Val(N)))
    re_in = D_in(real(yi), p_in)
    im_in = D_in(imag(yi))
    p_out = ForwardDiff.Partials{1, D_in}((ifelse(i == j, one(D_in), zero(D_in)),))
    return Complex(D_out(re_in, p_out), D_out(im_in))
end

# The buffer of the inner Jacobian `∂F/∂x`. A user `J` writes the storage of the inner problem's
# `jacobian_prototype`, such as the `nonzeros` of a `SparseMatrixCSC`, so the buffer takes that
# storage where both are given. An automatic-differentiation Jacobian fills a dense buffer.
@inline function _inner_jac_buffer(::AbstractNonlinearProblem, y, ::Type{T}, n::Int) where {T}
    return similar(y, T, n, n)
end
@inline function _inner_jac_buffer(prob::NonlinearProblem, y, ::Type{T}, n::Int) where {T}
    P = prob.jacobian_prototype
    return P === nothing || prob.J === nothing ? similar(y, T, n, n) : similar(P, T)
end

@inline function _prepare_inner_r3(backend, prob::NonlinearProblem, Fx_buf, x_buf, p)
    prob.J !== nothing && return nothing
    return prepare_ad(backend, prob, Fx_buf, x_buf, p)
end
@inline function _prepare_inner_r3(
        backend, prob::AbstractNonlinearProblem, Fx_buf, x_buf, p)
    return prepare_ad(backend, prob, Fx_buf, x_buf, p)
end

function _prepare_ad_reparam_r3(
        backend, reparam::Reparametrized{P, G, DG, C, K},
        r::AbstractArray, y::AbstractArray, p) where {P, G, DG, C, K}
    check_iterate_shape(y)
    _check_chart_indices(reparam.prob, y)
    n = length(y)
    T = eltype(y)
    R = real(T)
    b = backend isa AutoForwardDiff ? backend : AutoForwardDiff()
    N = chunk_size(b, n)
    x_buf = similar(y, T, n)
    Fx_buf = similar(y, T, n)
    v_work = similar(y, T, n)
    G_buf = similar(y, T, n, n)
    Jx_buf = _inner_jac_buffer(reparam.prob, y, T, n)
    W_mat = similar(y, T, n, n)
    J_jvp = similar(y, T, n, n)
    idx = similar(y, Int, n)
    copyto!(idx, collect(1:n))
    reparam.g(x_buf, y, p)
    inner_prep = _prepare_inner_r3(backend, reparam.prob, Fx_buf, x_buf, p)

    TgIn = ReparamMapTag{G, R}
    TgOut = ReparamCurvTag{G, R}
    D1 = dual_type(T, TgIn, N)
    y_dual = (DG === Nothing || (C && K)) ? similar(y, D1, n) : nothing
    x_dual = DG === Nothing ? similar(y, D1, n) : nothing
    G_dual = (DG !== Nothing && C && K) ? similar(y, D1, n, n) : nothing
    if DG === Nothing && C && K
        DN1 = _nested_dual_type(T, TgIn, TgOut, N, 1)
        y_nest1 = similar(y, DN1, n)
        x_nest1 = similar(y, DN1, n)
    else
        y_nest1 = nothing
        x_nest1 = nothing
    end
    return ReparametrizedPrep{
        N, typeof(x_buf), typeof(G_buf), typeof(Jx_buf), typeof(idx), typeof(inner_prep),
        typeof(y_dual), typeof(x_dual), typeof(G_dual), typeof(y_nest1), typeof(x_nest1)}(
        x_buf, Fx_buf, v_work, G_buf, Jx_buf, W_mat, J_jvp, idx, inner_prep,
        y_dual, x_dual, G_dual, y_nest1, x_nest1)
end

@inline function prepare_ad(
        backend::AutoForwardDiff, reparam::Reparametrized,
        r::AbstractArray, y::AbstractArray, p)
    return _prepare_ad_reparam_r3(backend, reparam, r, y, p)
end

@inline function prepare_ad(
        backend::AbstractADType, reparam::Reparametrized,
        r::AbstractArray, y::AbstractArray, p)
    return _prepare_ad_reparam_r3(backend, reparam, r, y, p)
end

@inline function prepare_ad(
        ::Nothing, reparam::Reparametrized,
        r::AbstractArray, y::AbstractArray, p)
    return _prepare_ad_reparam_r3(AutoForwardDiff(), reparam, r, y, p)
end

@inline function _inner_res_r3!(Fx, inner_prep, prob::NonlinearProblem, x, p)
    prob.F(Fx, x, p)
    return Fx
end
@inline function _inner_res_r3!(Fx, inner_prep, prob::StationarityProblem, x, p)
    return residual!!(Fx, inner_prep, prob, x, p)
end
@inline function _inner_res_r3!(Fx, inner_prep, prob::AbstractNonlinearProblem, x, p)
    return residual!!(Fx, prob, x, p)
end

@inline function _inner_jac_r3!(Jx, inner_prep, prob::NonlinearProblem, x, p)
    if prob.J !== nothing
        prob.J(Jx, x, p)
        return Jx
    end
    return jacobian!!(Jx, inner_prep, prob, x, p)
end
@inline function _inner_jac_r3!(Jx, inner_prep, prob::AbstractNonlinearProblem, x, p)
    return jacobian!!(Jx, inner_prep, prob, x, p)
end

function _eval_dg_r3!(
        G_buf::AbstractMatrix, prep::ReparametrizedPrep{N},
        reparam::Reparametrized{P, G, DG, C, K}, y::AbstractVector, p) where {
        N, P, G, DG, C, K}
    reparam.∂g(G_buf, y, p)
    return G_buf
end

function _eval_dg_r3!(
        G_buf::AbstractMatrix, prep::ReparametrizedPrep{N},
        reparam::Reparametrized{P, G, Nothing, C, K}, y::AbstractVector, p) where {
        N, P, G, C, K}
    n = length(y)
    y_dual = prep.y_dual
    x_dual = prep.x_dual
    idx = prep.idx
    for j0 in 0:N:(n - 1)
        m = min(N, n - j0)
        y_dual .= _seed_reparam_1.(y_dual, y, idx, j0)
        reparam.g(x_dual, y_dual, p)
        @views G_buf[:, (j0 + 1):(j0 + m)] .= dual_partial.(x_dual, (1:m)')
    end
    return G_buf
end

function residual!!(
        Fy::AbstractVector, prep::ReparametrizedPrep,
        reparam::Reparametrized{P, G, DG, false, K},
        y::AbstractVector, p) where {P, G, DG, K}
    reparam.g(prep.x_buf, y, p)
    _inner_res_r3!(Fy, prep.inner_prep, reparam.prob, prep.x_buf, p)
    return Fy
end

function residual!!(
        Fy::AbstractVector, prep::ReparametrizedPrep,
        reparam::Reparametrized{P, G, DG, true, K},
        y::AbstractVector, p) where {P, G, DG, K}
    reparam.g(prep.x_buf, y, p)
    _inner_res_r3!(prep.Fx_buf, prep.inner_prep, reparam.prob, prep.x_buf, p)
    _eval_dg_r3!(prep.G_buf, prep, reparam, y, p)
    mul!(Fy, transpose(prep.G_buf), prep.Fx_buf)
    return Fy
end

function residual!!(
        Fy::AbstractVector, reparam::Reparametrized, y::AbstractVector, p)
    prep = prepare_ad(AutoForwardDiff(), reparam, Fy, y, p)
    return residual!!(Fy, prep, reparam, y, p)
end

function jacobian!!(
        Jy::AbstractMatrix, prep::ReparametrizedPrep,
        reparam::Reparametrized{P, G, DG, false, K},
        y::AbstractVector, p) where {P, G, DG, K}
    reparam.g(prep.x_buf, y, p)
    _inner_jac_r3!(prep.Jx_buf, prep.inner_prep, reparam.prob, prep.x_buf, p)
    _eval_dg_r3!(prep.G_buf, prep, reparam, y, p)
    mul!(Jy, prep.Jx_buf, prep.G_buf)
    return Jy
end

function jacobian!!(
        Jy::AbstractMatrix, prep::ReparametrizedPrep,
        reparam::Reparametrized{P, G, DG, true, false},
        y::AbstractVector, p) where {P, G, DG}
    reparam.g(prep.x_buf, y, p)
    _inner_jac_r3!(prep.Jx_buf, prep.inner_prep, reparam.prob, prep.x_buf, p)
    _eval_dg_r3!(prep.G_buf, prep, reparam, y, p)
    mul!(prep.W_mat, prep.Jx_buf, prep.G_buf)
    mul!(Jy, transpose(prep.G_buf), prep.W_mat)
    return Jy
end

function _add_curvature_r3!(
        Jy::AbstractMatrix, prep::ReparametrizedPrep{N},
        reparam::Reparametrized{P, G, DG, true, true}, y::AbstractVector, p) where {
        N, P, G, DG}
    n = length(y)
    y_dual = prep.y_dual
    G_dual = prep.G_dual
    W_mat = prep.W_mat
    v_work = prep.v_work
    Fx_buf = prep.Fx_buf
    idx = prep.idx
    for j0 in 0:N:(n - 1)
        m = min(N, n - j0)
        y_dual .= _seed_reparam_1.(y_dual, y, idx, j0)
        reparam.∂g(G_dual, y_dual, p)
        for l in 1:m
            W_mat .= dual_partial.(G_dual, l)
            mul!(v_work, transpose(W_mat), Fx_buf)
            @views Jy[:, j0 + l] .+= v_work
        end
    end
    return Jy
end

function _add_curvature_r3!(
        Jy::AbstractMatrix, prep::ReparametrizedPrep{N},
        reparam::Reparametrized{P, G, Nothing, true, true}, y::AbstractVector, p) where {
        N, P, G}
    n = length(y)
    y_nest1 = prep.y_nest1
    x_nest1 = prep.x_nest1
    W_mat = prep.W_mat
    v_work = prep.v_work
    Fx_buf = prep.Fx_buf
    idx = prep.idx
    for j in 1:n
        for a0 in 0:N:(n - 1)
            ma = min(N, n - a0)
            y_nest1 .= _seed_reparam_nest1.(y_nest1, y, idx, a0, j)
            reparam.g(x_nest1, y_nest1, p)
            @views W_mat[:, (a0 + 1):(a0 + ma)] .= _nested_hess1.(x_nest1, (1:ma)')
        end
        mul!(v_work, transpose(W_mat), Fx_buf)
        @views Jy[:, j] .+= v_work
    end
    return Jy
end

function jacobian!!(
        Jy::AbstractMatrix, prep::ReparametrizedPrep,
        reparam::Reparametrized{P, G, DG, true, true},
        y::AbstractVector, p) where {P, G, DG}
    reparam.g(prep.x_buf, y, p)
    _inner_res_r3!(prep.Fx_buf, prep.inner_prep, reparam.prob, prep.x_buf, p)
    _inner_jac_r3!(prep.Jx_buf, prep.inner_prep, reparam.prob, prep.x_buf, p)
    _eval_dg_r3!(prep.G_buf, prep, reparam, y, p)
    mul!(prep.W_mat, prep.Jx_buf, prep.G_buf)
    mul!(Jy, transpose(prep.G_buf), prep.W_mat)
    _add_curvature_r3!(Jy, prep, reparam, y, p)
    return Jy
end

function jvp!!(
        Jv::AbstractVector, prep::ReparametrizedPrep,
        reparam::Reparametrized, y::AbstractVector, v::AbstractVector, p)
    jacobian!!(prep.J_jvp, prep, reparam, y, p)
    mul!(Jv, prep.J_jvp, v)
    return Jv
end

# --- Internal y-space problem and solver states ---

struct _InnerReparametrized{RP <: Reparametrized, WS} <: AbstractNonlinearProblem
    reparam::RP
    ws::WS
end

@inline function prepare_ad(
        ::AutoForwardDiff, iprob::_InnerReparametrized, r::AbstractArray, y::AbstractArray, p)
    return iprob.ws
end

@inline function prepare_ad(
        ::AbstractADType, iprob::_InnerReparametrized, r::AbstractArray, y::AbstractArray, p)
    return iprob.ws
end

@inline function prepare_ad(
        ::Nothing, iprob::_InnerReparametrized, r::AbstractArray, y::AbstractArray, p)
    return iprob.ws
end

@inline function residual!!(
        ::Nothing, iprob::_InnerReparametrized, y::Union{StaticVector, Number}, p)
    return residual!!(nothing, iprob.reparam, y, p)
end

@inline function residual!!(
        Fy::AbstractVector, iprob::_InnerReparametrized, y::AbstractVector, p)
    return residual!!(Fy, iprob.ws, iprob.reparam, y, p)
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, iprob::_InnerReparametrized,
        y::StaticVector{N, T}, p) where {N, T}
    return _reparam_jac_r1(backend, iprob.reparam, y, p)
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, iprob::_InnerReparametrized,
        y::Number, p)
    return _reparam_jac_r1(backend, iprob.reparam, y, p)
end

@inline function jacobian!!(
        ::Nothing, ::Nothing, iprob::_InnerReparametrized,
        y::Union{StaticVector, Number}, p)
    return _reparam_jac_r1(AutoForwardDiff(), iprob.reparam, y, p)
end

@inline function jacobian!!(
        Jy::AbstractMatrix, prep::ReparametrizedPrep, iprob::_InnerReparametrized,
        y::AbstractVector, p)
    return jacobian!!(Jy, prep, iprob.reparam, y, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, iprob::_InnerReparametrized,
        y::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return jvp!!(nothing, backend, iprob.reparam, y, v, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, iprob::_InnerReparametrized,
        y::T, v::T, p) where {T <: Number}
    return jvp!!(nothing, backend, iprob.reparam, y, v, p)
end

@inline function jvp!!(
        ::Nothing, ::Nothing, iprob::_InnerReparametrized,
        y::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return jvp!!(nothing, nothing, iprob.reparam, y, v, p)
end

@inline function jvp!!(
        ::Nothing, ::Nothing, iprob::_InnerReparametrized,
        y::T, v::T, p) where {T <: Number}
    return jvp!!(nothing, nothing, iprob.reparam, y, v, p)
end

@inline function jvp!!(
        Jv::AbstractVector, prep::ReparametrizedPrep, iprob::_InnerReparametrized,
        y::AbstractVector, v::AbstractVector, p)
    return jvp!!(Jv, prep, iprob.reparam, y, v, p)
end

@inline function init_state(m::Picard, iprob::_InnerReparametrized, y, p)
    Fy = buffer(y)
    jvpprep = prepare_ad(AutoForwardDiff(), iprob, Fy, y, p)
    merit = init_merit(m.globalization, y)
    if Fy === nothing
        return PicardState(jvpprep, nothing, nothing, nothing, merit)
    else
        g = init_globalization(m.globalization, y)
        return PicardState(jvpprep, Fy, buffer(y), g, merit)
    end
end

@inline function step!!(s::PicardState, m::Picard, iprob::_InnerReparametrized, y, Fy, p)
    ω = real(eltype(y))(m.damping)
    d = map!!(fi -> -ω * fi, s.d, Fy)
    g, gl, jvp, mer = s.g, m.globalization, s.jvpprep, s.merit
    y₊, F₊, _, info, merit = globalize_merit!!(
        g, gl, iprob, y, Fy, d, p, jvp, MeasuredSlope(), mer)
    return y₊, F₊, with_merit(s, merit), info
end

struct StaticReparamState{ST}
    inner::ST
end

struct DenseReparamState{ST, WS}
    inner::ST
    ws::WS
end

@inline function _init_reparam_r1(m, reparam::Reparametrized, y₀::StaticOrNumber, p)
    _check_chart_indices(reparam.prob, y₀)
    m_r = Adapt.adapt(ToReal{real(eltype(y₀))}(), m)
    iprob = _InnerReparametrized(reparam, nothing)
    return StaticReparamState(init_state(m_r, iprob, y₀, p))
end

function _init_reparam_r3(m, reparam::Reparametrized, y₀::AbstractVector, p)
    m_r = Adapt.adapt(ToReal{real(eltype(y₀))}(), m)
    ws = prepare_ad(_ad_backend_of(m_r), reparam, y₀, y₀, p)
    iprob = _InnerReparametrized(reparam, ws)
    return DenseReparamState(init_state(m_r, iprob, y₀, p), ws)
end

@inline function init_state(m::Newton, reparam::Reparametrized, y₀::StaticOrNumber, p)
    return _init_reparam_r1(m, reparam, y₀, p)
end
@inline function init_state(m, reparam::Reparametrized, y₀::StaticOrNumber, p)
    return _init_reparam_r1(m, reparam, y₀, p)
end

function init_state(m::Newton, reparam::Reparametrized, y₀::AbstractVector, p)
    return _init_reparam_r3(m, reparam, y₀, p)
end
function init_state(m, reparam::Reparametrized, y₀::AbstractVector, p)
    return _init_reparam_r3(m, reparam, y₀, p)
end

@inline function _solve_static_reparam!(
        s::StaticReparamState, reparam::Reparametrized, y₀, m, p, opt)
    m_r = Adapt.adapt(ToReal{real(eltype(y₀))}(), m)
    iprob = _InnerReparametrized(reparam, nothing)
    y_star, status, inner₊ = solve!!(s.inner, iprob, y₀, m_r, p, opt)
    return y_star, status, StaticReparamState(inner₊)
end

@inline function solve!!(
        s::StaticReparamState, reparam::Reparametrized, y₀, m,
        p = nothing, opt::Options{R} = Options(eltype(y₀))) where {R <: Real}
    y_star, status, state₊ = _solve_static_reparam!(s, reparam, y₀, m, p, opt)
    return reparam.g(y_star, p), status, state₊
end

function solve!!(
        s::DenseReparamState, reparam::Reparametrized, y₀::AbstractVector, method,
        p = nothing, options::Options{R} = Options(eltype(y₀))) where {R <: Real}
    m_r = Adapt.adapt(ToReal{real(eltype(y₀))}(), method)
    ws = s.ws
    iprob = _InnerReparametrized(reparam, ws)
    y_star, status, inner₊ = solve!!(s.inner, iprob, y₀, m_r, p, options)
    reparam.g(ws.x_buf, y_star, p)
    state₊ = inner₊ === s.inner ? s : DenseReparamState(inner₊, ws)
    return ws.x_buf, status, state₊
end

# --- `ReparametrizedSolver`, `init`, `solve!`, `solve`, `solve_batch!` ---

"""
    ReparametrizedSolver(prob, y, x, method, options, state, p)

A preallocated nonlinear solver for a [`Reparametrized`](@ref) problem `prob`, holding both the
chart iterate `y` and the primal coordinate `x = g(y, p)` alongside `method`, [`Options`](@ref)
`options`, the solver workspace `state`, and the parameters `p`.

Build it with [`init(reparam, y₀, method = Newton(), p = nothing; kwargs...)`](@ref init) and
solve with [`solve!(solver)`](@ref solve!) or [`solve!(solver, y, p)`](@ref solve!), which
updates `solver.y` to `y*`, evaluates `x* = g(y*, p)` into `solver.x`, and returns
`(solver.x, status)`.
"""
mutable struct ReparametrizedSolver{T, P <: Reparametrized, M, O <: Options, S,
    Y <: Union{AbstractVector{T}, T}, X <: Union{AbstractVector{T}, T}, Q}
    const prob::P
    const method::M
    const options::O
    const n::Int
    state::S
    y::Y
    x::X
    p::Q
    function ReparametrizedSolver(
            prob::P, y::Y,
            x::X,
            method::M,
            options::O,
            state::S,
            p::Q) where {
            P <: Reparametrized, M, O <: Options, S,
            Y <: Union{AbstractVector, Number}, X <: Union{AbstractVector, Number}, Q}
        return new{eltype(Y), P, M, O, S, Y, X, Q}(
            prob, method, options, length(y), state, y, x, p)
    end
end

function _init_reparametrized(reparam::Reparametrized, y₀, method, p, options, kwargs)
    y, m, opt, state = prepare_solve(reparam, y₀, method, p, options, kwargs)
    if y isa Union{StaticArray, Number}
        x = reparam.g(y, p)
    else
        x = copy(state.ws.x_buf)
    end
    return ReparametrizedSolver(reparam, y, x, m, opt, state, p)
end

function _solve_reparametrized(reparam::Reparametrized, y₀, method, p, options, kwargs)
    y, m, opt, state = prepare_solve(reparam, y₀, method, p, options, kwargs)
    x_star, status, _ = solve!!(state, reparam, y, m, p, opt)
    return x_star, status
end

function _solve_reparametrized_solver!(
        solver::ReparametrizedSolver{T, P, M, O, S, Y, X, Q},
        y = solver.y, p = solver.p) where {T, P, M, O, S, Y, X, Q}
    eltype(y) === T && length(y) == solver.n || throw(ArgumentError(
        "`solve!` needs an iterate like the template given to `init`, of length " *
        "$(solver.n) and element type $T; got length $(length(y)) and " *
        "element type $(eltype(y))"))
    y isa Y || throw(ArgumentError(
        "`solve!` needs an iterate of type $Y, like the one given to `init`; " *
        "got type $(typeof(y))"))
    p isa Q || throw(ArgumentError(
        "`solve!` needs parameters of type $Q, like those given to `init`; " *
        "got type $(typeof(p))"))
    solver.y = y
    solver.p = p
    st = solver.state
    reparam = solver.prob
    if st isa StaticReparamState
        y_star, status, state₊ = _solve_static_reparam!(
            st, reparam, y, solver.method, p, solver.options)
        solver.y = y_star
        solver.x = reparam.g(y_star, p)
        solver.state = state₊
        return solver.x, status
    else
        x_buf, status, state₊ = solve!!(
            st, reparam, y, solver.method, p, solver.options)
        copyto!(solver.x, x_buf)
        solver.state === state₊ || (solver.state = state₊)
        return solver.x, status
    end
end

function _prepare_batch(
        Y::AbstractVector,
        reparam::Reparametrized,
        method,
        options::Union{Nothing, Options},
        kwargs
)
    T = eltype(eltype(Y))
    R = real(T)
    opt = resolve_options(T, options, kwargs)
    m = Adapt.adapt(ToReal{R}(), method)
    state₀ = init_state(m, reparam, zero(eltype(Y)), nothing)
    return m, opt, state₀
end

function solve_batch!(
        Y::AbstractVector{<:Union{SVector, Number}},
        S::AbstractVector{SolverStatus{R}},
        reparam::Reparametrized,
        method = Newton(),
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        workgroupsize::Union{Nothing, Integer} = nothing,
        kwargs...
) where {R <: Real}
    R_Y = real(eltype(eltype(Y)))
    if R !== R_Y
        throw(ArgumentError(
            "SolverStatus real type $R does not match batch real element type $R_Y"))
    end
    if length(S) != length(Y)
        throw(DimensionMismatch(
            "status vector length $(length(S)) does not match batch length $(length(Y))"))
    end
    if get_backend(S) != get_backend(Y)
        throw(ArgumentError(
            "status vector backend $(get_backend(S)) does not match batch backend $(get_backend(Y))"))
    end
    _validate_batch(Y, p, order)
    _validate_batch_problem(reparam)
    m, opt, state₀ = _prepare_batch(Y, reparam, method, options, kwargs)
    isempty(Y) && return S
    return _launch!(Y, S, reparam, m, p, opt, state₀, order,
        _validate_workgroupsize(workgroupsize))
end

function solve_batch!(
        Y::AbstractVector{<:Union{SVector, Number}},
        reparam::Reparametrized,
        method = Newton(),
        p = nothing;
        order::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        options::Union{Nothing, Options} = nothing,
        workgroupsize::Union{Nothing, Integer} = nothing,
        kwargs...
)
    _validate_batch(Y, p, order)
    _validate_batch_problem(reparam)
    m, opt, state₀ = _prepare_batch(Y, reparam, method, options, kwargs)
    isempty(Y) && return _tally_return_codes(ReturnCode[])
    codes = allocate(get_backend(Y), ReturnCode, length(Y))
    return _tally_return_codes(Array(_launch!(
        Y, codes, reparam, m, p, opt, state₀, order,
        _validate_workgroupsize(workgroupsize))))
end

# The chart with the datum `b` of `init` or `solve` set on its inner `StationarityProblem`, as
# `_init_stationarity` stores it. Any other inner problem takes no datum.
function _with_datum(r::Reparametrized, b)
    b === nothing || throw(ArgumentError(
        "keyword `b` is only supported for a `Reparametrized` `StationarityProblem`"))
    return r
end

function _with_datum(
        r::Reparametrized{P, G, DG, C, K}, b) where {P <: StationarityProblem, G, DG, C, K}
    prob = _with_datum(r.prob, b)
    return Reparametrized{typeof(prob), G, DG, C, K}(prob, r.g, r.∂g)
end

# `CommonSolve.init`, `CommonSolve.solve` and `CommonSolve.solve!` on `Reparametrized` and
# `ReparametrizedSolver`, which add methods for types of this package to the functions imported
# in `src/GeometricSolvers.jl`, as `src/nonlinear/stationarity.jl` does.

function init(reparam::Reparametrized, y₀, m = Newton(), p = nothing;
        b = nothing, options = nothing, kw...)
    return _init_reparametrized(_with_datum(reparam, b), y₀, m, p, options, kw)
end

function solve(reparam::Reparametrized, y₀, m = Newton(), p = nothing;
        b = nothing, options = nothing, kw...)
    return _solve_reparametrized(_with_datum(reparam, b), y₀, m, p, options, kw)
end

function solve!(solver::ReparametrizedSolver)
    return _solve_reparametrized_solver!(solver)
end

function solve!(solver::ReparametrizedSolver, y, p = solver.p)
    return _solve_reparametrized_solver!(solver, y, p)
end
