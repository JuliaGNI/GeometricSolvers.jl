# Second-order automatic differentiation for `StationarityProblem`:
# - R1 (`StaticVector` and `Number`): tagged nested `ForwardDiff.Dual` (`StationarityInnerTag`
#   along `E` and `StationarityOuterTag` along `U`), evaluating `(∇S)_E` and `(∇²S)_{E, U}` in
#   one call to `S(x_nested, p)` and the Hessian–vector product `((∇²S)_{E, U}) v` in one call
#   to `S(x_nested1, p)` with a single outer partial (`K_out = 1`).
# - R3 (`Array` and `AbstractGPUArray`): `ChunkedHessianForwardDiff` for `AutoForwardDiff`,
#   using preallocated dual workspaces and device-resident index maps (`e_pos`, `u_pos`) so
#   that `residual!!`, `jacobian!!`, `gradient_hessian!!`, and `jvp!!` allocate 0 bytes on
#   `Array` and perform zero scalar indexing on `AbstractGPUArray`; `DIStationarity` for
#   other `AbstractADType` back ends on CPU `Array`.

struct StationarityInnerTag{S, T} end
struct StationarityOuterTag{S, T} end

@inline function inner_tag_type(
        ::AutoForwardDiff{C, Nothing}, prob::StationarityProblem, ::Type{T}) where {C, T}
    return typeof(ForwardDiff.Tag(StationarityInnerTag{typeof(prob.S), T}(), T))
end
@inline function inner_tag_type(
        backend::AutoForwardDiff, ::StationarityProblem, ::Type{T}) where {T}
    return typeof(ForwardDiff.Tag(StationarityInnerTag{typeof(backend.tag), T}(), T))
end

@inline function outer_tag_type(
        ::AutoForwardDiff{C, Nothing}, prob::StationarityProblem, ::Type{T}) where {C, T}
    return typeof(ForwardDiff.Tag(StationarityOuterTag{typeof(prob.S), T}(), T))
end
@inline function outer_tag_type(
        backend::AutoForwardDiff, ::StationarityProblem, ::Type{T}) where {T}
    return typeof(ForwardDiff.Tag(StationarityOuterTag{typeof(backend.tag), T}(), T))
end

"""
    gradient_hessian!!(F, J, prep, prob::StationarityProblem, x, p)

Evaluate both the stationarity residual `F = (∇S(x, p))_E + b` and its Jacobian
`J = (∇²S(x, p))_{E, U}` at full state `x` and parameters `p`. In R1 (`F === nothing` and
`J === nothing`), one nested `ForwardDiff.Dual` evaluation of `prob.S` returns `(F, J)` in a
single call to `prob.S`. In R3, `F` and `J` are written in place and returned as `(F, J)`.
"""
function gradient_hessian!! end

# --- R1 (`StaticVector` and `Number`) ---

@inline function _stationarity_residual_r1(
        backend::AutoForwardDiff, prob::StationarityProblem,
        x::StaticVector{M, T}, p, b_eff,
        ::Val{n} = _idx_len(prob.E, Val(M))) where {M, T <: Real, n}
    Tg_in = inner_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{Tg_in, T, n}
    xdual = SVector{M, D_in}(ntuple(Val(M)) do i
        pos_e = _idx_pos(prob.E, i)
        p_in = ForwardDiff.Partials{n, T}(
            ntuple(a -> ifelse(pos_e == a, one(T), zero(T)), Val(n)))
        D_in(@inbounds(x[i]), p_in)
    end)
    s_dual = prob.S(xdual, p)
    g_E = SVector{n, T}(ntuple(a -> ForwardDiff.partials(s_dual, a), Val(n)))
    return b_eff === nothing ? g_E : g_E + SVector{n, T}(b_eff)
end

@inline function _stationarity_residual_r1(
        backend::AutoForwardDiff, prob::StationarityProblem,
        x::T, p, b_eff) where {T <: Real}
    Tg_in = inner_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{Tg_in, T, 1}
    s_dual = prob.S(D_in(x, ForwardDiff.Partials{1, T}((one(T),))), p)
    g = ForwardDiff.partials(s_dual, 1)
    return b_eff === nothing ? g : g + T(b_eff)
end

@inline function _stationarity_grad_hess_r1(
        backend::AutoForwardDiff, prob::StationarityProblem,
        x::StaticVector{M, T}, p, b_eff,
        ::Val{n} = _idx_len(prob.E, Val(M))) where {M, T <: Real, n}
    Tg_in = inner_tag_type(backend, prob, T)
    Tg_out = outer_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{Tg_in, T, n}
    D_out = ForwardDiff.Dual{Tg_out, D_in, n}
    x_nested = SVector{M, D_out}(ntuple(Val(M)) do i
        pos_e = _idx_pos(prob.E, i)
        pos_u = _idx_pos(prob.U, i)
        p_in = ForwardDiff.Partials{n, T}(
            ntuple(a -> ifelse(pos_e == a, one(T), zero(T)), Val(n)))
        v_in = D_in(@inbounds(x[i]), p_in)
        p_out = ForwardDiff.Partials{n, D_in}(
            ntuple(b_col -> ifelse(pos_u == b_col, one(D_in), zero(D_in)), Val(n)))
        D_out(v_in, p_out)
    end)
    s_nested = prob.S(x_nested, p)
    s_val = ForwardDiff.value(s_nested)
    g_E = SVector{n, T}(ntuple(a -> ForwardDiff.partials(s_val, a), Val(n)))
    F = b_eff === nothing ? g_E : g_E + SVector{n, T}(b_eff)
    J = SMatrix{n, n, T, n * n}(ntuple(Val(n * n)) do idx
        a = mod1(idx, n)
        b_col = fld1(idx, n)
        outer_p = ForwardDiff.partials(s_nested, b_col)
        ForwardDiff.partials(outer_p, a)
    end)
    return F, J
end

@inline function _stationarity_grad_hess_r1(
        backend::AutoForwardDiff, prob::StationarityProblem,
        x::T, p, b_eff) where {T <: Real}
    Tg_in = inner_tag_type(backend, prob, T)
    Tg_out = outer_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{Tg_in, T, 1}
    D_out = ForwardDiff.Dual{Tg_out, D_in, 1}
    v_in = D_in(x, ForwardDiff.Partials{1, T}((one(T),)))
    x_nested = D_out(v_in, ForwardDiff.Partials{1, D_in}((one(D_in),)))
    s_nested = prob.S(x_nested, p)
    g = ForwardDiff.partials(ForwardDiff.value(s_nested), 1)
    F = b_eff === nothing ? g : g + T(b_eff)
    H = ForwardDiff.partials(ForwardDiff.partials(s_nested, 1), 1)
    return F, H
end

@inline function _stationarity_jvp_r1(
        backend::AutoForwardDiff, prob::StationarityProblem,
        x::StaticVector{M, T}, v::StaticVector{n, T}, p) where {M, n, T <: Real}
    Tg_in = inner_tag_type(backend, prob, T)
    Tg_out = outer_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{Tg_in, T, n}
    D_out1 = ForwardDiff.Dual{Tg_out, D_in, 1}
    x_nested1 = SVector{M, D_out1}(ntuple(Val(M)) do i
        pos_e = _idx_pos(prob.E, i)
        pos_u = _idx_pos(prob.U, i)
        p_in = ForwardDiff.Partials{n, T}(
            ntuple(a -> ifelse(pos_e == a, one(T), zero(T)), Val(n)))
        v_in = D_in(@inbounds(x[i]), p_in)
        vi = pos_u == 0 ? zero(D_in) : D_in(@inbounds(v[pos_u]))
        D_out1(v_in, ForwardDiff.Partials{1, D_in}((vi,)))
    end)
    s_nested1 = prob.S(x_nested1, p)
    outer_p1 = ForwardDiff.partials(s_nested1, 1)
    return SVector{n, T}(ntuple(a -> ForwardDiff.partials(outer_p1, a), Val(n)))
end

@inline function _stationarity_jvp_r1(
        backend::AutoForwardDiff, prob::StationarityProblem,
        x::T, v::T, p) where {T <: Real}
    Tg_in = inner_tag_type(backend, prob, T)
    Tg_out = outer_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{Tg_in, T, 1}
    D_out1 = ForwardDiff.Dual{Tg_out, D_in, 1}
    v_in = D_in(x, ForwardDiff.Partials{1, T}((one(T),)))
    x_nested1 = D_out1(v_in, ForwardDiff.Partials{1, D_in}((D_in(v),)))
    s_nested1 = prob.S(x_nested1, p)
    return ForwardDiff.partials(ForwardDiff.partials(s_nested1, 1), 1)
end

@inline function residual!!(
        ::Nothing, prob::StationarityProblem, x::StaticVector{N, T}, p) where {N, T}
    b_eff = _static_datum(prob, x, _resolve_datum(prob, p, nothing))
    return _stationarity_residual_r1(AutoForwardDiff(), prob, x, p, b_eff)
end

@inline function residual!!(
        ::Nothing, prob::StationarityProblem, x::Number, p)
    b_eff = _static_datum(prob, x, _resolve_datum(prob, p, nothing))
    return _stationarity_residual_r1(AutoForwardDiff(), prob, x, p, b_eff)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, backend::AutoForwardDiff,
        prob::StationarityProblem, x::StaticVector{N, T}, p) where {N, T}
    b_eff = _static_datum(prob, x, _resolve_datum(prob, p, nothing))
    return _stationarity_grad_hess_r1(backend, prob, x, p, b_eff)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, backend::AutoForwardDiff,
        prob::StationarityProblem, x::Number, p)
    b_eff = _static_datum(prob, x, _resolve_datum(prob, p, nothing))
    return _stationarity_grad_hess_r1(backend, prob, x, p, b_eff)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, ::Nothing, backend::AutoForwardDiff,
        prob::StationarityProblem, x::StaticVector{N, T}, p) where {N, T}
    return gradient_hessian!!(nothing, nothing, backend, prob, x, p)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, ::Nothing, backend::AutoForwardDiff,
        prob::StationarityProblem, x::Number, p)
    return gradient_hessian!!(nothing, nothing, backend, prob, x, p)
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff,
        prob::StationarityProblem, x::StaticVector{N, T}, p) where {N, T}
    return gradient_hessian!!(nothing, nothing, backend, prob, x, p)[2]
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff,
        prob::StationarityProblem, x::Number, p)
    return gradient_hessian!!(nothing, nothing, backend, prob, x, p)[2]
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, prob::StationarityProblem,
        x::StaticVector{M, T}, v::StaticVector{N, T}, p) where {M, N, T}
    return _stationarity_jvp_r1(backend, prob, x, v, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, prob::StationarityProblem,
        x::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    return _stationarity_jvp_r1(backend, prob, x, v, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, prob::StationarityProblem,
        x::T, v::T, p) where {T <: Number}
    return _stationarity_jvp_r1(backend, prob, x, v, p)
end

@inline function residual!!(
        ::Nothing, rprob::_ReducedStationarityProblem,
        u::StaticVector{N, T}, p) where {N, T}
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_residual_r1(AutoForwardDiff(), rprob.prob, x, p, rprob.b)
end

@inline function residual!!(
        ::Nothing, rprob::_ReducedStationarityProblem,
        u::Number, p)
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_residual_r1(AutoForwardDiff(), rprob.prob, x, p, rprob.b)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, backend::AutoForwardDiff,
        rprob::_ReducedStationarityProblem, u::StaticVector{N, T}, p) where {N, T}
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_grad_hess_r1(backend, rprob.prob, x, p, rprob.b)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, backend::AutoForwardDiff,
        rprob::_ReducedStationarityProblem, u::Number, p)
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_grad_hess_r1(backend, rprob.prob, x, p, rprob.b)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, ::Nothing, backend::AutoForwardDiff,
        rprob::_ReducedStationarityProblem, u::StaticVector{N, T}, p) where {N, T}
    return gradient_hessian!!(nothing, nothing, backend, rprob, u, p)
end

@inline function gradient_hessian!!(
        ::Nothing, ::Nothing, ::Nothing, backend::AutoForwardDiff,
        rprob::_ReducedStationarityProblem, u::Number, p)
    return gradient_hessian!!(nothing, nothing, backend, rprob, u, p)
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, rprob::_ReducedStationarityProblem,
        u::StaticVector{N, T}, p) where {N, T}
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_grad_hess_r1(backend, rprob.prob, x, p, rprob.b)[2]
end

@inline function jacobian!!(
        ::Nothing, backend::AutoForwardDiff, rprob::_ReducedStationarityProblem,
        u::Number, p)
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_grad_hess_r1(backend, rprob.prob, x, p, rprob.b)[2]
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, rprob::_ReducedStationarityProblem,
        u::StaticVector{N, T}, v::StaticVector{N, T}, p) where {N, T}
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_jvp_r1(backend, rprob.prob, x, v, p)
end

@inline function jvp!!(
        ::Nothing, backend::AutoForwardDiff, rprob::_ReducedStationarityProblem,
        u::T, v::T, p) where {T <: Number}
    x = _embed_unknowns(rprob.prob.U, rprob.x_ref, u)
    return _stationarity_jvp_r1(backend, rprob.prob, x, v, p)
end

# --- R3 (`ChunkedHessianForwardDiff` and `DIStationarity`) ---

"""
    ChunkedHessianForwardDiff{N, TgIn, TgOut}

The R3 AD preparation of `AutoForwardDiff` for [`StationarityProblem`](@ref): chunked nested
`ForwardDiff.Dual` workspaces and device-resident index maps `e_pos` and `u_pos` so that
[`residual!!`](@ref), [`jacobian!!`](@ref), [`gradient_hessian!!`](@ref), and [`jvp!!`](@ref)
allocate zero bytes on CPU `Array` and perform zero scalar indexing on `AbstractGPUArray`.
"""
struct ChunkedHessianForwardDiff{N, TgIn, TgOut, DX, DN, DN1, VX, IP, VC, MC}
    x_in::DX
    x_nest::DN
    x_nest1::DN1
    v_full::VX
    e_pos::IP
    u_pos::IP
    u_idx_cpu::Vector{Int}
    f_cpu::VC
    j_cpu::MC
    sym::Bool
end

function _build_pos_maps(prob::StationarityProblem, M::Int, n::Int)
    e_cpu = prob.E isa Colon ? collect(1:M) : Int[Int(i) for i in prob.E]
    u_cpu = prob.U isa Colon ? collect(1:M) : Int[Int(i) for i in prob.U]
    e_pos_cpu = zeros(Int, M)
    u_pos_cpu = zeros(Int, M)
    for a in 1:n
        e_pos_cpu[e_cpu[a]] = a
        u_pos_cpu[u_cpu[a]] = a
    end
    return e_pos_cpu, u_pos_cpu, e_cpu, u_cpu
end

function prepare_ad(
        backend::AutoForwardDiff, prob::StationarityProblem,
        r::AbstractArray, x::AbstractArray, p)
    check_iterate_shape(x)
    M = length(x)
    n = length(r)
    T = eltype(x)
    N = chunk_size(backend, n)
    TgIn = inner_tag_type(backend, prob, T)
    TgOut = outer_tag_type(backend, prob, T)
    D_in = ForwardDiff.Dual{TgIn, T, N}
    D_nest = ForwardDiff.Dual{TgOut, D_in, N}
    D_nest1 = ForwardDiff.Dual{TgOut, D_in, 1}
    x_in = similar(x, D_in, M)
    x_nest = similar(x, D_nest, M)
    x_nest1 = similar(x, D_nest1, M)
    v_full = similar(x, T, M)
    e_pos_cpu, u_pos_cpu, e_cpu, u_cpu = _build_pos_maps(prob, M, n)
    e_pos = similar(x, Int, M)
    copyto!(e_pos, e_pos_cpu)
    u_pos = similar(x, Int, M)
    copyto!(u_pos, u_pos_cpu)
    f_cpu = Vector{T}(undef, n)
    j_cpu = Matrix{T}(undef, n, n)
    sym = e_cpu == u_cpu
    return ChunkedHessianForwardDiff{
        N, TgIn, TgOut, typeof(x_in), typeof(x_nest), typeof(x_nest1),
        typeof(v_full), typeof(e_pos), typeof(f_cpu), typeof(j_cpu)}(
        x_in, x_nest, x_nest1, v_full, e_pos, u_pos, u_cpu, f_cpu, j_cpu, sym)
end

@inline function _seed_in(
        ::ForwardDiff.Dual{TgIn, T, N}, xi::T, pos_e::Int, a0::Int
) where {TgIn, T, N}
    p_in = ForwardDiff.Partials{N, T}(
        ntuple(k -> ifelse(pos_e == a0 + k, one(T), zero(T)), Val(N)))
    return ForwardDiff.Dual{TgIn, T, N}(xi, p_in)
end

@inline function _seed_nest(
        ::ForwardDiff.Dual{TgOut, ForwardDiff.Dual{TgIn, T, N}, N},
        xi::T, pos_e::Int, pos_u::Int, a0::Int, b0::Int
) where {TgOut, TgIn, T, N}
    D_in = ForwardDiff.Dual{TgIn, T, N}
    p_in = ForwardDiff.Partials{N, T}(
        ntuple(k -> ifelse(pos_e == a0 + k, one(T), zero(T)), Val(N)))
    v_in = D_in(xi, p_in)
    p_out = ForwardDiff.Partials{N, D_in}(
        ntuple(l -> ifelse(pos_u == b0 + l, one(D_in), zero(D_in)), Val(N)))
    return ForwardDiff.Dual{TgOut, D_in, N}(v_in, p_out)
end

@inline function _seed_nest1(
        ::ForwardDiff.Dual{TgOut, ForwardDiff.Dual{TgIn, T, N}, 1},
        xi::T, vi::T, pos_e::Int, a0::Int
) where {TgOut, TgIn, T, N}
    D_in = ForwardDiff.Dual{TgIn, T, N}
    p_in = ForwardDiff.Partials{N, T}(
        ntuple(k -> ifelse(pos_e == a0 + k, one(T), zero(T)), Val(N)))
    v_in = D_in(xi, p_in)
    p_out = ForwardDiff.Partials{1, D_in}((D_in(vi),))
    return ForwardDiff.Dual{TgOut, D_in, 1}(v_in, p_out)
end

@kernel function _scatter_v_kernel!(v_full, @Const(v), @Const(u_pos))
    i = @index(Global, Linear)
    @inbounds begin
        pos = u_pos[i]
        v_full[i] = pos == 0 ? zero(eltype(v_full)) : v[pos]
    end
end

@inline _add_datum_r3!(F::AbstractVector, ::Nothing) = F
@inline function _add_datum_r3!(F::AbstractVector{T}, b::Number) where {T}
    tb = T(b)
    F .+= tb
    return F
end
@inline function _add_datum_r3!(F::AbstractVector, b::AbstractVector)
    F .+= b
    return F
end

function _stationarity_residual_r3!(
        F::AbstractVector{T}, prep::ChunkedHessianForwardDiff{N},
        prob::StationarityProblem, x::AbstractVector{T}, p, b_eff) where {N, T}
    n = length(F)
    x_in = prep.x_in
    e_pos = prep.e_pos
    f_dest = F isa AbstractGPUArray ? prep.f_cpu : F
    for a0 in 0:N:(n - 1)
        ma = min(N, n - a0)
        x_in .= _seed_in.(x_in, x, e_pos, a0)
        s_dual = prob.S(x_in, p)
        @inbounds for k in 1:ma
            f_dest[a0 + k] = ForwardDiff.partials(s_dual, k)
        end
    end
    F isa AbstractGPUArray && copyto!(F, f_dest)
    _add_datum_r3!(F, b_eff)
    return F
end

function residual!!(
        F::AbstractVector, prep::ChunkedHessianForwardDiff,
        prob::StationarityProblem, x::AbstractVector, p)
    b_eff = _resolve_datum(prob, p, nothing)
    return _stationarity_residual_r3!(F, prep, prob, x, p, b_eff)
end

function jacobian!!(
        J::AbstractMatrix{T}, prep::ChunkedHessianForwardDiff{N},
        prob::StationarityProblem, x::AbstractVector{T}, p) where {N, T}
    n = size(J, 1)
    x_nest = prep.x_nest
    e_pos = prep.e_pos
    u_pos = prep.u_pos
    j_dest = J isa AbstractGPUArray ? prep.j_cpu : J
    sym = prep.sym
    for b0 in 0:N:(n - 1)
        mb = min(N, n - b0)
        a_start = sym ? b0 : 0
        for a0 in a_start:N:(n - 1)
            ma = min(N, n - a0)
            x_nest .= _seed_nest.(x_nest, x, e_pos, u_pos, a0, b0)
            s_nest = prob.S(x_nest, p)
            @inbounds for l in 1:mb
                outer_l = ForwardDiff.partials(s_nest, l)
                for k in 1:ma
                    j_dest[a0 + k, b0 + l] = ForwardDiff.partials(outer_l, k)
                end
            end
            if sym && a0 > b0
                @inbounds for l in 1:mb
                    for k in 1:ma
                        j_dest[b0 + l, a0 + k] = j_dest[a0 + k, b0 + l]
                    end
                end
            end
        end
    end
    J isa AbstractGPUArray && copyto!(J, j_dest)
    return J
end

function _stationarity_grad_hess_r3!(
        F::AbstractVector{T}, J::AbstractMatrix{T}, prep::ChunkedHessianForwardDiff{N},
        prob::StationarityProblem, x::AbstractVector{T}, p, b_eff) where {N, T}
    n = length(F)
    x_nest = prep.x_nest
    e_pos = prep.e_pos
    u_pos = prep.u_pos
    f_dest = F isa AbstractGPUArray ? prep.f_cpu : F
    j_dest = J isa AbstractGPUArray ? prep.j_cpu : J
    sym = prep.sym
    for b0 in 0:N:(n - 1)
        mb = min(N, n - b0)
        a_start = sym ? b0 : 0
        for a0 in a_start:N:(n - 1)
            ma = min(N, n - a0)
            x_nest .= _seed_nest.(x_nest, x, e_pos, u_pos, a0, b0)
            s_nest = prob.S(x_nest, p)
            if b0 == 0
                s_val = ForwardDiff.value(s_nest)
                @inbounds for k in 1:ma
                    f_dest[a0 + k] = ForwardDiff.partials(s_val, k)
                end
            end
            @inbounds for l in 1:mb
                s_out_l = ForwardDiff.partials(s_nest, l)
                for k in 1:ma
                    j_dest[a0 + k, b0 + l] = ForwardDiff.partials(s_out_l, k)
                end
            end
            if sym && a0 > b0
                @inbounds for l in 1:mb
                    for k in 1:ma
                        j_dest[b0 + l, a0 + k] = j_dest[a0 + k, b0 + l]
                    end
                end
            end
        end
    end
    F isa AbstractGPUArray && copyto!(F, f_dest)
    J isa AbstractGPUArray && copyto!(J, j_dest)
    _add_datum_r3!(F, b_eff)
    return F, J
end

function gradient_hessian!!(
        F::AbstractVector{T}, J::AbstractMatrix{T}, prep::ChunkedHessianForwardDiff{N},
        prob::StationarityProblem, x::AbstractVector{T}, p) where {N, T}
    return _stationarity_grad_hess_r3!(
        F, J, prep, prob, x, p, _resolve_datum(prob, p, nothing))
end

function gradient_hessian!!(
        F::AbstractVector{T}, J::AbstractMatrix{T}, prep::ChunkedHessianForwardDiff{N},
        ::AbstractADType, prob::StationarityProblem, x::AbstractVector{T}, p) where {N, T}
    return gradient_hessian!!(F, J, prep, prob, x, p)
end

function jvp!!(
        Jv::AbstractVector{T}, prep::ChunkedHessianForwardDiff{N},
        prob::StationarityProblem, x::AbstractVector{T},
        v::AbstractVector{T}, p) where {N, T}
    n = length(Jv)
    v_full = prep.v_full
    if prob.U isa Colon
        copyto!(v_full, v)
    elseif x isa AbstractGPUArray
        backend = get_backend(x)
        _scatter_v_kernel!(backend)(v_full, v, prep.u_pos; ndrange = length(x))
        isgpu(backend) && synchronize(backend)
    else
        fill!(v_full, zero(T))
        u_idx_cpu = prep.u_idx_cpu
        @inbounds for b_col in 1:n
            v_full[u_idx_cpu[b_col]] = v[b_col]
        end
    end
    x_nest1 = prep.x_nest1
    e_pos = prep.e_pos
    jv_dest = Jv isa AbstractGPUArray ? prep.f_cpu : Jv
    for a0 in 0:N:(n - 1)
        ma = min(N, n - a0)
        x_nest1 .= _seed_nest1.(x_nest1, x, v_full, e_pos, a0)
        s_nest1 = prob.S(x_nest1, p)
        outer_1 = ForwardDiff.partials(s_nest1, 1)
        @inbounds for k in 1:ma
            jv_dest[a0 + k] = ForwardDiff.partials(outer_1, k)
        end
    end
    Jv isa AbstractGPUArray && copyto!(Jv, jv_dest)
    return Jv
end

"""
    DIStationarity

The R3 AD preparation for [`StationarityProblem`](@ref) with any `AbstractADType` back end
other than `AutoForwardDiff`, on a CPU `Array`: the gradient of `S` gives the residual and the
Hessian of `S` gives the Jacobian, whose `(E, U)` block that Jacobian is.

The Jacobian and the Hessian-vector product are the `(E, U)` block of one Hessian of `S`, not the
derivative of a gradient map: a back end of finite differences would differentiate an already
finite-difference gradient, whose noise then swamps the outer difference. The Hessian is taken over
the whole state, so a state far longer than the unknown set costs more here than under
`AutoForwardDiff()`, which sweeps `E × U` only.
"""
struct DIStationarity{B <: AbstractADType, G, GP, HP, H}
    backend::B
    grad_full::G
    gradprep::GP
    hessprep::HP
    hess_full::H
    e_idx::Vector{Int}
    u_idx::Vector{Int}
end

function prepare_ad(
        backend::AbstractADType, prob::StationarityProblem,
        r::AbstractArray, x::AbstractArray, p)
    x isa AbstractGPUArray && throw(ArgumentError(
        "a device array iterate has an R3 stationarity derivative through " *
        "AutoForwardDiff() only: DifferentiationInterface raises a scalar-indexing " *
        "error on an MtlArray, a CuArray and a JLArray. Solve on an Array to use " *
        "another back end."))
    check_iterate_shape(x)
    M = length(x)
    n = length(r)
    _, _, e_cpu, u_cpu = _build_pos_maps(prob, M, n)
    grad_full = similar(x)
    gradprep = DI.prepare_gradient(prob.S, backend, x, DI.Constant(p))
    hess_full = similar(x, eltype(x), M, M)
    hessprep = DI.prepare_hessian(prob.S, backend, x, DI.Constant(p))
    return DIStationarity(backend, grad_full, gradprep, hessprep, hess_full, e_cpu, u_cpu)
end

function _stationarity_residual_r3!(
        F::AbstractVector, prep::DIStationarity,
        prob::StationarityProblem, x::AbstractVector, p, b_eff)
    DI.gradient!(prob.S, prep.grad_full, prep.gradprep, prep.backend, x, DI.Constant(p))
    e_idx = prep.e_idx
    @inbounds for a in eachindex(F, e_idx)
        F[a] = prep.grad_full[e_idx[a]]
    end
    _add_datum_r3!(F, b_eff)
    return F
end

function residual!!(
        F::AbstractVector, prep::DIStationarity,
        prob::StationarityProblem, x::AbstractVector, p)
    b_eff = _resolve_datum(prob, p, nothing)
    return _stationarity_residual_r3!(F, prep, prob, x, p, b_eff)
end

function _stationarity_hessian_r3!(
        prep::DIStationarity, prob::StationarityProblem, x::AbstractVector, p)
    DI.hessian!(prob.S, prep.hess_full, prep.hessprep, prep.backend, x, DI.Constant(p))
    return prep.hess_full
end

function jacobian!!(
        J::AbstractMatrix, prep::DIStationarity,
        prob::StationarityProblem, x::AbstractVector, p)
    isempty(J) && return J
    H = _stationarity_hessian_r3!(prep, prob, x, p)
    e_idx, u_idx = prep.e_idx, prep.u_idx
    @inbounds for c in eachindex(u_idx)
        for a in eachindex(e_idx)
            J[a, c] = H[e_idx[a], u_idx[c]]
        end
    end
    return J
end

function _stationarity_grad_hess_r3!(
        F::AbstractVector, J::AbstractMatrix, prep::DIStationarity,
        prob::StationarityProblem, x::AbstractVector, p, b_eff)
    _stationarity_residual_r3!(F, prep, prob, x, p, b_eff)
    jacobian!!(J, prep, prob, x, p)
    return F, J
end

function gradient_hessian!!(
        F::AbstractVector, J::AbstractMatrix, prep::DIStationarity,
        prob::StationarityProblem, x::AbstractVector, p)
    return _stationarity_grad_hess_r3!(
        F, J, prep, prob, x, p, _resolve_datum(prob, p, nothing))
end

function gradient_hessian!!(
        F::AbstractVector, J::AbstractMatrix, prep::DIStationarity,
        ::AbstractADType, prob::StationarityProblem, x::AbstractVector, p)
    return gradient_hessian!!(F, J, prep, prob, x, p)
end

function jvp!!(
        Jv::AbstractVector, prep::DIStationarity,
        prob::StationarityProblem, x::AbstractVector, v::AbstractVector, p)
    H = _stationarity_hessian_r3!(prep, prob, x, p)
    e_idx, u_idx = prep.e_idx, prep.u_idx
    T = eltype(Jv)
    @inbounds for a in eachindex(e_idx)
        acc = zero(T)
        for c in eachindex(u_idx)
            acc += H[e_idx[a], u_idx[c]] * v[c]
        end
        Jv[a] = acc
    end
    return Jv
end

# --- R3 `_ReducedStationarityProblem` forwarding ---

function prepare_ad(
        backend::AutoForwardDiff, rprob::_ReducedStationarityProblem,
        r::AbstractArray, u::AbstractArray, p)
    rprob.prep !== nothing && return rprob.prep
    return prepare_ad(backend, rprob.prob, r, rprob.x_ref, p)
end

function prepare_ad(
        backend::AbstractADType, rprob::_ReducedStationarityProblem,
        r::AbstractArray, u::AbstractArray, p)
    rprob.prep !== nothing && return rprob.prep
    return prepare_ad(backend, rprob.prob, r, rprob.x_ref, p)
end

function residual!!(
        F::AbstractArray, rprob::_ReducedStationarityProblem, u::AbstractArray, p)
    _embed_u!(rprob.x_ref, rprob.x_ref, u, rprob.u_pos)
    b_eff = rprob.has_b === nothing ? rprob.b : (rprob.has_b[] ? rprob.b : nothing)
    return _stationarity_residual_r3!(F, rprob.prep, rprob.prob, rprob.x_ref, p, b_eff)
end

function jacobian!!(
        J::AbstractMatrix, prep::Union{ChunkedHessianForwardDiff, DIStationarity},
        rprob::_ReducedStationarityProblem, u::AbstractVector, p)
    _embed_u!(rprob.x_ref, rprob.x_ref, u, rprob.u_pos)
    return jacobian!!(J, prep, rprob.prob, rprob.x_ref, p)
end

function gradient_hessian!!(
        F::AbstractVector, J::AbstractMatrix,
        prep::Union{ChunkedHessianForwardDiff, DIStationarity},
        rprob::_ReducedStationarityProblem, u::AbstractVector, p)
    _embed_u!(rprob.x_ref, rprob.x_ref, u, rprob.u_pos)
    b_eff = rprob.has_b === nothing ? rprob.b : (rprob.has_b[] ? rprob.b : nothing)
    return _stationarity_grad_hess_r3!(F, J, prep, rprob.prob, rprob.x_ref, p, b_eff)
end

function gradient_hessian!!(
        F::AbstractVector, J::AbstractMatrix,
        prep::Union{ChunkedHessianForwardDiff, DIStationarity},
        ::AbstractADType, rprob::_ReducedStationarityProblem, u::AbstractVector, p)
    return gradient_hessian!!(F, J, prep, rprob, u, p)
end

function jvp!!(
        Jv::AbstractVector, prep::Union{ChunkedHessianForwardDiff, DIStationarity},
        rprob::_ReducedStationarityProblem, u::AbstractVector, v::AbstractVector, p)
    _embed_u!(rprob.x_ref, rprob.x_ref, u, rprob.u_pos)
    return jvp!!(Jv, prep, rprob.prob, rprob.x_ref, v, p)
end
