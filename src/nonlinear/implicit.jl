# Implicit differentiation of `solve` for `ForwardDiff.Dual` inputs.
#
# When `x₀` or `p` carries a `ForwardDiff.Dual` (or `Complex{<:ForwardDiff.Dual}`) leaf, `solve`
# strips duals via `Adapt.adapt(StripDual(), ...)`, runs the primal nonlinear solve to `(x*, status)`,
# and — on `status.code === SUCCESS` — solves one unregularised linear system per partial at `x*`,
#
#     J(x*, p) ẋ_k = -∂_k F(x*, p),
#
# from a single factorisation of `J(x*, p)` and one dual residual evaluation `F(x̂, p)` with zero
# partials on `x̂`.

function _collect_dual_types!(duals::Vector{DataType}, ::Type{T}, seen::IdSet{Any}) where {T}
    (T in seen || T === Any || T === Union{} || T isa TypeVar) && return nothing
    push!(seen, T)
    if T isa Union
        _collect_dual_types!(duals, T.a, seen)
        _collect_dual_types!(duals, T.b, seen)
        return nothing
    elseif T isa UnionAll
        _collect_dual_types!(duals, Base.unwrap_unionall(T), seen)
        return nothing
    elseif T <: ForwardDiff.Dual
        T isa DataType && push!(duals, T)
        return nothing
    elseif T <: Complex
        if T isa DataType && !isempty(T.parameters) && T.parameters[1] isa Type
            _collect_dual_types!(duals, T.parameters[1], seen)
        end
        return nothing
    elseif T <: AbstractArray
        E = eltype(T)
        E isa Type && E !== T && _collect_dual_types!(duals, E, seen)
        return nothing
    elseif T <: Tuple
        if T isa DataType
            for P in T.parameters
                U = Base.unwrap_unionall(P)
                if U isa DataType && U.name === Core.TypeofVararg.name
                    !isempty(U.parameters) && U.parameters[1] isa Type &&
                        _collect_dual_types!(duals, U.parameters[1], seen)
                elseif P isa Type
                    _collect_dual_types!(duals, P, seen)
                end
            end
        end
        return nothing
    elseif T <: NamedTuple
        if T isa DataType && length(T.parameters) >= 2 && T.parameters[2] isa Type
            _collect_dual_types!(duals, T.parameters[2], seen)
        end
        return nothing
    elseif T isa DataType && isstructtype(T) && !isabstracttype(T) &&
           !(T <: Type) && !(T <: Core.Builtin) && !(T <: Symbol) && !(T <: AbstractString)
        for i in 1:fieldcount(T)
            ft = fieldtype(T, i)
            ft isa Type && _collect_dual_types!(duals, ft, seen)
        end
    end
    return nothing
end

function _has_dual_in_type(::Type{T}) where {T}
    duals = DataType[]
    _collect_dual_types!(duals, T, IdSet{Any}())
    return !isempty(duals)
end

@generated function has_dual_type(::Type{T}) where {T}
    return _has_dual_in_type(T) ? :(true) : :(false)
end

@generated function validate_dual_types(::Type{X}, ::Type{P}) where {X, P}
    duals = DataType[]
    seen = IdSet{Any}()
    _collect_dual_types!(duals, X, seen)
    _collect_dual_types!(duals, P, seen)
    isempty(duals) &&
        return :(throw(ArgumentError("expected ForwardDiff.Dual in x₀ or p")))
    for D in duals
        if length(D.parameters) < 3 || !(D.parameters[2] isa Type) ||
           !(D.parameters[3] isa Int) || _has_dual_in_type(D.parameters[2])
            return :(throw(ArgumentError(
                "solve does not support nested ForwardDiff.Dual inputs")))
        end
    end
    D1 = duals[1]
    Tg = D1.parameters[1]
    N = D1.parameters[3]
    for i in 2:length(duals)
        Di = duals[i]
        if Di.parameters[1] !== Tg || Di.parameters[3] != N
            return :(throw(ArgumentError(
                "solve requires a single ForwardDiff.Dual tag and partial count across x₀ and p")))
        end
    end
    V = reduce(promote_type, (D.parameters[2] for D in duals))
    D_out = ForwardDiff.Dual{Tg, V, N}
    return :($D_out)
end

@inline dual_partial(x::Real, ::Integer) = zero(x)
@inline dual_partial(z::Complex{<:Real}, ::Integer) = zero(z)

struct ZeroDual{D} end

@inline function (::ZeroDual{ForwardDiff.Dual{Tg, V, N}})(x::Real) where {Tg, V, N}
    z = zero(V)
    return ForwardDiff.Dual{Tg, V, N}(
        convert(V, x), ForwardDiff.Partials{N, V}(ntuple(_ -> z, Val(N))))
end

@inline function (zd::ZeroDual{D})(z::Complex) where {D}
    return Complex(zd(real(z)), zd(imag(z)))
end

struct NaNDual{D} end

@inline function (::NaNDual{ForwardDiff.Dual{Tg, V, N}})(x::Real) where {Tg, V, N}
    nan = V(NaN)
    return ForwardDiff.Dual{Tg, V, N}(
        convert(V, x), ForwardDiff.Partials{N, V}(ntuple(_ -> nan, Val(N))))
end

@inline function (nd::NaNDual{D})(z::Complex) where {D}
    return Complex(nd(real(z)), nd(imag(z)))
end

struct ExtractPartial{T}
    k::Int
end

@inline function (ep::ExtractPartial{T})(r) where {T}
    return convert(T, dual_partial(r, ep.k))
end

struct SetPartial{D}
    k::Int
end

@inline function (sp::SetPartial{ForwardDiff.Dual{Tg, V, N}})(
        xd::ForwardDiff.Dual{Tg, V, N}, dk::Real) where {Tg, V, N}
    val = ForwardDiff.value(xd)
    old = ForwardDiff.partials(xd).values
    vdk = convert(V, -dk)
    parts = ntuple(j -> ifelse(j == sp.k, vdk, old[j]), Val(N))
    return ForwardDiff.Dual{Tg, V, N}(val, ForwardDiff.Partials{N, V}(parts))
end

@inline function (sp::SetPartial{D})(
        zd::Complex{D}, dk::Complex) where {D <: ForwardDiff.Dual}
    return Complex(sp(real(zd), real(dk)), sp(imag(zd), imag(dk)))
end

@inline _seed_zero_duals(
    ::Type{ForwardDiff.Dual{Tg, V, N}}, x_star::Number
) where {Tg, V, N} = ZeroDual{ForwardDiff.Dual{Tg, V, N}}()(x_star)

@inline _seed_zero_duals(
    ::Type{ForwardDiff.Dual{Tg, V, N}}, x_star::StaticVector{M, T}
) where {Tg, V, N, M, T <: Number} = map(ZeroDual{ForwardDiff.Dual{Tg, V, N}}(), x_star)

function _seed_zero_duals(
        ::Type{ForwardDiff.Dual{Tg, V, N}}, x_star::AbstractVector{T}
) where {Tg, V, N, T <: Number}
    D = ForwardDiff.Dual{Tg, V, N}
    ET = T <: Complex ? Complex{D} : D
    out = similar(x_star, ET)
    out .= ZeroDual{D}().(x_star)
    return out
end

@inline _make_nan_duals(
    ::Type{ForwardDiff.Dual{Tg, V, N}}, x_star::Number
) where {Tg, V, N} = NaNDual{ForwardDiff.Dual{Tg, V, N}}()(x_star)

@inline _make_nan_duals(
    ::Type{ForwardDiff.Dual{Tg, V, N}}, x_star::StaticVector{M, T}
) where {Tg, V, N, M, T <: Number} = map(NaNDual{ForwardDiff.Dual{Tg, V, N}}(), x_star)

function _make_nan_duals(
        ::Type{ForwardDiff.Dual{Tg, V, N}}, x_star::AbstractVector{T}
) where {Tg, V, N, T <: Number}
    D = ForwardDiff.Dual{Tg, V, N}
    ET = T <: Complex ? Complex{D} : D
    out = similar(x_star, ET)
    out .= NaNDual{D}().(x_star)
    return out
end

@inline function _working_dual_type(
        ::Type{ForwardDiff.Dual{Tg, V, N}}, ::Type{T}) where {Tg, V, N, T <: Number}
    W = promote_type(real(T), V)
    return ForwardDiff.Dual{Tg, W, N}
end

@inline function _singular_status(st::SolverStatus{R}) where {R}
    return SolverStatus{R}(
        SINGULAR, st.iterations, st.fnorm, st.stepnorm, st.step_failures,
        st.promoted, st.rank, st.nullnorm, st.gradratio)
end

@inline function _tangent_jacprep(state::NewtonState, prob::NonlinearProblem, x_star, p_val)
    prob.J !== nothing && return nothing
    state.jacprep isa JacobianFunction &&
        return prepare_ad(AutoForwardDiff(), prob, state.Fx, x_star, p_val)
    return state.jacprep
end

@inline function _tangent_jacprep(state::PicardState, prob::NonlinearProblem, x_star, p_val)
    prob.J !== nothing && return nothing
    return prepare_ad(AutoForwardDiff(), prob, state.Fx, x_star, p_val)
end

@inline function _static_rhs_k(::Type{T}, r_dual::StaticVector{S}, k::Int) where {T, S}
    return SVector{S, T}(ntuple(i -> convert(T, dual_partial(r_dual[i], k)), Val(S)))
end

@inline function _solve_static_tangents(
        ::Val{N}, ::Type{T}, fac, r_dual::StaticVector) where {N, T}
    return ntuple(
        k -> -ldiv!!(nothing, fac, _static_rhs_k(T, r_dual, k)),
        Val(N))
end

@inline function _assemble_static_duals(
        ::Type{ForwardDiff.Dual{Tg, V, N}},
        x_star::StaticVector{S, T},
        tangents::NTuple{N, <:StaticVector{S}}
) where {Tg, V, N, S, T <: Real}
    D = ForwardDiff.Dual{Tg, V, N}
    return SVector{S, D}(ntuple(Val(S)) do i
        parts = ntuple(k -> convert(V, tangents[k][i]), Val(N))
        D(convert(V, x_star[i]), ForwardDiff.Partials{N, V}(parts))
    end)
end

@inline function _assemble_static_duals(
        ::Type{ForwardDiff.Dual{Tg, V, N}},
        x_star::StaticVector{S, T},
        tangents::NTuple{N, <:StaticVector{S}}
) where {Tg, V, N, S, T <: Complex}
    D = ForwardDiff.Dual{Tg, V, N}
    return SVector{S, Complex{D}}(ntuple(Val(S)) do i
        rparts = ntuple(k -> convert(V, real(tangents[k][i])), Val(N))
        iparts = ntuple(k -> convert(V, imag(tangents[k][i])), Val(N))
        Complex(
            D(convert(V, real(x_star[i])), ForwardDiff.Partials{N, V}(rparts)),
            D(convert(V, imag(x_star[i])), ForwardDiff.Partials{N, V}(iparts))
        )
    end)
end

@inline function _assemble_scalar_dual(
        ::Type{ForwardDiff.Dual{Tg, V, N}},
        x_star::T,
        fac,
        r_dual
) where {Tg, V, N, T <: Real}
    D = ForwardDiff.Dual{Tg, V, N}
    parts = ntuple(
        k -> convert(V, -ldiv!!(nothing, fac, convert(T, dual_partial(r_dual, k)))),
        Val(N))
    return D(convert(V, x_star), ForwardDiff.Partials{N, V}(parts))
end

@inline function _assemble_scalar_dual(
        ::Type{ForwardDiff.Dual{Tg, V, N}},
        x_star::T,
        fac,
        r_dual
) where {Tg, V, N, T <: Complex}
    D = ForwardDiff.Dual{Tg, V, N}
    ds = ntuple(
        k -> -ldiv!!(nothing, fac, convert(T, dual_partial(r_dual, k))),
        Val(N))
    rparts = ntuple(k -> convert(V, real(ds[k])), Val(N))
    iparts = ntuple(k -> convert(V, imag(ds[k])), Val(N))
    return Complex(
        D(convert(V, real(x_star)), ForwardDiff.Partials{N, V}(rparts)),
        D(convert(V, imag(x_star)), ForwardDiff.Partials{N, V}(iparts)))
end

@inline function _implicit_tangent_r1(
        ::Type{D}, state, prob::NonlinearProblem,
        x_star::Union{StaticVector, Number}, m,
        p, p_val, status::SolverStatus
) where {Tg, V, N, D <: ForwardDiff.Dual{Tg, V, N}}
    linsolve = m isa Newton ? m.linsolve : LUFactorization()
    fac_slot = m isa Newton ? state.fac : nothing
    jacprep = _tangent_jacprep(state, prob, x_star, p_val)
    Jmat = jacobian!!(nothing, jacprep, prob, x_star, p_val)
    fac = factorize!!(fac_slot, linsolve, Jmat)
    if returncode(fac) === SINGULAR
        return _make_nan_duals(D, x_star), _singular_status(status)
    end
    x_hat = _seed_zero_duals(D, x_star)
    r_dual = residual!!(nothing, prob, x_hat, p)
    if x_star isa Number
        return _assemble_scalar_dual(D, x_star, fac, r_dual), status
    else
        tangents = _solve_static_tangents(Val(N), eltype(x_star), fac, r_dual)
        return _assemble_static_duals(D, x_star, tangents), status
    end
end

@inline function _implicit_tangent(
        ::Type{D}, state, prob::NonlinearProblem,
        x_star::StaticVector, m,
        p, p_val, status::SolverStatus
) where {Tg, V, N, D <: ForwardDiff.Dual{Tg, V, N}}
    return _implicit_tangent_r1(D, state, prob, x_star, m, p, p_val, status)
end

@inline function _implicit_tangent(
        ::Type{D}, state, prob::NonlinearProblem,
        x_star::Number, m,
        p, p_val, status::SolverStatus
) where {Tg, V, N, D <: ForwardDiff.Dual{Tg, V, N}}
    return _implicit_tangent_r1(D, state, prob, x_star, m, p, p_val, status)
end

function _implicit_tangent(
        ::Type{D}, state, prob::NonlinearProblem,
        x_star::AbstractVector, m,
        p, p_val, status::SolverStatus
) where {Tg, V, N, D <: ForwardDiff.Dual{Tg, V, N}}
    linsolve = m isa Newton ? m.linsolve : LUFactorization()
    jacprep = _tangent_jacprep(state, prob, x_star, p_val)
    fac_slot = m isa Newton ? state.fac :
               init_factor(linsolve, prob, x_star, p_val, jacprep)
    J_slot = m isa Newton ? state.J : fac_slot.factors
    Jmat = jacobian!!(J_slot, jacprep, prob, x_star, p_val)
    fac = factorize!!(fac_slot, linsolve, Jmat)
    if returncode(fac) === SINGULAR
        return _make_nan_duals(D, x_star), _singular_status(status)
    end
    x_hat = _seed_zero_duals(D, x_star)
    r_dual = residual!!(similar(x_hat), prob, x_hat, p)
    T = eltype(x_star)
    b_buf = similar(x_star)
    d_buf = state.d
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
        prob::NonlinearProblem, x₀, method, p,
        options::Union{Nothing, Options}, kwargs)
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
    return _implicit_tangent(D, state, prob, x_star, m, p, p_val, status)
end
