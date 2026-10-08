# The R3 AD path on a device array: the own chunked forward mode (§3.1).
#
# DifferentiationInterface with `AutoForwardDiff` raises "Scalar indexing is disallowed" on an
# `MtlArray`, a `CuArray` and a `JLArray` — measured in part D — and Enzyme does not differentiate
# Metal kernels, so the portable path is chunked forward mode written here: `similar(x, Dual)`,
# seed with a broadcast, call the caller's residual, extract with a broadcast. It needs nothing of
# the residual but that it be generic in its element type, which is what a kernel-portable
# residual already is.
#
# `N` columns of the Jacobian come out of one residual evaluation, so a Jacobian costs `ceil(n/N)`
# evaluations. The four dual buffers are allocated once, in `prepare_ad`, and are state slots: no
# call allocates one.

"""
    ChunkedForwardDiff{N, Tg, P}

The AD preparation of the R3 device path: `N` Jacobian columns per residual evaluation, in one
pass of `ForwardDiff.Dual{Tg, T, N}` arithmetic. `Tg` is the tag, this solver's own where the
caller named none, and the same tag serves the Jacobian and the JVP. `P` is the type of the
parameters the preparation was built for: a replacement object of that type takes effect with no
new solver, and another type raises `ArgumentError` naming both types.

The four fields are the dual buffers, and they are the state slots that make a solve
allocation-free: `xdual` and `rdual` carry `N` partials for [`jacobian!!`](@ref), `xdual1` and
`rdual1` one partial for [`jvp!!`](@ref). A complex iterate gives
`Complex{ForwardDiff.Dual{Tg, real(T), N}}` buffers, so that a holomorphic residual is
differentiated in its complex argument.
"""
struct ChunkedForwardDiff{N, Tg, P, DX, DR, DX1, DR1}
    xdual::DX
    rdual::DR
    xdual1::DX1
    rdual1::DR1

    function ChunkedForwardDiff{N, Tg, P}(
            xdual::DX, rdual::DR, xdual1::DX1, rdual1::DR1
    ) where {N, Tg, P, DX, DR, DX1, DR1}
        return new{N, Tg, P, DX, DR, DX1, DR1}(xdual, rdual, xdual1, rdual1)
    end
end

"""
    ChunkedForwardDiff(backend::AutoForwardDiff, prob, r, x, p)

Allocate the four dual buffers for the residual of `prob`, an iterate like `x` and a residual
buffer like `r`, and fix the chunk size, the tag and the parameter type from `p`.
[`prepare_ad`](@ref) calls this for a device array iterate; it is written for any array type,
because the chunked mode is the one code that runs on every backend and is therefore also what a
CPU comparison tests. A non-vector iterate is refused here, as it is on the DI path.
"""
function ChunkedForwardDiff(
        backend::AutoForwardDiff, prob, r::AbstractArray, x::AbstractArray, p
)
    check_iterate_shape(x)
    N = chunk_size(backend, length(x))
    Tg = typeof(tag_of(backend, prob, x))
    return ChunkedForwardDiff{N, Tg, typeof(p)}(
        similar(x, dual_type(eltype(x), Tg, N)),
        similar(r, dual_type(eltype(r), Tg, N)),
        similar(x, dual_type(eltype(x), Tg, 1)),
        similar(r, dual_type(eltype(r), Tg, 1))
    )
end

function prepare_ad(
        backend::AutoForwardDiff, prob, r::AbstractGPUArray, x::AbstractGPUArray, p
)
    return ChunkedForwardDiff(backend, prob, r, x, p)
end

# The element type of a dual buffer. A complex iterate is differentiated as a holomorphic function
# of its complex argument, which is a `Dual` in each of the two real components and not a
# `Dual` whose value is complex.
function dual_type(::Type{T}, ::Type{Tg}, N::Integer) where {T <: Real, Tg}
    return ForwardDiff.Dual{Tg, T, N}
end

function dual_type(::Type{Complex{T}}, ::Type{Tg}, N::Integer) where {T <: Real, Tg}
    return Complex{ForwardDiff.Dual{Tg, T, N}}
end

# The seed of entry `i` for the chunk that starts after column `j0`: lane `k` carries a one
# exactly where `i` is column `j0 + k`, and a zero everywhere else. A lane of the last chunk that
# has no column left seeds nothing at all, because no index equals its column, and `jacobian!!`
# reads only the lanes whose column exists. The dual buffer's own element carries the tag and the
# number of partials, so that no type travels into the broadcast.
@inline function dual_seed(
        ::ForwardDiff.Dual{Tg, V, N}, xi, i::Integer, j0::Integer
) where {Tg, V, N}
    return ForwardDiff.Dual{Tg}(
        convert(V, xi), ntuple(k -> ifelse(i == j0 + k, one(V), zero(V)), Val(N))...
    )
end

@inline function dual_seed(
        d::Complex{<:ForwardDiff.Dual}, xi, i::Integer, j0::Integer
)
    return Complex(dual_seed(real(d), real(xi), i, j0), dual_zero(real(d), imag(xi)))
end

# The same dual with no perturbation: the imaginary component of a complex seed.
@inline function dual_zero(::ForwardDiff.Dual{Tg, V, N}, xi) where {Tg, V, N}
    return ForwardDiff.Dual{Tg}(convert(V, xi), ntuple(_ -> zero(V), Val(N))...)
end

# Partial `k` of a residual entry: one Jacobian entry, converted to the element type of `J` by the
# assignment that the broadcast makes.
@inline dual_partial(d::ForwardDiff.Dual, k::Integer) = ForwardDiff.partials(d, k)

@inline function dual_partial(d::Complex{<:ForwardDiff.Dual}, k::Integer)
    return complex(ForwardDiff.partials(real(d), k), ForwardDiff.partials(imag(d), k))
end

function jacobian!!(J, prep::ChunkedForwardDiff{N, Tg, P}, prob, x, p) where {N, Tg, P}
    check_parameter_type(P, p)
    n = length(x)
    xdual, rdual = prep.xdual, prep.rdual
    for j0 in 0:N:(n - 1)
        # the last chunk is partial where `N` does not divide `n`: its spare lanes are seeded with
        # zero by `dual_seed`, and only its `m` real columns are written
        m = min(N, n - j0)
        xdual .= dual_seed.(xdual, x, eachindex(x), j0)
        prob.F(rdual, xdual, p)
        @views J[:, (j0 + 1):(j0 + m)] .= dual_partial.(rdual, (1:m)')
    end
    return J
end
