# The R3 AD path of `AutoForwardDiff`: an own chunked forward mode, on every array type.
#
# DifferentiationInterface with `AutoForwardDiff` raises "Scalar indexing is disallowed" on an
# `MtlArray`, a `CuArray` and a `JLArray`, and Enzyme does not differentiate Metal kernels, so the
# portable path is chunked forward mode written here: `similar(x, Dual)`, seed with a broadcast,
# call the caller's residual, extract with a broadcast. It needs nothing of the residual but that
# it be generic in its element type, which is what a kernel-portable residual already is.
#
# `N` columns of the Jacobian come out of one residual evaluation, so a Jacobian costs `ceil(n/N)`
# evaluations. The four dual buffers are allocated once, in `prepare_ad`, and no call allocates
# one.

"""
    ChunkedForwardDiff{N, Tg}

The AD preparation of the R3 path of `AutoForwardDiff`: `N` Jacobian columns per residual
evaluation, in one pass of `ForwardDiff.Dual{Tg, T, N}` arithmetic. `Tg` is the caller's tag, or
this solver's own where the caller named none, and the same tag serves the Jacobian and the JVP.

The four fields are the dual buffers: `xdual` and `rdual` carry `N` partials for
[`jacobian!!`](@ref), `xdual1` and `rdual1` one partial for [`jvp!!`](@ref). A complex iterate
gives `Complex{ForwardDiff.Dual{Tg, real(T), N}}` buffers, so that a holomorphic residual is
differentiated in its complex argument.
"""
struct ChunkedForwardDiff{N, Tg, DX, DR, DX1, DR1}
    xdual::DX
    rdual::DR
    xdual1::DX1
    rdual1::DR1

    function ChunkedForwardDiff{N, Tg}(
            xdual::DX, rdual::DR, xdual1::DX1, rdual1::DR1
    ) where {N, Tg, DX, DR, DX1, DR1}
        return new{N, Tg, DX, DR, DX1, DR1}(xdual, rdual, xdual1, rdual1)
    end
end

# The caller's tag where it named one. Otherwise this solver's own, never `Nothing`: a
# `Dual{Nothing}` confuses the perturbations of a residual that differentiates in its own body.
tag_of(backend::AutoForwardDiff, prob, x) = backend.tag
function tag_of(::AutoForwardDiff{C, Nothing}, prob, x) where {C}
    ForwardDiff.Tag(prob.F, eltype(x))
end

# The caller's chunk size where it is positive, `ForwardDiff.pickchunksize(n)` otherwise, clamped
# to `1:n`: a chunk wider than the iterate would seed columns that do not exist.
function chunk_size(::AutoForwardDiff{C}, n::Integer) where {C}
    N = (C isa Integer && C > 0) ? C : ForwardDiff.pickchunksize(n)
    return max(1, min(N, n))
end

# The element type of a dual buffer. A complex iterate is differentiated as a holomorphic function
# of its complex argument, which is a `Dual` in each of the two real components and not a `Dual`
# whose value is complex.
dual_type(::Type{T}, ::Type{Tg}, N) where {T <: Real, Tg} = ForwardDiff.Dual{Tg, T, N}
function dual_type(::Type{Complex{T}}, ::Type{Tg}, N) where {T <: Real, Tg}
    return Complex{ForwardDiff.Dual{Tg, T, N}}
end

function prepare_ad(backend::AutoForwardDiff, prob, r::AbstractArray, x::AbstractArray, p)
    check_iterate_shape(x)
    N = chunk_size(backend, length(x))
    Tg = typeof(tag_of(backend, prob, x))
    return ChunkedForwardDiff{N, Tg}(
        similar(x, dual_type(eltype(x), Tg, N)),
        similar(r, dual_type(eltype(r), Tg, N)),
        similar(x, dual_type(eltype(x), Tg, 1)),
        similar(r, dual_type(eltype(r), Tg, 1))
    )
end

# A dual like `d` with the value `x` and the partials `lanes`. The element of the buffer carries
# the tag and the number of partials, so that no type travels into the broadcast. A complex dual
# takes the real parts of `x` and the lanes in its real component and the imaginary parts in its
# imaginary one: a real lane perturbs the real component only.
@inline function seed(::ForwardDiff.Dual{Tg, V}, x, lanes::Tuple) where {Tg, V}
    return ForwardDiff.Dual{Tg}(convert(V, x), map(l -> convert(V, l), lanes)...)
end

@inline function seed(d::Complex{<:ForwardDiff.Dual}, x, lanes::Tuple)
    return Complex(
        seed(real(d), real(x), map(real, lanes)), seed(imag(d), imag(x), map(imag, lanes))
    )
end

# Partial `k` of a residual entry: one Jacobian entry, converted to the element type of `J` by the
# assignment that the broadcast makes.
@inline dual_partial(d::ForwardDiff.Dual, k::Integer) = ForwardDiff.partials(d, k)

@inline function dual_partial(d::Complex{<:ForwardDiff.Dual}, k::Integer)
    return complex(ForwardDiff.partials(real(d), k), ForwardDiff.partials(imag(d), k))
end

function jacobian!!(J, prep::ChunkedForwardDiff{N}, prob, x, p) where {N}
    n = length(x)
    for j0 in 0:N:(n - 1)
        # lane `k` of entry `i` is one exactly where `i` is column `j0 + k`; the spare lanes of a
        # partial last chunk match no entry, so they seed zero and only its `m` columns are read
        lanes(i) = ntuple(k -> i == j0 + k, Val(N))
        m = min(N, n - j0)
        prep.xdual .= seed.(prep.xdual, x, lanes.(eachindex(x)))
        prob.F(prep.rdual, prep.xdual, p)
        @views J[:, (j0 + 1):(j0 + m)] .= dual_partial.(prep.rdual, (1:m)')
    end
    return J
end
