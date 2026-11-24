# The factor of `KroneckerFactorization`: the `n × n` blocks `γᵢ I - J`, each scaled by a power of
# two, of the Newton matrix `I - h A ⊗ J` of a stage system in the eigenbasis of the stage matrix
# `A = V Λ V⁻¹`, and the transforms `V` and `V⁻¹` between the stacked increments and that basis.
#
# The blocks fall into two lists. A block of the first list is factorised in the element type `T`
# of the iterate, and its solution enters the step once: a real eigenvalue for a real `T`, and each
# eigenvalue for a complex `T`. A block of the second list, only for a real `T`, is the first
# eigenvalue of a complex-conjugate pair, factorised in `complex(T)`; the solution of the second
# eigenvalue of the pair is the conjugate of its solution, so the pair enters the step as twice the
# real part of that solution. `μ` holds `1 / λ` of each block, so that `γ = μ / h` at a refresh;
# `h` is the step of the last refresh. `identity` marks the refresh of `h = 0`, whose Newton matrix
# is the identity, and `finite` is `false` after a refresh with a non-finite `h` or `J`, whose solve
# gives a direction of `NaN`. `info` is the position of the first block whose factor failed,
# counting the first list before the second, or `0`.
struct KroneckerFactor{T, C, R, FT, FC}
    facs::Vector{FT}
    M::Matrix{T}
    b::Vector{T}
    μ::Vector{T}
    V::Matrix{T}
    W::Matrix{T}
    pair_facs::Vector{FC}
    pair_M::Matrix{C}
    pair_b::Vector{C}
    pair_μ::Vector{C}
    pair_V::Matrix{C}
    pair_W::Matrix{C}
    h::R
    info::Int
    identity::Bool
    finite::Bool
end

@inline function with_refresh(
        fac::KroneckerFactor, h, info::Int, identity::Bool, finite::Bool)
    return KroneckerFactor(fac.facs, fac.M, fac.b, fac.μ, fac.V, fac.W, fac.pair_facs,
        fac.pair_M, fac.pair_b, fac.pair_μ, fac.pair_V, fac.pair_W, oftype(fac.h, h), info,
        identity, finite)
end

# The block factors of `block` on `n`-vectors of element type `S`, one per block.
function block_factors(block::LinearMethod, ::Type{S}, n::Int, count::Int) where {S}
    x = Vector{S}(undef, n)
    first = init_linear(block, nothing, x)
    facs = Vector{typeof(first)}(undef, count)
    for i in 1:count
        facs[i] = i == 1 ? first : init_linear(block, nothing, x)
    end
    return facs
end

# The factor of `block` for the stage matrix `A` and the stacked iterate `x` of element type `T`:
# the eigendecomposition of `A` in `Float64` (`ComplexF64` for a complex `A`), its checks, and the
# blocks, unfactorised.
function kronecker_factor(block::LinearMethod, A::AbstractMatrix, x::AbstractVector{T}) where {T}
    size(A, 1) == size(A, 2) || throw(ArgumentError(
        "KroneckerFactorization needs a square stage matrix, got one of size $(size(A))"))
    s = size(A, 1)
    length(x) % s == 0 || throw(ArgumentError(
        "KroneckerFactorization needs an iterate whose length is a multiple of the number of stages: the length $(length(x)) is not a multiple of s = $s"))
    T <: Real && !(eltype(A) <: Real) &&
        throw(ArgumentError(
            "KroneckerFactorization with a real iterate needs a real stage matrix, got $(eltype(A))"))
    E = eigen(eltype(A) <: Real ? Matrix{Float64}(A) : Matrix{ComplexF64}(A))
    λ, V = E.values, E.vectors
    κ = cond(V)
    bound = inv(sqrt(eps(real(T))))
    # `!(κ ≤ bound)` refuses a `NaN` condition number as well
    κ <= bound || throw(ArgumentError(
        "the stage matrix is not diagonalisable to the working precision: its eigenvectors have cond(V) = $κ, above eps($(real(T)))^(-1/2) = $bound"))
    any(iszero, λ) && throw(ArgumentError(
        "KroneckerFactorization needs an invertible stage matrix: the stage matrix has the eigenvalue 0"))
    W = inv(V)
    C = complex(T)
    n = length(x) ÷ s
    # a real eigenvalue of a real `A` has an imaginary part of exactly zero, and its eigenvector
    # and its row of `V⁻¹` are real up to rounding, which the real part drops
    single = T <: Complex ? collect(1:s) : findall(l -> iszero(imag(l)), λ)
    pairs = T <: Complex ? Int[] : findall(l -> imag(l) > 0, λ)
    conv(z) = T <: Real ? T(real(z)) : T(z)
    return KroneckerFactor(
        block_factors(block, T, n, length(single)), Matrix{T}(undef, n, n), Vector{T}(undef, n),
        conv.(inv.(λ[single])), conv.(V[:, single]), conv.(W[single, :]),
        block_factors(block, C, n, length(pairs)), Matrix{C}(undef, n, n), Vector{C}(undef, n),
        C.(inv.(λ[pairs])), C.(V[:, pairs]), C.(W[pairs, :]),
        real(T)(NaN), 0, false, true)
end

# The power of two `τ` by which the blocks of a refresh at the nonzero finite step `h` are scaled:
# the block `τ (γ I - J) = (μ / η) I - τ J`, with `η = h / τ`. For `|h| < 1`, `τ = 2^exponent(h)`,
# so that `|η|` lies in `[1, 2)` and `μ / η` stays finite where `γ = μ / h` overflows for a tiny
# `h`; otherwise `τ = 1`, so that `τ J` cannot overflow. A power of two scales without rounding,
# up to an underflow, so where `γ` is finite the block is `γ I - J` times `τ` exactly.
@inline step_scale(h) = abs(h) < 1 ? ldexp(one(h), exponent(h)) : one(h)

# The refresh of the factor at the step `h` and the Jacobian `J` of the field: each block
# `τ (γᵢ I - J)`, `γᵢ = μᵢ / h`, factorised with `block`.
function factorize_stages!!(fac::KroneckerFactor, block::LinearMethod, J::AbstractMatrix, h)
    iszero(h) && return with_refresh(fac, h, 0, true, true)
    (isfinite(h) && all(isfinite, J)) || return with_refresh(fac, h, 0, false, false)
    τ = step_scale(h)
    η = h / τ
    info = 0
    for i in eachindex(fac.facs)
        f = factorize!!(fac.facs[i], block, regularize!!(fac.M .= .-τ .* J, fac.μ[i] / η))
        fac.facs[i] = f
        info == 0 && f.info != 0 && (info = i)
    end
    for i in eachindex(fac.pair_facs)
        f = factorize!!(fac.pair_facs[i], block,
            regularize!!(fac.pair_M .= .-τ .* J, fac.pair_μ[i] / η))
        fac.pair_facs[i] = f
        info == 0 && f.info != 0 && (info = length(fac.facs) + i)
    end
    return with_refresh(fac, h, info, false, true)
end

# `b = ρ (W[i, :] ⊗ I) r`, the right-hand side of block `i` scaled by the `ρ = μ / η` of its block
# `ρ I - τ J`: with `I - h λ J = (h λ / τ) (ρ I - τ J)`, the block solution `(I - h λ J)⁻¹ y` is
# `(ρ I - τ J)⁻¹ (ρ y)`.
@inline function block_rhs!(
        b::AbstractVector, W::AbstractMatrix, i::Int, ρ, r::AbstractVector)
    n = length(b)
    @inbounds for l in 1:n
        acc = zero(eltype(b))
        for j in axes(W, 2)
            acc += W[i, j] * r[(j - 1) * n + l]
        end
        b[l] = ρ * acc
    end
    return b
end

# The solve of `(I - h A ⊗ J) d = r` with the refreshed factor: `d = (V ⊗ I) w`, with `w` the
# block solutions of `(V⁻¹ ⊗ I) r`. `d` and `r` are distinct vectors of `s n` entries.
function ldiv!!(d::AbstractVector{T}, fac::KroneckerFactor{T}, r::AbstractVector{T}) where {T}
    fac.identity && return copyto!(d, r)
    fac.finite || return fill!(d, T(NaN))
    n = length(fac.b)
    η = fac.h / step_scale(fac.h)
    fill!(d, zero(T))
    for i in eachindex(fac.facs)
        w = ldiv!!(fac.b, fac.facs[i], block_rhs!(fac.b, fac.W, i, fac.μ[i] / η, r))
        @inbounds for k in axes(fac.V, 1)
            v = fac.V[k, i]
            for l in 1:n
                d[(k - 1) * n + l] += v * w[l]
            end
        end
    end
    for i in eachindex(fac.pair_facs)
        w = ldiv!!(fac.pair_b, fac.pair_facs[i],
            block_rhs!(fac.pair_b, fac.pair_W, i, fac.pair_μ[i] / η, r))
        @inbounds for k in axes(fac.pair_V, 1)
            v = fac.pair_V[k, i]
            for l in 1:n
                d[(k - 1) * n + l] += 2 * real(v * w[l])
            end
        end
    end
    return d
end

# The Kronecker solve is that of the simplified Newton matrix `I - h A ⊗ J`, not of the Jacobian
# of the stacked residual, so a line search measures its slope.
exact_solve(::KroneckerFactor) = false

stored_factors(::KroneckerFactor) = nothing
