# Sparse 2×2-block nonlinear test problems matching GeometricBrackets'
# reduced Vlasov–Poisson sparsity pattern and sizes (N = 125…768).

using GeometricSolvers: NonlinearProblem
using Random: AbstractRNG, MersenneTwister
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse

# The element types of the sparse test problems: a real float type, and a complex type over one.
const SparseProblemEltype = Union{AbstractFloat, Complex{<:AbstractFloat}}

# A complex `T` draws one more number per entry, its imaginary part, after the real one, so that
# the real problems keep the draws of the seed.
function _sparse_circband_entries!(
        I::Vector{Int},
        J::Vector{Int},
        V::Vector{T},
        N::Int,
        row_offset::Int,
        col_offset::Int,
        ::Type{T},
        rng::AbstractRNG;
        skew::Bool = false
) where {T <: SparseProblemEltype}
    R = real(T)
    for i in 1:N, k in -3:3

        j = mod1(i + k, N)
        push!(I, row_offset + i)
        push!(J, col_offset + j)
        w = R(rand(rng)) / R(4)
        val = skew ? R(k) * (one(R) + w) : (k == 0 ? R(4) : w)
        push!(V, T <: Complex ? T(val, R(rand(rng)) / R(8)) : T(val))
    end
    return nothing
end

function _sparse_block_matrix(
        N::Int,
        ::Type{T} = Float64;
        seed::Int = 1
) where {T <: SparseProblemEltype}
    rng = MersenneTwister(seed)
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, 28 * N)
    sizehint!(J, 28 * N)
    sizehint!(V, 28 * N)
    _sparse_circband_entries!(I, J, V, N, 0, 0, T, rng; skew = false)
    _sparse_circband_entries!(I, J, V, N, 0, N, T, rng; skew = true)
    _sparse_circband_entries!(I, J, V, N, N, 0, T, rng; skew = false)
    _sparse_circband_entries!(I, J, V, N, N, N, T, rng; skew = false)
    return sparse(I, J, V, 2 * N, 2 * N)
end

"""
    sparse_bracket_problem(N::Int, ::Type{T} = Float64; seed::Int = 1)

Build a `2 × 2`-block sparse nonlinear test problem of dimension `n = 2N` with four
`7`-diagonal periodic banded blocks (`nnz(P) == 28N`, non-symmetric values `P != transpose(P)`),
cross-block nonlinear coupling (omitted at row `1` so `(1, N + 1)` is a stored zero at every
iterate), and a planted solution `z_star`. Returns `(prob_analytic, prob_ad, z0, p_val, z_star)`.

`T` is a real float type or a complex type over one. A complex `T` adds an imaginary part in
`[0, 1/8)` to each entry of the matrix and `cos(θ)/4` and `sin(θ)/4` to the planted solution; the
residual is holomorphic in `z`, so its analytic Jacobian is the complex derivative. The parameter
`p_val` is real for every `T`.
"""
function sparse_bracket_problem(
        N::Int,
        ::Type{T} = Float64;
        seed::Int = 1
) where {T <: SparseProblemEltype}
    R = real(T)
    n = 2 * N
    A = _sparse_block_matrix(N, T; seed = seed)
    P = copy(A)
    p_val = one(R)
    z_star = Vector{T}(undef, n)
    @inbounds for i in 1:N
        θ = R(2) * R(pi) * R(i) / R(N)
        z_star[i] = T <: Complex ? T(sin(θ) / R(2), cos(θ) / R(4)) : sin(θ) / R(2)
        z_star[N + i] = T <: Complex ? T(cos(θ) / R(2), sin(θ) / R(4)) : cos(θ) / R(2)
    end

    b = zeros(T, n)
    rowsA = rowvals(A)
    valsA = nonzeros(A)
    @inbounds for j in 1:n
        zj = z_star[j]
        for k in nzrange(A, j)
            b[rowsA[k]] += valsA[k] * zj
        end
    end
    @inbounds for i in 1:n
        ip = i <= N ? i + N : i - N
        zi = z_star[i]
        zp = z_star[ip]
        c = i == 1 ? zero(T) : one(T)
        b[i] += p_val * (zi^3 / T(5) + c * zp^2 / T(10))
    end

    z0 = [z_star[i] + one(T) / T(5) for i in 1:n]

    function F!(r, z, p)
        fill!(r, zero(eltype(r)))
        @inbounds for j in 1:n
            zj = z[j]
            for k in nzrange(A, j)
                r[rowsA[k]] += valsA[k] * zj
            end
        end
        @inbounds for i in 1:n
            ip = i <= N ? i + N : i - N
            zi = z[i]
            zp = z[ip]
            c = i == 1 ? zero(T) : one(T)
            r[i] += p * (zi^3 / T(5) + c * zp^2 / T(10)) - b[i]
        end
        return r
    end

    function J!(J::SparseMatrixCSC, z, p)
        valsJ = nonzeros(J)
        rowsJ = rowvals(J)
        copyto!(valsJ, valsA)
        @inbounds for j in 1:n
            zj = z[j]
            ip = j <= N ? j + N : j - N
            d_diag = p * (T(3) / T(5)) * zj^2
            d_part = ip == 1 ? zero(eltype(z)) : p * (one(T) / T(5)) * zj
            for k in nzrange(J, j)
                row = rowsJ[k]
                if row == j
                    valsJ[k] += d_diag
                elseif row == ip
                    valsJ[k] += d_part
                end
            end
        end
        return J
    end

    prob_analytic = NonlinearProblem(F!, J!; jacobian_prototype = P)
    prob_ad = NonlinearProblem(F!; jacobian_prototype = P)
    return (prob_analytic, prob_ad, z0, p_val, z_star)
end
