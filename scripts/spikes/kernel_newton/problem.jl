# The test problem of part E (§10.E): F(x) = A x + sin.(x)/10 - b, with one diagonally
# dominant A per n and a per-system x* from a fixed host seed, so that every solve has a known
# answer. The Jacobian J = A + diag(cos x)/10 has condition number κ∞ < 2 on every system.

const SEED_A = 0x5eed
const SEED_X = 0x5eedf00d

# A = 2I + R, R symmetric with a zero diagonal and off-diagonal row sums at most 1/2. Then
# ‖A‖∞ ≤ 2.5 and ‖A⁻¹‖∞ ≤ 2/3, so κ∞(A) < 2, and the diagonal of J moves by at most 0.1, so
# κ∞(J) < 2 as well.
function make_A(::Type{T}, n::Int) where {T}
    rng = Xoshiro(SEED_A + n)
    A = zeros(T, n, n)
    for j in 2:n, i in 1:(j - 1)

        r = (rand(rng, T) * 2 - 1) / (2 * (n - 1))
        A[i, j] = r
        A[j, i] = r
    end
    for i in 1:n
        A[i, i] = T(2)
    end
    @assert cond(A, Inf) < 2
    @assert maximum(sum(abs, view(A, i, :)) - abs(A[i, i]) for i in 1:n) <= T(1) / 2
    return SMatrix{n, n, T}(A)
end

# One x* per system, in [-1, 1]^n, from the fixed host seed.
function make_xstar(::Type{T}, n::Int, system::Int) where {T}
    rng = Xoshiro(SEED_X + system * 1000003 + n)
    x = rand(rng, T, n)
    for i in eachindex(x)
        x[i] = T(2) * x[i] - one(T)
    end
    return SVector{n, T}(x)
end

# b is built on the host, in T, from x*.
make_b(A::SMatrix{n, n, T}, xs::SVector{n, T}) where {n, T} = A * xs + sin.(xs) / 10

# F, the map the Newton iteration needs; each solver builds J in its own storage. The iterate may
# carry another element type: the SimpleNonlinearSolve baseline differentiates F with ForwardDiff,
# so its x holds dual numbers while A and b stay in T.
function residual(A::SMatrix{n, n, T}, b::SVector{n, T}, x::SVector{n, Tx}) where {n, T, Tx}
    A * x + sin.(x) / 10 - b
end
