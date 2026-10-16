# The value-form problems of the R1 tests: the out-of-place residual `F(x, p)` and the analytic
# `J(x, p)` beside it, so that a test can compare the two exactly.
#
# `test/helpers/adproblems.jl` holds the in-place stubs of the R3 tests, which fill a residual
# buffer: on an `SVector` such a stub cannot even be called, because `r .= …` does not index-assign
# an immutable array. These are its R1 counterparts — the same residuals, the same Jacobians, the
# same intent — and the comparisons they are written for are exact (`==`), because a tolerance
# would hide exactly the last bit that the seam tests read.

using ForwardDiff: ForwardDiff
using StaticArrays: SMatrix, SVector

"A cyclic shift by one, as the permutation of [`R1Coupled`](@ref): `[2, 3, …, n, 1]`."
r1_shift(::Type{T}, n::Int) where {T} = SVector{n, Int}([i == n ? 1 : i + 1 for i in 1:n])

"""
    diagonal_matrix(v)

The dense diagonal `SMatrix` with `v` on the diagonal. A problem's `J` returns an `SMatrix` in R1,
and a `Diagonal` is not one, so the analytic Jacobian of a diagonal problem is built here.
"""
function diagonal_matrix(v::SVector{n, T}) where {n, T}
    return SMatrix{n, n, T}(
        ntuple(k -> ((k - 1) % n) == ((k - 1) ÷ n) ? v[(k - 1) % n + 1] : zero(T), Val(n *
                                                                                       n)))
end

"The `n × n` identity, in the dense form a problem's `J` returns."
identity_matrix(::Type{T}, n::Int) where {T} = diagonal_matrix(SVector{n, T}(ones(T, n)))

"""
    r1_inputs(::Type{T}, n)

The iterate and the parameters of the R1 tests: no zero and no equal entry, so that a wrong column
cannot look right by accident.
"""
function r1_inputs(::Type{T}, n::Int) where {T <: Real}
    return SVector{n, T}([T(i) / T(n + 1) for i in 1:n]),
    SVector{n, T}([T(2i + 3) / T(n + 5) for i in 1:n])
end

function r1_inputs(::Type{Complex{T}}, n::Int) where {T <: Real}
    return SVector{n, Complex{T}}([Complex{T}(T(i) / T(n + 1), T(i + 2) / T(n + 4))
                                   for i in 1:n]),
    SVector{n, Complex{T}}([Complex{T}(T(2i + 3) / T(n + 5), -T(i + 1) / T(n + 6))
                            for i in 1:n])
end

"""
    r1_hostile_values(::Type{T})

The argument set of `evidence.md` for `T`: the two zeros, the smallest subnormal, the smallest
normal, `eps`, one, the largest finite values, the infinities, a `NaN`, and the scale family
`s = 10^k` for `k = -8:4:8`. A value is a value here and not a throw — the seam returns what the
problem computes, and the finiteness rule belongs to the loop. A complex `T` takes both of its
components from the same set.
"""
function r1_hostile_values(::Type{T}) where {T <: Real}
    return (
        zero(T), -zero(T), nextfloat(zero(T)), floatmin(T), eps(T), one(T), floatmax(T) / 2,
        floatmax(T), T(Inf), -T(Inf), T(NaN), map(k -> T(10)^k, -8:4:8)...)
end

function r1_hostile_values(::Type{Complex{T}}) where {T <: Real}
    return map(a -> Complex{T}(a, a), r1_hostile_values(T))
end

"""
    r1_forwarddiff_jacobian(F, x, p)

The reference Jacobian of the value-form residual `F` at `x`, from `ForwardDiff.jacobian` on the
host copies. It is a reference for a real element type only: `ForwardDiff.jacobian` throws on a
complex iterate, which is one of the reasons the R1 seeding exists.
"""
r1_forwarddiff_jacobian(F, x, p) = ForwardDiff.jacobian(z -> F(z, p), Array(x))

"A stand-in for a caller who differentiates this solver from outside: a tag this solver never
builds for itself, because its own tag names the residual."
struct CallerTag end

"""
    r1_tagged_scale(::Type{D})

`2` where the element type `D` carries [`CallerTag`](@ref)'s tag, `1` otherwise. The residual of
[`R1Tagged`](@ref) reads it off the element type of its argument, so the tag a differentiation
carries decides the answer, and a tag that was dropped on the way shows up as a value.
"""
r1_tagged_scale(::Type{<:ForwardDiff.Dual{<:ForwardDiff.Tag{CallerTag}}}) = 2
function r1_tagged_scale(::Type{<:Complex{<:ForwardDiff.Dual{<:ForwardDiff.Tag{CallerTag}}}})
    return 2
end
r1_tagged_scale(::Type) = 1

"""
    R1Tagged()

`F(x, p) = 2x` where the differentiation carries [`CallerTag`](@ref)'s tag, and `F(x, p) = x`
otherwise, because [`r1_tagged_scale`](@ref) reads the tag off `eltype(x)`. The tag is therefore a
value a test can compare: with the caller's tag the Jacobian is `2I` and the JVP `2v`, and with
this solver's own tag it is `I` and `v`.
"""
struct R1Tagged end

(::R1Tagged)(x, p) = x .* r1_tagged_scale(eltype(x))

"""
    R1Scaled()

The value form of `Scaled`: `F(x, p) = p .* x`, so the Jacobian is `Diagonal(p)`. The parameters
appear in the result, and a parameter that is differentiated by mistake shows up at once.
"""
struct R1Scaled end

(::R1Scaled)(x, p) = p .* x

r1_scaled_jacobian(x, p) = diagonal_matrix(p)

"""
    R1ScalarScaled()

The `Number` iterate of `R1Scaled`: `F(x, p) = p x` and `J(x, p) = p`, so that an `n = 1` problem
has a `Number` Jacobian and not a `1 × 1` matrix.
"""
struct R1ScalarScaled end

(::R1ScalarScaled)(x, p) = p * x

r1_scalar_scaled_jacobian(x, p) = p

"""
    R1Holomorphic()

The value form of `Holomorphic`: `F(x, p) = x .* x .+ p .* x`, holomorphic in `x`, so that in a
complex element type the Jacobian is the complex derivative `Diagonal(2x + p)` and not the real
Jacobian of its two components.
"""
struct R1Holomorphic end

(::R1Holomorphic)(x, p) = x .* x .+ p .* x

# `x + x + p`, not `2x + p`: the duals add the two halves of `x .* x` first, so this is the
# expression the AD evaluates and the comparison stays exact.
r1_holomorphic_jacobian(x, p) = diagonal_matrix(x .+ x .+ p)

"[`R1Closing`](@ref)'s scalar term: `x ⋅ ∂/∂s(s x)|_{s = 1}`, which is `x²`."
function r1_closing_term(xi, pi)
    return xi * ForwardDiff.derivative(s -> s * xi, one(real(eltype(pi))))
end

"""
    R1Closing()

A residual in the shape of `Nested`: `F(x, p) = x .* x`, written so that the residual itself
calls `ForwardDiff.derivative` over a variable its own body closes over — the inner body holds
`xi`, the outer dual, so the inner and the outer perturbation meet. This is the nested
differentiation a solver's tag rule exists for: with the outer tag missing the two are confused
and ForwardDiff raises `DualMismatchError`, where the tagged pass gives `J = Diagonal(2x)`.

The inner variable is a plain real in every element type, so a complex iterate gets the
holomorphic derivative and no `DimensionMismatch`. The derivative is `2x` exactly at every entry:
the duals evaluate `fl(1 ⋅ x_i) + fl(x_i ⋅ 1)`.
"""
struct R1Closing end

(::R1Closing)(x, p) = map(r1_closing_term, x, p)

r1_closing_jacobian(x, p) = diagonal_matrix(x .+ x)

"""
    R1Coupled(perm)

The value form of `Coupled`: `F(x, p) = x .* x .+ p .* x .+ 2 .* x[perm]`, so the Jacobian is a
diagonal plus the permutation matrix of `perm`. A diagonal Jacobian cannot see a transposed or a
shifted column, which is what this coupling is for.
"""
struct R1Coupled{P}
    perm::P
end

(f::R1Coupled)(x, p) = x .* x .+ p .* x .+ 2 .* x[f.perm]

# `2 x_i + p_i` on the diagonal, in the order the duals add it, and `2` in row `i` at column
# `perm[i]`: the derivative of `2 x[perm[i]]` is exactly two, and the other two terms contribute an
# exact zero there.
function r1_coupled_jacobian(x, p, perm)
    n = length(x)
    T = eltype(x)
    return SMatrix{n, n, T}(ntuple(k -> begin
            i = (k - 1) % n + 1
            j = (k - 1) ÷ n + 1
            d = x[i] + x[i] + p[i]
            if j != i
                perm[i] == j ? T(2) : zero(T)
            else
                perm[i] == i ? d + T(2) : d
            end
        end, Val(n * n)))
end

"""
    R1Cube()

`F(x, p) = x .* x .* x` with the analytic `J(x, p) = Diagonal(3 x .* x)`. The two round
differently — the duals evaluate `fl(x²) + fl((x + x) x)` and the analytic form `fl((3x) x)` — so
at the entries of [`r1_cube_input`](@ref) they differ in the last bit. That difference is what lets
a test see a call that reaches ForwardDiff where the problem's own `J` was asked for.
"""
struct R1Cube end

(::R1Cube)(x, p) = x .* x .* x

r1_cube_jacobian(x, p) = diagonal_matrix(3 .* x .* x)

"""
    r1_cube_input(::Type{T}, n)

The iterate of the [`R1Cube`](@ref) tests, at which `J(x, p) != ForwardDiff.jacobian` in the last
bit for at least one entry, in every element type the package supports. The values are stated and
measured rather than random, so that the test is the same in every run; `1.7` is the entry at which
the two roundings first differ in `Float64`.
"""
function r1_cube_input(::Type{T}, n::Int) where {T <: Real}
    return SVector{n, T}([
        T(5) / T(4), T(17) / T(10), T(9) / T(5), T(31) / T(10), T(59) / T(10),
        T(27) / T(10), T(19) / T(10)][1:n])
end

function r1_cube_input(::Type{Complex{T}}, n::Int) where {T <: Real}
    return SVector{n, Complex{T}}([Complex{T}(T(17) / T(10), T(3) / T(10)),
        Complex{T}(T(31) / T(10), -T(19) / T(10)),
        Complex{T}(T(7) / T(10), T(11) / T(10)),
        Complex{T}(T(23) / T(10), T(47) / T(10)),
        Complex{T}(T(59) / T(10), T(1) / T(10)),
        Complex{T}(T(19) / T(10), -T(7) / T(10)),
        Complex{T}(T(13) / T(10), T(21) / T(10))][1:n])
end
