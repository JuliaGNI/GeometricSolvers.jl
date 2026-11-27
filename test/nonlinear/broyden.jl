# The good Broyden update of `Newton(; reuse = k, update = BroydenUpdate(m))`: along a reused
# factor `B₀`, the step takes the direction `-Hₖ F` of the inverse
# `Hₖ = (I + uₖ sₖᴴ) ⋯ (I + u₁ s₁ᴴ) B₀⁻¹`, with `u = (s - H y) / (sᴴ H y)` from each accepted step
# `s` and its residual change `y`. The tests check the secant condition `H₊ y = s`, the edges of
# the update, the refusals of the constructor and of the state build, the R1 storage, a solve in a
# KernelAbstractions kernel on CPU, and the dispatch of the `Int32` step methods.

using Adapt: Adapt
using JET: JET
using KernelAbstractions: @index, @kernel, synchronize
using LinearAlgebra: I, dot, mul!, norm
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: BroydenUpdate, GMRESRefinement, LUFactorization,
                        LeastSquaresProblem,
                        NONFINITE, Newton, NonlinearProblem, Options, QRFactorization,
                        Refinement, SUCCESS, Static, init_state, solve
const GS = GeometricSolvers

include("../helpers/matrix.jl")

# The 3 × 3 linear residual `F(x) = A x - b` and a constant Jacobian `B₀ ≠ A`, in `T`; a complex
# `T` gives `A` and `B₀` an imaginary part of their own.
function linear_system(::Type{T}) where {T}
    R = real(T)
    A = T.(R[4 1 0; 1 3 1; 0 1 2])
    E = T.(R[3 0 1; 0 2 0; -1 0 4] ./ 10)
    if T <: Complex
        A = A .+ im .* T.(R[1 0 -1; 2 1 0; 0 -1 1] ./ 4)
        E = E .+ im .* T.(R[0 1 0; 0 0 1; 1 0 0] ./ 10)
    end
    return A, A .+ E, T.(R[1, 2, 3]), T.(R[1, -1, 2])
end

# The problem `F(x) = A x - b` and the method with the Jacobian `B₀`, on the storage of `x₀`: a
# `Vector` or a `JLArray` (R3), or an `SVector` (R1).
function linear_case(A, B₀, b, ::Type{<:SVector}; m_kw...)
    n = length(b)
    As = SMatrix{n, n}(A)
    Bs = SMatrix{n, n}(B₀)
    bs = SVector{n}(b)
    prob = NonlinearProblem((x, p) -> As * x - bs)
    return prob, Newton(; jacobian = (x, p) -> Bs, m_kw...)
end
function linear_case(A, B₀, b, ::Type{V}; m_kw...) where {V}
    Ad, Bd, bd = V(A), V(B₀), V(b)
    prob = NonlinearProblem((y, x, p) -> (mul!(y, Ad, x); y .-= bd; nothing))
    return prob, Newton(; jacobian = (J, x, p) -> copyto!(J, Bd), m_kw...)
end

storage(::Type{<:SVector}, x) = SVector{length(x)}(x)
storage(::Type{V}, x) where {V} = V(x)

# The iterates, the residuals and the step infos of `k` steps from `x₀`, and the state after them,
# with the method in the working precision, as `solve` adapts it.
function run_steps(m, prob, x₀, k)
    m = Adapt.adapt(GS.ToReal{real(eltype(x₀))}(), m)
    x = copy(x₀)
    state = init_state(m, prob, x, nothing)
    Fx = GS.primal_residual!!(GS.residual_buffer(state), prob, x, nothing)
    state = GS.start!!(state, m, prob, x, Fx, nothing, GS.rnorm(Fx))
    xs, Fs, infos = [copy(x)], [copy(Fx)], GS.StepInfo[]
    for _ in 1:k
        x, Fx, state, info = GS.step!!(state, m, prob, x, Fx, nothing)
        push!(xs, copy(x))
        push!(Fs, copy(Fx))
        push!(infos, info)
    end
    return xs, Fs, infos, state
end

# `H y` for the inverse that `state` holds: the solve of its factor, then its pairs.
function apply_inverse(state, y)
    H₀y = GS.ldiv!!(y isa SVector ? nothing : similar(y), state.fac, y)
    return GS.apply_pairs!!(state.pairs, H₀y)
end

const BACKENDS = (SVector, ARRAY_BACKENDS...)

@testset "the update meets the secant condition H₊ y = s, $T on $V" for T in ELTYPES,
    V in BACKENDS

    R = real(T)
    A, B₀, b, x₀ = linear_system(T)
    prob, m = linear_case(A, B₀, b, V; reuse = 3, update = BroydenUpdate(2))
    # after step `k` the state holds the pairs of steps `1..k`, and `Hₖ` meets the secant condition
    # of step `k`; with `B₀ ≠ A` the factor alone does not
    for k in 1:2
        xs, Fs, infos, state = run_steps(m, prob, storage(V, x₀), k)
        @test all(info -> info.code == SUCCESS, infos)
        @test state.pairs.count == k
        s = Array(xs[k + 1] - xs[k])
        y = Fs[k + 1] - Fs[k]
        H₀y = Array(GS.ldiv!!(y isa SVector ? nothing : similar(y), state.fac, y))
        Hy = Array(apply_inverse(state, y))
        @test norm(s) > 0
        @test norm(Hy - s) <= 8 * eps(R) * (norm(s) + norm(H₀y))
        @test norm(H₀y - s) > 100 * eps(R) * (norm(s) + norm(H₀y))
    end
end

@testset "a step along the reused factor takes the direction -Hⱼ F, $T on $V" for T in ELTYPES,
    V in BACKENDS

    # with `Static()`, step `j + 1` is `x_{j+1} - x_j = -Hⱼ F(x_j)` for the inverse that the state
    # holds after step `j`, and `-Hⱼ F ≠ -B₀⁻¹ F`
    R = real(T)
    A, B₀, b, x₀ = linear_system(T)
    prob, m = linear_case(A, B₀, b, V; reuse = 3, update = BroydenUpdate(2))
    for j in 1:2
        xs, Fs, infos, state = run_steps(m, prob, storage(V, x₀), j)
        xs₊, _, _, _ = run_steps(m, prob, storage(V, x₀), j + 1)
        s = Array(xs₊[j + 2] - xs₊[j + 1])
        d = -Array(apply_inverse(state, copy(Fs[j + 1])))
        d₀ = -Array(GS.ldiv!!(Fs[j + 1] isa SVector ? nothing : similar(Fs[j + 1]),
            state.fac, Fs[j + 1]))
        tol = 8 * eps(R) * (norm(Array(xs₊[j + 1])) + norm(Array(xs₊[j + 2])))
        @test norm(s - d) <= tol
        @test norm(s - d₀) > 100 * tol
    end
end

@testset "the update leaves H unchanged on the vectors z with sᴴ z = 0, $T" for T in ELTYPES
    # the pair acts as `z ↦ z + u (sᴴ z)`, so `H₊ B₀ z = z` where `sᴴ z = 0` in the conjugated
    # product; a product that does not conjugate `s` moves such a `z` of a complex `T`
    R = real(T)
    A, B₀, b, x₀ = linear_system(T)
    prob, m = linear_case(A, B₀, b, Array; reuse = 3, update = BroydenUpdate(2))
    xs, Fs, _, state = run_steps(m, prob, x₀, 1)
    s = xs[2] - xs[1]
    r = T.(R[1, 2, -1])
    z = r - s * (sum(conj.(s) .* r) / sum(conj.(s) .* s))
    @test abs(sum(conj.(s) .* z)) <= 8 * eps(R) * norm(s) * norm(z)
    # the pair moves `z` by `u (sᴴ z)`, at most `‖u‖` times the bound above, and the sum rounds
    u = state.pairs.u[:, 1]
    @test norm(GS.apply_pairs!!(state.pairs, copy(z)) - z) <=
          8 * eps(R) * (1 + norm(u) * norm(s)) * norm(z)
end

@testset "no pair where sᴴ H y = 0, and the next step factorises, $T on $V" for T in REAL_ELTYPES,
    V in BACKENDS

    # `A` rotates by a right angle and `B₀ = I`, so `H y = A s ⟂ s` exactly
    A = T[0 -1; 1 0]
    B₀ = Matrix{T}(I, 2, 2)
    prob, m = linear_case(A, B₀, zeros(T, 2), V; reuse = 3, update = BroydenUpdate(2))
    xs, Fs, infos, state = run_steps(m, prob, storage(V, T[1, 2]), 1)
    @test Array(xs[2]) == T[3, 1]
    @test state.pairs.count == 0
    @test state.rule.age == 0
    xs, Fs, infos, state = run_steps(m, prob, storage(V, T[1, 2]), 2)
    @test infos[2].code == SUCCESS
    @test all(isfinite, Array(xs[3]))
    @test Array(xs[3]) != Array(xs[2])
end

@testset "a step of zero length stores no pair, $T on $V" for T in REAL_ELTYPES,
    V in BACKENDS
    # `s = H y = 0` gives `sᴴ H y = 0` and `u = 0 / 0`: every entry of `u` is NaN
    z = storage(V, zeros(T, 2))
    pairs = GS.init_pairs(BroydenUpdate(1), 2, z)
    den = GS.pair_dot(z, z)
    @test iszero(den)
    stored, finite = GS.store_pair!!(pairs, copy(z), copy(z), den)
    @test !finite
    @test stored.count == 0
end

@testset "no pair where sᴴ H y overflows, $T on $V" for T in REAL_ELTYPES, V in BACKENDS
    # `F(x) = 2x - b` with `B₀ = I` from `x₀ = 0`: `s = b`, `H y = 2b`, and `sᴴ H y = 4‖b‖²` is
    # `Inf` for `b` near `floatmax / 8`
    big = floatmax(T) / 8
    A = Matrix{T}(2I, 2, 2)
    B₀ = Matrix{T}(I, 2, 2)
    prob, m = linear_case(A, B₀, T[big, big], V; reuse = 3, update = BroydenUpdate(2))
    xs, Fs, infos, state = run_steps(m, prob, storage(V, zeros(T, 2)), 1)
    @test infos[1].code == SUCCESS
    @test Array(xs[2]) == T[big, big]
    @test state.pairs.count == 0
    @test state.rule.age == 0
end

@testset "no pair where u overflows, and the next step factorises, $T" for T in REAL_ELTYPES
    # `F(x) = a x` from `x₀ = √floatmin / 4` along `B₀ = 2a`, `a = 1 / √floatmin`: `s = -x₀ / 2`,
    # `H y = -x₀ / 4` and `sᴴ H y = x₀² / 8`, finite and nonzero, but `u = -(x₀ / 4) / (x₀² / 8)`
    # overflows; the second step factorises afresh and halves the iterate again
    a = inv(sqrt(floatmin(T)))
    x₀ = sqrt(floatmin(T)) / 4
    prob = NonlinearProblem((x, p) -> a * x)
    m = Newton(;
        jacobian = (x, p) -> 2a, globalization = Static(), reuse = 2,
        update = BroydenUpdate(1))
    xs, Fs, infos, state = run_steps(m, prob, x₀, 1)
    s = xs[2] - xs[1]
    Hy = (Fs[2] - Fs[1]) / (2a)
    den = s * Hy
    @test isfinite(den) && !iszero(den) && den < inv(floatmax(T))
    @test !isfinite((s - Hy) * inv(den))
    @test state.pairs.count == 0
    @test state.rule.age == 0
    xs, Fs, infos, state = run_steps(m, prob, x₀, 2)
    @test infos[2].code == SUCCESS
    @test isfinite(xs[3])
    @test abs(xs[3] - x₀ / 4) <= 8 * eps(T) * x₀ / 4
end

@testset "the array form refuses a pair whose u overflows, $T" for T in REAL_ELTYPES
    pairs = GS.init_pairs(BroydenUpdate(1), 2, zeros(T, 2))
    ε = floatmin(T) / 8
    stored, finite = GS.store_pair!!(pairs, T[1, 0], T[ε, 1], ε)
    @test !finite
    @test stored.count == 0
    stored, finite = GS.store_pair!!(pairs, T[1, 0], T[1 / 2, 1], T(1) / 2)
    @test finite
    @test stored.count == 1
    @test stored.u[:, 1] == T[1, -2]
end

@testset "an array iterate holds min(m, reuse - 1) pairs, and a larger m steps as m = reuse - 1, $T on $V" for T in REAL_ELTYPES,
    V in BACKENDS

    # one factor of `reuse = k` stores at most `k - 1` pairs, so an array iterate allocates
    # `min(m, k - 1)` columns, while a value iterate keeps all `m` pairs as `NTuple`s; the value
    # iterate runs with `m = 6`, because `NTuple`s of 1000 entries take long to compile
    R = real(T)
    A, B₀, b, x₀ = linear_system(T)
    big_m = V <: SVector ? 6 : 1000
    for (reuse, m) in ((3, 2), (3, big_m), (4, big_m), (5, 4))
        prob, mk = linear_case(A, B₀, b, V; reuse, update = BroydenUpdate(m))
        _, m_ref = linear_case(A, B₀, b, V; reuse, update = BroydenUpdate(reuse - 1))
        pairs = init_state(mk, prob, storage(V, x₀), nothing).pairs
        if V <: SVector
            @test length(pairs.u) == length(pairs.s) == m
        else
            @test size(pairs.u) == size(pairs.s) == (3, min(m, reuse - 1))
        end
        # the first factor fills its `reuse - 1` pairs
        _, _, _, state = run_steps(mk, prob, storage(V, x₀), reuse - 1)
        @test state.pairs.count == reuse - 1
        # over two periods of the factor the iterates are those of the memory `reuse - 1`
        xs, _, _, _ = run_steps(mk, prob, storage(V, x₀), 2reuse)
        xs_ref, _, _, _ = run_steps(m_ref, prob, storage(V, x₀), 2reuse)
        @test all(zip(xs, xs_ref)) do (x, x_ref)
            norm(Array(x) - Array(x_ref)) <= 8 * eps(R) * norm(Array(x_ref))
        end
        @test Array(xs[end]) != Array(xs[1])
    end
end

@testset "a fresh factor clears the pairs, $T on $V" for T in REAL_ELTYPES, V in BACKENDS

    R = real(T)
    A, B₀, b, x₀ = linear_system(T)
    prob, m = linear_case(A, B₀, b, V; reuse = 3, update = BroydenUpdate(2))
    fresh_direction(state, F) = -Array(GS.ldiv!!(F isa SVector ? nothing : similar(F),
        state.fac, F))
    # at the end of the period of `reuse = 3`: steps 1 and 2 store a pair, step 3 none, and step 4
    # factorises afresh and steps along the factor alone
    xs, Fs, infos, state = run_steps(m, prob, storage(V, x₀), 3)
    @test state.pairs.count == 2
    @test state.rule.age == 0
    xs, Fs, infos, state = run_steps(m, prob, storage(V, x₀), 4)
    d = fresh_direction(state, Fs[4])
    s = Array(xs[5] - xs[4])
    @test norm(s - d) <= 8 * eps(R) * (norm(Array(xs[4])) + norm(Array(xs[5])))
    # the same after a forced refresh: `drop_factor!!` after step 2, and a stalled step
    for force in (s -> GS.drop_factor!!(s, m),
        s -> GS.with_rule(s, GS.FactorAge(s.rule.age, true, s.rule.reused)))
        xs, Fs, infos, state = run_steps(m, prob, storage(V, x₀), 2)
        @test state.pairs.count == 2
        x, Fx = copy(xs[3]), copy(Fs[3])
        x₊, F₊, state₊, info = GS.step!!(force(state), m, prob, x, Fx, nothing)
        @test info.code == SUCCESS
        @test norm(Array(x₊) - Array(xs[3]) - fresh_direction(state₊, Fs[3])) <=
              8 * eps(R) * (norm(Array(xs[3])) + norm(Array(x₊)))
        # the fresh step stores the first pair of the new factor
        @test state₊.pairs.count == 1
    end
    # `start!!` clears the pairs, also with `keep_factor = true`, whose factor the next solve reuses
    for keep in (false, true)
        mk = Newton(
            m.linsolve, m.globalization, m.jacobian, m.refinement, m.regularization,
            m.reuse, m.best, keep, m.update)
        xs, Fs, infos, state = run_steps(mk, prob, storage(V, x₀), 2)
        @test state.pairs.count == 2
        x = copy(xs[3])
        Fx = copy(Fs[3])
        started = GS.start!!(state, mk, prob, x, Fx, nothing, GS.rnorm(Fx))
        @test started.pairs.count == 0
        @test started.rule.age == (keep ? state.rule.age : 0)
    end
end

@testset "a reused step after a fresh factor applies the pair of that factor alone, $T on $V" for T in ELTYPES,
    V in BACKENDS

    # with `reuse = 3`, step 4 factorises afresh and stores its pair `(u, s)`, and step 5 takes the
    # direction `-(I + u sᴴ) B₀⁻¹ F` with that pair alone. The reference is dense algebra on the
    # iterates and residuals of the run, not `apply_pairs!!`, so a pair of steps 1 and 2 that the
    # step applies moves the step and not the reference. The step `x₆ - x₅` rounds at
    # `eps · (‖x₅‖ + ‖x₆‖)`.
    R = real(T)
    A, B₀, b, x₀ = linear_system(T)
    prob, m = linear_case(A, B₀, b, V; reuse = 3, update = BroydenUpdate(2))
    xs, Fs, infos, _ = run_steps(m, prob, storage(V, x₀), 5)
    @test all(info -> info.code == SUCCESS, infos)
    x₄, x₅, x₆ = Array(xs[4]), Array(xs[5]), Array(xs[6])
    F₄, F₅ = Array(Fs[4]), Array(Fs[5])
    s = x₅ - x₄
    Hy = B₀ \ (F₅ - F₄)
    u = (s - Hy) / dot(s, Hy)
    H₀F = B₀ \ F₅
    d = -(H₀F + u * dot(s, H₀F))
    @test norm(x₆ - x₅ - d) <= 8 * eps(R) * (norm(x₅) + norm(x₆))
    @test norm(x₆ - x₅ + H₀F) > 100 * eps(R) * (norm(x₅) + norm(x₆))
end

@testset "a solve with the update converges, $T on $V" for T in REAL_ELTYPES, V in BACKENDS
    # `F(x) = A x + x³ - b`, nonlinear, from `x₀`, along the Jacobian of the problem
    A, _, b, x₀ = linear_system(T)
    n = length(b)
    if V <: SVector
        As, bs = SMatrix{n, n}(A), SVector{n}(b)
        prob = NonlinearProblem((x, p) -> As * x + x .^ 3 - bs)
    else
        Ad, bd = V(A), V(b)
        prob = NonlinearProblem((y, x, p) -> (mul!(y, Ad, x); y .+= x .^ 3 .- bd; nothing))
    end
    for glob in (Static(), Backtracking())
        m = Newton(; reuse = 3, update = BroydenUpdate(2), globalization = glob)
        x, st = solve(prob, storage(V, x₀ ./ 4), m)
        @test st.code == SUCCESS
        @test x != storage(V, x₀ ./ 4)
        Fx = V <: SVector ? Array(prob.F(x, nothing)) :
             (yv = similar(storage(V, b)); prob.F(yv, x, nothing); Array(yv))
        @test norm(Fx) <= sqrt(eps(T)) * norm(b)
    end
end

@testset "the constructor refuses reuse outside 2 ≤ reuse ≤ m + 1, a refinement, a matrix-free method and KroneckerFactorization" begin
    @test Newton(; reuse = 3, update = BroydenUpdate(2)).update === BroydenUpdate(2)
    @test Newton(; reuse = 2, update = BroydenUpdate(1)).reuse === Int32(2)
    @test Newton(; reuse = 5, update = BroydenUpdate(4)).reuse === Int32(5)
    @test Newton().update === nothing
    # the range check compares `reuse - 1` with `m` and forms no `m + 1`, which overflows
    for big_m in (typemax(Int32), Int(typemax(Int32)) + 1, typemax(Int))
        @test Newton(; reuse = 3, update = BroydenUpdate(big_m)).update ===
              BroydenUpdate(big_m)
    end
    range_phrase = r"^ArgumentError: Newton with update = BroydenUpdate\(m\) needs 2 ≤ reuse ≤ m \+ 1"
    for reuse in (nothing, 1, 4, 10)
        @test_throws range_phrase Newton(; reuse, update = BroydenUpdate(2))
    end
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate takes no Refinement" Newton(;
        reuse = 3, update = BroydenUpdate(2), refinement = Refinement())
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate takes no Refinement" Newton(;
        reuse = 3, update = BroydenUpdate(2), refinement = true)
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate takes no GMRESRefinement" Newton(;
        reuse = 3, update = BroydenUpdate(2), refinement = GMRESRefinement())
    factor_phrase = r"^ArgumentError: Newton with a BroydenUpdate needs a factorising linear method other than KroneckerFactorization"
    @test_throws factor_phrase Newton(;
        reuse = 3, update = BroydenUpdate(2), linsolve = GMRES(), preconditioner = I)
    @test_throws factor_phrase Newton(; reuse = 3, update = BroydenUpdate(2),
        linsolve = KroneckerFactorization())
    @test_throws r"^ArgumentError: BroydenUpdate needs a memory m ≥ 1" BroydenUpdate(0)
    @test_throws r"^ArgumentError: BroydenUpdate needs a memory m ≥ 1" BroydenUpdate(-1)
    # the messages are constant, as a device kernel that rebuilds the method needs
    message(f) =
        try
            f()
            ""
        catch err
            err.msg
        end
    @test message(() -> Newton(; reuse = 7, update = BroydenUpdate(2))) ==
          "Newton with update = BroydenUpdate(m) needs 2 ≤ reuse ≤ m + 1: a factor serves at most reuse steps and collects at most reuse - 1 pairs, which the memory of m pairs holds"
    @test message(() -> Newton(; reuse = 3, update = BroydenUpdate(2), refinement = true)) ==
          "Newton with a BroydenUpdate takes no Refinement: a refinement corrects the solve of the factor, and the update steps along another inverse"
    @test message(() -> Newton(;
        reuse = 3, update = BroydenUpdate(2), refinement = GMRESRefinement())) ==
          "Newton with a BroydenUpdate takes no GMRESRefinement: a refinement corrects the solve of the factor, and the update steps along another inverse"
    @test message(() -> Newton(;
        reuse = 3, update = BroydenUpdate(2), linsolve = GMRES(), preconditioner = I)) ==
          "Newton with a BroydenUpdate needs a factorising linear method other than KroneckerFactorization: the update acts on the solve of a dense or sparse factor"
    @test message(() -> Newton(;
        reuse = 3, update = BroydenUpdate(2), linsolve = KroneckerFactorization())) ==
          "Newton with a BroydenUpdate needs a factorising linear method other than KroneckerFactorization: the update acts on the solve of a dense or sparse factor"
    @test message(() -> BroydenUpdate(0)) == "BroydenUpdate needs a memory m ≥ 1"
end

@testset "the state build refuses a factor in another precision and a LeastSquaresProblem, $T" for T in REAL_ELTYPES
    F = NonlinearProblem((y, x, p) -> (y .= x .^ 2 .- 2; nothing))
    other = T === Float64 ? Float32 : Float64
    m = Newton(; reuse = 3, update = BroydenUpdate(2), linsolve = LUFactorization(other))
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate needs a factor in the element type of the iterate" init_state(
        m, F, ones(T, 3), nothing)
    m_static = Newton(; reuse = 3, update = BroydenUpdate(2), linsolve = LUFactorization(other))
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate needs a factor in the element type of the iterate" init_state(
        m_static, NonlinearProblem((x, p) -> x .^ 2 .- 2), ones(SVector{3, T}), nothing)
    lsq = LeastSquaresProblem((y, x, p) -> (y .= [x[1] - 1, x[1] + 1, x[2]]; nothing), nothing, 3)
    m_qr = Newton(; reuse = 3, update = BroydenUpdate(2), linsolve = QRFactorization())
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate takes no LeastSquaresProblem" init_state(
        m_qr, lsq, ones(T, 2), nothing)
    lsq1 = LeastSquaresProblem((x, p) -> SVector(x - 1, x + 1), nothing, 2)
    @test_throws r"^ArgumentError: Newton with a BroydenUpdate takes no LeastSquaresProblem" init_state(
        m_qr, lsq1, one(T), nothing)
    # the same method in the precision of the iterate builds its state
    m_T = Newton(; reuse = 3, update = BroydenUpdate(2), linsolve = LUFactorization(T))
    @test init_state(m_T, F, ones(T, 3), nothing).pairs.count == 0
end

@testset "the R1 state with the pairs as NTuples is isbits, $T" for T in REAL_ELTYPES
    m = Newton(; reuse = 3, update = BroydenUpdate(2))
    prob = NonlinearProblem((x, p) -> x .^ 3 .- 2)
    state = init_state(m, prob, SVector{3, T}(1, 2, 3), nothing)
    @test isbitstype(typeof(state))
    @test state.pairs.u isa NTuple{2, SVector{3, T}}
    @test state.pairs.s isa NTuple{2, SVector{3, T}}
    xs, Fs, infos, state = run_steps(m, prob, SVector{3, T}(1, 2, 3), 2)
    @test isbitstype(typeof(state))
    @test state.pairs.count == 2
    # a `Number` iterate holds its pairs as numbers
    state₁ = init_state(m, NonlinearProblem((x, p) -> x^3 - 2), one(T), nothing)
    @test isbitstype(typeof(state₁))
    @test state₁.pairs.u isa NTuple{2, T}
end

@testset "a Number iterate meets the secant condition, $T" for T in ELTYPES
    R = real(T)
    a, b₀ = T(3) + (T <: Complex ? im * T(1) / 2 : zero(T)), T(4)
    prob = NonlinearProblem((x, p) -> a * x - 1)
    m = Newton(; jacobian = (x, p) -> b₀, reuse = 3, update = BroydenUpdate(2))
    xs, Fs, infos, state = run_steps(m, prob, one(T), 1)
    s = xs[2] - xs[1]
    y = Fs[2] - Fs[1]
    Hy = GS.apply_pairs!!(state.pairs, GS.ldiv!!(nothing, state.fac, y))
    @test state.pairs.count == 1
    @test abs(Hy - s) <= 8 * eps(R) * (abs(s) + abs(y / b₀))
end

@testset "the six-argument constructor and the Int32 step methods take the new parameter" begin
    m = Newton(; reuse = 3, update = BroydenUpdate(2))
    m6 = Newton(m.linsolve, m.globalization, m.jacobian, m.refinement, m.regularization,
        m.reuse)
    @test m6.update === nothing
    @test m6 === Newton(; reuse = 3)
    m9 = Newton(m.linsolve, m.globalization, m.jacobian, m.refinement, m.regularization,
        m.reuse, m.best, m.keep_factor, m.update)
    @test m9 === m
    @test Adapt.adapt(GS.ToReal{Float32}(), m).update === BroydenUpdate(2)
    @test GS.lift_scalar_operator(Newton(; jacobian = (x, p) -> 1.0, reuse = 3,
        update = BroydenUpdate(2))).update === BroydenUpdate(2)
    @test typeof(m).parameters[end] === BroydenUpdate{2}
    @test typeof(m6).parameters[end] === Nothing
    # both methods reach the `Int32` method of the step, whose signature names six parameters
    prob = NonlinearProblem((y, x, p) -> (y .= x .^ 2 .- 2; nothing))
    x = ones(3)
    methods_reached = map((m, m6)) do mm
        st = init_state(mm, prob, x, nothing)
        which(GS.newton_step!!,
            Tuple{typeof(st), typeof(mm), typeof(prob), typeof(x), typeof(x), Nothing})
    end
    @test methods_reached[1] === methods_reached[2]
    sig = Base.unwrap_unionall(methods_reached[1].sig)
    newton_type = Base.unwrap_unionall(sig.parameters[3])
    @test newton_type.parameters[6] === Int32
    # the signature leaves `best` and the update free
    @test newton_type.parameters[7] isa TypeVar
    @test newton_type.parameters[8] isa TypeVar
    st = init_state(m, prob, x, nothing)
    @test which(GS.refresh_diagnostics!!,
        Tuple{typeof(st), typeof(m), LeastSquaresProblem, typeof(x), typeof(x), Nothing}) ===
          which(GS.refresh_diagnostics!!,
        Tuple{typeof(st), typeof(m6), LeastSquaresProblem, typeof(x), typeof(x), Nothing})
end

@kernel function broyden_solve_kernel!(codes, xs, prob, method, state₀, opt, @Const(x₀))
    i = @index(Global, Linear)
    @inbounds begin
        x, st, _ = GS.solve!!(state₀, prob, x₀, method, nothing, opt)
        codes[i] = st.code
        xs[i] = x
    end
end

@testset "a solve with the update inside a KernelAbstractions kernel on CPU, $T" for T in REAL_ELTYPES
    A, _, b, x₀ = linear_system(T)
    As, bs = SMatrix{3, 3}(A), SVector{3}(b)
    prob = NonlinearProblem((x, p) -> As * x + x .^ 3 - bs)
    m = Adapt.adapt(GS.ToReal{T}(),
        Newton(; reuse = 3, update = BroydenUpdate(2), globalization = Backtracking()))
    opt = Options(T)
    q₀ = SVector{3}(x₀ ./ 4)
    state₀ = init_state(m, prob, q₀, nothing)
    x_host, st_host, _ = GS.solve!!(state₀, prob, q₀, m, nothing, opt)
    @test st_host.code == SUCCESS
    @test x_host != q₀
    codes = Vector{GS.ReturnCode}(undef, 2)
    xs = Vector{SVector{3, T}}(undef, 2)
    broyden_solve_kernel!(KA_BACKEND)(codes, xs, prob, m, state₀, opt, q₀; ndrange = 2)
    synchronize(KA_BACKEND)
    @test all(==(st_host.code), codes)
    @test all(x -> isapprox(x, x_host; rtol = 8 * eps(T)), xs)
end

allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "the R1 solve with the update is inferred and allocation-free, and the R3 solve allocates nothing, $T" for T in REAL_ELTYPES
    A, _, b, x₀ = linear_system(T)
    As, bs = SMatrix{3, 3}(A), SVector{3}(b)
    prob = NonlinearProblem((x, p) -> As * x + x .^ 3 - bs)
    m = Newton(; reuse = 3, update = BroydenUpdate(2))
    opt = Options(T)
    q₀ = SVector{3}(x₀ ./ 4)
    @test (@inferred solve(prob, q₀, m; options = opt)) isa Tuple
    m_T = Adapt.adapt(GS.ToReal{T}(), m)
    state = init_state(m_T, prob, q₀, nothing)
    run(s) = GS.solve!!(s, prob, q₀, m_T, nothing, opt)
    @test (@inferred run(state)) isa Tuple
    # under `--check-bounds=yes` a bounds check lets the local arrays of `StaticLU` escape to the
    # heap, so the check is skipped where a run forces `--check-bounds=yes`
    if Base.JLOptions().check_bounds != 1
        @test allocations(run, state) == 0
    end
    @test allocations(collect, [1, 2]) > 0
    JET.@test_opt run(state)
    # R3: a solve on the state that `init_state` built, from the same start each time
    Ad, bd = Matrix(A), Vector(b)
    prob3 = NonlinearProblem((y, x, p) -> (mul!(y, Ad, x); y .+= x .^ 3 .- bd; nothing))
    start = x₀ ./ 4
    state3 = init_state(m_T, prob3, start, nothing)
    x3 = similar(x₀)
    run3(s) = GS.solve!!(s, prob3, copyto!(x3, start), m_T, nothing, opt)
    _, st3, _ = run3(state3)
    @test st3.code == SUCCESS
    @test st3.iterations > 3
    if Base.JLOptions().check_bounds != 1
        @test allocations(run3, state3) == 0
    end
end
