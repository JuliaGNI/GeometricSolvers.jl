# The R1 value methods of the storage functions of §4.1 — check 1 of §13.L-a, and the storage half
# of check 7.
#
# Seven of the nine have their value method in `src/base/bang.jl`; `factorize!!` and `ldiv!!` have
# theirs there too, for a `Number` `J`, and forward to the `1 × 1` factors that part I builds. Each
# returns what §4.1's table promises, each is `@inferred`, and each allocates nothing through a
# function barrier. The seam then runs inside a KernelAbstractions kernel on `CPU()`, which is
# where a host-only operation or a heap buffer would show.

using ADTypes: AutoForwardDiff
using JET: JET, @test_opt
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
using StaticArrays: MMatrix, MVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: StaticLU, StaticQR, buffer, factorize!!, jacobian!!, jvp!!, ldiv!!,
                        map!!, mul!!, residual!!, returncode, update_iterate!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_BANG = 4

# An `MMatrix` of the element type `T`, column by column, with no zero entry.
function mmatrix(::Type{T}, n::Int) where {T}
    return MMatrix{n, n, T}(T[(i + 2j) / (n + 3) for i in 1:n for j in 1:n])
end

"`R1Scaled` with its analytic Jacobian: the problem the residual tests read."
r1_scaled_problem() = NonlinearProblem(R1Scaled(), r1_scaled_jacobian)

# The allocation assertions go through a function barrier whose arguments all have concrete
# types, and which calls the function once before it measures: at testset scope Julia 1.11 boxes
# the captured values and the assertion fails although the code allocates nothing.
#
# A value iterate of a `Number` or an `SVector` is `isbits`, so the compiler keeps the result in
# registers; an `MVector` is a value too, but `x + α d` and `map` on a mutable array return a new
# mutable array, which the heap holds, so nothing is asserted about its cost (`KNOWN_ISSUES.md`
# K7). `buffer` returns no array at all, and is asserted for a static matrix as well.
buffer_allocations(x::X) where {X} = (buffer(x); @allocated buffer(x))
function residual_allocations(y::Y, prob::P, x::X, p::Q) where {Y, P, X, Q}
    (residual!!(y, prob, x, p); @allocated residual!!(y, prob, x, p))
end
function update_allocations(x::X, α::A, d::D) where {X, A, D}
    (update_iterate!!(nothing, x, α, d); @allocated update_iterate!!(nothing, x, α, d))
end
map_allocations(x::X) where {X} = (map!!(-, nothing, x); @allocated map!!(-, nothing, x))
function map2_allocations(x::X, z::Z) where {X, Z}
    (map!!(+, nothing, x, z); @allocated map!!(+, nothing, x, z))
end
function mul_allocations(a::A, b::B) where {A, B}
    (mul!!(nothing, a, b); @allocated mul!!(nothing, a, b))
end
function factorize_allocations(m::M, jac::J) where {M, J}
    (factorize!!(nothing, m, jac); @allocated factorize!!(nothing, m, jac))
end
function ldiv_allocations(fac::F, r::R) where {F, R}
    (ldiv!!(nothing, fac, r); @allocated ldiv!!(nothing, fac, r))
end

"the control: the barrier sees an allocation where there is one"
collect_allocations(v::V) where {V} = (collect(v); @allocated collect(v))

# On a Julia it does not support, JET loads empty stubs that throw. JET 0.12 exports
# `JET_AVAILABLE` to say so; JET 0.9 and 0.10 name the same flag `JET_LOADABLE`.
const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

@testset "every storage function has a value method, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_BANG)
    v = MVector(x)
    m = mmatrix(T, N_BANG)
    s = SMatrix(m)
    prob = r1_scaled_problem()
    α = T(2)

    # `buffer` has no slot for a value iterate, whatever its shape: an `SVector`, an `MVector`
    # (which counts as a value, §4.3), an `MMatrix` — a static matrix, which is never an iterate,
    # and still gets no heap slot (§4.1) — and a `Number`. The second form is what `init` calls,
    # and it is `nothing` too: the R1 state spends no register on a buffer.
    @test buffer(x) === nothing
    @test buffer(v) === nothing
    @test buffer(m) === nothing
    @test buffer(one(T)) === nothing
    @test buffer(x, T, N_BANG) === nothing
    @test buffer(one(T), T) === nothing

    # `residual!!` is `F(x, p)` for every value iterate; `update_iterate!!` is `x + α d`; `map!!`
    # is `map`; `mul!!` is the product of a `SMatrix` with a `SVector` or a `SMatrix` (§4.1)
    @test residual!!(nothing, prob, x, p) == prob.F(x, p)
    @test residual!!(nothing, prob, v, p) == prob.F(v, p)
    @test residual!!(nothing, prob, one(T), T(3)) == T(3)
    @test update_iterate!!(nothing, x, α, x) === x + α * x
    @test update_iterate!!(nothing, v, α, v) == v + α * v
    @test map!!(-, nothing, x) == map(-, x)
    @test map!!(-, nothing, v) == map(-, v)
    @test map!!(+, nothing, x, x) == map(+, x, x)
    @test mul!!(nothing, s, x) == s * x
    @test mul!!(nothing, s, s) == s * s

    # and every one of them infers, the `MVector` result included: it is a new mutable array
    @test @inferred(buffer(x)) === nothing
    @test @inferred(buffer(m)) === nothing
    @test @inferred(buffer(one(T))) === nothing
    @test @inferred(residual!!(nothing, prob, x, p)) == prob.F(x, p)
    @test @inferred(residual!!(nothing, prob, v, p)) == prob.F(v, p)
    @test @inferred(update_iterate!!(nothing, x, α, x)) === x + α * x
    @test @inferred(update_iterate!!(nothing, v, α, v)) == v + α * v
    @test @inferred(map!!(-, nothing, x)) == map(-, x)
    @test @inferred(map!!(-, nothing, v)) == map(-, v)
    @test @inferred(mul!!(nothing, s, x)) == s * x
    @test @inferred(mul!!(nothing, s, s)) == s * s
end

# The value-returning calls of this testset are asserted for a value iterate of the working
# element type, where the result is `isbits`, and `buffer` is asserted for a static matrix as
# well, because it returns no array at all and so cannot allocate whatever the bounds-check flag
# is.
@testset "the value methods allocate nothing, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_BANG)
    m = mmatrix(T, N_BANG)
    s = SMatrix(m)
    prob = r1_scaled_problem()
    @test collect_allocations([one(T)]) > 0
    @test buffer_allocations(x) == 0
    @test buffer_allocations(one(T)) == 0
    @test buffer_allocations(m) == 0
    @test residual_allocations(nothing, prob, x, p) == 0
    @test update_allocations(x, T(2), x) == 0
    @test map_allocations(x) == 0
    @test map2_allocations(x, x) == 0
    @test mul_allocations(s, x) == 0
    @test mul_allocations(s, s) == 0
end

@testset "a Number J factorises through part I's 1 × 1 factors, $T" for T in ELTYPES
    for (method, F) in ((LUFactorization(), StaticLU), (QRFactorization(), StaticQR))
        fac = @inferred factorize!!(nothing, method, T(2))
        @test fac isa F{1, T, 1}
        @test isbits(fac)
        @test returncode(fac) == SUCCESS
        @test @inferred(ldiv!!(nothing, fac, T(4))) == T(2)
        # Under `--check-bounds=yes`, which `Pkg.test()` forces up to Julia 1.12, a bounds check
        # lets the local `MMatrix` of the static kernel and the local `MVector` of its backsolve
        # escape to the heap, so a run that forces it does not check these two — the same guard,
        # and the same reason, as `test/linear/static/lu.jl`.
        if Base.JLOptions().check_bounds != 1
            @test factorize_allocations(method, T(2)) == 0
            @test ldiv_allocations(fac, T(4)) == 0
        end
    end

    # a tensor-core marker has no `1 × 1` factor either, exactly as it has none for a matrix
    for marker in (TF32(), FP16(), BF16())
        @test_throws MethodError factorize!!(nothing, LUFactorization(marker), T(2))
    end
end

@testset "hostile inputs: a non-finite value is a value, not a throw, $T" for T in ELTYPES
    prob = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)
    p = SVector{N_BANG, T}(ntuple(_ -> one(T), Val(N_BANG)))
    unit = SVector{N_BANG, T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(N_BANG)))

    for a in r1_hostile_values(T)
        x = SVector{N_BANG, T}(fill(a, N_BANG))
        expected = prob.J(x, p) * unit
        # A non-finite value is a value, not a throw, and a `NaN` is compared with `isequal`,
        # because it is not `==` to itself. The unit direction is what makes the comparison of the
        # dual pass exact: the dense product is then one column of the Jacobian, with no sum to
        # round.
        @test isequal(residual!!(nothing, prob, x, p), prob.F(x, p))
        @test isequal(jvp!!(nothing, nothing, prob, x, unit, p), expected)
        @test jvp!!(nothing, AutoForwardDiff(), prob, x, unit, p) isa SVector{N_BANG, T}
        # The dual pass has an arithmetic of its own where `J` holds a non-finite entry that meets
        # the exact zero of the direction's other entries: `0 * Inf` is `NaN` in the dense product
        # and the dual computes the exact zero instead. Both are values; where `J` is finite they
        # agree.
        if all(isfinite, expected)
            @test isequal(jvp!!(nothing, AutoForwardDiff(), prob, x, unit, p), expected)
        end
    end
end

@testset "JET: no runtime dispatch in the value methods, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_BANG)
    prob = r1_scaled_problem()
    if JET_WORKS
        @test_opt buffer(x)
        @test_opt residual!!(nothing, prob, x, p)
        @test_opt update_iterate!!(nothing, x, T(2), x)
        @test_opt map!!(-, nothing, x)
        @test_opt factorize!!(nothing, LUFactorization(), T(2))
        @test_opt ldiv!!(nothing, factorize!!(nothing, QRFactorization(), T(2)), T(4))
    else
        @test_skip JET_WORKS  # issue #8
    end
end

# Check 7, the storage half: every call above runs inside a kernel, on values only. A `nothing`
# buffer slot is part of what a kernel carries — it has no storage and no register — so the tuple
# the kernel writes out holds it, and `mul!!` takes a static matrix there as it does on the host.
@kernel function seam_kernel!(out, prob, @Const(X), @Const(P), @Const(S), α)
    i = @index(Global)
    x = X[i]
    p = P[i]
    b = buffer(x)
    r = residual!!(nothing, prob, x, p)
    y = update_iterate!!(nothing, x, α, r)
    d = map!!(-, nothing, r)
    fac = factorize!!(nothing, LUFactorization(), one(eltype(x)) * 2)
    s = ldiv!!(nothing, fac, d[1])
    c = mul!!(nothing, S[i], y)
    out[i] = (y[1], s, d[1], b, c)
end

@testset "the storage seam inside a KernelAbstractions kernel on CPU(), $T" for T in ELTYPES
    backend = KA_BACKEND
    prob = r1_scaled_problem()
    X = [r1_inputs(T, N_BANG)[1] * T(i) for i in 1:3]
    P = [r1_inputs(T, N_BANG)[2] for i in 1:3]
    S = [diagonal_matrix(P[i] * T(i + 1)) for i in 1:3]
    out = Vector{Tuple{T, T, T, Nothing, SVector{N_BANG, T}}}(undef, 3)
    α = T(2)
    seam_kernel!(backend)(out, prob, X, P, S, α; ndrange = 3)
    KernelAbstractions.synchronize(backend)
    expected = map(1:3) do i
        r = P[i] .* X[i]
        y = X[i] + α * r
        return (y[1], -r[1] / 2, -r[1], nothing, S[i] * y)
    end
    # The four fields that move data are exact, and are asserted bit for bit: `y[1]`, `s` and
    # `d[1]` are `x + α d`, `-r[1] / 2` and `-r[1]` — plain `*`, `+` and a division by two — and
    # `b` is the `nothing` buffer slot of a value iterate, which the tuple carries out of the
    # kernel.
    #
    # The product is not exact in that sense. StaticArrays builds `A * B` for a static matrix and
    # a static vector from a chain of `muladd`s (`matrix_multiply_add.combine_products`), and
    # Julia's `muladd` reaches the backend as `fmul contract` and `fadd contract`: whether the
    # compiler contracts the chain into an FMA or leaves the two roundings is its own choice, and
    # the choice is not the same in every compilation context. On the Julia 1.11 floor — the
    # `[compat]` lower bound — the kernel's product and this closure's then differ by up to 2 ulp
    # for `ComplexF32` alone, where 1.13.2 makes the same choice in both and agrees to the bit.
    # `test/gpu/runtests.jl` carries a tolerance for the same reason. So the product is
    # compared at the ulp level of the working precision, and the four fields above it are what a
    # wrong `r`, `y`, `α` or factor has to move.
    for i in 1:3
        @test out[i][1:4] == expected[i][1:4]
        @test all(isapprox.(out[i][5], expected[i][5]; rtol = 8 * eps(real(T))))
    end
end
