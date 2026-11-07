# The reference solve of the vendor fast-path tests, `own_kernel_ldiv!` of
# `test/helpers/own_kernel.jl`, against the host solve `A \ b` on a `JLArray`.
#
# The CUDA agreement rows run only by hand on the Spark (decision 6b), so this file is where the
# reference they take their numbers from is checked in an ordinary run, and on a machine without
# a device. The bound is `4n · eps(real(TF)) · κ∞(A)`: one backward-stable solve gives
# `n · eps · κ∞`, and the rows assert `8n · eps · κ∞` against the vendor solve beside it.

using LinearAlgebra: cond, norm
using Test

using GeometricSolvers: LUFactorization, NonlinearProblem, QRFactorization, SVDFactorization
using GeometricSolvers: dense_factor, ka_factorize_lu!, ka_factorize_qr!, ka_factorize_svd!

include("../helpers/matrix.jl")
include("../helpers/own_kernel.jl")

own_kernel_noop(r, y, p) = nothing

# The own kernel factorisation of the matrix `dA` in `fac`, for the method `m`.
function own_kernel_factorize!(fac, dA::AbstractMatrix, ::LUFactorization)
    ka_factorize_lu!(fac, dA)
end
function own_kernel_factorize!(fac, dA::AbstractMatrix, ::QRFactorization)
    ka_factorize_qr!(fac, dA)
end
function own_kernel_factorize!(fac, dA::AbstractMatrix, ::SVDFactorization)
    ka_factorize_svd!(fac, dA)
end

# The relative error of `own_kernel_ldiv!` against the host solve, and its bound.
function own_kernel_row(::Type{T}, n::Int, m) where {T}
    A, b = cuda_well_conditioned(T, n)
    dA = JLArray(A)
    db = JLArray(b)
    dx = JLArray(zeros(T, n))
    fac = dense_factor(m, NonlinearProblem(dx, own_kernel_noop), dx, nothing)
    fac = own_kernel_factorize!(fac, dA, m)
    own_kernel_ldiv!(dx, fac, db)
    x = A \ b
    F = real(eltype(fac.factors))
    return norm(Array(dx) - x, Inf) / norm(x, Inf), 4 * n * Float64(eps(F)) * cond(A, Inf)
end

# The mixed rows: a `Float32` factorisation of the element types whose real type is `Float64`,
# where the factor and the solve are in another element type.
const MIXED_METHODS = (LUFactorization(Float32), QRFactorization(Float32))
const MIXED_ELTYPES = filter(T -> real(T) === Float64, ELTYPES)

@testset "own_kernel_ldiv! agrees with the host solve on JLArray" begin
    @testset "$m, T = $T, n = $n" for m in MIXED_METHODS, T in MIXED_ELTYPES, n in (1, 16)
        rel, bound = own_kernel_row(T, n, m)
        @test rel <= bound
    end

    @testset "SVD, T = $T" for T in REAL_ELTYPES
        for n in (1, 16)
            rel, bound = own_kernel_row(T, n, SVDFactorization())
            @test rel <= bound
        end
    end
end
