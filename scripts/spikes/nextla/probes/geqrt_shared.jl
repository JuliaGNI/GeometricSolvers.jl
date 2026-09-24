# `geqrt!` on Metal fails on private buffers, because `larfb!` calls a host `BLAS.trmm!` through
# `LinearAlgebra.mul!` (NextLA 0.2.3, `src/larfb.jl`). With `SharedStorage` arrays the host can
# read the buffers and `geqrt!` returns. This probe repeats that run 5 times for F32 and CF32 at
# the census size and prints the factorisation error ‖A0 - Q R‖/‖A0‖ and the orthogonality error
# ‖Q'Q - I‖, with Q built on the host from the reflectors.
#
# Usage, from the repository root:
#   JULIA_LOAD_PATH="@:$PWD/test/gpu/metal:@stdlib" julia --startup-file=no \
#       --project=scripts/spikes/nextla scripts/spikes/nextla/probes/geqrt_shared.jl

using LinearAlgebra
using Metal
using NextLA: NextLA
using Random

const m, n, ib = 45, 37, 8
const k = min(m, n)
const S = Metal.SharedStorage

shared(A::AbstractArray{T, N}) where {T, N} = MtlArray{T, N, S}(A)

function run_once(rng, ::Type{T}) where {T}
    A0 = randn(rng, T, m, n)
    A = shared(A0)
    Tm, tau, work = shared(zeros(T, ib, k)), shared(zeros(T, k)), shared(zeros(T, ib * n))
    NextLA.geqrt!(m, n, ib, A, Tm, tau, work)
    Metal.synchronize()
    Aq, tq = ComplexF64.(Array(A)), ComplexF64.(Array(tau))
    Q = Matrix{ComplexF64}(I, m, m)
    for i in 1:k
        v = zeros(ComplexF64, m)
        v[i] = 1
        v[(i + 1):m] .= Aq[(i + 1):m, i]
        Q -= (Q * v) * (tq[i] * v')
    end
    R = triu(Aq[1:k, :])
    fact = norm(ComplexF64.(A0) - Q[:, 1:k] * R) / norm(A0)
    orth = norm(Q' * Q - I)
    return fact, orth
end

rng = Xoshiro(0x5eed)
for (name, T) in (("F32", Float32), ("CF32", ComplexF32)), run in 1:5

    fact, orth = run_once(rng, T)
    println("$name run $run: fact $(round(fact; sigdigits = 2)), orth $(round(orth; sigdigits = 2))")
end
println("Packages: NextLA ", pkgversion(NextLA), ", Metal ", pkgversion(Metal))
