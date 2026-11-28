# The dependence of the consistent rows of known issue K149 on the summation order of the
# reductions: `J = u₂ (u₁ + u₂)ᵀ + δ u₁ u₁ᵀ`, `b = u₁` in the frame of `Xoshiro(7)`, `restart = 20`,
# `maxiter = 200`, `tol = 1e-8`, for a Float32 basis on Float64 with δ = 0.3 and for the Float16
# rows of the tests. `L = 0` keeps Base's reductions in the two dot products of the step; `L ≥ 1`
# replaces every reduction of the reduced-basis path (`arnoldi_dot`, `norm2`, the projections of
# the CGS2 step) and the sums of the operator by a sum over `L` interleaved accumulators, combined
# in order, as a SIMD loop of `L` lanes sums. Each line gives the outcome, `‖d‖`, the true residual
# and the breakdowns: how many there were, how many had a pivot at the round-off of the basis, and
# how many read `J` as singular. The script measures the code it loads: run it on a commit to read
# that commit's rows. Run from the package root, with the test environment:
#   julia --startup-file=no --project=test scripts/gmres_summation_order.jl

using GeometricSolvers
using GeometricSolvers: GMRES, gmres_solve!, init_gmres_workspace, rnorm
using LinearAlgebra: qr, norm
using Random: Xoshiro
using Printf: @printf

const LANES = Ref(0)
function lanesum(f, n, ::Type{S}) where {S}
    L = LANES[]
    acc = zeros(S, L)
    for i in 1:n
        acc[mod1(i, L)] += f(i)
    end
    s = zero(S)
    for l in 1:L
        s += acc[l]
    end
    return s
end
opsum(x) = LANES[] == 0 ? sum(x) : lanesum(i -> x[i], length(x), eltype(x))

const RECORD = Tuple{Bool, Bool}[]
@eval GeometricSolvers begin
    function arnoldi_dot(a, b)
        Main.LANES[] == 0 &&
            return sum(Broadcast.instantiate(Broadcast.broadcasted((x, y) -> conj(x) * y, a, b)))
        return Main.lanesum(i -> conj(a[i]) * b[i], length(b), eltype(b))
    end
    function norm2(a)
        R = real(eltype(a))
        Main.LANES[] == 0 && return isempty(a) ? zero(R) : R(sum(abs2, a))
        return Main.lanesum(i -> abs2(a[i]), length(a), R)
    end
    function cgs2_reduced_step!(ws::GMRESWorkspace{T, R}, j::Int) where {T, R}
        n = length(ws.w)
        @views ws.H[1:j, j] .= zero(T)
        for _ in 1:2
            for i in 1:j
                v = @view ws.V[:, i]
                ws.hd[i] = Main.LANES[] == 0 ? sum(k -> conj(T(v[k])) * ws.w[k], 1:n) :
                           Main.lanesum(k -> conj(T(v[k])) * ws.w[k], n, T)
            end
            for i in 1:j
                ws.H[i, j] += ws.hd[i]
            end
            for k in 1:n, i in 1:j

                ws.w[k] -= T(ws.V[k, i]) * ws.hd[i]
            end
        end
        return R(rnorm(ws.w))
    end
    @inline function singular_breakdown(
            ws::GMRESWorkspace{T, R, V}, a, w₀, gj, β) where {T, R, V}
        pivot = basis_breakdown(ws, a, w₀)
        singular = pivot && gj > sqrt(R(eps(real(eltype(V))))) * β
        push!(Main.RECORD, (pivot, singular))
        return singular
    end
end

const ROWS = ((Float32, false, 0.3), (Float32, true, 0.3), (Float16, false, 0.3),
    (Float16, true, 0.1), (Float16, true, 0.3))
for L in (0, 1, 2, 3, 4, 8, 16), (TB, mgs, δ) in ROWS

    LANES[] = L
    empty!(RECORD)
    T = Float64
    Q = T.(Matrix(qr(randn(Xoshiro(7), 50, 50)).Q))
    u₁ = Q * [fill(T(0.2), 25); zeros(T, 25)]
    u₂ = Q * [zeros(T, 25); fill(T(0.2), 25)]
    s = u₁ .+ u₂
    op = (Jv, v) -> (Jv .= u₂ .* opsum(s .* v) .+ T(δ) .* u₁ .* opsum(u₁ .* v); Jv)
    b = copy(u₁)
    ws = init_gmres_workspace(GMRES(; maxiter = 200, basis_eltype = TB), zeros(T, 50))
    d, info = gmres_solve!(ws, op, nothing, b, rnorm(b), 1e-8, Int32(200), Int32(20); mgs)
    @printf("L = %2d, %s basis, δ = %g, mgs = %-5s: converged %-5s after %3d, ‖d‖ %.3g, residual %.3g ‖b‖, breakdowns %d (pivot at round-off %d, singular %d)\n",
        L, TB, δ, mgs, info.converged, info.iterations, norm(d),
        rnorm(b .- op(similar(d), d)) / rnorm(b), length(RECORD), count(first, RECORD),
        count(last, RECORD))
end
@printf("Julia %s, %s, check-bounds %d\n", VERSION, Sys.MACHINE,
    Base.JLOptions().check_bounds)
