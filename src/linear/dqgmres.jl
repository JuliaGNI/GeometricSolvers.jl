# R5 matrix-free Newton–Krylov with fixed storage: the right-preconditioned DQGMRES of Saad and Wu
# (1996, doi:10.1002/(SICI)1099-1506(199607/08)3:4<329::AID-NLA86>3.0.CO;2-8), a linear method of
# `Newton` beside `GMRES`.
#
# Each new Krylov vector is orthogonalised against the last `m = min(memory, n)` basis vectors
# only, by modified Gram–Schmidt, so the Hessenberg matrix is banded with `m + 1` nonzeros per
# column. Its Givens rotations act on that band alone, and the solution is updated at every
# iteration along a direction `p_j = (z_j - Σ r_ij p_i) / r_jj` of the last `m` directions. A
# direction is stored as `p_j 2^e_j`, with `2^e_j ≤ |r_jj| < 2^(e_j+1)`, at the scale of `z_j`, so
# that no vector carries the reciprocal scale of the system; the band and the solution take the
# power of two back in their coefficients. The scaling is exact, so in the normal range the
# iterates are those of the unscaled directions. The basis keeps `m + 1` columns, the directions
# `m`, and both are rings: the vector of iteration `j` overwrites the one that the band no longer
# reaches. With `m` at least the iteration count, the orthogonalisation is complete and the
# iterate is that of unrestarted GMRES with modified Gram–Schmidt. The Hessenberg band and the rotations are on the host, every vector of length `n`
# is in the storage of the iterate, the loop allocates nothing after `init_dqgmres_workspace`, and
# it indexes no entry of a vector.

"""
    DQGMRES(; maxiter = 30, memory = 20, forcing = EisenstatWalkerChoice2(),
              preconditioner = nothing)

A matrix-free linear method of [`Newton`](@ref): the right-preconditioned DQGMRES of Saad and Wu
(1996, doi:10.1002/(SICI)1099-1506(199607/08)3:4<329::AID-NLA86>3.0.CO;2-8) on the
Jacobian–vector product of the problem, with the forcing term `forcing`
([`EisenstatWalkerChoice2`](@ref) by default, [`EisenstatWalkerChoice1`](@ref) or
[`FixedForcing`](@ref)) as the relative tolerance of each Newton step's inner solve, and the right
preconditioner `preconditioner`, which [`refresh!`](@ref) and [`apply_preconditioner!`](@ref) serve
as for [`GMRES`](@ref).

# Fields

- `maxiter::Int32`: the cap on the inner iterations of one Newton step, `≥ 1`. An iteration takes
  one Jacobian–vector product and one application of the preconditioner. A step that reaches the
  cap without meeting the tolerance of its forcing term counts a step-rule failure.
- `memory::Int32`: the window `≥ 1` of the incomplete orthogonalisation. On an iterate of length
  `n` the window is `m = min(memory, n)`: each Krylov vector is orthogonalised against the last
  `m`, and the solve stores `m + 1` basis vectors and `m` directions of the length of the iterate,
  whatever the iteration count. Where `m` is at least the iteration count, the iterate is that of
  unrestarted GMRES.
- `forcing::F`: the rule that sets the relative tolerance ``\\eta_k`` of step `k`.
- `preconditioner::P`: the right preconditioner, `nothing` for none; the solve forms
  ``J M^{-1} u = -F`` and returns ``d = M^{-1} u``.

The residual norm that the solve reports and stops on is the bound
``\\sqrt{\\lceil (j + 1)/(m + 1) \\rceil}\\, |g_{j+1}|`` of the true ``\\|F + J d\\|`` after `j`
iterations, with ``g_{j+1}`` the last entry of the rotated right-hand side: any `m + 1`
consecutive basis vectors are orthonormal, so the basis has at most that norm. With a complete
orthogonalisation the bound is ``|g_{j+1}|``, the residual norm of GMRES. A Krylov vector of norm
zero ends the solve; where the rotated diagonal entry of the band is then zero too, the solve
reports a breakdown and no convergence, as `GMRES` does on a singular operator. A product that is
not finite ends the step with `NONFINITE`, and the iterate stays.

A `Newton` with `DQGMRES` takes the iterate and the problems of a `Newton` with `GMRES`, and the
`forcing` and `preconditioner` keywords of `Newton` replace those of the method. It takes no
`reuse`, no [`Refinement`](@ref), no [`LevenbergMarquardt`](@ref), no [`PseudoTransient`](@ref) and
no [`DogLeg`](@ref), whose trial loop reads the Arnoldi relation of a `GMRES` solve, and `solve`
refuses a `ForwardDiff.Dual` in `x₀` or `p`; each refusal raises an `ArgumentError`, and so do
`maxiter < 1` and `memory < 1`.
"""
struct DQGMRES{F <: AbstractForcingTerm, P} <: LinearMethod{Nothing}
    maxiter::Int32
    memory::Int32
    forcing::F
    preconditioner::P
    # `adapt` rebuilds the method through this constructor, inside a device kernel too.
    function DQGMRES{F, P}(maxiter::Int32, memory::Int32, forcing::F,
            preconditioner::P) where {F <: AbstractForcingTerm, P}
        maxiter >= Int32(1) || throw(ArgumentError("DQGMRES maxiter must be ≥ 1"))
        memory >= Int32(1) || throw(ArgumentError("DQGMRES memory must be ≥ 1"))
        return new{F, P}(maxiter, memory, forcing, preconditioner)
    end
end

function DQGMRES(; maxiter::Integer = 30, memory::Integer = 20,
        forcing::AbstractForcingTerm = EisenstatWalkerChoice2(), preconditioner = nothing)
    m = to_int32(maxiter, "DQGMRES maxiter must fit in an Int32")
    k = to_int32(memory, "DQGMRES memory must fit in an Int32")
    return DQGMRES{typeof(forcing), typeof(preconditioner)}(m, k, forcing, preconditioner)
end

# The positional form, which `adapt` rebuilds the method through.
@inline function DQGMRES(maxiter::Int32, memory::Int32, forcing::AbstractForcingTerm,
        preconditioner)
    return DQGMRES{typeof(forcing), typeof(preconditioner)}(
        maxiter, memory, forcing, preconditioner)
end

Adapt.@adapt_structure DQGMRES

# The workspace of one DQGMRES solve: the ring `V` of `m + 1` basis vectors and the ring `P` of `m`
# directions, the solution `d`, the preconditioned basis vector `z`, the product `w` and the
# right-hand side `bb`, in the storage of the iterate; on the host the band `c` of `m + 2` entries
# of the current Hessenberg column, rows `j - m` to `j + 1`, the rings `cs` and `sn` of the last
# `m` rotations, and the ring `pe` of the exponents `e_j` of the stored directions `p_j 2^e_j`; and
# the counters of a solve, as in `MINRESQLPWorkspace`.
mutable struct DQGMRESWorkspace{T, R <: Real, M, V, SR, ST}
    V::M
    P::M
    d::V
    z::V
    w::V
    bb::V
    c::ST
    cs::SR
    sn::ST
    pe::Vector{Int}
    f_calls::Int32
    jvp_calls::Int32
    iterations::Int32
    max_inner::Int32
    step::Int32
    last_eta::R
    prev_fnorm::R
    linres::R
end

"""
    init_dqgmres_workspace(g::DQGMRES, x::AbstractVector)

Allocate the workspace of a [`DQGMRES`](@ref) solve on an iterate like `x`: with the window
`m = min(g.memory, length(x))`, the basis `V` of `m + 1` columns and the directions `P` of `m`
columns in the storage of `x`, four vectors like `x`, and the host band, rotations and direction
exponents of the window, whose sizes no solve changes.
"""
function init_dqgmres_workspace(g::DQGMRES, x::AbstractVector{T}) where {T}
    R = real(T)
    n = length(x)
    m = max(min(Int(g.memory), n), 1)
    V = buffer(x, T, n, m + 1)
    Pm = buffer(x, T, n, m)
    vs = ntuple(_ -> buffer(x), 4)
    c = zeros(T, m + 2)
    cs = zeros(R, m)
    sn = zeros(T, m)
    pe = zeros(Int, m)
    return DQGMRESWorkspace{T, R, typeof(V), typeof(vs[1]), typeof(cs), typeof(sn)}(
        V, Pm, vs..., c, cs, sn, pe,
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), R(NaN), R(NaN), R(NaN))
end

"""
    dqgmres_solve!(ws::DQGMRESWorkspace, jvp_op!, P, b, bnorm, tol, maxiter) -> (d, GMRESInfo)

Solve `J d = b` with right-preconditioned DQGMRES in the workspace `ws`, and return the direction
`d = M⁻¹ u` in the slot `ws.d` and a [`GMRESInfo`](@ref). `jvp_op!(Jv, v)` writes `J v` into
`Jv`, `P` is the right preconditioner, `bnorm` is ``\\|b\\|_2``, and the solve stops where the
bound of the residual norm that [`DQGMRES`](@ref) states is at most `tol * bnorm`, or after
`maxiter` iterations.

The orthogonalisation takes the conjugated inner product of [`arnoldi_dot`](@ref), so a complex
iterate is supported, and the rotations are those of `gmres_solve!`. A Krylov vector of norm zero
ends the solve with `breakdown = true`; a rotated diagonal entry of zero leaves the solution of
the last iteration, whose residual bound the solve reports, with no convergence. A norm that is not
finite, `bnorm` included, ends the solve with a residual of `NaN` and no convergence. `b = 0` takes
no product.
"""
function dqgmres_solve!(ws::DQGMRESWorkspace{T, R}, jvp_op!, P, b, bnorm, tol,
        maxiter) where {T, R}
    m = size(ws.P, 2)
    β₀ = R(bnorm)
    atol = R(tol) * β₀
    d, z, w, c = ws.d, ws.z, ws.w, ws.c
    fill!(d, zero(T))
    j = 0
    iszero(β₀) && return d, GMRESInfo{R}(Int32(0), zero(R), β₀, true, false)
    # `β₀ = Inf` makes the tolerance `atol` infinite, which every residual would meet
    isfinite(β₀) || return d, GMRESInfo{R}(Int32(0), R(NaN), β₀, false, false)
    @views ws.V[:, 1] .= b ./ β₀
    g = T(β₀)
    resnorm = β₀
    converged = resnorm <= atol
    breakdown = false
    while !converged && j < maxiter
        j += 1
        vj = @view ws.V[:, mod1(j, m + 1)]
        apply_preconditioner!(z, P, vj)
        jvp_op!(w, z)
        # the band of column j: `c[k]` is row `j - m - 1 + k`, and rows below `j - m + 1` are zero
        fill!(c, zero(T))
        for i in max(1, j - m + 1):j
            vi = @view ws.V[:, mod1(i, m + 1)]
            h = arnoldi_dot(vi, w)
            c[i - j + m + 1] = h
            w .-= h .* vi
        end
        hj = R(rnorm(w))
        if !isfinite(hj)
            resnorm = R(NaN)
            break
        end
        c[m + 2] = hj
        # the last `m` rotations on the band, then the rotation that removes `c[m + 2]`
        for i in max(1, j - m):(j - 1)
            a = i - j + m + 1
            q = mod1(i, m)
            t = ws.cs[q] * c[a] + ws.sn[q] * c[a + 1]
            c[a + 1] = -conj(ws.sn[q]) * c[a] + ws.cs[q] * c[a + 1]
            c[a] = t
        end
        a = c[m + 1]
        radius = R(hypot(abs(a), hj))
        if iszero(radius)
            # the band leaves row j without a pivot: no direction, and the iterate stays
            breakdown = true
            break
        end
        q = mod1(j, m)
        cs = R(abs(a) / radius)
        sn = iszero(abs(a)) ? T(hj) / T(radius) : (a / T(abs(a))) * T(hj) / T(radius)
        rjj = cs * a + sn * hj
        gj = cs * g
        g = -conj(sn) * g
        # the next basis vector in the slot of `v_{j-m}`, which the band of column `j` did not
        # read; a vector of norm zero ends the solve below and is not formed
        iszero(hj) || (@views ws.V[:, mod1(j + 1, m + 1)] .= w ./ hj)
        # the direction `p_j 2^e_j` of iteration j in the slot of `p_{j-m} 2^e_{j-m}`, which it
        # reads last: `w` holds `z_j` minus the directions of the band that keep their slots
        w .= z
        for i in max(1, j - m + 1):(j - 1)
            k = mod1(i, m)
            w .-= ldexp2(c[i - j + m + 1], -ws.pe[k]) .* @view(ws.P[:, k])
        end
        pj = @view ws.P[:, q]
        e = norm_exponent(R(abs(rjj)))
        if j > m
            pj .= (w .- ldexp2(c[1], -ws.pe[q]) .* pj) ./ ldexp2(rjj, -e)
        else
            pj .= w ./ ldexp2(rjj, -e)
        end
        d .+= ldexp2(gj, -e) .* pj
        ws.cs[q] = cs
        ws.sn[q] = sn
        ws.pe[q] = e
        resnorm = sqrt(R(cld(j + 1, m + 1))) * R(abs(g))
        if iszero(hj)
            breakdown = true
            converged = resnorm <= atol
            break
        end
        converged = resnorm <= atol
    end
    return d, GMRESInfo{R}(Int32(j), resnorm, β₀, converged, breakdown)
end

# The counters of a solve, from the seed of `solve`, as for a `GMRESWorkspace`.
max_inner_iterations(ws::DQGMRESWorkspace) = ws.max_inner
f_calls(ws::DQGMRESWorkspace) = ws.f_calls
jvp_calls(ws::DQGMRESWorkspace) = ws.jvp_calls
