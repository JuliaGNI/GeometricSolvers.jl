# Left preconditioning: the outer method runs on the residual `rˡ(y) = y - N(y)` of an inner
# method `N`, so every line-search trial of the outer costs one inner solve and its merit is
# `‖rˡ‖²` (`Brune:2015`, `CaiLi:2011`, `Dolean:2016`). With `N` the restricted additive Schwarz
# map of `src/composition/ras.jl` it is RASPEN, whose Jacobian is exact from the block factors of
# the last sweep; with any other `N` its derivative is a central difference of `rˡ`, one inner step
# at each of `x ± h v`.

"""
    LeftPreconditioned(outer, inner)

The left-preconditioned method of a [`NonlinearProblem`](@ref): the method `outer` runs on
the residual ``r^\\ell(y) = y - N(y)`` of the inner map ``N``, one [`step!!`](@ref) of `inner` from
`(y, F(y))`, instead of on `F` itself. Every trial point of `outer`'s globalization therefore
costs one inner step, and every merit value it reads is ``\\|r^\\ell\\|_2^2``. The zeros of
``r^\\ell`` are the fixed points of ``N``.

`LeftPreconditioned(Newton(), RAS(partition))` is RASPEN: the merit is on the restricted additive
Schwarz map, and the Jacobian of ``r^\\ell`` is the exact form of [`raspen_jvp!`](@ref) built from
the block factors of the last sweep (`Dolean:2016`, eq. 3.17). RASPEN needs converged local solves:
a `RAS` with a positive tolerance whose local solve does not converge returns `MAXITERS`, and a
`LeftPreconditioned` step that meets it, at its iterate or at the trial point where its outer step
reports `NONFINITE`, returns `MAXITERS` too, which ends the solve. A `RAS` with
a `balance_threshold` is refused at `init_state` with an `ArgumentError`, because a skipped sweep
returns its input and so reads as a zero of ``r^\\ell`` at a point that is not a root. A
[`Newton`](@ref) with [`GMRES`](@ref) as `outer` is refused at `init_state` with an
`ArgumentError`, because it solves a `NonlinearProblem` and `outer` solves ``r^\\ell``; as `inner`
it is taken. `outer` may be a combinator, such as [`Repeat`](@ref) or [`Multiplicative`](@ref),
whose members step on ``r^\\ell``; with a `RAS` inner, every member of an [`Additive`](@ref)
after the first costs one more sweep, at the iterate, because the RASPEN Jacobian reads the block
factors of the last sweep. A [`Deflated`](@ref) problem is refused at `init_state` with an
`ArgumentError`, because its stopping test reads the deflation factor from a `Newton` state. With
an `inner` other than a `RAS` the derivative of
``r^\\ell`` is a central difference, one inner step at each of ``x \\pm h v``, which
[`prepare_ad`](@ref) prepares at `init_state` whatever its globalization reads. With a
[`RAS`](@ref) inner that preparation is the RASPEN form, so every sweep forms the block factor at
the local iterate it returns even for a globalization that reads no slope: measured on the
5-block P8 problem at one inner iteration per block, a `LeftPreconditioned(Picard(), RAS)` under
[`Static`](@ref) makes 10 block-Jacobian calls per sweep against plain `RAS`'s 5.

The [`solve!!`](@ref) status of a `LeftPreconditioned` solve is the one of the loop: its `fnorm` is
``\\|F(x)\\|`` of the global residual at the returned iterate, and its convergence tests apply to
that residual. `init_state` raises the `ArgumentError` of an inner [`RAS`](@ref) for a problem
without block residuals.
"""
struct LeftPreconditioned{O, I}
    outer::O
    inner::I
end

Adapt.@adapt_structure LeftPreconditioned

"""
    LeftPreconditionedState

The state of a [`LeftPreconditioned`](@ref) method: the [`LeftPreconditionedProblem`](@ref) it
runs on, the outer method's state, and the buffer of the global residual that a step returns. The
outer state's residual workspace holds ``r^\\ell``; the problem's inner state holds the workspace
of the inner step, and its counters are the ones [`ras_counts`](@ref) reports.
"""
struct LeftPreconditionedState{P, S, FX}
    problem::P
    outer_state::S
    Fx::FX
end

"""
    RASPenPreparation

The AD preparation of a [`LeftPreconditioned`](@ref) method whose inner state is a
[`RASState`](@ref): the RAS state, whose block factors the RASPEN Jacobian product reads.
[`jacobian!!`](@ref) and [`jvp!!`](@ref) apply the exact form of
[`raspen_jacobian!`](@ref) and [`raspen_jvp!`](@ref).
"""
struct RASPenPreparation{A}
    ras::A
end

"""
    LeftPreconditionedAD

The AD preparation of a [`LeftPreconditioned`](@ref) method whose inner state is not a
[`RASState`](@ref): the iterate and residual buffers of the two points of a central difference,
and the direction buffer that seeds one column at a time. [`jvp!!`](@ref) evaluates ``r^\\ell`` at
``x \\pm h v`` and forms ``(r^\\ell(x + hv) - r^\\ell(x - hv)) / 2h``, and [`jacobian!!`](@ref)
takes one column per call of it.

A central difference, because one step of an inner method is not `ForwardDiff.Dual`-generic: a
method carries the real type of the working precision in its type, so a dual iterate reaches the
`StepInfo` of its globalization and the right-hand-side scaling of its linear factor in the wrong
type.
"""
struct LeftPreconditionedAD{XP, XM, RP, RM, E}
    xp::XP
    xm::XM
    rp::RP
    rm::RM
    e::E
end

@inline function init_state(m::LeftPreconditioned, prob::NonlinearProblem, x, p)
    x isa Union{Number, StaticArray} && throw(ArgumentError(
        "LeftPreconditioned takes a mutable vector iterate: an R1 iterate (a Number or an SVector) has no workspace for an inner step"))
    m.outer isa Newton && m.outer.linsolve isa GMRES &&
        throw(ArgumentError(
            "LeftPreconditioned does not take a Newton with GMRES as its outer method: a Newton with GMRES solves a NonlinearProblem, and the outer method solves the preconditioned residual; use a Newton with a factorization, such as the default LUFactorization(), or a Picard as the outer method, or put the Newton with GMRES in the inner method"))
    m.inner isa RAS && m.inner.balance_threshold !== nothing &&
        throw(ArgumentError(
            "LeftPreconditioned does not take a RAS with a balance_threshold: a skipped sweep returns its input, which reads as a zero of the preconditioned residual at a point that is not a root"))
    inner_state = init_state(m.inner, prob, x, p)
    lprob = LeftPreconditionedProblem(prob, m.inner, inner_state, buffer(x), Ref(SUCCESS))
    outer_state = init_state(m.outer, lprob, x, p)
    return LeftPreconditionedState(lprob, outer_state, buffer(x))
end

# A `Deflated` problem converges on the inner residual `‖F‖ = ‖M F‖ / M`, whose factor `M` its
# stopping test reads from the Jacobian preparation of a `Newton` state, and a `Picard` or a `RAS`
# inner method has no step on it. The reduced problem is the one a `Deflated(StationarityProblem)`
# solve builds its method's state for.
function init_state(
        m::LeftPreconditioned, ::Deflated{<:Union{NonlinearProblem, ReducedStatProb}}, x, p)
    throw(ArgumentError(
        "LeftPreconditioned does not take a Deflated problem: the stopping test of a Deflated problem reads its deflation factor from a Newton state; solve a Deflated problem with Newton()"))
end

@inline residual_buffer(state::LeftPreconditionedState) = state.Fx

# `rˡ(y) = y - N(y)`: the residual is the difference between the iterate and one inner step from
# it. The inner state is started at `y` on each evaluation, so successive evaluations do not share
# a merit anchor, and its buffers are reused. The two methods are the buffer and the value form of
# the residual seam, each stated on its own slot so that neither is ambiguous against it.
@inline function residual!!(y::AbstractArray, prob::LeftPreconditionedProblem, x, p)
    return preconditioned_residual(y, prob, x, p)
end

@inline function residual!!(::Nothing, prob::LeftPreconditionedProblem, x, p)
    return preconditioned_residual(nothing, prob, x, p)
end

# The counters of a method state that reports them, and the restart that keeps them: the inner map
# of a `LeftPreconditioned` solve is started again at every evaluation, and its counters belong to
# the solve, so they survive that restart.
@inline state_counts(st) = nothing
@inline restart_with_counts(st, ::Nothing) = st

@inline function preconditioned_residual(y, prob::LeftPreconditionedProblem, x, p)
    st = prob.state
    xw = copy!!(prob.work, x)
    Fx = residual!!(residual_buffer(st), prob.prob, xw, p)
    counts = state_counts(st)
    st = start!!(st, prob.inner, prob.prob, xw, Fx, p, rnorm(Fx))
    st = restart_with_counts(st, counts)
    x₊, _, _, info = step!!(st, prob.inner, prob.prob, xw, Fx, p)
    prob.code[] = info.code
    # a failed inner step returns its input, so `x - x₊` would read as an exact zero of `rˡ`: the
    # residual is non-finite instead, which makes the outer's merit `NaN` and its line search
    # reject the trial
    ok = info.code == SUCCESS
    if y === nothing
        r = map!!(-, nothing, x, x₊)
        return ok ? r : r * real(eltype(x))(NaN)
    end
    ok && return map!!(-, y, x, x₊)
    return fill!(y, real(eltype(y))(NaN))
end

@inline jacobian_prototype(::LeftPreconditionedProblem) = nothing

# A preconditioned problem has no analytic Jacobian of `rˡ`, so a `Picard` outer prepares its
# derivative through `prepare_ad` here, at `init_state`, whatever its globalization reads: with a
# `RAS` inner that sets the RASPEN flag of the inner state, and every sweep then forms the block
# factor at the local iterate it returns.
@inline function picard_jvpprep(m::Picard, prob::LeftPreconditionedProblem, Fx, x, p)
    return Fx === nothing ? nothing : prepare_ad(AutoForwardDiff(), prob, Fx, x, p)
end

# `Picard` on the preconditioned problem: the state and the step of `picard.jl`, whose
# `init_state` signature names `NonlinearProblem` alone so that the generic `Reparametrized` and
# `StationarityProblem` methods of `init_state` keep their precedence.
@inline init_state(m::Picard, prob::LeftPreconditionedProblem, x, p) = picard_state(m, prob, x, p)

function prepare_ad(
        backend::AutoForwardDiff, prob::LeftPreconditionedProblem, Fx::AbstractArray,
        x::AbstractArray, p)
    # the preparation reads the inner state's block factors, so the state that is a `RASState` —
    # the state of a [`RAS`](@ref), and of any member that forwards to one — is a RASPEN inner map,
    # and the sweeps whose factors it reads form the block factor at the iterate they return
    if prob.state isa RASState
        prob.state.blocks.raspen = true
        return RASPenPreparation(prob.state)
    end
    return LeftPreconditionedAD(
        similar(x), similar(x), similar(Fx), similar(Fx), similar(x))
end

function jvp!!(Jv, prep::RASPenPreparation, prob::LeftPreconditionedProblem, x, v, p)
    return raspen_jvp!(Jv, prep.ras, prob.prob, x, v, p)
end

function jacobian!!(J, prep::RASPenPreparation, prob::LeftPreconditionedProblem, x, p)
    return raspen_jacobian!(J, prep.ras, prob.prob, x, p)
end

# The RASPEN product reads the block factors and local iterates of the last sweep, so a member of
# an `Additive` outer that steps from `x` after another member's trial points needs the sweep at
# `x` again. A central difference evaluates `rˡ` itself and reads no such state.
@inline function resync_point!(prob::LeftPreconditionedProblem, F, x, p)
    prob.state isa RASState && residual!!(F, prob, x, p)
    return nothing
end

# The central difference of `rˡ` along `v`, written into `col`: one inner step at each of
# `x ± h v`, with ``h = \\sqrt[3]{\\varepsilon} \\max(1, \\|x\\|) / \\|v\\|``, the step whose
# truncation and rounding errors balance.
function difference_jvp!(
        col, prep::LeftPreconditionedAD, prob::LeftPreconditionedProblem, x, v, p)
    R = real(eltype(x))
    vnorm = rnorm(v)
    if iszero(vnorm)
        fill!(col, zero(eltype(col)))
        return col
    end
    h = cbrt(eps(R)) * max(one(R), rnorm(x)) / vnorm
    prep.xp .= x .+ h .* v
    prep.xm .= x .- h .* v
    rp = residual!!(prep.rp, prob, prep.xp, p)
    rm = residual!!(prep.rm, prob, prep.xm, p)
    col .= rp
    col .-= rm
    col ./= 2h
    return col
end

function jvp!!(Jv, prep::LeftPreconditionedAD, prob::LeftPreconditionedProblem, x, v, p)
    return difference_jvp!(Jv, prep, prob, x, v, p)
end

function jacobian!!(J, prep::LeftPreconditionedAD, prob::LeftPreconditionedProblem, x, p)
    for j in axes(J, 2)
        unit_direction!(prep.e, j)
        difference_jvp!(view(J, :, j), prep, prob, x, prep.e, p)
    end
    return J
end

# `e` set to the `j`-th unit vector in place, without a scalar write, so that a device direction
# buffer takes it.
@inline function unit_direction!(e, j)
    fill!(e, zero(eltype(e)))
    @views e[j:j] .= one(eltype(e))
    return e
end

@inline function start!!(
        state::LeftPreconditionedState, m::LeftPreconditioned, prob, x, Fx, p, fnorm₀)
    lprob = state.problem
    # the inner map is started again at every evaluation of `rˡ`, so its counters survive those
    # restarts and a solve would otherwise report every earlier solve too: this is the hook that
    # gives the counters of the solve that `start!!` opens
    inner = start!!(lprob.state, m.inner, prob, x, Fx, p, rnorm(Fx))
    lprob = LeftPreconditionedProblem(
        lprob.prob, lprob.inner, inner, lprob.work, lprob.code)
    Fxˡ = residual!!(residual_buffer(state.outer_state), lprob, x, p)
    outer_state = start!!(
        state.outer_state, m.outer, lprob, x, Fxˡ, p, rnorm(Fxˡ))
    return LeftPreconditionedState(lprob, outer_state, state.Fx)
end

@inline function step!!(
        state::LeftPreconditionedState, m::LeftPreconditioned, prob, x, Fx, p)
    lprob = state.problem
    R = real(eltype(x))
    Fxˡ = residual!!(residual_buffer(state.outer_state), lprob, x, p)
    fnormˡ = rnorm(Fxˡ)
    if !isfinite(fnormˡ)
        # an inner step that failed makes `rˡ` non-finite, and the outer's Jacobian would then
        # read the block factors of a sweep that formed none: the failure is this step's status,
        # which ends the solve. An inner local solve that did not converge keeps its `MAXITERS`,
        # so that the status names the cause; every other failure is `NONFINITE`
        code = lprob.code[] == MAXITERS ? MAXITERS : NONFINITE
        return x, Fx, state, StepInfo{R}(code, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    if iszero(fnormˡ)
        # `x` is a fixed point of the inner map, so the outer solves `rˡ = 0` here: its step is of
        # zero length, and the iterate and its global residual are unchanged
        return x, Fx, state,
        StepInfo{R}(SUCCESS, rnorm(Fx), zero(R), Int32(0), false, false)
    end
    x₊, _, outer_state, info = step!!(state.outer_state, m.outer, lprob, x, Fxˡ, p)
    F₊ = residual!!(state.Fx, prob, x₊, p)
    # an outer step whose own trial point found an unconverged local solve sees a non-finite
    # `rˡ` there and reports `NONFINITE`: the inner `MAXITERS` names the cause instead
    code = info.code == NONFINITE && lprob.code[] == MAXITERS ? MAXITERS : info.code
    info₊ = StepInfo{R}(
        code, rnorm(F₊), info.stepnorm, info.failures, info.promoted, info.limited)
    return x₊, F₊, LeftPreconditionedState(lprob, outer_state, state.Fx), info₊
end

@inline function finish!!(
        state::LeftPreconditionedState, m::LeftPreconditioned, prob, x, Fx,
        status::SolverStatus{R}) where {R}
    x_out, st, outer_state = finish!!(
        state.outer_state, m.outer, state.problem, x, Fx, status)
    return x_out, st, LeftPreconditionedState(state.problem, outer_state, state.Fx)
end

"""
    ras_counts(state::LeftPreconditionedState)

The counters of the [`RAS`](@ref) state of a [`LeftPreconditioned`](@ref) method's inner map, as
[`ras_counts`](@ref) reports them. A method whose inner map is not a `RAS` raises an
`ArgumentError`.
"""
ras_counts(state::LeftPreconditionedState) = ras_counts(state.problem.state)

"""
    ras_counts(state::MultiplicativeState)

The counters of the [`RAS`](@ref) member of a [`Multiplicative`](@ref) method, as
[`ras_counts`](@ref) reports them. A method with no `RAS` member raises an `ArgumentError`.
"""
ras_counts(state::MultiplicativeState) = _ras_member_counts(state.states)

function _ras_member_counts(states::Tuple)
    first(states) isa RASState && return ras_counts(first(states))
    return _ras_member_counts(Base.tail(states))
end

function _ras_member_counts(::Tuple{})
    throw(ArgumentError(
        "ras_counts needs a state with a RAS method: a RAS state, a LeftPreconditioned state whose inner method is a RAS, or a Multiplicative state with a RAS member"))
end

function ras_counts(state)
    throw(ArgumentError(
        "ras_counts needs a state with a RAS method: a RAS state, a LeftPreconditioned state whose inner method is a RAS, or a Multiplicative state with a RAS member"))
end
