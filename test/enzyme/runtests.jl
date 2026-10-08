# `AutoEnzyme()` through DifferentiationInterface, in an environment of its own.
#
# §3.2: the package codes against DI, so a back end other than ForwardDiff costs it no line of
# source. Enzyme is the back end that shows this, and it is also the back end whose releases break
# most often. Enzyme is therefore not a dependency of `test/Project.toml`: it lives here, in its
# own environment, driven by its own workflow (`.github/workflows/Enzyme.yml`), which is not a
# required status check. An Enzyme break then fails that job alone, and merges go on.
#
# Usage, from the repository root:
#   julia --startup-file=no --project=test/enzyme test/enzyme/runtests.jl
#
# What is checked is the claim of §3.2 and nothing more: the same `prepare_ad`, `jacobian!!` and
# `jvp!!` give, with `AutoEnzyme()`, what ForwardDiff gives. The device path is not touched —
# Enzyme does not differentiate Metal kernels, which is the reason the chunked mode exists.

using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using Test

using GeometricSolvers: DIJacobian, jacobian!!, jvp!!, prepare_ad

include("../helpers/matrix.jl")
include("../helpers/adproblems.jl")

const N_AD = 7

# Forward mode with runtime activity. A plain `AutoEnzyme()` raises
# `EnzymeRuntimeActivityError` on every residual here: the parameters are a `Constant` context,
# and Enzyme's static activity analysis cannot prove that a constant array broadcast into an
# active result is non-differentiable. That is a property of Enzyme's analysis and not of this
# package — the same residuals differentiate through ForwardDiff — and the documented remedy is
# runtime activity, so that is what the back end carries here. Recorded as an issue of its own
# in `KNOWN_ISSUES.md`.
const BACKEND = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))

@testset "AutoEnzyme() through DifferentiationInterface" begin
    @testset "$T" for T in REAL_ELTYPES
        prob = StubProblem(Coupled(cyclic_perm(Array, N_AD)))
        x, p, r = ad_inputs(Array, T, N_AD)

        prep = prepare_ad(BACKEND, prob, r, x, p)
        @test prep isa DIJacobian

        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        # Enzyme and ForwardDiff do the same arithmetic on this residual in a different order, so
        # the comparison carries a tolerance where the ForwardDiff tests are exact.
        @test J ≈ forwarddiff_jacobian(prob.F, x, p) rtol = 8 * eps(T)

        Jv = similar(x)
        v = T[(i + 2) / (N_AD + 1) for i in 1:N_AD]
        jvp!!(Jv, prep, prob, x, v, p)
        @test Jv ≈ J * v rtol = 8 * eps(T)

        # and the parameters are a constant here too: a replacement takes effect, and the
        # parameters are not differentiated
        scaled = StubProblem(Scaled())
        sprep = prepare_ad(BACKEND, scaled, r, x, p)
        jacobian!!(J, sprep, scaled, x, p)
        @test J ≈ forwarddiff_jacobian(Scaled(), x, p) rtol = 8 * eps(T)
        q = 3 .* p
        jacobian!!(J, sprep, scaled, x, q)
        @test J ≈ forwarddiff_jacobian(Scaled(), x, q) rtol = 8 * eps(T)
    end
end
