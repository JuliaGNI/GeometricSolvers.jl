# The vector fields of six ordinary differential equations `ẏ = f(y)`, in ODE form, with their
# Jacobians, and the stage systems of implicit Runge–Kutta tableaux built on them.
#
# The fields are those of the implicit-midpoint rows of the benchmark (`benchmark/problems.jl`),
# written anew in ODE form, because the GeometricProblems problems there are IODEs and HODEs and
# GeometricProblems is no test dependency. Each field `f!(dy, y)` writes into `dy` and allocates
# nothing on an iterate of its own element type; its Jacobian `jac!(J, y)` is analytic where the
# field is short, and ForwardDiff's elsewhere.
#
# The stage system of a tableau `A` and a step `h` from `yₙ` is `G(Z) = Z - h (A ⊗ I) f(yₙ + Z)`
# in the stacked increments `Z = (Z₁, …, Z_s)`, `Zᵢ = Yᵢ - yₙ`, as `NonlinearProblem` with
# `stage_matrix` takes it; the parameters are `p = (h = h, y = yₙ)`.

using ForwardDiff: ForwardDiff
using LinearAlgebra: SymTridiagonal, eigvals

# ── the vector fields ──────────────────────────────────────────────────────────

# The harmonic oscillator `q̇ = p`, `ṗ = -q`.
function oscillator!(dy, y)
    dy[1] = y[2]
    dy[2] = -y[1]
    return dy
end
function oscillator_jac!(J, y)
    fill!(J, zero(eltype(J)))
    J[1, 2] = one(eltype(J))
    J[2, 1] = -one(eltype(J))
    return J
end

# The pendulum `θ̇ = ω`, `ω̇ = -sin θ`.
function pendulum!(dy, y)
    dy[1] = y[2]
    dy[2] = -sin(y[1])
    return dy
end
function pendulum_jac!(J, y)
    fill!(J, zero(eltype(J)))
    J[1, 2] = one(eltype(J))
    J[2, 1] = -cos(y[1])
    return J
end

# The Lotka–Volterra system in two species, `u̇ = u (v - 2)`, `v̇ = v (1 - u)`.
function lotka_volterra_2d!(dy, y)
    dy[1] = y[1] * (y[2] - 2)
    dy[2] = y[2] * (1 - y[1])
    return dy
end
function lotka_volterra_2d_jac!(J, y)
    J[1, 1] = y[2] - 2
    J[1, 2] = y[1]
    J[2, 1] = -y[2]
    J[2, 2] = 1 - y[1]
    return J
end

# The Lotka–Volterra system in four species, `ẏᵢ = yᵢ (rᵢ + Σⱼ aᵢⱼ yⱼ)`.
const LV4_R = (1.0, 0.72, 1.53, 1.27)
const LV4_A = ((-1.0, -1.09, -1.52, 0.0), (0.0, -1.0, -0.44, -1.36),
    (-2.33, 0.0, -1.0, -0.47), (-1.21, -0.51, -0.35, -1.0))
function lotka_volterra_4d!(dy, y)
    for i in 1:4
        acc = LV4_R[i] + zero(eltype(y))
        for j in 1:4
            acc += LV4_A[i][j] * y[j]
        end
        dy[i] = y[i] * acc
    end
    return dy
end
function lotka_volterra_4d_jac!(J, y)
    for i in 1:4
        acc = LV4_R[i] + zero(eltype(y))
        for j in 1:4
            acc += LV4_A[i][j] * y[j]
        end
        for j in 1:4
            J[i, j] = y[i] * LV4_A[i][j] + (i == j ? acc : zero(acc))
        end
    end
    return J
end

# The double pendulum in angles and angular velocities, with unit masses, lengths and gravity.
function double_pendulum!(dy, y)
    θ₁, θ₂, ω₁, ω₂ = y[1], y[2], y[3], y[4]
    Δ = θ₁ - θ₂
    den = 3 - cos(2Δ)
    dy[1] = ω₁
    dy[2] = ω₂
    dy[3] = (-3sin(θ₁) - sin(θ₁ - 2θ₂) - 2sin(Δ) * (ω₂^2 + ω₁^2 * cos(Δ))) / den
    dy[4] = 2sin(Δ) * (2ω₁^2 + 2cos(θ₁) + ω₂^2 * cos(Δ)) / den
    return dy
end
double_pendulum_jac!(J, y) = ForwardDiff.jacobian!(J, double_pendulum!, similar(y), y)

# The periodic Toda lattice of four particles, `q̇ = p`, `ṗᵢ = e^{-(qᵢ - qᵢ₋₁)} - e^{-(qᵢ₊₁ - qᵢ)}`.
const TODA_N = 4
function toda!(dy, y)
    N = TODA_N
    for i in 1:N
        dy[i] = y[N + i]
        qm = y[mod1(i - 1, N)]
        qp = y[mod1(i + 1, N)]
        dy[N + i] = exp(-(y[i] - qm)) - exp(-(qp - y[i]))
    end
    return dy
end
toda_jac!(J, y) = ForwardDiff.jacobian!(J, toda!, similar(y), y)

# A complex linear field `ẏ = L y` for a complex element type.
const COMPLEX_L = ((0.0 + 1.0im, 0.3, 0.0), (-0.3, 0.0 + 2.0im, 0.1 - 0.2im),
    (0.0, -0.1 - 0.2im, -0.5 + 0.5im))
function complex_linear!(dy, y)
    for i in 1:3
        acc = zero(eltype(dy))
        for j in 1:3
            acc += COMPLEX_L[i][j] * y[j]
        end
        dy[i] = acc
    end
    return dy
end
function complex_linear_jac!(J, y)
    for i in 1:3, j in 1:3

        J[i, j] = COMPLEX_L[i][j]
    end
    return J
end

# Each field as `(name, f!, jac!, y₀)`, with `y₀` in `Float64`.
const ODE_FIELDS = [
    ("HarmonicOscillator", oscillator!, oscillator_jac!, [0.5, 0.0]),
    ("Pendulum", pendulum!, pendulum_jac!, [acos(0.4), 0.0]),
    ("LotkaVolterra2d", lotka_volterra_2d!, lotka_volterra_2d_jac!, [2.0, 1.0]),
    ("LotkaVolterra4d", lotka_volterra_4d!, lotka_volterra_4d_jac!, [0.3, 0.4, 0.2, 0.1]),
    ("DoublePendulum", double_pendulum!, double_pendulum_jac!, [0.7, -0.4, 0.2, 0.1]),
    ("TodaLattice", toda!, toda_jac!, [0.3, -0.1, 0.2, -0.4, 0.0, 0.1, -0.2, 0.1])
]

const COMPLEX_FIELD = ("ComplexLinear", complex_linear!, complex_linear_jac!,
    [1.0 + 0.5im, -0.2 + 0.1im, 0.3 - 0.4im])

# ── the tableaux ───────────────────────────────────────────────────────────────

# The collocation matrix of the nodes `c`: `A[i, j] = ∫₀^{cᵢ} ℓⱼ(t) dt` with `ℓⱼ` the Lagrange
# polynomials of the nodes, from `A V = Ĉ`, `V[j, k] = cⱼ^{k-1}`, `Ĉ[i, k] = cᵢ^k / k`, in
# `BigFloat` and rounded to `Float64` once.
function collocation_matrix(c)
    s = length(c)
    cb = big.(c)
    V = [cb[j]^(k - 1) for j in 1:s, k in 1:s]
    Ĉ = [cb[i]^k / k for i in 1:s, k in 1:s]
    return Float64.(Ĉ / V)
end

# The Gauss–Legendre nodes on `[0, 1]`, from the eigenvalues of the Jacobi matrix of the Legendre
# polynomials (Golub–Welsch).
function gauss_nodes(s)
    β = [k / sqrt(4k^2 - 1) for k in 1:(s - 1)]
    x = eigvals(SymTridiagonal(zeros(s), β))
    return (sort(x) .+ 1) ./ 2
end

gauss_tableau(s) = collocation_matrix(gauss_nodes(s))

# Radau IIA with three stages, of order five.
radau_iia3_tableau() = collocation_matrix([(4 - sqrt(6)) / 10, (4 + sqrt(6)) / 10, 1.0])

# The implicit midpoint rule, Gauss with one stage.
midpoint_tableau() = fill(0.5, 1, 1)

# Alexander's two-stage SDIRK of order two, whose `A` has the double eigenvalue `γ` with a single
# eigenvector and is not diagonalisable.
function sdirk2_tableau()
    γ = 1 - sqrt(2) / 2
    return [γ 0.0; 1-γ γ]
end

const STAGE_TABLEAUX = [
    ("Gauss s = 2", gauss_tableau(2)), ("Gauss s = 3", gauss_tableau(3)),
    ("Radau IIA s = 3", radau_iia3_tableau())]

# ── the stage system ───────────────────────────────────────────────────────────

# The stage residual `G(Z) = Z - h (A ⊗ I) f(yₙ + Z)` of the field `f!`, with `yₙ = p.y` and
# `h = p.h`. The buffers serve an iterate of their element type; a dual iterate of AD takes
# buffers of its own.
struct StageSystem{F, M, V}
    f!::F
    A::M
    Y::V
    fY::V
end

function StageSystem(f!, A, ::Type{T}, n) where {T}
    return StageSystem(f!, T.(A), zeros(T, n), zeros(T, n))
end

function (S::StageSystem)(G, Z, p)
    n = length(p.y)
    s = size(S.A, 1)
    Y, fY = eltype(Z) === eltype(S.Y) ? (S.Y, S.fY) : (similar(Z, n), similar(Z, n))
    copyto!(G, Z)
    for j in 1:s
        for k in 1:n
            Y[k] = p.y[k] + Z[(j - 1) * n + k]
        end
        S.f!(fY, Y)
        for i in 1:s
            a = p.h * S.A[i, j]
            for k in 1:n
                G[(i - 1) * n + k] -= a * fY[k]
            end
        end
    end
    return G
end

# The Jacobian of the field at `yₙ = p.y`, the simplified-Newton Jacobian of the stage solve, in
# the form `jac!(J, Z, p)`; `calls` counts its calls.
struct FrozenJacobian{J}
    jac!::J
    calls::Base.RefValue{Int}
end
FrozenJacobian(jac!) = FrozenJacobian(jac!, Ref(0))

function (F::FrozenJacobian)(J, Z, p)
    F.calls[] += 1
    F.jac!(J, p.y)
    return J
end

# The stage problem of the field `f!` and the tableau `A` in the element type `T`: the problem,
# its counting Jacobian, and `n`.
function stage_problem(f!, jac!, A, ::Type{T}, n) where {T}
    jac = FrozenJacobian(jac!)
    prob = NonlinearProblem(StageSystem(f!, A, T, n); stage_matrix = A, stage_jacobian = jac)
    return prob, jac
end
