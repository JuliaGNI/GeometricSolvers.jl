# Scalar line functions for the line-search tests. A line search sees a line function only
# through `φ(lf, α)` and `φ′(lf, α)`; the nonlinear solvers build theirs on a problem.

using GeometricSolvers
using GeometricSolvers: φ, φ′, norm2, rdot

# A line function from two closures, for the fixtures ported from `SimpleSolvers`.
struct Line{F, D}
    f::F
    d::D
end
GeometricSolvers.φ(l::Line, α) = l.f(α)
GeometricSolvers.φ′(l::Line, α) = l.d(α)

# A line function that records every step at which it is evaluated.
struct Watched{L, R}
    lf::L
    atφ::Vector{R}
    atφ′::Vector{R}
end
Watched(lf, ::Type{R}) where {R} = Watched(lf, R[], R[])
GeometricSolvers.φ(w::Watched, α) = (push!(w.atφ, α); φ(w.lf, α))
GeometricSolvers.φ′(w::Watched, α) = (push!(w.atφ′, α); φ′(w.lf, α))
evaluated(w::Watched) = length(w.atφ) + length(w.atφ′)

# The six test functions of Moré and Thuente (1994), §5, with their constants `c₁ = μ` and
# `c₂ = η` and the initial steps `10^-3, 10^-1, 10^1, 10^3`. They are isbits, so a kernel can
# hold them. `kind` 1–6 selects the function.
struct MoreThuente{R}
    kind::Int32
    β₁::R
    β₂::R
end

const MT_BETA = ((2.0, 0.0), (0.004, 0.0), (0.01, 0.0), (0.001, 0.001), (0.01, 0.001),
    (0.001, 0.01))
const MT_C = ((0.001, 0.1), (0.1, 0.1), (0.1, 0.1), (0.001, 0.001), (0.001, 0.001),
    (0.001, 0.001))
const MT_STEPS = (1e-3, 1e-1, 1e1, 1e3)

MoreThuente(R, kind) = MoreThuente{R}(Int32(kind), R(MT_BETA[kind][1]), R(MT_BETA[kind][2]))

mt_γ(β) = sqrt(1 + β^2) - β

function GeometricSolvers.φ(lf::MoreThuente{R}, α::R) where {R}
    β, β₂ = lf.β₁, lf.β₂
    if lf.kind == 1
        -α / (α^2 + β)
    elseif lf.kind == 2
        (α + β)^5 - 2 * (α + β)^4
    elseif lf.kind == 3
        l = R(39)
        φ₀ = α ≤ 1 - β ? 1 - α : α ≥ 1 + β ? α - 1 : (α - 1)^2 / (2β) + β / 2
        φ₀ + 2 * (1 - β) / (l * R(π)) * sin(l * R(π) * α / 2)
    else
        mt_γ(β) * sqrt((1 - α)^2 + β₂^2) + mt_γ(β₂) * sqrt(α^2 + β^2)
    end
end

function GeometricSolvers.φ′(lf::MoreThuente{R}, α::R) where {R}
    β, β₂ = lf.β₁, lf.β₂
    if lf.kind == 1
        (α^2 - β) / (α^2 + β)^2
    elseif lf.kind == 2
        5 * (α + β)^4 - 8 * (α + β)^3
    elseif lf.kind == 3
        l = R(39)
        dφ₀ = α ≤ 1 - β ? -one(R) : α ≥ 1 + β ? one(R) : (α - 1) / β
        dφ₀ + (1 - β) * cos(l * R(π) * α / 2)
    else
        mt_γ(β) * (α - 1) / sqrt((1 - α)^2 + β₂^2) + mt_γ(β₂) * α / sqrt(α^2 + β^2)
    end
end

# Isbits merits for a kernel: a cubic c₀ + c₁α + c₂α² + c₃α³, the kink 1 - α below a and
# 1 - a + 2(α - a) above it, and the cliff 1 + 1000α with the lying slope -2.
struct Cubic{R}
    c::NTuple{4, R}
end
GeometricSolvers.φ(l::Cubic, α) = evalpoly(α, l.c)
GeometricSolvers.φ′(l::Cubic, α) = evalpoly(α, (l.c[2], 2l.c[3], 3l.c[4]))
struct Kink{R}
    a::R
end
GeometricSolvers.φ(l::Kink, α) = α < l.a ? 1 - α : 1 - l.a + 2 * (α - l.a)
GeometricSolvers.φ′(l::Kink, α) = α < l.a ? -one(α) : 2one(α)
# φ rises for every α > 0 while φ′(0) = -2: the lower end of a Bisection bracket stays at 0
struct Rising end
GeometricSolvers.φ(::Rising, α) = α > 0 ? 1 + α : one(α)
GeometricSolvers.φ′(::Rising, α) = α > 0 ? one(α) : -2one(α)
struct Cliff end
GeometricSolvers.φ(::Cliff, α) = α > 0 ? 1 + 1000α : one(α)
GeometricSolvers.φ′(::Cliff, α) = -2one(α)

# The merit φ(α) = ‖r(x + α d)‖² of r(x) = x² - 2, componentwise, along its Newton direction
# d = -r(x) / 2x: an array line function for R3, on any array type.
struct NewtonLine{V}
    x::V
    d::V
end
square_residual(y) = y .^ 2 .- 2
function NewtonLine(x::AbstractVector)
    NewtonLine(x, -square_residual(x) ./ (2 .* x))
end
GeometricSolvers.φ(l::NewtonLine, α) = norm2(square_residual(l.x .+ α .* l.d))
function GeometricSolvers.φ′(l::NewtonLine, α)
    y = l.x .+ α .* l.d
    2 * rdot(square_residual(y), 2 .* y .* l.d)
end
