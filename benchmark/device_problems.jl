# The problems of the device rows: the inline `MVector` adapter that runs Y1's in-place residuals
# out of place on `SVector`s, the `isbits` midpoint residuals, the §10.E problem in R1 form (for
# the R2 throughput grid) and in R3 form (for the R3 and LU rows), the R1 size band of each
# backend, and the classifier of a launch that throws.
#
# It needs `problems.jl` included first, for `standard_rows`, `midpoint_rows`,
# `MIDPOINT_SCENARIOS`, `midpoint_problem` and `problem_kind`.

using GeometricSolvers
using LinearAlgebra: I, diagind, norm
using Random: Xoshiro
using StaticArrays: SMatrix, SVector, MVector

"""
    InlineAdapter(f!)

The out-of-place residual `F(x, p)` of an in-place residual `f!(y, x, p)` on an `SVector` `x`: it
copies `x` into an `MVector`, calls `f!` with `@inline` at the call site, and returns the result
as an `SVector`. An `MVector` that reaches a call which is not inlined escapes to the heap, which a
kernel allocates from a device heap or not at all; the inlined call keeps it on the stack.
"""
struct InlineAdapter{F}
    f!::F
end

@inline function (a::InlineAdapter)(x::SVector{N, S}, p) where {N, S}
    y = MVector{N, S}(undef)
    @inline a.f!(y, MVector{N, S}(x), p)
    return SVector{N, S}(y)
end

"""
    MidpointResidual{K}(fs, t̃, h, q₀, p₀)

The stage residual of `midpoint_system` in `problems.jl` for a problem of kind `K` (`:ode`,
`:iode` or `:hode`), with the initial state held as `SVector`s and GeometricProblems' parameters
passed as `p`: the residuals of `midpoint_system` capture host `Vector`s, which a kernel cannot
read. GeometricProblems' vector fields are called with `@inline`, so that the `MVector`s they
write stay off the heap.
"""
struct MidpointResidual{K, Fs, S, Q, P}
    fs::Fs
    t̃::S
    h::S
    q₀::Q
    p₀::P
end

function (r::MidpointResidual{:ode})(y, V, par)
    @inline r.fs.v(y, r.t̃, r.q₀ .+ (r.h / 2) .* SVector(V), par)
    y .-= V
    return nothing
end

function (r::MidpointResidual{:iode})(y, V, par)
    Q = r.q₀ .+ (r.h / 2) .* SVector(V)
    f = similar(y)
    @inline r.fs.ϑ(y, r.t̃, Q, SVector(V), par)
    @inline r.fs.f(f, r.t̃, Q, SVector(V), par)
    y .-= r.p₀ .+ (r.h / 2) .* f
    return nothing
end

function (r::MidpointResidual{:hode, Fs, S, <:SVector{D}})(y, X, par) where {Fs, S, D}
    V = SVector{D}(ntuple(i -> X[i], Val(D)))
    W = SVector{D}(ntuple(i -> X[D + i], Val(D)))
    v, f = MVector{D, eltype(y)}(undef), MVector{D, eltype(y)}(undef)
    @inline r.fs.v(v, r.t̃, r.q₀ .+ (r.h / 2) .* V, r.p₀ .+ (r.h / 2) .* W, par)
    @inline r.fs.f(f, r.t̃, r.q₀ .+ (r.h / 2) .* V, r.p₀ .+ (r.h / 2) .* W, par)
    for i in 1:D
        y[i] = v[i] - V[i]
        y[D + i] = f[i] - W[i]
    end
    return nothing
end

"""
    midpoint_residual(prob)

The [`MidpointResidual`](@ref) of the GeometricProblems problem `prob`.
"""
function midpoint_residual(prob)
    k = problem_kind(prob)
    q₀ = SVector{length(prob.ics.q)}(collect(prob.ics.q))
    p₀ = k == :ode ? nothing : SVector{length(prob.ics.p)}(collect(prob.ics.p))
    t̃ = prob.timespan[1] + prob.timestep / 2
    return MidpointResidual{k, typeof(prob.functions), typeof(t̃), typeof(q₀), typeof(p₀)}(
        prob.functions, t̃, prob.timestep, q₀, p₀)
end

"""
    device_rows(T)

Y1's 79 rows in the element type `T`, in Y1's order, each a named tuple of its labels `set`,
`problem` and `start`, its start `x0` as an `SVector`, its out-of-place residual `F`, an
[`InlineAdapter`](@ref), and its parameters `p`: `nothing` for a standard row, GeometricProblems'
parameters for a midpoint row, whose residual is a [`MidpointResidual`](@ref).
"""
function device_rows(::Type{T}) where {T}
    rows = Any[(; r.set, r.problem, r.start, x0 = SVector{length(r.x0)}(r.x0),
                   F = InlineAdapter(r.F), p = nothing)
               for r in standard_rows(T)]
    for ((name, h), r) in zip(MIDPOINT_SCENARIOS, midpoint_rows(T))
        prob = midpoint_problem(name, h, T)
        push!(rows,
            (; r.set, r.problem, r.start, x0 = SVector{length(r.x0)}(r.x0),
                F = InlineAdapter(midpoint_residual(prob)), p = prob.parameters))
    end
    return rows
end

"""
    BANDS

The R1 size band of each backend: the largest `n` of a system that an R2 cell launches, or
`nothing` for no band.
"""
const BANDS = Dict{String, Union{Nothing, Int}}(
    "cpu" => nothing, "metal" => 24, "cuda" => 24, "rocm" => 8)

"""
    band_status(backend, n)

The status `above band <band>` of a system of dimension `n` above the band of `backend`, or
`nothing` for a system within it.
"""
function band_status(backend::AbstractString, n::Integer)
    b = BANDS[backend]
    return b !== nothing && n > b ? "above band $b" : nothing
end

"""
    tenE_matrix(n, T)

The matrix `A = 2I + R` of the §10.E problem in the element type `T`: `R` has a zero diagonal and
entries drawn uniformly from `[-1, 1]` and scaled by `1 / (2(n − 1))`, so that each of its
off-diagonal row sums is at most 1/2. One matrix per `n`, from the seed `n`.
"""
function tenE_matrix(n::Integer, ::Type{T}) where {T}
    R = 2 .* rand(Xoshiro(n), n, n) .- 1
    R[diagind(R)] .= 0
    R .*= 0.5 / (n - 1)
    return T.(2I + R)
end

"""
    TenEResidual(A)

The §10.E residual in R1 form, `F(x, b) = A x + sin.(x) / 10 − b`, with `A` an `SMatrix` and the
system's own `b` passed as its parameter.
"""
struct TenEResidual{M}
    A::M
end

(r::TenEResidual)(x, b) = r.A * x .+ sin.(x) ./ 10 .- b

"""
    tenE_batch(n, n_sys, T)

The §10.E problem in R1 form for `n_sys` systems of dimension `n`: the [`TenEResidual`](@ref) of
`tenE_matrix(n, T)` and the vector of the systems' right-hand sides `b = A x* + sin.(x*) / 10`,
built in `T` on the host, each with its own `x*` drawn uniformly from `[-1, 1]ⁿ` from the seed 1.
"""
function tenE_batch(n::Integer, n_sys::Integer, ::Type{T}) where {T}
    A = SMatrix{n, n, T}(tenE_matrix(n, T))
    rng = Xoshiro(1)
    B = map(1:n_sys) do _
        xs = SVector{n, T}(T.(2 .* rand(rng, n) .- 1))
        return A * xs .+ sin.(xs) ./ 10
    end
    return TenEResidual(A), B
end

"""
    tenE_r3(n, T)

The §10.E problem in R3 form, of dimension `n` in the element type `T`: the matrix
`A = tenE_matrix(n, T)` and the right-hand side `b = A x* + sin.(x*) / 10`, with `x*` drawn
uniformly from `[-1, 1]ⁿ` from the seed 1 and rounded to `T`.
"""
function tenE_r3(n::Integer, ::Type{T}) where {T}
    A = tenE_matrix(n, T)
    xs = T.(2 .* rand(Xoshiro(1), n) .- 1)
    return A, A * xs .+ sin.(xs) ./ 10
end

"""
    r3_F!(y, x, p)
    r3_J!(J, x, p)

The §10.E residual in R3 form, `y = A x + sin.(x) / 10 − b`, and its Jacobian
`J = A + diag(cos.(x)) / 10`, with broadcasts only, for `p = (; A, b, Id)`, `Id` the identity
matrix on the array type of `A`.
"""
r3_F!(y, x, p) = (y .= p.A * x .+ sin.(x) ./ 10 .- p.b; nothing)
r3_J!(J, x, p) = (J .= p.A .+ p.Id .* reshape(cos.(x) ./ 10, 1, :); nothing)

# The exception that a kernel launch wraps.
function unwrap(e)
    e isa TaskFailedException && return unwrap(e.task.exception)
    e isa CompositeException && return unwrap(first(e.exceptions))
    e isa LoadError && return unwrap(e.error)
    return e
end

# The first line of a message as one field of a CSV line: no comma, no quote, at most 160
# characters.
function csv_text(s::AbstractString)
    line = first(split(s, '\n'))
    line = length(line) > 160 ? first(line, 160) * "…" : line
    return replace(line, ',' => ';', '"' => '\'', '\r' => ' ')
end

"""
    launch_status(e)

The status of a launch that threw `e`: `no compile: <first line>` for a kernel that does not
compile (an `InvalidIRError`, a `KernelError`, or POCL's `Failed to compile program`), `kernel
error: <first line>` for a kernel that threw (a `KernelException`, or a message that holds
`exception thrown during kernel execution` or AMDGPU's `GPU Kernel Exception`), and `nothing`
for any other exception. The first line names the type of the exception.
"""
function launch_status(e)
    u = unwrap(e)
    name = nameof(typeof(u))
    msg = sprint(showerror, u)
    line = csv_text(startswith(msg, string(name)) ? msg : "$name: $msg")
    if name in (:InvalidIRError, :KernelError) || occursin("Failed to compile program", msg)
        return "no compile: $line"
    elseif name === :KernelException ||
           occursin("exception thrown during kernel execution", msg) ||
           occursin("GPU Kernel Exception", msg)
        return "kernel error: $line"
    end
    return nothing
end
