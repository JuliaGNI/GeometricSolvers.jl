# The standard test set: the 23 nonlinear systems of NonlinearProblemLibrary.jl.
#
# Sources. The residuals, starts, titles and roots are copied from NonlinearProblemLibrary.jl
# 0.1.8 (SciML/DiffEqProblemLibrary.jl, `lib/NonlinearProblemLibrary`, MIT licence), which
# implements John Burkardt's `test_nonlin` collection, after J. J. Moré, B. S. Garbow and
# K. E. Hillstrom, "Testing unconstrained optimization software", ACM Transactions on
# Mathematical Software 7 (1981), 17–41. The scaled starts `10 u0` and `100 u0` are those of
# Moré, Garbow and Hillstrom.
#
# Each residual is an in-place `F(y, x, p)` of this package's `NonlinearProblem`. It computes in
# the element type `T = eltype(x)`: every non-integer constant is written as `T(c)`, so that a
# `Float32` iterate computes in `Float32` and a dual number of the chunked forward mode stays a
# dual number. `test/verification/standard.jl` parses this file and fails on a floating-point
# literal outside a call `T(...)`, and `scripts/standard/check_transcription.jl` compares each
# residual with the library's in `Float64`.

using GeometricSolvers: NonlinearProblem

"""
    StandardProblem

One problem of the standard test set: its `number` and `title` in the library, its size `n`, the
`NonlinearProblem` `prob` of its residual, its start `u0(T)` in the element type `T`, and its
root `root(T)`, or `nothing` where the source states none.
"""
struct StandardProblem{P <: NonlinearProblem, S, R}
    number::Int
    title::String
    n::Int
    prob::P
    u0::S
    root::R
end

function StandardProblem(number, title, n, f!, u0, root)
    return StandardProblem(number, title, n, NonlinearProblem(f!), u0, root)
end

"""
    standard_starts(sp::StandardProblem, T)

The starts of `sp` in the element type `T`, as pairs of a label and an iterate: `u0`, `10 u0`
and `100 u0`, where a start equal to an earlier one appears once.
"""
function standard_starts(sp::StandardProblem, ::Type{T}) where {T}
    u0 = sp.u0(T)
    starts = Pair{String, Vector{T}}["u0" => u0]
    for (label, s) in (("10u0", 10), ("100u0", 100))
        x = s .* u0
        any(st -> st.second == x, starts) || push!(starts, label => x)
    end
    return starts
end

# ------------------------------------- Problem 1 ------------------------------------------
function p1_f!(out, x, p)
    n = length(x)
    out[1] = 1 - x[1]
    @views @. out[2:n] = 10 * (x[2:n] - x[1:(n - 1)] * x[1:(n - 1)])
    return nothing
end

# ------------------------------------- Problem 2 ------------------------------------------
function p2_f!(out, x, p)
    T = eltype(x)
    out[1] = x[1] + 10 * x[2]
    out[2] = sqrt(T(5)) * (x[3] - x[4])
    out[3] = (x[2] - 2 * x[3])^2
    out[4] = sqrt(T(10)) * (x[1] - x[4]) * (x[1] - x[4])
    return nothing
end

# ------------------------------------- Problem 3 ------------------------------------------
function p3_f!(out, x, p)
    T = eltype(x)
    out[1] = 10000 * x[1] * x[2] - 1
    out[2] = exp(-x[1]) + exp(-x[2]) - T(1.0001)
    return nothing
end

# ------------------------------------- Problem 4 ------------------------------------------
function p4_f!(out, x, p)
    T = eltype(x)
    temp1 = x[2] - x[1] * x[1]
    temp2 = x[4] - x[3] * x[3]

    out[1] = -200 * x[1] * temp1 - (1 - x[1])
    out[2] = 200 * temp1 + T(20.2) * (x[2] - 1) + T(19.8) * (x[4] - 1)
    out[3] = -180 * x[3] * temp2 - (1 - x[3])
    out[4] = 180 * temp2 + T(20.2) * (x[4] - 1) + T(19.8) * (x[2] - 1)
    return nothing
end

# ------------------------------------- Problem 5 ------------------------------------------
function p5_f!(out, x, p)
    T = eltype(x)
    temp1 = atan(x[2] / x[1]) / (2 * T(pi))
    temp = ifelse(0 < x[1], temp1, ifelse(x[1] < 0, temp1 + T(0.5), T(0.25) * sign(x[2])))
    out[1] = 10 * (x[3] - 10 * temp)
    out[2] = 10 * (sqrt(x[1] * x[1] + x[2] * x[2]) - 1)
    out[3] = x[3]
    return nothing
end

# ------------------------------------- Problem 6 ------------------------------------------
function p6_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    out .= 0
    for i in 1:29
        ti = T(i) / 29
        sum1 = zero(T)
        temp = one(T)
        for j in 2:n
            sum1 = sum1 + j * temp * x[j]
            temp = ti * temp
        end

        sum2 = zero(T)
        temp = one(T)
        for j in 1:n
            sum2 = sum2 + temp * x[j]
            temp = ti * temp
        end
        temp = 1 / ti

        for k in 1:n
            out[k] = out[k] + temp * (sum1 - sum2 * sum2 - 1) * (k - 2 * ti * sum2)
            temp = ti * temp
        end
    end

    out[1] += x[1] * (3 - 2 * x[2] + 2 * x[1]^2)
    out[2] += x[2] * (1 - x[2]) - 1
    return nothing
end

# ------------------------------------- Problem 7 ------------------------------------------
function p7_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    out .= 0
    for j in 1:n
        t1 = one(T)
        t2 = x[j]
        for i in 1:n
            out[i] += t2
            t3 = 2 * x[j] * t2 - t1
            t1 = t2
            t2 = t3
        end
    end

    for i in 1:n
        out[i] = out[i] / n + ifelse(i % 2 == 0, one(T) / (i * i - 1), zero(T))
    end
    return nothing
end

# ------------------------------------- Problem 8 ------------------------------------------
function p8_f!(out, x, p)
    n = length(x)
    sum_x = sum(x)
    @views @. out[1:(n - 1)] = x[1:(n - 1)] + sum_x - (n + 1)
    out[n] = prod(x) - 1
    return nothing
end

# ------------------------------------- Problem 9 ------------------------------------------
function discrete_bvf_kernel(xk, k, h)
    T = typeof(xk)
    return 2 * xk + T(0.5) * h^2 * (xk + k * h + 1)^3
end

function p9_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    h = one(T) / (n + 1)
    out[1] = discrete_bvf_kernel(x[1], 1, h) - x[2]
    for k in 2:(n - 1)
        out[k] = discrete_bvf_kernel(x[k], k, h) - x[k - 1] - x[k + 1]
    end
    out[n] = discrete_bvf_kernel(x[n], n, h) - x[n - 1]
    return nothing
end

# ------------------------------------- Problem 10 -----------------------------------------
function p10_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    h = one(T) / (n + 1)
    for k in 1:n
        tk = T(k) / (n + 1)
        sum1 = zero(T)
        for j in 1:k
            tj = j * h
            sum1 = sum1 + tj * (x[j] + tj + 1)^3
        end
        sum2 = zero(T)
        for j in k:n
            tj = j * h
            sum2 = sum2 + (1 - tj) * (x[j] + tj + 1)^3
        end

        out[k] = x[k] + h * ((1 - tk) * sum1 + tk * sum2) / 2
    end
    return nothing
end

# ------------------------------------- Problem 11 -----------------------------------------
function p11_f!(out, x, p)
    n = length(x)
    c_sum = sum(cos, x)
    for k in 1:n
        sxk, cxk = sincos(x[k])
        out[k] = n - c_sum + k * (1 - cxk) - sxk
    end
    return nothing
end

# ------------------------------------- Problem 12 -----------------------------------------
function p12_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    sum1 = zero(T)
    for j in 1:n
        sum1 += j * (x[j] - 1)
    end
    for k in 1:n
        out[k] = x[k] - 1 + k * sum1 * (1 + 2 * sum1 * sum1)
    end
    return nothing
end

# ------------------------------------- Problem 13 -----------------------------------------
broyden_tridiag_kernel(xk) = (3 - 2 * xk) * xk + 1

function p13_f!(out, x, p)
    n = length(x)
    out[1] = broyden_tridiag_kernel(x[1]) - 2 * x[2]
    for k in 2:(n - 1)
        out[k] = broyden_tridiag_kernel(x[k]) - x[k - 1] - 2 * x[k + 1]
    end
    out[n] = broyden_tridiag_kernel(x[n]) - x[n - 1]
    return nothing
end

# ------------------------------------- Problem 14 -----------------------------------------
function p14_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    ml = 5
    mu = 1
    for k in 1:n
        k1 = max(1, k - ml)
        k2 = min(n, k + mu)

        temp = zero(T)
        for j in k1:k2
            if j != k
                temp += x[j] * (1 + x[j])
            end
        end
        out[k] = x[k] * (2 + 5 * x[k] * x[k]) + 1 - temp
    end
    return nothing
end

# ------------------------------------- Problem 15 -----------------------------------------
function p15_f!(out, x, p)
    T = eltype(x)
    out[1] = (x[1] * x[1] + x[2] * x[3]) - T(0.0001)
    out[2] = (x[1] * x[2] + x[2] * x[4]) - 1
    out[3] = (x[3] * x[1] + x[4] * x[3]) - 0
    out[4] = (x[3] * x[2] + x[4] * x[4]) - T(0.0001)
    return nothing
end

# ------------------------------------- Problem 16 -----------------------------------------
function p16_f!(out, x, p)
    T = eltype(x)
    out[1] = (x[1] * x[1] + x[2] * x[4] + x[3] * x[7]) - T(0.0001)
    out[2] = (x[1] * x[2] + x[2] * x[5] + x[3] * x[8]) - 1
    out[3] = x[1] * x[3] + x[2] * x[6] + x[3] * x[9]
    out[4] = x[4] * x[1] + x[5] * x[4] + x[6] * x[7]
    out[5] = (x[4] * x[2] + x[5] * x[5] + x[6] * x[8]) - T(0.0001)
    out[6] = x[4] * x[3] + x[5] * x[6] + x[6] * x[9]
    out[7] = x[7] * x[1] + x[8] * x[4] + x[9] * x[7]
    out[8] = x[7] * x[2] + x[8] * x[5] + x[9] * x[8]
    out[9] = (x[7] * x[3] + x[8] * x[6] + x[9] * x[9]) - T(0.0001)
    return nothing
end

# ------------------------------------- Problem 17 -----------------------------------------
function p17_f!(out, x, p)
    out[1] = x[1] + x[2] - 3
    out[2] = x[1]^2 + x[2]^2 - 9
    return nothing
end

# ------------------------------------- Problem 18 -----------------------------------------
function p18_f!(out, x, p)
    T = eltype(x)
    if iszero(x[1])
        out[1] = zero(T)
    else
        out[1] = x[2]^2 * (1 - exp(-x[1] * x[1])) / x[1]
    end

    if iszero(x[2])
        out[2] = zero(T)
    else
        out[2] = x[1] * (1 - exp(-x[2] * x[2])) / x[2]
    end

    return nothing
end

# ------------------------------------- Problem 19 -----------------------------------------
function p19_f!(out, x, p)
    tmp = x[1]^2 + x[2]^2
    out[1] = x[1] * tmp
    out[2] = x[2] * tmp
    return nothing
end

# ------------------------------------- Problem 20 -----------------------------------------
function p20_f!(out, x, p)
    out[1] = x[1] * (x[1] - 5)^2
    return nothing
end

# ------------------------------------- Problem 21 -----------------------------------------
function p21_f!(out, x, p)
    out[1] = x[1] - x[2]^3 + 5 * x[2]^2 - 2 * x[2] - 13
    out[2] = x[1] + x[2]^3 + x[2]^2 - 14 * x[2] - 29
    return nothing
end

# ------------------------------------- Problem 22 -----------------------------------------
function p22_f!(out, x, p)
    T = eltype(x)
    out[1] = x[1] * x[1] - x[2] + 1
    out[2] = x[1] - cospi(T(0.5) * x[2])
    return nothing
end

# ------------------------------------- Problem 23 -----------------------------------------
function p23_f!(out, x, p)
    T = eltype(x)
    n = length(x)
    c = T(0.9)
    for i in 1:n
        out[i] = x[i]
    end
    for i in 1:n
        μi = T(i) / n
        s = zero(T)
        for j in 1:n
            μj = T(j) / n
            s = s + (μi * x[j]) / (μi + μj)
        end
        term = 1 - c * s / (2 * n)
        out[i] -= 1 / term
    end
    return nothing
end

# ------------------------------------- The set --------------------------------------------
"""
    STANDARD_PROBLEMS

The 23 problems of the standard test set, in the order and with the titles of the library.
"""
const STANDARD_PROBLEMS = [
    StandardProblem(1, "Generalized Rosenbrock function", 10, p1_f!,
        T -> [T(-1.2); ones(T, 9)], T -> ones(T, 10)),
    StandardProblem(2, "Powell singular function", 4, p2_f!,
        T -> T[3, -1, 0, 1], T -> zeros(T, 4)),
    StandardProblem(3, "Powell badly scaled function", 2, p3_f!,
        T -> T[0, 1], T -> [T(1.098159e-5), T(9.106146)]),
    StandardProblem(4, "Wood function", 4, p4_f!,
        T -> T[-3, -1, -3, -1], T -> ones(T, 4)),
    StandardProblem(5, "Helical valley function", 3, p5_f!,
        T -> T[-1, 0, 0], T -> T[1, 0, 0]),
    StandardProblem(6, "Watson function", 2, p6_f!,
        T -> zeros(T, 2), nothing),
    StandardProblem(7, "Chebyquad function", 2, p7_f!,
        T -> T[(2 * i - 2) // 3 for i in 1:2],
        T -> [2 * T(0.2113248654051871) - 1, 2 * T(0.7886751345948129) - 1]),
    StandardProblem(8, "Brown almost linear function", 10, p8_f!,
        T -> fill(T(1 // 2), 10), T -> ones(T, 10)),
    StandardProblem(9, "Discrete boundary value function", 10, p9_f!,
        T -> T[i * (i - 11) // 121 for i in 1:10], nothing),
    StandardProblem(10, "Discrete integral equation function", 10, p10_f!,
        T -> T[i * (i - 11) // 121 for i in 1:10], nothing),
    StandardProblem(11, "Trigonometric function", 10, p11_f!,
        T -> fill(T(1 // 10), 10), nothing),
    StandardProblem(12, "Variably dimensioned function", 10, p12_f!,
        T -> T[1 - i / 10 for i in 1:10], T -> ones(T, 10)),
    StandardProblem(13, "Broyden tridiagonal function", 10, p13_f!,
        T -> fill(-one(T), 10), nothing),
    StandardProblem(14, "Broyden banded function", 10, p14_f!,
        T -> fill(-one(T), 10), nothing),
    StandardProblem(15, "Hammarling 2 by 2 matrix square root problem", 4, p15_f!,
        T -> T[1, 0, 0, 1], T -> [T(0.01), 50, 0, T(0.01)]),
    StandardProblem(16, "Hammarling 3 by 3 matrix square root problem", 9, p16_f!,
        T -> T[1, 0, 0, 0, 1, 0, 0, 0, 1], T -> [
            T(0.01), 50, 0, 0, T(0.01), 0, 0, 0, T(0.01)]),
    StandardProblem(17, "Dennis and Schnabel 2 by 2 example", 2, p17_f!,
        T -> T[1, 5], T -> T[0, 3]),
    StandardProblem(18, "Sample problem 18", 2, p18_f!,
        T -> T[2, 2], T -> zeros(T, 2)),
    StandardProblem(19, "Sample problem 19", 2, p19_f!,
        T -> T[3, 3], T -> zeros(T, 2)),
    StandardProblem(20, "Scalar problem f(x) = x(x - 5)^2", 1, p20_f!,
        T -> T[1], T -> T[5]),
    StandardProblem(21, "Freudenstein-Roth function", 2, p21_f!,
        T -> [T(0.5), -2], T -> T[5, 4]),
    StandardProblem(22, "Boggs function", 2, p22_f!,
        T -> T[1, 0], T -> T[0, 1]),
    StandardProblem(23, "Chandrasekhar function", 10, p23_f!,
        T -> ones(T, 10), nothing)
]
