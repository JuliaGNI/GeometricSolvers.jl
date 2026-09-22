# Accuracy of Float16 dot products under five accumulation policies, against a BigFloat reference.
# Error is reported relative to the condition-free scale sum(|a_i b_i|), in units of eps(Float16).
using Random, Printf

twosum(a, b) = (s = a + b; bb = s - a; (s, (a - (s - bb)) + (b - bb)))
twoprod(a, b) = (p = a * b; (p, fma(a, b, -p)))

naive(a, b) = (s = zero(eltype(a)); for i in eachindex(a)
        s += a[i] * b[i]
    end; s)
f32acc(a, b) = (s = 0.0f0; for i in eachindex(a)
        s += Float32(a[i]) * Float32(b[i])
    end; Float16(s))
function neumaier(a, b)       # compensated sum, products rounded to Float16
    s = c = zero(eltype(a))
    for i in eachindex(a)
        s, e = twosum(s, a[i] * b[i])
        c += e
    end
    s + c
end
function dot2(a, b)           # Ogita–Rump–Oishi Dot2: TwoProd via fma, TwoSum
    s = c = zero(eltype(a))
    for i in eachindex(a)
        p, π = twoprod(a[i], b[i])
        s, σ = twosum(s, p)
        c += π + σ
    end
    s + c
end
function twiceprec(a, b)      # Base.TwicePrecision{Float16}
    s = Base.TwicePrecision{Float16}(zero(Float16))
    for i in eachindex(a)
        s += Base.TwicePrecision{Float16}(a[i]) * b[i]
    end
    Float16(s)
end

const POLICIES = (
    naive = naive, neumaier = neumaier, dot2 = dot2, twiceprec = twiceprec, f32acc = f32acc)

function run(scale)
    @printf("scale = %g\n%8s", scale, "n")
    foreach(k -> @printf("%11s", k), keys(POLICIES))
    println()
    for n in (10, 100, 1000, 10000)
        errs = zeros(length(POLICIES))
        for seed in 1:5
            rng = Xoshiro(seed)
            a = Float16.(scale .* randn(rng, n))
            b = Float16.(scale .* randn(rng, n))
            exact = sum(big.(a) .* big.(b))
            mag = sum(abs.(big.(a) .* big.(b)))
            for (j, f) in enumerate(values(POLICIES))
                errs[j] = max(errs[j], Float64(abs(big(f(a, b)) - exact) / mag /
                                               eps(Float16)))
            end
        end
        @printf("%8d", n)
        foreach(e -> @printf("%11.3f", e), errs)
        println()
    end
end

# ‖a‖² = dot(a, a): no cancellation, so the error is relative to the exact value itself.
function run_norm2(scale)
    @printf("norm², scale = %g\n%8s", scale, "n")
    foreach(k -> @printf("%11s", k), keys(POLICIES))
    println()
    for n in (10, 100, 1000, 10000)
        errs = zeros(length(POLICIES))
        for seed in 1:5
            a = Float16.(scale .* randn(Xoshiro(seed), n))
            exact = sum(big.(a) .^ 2)
            for (j, f) in enumerate(values(POLICIES))
                errs[j] = max(errs[j], Float64(abs(big(f(a, a)) - exact) / exact /
                                               eps(Float16)))
            end
        end
        @printf("%8d", n)
        foreach(e -> @printf("%11.3g", e), errs)
        println()
    end
end

println("public TwicePrecision: ", Base.ispublic(Base, :TwicePrecision), "   julia ", VERSION)
run(1.0)      # ordinary magnitudes
run(1e-2)     # products near the Float16 normal range floor (6.1e-5)
run_norm2(1.0)
run_norm2(10.0)   # ‖a‖² passes floatmax(Float16) = 65504 at n ≈ 650
