"""
    certify(prob::NonlinearProblem, x, p = nothing) -> (; certified, enclosure, guaranteed)

Test with interval arithmetic whether `prob` has a root near the iterate `x`, for the parameters
`p`, when `IntervalArithmetic` is loaded. The package extension
`GeometricSolversIntervalArithmeticExt` adds the method; without it, `certify` has no method.

The test is Krawczyk's: with `Y` the inverse of the Jacobian at `x`, and a box `X` around `x`, the
operator

``K(X) = x - Y F(x) + (I - Y J(X)) (X - x)``

is evaluated in intervals. When `K(X)` lies in the interior of `X`, the box `X` holds exactly one
root of `F`, and `K(X)` encloses it. The first box has the radius of one `eps` of `x`, and each
failed test widens the box to twice the distance of `K(X)` from `x`, plus one `eps` of `x`, up to
10 boxes.

The result is a `NamedTuple` of one concrete type for both outcomes:

- `certified`: whether the test holds.
- `enclosure`: `K(X)` when `certified`, in the storage of `x` with entries `Interval{T}` for the
  element type `T` of `x`: an `Interval{T}` for a `Number`, an `SVector` or an `MVector` of them
  for an `SVector` or an `MVector`, and a `Vector` of them for a `Vector`. With no certificate,
  every entry is the whole real line.
- `guaranteed`: whether every interval of `F(x)` and `J(X)` is guaranteed, that is, carries no
  NG flag of IntervalArithmetic. A number that is not an interval, such as the `1` of
  `1 - x[1]`, sets the flag where it meets an interval, and the certificate holds without the
  guarantee. With no certificate, `guaranteed` is `false`.

The call form of `F` and `J` follows the storage of `x`, as for a solve: `F(x, p)` and `J(x, p)`
for a real floating-point `Number` and a `StaticVector`, and `F(y, x, p)` and `J(A, x, p)` for a
host `Vector`. Without `J`, `ForwardDiff` differentiates `F`. `F` and `J` must accept intervals.
A singular Jacobian at `x`, a non-finite `x`, `F(x)` or `Y`, a box radius that is not finite, and
a test that fails on the tenth box each give `certified == false`; none of these throws. A
complex iterate, a batch, a device array, a [`StationarityProblem`](@ref) and a
[`LeastSquaresProblem`](@ref) have no method.

`certify` is `public` and not exported.
"""
function certify end
