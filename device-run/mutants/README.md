# The named mutants of part RO

One patch per mutant against `915e45b`, applied by `../run.sh` one at a time with `git apply`.

| mutant | change | caught by |
|:--|:--|:--|
| N1 | `trsv!` on the whole `m × n` factor and `d` (line 66) | Error in "rectangular QR, m = 6 > n = 3: Float32", `DimensionMismatch` |
| N2 | `ormqr!` with `'N'` (line 65) | Fail in "agreement with own KA kernels at n = 16", Float32, the QR row |
| N3 | `conj` of the `m × n` complex `τ` after `geqrf!` (after line 52) | Fail in "rectangular QR, m = 6 > n = 3: ComplexF32" |
| N4a | the method `backsolve_shifted!` deleted (lines 70–81) | 8 Fail at line 47 of `test/gpu/rocm/runtests.jl` |
| N4b | N4a, and line 47 of `test/gpu/rocm/runtests.jl` deleted | Fail in "rectangular QR, m = 6 > n = 3: ComplexF32", the same-type `ldiv!!` row |
| N5 | `trsv!` on the whole `d` (line 66) | Error in "rectangular QR, m = 6 > n = 3: Float32", `DimensionMismatch` |
