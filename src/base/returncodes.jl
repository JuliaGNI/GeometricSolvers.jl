"""
    ReturnCode

Why a solve or a step ended, stored in one byte so that a status fits a GPU register budget:

- `SUCCESS`: the stopping test of [`Options`](@ref) holds.
- `STALLED`: the iterate or the residual stopped improving before the test held.
- `MAXITERS`: the iteration cap was reached.
- `SINGULAR`: a factorisation found a zero pivot; it reports this and does not throw.
- `NONFINITE`: a residual, a step or a norm is not finite.
- `LINESEARCH_FAILED`: the globalisation found no acceptable step.
"""
@enum ReturnCode::UInt8 SUCCESS STALLED MAXITERS SINGULAR NONFINITE LINESEARCH_FAILED
