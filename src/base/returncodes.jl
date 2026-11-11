"""
    ReturnCode

Why a solve or a step ended, stored in one byte so that a status fits a GPU register budget:

- `SUCCESS`: the residual test of [`Options`](@ref) holds.
- `STALLED`: the iterate or the residual stopped improving before the test held.
- `MAXITERS`: the iteration cap was reached: the cap of the solve, or the cap of an inner
  iteration inside a step, such as a [`RAS`](@ref) local solve that did not converge.
- `SINGULAR`: a factorisation failed, as [`factorize!!`](@ref) states; it does not throw.
- `NONFINITE`: a residual, a step or a norm is not finite.
- `LINESEARCH_FAILED`: the globalisation found no acceptable step.
"""
@enum ReturnCode::UInt8 SUCCESS STALLED MAXITERS SINGULAR NONFINITE LINESEARCH_FAILED

"""
    sciml_retcode(code::ReturnCode) -> SciMLBase.ReturnCode.T

Map a [`ReturnCode`](@ref) to a `SciMLBase.ReturnCode.T` when `SciMLBase` is loaded. A SciML caller
reads `sciml_retcode(status.code)`:

- `SUCCESS` → `SciMLBase.ReturnCode.Success`
- `STALLED` → `SciMLBase.ReturnCode.Stalled`
- `MAXITERS` → `SciMLBase.ReturnCode.MaxIters`
- `SINGULAR` → `SciMLBase.ReturnCode.InternalLinearSolveFailed`
- `NONFINITE` → `SciMLBase.ReturnCode.Unstable`
- `LINESEARCH_FAILED` → `SciMLBase.ReturnCode.InternalLineSearchFailed`
"""
function sciml_retcode end
