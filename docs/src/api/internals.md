```@meta
CurrentModule = GeometricSolvers
```

# Internals

These names are not exported. Reach them as `GeometricSolvers.name`.

## The storage seam

The storage seam lets an algorithm read each function's result rather than the storage itself,
so the storage-specific operations stay at the leaves. Each function has a value method for a
`StaticArray` or a `Number` iterate (R1) and an in-place method for an `AbstractArray` buffer slot
(R3).

```@docs
buffer
residual!!
update_iterate!!
map!!
mul!!
copy!!
scale!!
axpby!!
regularize!!
```

## The solver loop and its helpers

A nonlinear method enters the loop of [`solve!!`](@ref) through [`init_state`](@ref), which builds
its state, [`step!!`](@ref), and the two hooks [`start!!`](@ref) and [`finish!!`](@ref), which run
once before the first step and at every return. [`residual_buffer`](@ref) names the slot of the
state that holds `F(x₀)`, and [`refine!!`](@ref) refines the linear direction `d` in working
precision before globalization; [`gmres_ir!!`](@ref) is the GMRES-IR stage that follows a failed
pass of a [`GMRESRefinement`](@ref).

```@docs
init_state
residual_buffer
step!!
start!!
finish!!
refine!!
sync_merit
init_globalization
LineFunction
globalize!!
LineSearch
linesearch
φ
φ′
ExactStep
InexactStep
MeasuredSlope
LineSearchResult
roundoff
smallest_step
sufficient_decrease
classify
backtrack_step
zoom_step
LinearMethod
compute_forcing
max_inner_iterations
GMRESWorkspace
init_gmres_workspace
GMRESInfo
gmres_solve!
gmres_ir!!
promote_lu_factors!
arnoldi_dot
R5Operator
R5JVP
CountingResidual
sat16
with_calls
jacobi_rotation
ToReal
StripDual
parameters
record
converged
isconverged
isstalled
norm2
rnorm
rdot
BatchEliminatedJVPPrep
PIC1DParams
pic_gather
pic_inner_residual
pic_inner_jacobian
deposit_pic_moments!
pic_outer_residual!
counting_problem
```
