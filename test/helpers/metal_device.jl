# The paravirtual GPU of the GitHub macOS runners throws device exceptions at random once a process
# has run enough kernels, in kernels that pass on a real Apple GPU (issue #34). The Metal device
# tests therefore skip on a virtualized GPU and run on a real one.

on_virtual_gpu() = Metal.functional() && Metal.MTL.is_virtual(Metal.device())
