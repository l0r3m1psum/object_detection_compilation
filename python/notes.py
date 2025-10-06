import tvm

# Optimization passes transform objects of type IRModule
tvm.ir.IRModule
# That contains relax.Function and tir.PrimFunc
tvm.relax.Function # high level
tvm.tir.PrimFunc # low level

# Encapsulate the result of a compilation. Contains PackedFunc
mod: tvm.runtime.Module
arr: tvm.runtime.NDArray
fun: tvm.runtime.PackedFunc
# fun = mod['forward']; fun(arr)

# MetaSchedule is a drop-in replacement for AutoTVM and AutoScheduler


# tvm.compile -> tvm.relax.VMExecutable
tvm.relax.VirtualMachine
