import tvm
from tvm import ir, relax
from tvm.script import ir as I, relax as R

import numpy

from typing import List, Dict
import ctypes

testerino_lib = ctypes.CDLL("build/my_codegen.dll")

# relax.backend.metal.coreml
@tvm._ffi.register_func("relax.ext.my_target")
def my_compiler(
    funcs: List[relax.Function],
    options,
    constant_names: Dict[relax.Constant, str]
) -> List[tvm.runtime.Module]:
    create = tvm._ffi.get_global_func("tvm.my_target_runtime.create")
    compiled_funcs = [
        create("symbol", "model_path"),
    ]
    return compiled_funcs

@ir.transform.module_pass(opt_level=0)
class MyDispatch:
    def transform_module(self, mod: ir.IRModule, _ctx: ir.transform.PassContext) -> ir.IRModule:
        pattern = relax.dpl.is_op("relax.add")(
            relax.dpl.wildcard(),
            relax.dpl.wildcard()
        )
        patterns = (
            relax.transform.FusionPattern("my_target.pattern", pattern),
        )

        mod = relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True)(mod)
        mod = relax.transform.RunCodegen()(mod)
        return mod

@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor((16,), dtype="float32")):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            gv = R.add(x, R.const(1.))
            R.output(gv)
        return gv

mod = Module
mod = MyDispatch()(mod)
mod.show()
dev = tvm.cpu()
target = tvm.target.Target("llvm")
ex = relax.build(mod, target)
lib_path = "build/deploy_output.dll"
ex.export_library(
    lib_path,
    workspace_dir="build",
    options=[
        "--target=x86_64-pc-windows-msvc",
        "-save-temps=build", "-v", "-Wl,-verbose",
        "-g",
    ]
)
loaded_ex = tvm.runtime.load_module(lib_path)
vm = relax.VirtualMachine(loaded_ex, dev)
x = tvm.nd.array(numpy.ones(16, dtype="float32"), dev)
y = vm['forward'](x)
print(x)
print(y)
