import tvm
import numpy
import ctypes

my_dev_lib = ctypes.CDLL("build/my_device_api.dll")

dev = tvm.device("ext_dev")
A = tvm.nd.array(numpy.ones(16), dev)

from tvm import ir, tir
from tvm.script import tir as T

ir.register_op_attr("tir.my_dev.func", "TCallEffectKind", tir.CallEffectKind.Pure)
ir.register_op_attr("tir.my_dev.func", "TScriptPrinterName", "tir.my_dev.func")

@ir.register_intrin_lowering("tir.my_dev.func", "default")
def coproc_dep_pop(op):
    if op.args[0].op.name != op.args[1].op.name or op.args[0].op.name != "tir.tvm_access_ptr":
        raise ValueError()
    len1 = op.args[0].args[3]
    len2 = op.args[1].args[3]
    if len1 != len2: raise ValueError()
    return tir.call_extern(
        "void", "my_func", op.args[0], op.args[1], op.args[0].args[3]
    )

@T.prim_func
def tir_func(A: T.Buffer((16,), "float64"), B: T.Buffer((16,), "float64")):
    tir.call_intrin("void", "tir.my_dev.func", A.access_ptr("r"), B.access_ptr("w"))

tir_func.show()
target = tvm.target.Target("ext_dev")
ex = tir.build(tir_func, target)
B = tvm.nd.array(numpy.zeros(16), dev)
ex["tir_func"](A, B)
print(B)
