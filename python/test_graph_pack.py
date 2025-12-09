import tvm
from tvm import relax
from tvm import topi
from tvm import te
from tvm import tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import vtar.relax.transform

# inp_dtype: int8
# wgt_dtype: int8
# out_dtype: int8
# acc_dtype: int32
# BATCH: 1
# BLOCK_IN: 16
# BLOCK_OUT: 16

from tvm import dlight as dl
from tvm import ir

inp_width = 8
int8_max = 127

@I.ir_module
class ALUOperations:
	@R.function
	def main(
		x: R.Tensor((1, 128 + 1, 1, 16), dtype="int32"),
		y: R.Tensor((1, 128 + 1, 1, 16), dtype="int32"),
	):
		with R.dataflow():
			# TODO: x + y * x
			lv = x + y
			lv1 = R.right_shift(lv, R.const(inp_width))
			lv2 = R.maximum(lv1, R.const(0)) # ReLU
			lv3 = R.minimum(lv2, R.const(int8_max))
			lv4 = lv3.astype("int8")
			gv = lv4
			R.output(gv)
		return gv

mod = ALUOperations
almost_end2end_pipeline = ir.transform.Sequential([
	# TODO: integer only quantization
	# vtar.relax.transform.GraphPack(),
	# relax.transform.CanonicalizeBindings(), # removes redundant assignments
	relax.get_pipeline('vtar_zero'),
	############################################################################
	tir.transform.ForceNarrowIndexToInt32(),
	dl.ApplyDefaultSchedule(
		vtar.dlight.ALU(),
	),
	vtar.get_vtar_tir_transform(),
])

env = vtar.get_env()

env.INP_BUFF_SIZE//8, env.WGT_BUFF_SIZE//(8*8), env.ACC_BUFF_SIZE//32, env.OUT_BUFF_SIZE//8
(4096, 4096, 4096, 4096)
4096//16 == 256
# (1, 128, 1, 16) we can load at most two of this in ACC_BUFF, hence anything
# bigger needs to be split

target = tvm.target.Target(env.target, host=env.target_host)
mod.show()
with target:
	mod = almost_end2end_pipeline(mod)
mod.show()
ex = tvm.compile(mod, target=target)
import numpy
import ctypes; vta_fsim = ctypes.CDLL("vta_fsim")
dev = tvm.device('ext_dev')
vm = relax.VirtualMachine(ex, dev)
# Perché devono essere allocati su dev???
a = tvm.nd.array(numpy.ones((1, 128 + 1, 1, 16), dtype='int32')*1024, dev)
res = vm['main'](a, a)
raise SystemExit(0)

import numpy
from vtar.relax.transform import _get_shape

# It is necessary to use vtar_zero because otherwise we get
#     TVMError: CodeGenVM cannot handle this intrinsic now:
#     Op(relax.nn.conv2d)
# This is because CodeGenVM::VisitExpr_(const CallNode*) can handle very few
# operatons and Relax nodes are not among them; we have to convert them to
# R.call_tir with our custom legalize ops.
irmods = (ConvModel, ConvModelPacked)
vms = []
for irmod in irmods:
	irmod = relax.get_pipeline('vtar_zero')(irmod)
	irmod = tvm.transform.PrintIR()(irmod)
	dev = tvm.device('llvm')
	target = tvm.target.Target('llvm')
	vmexec = tvm.compile(irmod, target)
	vms.append(relax.VirtualMachine(vmexec, dev))

seed = 42
rng = numpy.random.default_rng(seed)
params = [_get_shape(param) for param in ConvModel['main'].params]
x = tvm.nd.array((rng.random(params[0])*255).astype('int8'), device=dev)
w = tvm.nd.array((rng.random(params[1])*255).astype('int8'), device=dev)
b = tvm.nd.array((rng.random(params[2])*255).astype('int32'), device=dev)
ok = numpy.all(numpy.equal(vms[0]['main'](x, w, b).numpy(), vms[1]['main'](x, w, b).numpy()))
if not ok:
	raise ValueError("Packed and not packed are not equal...")
