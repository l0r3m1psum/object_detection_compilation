import tvm
from tvm import relax
from tvm import topi
from tvm import te
from tvm import tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import vtar.relax.transform

env = vtar.get_env()

env.INP_BUFF_SIZE//8, env.WGT_BUFF_SIZE//(8*8), env.ACC_BUFF_SIZE//32, env.OUT_BUFF_SIZE//8
(4096, 4096, 4096, 4096)
4096//16 == 256
# (1, 128, 1, 16) we can load at most two of this in ACC_BUFF, hence anything
# bigger needs to be split

# TODO: move this stuff in the test_vta_compiler.py file
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
