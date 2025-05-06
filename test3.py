import tempfile
import numpy as np
import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn

class RelaxModel(nn.Module):
	def __init__(self):
		super(RelaxModel, self).__init__()
		self.fc1 = nn.Linear(784, 256)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(256, 10, bias=False)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		return x


input_shape = (1, 784)
mod, params = RelaxModel().export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
mod.show()

import tvm.relax.backend.cuda.cublas as _cublas

@tvm.transform.module_pass(opt_level=0, name="CublasDispatch")
class CublasDispatch:
	def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
		if not tvm.get_global_func("relax.ext.cublas", True):
			raise Exception("CUBLAS is not enabled.")

		patterns = [relax.backend.get_pattern("cublas.matmul_transposed_bias_relu")]
		# Note in real-world cases, we usually get all patterns
		# patterns = relax.backend.get_patterns_with_prefix("cublas")

		# Fuse ops by patterns and then run codegen
		mod = relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True)(mod)
		mod = relax.transform.RunCodegen()(mod)
		return mod


mod = CublasDispatch()(mod)
mod.show()

device = tvm.cuda(0)
target = tvm.target.Target.from_device(device)

from tvm import dlight as dl

with target:
	mod = tvm.ir.transform.Sequential(
		[
			relax.get_pipeline("zero"),
			dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
				dl.gpu.Matmul(),
				dl.gpu.GEMV(),
				dl.gpu.Reduction(),
				dl.gpu.GeneralReduction(),
				dl.gpu.Fallback(),
			),
		]
	)(mod)

mod.show()

ex = tvm.compile(mod, target=target)
vm = relax.VirtualMachine(ex, device)
data = tvm.nd.array(np.random.rand(*input_shape).astype("float32"), device)
gpu_params = [tvm.nd.array(np.random.rand(*p.shape).astype(p.dtype), device) for _, p in params]
gpu_out = vm["forward"](data, *gpu_params).numpy()
print(gpu_out)
