import os

import utils

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
# mod.show()
gen = 'gpu'

if gen == 'gpu':
	target = tvm.target.Target('cuda')
	with target:
		cuda_mod = tvm.ir.transform.Sequential([
			tvm.relax.get_pipeline("zero"),
			tvm.dlight.ApplyDefaultSchedule(
				tvm.dlight.gpu.Matmul(),
				tvm.dlight.gpu.GEMV(),
				tvm.dlight.gpu.Reduction(),
				tvm.dlight.gpu.GeneralReduction(),
				tvm.dlight.gpu.Fallback(),
			),
		])(mod)

	ex = tvm.compile(cuda_mod, target)
elif gen == 'cpu':
	target = tvm.target.Target('llvm')
	# ex = tvm.compile(mod, target=target)
	ex = tvm.relax.build(
		mod,
		target=target,
		params=None,
		relax_pipeline="default",
		tir_pipeline="default",
		exec_mode="compiled"
	)
else:
	assert False

if not os.path.exists('build'): os.mkdir('build')
ex.export_library(**utils.get_export_library_args('mlp'))
