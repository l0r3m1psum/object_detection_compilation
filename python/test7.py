import os

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

target = tvm.target.Target('llvm')
ex = tvm.compile(mod, target=target)
if not os.path.exists('build'): os.mkdir('build')
ex.export_library('build/mlp.dll')
