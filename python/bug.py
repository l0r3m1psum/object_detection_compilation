import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm import dlight as dl

class MLPModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 256)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(256, 10)
	def forward(self, x):
		x = self.fc1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		return x

mod, param_spec = MLPModel().export_tvm(
	spec={'forward': {'x': nn.spec.Tensor((1, 784), 'float32')}}
)

zero_pipeline = relax.get_pipeline('zero')
mod_op = zero_pipeline(mod)

with tvm.target.Target('cuda'):
	gpu_mod = dl.ApplyDefaultSchedule(
		dl.gpu.Matmul(),
		dl.gpu.Fallback(),
	)(mod_op)

exec = tvm.compile(gpu_mod, target='cuda')
dev = tvm.device('cuda', 0)
vm = relax.VirtualMachine(exec, dev)

# NULL pointer dereference
relax.op.reshape(None, (1,2,3))
