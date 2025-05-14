import tvm
from tvm import relax
from tvm.relax.frontend import nn
import numpy
import utils

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
print(param_spec)
# mod.show()

zero_pipeline = relax.get_pipeline('zero')
mod_op = zero_pipeline(mod)
# mod_op.show()

device = tvm.cpu()
target = tvm.target.Target('llvm')
# target = tvm.target.Target('c')
ex = tvm.compile(mod_op, target)
vm = relax.VirtualMachine(ex, device) # An LLVM's assersion is violated for some reason...

data = numpy.random.rand(1, 784).astype('float32')
tvm_data = tvm.nd.array(data, device=device)
params = [
	tvm.nd.array(numpy.random.rand(*param.shape).astype('float32'), device=device)
	for _, param in param_spec]
print(vm['forward'](tvm_data, *params).numpy())

from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class TVMScriptModule:
	@R.function
	def main(
		x: R.Tensor((1, 784), dtype='float32'),
		fc1_weight: R.Tensor((256, 784), dtype='float32'),
		fc1_bias: R.Tensor((256,), dtype='float32'),
		fc2_weight: R.Tensor((10, 256), dtype='float32'),
		fc2_bias: R.Tensor((10,), dtype='float32'),
	) -> R.Tensor((1, 10), dtype='float32'):
		R.func_attr({"num_input": 1})
		# Constructing a dataflow graph.
		with R.dataflow():
			permute_dims = R.permute_dims(fc1_weight, axes=None) # transpose
			matmul = R.matmul(x, permute_dims, out_dtype='void')
			add = R.add(matmul, fc1_bias)
			relu = R.nn.relu(add)
			permute_dims1 = R.permute_dims(fc2_weight, axes=None) # transpose
			matmul1 = R.matmul(relu, permute_dims1, out_dtype='void')
			add1 = R.add(matmul1, fc2_bias)
			gv = add1
			R.output(gv)
		return gv

# Identical to the one generated via tvm.relax.frontend.nn
mod_from_script = TVMScriptModule
# mod_from_script.show()

# 'zero' pipeline simply lowers down Relax to TensorIR, i.e. from datafow graph
# to code and fuses trivial operations like matmul and add (but not transpose).
# mod_op.show()

print(mod.get_global_vars())
# print(mod['forward']) # come mod.show() ma stampa solo la funzione

# Converts each Relax operation in TensorIR representation without any
# optimization whatsoever.
mod_legal = relax.transform.LegalizeOps()(mod)
# mod_legal.show()
print(mod_legal.get_global_vars())

# zero_pileline applies LegalizeOps again but all TVM's passes are idempotent.
mod_legal_opt = zero_pipeline(mod_legal)

from tvm import dlight as dl

# TODO: come posso ottenere la param_spec di mod_legal_opt?
with tvm.target.Target('cuda'):
	gpu_mod = dl.ApplyDefaultSchedule(
		dl.gpu.Matmul(),
		dl.gpu.Fallback(),
	)(mod_op)

exec = tvm.compile(gpu_mod, target='cuda')
dev = tvm.device('cuda', 0)
vm = relax.VirtualMachine(exec, dev)

data = numpy.random.rand(1, 784).astype('float32')
tvm_data = tvm.nd.array(data, dev)
gpu_params = [
		tvm.nd.array(numpy.random.rand(*param.shape).astype('float32'), device=dev)
		for _, param in param_spec]
gpu_out = vm['forward'](tvm_data, *gpu_params).numpy()
print(gpu_out)

import torch
import torchvision
from tvm.relax.frontend.torch import from_exported_program

if False:
	# Assumes CIFAR10 size (32x32)
	class ConvModel(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.conv1 = torch.nn.Conv2d(3, 6, 5)
			self.pool = torch.nn.MaxPool2d(2, 2)
			self.conv2 = torch.nn.Conv2d(6, 16, 5)
			self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
			self.fc2 = torch.nn.Linear(120, 84)
			self.fc3 = torch.nn.Linear(84, 10)

		def forward(self, x):
			x = self.pool(torch.nn.functional.relu(self.conv1(x)))
			x = self.pool(torch.nn.functional.relu(self.conv2(x)))
			# NOTE: torch.flatten is not supported for some reason...
			# x = torch.flatten(x, 1)
			x = x.view(x.size(0), -1)
			x = torch.nn.functional.relu(self.fc1(x))
			x = torch.nn.functional.relu(self.fc2(x))
			x = self.fc3(x)
			return x

	example_args = torch.randn(1, 3, 32, 32, dtype=torch.float32),

	with torch.no_grad():
		exported_program = torch.export.export(ConvModel().eval(), example_args)
	mod_from_torch = from_exported_program(
		exported_program,
		keep_params_as_input=True,
		unwrap_unit_return_tuple=True
	)
	mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
	mod_from_torch.show()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
	weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
model = model.eval()

# model.rpn = torch.compiler.disable(model.rpn)

utils.make_relu_not_inplace(model.backbone)

# Compiling the whole model is very hard, we are going to settle for the backbone
example_args = torch.randn(1, 3, 1024, 1024, dtype=torch.float32),
with torch.no_grad():
	exported_program = torch.export.export(model.backbone, example_args)
utils.replace_add_inplace_with_add(exported_program.graph)
exported_program.graph.lint()
mod_from_torch = from_exported_program(
	exported_program,
	keep_params_as_input=True,
	unwrap_unit_return_tuple=True
)
mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# mod_from_torch.show()

mod_from_torch_opt = zero_pipeline(mod_from_torch)
with tvm.target.Target('cuda'):
	gpu_mod = dl.ApplyDefaultSchedule(
		dl.gpu.Matmul(),
		dl.gpu.Fallback(),
	)(mod_from_torch_opt)

exec = tvm.compile(gpu_mod, target='cuda')
dev = tvm.device('cuda', 0)
vm = relax.VirtualMachine(exec, dev)

data = numpy.random.rand(1, 3, 1024, 1024).astype('float32')
tvm_data = tvm.nd.array(data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in params_from_torch['main']]
gpu_out = [ndarray.numpy() for ndarray in vm['main'](tvm_data, *gpu_params)]
print(gpu_out)
