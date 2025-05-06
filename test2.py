import numpy
import torch
import torchvision
import tvm
import tvm.relax.frontend.torch
import utils

# here you have to use mod['main']
def resnet18_from_torchvision() -> tvm.IRModule:
	resnet18_model = torchvision.models.resnet.resnet18
	resnet18_weights = torchvision.models.resnet.ResNet18_Weights.DEFAULT

	torch_model = resnet18_model(weights=resnet18_weights).eval()
	utils.make_relu_not_inplace(torch_model)

	example_args = torch.randn(2, 3, 224, 224, dtype=torch.float32),

	with torch.no_grad():
		exported_program = torch.export.export(torch_model, example_args)

	utils.replace_flatten_with_view(exported_program.graph, example_args[0].shape)
	utils.replace_add_inplace_with_add(exported_program.graph)

	mod = tvm.relax.frontend.torch.from_exported_program(exported_program,
		keep_params_as_input=True)

	mod, params = tvm.relax.frontend.detach_params(mod)

	return mod

if True:
	class MLPModel(tvm.relax.frontend.nn.Module):
		def __init__(self):
			super().__init__()
			self.fc1 = tvm.relax.frontend.nn.Linear(784, 256)
			self.relu1 = tvm.relax.frontend.nn.ReLU()
			self.fc2 = tvm.relax.frontend.nn.Linear(256, 10)
		def forward(self, x):
			x = self.fc1(x)
			x = self.relu1(x)
			x = self.fc2(x)
			return x

	mod, param_spec = MLPModel().export_tvm(
		spec={'forward': {'x': tvm.relax.frontend.nn.spec.Tensor((1, 784), 'float32')}}
	)

TOTAL_TRIALS = 10
# target = tvm.target.Target('cuda')
# print(tvm.target.Target.list_kinds())
target = tvm.target.Target.from_device('cuda:0')
# target = tvm.target.Target('llvm -num-cores 24')
work_dir = 'tuning_logs'

# Using a small MLP it takes reasonable time... But still takes half an hour and
# there are all N/A in time based metrics which i dont know if it is a bug...
mod = tvm.relax.get_pipeline('static_shape_tuning', target=target,
	total_trials=TOTAL_TRIALS)(mod)
mod['forward'].show()
