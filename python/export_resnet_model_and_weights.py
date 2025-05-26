import torch
import torchvision
import safetensors.torch
import tvm
import tvm.relax.frontend.torch

import ssl
import os

import utils

def compile(mod, gen):
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
		print(cuda_mod)
		ex = tvm.compile(cuda_mod, target)
	elif gen == 'cpu':
		ex = tvm.compile(mod, 'llvm')
	return ex


ssl._create_default_https_context = ssl._create_stdlib_context
torch.hub.set_dir(os.path.expandvars('%installdir%\\Programs\\hub'))
model = torchvision.models.resnet.resnet18(weights='DEFAULT').eval()
safetensors.torch.save_file(model.state_dict(), 'build\\resnet18.safetensors')

utils.make_relu_not_inplace(model)
# FIXME if batchsize is 2 this does not work???
example_args = torch.randn(1, 3, 224, 224, dtype=torch.float32),
with torch.no_grad():
	exported_program = torch.export.export(model, example_args)

# utils.replace_flatten_with_view(exported_program.graph, example_args[0].shape)
# utils.replace_add_inplace_with_add(exported_program.graph)
mod = tvm.relax.frontend.torch.from_exported_program(exported_program, keep_params_as_input=True)

# breakpoint()

mod, params = tvm.relax.frontend.detach_params(mod)
print(mod)
ex = compile(mod, 'gpu')
input()
ex.export_library(**utils.get_export_library_args('resnet18'))

# TODO: tvm.relax.transform.BindParams to bind the weights to the model