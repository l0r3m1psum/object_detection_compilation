import torch
import torchvision
import safetensors.torch
import safetensors.numpy
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
	else:
		raise ValueError("bad gen argument")
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
print(ex.as_text())
# print(ex.as_python())
import collections
res = collections.OrderedDict((str(k), {"dtype": v.numpy().dtype.name, "shape": v.numpy().shape, "data": v.numpy().tobytes()}) for k, v in zip(mod['main'].params[1:], params['main']) )
safetensors.serialize_file(res, 'build\\resnet18.safetensors', metadata={str(key): str(value) for value, key in enumerate(mod['main'].params[1:])})
breakpoint()
# TODO: invoke after compilation...
ex.export_library(**utils.get_export_library_args('resnet18'))

# TODO: tvm.relax.transform.BindParams to bind the weights to the model
