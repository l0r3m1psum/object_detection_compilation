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
		# print(cuda_mod)
		ex = tvm.build(cuda_mod, target)
	elif gen == 'cpu':
		ex = tvm.build(mod, 'llvm')
	else:
		raise ValueError("bad gen argument")
	return ex

def save_parameters(mod, params, path: str) -> None:
	params_name_without_input = mod['main'].params[1:]
	params_dict = {
		str(k): {
			"dtype": v.numpy().dtype.name,
			"shape": v.numpy().shape,
			"data": safetensors.numpy._tobytes(v.numpy())
		}
		for k, v in zip(params_name_without_input, params['main'])
	}
	order_metadata = {str(key): str(value) for value, key in enumerate(params_name_without_input)}
	safetensors.serialize_file(params_dict, path, metadata=order_metadata)

quantized = False

ssl._create_default_https_context = ssl._create_stdlib_context
torch.hub.set_dir(os.path.expandvars('%installdir%\\Programs\\hub'))

if quantized:
	model = torchvision.models.quantization.resnet18(
		weights=torchvision.models.quantization.ResNet18_QuantizedWeights.DEFAULT,
		progress=False, quantize=True
	).eval()
else:
	model = torchvision.models.resnet.resnet18(weights='DEFAULT').eval()

model_name = 'quantized_resnet18' if quantized else 'resnet18'
input_dtype = torch.int8 if quantized else torch.float32

utils.make_relu_not_inplace(model)
quant_params = utils.export_quant_dequant_params(model)
# FIXME if batchsize is 2 this does not work???
example_args = torch.randn(1, 3, 224, 224).to(torch.float32),
with torch.no_grad():
	exported_program = torch.export.export(model, example_args)

# utils.replace_flatten_with_view(exported_program.graph, example_args[0].shape)
# utils.replace_add_inplace_with_add(exported_program.graph)
mod = tvm.relax.frontend.torch.from_exported_program(exported_program, keep_params_as_input=True)

# breakpoint()

mod, params = tvm.relax.frontend.detach_params(mod)
# print(mod)
ex = compile(mod, 'gpu')
# print(ex.as_text())
# print(ex.as_python())
save_parameters(mod, params, f'build\\{model_name}.safetensors')
# breakpoint()
# TODO: invoke model after compilation...
ex.export_library(**utils.get_export_library_args(model_name))

# TODO: tvm.relax.transform.BindParams to bind the weights to the model
