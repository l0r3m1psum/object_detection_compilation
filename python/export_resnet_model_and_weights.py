import torch
import torchvision
import safetensors.torch
import tvm
import tvm.relax.frontend.torch

import ssl
import os

import utils

ssl._create_default_https_context = ssl._create_stdlib_context
torch.hub.set_dir(os.path.expandvars('%installdir%\\Programs\\hub'))
model = torchvision.models.resnet.resnet18(weights='DEFAULT').eval()
safetensors.torch.save_file(model.state_dict(), 'build\\resnet50.safetensors')

utils.make_relu_not_inplace(model)
# FIXME if batchsize is 2 this does not work???
example_args = torch.randn(1, 3, 224, 224, dtype=torch.float32),
with torch.no_grad():
	exported_program = torch.export.export(model, example_args)

# utils.replace_flatten_with_view(exported_program.graph, example_args[0].shape)
# utils.replace_add_inplace_with_add(exported_program.graph)
mod = tvm.relax.frontend.torch.from_exported_program(exported_program, keep_params_as_input=True)

# mod, params = tvm.relax.frontend.detach_params(mod)
ex = tvm.compile(mod, 'llvm')
ex.export_library(**utils.get_export_library_args('resnet'))

# TODO: tvm.relax.transform.BindParams to bind the weights to the model