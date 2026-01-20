"""
ssl._create_default_https_context = ssl._create_stdlib_context
torch.hub.set_dir(os.path.expandvars('%installdir%\\Programs\\hub'))

model = torchvision.models.quantization.resnet18(
	weights=torchvision.models.quantization.ResNet18_QuantizedWeights.DEFAULT,
	progress=False, quantize=True
).eval()

example_args = torch.randn(1, 3, 224, 224).to(torch.float32),
with torch.no_grad():
	exported_program = torch.export.export(model, example_args)

mod = tvm.relax.frontend.torch.from_exported_program(exported_program)
"""

import torch

# TODO: create an interator class that does the dfs
def make_relu_not_inplace(module: torch.nn.Module) -> None:
	for name, submodule in module.named_children():
		if name == 'relu':
			submodule.inplace = False
		make_relu_not_inplace(submodule)

# Saddly "quantized nyi in meta tensors" (nyi = not yet implemented)
def make_quantization_parameter_constants(module: torch.nn.Module) -> None:
	for name, submodule in module.named_children():
		if name == 'quant':
			self = submodule
			scale = float(self.scale)
			zero_point = int(self.zero_point)
			submodule.forward = lambda X: torch.quantize_per_tensor(X, scale, zero_point, self.dtype)
		make_quantization_parameter_constants(submodule)

import types

def my_named_parameters(self):
	yield from (('weight', self.weight()), ('bias', self.bias()),)

def give_children_named_parameters(module: torch.nn.Module) -> None:
	for name, submodule in module.named_children():
		if isinstance(submodule, torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule):
			if name.startswith('conv'):
				submodule.named_parameters = types.MethodType(my_named_parameters, submodule)
		give_children_named_parameters(submodule)

def top_module_named_parameters(module: torch.nn.Module):
	for name, submodule in module.named_children():
		yield from submodule.named_parameters()
		top_module_named_parameters(submodule)

def give_named_parameters(module: torch.nn.Module) -> None:
	module.named_parameters = types.MethodType(top_module_named_parameters, module)
	give_children_named_parameters(module)

def export_quant_dequant_params(module: torch.nn.Module):
	res = []
	for name, submodule in module.named_children():
		try:
			scale = float(submodule.scale)
			zero_point =  int(submodule.zero_point)
			# QuantizedLinear has also the qscheme attribute
			# Quantize        has also the dtype   attribute
			res.append((name, scale, zero_point))
		except AttributeError:
			pass
		res.extend(export_quant_dequant_params(submodule))
	return res

# https://pytorch.org/docs/stable/torch.compiler_transformations.html
# TODO: maybe use https://pytorch.org/docs/stable/fx.html#subgraph-rewriting-with-replace-pattern instead
def replace_add_inplace_with_add(graph: torch.fx.Graph) -> None:
	for node in graph.nodes:
		if node.op == 'call_function' and node.target == torch.ops.aten.add_.Tensor:
			node.target = torch.ops.aten.add.Tensor
	graph.lint()

# TODO: verify that torch.flatten(x, 1) is equal to x.view(x.size(0), -1)
# Probably a better way to do it is use: https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
def replace_flatten_with_view(graph: torch.fx.Graph, input_shape: torch.Size) -> None:
	for node in graph.nodes:
		if (node.op == 'call_function'
			and node.target == torch.ops.aten.flatten.using_ints
			and node.args[1] == 1):
			node.target = torch.ops.aten.view.default
			input_node: torch.fx.node.Node = node.args[0]
			batch_size = input_shape[0]
			node.args = input_node, [batch_size, -1]
	graph.lint()
