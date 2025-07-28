# Ignoring this guide I can directly export the model quantized by the PyTorch
# team
# https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27
import torch, torchvision
import tvm.relax.frontend.torch
import tvm.relax.frontend.onnx
import onnx
import utils

importer = tvm.relax.frontend.torch.fx_translator.TorchFXImporter()

def convert_quant(node: torch.fx.Node):
	x = x = importer.env[node.args[0]]
	module = importer.named_modules[node.target] # getattr(fx_graph, node.target)
	scale = module.scale.item()
	zero_point = module.zero_point.item()
	return importer.block_builder.emit(
		tvm.relax.op.quantize(x, tvm.relax.const(scale), tvm.relax.const(zero_point, 'int8'))
	)

def convert_convrelu2d(node: torch.fx.Node):
	x = importer.env[node.args[0]]
	module = importer.named_modules[node.target]
	breakpoint()
	weight = importer.params[module.weight()]
	bias = importer.params.get(module.bias(), None)

	conv2d = self.block_builder.emit(
		tvm.relax.op.nn.conv2d(
			x,
			weight,
			strides=module.strides,
			padding=module.padding,
			dilation=module.dilation,
			groups=module.groups,
			data_layout="NCHW",
			kernel_layout="OIHW",
			out_dtype="float32",
		)
	)

	if bias is None:
		return conv2d
	breakpoint()
	assert len(self.shape_of(bias)) == 1
	bias = tvm.relax.op.reshape(bias, (1, -1, 1, 1))
	return self.block_builder.emit(tvm.relax.op.add(conv2d, bias))

model = torchvision.models.quantization.resnet18(
	weights=torchvision.models.quantization.ResNet18_QuantizedWeights.DEFAULT,
	progress=False, quantize=True
).eval()
path = 'build/qresnet18.onnx'
example_args = torch.randn(1, 3, 224, 224)

# print(model)
utils.make_quantization_parameter_constants(model)
utils.give_named_parameters(model)
print(len(list(model.named_parameters())))

fx_graph = torch.fx.symbolic_trace(model, concrete_args={"x": example_args})
# print(fx_graph.graph)

custom_convert_map = {
	torch.ao.nn.quantized.modules.Quantize: convert_quant,
	torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d: convert_convrelu2d,
}
mod = importer.from_fx(
	fx_graph,
	[((1, 3, 224, 224), "float32")],
	keep_params_as_input=False,
	unwrap_unit_return_tuple=False,
	no_bind_return_tuple=False,
	custom_convert_map=custom_convert_map
)
# exported_program = torch.export.export(model, (example_args, ))

torch.onnx.export(model, example_args, path)

model_onnx = onnx.load(path)
onnx.checker.check_model(model_onnx)

# It is very broken...
tvm.relax.frontend.onnx.from_onnx(model_onnx)