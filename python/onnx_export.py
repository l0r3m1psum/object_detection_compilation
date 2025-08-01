# Ignoring this guide I can directly export the model quantized by the PyTorch
# team
# https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27
import torch, torchvision
import tvm.relax.frontend.torch
import tvm.relax.frontend.onnx
import onnx
import utils

"""
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
"""

class QuantizeLinear(tvm.relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#inputs
		x = inputs[0]
		y_scale = inputs[1]
		y_zero_point = inputs[2]

		return tvm.relax.op.quantize(x, y_scale, y_zero_point)

class QLinearConv(tvm.relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		input0_struct_info = inputs[0].args[0].struct_info
		# https://onnx.ai/onnx/operators/onnx__QLinearConv.html#inputs
		x            = inputs[0]
		x_scale      = inputs[1]
		x_zero_point = inputs[2]
		w            = inputs[3]
		w_scale      = inputs[4]
		w_zero_point = inputs[5]
		y_scale      = inputs[6]
		y_zero_point = inputs[7]
		B            = inputs[8]

		if hasattr(input0_struct_info, "ndim"):
			ndim = input0_struct_info.ndim
		else:
			ndim = len(input0_struct_info.shape)

		if ndim == 3:
			op = tvm.relax.op.nn.conv1d
			data_layout = "NCW"
			kernel_layout = "OIW"
		elif ndim == 4:
			op = tvm.relax.op.nn.conv2d
			data_layout = "NCHW"
			kernel_layout = "OIHW"
		elif ndim == 5:
			op = tvm.relax.op.nn.conv3d
			data_layout = "NCDHW"
			kernel_layout = "OIDHW"
		else:
			raise NotImplementedError("Ndim > 5 not supported for convolution.")

		conv_out = bb.normalize(
			op(
				data=x,
				weight=w,
				strides=attr.get("strides", 1),
				padding=attr.get("pads", 0),
				dilation=attr.get("dilations", 1),
				groups=attr.get("group", 1),
				data_layout=data_layout,
				kernel_layout=kernel_layout,
			)
		)
		# FIXME: this is 100% wrong
		if B is not None:
			bias = tvm.relax.op.reshape(B, [1, -1] + [1] * (ndim - 2))
			conv_out = tvm.relax.op.add(tvm.relax.op.astype(conv_out, "int32"), bias)

		# FIXME: this is wrong
		return tvm.relax.op.astype(conv_out, "int8")

class DequantizeLinear(tvm.relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#inputs
		x = inputs[0]
		x_scale = inputs[1]
		x_zero_point = inputs[2]

		return tvm.relax.op.dequantize(x, x_scale, x_zero_point)

class QLinearAdd(tvm.relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearAdd
		A            = inputs[0]
		A_scale      = inputs[1]
		A_zero_point = inputs[2]
		B            = inputs[3]
		B_scale      = inputs[4]
		B_zero_point = inputs[5]
		C_scale      = inputs[6]
		C_zero_point = inputs[7]
		# FIXME: this is wrong
		return tvm.relax.op.add(A, B)

class QGemm(tvm.relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QGemm
		alpha  = attr.get('alpha', 0.0)
		transA = attr.get('transA', 0)
		transB = attr.get('transB', 0)
		A            = inputs[0]
		a_scale      = inputs[1]
		a_zero_point = inputs[2]
		B            = inputs[3]
		b_scale      = inputs[4]
		b_zero_point = inputs[5]
		C            = inputs[6]
		y_scale      = inputs[7]
		y_zero_point = inputs[8]
		# FIXME: this is wrong
		# TODO: tvm.relax.op.permute_dims
		assert A.args[0].struct_info.ndim == B.struct_info.ndim
		assert B.struct_info.ndim == 2
		AT = tvm.relax.op.permute_dims(A) if transA else A
		BT = tvm.relax.op.permute_dims(B) if transB else B
		AB = tvm.relax.op.matmul(AT, BT)
		alphaAB = tvm.relax.op.multiply(AB, alpha) if alpha != 1.0 else AB  
		# FIXME: this is wrong
		return tvm.relax.op.astype(tvm.relax.op.add(tvm.relax.op.astype(alphaAB, "int32"), C), "int8")

class QLinearGlobalAveragePool(tvm.relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearGlobalAveragePool
		X = inputs[0]
		x_scale = inputs[0]
		x_zero_point = inputs[0]
		y_scale = inputs[0]
		y_zero_point = inputs[0]

		# FIXME: this is wrong
		return tvm.relax.op.collapse_sum_to(X, X.struct_info.shape.values[:2]+[1,1])

# TODO: Capire come fare la requantizzatione da int32 a int8

convert_map = {
	"QuantizeLinear": QuantizeLinear,
	"QLinearConv": QLinearConv,
	"QGemm": QGemm,
	"QLinearAdd": QLinearAdd,
	"QLinearGlobalAveragePool": QLinearGlobalAveragePool,
	"DequantizeLinear": DequantizeLinear,
}

original_get_convert_map = tvm.relax.frontend.onnx.onnx_frontend._get_convert_map
def my_get_convert_map() -> dict:
	return original_get_convert_map() | convert_map
tvm.relax.frontend.onnx.onnx_frontend._get_convert_map = my_get_convert_map

path = "build/resnet18_int8.onnx"
model_onnx = onnx.load(path)

mod = tvm.relax.frontend.onnx.from_onnx(model_onnx, keep_params_in_input=False)

import os, sys
sys.path.append(os.path.join(os.getcwd(), "submodules\\tvm\\vta\\python"))
import vta

import ctypes
# ctypes.cdll.kernel32.DebugBreak()

if False:
	env = vta.get_env()
	mod = vta.top.graph_pack(
		mod["main"],
		env.BATCH,
		env.BLOCK_OUT,
		env.WGT_WIDTH,
	)

zero_pipeline = tvm.relax.get_pipeline('zero')
mod = zero_pipeline(mod)

vta.build(mod)
