from tvm import relax
from tvm import ir
from tvm import topi
import tvm.relax.frontend.onnx

import onnx

from typing import Dict, List
import warnings

def clamp(data, min, max):
	res = relax.op.minimum(data, relax.const(max))
	res = relax.op.maximum(relax.const(min), res)
	return res

def requantize(s, x, z):
	return clamp(relax.op.round(s*x + z.astype("float32")), -128., 127.).astype("int8")

def get_data(expr: relax.Expr, params: Dict[str, relax.Expr]) -> float|int:
	keep_params_in_input = hasattr(expr, 'data')
	if keep_params_in_input:
		array = expr.data
	else:
		_, array = params[str(expr)]
	res = array.numpy().item()
	return res

# TODO: This allows to avoid taking certain parameters from the function input and
# hardcoding  them as constants allowing for constant folding optimizations etc...
# This is a bit of an hack, the way that it should be done instead is to create
# a new function with the paramiters binded the value of the metadata so that
# they are effectivelly constant in the new function.
def get_arg(expr: relax.Expr, params: Dict[str, relax.Expr]) -> relax.Expr:
	keep_params_in_input = bool(params)
	if keep_params_in_input:
		_, array = params[str(expr)]
		if isinstance(array, relax.Expr):
			res = array
		else:
			# Because non scalars are picked from "metadata"
			if array.shape:
				res = expr
			else:
				res = relax.const(array)
	else:
		res = expr
	return res

# TODO: implement custom relax function for quantization and dequantization with
# non 'int8' zero_point.

class QuantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-10
		x = inputs[0]
		y_scale = get_arg(inputs[1], params[1])
		y_zero_point = get_arg(inputs[2], params[1])

		y_zero_point_dtype = y_zero_point.struct_info.dtype
		if y_zero_point_dtype != "int8":
			warnings.warn("Unsupported zero_point dtype for quantization '%s' casting it to 'int8" % y_zero_point_dtype)
			# TODO: add check for cast to not lose precision
			y_zero_point = y_zero_point.astype("int8")

		return relax.op.quantize(x, y_scale, y_zero_point)

class DequantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-10
		x = inputs[0]
		x_scale = get_arg(inputs[1], params[1])
		x_zero_point = get_arg(inputs[2], params[1])
		assert len(x_scale.data.shape) == len(x_zero_point.data.shape)

		x_zero_point_dtype = x_zero_point.struct_info.dtype
		if x_zero_point.struct_info.shape:
			warnings.warn("Non scalar zero_point is supported at the Relax "
				"level but it will fail when calling "
				"relax.transform.LegalizeOps")
		if x_zero_point_dtype != "int8":
			warnings.warn("Unsupported zero_point dtype for dequantization '%s' casting it to 'int8" % x_zero_point_dtype)
			# TODO: add check for cast to not lose precision
			x_zero_point = x_zero_point.astype("int8")

		# NOTE: this is very best effort we assume that the data is a
		# convolution kernel in OIHW, we do not know how to drop this
		# assumption. The axis to is the first because we assume per-tensor
		# dequantization.
		is_conv_kernel = len(x_scale.data.shape) != 0
		axis = 0 if is_conv_kernel else -1
		# TODO: add check for cast to not loose precision
		return relax.op.dequantize(x, x_scale, x_zero_point, axis)

class QLinearConv(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QLinearConv.html#qlinearconv-10
		X   = inputs[0]
		X_s = get_data(inputs[1], params[1])
		X_z = get_arg(inputs[2], params[1])
		W   = get_arg(inputs[3], params[1])
		W_s = get_data(inputs[4], params[1])
		W_z = get_arg(inputs[5], params[1])
		Y_s = get_data(inputs[6], params[1])
		Y_z = get_arg(inputs[7], params[1])
		B   = get_arg(inputs[8], params[1])
		assert len(X.struct_info.shape) == 4
		assert len(W.struct_info.shape) == 4
		assert get_data(inputs[5], params[1]) == 0

		M = relax.const((X_s*W_s)/Y_s)
		conv = relax.op.nn.conv2d(
			data=(X.astype("int32") - X_z.astype("int32")),
			weight=(W.astype("int32") - W_z.astype("int32")),
			strides=attr.get("strides", 1),
			padding=attr.get("pads", 0),
			dilation=attr.get("dilations", 1),
			groups=attr.get("group", 1),
			data_layout="NCHW",
			kernel_layout="OIHW",
			out_dtype="int32"
		)
		res = (conv + relax.op.reshape(B, (1, -1, 1, 1))).astype("float32")
		res = requantize(M, res, Y_z)
		return res

class QLinearAdd(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearAdd
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/micro/kernels/add_common.cc#L48C62-L48C64
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/reference/integer_ops/add.h#L211
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/reference/integer_ops/add.h#L204
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/reference/integer_ops/add.h#L180
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/common.h#L269
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/common.cc#L22

		# C = (A_s * (A - A_z) + B_s * (B - B_z))/C_s + C_z
		A   = inputs[0]
		A_s = get_data(inputs[1], params[1])
		A_z = get_arg(inputs[2], params[1])
		B   = inputs[3]
		B_s = get_data(inputs[4], params[1])
		B_z = get_arg(inputs[5], params[1])
		C_s = get_data(inputs[6], params[1])
		C_z = get_arg(inputs[7], params[1])

		M_1 = relax.const(A_s/C_s)
		M_2 = relax.const(B_s/C_s)
		res = M_1*(A.astype("int32") - A_z.astype("int32")).astype("float32") \
			+ M_2*(B.astype("int32") - B_z.astype("int32")).astype("float32")
		res = requantize(relax.const(1.0), res, C_z)
		return res

class QGemm(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QGemm
		alpha  = attr.get('alpha', 0.0)
		transA = attr.get('transA', 0)
		transB = attr.get('transB', 0)

		assert inputs[0].args[0].struct_info.ndim == inputs[3].struct_info.ndim
		assert inputs[3].struct_info.ndim == 2
		assert alpha == 1.0, "alpha != 1.0 requires some work to keep it integer only"

		# A has shape MxN and B has shape NxO
		n = relax.const(topi.utils.get_const_tuple(relax.get_shape_of(inputs[0]))[1])
		A   = inputs[0]
		A_s = get_data(inputs[1], params[1])
		A_z = get_arg(inputs[2], params[1])
		B   = get_arg(inputs[3], params[1])
		B_s = get_data(inputs[4], params[1])
		B_z = get_arg(inputs[5], params[1])
		C   = get_arg(inputs[6], params[1])
		Y_s = get_data(inputs[7], params[1])
		Y_z = get_arg(inputs[8], params[1])
		AT = relax.op.permute_dims(A) if transA else A
		BT = relax.op.permute_dims(B) if transB else B

		M = relax.const((A_s*B_s)/Y_s)
		matmul = relax.op.matmul(
			(AT.astype("int32") - A_z.astype("int32")),
			(BT.astype("int32") - B_z.astype("int32")),
			out_dtype="int32"
		)
		res = (matmul + C).astype("float32")
		res = requantize(M, res, Y_z)
		return res

class QLinearMatMul(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html#qlinearmatmul-10
		A = inputs[0]
		A_s = get_data(inputs[1], params[1])
		A_z = get_arg(inputs[2], params[1])
		B = get_arg(inputs[3], params[1])
		B_s = get_data(inputs[4], params[1])
		B_z = get_arg(inputs[5], params[1])
		Y_s = get_data(inputs[6], params[1])
		Y_z = get_arg(inputs[7], params[1])

		M = relax.const((A_s*B_s)/Y_s)
		matmul = relax.op.matmul(
			(A.astype("int32") - A_z.astype("int32")),
			(B.astype("int32") - B_z.astype("int32")),
			out_dtype="int32"
		)
		res = matmul.astype("float32")
		res = requantize(M, res, Y_z)
		return res

class QLinearGlobalAveragePool(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearGlobalAveragePool
		x = inputs[0]
		x_s = get_data(inputs[1], params[1])
		x_z = get_arg(inputs[2], params[1])
		y_s = get_data(inputs[3], params[1])
		y_z = get_arg(inputs[4], params[1])

		M = relax.const(x_s/y_s)
		# NOTE: that astype("int32") is needed because for some reason Relax
		# infers the type of the avg_pool2d expression to be an "int64". Is this
		# a TVM bug?
		avg_pool2d = relax.op.nn.avg_pool2d(
			data=(x).astype("int32"),
			pool_size=x.struct_info.shape.values[2:],
		).astype("int32") - x_z.astype("int32")
		res = avg_pool2d.astype("float32")
		res = requantize(M, res, y_z)
		return res

convert_map = {
	"QuantizeLinear": QuantizeLinear,
	"DequantizeLinear": DequantizeLinear,
	"QLinearConv": QLinearConv,
	"QGemm": QGemm,
	"QLinearMatMul": QLinearMatMul,
	"QLinearAdd": QLinearAdd,
	"QLinearGlobalAveragePool": QLinearGlobalAveragePool,
}

def from_onnx(
		model: onnx.GraphProto,
		shape_dict: Dict[str, List] | None = None,
		dtype_dict: str | Dict[str, str] | None = 'float32',
		opset: int | None = None,
		keep_params_in_input: bool = False,
		sanitize_input_names: bool = True
	) -> ir.IRModule:

	def my_get_convert_map() -> dict:
		return original_get_convert_map() | convert_map

	original_get_convert_map = relax.frontend.onnx.onnx_frontend._get_convert_map
	relax.frontend.onnx.onnx_frontend._get_convert_map = my_get_convert_map
	try:
		res = relax.frontend.onnx.from_onnx(
			model,
			shape_dict,
			dtype_dict,
			opset,
			keep_params_in_input,
			sanitize_input_names,
		)
	finally:
		relax.frontend.onnx.onnx_frontend._get_convert_map = original_get_convert_map
	return res

from vtar.tir.util import prod # TODO: move prod in a util module inside vtar

def make_initializers_hollow(model: onnx.GraphProto):

	for tensor in model.graph.initializer:
		tensor.data_location = onnx.TensorProto.EXTERNAL

		tensor.ClearField("raw_data")

		tensor.external_data.clear()
		entry = tensor.external_data.add()
		entry.key = "location"
		entry.value = "weights.bin"

	print(onnx.printer.to_text(model))

def convert_weights_to_inputs(model: onnx.GraphProto):
	graph = model.graph

	initializers_to_keep = []
	existing_input_names = {inp.name for inp in graph.input}
	limit_elements = 1

	for init in graph.initializer:
		size = prod(init.dims)

		if size <= limit_elements:
			initializers_to_keep.append(init)
		elif init.name not in existing_input_names:
			input_arg = onnx.helper.make_tensor_value_info(
				name=init.name,
				elem_type=init.data_type,
				shape=init.dims
			)

			graph.input.append(input_arg)

	graph.initializer.clear()
	graph.initializer.extend(initializers_to_keep)

	print(onnx.printer.to_text(model))

def move_constants_to_initializers(model: onnx.GraphProto):
	graph = model.graph

	new_nodes = []

	for node in graph.node:
		if node.op_type == "Constant":
			# TODO: why is it done like this?
			tensor_attr = next((attr for attr in node.attribute if attr.name == "value"), None)

			if tensor_attr:
				tensor_proto = tensor_attr.t
				tensor_proto.name = node.output[0]
				graph.initializer.append(tensor_proto)
				continue

		new_nodes.append(node)

	del graph.node[:]
	graph.node.extend(new_nodes)

	print(onnx.printer.to_text(model))
