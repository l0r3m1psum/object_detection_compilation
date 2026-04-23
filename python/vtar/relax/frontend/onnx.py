from tvm import ir, topi, relax
from tvm.script import relax as R
import tvm.relax.frontend.onnx
from tvm.relax.frontend.onnx.onnx_frontend import get_constant

from ... import relax as vtar_relax

import onnx
import numpy

from typing import Dict, List, Tuple, Optional
import warnings
import math

from ... import topi as mytopi

# TODO: implement custom relax function for quantization and dequantization with
# non 'int8' zero_point.

# TODO: what is the difference between using get_constant(inputs[n], params) and
# just inputs[n]?

def convert_zero_point_for_relax(zero_point: relax.Constant) -> relax.Constant:
	"""Input and output type are relax.Constant because quantize and dequantize
	only support immediate values for scale and zero point."""
	zero_point_dtype = zero_point.struct_info.dtype
	if zero_point_dtype != "int8":
		if not ((0 <= zero_point.data.numpy()) & (zero_point.data.numpy() < 128)).all():
			warnings.warn("Relax only supports signed integers types for "
				"zero points. The values of this particular zero_point "
				"cannot be safely casted to int8. Clipping is going to be "
				"performed.")
		zero_point = relax.const(numpy.clip(zero_point.data.numpy(), -128, 127), dtype="int8")
	return zero_point

class QuantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v13(cls, bb, inputs, attr, params: List[Dict[str, relax.Var]]):
		# https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13
		x = inputs[0]
		y_scale = get_constant(inputs[1], params)
		y_zero_point = get_constant(inputs[2], params)
		axis = attr.get("axis", 1)

		y_zero_point  = convert_zero_point_for_relax(y_zero_point)

		return relax.op.quantize(x, y_scale, y_zero_point, axis)

class DequantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v13(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-13
		params_var, params_val = params
		x = inputs[0]
		x_scale = get_constant(inputs[1], params)
		x_zero_point = get_constant(inputs[2], params)
		axis = attr.get("axis", 1)

		x_zero_point  = convert_zero_point_for_relax(x_zero_point)

		return relax.op.dequantize(x, x_scale, x_zero_point, axis)

# Bilinear operators ###########################################################

class QLinearConv(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QLinearConv.html#qlinearconv-10
		X   = inputs[0]
		X_s = get_constant(inputs[1], params)
		X_z = get_constant(inputs[2], params)
		W   = get_constant(inputs[3], params)
		W_s = get_constant(inputs[4], params)
		W_z = get_constant(inputs[5], params)
		Y_s = get_constant(inputs[6], params)
		Y_z = get_constant(inputs[7], params)
		B   = get_constant(inputs[8], params)
		assert len(X.struct_info.shape) == 4
		assert len(W.struct_info.shape) == 4
		# assert get_constant(inputs[5], params) == 0

		auto_pad = attr.get("auto_pad", b"NOTSET").decode()

		# NOTE: LLM generated but it seems good
		ih, iw = topi.utils.get_const_tuple(relax.get_shape_of(X))[2:]
		kh, kw = topi.utils.get_const_tuple(relax.get_shape_of(W))[2:]
		sh, sw = attr.get("strides", (1, 1))
		dh, dw = attr.get("dilations", (1, 1))
		# 1. Calculate Effective Kernel Size (accounting for dilation)
		k_eff_h = dh * (kh - 1) + 1
		k_eff_w = dw * (kw - 1) + 1
		# 2. Calculate Output Size (Ceil mode for SAME)
		oh = math.ceil(ih / sh)
		ow = math.ceil(iw / sw)
		# 3. Calculate Total Padding needed
		pad_h_total = max(0, (oh - 1) * sh + k_eff_h - ih)
		pad_w_total = max(0, (ow - 1) * sw + k_eff_w - iw)

		if auto_pad == "NOTSET":
			pad_left, pad_right, pad_top, pad_bottom = attr.get("pads", (0, 0, 0, 0))
		elif auto_pad == "SAME_UPPER":
			pad_left = pad_w_total // 2
			pad_right = pad_w_total - pad_left
			pad_top = pad_h_total // 2
			pad_bottom = pad_h_total - pad_top
		elif auto_pad == "SAME_LOWER":
			pad_right = pad_w_total // 2
			pad_left = pad_w_total - pad_right
			pad_bottom = pad_h_total // 2
			pad_top = pad_h_total - pad_bottom
		elif auto_pad == "VALID":
			pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
		else:
			raise ValueError("Invalid auto_pad attribute '%s'" % auto_pad)
		padding = pad_left, pad_right, pad_top, pad_bottom

		args = [
			X, X_s, X_z,
			W, W_s, W_z,
			   Y_s, Y_z,
		]
		if B:
			args.append(B)
		kwargs = dict(
			strides=attr.get("strides", 1),
			padding=padding,
			dilation=attr.get("dilations", 1),
			groups=attr.get("group", 1),
			data_layout="NCHW",
			kernel_layout="OIHW",
		)
		res = vtar_relax.op.qnn.conv2d(*args, **kwargs)
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

		A   = inputs[0]
		A_s = get_constant(inputs[1], params)
		A_z = get_constant(inputs[2], params)
		B   = get_constant(inputs[3], params)
		B_s = get_constant(inputs[4], params)
		B_z = get_constant(inputs[5], params)
		C   = get_constant(inputs[6], params)
		Y_s = get_constant(inputs[7], params)
		Y_z = get_constant(inputs[8], params)

		A = relax.op.permute_dims(A) if transA else A
		B = relax.op.permute_dims(B) if transB else B
		# TODO: if both are constant pre-multiply.
		A_s = A_s if alpha == 1.0 else relax.const(alpha)*A_s

		res = vtar_relax.op.qnn.linear(
			A, A_s, A_z,
			B, B_s, B_z,
			   Y_s, Y_z,
			C,
		)
		return res

class QLinearMatMul(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html#qlinearmatmul-10
		A = inputs[0]
		A_s = get_constant(inputs[1], params).data.numpy()
		A_z = get_constant(inputs[2], params)
		B = get_constant(inputs[3], params)
		B_s = get_constant(inputs[4], params).data.numpy()
		B_z = get_constant(inputs[5], params)
		Y_s = get_constant(inputs[6], params).data.numpy()
		Y_z = get_constant(inputs[7], params)

		if len(relax.get_shape_of(A)) == 1:
			A = bb.normalize(relax.op.expand_dims(A, axis=0))

		if len(relax.get_shape_of(B)) == 1:
			B = bb.normalize(relax.op.expand_dims(B, axis=1))

		M = relax.const((A_s*B_s)/Y_s)
		matmul = do_matmul(A, A_z, B, B_z)
		res = matmul.astype("float32")
		res = requantize(M, res, Y_z)
		return res

class QLinearMul(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls: type, bb: relax.BlockBuilder, inputs: relax.frontend.onnx.onnx_frontend.onnx_input, attr: dict, params: List[Dict[str, relax.Var]]) -> relax.Expr:
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearMul
		A = inputs[0]
		A_s = get_constant(inputs[1], params).data.numpy()
		A_z = get_constant(inputs[2], params)
		B = get_constant(inputs[3], params)
		B_s = get_constant(inputs[4], params).data.numpy()
		B_z = get_constant(inputs[5], params)
		C_s = get_constant(inputs[6], params).data.numpy()
		C_z = get_constant(inputs[7], params)

		M = relax.const((A_s*B_s)/C_s)
		if False:
			mul = (A.astype("int32") - const_astype(A_z, "int32"))*(const_astype(B, "int32") - const_astype(B_z, "int32"))
		else:
			# This version can be performed by VTA loading from int8 and
			# outputing int32 using the GEMM core with diagonal matrices in 1
			# clock cycle.
			mul = (
				const_astype(A_z, "int32")*const_astype(B_z, "int32")
				- const_astype(B_z, "int32")*A.astype("int32")
				- const_astype(A_z, "int32")*const_astype(B, "int32")
				+ A.astype("int32")*const_astype(B, "int32")
			)
		res = mul.astype("float32")
		res = requantize(M, res, C_z)
		return res

# Linear Operators #############################################################

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
		A_s = get_constant(inputs[1], params)
		A_z = get_constant(inputs[2], params)
		B   = inputs[3]
		B_s = get_constant(inputs[4], params)
		B_z = get_constant(inputs[5], params)
		C_s = get_constant(inputs[6], params)
		C_z = get_constant(inputs[7], params)

		res = vtar_relax.op.qnn.add(
			A, A_s, A_z,
			B, B_s, B_z,
			   C_s, C_z,
		)

		return res

# NOTE: QLinearSplit does not exits...
# Concatenation and Split as linear operations
# c(x, y) = [I 0]' x + [0 I]' y
# s(x) = [I 0] x, [0 I] x
# c(x, y) = [I 0]' (s_x/s_c)(x - z_x) + [0 I]' (s_y/s_c)(y - z_y) + z_c
class QLinearConcat(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls: type, bb: relax.BlockBuilder, inputs: relax.frontend.onnx.onnx_frontend.onnx_input, attr: dict, params: List[Dict[str, relax.Var]]) -> relax.Expr:
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearConcat
		y_s = get_constant(inputs[0], params).data.numpy()
		y_z = get_constant(inputs[1], params)
		xs = [get_constant(input, params) for i, input in enumerate(inputs[2:]) if i % 3 == 0]
		xs_s = [get_constant(input, params).data.numpy() for i, input in enumerate(inputs[2:]) if i % 3 == 1]
		xs_z = [get_constant(input, params) for i, input in enumerate(inputs[2:]) if i % 3 == 2]

		axis = attr['axis']
		# Scales and zero_zeropoint needs to be rasheped based on the axis?

		Ms = [relax.const(x_s/y_s) for x_s in xs_s]

		res = relax.op.concat(
			[M*(x.astype("int32") - const_astype(x_z, "int32")).astype("float32")
			for M, x, x_z in zip(Ms, xs, xs_z)],
			axis
		)

		res = requantize(relax.const(1.0), res, y_z)
		return res

# Single Argument Operators ####################################################

# TODO: Average pool performs the average by dividing by a constant n. This
# means that it can be implemented without using an explicit integer division
# using Chapter 10 of Hacker's Delight 2nd Edition.

class QLinearGlobalAveragePool(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearGlobalAveragePool
		x = inputs[0]
		x_s = get_constant(inputs[1], params)
		x_z = get_constant(inputs[2], params)
		y_s = get_constant(inputs[3], params)
		y_z = get_constant(inputs[4], params)
		channels_last = attr["channels_last"]

		# relax.op.nn.avg_pool2d(...).astype("int32")
		# NOTE: that astype("int32") is needed because for some reason Relax
		# infers the type of the avg_pool2d expression to be an "int64". Is this
		# a TVM bug?
		res = vtar_relax.op.qnn.avg_pool2d(
			x, x_s, x_z,
			   y_s, y_z,
			pool_size=x.struct_info.shape.values[2:],
			layout="NHWC" if channels_last else "NCHW",
		)
		return res

class QLinearAveragePool(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls: type, bb: relax.BlockBuilder, inputs: relax.frontend.onnx.onnx_frontend.onnx_input, attr: dict, params: List[Dict[str, relax.Var]]) -> relax.Expr:
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearAveragePool
		x = inputs[0]
		x_s = get_constant(inputs[1], params)
		x_z = get_constant(inputs[2], params)
		y_s = get_constant(inputs[3], params)
		y_z = get_constant(inputs[4], params)

		auto_pad = attr['auto_pad']
		ceil_mode = attr['ceil_mode']
		channels_last = attr['channels_last']
		count_include_pad = attr['count_include_pad']
		kernel_shape = attr['kernel_shape']
		pads = attr.get("pads", [0])
		strides = attr['strides']

		res = vtar_relax.op.qnn.avg_pool2d(
			x, x_s, x_z,
			   y_s, y_z,
			strides=strides,
			padding=pads,
			ceil_mode=ceil_mode,
			count_include_pad=count_include_pad,
			layout="NHWC" if channels_last else "NCHW",
		)

		return res

class QLinearLeakyRelu(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls: type, bb: relax.BlockBuilder, inputs: relax.frontend.onnx.onnx_frontend.onnx_input, attr: dict, params: List[Dict[str, relax.Var]]) -> relax.Expr:
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearLeakyRelu
		# The function is homogeneous and could be used in re-scale.
		# y = a*max(0,x)
		# q_y = s_x/s_y * a * max(0,q_x-z_x) + z_y
		x = inputs[0]
		x_s = get_constant(inputs[1], params)
		x_z = get_constant(inputs[2], params)
		y_s = get_constant(inputs[3], params)
		y_z = get_constant(inputs[4], params)

		alpha = attr["alpha"]

		relax.op.nn.leakyrelu
		breakpoint()

# com.microsoft.QLinearReduceMean
# com.microsoft.QLinearSigmoid
# com.microsoft.QLinearSoftmax
# com.microsoft.QLinearWhere

convert_map = {
	"QuantizeLinear": QuantizeLinear,
	"DequantizeLinear": DequantizeLinear,
	"QLinearConv": QLinearConv,
	"QGemm": QGemm,
	"QLinearMatMul": QLinearMatMul,
	"QLinearAdd": QLinearAdd,
	"QLinearGlobalAveragePool": QLinearGlobalAveragePool,
	"QLinearAveragePool": QLinearAveragePool,
	"QLinearConcat": QLinearConcat,
	"QLinearLeakyRelu": QLinearLeakyRelu,
	"QLinearMul": QLinearMul,
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

# onnxscript monkeypatches the onnx module adding methods like remove and uses.
# https://github.com/onnx/onnx/issues/6404#issuecomment-2403738628
def move_constants_to_initializers(model: onnx.GraphProto):
	graph = model.graph

	new_nodes = []

	for node in graph.node:
		if node.op_type == "Constant":
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
