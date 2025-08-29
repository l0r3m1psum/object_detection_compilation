from tvm import relax
from tvm import ir
import tvm.relax.frontend.onnx

from onnx import GraphProto

from typing import Dict, List

class QuantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#inputs
		x = inputs[0]
		y_scale = inputs[1]
		y_zero_point = inputs[2]

		return relax.op.quantize(x, y_scale, y_zero_point)

class QLinearConv(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
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
			op = relax.op.nn.conv1d
			data_layout = "NCW"
			kernel_layout = "OIW"
		elif ndim == 4:
			op = relax.op.nn.conv2d
			data_layout = "NCHW"
			kernel_layout = "OIHW"
		elif ndim == 5:
			op = relax.op.nn.conv3d
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
			print(B)
			bias = relax.op.reshape(B, [1, -1] + [1] * (ndim - 2))
			conv_out = relax.op.add(relax.op.astype(conv_out, "int32"), bias)

		# FIXME: this is wrong
		return relax.op.astype(conv_out, "int8")

class DequantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#inputs
		x = inputs[0]
		x_scale = inputs[1]
		x_zero_point = inputs[2]

		return relax.op.dequantize(x, x_scale, x_zero_point)

def clamp(data, min, max):
	res = relax.op.minimum(data, relax.const(max))
	res = relax.op.maximum(relax.const(min), res)
	return res

class QLinearAdd(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearAdd
		# C = (A_s * (A - A_z) + B_s * (B - B_z))/C_s + C_z
		A   = inputs[0].astype("int32")
		A_s = inputs[1].data.numpy().item()
		A_z = inputs[2].astype("int32")
		B   = inputs[3].astype("int32")
		B_s = inputs[4].data.numpy().item()
		B_z = inputs[5].astype("int32")
		C_s = inputs[6].data.numpy().item()
		C_z = inputs[7].astype("int32")
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/micro/kernels/add_common.cc#L48C62-L48C64
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/reference/integer_ops/add.h#L211
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/reference/integer_ops/add.h#L204
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/reference/integer_ops/add.h#L180
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/common.h#L269
		# https://github.com/tensorflow/tflite-micro/blob/3b209129cc4ca0d9de64e23bd2b15def90345a7f/tensorflow/lite/kernels/internal/common.cc#L22
		shift = 20
		s1 = relax.const(int((A_s/C_s) * (1 << shift)))
		s2 = relax.const(int((B_s/C_s) * (1 << shift)))
		C = (relax.op.right_shift(s1*(A - A_z), relax.const(shift))
			+ relax.op.right_shift(s2*(B - B_z), relax.const(shift))
			+ C_z)
		C = clamp(C, -128, 127).astype("int8")
		return C

class QGemm(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
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
		# TODO: relax.op.permute_dims
		assert A.args[0].struct_info.ndim == B.struct_info.ndim
		assert B.struct_info.ndim == 2
		AT = relax.op.permute_dims(A) if transA else A
		BT = relax.op.permute_dims(B) if transB else B
		AB = relax.op.matmul(AT, BT)
		alphaAB = relax.op.multiply(AB, alpha) if alpha != 1.0 else AB
		# FIXME: this is wrong
		return relax.op.astype(relax.op.add(relax.op.astype(alphaAB, "int32"), C), "int8")

# TODO: https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html

class QLinearGlobalAveragePool(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearGlobalAveragePool
		x = inputs[0]
		x_s = inputs[1].data.numpy().item()
		x_z = inputs[2].astype("int32")
		y_s = inputs[3].data.numpy().item()
		y_z = inputs[4].astype("int32")

		shift = 20
		s = relax.const(int((x_s/y_s) * (1 << shift)))
		avg = relax.op.nn.avg_pool2d(
			data=x.astype("int32"),
			pool_size=x.struct_info.shape.values[2:],
		)
		res = relax.op.right_shift(s*(avg - x_z), relax.const(shift)) + y_z
		res = clamp(res, -128, 127).astype("int8")
		return res

convert_map = {
	"QuantizeLinear": QuantizeLinear,
	"QLinearConv": QLinearConv,
	"QGemm": QGemm,
	"QLinearAdd": QLinearAdd,
	"QLinearGlobalAveragePool": QLinearGlobalAveragePool,
	"DequantizeLinear": DequantizeLinear,
}

def from_onnx(
		model: GraphProto,
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
