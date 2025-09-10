from tvm import relax
from tvm import ir
from tvm import topi
import tvm.relax.frontend.onnx

from onnx import GraphProto

from typing import Dict, List

def clamp(data, min, max):
	res = relax.op.minimum(data, relax.const(max))
	res = relax.op.maximum(relax.const(min), res)
	return res

def get_data(
		expr: relax.Expr,
		params: Dict[str, relax.Expr]
	) -> float|int:
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
def get_arg(
		expr: relax.Expr,
		params: Dict[str, relax.Expr]
	) -> relax.Expr:
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

class QuantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#inputs
		x = inputs[0]
		y_scale = get_arg(inputs[1], params[1])
		y_zero_point = get_arg(inputs[2], params[1])

		return relax.op.quantize(x, y_scale, y_zero_point)

class QLinearConv(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		input0_struct_info = inputs[0].args[0].struct_info
		# https://onnx.ai/onnx/operators/onnx__QLinearConv.html#inputs
		X   = inputs[0]
		X_s = get_data(inputs[1], params[1])
		X_z = get_arg(inputs[2], params[1]).astype("int32")
		W   = get_arg(inputs[3], params[1])
		W_s = get_data(inputs[4], params[1])
		W_z = get_arg(inputs[5], params[1]).astype("int32")
		Y_s = get_data(inputs[6], params[1])
		Y_z = get_arg(inputs[7], params[1]).astype("int32")
		B   = get_arg(inputs[8], params[1])
		assert len(X.struct_info.shape) == 4
		assert len(W.struct_info.shape) == 4
		assert get_data(inputs[5], params[1]) == 0

		conv = relax.op.nn.conv2d(
			data=X,
			weight=W,
			strides=attr.get("strides", 1),
			padding=attr.get("pads", 0),
			dilation=attr.get("dilations", 1),
			groups=attr.get("group", 1),
			data_layout="NCHW",
			kernel_layout="OIHW",
			out_dtype="int32"
		)

		shift = 20
		_, c, h, w = topi.utils.get_const_tuple(relax.get_shape_of(W))
		m = relax.const(int(X_s*W_s/Y_s * (1<<shift)))
		s3 = relax.const(int(Y_s * (1<<shift)))
		res = relax.const(
			int(c*h*w
				*get_data(inputs[2], params[1])
				*get_data(inputs[5], params[1])/Y_s)) \
		+ relax.op.right_shift(m*(
			conv
			# FIXME: the reshape should be correct wrt the batch dimension wich is not necessarelly 1.
			+ (-X_z)*relax.op.reshape(bb.normalize(relax.op.sum(W.astype("int32"), axis=(1,2,3))), (1, -1, 1, 1))
			# TODO: right now we are asserting that W_z is zero, in the future we have to remove this assumption.
			# + (-W_z)*relax.op.sum(W.astype("int32")) # TODO: sum the last three dimensions
		), relax.const(shift)) \
		+ relax.op.reshape(bb.normalize(relax.op.left_shift(B/s3, relax.const(shift))), (1,-1,1,1)) + Y_z
		# FIXME: the VTA accellerator does not support left shift...

		res = clamp(res, -128, 127).astype("int8")
		return res

class DequantizeLinear(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v10(cls, bb, inputs, attr, params):
		# https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#inputs
		x = inputs[0]
		x_scale = get_arg(inputs[1], params[1])
		x_zero_point = get_arg(inputs[2], params[1])

		return relax.op.dequantize(x, x_scale, x_zero_point)

class QLinearAdd(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearAdd
		# C = (A_s * (A - A_z) + B_s * (B - B_z))/C_s + C_z
		# We cast to "int32" because is what VTA internally does and we hope
		# that gets mapped to TVM later in the compilation.
		A   = inputs[0].astype("int32")
		A_s = get_data(inputs[1], params[1])
		A_z = get_arg(inputs[2], params[1]).astype("int32")
		B   = inputs[3].astype("int32")
		B_s = get_data(inputs[4], params[1])
		B_z = get_arg(inputs[5], params[1]).astype("int32")
		C_s = get_data(inputs[6], params[1])
		C_z = get_arg(inputs[7], params[1]).astype("int32")
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

		assert inputs[0].args[0].struct_info.ndim == inputs[3].struct_info.ndim
		assert inputs[3].struct_info.ndim == 2
		assert alpha == 1.0, "alpha != 1.0 requires some work to keep it integer only"

		# A has shape MxN and B has shape NxO
		n = relax.const(topi.utils.get_const_tuple(relax.get_shape_of(inputs[0]))[1])
		A   = inputs[0]
		A_s = get_data(inputs[1], params[1])
		A_z = get_arg(inputs[2], params[1]).astype("int32")
		B   = get_arg(inputs[3], params[1])
		B_s = get_data(inputs[4], params[1])
		B_z = get_arg(inputs[5], params[1]).astype("int32")
		C   = get_arg(inputs[6], params[1]).astype("int32")
		Y_s = get_data(inputs[7], params[1])
		Y_z = get_arg(inputs[8], params[1]).astype("int32")
		AT = relax.op.permute_dims(A) if transA else A
		BT = relax.op.permute_dims(B) if transB else B
		AB = relax.op.matmul(AT, BT, out_dtype="int32")

		# Reductions should happen in on the CPU while VTA is doing the matmul
		a2 = relax.op.sum(BT.astype("int32"), axis=0) # reduce over columns
		a1 = relax.op.sum(AT.astype("int32"), axis=1) # reduce over rows

		shift = 20
		m = relax.const(int((A_s*B_s)/Y_s * (1<<shift)))
		# - A_z*a2 - B_z*a1 is broadcasted
		res = (Y_z
			+ relax.op.right_shift(m*(n*A_z*B_z - A_z*a2 - B_z*a1 + AB), relax.const(shift))
			# TOOD: check that this is right (problably not)
			+ relax.op.left_shift(C/relax.const(int(Y_s * (1 << shift))), relax.const(shift))
		)
		res = clamp(res, -128, 127).astype("int8")
		return res

# TODO: https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html

class QLinearGlobalAveragePool(relax.frontend.onnx.onnx_frontend.OnnxOpConverter):
	@classmethod
	def _impl_v1(cls, bb, inputs, attr, params):
		# https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearGlobalAveragePool
		x = inputs[0]
		x_s = get_data(inputs[1], params[1])
		x_z = get_arg(inputs[2], params[1]).astype("int32")
		y_s = get_data(inputs[3], params[1])
		y_z = get_arg(inputs[4], params[1]).astype("int32")

		shift = 20
		s = relax.const(int((x_s/y_s) * (1 << shift)))
		avg = relax.op.nn.avg_pool2d(
			data=x.astype("int32"),
			pool_size=x.struct_info.shape.values[2:],
		).astype("int32") # This is a workaround because the vtar_zero pipeline is convinced that this is an int64
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
