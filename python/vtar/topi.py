from tvm import topi
from tvm import te
from tvm import tir

from typing import Tuple

# copied from vta.top.vta_conv2d
# https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.conv2d_NCHWc
def conv2d_NCHWnc(
	data: te.Tensor,
	kernel: te.Tensor,
	strides: Tuple[int, int],
	padding: Tuple[int, int],
	dilation: Tuple[int, int],
	out_dtype='int32'
) -> te.Tensor:
	# TODO: error checking
	# TODO: add support for single int or list for strides etc like the other operators topi

	# if not is_packed_layout(layout):
	# 	raise topi.InvalidShapeError()
	assert dilation == (1, 1)
	assert len(data.shape) == 6
	assert len(kernel.shape) == 6

	dshape = topi.utils.get_const_tuple(data.shape)
	kshape = topi.utils.get_const_tuple(kernel.shape)

	if any(padding):
		pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0])
	else:
		pad_data = data

	hstride, wstride = strides
	oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // hstride + 1)
	owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // wstride + 1)
	oshape = (dshape[0], kshape[0], oheight, owidth, dshape[4], kshape[4])

	d_i = te.reduce_axis((0, kshape[2]), name="d_i")
	d_j = te.reduce_axis((0, kshape[3]), name="d_j")
	k_o = te.reduce_axis((0, dshape[1]), name="k_o")
	k_i = te.reduce_axis((0, dshape[-1]), name="k_i")
	res = te.compute(
		oshape,
		lambda b_o, c_o, i, j, b_i, c_i: te.sum(
			pad_data[b_o, k_o, i * hstride + d_i, j * wstride + d_j, b_i, k_i].astype(out_dtype)
			* kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
			axis=[k_o, d_i, d_j, k_i],
		),
		name="conv2d_NCHWnc",
		tag="conv2d_dense",
	)

	return res

def avg_pool2d_int(
	data: te.Tensor,
	pool_size: Tuple[int, int],
	strides: Tuple[int, int],
	dilation: Tuple[int, int],
	padding: Tuple[int, int],
	count_include_pad: bool = False,
	layout: str = "NCHW", # TODO: handle factored layouts
	out_layout: str = "NCHW",
) -> te.Tensor:

	# TODO: it should be trivial to add support for dilation
	assert dilation == (1, 1)
	assert layout == "NCHW"
	assert out_layout == "NCHW"
	assert not count_include_pad

	pool_h, pool_w = pool_size
	N = te.const(pool_h * pool_w, data.dtype)

	ry = te.reduce_axis((0, pool_h), name="ry")
	rx = te.reduce_axis((0, pool_w), name="rx")

	def reducer_intr(lhs: Tuple[tir.Var], rhs: Tuple[tir.Var]) -> Tuple[tir.PrimExpr, tir.PrimExpr]:
		acc_quot, acc_rem = lhs
		new_quot, new_rem = rhs
		new_acc_quot = acc_quot + new_quot
		new_acc_rem = acc_rem + new_rem
		# NOTE: N must be less than INT32_MIN because of negation in two's complement
		quot_correction_pos = tir.Select(new_acc_rem >= N, 1, 0)
		quot_correction_neg = tir.Select(new_acc_rem <= -N, -1, 0)
		quot_correction = quot_correction_pos + quot_correction_neg
		rem_correction_pos = tir.Select(new_acc_rem >= N, -N, 0)
		rem_correction_neg = tir.Select(new_acc_rem <= -N, N, 0)
		rem_correction = rem_correction_pos + rem_correction_neg
		return (new_acc_quot + quot_correction, new_acc_rem + rem_correction)

	def reducer_identity(dtype1: str, dtype2: str) -> Tuple[tir.PrimExpr, tir.PrimExpr]:
		return (te.const(0, dtype1), te.const(0, dtype2))

	dist_avg_reducer = te.comm_reducer(reducer_intr, reducer_identity, name="dist_avg")

	if any(padding):
		pad_data = topi.nn.pad(data, (0, 0, padding[0], padding[1]))
	else:
		pad_data = data

	hstride, wstride = strides
	dshape = topi.utils.get_const_tuple(data.shape)
	pad_data_shape = topi.utils.get_const_tuple(pad_data.shape)
	oheight = (pad_data_shape[2] - pool_h) // hstride + 1
	owidth = (pad_data_shape[3] - pool_w) // wstride + 1
	oshape = (dshape[0], dshape[1], oheight, owidth)

	sum_quot, sum_rem = te.compute(
		oshape,
		lambda n, c, h, w: dist_avg_reducer(
			(
				te.truncdiv(pad_data[n, c, h*hstride + ry, w*wstride + rx], N),
				te.truncmod(pad_data[n, c, h*hstride + ry, w*wstride + rx], N)
			),
			axis=[ry, rx]
		),
		name="pool_distributive_accum"
	)
	res = te.compute(
		oshape,
		lambda n, c, h, w: sum_quot[n, c, h*hstride, w*wstride]    \
			+ te.truncdiv(sum_rem[n, c, h*hstride, w*wstride], N),
		name="res"
	)

	return res

def bidi_shift(x: te.Tensor, a: te.Tensor) -> te.Tensor:
	shape = topi.utils.get_const_tuple(x.shape)
	# FIXME: the hack I have done here makes the test test_shift_bidirectional fail
	if len(shape) == 4:
		res = te.compute(
			topi.utils.get_const_tuple(x.shape),
			lambda n, c, h, w: tir.Select(
				a[0, c, 0, 0] >= 0,
				x[n, c, h, w] >> a[0, c, 0, 0],
				x[n, c, h, w] << -a[0, c, 0, 0]
			),
			"res",
		)
	else:
		res = te.compute(
			topi.utils.get_const_tuple(x.shape),
			lambda no, co, h, w, ni, ci: tir.Select(
				a[0, co, 0, 0, 0, ci] >= 0,
				x[no, co, h, w, ni, ci] >> a[0, co, 0, 0, 0, ci],
				x[no, co, h, w, ni, ci] << -a[0, co, 0, 0, 0, ci]
			),
			"res",
		)
	return res

def sq_ioa_conv2d_NCHWnc(
	data: te.Tensor,
	kernel: te.Tensor,
	bias: te.Tensor,
	scale: int | te.Tensor,
	strides: Tuple[int, int],
	padding: Tuple[int, int],
	dilation: Tuple[int, int],
	acc_dtype: str = 'int32', # accumulator
	out_dtype: str = 'int8',
):
	"""Symmetrically quantized, integer-only-arithmetic 2D convolution in packed
	format."""

	res = conv2d_NCHWnc(
		data=data,
		kernel=kernel,
		padding=padding,
		strides=strides,
		dilation=dilation,
		out_dtype=acc_dtype,
	)
	res = topi.add(res, bias)
	if isinstance(scale, te.Tensor):
		res = bidi_shift(res, scale)
	else:
		res = topi.right_shift(res, scale) if scale >= 0 else topi.left_shift(res, -scale)
	res = topi.minimum(res, te.max_value(out_dtype))
	res = topi.maximum(te.min_value(out_dtype), res)
	res = topi.cast(res, out_dtype)
	return res

resnet18_workloads = (
	#                   inp     out    H    W    I    O  R  S pt pl pb pb dh dw sh sw
	topi.nn.Workload("int8", "int8", 224, 224,   3,  64, 7, 7, 3, 3, 3, 3, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",  56,  56,  64,  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
	topi.nn.Workload("int8", "int8",  56,  56,  64, 128, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",  56,  56,  64, 128, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",  28,  28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
	topi.nn.Workload("int8", "int8",  28,  28, 128, 256, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",  28,  28, 128, 256, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",  14,  14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
	topi.nn.Workload("int8", "int8",  14,  14, 256, 512, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",  14,  14, 256, 512, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2),
	topi.nn.Workload("int8", "int8",   7,   7, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
)

def sq_ioa_conv2d_NCHWnc_from_workload(
	wl: topi.nn.Workload, BATCH: int, BLOCK_IN: int, BLOCK_OUT: int, imm_scale=None
) -> tir.PrimFunc:
	if BATCH != 1:
		raise ValueError("ONLY single batch is supported for now.")
	N = 1
	H, W = wl.height, wl.width
	CI, CO = wl.in_filter, wl.out_filter
	KH, KW = wl.kernel_h, wl.kernel_w
	strides = (wl.stride_h, wl.stride_w)
	if wl.padt != wl.padb or wl.padl != wl.padr:
		raise ValueError("Padding top should should be equal to padding bottom"
			" and padding left should be equal to padding right.")
	padding = (wl.padt, wl.padl)
	dilation = (wl.dilation_h, wl.dilation_w)

	data_shape   = (N // BATCH, CI // BLOCK_IN, H, W, BATCH, BLOCK_IN)
	kernel_shape = (CO // BLOCK_OUT, CI // BLOCK_IN, KH, KW, BLOCK_OUT, BLOCK_IN)
	bias_shape   = (1, CO // BLOCK_OUT, 1, 1, 1, BLOCK_OUT)

	if wl.in_dtype != "int8" or wl.out_dtype != "int8":
		raise ValueError("Only int8 input and output types supported.")

	acc_dtype = "int32"

	data   = te.placeholder(data_shape, wl.in_dtype, "data")
	kernel = te.placeholder(kernel_shape, wl.in_dtype, "kernel")
	bias   = te.placeholder(bias_shape, acc_dtype, "bias")
	scale  = te.placeholder(bias_shape, acc_dtype, "scale") if not imm_scale else imm_scale

	res = sq_ioa_conv2d_NCHWnc(
	    data=data,
	    kernel=kernel,
	    bias=bias,
	    scale=scale,
	    strides=strides,
	    padding=padding,
	    dilation=dilation,
	    acc_dtype=acc_dtype,
	    out_dtype=wl.out_dtype,
	)

	func = te.create_prim_func((data, kernel, bias, scale,  res))

	return func
