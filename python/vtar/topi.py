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
	layout: str,
	out_layout: str,
	out_dtype='int32'
) -> te.Tensor:
	# TODO: error checking
	# TODO: add support for single int or list for strides ecc like the other operators topi

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
		name="res",
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
	res = te.compute(
		topi.utils.get_const_tuple(x.shape),
		lambda *i: tir.Select(a[*i] >= 0, x[*i] >> a[*i], x[*i] << -a[*i]),
		"res",
	)
	return res
