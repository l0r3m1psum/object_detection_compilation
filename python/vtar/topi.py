from tvm import topi
from tvm import te

from typing import Tuple

# copied from vta.top.vta_conv2d
# https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.conv2d_NCHWc
def conv2d_NCHWnc(
		data: te.Tensor,
		kernel: te.Tensor,
		strides: Tuple[int],
		padding: Tuple[int],
		dilation: Tuple[int],
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

	if padding[0]:
		pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0])
	else:
		pad_data = data

	oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
	owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
	oshape = (dshape[0], kshape[0], oheight, owidth, dshape[4], kshape[4])

	d_i = te.reduce_axis((0, kshape[2]), name="d_i")
	d_j = te.reduce_axis((0, kshape[3]), name="d_j")
	k_o = te.reduce_axis((0, dshape[1]), name="k_o")
	k_i = te.reduce_axis((0, dshape[-1]), name="k_i")
	hstride, wstride = strides
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
