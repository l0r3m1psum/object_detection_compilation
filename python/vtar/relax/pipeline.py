import tvm
from tvm import relax
from tvm import tir
from tvm import topi

import vtar.topi

import re
import copy


# relax.BlockBuilder.call_te transforms relax.Var in te.Tensor allowing easy use
# of topi operations.

def customize_legalize_conv2d(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
	data = call.args[0]
	kernel = call.args[1]
	strides = topi.utils.get_const_tuple(call.attrs.strides)
	padding = topi.utils.get_const_tuple(call.attrs.padding)
	dilation = topi.utils.get_const_tuple(call.attrs.dilation)
	layout = call.attrs.data_layout
	out_dtype = call.attrs.out_dtype
	# for conv2d
	kernel_layout = call.attrs.kernel_layout
	# For conv2d_NCHWnc
	out_layout = call.attrs.out_layout

	if layout == "NCHW":
		res = bb.call_te(
			topi.nn.conv2d,
			data, kernel, strides, padding, dilation, layout,
			kernel_layout, out_dtype)
	elif re.match("^NCHW\\d+n\\d+c$", layout):
		res = bb.call_te(
			vtar.topi.conv2d_NCHWnc,
			data, kernel, strides, padding, dilation, layout,
			out_layout, out_dtype)
	else:
		raise ValueError("Unsupported conv2d layout '%s', if if it matches "
			"NCHW\\d+c it may be trivial to add support for it." % layout)

	return res

def customize_legalize_avg_pool2d(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
	data = call.args[0]
	pool_size = topi.utils.get_const_tuple(call.attrs.pool_size)
	strides = topi.utils.get_const_tuple(call.attrs.strides)
	padding = topi.utils.get_const_tuple(call.attrs.padding)
	dilation = topi.utils.get_const_tuple(call.attrs.dilation)
	ceil_mode = call.attrs.ceil_mode
	count_include_pad = call.attrs.count_include_pad
	layout = call.attrs.layout
	out_layout = call.attrs.out_layout

	in_dtype = data.struct_info.dtype
	out_dtype = call.struct_info.dtype
	if in_dtype == out_dtype and in_dtype.startswith("int"):
		res = bb.call_te(
			vtar.topi.avg_pool2d_int,
			data,
			pool_size,
			strides,
			dilation,
			padding[-2:] if len(padding) == 4 else padding
		)
	else:
		# Taken from relax.transform.legalize_ops.nn._nn_avg_pool2d
		res = bb.call_te(
			topi.nn.pool2d,
			call.args[0],
			kernel=call.attrs.pool_size,
			stride=call.attrs.strides,
			dilation=call.attrs.dilation,
			padding=call.attrs.padding,
			pool_type="avg",
			ceil_mode=call.attrs.ceil_mode,
			layout=call.attrs.layout,
			count_include_pad=call.attrs.count_include_pad,
			primfunc_name_hint="avg_pool2d",
		)

	return res

# This is what zero_pipeline does but wiht the custom LegalizeOps
@tvm.relax.register_pipeline("vtar_zero")
def vtar_zero_pipeline():
	@tvm.transform.module_pass(opt_level=0)
	def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
		seq = tvm.transform.Sequential(
			[
				# TODO: look into register_legalize
				relax.transform.LegalizeOps(
					{"relax.nn.conv2d": customize_legalize_conv2d,}, True
				),
				relax.transform.AnnotateTIROpPattern(),
				relax.transform.FoldConstant(),
				relax.transform.FuseOps(),
				relax.transform.FuseTIR(),
			]
		)
		mod = seq(mod)
		return mod
	return _pipeline
