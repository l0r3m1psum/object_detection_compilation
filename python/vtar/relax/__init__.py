import tvm
from tvm import relax
from tvm import tir
from tvm import topi

import vtar

import os, sys
sys.path.append(os.path.join(os.getcwd(), "submodules\\tvm\\vta\\python"))

import vta

# relax.BlockBuilder.call_te transforms relax.Var in te.Tensor allowing easy use
# of topi operations.

def customize_legalize_conv2d(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
	data = call.args[0]
	kernel = call.args[1]
	strides = topi.utils.get_const_tuple(call.attrs.strides)
	padding = topi.utils.get_const_tuple(call.attrs.padding)
	dilation = topi.utils.get_const_tuple(call.attrs.dilation)
	layout = call.attrs.data_layout
	out_layout = call.attrs.out_layout
	out_dtype = vta.get_env().acc_dtype

	# TODO: dispatch to other conv2d in the topi and error out for unsupported
	# formats.

	return bb.call_te(vtar.topi.conv2d_NCHWnc, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype)

# This is what zero_pipeline does but wiht the custom LegalizeOps and MakePackedAPI
@tvm.relax.register_pipeline("vtar_zero")
def vtar_zero_pipeline():
	@tvm.transform.module_pass(opt_level=0)
	def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
		seq = tvm.transform.Sequential(
			[
				relax.transform.LegalizeOps(
					{"relax.nn.conv2d": customize_legalize_conv2d,}, True
				),
				relax.transform.AnnotateTIROpPattern(),
				relax.transform.FoldConstant(),
				relax.transform.FuseOps(),
				relax.transform.FuseTIR(),
				tir.transform.MakePackedAPI(),
			]
		)
		mod = seq(mod)
		return mod
	return _pipeline
