import tvm

from tvm import relax, te, tir

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import numpy

from typing import List, Tuple

@I.ir_module
class TVMScriptModule:
	@R.function
	def main(
		x: R.Tensor((1, 784), dtype='float32'),
		fc1_weight: R.Tensor((256, 784), dtype='float32'),
		fc1_bias: R.Tensor((256,), dtype='float32'),
		fc2_weight: R.Tensor((10, 256), dtype='float32'),
		fc2_bias: R.Tensor((10,), dtype='float32'),
	) -> R.Tensor((1, 10), dtype='float32'):
		R.func_attr({"num_input": 1})
		with R.dataflow():
			permute_dims = R.permute_dims(fc1_weight) # transpose
			matmul = R.matmul(x, permute_dims)
			add = R.add(matmul, fc1_bias)
			relu = R.nn.relu(add)
			permute_dims1 = R.permute_dims(fc2_weight) # transpose
			matmul1 = R.matmul(relu, permute_dims1)
			add1 = R.add(matmul1, fc2_bias)
			gv = add1
			R.output(gv)
		return gv

in_chans = 3,
out_chans = 2
kernel_h, kernel_w = 5, 5
# FIXME: the network has to be quantized first.
@I.ir_module
class ConvModel:
	@R.function
	def main(
		x: R.Tensor((1, 3, 32, 32), dtype="float32"),
		conv1_weight: R.Tensor((2, 3, 5, 5), dtype="float32"),
		conv1_bias: R.Tensor((2,), dtype="float32"),
	) -> R.Tensor((1, 2, 28, 28)):
		R.func_attr({"num_input": 1})
		with R.dataflow():
			conv1 = R.nn.conv2d(x, conv1_weight, strides=1, padding=0, dilation=1,
				data_layout='NCHW', kernel_layout='OIHW')
			add1 = R.add(conv1, R.reshape(conv1_bias, (1, -1, 1, 1)))
			gv = add1
			R.output(gv)
		return gv

"""
inp_dtype: int8
wgt_dtype: int8
out_dtype: int8
acc_dtype: int32
BATCH: 1
BLOCK_IN: 16
BLOCK_OUT: 16
"""

# To interpret look at test_benchmark_topi_conv2d.py:run_conv2d
# H=56, W=56, I=64, O=64, kH=3, kW=3
@I.ir_module
class ConvModelVTA:
	@R.function
	def main(
		#             (1//BATCH,      64//BLOCK_IN, 56, 56, BATCH,     BLOCK_IN)
		x:            R.Tensor((1//1,   64//16, 56, 56, 1,  16), dtype="int8"),
		#             (64//BLOCK_OUT, 64//BLOCK_IN, 3,  3,  BLOCK_OUT, BLOCK_IN)
		conv1_weight: R.Tensor((64//16, 64//16, 3,  3,  16, 16), dtype="int8"),
		#             (1//BATCH,      64//BLOCK_IN, 1,  1,  BATCH,     BLOCK_OUT)
		conv1_bias:   R.Tensor((1//1,   64//16, 1,  1,  1,  16), dtype="int32"),
	):
		R.func_attr({"num_input": 1})
		with R.dataflow():
			conv1 = R.nn.conv2d(x, conv1_weight, strides=1, padding=1, dilation=1,
				data_layout="NCHW1n16c", kernel_layout="OIHW16o16i")
			astype1 = R.astype(conv1, "int32")
			add1 = R.add(astype1, conv1_bias)
			gv = add1
			R.output(gv)
		return gv

# TVM supports data layouts for convolutions with "factors" i.e.
# NCHW([1-9][0-9]*n)?([1-9][0-9]*w)?([1-9][0-9]*h)?([1-9][0-9]*w)?
# where the result resulting shape is {N/n}{C/c}{H/h}{W/w}nchw

# Translating from vta/python/vta/top/graphpack.py

def _get_shape(data: tvm.relax.Var) -> List[int]:
	res = [int(i) for i in data.struct_info.shape]
	return res

def _pack_batch_channel(data: tvm.relax.Var, bfactor: int, cfactor: int) -> tvm.relax.Expr:
	"""Pack the data batch and channel dimension."""
	dshape = _get_shape(data)
	if dshape[0] % bfactor != 0: raise ValueError("bad batch factor")
	if dshape[1] % cfactor != 0: raise ValueError("bad channel factor")
	data = tvm.relax.op.reshape(
		data,
		(
			dshape[0] // bfactor, # N/n | 0
			bfactor,              # n   | 1
			dshape[1] // cfactor, # C/c | 2
			cfactor,              # c   | 3
			dshape[2],            # H   | 4
			dshape[3],            # W   | 5
		),
	)
	data = tvm.relax.op.permute_dims(data,
		axes=(0, 2, 4, 5, 1, 3) # N/n C/c H W n c
	)
	return data

# NOTE: I do not understand this function..
def _weight_shape_match(data: tvm.relax.Var, cfactor_out: int) -> tvm.relax.Expr:
	"""Pad the weight if the shape[0] not divisible by cfactor_out."""
	O, I, H, W = _get_shape(data)

	pad_width = O % cfactor_out
	if pad_width != 0:
		data = tvm.relax.op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0], [0, 0]])
	return data

def _pack_weight(data: tvm.relax.Var, cfactor: int):
	"""Pack the weight into packed format."""
	dshape = _get_shape(data)
	if dshape[0] % cfactor != 0: raise ValueError("bad channel factor for dshape[0]")
	if dshape[1] % cfactor != 0: raise ValueError("bad channel factor for dshape[1]")
	data = op.reshape(
		data,
		newshape=(
			dshape[0] // cfactor, # O/o | 0
			cfactor,              # o   | 1
			dshape[1] // cfactor, # I/i | 2
			cfactor,              # i   | 3
			dshape[2],            # H   | 4
			dshape[3],            # W   | 5
		),
	)
	data = op.permute_dims(data,
		axes=(0, 2, 4, 5, 1, 3) # O/o I/i H W o i
	)
	return data

def _pack_const(data: tvm.relax.Var, bfactor: int, cfactor: int):
	"""Pack a constant parameter."""
	dshape = _get_shape(dshape)
	if len(dshape) != 3: raise ValueError("")
	if dshape[0] % cfactor != 0: raise ValueError("")
	data = tvm.relax.op.reshape(data, newshape=(dshape[0] // cfactor, cfactor, dshape[1], dshape[2], 1))
	data = tvm.relax.op.permute_dims(data, (0, 2, 3, 4, 1))

	# broadcast batch dimension to bfactor
	data = tvm.relax.op.broadcast_to(
		data, shape=(dshape[0] // cfactor, dshape[1], dshape[2], bfactor, cfactor)
	)
	return data

# TODO: per evitare di specificare bitpack_start e bitpack_end serve un'algoritmo
# che frovi i sottografi di operazioni supportate (per fare il "packing") in automatico

# Per il momento scrivere un "packer" che non considera inizio e fine ma che
# "esplode" se trova operazioni non supportate.
@tvm.relax.expr_functor.mutator
class ReluAndMatmulRewriter(tvm.relax.expr_functor.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)
		self.bitpack_start = "relax.nn.conv2d"
		self.bitpack_end = "relax.add"
		self.start_pack = False
		self.bfactor = 1
		self.cfactor = 1

		self.conv_data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
		self.conv_kernel_layout = "OIHW%do%di" % (self.cfactor, self.cfactor)

	def visit_call_(self, call: tvm.relax.Call) -> tvm.relax.Expr:
		# NOTE: pretty sure that this is useless...
		args = [self.visit_expr(arg) for arg in call.args]

		if call.op.name == self.bitpack_start:
			self.start_pack = True
			return _pack_batch_channel(args[0], self.bfactor, self.cfactor)

		if call.op.name == self.bitpack_end:
			self.start_pack = False
			old_shape = _get_shape(call.args[0])
			breakpoint()
			return tvm.relax.op.reshape(args[0], old_shape)

		if self.start_pack:
			if (call.op.name == "relax.nn.conv2d"
					and call.attrs['out_dtype'] == "int32"):
				data, weight = args
				data_shape = _get_shape(data)
				weight_shape = _get_shape(weight)
				weight = _weight_shape_match(weight, self.cfactor)
				weight = _pack_weight(weight, self.cfactor)
				# TODO: tvm.topi.nn.bitpack
				res = tvm.relax.op.nn.conv2d(
					data,
					weight,
					call.attrs['strides'],
					call.attrs['padding'],
					call.attrs['dilation'],
					call.attrs['groups'],
					self.conv_data_layout,
					self.conv_kernel_layout,
					call.attrs['out_layout'],
					call.attrs['out_dtype'],
				)
				return res

			if (call.op.name == "relax.nn.conv2d_transpose"
					and call.attrs['out_dtype'] == "int32"):
				raise Exception("not supported")

			if call.op.name == "relax.add":
				if _get_shape(call.args[0]) == _get_shape(call.args[1]):
					return super().visit_call_(call)

				# NOTE: why 3?
				if len(_get_shape(call.args[1])) == 3:
					data, const = args
					const, input_shape = _const_shape_match(const, input_types[1].shape, self.cfactor)
					const = _pack_const(const, self.bfactor, self.cfactor)
					return tvm.relax.op.add(data, const)

		return super().visit_call_(call)

# A TVM pass
@tvm.transform.module_pass(opt_level=0, name="ReluToGeluAndQuantizeMatmul")
class ReluToGeluAndQuantizeMatmul:
	def transform_module(self, mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
		"""IRModule-level transformation"""
		rewriter = ReluAndMatmulRewriter(mod)
		for g_var, func in mod.functions_items():
			if isinstance(func, tvm.relax.Function):
				func = rewriter.visit_expr(func)
				rewriter.builder_.update_func(g_var, func)
		return rewriter.builder_.get()

zero_pipeline = tvm.relax.get_pipeline('zero')

from tvm import topi
import os, sys
sys.path.append(os.path.join(os.getcwd(), "submodules\\tvm\\vta\\python"))
import vta.testing

os.environ["TVM_WIN_CC"] = "clang_wrapper.bat"

def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
	print("af")
	return bb.call_te(topi.add, call.args[0], call.args[1])

# copied from vta.top.vta_conv2d
# https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.conv2d_NCHWc
def topi_conv2d_NCHWnc(
		data: te.Tensor,
		kernel: te.Tensor,
		strides: Tuple[int],
		padding: Tuple[int],
		dilation: Tuple[int],
		layout: str,
		out_layout: str,
		out_dtype='int32'
	) -> te.Tensor:
	# if not is_packed_layout(layout):
	# 	raise topi.InvalidShapeError()
	assert dilation == (1, 1)
	assert len(data.shape) == 6
	assert len(kernel.shape) == 6

	ishape = topi.utils.get_const_tuple(data.shape)
	kshape = topi.utils.get_const_tuple(kernel.shape)

	if padding[0]:
		pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0])
	else:
		pad_data = data

	oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
	owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
	oshape = (topi.utils.get_const_int(data.shape[0]), topi.utils.get_const_int(kernel.shape[0]), oheight, owidth, topi.utils.get_const_int(data.shape[4]), topi.utils.get_const_int(kernel.shape[4]))

	d_i = te.reduce_axis((0, kshape[2]), name="d_i")
	d_j = te.reduce_axis((0, kshape[3]), name="d_j")
	k_o = te.reduce_axis((0, ishape[1]), name="k_o")
	k_i = te.reduce_axis((0, ishape[-1]), name="k_i")
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

def customize_legalize_conv2d(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
	data = call.args[0]
	kernel = call.args[1]
	strides = topi.utils.get_const_tuple(call.attrs.strides)
	padding = topi.utils.get_const_tuple(call.attrs.padding)
	dilation = topi.utils.get_const_tuple(call.attrs.dilation)
	layout = call.attrs.data_layout
	out_layout = call.attrs.out_layout
	out_dtype = vta.get_env().acc_dtype

	return bb.call_te(topi_conv2d_NCHWnc, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype)


mod = ConvModelVTA
# mod, _ = tvm.relax.frontend.detach_params(mod)
# mod = tvm.lower(mod)
mod = relax.transform.LegalizeOps(
	{
		"relax.add": customize_legalize_add,
		"relax.nn.conv2d": customize_legalize_conv2d,
	},
	True
)(mod)
mod = zero_pipeline(mod)
seq = tvm.transform.Sequential(
	[
				relax.transform.RewriteDataflowReshape(),
				relax.transform.ToNonDataflow(),
				relax.transform.RemovePurityChecking(),
				relax.transform.CallTIRRewrite(),
				tir.transform.MakePackedAPI(),
				# relax.transform.StaticPlanBlockMemory(),
				# relax.transform.LowerAllocTensor(),
				# relax.transform.KillAfterLastUse(),
				# relax.transform.LowerRuntimeBuiltin(),
				# relax.transform.VMShapeLower(),
				# relax.transform.AttachGlobalSymbol(),
	]
)
mod = seq(mod)

print(mod)
# print(mod['fused_cast_add'].buffer_map) # this needs to be dropped

# FIXME: This pass must be called after MakePackedAPI
vta.build(mod)

raise SystemExit(0)

def to_int_list(x: tvm.ir.Array) -> List[int]:
	return [int(n) for n in x]

def make_closure_test_hardcoded_relax(mod):
	def test_hardcoded_relax(env: vta.Environment, remote: tvm.rpc.RPCSession) -> None:
		nonlocal mod
		x            = te.placeholder((1, 4, 56, 56, 1, 16), name="x",            dtype="int8")
		conv1_weight = te.placeholder((4, 4, 3, 3, 16, 16),  name="conv1_weight", dtype="int8")
		conv1_bias   = te.placeholder((1, 4, 1, 1, 1, 16),   name="conv1_bias",   dtype="int32")
		res          = te.placeholder((1, 4, 56, 56, 1, 16), name="res",          dtype="int32")
		mod = vta.build(
			mod['fused_cast_add'],
			(conv1_weight, conv1_bias, res),
			target=tvm.target.Target(env.target, host=env.target_host),
			name="conv2d"
		)
		mod.save("build/conv2d.o")
		remote.upload("build/conv2d.o")
		f = remote.load_module("conv2d.o")
		dev = remote.device(str(env.target))
		time_f = f.time_evaluator("conv2d", dev, number=1)

		data_np   = numpy.random.randint(0, 10, size=to_int_list(x.shape)).astype(x.dtype)
		kernel_np = numpy.random.randint(0, 10, size=to_int_list(conv1_weight.shape)).astype(conv1_weight.dtype)
		bias_np   = numpy.random.randint(0, 10, size=to_int_list(conv1_bias.shape)).astype(conv1_bias.dtype)
		res_np    = numpy.zeros(to_int_list(res.shape)).astype(res.dtype)

		data_arr = tvm.nd.array(data_np, dev)
		kernel_arr = tvm.nd.array(kernel_np, dev)
		bias_arr = tvm.nd.array(bias_np, dev)
		res_arr = tvm.nd.array(res_np, dev)

		cost = time_f(kernel_arr, bias_arr, res_arr)
		print(cost)
	return test_hardcoded_relax

vta.testing.run(make_closure_test_hardcoded_relax(mod))
