import tvm

from tvm import relax
from tvm import topi

from typing import List, Tuple, Dict

# TVM supports data layouts for convolutions with "factors" i.e.
# NCHW([1-9][0-9]*n)?([1-9][0-9]*w)?([1-9][0-9]*h)?([1-9][0-9]*w)?
# where the result resulting shape is {N/n}{C/c}{H/h}{W/w}nchw

# Translating from vta/python/vta/top/graphpack.py

def _get_shape(data: relax.Var) -> Tuple[int]:
	res = topi.utils.get_const_tuple(relax.get_shape_of(data))
	return res

# We can't just use relax.op.* but we have to wrap them in a bb.emit because
# this bunds the expression to a variable ans populates the struct_info
# correctly. This is necessary to be able to do relax.get_shape_of because
# "GetShapeOf can only be applied to normalized expr".

def pad_channel(bb: relax.BlockBuilder, data: relax.Var, c: int) -> relax.Expr:
	dshape = _get_shape(data)
	N, C, H, W = dshape
	pad_width = C % c;
	pad_widths = ((0, 0), (0, pad_width), (0, 0), (0, 0))
	if pad_width != 0:
		raise Exception("Bugged in TVM 0.18")
		res = bb.emit(relax.op.nn.pad(data, pad_widths))
	else:
		res = data
	return res

# https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.unpack_NCHWc_to_nchw

def unpack_NCHWnc_to_nchw(bb: relax.BlockBuilder, data: relax.Var) -> relax.Expr:
	dshape = _get_shape(data)
	Nn, Cc, H, W, n, c = dshape

	res = bb.emit(relax.op.permute_dims(data, (0, 4, 1, 5, 2, 3)))
	res = bb.emit(relax.op.reshape(res, (Nn*n, Cc*c, H, W)))

	return res

def pack_nchw_to_NCHWc(bb: relax.BlockBuilder, data: relax.Var, c: int) -> relax.Expr:
	dshape = _get_shape(data)
	N, C, H, W = dshape # O I H W for kernels
	if C % c != 0: raise ValueError("Bad channel factor")

	res = bb.emit(relax.op.reshape(data, (N, C//c, c, H, W)))
	res = bb.emit(relax.op.permute_dims(res, (0, 1, 3, 4, 2))) # N C//c H W c

	return res

def pack_nchw_to_NCHWnc(bb: relax.BlockBuilder, data: relax.Var, n: int, c: int) -> relax.Expr:
	dshape = _get_shape(data)
	N, C, H, W = dshape # O I H W for kernels

	if N % n != 0: raise ValueError("Bad batch factor")
	if C % c != 0: raise ValueError("Bad channel factor")

	res = bb.emit(relax.op.reshape(data, (N//n, n, C//c, c, H, W)))
	res = bb.emit(relax.op.permute_dims(res, (0, 2, 4, 5, 1, 3))) # N/n C/c H W n c

	return res

def is_broadcastable(shp1, shp2):
	if len(shp1) == 0 or len(shp2) == 0:
		return True
	if len(shp1) != len(shp2):
		return False
	for a, b in zip(shp1[::-1], shp2[::-1]):
		if a == 1 or b == 1 or a == b:
			pass
		else:
			return False
	return True

# TODO: per evitare di specificare bitpack_start e bitpack_end serve un'algoritmo
# che frovi i sottografi di operazioni supportate (per fare il "packing") in automatico

@relax.expr_functor.mutator
class GraphPacker(relax.expr_functor.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)
		self.bitpack_start = "relax.nn.max_pool2d"
		self.bitpack_end = "relax.nn.avg_pool2d"
		self.start_pack = False

		self.bfactor = 1 # env.BATCH
		self.cfactor = 16 # env.BLOCK_OUT
		# self.weight_bits = 8 # env.WGT_WIDTH

		self.conv_data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
		self.conv_kernel_layout = "OIHW%do%di" % (self.cfactor, self.cfactor)

	def visit_call_(self, call: relax.Call) -> relax.Expr:
		packed_args = [self.visit_expr(arg) for arg in call.args]

		# TODO: check that both start and end are present in the graph otherwise
		# the transformation can't work.

		res = None
		if call.op.name == self.bitpack_start:
			self.start_pack = True
			res = self.builder_.emit(call)
			res = pack_nchw_to_NCHWnc(self.builder_, res, self.bfactor, self.cfactor)
		elif call.op.name == self.bitpack_end:
			self.start_pack = False
			# This is too strict to allow function like reshape that has two
			# arguments one tensor and one shape.
			if len(packed_args) != 1:
				raise ValueError("The last node should have only one input.")
			res = unpack_NCHWnc_to_nchw(self.builder_, packed_args[0])
			res = self.builder_.emit(relax.Call(call.op, (res,), call.attrs))
		elif self.start_pack:
			# TODO: add way more checks like input dtypes...
			if (call.op.name == "relax.nn.conv2d"
					and call.attrs['out_dtype'] == "int32"):
				data, weight = packed_args
				weight = pad_channel(self.builder_, weight, self.cfactor)
				weight = pack_nchw_to_NCHWnc(self.builder_, weight, self.cfactor, self.cfactor)
				# TODO: topi.nn.bitpack
				res = self.builder_.emit(relax.op.nn.conv2d(
					data,
					weight,
					call.attrs['strides'],
					call.attrs['padding'],
					call.attrs['dilation'],
					call.attrs['groups'],
					self.conv_data_layout,
					self.conv_kernel_layout,
					# bb.emit infers the out_layout
					out_dtype=call.attrs['out_dtype']
				))
			elif call.op.name == 'relax.add' or call.op.name == 'relax.multiply':
				# This code should work for all elementwise binary functions.
				arg0_shape = _get_shape(packed_args[0])
				arg1_shape = _get_shape(packed_args[1])
				if arg0_shape == arg1_shape:
					pass
				elif is_broadcastable(arg0_shape, arg1_shape):
					pass
				elif not arg0_shape or not arg1_shape: # one of the two is a scalar
					pass
				else:
					# TODO: make this commutative.
					# TODO: generalize this for any 4 dimensional broadcasting
					data, bias = packed_args
					data_shape = _get_shape(data)
					bias_shape = _get_shape(bias)
					if (len(bias_shape) != 4 or
						(bias_shape[0] != 1 or bias_shape[2] != 1 or bias_shape[3] != 1)):
						raise ValueError("Broadcasted %s is only supported channel dimension" % call.op.name)
					bias = pad_channel(self.builder_, bias, self.cfactor)
					bias = pack_nchw_to_NCHWnc(self.builder_, bias, 1, self.cfactor)
					res = self.builder_.emit(relax.Call(call.op, (data, bias)))
			elif call.op.name == 'relax.reshape':
				# Data in packed_args is passed in pack_nchw_to_NCHWnc
				(data, shape) = packed_args
				shape = topi.utils.get_const_tuple(shape)
				assert len(shape) == 4 and shape[0] == 1 and shape[2] == 1 and shape[3] == 1, "only reshaping for broadcast is supported"
				data = self.builder_.emit(call)
				data = pad_channel(self.builder_, data, self.cfactor)
				data = pack_nchw_to_NCHWnc(self.builder_, data, 1, self.cfactor)
				res = data
				# # self.unpack_transpose = False
				# # N C H W n c
				# data = bb.emit(relax.op.permute_dims(data, (0, 4, 1, 5, 2, 3)))
				# # N n C c H W
				# new_shape = _get_shape(call.args[0]) # N C H W
				# res = pad_channel(self.builder_, data, new_shape[1])
			elif call.op.name == 'relax.pad':
				assert False, "pad"
			# TODO: should I check which operations we are packing and make sure
			# that they are all supported by VTA?

		if res is None:
			res = relax.Call(call.op, packed_args, call.attrs)

		return res

# TODO: vedere cosa stampa su un modello vero i.e. ResNet
# TODO: assert that the Module contains only the main function.
@tvm.transform.module_pass(opt_level=0)
class GraphPack:
	def transform_module(self, mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
		"""IRModule-level transformation"""

		# https://matt.might.net/articles/a-normalization/
		mod = relax.transform.Normalize()(mod) # should remove nested relax calls

		rewriter = GraphPacker(mod)
		for g_var, func in mod.functions_items():
			if isinstance(func, relax.Function):
				func = rewriter.visit_expr(func)
				rewriter.builder_.update_func(g_var, func)
		mod = rewriter.builder_.get()

		return mod

# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA. Note that the start_pack and
# stop_pack interval is exclusive on both ends (start_pack, stop_pack).

# Graphpack expects to receive as input a quantized model at least in the
# ``start_pack`` and ``stop_pack`` range. Also all operations in that range
# shall be supported by VTA and the subgraph should be in A-Normal Form (ANF).

# https://mlc.ai/chapter_graph_optimization/index.html#fuse-linear-and-relu
@relax.expr_functor.mutator
class UnnecessaryDequantizeQuantizeWrappingRemover(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)

	def visit_call_(self, call):

		try:
			prev = self.lookup_binding(call.args[0]) or call
			prev_prev = self.lookup_binding(prev.args[0]) or call
		# except tvm.error.TVMError as e:
		# FIXME: this catch all is terrible.
		except:
			prev = call
			prev_prev = call

		wrapper_in_dequant_quant = (
			call.op.name == 'relax.quantize'
			and prev_prev.op.name == 'relax.dequantize'
			and call.args[1].data.numpy() == prev_prev.args[1].data.numpy() # same scale
			and call.args[2].data.numpy() == prev_prev.args[2].data.numpy() # same zero_point
		)

		# FIXME: how do I check that in the pattern "dequant -> X -> duant"
		# the X node has only one outgoing edge to dequant and no other node is
		# using it downstream?

		if wrapper_in_dequant_quant:
			if prev.op.name == 'relax.reshape':
				res = relax.Call(prev.op, (prev_prev.args[0], prev.args[1]), prev.attrs)
			elif prev.op.name == 'relax.nn.max_pool2d':
				res = relax.Call(prev.op, (prev_prev.args[0],), prev.attrs)
			else:
				res = call
		else:
			res = call

		return res

# Apparently it is a common pattern to wrap non quantized operations in
# dequantize quantize blocks to fake support for them and leave the support to
# the backend/runtime.
# https://github.com/onnx/onnx/issues/5895#issuecomment-1928446285
# TODO: in the future this should support also binary operations like addition
@tvm.ir.transform.module_pass(opt_level=0)
class RemoveUnnecessaryDequantizeQuantizeWrapping:
	def transform_module(self, mod, ctx):
		rewriter = UnnecessaryDequantizeQuantizeWrappingRemover(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = rewriter.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

def to_dict(attrs: tvm.ir.Attrs) -> Dict:
	if str(attrs).startswith('relax.attrs.QuantizeAttrs'):
		return {'axis': attrs.axis, 'out_dtype': attrs.out_dtype}
	assert isinstance(attrs, tvm.ir.Attrs)
	return {key: attrs[key] for key in attrs.keys()}

@relax.expr_functor.mutator
class MaxpoolDequantizeQuantizeWrapper(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)

	def visit_call_(self, call):

		if call.op.name == 'relax.nn.max_pool2d':
			# TODO: This is a naive search for quantization values. A proper
			# algorithm should be used here.
			try:
				quantize = call
				while quantize.op.name != 'relax.quantize':
					quantize = self.lookup_binding(call.args[0])
			except Exception as e:
				raise ValueError("could not find quantization values") from e

			# breakpoint()
			res = self.builder_.emit(relax.op.dequantize(call.args[0], *quantize.args[1:]))
			res = self.builder_.emit(relax.op.nn.max_pool2d(res, **to_dict(call.attrs)))
			res = self.builder_.emit(relax.op.quantize(res, *quantize.args[1:]))
		else:
			res = call

		return res

@tvm.ir.transform.module_pass(opt_level=0)
class WrapMaxpoolDequantizeQuantize:
	def transform_module(self, mod, ctx):
		rewriter = MaxpoolDequantizeQuantizeWrapper(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = rewriter.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

@relax.expr_functor.mutator
class Matcher(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)

		wc = relax.dpl.wildcard
		is_op = relax.dpl.is_op

		self.data_scale = wc()
		self.data_zero_point = wc()
		self.weight_scale = wc()
		self.weight_zero_point = wc()
		self.bias_scale = wc()
		self.bias_zero_point = wc()
		self.output_scale = wc()
		self.output_zero_point = wc()

		# Are all commutative "is_op" operators commutative?
		# TODO: relu = is_op("relax.nn.relu")(prev) | is_op("relax.max")(prev, ...)

		# This should match all the quantized convolutions in ResNet
		qdata = is_op("relax.dequantize")(wc(), self.data_scale, self.data_zero_point)
		qweight = is_op("relax.dequantize")(wc(), self.weight_scale, self.weight_zero_point)
		qbias = is_op("relax.dequantize")(wc(), self.bias_scale, self.bias_zero_point)
		qreshape = is_op("relax.reshape")(qbias, wc()) | qbias # reshape is optional
		conv = is_op("relax.nn.conv2d")(qdata, qweight)
		add = is_op("relax.add")(conv, qreshape)
		relu = is_op("relax.nn.relu")(add) | add # ReLU is optional
		quant = is_op("relax.quantize")(relu, self.output_scale, self.output_zero_point)
		cast = is_op("relax.astype")(quant)
		self.quant_conv2d_pattern = cast

		a = is_op("relax.dequantize")(wc(), wc(), wc())
		b = is_op("relax.reshape")(a, wc()) | is_op("relax.nn.max_pool2d")(a)
		c = is_op("relax.astype")(is_op("relax.quantize")(b, wc(), wc()))
		self.quant_useless_pattern = c

		# FIXME: this also fuses only relax.astype for some reason...
		qadd = is_op("relax.add")(
			is_op("relax.dequantize")(wc(), wc(), wc()),
			is_op("relax.dequantize")(wc(), wc(), wc())
		)
		relu = is_op("relax.nn.relu")(qadd) | qadd.dup()
		add = is_op("relax.astype")(is_op("relax.quantize")(relu, wc(), wc()))
		self.quant_add_pattern = add

		qmatmul = is_op("relax.matmul")(
			is_op("relax.dequantize")(wc(), wc(), wc()),
			is_op("relax.permute_dims")(is_op("relax.dequantize")(wc(), wc(), wc())),
		)
		add = is_op("relax.add")(qmatmul, is_op("relax.dequantize")(wc(), wc(), wc())) | qmatmul
		relu = is_op("relax.nn.relu")(add) | add
		linear = is_op("relax.astype")(is_op("relax.quantize")(relu, wc(), wc()))
		# TODO: add support for relax.nn.linear
		self.quant_linear_pattern = linear

		qmean = is_op("relax.mean")(is_op("relax.dequantize")(wc(), wc(), wc()))
		avg_pool = is_op("relax.astype")(is_op("relax.quantize")(qmean, wc(), wc()))
		# TODO: add support for relax.nn.avg_pool2d
		self.quant_avg_pool_pattern = avg_pool

		self.var2val = {}
		self.conv_cnt = 0
		self.match_cnt = 0

	def visit_function_(self, func: relax.Function) -> relax.Expr:
		self.var2val = relax.analysis.get_var2val(func)
		return super().visit_function_(func)

	def visit_call_(self, call: relax.Call) -> relax.Expr:
		if call.op.name == "relax.nn.conv2d":
			self.conv_cnt += 1
		matched_expr = self.quant_conv2d_pattern.extract_matched_expr(call, self.var2val)
		if matched_expr:
			self.match_cnt += 1
			data_scale = matched_expr[self.data_scale]
			data_zero_point = matched_expr[self.data_zero_point]
			weight_scale = matched_expr[self.weight_scale]
			weight_zero_point = matched_expr[self.weight_zero_point]
			bias_scale = matched_expr[self.bias_scale]
			bias_zero_point = matched_expr[self.bias_zero_point]
			output_scale = matched_expr[self.output_scale]
			output_zero_point = matched_expr[self.output_zero_point]

		return super().visit_call_(call)
