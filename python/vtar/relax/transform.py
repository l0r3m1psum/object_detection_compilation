import tvm

from tvm import ir
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
		self.bitpack_end = "relax.mean" # "relax.nn.avg_pool2d"
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
				# NOTE: I don't know why this if before was not necessary and now it is...
				if len(data.struct_info.shape) != 6:
					data = pack_nchw_to_NCHWnc(self.builder_, data, self.bfactor, self.cfactor)
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
					# NOTE: I don't know why this if before was not necessary and now it is...
					if len(bias_shape) != 6:
						if (len(bias_shape) != 4 or
							(bias_shape[0] != 1 or bias_shape[2] != 1 or bias_shape[3] != 1)):
							breakpoint()
							raise ValueError("Broadcasted %s is only supported channel dimension" % call.op.name)
						bias = pad_channel(self.builder_, bias, self.cfactor)
						bias = pack_nchw_to_NCHWnc(self.builder_, bias, 1, self.cfactor)
					# NOTE: I don't know why this if before was not necessary and now it is...
					if len(data_shape) != 6:
						data = pack_nchw_to_NCHWnc(self.builder_, data, 1, self.cfactor)
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

################################################################################

# https://mlc.ai/chapter_graph_optimization/index.html#fuse-linear-and-relu
@relax.expr_functor.mutator
class UnnecessaryDequantizeQuantizeWrappingRemover(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)
		# TODO: make this more general

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

		res = call
		if wrapper_in_dequant_quant:
			if prev.op.name == 'relax.reshape':
				res = relax.Call(prev.op, (prev_prev.args[0], prev.args[1]), prev.attrs)
			elif prev.op.name == 'relax.nn.max_pool2d':
				res = relax.Call(prev.op, (prev_prev.args[0],), prev.attrs)
			elif prev.op.name == 'relax.nn.relu':
				res = relax.Call(prev.op, (prev_prev.args[0],), prev.attrs)

		if res is call:
			res = super().visit_call_(call)

		return res

# Apparently it is a common pattern to wrap non quantized operations in
# dequantize quantize blocks to fake support for them and leave the support to
# the backend/runtime.
# https://github.com/onnx/onnx/issues/5895#issuecomment-1928446285
# TODO: in the future this should support also binary operations like addition
@ir.transform.module_pass(opt_level=0)
class RemoveUnnecessaryDequantizeQuantizeWrapping:
	def transform_module(self, mod, ctx):
		rewriter = UnnecessaryDequantizeQuantizeWrappingRemover(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = rewriter.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

def to_dict(attrs: ir.Attrs) -> Dict:
	if str(attrs).startswith('relax.attrs.QuantizeAttrs'):
		return {'axis': attrs.axis, 'out_dtype': attrs.out_dtype}
	assert isinstance(attrs, ir.Attrs)
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

@ir.transform.module_pass(opt_level=0)
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
class ConstAstypeSimplifier(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)
		self.pattern = relax.dpl.is_op("relax.astype")(relax.dpl.is_const())

	def visit_call_(self, call):

		res = call
		if self.pattern.match(call):
			res = relax.const(call.args[0].data.numpy().astype(call.struct_info.dtype))

		return res

@ir.transform.module_pass(opt_level=0)
class SimplifyConstAstype:
	def transform_module(self, mod, ctx):
		rewriter = ConstAstypeSimplifier(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = rewriter.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

@relax.expr_functor.mutator
class RingSimplifier(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)
		# Yes, the pattern of commutative operations is commutative.
		self.const = relax.dpl.is_const()
		self.var = relax.dpl.wildcard()
		self.op = (
			relax.dpl.is_op("relax.multiply")
			| relax.dpl.is_op("relax.add")
			| relax.dpl.is_op("relax.subtract")
		)
		self.pattern = self.op(self.const, self.var) | self.op(self.var, self.const)

	def visit_function_(self, func: relax.Function) -> relax.Expr:
		self.var2val = relax.analysis.get_var2val(func)
		return super().visit_function_(func)

	def visit_call_(self, call):

		res = call
		matched_expr = self.pattern.extract_matched_expr(call, self.var2val)
		if matched_expr:
			matched_const = matched_expr[self.const]
			const = self.visit_expr(matched_const)
			var = self.visit_expr(matched_expr[self.var])
			const_np = const.data.numpy()
			# TODO: what if both are const?
			if call.op.name == "relax.multiply":
				if (const_np == 1).all():
					res = var
				elif (const_np == 0).all():
					res = const
			if call.op.name == "relax.add":
				if (const_np == 0).all():
					res = var
			if call.op.name == "relax.subtract":
				if (const_np == 0).all():
					is_lhs = matched_const.same_as(self.builder_.lookup_binding(call.args[0]))
					if is_lhs:
						res = relax.op.negative(var)
					else:
						res = var

		if res is call:
			new_args = [self.visit_expr(arg) for arg in call.args]
			res = relax.Call(call.op, new_args, call.attrs)
		return res

@ir.transform.module_pass(opt_level=0)
class SimplifyRing:
	def transform_module(self, mod, ctx):
		rewriter = RingSimplifier(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				old_func = func
				updated_func = rewriter.visit_expr(old_func)
				while not ir.structural_equal(updated_func, old_func):
					old_func = updated_func
					updated_func = rewriter.visit_expr(old_func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

@relax.expr_functor.mutator
class AddChainSimplifier(relax.PyExprMutator):
	def __init__(self, mod: tvm.IRModule) -> None:
		super().__init__(mod)
		# lvN: = R.add(metadata["relax.expr.Constant"][N], lvO)
		# lvM: = R.add(lvN, metadata["relax.expr.Constant"][M])
		self.c1 = relax.dpl.is_const()
		self.c2 = relax.dpl.is_const()
		self.var = relax.dpl.wildcard()
		# TODO: make this invariant to "chain" length
		self.pattern = relax.dpl.is_op("relax.add")(
			relax.dpl.is_op("relax.add")(self.c1, self.var),
			self.c2
		)

	def visit_function_(self, func: relax.Function) -> relax.Expr:
		self.var2val = relax.analysis.get_var2val(func)
		return super().visit_function_(func)

	def visit_call_(self, call: relax.Call) -> relax.Expr:
		# We rebuild the graph.
		new_args = [self.visit_expr(arg) for arg in call.args]

		res = call
		matched_expr = self.pattern.extract_matched_expr(call, self.var2val)
		if matched_expr:
			new_data = matched_expr[self.c1].data.numpy() + matched_expr[self.c2].data.numpy()
			new_var = self.visit_expr(matched_expr[self.var])

			res = relax.op.add(new_var, relax.const(new_data))

		if res is call:
			res = relax.Call(call.op, new_args, call.attrs)
		return res

@ir.transform.module_pass(opt_level=0)
class AddChainSimplify:
	def transform_module(self, mod, ctx):
		rewriter = AddChainSimplifier(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				old_func = func
				updated_func = rewriter.visit_expr(old_func)
				while not ir.structural_equal(updated_func, old_func):
					old_func = updated_func
					updated_func = rewriter.visit_expr(old_func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

@relax.expr_functor.mutator
class FunctionToBindRewriter(relax.PyExprMutator):
	def __init__(self, mod: ir.IRModule) -> None:
		super().__init__(mod)
		self.mod = mod

	def visit_call_(self, call: relax.Call) -> relax.Expr:
		res = call
		if isinstance(call.op, ir.GlobalVar):

			params_to_bind = {}
			params_to_leave = []
			for arg, var in zip(call.args, self.mod[call.op].params):
				if isinstance(arg, relax.Constant) \
					and not arg.data.numpy().shape:
					params_to_bind[var] = arg
				else:
					params_to_leave.append(arg)
			if params_to_bind:
				new_func = self.mod[call.op].bind_params(params_to_bind)
				new_func_name = self.builder_.get_unique_name(call.op.name_hint + "_binded")
				# This populates the struct_info of gv!
				gv = self.builder_.add_func(new_func, new_func_name)
				res = relax.Call(gv, params_to_leave, call.attrs)
		if res is call:
			res = super().visit_call_(call)
		return res

@ir.transform.module_pass(opt_level=0)
class BindScalarToFunctions:
	def transform_module(self, mod, ctx):
		rewriter = FunctionToBindRewriter(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = rewriter.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()
