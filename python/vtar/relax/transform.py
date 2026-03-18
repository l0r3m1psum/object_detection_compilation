import tvm
from tvm import ir, relax, topi, tir

import numpy
from typing import List, Tuple, Dict, Set
from collections import defaultdict

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
	def __init__(self, mod: tvm.IRModule, bitpack_start: str = "relax.nn.max_pool2d", bitpack_end: str = "relax.nn.avg_pool2d") -> None:
		super().__init__(mod)
		self.bitpack_start = bitpack_start
		self.bitpack_end = bitpack_end
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
			elif call.op.name == "relax.nn.avg_pool2d":
				data, = packed_args
				res = self.builder_.emit(relax.op.nn.avg_pool2d(
					data,
					call.attrs["pool_size"],
					call.attrs['strides'],
					call.attrs['padding'],
					call.attrs['dilation'],
					call.attrs['ceil_mode'],
					call.attrs['count_include_pad'],
					self.conv_data_layout,
				))
			elif (
				call.op.name == 'relax.add'
				or call.op.name == 'relax.multiply'
				or call.op.name == "relax.right_shift"
				or call.op.name == "relax.left_shift"
				or call.op.name == "relax.greater_equal"
				or call.op.name == "relax.less_equal"
				or call.op.name == "relax.greater"
				or call.op.name == "relax.less"
				or call.op.name == "relax.bidi_shift"
			):
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
							raise ValueError("Broadcasted %s is only supported channel dimension" % call.op.name)
						bias = pad_channel(self.builder_, bias, self.cfactor)
						bias = pack_nchw_to_NCHWnc(self.builder_, bias, 1, self.cfactor)
					# NOTE: I don't know why this if before was not necessary and now it is...
					if len(data_shape) != 6:
						data = pack_nchw_to_NCHWnc(self.builder_, data, 1, self.cfactor)
					res = self.builder_.emit(relax.Call(call.op, (data, bias)))
			elif call.op.name == "relax.where":
				arg0_shape = _get_shape(packed_args[0])
				arg1_shape = _get_shape(packed_args[1])
				arg2_shape = _get_shape(packed_args[2])
				if arg0_shape == arg1_shape == arg2_shape:
					pass
				elif is_broadcastable(arg0_shape, arg1_shape) and is_broadcastable(arg1_shape, arg2_shape):
					pass
				elif (
					(not arg0_shape and not arg1_shape)
					or (not arg1_shape and not arg2_shape)
					or (not arg0_shape and not arg2_shape)
				): # two of the three are a scalar
					pass
				else:
					arg0, arg1, arg2 = packed_args
					if len(arg0_shape) != 6:
						arg0 = pack_nchw_to_NCHWnc(self.builder_, arg0, 1, self.cfactor)
					if len(arg1_shape) != 6:
						arg1 = pack_nchw_to_NCHWnc(self.builder_, arg1, 1, self.cfactor)
					if len(arg2_shape) != 6:
						arg2 = pack_nchw_to_NCHWnc(self.builder_, arg2, 1, self.cfactor)
					res = self.builder_.emit(relax.Call(call.op, (arg0, arg1, arg2)))
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
	def __init__(self, bitpack_start: str = "relax.nn.max_pool2d", bitpack_end: str = "relax.nn.avg_pool2d") -> None:
		self.bitpack_start = bitpack_start
		self.bitpack_end = bitpack_end

	def transform_module(self, mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
		"""IRModule-level transformation"""

		# https://matt.might.net/articles/a-normalization/
		mod = relax.transform.Normalize()(mod) # should remove nested relax calls

		rewriter = GraphPacker(mod, self.bitpack_start, self.bitpack_end)
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
# TODO: rename in QDQPairSimplify
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
			# TODO: add support for division by one and shifts by zero.

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

@relax.expr_functor.visitor
class OpAttrCounter(relax.PyExprVisitor):
	def __init__(self):
		super().__init__()
		self.counts = defaultdict(int)

	def visit_call_(self, call: relax.Call) -> None:
		is_non_module_method = isinstance(call.op, ir.Op)
		if is_non_module_method:
			op_name = call.op.name
			# TODO: some attributes can be "compressed" thing of
			# stride=1 vs stride=(1,1), also integers are printed as
			# T.int64...

			if call.attrs is None:
				dict_attrs = {}
			if isinstance(call.attrs, (ir.Attrs, ir.DictAttrs)):
				dict_attrs = dict(call.attrs)
			elif isinstance(call.attrs, tvm.runtime.Object):
				# FIXME: how do I discriminate over non QuantizeAttrs?
				dict_attrs = {
					"axis": call.attrs.axis,
					"out_dtype": call.attrs.out_dtype,
				}
			attrs_key = str(dict_attrs)

			self.counts[(op_name, attrs_key)] += 1

	def print_report(self) -> None:
		print("%-30s | %-5s | Attributes" % ('Operation', 'Count'))
		print("-" * 80)

		sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)

		for (op, attrs), count in sorted_counts:
			attr_display = (attrs[:75] + '..') if len(attrs) > 75 else attrs

			print("%-30s | %-5d | %s" % (op, count, attr_display))

@ir.transform.module_pass(opt_level=0)
def print_report(mod, ctx):
	counter = OpAttrCounter()
	for function in mod.functions.values():
		if isinstance(function, relax.Function):
			counter.visit_expr(function)
	counter.print_report()
	return mod

@tvm.ir.transform.module_pass(opt_level=0)
class RemoveRelu:
	def transform_module(self, mod, ctx):
		pat_input = relax.dpl.wildcard()
		pat_const = relax.dpl.is_const()

		pattern = relax.dpl.is_op("relax.maximum")(pat_const, pat_input)
		pattern = relax.dpl.is_op("relax.astype")(pattern)
		pattern = relax.dpl.is_op("relax.nn.relu")(pattern)
		# NOTE: this pattern matches only if used from relax.dpl.rewrite_call
		# and not if used bear from a relax.PyExprVisitor.visit_call_
		# the problem may be caused by the fact that if the operation is
		# interleaved with other like so
		# relax.maximum
		# relax.add
		# relax.astype
		# relax.relu

		def rewriter(call, match_map) -> relax.Expr:
			const_expr = match_map[pat_const]
			input_expr = match_map[pat_input]

			const_np = const_expr.data.numpy()
			if (const_np <= 0).all():
				res = relax.op.maximum(relax.const(0, input_expr.struct_info.dtype), input_expr)
				res = relax.op.astype(res, dtype=call.struct_info.dtype)
			else:
				res = call

			return res

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				new_func = relax.dpl.rewrite_call(pattern, rewriter, func)
				new_func = relax.analysis.remove_all_unused(new_func)
				mod.update_func(global_var, new_func)

		return mod

@tvm.ir.transform.module_pass(opt_level=0)
class RewriteBidiShift:
	def transform_module(self, mod, ctx):
		data_pat = relax.dpl.wildcard()
		shift_pat = relax.dpl.is_const()
		zero_pat = relax.dpl.is_const()

		pattern = relax.dpl.is_op("relax.where")(
			(
				relax.dpl.is_op("relax.greater_equal")
				| relax.dpl.is_op("relax.greater")
			)(shift_pat, zero_pat)
			| (
				relax.dpl.is_op("relax.less_equal")
				| relax.dpl.is_op("relax.less")
			)(zero_pat, shift_pat),
			relax.dpl.is_op("relax.right_shift")(data_pat, shift_pat),
			relax.dpl.is_op("relax.left_shift")(
				data_pat,
				relax.dpl.is_op("relax.negative")(shift_pat)
			),
		)

		def rewriter(call, match_map) -> relax.Expr:
			const_expr = match_map[zero_pat]

			const_np = const_expr.data.numpy()
			res = call
			if (const_np == 0).all():
				res = relax.Call(
					ir.Op.get("relax.bidi_shift"),
					(match_map[data_pat], match_map[shift_pat])
				)

			return res

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				new_func = relax.dpl.rewrite_call(pattern, rewriter, func)
				new_func = relax.analysis.remove_all_unused(new_func)
				mod.update_func(global_var, new_func)

		return mod

def optimize_scales(s_y: float, s_i: numpy.ndarray) -> numpy.ndarray:
	"""
	s_y: output scale of the "summation"
	s_i: input scales of every addition the "summation"
	"""
	if len(s_i.shape) != 1: raise ValueError("s_i must be a vector")
	K = s_i.size
	best_error = numpy.inf
	best_result = None

	exact_n = numpy.log2(s_i/s_y)
	floor_choice = numpy.floor(exact_n)
	ceil_choice = numpy.ceil(exact_n)
	choices = numpy.stack((floor_choice, ceil_choice))
	iota = numpy.arange(K, dtype="int64")
	mask = numpy.ones(K, dtype="int64") << iota

	for i in range(2**K):
		# Consider that a number written with bits that counts from 0 to 2**K-1
		# enumerates all possible {0,1}**K strings. Hence using a mask to detect
		# if a bit is set or not we can use it to select from one row or another
		# of the choices matrix.
		row_indices = ((i & mask) > 0).astype(int)
		n = choices[row_indices, iota]
		sum_num = numpy.sum(2**n * s_i)
		sum_den = numpy.sum(2**(2*n))
		delta_s_y = (sum_num - s_y * sum_den) / (1 + sum_den)
		delta_s_i = (2**n) * (s_y + delta_s_y) - s_i
		delta_s = numpy.concatenate(((delta_s_y,), delta_s_i))
		error = numpy.linalg.norm(delta_s)
		if error < best_error:
			best_error = error
			best_result = n

	return best_result.astype(int)

def find_connected_linear_ops(
	root_var: relax.Var,
	bindings: Dict[relax.Var, relax.Expr],
) -> Tuple[List[relax.Var], List[float]]:
	connected_linear_ops = []
	leaf_scales = []

	def _trace(var):
		expr = bindings[var]

		if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
			if expr.op.name == "relax.qnn.add": # Linear ops
				connected_linear_ops.append(var)
				(
					a, s_a, z_a,
					b, s_b, z_b,
					s_c, z_c,
				) = expr.args

				if (
					(s_a.data.numpy() <= 0).any()
					or (s_b.data.numpy() <= 0).any()
					or (s_c.data.numpy() <= 0).any()
				):
					raise ValueError("All scales of quantized addition must be"
						" strictly positive.")
				if (z_c.data.numpy() != 0).all():
					raise ValueError("Quantized addition with output zero point"
						" different form zero is not supported.")

				if isinstance(a, relax.Var):
					expr = bindings[a]
					if (
						isinstance(expr, relax.Call)
						and hasattr(expr.op, "name")
						and expr.op.name != "relax.qnn.add"
						and expr.op.name != "relax.nn.relu"
					):
						leaf_scales.append(s_a.data.numpy())
					else:
						_trace(a)
				if isinstance(b, relax.Var):
					expr = bindings[b]
					if (
						isinstance(expr, relax.Call)
						and hasattr(expr.op, "name")
						and expr.op.name != "relax.qnn.add"
						and expr.op.name != "relax.nn.relu"
					):
						leaf_scales.append(s_b.data.numpy())
					else:
						_trace(b)
			elif expr.op.name == "relax.nn.relu": # Homogeneous functions
				_trace(expr.args[0])

	_trace(root_var)
	return connected_linear_ops, leaf_scales

def rebuild_tree(root_var: relax.Var, var2val: Dict, pots: List[int]) -> Dict[relax.Var, dict]:
	# Use an iterator so we pop the exact n_i corresponding to the leaves in DFS
	# order as the find_connected_linear_ops function does.
	pot_iter = iter(pots)
	tree_nodes = {}

	def _trace(var: relax.Var, is_root: bool = False):
		expr = var2val[var]

		if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
			if expr.op.name == "relax.qnn.add":
				a, s_a, z_a, b, s_b, z_b, s_c, z_c = expr.args

				node_info = {
					"op": "qnn.add",
					"is_root": is_root,
				}
				if is_root:
					node_info["out_dtype"] = root_var.struct_info.dtype

				# Checks if leaves needs to be produced for LHS.
				if isinstance(a, relax.Var):
					expr_a = var2val.get(a)
					if isinstance(expr_a, relax.Call) and getattr(expr_a.op, "name", "") in ["relax.qnn.add", "relax.nn.relu"]:
						node_info["lhs_is_leaf"] = False
						_trace(a)
					else:
						node_info["lhs_is_leaf"] = True
						node_info["lhs_n"] = next(pot_iter)
				else:
					node_info["lhs_is_leaf"] = True
					node_info["lhs_n"] = next(pot_iter)

				# Checks if leaves needs to be produced for RHS.
				if isinstance(b, relax.Var):
					expr_b = var2val.get(b)
					if isinstance(expr_b, relax.Call) and getattr(expr_b.op, "name", "") in ["relax.qnn.add", "relax.nn.relu"]:
						node_info["rhs_is_leaf"] = False
						_trace(b)
					else:
						node_info["rhs_is_leaf"] = True
						node_info["rhs_n"] = next(pot_iter)
				else:
					node_info["rhs_is_leaf"] = True
					node_info["rhs_n"] = next(pot_iter)

				tree_nodes[var] = node_info

			elif expr.op.name == "relax.nn.relu":
				node_info = {
					"op": "nn.relu",
					"is_root": is_root,
				}
				if is_root:
					node_info["out_dtype"] = root_var.struct_info.dtype

				_trace(expr.args[0])
				tree_nodes[var] = node_info

	_trace(root_var, is_root=True)

	return tree_nodes

@relax.expr_functor.mutator
class ReScaleMutator(relax.PyExprMutator):

	def visit_function_(self, func: relax.Function) -> relax.Function:
		# Instead of roots_and_pots, we now track all individual nodes of the tree
		self.tree_nodes: Dict[relax.Var, dict] = {}
		self.already_connected_linear_ops: Set[relax.Var] = set()
		self.var2val: Dict[relax.Var, relax.Expr] = relax.analysis.get_var2val(func)

		return super().visit_function_(func)

	def visit_dataflow_block_(self, block: relax.DataflowBlock) -> relax.DataflowBlock:
		# Backward pass: identify roots and build the configuration dictionary
		for binding in reversed(block.bindings):
			var, expr = binding.var, binding.value
			if isinstance(expr, relax.Call) and getattr(expr.op, "name", "") == "relax.qnn.add":

				if var in self.already_connected_linear_ops:
					continue

				connected_linear_ops, leaf_scales = find_connected_linear_ops(var, self.var2val)

				self.already_connected_linear_ops.update(connected_linear_ops)

				(
					_, _, _,
					_, _, _,
					s_y, _,
				) = self.var2val[connected_linear_ops[0]].args

				pots = optimize_scales(s_y.data.numpy().item(), numpy.array(leaf_scales))

				# Update the mutator's node mapping with all nodes in this specific tree
				tree_dict = rebuild_tree(var, self.var2val, pots)
				self.tree_nodes.update(tree_dict)

		# Standard forward pass mutation over bindings using our mapping
		return super().visit_dataflow_block_(block)

	def _build_leaf(self, arg: relax.Expr, zero_point: relax.Expr, n: int) -> relax.Expr:
		"""Helper method: Emits (a - z_a) << n / >> -n"""
		val_i32 = self.builder_.emit(relax.op.astype(arg, "int32"))
		if isinstance(zero_point, relax.Constant) and (zero_point.data.numpy() == 0).all():
			diff = val_i32
		else:
			zp_i32 = self.builder_.emit(relax.op.astype(zero_point, "int32"))
			diff = self.builder_.emit(relax.op.subtract(val_i32, zp_i32))

		n_val = int(n)
		if n_val > 0:
			shift_const = relax.const(n_val, "int32")
			return self.builder_.emit(relax.op.left_shift(diff, shift_const))
		elif n_val < 0:
			shift_amount = -n_val
			rounding_bias = relax.const(1 << (shift_amount - 1), "int32")
			diff_rounded = self.builder_.emit(relax.op.add(diff, rounding_bias))
			shift_const = relax.const(shift_amount, "int32")
			return self.builder_.emit(relax.op.right_shift(diff_rounded, shift_const))
		else:
			return diff

	def visit_var_binding_(self, binding: relax.VarBinding) -> None:
		# If the currently visited variable is part of an optimized tree
		if binding.var in self.tree_nodes:
			info = self.tree_nodes[binding.var]
			expr = self.var2val[binding.var]

			if info["op"] == "qnn.add":
				a, s_a, z_a, b, s_b, z_b, s_c, z_c = expr.args

				# `self.visit_expr()` acts as a lookup. If `a` or `b` are internal nodes
				# they will correctly return the newly mapped int32 variables generated previously.
				if info["lhs_is_leaf"]:
					lhs = self._build_leaf(self.visit_expr(a), self.visit_expr(z_a), info["lhs_n"])
				else:
					lhs = self.visit_expr(a)

				if info["rhs_is_leaf"]:
					rhs = self._build_leaf(self.visit_expr(b), self.visit_expr(z_b), info["rhs_n"])
				else:
					rhs = self.visit_expr(b)

				res = self.builder_.emit(relax.op.add(lhs, rhs))

			elif info["op"] == "nn.relu":
				inner_expr = self.visit_expr(expr.args[0])
				res = self.builder_.emit(relax.op.nn.relu(inner_expr))

			# If this is the root of the tree, it is finally cast back to the original datatype
			if info["is_root"]:
				out_dtype = info["out_dtype"]
				# Enforce "int32" type for minimum/maximum bounds to match the type of `res`
				res = self.builder_.emit(relax.op.minimum(res, relax.const(tir.max_value(out_dtype).value, "int32")))
				res = self.builder_.emit(relax.op.maximum(relax.const(tir.min_value(out_dtype).value, "int32"), res))
				res = self.builder_.emit(relax.op.astype(res, out_dtype))

			# Rebind the original variable ID to the AST
			if isinstance(binding.var, relax.DataflowVar):
				new_var = self.builder_.emit(res, name_hint=binding.var.name_hint)
			else:
				new_var = self.builder_.emit_output(res, name_hint=binding.var.name_hint)

			self.set_var_remap(binding.var.vid, new_var)
			return

		# Fallback to standard AST propagation for operations outside of the qnn.add trees
		super().visit_var_binding_(binding)

@ir.transform.module_pass(opt_level=0)
class ReScale:
	def transform_module(self, mod, ctx):
		rewriter = ReScaleMutator(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = rewriter.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				rewriter.builder_.update_func(global_var, updated_func)

		return rewriter.builder_.get()

@ir.transform.module_pass(opt_level=0)
class RewriteQDQPatterns:
	def transform_module(self, mod, ctx):

		qconv2d_x = relax.dpl.wildcard()
		qconv2d_x_s = relax.dpl.is_const()
		qconv2d_x_zp = relax.dpl.is_const()

		qconv2d_w = relax.dpl.wildcard()
		qconv2d_w_s = relax.dpl.is_const()
		qconv2d_w_zp = relax.dpl.is_const()

		qconv2d_b = relax.dpl.is_const()
		qconv2d_b_s = relax.dpl.is_const()
		qconv2d_b_zp = relax.dpl.is_const()

		qconv2d_y_s = relax.dpl.is_const()
		qconv2d_y_zp = relax.dpl.is_const()

		qconv2d_relu = relax.dpl.is_op("relax.nn.relu")

		qconv2d_conv = relax.dpl.is_op("relax.nn.conv2d")(
			relax.dpl.is_op("relax.dequantize")(
				qconv2d_x, qconv2d_x_s, qconv2d_x_zp
			),
			relax.dpl.is_op("relax.dequantize")(
				qconv2d_w, qconv2d_w_s, qconv2d_w_zp
			),
		)
		qconv2d = relax.dpl.is_op("relax.add")(
			qconv2d_conv,
			relax.dpl.is_op("relax.reshape")(
				relax.dpl.is_op("relax.dequantize")(qconv2d_b, qconv2d_b_s, qconv2d_b_zp) | qconv2d_b,
				relax.dpl.wildcard()
			)
		) | qconv2d_conv
		qconv2d = qconv2d_relu(qconv2d) | qconv2d
		qconv2d = relax.dpl.is_op("relax.quantize")(
			qconv2d, qconv2d_y_s, qconv2d_y_zp
		)

		qadd_a = relax.dpl.wildcard()
		qadd_a_s = relax.dpl.is_const()
		qadd_a_zp = relax.dpl.is_const()

		qadd_b = relax.dpl.wildcard()
		qadd_b_s = relax.dpl.is_const()
		qadd_b_zp = relax.dpl.is_const()

		qadd_c_s = relax.dpl.is_const()
		qadd_c_zp = relax.dpl.is_const()

		qadd_relu = relax.dpl.is_op("relax.nn.relu")

		qadd = relax.dpl.is_op("relax.add")(
			relax.dpl.is_op("relax.dequantize")(qadd_a, qadd_a_s, qadd_a_zp),
			relax.dpl.is_op("relax.dequantize")(qadd_b, qadd_b_s, qadd_b_zp),
		)
		qadd = qadd_relu(qadd) | qadd
		qadd = relax.dpl.is_op("relax.quantize")(qadd, qadd_c_s, qadd_c_zp)

		# A single pattern must be used in this case because is this is
		# splitted in two passes of relax.dpl.rewrite_call (or using
		# relax.transform.FuseOpsByPattern with two different patterns) there
		# are some dequantize which have out degree 2 i.e are used by more than
		# one node after. If two patterns are used the dequantize node are
		# removed by the first and the second one can't use it.
		pattern = qadd | qconv2d

		def rewriter(call: relax.Call, match_map: ir.Map) -> relax.Expr:
			if qadd in match_map:
				res = relax.Call(
					ir.Op.get("relax.qnn.add"),
					(
						match_map[qadd_a], match_map[qadd_a_s], match_map[qadd_a_zp],
						match_map[qadd_b], match_map[qadd_b_s], match_map[qadd_b_zp],
						match_map[qadd_c_s], match_map[qadd_c_zp],
					)
				)
				if qadd_relu in match_map:
					res = relax.op.nn.relu(res)
				return res
			else:
				args = [
					match_map[qconv2d_x], match_map[qconv2d_x_s], match_map[qconv2d_x_zp],
					match_map[qconv2d_w], match_map[qconv2d_w_s], match_map[qconv2d_w_zp],
					match_map[qconv2d_y_s], match_map[qconv2d_y_zp],
				]
				if qconv2d_b in match_map:
					if qconv2d_b_s in match_map:
						# NOTE: is there a way to avoid re dequantization?
						b_s_old = match_map[qconv2d_b_s].data.numpy()
						b_zp_old = match_map[qconv2d_b_zp].data.numpy()
						b_float = (match_map[qconv2d_b].data.numpy() - b_zp_old) * b_s_old
					else:
						# 1. It is a float tensor (unquantized)
						b_float = match_map[qconv2d_b].data.numpy()

					# 2. Requantize using the ONNX convention
					b_s = (
						match_map[qconv2d_x_s].data.numpy().astype("float64")
						* match_map[qconv2d_w_s].data.numpy().astype("float64")
					).astype("float32")
					b_zp = numpy.zeros_like(b_s)

					b = relax.const(
						numpy.clip(
							numpy.round(b_float / b_s + b_zp),
							numpy.iinfo("int32").min,
							numpy.iinfo("int32").max,
						).astype("int32")
					)
					args.append(b)
				new_attrs = {str(k): v for k, v in dict(match_map[qconv2d_conv].attrs).items()}
				new_attrs['out_dtype'] = 'int32'
				res = relax.Call(
					ir.Op.get("relax.qnn.conv2d"),
					args,
					attrs=ir.make_node("relax.attrs.Conv2DAttrs", **new_attrs)
				)
				if qconv2d_relu in match_map:
					res = relax.op.nn.relu(res)
				return res

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				new_func = relax.dpl.rewrite_call(pattern, rewriter, func)
				new_func = relax.analysis.remove_all_unused(new_func)
				mod.update_func(global_var, new_func)

		return mod

# TODO: write test for this function.
def get_strictly_power_of_two(arr) -> Tuple[numpy.ndarray, numpy.ndarray]:
	x = numpy.asanyarray(arr, dtype=numpy.float32)
	x_bits = x.view(numpy.int32)

	SIGN_MASK     = numpy.asarray(-2147483648, dtype='int32') # 0x80000000
	EXPONENT_MASK = 0x7F800000
	MANTISSA_MASK = 0x007FFFFF
	EXPONENT_SIZE_MASK = 0xFF
	MANTISSA_SIZE = 23
	BIAS = 127

	# https://it.wikipedia.org/wiki/IEEE_754#Precisione_singola_(32_bit)
	not_inf_or_nan = (x_bits < EXPONENT_MASK)
	not_neg_inf_or_nan = (x_bits > 0) & not_inf_or_nan
	has_zero_mantissa = (x_bits & (MANTISSA_MASK | SIGN_MASK)) == 0
	is_pow2 = not_neg_inf_or_nan & has_zero_mantissa

	powers = ((x_bits >> MANTISSA_SIZE) & EXPONENT_SIZE_MASK) - BIAS

	return is_pow2, powers
	# return numpy.where(is_pow2, powers, numpy.nan)

def clamp(data: relax.Expr, min, max) -> relax.Expr:
	res = relax.op.minimum(data, relax.const(max))
	res = relax.op.maximum(relax.const(min), res)
	return res

def const_astype(x: relax.Constant, dtype: str) -> relax.Constant:
	# TODO: check that conversion is safe to do
	return relax.const(x.data.numpy().astype(dtype))

def requantize(s: relax.Constant, x: relax.Expr, z: relax.Constant) -> relax.Expr:
	res = x
	if not (s.data.numpy() == 1).all():
		res *= s
	if not (z.data.numpy() == 0).all():
		res += const_astype(z, "float32")
	res = relax.op.round(res)
	res = clamp(res, -128., 127.).astype("int8")
	return res

from typing import Optional

def integer_only_arithmetic(
	M: relax.Constant,
	s_w: numpy.ndarray,
	q_w: relax.Constant,
	z_w: relax.Constant,
	B: Optional[relax.Constant]
) -> Tuple[numpy.ndarray, relax.Expr, Optional[relax.Constant]]:
	"""From "Speed up integer-arithmetic-only inference via bit-shifting" """

	# Global scale optimization
	is_pow2, powers = get_strictly_power_of_two(s_w)
	if (M.data.numpy() == s_w).all() and is_pow2.all():
		# Powers needs to be negated because... Look at ioa_requantize
		return numpy.array(-powers.item()).astype("int32"), q_w, B

	M = M.data.numpy()
	n = numpy.floor(-numpy.log2(M)).astype("int32")
	# if (n < 0).any(): print(n, M)
	M_star = 2.**(-n)
	assert ((2.**(-(n + 1)) <= M) & (M <= M_star)).all(), "M = %s, n = %s" % (M, n)
	assert ((1 <= M_star/M) & (M_star/M < 2)).all()

	s_star_w = M_star/M * s_w
	assert (s_star_w >= s_w).all()

	q_star_w = (const_astype(q_w, "int32") - const_astype(z_w, "int32")).astype("float32")
	q_star_w = requantize(relax.const(s_w/s_star_w), q_star_w , z_w)
	if B:
		B = relax.const(numpy.round(s_w/s_star_w*B.data.numpy()).astype("int32"))
	return n, q_star_w, B

def ioa_requantize(
	bb: relax.BlockBuilder,
	N: numpy.ndarray,
	x: relax.Expr,
	z: relax.Constant
) -> relax.Expr:
	# This is horrible. N is the inverted exponent of 2 to perform the
	# multiplication in IOA i.e. M_star = 2**(-n).
	is_pos = -N > 0
	is_neg = -N < 0
	magnitude = relax.const(numpy.where(is_neg, -(-N), -N))
	# https://docs.amd.com/r/en-US/ug1399-vitis-hls/Class-Methods-and-Operators#:~:text=b%2B1%3B%0A%7D-,Shift%20Operators,-Each%20shift%20operator
	is_scalar = not is_pos.shape
	# This implements round to nearest after multiplication
	# because the semantics of x >> n is floor(x >> n) and to get round(x >> n)
	# we need to do x + 2^(n-1) >> n.
	# Note that 2^(n-1) is 1 << (n-1) and could be calculated inside VTA.
	# TODO: check that for left_shift is right also
	if is_scalar:
		x = x + relax.const(2**(magnitude.data.numpy().item()-1))
		res = relax.op.left_shift(x, magnitude) if is_pos \
			else relax.op.right_shift(x, magnitude)
	else:
		x = x + relax.const(2**(magnitude.data.numpy()-1))
		A = relax.const(N)
		res = relax.op.where(
			A >= relax.const(0),
			relax.op.right_shift(x, A),
			relax.op.left_shift(x, -A)
		)

	# FIXME: handle y zero point when different from zero.
	if not (z.data.numpy() == 0).all():
		print("Non zero z_y = %f" % z.data.numpy())

	return clamp(res + const_astype(z, "int32"), -128, 127).astype("int8")

def reshape_if_needed(x: relax.Constant, shape: Tuple[int, int, int, int]) -> relax.Constant:
	x_np = x.data.numpy()
	# FIXME: GrapPhack breaks if scalar are reshaped to (1, 1, 1, 1)...
	res = relax.const(x_np.reshape(shape)) if x_np.shape else x
	return res

@relax.expr_functor.mutator
class ReQuantizeMutator(relax.PyExprMutator):
	def visit_call_(self, call: relax.Call) -> relax.Call:
		# Instead of roots_and_pots, we now track all individual nodes of the tree
		if getattr(call.op, "name", "") == "relax.qnn.conv2d":
			(
				x, x_s, x_zp,
				w, w_s, w_zp,
				y_s, y_zp,
			) = call.args[:8]
			x_s = x_s.data.numpy()
			w_s = w_s.data.numpy()
			y_s = y_s.data.numpy()
			b = call.args[8] if len(call.args) == 9 else None

			if (x_s == y_s).all():
				m = relax.const(w_s)
			else:
				m = relax.const((x_s*w_s)/y_s)

			N, W, B = integer_only_arithmetic(
				reshape_if_needed(m, (-1, 1, 1, 1)),
				w_s.reshape((-1, 1, 1, 1)),
				w,
				reshape_if_needed(w_zp, (-1, 1, 1, 1)),
				reshape_if_needed(b, (-1, 1, 1, 1)) if b else b,
			)
			W = self.builder_.normalize(W)
			x_zp = reshape_if_needed(x_zp, (1, -1, 1, 1))
			w_zp = reshape_if_needed(w_zp, (1, -1, 1, 1))
			# TODO: based on kwargs some of those terms can be drastically
			# simplified.
			# TODO: implement "scalarization" of tensors with all the same value.
			kwargs = call.attrs if call.attrs else {}
			conv = (
				const_astype(x_zp, "int32")*const_astype(w_zp, "int32")*relax.op.nn.conv2d(relax.op.ones_like(x), relax.op.ones_like(W), **kwargs)
				- const_astype(w_zp, "int32")*relax.op.nn.conv2d(x, relax.op.ones_like(W), **kwargs)
				- const_astype(x_zp, "int32")*relax.op.nn.conv2d(relax.op.ones_like(x), W, **kwargs)
				+ relax.op.nn.conv2d(x, W, **kwargs)
			)
			res = conv + reshape_if_needed(B, (1, -1, 1, 1)) if B else conv
			y_zp = reshape_if_needed(y_zp, (1, -1, 1, 1))
			# The conditional reshape is needed because the dlight schedule is
			# fragile (also the transform that binds scalars does not recognize
			# tensors of shape (1, 1, 1, 1) as equivalent to scalars)
			return ioa_requantize(self.builder_, N.reshape(1, -1, 1, 1) if N.shape else N, res, y_zp)

		return super().visit_call_(call)

@ir.transform.module_pass(opt_level=0)
class ReQuantize:
	def transform_module(self, mod: ir.IRModule, _ctx: ir.transform.PassContext) -> ir.IRModule:
		mutator = ReQuantizeMutator(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = mutator.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				mod.update_func(global_var, updated_func)

		return mod

@relax.expr_functor.mutator
class LowerQNNOpsMutator(relax.PyExprMutator):
	def visit_call_(self, call: relax.Call) -> relax.Call:
		res = call
		op_name = getattr(call.op, "name", "")
		if op_name == "relax.qnn.conv2d":
			(
				x, x_s, x_zp,
				w, w_s, w_zp,
				y_s, y_zp,
			) = call.args[:8]
			b = call.args[8] if len(call.args) == 9 else None

			kwargs = call.attrs if call.attrs else {}
			m = relax.const(
				(x_s.data.numpy().astype("float64")
					* w_s.data.numpy().astype("float64"))
				/ y_s.data.numpy().astype("float64"),
				dtype="float32"
			)
			x_ones = relax.op.ones_like(x)
			w_ones = relax.op.ones_like(w)
			x_zp_int = reshape_if_needed(const_astype(x_zp, "int32"), (1, -1, 1, 1))
			w_zp_int = reshape_if_needed(const_astype(w_zp, "int32"), (1, -1, 1, 1))

			x_zp_zero = (x_zp.data.numpy() == 0).all()
			w_zp_zero = (w_zp.data.numpy() == 0).all()

			res = relax.op.nn.conv2d(x, w, **kwargs)
			if not x_zp_zero:
				res -= x_zp_int*relax.op.nn.conv2d(x_ones, w, **kwargs)
			if not w_zp_zero:
				res -= w_zp_int*relax.op.nn.conv2d(x, w_ones, **kwargs)
			if not x_zp_zero and not w_zp_zero:
				res += x_zp_int*w_zp_int*relax.op.nn.conv2d(x_ones, w_ones, **kwargs)
			if b:
				res += reshape_if_needed(b, (1, -1, 1, 1))

			res = requantize(
				reshape_if_needed(m, (1, -1, 1, 1)),
				res.astype("float32"),
				reshape_if_needed(y_zp, (1, -1, 1, 1))
			)
		elif op_name == "relax.qnn.add":
			(
				a, a_s, a_zp,
				b, b_s, b_zp,
				c_s, c_zp,
			) = call.args
			# C = (A_s * (A - A_z) + B_s * (B - B_z))/C_s + C_z
			# C = A_s/C_s * (A - A_z) + B_s/C_s * (B - B_z) + C_z
			c_s_float = c_s.data.numpy().astype("float64")
			a_m_float = a_s.data.numpy().astype("float64") / c_s_float
			b_m_float = b_s.data.numpy().astype("float64") / c_s_float
			a_m = relax.const(a_m_float, dtype="float32")
			b_m = relax.const(b_m_float, dtype="float32")

			lhs = a.astype("int32")
			if not (a_zp.data.numpy() == 0).all():
				lhs -= const_astype(a_zp, "int32")
			lhs = lhs.astype("float32")
			if not (a_m_float == 1).all():
				lhs *= a_m

			rhs = b.astype("int32")
			if not (b_zp.data.numpy() == 0).all():
				rhs -= const_astype(b_zp, "int32")
			rhs = rhs.astype("float32")
			if not (b_m_float == 1).all():
				rhs *= b_m

			res = lhs + rhs
			res = requantize(relax.const(1.0), res, c_zp)

		if res is call:
			res = super().visit_call_(call)

		return res

@ir.transform.module_pass(opt_level=0)
class LowerQNNOps:
	def transform_module(self, mod: ir.IRModule, _ctx: ir.transform.PassContext) -> ir.IRModule:
		mutator = LowerQNNOpsMutator(mod)

		for global_var, func in mod.functions.items():
			if isinstance(func, relax.Function):
				updated_func = mutator.visit_expr(func)
				updated_func = relax.analysis.remove_all_unused(updated_func)
				mod.update_func(global_var, updated_func)

		return mod
