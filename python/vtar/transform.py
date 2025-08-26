import tvm

# TVM supports data layouts for convolutions with "factors" i.e.
# NCHW([1-9][0-9]*n)?([1-9][0-9]*w)?([1-9][0-9]*h)?([1-9][0-9]*w)?
# where the result resulting shape is {N/n}{C/c}{H/h}{W/w}nchw

# Translating from vta/python/vta/top/graphpack.py

# topi.utils.get_const_tuple
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
