import tvm
from tvm import relax, tir, te, ir, topi

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

@I.ir_module
class MyModule:
	@T.prim_func
	def add_one(A: T.Buffer((10,), "int32"), B: T.Buffer((10,), "int32")):
		for i in range(10):
			B[i] = A[i] + 1

def create_add_primfunc(n_val: int) -> tir.PrimFunc:
	dtype = "float32"
	shape = (n_val,)

	var_A = tir.Var("A", 'handle')
	var_B = tir.Var("B", 'handle')
	var_C = tir.Var("C", 'handle')
	var_n = tir.Var("n", "int32")

	buffer_A = tir.decl_buffer(shape, dtype, name="A", strides=[1], elem_offset=None, scope="global", axis_separators=None)
	buffer_B = tir.decl_buffer(shape, dtype, name="B", strides=[1], elem_offset=None, scope="global", axis_separators=None)
	buffer_C = tir.decl_buffer(shape, dtype, name="C", strides=[1], elem_offset=None, scope="global", axis_separators=None)

	buffer_map = {
		var_A: buffer_A,
		var_B: buffer_B,
		var_C: buffer_C,
	}

	# C[i] = A[i] + B[i]
	loop_var_i = tir.Var("i", "int32")
	load_A = tir.BufferLoad(buffer_A, [loop_var_i])
	load_B = tir.BufferLoad(buffer_B, [loop_var_i])
	add_expr = tir.Add(load_A, load_B)
	store_C = tir.BufferStore(buffer_C, add_expr, [loop_var_i])

	for_loop = tir.For(loop_var_i, 0, var_n, tir.ForKind.SERIAL, store_C, annotations={})

	# body = tir.SeqStmt([for_loop])
	body = for_loop

	prim_func = tir.PrimFunc(
		params=[var_A, var_B, var_C, var_n],
		body=body,
		buffer_map=buffer_map,
		attrs=tvm.ir.make_node("DictAttrs", **{"global_symbol": "add_func", "tir.noalias": True}),
		ret_type=None
	)

	call = tir.Call("void", tvm.ir.Op.get("tir.sinh"), (var_A, var_B, var_C, var_n))
	prim_func2 = tir.PrimFunc(
		params=[var_A, var_B, var_C, var_n],
		body=tir.stmt_seq(call),
		buffer_map=buffer_map,
		attrs=tvm.ir.make_node("DictAttrs", **{"global_symbol": "add_func_call", "tir.noalias": True}),
		ret_type=None
	)
	print(prim_func2)

	ib = tir.ir_builder.create()
	n = te.var("n")
	A = ib.allocate("float32", n, name="A")
	with ib.for_range(0, n, name="i") as i:
		with ib.if_scope((i % 2) == 0):
			A[i] = A[i] + 1
	ib.emit(tir.call_tir(MyModule.get_global_var("add_one")))
	stmt = ib.get()
	func = tir.PrimFunc((), stmt)
	print(func)

	return prim_func


@I.ir_module
class Module:
	@T.prim_func(private=True)
	def conv2d_NCHWnc(lv3: T.Buffer((T.int64(1), T.int64(4), T.int64(56), T.int64(56), T.int64(1), T.int64(16)), "int8"), lv4: T.Buffer((T.int64(4), T.int64(4), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "int8"), res: T.Buffer((T.int64(1), T.int64(4), T.int64(58), T.int64(58), T.int64(1), T.int64(16)), "int32")):
		T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
		# with T.block("root"):
		PadInput = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(58), T.int64(58), T.int64(1), T.int64(16)), "int8")
		for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), T.int64(4), T.int64(58), T.int64(58), T.int64(1), T.int64(16)):
			with T.block("PadInput"):
				v_i0, v_i1, v_i2, v_i3, v_i4, v_i5 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
				T.reads(lv3[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1), v_i4, v_i5])
				T.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5])
				PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(57) and T.int64(1) <= v_i3 and v_i3 < T.int64(57), lv3[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1), v_i4, v_i5], T.int8(0))
		for b_o, c_o, i, j, b_i, c_i, k_o, d_i, d_j, k_i in T.grid(T.int64(1), T.int64(4), T.int64(58), T.int64(58), T.int64(1), T.int64(16), T.int64(4), T.int64(1), T.int64(1), T.int64(16)):
			with T.block("res"):
				v_b_o, v_c_o, v_i, v_j, v_b_i, v_c_i, v_k_o, v_d_i, v_d_j, v_k_i = T.axis.remap("SSSSSSRRRR", [b_o, c_o, i, j, b_i, c_i, k_o, d_i, d_j, k_i])
				T.reads(PadInput[v_b_o, v_k_o, v_i + v_d_i, v_j + v_d_j, v_b_i, v_k_i], lv4[v_c_o, v_k_o, v_d_i, v_d_j, v_c_i, v_k_i])
				T.writes(res[v_b_o, v_c_o, v_i, v_j, v_b_i, v_c_i])
				with T.init():
					res[v_b_o, v_c_o, v_i, v_j, v_b_i, v_c_i] = 0
				res[v_b_o, v_c_o, v_i, v_j, v_b_i, v_c_i] = res[v_b_o, v_c_o, v_i, v_j, v_b_i, v_c_i] + T.Cast("int32", PadInput[v_b_o, v_k_o, v_i + v_d_i, v_j + v_d_j, v_b_i, v_k_i]) * T.Cast("int32", lv4[v_c_o, v_k_o, v_d_i, v_d_j, v_c_i, v_k_i])

	@T.prim_func(private=True)
	def fused_reshape1_transpose1(conv1_w: T.Buffer((T.int64(64), T.int64(64), T.int64(1), T.int64(1)), "int8"), T_transpose_intermediate: T.Buffer((T.int64(4), T.int64(4), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "int8")):
		T.func_attr({"tir.noalias": T.bool(True)})
		# with T.block("root"):
		T_reshape_intermediate = T.alloc_buffer((T.int64(4), T.int64(16), T.int64(4), T.int64(16), T.int64(1), T.int64(1)), "int8")
		for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(T.int64(4), T.int64(16), T.int64(4), T.int64(16), T.int64(1), T.int64(1)):
			with T.block("T_reshape"):
				v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
				T.reads(conv1_w[(v_ax0 * T.int64(16) + (v_ax2 * T.int64(16) + v_ax3 + v_ax4 + v_ax5) // T.int64(64) + v_ax1) % T.int64(64), (v_ax2 * T.int64(16) + v_ax3 + v_ax4 + v_ax5) % T.int64(64), T.int64(0), T.int64(0)])
				T.writes(T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5])
				T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5] = conv1_w[(v_ax0 * T.int64(16) + (v_ax2 * T.int64(16) + v_ax3 + v_ax4 + v_ax5) // T.int64(64) + v_ax1) % T.int64(64), (v_ax2 * T.int64(16) + v_ax3 + v_ax4 + v_ax5) % T.int64(64), T.int64(0), T.int64(0)]
		for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(T.int64(4), T.int64(4), T.int64(1), T.int64(1), T.int64(16), T.int64(16)):
			with T.block("T_transpose"):
				v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
				T.reads(T_reshape_intermediate[v_ax0, v_ax4, v_ax1, v_ax5, v_ax2, v_ax3])
				T.writes(T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5])
				T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5] = T_reshape_intermediate[v_ax0, v_ax4, v_ax1, v_ax5, v_ax2, v_ax3]

	@T.prim_func(private=True)
	def fused_reshape_transpose(lv2_1: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "int8"), T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(4), T.int64(56), T.int64(56), T.int64(1), T.int64(16)), "int8")):
		T.func_attr({"tir.noalias": T.bool(True)})
		# with T.block("root"):
		T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4), T.int64(16), T.int64(56), T.int64(56)), "int8")
		for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(16), T.int64(56), T.int64(56)):
			with T.block("T_reshape"):
				v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
				T.reads(lv2_1[T.int64(0), (v_ax2 * T.int64(16) + (v_ax5 // T.int64(56) + v_ax4) // T.int64(56) + v_ax3) % T.int64(64), (v_ax5 // T.int64(56) + v_ax4) % T.int64(56), v_ax5 % T.int64(56)])
				T.writes(T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5])
				T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5] = lv2_1[T.int64(0), (v_ax2 * T.int64(16) + (v_ax5 // T.int64(56) + v_ax4) // T.int64(56) + v_ax3) % T.int64(64), (v_ax5 // T.int64(56) + v_ax4) % T.int64(56), v_ax5 % T.int64(56)]
		for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(T.int64(1), T.int64(4), T.int64(56), T.int64(56), T.int64(1), T.int64(16)):
			with T.block("T_transpose"):
				v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
				T.reads(T_reshape_intermediate[v_ax0, v_ax4, v_ax1, v_ax5, v_ax2, v_ax3])
				T.writes(T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5])
				T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_ax5] = T_reshape_intermediate[v_ax0, v_ax4, v_ax1, v_ax5, v_ax2, v_ax3]

	@T.prim_func(private=True)
	def max_pool2d(lv: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "int8"), pool_max: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "int8")):
		T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
		# with T.block("root"):
		for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(1), T.int64(64), T.int64(56), T.int64(56), T.int64(1), T.int64(1)):
			with T.block("pool_max"):
				v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
				T.reads(lv[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1])
				T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
				T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
				with T.init():
					pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.int8(-128)
				pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(pool_max[v_ax0, v_ax1, v_ax2, v_ax3], lv[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1])

	@T.prim_func(private=True)
	def quantize(input: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), quantized: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "int8")):
		T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
		# with T.block("root"):
		for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(64), T.int64(56), T.int64(56)):
			with T.block("quantized"):
				v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
				T.reads(input[v_i0, v_i1, v_i2, v_i3])
				T.writes(quantized[v_i0, v_i1, v_i2, v_i3])
				quantized[v_i0, v_i1, v_i2, v_i3] = T.Cast("int8", T.max(T.min(T.round(input[v_i0, v_i1, v_i2, v_i3] * T.float32(100.00000223517424)) + T.float32(-128.0), T.float32(127.0)), T.float32(-128.0)))

	@R.function
	def main(input: R.Tensor((1, 64, 56, 56), dtype="float32"), conv1_w: R.Tensor((64, 64, 1, 1), dtype="int8"), conv1_b: R.Tensor((64,), dtype="int32")) -> R.Tensor((1, 4, 58, 58, 1, 16), dtype="int32"):
		R.func_attr({"num_input": 1})
		cls = Module
		with R.dataflow():
			lv = R.call_tir(cls.quantize, (input,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="int8"))
			lv2_1 = R.call_tir(cls.max_pool2d, (lv,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="int8"))
			lv_1 = R.call_tir(cls.fused_reshape_transpose, (lv2_1,), out_sinfo=R.Tensor((1, 4, 56, 56, 1, 16), dtype="int8"))
			lv1 = R.call_tir(cls.fused_reshape1_transpose1, (conv1_w,), out_sinfo=R.Tensor((4, 4, 1, 1, 16, 16), dtype="int8"))
			gv = R.call_tir(cls.conv2d_NCHWnc, (lv_1, lv1), out_sinfo=R.Tensor((1, 4, 58, 58, 1, 16), dtype="int32"))
			R.output(gv)
		return gv

from typing import Tuple
def _get_shape(data: relax.Var) -> Tuple[int]:
	res = topi.utils.get_const_tuple(relax.get_shape_of(data))
	return res

import collections

@relax.expr_functor.visitor
class MyExprVisitor(relax.PyExprVisitor):
	def __init__(self):
		super().__init__()
		self.ib = tir.ir_builder.create()
		self.params = collections.OrderedDict()
		self.locals = {}

	def visit_function_(self, func: relax.Function) -> None:
		for param in func.params:
			self.params[str(param)] = tir.decl_buffer(_get_shape(param), param.struct_info.dtype)
			# self.params[str(param)] = tir.Var(str(param), param.struct_info.dtype)
		super().visit_function_(func)

	def visit_binding(self, binding: relax.Binding) -> None:
		if isinstance(binding.value, relax.Call):
			call = binding.value
			var = binding.var
			out_sinfo = call.sinfo_args[0]
			buf_var = self.ib.allocate(out_sinfo.dtype, topi.utils.get_const_tuple(out_sinfo.shape), str(var))
			self.locals[str(var)] = buf_var

			func_args = call.args[1]
			new_func_args = []
			for func_arg in func_args:
				# Local variables shadow parameters
				if str(func_arg) in self.locals:
					# new_func_args.append(self.locals[str(func_arg)])
					new_func_args.append(tir.Var("placeholder_FIXME", "float32"))
				elif str(func_arg) in self.params:
					new_func_args.append(self.params[str(func_arg)])
				else:
					raise ValueError("arg not found")
			# new_func_args.append(buf_var._buffer)
			# buf_var = call.args[0](*new_func_args)
			self.ib.emit(tir.call_tir(call.args[0], *new_func_args))

	def get_func(self) -> tir.PrimFunc:
		return tir.PrimFunc(self.params.values(), self.ib.get())

# TODO: understand how tir Buffer and Var work.

visitor = MyExprVisitor()
visitor.visit_expr(Module["main"])
func = visitor.get_func()
print(func)

if True:
	n = 128
	my_add_func = create_add_primfunc(n)

	print(my_add_func)
