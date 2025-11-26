from typing import Tuple

import tvm
from tvm import tir
from ..environment import Environment

def get_alu_op(
		env: Environment, analyzer: tvm.arith.Analyzer, value: tir.PrimExpr
	) -> Tuple[int, tir.PrimExpr, tir.PrimExpr, Exception]:
	"""If the expression can be implemented with the VTA ALU core its opcode,
	lhs and rhs are returned."""
	alu_opcode = 0
	lhs = rhs = tir.const(0, "int32")
	error = None

	if isinstance(value, tir.expr.BinaryOpExpr):
		lhs = value.a
		rhs = value.b
		if   isinstance(value, tir.Add): alu_opcode = env.dev.ALU_OPCODE_ADD
		elif isinstance(value, tir.Sub): alu_opcode = env.dev.ALU_OPCODE_SUB
		elif isinstance(value, tir.Mul): alu_opcode = env.dev.ALU_OPCODE_MUL
		elif isinstance(value, tir.Min): alu_opcode = env.dev.ALU_OPCODE_MIN
		elif isinstance(value, tir.Max): alu_opcode = env.dev.ALU_OPCODE_MAX
		else:
			error = RuntimeError("Binary op not supported %s" % value.op.name)
	elif isinstance(value, tir.Call):
		if value.op.name == "tir.shift_left":
			alu_opcode = env.dev.ALU_OPCODE_SHR
			lhs = value.args[0]
			rhs = analyzer.simplify(-value.args[1])
		elif value.op.name == "tir.shift_right":
			alu_opcode = env.dev.ALU_OPCODE_SHR
			lhs = value.args[0]
			rhs = value.args[1]
		else:
			error = RuntimeError(
				"Function call not recognized %s" % (value.op.name)
			)
	elif isinstance(value, tir.BufferLoad):
		alu_opcode = env.dev.ALU_OPCODE_SHR
		lhs = value
		rhs = tvm.tir.const(0, "int32")
	else:
		error = RuntimeError(
			"Expression not recognized %s, %s"
			% (type(value), str(value))
		)

	supported_super = (tir.BufferLoad, tir.IntImm)
	if not isinstance(lhs, supported_super):
		error = RuntimeError("Lhs not recognized %s expected one of %s"
			% (type(lhs), supported_super))
	if not isinstance(rhs, supported_super):
		error = RuntimeError("Rhs not recognized %s expected one of %s"
			% (type(lhs), supported_super))

	return alu_opcode, lhs, rhs, error
