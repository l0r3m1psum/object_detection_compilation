from . import qnn

from tvm import relax, ir

# TODO: right now the semantic of bidi_shift is based on the Vitis HLS right
# shift a more natural semantic is to base it on the left shift since at that
# point bidi_shift(x, n) === floor(x * 2**n), while now the exponent of the 2 is
# negated.
# https://docs.amd.com/r/en-US/ug1399-vitis-hls/Class-Methods-and-Operators#:~:text=A%20negative%20value%20supplied%20to%20the%20signed%20RHS%20versions%20reverses%20the%20shift%20operations%20direction.

def infer_struct_info_bidi_shift_op(call: relax.Call, ctx: relax.BlockBuilder) -> relax.StructInfo:
    right_shift_op = ir.Op.get("relax.right_shift")
    infer_fn = right_shift_op.get_attr("FInferStructInfo")
    return infer_fn(call, ctx)

ir.register_op_attr("relax.bidi_shift", "FPurity", True)
ir.register_op_attr("relax.bidi_shift", "FInferStructInfo", infer_struct_info_bidi_shift_op)
bidi_shift_op = ir.Op.get("relax.bidi_shift")
bidi_shift_op.set_num_inputs(2)
bidi_shift_op.add_argument("data", "Tensor", "The input tensor.")
bidi_shift_op.add_argument("shift", "Tensor", "The direction and magnitude of the shift.")

def bidi_shift(data: relax.Expr, shift: relax.Expr) -> relax.Call:
    op = ir.Op.get("relax.bidi_shift")
    return relax.Call(op, (data, shift))

