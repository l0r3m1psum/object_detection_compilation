# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, invalid-name
"""VTA specific buildin for runtime."""
import tvm
import tvm.script.relax

from .tir import transform
from .environment import get_env, Environment

from types import SimpleNamespace

if not hasattr(tvm.script.relax, "qnn"):
    tvm.script.relax.qnn = SimpleNamespace()

def infer_struct_info(call: tvm.relax.Call, ctx: tvm.relax.BlockBuilder) -> tvm.relax.StructInfo:
    right_shift_op = tvm.ir.Op.get("relax.right_shift")
    infer_fn = right_shift_op.get_attr("FInferStructInfo")
    return infer_fn(call, ctx)

tvm.ir.register_op_attr("relax.bidi_shift", "FPurity", True)
tvm.ir.register_op_attr("relax.bidi_shift", "FInferStructInfo", infer_struct_info)
bidi_shift_op = tvm.ir.Op.get("relax.bidi_shift")
bidi_shift_op.set_num_inputs(2)
bidi_shift_op.add_argument("data", "Tensor", "The input tensor.")
bidi_shift_op.add_argument("shift", "Tensor", "The direction and magnitude of the shift.")

def bidi_shift(data: tvm.relax.Expr, shift: tvm.relax.Expr) -> tvm.relax.Call:
    op = tvm.ir.Op.get("relax.bidi_shift")
    return tvm.relax.Call(op, (data, shift))

tvm.script.relax.bidi_shift = bidi_shift

def infer_struct_info_qnn_add_op(call: tvm.relax.Call, ctx: tvm.relax.BlockBuilder) -> tvm.relax.StructInfo:
    if len(call.args) != 8:
        raise ValueError("relax.qnn.add expects exactly 8 arguments.")

    sinfo = []
    for i, arg in enumerate(call.args):
        if not isinstance(arg.struct_info, tvm.relax.TensorStructInfo):
            raise ValueError(f"Argument {i} must be a Tensor.")
        sinfo.append(arg.struct_info)

    a_sinfo = sinfo[0]
    a_scale_sinfo = sinfo[1]
    a_zp_sinfo = sinfo[2]
    b_sinfo = sinfo[3]
    b_scale_sinfo = sinfo[4]
    b_zp_sinfo = sinfo[5]
    c_scale_sinfo = sinfo[6]
    c_zp_sinfo = sinfo[7]

    def is_int(dtype):
        return dtype.startswith("int") or dtype.startswith("uint")

    if not (a_scale_sinfo.dtype.startswith("float") and
            b_scale_sinfo.dtype.startswith("float") and
            c_scale_sinfo.dtype.startswith("float")):
        raise ValueError("All scales must be float tensors.")

    if not (is_int(a_sinfo.dtype) and is_int(b_sinfo.dtype)):
        raise ValueError("All input addend must be integer tensors.")

    if not (is_int(a_zp_sinfo.dtype) and
            is_int(b_zp_sinfo.dtype) and
            is_int(c_zp_sinfo.dtype)):
        raise ValueError("All zero points must be integer tensors.")

    def get_broadcast_shape(shape1, shape2):
           if shape1 is None or shape2 is None:
               return None

           dummy_struct_info1 = tvm.relax.TensorStructInfo(shape1.values, dtype="float32")
           dummy_struct_info2 = tvm.relax.TensorStructInfo(shape2.values, dtype="float32")
           dummy_var1 = tvm.relax.Var("tmp1", dummy_struct_info1)
           dummy_var2 = tvm.relax.Var("tmp2", dummy_struct_info2)
           dummy_add = tvm.relax.op.add(dummy_var1, dummy_var2)

           # We use the internal TVM facilities to implement broadcast semantics
           normalized = ctx.normalize(dummy_add)

           return normalized.struct_info.shape

    # (a_scale * (a - a_zero_point) + b_scale * (b - b_zero_point))/c_scale + c_zero_point
    out_shape1 = get_broadcast_shape(a_sinfo.shape, a_zp_sinfo.shape)
    out_shape1 = get_broadcast_shape(out_shape1, a_scale_sinfo.shape)
    out_shape2 = get_broadcast_shape(b_sinfo.shape, b_zp_sinfo.shape)
    out_shape2 = get_broadcast_shape(out_shape2, b_scale_sinfo.shape)
    out_shape = get_broadcast_shape(out_shape1, out_shape2)
    out_shape = get_broadcast_shape(out_shape, c_scale_sinfo.shape)
    out_shape = get_broadcast_shape(out_shape, c_zp_sinfo.shape)

    # TODO: this is best effort and the resulting type should be calculated or
    # constrains should be added to ensure correctness. Same for vdevice
    out_dtype = a_sinfo.dtype
    out_vdevice = a_sinfo.vdevice

    # TODO: I have no idea if this is correct.
    if out_shape is None:
        out_ndim = max(s.ndim for s in sinfo) if all(s.ndim >= 0 for s in sinfo) else -1
        return tvm.relax.TensorStructInfo(dtype=out_dtype, ndim=out_ndim, vdevice=out_vdevice)

    return tvm.relax.TensorStructInfo(out_shape, dtype=out_dtype, vdevice=out_vdevice)

tvm.ir.register_op_attr("relax.qnn.add", "FPurity", True)
tvm.ir.register_op_attr("relax.qnn.add", "FInferStructInfo", infer_struct_info_qnn_add_op)
qnn_add_op = tvm.ir.Op.get("relax.qnn.add")
qnn_add_op.set_num_inputs(8)
qnn_add_op.add_argument("a", "Tensor", "LHS addend.")
qnn_add_op.add_argument("a_scale", "Tensor", "Scale of the LHS addend.")
qnn_add_op.add_argument("a_zero_point", "Tensor", "Zero point of the LHS addend.")
qnn_add_op.add_argument("b", "Tensor", "RHS addend.")
qnn_add_op.add_argument("b_scale", "Tensor", "Scale of the RHS addend.")
qnn_add_op.add_argument("b_zero_point", "Tensor", "Zero point of the RHS addend.")
qnn_add_op.add_argument("c_scale", "Tensor", "Scale of the result.")
qnn_add_op.add_argument("c_zero_point", "Tensor", "Zero point of the result.")

def qnn_add(
    a: tvm.relax.Expr, s_a: tvm.relax.Expr, z_a: tvm.relax.Expr,
    b: tvm.relax.Expr, s_b: tvm.relax.Expr, z_b: tvm.relax.Expr,
    s_c: tvm.relax.Expr, z_c: tvm.relax.Expr,
) -> tvm.relax.Call:
    op = tvm.ir.Op.get("relax.qnn.add")
    return tvm.relax.Call(op, (a, s_a, z_a, b, s_b, z_b, s_c, z_c))

tvm.script.relax.qnn.add = qnn_add

def infer_struct_info_qnn_conv2d_op(call: tvm.relax.Call, ctx: tvm.relax.BlockBuilder) -> tvm.relax.StructInfo:
    if len(call.args) not in (8, 9):
        raise ValueError("relax.qnn.conv2d expects either 8 or 9 arguments.")

    sinfo =[]
    for i, arg in enumerate(call.args):
        if not isinstance(arg.struct_info, tvm.relax.TensorStructInfo):
            raise ValueError(f"Argument {i} must be a Tensor.")
        sinfo.append(arg.struct_info)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    w_sinfo = sinfo[3]
    w_scale_sinfo = sinfo[4]
    w_zp_sinfo = sinfo[5]
    y_scale_sinfo = sinfo[6]
    y_zp_sinfo = sinfo[7]
    b_sinfo = sinfo[8] if len(call.args) == 9 else None

    def is_int(dtype):
        return dtype.startswith("int") or dtype.startswith("uint")

    if not (x_scale_sinfo.dtype.startswith("float") and
            w_scale_sinfo.dtype.startswith("float") and
            y_scale_sinfo.dtype.startswith("float")):
        raise ValueError("All scales must be float tensors.")

    if not (is_int(x_sinfo.dtype) and is_int(w_sinfo.dtype)):
        raise ValueError("Input and weight must be integer tensors.")

    if b_sinfo and not is_int(b_sinfo.dtype):
        raise ValueError("Bias must be an integer tensors.")

    if not (is_int(x_zp_sinfo.dtype) and
            is_int(w_zp_sinfo.dtype) and
            is_int(y_zp_sinfo.dtype)):
        raise ValueError("All zero points must be integer tensors.")

    out_shape = None
    if x_sinfo.shape is not None and w_sinfo.shape is not None:
        dummy_x_sinfo = tvm.relax.TensorStructInfo(x_sinfo.shape, dtype="float32")
        dummy_w_sinfo = tvm.relax.TensorStructInfo(w_sinfo.shape, dtype="float32")
        dummy_x = tvm.relax.Var("tmp_x", dummy_x_sinfo)
        dummy_w = tvm.relax.Var("tmp_w", dummy_w_sinfo)

        kwargs = {}
        if call.attrs is not None:
            for k, v in call.attrs.items():
                kwargs[k] = v

        dummy_conv = tvm.relax.op.nn.conv2d(dummy_x, dummy_w, **kwargs)
        normalized = ctx.normalize(dummy_conv)
        out_shape = normalized.struct_info.shape

    # https://onnx.ai/onnx/operators/onnx__QLinearConv.html#qlinearconv-10
    # In ONNX y_zero_point has the same type as y hence we use it to determine
    # the output type.
    out_dtype = y_zp_sinfo.dtype
    out_vdevice = x_sinfo.vdevice

    # TODO: I have no idea if this is correct.
    if out_shape is None:
        # Best effort ndim deduction if dynamic shape is heavily missing
        out_ndim = x_sinfo.ndim if x_sinfo.ndim >= 0 else -1
        return tvm.relax.TensorStructInfo(dtype=out_dtype, ndim=out_ndim, vdevice=out_vdevice)

    return tvm.relax.TensorStructInfo(out_shape, dtype=out_dtype, vdevice=out_vdevice)

tvm.ir.register_op_attr("relax.qnn.conv2d", "FPurity", True)
tvm.ir.register_op_attr("relax.qnn.conv2d", "FInferStructInfo", infer_struct_info_qnn_conv2d_op)

qnn_conv2d_op = tvm.ir.Op.get("relax.qnn.conv2d")
qnn_conv2d_op.set_num_inputs(9)
qnn_conv2d_op.add_argument("x", "Tensor", "Input tensor.")
qnn_conv2d_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_conv2d_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_conv2d_op.add_argument("w", "Tensor", "Weight tensor.")
qnn_conv2d_op.add_argument("w_scale", "Tensor", "Scale of the weight.")
qnn_conv2d_op.add_argument("w_zero_point", "Tensor", "Zero point of the weight.")
qnn_conv2d_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_conv2d_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")
qnn_conv2d_op.add_argument("B", "Optional[Tensor]", "Optional bias tensor.")

def qnn_conv2d(
    x: tvm.relax.Expr, x_scale: tvm.relax.Expr, x_zero_point: tvm.relax.Expr,
    w: tvm.relax.Expr, w_scale: tvm.relax.Expr, w_zero_point: tvm.relax.Expr,
    y_scale: tvm.relax.Expr, y_zero_point: tvm.relax.Expr,
    B: tvm.relax.Expr = None,
    **kwargs
) -> tvm.relax.Call:

    op = tvm.ir.Op.get("relax.qnn.conv2d")
    args =[x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point]
    if B is not None:
        args.append(B)

    # NOTE: maybe we should use tvm.relax.op.op_attrs.Conv2DAttrs
    attrs = tvm.ir.make_node("DictAttrs", **kwargs) if kwargs else None

    return tvm.relax.Call(op, tuple(args), attrs=attrs)

tvm.script.relax.qnn.conv2d = qnn_conv2d

# Register key ops
tvm.ir.register_op_attr("tir.vta.coproc_sync", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.coproc_sync", "TScriptPrinterName", "tir.vta.coproc_sync")

tvm.ir.register_op_attr("tir.vta.coproc_dep_push", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.coproc_dep_push", "TScriptPrinterName", "tir.vta.coproc_dep_push")

tvm.ir.register_op_attr("tir.vta.coproc_dep_pop", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.coproc_dep_pop", "TScriptPrinterName", "tir.vta.coproc_dep_pop")

tvm.ir.register_op_attr("tir.vta.uop_push", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.uop_push", "TGlobalSymbol", "VTAUopPush")
tvm.ir.register_op_attr("tir.vta.uop_push", "TScriptPrinterName", "tir.vta.uop_push")

tvm.ir.register_op_attr("tir.vta.command_handle", "TGlobalSymbol", "VTATLSCommandHandle")
tvm.ir.register_op_attr("tir.vta.command_handle", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.command_handle", "TScriptPrinterName", "tir.vta.command_handle")

tvm.ir.register_op_attr("tir.vta.coproc_read_barrier", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.coproc_read_barrier", "TScriptPrinterName", "tir.vta.coproc_read_barrier")

tvm.ir.register_op_attr("tir.vta.coproc_write_barrier", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque)
tvm.ir.register_op_attr("tir.vta.coproc_write_barrier", "TScriptPrinterName", "tir.vta.coproc_write_barrier")

# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.inp_scope)
def mem_info_inp_buffer():
    spec = get_env()
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=spec.INP_ELEM_BITS,
        max_simd_bits=spec.INP_ELEM_BITS,
        max_num_bits=spec.INP_BUFF_SIZE * 8,
        head_address=None,
    )


@tvm.register_func("tvm.info.mem.%s" % Environment.wgt_scope)
def mem_info_wgt_buffer():
    spec = get_env()
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=spec.WGT_ELEM_BITS,
        max_simd_bits=spec.WGT_ELEM_BITS,
        max_num_bits=spec.WGT_BUFF_SIZE * 8,
        head_address=None,
    )


@tvm.register_func("tvm.info.mem.%s" % Environment.acc_scope)
def mem_info_acc_buffer():
    spec = get_env()
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=spec.ACC_ELEM_BITS,
        max_simd_bits=spec.ACC_ELEM_BITS,
        max_num_bits=spec.ACC_BUFF_SIZE * 8,
        head_address=None,
    )


# TVM Op related registration
@tvm.ir.register_intrin_lowering("tir.vta.coproc_sync", "default")
def coproc_sync(op):
    _ = op
    return tvm.tir.call_extern(
        "int32",
        "VTASynchronize",
        get_env().dev.command_handle,
        tvm.runtime.const(1 << 31, dtype="uint32"),
    )


@tvm.ir.register_intrin_lowering("tir.vta.coproc_dep_push", "default")
def coproc_dep_push(op):
    return tvm.tir.call_extern(
        "int32", "VTADepPush", get_env().dev.command_handle, op.args[0], op.args[1]
    )


@tvm.ir.register_intrin_lowering("tir.vta.coproc_dep_pop", "default")
def coproc_dep_pop(op):
    return tvm.tir.call_extern(
        "int32", "VTADepPop", get_env().dev.command_handle, op.args[0], op.args[1]
    )
