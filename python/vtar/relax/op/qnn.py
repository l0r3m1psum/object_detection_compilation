import tvm
from tvm import ir, relax

from typing import Optional, Tuple, Union

def is_int(dtype): return dtype.startswith("int") or dtype.startswith("uint")
def is_float(dtype): return dtype.startswith("float")
def get_tensors_sinfo(args):
    sinfo = []
    for i, arg in enumerate(args):
        if not isinstance(arg.struct_info, relax.TensorStructInfo):
            raise ValueError(f"Argument {i} must be a Tensor.")
        sinfo.append(arg.struct_info)
    return sinfo

def infer_struct_info_qnn_add_op(call: relax.Call, ctx: relax.BlockBuilder) -> relax.StructInfo:
    if len(call.args) != 8:
        raise ValueError("relax.qnn.add expects exactly 8 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    a_sinfo = sinfo[0]
    a_scale_sinfo = sinfo[1]
    a_zp_sinfo = sinfo[2]
    b_sinfo = sinfo[3]
    b_scale_sinfo = sinfo[4]
    b_zp_sinfo = sinfo[5]
    c_scale_sinfo = sinfo[6]
    c_zp_sinfo = sinfo[7]

    if not (is_float(a_scale_sinfo.dtype) and
            is_float(b_scale_sinfo.dtype) and
            is_float(c_scale_sinfo.dtype)):
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

       dummy_struct_info1 = relax.TensorStructInfo(shape1.values, dtype="float32")
       dummy_struct_info2 = relax.TensorStructInfo(shape2.values, dtype="float32")
       dummy_var1 = relax.Var("tmp1", dummy_struct_info1)
       dummy_var2 = relax.Var("tmp2", dummy_struct_info2)
       dummy_add = relax.op.add(dummy_var1, dummy_var2)

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

    # https://github.com/microsoft/onnxruntime/blob/6ee4ea3b05423aaa3ecd3698a56b83eb45f4b2ad/docs/ContribOperators.md#com.microsoft.QLinearAdd
    # We follow the ONNX semantics of the output having the same device as the
    # two quantized input tensors. Same for vdevice.
    out_dtype = a_sinfo.dtype
    out_vdevice = a_sinfo.vdevice

    # TODO: I have no idea if this is correct.
    if out_shape is None:
        out_ndim = max(s.ndim for s in sinfo) if all(s.ndim >= 0 for s in sinfo) else -1
        return relax.TensorStructInfo(dtype=out_dtype, ndim=out_ndim, vdevice=out_vdevice)

    return relax.TensorStructInfo(out_shape, dtype=out_dtype, vdevice=out_vdevice)

ir.register_op_attr("relax.qnn.add", "FPurity", True)
ir.register_op_attr("relax.qnn.add", "FInferStructInfo", infer_struct_info_qnn_add_op)
qnn_add_op = ir.Op.get("relax.qnn.add")
qnn_add_op.set_num_inputs(8)
qnn_add_op.add_argument("a", "Tensor", "LHS addend.")
qnn_add_op.add_argument("a_scale", "Tensor", "Scale of the LHS addend.")
qnn_add_op.add_argument("a_zero_point", "Tensor", "Zero point of the LHS addend.")
qnn_add_op.add_argument("b", "Tensor", "RHS addend.")
qnn_add_op.add_argument("b_scale", "Tensor", "Scale of the RHS addend.")
qnn_add_op.add_argument("b_zero_point", "Tensor", "Zero point of the RHS addend.")
qnn_add_op.add_argument("c_scale", "Tensor", "Scale of the result.")
qnn_add_op.add_argument("c_zero_point", "Tensor", "Zero point of the result.")

def add(
    a: relax.Expr, s_a: relax.Expr, z_a: relax.Expr,
    b: relax.Expr, s_b: relax.Expr, z_b: relax.Expr,
    s_c: relax.Expr, z_c: relax.Expr,
) -> relax.Call:
    op = ir.Op.get("relax.qnn.add")
    return relax.Call(op, (a, s_a, z_a, b, s_b, z_b, s_c, z_c))

def conv2d_attrs_to_dict(attrs: ir.DictAttrs) -> dict:
    strides = [int(s) for s in attrs["strides"]]
    padding = [int(p) for p in attrs["padding"]]
    dilation = [int(d) for d in attrs["dilation"]]
    groups = int(attrs["groups"])
    data_layout = str(attrs["data_layout"])
    kernel_layout = str(attrs["kernel_layout"])
    out_layout = attrs["out_layout"]
    out_layout = str(out_layout) if out_layout else data_layout
    out_dtype = "void"
    res = {
        "strides": strides,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
        "out_layout": out_layout,
        "out_dtype": out_dtype,
    }
    return res

def infer_struct_info_qnn_conv2d_op(call: relax.Call, ctx: relax.BlockBuilder) -> relax.StructInfo:
    if len(call.args) not in (8, 9):
        raise ValueError("relax.qnn.conv2d expects either 8 or 9 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    w_sinfo = sinfo[3]
    w_scale_sinfo = sinfo[4]
    w_zp_sinfo = sinfo[5]
    y_scale_sinfo = sinfo[6]
    y_zp_sinfo = sinfo[7]
    b_sinfo = sinfo[8] if len(call.args) == 9 else None

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(w_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
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
        dummy_x_sinfo = relax.TensorStructInfo(x_sinfo.shape, dtype="float32")
        dummy_w_sinfo = relax.TensorStructInfo(w_sinfo.shape, dtype="float32")
        dummy_b_sinfo = relax.TensorStructInfo(b_sinfo.shape, dtype="float32") if b_sinfo is not None else None
        dummy_x = relax.Var("tmp_x", dummy_x_sinfo)
        dummy_w = relax.Var("tmp_w", dummy_w_sinfo)
        dummy_b = relax.Var("tmp_b", dummy_b_sinfo) if b_sinfo is not None else None

        attrs = conv2d_attrs_to_dict(call.attrs)
        # attrs = ir.make_node("relax.attrs.Conv2DAttrs", **attrs)
        dummy_conv = relax.op.nn.conv2d(dummy_x, dummy_w, **attrs)
        # FIXME: this line is commented out because if the expression in not in
        # ANF (A normal form) ctx.normalize emits a spurious operation in the
        # dataflow graph.
        # dummy_conv = dummy_conv + dummy_b if dummy_b is not None else dummy_conv
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
        return relax.TensorStructInfo(dtype=out_dtype, ndim=out_ndim, vdevice=out_vdevice)

    return relax.TensorStructInfo(out_shape, dtype=out_dtype, vdevice=out_vdevice)

ir.register_op_attr("relax.qnn.conv2d", "FPurity", True)
ir.register_op_attr("relax.qnn.conv2d", "FInferStructInfo", infer_struct_info_qnn_conv2d_op)

qnn_conv2d_op = ir.Op.get("relax.qnn.conv2d")
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

def conv2d(
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    w: relax.Expr, w_scale: relax.Expr, w_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    B: relax.Expr | None = None,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: str | None = None,
) -> relax.Call:

    op = ir.Op.get("relax.qnn.conv2d")
    args = [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point]
    if B is not None:
        args.append(B)

    attrs = ir.make_node(
        "DictAttrs",
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=out_layout,
    )

    return relax.Call(op, args, attrs)

def avg_pool2d_attrs_to_dict(attrs: ir.DictAttrs) -> dict:
    pool_size = [int(s) for s in attrs["pool_size"]]
    strides = [int(s) for s in attrs["strides"]]
    padding = [int(p) for p in attrs["padding"]]
    dilation = [int(d) for d in attrs["dilation"]]
    ceil_mode = int(attrs["ceil_mode"])
    count_include_pad = int(attrs["count_include_pad"])
    layout = str(attrs["layout"])
    out_layout = attrs["out_layout"]
    out_layout = str(out_layout) if out_layout else layout
    res = {
        "pool_size": pool_size,
        "strides": strides,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "layout": layout,
        "out_layout": out_layout,
    }
    return res

def infer_struct_info_qnn_avg_pool2d_op(call: relax.Call, ctx: relax.BlockBuilder) -> relax.StructInfo:
    if len(call.args) != 5:
        raise ValueError("relax.qnn.avg_pool2d expects 5 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    y_scale_sinfo = sinfo[3]
    y_zp_sinfo = sinfo[4]

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
        raise ValueError("All scales must be float tensors.")

    if not is_int(x_sinfo.dtype):
        raise ValueError("The input must be integer tensors.")

    if not (is_int(x_zp_sinfo.dtype) and
            is_int(y_zp_sinfo.dtype)):
        raise ValueError("All zero points must be integer tensors.")

    out_shape = None
    if x_sinfo.shape is not None:
        dummy_x_sinfo = relax.TensorStructInfo(x_sinfo.shape, dtype="float32")
        dummy_x = relax.Var("tmp_x", dummy_x_sinfo)

        attrs = avg_pool2d_attrs_to_dict(call.attrs)
        dummy_conv = relax.op.nn.avg_pool2d(dummy_x, **attrs)
        normalized = ctx.normalize(dummy_conv)
        out_shape = normalized.struct_info.shape

    out_dtype = x_sinfo.dtype
    out_vdevice = x_sinfo.vdevice

    # TODO: add test for symbolic sizes in shapes...

    return relax.TensorStructInfo(out_shape, dtype=out_dtype, vdevice=out_vdevice)

ir.register_op_attr("relax.qnn.avg_pool2d", "FPurity", True)
ir.register_op_attr("relax.qnn.avg_pool2d", "FInferStructInfo", infer_struct_info_qnn_avg_pool2d_op)

qnn_avg_pool2d_op = ir.Op.get("relax.qnn.avg_pool2d")
qnn_avg_pool2d_op.set_num_inputs(5)
qnn_avg_pool2d_op.add_argument("x", "Tensor", "Input tensor.")
qnn_avg_pool2d_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_avg_pool2d_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_avg_pool2d_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_avg_pool2d_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")

def avg_pool2d(
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    pool_size: int | tuple[int, int] = (1, 1),
    strides: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, ...] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    layout: str = 'NCHW',
    out_layout: str | None = None
) -> relax.Call:
    op = ir.Op.get("relax.qnn.avg_pool2d")
    args = (x, x_scale, x_zero_point, y_scale, y_zero_point)
    attrs = ir.make_node(
        "DictAttrs",
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        layout=layout,
        out_layout=out_layout,
    )
    return relax.Call(op, args, attrs)

def infer_struct_info_qnn_linear_op(call: relax.Call, ctx: relax.BlockBuilder) -> relax.StructInfo:
    if len(call.args) not in (8, 9):
        raise ValueError("relax.qnn.linear expects either 8 or 9 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    w_sinfo = sinfo[3]
    w_scale_sinfo = sinfo[4]
    w_zp_sinfo = sinfo[5]
    y_scale_sinfo = sinfo[6]
    y_zp_sinfo = sinfo[7]
    b_sinfo = sinfo[8] if len(call.args) == 9 else None

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(w_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
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
        dummy_x_sinfo = relax.TensorStructInfo(x_sinfo.shape, dtype="float32")
        dummy_w_sinfo = relax.TensorStructInfo(w_sinfo.shape, dtype="float32")
        dummy_b_sinfo = relax.TensorStructInfo(b_sinfo.shape, dtype="float32") if b_sinfo is not None else None
        dummy_x = relax.Var("tmp_x", dummy_x_sinfo)
        dummy_w = relax.Var("tmp_w", dummy_w_sinfo)
        dummy_b = relax.Var("tmp_b", dummy_b_sinfo) if b_sinfo is not None else None

        # relax.on.linear expects x to be transposed...
        attrs = ir.make_node("relax.attrs.MatmulAttrs", out_dtype="void")
        dummy_linear = relax.Call(ir.Op.get("relax.matmul"), (dummy_x, dummy_w), attrs)
        # dummy_linear = dummy_linear + dummy_b if dummy_b is not None else dummy_linear
        normalized = ctx.normalize(dummy_linear)
        out_shape = normalized.struct_info.shape

    # https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html#qlinearmatmul-10
    # In ONNX y_zero_point has the same type as y hence we use it to determine
    # the output type.
    out_dtype = y_zp_sinfo.dtype
    out_vdevice = x_sinfo.vdevice

    # TODO: I have no idea if this is correct.
    if out_shape is None:
        # Best effort ndim deduction if dynamic shape is heavily missing
        out_ndim = x_sinfo.ndim if x_sinfo.ndim >= 0 else -1
        return relax.TensorStructInfo(dtype=out_dtype, ndim=out_ndim, vdevice=out_vdevice)

    return relax.TensorStructInfo(out_shape, dtype=out_dtype, vdevice=out_vdevice)

ir.register_op_attr("relax.qnn.linear", "FPurity", True)
ir.register_op_attr("relax.qnn.linear", "FInferStructInfo", infer_struct_info_qnn_linear_op)

# beta is not present between the parameters because we expect it to be constant
# folded in B.
qnn_linear_op = ir.Op.get("relax.qnn.linear")
qnn_linear_op.set_num_inputs(9)
qnn_linear_op.add_argument("x", "Tensor", "Input tensor.")
qnn_linear_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_linear_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_linear_op.add_argument("w", "Tensor", "Weight tensor.")
qnn_linear_op.add_argument("w_scale", "Tensor", "Scale of the weight.")
qnn_linear_op.add_argument("w_zero_point", "Tensor", "Zero point of the weight.")
qnn_linear_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_linear_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")
qnn_linear_op.add_argument("B", "Optional[Tensor]", "Optional bias tensor.")

def linear(
    a: relax.Expr,
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    w: relax.Expr, w_scale: relax.Expr, w_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    B: relax.Expr | None = None,
) -> relax.Call:
    pass
    op = ir.Op.get("relax.qnn.linear")
    args = [a, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point]
    if B is not None:
        args.append(B)
    return relax.Call(op, tuple(args))
