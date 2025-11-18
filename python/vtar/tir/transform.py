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
"""Additional Transformation Passes. for VTA"""
# pylint: disable=len-as-condition, no-else-return, unused-argument, invalid-name
import tvm
from tvm import te
from tvm.topi import utils
from tvm.script import tir as T

from ..environment import get_env


def _match_pragma(stmt, key):
    """Internal helper to match stmt to pragma stmt.

    Parameters
    ----------
    stmt : Stmt
        The AttrStmt

    key : str
        The pragma key
    """
    return key in stmt.annotations
    return (stmt.attr_key == "pragma_" + key) or (
        stmt.attr_key == "pragma_scope" and stmt.value.value == key
    )


"""
NOTE: this may be wrong...
Let a[I][J][K][L] be an array
BaseAddress + ( (i * I * J * K) +
                (j * J * K) +
                (k * K) +
                (l * 1) )
            * ElementSize
\sum_i index_i * \prod_{j=i}^{4-1} dim_j
"""

from tvm import tir, ir
from typing import List, Tuple

def prod(iterable, /, start=1):
    res = start
    for element in iterable:
        res *= element
    return res

def get_strides(buf: tir.Buffer) -> List[tir.PrimExpr]:
    if buf.strides:
        res = buf.strides
    else:
        start = tir.IntImm("int32", 1)
        res = [prod(buf.shape[i+1:], start=start) for i in range(len(buf.shape))]
    return res

def _check_compact(buf: tir.Buffer):
    """By compact they mean that the strides are exactly the cumprod of the
    shape dimensions. If the strides were greater the cumprod than the buffer
    is a slice of a bigger (in number of elements) contigous (or compact) memory
    allocation. According to the documentation in buffer.h if buf.strides is
    empty the buffer is contigous"""
    ndim = len(buf.shape)
    print(buf.shape[0].dtype)
    size = tir.const(1, buf.shape[0].dtype)
    strides = get_strides(buf)
    for i in reversed(range(ndim)):
        if not utils.equal_const_int(size - strides[i], 0):
            raise RuntimeError(
                "Cannot prove compact: shape=%s, strides=%s" % (buf.shape, strides)
            )
        size = size * buf.shape[i]

def _fold_buffer_dim(buf: tir.Buffer, scope: str, elem_block: int) -> Tuple[List[int], List[int]]:
    """
    scope: only used for error reporting
    """
    ndim = len(buf.shape)
    buf_strides = get_strides(buf)
    x_size = 1
    base = 0
    for i in range(1, ndim + 1):
        if not utils.equal_const_int(buf_strides[ndim - i] - x_size, 0):
            raise RuntimeError("scope %s needs to have block=%d" % (scope, elem_block))
        x_size = x_size * buf.shape[ndim - i]
        if utils.equal_const_int(x_size - elem_block, 0):
            base = i + 1
            break
    if base == 0:
        raise RuntimeError(
            "scope %s need to have block=%d, shape=%s" % (scope, elem_block, buf.shape)
        )
    shape = [elem_block]
    strides = [1]

    if base < ndim + 1 and not utils.equal_const_int(buf_strides[ndim - base], elem_block):
        shape.append(1)
        strides.append(elem_block)

    analyzer = tvm.arith.Analyzer()
    while base < ndim + 1:
        x_size = 1
        x_stride = buf_strides[ndim - base]
        next_base = base
        if not utils.equal_const_int(tir.indexmod(x_stride, elem_block), 0):
            raise RuntimeError(
                "scope %s need to have block=%d, shape=%s, strides=%s"
                % (scope, elem_block, buf.shape, buf_strides)
            )
        for i in range(base, ndim + 1):
            k = ndim - i
            if not utils.equal_const_int(x_size * x_stride - buf_strides[k], 0):
                break
            x_size = x_size * buf.shape[k]
            next_base = i + 1
        shape.append(analyzer.simplify(x_size))
        strides.append(x_stride)
        assert next_base != base
        base = next_base

    strides = list(reversed(strides))
    shape = list(reversed(shape))
    return shape, strides

def _get_2d_pattern(buf: tir.Buffer, elem_width: int, elem_bytes: int, dtype: str, scope: str, allow_fold: bool) -> Tuple[int, int, int, int]:
    """
    elem_width: use only to calculate elem_block = (OUT|ACC|INP|WGT)_WIDTH i.e. int8 or int32
    elem_bytes: use only to calculate elem_block = (OUT|ACC|INP|WGT)_ELEM_BYTES i.e. number of bytes in the matrix (OUT|ACC|INP|WGT)
    dtype: can be ignored
    scope: is used just for error reporting
    allow_fold: is false only when performing a load with padding
    """
    elem_bits = elem_bytes * 8
    elem_block = elem_bits // elem_width # number of element in a matrix (OUT|ACC|INP|WGT)
    shape, strides = buf.shape, buf.strides

    # breakpoint()
    if not utils.equal_const_int(tir.indexmod(buf.elem_offset, elem_block), 0):
        raise RuntimeError("scope %s need to have block=%d" % (scope, elem_block))

    if allow_fold:
        shape, strides = _fold_buffer_dim(buf, scope, elem_block)
    else:
        shape = list(x for x in shape)
        strides = list(x for x in strides)

    def raise_error():
        raise RuntimeError(
            (
                "Scope[%s]: cannot detect 2d pattern with elem_block=%d:"
                + " shape=%s, strides=%s"
            )
            % (scope, elem_block, buf.shape, buf.strides)
        )

    ndim = len(shape)

    # Check if the inner-tensor is already flat
    flat = utils.equal_const_int(shape[-1], elem_block)

    if flat:
        if not utils.equal_const_int(strides[-1], 1):
            raise_error()

        if ndim == 1:
            x_size = 1
            x_stride = 1
            y_size = 1
            return x_size, y_size, x_stride, tir.indexdiv(buf.elem_offset, elem_block)

        if not utils.equal_const_int(strides[-2] - elem_block, 0):
            raise_error()

        if ndim == 2:
            x_size = shape[-2]
            x_stride = shape[-2]
            y_size = 1
            return x_size, y_size, x_stride, tir.indexdiv(buf.elem_offset, elem_block)

        if not utils.equal_const_int(tir.indexmod(strides[-3], elem_block), 0):
            raise_error()

        if ndim == 3:
            x_size = shape[-2]
            x_stride = tir.indexdiv(strides[-3], elem_block)
            y_size = shape[-3]
            return x_size, y_size, x_stride, tir.indexdiv(buf.elem_offset, elem_block)
    else:
        if not utils.equal_const_int(strides[-1], 1):
            raise_error()
        if not utils.equal_const_int(strides[-2] - shape[-1], 0):
            raise_error()
        if not utils.equal_const_int(shape[-1] * shape[-2], elem_block):
            raise_error()

        if ndim == 2:
            x_size = 1
            x_stride = 1
            y_size = 1
            return x_size, y_size, x_stride, tir.indexdiv(buf.elem_offset, elem_block)

        if not utils.equal_const_int(strides[-3], elem_block):
            raise_error()

        if ndim == 3:
            x_size = shape[-3]
            x_stride = shape[-3]
            y_size = 1
            return x_size, y_size, x_stride, tir.indexdiv(buf.elem_offset, elem_block)

        if not utils.equal_const_int(tir.indexmod(strides[-4], elem_block), 0):
            raise_error()

        if ndim == 4:
            x_size = shape[-3]
            x_stride = tir.indexdiv(strides[-4], elem_block)
            y_size = shape[-4]
            return x_size, y_size, x_stride, tir.indexdiv(buf.elem_offset, elem_block)

    raise_error()

def do_inject_dma_intin_transform(stmt: tir.Stmt) -> tir.Stmt | None:
    """
    This function tries to match for a computation like this

    for i0, i1, i2, i3 in T.grid(o, m, BATCH, BLOCK_OUT):
        with T.block("D"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(D_buf[v_i0, v_i1, v_i2, v_i3])
            T.writes(D[v_i0, v_i1, v_i2, v_i3])
            D[v_i0, v_i1, v_i2, v_i3] = T.Cast("int8", D_buf[v_i0, v_i1, v_i2, v_i3])

    T.call_extern("int32", "VTALoadBuffer2D",
    src_dram_addr=A_2, src_offset=0, x_size=m, y_size=o, x_stride=m, pad=(0, 0, 0, 0), dst_sram_index=0, 3)

    An if expression with a condition based on an index is padding in TIR

    for i in range(140):
        with T.block("block"):
            vi = T.axis.remap("S", [i])
            y[vi] = T.if_then_else(vi >= 6 and vi < 134, x[vi - 6], 0, dtype="int32")
    """
    env = get_env()

    if _match_pragma(stmt, "dma_copy"):
        innermost_loop_body = stmt
        # This two list should be the ones in `for indices in T.grid(*extents)`
        indices: List[tir.IntImm] = []
        extents: List[tir.Var] = []
        while isinstance(innermost_loop_body, tvm.tir.For):
            indices.append(innermost_loop_body.loop_var)
            extents.append(innermost_loop_body.extent)
            innermost_loop_body = innermost_loop_body.body

        reads = innermost_loop_body.block.reads
        writes = innermost_loop_body.block.writes
        if len(reads) != 1:
            raise ValueError("Can only load from one buffer at the time not %s" % reads)
        if len(writes) != 1:
            raise ValueError("Can only write to one buffer at the time not %s" % writes)
        src = reads[0].buffer
        dst = writes[0].buffer
        expr = innermost_loop_body.block.body.value

        if  isinstance(expr, tir.IfThenElse):
            raise ValueError("Padding not yet supported by the compiler")
        else:
            pad_before = []
            pad_after = []
            pad_value = 0

        if dst.scope() == "global": # Store
            if pad_before or pad_after:
                raise RuntimeError("Do not support copy into DRAM with pad")
            if src.scope() == env.acc_scope:
                elem_width = env.OUT_WIDTH
                elem_bytes = env.OUT_ELEM_BYTES
                mem_type = env.dev.MEM_ID_OUT
                data_type = "int%d" % env.OUT_WIDTH
                task_qid = env.dev.QID_STORE_OUT
            else:
                raise RuntimeError("Do not support copy %s->dram" % (src.scope()))
            _check_compact(src)
            x_size, y_size, x_stride, offset = _get_2d_pattern(
                dst, elem_width, elem_bytes, data_type, src.scope(), allow_fold=True
            )
            irb = tvm.tir.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope", env.dev.get_task_qid(task_qid))
            irb.emit(
                tvm.tir.call_extern(
                    "int32",
                    "VTAStoreBuffer2D",
                    env.dev.command_handle,
                    src.access_ptr("r", "int32"),
                    mem_type,
                    dst.data,
                    offset,
                    x_size,
                    y_size,
                    x_stride,
                )
            )
            return irb.get()
        elif src.scope() == "global": # Load
            if dst.scope() == env.acc_scope:
                elem_width = env.ACC_WIDTH
                elem_bytes = env.ACC_ELEM_BYTES
                mem_type = env.dev.MEM_ID_ACC
                data_type = "int%d" % env.ACC_WIDTH
                task_qid = env.dev.QID_LOAD_OUT
            elif dst.scope() == env.inp_scope:
                elem_width = env.INP_WIDTH
                elem_bytes = env.INP_ELEM_BYTES
                mem_type = env.dev.MEM_ID_INP
                data_type = "int%d" % env.INP_WIDTH
                task_qid = env.dev.QID_LOAD_INP
            elif dst.scope() == env.wgt_scope:
                elem_width = env.WGT_WIDTH
                elem_bytes = env.WGT_ELEM_BYTES
                mem_type = env.dev.MEM_ID_WGT
                data_type = "int%d" % env.WGT_WIDTH
                task_qid = env.dev.QID_LOAD_WGT
            else:
                raise RuntimeError("Do not support copy dram->%s" % (dst.scope()))
            # collect pad statistics
            if pad_before:
                assert pad_after
                ndim = len(pad_before)
                if ndim <= 2 or ndim > 5:
                    raise ValueError("Limitation of 2D pad load forbid ndim=%d" % ndim)
                if ndim == 5:
                    # This case occurs when batch size N > 1
                    y_pad_before = pad_before[1]
                    x_pad_before = pad_before[2]
                    y_pad_after = pad_after[1]
                    x_pad_after = pad_after[2]
                    for dim in range(3, ndim):
                        if not utils.equal_const_int(pad_before[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                        if not utils.equal_const_int(pad_after[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                else:
                    y_pad_before = pad_before[0]
                    x_pad_before = pad_before[1]
                    y_pad_after = pad_after[0]
                    x_pad_after = pad_after[1]
                    for dim in range(2, ndim):
                        if not utils.equal_const_int(pad_before[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                        if not utils.equal_const_int(pad_after[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                allow_fold = False
            else:
                x_pad_before = 0
                y_pad_before = 0
                x_pad_after = 0
                y_pad_after = 0
                allow_fold = True

            _check_compact(dst)
            x_size, y_size, x_stride, offset = _get_2d_pattern(
                src, elem_width, elem_bytes, data_type, dst.scope(), allow_fold=allow_fold
            )

            if data_type != src.dtype:
                if not (data_type == "int%d" % env.ACC_WIDTH and src.dtype == "int%d" % env.INP_WIDTH):
                    raise ValueError("Bad data types for load")
                mem_type = env.dev.MEM_ID_ACC_8BIT

            irb = tvm.tir.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope", env.dev.get_task_qid(task_qid))
            # This is needed because otherwise the InplaceOpVerifier of the
            # StorageRewtire transform wrongly determines that the load
            # operation can be performed in-place i.e. the second load
            # overwrites the first.
            irb.scope_attr(env.dev.vta_axis, "extern_scope", True)

            irb.emit(
                tvm.tir.call_extern(
                    "int32",
                    "VTALoadBuffer2D",
                    env.dev.command_handle,
                    src.data,
                    offset,
                    x_size,
                    y_size,
                    x_stride,
                    x_pad_before,
                    y_pad_before,
                    x_pad_after,
                    y_pad_after,
                    dst.access_ptr("w", "int32"),
                    mem_type,
                )
            )
            return irb.get()

        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))
    return None

# The original code used this function which searched for the dma_copy pragma.
# https://mlc.ai/docs/reference/api/tir/transform.html#tvm.tir.transform.InjectCopyIntrin

def inject_dma_intin_transform(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, None, do_inject_dma_intin_transform, ["tir.For"])
    )

# TODO: implement this using tir.PyStmtExprMutator
def InjectDMAIntrin() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        inject_dma_intin_transform, opt_level=0, name="tir.vta.InjectDMAIntrin"
    )

def do_inject_alu_intin_transform(stmt: tir.Stmt) -> tir.Stmt | None:
    """
    This function tries to match for a computation like this

    for i0, i1, i2, i3 in T.grid(o, m, BATCH, BLOCK_OUT):
        with T.block("C_buf"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A_buf[v_i0, v_i1, v_i2, v_i3], B_buf[v_i0, v_i1, v_i2, v_i3])
            T.writes(C_buf[v_i0, v_i1, v_i2, v_i3])
            C_buf[v_i0, v_i1, v_i2, v_i3] = A_buf[v_i0, v_i1, v_i2, v_i3] \
                + B_buf[v_i0, v_i1, v_i2, v_i3]

    where o = O//BATCH and m = M//BLOCK_OUT, and turn it into

    T.call_extern("int32", "VTAUopLoopBegin", extent=o, dst_factor=1, src_factor=1, wgt_factor=0)
    T.call_extern("int32", "VTAUopLoopBegin", extent=m, dst_factor=1, src_factor=1, wgt_factor=0)
    T.tir.vta.uop_push(mode=1, rst_out=0, dst_idx=0, src_idx=m, wgt_idx=0, opcode=2, use_imm=0, imm=0)
    T.call_extern("int32", "VTAUopLoopEnd")
    T.call_extern("int32", "VTAUopLoopEnd")

    If the extents are 1 the VTAUopLoopBegin and respective VTAUopLoopEnd can be
    omitted.
    """
    env = get_env()
    analyzer = tvm.arith.Analyzer()

    res = None
    if _match_pragma(stmt, "alu"):

        innermost_loop_body = stmt
        # This two list should be the ones in `for indices in T.grid(*extents)`
        indices: List[tir.IntImm] = []
        extents: List[tir.Var] = []
        while isinstance(innermost_loop_body, tvm.tir.For):
            indices.append(innermost_loop_body.loop_var)
            extents.append(innermost_loop_body.extent)
            innermost_loop_body = innermost_loop_body.body

        # dst_var[*dst_idx] = value
        # where values is lhs op rhs
        # and dst_index is dst_idx = T.axis.remap("SSS", indices)
        S = 0
        if not all([iter_var.iter_type == S for iter_var in innermost_loop_body.block.iter_vars]):
            raise ValueError("Axis should be all spatially remapped")
        dst_var: tir.Var = innermost_loop_body.block.body.buffer.data
        dst_idx = innermost_loop_body.block.body.indices
        value = innermost_loop_body.block.body.value

        if isinstance(value, tir.expr.BinaryOpExpr):
            lhs = value.a
            rhs = value.b
            if   isinstance(value, tir.Add): alu_opcode = env.dev.ALU_OPCODE_ADD
            elif isinstance(value, tir.Sub): alu_opcode = env.dev.ALU_OPCODE_SUB
            elif isinstance(value, tir.Mul): alu_opcode = env.dev.ALU_OPCODE_MUL
            elif isinstance(value, tir.Min): alu_opcode = env.dev.ALU_OPCODE_MIN
            elif isinstance(value, tir.Max): alu_opcode = env.dev.ALU_OPCODE_MAX
            else:
                raise RuntimeError("Binary op not supported %s" % value.op.name)
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
                raise RuntimeError(
                    "Function call not recognized %s" % (value.op.name)
                )
        elif isinstance(value, tir.BufferLoad):
            alu_opcode = env.dev.ALU_OPCODE_SHR
            lhs = value
            rhs = tvm.tir.const(0, "int32")
        else:
            raise RuntimeError(
                "Expression not recognized %s, %s, %s"
                % (type(value), str(value), str(stmt))
            )

        dst_coeff = extents
        # Check if lhs/rhs is immediate
        imm_val = None
        if isinstance(rhs, tvm.tir.IntImm):
            src_coeff = lhs.indices
            imm_val = rhs
        if isinstance(lhs, tvm.tir.IntImm):
            src_coeff = rhs.indices
            imm_val = lhs
        if imm_val is None:
            imm_val = 0
            src_lhs_coeff = lhs.indices
            src_rhs_coeff = rhs.indices

            lhs_equal = True
            rhs_equal = True
            for i, coef in enumerate(dst_idx):
                if not tvm.ir.structural_equal(coef, src_lhs_coeff[i]):
                    lhs_equal = False
                if not tvm.ir.structural_equal(coef, src_rhs_coeff[i]):
                    rhs_equal = False

            if not (lhs_equal and rhs_equal):
                raise ValueError("lhs and rhs must have the same indices")
            # NOTE(Diego): the original implementation seems like it did
            # something more general
            src_coeff = src_rhs_coeff
        src_coeff = extents

        # At this point is either lhs op rhs or lhs op imm

        if len(extents) == 4:
            src_coeff = [prod(extents[0:-1]), prod(extents[1:-1]), prod(extents[2:-1]), prod(extents[3:-1])]
            src_coeff = list(reversed(src_coeff))
            dst_coeff = src_coeff
            remove_last_two = slice(0, -2, 1)
            extents = extents[remove_last_two]
            irb = tvm.tir.ir_builder.create()
            for i, extent in enumerate(extents):
                if extent != 1:
                    irb.emit(tvm.tir.call_extern("int32", "VTAUopLoopBegin", extent, dst_coeff[i], src_coeff[i], 0))
            irb.emit(
                tvm.tir.call_intrin(
                    "int32",
                    "tir.vta.uop_push",
                    1, # alu mode
                    0, # do not reset accumulator
                    dst_coeff[-1], # dst_index # FIXME: this is wrong!
                    src_coeff[-1], # src_index
                    0,             # wgt_index
                    alu_opcode,
                    int(bool(imm_val)),
                    imm_val,
                )
            )
            for extent in extents:
                if extent != 1:
                    irb.emit(tvm.tir.call_extern("int32", "VTAUopLoopEnd"))
            res = irb.get()
        else:
            raise ValueError("Only quadruply nested loops")

        # ExceptionGroup("The optimization could not be performed because...", execs)
    return res

def inject_alu_intin_transform(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, None, do_inject_alu_intin_transform, ["tir.For"])
    )

# TODO: implement this using tir.PyStmtExprMutator
def InjectALUIntrin() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        inject_alu_intin_transform, opt_level=0, name="tir.vta.InjectALUIntrin"
    )

# TODO: make this behave as any other pass.
@tvm.tir.transform.prim_func_pass(opt_level=0)
def InjectDebug(f, *_):
    env = get_env()
    debug_flag = 1
    debug = tir.call_extern("int32", "VTASetDebugMode", env.dev.command_handle, debug_flag)

    return f.with_body(tir.stmt_seq(debug, f.body))

from tvm import ir

# TODO: make this behave as any other pass.
@tvm.tir.transform.prim_func_pass(opt_level=0)
def InjectDeclBuffer(f, *_):
    alloc_buffers = f.body.block.alloc_buffers
    breakpoint()
    match_buffers = []
    for alloc_buffer in alloc_buffers:
        region = [ir.Range(n) for n in alloc_buffer.shape]
        match_buffers.append(tir.MatchBufferRegion(alloc_buffer, tir.BufferRegion(alloc_buffer, region)))
    block = f.body.block
    body = tir.Block(block.iter_vars, block.reads, block.reads, block.name_hint,
        block.body, block.init, block.alloc_buffers, match_buffers,
        block.annotations)
    return f.with_body(body)


def do_annotate_alu_coproc_scope(stmt: tir.Stmt) -> tir.Stmt | None:
    env = get_env()
    res = None
    if _match_pragma(stmt, "alu"):
        irb = tir.ir_builder.create()
        irb.scope_attr(
            env.dev.vta_axis, "coproc_scope", env.dev.get_task_qid(env.dev.QID_COMPUTE)
        )
        irb.scope_attr(
            env.dev.vta_axis, "coproc_uop_scope", tir.StringImm("VTAPushALUOp")
        )
        irb.emit(stmt)
        res = irb.get()
    if _match_pragma(stmt, "skip_alu"):
        res = tir.Evaluate(0)
    return res

def annotate_alu_coproc_scope(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, None, do_annotate_alu_coproc_scope, ["tir.For"])
    )

# TODO: implement this using tir.PyStmtExprMutator
def AnnotateALUCoProcScope() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        annotate_alu_coproc_scope, opt_level=0, name="tir.vta.AnnotateALUCoProcScope"
    )

def do_inject_coproc_sync(stmt: tir.Stmt) -> tir.Stmt | None:
    # if _match_pragma(stmt, "coproc_sync"):
    if stmt.attr_key == "coproc_sync":
        sync = tir.Call("int32", "tir.vta.coproc_sync", [])
        body = tir.SeqStmt([stmt.body, tir.Evaluate(sync)])
        return body

    if False and _match_pragma(stmt, "trim_loop"):
        op = stmt.body
        if not isinstance(op, tir.For): raise ValueError("Not an instance of tir.For")
        return tvm.tir.For(
            op.loop_var, op.min, 2, op.kind, op.body, op.thread_binding, op.annotations
        )
    return None

def inject_coproc_sync(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, None, do_inject_coproc_sync, ["tir.AttrStmt"])
    )

# TODO: implement this using tir.PyStmtExprMutator
def InjectCoProcSync() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        inject_coproc_sync, opt_level=0, name="tir.vta.InjectCoProcSync"
    )

from tvm.tir.transform import _ffi_api

def LiftAttrScope(s: str):
    return _ffi_api.LiftAttrScope(s)

def CoProcSync():
    return _ffi_api.CoProcSync()


def lift_alloc_to_scope_begin(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    lift_stmt = [[]]

    def _merge_block(slist, body):
        for op in slist:
            if op.body == body:
                body = op
            elif isinstance(op, tvm.tir.Allocate):
                body = tvm.tir.Allocate(op.buffer_var, op.dtype, op.extents, op.condition, body)
            elif isinstance(op, tvm.tir.AttrStmt):
                body = tvm.tir.AttrStmt(op.node, op.attr_key, op.value, body)
            elif isinstance(op, tvm.tir.For):
                body = tvm.tir.For(
                    op.loop_var,
                    op.min,
                    op.extent,
                    op.kind,
                    body,
                    op.thread_binding,
                    op.annotations,
                )
            else:
                raise RuntimeError("unexpected op")
        del slist[:]
        return body

    def do_lift_alloc_to_scope_begin_pre_order(op: tir.Stmt) -> tir.Stmt | None:
        if isinstance(op, tvm.tir.For):
            lift_stmt.append([])
        elif isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == "virtual_thread":
                lift_stmt.append([])

    def do_lift_alloc_to_scope_begin_post_order(op: tir.Stmt) -> tir.Stmt | None:
        if isinstance(op, tvm.tir.Allocate):
            lift_stmt[-1].append(op)
            return op.body
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == "storage_scope":
                lift_stmt[-1].append(op)
                return op.body
            if op.attr_key == "virtual_thread":
                return _merge_block(lift_stmt.pop() + [op], op.body)
            return op
        if isinstance(op, tvm.tir.For):
            return _merge_block(lift_stmt.pop() + [op], op.body)
        raise RuntimeError("not reached")

    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, do_lift_alloc_to_scope_begin_pre_order, do_lift_alloc_to_scope_begin_post_order, ["tir.Allocate", "tir.AttrStmt", "tir.For"])
    )

# TODO: implement this using tir.PyStmtExprMutator
def LiftAllocToScopeBegin() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        lift_alloc_to_scope_begin, opt_level=0, name="tir.vta.LiftAllocToScopeBegin"
    )

def _fold_outermost_loop(body):
    stmt = body
    if not isinstance(stmt, tvm.tir.For):
        return None, body, None

    loop_var = stmt.loop_var
    gemm_offsets = [None, None, None]
    fail = [False]
    builtin_uop_push = tvm.ir.Op.get("tir.vta.uop_push")

    def _post_order(op):
        assert isinstance(op, tvm.tir.Call)
        base_args = 2
        if op.op.same_as(builtin_uop_push):
            args = []
            args += op.args[:base_args]
            for i in range(3):
                m = tvm.arith.detect_linear_equation(op.args[i + base_args], [loop_var])
                if not m:
                    fail[0] = True
                    return op
                if gemm_offsets[i] is not None:
                    if not tvm.ir.structural_equal(m[0], gemm_offsets[i]):
                        fail[0] = True
                        return op
                    args.append(m[1])
                else:
                    gemm_offsets[i] = m[0]
                    args.append(m[1])
            args += op.args[base_args + 3 :]
            return tvm.tir.call_intrin("int32", builtin_uop_push, *args)
        if op.op.name not in ("tir.vta.command_handle", "tir.tvm_thread_context"):
            raise RuntimeError("unexpected op %s" % op)
        return op

    ret = tvm.tir.stmt_functor.ir_transform(stmt.body, None, _post_order, ["tir.Call"])

    if not fail[0] and all(x is not None for x in gemm_offsets):

        def _visit(op):
            if op.same_as(loop_var):
                fail[0] = True

        tvm.tir.stmt_functor.post_order_visit(ret, _visit)
        if not fail[0]:
            begin = tvm.tir.call_extern("int32", "VTAUopLoopBegin", stmt.extent, *gemm_offsets)
            end = tvm.tir.call_extern("int32", "VTAUopLoopEnd")
            return [begin, ret, end]
    raise ValueError("Failed to fold the GEMM instructions..")


def do_fold_uop_loop(stmt: tir.Stmt) -> tir.Stmt | None:
    """
    with T.attr(T.iter_var(vta, None, "ThreadIndex", "vta"), "coproc_uop_scope", "VTAPushGEMMOp"):
        for I, J in T.grid(N, M):
            T.call_extern("int32", "SomethingCompute", C_local_acc_buffer, C_local_acc_buffer, C_local_acc_buffer)

    becomes

    T.call_extern("int32", "VTAUopLoopBegin", extent=o, dst_factor=1, src_factor=1, wgt_factor=0)
    T.call_extern("int32", "VTAUopLoopBegin", extent=m, dst_factor=1, src_factor=1, wgt_factor=0)
    T.tir.vta.uop_push(mode=1, rst_out=0, dst_idx=0, src_idx=m, wgt_idx=0, opcode=2, use_imm=0, imm=0)
    T.call_extern("int32", "VTAUopLoopEnd")
    T.call_extern("int32", "VTAUopLoopEnd")
    """
    env = get_env()
    if (
        stmt.attr_key == "coproc_uop_scope"
        and isinstance(stmt.value, tvm.tir.StringImm)
        and stmt.value.value == env.dev.vta_push_uop.value
    ):
        body = stmt.body
        begins = []
        ends = []
        try:
            for _ in range(2):
                begin, body, end = _fold_outermost_loop(body)
                if begin is not None:
                    begins.append(begin)
                if end is not None:
                    ends.append(end)
        except ValueError:
            pass
        if body == stmt.body:
            return stmt
        ends = list(reversed(ends))
        body = tvm.tir.stmt_seq(*(begins + [body] + ends))
        return tvm.tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, body)

    return stmt

def fold_uop_loop(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, None, do_fold_uop_loop, ["tir.AttrStmt"])
    )

def FoldUopLoop() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        fold_uop_loop, opt_level=0, name="tir.vta.FoldUopLoop"
    )

"""
BufferStore(tir::Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices, ffi::Optional<PrimExpr> predicate) // Leaf
Evaluate(PrimExpr value) // Leaf
LetStmt(tir::Var var, PrimExpr value, tir::Stmt body)
AttrStmt(ffi::Any node, ffi::String attr_key, PrimExpr value, tir::Stmt body)
AssertStmt(PrimExpr condition, PrimExpr message, tir::Stmt body)
BufferRealize(tir::Buffer buffer, ffi::Array<Range> bounds, PrimExpr condition, tir::Stmt body)
Allocate(tir::Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents, PrimExpr condition, tir::Stmt body, ffi::Map<ffi::String, ffi::Any> annotations)
AllocateConst(tir::Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents, ObjectRef data_or_idx, tir::Stmt body, ffi::Map<ffi::String, ffi::Any> annotations)
DeclBuffer(tir::Buffer, tir::Stmt body)
SeqStmt(ffi::Array<tir::Stmt> seq)
IfThenElse(PrimExpr condition, tir::Stmt then_case, ffi::Optional<tir::Stmt> else_case)
For(tir::Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, tir::Stmt body, ffi::Map<ffi::String, ffi::Any>annotations)
While(PrimExpr condition, tir::Stmt body)
Block(
  ffi::Array<IterVar> iter_vars,
  ffi::Array<BufferRegion> reads,
  ffi::Array<BufferRegion> writes,
  ffi::String name_hint,
  tir::Stmt body,
  ffi::Optional<tir::Stmt> init,
  ffi::Array<tir::Buffer> alloc_buffers,
  ffi::Array<MatchBufferRegion> match_buffers,
  ffi::Map<ffi::String, ffi::Any> annotations
)
BlockRealize(ffi::Array<PrimExpr> iter_values, PrimExpr predicate, Block block)

"""
def visit(stmt, pattern: str):
    if isinstance(stmt, tir.Stmt):
        print(pattern + str(type(stmt)), end=' ')
        bodies = []

        if isinstance(stmt, tir.BufferStore):
            pass
        elif isinstance(stmt, tir.Evaluate):
            # print(stmt.value, end='')
            pass
        elif isinstance(stmt, tir.LetStmt):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.AttrStmt):
            bodies.append(stmt.body)
            print(hash(stmt.node), stmt.node, stmt.attr_key, stmt.value, end='')
        elif isinstance(stmt, tir.AssertStmt):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.BufferRealize):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.Allocate):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.AllocateConst):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.DeclBuffer):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.SeqStmt):
            bodies.extend(stmt.seq)
        elif isinstance(stmt, tir.IfThenElse):
            bodies.append(stmt.then_case)
            if stmt.else_case:
                bodies.append(else_case)
        elif isinstance(stmt, tir.For):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.While):
            bodies.append(stmt.body)
        elif isinstance(stmt, tir.Block):
            bodies.append(stmt.body)
            print(stmt.name_hint, end='')
        elif isinstance(stmt, tir.BlockRealize):
            bodies.append(stmt.block)
        else:
            assert False

        if isinstance(stmt, tir.SeqStmt):
            pattern += "| "
        else:
            pattern += "  "

        print()
        for body in bodies:
            visit(body, pattern)

def print_debug(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    visit(func.body, "")
    return func

def PrintDebug() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        print_debug, opt_level=0, name="tir.vta.PrintDebug"
    )

# For some reason the one in DevContext.vta_axis is created multiple times,
# hence it hashes to different values and tir.transform.CoProcSync does not like
# it.
vta_axis = tvm.te.thread_axis("vta")

def do_replace_vta_var(stmt: tir.Stmt) -> tir.Stmt | None:
    res = stmt
    node = stmt.node
    if isinstance(node, tir.IterVar) and node.var.name == "vta":
        attr_stmt = tir.AttrStmt(vta_axis, stmt.attr_key, stmt.value, stmt.body)
        res = attr_stmt
    return res

def replace_vta_var(func: tir.PrimFunc, mod: ir.IRModule, ctx: ir.transform.PassContext) -> tir.PrimFunc:
    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(func.body, None, do_replace_vta_var, ["tir.AttrStmt"])
    )

def ReplaceVTAVar() -> tir.transform.PrimFuncPass:
    return tir.transform.prim_func_pass(
        replace_vta_var, opt_level=0, name="tir.vta.ReplaceVtaVar"
    )
