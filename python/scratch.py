import tvm
from tvm import tir, arith, ir, te, topi, relax, dlight as dl
from tvm.script import tir as T
from tvm.script import ir as I
from tvm.script import relax as R

import shutil
# shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
#     "submodules/tvm-vta/config/vta_config.json")
import vtar
shutil.copy("submodules/tvm-vta/config/fsim_sample.json",
    "submodules/tvm-vta/config/vta_config.json")
import vtar.relax.frontend.onnx
import vtar.relax.transform

import os
from typing import List, Tuple, Dict

import numpy
import onnx
import PIL.Image

def analyze_index_map():
    i = tir.Var("i", "int32")
    j = tir.Var("j", "int32")

    input_iters = {
        i: ir.Range(0, 4),
        j: ir.Range(0, 8)
    }

    # It works in both cases...
    access_index = [i * 8 + j]
    access_index = [i, j]

    print(f"Original Expression: {access_index}")

    result = arith.detect_iter_map(access_index, input_iters)

    if result.indices:
        iter_sum = result.indices[0]

        print(f"Result Type: {type(iter_sum)}")
        print(f"Base (offset): {iter_sum.base}")

        print("Args (components):")
        for arg in iter_sum.args:
            print(f"  Iterator: {arg.source.source}, Scale: {arg.scale}, Extent: {arg.extent}")
        print(f"\nSummary: The expression '{access_index}' successfully maps to a fused iteration space of 0..32.")
    else:
        print("Could not detect a valid affine map.")

N = tir.Var("N", "int32")
M = tir.Var("M", "int32")

@T.prim_func
def row_major(A: T.Buffer((N, M), "float32"), B: T.Buffer((N, M), "float32")):
    for i, j in T.grid(N, M):
        with T.block("block_row"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[2*vi, vj+2] = A[vi, vj]

@T.prim_func
def row_major2(A: T.Buffer((N, M), "float32"), B: T.Buffer((N, M), "float32")):
    for ij in T.grid(N*M):
        with T.block("block_row"):
            vi = T.axis.spatial(N, ij // N)
            vj = T.axis.spatial(M, ij % N)
            B[vi, vj] = A[vi, vj]

def analyze_tensorir_func_automatic(func):
    print(f"--- Analyzing Function: {func.attrs.get('global_symbol', 'unnamed')} ---")

    loop_stack: List[Tuple[tir.Var, ir.Range]] = []
    block_map_stack: List[Tuple[tir.Var, tir.Var]] = []
    buffer_var_map: Dict[tir.Var, tir.Buffer] = {
        buffer.data: buffer
        for param_var, buffer in func.buffer_map.items()
    }

    read_write_regions = {}

    # Helper to print results cleanly
    def print_analysis(buffer_name: str, access_type: str, indices, result):
        print(f"\nAccess: {buffer_name} ({access_type})")
        print(f"  Indices: {indices}")
        if result.indices:
            print("  [✓] Affine Map Detected:")
            for i, iter_sum in enumerate(result.indices):
                args_str = ", ".join(
                    [f"{arg.scale}*{arg.source.source}" for arg in iter_sum.args]
                )
                print(f"      Dim {i}: Base={iter_sum.base}, Args=[{args_str}]")
        else:
            print("  [x] Could not detect valid affine map.")

    # Helper to traverse Expressions (recursively) to find BufferLoads
    # (Since ir_transform only visits Statements)
    def find_and_analyze_loads(expr, input_iters, param_map):
        if isinstance(expr, tir.BufferLoad):
            # Substitute Block Vars -> Loop Vars
            mapped_indices = [
                tir.stmt_functor.substitute(idx, param_map) for idx in expr.indices
            ]
            # Analyze
            res = arith.detect_iter_map(mapped_indices, input_iters)
            print_analysis(expr.buffer.name, "Load", mapped_indices, res)
            return

        # Recursive step for common expression types
        # Note: This is a simplified traversal. TVM expressions are trees.
        if hasattr(expr, "a"): find_and_analyze_loads(expr.a, input_iters, param_map)
        if hasattr(expr, "b"): find_and_analyze_loads(expr.b, input_iters, param_map)
        if hasattr(expr, "value"): find_and_analyze_loads(expr.value, input_iters, param_map)
        if hasattr(expr, "condition"): find_and_analyze_loads(expr.condition, input_iters, param_map)
        if hasattr(expr, "args"):
            for arg in expr.args: find_and_analyze_loads(arg, input_iters, param_map)

    def preorder(stmt):
        if isinstance(stmt, tir.For):
            loop_stack.append((stmt.loop_var, ir.Range(stmt.min, stmt.extent)))
        elif isinstance(stmt, tir.BlockRealize):
            mapping = {
                var.var: binding
                for var, binding in zip(stmt.block.iter_vars, stmt.iter_values)
            }
            block_map_stack.append(mapping)

            ####################################################################

            block = stmt.block
            for alloc_buffer in block.alloc_buffers:
                buffer_var_map[alloc_buffer.data] = alloc_buffer
            for match_buffer in block.match_buffers:
                buffer_var_map[match_buffer.buffer.data] = match_buffer.buffer

            print(f"\n[Block: '{block.name_hint}'] Analysis:")
            rw_regions = tir.analysis.get_block_read_write_region(block, buffer_var_map)
            reads, writes = rw_regions[0], rw_regions[1]
            read_write_regions[block.name_hint] = rw_regions

            print(reads, writes)

        elif isinstance(stmt, tir.BufferStore):

            input_iters = {var: range for var, range in loop_stack}
            param_map = block_map_stack[-1] if block_map_stack else {}

            # B. Analyze the Store (Write)
            mapped_indices = [
                tir.stmt_functor.substitute(idx, param_map) for idx in stmt.indices
            ]
            res = arith.detect_iter_map(mapped_indices, input_iters)
            print_analysis(stmt.buffer.name, "Store", mapped_indices, res)
            # print(arith.normalize_to_iter_sum(mapped_indices[0], input_iters))
            # print(arith.iter_map_simplify(mapped_indices, input_iters))
            # print(arith.normalize_iter_map_to_expr(res.indices[0]))

            # C. Analyze the Value (Reads)
            # The value being stored is an Expression, traverse it to find Loads
            find_and_analyze_loads(stmt.value, input_iters, param_map)

        # print(type(stmt))
        return None

    def postorder(stmt):
        if isinstance(stmt, tir.For):
            loop_stack.pop()
        elif isinstance(stmt, tir.BlockRealize):
            block_map_stack.pop()
        # TODO: make a stack of buffer_var_map
        return stmt

    _ = tir.stmt_functor.ir_transform(func.body, preorder, postorder)

    return read_write_regions

@T.prim_func
def main(
        data: T.Buffer((1, 16, 14, 14, 1, 16), "int8"),
        kernel: T.Buffer((16, 16, 3, 3, 16, 16), "int8"),
        res: T.Buffer((1, 16, 14, 14, 1, 16), "int8")
    ):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    data_buf = T.alloc_buffer((1, 16, 16, 16, 1, 16), "int8")
    res_conv = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    res_shr = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    res_max = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    res_min = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    kernel_local_wgt_buffer = T.alloc_buffer((16, 16, 3, 3, 16, 16), "int8", scope="local.wgt_buffer")
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(16, 16, 3, 3, 16, 16):
        with T.block("kernel_local.wgt_buffer"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(kernel[v0, v1, v2, v3, v4, v5])
            T.writes(kernel_local_wgt_buffer[v0, v1, v2, v3, v4, v5])
            kernel_local_wgt_buffer[v0, v1, v2, v3, v4, v5] = kernel[v0, v1, v2, v3, v4, v5]
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 16, 16, 16, 1, 16):
        with T.block("data_buf"):
            v_i0, v_i1, v_i2, v_i3, v_i4, v_i5 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
            T.reads(data[v_i0, v_i1, v_i2 - 1, v_i3 - 1, v_i4, v_i5])
            T.writes(data_buf[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5])
            data_buf[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5] = T.if_then_else(1 <= v_i2 and v_i2 < 15 and 1 <= v_i3 and v_i3 < 15, data[v_i0, v_i1, v_i2 - 1, v_i3 - 1, v_i4, v_i5], T.int8(0))
    for co_0_1 in T.thread_binding(2, thread="cthread"):
        for co_0_0, bo_0, i_0, j_0 in T.grid(1, 1, 2, 1):
            for ax0_init, ax1_init, ax2_init, ax3_init, ax4_init, ax5_init in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_conv_init"):
                    v_bo = T.axis.spatial(1, ax0_init)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1_init)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2_init)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3_init, ax4_init, ax5_init])
                    T.reads()
                    T.writes(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci] = 0
            for ax6_0, ax0, ax1, ax2, ax6_1, ax7, ax8, ax3, ax4, ax5, ax9 in T.grid(16, 1, 8, 7, 1, 3, 3, 14, 1, 16, 16):
                with T.block("res_conv_update"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    v_ic = T.axis.reduce(16, ax6_0 + ax6_1)
                    v_dy, v_dx, v_ic_tns = T.axis.remap("RRR", [ax7, ax8, ax9])
                    T.reads(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci], data_buf[v_bo, v_ic, v_i + v_dy, v_j + v_dx, v_bi, v_ic_tns], kernel_local_wgt_buffer[v_co, v_ic, v_dy, v_dx, v_ci, v_ic_tns])
                    T.writes(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci] += T.Cast("int32", data_buf[v_bo, v_ic, v_i + v_dy, v_j + v_dx, v_bi, v_ic_tns]) * T.Cast("int32", kernel_local_wgt_buffer[v_co, v_ic, v_dy, v_dx, v_ci, v_ic_tns])
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_shr"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    T.reads(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.shift_right(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci], 8)
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_max"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    T.reads(res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.max(res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci], 0)
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_min"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    T.reads(res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.min(res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci], 127)
            for bo_1, co_1, i_1, j_1, bi, ci in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res"):
                    v_bo = T.axis.spatial(1, bo_0 + bo_1)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + co_1)
                    v_i = T.axis.spatial(14, i_0 * 7 + i_1)
                    v_j = T.axis.spatial(14, j_0 * 14 + j_1)
                    v_bi, v_ci = T.axis.remap("SS", [bi, ci])
                    T.reads(res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.Cast("int8", res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci])

@T.prim_func
def main2(
        data: T.Buffer((1, 16, 14, 14, 1, 16), "int8"),
        kernel: T.Buffer((16, 16, 3, 3, 16, 16), "int8"),
        res: T.Buffer((1, 16, 14, 14, 1, 16), "int8")
    ):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    data_buf = T.alloc_buffer((1, 16, 16, 16, 1, 16), "int8")
    res_conv = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    res_shr = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    res_max = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    res_min = T.alloc_buffer((1, 16, 14, 14, 1, 16), "int32")
    kernel_local_wgt_buffer = T.alloc_buffer((16, 16, 3, 3, 16, 16), "int8", scope="local.wgt_buffer")
    for co_0_1 in T.thread_binding(2, thread="cthread"):
        for co_0_0, bo_0, i_0, j_0 in T.grid(1, 1, 2, 1):
            for ax0_init, ax1_init, ax2_init, ax3_init, ax4_init, ax5_init in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_conv_init"):
                    v_bo = T.axis.spatial(1, ax0_init)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1_init)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2_init)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3_init, ax4_init, ax5_init])
                    T.reads()
                    T.writes(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci] = 0
            for ax6_0 in range(16):
                for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 1, 9, 16, 1, 16):
                    with T.block("data_buf"):
                        v_i0 = T.axis.spatial(1, ax0)
                        v_i1 = T.axis.spatial(16, ax6_0 + ax1)
                        v_i2 = T.axis.spatial(16, i_0 * 7 + ax2)
                        v_i3, v_i4, v_i5 = T.axis.remap("SSS", [ax3, ax4, ax5])
                        T.reads(data[v_i0, v_i1, v_i2 - 1, v_i3 - 1, v_i4, v_i5])
                        T.writes(data_buf[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5])
                        data_buf[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5] = T.if_then_else(1 <= v_i2 and v_i2 < 15 and 1 <= v_i3 and v_i3 < 15, data[v_i0, v_i1, v_i2 - 1, v_i3 - 1, v_i4, v_i5], T.int8(0))
                for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(8, 1, 3, 3, 16, 16):
                    with T.block("kernel_local.wgt_buffer"):
                        v0 = T.axis.spatial(16, co_0_1 * 8 + ax0)
                        v1 = T.axis.spatial(16, ax6_0 + ax1)
                        v2, v3, v4, v5 = T.axis.remap("SSSS", [ax2, ax3, ax4, ax5])
                        T.reads(kernel[v0, v1, v2, v3, v4, v5])
                        T.writes(kernel_local_wgt_buffer[v0, v1, v2, v3, v4, v5])
                        kernel_local_wgt_buffer[v0, v1, v2, v3, v4, v5] = kernel[v0, v1, v2, v3, v4, v5]
                for ax0, ax1, ax2, ax6_1, ax7, ax8, ax3, ax4, ax5, ax9 in T.grid(1, 8, 7, 1, 3, 3, 14, 1, 16, 16):
                    with T.block("res_conv_update"):
                        v_bo = T.axis.spatial(1, ax0)
                        v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                        v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                        v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                        v_ic = T.axis.reduce(16, ax6_0 + ax6_1)
                        v_dy, v_dx, v_ic_tns = T.axis.remap("RRR", [ax7, ax8, ax9])
                        T.reads(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci], data_buf[v_bo, v_ic, v_i + v_dy, v_j + v_dx, v_bi, v_ic_tns], kernel_local_wgt_buffer[v_co, v_ic, v_dy, v_dx, v_ci, v_ic_tns])
                        T.writes(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                        res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci] += \
                            T.Cast("int32", data_buf[v_bo, v_ic, v_i + v_dy, v_j + v_dx, v_bi, v_ic_tns]) * T.Cast("int32", kernel_local_wgt_buffer[v_co, v_ic, v_dy, v_dx, v_ci, v_ic_tns])
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_shr"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    T.reads(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.shift_right(res_conv[v_bo, v_co, v_i, v_j, v_bi, v_ci], 8)
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_max"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    T.reads(res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.max(res_shr[v_bo, v_co, v_i, v_j, v_bi, v_ci], 0)
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res_min"):
                    v_bo = T.axis.spatial(1, ax0)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + ax1)
                    v_i = T.axis.spatial(14, i_0 * 7 + ax2)
                    v_j, v_bi, v_ci = T.axis.remap("SSS", [ax3, ax4, ax5])
                    T.reads(res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.min(res_max[v_bo, v_co, v_i, v_j, v_bi, v_ci], 127)
            for bo_1, co_1, i_1, j_1, bi, ci in T.grid(1, 8, 7, 14, 1, 16):
                with T.block("res"):
                    v_bo = T.axis.spatial(1, bo_0 + bo_1)
                    v_co = T.axis.spatial(16, co_0_0 * 16 + co_0_1 * 8 + co_1)
                    v_i = T.axis.spatial(14, i_0 * 7 + i_1)
                    v_j = T.axis.spatial(14, j_0 * 14 + j_1)
                    v_bi, v_ci = T.axis.remap("SS", [bi, ci])
                    T.reads(res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    T.writes(res[v_bo, v_co, v_i, v_j, v_bi, v_ci])
                    res[v_bo, v_co, v_i, v_j, v_bi, v_ci] = T.Cast("int8", res_min[v_bo, v_co, v_i, v_j, v_bi, v_ci])

@I.ir_module
class BidirectionalShiftModule:
    @R.function
    def main(data: R.Tensor((4,), "int32"), shift_map: R.Tensor((4,), "int32")) -> R.Tensor((4,), "int32"):
        with R.dataflow():
            is_positive = shift_map > R.const(0, "int32")
            is_negative = shift_map < R.const(0, "int32")
            magnitude = R.where(is_negative, -shift_map, shift_map)
            right_shifted = R.right_shift(data, magnitude)
            left_shifted  = R.left_shift(data, magnitude)
            result = R.where(is_positive, right_shifted, left_shifted)
            R.output(result)
        return result

if __name__ == "__main__":
    from vtar.relax.transform import print_report
    convert_layout = relax.transform.ConvertLayout({
        "relax.nn.conv2d": ["NCHW1n16c", "OIHW16o16i"],
    })

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/mnist-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    # TODO: improve RemoveUnnecessaryDequantizeQuantizeWrapping
    mod = vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping()(mod)
    mod = relax.transform.FoldConstant()(mod)
    # mod = vtar.relax.transform.GraphPack(bitpack_start="relax.quantize", bitpack_end="relax.dequantize")
    mod.show()
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    raise SystemExit(0)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/vgg16-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/resnet50-v1-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/densenet-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/squeezenet1.0-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    # All of the above should go on VTA with no problems!

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/mobilenetv2-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/efficientnet-lite4-11-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    raise SystemExit(0)

    mod = relax.transform.FoldConstant()(mod)
    mod.show()

    raise SystemExit(0)

    # The "metadata" notation is used every time a Relax constant is not a
    # scalar as it can be seen by (which seem to be bugged but okay)
    program, metadata = relax.utils.metadata_partitioner(relax.const(numpy.ones(1)).script(show_meta=True))
    # There is no "metatadata" attribute anywhere in the IRModule it is just
    # printed that way to serialize all the information in a printable format.

    x = te.placeholder((16,), "float32", "x")
    dtype = te.const(topi.nn.SQNN_DTYPE_TO_CODE['int8'])
    scale = te.compute((1,), lambda i: tir.const(0.5, "float32"), "scale")
    zero_point = te.compute((1,), lambda i: tir.const(3,), "zero_point")
    y = topi.nn.simulated_quantize(x, dtype, scale, zero_point)
    f = te.create_prim_func((x, y))

    onnx_model = onnx.load("build/resnet18_int8_per_tensor.onnx")
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model, keep_params_in_input=True)
    scalar_params = {}
    non_scalar_attr = []
    for param, data in zip(mod['main'].params[1:], mod['main'].attrs['params']):
        is_scalar = not param.struct_info.shape.values
        if is_scalar:
            scalar_params[param] = data
        else:
            non_scalar_attr.append(data)
    mod = relax.transform.BindParams('main', scalar_params)(mod)
    mod['main'] = mod['main'].with_attr("params", non_scalar_attr)

    if False:
        analyze_index_map()
        analyze_tensorir_func_automatic(row_major)
        sch = tir.Schedule(row_major)
        block = sch.get_block("block_row")
        i, j = sch.get_loops(block)
        ij = sch.fuse(i, j)
        func = sch.mod["main"]
        func.show()
        analyze_tensorir_func_automatic(func)
    sch = tir.Schedule(main)
    kernel_cache = sch.get_block("kernel_local.wgt_buffer")
    kernel_cache_loops = sch.get_loops(kernel_cache)
    kernel_cache_fused = sch.fuse(kernel_cache_loops[-4], kernel_cache_loops[-3])
    new_block = sch.blockize(kernel_cache_loops[0])
    sch.mod["main"].show()
    res = analyze_tensorir_func_automatic(sch.mod["main"])
    sch = tir.Schedule(main2)
    kernel_cache = sch.get_block("kernel_local.wgt_buffer")
    kernel_cache_loops = sch.get_loops(kernel_cache)
    _ = sch.blockize(kernel_cache_loops[-6])
    sch.mod["main"].show()
    res = analyze_tensorir_func_automatic(sch.mod["main"])
    if False:
        analyze_tensorir_func_automatic(row_major)
        # Notice how extent in IterMark and IterSplit differs signaling the presence
        # of a modulus operation.
        analyze_tensorir_func_automatic(row_major2)
