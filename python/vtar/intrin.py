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
"""VTA related intrinsics"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te

from tvm.script import tir as T

from .environment import get_env

def get_gemm_desc() -> tvm.tir.PrimFunc:
    env = get_env()

    wgt_lanes = env.WGT_ELEM_BITS // env.WGT_WIDTH
    assert wgt_lanes == env.BLOCK_OUT * env.BLOCK_IN
    wgt_shape = (env.BLOCK_OUT, env.BLOCK_IN)
    assert wgt_shape[0] * wgt_shape[1] == wgt_lanes

    inp_lanes = env.INP_ELEM_BITS // env.INP_WIDTH
    assert inp_lanes == env.BATCH * env.BLOCK_IN
    inp_shape = (env.BATCH, env.BLOCK_IN)
    assert inp_shape[0] * inp_shape[1] == inp_lanes

    out_lanes = env.ACC_ELEM_BITS // env.ACC_WIDTH
    assert out_lanes == env.BATCH * env.BLOCK_OUT
    out_shape = (env.BATCH, env.BLOCK_OUT)
    assert out_shape[0] * out_shape[1] == out_lanes

    wgt = te.placeholder(
        (wgt_shape[0], wgt_shape[1]), dtype="int%d" % env.WGT_WIDTH, name=env.wgt_scope
    )
    inp = te.placeholder(
        (inp_shape[0], inp_shape[1]), dtype="int%d" % env.INP_WIDTH, name=env.inp_scope
    )
    k = te.reduce_axis((0, wgt_shape[1]), name="k")
    out_dtype = "int%d" % env.ACC_WIDTH
    out = te.compute(
        (out_shape[0], out_shape[1]),
        lambda i, j: te.sum(inp[i, k].astype(out_dtype) * wgt[j, k].astype(out_dtype), axis=[k]),
        name="out",
    )

    gemm_desc = te.create_prim_func([wgt, inp, out]).with_attr({"global_symbol": "gemm"})

    return gemm_desc

@T.prim_func
def gemm_desc(
        local_wgt_buffer: T.Buffer((16, 16), "int8"),
        local_inp_buffer: T.Buffer((1, 16), "int8"),
        out: T.Buffer((1, 16), "int32")
    ) -> None:
    T.func_attr({"tir.noalias": T.bool(True)})
    with T.block("root"):
        T.reads(out[0:1, 0:16], local_inp_buffer[0:1, 0:16], local_wgt_buffer[0:16, 0:16])
        T.writes(out[0:1, 0:16])
        for i, j, k in T.grid(1, 16, 16):
            with T.block("out"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(out[v_i, v_j], local_inp_buffer[v_i, v_k], local_wgt_buffer[v_j, v_k])
                T.writes(out[v_i, v_j])
                # with T.init(): out[v_i, v_j] = 0
                out[v_i, v_j] += T.Cast("int32", local_inp_buffer[v_i, v_k]) \
                    * T.Cast("int32", local_wgt_buffer[v_j, v_k])

# TODO: this should be created dynamically with something like ir_builder so
# that the width of the data types and the other parameters can be determined at
# runtime.
assert get_env().acc_dtype == "int32"
assert get_env().wgt_dtype == "int8"
assert get_env().inp_dtype == "int8"
assert get_env().dev.get_task_qid(get_env().dev.QID_COMPUTE) == 2
assert get_env().dev.vta_push_uop == "VTAPushGEMMOp"
get_env().dev.vta_axis
@T.prim_func
def gemm_intrin(
        local_wgt_buffer: T.Buffer((16, 16), "int8"),
        local_inp_buffer: T.Buffer((1, 16), "int8"),
        out: T.Buffer((1, 16), "int32")
    ) -> None:
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    vta = T.int32()
    with T.attr(T.iter_var(vta, None, "ThreadIndex", "vta"), "coproc_scope", 2), \
        T.attr(T.iter_var(vta, None, "ThreadIndex", "vta"), "coproc_uop_scope", "VTAPushGEMMOp"):
        for i, j, k in T.grid(1, 16, 16):
            with T.block("out"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(out[v_i, v_j], local_inp_buffer[v_i, v_k], local_wgt_buffer[v_j, v_k])
                T.writes(out[v_i, v_j])
                # with T.init():
                #     T.call_intrin("void", "tir.vta.uop_push",
                #         0, # mode = 0 is GEMM
                #         1, # reset_out = 1 is "reset the accumulator"
                #         out.access_ptr("rw", "int32"), # dst_index
                #         0, # src_index
                #         0, # wgt_index
                #         0, 0, 0 # parameters for ALU (ignored)
                #     )
                T.call_intrin("void", "tir.vta.uop_push",
                    0, # mode = 0 is GEMM
                    0, # reset_out = 0 is "do not reset the accumulator"
                    out.access_ptr("rw", "int32"), # dst_index
                    local_inp_buffer.access_ptr("r", "int32"), # src_index
                    local_wgt_buffer.access_ptr("r", "int32"), # wgt_index
                    0, 0, 0 # parameters for ALU (ignored)
                )


tvm.tir.TensorIntrin.register("vta_gemm_intrin", gemm_desc, gemm_intrin)

def gemm(env, mock=False):
    """Matrix-matrix multiply intrinsic

    Parameters
    ----------
    env : Environment
        The Environment

    mock : bool
        Whether create a mock version.
    """
    wgt_lanes = env.WGT_ELEM_BITS // env.WGT_WIDTH
    assert wgt_lanes == env.BLOCK_OUT * env.BLOCK_IN
    wgt_shape = (env.BLOCK_OUT, env.BLOCK_IN)
    assert wgt_shape[0] * wgt_shape[1] == wgt_lanes

    inp_lanes = env.INP_ELEM_BITS // env.INP_WIDTH
    assert inp_lanes == env.BATCH * env.BLOCK_IN
    inp_shape = (env.BATCH, env.BLOCK_IN)
    assert inp_shape[0] * inp_shape[1] == inp_lanes

    out_lanes = env.ACC_ELEM_BITS // env.ACC_WIDTH
    assert out_lanes == env.BATCH * env.BLOCK_OUT
    out_shape = (env.BATCH, env.BLOCK_OUT)
    assert out_shape[0] * out_shape[1] == out_lanes

    ############################################################################

    wgt = te.placeholder(
        (wgt_shape[0], wgt_shape[1]), dtype="int%d" % env.WGT_WIDTH, name=env.wgt_scope
    )
    inp = te.placeholder(
        (inp_shape[0], inp_shape[1]), dtype="int%d" % env.INP_WIDTH, name=env.inp_scope
    )
    k = te.reduce_axis((0, wgt_shape[1]), name="k")
    out_dtype = "int%d" % env.ACC_WIDTH
    out = te.compute(
        (out_shape[0], out_shape[1]),
        lambda i, j: te.sum(inp[i, k].astype(out_dtype) * wgt[j, k].astype(out_dtype), axis=[k]),
        name="out",
    )

    gemm_desc = te.create_prim_func([wgt, inp, out]).with_attr({"global_symbol": "gemm"})

    ############################################################################

    wgt_layout = tvm.tir.decl_buffer(
        wgt.shape,
        wgt.dtype,
        env.wgt_scope,
        scope=env.wgt_scope,
        offset_factor=wgt_lanes,
        data_alignment=wgt_lanes,
    )
    inp_layout = tvm.tir.decl_buffer(
        inp.shape,
        inp.dtype,
        env.inp_scope,
        scope=env.inp_scope,
        offset_factor=inp_lanes,
        data_alignment=inp_lanes,
    )
    out_layout = tvm.tir.decl_buffer(
        out.shape,
        out.dtype,
        env.acc_scope,
        scope=env.acc_scope,
        offset_factor=out_lanes,
        data_alignment=out_lanes,
    )

    def intrin_func(ins, outs):
        """Matrix-matrix multiply intrinsic function"""
        dinp, dwgt = ins
        dout = outs[0]

        def instr(index):
            """Generate matrix-matrix multiply VTA instruction"""
            irb = tvm.tir.ir_builder.create()
            dev = env.dev
            irb.scope_attr(dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE))
            irb.scope_attr(dev.vta_axis, "coproc_uop_scope", dev.vta_push_uop)
            if index in (0, 2):
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        0,
                        0,
                        dout.access_ptr("rw", "int32"),
                        dinp.access_ptr("r", "int32"),
                        dwgt.access_ptr("r", "int32"),
                        0, 0, 0,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        0,
                        1,
                        dout.access_ptr("rw", "int32"),
                        0,
                        0,
                        0, 0, 0,
                    )
                )
            return irb.get()

        # return a triple of normal-set, reset, update
        nop = tvm.tir.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(0), instr(1), instr(2))

    # https://heterocl.csl.cornell.edu/doc/heterocl.tvm.tensor_intrin.html
    # body and reduce_update are identical
    body, reduce_init, reduce_update = intrin_func((inp_layout, wgt_layout), (out_layout,))
    gemm_intrin = tvm.tir.PrimFunc(
        (inp_layout, wgt_layout, out_layout,),
        body
    )
    breakpoint()
    return None

    return te.decl_tensor_intrin(
        out.op, intrin_func, name="GEMM", binds={inp: inp_layout, wgt: wgt_layout, out: out_layout}
    )

# gemm(get_env())
