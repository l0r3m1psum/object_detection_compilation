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
from .tir import transform
from .environment import get_env, Environment


def get_vtar_tir_transform() -> tvm.ir.transform.Pass:
    # Some documentation for old transformations is still available here
    # https://mlc.ai/docs/reference/api/tir/transform.html
    return tvm.transform.Sequential([
        # vtar.tir.transform.InjectConv2DTransposeSkip(), # TODO
        transform.InjectDMAIntrin(),
        # transform.InjectSkipCopy(), # TODO: Just for debug
        transform.AnnotateALUCoProcScope(),
        transform.LiftAttrScope("coproc_uop_scope"),
        transform.LiftAllocToScopeBegin(),
        transform.LiftAttrScope("coproc_scope"),
        transform.LiftAttrScope("extern_scope"),
        transform.CoProcSync(), # This inserts the copro_(dep_push|dep_pop|sync)
        # transform.InjectDebug,
        transform.InjectALUIntrin(),
        # Taken from tvm.tir.get_default_tir_pipeline in pipeline.py ###########
        tvm.tir.transform.ConvertBlocksToOpaque(),
        tvm.tir.transform.CompactBufferAllocation(),
        tvm.tir.transform.LowerMatchBuffer(),
        tvm.tir.transform.LowerOpaqueBlock(),
        tvm.tir.transform.FlattenBuffer(), tvm.ir.transform.PrintIR("After FlattenBuffer"),
        ########################################################################
        tvm.tir.transform.StorageRewrite(), tvm.ir.transform.PrintIR("After StorageRewrite"),
        tvm.tir.transform.LowerDeviceStorageAccessInfo(), tvm.ir.transform.PrintIR("After LowerDeviceStorageAccessInfo"),
        # transform.FoldUopLoop(), # TODO
        # transform.CPUAccessRewrite(), # TODO
    ])


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
