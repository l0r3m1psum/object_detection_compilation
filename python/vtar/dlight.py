"""Dlight stands for Dynamic shape aware Light weight scheduler
https://discuss.tvm.apache.org/t/dlight-enabling-fast-and-efficient-kernel-generation-by-hardware-information/16273/4?u=l0r3m
"""
from typing import Callable, List, Union
import warnings

import tvm
from tvm import dlight as dl
from tvm import tir

import vtar.tir.util


class VTAScheduleRule(dl.ScheduleRule):

    def is_target_available(self, target: tvm.target.Target) -> bool:
        # TODO: implement this correctly
        breakpoint()
        return super().is_target_available(target) and "vta" == target.kind.name

# NOTE: those checks are best effort, some better idiom recognition strategy is needed
# https://www.emmtrix.com/wiki/Idiom_Recognizer
# https://dl.acm.org/doi/10.1145/224538.224655
# https://ieeexplore.ieee.org/document/5372793

def is_alu_and_load(sch: tir.Schedule, block_info: dl.BlockInfo) -> bool:
    block = sch.get(block_info.block_rv)
    read_num_condition = 1 <= len(block.reads) <= 2
    write_num_condition = len(block.writes) == 1
    iter_dom_condition = block_info.is_injective()
    iter_num_cond = 2 <= len(block_info.iters) <= 4 # This condition might be too restrictive
    is_buffer_store = isinstance(block.body, tir.BufferStore)
    if not (read_num_condition and write_num_condition and iter_dom_condition
        and iter_num_cond and is_buffer_store):
        return False
    value = block.body.value
    alu_opcode, lhs, rhs, error = vtar.tir.util.get_alu_op(vtar.get_env(), tvm.arith.Analyzer(), value)
    if error: return False
    return True

def is_store(sch: tir.Schedule, block_info: dl.BlockInfo) -> bool:
    block = sch.get(block_info.block_rv)
    read_num_condition = len(block.reads) == 1
    write_num_condition = len(block.writes) == 1
    iter_dom_condition = block_info.is_injective()
    iter_num_cond = 2 <= len(block_info.iters) <= 4 # This condition might be too restrictive
    is_buffer_store = isinstance(block.body, tir.BufferStore)
    if not (read_num_condition and write_num_condition and iter_dom_condition
        and iter_num_cond and is_buffer_store):
        return False
    value = block.body.value
    if not isinstance(value, tir.Cast):
        return False
    return True

class ALU(VTAScheduleRule):
    def apply(
        self,
        func: tir.PrimFunc,
        target: tvm.target.Target,
        tunable: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        """Apply the ScheduleRule to the given PrimFunc.

        Parameters
        ----------
        func : tir.PrimFunc
            The PrimFunc to apply the ScheduleRule to.
        target : Target
            The compilation target the schedule is supposed to be built for.
        tunable : bool
            Whether the schedule is allowed to contain tunable instructions.

        Returns
        -------
        results : Union[None, tir.Schedule, List[tir.Schedule]]
            Either a Schedule, a list of Schedules, or None, where None means that the rule
            is not applicable to the given PrimFunc.
        """
        # For now we only support a function with 2 input and one output.
        if len(func.params) != 2 + 1:
            return None

        sch = tir.Schedule(func)
        block_infos = dl.normalize_prim_func(sch)

        # TODO: add type checking...

        # TODO: check that only the first block loads from global memory...

        if not all(([is_alu_and_load(sch, block_info) for block_info in block_infos[:-1]])):
            return None

        if not is_store(sch, block_infos[-1]):
            return None

        if len(block_infos) < 2:
            return None

        alu_and_load_block = block_infos[0].block_rv
        cast_block = block_infos[-1].block_rv
        env = vtar.get_env()
        lhs_cache = sch.cache_read(alu_and_load_block, 0, env.acc_scope)
        rhs_cache = sch.cache_read(alu_and_load_block, 1, env.acc_scope)
        sch.set_scope(alu_and_load_block, 0, env.acc_scope)
        sch.annotate(sch.get_loops(lhs_cache)[0], env.dma_copy, True)
        sch.annotate(sch.get_loops(rhs_cache)[0], env.dma_copy, True)
        sch.annotate(sch.get_loops(alu_and_load_block)[0], env.alu, True)
        sch.annotate(sch.get_loops(cast_block)[0], env.dma_copy, True)

        for block_info in block_infos[1:-1]:
            intermediate_alu_block = block_info.block_rv
            sch.set_scope(intermediate_alu_block, 0, env.acc_scope)
            sch.annotate(sch.get_loops(intermediate_alu_block)[0], env.alu, True)

        return sch

def is_conv2d(sch: tir.Schedule, block_info: dl.BlockInfo) -> bool:
    block = sch.get(block_info.block_rv)
    read_num_condition = len(block.reads) == 2 # must read data and weight
    write_num_condition = len(block.writes) == 1
    iter_dom_condition = not block_info.is_injective() # must have reductions
    iter_num_cond = 2 <= len(block_info.iters) <= 4 # This condition might be too restrictive
    is_buffer_store = isinstance(block.body, tir.BufferStore)
    if not (read_num_condition and write_num_condition and iter_dom_condition
        and iter_num_cond and is_buffer_store):
        return False
    value = block.body.value
    alu_opcode, lhs, rhs, error = vtar.tir.util.get_alu_op(vtar.get_env(), tvm.arith.Analyzer(), value)
    if error: return False
    return True

# Taken from tvm.dlight.analysis.common_analysis
def _iter_kind(i: tir.IterVar) -> str:
    return {
        tir.IterVar.DataPar: "S",
        tir.IterVar.CommReduce: "R",
    }.get(i.iter_type, "O") # O as in Other I guess.

class Conv2D(VTAScheduleRule):
    def apply(
        self,
        func: tir.PrimFunc,
        target: tvm.target.Target,
        tunable: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:

        # data, weight, bias + output
        if len(func.params) != 3 + 1:
            return None

        if any(len(param.shape) != 6 for param in func.struct_info.params):
            return None

        sch = tir.Schedule(func)
        # For some reason normalization removes loops with extent 1, so for us
        # is not usable...
        # block_infos = dl.normalize_prim_func(sch)
        blocks: List[tir.schedule.BlockRV] = sch.get_child_blocks(sch.get_block("root"))

        # Then initial padding is optional
        block0_iter_kinds = [_iter_kind(iter_var) for iter_var in sch.get(blocks[0]).iter_vars]
        block0_is_injective = all(iter_kind == "S" for iter_kind in block0_iter_kinds)
        if not block0_is_injective:
            # For the moment we skip it
            return None

        if len(blocks) != 8:
            return None

        env = vtar.get_env()

        data_cache = blocks[0] # Load
        res_conv_block = blocks[1] # GEMM
        kernel_cache = sch.cache_read(res_conv_block, 1, env.wgt_scope) # Load
        res_bias_block = blocks[2] # ALU
        bias_cache = sch.cache_read(res_bias_block, 1, env.acc_scope) # Load
        res_shr_block = blocks[3] # ALU
        res_add_block = blocks[4] # ALU
        res_min_block = blocks[5] # ALU
        res_max_block = blocks[6] # ALU
        res_block = blocks[7] # Store

        # TODO: take this numbers form data and weight
        b_block = 1 // env.BATCH
        oc_block = 128 // env.BLOCK_OUT
        ic_block = 16 // env.BLOCK_IN
        h_block = 7
        w_block = 14

        b, oc, y, x, b_tns, oc_tns = sch.get_loops(res_block)
        b_out, b_inn = sch.split(b, (None, b_block))
        oc_out, oc_inn = sch.split(oc, (None, oc_block))
        y_out, y_inn = sch.split(y, (None, h_block))
        x_out, x_inn = sch.split(x, (None, w_block))
        sch.reorder(b_out, oc_out, y_out, x_out, b_inn, oc_inn, y_inn, x_inn, b_tns, oc_tns)

        sch.compute_at(res_max_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_min_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_add_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_shr_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_bias_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_conv_block, x_out, preserve_unit_loops=True)

        # sch.mod["main"].show()
        # return sch

        # oc = output_channel (spatial axis)
        # ic = input_channel (reduce axis)
        (
            b_out, oc_out, y_out, x_out, # outer
            b_inn, oc_inn, y_inn, x_inn, # inner
            b_tns, oc_tns, ic, dy, dx, ic_tns # bi, ci, ic, dy, dx, ic_tns
        ) = sch.get_loops(res_conv_block)
        ic_out, ic_inn = sch.split(ic, (None, ic_block))
        sch.reorder(
            ic_out, b_inn, oc_inn, # RSS
            y_inn, ic_inn, dy, dx, x_inn, # SRRRS
            b_tns, oc_tns, ic_tns # SSR
        )

        v_threads = 2
        _, tx = sch.split(oc_out, (None, v_threads))
        sch.reorder(tx, b_out)
        warnings.warn(UserWarning("skipping thread binding because it is not "
            "supported in codegen_llvm.cc::GetThreadIndex"))
        # TODO: what I need here is probably software pipeline but
        # tir.transform.InjectSoftwarePipeline should be only for CUDA
        if False:
            sch.bind(tx, "cthread")

        conv_init = sch.decompose_reduction(res_conv_block, ic_out)

        # sch.mod["main"].show()

        sch.compute_at(data_cache, ic_out)
        sch.compute_at(kernel_cache, ic_out)
        # sch.compute_at(bias_cache, ic_out)

        sch.set_scope(data_cache, 0, env.inp_scope)
        sch.set_scope(res_conv_block, 0, env.acc_scope)
        sch.set_scope(res_shr_block, 0, env.acc_scope)
        sch.set_scope(res_add_block, 0, env.acc_scope)
        sch.set_scope(res_min_block, 0, env.acc_scope)
        sch.set_scope(res_max_block, 0, env.acc_scope)

        sch.annotate(sch.get_loops(data_cache)[-3], env.dma_copy, 0)
        sch.annotate(sch.get_loops(kernel_cache)[-5], env.dma_copy, 0)
        sch.annotate(sch.get_loops(res_block)[-4], env.dma_copy, 0)
        sch.annotate(sch.get_loops(res_shr_block)[-6], env.alu, 0)
        sch.annotate(sch.get_loops(res_add_block)[-6], env.alu, 0)
        sch.annotate(sch.get_loops(res_min_block)[-6], env.alu, 0)
        sch.annotate(sch.get_loops(res_max_block)[-6], env.alu, 0)

        init_loops = sch.get_loops(conv_init)
        ij_init = sch.fuse(init_loops[-2], init_loops[-1])
        sch.tensorize(ij_init, "vta_init_intrin1")
        conv_loops = sch.get_loops(res_conv_block)
        ij_conv = sch.fuse(conv_loops[-3], conv_loops[-2])
        sch.tensorize(ij_conv, "vta_gemm_intrin1")

        sch.mod["main"].show()

        return sch
