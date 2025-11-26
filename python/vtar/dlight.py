"""Dlight stands for Dynamic shape aware Light weight scheduler
https://discuss.tvm.apache.org/t/dlight-enabling-fast-and-efficient-kernel-generation-by-hardware-information/16273/4?u=l0r3m
"""
from typing import Callable, List, Union

import tvm
from tvm import dlight as dl
from tvm import tir

import vtar.tir.util


class VTAScheduleRule(dl.ScheduleRule):

    def is_target_available(self, target: tvm.target.Target) -> bool:
        # TODO: implement this correctly
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

        sch.mod.show()

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
        sch.mod.show()
        return sch
