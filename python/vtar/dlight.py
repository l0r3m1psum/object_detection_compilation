"""Dlight stands for Dynamic shape aware Light weight scheduler
https://discuss.tvm.apache.org/t/dlight-enabling-fast-and-efficient-kernel-generation-by-hardware-information/16273/4?u=l0r3m
"""
from typing import Callable, List, Union

import tvm
from tvm import dlight as dl
from tvm import tir

import vtar

class VTAScheduleRule(dl.ScheduleRule):

    def is_target_available(self, target: tvm.target.Target) -> bool:
        # TODO: implement this correctly
        return super().is_target_available(target) and "vta" == target.kind.name

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
        sch = tir.Schedule(func)
        block_infos = dl.normalize_prim_func(sch)
        print(block_infos)
        sch.mod.show()
        # In the TVM compiler the schedule rule implement ways to recognize if
        # the computation performed is the expected one e.g.
        # tvm.dlight.analysis.gemv.is_gemv
        # Here we could simplify our lives by checking for specific attributes...
        # TODO: normalize to 4 loops by adding unit loops...
        if len(block_infos) != 2:
            return None
        # block_name = block_infos[0].name
        alu_block = block_infos[0].block_rv
        cast_block = block_infos[1].block_rv
        env = vtar.get_env()
        lhs_cache = sch.cache_read(alu_block, 0, env.acc_scope)
        rhs_cache = sch.cache_read(alu_block, 1, env.acc_scope)
        sch.set_scope(alu_block, 0, env.acc_scope)
        sch.annotate(sch.get_loops(lhs_cache)[0], env.dma_copy, True)
        sch.annotate(sch.get_loops(rhs_cache)[0], env.dma_copy, True)
        sch.annotate(sch.get_loops(alu_block)[0], env.alu, True)
        sch.annotate(sch.get_loops(cast_block)[0], env.dma_copy, True)
        sch.mod.show()
        return sch
