"""Dlight stands for Dynamic shape aware Light weight scheduler
https://discuss.tvm.apache.org/t/dlight-enabling-fast-and-efficient-kernel-generation-by-hardware-information/16273/4?u=l0r3m
"""
from typing import Callable, List, Union
import warnings

import tvm
from tvm import dlight as dl
from tvm import tir
from tvm import topi
from tvm import ir

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

        if len(blocks) != 7:
            return None

        env = vtar.get_env()

        data_cache = blocks[0] # Load
        res_conv_block = blocks[1] # GEMM
        kernel_cache = sch.cache_read(res_conv_block, 1, env.wgt_scope) # Load
        res_bias_block = blocks[2] # ALU
        bias_cache = sch.cache_read(res_bias_block, 1, env.acc_scope) # Load
        res_shr_block = blocks[3] # ALU
        # res_add_block = blocks[4] # ALU
        res_min_block = blocks[4] # ALU
        res_max_block = blocks[5] # ALU
        res_block = blocks[6] # Store

        b_block = 1 // env.BATCH
        oc_block = 128 // env.BLOCK_OUT
        ic_block = 16 // env.BLOCK_IN
        h_block = 7
        w_block = 14

        # We skip kernels that can't be spited by this hard-coded numbers...
        N, C, H, W, n, c = topi.utils.get_const_tuple(func.struct_info.params[0].shape)
        if C % ic_block != 0:
            return None

        N, C, H, W, n, c = topi.utils.get_const_tuple(func.struct_info.params[3].shape)
        if N % b_block != 0 or C % oc_block != 0 or H % h_block != 0 or W % w_block:
            return None

        b, oc, y, x, b_tns, oc_tns = sch.get_loops(res_block)
        b_out, b_inn = sch.split(b, (None, b_block))
        oc_out, oc_inn = sch.split(oc, (None, oc_block))
        y_out, y_inn = sch.split(y, (None, h_block))
        x_out, x_inn = sch.split(x, (None, w_block))
        sch.reorder(b_out, oc_out, y_out, x_out, b_inn, oc_inn, y_inn, x_inn, b_tns, oc_tns)

        sch.compute_at(res_max_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_min_block, x_out, preserve_unit_loops=True)
        # sch.compute_at(res_add_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_shr_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_bias_block, x_out, preserve_unit_loops=True)
        sch.compute_at(res_conv_block, x_out, preserve_unit_loops=True)

        # sch.mod["main"].show()

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
        # If it is not divisible T.where is generated which is then lowered to
        # an "if" which we can't compile!
        if sch.get(oc_out).extent % 2 != 0:
            return None
        _, tx = sch.split(oc_out, (None, v_threads))
        sch.reorder(tx, b_out)
        if False:
            sch.bind(tx, "vthread.x")

        conv_init = sch.decompose_reduction(res_conv_block, ic_out)

        # sch.mod["main"].show()

        sch.compute_at(data_cache, ic_out)
        sch.compute_at(kernel_cache, ic_out)
        sch.compute_at(bias_cache, x_out) # NOTE: not sure this is correct...

        sch.set_scope(data_cache, 0, env.inp_scope)
        sch.set_scope(res_conv_block, 0, env.acc_scope)
        sch.set_scope(res_bias_block, 0, env.acc_scope)
        sch.set_scope(res_shr_block, 0, env.acc_scope)
        # sch.set_scope(res_add_block, 0, env.acc_scope)
        sch.set_scope(res_min_block, 0, env.acc_scope)
        sch.set_scope(res_max_block, 0, env.acc_scope)

        sch.annotate(sch.get_loops(data_cache)[-3], env.dma_copy, 0)
        sch.annotate(sch.get_loops(kernel_cache)[-5], env.dma_copy, 0)
        sch.annotate(sch.get_loops(bias_cache)[-2], env.dma_copy, 0) # NOTE: not sure this is correct...
        sch.annotate(sch.get_loops(res_block)[-4], env.dma_copy, 0)
        sch.annotate(sch.get_loops(res_bias_block)[-5], env.alu, 0)
        sch.annotate(sch.get_loops(res_shr_block)[-6], env.alu, 0)
        # sch.annotate(sch.get_loops(res_add_block)[-6], env.alu, 0)
        sch.annotate(sch.get_loops(res_min_block)[-6], env.alu, 0)
        sch.annotate(sch.get_loops(res_max_block)[-6], env.alu, 0)

        # sch.mod["main"].show()

        init_loops = sch.get_loops(conv_init)
        ij_init = sch.fuse(init_loops[-2], init_loops[-1])
        sch.tensorize(ij_init, "vta_init_intrin1")
        conv_loops = sch.get_loops(res_conv_block)
        ij_conv = sch.fuse(conv_loops[-3], conv_loops[-2])
        sch.tensorize(ij_conv, "vta_gemm_intrin1")

        # sch.mod["main"].show()

        return sch

def is_reduction(sch: tir.Schedule, block_rv: tir.schedule.BlockRV) -> bool:
    block = sch.get(block_rv)
    res = any(
        iter_var.iter_type == tir.IterVar.CommReduce
        for iter_var in block.iter_vars
    )
    return res

def is_injective(sch: tir.Schedule, block_rv: tir.schedule.BlockRV) -> bool:
    block = sch.get(block_rv)
    res = all(
        iter_var.iter_type == tir.IterVar.DataPar
        for iter_var in block.iter_vars
    )
    return res

def is_conv2d(sch: tir.Schedule, block_rv: tir.schedule.BlockRV) -> bool:
    block = sch.get(block_rv)
    res = block.name_hint.startswith("conv2d_NCHWnc")
    return res

def normalize_bidi_shift(sch: tir.Schedule, child_rvs: List[tir.schedule.BlockRV]) -> None:
    # We look if there is a bidi shift operation and in case we inline it.
    where_rvs = [
        child_rv for child_rv in child_rvs
        if isinstance(sch.get(child_rv).body.value, tir.Select)
    ]
    for where_rv in where_rvs:
        less_equal_rv, shift_right_rv, shift_left_rv = [
            producer
            for producer in sch.get_producers(where_rv)
        ]
        less_equal = sch.get(less_equal_rv)
        shift_right = sch.get(shift_right_rv)
        shift_left = sch.get(shift_left_rv)
        negate_prod = sch.get_producers(shift_left_rv)
        is_ok = True
        if len(negate_prod) == 2:
            negate = sch.get(negate_prod[1])
        else:
            is_ok = False
        # Tries to match for tir.Select(0 <= s, x >> s, x << -s)
        is_ok = (
            is_ok
            # first level checks.
            and isinstance(less_equal.body.value, tir.LE)
            and isinstance(shift_right.body.value, tir.Call)
            and isinstance(shift_left.body.value, tir.Call)
            # second level checks
            and less_equal.body.value.a == 0
            and shift_right.body.value.op.name == "tir.shift_right"
            and shift_left.body.value.op.name == "tir.shift_left"
            and ir.structural_equal(less_equal.body.value.b, shift_right.body.value.args[1], True)
            and ir.structural_equal(shift_right.body.value.args[0], shift_left.body.value.args[0], True)
            # third level checks
            and isinstance(negate.body.value, tir.Mul)
            and negate.body.value.b == -1
            and ir.structural_equal(shift_right.body.value.args[1], negate.body.value.a, True)
        )
        if is_ok:
            sch.compute_inline(negate_prod[1])
            sch.compute_inline(shift_left_rv)
            sch.compute_inline(shift_right_rv)
            sch.compute_inline(less_equal_rv)
            # sch.annotate(where_rv, env.alu, 0)

def not_in(sch, block_rv: tir.schedule.BlockRV, rvs: List[tir.schedule.BlockRV]) -> bool:
    """BlockRV works differently then SBlockRV..."""
    block = sch.get(block_rv)
    res = all(not block.same_as(sch.get(rv)) for rv in rvs)
    return res

# Modeled after https://github.com/apache/tvm/blob/v0.23.0/python/tvm/dlight/adreno/convolution.py
class Conv2DPrime(VTAScheduleRule):
    def apply(
        self,
        func: tir.PrimFunc,
        target: tvm.target.Target,
        tunable: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        env = vtar.get_env()
        sch = tir.Schedule(func)

        root_rv = dl.analysis.get_root_block(sch)
        child_rvs = sch.get_child_blocks(root_rv)
        normalize_bidi_shift(sch, child_rvs)
        child_rvs = sch.get_child_blocks(root_rv)

        reduction_block_rvs = [
            child_rv for child_rv in child_rvs if is_reduction(sch, child_rv)
        ]

        # Right now only one convolution is supported even though in the future
        # it might be necessary to support two for asymmetric weight quantization.
        if len(reduction_block_rvs) != 1 or not is_conv2d(sch, reduction_block_rvs[0]):
            return None

        conv2d_rv = reduction_block_rvs[0]
        sch.set_scope(conv2d_rv, 0, env.acc_scope)
        conv2d_prod_rvs = sch.get_producers(conv2d_rv)
        if conv2d_prod_rvs:
            data_cache_rv = conv2d_prod_rvs[0]
            sch.set_scope(data_cache_rv, 0, env.inp_scope)
        else:
            data_cache_rv = sch.cache_read(conv2d_rv, 0, env.inp_scope)
        kernel_cache_rv = sch.cache_read(conv2d_rv, 1, env.wgt_scope)
        sch.annotate(data_cache_rv, env.dma_copy, 0) # FIXME: annotate loop
        sch.annotate(kernel_cache_rv, env.dma_copy, 0) # FIXME: annotate loop
        input_rvs = (data_cache_rv, kernel_cache_rv)

        output_rvs = sch.get_output_blocks(root_rv)
        for output_rv in output_rvs:
            # No need to set_scope
            sch.annotate(output_rv, env.dma_copy, 0) # FIXME: annotate loop

        remaining_block_rvs = [
            child_rv for child_rv in child_rvs if (
                not_in(sch, child_rv, reduction_block_rvs)
                and not_in(sch, child_rv, input_rvs)
                and not_in(sch, child_rv, output_rvs)
            )
        ]

        from .tir.util import get_alu_op
        from tvm import arith
        analyzer = arith.Analyzer()
        for remaining_block_rv in remaining_block_rvs:
            if not is_injective(sch, remaining_block_rv):
                return None
            _, _, _, err = get_alu_op(env, analyzer, sch.get(remaining_block_rv).body.value)
            if err:
                return None

        # data:   NCHWnc
        # kernel: OIRSoi
        # output: NOĤŴno
        (
            b_o, c_o, i, j, b_i, c_i, # NOĤŴno (all S)
            k_o, d_i, d_j, k_i        # IRSi   (all R)
        ) = sch.get_loops(conv2d_rv)
        # TODO: determine this splits such that the maximum amount of SRAM is used.
        b_block = 1 // env.BATCH
        oc_block = 128 // env.BLOCK_OUT
        ic_block = 16 // env.BLOCK_IN
        h_block = 7
        w_block = 14
        b_out, b_inn = sch.split(b_o, (None, b_block))
        oc_out, oc_inn = sch.split(c_o, (None, oc_block))
        y_out, y_inn = sch.split(i, (None, h_block))
        x_out, x_inn = sch.split(j, (None, w_block))
        ic_out, ic_inn = sch.split(k_o, (None, ic_block))
        sch.reorder(
            b_out, oc_out, y_out, x_out,
            ic_out, b_inn, oc_inn, # RSS
            y_inn, ic_inn, d_i, d_j, x_inn, # SRRRS
            b_i, c_i, k_i # SSR

        )

        # vthreads double the memory usage. Hence this must be taken into
        # account.
        v_threads = 2
        # If it is not divisible T.where is generated which is then lowered to
        # an "if" which we can't compile!
        if sch.get(oc_out).extent % 2 != 0: return None # FIXME
        _, tx = sch.split(oc_out, (None, v_threads))
        sch.reorder(tx, b_out)
        sch.bind(tx, "vthread.x")

        conv2d_init_rv = sch.decompose_reduction(conv2d_rv, ic_out)
        sch.compute_at(data_cache_rv, ic_out, preserve_unit_loops=True)
        sch.compute_at(kernel_cache_rv, ic_out, preserve_unit_loops=True)

        arg_buffers = set(func.buffer_map.values())
        # Hopefully topologically sorted.
        for remaining_block_rv in remaining_block_rvs:
            block = sch.get(remaining_block_rv)
            _, lhs, rhs, _ = get_alu_op(env, analyzer, sch.get(remaining_block_rv).body.value)
            # tir.IntImm, tir.BufferLoad

            lhs_idx, rhs_idx = 0, 1
            if isinstance(block.body.value, tir.Select):
                lhs_idx, rhs_idx = rhs_idx, lhs_idx
            remaining_block_cache_rv = None
            if isinstance(lhs, tir.BufferLoad):
                if lhs.buffer in arg_buffers:
                    remaining_block_cache_rv = sch.cache_read(remaining_block_rv, lhs_idx, env.acc_scope)
            if isinstance(rhs, tir.BufferLoad):
                if rhs.buffer in arg_buffers:
                    remaining_block_cache_rv = sch.cache_read(remaining_block_rv, rhs_idx, env.acc_scope)


            sch.get(remaining_block_rv).show()
            if remaining_block_cache_rv:
                sch.reverse_compute_at(remaining_block_cache_rv, x_out, preserve_unit_loops=True)
            # TODO: split by hand...
            # sch.reverse_compute_at(remaining_block_rv, x_out, preserve_unit_loops=True)
            sch.set_scope(remaining_block_rv, 0, env.acc_scope)
            sch.annotate(remaining_block_rv, env.alu, 0) # FIXME: annotate loop

        # for output_rv in output_rvs:
        #     sch.compute_at(output_rv, x_out)

        return sch

        # Schedule conv2d to use all the SRAM in inp e wgt
        # Schedule ALU blocks (can optionally load to acc if arg is not scalar.)

        # From the output there should be one produces until the inputs...
        producer_rvs = [output_rv]
        while True:
            if len(producer_rvs) == 0:
                break
            if len(producer_rvs) > 1:
                # Only "directed path" operation are supported.
                print("bad")
                return None
            producer_rv = producer_rvs[0]
            producer = sch.get(producer_rv)
            if not all(iter_var.iter_type == tir.IterVar.DataPar for iter_var in producer.iter_vars):
                if not sch.get(producer_rv).same_as(sch.get(conv2d_rv)):
                    # There must be only one reduction block and it must be the
                    # convolution
                    return None
            producer_rvs = sch.get_producers(producer_rvs[0])

        return sch
