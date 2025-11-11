# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/matrix_multiply.html
# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/optimize/matrix_multiply_opt.html

import tvm
from tvm import te
import ctypes
vta_fsim = ctypes.CDLL("vta_fsim")
import numpy
import vtar

env = vtar.get_env()

assert env.BATCH == 1
assert env.BLOCK_IN == 16
assert env.BLOCK_OUT == 16

assert env.inp_dtype == "int8"
assert env.wgt_dtype == "int8"
assert env.acc_dtype == "int32"

if False:
    # This works only in tvm 0.21
    from tvm import tir
    from tvm.script import tir as T

    @T.prim_func
    def my_func(a: T.Buffer((10, 10), "float32"),
                b: T.Buffer((10, 10), "float32")):
        for i, j in T.grid(10, 10):
            with T.block("compute_block"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(a[vi, vj])
                T.writes(b[vi, vj])
                b[vi, vj] = a[vi, vj] * 2.0

    @tir.functor.visitor
    class MyBlockIterVarCollector(tir.PyStmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.block_iter_maps = {}

        def visit_block_realize_(self, block_realize: tir.BlockRealize):
            block = block_realize.block
            block_name = block.name_hint if block.name_hint else "unnamed_block"
            # logical_iter_vars = [iter_var.var for iter_var in block.iter_vars]
            logical_iter_vars = block.iter_vars
            bound_iter_vars = block_realize.iter_values

            self.block_iter_maps[block_name] = {
                "logical_iter_vars": [str(v.var) for v in logical_iter_vars],
                "bound_iter_vars": [str(v) for v in bound_iter_vars]
            }

            self.visit_stmt(block)

    collector = MyBlockIterVarCollector()
    collector.visit_stmt(my_func.body)

    print(my_func)
    print(collector.block_iter_maps)

def f():
    BATCH, BLOCK_OUT = 3, 2
    O, M = 3, 4
    o, m = O//BATCH, M//BLOCK_OUT

    A_orig = numpy.arange(O*M).reshape((O, M)).astype("int8")
    B_orig = numpy.arange(O*M).reshape((O, M)).astype("int8")

    A_pack = A_orig.reshape(o, BATCH, m, BLOCK_OUT).transpose((0, 2, 1, 3))
    B_pack = B_orig.reshape(o, BATCH, m, BLOCK_OUT).transpose((0, 2, 1, 3))

    print("Tensorization/matricization is vectorization applied also to the "
        "batch dimension.")
    print(A_orig)
    print(A_pack)
# f()

def vta_alu():
    # A and B originally have shape (O, M). To use VTA to accelerate the alu
    # (element-wise) operations we have to reshape them to
    # (o, m, BATCH, BLOCK_OUT) where o = O//BATCH and m = M//BLOCK_OUT.
    O, M = 1, 1024
    o, m = O//env.BATCH, M//env.BLOCK_OUT
    shape = (o, m, env.BATCH, env.BLOCK_OUT)

    # This could be used to schedule the computation in such a way that the data
    # layout is changed on the fly.
    # https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule.transform_layout

    A = te.placeholder(shape, name="A", dtype=env.acc_dtype)
    B = te.placeholder(shape, name="B", dtype=env.acc_dtype)
    C = te.compute(shape, lambda *i: A(*i) + B(*i), "C")
    D = te.compute(shape, lambda *i: C(*i).astype(env.inp_dtype), "D")
    # Since TVM Schedule.reverse_compute_inline cannot breakup block reads more
    # than one buffer we have to do the split at the TE level instead of
    # scheduling with:
    # s.reverse_compute_inline(s.get_block("D"))
    # s.cache_write(s.get_block("D"), 0, env.acc_scope)

    alu = te.create_prim_func([A, B, D]).with_attr({"global_symbol": "alu"})
    Module = tvm.IRModule({"alu": alu})

    s = tvm.tir.Schedule(Module)
    s.work_on('alu')
    s.cache_read(s.get_block("C"), 0, env.acc_scope)
    s.cache_read(s.get_block("C"), 1, env.acc_scope)
    s.set_scope(s.get_block("C"), 0, env.acc_scope)
    s.annotate(s.get_loops(s.get_block("A_local.acc_buffer"))[0], env.dma_copy, True)
    s.annotate(s.get_loops(s.get_block("B_local.acc_buffer"))[0], env.dma_copy, True)
    s.annotate(s.get_loops(s.get_block("C"))[0], env.alu, True)
    s.annotate(s.get_loops(s.get_block("D"))[0], env.dma_copy, True)
    s.mod.show()
    return s.mod

mod = vta_alu()
mod = vtar.get_vtar_tir_transform()(mod)
mod.show(syntax_sugar=True)

# mod = Module

if False:
    import os
    rng = numpy.random.default_rng(42)
    # from tvm import relax
    # os.environ["TVM_WIN_CC"] = "clang_wrapper.bat"
    ex = tvm.tir.build(mod, tvm.target.Target(env.target, host=env.target_host))
    # TODO: start from an IRModule with a Relax main
    dev = tvm.device(str(env.target))
    # vm = relax.VirtualMachine(ex, dev)
    A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
    ex(A, B, C)
    numpy.testing.assert_equal(C.numpy(), A.numpy() + B.numpy())
    print(C)

# Computation Declaration ######################################################

def g():
    BATCH, BLOCK_IN, BLOCK_OUT = 2, 2, 2
    O, N, M = 4, 8, 6
    o, n, m = O//BATCH, N//BLOCK_IN, M//BLOCK_OUT

    A_orig = numpy.arange(O*N).reshape((O, N))
    B_orig = numpy.arange(M*N).reshape((M, N))

    A_pack = A_orig.reshape(o, BATCH, n, BLOCK_IN).transpose((0, 2, 1, 3))
    B_pack = B_orig.reshape(m, BLOCK_OUT, n, BLOCK_IN).transpose((0, 2, 1, 3))

    C_orig = numpy.einsum("ik,jk->ij", A_orig, B_orig) # A_orig @ B_orig.T
    C_pack = numpy.einsum("IKik,JKjk->IJij", A_pack, B_pack)

    # Here we are essentially doing block matrix multiplication where if
    # normally one would write matrix multiplication in Einstein notation as
    # IJ = IK*KJ in this case it becomes becomes IJij = IKik*KJkj where IJ are
    # meta-rows and meta-columns indices.

    print(A_orig)
    print(A_pack)
    print(B_orig)
    print(B_pack)
    print(C_orig)
    print(C_pack)
# g()

from tvm import ir

def schedule_like_matmul_tutorial(mod: ir.IRModule) -> ir.IRModule:
    """This function schedules the computation like the Simple Matrix Multiply
    tutorial
    https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/matrix_multiply.html
    but tensorization does not work..."""
    sch = tvm.tir.Schedule(mod)
    sch.work_on('gemm')
    C_block = sch.get_block("C")
    A_cache = sch.cache_read(C_block, 0, env.inp_scope)
    B_cache = sch.cache_read(C_block, 1, env.wgt_scope)
    sch.set_scope(C_block, 0, env.acc_scope)
    I, J, i, j, K, k = sch.get_loops(C_block)
    sch.compute_at(A_cache, K)
    sch.compute_at(B_cache, K)
    sch.decompose_reduction(C_block, K)
    sch.annotate(sch.get_loops(A_cache)[-1], env.dma_copy, True)
    sch.annotate(sch.get_loops(B_cache)[-1], env.dma_copy, True)
    sch.annotate(sch.get_loops(sch.get_block("D"))[0], env.dma_copy, True)
    sch.reorder_block_iter_var(C_block, (4, 0, 1, 2, 3, 5))
    # sch.tensorize(C_block, "vta_gemm_intrin")
    return sch.mod

def vta_gemm():
    # A and B originally have shape (O, N) and (M, N) respectively. To use VTA
    # to accelerate the gemm operations we have to reshape them to
    # (o, n, BATCH, BLOCK_IN) and (m, n, BLOCK_OUT, BLOCK_IN) respectively where
    # o = O//BATCH, n = N//BLOCK_IN and m = M//BLOCK_OUT.

    O, N, M = 1, 256, 256
    o, n, m = O//env.BATCH, N//env.BLOCK_IN, M//env.BLOCK_OUT
    out_shape = (o, m, env.BATCH, env.BLOCK_OUT)

    K = te.reduce_axis((0, n), name="K") # Outer input feature reduction axis
    k = te.reduce_axis((0, env.BLOCK_IN), name="k") # Inner input feature reduction axis

    A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
    B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
    C = te.compute(out_shape,
        lambda I, J, i, j: te.sum(
            A[I, K, i, k].astype(env.acc_dtype) * B[J, K, j, k].astype(env.acc_dtype),
            axis=[K, k],
        ),
        name="C")
    D = te.compute(out_shape, lambda I, J, i, j: C(I, J, i, j).astype(env.inp_dtype), name="D")

    # bo = I = batch outer
    # co = J = channel outer
    # bi = i = batch inner
    # ci = j = channel inner
    # ko = K = reduce outer
    # ki = k = reduce inner

    gemm = te.create_prim_func([A, B, D]).with_attr({"global_symbol": "gemm"})
    mod = tvm.IRModule({"gemm": gemm})
    # mod['gemm'].show(ir_prefix="IR")

    sch = tvm.tir.Schedule(mod)
    sch.work_on('gemm')
    C_block = sch.get_block("C")
    I, J, i, j, K, k = sch.get_loops(C_block)
    sch.reorder(K, I, J, i, j, k)
    A_cache = sch.cache_read(C_block, 0, env.inp_scope)
    B_cache = sch.cache_read(C_block, 1, env.wgt_scope)
    sch.set_scope(C_block, 0, env.acc_scope)
    C_init = sch.decompose_reduction(C_block, K)
    ij = sch.fuse(i, j)
    sch.compute_at(A_cache, K)
    sch.compute_at(B_cache, K)
    sch.mod['gemm'].show(ir_prefix="IR")
    sch.annotate(sch.get_loops(A_cache)[-1], env.dma_copy, True)
    sch.annotate(sch.get_loops(B_cache)[1], env.dma_copy, True)
    sch.annotate(sch.get_loops(sch.get_block("D"))[0], env.dma_copy, True)
    sch.tensorize(ij, "vta_gemm_intrin1")
    sch.mod['gemm'].show(ir_prefix="IR")

    return sch.mod

mod = vta_gemm()
mod = vtar.get_vtar_tir_transform()(mod)
mod.show(syntax_sugar=True)
