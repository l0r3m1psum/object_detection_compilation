# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/matrix_multiply.html
# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/optimize/matrix_multiply_opt.html

import tvm
from tvm import te
import ctypes
vta_fsim = ctypes.CDLL("vta_fsim")
import numpy
import vtar

env = vtar.get_env()

assert env.TARGET == "sim"

assert env.BATCH == 1
assert env.BLOCK_IN == 16
assert env.BLOCK_OUT == 16

assert env.inp_dtype == "int8"
assert env.wgt_dtype == "int8"
assert env.acc_dtype == "int32"

rng = numpy.random.default_rng(42)

target = tvm.target.Target(env.target, host=env.target_host)
dev = tvm.device(str(env.target))

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
    mod = tvm.IRModule({"alu": alu})

    sch = tvm.tir.Schedule(mod)
    sch.work_on('alu')

    C_block = sch.get_block("C")
    A_cache = sch.cache_read(C_block, 0, env.acc_scope)
    B_cache = sch.cache_read(C_block, 1, env.acc_scope)
    sch.set_scope(C_block, 0, env.acc_scope)
    sch.annotate(sch.get_loops(A_cache)[0], env.dma_copy, True)
    sch.annotate(sch.get_loops(B_cache)[0], env.dma_copy, True)
    sch.annotate(sch.get_loops(C_block)[0], env.alu, True)
    sch.annotate(sch.get_loops(sch.get_block("D"))[0], env.dma_copy, True)

    mod = sch.mod
    # This optimization is done automatically by tir.transform.StorageRewrite
    # mod = vtar.tir.transform.ReplaceVarOcurrence("C_local.acc_buffer", "A_local.acc_buffer")(mod)
    return mod

mod = vta_alu()
mod['alu'].show()

ex = tvm.tir.build(mod, target, vtar.get_vtar_tir_transform())
A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
ex(A, B, C)
numpy.testing.assert_equal(C.numpy(), A.numpy() + B.numpy())
# print(C)

if False:
    from tvm import rpc
    host = "192.168.137.48"
    port = 9091
    remote = rpc.connect(host, port)
    ex.export_library("build/alu.tar")
    remote.upload("build/alu.tar")
    func = remote.load_module("alu.tar")
    dev = remote.ext_dev(0)
    A = tvm.nd.empty((1, 64, 1, 16), "int32", dev)
    B = tvm.nd.empty((1, 64, 1, 16), "int32", dev)
    C = tvm.nd.empty((1, 64, 1, 16), "int8", dev)
    func(A, B, C)

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
    sch.annotate(sch.get_loops(A_cache)[-1], env.dma_copy, True)
    sch.annotate(sch.get_loops(B_cache)[1], env.dma_copy, True)
    sch.annotate(sch.get_loops(sch.get_block("D"))[0], env.dma_copy, True)
    sch.tensorize(ij, "vta_gemm_intrin1")
    I_init, J_init, i_init, j_init = sch.get_loops(C_init)
    ij_init = sch.fuse(i_init, j_init)
    sch.tensorize(ij_init, "vta_init_intrin1")

    return sch.mod

mod = vta_gemm()
mod['gemm'].show(ir_prefix="IR")

O, N, M = 1, 256, 256
o, n, m = O//env.BATCH, N//env.BLOCK_IN, M//env.BLOCK_OUT
A_orig = rng.integers(-128, 128, (O, N)).astype(env.inp_dtype)
B_orig = rng.integers(-128, 128, (M, N)).astype(env.wgt_dtype)
C_orig = (A_orig.astype(env.acc_dtype) @ B_orig.T.astype(env.acc_dtype)).astype(env.out_dtype)
A_pack = A_orig.reshape(o, env.BATCH, n, env.BLOCK_IN).transpose((0, 2, 1, 3))
B_pack = B_orig.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose((0, 2, 1, 3))
C_pack = C_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))

ex = tvm.tir.build(mod, target, vtar.get_vtar_tir_transform())
A = tvm.nd.array(A_pack, dev)
B = tvm.nd.array(B_pack, dev)
C = tvm.nd.array(numpy.zeros((o, m, env.BATCH, env.BLOCK_OUT), dtype=env.out_dtype), dev)
ex(A, B, C)
numpy.testing.assert_equal(C.numpy(), C_pack)
# https://tvm.hyper.ai/docs/0.12.0/topic/vta/tutorials/mat_mul

# FIXME: how do I put data in the NDArrays???
# TODO: export one and load from another file
if False:
    ex.export_library("build/gemm.tar")
    remote.upload("build/gemm.tar")
    func = remote.load_module("gemm.tar")
    dev = remote.ext_dev(0)
    A = tvm.nd.empty((1, 16, 1, 16), "int8", dev)
    B = tvm.nd.empty((16, 16, 16, 16), "int8", dev)
    C = tvm.nd.empty((1, 16, 1, 16), "int8", dev).copyfrom(numpy.zeros((1, 16, 1, 16), dtype='int8'))
    # tmp, tmp_shape = tvm.nd.numpyasarray(numpy.zeros((1, 16, 1, 16), dtype='int8'))
    # breakpoint()
    # tmp.copyto(C)
    func(A, B, C)

def h():
    # FIXME: this explanation is wrong.

    # Say we configure VTA to have ACC_BUFF_SIZE//32 == 1024, we cannot store
    # two 1x1024 matrices in its memory to add them together hence we have to
    # perform the computation piecewise slitting the two matrices in blocks of
    # suitable size. In general if ACC_BUFF_SIZE//32 == ω*μ*BATCH*BLOCK_OUT to
    # perform (o/ω) * (m/μ) ALU operations loading two tensors of shape
    # (ω, μ/2, BATCH, BLOCK_OUT) at the time.

    BATCH, BLOCK_OUT = 1, 16
    O, M = 1, 1024
    o, m = O//BATCH, M//BLOCK_OUT
    ω, μ = 1, 64

    A_orig = numpy.arange(O*M).reshape((O, M)).astype("int32")
    B_orig = numpy.arange(O*M).reshape((O, M)).astype("int32")
    C_orig = (A_orig+B_orig).astype("int8")

    A_pack = A_orig.reshape(o, BATCH, m, BLOCK_OUT).transpose((0, 2, 1, 3))
    B_pack = B_orig.reshape(o, BATCH, m, BLOCK_OUT).transpose((0, 2, 1, 3))
    C_pack = C_orig.reshape(o, BATCH, m, BLOCK_OUT).transpose((0, 2, 1, 3))

    import itertools
    def grid(*ns):
        for i in itertools.product(*(range(n) for n in ns)):
            print(i)
        return itertools.product(*(range(n) for n in ns))

    acc_buff = numpy.zeros((ω, μ, BATCH, BLOCK_OUT))
    res = numpy.zeros_like(C_pack)
    for boo, coo in grid(o//ω, m//(μ//2)):
        bos = slice((o//ω)*boo, (o//ω)*(boo+1)) # batch outer slice
        cos = slice((μ//2)*coo, (μ//2)*(coo+1)) # channel outer slice
        acc_buff[:, :μ//2, :, :] = A_pack[bos, cos, :, :] # Load
        acc_buff[:, μ//2:, :, :] = B_pack[bos, cos, :, :] # Load
        acc_buff[:, :μ//2, :, :] += acc_buff[:, μ//2:, :, :] # ALU
        res[bos, cos, :, :] = acc_buff[:, :μ//2, :, :].astype('int8') # Store
    numpy.testing.assert_equal(res, C_pack)
h()

def vta_alu_blocked():
    O, M = 1, 1024
    o, m = O//env.BATCH, M//env.BLOCK_OUT
    shape = (o, m, env.BATCH, env.BLOCK_OUT)
    ω, μ = 1, 64

    A = te.placeholder(shape, name="A", dtype=env.acc_dtype)
    B = te.placeholder(shape, name="B", dtype=env.acc_dtype)
    C = te.compute(shape, lambda bo, co, bi, ci: A(bo, co, bi, ci) + B(bo, co, bi, ci), "C")
    D = te.compute(shape, lambda bo, co, bi, ci: C(bo, co, bi, ci).astype(env.out_dtype), "D")

    alu = te.create_prim_func([A, B, D]).with_attr({"global_symbol": "alu"})
    mod = tvm.IRModule({"alu": alu})

    sch = tvm.tir.Schedule(mod)
    sch.work_on('alu')

    # Loop names:
    #   bo: batch_outer    |  boo: batch_outer_outer
    #   co: channel_outer  |  coo: channel_outer_outer
    #   bi: batch_inner    |  boi: batch_outer_inner
    #   ci: channel_inner  |  coi: channel_outer_inner

    C_block = sch.get_block("C")
    bo, co, bi, ci = sch.get_loops(C_block)
    boo, boi = sch.split(bo, (1, 1//1))
    coo, coi = sch.split(co, (2, 64//2))
    sch.reorder(boo, coo, boi, coi, bi, ci)
    D_block = sch.get_block("D")
    bo_, co_, bi_, ci_ = sch.get_loops(D_block)
    boo_, boi_ = sch.split(bo_, (1, 1//1))
    coo_, coi_ = sch.split(co_, (2, 64//2))
    sch.reorder(boo_, coo_, boi_, coi_, bi_, ci_)
    A_cache = sch.cache_read(C_block, 0, env.acc_scope)
    B_cache = sch.cache_read(C_block, 1, env.acc_scope)
    sch.compute_at(A_cache, coo, preserve_unit_loops=True)
    sch.compute_at(B_cache, coo, preserve_unit_loops=True)
    sch.merge(coo, coo_)
    sch.set_scope(C_block, 0, env.acc_scope)
    sch.annotate(sch.get_loops(A_cache)[2], env.dma_copy, True)
    sch.annotate(sch.get_loops(B_cache)[2], env.dma_copy, True)
    sch.annotate(sch.get_loops(C_block)[2], env.alu, True)
    sch.annotate(sch.get_loops(D_block)[2], env.dma_copy, True)
    return sch.mod

mod = vta_alu_blocked()
mod['alu'].show()

ex = tvm.tir.build(mod, target, vtar.get_vtar_tir_transform())
A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
ex(A, B, C)
numpy.testing.assert_equal(C.numpy(), A.numpy() + B.numpy())
