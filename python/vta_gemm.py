# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/matrix_multiply.html
# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/optimize/matrix_multiply_opt.html

import tvm
from tvm import te
import vtar

env = vtar.get_env()

assert env.BATCH == 1
assert env.BLOCK_IN == 16
assert env.BLOCK_OUT == 16

assert env.inp_dtype == "int8"
assert env.wgt_dtype == "int8"
assert env.acc_dtype == "int32"

import tvm
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
    import numpy
    BATCH, BLOCK_OUT = 3, 2
    O, M = 3, 4
    x = numpy.arange(M)
    x = numpy.vstack((x, x, x))
    print(x)
    x = numpy.reshape(x, (O//BATCH, M//BLOCK_OUT, BATCH, BLOCK_OUT))
    x = numpy.transpose(x, (2, 3, 0, 1,)) # FIXME: this is wrong
    print(x)

def test():
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    k = te.reduce_axis((0, 128), "k")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    func = te.create_prim_func([A, B, C])
    print(func.script())

test()

def test():
    n = te.var('n')
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((m, l), name='B')
    k = te.reduce_axis((0, l), name='k')
    C = te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k))
    func = te.create_prim_func([A, B, C])
    print(func)
test()

def vta_alu():
    # A and B originally have shape (O, M). To use VTA to accellerate the alu
    # (elementwise operations) we have to reshape them to
    # (o, m, BATCH, BLOCK_OUT) where o = O//BATCH and m = M//BLOCK_OUT.
    # TODO: find how to reshape

    M, O = 1024, 1
    m, o = M//env.BLOCK_OUT, O//env.BATCH
    # o, m = te.var("o"), te.var("m")
    shape = (o, m, env.BATCH, env.BLOCK_OUT)
    # FIXME: How do I make InjectALUIntrin work when dtype=env.inp_dtype and
    # astype becomes a Cast?
    A = te.placeholder(shape, name="A", dtype=env.acc_dtype)
    B = te.placeholder(shape, name="B", dtype=env.acc_dtype)

    A_buf = te.compute(shape, lambda *i: A(*i), "A_buf")
    B_buf = te.compute(shape, lambda *i: B(*i), "B_buf")
    C_buf = te.compute(
        shape,
        lambda *i: A_buf(*i).astype(env.acc_dtype) + B_buf(*i).astype(env.acc_dtype),
        name="C_buf",
    )
    D_buf = te.compute(shape, lambda *i: C_buf(*i) >> 2, name="D_buf")

    D = te.compute(shape, lambda *i: D_buf(*i).astype(env.inp_dtype), name="D")

    # https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc
    alu = te.create_prim_func([A, B, D]).with_attr({"global_symbol": "alu"})
    Module = tvm.IRModule({"alu": alu})
    s = tvm.tir.Schedule(Module)
    s.work_on('alu')
    s.mod.show()

    s.set_scope(s.get_block("A_buf"), 0, env.acc_scope)
    s.set_scope(s.get_block("B_buf"), 0, env.acc_scope)
    s.set_scope(s.get_block("C_buf"), 0, env.acc_scope)
    s.set_scope(s.get_block("D_buf"), 0, env.acc_scope)

    s.annotate(s.get_loops(s.get_block("A_buf"))[0], env.dma_copy, True)
    s.annotate(s.get_loops(s.get_block("B_buf"))[0], env.dma_copy, True)
    s.annotate(s.get_loops(s.get_block("C_buf"))[0], env.alu, True)
    s.annotate(s.get_loops(s.get_block("D_buf"))[0], env.alu, True)
    s.annotate(s.get_loops(s.get_block("D"))[0], env.dma_copy, True)

    s.mod.show()

    # s.trace.show()

    mod = s.mod

    # https://mlc.ai/docs/reference/api/tir/transform.html
    # mod = vtar.tir.transform.InjectConv2DTransposeSkip()(mod) # TODO
    mod = vtar.tir.transform.InjectDMAIntrin()(mod)
    # mod = vtar.tir.transform.InjectSkipCopy()(mod) # Just for debug
    mod = vtar.tir.transform.AnnotateALUCoProcScope()(mod)
    # mod = tvm.tir.transform.LiftAttrScope("coproc_uop_scope")(mod) # DEPRECATED
    # mod = vtar.tir.transform.LiftAllocToScopeBegin()(mod)
    # mod = tvm.tir.transform.LiftAttrScope("coproc_scope")(mod) # DEPRECATED
    mod = vtar.tir.transform.InjectCoProcSync()(mod)
    # mod = tvm.tir.transform.StorageRewrite()(mod) # BROKEN!
    mod = vtar.tir.transform.InjectDebug(mod)
    mod = vtar.tir.transform.InjectALUIntrin()(mod)
    # mod = tvm.tir.transform.LowerDeviceStorageAccessInfo()(mod) # BROKEN!
    # mod = vtar.tir.transform.FoldUopLoop()(mod) # TODO
    # mod = vtar.tir.transform.CPUAccessRewrite()(mod) # TODO
    mod.show()
    print(tvm.tir.analysis.analysis.verify_well_formed(mod))
    return s.mod

mod = vta_alu()

import os
import numpy
rng = numpy.random.default_rng(42)
# from tvm import relax
os.environ["TVM_WIN_CC"] = "clang_wrapper.bat"
device = tvm.cpu()
ex = tvm.compile(mod)
# vm = relax.VirtualMachine(ex, device)
A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"))
B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"))
C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"))
ex(A, B, C)

raise SystemExit(0)

# Computation Declaration ######################################################

# Output channel factor m - total 16x16=256 output channels
m = 16
# Input channel factor n - total 16x16=256 input channels
n = 16
# Batch factor o (we use single batch inference)
o = 1
# A placeholder tensor in tiled data format
A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
# B placeholder tensor in tiled data format
B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
# A copy buffer
A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
# B copy buffer
B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf")

# Outer input feature reduction axis
ko = te.reduce_axis((0, n), name="ko")
# Inner input feature reduction axis
ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
# Describe the in-VTA matrix multiplication
C_buf = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda bo, co, bi, ci: te.sum(
        A_buf[bo, ko, bi, ki].astype(env.acc_dtype) * B_buf[co, ko, ci, ki].astype(env.acc_dtype),
        axis=[ko, ki],
    ),
    name="C_buf",
)

# Cast to output type, and send to main memory
C = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
)

# Scheduling the Computation ###################################################

gemm = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "gemm"})
Module = tvm.IRModule({"gemm": gemm})
s = tvm.tir.Schedule(Module)
s.work_on('gemm')

s.mod.show()

s.set_scope(s.get_block("A_buf"), 0, env.inp_scope)
s.set_scope(s.get_block("B_buf"), 0, env.wgt_scope)
s.set_scope(s.get_block("C_buf"), 0, env.acc_scope)

s.mod.show()

s.compute_at(s.get_block("A_buf"), s.get_loops(s.get_block("C_buf"))[0])
s.compute_at(s.get_block("B_buf"), s.get_loops(s.get_block("C_buf"))[0])

s.mod.show()

s.annotate(s.get_loops(s.get_block("C_buf"))[-1], env.dma_copy, True)

if False:
    s = te.create_schedule(C.op)

    # Set the intermediate tensor's scope to VTA's on-chip buffers
    s[A_buf].set_scope(env.inp_scope) # env.inp_scope: ro, shape (env.BATCH, env.BLOCK_IN), type env.inp_dtype, contains 2 ^ LOG_INP_BUFF_SIZE matrix elements
    s[B_buf].set_scope(env.wgt_scope) # env.wgt_scope: ro, shape (env.BLOCK_OUT, env.BLOCK_IN), type env.wgt_dtype, contains 2 ^ LOG_WGT_BUFF_SIZE matrix elements
    s[C_buf].set_scope(env.acc_scope) # env.acc_scope: rw, shape (env.BATCH, env.BLOCK_OUT), type env.acc_dtype, contains 2 ^ LOG_ACC_BUFF_SIZE matrix elements

    # Move buffer copy into matrix multiply loop
    s[A_buf].compute_at(s[C_buf], ko)
    s[B_buf].compute_at(s[C_buf], ko)

    # Tag the buffer copies with the DMA pragma to insert a DMA transfer
    s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
    s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
    s[C].pragma(s[C].op.axis[0], env.dma_copy)

    s[C_buf].reorder(
        ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1], s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki
    )
    s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)
