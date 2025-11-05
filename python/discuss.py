from tvm import ir
from tvm import tir
from tvm import te
from tvm.script import tir as T

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
                out[v_i, v_j] += T.Cast("int32", local_inp_buffer[v_i, v_k]) \
                    * T.Cast("int32", local_wgt_buffer[v_j, v_k])

@T.prim_func
def gemm_intrin(
        local_wgt_buffer: T.Buffer((16, 16), "int8"),
        local_inp_buffer: T.Buffer((1, 16), "int8"),
        out: T.Buffer((1, 16), "int32")
    ) -> None:
    T.func_attr({"tir.noalias": T.bool(True)})
    with T.block("root"):
        T.reads(out[0:1, 0:16], local_inp_buffer[0:1, 0:16], local_wgt_buffer[0:16, 0:16])
        T.writes(out[0:1, 0:16])
        T.evaluate(T.call_extern("int32", "VTADoTheThing",
            out.data, local_inp_buffer.data, local_wgt_buffer.data))

tir.TensorIntrin.register("gemm_intrin", gemm_desc, gemm_intrin)

BATCH, BLOCK_IN, BLOCK_OUT = 1, 16, 16
O, N, M = 1, 256, 256
o, n, m = O//BATCH, N//BLOCK_IN, M//BLOCK_OUT
out_shape = (o, m, BATCH, BLOCK_OUT)

K = te.reduce_axis((0, n), name="K")
k = te.reduce_axis((0, BLOCK_IN), name="k")

A = te.placeholder((o, n, BATCH, BLOCK_IN), name="A", dtype="int8")
B = te.placeholder((m, n, BLOCK_OUT, BLOCK_IN), name="B", dtype="int8")
C = te.compute(out_shape,
    lambda I, J, i, j: te.sum(
        A[I, K, i, k].astype("int32") * B[J, K, j, k].astype("int32"),
        axis=[K, k],
    ),
    name="C")
D = te.compute(out_shape, lambda I, J, i, j: C(I, J, i, j).astype("int8"), name="D")

gemm = te.create_prim_func([A, B, D]).with_attr({"global_symbol": "gemm"})
mod = ir.IRModule({"gemm": gemm})

sch = tir.Schedule(mod)
sch.work_on('gemm')
C_block = sch.get_block("C")
A_cache = sch.reindex_cache_read(C_block, 0, "global", lambda I, J, i, j, K, k: (I, K, i, k))
B_cache = sch.reindex_cache_read(C_block, 1, "global", lambda I, J, i, j, K, k: (J, K, j, k))
I, J, i, j, K, k = sch.get_loops(C_block)
sch.compute_at(A_cache, K)
sch.compute_at(B_cache, K)
sch.decompose_reduction(C_block, K)
sch.reorder_block_iter_var(C_block, (4, 0, 1, 2, 3, 5))
sch.mod.show()
sch.tensorize(C_block, "gemm_intrin")
