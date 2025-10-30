from tvm.script import tir as T
from tvm import tir
from tvm import te

@T.prim_func
def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

@T.prim_func
def mma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data, C.elem_offset // 256,
                A.data, A.elem_offset // 256,
                B.data, B.elem_offset // 256,
                C.data, C.elem_offset // 256,
                dtype="handle",
            )
        )

tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)

@T.prim_func
def alu_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("update"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]

@T.prim_func
def alu_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.call_extern("int32", "SomethingSomething", A.data, B.data, C.data))

tir.TensorIntrin.register("test_alu_intrin", alu_desc, alu_intrin)

def align_buffers_to_128(func: tir.PrimFunc) -> tir.PrimFunc:
    data_alignment = 128
    buffer_map = {}
    buffer_map_ = {} # FIXME: this variable has a shit name.
    for var, buffer in func.buffer_map.items():
        buffer_map[var] = tir.decl_buffer(
            buffer.shape, buffer.dtype, buffer.name, buffer.data,
            buffer.strides, buffer.elem_offset, buffer.scope, data_alignment,
            buffer.offset_factor, # buffer.buffer_type, buffer.axis_separators,
            # buffer.span
        )
        buffer_map_[buffer.name] = buffer_map[var]
    def fixup(stmt: tir.Stmt) -> tir.Stmt | None:
        # block.reads, block.writes, block.alloc_buffers is the block signature
        if isinstance(stmt, tir.BufferLoad):
            return tir.BufferLoad(buffer_map_[stmt.buffer.name], stmt.indices, stmt.predicate)
        elif isinstance(stmt, tir.BufferStore):
            return tir.BufferStore(buffer_map_[stmt.buffer.name], stmt.value, stmt.indices, stmt.predicate)
        elif isinstance(stmt, tir.Block):
            writes = [
                tir.BufferRegion(buffer_map_[write.buffer.name], write.region)
                for write in stmt.writes
            ]
            reads = [
                tir.BufferRegion(buffer_map_[read.buffer.name], read.region)
                for read in stmt.reads
            ]
            res = tir.Block(stmt.iter_vars, reads, writes, stmt.name_hint,
            stmt.body, stmt.init, stmt.alloc_buffers, stmt.match_buffers, stmt.annotations)
            return res
        else:
            return None
        return None
    body = tir.stmt_functor.ir_transform(func.body, None, fixup,
        ["tir.BufferStore", "tir.BufferLoad", "tir.Block"])
    res = tir.PrimFunc(
        func.params, body, func.ret_type, buffer_map, func.attrs, func.span
    )
    return res

A = te.placeholder((128, 128), "float32", "A")
B = te.placeholder((128, 128), "float32", "B")
k = te.reduce_axis((0, 128), "k")
C = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), "C")
te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm"})
# FIXME: for some reason this is not equivalent to the TVMScript one!
te_func = align_buffers_to_128(te_func)
print(te_func)

try:
    sch = tir.Schedule(te_func)
    C = sch.get_block("C")
    i, j, k = sch.get_loops(C)
    sch.decompose_reduction(C, i)
    i, j, k = sch.get_loops(sch.get_block("C_update"))
    i0, i1 = sch.split(i, (8, 16))
    j0, j1 = sch.split(j, (8, 16))
    k0, k1 = sch.split(k, (8, 16))
    sch.reorder(i0, j0, k0, i1, j1, k1)
    i0, j0, k0, i1, j1, k1 = sch.get_loops(sch.get_block("C_update"))
    print(sch.mod["main"].script())
    sch.tensorize(i1, "test_mma_intrin")
    print(sch.mod["main"].script())
except Exception as e:
    print(e)

@T.prim_func
def before(
        A: T.Buffer((128, 128), "float32", align=128),
        B: T.Buffer((128, 128), "float32", align=128),
        C: T.Buffer((128, 128), "float32", align=128),
    ) -> None:
    # with T.block("root")
    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] += A[vi, vk] * B[vj, vk]

# TODO: use Schedule.cache_read to schedule the computation to something
# that can be tensorized to VTA
# https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule.cache_read
# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/optimize/matrix_multiply_opt.html#sphx-glr-topic-vta-tutorials-optimize-matrix-multiply-opt-py

# TODO: the TIR pattern for padding is to use if_then_else when loading e.g.
# y[vi] = T.if_then_else(vi >= 6 and vi < 134, x[vi - 6], 0, dtype="int32")
# study how to schedule it to VTALoadBuffer2D
# https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule.decompose_padding

sch = tir.Schedule(before)
C = sch.get_block("C")
i, j, k = sch.get_loops(C)
sch.decompose_reduction(C, i)
i, j, k = sch.get_loops(sch.get_block("C_update"))
i0, i1 = sch.split(i, (8, 16))
j0, j1 = sch.split(j, (8, 16))
k0, k1 = sch.split(k, (8, 16))
sch.reorder(i0, j0, k0, i1, j1, k1)
i0, j0, k0, i1, j1, k1 = sch.get_loops(sch.get_block("C_update"))
print(sch.mod["main"].script())
sch.tensorize(i1, "test_mma_intrin")
print(sch.mod["main"].script())

@T.prim_func
def before(
        A: T.Buffer((128, 128), "float32", align=128),
        B: T.Buffer((128, 128), "float32", align=128),
        C: T.Buffer((128, 128), "float32", align=128),
    ) -> None:
    # with T.block("root")
    for i, j in T.grid(128, 128):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj], B[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tir.Schedule(before)
i, j = sch.get_loops(sch.get_block("update"))
i0, i1 = sch.split(i, (8, 16))
j0, j1 = sch.split(j, (8, 16))
sch.reorder(i0, j0, i1, j1)
i0, j0, i1, j1 = sch.get_loops(sch.get_block("update"))
sch.tensorize(i1, "test_alu_intrin")
print(sch.mod["main"].script())

################################################################################

@T.prim_func
def vta_gemm_desc(
        A: T.Buffer((16, 16), "int8"),
        B: T.Buffer((2, 16), "int8"),
        C: T.Buffer((2, 16), "int32")
    ) -> None:
    """Calculates the entries of a row vector (C) by doing a dot product of
    a row vector (B) and the rows of a matrix (A). The computation can be
    written as
        c' = b' A'
    in matrix notation or as
        c_j = b_k a_jk
    in Einstein notation.
    """

    with T.block("root"):
        T.reads(C[0:2, 0:16], B[0:2, 0:16], A[0:16, 0:16])
        T.writes(C[0:2, 0:16])
        for i, j, k in T.grid(2, 16, 16):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(C[vi, vj], B[vi, vk], A[vj, vk])
                T.writes(C[vi, vj])
                # with T.init(): C[vi, vj] = 0
                C[vi, vj] += B[vi, vk].astype("int32") * A[vj, vk].astype("int32")

@T.prim_func
def vta_gemm_intrin(
        A: T.Buffer((16, 16), "int8"),
        B: T.Buffer((2, 16), "int8"),
        C: T.Buffer((2, 16), "int32")
    ) -> None:
    T.func_attr({"tir.noalias": T.bool(True)})
    with T.block("root"):
        T.reads(A[0:16, 0:16], B[0:2, 0:16], C[0:2, 0:16])
        T.writes(C[0:2, 0:16])
        # with T.init(): T.evaluate(T.call_extern("int32", "SomethingInit", C.data))
        T.evaluate(T.call_extern("int32", "SomethingCompute", A.data, B.data, C.data))

tir.TensorIntrin.register("test_vta_gemm_intrin", vta_gemm_desc, vta_gemm_intrin)

@T.prim_func
def before(
        A: T.Buffer((128, 128), "int8"),
        B: T.Buffer((128, 128), "int8"),
        C: T.Buffer((128, 128), "int32"),
    ) -> None:
    # with T.block("root")
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            # with T.init(): C[vi, vj] = 0
            C[vi, vj] += B[vi, vk].astype("int32") * A[vj, vk].astype("int32")

sch = tir.Schedule(before)
i, j, k = sch.get_loops(sch.get_block("update"))
i0, i1 = sch.split(i, (64, 2))
j0, j1 = sch.split(j, (8, 16))
k0, k1 = sch.split(k, (8, 16))
sch.reorder(i0, j0, k0, i1, j1, k1)
sch.mod.show()
sch.tensorize(i1, "test_vta_gemm_intrin")
sch.mod.show()

@T.prim_func
def before(
        X: T.Buffer((16, 16), "int8"),
        Y: T.Buffer((2, 16), "int8"),
        Z: T.Buffer((2, 16), "int32"),
    ) -> None:
    with T.block("root"):
        T.reads(Z[0:2, 0:16], Y[0:2, 0:16], X[0:16, 0:16])
        T.writes(Z[0:2, 0:16])
        for i, j, k in T.grid(2, 16, 16):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(Z[vi, vj], Y[vi, vk], X[vj, vk])
                T.writes(Z[vi, vj])
                # with T.init(): Z[vi, vj] = 0
                Z[vi, vj] += Y[vi, vk].astype("int32") * X[vj, vk].astype("int32")

sch = tir.Schedule(before)
i, j, k = sch.get_loops(sch.get_block("update"))
#i0, i1 = sch.split(i, (16, 1))
sch.mod.show()
sch.tensorize(i, "test_vta_gemm_intrin")
sch.mod.show()

# Block Signature:
#   * iterator variables domain and binding
#         vi = T.axis.spatial(128, i0 * 2 + i1)
#         vj = T.axis.spatial(128, j0 * 16 + j1)
#         vk = T.axis.reduce(128, k0 * 16 + k1)
#     e.g. vi has spatial domain of size 128 and it is bounded to f(i0, i1)
#     where f(x, y) = x*2 + y
#   * producer consumer dependency relation
#         T.reads(C[vi_o * 2 + vi_i, vj_o * 16 + vj_i],
#                 A[vi_o*2:vi_o*2 + 2, vk_o * 16 + vk_i])
#         T.writes(C[vi_o * 2 + vi_i, vj_o * 16 + vj_i])
#     e.g. A is read (consumed) in the interval of rows from vi_o*2 to vi_o*2+2
#     and in the column at vk_o * 16 + vk_i