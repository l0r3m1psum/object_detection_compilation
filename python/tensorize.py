from tvm.script import tir as T
from tvm import tir

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

@T.prim_func
def before(
        A: T.Buffer((128, 128), "float32", align=128),
        B: T.Buffer((128, 128), "float32", align=128),
        C: T.Buffer((128, 128), "float32", align=128),
    ) -> None:
    # with T.block("root")
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            # with T.init(): C[vi, vj] = T.float32(0)
            C[vi, vj] += A[vi, vk] * B[vj, vk]

sch = tir.Schedule(before)
i, j, k = sch.get_loops(sch.get_block("update"))
i0, i1 = sch.split(i, (8, 16))
j0, j1 = sch.split(j, (8, 16))
k0, k1 = sch.split(k, (8, 16))
sch.reorder(i0, j0, k0, i1, j1, k1)
i0, j0, k0, i1, j1, k1 = sch.get_loops(sch.get_block("update"))
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