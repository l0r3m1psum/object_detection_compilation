from typing import Tuple

from tvm import tir
from tvm.script import tir as T

def make_vta_gemm_intrinsic(n: int) -> Tuple[tir.PrimFunc, tir.PrimFunc]:
    if n < 1:
        raise ValueError("n must be greater than 1")

    @T.prim_func
    def vta_gemm_desc(
            A: T.Buffer((16, 16), "int8"),
            B: T.Buffer((n, 16), "int8"),
            C: T.Buffer((n, 16), "int32")
        ) -> None:

        with T.block("root"):
            T.reads(C[0:n, 0:16], B[0:n, 0:16], A[0:16, 0:16])
            T.writes(C[0:n, 0:16])
            for i, j, k in T.grid(n, 16, 16):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    T.reads(C[vi, vj], B[vi, vk], A[vj, vk])
                    T.writes(C[vi, vj])
                    C[vi, vj] += B[vi, vk].astype("int32") * A[vj, vk].astype("int32")

    @T.prim_func
    def vta_gemm_intrin(
            A: T.Buffer((16, 16), "int8"),
            B: T.Buffer((n, 16), "int8"),
            C: T.Buffer((n, 16), "int32")
        ) -> None:
        T.func_attr({"tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads(A[0:16, 0:16], B[0:n, 0:16], C[0:n, 0:16])
            T.writes(C[0:n, 0:16])
            T.evaluate(T.call_extern("int32", "SomethingCompute", A.data, B.data, C.data))

    return vta_gemm_desc, vta_gemm_intrin

def test_vta_gemm_intrin(n: int) -> None:
    if n < 1:
        raise ValueError("n must be greater than 1")

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
                C[vi, vj] += B[vi, vk].astype("int32") * A[vj, vk].astype("int32")

    sch = tir.Schedule(before)
    i, j, k = sch.get_loops(sch.get_block("update"))
    i0, i1 = sch.split(i, (128//n, n))
    j0, j1 = sch.split(j, (128//16, 16))
    k0, k1 = sch.split(k, (128//16, 16))
    sch.reorder(i0, j0, k0, i1, j1, k1)
    sch.mod.show()
    sch.tensorize(i1, "test_vta_gemm_intrin%d" % n)
    sch.mod.show()

n = 4; tir.TensorIntrin.register("test_vta_gemm_intrin%d" % n, *make_vta_gemm_intrinsic(n)); test_vta_gemm_intrin(n)
n = 2; tir.TensorIntrin.register("test_vta_gemm_intrin%d" % n, *make_vta_gemm_intrinsic(n)); test_vta_gemm_intrin(n)
n = 1; tir.TensorIntrin.register("test_vta_gemm_intrin%d" % n, *make_vta_gemm_intrinsic(n)); test_vta_gemm_intrin(n)
