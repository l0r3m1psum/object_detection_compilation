import numpy
rng = numpy.random.default_rng(42)

import tvm
# TVMScript (tvm.script) is a domain specific dialect embedded in Python's AST.
# It is a convinient way to write TensorIR and Relax programs.
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
# Tensor Expresion (tvm.te) is a succint way to generate Tensor IR
from tvm import te

@T.prim_func
def main(
    A: T.Buffer((128,), "float32"),
    B: T.Buffer((128,), "float32"),
    C: T.Buffer((128,), "float32"),
) -> None:
    for i in range(128):
        with T.block("C"):
            vi = T.axis.spatial(128, i)
            C[vi] = A[vi] + B[vi]

# Implements
# Y_{i,j} = \sum_k A_{i,k} B_{k,j}
# C_{i,j} = \relu(Y_{i,j}) = \max(Y_{i,j}, 0)
@I.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

@I.ir_module
class VerboseModule:
    @T.prim_func
    def mm_relu(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    with T.block("Y"):
                        vi = T.axis.spatial(128, i)
                        vj = T.axis.spatial(128, j)
                        vk = T.axis.reduce(128, k)
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(Y[vi, vj])
                        with T.init():
                            Y[vi, vj] = T.float32(0)
                        Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i in range(128):
            for j in range(128):
                with T.block("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j)
                    T.reads(Y[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

@I.ir_module
class ConciseModule:
    @T.prim_func
    def mm_relu(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] += A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

assert tvm.ir.structural_equal(MyModule, ConciseModule)
assert tvm.ir.structural_equal(MyModule, VerboseModule)

dtype = 'float32'
@I.ir_module
class DynamicShapeModule:
    @T.prim_func
    def mm_relu(a: T.handle, b: T.handle, c: T.handle):
        # Dynamic shape definition
        M, N, K = T.int32(), T.int32(), T.int32()

        # Bind the input buffers with the dynamic shapes
        A = T.match_buffer(a, (M, K), dtype)
        B = T.match_buffer(b, (K, N), dtype)
        C = T.match_buffer(c, (M, N), dtype)
        Y = T.alloc_buffer((M, N), dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.cast(T.float32(0), dtype)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(M, N):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.cast(T.float32(0), dtype))

def evaluate_dynamic_shape(
        lib: tvm.runtime.Module,
        m: int, n: int, k: int,
    ) -> numpy.ndarray:
    A = tvm.nd.array(rng.uniform(size=(m, k)).astype("float32"))
    B = tvm.nd.array(rng.uniform(size=(k, n)).astype("float32"))
    C = tvm.nd.array(numpy.empty((m, n), dtype="float32"))
    lib(A, B, C)
    return C.numpy()

if False:
    dyn_shape_lib = tvm.compile(DynamicShapeModule, target="llvm")
    mmmul_relu_small = evaluate_dynamic_shape(dyn_shape_lib, m=4, n=4, k=4)
    mmmul_relu_big = evaluate_dynamic_shape(dyn_shape_lib, m=64, n=64, k=128)
    print(mmmul_relu_small)
    print(mmmul_relu_big)

def create_te_module() -> tvm.IRModule:
    A = te.placeholder((128, 128), 'float32', name='A')
    B = te.placeholder((128, 128), 'float32', name='B')
    k = te.reduce_axis((0, 128), 'k')
    # i and j are automatically mapped to spatial axis
    Y = te.compute(
        (128, 128),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name='Y'
    )
    C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name='C')
    te_func = te.create_prim_func((A, B, C)).with_attr({"global_symbol": "mm_relu"})
    # Tensor Expression automatically includes this attribute, removing it is a
    # pessimization but it makes the module equivalent to the other modules.
    te_func = te_func.without_attr('tir.noalias')
    TEModule = tvm.IRModule({'mm_relu': te_func})
    return TEModule

TEModule = create_te_module()
assert tvm.ir.structural_equal(MyModule, TEModule)

def create_dynamic_te_module() -> tvm.IRModule:
    M, N, K = te.var("M"), te.var("N"), te.var("K")
    A = te.placeholder((M, K), "float32", name="A")
    B = te.placeholder((K, N), "float32", name="B")
    k = te.reduce_axis((0, K), "k")
    Y = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
    C = te.compute((M, N), lambda i, j: te.max(Y[i, j], 0), name="C")

    dyn_te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
    dyn_te_func = dyn_te_func.without_attr('tir.noalias')
    DynamicTEModule = tvm.IRModule({"mm_relu": dyn_te_func})
    return DynamicTEModule

DynamicTEModule = create_dynamic_te_module()
assert tvm.ir.structural_equal(DynamicShapeModule, DynamicTEModule)

def evaluate(mod: tvm.IRModule) -> tvm.runtime.module.BenchmarkResult:
    a_numpy = rng.uniform(size=(128, 128)).astype('float32')
    b_numpy = rng.uniform(size=(128, 128)).astype('float32')
    c_numpy = a_numpy @ b_numpy

    a_tvm = tvm.nd.array(a_numpy)
    b_tvm = tvm.nd.array(b_numpy)
    c_tvm = tvm.nd.array(numpy.empty((128, 128), dtype='float32'))

    lib = tvm.tir.build(mod, target='llvm')
    lib(a_tvm, b_tvm, c_tvm)
    numpy.testing.assert_allclose(c_tvm.numpy(), c_numpy, rtol=1e-5)

    f_timer = lib.time_evaluator('mm_relu', tvm.cpu())
    return f_timer(a_tvm, b_tvm, c_tvm)

# print(evaluate(MyModule))

import warnings
warnings.filterwarnings("ignore")

sch = tvm.tir.Schedule(MyModule)
sch.work_on('mm_relu')
sch.mod.show()

block_Y = sch.get_block('Y')
i, j, k = sch.get_loops(block_Y) # Loop tiling
j0, j1 = sch.split(j, factors=[None, 8])
sch.mod.show()

sch.reorder(j0, k, j1)
sch.mod.show()
# print(evaluate(sch.mod))

block_C = sch.get_block("C")
sch.reverse_compute_at(block_C, j0)
sch.mod.show()

# segregate the initialization of Y’s elements from the reduction
sch.decompose_reduction(block_Y, k)
sch.mod.show()

print(evaluate(sch.mod))

sch.trace.show()
