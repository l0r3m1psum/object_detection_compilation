import tvm
from tvm import testing
from tvm import te, tir, relax, ir

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import vtar.relax.transform
import numpy
import ctypes

# Needed to register the ext_dev
vta_fsim = ctypes.CDLL("vta_fsim")
env = vtar.get_env()
target = tvm.target.Target(env.target, host=env.target_host)
dev = tvm.device(str(env.target))
rng = numpy.random.default_rng(42)

# Scheduling tests #############################################################

def test_blocked_load_store():
    BATCH, BLOCK_OUT = 1, 16
    shape = (16, 16, BATCH, BLOCK_OUT)
    A = te.placeholder(shape, env.acc_dtype, "A")
    B = te.compute(shape, lambda bo, co, bi, ci: A(bo, co, bi, ci).astype(env.out_dtype), "B")
    f = te.create_prim_func((A, B))

    sch = tir.Schedule(f)
    block = sch.get_block("B")
    bo, co, bi, ci = sch.get_loops(block)
    boo, boi = sch.split(bo, (None, 4))
    coo, coi = sch.split(co, (None, 4))
    sch.reorder(boo, coo, boi, coi, bi, ci)
    cache = sch.cache_read(block, 0, env.acc_scope)
    sch.compute_at(cache, coo)
    sch.annotate(sch.get_loops(cache)[2], env.dma_copy, 0)
    sch.annotate(sch.get_loops(block)[2], env.dma_copy, 0)

    mod = sch.mod
    ex = tir.build(mod, target, vtar.get_vtar_tir_transform())
    A = tvm.nd.array((rng.uniform(size=shape)*10).astype(env.acc_dtype), dev)
    B = tvm.nd.array(numpy.zeros(shape, dtype=env.out_dtype), dev)
    ex(A, B)
    numpy.testing.assert_equal(B.numpy(), A.numpy().astype(env.out_dtype))

def test_simple_vta_alu():
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
    D = te.compute(shape, lambda *i: C(*i).astype(env.out_dtype), "D")

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

    ex = tvm.tir.build(mod, target, vtar.get_vtar_tir_transform())
    A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
    ex(A, B, C)
    numpy.testing.assert_equal(C.numpy(), A.numpy() + B.numpy())

def test_simple_vta_gemm():
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
    # K, I, J order is the outer product formulation of matrix multiplication.
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

    mod = sch.mod

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

def test_blocked_vta_alu():
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

    mod = sch.mod

    ex = tvm.tir.build(mod, target, vtar.get_vtar_tir_transform())
    A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
    ex(A, B, C)
    numpy.testing.assert_equal(C.numpy(), A.numpy() + B.numpy())

# python python/test_vta_compiler.py -s -k test_blocked_vta_gemm
def test_blocked_vta_gemm():
    # NOTE: this is a gemv
    batch_size = 1
    in_chann = 1024
    out_chann = 1024
    assert batch_size % env.BATCH == 0
    assert in_chann % env.BLOCK_IN == 0
    assert out_chann % env.BLOCK_OUT == 0
    blocked_batch_size = batch_size // env.BATCH
    blocked_in_chann = in_chann // env.BLOCK_IN
    blocked_out_chann = out_chann // env.BLOCK_OUT

    # data is a "big vector" and we want to multiply it with weight which is a
    # "big matrix"
    data_shape = (blocked_batch_size, blocked_in_chann, env.BATCH, env.BLOCK_IN)
    weight_shape = (blocked_out_chann, blocked_in_chann, env.BLOCK_OUT, env.BLOCK_IN)
    output_shape = (blocked_batch_size, blocked_out_chann, env.BATCH, env.BLOCK_OUT)

    ro = te.reduce_axis((0, blocked_in_chann), "ro")  # reduction outer
    ri = te.reduce_axis((0, env.BLOCK_IN), "ri") # reduction inner

    data = te.placeholder(data_shape, env.inp_dtype, "data")
    weight = te.placeholder(weight_shape, env.wgt_dtype, "weight")

    res_gemm = te.compute(
        output_shape,
        lambda bo, co, bi, ci: te.sum(
            data[bo, ro, bi, ri].astype(env.acc_dtype)
            * weight[co, ro, ci, ri].astype(env.acc_dtype),
            axis=[ro, ri],
        ),
        name="res_gemm",
    )

    inp_max = (1 << (env.INP_WIDTH - 1)) - 1
    res_shr = te.compute(output_shape, lambda bo, co, bi, ci: res_gemm(bo, co, bi, ci) >> env.INP_WIDTH, "res_shr")
    res_max = te.compute(output_shape, lambda bo, co, bi, ci: te.max(res_shr(bo, co, bi, ci), 0), "res_max")
    res_min = te.compute(output_shape, lambda bo, co, bi, ci: te.min(res_max(bo, co, bi, ci), inp_max), "res_min")
    res = te.compute(output_shape, lambda bo, co, bi, ci: res_min(bo, co, bi, ci).astype(env.inp_dtype), "res")

    gemm = te.create_prim_func([data, weight, res]).with_attr({"global_symbol": "gemm"})

    batch_block = 1 // env.BATCH
    input_block = 256 // env.BLOCK_IN
    output_block = 256 // env.BLOCK_OUT

    sch = tir.Schedule(gemm)
    gemm_block = sch.get_block("res_gemm")

    bo, co, bi, ci, ro, ri = sch.get_loops(gemm_block)
    boo, boi = sch.split(bo, (None, batch_block))
    coo, coi = sch.split(co, (None, output_block))
    roo, roi = sch.split(ro, (None, input_block))
    sch.reorder(boo, coo, roo, roi, boi, coi, bi, ci, ri)

    data_cache = sch.cache_read(gemm_block, 0, env.inp_scope)
    weight_cache = sch.cache_read(gemm_block, 1, env.wgt_scope)
    gemm_init = sch.decompose_reduction(gemm_block, roo)
    sch.compute_at(data_cache, coo)
    sch.compute_at(weight_cache, coo)
    sch.reverse_compute_at(sch.get_block("res_shr"), boo)
    sch.reverse_compute_at(sch.get_block("res_max"), boo)
    sch.reverse_compute_at(sch.get_block("res_min"), boo)
    sch.reverse_compute_at(sch.get_block("res"), boo)

    sch.mod["main"].show()


# Relax tests ##################################################################

def test_trivial_graphpack():
    # H=56, W=56, I=64, O=64, kH=3, kW=3
    @I.ir_module
    class ConvModel:
        @R.function
        def main(
            x:            R.Tensor((1,  64, 56, 56), dtype="int8"),
            conv1_weight: R.Tensor((64, 64, 3,  3),  dtype="int8"),
            conv1_bias:   R.Tensor((1,  64, 1,  1),  dtype="int32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                mp1 = R.nn.max_pool2d(x)
                conv1 = R.nn.conv2d(mp1, conv1_weight, strides=1, padding=1, dilation=1,
                    out_dtype="int32")
                add1 = R.add(conv1, conv1_bias)
                avg1 = R.nn.avg_pool2d(add1)
                gv = avg1
                R.output(gv)
            return gv

    # TODO: check that the dimension factoring is correct, why in
    # https://tvm.apache.org/docs/v0.16.0/topic/vta/tutorials/optimize/convolution_opt.html#sphx-glr-topic-vta-tutorials-optimize-convolution-opt-py
    # they use also BLOCK_OUT
    # To interpret look at test_benchmark_topi_conv2d.py:run_conv2d
    # H=56, W=56, I=64, O=64, kH=3, kW=3
    @I.ir_module
    class ConvModelPacked:
        @R.function
        def main(
                x: R.Tensor((1, 64, 56, 56), dtype="int8"),
                conv1_weight: R.Tensor((64, 64, 3, 3), dtype="int8"),
                conv1_bias: R.Tensor((1, 64, 1, 1), dtype="int32")
            ) -> R.Tensor((1, 64, 56, 56), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.nn.max_pool2d(x)
                lv1 = R.reshape(lv, R.shape([1, 1, 4, 16, 56, 56]))
                #             (1//BATCH,      64//BLOCK_IN, 56, 56, BATCH,     BLOCK_IN)
                lv2: R.Tensor((1//1,   64//16, 56, 56, 1,  16), dtype="int8") = R.permute_dims(lv1, axes=[0, 2, 4, 5, 1, 3])

                lv3 = R.reshape(conv1_weight, R.shape([4, 16, 4, 16, 3, 3]))
                #             (64//BLOCK_IN,  64//BLOCK_IN, 3,  3,  BLOCK_IN,  BLOCK_IN)
                lv4: R.Tensor((64//16, 64//16, 3,  3,  16, 16), dtype="int8") = R.permute_dims(lv3, axes=[0, 2, 4, 5, 1, 3])
                lv5 = R.nn.conv2d(lv2, lv4, strides=1, padding=1, dilation=1, data_layout="NCHW1n16c", kernel_layout="OIHW16o16i", out_layout="NCHW1n16c", out_dtype="int32")

                lv6 = R.reshape(conv1_bias, R.shape([1, 1, 4, 16, 1, 1]))
                #             (1,             64//BLOCK_IN, 1,  1,  1,         BLOCK_IN)
                lv7: R.Tensor((1,      64//16, 1,  1,  1,  16), dtype="int32") = R.permute_dims(lv6, axes=[0, 2, 4, 5, 1, 3])
                lv8 = R.add(lv5, lv7)

                lv9 = R.permute_dims(lv8, axes=[0, 4, 1, 5, 2, 3])
                lv10 = R.reshape(lv9, R.shape([1, 64, 56, 56]))
                gv = R.nn.avg_pool2d(lv10)

                R.output(gv)
            return gv

    ir.assert_structural_equal(
        ir.transform.Sequential([
            vtar.relax.transform.GraphPack(),
            relax.transform.CanonicalizeBindings(), # removes redundant assignments
        ])(ConvModel),
        ConvModelPacked
    )

def test_trivial_remove_unnecessary_dequantize_quantize_wrapping():
    @I.ir_module
    class DequantReshapeQuant:
        @R.function
        def main(x: R.Tensor((1, 8), dtype="int8")):
            with R.dataflow():
                lv = R.dequantize(x, R.const(1.0), R.const(1, dtype="int8"))
                lv1 = R.reshape(lv, R.shape((-1,)))
                lv2 = R.quantize(lv1, R.const(1.0), R.const(1, dtype="int8"))
                gv = lv2
                R.output(gv)
            return gv

    @I.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor((1, 8), dtype="int8")):
            with R.dataflow():
                lv = R.reshape(x, R.shape([8]))
                gv = lv
                R.output(gv)
            return gv

    # TODO: DequantMaxpool2DQuant -> Maxpool2D

    tvm.ir.assert_structural_equal(
        vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping()(DequantReshapeQuant),
        Reshape
    )

if __name__ == '__main__':
    testing.main()
