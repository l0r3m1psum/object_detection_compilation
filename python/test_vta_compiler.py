import tvm
from tvm import testing
from tvm import te, tir, relax, ir, dlight as dl, topi, arith
import tvm.topi.testing

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import onnx

import vtar.relax.transform
import vtar.relax.frontend.onnx
import numpy
import ctypes
import pytest
import warnings
import os

if onnx.__version__ != "1.18.0":
    warnings.warn("The version of ONNX is different from 1.18.0 which is the "
        "one tested to work.")

# Needed to register the ext_dev
vta_fsim = ctypes.CDLL("vta_fsim")
env = vtar.get_env()
target = tvm.target.Target(env.target, host=env.target_host)
dev = tvm.device(str(env.target))
rng = numpy.random.default_rng(42)

assert \
    (env.INP_BUFF_SIZE//8, env.WGT_BUFF_SIZE//(8*8),
        env.ACC_BUFF_SIZE//32, env.OUT_BUFF_SIZE//8) == (4096, 4096, 4096, 4096)
assert 4096//16 == 256
# (1, 128, 1, 16) we can load at most two of this in ACC_BUFF, hence anything
# bigger needs to be split

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
    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())
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

    sch = tir.Schedule(mod)
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

    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())
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

    sch = tir.Schedule(mod)
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

    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())
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

    sch = tir.Schedule(mod)
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

    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())
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
    res_shr_block = sch.get_block("res_shr")
    res_max_block = sch.get_block("res_max")
    res_min_block = sch.get_block("res_min")
    res_block = sch.get_block("res")

    bo, co, bi, ci, ro, ri = sch.get_loops(gemm_block)
    boo, boi = sch.split(bo, (None, batch_block))
    coo, coi = sch.split(co, (None, output_block))
    roo, roi = sch.split(ro, (None, input_block))
    sch.reorder(boo, coo, roo, roi, boi, coi, bi, ci, ri)

    data_cache = sch.cache_read(gemm_block, 0, env.inp_scope)
    weight_cache = sch.cache_read(gemm_block, 1, env.wgt_scope)
    # decompose_reduction works only if the index is the outermost reduction
    # index or an index after that.
    gemm_init = sch.decompose_reduction(gemm_block, roo)
    sch.compute_at(data_cache, roo)
    sch.compute_at(weight_cache, roo)

    sch.reverse_compute_at(res_shr_block, coo)
    sch.reverse_compute_at(res_max_block, coo)
    sch.reverse_compute_at(res_min_block, coo)
    sch.reverse_compute_at(res_block, coo)

    sch.set_scope(res_shr_block, 0, env.acc_scope)
    sch.set_scope(res_max_block, 0, env.acc_scope)
    sch.set_scope(res_min_block, 0, env.acc_scope)

    sch.annotate(sch.get_loops(res_shr_block)[-2], env.alu, 0)
    sch.annotate(sch.get_loops(res_max_block)[-2], env.alu, 0)
    sch.annotate(sch.get_loops(res_min_block)[-2], env.alu, 0)

    sch.set_scope(gemm_block, 0, env.acc_scope)
    bici = sch.fuse(bi, ci)
    sch.tensorize(bici, "vta_gemm_intrin1")

    _, _, I_init, J_init, i_init, j_init = sch.get_loops(gemm_init)
    ij_init = sch.fuse(i_init, j_init)
    sch.tensorize(ij_init, "vta_init_intrin1")

    sch.annotate(sch.get_loops(data_cache)[-2], env.dma_copy, 0)
    sch.annotate(sch.get_loops(weight_cache)[-4], env.dma_copy, 0)
    sch.annotate(sch.get_loops(res_block)[-2], env.dma_copy, 0)

    mod = sch.mod

    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())

    data_np = rng.integers(-128, 128, size=(batch_size, in_chann)).astype(data.dtype)
    weight_np = rng.integers(-128, 128, size=(out_chann, in_chann)).astype(weight.dtype)

    data_packed = data_np.reshape(
        blocked_batch_size, env.BATCH, blocked_in_chann, env.BLOCK_IN
    ).transpose((0, 2, 1, 3))
    weight_packed = weight_np.reshape(
        blocked_out_chann, env.BLOCK_OUT, blocked_in_chann, env.BLOCK_IN
    ).transpose((0, 2, 1, 3))

    data_nd = tvm.nd.array(data_packed, dev)
    weight_nd = tvm.nd.array(weight_packed, dev)
    res_nd = tvm.nd.array(numpy.zeros(output_shape).astype(res.dtype), dev)

    ex(data_nd, weight_nd, res_nd)

    res_ref = numpy.dot(data_np.astype(env.acc_dtype), weight_np.T.astype(env.acc_dtype))
    res_ref = res_ref >> env.INP_WIDTH
    res_ref = numpy.clip(res_ref, 0, inp_max)
    res_ref = res_ref.astype(res.dtype)
    res_ref = res_ref.reshape(
        blocked_batch_size, env.BATCH, blocked_out_chann, env.BLOCK_OUT
    ).transpose((0, 2, 1, 3))
    numpy.testing.assert_equal(res_ref, res_nd.numpy())

def test_blocked_vta_conv2d():
    batch_size = 1
    height = 14
    width = 14
    in_channels = 256
    out_channels = 256
    kernel_h = 3
    kernel_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    assert batch_size % env.BATCH == 0
    assert in_channels % env.BLOCK_IN == 0
    assert out_channels % env.BLOCK_OUT == 0
    
    data_shape = ( # NCHW1n16c
        batch_size // env.BATCH,
        in_channels // env.BLOCK_IN,
        height,
        width,
        env.BATCH,
        env.BLOCK_IN,
    )
    kernel_shape = ( # OIHW16o16i
        out_channels // env.BLOCK_OUT,
        in_channels // env.BLOCK_IN,
        kernel_h,
        kernel_w,
        env.BLOCK_OUT,
        env.BLOCK_IN,
    )
    fout_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    fout_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    output_shape = ( # NOHW1n16o
        batch_size // env.BATCH,
        out_channels // env.BLOCK_OUT,
        fout_height,
        fout_width,
        env.BATCH,
        env.BLOCK_OUT,
    )
    
    dy = te.reduce_axis((0, kernel_h), "dy")
    dx = te.reduce_axis((0, kernel_w), "dx")
    ic = te.reduce_axis((0, in_channels // env.BLOCK_IN), "ic")
    ic_tns = te.reduce_axis((0, env.BLOCK_IN), "ic_tns")
    
    data = te.placeholder(data_shape, env.inp_dtype, "data")
    kernel = te.placeholder(kernel_shape, env.wgt_dtype, "kernel")
    
    data_buf = topi.nn.pad(data, (0, 0, pad_h, pad_w, 0, 0), name="data_buf")
    res_conv = te.compute(
        output_shape,
        lambda bo, co, i, j, bi, ci: te.sum(
            data_buf[bo, ic, i * stride_h + dy, j * stride_w + dx, bi, ic_tns].astype(env.acc_dtype)
            * kernel[co, ic, dy, dx, ci, ic_tns].astype(env.acc_dtype),
            axis=[ic, dy, dx, ic_tns],
        ),
        "res_conv",
    )
    inp_max = (1 << (env.INP_WIDTH - 1)) - 1
    res_shr = te.compute(output_shape, lambda bo, co, i, j, bi, ci: res_conv(bo, co, i, j, bi, ci) >> 8, "res_shr")
    res_max = te.compute(output_shape, lambda bo, co, i, j, bi, ci: te.max(res_shr(bo, co, i, j, bi, ci), 0), "res_max")
    res_min = te.compute(output_shape, lambda bo, co, i, j, bi, ci: te.min(res_max(bo, co, i, j, bi, ci), inp_max), "res_min")
    res = te.compute(output_shape, lambda bo, co, i, j, bi, ci: res_min(bo, co, i, j, bi, ci).astype(env.inp_dtype), "res")
    
    conv2d = te.create_prim_func((data, kernel, res))
    
    sch = tir.Schedule(conv2d)
    res_conv_block = sch.get_block("res_conv")
    res_shr_block = sch.get_block("res_shr")
    res_max_block = sch.get_block("res_max")
    res_min_block = sch.get_block("res_min")
    res_block = sch.get_block("res")
    kernel_cache = sch.cache_read(res_conv_block, 1, env.wgt_scope)
    data_cache = sch.get_block("data_buf")
    
    sch.mod["main"].show()

    b_block = 1 // env.BATCH
    oc_block = 128 // env.BLOCK_OUT
    h_block = 7
    w_block = 14

    b, oc, y, x, b_tns, oc_tns = sch.get_loops(res_block)
    b_out, b_inn = sch.split(b, (None, b_block))
    oc_out, oc_inn = sch.split(oc, (None, oc_block))
    y_out, y_inn = sch.split(y, (None, h_block))
    x_out, x_inn = sch.split(x, (None, w_block))
    sch.reorder(b_out, oc_out, y_out, x_out, b_inn, oc_inn, y_inn, x_inn, b_tns, oc_tns)

    sch.compute_at(res_min_block, x_out, preserve_unit_loops=True)
    sch.compute_at(res_max_block, x_out, preserve_unit_loops=True)
    sch.compute_at(res_shr_block, x_out, preserve_unit_loops=True)
    sch.compute_at(res_conv_block, x_out, preserve_unit_loops=True)

    ic_block = 16 // env.BLOCK_IN

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
    _, tx = sch.split(oc_out, (None, v_threads))
    sch.reorder(tx, b_out)
    # This can't be used because InjectVirtualThread doubles memory usage
    # (because v_threads=2) and than StorageRewrite fails because of the
    # MemoryInfo node.
    # sch.bind(tx, "vthread.x")

    conv_init = sch.decompose_reduction(res_conv_block, ic_out)

    sch.mod["main"].show()

    sch.compute_at(data_cache, ic_out)
    sch.compute_at(kernel_cache, ic_out)

    sch.set_scope(data_cache, 0, env.inp_scope)
    sch.set_scope(res_conv_block, 0, env.acc_scope)
    sch.set_scope(res_shr_block, 0, env.acc_scope)
    sch.set_scope(res_max_block, 0, env.acc_scope)
    sch.set_scope(res_min_block, 0, env.acc_scope)

    sch.annotate(sch.get_loops(data_cache)[-3], env.dma_copy, 0)
    sch.annotate(sch.get_loops(kernel_cache)[-5], env.dma_copy, 0)
    sch.annotate(sch.get_loops(res_block)[-4], env.dma_copy, 0)
    sch.annotate(sch.get_loops(res_shr_block)[-6], env.alu, 0)
    sch.annotate(sch.get_loops(res_max_block)[-6], env.alu, 0)
    sch.annotate(sch.get_loops(res_min_block)[-6], env.alu, 0)

    init_loops = sch.get_loops(conv_init)
    ij_init = sch.fuse(init_loops[-2], init_loops[-1])
    sch.tensorize(ij_init, "vta_init_intrin1")
    conv_loops = sch.get_loops(res_conv_block)
    ij_conv = sch.fuse(conv_loops[-3], conv_loops[-2])
    sch.tensorize(ij_conv, "vta_gemm_intrin1")

    sch.mod["main"].show()

    mod = sch.mod

    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())

    data_np = rng.integers(
        -128, 128, size=(batch_size, in_channels, height, width)
    ).astype(data.dtype)
    kernel_np = rng.integers(
        -128, 128, size=(out_channels, in_channels, kernel_h, kernel_w)
    ).astype(kernel.dtype)
    data_packed = data_np.reshape(
        batch_size // env.BATCH, env.BATCH, in_channels // env.BLOCK_IN, env.BLOCK_IN, height, width
    ).transpose((0, 2, 4, 5, 1, 3))

    kernel_packed = kernel_np.reshape(
        out_channels // env.BLOCK_OUT,
        env.BLOCK_OUT,
        in_channels // env.BLOCK_IN,
        env.BLOCK_IN,
        kernel_h,
        kernel_w,
    ).transpose((0, 2, 4, 5, 1, 3))

    data_nd = tvm.nd.array(data_packed, dev)
    kernel_nd = tvm.nd.array(kernel_packed, dev)
    res_nd = tvm.nd.array(numpy.zeros(output_shape).astype(res.dtype), dev)

    ex(data_nd, kernel_nd, res_nd)

    res_ref = topi.testing.conv2d_nchw_python(
        data_np.astype(env.acc_dtype),
        kernel_np.astype(env.acc_dtype),
        (stride_h, stride_w),
        (pad_h, pad_w),
    ).astype(env.acc_dtype)
    res_ref = res_ref >> env.INP_WIDTH
    res_ref = numpy.clip(res_ref, 0, inp_max)
    res_ref = res_ref.astype(res.dtype)
    res_ref = res_ref.reshape(
        batch_size // env.BATCH,
        env.BATCH,
        out_channels // env.BLOCK_OUT,
        env.BLOCK_OUT,
        fout_height,
        fout_width,
    ).transpose((0, 2, 4, 5, 1, 3))
    tvm.testing.assert_allclose(res_ref, res_nd.numpy())

def test_shift_bidirectional():
    O, M = 1, 1024
    o, m = O//env.BATCH, M//env.BLOCK_OUT
    shape = (o, m, env.BATCH, env.BLOCK_OUT)

    A = te.placeholder(shape, name="A", dtype=env.acc_dtype)
    B = te.placeholder(shape, name="B", dtype=env.acc_dtype)
    C = vtar.topi.bidi_shift(A, B)
    D = te.compute(shape, lambda *i: C(*i).astype(env.out_dtype), "D")

    alu = te.create_prim_func([A, B, D]).with_attr({"global_symbol": "shift"})
    mod = tvm.IRModule({"alu": alu})

    sch = tir.Schedule(mod)
    sch.work_on('alu')

    sch.mod.show()

    C_block = sch.get_block("res")
    A_cache = sch.cache_read(C_block, 0, env.acc_scope)
    B_cache = sch.cache_read(C_block, 1, env.acc_scope)
    sch.set_scope(C_block, 0, env.acc_scope)
    sch.annotate(sch.get_loops(A_cache)[0], env.dma_copy, True)
    sch.annotate(sch.get_loops(B_cache)[0], env.dma_copy, True)
    sch.annotate(sch.get_loops(C_block)[0], env.alu, True)
    sch.annotate(sch.get_loops(sch.get_block("D"))[0], env.dma_copy, True)

    sch.mod.show()

    mod = sch.mod

    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())
    A = tvm.nd.array(((rng.uniform(size=shape)-.5) * 10).astype("int32"), dev)
    B = tvm.nd.array(((rng.uniform(size=shape)-.5) * 10).astype("int32"), dev)
    C = tvm.nd.array(numpy.zeros(shape, dtype="int8"), dev)
    # FIXME: this should work with the argument reversed...
    ex(B, A, C)
    A_np = A.numpy()
    B_np = B.numpy()
    C_np = numpy.where(B_np >= 0,  A_np >> B_np, A_np << -B_np).astype("int8")
    numpy.testing.assert_equal(C.numpy(), C_np)

def test_dlight_conv2d():
    N, C, H, W = 1, 256, 14, 14
    O, R, S = 256, 3, 3
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    assert N % env.BATCH == 0
    assert C % env.BLOCK_IN == 0
    assert O % env.BLOCK_OUT == 0

    data_shape = (N//env.BATCH, C//env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN,)
    kernel_shape = (O//env.BLOCK_OUT, C//env.BLOCK_IN, R, S, env.BLOCK_OUT, env.BLOCK_IN,)
    ewise_shape = (1, O//env.BLOCK_OUT, 1, 1, 1, env.BLOCK_OUT,)

    data = te.placeholder(data_shape, env.inp_dtype, "data")
    kernel = te.placeholder(kernel_shape, env.wgt_dtype, "kernel")
    bias = te.placeholder(ewise_shape, env.acc_dtype, "bias")
    # offset = te.placeholder(ewise_shape, env.acc_dtype, "offset")
    shift = te.placeholder(ewise_shape, env.acc_dtype, "shift")

    # convolution
    conv = vtar.topi.conv2d_NCHWnc(data, kernel, strides, padding, dilation)
    out_shape = topi.utils.get_const_tuple(conv.shape)
    res_bias = te.compute(out_shape, lambda bo, co, i, j, bi, ci: conv[bo, co, i, j, bi, ci] + bias[0, co, 0, 0, 0, ci], "res_bias")
    # mul_pow2_round_nst
    # res_add = te.compute(out_shape, lambda bo, co, i, j, bi, ci: res_bias[bo, co, i, j, bi, ci] + (1 << (shift[0, co, 0, 0, 0, c]-1)), "res_add")
    # res_add = te.compute(out_shape, lambda bo, co, i, j, bi, ci: res_bias[bo, co, i, j, bi, ci] + offset[0, co, 0, 0, 0, c], "res_add")
    # TODO: the offset can be incorporated in the bias! if the bias is not
    # present it become the bias (only at that moment it could make sense to
    # make the 1 << (s-1) inside of VTA avoiding a memory load).
    res_add = res_bias
    res_neg = te.compute(ewise_shape, lambda *i: -shift[*i], "res_neg")
    res_cond = te.compute(ewise_shape, lambda *i: shift[*i] >= 0, "res_cond")
    res_shl = te.compute(out_shape, lambda bo, co, i, j, bi, ci: res_add[bo, co, i, j, bi, ci] << res_neg[0, co, 0, 0, 0, ci], "res_shl")
    res_shr = te.compute(out_shape, lambda bo, co, i, j, bi, ci: res_add[bo, co, i, j, bi, ci] >> shift[0, co, 0, 0, 0, ci], "res_shr")
    res_where = te.compute(out_shape, lambda bo, co, i, j, bi, ci: tir.Select(res_cond[0, co, 0, 0, 0, ci], res_shr[bo, co, i, j, bi, ci], res_shl[bo, co, i, j, bi, ci]), "res_where")
    # saturate cast
    res_min = te.compute(out_shape, lambda *i: te.min(res_where(*i), 127), "res_min")
    res_max = te.compute(out_shape, lambda *i: te.max(-128, res_min(*i)), "res_max")
    res = te.compute(out_shape, lambda *i: res_max(*i).astype(env.inp_dtype), "res")

    conv2d = te.create_prim_func((data, kernel, bias, shift,  res))
    mod = ir.IRModule.from_expr(conv2d)
    mod.show()

    seq = ir.transform.Sequential([
        tir.transform.ForceNarrowIndexToInt32(),
        dl.ApplyDefaultSchedule(
            vtar.dlight.Conv2DPrime(),
        ),
        vtar.tir.transform.FixSelectCondition,
    ])
    with target:
        mod = seq(mod)
    mod.show()
    ex = tir.build(mod, target, vtar.tir.get_vtar_tir_transform())

def test_resnet18_layers():

    workloads = vtar.topi.resnet18_workloads

    for idx, wl in enumerate(workloads):
        # We can't execute the 3x3 convolution
        if idx == 0:
            continue
        print(idx, wl)

        func = vtar.topi.sq_ioa_conv2d_NCHWnc_from_workload(wl, env.BATCH, env.BLOCK_IN, env.BLOCK_OUT)

        data_shape = [int(v) for v in func.struct_info.params[0].shape.values]
        kernel_shape = [int(v) for v in func.struct_info.params[1].shape.values]
        bias_shape = [int(v) for v in func.struct_info.params[2].shape.values]
        res_shape = [int(v) for v in func.struct_info.params[-1].shape.values]

        ex_cpu = tir.build(func, tvm.target.Target("llvm"))
        ex_vta = tir.build(func, target, vtar.tir.get_actual_pipeline())
        data_np = rng.integers(-128, 128, data_shape).astype('int8')
        kernel_np = rng.integers(-128, 128, kernel_shape).astype('int8')
        bias_np = rng.integers(-128, 128, bias_shape).astype('int32')
        # VTA can only shift by small values!
        scale_np = rng.integers(-7, 8, bias_shape).astype('int32')

        res_zeros = numpy.zeros(res_shape, dtype='int8')
        cpu_dev = tvm.device('cpu')

        res_cpu = tvm.nd.array(res_zeros, cpu_dev)
        ex_cpu(
            tvm.nd.array(data_np, cpu_dev),
            tvm.nd.array(kernel_np, cpu_dev),
            tvm.nd.array(bias_np, cpu_dev),
            tvm.nd.array(scale_np, cpu_dev),
            res_cpu
        )

        res_vta = tvm.nd.array(res_zeros, dev)
        ex_vta(
            tvm.nd.array(data_np, dev),
            tvm.nd.array(kernel_np, dev),
            tvm.nd.array(bias_np, dev),
            tvm.nd.array(scale_np, dev),
            res_vta
        )

        numpy.testing.assert_equal(res_cpu.numpy(), res_vta.numpy())

def test_broadcast():
    # TODO: test also when w_z is a vector in the case of per-channel quantization.
    N, C, H, W = 1, 256, 14, 14
    O, R, S = 256, 3, 3
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    assert N % env.BATCH == 0
    assert C % env.BLOCK_IN == 0
    assert O % env.BLOCK_OUT == 0

    data_shape = (N//env.BATCH, C//env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN,)
    minimal_shape = (1, 1, 1, 1, env.BLOCK_OUT, env.BLOCK_IN,)
    kernel_shape = (O//env.BLOCK_OUT, C//env.BLOCK_IN, R, S, env.BLOCK_OUT, env.BLOCK_IN,)

    x = te.placeholder(data_shape, env.inp_dtype, "x")
    # This is the minimal amount of data that VTA load in wgt memory. For us is
    # going to contain only the w_z value the zero point of the weight.
    w = te.placeholder(minimal_shape, env.wgt_dtype, "w")
    # Broadcasting is used to obtain the correct dimension for the convolution.
    wb = topi.broadcast_to(w, kernel_shape)
    y = vtar.topi.conv2d_NCHWnc(x, wb, strides, padding, dilation)
    out_shape = topi.utils.get_const_tuple(y.shape)
    res = te.compute(out_shape, lambda *i: y(*i).astype(env.inp_dtype), "res")

    f = te.create_prim_func((x, w, res))
    sch = tir.Schedule(f)
    broadcast_block_rv = sch.get_block("T_broadcast_to")
    # Inlining the computation should allow to use the classic VTA compilation path.
    sch.compute_inline(broadcast_block_rv)
    mod = sch.mod
    mod.show()

    ex_cpu = tir.build(mod, tvm.target.Target("llvm"))
    ex_vta = tir.build(mod, target, vtar.tir.get_actual_pipeline())
    data_np = rng.integers(-128, 128, data_shape).astype('int8')
    kernel_np = rng.integers(-128, 128, minimal_shape).astype('int8')
    res_zeros = numpy.zeros(out_shape, dtype='int8')
    cpu_dev = tvm.device('cpu')

    res_cpu = tvm.nd.array(res_zeros, cpu_dev)
    ex_cpu(
        tvm.nd.array(data_np, cpu_dev),
        tvm.nd.array(kernel_np, cpu_dev),
        res_cpu
    )

    res_vta = tvm.nd.array(res_zeros, dev)
    ex_vta(
        tvm.nd.array(data_np, dev),
        tvm.nd.array(kernel_np, dev),
        res_vta
    )

    numpy.testing.assert_equal(res_cpu.numpy(), res_vta.numpy())

def test_loop_fission_for_virtual_threading_in_vta():
    # VTA can only execute nested loop with one statement inside, virtual
    # threading split the BufferStore in a statement in multiple ones hence
    # the need for loop fission AKA unroll-and-jam
    shape = (16, 16)
    A = te.placeholder(shape, "float32", "A")
    B = te.compute(shape, lambda i, j: A[i, j] + 1, "B")
    func = te.create_prim_func((A, B))
    func.show()

    sch = tir.Schedule(func)
    block = sch.get_block("B")
    i, j = sch.get_loops(block)
    ij = sch.fuse(i, j)
    ijo, iji = sch.split(ij, factors=(2, None))
    sch.bind(ijo, "vthread.x")
    mod = sch.mod
    mod.show()

    transform_pass = ir.transform.Sequential([
        # Turn TensorIR blocks into "opaque" blocks (removes dependency tracking)
        tir.transform.ConvertBlocksToOpaque(),
        # Remove the blocks entirely, resulting in flat TIR (like 'before_virtual_thread')
        tir.transform.LowerOpaqueBlock(),
        tir.transform.FlattenBuffer(),
        tir.transform.InjectVirtualThread(),
        tir.transform.Simplify(),
        vtar.tir.transform.LoopFission(),
    ])

    mod = transform_pass(mod)
    mod.show()

    assert mod['main'].body.thread_binding is None, "virtual threads were not injected"

    A = te.placeholder((128,), name="A", dtype="float32")
    B = te.compute((128,), lambda i: A[i] * 2.0, name="B")
    C = te.compute((128,), lambda i: B[i] + 1.0, name="C")
    initial_func = te.create_prim_func([A, B, C])
    initial_func.show()

    sch = tvm.tir.Schedule(initial_func)

    block_b = sch.get_block("B")
    block_c = sch.get_block("C")

    loop_b, = sch.get_loops(block_b)
    sch.reverse_compute_at(block_c, loop_b)
    loop_b_o, loop_b_i = sch.split(loop_b, factors=(2, None))
    sch.bind(loop_b_o, "vthread.x")
    sch.annotate(loop_b_i, "alu", 0)

    mod = sch.mod
    mod["main"].show()
    mod = transform_pass(mod)
    mod.show()

    assert isinstance(mod['main'].body.seq[0], tir.For), "loop fission was not applied"
    assert isinstance(mod['main'].body.seq[1], tir.For), "loop fission was not applied"
    assert isinstance(mod['main'].body.seq[2], tir.For), "loop fission was not applied"
    assert isinstance(mod['main'].body.seq[3], tir.For), "loop fission was not applied"
    assert mod['main'].body.seq[0].thread_binding is None, "virtual threads were not injected"
    assert mod['main'].body.seq[1].thread_binding is None, "virtual threads were not injected"
    assert mod['main'].body.seq[2].thread_binding is None, "virtual threads were not injected"
    assert mod['main'].body.seq[3].thread_binding is None, "virtual threads were not injected"

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

    mod = ConvModel
    # R.layout_transform is R.reshape followed by R.permute_dims
    mod = relax.transform.ConvertLayout({
        "relax.nn.conv2d": ["NCHW1n16c", "OIHW16o16i"],
    })(mod)
    mod.show()
    # TODO: convert vtar.relax.transform.GraphPack to use R.layout_transform

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

    mod1 = ir.transform.Sequential([
        vtar.relax.transform.GraphPack(),
        relax.transform.CanonicalizeBindings(), # removes redundant assignments
    ])(ConvModel)
    mod2 = ConvModelPacked
    mod1.show()
    mod2.show()
    ir.assert_structural_equal(mod1, mod2)

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
                lv = R.reshape(x, R.shape((8,)))
                gv = lv
                R.output(gv)
            return gv

    # TODO: DequantMaxpool2DQuant -> Maxpool2D

    tvm.ir.assert_structural_equal(
        vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping()(DequantReshapeQuant),
        Reshape
    )

def test_trivial_end2end_compilation():
    inp_width = 8
    int8_max = 127

    @I.ir_module
    class ALUOperations:
        @R.function
        def main(
            x: R.Tensor((1, 128 + 1, 1, 16), dtype="int32"),
            y: R.Tensor((1, 128 + 1, 1, 16), dtype="int32"),
        ):
            with R.dataflow():
                # TODO: x + y * x
                lv = x + y
                lv1 = R.right_shift(lv, R.const(inp_width))
                lv2 = R.maximum(lv1, R.const(0)) # ReLU
                lv3 = R.minimum(lv2, R.const(int8_max))
                lv4 = lv3.astype("int8")
                gv = lv4
                R.output(gv)
            return gv

    mod = ALUOperations
    almost_end2end_pipeline = ir.transform.Sequential([
        # TODO: integer only quantization
        # vtar.relax.transform.GraphPack(),
        # relax.transform.CanonicalizeBindings(), # removes redundant assignments
        relax.get_pipeline('vtar_zero'),
        ############################################################################
        tir.transform.ForceNarrowIndexToInt32(),
        dl.ApplyDefaultSchedule(
            vtar.dlight.ALU(),
        ),
        vtar.tir.get_vtar_tir_transform(),
    ])

    with target:
        mod = almost_end2end_pipeline(mod)
    # mod.show()
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)
    # The nd.array needs to be given the dev argument probably because the
    # allocation needs to be performed in a CMA to be loaded with VTA's DMA.
    a = tvm.nd.array(numpy.ones((1, 128 + 1, 1, 16), dtype='int32')*1024, dev)
    res = vm['main'](a, a)

def test_trivial_quantized_gemm():
    @I.ir_module
    class QantizedGEMM:
        @R.function
        def main(
            x: R.Tensor((1, 64, 1, 16), "float32"),
            w: R.Tensor((64, 64, 16, 16), "float32"),
        ):
            with R.dataflow():
                lv = R.quantize(x, R.const(2., "float32"), R.const(-10, "int8"))
                lv1 = R.quantize(w, R.const(2., "float32"), R.const(-5, "int8"))
                lv2 = R.einsum((lv.astype("int32"), lv1.astype("int32")), "IKik,JKjk->IJij") # matmul with packed format
                lv3 = R.nn.relu(lv2)
                lv4 = lv3.astype("int8")
                lv5 = R.dequantize(lv4, R.const(2., "float32"), R.const(+3, "int8"))
                gv = lv5
                R.output(gv)
            return gv

    data = relax.dpl.is_op("relax.astype")(relax.dpl.wildcard())
    weight = relax.dpl.is_op("relax.astype")(relax.dpl.wildcard())
    out = relax.dpl.is_tuple((data, weight))
    out = relax.dpl.is_op("relax.einsum")(out)
    out = relax.dpl.is_op("relax.nn.relu")(out)
    out =relax.dpl.is_op("relax.astype")(out)
    quantized_pat = out

    patterns = (
        relax.transform.FusionPattern("vtar.quantized_compute", quantized_pat),
    )

    mod = QantizedGEMM
    mod = relax.transform.FuseOpsByPattern(patterns)(mod)
    mod.show()

    pytest.skip("relax.dpl.is_tuple seems to be broken on the TVM side.")

def test_constant_folding_patches():
    @I.ir_module
    class BeforeModule:
        @R.function
        def main(x: R.Tensor((1, 4), dtype="float32")):
            with R.dataflow():
                c1 = R.const(numpy.array(((1, 2, 3, 4),), dtype="float32"))
                c2 = R.const(numpy.array(((10, 20, 30, 40),), dtype="float32"))
                c3 = R.astype(R.const(numpy.array([1]), dtype="int32"), dtype="float32")
                c4 = R.add(c1, c3)
                tt = R.astype(R.const(0), dtype="float32")
                # relax.transform.FoldConstant does constant folding by execution
                # hence it completely stops the analysis if it encounters a non
                # compile-time known constant. This is a shame since it misses to do
                # some trivial optimizations when multiplying or adding by zero or
                # one.
                c5 = R.multiply(x, tt)
                c6 = R.add(c5, c4)
                intermediate = R.add(c6, c2)
                lv1 = R.add(x, intermediate)
                # Given the previously explained limitations this pattern is
                # also out of scope for relax.transform.FoldConstant
                lv2 = R.add(lv1, R.const(numpy.array(((1, 1, 1, 1),), dtype="float32")))
                gv = R.add(R.const(numpy.array(((1, 1, 1, 1),), dtype="float32")), lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class ReferenceModule:
        @R.function
        def main(x: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((1, 4), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((1, 4), dtype="float32") = R.add(x, R.const(numpy.array(((11+1+2, 22+1+2, 33+1+2, 44+1+2),), dtype="float32")))
                R.output(gv)
            return gv

    pipeline = tvm.transform.Sequential((
        vtar.relax.transform.SimplifyConstAstype(),
        relax.transform.CanonicalizeBindings(), # necessary
        vtar.relax.transform.SimplifyRing(),
        relax.transform.FoldConstant(),
        tvm.ir.transform.PrintIR(),
        vtar.relax.transform.AddChainSimplify(),
        vtar.relax.transform.AddChainSimplify(),
        relax.transform.CanonicalizeBindings(),
        tvm.ir.transform.PrintIR(),
    ))
    mod = BeforeModule
    mod = pipeline(mod)
    AfterModule = mod

    ir.assert_structural_equal(AfterModule, ReferenceModule)

def test_qconv2d_operator_fusion():
    @I.ir_module
    class Module:
        @R.function
        def qconv(
            x: R.Tensor((1, 1, 1, 1), dtype="int8"),
            w: R.Tensor((1, 1, 1, 1), dtype="int8"),
            b: R.Tensor((1, 1, 1, 1), dtype="int32"),
        ):
            with R.dataflow():
                conv = (R.nn.conv2d(x, w, out_dtype="int32") + b)
                mul_round_nst = R.left_shift(conv + R.const(2**3), R.const(2))
                clamp_astype = R.maximum(R.const(-128), R.minimum(mul_round_nst, R.const(127))).astype("int8")
                R.output(clamp_astype)
            return clamp_astype

    s = relax.const(numpy.ones(10, dtype="int32")*2)
    @I.ir_module
    class Module1:
        @R.function
        def qconv(
            x: R.Tensor((1, 1, 1, 1), dtype="int8"),
            w: R.Tensor((1, 1, 1, 1), dtype="int8"),
            b: R.Tensor((1, 1, 1, 1), dtype="int32"),
        ):
            with R.dataflow():
                conv = (R.nn.conv2d(x, w, out_dtype="int32") + b)
                # Using the same constant for both addition and shift is wrong
                # semantically but the pattern should match anyway.
                x = conv + s
                mul_round_nst = R.bidi_shift(x, s)
                clamp_astype = R.maximum(R.const(-128), R.minimum(mul_round_nst, R.const(127))).astype("int8")
                R.output(clamp_astype)
            return clamp_astype

    patterns = relax.backend.get_patterns_with_prefix("vtar")
    mod = Module
    mod.show()
    assert len(mod["qconv"].body.blocks[0].bindings) > 1
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOpsByPattern(patterns)(mod)
    mod.show()
    assert len(mod["qconv"].body.blocks[0].bindings) == 1

    mod = Module1
    mod.show()
    assert len(mod["qconv"].body.blocks[0].bindings) > 1
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOpsByPattern(patterns)(mod)
    mod.show()
    assert len(mod["qconv"].body.blocks[0].bindings) == 1

def test_simple_requantize():
    # https://en.wikipedia.org/wiki/Normal_distribution
    # https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution

    box = numpy.broadcast_to(1/9, (1, 1, 3, 3)).astype("float32")
    inp = rng.random((1, 1, 10, 10)).astype("float32")

    s_x, z_x = vtar.utils.asymmetric_scale_zero_point(0, 1, "int8")
    s_w, z_w = vtar.utils.symmetric_scale_zero_point(0, 1, "int8")
    s_y, z_y = vtar.utils.asymmetric_scale_zero_point(0, 1, "int8")

    q_box = numpy.clip(numpy.round(box/s_w) + z_w, -128, 127).astype("int8")

    @R.function
    def main(x: R.Tensor((1, 1, 10, 10), dtype="float32")):
        with R.dataflow():
            gv = R.nn.conv2d(x, R.const(box, "float32"))
            R.output(gv)
        return gv

    @R.function
    def q_main(x: R.Tensor((1, 1, 10, 10), dtype="float32")):
        with R.dataflow():
            q_x = R.quantize(x, R.const(s_x, "float32"), R.const(z_x, "int8"))
            lv = R.qnn.conv2d(
                q_x,                    R.const(s_x, "float32"), R.const(z_x, "int8"),
                R.const(q_box, "int8"), R.const(s_w, "float32"), R.const(z_w, "int8"),
                                        R.const(s_y, "float32"), R.const(z_y, "int8"),
            )
            gv = R.dequantize(lv, R.const(s_y, "float32"), R.const(z_y, "int8"))
            R.output(gv)
        return gv

    mod = ir.IRModule({"main": main})
    q_mod = ir.IRModule({"main": q_main})

    vm = relax.VirtualMachine(tvm.compile(mod), tvm.cpu())

    zero = relax.get_pipeline("vtar_zero") # To legalize bidi_shift
    q_mod_iao = zero(vtar.relax.transform.ReQuantize()(q_mod))
    q_vm_iao = relax.VirtualMachine(tvm.compile(q_mod_iao), tvm.cpu())
    q_mod = zero(vtar.relax.transform.LowerQNNOps()(q_mod))
    q_vm = relax.VirtualMachine(tvm.compile(q_mod), tvm.cpu())

    inp_arr = tvm.nd.array(inp)
    res = vm["main"](inp_arr).numpy()
    q_res = q_vm["q_main"](inp_arr).numpy()
    q_res_iao = q_vm_iao["q_main"](inp_arr).numpy()

    print(res, q_res, q_res_iao, sep="\n")

    q_err = numpy.linalg.norm((res - q_res).flatten(), ord=numpy.inf)
    q_err_ioa = numpy.linalg.norm((res - q_res_iao).flatten(), ord=numpy.inf)

    print(q_err, q_err_ioa, sep="\n")

    numpy.testing.assert_allclose(q_err, q_err_ioa)

# ONNX tests ###################################################################

onnx_text_quantized_bottleneck = """\
<
   ir_version: 6,
   opset_import: ["" : 11, "com.microsoft" : 1]
>
main_graph (
    float[1,3,224,224] input,
    int8[64,3,7,7] "Conv_193_q", int32[64] "Conv_194_q",
    int8[64,64,3,3] "Conv_196_q", int32[64] "Conv_197_q",
    int8[64,64,3,3] "Conv_199_q", int32[64] "Conv_200_q",
    int8[64,64,3,3] "Conv_202_q", int32[64] "Conv_203_q",
    int8[64,64,3,3] "Conv_205_q", int32[64] "Conv_206_q",
    int8[10,64] "fc_weight_q", int32[10] "fc_bias_q"
) => (float[1,10] output)
<
    float input_s = {0.0186584}, int8 input_zp = {-14},
    float "/conv1/Conv_s" = {0.0288692}, int8 "/conv1/Conv_zp" = {-128},
    float "Conv_193_s" = {0.00268506}, int8 "Conv_193_zp" = {0},
    float "/l0/conv1/Conv_s" = {0.0219578}, int8 "/l0/conv1/Conv_zp" = {-128},
    float "MaxPool_s" = {0.0288692}, int8 "MaxPool_zp" = {-128},
    float "Conv_196_s" = {0.00252428}, int8 "Conv_196_zp" = {0},
    float "/l0/conv2/Conv_s" = {0.0414506}, int8 "/l0/conv2/Conv_zp" = {16},
    float "Conv_199_s" = {0.00593825}, int8 "Conv_199_zp" = {0},
    float "/l0/Add_s" = {0.0333703}, int8 "/l0/Add_zp" = {-128},
    float "/l1/conv1/Conv_s" = {0.0201473}, int8 "/l1/conv1/Conv_zp" = {-128},
    float "Conv_202_s" = {0.00179852}, int8 "Conv_202_zp" = {0},
    float "/l1/conv2/Conv_s" = {0.0872713}, int8 "/l1/conv2/Conv_zp" = {32},
    float "Conv_205_s" = {0.006156}, int8 "Conv_205_zp" = {0},
    float "/l1/Add_s" = {0.0397232}, int8 "/l1/Add_zp" = {-128},
    float "GlobalAveragePool_s" = {0.05131}, int8 "GlobalAveragePool_zp" = {-128},
    float output_s = {0.240899}, int8 output_zp = {-47},
    float "/Flatten_s" = {0.05131}, int8 "/Flatten_zp" = {-128},
    float "fc_weight_s" = {0.000817349}, int8 "fc_weight_zp" = {0}
>
{
    input_q = QuantizeLinear (input, input_s, input_zp)

    "/conv1/Conv_q" = QLinearConv
        <dilations: ints = [1, 1], group: int = 1, kernel_shape: ints = [7, 7], pads: ints = [3, 3, 3, 3], strides: ints = [2, 2]>
        (input_q, input_s, input_zp, "Conv_193_q", "Conv_193_s", "Conv_193_zp", "/conv1/Conv_s", "/conv1/Conv_zp", "Conv_194_q")

    "/relu/Relu" = DequantizeLinear ("/conv1/Conv_q", "/conv1/Conv_s", "/conv1/Conv_zp")
    "MaxPool" = MaxPool
        <ceil_mode: int = 0, dilations: ints = [1, 1], kernel_shape: ints = [3, 3], pads: ints = [1, 1, 1, 1], strides: ints = [2, 2]>
        ("/relu/Relu")
    "MaxPool_q" = QuantizeLinear ("MaxPool", "MaxPool_s", "MaxPool_zp")

    "/l0/conv1/Conv_q" = QLinearConv
        <dilations: ints = [1, 1], group: int = 1, kernel_shape: ints = [3, 3], pads: ints = [1, 1, 1, 1], strides: ints = [1, 1]>
        ("MaxPool_q", "MaxPool_s", "MaxPool_zp", "Conv_196_q", "Conv_196_s", "Conv_196_zp", "/l0/conv1/Conv_s", "/l0/conv1/Conv_zp", "Conv_197_q")
    "/l0/conv2/Conv_q" = QLinearConv
        <dilations: ints = [1, 1], group: int = 1, kernel_shape: ints = [3, 3], pads: ints = [1, 1, 1, 1], strides: ints = [1, 1]>
        ("/l0/conv1/Conv_q", "/l0/conv1/Conv_s", "/l0/conv1/Conv_zp", "Conv_199_q", "Conv_199_s", "Conv_199_zp", "/l0/conv2/Conv_s", "/l0/conv2/Conv_zp", "Conv_200_q")
    "/l0/Add_q" = com.microsoft.QLinearAdd
        ("/l0/conv2/Conv_q", "/l0/conv2/Conv_s", "/l0/conv2/Conv_zp", "MaxPool_q", "MaxPool_s", "MaxPool_zp", "/l0/Add_s", "/l0/Add_zp")
    "/l1/conv1/Conv_q" = QLinearConv
        <dilations: ints = [1, 1], group: int = 1, kernel_shape: ints = [3, 3], pads: ints = [1, 1, 1, 1], strides: ints = [1, 1]>
        ("/l0/Add_q", "/l0/Add_s", "/l0/Add_zp", "Conv_202_q", "Conv_202_s", "Conv_202_zp", "/l1/conv1/Conv_s", "/l1/conv1/Conv_zp", "Conv_203_q")
    "/l1/conv2/Conv_q" = QLinearConv
        <dilations: ints = [1, 1], group: int = 1, kernel_shape: ints = [3, 3], pads: ints = [1, 1, 1, 1], strides: ints = [1, 1]>
        ("/l1/conv1/Conv_q", "/l1/conv1/Conv_s", "/l1/conv1/Conv_zp", "Conv_205_q", "Conv_205_s", "Conv_205_zp", "/l1/conv2/Conv_s", "/l1/conv2/Conv_zp", "Conv_206_q")
    "/l1/Add_q" = com.microsoft.QLinearAdd
        ("/l1/conv2/Conv_q", "/l1/conv2/Conv_s", "/l1/conv2/Conv_zp", "/l0/Add_q", "/l0/Add_s", "/l0/Add_zp", "/l1/Add_s", "/l1/Add_zp")

    "GlobalAveragePool_q" = com.microsoft.QLinearGlobalAveragePool
        <channels_last: int = 0>
        ("/l1/Add_q", "/l1/Add_s", "/l1/Add_zp", "GlobalAveragePool_s", "GlobalAveragePool_zp")

    "GlobalAveragePool" = DequantizeLinear ("GlobalAveragePool_q", "GlobalAveragePool_s", "GlobalAveragePool_zp")
    "/Flatten" = Flatten <axis: int = 1> ("GlobalAveragePool")
    "/Flatten_q" = QuantizeLinear ("/Flatten", "/Flatten_s", "/Flatten_zp")

    output_q = com.microsoft.QGemm
        <alpha: float = 1., transB: int = 1>
        ("/Flatten_q", "/Flatten_s", "/Flatten_zp", "fc_weight_q", "fc_weight_s", "fc_weight_zp", "fc_bias_q", output_s, output_zp)

    output = DequantizeLinear (output_q, output_s, output_zp)
}
"""

def my_instrument(func, func_symbol: str, before_run: bool, ret_value, *args) -> int:
    after_run = not before_run
    # NOTE: why ret_value is always None???
    if after_run and "conv" in func_symbol:
        print(func_symbol)
    return relax.VMInstrumentReturnKind.NO_OP

def test_onnx_import_simple_bottleneck():
    onnx_model = onnx.parser.parse_model(onnx_text_quantized_bottleneck)
    onnx.checker.check_model(onnx_model)
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping()(mod)
    mod.show()
    mod = vtar.relax.transform.LowerQNNOps()(mod)
    mod.show()
    try:
        # NOTE: this is bugged since it can't handle addition the addition of the
        # skip connection in InferLayoutBinaryEwise
        # https://daobook.github.io/tvm/docs/arch/convert_layout.html
        mod = relax.transform.ConvertLayout({
            "relax.nn.conv2d": ["NCHW1n16c", "OIHW16o16i"],
        })(mod)
        raise ValueError("This should not be reached because ConvertLayout is bugged")
    except tvm.error.TVMError:
        pass
    mod = vtar.relax.transform.GraphPack()(mod)
    mod = relax.get_pipeline('vtar_zero')(mod)
    mod.show()
    # FIXME: this crashes if we use ext_dev(0) because look at test_trivial_end2end_compilation
    dev = tvm.runtime.device('cpu')
    target = tvm.target.Target.from_device(dev)
    # tvm.target.Target("llvm", host="llvm")
    params_spec = [
        (topi.utils.get_const_tuple(relax.get_shape_of(param)), param.struct_info.dtype)
        for param in mod['main'].params
    ]
    params = [tvm.nd.array((rng.random(shape)*10).astype(dtype), dev) for shape, dtype in params_spec]
    ex = relax.build(mod, target)
    path = "build/bottleneck_int8.dll"
    ex.export_library(path)
    rt_mod = tvm.runtime.load_module(path)
    vm = relax.VirtualMachine(rt_mod, dev)
    vm.set_instrument(my_instrument) # could be used for quantization in TVM
    time_f = vm.time_evaluator("main", dev, number=1)
    res = time_f(*params)
    print(print(rt_mod["stats"]()))
    print(res)

def test_onnx_import_keep_params_in_input_or_not():
    onnx_model = onnx.parser.parse_model(onnx_text_quantized_bottleneck)
    onnx.checker.check_model(onnx_model)
    _ = vtar.relax.frontend.onnx.from_onnx(onnx_model, keep_params_in_input=False)
    _ = vtar.relax.frontend.onnx.from_onnx(onnx_model, keep_params_in_input=True)

# Misc. tests ##################################################################

def test_relax_metadata_serialization():

    value = relax.const(numpy.ones(1))
    text = value.script(show_meta=True)
    # The "metadata" notation is used every time a Relax constant is not a
    # scalar as it can be seen by (which seem to be bugged but okay)
    program, metadata = relax.utils.metadata_partitioner(text)
    # There is no "metatadata" attribute anywhere in the IRModule it is just
    # printed that way to serialize all the information in a printable format.
    value_ = ir.load_json(metadata.replace("\\\"", "\""))["relax.expr.Constant"][0]
    struct_info = value_.struct_info

    assert (value_.data.numpy() == value.data.numpy()).item()

# TODO: use a similar algorithm for InjectDMAIntrin
def test_can_prove_data_load_invariant_of_access_order():
    N, M = tir.Var("N", "int32"), tir.Var("M", "int32")

    @T.prim_func
    def row_major(A: T.Buffer((N, M), "float32"), B: T.Buffer((N, M), "float32")):
        for i, j in T.grid(N, M):
            with T.block("block_row"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj]

    @T.prim_func
    def col_major(A: T.Buffer((N, M), "float32"), B: T.Buffer((N, M), "float32")):
        for j, i in T.grid(M, N):
            with T.block("block_col"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj]

    def get_access_region(func):
        sch = tir.Schedule(func)
        block_rv = sch.get_block("root")
        block_stmt = sch.get(block_rv)

        # Why func.buffer_map alone is not enough?
        buffer_var_map = {buf.data: buf for var, buf in func.buffer_map.items()}
        read_regions, write_regions = tir.analysis.get_block_read_write_region(
            block_stmt,
            buffer_var_map
        )
        return read_regions[0].region

    region_1 = get_access_region(row_major)
    region_2 = get_access_region(col_major)

    analyzer = arith.Analyzer()

    # print("Loop 1 Region:", region_1)
    # print("Loop 2 Region:", region_2)

    is_equivalent = True
    for r1, r2 in zip(region_1, region_2):
        min_eq = analyzer.can_prove_equal(r1.min, r2.min)
        ext_eq = analyzer.can_prove_equal(r1.extent, r2.extent)

        if not (min_eq and ext_eq):
            is_equivalent = False
            break

    total_elements_accessed = 1
    for r in region_1:
        total_elements_accessed *= r.extent

    buffer_shape = row_major.buffer_map[row_major.params[0]].shape
    total_buffer_size = 1
    for dim in buffer_shape:
        total_buffer_size *= dim

    is_contiguous = analyzer.can_prove_equal(total_elements_accessed, total_buffer_size)

    assert is_equivalent and is_contiguous

    print(f"Proof of Equivalence: {is_equivalent}")
    print(f"Proof of Contiguity: {is_contiguous}")
    print(f"Volume Accessed: {total_elements_accessed}")
    print(f"Buffer Size:     {total_buffer_size}")

@pytest.mark.skipif("HAS_VTA" not in os.environ, reason="Pynq with VTA is not connected")
def test_pynq_remote():
    # FIXME: implement this properly
    from tvm import rpc
    from tvm.contrib import utils

    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]).with_attr("global_symbol", "add_one"))

    local_demo = False

    if local_demo:
        target = "llvm"
    else:
        # uname -i
        target = "llvm -mtriple=aarch64-linux-gnueabihf"

    func = tvm.compile(mod, target=target)
    # save the lib at a local temp folder
    temp = utils.tempdir()
    path = temp.relpath("lib.tar")
    func.export_library(path)

    if local_demo:
        remote = rpc.LocalSession()
    else:
        host = "192.168.137.48"
        port = 9091
        remote = rpc.connect(host, port)

    import vtar
    import os
    import shutil

    # set "PYTHONPATH=%CD%\..\submodules\tvm\vta\python"
    # The bitsream should be inside "zcu104\0_0_1\1x16_i8w8a32_15_15_18_17.bit"
    os.environ["VTA_CACHE_PATH"] = os.path.join(os.environ["installdir"], "Programs/bitstreams")
    # The code expect the HOME environment variable to exists.
    os.environ["HOME"] = "workaround"
    shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
            "submodules/tvm-vta/config/vta_config.json")
    # vtar.reconfig_runtime(remote)
    # vtar.program_fpga(remote, bitstream=None)

    remote.upload(path)
    func = remote.load_module("lib.tar")

    dev = remote.cpu()
    a = tvm.nd.array(numpy.random.uniform(size=1024).astype(A.dtype), dev)
    b = tvm.nd.array(numpy.zeros(1024, dtype=A.dtype), dev)
    # the function will run on the remote device
    func(a, b)
    numpy.testing.assert_equal(b.numpy(), a.numpy() + 1)

    time_f = func.time_evaluator(func.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print("%g secs/op" % cost)

@pytest.mark.skipif("HAS_VTA" not in os.environ, reason="Pynq with VTA is not connected")
def test_vta_remote():
    from tvm import rpc
    # FIXME: implement this properly
    os.environ["VTA_CACHE_PATH"] = os.path.join(os.environ["installdir"], "\\Programs\\bitstreams")

    host = "192.168.137.48"
    port = 9091
    remote = rpc.connect(host, port)
    # vtar.program_fpga(remote, bitstream=None)
    # rng = numpy.random.default_rng(42)
    dev = remote.ext_dev(0)
    # A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    A = tvm.nd.empty((1, 64, 1, 16), "int32", dev)
    # B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
    B = tvm.nd.empty((1, 64, 1, 16), "int32", dev)
    # C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
    C = tvm.nd.empty((1, 64, 1, 16), "int8", dev)

    remote.upload("vta.tar")
    func = remote.load_module("vta.tar")
    func(A, B, C)

if __name__ == '__main__':
    testing.main()
