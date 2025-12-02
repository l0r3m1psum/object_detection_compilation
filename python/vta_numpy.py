"""VTA operations performed with NumPy for documentation and understanding
purposes."""
import itertools

import numpy

def grid(*ns): return itertools.product(*(range(n) for n in ns))

def prod(iterable, /, start=1):
    res = start
    for element in iterable:
        res *= element
    return res

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
f()


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
g()

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

def load2d(
        src_dram_addr: numpy.ndarray, src_elem_offset: int, x_size: int, y_size: int, x_stride: int,
        x_pad_before: int, y_pad_before: int, x_pad_after: int, y_pad_after: int,
        dst_sram_index: int, dst_memory_type: int
    ) -> numpy.ndarray:
    """VTALoadBuffer2D"""
    sram_shape = (x_pad_before + x_size + x_pad_after, y_pad_before + y_size + y_pad_after)
    sram_len = sram_shape[0] * sram_shape[1]
    # vta/hw_spec_const.h
    sram_dtype = ("void", "int8", "int8", "int32", "int8", "int8")[dst_memory_type]
    sram = numpy.empty((4096,), sram_dtype)
    sram_slice = slice(0 + dst_sram_index, sram_len + dst_sram_index)

    rows, cols = src_dram_addr.shape
    x_offset, y_offset = src_elem_offset//rows, src_elem_offset%cols
    stride, reminder = divmod(x_stride, rows)
    if reminder:
        raise ValueError("x_stride must be a multiple of the rows of src_dram_addr")
    dram_x_slice = slice(0 + x_offset, x_size + x_offset, stride)
    dram_y_slice = slice(0 + y_offset, y_size + y_offset, 1)

    pad_shape = ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after))

    sram[sram_slice] = numpy.pad(
        src_dram_addr[dram_x_slice, dram_y_slice], pad_shape, constant_values=0
    ).flatten()
    return sram[sram_slice].reshape(sram_shape)

def store2d(
        src_sram_index: int, src_memory_type: int,
        dst_dram_addr: numpy.ndarray , dst_elem_offset: int, x_size: int, y_size: int, x_stride: int
    ) -> None:
    pass

A = numpy.arange(100*100, dtype='int32').reshape((100, 100))

res = load2d(
    A,
    100*10 + 20, # skip 10 rows and 20 columns
    30, 20, 100, # take a (30, 20) sub-matrix of consecutive rows (stride = 100)
    1, 1, 1, 1,  # pad the output with an halo of zeros of width 1
    0, 3,        # put it at the beginning of the ACC SRAM
)

shape = (8, 8, 16, 16)
WGT = numpy.arange(prod(shape), dtype="int8").reshape(shape)
wgt = numpy.empty((4, 4, 16, 16), dtype="int8"); assert wgt.size = 4096
inp = numpy.empty((4, 64, 1, 16), dtype="int8"); assert inp.size = 4096
out = numpy.empty((4, 64, 1, 16), dtype="int8"); assert out.size = 4096

# This needs to be lowered to a VTALoadBuffer2D
for i, j in grid(2, 2):
    wgt[:, :, :, :] = WGT[i*4: i*4 + 4, j*4: j*4 + 4, :, :]
