# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/matrix_multiply.html
# https://tvm.apache.org/docs/v0.8.0/topic/vta/tutorials/optimize/matrix_multiply_opt.html

import tvm
from tvm import te
import vta

env = vta.get_env()

assert env.BATCH == 1
assert env.BLOCK_IN == 16
assert env.BLOCK_OUT == 16

assert env.inp_dtype == "uint8"
assert env.wgt_dtype == "uint8"
assert env.acc_dtype == "uint32"

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
