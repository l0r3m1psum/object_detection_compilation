import tvm
from tvm import relax
from tvm import topi
from tvm import te
from tvm import tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import os, sys
sys.path.append(os.path.join(os.getcwd(), "submodules/tvm/vta/python"))

import vtar.relax.transform

# inp_dtype: int8
# wgt_dtype: int8
# out_dtype: int8
# acc_dtype: int32
# BATCH: 1
# BLOCK_IN: 16
# BLOCK_OUT: 16

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
			conv1 = R.nn.conv2d(x, conv1_weight, strides=1, padding=1, dilation=1,
				out_dtype="int32")
			add1 = R.add(conv1, conv1_bias)
			gv = add1
			R.output(gv)
		return gv

# To interpret look at test_benchmark_topi_conv2d.py:run_conv2d
# H=56, W=56, I=64, O=64, kH=3, kW=3
@I.ir_module
class ConvModelVTA:
	@R.function
	def main(
		#             (1//BATCH,      64//BLOCK_IN, 56, 56, BATCH,     BLOCK_IN)
		x:            R.Tensor((1//1,   64//16, 56, 56, 1,  16), dtype="int8"),
		#             (64//BLOCK_OUT, 64//BLOCK_IN, 3,  3,  BLOCK_OUT, BLOCK_IN)
		conv1_weight: R.Tensor((64//16, 64//16, 3,  3,  16, 16), dtype="int8"),
		#             (1//BATCH,      64//BLOCK_IN, 1,  1,  BATCH,     BLOCK_OUT)
		conv1_bias:   R.Tensor((1//1,   64//16, 1,  1,  1,  16), dtype="int32"),
	):
		R.func_attr({"num_input": 1})
		with R.dataflow():
			conv1 = R.nn.conv2d(x, conv1_weight, strides=1, padding=1, dilation=1,
				data_layout="NCHW1n16c", kernel_layout="OIHW16o16i", out_dtype="int32")
			add1 = R.add(conv1, conv1_bias)
			gv = add1
			R.output(gv)
		return gv

print(ConvModel)
print(ConvModelVTA)

res = vtar.relax.transform.ReluToGeluAndQuantizeMatmul()(ConvModel)
print(res)
