import tvm

from tvm import relax, te, tir, topi

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import vtar.topi
import vtar.relax

import numpy

from typing import List, Tuple

import os, sys
sys.path.append(os.path.join(os.getcwd(), "submodules\\tvm\\vta\\python"))
import vta.testing

os.environ["TVM_WIN_CC"] = "clang_wrapper.bat"

# inp_dtype: int8
# wgt_dtype: int8
# out_dtype: int8
# acc_dtype: int32
# BATCH: 1
# BLOCK_IN: 16
# BLOCK_OUT: 16

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

mod = ConvModelVTA
zero_pipeline = relax.get_pipeline('vtar_zero')
mod = zero_pipeline(mod)

def make_closure_test_hardcoded_relax(mod):
	def test_hardcoded_relax(env: vta.Environment, remote: tvm.rpc.RPCSession) -> None:
		nonlocal mod

		# TODO: this are not needed anymore since we are compiling with Relax
		data   = te.placeholder((1, 4, 56, 56, 1, 16), name="data",   dtype="int8")
		kernel = te.placeholder((4, 4, 3, 3, 16, 16),  name="kernel", dtype="int8")
		bias   = te.placeholder((1, 4, 1, 1, 1, 16),   name="bias",   dtype="int32")
		res    = te.placeholder((1, 4, 56, 56, 1, 16), name="res",    dtype="int32")

		data_shape   = topi.utils.get_const_tuple(data.shape)
		kernel_shape = topi.utils.get_const_tuple(kernel.shape)
		bias_shape   = topi.utils.get_const_tuple(bias.shape)
		res_shape    = topi.utils.get_const_tuple(res.shape)

		data_np   = numpy.random.randint(0, 10, size=data_shape).astype(data.dtype)
		kernel_np = numpy.random.randint(0, 10, size=kernel_shape).astype(kernel.dtype)
		bias_np   = numpy.random.randint(0, 10, size=bias_shape).astype(bias.dtype)
		res_np    = numpy.zeros(res_shape).astype(res.dtype)

		dev = tvm.device(str(env.target))
		target = tvm.target.Target(env.target, host=env.target_host)

		with vta.build_config():
			ex = relax.build(mod, target)
		vm = relax.VirtualMachine(ex, dev)
		# _ = vm['main'](data_arr, kernel_arr, bias_arr)
		# FIXME: make this cross platform
		ex.export_library('build/conv2d.dll')
		remote.upload("build/conv2d.dll")
		f = remote.load_module("conv2d.dll")
		devr = remote.device(str(env.target))
		time_f = f.time_evaluator(f.entry_name, devr, number=1)

		data_arr   = tvm.nd.array(data_np, devr)
		kernel_arr = tvm.nd.array(kernel_np, devr)
		bias_arr   = tvm.nd.array(bias_np, devr)
		res_arr    = tvm.nd.array(res_np, devr)

		cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
		print(cost)
	return test_hardcoded_relax

vta.testing.run(make_closure_test_hardcoded_relax(mod))
