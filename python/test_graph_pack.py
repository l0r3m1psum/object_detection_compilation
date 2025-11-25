import tvm
from tvm import relax
from tvm import topi
from tvm import te
from tvm import tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

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

tvm.ir.assert_structural_equal(
	tvm.transform.Sequential([
		vtar.relax.transform.GraphPack(),
		relax.transform.CanonicalizeBindings(), # removes redundant assignments
	])(ConvModel),
	ConvModelPacked
)

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

from tvm import dlight as dl
from tvm import ir

@I.ir_module
class ALUOperations:
	@R.function
	def main(
		x: R.Tensor((1, 64, 1, 16), dtype="int32"),
		y: R.Tensor((1, 64, 1, 16), dtype="int32"),
	):
		with R.dataflow():
			# TODO: x + y * x
			lv = (x + y).astype("int8")
			gv = lv
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
	vtar.get_vtar_tir_transform(),
])

env = vtar.get_env()
target = tvm.target.Target(env.target, host=env.target_host)
with target:
	mod = almost_end2end_pipeline(mod)
ex = tvm.compile(mod, target=target)
raise SystemExit(0)

import numpy
from vtar.relax.transform import _get_shape

# It is necessary to use vtar_zero because otherwise we get
#     TVMError: CodeGenVM cannot handle this intrinsic now:
#     Op(relax.nn.conv2d)
# This is because CodeGenVM::VisitExpr_(const CallNode*) can handle very few
# operatons and Relax nodes are not among them; we have to convert them to
# R.call_tir with our custom legalize ops.
irmods = (ConvModel, ConvModelPacked)
vms = []
for irmod in irmods:
	irmod = relax.get_pipeline('vtar_zero')(irmod)
	irmod = tvm.transform.PrintIR()(irmod)
	dev = tvm.device('llvm')
	target = tvm.target.Target('llvm')
	vmexec = tvm.compile(irmod, target)
	vms.append(relax.VirtualMachine(vmexec, dev))

seed = 42
rng = numpy.random.default_rng(seed)
params = [_get_shape(param) for param in ConvModel['main'].params]
x = tvm.nd.array((rng.random(params[0])*255).astype('int8'), device=dev)
w = tvm.nd.array((rng.random(params[1])*255).astype('int8'), device=dev)
b = tvm.nd.array((rng.random(params[2])*255).astype('int32'), device=dev)
ok = numpy.all(numpy.equal(vms[0]['main'](x, w, b).numpy(), vms[1]['main'](x, w, b).numpy()))
if not ok:
	raise ValueError("Packed and not packed are not equal...")
