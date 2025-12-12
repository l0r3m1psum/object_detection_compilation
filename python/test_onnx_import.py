# import fix_tvm_and_vtar_env

import tvm
from tvm import relax
import onnx
if onnx.__version__ != '1.16.1':
	# https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/issues/541#issuecomment-2568218608
	import warnings
	# it fails in onnx.checker.check_model...
	warnings.warn("The onnx version is not the expected one, 1.16.1 seems to be the only working one")

import vtar
# This is needed to avoid:
# InternalError: Check failed: (allow_missing) is false: Device API ext_dev is not enabled.
# import vtar.testing
# import vtar.testing.simulator

import vtar.relax.frontend.onnx
import vtar.relax.transform

def prod(iterable, /, start=1):
	res = start
	for element in iterable:
		res *= element
	return res

def make_conv2d(
		counter, nodes, initializers,
		x_name: str, x_scale_name: str, x_zero_point_name: str,
		O: int, I: int, H: int, W: int,
		strides, pads
	) -> str:
	# The Bottlenect block in ResNet does not use bias, but when quantized the
	# bias is used to fuse the ReLU in the convolutional layer.

	weight_shape = (O, I, H, H) # OIHW
	scalar_shape = ()
	bias_shape = (O,)
	global conv_counter
	counter[0] += 1
	prefix = "conv%d_" % counter[0]

	w_name = prefix + "w"
	w_scale_name = prefix + "w_scale"
	w_zero_point_name = prefix + "w_zero_point"
	y_name = prefix + "y"
	y_scale_name = prefix + "y_scale"
	y_zero_point_name = prefix + "y_zero_point"
	b_name = prefix + "b"

	w            = onnx.helper.make_tensor(w_name,            onnx.TensorProto.INT8,  weight_shape, [i % 256 for i in range(prod(weight_shape))])
	w_scale      = onnx.helper.make_tensor(w_scale_name,      onnx.TensorProto.FLOAT, scalar_shape, [1])
	w_zero_point = onnx.helper.make_tensor(w_zero_point_name, onnx.TensorProto.INT8,  scalar_shape, [0])
	y_scale      = onnx.helper.make_tensor(y_scale_name,      onnx.TensorProto.FLOAT, scalar_shape, [1])
	y_zero_point = onnx.helper.make_tensor(y_zero_point_name, onnx.TensorProto.INT8,  scalar_shape, [0])
	b            = onnx.helper.make_tensor(b_name,            onnx.TensorProto.INT32, bias_shape,   [i % 256 for i in range(O)])
	initializers.extend([w, w_scale, w_zero_point, y_scale, y_zero_point, b])

	conv_node = onnx.helper.make_node(
		"QLinearConv",
		[
			x_name, x_scale_name, x_zero_point_name,
			w_name, w_scale_name, w_zero_point_name,
			y_scale_name, y_zero_point_name,
			b_name,
		],
		[y_name],
		kernel_shape=[H, W],
		strides=strides,
		pads=pads,
		name=prefix[:-1]
	)
	nodes.append(conv_node)

	return y_name, y_scale_name, y_zero_point_name

# this function is extremelly bad and just used to create a very simple model
def create_quantized_bottleneck_model():
	model_name = "quantized_bottleneck_model"
	input_shape = (1, 64, 56, 56)

	nodes = []
	initializers = []

	input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)
	input_scale = onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [], [0.01])
	input_zero_point = onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [], [128])
	initializers.extend([input_scale, input_zero_point])

	quantize_node = onnx.helper.make_node(
		"QuantizeLinear",
		["input", "input_scale", "input_zero_point"],
		["quantized_input"],
		name="quantize_input"
	)
	nodes.append(quantize_node)

	maxpool_node = onnx.helper.make_node(
		"MaxPool",
		["quantized_input"],
		["maxpool_output"],
		kernel_shape=[1, 1],
		strides=[1, 1],
		name="maxpool_layer"
	)
	nodes.append(maxpool_node)

	conv_counter = [0]
	conv1, conv1_scale, conv1_zero_point = make_conv2d(
		conv_counter, nodes, initializers,
		"maxpool_output", "input_scale", "input_zero_point",
		64, 64, 1, 1,
		[1,1], [0,0,0,0]
	)

	add_output_scale = onnx.helper.make_tensor("add_output_scale", onnx.TensorProto.FLOAT, [], [0.035])
	add_output_zero_point = onnx.helper.make_tensor("add_output_zero_point", onnx.TensorProto.INT8, [], [0])
	initializers.extend([add_output_scale, add_output_zero_point])

	qlinear_add_node = onnx.helper.make_node(
		"QLinearAdd",
		[
			"maxpool_output", "input_scale", "input_zero_point",
			conv1, conv1_scale, conv1_zero_point,
			"add_output_scale", "add_output_zero_point"
		],
		["bottleneck_output"],
		name="bottleneck_residual_add"
	)
	nodes.append(qlinear_add_node)

	gap_output_scale = onnx.helper.make_tensor("add_output_scale", onnx.TensorProto.FLOAT, [], [0.035])
	gap_output_zero_point = onnx.helper.make_tensor("add_output_zero_point", onnx.TensorProto.INT8, [], [0])
	global_average_pool_node = onnx.helper.make_node(
		"QLinearGlobalAveragePool",
		["bottleneck_output", "add_output_scale", "add_output_zero_point",
		gap_output_scale.name, gap_output_zero_point.name],
		["gap_output"],
		name="global_average_pool_layer"
	)
	nodes.append(global_average_pool_node)

	output_tensor_shape = (1, 64, 1, 1)
	dequantize_node = onnx.helper.make_node(
		"DequantizeLinear",
		["gap_output", gap_output_scale.name, gap_output_zero_point.name],
		["output"],
		name="dequantize_output"
	)
	nodes.append(dequantize_node)

	output_tensor = onnx.helper.make_tensor_value_info(
		"output", onnx.TensorProto.FLOAT, output_tensor_shape
	)

	graph = onnx.helper.make_graph(
		nodes,
		model_name,
		[input_tensor],
		[output_tensor],
		initializers,
	)

	model = onnx.helper.make_model(graph)
	# onnx.checker.check_model(model)
	return model

def compile(mod):
	# relax.transform.Normalize()
	# relax.transform.CanonicalizeBindings()
	# tvm.transform.PrintIR()
	mod = tvm.transform.Sequential(
	[
		tvm.transform.PrintIR(),
		relax.transform.FoldConstant(),
		vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping(),
		# vtar.relax.transform.GraphPack(),
		# tvm.transform.PrintIR(),
		# vtar.relax.transform.WrapMaxpoolDequantizeQuantize(),
		# tvm.transform.PrintIR(),
		relax.get_pipeline('vtar_zero'),
		tvm.transform.PrintIR(),
	])(mod)
	return mod

model_onnx = create_quantized_bottleneck_model()

import numpy
from tvm import topi

def remove_unused_arguments(mod):
	unused_params = {
		str(param): (param.checked_type.dtype, topi.utils.get_const_tuple(relax.get_shape_of(param)))
		for param in mod['main'].params
	}
	for i in range(len(mod['main'].body.blocks)):
		for j in range(len(mod['main'].body.blocks[i].bindings)):
			for k in range(len(mod['main'].body.blocks[i].bindings[j].value.args)):
				arg = mod['main'].body.blocks[i].bindings[j].value.args[k]
				_ = unused_params.pop(str(arg), None)
	params_to_remove = {
		key: tvm.nd.array(numpy.zeros(shape, dtype=dtype))
		for key, (dtype, shape) in unused_params.items()
	}
	# dict(zip(mod['main'].params[1:], params['main']))
	mod.update_func(
		mod.get_global_var('main'),
		mod['main'].bind_params(params_to_remove)
	)
	return mod

breakpoint()
mod = vtar.relax.frontend.onnx.from_onnx(model_onnx, keep_params_in_input=True)
mod = Module
mod, params = relax.frontend.detach_params(mod)
# mod = remove_unused_arguments(mod)

from tvm import tir

def make_closure_test_onnx_import(mod):
	def test_onnx_import(env: vtar.Environment, remote: tvm.rpc.RPCSession) -> None:
		nonlocal mod

		dev = tvm.device(str(env.target))
		target = tvm.target.Target(env.target, host=env.target_host)
		print(target)

		mod = compile(mod)
		with vtar.build_config():
			# The "main" disapears in tvm.relax.build, after _vmcodegen is called.
			# This problem seem to happen also in newer versions of TVM (0.20.0),
			# the cases in which it seems to work are just because TVM is able
			# to fuse all operations in a single function and _vmlink treats
			# that only function as the "main".
			ex = relax.build(mod, target)
		print(ex.as_python())
		# TODO: make it run (see if it is correct) and see how much is really offloaded to VTA
		# vm = relax.VirtualMachine(ex, dev)
		ex.export_library('build/qbottleneck.dll')
		remote.upload("build/qbottleneck.dll")
		f = remote.load_module("qbottleneck.dll")
		devr = remote.device(str(env.target))
		time_f = f.time_evaluator(f.entry_name, devr, number=1)
	return test_onnx_import

vtar.testing.run(make_closure_test_onnx_import(mod))

raise SystemExit(0)

path = "build/resnet18_int8.onnx"
# https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx
# path = "build/resnet50-v1-12-int8.onnx"
model_onnx = onnx.load(path)
mod = vtar.relax.frontend.onnx.from_onnx(model_onnx)
mod = compile(mod)

with vta.build_config():
	ex = relax.build(mod, target)
vm = relax.VirtualMachine(ex, dev)
# FIXME: make this cross platform
ex.export_library('build/resnet18_int8.dll')
remote = tvm.rpc.LocalSession()
remote.upload("build/resnet18_int8.dll")
f = remote.load_module("resnet18_int8.dll")
devr = remote.device(str(env.target))
time_f = f.time_evaluator(f.entry_name, devr, number=1)

# TODO: relax.op.image.resize2d

print(
	"inp_dtype: %s\n"
	"wgt_dtype: %s\n"
	"out_dtype: %s\n"
	"acc_dtype: %s\n"
	"BATCH: %d\n"
	"BLOCK_IN: %d\n"
	"BLOCK_OUT: %d" \
		% (env.inp_dtype,
			env.wgt_dtype,
			env.out_dtype,
			env.acc_dtype,
			env.BATCH,
			env.BLOCK_IN,
			env.BLOCK_OUT)
)

raise SystemExit(0)

if False:
	mod = vta.top.graph_pack(
		mod["main"],
		env.BATCH,
		env.BLOCK_OUT,
		env.WGT_WIDTH,
	)

zero_pipeline = relax.get_pipeline('zero')
mod = zero_pipeline(mod)

vta.build(mod)
