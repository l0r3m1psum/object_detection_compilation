import fix_tvm_and_vtar_env
import os
os.environ["path"] = os.environ["path"] + ";" + os.path.join(os.getcwd(), r"submodules\tvm\build\RelWithDebInfo")

import tvm
from tvm import relay
import vta.testing
# from vta.testing import simulator
from tvm.contrib import graph_executor

import onnx

from PIL import Image
import numpy as np

import pickle

def closure(graph, lib, params):
	def myrun(env: vta.Environment, remote: tvm.rpc.RPCSession) -> None:
		nonlocal graph, lib, params

		lib.export_library("build/graphlib.tar")

		if env.TARGET == "zcu104":
			vta.reconfig_runtime(remote)
			vta.program_fpga(remote, bitstream=None)

		if env.TARGET == "sim":
			simulator.clear_stats()

		ctx = remote.ext_dev(0)
		remote.upload("build/graphlib.tar")
		lib = remote.load_module("graphlib.tar")
		m = graph_executor.create(graph, lib, ctx)

		# https://share.google/images/qCdEdjEnk9q0CzHmI
		image = Image.open("chainsaw.jpg").resize((224, 224))
		# plt.imshow(image)
		# plt.show(block=True)
		image = np.array(image) - np.array([123.0, 117.0, 104.0])
		image /= np.array([58.395, 57.12, 57.375])
		image = image.transpose((2, 0, 1))
		image = image[np.newaxis, :]
		image = np.repeat(image, env.BATCH, axis=0)

		m.set_input(**params)
		m.set_input("input", image)

		num = 4  # number of times we run module for a single measurement
		rep = 3  # number of measurements (we derive std dev from this)
		timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

		timer()

		if env.TARGET == "sim":
			sim_stats = simulator.stats()
			print("Execution statistics:")
			for k, v in sim_stats.items():
				# Since we execute the workload many times, we need to normalize stats
				# Note that there is always one warm up run
				# Therefore we divide the overall stats by (num * rep + 1)
				print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))

		tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 10), "float32", remote.cpu(0)))

		for b in range(env.BATCH):
			top_categories = np.argsort(tvm_output.numpy()[b])
			print(top_categories[-5:])
			break
			# Report top-5 classification results
			print("{} prediction for sample {}".format("resnet18_v1", b))
			print("\t#1:", synset[top_categories[-1]])
			print("\t#2:", synset[top_categories[-2]])
			print("\t#3:", synset[top_categories[-3]])
			# This just checks that one of the 3 top categories;
			# this is by no means an accurate assessment of how quantization affects
			# classification accuracy but is meant to catch changes to the quantization
			# pass that would accuracy in the CI.
			detected = False
			for k in top_categories[-3:]:
				if "chain saw" in synset[k]:
					detected = True
			if not detected: raise Exception("chainsaw not detected")

	return myrun

from tvm import autotvm
env = vta.get_env()


with autotvm.tophub.context(env.target):
	path = "build/resnet18.onnx"
	model_onnx = onnx.load(path)

	mod, params = relay.frontend.onnx.from_onnx(model_onnx, {"input": (env.BATCH, 3, 224, 224)})

	if not os.path.exists("build/relay_prog.pkl"):
		with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
			mod = relay.quantize.quantize(mod, params=params)

		relay_prog = vta.top.graph_pack(
			mod['main'],
			env.BATCH,
			env.BLOCK_OUT,
			env.WGT_WIDTH,
			start_name="nn.max_pool2d",
			stop_name="nn.global_avg_pool2d"
		)
		with open('build/relay_prog.pkl', 'wb') as f:
			pickle.dump(relay_prog, f)
	else:
		with open('build/relay_prog.pkl', 'rb') as f:
			relay_prog = pickle.load(f)


	assert env.BLOCK_IN == env.BLOCK_OUT

	synset = ["tench", "English springer", "cassette player", "chain saw", "church",
		"French horn", "garbage truck", "gas pump", "golf ball", "parachute",]

	with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
		graph, lib, params = relay.build(
			relay_prog, target=env.target, params=params, target_host=env.target_host
		)


	vta.testing.run(closure(graph, lib, params))
