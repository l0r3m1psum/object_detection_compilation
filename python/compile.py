import tvm
from tvm import ir
from tvm.script import ir as I, relax as R

import os
import shutil

import numpy
import onnx

# PIL.Image.open(path)
@I.ir_module
class ImagePreprocessing:
	@R.function
	def preprocessings(input_image: R.Tensor(("h", "w", 3), dtype="uint8")) -> R.Tensor((1, 3, 224, 224), dtype="float32"):
		with R.dataflow():
			expand = R.expand_dims(input_image, axis=0)
			resize = R.image.resize2d(expand, size=(256, 256), layout="NHWC")
			crop = R.strided_slice(
				resize,
				axes=(1, 2),
				begin=(16, 16), # (256 - 224)//2 == 16
				end=(240, 240) # (256 + 224)//2 == 240
			)
			x_f32 = R.astype(crop, dtype="float32")
			x_scaled = x_f32/R.const(255.0, dtype="float32")
			imagenet_mean = R.const(numpy.array((0.485, 0.456, 0.406), dtype="float32"))
			imagenet_std = R.const(numpy.array((0.229, 0.224, 0.225), dtype="float32"))
			x_norm = (x_scaled - imagenet_mean)/imagenet_std
			output = R.permute_dims(x_norm, axes=(0, 3, 1, 2))
			R.output(output)
		return output

if True:
	mod = ImagePreprocessing
	mod.show()

	from tvm import relax
	target = tvm.target.Target("llvm")
	# When compiling with dynamically shaped tensors R.match_cast and T.match_buffer
	# and are generated to get the dimensions at runtime.
	ex = relax.build(mod, target)
	relax.get_pipeline('zero')(mod).show()
	vm = relax.VirtualMachine(ex, tvm.cpu())

	x = tvm.nd.array(numpy.ones((400, 600, 3), dtype="uint8"))
	y = vm["preprocessings"](x)

# shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
#     "submodules/tvm-vta/config/vta_config.json")
import vtar
shutil.copy("submodules/tvm-vta/config/fsim_sample.json",
    "submodules/tvm-vta/config/vta_config.json")
import vtar.relax.frontend.onnx

# TODO: make this library load when importing the vtar module
import ctypes
vta_fsim = ctypes.CDLL("vta_fsim")

env = vtar.get_env()
target = tvm.target.Target(env.target, host=env.target_host)

seq = tvm.transform.Sequential([
	vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping(),
	# Constant folding in TVM has some serious limitations. It can
	# only fold code by executing it hence if x is not know at
	# compile time even simple expressions like x + 0 are not
	# simplified.
	vtar.relax.transform.SimplifyConstAstype(),
	relax.transform.CanonicalizeBindings(), # necessary
	vtar.relax.transform.SimplifyRing(),
	# Some constant folding needs to be performed before graphpack
	# because TVM does not now how to execute NCHWnc convolution
	relax.transform.FoldConstant(),
	vtar.relax.transform.GraphPack(bitpack_end="relax.mean"),
	# GraphPack inserts some reshape, permute and pad that can be
	# folded away.
	relax.transform.FoldConstant(),
	vtar.relax.transform.AddChainSimplify(),
	relax.transform.CanonicalizeBindings(), # TODO: is this necessary?
	# TODO: write transform to put ReLU before astype
])

if not os.path.exists("build/resnet18_int8.json")
	onnx_model = onnx.load("build/resnet18_int8.onnx")
	mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
	mod = seq(mod)
	mod.show()
	with open("build/resnet18_int8.json", "w") as f:
		f.write(ir.save_json(mod))
else:
	with open("build/resnet18_int8.json") as f:
		mod = ir.load_json(f.read())

mod = vtar.relax.vtar_actual_pipeline()(mod)
mod.show()

ex = tvm.compile(
	mod,
	target=target,
	relax_pipeline=None,
	tir_pipeline=vtar.tir.get_actual_pipeline(),
)

ex.export_library(
	"build/resnet18_int8.dll",
	workspace_dir='build',
	options=(
		"-v", "-Wl,-verbose",
		"-g",
		"-L", os.path.expandvars('%installdir%\\Programs\\TVM\\lib'),
		"-l", "vta_fsim",
		"-l", "tvm_runtime",
		"-Wl,/DEBUG:FULL,/PDB:build\\resnet18_int8.pdb",
	)
)
