from tvm import relax
import onnx
import vtar.relax.frontend.onnx

import os, sys
sys.path.append(os.path.join(os.getcwd(), "submodules/tvm/vta/python"))
import vta

path = "build/resnet18_int8.onnx"
model_onnx = onnx.load(path)

mod = vtar.relax.frontend.onnx.from_onnx(model_onnx, keep_params_in_input=False)

print(mod)

env = vta.get_env()

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
