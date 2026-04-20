import tvm
from tvm import tir, arith, ir, te, topi, relax, dlight as dl
from tvm.script import tir as T
from tvm.script import ir as I
from tvm.script import relax as R

import shutil
# shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
#     "submodules/tvm-vta/config/vta_config.json")
import vtar
shutil.copy("submodules/tvm-vta/config/fsim_sample.json",
    "submodules/tvm-vta/config/vta_config.json")
import vtar.relax.frontend.onnx
import vtar.relax.transform

import os
from typing import List, Tuple, Dict

import numpy
import onnx
import PIL.Image


def main():
    # From Qualcomm zoo
    # onnx_model = onnx.load(r"C:\Users\Diego\Downloads\resnet18-resnet18-w8a8.onnx\job_jgovrkr45_optimized_onnx\model.onnx")
    # From STMicroelectronics zoo
    # onnx_model = onnx.load(r"build/resnet18wd4_pt_224_qdq_int8.onnx")
    # Despite the name this is the one taken from the ONNX zoo (ported from Gluon)
    # and then quantized by me (look at quantize.bat).
    onnx_model = onnx.load(r"build\resnet18-v2-7_int8_axis.onnx")
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    # mod = relax.transform.BindSymbolicVars({"batch_size": 64})(mod)
    mod.show()
    mod = vtar.relax.transform.RewriteQDQPatterns()(mod)
    # mod = vtar.relax.transform.ReScale()(mod)
    mod = vtar.relax.transform.ReQuantize()(mod)
    mod = vtar.relax.transform.LowerQNNOps()(mod)
    mod = relax.transform.FoldConstant()(mod)
    mod.show()
    ex = tvm.compile(mod)
    ex.export_library(r"build\gluon_resnet18_v1b_Opset18_ioa.dll")

    raise SystemExit(0)
    # from tvm.contrib.download import download
    from vtar.relax.transform import print_report
    convert_layout = relax.transform.ConvertLayout({
        "relax.nn.conv2d": ["NCHW1n16c", "OIHW16o16i"],
    })

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/mnist-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    # TODO: improve RemoveUnnecessaryDequantizeQuantizeWrapping
    mod = vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping()(mod)
    mod = relax.transform.FoldConstant()(mod)
    # mod = vtar.relax.transform.GraphPack(bitpack_start="relax.quantize", bitpack_end="relax.dequantize")
    mod = tir.transform.ForceNarrowIndexToInt32()(mod)
    mod.show()
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    raise SystemExit(0)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/vgg16-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/resnet50-v1-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/densenet-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/squeezenet1.0-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    # All of the above should go on VTA with no problems!

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/mobilenetv2-12-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    onnx_model = onnx.load(os.path.expandvars("%installdir%/Zoo/efficientnet-lite4-11-int8.onnx"))
    mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
    mod = print_report(mod)
    mod = relax.transform.LegalizeOps()(mod)

    raise SystemExit(0)

    mod = relax.transform.FoldConstant()(mod)
    mod.show()

    raise SystemExit(0)

    x = te.placeholder((16,), "float32", "x")
    dtype = te.const(topi.nn.SQNN_DTYPE_TO_CODE['int8'])
    scale = te.compute((1,), lambda i: tir.const(0.5, "float32"), "scale")
    zero_point = te.compute((1,), lambda i: tir.const(3,), "zero_point")
    y = topi.nn.simulated_quantize(x, dtype, scale, zero_point)
    f = te.create_prim_func((x, y))

if __name__ == "__main__":
    main()
