import shutil
import tvm
from tvm import tir

shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
    "submodules/tvm-vta/config/vta_config.json")
import vtar
shutil.copy("submodules/tvm-vta/config/fsim_sample.json",
    "submodules/tvm-vta/config/vta_config.json")

env = vtar.get_env()
target = tvm.target.Target(env.target, host=env.target_host)
workloads = vtar.topi.resnet18_workloads

for idx, wl in enumerate(workloads):
    if idx == 0:
        continue
    func = vtar.topi.sq_ioa_conv2d_NCHWnc_from_workload(
        wl, env.BATCH, env.BLOCK_IN, env.BLOCK_OUT
    )
    ex = tir.build(func, target, vtar.tir.get_actual_pipeline())
    ex.export_library(
        "build/conv2d_pc_%d_%dx%dx%d.tar" % (idx, wl.height, wl.width, wl.in_filter)
    )
    func = vtar.topi.sq_ioa_conv2d_NCHWnc_from_workload(
        wl, env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, imm_scale=3
    )
    ex = tir.build(func, target, vtar.tir.get_actual_pipeline())
    ex.export_library(
        "build/conv2d_pt_%d_%dx%dx%d.tar" % (idx, wl.height, wl.width, wl.in_filter)
    )

# TODO: Export also a Relax model that can run on the remote.
# TODO: Check that this is the correct way to export from a PC to the ZCU104 and
# that no compilation flags are missing.
