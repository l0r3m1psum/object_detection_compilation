import shutil
import csv
import tvm
from tvm import tir

shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
    "submodules/tvm-vta/config/vta_config.json")
import vtar
shutil.copy("submodules/tvm-vta/config/fsim_sample.json",
    "submodules/tvm-vta/config/vta_config.json")

def write_params_csv(func: tir.PrimFunc, name: str) -> None:
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, param in enumerate(func.struct_info.params):
            shape = [v.value for v in param.shape.values]
            dtype = param.dtype
            writer.writerow(shape + [dtype])


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
    ex.export_library("build/resnet_conv_%d_per_axis.tar" % idx)
    write_params_csv(func, "build/resnet_conv_%d_per_axis.csv" % idx)

    func = vtar.topi.sq_ioa_conv2d_NCHWnc_from_workload(
        wl, env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, imm_scale=3
    )
    ex = tir.build(func, target, vtar.tir.get_actual_pipeline())
    ex.export_library("build/resnet_conv_%d_per_tensor.tar" % idx)
    write_params_csv(func, "build/resnet_conv_%d_per_tensor.csv" % idx)

# TODO: Export also a Relax model that can run on the remote.
