import numpy

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]).with_attr("global_symbol", "add_one"))

local_demo = False

if local_demo:
    target = "llvm"
else:
	# uname -i
    target = "llvm -mtriple=aarch64-linux-gnueabihf"

func = tvm.compile(mod, target=target)
# save the lib at a local temp folder
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)

if local_demo:
    remote = rpc.LocalSession()
else:
    host = "192.168.137.48"
    port = 9091
    remote = rpc.connect(host, port)

import vtar
import os
import shutil

# set "PYTHONPATH=%CD%\..\submodules\tvm\vta\python"
# The bitsream should be inside "zcu104\0_0_1\1x16_i8w8a32_15_15_18_17.bit"
os.environ["VTA_CACHE_PATH"] = os.path.join(os.environ["installdir"], "Programs/bitstreams")
# The code expect the HOME environment variable to exists.
os.environ["HOME"] = "workaround"
shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
        "submodules/tvm-vta/config/vta_config.json")
# vtar.reconfig_runtime(remote)
# vtar.program_fpga(remote, bitstream=None)

remote.upload(path)
func = remote.load_module("lib.tar")

dev = remote.cpu()
a = tvm.nd.array(numpy.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.nd.array(numpy.zeros(1024, dtype=A.dtype), dev)
# the function will run on the remote device
func(a, b)
numpy.testing.assert_equal(b.numpy(), a.numpy() + 1)

time_f = func.time_evaluator(func.entry_name, dev, number=10)
cost = time_f(a, b).mean
print("%g secs/op" % cost)