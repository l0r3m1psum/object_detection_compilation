import numpy

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils
import vtar

import os
os.environ["VTA_CACHE_PATH"] = os.path.join(os.environ["installdir"], "\\Programs\\bitstreams")

host = "192.168.137.48"
port = 9091
remote = rpc.connect(host, port)
# vtar.program_fpga(remote, bitstream=None)
# rng = numpy.random.default_rng(42)
dev = remote.ext_dev(0)
# A = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
A = tvm.nd.empty((1, 64, 1, 16), "int32", dev)
# B = tvm.nd.array((rng.uniform(size=(1, 64, 1, 16)) * 10).astype("int32"), dev)
B = tvm.nd.empty((1, 64, 1, 16), "int32", dev)
# C = tvm.nd.array(numpy.zeros((1, 64, 1, 16), dtype="int8"), dev)
C = tvm.nd.empty((1, 64, 1, 16), "int8", dev)

remote.upload("vta.tar")
func = remote.load_module("vta.tar")
func(A, B, C)
