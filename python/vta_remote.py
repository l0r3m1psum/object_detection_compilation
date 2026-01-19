import tvm
from tvm import rpc
from tvm import relax

import numpy

import os

host = "192.168.137.48"
port = 9092
remote = rpc.connect(host, port)

remote.upload("build/resnet18_int8.tar")
lib = remote.load_module("resnet18_int8.tar")
bitstream_path = os.path.join(os.environ["installdir"], "Programs/bitstreams/zcu104/0_0_1/1x16_i8w8a32_15_15_18_17.bit")
remote.upload(bitstream_path)
init = remote.get_function("tvm.contrib.vta.init")
init("1x16_i8w8a32_15_15_18_17.bit")
start_recording = remote.get_function("tvm.contrib.vta.start_recording")
stop_recording = remote.get_function("tvm.contrib.vta.stop_recording")
reset_recording = remote.get_function("tvm.contrib.vta.reset_recording")
mark_recording = remote.get_function("tvm.contrib.vta.mark_recording")

dev = remote.ext_dev(0)
x = tvm.nd.array(numpy.ones((1, 3, 224, 224), dtype="float32"), dev)
vm = relax.VirtualMachine(lib, dev)
# vm["main"](x) # This is broken in TVM 0.20.0
vm.set_input("main", x)
vm.invoke_stateful("main")
y = vm.get_outputs("main")

start_recording(0.0002)
vm.invoke_stateful("main")
res = stop_recording()
INT = res.numpy()[:, 3]
print(INT.min(), INT.max())
