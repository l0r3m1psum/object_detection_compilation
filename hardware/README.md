Vitis serial stuff requires a [driver](https://www.silabs.com/documents/public/software/CP210x_Windows_Drivers.zip)

https://www.hackster.io/matjaz4/gnu-radio-toolkit-on-axu2cgb-zynq-ultrascale-board-part1-81a40b

https://www.hackster.io/mabushaqraedu0/axi-lite-slave-custom-ip-core-to-control-led-under-petalinux-b70b1d
https://www.hackster.io/caglayandokme/jtag-booting-embedded-linux-on-zynq-soc-cec756

THE GIUDE TO FOLLOW
https://www.centennialsoftwaresolutions.com/post/install-petalinux-tools-2023-1-on-wsl2-running-on-windows-10-build-and-run-the-vck190-bsp-on-qemu
THEN this fix for rlwrap
https://community.element14.com/technologies/fpga-group/f/forum/55215/vivado-vitis-and-petalinux-on-windows-sublayer-for-linux-wsl2-solve-xsdb-segmentation-fault


https://xilinx.github.io/Embedded-Design-Tutorials/docs/2023.1/build/html/docs/Introduction/Versal-EDT/docs/4-boot-and-config.html
```
petalinux-create -t project -n petalinux_AXU2CGB --template zynqMP
cd petalinux_AXU2CGB
petalinux-config --get-hw-description /mnt/c/Users/Diego/Source/AXU2CGA_AXU2CGB/vivado/design_1_wrapper.xsa
# Remove <cpu2> and <cpu3> from /pmu
vi components/plnx_workspace/device-tree/device-tree/zynqmp.dtsi
petalinux-buils
```

wsl files
https://askubuntu.com/a/1422805
https://learn.microsoft.com/en-us/windows/wsl/connect-usb


https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842019/Zynq+UltraScale+FSBL
https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841724/PMU+Firmware


To access WSL from Windows batch
```
git config --global --add safe.directory "*"
net use D: \\wsl.localhost\Ubuntu-22.04
```

https://www.centennialsoftwaresolutions.com/post/booting-u-boot-via-jtag-with-petalinux-tools

https://adaptivesupport.amd.com/s/question/0D52E00006mfjcgSAA/vfs-cannot-open-root-device-mmcblk0p2-or-unknownblock1792-error-30-kria-kv260?language=ja
https://adaptivesupport.amd.com/s/question/0D54U000076gLY5SAM/petalinux-20222-custom-board-not-booting-to-sd1?language=en_US
https://unix.stackexchange.com/questions/491315/vfs-cannot-open-root-device-mmcblk1p1-or-unknown-block179-33
https://docs.amd.com/r/en-US/ug1144-petalinux-tools-reference-guide/Partitioning-and-Formatting-an-SD-Card

https://learn.microsoft.com/en-us/windows/wsl/connect-usb

Connect to Pynq via IP

https://pynq.readthedocs.io/en/latest/getting_started/network_connection.html
https://superuser.com/a/996144
```
netsh interface ip set address name="Ethernet 2" static 192.168.2.1 255.255.255.0
```

```
cffi-1.17.1-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
cloudpickle-1.1.1-py2.py3-none-any.whl
numpy-1.19.0-cp38-cp38-manylinux2014_aarch64.whl
pycparser-2.22-py3-none-any.whl
cython-3.1.6-cp38-cp38-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl

ssh xilinx@192.168.2.99
sudo python3 -c "from pynq import Bitstream; bitstream_path = '1x16_i8w8a32_15_15_18_17.bit'; bitstream = Bitstream(bitstream_path); bitstream.download()"
. cpython-3.8/tvm_runtime/bin/activate
cd tvm/
# sudo -E ./apps/vta_rpc/start_rpc_server.sh
cd python # Since sudo -E ignores PYTHONPATH
sudo -E python -m tvm.exec.rpc_server --load-lib=libvta.so
```

Instructions for cross-compiling code for the VTA
```
# This must happen before importing vtar in this way the vtar.Environment
# object is initialized for cross-compilaiton.
shutil.copy("submodules/tvm-vta/config/zcu104_sample.json",
    "submodules/tvm-vta/config/vta_config.json")
import vtar
shutil.copy("submodules/tvm-vta/config/fsim_sample.json",
    "submodules/tvm-vta/config/vta_config.json")

env = vtar.get_env()
target = tvm.target.Target(env.target, host=env.target_host)
ex = tir.build(mod, target, vtar.get_vtar_tir_transform())
ex.export_library("build/gemm.tar")
```

Intructions for running code on the VTA on the ZCU104 with Pynq image
```
# Get Pynq from https://github.com/Xilinx/PYNQ/releases/tag/v2.5
curl -O "https://files.pythonhosted.org/packages/53/93/7e547ab4105969cc8c93b38a667b82a835dd2cc78f3a7dad6130cfd41e1d/cffi-1.17.1-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
curl -O "https://files.pythonhosted.org/packages/24/fb/4f92f8c0f40a0d728b4f3d5ec5ff84353e705d8ff5e3e447620ea98b06bd/cloudpickle-1.1.1-py2.py3-none-any.whl"
curl -O "https://files.pythonhosted.org/packages/5e/af/c5c302d5ddaadb1875552d4eb109925f1e818832d5f5b31663069d2c4dba/numpy-1.19.0-cp38-cp38-manylinux2014_aarch64.whl"
curl -O "https://files.pythonhosted.org/packages/13/a3/a812df4e2dd5696d1f351d58b8fe16a405b234ad2886a0dab9183fb78109/pycparser-2.22-py3-none-any.whl"
curl -O "https://files.pythonhosted.org/packages/0a/04/50bcc8c74bbc35c6f4b92c03bee9e930bb1685b019bb89d700b5642e2598/cython-3.1.6-cp38-cp38-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"

# Download the cpyhton3.8 zip form GitHub
# Download the tvm0.20 zip from Github together with its dependencies
# Download the tvm0.18 zip from Github together with its dependencies (this still contains libvta)
# Upload the bitstream to the Pynq
# Upload the exported library

ssh xilinx@192.168.137.48
sudo python3 -c "import pynq; pynq.Bitstream('1x16_i8w8a32_15_15_18_17.bit').download()"
# build cpython
# create and activate a virtual environment
# install the wheels
# build tvm0.18 runtime with "-frtti" and configuring the driver for zcu104 (https://tvm.hyper.ai/docs/0.12.0/topic/vta/install)
# copy libvta.so in tvm0.20/python/tvm
# build tvm0.20 runtime (https://tvm.apache.org/docs/how_to/tutorials/cross_compilation_and_rpc.html)
cd tvm0.20/python # So that tvm can be imported without modifing the PYTHONPATH
```
Run this with sudo -E
```
import tvm # Must be imported before loading libvta
import ctypes
import numpy
libvta = ctypes.CDLL("./libvta.so", ctypes.RTLD_GLOBAL)
# ref = tvm.get_global_func("device_api.ext_dev")() # This sets the global ext_dev pointer
func = tvm.runtime.load_module("../../alu.tar")
dev = tvm.ext_dev(0)
A = tvm.nd.array(numpy.ones((1, 64, 1, 16), dtype='int32'), dev)
B = tvm.nd.array(numpy.ones((1, 64, 1, 16), dtype='int32'), dev)
C = tvm.nd.empty((1, 64, 1, 16), 'int8', dev)
func(A, B, C)
```
