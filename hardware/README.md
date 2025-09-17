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

ssh xilinx@192.168.2.99
. cpython-3.8/tvm_runtime/bin/activate
cd tvm/
sudo -E ./apps/vta_rpc/start_rpc_server.sh
```
