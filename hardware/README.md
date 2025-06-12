Vitis serial stuff requires a [driver](https://www.silabs.com/documents/public/software/CP210x_Windows_Drivers.zip)

https://www.hackster.io/matjaz4/gnu-radio-toolkit-on-axu2cgb-zynq-ultrascale-board-part1-81a40b

https://www.hackster.io/mabushaqraedu0/axi-lite-slave-custom-ip-core-to-control-led-under-petalinux-b70b1d
https://www.hackster.io/caglayandokme/jtag-booting-embedded-linux-on-zynq-soc-cec756
https://xilinx.github.io/Embedded-Design-Tutorials/docs/2023.1/build/html/docs/Introduction/Versal-EDT/docs/4-boot-and-config.html



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