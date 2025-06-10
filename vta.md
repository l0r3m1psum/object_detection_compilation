# VTA

TODO: verilator
https://www.chisel-lang.org/docs/installation#java-versions

```
curl -O "https://github.com/sbt/sbt/releases/download/v1.11.1/sbt-1.11.1.zip"
curl -O "https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.zip"
```

setup_env.sh
```
@echo off
set "PATH=%installdir%\Programs\sbt\bin;%PATH%"
REM https://stackoverflow.com/a/76151135
set "PATH=%installdir%\Programs\Java\jdk-17.0.12\bin;%PATH%"
```

# Pynq

Build requires Ubuntu 20.04,
[Vivado 2022.1](https://www.xilinx.com/member/forms/download/xef.html?filename=Xilinx_Unified_2022.1_0420_0327_Lin64.bin),
[petalinux-2022.1](https://www.xilinx.com/member/forms/download/xef.html?filename=petalinux-v2022.1-04191534-installer.run)

```
wget 'https://bit.ly/pynq_aarch64_v3_0_1' 'https://bit.ly/pynq_arm_v3_1' \
    'https://bit.ly/pynq_sdist_v3_0_1'
cp pynq_aarch64_v3_0_1 PYNQ/sdbuild/prebuilt/pynq_rootfs.aarch64.tar.gz
cp pynq_arm_v3_1       PYNQ/sdbuild/prebuilt/pynq_rootfs.arm.tar.gz
cp pynq_sdist_v3_0_1   PYNQ/sdbuild/prebuilt/pynq_sdist.tar.gz
```

[Pynq website](https://www.pynq.io/)
[Pynq repo](https://github.com/Xilinx/PYNQ)
[Pynq documentation](https://pynq.readthedocs.io/en/latest/)

[Petalinux website](http://www.xilinx.com/petalinux)
[Petalinux wiki](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842250/PetaLinux)
[PetaLinux 101 - Getting Started Quickly](https://www.youtube.com/watch?v=k03r2Ud42jY)

[Yocto](https://www.yoctoproject.org/)
[Getting Started with the Yocto Project - New Developer Screencast Tutorial](https://www.youtube.com/watch?v=zNLYanJAQ3s)
[Buildroot](https://buildroot.org/) is an easier alternative to Yocto

# AXU2CGB

https://www.hackster.io/mabushaqraedu0/axi-lite-slave-custom-ip-core-to-control-led-under-petalinux-b70b1d
https://www.hackster.io/matjaz4/gnu-radio-toolkit-on-axu2cgb-zynq-ultrascale-board-part2-ace235
