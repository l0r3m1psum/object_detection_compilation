# TOOD: https://github.com/pulp-platform/pulp/blob/b6ae54700b76395b049742ebfc52c5aaf6e148a5/fpga/pulp-zcu102/constraints/zcu102.xdc

set_property -dict { PACKAGE_PIN AK15 IOSTANDARD DIFF_SSTL12  } [get_ports sys_clk_p];
set_property -dict { PACKAGE_PIN AL12 IOSTANDARD LVCMOS33 } [get_ports led];
