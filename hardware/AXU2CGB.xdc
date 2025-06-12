set_property -dict { PACKAGE_PIN AB11 IOSTANDARD DIFF_SSTL12 } [get_ports sys_clk_p];
set_property -dict { PACKAGE_PIN W13  IOSTANDARD LVCMOS33    } [get_ports {led[0]}];
set_property -dict { PACKAGE_PIN Y12  IOSTANDARD LVCMOS33    } [get_ports {led[1]}];
set_property -dict { PACKAGE_PIN AA12 IOSTANDARD LVCMOS33    } [get_ports {led[2]}];
set_property -dict { PACKAGE_PIN AB13 IOSTANDARD LVCMOS33    } [get_ports {led[3]}];
set_property -dict { PACKAGE_PIN AA13 IOSTANDARD LVCMOS33    } [get_ports {key[0]}];
set_property -dict { PACKAGE_PIN AE14 IOSTANDARD LVCMOS33    } [get_ports {key[1]}];
set_property -dict { PACKAGE_PIN AE15 IOSTANDARD LVCMOS33    } [get_ports {key[2]}];
set_property -dict { PACKAGE_PIN AG14 IOSTANDARD LVCMOS33    } [get_ports {key[3]}];
set_property -dict { PACKAGE_PIN AB19 IOSTANDARD LVCMOS33    } [get_ports uart_tx_];

# create_clock -period 5.000 -name sys_clk_p -waveform {0.000 2.500} [get_ports sys_clk_p];