# https://projectf.io/posts/vivado-tcl-build-script/
# https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_1/ug835-vivado-tcl-commands.pdf

read_verilog "top.v"

read_xdc "AXU2CGB.xdc"

# synth_design -top "top" -part "xczu9eg-ffvb1156-2-e"
synth_design -top "top" -part "xczu2cg-sfvc784-1-e"

opt_design
place_design
route_design

write_bitstream -force "hello.bit"

################################################################################

open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {hello.bit} [current_hw_device]
program_hw_devices [current_hw_device]

################################################################################

if 0 {
	# https://adaptivesupport.amd.com/s/question/0D52E00006hpPHtSAM/error-common-1769-command-failed-writehwplatform-is-only-supported-for-synthesized-implemented-or-checkpoint-designs?language=en_US#:~:text=only%20for%20project%20mode
	write_checkpoint -force "hello_checkpoint.dcp"
	open_checkpoint "hello_checkpoint.dcp"

	# get_property platform.name [current_project]
	set_property platform.name plt_zcu102_hello [current_project]
	# get_property platform.board_id [current_project]
	set_property platform.board_id zcu102 [current_project]
	# TODO: hardware handoff
	# https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841693/HSI+debugging+and+optimization+techniques?f=print
	# https://github.com/Xilinx/workflow-decoupling-docs
	write_hw_platform -fixed -include_bit -force "hello.xsa"
}