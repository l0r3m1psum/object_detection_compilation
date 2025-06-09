# https://projectf.io/posts/vivado-tcl-build-script/
# https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_1/ug835-vivado-tcl-commands.pdf

# read design sources (add one line for each file)
read_verilog "mytop.v"

# read constraints
read_xdc "myconstrains.xdc"

# synth
synth_design -top "processore_wrapper" -part "xczu9eg-ffvb1156-2-e"

# place and route
opt_design
place_design
route_design

# write bitstream
write_bitstream -force "hello.bit"

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