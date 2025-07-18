`timescale 1 ps / 1 ps

module top
	processore_zynq_ultra_ps_e_0_0 zynq_ultra_ps_e_0();
endmodule


module processore_zynq_ultra_ps_e_0_0 (
	pl_resetn0,
	pl_clk0
);

	(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME PL_RESETN0, POLARITY ACTIVE_LOW, INSERT_VIP 0" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 PL_RESETN0 RST" *)
	output wire pl_resetn0;
	(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME PL_CLK0, FREQ_HZ 99990005, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN processore_zynq_ultra_ps_e_0_0_pl_clk0, INSERT_VIP 0" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 PL_CLK0 CLK" *)
	output wire pl_clk0;

	zynq_ultra_ps_e_v3_4_0_zynq_ultra_ps_e #(
		.C_DP_USE_AUDIO(0),
		.C_DP_USE_VIDEO(0),
		.C_MAXIGP0_DATA_WIDTH(128),
		.C_MAXIGP1_DATA_WIDTH(128),
		.C_MAXIGP2_DATA_WIDTH(32),
		.C_SAXIGP0_DATA_WIDTH(128),
		.C_SAXIGP1_DATA_WIDTH(128),
		.C_SAXIGP2_DATA_WIDTH(128),
		.C_SAXIGP3_DATA_WIDTH(128),
		.C_SAXIGP4_DATA_WIDTH(128),
		.C_SAXIGP5_DATA_WIDTH(128),
		.C_SAXIGP6_DATA_WIDTH(128),
		.C_USE_DIFF_RW_CLK_GP0(0),
		.C_USE_DIFF_RW_CLK_GP1(0),
		.C_USE_DIFF_RW_CLK_GP2(0),
		.C_USE_DIFF_RW_CLK_GP3(0),
		.C_USE_DIFF_RW_CLK_GP4(0),
		.C_USE_DIFF_RW_CLK_GP5(0),
		.C_USE_DIFF_RW_CLK_GP6(0),
		.C_EN_FIFO_ENET0("0"),
		.C_EN_FIFO_ENET1("0"),
		.C_EN_FIFO_ENET2("0"),
		.C_EN_FIFO_ENET3("0"),
		.C_PL_CLK0_BUF("TRUE"),
		.C_PL_CLK1_BUF("FALSE"),
		.C_PL_CLK2_BUF("FALSE"),
		.C_PL_CLK3_BUF("FALSE"),
		.C_TRACE_PIPELINE_WIDTH(8),
		.C_EN_EMIO_TRACE(0),
		.C_TRACE_DATA_WIDTH(32),
		.C_USE_DEBUG_TEST(0),
		.C_SD0_INTERNAL_BUS_WIDTH(5),
		.C_SD1_INTERNAL_BUS_WIDTH(8),
		.C_NUM_F2P_0_INTR_INPUTS(1),
		.C_NUM_F2P_1_INTR_INPUTS(1),
		.C_EMIO_GPIO_WIDTH(1),
		.C_NUM_FABRIC_RESETS(1)
	) inst (
		.maxihpm0_fpd_aclk(1'B0), .maxigp0_awid(), .maxigp0_awaddr(), .maxigp0_awlen(), .maxigp0_awsize(), .maxigp0_awburst(), .maxigp0_awlock(), .maxigp0_awcache(), .maxigp0_awprot(), .maxigp0_awvalid(), .maxigp0_awuser(), .maxigp0_awready(1'B0), .maxigp0_wdata(), .maxigp0_wstrb(), .maxigp0_wlast(), .maxigp0_wvalid(), .maxigp0_wready(1'B0), .maxigp0_bid(16'B0), .maxigp0_bresp(2'B0), .maxigp0_bvalid(1'B0), .maxigp0_bready(), .maxigp0_arid(), .maxigp0_araddr(), .maxigp0_arlen(), .maxigp0_arsize(), .maxigp0_arburst(), .maxigp0_arlock(), .maxigp0_arcache(), .maxigp0_arprot(), .maxigp0_arvalid(), .maxigp0_aruser(), .maxigp0_arready(1'B0), .maxigp0_rid(16'B0), .maxigp0_rdata(128'B0), .maxigp0_rresp(2'B0), .maxigp0_rlast(1'B0), .maxigp0_rvalid(1'B0), .maxigp0_rready(), .maxigp0_awqos(), .maxigp0_arqos(),
		.maxihpm1_fpd_aclk(1'B0), .maxigp1_awid(), .maxigp1_awaddr(), .maxigp1_awlen(), .maxigp1_awsize(), .maxigp1_awburst(), .maxigp1_awlock(), .maxigp1_awcache(), .maxigp1_awprot(), .maxigp1_awvalid(), .maxigp1_awuser(), .maxigp1_awready(1'B0), .maxigp1_wdata(), .maxigp1_wstrb(), .maxigp1_wlast(), .maxigp1_wvalid(), .maxigp1_wready(1'B0), .maxigp1_bid(16'B0), .maxigp1_bresp(2'B0), .maxigp1_bvalid(1'B0), .maxigp1_bready(), .maxigp1_arid(), .maxigp1_araddr(), .maxigp1_arlen(), .maxigp1_arsize(), .maxigp1_arburst(), .maxigp1_arlock(), .maxigp1_arcache(), .maxigp1_arprot(), .maxigp1_arvalid(), .maxigp1_aruser(), .maxigp1_arready(1'B0), .maxigp1_rid(16'B0), .maxigp1_rdata(128'B0), .maxigp1_rresp(2'B0), .maxigp1_rlast(1'B0), .maxigp1_rvalid(1'B0), .maxigp1_rready(), .maxigp1_awqos(), .maxigp1_arqos(),

		.maxihpm0_lpd_aclk(1'B0), // TYPO???

		.maxigp2_awid(), .maxigp2_awaddr(), .maxigp2_awlen(), .maxigp2_awsize(), .maxigp2_awburst(), .maxigp2_awlock(), .maxigp2_awcache(), .maxigp2_awprot(), .maxigp2_awvalid(), .maxigp2_awuser(), .maxigp2_awready(1'B0), .maxigp2_wdata(), .maxigp2_wstrb(), .maxigp2_wlast(), .maxigp2_wvalid(), .maxigp2_wready(1'B0), .maxigp2_bid(16'B0), .maxigp2_bresp(2'B0), .maxigp2_bvalid(1'B0), .maxigp2_bready(), .maxigp2_arid(), .maxigp2_araddr(), .maxigp2_arlen(), .maxigp2_arsize(), .maxigp2_arburst(), .maxigp2_arlock(), .maxigp2_arcache(), .maxigp2_arprot(), .maxigp2_arvalid(), .maxigp2_aruser(), .maxigp2_arready(1'B0), .maxigp2_rid(16'B0), .maxigp2_rdata(32'B0), .maxigp2_rresp(2'B0), .maxigp2_rlast(1'B0), .maxigp2_rvalid(1'B0), .maxigp2_rready(), .maxigp2_awqos(), .maxigp2_arqos(),

		.ddrc_ext_refresh_rank0_req(1'B0),
		.ddrc_ext_refresh_rank1_req(1'B0),
		.ddrc_refresh_pl_clk(1'B0),

		.saxihpc0_fpd_rclk(1'B0), .saxihpc0_fpd_wclk(1'B0), .saxihpc0_fpd_aclk(1'B0), .saxigp0_aruser(1'B0), .saxigp0_awuser(1'B0), .saxigp0_awid(6'B0), .saxigp0_awaddr(49'B0), .saxigp0_awlen(8'B0), .saxigp0_awsize(3'B0), .saxigp0_awburst(2'B0), .saxigp0_awlock(1'B0), .saxigp0_awcache(4'B0), .saxigp0_awprot(3'B0), .saxigp0_awvalid(1'B0), .saxigp0_awready(), .saxigp0_wdata(128'B0), .saxigp0_wstrb(16'B0), .saxigp0_wlast(1'B0), .saxigp0_wvalid(1'B0), .saxigp0_wready(), .saxigp0_bid(), .saxigp0_bresp(), .saxigp0_bvalid(), .saxigp0_bready(1'B0), .saxigp0_arid(6'B0), .saxigp0_araddr(49'B0), .saxigp0_arlen(8'B0), .saxigp0_arsize(3'B0), .saxigp0_arburst(2'B0), .saxigp0_arlock(1'B0), .saxigp0_arcache(4'B0), .saxigp0_arprot(3'B0), .saxigp0_arvalid(1'B0), .saxigp0_arready(), .saxigp0_rid(), .saxigp0_rdata(), .saxigp0_rresp(), .saxigp0_rlast(), .saxigp0_rvalid(), .saxigp0_rready(1'B0), .saxigp0_awqos(4'B0), .saxigp0_arqos(4'B0), .saxigp0_rcount(), .saxigp0_wcount(), .saxigp0_racount(), .saxigp0_wacount(),
		.saxihpc1_fpd_rclk(1'B0), .saxihpc1_fpd_wclk(1'B0), .saxihpc1_fpd_aclk(1'B0), .saxigp1_aruser(1'B0), .saxigp1_awuser(1'B0), .saxigp1_awid(6'B0), .saxigp1_awaddr(49'B0), .saxigp1_awlen(8'B0), .saxigp1_awsize(3'B0), .saxigp1_awburst(2'B0), .saxigp1_awlock(1'B0), .saxigp1_awcache(4'B0), .saxigp1_awprot(3'B0), .saxigp1_awvalid(1'B0), .saxigp1_awready(), .saxigp1_wdata(128'B0), .saxigp1_wstrb(16'B0), .saxigp1_wlast(1'B0), .saxigp1_wvalid(1'B0), .saxigp1_wready(), .saxigp1_bid(), .saxigp1_bresp(), .saxigp1_bvalid(), .saxigp1_bready(1'B0), .saxigp1_arid(6'B0), .saxigp1_araddr(49'B0), .saxigp1_arlen(8'B0), .saxigp1_arsize(3'B0), .saxigp1_arburst(2'B0), .saxigp1_arlock(1'B0), .saxigp1_arcache(4'B0), .saxigp1_arprot(3'B0), .saxigp1_arvalid(1'B0), .saxigp1_arready(), .saxigp1_rid(), .saxigp1_rdata(), .saxigp1_rresp(), .saxigp1_rlast(), .saxigp1_rvalid(), .saxigp1_rready(1'B0), .saxigp1_awqos(4'B0), .saxigp1_arqos(4'B0), .saxigp1_rcount(), .saxigp1_wcount(), .saxigp1_racount(), .saxigp1_wacount(),

		.saxihp0_fpd_rclk(1'B0), .saxihp0_fpd_wclk(1'B0), .saxihp0_fpd_aclk(1'B0), .saxigp2_aruser(1'B0), .saxigp2_awuser(1'B0), .saxigp2_awid(6'B0), .saxigp2_awaddr(49'B0), .saxigp2_awlen(8'B0), .saxigp2_awsize(3'B0), .saxigp2_awburst(2'B0), .saxigp2_awlock(1'B0), .saxigp2_awcache(4'B0), .saxigp2_awprot(3'B0), .saxigp2_awvalid(1'B0), .saxigp2_awready(), .saxigp2_wdata(128'B0), .saxigp2_wstrb(16'B0), .saxigp2_wlast(1'B0), .saxigp2_wvalid(1'B0), .saxigp2_wready(), .saxigp2_bid(), .saxigp2_bresp(), .saxigp2_bvalid(), .saxigp2_bready(1'B0), .saxigp2_arid(6'B0), .saxigp2_araddr(49'B0), .saxigp2_arlen(8'B0), .saxigp2_arsize(3'B0), .saxigp2_arburst(2'B0), .saxigp2_arlock(1'B0), .saxigp2_arcache(4'B0), .saxigp2_arprot(3'B0), .saxigp2_arvalid(1'B0), .saxigp2_arready(), .saxigp2_rid(), .saxigp2_rdata(), .saxigp2_rresp(), .saxigp2_rlast(), .saxigp2_rvalid(), .saxigp2_rready(1'B0), .saxigp2_awqos(4'B0), .saxigp2_arqos(4'B0), .saxigp2_rcount(), .saxigp2_wcount(), .saxigp2_racount(), .saxigp2_wacount(),
		.saxihp1_fpd_rclk(1'B0), .saxihp1_fpd_wclk(1'B0), .saxihp1_fpd_aclk(1'B0), .saxigp3_aruser(1'B0), .saxigp3_awuser(1'B0), .saxigp3_awid(6'B0), .saxigp3_awaddr(49'B0), .saxigp3_awlen(8'B0), .saxigp3_awsize(3'B0), .saxigp3_awburst(2'B0), .saxigp3_awlock(1'B0), .saxigp3_awcache(4'B0), .saxigp3_awprot(3'B0), .saxigp3_awvalid(1'B0), .saxigp3_awready(), .saxigp3_wdata(128'B0), .saxigp3_wstrb(16'B0), .saxigp3_wlast(1'B0), .saxigp3_wvalid(1'B0), .saxigp3_wready(), .saxigp3_bid(), .saxigp3_bresp(), .saxigp3_bvalid(), .saxigp3_bready(1'B0), .saxigp3_arid(6'B0), .saxigp3_araddr(49'B0), .saxigp3_arlen(8'B0), .saxigp3_arsize(3'B0), .saxigp3_arburst(2'B0), .saxigp3_arlock(1'B0), .saxigp3_arcache(4'B0), .saxigp3_arprot(3'B0), .saxigp3_arvalid(1'B0), .saxigp3_arready(), .saxigp3_rid(), .saxigp3_rdata(), .saxigp3_rresp(), .saxigp3_rlast(), .saxigp3_rvalid(), .saxigp3_rready(1'B0), .saxigp3_awqos(4'B0), .saxigp3_arqos(4'B0), .saxigp3_rcount(), .saxigp3_wcount(), .saxigp3_racount(), .saxigp3_wacount(),
		.saxihp2_fpd_rclk(1'B0), .saxihp2_fpd_wclk(1'B0), .saxihp2_fpd_aclk(1'B0), .saxigp4_aruser(1'B0), .saxigp4_awuser(1'B0), .saxigp4_awid(6'B0), .saxigp4_awaddr(49'B0), .saxigp4_awlen(8'B0), .saxigp4_awsize(3'B0), .saxigp4_awburst(2'B0), .saxigp4_awlock(1'B0), .saxigp4_awcache(4'B0), .saxigp4_awprot(3'B0), .saxigp4_awvalid(1'B0), .saxigp4_awready(), .saxigp4_wdata(128'B0), .saxigp4_wstrb(16'B0), .saxigp4_wlast(1'B0), .saxigp4_wvalid(1'B0), .saxigp4_wready(), .saxigp4_bid(), .saxigp4_bresp(), .saxigp4_bvalid(), .saxigp4_bready(1'B0), .saxigp4_arid(6'B0), .saxigp4_araddr(49'B0), .saxigp4_arlen(8'B0), .saxigp4_arsize(3'B0), .saxigp4_arburst(2'B0), .saxigp4_arlock(1'B0), .saxigp4_arcache(4'B0), .saxigp4_arprot(3'B0), .saxigp4_arvalid(1'B0), .saxigp4_arready(), .saxigp4_rid(), .saxigp4_rdata(), .saxigp4_rresp(), .saxigp4_rlast(), .saxigp4_rvalid(), .saxigp4_rready(1'B0), .saxigp4_awqos(4'B0), .saxigp4_arqos(4'B0), .saxigp4_rcount(), .saxigp4_wcount(), .saxigp4_racount(), .saxigp4_wacount(),
		.saxihp3_fpd_rclk(1'B0), .saxihp3_fpd_wclk(1'B0), .saxihp3_fpd_aclk(1'B0), .saxigp5_aruser(1'B0), .saxigp5_awuser(1'B0), .saxigp5_awid(6'B0), .saxigp5_awaddr(49'B0), .saxigp5_awlen(8'B0), .saxigp5_awsize(3'B0), .saxigp5_awburst(2'B0), .saxigp5_awlock(1'B0), .saxigp5_awcache(4'B0), .saxigp5_awprot(3'B0), .saxigp5_awvalid(1'B0), .saxigp5_awready(), .saxigp5_wdata(128'B0), .saxigp5_wstrb(16'B0), .saxigp5_wlast(1'B0), .saxigp5_wvalid(1'B0), .saxigp5_wready(), .saxigp5_bid(), .saxigp5_bresp(), .saxigp5_bvalid(), .saxigp5_bready(1'B0), .saxigp5_arid(6'B0), .saxigp5_araddr(49'B0), .saxigp5_arlen(8'B0), .saxigp5_arsize(3'B0), .saxigp5_arburst(2'B0), .saxigp5_arlock(1'B0), .saxigp5_arcache(4'B0), .saxigp5_arprot(3'B0), .saxigp5_arvalid(1'B0), .saxigp5_arready(), .saxigp5_rid(), .saxigp5_rdata(), .saxigp5_rresp(), .saxigp5_rlast(), .saxigp5_rvalid(), .saxigp5_rready(1'B0), .saxigp5_awqos(4'B0), .saxigp5_arqos(4'B0), .saxigp5_rcount(), .saxigp5_wcount(), .saxigp5_racount(), .saxigp5_wacount(),

		.saxi_lpd_rclk(1'B0), .saxi_lpd_wclk(1'B0), .saxi_lpd_aclk(1'B0),
		.saxigp6_aruser(1'B0), .saxigp6_awuser(1'B0), .saxigp6_awid(6'B0), .saxigp6_awaddr(49'B0), .saxigp6_awlen(8'B0), .saxigp6_awsize(3'B0), .saxigp6_awburst(2'B0), .saxigp6_awlock(1'B0), .saxigp6_awcache(4'B0), .saxigp6_awprot(3'B0), .saxigp6_awvalid(1'B0), .saxigp6_awready(), .saxigp6_wdata(128'B0), .saxigp6_wstrb(16'B0), .saxigp6_wlast(1'B0), .saxigp6_wvalid(1'B0), .saxigp6_wready(), .saxigp6_bid(), .saxigp6_bresp(), .saxigp6_bvalid(), .saxigp6_bready(1'B0), .saxigp6_arid(6'B0), .saxigp6_araddr(49'B0), .saxigp6_arlen(8'B0), .saxigp6_arsize(3'B0), .saxigp6_arburst(2'B0), .saxigp6_arlock(1'B0), .saxigp6_arcache(4'B0), .saxigp6_arprot(3'B0), .saxigp6_arvalid(1'B0), .saxigp6_arready(), .saxigp6_rid(), .saxigp6_rdata(), .saxigp6_rresp(), .saxigp6_rlast(), .saxigp6_rvalid(), .saxigp6_rready(1'B0), .saxigp6_awqos(4'B0), .saxigp6_arqos(4'B0), .saxigp6_rcount(), .saxigp6_wcount(), .saxigp6_racount(), .saxigp6_wacount(),

		.saxiacp_fpd_aclk(1'B0), .saxiacp_awuser(2'B0), .saxiacp_aruser(2'B0), .saxiacp_awid(5'B0), .saxiacp_awaddr(40'B0), .saxiacp_awlen(8'B0), .saxiacp_awsize(3'B0), .saxiacp_awburst(2'B0), .saxiacp_awlock(1'B0), .saxiacp_awcache(4'B0), .saxiacp_awprot(3'B0), .saxiacp_awvalid(1'B0), .saxiacp_awready(), .saxiacp_wdata(128'B0), .saxiacp_wstrb(16'B0), .saxiacp_wlast(1'B0), .saxiacp_wvalid(1'B0), .saxiacp_wready(), .saxiacp_bid(), .saxiacp_bresp(), .saxiacp_bvalid(), .saxiacp_bready(1'B0), .saxiacp_arid(5'B0), .saxiacp_araddr(40'B0), .saxiacp_arlen(8'B0), .saxiacp_arsize(3'B0), .saxiacp_arburst(2'B0), .saxiacp_arlock(1'B0), .saxiacp_arcache(4'B0), .saxiacp_arprot(3'B0), .saxiacp_arvalid(1'B0), .saxiacp_arready(), .saxiacp_rid(), .saxiacp_rdata(), .saxiacp_rresp(), .saxiacp_rlast(), .saxiacp_rvalid(), .saxiacp_rready(1'B0), .saxiacp_awqos(4'B0), .saxiacp_arqos(4'B0),

		.sacefpd_aclk(1'B0), .sacefpd_wuser(1'B0), .sacefpd_buser(), .sacefpd_ruser(), .sacefpd_awuser(16'B0), .sacefpd_awsnoop(3'B0), .sacefpd_awsize(3'B0), .sacefpd_awregion(4'B0), .sacefpd_awqos(4'B0), .sacefpd_awprot(3'B0), .sacefpd_awlen(8'B0), .sacefpd_awid(6'B0), .sacefpd_awdomain(2'B0), .sacefpd_awcache(4'B0), .sacefpd_awburst(2'B0), .sacefpd_awbar(2'B0), .sacefpd_awaddr(44'B0), .sacefpd_awlock(1'B0), .sacefpd_awvalid(1'B0), .sacefpd_awready(), .sacefpd_wstrb(16'B0), .sacefpd_wdata(128'B0), .sacefpd_wlast(1'B0), .sacefpd_wvalid(1'B0), .sacefpd_wready(), .sacefpd_bresp(), .sacefpd_bid(), .sacefpd_bvalid(), .sacefpd_bready(1'B0), .sacefpd_aruser(16'B0), .sacefpd_arsnoop(4'B0), .sacefpd_arsize(3'B0), .sacefpd_arregion(4'B0), .sacefpd_arqos(4'B0), .sacefpd_arprot(3'B0), .sacefpd_arlen(8'B0), .sacefpd_arid(6'B0), .sacefpd_ardomain(2'B0), .sacefpd_arcache(4'B0), .sacefpd_arburst(2'B0), .sacefpd_arbar(2'B0), .sacefpd_araddr(44'B0), .sacefpd_arlock(1'B0), .sacefpd_arvalid(1'B0), .sacefpd_arready(), .sacefpd_rresp(), .sacefpd_rid(), .sacefpd_rdata(), .sacefpd_rlast(), .sacefpd_rvalid(), .sacefpd_rready(1'B0), .sacefpd_acsnoop(), .sacefpd_acprot(), .sacefpd_acaddr(), .sacefpd_acvalid(), .sacefpd_acready(1'B0), .sacefpd_cddata(128'B0), .sacefpd_cdlast(1'B0), .sacefpd_cdvalid(1'B0), .sacefpd_cdready(), .sacefpd_crresp(5'B0), .sacefpd_crvalid(1'B0), .sacefpd_crready(), .sacefpd_wack(1'B0), .sacefpd_rack(1'B0),

		.emio_can0_phy_tx(), .emio_can0_phy_rx(1'B0),
		.emio_can1_phy_tx(), .emio_can1_phy_rx(1'B0),

		.emio_enet0_gmii_rx_clk(1'B0), .emio_enet0_speed_mode(), .emio_enet0_gmii_crs(1'B0), .emio_enet0_gmii_col(1'B0), .emio_enet0_gmii_rxd(8'B0), .emio_enet0_gmii_rx_er(1'B0), .emio_enet0_gmii_rx_dv(1'B0), .emio_enet0_gmii_tx_clk(1'B0), .emio_enet0_gmii_txd(), .emio_enet0_gmii_tx_en(), .emio_enet0_gmii_tx_er(), .emio_enet0_mdio_mdc(), .emio_enet0_mdio_i(1'B0), .emio_enet0_mdio_o(), .emio_enet0_mdio_t(),
		.emio_enet1_gmii_rx_clk(1'B0), .emio_enet1_speed_mode(), .emio_enet1_gmii_crs(1'B0), .emio_enet1_gmii_col(1'B0), .emio_enet1_gmii_rxd(8'B0), .emio_enet1_gmii_rx_er(1'B0), .emio_enet1_gmii_rx_dv(1'B0), .emio_enet1_gmii_tx_clk(1'B0), .emio_enet1_gmii_txd(), .emio_enet1_gmii_tx_en(), .emio_enet1_gmii_tx_er(), .emio_enet1_mdio_mdc(), .emio_enet1_mdio_i(1'B0), .emio_enet1_mdio_o(), .emio_enet1_mdio_t(),
		.emio_enet2_gmii_rx_clk(1'B0), .emio_enet2_speed_mode(), .emio_enet2_gmii_crs(1'B0), .emio_enet2_gmii_col(1'B0), .emio_enet2_gmii_rxd(8'B0), .emio_enet2_gmii_rx_er(1'B0), .emio_enet2_gmii_rx_dv(1'B0), .emio_enet2_gmii_tx_clk(1'B0), .emio_enet2_gmii_txd(), .emio_enet2_gmii_tx_en(), .emio_enet2_gmii_tx_er(), .emio_enet2_mdio_mdc(), .emio_enet2_mdio_i(1'B0), .emio_enet2_mdio_o(), .emio_enet2_mdio_t(),
		.emio_enet3_gmii_rx_clk(1'B0), .emio_enet3_speed_mode(), .emio_enet3_gmii_crs(1'B0), .emio_enet3_gmii_col(1'B0), .emio_enet3_gmii_rxd(8'B0), .emio_enet3_gmii_rx_er(1'B0), .emio_enet3_gmii_rx_dv(1'B0), .emio_enet3_gmii_tx_clk(1'B0), .emio_enet3_gmii_txd(), .emio_enet3_gmii_tx_en(), .emio_enet3_gmii_tx_er(), .emio_enet3_mdio_mdc(), .emio_enet3_mdio_i(1'B0), .emio_enet3_mdio_o(), .emio_enet3_mdio_t(),

		.emio_enet0_tx_r_data_rdy(1'B0), .emio_enet0_tx_r_rd(), .emio_enet0_tx_r_valid(1'B0), .emio_enet0_tx_r_data(8'B0), .emio_enet0_tx_r_sop(1'B1), .emio_enet0_tx_r_eop(1'B1), .emio_enet0_tx_r_err(1'B0), .emio_enet0_tx_r_underflow(1'B0), .emio_enet0_tx_r_flushed(1'B0), .emio_enet0_tx_r_control(1'B0), .emio_enet0_dma_tx_end_tog(), .emio_enet0_dma_tx_status_tog(1'B0), .emio_enet0_tx_r_status(), .emio_enet0_rx_w_wr(), .emio_enet0_rx_w_data(), .emio_enet0_rx_w_sop(), .emio_enet0_rx_w_eop(), .emio_enet0_rx_w_status(), .emio_enet0_rx_w_err(), .emio_enet0_rx_w_overflow(1'B0), .emio_enet0_rx_w_flush(), .emio_enet0_tx_r_fixed_lat(), .fmio_gem0_fifo_tx_clk_to_pl_bufg(), .fmio_gem0_fifo_rx_clk_to_pl_bufg(),
		.emio_enet1_tx_r_data_rdy(1'B0), .emio_enet1_tx_r_rd(), .emio_enet1_tx_r_valid(1'B0), .emio_enet1_tx_r_data(8'B0), .emio_enet1_tx_r_sop(1'B1), .emio_enet1_tx_r_eop(1'B1), .emio_enet1_tx_r_err(1'B0), .emio_enet1_tx_r_underflow(1'B0), .emio_enet1_tx_r_flushed(1'B0), .emio_enet1_tx_r_control(1'B0), .emio_enet1_dma_tx_end_tog(), .emio_enet1_dma_tx_status_tog(1'B0), .emio_enet1_tx_r_status(), .emio_enet1_rx_w_wr(), .emio_enet1_rx_w_data(), .emio_enet1_rx_w_sop(), .emio_enet1_rx_w_eop(), .emio_enet1_rx_w_status(), .emio_enet1_rx_w_err(), .emio_enet1_rx_w_overflow(1'B0), .emio_enet1_rx_w_flush(), .emio_enet1_tx_r_fixed_lat(), .fmio_gem1_fifo_tx_clk_to_pl_bufg(), .fmio_gem1_fifo_rx_clk_to_pl_bufg(),
		.emio_enet2_tx_r_data_rdy(1'B0), .emio_enet2_tx_r_rd(), .emio_enet2_tx_r_valid(1'B0), .emio_enet2_tx_r_data(8'B0), .emio_enet2_tx_r_sop(1'B1), .emio_enet2_tx_r_eop(1'B1), .emio_enet2_tx_r_err(1'B0), .emio_enet2_tx_r_underflow(1'B0), .emio_enet2_tx_r_flushed(1'B0), .emio_enet2_tx_r_control(1'B0), .emio_enet2_dma_tx_end_tog(), .emio_enet2_dma_tx_status_tog(1'B0), .emio_enet2_tx_r_status(), .emio_enet2_rx_w_wr(), .emio_enet2_rx_w_data(), .emio_enet2_rx_w_sop(), .emio_enet2_rx_w_eop(), .emio_enet2_rx_w_status(), .emio_enet2_rx_w_err(), .emio_enet2_rx_w_overflow(1'B0), .emio_enet2_rx_w_flush(), .emio_enet2_tx_r_fixed_lat(), .fmio_gem2_fifo_tx_clk_to_pl_bufg(), .fmio_gem2_fifo_rx_clk_to_pl_bufg(),
		.emio_enet3_tx_r_data_rdy(1'B0), .emio_enet3_tx_r_rd(), .emio_enet3_tx_r_valid(1'B0), .emio_enet3_tx_r_data(8'B0), .emio_enet3_tx_r_sop(1'B1), .emio_enet3_tx_r_eop(1'B1), .emio_enet3_tx_r_err(1'B0), .emio_enet3_tx_r_underflow(1'B0), .emio_enet3_tx_r_flushed(1'B0), .emio_enet3_tx_r_control(1'B0), .emio_enet3_dma_tx_end_tog(), .emio_enet3_dma_tx_status_tog(1'B0), .emio_enet3_tx_r_status(), .emio_enet3_rx_w_wr(), .emio_enet3_rx_w_data(), .emio_enet3_rx_w_sop(), .emio_enet3_rx_w_eop(), .emio_enet3_rx_w_status(), .emio_enet3_rx_w_err(), .emio_enet3_rx_w_overflow(1'B0), .emio_enet3_rx_w_flush(), .emio_enet3_tx_r_fixed_lat(), .fmio_gem3_fifo_tx_clk_to_pl_bufg(), .fmio_gem3_fifo_rx_clk_to_pl_bufg(),

		.emio_enet0_tx_sof(), .emio_enet0_sync_frame_tx(), .emio_enet0_delay_req_tx(), .emio_enet0_pdelay_req_tx(), .emio_enet0_pdelay_resp_tx(), .emio_enet0_rx_sof(), .emio_enet0_sync_frame_rx(), .emio_enet0_delay_req_rx(), .emio_enet0_pdelay_req_rx(), .emio_enet0_pdelay_resp_rx(), .emio_enet0_tsu_inc_ctrl(2'B0), .emio_enet0_tsu_timer_cmp_val(),
		.emio_enet1_tx_sof(), .emio_enet1_sync_frame_tx(), .emio_enet1_delay_req_tx(), .emio_enet1_pdelay_req_tx(), .emio_enet1_pdelay_resp_tx(), .emio_enet1_rx_sof(), .emio_enet1_sync_frame_rx(), .emio_enet1_delay_req_rx(), .emio_enet1_pdelay_req_rx(), .emio_enet1_pdelay_resp_rx(), .emio_enet1_tsu_inc_ctrl(2'B0), .emio_enet1_tsu_timer_cmp_val(),
		.emio_enet2_tx_sof(), .emio_enet2_sync_frame_tx(), .emio_enet2_delay_req_tx(), .emio_enet2_pdelay_req_tx(), .emio_enet2_pdelay_resp_tx(), .emio_enet2_rx_sof(), .emio_enet2_sync_frame_rx(), .emio_enet2_delay_req_rx(), .emio_enet2_pdelay_req_rx(), .emio_enet2_pdelay_resp_rx(), .emio_enet2_tsu_inc_ctrl(2'B0), .emio_enet2_tsu_timer_cmp_val(),
		.emio_enet3_tx_sof(), .emio_enet3_sync_frame_tx(), .emio_enet3_delay_req_tx(), .emio_enet3_pdelay_req_tx(), .emio_enet3_pdelay_resp_tx(), .emio_enet3_rx_sof(), .emio_enet3_sync_frame_rx(), .emio_enet3_delay_req_rx(), .emio_enet3_pdelay_req_rx(), .emio_enet3_pdelay_resp_rx(), .emio_enet3_tsu_inc_ctrl(2'B0), .emio_enet3_tsu_timer_cmp_val(),

		.fmio_gem_tsu_clk_to_pl_bufg(),
		.fmio_gem_tsu_clk_from_pl(1'B0),
		.emio_enet_tsu_clk(1'B0),
		.emio_enet0_enet_tsu_timer_cnt(),
		.emio_enet0_ext_int_in(1'B0),
		.emio_enet1_ext_int_in(1'B0),
		.emio_enet2_ext_int_in(1'B0),
		.emio_enet3_ext_int_in(1'B0),
		.emio_enet0_dma_bus_width(),
		.emio_enet1_dma_bus_width(),
		.emio_enet2_dma_bus_width(),
		.emio_enet3_dma_bus_width(),
		.emio_gpio_i(1'B0),
		.emio_gpio_o(),
		.emio_gpio_t(),

		.emio_i2c0_scl_i(1'B0), .emio_i2c0_scl_o(), .emio_i2c0_scl_t(), .emio_i2c0_sda_i(1'B0), .emio_i2c0_sda_o(), .emio_i2c0_sda_t(),
		.emio_i2c1_scl_i(1'B0), .emio_i2c1_scl_o(), .emio_i2c1_scl_t(), .emio_i2c1_sda_i(1'B0), .emio_i2c1_sda_o(), .emio_i2c1_sda_t(),

		.emio_uart0_txd(), .emio_uart0_rxd(1'B0), .emio_uart0_ctsn(1'B0), .emio_uart0_rtsn(), .emio_uart0_dsrn(1'B0), .emio_uart0_dcdn(1'B0), .emio_uart0_rin(1'B0), .emio_uart0_dtrn(),
		.emio_uart1_txd(), .emio_uart1_rxd(1'B0), .emio_uart1_ctsn(1'B0), .emio_uart1_rtsn(), .emio_uart1_dsrn(1'B0), .emio_uart1_dcdn(1'B0), .emio_uart1_rin(1'B0), .emio_uart1_dtrn(),

		.emio_sdio0_clkout(), .emio_sdio0_fb_clk_in(1'B0), .emio_sdio0_cmdout(), .emio_sdio0_cmdin(1'B0), .emio_sdio0_cmdena(), .emio_sdio0_datain(5'B0), .emio_sdio0_dataout(), .emio_sdio0_dataena(), .emio_sdio0_cd_n(1'B0), .emio_sdio0_wp(1'B1), .emio_sdio0_ledcontrol(), .emio_sdio0_buspower(), .emio_sdio0_bus_volt(),
		.emio_sdio1_clkout(), .emio_sdio1_fb_clk_in(1'B0), .emio_sdio1_cmdout(), .emio_sdio1_cmdin(1'B0), .emio_sdio1_cmdena(), .emio_sdio1_datain(8'B0), .emio_sdio1_dataout(), .emio_sdio1_dataena(), .emio_sdio1_cd_n(1'B0), .emio_sdio1_wp(1'B1), .emio_sdio1_ledcontrol(), .emio_sdio1_buspower(), .emio_sdio1_bus_volt(),


		.emio_spi0_sclk_i(1'B0), .emio_spi0_sclk_o(), .emio_spi0_sclk_t(), .emio_spi0_m_i(1'B0), .emio_spi0_m_o(), .emio_spi0_mo_t(), .emio_spi0_s_i(1'B0), .emio_spi0_s_o(), .emio_spi0_so_t(), .emio_spi0_ss_i_n(1'B1), .emio_spi0_ss_o_n(), .emio_spi0_ss1_o_n(), .emio_spi0_ss2_o_n(), .emio_spi0_ss_n_t(),
		.emio_spi1_sclk_i(1'B0), .emio_spi1_sclk_o(), .emio_spi1_sclk_t(), .emio_spi1_m_i(1'B0), .emio_spi1_m_o(), .emio_spi1_mo_t(), .emio_spi1_s_i(1'B0), .emio_spi1_s_o(), .emio_spi1_so_t(), .emio_spi1_ss_i_n(1'B1), .emio_spi1_ss_o_n(), .emio_spi1_ss1_o_n(), .emio_spi1_ss2_o_n(), .emio_spi1_ss_n_t(),

		.pl_ps_trace_clk(1'B0),
		.ps_pl_tracectl(),
		.ps_pl_tracedata(),
		.trace_clk_out(),

		.emio_ttc0_wave_o(), .emio_ttc0_clk_i(3'B0),
		.emio_ttc1_wave_o(), .emio_ttc1_clk_i(3'B0),
		.emio_ttc2_wave_o(), .emio_ttc2_clk_i(3'B0),
		.emio_ttc3_wave_o(), .emio_ttc3_clk_i(3'B0),

		.emio_wdt0_clk_i(1'B0), .emio_wdt0_rst_o(),
		.emio_wdt1_clk_i(1'B0), .emio_wdt1_rst_o(),

		.emio_hub_port_overcrnt_usb3_0(1'B0),
		.emio_hub_port_overcrnt_usb3_1(1'B0),
		.emio_hub_port_overcrnt_usb2_0(1'B0),
		.emio_hub_port_overcrnt_usb2_1(1'B0),
		.emio_u2dsport_vbus_ctrl_usb3_0(),
		.emio_u2dsport_vbus_ctrl_usb3_1(),
		.emio_u3dsport_vbus_ctrl_usb3_0(),
		.emio_u3dsport_vbus_ctrl_usb3_1(),

		.adma_fci_clk(8'B0),
		.pl2adma_cvld(8'B0),
		.pl2adma_tack(8'B0),
		.adma2pl_cack(),
		.adma2pl_tvld(),
		.perif_gdma_clk(8'B0),
		.perif_gdma_cvld(8'B0),
		.perif_gdma_tack(8'B0),
		.gdma_perif_cack(),
		.gdma_perif_tvld(),
		.pl_clock_stop(4'B0),
		.pll_aux_refclk_lpd(2'B0),
		.pll_aux_refclk_fpd(3'B0),
		.dp_audio_ref_clk(),
		.dp_video_ref_clk(),
		.dp_s_axis_audio_tdata(32'B0),
		.dp_s_axis_audio_tid(1'B0),
		.dp_s_axis_audio_tvalid(1'B0),
		.dp_s_axis_audio_tready(),
		.dp_m_axis_mixed_audio_tdata(),
		.dp_m_axis_mixed_audio_tid(),
		.dp_m_axis_mixed_audio_tvalid(),
		.dp_m_axis_mixed_audio_tready(1'B0),
		.dp_s_axis_audio_clk(1'B0),
		.dp_live_video_in_vsync(1'B0),
		.dp_live_video_in_hsync(1'B0),
		.dp_live_video_in_de(1'B0),
		.dp_live_video_in_pixel1(36'B0),
		.dp_video_in_clk(1'B0),
		.dp_video_out_hsync(),
		.dp_video_out_vsync(),
		.dp_video_out_pixel1(),
		.dp_aux_data_in(1'B0),
		.dp_aux_data_out(),
		.dp_aux_data_oe_n(),
		.dp_live_gfx_alpha_in(8'B0),
		.dp_live_gfx_pixel1_in(36'B0),
		.dp_hot_plug_detect(1'B0),
		.dp_external_custom_event1(1'B0),
		.dp_external_custom_event2(1'B0),
		.dp_external_vsync_event(1'B0),
		.dp_live_video_de_out(),
		.pl_ps_eventi(1'B0),
		.ps_pl_evento(),
		.ps_pl_standbywfe(),
		.ps_pl_standbywfi(),
		.pl_ps_apugic_irq(4'B0),
		.pl_ps_apugic_fiq(4'B0),
		.rpu_eventi0(1'B0),
		.rpu_eventi1(1'B0),
		.rpu_evento0(),
		.rpu_evento1(),
		.nfiq0_lpd_rpu(1'B1),
		.nfiq1_lpd_rpu(1'B1),
		.nirq0_lpd_rpu(1'B1),
		.nirq1_lpd_rpu(1'B1),
		.irq_ipi_pl_0(),
		.irq_ipi_pl_1(),
		.irq_ipi_pl_2(),
		.irq_ipi_pl_3(),
		.stm_event(60'B0),
		.pl_ps_trigger_0(1'B0),
		.pl_ps_trigger_1(1'B0),
		.pl_ps_trigger_2(1'B0),
		.pl_ps_trigger_3(1'B0),
		.ps_pl_trigack_0(),
		.ps_pl_trigack_1(),
		.ps_pl_trigack_2(),
		.ps_pl_trigack_3(),
		.ps_pl_trigger_0(),
		.ps_pl_trigger_1(),
		.ps_pl_trigger_2(),
		.ps_pl_trigger_3(),
		.pl_ps_trigack_0(1'B0),
		.pl_ps_trigack_1(1'B0),
		.pl_ps_trigack_2(1'B0),
		.pl_ps_trigack_3(1'B0),
		.ftm_gpo(),
		.ftm_gpi(32'B0),
		.pl_ps_irq0(1'B0),
		.pl_ps_irq1(1'B0),
		.pl_resetn0(pl_resetn0),
		.pl_resetn1(),
		.pl_resetn2(),
		.pl_resetn3(),
		.osc_rtc_clk(),
		.pl_pmu_gpi(32'B0),
		.pmu_pl_gpo(),
		.aib_pmu_afifm_fpd_ack(1'B0),
		.aib_pmu_afifm_lpd_ack(1'B0),
		.pmu_aib_afifm_fpd_req(),
		.pmu_aib_afifm_lpd_req(),
		.pmu_error_from_pl(4'B0),
		.pmu_error_to_pl(),
		.pl_acpinact(1'B0),
		.pl_clk0(pl_clk0),
		.pl_clk1(),
		.pl_clk2(),
		.pl_clk3(),
		.ps_pl_irq_can0(),
		.ps_pl_irq_can1(),
		.ps_pl_irq_enet0(),
		.ps_pl_irq_enet1(),
		.ps_pl_irq_enet2(),
		.ps_pl_irq_enet3(),
		.ps_pl_irq_enet0_wake(),
		.ps_pl_irq_enet1_wake(),
		.ps_pl_irq_enet2_wake(),
		.ps_pl_irq_enet3_wake(),
		.ps_pl_irq_gpio(),
		.ps_pl_irq_i2c0(),
		.ps_pl_irq_i2c1(),
		.ps_pl_irq_uart0(),
		.ps_pl_irq_uart1(),
		.ps_pl_irq_sdio0(),
		.ps_pl_irq_sdio1(),
		.ps_pl_irq_sdio0_wake(),
		.ps_pl_irq_sdio1_wake(),
		.ps_pl_irq_spi0(),
		.ps_pl_irq_spi1(),
		.ps_pl_irq_qspi(),
		.ps_pl_irq_ttc0_0(),
		.ps_pl_irq_ttc0_1(),
		.ps_pl_irq_ttc0_2(),
		.ps_pl_irq_ttc1_0(),
		.ps_pl_irq_ttc1_1(),
		.ps_pl_irq_ttc1_2(),
		.ps_pl_irq_ttc2_0(),
		.ps_pl_irq_ttc2_1(),
		.ps_pl_irq_ttc2_2(),
		.ps_pl_irq_ttc3_0(),
		.ps_pl_irq_ttc3_1(),
		.ps_pl_irq_ttc3_2(),
		.ps_pl_irq_csu_pmu_wdt(),
		.ps_pl_irq_lp_wdt(),
		.ps_pl_irq_usb3_0_endpoint(),
		.ps_pl_irq_usb3_0_otg(),
		.ps_pl_irq_usb3_1_endpoint(),
		.ps_pl_irq_usb3_1_otg(),
		.ps_pl_irq_adma_chan(),
		.ps_pl_irq_usb3_0_pmu_wakeup(),
		.ps_pl_irq_gdma_chan(),
		.ps_pl_irq_csu(),
		.ps_pl_irq_csu_dma(),
		.ps_pl_irq_efuse(),
		.ps_pl_irq_xmpu_lpd(),
		.ps_pl_irq_ddr_ss(),
		.ps_pl_irq_nand(),
		.ps_pl_irq_fp_wdt(),
		.ps_pl_irq_pcie_msi(),
		.ps_pl_irq_pcie_legacy(),
		.ps_pl_irq_pcie_dma(),
		.ps_pl_irq_pcie_msc(),
		.ps_pl_irq_dport(),
		.ps_pl_irq_fpd_apb_int(),
		.ps_pl_irq_fpd_atb_error(),
		.ps_pl_irq_dpdma(),
		.ps_pl_irq_apm_fpd(),
		.ps_pl_irq_gpu(),
		.ps_pl_irq_sata(),
		.ps_pl_irq_xmpu_fpd(),
		.ps_pl_irq_apu_cpumnt(),
		.ps_pl_irq_apu_cti(),
		.ps_pl_irq_apu_pmu(),
		.ps_pl_irq_apu_comm(),
		.ps_pl_irq_apu_l2err(),
		.ps_pl_irq_apu_exterr(),
		.ps_pl_irq_apu_regs(),
		.ps_pl_irq_intf_ppd_cci(),
		.ps_pl_irq_intf_fpd_smmu(),
		.ps_pl_irq_atb_err_lpd(),
		.ps_pl_irq_aib_axi(),
		.ps_pl_irq_ams(),
		.ps_pl_irq_lpd_apm(),
		.ps_pl_irq_rtc_alaram(),
		.ps_pl_irq_rtc_seconds(),
		.ps_pl_irq_clkmon(),
		.ps_pl_irq_ipi_channel0(),
		.ps_pl_irq_ipi_channel1(),
		.ps_pl_irq_ipi_channel2(),
		.ps_pl_irq_ipi_channel7(),
		.ps_pl_irq_ipi_channel8(),
		.ps_pl_irq_ipi_channel9(),
		.ps_pl_irq_ipi_channel10(),
		.ps_pl_irq_rpu_pm(),
		.ps_pl_irq_ocm_error(),
		.ps_pl_irq_lpd_apb_intr(),
		.ps_pl_irq_r5_core0_ecc_error(),
		.ps_pl_irq_r5_core1_ecc_error(),
		.test_adc_clk(4'B0),
		.test_adc_in(32'B0),
		.test_adc2_in(32'B0),
		.test_db(),
		.test_adc_out(),
		.test_ams_osc(),
		.test_mon_data(),
		.test_dclk(1'B0),
		.test_den(1'B0),
		.test_dwe(1'B0),
		.test_daddr(8'B0),
		.test_di(16'B0),
		.test_drdy(),
		.test_do(),
		.test_convst(1'B0),
		.pstp_pl_clk(4'B0),
		.pstp_pl_in(32'B0),
		.pstp_pl_out(),
		.pstp_pl_ts(32'B0),
		.fmio_test_gem_scanmux_1(1'B0),
		.fmio_test_gem_scanmux_2(1'B0),
		.test_char_mode_fpd_n(1'B0),
		.test_char_mode_lpd_n(1'B0),
		.fmio_test_io_char_scan_clock(1'B0),
		.fmio_test_io_char_scanenable(1'B0),
		.fmio_test_io_char_scan_in(1'B0),
		.fmio_test_io_char_scan_out(),
		.fmio_test_io_char_scan_reset_n(1'B0),
		.fmio_char_afifslpd_test_select_n(1'B0),
		.fmio_char_afifslpd_test_input(1'B0),
		.fmio_char_afifslpd_test_output(),
		.fmio_char_afifsfpd_test_select_n(1'B0),
		.fmio_char_afifsfpd_test_input(1'B0),
		.fmio_char_afifsfpd_test_output(),
		.io_char_audio_in_test_data(1'B0),
		.io_char_audio_mux_sel_n(1'B0),
		.io_char_video_in_test_data(1'B0),
		.io_char_video_mux_sel_n(1'B0),
		.io_char_video_out_test_data(),
		.io_char_audio_out_test_data(),
		.fmio_test_qspi_scanmux_1_n(1'B0),
		.fmio_test_sdio_scanmux_1(1'B0),
		.fmio_test_sdio_scanmux_2(1'B0),
		.fmio_sd0_dll_test_in_n(4'B0),
		.fmio_sd0_dll_test_out(),
		.fmio_sd1_dll_test_in_n(4'B0),
		.fmio_sd1_dll_test_out(),
		.test_pl_scan_chopper_si(1'B0),
		.test_pl_scan_chopper_so(),
		.test_pl_scan_chopper_trig(1'B0),
		.test_pl_scan_clk0(1'B0),
		.test_pl_scan_clk1(1'B0),
		.test_pl_scan_edt_clk(1'B0),
		.test_pl_scan_edt_in_apu(1'B0),
		.test_pl_scan_edt_in_cpu(1'B0),
		.test_pl_scan_edt_in_ddr(4'B0),
		.test_pl_scan_edt_in_fp(10'B0),
		.test_pl_scan_edt_in_gpu(4'B0),
		.test_pl_scan_edt_in_lp(9'B0),
		.test_pl_scan_edt_in_usb3(2'B0),
		.test_pl_scan_edt_out_apu(),
		.test_pl_scan_edt_out_cpu0(),
		.test_pl_scan_edt_out_cpu1(),
		.test_pl_scan_edt_out_cpu2(),
		.test_pl_scan_edt_out_cpu3(),
		.test_pl_scan_edt_out_ddr(),
		.test_pl_scan_edt_out_fp(),
		.test_pl_scan_edt_out_gpu(),
		.test_pl_scan_edt_out_lp(),
		.test_pl_scan_edt_out_usb3(),
		.test_pl_scan_edt_update(1'B0),
		.test_pl_scan_reset_n(1'B0),
		.test_pl_scanenable(1'B0),
		.test_pl_scan_pll_reset(1'B0),
		.test_pl_scan_spare_in0(1'B0),
		.test_pl_scan_spare_in1(1'B0),
		.test_pl_scan_spare_out0(),
		.test_pl_scan_spare_out1(),
		.test_pl_scan_wrap_clk(1'B0),
		.test_pl_scan_wrap_ishift(1'B0),
		.test_pl_scan_wrap_oshift(1'B0),
		.test_pl_scan_slcr_config_clk(1'B0),
		.test_pl_scan_slcr_config_rstn(1'B0),
		.test_pl_scan_slcr_config_si(1'B0),
		.test_pl_scan_spare_in2(1'B0),
		.test_pl_scanenable_slcr_en(1'B0),
		.test_pl_pll_lock_out(),
		.test_pl_scan_slcr_config_so(),
		.tst_rtc_calibreg_in(21'B0),
		.tst_rtc_calibreg_out(),
		.tst_rtc_calibreg_we(1'B0),
		.tst_rtc_clk(1'B0),
		.tst_rtc_osc_clk_out(),
		.tst_rtc_sec_counter_out(),
		.tst_rtc_seconds_raw_int(),
		.tst_rtc_testclock_select_n(1'B0),
		.tst_rtc_tick_counter_out(),
		.tst_rtc_timesetreg_in(32'B0),
		.tst_rtc_timesetreg_out(),
		.tst_rtc_disable_bat_op(1'B0),
		.tst_rtc_osc_cntrl_in(4'B0),
		.tst_rtc_osc_cntrl_out(),
		.tst_rtc_osc_cntrl_we(1'B0),
		.tst_rtc_sec_reload(1'B0),
		.tst_rtc_timesetreg_we(1'B0),
		.tst_rtc_testmode_n(1'B0),
		.test_usb0_funcmux_0_n(1'B0),
		.test_usb1_funcmux_0_n(1'B0),
		.test_usb0_scanmux_0_n(1'B0),
		.test_usb1_scanmux_0_n(1'B0),
		.lpd_pll_test_out(),
		.pl_lpd_pll_test_ck_sel_n(3'B0),
		.pl_lpd_pll_test_fract_clk_sel_n(1'B0),
		.pl_lpd_pll_test_fract_en_n(1'B0),
		.pl_lpd_pll_test_mux_sel(1'B0),
		.pl_lpd_pll_test_sel(4'B0),
		.fpd_pll_test_out(),
		.pl_fpd_pll_test_ck_sel_n(3'B0),
		.pl_fpd_pll_test_fract_clk_sel_n(1'B0),
		.pl_fpd_pll_test_fract_en_n(1'B0),
		.pl_fpd_pll_test_mux_sel(2'B0),
		.pl_fpd_pll_test_sel(4'B0),
		.fmio_char_gem_selection(2'B0),
		.fmio_char_gem_test_select_n(1'B0),
		.fmio_char_gem_test_input(1'B0),
		.fmio_char_gem_test_output(),
		.test_ddr2pl_dcd_skewout(),
		.test_pl2ddr_dcd_sample_pulse(1'B0),
		.test_bscan_en_n(1'B0),
		.test_bscan_tdi(1'B0),
		.test_bscan_updatedr(1'B0),
		.test_bscan_shiftdr(1'B0),
		.test_bscan_reset_tap_b(1'B0),
		.test_bscan_misr_jtag_load(1'B0),
		.test_bscan_intest(1'B0),
		.test_bscan_extest(1'B0),
		.test_bscan_clockdr(1'B0),
		.test_bscan_ac_mode(1'B0),
		.test_bscan_ac_test(1'B0),
		.test_bscan_init_memory(1'B0),
		.test_bscan_mode_c(1'B0),
		.test_bscan_tdo(),
		.i_dbg_l0_txclk(1'B0),
		.i_dbg_l0_rxclk(1'B0),
		.i_dbg_l1_txclk(1'B0),
		.i_dbg_l1_rxclk(1'B0),
		.i_dbg_l2_txclk(1'B0),
		.i_dbg_l2_rxclk(1'B0),
		.i_dbg_l3_txclk(1'B0),
		.i_dbg_l3_rxclk(1'B0),
		.i_afe_rx_symbol_clk_by_2_pl(1'B0),
		.pl_fpd_spare_0_in(1'B0),
		.pl_fpd_spare_1_in(1'B0),
		.pl_fpd_spare_2_in(1'B0),
		.pl_fpd_spare_3_in(1'B0),
		.pl_fpd_spare_4_in(1'B0),
		.fpd_pl_spare_0_out(),
		.fpd_pl_spare_1_out(),
		.fpd_pl_spare_2_out(),
		.fpd_pl_spare_3_out(),
		.fpd_pl_spare_4_out(),
		.pl_lpd_spare_0_in(1'B0),
		.pl_lpd_spare_1_in(1'B0),
		.pl_lpd_spare_2_in(1'B0),
		.pl_lpd_spare_3_in(1'B0),
		.pl_lpd_spare_4_in(1'B0),
		.lpd_pl_spare_0_out(),
		.lpd_pl_spare_1_out(),
		.lpd_pl_spare_2_out(),
		.lpd_pl_spare_3_out(),
		.lpd_pl_spare_4_out(),
		.o_dbg_l0_phystatus(),
		.o_dbg_l0_rxdata(),
		.o_dbg_l0_rxdatak(),
		.o_dbg_l0_rxvalid(),
		.o_dbg_l0_rxstatus(),
		.o_dbg_l0_rxelecidle(),
		.o_dbg_l0_rstb(),
		.o_dbg_l0_txdata(),
		.o_dbg_l0_txdatak(),
		.o_dbg_l0_rate(),
		.o_dbg_l0_powerdown(),
		.o_dbg_l0_txelecidle(),
		.o_dbg_l0_txdetrx_lpback(),
		.o_dbg_l0_rxpolarity(),
		.o_dbg_l0_tx_sgmii_ewrap(),
		.o_dbg_l0_rx_sgmii_en_cdet(),
		.o_dbg_l0_sata_corerxdata(),
		.o_dbg_l0_sata_corerxdatavalid(),
		.o_dbg_l0_sata_coreready(),
		.o_dbg_l0_sata_coreclockready(),
		.o_dbg_l0_sata_corerxsignaldet(),
		.o_dbg_l0_sata_phyctrltxdata(),
		.o_dbg_l0_sata_phyctrltxidle(),
		.o_dbg_l0_sata_phyctrltxrate(),
		.o_dbg_l0_sata_phyctrlrxrate(),
		.o_dbg_l0_sata_phyctrltxrst(),
		.o_dbg_l0_sata_phyctrlrxrst(),
		.o_dbg_l0_sata_phyctrlreset(),
		.o_dbg_l0_sata_phyctrlpartial(),
		.o_dbg_l0_sata_phyctrlslumber(),
		.o_dbg_l1_phystatus(),
		.o_dbg_l1_rxdata(),
		.o_dbg_l1_rxdatak(),
		.o_dbg_l1_rxvalid(),
		.o_dbg_l1_rxstatus(),
		.o_dbg_l1_rxelecidle(),
		.o_dbg_l1_rstb(),
		.o_dbg_l1_txdata(),
		.o_dbg_l1_txdatak(),
		.o_dbg_l1_rate(),
		.o_dbg_l1_powerdown(),
		.o_dbg_l1_txelecidle(),
		.o_dbg_l1_txdetrx_lpback(),
		.o_dbg_l1_rxpolarity(),
		.o_dbg_l1_tx_sgmii_ewrap(),
		.o_dbg_l1_rx_sgmii_en_cdet(),
		.o_dbg_l1_sata_corerxdata(),
		.o_dbg_l1_sata_corerxdatavalid(),
		.o_dbg_l1_sata_coreready(),
		.o_dbg_l1_sata_coreclockready(),
		.o_dbg_l1_sata_corerxsignaldet(),
		.o_dbg_l1_sata_phyctrltxdata(),
		.o_dbg_l1_sata_phyctrltxidle(),
		.o_dbg_l1_sata_phyctrltxrate(),
		.o_dbg_l1_sata_phyctrlrxrate(),
		.o_dbg_l1_sata_phyctrltxrst(),
		.o_dbg_l1_sata_phyctrlrxrst(),
		.o_dbg_l1_sata_phyctrlreset(),
		.o_dbg_l1_sata_phyctrlpartial(),
		.o_dbg_l1_sata_phyctrlslumber(),
		.o_dbg_l2_phystatus(),
		.o_dbg_l2_rxdata(),
		.o_dbg_l2_rxdatak(),
		.o_dbg_l2_rxvalid(),
		.o_dbg_l2_rxstatus(),
		.o_dbg_l2_rxelecidle(),
		.o_dbg_l2_rstb(),
		.o_dbg_l2_txdata(),
		.o_dbg_l2_txdatak(),
		.o_dbg_l2_rate(),
		.o_dbg_l2_powerdown(),
		.o_dbg_l2_txelecidle(),
		.o_dbg_l2_txdetrx_lpback(),
		.o_dbg_l2_rxpolarity(),
		.o_dbg_l2_tx_sgmii_ewrap(),
		.o_dbg_l2_rx_sgmii_en_cdet(),
		.o_dbg_l2_sata_corerxdata(),
		.o_dbg_l2_sata_corerxdatavalid(),
		.o_dbg_l2_sata_coreready(),
		.o_dbg_l2_sata_coreclockready(),
		.o_dbg_l2_sata_corerxsignaldet(),
		.o_dbg_l2_sata_phyctrltxdata(),
		.o_dbg_l2_sata_phyctrltxidle(),
		.o_dbg_l2_sata_phyctrltxrate(),
		.o_dbg_l2_sata_phyctrlrxrate(),
		.o_dbg_l2_sata_phyctrltxrst(),
		.o_dbg_l2_sata_phyctrlrxrst(),
		.o_dbg_l2_sata_phyctrlreset(),
		.o_dbg_l2_sata_phyctrlpartial(),
		.o_dbg_l2_sata_phyctrlslumber(),
		.o_dbg_l3_phystatus(),
		.o_dbg_l3_rxdata(),
		.o_dbg_l3_rxdatak(),
		.o_dbg_l3_rxvalid(),
		.o_dbg_l3_rxstatus(),
		.o_dbg_l3_rxelecidle(),
		.o_dbg_l3_rstb(),
		.o_dbg_l3_txdata(),
		.o_dbg_l3_txdatak(),
		.o_dbg_l3_rate(),
		.o_dbg_l3_powerdown(),
		.o_dbg_l3_txelecidle(),
		.o_dbg_l3_txdetrx_lpback(),
		.o_dbg_l3_rxpolarity(),
		.o_dbg_l3_tx_sgmii_ewrap(),
		.o_dbg_l3_rx_sgmii_en_cdet(),
		.o_dbg_l3_sata_corerxdata(),
		.o_dbg_l3_sata_corerxdatavalid(),
		.o_dbg_l3_sata_coreready(),
		.o_dbg_l3_sata_coreclockready(),
		.o_dbg_l3_sata_corerxsignaldet(),
		.o_dbg_l3_sata_phyctrltxdata(),
		.o_dbg_l3_sata_phyctrltxidle(),
		.o_dbg_l3_sata_phyctrltxrate(),
		.o_dbg_l3_sata_phyctrlrxrate(),
		.o_dbg_l3_sata_phyctrltxrst(),
		.o_dbg_l3_sata_phyctrlrxrst(),
		.o_dbg_l3_sata_phyctrlreset(),
		.o_dbg_l3_sata_phyctrlpartial(),
		.o_dbg_l3_sata_phyctrlslumber(),
		.dbg_path_fifo_bypass(),
		.i_afe_pll_pd_hs_clock_r(1'B0),
		.i_afe_mode(1'B0),
		.i_bgcal_afe_mode(1'B0),
		.o_afe_cmn_calib_comp_out(),
		.i_afe_cmn_bg_enable_low_leakage(1'B0),
		.i_afe_cmn_bg_iso_ctrl_bar(1'B0),
		.i_afe_cmn_bg_pd(1'B0),
		.i_afe_cmn_bg_pd_bg_ok(1'B0),
		.i_afe_cmn_bg_pd_ptat(1'B0),
		.i_afe_cmn_calib_en_iconst(1'B0),
		.i_afe_cmn_calib_enable_low_leakage(1'B0),
		.i_afe_cmn_calib_iso_ctrl_bar(1'B0),
		.o_afe_pll_dco_count(),
		.o_afe_pll_clk_sym_hs(),
		.o_afe_pll_fbclk_frac(),
		.o_afe_rx_pipe_lfpsbcn_rxelecidle(),
		.o_afe_rx_pipe_sigdet(),
		.o_afe_rx_symbol(),
		.o_afe_rx_symbol_clk_by_2(),
		.o_afe_rx_uphy_save_calcode(),
		.o_afe_rx_uphy_startloop_buf(),
		.o_afe_rx_uphy_rx_calib_done(),
		.i_afe_rx_rxpma_rstb(1'B0),
		.i_afe_rx_uphy_restore_calcode_data(8'B0),
		.i_afe_rx_pipe_rxeqtraining(1'B0),
		.i_afe_rx_iso_hsrx_ctrl_bar(1'B0),
		.i_afe_rx_iso_lfps_ctrl_bar(1'B0),
		.i_afe_rx_iso_sigdet_ctrl_bar(1'B0),
		.i_afe_rx_hsrx_clock_stop_req(1'B0),
		.o_afe_rx_uphy_save_calcode_data(),
		.o_afe_rx_hsrx_clock_stop_ack(),
		.o_afe_pg_avddcr(),
		.o_afe_pg_avddio(),
		.o_afe_pg_dvddcr(),
		.o_afe_pg_static_avddcr(),
		.o_afe_pg_static_avddio(),
		.i_pll_afe_mode(1'B0),
		.i_afe_pll_coarse_code(11'B0),
		.i_afe_pll_en_clock_hs_div2(1'B0),
		.i_afe_pll_fbdiv(16'B0),
		.i_afe_pll_load_fbdiv(1'B0),
		.i_afe_pll_pd(1'B0),
		.i_afe_pll_pd_pfd(1'B0),
		.i_afe_pll_rst_fdbk_div(1'B0),
		.i_afe_pll_startloop(1'B0),
		.i_afe_pll_v2i_code(6'B0),
		.i_afe_pll_v2i_prog(5'B0),
		.i_afe_pll_vco_cnt_window(1'B0),
		.i_afe_rx_mphy_gate_symbol_clk(1'B0),
		.i_afe_rx_mphy_mux_hsb_ls(1'B0),
		.i_afe_rx_pipe_rx_term_enable(1'B0),
		.i_afe_rx_uphy_biasgen_iconst_core_mirror_enable(1'B0),
		.i_afe_rx_uphy_biasgen_iconst_io_mirror_enable(1'B0),
		.i_afe_rx_uphy_biasgen_irconst_core_mirror_enable(1'B0),
		.i_afe_rx_uphy_enable_cdr(1'B0),
		.i_afe_rx_uphy_enable_low_leakage(1'B0),
		.i_afe_rx_rxpma_refclk_dig(1'B0),
		.i_afe_rx_uphy_hsrx_rstb(1'B0),
		.i_afe_rx_uphy_pdn_hs_des(1'B0),
		.i_afe_rx_uphy_pd_samp_c2c(1'B0),
		.i_afe_rx_uphy_pd_samp_c2c_eclk(1'B0),
		.i_afe_rx_uphy_pso_clk_lane(1'B0),
		.i_afe_rx_uphy_pso_eq(1'B0),
		.i_afe_rx_uphy_pso_hsrxdig(1'B0),
		.i_afe_rx_uphy_pso_iqpi(1'B0),
		.i_afe_rx_uphy_pso_lfpsbcn(1'B0),
		.i_afe_rx_uphy_pso_samp_flops(1'B0),
		.i_afe_rx_uphy_pso_sigdet(1'B0),
		.i_afe_rx_uphy_restore_calcode(1'B0),
		.i_afe_rx_uphy_run_calib(1'B0),
		.i_afe_rx_uphy_rx_lane_polarity_swap(1'B0),
		.i_afe_rx_uphy_startloop_pll(1'B0),
		.i_afe_rx_uphy_hsclk_division_factor(2'B0),
		.i_afe_rx_uphy_rx_pma_opmode(8'B0),
		.i_afe_tx_enable_hsclk_division(2'B0),
		.i_afe_tx_enable_ldo(1'B0),
		.i_afe_tx_enable_ref(1'B0),
		.i_afe_tx_enable_supply_hsclk(1'B0),
		.i_afe_tx_enable_supply_pipe(1'B0),
		.i_afe_tx_enable_supply_serializer(1'B0),
		.i_afe_tx_enable_supply_uphy(1'B0),
		.i_afe_tx_hs_ser_rstb(1'B0),
		.i_afe_tx_hs_symbol(20'B0),
		.i_afe_tx_mphy_tx_ls_data(1'B0),
		.i_afe_tx_pipe_tx_enable_idle_mode(2'B0),
		.i_afe_tx_pipe_tx_enable_lfps(2'B0),
		.i_afe_tx_pipe_tx_enable_rxdet(1'B0),
		.i_afe_TX_uphy_txpma_opmode(8'B0),
		.i_afe_TX_pmadig_digital_reset_n(1'B0),
		.i_afe_TX_serializer_rst_rel(1'B0),
		.i_afe_TX_pll_symb_clk_2(1'B0),
		.i_afe_TX_ana_if_rate(2'B0),
		.i_afe_TX_en_dig_sublp_mode(1'B0),
		.i_afe_TX_LPBK_SEL(3'B0),
		.i_afe_TX_iso_ctrl_bar(1'B0),
		.i_afe_TX_ser_iso_ctrl_bar(1'B0),
		.i_afe_TX_lfps_clk(1'B0),
		.i_afe_TX_serializer_rstb(1'B0),
		.o_afe_TX_dig_reset_rel_ack(),
		.o_afe_TX_pipe_TX_dn_rxdet(),
		.o_afe_TX_pipe_TX_dp_rxdet(),
		.i_afe_tx_pipe_tx_fast_est_common_mode(1'B0),
		.o_dbg_l0_txclk(),
		.o_dbg_l0_rxclk(),
		.o_dbg_l1_txclk(),
		.o_dbg_l1_rxclk(),
		.o_dbg_l2_txclk(),
		.o_dbg_l2_rxclk(),
		.o_dbg_l3_txclk(),
		.o_dbg_l3_rxclk(),
		.emio_i2c0_scl_t_n(),
		.emio_i2c0_sda_t_n(),
		.emio_enet0_mdio_t_n(),
		.emio_enet1_mdio_t_n(),
		.emio_enet2_mdio_t_n(),
		.emio_enet3_mdio_t_n(),
		.emio_gpio_t_n(),
		.emio_i2c1_scl_t_n(),
		.emio_i2c1_sda_t_n(),
		.emio_spi0_sclk_t_n(),
		.emio_spi0_mo_t_n(),
		.emio_spi0_so_t_n(),
		.emio_spi0_ss_n_t_n(),
		.emio_spi1_sclk_t_n(),
		.emio_spi1_mo_t_n(),
		.emio_spi1_so_t_n(),
		.emio_spi1_ss_n_t_n()
	);
endmodule

module zynq_ultra_ps_e_v3_4_0 #(
	parameter C_MAXIGP0_DATA_WIDTH = 128,
	parameter C_MAXIGP1_DATA_WIDTH = 128,
	parameter C_MAXIGP2_DATA_WIDTH = 128,

	parameter C_SAXIGP0_DATA_WIDTH = 128,
	parameter C_SAXIGP1_DATA_WIDTH = 128,
	parameter C_SAXIGP2_DATA_WIDTH = 128,
	parameter C_SAXIGP3_DATA_WIDTH = 128,
	parameter C_SAXIGP4_DATA_WIDTH = 128,
	parameter C_SAXIGP5_DATA_WIDTH = 128,
	parameter C_SAXIGP6_DATA_WIDTH = 128,
	parameter C_SD0_INTERNAL_BUS_WIDTH = 8,
	parameter C_SD1_INTERNAL_BUS_WIDTH = 8,
	parameter C_PL_CLK0_BUF = "true",
	parameter C_PL_CLK1_BUF = "true",
	parameter C_PL_CLK2_BUF = "true",
	parameter C_PL_CLK3_BUF = "true",

	parameter integer C_NUM_F2P_0_INTR_INPUTS = 0,
	parameter integer C_NUM_F2P_1_INTR_INPUTS = 0,

	parameter C_NUM_FABRIC_RESETS = 1,
	parameter C_EMIO_GPIO_WIDTH = 96,

	//parameter C_TRISTATE_INVERTED = 1,

	parameter C_USE_DIFF_RW_CLK_GP0 = 0,
	parameter C_USE_DIFF_RW_CLK_GP1 = 0,
	parameter C_USE_DIFF_RW_CLK_GP2 = 0,
	parameter C_USE_DIFF_RW_CLK_GP3 = 0,
	parameter C_USE_DIFF_RW_CLK_GP4 = 0,
	parameter C_USE_DIFF_RW_CLK_GP5 = 0,
	parameter C_USE_DIFF_RW_CLK_GP6 = 0,

	parameter C_TRACE_PIPELINE_WIDTH = 8,
	parameter C_EN_EMIO_TRACE = 0,
	parameter C_EN_FIFO_ENET0 = "0",
	parameter C_EN_FIFO_ENET1 = "0",
	parameter C_EN_FIFO_ENET2 = "0",
	parameter C_EN_FIFO_ENET3 = "0",
	parameter C_TRACE_DATA_WIDTH = 32,

	parameter C_USE_DEBUG_TEST = 0,
	parameter C_DP_USE_AUDIO = 0,
	parameter C_DP_USE_VIDEO = 0
) (
	// maxigp0
	input  wire          maxihpm0_fpd_aclk,
	output wire          dp_video_ref_clk,
	output wire          dp_audio_ref_clk,
	output wire [15:0]   maxigp0_awid,
	output wire [39:0]   maxigp0_awaddr,
	output wire [7:0]    maxigp0_awlen,
	output wire [2:0]    maxigp0_awsize,
	output wire [1:0]    maxigp0_awburst,
	output wire          maxigp0_awlock,
	output wire [3:0]    maxigp0_awcache,
	output wire [2:0]    maxigp0_awprot,
	output wire          maxigp0_awvalid,
	output wire [15:0]   maxigp0_awuser,
	input  wire          maxigp0_awready,
	output wire [C_MAXIGP0_DATA_WIDTH-1 :0]  maxigp0_wdata,
	output wire [(C_MAXIGP0_DATA_WIDTH/8)-1 :0]   maxigp0_wstrb,
	output wire          maxigp0_wlast,
	output wire          maxigp0_wvalid,
	input  wire          maxigp0_wready,
	input  wire [15:0]   maxigp0_bid,
	input  wire [1:0]    maxigp0_bresp,
	input  wire          maxigp0_bvalid,
	output wire          maxigp0_bready,
	output wire [15:0]   maxigp0_arid,
	output wire [39:0]   maxigp0_araddr,
	output wire [7:0]    maxigp0_arlen,
	output wire [2:0]    maxigp0_arsize,
	output wire [1:0]    maxigp0_arburst,
	output wire          maxigp0_arlock,
	output wire [3:0]    maxigp0_arcache,
	output wire [2:0]    maxigp0_arprot,
	output wire          maxigp0_arvalid,
	output wire [15:0]   maxigp0_aruser,
	input  wire          maxigp0_arready,
	input  wire [15:0]   maxigp0_rid,
	input  wire [C_MAXIGP0_DATA_WIDTH-1 :0]  maxigp0_rdata,
	input  wire [1:0]    maxigp0_rresp,
	input  wire          maxigp0_rlast,
	input  wire          maxigp0_rvalid,
	output wire          maxigp0_rready,
	output wire [3:0]    maxigp0_awqos,
	output wire [3:0]    maxigp0_arqos,

	// maxigp1
	input  wire          maxihpm1_fpd_aclk,
	output wire [15:0]   maxigp1_awid,
	output wire [39:0]   maxigp1_awaddr,
	output wire [7:0]    maxigp1_awlen,
	output wire [2:0]    maxigp1_awsize,
	output wire [1:0]    maxigp1_awburst,
	output wire          maxigp1_awlock,
	output wire [3:0]    maxigp1_awcache,
	output wire [2:0]    maxigp1_awprot,
	output wire          maxigp1_awvalid,
	output wire [15:0]   maxigp1_awuser,
	input  wire          maxigp1_awready,
	output wire [C_MAXIGP1_DATA_WIDTH-1 :0]  maxigp1_wdata,
	output wire [(C_MAXIGP1_DATA_WIDTH/8)-1 :0]   maxigp1_wstrb,
	output wire          maxigp1_wlast,
	output wire          maxigp1_wvalid,
	input  wire          maxigp1_wready,
	input  wire [15:0]   maxigp1_bid,
	input  wire [1:0]    maxigp1_bresp,
	input  wire          maxigp1_bvalid,
	output wire          maxigp1_bready,
	output wire [15:0]   maxigp1_arid,
	output wire [39:0]   maxigp1_araddr,
	output wire [7:0]    maxigp1_arlen,
	output wire [2:0]    maxigp1_arsize,
	output wire [1:0]    maxigp1_arburst,
	output wire          maxigp1_arlock,
	output wire [3:0]    maxigp1_arcache,
	output wire [2:0]    maxigp1_arprot,
	output wire          maxigp1_arvalid,
	output wire [15:0]   maxigp1_aruser,
	input  wire          maxigp1_arready,
	input  wire [15:0]   maxigp1_rid,
	input  wire [C_MAXIGP1_DATA_WIDTH-1 :0]  maxigp1_rdata,
	input  wire [1:0]    maxigp1_rresp,
	input  wire          maxigp1_rlast,
	input  wire          maxigp1_rvalid,
	output wire          maxigp1_rready,
	output wire [3:0]    maxigp1_awqos,
	output wire [3:0]    maxigp1_arqos,
	// maxigp2
	input  wire          maxihpm0_lpd_aclk,
	output wire [15:0]   maxigp2_awid,
	output wire [39:0]   maxigp2_awaddr,
	output wire [7:0]    maxigp2_awlen,
	output wire [2:0]    maxigp2_awsize,
	output wire [1:0]    maxigp2_awburst,
	output wire          maxigp2_awlock,
	output wire [3:0]    maxigp2_awcache,
	output wire [2:0]    maxigp2_awprot,
	output wire          maxigp2_awvalid,
	output wire [15:0]   maxigp2_awuser,
	input  wire          maxigp2_awready,
	output wire [C_MAXIGP2_DATA_WIDTH-1 :0]  maxigp2_wdata,
	output wire [(C_MAXIGP2_DATA_WIDTH/8)-1 :0]   maxigp2_wstrb,
	output wire          maxigp2_wlast,
	output wire          maxigp2_wvalid,
	input  wire          maxigp2_wready,
	input  wire [15:0]   maxigp2_bid,
	input  wire [1:0]    maxigp2_bresp,
	input  wire          maxigp2_bvalid,
	output wire          maxigp2_bready,
	output wire [15:0]   maxigp2_arid,
	output wire [39:0]   maxigp2_araddr,
	output wire [7:0]    maxigp2_arlen,
	output wire [2:0]    maxigp2_arsize,
	output wire [1:0]    maxigp2_arburst,
	output wire          maxigp2_arlock,
	output wire [3:0]    maxigp2_arcache,
	output wire [2:0]    maxigp2_arprot,
	output wire          maxigp2_arvalid,
	output wire [15:0]   maxigp2_aruser,
	input  wire          maxigp2_arready,
	input  wire [15:0]   maxigp2_rid,
	input  wire [C_MAXIGP2_DATA_WIDTH-1 :0]  maxigp2_rdata,
	input  wire [1:0]    maxigp2_rresp,
	input  wire          maxigp2_rlast,
	input  wire          maxigp2_rvalid,
	output wire          maxigp2_rready,
	output wire [3:0]    maxigp2_awqos,
	output wire [3:0]    maxigp2_arqos,
	// saxigp0
	input  wire          saxihpc0_fpd_aclk,
	input  wire          saxihpc0_fpd_rclk,
	input  wire          saxihpc0_fpd_wclk,
	input  wire          saxigp0_aruser,
	input  wire          saxigp0_awuser,
	input  wire [5:0]    saxigp0_awid,
	input  wire [48:0]   saxigp0_awaddr,
	input  wire [7:0]    saxigp0_awlen ,
	input  wire [2:0]    saxigp0_awsize,
	input  wire [1:0]    saxigp0_awburst,
	input  wire          saxigp0_awlock,
	input  wire [3:0]    saxigp0_awcache,
	input  wire [2:0]    saxigp0_awprot,
	input  wire          saxigp0_awvalid,
	output wire          saxigp0_awready,
	input  wire [C_SAXIGP0_DATA_WIDTH-1:0]  saxigp0_wdata,
	input  wire [(C_SAXIGP0_DATA_WIDTH/8) -1 :0]   saxigp0_wstrb,
	input  wire          saxigp0_wlast,
	input  wire          saxigp0_wvalid,
	output wire          saxigp0_wready,
	output wire [5:0]    saxigp0_bid,
	output wire [1:0]    saxigp0_bresp,
	output wire          saxigp0_bvalid,
	input  wire          saxigp0_bready,
	input  wire [5:0]    saxigp0_arid,
	input  wire [48:0]   saxigp0_araddr,
	input  wire [7:0]    saxigp0_arlen,
	input  wire [2:0]    saxigp0_arsize,
	input  wire [1:0]    saxigp0_arburst,
	input  wire          saxigp0_arlock,
	input  wire [3:0]    saxigp0_arcache,
	input  wire [2:0]    saxigp0_arprot,
	input  wire          saxigp0_arvalid,
	output wire          saxigp0_arready,
	output wire [5:0]    saxigp0_rid,
	output wire [C_SAXIGP0_DATA_WIDTH-1:0]  saxigp0_rdata,
	output wire [1:0]    saxigp0_rresp,
	output wire          saxigp0_rlast,
	output wire          saxigp0_rvalid,
	input  wire          saxigp0_rready,
	input  wire [3:0]    saxigp0_awqos,
	input  wire [3:0]    saxigp0_arqos,
	output wire [7:0]    saxigp0_rcount,
	output wire [7:0]    saxigp0_wcount,
	output wire [3:0]    saxigp0_racount,
	output wire [3:0]    saxigp0_wacount,
	// saxigp1
	input  wire          saxihpc1_fpd_aclk,
	input  wire          saxihpc1_fpd_rclk,
	input  wire          saxihpc1_fpd_wclk,
	input  wire          saxigp1_aruser,
	input  wire          saxigp1_awuser,
	input  wire [5:0]    saxigp1_awid,
	input  wire [48:0]   saxigp1_awaddr,
	input  wire [7:0]    saxigp1_awlen ,
	input  wire [2:0]    saxigp1_awsize,
	input  wire [1:0]    saxigp1_awburst,
	input  wire          saxigp1_awlock,
	input  wire [3:0]    saxigp1_awcache,
	input  wire [2:0]    saxigp1_awprot,
	input  wire          saxigp1_awvalid,
	output wire          saxigp1_awready,
	input  wire [C_SAXIGP1_DATA_WIDTH-1:0]  saxigp1_wdata,
	input  wire [(C_SAXIGP1_DATA_WIDTH/8) -1 :0]   saxigp1_wstrb,
	input  wire          saxigp1_wlast,
	input  wire          saxigp1_wvalid,
	output wire          saxigp1_wready,
	output wire [5:0]    saxigp1_bid,
	output wire [1:0]    saxigp1_bresp,
	output wire          saxigp1_bvalid,
	input  wire          saxigp1_bready,
	input  wire [5:0]    saxigp1_arid,
	input  wire [48:0]   saxigp1_araddr,
	input  wire [7:0]    saxigp1_arlen,
	input  wire [2:0]    saxigp1_arsize,
	input  wire [1:0]    saxigp1_arburst,
	input  wire          saxigp1_arlock,
	input  wire [3:0]    saxigp1_arcache,
	input  wire [2:0]    saxigp1_arprot,
	input  wire          saxigp1_arvalid,
	output wire          saxigp1_arready,
	output wire [5:0]    saxigp1_rid,
	output wire [C_SAXIGP1_DATA_WIDTH-1:0]  saxigp1_rdata,
	output wire [1:0]    saxigp1_rresp,
	output wire          saxigp1_rlast,
	output wire          saxigp1_rvalid,
	input  wire          saxigp1_rready,
	input  wire [3:0]    saxigp1_awqos,
	input  wire [3:0]    saxigp1_arqos,
	output wire [7:0]    saxigp1_rcount,
	output wire [7:0]    saxigp1_wcount,
	output wire [3:0]    saxigp1_racount,
	output wire [3:0]    saxigp1_wacount,
	// saxigp2
	input  wire          saxihp0_fpd_aclk,
	input  wire          saxihp0_fpd_rclk,
	input  wire          saxihp0_fpd_wclk,
	input  wire          saxigp2_aruser,
	input  wire          saxigp2_awuser,
	input  wire [5:0]    saxigp2_awid,
	input  wire [48:0]   saxigp2_awaddr,
	input  wire [7:0]    saxigp2_awlen ,
	input  wire [2:0]    saxigp2_awsize,
	input  wire [1:0]    saxigp2_awburst,
	input  wire          saxigp2_awlock,
	input  wire [3:0]    saxigp2_awcache,
	input  wire [2:0]    saxigp2_awprot,
	input  wire          saxigp2_awvalid,
	output wire          saxigp2_awready,
	input  wire [C_SAXIGP2_DATA_WIDTH-1:0]  saxigp2_wdata,
	input  wire [(C_SAXIGP2_DATA_WIDTH/8) -1 :0]   saxigp2_wstrb,
	input  wire          saxigp2_wlast,
	input  wire          saxigp2_wvalid,
	output wire          saxigp2_wready,
	output wire [5:0]    saxigp2_bid,
	output wire [1:0]    saxigp2_bresp,
	output wire          saxigp2_bvalid,
	input  wire          saxigp2_bready,
	input  wire [5:0]    saxigp2_arid,
	input  wire [48:0]   saxigp2_araddr,
	input  wire [7:0]    saxigp2_arlen,
	input  wire [2:0]    saxigp2_arsize,
	input  wire [1:0]    saxigp2_arburst,
	input  wire          saxigp2_arlock,
	input  wire [3:0]    saxigp2_arcache,
	input  wire [2:0]    saxigp2_arprot,
	input  wire          saxigp2_arvalid,
	output wire          saxigp2_arready,
	output wire [5:0]    saxigp2_rid,
	output wire [C_SAXIGP2_DATA_WIDTH-1:0]  saxigp2_rdata,
	output wire [1:0]    saxigp2_rresp,
	output wire          saxigp2_rlast,
	output wire          saxigp2_rvalid,
	input  wire          saxigp2_rready,
	input  wire [3:0]    saxigp2_awqos,
	input  wire [3:0]    saxigp2_arqos,
	output wire [7:0]    saxigp2_rcount,
	output wire [7:0]    saxigp2_wcount,
	output wire [3:0]    saxigp2_racount,
	output wire [3:0]    saxigp2_wacount,
	// saxigp3
	input  wire          saxihp1_fpd_aclk,
	input  wire          saxihp1_fpd_rclk,
	input  wire          saxihp1_fpd_wclk,
	input  wire          saxigp3_aruser,
	input  wire          saxigp3_awuser,
	input  wire [5:0]    saxigp3_awid,
	input  wire [48:0]   saxigp3_awaddr,
	input  wire [7:0]    saxigp3_awlen ,
	input  wire [2:0]    saxigp3_awsize,
	input  wire [1:0]    saxigp3_awburst,
	input  wire          saxigp3_awlock,
	input  wire [3:0]    saxigp3_awcache,
	input  wire [2:0]    saxigp3_awprot,
	input  wire          saxigp3_awvalid,
	output wire          saxigp3_awready,
	input  wire [C_SAXIGP3_DATA_WIDTH-1:0]  saxigp3_wdata,
	input  wire [(C_SAXIGP3_DATA_WIDTH/8) -1 :0]   saxigp3_wstrb,
	input  wire          saxigp3_wlast,
	input  wire          saxigp3_wvalid,
	output wire          saxigp3_wready,
	output wire [5:0]    saxigp3_bid,
	output wire [1:0]    saxigp3_bresp,
	output wire          saxigp3_bvalid,
	input  wire          saxigp3_bready,
	input  wire [5:0]    saxigp3_arid,
	input  wire [48:0]   saxigp3_araddr,
	input  wire [7:0]    saxigp3_arlen,
	input  wire [2:0]    saxigp3_arsize,
	input  wire [1:0]    saxigp3_arburst,
	input  wire          saxigp3_arlock,
	input  wire [3:0]    saxigp3_arcache,
	input  wire [2:0]    saxigp3_arprot,
	input  wire          saxigp3_arvalid,
	output wire          saxigp3_arready,
	output wire [5:0]    saxigp3_rid,
	output wire [C_SAXIGP3_DATA_WIDTH-1:0]  saxigp3_rdata,
	output wire [1:0]    saxigp3_rresp,
	output wire          saxigp3_rlast,
	output wire          saxigp3_rvalid,
	input  wire          saxigp3_rready,
	input  wire [3:0]    saxigp3_awqos,
	input  wire [3:0]    saxigp3_arqos,
	output wire [7:0]    saxigp3_rcount,
	output wire [7:0]    saxigp3_wcount,
	output wire [3:0]    saxigp3_racount,
	output wire [3:0]    saxigp3_wacount,
	// saxigp4
	input  wire          saxihp2_fpd_aclk,
	input  wire          saxihp2_fpd_rclk,
	input  wire          saxihp2_fpd_wclk,
	input  wire          saxigp4_aruser,
	input  wire          saxigp4_awuser,
	input  wire [5:0]    saxigp4_awid,
	input  wire [48:0]   saxigp4_awaddr,
	input  wire [7:0]    saxigp4_awlen ,
	input  wire [2:0]    saxigp4_awsize,
	input  wire [1:0]    saxigp4_awburst,
	input  wire          saxigp4_awlock,
	input  wire [3:0]    saxigp4_awcache,
	input  wire [2:0]    saxigp4_awprot,
	input  wire          saxigp4_awvalid,
	output wire          saxigp4_awready,
	input  wire [C_SAXIGP4_DATA_WIDTH-1:0]  saxigp4_wdata,
	input  wire [(C_SAXIGP4_DATA_WIDTH/8) -1 :0]   saxigp4_wstrb,
	input  wire          saxigp4_wlast,
	input  wire          saxigp4_wvalid,
	output wire          saxigp4_wready,
	output wire [5:0]    saxigp4_bid,
	output wire [1:0]    saxigp4_bresp,
	output wire          saxigp4_bvalid,
	input  wire          saxigp4_bready,
	input  wire [5:0]    saxigp4_arid,
	input  wire [48:0]   saxigp4_araddr,
	input  wire [7:0]    saxigp4_arlen,
	input  wire [2:0]    saxigp4_arsize,
	input  wire [1:0]    saxigp4_arburst,
	input  wire          saxigp4_arlock,
	input  wire [3:0]    saxigp4_arcache,
	input  wire [2:0]    saxigp4_arprot,
	input  wire          saxigp4_arvalid,
	output wire          saxigp4_arready,
	output wire [5:0]    saxigp4_rid,
	output wire [C_SAXIGP4_DATA_WIDTH-1:0]  saxigp4_rdata,
	output wire [1:0]    saxigp4_rresp,
	output wire          saxigp4_rlast,
	output wire          saxigp4_rvalid,
	input  wire          saxigp4_rready,
	input  wire [3:0]    saxigp4_awqos,
	input  wire [3:0]    saxigp4_arqos,
	output wire [7:0]    saxigp4_rcount,
	output wire [7:0]    saxigp4_wcount,
	output wire [3:0]    saxigp4_racount,
	output wire [3:0]    saxigp4_wacount,
	// saxigp5
	input  wire          saxihp3_fpd_aclk,
	input  wire          saxihp3_fpd_rclk,
	input  wire          saxihp3_fpd_wclk,
	input  wire          saxigp5_aruser,
	input  wire          saxigp5_awuser,
	input  wire [5:0]    saxigp5_awid,
	input  wire [48:0]   saxigp5_awaddr,
	input  wire [7:0]    saxigp5_awlen ,
	input  wire [2:0]    saxigp5_awsize,
	input  wire [1:0]    saxigp5_awburst,
	input  wire          saxigp5_awlock,
	input  wire [3:0]    saxigp5_awcache,
	input  wire [2:0]    saxigp5_awprot,
	input  wire          saxigp5_awvalid,
	output wire          saxigp5_awready,
	input  wire [C_SAXIGP5_DATA_WIDTH-1:0]  saxigp5_wdata,
	input  wire [(C_SAXIGP5_DATA_WIDTH/8) -1 :0]   saxigp5_wstrb,
	input  wire          saxigp5_wlast,
	input  wire          saxigp5_wvalid,
	output wire          saxigp5_wready,
	output wire [5:0]    saxigp5_bid,
	output wire [1:0]    saxigp5_bresp,
	output wire          saxigp5_bvalid,
	input  wire          saxigp5_bready,
	input  wire [5:0]    saxigp5_arid,
	input  wire [48:0]   saxigp5_araddr,
	input  wire [7:0]    saxigp5_arlen,
	input  wire [2:0]    saxigp5_arsize,
	input  wire [1:0]    saxigp5_arburst,
	input  wire          saxigp5_arlock,
	input  wire [3:0]    saxigp5_arcache,
	input  wire [2:0]    saxigp5_arprot,
	input  wire          saxigp5_arvalid,
	output wire          saxigp5_arready,
	output wire [5:0]    saxigp5_rid,
	output wire [C_SAXIGP5_DATA_WIDTH-1:0]  saxigp5_rdata,
	output wire [1:0]    saxigp5_rresp,
	output wire          saxigp5_rlast,
	output wire          saxigp5_rvalid,
	input  wire          saxigp5_rready,
	input  wire [3:0]    saxigp5_awqos,
	input  wire [3:0]    saxigp5_arqos,
	output wire [7:0]    saxigp5_rcount,
	output wire [7:0]    saxigp5_wcount,
	output wire [3:0]    saxigp5_racount,
	output wire [3:0]    saxigp5_wacount,
	// saxigp6
	input  wire          saxi_lpd_aclk,
	input  wire          saxi_lpd_rclk,
	input  wire          saxi_lpd_wclk,
	input  wire          saxigp6_aruser,
	input  wire          saxigp6_awuser,
	input  wire [5:0]    saxigp6_awid,
	input  wire [48:0]   saxigp6_awaddr,
	input  wire [7:0]    saxigp6_awlen ,
	input  wire [2:0]    saxigp6_awsize,
	input  wire [1:0]    saxigp6_awburst,
	input  wire          saxigp6_awlock,
	input  wire [3:0]    saxigp6_awcache,
	input  wire [2:0]    saxigp6_awprot,
	input  wire          saxigp6_awvalid,
	output wire          saxigp6_awready,
	input  wire [C_SAXIGP6_DATA_WIDTH-1:0]  saxigp6_wdata,
	input  wire [(C_SAXIGP6_DATA_WIDTH/8) -1 :0]   saxigp6_wstrb,
	input  wire          saxigp6_wlast,
	input  wire          saxigp6_wvalid,
	output wire          saxigp6_wready,
	output wire [5:0]    saxigp6_bid,
	output wire [1:0]    saxigp6_bresp,
	output wire          saxigp6_bvalid,
	input  wire          saxigp6_bready,
	input  wire [5:0]    saxigp6_arid,
	input  wire [48:0]   saxigp6_araddr,
	input  wire [7:0]    saxigp6_arlen,
	input  wire [2:0]    saxigp6_arsize,
	input  wire [1:0]    saxigp6_arburst,
	input  wire          saxigp6_arlock,
	input  wire [3:0]    saxigp6_arcache,
	input  wire [2:0]    saxigp6_arprot,
	input  wire          saxigp6_arvalid,
	output wire          saxigp6_arready,
	output wire [5:0]    saxigp6_rid,
	output wire [C_SAXIGP6_DATA_WIDTH-1:0]  saxigp6_rdata,
	output wire [1:0]    saxigp6_rresp,
	output wire          saxigp6_rlast,
	output wire          saxigp6_rvalid,
	input  wire          saxigp6_rready,
	input  wire [3:0]    saxigp6_awqos,
	input  wire [3:0]    saxigp6_arqos,
	output wire [7:0]    saxigp6_rcount,
	output wire [7:0]    saxigp6_wcount,
	output wire [3:0]    saxigp6_racount,
	output wire [3:0]    saxigp6_wacount,
	// saxiacp
	input  wire          saxiacp_fpd_aclk,
	input  wire [39:0]   saxiacp_awaddr,
	input  wire [4:0]    saxiacp_awid,
	input  wire [7:0]    saxiacp_awlen,
	input  wire [2:0]    saxiacp_awsize,
	input  wire [1:0]    saxiacp_awburst,
	input  wire          saxiacp_awlock,
	input  wire [3:0]    saxiacp_awcache,
	input  wire [2:0]    saxiacp_awprot,
	input  wire          saxiacp_awvalid,
	output wire          saxiacp_awready,
	input  wire [1:0]    saxiacp_awuser,
	input  wire [3:0]    saxiacp_awqos,
	input  wire          saxiacp_wlast,
	input  wire [127:0]  saxiacp_wdata,
	input  wire [15:0]   saxiacp_wstrb,
	input  wire          saxiacp_wvalid,
	output wire          saxiacp_wready,
	output wire [1:0]    saxiacp_bresp,
	output wire [4:0]    saxiacp_bid,
	output wire          saxiacp_bvalid,
	input  wire          saxiacp_bready,
	input  wire [39:0]   saxiacp_araddr,
	input  wire [4:0]    saxiacp_arid,
	input  wire [7:0]    saxiacp_arlen,
	input  wire [2:0]    saxiacp_arsize,
	input  wire [1:0]    saxiacp_arburst,
	input  wire          saxiacp_arlock,
	input  wire [3:0]    saxiacp_arcache,
	input  wire [2:0]    saxiacp_arprot,
	input  wire          saxiacp_arvalid,
	output wire          saxiacp_arready,
	input  wire [1:0]    saxiacp_aruser,
	input  wire [3:0]    saxiacp_arqos,
	output wire [4:0]    saxiacp_rid,
	output wire          saxiacp_rlast,
	output wire [127:0]  saxiacp_rdata,
	output wire [1:0]    saxiacp_rresp,
	output wire          saxiacp_rvalid,
	input  wire          saxiacp_rready,
	// sacefpd
	input  wire          sacefpd_aclk,
	input  wire          sacefpd_awvalid,
	output wire          sacefpd_awready,
	input  wire [5:0]    sacefpd_awid,
	input  wire [43:0]   sacefpd_awaddr,
	input  wire [3:0]    sacefpd_awregion,
	input  wire [7:0]    sacefpd_awlen,
	input  wire [2:0]    sacefpd_awsize,
	input  wire [1:0]    sacefpd_awburst,
	input  wire          sacefpd_awlock,
	input  wire [3:0]    sacefpd_awcache,
	input  wire [2:0]    sacefpd_awprot,
	input  wire [1:0]    sacefpd_awdomain,
	input  wire [2:0]    sacefpd_awsnoop,
	input  wire [1:0]    sacefpd_awbar,
	input  wire [3:0]    sacefpd_awqos,
	input  wire          sacefpd_wvalid,
	output wire          sacefpd_wready,
	input  wire [127:0]  sacefpd_wdata,
	input  wire [15:0]   sacefpd_wstrb,
	input  wire          sacefpd_wlast,
	input  wire          sacefpd_wuser,
	output wire          sacefpd_bvalid,
	input  wire          sacefpd_bready,
	output wire [5:0]    sacefpd_bid,
	output wire [1:0]    sacefpd_bresp,
	output wire          sacefpd_buser,
	input  wire          sacefpd_arvalid,
	output wire          sacefpd_arready,
	input  wire [5:0]    sacefpd_arid,
	input  wire [43:0]   sacefpd_araddr,
	input  wire [3:0]    sacefpd_arregion,
	input  wire [7:0]    sacefpd_arlen,
	input  wire [2:0]    sacefpd_arsize,
	input  wire [1:0]    sacefpd_arburst,
	input  wire          sacefpd_arlock,
	input  wire [3:0]    sacefpd_arcache,
	input  wire [2:0]    sacefpd_arprot,
	input  wire [1:0]    sacefpd_ardomain,
	input  wire [3:0]    sacefpd_arsnoop,
	input  wire [1:0]    sacefpd_arbar,
	input  wire [3:0]    sacefpd_arqos,
	output wire          sacefpd_rvalid,
	input  wire          sacefpd_rready,
	output wire [5:0]    sacefpd_rid,
	output wire [127:0]  sacefpd_rdata,
	output wire [3:0]    sacefpd_rresp,
	output wire          sacefpd_rlast,
	output wire          sacefpd_ruser,
	output wire          sacefpd_acvalid,
	input  wire          sacefpd_acready,
	output wire [43:0]   sacefpd_acaddr,
	output wire [3:0]    sacefpd_acsnoop,
	output wire [2:0]    sacefpd_acprot,
	input  wire          sacefpd_crvalid,
	output wire          sacefpd_crready,
	input  wire [4:0]    sacefpd_crresp,
	input  wire          sacefpd_cdvalid,
	output wire          sacefpd_cdready,
	input  wire [127:0]  sacefpd_cddata,
	input  wire          sacefpd_cdlast,
	input  wire          sacefpd_wack,
	input  wire          sacefpd_rack,


	///////////////////////////////////////////////
	// not using anything below this line yet... //
	///////////////////////////////////////////////

	// can0
	output wire emio_can0_phy_tx,
	input  wire emio_can0_phy_rx,
	// can1
	output wire emio_can1_phy_tx,
	input  wire emio_can1_phy_rx,
	// enet0
	input  wire emio_enet0_gmii_rx_clk,
	output wire [2:0] emio_enet0_speed_mode,
	input  wire emio_enet0_gmii_crs,
	input  wire emio_enet0_gmii_col,
	input  wire [7:0] emio_enet0_gmii_rxd,
	input  wire emio_enet0_gmii_rx_er,
	input  wire emio_enet0_gmii_rx_dv,
	input  wire emio_enet0_gmii_tx_clk,
	output wire [7:0] emio_enet0_gmii_txd,
	output wire emio_enet0_gmii_tx_en,
	output wire emio_enet0_gmii_tx_er,
	output wire emio_enet0_mdio_mdc,
	input  wire emio_enet0_mdio_i,
	output wire emio_enet0_mdio_o,
	output wire emio_enet0_mdio_t,
	output wire emio_enet0_mdio_t_n,
	// enet1
	input  wire emio_enet1_gmii_rx_clk,
	output wire [2:0] emio_enet1_speed_mode,
	input  wire emio_enet1_gmii_crs,
	input  wire emio_enet1_gmii_col,
	input  wire [7:0] emio_enet1_gmii_rxd,
	input  wire emio_enet1_gmii_rx_er,
	input  wire emio_enet1_gmii_rx_dv,
	input  wire emio_enet1_gmii_tx_clk,
	output wire [7:0] emio_enet1_gmii_txd,
	output wire emio_enet1_gmii_tx_en,
	output wire emio_enet1_gmii_tx_er,
	output wire emio_enet1_mdio_mdc,
	input  wire emio_enet1_mdio_i,
	output wire emio_enet1_mdio_o,
	output wire emio_enet1_mdio_t,
	output wire emio_enet1_mdio_t_n,
	// enet2
	input  wire emio_enet2_gmii_rx_clk,
	output wire [2:0] emio_enet2_speed_mode,
	input  wire emio_enet2_gmii_crs,
	input  wire emio_enet2_gmii_col,
	input  wire [7:0] emio_enet2_gmii_rxd,
	input  wire emio_enet2_gmii_rx_er,
	input  wire emio_enet2_gmii_rx_dv,
	input  wire emio_enet2_gmii_tx_clk,
	output wire [7:0] emio_enet2_gmii_txd,
	output wire emio_enet2_gmii_tx_en,
	output wire emio_enet2_gmii_tx_er,
	output wire emio_enet2_mdio_mdc,
	input  wire emio_enet2_mdio_i,
	output wire emio_enet2_mdio_o,
	output wire emio_enet2_mdio_t,
	output wire emio_enet2_mdio_t_n,
	// enet3
	input  wire emio_enet3_gmii_rx_clk,
	output wire [2:0] emio_enet3_speed_mode,
	input  wire emio_enet3_gmii_crs,
	input  wire emio_enet3_gmii_col,
	input  wire [7:0] emio_enet3_gmii_rxd,
	input  wire emio_enet3_gmii_rx_er,
	input  wire emio_enet3_gmii_rx_dv,
	input  wire emio_enet3_gmii_tx_clk,
	output wire [7:0] emio_enet3_gmii_txd,
	output wire emio_enet3_gmii_tx_en,
	output wire emio_enet3_gmii_tx_er,
	output wire emio_enet3_mdio_mdc,
	input  wire emio_enet3_mdio_i,
	output wire emio_enet3_mdio_o,
	output wire emio_enet3_mdio_t,
	output wire emio_enet3_mdio_t_n,
	// fifoif0
	input  wire emio_enet0_tx_r_data_rdy,
	output wire emio_enet0_tx_r_rd,
	input  wire emio_enet0_tx_r_valid,
	input  wire [7:0] emio_enet0_tx_r_data,
	input  wire emio_enet0_tx_r_sop,
	input  wire emio_enet0_tx_r_eop,
	input  wire emio_enet0_tx_r_err,
	input  wire emio_enet0_tx_r_underflow,
	input  wire emio_enet0_tx_r_flushed,
	input  wire emio_enet0_tx_r_control,
	output wire emio_enet0_dma_tx_end_tog,
	input  wire emio_enet0_dma_tx_status_tog,
	output wire [3:0] emio_enet0_tx_r_status,
	output wire emio_enet0_rx_w_wr,
	output wire [7:0] emio_enet0_rx_w_data,
	output wire emio_enet0_rx_w_sop,
	output wire emio_enet0_rx_w_eop,
	output wire [44:0] emio_enet0_rx_w_status,
	output wire emio_enet0_rx_w_err,
	input  wire emio_enet0_rx_w_overflow,
	input  wire emio_enet0_signal_detect,
	output wire emio_enet0_rx_w_flush,
	output wire emio_enet0_tx_r_fixed_lat,
	// fifoif1
	input  wire emio_enet1_tx_r_data_rdy,
	output wire emio_enet1_tx_r_rd,
	input  wire emio_enet1_tx_r_valid,
	input  wire [7:0] emio_enet1_tx_r_data,
	input  wire emio_enet1_tx_r_sop,
	input  wire emio_enet1_tx_r_eop,
	input  wire emio_enet1_tx_r_err,
	input  wire emio_enet1_tx_r_underflow,
	input  wire emio_enet1_tx_r_flushed,
	input  wire emio_enet1_tx_r_control,
	output wire emio_enet1_dma_tx_end_tog,
	input  wire emio_enet1_dma_tx_status_tog,
	output wire [3:0] emio_enet1_tx_r_status,
	output wire emio_enet1_rx_w_wr,
	output wire [7:0] emio_enet1_rx_w_data,
	output wire emio_enet1_rx_w_sop,
	output wire emio_enet1_rx_w_eop,
	output wire [44:0] emio_enet1_rx_w_status,
	output wire emio_enet1_rx_w_err,
	input  wire emio_enet1_rx_w_overflow,
	input  wire emio_enet1_signal_detect,
	output wire emio_enet1_rx_w_flush,
	output wire emio_enet1_tx_r_fixed_lat,
	// fifoif2
	input  wire emio_enet2_tx_r_data_rdy,
	output wire emio_enet2_tx_r_rd,
	input  wire emio_enet2_tx_r_valid,
	input  wire [7:0] emio_enet2_tx_r_data,
	input  wire emio_enet2_tx_r_sop,
	input  wire emio_enet2_tx_r_eop,
	input  wire emio_enet2_tx_r_err,
	input  wire emio_enet2_tx_r_underflow,
	input  wire emio_enet2_tx_r_flushed,
	input  wire emio_enet2_tx_r_control,
	output wire emio_enet2_dma_tx_end_tog,
	input  wire emio_enet2_dma_tx_status_tog,
	output wire [3:0] emio_enet2_tx_r_status,
	output wire emio_enet2_rx_w_wr,
	output wire [7:0] emio_enet2_rx_w_data,
	output wire emio_enet2_rx_w_sop,
	output wire emio_enet2_rx_w_eop,
	output wire [44:0] emio_enet2_rx_w_status,
	output wire emio_enet2_rx_w_err,
	input  wire emio_enet2_rx_w_overflow,
	input  wire emio_enet2_signal_detect,
	output wire emio_enet2_rx_w_flush,
	output wire emio_enet2_tx_r_fixed_lat,
	// fifoif3
	input  wire emio_enet3_tx_r_data_rdy,
	output wire emio_enet3_tx_r_rd,
	input  wire emio_enet3_tx_r_valid,
	input  wire [7:0] emio_enet3_tx_r_data,
	input  wire emio_enet3_tx_r_sop,
	input  wire emio_enet3_tx_r_eop,
	input  wire emio_enet3_tx_r_err,
	input  wire emio_enet3_tx_r_underflow,
	input  wire emio_enet3_tx_r_flushed,
	input  wire emio_enet3_tx_r_control,
	output wire emio_enet3_dma_tx_end_tog,
	input  wire emio_enet3_dma_tx_status_tog,
	output wire [3:0] emio_enet3_tx_r_status,
	output wire emio_enet3_rx_w_wr,
	output wire [7:0] emio_enet3_rx_w_data,
	output wire emio_enet3_rx_w_sop,
	output wire emio_enet3_rx_w_eop,
	output wire [44:0] emio_enet3_rx_w_status,
	output wire emio_enet3_rx_w_err,
	input  wire emio_enet3_rx_w_overflow,
	input  wire emio_enet3_signal_detect,
	output wire emio_enet3_rx_w_flush,
	output wire emio_enet3_tx_r_fixed_lat,
	// gem0_fmio
	//input wire  fmio_gem0_fifo_tx_clk_from_pl,
	//input wire  fmio_gem0_fifo_rx_clk_from_pl,
	output wire fmio_gem0_fifo_tx_clk_to_pl_bufg,
	output wire fmio_gem0_fifo_rx_clk_to_pl_bufg,
	// gem1_fmio
	//input wire  fmio_gem1_fifo_tx_clk_from_pl,
	//input wire  fmio_gem1_fifo_rx_clk_from_pl,
	output wire fmio_gem1_fifo_tx_clk_to_pl_bufg,
	output wire fmio_gem1_fifo_rx_clk_to_pl_bufg,
	// gem2_fmio
	//input wire  fmio_gem2_fifo_tx_clk_from_pl,
	//input wire  fmio_gem2_fifo_rx_clk_from_pl,
	output wire fmio_gem2_fifo_tx_clk_to_pl_bufg,
	output wire fmio_gem2_fifo_rx_clk_to_pl_bufg,
	// gem3_fmio
	//input wire  fmio_gem3_fifo_tx_clk_from_pl,
	//input wire  fmio_gem3_fifo_rx_clk_from_pl,
	output wire fmio_gem3_fifo_tx_clk_to_pl_bufg,
	output wire fmio_gem3_fifo_rx_clk_to_pl_bufg,
	// gem0_1588
	output wire emio_enet0_tx_sof,
	output wire emio_enet0_sync_frame_tx,
	output wire emio_enet0_delay_req_tx,
	output wire emio_enet0_pdelay_req_tx,
	output wire emio_enet0_pdelay_resp_tx,
	output wire emio_enet0_rx_sof,
	output wire emio_enet0_sync_frame_rx,
	output wire emio_enet0_delay_req_rx,
	output wire emio_enet0_pdelay_req_rx,
	output wire emio_enet0_pdelay_resp_rx,
	input  wire [1:0] emio_enet0_tsu_inc_ctrl,
	output wire emio_enet0_tsu_timer_cmp_val,
	//gem1_1588
	output wire emio_enet1_tx_sof,
	output wire emio_enet1_sync_frame_tx,
	output wire emio_enet1_delay_req_tx,
	output wire emio_enet1_pdelay_req_tx,
	output wire emio_enet1_pdelay_resp_tx,
	output wire emio_enet1_rx_sof,
	output wire emio_enet1_sync_frame_rx,
	output wire emio_enet1_delay_req_rx,
	output wire emio_enet1_pdelay_req_rx,
	output wire emio_enet1_pdelay_resp_rx,
	input  wire [1:0] emio_enet1_tsu_inc_ctrl,
	output wire emio_enet1_tsu_timer_cmp_val,
	//gem2_1588
	output wire emio_enet2_tx_sof,
	output wire emio_enet2_sync_frame_tx,
	output wire emio_enet2_delay_req_tx,
	output wire emio_enet2_pdelay_req_tx,
	output wire emio_enet2_pdelay_resp_tx,
	output wire emio_enet2_rx_sof,
	output wire emio_enet2_sync_frame_rx,
	output wire emio_enet2_delay_req_rx,
	output wire emio_enet2_pdelay_req_rx,
	output wire emio_enet2_pdelay_resp_rx,
	input  wire [1:0] emio_enet2_tsu_inc_ctrl,
	output wire emio_enet2_tsu_timer_cmp_val,
	//gem3_1588
	output wire emio_enet3_tx_sof,
	output wire emio_enet3_sync_frame_tx,
	output wire emio_enet3_delay_req_tx,
	output wire emio_enet3_pdelay_req_tx,
	output wire emio_enet3_pdelay_resp_tx,
	output wire emio_enet3_rx_sof,
	output wire emio_enet3_sync_frame_rx,
	output wire emio_enet3_delay_req_rx,
	output wire emio_enet3_pdelay_req_rx,
	output wire emio_enet3_pdelay_resp_rx,
	input  wire [1:0] emio_enet3_tsu_inc_ctrl,
	output wire emio_enet3_tsu_timer_cmp_val,
	// gem_tsu
	input wire  fmio_gem_tsu_clk_from_pl,
	output wire fmio_gem_tsu_clk_to_pl_bufg,
	input  wire emio_enet_tsu_clk,
	output wire [93:0] emio_enet0_enet_tsu_timer_cnt,
	// gem_misc
	input  wire emio_enet0_ext_int_in,
	input  wire emio_enet1_ext_int_in,
	input  wire emio_enet2_ext_int_in,
	input  wire emio_enet3_ext_int_in,
	output wire [1:0] emio_enet0_dma_bus_width,
	output wire [1:0] emio_enet1_dma_bus_width,
	output wire [1:0] emio_enet2_dma_bus_width,
	output wire [1:0] emio_enet3_dma_bus_width,
	// gpio
	input  wire [(C_EMIO_GPIO_WIDTH -1):0] emio_gpio_i,
	output wire [(C_EMIO_GPIO_WIDTH -1):0] emio_gpio_o,
	output wire [(C_EMIO_GPIO_WIDTH -1):0] emio_gpio_t,
	output wire [(C_EMIO_GPIO_WIDTH -1):0] emio_gpio_t_n,
	// i2c0
	input  wire emio_i2c0_scl_i,
	output wire emio_i2c0_scl_o,
	output wire emio_i2c0_scl_t_n,
	output wire emio_i2c0_scl_t,
	input  wire emio_i2c0_sda_i,
	output wire emio_i2c0_sda_o,
	output wire emio_i2c0_sda_t_n,
	output wire emio_i2c0_sda_t,
	// i2c1
	input  wire emio_i2c1_scl_i,
	output wire emio_i2c1_scl_o,
	output wire emio_i2c1_scl_t,
	output wire emio_i2c1_scl_t_n,
	input  wire emio_i2c1_sda_i,
	output wire emio_i2c1_sda_o,
	output wire emio_i2c1_sda_t,
	output wire emio_i2c1_sda_t_n,
	// uart0
	output wire emio_uart0_txd,
	input  wire emio_uart0_rxd,
	input  wire emio_uart0_ctsn,
	output wire emio_uart0_rtsn,
	input  wire emio_uart0_dsrn,
	input  wire emio_uart0_dcdn,
	input  wire emio_uart0_rin,
	output wire emio_uart0_dtrn,
	// uart1
	output wire emio_uart1_txd,
	input  wire emio_uart1_rxd,
	input  wire emio_uart1_ctsn,
	output wire emio_uart1_rtsn,
	input  wire emio_uart1_dsrn,
	input  wire emio_uart1_dcdn,
	input  wire emio_uart1_rin,
	output wire emio_uart1_dtrn,
	// sdio0
	output wire emio_sdio0_clkout,
	input  wire emio_sdio0_fb_clk_in,
	output wire emio_sdio0_cmdout,
	input  wire emio_sdio0_cmdin,
	output wire emio_sdio0_cmdena,
	input  wire [C_SD0_INTERNAL_BUS_WIDTH-1:0] emio_sdio0_datain,
	output wire [C_SD0_INTERNAL_BUS_WIDTH-1:0] emio_sdio0_dataout,
	output wire [C_SD0_INTERNAL_BUS_WIDTH-1:0] emio_sdio0_dataena,
	input  wire emio_sdio0_cd_n,
	input  wire emio_sdio0_wp,
	output wire emio_sdio0_ledcontrol,
	output wire emio_sdio0_buspower,
	output wire [2:0] emio_sdio0_bus_volt,
	// sdio1
	output wire emio_sdio1_clkout,
	input  wire emio_sdio1_fb_clk_in,
	output wire emio_sdio1_cmdout,
	input  wire emio_sdio1_cmdin,
	output wire emio_sdio1_cmdena,
	input  wire [C_SD1_INTERNAL_BUS_WIDTH-1:0] emio_sdio1_datain,
	output wire [C_SD1_INTERNAL_BUS_WIDTH-1:0] emio_sdio1_dataout,
	output wire [C_SD1_INTERNAL_BUS_WIDTH-1:0] emio_sdio1_dataena,
	input  wire emio_sdio1_cd_n,
	input  wire emio_sdio1_wp,
	output wire emio_sdio1_ledcontrol,
	output wire emio_sdio1_buspower,
	output wire [2:0] emio_sdio1_bus_volt,
	// spi0
	input  wire emio_spi0_sclk_i,
	output wire emio_spi0_sclk_o,
	output wire emio_spi0_sclk_t,
	output wire emio_spi0_sclk_t_n,
	input  wire emio_spi0_m_i,
	output wire emio_spi0_m_o,
	output wire emio_spi0_mo_t,
	output wire emio_spi0_mo_t_n,
	input  wire emio_spi0_s_i,
	output wire emio_spi0_s_o,
	output wire emio_spi0_so_t,
	output wire emio_spi0_so_t_n,
	input  wire emio_spi0_ss_i_n,
	output wire emio_spi0_ss_o_n,
	output wire emio_spi0_ss1_o_n,
	output wire emio_spi0_ss2_o_n,
	output wire emio_spi0_ss_n_t,
	output wire emio_spi0_ss_n_t_n,
	// spi1
	input  wire emio_spi1_sclk_i,
	output wire emio_spi1_sclk_o,
	output wire emio_spi1_sclk_t,
	output wire emio_spi1_sclk_t_n,
	input  wire emio_spi1_m_i,
	output wire emio_spi1_m_o,
	output wire emio_spi1_mo_t,
	output wire emio_spi1_mo_t_n,
	input  wire emio_spi1_s_i,
	output wire emio_spi1_s_o,
	output wire emio_spi1_so_t,
	output wire emio_spi1_so_t_n,
	input  wire emio_spi1_ss_i_n,
	output wire emio_spi1_ss_o_n,
	output wire emio_spi1_ss1_o_n,
	output wire emio_spi1_ss2_o_n,
	output wire emio_spi1_ss_n_t,
	output wire emio_spi1_ss_n_t_n,
	// trace
	input wire pl_ps_trace_clk,
	output wire ps_pl_tracectl,
	output wire [C_TRACE_DATA_WIDTH-1:0] ps_pl_tracedata,
	output reg trace_clk_out,
	// ttc0
	output wire [2:0] emio_ttc0_wave_o,
	input  wire [2:0] emio_ttc0_clk_i,
	// ttc1
	output wire [2:0] emio_ttc1_wave_o,
	input  wire [2:0] emio_ttc1_clk_i,
	// ttc2
	output wire [2:0] emio_ttc2_wave_o,
	input  wire [2:0] emio_ttc2_clk_i,
	// ttc3
	output wire [2:0] emio_ttc3_wave_o,
	input  wire [2:0] emio_ttc3_clk_i,
	// wdt0
	input  wire emio_wdt0_clk_i,
	output wire emio_wdt0_rst_o,
	// wdt1
	input  wire emio_wdt1_clk_i,
	output wire emio_wdt1_rst_o,
	// usb3
	input  wire emio_hub_port_overcrnt_usb3_0,
	input  wire emio_hub_port_overcrnt_usb3_1,
	input  wire emio_hub_port_overcrnt_usb2_0,
	input  wire emio_hub_port_overcrnt_usb2_1,
	output wire emio_u2dsport_vbus_ctrl_usb3_0,
	output wire emio_u2dsport_vbus_ctrl_usb3_1,
	output wire emio_u3dsport_vbus_ctrl_usb3_0,
	output wire emio_u3dsport_vbus_ctrl_usb3_1,
	//adma
	input  wire [7:0] adma_fci_clk,
	input  wire [7:0] pl2adma_cvld,
	input  wire [7:0] pl2adma_tack,
	output wire [7:0] adma2pl_cack,
	output wire [7:0] adma2pl_tvld,
	//gdma
	input  wire [7:0] perif_gdma_clk,
	input  wire [7:0] perif_gdma_cvld,
	input  wire [7:0] perif_gdma_tack,
	output wire [7:0] gdma_perif_cack,
	output wire [7:0] gdma_perif_tvld,
	// clk
	input  wire [3:0] pl_clock_stop,
	input  wire [1:0] pll_aux_refclk_lpd,
	input  wire [2:0] pll_aux_refclk_fpd,
	// audio
	input  wire [31:0] dp_s_axis_audio_tdata,
	input  wire dp_s_axis_audio_tid,
	input  wire dp_s_axis_audio_tvalid,
	output wire dp_s_axis_audio_tready,
	output wire [31:0] dp_m_axis_mixed_audio_tdata,
	output wire dp_m_axis_mixed_audio_tid,
	output wire dp_m_axis_mixed_audio_tvalid,
	input  wire dp_m_axis_mixed_audio_tready,
	input  wire dp_s_axis_audio_clk,
	// video
	input  wire dp_live_video_in_vsync,
	input  wire dp_live_video_in_hsync,
	input  wire dp_live_video_in_de,
	input  wire [35:0] dp_live_video_in_pixel1,
	input  wire dp_video_in_clk,
	output wire dp_video_out_hsync,
	output wire dp_video_out_vsync,
	output wire [35:0] dp_video_out_pixel1,
	input  wire dp_aux_data_in,
	output wire dp_aux_data_out,
	output wire dp_aux_data_oe_n,
	input  wire [7:0] dp_live_gfx_alpha_in,
	input  wire [35:0] dp_live_gfx_pixel1_in,
	input  wire dp_hot_plug_detect,
	input  wire dp_external_custom_event1,
	input  wire dp_external_custom_event2,
	input  wire dp_external_vsync_event,
	output wire dp_live_video_de_out,
	// event_apu
	input  wire pl_ps_eventi,
	output wire ps_pl_evento,
	output wire [3:0] ps_pl_standbywfe,
	output wire [3:0] ps_pl_standbywfi,
	input  wire [3:0] pl_ps_apugic_irq,
	input  wire [3:0] pl_ps_apugic_fiq,
	// event_rpu
	input  wire rpu_eventi0,
	input  wire rpu_eventi1,
	output wire rpu_evento0,
	output wire rpu_evento1,
	input  wire nfiq0_lpd_rpu,
	input  wire nfiq1_lpd_rpu,
	input  wire nirq0_lpd_rpu,
	input  wire nirq1_lpd_rpu,
	// ipi
	output wire irq_ipi_pl_0,
	output wire irq_ipi_pl_1,
	output wire irq_ipi_pl_2,
	output wire irq_ipi_pl_3,
	// stm
	input  wire [59:0] stm_event,
	// ftm
	input  wire pl_ps_trigack_0,
	input  wire pl_ps_trigack_1,
	input  wire pl_ps_trigack_2,
	input  wire pl_ps_trigack_3,
	input  wire pl_ps_trigger_0,
	input  wire pl_ps_trigger_1,
	input  wire pl_ps_trigger_2,
	input  wire pl_ps_trigger_3,
	output wire ps_pl_trigack_0,
	output wire ps_pl_trigack_1,
	output wire ps_pl_trigack_2,
	output wire ps_pl_trigack_3,
	output wire ps_pl_trigger_0,
	output wire ps_pl_trigger_1,
	output wire ps_pl_trigger_2,
	output wire ps_pl_trigger_3,
	output wire [31:0] ftm_gpo,
	input  wire [31:0] ftm_gpi,
	// irq
	input  wire [(C_NUM_F2P_0_INTR_INPUTS-1):0] pl_ps_irq0,
	input  wire [(C_NUM_F2P_1_INTR_INPUTS-1):0] pl_ps_irq1,
	//output wire [99:0] ps_pl_irq_lpd,
	//output wire [63:0] ps_pl_irq_fpd,

	//resets using gpio

	output wire pl_resetn0,
	output wire pl_resetn1,
	output wire pl_resetn2,
	output wire pl_resetn3,

	output wire ps_pl_irq_can0,
	output wire ps_pl_irq_can1,
	output wire ps_pl_irq_enet0,
	output wire ps_pl_irq_enet1,
	output wire ps_pl_irq_enet2,
	output wire ps_pl_irq_enet3,
	output wire ps_pl_irq_enet0_wake,
	output wire ps_pl_irq_enet1_wake,
	output wire ps_pl_irq_enet2_wake,
	output wire ps_pl_irq_enet3_wake,
	output wire ps_pl_irq_gpio,
	output wire ps_pl_irq_i2c0,
	output wire ps_pl_irq_i2c1,
	output wire ps_pl_irq_uart0,
	output wire ps_pl_irq_uart1,
	output wire ps_pl_irq_sdio0,
	output wire ps_pl_irq_sdio1,
	output wire ps_pl_irq_sdio0_wake,
	output wire ps_pl_irq_sdio1_wake,
	output wire ps_pl_irq_spi0,
	output wire ps_pl_irq_spi1,
	output wire ps_pl_irq_qspi,
	output wire ps_pl_irq_ttc0_0,
	output wire ps_pl_irq_ttc0_1,
	output wire ps_pl_irq_ttc0_2,
	output wire ps_pl_irq_ttc1_0,
	output wire ps_pl_irq_ttc1_1,
	output wire ps_pl_irq_ttc1_2,
	output wire ps_pl_irq_ttc2_0,
	output wire ps_pl_irq_ttc2_1,
	output wire ps_pl_irq_ttc2_2,
	output wire ps_pl_irq_ttc3_0,
	output wire ps_pl_irq_ttc3_1,
	output wire ps_pl_irq_ttc3_2,
	output wire ps_pl_irq_csu_pmu_wdt,
	output wire ps_pl_irq_lp_wdt,
	output wire [3:0] ps_pl_irq_usb3_0_endpoint,
	output wire ps_pl_irq_usb3_0_otg,
	output wire [3:0] ps_pl_irq_usb3_1_endpoint,
	output wire ps_pl_irq_usb3_1_otg,
	output wire [7:0] ps_pl_irq_adma_chan,
	output wire [1:0] ps_pl_irq_usb3_0_pmu_wakeup,
	output wire [7:0] ps_pl_irq_gdma_chan,
	output wire ps_pl_irq_csu,
	output wire ps_pl_irq_csu_dma,
	output wire ps_pl_irq_efuse,
	output wire ps_pl_irq_xmpu_lpd,
	output wire ps_pl_irq_ddr_ss,
	output wire ps_pl_irq_nand,
	output wire ps_pl_irq_fp_wdt,
	output wire [1:0] ps_pl_irq_pcie_msi,
	output wire ps_pl_irq_pcie_legacy,
	output wire ps_pl_irq_pcie_dma,
	output wire ps_pl_irq_pcie_msc,
	output wire ps_pl_irq_dport,
	output wire ps_pl_irq_fpd_apb_int,
	output wire ps_pl_irq_fpd_atb_error,
	output wire ps_pl_irq_dpdma,
	output wire ps_pl_irq_apm_fpd,
	output wire ps_pl_irq_gpu,
	output wire ps_pl_irq_sata,
	output wire ps_pl_irq_xmpu_fpd,
	output wire [3:0] ps_pl_irq_apu_cpumnt,
	output wire [3:0] ps_pl_irq_apu_cti,
	output wire [3:0] ps_pl_irq_apu_pmu,
	output wire [3:0] ps_pl_irq_apu_comm,
	output wire ps_pl_irq_apu_l2err,
	output wire ps_pl_irq_apu_exterr,
	output wire ps_pl_irq_apu_regs,
	output wire ps_pl_irq_intf_ppd_cci,
	output wire ps_pl_irq_intf_fpd_smmu,
	output wire ps_pl_irq_atb_err_lpd,
	output wire ps_pl_irq_aib_axi,
	output wire ps_pl_irq_ams,
	output wire ps_pl_irq_lpd_apm,
	output wire ps_pl_irq_rtc_alaram,
	output wire ps_pl_irq_rtc_seconds,
	output wire ps_pl_irq_clkmon,
	output wire ps_pl_irq_ipi_channel0,
	output wire ps_pl_irq_ipi_channel1,
	output wire ps_pl_irq_ipi_channel2,
	output wire ps_pl_irq_ipi_channel7,
	output wire ps_pl_irq_ipi_channel8,
	output wire ps_pl_irq_ipi_channel9,
	output wire ps_pl_irq_ipi_channel10,
	output wire [1:0] ps_pl_irq_rpu_pm,
	output wire ps_pl_irq_ocm_error,
	output wire ps_pl_irq_lpd_apb_intr,
	output wire ps_pl_irq_r5_core0_ecc_error,
	output wire ps_pl_irq_r5_core1_ecc_error,


	// rtc
	output wire osc_rtc_clk,
	// pmu
	input  wire [31:0] pl_pmu_gpi,
	output wire [31:0] pmu_pl_gpo,
	input  wire aib_pmu_afifm_fpd_ack,
	input  wire aib_pmu_afifm_lpd_ack,
	output wire pmu_aib_afifm_fpd_req,
	output wire pmu_aib_afifm_lpd_req,
	output wire [46:0] pmu_error_to_pl,
	input  wire [3:0] pmu_error_from_pl,
	// misc
	input  wire ddrc_ext_refresh_rank0_req,
	input  wire ddrc_ext_refresh_rank1_req,
	input  wire ddrc_refresh_pl_clk,
	input  wire pl_acpinact,

	//For Clock buffering
	//FCLK
	output    wire pl_clk3,
	output    wire pl_clk2,
	output    wire pl_clk1,
	output    wire pl_clk0,

	///////////////////////////
	// ACE interface allotment
	//////////////////////////
	input  wire [15:0] sacefpd_awuser,
	input  wire [15:0] sacefpd_aruser,

	//Debug and Test signals
	input  [3:0]  test_adc_clk,
	input  [31:0] test_adc_in,
	input  [31:0] test_adc2_in,
	output [15:0] test_db,
	output [19:0] test_adc_out,
	output [7:0]  test_ams_osc,
	output [15:0] test_mon_data,
	input         test_dclk,
	input         test_den,
	input         test_dwe,
	input  [7:0]  test_daddr,
	input  [15:0] test_di,
	output        test_drdy,
	output [15:0] test_do,
	input         test_convst,
	input  [3:0]  pstp_pl_clk,
	input  [31:0] pstp_pl_in,
	output [31:0] pstp_pl_out,
	input  [31:0] pstp_pl_ts,
	input         fmio_test_gem_scanmux_1,
	input         fmio_test_gem_scanmux_2,
	input         test_char_mode_fpd_n,
	input         test_char_mode_lpd_n,
	input         fmio_test_io_char_scan_clock,
	input         fmio_test_io_char_scanenable,
	input         fmio_test_io_char_scan_in,
	output        fmio_test_io_char_scan_out,
	input         fmio_test_io_char_scan_reset_n,
	input         fmio_char_afifslpd_test_select_n,
	input         fmio_char_afifslpd_test_input,
	output        fmio_char_afifslpd_test_output,
	input         fmio_char_afifsfpd_test_select_n,
	input         fmio_char_afifsfpd_test_input,
	output        fmio_char_afifsfpd_test_output,
	input         io_char_audio_in_test_data,
	input         io_char_audio_mux_sel_n,
	input         io_char_video_in_test_data,
	input         io_char_video_mux_sel_n,
	output        io_char_video_out_test_data,
	output        io_char_audio_out_test_data,
	input         fmio_test_qspi_scanmux_1_n,
	input         fmio_test_sdio_scanmux_1,
	input         fmio_test_sdio_scanmux_2,
	input   [3:0] fmio_sd0_dll_test_in_n,
	output  [7:0] fmio_sd0_dll_test_out,
	input   [3:0] fmio_sd1_dll_test_in_n,
	output  [7:0] fmio_sd1_dll_test_out,
	input         test_pl_scan_chopper_si,
	output        test_pl_scan_chopper_so,
	input         test_pl_scan_chopper_trig,
	input         test_pl_scan_clk0,
	input         test_pl_scan_clk1,
	input         test_pl_scan_edt_clk,
	input         test_pl_scan_edt_in_apu,
	input         test_pl_scan_edt_in_cpu,
	input   [3:0] test_pl_scan_edt_in_ddr,
	input   [9:0] test_pl_scan_edt_in_fp,
	input   [3:0] test_pl_scan_edt_in_gpu,
	input   [8:0] test_pl_scan_edt_in_lp,
	input   [1:0] test_pl_scan_edt_in_usb3,
	output        test_pl_scan_edt_out_apu,
	output        test_pl_scan_edt_out_cpu0,
	output        test_pl_scan_edt_out_cpu1,
	output        test_pl_scan_edt_out_cpu2,
	output        test_pl_scan_edt_out_cpu3,
	output  [3:0] test_pl_scan_edt_out_ddr,
	output  [9:0] test_pl_scan_edt_out_fp,
	output  [3:0] test_pl_scan_edt_out_gpu,
	output  [8:0] test_pl_scan_edt_out_lp,
	output  [1:0] test_pl_scan_edt_out_usb3,
	input         test_pl_scan_edt_update,
	input         test_pl_scan_reset_n,
	input         test_pl_scanenable,
	input         test_pl_scan_pll_reset,
	input         test_pl_scan_spare_in0,
	input         test_pl_scan_spare_in1,
	output        test_pl_scan_spare_out0,
	output        test_pl_scan_spare_out1,
	input         test_pl_scan_wrap_clk,
	input         test_pl_scan_wrap_ishift,
	input         test_pl_scan_wrap_oshift,
	input         test_pl_scan_slcr_config_clk,
	input         test_pl_scan_slcr_config_rstn,
	input         test_pl_scan_slcr_config_si,
	input         test_pl_scan_spare_in2,
	input         test_pl_scanenable_slcr_en,
	output  [4:0] test_pl_pll_lock_out,
	output        test_pl_scan_slcr_config_so,
	input  [20:0] tst_rtc_calibreg_in,
	output [20:0] tst_rtc_calibreg_out,
	input         tst_rtc_calibreg_we,
	input         tst_rtc_clk,
	output        tst_rtc_osc_clk_out,
	output [31:0] tst_rtc_sec_counter_out,
	output        tst_rtc_seconds_raw_int,
	input         tst_rtc_testclock_select_n,
	output [15:0] tst_rtc_tick_counter_out,
	input  [31:0] tst_rtc_timesetreg_in,
	output [31:0] tst_rtc_timesetreg_out,
	input         tst_rtc_disable_bat_op,
	input   [3:0] tst_rtc_osc_cntrl_in,
	output  [3:0] tst_rtc_osc_cntrl_out,
	input         tst_rtc_osc_cntrl_we,
	input         tst_rtc_sec_reload,
	input         tst_rtc_timesetreg_we,
	input         tst_rtc_testmode_n,
	input         test_usb0_funcmux_0_n,
	input         test_usb1_funcmux_0_n,
	input         test_usb0_scanmux_0_n,
	input         test_usb1_scanmux_0_n,
	output [31:0] lpd_pll_test_out,
	input   [2:0] pl_lpd_pll_test_ck_sel_n,
	input         pl_lpd_pll_test_fract_clk_sel_n,
	input         pl_lpd_pll_test_fract_en_n,
	input         pl_lpd_pll_test_mux_sel,
	input   [3:0] pl_lpd_pll_test_sel,
	output [31:0] fpd_pll_test_out,
	input   [2:0] pl_fpd_pll_test_ck_sel_n,
	input         pl_fpd_pll_test_fract_clk_sel_n,
	input         pl_fpd_pll_test_fract_en_n,
	input   [1:0] pl_fpd_pll_test_mux_sel,
	input   [3:0] pl_fpd_pll_test_sel,
	input   [1:0] fmio_char_gem_selection,
	input         fmio_char_gem_test_select_n,
	input         fmio_char_gem_test_input,
	output        fmio_char_gem_test_output,
	output        test_ddr2pl_dcd_skewout,
	input         test_pl2ddr_dcd_sample_pulse,
	input         test_bscan_en_n,
	input         test_bscan_tdi,
	input         test_bscan_updatedr,
	input         test_bscan_shiftdr,
	input         test_bscan_reset_tap_b,
	input         test_bscan_misr_jtag_load,
	input         test_bscan_intest,
	input         test_bscan_extest,
	input         test_bscan_clockdr,
	input         test_bscan_ac_mode,
	input         test_bscan_ac_test,
	input         test_bscan_init_memory,
	input         test_bscan_mode_c,
	output        test_bscan_tdo,
	input         i_dbg_l0_txclk,
	input         i_dbg_l0_rxclk,
	input         i_dbg_l1_txclk,
	input         i_dbg_l1_rxclk,
	input         i_dbg_l2_txclk,
	input         i_dbg_l2_rxclk,
	input         i_dbg_l3_txclk,
	input         i_dbg_l3_rxclk,
	input         i_afe_rx_symbol_clk_by_2_pl,
	input         pl_fpd_spare_0_in,
	input         pl_fpd_spare_1_in,
	input         pl_fpd_spare_2_in,
	input         pl_fpd_spare_3_in,
	input         pl_fpd_spare_4_in,
	output        fpd_pl_spare_0_out,
	output        fpd_pl_spare_1_out,
	output        fpd_pl_spare_2_out,
	output        fpd_pl_spare_3_out,
	output        fpd_pl_spare_4_out,
	input         pl_lpd_spare_0_in,
	input         pl_lpd_spare_1_in,
	input         pl_lpd_spare_2_in,
	input         pl_lpd_spare_3_in,
	input         pl_lpd_spare_4_in,
	output        lpd_pl_spare_0_out,
	output        lpd_pl_spare_1_out,
	output        lpd_pl_spare_2_out,
	output        lpd_pl_spare_3_out,
	output        lpd_pl_spare_4_out,
	output        o_dbg_l0_phystatus,
	output [19:0] o_dbg_l0_rxdata,
	output  [1:0] o_dbg_l0_rxdatak,
	output        o_dbg_l0_rxvalid,
	output  [2:0] o_dbg_l0_rxstatus,
	output        o_dbg_l0_rxelecidle,
	output        o_dbg_l0_rstb,
	output [19:0] o_dbg_l0_txdata,
	output  [1:0] o_dbg_l0_txdatak,
	output  [1:0] o_dbg_l0_rate,
	output  [1:0] o_dbg_l0_powerdown,
	output        o_dbg_l0_txelecidle,
	output        o_dbg_l0_txdetrx_lpback,
	output        o_dbg_l0_rxpolarity,
	output        o_dbg_l0_tx_sgmii_ewrap,
	output        o_dbg_l0_rx_sgmii_en_cdet,
	output [19:0] o_dbg_l0_sata_corerxdata,
	output  [1:0] o_dbg_l0_sata_corerxdatavalid,
	output        o_dbg_l0_sata_coreready,
	output        o_dbg_l0_sata_coreclockready,
	output        o_dbg_l0_sata_corerxsignaldet,
	output [19:0] o_dbg_l0_sata_phyctrltxdata,
	output        o_dbg_l0_sata_phyctrltxidle,
	output  [1:0] o_dbg_l0_sata_phyctrltxrate,
	output  [1:0] o_dbg_l0_sata_phyctrlrxrate,
	output        o_dbg_l0_sata_phyctrltxrst,
	output        o_dbg_l0_sata_phyctrlrxrst,
	output        o_dbg_l0_sata_phyctrlreset,
	output        o_dbg_l0_sata_phyctrlpartial,
	output        o_dbg_l0_sata_phyctrlslumber,
	output        o_dbg_l1_phystatus,
	output [19:0] o_dbg_l1_rxdata,
	output  [1:0] o_dbg_l1_rxdatak,
	output        o_dbg_l1_rxvalid,
	output  [2:0] o_dbg_l1_rxstatus,
	output        o_dbg_l1_rxelecidle,
	output        o_dbg_l1_rstb,
	output [19:0] o_dbg_l1_txdata,
	output  [1:0] o_dbg_l1_txdatak,
	output  [1:0] o_dbg_l1_rate,
	output  [1:0] o_dbg_l1_powerdown,
	output        o_dbg_l1_txelecidle,
	output        o_dbg_l1_txdetrx_lpback,
	output        o_dbg_l1_rxpolarity,
	output        o_dbg_l1_tx_sgmii_ewrap,
	output        o_dbg_l1_rx_sgmii_en_cdet,
	output [19:0] o_dbg_l1_sata_corerxdata,
	output  [1:0] o_dbg_l1_sata_corerxdatavalid,
	output        o_dbg_l1_sata_coreready,
	output        o_dbg_l1_sata_coreclockready,
	output        o_dbg_l1_sata_corerxsignaldet,
	output [19:0] o_dbg_l1_sata_phyctrltxdata,
	output        o_dbg_l1_sata_phyctrltxidle,
	output  [1:0] o_dbg_l1_sata_phyctrltxrate,
	output  [1:0] o_dbg_l1_sata_phyctrlrxrate,
	output        o_dbg_l1_sata_phyctrltxrst,
	output        o_dbg_l1_sata_phyctrlrxrst,
	output        o_dbg_l1_sata_phyctrlreset,
	output        o_dbg_l1_sata_phyctrlpartial,
	output        o_dbg_l1_sata_phyctrlslumber,
	output        o_dbg_l2_phystatus,
	output [19:0] o_dbg_l2_rxdata,
	output  [1:0] o_dbg_l2_rxdatak,
	output        o_dbg_l2_rxvalid,
	output  [2:0] o_dbg_l2_rxstatus,
	output        o_dbg_l2_rxelecidle,
	output        o_dbg_l2_rstb,
	output [19:0] o_dbg_l2_txdata,
	output  [1:0] o_dbg_l2_txdatak,
	output  [1:0] o_dbg_l2_rate,
	output  [1:0] o_dbg_l2_powerdown,
	output        o_dbg_l2_txelecidle,
	output        o_dbg_l2_txdetrx_lpback,
	output        o_dbg_l2_rxpolarity,
	output        o_dbg_l2_tx_sgmii_ewrap,
	output        o_dbg_l2_rx_sgmii_en_cdet,
	output [19:0] o_dbg_l2_sata_corerxdata,
	output  [1:0] o_dbg_l2_sata_corerxdatavalid,
	output        o_dbg_l2_sata_coreready,
	output        o_dbg_l2_sata_coreclockready,
	output        o_dbg_l2_sata_corerxsignaldet,
	output  [19:0] o_dbg_l2_sata_phyctrltxdata,
	output        o_dbg_l2_sata_phyctrltxidle,
	output  [1:0] o_dbg_l2_sata_phyctrltxrate,
	output  [1:0] o_dbg_l2_sata_phyctrlrxrate,
	output        o_dbg_l2_sata_phyctrltxrst,
	output        o_dbg_l2_sata_phyctrlrxrst,
	output        o_dbg_l2_sata_phyctrlreset,
	output        o_dbg_l2_sata_phyctrlpartial,
	output        o_dbg_l2_sata_phyctrlslumber,
	output        o_dbg_l3_phystatus,
	output [19:0] o_dbg_l3_rxdata,
	output  [1:0] o_dbg_l3_rxdatak,
	output        o_dbg_l3_rxvalid,
	output  [2:0] o_dbg_l3_rxstatus,
	output        o_dbg_l3_rxelecidle,
	output        o_dbg_l3_rstb,
	output [19:0] o_dbg_l3_txdata,
	output  [1:0] o_dbg_l3_txdatak,
	output  [1:0] o_dbg_l3_rate,
	output  [1:0] o_dbg_l3_powerdown,
	output        o_dbg_l3_txelecidle,
	output        o_dbg_l3_txdetrx_lpback,
	output        o_dbg_l3_rxpolarity,
	output        o_dbg_l3_tx_sgmii_ewrap,
	output        o_dbg_l3_rx_sgmii_en_cdet,
	output [19:0] o_dbg_l3_sata_corerxdata,
	output  [1:0] o_dbg_l3_sata_corerxdatavalid,
	output        o_dbg_l3_sata_coreready,
	output        o_dbg_l3_sata_coreclockready,
	output        o_dbg_l3_sata_corerxsignaldet,
	output  [19:0] o_dbg_l3_sata_phyctrltxdata,
	output        o_dbg_l3_sata_phyctrltxidle,
	output  [1:0] o_dbg_l3_sata_phyctrltxrate,
	output  [1:0] o_dbg_l3_sata_phyctrlrxrate,
	output        o_dbg_l3_sata_phyctrltxrst,
	output        o_dbg_l3_sata_phyctrlrxrst,
	output        o_dbg_l3_sata_phyctrlreset,
	output        o_dbg_l3_sata_phyctrlpartial,
	output        o_dbg_l3_sata_phyctrlslumber,
	output        dbg_path_fifo_bypass,
	input         i_afe_pll_pd_hs_clock_r,
	input         i_afe_mode,
	input         i_bgcal_afe_mode,
	output        o_afe_cmn_calib_comp_out,
	input         i_afe_cmn_bg_enable_low_leakage,
	input         i_afe_cmn_bg_iso_ctrl_bar,
	input         i_afe_cmn_bg_pd,
	input         i_afe_cmn_bg_pd_bg_ok,
	input         i_afe_cmn_bg_pd_ptat,
	input         i_afe_cmn_calib_en_iconst,
	input         i_afe_cmn_calib_enable_low_leakage,
	input         i_afe_cmn_calib_iso_ctrl_bar,
	output [12:0] o_afe_pll_dco_count,
	output        o_afe_pll_clk_sym_hs,
	output        o_afe_pll_fbclk_frac,
	output        o_afe_rx_pipe_lfpsbcn_rxelecidle,
	output        o_afe_rx_pipe_sigdet,
	output [19:0] o_afe_rx_symbol,
	output        o_afe_rx_symbol_clk_by_2,
	output        o_afe_rx_uphy_save_calcode,
	output        o_afe_rx_uphy_startloop_buf,
	output        o_afe_rx_uphy_rx_calib_done,
	input         i_afe_rx_rxpma_rstb,
	input   [7:0] i_afe_rx_uphy_restore_calcode_data,
	input         i_afe_rx_pipe_rxeqtraining,
	input         i_afe_rx_iso_hsrx_ctrl_bar,
	input         i_afe_rx_iso_lfps_ctrl_bar,
	input         i_afe_rx_iso_sigdet_ctrl_bar,
	input         i_afe_rx_hsrx_clock_stop_req,
	output  [7:0] o_afe_rx_uphy_save_calcode_data,
	output        o_afe_rx_hsrx_clock_stop_ack,
	output        o_afe_pg_avddcr,
	output        o_afe_pg_avddio,
	output        o_afe_pg_dvddcr,
	output        o_afe_pg_static_avddcr,
	output        o_afe_pg_static_avddio,
	input         i_pll_afe_mode,
	input  [10:0] i_afe_pll_coarse_code,
	input         i_afe_pll_en_clock_hs_div2,
	input  [15:0] i_afe_pll_fbdiv,
	input         i_afe_pll_load_fbdiv,
	input         i_afe_pll_pd,
	input         i_afe_pll_pd_pfd,
	input         i_afe_pll_rst_fdbk_div,
	input         i_afe_pll_startloop,
	input   [5:0] i_afe_pll_v2i_code,
	input   [4:0] i_afe_pll_v2i_prog,
	input         i_afe_pll_vco_cnt_window,
	input         i_afe_rx_mphy_gate_symbol_clk,
	input         i_afe_rx_mphy_mux_hsb_ls,
	input         i_afe_rx_pipe_rx_term_enable,
	input         i_afe_rx_uphy_biasgen_iconst_core_mirror_enable,
	input         i_afe_rx_uphy_biasgen_iconst_io_mirror_enable,
	input         i_afe_rx_uphy_biasgen_irconst_core_mirror_enable,
	input         i_afe_rx_uphy_enable_cdr,
	input         i_afe_rx_uphy_enable_low_leakage,
	input         i_afe_rx_rxpma_refclk_dig,
	input         i_afe_rx_uphy_hsrx_rstb,
	input         i_afe_rx_uphy_pdn_hs_des,
	input         i_afe_rx_uphy_pd_samp_c2c,
	input         i_afe_rx_uphy_pd_samp_c2c_eclk,
	input         i_afe_rx_uphy_pso_clk_lane,
	input         i_afe_rx_uphy_pso_eq,
	input         i_afe_rx_uphy_pso_hsrxdig,
	input         i_afe_rx_uphy_pso_iqpi,
	input         i_afe_rx_uphy_pso_lfpsbcn,
	input         i_afe_rx_uphy_pso_samp_flops,
	input         i_afe_rx_uphy_pso_sigdet,
	input         i_afe_rx_uphy_restore_calcode,
	input         i_afe_rx_uphy_run_calib,
	input         i_afe_rx_uphy_rx_lane_polarity_swap,
	input         i_afe_rx_uphy_startloop_pll,
	input   [1:0] i_afe_rx_uphy_hsclk_division_factor,
	input   [7:0] i_afe_rx_uphy_rx_pma_opmode,
	input   [1:0] i_afe_tx_enable_hsclk_division,
	input         i_afe_tx_enable_ldo,
	input         i_afe_tx_enable_ref,
	input         i_afe_tx_enable_supply_hsclk,
	input         i_afe_tx_enable_supply_pipe,
	input         i_afe_tx_enable_supply_serializer,
	input         i_afe_tx_enable_supply_uphy,
	input         i_afe_tx_hs_ser_rstb,
	input  [19:0] i_afe_tx_hs_symbol,
	input         i_afe_tx_mphy_tx_ls_data,
	input   [1:0] i_afe_tx_pipe_tx_enable_idle_mode,
	input   [1:0] i_afe_tx_pipe_tx_enable_lfps,
	input         i_afe_tx_pipe_tx_enable_rxdet,
	input   [7:0] i_afe_TX_uphy_txpma_opmode,
	input         i_afe_TX_pmadig_digital_reset_n,
	input         i_afe_TX_serializer_rst_rel,
	input         i_afe_TX_pll_symb_clk_2,
	input   [1:0] i_afe_TX_ana_if_rate,
	input         i_afe_TX_en_dig_sublp_mode,
	input   [2:0] i_afe_TX_LPBK_SEL,
	input         i_afe_TX_iso_ctrl_bar,
	input         i_afe_TX_ser_iso_ctrl_bar,
	input         i_afe_TX_lfps_clk,
	input         i_afe_TX_serializer_rstb,
	output        o_afe_TX_dig_reset_rel_ack,
	output        o_afe_TX_pipe_TX_dn_rxdet,
	output        o_afe_TX_pipe_TX_dp_rxdet,
	input         i_afe_tx_pipe_tx_fast_est_common_mode,
	output        o_dbg_l0_txclk,
	output        o_dbg_l0_rxclk,
	output        o_dbg_l1_txclk,
	output        o_dbg_l1_rxclk,
	output        o_dbg_l2_txclk,
	output        o_dbg_l2_rxclk,
	output        o_dbg_l3_txclk,
	output        o_dbg_l3_rxclk
);
	// [5:0]AxUSER driven by PL-AXI_ID
	// [9:6]AxUSER tied to 4'b1111
	// [15:10]AxUSER tied to 6'b000000

	//Write channel
	// assign sacefpd_awuser[15:10] = 6'b000000;
	// assign sacefpd_awuser[9:6] = 4'b1111;


	//Read Channel
	// assign sacefpd_aruser[15:10] = 6'b000000;
	// assign sacefpd_aruser[9:6] = 4'b1111;

	wire emio_sdio0_cmdena_i;
	wire [C_SD0_INTERNAL_BUS_WIDTH-1:0] emio_sdio0_dataena_i;
	wire emio_sdio1_cmdena_i;
	wire [C_SD1_INTERNAL_BUS_WIDTH-1:0] emio_sdio1_dataena_i;

	wire [3:0] pl_clk_unbuffered;
	wire [3:0] pl_clk_buffered;

	wire [127:0] maxigp0_wdata_i;
	wire [127:0] maxigp1_wdata_i;
	wire [127:0] maxigp2_wdata_i;
	wire [127:0] saxigp0_wdata_i;
	wire [127:0] saxigp1_wdata_i;
	wire [127:0] saxigp2_wdata_i;
	wire [127:0] saxigp3_wdata_i;
	wire [127:0] saxigp4_wdata_i;
	wire [127:0] saxigp5_wdata_i;
	wire [127:0] saxigp6_wdata_i;

	wire [127:0] maxigp0_rdata_i;
	wire [127:0] maxigp1_rdata_i;
	wire [127:0] maxigp2_rdata_i;
	wire [127:0] saxigp0_rdata_i;
	wire [127:0] saxigp1_rdata_i;
	wire [127:0] saxigp2_rdata_i;
	wire [127:0] saxigp3_rdata_i;
	wire [127:0] saxigp4_rdata_i;
	wire [127:0] saxigp5_rdata_i;
	wire [127:0] saxigp6_rdata_i;

	wire [15:0] maxigp0_wstrb_i;
	wire [15:0] maxigp1_wstrb_i;
	wire [15:0] maxigp2_wstrb_i;
	wire [15:0] saxigp0_wstrb_i;
	wire [15:0] saxigp1_wstrb_i;
	wire [15:0] saxigp2_wstrb_i;
	wire [15:0] saxigp3_wstrb_i;
	wire [15:0] saxigp4_wstrb_i;
	wire [15:0] saxigp5_wstrb_i;
	wire [15:0] saxigp6_wstrb_i;

	wire [7:0]   irq_f2p_0_i;
	wire [7:0]   irq_f2p_0_null = 8'h00;
	wire [7:0]   irq_f2p_1_i;
	wire [7:0]   irq_f2p_1_null = 8'h00;

	wire [95:0] emio_gpio_o_temp;
	wire [95:0] emio_gpio_i_temp;
	wire [95:0] emio_gpio_t_temp;

	wire saxihpc0_fpd_rclk_temp;
	wire saxihpc0_fpd_wclk_temp;
	wire saxihpc1_fpd_rclk_temp;
	wire saxihpc1_fpd_wclk_temp;
	wire saxihp0_fpd_rclk_temp;
	wire saxihp0_fpd_wclk_temp;
	wire saxihp1_fpd_rclk_temp;
	wire saxihp1_fpd_wclk_temp;
	wire saxihp2_fpd_rclk_temp;
	wire saxihp2_fpd_wclk_temp;
	wire saxihp3_fpd_rclk_temp;
	wire saxihp3_fpd_wclk_temp;
	wire saxi_lpd_rclk_temp;
	wire saxi_lpd_wclk_temp;

	wire emio_i2c0_sda_tri;
	wire emio_i2c0_scl_tri;
	wire emio_i2c1_sda_tri;
	wire emio_i2c1_scl_tri;

	wire emio_enet0_mdio_tri;
	wire emio_enet1_mdio_tri;
	wire emio_enet2_mdio_tri;
	wire emio_enet3_mdio_tri;
	wire emio_gpio_tri;
	wire emio_spi0_sclk_tri;
	wire emio_spi0_mo_tri;
	wire emio_spi0_so_tri;
	wire emio_spi0_ss_n_tri;
	wire emio_spi1_sclk_tri;
	wire emio_spi1_mo_tri;
	wire emio_spi1_so_tri;
	wire emio_spi1_ss_n_tri;

	wire [18:0] ps_pl_irq_lpd_low;
	wire [19:0] ps_pl_irq_fpd_low;

	wire dp_video_ref_clk_i;
	wire dp_audio_ref_clk_i;

	wire fmio_gem0_fifo_rx_clk;
	wire fmio_gem0_fifo_tx_clk;
	wire fmio_gem0_fifo_rx_clk_temp;
	wire fmio_gem0_fifo_tx_clk_temp;
	wire fmio_gem1_fifo_rx_clk;
	wire fmio_gem1_fifo_tx_clk;
	wire fmio_gem1_fifo_rx_clk_temp;
	wire fmio_gem1_fifo_tx_clk_temp;
	wire fmio_gem2_fifo_rx_clk;
	wire fmio_gem2_fifo_tx_clk;
	wire fmio_gem2_fifo_rx_clk_temp;
	wire fmio_gem2_fifo_tx_clk_temp;
	wire fmio_gem3_fifo_rx_clk;
	wire fmio_gem3_fifo_tx_clk;
	wire fmio_gem3_fifo_rx_clk_temp;
	wire fmio_gem3_fifo_tx_clk_temp;

	generate
	if (C_EN_FIFO_ENET0 == "1") begin
		assign fmio_gem0_fifo_rx_clk = fmio_gem0_fifo_rx_clk_temp;
		assign fmio_gem0_fifo_tx_clk = fmio_gem0_fifo_tx_clk_temp;
	end else begin
		assign fmio_gem0_fifo_rx_clk = 0;
		assign fmio_gem0_fifo_tx_clk = 0;
	end
	endgenerate

	generate
	if (C_EN_FIFO_ENET1 == "1") begin
		assign fmio_gem1_fifo_rx_clk = fmio_gem1_fifo_rx_clk_temp;
		assign fmio_gem1_fifo_tx_clk = fmio_gem1_fifo_tx_clk_temp;
	end else begin
		assign fmio_gem1_fifo_rx_clk = 0;
		assign fmio_gem1_fifo_tx_clk = 0;
	end
	endgenerate

	generate
	if (C_EN_FIFO_ENET2 == "1") begin
		assign fmio_gem2_fifo_rx_clk = fmio_gem2_fifo_rx_clk_temp;
		assign fmio_gem2_fifo_tx_clk = fmio_gem2_fifo_tx_clk_temp;
	end else begin
		assign fmio_gem2_fifo_rx_clk = 0;
		assign fmio_gem2_fifo_tx_clk = 0;
	end
	endgenerate

	generate
	if (C_EN_FIFO_ENET3 == "1") begin
		assign fmio_gem3_fifo_rx_clk = fmio_gem3_fifo_rx_clk_temp;
		assign fmio_gem3_fifo_tx_clk = fmio_gem3_fifo_tx_clk_temp;
	end else begin
		assign fmio_gem3_fifo_rx_clk = 0;
		assign fmio_gem3_fifo_tx_clk = 0;
	end
	endgenerate

	assign fmio_gem0_fifo_rx_clk_to_pl_bufg = fmio_gem0_fifo_rx_clk_temp;
	assign fmio_gem0_fifo_tx_clk_to_pl_bufg = fmio_gem0_fifo_tx_clk_temp;
	assign fmio_gem1_fifo_rx_clk_to_pl_bufg = fmio_gem1_fifo_rx_clk_temp;
	assign fmio_gem1_fifo_tx_clk_to_pl_bufg = fmio_gem1_fifo_tx_clk_temp;
	assign fmio_gem2_fifo_rx_clk_to_pl_bufg = fmio_gem2_fifo_rx_clk_temp;
	assign fmio_gem2_fifo_tx_clk_to_pl_bufg = fmio_gem2_fifo_tx_clk_temp;
	assign fmio_gem3_fifo_rx_clk_to_pl_bufg = fmio_gem3_fifo_rx_clk_temp;
	assign fmio_gem3_fifo_tx_clk_to_pl_bufg = fmio_gem3_fifo_tx_clk_temp;

	// Adding tristate inverters

	assign  emio_sdio0_cmdena = ~emio_sdio0_cmdena_i;
	assign  emio_sdio0_dataena = ~emio_sdio0_dataena_i;
	assign  emio_sdio1_cmdena  = ~emio_sdio1_cmdena_i;
	assign  emio_sdio1_dataena = ~emio_sdio1_dataena_i;

	assign emio_i2c0_sda_t_n = emio_i2c0_sda_tri;
	assign emio_i2c0_sda_t = ~emio_i2c0_sda_tri;

	assign emio_i2c0_scl_t_n = emio_i2c0_scl_tri;
	assign emio_i2c0_scl_t = ~emio_i2c0_scl_tri;

	assign emio_i2c1_sda_t_n = emio_i2c1_sda_tri;
	assign emio_i2c1_sda_t = ~emio_i2c1_sda_tri;

	assign emio_i2c1_scl_t_n = emio_i2c1_scl_tri;
	assign emio_i2c1_scl_t = ~emio_i2c1_scl_tri;

	assign emio_enet0_mdio_t_n = emio_enet0_mdio_tri;
	assign emio_enet0_mdio_t = ~emio_enet0_mdio_tri;

	assign emio_enet1_mdio_t_n = emio_enet1_mdio_tri;
	assign emio_enet1_mdio_t = ~emio_enet1_mdio_tri;

	assign emio_enet2_mdio_t_n = emio_enet2_mdio_tri;
	assign emio_enet2_mdio_t = ~emio_enet2_mdio_tri;

	assign emio_enet3_mdio_t_n = emio_enet3_mdio_tri;
	assign emio_enet3_mdio_t = ~emio_enet3_mdio_tri;

	assign emio_spi0_sclk_t_n = emio_spi0_sclk_tri;
	assign emio_spi0_sclk_t = ~emio_spi0_sclk_tri;

	assign emio_spi1_sclk_t_n = emio_spi1_sclk_tri;
	assign emio_spi1_sclk_t = ~emio_spi1_sclk_tri;

	assign emio_spi0_mo_t_n = emio_spi0_mo_tri;
	assign emio_spi0_mo_t = ~emio_spi0_mo_tri;

	assign emio_spi1_mo_t_n = emio_spi1_mo_tri;
	assign emio_spi1_mo_t = ~emio_spi1_mo_tri;

	assign emio_spi0_so_t_n = emio_spi0_so_tri;
	assign emio_spi0_so_t = ~emio_spi0_so_tri;

	assign emio_spi1_so_t_n = emio_spi1_so_tri;
	assign emio_spi1_so_t = ~emio_spi1_so_tri;

	assign emio_spi0_ss_n_t_n = emio_spi0_ss_n_tri;
	assign emio_spi0_ss_n_t = ~emio_spi0_ss_n_tri;

	assign emio_spi1_ss_n_t_n = emio_spi1_ss_n_tri;
	assign emio_spi1_ss_n_t = ~emio_spi1_ss_n_tri;

	generate
	if (C_USE_DIFF_RW_CLK_GP0 == 0) begin : clk_assign0
		assign saxihpc0_fpd_rclk_temp  =  saxihpc0_fpd_aclk;
		assign saxihpc0_fpd_wclk_temp  =  saxihpc0_fpd_aclk;
	end else begin
		assign saxihpc0_fpd_rclk_temp  =  saxihpc0_fpd_rclk;
		assign saxihpc0_fpd_wclk_temp  =  saxihpc0_fpd_wclk;
	end

	if (C_USE_DIFF_RW_CLK_GP1 == 0) begin : clk_assign1
		assign saxihpc1_fpd_rclk_temp  =  saxihpc1_fpd_aclk;
		assign saxihpc1_fpd_wclk_temp  =  saxihpc1_fpd_aclk;
	end else begin
		assign saxihpc1_fpd_rclk_temp  =  saxihpc1_fpd_rclk;
		assign saxihpc1_fpd_wclk_temp  =  saxihpc1_fpd_wclk;
	end

	if (C_USE_DIFF_RW_CLK_GP2 == 0) begin : clk_assign2
		assign saxihp0_fpd_rclk_temp  =  saxihp0_fpd_aclk;
		assign saxihp0_fpd_wclk_temp  =  saxihp0_fpd_aclk;
	end else begin
		assign saxihp0_fpd_rclk_temp  =  saxihp0_fpd_rclk;
		assign saxihp0_fpd_wclk_temp  =  saxihp0_fpd_wclk;
	end

	if (C_USE_DIFF_RW_CLK_GP3 == 0) begin : clk_assign3
		assign saxihp1_fpd_rclk_temp  =  saxihp1_fpd_aclk;
		assign saxihp1_fpd_wclk_temp  =  saxihp1_fpd_aclk;
	end else begin
		assign saxihp1_fpd_rclk_temp  =  saxihp1_fpd_rclk;
		assign saxihp1_fpd_wclk_temp  =  saxihp1_fpd_wclk;
	end

	if (C_USE_DIFF_RW_CLK_GP4 == 0) begin : clk_assign4
		assign saxihp2_fpd_rclk_temp  =  saxihp2_fpd_aclk;
		assign saxihp2_fpd_wclk_temp  =  saxihp2_fpd_aclk;
	end else begin
		assign saxihp2_fpd_rclk_temp  =  saxihp2_fpd_rclk;
		assign saxihp2_fpd_wclk_temp  =  saxihp2_fpd_wclk;
	end

	if (C_USE_DIFF_RW_CLK_GP5 == 0) begin : clk_assign5
		assign saxihp3_fpd_rclk_temp  =  saxihp3_fpd_aclk;
		assign saxihp3_fpd_wclk_temp  =  saxihp3_fpd_aclk;
	end else begin
		assign saxihp3_fpd_rclk_temp  =  saxihp3_fpd_rclk;
		assign saxihp3_fpd_wclk_temp  =  saxihp3_fpd_wclk;
	end

	if (C_USE_DIFF_RW_CLK_GP6 == 0) begin : clk_assign6
		assign saxi_lpd_rclk_temp  =  saxi_lpd_aclk;
		assign saxi_lpd_wclk_temp  =  saxi_lpd_aclk;
	end else begin
		assign saxi_lpd_rclk_temp  =  saxi_lpd_rclk;
		assign saxi_lpd_wclk_temp  =  saxi_lpd_wclk;
	end
	endgenerate

	generate
	if ( (C_PL_CLK0_BUF == "true")  | (C_PL_CLK0_BUF == "TRUE") | (C_PL_CLK0_BUF == 1) | (C_PL_CLK0_BUF == 'b1)) begin : buffer_pl_clk_0
		BUFG_PS PL_CLK_0_BUFG (.I(pl_clk_unbuffered[0]), .O(pl_clk_buffered[0]));
	end else begin
		assign pl_clk_buffered[0] = pl_clk_unbuffered[0];
	end if ( (C_PL_CLK1_BUF == "true") | (C_PL_CLK1_BUF == "TRUE") | (C_PL_CLK1_BUF == 1) | (C_PL_CLK1_BUF == 'b1)) begin : buffer_pl_clk_1
		BUFG_PS PL_CLK_1_BUFG (.I(pl_clk_unbuffered[1]), .O(pl_clk_buffered[1]));
	end else begin
		assign pl_clk_buffered[1] = pl_clk_unbuffered[1];
	end

	if ( (C_PL_CLK2_BUF == "true")  | (C_PL_CLK2_BUF == "TRUE") | (C_PL_CLK2_BUF == 1) | (C_PL_CLK2_BUF == 'b1))begin : buffer_pl_clk_2
		BUFG_PS PL_CLK_2_BUFG (.I(pl_clk_unbuffered[2]), .O(pl_clk_buffered[2]));
	end else begin
		assign pl_clk_buffered[2] = pl_clk_unbuffered[2];
	end

	if ( (C_PL_CLK3_BUF == "true") | (C_PL_CLK3_BUF == "TRUE") | (C_PL_CLK3_BUF == 1) | (C_PL_CLK3_BUF == 'b1)) begin : buffer_pl_clk_3
		BUFG_PS PL_CLK_3_BUFG (.I(pl_clk_unbuffered[3]), .O(pl_clk_buffered[3]));
	end else begin
		assign pl_clk_buffered[3] = pl_clk_unbuffered[3];
	end
	endgenerate

	// Assigning the F2P IRQ

	//input  wire [7:0] pl_ps_irq0,
	//input  wire [7:0] pl_ps_irq1,
	//output wire [99:0] ps_pl_irq_lpd,
	//output wire [63:0] ps_pl_irq_fpd,

	generate
	if (C_NUM_F2P_0_INTR_INPUTS == 0) begin : irq_f2p_0_select_null
		assign irq_f2p_0_i[7:0] = {irq_f2p_0_null[7:0]};
	end else if(C_NUM_F2P_0_INTR_INPUTS == 8) begin : irq_f2p_select_all
		assign irq_f2p_0_i[7:0] = {pl_ps_irq0[7:0]};
	end else begin : irq_f2p_select
		assign irq_f2p_0_i[7:0] = {irq_f2p_0_null[(7-C_NUM_F2P_0_INTR_INPUTS):0], pl_ps_irq0[(C_NUM_F2P_0_INTR_INPUTS-1):0]};
	end

	if (C_NUM_F2P_1_INTR_INPUTS == 0)  begin : irq_f2p_1_select_null
		assign irq_f2p_1_i[7:0] = {irq_f2p_1_null[7:0]};
	end else if(C_NUM_F2P_1_INTR_INPUTS == 8) begin : irq_f2p_select_all1
		assign irq_f2p_1_i[7:0] = {pl_ps_irq1[7:0]};
	end else begin : irq_f2p_select_1
		assign irq_f2p_1_i[7:0] = {irq_f2p_1_null[(7-C_NUM_F2P_1_INTR_INPUTS):0], pl_ps_irq1[(C_NUM_F2P_1_INTR_INPUTS-1):0]};
	end
	endgenerate

	// workaround for reset signals using gpio_o signals

	generate
	assign emio_gpio_o =  emio_gpio_o_temp[(C_EMIO_GPIO_WIDTH -1):0];
	assign emio_gpio_t =  ~emio_gpio_t_temp[(C_EMIO_GPIO_WIDTH -1):0];
	assign emio_gpio_t_n =  emio_gpio_t_temp[(C_EMIO_GPIO_WIDTH -1):0];
	assign emio_gpio_i_temp =  {{(96-C_EMIO_GPIO_WIDTH){1'b0}},emio_gpio_i};

	if(C_NUM_FABRIC_RESETS == 1) begin
		assign pl_resetn0 = emio_gpio_o_temp[95];
		assign pl_resetn1 = 1'b1;
		assign pl_resetn2 = 1'b1;
		assign pl_resetn3 = 1'b1;
	end else if(C_NUM_FABRIC_RESETS == 2) begin
		assign pl_resetn0 = emio_gpio_o_temp[95];
		assign pl_resetn1 = emio_gpio_o_temp[94];
		assign pl_resetn2 = 1'b1;
		assign pl_resetn3 = 1'b1;
	end else if(C_NUM_FABRIC_RESETS == 3) begin
		assign pl_resetn0 = emio_gpio_o_temp[95];
		assign pl_resetn1 = emio_gpio_o_temp[94];
		assign pl_resetn2 = emio_gpio_o_temp[93];
		assign pl_resetn3 = 1'b1;
	end else begin
		assign pl_resetn0 = emio_gpio_o_temp[95];
		assign pl_resetn1 = emio_gpio_o_temp[94];
		assign pl_resetn2 = emio_gpio_o_temp[93];
		assign pl_resetn3 = emio_gpio_o_temp[92];
	end
	endgenerate

	assign pl_clk0 = ((C_PL_CLK0_BUF == "true") | (C_PL_CLK0_BUF == "TRUE") | (C_PL_CLK0_BUF == 1) | (C_PL_CLK0_BUF == 'b1)) ? pl_clk_buffered[0] : pl_clk_unbuffered[0];
	assign pl_clk1 = ((C_PL_CLK1_BUF == "true") | (C_PL_CLK1_BUF == "TRUE") | (C_PL_CLK1_BUF == 1) | (C_PL_CLK1_BUF == 'b1)) ? pl_clk_buffered[1] : pl_clk_unbuffered[1];
	assign pl_clk2 = ((C_PL_CLK2_BUF == "true") | (C_PL_CLK2_BUF == "TRUE") | (C_PL_CLK2_BUF == 1) | (C_PL_CLK2_BUF == 'b1)) ? pl_clk_buffered[2] : pl_clk_unbuffered[2];
	assign pl_clk3 = ((C_PL_CLK3_BUF == "true") | (C_PL_CLK3_BUF == "TRUE") | (C_PL_CLK3_BUF == 1) | (C_PL_CLK3_BUF == 'b1)) ? pl_clk_buffered[3] : pl_clk_unbuffered[3];

	wire [C_TRACE_DATA_WIDTH-1:0] trace_data_i;
	wire trace_ctl_i;
	(* keep = "true" *) reg   trace_ctl_pipe [(C_TRACE_PIPELINE_WIDTH - 1):0];
	(* keep = "true" *) reg   [C_TRACE_DATA_WIDTH-1:0] trace_data_pipe [(C_TRACE_PIPELINE_WIDTH - 1):0];

	// maxigp0
	assign maxigp0_wdata = (C_MAXIGP0_DATA_WIDTH == 128) ? maxigp0_wdata_i : ( (C_MAXIGP0_DATA_WIDTH == 64) ? maxigp0_wdata_i[63:0] : maxigp0_wdata_i[31:0] );
	assign maxigp0_wstrb = (C_MAXIGP0_DATA_WIDTH > 0) ? maxigp0_wstrb_i[(C_MAXIGP0_DATA_WIDTH/8)-1:0] : maxigp0_wstrb_i[(32/8)-1:0];
	assign maxigp0_rdata_i = (C_MAXIGP0_DATA_WIDTH == 128) ? maxigp0_rdata : ((C_MAXIGP0_DATA_WIDTH == 64) ? {64'b0, maxigp0_rdata} : {96'b0, maxigp0_rdata});

	// maxigp1
	assign maxigp1_wdata = (C_MAXIGP1_DATA_WIDTH == 128) ? maxigp1_wdata_i : ( (C_MAXIGP1_DATA_WIDTH == 64) ? maxigp1_wdata_i[63:0] : maxigp1_wdata_i[31:0] );
	assign maxigp1_wstrb = (C_MAXIGP1_DATA_WIDTH > 0) ? maxigp1_wstrb_i[(C_MAXIGP1_DATA_WIDTH/8)-1:0] : maxigp1_wstrb_i[(32/8)-1:0];
	assign maxigp1_rdata_i = (C_MAXIGP1_DATA_WIDTH == 128) ? maxigp1_rdata : ((C_MAXIGP1_DATA_WIDTH == 64) ? {64'b0, maxigp1_rdata} : {96'b0, maxigp1_rdata});

	// maxigp2
	assign maxigp2_wdata = (C_MAXIGP2_DATA_WIDTH == 128) ? maxigp2_wdata_i : ( (C_MAXIGP2_DATA_WIDTH == 64) ? maxigp2_wdata_i[63:0] : maxigp2_wdata_i[31:0] );
	assign maxigp2_wstrb = (C_MAXIGP2_DATA_WIDTH > 0) ? maxigp2_wstrb_i[(C_MAXIGP2_DATA_WIDTH/8)-1:0] : maxigp2_wstrb_i[(32/8)-1:0];
	assign maxigp2_rdata_i = (C_MAXIGP2_DATA_WIDTH == 128) ? maxigp2_rdata : ((C_MAXIGP2_DATA_WIDTH == 64) ? {64'b0, maxigp2_rdata} : {96'b0, maxigp2_rdata});

	// saxigp0
	assign saxigp0_wdata_i = (C_SAXIGP0_DATA_WIDTH == 128) ? saxigp0_wdata : ((C_SAXIGP0_DATA_WIDTH == 64) ? {64'b0, saxigp0_wdata} : {96'b0, saxigp0_wdata});
	assign saxigp0_wstrb_i = (C_SAXIGP0_DATA_WIDTH == 128) ? saxigp0_wstrb : ((C_SAXIGP0_DATA_WIDTH == 64) ? {8'b0, saxigp0_wstrb} : {12'b0, saxigp0_wstrb});
	assign saxigp0_rdata = (C_SAXIGP0_DATA_WIDTH == 128) ? saxigp0_rdata_i : ( (C_SAXIGP0_DATA_WIDTH == 64) ? saxigp0_rdata_i[63:0] : saxigp0_rdata_i[31:0] );

	// saxigp1
	assign saxigp1_wdata_i = (C_SAXIGP1_DATA_WIDTH == 128) ? saxigp1_wdata : ((C_SAXIGP1_DATA_WIDTH == 64) ? {64'b0, saxigp1_wdata} : {96'b0, saxigp1_wdata});
	assign saxigp1_wstrb_i = (C_SAXIGP1_DATA_WIDTH == 128) ? saxigp1_wstrb : ((C_SAXIGP1_DATA_WIDTH == 64) ? {8'b0, saxigp1_wstrb} : {12'b0, saxigp1_wstrb});
	assign saxigp1_rdata = (C_SAXIGP1_DATA_WIDTH == 128) ? saxigp1_rdata_i : ( (C_SAXIGP1_DATA_WIDTH == 64) ? saxigp1_rdata_i[63:0] : saxigp1_rdata_i[31:0] );

	// saxigp2
	assign saxigp2_wdata_i = (C_SAXIGP2_DATA_WIDTH == 128) ? saxigp2_wdata : ((C_SAXIGP2_DATA_WIDTH == 64) ? {64'b0, saxigp2_wdata} : {96'b0, saxigp2_wdata});
	assign saxigp2_wstrb_i = (C_SAXIGP2_DATA_WIDTH == 128) ? saxigp2_wstrb : ((C_SAXIGP2_DATA_WIDTH == 64) ? {8'b0, saxigp2_wstrb} : {12'b0, saxigp2_wstrb});
	assign saxigp2_rdata = (C_SAXIGP2_DATA_WIDTH == 128) ? saxigp2_rdata_i : ( (C_SAXIGP2_DATA_WIDTH == 64) ? saxigp2_rdata_i[63:0] : saxigp2_rdata_i[31:0] );

	// saxigp3
	assign saxigp3_wdata_i = (C_SAXIGP3_DATA_WIDTH == 128) ? saxigp3_wdata : ((C_SAXIGP3_DATA_WIDTH == 64) ? {64'b0, saxigp3_wdata} : {96'b0, saxigp3_wdata});
	assign saxigp3_wstrb_i = (C_SAXIGP3_DATA_WIDTH == 128) ? saxigp3_wstrb : ((C_SAXIGP3_DATA_WIDTH == 64) ? {8'b0, saxigp3_wstrb} : {12'b0, saxigp3_wstrb});
	assign saxigp3_rdata = (C_SAXIGP3_DATA_WIDTH == 128) ? saxigp3_rdata_i : ( (C_SAXIGP3_DATA_WIDTH == 64) ? saxigp3_rdata_i[63:0] : saxigp3_rdata_i[31:0] );

	// saxigp4
	assign saxigp4_wdata_i = (C_SAXIGP4_DATA_WIDTH == 128) ? saxigp4_wdata : ((C_SAXIGP4_DATA_WIDTH == 64) ? {64'b0, saxigp4_wdata} : {96'b0, saxigp4_wdata});
	assign saxigp4_wstrb_i = (C_SAXIGP4_DATA_WIDTH == 128) ? saxigp4_wstrb : ((C_SAXIGP4_DATA_WIDTH == 64) ? {8'b0, saxigp4_wstrb} : {12'b0, saxigp4_wstrb});
	assign saxigp4_rdata = (C_SAXIGP4_DATA_WIDTH == 128) ? saxigp4_rdata_i : ( (C_SAXIGP4_DATA_WIDTH == 64) ? saxigp4_rdata_i[63:0] : saxigp4_rdata_i[31:0] );

	// saxigp5
	assign saxigp5_wdata_i = (C_SAXIGP5_DATA_WIDTH == 128) ? saxigp5_wdata : ((C_SAXIGP5_DATA_WIDTH == 64) ? {64'b0, saxigp5_wdata} : {96'b0, saxigp5_wdata});
	assign saxigp5_wstrb_i = (C_SAXIGP5_DATA_WIDTH == 128) ? saxigp5_wstrb : ((C_SAXIGP5_DATA_WIDTH == 64) ? {8'b0, saxigp5_wstrb} : {12'b0, saxigp5_wstrb});
	assign saxigp5_rdata = (C_SAXIGP5_DATA_WIDTH == 128) ? saxigp5_rdata_i : ( (C_SAXIGP5_DATA_WIDTH == 64) ? saxigp5_rdata_i[63:0] : saxigp5_rdata_i[31:0] );

	// saxigp6
	assign saxigp6_wdata_i = (C_SAXIGP6_DATA_WIDTH == 128) ? saxigp6_wdata : ((C_SAXIGP6_DATA_WIDTH == 64) ? {64'b0, saxigp6_wdata} : {96'b0, saxigp6_wdata});
	assign saxigp6_wstrb_i = (C_SAXIGP6_DATA_WIDTH == 128) ? saxigp6_wstrb : ((C_SAXIGP6_DATA_WIDTH == 64) ? {8'b0, saxigp6_wstrb} : {12'b0, saxigp6_wstrb});
	assign saxigp6_rdata = (C_SAXIGP6_DATA_WIDTH == 128) ? saxigp6_rdata_i : ( (C_SAXIGP6_DATA_WIDTH == 64) ? saxigp6_rdata_i[63:0] : saxigp6_rdata_i[31:0] );

	// Trace Pipeline and clk out
	integer j;
	generate
	if (C_EN_EMIO_TRACE == 1) begin //Enable EMIO Trace
		always @(posedge pl_ps_trace_clk) begin
			//Add date to end of the pipeline
			trace_ctl_pipe[C_TRACE_PIPELINE_WIDTH - 1] <= trace_ctl_i;
			trace_data_pipe[C_TRACE_PIPELINE_WIDTH - 1] <= trace_data_i;
			for (j=(C_TRACE_PIPELINE_WIDTH-1); j>0; j=j-1) begin
				trace_ctl_pipe[j-1] <= trace_ctl_pipe[j];
				trace_data_pipe[j-1] <= trace_data_pipe[j];
			end
			//Create Divide by two clock
			trace_clk_out <= ~trace_clk_out;
		end
	end else begin
		always @(posedge pl_ps_trace_clk) begin
			trace_clk_out <= ~trace_clk_out;
		end
	end
	endgenerate

	assign ps_pl_tracectl = trace_ctl_pipe[0];
	assign ps_pl_tracedata = trace_data_pipe[0];

	generate
	if (C_DP_USE_VIDEO == 0) begin
		assign dp_video_ref_clk = dp_video_ref_clk_i;
	end else begin
		BUFG_PS DP_VID_CLK_BUFG (.I(dp_video_ref_clk_i), .O(dp_video_ref_clk));
	end
	endgenerate

	generate
	if(C_DP_USE_AUDIO == 0) begin
		assign dp_audi_ref_clk = dp_audio_ref_clk_i;
	end else begin
		BUFG_PS DP_AUD_CLK_BUFG(.I(dp_audio_ref_clk_i), .O(dp_audio_ref_clk));
	end
	endgenerate

	generate
	if (C_USE_DEBUG_TEST == 1) begin
		PS8_TEST PS8_TEST_i(
			.MAXIGP0ACLK(maxihpm0_fpd_aclk), .MAXIGP0AWID(maxigp0_awid), .MAXIGP0AWADDR(maxigp0_awaddr), .MAXIGP0AWLEN(maxigp0_awlen), .MAXIGP0AWSIZE(maxigp0_awsize), .MAXIGP0AWBURST(maxigp0_awburst), .MAXIGP0AWLOCK(maxigp0_awlock), .MAXIGP0AWCACHE(maxigp0_awcache), .MAXIGP0AWPROT(maxigp0_awprot), .MAXIGP0AWVALID(maxigp0_awvalid), .MAXIGP0AWUSER(maxigp0_awuser), .MAXIGP0AWREADY(maxigp0_awready), .MAXIGP0WDATA(maxigp0_wdata_i), .MAXIGP0WSTRB(maxigp0_wstrb_i), .MAXIGP0WLAST(maxigp0_wlast), .MAXIGP0WVALID(maxigp0_wvalid), .MAXIGP0WREADY(maxigp0_wready), .MAXIGP0BID(maxigp0_bid), .MAXIGP0BRESP(maxigp0_bresp), .MAXIGP0BVALID(maxigp0_bvalid), .MAXIGP0BREADY(maxigp0_bready), .MAXIGP0ARID(maxigp0_arid), .MAXIGP0ARADDR(maxigp0_araddr), .MAXIGP0ARLEN(maxigp0_arlen), .MAXIGP0ARSIZE(maxigp0_arsize), .MAXIGP0ARBURST(maxigp0_arburst), .MAXIGP0ARLOCK(maxigp0_arlock), .MAXIGP0ARCACHE(maxigp0_arcache), .MAXIGP0ARPROT(maxigp0_arprot), .MAXIGP0ARVALID(maxigp0_arvalid), .MAXIGP0ARUSER(maxigp0_aruser), .MAXIGP0ARREADY(maxigp0_arready), .MAXIGP0RID(maxigp0_rid), .MAXIGP0RDATA(maxigp0_rdata_i), .MAXIGP0RRESP(maxigp0_rresp), .MAXIGP0RLAST(maxigp0_rlast), .MAXIGP0RVALID(maxigp0_rvalid), .MAXIGP0RREADY(maxigp0_rready), .MAXIGP0AWQOS(maxigp0_awqos), .MAXIGP0ARQOS(maxigp0_arqos),

			.MAXIGP1ACLK(maxihpm1_fpd_aclk), .MAXIGP1AWID(maxigp1_awid), .MAXIGP1AWADDR(maxigp1_awaddr), .MAXIGP1AWLEN(maxigp1_awlen), .MAXIGP1AWSIZE(maxigp1_awsize), .MAXIGP1AWBURST(maxigp1_awburst), .MAXIGP1AWLOCK(maxigp1_awlock), .MAXIGP1AWCACHE(maxigp1_awcache), .MAXIGP1AWPROT(maxigp1_awprot), .MAXIGP1AWVALID(maxigp1_awvalid), .MAXIGP1AWUSER(maxigp1_awuser), .MAXIGP1AWREADY(maxigp1_awready), .MAXIGP1WDATA(maxigp1_wdata_i), .MAXIGP1WSTRB(maxigp1_wstrb_i), .MAXIGP1WLAST(maxigp1_wlast), .MAXIGP1WVALID(maxigp1_wvalid), .MAXIGP1WREADY(maxigp1_wready), .MAXIGP1BID(maxigp1_bid), .MAXIGP1BRESP(maxigp1_bresp), .MAXIGP1BVALID(maxigp1_bvalid), .MAXIGP1BREADY(maxigp1_bready), .MAXIGP1ARID(maxigp1_arid), .MAXIGP1ARADDR(maxigp1_araddr), .MAXIGP1ARLEN(maxigp1_arlen), .MAXIGP1ARSIZE(maxigp1_arsize), .MAXIGP1ARBURST(maxigp1_arburst), .MAXIGP1ARLOCK(maxigp1_arlock), .MAXIGP1ARCACHE(maxigp1_arcache), .MAXIGP1ARPROT(maxigp1_arprot), .MAXIGP1ARVALID(maxigp1_arvalid), .MAXIGP1ARUSER(maxigp1_aruser), .MAXIGP1ARREADY(maxigp1_arready), .MAXIGP1RID(maxigp1_rid), .MAXIGP1RDATA(maxigp1_rdata_i), .MAXIGP1RRESP(maxigp1_rresp), .MAXIGP1RLAST(maxigp1_rlast), .MAXIGP1RVALID(maxigp1_rvalid), .MAXIGP1RREADY(maxigp1_rready), .MAXIGP1AWQOS(maxigp1_awqos), .MAXIGP1ARQOS(maxigp1_arqos),

			.MAXIGP2ACLK(maxihpm0_lpd_aclk), .MAXIGP2AWID(maxigp2_awid), .MAXIGP2AWADDR(maxigp2_awaddr), .MAXIGP2AWLEN(maxigp2_awlen), .MAXIGP2AWSIZE(maxigp2_awsize), .MAXIGP2AWBURST(maxigp2_awburst), .MAXIGP2AWLOCK(maxigp2_awlock), .MAXIGP2AWCACHE(maxigp2_awcache), .MAXIGP2AWPROT(maxigp2_awprot), .MAXIGP2AWVALID(maxigp2_awvalid), .MAXIGP2AWUSER(maxigp2_awuser), .MAXIGP2AWREADY(maxigp2_awready), .MAXIGP2WDATA(maxigp2_wdata_i), .MAXIGP2WSTRB(maxigp2_wstrb_i), .MAXIGP2WLAST(maxigp2_wlast), .MAXIGP2WVALID(maxigp2_wvalid), .MAXIGP2WREADY(maxigp2_wready), .MAXIGP2BID(maxigp2_bid), .MAXIGP2BRESP(maxigp2_bresp), .MAXIGP2BVALID(maxigp2_bvalid), .MAXIGP2BREADY(maxigp2_bready), .MAXIGP2ARID(maxigp2_arid), .MAXIGP2ARADDR(maxigp2_araddr), .MAXIGP2ARLEN(maxigp2_arlen), .MAXIGP2ARSIZE(maxigp2_arsize), .MAXIGP2ARBURST(maxigp2_arburst), .MAXIGP2ARLOCK(maxigp2_arlock), .MAXIGP2ARCACHE(maxigp2_arcache), .MAXIGP2ARPROT(maxigp2_arprot), .MAXIGP2ARVALID(maxigp2_arvalid), .MAXIGP2ARUSER(maxigp2_aruser), .MAXIGP2ARREADY(maxigp2_arready), .MAXIGP2RID(maxigp2_rid), .MAXIGP2RDATA(maxigp2_rdata_i), .MAXIGP2RRESP(maxigp2_rresp), .MAXIGP2RLAST(maxigp2_rlast), .MAXIGP2RVALID(maxigp2_rvalid), .MAXIGP2RREADY(maxigp2_rready), .MAXIGP2AWQOS(maxigp2_awqos), .MAXIGP2ARQOS(maxigp2_arqos),

			.SAXIGP0RCLK(saxihpc0_fpd_rclk_temp), .SAXIGP0WCLK(saxihpc0_fpd_wclk_temp), .SAXIGP0ARUSER(saxigp0_aruser), .SAXIGP0AWUSER(saxigp0_awuser), .SAXIGP0AWID(saxigp0_awid), .SAXIGP0AWADDR(saxigp0_awaddr), .SAXIGP0AWLEN(saxigp0_awlen), .SAXIGP0AWSIZE(saxigp0_awsize), .SAXIGP0AWBURST(saxigp0_awburst), .SAXIGP0AWLOCK(saxigp0_awlock), .SAXIGP0AWCACHE(saxigp0_awcache), .SAXIGP0AWPROT(saxigp0_awprot), .SAXIGP0AWVALID(saxigp0_awvalid), .SAXIGP0AWREADY(saxigp0_awready), .SAXIGP0WDATA(saxigp0_wdata_i), .SAXIGP0WSTRB(saxigp0_wstrb_i), .SAXIGP0WLAST(saxigp0_wlast), .SAXIGP0WVALID(saxigp0_wvalid), .SAXIGP0WREADY(saxigp0_wready), .SAXIGP0BID(saxigp0_bid), .SAXIGP0BRESP(saxigp0_bresp), .SAXIGP0BVALID(saxigp0_bvalid), .SAXIGP0BREADY(saxigp0_bready), .SAXIGP0ARID(saxigp0_arid), .SAXIGP0ARADDR(saxigp0_araddr), .SAXIGP0ARLEN(saxigp0_arlen), .SAXIGP0ARSIZE(saxigp0_arsize), .SAXIGP0ARBURST(saxigp0_arburst), .SAXIGP0ARLOCK(saxigp0_arlock), .SAXIGP0ARCACHE(saxigp0_arcache), .SAXIGP0ARPROT(saxigp0_arprot), .SAXIGP0ARVALID(saxigp0_arvalid), .SAXIGP0ARREADY(saxigp0_arready), .SAXIGP0RID(saxigp0_rid), .SAXIGP0RDATA(saxigp0_rdata_i), .SAXIGP0RRESP(saxigp0_rresp), .SAXIGP0RLAST(saxigp0_rlast), .SAXIGP0RVALID(saxigp0_rvalid), .SAXIGP0RREADY(saxigp0_rready), .SAXIGP0AWQOS(saxigp0_awqos), .SAXIGP0ARQOS(saxigp0_arqos), .SAXIGP0RCOUNT(saxigp0_rcount), .SAXIGP0WCOUNT(saxigp0_wcount), .SAXIGP0RACOUNT(saxigp0_racount), .SAXIGP0WACOUNT(saxigp0_wacount),

			.SAXIGP1RCLK(saxihpc1_fpd_rclk_temp), .SAXIGP1WCLK(saxihpc1_fpd_wclk_temp), .SAXIGP1ARUSER(saxigp1_aruser), .SAXIGP1AWUSER(saxigp1_awuser), .SAXIGP1AWID(saxigp1_awid), .SAXIGP1AWADDR(saxigp1_awaddr), .SAXIGP1AWLEN(saxigp1_awlen), .SAXIGP1AWSIZE(saxigp1_awsize), .SAXIGP1AWBURST(saxigp1_awburst), .SAXIGP1AWLOCK(saxigp1_awlock), .SAXIGP1AWCACHE(saxigp1_awcache), .SAXIGP1AWPROT(saxigp1_awprot), .SAXIGP1AWVALID(saxigp1_awvalid), .SAXIGP1AWREADY(saxigp1_awready), .SAXIGP1WDATA(saxigp1_wdata_i), .SAXIGP1WSTRB(saxigp1_wstrb_i), .SAXIGP1WLAST(saxigp1_wlast), .SAXIGP1WVALID(saxigp1_wvalid), .SAXIGP1WREADY(saxigp1_wready), .SAXIGP1BID(saxigp1_bid), .SAXIGP1BRESP(saxigp1_bresp), .SAXIGP1BVALID(saxigp1_bvalid), .SAXIGP1BREADY(saxigp1_bready), .SAXIGP1ARID(saxigp1_arid), .SAXIGP1ARADDR(saxigp1_araddr), .SAXIGP1ARLEN(saxigp1_arlen), .SAXIGP1ARSIZE(saxigp1_arsize), .SAXIGP1ARBURST(saxigp1_arburst), .SAXIGP1ARLOCK(saxigp1_arlock), .SAXIGP1ARCACHE(saxigp1_arcache), .SAXIGP1ARPROT(saxigp1_arprot), .SAXIGP1ARVALID(saxigp1_arvalid), .SAXIGP1ARREADY(saxigp1_arready), .SAXIGP1RID(saxigp1_rid), .SAXIGP1RDATA(saxigp1_rdata_i), .SAXIGP1RRESP(saxigp1_rresp), .SAXIGP1RLAST(saxigp1_rlast), .SAXIGP1RVALID(saxigp1_rvalid), .SAXIGP1RREADY(saxigp1_rready), .SAXIGP1AWQOS(saxigp1_awqos), .SAXIGP1ARQOS(saxigp1_arqos), .SAXIGP1RCOUNT(saxigp1_rcount), .SAXIGP1WCOUNT(saxigp1_wcount), .SAXIGP1RACOUNT(saxigp1_racount), .SAXIGP1WACOUNT(saxigp1_wacount),

			.SAXIGP2RCLK(saxihp0_fpd_rclk_temp), .SAXIGP2WCLK(saxihp0_fpd_wclk_temp), .SAXIGP2ARUSER(saxigp2_aruser), .SAXIGP2AWUSER(saxigp2_awuser), .SAXIGP2AWID(saxigp2_awid), .SAXIGP2AWADDR(saxigp2_awaddr), .SAXIGP2AWLEN(saxigp2_awlen), .SAXIGP2AWSIZE(saxigp2_awsize), .SAXIGP2AWBURST(saxigp2_awburst), .SAXIGP2AWLOCK(saxigp2_awlock), .SAXIGP2AWCACHE(saxigp2_awcache), .SAXIGP2AWPROT(saxigp2_awprot), .SAXIGP2AWVALID(saxigp2_awvalid), .SAXIGP2AWREADY(saxigp2_awready), .SAXIGP2WDATA(saxigp2_wdata_i), .SAXIGP2WSTRB(saxigp2_wstrb_i), .SAXIGP2WLAST(saxigp2_wlast), .SAXIGP2WVALID(saxigp2_wvalid), .SAXIGP2WREADY(saxigp2_wready), .SAXIGP2BID(saxigp2_bid), .SAXIGP2BRESP(saxigp2_bresp), .SAXIGP2BVALID(saxigp2_bvalid), .SAXIGP2BREADY(saxigp2_bready), .SAXIGP2ARID(saxigp2_arid), .SAXIGP2ARADDR(saxigp2_araddr), .SAXIGP2ARLEN(saxigp2_arlen), .SAXIGP2ARSIZE(saxigp2_arsize), .SAXIGP2ARBURST(saxigp2_arburst), .SAXIGP2ARLOCK(saxigp2_arlock), .SAXIGP2ARCACHE(saxigp2_arcache), .SAXIGP2ARPROT(saxigp2_arprot), .SAXIGP2ARVALID(saxigp2_arvalid), .SAXIGP2ARREADY(saxigp2_arready), .SAXIGP2RID(saxigp2_rid), .SAXIGP2RDATA(saxigp2_rdata_i), .SAXIGP2RRESP(saxigp2_rresp), .SAXIGP2RLAST(saxigp2_rlast), .SAXIGP2RVALID(saxigp2_rvalid), .SAXIGP2RREADY(saxigp2_rready), .SAXIGP2AWQOS(saxigp2_awqos), .SAXIGP2ARQOS(saxigp2_arqos), .SAXIGP2RCOUNT(saxigp2_rcount), .SAXIGP2WCOUNT(saxigp2_wcount), .SAXIGP2RACOUNT(saxigp2_racount), .SAXIGP2WACOUNT(saxigp2_wacount),

			.SAXIGP3RCLK(saxihp1_fpd_rclk_temp), .SAXIGP3WCLK(saxihp1_fpd_wclk_temp), .SAXIGP3ARUSER(saxigp3_aruser), .SAXIGP3AWUSER(saxigp3_awuser), .SAXIGP3AWID(saxigp3_awid), .SAXIGP3AWADDR(saxigp3_awaddr), .SAXIGP3AWLEN(saxigp3_awlen), .SAXIGP3AWSIZE(saxigp3_awsize), .SAXIGP3AWBURST(saxigp3_awburst), .SAXIGP3AWLOCK(saxigp3_awlock), .SAXIGP3AWCACHE(saxigp3_awcache), .SAXIGP3AWPROT(saxigp3_awprot), .SAXIGP3AWVALID(saxigp3_awvalid), .SAXIGP3AWREADY(saxigp3_awready), .SAXIGP3WDATA(saxigp3_wdata_i), .SAXIGP3WSTRB(saxigp3_wstrb_i), .SAXIGP3WLAST(saxigp3_wlast), .SAXIGP3WVALID(saxigp3_wvalid), .SAXIGP3WREADY(saxigp3_wready), .SAXIGP3BID(saxigp3_bid), .SAXIGP3BRESP(saxigp3_bresp), .SAXIGP3BVALID(saxigp3_bvalid), .SAXIGP3BREADY(saxigp3_bready), .SAXIGP3ARID(saxigp3_arid), .SAXIGP3ARADDR(saxigp3_araddr), .SAXIGP3ARLEN(saxigp3_arlen), .SAXIGP3ARSIZE(saxigp3_arsize), .SAXIGP3ARBURST(saxigp3_arburst), .SAXIGP3ARLOCK(saxigp3_arlock), .SAXIGP3ARCACHE(saxigp3_arcache), .SAXIGP3ARPROT(saxigp3_arprot), .SAXIGP3ARVALID(saxigp3_arvalid), .SAXIGP3ARREADY(saxigp3_arready), .SAXIGP3RID(saxigp3_rid), .SAXIGP3RDATA(saxigp3_rdata_i), .SAXIGP3RRESP(saxigp3_rresp), .SAXIGP3RLAST(saxigp3_rlast), .SAXIGP3RVALID(saxigp3_rvalid), .SAXIGP3RREADY(saxigp3_rready), .SAXIGP3AWQOS(saxigp3_awqos), .SAXIGP3ARQOS(saxigp3_arqos), .SAXIGP3RCOUNT(saxigp3_rcount), .SAXIGP3WCOUNT(saxigp3_wcount), .SAXIGP3RACOUNT(saxigp3_racount), .SAXIGP3WACOUNT(saxigp3_wacount),

			.SAXIGP4RCLK(saxihp2_fpd_rclk_temp), .SAXIGP4WCLK(saxihp2_fpd_wclk_temp), .SAXIGP4ARUSER(saxigp4_aruser), .SAXIGP4AWUSER(saxigp4_awuser), .SAXIGP4AWID(saxigp4_awid), .SAXIGP4AWADDR(saxigp4_awaddr), .SAXIGP4AWLEN(saxigp4_awlen), .SAXIGP4AWSIZE(saxigp4_awsize), .SAXIGP4AWBURST(saxigp4_awburst), .SAXIGP4AWLOCK(saxigp4_awlock), .SAXIGP4AWCACHE(saxigp4_awcache), .SAXIGP4AWPROT(saxigp4_awprot), .SAXIGP4AWVALID(saxigp4_awvalid), .SAXIGP4AWREADY(saxigp4_awready), .SAXIGP4WDATA(saxigp4_wdata_i), .SAXIGP4WSTRB(saxigp4_wstrb_i), .SAXIGP4WLAST(saxigp4_wlast), .SAXIGP4WVALID(saxigp4_wvalid), .SAXIGP4WREADY(saxigp4_wready), .SAXIGP4BID(saxigp4_bid), .SAXIGP4BRESP(saxigp4_bresp), .SAXIGP4BVALID(saxigp4_bvalid), .SAXIGP4BREADY(saxigp4_bready), .SAXIGP4ARID(saxigp4_arid), .SAXIGP4ARADDR(saxigp4_araddr), .SAXIGP4ARLEN(saxigp4_arlen), .SAXIGP4ARSIZE(saxigp4_arsize), .SAXIGP4ARBURST(saxigp4_arburst), .SAXIGP4ARLOCK(saxigp4_arlock), .SAXIGP4ARCACHE(saxigp4_arcache), .SAXIGP4ARPROT(saxigp4_arprot), .SAXIGP4ARVALID(saxigp4_arvalid), .SAXIGP4ARREADY(saxigp4_arready), .SAXIGP4RID(saxigp4_rid), .SAXIGP4RDATA(saxigp4_rdata_i), .SAXIGP4RRESP(saxigp4_rresp), .SAXIGP4RLAST(saxigp4_rlast), .SAXIGP4RVALID(saxigp4_rvalid), .SAXIGP4RREADY(saxigp4_rready), .SAXIGP4AWQOS(saxigp4_awqos), .SAXIGP4ARQOS(saxigp4_arqos), .SAXIGP4RCOUNT(saxigp4_rcount), .SAXIGP4WCOUNT(saxigp4_wcount), .SAXIGP4RACOUNT(saxigp4_racount), .SAXIGP4WACOUNT(saxigp4_wacount),

			.SAXIGP5RCLK(saxihp3_fpd_rclk_temp), .SAXIGP5WCLK(saxihp3_fpd_wclk_temp), .SAXIGP5ARUSER(saxigp5_aruser), .SAXIGP5AWUSER(saxigp5_awuser), .SAXIGP5AWID(saxigp5_awid), .SAXIGP5AWADDR(saxigp5_awaddr), .SAXIGP5AWLEN(saxigp5_awlen), .SAXIGP5AWSIZE(saxigp5_awsize), .SAXIGP5AWBURST(saxigp5_awburst), .SAXIGP5AWLOCK(saxigp5_awlock), .SAXIGP5AWCACHE(saxigp5_awcache), .SAXIGP5AWPROT(saxigp5_awprot), .SAXIGP5AWVALID(saxigp5_awvalid), .SAXIGP5AWREADY(saxigp5_awready), .SAXIGP5WDATA(saxigp5_wdata_i), .SAXIGP5WSTRB(saxigp5_wstrb_i), .SAXIGP5WLAST(saxigp5_wlast), .SAXIGP5WVALID(saxigp5_wvalid), .SAXIGP5WREADY(saxigp5_wready), .SAXIGP5BID(saxigp5_bid), .SAXIGP5BRESP(saxigp5_bresp), .SAXIGP5BVALID(saxigp5_bvalid), .SAXIGP5BREADY(saxigp5_bready), .SAXIGP5ARID(saxigp5_arid), .SAXIGP5ARADDR(saxigp5_araddr), .SAXIGP5ARLEN(saxigp5_arlen), .SAXIGP5ARSIZE(saxigp5_arsize), .SAXIGP5ARBURST(saxigp5_arburst), .SAXIGP5ARLOCK(saxigp5_arlock), .SAXIGP5ARCACHE(saxigp5_arcache), .SAXIGP5ARPROT(saxigp5_arprot), .SAXIGP5ARVALID(saxigp5_arvalid), .SAXIGP5ARREADY(saxigp5_arready), .SAXIGP5RID(saxigp5_rid), .SAXIGP5RDATA(saxigp5_rdata_i), .SAXIGP5RRESP(saxigp5_rresp), .SAXIGP5RLAST(saxigp5_rlast), .SAXIGP5RVALID(saxigp5_rvalid), .SAXIGP5RREADY(saxigp5_rready), .SAXIGP5AWQOS(saxigp5_awqos), .SAXIGP5ARQOS(saxigp5_arqos), .SAXIGP5RCOUNT(saxigp5_rcount), .SAXIGP5WCOUNT(saxigp5_wcount), .SAXIGP5RACOUNT(saxigp5_racount), .SAXIGP5WACOUNT(saxigp5_wacount),

			.SAXIGP6RCLK(saxi_lpd_rclk_temp), .SAXIGP6WCLK(saxi_lpd_wclk_temp), .SAXIGP6ARUSER(saxigp6_aruser), .SAXIGP6AWUSER(saxigp6_awuser), .SAXIGP6AWID(saxigp6_awid), .SAXIGP6AWADDR(saxigp6_awaddr), .SAXIGP6AWLEN(saxigp6_awlen), .SAXIGP6AWSIZE(saxigp6_awsize), .SAXIGP6AWBURST(saxigp6_awburst), .SAXIGP6AWLOCK(saxigp6_awlock), .SAXIGP6AWCACHE(saxigp6_awcache), .SAXIGP6AWPROT(saxigp6_awprot), .SAXIGP6AWVALID(saxigp6_awvalid), .SAXIGP6AWREADY(saxigp6_awready), .SAXIGP6WDATA(saxigp6_wdata_i), .SAXIGP6WSTRB(saxigp6_wstrb_i), .SAXIGP6WLAST(saxigp6_wlast), .SAXIGP6WVALID(saxigp6_wvalid), .SAXIGP6WREADY(saxigp6_wready), .SAXIGP6BID(saxigp6_bid), .SAXIGP6BRESP(saxigp6_bresp), .SAXIGP6BVALID(saxigp6_bvalid), .SAXIGP6BREADY(saxigp6_bready), .SAXIGP6ARID(saxigp6_arid), .SAXIGP6ARADDR(saxigp6_araddr), .SAXIGP6ARLEN(saxigp6_arlen), .SAXIGP6ARSIZE(saxigp6_arsize), .SAXIGP6ARBURST(saxigp6_arburst), .SAXIGP6ARLOCK(saxigp6_arlock), .SAXIGP6ARCACHE(saxigp6_arcache), .SAXIGP6ARPROT(saxigp6_arprot), .SAXIGP6ARVALID(saxigp6_arvalid), .SAXIGP6ARREADY(saxigp6_arready), .SAXIGP6RID(saxigp6_rid), .SAXIGP6RDATA(saxigp6_rdata_i), .SAXIGP6RRESP(saxigp6_rresp), .SAXIGP6RLAST(saxigp6_rlast), .SAXIGP6RVALID(saxigp6_rvalid), .SAXIGP6RREADY(saxigp6_rready), .SAXIGP6AWQOS(saxigp6_awqos), .SAXIGP6ARQOS(saxigp6_arqos), .SAXIGP6RCOUNT(saxigp6_rcount), .SAXIGP6WCOUNT(saxigp6_wcount), .SAXIGP6RACOUNT(saxigp6_racount), .SAXIGP6WACOUNT(saxigp6_wacount),

			.SAXIACPACLK(saxiacp_fpd_aclk), .SAXIACPAWADDR(saxiacp_awaddr), .SAXIACPAWID(saxiacp_awid), .SAXIACPAWLEN(saxiacp_awlen), .SAXIACPAWSIZE(saxiacp_awsize), .SAXIACPAWBURST(saxiacp_awburst), .SAXIACPAWLOCK(saxiacp_awlock), .SAXIACPAWCACHE(saxiacp_awcache), .SAXIACPAWPROT(saxiacp_awprot), .SAXIACPAWVALID(saxiacp_awvalid), .SAXIACPAWREADY(saxiacp_awready), .SAXIACPAWUSER(saxiacp_awuser), .SAXIACPAWQOS(saxiacp_awqos), .SAXIACPWLAST(saxiacp_wlast), .SAXIACPWDATA(saxiacp_wdata), .SAXIACPWSTRB(saxiacp_wstrb), .SAXIACPWVALID(saxiacp_wvalid), .SAXIACPWREADY(saxiacp_wready), .SAXIACPBRESP(saxiacp_bresp), .SAXIACPBID(saxiacp_bid), .SAXIACPBVALID(saxiacp_bvalid), .SAXIACPBREADY(saxiacp_bready), .SAXIACPARADDR(saxiacp_araddr), .SAXIACPARID(saxiacp_arid), .SAXIACPARLEN(saxiacp_arlen), .SAXIACPARSIZE(saxiacp_arsize), .SAXIACPARBURST(saxiacp_arburst), .SAXIACPARLOCK(saxiacp_arlock), .SAXIACPARCACHE(saxiacp_arcache), .SAXIACPARPROT(saxiacp_arprot), .SAXIACPARVALID(saxiacp_arvalid), .SAXIACPARREADY(saxiacp_arready), .SAXIACPARUSER(saxiacp_aruser), .SAXIACPARQOS(saxiacp_arqos), .SAXIACPRID(saxiacp_rid), .SAXIACPRLAST(saxiacp_rlast), .SAXIACPRDATA(saxiacp_rdata), .SAXIACPRRESP(saxiacp_rresp), .SAXIACPRVALID(saxiacp_rvalid), .SAXIACPRREADY(saxiacp_rready),

			.PLACECLK (sacefpd_aclk),

			.SACEFPDAWVALID(sacefpd_awvalid), .SACEFPDAWREADY(sacefpd_awready), .SACEFPDAWID(sacefpd_awid), .SACEFPDAWADDR(sacefpd_awaddr), .SACEFPDAWREGION(sacefpd_awregion), .SACEFPDAWLEN(sacefpd_awlen), .SACEFPDAWSIZE(sacefpd_awsize), .SACEFPDAWBURST(sacefpd_awburst), .SACEFPDAWLOCK(sacefpd_awlock), .SACEFPDAWCACHE(sacefpd_awcache), .SACEFPDAWPROT(sacefpd_awprot), .SACEFPDAWDOMAIN(sacefpd_awdomain), .SACEFPDAWSNOOP(sacefpd_awsnoop), .SACEFPDAWBAR(sacefpd_awbar), .SACEFPDAWQOS(sacefpd_awqos), .SACEFPDAWUSER({6'b000000,4'b1111,sacefpd_awuser[5:0]}), .SACEFPDWVALID(sacefpd_wvalid), .SACEFPDWREADY(sacefpd_wready), .SACEFPDWDATA(sacefpd_wdata), .SACEFPDWSTRB(sacefpd_wstrb), .SACEFPDWLAST(sacefpd_wlast), .SACEFPDWUSER(sacefpd_wuser), .SACEFPDBVALID(sacefpd_bvalid), .SACEFPDBREADY(sacefpd_bready), .SACEFPDBID(sacefpd_bid), .SACEFPDBRESP(sacefpd_bresp), .SACEFPDBUSER(sacefpd_buser), .SACEFPDARVALID(sacefpd_arvalid), .SACEFPDARREADY(sacefpd_arready), .SACEFPDARID(sacefpd_arid), .SACEFPDARADDR(sacefpd_araddr), .SACEFPDARREGION(sacefpd_arregion), .SACEFPDARLEN(sacefpd_arlen), .SACEFPDARSIZE(sacefpd_arsize), .SACEFPDARBURST(sacefpd_arburst), .SACEFPDARLOCK(sacefpd_arlock), .SACEFPDARCACHE(sacefpd_arcache), .SACEFPDARPROT(sacefpd_arprot), .SACEFPDARDOMAIN(sacefpd_ardomain), .SACEFPDARSNOOP(sacefpd_arsnoop), .SACEFPDARBAR(sacefpd_arbar), .SACEFPDARQOS(sacefpd_arqos), .SACEFPDARUSER({6'b000000,4'b1111,sacefpd_aruser[5:0]}), .SACEFPDRVALID(sacefpd_rvalid), .SACEFPDRREADY(sacefpd_rready), .SACEFPDRID(sacefpd_rid), .SACEFPDRDATA(sacefpd_rdata), .SACEFPDRRESP(sacefpd_rresp), .SACEFPDRLAST(sacefpd_rlast), .SACEFPDRUSER(sacefpd_ruser), .SACEFPDACVALID(sacefpd_acvalid), .SACEFPDACREADY(sacefpd_acready), .SACEFPDACADDR(sacefpd_acaddr), .SACEFPDACSNOOP(sacefpd_acsnoop), .SACEFPDACPROT(sacefpd_acprot), .SACEFPDCRVALID(sacefpd_crvalid), .SACEFPDCRREADY(sacefpd_crready), .SACEFPDCRRESP(sacefpd_crresp), .SACEFPDCDVALID(sacefpd_cdvalid), .SACEFPDCDREADY(sacefpd_cdready), .SACEFPDCDDATA(sacefpd_cddata), .SACEFPDCDLAST(sacefpd_cdlast), .SACEFPDWACK(sacefpd_wack), .SACEFPDRACK(sacefpd_rack),

			.EMIOCAN0PHYTX(emio_can0_phy_tx), .EMIOCAN0PHYRX(emio_can0_phy_rx), .EMIOCAN1PHYTX(emio_can1_phy_tx), .EMIOCAN1PHYRX(emio_can1_phy_rx),

			.EMIOENET0GMIIRXCLK(emio_enet0_gmii_rx_clk), .EMIOENET0SPEEDMODE(emio_enet0_speed_mode), .EMIOENET0GMIICRS(emio_enet0_gmii_crs), .EMIOENET0GMIICOL(emio_enet0_gmii_col), .EMIOENET0GMIIRXD(emio_enet0_gmii_rxd), .EMIOENET0GMIIRXER(emio_enet0_gmii_rx_er), .EMIOENET0GMIIRXDV(emio_enet0_gmii_rx_dv), .EMIOENET0GMIITXCLK(emio_enet0_gmii_tx_clk), .EMIOENET0GMIITXD(emio_enet0_gmii_txd), .EMIOENET0GMIITXEN(emio_enet0_gmii_tx_en), .EMIOENET0GMIITXER(emio_enet0_gmii_tx_er), .EMIOENET0MDIOMDC(emio_enet0_mdio_mdc), .EMIOENET0MDIOI(emio_enet0_mdio_i), .EMIOENET0MDIOO(emio_enet0_mdio_o), .EMIOENET0MDIOTN(emio_enet0_mdio_tri),

			.EMIOENET1GMIIRXCLK(emio_enet1_gmii_rx_clk), .EMIOENET1SPEEDMODE(emio_enet1_speed_mode), .EMIOENET1GMIICRS(emio_enet1_gmii_crs), .EMIOENET1GMIICOL(emio_enet1_gmii_col), .EMIOENET1GMIIRXD(emio_enet1_gmii_rxd), .EMIOENET1GMIIRXER(emio_enet1_gmii_rx_er), .EMIOENET1GMIIRXDV(emio_enet1_gmii_rx_dv), .EMIOENET1GMIITXCLK(emio_enet1_gmii_tx_clk), .EMIOENET1GMIITXD(emio_enet1_gmii_txd), .EMIOENET1GMIITXEN(emio_enet1_gmii_tx_en), .EMIOENET1GMIITXER(emio_enet1_gmii_tx_er), .EMIOENET1MDIOMDC(emio_enet1_mdio_mdc), .EMIOENET1MDIOI(emio_enet1_mdio_i), .EMIOENET1MDIOO(emio_enet1_mdio_o), .EMIOENET1MDIOTN(emio_enet1_mdio_tri),

			.EMIOENET2GMIIRXCLK(emio_enet2_gmii_rx_clk), .EMIOENET2SPEEDMODE(emio_enet2_speed_mode), .EMIOENET2GMIICRS(emio_enet2_gmii_crs), .EMIOENET2GMIICOL(emio_enet2_gmii_col), .EMIOENET2GMIIRXD(emio_enet2_gmii_rxd), .EMIOENET2GMIIRXER(emio_enet2_gmii_rx_er), .EMIOENET2GMIIRXDV(emio_enet2_gmii_rx_dv), .EMIOENET2GMIITXCLK(emio_enet2_gmii_tx_clk), .EMIOENET2GMIITXD(emio_enet2_gmii_txd), .EMIOENET2GMIITXEN(emio_enet2_gmii_tx_en), .EMIOENET2GMIITXER(emio_enet2_gmii_tx_er), .EMIOENET2MDIOMDC(emio_enet2_mdio_mdc), .EMIOENET2MDIOI(emio_enet2_mdio_i), .EMIOENET2MDIOO(emio_enet2_mdio_o), .EMIOENET2MDIOTN(emio_enet2_mdio_tri),

			.EMIOENET3GMIIRXCLK(emio_enet3_gmii_rx_clk), .EMIOENET3SPEEDMODE(emio_enet3_speed_mode), .EMIOENET3GMIICRS(emio_enet3_gmii_crs), .EMIOENET3GMIICOL(emio_enet3_gmii_col), .EMIOENET3GMIIRXD(emio_enet3_gmii_rxd), .EMIOENET3GMIIRXER(emio_enet3_gmii_rx_er), .EMIOENET3GMIIRXDV(emio_enet3_gmii_rx_dv), .EMIOENET3GMIITXCLK(emio_enet3_gmii_tx_clk), .EMIOENET3GMIITXD(emio_enet3_gmii_txd), .EMIOENET3GMIITXEN(emio_enet3_gmii_tx_en), .EMIOENET3GMIITXER(emio_enet3_gmii_tx_er), .EMIOENET3MDIOMDC(emio_enet3_mdio_mdc), .EMIOENET3MDIOI(emio_enet3_mdio_i), .EMIOENET3MDIOO(emio_enet3_mdio_o), .EMIOENET3MDIOTN(emio_enet3_mdio_tri),

			.EMIOENET0TXRDATARDY(emio_enet0_tx_r_data_rdy), .EMIOENET0TXRRD(emio_enet0_tx_r_rd), .EMIOENET0TXRVALID(emio_enet0_tx_r_valid), .EMIOENET0TXRDATA(emio_enet0_tx_r_data), .EMIOENET0TXRSOP(emio_enet0_tx_r_sop), .EMIOENET0TXREOP(emio_enet0_tx_r_eop), .EMIOENET0TXRERR(emio_enet0_tx_r_err), .EMIOENET0TXRUNDERFLOW(emio_enet0_tx_r_underflow), .EMIOENET0TXRFLUSHED(emio_enet0_tx_r_flushed), .EMIOENET0TXRCONTROL(emio_enet0_tx_r_control), .EMIOENET0DMATXENDTOG(emio_enet0_dma_tx_end_tog), .EMIOENET0DMATXSTATUSTOG(emio_enet0_dma_tx_status_tog), .EMIOENET0TXRSTATUS(emio_enet0_tx_r_status), .EMIOENET0RXWWR(emio_enet0_rx_w_wr), .EMIOENET0RXWDATA(emio_enet0_rx_w_data), .EMIOENET0RXWSOP(emio_enet0_rx_w_sop), .EMIOENET0RXWEOP(emio_enet0_rx_w_eop), .EMIOENET0RXWSTATUS(emio_enet0_rx_w_status), .EMIOENET0RXWERR(emio_enet0_rx_w_err), .EMIOENET0RXWOVERFLOW(emio_enet0_rx_w_overflow), .FMIOGEM0SIGNALDETECT(emio_enet0_signal_detect), .EMIOENET0RXWFLUSH(emio_enet0_rx_w_flush), .EMIOGEM0TXRFIXEDLAT(emio_enet0_tx_r_fixed_lat), .FMIOGEM0FIFOTXCLKFROMPL(fmio_gem0_fifo_tx_clk), .FMIOGEM0FIFORXCLKFROMPL(fmio_gem0_fifo_rx_clk), .FMIOGEM0FIFOTXCLKTOPLBUFG(fmio_gem0_fifo_tx_clk_temp), .FMIOGEM0FIFORXCLKTOPLBUFG(fmio_gem0_fifo_rx_clk_temp),

			.EMIOENET1TXRDATARDY(emio_enet1_tx_r_data_rdy), .EMIOENET1TXRRD(emio_enet1_tx_r_rd), .EMIOENET1TXRVALID(emio_enet1_tx_r_valid), .EMIOENET1TXRDATA(emio_enet1_tx_r_data), .EMIOENET1TXRSOP(emio_enet1_tx_r_sop), .EMIOENET1TXREOP(emio_enet1_tx_r_eop), .EMIOENET1TXRERR(emio_enet1_tx_r_err), .EMIOENET1TXRUNDERFLOW(emio_enet1_tx_r_underflow), .EMIOENET1TXRFLUSHED(emio_enet1_tx_r_flushed), .EMIOENET1TXRCONTROL(emio_enet1_tx_r_control), .EMIOENET1DMATXENDTOG(emio_enet1_dma_tx_end_tog), .EMIOENET1DMATXSTATUSTOG(emio_enet1_dma_tx_status_tog), .EMIOENET1TXRSTATUS(emio_enet1_tx_r_status), .EMIOENET1RXWWR(emio_enet1_rx_w_wr), .EMIOENET1RXWDATA(emio_enet1_rx_w_data), .EMIOENET1RXWSOP(emio_enet1_rx_w_sop), .EMIOENET1RXWEOP(emio_enet1_rx_w_eop), .EMIOENET1RXWSTATUS(emio_enet1_rx_w_status), .EMIOENET1RXWERR(emio_enet1_rx_w_err), .EMIOENET1RXWOVERFLOW(emio_enet1_rx_w_overflow), .FMIOGEM1SIGNALDETECT(emio_enet1_signal_detect), .EMIOENET1RXWFLUSH(emio_enet1_rx_w_flush), .EMIOGEM1TXRFIXEDLAT(emio_enet1_tx_r_fixed_lat), .FMIOGEM1FIFOTXCLKFROMPL(fmio_gem1_fifo_tx_clk), .FMIOGEM1FIFORXCLKFROMPL(fmio_gem1_fifo_rx_clk), .FMIOGEM1FIFOTXCLKTOPLBUFG(fmio_gem1_fifo_tx_clk_temp), .FMIOGEM1FIFORXCLKTOPLBUFG(fmio_gem1_fifo_rx_clk_temp),

			.EMIOENET2TXRDATARDY(emio_enet2_tx_r_data_rdy), .EMIOENET2TXRRD(emio_enet2_tx_r_rd), .EMIOENET2TXRVALID(emio_enet2_tx_r_valid), .EMIOENET2TXRDATA(emio_enet2_tx_r_data), .EMIOENET2TXRSOP(emio_enet2_tx_r_sop), .EMIOENET2TXREOP(emio_enet2_tx_r_eop), .EMIOENET2TXRERR(emio_enet2_tx_r_err), .EMIOENET2TXRUNDERFLOW(emio_enet2_tx_r_underflow), .EMIOENET2TXRFLUSHED(emio_enet2_tx_r_flushed), .EMIOENET2TXRCONTROL(emio_enet2_tx_r_control), .EMIOENET2DMATXENDTOG(emio_enet2_dma_tx_end_tog), .EMIOENET2DMATXSTATUSTOG(emio_enet2_dma_tx_status_tog), .EMIOENET2TXRSTATUS(emio_enet2_tx_r_status), .EMIOENET2RXWWR(emio_enet2_rx_w_wr), .EMIOENET2RXWDATA(emio_enet2_rx_w_data), .EMIOENET2RXWSOP(emio_enet2_rx_w_sop), .EMIOENET2RXWEOP(emio_enet2_rx_w_eop), .EMIOENET2RXWSTATUS(emio_enet2_rx_w_status), .EMIOENET2RXWERR(emio_enet2_rx_w_err), .EMIOENET2RXWOVERFLOW(emio_enet2_rx_w_overflow), .FMIOGEM2SIGNALDETECT(emio_enet2_signal_detect), .EMIOENET2RXWFLUSH(emio_enet2_rx_w_flush), .EMIOGEM2TXRFIXEDLAT(emio_enet2_tx_r_fixed_lat), .FMIOGEM2FIFOTXCLKFROMPL(fmio_gem2_fifo_tx_clk), .FMIOGEM2FIFORXCLKFROMPL(fmio_gem2_fifo_rx_clk), .FMIOGEM2FIFOTXCLKTOPLBUFG(fmio_gem2_fifo_tx_clk_temp), .FMIOGEM2FIFORXCLKTOPLBUFG(fmio_gem2_fifo_rx_clk_temp),

			.EMIOENET3TXRDATARDY(emio_enet3_tx_r_data_rdy), .EMIOENET3TXRRD(emio_enet3_tx_r_rd), .EMIOENET3TXRVALID(emio_enet3_tx_r_valid), .EMIOENET3TXRDATA(emio_enet3_tx_r_data), .EMIOENET3TXRSOP(emio_enet3_tx_r_sop), .EMIOENET3TXREOP(emio_enet3_tx_r_eop), .EMIOENET3TXRERR(emio_enet3_tx_r_err), .EMIOENET3TXRUNDERFLOW(emio_enet3_tx_r_underflow), .EMIOENET3TXRFLUSHED(emio_enet3_tx_r_flushed), .EMIOENET3TXRCONTROL(emio_enet3_tx_r_control), .EMIOENET3DMATXENDTOG(emio_enet3_dma_tx_end_tog), .EMIOENET3DMATXSTATUSTOG(emio_enet3_dma_tx_status_tog), .EMIOENET3TXRSTATUS(emio_enet3_tx_r_status), .EMIOENET3RXWWR(emio_enet3_rx_w_wr), .EMIOENET3RXWDATA(emio_enet3_rx_w_data), .EMIOENET3RXWSOP(emio_enet3_rx_w_sop), .EMIOENET3RXWEOP(emio_enet3_rx_w_eop), .EMIOENET3RXWSTATUS(emio_enet3_rx_w_status), .EMIOENET3RXWERR(emio_enet3_rx_w_err), .EMIOENET3RXWOVERFLOW(emio_enet3_rx_w_overflow), .FMIOGEM3SIGNALDETECT(emio_enet3_signal_detect), .EMIOENET3RXWFLUSH(emio_enet3_rx_w_flush), .EMIOGEM3TXRFIXEDLAT(emio_enet3_tx_r_fixed_lat), .FMIOGEM3FIFOTXCLKFROMPL(fmio_gem3_fifo_tx_clk), .FMIOGEM3FIFORXCLKFROMPL(fmio_gem3_fifo_rx_clk), .FMIOGEM3FIFOTXCLKTOPLBUFG(fmio_gem3_fifo_tx_clk_temp), .FMIOGEM3FIFORXCLKTOPLBUFG(fmio_gem3_fifo_rx_clk_temp),

			.EMIOGEM0TXSOF(emio_enet0_tx_sof), .EMIOGEM0SYNCFRAMETX(emio_enet0_sync_frame_tx), .EMIOGEM0DELAYREQTX(emio_enet0_delay_req_tx), .EMIOGEM0PDELAYREQTX(emio_enet0_pdelay_req_tx), .EMIOGEM0PDELAYRESPTX(emio_enet0_pdelay_resp_tx), .EMIOGEM0RXSOF(emio_enet0_rx_sof), .EMIOGEM0SYNCFRAMERX(emio_enet0_sync_frame_rx), .EMIOGEM0DELAYREQRX(emio_enet0_delay_req_rx), .EMIOGEM0PDELAYREQRX(emio_enet0_pdelay_req_rx), .EMIOGEM0PDELAYRESPRX(emio_enet0_pdelay_resp_rx), .EMIOGEM0TSUINCCTRL(emio_enet0_tsu_inc_ctrl), .EMIOGEM0TSUTIMERCMPVAL(emio_enet0_tsu_timer_cmp_val),

			.EMIOGEM1TXSOF(emio_enet1_tx_sof), .EMIOGEM1SYNCFRAMETX(emio_enet1_sync_frame_tx), .EMIOGEM1DELAYREQTX(emio_enet1_delay_req_tx), .EMIOGEM1PDELAYREQTX(emio_enet1_pdelay_req_tx), .EMIOGEM1PDELAYRESPTX(emio_enet1_pdelay_resp_tx), .EMIOGEM1RXSOF(emio_enet1_rx_sof), .EMIOGEM1SYNCFRAMERX(emio_enet1_sync_frame_rx), .EMIOGEM1DELAYREQRX(emio_enet1_delay_req_rx), .EMIOGEM1PDELAYREQRX(emio_enet1_pdelay_req_rx), .EMIOGEM1PDELAYRESPRX(emio_enet1_pdelay_resp_rx), .EMIOGEM1TSUINCCTRL(emio_enet1_tsu_inc_ctrl), .EMIOGEM1TSUTIMERCMPVAL(emio_enet1_tsu_timer_cmp_val),

			.EMIOGEM2TXSOF(emio_enet2_tx_sof), .EMIOGEM2SYNCFRAMETX(emio_enet2_sync_frame_tx), .EMIOGEM2DELAYREQTX(emio_enet2_delay_req_tx), .EMIOGEM2PDELAYREQTX(emio_enet2_pdelay_req_tx), .EMIOGEM2PDELAYRESPTX(emio_enet2_pdelay_resp_tx), .EMIOGEM2RXSOF(emio_enet2_rx_sof), .EMIOGEM2SYNCFRAMERX(emio_enet2_sync_frame_rx), .EMIOGEM2DELAYREQRX(emio_enet2_delay_req_rx), .EMIOGEM2PDELAYREQRX(emio_enet2_pdelay_req_rx), .EMIOGEM2PDELAYRESPRX(emio_enet2_pdelay_resp_rx), .EMIOGEM2TSUINCCTRL(emio_enet2_tsu_inc_ctrl), .EMIOGEM2TSUTIMERCMPVAL(emio_enet2_tsu_timer_cmp_val),

			.EMIOGEM3TXSOF(emio_enet3_tx_sof), .EMIOGEM3SYNCFRAMETX(emio_enet3_sync_frame_tx), .EMIOGEM3DELAYREQTX(emio_enet3_delay_req_tx), .EMIOGEM3PDELAYREQTX(emio_enet3_pdelay_req_tx), .EMIOGEM3PDELAYRESPTX(emio_enet3_pdelay_resp_tx), .EMIOGEM3RXSOF(emio_enet3_rx_sof), .EMIOGEM3SYNCFRAMERX(emio_enet3_sync_frame_rx), .EMIOGEM3DELAYREQRX(emio_enet3_delay_req_rx), .EMIOGEM3PDELAYREQRX(emio_enet3_pdelay_req_rx), .EMIOGEM3PDELAYRESPRX(emio_enet3_pdelay_resp_rx), .EMIOGEM3TSUINCCTRL(emio_enet3_tsu_inc_ctrl), .EMIOGEM3TSUTIMERCMPVAL(emio_enet3_tsu_timer_cmp_val),

			.FMIOGEMTSUCLKFROMPL(fmio_gem_tsu_clk_from_pl), .FMIOGEMTSUCLKTOPLBUFG(fmio_gem_tsu_clk_to_pl_bufg), .EMIOENETTSUCLK(emio_enet_tsu_clk), .EMIOENET0GEMTSUTIMERCNT(emio_enet0_enet_tsu_timer_cnt),

			.EMIOENET0EXTINTIN(emio_enet0_ext_int_in), .EMIOENET1EXTINTIN(emio_enet1_ext_int_in), .EMIOENET2EXTINTIN(emio_enet2_ext_int_in), .EMIOENET3EXTINTIN(emio_enet3_ext_int_in),

			.EMIOENET0DMABUSWIDTH(emio_enet0_dma_bus_width), .EMIOENET1DMABUSWIDTH(emio_enet1_dma_bus_width), .EMIOENET2DMABUSWIDTH(emio_enet2_dma_bus_width), .EMIOENET3DMABUSWIDTH(emio_enet3_dma_bus_width),

			.EMIOGPIOI(emio_gpio_i_temp), .EMIOGPIOO(emio_gpio_o_temp), .EMIOGPIOTN(emio_gpio_t_temp),

			.EMIOI2C0SCLI(emio_i2c0_scl_i), .EMIOI2C0SCLO(emio_i2c0_scl_o), .EMIOI2C0SCLTN(emio_i2c0_scl_tri), .EMIOI2C0SDAI(emio_i2c0_sda_i), .EMIOI2C0SDAO(emio_i2c0_sda_o), .EMIOI2C0SDATN(emio_i2c0_sda_tri),

			.EMIOI2C1SCLI(emio_i2c1_scl_i), .EMIOI2C1SCLO(emio_i2c1_scl_o), .EMIOI2C1SCLTN(emio_i2c1_scl_tri), .EMIOI2C1SDAI(emio_i2c1_sda_i), .EMIOI2C1SDAO(emio_i2c1_sda_o), .EMIOI2C1SDATN(emio_i2c1_sda_tri),

			.EMIOUART0TX(emio_uart0_txd), .EMIOUART0RX(emio_uart0_rxd), .EMIOUART0CTSN(emio_uart0_ctsn), .EMIOUART0RTSN(emio_uart0_rtsn), .EMIOUART0DSRN(emio_uart0_dsrn), .EMIOUART0DCDN(emio_uart0_dcdn), .EMIOUART0RIN(emio_uart0_rin), .EMIOUART0DTRN(emio_uart0_dtrn),

			.EMIOUART1TX(emio_uart1_txd), .EMIOUART1RX(emio_uart1_rxd), .EMIOUART1CTSN(emio_uart1_ctsn), .EMIOUART1RTSN(emio_uart1_rtsn), .EMIOUART1DSRN(emio_uart1_dsrn), .EMIOUART1DCDN(emio_uart1_dcdn), .EMIOUART1RIN(emio_uart1_rin), .EMIOUART1DTRN(emio_uart1_dtrn),

			.EMIOSDIO0CLKOUT(emio_sdio0_clkout), .EMIOSDIO0FBCLKIN(emio_sdio0_fb_clk_in), .EMIOSDIO0CMDOUT(emio_sdio0_cmdout), .EMIOSDIO0CMDIN(emio_sdio0_cmdin), .EMIOSDIO0CMDENA(emio_sdio0_cmdena_i), .EMIOSDIO0DATAIN(emio_sdio0_datain), .EMIOSDIO0DATAOUT(emio_sdio0_dataout), .EMIOSDIO0DATAENA(emio_sdio0_dataena_i), .EMIOSDIO0CDN(emio_sdio0_cd_n), .EMIOSDIO0WP(emio_sdio0_wp), .EMIOSDIO0LEDCONTROL(emio_sdio0_ledcontrol), .EMIOSDIO0BUSPOWER(emio_sdio0_buspower), .EMIOSDIO0BUSVOLT(emio_sdio0_bus_volt),

			.EMIOSDIO1CLKOUT(emio_sdio1_clkout), .EMIOSDIO1FBCLKIN(emio_sdio1_fb_clk_in), .EMIOSDIO1CMDOUT(emio_sdio1_cmdout), .EMIOSDIO1CMDIN(emio_sdio1_cmdin), .EMIOSDIO1CMDENA(emio_sdio1_cmdena_i), .EMIOSDIO1DATAIN(emio_sdio1_datain), .EMIOSDIO1DATAOUT(emio_sdio1_dataout), .EMIOSDIO1DATAENA(emio_sdio1_dataena_i), .EMIOSDIO1CDN(emio_sdio1_cd_n), .EMIOSDIO1WP(emio_sdio1_wp), .EMIOSDIO1LEDCONTROL(emio_sdio1_ledcontrol), .EMIOSDIO1BUSPOWER(emio_sdio1_buspower), .EMIOSDIO1BUSVOLT(emio_sdio1_bus_volt),

			.EMIOSPI0SCLKI(emio_spi0_sclk_i), .EMIOSPI0SCLKO(emio_spi0_sclk_o), .EMIOSPI0SCLKTN(emio_spi0_sclk_tri), .EMIOSPI0MI(emio_spi0_m_i), .EMIOSPI0MO(emio_spi0_m_o), .EMIOSPI0MOTN(emio_spi0_mo_tri), .EMIOSPI0SI(emio_spi0_s_i), .EMIOSPI0SO(emio_spi0_s_o), .EMIOSPI0STN(emio_spi0_so_tri), .EMIOSPI0SSIN(emio_spi0_ss_i_n), .EMIOSPI0SSON({emio_spi0_ss2_o_n,emio_spi0_ss1_o_n,emio_spi0_ss_o_n}), .EMIOSPI0SSNTN(emio_spi0_ss_n_tri),

			.EMIOSPI1SCLKI(emio_spi1_sclk_i), .EMIOSPI1SCLKO(emio_spi1_sclk_o), .EMIOSPI1SCLKTN(emio_spi1_sclk_tri), .EMIOSPI1MI(emio_spi1_m_i), .EMIOSPI1MO(emio_spi1_m_o), .EMIOSPI1MOTN(emio_spi1_mo_tri), .EMIOSPI1SI(emio_spi1_s_i), .EMIOSPI1SO(emio_spi1_s_o), .EMIOSPI1STN(emio_spi1_so_tri), .EMIOSPI1SSIN(emio_spi1_ss_i_n), .EMIOSPI1SSON({emio_spi1_ss2_o_n,emio_spi1_ss1_o_n,emio_spi1_ss_o_n}), .EMIOSPI1SSNTN(emio_spi1_ss_n_tri),

			.PLPSTRACECLK(pl_ps_trace_clk), .PSPLTRACECTL(trace_ctl_i), .PSPLTRACEDATA(trace_data_i),

			.EMIOTTC0WAVEO(emio_ttc0_wave_o), .EMIOTTC0CLKI(emio_ttc0_clk_i),

			.EMIOTTC1WAVEO(emio_ttc1_wave_o), .EMIOTTC1CLKI(emio_ttc1_clk_i),

			.EMIOTTC2WAVEO(emio_ttc2_wave_o), .EMIOTTC2CLKI(emio_ttc2_clk_i),

			.EMIOTTC3WAVEO(emio_ttc3_wave_o), .EMIOTTC3CLKI(emio_ttc3_clk_i),

			.EMIOWDT0CLKI(emio_wdt0_clk_i), .EMIOWDT0RSTO(emio_wdt0_rst_o),

			.EMIOWDT1CLKI(emio_wdt1_clk_i), .EMIOWDT1RSTO(emio_wdt1_rst_o),

			.EMIOHUBPORTOVERCRNTUSB30(emio_hub_port_overcrnt_usb3_0), .EMIOHUBPORTOVERCRNTUSB31(emio_hub_port_overcrnt_usb3_1), .EMIOHUBPORTOVERCRNTUSB20(emio_hub_port_overcrnt_usb2_0), .EMIOHUBPORTOVERCRNTUSB21(emio_hub_port_overcrnt_usb2_1),

			.EMIOU2DSPORTVBUSCTRLUSB30(emio_u2dsport_vbus_ctrl_usb3_0), .EMIOU2DSPORTVBUSCTRLUSB31(emio_u2dsport_vbus_ctrl_usb3_1), .EMIOU3DSPORTVBUSCTRLUSB30(emio_u3dsport_vbus_ctrl_usb3_0), .EMIOU3DSPORTVBUSCTRLUSB31(emio_u3dsport_vbus_ctrl_usb3_1),

			.ADMAFCICLK(adma_fci_clk), .PL2ADMACVLD(pl2adma_cvld), .PL2ADMATACK(pl2adma_tack), .ADMA2PLCACK(adma2pl_cack), .ADMA2PLTVLD(adma2pl_tvld),

			.GDMAFCICLK(perif_gdma_clk), .PL2GDMACVLD(perif_gdma_cvld), .PL2GDMATACK(perif_gdma_tack), .GDMA2PLCACK(gdma_perif_cack), .GDMA2PLTVLD(gdma_perif_tvld),

			.PLFPGASTOP(pl_clock_stop),

			.PLLAUXREFCLKLPD(pll_aux_refclk_lpd), .PLLAUXREFCLKFPD(pll_aux_refclk_fpd),

			.DPSAXISAUDIOTDATA(dp_s_axis_audio_tdata), .DPSAXISAUDIOTID(dp_s_axis_audio_tid), .DPSAXISAUDIOTVALID(dp_s_axis_audio_tvalid), .DPSAXISAUDIOTREADY(dp_s_axis_audio_tready),

			.DPMAXISMIXEDAUDIOTDATA(dp_m_axis_mixed_audio_tdata), .DPMAXISMIXEDAUDIOTID(dp_m_axis_mixed_audio_tid), .DPMAXISMIXEDAUDIOTVALID(dp_m_axis_mixed_audio_tvalid), .DPMAXISMIXEDAUDIOTREADY(dp_m_axis_mixed_audio_tready),

			.DPSAXISAUDIOCLK(dp_s_axis_audio_clk),

			.DPLIVEVIDEOINVSYNC(dp_live_video_in_vsync), .DPLIVEVIDEOINHSYNC(dp_live_video_in_hsync), .DPLIVEVIDEOINDE(dp_live_video_in_de), .DPLIVEVIDEOINPIXEL1(dp_live_video_in_pixel1),

			.DPVIDEOINCLK(dp_video_in_clk), .DPVIDEOOUTHSYNC(dp_video_out_hsync), .DPVIDEOOUTVSYNC(dp_video_out_vsync), .DPVIDEOOUTPIXEL1(dp_video_out_pixel1),

			.DPAUXDATAIN(dp_aux_data_in), .DPAUXDATAOUT(dp_aux_data_out), .DPAUXDATAOEN(dp_aux_data_oe_n),

		.DPLIVEGFXALPHAIN (dp_live_gfx_alpha_in),
		.DPLIVEGFXPIXEL1IN (dp_live_gfx_pixel1_in),
		.DPHOTPLUGDETECT (dp_hot_plug_detect),
		.DPEXTERNALCUSTOMEVENT1 (dp_external_custom_event1),
		.DPEXTERNALCUSTOMEVENT2 (dp_external_custom_event2),
		.DPEXTERNALVSYNCEVENT (dp_external_vsync_event),
		.DPLIVEVIDEODEOUT (dp_live_video_de_out),

			.PLPSEVENTI(pl_ps_eventi), .PSPLEVENTO(ps_pl_evento),

			.PSPLSTANDBYWFE(ps_pl_standbywfe), .PSPLSTANDBYWFI(ps_pl_standbywfi),
			.PLPSAPUGICIRQ(pl_ps_apugic_irq), .PLPSAPUGICFIQ(pl_ps_apugic_fiq),

			.RPUEVENTI0(rpu_eventi0), .RPUEVENTI1(rpu_eventi1),
			.RPUEVENTO0(rpu_evento0), .RPUEVENTO1(rpu_evento1),

		.NFIQ0LPDRPU(nfiq0_lpd_rpu), .NFIQ1LPDRPU(nfiq1_lpd_rpu), .NIRQ0LPDRPU(nirq0_lpd_rpu), .NIRQ1LPDRPU(nirq1_lpd_rpu),

		.STMEVENT (stm_event),

		.PLPSTRIGACK({pl_ps_trigack_3, pl_ps_trigack_2, pl_ps_trigack_1, pl_ps_trigack_0}), .PLPSTRIGGER({pl_ps_trigger_3, pl_ps_trigger_2, pl_ps_trigger_1, pl_ps_trigger_0}),
		.PSPLTRIGACK({ps_pl_trigack_3, ps_pl_trigack_2, ps_pl_trigack_1, ps_pl_trigack_0}), .PSPLTRIGGER({ps_pl_trigger_3, ps_pl_trigger_2, ps_pl_trigger_1, ps_pl_trigger_0}),

		.FTMGPO(ftm_gpo), .FTMGPI(ftm_gpi),
		.PLPSIRQ0(irq_f2p_0_i), .PLPSIRQ1(irq_f2p_1_i),

		.PSPLIRQLPD({ps_pl_irq_lpd_low[18:8], ps_pl_irq_xmpu_lpd, ps_pl_irq_efuse, ps_pl_irq_csu_dma, ps_pl_irq_csu, ps_pl_irq_adma_chan, ps_pl_irq_usb3_0_pmu_wakeup, ps_pl_irq_usb3_1_otg, ps_pl_irq_usb3_1_endpoint, ps_pl_irq_usb3_0_otg, ps_pl_irq_usb3_0_endpoint, ps_pl_irq_enet3_wake, ps_pl_irq_enet3, ps_pl_irq_enet2_wake, ps_pl_irq_enet2, ps_pl_irq_enet1_wake, ps_pl_irq_enet1, ps_pl_irq_enet0_wake, ps_pl_irq_enet0, ps_pl_irq_ams, ps_pl_irq_aib_axi, ps_pl_irq_atb_err_lpd, ps_pl_irq_csu_pmu_wdt, ps_pl_irq_lp_wdt, ps_pl_irq_sdio1_wake, ps_pl_irq_sdio0_wake, ps_pl_irq_sdio1, ps_pl_irq_sdio0, ps_pl_irq_ttc3_2, ps_pl_irq_ttc3_1, ps_pl_irq_ttc3_0, ps_pl_irq_ttc2_2, ps_pl_irq_ttc2_1, ps_pl_irq_ttc2_0, ps_pl_irq_ttc1_2, ps_pl_irq_ttc1_1, ps_pl_irq_ttc1_0, ps_pl_irq_ttc0_2, ps_pl_irq_ttc0_1, ps_pl_irq_ttc0_0, ps_pl_irq_ipi_channel0, ps_pl_irq_ipi_channel1, ps_pl_irq_ipi_channel2, ps_pl_irq_ipi_channel7, ps_pl_irq_ipi_channel8, ps_pl_irq_ipi_channel9, ps_pl_irq_ipi_channel10, ps_pl_irq_clkmon, ps_pl_irq_rtc_seconds, ps_pl_irq_rtc_alaram, ps_pl_irq_lpd_apm, ps_pl_irq_can1, ps_pl_irq_can0, ps_pl_irq_uart1, ps_pl_irq_uart0, ps_pl_irq_spi1, ps_pl_irq_spi0, ps_pl_irq_i2c1, ps_pl_irq_i2c0, ps_pl_irq_gpio, ps_pl_irq_qspi, ps_pl_irq_nand, ps_pl_irq_r5_core1_ecc_error, ps_pl_irq_r5_core0_ecc_error, ps_pl_irq_lpd_apb_intr, ps_pl_irq_ocm_error, ps_pl_irq_rpu_pm, ps_pl_irq_lpd_low[7:0]}), .PSPLIRQFPD({ps_pl_irq_fpd_low[19:12], ps_pl_irq_intf_fpd_smmu, ps_pl_irq_intf_ppd_cci, ps_pl_irq_apu_regs, ps_pl_irq_apu_exterr, ps_pl_irq_apu_l2err, ps_pl_irq_apu_comm, ps_pl_irq_apu_pmu, ps_pl_irq_apu_cti, ps_pl_irq_apu_cpumnt, ps_pl_irq_xmpu_fpd, ps_pl_irq_sata, ps_pl_irq_gpu, ps_pl_irq_gdma_chan, ps_pl_irq_apm_fpd, ps_pl_irq_dpdma, ps_pl_irq_fpd_atb_error, ps_pl_irq_fpd_apb_int, ps_pl_irq_dport, ps_pl_irq_pcie_msc, ps_pl_irq_pcie_dma, ps_pl_irq_pcie_legacy, ps_pl_irq_pcie_msi, ps_pl_irq_fp_wdt, ps_pl_irq_ddr_ss, ps_pl_irq_fpd_low[11:0]}),
		.OSCRTCCLK (osc_rtc_clk),
		.PLPMUGPI (pl_pmu_gpi),
		.PMUPLGPO (pmu_pl_gpo),
		.AIBPMUAFIFMFPDACK (aib_pmu_afifm_fpd_ack),
		.AIBPMUAFIFMLPDACK (aib_pmu_afifm_lpd_ack),
		.PMUAIBAFIFMFPDREQ (pmu_aib_afifm_fpd_req),
		.PMUAIBAFIFMLPDREQ (pmu_aib_afifm_lpd_req),
		.PMUERRORTOPL (pmu_error_to_pl),
		.PMUERRORFROMPL (pmu_error_from_pl),
		.DDRCEXTREFRESHRANK0REQ (ddrc_ext_refresh_rank0_req),
		.DDRCEXTREFRESHRANK1REQ (ddrc_ext_refresh_rank1_req),
		.DDRCREFRESHPLCLK (ddrc_refresh_pl_clk),
		.PLACPINACT (pl_acpinact),
		.PLCLK (pl_clk_unbuffered),


		.TESTADCCLK (test_adc_clk),
		.TESTADCIN (test_adc_in),
		.TESTADC2IN (test_adc2_in),
		.TESTDB (test_db),
		.TESTADCOUT (test_adc_out),
		.TESTAMSOSC (test_ams_osc),
		.TESTMONDATA (test_mon_data),
		.TESTDCLK (test_dclk),
		.TESTDEN (test_den),
		.TESTDWE (test_dwe),
		.TESTDADDR (test_daddr),
		.TESTDI (test_di),
		.TESTDRDY (test_drdy),
		.TESTDO (test_do),
		.TESTCONVST (test_convst),
		.PSTPPLCLK (pstp_pl_clk),
		.PSTPPLIN (pstp_pl_in),
		.PSTPPLOUT (pstp_pl_out),
		.PSTPPLTS (pstp_pl_ts),
		.FMIOTESTGEMSCANMUX1 (fmio_test_gem_scanmux_1),
		.FMIOTESTGEMSCANMUX2 (fmio_test_gem_scanmux_2),
		.TESTCHARMODEFPDN (test_char_mode_fpd_n),
		.TESTCHARMODELPDN (test_char_mode_lpd_n),
		.FMIOTESTIOCHARSCANCLOCK (fmio_test_io_char_scan_clock),
		.FMIOTESTIOCHARSCANENABLE (fmio_test_io_char_scanenable),
		.FMIOTESTIOCHARSCANIN (fmio_test_io_char_scan_in),
		.FMIOTESTIOCHARSCANOUT (fmio_test_io_char_scan_out),
		.FMIOTESTIOCHARSCANRESETN (fmio_test_io_char_scan_reset_n),
		.FMIOCHARAFIFSLPDTESTSELECTN (fmio_char_afifslpd_test_select_n),
		.FMIOCHARAFIFSLPDTESTINPUT (fmio_char_afifslpd_test_input),
		.FMIOCHARAFIFSLPDTESTOUTPUT (fmio_char_afifslpd_test_output),
		.FMIOCHARAFIFSFPDTESTSELECTN (fmio_char_afifsfpd_test_select_n),
		.FMIOCHARAFIFSFPDTESTINPUT (fmio_char_afifsfpd_test_input),
		.FMIOCHARAFIFSFPDTESTOUTPUT (fmio_char_afifsfpd_test_output),
		.IOCHARAUDIOINTESTDATA (io_char_audio_in_test_data),
		.IOCHARAUDIOMUXSELN (io_char_audio_mux_sel_n),
		.IOCHARVIDEOINTESTDATA (io_char_video_in_test_data),
		.IOCHARVIDEOMUXSELN (io_char_video_mux_sel_n),
		.IOCHARVIDEOOUTTESTDATA (io_char_video_out_test_data),
		.IOCHARAUDIOOUTTESTDATA (io_char_audio_out_test_data),
		.FMIOTESTQSPISCANMUX1N (fmio_test_qspi_scanmux_1_n),
		.FMIOTESTSDIOSCANMUX1 (fmio_test_sdio_scanmux_1),
		.FMIOTESTSDIOSCANMUX2 (fmio_test_sdio_scanmux_2),
		.FMIOSD0DLLTESTINN (fmio_sd0_dll_test_in_n),
		.FMIOSD0DLLTESTOUT (fmio_sd0_dll_test_out),
		.FMIOSD1DLLTESTINN (fmio_sd1_dll_test_in_n),
		.FMIOSD1DLLTESTOUT (fmio_sd1_dll_test_out),
		.TESTPLSCANCHOPPERSI (test_pl_scan_chopper_si),
		.TESTPLSCANCHOPPERSO (test_pl_scan_chopper_so),
		.TESTPLSCANCHOPPERTRIG (test_pl_scan_chopper_trig),
		.TESTPLSCANCLK0 (test_pl_scan_clk0),
		.TESTPLSCANCLK1 (test_pl_scan_clk1),
		.TESTPLSCANEDTCLK (test_pl_scan_edt_clk),
		.TESTPLSCANEDTINAPU (test_pl_scan_edt_in_apu),
		.TESTPLSCANEDTINCPU (test_pl_scan_edt_in_cpu),
		.TESTPLSCANEDTINDDR (test_pl_scan_edt_in_ddr),
		.TESTPLSCANEDTINFP (test_pl_scan_edt_in_fp),
		.TESTPLSCANEDTINGPU (test_pl_scan_edt_in_gpu),
		.TESTPLSCANEDTINLP (test_pl_scan_edt_in_lp),
		.TESTPLSCANEDTINUSB3 (test_pl_scan_edt_in_usb3),
		.TESTPLSCANEDTOUTAPU (test_pl_scan_edt_out_apu),
		.TESTPLSCANEDTOUTCPU0 (test_pl_scan_edt_out_cpu0),
		.TESTPLSCANEDTOUTCPU1 (test_pl_scan_edt_out_cpu1),
		.TESTPLSCANEDTOUTCPU2 (test_pl_scan_edt_out_cpu2),
		.TESTPLSCANEDTOUTCPU3 (test_pl_scan_edt_out_cpu3),
		.TESTPLSCANEDTOUTDDR (test_pl_scan_edt_out_ddr),
		.TESTPLSCANEDTOUTFP (test_pl_scan_edt_out_fp),
		.TESTPLSCANEDTOUTGPU (test_pl_scan_edt_out_gpu),
		.TESTPLSCANEDTOUTLP (test_pl_scan_edt_out_lp),
		.TESTPLSCANEDTOUTUSB3 (test_pl_scan_edt_out_usb3),
		.TESTPLSCANEDTUPDATE (test_pl_scan_edt_update),
		.TESTPLSCANRESETN (test_pl_scan_reset_n),
		.TESTPLSCANENABLE (test_pl_scanenable),
		.TESTPLSCANPLLRESET (test_pl_scan_pll_reset),
		.TESTPLSCANSPAREIN0 (test_pl_scan_spare_in0),
		.TESTPLSCANSPAREIN1 (test_pl_scan_spare_in1),
		.TESTPLSCANSPAREOUT0 (test_pl_scan_spare_out0),
		.TESTPLSCANSPAREOUT1 (test_pl_scan_spare_out1),
		.TESTPLSCANWRAPCLK (test_pl_scan_wrap_clk),
		.TESTPLSCANWRAPISHIFT (test_pl_scan_wrap_ishift),
		.TESTPLSCANWRAPOSHIFT (test_pl_scan_wrap_oshift),
		.TESTPLSCANSLCRCONFIGCLK (test_pl_scan_slcr_config_clk),
		.TESTPLSCANSLCRCONFIGRSTN (test_pl_scan_slcr_config_rstn),
		.TESTPLSCANSLCRCONFIGSI (test_pl_scan_slcr_config_si),
		.TESTPLSCANSPAREIN2 (test_pl_scan_spare_in2),
		.TESTPLSCANENABLESLCREN (test_pl_scanenable_slcr_en),
		.TESTPLPLLLOCKOUT (test_pl_pll_lock_out),
		.TESTPLSCANSLCRCONFIGSO (test_pl_scan_slcr_config_so),
		.TSTRTCCALIBREGIN (tst_rtc_calibreg_in),
		.TSTRTCCALIBREGOUT (tst_rtc_calibreg_out),
		.TSTRTCCALIBREGWE (tst_rtc_calibreg_we),
		.TSTRTCCLK (tst_rtc_clk),
		.TSTRTCOSCCLKOUT (tst_rtc_osc_clk_out),
		.TSTRTCSECCOUNTEROUT (tst_rtc_sec_counter_out),
		.TSTRTCSECONDSRAWINT (tst_rtc_seconds_raw_int),
		.TSTRTCTESTCLOCKSELECTN (tst_rtc_testclock_select_n),
		.TSTRTCTICKCOUNTEROUT (tst_rtc_tick_counter_out),
		.TSTRTCTIMESETREGIN (tst_rtc_timesetreg_in),
		.TSTRTCTIMESETREGOUT (tst_rtc_timesetreg_out),
		.TSTRTCDISABLEBATOP (tst_rtc_disable_bat_op),
		.TSTRTCOSCCNTRLIN (tst_rtc_osc_cntrl_in),
		.TSTRTCOSCCNTRLOUT (tst_rtc_osc_cntrl_out),
		.TSTRTCOSCCNTRLWE (tst_rtc_osc_cntrl_we),
		.TSTRTCSECRELOAD (tst_rtc_sec_reload),
		.TSTRTCTIMESETREGWE (tst_rtc_timesetreg_we),
		.TSTRTCTESTMODEN (tst_rtc_testmode_n),
		.TESTUSB0FUNCMUX0N (test_usb0_funcmux_0_n),
		.TESTUSB1FUNCMUX0N (test_usb1_funcmux_0_n),
		.TESTUSB0SCANMUX0N (test_usb0_scanmux_0_n),
		.TESTUSB1SCANMUX0N (test_usb1_scanmux_0_n),
		.LPDPLLTESTOUT (lpd_pll_test_out),
		.PLLPDPLLTESTCKSELN (pl_lpd_pll_test_ck_sel_n),
		.PLLPDPLLTESTFRACTCLKSELN (pl_lpd_pll_test_fract_clk_sel_n),
		.PLLPDPLLTESTFRACTENN (pl_lpd_pll_test_fract_en_n),
		.PLLPDPLLTESTMUXSEL (pl_lpd_pll_test_mux_sel),
		.PLLPDPLLTESTSEL (pl_lpd_pll_test_sel),
		.FPDPLLTESTOUT (fpd_pll_test_out),
		.PLFPDPLLTESTCKSELN (pl_fpd_pll_test_ck_sel_n),
		.PLFPDPLLTESTFRACTCLKSELN (pl_fpd_pll_test_fract_clk_sel_n),
		.PLFPDPLLTESTFRACTENN (pl_fpd_pll_test_fract_en_n),
		.PLFPDPLLTESTMUXSEL (pl_fpd_pll_test_mux_sel),
		.PLFPDPLLTESTSEL (pl_fpd_pll_test_sel),
		.FMIOCHARGEMSELECTION (fmio_char_gem_selection),
		.FMIOCHARGEMTESTSELECTN (fmio_char_gem_test_select_n),
		.FMIOCHARGEMTESTINPUT (fmio_char_gem_test_input),
		.FMIOCHARGEMTESTOUTPUT (fmio_char_gem_test_output),
		.TESTDDR2PLDCDSKEWOUT (test_ddr2pl_dcd_skewout),
		.TESTPL2DDRDCDSAMPLEPULSE (test_pl2ddr_dcd_sample_pulse),
		.TESTBSCANENN (test_bscan_en_n),
		.TESTBSCANTDI (test_bscan_tdi),
		.TESTBSCANUPDATEDR (test_bscan_updatedr),
		.TESTBSCANSHIFTDR (test_bscan_shiftdr),
		.TESTBSCANRESETTAPB (test_bscan_reset_tap_b),
		.TESTBSCANMISRJTAGLOAD (test_bscan_misr_jtag_load),
		.TESTBSCANINTEST (test_bscan_intest),
		.TESTBSCANEXTEST (test_bscan_extest),
		.TESTBSCANCLOCKDR (test_bscan_clockdr),
		.TESTBSCANACMODE (test_bscan_ac_mode),
		.TESTBSCANACTEST (test_bscan_ac_test),
		.TESTBSCANINITMEMORY (test_bscan_init_memory),
		.TESTBSCANMODEC (test_bscan_mode_c),
		.TESTBSCANTDO (test_bscan_tdo),
		.IDBGL0TXCLK (i_dbg_l0_txclk),
		.IDBGL0RXCLK (i_dbg_l0_rxclk),
		.IDBGL1TXCLK (i_dbg_l1_txclk),
		.IDBGL1RXCLK (i_dbg_l1_rxclk),
		.IDBGL2TXCLK (i_dbg_l2_txclk),
		.IDBGL2RXCLK (i_dbg_l2_rxclk),
		.IDBGL3TXCLK (i_dbg_l3_txclk),
		.IDBGL3RXCLK (i_dbg_l3_rxclk),
		.IAFERXSYMBOLCLKBY2PL (i_afe_rx_symbol_clk_by_2_pl),
		.PLFPDSPARE0IN (pl_fpd_spare_0_in),
		.PLFPDSPARE1IN (pl_fpd_spare_1_in),
		.PLFPDSPARE2IN (pl_fpd_spare_2_in),
		.PLFPDSPARE3IN (pl_fpd_spare_3_in),
		.PLFPDSPARE4IN (pl_fpd_spare_4_in),
		.FPDPLSPARE0OUT (fpd_pl_spare_0_out),
		.FPDPLSPARE1OUT (fpd_pl_spare_1_out),
		.FPDPLSPARE2OUT (fpd_pl_spare_2_out),
		.FPDPLSPARE3OUT (fpd_pl_spare_3_out),
		.FPDPLSPARE4OUT (fpd_pl_spare_4_out),
		.PLLPDSPARE0IN (pl_lpd_spare_0_in),
		.PLLPDSPARE1IN (pl_lpd_spare_1_in),
		.PLLPDSPARE2IN (pl_lpd_spare_2_in),
		.PLLPDSPARE3IN (pl_lpd_spare_3_in),
		.PLLPDSPARE4IN (pl_lpd_spare_4_in),
		.LPDPLSPARE0OUT (lpd_pl_spare_0_out),
		.LPDPLSPARE1OUT (lpd_pl_spare_1_out),
		.LPDPLSPARE2OUT (lpd_pl_spare_2_out),
		.LPDPLSPARE3OUT (lpd_pl_spare_3_out),
		.LPDPLSPARE4OUT (lpd_pl_spare_4_out),
		.ODBGL0PHYSTATUS (o_dbg_l0_phystatus),
		.ODBGL0RXDATA (o_dbg_l0_rxdata),
		.ODBGL0RXDATAK (o_dbg_l0_rxdatak),
		.ODBGL0RXVALID (o_dbg_l0_rxvalid),
		.ODBGL0RXSTATUS (o_dbg_l0_rxstatus),
		.ODBGL0RXELECIDLE (o_dbg_l0_rxelecidle),
		.ODBGL0RSTB (o_dbg_l0_rstb),
		.ODBGL0TXDATA (o_dbg_l0_txdata),
		.ODBGL0TXDATAK (o_dbg_l0_txdatak),
		.ODBGL0RATE (o_dbg_l0_rate),
		.ODBGL0POWERDOWN (o_dbg_l0_powerdown),
		.ODBGL0TXELECIDLE (o_dbg_l0_txelecidle),
		.ODBGL0TXDETRXLPBACK (o_dbg_l0_txdetrx_lpback),
		.ODBGL0RXPOLARITY (o_dbg_l0_rxpolarity),
		.ODBGL0TXSGMIIEWRAP (o_dbg_l0_tx_sgmii_ewrap),
		.ODBGL0RXSGMIIENCDET (o_dbg_l0_rx_sgmii_en_cdet),
		.ODBGL0SATACORERXDATA (o_dbg_l0_sata_corerxdata),
		.ODBGL0SATACORERXDATAVALID (o_dbg_l0_sata_corerxdatavalid),
		.ODBGL0SATACOREREADY (o_dbg_l0_sata_coreready),
		.ODBGL0SATACORECLOCKREADY (o_dbg_l0_sata_coreclockready),
		.ODBGL0SATACORERXSIGNALDET (o_dbg_l0_sata_corerxsignaldet),
		.ODBGL0SATAPHYCTRLTXDATA (o_dbg_l0_sata_phyctrltxdata),
		.ODBGL0SATAPHYCTRLTXIDLE (o_dbg_l0_sata_phyctrltxidle),
		.ODBGL0SATAPHYCTRLTXRATE (o_dbg_l0_sata_phyctrltxrate),
		.ODBGL0SATAPHYCTRLRXRATE (o_dbg_l0_sata_phyctrlrxrate),
		.ODBGL0SATAPHYCTRLTXRST (o_dbg_l0_sata_phyctrltxrst),
		.ODBGL0SATAPHYCTRLRXRST (o_dbg_l0_sata_phyctrlrxrst),
		.ODBGL0SATAPHYCTRLRESET (o_dbg_l0_sata_phyctrlreset),
		.ODBGL0SATAPHYCTRLPARTIAL (o_dbg_l0_sata_phyctrlpartial),
		.ODBGL0SATAPHYCTRLSLUMBER (o_dbg_l0_sata_phyctrlslumber),
		.ODBGL1PHYSTATUS (o_dbg_l1_phystatus),
		.ODBGL1RXDATA (o_dbg_l1_rxdata),
		.ODBGL1RXDATAK (o_dbg_l1_rxdatak),
		.ODBGL1RXVALID (o_dbg_l1_rxvalid),
		.ODBGL1RXSTATUS (o_dbg_l1_rxstatus),
		.ODBGL1RXELECIDLE (o_dbg_l1_rxelecidle),
		.ODBGL1RSTB (o_dbg_l1_rstb),
		.ODBGL1TXDATA (o_dbg_l1_txdata),
		.ODBGL1TXDATAK (o_dbg_l1_txdatak),
		.ODBGL1RATE (o_dbg_l1_rate),
		.ODBGL1POWERDOWN (o_dbg_l1_powerdown),
		.ODBGL1TXELECIDLE (o_dbg_l1_txelecidle),
		.ODBGL1TXDETRXLPBACK (o_dbg_l1_txdetrx_lpback),
		.ODBGL1RXPOLARITY (o_dbg_l1_rxpolarity),
		.ODBGL1TXSGMIIEWRAP (o_dbg_l1_tx_sgmii_ewrap),
		.ODBGL1RXSGMIIENCDET (o_dbg_l1_rx_sgmii_en_cdet),
		.ODBGL1SATACORERXDATA (o_dbg_l1_sata_corerxdata),
		.ODBGL1SATACORERXDATAVALID (o_dbg_l1_sata_corerxdatavalid),
		.ODBGL1SATACOREREADY (o_dbg_l1_sata_coreready),
		.ODBGL1SATACORECLOCKREADY (o_dbg_l1_sata_coreclockready),
		.ODBGL1SATACORERXSIGNALDET (o_dbg_l1_sata_corerxsignaldet),
		.ODBGL1SATAPHYCTRLTXDATA (o_dbg_l1_sata_phyctrltxdata),
		.ODBGL1SATAPHYCTRLTXIDLE (o_dbg_l1_sata_phyctrltxidle),
		.ODBGL1SATAPHYCTRLTXRATE (o_dbg_l1_sata_phyctrltxrate),
		.ODBGL1SATAPHYCTRLRXRATE (o_dbg_l1_sata_phyctrlrxrate),
		.ODBGL1SATAPHYCTRLTXRST (o_dbg_l1_sata_phyctrltxrst),
		.ODBGL1SATAPHYCTRLRXRST (o_dbg_l1_sata_phyctrlrxrst),
		.ODBGL1SATAPHYCTRLRESET (o_dbg_l1_sata_phyctrlreset),
		.ODBGL1SATAPHYCTRLPARTIAL (o_dbg_l1_sata_phyctrlpartial),
		.ODBGL1SATAPHYCTRLSLUMBER (o_dbg_l1_sata_phyctrlslumber),
		.ODBGL2PHYSTATUS (o_dbg_l2_phystatus),
		.ODBGL2RXDATA (o_dbg_l2_rxdata),
		.ODBGL2RXDATAK (o_dbg_l2_rxdatak),
		.ODBGL2RXVALID (o_dbg_l2_rxvalid),
		.ODBGL2RXSTATUS (o_dbg_l2_rxstatus),
		.ODBGL2RXELECIDLE (o_dbg_l2_rxelecidle),
		.ODBGL2RSTB (o_dbg_l2_rstb),
		.ODBGL2TXDATA (o_dbg_l2_txdata),
		.ODBGL2TXDATAK (o_dbg_l2_txdatak),
		.ODBGL2RATE (o_dbg_l2_rate),
		.ODBGL2POWERDOWN (o_dbg_l2_powerdown),
		.ODBGL2TXELECIDLE (o_dbg_l2_txelecidle),
		.ODBGL2TXDETRXLPBACK (o_dbg_l2_txdetrx_lpback),
		.ODBGL2RXPOLARITY (o_dbg_l2_rxpolarity),
		.ODBGL2TXSGMIIEWRAP (o_dbg_l2_tx_sgmii_ewrap),
		.ODBGL2RXSGMIIENCDET (o_dbg_l2_rx_sgmii_en_cdet),
		.ODBGL2SATACORERXDATA (o_dbg_l2_sata_corerxdata),
		.ODBGL2SATACORERXDATAVALID (o_dbg_l2_sata_corerxdatavalid),
		.ODBGL2SATACOREREADY (o_dbg_l2_sata_coreready),
		.ODBGL2SATACORECLOCKREADY (o_dbg_l2_sata_coreclockready),
		.ODBGL2SATACORERXSIGNALDET (o_dbg_l2_sata_corerxsignaldet),
		.ODBGL2SATAPHYCTRLTXDATA (o_dbg_l2_sata_phyctrltxdata),
		.ODBGL2SATAPHYCTRLTXIDLE (o_dbg_l2_sata_phyctrltxidle),
		.ODBGL2SATAPHYCTRLTXRATE (o_dbg_l2_sata_phyctrltxrate),
		.ODBGL2SATAPHYCTRLRXRATE (o_dbg_l2_sata_phyctrlrxrate),
		.ODBGL2SATAPHYCTRLTXRST (o_dbg_l2_sata_phyctrltxrst),
		.ODBGL2SATAPHYCTRLRXRST (o_dbg_l2_sata_phyctrlrxrst),
		.ODBGL2SATAPHYCTRLRESET (o_dbg_l2_sata_phyctrlreset),
		.ODBGL2SATAPHYCTRLPARTIAL (o_dbg_l2_sata_phyctrlpartial),
		.ODBGL2SATAPHYCTRLSLUMBER (o_dbg_l2_sata_phyctrlslumber),
		.ODBGL3PHYSTATUS (o_dbg_l3_phystatus),
		.ODBGL3RXDATA (o_dbg_l3_rxdata),
		.ODBGL3RXDATAK (o_dbg_l3_rxdatak),
		.ODBGL3RXVALID (o_dbg_l3_rxvalid),
		.ODBGL3RXSTATUS (o_dbg_l3_rxstatus),
		.ODBGL3RXELECIDLE (o_dbg_l3_rxelecidle),
		.ODBGL3RSTB (o_dbg_l3_rstb),
		.ODBGL3TXDATA (o_dbg_l3_txdata),
		.ODBGL3TXDATAK (o_dbg_l3_txdatak),
		.ODBGL3RATE (o_dbg_l3_rate),
		.ODBGL3POWERDOWN (o_dbg_l3_powerdown),
		.ODBGL3TXELECIDLE (o_dbg_l3_txelecidle),
		.ODBGL3TXDETRXLPBACK (o_dbg_l3_txdetrx_lpback),
		.ODBGL3RXPOLARITY (o_dbg_l3_rxpolarity),
		.ODBGL3TXSGMIIEWRAP (o_dbg_l3_tx_sgmii_ewrap),
		.ODBGL3RXSGMIIENCDET (o_dbg_l3_rx_sgmii_en_cdet),
		.ODBGL3SATACORERXDATA (o_dbg_l3_sata_corerxdata),
		.ODBGL3SATACORERXDATAVALID (o_dbg_l3_sata_corerxdatavalid),
		.ODBGL3SATACOREREADY (o_dbg_l3_sata_coreready),
		.ODBGL3SATACORECLOCKREADY (o_dbg_l3_sata_coreclockready),
		.ODBGL3SATACORERXSIGNALDET (o_dbg_l3_sata_corerxsignaldet),
		.ODBGL3SATAPHYCTRLTXDATA (o_dbg_l3_sata_phyctrltxdata),
		.ODBGL3SATAPHYCTRLTXIDLE (o_dbg_l3_sata_phyctrltxidle),
		.ODBGL3SATAPHYCTRLTXRATE (o_dbg_l3_sata_phyctrltxrate),
		.ODBGL3SATAPHYCTRLRXRATE (o_dbg_l3_sata_phyctrlrxrate),
		.ODBGL3SATAPHYCTRLTXRST (o_dbg_l3_sata_phyctrltxrst),
		.ODBGL3SATAPHYCTRLRXRST (o_dbg_l3_sata_phyctrlrxrst),
		.ODBGL3SATAPHYCTRLRESET (o_dbg_l3_sata_phyctrlreset),
		.ODBGL3SATAPHYCTRLPARTIAL (o_dbg_l3_sata_phyctrlpartial),
		.ODBGL3SATAPHYCTRLSLUMBER (o_dbg_l3_sata_phyctrlslumber),
		.DBGPATHFIFOBYPASS (dbg_path_fifo_bypass),
		.IAFEPLLPDHSCLOCKR (i_afe_pll_pd_hs_clock_r),
		.IAFEMODE (i_afe_mode),
		.IBGCALAFEMODE (i_bgcal_afe_mode),
		.OAFECMNCALIBCOMPOUT (o_afe_cmn_calib_comp_out),
		.IAFECMNBGENABLELOWLEAKAGE (i_afe_cmn_bg_enable_low_leakage),
		.IAFECMNBGISOCTRLBAR (i_afe_cmn_bg_iso_ctrl_bar),
		.IAFECMNBGPD (i_afe_cmn_bg_pd),
		.IAFECMNBGPDBGOK (i_afe_cmn_bg_pd_bg_ok),
		.IAFECMNBGPDPTAT (i_afe_cmn_bg_pd_ptat),
		.IAFECMNCALIBENICONST (i_afe_cmn_calib_en_iconst),
		.IAFECMNCALIBENABLELOWLEAKAGE (i_afe_cmn_calib_enable_low_leakage),
		.IAFECMNCALIBISOCTRLBAR (i_afe_cmn_calib_iso_ctrl_bar),
		.OAFEPLLDCOCOUNT (o_afe_pll_dco_count),
		.OAFEPLLCLKSYMHS (o_afe_pll_clk_sym_hs),
		.OAFEPLLFBCLKFRAC (o_afe_pll_fbclk_frac),
		.OAFERXPIPELFPSBCNRXELECIDLE (o_afe_rx_pipe_lfpsbcn_rxelecidle),
		.OAFERXPIPESIGDET (o_afe_rx_pipe_sigdet),
		.OAFERXSYMBOL (o_afe_rx_symbol),
		.OAFERXSYMBOLCLKBY2 (o_afe_rx_symbol_clk_by_2),
		.OAFERXUPHYSAVECALCODE (o_afe_rx_uphy_save_calcode),
		.OAFERXUPHYSTARTLOOPBUF (o_afe_rx_uphy_startloop_buf),
		.OAFERXUPHYRXCALIBDONE (o_afe_rx_uphy_rx_calib_done),
		.IAFERXRXPMARSTB (i_afe_rx_rxpma_rstb),
		.IAFERXUPHYRESTORECALCODEDATA (i_afe_rx_uphy_restore_calcode_data),
		.IAFERXPIPERXEQTRAINING (i_afe_rx_pipe_rxeqtraining),
		.IAFERXISOHSRXCTRLBAR (i_afe_rx_iso_hsrx_ctrl_bar),
		.IAFERXISOLFPSCTRLBAR (i_afe_rx_iso_lfps_ctrl_bar),
		.IAFERXISOSIGDETCTRLBAR (i_afe_rx_iso_sigdet_ctrl_bar),
		.IAFERXHSRXCLOCKSTOPREQ (i_afe_rx_hsrx_clock_stop_req),
		.OAFERXUPHYSAVECALCODEDATA (o_afe_rx_uphy_save_calcode_data),
		.OAFERXHSRXCLOCKSTOPACK (o_afe_rx_hsrx_clock_stop_ack),
		.OAFEPGAVDDCR (o_afe_pg_avddcr),
		.OAFEPGAVDDIO (o_afe_pg_avddio),
		.OAFEPGDVDDCR (o_afe_pg_dvddcr),
		.OAFEPGSTATICAVDDCR (o_afe_pg_static_avddcr),
		.OAFEPGSTATICAVDDIO (o_afe_pg_static_avddio),
		.IPLLAFEMODE (i_pll_afe_mode),
		.IAFEPLLCOARSECODE (i_afe_pll_coarse_code),
		.IAFEPLLENCLOCKHSDIV2 (i_afe_pll_en_clock_hs_div2),
		.IAFEPLLFBDIV (i_afe_pll_fbdiv),
		.IAFEPLLLOADFBDIV (i_afe_pll_load_fbdiv),
		.IAFEPLLPD (i_afe_pll_pd),
		.IAFEPLLPDPFD (i_afe_pll_pd_pfd),
		.IAFEPLLRSTFDBKDIV (i_afe_pll_rst_fdbk_div),
		.IAFEPLLSTARTLOOP (i_afe_pll_startloop),
		.IAFEPLLV2ICODE (i_afe_pll_v2i_code),
		.IAFEPLLV2IPROG (i_afe_pll_v2i_prog),
		.IAFEPLLVCOCNTWINDOW (i_afe_pll_vco_cnt_window),
		.IAFERXMPHYGATESYMBOLCLK (i_afe_rx_mphy_gate_symbol_clk),
		.IAFERXMPHYMUXHSBLS (i_afe_rx_mphy_mux_hsb_ls),
		.IAFERXPIPERXTERMENABLE (i_afe_rx_pipe_rx_term_enable),
		.IAFERXUPHYBIASGENICONSTCOREMIRRORENABLE (i_afe_rx_uphy_biasgen_iconst_core_mirror_enable),
		.IAFERXUPHYBIASGENICONSTIOMIRRORENABLE (i_afe_rx_uphy_biasgen_iconst_io_mirror_enable),
		.IAFERXUPHYBIASGENIRCONSTCOREMIRRORENABLE (i_afe_rx_uphy_biasgen_irconst_core_mirror_enable),
		.IAFERXUPHYENABLECDR (i_afe_rx_uphy_enable_cdr),
		.IAFERXUPHYENABLELOWLEAKAGE (i_afe_rx_uphy_enable_low_leakage),
		.IAFERXRXPMAREFCLKDIG (i_afe_rx_rxpma_refclk_dig),
		.IAFERXUPHYHSRXRSTB (i_afe_rx_uphy_hsrx_rstb),
		.IAFERXUPHYPDNHSDES (i_afe_rx_uphy_pdn_hs_des),
		.IAFERXUPHYPDSAMPC2C (i_afe_rx_uphy_pd_samp_c2c),
		.IAFERXUPHYPDSAMPC2CECLK (i_afe_rx_uphy_pd_samp_c2c_eclk),
		.IAFERXUPHYPSOCLKLANE (i_afe_rx_uphy_pso_clk_lane),
		.IAFERXUPHYPSOEQ (i_afe_rx_uphy_pso_eq),
		.IAFERXUPHYPSOHSRXDIG (i_afe_rx_uphy_pso_hsrxdig),
		.IAFERXUPHYPSOIQPI (i_afe_rx_uphy_pso_iqpi),
		.IAFERXUPHYPSOLFPSBCN (i_afe_rx_uphy_pso_lfpsbcn),
		.IAFERXUPHYPSOSAMPFLOPS (i_afe_rx_uphy_pso_samp_flops),
		.IAFERXUPHYPSOSIGDET (i_afe_rx_uphy_pso_sigdet),
		.IAFERXUPHYRESTORECALCODE (i_afe_rx_uphy_restore_calcode),
		.IAFERXUPHYRUNCALIB (i_afe_rx_uphy_run_calib),
		.IAFERXUPHYRXLANEPOLARITYSWAP (i_afe_rx_uphy_rx_lane_polarity_swap),
		.IAFERXUPHYSTARTLOOPPLL (i_afe_rx_uphy_startloop_pll),
		.IAFERXUPHYHSCLKDIVISIONFACTOR (i_afe_rx_uphy_hsclk_division_factor),
		.IAFERXUPHYRXPMAOPMODE (i_afe_rx_uphy_rx_pma_opmode),
		.IAFETXENABLEHSCLKDIVISION (i_afe_tx_enable_hsclk_division),
		.IAFETXENABLELDO (i_afe_tx_enable_ldo),
		.IAFETXENABLEREF (i_afe_tx_enable_ref),
		.IAFETXENABLESUPPLYHSCLK (i_afe_tx_enable_supply_hsclk),
		.IAFETXENABLESUPPLYPIPE (i_afe_tx_enable_supply_pipe),
		.IAFETXENABLESUPPLYSERIALIZER (i_afe_tx_enable_supply_serializer),
		.IAFETXENABLESUPPLYUPHY (i_afe_tx_enable_supply_uphy),
		.IAFETXHSSERRSTB (i_afe_tx_hs_ser_rstb),
		.IAFETXHSSYMBOL (i_afe_tx_hs_symbol),
		.IAFETXMPHYTXLSDATA (i_afe_tx_mphy_tx_ls_data),
		.IAFETXPIPETXENABLEIDLEMODE (i_afe_tx_pipe_tx_enable_idle_mode),
		.IAFETXPIPETXENABLELFPS (i_afe_tx_pipe_tx_enable_lfps),
		.IAFETXPIPETXENABLERXDET (i_afe_tx_pipe_tx_enable_rxdet),
		.IAFETXUPHYTXPMAOPMODE (i_afe_TX_uphy_txpma_opmode),
		.IAFETXPMADIGDIGITALRESETN (i_afe_TX_pmadig_digital_reset_n),
		.IAFETXSERIALIZERRSTREL (i_afe_TX_serializer_rst_rel),
		.IAFETXPLLSYMBCLK2 (i_afe_TX_pll_symb_clk_2),
		.IAFETXANAIFRATE (i_afe_TX_ana_if_rate),
		.IAFETXENDIGSUBLPMODE (i_afe_TX_en_dig_sublp_mode),
		.IAFETXLPBKSEL (i_afe_TX_LPBK_SEL),
		.IAFETXISOCTRLBAR (i_afe_TX_iso_ctrl_bar),
		.IAFETXSERISOCTRLBAR (i_afe_TX_ser_iso_ctrl_bar),
		.IAFETXLFPSCLK (i_afe_TX_lfps_clk),
		.IAFETXSERIALIZERRSTB (i_afe_TX_serializer_rstb),
		.OAFETXDIGRESETRELACK (o_afe_TX_dig_reset_rel_ack),
		.OAFETXPIPETXDNRXDET (o_afe_TX_pipe_TX_dn_rxdet),
		.OAFETXPIPETXDPRXDET (o_afe_TX_pipe_TX_dp_rxdet),
		.IAFETXPIPETXFASTESTCOMMONMODE (i_afe_tx_pipe_tx_fast_est_common_mode),

			.ODBGL0TXCLK(o_dbg_l0_txclk), .ODBGL0RXCLK(o_dbg_l0_rxclk),
			.ODBGL1TXCLK(o_dbg_l1_txclk), .ODBGL1RXCLK(o_dbg_l1_rxclk),
			.ODBGL2TXCLK(o_dbg_l2_txclk), .ODBGL2RXCLK(o_dbg_l2_rxclk),
			.ODBGL3TXCLK(o_dbg_l3_txclk), .ODBGL3RXCLK(o_dbg_l3_rxclk),

			.DPVIDEOREFCLK(dp_video_ref_clk_i),
			.DPAUDIOREFCLK(dp_audio_ref_clk_i)
		);
	end else begin
		PS8 PS8_i(
			// Master AXI General Purpose 0
			// set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {0}] [get_bd_cells zynq_ultra_ps_e]
			.MAXIGP0ACLK(maxihpm0_fpd_aclk), .MAXIGP0AWID(maxigp0_awid), .MAXIGP0AWADDR(maxigp0_awaddr), .MAXIGP0AWLEN(maxigp0_awlen), .MAXIGP0AWSIZE(maxigp0_awsize), .MAXIGP0AWBURST(maxigp0_awburst), .MAXIGP0AWLOCK(maxigp0_awlock), .MAXIGP0AWCACHE(maxigp0_awcache), .MAXIGP0AWPROT(maxigp0_awprot), .MAXIGP0AWVALID(maxigp0_awvalid), .MAXIGP0AWUSER(maxigp0_awuser), .MAXIGP0AWREADY(maxigp0_awready), .MAXIGP0WDATA(maxigp0_wdata_i), .MAXIGP0WSTRB(maxigp0_wstrb_i), .MAXIGP0WLAST(maxigp0_wlast), .MAXIGP0WVALID(maxigp0_wvalid), .MAXIGP0WREADY(maxigp0_wready), .MAXIGP0BID(maxigp0_bid), .MAXIGP0BRESP(maxigp0_bresp), .MAXIGP0BVALID(maxigp0_bvalid), .MAXIGP0BREADY(maxigp0_bready), .MAXIGP0ARID(maxigp0_arid), .MAXIGP0ARADDR(maxigp0_araddr), .MAXIGP0ARLEN(maxigp0_arlen), .MAXIGP0ARSIZE(maxigp0_arsize), .MAXIGP0ARBURST(maxigp0_arburst), .MAXIGP0ARLOCK(maxigp0_arlock), .MAXIGP0ARCACHE(maxigp0_arcache), .MAXIGP0ARPROT(maxigp0_arprot), .MAXIGP0ARVALID(maxigp0_arvalid), .MAXIGP0ARUSER(maxigp0_aruser), .MAXIGP0ARREADY(maxigp0_arready), .MAXIGP0RID(maxigp0_rid), .MAXIGP0RDATA(maxigp0_rdata_i), .MAXIGP0RRESP(maxigp0_rresp), .MAXIGP0RLAST(maxigp0_rlast), .MAXIGP0RVALID(maxigp0_rvalid), .MAXIGP0RREADY(maxigp0_rready), .MAXIGP0AWQOS(maxigp0_awqos), .MAXIGP0ARQOS(maxigp0_arqos),

			// Master AXI General Purpose 1
			// set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ultra_ps_e]
			.MAXIGP1ACLK(maxihpm1_fpd_aclk), .MAXIGP1AWID(maxigp1_awid), .MAXIGP1AWADDR(maxigp1_awaddr), .MAXIGP1AWLEN(maxigp1_awlen), .MAXIGP1AWSIZE(maxigp1_awsize), .MAXIGP1AWBURST(maxigp1_awburst), .MAXIGP1AWLOCK(maxigp1_awlock), .MAXIGP1AWCACHE(maxigp1_awcache), .MAXIGP1AWPROT(maxigp1_awprot), .MAXIGP1AWVALID(maxigp1_awvalid), .MAXIGP1AWUSER(maxigp1_awuser), .MAXIGP1AWREADY(maxigp1_awready), .MAXIGP1WDATA(maxigp1_wdata_i), .MAXIGP1WSTRB(maxigp1_wstrb_i), .MAXIGP1WLAST(maxigp1_wlast), .MAXIGP1WVALID(maxigp1_wvalid), .MAXIGP1WREADY(maxigp1_wready), .MAXIGP1BID(maxigp1_bid), .MAXIGP1BRESP(maxigp1_bresp), .MAXIGP1BVALID(maxigp1_bvalid), .MAXIGP1BREADY(maxigp1_bready), .MAXIGP1ARID(maxigp1_arid), .MAXIGP1ARADDR(maxigp1_araddr), .MAXIGP1ARLEN(maxigp1_arlen), .MAXIGP1ARSIZE(maxigp1_arsize), .MAXIGP1ARBURST(maxigp1_arburst), .MAXIGP1ARLOCK(maxigp1_arlock), .MAXIGP1ARCACHE(maxigp1_arcache), .MAXIGP1ARPROT(maxigp1_arprot), .MAXIGP1ARVALID(maxigp1_arvalid), .MAXIGP1ARUSER(maxigp1_aruser), .MAXIGP1ARREADY(maxigp1_arready), .MAXIGP1RID(maxigp1_rid), .MAXIGP1RDATA(maxigp1_rdata_i), .MAXIGP1RRESP(maxigp1_rresp), .MAXIGP1RLAST(maxigp1_rlast), .MAXIGP1RVALID(maxigp1_rvalid), .MAXIGP1RREADY(maxigp1_rready), .MAXIGP1AWQOS(maxigp1_awqos), .MAXIGP1ARQOS(maxigp1_arqos),

			// Master AXI General Purpose 2
			// set_property -dict [list CONFIG.PSU__USE__M_AXI_GP2 {0}] [get_bd_cells zynq_ultra_ps_e]
			.MAXIGP2ACLK(maxihpm0_lpd_aclk), .MAXIGP2AWID(maxigp2_awid), .MAXIGP2AWADDR(maxigp2_awaddr), .MAXIGP2AWLEN(maxigp2_awlen), .MAXIGP2AWSIZE(maxigp2_awsize), .MAXIGP2AWBURST(maxigp2_awburst), .MAXIGP2AWLOCK(maxigp2_awlock), .MAXIGP2AWCACHE(maxigp2_awcache), .MAXIGP2AWPROT(maxigp2_awprot), .MAXIGP2AWVALID(maxigp2_awvalid), .MAXIGP2AWUSER(maxigp2_awuser), .MAXIGP2AWREADY(maxigp2_awready), .MAXIGP2WDATA(maxigp2_wdata_i), .MAXIGP2WSTRB(maxigp2_wstrb_i), .MAXIGP2WLAST(maxigp2_wlast), .MAXIGP2WVALID(maxigp2_wvalid), .MAXIGP2WREADY(maxigp2_wready), .MAXIGP2BID(maxigp2_bid), .MAXIGP2BRESP(maxigp2_bresp), .MAXIGP2BVALID(maxigp2_bvalid), .MAXIGP2BREADY(maxigp2_bready), .MAXIGP2ARID(maxigp2_arid), .MAXIGP2ARADDR(maxigp2_araddr), .MAXIGP2ARLEN(maxigp2_arlen), .MAXIGP2ARSIZE(maxigp2_arsize), .MAXIGP2ARBURST(maxigp2_arburst), .MAXIGP2ARLOCK(maxigp2_arlock), .MAXIGP2ARCACHE(maxigp2_arcache), .MAXIGP2ARPROT(maxigp2_arprot), .MAXIGP2ARVALID(maxigp2_arvalid), .MAXIGP2ARUSER(maxigp2_aruser), .MAXIGP2ARREADY(maxigp2_arready), .MAXIGP2RID(maxigp2_rid), .MAXIGP2RDATA(maxigp2_rdata_i), .MAXIGP2RRESP(maxigp2_rresp), .MAXIGP2RLAST(maxigp2_rlast), .MAXIGP2RVALID(maxigp2_rvalid), .MAXIGP2RREADY(maxigp2_rready), .MAXIGP2AWQOS(maxigp2_awqos), .MAXIGP2ARQOS(maxigp2_arqos),

			// Slave AXI General Purpose 0
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP0 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP0RCLK(saxihpc0_fpd_rclk_temp), .SAXIGP0WCLK(saxihpc0_fpd_wclk_temp), .SAXIGP0ARUSER(saxigp0_aruser), .SAXIGP0AWUSER(saxigp0_awuser), .SAXIGP0AWID(saxigp0_awid), .SAXIGP0AWADDR(saxigp0_awaddr), .SAXIGP0AWLEN(saxigp0_awlen), .SAXIGP0AWSIZE(saxigp0_awsize), .SAXIGP0AWBURST(saxigp0_awburst), .SAXIGP0AWLOCK(saxigp0_awlock), .SAXIGP0AWCACHE(saxigp0_awcache), .SAXIGP0AWPROT(saxigp0_awprot), .SAXIGP0AWVALID(saxigp0_awvalid), .SAXIGP0AWREADY(saxigp0_awready), .SAXIGP0WDATA(saxigp0_wdata_i), .SAXIGP0WSTRB(saxigp0_wstrb_i), .SAXIGP0WLAST(saxigp0_wlast), .SAXIGP0WVALID(saxigp0_wvalid), .SAXIGP0WREADY(saxigp0_wready), .SAXIGP0BID(saxigp0_bid), .SAXIGP0BRESP(saxigp0_bresp), .SAXIGP0BVALID(saxigp0_bvalid), .SAXIGP0BREADY(saxigp0_bready), .SAXIGP0ARID(saxigp0_arid), .SAXIGP0ARADDR(saxigp0_araddr), .SAXIGP0ARLEN(saxigp0_arlen), .SAXIGP0ARSIZE(saxigp0_arsize), .SAXIGP0ARBURST(saxigp0_arburst), .SAXIGP0ARLOCK(saxigp0_arlock), .SAXIGP0ARCACHE(saxigp0_arcache), .SAXIGP0ARPROT(saxigp0_arprot), .SAXIGP0ARVALID(saxigp0_arvalid), .SAXIGP0ARREADY(saxigp0_arready), .SAXIGP0RID(saxigp0_rid), .SAXIGP0RDATA(saxigp0_rdata_i), .SAXIGP0RRESP(saxigp0_rresp), .SAXIGP0RLAST(saxigp0_rlast), .SAXIGP0RVALID(saxigp0_rvalid), .SAXIGP0RREADY(saxigp0_rready), .SAXIGP0AWQOS(saxigp0_awqos), .SAXIGP0ARQOS(saxigp0_arqos), .SAXIGP0RCOUNT(saxigp0_rcount), .SAXIGP0WCOUNT(saxigp0_wcount), .SAXIGP0RACOUNT(saxigp0_racount), .SAXIGP0WACOUNT(saxigp0_wacount),

			// Slave AXI General Purpose 1
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP1 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP1RCLK(saxihpc1_fpd_rclk_temp), .SAXIGP1WCLK(saxihpc1_fpd_wclk_temp), .SAXIGP1ARUSER(saxigp1_aruser), .SAXIGP1AWUSER(saxigp1_awuser), .SAXIGP1AWID(saxigp1_awid), .SAXIGP1AWADDR(saxigp1_awaddr), .SAXIGP1AWLEN(saxigp1_awlen), .SAXIGP1AWSIZE(saxigp1_awsize), .SAXIGP1AWBURST(saxigp1_awburst), .SAXIGP1AWLOCK(saxigp1_awlock), .SAXIGP1AWCACHE(saxigp1_awcache), .SAXIGP1AWPROT(saxigp1_awprot), .SAXIGP1AWVALID(saxigp1_awvalid), .SAXIGP1AWREADY(saxigp1_awready), .SAXIGP1WDATA(saxigp1_wdata_i), .SAXIGP1WSTRB(saxigp1_wstrb_i), .SAXIGP1WLAST(saxigp1_wlast), .SAXIGP1WVALID(saxigp1_wvalid), .SAXIGP1WREADY(saxigp1_wready), .SAXIGP1BID(saxigp1_bid), .SAXIGP1BRESP(saxigp1_bresp), .SAXIGP1BVALID(saxigp1_bvalid), .SAXIGP1BREADY(saxigp1_bready), .SAXIGP1ARID(saxigp1_arid), .SAXIGP1ARADDR(saxigp1_araddr), .SAXIGP1ARLEN(saxigp1_arlen), .SAXIGP1ARSIZE(saxigp1_arsize), .SAXIGP1ARBURST(saxigp1_arburst), .SAXIGP1ARLOCK(saxigp1_arlock), .SAXIGP1ARCACHE(saxigp1_arcache), .SAXIGP1ARPROT(saxigp1_arprot), .SAXIGP1ARVALID(saxigp1_arvalid), .SAXIGP1ARREADY(saxigp1_arready), .SAXIGP1RID(saxigp1_rid), .SAXIGP1RDATA(saxigp1_rdata_i), .SAXIGP1RRESP(saxigp1_rresp), .SAXIGP1RLAST(saxigp1_rlast), .SAXIGP1RVALID(saxigp1_rvalid), .SAXIGP1RREADY(saxigp1_rready), .SAXIGP1AWQOS(saxigp1_awqos), .SAXIGP1ARQOS(saxigp1_arqos), .SAXIGP1RCOUNT(saxigp1_rcount), .SAXIGP1WCOUNT(saxigp1_wcount), .SAXIGP1RACOUNT(saxigp1_racount), .SAXIGP1WACOUNT(saxigp1_wacount),

			// Slave AXI General Purpose 2
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP2RCLK(saxihp0_fpd_rclk_temp), .SAXIGP2WCLK(saxihp0_fpd_wclk_temp), .SAXIGP2ARUSER(saxigp2_aruser), .SAXIGP2AWUSER(saxigp2_awuser), .SAXIGP2AWID(saxigp2_awid), .SAXIGP2AWADDR(saxigp2_awaddr), .SAXIGP2AWLEN(saxigp2_awlen), .SAXIGP2AWSIZE(saxigp2_awsize), .SAXIGP2AWBURST(saxigp2_awburst), .SAXIGP2AWLOCK(saxigp2_awlock), .SAXIGP2AWCACHE(saxigp2_awcache), .SAXIGP2AWPROT(saxigp2_awprot), .SAXIGP2AWVALID(saxigp2_awvalid), .SAXIGP2AWREADY(saxigp2_awready), .SAXIGP2WDATA(saxigp2_wdata_i), .SAXIGP2WSTRB(saxigp2_wstrb_i), .SAXIGP2WLAST(saxigp2_wlast), .SAXIGP2WVALID(saxigp2_wvalid), .SAXIGP2WREADY(saxigp2_wready), .SAXIGP2BID(saxigp2_bid), .SAXIGP2BRESP(saxigp2_bresp), .SAXIGP2BVALID(saxigp2_bvalid), .SAXIGP2BREADY(saxigp2_bready), .SAXIGP2ARID(saxigp2_arid), .SAXIGP2ARADDR(saxigp2_araddr), .SAXIGP2ARLEN(saxigp2_arlen), .SAXIGP2ARSIZE(saxigp2_arsize), .SAXIGP2ARBURST(saxigp2_arburst), .SAXIGP2ARLOCK(saxigp2_arlock), .SAXIGP2ARCACHE(saxigp2_arcache), .SAXIGP2ARPROT(saxigp2_arprot), .SAXIGP2ARVALID(saxigp2_arvalid), .SAXIGP2ARREADY(saxigp2_arready), .SAXIGP2RID(saxigp2_rid), .SAXIGP2RDATA(saxigp2_rdata_i), .SAXIGP2RRESP(saxigp2_rresp), .SAXIGP2RLAST(saxigp2_rlast), .SAXIGP2RVALID(saxigp2_rvalid), .SAXIGP2RREADY(saxigp2_rready), .SAXIGP2AWQOS(saxigp2_awqos), .SAXIGP2ARQOS(saxigp2_arqos), .SAXIGP2RCOUNT(saxigp2_rcount), .SAXIGP2WCOUNT(saxigp2_wcount), .SAXIGP2RACOUNT(saxigp2_racount), .SAXIGP2WACOUNT(saxigp2_wacount),

			// Slave AXI General Purpose 3
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP3 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP3RCLK(saxihp1_fpd_rclk_temp), .SAXIGP3WCLK(saxihp1_fpd_wclk_temp), .SAXIGP3ARUSER(saxigp3_aruser), .SAXIGP3AWUSER(saxigp3_awuser), .SAXIGP3AWID(saxigp3_awid), .SAXIGP3AWADDR(saxigp3_awaddr), .SAXIGP3AWLEN(saxigp3_awlen), .SAXIGP3AWSIZE(saxigp3_awsize), .SAXIGP3AWBURST(saxigp3_awburst), .SAXIGP3AWLOCK(saxigp3_awlock), .SAXIGP3AWCACHE(saxigp3_awcache), .SAXIGP3AWPROT(saxigp3_awprot), .SAXIGP3AWVALID(saxigp3_awvalid), .SAXIGP3AWREADY(saxigp3_awready), .SAXIGP3WDATA(saxigp3_wdata_i), .SAXIGP3WSTRB(saxigp3_wstrb_i), .SAXIGP3WLAST(saxigp3_wlast), .SAXIGP3WVALID(saxigp3_wvalid), .SAXIGP3WREADY(saxigp3_wready), .SAXIGP3BID(saxigp3_bid), .SAXIGP3BRESP(saxigp3_bresp), .SAXIGP3BVALID(saxigp3_bvalid), .SAXIGP3BREADY(saxigp3_bready), .SAXIGP3ARID(saxigp3_arid), .SAXIGP3ARADDR(saxigp3_araddr), .SAXIGP3ARLEN(saxigp3_arlen), .SAXIGP3ARSIZE(saxigp3_arsize), .SAXIGP3ARBURST(saxigp3_arburst), .SAXIGP3ARLOCK(saxigp3_arlock), .SAXIGP3ARCACHE(saxigp3_arcache), .SAXIGP3ARPROT(saxigp3_arprot), .SAXIGP3ARVALID(saxigp3_arvalid), .SAXIGP3ARREADY(saxigp3_arready), .SAXIGP3RID(saxigp3_rid), .SAXIGP3RDATA(saxigp3_rdata_i), .SAXIGP3RRESP(saxigp3_rresp), .SAXIGP3RLAST(saxigp3_rlast), .SAXIGP3RVALID(saxigp3_rvalid), .SAXIGP3RREADY(saxigp3_rready), .SAXIGP3AWQOS(saxigp3_awqos), .SAXIGP3ARQOS(saxigp3_arqos), .SAXIGP3RCOUNT(saxigp3_rcount), .SAXIGP3WCOUNT(saxigp3_wcount), .SAXIGP3RACOUNT(saxigp3_racount), .SAXIGP3WACOUNT(saxigp3_wacount),

			// Slave AXI General Purpose 4
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP4 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP4RCLK(saxihp2_fpd_rclk_temp), .SAXIGP4WCLK(saxihp2_fpd_wclk_temp), .SAXIGP4ARUSER(saxigp4_aruser), .SAXIGP4AWUSER(saxigp4_awuser), .SAXIGP4AWID(saxigp4_awid), .SAXIGP4AWADDR(saxigp4_awaddr), .SAXIGP4AWLEN(saxigp4_awlen), .SAXIGP4AWSIZE(saxigp4_awsize), .SAXIGP4AWBURST(saxigp4_awburst), .SAXIGP4AWLOCK(saxigp4_awlock), .SAXIGP4AWCACHE(saxigp4_awcache), .SAXIGP4AWPROT(saxigp4_awprot), .SAXIGP4AWVALID(saxigp4_awvalid), .SAXIGP4AWREADY(saxigp4_awready), .SAXIGP4WDATA(saxigp4_wdata_i), .SAXIGP4WSTRB(saxigp4_wstrb_i), .SAXIGP4WLAST(saxigp4_wlast), .SAXIGP4WVALID(saxigp4_wvalid), .SAXIGP4WREADY(saxigp4_wready), .SAXIGP4BID(saxigp4_bid), .SAXIGP4BRESP(saxigp4_bresp), .SAXIGP4BVALID(saxigp4_bvalid), .SAXIGP4BREADY(saxigp4_bready), .SAXIGP4ARID(saxigp4_arid), .SAXIGP4ARADDR(saxigp4_araddr), .SAXIGP4ARLEN(saxigp4_arlen), .SAXIGP4ARSIZE(saxigp4_arsize), .SAXIGP4ARBURST(saxigp4_arburst), .SAXIGP4ARLOCK(saxigp4_arlock), .SAXIGP4ARCACHE(saxigp4_arcache), .SAXIGP4ARPROT(saxigp4_arprot), .SAXIGP4ARVALID(saxigp4_arvalid), .SAXIGP4ARREADY(saxigp4_arready), .SAXIGP4RID(saxigp4_rid), .SAXIGP4RDATA(saxigp4_rdata_i), .SAXIGP4RRESP(saxigp4_rresp), .SAXIGP4RLAST(saxigp4_rlast), .SAXIGP4RVALID(saxigp4_rvalid), .SAXIGP4RREADY(saxigp4_rready), .SAXIGP4AWQOS(saxigp4_awqos), .SAXIGP4ARQOS(saxigp4_arqos), .SAXIGP4RCOUNT(saxigp4_rcount), .SAXIGP4WCOUNT(saxigp4_wcount), .SAXIGP4RACOUNT(saxigp4_racount), .SAXIGP4WACOUNT(saxigp4_wacount),

			// Slave AXI General Purpose 5
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP5 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP5RCLK(saxihp3_fpd_rclk_temp), .SAXIGP5WCLK(saxihp3_fpd_wclk_temp), .SAXIGP5ARUSER(saxigp5_aruser), .SAXIGP5AWUSER(saxigp5_awuser), .SAXIGP5AWID(saxigp5_awid), .SAXIGP5AWADDR(saxigp5_awaddr), .SAXIGP5AWLEN(saxigp5_awlen), .SAXIGP5AWSIZE(saxigp5_awsize), .SAXIGP5AWBURST(saxigp5_awburst), .SAXIGP5AWLOCK(saxigp5_awlock), .SAXIGP5AWCACHE(saxigp5_awcache), .SAXIGP5AWPROT(saxigp5_awprot), .SAXIGP5AWVALID(saxigp5_awvalid), .SAXIGP5AWREADY(saxigp5_awready), .SAXIGP5WDATA(saxigp5_wdata_i), .SAXIGP5WSTRB(saxigp5_wstrb_i), .SAXIGP5WLAST(saxigp5_wlast), .SAXIGP5WVALID(saxigp5_wvalid), .SAXIGP5WREADY(saxigp5_wready), .SAXIGP5BID(saxigp5_bid), .SAXIGP5BRESP(saxigp5_bresp), .SAXIGP5BVALID(saxigp5_bvalid), .SAXIGP5BREADY(saxigp5_bready), .SAXIGP5ARID(saxigp5_arid), .SAXIGP5ARADDR(saxigp5_araddr), .SAXIGP5ARLEN(saxigp5_arlen), .SAXIGP5ARSIZE(saxigp5_arsize), .SAXIGP5ARBURST(saxigp5_arburst), .SAXIGP5ARLOCK(saxigp5_arlock), .SAXIGP5ARCACHE(saxigp5_arcache), .SAXIGP5ARPROT(saxigp5_arprot), .SAXIGP5ARVALID(saxigp5_arvalid), .SAXIGP5ARREADY(saxigp5_arready), .SAXIGP5RID(saxigp5_rid), .SAXIGP5RDATA(saxigp5_rdata_i), .SAXIGP5RRESP(saxigp5_rresp), .SAXIGP5RLAST(saxigp5_rlast), .SAXIGP5RVALID(saxigp5_rvalid), .SAXIGP5RREADY(saxigp5_rready), .SAXIGP5AWQOS(saxigp5_awqos), .SAXIGP5ARQOS(saxigp5_arqos), .SAXIGP5RCOUNT(saxigp5_rcount), .SAXIGP5WCOUNT(saxigp5_wcount), .SAXIGP5RACOUNT(saxigp5_racount), .SAXIGP5WACOUNT(saxigp5_wacount),

			// Slave AXI General Purpose 6
			// set_property -dict [list CONFIG.PSU__USE__S_AXI_GP6 {1}] [get_bd_cells zynq_ultra_ps_e]
			.SAXIGP6RCLK(saxi_lpd_rclk_temp), .SAXIGP6WCLK(saxi_lpd_wclk_temp), .SAXIGP6ARUSER(saxigp6_aruser), .SAXIGP6AWUSER(saxigp6_awuser), .SAXIGP6AWID(saxigp6_awid), .SAXIGP6AWADDR(saxigp6_awaddr), .SAXIGP6AWLEN(saxigp6_awlen), .SAXIGP6AWSIZE(saxigp6_awsize), .SAXIGP6AWBURST(saxigp6_awburst), .SAXIGP6AWLOCK(saxigp6_awlock), .SAXIGP6AWCACHE(saxigp6_awcache), .SAXIGP6AWPROT(saxigp6_awprot), .SAXIGP6AWVALID(saxigp6_awvalid), .SAXIGP6AWREADY(saxigp6_awready), .SAXIGP6WDATA(saxigp6_wdata_i), .SAXIGP6WSTRB(saxigp6_wstrb_i), .SAXIGP6WLAST(saxigp6_wlast), .SAXIGP6WVALID(saxigp6_wvalid), .SAXIGP6WREADY(saxigp6_wready), .SAXIGP6BID(saxigp6_bid), .SAXIGP6BRESP(saxigp6_bresp), .SAXIGP6BVALID(saxigp6_bvalid), .SAXIGP6BREADY(saxigp6_bready), .SAXIGP6ARID(saxigp6_arid), .SAXIGP6ARADDR(saxigp6_araddr), .SAXIGP6ARLEN(saxigp6_arlen), .SAXIGP6ARSIZE(saxigp6_arsize), .SAXIGP6ARBURST(saxigp6_arburst), .SAXIGP6ARLOCK(saxigp6_arlock), .SAXIGP6ARCACHE(saxigp6_arcache), .SAXIGP6ARPROT(saxigp6_arprot), .SAXIGP6ARVALID(saxigp6_arvalid), .SAXIGP6ARREADY(saxigp6_arready), .SAXIGP6RID(saxigp6_rid), .SAXIGP6RDATA(saxigp6_rdata_i), .SAXIGP6RRESP(saxigp6_rresp), .SAXIGP6RLAST(saxigp6_rlast), .SAXIGP6RVALID(saxigp6_rvalid), .SAXIGP6RREADY(saxigp6_rready), .SAXIGP6AWQOS(saxigp6_awqos), .SAXIGP6ARQOS(saxigp6_arqos), .SAXIGP6RCOUNT(saxigp6_rcount), .SAXIGP6WCOUNT(saxigp6_wcount), .SAXIGP6RACOUNT(saxigp6_racount), .SAXIGP6WACOUNT(saxigp6_wacount),

			// Slave AXI Accelerator Coherency Port
			// https://developer.arm.com/documentation/ddi0434/c/snoop-control-unit/about-the-scu/accelerator-coherency-port
			.SAXIACPACLK(saxiacp_fpd_aclk), .SAXIACPAWADDR(saxiacp_awaddr), .SAXIACPAWID(saxiacp_awid), .SAXIACPAWLEN(saxiacp_awlen), .SAXIACPAWSIZE(saxiacp_awsize), .SAXIACPAWBURST(saxiacp_awburst), .SAXIACPAWLOCK(saxiacp_awlock), .SAXIACPAWCACHE(saxiacp_awcache), .SAXIACPAWPROT(saxiacp_awprot), .SAXIACPAWVALID(saxiacp_awvalid), .SAXIACPAWREADY(saxiacp_awready), .SAXIACPAWUSER(saxiacp_awuser), .SAXIACPAWQOS(saxiacp_awqos), .SAXIACPWLAST(saxiacp_wlast), .SAXIACPWDATA(saxiacp_wdata), .SAXIACPWSTRB(saxiacp_wstrb), .SAXIACPWVALID(saxiacp_wvalid), .SAXIACPWREADY(saxiacp_wready), .SAXIACPBRESP(saxiacp_bresp), .SAXIACPBID(saxiacp_bid), .SAXIACPBVALID(saxiacp_bvalid), .SAXIACPBREADY(saxiacp_bready), .SAXIACPARADDR(saxiacp_araddr), .SAXIACPARID(saxiacp_arid), .SAXIACPARLEN(saxiacp_arlen), .SAXIACPARSIZE(saxiacp_arsize), .SAXIACPARBURST(saxiacp_arburst), .SAXIACPARLOCK(saxiacp_arlock), .SAXIACPARCACHE(saxiacp_arcache), .SAXIACPARPROT(saxiacp_arprot), .SAXIACPARVALID(saxiacp_arvalid), .SAXIACPARREADY(saxiacp_arready), .SAXIACPARUSER(saxiacp_aruser), .SAXIACPARQOS(saxiacp_arqos), .SAXIACPRID(saxiacp_rid), .SAXIACPRLAST(saxiacp_rlast), .SAXIACPRDATA(saxiacp_rdata), .SAXIACPRRESP(saxiacp_rresp), .SAXIACPRVALID(saxiacp_rvalid), .SAXIACPRREADY(saxiacp_rready),
			.PLACECLK (sacefpd_aclk),

			// Full Power Domain
			.SACEFPDAWVALID(sacefpd_awvalid), .SACEFPDAWREADY(sacefpd_awready), .SACEFPDAWID(sacefpd_awid), .SACEFPDAWADDR(sacefpd_awaddr), .SACEFPDAWREGION(sacefpd_awregion), .SACEFPDAWLEN(sacefpd_awlen), .SACEFPDAWSIZE(sacefpd_awsize), .SACEFPDAWBURST(sacefpd_awburst), .SACEFPDAWLOCK(sacefpd_awlock), .SACEFPDAWCACHE(sacefpd_awcache), .SACEFPDAWPROT(sacefpd_awprot), .SACEFPDAWDOMAIN(sacefpd_awdomain), .SACEFPDAWSNOOP(sacefpd_awsnoop), .SACEFPDAWBAR(sacefpd_awbar), .SACEFPDAWQOS(sacefpd_awqos), .SACEFPDAWUSER({6'b000000,4'b1111,sacefpd_awuser[5:0]}), .SACEFPDWVALID(sacefpd_wvalid), .SACEFPDWREADY(sacefpd_wready), .SACEFPDWDATA(sacefpd_wdata), .SACEFPDWSTRB(sacefpd_wstrb), .SACEFPDWLAST(sacefpd_wlast), .SACEFPDWUSER(sacefpd_wuser), .SACEFPDBVALID(sacefpd_bvalid), .SACEFPDBREADY(sacefpd_bready), .SACEFPDBID(sacefpd_bid), .SACEFPDBRESP(sacefpd_bresp), .SACEFPDBUSER(sacefpd_buser), .SACEFPDARVALID(sacefpd_arvalid), .SACEFPDARREADY(sacefpd_arready), .SACEFPDARID(sacefpd_arid), .SACEFPDARADDR(sacefpd_araddr), .SACEFPDARREGION(sacefpd_arregion), .SACEFPDARLEN(sacefpd_arlen), .SACEFPDARSIZE(sacefpd_arsize), .SACEFPDARBURST(sacefpd_arburst), .SACEFPDARLOCK(sacefpd_arlock), .SACEFPDARCACHE(sacefpd_arcache), .SACEFPDARPROT(sacefpd_arprot), .SACEFPDARDOMAIN(sacefpd_ardomain), .SACEFPDARSNOOP(sacefpd_arsnoop), .SACEFPDARBAR(sacefpd_arbar), .SACEFPDARQOS(sacefpd_arqos), .SACEFPDARUSER({6'b000000,4'b1111,sacefpd_aruser[5:0]}), .SACEFPDRVALID(sacefpd_rvalid), .SACEFPDRREADY(sacefpd_rready), .SACEFPDRID(sacefpd_rid), .SACEFPDRDATA(sacefpd_rdata), .SACEFPDRRESP(sacefpd_rresp), .SACEFPDRLAST(sacefpd_rlast), .SACEFPDRUSER(sacefpd_ruser), .SACEFPDACVALID(sacefpd_acvalid), .SACEFPDACREADY(sacefpd_acready), .SACEFPDACADDR(sacefpd_acaddr), .SACEFPDACSNOOP(sacefpd_acsnoop), .SACEFPDACPROT(sacefpd_acprot), .SACEFPDCRVALID(sacefpd_crvalid), .SACEFPDCRREADY(sacefpd_crready), .SACEFPDCRRESP(sacefpd_crresp), .SACEFPDCDVALID(sacefpd_cdvalid), .SACEFPDCDREADY(sacefpd_cdready), .SACEFPDCDDATA(sacefpd_cddata), .SACEFPDCDLAST(sacefpd_cdlast), .SACEFPDWACK(sacefpd_wack), .SACEFPDRACK(sacefpd_rack),

			// Extended Multiplexed I/O

			// EMIO Controller Area Network 0
			.EMIOCAN0PHYTX(emio_can0_phy_tx), .EMIOCAN0PHYRX(emio_can0_phy_rx),

			// EMIO Controller Area Network 1
			.EMIOCAN1PHYTX(emio_can1_phy_tx), .EMIOCAN1PHYRX(emio_can1_phy_rx),

			// Gigabit Ethernet Media Access Controller

			// EMIO Ethernet 0
			.EMIOENET0GMIIRXCLK(emio_enet0_gmii_rx_clk), .EMIOENET0SPEEDMODE(emio_enet0_speed_mode), .EMIOENET0GMIICRS(emio_enet0_gmii_crs), .EMIOENET0GMIICOL(emio_enet0_gmii_col), .EMIOENET0GMIIRXD(emio_enet0_gmii_rxd), .EMIOENET0GMIIRXER(emio_enet0_gmii_rx_er), .EMIOENET0GMIIRXDV(emio_enet0_gmii_rx_dv), .EMIOENET0GMIITXCLK(emio_enet0_gmii_tx_clk), .EMIOENET0GMIITXD(emio_enet0_gmii_txd), .EMIOENET0GMIITXEN(emio_enet0_gmii_tx_en), .EMIOENET0GMIITXER(emio_enet0_gmii_tx_er), .EMIOENET0MDIOMDC(emio_enet0_mdio_mdc), .EMIOENET0MDIOI(emio_enet0_mdio_i), .EMIOENET0MDIOO(emio_enet0_mdio_o), .EMIOENET0MDIOTN(emio_enet0_mdio_tri),

			// EMIO Ethernet 1
			.EMIOENET1GMIIRXCLK(emio_enet1_gmii_rx_clk), .EMIOENET1SPEEDMODE(emio_enet1_speed_mode), .EMIOENET1GMIICRS(emio_enet1_gmii_crs), .EMIOENET1GMIICOL(emio_enet1_gmii_col), .EMIOENET1GMIIRXD(emio_enet1_gmii_rxd), .EMIOENET1GMIIRXER(emio_enet1_gmii_rx_er), .EMIOENET1GMIIRXDV(emio_enet1_gmii_rx_dv), .EMIOENET1GMIITXCLK(emio_enet1_gmii_tx_clk), .EMIOENET1GMIITXD(emio_enet1_gmii_txd), .EMIOENET1GMIITXEN(emio_enet1_gmii_tx_en), .EMIOENET1GMIITXER(emio_enet1_gmii_tx_er), .EMIOENET1MDIOMDC(emio_enet1_mdio_mdc), .EMIOENET1MDIOI(emio_enet1_mdio_i), .EMIOENET1MDIOO(emio_enet1_mdio_o), .EMIOENET1MDIOTN(emio_enet1_mdio_tri),

			// EMIO Ethernet 2
			.EMIOENET2GMIIRXCLK(emio_enet2_gmii_rx_clk), .EMIOENET2SPEEDMODE(emio_enet2_speed_mode), .EMIOENET2GMIICRS(emio_enet2_gmii_crs), .EMIOENET2GMIICOL(emio_enet2_gmii_col), .EMIOENET2GMIIRXD(emio_enet2_gmii_rxd), .EMIOENET2GMIIRXER(emio_enet2_gmii_rx_er), .EMIOENET2GMIIRXDV(emio_enet2_gmii_rx_dv), .EMIOENET2GMIITXCLK(emio_enet2_gmii_tx_clk), .EMIOENET2GMIITXD(emio_enet2_gmii_txd), .EMIOENET2GMIITXEN(emio_enet2_gmii_tx_en), .EMIOENET2GMIITXER(emio_enet2_gmii_tx_er), .EMIOENET2MDIOMDC(emio_enet2_mdio_mdc), .EMIOENET2MDIOI(emio_enet2_mdio_i), .EMIOENET2MDIOO(emio_enet2_mdio_o), .EMIOENET2MDIOTN(emio_enet2_mdio_tri),

			// EMIO Ethernet 3
			.EMIOENET3GMIIRXCLK(emio_enet3_gmii_rx_clk), .EMIOENET3SPEEDMODE(emio_enet3_speed_mode), .EMIOENET3GMIICRS(emio_enet3_gmii_crs), .EMIOENET3GMIICOL(emio_enet3_gmii_col), .EMIOENET3GMIIRXD(emio_enet3_gmii_rxd), .EMIOENET3GMIIRXER(emio_enet3_gmii_rx_er), .EMIOENET3GMIIRXDV(emio_enet3_gmii_rx_dv), .EMIOENET3GMIITXCLK(emio_enet3_gmii_tx_clk), .EMIOENET3GMIITXD(emio_enet3_gmii_txd), .EMIOENET3GMIITXEN(emio_enet3_gmii_tx_en), .EMIOENET3GMIITXER(emio_enet3_gmii_tx_er), .EMIOENET3MDIOMDC(emio_enet3_mdio_mdc), .EMIOENET3MDIOI(emio_enet3_mdio_i), .EMIOENET3MDIOO(emio_enet3_mdio_o), .EMIOENET3MDIOTN(emio_enet3_mdio_tri),

			// EMIO Ethernet 0 Rx Tx
			.EMIOENET0TXRDATARDY(emio_enet0_tx_r_data_rdy), .EMIOENET0TXRRD(emio_enet0_tx_r_rd), .EMIOENET0TXRVALID(emio_enet0_tx_r_valid), .EMIOENET0TXRDATA(emio_enet0_tx_r_data), .EMIOENET0TXRSOP(emio_enet0_tx_r_sop), .EMIOENET0TXREOP(emio_enet0_tx_r_eop), .EMIOENET0TXRERR(emio_enet0_tx_r_err), .EMIOENET0TXRUNDERFLOW(emio_enet0_tx_r_underflow), .EMIOENET0TXRFLUSHED(emio_enet0_tx_r_flushed), .EMIOENET0TXRCONTROL(emio_enet0_tx_r_control), .EMIOENET0DMATXENDTOG(emio_enet0_dma_tx_end_tog), .EMIOENET0DMATXSTATUSTOG(emio_enet0_dma_tx_status_tog), .EMIOENET0TXRSTATUS(emio_enet0_tx_r_status), .EMIOENET0RXWWR(emio_enet0_rx_w_wr), .EMIOENET0RXWDATA(emio_enet0_rx_w_data), .EMIOENET0RXWSOP(emio_enet0_rx_w_sop), .EMIOENET0RXWEOP(emio_enet0_rx_w_eop), .EMIOENET0RXWSTATUS(emio_enet0_rx_w_status), .EMIOENET0RXWERR(emio_enet0_rx_w_err), .EMIOENET0RXWOVERFLOW(emio_enet0_rx_w_overflow), .FMIOGEM0SIGNALDETECT(emio_enet0_signal_detect), .EMIOENET0RXWFLUSH(emio_enet0_rx_w_flush), .EMIOGEM0TXRFIXEDLAT(emio_enet0_tx_r_fixed_lat), .FMIOGEM0FIFOTXCLKFROMPL(fmio_gem0_fifo_tx_clk), .FMIOGEM0FIFORXCLKFROMPL(fmio_gem0_fifo_rx_clk), .FMIOGEM0FIFOTXCLKTOPLBUFG(fmio_gem0_fifo_tx_clk_temp), .FMIOGEM0FIFORXCLKTOPLBUFG(fmio_gem0_fifo_rx_clk_temp),

			// EMIO Ethernet 1 Rx Tx
			.EMIOENET1TXRDATARDY(emio_enet1_tx_r_data_rdy), .EMIOENET1TXRRD(emio_enet1_tx_r_rd), .EMIOENET1TXRVALID(emio_enet1_tx_r_valid), .EMIOENET1TXRDATA(emio_enet1_tx_r_data), .EMIOENET1TXRSOP(emio_enet1_tx_r_sop), .EMIOENET1TXREOP(emio_enet1_tx_r_eop), .EMIOENET1TXRERR(emio_enet1_tx_r_err), .EMIOENET1TXRUNDERFLOW(emio_enet1_tx_r_underflow), .EMIOENET1TXRFLUSHED(emio_enet1_tx_r_flushed), .EMIOENET1TXRCONTROL(emio_enet1_tx_r_control), .EMIOENET1DMATXENDTOG(emio_enet1_dma_tx_end_tog), .EMIOENET1DMATXSTATUSTOG(emio_enet1_dma_tx_status_tog), .EMIOENET1TXRSTATUS(emio_enet1_tx_r_status), .EMIOENET1RXWWR(emio_enet1_rx_w_wr), .EMIOENET1RXWDATA(emio_enet1_rx_w_data), .EMIOENET1RXWSOP(emio_enet1_rx_w_sop), .EMIOENET1RXWEOP(emio_enet1_rx_w_eop), .EMIOENET1RXWSTATUS(emio_enet1_rx_w_status), .EMIOENET1RXWERR(emio_enet1_rx_w_err), .EMIOENET1RXWOVERFLOW(emio_enet1_rx_w_overflow), .FMIOGEM1SIGNALDETECT(emio_enet1_signal_detect), .EMIOENET1RXWFLUSH(emio_enet1_rx_w_flush), .EMIOGEM1TXRFIXEDLAT(emio_enet1_tx_r_fixed_lat), .FMIOGEM1FIFOTXCLKFROMPL(fmio_gem1_fifo_tx_clk), .FMIOGEM1FIFORXCLKFROMPL(fmio_gem1_fifo_rx_clk), .FMIOGEM1FIFOTXCLKTOPLBUFG(fmio_gem1_fifo_tx_clk_temp), .FMIOGEM1FIFORXCLKTOPLBUFG(fmio_gem1_fifo_rx_clk_temp),

			// EMIO Ethernet 2 Rx Tx
			.EMIOENET2TXRDATARDY(emio_enet2_tx_r_data_rdy), .EMIOENET2TXRRD(emio_enet2_tx_r_rd), .EMIOENET2TXRVALID(emio_enet2_tx_r_valid), .EMIOENET2TXRDATA(emio_enet2_tx_r_data), .EMIOENET2TXRSOP(emio_enet2_tx_r_sop), .EMIOENET2TXREOP(emio_enet2_tx_r_eop), .EMIOENET2TXRERR(emio_enet2_tx_r_err), .EMIOENET2TXRUNDERFLOW(emio_enet2_tx_r_underflow), .EMIOENET2TXRFLUSHED(emio_enet2_tx_r_flushed), .EMIOENET2TXRCONTROL(emio_enet2_tx_r_control), .EMIOENET2DMATXENDTOG(emio_enet2_dma_tx_end_tog), .EMIOENET2DMATXSTATUSTOG(emio_enet2_dma_tx_status_tog), .EMIOENET2TXRSTATUS(emio_enet2_tx_r_status), .EMIOENET2RXWWR(emio_enet2_rx_w_wr), .EMIOENET2RXWDATA(emio_enet2_rx_w_data), .EMIOENET2RXWSOP(emio_enet2_rx_w_sop), .EMIOENET2RXWEOP(emio_enet2_rx_w_eop), .EMIOENET2RXWSTATUS(emio_enet2_rx_w_status), .EMIOENET2RXWERR(emio_enet2_rx_w_err), .EMIOENET2RXWOVERFLOW(emio_enet2_rx_w_overflow), .FMIOGEM2SIGNALDETECT(emio_enet2_signal_detect), .EMIOENET2RXWFLUSH(emio_enet2_rx_w_flush), .EMIOGEM2TXRFIXEDLAT(emio_enet2_tx_r_fixed_lat), .FMIOGEM2FIFOTXCLKFROMPL(fmio_gem2_fifo_tx_clk), .FMIOGEM2FIFORXCLKFROMPL(fmio_gem2_fifo_rx_clk), .FMIOGEM2FIFOTXCLKTOPLBUFG(fmio_gem2_fifo_tx_clk_temp), .FMIOGEM2FIFORXCLKTOPLBUFG(fmio_gem2_fifo_rx_clk_temp),

			// EMIO Ethernet 3 Rx Tx
			.EMIOENET3TXRDATARDY(emio_enet3_tx_r_data_rdy), .EMIOENET3TXRRD(emio_enet3_tx_r_rd), .EMIOENET3TXRVALID(emio_enet3_tx_r_valid), .EMIOENET3TXRDATA(emio_enet3_tx_r_data), .EMIOENET3TXRSOP(emio_enet3_tx_r_sop), .EMIOENET3TXREOP(emio_enet3_tx_r_eop), .EMIOENET3TXRERR(emio_enet3_tx_r_err), .EMIOENET3TXRUNDERFLOW(emio_enet3_tx_r_underflow), .EMIOENET3TXRFLUSHED(emio_enet3_tx_r_flushed), .EMIOENET3TXRCONTROL(emio_enet3_tx_r_control), .EMIOENET3DMATXENDTOG(emio_enet3_dma_tx_end_tog), .EMIOENET3DMATXSTATUSTOG(emio_enet3_dma_tx_status_tog), .EMIOENET3TXRSTATUS(emio_enet3_tx_r_status), .EMIOENET3RXWWR(emio_enet3_rx_w_wr), .EMIOENET3RXWDATA(emio_enet3_rx_w_data), .EMIOENET3RXWSOP(emio_enet3_rx_w_sop), .EMIOENET3RXWEOP(emio_enet3_rx_w_eop), .EMIOENET3RXWSTATUS(emio_enet3_rx_w_status), .EMIOENET3RXWERR(emio_enet3_rx_w_err), .EMIOENET3RXWOVERFLOW(emio_enet3_rx_w_overflow), .FMIOGEM3SIGNALDETECT(emio_enet3_signal_detect), .EMIOENET3RXWFLUSH(emio_enet3_rx_w_flush), .EMIOGEM3TXRFIXEDLAT(emio_enet3_tx_r_fixed_lat), .FMIOGEM3FIFOTXCLKFROMPL(fmio_gem3_fifo_tx_clk), .FMIOGEM3FIFORXCLKFROMPL(fmio_gem3_fifo_rx_clk), .FMIOGEM3FIFOTXCLKTOPLBUFG(fmio_gem3_fifo_tx_clk_temp), .FMIOGEM3FIFORXCLKTOPLBUFG(fmio_gem3_fifo_rx_clk_temp),

			// EMIO Gigabit Ethernet MAC 0
			.EMIOGEM0TXSOF(emio_enet0_tx_sof), .EMIOGEM0SYNCFRAMETX(emio_enet0_sync_frame_tx), .EMIOGEM0DELAYREQTX(emio_enet0_delay_req_tx), .EMIOGEM0PDELAYREQTX(emio_enet0_pdelay_req_tx), .EMIOGEM0PDELAYRESPTX(emio_enet0_pdelay_resp_tx), .EMIOGEM0RXSOF(emio_enet0_rx_sof), .EMIOGEM0SYNCFRAMERX(emio_enet0_sync_frame_rx), .EMIOGEM0DELAYREQRX(emio_enet0_delay_req_rx), .EMIOGEM0PDELAYREQRX(emio_enet0_pdelay_req_rx), .EMIOGEM0PDELAYRESPRX(emio_enet0_pdelay_resp_rx), .EMIOGEM0TSUINCCTRL(emio_enet0_tsu_inc_ctrl), .EMIOGEM0TSUTIMERCMPVAL(emio_enet0_tsu_timer_cmp_val),

			// EMIO Gigabit Ethernet MAC 1
			.EMIOGEM1TXSOF(emio_enet1_tx_sof), .EMIOGEM1SYNCFRAMETX(emio_enet1_sync_frame_tx), .EMIOGEM1DELAYREQTX(emio_enet1_delay_req_tx), .EMIOGEM1PDELAYREQTX(emio_enet1_pdelay_req_tx), .EMIOGEM1PDELAYRESPTX(emio_enet1_pdelay_resp_tx), .EMIOGEM1RXSOF(emio_enet1_rx_sof), .EMIOGEM1SYNCFRAMERX(emio_enet1_sync_frame_rx), .EMIOGEM1DELAYREQRX(emio_enet1_delay_req_rx), .EMIOGEM1PDELAYREQRX(emio_enet1_pdelay_req_rx), .EMIOGEM1PDELAYRESPRX(emio_enet1_pdelay_resp_rx), .EMIOGEM1TSUINCCTRL(emio_enet1_tsu_inc_ctrl), .EMIOGEM1TSUTIMERCMPVAL(emio_enet1_tsu_timer_cmp_val),

			// EMIO Gigabit Ethernet MAC 2
			.EMIOGEM2TXSOF(emio_enet2_tx_sof), .EMIOGEM2SYNCFRAMETX(emio_enet2_sync_frame_tx), .EMIOGEM2DELAYREQTX(emio_enet2_delay_req_tx), .EMIOGEM2PDELAYREQTX(emio_enet2_pdelay_req_tx), .EMIOGEM2PDELAYRESPTX(emio_enet2_pdelay_resp_tx), .EMIOGEM2RXSOF(emio_enet2_rx_sof), .EMIOGEM2SYNCFRAMERX(emio_enet2_sync_frame_rx), .EMIOGEM2DELAYREQRX(emio_enet2_delay_req_rx), .EMIOGEM2PDELAYREQRX(emio_enet2_pdelay_req_rx), .EMIOGEM2PDELAYRESPRX(emio_enet2_pdelay_resp_rx), .EMIOGEM2TSUINCCTRL(emio_enet2_tsu_inc_ctrl), .EMIOGEM2TSUTIMERCMPVAL(emio_enet2_tsu_timer_cmp_val),

			// EMIO Gigabit Ethernet MAC 3
			.EMIOGEM3TXSOF(emio_enet3_tx_sof), .EMIOGEM3SYNCFRAMETX(emio_enet3_sync_frame_tx), .EMIOGEM3DELAYREQTX(emio_enet3_delay_req_tx), .EMIOGEM3PDELAYREQTX(emio_enet3_pdelay_req_tx), .EMIOGEM3PDELAYRESPTX(emio_enet3_pdelay_resp_tx), .EMIOGEM3RXSOF(emio_enet3_rx_sof), .EMIOGEM3SYNCFRAMERX(emio_enet3_sync_frame_rx), .EMIOGEM3DELAYREQRX(emio_enet3_delay_req_rx), .EMIOGEM3PDELAYREQRX(emio_enet3_pdelay_req_rx), .EMIOGEM3PDELAYRESPRX(emio_enet3_pdelay_resp_rx), .EMIOGEM3TSUINCCTRL(emio_enet3_tsu_inc_ctrl), .EMIOGEM3TSUTIMERCMPVAL(emio_enet3_tsu_timer_cmp_val),

			.FMIOGEMTSUCLKFROMPL(fmio_gem_tsu_clk_from_pl), .FMIOGEMTSUCLKTOPLBUFG(fmio_gem_tsu_clk_to_pl_bufg),
			.EMIOENETTSUCLK(emio_enet_tsu_clk), .EMIOENET0GEMTSUTIMERCNT(emio_enet0_enet_tsu_timer_cnt),

			.EMIOENET0EXTINTIN(emio_enet0_ext_int_in), .EMIOENET1EXTINTIN(emio_enet1_ext_int_in), .EMIOENET2EXTINTIN(emio_enet2_ext_int_in), .EMIOENET3EXTINTIN(emio_enet3_ext_int_in),

			.EMIOENET0DMABUSWIDTH(emio_enet0_dma_bus_width), .EMIOENET1DMABUSWIDTH(emio_enet1_dma_bus_width), .EMIOENET2DMABUSWIDTH(emio_enet2_dma_bus_width), .EMIOENET3DMABUSWIDTH(emio_enet3_dma_bus_width),

			// EMIO General Purpose Input Output
			.EMIOGPIOI(emio_gpio_i_temp), .EMIOGPIOO(emio_gpio_o_temp), .EMIOGPIOTN(emio_gpio_t_temp),

			// EMIO Inter Integrated Circuit 0
			.EMIOI2C0SCLI(emio_i2c0_scl_i), .EMIOI2C0SCLO(emio_i2c0_scl_o), .EMIOI2C0SCLTN(emio_i2c0_scl_tri), .EMIOI2C0SDAI(emio_i2c0_sda_i), .EMIOI2C0SDAO(emio_i2c0_sda_o), .EMIOI2C0SDATN(emio_i2c0_sda_tri),

			// EMIO Inter Integrated Circuit 1
			.EMIOI2C1SCLI(emio_i2c1_scl_i), .EMIOI2C1SCLO(emio_i2c1_scl_o), .EMIOI2C1SCLTN(emio_i2c1_scl_tri), .EMIOI2C1SDAI(emio_i2c1_sda_i), .EMIOI2C1SDAO(emio_i2c1_sda_o), .EMIOI2C1SDATN(emio_i2c1_sda_tri),

			// EMIO Universal Asynchronous Receiver-Transmitter 0
			.EMIOUART0TX(emio_uart0_txd), .EMIOUART0RX(emio_uart0_rxd), .EMIOUART0CTSN(emio_uart0_ctsn), .EMIOUART0RTSN(emio_uart0_rtsn), .EMIOUART0DSRN(emio_uart0_dsrn), .EMIOUART0DCDN(emio_uart0_dcdn), .EMIOUART0RIN(emio_uart0_rin), .EMIOUART0DTRN(emio_uart0_dtrn),

			// EMIO Universal Asynchronous Receiver-Transmitter 1
			.EMIOUART1TX(emio_uart1_txd), .EMIOUART1RX(emio_uart1_rxd), .EMIOUART1CTSN(emio_uart1_ctsn), .EMIOUART1RTSN(emio_uart1_rtsn), .EMIOUART1DSRN(emio_uart1_dsrn), .EMIOUART1DCDN(emio_uart1_dcdn), .EMIOUART1RIN(emio_uart1_rin), .EMIOUART1DTRN(emio_uart1_dtrn),

			// EMIO Secure Digital 0
			.EMIOSDIO0CLKOUT(emio_sdio0_clkout), .EMIOSDIO0FBCLKIN(emio_sdio0_fb_clk_in), .EMIOSDIO0CMDOUT(emio_sdio0_cmdout), .EMIOSDIO0CMDIN(emio_sdio0_cmdin), .EMIOSDIO0CMDENA(emio_sdio0_cmdena_i), .EMIOSDIO0DATAIN(emio_sdio0_datain), .EMIOSDIO0DATAOUT(emio_sdio0_dataout), .EMIOSDIO0DATAENA(emio_sdio0_dataena_i), .EMIOSDIO0CDN(emio_sdio0_cd_n), .EMIOSDIO0WP(emio_sdio0_wp), .EMIOSDIO0LEDCONTROL(emio_sdio0_ledcontrol), .EMIOSDIO0BUSPOWER(emio_sdio0_buspower), .EMIOSDIO0BUSVOLT(emio_sdio0_bus_volt),

			// EMIO Secure Digital 1
			.EMIOSDIO1CLKOUT(emio_sdio1_clkout), .EMIOSDIO1FBCLKIN(emio_sdio1_fb_clk_in), .EMIOSDIO1CMDOUT(emio_sdio1_cmdout), .EMIOSDIO1CMDIN(emio_sdio1_cmdin), .EMIOSDIO1CMDENA(emio_sdio1_cmdena_i), .EMIOSDIO1DATAIN(emio_sdio1_datain), .EMIOSDIO1DATAOUT(emio_sdio1_dataout), .EMIOSDIO1DATAENA(emio_sdio1_dataena_i), .EMIOSDIO1CDN(emio_sdio1_cd_n), .EMIOSDIO1WP(emio_sdio1_wp), .EMIOSDIO1LEDCONTROL(emio_sdio1_ledcontrol), .EMIOSDIO1BUSPOWER(emio_sdio1_buspower), .EMIOSDIO1BUSVOLT(emio_sdio1_bus_volt),

			// EMIO Serial Peripheral Interface 0
			.EMIOSPI0SCLKI(emio_spi0_sclk_i), .EMIOSPI0SCLKO(emio_spi0_sclk_o), .EMIOSPI0SCLKTN(emio_spi0_sclk_tri), .EMIOSPI0MI(emio_spi0_m_i), .EMIOSPI0MO(emio_spi0_m_o), .EMIOSPI0MOTN(emio_spi0_mo_tri), .EMIOSPI0SI(emio_spi0_s_i), .EMIOSPI0SO(emio_spi0_s_o), .EMIOSPI0STN(emio_spi0_so_tri), .EMIOSPI0SSIN(emio_spi0_ss_i_n), .EMIOSPI0SSON({emio_spi0_ss2_o_n,emio_spi0_ss1_o_n,emio_spi0_ss_o_n}), .EMIOSPI0SSNTN(emio_spi0_ss_n_tri),

			// EMIO Serial Peripheral Interface 1
			.EMIOSPI1SCLKI(emio_spi1_sclk_i), .EMIOSPI1SCLKO(emio_spi1_sclk_o), .EMIOSPI1SCLKTN(emio_spi1_sclk_tri), .EMIOSPI1MI(emio_spi1_m_i), .EMIOSPI1MO(emio_spi1_m_o), .EMIOSPI1MOTN(emio_spi1_mo_tri), .EMIOSPI1SI(emio_spi1_s_i), .EMIOSPI1SO(emio_spi1_s_o), .EMIOSPI1STN(emio_spi1_so_tri), .EMIOSPI1SSIN(emio_spi1_ss_i_n), .EMIOSPI1SSON({emio_spi1_ss2_o_n,emio_spi1_ss1_o_n,emio_spi1_ss_o_n}), .EMIOSPI1SSNTN(emio_spi1_ss_n_tri),

			// Programmable Logic & Processing System
			.PLPSTRACECLK(pl_ps_trace_clk), .PSPLTRACECTL(trace_ctl_i), .PSPLTRACEDATA(trace_data_i),s

			// EMIO Timing Trigger and Control 0
			.EMIOTTC0WAVEO(emio_ttc0_wave_o), .EMIOTTC0CLKI(emio_ttc0_clk_i),

			// EMIO Timing Trigger and Control 1
			.EMIOTTC1WAVEO(emio_ttc1_wave_o), .EMIOTTC1CLKI(emio_ttc1_clk_i),

			// EMIO Timing Trigger and Control 2
			.EMIOTTC2WAVEO(emio_ttc2_wave_o), .EMIOTTC2CLKI(emio_ttc2_clk_i),

			// EMIO Timing Trigger and Control 3
			.EMIOTTC3WAVEO(emio_ttc3_wave_o), .EMIOTTC3CLKI(emio_ttc3_clk_i),

			// EMIO Watchdog Timer
			.EMIOWDT0CLKI(emio_wdt0_clk_i), .EMIOWDT0RSTO(emio_wdt0_rst_o),

			// EMIO Watchdog Timer
			.EMIOWDT1CLKI(emio_wdt1_clk_i), .EMIOWDT1RSTO(emio_wdt1_rst_o),

			.EMIOHUBPORTOVERCRNTUSB30(emio_hub_port_overcrnt_usb3_0), .EMIOHUBPORTOVERCRNTUSB31(emio_hub_port_overcrnt_usb3_1), .EMIOHUBPORTOVERCRNTUSB20(emio_hub_port_overcrnt_usb2_0), .EMIOHUBPORTOVERCRNTUSB21(emio_hub_port_overcrnt_usb2_1),

			.EMIOU2DSPORTVBUSCTRLUSB30(emio_u2dsport_vbus_ctrl_usb3_0), .EMIOU2DSPORTVBUSCTRLUSB31(emio_u2dsport_vbus_ctrl_usb3_1), .EMIOU3DSPORTVBUSCTRLUSB30(emio_u3dsport_vbus_ctrl_usb3_0), .EMIOU3DSPORTVBUSCTRLUSB31(emio_u3dsport_vbus_ctrl_usb3_1),

			.ADMAFCICLK(adma_fci_clk), .PL2ADMACVLD(pl2adma_cvld), .PL2ADMATACK(pl2adma_tack), .ADMA2PLCACK(adma2pl_cack), .ADMA2PLTVLD(adma2pl_tvld),

			.GDMAFCICLK(perif_gdma_clk), .PL2GDMACVLD(perif_gdma_cvld), .PL2GDMATACK(perif_gdma_tack), .GDMA2PLCACK(gdma_perif_cack), .GDMA2PLTVLD(gdma_perif_tvld),

			.PLFPGASTOP(pl_clock_stop),

			// Phase-Locked Loop
			.PLLAUXREFCLKLPD(pll_aux_refclk_lpd), .PLLAUXREFCLKFPD(pll_aux_refclk_fpd),

			// Display Port

			// DP Slave AXI
			.DPSAXISAUDIOTDATA(dp_s_axis_audio_tdata), .DPSAXISAUDIOTID(dp_s_axis_audio_tid), .DPSAXISAUDIOTVALID(dp_s_axis_audio_tvalid), .DPSAXISAUDIOTREADY(dp_s_axis_audio_tready),

			// DP Master AXI
			.DPMAXISMIXEDAUDIOTDATA(dp_m_axis_mixed_audio_tdata), .DPMAXISMIXEDAUDIOTID(dp_m_axis_mixed_audio_tid), .DPMAXISMIXEDAUDIOTVALID(dp_m_axis_mixed_audio_tvalid), .DPMAXISMIXEDAUDIOTREADY(dp_m_axis_mixed_audio_tready),

			.DPSAXISAUDIOCLK(dp_s_axis_audio_clk),

			.DPLIVEVIDEOINVSYNC(dp_live_video_in_vsync), .DPLIVEVIDEOINHSYNC(dp_live_video_in_hsync), .DPLIVEVIDEOINDE(dp_live_video_in_de), .DPLIVEVIDEOINPIXEL1(dp_live_video_in_pixel1),

			.DPVIDEOINCLK(dp_video_in_clk), .DPVIDEOOUTHSYNC(dp_video_out_hsync), .DPVIDEOOUTVSYNC(dp_video_out_vsync), .DPVIDEOOUTPIXEL1(dp_video_out_pixel1),

			.DPAUXDATAIN(dp_aux_data_in), .DPAUXDATAOUT(dp_aux_data_out), .DPAUXDATAOEN(dp_aux_data_oe_n),

			.DPLIVEGFXALPHAIN(dp_live_gfx_alpha_in), .DPLIVEGFXPIXEL1IN(dp_live_gfx_pixel1_in),

			.DPHOTPLUGDETECT(dp_hot_plug_detect),

			.DPEXTERNALCUSTOMEVENT1(dp_external_custom_event1), .DPEXTERNALCUSTOMEVENT2(dp_external_custom_event2),

			.DPEXTERNALVSYNCEVENT(dp_external_vsync_event),
			.DPLIVEVIDEODEOUT(dp_live_video_de_out),


			.PLPSEVENTI(pl_ps_eventi), .PSPLEVENTO(ps_pl_evento),
			.PSPLSTANDBYWFE(ps_pl_standbywfe), .PSPLSTANDBYWFI(ps_pl_standbywfi),
			.PLPSAPUGICIRQ(pl_ps_apugic_irq), .PLPSAPUGICFIQ(pl_ps_apugic_fiq),

			.RPUEVENTI0(rpu_eventi0), .RPUEVENTI1(rpu_eventi1),
			.RPUEVENTO0(rpu_evento0), .RPUEVENTO1(rpu_evento1),
			.NFIQ0LPDRPU(nfiq0_lpd_rpu), .NFIQ1LPDRPU(nfiq1_lpd_rpu),
			.NIRQ0LPDRPU(nirq0_lpd_rpu), .NIRQ1LPDRPU(nirq1_lpd_rpu),

			.STMEVENT(stm_event),
			.PLPSTRIGACK({pl_ps_trigack_3, pl_ps_trigack_2, pl_ps_trigack_1, pl_ps_trigack_0}), .PLPSTRIGGER({pl_ps_trigger_3, pl_ps_trigger_2, pl_ps_trigger_1, pl_ps_trigger_0}),
			.PSPLTRIGACK({ps_pl_trigack_3, ps_pl_trigack_2, ps_pl_trigack_1, ps_pl_trigack_0}), .PSPLTRIGGER({ps_pl_trigger_3, ps_pl_trigger_2, ps_pl_trigger_1, ps_pl_trigger_0}),
			.FTMGPO(ftm_gpo), .FTMGPI(ftm_gpi),
			.PLPSIRQ0(irq_f2p_0_i), .PLPSIRQ1(irq_f2p_1_i),
			.PSPLIRQLPD({ps_pl_irq_lpd_low[18:8], ps_pl_irq_xmpu_lpd, ps_pl_irq_efuse, ps_pl_irq_csu_dma, ps_pl_irq_csu, ps_pl_irq_adma_chan, ps_pl_irq_usb3_0_pmu_wakeup, ps_pl_irq_usb3_1_otg, ps_pl_irq_usb3_1_endpoint, ps_pl_irq_usb3_0_otg, ps_pl_irq_usb3_0_endpoint, ps_pl_irq_enet3_wake, ps_pl_irq_enet3, ps_pl_irq_enet2_wake, ps_pl_irq_enet2, ps_pl_irq_enet1_wake, ps_pl_irq_enet1, ps_pl_irq_enet0_wake, ps_pl_irq_enet0, ps_pl_irq_ams, ps_pl_irq_aib_axi, ps_pl_irq_atb_err_lpd, ps_pl_irq_csu_pmu_wdt, ps_pl_irq_lp_wdt, ps_pl_irq_sdio1_wake, ps_pl_irq_sdio0_wake, ps_pl_irq_sdio1, ps_pl_irq_sdio0, ps_pl_irq_ttc3_2, ps_pl_irq_ttc3_1, ps_pl_irq_ttc3_0, ps_pl_irq_ttc2_2, ps_pl_irq_ttc2_1, ps_pl_irq_ttc2_0, ps_pl_irq_ttc1_2, ps_pl_irq_ttc1_1, ps_pl_irq_ttc1_0, ps_pl_irq_ttc0_2, ps_pl_irq_ttc0_1, ps_pl_irq_ttc0_0, ps_pl_irq_ipi_channel0, ps_pl_irq_ipi_channel1, ps_pl_irq_ipi_channel2, ps_pl_irq_ipi_channel7, ps_pl_irq_ipi_channel8, ps_pl_irq_ipi_channel9, ps_pl_irq_ipi_channel10, ps_pl_irq_clkmon, ps_pl_irq_rtc_seconds, ps_pl_irq_rtc_alaram, ps_pl_irq_lpd_apm, ps_pl_irq_can1, ps_pl_irq_can0, ps_pl_irq_uart1, ps_pl_irq_uart0, ps_pl_irq_spi1, ps_pl_irq_spi0, ps_pl_irq_i2c1, ps_pl_irq_i2c0, ps_pl_irq_gpio, ps_pl_irq_qspi, ps_pl_irq_nand, ps_pl_irq_r5_core1_ecc_error, ps_pl_irq_r5_core0_ecc_error, ps_pl_irq_lpd_apb_intr, ps_pl_irq_ocm_error, ps_pl_irq_rpu_pm, ps_pl_irq_lpd_low[7:0]}), .PSPLIRQFPD({ps_pl_irq_fpd_low[19:12], ps_pl_irq_intf_fpd_smmu, ps_pl_irq_intf_ppd_cci, ps_pl_irq_apu_regs, ps_pl_irq_apu_exterr, ps_pl_irq_apu_l2err, ps_pl_irq_apu_comm, ps_pl_irq_apu_pmu, ps_pl_irq_apu_cti, ps_pl_irq_apu_cpumnt, ps_pl_irq_xmpu_fpd, ps_pl_irq_sata, ps_pl_irq_gpu, ps_pl_irq_gdma_chan, ps_pl_irq_apm_fpd, ps_pl_irq_dpdma, ps_pl_irq_fpd_atb_error, ps_pl_irq_fpd_apb_int, ps_pl_irq_dport, ps_pl_irq_pcie_msc, ps_pl_irq_pcie_dma, ps_pl_irq_pcie_legacy, ps_pl_irq_pcie_msi, ps_pl_irq_fp_wdt, ps_pl_irq_ddr_ss, ps_pl_irq_fpd_low[11:0]}),
			.OSCRTCCLK(osc_rtc_clk),
			.PLPMUGPI(pl_pmu_gpi),

			// Platform Managment Unit
			.PMUPLGPO(pmu_pl_gpo), .AIBPMUAFIFMFPDACK(aib_pmu_afifm_fpd_ack), .AIBPMUAFIFMLPDACK(aib_pmu_afifm_lpd_ack), .PMUAIBAFIFMFPDREQ(pmu_aib_afifm_fpd_req), .PMUAIBAFIFMLPDREQ(pmu_aib_afifm_lpd_req), .PMUERRORTOPL(pmu_error_to_pl), .PMUERRORFROMPL(pmu_error_from_pl),

			// Double Data Rate Controller
			.DDRCEXTREFRESHRANK0REQ(ddrc_ext_refresh_rank0_req), .DDRCEXTREFRESHRANK1REQ(ddrc_ext_refresh_rank1_req), .DDRCREFRESHPLCLK(ddrc_refresh_pl_clk),

			.PLACPINACT(pl_acpinact),
			.PLCLK(pl_clk_unbuffered),
			.DPVIDEOREFCLK(dp_video_ref_clk_i),
			.DPAUDIOREFCLK(dp_audio_ref_clk_i)
		);

		assign test_db = 16'h0000;
		assign test_adc_out = 20'h00000;
		assign test_ams_osc = 8'h00;
		assign test_mon_data =  16'h0000;
		assign test_drdy = 1'b0;
		assign test_do = 16'h0000;
		assign pstp_pl_out = 32'h00000000;
		assign fmio_test_io_char_scan_out = 1'b0;
		assign fmio_char_afifslpd_test_output = 1'b0;
		assign fmio_char_afifsfpd_test_output = 1'b0;
		assign io_char_video_out_test_data = 1'b0;
		assign io_char_audio_out_test_data = 1'b0;
		assign fmio_sd0_dll_test_out = 8'h00;
		assign fmio_sd1_dll_test_out = 8'h00;
		assign test_pl_scan_chopper_so = 1'b0;
		assign test_pl_scan_edt_out_apu = 1'b0;
		assign test_pl_scan_edt_out_cpu0 = 1'b0;
		assign test_pl_scan_edt_out_cpu1 = 1'b0;
		assign test_pl_scan_edt_out_cpu2 = 1'b0;
		assign test_pl_scan_edt_out_cpu3 = 1'b0;
		assign test_pl_scan_edt_out_ddr = 4'h0;
		assign test_pl_scan_edt_out_fp = 10'b0000000000;
		assign test_pl_scan_edt_out_gpu = 4'h0;
		assign test_pl_scan_edt_out_lp = 9'b000000000;
		assign test_pl_scan_edt_out_usb3 = 2'b00;
		assign test_pl_scan_spare_out0 = 1'b0;
		assign test_pl_scan_spare_out1 = 1'b0;
		assign test_pl_pll_lock_out = 5'b00000;
		assign test_pl_scan_slcr_config_so = 1'b0;
		assign tst_rtc_calibreg_out = 21'b000000000000000000000;
		assign tst_rtc_osc_clk_out = 1'b0;
		assign tst_rtc_sec_counter_out = 32'h00000000;
		assign tst_rtc_seconds_raw_int = 1'b0;
		assign tst_rtc_tick_counter_out = 16'h0000;
		assign tst_rtc_timesetreg_out = 32'h00000000;
		assign tst_rtc_osc_cntrl_out = 4'h0;
		assign lpd_pll_test_out = 32'h00000000;
		assign fpd_pll_test_out = 32'h00000000;
		assign fmio_char_gem_test_output = 1'b0;
		assign test_ddr2pl_dcd_skewout = 1'b0;
		assign test_bscan_tdo = 1'b0;
		assign fpd_pl_spare_0_out = 1'b0;
		assign fpd_pl_spare_1_out = 1'b0;
		assign fpd_pl_spare_2_out = 1'b0;
		assign fpd_pl_spare_3_out = 1'b0;
		assign fpd_pl_spare_4_out = 1'b0;
		assign lpd_pl_spare_0_out = 1'b0;
		assign lpd_pl_spare_1_out = 1'b0;
		assign lpd_pl_spare_2_out = 1'b0;
		assign lpd_pl_spare_3_out = 1'b0;
		assign lpd_pl_spare_4_out = 1'b0;
		assign o_dbg_l0_phystatus = 1'b0;
		assign o_dbg_l0_rxdata = 20'h00000;
		assign o_dbg_l0_rxdatak = 2'b00;
		assign o_dbg_l0_rxvalid = 1'b0;
		assign o_dbg_l0_rxstatus = 3'b000;
		assign o_dbg_l0_rxelecidle = 1'b0;
		assign o_dbg_l0_rstb = 1'b0;
		assign o_dbg_l0_txdata = 20'h00000;
		assign o_dbg_l0_txdatak = 2'b00;
		assign o_dbg_l0_rate = 2'b00;
		assign o_dbg_l0_powerdown = 2'b00;
		assign o_dbg_l0_txelecidle = 1'b0;
		assign o_dbg_l0_txdetrx_lpback = 1'b0;
		assign o_dbg_l0_rxpolarity = 1'b0;
		assign o_dbg_l0_tx_sgmii_ewrap = 1'b0;
		assign o_dbg_l0_rx_sgmii_en_cdet = 1'b0;
		assign o_dbg_l0_sata_corerxdata = 20'h00000;
		assign o_dbg_l0_sata_corerxdatavalid = 2'b00;
		assign o_dbg_l0_sata_coreready = 1'b0;
		assign o_dbg_l0_sata_coreclockready = 1'b0;
		assign o_dbg_l0_sata_corerxsignaldet = 1'b0;
		assign o_dbg_l0_sata_phyctrltxdata = 20'h00000;
		assign o_dbg_l0_sata_phyctrltxidle = 1'b0;
		assign o_dbg_l0_sata_phyctrltxrate = 2'b00;
		assign o_dbg_l0_sata_phyctrlrxrate = 2'b00;
		assign o_dbg_l0_sata_phyctrltxrst = 1'b0;
		assign o_dbg_l0_sata_phyctrlrxrst = 1'b0;
		assign o_dbg_l0_sata_phyctrlreset = 1'b0;
		assign o_dbg_l0_sata_phyctrlpartial = 1'b0;
		assign o_dbg_l0_sata_phyctrlslumber = 1'b0;
		assign o_dbg_l1_phystatus = 1'b0;
		assign o_dbg_l1_rxdata = 20'h00000;
		assign o_dbg_l1_rxdatak = 2'b00;
		assign o_dbg_l1_rxvalid = 1'b0;
		assign o_dbg_l1_rxstatus = 3'b000;
		assign o_dbg_l1_rxelecidle = 1'b0;
		assign o_dbg_l1_rstb = 1'b0;
		assign o_dbg_l1_txdata = 20'h00000;
		assign o_dbg_l1_txdatak = 2'b00;
		assign o_dbg_l1_rate = 2'b00;
		assign o_dbg_l1_powerdown = 2'b00;
		assign o_dbg_l1_txelecidle = 1'b0;
		assign o_dbg_l1_txdetrx_lpback = 1'b0;
		assign o_dbg_l1_rxpolarity = 1'b0;
		assign o_dbg_l1_tx_sgmii_ewrap = 1'b0;
		assign o_dbg_l1_rx_sgmii_en_cdet = 1'b0;
		assign o_dbg_l1_sata_corerxdata  = 20'h00000;
		assign o_dbg_l1_sata_corerxdatavalid = 2'b00;
		assign o_dbg_l1_sata_coreready = 1'b0;
		assign o_dbg_l1_sata_coreclockready = 1'b0;
		assign o_dbg_l1_sata_corerxsignaldet = 1'b0;
		assign o_dbg_l1_sata_phyctrltxdata  = 20'h00000;
		assign o_dbg_l1_sata_phyctrltxidle = 1'b0;
		assign o_dbg_l1_sata_phyctrltxrate = 2'b00;
		assign o_dbg_l1_sata_phyctrlrxrate = 2'b00;
		assign o_dbg_l1_sata_phyctrltxrst = 1'b0;
		assign o_dbg_l1_sata_phyctrlrxrst = 1'b0;
		assign o_dbg_l1_sata_phyctrlreset = 1'b0;
		assign o_dbg_l1_sata_phyctrlpartial = 1'b0;
		assign o_dbg_l1_sata_phyctrlslumber = 1'b0;
		assign o_dbg_l2_phystatus = 1'b0;
		assign o_dbg_l2_rxdata = 20'h00000;
		assign o_dbg_l2_rxdatak = 2'b00;
		assign o_dbg_l2_rxvalid = 1'b0;
		assign o_dbg_l2_rxstatus = 3'b000;
		assign o_dbg_l2_rxelecidle = 1'b0;
		assign o_dbg_l2_rstb = 1'b0;
		assign o_dbg_l2_txdata = 20'h00000;
		assign o_dbg_l2_txdatak = 2'b00;
		assign o_dbg_l2_rate = 2'b00;
		assign o_dbg_l2_powerdown = 2'b00;
		assign o_dbg_l2_txelecidle = 1'b0;
		assign o_dbg_l2_txdetrx_lpback = 1'b0;
		assign o_dbg_l2_rxpolarity = 1'b0;
		assign o_dbg_l2_tx_sgmii_ewrap = 1'b0;
		assign o_dbg_l2_rx_sgmii_en_cdet = 1'b0;
		assign o_dbg_l2_sata_corerxdata = 20'h00000;
		assign o_dbg_l2_sata_corerxdatavalid = 2'b00;
		assign o_dbg_l2_sata_coreready = 1'b0;
		assign o_dbg_l2_sata_coreclockready = 1'b0;
		assign o_dbg_l2_sata_corerxsignaldet = 1'b0;
		assign o_dbg_l2_sata_phyctrltxdata = 20'h00000;
		assign o_dbg_l2_sata_phyctrltxidle = 1'b0;
		assign o_dbg_l2_sata_phyctrltxrate = 2'b00;
		assign o_dbg_l2_sata_phyctrlrxrate = 2'b00;
		assign o_dbg_l2_sata_phyctrltxrst = 1'b0;
		assign o_dbg_l2_sata_phyctrlrxrst = 1'b0;
		assign o_dbg_l2_sata_phyctrlreset = 1'b0;
		assign o_dbg_l2_sata_phyctrlpartial = 1'b0;
		assign o_dbg_l2_sata_phyctrlslumber = 1'b0;
		assign o_dbg_l3_phystatus = 1'b0;
		assign o_dbg_l3_rxdata = 20'h00000;
		assign o_dbg_l3_rxdatak = 2'b00;
		assign o_dbg_l3_rxvalid = 1'b0;
		assign o_dbg_l3_rxstatus = 3'b000;
		assign o_dbg_l3_rxelecidle = 1'b0;
		assign o_dbg_l3_rstb = 1'b0;
		assign o_dbg_l3_txdata = 20'h00000;
		assign o_dbg_l3_txdatak = 2'b00;
		assign o_dbg_l3_rate = 2'b00;
		assign o_dbg_l3_powerdown = 2'b00;
		assign o_dbg_l3_txelecidle = 1'b0;
		assign o_dbg_l3_txdetrx_lpback = 1'b0;
		assign o_dbg_l3_rxpolarity = 1'b0;
		assign o_dbg_l3_tx_sgmii_ewrap = 1'b0;
		assign o_dbg_l3_rx_sgmii_en_cdet = 1'b0;
		assign o_dbg_l3_sata_corerxdata = 20'h00000;
		assign o_dbg_l3_sata_corerxdatavalid = 2'b00;
		assign o_dbg_l3_sata_coreready = 1'b0;
		assign o_dbg_l3_sata_coreclockready = 1'b0;
		assign o_dbg_l3_sata_corerxsignaldet = 1'b0;
		assign o_dbg_l3_sata_phyctrltxdata = 20'h00000;
		assign o_dbg_l3_sata_phyctrltxidle = 1'b0;
		assign o_dbg_l3_sata_phyctrltxrate = 2'b00;
		assign o_dbg_l3_sata_phyctrlrxrate = 2'b00;
		assign o_dbg_l3_sata_phyctrltxrst = 1'b0;
		assign o_dbg_l3_sata_phyctrlrxrst = 1'b0;
		assign o_dbg_l3_sata_phyctrlreset = 1'b0;
		assign o_dbg_l3_sata_phyctrlpartial = 1'b0;
		assign o_dbg_l3_sata_phyctrlslumber = 1'b0;
		assign dbg_path_fifo_bypass = 1'b0;
		assign o_afe_cmn_calib_comp_out = 1'b0;
		assign o_afe_pll_dco_count = 13'b0000000000000;
		assign o_afe_pll_clk_sym_hs = 1'b0;
		assign o_afe_pll_fbclk_frac = 1'b0;
		assign o_afe_rx_pipe_lfpsbcn_rxelecidle = 1'b0;
		assign o_afe_rx_pipe_sigdet = 1'b0;
		assign o_afe_rx_symbol  = 20'h00000;
		assign o_afe_rx_symbol_clk_by_2 = 1'b0;
		assign o_afe_rx_uphy_save_calcode = 1'b0;
		assign o_afe_rx_uphy_startloop_buf = 1'b0;
		assign o_afe_rx_uphy_rx_calib_done = 1'b0;
		assign o_afe_rx_uphy_save_calcode_data = 8'h00;
		assign o_afe_rx_hsrx_clock_stop_ack = 1'b0;
		assign o_afe_pg_avddcr = 1'b0;
		assign o_afe_pg_avddio = 1'b0;
		assign o_afe_pg_dvddcr = 1'b0;
		assign o_afe_pg_static_avddcr = 1'b0;
		assign o_afe_pg_static_avddio = 1'b0;
		assign o_afe_TX_dig_reset_rel_ack  = 1'b0;
		assign o_afe_TX_pipe_TX_dn_rxdet = 1'b0;
		assign o_afe_TX_pipe_TX_dp_rxdet = 1'b0;
		assign o_dbg_l0_txclk = 1'b0;
		assign o_dbg_l0_rxclk = 1'b0;
		assign o_dbg_l1_txclk = 1'b0;
		assign o_dbg_l1_rxclk = 1'b0;
		assign o_dbg_l2_txclk = 1'b0;
		assign o_dbg_l2_rxclk = 1'b0;
		assign o_dbg_l3_txclk = 1'b0;
		assign o_dbg_l3_rxclk = 1'b0;
	end
	endgenerate
endmodule
