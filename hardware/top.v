`timescale 1 ps / 1 ps

module top(
   input sys_clk_n,
   input sys_clk_p,
   input [3:0] key,      /* active low */
   output reg [3:0] led, /*active low */
   output uart_tx_
);
   localparam clock_Mhz = 25_000_000;

   reg[31:0] timer_cnt = 0;
   wire sys_clk;
   wire rst_n = key[0];

   IBUFDS IBUFDS_inst (
      .O(sys_clk),
      .I(sys_clk_p),
      .IB(sys_clk_n)
   );

   wire tx_data_ready;
   uart_tx #(
      .CLK_FRE(25)
   ) transmitter (
      sys_clk,
      rst_n,
      65,
      tx_data_ready,
      uart_tx_
   );

   always @(posedge sys_clk) begin
       if (!rst_n) begin
         led[0] <= 1'b1 ;
         timer_cnt <= 32'd0 ;
       end else if (timer_cnt >= (clock_Mhz-1)) begin
           led[0] <= ~led[0];
           timer_cnt <= 32'd0;
       end else begin
           led[0] <= led[0];
           timer_cnt <= timer_cnt + 32'd1;
       end
       led[1] <= 1;
       led[2] <= 1;
       led[3] <= 1;
   end
endmodule

module uart_tx #(
   parameter CLK_FRE = 50,      //clock frequency(Mhz)
   parameter BAUD_RATE = 115200 //serial baud rate
) (
   input                        clk,              //clock input
   input                        rst_n,            //asynchronous reset input, low active
   input[7:0]                   tx_data,          //data to send
   input                        tx_data_valid,    //data to be sent is valid
   output reg                   tx_data_ready,    //send ready
   output                       tx_pin            //serial data output
);

   //calculates the clock cycle for baud rate
   localparam                       CYCLE = CLK_FRE * 1000000 / BAUD_RATE;
   //state machine code
   localparam                       S_IDLE       = 1;
   localparam                       S_START      = 2; //start bit
   localparam                       S_SEND_BYTE  = 3; //data bits
   localparam                       S_STOP       = 4; //stop bit
   reg[2:0]                         state;
   reg[2:0]                         next_state;
   reg[15:0]                        cycle_cnt; //baud counter
   reg[2:0]                         bit_cnt;//bit counter
   reg[7:0]                         tx_data_latch; //latch data to send
   reg                              tx_reg; //serial data output
   assign tx_pin = tx_reg;

   always@(posedge clk or negedge rst_n) begin
      if(rst_n == 1'b0)
         state <= S_IDLE;
      else
         state <= next_state;
   end

   always @(*) begin
      case(state)
         S_IDLE:
            if (tx_data_valid == 1'b1)
               next_state <= S_START;
            else
               next_state <= S_IDLE;
         S_START:
            if (cycle_cnt == CYCLE - 1)
               next_state <= S_SEND_BYTE;
            else
               next_state <= S_START;
         S_SEND_BYTE:
            if (cycle_cnt == CYCLE - 1  && bit_cnt == 3'd7)
               next_state <= S_STOP;
            else
               next_state <= S_SEND_BYTE;
         S_STOP:
            if (cycle_cnt == CYCLE - 1)
               next_state <= S_IDLE;
            else
               next_state <= S_STOP;
         default:
            next_state <= S_IDLE;
      endcase
   end

   always @(posedge clk or negedge rst_n) begin
      if (rst_n == 1'b0) begin
         tx_data_ready <= 1'b0;
      end else if(state == S_IDLE)
         if (tx_data_valid == 1'b1)
            tx_data_ready <= 1'b0;
         else
            tx_data_ready <= 1'b1;
      else if (state == S_STOP && cycle_cnt == CYCLE - 1)
         tx_data_ready <= 1'b1;
   end

   always @(posedge clk or negedge rst_n) begin
      if (rst_n == 1'b0) begin
         tx_data_latch <= 8'd0;
      end else if (state == S_IDLE && tx_data_valid == 1'b1)
         tx_data_latch <= tx_data;
   end

   always @(posedge clk or negedge rst_n) begin
      if (rst_n == 1'b0) begin
         bit_cnt <= 3'd0;
      end else if(state == S_SEND_BYTE)
         if (cycle_cnt == CYCLE - 1)
            bit_cnt <= bit_cnt + 3'd1;
         else
            bit_cnt <= bit_cnt;
      else
         bit_cnt <= 3'd0;
   end


   always @(posedge clk or negedge rst_n) begin
      if (rst_n == 1'b0)
         cycle_cnt <= 16'd0;
      else if ((state == S_SEND_BYTE && cycle_cnt == CYCLE - 1) || next_state != state)
         cycle_cnt <= 16'd0;
      else
         cycle_cnt <= cycle_cnt + 16'd1;
   end

   always @(posedge clk or negedge rst_n) begin
      if(rst_n == 1'b0)
         tx_reg <= 1'b1;
      else
         case(state)
            S_IDLE,S_STOP:
               tx_reg <= 1'b1;
            S_START:
               tx_reg <= 1'b0;
            S_SEND_BYTE:
               tx_reg <= tx_data_latch[bit_cnt];
            default:
               tx_reg <= 1'b1;
         endcase
   end

endmodule
