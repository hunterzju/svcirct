`timescale 10ns/1ns
// FPGA projects using Verilog/ VHDL
// fpga4student.com: FPGA projects, Verilog projects, VHDL projects
// Verilog code for up-down counter with testbench
// Testbench Verilog code for up-down counter
module updowncounter_tb();
reg clk, reset,up_down;
wire [3:0] counter;

up_down_counter dut(clk, reset,up_down, counter);
initial begin 
clk=0;
forever #5ns clk=~clk;
end
initial begin
reset=1;
up_down=0;
#20ns;
reset=0;
#200ns;
up_down=1;
end
endmodule 
