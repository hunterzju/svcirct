module dff_sync_reset (
input  wire data  , // Data Input
input  wire clk   , // Clock Input
input  wire reset , // Reset input 
output reg  q       // Q output
);
//-------------Code Starts Here---------
always_ff @ ( posedge clk)
if (~reset) begin
  q <= 1'b0;
end  else begin
  q <= data;
end

endmodule //End Of Module dff_sync_reset

// testbench
module tb_top ();
    reg clk;
    reg reset;
    reg d;
    wire q;

    dff_sync_reset dff_sync_reset_0(.data(d),
                                    .clk(clk),
                                    .reset(reset),
                                    .q(q));
    
    always #10  clk <= ~clk;

    initial begin
        reset <= 0;
        d <= 0;

        #10 reset <= 1;
        #5  d <= 1;
        #8  d <= 0;
        #2  d <= 1;
        #10 d <= 0;
    end

endmodule
