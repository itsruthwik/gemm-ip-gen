`timescale 1ns/1ps

module tb_16x8;
    reg clk = 0; reg reset = 1; reg start_mat_mul = 0;
    reg [63:0] a_data [0:1]; reg [63:0] b_data = 0;
    wire [63:0] a_out [0:1]; wire [63:0] b_out [0:1];
    wire done [0:1];

    tensor_slice_int8 row0 (.clk(clk), .reset(reset), .pe_reset(1'b0), .start_mat_mul(start_mat_mul), .done_mat_mul(done[0]),
        .a_data(a_data[0]), .b_data(b_data), .a_data_in(64'b0), .b_data_in(64'b0), .a_data_out(a_out[0]), .b_data_out(b_out[0]),
        .validity_mask_a_rows(8'hFF), .validity_mask_a_cols_b_rows(8'hFF), .validity_mask_b_cols(8'hFF),
        .slice_dtype(2'b00), .slice_mode(1'b0), .op(3'b000), .preload(1'b0), .no_rounding(1'b0),
        .final_mat_mul_size(8'd8), .a_loc(5'd0), .b_loc(5'd0));

    tensor_slice_int8 row1 (.clk(clk), .reset(reset), .pe_reset(1'b0), .start_mat_mul(start_mat_mul), .done_mat_mul(done[1]),
        .a_data(a_data[1]), .b_data(64'h0), .a_data_in(64'b0), .b_data_in(b_out[0]), .a_data_out(a_out[1]), .b_data_out(b_out[1]),
        .validity_mask_a_rows(8'hFF), .validity_mask_a_cols_b_rows(8'hFF), .validity_mask_b_cols(8'hFF),
        .slice_dtype(2'b00), .slice_mode(1'b0), .op(3'b000), .preload(1'b0), .no_rounding(1'b0),
        .final_mat_mul_size(8'd8), .a_loc(5'd1), .b_loc(5'd0));

    always #5 clk = ~clk;

    initial begin
        $display("Running 16x8 Test");
        #20 reset = 0;
        a_data[0] = 64'h0101010101010101; a_data[1] = 64'h0101010101010101; b_data = 64'h0101010101010101;
        #10 start_mat_mul = 1;
        #10 start_mat_mul = 0;
        fork
            begin repeat(7) @(posedge clk); a_data[0] <= 0; a_data[1] <= 0; b_data <= 0; end
            wait(done[1]);
        join
        $display("16x8 Latency: %0d cycles", ($time-30)/10);
        #20 $finish;
    end
endmodule
