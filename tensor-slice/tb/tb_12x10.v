`timescale 1ns/1ps

module tb_12x10;
    reg clk = 0; reg reset = 1; reg start_mat_mul = 0;
    reg [63:0] a_data [0:1]; reg [63:0] b_data [0:1];
    wire [63:0] a_out [0:1][0:1]; wire [63:0] b_out [0:1][0:1];
    wire done [0:1][0:1];

    genvar r, c;
    generate
        for (r=0; r<2; r=r+1) begin : row
            for (c=0; c<2; c=c+1) begin : col
                wire [63:0] a_in = (c==0) ? 64'b0 : a_out[r][c-1];
                wire [63:0] b_in = (r==0) ? 64'b0 : b_out[r-1][c];
                tensor_slice_int8 slice (
                    .clk(clk), .reset(reset), .pe_reset(1'b0),
                    .start_mat_mul(start_mat_mul), .done_mat_mul(done[r][c]),
                    .a_data(a_data[r]), .b_data(b_data[c]),
                    .a_data_in(a_in), .b_data_in(b_in),
                    .a_data_out(a_out[r][c]), .b_data_out(b_out[r][c]),
                    .validity_mask_a_rows((r==0) ? 8'hFF : 8'h0F), 
                    .validity_mask_a_cols_b_rows(8'hFF),           
                    .validity_mask_b_cols((c==0) ? 8'hFF : 8'h03), 
                    .slice_dtype(2'b00), .slice_mode(1'b0), .op(3'b000), .preload(1'b0), .no_rounding(1'b0),
                    .final_mat_mul_size(8'd10), .a_loc(r[4:0]), .b_loc(c[4:0])
                );
            end
        end
    endgenerate

    always #5 clk = ~clk;

    initial begin
        $display("Running 12x10 Test");
        #20 reset = 0;
        a_data[0] = 64'h0101010101010101; a_data[1] = 64'h0101010101010101;
        b_data[0] = 64'h0101010101010101; b_data[1] = 64'h0101010101010101;
        #10 start_mat_mul = 1;
        #10 start_mat_mul = 0;
        fork
            begin repeat(9) @(posedge clk); a_data[0] <= 0; a_data[1] <= 0; b_data[0] <= 0; b_data[1] <= 0; end
            wait(done[1][1]);
        join
        $display("12x10 Latency: %0d cycles", ($time-30)/10);
        #20 $finish;
    end
endmodule
