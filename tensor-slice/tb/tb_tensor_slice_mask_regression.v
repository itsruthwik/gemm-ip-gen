`timescale 1ns/1ps

module tb_tensor_slice_mask_row;
    reg clk = 0, reset = 1, pe_reset = 0, start_mat_mul = 0;
    reg [63:0] a_data = 0, b_data = 0, a_data_in = 0, b_data_in = 0;
    wire [63:0] a_data_out, b_data_out;
    wire [127:0] c_data_out;
    wire c_data_available, done_mat_mul;
    reg [7:0] validity_mask_a_rows = 8'h0F;
    reg [7:0] validity_mask_a_cols_b_rows = 8'hFF;
    reg [7:0] validity_mask_b_cols = 8'hFF;
    integer kk, row_idx = 0, errors = 0;
    reg [63:0] expected [0:7];

    tensor_slice_int8 dut (
        .clk(clk), .reset(reset), .pe_reset(pe_reset),
        .start_mat_mul(start_mat_mul), .done_mat_mul(done_mat_mul),
        .a_data(a_data), .b_data(b_data), .a_data_in(a_data_in), .b_data_in(b_data_in),
        .a_data_out(a_data_out), .b_data_out(b_data_out), .c_data_out(c_data_out), .c_data_available(c_data_available),
        .validity_mask_a_rows(validity_mask_a_rows), .validity_mask_a_cols_b_rows(validity_mask_a_cols_b_rows),
        .validity_mask_b_cols(validity_mask_b_cols), .slice_dtype(2'd0), .slice_mode(1'b0), .op(3'd0),
        .preload(1'b0), .no_rounding(1'b0), .final_mat_mul_size(8'd8), .a_loc(5'd0), .b_loc(5'd0)
    );

    always #5 clk = ~clk;

    initial begin
        integer r;
        for (r = 0; r < 8; r = r + 1) begin
            expected[r] = 64'd0;
            if (r < 4) expected[r][r*8 +: 8] = r + 1;
        end
        repeat (2) @(posedge clk);
        @(negedge clk); reset = 1'b0;
        for (kk = 0; kk < 8; kk = kk + 1) begin
            @(negedge clk);
            start_mat_mul = (kk == 0);
            a_data = 64'd0;
            b_data = 64'd0;
            a_data[kk*8 +: 8] = kk + 1;
            b_data[kk*8 +: 8] = 8'd1;
            @(posedge clk);
        end
        @(negedge clk);
        start_mat_mul = 1'b0;
        a_data = 64'd0;
        b_data = 64'd0;
        wait (done_mat_mul);
        repeat (2) @(posedge clk);
        if ((errors == 0) && (row_idx == 8)) $display("mask_row PASSED");
        else begin
            if (row_idx != 8) $display("mask_row FAILED: saw %0d rows", row_idx);
            $display("mask_row FAILED with %0d errors", errors);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (c_data_available) begin
            if (c_data_out[63:0] !== expected[row_idx]) begin
                $display("mask_row ERROR row%0d data %h expected %h", row_idx, c_data_out[63:0], expected[row_idx]);
                errors = errors + 1;
            end
            row_idx = row_idx + 1;
        end
    end
endmodule

module tb_tensor_slice_mask_col;
    reg clk = 0, reset = 1, pe_reset = 0, start_mat_mul = 0;
    reg [63:0] a_data = 0, b_data = 0, a_data_in = 0, b_data_in = 0;
    wire [63:0] a_data_out, b_data_out;
    wire [127:0] c_data_out;
    wire c_data_available, done_mat_mul;
    reg [7:0] validity_mask_a_rows = 8'hFF;
    reg [7:0] validity_mask_a_cols_b_rows = 8'hFF;
    reg [7:0] validity_mask_b_cols = 8'h07;
    integer kk, row_idx = 0, errors = 0;
    reg [63:0] expected;

    tensor_slice_int8 dut (
        .clk(clk), .reset(reset), .pe_reset(pe_reset),
        .start_mat_mul(start_mat_mul), .done_mat_mul(done_mat_mul),
        .a_data(a_data), .b_data(b_data), .a_data_in(a_data_in), .b_data_in(b_data_in),
        .a_data_out(a_data_out), .b_data_out(b_data_out), .c_data_out(c_data_out), .c_data_available(c_data_available),
        .validity_mask_a_rows(validity_mask_a_rows), .validity_mask_a_cols_b_rows(validity_mask_a_cols_b_rows),
        .validity_mask_b_cols(validity_mask_b_cols), .slice_dtype(2'd0), .slice_mode(1'b0), .op(3'd0),
        .preload(1'b0), .no_rounding(1'b0), .final_mat_mul_size(8'd8), .a_loc(5'd0), .b_loc(5'd0)
    );

    always #5 clk = ~clk;

    initial begin
        expected = 64'h0000000000080808;
        repeat (2) @(posedge clk);
        @(negedge clk); reset = 1'b0;
        for (kk = 0; kk < 8; kk = kk + 1) begin
            @(negedge clk);
            start_mat_mul = (kk == 0);
            a_data = 64'h0101010101010101;
            b_data = 64'h0101010101010101;
            @(posedge clk);
        end
        @(negedge clk);
        start_mat_mul = 1'b0;
        a_data = 64'd0;
        b_data = 64'd0;
        wait (done_mat_mul);
        repeat (2) @(posedge clk);
        if ((errors == 0) && (row_idx == 8)) $display("mask_col PASSED");
        else begin
            if (row_idx != 8) $display("mask_col FAILED: saw %0d rows", row_idx);
            $display("mask_col FAILED with %0d errors", errors);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (c_data_available) begin
            if (c_data_out[63:0] !== expected) begin
                $display("mask_col ERROR row%0d data %h expected %h", row_idx, c_data_out[63:0], expected);
                errors = errors + 1;
            end
            row_idx = row_idx + 1;
        end
    end
endmodule

module tb_tensor_slice_mask_temporal;
    reg clk = 0, reset = 1, pe_reset = 0, start_mat_mul = 0;
    reg [63:0] a_data = 0, b_data = 0, a_data_in = 0, b_data_in = 0;
    wire [63:0] a_data_out, b_data_out;
    wire [127:0] c_data_out;
    wire c_data_available, done_mat_mul;
    reg [7:0] validity_mask_a_rows = 8'hFF;
    reg [7:0] validity_mask_a_cols_b_rows = 8'h55;
    reg [7:0] validity_mask_b_cols = 8'hFF;
    integer kk, row_idx = 0, errors = 0;
    reg [63:0] expected;

    tensor_slice_int8 dut (
        .clk(clk), .reset(reset), .pe_reset(pe_reset),
        .start_mat_mul(start_mat_mul), .done_mat_mul(done_mat_mul),
        .a_data(a_data), .b_data(b_data), .a_data_in(a_data_in), .b_data_in(b_data_in),
        .a_data_out(a_data_out), .b_data_out(b_data_out), .c_data_out(c_data_out), .c_data_available(c_data_available),
        .validity_mask_a_rows(validity_mask_a_rows), .validity_mask_a_cols_b_rows(validity_mask_a_cols_b_rows),
        .validity_mask_b_cols(validity_mask_b_cols), .slice_dtype(2'd0), .slice_mode(1'b0), .op(3'd0),
        .preload(1'b0), .no_rounding(1'b0), .final_mat_mul_size(8'd8), .a_loc(5'd0), .b_loc(5'd0)
    );

    always #5 clk = ~clk;

    initial begin
        expected = 64'h0404040404040404;
        repeat (2) @(posedge clk);
        @(negedge clk); reset = 1'b0;
        for (kk = 0; kk < 8; kk = kk + 1) begin
            @(negedge clk);
            start_mat_mul = (kk == 0);
            a_data = 64'h0101010101010101;
            b_data = 64'h0101010101010101;
            @(posedge clk);
        end
        @(negedge clk);
        start_mat_mul = 1'b0;
        a_data = 64'd0;
        b_data = 64'd0;
        wait (done_mat_mul);
        repeat (2) @(posedge clk);
        if ((errors == 0) && (row_idx == 8)) $display("mask_temporal PASSED");
        else begin
            if (row_idx != 8) $display("mask_temporal FAILED: saw %0d rows", row_idx);
            $display("mask_temporal FAILED with %0d errors", errors);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (c_data_available) begin
            if (c_data_out[63:0] !== expected) begin
                $display("mask_temporal ERROR row%0d data %h expected %h", row_idx, c_data_out[63:0], expected);
                errors = errors + 1;
            end
            row_idx = row_idx + 1;
        end
    end
endmodule

module tb_tensor_slice_mask_5x5;
    reg clk = 0, reset = 1, pe_reset = 0, start_mat_mul = 0;
    reg [63:0] a_data = 0, b_data = 0, a_data_in = 0, b_data_in = 0;
    wire [63:0] a_data_out, b_data_out;
    wire [127:0] c_data_out;
    wire c_data_available, done_mat_mul;
    reg [7:0] validity_mask_a_rows = 8'h1F;
    reg [7:0] validity_mask_a_cols_b_rows = 8'h1F;
    reg [7:0] validity_mask_b_cols = 8'h1F;
    integer kk, row_idx = 0, errors = 0;
    reg [63:0] expected [0:7];

    tensor_slice_int8 dut (
        .clk(clk), .reset(reset), .pe_reset(pe_reset),
        .start_mat_mul(start_mat_mul), .done_mat_mul(done_mat_mul),
        .a_data(a_data), .b_data(b_data), .a_data_in(a_data_in), .b_data_in(b_data_in),
        .a_data_out(a_data_out), .b_data_out(b_data_out), .c_data_out(c_data_out), .c_data_available(c_data_available),
        .validity_mask_a_rows(validity_mask_a_rows), .validity_mask_a_cols_b_rows(validity_mask_a_cols_b_rows),
        .validity_mask_b_cols(validity_mask_b_cols), .slice_dtype(2'd0), .slice_mode(1'b0), .op(3'd0),
        .preload(1'b0), .no_rounding(1'b0), .final_mat_mul_size(8'd5), .a_loc(5'd0), .b_loc(5'd0)
    );

    always #5 clk = ~clk;

    initial begin
        integer r, c;
        for (r = 0; r < 8; r = r + 1) begin
            expected[r] = 64'd0;
            if (r < 5) begin
                for (c = 0; c < 5; c = c + 1) expected[r][c*8 +: 8] = 8'd5;
            end
        end
        repeat (2) @(posedge clk);
        @(negedge clk); reset = 1'b0;
        for (kk = 0; kk < 5; kk = kk + 1) begin
            @(negedge clk);
            start_mat_mul = (kk == 0);
            a_data = 64'h0000000101010101;
            b_data = 64'h0000000101010101;
            @(posedge clk);
        end
        @(negedge clk);
        start_mat_mul = 1'b0;
        a_data = 64'd0;
        b_data = 64'd0;
        wait (done_mat_mul);
        repeat (2) @(posedge clk);
        if ((errors == 0) && (row_idx == 8)) $display("mask_5x5 PASSED");
        else begin
            if (row_idx != 8) $display("mask_5x5 FAILED: saw %0d rows", row_idx);
            $display("mask_5x5 FAILED with %0d errors", errors);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (c_data_available) begin
            if (c_data_out[63:0] !== expected[row_idx]) begin
                $display("mask_5x5 ERROR row%0d data %h expected %h", row_idx, c_data_out[63:0], expected[row_idx]);
                errors = errors + 1;
            end
            row_idx = row_idx + 1;
        end
    end
endmodule
