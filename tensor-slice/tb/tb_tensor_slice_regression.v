`timescale 1ns/1ps

module tb_tensor_slice_regression;
    reg clk = 0;
    reg reset = 1;
    reg pe_reset = 0;
    reg start_mat_mul = 0;
    reg [63:0] a_data = 0;
    reg [63:0] b_data = 0;
    reg [63:0] a_data_in = 0;
    reg [63:0] b_data_in = 0;
    wire [63:0] a_data_out;
    wire [63:0] b_data_out;
    wire [127:0] c_data_out;
    wire c_data_available;
    wire done_mat_mul;

    localparam integer NUM_OPS = 4;

    reg [7:0] validity_mask_a_rows = 8'hFF;
    reg [7:0] validity_mask_a_cols_b_rows = 8'hFF;
    reg [7:0] validity_mask_b_cols = 8'hFF;
    reg [1:0] slice_dtype = 2'd0;
    reg slice_mode = 1'b0;
    reg [2:0] op = 3'd0;
    reg preload = 1'b0;
    reg no_rounding = 1'b0;
    reg [7:0] final_mat_mul_size = 8'd8;
    reg [4:0] a_loc = 5'd0;
    reg [4:0] b_loc = 5'd0;

    integer cycle_ctr = 0;
    integer drive_op_idx = 0;
    integer check_op_idx = 0;
    integer errors = 0;
    integer done_count = 0;
    integer launch_cycle [0:NUM_OPS-1];
    integer first_avail_cycle [0:NUM_OPS-1];
    integer done_cycle [0:NUM_OPS-1];
    integer row_idx [0:NUM_OPS-1];
    integer expect_first_avail_delta [0:NUM_OPS-1];
    integer expect_done_delta [0:NUM_OPS-1];
    reg [63:0] expected [0:NUM_OPS-1][0:7];

    tensor_slice_int8 dut (
        .clk(clk),
        .reset(reset),
        .pe_reset(pe_reset),
        .start_mat_mul(start_mat_mul),
        .done_mat_mul(done_mat_mul),
        .a_data(a_data),
        .b_data(b_data),
        .a_data_in(a_data_in),
        .b_data_in(b_data_in),
        .a_data_out(a_data_out),
        .b_data_out(b_data_out),
        .c_data_out(c_data_out),
        .c_data_available(c_data_available),
        .validity_mask_a_rows(validity_mask_a_rows),
        .validity_mask_a_cols_b_rows(validity_mask_a_cols_b_rows),
        .validity_mask_b_cols(validity_mask_b_cols),
        .slice_dtype(slice_dtype),
        .slice_mode(slice_mode),
        .op(op),
        .preload(preload),
        .no_rounding(no_rounding),
        .final_mat_mul_size(final_mat_mul_size),
        .a_loc(a_loc),
        .b_loc(b_loc)
    );

    always #5 clk = ~clk;

    task automatic drive_identity_op(
        input integer variant,
        input integer k_cycles,
        input integer use_chain_inputs
    );
        integer kk;
        reg [63:0] a_cycle;
        reg [63:0] b_cycle;
        begin
            for (kk = 0; kk < k_cycles; kk = kk + 1) begin
                a_cycle = 64'd0;
                b_cycle = 64'd0;
                a_cycle[kk*8 +: 8] = (variant == 0) ? (kk + 1) : (8 - kk);
                b_cycle[kk*8 +: 8] = 8'd1;
                @(negedge clk);
                start_mat_mul = (kk == 0);
                if (use_chain_inputs != 0) begin
                    a_data = 64'd0;
                    b_data = 64'd0;
                    a_data_in = a_cycle;
                    b_data_in = b_cycle;
                end else begin
                    a_data = a_cycle;
                    b_data = b_cycle;
                    a_data_in = 64'd0;
                    b_data_in = 64'd0;
                end
                @(posedge clk);
            end
            @(negedge clk);
            start_mat_mul = 1'b0;
            a_data = 64'd0;
            b_data = 64'd0;
            a_data_in = 64'd0;
            b_data_in = 64'd0;
        end
    endtask

    task automatic drive_masked_ones_op(input integer k_cycles, input integer use_chain_inputs);
        integer kk;
        reg [63:0] a_cycle;
        reg [63:0] b_cycle;
        begin
            a_cycle = 64'd0;
            b_cycle = 64'd0;
            for (kk = 0; kk < 5; kk = kk + 1) begin
                a_cycle[kk*8 +: 8] = 8'd1;
                b_cycle[kk*8 +: 8] = 8'd1;
            end

            for (kk = 0; kk < k_cycles; kk = kk + 1) begin
                @(negedge clk);
                start_mat_mul = (kk == 0);
                if (use_chain_inputs != 0) begin
                    a_data = 64'd0;
                    b_data = 64'd0;
                    a_data_in = a_cycle;
                    b_data_in = b_cycle;
                end else begin
                    a_data = a_cycle;
                    b_data = b_cycle;
                    a_data_in = 64'd0;
                    b_data_in = 64'd0;
                end
                @(posedge clk);
            end
            @(negedge clk);
            start_mat_mul = 1'b0;
            a_data = 64'd0;
            b_data = 64'd0;
            a_data_in = 64'd0;
            b_data_in = 64'd0;
        end
    endtask

    initial begin
        integer r;
        integer c;
        for (r = 0; r < 8; r = r + 1) begin
            expected[0][r] = 64'd0;
            expected[0][r][r*8 +: 8] = r + 1;
            expected[1][r] = 64'd0;
            expected[1][r][r*8 +: 8] = 8 - r;
            expected[2][r] = 64'd0;
            if (r < 5) begin
                for (c = 0; c < 5; c = c + 1) begin
                    expected[2][r][c*8 +: 8] = 8'd5;
                end
            end
            expected[3][r] = 64'd0;
            expected[3][r][r*8 +: 8] = r + 1;
        end
        for (r = 0; r < NUM_OPS; r = r + 1) begin
            launch_cycle[r] = -1;
            first_avail_cycle[r] = -1;
            done_cycle[r] = -1;
            row_idx[r] = 0;
        end
        expect_first_avail_delta[0] = 17;
        expect_done_delta[0] = 25;
        expect_first_avail_delta[1] = 17;
        expect_done_delta[1] = 25;
        expect_first_avail_delta[2] = 14;
        expect_done_delta[2] = 22;
        expect_first_avail_delta[3] = 33;
        expect_done_delta[3] = 41;

        $display("=== tensor_slice_int8 regression start ===");

        repeat (2) @(posedge clk);
        @(negedge clk);
        reset = 1'b0;

        validity_mask_a_rows = 8'hFF;
        validity_mask_a_cols_b_rows = 8'hFF;
        validity_mask_b_cols = 8'hFF;
        final_mat_mul_size = 8'd8;
        a_loc = 5'd0;
        b_loc = 5'd0;
        drive_identity_op(0, 8, 0);
        fork
            begin : wait_done0
                wait (done_mat_mul);
            end
            begin : timeout0
                repeat (80) @(posedge clk);
                $display("TIMEOUT op0");
                errors = errors + 1;
                disable wait_done0;
            end
        join_any
        disable wait_done0;
        disable timeout0;
        @(posedge clk);
        drive_op_idx = 1;

        drive_identity_op(1, 8, 0);
        fork
            begin : wait_done1
                wait (done_mat_mul);
            end
            begin : timeout1
                repeat (80) @(posedge clk);
                $display("TIMEOUT op1");
                errors = errors + 1;
                disable wait_done1;
            end
        join_any
        disable wait_done1;
        disable timeout1;
        @(posedge clk);
        drive_op_idx = 2;

        validity_mask_a_rows = 8'h1F;
        validity_mask_a_cols_b_rows = 8'h1F;
        validity_mask_b_cols = 8'h1F;
        final_mat_mul_size = 8'd5;
        a_loc = 5'd0;
        b_loc = 5'd0;
        drive_masked_ones_op(5, 0);
        fork
            begin : wait_done2
                wait (done_mat_mul);
            end
            begin : timeout2
                repeat (80) @(posedge clk);
                $display("TIMEOUT op2");
                errors = errors + 1;
                disable wait_done2;
            end
        join_any
        disable wait_done2;
        disable timeout2;
        @(posedge clk);
        drive_op_idx = 3;

        validity_mask_a_rows = 8'hFF;
        validity_mask_a_cols_b_rows = 8'hFF;
        validity_mask_b_cols = 8'hFF;
        final_mat_mul_size = 8'd8;
        a_loc = 5'd1;
        b_loc = 5'd1;
        drive_identity_op(0, 8, 1);
        fork
            begin : wait_done3
                wait (done_mat_mul);
            end
            begin : timeout3
                repeat (120) @(posedge clk);
                $display("TIMEOUT op3");
                errors = errors + 1;
                disable wait_done3;
            end
        join_any
        disable wait_done3;
        disable timeout3;
        repeat (10) @(posedge clk);

        if (done_count != NUM_OPS) begin
            $display("ERROR saw %0d done pulses, expected %0d", done_count, NUM_OPS);
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("tensor_slice regression PASSED");
        end else begin
            $display("tensor_slice regression FAILED with %0d errors", errors);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (reset) begin
            cycle_ctr <= 0;
        end else begin
            cycle_ctr <= cycle_ctr + 1;
        end

        if (c_data_available === 1'b1) begin
            if (first_avail_cycle[check_op_idx] == -1) begin
                first_avail_cycle[check_op_idx] = cycle_ctr;
                if ((cycle_ctr - launch_cycle[check_op_idx]) !== expect_first_avail_delta[check_op_idx]) begin
                    $display("ERROR op%0d first c_data_available delta %0d, expected %0d", check_op_idx, cycle_ctr - launch_cycle[check_op_idx], expect_first_avail_delta[check_op_idx]);
                    errors = errors + 1;
                end
            end

            if (row_idx[check_op_idx] >= 8) begin
                $display("ERROR op%0d extra output row at cycle %0d", check_op_idx, cycle_ctr);
                errors = errors + 1;
            end else if (c_data_out[63:0] !== expected[check_op_idx][row_idx[check_op_idx]]) begin
                $display("ERROR op%0d row%0d data %h expected %h", check_op_idx, row_idx[check_op_idx], c_data_out[63:0], expected[check_op_idx][row_idx[check_op_idx]]);
                errors = errors + 1;
            end
            row_idx[check_op_idx] = row_idx[check_op_idx] + 1;
        end

        if (done_mat_mul === 1'b1) begin
            done_cycle[check_op_idx] = cycle_ctr;
            if ((cycle_ctr - launch_cycle[check_op_idx]) !== expect_done_delta[check_op_idx]) begin
                $display("ERROR op%0d done_mat_mul delta %0d, expected %0d", check_op_idx, cycle_ctr - launch_cycle[check_op_idx], expect_done_delta[check_op_idx]);
                errors = errors + 1;
            end
            if (row_idx[check_op_idx] !== 8) begin
                $display("ERROR op%0d saw %0d rows before done, expected 8", check_op_idx, row_idx[check_op_idx]);
                errors = errors + 1;
            end
            done_count = done_count + 1;
            check_op_idx = check_op_idx + 1;
        end

        if (start_mat_mul === 1'b1 && !reset) begin
            launch_cycle[drive_op_idx] = cycle_ctr;
        end
    end

endmodule
