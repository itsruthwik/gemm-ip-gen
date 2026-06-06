`timescale 1ns/1ps
// Self-checking unit testbench for tensor_slice_int8 bias-preload.
// Row/col protocol: each beat is one A row (8 K-values in bytes) or one B col.
// K is controlled by final_mat_mul_size (in_phase = local_idx < final_mat_mul_size).
module tb_tensor_slice_bias_preload;
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

    integer errors = 0;
    integer row_idx = 0;
    integer op_num = 0;
    integer cycle_ctr = 0;
    reg [63:0] expected [0:7];

    tensor_slice_int8 dut (
        .clk(clk), .reset(reset), .pe_reset(pe_reset),
        .start_mat_mul(start_mat_mul), .done_mat_mul(done_mat_mul),
        .a_data(a_data), .b_data(b_data),
        .a_data_in(a_data_in), .b_data_in(b_data_in),
        .a_data_out(a_data_out), .b_data_out(b_data_out),
        .c_data_out(c_data_out), .c_data_available(c_data_available),
        .validity_mask_a_rows(validity_mask_a_rows),
        .validity_mask_a_cols_b_rows(validity_mask_a_cols_b_rows),
        .validity_mask_b_cols(validity_mask_b_cols),
        .slice_dtype(slice_dtype), .slice_mode(slice_mode),
        .op(op), .preload(preload), .no_rounding(no_rounding),
        .final_mat_mul_size(final_mat_mul_size),
        .a_loc(a_loc), .b_loc(b_loc)
    );

    always #5 clk = ~clk;

    function integer sat8(input integer v);
        if (v > 127) sat8 = 127;
        else if (v < -128) sat8 = -128;
        else sat8 = v;
    endfunction

    // Compute expected row: for each col c (0..7), acc = sum_k(A[row][k] * B[k][c]) + bias[c]
    function [63:0] pack_exp(
        input [7:0] k_len,              // K dimension
        input integer row_val,           // A value for this row (all K positions)
        input integer b0, input integer b1, // bias bytes 0,1
        input [7:0] ma, input [7:0] mb   // row/col masks
    );
        reg [63:0] rv;
        integer c;
        begin
            rv = 64'd0;
            for (c = 0; c < 8; c = c + 1) begin
                if ((ma[0] ^ ma[0] == 0) ? 1 : 0) begin  // always true hack
                end
            end
        end
    endfunction

    // Row/col protocol: drive num_rows A beats and num_cols B beats.
    // Each beat: 64 bits = 8 bytes = 8 K-values for one row/col.
    // Block-diagonal: A[row_val] replicated across all K bytes.
    // B all ones (identity-like).
    task feed_block_diag(
        input integer num_beats,        // beats to drive (rows/cols)
        input integer row_val_start,    // A value for row 0
        input integer row_val_step,     // step between rows
        input integer use_pre           // preload bias first
    );
        integer t;
        reg [63:0] a_beat;
        reg [63:0] b_beat;
        integer kk;
        begin
            if (use_pre) begin
                @(negedge clk);
                preload = 1; start_mat_mul = 0;
                a_data = 0; a_data_in = 0; b_data_in = 0;
                @(posedge clk);
            end
            for (t = 0; t < num_beats; t = t + 1) begin
                a_beat = 64'd0;
                b_beat = 64'd0;
                for (kk = 0; kk < 8; kk = kk + 1) begin
                    a_beat[kk*8 +: 8] = 8'(row_val_start + t * row_val_step);
                    b_beat[kk*8 +: 8] = 8'd1;
                end
                @(negedge clk);
                preload = 0; start_mat_mul = (t == 0);
                a_data = a_beat; b_data = b_beat;
                a_data_in = 0; b_data_in = 0;
                @(posedge clk);
            end
            @(negedge clk);
            start_mat_mul = 0; a_data = 0; b_data = 0;
            a_data_in = 0; b_data_in = 0;
        end
    endtask

    // Compute expected row: for each col c, acc = sum_k(A[row][k] * B[k][c]) + bias[c]
    // With block-diagonal: A = row_val for all k, B = 1 for all (k,c), K = k_len
    // Sum = k_len * row_val, then add bias at position c
    function [63:0] expect_row(
        input integer k_len,
        input integer row_val,
        input integer b0,
        input integer b1,
        input [7:0] row_mask,
        input [7:0] col_mask,
        input integer row_index
    );
        reg [63:0] rv;
        integer c;
        integer sum_before_bias;
        integer total;
        begin
            rv = 64'd0;
            if (row_mask[row_index]) begin
                sum_before_bias = k_len * row_val;
                for (c = 0; c < 8; c = c + 1) begin
                    if (col_mask[c]) begin
                        if (c == 0)
                            total = sum_before_bias + b0;
                        else if (c == 1)
                            total = sum_before_bias + b1;
                        else
                            total = sum_before_bias;
                        rv[c*8 +: 8] = 8'(sat8(total));
                    end else begin
                        rv[c*8 +: 8] = 8'd0;
                    end
                end
            end else begin
                rv = 64'd0;
            end
            expect_row = rv;
        end
    endfunction

    // Run one test case
    task run_one(
        input integer k_len,
        input integer use_pre,
        input integer b0, input integer b1,
        input [7:0] ma, input [7:0] mb,
        input integer row_val_start, input integer row_val_step,
        input [1024:1] tname
    );
        integer r;
        begin
            $display("=== %0s ===", tname);
            op_num = op_num + 1;
            row_idx = 0;
            validity_mask_a_rows = ma;
            validity_mask_b_cols = mb;
            final_mat_mul_size = k_len;

            for (r = 0; r < 8; r = r + 1)
                expected[r] = expect_row(k_len, row_val_start + r * row_val_step,
                                          b0, b1, ma, mb, r);

            // Set bias data for preload
            if (use_pre) begin
                b_data = 64'd0;
                b_data[0*8 +: 8] = b0;
                b_data[1*8 +: 8] = b1;
            end

            feed_block_diag(k_len, row_val_start, row_val_step, use_pre);
            @(negedge clk);
            wait (done_mat_mul);
            @(posedge clk);
            #1;

            if (row_idx !== 8) begin
                $display("FAIL op%0d %0s: got %0d rows expected 8", op_num, tname, row_idx);
                errors = errors + 1;
            end
        end
    endtask

    // ── Main test sequence ─────────────────────────────────────────────────────
    initial begin
        repeat (3) @(posedge clk); @(negedge clk); reset = 0;

        // Test 1: No-preload regression. A rows = 1,2,3,...,8. B = all ones.
        // cell(r,c) = 8 * (r+1) (K=8 sum of (r+1)*1)
        run_one(8, 0, 0, 0, 8'hFF, 8'hFF, 1, 1, "Test 1: no-preload regression");

        // Test 2: Zero-bias preload (should be same as no-preload)
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(8, 1, 0, 0, 8'hFF, 8'hFF, 1, 1, "Test 2: zero-bias preload");

        // Test 3: Positive and negative bias
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(8, 1, 5, -3, 8'hFF, 8'hFF, 1, 1, "Test 3: pos/neg bias");

        // Test 4: Saturation bias. b0=127, b1=-128.
        // For r=7: 8*8=64 + 127=191 → sat to 127. 64 - 128 = -64 → OK.
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(8, 1, 127, -128, 8'hFF, 8'hFF, 1, 1, "Test 4: saturation bias");

        // Test 5: Masked lanes. Only rows 0,1 and cols 0,1.
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(8, 1, 5, -3, 8'h03, 8'h03, 1, 1, "Test 5: masked lanes");

        // Test 6: Back-to-back: preload then no-preload
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(8, 1, 5, -3, 8'hFF, 8'hFF, 1, 1, "Test 6a: preload op");
        run_one(8, 0, 0, 0, 8'hFF, 8'hFF, 1, 1, "Test 6b: no-preload after preload");

        #100;
        if (errors == 0) $display("TENSOR_SLICE_BIAS_PRELOAD_PASSED");
        else             $display("TENSOR_SLICE_BIAS_PRELOAD_FAILED with %0d errors", errors);
        $finish;
    end

    // ── Output collector ──────────────────────────────────────────────────────
    always @(posedge clk) begin
        cycle_ctr <= cycle_ctr + 1;
        if (reset) begin
            row_idx <= 0;
        end else if (c_data_available && row_idx < 8) begin
            if (c_data_out[63:0] !== expected[row_idx]) begin
                $display("FAIL op%0d row%0d: got %h expected %h",
                         op_num, row_idx, c_data_out[63:0], expected[row_idx]);
                errors = errors + 1;
            end
            row_idx <= row_idx + 1;
        end
    end

endmodule
