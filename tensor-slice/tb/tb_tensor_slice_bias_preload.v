`timescale 1ns/1ps
// Self-checking unit testbench for tensor_slice_int8 bias-preload.
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
    reg [7:0] final_mat_mul_size = 8'd4;
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

    // ── Helpers ───────────────────────────────────────────────────────────────
    function integer sat8(input integer v);
        if (v > 127) sat8 = 127;
        else if (v < -128) sat8 = -128;
        else sat8 = v;
    endfunction

    function [63:0] pack_exp(input integer k, input integer r, input integer b0, input integer b1,
                              input integer ma, input integer mb);
        reg [63:0] rv;
        integer c, bc;
        begin
            rv = 64'd0;
            for (c = 0; c < 8; c = c + 1) begin
                if (((ma >> r) & 1) && ((mb >> c) & 1)) begin
                    bc = (c == 0) ? b0 : ((c == 1) ? b1 : 0);
                    bc = bc + ((r == 0 && c == 0) ? k * (k + 1) / 2 : 0);
                end else begin
                    bc = 0;
                end
                rv[c*8 +: 8] = 8'(sat8(bc));
            end
            pack_exp = rv;
        end
    endfunction

    // ── Drive one feed cycle ──────────────────────────────────────────────────
    task feed(input integer k, input integer use_pre);
        integer kk;
        reg [63:0] ac, bc;
        begin
            if (use_pre) begin
                @(negedge clk);
                preload = 1; start_mat_mul = 0; a_data = 0;
                a_data_in = 0; b_data_in = 0;
                @(posedge clk);
            end
            for (kk = 0; kk < k; kk = kk + 1) begin
                ac = 64'd0; bc = 64'd0;
                ac[0*8 +: 8] = 8'(kk + 1);
                bc[0*8 +: 8] = 8'd1;
                @(negedge clk);
                preload = 0; start_mat_mul = (kk == 0);
                a_data = ac; b_data = bc;
                a_data_in = 0; b_data_in = 0;
                @(posedge clk);
            end
            @(negedge clk);
            start_mat_mul = 0; a_data = 0; b_data = 0;
            a_data_in = 0; b_data_in = 0;
        end
    endtask

    // ── Run one test case ──────────────────────────────────────────────────────
    // Returns when done_mat_mul fires.
    task run_one(input integer k, input integer use_pre,
                 input integer b0, input integer b1,
                 input integer ma, input integer mb,
                 input [80*8:1] tname);
        integer r;
        begin
            $display("=== %s ===", tname);
            op_num = op_num + 1;
            row_idx = 0;
            validity_mask_a_rows = ma; validity_mask_b_cols = mb;
            final_mat_mul_size = k;
            for (r = 0; r < 8; r = r + 1)
                expected[r] = pack_exp(k, r, b0, b1, ma, mb);

            if (use_pre) begin
                b_data = 64'd0;
                b_data[0*8 +: 8] = b0;
                b_data[1*8 +: 8] = b1;
            end

            feed(k, use_pre);
            @(negedge clk);
            wait (done_mat_mul);
            @(posedge clk);
            #1; // avoid race with always block

            if (row_idx !== 8) begin
                $display("FAIL op%0d %s: got %0d rows expected 8", op_num, tname, row_idx);
                errors = errors + 1;
            end
        end
    endtask

    // ── Main test sequence ─────────────────────────────────────────────────────
    initial begin
        repeat (3) @(posedge clk); @(negedge clk); reset = 0;

        // 1. No-preload regression
        run_one(4, 0, 0, 0, 8'hFF, 8'hFF, "Test 1: no-preload regression");

        // 2. Zero-bias preload
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(4, 1, 0, 0, 8'hFF, 8'hFF, "Test 2: zero-bias preload");

        // 3. Positive and negative bias
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(4, 1, 5, -3, 8'hFF, 8'hFF, "Test 3: pos/neg bias");

        // 4. Saturation
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(4, 1, 127, -128, 8'hFF, 8'hFF, "Test 4: saturation bias");

        // 5. Masked lanes
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(4, 1, 5, -3, 8'h03, 8'h03, "Test 5: masked lanes");

        // 6. Back-to-back: preload then no-preload
        reset = 1; @(posedge clk); @(negedge clk); reset = 0;
        run_one(4, 1, 5, -3, 8'hFF, 8'hFF, "Test 6a: preload op");
        run_one(4, 0, 0, 0, 8'hFF, 8'hFF, "Test 6b: no-preload after preload");

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
