`timescale 1ns/1ps
// Multi-tile Vitis back-to-back regression: 2 vectors, NO reset between them.
// Shape: 16x8x16 (2x2 grid).  All A=1, B=2, bias=0.  Expect C[i][j] = 16 = 0x10.
module tb_vitis_bb_16x16;
    reg ap_clk = 0; reg ap_rst = 1; reg ap_ce = 1;
    reg [127:0] a_tdata = 0, bias_tdata = 0, b_tdata = 0;
    reg a_tvalid = 0, bias_tvalid = 0, b_tvalid = 0;
    wire a_tready, bias_tready, b_tready;
    wire [127:0] c_tdata; wire c_tvalid; reg c_tready = 1;

    vitis_16x8x16 dut (
        .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_ce(ap_ce),
        .a_tdata(a_tdata), .a_tvalid(a_tvalid), .a_tready(a_tready),
        .bias_tdata(bias_tdata), .bias_tvalid(bias_tvalid), .bias_tready(bias_tready),
        .b_tdata(b_tdata), .b_tvalid(b_tvalid), .b_tready(b_tready),
        .c_tdata(c_tdata), .c_tvalid(c_tvalid), .c_tready(c_tready)
    );
    always #5 ap_clk = ~ap_clk;

    // All A rows = 1..8, all B cols = 2.  16 A rows, 16 B cols, K=8.
    // A: 2 tile-rows × 64 = 128 bits. Row t byte k = A[t][k].
    // B: 2 tile-cols × 64 = 128 bits. Col t byte k = B[k][t].
    // Expected: C[i][j] = K * 1 * 2 = 8*2 = 16 = 0x10.  But A=1..8 per row →
    // C[i][j] = sum_k A[i][k]*B[k][j] = (1+2+...+8)*2 = 36*2 = 72 = 0x48.
    // Use uniform A=1, B=2 for simplicity: C = 16.
    integer v, i, fail;

    initial begin
        $display("=== Vitis back-to-back 16x8x16 (2 vectors, no reset) ===");
        fail = 0;
        ap_rst <= 1; repeat(2) @(posedge ap_clk); ap_rst <= 0;

        for (v = 0; v < 2; v = v + 1) begin
            $display("--- Vector %0d ---", v);
            // Feed bias (all zeros, 128 bits = 2 tile-cols × 64)
            @(posedge ap_clk);
            bias_tdata <= 128'd0; bias_tvalid <= 1;
            @(posedge ap_clk);
            bias_tvalid <= 0;

            // Feed 16 beats: rows 0-15 paired with cols 0-15.
            // All A = 1, all B = 2.  Expect C = 16 = 0x10.
            a_tdata <= 128'h01010101010101010101010101010101;
            b_tdata <= 128'h02020202020202020202020202020202;
            a_tvalid <= 1; b_tvalid <= 1;
            @(posedge ap_clk);
            for (i = 1; i < 16; i = i + 1) begin
                a_tdata <= 128'h01010101010101010101010101010101;
                b_tdata <= 128'h02020202020202020202020202020202;
                @(posedge ap_clk);
            end
            a_tvalid <= 0; b_tvalid <= 0;

            // Drain 16 output rows (2 tile-rows × 8).  Output is 128 bits (2 tile-cols × 64).
            for (i = 0; i < 16; i = i + 1) begin
                while (!c_tvalid) @(posedge ap_clk);
                if (c_tdata !== 128'h10101010101010101010101010101010) begin
                    $display("  FAIL row %0d: got 0x%032x expected 0x10101010101010101010101010101010",
                             i, c_tdata);
                    fail = fail + 1;
                end
                @(posedge ap_clk);
            end
        end

        if (fail == 0) $display("ALL_PASS"); else $display("FAILURES=%0d", fail);
        $finish;
    end
endmodule
