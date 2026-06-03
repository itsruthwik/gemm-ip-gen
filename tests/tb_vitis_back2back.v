`timescale 1ns/1ps
// Minimal Vitis back-to-back test: 2 vectors, NO reset between them.
module tb_vitis_bb;
    reg ap_clk = 0; reg ap_rst = 1; reg ap_ce = 1;
    reg [63:0] a_tdata = 0, bias_tdata = 0, b_tdata = 0;
    reg a_tvalid = 0, bias_tvalid = 0, b_tvalid = 0;
    wire a_tready, bias_tready, b_tready;
    wire [63:0] c_tdata; wire c_tvalid; reg c_tready = 1;

    vitis_8x8x8 dut (
        .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_ce(ap_ce),
        .a_tdata(a_tdata), .a_tvalid(a_tvalid), .a_tready(a_tready),
        .bias_tdata(bias_tdata), .bias_tvalid(bias_tvalid), .bias_tready(bias_tready),
        .b_tdata(b_tdata), .b_tvalid(b_tvalid), .b_tready(b_tready),
        .c_tdata(c_tdata), .c_tvalid(c_tvalid), .c_tready(c_tready)
    );
    always #5 ap_clk = ~ap_clk;

    // All A = 1, all B = 2, bias = 0. Expect C[i][j] = 16 = 0x10
    integer v, i, fail;

    initial begin
        $display("=== Vitis back-to-back (2 vectors, no reset) ===");
        fail = 0;
        ap_rst <= 1; repeat(2) @(posedge ap_clk); ap_rst <= 0;

        for (v = 0; v < 2; v = v + 1) begin
            $display("--- Vector %0d ---", v);
            // Feed bias
            @(posedge ap_clk);
            bias_tdata <= 64'h0; bias_tvalid <= 1;
            @(posedge ap_clk);
            bias_tvalid <= 0;

            // Feed 8 beats: A rows (all 1s), B cols (all 2s)
            a_tdata <= 64'h0101010101010101; b_tdata <= 64'h0202020202020202;
            a_tvalid <= 1; b_tvalid <= 1;
            @(posedge ap_clk);
            for (i = 1; i < 8; i = i + 1) begin
                a_tdata <= 64'h0101010101010101; b_tdata <= 64'h0202020202020202;
                @(posedge ap_clk);
            end
            a_tvalid <= 0; b_tvalid <= 0;

            // Drain 8 output rows
            for (i = 0; i < 8; i = i + 1) begin
                while (!c_tvalid) @(posedge ap_clk);
                if (c_tdata !== 64'h1010101010101010) begin
                    $display("  FAIL row %0d: got 0x%016x expected 0x1010101010101010", i, c_tdata);
                    fail = fail + 1;
                end
                @(posedge ap_clk);
            end
        end

        if (fail == 0) $display("ALL_PASS"); else $display("FAILURES=%0d", fail);
        $finish;
    end
endmodule
