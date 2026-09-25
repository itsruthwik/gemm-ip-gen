// cmvu_regbank — the core bypassable register-bank primitive of the CMVU pipeline.
//
// Every pipeline stage (input, multiply, adder-tree level, reduce, output, ...) is built from
// one of these banks. Whether a bank is a real clocked register or a combinational pass-through
// is a STATIC, compile-time choice (`parameter PRESENT`), never a runtime signal — so the
// resulting pipeline's latency is simply the COUNT of PRESENT banks on a given input->output
// path. PRESENT=1 => a real clocked register (clock-enable `ena`, async reset `rst`); PRESENT=0
// => combinational pass-through (the bank is bypassed, contributing 0 latency).
//
// `ena`/`rst` are the genuinely-dynamic pins (they stall/clear an *already-present* register —
// they never change pipeline depth). Depth is the parameter's job.
module cmvu_regbank #(
    parameter int unsigned W       = 1,
    parameter bit          PRESENT = 1'b1
) (
    input  wire         clk,
    input  wire         rst,   // async reset
    input  wire         ena,   // clock enable
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (PRESENT) begin : g_reg
            reg [W-1:0] r;
            always @(posedge clk or posedge rst) begin
                if (rst)      r <= '0;
                else if (ena) r <= d;
            end
            assign q = r;
        end else begin : g_bypass
            assign q = d;       // bank bypassed -> combinational
        end
    endgenerate
endmodule
