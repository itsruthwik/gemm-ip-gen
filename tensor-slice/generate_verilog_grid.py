#!/usr/bin/env python3
"""
generate_verilog_grid.py
Generates a Verilog wrapper stitching tensor_slice_int8 modules into an
M×N output grid with K inner dimension.

Key design:
  - All masks resolved in Python, emitted as literals (no generate arithmetic)
  - Slices unrolled explicitly for portability
  - One shared clk_cnt / slice_start drives the entire grid
  - Output mux: each tile-row (r) streams 8 result rows in order,
    staggered correctly against other tile-rows
  - out_last fires after the LAST output row of the LAST tile-row is captured
"""
import argparse
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_mask(total, loc):
    remain = total - loc * 8
    if remain <= 0:  return 0x00
    if remain >= 8:  return 0xFF
    return (0xFF >> (8 - remain)) & 0xFF


def vm(val):
    return f"8'h{val:02X}"


# ── main generator ────────────────────────────────────────────────────────────

def generate_grid_verilog(m, k, n, module_name="gemm_grid_wrapper"):
    """
    Parameters
    ----------
    m : int   Output rows  (A rows)
    k : int   Inner dimension
    n : int   Output cols  (B cols)
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 128   # 8 INT16 values per column tile

    # ── Latency math ─────────────────────────────────────────────────────────
    # tensor_slice_int8 internal:
    #   l_config  = (a_loc + b_loc)*8 + 7 + k + 3
    #   c_avail   asserted when clk_cnt == l_config - 1 (readout_ptr=0)
    #   done_mat_mul when clk_cnt == l_config + 8
    # Slice (r, c) → a_loc=r, b_loc=c
    P = 3

    def readout_start(r, c):   # clk_cnt value when c_avail first goes high
        return (r + c) * 8 + 7 + k + P

    def done_cycle(r, c):      # clk_cnt when done_mat_mul goes high
        return readout_start(r, c) + 9   # 8 rows drain → ptr reaches 7 then done

    # The LAST slice to complete is always the bottom-right corner
    last_done = done_cycle(grid_rows - 1, grid_cols - 1)
    # We need slice_start to stay high until clk_cnt reaches last_done
    TOTAL_CYCLES = last_done      # cycle counter runs 0 .. TOTAL_CYCLES-1

    # ── Alignment: align each column c to the last column ────────────────────
    # Slice (r, c) readout starts at readout_start(r,c).
    # Slice (r, GRID_COLS-1) readout starts at readout_start(r, GRID_COLS-1).
    # To align column c to GRID_COLS-1, we delay by:
    #   align_delay(c) = (GRID_COLS-1 - c) * 8
    def align_delay(c):
        return (grid_cols - 1 - c) * 8

    # ── Build mask literals ───────────────────────────────────────────────────
    k_mask_val    = compute_mask(k, 0)
    row_mask_vals = [compute_mask(m, r) for r in range(grid_rows)]
    col_mask_vals = [compute_mask(n, c) for c in range(grid_cols)]

    # ── Slice instantiations ──────────────────────────────────────────────────
    inst_lines = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            inst_lines.append(f"""\
        tensor_slice_int8 slice_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_reset),
            .start_mat_mul(slice_start),
            .done_mat_mul(done_mat_mul[{r*grid_cols+c}]),
            .a_data(in_valid_d ? a_rows_d[{r}*64 +: 64] : 64'b0),
            .b_data(in_valid_d ? b_cols_d[{c}*64 +: 64] : 64'b0),
            .a_data_in(a_chain_{r}_{c}),
            .b_data_in(b_chain_{r}_{c}),
            .a_data_out(a_chain_{r}_{c+1}),
            .b_data_out(b_chain_{r+1}_{c}),
            .c_data_out(c_data_{r}_{c}),
            .c_data_available(c_avail_{r}_{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows({vm(k_mask_val)}),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op(3'd0),
            .preload(1'b0), .no_rounding(1'b0),
            .final_mat_mul_size(8'd{k}),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")
    insts = "\n".join(inst_lines)

    # ── Wire declarations ─────────────────────────────────────────────────────
    chain_wires = []
    for r in range(grid_rows):
        for c in range(grid_cols + 1):
            chain_wires.append(f"    wire [63:0] a_chain_{r}_{c};")
    for r in range(grid_rows + 1):
        for c in range(grid_cols):
            chain_wires.append(f"    wire [63:0] b_chain_{r}_{c};")
    for r in range(grid_rows):
        for c in range(grid_cols):
            chain_wires.append(f"    wire [127:0] c_data_{r}_{c};")
            chain_wires.append(f"    wire         c_avail_{r}_{c};")

    # ── Boundary conditions ───────────────────────────────────────────────────
    boundary = []
    for r in range(grid_rows):
        boundary.append(f"    assign a_chain_{r}_0 = 64'b0;")
    for c in range(grid_cols):
        boundary.append(f"    assign b_chain_0_{c} = 64'b0;")

    # ── Alignment delay buffers ───────────────────────────────────────────────
    # For each (r, c) we delay so that all columns in the same row-tile fire
    # simultaneously. Then we simply OR per-row availability.
    align_decl   = []
    align_assign = []
    align_always = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            d     = align_delay(c)
            acd   = f"acd_{r}_{c}"    # aligned c_data
            aca   = f"aca_{r}_{c}"    # aligned c_avail
            if d == 0:
                align_decl.append(f"    wire [127:0] {acd} = c_data_{r}_{c};")
                align_decl.append(f"    wire         {aca} = c_avail_{r}_{c};")
            else:
                align_decl.append(f"    reg [127:0] {acd}_pipe [0:{d-1}];")
                align_decl.append(f"    reg         {aca}_pipe [0:{d-1}];")
                align_decl.append(f"    wire [127:0] {acd};")
                align_decl.append(f"    wire         {aca};")
                align_assign.append(f"    assign {acd} = {acd}_pipe[{d-1}];")
                align_assign.append(f"    assign {aca} = {aca}_pipe[{d-1}];")
                blk = []
                blk.append(f"    always @(posedge clk) begin")
                blk.append(f"        if (slice_reset) begin")
                for dd in range(d):
                    blk.append(f"            {aca}_pipe[{dd}] <= 1'b0;")
                blk.append(f"        end else if (en) begin")
                blk.append(f"            {acd}_pipe[0] <= c_data_{r}_{c};")
                blk.append(f"            {aca}_pipe[0] <= c_avail_{r}_{c};")
                for dd in range(1, d):
                    blk.append(f"            {acd}_pipe[{dd}] <= {acd}_pipe[{dd-1}];")
                    blk.append(f"            {aca}_pipe[{dd}] <= {aca}_pipe[{dd-1}];")
                blk.append(f"        end")
                blk.append(f"    end")
                align_always.extend(blk)

    # ── Per-row-tile availability ─────────────────────────────────────────────
    # After alignment, all columns in tile-row r fire at the same time.
    # Use column GRID_COLS-1 (the last, delay=0) to detect row availability.
    row_avail_decl = []
    for r in range(grid_rows):
        row_avail_decl.append(
            f"    wire row_avail_{r} = aca_{r}_{grid_cols-1};"
        )

    # ── Output mux inside always block ───────────────────────────────────────
    mux_lines = []
    for r in range(grid_rows):
        mux_lines.append(f"                if (row_avail_{r}) begin")
        for c in range(grid_cols):
            mux_lines.append(
                f"                    c_row[{c}*128 +: 128] <= acd_{r}_{c};"
            )
        mux_lines.append(f"                end")

    any_avail_expr = " | ".join(f"row_avail_{r}" for r in range(grid_rows))

    # out_last: fires one cycle after the very last row of the last tile-row
    # We count output rows and assert out_last when out_row_count == M_tiles*8
    total_output_rows = grid_rows * 8

    # ── Assemble Verilog ──────────────────────────────────────────────────────
    verilog = f"""\
// Auto-generated by generate_verilog_grid.py
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices
// TOTAL_CYCLES = {TOTAL_CYCLES}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,        // 1=reset, 0=running
    input  wire                   en,
    input  wire [{a_width-1}:0]  a_rows,     // packed A column-tiles, one K-step
    input  wire [{b_width-1}:0]  b_cols,     // packed B row-tiles,    one K-step
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]  c_row,
    output reg                    out_valid,
    output reg                    out_last
);

    // ---- Control ----
    localparam integer TOTAL_CYCLES    = {TOTAL_CYCLES};
    localparam integer TOTAL_OUT_ROWS  = {total_output_rows};

    reg [15:0] cycle;
    reg [15:0] out_row_count;
    reg [{a_width-1}:0] a_rows_d;
    reg [{b_width-1}:0] b_cols_d;
    reg in_valid_d;

    wire transaction_start = en & in_valid & (cycle == 0 || cycle >= TOTAL_CYCLES);
    wire slice_reset = rst | transaction_start;
    wire slice_start;

    // Buffer one input beat so the slice start pulse is visible before the
    // tensor_slice samples beat zero. A new in_valid after TOTAL_CYCLES starts
    // a fresh transaction without relying on top-level reset.
    always @(posedge clk) begin
        if (rst) begin
            cycle <= 16'd0;
            a_rows_d <= {a_width}'d0;
            b_cols_d <= {b_width}'d0;
            in_valid_d <= 1'b0;
        end else if (transaction_start) begin
            a_rows_d <= a_rows;
            b_cols_d <= b_cols;
            in_valid_d <= in_valid;
            cycle <= 16'd1;
        end else if (en && cycle < TOTAL_CYCLES) begin
            a_rows_d <= a_rows;
            b_cols_d <= b_cols;
            in_valid_d <= in_valid;
            cycle <= cycle + 1;
        end
    end

    // Launch one cycle after the first input beat is captured.
    assign slice_start = en & (cycle == 16'd1);

    // ---- Wire declarations ----
    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;

{chr(10).join(chain_wires)}

    // ---- Systolic chain boundaries ----
{chr(10).join(boundary)}

    // ---- Slice instantiations ----
{insts}

    // ---- Alignment buffers ----
{chr(10).join(align_decl)}
{chr(10).join(align_assign)}
{chr(10).join(align_always)}

    // ---- Per-row-tile availability ----
{chr(10).join(row_avail_decl)}

    wire any_avail = {any_avail_expr};

    // ---- Output logic ----
    always @(posedge clk) begin
        if (slice_reset) begin
            c_row         <= {c_width}'d0;
            out_valid     <= 1'b0;
            out_last      <= 1'b0;
            out_row_count <= 16'd0;
        end else if (en) begin
            out_valid <= any_avail;
            if (any_avail) begin
                out_row_count <= out_row_count + 1;
                out_last      <= (out_row_count + 1 == TOTAL_OUT_ROWS);
{chr(10).join(mux_lines)}
            end else begin
                out_last <= 1'b0;
            end
        end
    end

endmodule
"""
    return verilog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Verilog grid wrapper for tensor_slice_int8 modules"
    )
    parser.add_argument("--m",      type=int, required=True)
    parser.add_argument("--k",      type=int, required=True)
    parser.add_argument("--n",      type=int, required=True)
    parser.add_argument("--name",   type=str, default="gemm_grid_wrapper")
    parser.add_argument("--output", type=str, default="gemm_grid.v")
    args = parser.parse_args()

    content = generate_grid_verilog(args.m, args.k, args.n, args.name)
    Path(args.output).write_text(content)
    print(f"Generated {args.output}  (M={args.m}, K={args.k}, N={args.n})")
