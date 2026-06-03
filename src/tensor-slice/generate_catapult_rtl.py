#!/usr/bin/env python3
"""
generate_verilog_grid.py
Catapult RTL generator — tensor-slice grid with clk/rst/en protocol.
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

from _generate_rtl_common import tail_mask_hex, vm, total_cycles as _total_cycles


# ── main generator ────────────────────────────────────────────────────────────

def generate_grid_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="direct"):
    """
    Parameters
    ----------
    m : int   Output rows  (A rows)
    k : int   Inner dimension
    n : int   Output cols  (B cols)
    feed_mode : str   "direct" (current) or "chained" (new location-aware)
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 128   # 8 INT16 values per column tile

    # ── Mode selection ───────────────────────────────────────────────────────
    is_chained = (feed_mode == "chained")
    chain_mode_bit = "1'b1" if is_chained else "1'b0"
    # Chained: FEED must cover all tiles' capture windows: max_loc_delay + 8
    max_loc_delay = (grid_rows - 1 + grid_cols - 1) * 8
    feed_len = max_loc_delay + 8 if is_chained else 8
    a_loc_val = lambda r: f"5'd{r}" if is_chained else "5'd0"
    b_loc_val = lambda c: f"5'd{c}" if is_chained else "5'd0"

    # ── Latency math ─────────────────────────────────────────────────────────
    TOTAL_CYCLES = _total_cycles(m, k, n, feed_mode=feed_mode)

    # ── Alignment ────────────────────────────────────────────────────────────
    if is_chained:
        # Per-column + per-tile-row delay for output sequencing.
        def align_delay(c):
            return (grid_cols - 1 - c) * 8
    else:
        def align_delay(c):
            return 0  # no per-column delay (direct feed)

    # ── Build mask literals ───────────────────────────────────────────────────
    k_mask_val    = tail_mask_hex(k, 0)
    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

    # ── Per-tile data wires ──────────────────────────────────────────────────
    # In chained mode, each tile captures during its local window:
    # local_feed = feed_idx - (r+c)*8 gives 0..7 for that tile.
    data_wires = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            a_base = r * 8
            b_base = c * 8
            a_hi   = (r + 1) * 64 - 1
            a_lo   = r * 64
            b_hi   = (c + 1) * 64 - 1
            b_lo   = c * 64
            if is_chained:
                loc_offset = (r + c) * 8
                data_wires.append(f"    wire signed [15:0] lf_{r}_{c} = $signed(feed_idx) - {loc_offset};")
                data_wires.append(f"    wire lv_{r}_{c} = in_valid_d && (lf_{r}_{c} >= 0) && (lf_{r}_{c} < 8);")
                data_wires.append(f"    wire [63:0] a_data_{r}_{c} = ({c} == 0 && lv_{r}_{c}) ? a_buf[{a_base} + lf_{r}_{c}][{a_hi}:{a_lo}] : 64'b0;")
                data_wires.append(f"    wire [63:0] b_data_{r}_{c} = ({r} == 0 && lv_{r}_{c}) ? b_buf[{b_base} + lf_{r}_{c}][{b_hi}:{b_lo}] : 64'b0;")
            else:
                data_wires.append(f"    wire [63:0] a_data_{r}_{c} = in_valid_d ? a_buf[{a_base} + feed_idx][{a_hi}:{a_lo}] : 64'b0;")
                data_wires.append(f"    wire [63:0] b_data_{r}_{c} = in_valid_d ? b_buf[{b_base} + feed_idx][{b_hi}:{b_lo}] : 64'b0;")

    # ── Slice instantiations ──────────────────────────────────────────────────
    inst_lines = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            inst_lines.append(f"""\
        tensor_slice_int8 slice_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_reset),
            .start_mat_mul(slice_start),
            .done_mat_mul(done_mat_mul[{r*grid_cols+c}]),
            .a_data(a_data_{r}_{c}),
            .b_data(preload_valid ? bias_cols[{c}*64 +: 64] : b_data_{r}_{c}),
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
            .preload(preload_valid), .no_rounding(1'b0),
            .final_mat_mul_size(8'd{k}),
            .a_loc({a_loc_val(r)}),  // {feed_mode} feed
            .b_loc({b_loc_val(c)}),
            .rowcol_chain_mode({chain_mode_bit})
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
            d     = align_delay(c) + r * 8  # per-column + tile-row staggering
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
                blk.append(f"        end else if (transaction_active) begin")
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

    # ── Output mux ───────────────────────────────────────────────────────────
    mux_lines = []
    for r in range(grid_rows):
        mux_lines.append(f"        if (row_avail_{r}) begin")
        for c in range(grid_cols):
            mux_lines.append(
                f"            row_mux[{c}*128 +: 128] = acd_{r}_{c};"
            )
        mux_lines.append(f"        end")

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
    input  wire [{a_width-1}:0]   a_rows,     // packed A column-tiles, one K-step
    input  wire [{b_width-1}:0]   b_cols,     // packed B row-tiles,    one K-step
    input  wire [{b_width-1}:0]   bias_cols,  // packed bias tiles, one per column-tile
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);

    // ---- Control (buffered row/col) ----
    localparam integer INPUT_BEATS     = {max(m,n)};
    localparam integer FEED_BEATS      = {feed_len};
    localparam integer TOTAL_CYCLES    = {TOTAL_CYCLES};
    localparam integer TOTAL_OUT_ROWS  = {total_output_rows};

    // FSM: IDLE -> COLLECT (buffer max(M,N) beats) -> FEED (replay 8 beats) -> DRAIN
    localparam [1:0] FSM_IDLE=2'd0, FSM_COLLECT=2'd1, FSM_FEED=2'd2, FSM_DRAIN=2'd3;

    reg [1:0] fsm_state;
    reg [15:0] cycle;
    reg [15:0] beat_cnt;
    reg [15:0] feed_idx;
    reg [{a_width-1}:0] a_buf [0:INPUT_BEATS-1];
    reg [{b_width-1}:0] b_buf [0:INPUT_BEATS-1];
    reg transaction_active;
    wire in_valid_d = (fsm_state == FSM_FEED);  // combinational: no NBA delay on launch
    reg [{c_width-1}:0] row_mux;
    reg [{c_width-1}:0] out_fifo [0:TOTAL_OUT_ROWS-1];
    reg [15:0] out_wr_ptr;
    reg [15:0] out_rd_ptr;
    reg [15:0] out_count;
    reg [15:0] out_row_count;

    wire slice_reset = rst;
    wire slice_start;
    wire output_take = en & (out_count != 0);

    always @(posedge clk) begin
        if (rst) begin
            fsm_state <= FSM_IDLE;
            cycle <= 16'd0;
            beat_cnt <= 16'd0;
            feed_idx <= 16'd0;
            transaction_active <= 1'b0;
        end else if (en) begin
            case (fsm_state)
                FSM_IDLE: begin
                    cycle <= 16'd0;
                    if (preload_valid) begin
                        // bias loaded directly via b_data mux; move to COLLECT
                        fsm_state <= FSM_COLLECT;
                        beat_cnt <= 16'd0;
                    end
                end

                FSM_COLLECT: begin
                    cycle <= cycle + 1;
                    if (in_valid) begin
                        a_buf[beat_cnt] <= a_rows;
                        b_buf[beat_cnt] <= b_cols;
                        beat_cnt <= beat_cnt + 1;
                    end
                    // Transition when enough beats collected (NBA, so +1 cycle)
                    if (beat_cnt >= INPUT_BEATS) begin
                        fsm_state <= FSM_FEED;
                        feed_idx <= 16'd0;
                        transaction_active <= 1'b1;
                    end
                end

                FSM_FEED: begin
                    cycle <= cycle + 1;
                    feed_idx <= feed_idx + 1;
                    if (feed_idx + 1 >= FEED_BEATS) begin
                        fsm_state <= FSM_DRAIN;
                    end
                end

                FSM_DRAIN: begin
                    cycle <= cycle + 1;
                    if (cycle + 1 >= TOTAL_CYCLES) begin
                        transaction_active <= 1'b0;
                        fsm_state <= FSM_IDLE;
                    end
                end
            endcase
        end
    end

    // Launch tensor_slice on the first FEED cycle (feed_idx=0).
    assign slice_start = (fsm_state == FSM_FEED) && (feed_idx == 16'd0);


    // ---- Wire declarations ----
    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;

{chr(10).join(chain_wires)}

    // ---- Per-tile data wires (row/col buffer indexing) ----
{chr(10).join(data_wires)}

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

    always @(*) begin
        row_mux = {c_width}'d0;
{chr(10).join(mux_lines)}
    end

    // ---- Output FIFO/hold logic ----
    integer fifo_i;
    always @(posedge clk) begin
        if (slice_reset) begin
            out_wr_ptr    <= 16'd0;
            out_rd_ptr    <= 16'd0;
            out_count     <= 16'd0;
            out_row_count <= 16'd0;
            for (fifo_i = 0; fifo_i < TOTAL_OUT_ROWS; fifo_i = fifo_i + 1) begin
                out_fifo[fifo_i] <= {c_width}'d0;
            end
        end else begin
            if (any_avail) begin
                out_fifo[out_wr_ptr] <= row_mux;
                out_wr_ptr <= out_wr_ptr + 1;
            end
            if (output_take) begin
                out_rd_ptr <= out_rd_ptr + 1;
                out_row_count <= out_row_count + 1;
            end
            case ({{any_avail, output_take}})
                2'b10: out_count <= out_count + 1;
                2'b01: out_count <= out_count - 1;
                default: out_count <= out_count;
            endcase
        end
    end

    always @(*) begin
        if (out_count != 0) begin
            c_row = out_fifo[out_rd_ptr];
            out_valid = 1'b1;
            out_last = (out_row_count + 1 == TOTAL_OUT_ROWS);
        end else begin
            c_row = {c_width}'d0;
            out_valid = 1'b0;
            out_last = 1'b0;
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
