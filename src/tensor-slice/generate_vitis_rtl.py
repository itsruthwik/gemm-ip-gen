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

def generate_vitis_rtl(m, k, n, module_name="gemm_vitis", feed_mode="direct"):
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
    max_loc_delay = (grid_rows - 1 + grid_cols - 1) * 8
    feed_len = max_loc_delay + 8 if is_chained else 8

    # ── Latency math ─────────────────────────────────────────────────────────
    TOTAL_CYCLES = _total_cycles(m, k, n, feed_mode=feed_mode)

    # ── Alignment ────────────────────────────────────────────────────────────
    if is_chained:
        def align_delay(r, c):
            return (grid_cols - 1 - c) * 8 + r * grid_cols * 8
    else:
        def align_delay(r, c):
            return r * 8

    # ── Build mask literals ───────────────────────────────────────────────────
    k_mask_val    = tail_mask_hex(k, 0)
    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

    # ── Per-tile data wires ──────────────────────────────────────────────────
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
    a_loc_val = lambda r: f"5'd{r}" if is_chained else "5'd0"
    b_loc_val = lambda c: f"5'd{c}" if is_chained else "5'd0"
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
            .a_loc({a_loc_val(r)}),
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
            d     = align_delay(r, c)
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
    c_stream_width = grid_cols * 64  # int8 packed output

    verilog = f"""\
// Auto-generated by generate_vitis_rtl.py (Catapult core + AXI-Stream wrapper)
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices
// TOTAL_CYCLES = {TOTAL_CYCLES}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   ap_clk,
    input  wire                   ap_rst,
    input  wire                   ap_ce,

    // AXI-Stream inputs
    input  wire [{a_width-1}:0]   a_tdata,
    input  wire                   a_tvalid,
    output wire                   a_tready,

    input  wire [{b_width-1}:0]   bias_tdata,
    input  wire                   bias_tvalid,
    output wire                   bias_tready,

    input  wire [{b_width-1}:0]   b_tdata,
    input  wire                   b_tvalid,
    output wire                   b_tready,

    // AXI-Stream output
    output wire [{c_stream_width-1}:0] c_tdata,
    output wire                    c_tvalid,
    input  wire                    c_tready
);

    // ═══════════════════════════════════════════════════════════════════════════
    //  Protocol FSM — drives internal Catapult signals from AXI-Stream
    // ═══════════════════════════════════════════════════════════════════════════
    // Map Vitis port names to internal Catapult signal names
    wire clk = ap_clk;
    wire rst = ap_rst;
    localparam FSM_IDLE=3'd0, FSM_PRELOAD=3'd1;
    localparam FSM_COLLECT=3'd2, FSM_FEED=3'd3, FSM_DRAIN=3'd4;

    reg [2:0]  fsm_state;
    reg [15:0] fsm_cnt;
    reg [15:0] feed_idx;

    // Beat buffers: collect max(M,N) beats during COLLECT, then feed consecutively
    reg [{a_width-1}:0] a_buf [0:{max(m,n)-1}];
    reg [{b_width-1}:0] b_buf [0:{max(m,n)-1}];

    // Independent per-channel write pointers.  Each channel captures beats as
    // they arrive (tolerating FIFO read-latency skew from asymmetric widths).
    // Beats are paired by index when both channels have collected K entries.
    reg [15:0] a_wr_ptr;
    reg [15:0] b_wr_ptr;

    // Internal control signals
    reg        en;
    reg [{b_width-1}:0] bias_cols_int;
    reg        preload_valid;
    reg        in_valid;

    // AXI-Stream ready signals.
    // Vitis may select FIFO implementations with different read latencies for
    // asymmetric widths (e.g. 128-bit vs 64-bit).  Each channel independently
    // accepts beats up to K — no cross-channel dependency, no deadlock.
    assign bias_tready = (fsm_state == FSM_IDLE);
    assign a_tready    = ((fsm_state == FSM_PRELOAD || fsm_state == FSM_COLLECT) && (a_wr_ptr < {max(m,n)}));
    assign b_tready    = ((fsm_state == FSM_PRELOAD || fsm_state == FSM_COLLECT) && (b_wr_ptr < {max(m,n)}));

    // Preload → Collect → Feed pipeline
    always @(posedge clk) begin
        if (rst) begin
            fsm_state <= FSM_IDLE;
            fsm_cnt   <= 0;
            feed_idx  <= 0;
            a_wr_ptr  <= 0;
            b_wr_ptr  <= 0;
            en        <= 0;
            bias_cols_int <= 0;
            preload_valid <= 0;
            in_valid      <= 0;
        end else if (ap_ce) begin
            case (fsm_state)
                FSM_IDLE: begin
                    en <= 0; preload_valid <= 0; in_valid <= 0;
                    // Clear wr_ptrs in IDLE so a_tready/b_tready see 0
                    // in PRELOAD (combinational, reads old reg values).
                    a_wr_ptr  <= 0;
                    b_wr_ptr  <= 0;
                    if (bias_tvalid && bias_tready) begin
                        bias_cols_int <= bias_tdata;
                        fsm_state <= FSM_PRELOAD;
                        fsm_cnt   <= 0;
                        en        <= 1;
                        preload_valid <= 1;
                    end
                end

                FSM_PRELOAD: begin
                    // Bias was preloaded on the IDLE→PRELOAD transition.
                    // Accept first A/B beat here to avoid 1-cycle handshake gap.
                    // wr_ptrs already 0 from IDLE; capture bumps them to 1.
                    fsm_state <= FSM_COLLECT;
                    preload_valid <= 0;
                    in_valid  <= 0;
                    if (a_tvalid && a_tready) begin
                        a_buf[0] <= a_tdata;
                        a_wr_ptr <= 1;
                    end
                    if (b_tvalid && b_tready) begin
                        b_buf[0] <= b_tdata;
                        b_wr_ptr <= 1;
                    end
                end

                FSM_COLLECT: begin
                    // Each channel captured independently — tolerates FIFO skew.
                    if (a_tvalid && a_tready) begin
                        a_buf[a_wr_ptr] <= a_tdata;
                        a_wr_ptr <= a_wr_ptr + 1;
                    end
                    if (b_tvalid && b_tready) begin
                        b_buf[b_wr_ptr] <= b_tdata;
                        b_wr_ptr <= b_wr_ptr + 1;
                    end
                    // Transition when both channels have collected K beats
                    if (a_wr_ptr >= {max(m,n)} && b_wr_ptr >= {max(m,n)}) begin
                        fsm_state <= FSM_FEED;
                        fsm_cnt   <= 0;
                        feed_idx  <= 0;
                        in_valid  <= 1;
                    end
                end

                FSM_FEED: begin
                    // Feed buffered beats on consecutive cycles.
                    // No dependency on tvalid — data is from local buffers.
                    feed_idx <= feed_idx + 1;
                    if (feed_idx + 1 >= FEED_BEATS) begin
                        fsm_state <= FSM_DRAIN;
                        fsm_cnt   <= 0;
                        in_valid  <= 0;
                    end
                end

                FSM_DRAIN: begin
                    if (output_take) begin
                        fsm_cnt  <= fsm_cnt + 1;
                        if (out_last) begin
                            fsm_state <= FSM_IDLE;
                            en        <= 0;
                        end
                    end
                end
            endcase
        end
    end

    // ── Map bias to internal name ─────────────────────────────────────────────
    wire [{b_width-1}:0] bias_cols = bias_cols_int;

    // ── Output packing: extract int8 bytes from each 128-bit tile ────────────
    // Catapult c_row has gc*128 bits (int16 lanes per tile).
    // Vitis c_tdata needs gc*64 bits (int8 packed per tile).
    // Lower 64 bits of each 128-bit tile contain the int8 values.
    function [{c_stream_width-1}:0] pack_c_row;
        input [{c_width-1}:0] row_in;
        integer t;
    begin
        for (t = 0; t < {grid_cols}; t = t + 1) begin
            pack_c_row[t * 64 +: 64] = row_in[t * 128 +: 64];
        end
    end
    endfunction

    // ═══════════════════════════════════════════════════════════════════════════
    //  Direct-feed grid core (row/col, matched to Catapult generator)
    // ═══════════════════════════════════════════════════════════════════════════

    // ---- Control ----
    localparam integer FEED_BEATS      = {feed_len};
    localparam integer TOTAL_CYCLES    = {TOTAL_CYCLES};
    localparam integer TOTAL_OUT_ROWS  = {total_output_rows};

    reg [15:0] cycle;
    wire transaction_active = (fsm_state == FSM_COLLECT) || (fsm_state == FSM_FEED) || (fsm_state == FSM_DRAIN);
    wire in_valid_d = (fsm_state == FSM_FEED);
    reg [{c_width-1}:0] row_mux;
    reg [{c_width-1}:0] out_fifo [0:TOTAL_OUT_ROWS-1];
    reg [15:0] out_wr_ptr;
    reg [15:0] out_rd_ptr;
    reg [15:0] out_count;
    reg [15:0] out_row_count;

    reg  [{c_width-1}:0] c_row;
    reg                   out_valid;
    reg                   out_last;

    wire slice_reset = rst;
    wire clear_fifo = (fsm_state == FSM_PRELOAD);
    wire slice_start;
    wire output_take = en & (fsm_state == FSM_DRAIN) & (out_count != 0) & c_tready;
    assign c_tvalid = output_take;
    assign c_tdata = output_take ? pack_c_row(c_row) : {c_stream_width}'d0;

    // Track cycle counter (free-running during transaction)
    always @(posedge clk) begin
        if (rst) begin
            cycle <= 16'd0;
        end else if (ap_ce) begin
            if (fsm_state != FSM_IDLE) begin
                cycle <= cycle + 1;
            end else begin
                cycle <= 16'd0;
            end
        end
    end

    // Launch tensor_slice on first FEED cycle (feed_idx=0)
    assign slice_start = (fsm_state == FSM_FEED) && (feed_idx == 16'd0);

    // ---- Wire declarations ----
    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;

{chr(10).join(chain_wires)}

    // ---- Systolic chain boundaries ----
{chr(10).join(boundary)}

    // ---- Per-tile data wires (row/col buffer indexing) ----
{chr(10).join(data_wires)}

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
        if (slice_reset || clear_fifo) begin
            out_wr_ptr    <= 16'd0;
            out_rd_ptr    <= 16'd0;
            out_count     <= 16'd0;
            out_row_count <= 16'd0;
            for (fifo_i = 0; fifo_i < TOTAL_OUT_ROWS; fifo_i = fifo_i + 1) begin
                out_fifo[fifo_i] <= {c_width}'d0;
            end
        end else if (ap_ce) begin
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
