#!/usr/bin/env python3
"""
generate_vitis_rtl.py
Vitis RTL generator — tensor-slice grid with AXI-Stream protocol + Catapult core.

Key design:
  - AXI-Stream protocol FSM → drives Catapult-style core
  - Single continuous operation: collect K beats, feed tiles, drain
  - No K-step chunking
"""
import argparse
from pathlib import Path

from _generate_rtl_common import tail_mask_hex, vm, total_cycles as _total_cycles


def generate_vitis_sim_rtl(m, k, n, module_name="gemm_vitis"):
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_stream_width = grid_cols * 64
    input_beats = max(m, n)
    k_chunks = (k + 7) // 8
    total_input_beats = k_chunks * input_beats
    total_output_rows = m
    latency = max(0, k + n - k_chunks * input_beats)
    behav_name = f"{module_name}_behav_grid"

    return f"""\
// Auto-generated simulation model by generate_vitis_rtl.py
// Chunked behavioral MxKxN GEMM. Not intended for synthesis.
// Dimensions: M={m}, K={k}, N={n}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   ap_clk,
    input  wire                   ap_rst,
    input  wire                   ap_ce,

    input  wire [{a_width-1}:0]   a_tdata,
    input  wire                   a_tvalid,
    output wire                   a_tready,

    input  wire [{b_width-1}:0]   bias_tdata,
    input  wire                   bias_tvalid,
    output wire                   bias_tready,

    input  wire [{b_width-1}:0]   b_tdata,
    input  wire                   b_tvalid,
    output wire                   b_tready,

    output wire [{c_stream_width-1}:0] c_tdata,
    output wire                   c_tvalid,
    input  wire                   c_tready
);

    wire [{c_stream_width-1}:0] behav_c_tdata;
    wire                       behav_c_tvalid;

    {behav_name} grid (
        .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_ce(ap_ce),
        .a_tdata(a_tdata), .a_tvalid(a_tvalid), .a_tready(a_tready),
        .bias_tdata(bias_tdata), .bias_tvalid(bias_tvalid), .bias_tready(bias_tready),
        .b_tdata(b_tdata), .b_tvalid(b_tvalid), .b_tready(b_tready),
        .c_tdata(behav_c_tdata), .c_tvalid(behav_c_tvalid), .c_tready(c_tready)
    );

    assign c_tdata = behav_c_tdata;
    assign c_tvalid = behav_c_tvalid;

endmodule

module {behav_name}(
    input  wire                   ap_clk,
    input  wire                   ap_rst,
    input  wire                   ap_ce,

    input  wire [{a_width-1}:0]   a_tdata,
    input  wire                   a_tvalid,
    output wire                   a_tready,

    input  wire [{b_width-1}:0]   bias_tdata,
    input  wire                   bias_tvalid,
    output wire                   bias_tready,

    input  wire [{b_width-1}:0]   b_tdata,
    input  wire                   b_tvalid,
    output wire                   b_tready,

    output reg  [{c_stream_width-1}:0] c_tdata,
    output reg                    c_tvalid,
    input  wire                   c_tready
);

    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer TOTAL_INPUT_BEATS = {total_input_beats};
    localparam integer TOTAL_ROWS = {total_output_rows};
    localparam integer LATENCY = {latency};

    localparam [2:0] C_IDLE=3'd0, C_COLLECT=3'd1, C_WAIT=3'd2, C_DONE=3'd3;
    localparam [1:0] O_IDLE=2'd0, O_OUTPUT=2'd1;

    // ── Collection pipeline registers ──────────────────────────────────────
    reg [2:0]  coll_state;
    reg [15:0] beat_count;
    reg [15:0] wait_count;

    // ── Output pipeline registers ──────────────────────────────────────────
    reg [1:0]  out_state;
    reg [15:0] out_row_idx;

    // ── Double-buffered storage ────────────────────────────────────────────
    reg signed [7:0] amat_0 [0:{m-1}][0:{k-1}];
    reg signed [7:0] amat_1 [0:{m-1}][0:{k-1}];
    reg signed [7:0] bmat_0 [0:{k-1}][0:{n-1}];
    reg signed [7:0] bmat_1 [0:{k-1}][0:{n-1}];
    reg signed [7:0] bias_0 [0:{n-1}];
    reg signed [7:0] bias_1 [0:{n-1}];
    reg signed [31:0] cmat_0 [0:{m-1}][0:{n-1}];
    reg signed [31:0] cmat_1 [0:{m-1}][0:{n-1}];

    // ── Buffer control ─────────────────────────────────────────────────────
    reg        coll_buf;
    reg        out_buf;
    reg        buf_ready_0;
    reg        buf_ready_1;
    wire       buf_ready_cur = (out_buf == 0) ? buf_ready_0 : buf_ready_1;
    reg        next_use;

    // ── Pending start (for back-to-back pipelining) ────────────────────────
    reg        pending_start;
    reg        pending_buf;

    // ── BEH timing ─────────────────────────────────────────────────────────
    reg [31:0] beh_cyc;
    reg        beh_first_input;
    reg [31:0] prev_beh_start;
    reg [31:0] beh_ii_val;

    wire coll_ready = (coll_state == C_COLLECT) && (beat_count < TOTAL_INPUT_BEATS) && ap_ce;
    wire input_fire = coll_ready && a_tvalid && b_tvalid;

    assign bias_tready = ((coll_buf ^ 1'b1) != out_buf || !buf_ready_cur) && ap_ce;
    assign a_tready = coll_ready && b_tvalid;
    assign b_tready = coll_ready && a_tvalid;

    integer i;
    integer j;
    integer kk;
    integer tile;
    integer lane;
    integer actual_row;
    integer actual_col;
    integer chunk_idx;
    integer beat_in_chunk;
    reg signed [31:0] sum;
    reg [7:0] sat;

    function [7:0] sat_int8;
        input signed [31:0] x;
        begin
            if (x > 32'sd127) sat_int8 = 8'h7f;
            else if (x < -32'sd128) sat_int8 = 8'h80;
            else sat_int8 = x[7:0];
        end
    endfunction

    // ═══════════════════════════════════════════════════════════════════════
    // Collection pipeline: C_IDLE → C_COLLECT → C_WAIT → C_DONE → C_IDLE
    // With pending-start mechanism for back-to-back pipelining.
    // ═══════════════════════════════════════════════════════════════════════
    always @(posedge ap_clk) begin
        if (ap_rst) begin
            coll_state <= C_IDLE;
            beat_count <= 16'd0;
            wait_count <= 16'd0;
            coll_buf <= 1'b0;
            buf_ready_0 <= 1'b0;
            buf_ready_1 <= 1'b0;
            pending_start <= 1'b0;
            pending_buf <= 1'b0;
            beh_cyc <= 32'd0;
            beh_first_input <= 1'b1;
            prev_beh_start <= 32'd0;
            beh_ii_val <= 32'd0;
        end else if (ap_ce) begin
            beh_cyc <= beh_cyc + 1;

            // Bias acceptance: use the OTHER buffer unless output is draining from it.
            // If coll_state is not C_IDLE, defer the start via pending flags.
            if (bias_tvalid && bias_tready) begin
                next_use = coll_buf ^ 1'b1;
                if (next_use == 0) begin
                    for (j = 0; j < {n}; j = j + 1)
                        bias_0[j] <= bias_tdata[(j / 8) * 64 + (j % 8) * 8 +: 8];
                end else begin
                    for (j = 0; j < {n}; j = j + 1)
                        bias_1[j] <= bias_tdata[(j / 8) * 64 + (j % 8) * 8 +: 8];
                end
                if (coll_state == C_IDLE) begin
                    coll_buf <= next_use;
                    beh_first_input <= 1'b1;
                    beat_count <= 16'd0;
                    wait_count <= 16'd0;
                    coll_state <= C_COLLECT;
                end else begin
                    pending_start <= 1'b1;
                    pending_buf <= next_use;
                end
            end

            case (coll_state)
                C_IDLE: begin
                    // Service pending start (deferred from C_WAIT interrupt)
                    if (pending_start) begin
                        coll_buf <= pending_buf;
                        beh_first_input <= 1'b1;
                        beat_count <= 16'd0;
                        wait_count <= 16'd0;
                        pending_start <= 1'b0;
                        coll_state <= C_COLLECT;
                    end
                end

                C_COLLECT: begin
                    if (input_fire) begin
                        if (beh_first_input) begin
                            if (prev_beh_start == 0)
                                beh_ii_val <= 32'd0;
                            else
                                beh_ii_val <= beh_cyc - prev_beh_start;
                            prev_beh_start <= beh_cyc;
                            $display("BEH_START beh_cyc=%0d", beh_cyc);
                            beh_first_input <= 1'b0;
                        end
                        chunk_idx = beat_count / INPUT_BEATS;
                        beat_in_chunk = beat_count % INPUT_BEATS;
                        if (beat_in_chunk < {m}) begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                kk = chunk_idx * 8 + lane;
                                if (kk < {k}) begin
                                    if (coll_buf == 0)
                                        amat_0[beat_in_chunk][kk] = a_tdata[(beat_in_chunk / 8) * 64 + lane * 8 +: 8];
                                    else
                                        amat_1[beat_in_chunk][kk] = a_tdata[(beat_in_chunk / 8) * 64 + lane * 8 +: 8];
                                end
                            end
                        end
                        if (beat_in_chunk < {n}) begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                kk = chunk_idx * 8 + lane;
                                if (kk < {k}) begin
                                    if (coll_buf == 0)
                                        bmat_0[kk][beat_in_chunk] = b_tdata[(beat_in_chunk / 8) * 64 + lane * 8 +: 8];
                                    else
                                        bmat_1[kk][beat_in_chunk] = b_tdata[(beat_in_chunk / 8) * 64 + lane * 8 +: 8];
                                end
                            end
                        end
                        beat_count <= beat_count + 1;
                    end
                    if (beat_count + 1 >= TOTAL_INPUT_BEATS) begin
                        if (coll_buf == 0) begin
                            for (i = 0; i < {m}; i = i + 1) begin
                                for (j = 0; j < {n}; j = j + 1) begin
                                    sum = bias_0[j];
                                    for (kk = 0; kk < {k}; kk = kk + 1)
                                        sum = sum + (amat_0[i][kk] * bmat_0[kk][j]);
                                    cmat_0[i][j] <= sum;
                                end
                            end
                        end else begin
                            for (i = 0; i < {m}; i = i + 1) begin
                                for (j = 0; j < {n}; j = j + 1) begin
                                    sum = bias_1[j];
                                    for (kk = 0; kk < {k}; kk = kk + 1)
                                        sum = sum + (amat_1[i][kk] * bmat_1[kk][j]);
                                    cmat_1[i][j] <= sum;
                                end
                            end
                        end
                        if (LATENCY == 0) begin
                            if (coll_buf == 0) buf_ready_0 <= 1'b1; else buf_ready_1 <= 1'b1;
                            beat_count <= 16'd0;
                            coll_state <= C_IDLE;
                        end else begin
                            wait_count <= 16'd0;
                            coll_state <= C_WAIT;
                        end
                    end
                end

                C_WAIT: begin
                    wait_count <= wait_count + 1;
                    if (wait_count + 1 >= LATENCY) begin
                        if (coll_buf == 0) buf_ready_0 <= 1'b1; else buf_ready_1 <= 1'b1;
                        beat_count <= 16'd0;
                        coll_state <= C_IDLE;
                        // Service pending start immediately (same cycle)
                        if (pending_start) begin
                            coll_buf <= pending_buf;
                            beh_first_input <= 1'b1;
                            beat_count <= 16'd0;
                            wait_count <= 16'd0;
                            pending_start <= 1'b0;
                            coll_state <= C_COLLECT;
                        end
                    end
                end

                default: coll_state <= C_IDLE;
            endcase
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // Output pipeline: O_IDLE → O_OUTPUT → O_IDLE
    // Checks both buffers and picks whichever is ready.
    // ═══════════════════════════════════════════════════════════════════════
    always @(posedge ap_clk) begin
        if (ap_rst) begin
            out_state <= O_IDLE;
            out_row_idx <= 16'd0;
            c_tdata <= {c_stream_width}'d0;
            c_tvalid <= 1'b0;
            out_buf <= 1'b1;
        end else if (ap_ce) begin
            case (out_state)
                O_IDLE: begin
                    c_tdata <= {c_stream_width}'d0;
                    c_tvalid <= 1'b0;
                    out_row_idx <= 16'd0;
                    if (buf_ready_0) begin
                        out_buf <= 1'b0;
                        out_state <= O_OUTPUT;
                    end else if (buf_ready_1) begin
                        out_buf <= 1'b1;
                        out_state <= O_OUTPUT;
                    end
                end

                O_OUTPUT: begin
                    if (!c_tvalid || c_tready) begin
                        c_tdata <= {c_stream_width}'d0;
                        actual_row = out_row_idx;
                        for (tile = 0; tile < {grid_cols}; tile = tile + 1) begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                actual_col = tile * 8 + lane;
                                if (actual_row < {m} && actual_col < {n}) begin
                                    if (out_buf == 0)
                                        sat = sat_int8(cmat_0[actual_row][actual_col]);
                                    else
                                        sat = sat_int8(cmat_1[actual_row][actual_col]);
                                end else begin
                                    sat = 8'd0;
                                end
                                c_tdata[tile * 64 + lane * 8 +: 8] <= sat;
                            end
                        end
                        c_tvalid <= 1'b1;
                        out_row_idx <= out_row_idx + 1;
                        if (out_row_idx + 1 >= TOTAL_ROWS) begin
                            $display("BEH_II=%0d", beh_ii_val);
                            $display("BEH_DONE beh_cyc=%0d", beh_cyc);
                            if (out_buf == 0) buf_ready_0 <= 1'b0; else buf_ready_1 <= 1'b0;
                            out_state <= O_IDLE;
                        end
                    end
                end

                default: out_state <= O_IDLE;
            endcase
        end
    end

endmodule
"""


def generate_vitis_synth_rtl(m, k, n, module_name="gemm_vitis", feed_mode="chained"):
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 128
    c_stream_width = grid_cols * 64
    input_beats = max(m, n)
    total_output_rows = grid_rows * 8
    k_chunks = (k + 7) // 8
    last_k_size = k - (k_chunks - 1) * 8
    last_k_mask = tail_mask_hex(k, k_chunks - 1)

    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

    def align_delay(r, c):
        return (grid_cols - 1 - c) * 8 + r * grid_cols * 8

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

    boundary = []
    for r in range(grid_rows):
        boundary.append(f"    assign a_chain_{r}_0 = 64'b0;")
    for c in range(grid_cols):
        boundary.append(f"    assign b_chain_0_{c} = 64'b0;")

    data_wires = []
    for r in range(grid_rows):
        a_hi = (r + 1) * 64 - 1
        a_lo = r * 64
        for c in range(grid_cols):
            b_hi = (c + 1) * 64 - 1
            b_lo = c * 64
            data_wires.append(
                f"    wire [63:0] a_data_{r}_{c} = (in_beat_active && ({c} == 0)) ? a_tdata[{a_hi}:{a_lo}] : 64'b0;"
            )
            data_wires.append(
                f"    wire [63:0] b_data_{r}_{c} = (in_beat_active && ({r} == 0)) ? b_tdata[{b_hi}:{b_lo}] : 64'b0;"
            )

    inst_lines = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            inst_lines.append(f"""\
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_pe_reset),
            .start_mat_mul(slice_start),
            .done_mat_mul(done_mat_mul[{r*grid_cols+c}]),
            .a_data(a_data_{r}_{c}),
            .b_data(preload_d ? bias_cols[{c}*64 +: 64] : b_data_{r}_{c}),
            .a_data_in(a_chain_{r}_{c}),
            .b_data_in(b_chain_{r}_{c}),
            .a_data_out(a_chain_{r}_{c+1}),
            .b_data_out(b_chain_{r+1}_{c}),
            .c_data_out(c_data_{r}_{c}),
            .c_data_available(c_avail_{r}_{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows(current_k_mask),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op(3'd0),
            .preload(preload_d), .no_rounding(1'b0),
            .final_mat_mul_size(current_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    align_decl = []
    align_assign = []
    align_always = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            d = align_delay(r, c)
            acd = f"acd_{r}_{c}"
            aca = f"aca_{r}_{c}"
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
                blk = ["    always @(posedge clk) begin", "        if (slice_reset) begin"]
                for dd in range(d):
                    blk.append(f"            {acd}_pipe[{dd}] <= 128'd0;")
                    blk.append(f"            {aca}_pipe[{dd}] <= 1'b0;")
                blk.append("        end else if (transaction_active) begin")
                blk.append(f"            {acd}_pipe[0] <= c_data_{r}_{c};")
                blk.append(f"            {aca}_pipe[0] <= c_avail_{r}_{c};")
                for dd in range(1, d):
                    blk.append(f"            {acd}_pipe[{dd}] <= {acd}_pipe[{dd-1}];")
                    blk.append(f"            {aca}_pipe[{dd}] <= {aca}_pipe[{dd-1}];")
                blk.append("        end")
                blk.append("    end")
                align_always.extend(blk)

    row_avail_decl = [f"    wire row_avail_{r} = aca_{r}_{grid_cols-1};" for r in range(grid_rows)]
    any_avail_expr = " | ".join(f"row_avail_{r}" for r in range(grid_rows))
    mux_lines = []
    for r in range(grid_rows):
        mux_lines.append(f"        if (row_avail_{r}) begin")
        for c in range(grid_cols):
            mux_lines.append(f"            row_mux[{c}*128 +: 128] = acd_{r}_{c};")
        mux_lines.append("        end")

    return f"""\
// Auto-generated by generate_vitis_rtl.py
// Chunked structural tensor-slice synth wrapper with AXI-Stream ports
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices
`timescale 1ns/1ps

module {module_name}(
    input  wire                   ap_clk,
    input  wire                   ap_rst,
    input  wire                   ap_ce,

    input  wire [{a_width-1}:0]   a_tdata,
    input  wire                   a_tvalid,
    output wire                   a_tready,

    input  wire [{b_width-1}:0]   bias_tdata,
    input  wire                   bias_tvalid,
    output wire                   bias_tready,

    input  wire [{b_width-1}:0]   b_tdata,
    input  wire                   b_tvalid,
    output wire                   b_tready,

    output wire [{c_stream_width-1}:0] c_tdata,
    output wire                   c_tvalid,
    input  wire                   c_tready
);

    wire clk = ap_clk;
    wire rst = ap_rst;

    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer LAST_K_SIZE = {last_k_size};
    localparam integer TOTAL_OUT_ROWS = {total_output_rows};
    localparam [1:0] S_IDLE=2'd0, S_PRELOAD=2'd1, S_RUN=2'd2, S_WAIT=2'd3;

    reg [1:0] state;
    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] out_row_count;
    reg [{b_width-1}:0] bias_cols;
    reg preload_d;
    reg transaction_active;
    reg [{c_width-1}:0] row_mux;
    reg [{c_stream_width-1}:0] skid_data;
    reg skid_valid;
    reg skid_last;

    wire input_ready = (state == S_RUN) && (beat_count < INPUT_BEATS) && !skid_valid;
    wire input_fire = input_ready && a_tvalid && b_tvalid;
    wire in_beat_active = input_fire;
    wire slice_start = input_fire && (beat_count == 16'd0);
    wire slice_pe_reset = slice_start && (chunk_idx == 16'd0);
    wire final_chunk = (chunk_idx == K_CHUNKS - 1);
    wire [7:0] current_k_size = final_chunk ? 8'd{last_k_size} : 8'd8;
    wire [7:0] current_k_mask = final_chunk ? {vm(last_k_mask)} : 8'hFF;
    wire output_fire = skid_valid && c_tready;

    assign bias_tready = (state == S_IDLE) && ap_ce;
    assign a_tready = input_ready && b_tvalid && ap_ce;
    assign b_tready = input_ready && a_tvalid && ap_ce;
    assign c_tdata = skid_data;
    assign c_tvalid = skid_valid;

    wire slice_reset = rst;
    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;
    wire all_slices_done = &done_mat_mul;

{chr(10).join(chain_wires)}

{chr(10).join(data_wires)}

{chr(10).join(boundary)}

{chr(10).join(inst_lines)}

{chr(10).join(align_decl)}
{chr(10).join(align_assign)}
{chr(10).join(align_always)}

{chr(10).join(row_avail_decl)}

    wire any_avail = transaction_active && final_chunk && ({any_avail_expr});

    always @(*) begin
        row_mux = {c_width}'d0;
{chr(10).join(mux_lines)}
    end

    function [{c_stream_width-1}:0] pack_c_row;
        input [{c_width-1}:0] row_in;
        integer t;
        begin
            for (t = 0; t < {grid_cols}; t = t + 1)
                pack_c_row[t * 64 +: 64] = row_in[t * 128 +: 64];
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            out_row_count <= 16'd0;
            bias_cols <= {b_width}'d0;
            preload_d <= 1'b0;
            transaction_active <= 1'b0;
            skid_data <= {c_stream_width}'d0;
            skid_valid <= 1'b0;
            skid_last <= 1'b0;
        end else if (ap_ce) begin
            preload_d <= 1'b0;

            if (output_fire) begin
                skid_valid <= 1'b0;
                skid_last <= 1'b0;
            end

            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    out_row_count <= 16'd0;
                    transaction_active <= 1'b0;
                    if (bias_tvalid && bias_tready) begin
                        bias_cols <= bias_tdata;
                        preload_d <= 1'b1;
                        transaction_active <= 1'b1;
                        state <= S_PRELOAD;
                    end
                end

                S_PRELOAD: begin
                    state <= S_RUN;
                end

                S_RUN: begin
                    if (input_fire) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS)
                            state <= S_WAIT;
                    end

                    if (output_fire && skid_last) begin
                        state <= S_IDLE;
                    end

                    if (any_avail && !skid_valid) begin
                        skid_data <= pack_c_row(row_mux);
                        skid_valid <= 1'b1;
                        skid_last <= (out_row_count + 16'd1 == TOTAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == TOTAL_OUT_ROWS) begin
                            transaction_active <= 1'b0;
                        end
                    end
                end

                S_WAIT: begin
                    if (output_fire && skid_last) begin
                        state <= S_IDLE;
                    end

                    if (any_avail && !skid_valid) begin
                        skid_data <= pack_c_row(row_mux);
                        skid_valid <= 1'b1;
                        skid_last <= (out_row_count + 16'd1 == TOTAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == TOTAL_OUT_ROWS)
                            transaction_active <= 1'b0;
                    end else if (!final_chunk && all_slices_done) begin
                        chunk_idx <= chunk_idx + 16'd1;
                        beat_count <= 16'd0;
                        state <= S_RUN;
                    end
                end
            endcase
        end
    end

endmodule
"""


def _generate_buffered_vitis_synth_rtl(m, k, n, module_name="gemm_vitis", feed_mode="chained"):
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 128   # Catapult internal: 8 INT16 per column tile

    # ── Chained feed ─────────────────────────────────────────────────────────
    max_loc_delay = (grid_rows - 1 + grid_cols - 1) * 8
    feed_len = max_loc_delay + max(m, n)   # feed one row/col per cycle
    input_beats = max(m, n)

    # ── Latency ─────────────────────────────────────────────────────────────
    TOTAL_CYCLES = _total_cycles(m, k, n, feed_mode="chained")

    # ── Alignment ────────────────────────────────────────────────────────────
    def align_delay(r, c):
        return (grid_cols - 1 - c) * 8 + r * grid_cols * 8

    # ── Masks ────────────────────────────────────────────────────────────────
    k_mask_val = tail_mask_hex(k, 0)
    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

    # ── Per-tile data wires ──────────────────────────────────────────────────
    data_wires = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            a_hi = (r + 1) * 64 - 1
            a_lo = r * 64
            b_hi = (c + 1) * 64 - 1
            b_lo = c * 64
            loc_offset = (r + c) * 8
            data_wires.append(
                f"    wire lv_{r}_{c} = in_valid_d && (core_feed_idx < {input_beats});")
            data_wires.append(
                f"    wire [63:0] a_data_{r}_{c} = ({c} == 0 && lv_{r}_{c}) "
                f"? a_buf[core_feed_idx][{a_hi}:{a_lo}] : 64'b0;")
            data_wires.append(
                f"    wire [63:0] b_data_{r}_{c} = ({r} == 0 && lv_{r}_{c}) "
                f"? b_buf[core_feed_idx][{b_hi}:{b_lo}] : 64'b0;")

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
            .b_data(bias_phase_core ? bias_cols[{c}*64 +: 64] : b_data_{r}_{c}),
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
            .preload(preload_d), .no_rounding(1'b0),
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
    align_decl = []
    align_assign = []
    align_always = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            d = align_delay(r, c)
            acd = f"acd_{r}_{c}"
            aca = f"aca_{r}_{c}"
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
            mux_lines.append(f"            row_mux[{c}*128 +: 128] = acd_{r}_{c};")
        mux_lines.append(f"        end")

    any_avail_expr = " | ".join(f"row_avail_{r}" for r in range(grid_rows))

    total_output_rows = grid_rows * 8
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

    wire clk = ap_clk;
    wire rst = ap_rst;

    // ═══════════════════════════════════════════════════════════════════════════
    //  Protocol FSM — drives internal Catapult signals from AXI-Stream
    // ═══════════════════════════════════════════════════════════════════════════
    localparam FSM_IDLE=3'd0, FSM_PRELOAD=3'd1;
    localparam FSM_COLLECT=3'd2, FSM_FEED=3'd3, FSM_DRAIN=3'd4;

    reg [2:0]  fsm_state;
    reg [15:0] fsm_cnt;
    reg [15:0] feed_idx;

    // Beat buffers: collect K beats during COLLECT
    reg [{a_width-1}:0] a_buf [0:{input_beats-1}];
    reg [{b_width-1}:0] b_buf [0:{input_beats-1}];

    reg [15:0] a_wr_ptr;
    reg [15:0] b_wr_ptr;

    // Internal control signals
    reg        en;
    reg [{b_width-1}:0] bias_cols_int;
    reg        preload_valid;
    reg        in_valid;

    assign bias_tready = (fsm_state == FSM_IDLE);
    assign a_tready    = ((fsm_state == FSM_PRELOAD || fsm_state == FSM_COLLECT)
                          && (a_wr_ptr < {input_beats}));
    assign b_tready    = ((fsm_state == FSM_PRELOAD || fsm_state == FSM_COLLECT)
                          && (b_wr_ptr < {input_beats}));

    // Protocol FSM: Preload → Collect → Feed → Drain
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
                    if (a_tvalid && a_tready) begin
                        a_buf[a_wr_ptr] <= a_tdata;
                        a_wr_ptr <= a_wr_ptr + 1;
                    end
                    if (b_tvalid && b_tready) begin
                        b_buf[b_wr_ptr] <= b_tdata;
                        b_wr_ptr <= b_wr_ptr + 1;
                    end
                    if (a_wr_ptr >= {input_beats} && b_wr_ptr >= {input_beats}) begin
                        fsm_state <= FSM_FEED;
                        fsm_cnt   <= 0;
                        feed_idx  <= 0;
                        in_valid  <= 1;
                    end
                end

                FSM_FEED: begin
                    in_valid <= 1;
                    if (core_state == CORE_DRAIN) begin
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

    wire [{b_width-1}:0] bias_cols = bias_cols_int;

    // ── Output packing: int8 bytes from 128-bit tile ─────────────────────────
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
    //  Grid core — single continuous operation
    // ═══════════════════════════════════════════════════════════════════════════

    localparam integer FEED_BEATS  = {feed_len};
    localparam integer TOTAL_CYCLES = {TOTAL_CYCLES};
    localparam integer TOTAL_OUT_ROWS = {total_output_rows};

    localparam [1:0] CORE_IDLE=2'd0, CORE_PRELOAD=2'd1,
                     CORE_FEED=2'd2, CORE_DRAIN=2'd3;

    reg [2:0]  core_state;
    reg [15:0] cycle;
    reg        preload_d;
    wire       bias_phase_core = (core_state == CORE_PRELOAD);
    wire       transaction_active = (core_state == CORE_PRELOAD) || (core_state == CORE_FEED) || (core_state == CORE_DRAIN);
    wire       in_valid_d = (core_state == CORE_FEED);

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
    wire output_take = en & (core_state == CORE_DRAIN) & (out_count != 0) & c_tready;
    assign c_tvalid = output_take;
    assign c_tdata = output_take ? pack_c_row(c_row) : {c_stream_width}'d0;

    always @(posedge clk) begin
        if (rst) begin
            cycle <= 16'd0;
        end else if (ap_ce) begin
            if (core_state != CORE_IDLE) begin
                cycle <= cycle + 1;
            end else begin
                cycle <= 16'd0;
            end
        end
    end

    wire core_trigger = (fsm_state == FSM_FEED) && (core_state == CORE_IDLE);
    reg [15:0] core_feed_idx;

    always @(posedge clk) begin
        if (rst) begin
            core_state <= CORE_IDLE;
            core_feed_idx <= 16'd0;
            preload_d <= 1'b0;
            output_collect_active <= 1'b0;
        end else if (ap_ce) begin
            case (core_state)
                CORE_IDLE: begin
                    core_feed_idx <= 16'd0;
                    preload_d <= 1'b0;
                    output_collect_active <= 1'b0;
                    if (core_trigger) begin
                        core_state <= CORE_PRELOAD;
                        core_feed_idx <= 16'd0;
                        preload_d <= 1'b1;
                        output_collect_active <= 1'b1;
                    end
                end

                CORE_PRELOAD: begin
                    // Bias preload — slice sees preload=1, start_mat_mul=0
                    core_state <= CORE_FEED;
                    preload_d <= 1'b0;
                end

                CORE_FEED: begin
                    core_feed_idx <= core_feed_idx + 1;
                    if (core_feed_idx + 1 >= FEED_BEATS) begin
                        core_state <= CORE_DRAIN;
                    end
                end

                CORE_DRAIN: begin
                    if (output_collect_active && (out_wr_ptr == TOTAL_OUT_ROWS) && !any_avail)
                        output_collect_active <= 1'b0;
                    if ((output_collect_active == 1'b0 && out_count == 0)
                        || (cycle + 1 >= TOTAL_CYCLES))
                        core_state <= CORE_IDLE;
                end
            endcase
        end
    end

    assign slice_start = (core_state == CORE_FEED) && (core_feed_idx == 16'd0);

    // ---- Wire declarations ----
    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;

{chr(10).join(chain_wires)}

    // ---- Systolic chain boundaries ----
{chr(10).join(boundary)}

    // ---- Per-tile data wires ----
{chr(10).join(data_wires)}

    // ---- Slice instantiations ----
{insts}

    // ---- Alignment buffers ----
{chr(10).join(align_decl)}
{chr(10).join(align_assign)}
{chr(10).join(align_always)}

    // ---- Per-row-tile availability ----
{chr(10).join(row_avail_decl)}

    reg output_collect_active;
    wire any_avail = output_collect_active && ({any_avail_expr});

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


def generate_vitis_rtl(m, k, n, module_name="gemm_vitis", feed_mode="chained"):
    return generate_vitis_synth_rtl(m, k, n, module_name, feed_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Vitis grid wrapper for tensor_slice_int8 modules")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--name", type=str, default="gemm_vitis")
    parser.add_argument("--output", type=str, default="gemm_vitis.v")
    parser.add_argument("--sim-output", type=str, default=None)
    parser.add_argument("--synth-output", type=str, default=None)
    args = parser.parse_args()

    if args.sim_output or args.synth_output:
        sim_output = args.sim_output or args.output.replace(".v", "_sim.v")
        synth_output = args.synth_output or args.output.replace(".v", "_synth.v")
        Path(sim_output).write_text(generate_vitis_sim_rtl(args.m, args.k, args.n, args.name))
        Path(synth_output).write_text(generate_vitis_synth_rtl(args.m, args.k, args.n, args.name))
        print(f"Generated {sim_output} and {synth_output}  (M={args.m}, K={args.k}, N={args.n})")
    else:
        content = generate_vitis_synth_rtl(args.m, args.k, args.n, args.name)
        Path(args.output).write_text(content)
        print(f"Generated {args.output}  (M={args.m}, K={args.k}, N={args.n})")
