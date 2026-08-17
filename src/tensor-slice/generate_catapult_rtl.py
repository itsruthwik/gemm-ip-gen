#!/usr/bin/env python3
"""
generate_catapult_rtl.py
Catapult RTL generator — tensor-slice grid with clk/rst/en protocol.
M x N output grid with K inner dimension.

Key design:
  - Single continuous operation: collect K beats, feed tiles in chained fashion,
    drain outputs once.
  - All masks resolved in Python, emitted as literals (no generate arithmetic)
  - Slices unrolled explicitly for portability
  - Output mux: each tile-row (r) streams 8 result rows in order,
    staggered correctly against other tile-rows
"""
import argparse
from pathlib import Path

from _generate_rtl_common import tail_mask_hex, vm, total_cycles as _total_cycles


def generate_sim_verilog(m, k, n, module_name="gemm_grid_wrapper", full_k_spatial=False,
                         requant_shift=0, requant_bits=None, weight_rom=None, emit_rom=True):
    # Body of requant_acc(): applied ONCE to the fully-accumulated dot product.
    # Emit requant_acc() ONLY when it is used. A declared-but-unused function is
    # dead Verilog, but it still perturbs synthesis (measured -1.2% Fmax on a
    # requant_shift==0 design), so requant_shift==0 must reproduce the original
    # output byte-for-byte.
    _use_requant = bool(requant_shift and requant_shift > 0)
    if _use_requant:
        _w = requant_bits or 8
        _half = 1 << (requant_shift - 1)
        requant_fn = f"""
    // Requantise the FULL contraction once, after everything has been summed
    // (in-slice and cross-chunk alike are folded into `acc` here, which is the
    // whole point of the behavioural model). Round-half-up, shift, then reduce
    // to the output width.
    function signed [15:0] requant_acc;
        input signed [31:0] x;
        reg signed [31:0] r;
        begin
            r = (x + 32'sd{_half}) >>> {requant_shift};
            requant_acc = $signed(r[{_w-1}:0]);
        end
    endfunction
"""
    else:
        requant_fn = ""
    _sat_call = "requant_acc(acc)" if _use_requant else "sat_int8(acc)"
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    a_chunk_width = a_width
    b_chunk_width = b_width
    c_width = grid_cols * 128
    input_beats = max(m, n)
    k_chunks = (k + 7) // 8
    if full_k_spatial:
        # Narrow per-beat word: one tile, 64 bits per K-chunk, no grid padding.
        a_width = 64 * k_chunks
        b_width = 64 * k_chunks
    bias_width = grid_cols * 64
    total_input_beats = input_beats if full_k_spatial else k_chunks * input_beats
    total_output_rows = m
    latency = max(0, k + n - total_input_beats)
    # First-output offset == catapult.latency_cycles(full_k_spatial=...): feed
    # beats + the systolic K+N wave remainder. Full-K mode feeds every K chunk
    # spatially in one max(M,N)-beat pass, so its first_out drops accordingly;
    # the C++ core and the wrapper's DRAIN capture window use the same formula.
    first_out = total_input_beats + latency
    behav_name = f"{module_name}_behav_grid"
    mode_comment = (
        "Full K-spatial behavioral MxKxN GEMM. Not intended for synthesis."
        if full_k_spatial
        else "Chunked behavioral MxKxN GEMM. Not intended for synthesis."
    )
    # Frames in flight: the feed of frame t+1 overlaps the compute/drain of
    # frame t, so back-to-back frames sustain a frame II of total_beats+1
    # (the preload step plus the data beats). Slot count covers the deepest
    # overlap plus one spare so an allocating frame never lands on a slot
    # that is still draining.
    frame_period = total_input_beats + 1
    slots = -(-(first_out + 1 + total_output_rows) // frame_period) + 1
    # Per-slot unpack: capture a_rows/b_cols into the allocating slot's amat/
    # bmat partition at beat index `cc` (== that frame's clk_cnt while
    # collecting). `ws` is the slot index resolved this cycle.
    single_unpack = f"""\
                    cc_chunk = cc / INPUT_BEATS;
                    cc_beat  = cc % INPUT_BEATS;
                    if (cc_beat < {m}) begin
                        for (lane = 0; lane < 8; lane = lane + 1) begin
                            kk = cc_chunk * 8 + lane;
                            if (kk < {k})
                                amat[ws * {m} + cc_beat][kk] = a_rows[(cc_beat / 8) * 64 + lane * 8 +: 8];
                        end
                    end
                    if (cc_beat < {n}) begin
                        for (lane = 0; lane < 8; lane = lane + 1) begin
                            kk = cc_chunk * 8 + lane;
                            if (kk < {k})
                                bmat[ws * {k} + kk][cc_beat] = b_cols[(cc_beat / 8) * 64 + lane * 8 +: 8];
                        end
                    end"""
    if full_k_spatial:
        single_unpack = f"""\
                    cc_beat = cc;
                    if (cc_beat < {m}) begin
                        for (kk = 0; kk < {k}; kk = kk + 1) begin
                            cc_chunk = kk / 8;
                            lane = kk % 8;
                            amat[ws * {m} + cc_beat][kk] = a_rows[cc_chunk * 64 + lane * 8 +: 8];
                        end
                    end
                    if (cc_beat < {n}) begin
                        for (kk = 0; kk < {k}; kk = kk + 1) begin
                            cc_chunk = kk / 8;
                            lane = kk % 8;
                            bmat[ws * {k} + kk][cc_beat] = b_cols[cc_chunk * 64 + lane * 8 +: 8];
                        end
                    end"""

    # Weight-stationary (const-weight): the top sim wrapper drops its external b_cols
    # port and feeds the inner behav_grid from the shared ROM (w_rom_out). The behav_grid
    # keeps its internal b_cols input, now driven by the ROM. Bias stays external.
    # Full-K is supported: b_width is already the narrow 64*k_chunks word above, and
    # the ROM holds one beat per presented input cycle (total_input_beats ==
    # input_beats under full-K), so the widened word carries every K chunk without
    # adding beats. The ROM must be built with the matching narrow full-K packer —
    # gemm_ip.weights.build_weight_rom_full_k.
    _sim_ws = weight_rom is not None
    if _sim_ws:
        sim_b_cols_port = ""
        sim_b_src = "w_rom_out"
        sim_rom_block = _weight_rom_block(b_width, weight_rom) if emit_rom else ""
    else:
        sim_b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        sim_b_src = "b_cols"
        sim_rom_block = ""

    return f"""\
// Auto-generated simulation model by generate_catapult_rtl.py
// {mode_comment}
// Dimensions: M={m}, K={k}, N={n}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{sim_b_cols_port}    input  wire [{bias_width-1}:0]   bias_cols,
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{sim_rom_block}
    wire [{c_width-1}:0] behav_c_row;
    wire                 behav_out_valid;
    wire                 behav_out_last;

    {behav_name} grid (
        .clk(clk), .rst(rst), .en(en),
        .a_rows(a_rows), .b_cols({sim_b_src}), .bias_cols(bias_cols),
        .preload_valid(preload_valid), .in_valid(in_valid),
        .c_row(behav_c_row), .out_valid(behav_out_valid), .out_last(behav_out_last)
    );

    always @(posedge clk) begin
        if (rst) begin
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
        end else if (en) begin
            c_row <= behav_c_row;
            out_valid <= behav_out_valid;
            out_last <= behav_out_last;
        end
    end

endmodule

module {behav_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
    input  wire [{b_width-1}:0]   b_cols,
    input  wire [{bias_width-1}:0]   bias_cols,
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);

    localparam integer INPUT_BEATS       = {input_beats};
    localparam integer K_CHUNKS          = {k_chunks};
    localparam integer TOTAL_INPUT_BEATS = {total_input_beats};
    localparam integer FIRST_OUT         = {first_out};
    localparam integer TOTAL_ROWS        = {total_output_rows};
    localparam integer FRAME_SLOTS       = {slots};

    // ── Frame-slot storage: up to FRAME_SLOTS frames in flight ─────────────
    // Pure behavioral scheduler (no tensor-slice structure): each frame gets
    // a private operand partition and cycle counter; results are computed at
    // emit time in full precision. Feed of frame t+1 overlaps compute/drain
    // of frame t, so back-to-back frames sustain frame II = TOTAL_INPUT_BEATS+1
    // (~one result row per cycle) while each frame keeps its own FIRST_OUT.
    reg signed [7:0]  amat [0:{slots * m - 1}][0:{k - 1}];
    reg signed [7:0]  bmat [0:{slots * k - 1}][0:{n - 1}];
    reg signed [7:0]  bias_buf [0:{slots * n - 1}];
    reg [31:0] cc_slot [0:{slots - 1}];
    reg        slot_run [0:{slots - 1}];
    reg [31:0] wr_slot;
    reg        feeding;

    // ── Diagnostics (parsed by the testbench; not load-bearing) ────────────
    reg [31:0] beh_cyc;
    reg [31:0] prev_beh_start;
    reg [31:0] beh_ii_val;

    integer i, j, kk, lane, tile, s;
    integer ws, cc, scc, cc_chunk, cc_beat, out_idx, actual_row, actual_col;
    reg signed [31:0] acc;
    reg signed [15:0] sat;

    function signed [15:0] sat_int8;
        input signed [31:0] x;
        begin
            if (x > 32'sd127) sat_int8 = 16'sd127;
            else if (x < -32'sd128) sat_int8 = -16'sd128;
            else sat_int8 = x[15:0];
        end
    endfunction
{requant_fn}
    // ═══════════════════════════════════════════════════════════════════════
    // Frame-slot clk_cnt schedule. A frame starts at the first in_valid call
    // after a non-in_valid call (the wrapper protocol always inserts at least
    // the preload step between frames): allocate the next slot, capture its
    // TOTAL_INPUT_BEATS beats, then emit one row per cycle for that slot's
    // cc in [FIRST_OUT+1, FIRST_OUT+1+TOTAL_ROWS) and retire it. Emission
    // windows of consecutive frames cannot overlap (TOTAL_ROWS <= the minimum
    // frame period TOTAL_INPUT_BEATS+1). Everything advances on en only, so
    // wrapper stalls keep the model aligned with the run()-call schedule.
    // ═══════════════════════════════════════════════════════════════════════
    always @(posedge clk) begin
        if (rst) begin
            for (s = 0; s < FRAME_SLOTS; s = s + 1) begin
                cc_slot[s]  <= 32'd0;
                slot_run[s] <= 1'b0;
            end
            wr_slot        <= FRAME_SLOTS - 1;
            feeding        <= 1'b0;
            c_row          <= {c_width}'d0;
            out_valid      <= 1'b0;
            out_last       <= 1'b0;
            beh_cyc        <= 32'd0;
            prev_beh_start <= 32'd0;
            beh_ii_val     <= 32'd0;
        end else if (en) begin
            beh_cyc   <= beh_cyc + 1;
            c_row     <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            // ── capture into the feeding frame's slot ────────────────────
            ws = wr_slot;
            cc = 0;
            if (in_valid) begin
                if (!feeding) begin
                    ws = (wr_slot + 1) % FRAME_SLOTS;
                    if (slot_run[ws]) $display("BEH_SLOT_OVERFLOW beh_cyc=%0d", beh_cyc);
                    wr_slot      <= ws;
                    feeding      <= 1'b1;
                    slot_run[ws] <= 1'b1;
                    cc_slot[ws]  <= 32'd1;   // cc = 0 is consumed this cycle
                    for (j = 0; j < {n}; j = j + 1)
                        bias_buf[ws * {n} + j] <= bias_cols[(j / 8) * 64 + (j % 8) * 8 +: 8];
                    if (prev_beh_start == 0) beh_ii_val <= 32'd0;
                    else                     beh_ii_val <= beh_cyc - prev_beh_start;
                    prev_beh_start <= beh_cyc;
                    $display("BEH_START beh_cyc=%0d", beh_cyc);
                    cc = 0;
                end else begin
                    cc = cc_slot[ws];
                end
                if (cc < TOTAL_INPUT_BEATS) begin
{single_unpack}
                end
            end else begin
                feeding <= 1'b0;
            end

            // ── per-slot emit (compute-at-emit) + advance / retire ───────
            // slot_run[] reads pre-edge values, so the slot allocated THIS
            // cycle (cc already set to 1 above) is naturally skipped.
            for (s = 0; s < FRAME_SLOTS; s = s + 1) begin
                if (slot_run[s]) begin
                    scc = cc_slot[s];
                    if (scc >= FIRST_OUT + 1 && scc < FIRST_OUT + 1 + TOTAL_ROWS) begin
                        out_idx    = scc - (FIRST_OUT + 1);
                        actual_row = out_idx;
                        for (tile = 0; tile < {grid_cols}; tile = tile + 1) begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                actual_col = tile * 8 + lane;
                                if (actual_row < {m} && actual_col < {n}) begin
                                    acc = bias_buf[s * {n} + actual_col];
                                    for (kk = 0; kk < {k}; kk = kk + 1)
                                        acc = acc + (amat[s * {m} + actual_row][kk] * bmat[s * {k} + kk][actual_col]);
                                    sat = {_sat_call};
                                end else begin
                                    sat = 16'sd0;
                                end
                                c_row[tile * 128 + lane * 16 +: 16] <= sat;
                            end
                        end
                        out_valid <= 1'b1;
                        out_last  <= (out_idx + 1 == TOTAL_ROWS) ? 1'b1 : 1'b0;
                    end
                    cc_slot[s] <= scc + 1;
                    if (scc + 1 >= FIRST_OUT + 1 + TOTAL_ROWS) begin
                        slot_run[s] <= 1'b0;
                        cc_slot[s]  <= 32'd0;
                        $display("BEH_II=%0d", beh_ii_val);
                        $display("BEH_DONE beh_cyc=%0d", beh_cyc);
                    end
                end
            end
        end
    end

endmodule
"""


def _weight_rom_block(b_width, weight_rom):
    """Shared const-weight ROM: declaration + inline init + feed_ptr + w_rom_out wire.

    Emitted once (above the `ifndef SYNTHESIS` split in the combined core) so a single
    ROM feeds both the behavioral-sim and structural-synth branches. `feed_ptr` advances
    one entry per presented input beat (mirrors the free-running `in_valid`), reproducing
    the exact order the external `b_cols` port received (pack_b_chunk feed order).
    """
    hexw = (b_width + 3) // 4
    mask = (1 << b_width) - 1
    nbeats = len(weight_rom)
    rom_init = "\n".join(
        f"        w_rom[{i}] = {b_width}'h{(int(v) & mask):0{hexw}x};"
        for i, v in enumerate(weight_rom)
    )
    return f"""
    // Weight-stationary const-weight ROM (baked; no external b_cols port). One beat
    // per presented input cycle; feeds both the sim and synth branches below.
    reg [{b_width - 1}:0] w_rom [0:{nbeats - 1}];
    initial begin
{rom_init}
    end
    reg [15:0] feed_ptr;
    always @(posedge clk) begin
        if (rst) feed_ptr <= 16'd0;
        else if (en) begin
            if (!in_valid) feed_ptr <= 16'd0;
            else if (feed_ptr < 16'd{nbeats - 1}) feed_ptr <= feed_ptr + 16'd1;
        end
    end
    wire [{b_width - 1}:0] w_rom_out = w_rom[feed_ptr];
"""


def generate_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="chained", debug=False,
                           weight_rom=None, emit_rom=True):
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 128
    input_beats = max(m, n)
    total_output_rows = grid_rows * 8
    k_chunks = (k + 7) // 8
    last_k_size = k - (k_chunks - 1) * 8
    last_k_mask = tail_mask_hex(k, k_chunks - 1)

    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

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
                f"    wire [63:0] a_data_{r}_{c} = (in_beat_active && ({c} == 0)) ? a_rows_q[{a_hi}:{a_lo}] : 64'b0;"
            )
            data_wires.append(
                f"    wire [63:0] b_data_{r}_{c} = (in_beat_active && ({r} == 0)) ? b_cols_q[{b_hi}:{b_lo}] : 64'b0;"
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
            .b_data(preload_d ? bias_cols_q[{c}*64 +: 64] : b_data_{r}_{c}),
            .a_data_in(a_chain_{r}_{c}),
            .b_data_in(b_chain_{r}_{c}),
            .a_data_out(a_chain_{r}_{c+1}),
            .b_data_out(b_chain_{r+1}_{c}),
            .c_data_out(c_data_{r}_{c}),
            .c_data_available(c_avail_{r}_{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows(current_k_mask),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op({{2'b00, op0_{r}}}),
            .preload(preload_d), .no_rounding(1'b0),
            .final_mat_mul_size(current_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    # ── Zero-storage output collector (op[0] readout gate) ──────────────────
    # tensor_slice_int8 contract: op[0] == out_ctrl. With op[0]=1 the tile
    # HOLDS its completed result internally (no burst, including after
    # intermediate K-chunks); with op[0]=0 it shifts one result row per cycle
    # on c_data_out once ready (c_data_available qualifies each row). The
    # legacy free-run behaviour is op[0] tied 0. All tiles finish together,
    # so column tiles of one tile-row concatenate as pure wiring; the wrapper
    # holds every tile-row and releases them one at a time, in row-major
    # order, for their 8-row bursts. No parking storage and no delay pyramid
    # (the old alignment shift-lines cost sum-of-delays x 129 FFs — ~6.2k on
    # a 2x2 grid, ~29k on 4x2).
    op0_decl = []
    for r in range(grid_rows):
        release = (f"emit_phase && (cur_row_tile == 16'd{r})"
                   if grid_rows > 1 else "emit_phase")
        op0_decl.append(f"    wire op0_{r} = !({release});")
    row_avail_decl = []
    row_data_decl = []
    for r in range(grid_rows):
        avail_terms = " & ".join(f"c_avail_{r}_{c}" for c in range(grid_cols))
        concat = "{" + ", ".join(f"c_data_{r}_{c}" for c in range(grid_cols - 1, -1, -1)) + "}"
        row_avail_decl.append(f"    wire row_avail_{r} = {avail_terms};")
        row_data_decl.append(f"    wire [{c_width-1}:0] row_data_{r} = {concat};")
    mux_lines = []
    for r in range(grid_rows):
        cond = (f"(cur_row_tile == 16'd{r}) && row_avail_{r}"
                if grid_rows > 1 else f"row_avail_{r}")
        mux_lines.append(f"        if ({cond}) begin")
        mux_lines.append(f"            row_mux = row_data_{r};")
        mux_lines.append("            row_take = 1'b1;")
        mux_lines.append("        end")

    debug_block = ""
    if debug:
        debug_block = """
    always @(posedge clk) begin
        if (!slice_reset && (transaction_active || row_take || |done_mat_mul)) begin
            $display("DBG t=%0t st=%0d beat=%0d done=%b take=%0b out_rows=%0d",
                     $time, state, beat_count, done_mat_mul, row_take, out_row_count);
        end
    end
"""

    # ── Weight-stationary (const-weight) variant ────────────────────────────
    # weight_rom = per-beat b_cols values (grid_cols*64 bits each), already in the
    # pack_b_chunk feed order (see gemm_ip/weights.py). When present, drop the
    # external b_cols port and source B from an internal ROM: one beat per presented
    # input cycle. feed_ptr mirrors in_valid exactly as the external b_cols port did
    # (the TB free-runs one beat/clock while in_valid is high), so timing/systolic
    # feed are byte-identical to the streamed path. Bias stays external.
    ws = weight_rom is not None
    if ws:
        b_cols_port = ""
        b_cols_q_src = "w_rom_out"
        # emit_rom=False when the combined core provides the shared ROM above `ifndef.
        w_rom_block = _weight_rom_block(b_width, weight_rom) if emit_rom else ""
    else:
        b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        b_cols_q_src = "b_cols"
        w_rom_block = ""

    return f"""\
// Auto-generated by generate_catapult_rtl.py
// Chunked structural tensor-slice synth wrapper
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{b_cols_port}    input  wire [{b_width-1}:0]   bias_cols,
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{w_rom_block}

    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer LAST_K_SIZE = {last_k_size};
    localparam integer TOTAL_OUT_ROWS = {total_output_rows};
    localparam [1:0] S_IDLE=2'd0, S_PRELOAD=2'd1, S_RUN=2'd2, S_WAIT=2'd3;

    reg [1:0] state;
    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] out_row_count;
    reg preload_d;
    reg transaction_active;
    reg [{c_width-1}:0] row_mux;
    reg row_take;

    // ── Input pipeline stage ────────────────────────────────────────────────
    // Register the whole input bundle (en-gated, so the core stays self-timed),
    // keeping the beat decode and data gating muxes off the path into the
    // tensor_slice input pins.
    reg [{a_width-1}:0] a_rows_q;
    reg [{b_width-1}:0] b_cols_q;
    reg [{b_width-1}:0] bias_cols_q;
    reg preload_valid_q;
    reg in_valid_q;

    always @(posedge clk) begin
        if (rst) begin
            a_rows_q        <= {a_width}'d0;
            b_cols_q        <= {b_width}'d0;
            bias_cols_q     <= {b_width}'d0;
            preload_valid_q <= 1'b0;
            in_valid_q      <= 1'b0;
        end else if (en) begin
            a_rows_q        <= a_rows;
            b_cols_q        <= {b_cols_q_src};
            bias_cols_q     <= bias_cols;
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = (state == S_RUN) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire slice_pe_reset = slice_start && (chunk_idx == 16'd0);
    wire final_chunk = (chunk_idx == K_CHUNKS - 1);
    wire [7:0] current_k_size = final_chunk ? 8'd{last_k_size} : 8'd8;
    wire [7:0] current_k_mask = final_chunk ? {vm(last_k_mask)} : 8'hFF;

    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;
    wire all_slices_done = &done_mat_mul;

    // Output collector control: hold every tile-row (out_ctrl=1) until all
    // tiles have finished the final chunk, then release one tile-row at a
    // time for its 8-row burst, in row-major order.
    wire emit_phase = transaction_active && final_chunk && all_slices_done;
    wire [15:0] cur_row_tile = out_row_count >> 3;

{chr(10).join(op0_decl)}

{chr(10).join(chain_wires)}

{chr(10).join(data_wires)}

{chr(10).join(boundary)}

{chr(10).join(inst_lines)}

{chr(10).join(row_avail_decl)}
{chr(10).join(row_data_decl)}

{debug_block}

    always @(*) begin
        row_mux  = {c_width}'d0;
        row_take = 1'b0;
{chr(10).join(mux_lines)}
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            out_row_count <= 16'd0;
            preload_d <= 1'b0;
            transaction_active <= 1'b0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
        end else if (en) begin
            preload_d <= 1'b0;
            out_valid <= 1'b0;
            out_last <= 1'b0;

            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    out_row_count <= 16'd0;
                    transaction_active <= 1'b0;
                    if (preload_valid_q) begin
                        preload_d <= 1'b1;
                        transaction_active <= 1'b1;
                        state <= S_PRELOAD;
                    end
                end

                S_PRELOAD: begin
                    state <= S_RUN;
                end

                S_RUN: begin
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS)
                            state <= S_WAIT;
                    end

                    if (emit_phase && row_take) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= (out_row_count + 16'd1 == TOTAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == TOTAL_OUT_ROWS) begin
                            transaction_active <= 1'b0;
                            state <= S_IDLE;
                        end
                    end
                end

                S_WAIT: begin
                    if (emit_phase && row_take) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= (out_row_count + 16'd1 == TOTAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == TOTAL_OUT_ROWS) begin
                            transaction_active <= 1'b0;
                            state <= S_IDLE;
                        end
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


def _generate_buffered_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="chained", debug=False):
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 128   # 8 INT16 values per column tile

    # ── Chained feed ─────────────────────────────────────────────────────────
    max_loc_delay = (grid_rows - 1 + grid_cols - 1) * 8
    feed_len = max_loc_delay + max(m, n)   # feed one row/col per cycle
    input_beats = max(m, n)

    # ── Latency ─────────────────────────────────────────────────────────────
    TOTAL_CYCLES = _total_cycles(m, k, n, feed_mode="chained")

    # ── Alignment ────────────────────────────────────────────────────────────
    def align_delay(c):
        return (grid_cols - 1 - c) * 8

    # ── Masks ────────────────────────────────────────────────────────────────
    k_mask_val    = tail_mask_hex(k, 0)   # single K mask (no K-step)
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
                f"    wire lv_{r}_{c} = in_valid_d && (feed_idx < {input_beats});")
            data_wires.append(
                f"    wire [63:0] a_data_{r}_{c} = ({c} == 0 && lv_{r}_{c}) "
                f"? a_buf[feed_idx][{a_hi}:{a_lo}] : 64'b0;")
            data_wires.append(
                f"    wire [63:0] b_data_{r}_{c} = ({r} == 0 && lv_{r}_{c}) "
                f"? b_buf[feed_idx][{b_hi}:{b_lo}] : 64'b0;")

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
            .b_data(bias_phase ? bias_cols[{c}*64 +: 64] : b_data_{r}_{c}),
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
            d = align_delay(c) + r * 8
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

    # ── Debug traces ─────────────────────────────────────────────────────────
    c_avail_vec = ", ".join(
        f"c_avail_{r}_{c}" for r in reversed(range(grid_rows)) for c in reversed(range(grid_cols)))
    row_avail_vec = ", ".join(f"row_avail_{r}" for r in reversed(range(grid_rows)))
    aca_vec = ", ".join(
        f"aca_{r}_{c}" for r in reversed(range(grid_rows)) for c in reversed(range(grid_cols)))
    debug_block = ""
    if debug:
        debug_block = f"""
    // ---- Debug traces ----
    always @(posedge clk) begin
        if (!slice_reset && (transaction_active || any_avail || |done_mat_mul)) begin
            $display("DBG t=%0t cyc=%0d st=%0d feed=%0d trans=%0b done=%b c_avail=%b aca=%b row_avail=%b any=%0b wr=%0d rd=%0d cnt=%0d out_rows=%0d",
                     $time, cycle, fsm_state, feed_idx,
                     transaction_active, done_mat_mul,
                     {{{c_avail_vec}}}, {{{aca_vec}}}, {{{row_avail_vec}}},
                     any_avail, out_wr_ptr, out_rd_ptr, out_count, out_row_count);
        end
    end
"""

    total_output_rows = grid_rows * 8

    verilog = f"""\
// Auto-generated by generate_catapult_rtl.py
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices
// TOTAL_CYCLES = {TOTAL_CYCLES}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,        // 1=reset, 0=running
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,     // packed A: grid_rows*64 bits per K-position
    input  wire [{b_width-1}:0]   b_cols,     // packed B: grid_cols*64 bits per K-position
    input  wire [{b_width-1}:0]   bias_cols,  // packed bias tiles, one per column-tile
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);

    localparam integer INPUT_BEATS     = {input_beats};
    localparam integer FEED_BEATS      = {feed_len};
    localparam integer TOTAL_CYCLES    = {TOTAL_CYCLES};
    localparam integer TOTAL_OUT_ROWS  = {total_output_rows};

    // FSM: IDLE -> COLLECT -> PRELOAD -> FEED -> DRAIN
    localparam [2:0] FSM_IDLE=3'd0, FSM_COLLECT=3'd1,
                     FSM_PRELOAD=3'd2, FSM_FEED=3'd3, FSM_DRAIN=3'd4;

    reg [2:0] fsm_state;
    reg [15:0] cycle;
    reg [15:0] beat_cnt;
    reg [15:0] feed_idx;
    reg        preload_d;         // bias preload for slice
    wire       bias_phase = (fsm_state == FSM_PRELOAD);
    reg        transaction_active;
    reg        output_collect_active;
    wire       in_valid_d = (fsm_state == FSM_FEED);

    reg [{a_width-1}:0] a_buf [0:{input_beats-1}];
    reg [{b_width-1}:0] b_buf [0:{input_beats-1}];

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
            preload_d <= 1'b0;
            output_collect_active <= 1'b0;
            transaction_active <= 1'b0;
        end else if (en) begin
            case (fsm_state)
                FSM_IDLE: begin
                    cycle <= 16'd0;
                    beat_cnt <= 16'd0;
                    preload_d <= 1'b0;
                    output_collect_active <= 1'b0;
                    if (preload_valid) begin
                        fsm_state <= FSM_COLLECT;
                        beat_cnt <= 16'd0;
                        transaction_active <= 1'b1;
                    end
                end

                FSM_COLLECT: begin
                    cycle <= cycle + 1;
                    if (in_valid && (beat_cnt < INPUT_BEATS)) begin
                        a_buf[beat_cnt] <= a_rows;
                        b_buf[beat_cnt] <= b_cols;
                        beat_cnt <= beat_cnt + 1;
                    end
                    if (beat_cnt >= INPUT_BEATS) begin
                        fsm_state <= FSM_PRELOAD;
                        preload_d <= 1'b1;
                    end
                end

                FSM_PRELOAD: begin
                    // Bias preload cycle — slice sees preload=1, start_mat_mul=0
                    cycle <= cycle + 1;
                    preload_d <= 1'b0;  // clear for next cycle
                    output_collect_active <= 1'b1;
                    fsm_state <= FSM_FEED;
                    feed_idx <= 16'd0;
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
                    if (output_collect_active && (out_wr_ptr == TOTAL_OUT_ROWS) && !any_avail)
                        output_collect_active <= 1'b0;
                    if ((output_collect_active == 1'b0 && out_count == 0) || (cycle + 1 >= TOTAL_CYCLES)) begin
                        transaction_active <= 1'b0;
                        fsm_state <= FSM_IDLE;
                    end
                end
            endcase
        end
    end

    assign slice_start = (fsm_state == FSM_FEED) && (feed_idx == 16'd0);

    // ---- Wire declarations ----
    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;

{chr(10).join(chain_wires)}

    // ---- Per-tile data wires ----
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

    wire any_avail = output_collect_active && ({any_avail_expr});

{debug_block}

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


def generate_grid_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="chained", debug=False):
    return generate_synth_verilog(m, k, n, module_name, feed_mode, debug)


def _widen_output_saturation(text, out_bits):
    """Rewrite the hardcoded int8 (±127) output-saturation bounds to an
    ``out_bits``-wide signed range.

    The Catapult core always packs 16-bit output lanes (``c_row[...+:16]``,
    ``c_bits = grid_cols * 128``); only the clamp *value* was int8, which
    silently capped the GEMM result at ±127 regardless of the configured
    ``output_precision``. This rewrites the clamp to the output range
    (e.g. ±32767 for a 16-bit ``output_precision``). The ``32'sd127`` /
    ``16'sd127`` / ``-32'sd128`` / ``-16'sd128`` tokens appear ONLY inside the
    ``sat_int8`` / ``sat_int8_to_i16`` functions, so the substitution is exact.
    No-op for ``out_bits == 8`` (preserves the legacy int8 contract).
    """
    if out_bits is None or out_bits == 8:
        return text
    if out_bits > 16:
        # The tensor_slice output lane is physically 16 bits, so the saturation
        # clamp can be at most ±2^15. Cap here rather than rejecting the design:
        # the GEMM result is lossless as long as its raw integer magnitude fits
        # 16 bits, which the Keras-vs-C-sim gate certifies per design (a design
        # that genuinely overflows would saturate and fail that bit-exact check).
        import sys as _sys
        print(f"  [gemm-ip-gen] WARNING: output_precision is {out_bits} bits but the "
              f"Catapult tensor_slice lane is 16 bits; clamping saturation to 16 bits. "
              f"Correctness depends on the raw GEMM result fitting 16 bits "
              f"(verify via Keras-vs-C-sim).", file=_sys.stderr)
        out_bits = 16
    pos = (1 << (out_bits - 1)) - 1          # e.g. 32767
    neg = (1 << (out_bits - 1))              # e.g. 32768
    subs = [("32'sd127", f"32'sd{pos}"), ("16'sd127", f"16'sd{pos}"),
            ("-32'sd128", f"-32'sd{neg}"), ("-16'sd128", f"-16'sd{neg}")]
    for old, new in subs:
        text = text.replace(old, new)
    return text


def _split_module(text, name):
    """Split a single-module Verilog string into (header, body).

    header = 'module {name}(' … '\\n);'  (port list, inclusive).
    body   = everything after the port-list close up to (excluding) the final endmodule.
    """
    start = text.index(f"module {name}(")
    pclose = text.index("\n);", start) + len("\n);")
    end = text.rindex("endmodule")
    return text[start:pclose], text[pclose:end]


def _split_after_first_endmodule(text):
    """Split at the first endmodule → (first_module_incl_endmodule, remainder)."""
    idx = text.index("endmodule") + len("endmodule")
    rest = text[idx:].lstrip("\n")
    if rest and not rest.endswith("\n"):
        rest += "\n"
    return text[:idx], rest


def generate_combined_core_verilog(m, k, n, module_name="gemm_grid_wrapper", out_bits=8,
                                   requant_shift=0, requant_bits=None, weight_rom=None):
    """Generate a single {module_name}.v with ifndef SYNTHESIS guard.

    ``ifndef SYNTHESIS`` — behavioral simulation model (wrapper + behav_grid).
    Used by Catapult SCVerify for RTL vs C++ co-simulation.

    ``else`` — structural synth wrapper with tensor_slice_int8 black-box slices.
    Used by Catapult HLS → downstream synthesis (Design Compiler).

    Both share the same port list so the ac_blackbox() binding is identical.

    ``out_bits`` is the result-lane width derived from ``output_precision``
    (default 8 = legacy int8 clamp; 16 = honor a fixed<16,…> output_precision).
    """
    # Weight-stationary: hoist a SINGLE weight ROM above the `ifndef so both the sim
    # and synth branches read the same w_rom_out (one source of truth; sim ≡ synth
    # weights by construction). The combined core becomes ONE module with the `ifndef
    # INSIDE it; the sim's behav_grid helper stays a trailing module.
    if weight_rom is not None:
        sim_top = generate_sim_verilog(m, k, n, module_name, requant_shift=requant_shift,
                                       requant_bits=requant_bits, weight_rom=weight_rom, emit_rom=False)
        synth_top = generate_synth_verilog(m, k, n, module_name, weight_rom=weight_rom, emit_rom=False)
        b_width = ((n + 7) // 8) * 64
        header, syn_body = _split_module(synth_top, module_name)   # header incl. 'module..);'
        sim_wrapper, sim_behav = _split_after_first_endmodule(sim_top)
        _, sim_body = _split_module(sim_wrapper, module_name)
        rom_block = _weight_rom_block(b_width, weight_rom)
        out = (
            f"// Auto-generated by generate_catapult_rtl.py\n"
            f"// Combined core (weight-stationary): M={m}, K={k}, N={n}\n"
            f"//   shared const-weight ROM above `ifndef feeds both branches\n"
            f"{header}\n{rom_block}\n"
            f"`ifndef SYNTHESIS\n{sim_body}`else\n{syn_body}`endif\n"
            f"endmodule\n\n"
            f"`ifndef SYNTHESIS\n{sim_behav}`endif\n"
        )
        return _widen_output_saturation(out, out_bits)

    sim_top = generate_sim_verilog(m, k, n, module_name,
                                   requant_shift=requant_shift, requant_bits=requant_bits)
    synth_top = generate_synth_verilog(m, k, n, module_name)

    lines = []
    lines.append("// Auto-generated by generate_catapult_rtl.py")
    lines.append(f"// Combined core: M={m}, K={k}, N={n}")
    lines.append("//   ifndef SYNTHESIS → behavioral simulation model (SCVerify)")
    lines.append("//   else            → structural synth wrapper  (HLS synthesis)")
    lines.append("")
    lines.append("`ifndef SYNTHESIS")
    lines.append("")
    for l in sim_top.splitlines():
        lines.append(l)
    lines.append("")
    lines.append("`else")
    lines.append("")
    for l in synth_top.splitlines():
        lines.append(l)
    lines.append("")
    lines.append("`endif")
    return _widen_output_saturation("\n".join(lines) + "\n", out_bits)


def _k_spatial_partitions(k, k_spatial):
    k_chunks = (k + 7) // 8
    if k_spatial < 1:
        raise ValueError("k_spatial must be >= 1")
    if k_spatial > k_chunks:
        raise ValueError(
            f"k_spatial={k_spatial} exceeds K_CHUNKS={k_chunks}; "
            "v1 requires at most one spatial grid per K chunk"
        )
    base = k_chunks // k_spatial
    extra = k_chunks % k_spatial
    out = []
    start = 0
    for p in range(k_spatial):
        count = base + (1 if p < extra else 0)
        out.append((start, start + count - 1))
        start += count
    return out


def generate_k_spatial_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                                     requant_shift=0, requant_bits=None,
                                     weight_rom=None, emit_rom=True):
    """Structural K-spatial core.

    Tensor-slice outputs (and therefore the K-chunk partials) are 16-bit, as in
    the current architecture. Only the post-accum32 step changes: the summed
    partials are requantised rather than clamped.

    The structural body instantiates multiple tensor-slice grids and exposes the
    intended partition/control topology. Functional RTL simulation for this
    experimental mode is provided by the combined core's behavioral branch.
    """
    # As in the sim branch: emit the function only when used, so a
    # requant_shift==0 core is byte-identical to the pre-requant generator.
    if requant_shift and requant_shift > 0:
        _w = requant_bits or 8
        _half = 1 << (requant_shift - 1)
        requant_fn = f"""
    // Requantise the summed cross-chunk result once, replacing the old clamp.
    function signed [15:0] requant_acc;
        input signed [31:0] x;
        reg signed [31:0] r;
        begin
            r = (x + 32'sd{_half}) >>> {requant_shift};
            requant_acc = $signed(r[{_w-1}:0]);
        end
    endfunction
"""
    else:
        requant_fn = ""
    if k_spatial == 1:
        return generate_synth_verilog(m, k, n, module_name,
                                      weight_rom=weight_rom, emit_rom=emit_rom)

    partitions = _k_spatial_partitions(k, k_spatial)
    # Weight-stationary: same contract as the chunked emitter — drop the external
    # b_cols port and register B from the shared ROM instead. emit_rom=False when the
    # combined core hoists one ROM above the `ifndef so both branches read it.
    ksp_ws = weight_rom is not None
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    bias_width = b_width
    a_chunk_width = a_width
    b_chunk_width = b_width
    c_width = grid_cols * 128
    input_beats = max(m, n)
    k_chunks = (k + 7) // 8
    full_k_spatial = k_spatial == k_chunks
    if full_k_spatial:
        # Narrow per-beat word: partition p carries one 64-bit tile (its K-chunk);
        # the wrapper routes it to the row/col tile by beat index, so no grid padding.
        a_width = 64 * k_chunks
        b_width = 64 * k_chunks
        a_chunk_width = 64
        b_chunk_width = 64
    total_output_rows = grid_rows * 8

    if ksp_ws:
        ksp_b_cols_port = ""
        ksp_b_cols_src = "w_rom_out"
        ksp_rom_block = _weight_rom_block(b_width, weight_rom) if emit_rom else ""
    else:
        ksp_b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        ksp_b_cols_src = "b_cols"
        ksp_rom_block = ""

    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]
    part_comments = "\n".join(
        f"// K_SPATIAL_PARTITION {p}: chunks {lo}..{hi}" for p, (lo, hi) in enumerate(partitions)
    )

    decls = []
    insts = []
    for p, (lo, hi) in enumerate(partitions):
        decls.append(f"    // Spatial grid {p}: contiguous K chunks {lo}..{hi}")
        if full_k_spatial:
            decls.append(f"    wire part{p}_active = 1'b1;")
            decls.append(f"    wire part{p}_first_chunk = 1'b1;")
            decls.append(f"    wire part{p}_last_chunk = 1'b1;")
            part_k_size = k - lo * 8 if hi == k_chunks - 1 else 8
            part_k_mask = tail_mask_hex(k, hi) if hi == k_chunks - 1 else 0xFF
        else:
            decls.append(f"    wire part{p}_active = (chunk_idx >= 16'd{lo}) && (chunk_idx <= 16'd{hi});")
            decls.append(f"    wire part{p}_first_chunk = (chunk_idx == 16'd{lo});")
            decls.append(f"    wire part{p}_last_chunk = (chunk_idx == 16'd{hi});")
            part_k_size = (k - hi * 8) or 8
            part_k_mask = tail_mask_hex(k, hi)
        decls.append(
            f"    wire [7:0] part{p}_k_size = part{p}_last_chunk ? "
            f"((16'd{hi} == K_CHUNKS - 1) ? 8'd{part_k_size} : 8'd8) : 8'd8;"
        )
        decls.append(
            f"    wire [7:0] part{p}_k_mask = part{p}_last_chunk ? "
            f"((16'd{hi} == K_CHUNKS - 1) ? {vm(part_k_mask)} : 8'hFF) : 8'hFF;"
        )
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = p * grid_rows * grid_cols + r * grid_cols + c
                if full_k_spatial:
                    # Narrow word: partition p carries one 64-bit tile at p*64; route
                    # it to row-tile r / col-tile c by beat index (beat_count/8 == tile).
                    a_expr = f"a_rows_q[{p}*{a_chunk_width} + 63:{p}*{a_chunk_width}]"
                    b_expr = f"b_cols_q[{p}*{b_chunk_width} + 63:{p}*{b_chunk_width}]"
                    a_route = f" && (beat_count >> 3 == {r})"
                    b_route = f" && (beat_count >> 3 == {c})"
                else:
                    a_expr = f"a_rows_q[{(r + 1) * 64 - 1}:{r * 64}]"
                    b_expr = f"b_cols_q[{(c + 1) * 64 - 1}:{c * 64}]"
                    a_route = ""
                    b_route = ""
                insts.append(f"""\
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_start && part{p}_first_chunk),
            .start_mat_mul(slice_start && part{p}_active),
            .done_mat_mul(done_mat_mul[{idx}]),
            .a_data((in_beat_active && part{p}_active && ({c} == 0){a_route}) ? {a_expr} : 64'b0),
            .b_data((in_beat_active && part{p}_active && ({r} == 0){b_route}) ? {b_expr} : 64'b0),
            .a_data_in(64'b0),
            .b_data_in(64'b0),
            .a_data_out(),
            .b_data_out(),
            .c_data_out(partial_c_p{p}_r{r}_c{c}),
            .c_data_available(partial_avail_p{p}_r{r}_c{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows(part{p}_k_mask),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op({{2'b00, op0_{r}}}),
            .preload(1'b0), .no_rounding(1'b0),
            .final_mat_mul_size(part{p}_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    partial_wires = []
    for p in range(k_spatial):
        for r in range(grid_rows):
            for c in range(grid_cols):
                partial_wires.append(f"    wire [127:0] partial_c_p{p}_r{r}_c{c};")
                partial_wires.append(f"    wire         partial_avail_p{p}_r{r}_c{c};")

    row_avail = []
    row_mux_cases = []
    for r in range(grid_rows):
        avail_terms = " & ".join(
            f"partial_avail_p{p}_r{r}_c{c}" for p in range(k_spatial) for c in range(grid_cols)
        )
        row_avail.append(f"    wire row_avail_{r} = {avail_terms};")
        row_mux_cases.append(f"        if (row_avail_{r}) begin")
        for c in range(grid_cols):
            for lane in range(8):
                terms = " + ".join(
                    f"$signed(partial_c_p{p}_r{r}_c{c}[{lane}*16 +: 16])" for p in range(k_spatial)
                )
                row_mux_cases.append(
                    f"            accum32 = {terms} + $signed({{ {{24{{bias_cols_q[{c}*64 + {lane}*8 + 7]}}}}, bias_cols_q[{c}*64 + {lane}*8 +: 8] }});"
                )
                # Slice outputs (and hence the partials) remain 16-bit -- unchanged.
                # The ONLY change is what happens after accum32: requantise the
                # cross-chunk sum instead of clamping it, matching requant_acc()
                # in the behavioural branch.
                _drain = ("requant_acc(accum32)" if requant_shift
                          else "sat_int8_to_i16(accum32)")
                row_mux_cases.append(
                    f"            row_mux[{c}*128 + {lane}*16 +: 16] = {_drain};"
                )
        row_mux_cases.append("        end")
    any_avail_expr = " | ".join(f"row_avail_{r}" for r in range(grid_rows))
    op0_lines = "\n".join(
        f"    wire op0_{r} = !((state == S_OUTPUT) && (cur_row_tile == 16'd{r}));"
        for r in range(grid_rows)
    )

    if full_k_spatial:
        wait_body = """\
                    if (all_slices_done) begin
                        state <= S_OUTPUT;
                    end"""
    else:
        wait_body = f"""\
                    if (all_slices_done) begin
                        if (chunk_idx + 16'd1 == K_CHUNKS) begin
                            state <= S_OUTPUT;
                        end else begin
                            chunk_idx <= chunk_idx + 16'd1;
                            beat_count <= 16'd0;
                            state <= S_RUN;
                        end
                    end"""

    return f"""\
// Auto-generated by generate_catapult_rtl.py
// Experimental K-spatial structural tensor-slice synth wrapper
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices, K_SPATIAL={k_spatial}
// WARNING: K-spatial partial outputs are INT16; correctness requires every partition partial sum to fit INT16.
{part_comments}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{ksp_b_cols_port}    input  wire [{bias_width-1}:0]   bias_cols,
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{ksp_rom_block}
    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer K_SPATIAL = {k_spatial};
    localparam integer TOTAL_OUT_ROWS = {total_output_rows};
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_WAIT=2'd2, S_OUTPUT=2'd3;

    reg [1:0] state;
    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] out_row_count;
    reg signed [31:0] accum32;
    reg [{c_width-1}:0] row_mux;

    // Input pipeline stage (same rationale as the chunked emitter): register
    // the whole bundle en-gated so the beat decode + partition gating muxes
    // start from local registers instead of chaining from the Catapult
    // wrapper into the tensor_slice input pins.
    reg [{a_width-1}:0] a_rows_q;
    reg [{b_width-1}:0] b_cols_q;
    reg [{bias_width-1}:0] bias_cols_q;
    reg preload_valid_q;
    reg in_valid_q;

    always @(posedge clk) begin
        if (rst) begin
            a_rows_q        <= {a_width}'d0;
            b_cols_q        <= {b_width}'d0;
            bias_cols_q     <= {bias_width}'d0;
            preload_valid_q <= 1'b0;
            in_valid_q      <= 1'b0;
        end else if (en) begin
            a_rows_q        <= a_rows;
            b_cols_q        <= {ksp_b_cols_src};
            bias_cols_q     <= bias_cols;
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = (state == S_RUN) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire [{k_spatial * grid_rows * grid_cols - 1}:0] done_mat_mul;
    wire all_slices_done = &done_mat_mul;

    // op[0] readout gate (op[0] == out_ctrl on the tensor slice): hold every
    // tile-row's completed result inside the tiles until S_OUTPUT, then
    // release one tile-row at a time, in row-major order. The released row's
    // partition partials are summed combinationally below; no parking
    // storage is needed and intermediate-chunk bursts never occur.
    wire [15:0] cur_row_tile = out_row_count >> 3;
{op0_lines}

{chr(10).join(decls)}

{chr(10).join(partial_wires)}

{chr(10).join(insts)}

{chr(10).join(row_avail)}

    wire any_avail = {any_avail_expr};

    function [15:0] sat_int8_to_i16;
        input signed [31:0] x;
        begin
            if (x > 32'sd127) sat_int8_to_i16 = 16'sd127;
            else if (x < -32'sd128) sat_int8_to_i16 = -16'sd128;
            else sat_int8_to_i16 = x[15:0];
        end
    endfunction
{requant_fn}
    always @(*) begin
        row_mux = {c_width}'d0;
        accum32 = 32'sd0;
{chr(10).join(row_mux_cases)}
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            out_row_count <= 16'd0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
        end else if (en) begin
            out_valid <= 1'b0;
            out_last <= 1'b0;
            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    out_row_count <= 16'd0;
                    if (preload_valid_q) state <= S_RUN;
                end
                S_RUN: begin
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS)
                            state <= S_WAIT;
                    end
                end
                S_WAIT: begin
{wait_body}
                end
                S_OUTPUT: begin
                    if (any_avail) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= (out_row_count + 16'd1 == TOTAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == TOTAL_OUT_ROWS)
                            state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
"""


def generate_k_spatial_sim_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                                   requant_shift=0, requant_bits=None,
                                   weight_rom=None, emit_rom=True):
    k_chunks = (k + 7) // 8
    sim = generate_sim_verilog(m, k, n, module_name, full_k_spatial=(k_spatial == k_chunks),
                               requant_shift=requant_shift, requant_bits=requant_bits,
                               weight_rom=weight_rom, emit_rom=emit_rom)
    if k_spatial == 1:
        return sim
    banner = (
        f"// Experimental K-spatial behavioral simulation model, K_SPATIAL={k_spatial}\n"
        "// WARNING: structural K-spatial partial outputs are INT16; simulation uses exact INT32 reference math.\n"
    )
    return banner + sim


def generate_k_spatial_combined_core_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1, out_bits=8,
                                            requant_shift=0, requant_bits=None, weight_rom=None):
    if k_spatial == 1:
        return generate_combined_core_verilog(m, k, n, module_name, out_bits=out_bits,
                                              requant_shift=requant_shift, requant_bits=requant_bits,
                                              weight_rom=weight_rom)
    _k_spatial_partitions(k, k_spatial)
    # Weight-stationary: each branch emits its own ROM. Unlike the chunked combined
    # core — which splits the two modules apart to hoist a single shared ROM above the
    # `ifndef — the K-spatial core keeps sim and synth as whole modules, so hoisting
    # would mean the same module surgery on an experimental structural body. Both ROMs
    # are built from the same weight_rom, so sim and synth weights stay identical by
    # construction; only one branch is ever compiled.
    sim_top = generate_k_spatial_sim_verilog(m, k, n, module_name, k_spatial,
                                            requant_shift=requant_shift, requant_bits=requant_bits,
                                            weight_rom=weight_rom)
    synth_top = generate_k_spatial_synth_verilog(m, k, n, module_name, k_spatial,
                                                requant_shift=requant_shift, requant_bits=requant_bits,
                                                weight_rom=weight_rom)

    lines = []
    lines.append("// Auto-generated by generate_catapult_rtl.py")
    lines.append(f"// Combined K-spatial core: M={m}, K={k}, N={n}, K_SPATIAL={k_spatial}")
    lines.append("//   ifndef SYNTHESIS -> behavioral simulation model")
    lines.append("//   else             -> experimental structural K-spatial wrapper")
    lines.append("// WARNING: INT16 partial overflow is possible in K-spatial structural mode.")
    lines.append("")
    lines.append("`ifndef SYNTHESIS")
    lines.append("")
    lines.extend(sim_top.splitlines())
    lines.append("")
    lines.append("`else")
    lines.append("")
    lines.extend(synth_top.splitlines())
    lines.append("")
    lines.append("`endif")
    return _widen_output_saturation("\n".join(lines) + "\n", out_bits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Verilog grid wrapper for tensor_slice_int8 modules")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--name", type=str, default="gemm_grid_wrapper")
    parser.add_argument("--output", type=str, default="gemm_grid.v")
    parser.add_argument("--sim-output", type=str, default=None)
    parser.add_argument("--synth-output", type=str, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Emit Verilog $display debug traces")
    args = parser.parse_args()

    if args.sim_output or args.synth_output:
        sim_output = args.sim_output or args.output.replace(".v", "_sim.v")
        synth_output = args.synth_output or args.output.replace(".v", "_synth.v")
        Path(sim_output).write_text(generate_sim_verilog(args.m, args.k, args.n, args.name))
        Path(synth_output).write_text(generate_synth_verilog(args.m, args.k, args.n, args.name, debug=args.debug))
        print(f"Generated {sim_output} and {synth_output}  (M={args.m}, K={args.k}, N={args.n})")
    else:
        content = generate_synth_verilog(args.m, args.k, args.n, args.name, debug=args.debug)
        Path(args.output).write_text(content)
        print(f"Generated {args.output}  (M={args.m}, K={args.k}, N={args.n})")
