#!/usr/bin/env python3
"""
Generate self-checking Verilog testbenches for tensor-slice GEMM RTL.

Drives the Catapult ``{core}`` with ``clk/rst/en``, ``a_rows/b_cols/bias_cols``,
``preload_valid/in_valid``, and checks ``c_row/out_valid/out_last``.

Stimulus: random INT8 matrices A, B are generated in Python, multiplied with
int32 accumulation, bias added, saturated to int8/result type, and embedded as
Verilog literals for self-checking.
"""
import argparse
import numpy as np
from pathlib import Path


# ── Low-level helpers (shared) ─────────────────────────────────────────────────


def pack_a_row(A, t, grid_rows, m):
    """Pack one row t of A into grid_rows*64 bits (row/col contract).

    Row t goes to tile-row (t // 8), filling the 8 K-bytes of that row.
    Other tile-row lanes get 0.

    NOTE: For K > 8 this overflows 64-bit tile-row slots.  Use
    ``pack_a_chunk`` for the chunked K>8 protocol.
    """
    val = 0
    K = A.shape[1]  # inner dimension (K)
    tile_r = t // 8
    if t < m:
        row_word = 0
        for k in range(K):
            byte = int(A[t, k]) & 0xFF
            row_word |= byte << (k * 8)
        val = row_word << (tile_r * 64)
    return val


def pack_a_chunk(A, t, chunk, grid_rows, m, k):
    """Pack 8 K-bytes of row t starting at chunk*8.

    Each tile-row gets exactly 8 bytes (64 bits).  For K>8 the chunks
    are serialised across beats: chunk 0→k=0..7, chunk 1→k=8..15, etc.
    """
    val = 0
    tile_r = t // 8
    if t < m:
        row_word = 0
        k_start = chunk * 8
        k_end = min(k, (chunk + 1) * 8)
        for kk in range(k_start, k_end):
            byte = int(A[t, kk]) & 0xFF
            row_word |= byte << ((kk % 8) * 8)
        val = row_word << (tile_r * 64)
    return val


def pack_b_col(B, col_idx, grid_cols, n):
    """Pack one column of B into grid_cols*64 bits (row/col contract).

    Column col_idx goes to tile-col (col_idx // 8), filling the 8 K-rows.
    Other tile-col lanes get 0.

    NOTE: For K > 8 this overflows 64-bit tile-col slots.  Use
    ``pack_b_chunk`` for the chunked K>8 protocol.
    """
    val = 0
    K = B.shape[0]  # inner dimension (K)
    tile_c = col_idx // 8
    if col_idx < n:
        col_word = 0
        for k in range(K):
            byte = int(B[k, col_idx]) & 0xFF
            col_word |= byte << (k * 8)
        val = col_word << (tile_c * 64)
    return val


def pack_b_chunk(B, col_idx, chunk, grid_cols, n, k):
    """Pack 8 K-bytes of column col_idx starting at chunk*8."""
    val = 0
    tile_c = col_idx // 8
    if col_idx < n:
        col_word = 0
        k_start = chunk * 8
        k_end = min(k, (chunk + 1) * 8)
        for kk in range(k_start, k_end):
            byte = int(B[kk, col_idx]) & 0xFF
            col_word |= byte << ((kk % 8) * 8)
        val = col_word << (tile_c * 64)
    return val


def pack_a_full_k_spatial(A, t, grid_rows, m, k):
    """Pack all K chunks of one A row into a widened full-K spatial beat."""
    val = 0
    k_chunks = (k + 7) // 8
    chunk_width = grid_rows * 64
    for chunk in range(k_chunks):
        val |= pack_a_chunk(A, t, chunk, grid_rows, m, k) << (chunk * chunk_width)
    return val


def pack_b_full_k_spatial(B, col_idx, grid_cols, n, k):
    """Pack all K chunks of one B column into a widened full-K spatial beat."""
    val = 0
    k_chunks = (k + 7) // 8
    chunk_width = grid_cols * 64
    for chunk in range(k_chunks):
        val |= pack_b_chunk(B, col_idx, chunk, grid_cols, n, k) << (chunk * chunk_width)
    return val


def pack_a_full_k_spatial_narrow(A, t, m, k):
    """Narrow full-K spatial A beat (64*k_chunks bits): one row tile at position 0,
    K chunk c at bits [c*64 : c*64+64], no tile-row offset."""
    val = 0
    if t < m:
        for kk in range(k):
            byte = int(A[t, kk]) & 0xFF
            val |= byte << ((kk // 8) * 64 + (kk % 8) * 8)
    return val


def pack_b_full_k_spatial_narrow(B, col_idx, n, k):
    """Narrow full-K spatial B beat (64*k_chunks bits): one col tile at position 0,
    K chunk c at bits [c*64 : c*64+64], no tile-col offset."""
    val = 0
    if col_idx < n:
        for kk in range(k):
            byte = int(B[kk, col_idx]) & 0xFF
            val |= byte << ((kk // 8) * 64 + (kk % 8) * 8)
    return val


def pack_bias(biases, grid_cols, n):
    """Pack bias array into a single integer (same layout as B)."""
    val = 0
    for c in range(grid_cols):
        for col in range(8):
            actual_col = c * 8 + col
            if actual_col < n:
                byte = int(biases[actual_col]) & 0xFF
            else:
                byte = 0
            val |= byte << ((c * 8 + col) * 8)
    return val


def pack_c_row(C_sat, r_tile, row_in_tile, grid_cols, m, n, protocol="catapult"):
    """Pack one output row of C: 128 bits per column tile — eight signed INT16
    result lanes.

    ``C_sat`` is the Catapult matmul + bias, saturated to the result type. The
    ``protocol`` argument is accepted for call-site compatibility but unused.
    """
    del protocol
    actual_row = r_tile * 8 + row_in_tile
    val = 0
    for c in range(grid_cols):
        tile_val = 0
        for col in range(8):
            actual_col = c * 8 + col
            if actual_row < m and actual_col < n:
                lane = int(C_sat[actual_row, actual_col]) & 0xFFFF
            else:
                lane = 0
            tile_val |= lane << (col * 16)
        val |= tile_val << (c * 128)
    return val


def hex_literal(val, width_bytes):
    """Return a Verilog hex literal of the given byte width."""
    bits = width_bytes * 8
    return f"{bits}'h{val:0{width_bytes*2}x}"


def _random_matrices(m, k, n, seed, fixed_B=None, max_val=None):
    rng = np.random.default_rng(seed)
    # Default range keeps the *integer* matmul within int8 for the Catapult
    # int8-saturated golden.
    if max_val is None:
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
    A = rng.integers(-max_val, max_val + 1, size=(m, k), dtype=np.int8)
    # Weight-stationary: B is the baked const weight (same every vector), not random.
    if fixed_B is not None:
        B = np.asarray(fixed_B, dtype=np.int8)
    else:
        B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)
    biases = rng.integers(-8, 8, size=(n,), dtype=np.int8)
    C_ref = A.astype(np.int32) @ B.astype(np.int32) + biases.astype(np.int32)
    C_sat = np.clip(C_ref, -128, 127).astype(np.int8)
    return A, B, biases, C_sat


def _gen_all_stimulus(m, k, n, num_vectors, base_seed, fixed_B=None):
    """Generate stimulus and golden data for *num_vectors* random tests.

    Row/col contract: one A row + one B column per beat.
    K ≤ 8: max(M,N) beats total.
    K > 8: k_chunks × max(M,N) beats — each chunk carries 8 K-bytes.
    Returns golden in Catapult format: grid_rows*8 rows (zero-padded beyond m).
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one row + one col per cycle
    k_chunks = (k + 7) // 8

    all_a_stim, all_b_stim, all_bias, all_golden = [], [], [], []

    for v in range(num_vectors):
        seed = base_seed + v
        A, B, biases, C_sat = _random_matrices(m, k, n, seed, fixed_B=fixed_B)

        a_stim, b_stim = [], []
        for chunk in range(k_chunks):
            for t in range(input_beats):
                a_stim.append(pack_a_chunk(A, t, chunk, grid_rows, m, k))
                b_stim.append(pack_b_chunk(B, t, chunk, grid_cols, n, k))
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        all_bias.append(pack_bias(biases, grid_cols, n))

        golden = []
        for rt in range(grid_rows):
            for row in range(8):
                golden.append(pack_c_row(C_sat, rt, row, grid_cols, m, n, "catapult"))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols


def _gen_all_stimulus_catapult_full_k_spatial(m, k, n, num_vectors, base_seed, fixed_B=None):
    """Generate widened full-K spatial Catapult stimulus.

    Each beat carries one logical A row and one logical B column, with all
    K chunks packed spatially into widened A/B words.

    ``fixed_B`` pins B across every vector, as the chunked generator does. It is
    required for weight-stationary, where the DUT's baked ROM holds one B for the
    whole run: without it each vector would be checked against a fresh random B
    that the ROM never contained.
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)

    all_a_stim, all_b_stim, all_bias, all_golden = [], [], [], []

    for v in range(num_vectors):
        seed = base_seed + v
        A, B, biases, C_sat = _random_matrices(m, k, n, seed, fixed_B=fixed_B)

        a_stim, b_stim = [], []
        for t in range(input_beats):
            a_stim.append(pack_a_full_k_spatial_narrow(A, t, m, k))
            b_stim.append(pack_b_full_k_spatial_narrow(B, t, n, k))
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        all_bias.append(pack_bias(biases, grid_cols, n))

        golden = []
        for rt in range(grid_rows):
            for row in range(8):
                golden.append(pack_c_row(C_sat, rt, row, grid_cols, m, n, "catapult"))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols



def _gen_catapult_tb(m, k, n, module_name, base_seed, all_a_stim, all_b_stim, all_bias, all_golden, timing=False, back2back=False, full_k_spatial=False, weights_in_core=False):
    """Generate a multi-vector Catapult testbench.

    back2back=False (default): Reset between every vector; check each vector
        before feeding the next.  Even vectors get reset, odd vectors don't.

    back2back=True: Reset only on the first vector; feed all vectors in quick
        succession; outputs are checked sequentially by out_last count.
        Tests double-buffer pipelining.
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one A row + one B col per cycle
    k_chunks = (k + 7) // 8
    total_input_beats = input_beats if full_k_spatial else k_chunks * input_beats
    a_bytes = grid_rows * 8
    b_bytes = grid_cols * 8
    bias_bytes = b_bytes
    if full_k_spatial:
        # Narrow word: 64*k_chunks bits = 8*k_chunks bytes (one tile, all K chunks);
        # the wrapper RTL routes the tile by beat index.  Bias stays grid_cols*64.
        a_bytes = 8 * k_chunks
        b_bytes = 8 * k_chunks
    c_bytes = grid_cols * 16
    aw = a_bytes * 8
    bw = b_bytes * 8
    biasw = bias_bytes * 8
    cw = c_bytes * 8

    num_vectors = len(all_a_stim)
    total_out_rows = m
    b2b_flag = 1 if back2back else 0
    # Weight-stationary DUT has no external b_cols port (weights baked in ROM).
    dut_b_cols = "" if weights_in_core else ".b_cols(b_cols), "

    # Build memory initialization blocks
    a_init, b_init, bias_init, golden_init = [], [], [], []
    for v in range(num_vectors):
        for t in range(total_input_beats):
            a_init.append(f"        a_stim[{v}][{t}] = {hex_literal(all_a_stim[v][t], a_bytes)};")
            b_init.append(f"        b_stim[{v}][{t}] = {hex_literal(all_b_stim[v][t], b_bytes)};")
        bias_init.append(f"        bias_stim[{v}] = {hex_literal(all_bias[v], bias_bytes)};")
        for row in range(total_out_rows):
            golden_init.append(f"        golden[{v}][{row}] = {hex_literal(all_golden[v][row], c_bytes)};")

    t_first_out = '                if (out_row_idx == 0) $display("T:first_output=%0d", $realtime);' if timing else ""
    t_last_out  = '            $display("T:last_output=%0d", $realtime);' if timing else ""

    mode_tag = "full-k-spatial " if full_k_spatial else ""
    mode_tag += "back2back" if back2back else "sequential"
    if back2back:
        # ── Back-to-back initial block ──
        init_block = f"""
    initial begin
        $display("=== Catapult TB {m}x{k}x{n}  ({num_vectors} vectors) ===");
        pass_count = 0; fail_count = 0;
        out_last_count = 0;
        out_row_idx = 0;

        $display("MODE back2back");

        for (vec_idx = 0; vec_idx < NV; vec_idx = vec_idx + 1) begin
            $display("--- Vector %0d (seed %0d) ---", vec_idx, {base_seed} + vec_idx);

            // Reset only on first vector
            if (vec_idx == 0) begin
                preload_valid <= 0; in_valid <= 0;
                rst <= 1;
                @(posedge clk);
                rst <= 0; en <= 1;
                @(posedge clk);
            end

            // Preload bias
            preload_valid <= 1;
            bias_cols <= bias_stim[vec_idx];
            @(posedge clk);
            preload_valid <= 0;

            // Feed data
            in_valid <= 1;
            a_rows <= a_stim[vec_idx][0];
            b_cols <= b_stim[vec_idx][0];
            @(posedge clk);
            for (t = 1; t < TOTAL_INPUT_BEATS; t = t + 1) begin
                a_rows <= a_stim[vec_idx][t];
                b_cols <= b_stim[vec_idx][t];
                @(posedge clk);
            end
            in_valid <= 0;
            a_rows <= 0;
            b_cols <= 0;

            // True back-to-back cadence: the frame-slot sim model pipelines
            // frames (feed of frame t+1 overlaps compute/drain of frame t), so
            // the next vector's preload follows immediately — the only gap is
            // the preload step itself, giving a sustained frame II of
            // TOTAL_INPUT_BEATS+1 cycles.
        end

        // Wait for all out_last events
        cycle_ctr = 0;
        while (out_last_count < NV && cycle_ctr < 2000) begin
            @(posedge clk);
            cycle_ctr = cycle_ctr + 1;
        end
        if (out_last_count != NV) begin
            $display("TIMEOUT: only %0d out_last events, expected %0d", out_last_count, NV);
            fail_count = fail_count + 1;
        end
        @(posedge clk);

        if (fail_count == 0) $display("ALL_PASS  (%0d vectors)", NV);
        else $display("FAILURES=%0d", fail_count);
        $finish;
    end
"""
        # ── Back-to-back output checking always block ──
        always_block = f"""
    always @(posedge clk) begin
        if (out_last) begin
{t_last_out}
            if (out_row_idx != TOTAL_ROWS - 1)
                $display("FAIL vec %0d out_last row %0d", out_last_count, out_row_idx);
            out_last_count <= out_last_count + 1;
        end
        if (out_valid) begin
            if (out_row_idx < TOTAL_ROWS) begin
{t_first_out}
                if (c_row !== golden[out_last_count][out_row_idx]) begin
                    $display("FAIL vec %0d row %0d: got %h expected %h", out_last_count, out_row_idx, c_row, golden[out_last_count][out_row_idx]);
                    fail_count = fail_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                end
                out_row_idx <= out_row_idx + 1;
            end else begin
                $display("FAIL vec %0d extra row %0d", out_last_count, out_row_idx);
                fail_count = fail_count + 1;
            end
        end
        if (out_last) out_row_idx <= 0;
    end
"""
    else:
        # ── Sequential initial block ──
        init_block = f"""
    initial begin
        $display("=== Catapult TB {m}x{k}x{n}  ({num_vectors} vectors) ===");
        pass_count = 0; fail_count = 0;

        $display("MODE sequential");

        for (vec_idx = 0; vec_idx < NV; vec_idx = vec_idx + 1) begin
            $display("--- Vector %0d (seed %0d) ---", vec_idx, {base_seed} + vec_idx);
            out_row_idx = 0;

            // Even vectors: full reset.  Odd vectors: back-to-back without reset.
            if (vec_idx == 0 || (vec_idx % 2) == 0) begin
                preload_valid <= 0; in_valid <= 0;
                rst <= 1;
                @(posedge clk);
                rst <= 0; en <= 1;
                @(posedge clk);
            end

            // Preload bias
            preload_valid <= 1;
            bias_cols <= bias_stim[vec_idx];
            @(posedge clk);
            preload_valid <= 0;

            // Feed data
            in_valid <= 1;
            a_rows <= a_stim[vec_idx][0];
            b_cols <= b_stim[vec_idx][0];
            @(posedge clk);
            for (t = 1; t < TOTAL_INPUT_BEATS; t = t + 1) begin
                a_rows <= a_stim[vec_idx][t];
                b_cols <= b_stim[vec_idx][t];
                @(posedge clk);
            end
            in_valid <= 0;
            a_rows <= 0;
            b_cols <= 0;

            // Wait for out_last with cycle-count timeout
            cycle_ctr = 0;
            while (!out_last && cycle_ctr < 2000) begin
                @(posedge clk);
                cycle_ctr = cycle_ctr + 1;
            end
            if (cycle_ctr >= 2000) begin
                $display("TIMEOUT vec %0d", vec_idx);
                fail_count = fail_count + 1;
            end
            @(posedge clk);

            if (out_row_idx != TOTAL_ROWS) begin
                $display("FAIL vec %0d row count: got %0d expected %0d", vec_idx, out_row_idx, TOTAL_ROWS);
                fail_count = fail_count + 1;
            end
        end

        if (fail_count == 0) $display("ALL_PASS  (%0d vectors)", NV);
        else $display("FAILURES=%0d", fail_count);
        $finish;
    end
"""
        # ── Sequential output checking always block ──
        always_block = f"""
    always @(posedge clk) begin
        if (out_last) begin
{t_last_out}
            out_last_count = out_last_count + 1;
            if (out_row_idx != TOTAL_ROWS - 1) begin
                $display("FAIL vec %0d out_last row %0d", vec_idx, out_row_idx);
                fail_count = fail_count + 1;
            end
        end
        if (out_valid && out_row_idx < TOTAL_ROWS) begin
{t_first_out}
            if (c_row !== golden[vec_idx][out_row_idx]) begin
                $display("FAIL vec %0d row %0d: got %h expected %h", vec_idx, out_row_idx, c_row, golden[vec_idx][out_row_idx]);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
            out_row_idx <= out_row_idx + 1;
        end else if (out_valid) begin
            $display("FAIL vec %0d extra row %0d", vec_idx, out_row_idx);
            fail_count = fail_count + 1;
        end
    end
"""

    return f"""\
`timescale 1ns/1ps
// Auto-generated Catapult-core multi-vector testbench  (row/col streaming)
// M={m}, K={k}, N={n}  |  base_seed={base_seed}  |  {num_vectors} vectors
// mode: {mode_tag}

module tb_catapult_{m}x{k}x{n};

    localparam NV = {num_vectors};
    localparam INPUT_BEATS = {input_beats};
    localparam K_CHUNKS = {k_chunks};
    localparam TOTAL_INPUT_BEATS = {total_input_beats};
    localparam TOTAL_ROWS = {total_out_rows};

    reg  clk = 0;
    reg  rst = 1;
    reg  en  = 1;
    reg  preload_valid = 0;
    reg  in_valid = 0;
    reg  [{aw - 1}:0] a_rows = 0;
    reg  [{bw - 1}:0] b_cols = 0;
    reg  [{biasw - 1}:0] bias_cols = 0;
    wire [{cw - 1}:0] c_row;
    wire out_valid;
    wire out_last;

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en),
        .a_rows(a_rows), {dut_b_cols}.bias_cols(bias_cols),
        .preload_valid(preload_valid), .in_valid(in_valid),
        .c_row(c_row), .out_valid(out_valid), .out_last(out_last)
    );

    always #5 clk = ~clk;

    // Stimulus & golden memories
    reg [{aw - 1}:0] a_stim    [0:NV-1][0:TOTAL_INPUT_BEATS - 1];
    reg [{bw - 1}:0] b_stim    [0:NV-1][0:TOTAL_INPUT_BEATS - 1];
    reg [{biasw - 1}:0] bias_stim [0:NV-1];
    reg [{cw - 1}:0] golden    [0:NV-1][0:TOTAL_ROWS - 1];

    initial begin
{chr(10).join(a_init)}
{chr(10).join(b_init)}
{chr(10).join(bias_init)}
{chr(10).join(golden_init)}
    end

    integer vec_idx, out_row_idx;
    integer t;
    integer pass_count, fail_count;
    integer cycle_ctr, out_last_count;
{init_block}

{always_block}

endmodule
"""


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_tb(m, k, n, module_name="gemm_grid_wrapper", seed=42, protocol="catapult",
                num_vectors=10, timing=False, back2back=False, full_k_spatial=False,
                weights_in_core=False, fixed_B=None):
    """Generate a self-checking multi-vector Verilog testbench.

    Args:
        m, k, n: GEMM dimensions.
        module_name: Name of the DUT module to instantiate.
        seed: Base random seed (vectors use seed, seed+1, ..., seed+N-1).
        protocol: ``"catapult"`` (the only supported protocol).
        num_vectors: Number of random test vectors (default 10).
        timing: If True, emit ``$display`` with ``$realtime`` at key events.
        back2back: If True, feed vectors in quick succession without waiting
            for output before starting the next vector (tests double-buffer
            pipelining). Outputs are checked sequentially by out_last count.

    Returns:
        Verilog source as a string.
    """
    if full_k_spatial:
        all_a, all_b, all_bias, all_golden, gr, gc = _gen_all_stimulus_catapult_full_k_spatial(
            m, k, n, num_vectors, seed, fixed_B=fixed_B)
    else:
        all_a, all_b, all_bias, all_golden, gr, gc = _gen_all_stimulus(
            m, k, n, num_vectors, seed, fixed_B=fixed_B)
    return _gen_catapult_tb(m, k, n, module_name, seed, all_a, all_b, all_bias, all_golden, timing=timing, back2back=back2back, full_k_spatial=full_k_spatial, weights_in_core=weights_in_core)


def generate_tb_with_data(m, k, n, module_name, seed, protocol, A, B, biases, C_sat, timing=False):
    """Single-vector testbench (backward compat — used for manual timing runs).

    ``C_sat`` bakes bias into the Catapult int8/result-type saturated golden.
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one row + one col per cycle
    k_chunks = (k + 7) // 8
    a_stim, b_stim = [], []
    for chunk in range(k_chunks):
        for t in range(input_beats):
            a_stim.append(pack_a_chunk(A, t, chunk, grid_rows, m, k))
            b_stim.append(pack_b_chunk(B, t, chunk, grid_cols, n, k))
    # Catapult: need (grid_rows*8) rows of golden
    bias_packed = pack_bias(biases, grid_cols, n)
    total_out_rows = m
    cat_golden = []
    for rt in range(grid_rows):
        for row in range(8):
            cat_golden.append(pack_c_row(C_sat, rt, row, grid_cols, m, n, "catapult"))
    return _gen_catapult_tb(m, k, n, module_name, seed, [a_stim], [b_stim], [bias_packed], [cat_golden], timing=timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate self-checking Verilog testbench for tensor-slice GEMM"
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--name", type=str, default="gemm_grid_wrapper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=("catapult",), default="catapult")
    parser.add_argument("--output", type=str, default="tb_gemm.v")
    args = parser.parse_args()

    content = generate_tb(args.m, args.k, args.n, args.name, args.seed, args.protocol)
    Path(args.output).write_text(content)
    print(f"Generated {args.output}  ({args.protocol}, M={args.m}, K={args.k}, N={args.n}, seed={args.seed})")
