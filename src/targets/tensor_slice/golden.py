#!/usr/bin/env python3
"""
Generate self-checking Verilog testbenches for tensor-slice GEMM RTL.

Drives the Catapult ``{core}`` with ``clk/rst/en``, ``a_rows/b_cols``,
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


def pack_a_k_spatial_narrow(A, t, pass_idx, m, k, k_spatial):
    """Narrow K-spatial A beat (64*k_spatial bits) for pass ``pass_idx``.

    Partition ``p`` carries K chunk ``pass_idx*k_spatial + p`` at bits
    ``[p*64 : p*64+64)``; ``k_spatial == k_chunks`` (``pass_idx`` always 0)
    reproduces today's full-K narrow beat exactly.
    """
    val = 0
    if t < m:
        for p in range(k_spatial):
            chunk = pass_idx * k_spatial + p
            k_start = chunk * 8
            for lane in range(8):
                kk = k_start + lane
                if kk < k:
                    byte = int(A[t, kk]) & 0xFF
                    val |= byte << (p * 64 + lane * 8)
    return val


def pack_b_k_spatial_narrow(B, col_idx, pass_idx, n, k, k_spatial):
    """Narrow K-spatial B beat (64*k_spatial bits) for pass ``pass_idx``.

    See :func:`pack_a_k_spatial_narrow` for the partition/chunk layout.
    """
    val = 0
    if col_idx < n:
        for p in range(k_spatial):
            chunk = pass_idx * k_spatial + p
            k_start = chunk * 8
            for lane in range(8):
                kk = k_start + lane
                if kk < k:
                    byte = int(B[kk, col_idx]) & 0xFF
                    val |= byte << (p * 64 + lane * 8)
    return val


def pack_a_full_k_spatial_narrow(A, t, m, k):
    """Narrow full-K spatial A beat (64*k_chunks bits): today's ``ks=k_chunks``
    endpoint of :func:`pack_a_k_spatial_narrow` (single pass)."""
    k_chunks = (k + 7) // 8
    return pack_a_k_spatial_narrow(A, t, 0, m, k, k_chunks)


def pack_b_full_k_spatial_narrow(B, col_idx, n, k):
    """Narrow full-K spatial B beat (64*k_chunks bits): today's ``ks=k_chunks``
    endpoint of :func:`pack_b_k_spatial_narrow` (single pass)."""
    k_chunks = (k + 7) // 8
    return pack_b_k_spatial_narrow(B, col_idx, 0, n, k, k_chunks)


def pack_bias(bias_codes, grid_cols, n):
    """Pack the per-column bias CODES (already at the 16-bit stage-2
    intermediate scale -- decision 4 in the tensor-slice-bias-in-rtl plan)
    into a single integer: one 16-bit signed lane per column, same per-tile
    layout as ``c_row`` (128 bits/tile) regardless of the output lane width.
    """
    val = 0
    for c in range(grid_cols):
        for col in range(8):
            actual_col = c * 8 + col
            code = int(bias_codes[actual_col]) & 0xFFFF if actual_col < n else 0
            val |= code << ((c * 8 + col) * 16)
    return val


def pack_c_row(C_out, r_tile, row_in_tile, grid_cols, m, n, out_width=8):
    """Pack one output row of C: ``out_width`` bits per lane, 8 lanes/column
    tile (``8*out_width`` bits/tile).

    ``C_out`` is the two-stage-requantised (round-half-up, wrap, no
    saturation) reference result.
    """
    mask = (1 << out_width) - 1
    actual_row = r_tile * 8 + row_in_tile
    val = 0
    for c in range(grid_cols):
        tile_val = 0
        for col in range(8):
            actual_col = c * 8 + col
            if actual_row < m and actual_col < n:
                lane = int(C_out[actual_row, actual_col]) & mask
            else:
                lane = 0
            tile_val |= lane << (col * out_width)
        val |= tile_val << (c * 8 * out_width)
    return val


def hex_literal(val, width_bytes):
    """Return a Verilog hex literal of the given byte width."""
    bits = width_bytes * 8
    return f"{bits}'h{val:0{width_bytes*2}x}"


def _wrap_signed(x, bits):
    """Wrap a Python int to a *bits*-wide two's-complement value (Verilog
    truncating assignment semantics: low *bits* bits, reinterpreted signed)."""
    m = 1 << bits
    x = int(x) % m
    if x >= (1 << (bits - 1)):
        x -= m
    return x


def _round_shift(x, s):
    """Round-half-up shift right by *s* (identity for s<=0). Python's ``>>``
    on an int is an arithmetic (floor) shift, matching Verilog's ``>>>`` on a
    signed value -- exactly the ``(x + 2**(s-1)) >>> s`` the RTL functions do."""
    if not s or s <= 0:
        return int(x)
    return (int(x) + (1 << (s - 1))) >> s


def two_stage_reference(A, B, bias_codes, s1=0, s2=0, out_width=8):
    """Reference matching stage1()/stage2() in rtl.py bit-for-bit: exact full-K
    sum (the sim branch folds in-slice and cross-chunk accumulation together --
    phase 1 does not structurally partition K, see the plan's "Sim vs synth
    branches" note), stage 1 (round-half-up shift by s1, wrap to 16), bias add
    (16-bit wrap), stage 2 (round-half-up shift by s2, wrap to out_width). No
    saturation anywhere. Returns an (m, n) int array of signed out_width codes.
    """
    m, k = A.shape
    _, n = B.shape
    raw = A.astype(np.int64) @ B.astype(np.int64)
    out = np.zeros((m, n), dtype=np.int64)
    for row in range(m):
        for col in range(n):
            p1 = _wrap_signed(_round_shift(int(raw[row, col]), s1), 16)
            bias = int(bias_codes[col]) if bias_codes is not None else 0
            biased = _wrap_signed(p1 + bias, 16)
            p2 = _round_shift(biased, s2)
            out[row, col] = _wrap_signed(p2, out_width)
    return out


def single_round_reference(A, B, bias_codes, s1=0, s2=0, out_width=8):
    """The ideal SINGLE-round reference decision 7 in the plan asks for: one
    round-half-up shift by ``s1 + s2`` applied directly to the exact sum, bias
    rescaled up to the raw (pre-shift) scale so it lands at the same point in
    the computation as ``two_stage_reference``'s bias add (just without the
    intermediate stage-1 round). Used only to MEASURE the S1>0 double-
    rounding delta -- never the pass/fail golden (that's always
    ``two_stage_reference``, which is what phase 1's folded sim RTL actually
    computes).
    """
    m, k = A.shape
    _, n = B.shape
    raw = A.astype(np.int64) @ B.astype(np.int64)
    total = int(s1) + int(s2)
    out = np.zeros((m, n), dtype=np.int64)
    for row in range(m):
        for col in range(n):
            bias = int(bias_codes[col]) if bias_codes is not None else 0
            biased_raw = int(raw[row, col]) + (bias << int(s1))
            out[row, col] = _wrap_signed(_round_shift(biased_raw, total), out_width)
    return out


def _random_matrices(m, k, n, seed, fixed_B=None, max_val=None, bias_codes=None,
                     s1=0, s2=0, out_width=8, has_bias=True):
    rng = np.random.default_rng(seed)
    # Default range keeps the *integer* matmul comfortably inside the
    # out_width range with s1=s2=0 (the common default-case regression), so
    # switching from the legacy saturate to round+wrap changes nothing for
    # these vectors -- no term ever approaches the wrap boundary.
    if max_val is None:
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
    A = rng.integers(-max_val, max_val + 1, size=(m, k), dtype=np.int8)
    # Weight-stationary: B is the baked const weight (same every vector), not random.
    if fixed_B is not None:
        B = np.asarray(fixed_B, dtype=np.int8)
    else:
        B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)
    # Bias is compile-time now (decision 4): it must be an explicit, FIXED
    # array supplied by the caller (the RTL core bakes one bias_rom for the
    # whole run, not a fresh one per vector) -- no more per-vector random
    # bias. `has_bias=False` (or no bias_codes) folds the add away (zeros).
    if has_bias and bias_codes is not None:
        biases = np.asarray(bias_codes, dtype=np.int64)
    else:
        biases = np.zeros((n,), dtype=np.int64)
    C_out = two_stage_reference(A, B, biases, s1=s1, s2=s2, out_width=out_width)
    return A, B, biases, C_out


def _gen_all_stimulus(m, k, n, num_vectors, base_seed, fixed_B=None,
                      bias_codes=None, s1=0, s2=0, out_width=8, has_bias=True):
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
        A, B, biases, C_out = _random_matrices(m, k, n, seed, fixed_B=fixed_B,
                                               bias_codes=bias_codes, s1=s1, s2=s2,
                                               out_width=out_width, has_bias=has_bias)

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
                golden.append(pack_c_row(C_out, rt, row, grid_cols, m, n, out_width=out_width))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols


def _gen_all_stimulus_catapult_k_spatial(m, k, n, num_vectors, base_seed, k_spatial, fixed_B=None,
                                         bias_codes=None, s1=0, s2=0, out_width=8, has_bias=True):
    """Generate widened K-spatial Catapult stimulus for ``k_spatial`` partitions.

    Each pass' beat carries one logical A row and one logical B column, with
    ``k_spatial`` K chunks packed spatially into a narrow A/B word;
    ``k_spatial == k_chunks`` (one pass) reproduces today's full-K spatial
    stimulus exactly.

    ``fixed_B`` pins B across every vector, as the chunked generator does. It is
    required for weight-stationary, where the DUT's baked ROM holds one B for the
    whole run: without it each vector would be checked against a fresh random B
    that the ROM never contained.
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)
    k_chunks = (k + 7) // 8
    passes = -(-k_chunks // k_spatial)

    all_a_stim, all_b_stim, all_bias, all_golden = [], [], [], []

    for v in range(num_vectors):
        seed = base_seed + v
        A, B, biases, C_out = _random_matrices(m, k, n, seed, fixed_B=fixed_B,
                                               bias_codes=bias_codes, s1=s1, s2=s2,
                                               out_width=out_width, has_bias=has_bias)

        a_stim, b_stim = [], []
        for pass_idx in range(passes):
            for t in range(input_beats):
                a_stim.append(pack_a_k_spatial_narrow(A, t, pass_idx, m, k, k_spatial))
                b_stim.append(pack_b_k_spatial_narrow(B, t, pass_idx, n, k, k_spatial))
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        all_bias.append(pack_bias(biases, grid_cols, n))

        golden = []
        for rt in range(grid_rows):
            for row in range(8):
                golden.append(pack_c_row(C_out, rt, row, grid_cols, m, n, out_width=out_width))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols



def _gen_catapult_tb(m, k, n, module_name, base_seed, all_a_stim, all_b_stim, all_bias, all_golden, timing=False, back2back=False, k_spatial=1, weights_in_core=False, out_width=8):
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
    passes = -(-k_chunks // k_spatial)
    total_input_beats = passes * input_beats
    a_bytes = grid_rows * 8
    b_bytes = grid_cols * 8
    if k_spatial > 1:
        # Narrow word: 64*k_spatial bits = 8*k_spatial bytes (one tile, k_spatial
        # K chunks per pass); the wrapper RTL routes the tile by beat index.
        a_bytes = 8 * k_spatial
        b_bytes = 8 * k_spatial
    # Bias is compile-time now (decision 4): no bias_cols port/width at all.
    c_bytes = grid_cols * out_width // 8
    aw = a_bytes * 8
    bw = b_bytes * 8
    cw = c_bytes * 8

    num_vectors = len(all_a_stim)
    total_out_rows = m
    b2b_flag = 1 if back2back else 0
    # Weight-stationary DUT has no external b_cols port (weights baked in ROM).
    dut_b_cols = "" if weights_in_core else ".b_cols(b_cols), "
    # Bias is compile-time now (decision 4): no bias_cols port at all.

    # Build memory initialization blocks
    a_init, b_init, bias_init, golden_init = [], [], [], []
    for v in range(num_vectors):
        for t in range(total_input_beats):
            a_init.append(f"        a_stim[{v}][{t}] = {hex_literal(all_a_stim[v][t], a_bytes)};")
            b_init.append(f"        b_stim[{v}][{t}] = {hex_literal(all_b_stim[v][t], b_bytes)};")
        for row in range(total_out_rows):
            golden_init.append(f"        golden[{v}][{row}] = {hex_literal(all_golden[v][row], c_bytes)};")

    t_first_out = '                if (out_row_idx == 0) $display("T:first_output=%0d", $realtime);' if timing else ""
    t_last_out  = '            $display("T:last_output=%0d", $realtime);' if timing else ""

    mode_tag = "full-k-spatial " if k_spatial > 1 else ""
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

            // Preload (bias is compile-time now, baked into the core)
            preload_valid <= 1;
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

            // Preload (bias is compile-time now, baked into the core)
            preload_valid <= 1;
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
    wire [{cw - 1}:0] c_row;
    wire out_valid;
    wire out_last;

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en),
        .a_rows(a_rows), {dut_b_cols}
        .preload_valid(preload_valid), .in_valid(in_valid),
        .c_row(c_row), .out_valid(out_valid), .out_last(out_last)
    );

    always #5 clk = ~clk;

    // Stimulus & golden memories (bias is compile-time now -- no bias_stim)
    reg [{aw - 1}:0] a_stim    [0:NV-1][0:TOTAL_INPUT_BEATS - 1];
    reg [{bw - 1}:0] b_stim    [0:NV-1][0:TOTAL_INPUT_BEATS - 1];
    reg [{cw - 1}:0] golden    [0:NV-1][0:TOTAL_ROWS - 1];

    initial begin
{chr(10).join(a_init)}
{chr(10).join(b_init)}
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


def _gen_all_stimulus_fold_n(core_m, k, core_n, n_passes, base_seed, k_spatial, fixed_b_full=None,
                             bias_codes=None, s1=0, s2=0, out_width=8, has_bias=False):
    """Group-aware fold-N stimulus: ONE shared A and ONE shared full-width B,
    sliced into ``n_passes`` groups of ``core_n`` columns each (group g is
    columns ``[g*core_n, (g+1)*core_n)`` of the full B) -- unlike the plain
    K/M-fold regressions (independent random vectors per frame), fold-N needs
    every frame fed the SAME A so frame g's checked golden is really
    ``A @ B[:, group g]``, exactly what the wrapper's multi-frame RUN loop
    composes back into one logical GEMM (see docs/wrapper_run_loop.md).

    Returns the same 6-tuple ``generate_tb``'s other stimulus builders do,
    plus the full B (``[k, n_passes*core_n]``) for the caller's own use (e.g.
    building a matching weight ROM for a weights_in_core RTL regression case).
    """
    grid_rows = (core_m + 7) // 8
    grid_cols = (core_n + 7) // 8  # per-group/per-frame core grid cols
    input_beats = max(core_m, core_n)
    k_chunks = (k + 7) // 8
    passes = -(-k_chunks // k_spatial)  # always 1 under fold-N (single K pass)

    rng = np.random.default_rng(base_seed)
    max_val = max(1, int((127 / max(k, 1)) ** 0.5))
    A = rng.integers(-max_val, max_val + 1, size=(core_m, k), dtype=np.int8)
    n_full = n_passes * core_n
    if fixed_b_full is not None:
        B_full = np.asarray(fixed_b_full, dtype=np.int8)
    else:
        B_full = rng.integers(-max_val, max_val + 1, size=(k, n_full), dtype=np.int8)

    all_a_stim, all_b_stim, all_bias, all_golden = [], [], [], []
    for g in range(n_passes):
        B_g = B_full[:, g * core_n:(g + 1) * core_n]
        # Bias is compile-time (decision 4), baked as ONE n_passes*core_n-wide
        # ROM (like the weight ROM): group g's REAL bias values live at
        # bias_codes[g*core_n:(g+1)*core_n] -- this can differ per group
        # (see the RTL's out_grp-indexed lookup, frozen per-frame so the
        # already-advanced live group counter is never read at emit time).
        if has_bias and bias_codes is not None:
            biases_g = np.asarray(bias_codes[g * core_n:(g + 1) * core_n], dtype=np.int64)
        else:
            biases_g = np.zeros((core_n,), dtype=np.int64)
        C_out = two_stage_reference(A, B_g, biases_g, s1=s1, s2=s2, out_width=out_width)

        a_stim, b_stim = [], []
        for pass_idx in range(passes):
            for t in range(input_beats):
                a_stim.append(pack_a_k_spatial_narrow(A, t, pass_idx, core_m, k, k_spatial))
                b_stim.append(pack_b_k_spatial_narrow(B_g, t, pass_idx, core_n, k, k_spatial))
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        all_bias.append(pack_bias(biases_g, grid_cols, core_n))

        golden = []
        for rt in range(grid_rows):
            for row in range(8):
                golden.append(pack_c_row(C_out, rt, row, grid_cols, core_m, core_n, out_width=out_width))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols, B_full


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_tb(m, k, n, module_name="gemm_grid_wrapper", seed=42, protocol="catapult",
                num_vectors=10, timing=False, back2back=False, k_spatial=1,
                weights_in_core=False, fixed_B=None, fold_n_groups=None,
                bias_codes=None, s1=0, s2=0, out_width=8, has_bias=None):
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
        k_spatial: Number of parallel K partitions (passes over K =
            ceil(k_chunks/k_spatial)); 1 is today's chunked layout,
            k_chunks is today's full-K layout, values in between are the
            general pass-sequenced K-spatial layout.

    Returns:
        Verilog source as a string.
    """
    if fold_n_groups and fold_n_groups > 1:
        # Fold-N (FoldAxis="n"): group-aware stimulus -- ONE shared A, ONE
        # shared full-width B sliced by group; overrides num_vectors with
        # fold_n_groups (one vector per group/frame, checked in frame order
        # by the existing back2back path).
        all_a, all_b, all_bias, all_golden, gr, gc, _ = _gen_all_stimulus_fold_n(
            m, k, n, fold_n_groups, seed, k_spatial, fixed_b_full=fixed_B,
            bias_codes=bias_codes, s1=s1, s2=s2, out_width=out_width,
            has_bias=bool(has_bias) if has_bias is not None else (bias_codes is not None))
    elif k_spatial > 1:
        all_a, all_b, all_bias, all_golden, gr, gc = _gen_all_stimulus_catapult_k_spatial(
            m, k, n, num_vectors, seed, k_spatial, fixed_B=fixed_B,
            bias_codes=bias_codes, s1=s1, s2=s2, out_width=out_width,
            has_bias=(bias_codes is not None) if has_bias is None else bool(has_bias))
    else:
        all_a, all_b, all_bias, all_golden, gr, gc = _gen_all_stimulus(
            m, k, n, num_vectors, seed, fixed_B=fixed_B,
            bias_codes=bias_codes, s1=s1, s2=s2, out_width=out_width,
            has_bias=(bias_codes is not None) if has_bias is None else bool(has_bias))
    return _gen_catapult_tb(m, k, n, module_name, seed, all_a, all_b, all_bias, all_golden, timing=timing, back2back=back2back, k_spatial=k_spatial, weights_in_core=weights_in_core, out_width=out_width)


def generate_tb_with_data(m, k, n, module_name, seed, protocol, A, B, biases, C_out, timing=False, out_width=8):
    """Single-vector testbench (backward compat — used for manual timing runs).

    ``C_out`` is the caller's own two-stage-requantised golden (see
    ``two_stage_reference``).
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
            cat_golden.append(pack_c_row(C_out, rt, row, grid_cols, m, n, out_width=out_width))
    return _gen_catapult_tb(m, k, n, module_name, seed, [a_stim], [b_stim], [bias_packed], [cat_golden], timing=timing, out_width=out_width)


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
