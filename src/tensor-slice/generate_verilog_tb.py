#!/usr/bin/env python3
"""
Generate self-checking Verilog testbenches for tensor-slice GEMM RTL.

Supports two protocols via ``--protocol catapult|vitis``:

- **catapult** (default): Drives ``{core}`` with ``clk/rst/en``,
  ``a_rows/b_cols/bias_cols``, ``preload_valid/in_valid``, and checks
  ``c_row/out_valid/out_last``.

- **vitis**: Drives ``{wrapper}`` with ``ap_clk/ap_rst/ap_ce``,
  AXI-stream FIFO handshake (``a_tdata/a_tvalid/a_tready``,
  ``bias_tdata/…``, ``b_tdata/…``), and reads ``c_tdata/c_tvalid``.

Both variants use the same golden reference: random INT8 matrices A, B are
generated in Python, multiplied with int32 accumulation, saturated to int8,
and embedded as Verilog literals for self-checking.
"""
import argparse
import numpy as np
from pathlib import Path


# ── Low-level helpers (shared) ─────────────────────────────────────────────────


def pack_a_row(A, t, grid_rows, m):
    """Pack one row t of A into grid_rows*64 bits (row/col contract).

    Row t goes to tile-row (t // 8), filling the 8 K-bytes of that row.
    Other tile-row lanes get 0.
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


def pack_b_col(B, col_idx, grid_cols, n):
    """Pack one column of B into grid_cols*64 bits (row/col contract).

    Column col_idx goes to tile-col (col_idx // 8), filling the 8 K-rows.
    Other tile-col lanes get 0.
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
    """Pack one output row of C.

    Catapult grid: 128 bits per column tile — data at byte offsets 0,8,16…
    within each tile (lower 64 bits valid, upper 64 zero-padded).
    Vitis wrapper: 64 bits per column tile — 8 int8 lanes packed adjacently.
    """
    actual_row = r_tile * 8 + row_in_tile
    val = 0
    for c in range(grid_cols):
        tile_val = 0
        for col in range(8):
            actual_col = c * 8 + col
            if actual_row < m and actual_col < n:
                byte = int(C_sat[actual_row, actual_col]) & 0xFF
            else:
                byte = 0
            if protocol == "vitis":
                tile_val |= byte << (col * 8)
            else:
                tile_val |= byte << (col * 8)
        if protocol == "vitis":
            val |= tile_val << (c * 64)
        else:
            val |= tile_val << (c * 128)
    return val


def hex_literal(val, width_bytes):
    """Return a Verilog hex literal of the given byte width."""
    bits = width_bytes * 8
    return f"{bits}'h{val:0{width_bytes*2}x}"


def _random_matrices(m, k, n, seed):
    rng = np.random.default_rng(seed)
    max_val = max(1, int((127 / max(k, 1)) ** 0.5))
    A = rng.integers(-max_val, max_val + 1, size=(m, k), dtype=np.int8)
    B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)
    biases = rng.integers(-8, 8, size=(n,), dtype=np.int8)
    C_ref = A.astype(np.int32) @ B.astype(np.int32) + biases.astype(np.int32)
    C_sat = np.clip(C_ref, -128, 127).astype(np.int8)
    return A, B, biases, C_sat


def _gen_all_stimulus(m, k, n, num_vectors, base_seed):
    """Generate stimulus and golden data for *num_vectors* random tests.

    Row/col contract: one A row + one B column per beat, max(M,N) beats total.
    Returns golden in Catapult format: grid_rows*8 rows (zero-padded beyond m).
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one row + one col per cycle

    all_a_stim, all_b_stim, all_bias, all_golden = [], [], [], []

    for v in range(num_vectors):
        seed = base_seed + v
        A, B, biases, C_sat = _random_matrices(m, k, n, seed)

        a_stim = [pack_a_row(A, t, grid_rows, m) for t in range(input_beats)]
        b_stim = [pack_b_col(B, t, grid_cols, n) for t in range(input_beats)]
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        all_bias.append(pack_bias(biases, grid_cols, n))

        golden = []
        for rt in range(grid_rows):
            for row in range(8):
                golden.append(pack_c_row(C_sat, rt, row, grid_cols, m, n, "catapult"))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols


def _gen_all_stimulus_vitis(m, k, n, num_vectors, base_seed):
    """Same as _gen_all_stimulus but golden uses Vitis packing (gc*64 bits/row, grid_rows*8 rows)."""
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one row + one col per cycle

    all_a_stim, all_b_stim, all_bias, all_golden = [], [], [], []

    for v in range(num_vectors):
        seed = base_seed + v
        A, B, biases, C_sat = _random_matrices(m, k, n, seed)

        a_stim = [pack_a_row(A, t, grid_rows, m) for t in range(input_beats)]
        b_stim = [pack_b_col(B, t, grid_cols, n) for t in range(input_beats)]
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        all_bias.append(pack_bias(biases, grid_cols, n))

        golden = []
        for rt in range(grid_rows):
            for row in range(8):
                golden.append(pack_c_row(C_sat, rt, row, grid_cols, m, n, "vitis"))
        all_golden.append(golden)

    return all_a_stim, all_b_stim, all_bias, all_golden, grid_rows, grid_cols


# ── Catapult testbench ─────────────────────────────────────────────────────────


def _gen_catapult_tb(m, k, n, module_name, base_seed, all_a_stim, all_b_stim, all_bias, all_golden, timing=False):
    """Generate a multi-vector Catapult testbench with alternating reset/no-reset.

    Even vectors: full reset (catches init-path bugs).
    Odd vectors:  back-to-back without reset (catches state-leakage bugs).
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one A row + one B col per cycle
    a_bytes = grid_rows * 8
    b_bytes = grid_cols * 8
    c_bytes = grid_cols * 16
    aw = a_bytes * 8
    bw = b_bytes * 8
    cw = c_bytes * 8

    num_vectors = len(all_a_stim)
    total_out_rows = grid_rows * 8

    # Build memory initialization blocks
    a_init, b_init, bias_init, golden_init = [], [], [], []
    for v in range(num_vectors):
        for t in range(input_beats):
            a_init.append(f"        a_stim[{v}][{t}] = {hex_literal(all_a_stim[v][t], a_bytes)};")
            b_init.append(f"        b_stim[{v}][{t}] = {hex_literal(all_b_stim[v][t], b_bytes)};")
        bias_init.append(f"        bias_stim[{v}] = {hex_literal(all_bias[v], b_bytes)};")
        for row in range(total_out_rows):
            golden_init.append(f"        golden[{v}][{row}] = {hex_literal(all_golden[v][row], c_bytes)};")

    t_first_out = '                if (out_row_idx == 0) $display("T:first_output=%0d", $realtime);' if timing else ""
    t_last_out  = '            $display("T:last_output=%0d", $realtime);' if timing else ""

    return f"""\
`timescale 1ns/1ps
// Auto-generated Catapult-core multi-vector testbench  (row/col streaming)
// M={m}, K={k}, N={n}  |  base_seed={base_seed}  |  {num_vectors} vectors
// All vectors: brief rst pulse between runs (pe_reset required)

module tb_catapult_{m}x{k}x{n};

    localparam NV = {num_vectors};
    localparam INPUT_BEATS = {input_beats};
    localparam TOTAL_ROWS = {total_out_rows};

    reg  clk = 0;
    reg  rst = 1;
    reg  en  = 1;
    reg  preload_valid = 0;
    reg  in_valid = 0;
    reg  [{aw - 1}:0] a_rows = 0;
    reg  [{bw - 1}:0] b_cols = 0;
    reg  [{bw - 1}:0] bias_cols = 0;
    wire [{cw - 1}:0] c_row;
    wire out_valid;
    wire out_last;

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en),
        .a_rows(a_rows), .b_cols(b_cols), .bias_cols(bias_cols),
        .preload_valid(preload_valid), .in_valid(in_valid),
        .c_row(c_row), .out_valid(out_valid), .out_last(out_last)
    );

    always #5 clk = ~clk;

    // Stimulus & golden memories
    reg [{aw - 1}:0] a_stim    [0:NV-1][0:INPUT_BEATS - 1];
    reg [{bw - 1}:0] b_stim    [0:NV-1][0:INPUT_BEATS - 1];
    reg [{bw - 1}:0] bias_stim [0:NV-1];
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

    initial begin
        $display("=== Catapult TB {m}x{k}x{n}  ({num_vectors} vectors) ===");
        pass_count = 0; fail_count = 0;

        for (vec_idx = 0; vec_idx < NV; vec_idx = vec_idx + 1) begin
            $display("--- Vector %0d (seed %0d) ---", vec_idx, {base_seed} + vec_idx);
            out_row_idx = 0; out_last_count = 0;

            // Reset between vectors (Catapult requires pe_reset)
            preload_valid <= 0; in_valid <= 0;
            rst <= 1;
            @(posedge clk);
            rst <= 0;
            en  <= 1;
            @(posedge clk);

            // Preload bias (1 cycle — bias loads via primary b_data port on all tiles)
            preload_valid <= 1;
            bias_cols <= bias_stim[vec_idx];
            @(posedge clk);
            preload_valid <= 0;

            // Feed INPUT_BEATS row/col pairs — set in_valid + first beat together
            in_valid <= 1;
            a_rows <= a_stim[vec_idx][0];
            b_cols <= b_stim[vec_idx][0];
            @(posedge clk);
            for (t = 1; t < INPUT_BEATS; t = t + 1) begin
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
            @(posedge clk);  // one more cycle for output pipeline

            if (out_row_idx != TOTAL_ROWS) begin
                $display("FAIL vec %0d row count: got %0d expected %0d", vec_idx, out_row_idx, TOTAL_ROWS);
                fail_count = fail_count + 1;
            end
        end

        if (fail_count == 0) $display("ALL_PASS  (%0d vectors)", NV);
        else $display("FAILURES=%0d", fail_count);
        $finish;
    end

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

endmodule
"""


# ── Vitis testbench ────────────────────────────────────────────────────────────


def _gen_vitis_tb(m, k, n, module_name, base_seed, all_a_stim, all_b_stim, all_bias, all_golden, timing=False):
    """Generate a multi-vector Vitis testbench (simple valid/ready/last protocol)."""
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one A row + one B col per cycle
    a_bytes = grid_rows * 8
    b_bytes = grid_cols * 8
    c_bytes = grid_cols * 8
    aw = a_bytes * 8
    bw = b_bytes * 8
    cw = c_bytes * 8

    num_vectors = len(all_a_stim)
    total_out_rows = grid_rows * 8

    a_init, b_init, bias_init, golden_init = [], [], [], []
    for v in range(num_vectors):
        for t in range(input_beats):
            a_init.append(f"        a_stim[{v}][{t}] = {hex_literal(all_a_stim[v][t], a_bytes)};")
            b_init.append(f"        b_stim[{v}][{t}] = {hex_literal(all_b_stim[v][t], b_bytes)};")
        bias_init.append(f"        bias_stim[{v}] = {hex_literal(all_bias[v], b_bytes)};")
        for row in range(total_out_rows):
            golden_init.append(f"        golden[{v}][{row}] = {hex_literal(all_golden[v][row], c_bytes)};")

    vt_first_out = '            if (out_row_idx == 0) $display("T:first_output=%0d", $realtime);' if timing else ""
    vt_last_out  = f'            if (out_row_idx == {total_out_rows} - 1) $display("T:last_output=%0d", $realtime);' if timing else ""

    return f"""\
`timescale 1ns/1ps
// Auto-generated Vitis-RTL multi-vector testbench  (row/col streaming, valid/ready/last)
// M={m}, K={k}, N={n}  |  base_seed={base_seed}  |  {num_vectors} vectors

module tb_vitis_{m}x{k}x{n};

    localparam NV = {num_vectors};
    localparam INPUT_BEATS = {input_beats};

    reg  ap_clk = 0;
    reg  ap_rst = 1;
    reg  ap_ce = 1;

    reg  [{aw - 1}:0] a_tdata = 0;
    reg        a_tvalid = 0;
    wire       a_tready;

    reg  [{bw - 1}:0] bias_tdata = 0;
    reg        bias_tvalid = 0;
    wire       bias_tready;

    reg  [{bw - 1}:0] b_tdata = 0;
    reg        b_tvalid = 0;
    wire       b_tready;

    wire [{cw - 1}:0] c_tdata;
    wire       c_tvalid;
    reg        c_tready = 1;

    {module_name} dut (
        .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_ce(ap_ce),
        .a_tdata(a_tdata), .a_tvalid(a_tvalid), .a_tready(a_tready),
        .bias_tdata(bias_tdata), .bias_tvalid(bias_tvalid), .bias_tready(bias_tready),
        .b_tdata(b_tdata), .b_tvalid(b_tvalid), .b_tready(b_tready),
        .c_tdata(c_tdata), .c_tvalid(c_tvalid), .c_tready(c_tready)
    );

    always #5 ap_clk = ~ap_clk;

    // Stimulus & golden memories
    reg [{aw - 1}:0] a_stim    [0:NV-1][0:INPUT_BEATS - 1];
    reg [{bw - 1}:0] b_stim    [0:NV-1][0:INPUT_BEATS - 1];
    reg [{bw - 1}:0] bias_stim [0:NV-1];
    reg [{cw - 1}:0] golden    [0:NV-1][0:{total_out_rows - 1}];

    initial begin
{chr(10).join(a_init)}
{chr(10).join(b_init)}
{chr(10).join(bias_init)}
{chr(10).join(golden_init)}
    end

    integer vec_idx, out_row_idx;
    integer t;
    integer pass_count, fail_count;

    initial begin
        $display("=== Vitis TB {m}x{k}x{n}  ({num_vectors} vectors) ===");
        pass_count = 0; fail_count = 0;

        for (vec_idx = 0; vec_idx < NV; vec_idx = vec_idx + 1) begin
            $display("--- Vector %0d (seed %0d) ---", vec_idx, {base_seed} + vec_idx);
            out_row_idx = 0;

            // Even vectors: full reset.  Odd vectors: back-to-back without reset.
            if (vec_idx == 0 || (vec_idx % 2) == 0) begin
                ap_rst <= 1;
                repeat (2) @(posedge ap_clk);
                ap_rst <= 0;
            end

            // Feed bias (1 beat)
            @(posedge ap_clk);
            bias_tdata  <= bias_stim[vec_idx];
            bias_tvalid <= 1;
            @(posedge ap_clk);
            bias_tvalid <= 0;

            // Feed A+B (INPUT_BEATS row/col pairs)
            a_tdata <= a_stim[vec_idx][0];
            b_tdata <= b_stim[vec_idx][0];
            a_tvalid <= 1;
            b_tvalid <= 1;
            @(posedge ap_clk);
            for (t = 1; t < INPUT_BEATS; t = t + 1) begin
                a_tdata <= a_stim[vec_idx][t];
                b_tdata <= b_stim[vec_idx][t];
                @(posedge ap_clk);
            end
            a_tvalid <= 0;
            b_tvalid <= 0;

            // Drain results (all grid_rows*8 rows, including zero-padded)
            for (int i = 0; i < {total_out_rows}; i++) begin
                while (!c_tvalid) @(posedge ap_clk);
{vt_first_out}
{vt_last_out}
                if (c_tdata !== golden[vec_idx][out_row_idx]) begin
                    $display("FAIL vec %0d row %0d: got %h expected %h", vec_idx, out_row_idx, c_tdata, golden[vec_idx][out_row_idx]);
                    fail_count = fail_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                end
                out_row_idx = out_row_idx + 1;
                @(posedge ap_clk);
            end
        end

        if (fail_count == 0) $display("ALL_PASS  (%0d vectors)", NV);
        else $display("FAILURES=%0d", fail_count);
        $finish;
    end

endmodule
"""


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_tb(m, k, n, module_name="gemm_grid_wrapper", seed=42, protocol="catapult",
                num_vectors=10, timing=False):
    """Generate a self-checking multi-vector Verilog testbench.

    Args:
        m, k, n: GEMM dimensions.
        module_name: Name of the DUT module to instantiate.
        seed: Base random seed (vectors use seed, seed+1, ..., seed+N-1).
        protocol: ``"catapult"`` or ``"vitis"``.
        num_vectors: Number of random test vectors (default 10).
        timing: If True, emit ``$display`` with ``$realtime`` at key events.

    Even vectors run with reset; odd vectors run back-to-back without reset.
    Returns:
        Verilog source as a string.
    """
    if protocol == "vitis":
        all_a, all_b, all_bias, all_golden, gr, gc = _gen_all_stimulus_vitis(
            m, k, n, num_vectors, seed)
        return _gen_vitis_tb(m, k, n, module_name, seed, all_a, all_b, all_bias, all_golden, timing=timing)
    all_a, all_b, all_bias, all_golden, gr, gc = _gen_all_stimulus(
        m, k, n, num_vectors, seed)
    return _gen_catapult_tb(m, k, n, module_name, seed, all_a, all_b, all_bias, all_golden, timing=timing)


def generate_tb_with_data(m, k, n, module_name, seed, protocol, A, B, biases, C_sat, timing=False):
    """Single-vector testbench (backward compat — used for manual timing runs)."""
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)  # row/col: one row + one col per cycle
    a_stim = [pack_a_row(A, t, grid_rows, m) for t in range(input_beats)]
    b_stim = [pack_b_col(B, t, grid_cols, n) for t in range(input_beats)]
    bias_packed = pack_bias(biases, grid_cols, n)
    golden = []
    for actual_row in range(m):
        rt = actual_row // 8
        rl = actual_row % 8
        golden.append(pack_c_row(C_sat, rt, rl, grid_cols, m, n, protocol))
    if protocol == "vitis":
        return _gen_vitis_tb(m, k, n, module_name, seed, [a_stim], [b_stim], [bias_packed], [golden], timing=timing)
    # Catapult: need (grid_rows*8) rows of golden
    total_out_rows = grid_rows * 8
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
    parser.add_argument("--protocol", choices=("catapult", "vitis"), default="catapult")
    parser.add_argument("--output", type=str, default="tb_gemm.v")
    args = parser.parse_args()

    content = generate_tb(args.m, args.k, args.n, args.name, args.seed, args.protocol)
    Path(args.output).write_text(content)
    print(f"Generated {args.output}  ({args.protocol}, M={args.m}, K={args.k}, N={args.n}, seed={args.seed})")
