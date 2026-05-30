#!/usr/bin/env python3
"""
generate_verilog_tb.py
Generates a self-checking Verilog testbench for gemm_grid_wrapper.
Two random INT8 matrices are generated in Python; their product is
computed as the golden reference and embedded as literals in the TB.
"""
import argparse
import numpy as np
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def pack_a_cycle(A, k_step, grid_rows, m):
    """
    Pack one K-cycle slice of A into a single integer.
    Layout: bits [(r*8+row)*8 +: 8] = A[r*8+row][k_step]
    """
    val = 0
    for r in range(grid_rows):
        for row in range(8):
            actual_row = r * 8 + row
            if actual_row < m:
                byte = int(A[actual_row, k_step]) & 0xFF
            else:
                byte = 0
            val |= byte << ((r * 8 + row) * 8)
    return val


def pack_b_cycle(B, k_step, grid_cols, n):
    """
    Pack one K-cycle slice of B into a single integer.
    Layout: bits [(c*8+col)*8 +: 8] = B[k_step][c*8+col]
    """
    val = 0
    for c in range(grid_cols):
        for col in range(8):
            actual_col = c * 8 + col
            if actual_col < n:
                byte = int(B[k_step, actual_col]) & 0xFF
            else:
                byte = 0
            val |= byte << ((c * 8 + col) * 8)
    return val


def pack_c_row(C_sat, r_tile, row_in_tile, grid_cols, m, n):
    """
    Pack one output row of C into a single integer.
    Layout: bits [c*128 + col*8 +: 8] = C_sat[r_tile*8+row_in_tile][c*8+col]
    (Each column tile has 128 bits, bottom 64 bits contain the 8 INT8 elements, top 64 bits are zero).
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
            tile_val |= byte << (col * 8)
        val |= tile_val << (c * 128)
    return val


def hex_literal(val, width_bytes):
    """Return a Verilog hex literal of the given byte width."""
    bits = width_bytes * 8
    hex_str = f"{val:0{width_bytes*2}x}"
    return f"{bits}'h{hex_str}"


# ── main generator ────────────────────────────────────────────────────────────

def generate_tb(m, k, n, module_name="gemm_grid_wrapper", seed=42):
    rng = np.random.default_rng(seed)
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    a_bytes   = grid_rows * 8
    b_bytes   = grid_cols * 8
    c_bytes   = grid_cols * 16

    # ── Random INT8 matrices ──────────────────────────────────────────────────
    # Keep values small enough that K accumulations won't saturate:
    #   |K * max_a * max_b| < 128  → max_val ≈ sqrt(127/K)
    max_val = max(1, int((127 / max(k, 1)) ** 0.5))
    A = rng.integers(-max_val, max_val + 1, size=(m, k), dtype=np.int8)
    B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)

    # ── Golden reference ──────────────────────────────────────────────────────
    C_ref = A.astype(np.int32) @ B.astype(np.int32)
    C_sat = np.clip(C_ref, -128, 127).astype(np.int8)

    # ── Pack stimulus per K cycle ─────────────────────────────────────────────
    a_stim = [pack_a_cycle(A, kk, grid_rows, m) for kk in range(k)]
    b_stim = [pack_b_cycle(B, kk, grid_cols, n) for kk in range(k)]

    # ── Pack expected output rows ─────────────────────────────────────────────
    # The wrapper emits grid_rows*8 rows in order:
    #   tile-row 0: rows 0..7, tile-row 1: rows 8..15, ...
    expected = []
    for rt in range(grid_rows):
        for row in range(8):
            expected.append(pack_c_row(C_sat, rt, row, grid_cols, m, n))

    total_out_rows = grid_rows * 8
    # The generated wrapper launches the slice array one cycle after `rst`
    # goes high and the checker counts from the first active wrapper cycle,
    # which places the first observed out_valid two cycles after the raw
    # slice readout-start formula.
    first_valid_cycle = (grid_cols - 1) * 8 + 7 + k + 3 + 2
    last_valid_cycle = first_valid_cycle + total_out_rows - 1

    # ── Build Verilog stimulus block ──────────────────────────────────────────
    stim_lines = []
    for kk in range(k):
        stim_lines.append(
            f"        a_rows = {hex_literal(a_stim[kk], a_bytes)};\n"
            f"        b_cols = {hex_literal(b_stim[kk], b_bytes)};\n"
            f"        @(posedge clk);"
        )
    stim_block = "\n".join(stim_lines)

    # ── Build checker block ───────────────────────────────────────────────────
    check_lines = []
    check_lines.append(f"    integer out_row_idx;")
    check_lines.append(f"    integer pass_count, fail_count;")
    check_lines.append(f"    integer cycle_ctr;")
    check_lines.append(f"    integer out_last_count;")
    check_lines.append(f"    integer first_valid_seen;")
    check_lines.append(f"    reg [{c_bytes*8-1}:0] golden [{total_out_rows-1}:0];")
    check_lines.append(f"    initial begin")
    for idx, val in enumerate(expected):
        check_lines.append(
            f"        golden[{idx}] = {hex_literal(val, c_bytes)};"
        )
    check_lines.append(f"        out_row_idx = 0;")
    check_lines.append(f"        pass_count  = 0;")
    check_lines.append(f"        fail_count  = 0;")
    check_lines.append(f"        cycle_ctr = 0;")
    check_lines.append(f"        out_last_count = 0;")
    check_lines.append(f"        first_valid_seen = 0;")
    check_lines.append(f"    end")

    check_block = "\n".join(check_lines)

    # ── Matrix summary comment ────────────────────────────────────────────────
    a_preview = "\n".join(
        f"//   {list(A[r])}" for r in range(min(m, 4))
    ) + ("\n//   ..." if m > 4 else "")
    b_preview = "\n".join(
        f"//   {list(B[r])}" for r in range(min(k, 4))
    ) + ("\n//   ..." if k > 4 else "")
    c_preview = "\n".join(
        f"//   {list(C_sat[r])}" for r in range(min(m, 4))
    ) + ("\n//   ..." if m > 4 else "")

    tb = f"""\
`timescale 1ns/1ps
// Auto-generated self-checking testbench
// M={m}, K={k}, N={n}  |  seed={seed}  |  max_element=±{max_val}
//
// A ({m}×{k}):
{a_preview}
// B ({k}×{n}):
{b_preview}
// C = A@B saturated to INT8 ({m}×{n}):
{c_preview}

module tb_{m}x{k}x{n};
    // ---- DUT ports ----
    reg  clk = 0;
    reg  rst = 1;
    reg  en  = 1;
    reg  in_valid = 0;
    reg  [{a_bytes*8-1}:0] a_rows = 0;
    reg  [{b_bytes*8-1}:0] b_cols = 0;
    wire [{c_bytes*8-1}:0] c_row;
    wire out_valid;
    wire out_last;

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en),
        .a_rows(a_rows), .b_cols(b_cols), .in_valid(in_valid),
        .c_row(c_row), .out_valid(out_valid), .out_last(out_last)
    );

    always #5 clk = ~clk;

    // ---- Checker state ----
{check_block}

    // ---- Stimulus ----
    initial begin
        $display("=== TB {m}x{k}x{n} start ===");
        #20;
        @(posedge clk);
        rst      = 0;
        in_valid = 1;
{stim_block}
        in_valid = 0;
        a_rows = 0;
        b_cols = 0;

        // Wait for completion or timeout
        fork
            begin : wait_done
                wait (out_last);
                @(posedge clk);
                disable timeout;
            end
            begin : timeout
                repeat ({last_valid_cycle + 200}) @(posedge clk);
                $display("TIMEOUT waiting for out_last");
                fail_count = fail_count + 1;
                disable wait_done;
            end
        join
        #100;
        if (out_row_idx != {total_out_rows}) begin
            $display("FAIL row count: got %0d expected %0d", out_row_idx, {total_out_rows});
            fail_count = fail_count + 1;
        end
        if (out_last_count != 1) begin
            $display("FAIL out_last pulse count: got %0d expected 1", out_last_count);
            fail_count = fail_count + 1;
        end
        $display("=== RESULTS: %0d PASS / %0d FAIL ===", pass_count, fail_count);
        if (fail_count == 0)
            $display("SIMULATION PASSED");
        else
            $display("SIMULATION FAILED");
        $finish;
    end

    // ---- Sample and check output ----
    always @(posedge clk) begin
        if (rst) begin
            cycle_ctr <= 0;
        end else begin
            cycle_ctr <= cycle_ctr + 1;
        end

        if (out_last) begin
            out_last_count = out_last_count + 1;
            if (!out_valid) begin
                $display("FAIL out_last asserted without out_valid at cycle %0d", cycle_ctr);
                fail_count = fail_count + 1;
            end
            if (out_row_idx != {total_out_rows - 1}) begin
                $display("FAIL out_last asserted at row %0d expected %0d",
                         out_row_idx, {total_out_rows - 1});
                fail_count = fail_count + 1;
            end
            if (cycle_ctr !== {last_valid_cycle}) begin
                $display("FAIL out_last cycle: got %0d expected %0d",
                         cycle_ctr, {last_valid_cycle});
                fail_count = fail_count + 1;
            end
        end

        if (out_valid && (out_row_idx >= {total_out_rows})) begin
            $display("FAIL extra out_valid beyond expected rows at cycle %0d", cycle_ctr);
            fail_count = fail_count + 1;
        end else if (out_valid && out_row_idx < {total_out_rows}) begin
            if (!first_valid_seen) begin
                first_valid_seen <= 1;
                if (cycle_ctr !== {first_valid_cycle}) begin
                    $display("FAIL first out_valid cycle: got %0d expected %0d",
                             cycle_ctr, {first_valid_cycle});
                    fail_count = fail_count + 1;
                end
            end
            if (cycle_ctr !== ({first_valid_cycle} + out_row_idx)) begin
                $display("FAIL out_valid timing row %0d: got cycle %0d expected %0d",
                         out_row_idx, cycle_ctr, ({first_valid_cycle} + out_row_idx));
                fail_count = fail_count + 1;
            end
            if (c_row !== golden[out_row_idx]) begin
                $display("FAIL row %0d: got %h, expected %h",
                         out_row_idx, c_row, golden[out_row_idx]);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
            out_row_idx = out_row_idx + 1;
        end
    end

endmodule
"""
    return tb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a self-checking Verilog testbench for gemm_grid_wrapper"
    )
    parser.add_argument("--m",      type=int, required=True)
    parser.add_argument("--k",      type=int, required=True)
    parser.add_argument("--n",      type=int, required=True)
    parser.add_argument("--name",   type=str, default="gemm_grid_wrapper")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default="tb_grid.v")
    args = parser.parse_args()

    content = generate_tb(args.m, args.k, args.n, args.name, args.seed)
    Path(args.output).write_text(content)
    print(f"Generated {args.output}  (M={args.m}, K={args.k}, N={args.n}, seed={args.seed})")
