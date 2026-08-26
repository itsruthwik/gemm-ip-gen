"""tensor_slice geometry: tile counts, stream widths, latency, and validity masks.

This is the canonical home for the 8-lane tiling geometry of the tensor_slice
hardblock. It is deliberately self-contained (no ``gemm_ip`` imports) so the RTL
generation path — ``rtl.py`` / ``golden.py`` and ``run_rtl_tests.py`` — can import
it with only the target directory on ``sys.path``.
"""
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

LANE_WIDTH = 8  # tensor_slice processes 8 int8 lanes per tile per cycle

# Standalone slice IP RTL for this target (external; not required for the
# behavioral RTL path, only for structural synthesis).
TENSOR_SLICE_SRC = Path(__file__).resolve().parent / "tensor_slice_int8.v"


# ── Helpers ────────────────────────────────────────────────────────────────────
# _ceil_div is kept local (rather than imported from gemm_ip.common) so this
# module stays import-free of the core — see the module docstring.


def _ceil_div(a, b):
    return (a + b - 1) // b


# ── Width and tile calculations ────────────────────────────────────────────────


def grid_rows(m):
    """Number of 8-lane row tiles needed to cover *m* rows."""
    return _ceil_div(m, LANE_WIDTH)


def grid_cols(n):
    """Number of 8-lane column tiles needed to cover *n* columns."""
    return _ceil_div(n, LANE_WIDTH)


def k_chunks(k):
    """Number of 8-lane K chunks needed to cover *k* reduction lanes."""
    return _ceil_div(k, LANE_WIDTH)


def k_chunk_size(k, chunk_index):
    """Valid K lanes in one 8-lane chunk."""
    remain = int(k) - int(chunk_index) * LANE_WIDTH
    if remain <= 0:
        return 0
    return min(LANE_WIDTH, remain)


def k_chunk_mask(k, chunk_index):
    """8-bit validity mask for one K chunk."""
    return tail_mask_hex(k, chunk_index)


def _validate_gemm_k_spatial(k, gemm_k_spatial):
    """Validate/normalise the spatial K-partition count against K_CHUNKS."""
    k_chunks = _ceil_div(k, LANE_WIDTH)
    if gemm_k_spatial is None:
        return k_chunks
    k_spatial = int(gemm_k_spatial)
    if k_spatial < 1:
        raise ValueError("gemm_k_spatial must be >= 1")
    if k_spatial > k_chunks:
        raise ValueError(
            f"gemm_k_spatial={k_spatial} exceeds K_CHUNKS={k_chunks}; "
            "v1 requires at most one spatial grid per K chunk"
        )
    return k_spatial


def a_stream_width(m):
    """Bit-width of the activation stream packet for *m* rows."""
    return grid_rows(m) * 64


def b_stream_width(n):
    """Bit-width of the weight (and bias) stream packet for *n* columns."""
    return grid_cols(n) * 64


def c_stream_width(n, out_bits=8):
    """Bit-width of the result stream packet for *n* columns.

    ``out_bits`` is the per-lane result width from ``output_precision`` (default
    8 = legacy int8). 8 lanes per column tile, so width = grid_cols * 8 * out_bits.
    """
    return grid_cols(n) * 8 * out_bits


def bias_stream_width(n):
    """Bit-width of the bias stream packet for *n* columns."""
    return b_stream_width(n)


# ── Validity masks (for tail dimensions) ───────────────────────────────────────


def tail_mask_hex(total, chunk_index):
    """Return an 8-bit validity mask for the *chunk_index*-th 8-lane tile."""
    remain = total - chunk_index * LANE_WIDTH
    if remain <= 0:
        return 0x00
    if remain >= LANE_WIDTH:
        return 0xFF
    return (0xFF >> (LANE_WIDTH - remain)) & 0xFF


def vm(val):
    """Format a mask byte as a Verilog literal."""
    return f"8'h{val:02X}"


# ── Latency / cycle counts ─────────────────────────────────────────────────────


def total_cycles(m, k, n, P=3, feed_mode="chained"):
    """Conservative total cycle count for the grid counter.

    Row/col: preload(1) + collect(max(M,N)) + transition(2) +
    feed(max_loc_delay + max(M,N)) + compute + output_align + margin.
    """
    gr = (m + 7) // 8
    gc = (n + 7) // 8
    last_loc = (gr - 1 + gc - 1) * 8
    beats = max(m, n)
    local_compute = 7 + k + P
    output_align = (gc - 1) * 8 + (gr - 1) * gc * 8
    margin = 60
    return 1 + beats + 2 + (last_loc + beats) + last_loc + local_compute + output_align + 8 + margin


def latency_cycles(k_val, grid_rows_val, grid_cols_val, m=None, n=None,
                   k=None, feed_mode="direct"):
    """First output available at this cycle (0-based).  Conservative estimate;
    exact transaction control uses total_cycles().
    """
    if m is not None and n is not None and k is not None:
        # Conservative: total_cycles minus output FIFO drain margin (~20 cycles)
        return max(0, total_cycles(m, k, n, feed_mode=feed_mode) - 20 - (grid_rows_val * 8))
    # Fallback
    return (grid_cols_val - 1) * 8 + k_val + 10


def dead_cycles_raw(grid_cols_val):
    """Cycles from first output to drain completion (pre-trim)."""
    return (grid_cols_val - 1) * 8 + 10


def dead_cycles(grid_cols_val):
    """Same as dead_cycles_raw plus 1 for pipeline register."""
    return dead_cycles_raw(grid_cols_val) + 1


# ── Verilog comment helpers ─────────────────────────────────────────────────────


def tile_comment(m, k, n, gr, gc):
    return (
        f"// Dimensions: M={m}, K={k}, N={n}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}\n"
        f"// A tail (rows after tile {gr - 1}): {vm(tail_mask_hex(m, gr - 1))}\n"
        f"// B tail (cols after tile {gc - 1}): {vm(tail_mask_hex(n, gc - 1))}\n"
    )


def verilog_tile_comment(item):
    """Return a Verilog comment summarising grid dimensions and tail masks."""
    gr = item["grid_rows"]
    gc = item["grid_cols"]
    return (
        f"// Dimensions: n_in={item['k']}, n_out={item['n']}, m={item['m']}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}\n"
        f"// A tail mask (rows after tile {gr - 1}): {vm(tail_mask_hex(item['m'], gr - 1))}\n"
        f"// B tail mask (cols after tile {gc - 1}): {vm(tail_mask_hex(item['n'], gc - 1))}\n"
    )
