"""Shared helpers for generate_catapult_rtl.py and generate_vitis_rtl.py."""


def grid_rows(m):
    return (m + 7) // 8


def grid_cols(n):
    return (n + 7) // 8


def k_steps_ceil(k):
    return (k + 7) // 8


def a_width_bits(m):
    return grid_rows(m) * 64


def b_width_bits(n):
    return grid_cols(n) * 64


def c_width_bits(n):
    return grid_cols(n) * 64


def tail_mask_hex(total, chunk_index):
    remain = total - chunk_index * 8
    if remain <= 0:
        return 0x00
    if remain >= 8:
        return 0xFF
    return (0xFF >> (8 - remain)) & 0xFF


def vm(val):
    return f"8'h{val:02X}"


def total_cycles(m, k, n, P=3, feed_mode="direct"):
    """Total cycle count for the grid counter.

    Direct row/col: preload(1) + collect(max(M,N)) + transition(1) +
    feed(8 beats simultaneous) + grid propagation + stagger + MAC + margin.

    Chained row/col: preload(1) + collect(max(M,N)) + transition(1) +
    feed(max(M,N) beats) + last-tile loc_delay + local compute +
    output alignment + FIFO drain margin.
    """
    gr = grid_rows(m)
    gc = grid_cols(n)
    input_beats = max(m, n)

    if feed_mode == "chained":
        # Conservative: collect + feed(8) + last tile location + local compute
        # + output alignment (per-column + per-tile-row) + readout + margin
        last_loc = (gr - 1 + gc - 1) * 8
        local_compute = 7 + k + P
        output_align = (gc - 1) * 8 + (gr - 1) * gc * 8
        margin = 16
        return 1 + input_beats + 1 + 8 + last_loc + local_compute + output_align + 8 + margin

    # Direct mode
    compute_drain = (gr - 1 + gc - 1) * 8 + 7 + P + 9
    return 1 + input_beats + 1 + 8 + compute_drain


def tile_comment(m, k, n, gr, gc, ks):
    return (
        f"// Dimensions: M={m}, K={k}, N={n}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}, K_STEPS={ks}\n"
        f"// A tail (rows after tile {gr - 1}): {vm(tail_mask_hex(m, gr - 1))}\n"
        f"// B tail (cols after tile {gc - 1}): {vm(tail_mask_hex(n, gc - 1))}\n"
    )
