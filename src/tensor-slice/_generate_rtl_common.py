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


def total_cycles(m, k, n, P=3):
    """Total cycle count for the grid counter.

    Uses the proven Catapult formula: done_cycle of the bottom-right tile
    = (gr-1 + gc-1)*8 + 7 + k + P + 9.
    Shared by both Catapult and Vitis RTL generators.
    """
    gr = grid_rows(m)
    gc = grid_cols(n)
    return (gr - 1 + gc - 1) * 8 + 7 + k + P + 9


def tile_comment(m, k, n, gr, gc, ks):
    return (
        f"// Dimensions: M={m}, K={k}, N={n}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}, K_STEPS={ks}\n"
        f"// A tail (rows after tile {gr - 1}): {vm(tail_mask_hex(m, gr - 1))}\n"
        f"// B tail (cols after tile {gc - 1}): {vm(tail_mask_hex(n, gc - 1))}\n"
    )
